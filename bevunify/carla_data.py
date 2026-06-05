"""Self-contained CARLA data infra — no carla code lives in third_party anymore.

Bundles, in one file:
  - ``Sample``                 — JSON sample dataclass
  - ``LoadDataTransform``      — runtime transform with VR (per-cam image+extrinsic
                                  swap) and CTS (cross-platform sedan-source) hooks
  - ``CarlaGeneratedDataset``  — per-scene torch Dataset
  - ``encode`` / ``decode``    — 12-class BEV PNG bit-pack helpers
  - ``get_carla_split``        — reads ``bevunify/splits/carla/{split}.txt``

The toggle-aware subclass (``CarlaToggleLoadDataTransform`` in transforms_toggle.py)
inherits from ``LoadDataTransform`` here and remaps ``bev*`` keys to ``vehicle*``.
"""
import json
import logging
import math
import pathlib

import numpy as np
import torch
import torchvision
from PIL import Image

log = logging.getLogger(__name__)


# ── splits ────────────────────────────────────────────────────────────────────

_SPLITS_DIR = pathlib.Path(__file__).resolve().parent / "splits" / "carla"


def get_carla_split(split):
    """Read ``splits/carla/{split}.txt`` → list of scene names."""
    return (_SPLITS_DIR / f"{split}.txt").read_text().strip().split("\n")


# ── BEV bit-pack helpers (12-class PNG ↔ binary stack) ────────────────────────

def encode(x):
    """(h, w, c) np.uint8 {0, 255} → (h, w) int32 bit-packed."""
    n = x.shape[2]
    assert x.ndim == 3 and x.dtype == np.uint8
    shift = np.arange(n, dtype=np.int32)[None, None]
    binary = (x > 0)
    return ((binary << shift).sum(-1)).astype(np.int32)


def decode(img, n):
    """returns (h, w, n) np.int32 {0, 1}."""
    shift = np.arange(n, dtype=np.int32)[None, None]
    x = np.array(img)[..., None]
    return (x >> shift) & 1


# ── Sample dataclass ──────────────────────────────────────────────────────────

class Sample(dict):
    def __init__(self, token, scene, intrinsics, extrinsics, images, view, bev, **kwargs):
        super().__init__(**kwargs)
        self.token = token
        self.scene = scene
        self.view = view
        self.bev = bev
        self.images = images
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    def __getattr__(self, key):
        return super().__getitem__(key)

    def __setattr__(self, key, val):
        self[key] = val
        return super().__setattr__(key, val)


# ── Runtime transform — VR/CTS hooks built in ─────────────────────────────────

class LoadDataTransform(torchvision.transforms.ToTensor):
    """Load CARLA images + pre-rendered BEV labels.

    Per-cam viewpoint perturbation (VR):
      Normal: image_swap=False, extrinsic_swap=False
      ER    : extrinsic_swap=True  (extrinsic only)
      VR    : image_swap=True      (image only — PRIMARY)
      CR    : both swap

    Cross-platform transfer (CTS):
      cts_img_to_sedan=True  → load sedan RGB instead of target's
      cts_ext_override={cam: 4x4} → replace target extrinsic with sedan's
    """

    def __init__(self, dataset_dir, labels_dir, image_config, num_classes,
                 augment="none", split_intrin_extrin=False, val_perturb=None,
                 extrinsic_noise_deg=0.0, label_indices=None,
                 eval_viewpoint_variant=None, viewpoint_metadata_path=None,
                 eval_image_swap=True, eval_extrinsic_swap=True, eval_target_cameras=None):
        super().__init__()
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.labels_dir = pathlib.Path(labels_dir)
        self.image_config = image_config
        self.num_classes = num_classes
        self.label_indices = label_indices
        self.split_intrin_extrin = split_intrin_extrin
        self.val_perturb = val_perturb
        self.extrinsic_noise_deg = float(extrinsic_noise_deg)

        # GeoBEV viewpoint eval setup
        self.eval_viewpoint_variant = eval_viewpoint_variant
        self.eval_image_swap = bool(eval_image_swap)
        self.eval_extrinsic_swap = bool(eval_extrinsic_swap)
        if eval_target_cameras in (None, "ALL", "all"):
            self.eval_target_cameras = None
        else:
            self.eval_target_cameras = (set(eval_target_cameras)
                                        if not isinstance(eval_target_cameras, str)
                                        else {eval_target_cameras})
        # Store the metadata path so lazy-loaders (eval.py:_mutate_vr, tests/probe_*)
        # can pick it up later via getattr(transform, "viewpoint_metadata_path"). Without
        # this, the transform discards the path after __init__ and any code mutating
        # eval_viewpoint_variant post-construction cannot load the metadata → VR/ER swap
        # silently no-ops in get_cameras (do_swap_img/extrinsic_swap conditions require
        # self.viewpoint_metadata != None).
        self.viewpoint_metadata_path = viewpoint_metadata_path
        self.viewpoint_metadata = None
        self.vr_root = None

        # GT cache: token -> dict from get_bev(). Populated once before the VR/CTS
        # config loop (see eval._warm_gt_cache). With DataLoader fork on Linux this
        # dict is COW-shared across workers. Skip 3 disk reads per sample per config
        # (bev.png + visibility.png + aux.npz). LOSSLESS — same tensors as on-disk.
        self.gt_cache = None

        # Image cache: str(image_path) -> (img_tensor (3,h,w) float32, image_w, image_h).
        # Stores ORIGINAL-image processed tensors only (skipping VR-swapped paths since
        # they explode by 90×). Hit rate per VR config: Normal/ER = 6/6, VR/CR per_cam
        # = 5/6, all_cam = 0/6. The cached tensor is the exact output of img_transform
        # after resize + crop (uses image's original width/height for intrinsics rescale).
        # LOSSLESS — bit-identical to the uncached pipeline.
        self.image_cache = None
        if eval_viewpoint_variant is not None and eval_viewpoint_variant != "yaw0pitch0roll0":
            assert viewpoint_metadata_path is not None, \
                "eval_viewpoint_variant set but viewpoint_metadata_path is None"
            self.viewpoint_metadata = json.loads(pathlib.Path(viewpoint_metadata_path).read_text())
            self.vr_root = pathlib.Path(self.viewpoint_metadata["vr_root"])

        # CTS cross-platform eval hooks (default off; set by eval.py protocol=cts)
        self.cts_platform = None        # "suv" | "bus" (target DB platform tag)
        self.cts_img_to_sedan = False
        self.cts_ext_override = None    # {cam_channel: 4x4 sedan extrinsic}

        xform = {"none": []}[augment] + [torchvision.transforms.ToTensor()]
        self.img_transform = torchvision.transforms.Compose(xform)
        self.to_tensor = super().__call__

    # ── VR helpers ────────────────────────────────────────────────────────────

    def _swap_image_path(self, image_path, cam_channel):
        """sedan baseline (10 Hz frame N) -> VR variant under vr_root (20 Hz frame 2 N).
        Filename rebuilt from scratch: drops platform tag, doubles frame index."""
        import re
        v = self.eval_viewpoint_variant
        fn = pathlib.PurePosixPath(image_path).name
        m = re.search(r"scene-(\d+)-frame-(\d+)", fn)
        scene_idx, norm_frame = int(m.group(1)), int(m.group(2))
        vr_frame = norm_frame * 2
        new_fn = f"SimBEV-scene-{scene_idx:04d}-frame-{vr_frame:04d}-RGB-{cam_channel}_{v}.jpg"
        return self.vr_root / "sweeps" / f"RGB-{cam_channel}_{v}" / new_fn

    def _extrinsic_correction(self, scene_name, cam_channel):
        """E_variant = correction @ E_baseline."""
        from pyquaternion import Quaternion
        v = self.eval_viewpoint_variant
        baseline = self.viewpoint_metadata["scenes"][scene_name][cam_channel]["yaw0pitch0roll0"]
        variant  = self.viewpoint_metadata["scenes"][scene_name][cam_channel][v]

        def cam_from_egocam(s2e):
            R = Quaternion(s2e["sensor2ego_rotation"]).rotation_matrix
            t = np.asarray(s2e["sensor2ego_translation"])
            T = np.eye(4); T[:3, :3] = R.T; T[:3, 3] = -R.T @ t
            return T

        def egocam_from_cam(s2e):
            R = Quaternion(s2e["sensor2ego_rotation"]).rotation_matrix
            t = np.asarray(s2e["sensor2ego_translation"])
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
            return T

        return cam_from_egocam(variant) @ egocam_from_cam(baseline)

    # ── CTS helper ────────────────────────────────────────────────────────────

    def _cts_path_to_sedan(self, image_path, cam_channel):
        """Target (suv/bus) RGB rel-path -> sedan equivalent at same scene/frame."""
        plat = self.cts_platform
        return (image_path
                .replace(f"RGB-{plat}-{cam_channel}/", f"RGB-{cam_channel}/")
                .replace(f"RGB-{plat}__{cam_channel}",  f"RGB-subcompact__{cam_channel}"))

    def _remap_perturbed_path(self, image_path):
        """val_perturb hook: redirect to a single-axis perturbation sweep."""
        vp = self.val_perturb
        yaw = pitch = roll = 0
        axis, angle = vp["axis"], int(vp["angle"])
        if axis == "yaw":   yaw = angle
        elif axis == "pitch": pitch = angle
        elif axis == "roll":  roll = angle
        tag = f"yaw{yaw}pitch{pitch}roll{roll}"
        p = pathlib.Path(image_path)
        return pathlib.Path(vp["sweeps_root"]) / f"{p.parent.name}_{tag}" / f"{p.stem}_{tag}{p.suffix}"

    # ── main load steps ──────────────────────────────────────────────────────

    def get_cameras(self, sample, h, w, top_crop):
        images, intrinsics = [], []
        cam_channels = (sample.get("cam_channels", None) if isinstance(sample, dict)
                        else getattr(sample, "cam_channels", None))

        for i, (image_path, I_original) in enumerate(zip(sample.images, sample.intrinsics)):
            h_resize = h + top_crop
            w_resize = w

            cam_channel = cam_channels[i] if cam_channels else None
            is_target = (self.eval_target_cameras is None) or (cam_channel in self.eval_target_cameras)
            do_swap_img = bool(self.eval_viewpoint_variant and self.viewpoint_metadata
                               and self.eval_image_swap and is_target)

            if do_swap_img:
                resolved_path = self._swap_image_path(image_path, cam_channel)
            elif self.cts_img_to_sedan:
                resolved_path = self.dataset_dir / self._cts_path_to_sedan(image_path, cam_channel)
            elif self.val_perturb is not None:
                resolved_path = self._remap_perturbed_path(image_path)
            else:
                resolved_path = self.dataset_dir / image_path

            # Image cache hit → skip Image.open + resize + crop + ToTensor (~14ms/cam).
            cache_key = str(resolved_path)
            hit = (self.image_cache is not None and cache_key in self.image_cache)
            if hit:
                img_t, image_w, image_h = self.image_cache[cache_key]
            else:
                image = Image.open(resolved_path)
                image_new = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
                image_new = image_new.crop((0, top_crop, image_new.width, image_new.height))
                img_t = self.img_transform(image_new)
                image_w, image_h = image.width, image.height
                # Cache ORIGINAL paths only. VR-swap / CTS-sedan-img / val_perturb hit
                # unique-per-(variant,cam,sample) paths and would explode the cache by
                # ~1.3 MB × (per_cam 3792 + all_cam 22752) × 30 configs ≈ 500-600 GB,
                # blowing past the container's 400 GiB cgroup memory.max → silent OOM.
                # Matches the class docstring and _warm_image_cache contract.
                is_original = not (do_swap_img or self.cts_img_to_sedan
                                   or self.val_perturb is not None)
                if self.image_cache is not None and is_original:
                    self.image_cache[cache_key] = (img_t, image_w, image_h)

            I = np.float32(I_original)
            I[0, 0] *= w_resize / image_w
            I[0, 2] *= w_resize / image_w
            I[1, 1] *= h_resize / image_h
            I[1, 2] *= h_resize / image_h
            I[1, 2] -= top_crop

            images.append(img_t)
            intrinsics.append(torch.tensor(I))

        extrinsics = torch.tensor(np.float32(sample.extrinsics))

        # ER/CR — per-cam extrinsic correction
        if (self.eval_viewpoint_variant and self.viewpoint_metadata
                and self.eval_extrinsic_swap and cam_channels):
            for k, cam_channel in enumerate(cam_channels):
                is_target = (self.eval_target_cameras is None) or (cam_channel in self.eval_target_cameras)
                if is_target:
                    correction = self._extrinsic_correction(sample.scene, cam_channel)
                    extrinsics[k] = torch.tensor(correction.astype(np.float32)) @ extrinsics[k]

        # CTS — replace target extrinsic with sedan's (constant per cam)
        if self.cts_ext_override is not None and cam_channels:
            for k, cam_channel in enumerate(cam_channels):
                if cam_channel in self.cts_ext_override:
                    extrinsics[k] = torch.tensor(np.float32(self.cts_ext_override[cam_channel]))

        # Train-only extrinsic noise
        if self.extrinsic_noise_deg > 0.0 and self.val_perturb is None:
            std_rad = self.extrinsic_noise_deg * math.pi / 180.0
            for k in range(extrinsics.shape[0]):
                ax = torch.randn(3) * std_rad
                cz, sz = math.cos(ax[0].item()), math.sin(ax[0].item())
                cx, sx = math.cos(ax[1].item()), math.sin(ax[1].item())
                cy, sy = math.cos(ax[2].item()), math.sin(ax[2].item())
                Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float32)
                Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float32)
                Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
                R4 = torch.eye(4, dtype=torch.float32); R4[:3, :3] = Ry @ Rx @ Rz
                extrinsics[k] = R4 @ extrinsics[k]

        # lidar2img = K_4x4 @ E
        lidar2img_list = []
        for I, E in zip(intrinsics, extrinsics):
            K_4x4 = torch.eye(4, dtype=torch.float32); K_4x4[:3, :3] = I
            lidar2img_list.append(K_4x4 @ E)

        result = {"image": torch.stack(images, 0),
                  "lidar2img": torch.stack(lidar2img_list, 0)}
        if self.split_intrin_extrin:
            result["intrinsics"] = torch.stack(intrinsics, 0)
            result["extrinsics"] = extrinsics
        return result

    def get_bev(self, sample):
        # GT cache hit: skip 3 disk reads (bev.png / visibility.png / aux.npz).
        # GT is invariant under VR/CTS perturbations, so a single warm pass is enough.
        if self.gt_cache is not None and sample.token in self.gt_cache:
            return self.gt_cache[sample.token]

        scene_dir = self.labels_dir / sample.scene
        bev = None
        if sample.bev is not None:
            bev = Image.open(scene_dir / sample.bev)
            bev = decode(bev, self.num_classes)                   # (h, w, num_classes) {0,1}
            bev = (255 * bev).astype(np.uint8)
            bev = self.to_tensor(bev)                              # (num_classes, h, w) [0,1]

            # CVT-style 12-class → vehicle subset merge.
            if self.label_indices is not None and len(self.label_indices) > 0:
                merged = []
                for idx_group in self.label_indices:
                    merged.append(bev[list(idx_group)].amax(dim=0))
                bev = torch.stack(merged, dim=0)

        result = {"bev": bev, "view": torch.tensor(sample.view)}

        if "visibility" in sample:
            visibility = Image.open(scene_dir / sample.visibility)
            result["bev_visibility"] = np.array(visibility, dtype=np.uint8)

        if "aux" in sample:
            aux = np.load(scene_dir / sample.aux)["aux"]
            result["bev_center"] = self.to_tensor(aux[..., 1])
            if aux.shape[-1] >= 4:
                result["bev_offset"] = torch.tensor(aux[..., 2:4]).permute(2, 0, 1).float()

        if "pose" in sample:
            result["pose"] = np.float32(sample["pose"])

        if self.gt_cache is not None:
            self.gt_cache[sample.token] = result
        return result

    def __call__(self, batch):
        if not isinstance(batch, Sample):
            batch = Sample(**batch)
        result = dict()
        result.update(self.get_cameras(batch, **self.image_config))
        result.update(self.get_bev(batch))
        result["token"] = batch.token
        result["scene"] = batch.scene
        return result


# ── Runtime Dataset ───────────────────────────────────────────────────────────

class CarlaGeneratedDataset(torch.utils.data.Dataset):
    def __init__(self, scene_name, labels_dir, transform=None):
        self.samples = json.loads((pathlib.Path(labels_dir) / f"{scene_name}.json").read_text())
        self.transform = transform
        self.scene_name = scene_name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Linear-probe replacement on OSError: walk forward to skip corrupted samples.
        n = len(self)
        for offset in range(n):
            j = (idx + offset) % n
            try:
                data = Sample(**self.samples[j])
                if self.transform is not None:
                    data = self.transform(data)
                return data
            except OSError:
                if offset == 0:
                    log.warning(f"Skipping corrupted sample {idx}, scanning forward...")
                continue
        raise RuntimeError(f"No loadable samples in dataset (size={n})")
