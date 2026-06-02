"""Empirical probe: does VR per-cam image swap actually change CAM_FRONT?

Builds a CarlaToggleLoadDataTransform directly on sedan_eval/scene_0220 frame 0
(no hydra/no model — just exercise the data layer), then for each viewpoint
variant flips the swap hooks the same way bevunify.eval._mutate_vr does and
records the CAM_FRONT image tensor.

Pass criteria
  - MSE(Normal vs Normal) == 0.0 (trivial sanity)
  - MSE(Normal vs each VR variant CAM_FRONT) > 1e-3 (swap took effect)
  - MSE(Normal vs each VR variant CAM_BACK) == 0.0 (per-cam targeting works
    — other cams stay baseline)

If any "Normal vs VR" MSE is near zero, that is a BLOCKER: the eval grid
would silently use baseline images while reporting VR numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bevunify.carla_data import CarlaGeneratedDataset
from bevunify.transforms_toggle import CarlaToggleLoadDataTransform


SCENE = "scene_0220"
FRAME = 0
VARIANTS = [
    ("Normal",        None,                  False),
    ("VR_yaw20",      "yaw20pitch0roll0",    True),
    ("VR_yawneg20",   "yaw-20pitch0roll0",   True),
    ("VR_pitchneg20", "yaw0pitch-20roll0",   True),
    ("VR_roll12",     "yaw0pitch0roll12",    True),
]
PNG_OUT = Path("/tmp/probe_vr_image_swap.png")


def load_carla_cfg():
    with open(REPO_ROOT / "config" / "data" / "carla.yaml") as f:
        cfg = yaml.safe_load(f)
    # Resolve image config (handles the hydra defaults entry).
    img_yaml = REPO_ROOT / "config" / "data" / "img_params" / "scale_0_3.yaml"
    with open(img_yaml) as f:
        img_cfg = yaml.safe_load(f)
    # The hydra-merged carla.yaml has `image:` at top level; prefer that.
    image = cfg.get("image") or img_cfg
    return cfg, image


def build_transform(cfg, image):
    """Mirror what get_data_carla does for split=val."""
    dataset_dir = Path(cfg["val_dataset_dir"])
    labels_dir = Path(cfg["val_labels_dir"])
    return CarlaToggleLoadDataTransform(
        dataset_dir,
        labels_dir,
        image,
        num_classes=cfg["num_classes"],
        augment="none",
        split_intrin_extrin=cfg["split_intrin_extrin"],
        val_perturb=None,
        extrinsic_noise_deg=0.0,
        label_indices=cfg["label_indices"],
        eval_viewpoint_variant=None,          # set per-trial below
        viewpoint_metadata_path=cfg["viewpoint_metadata_path"],
        eval_image_swap=False,                # set per-trial below
        eval_extrinsic_swap=False,
        eval_target_cameras=None,
        training=False,
        # gt_* default True
    ), labels_dir


def mutate(transform, variant, image_swap, target_cams):
    """Same surface as bevunify.eval._mutate_vr (carla_data.py:153-156)."""
    transform.eval_viewpoint_variant = variant
    transform.eval_image_swap = bool(image_swap)
    transform.eval_extrinsic_swap = False    # we're probing image swap only
    transform.eval_target_cameras = (set(target_cams) if target_cams else None)
    if variant is not None and getattr(transform, "viewpoint_metadata", None) is None:
        import json
        meta = json.loads(Path(transform_metadata_path(transform)).read_text())
        transform.viewpoint_metadata = meta
        transform.vr_root = Path(meta["vr_root"])


def transform_metadata_path(transform):
    # We stored the path on the transform during construction (via super().__init__).
    # If the metadata wasn't preloaded, look up via the carla.yaml constant.
    cfg, _ = load_carla_cfg()
    return cfg["viewpoint_metadata_path"]


def get_cam_image(sample, cam_channels, cam_name):
    """Return (3, H, W) tensor for cam_name out of the per-sample stack."""
    idx = cam_channels.index(cam_name)
    return sample["image"][idx].clone()


def main():
    cfg, image = load_carla_cfg()
    transform, labels_dir = build_transform(cfg, image)
    ds = CarlaGeneratedDataset(SCENE, labels_dir, transform=transform)
    if FRAME >= len(ds):
        raise RuntimeError(f"frame {FRAME} out of range (scene has {len(ds)})")

    # Sanity: peek at the underlying JSON to recover cam_channels reliably.
    cam_channels = list(ds.samples[FRAME]["cam_channels"])
    print(f"[info] scene={SCENE} frame={FRAME} cam_channels={cam_channels}")
    assert "CAM_FRONT" in cam_channels and "CAM_BACK" in cam_channels

    # --- Pass 1: per-trial, target_cams={CAM_FRONT} -------------------------
    print("\n=== per-cam target = CAM_FRONT ===")
    cam_front_images = {}
    cam_back_images = {}
    for name, variant, image_swap in VARIANTS:
        mutate(transform, variant, image_swap, {"CAM_FRONT"})
        sample = ds[FRAME]
        cam_front_images[name] = get_cam_image(sample, cam_channels, "CAM_FRONT")
        cam_back_images[name] = get_cam_image(sample, cam_channels, "CAM_BACK")
        print(f"  loaded [{name}] variant={variant} image_swap={image_swap} "
              f"front.shape={tuple(cam_front_images[name].shape)} "
              f"front.mean={cam_front_images[name].mean().item():.4f}")

    # --- MSE Normal vs each variant, CAM_FRONT (target) and CAM_BACK (non-target).
    print("\n=== MSE: CAM_FRONT (TARGET — should differ for VR variants) ===")
    normal_front = cam_front_images["Normal"]
    normal_back = cam_back_images["Normal"]
    front_results = {}
    back_results = {}
    for name, _, _ in VARIANTS:
        mse_f = torch.mean((cam_front_images[name] - normal_front) ** 2).item()
        mse_b = torch.mean((cam_back_images[name] - normal_back) ** 2).item()
        front_results[name] = mse_f
        back_results[name] = mse_b

    blockers = []
    for name, _, image_swap in VARIANTS:
        mse_f = front_results[name]
        if name == "Normal":
            ok = mse_f == 0.0
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] {name}: CAM_FRONT MSE={mse_f:.6e} (expect == 0)")
            if not ok:
                blockers.append((name, "Normal vs Normal CAM_FRONT not zero"))
        else:
            ok = mse_f > 1e-3
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] {name}: CAM_FRONT MSE={mse_f:.6e} (expect > 1e-3)")
            if not ok:
                blockers.append((name, f"CAM_FRONT MSE={mse_f:.2e} suggests VR image NOT swapped"))

    print("\n=== MSE: CAM_BACK (NON-TARGET — should stay = Normal) ===")
    for name, _, _ in VARIANTS:
        mse_b = back_results[name]
        ok = mse_b == 0.0
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name}: CAM_BACK MSE={mse_b:.6e} (expect == 0)")
        if not ok:
            blockers.append((name, f"CAM_BACK changed (MSE={mse_b:.2e}) — per-cam target leaked"))

    # --- Montage -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(VARIANTS), figsize=(4 * len(VARIANTS), 4))
        for ax, (name, _, _) in zip(axes, VARIANTS):
            img = cam_front_images[name].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(f"{name}\nMSE={front_results[name]:.2e}", fontsize=10)
            ax.axis("off")
        fig.suptitle(f"VR image swap probe — {SCENE} frame {FRAME} CAM_FRONT (target)")
        fig.tight_layout()
        fig.savefig(PNG_OUT, dpi=110)
        plt.close(fig)
        print(f"\n[saved montage] {PNG_OUT}")
    except Exception as e:
        print(f"\n[warn] montage failed: {e}")

    # --- Summary ------------------------------------------------------------
    print("\n=== SUMMARY ===")
    print(f"  PNG: {PNG_OUT}")
    print(f"  CAM_FRONT MSEs: " + ", ".join(f"{n}={front_results[n]:.2e}" for n, _, _ in VARIANTS))
    print(f"  CAM_BACK  MSEs: " + ", ".join(f"{n}={back_results[n]:.2e}" for n, _, _ in VARIANTS))
    if blockers:
        print("  BLOCKERS:")
        for n, why in blockers:
            print(f"    - {n}: {why}")
        sys.exit(1)
    print("  ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
