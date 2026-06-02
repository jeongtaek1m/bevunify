"""Core helpers for the toolkit: lidar2cam projection + scene/calibration loading.

Loads ground-truth metadata straight from disk (no DataModule needed), so any
projection viz can run without touching Hydra/Lightning.

Frame conventions (CARLA GeoBEV):
  * scene_*.json:  extrinsics[k] is 4x4 ego->cam (cam_from_ego); intrinsics[k] is 3x3
  * LIDAR npz:     points (N,3) in the LIDAR SENSOR frame
  * LIDAR pose:    sensor2ego_translation=(0,0,1.8), sensor2ego_rotation=I
                   so p_ego = p_lidar + (0, 0, 1.8)
  * Image:         1600 x 900
  * VR metadata:   {scenes:{scene:{cam:{variant:{sensor2ego_rotation,sensor2ego_translation}}}}}
                   correction = inv(s2e_variant_T) @ s2e_baseline_T,  E_new = correction @ E_base
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pyquaternion import Quaternion


# ── Paths ───────────────────────────────────────────────────────────────────
GEOBEV_ROOT = Path("/home/hanyan_arch/data/carla_geobev")
LABELS_ROOT = Path("/home/hanyan_arch/data/carla_geobev_labels/gaussianlss")
VR_METADATA = Path("/NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_VR/viewpoint_metadata.json")

PLATFORM_TAG = {"sedan": "subcompact", "suv": "suv", "bus": "bus"}
CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT"]

# LIDAR_TOP at (0,0,1.8) in ego with identity rotation (calibrated_sensor.json).
LIDAR_T_EGO = np.array([0.0, 0.0, 1.8], dtype=np.float32)

IMG_W, IMG_H = 1600, 900


# ── Data containers ─────────────────────────────────────────────────────────
@dataclass
class Frame:
    """One (scene, frame_idx) snapshot of a single platform's calibration + paths."""
    platform: str
    scene: str
    frame_idx: int
    cam_channels: List[str]
    intrinsics: np.ndarray              # (6, 3, 3)
    extrinsics: np.ndarray              # (6, 4, 4) ego->cam
    image_paths: List[Path]
    lidar_path: Path
    token: str


# ── Loading ─────────────────────────────────────────────────────────────────
def load_vr_metadata(path: Path = VR_METADATA) -> dict:
    return json.loads(Path(path).read_text())


def load_frame(platform: str, scene: str, frame_idx: int,
               labels_root: Path = LABELS_ROOT,
               sweeps_root: Path = GEOBEV_ROOT) -> Frame:
    plat_dir = labels_root / f"{platform}_eval"
    samples = json.loads((plat_dir / f"{scene}.json").read_text())
    if frame_idx >= len(samples):
        raise IndexError(f"{scene} only has {len(samples)} frames (req {frame_idx})")
    s = samples[frame_idx]
    return Frame(
        platform=platform, scene=scene, frame_idx=frame_idx,
        cam_channels=list(s["cam_channels"]),
        intrinsics=np.array(s["intrinsics"], dtype=np.float32),
        extrinsics=np.array(s["extrinsics"], dtype=np.float32),
        image_paths=[sweeps_root / p for p in s["images"]],
        lidar_path=sweeps_root / "sweeps" / "LIDAR" /
                   f"SimBEV-{scene.replace('_', '-')}-frame-{frame_idx:04d}-LIDAR.npz",
        token=s["token"],
    )


def load_lidar_ego(lidar_path: Path) -> np.ndarray:
    """Return lidar points lifted into ego frame (with LIDAR_T_EGO offset applied)."""
    pts_lidar = np.load(lidar_path)["data"]              # (N,3) sensor frame
    return pts_lidar.astype(np.float32) + LIDAR_T_EGO    # (N,3) ego frame


def load_image(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p))


# ── Projection ──────────────────────────────────────────────────────────────
def project_ego_to_cam(pts_ego: np.ndarray, K: np.ndarray,
                       E_ego_to_cam: np.ndarray,
                       z_min: float = 0.1
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project (N,3) ego-frame points onto the image plane.

    Returns (uv, depth, mask) where mask covers in-front-of-camera AND in-bounds.
    uv shape (M,2), depth (M,), all already filtered by mask (i.e. shape matches each other).
    """
    R = E_ego_to_cam[:3, :3]; t = E_ego_to_cam[:3, 3]
    p_cam = pts_ego @ R.T + t                # (N,3)
    z = p_cam[:, 2]
    front = z > z_min
    p_cam_f = p_cam[front]
    uv = (K @ p_cam_f.T).T
    u = uv[:, 0] / uv[:, 2]
    v = uv[:, 1] / uv[:, 2]
    in_img = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    out_uv = np.stack([u[in_img], v[in_img]], axis=1)
    out_z = p_cam_f[in_img, 2]
    full_mask = np.zeros(len(pts_ego), dtype=bool)
    front_idx = np.nonzero(front)[0]
    full_mask[front_idx[in_img]] = True
    return out_uv, out_z, full_mask


# ── VR mutations ────────────────────────────────────────────────────────────
def vr_extrinsic_correction(meta: dict, scene: str, cam: str, variant: str
                            ) -> np.ndarray:
    """Returns 4x4 correction; new_E = correction @ baseline_E (matches carla_data.py)."""
    def egocam_from_cam(s2e):       # cam (in cam frame) -> ego  (sensor2ego_T)
        R = Quaternion(s2e["sensor2ego_rotation"]).rotation_matrix
        t = np.asarray(s2e["sensor2ego_translation"])
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        return T
    def cam_from_egocam(s2e):       # inverse of the above
        R = Quaternion(s2e["sensor2ego_rotation"]).rotation_matrix
        t = np.asarray(s2e["sensor2ego_translation"])
        T = np.eye(4); T[:3, :3] = R.T; T[:3, 3] = -R.T @ t
        return T
    base = meta["scenes"][scene][cam]["yaw0pitch0roll0"]
    var = meta["scenes"][scene][cam][variant]
    return cam_from_egocam(var) @ egocam_from_cam(base)


def vr_image_path(meta: dict, scene: str, cam: str, variant: str, frame_idx: int
                  ) -> Path:
    """Path under vr_root for the perturbed image (no platform tag — sedan only)."""
    vr_root = Path(meta["vr_root"])
    scene_tok = scene.replace("_", "-")  # scene_0220 → scene-0220 (on-disk naming)
    return vr_root / "sweeps" / f"RGB-{cam}_{variant}" / \
           f"SimBEV-{scene_tok}-frame-{frame_idx:04d}-RGB-{cam}_{variant}.jpg"


def apply_vr(frame: Frame, meta: dict, variant: str, targets: Optional[List[str]],
             image_swap: bool, extrinsic_swap: bool
             ) -> Tuple[np.ndarray, List[Path]]:
    """Return (extrinsics_new, image_paths_new) after applying VR perturbation."""
    E_new = frame.extrinsics.copy()
    paths_new = list(frame.image_paths)
    cams = targets if targets else frame.cam_channels
    for cam in cams:
        if cam not in frame.cam_channels:
            continue
        k = frame.cam_channels.index(cam)
        if extrinsic_swap:
            corr = vr_extrinsic_correction(meta, frame.scene, cam, variant)
            E_new[k] = corr @ E_new[k]
        if image_swap:
            paths_new[k] = vr_image_path(meta, frame.scene, cam, variant,
                                          frame.frame_idx)
    return E_new, paths_new


# ── CTS mutations ───────────────────────────────────────────────────────────
def build_sedan_ext_map(labels_root: Path = LABELS_ROOT) -> Dict[str, np.ndarray]:
    """Read one sedan_eval scene → {cam: 4x4 sedan extrinsic}. Rig is rigid wrt ego."""
    first = next(iter(sorted((labels_root / "sedan_eval").glob("scene_*.json"))))
    s = json.loads(first.read_text())[0]
    return {ch: np.array(e, dtype=np.float32)
            for ch, e in zip(s["cam_channels"], s["extrinsics"])}


def cts_path_to_sedan(image_path: Path, plat: str) -> Path:
    """Target-platform image rel-path → sedan equivalent at same scene/frame."""
    s = str(image_path)
    return Path(s.replace(f"RGB-{plat}-", "RGB-")
                 .replace(f"RGB-{plat}__", "RGB-subcompact__"))


def apply_cts(frame: Frame, condition: str, sedan_ext: Dict[str, np.ndarray]
              ) -> Tuple[np.ndarray, List[Path]]:
    """CTS condition matrix:
        NORMAL: img=sedan, ext=sedan
        EXT:    img=sedan, ext=target  (only ext stays target → primary EXT axis)
        IMG:    img=target, ext=sedan  (only img stays target → primary IMG axis)
        CAL:    img=target, ext=target (= baseline target run, 'full transfer')
    """
    flags = {"NORMAL": (True, True), "EXT": (True, False),
             "IMG":    (False, True), "CAL": (False, False)}
    img_to_sedan, ext_to_sedan = flags[condition]
    E_new = frame.extrinsics.copy()
    paths_new = list(frame.image_paths)
    for k, cam in enumerate(frame.cam_channels):
        if ext_to_sedan and cam in sedan_ext:
            E_new[k] = sedan_ext[cam]
        if img_to_sedan:
            paths_new[k] = GEOBEV_ROOT / cts_path_to_sedan(
                Path(*paths_new[k].parts[paths_new[k].parts.index("sweeps"):]),
                frame.platform)
    return E_new, paths_new
