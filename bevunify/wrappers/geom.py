"""Shared geometry / path helpers for the model wrappers.

The unified DataModule (GaussianLSS, split_intrin_extrin=True) emits per camera:
  intrinsics  (B,N,3,3)
  extrinsics  (B,N,4,4)  with  x_cam = E @ x_lidar   (i.e. lidar/ego -> cam)

Most repos (LSS, LaRa, PointBeV) want the inverse rigid transform (cam -> ego):
  x_ego = R @ x_cam + t .
"""
import sys
from pathlib import Path
import torch

# repo root of bevunify (this file: bevunify/bevunify/wrappers/geom.py)
BEVUNIFY_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo(repo_root: str) -> str:
    """Resolve a (possibly relative, e.g. 'third_party/PointBeV') repo path against the
    bevunify repo root, so vendored repos work regardless of CWD."""
    p = Path(repo_root)
    return str(p if p.is_absolute() else BEVUNIFY_ROOT / repo_root)


def add_repo_to_path(repo_root: str) -> str:
    repo_root = resolve_repo(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def ego_from_cam(extrinsics: torch.Tensor):
    """Invert lidar->cam extrinsics to cam->ego rotation/translation.

    extrinsics: (..., 4, 4).  Returns rots (...,3,3), trans (...,3).
    """
    R = extrinsics[..., :3, :3]
    t = extrinsics[..., :3, 3]
    R_inv = R.transpose(-1, -2)
    t_inv = -torch.matmul(R_inv, t.unsqueeze(-1)).squeeze(-1)
    return R_inv, t_inv


def rots_trans(batch):
    """cam->ego rotation/translation. Prefers the materialized `ego_from_cam` produced
    by bevunify.datagen; falls back to inverting `extrinsics` (identical math) so old
    labels still work."""
    efc = batch.get("ego_from_cam") if isinstance(batch, dict) else None
    if efc is not None:
        return efc[..., :3, :3], efc[..., :3, 3]
    return ego_from_cam(batch["extrinsics"])


def intrinsics_to_4x4(intrinsics: torch.Tensor) -> torch.Tensor:
    """(...,3,3) -> (...,4,4) homogeneous projection (eye padded)."""
    *batch, _, _ = intrinsics.shape
    out = torch.eye(4, dtype=intrinsics.dtype, device=intrinsics.device)
    out = out.expand(*batch, 4, 4).clone()
    out[..., :3, :3] = intrinsics
    return out
