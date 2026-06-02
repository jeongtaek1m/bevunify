"""Shared plotting: 2x3 grid of 6 cameras with lidar2cam projection overlay."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .projection import (CAMS, IMG_W, IMG_H, load_image, load_lidar_ego,
                          project_ego_to_cam, Frame)


def render_projection(frame: Frame, extrinsics: np.ndarray, image_paths,
                      out_path: Path, title: str = "",
                      highlight_cams: Optional[List[str]] = None,
                      point_size: float = 1.4, alpha: float = 0.55,
                      max_depth: float = 60.0) -> None:
    """Project the frame's lidar onto every cam with the given (E_k, img_k) pair
    and save a 2x3 grid. `highlight_cams` get a red border to mark perturbed targets.
    """
    pts_ego = load_lidar_ego(frame.lidar_path)
    highlight_cams = set(highlight_cams or [])

    fig, axes = plt.subplots(2, 3, figsize=(18, 7))
    for k, cam in enumerate(CAMS):
        ax = axes[k // 3, k % 3]
        try:
            img = load_image(image_paths[k])
        except FileNotFoundError:
            ax.text(0.5, 0.5, f"missing: {image_paths[k].name}", ha="center",
                    va="center", transform=ax.transAxes, fontsize=7, color="red")
            ax.axis("off")
            continue
        ax.imshow(img)
        uv, z, _ = project_ego_to_cam(pts_ego, frame.intrinsics[k], extrinsics[k])
        if len(uv):
            c = np.clip(z / max_depth, 0.0, 1.0)
            ax.scatter(uv[:, 0], uv[:, 1], c=c, cmap="turbo_r",
                       s=point_size, alpha=alpha, edgecolors="none")
        ax.set_xlim(0, IMG_W); ax.set_ylim(IMG_H, 0)
        ax.set_title(cam, fontsize=8,
                     color=("red" if cam in highlight_cams else "black"),
                     fontweight=("bold" if cam in highlight_cams else "normal"))
        ax.set_xticks([]); ax.set_yticks([])
        if cam in highlight_cams:
            for spine in ax.spines.values():
                spine.set_edgecolor("red"); spine.set_linewidth(2.5)
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
