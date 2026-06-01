"""Dataset module mirroring GaussianLSS ``nuscenes_dataset_generated`` but using the
toggle-aware transform. Registered as ``nuscenes_toggle`` by ``bevunify.common``.

Reuses the host ``NuScenesGeneratedDataset`` (JSON + gt_box.npz reader) unchanged;
only the transform differs.
"""
from pathlib import Path

from GaussianLSS.data.common import get_split
from GaussianLSS.data.nuscenes_dataset_generated import NuScenesGeneratedDataset

from .transforms_toggle import ToggleLoadDataTransform


def get_data(
    dataset_dir,
    labels_dir,
    split,
    version,
    image=None,                         # image config
    **dataset_kwargs                    # carries gt_center / gt_offset / gt_visibility, etc.
):
    out = []
    dataset_dir = Path(dataset_dir)
    labels_dir = Path(labels_dir)

    training = split == "train"
    transform = ToggleLoadDataTransform(
        dataset_dir, labels_dir, image, training=training, **dataset_kwargs
    )

    split = f"mini_{split}" if version == "v1.0-mini" else split
    split_scenes = get_split(split, "nuscenes")

    for s in split_scenes:
        out.append(NuScenesGeneratedDataset(s, labels_dir, transform=transform))
    return out
