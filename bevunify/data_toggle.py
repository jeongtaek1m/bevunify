"""Dataset adapters for the unified pipeline — one file, two ``get_data_*`` functions.

Both functions return a list of ``Dataset`` instances built around the host
generated-dataset readers and the toggle-aware transforms in ``transforms_toggle``.
They are registered as host modules by ``bevunify.common`` (``nuscenes_toggle``
and ``carla_toggle``), so the same wrappers and unified loss/metric plug into
either dataset interchangeably.
"""
from pathlib import Path

from GaussianLSS.data.common import get_split
from GaussianLSS.data.nuscenes_dataset_generated import NuScenesGeneratedDataset

from .carla_data import CarlaGeneratedDataset, get_carla_split
from .transforms_toggle import ToggleLoadDataTransform, CarlaToggleLoadDataTransform


# ── nuScenes ───────────────────────────────────────────────────────────────────

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


# ── CARLA ──────────────────────────────────────────────────────────────────────

# Kwargs the carla LoadDataTransform (host) + toggle subclass actually consume.
# Anything else in ``cfg.data`` (img_params/cameras/bev_aug_conf/vehicle/ped/cts_*/etc.)
# is config metadata and gets filtered out to avoid TypeError.
_CARLA_TRANSFORM_KWARGS = (
    "split_intrin_extrin",
    "val_perturb",
    "extrinsic_noise_deg",
    "label_indices",
    "eval_viewpoint_variant",
    "viewpoint_metadata_path",
    "eval_image_swap",
    "eval_extrinsic_swap",
    "eval_target_cameras",
    # toggle subclass
    "gt_center",
    "gt_offset",
    "gt_visibility",
    "extrin_noise",
)


def get_data_carla(
    dataset_dir,
    labels_dir,
    split,
    version,
    image=None,
    num_classes=12,
    augment="none",
    # split-specific (None → top-level)
    val_dataset_dir=None,
    val_labels_dir=None,
    val_version=None,
    **dataset_kwargs,
):
    if split in ("val", "test"):
        if val_dataset_dir is not None: dataset_dir = val_dataset_dir
        if val_labels_dir  is not None: labels_dir  = val_labels_dir
        if val_version     is not None: version     = val_version

    dataset_dir = Path(dataset_dir)
    labels_dir = Path(labels_dir)

    transform_kwargs = {k: dataset_kwargs[k] for k in _CARLA_TRANSFORM_KWARGS if k in dataset_kwargs}
    transform_kwargs["training"] = split == "train"

    transform = CarlaToggleLoadDataTransform(
        dataset_dir, labels_dir, image, num_classes,
        augment if split == "train" else "none",
        **transform_kwargs,
    )

    split_scenes = get_carla_split(split)

    return [CarlaGeneratedDataset(s, labels_dir, transform=transform) for s in split_scenes]
