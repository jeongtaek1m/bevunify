"""Augmented GaussianLSS data generation.

Modifies the GaussianLSS generation to ALSO store, per camera, the extrinsic
conventions other models use, so each model gets a well-defined transform on disk
(no per-wrapper inversion guesswork):

  extrinsics    (4x4)  cam_from_ego  : ego/lidar-flat -> cam   (GaussianLSS native; = E)
  ego_from_cam  (4x4)  cam -> ego/lidar-flat                   (= inv(E); LSS/LaRa/PointBeV/simple_bev)
  intrinsics    (3x3)                                          (CVT, simple_bev pix_T_cams)

These are exact transforms of the cached E, materialized explicitly per request.
Writes to a NEW labels dir (does NOT touch the shared /data/.../labels).

Usage (host env):
    cd /home/jeongtae/bevseg/bevunify
    $PY -m bevunify.datagen \
        --dataset_dir /data/datasets/nuscenes \
        --labels_dir  /data/datasets/nuscenes/labels_aug \
        --version v1.0-trainval [--splits val train] [--scenes scene-0003 ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import bevunify  # noqa: F401  bootstraps GaussianLSS path
from GaussianLSS.data.common import get_split
from GaussianLSS.data.nuscenes_dataset import NuScenesSingleton, NuScenesDataset
from GaussianLSS.data.transforms import SaveDataTransform


class AugNuScenesDataset(NuScenesDataset):
    """Host raw dataset + extra per-camera extrinsic conventions."""

    def parse_sample_record(self, sample_record, camera_rig):
        d = super().parse_sample_record(sample_record, camera_rig)
        E = np.array(d["extrinsics"], dtype=np.float64)          # (N,4,4) cam_from_ego
        d["ego_from_cam"] = np.linalg.inv(E).tolist()            # (N,4,4) cam -> ego
        return d


def generate(dataset_dir, labels_dir, version, splits, bev, cameras, scenes=None):
    dataset_dir, labels_dir = Path(dataset_dir), Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    helper = NuScenesSingleton(dataset_dir, version)
    transform = SaveDataTransform(labels_dir)

    for split in splits:
        split_name = f"mini_{split}" if version == "v1.0-mini" else split
        split_scenes = set(get_split(split_name, "nuscenes"))
        print(f"[{split}] {len(split_scenes)} scenes")

        for scene_name, scene_record in tqdm(list(helper.get_scenes())):
            if scene_name not in split_scenes:
                continue
            if scenes and scene_name not in scenes:
                continue
            scene_dir = labels_dir / scene_name
            scene_dir.mkdir(exist_ok=True, parents=False)

            ds = AugNuScenesDataset(scene_name, scene_record, helper,
                                    bev=bev, cameras=cameras, transform=transform)
            info = [ds[i] for i in range(len(ds))]               # saves gt_box npz, returns json dict
            (labels_dir / f"{scene_name}.json").write_text(json.dumps(info))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="/data/datasets/nuscenes")
    ap.add_argument("--labels_dir", default="/data/datasets/nuscenes/labels_aug")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--splits", nargs="+", default=["val", "train"])
    ap.add_argument("--scenes", nargs="+", default=None, help="optional subset for testing")
    args = ap.parse_args()

    bev = dict(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0)
    cameras = [[0, 1, 2, 3, 4, 5]]
    generate(args.dataset_dir, args.labels_dir, args.version, args.splits, bev, cameras, args.scenes)


if __name__ == "__main__":
    main()
