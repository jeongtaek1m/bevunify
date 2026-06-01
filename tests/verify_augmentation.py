"""Verify the central augmentation manager (config + transform behaviour).

    cd /home/jeongtae/bevseg/bevunify
    $PY tests/verify_augmentation.py

Checks:
  1. each augmentation/*.yaml preset composes and sets the expected data flags;
  2. extrinsic-noise perturbs the per-camera extrinsic (and recomputes ego_from_cam)
     while leaving the BEV GT byte-identical (image + GT must stay clean);
  3. image-warp produces 224x480 images with intrinsics carried through.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bevunify  # noqa: E402  bootstraps GaussianLSS path
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from GaussianLSS.data.transforms import Sample
from bevunify.transforms_toggle import ToggleLoadDataTransform

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
DATA_DIR = "/data/datasets/nuscenes"
LABELS_DIR = "/data/datasets/nuscenes/labels_aug"
SCENE = "scene-0001"


def cfg_for(aug):
    overrides = [
        "+experiment=cvt",
        f"augmentation={aug}",
        f"data.dataset_dir={DATA_DIR}",
        f"data.labels_dir={LABELS_DIR}",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=overrides)


def build_transform(data_cfg, training=True):
    d = OmegaConf.to_container(data_cfg, resolve=True)
    args = (d.pop("dataset_dir"), d.pop("labels_dir"), d.pop("image"))
    for k in ("dataset", "version", "cameras"):
        d.pop(k, None)
    return ToggleLoadDataTransform(*args, training=training, **d)


def raw_record():
    import json
    from pathlib import Path
    samples = json.loads((Path(LABELS_DIR) / f"{SCENE}.json").read_text())
    return samples[0]


def main():
    # --- 1. presets compose with the expected flags ---------------------------
    expect = {
        "none":        (False, False, False),
        "warp":        (True,  False, False),
        "extrin_noise":(False, False, True),
        "warp_extrin": (True,  False, True),
    }
    for aug, (aimg, abev, enoise) in expect.items():
        c = cfg_for(aug)
        assert bool(c.data.augment_img) == aimg, f"{aug}: augment_img"
        assert bool(c.data.augment_bev) == abev, f"{aug}: augment_bev"
        assert bool(c.data.extrin_noise.enabled) == enoise, f"{aug}: extrin_noise.enabled"
        print(f"[compose] {aug:13s} augment_img={aimg} augment_bev={abev} extrin_noise={enoise}  OK")

    # --- 2. extrinsic noise perturbs extrinsics, leaves GT untouched ----------
    cfg = cfg_for("extrin_noise")
    t_noise = build_transform(cfg.data, training=True)
    cfg_off = cfg_for("none")
    t_clean = build_transform(cfg_off.data, training=True)

    rec = raw_record()
    out_clean = t_clean(Sample(**rec))
    out_noise = t_noise(Sample(**rec))

    ext_clean = out_clean["extrinsics"].numpy()
    ext_noise = out_noise["extrinsics"].numpy()
    dext = np.abs(ext_clean - ext_noise).max()
    assert dext > 1e-4, f"extrinsic noise had no effect (max|d|={dext})"
    print(f"[extrin]  extrinsics perturbed: max|delta|={dext:.4f}  OK")

    # GT must be byte-identical (image + GT stay clean under extrinsic noise)
    gt_clean = out_clean["vehicle"].numpy()
    gt_noise = out_noise["vehicle"].numpy()
    assert np.array_equal(gt_clean, gt_noise), "extrinsic noise changed the BEV GT!"
    print(f"[extrin]  vehicle GT identical under noise (sum={int(gt_noise.sum())})  OK")

    # ego_from_cam must equal inv(noised extrinsics)
    if "ego_from_cam" in out_noise:
        efc = out_noise["ego_from_cam"].numpy()
        recomputed = np.linalg.inv(ext_noise)
        assert np.allclose(efc, recomputed, atol=1e-4), "ego_from_cam != inv(noised E)"
        print(f"[extrin]  ego_from_cam == inv(noised extrinsics)  OK")

    # --- 3. image warp -> 224x480 with intrinsics carried through -------------
    cfg_w = cfg_for("warp")
    t_warp = build_transform(cfg_w.data, training=True)
    out_warp = t_warp(Sample(**rec))
    img = out_warp["image"]
    assert tuple(img.shape[-2:]) == (224, 480), f"warp image shape {tuple(img.shape)}"
    assert "intrinsics" in out_warp, "warp dropped intrinsics"
    assert "cam_idx" in out_warp, "warp path dropped cam_idx (CVT needs it)"
    print(f"[warp]    image {tuple(img.shape)}, intrinsics + cam_idx present  OK")

    print("\nALL AUGMENTATION CHECKS PASSED")


if __name__ == "__main__":
    main()
