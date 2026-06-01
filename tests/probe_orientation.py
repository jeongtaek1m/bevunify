"""Orientation probe — the single most important correctness check before trusting
any cross-model IoU number.

Every model's predicted BEV must land in the SAME (row=Y-left, col=X-forward) cell as
the GaussianLSS GT. Resolution/range already match (all 200x200 / +-50m / 0.5m); the
risk is axis-order / flips (PointBeV's .flip(1,2), simple_bev's (Z,X) layout) and the
cam->ego extrinsic inversion.

How to use (host env, after a real batch is available):

    cd /home/jeongtae/bevseg/bevunify
    /home/jeongtae/miniconda3/envs/GaussianLSS/bin/python tests/probe_orientation.py +experiment=lss

Strategy:
  1. Pull ONE real sample from the unified DataModule (val split).
  2. Show the GT mask `batch['vehicle']` and note where vehicles are (e.g. a box front-left).
  3. Run the model wrapper's forward; sigmoid the `vehicle` logits.
  4. Overlay / argmax both. The bright region MUST coincide with the GT region.
  5. If it is transposed or mirrored, set the wrapper's axis_fix accordingly:
       - PointBeVWrapper(axis_fix="flip_y" | "transpose" | "none")
       - SimpleBEVWrapper(axis_fix="transpose" | "none")
     and re-run until GT and prediction align on an ASYMMETRIC sample.

This file is intentionally a scaffold: it documents the procedure and gives the loading
boilerplate. Fill in the visualization to your preference (matplotlib / save PNG).
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bevunify  # noqa: F401,E402
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from bevunify.common import setup_config, setup_data_module, setup_network
import os

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")


def load_one(experiment):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=[f"+experiment={experiment}"])
    setup_config(cfg)
    dm = setup_data_module(cfg)
    val = dm.get_split("val", loader=False)[0]      # first scene dataset
    sample = val[0]
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in sample.items()}
    return cfg, batch


def main():
    exp = "lss"
    for a in sys.argv[1:]:
        if a.startswith("+experiment="):
            exp = a.split("=", 1)[1]
    cfg, batch = load_one(exp)
    gt = batch[cfg.key][0, 0].numpy()                # (200,200) GT mask
    print("GT mask: nonzero pixels =", int((gt > 0.5).sum()),
          "| centroid(row,col) =", np.argwhere(gt > 0.5).mean(0) if (gt > 0.5).any() else None)

    model = setup_network(cfg).eval()
    with torch.no_grad():
        pred = model(batch)[cfg.key][0, 0].sigmoid().numpy()
    hot = pred > 0.5
    print("PRED mask: nonzero pixels =", int(hot.sum()),
          "| centroid(row,col) =", np.argwhere(hot).mean(0) if hot.any() else None)
    print("\nCompare the two centroids. If they disagree, adjust the wrapper's axis_fix.")


if __name__ == "__main__":
    main()
