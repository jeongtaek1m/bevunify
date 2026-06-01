"""Verify the native-faithful loss wiring (CPU, no GPU): every experiment composes,
and the loss (incl. the new FocalCenterLoss / BalancedMSECenterLoss / FootprintOffsetLoss)
runs to a finite scalar on dummy tensors via MultipleLoss.

    cd /home/jeongtae/bevseg/bevunify
    $PY tests/verify_losses.py
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bevunify  # noqa: E402
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from GaussianLSS.losses import MultipleLoss

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
B = 2


def mk_pred():
    return {
        "vehicle": torch.randn(B, 1, 200, 200),
        "vehicle_center": torch.randn(B, 1, 200, 200),
        "vehicle_offset": torch.randn(B, 2, 200, 200),
    }


def mk_batch():
    vis = torch.randint(0, 4, (B, 200, 200))            # 0..3 -> vis>=2 non-empty
    veh = (torch.rand(B, 1, 200, 200) > 0.9).float()
    return {
        "vehicle": veh,
        "vehicle_center": torch.rand(B, 1, 200, 200),
        "vehicle_offset": torch.randn(B, 2, 200, 200),
        "vehicle_visibility": vis,
    }


def main():
    for exp in ["gaussianlss", "cvt", "lara", "lss", "pointbev", "simplebev"]:
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
            cfg = compose(config_name="config", overrides=[f"+experiment={exp}"])
        loss_func = MultipleLoss(instantiate(cfg.loss))
        total, details, _ = loss_func(mk_pred(), mk_batch())
        assert torch.isfinite(total), f"{exp}: non-finite loss {total}"
        terms = {k: round(float(v), 4) for k, v in details.items()}
        bevcls = cfg.loss.bev._target_.split(".")[-1]
        centercls = cfg.loss.center._target_.split(".")[-1] if cfg.loss.get("center") else "-"
        print(f"[{exp:11s}] loss={float(total):.4f}  terms={terms}  (bev={bevcls}, center={centercls})")
    print("\nALL LOSS CHECKS PASSED")


if __name__ == "__main__":
    main()
