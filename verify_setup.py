"""Static verification of the unified project — no GPU / no training needed.

1. Composes every experiment config and prints model target + GT/loss wiring.
2. Imports the host spine and every wrapper module (syntax/import check).

Run (host env):
    cd /home/jeongtae/bevseg/bevunify
    /home/jeongtae/miniconda3/envs/GaussianLSS/bin/python verify_setup.py
"""
import importlib
import os
import traceback

import bevunify  # noqa: F401  (bootstraps GaussianLSS path)
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

EXPERIMENTS = ["gaussianlss", "cvt", "lara", "pointbev", "lss", "simplebev"]
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")


def gt_flags(data):
    return {k: data.get(k) for k in ("vehicle", "ped", "gt_center", "gt_offset", "gt_visibility",
                                     "split_intrin_extrin")}


def main():
    print("=" * 70, "\nCONFIG COMPOSE\n", "=" * 70, sep="")
    ok = 0
    for exp in EXPERIMENTS:
        GlobalHydra.instance().clear()
        try:
            with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
                cfg = compose(config_name="config", overrides=[f"+experiment={exp}"])
            print(f"\n[{exp}]")
            print("  model._target_ :", cfg.model._target_)
            print("  loss keys      :", [k for k in cfg.loss.keys() if not k.endswith("_weight")])
            print("  gt flags       :", gt_flags(cfg.data))
            print("  metric         :", list(cfg.metrics.keys()), "| key =", cfg.key)
            opt = cfg.optimizer
            opt_s = opt.get("_target_", "AdamW").split(".")[-1] + f"(lr={opt.get('lr')})"
            sch = cfg.get("scheduler")
            if sch is None:
                sch_s = "None(constant)"
            else:
                sch_s = sch.get("_target_", "OneCycleLR").split(".")[-1]
            print(f"  recipe         : opt={opt_s} sched={sch_s} "
                  f"max_steps={cfg.trainer.max_steps} max_epochs={cfg.trainer.max_epochs}")
            ok += 1
        except Exception:
            print(f"\n[{exp}] COMPOSE FAILED")
            traceback.print_exc()
    print(f"\ncompose: {ok}/{len(EXPERIMENTS)} OK")

    print("\n" + "=" * 70, "\nWRAPPER MODULE IMPORTS\n", "=" * 70, sep="")
    for mod in ["bevunify.common", "bevunify.transforms_toggle", "bevunify.data_toggle",
                "bevunify.wrappers.geom", "bevunify.wrappers.repo_compose",
                "bevunify.wrappers.cvt", "bevunify.wrappers.lara", "bevunify.wrappers.pointbev",
                "bevunify.wrappers.lss", "bevunify.wrappers.simplebev"]:
        try:
            importlib.import_module(mod)
            print(f"  OK   {mod}")
        except Exception as e:
            print(f"  FAIL {mod}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
