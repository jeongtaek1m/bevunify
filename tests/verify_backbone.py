"""Verify the config-managed backbone knob (CPU-only, no net build / no GPU).

    cd /home/jeongtae/bevseg/bevunify
    $PY tests/verify_backbone.py

Checks:
  1. each experiment composes and exposes its backbone knob in cfg.model;
  2. for the 3 repos whose backbone lives in their own hydra config (cvt/lara/
     pointbev), the dotted node the wrapper assigns to actually PRE-EXISTS in the
     composed repo cfg (so the knob overwrites a real node, not a phantom key).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bevunify  # noqa: E402  bootstraps GaussianLSS path
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from bevunify.wrappers.geom import add_repo_to_path
from bevunify.wrappers.repo_compose import compose_repo_cfg

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")

# experiment -> (dotted path of the backbone knob in the bevunify cfg.model, expected value)
BEVUNIFY_KNOB = {
    "gaussianlss": ("model.backbone._target_", None),       # block knob; just assert present
    "simplebev":   ("model.encoder_type", "res101"),
    "cvt":         ("model.backbone", "efficientnet-b4"),
    "lara":        ("model.backbone", "b4"),
    "pointbev":    ("model.backbone", "b4"),
    "lss":         ("model.backbone", "efficientnet-b0"),
}


def check_bevunify_knobs():
    for exp, (path, expected) in BEVUNIFY_KNOB.items():
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
            cfg = compose(config_name="config", overrides=[f"+experiment={exp}"])
        val = OmegaConf.select(cfg, path)
        assert val is not None, f"{exp}: knob {path} missing"
        if expected is not None:
            assert val == expected, f"{exp}: {path}={val} (expected {expected})"
        print(f"[bevunify] {exp:11s} {path} = {val}  OK")


def check_repo_nodes():
    # (repo dir, config_name, overrides, env setup, dotted node, native value)
    GAUSSIANLSS = None
    cases = []

    cvt_root = add_repo_to_path("third_party/cross_view_transformers")
    cases.append(("cvt", f"{cvt_root}/config", "config", ["+experiment=cvt_nuscenes_vehicle"],
                  lambda: None, "model.encoder.backbone.model_name", "efficientnet-b4"))

    lara_root = add_repo_to_path("third_party/LaRa")
    def _lara_env():
        os.environ.setdefault("WEIGHTS_PATH", "")
    cases.append(("lara", f"{lara_root}/configs", "train", ["experiment=LaRa_inCamrays_outCoord"],
                  _lara_env, "model.net.cam_encoder.version", "b4"))

    pbev_root = add_repo_to_path("third_party/PointBeV")
    def _pbev_env():
        os.environ.setdefault("PROJECT_ROOT", pbev_root)
        import importlib
        importlib.import_module("hydra_plugins.resolvers")
    cases.append(("pointbev", f"{pbev_root}/configs", "train", [],
                  _pbev_env, "model.net.backbone.version", "b4"))

    for name, cdir, cname, ov, envfn, node, native in cases:
        envfn()
        cfg = compose_repo_cfg(config_dir=cdir, config_name=cname, overrides=ov)
        OmegaConf.set_struct(cfg, False)
        val = OmegaConf.select(cfg, node)
        assert val is not None, f"{name}: repo node {node} does NOT exist (knob would be a no-op)"
        assert str(val) == native, f"{name}: repo node {node}={val} (expected native {native})"
        # confirm the wrapper's assignment actually takes
        OmegaConf.update(cfg, node, "b0" if native in ("b4",) else "efficientnet-b0", force_add=False)
        assert str(OmegaConf.select(cfg, node)) in ("b0", "efficientnet-b0")
        print(f"[repo]     {name:11s} {node} pre-exists = {native}, settable  OK")


if __name__ == "__main__":
    check_bevunify_knobs()
    check_repo_nodes()
    print("\nALL BACKBONE-KNOB CHECKS PASSED")
