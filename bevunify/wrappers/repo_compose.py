"""Compose a source repo's own Hydra config in isolation, so a wrapper can
instantiate that repo's exact model architecture without re-porting its nested
config into bevunify.

NOTE: this clears the global Hydra instance. It is only called from wrapper
``__init__`` during ``setup_experiment`` — i.e. AFTER the bevunify cfg has already
been composed and resolved (``setup_config`` calls ``OmegaConf.resolve``), so the
already-materialised cfg is unaffected.
"""
import os
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra


def compose_repo_cfg(config_dir: str, config_name: str, overrides=None):
    config_dir = os.path.abspath(config_dir)
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base="1.3", config_dir=config_dir):
            cfg = compose(config_name=config_name, overrides=list(overrides or []))
    finally:
        GlobalHydra.instance().clear()
    return cfg
