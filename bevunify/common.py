"""Setup spine for bevunify.

Reuses the GaussianLSS host spine, with two changes:
 - the model module is ``UnifiedModelModule`` (generalized optimizer/scheduler), so
   each model can use its own training recipe (config/recipe/<model>.yaml);
 - the toggle-aware dataset is registered under ``nuscenes_toggle``.
Any model exposed as an ``nn.Module`` returning the canonical pred dict plugs in.
"""
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchmetrics import MetricCollection

from GaussianLSS.common import (  # noqa: F401  (re-exported)
    setup_config,
    setup_network,
    setup_data_module,
    load_backbone,
)
from GaussianLSS.losses import MultipleLoss
from GaussianLSS.data import MODULES

from . import data_toggle
from .model_module import UnifiedModelModule

# Make `data.dataset: nuscenes_toggle` resolvable by the host DataModule.
MODULES["nuscenes_toggle"] = data_toggle


def setup_model_module(cfg):
    backbone = setup_network(cfg)
    loss_func = MultipleLoss(instantiate(cfg.loss))
    metrics = MetricCollection({k: v for k, v in instantiate(cfg.metrics).items()},
                               compute_groups=False)
    return UnifiedModelModule(backbone, loss_func, metrics,
                              cfg.optimizer, cfg.scheduler,
                              cfg=cfg, val_only=cfg.val_only)


def setup_viz(cfg):
    # visualization is optional (the `null` group default leaves no key).
    viz_cfg = OmegaConf.select(cfg, "visualization")
    return instantiate(viz_cfg) if viz_cfg is not None else None


def setup_experiment(cfg):
    return setup_model_module(cfg), setup_data_module(cfg), setup_viz(cfg)
