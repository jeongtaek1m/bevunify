"""Unified evaluation entry — same GT, same IoU metric, fair across models.

    python -m bevunify.eval +experiment=cvt ckpt=/path/to/last.ckpt
"""
from pathlib import Path
import logging

import torch
import hydra
import pytorch_lightning as pl

from bevunify.common import setup_config, setup_experiment, load_backbone

log = logging.getLogger(__name__)

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "config")
CONFIG_NAME = "config.yaml"


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    torch.set_float32_matmul_precision("high")
    setup_config(cfg)

    model_module, data_module, _ = setup_experiment(cfg)

    ckpt_path = cfg.get("ckpt", None)
    if ckpt_path is not None:
        model_module.backbone = load_backbone(ckpt_path, backbone=model_module.backbone)

    logger = pl.loggers.CSVLogger(save_dir=cfg.experiment.save_dir, name=cfg.experiment.project)
    trainer = pl.Trainer(logger=logger, **cfg.trainer)
    trainer.validate(model_module, datamodule=data_module)


if __name__ == "__main__":
    main()
