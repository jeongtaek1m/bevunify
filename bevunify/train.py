"""Unified training entry. One switch trains any model on GaussianLSS GT:

    python -m bevunify.train +experiment=gaussianlss
    python -m bevunify.train +experiment=cvt
    python -m bevunify.train +experiment=lss
    ...

Mirrors GaussianLSS/scripts/train.py; uses bevunify config + the toggle dataset.
"""
from pathlib import Path
import logging

import torch
import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary

# importing bevunify.common registers the 'nuscenes_toggle' dataset
from bevunify.common import setup_config, setup_experiment, load_backbone

log = logging.getLogger(__name__)

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "config")
CONFIG_NAME = "config.yaml"


def maybe_resume_training(experiment):
    # Scope the resume glob to THIS project's directory — a bare save_dir/**/{uuid}
    # could silently resume a same-uuid checkpoint from a DIFFERENT project.
    save_dir = Path(experiment.save_dir).resolve() / experiment.project
    checkpoints = sorted(save_dir.glob(f"**/{experiment.uuid}/checkpoints/*.ckpt"))
    if not checkpoints:
        return None
    log.info(f"Found {checkpoints[-1]}.")
    return checkpoints[-1]


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    torch.set_float32_matmul_precision("high")
    setup_config(cfg)
    if cfg.experiment.get("seed"):
        pl.seed_everything(cfg.experiment.seed, workers=True)
    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)

    model_module, data_module, viz_fn = setup_experiment(cfg)

    ckpt_path = maybe_resume_training(cfg.experiment)
    if ckpt_path is not None:
        model_module.backbone = load_backbone(ckpt_path)

    # logger: per-model W&B project (experiment.project), or CSV fallback.
    if cfg.experiment.get("logger", "wandb") == "wandb":
        logger = pl.loggers.WandbLogger(project=cfg.experiment.project,
                                        name=cfg.experiment.uuid,
                                        save_dir=cfg.experiment.save_dir)
    else:
        logger = pl.loggers.CSVLogger(save_dir=cfg.experiment.save_dir, name=cfg.experiment.project)

    from bevunify.viz_callback import ValVizCallback
    callbacks = [
        ModelSummary(max_depth=2),
        LearningRateMonitor(logging_interval="epoch"),
        # full state (optimizer/scheduler/epoch) so `trainer.fit(ckpt_path=...)` can
        # actually resume — save_weights_only=True breaks resume (KeyError on optim state).
        # dirpath fixed to a uuid-named folder (not the wandb run-id default) so
        # maybe_resume_training's glob **/{uuid}/checkpoints/*.ckpt finds it on relaunch.
        ModelCheckpoint(
            dirpath=str(Path(cfg.experiment.save_dir) / cfg.experiment.project
                        / str(cfg.experiment.uuid) / "checkpoints"),
            save_weights_only=False, filename="last",
            monitor="val/metrics/mIoU", mode="max"),
        # 6 cam | GT | pred every N val steps (val is not shuffled -> stable across epochs)
        ValVizCallback(key=cfg.key,
                       every_n_steps=cfg.experiment.get("val_viz_interval", 100),
                       train_every_n_steps=cfg.experiment.get("train_viz_interval", 0),
                       out_dir=str(Path(cfg.experiment.save_dir) / "val_viz" / cfg.experiment.project)),
    ]

    dev = cfg.trainer.devices
    # static_graph=True: required because CVT's EfficientNet uses reentrant gradient
    # checkpointing, which trips DDP's default "marked ready twice" guard.
    strategy = "auto" if (isinstance(dev, int) and dev == 1) else DDPStrategy(static_graph=True)
    trainer = pl.Trainer(logger=logger, callbacks=callbacks, strategy=strategy, **cfg.trainer)
    if trainer.global_rank == 0:
        log.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))

    trainer.fit(model_module, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
