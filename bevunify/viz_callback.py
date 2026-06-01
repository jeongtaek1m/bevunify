"""Validation visualization callback: every N val steps, render `6 cam | GT | pred`
for the first sample of the batch. Logs to W&B (if the logger is WandbLogger) and
always saves a PNG to disk. Works for any model (pred[key] is already in the
GaussianLSS GT frame because each wrapper applies its own axis fix).

val is NOT shuffled, so the same batch_idx shows the same scene across epochs.
"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl

CAM_NAMES = ["FRONT_LEFT", "FRONT", "FRONT_RIGHT", "BACK_LEFT", "BACK", "BACK_RIGHT"]


class ValVizCallback(pl.Callback):
    def __init__(self, key="vehicle", every_n_steps=100, train_every_n_steps=0, out_dir="val_viz"):
        super().__init__()
        self.key = key
        self.n = max(1, int(every_n_steps))        # val: every N val batches
        self.train_n = int(train_every_n_steps)    # train: every N global steps (0 = off)
        self.out_dir = out_dir

    @staticmethod
    def _cam(t):
        return t.permute(1, 2, 0).clamp(0, 1).numpy()

    @staticmethod
    def _colorize(gt, vis):
        """RGB GT: red=vis1 (ignored by vis>=2 loss), green=vis>=2 (supervised)."""
        h, w = gt.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        veh = gt > 0.5
        rgb[veh & (vis == 1)] = (1.0, 0.25, 0.25)
        rgb[veh & (vis >= 2) & (vis != 255)] = (0.3, 1.0, 0.45)
        return rgb

    def _figure(self, imgs, gt, pr, vis, title):
        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1.4, 1.4], wspace=0.05, hspace=0.12)
        grid = [[0, 1, 2], [3, 4, 5]]
        for r in range(2):
            for c in range(3):
                ax = fig.add_subplot(gs[r, c])
                ax.imshow(self._cam(imgs[grid[r][c]]))
                ax.set_title(CAM_NAMES[grid[r][c]], fontsize=8); ax.axis("off")
        axg = fig.add_subplot(gs[:, 3])
        if vis is not None:
            axg.imshow(self._colorize(gt, vis), origin="upper")
            axg.set_title("GT  red=vis1 green=vis>=2  front^ left<", fontsize=8)
        else:
            axg.imshow(gt, cmap="magma", vmin=0, vmax=1, origin="upper")
            axg.set_title("GT (vehicle)  front^ left<", fontsize=9)
        axg.scatter([100], [100], c="cyan", s=14, marker="^"); axg.axis("off")
        axp = fig.add_subplot(gs[:, 4])
        axp.imshow(pr, cmap="magma", vmin=0, vmax=1, origin="upper")
        axp.scatter([100], [100], c="cyan", s=14, marker="^")
        axp.set_title("pred (sigmoid)", fontsize=9); axp.axis("off")
        fig.suptitle(f"{title}   | 6 cam | GT | pred", fontsize=10)
        return fig

    def _emit(self, trainer, pl_module, batch, title, fname, wandb_key):
        if self.key not in batch:
            return
        was_training = pl_module.training
        with torch.no_grad():
            pred = pl_module(batch)
        if was_training:
            pl_module.train()
        if self.key not in pred:
            return
        imgs = batch["image"][0].detach().float().cpu()
        gt = batch[self.key][0, 0].detach().float().cpu().numpy()
        pr = pred[self.key][0, 0].sigmoid().detach().float().cpu().numpy()
        visk = f"{self.key}_visibility"
        vis = batch[visk][0].detach().cpu().numpy() if visk in batch else None
        fig = self._figure(imgs, gt, pr, vis, title)
        try:
            from pytorch_lightning.loggers import WandbLogger
            if isinstance(trainer.logger, WandbLogger):
                import wandb
                trainer.logger.experiment.log({wandb_key: wandb.Image(fig)}, step=trainer.global_step)
        except Exception:
            pass
        os.makedirs(self.out_dir, exist_ok=True)
        fig.savefig(os.path.join(self.out_dir, fname), dpi=100, bbox_inches="tight")
        plt.close(fig)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.is_global_zero or batch_idx % self.n != 0:
            return
        self._emit(trainer, pl_module, batch,
                   f"val  epoch {trainer.current_epoch}  batch {batch_idx}",
                   f"ep{trainer.current_epoch:02d}_b{batch_idx:04d}.png", "val_viz")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.train_n <= 0 or not trainer.is_global_zero:
            return
        if trainer.global_step % self.train_n != 0:
            return
        self._emit(trainer, pl_module, batch,
                   f"train  step {trainer.global_step}",
                   f"train_step{trainer.global_step:06d}.png", "train_viz")
