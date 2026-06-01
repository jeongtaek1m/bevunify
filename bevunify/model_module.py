"""ModelModule with a generalized optimizer/scheduler so each model can use its OWN
training recipe (config/recipe/<model>.yaml).

Subclasses the host ModelModule and overrides ONLY configure_optimizers:
 - optimizer: ``_target_`` -> instantiate(params=...); else AdamW(**args)
 - scheduler: None       -> constant LR (no scheduler)
              ``_target_`` -> instantiate(optimizer=...)  (e.g. StepLR for LaRa)
              else        -> OneCycleLR(**args)           (host default)
   An optional ``interval`` key ('step' | 'epoch', default 'step') is honored.
"""
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from GaussianLSS.model.model_module import ModelModule


class UnifiedModelModule(ModelModule):
    def configure_optimizers(self, disable_scheduler=False):
        params = [p for p in self.backbone.parameters() if p.requires_grad]
        params += [p for p in self.loss_func.parameters() if p.requires_grad]

        opt_cfg = self.optimizer_args
        if opt_cfg is not None and "_target_" in opt_cfg:
            optimizer = instantiate(opt_cfg, params=params)
        else:
            optimizer = torch.optim.AdamW(params, **OmegaConf.to_container(opt_cfg, resolve=True))

        sch_cfg = self.scheduler_args
        if disable_scheduler or sch_cfg is None:
            return optimizer

        sch = OmegaConf.to_container(sch_cfg, resolve=True)
        interval = sch.pop("interval", "step")
        if "_target_" in sch:
            scheduler = instantiate(sch, optimizer=optimizer)
        else:
            # OneCycleLR needs total_steps; auto-fill for epoch-based runs (max_steps=-1).
            if sch.get("total_steps") in (None, -1):
                sch["total_steps"] = int(self.trainer.estimated_stepping_batches)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **sch)

        return [optimizer], [{"scheduler": scheduler, "interval": interval}]
