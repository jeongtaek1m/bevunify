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
        # Optional image-backbone LR multiplier (BEVFormer-native paramwise_cfg:
        # img_backbone lr_mult). Opt-in via two recipe keys; absent -> single
        # param group, behavior identical to before. AdamW path only.
        opt_dict = (OmegaConf.to_container(opt_cfg, resolve=True)
                    if opt_cfg is not None else {})
        lr_mult = opt_dict.pop("backbone_lr_mult", None)
        prefixes = opt_dict.pop("backbone_prefixes", None)
        self._group_lr_mults = [1.0]
        if lr_mult is not None and prefixes and "_target_" not in opt_dict:
            bb, rest = [], []
            for n, p in self.backbone.named_parameters():
                if p.requires_grad:
                    (bb if any(n.startswith(pf) for pf in prefixes) else rest).append(p)
            rest += [p for p in self.loss_func.parameters() if p.requires_grad]
            params = [{"params": rest},
                      {"params": bb, "lr": opt_dict["lr"] * lr_mult}]
            self._group_lr_mults = [1.0, lr_mult]

        if opt_cfg is not None and "_target_" in opt_dict:
            optimizer = instantiate(opt_cfg, params=params)
        else:
            optimizer = torch.optim.AdamW(params, **opt_dict)

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
            # OneCycle drives each group's lr from max_lr — a scalar would erase the
            # backbone lr_mult, so expand to a per-group list when groups exist.
            if (len(self._group_lr_mults) > 1
                    and not isinstance(sch.get("max_lr"), (list, tuple))):
                sch["max_lr"] = [sch["max_lr"] * m for m in self._group_lr_mults]
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **sch)

        return [optimizer], [{"scheduler": scheduler, "interval": interval}]
