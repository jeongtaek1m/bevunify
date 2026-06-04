"""PointBeV wrapper (SCAFFOLD — geometry/orientation needs the probe to confirm).

PointBeV.forward(imgs, rots, trans, intrins, bev_aug, egoTin_to_seq, **kwargs) with a
temporal dim T. We feed a single frame (T=1, nq=1) and take the present-frame output.
Outputs: preds['bev']['binimg'|'centerness'|'offsets'] at (B,nq,1|2,200,200).

PointBeV builds its grid with meshgrid(indexing='ij').flip(1,2) -> row=Y points right,
whereas the GaussianLSS frame has row=Y pointing left. ``axis_fix`` applies the
alignment; the exact op MUST be confirmed with tests/probe_orientation.py.

Requires: pip install rich . Consumes seg + center + offset.
"""
import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from hydra.utils import instantiate

from .geom import add_repo_to_path, rots_trans
from .repo_compose import compose_repo_cfg


class PointBeVWrapper(nn.Module):
    def __init__(self, key, repo_root, config_name="train", axis_fix="none", backbone="b4"):
        super().__init__()
        self.key = key
        self.axis_fix = axis_fix
        # PointBeV's backbone has NO internal normalization; the original loader applies
        # ImageNet mean/std (configs/data/nuscenes.yaml normalize_img=True). The unified
        # loader feeds [0,1], so normalize here (mirrors the LSS/LaRa wrapper fix).
        self.register_buffer("_imnet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("_imnet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
        repo_root = add_repo_to_path(repo_root)
        os.environ.setdefault("PROJECT_ROOT", repo_root)   # PointBeV paths use ${oc.env:PROJECT_ROOT}

        # Register PointBeV's custom OmegaConf resolvers (get_in_c_neck, eval, ...).
        # They live in <repo>/hydra_plugins/resolvers.py and register on import.
        import importlib
        importlib.import_module("hydra_plugins.resolvers")

        cfg = compose_repo_cfg(config_dir=f"{repo_root}/configs", config_name=config_name)
        # backbone knob (PointBeV EfficientNet; native 'b4', also 'b0'). To switch
        # to a ResNet backbone, override the repo's net/backbone config group instead.
        OmegaConf.set_struct(cfg, False)
        cfg.model.net.backbone.version = backbone
        self.net = instantiate(cfg.model.net)

    def _fix_axes(self, t):
        if self.axis_fix == "flip_y":
            return torch.flip(t, dims=[-2])       # VERIFY against the probe
        if self.axis_fix == "transpose":
            return t.transpose(-1, -2)
        return t

    def forward(self, batch):
        image = batch["image"]                                   # (B,N,3,H,W) in [0,1]
        image = (image - self._imnet_mean) / self._imnet_std     # ImageNet norm (matches original loader)
        rots, trans = rots_trans(batch)                          # (B,N,3,3), (B,N,3) cam->ego
        intrins = batch["intrinsics"]                            # (B,N,3,3)
        B = image.shape[0]

        imgs = image[:, None]                                    # (B,1,N,3,H,W)
        rots = rots[:, None]                                     # (B,1,N,3,3)
        trans = trans[:, None].unsqueeze(-1)                     # (B,1,N,3,1)
        intrins = intrins[:, None]                               # (B,1,N,3,3)
        eye = torch.eye(4, device=image.device).reshape(1, 1, 4, 4).expand(B, 1, 4, 4)
        bev_aug = batch["bev_augm"][:, None] if "bev_augm" in batch else eye   # (B,1,4,4)
        egoTin_to_seq = eye                                                    # (B,1,4,4), nq=1

        preds = self.net(imgs=imgs, rots=rots, trans=trans, intrins=intrins,
                         bev_aug=bev_aug, egoTin_to_seq=egoTin_to_seq)
        bev = preds["bev"]
        masks = preds.get("masks", {}).get("bev", {})

        def take(name):
            x = bev[name][:, -1]                                 # (B,C,200,200) present frame
            return self._fix_axes(x)

        out = {self.key: take("binimg")}
        # PointBeV is SPARSE: only the coarse+fine *sampled* cells are real predictions;
        # the rest are zero-filled by .dense(). Surface that sampled mask (coarse u fine
        # union, native pointbev/models/sampled.py) so the loss supervises ONLY those
        # cells -> matches native (loss*mask).sum()/mask.sum() instead of diluting the
        # ~5k-cell gradient over the full ~40k grid. In dense val mode the mask is
        # all-ones (no-op); metrics read pred[key] only, so they are unaffected.
        if "binimg" in masks:
            out["_sampled_mask"] = self._fix_axes(masks["binimg"][:, -1]).float()
        if "centerness" in bev:
            # PointBeV's head already applies sigmoid to centerness; the shared CenterLoss
            # re-applies sigmoid -> double sigmoid (center loss froze at ~0.498). Undo the
            # head's sigmoid here so the loss sees logits (mirrors the simple_bev wrapper).
            c = take("centerness").clamp(1e-6, 1 - 1e-6)
            out[f"{self.key}_center"] = torch.logit(c)
        if "offsets" in bev:
            # audit: spatial already in GT frame (axis_fix='none'); offset needs ch0<->ch1 swap
            out[f"{self.key}_offset"] = take("offsets")[:, [1, 0]]
        return out
