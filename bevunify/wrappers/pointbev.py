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
from hydra.utils import instantiate

from .geom import add_repo_to_path, rots_trans
from .repo_compose import compose_repo_cfg


class PointBeVWrapper(nn.Module):
    def __init__(self, key, repo_root, config_name="train", axis_fix="none"):
        super().__init__()
        self.key = key
        self.axis_fix = axis_fix
        add_repo_to_path(repo_root)
        os.environ.setdefault("PROJECT_ROOT", repo_root)   # PointBeV paths use ${oc.env:PROJECT_ROOT}

        # Register PointBeV's custom OmegaConf resolvers (get_in_c_neck, eval, ...).
        # They live in <repo>/hydra_plugins/resolvers.py and register on import.
        import importlib
        importlib.import_module("hydra_plugins.resolvers")

        cfg = compose_repo_cfg(config_dir=f"{repo_root}/configs", config_name=config_name)
        self.net = instantiate(cfg.model.net)

    def _fix_axes(self, t):
        if self.axis_fix == "flip_y":
            return torch.flip(t, dims=[-2])       # VERIFY against the probe
        if self.axis_fix == "transpose":
            return t.transpose(-1, -2)
        return t

    def forward(self, batch):
        image = batch["image"]                                   # (B,N,3,H,W)
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

        def take(name):
            x = bev[name][:, -1]                                 # (B,C,200,200) present frame
            return self._fix_axes(x)

        out = {self.key: take("binimg")}
        if "centerness" in bev:
            out[f"{self.key}_center"] = take("centerness")
        if "offsets" in bev:
            # audit: spatial already in GT frame (axis_fix='none'); offset needs ch0<->ch1 swap
            out[f"{self.key}_offset"] = take("offsets")[:, [1, 0]]
        return out
