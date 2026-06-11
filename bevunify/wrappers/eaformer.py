"""EAFormer wrapper (Epipolar Attention Field Transformer, WACV 2025).

A fork of the vendored CVT model (third_party/eaformer) that replaces CVT's
learnable camera-aware positional encoding with Epipolar Attention Fields and an
ASPP head. The wiring mirrors ``cvt.py``: compose the fork's own Hydra config in
isolation, set the canonical output keys, and instantiate ``cfg.model``.

Like CVT, this needs the unified DataModule with ``split_intrin_extrin=True`` so
the batch carries ``intrinsics`` / ``extrinsics`` / ``cam_idx``. Extrinsic
convention ``x_cam = E @ x_ego`` matches the loader, and the output is already in
the GaussianLSS BEV frame (row=-X, col=-Y) — no axis flip (shares CVT's bev.grid).

Optional ``eaf_*`` / ``use_*`` kwargs (default ``None`` -> keep the fork-config
value) let ablations be driven from the bevunify config or CLI, e.g.
``model.eaf_mode=additive`` or ``model.use_eaf=false``.
"""
import torch.nn as nn
from omegaconf import OmegaConf
from hydra.utils import instantiate

from .geom import add_repo_to_path
from .repo_compose import compose_repo_cfg


class EAFormerWrapper(nn.Module):
    def __init__(self, key, repo_root, config_name="config", center=True,
                 experiment="eaformer_nuscenes_vehicle", backbone="efficientnet-b4",
                 image_h=None, image_w=None,
                 use_eaf=None, use_pe=None, eaf_mode=None, eaf_lambda=None,
                 eaf_learnable_lambda=None, eaf_bev_range_m=None, eaf_d0=None,
                 eaf_clamp_qi=None):
        super().__init__()
        self.key = key
        repo_root = add_repo_to_path(repo_root)

        cfg = compose_repo_cfg(
            config_dir=f"{repo_root}/config",
            config_name=config_name,
            overrides=[f"+experiment={experiment}"],
        )
        # Canonical output keys (${key} can't interpolate into a YAML dict KEY).
        outputs = {key: [0, 1]}
        if center:
            outputs[f"{key}_center"] = [1, 2]
        OmegaConf.set_struct(cfg, False)
        # Sync image resolution + backbone with the unified loader (see cvt.py).
        if image_h is not None and OmegaConf.select(cfg, "data.image") is not None:
            cfg.data.image.h = image_h
            cfg.data.image.w = image_w
        cfg.model.encoder.backbone.model_name = backbone
        cfg.model.outputs = OmegaConf.create(outputs)

        # Optional EAF ablation overrides (None -> keep fork-config default).
        cv = cfg.model.encoder.cross_view
        for name, val in dict(use_eaf=use_eaf, use_pe=use_pe, eaf_mode=eaf_mode,
                              eaf_lambda=eaf_lambda, eaf_learnable_lambda=eaf_learnable_lambda,
                              eaf_bev_range_m=eaf_bev_range_m, eaf_d0=eaf_d0,
                              eaf_clamp_qi=eaf_clamp_qi).items():
            if val is not None:
                cv[name] = val

        self.net = instantiate(cfg.model)

    def forward(self, batch):
        # Output keys == canonical keys -> matches the GaussianLSS loss/metric contract.
        return self.net(batch)
