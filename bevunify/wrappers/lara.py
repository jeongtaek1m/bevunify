"""LaRa wrapper.

The LaRa repo config exposes the raw net at cfg.model.net (the LightningModule
BEVBinaryOccupancy wraps it). We instantiate just the net and adapt I/O.

LaRa.forward(imgs, rots, trans, intrins) -> (B,1,200,200) logits. rots/trans follow
the LSS cam->ego convention. Consumes binary seg only.

Requires: pip install fairscale ; env WEIGHTS_PATH (EfficientNet cam-encoder weights).
"""
import os
import torch
import torch.nn as nn
from hydra.utils import instantiate

from .geom import add_repo_to_path, rots_trans
from .repo_compose import compose_repo_cfg


class LaRaWrapper(nn.Module):
    def __init__(self, key, repo_root, config_name="train",
                 experiment="LaRa_inCamrays_outCoord", imagenet_norm=True):
        super().__init__()
        self.key = key
        self.imagenet_norm = imagenet_norm
        # LaRa's pretrained EfficientNet (CamEncode) was trained with ImageNet-normalized
        # images (normalize_img); the unified loader feeds [0,1]. Normalize here (LaRa-only).
        self.register_buffer("_imnet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("_imnet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
        repo_root = add_repo_to_path(repo_root)
        os.environ.setdefault("WEIGHTS_PATH", "")  # CamEncode reads ${oc.env:WEIGHTS_PATH}

        # the experiment fills mandatory input_embeddings / query_generators.
        cfg = compose_repo_cfg(config_dir=f"{repo_root}/configs", config_name=config_name,
                               overrides=[f"experiment={experiment}"])
        # VERIFY: grid_conf/data_aug_conf in LaRa default to 200x200 / +-50m (matches host).
        self.net = instantiate(cfg.model.net)

    def forward(self, batch):
        imgs = batch["image"]                    # (B,N,3,H,W) in [0,1]
        if self.imagenet_norm:
            imgs = (imgs - self._imnet_mean) / self._imnet_std
        intrins = batch["intrinsics"]            # (B,N,3,3)
        rots, trans = rots_trans(batch)                   # (B,N,3,3), (B,N,3) cam->ego
        out = self.net(imgs, rots, trans, intrins)        # (B,1,200,200) row=+X col=+Y
        out = out.flip(-2).flip(-1)                        # -> GaussianLSS GT frame (row=-X col=-Y); same as LSS
        return {self.key: out}
