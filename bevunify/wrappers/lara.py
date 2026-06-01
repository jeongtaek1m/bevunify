"""LaRa wrapper.

The LaRa repo config exposes the raw net at cfg.model.net (the LightningModule
BEVBinaryOccupancy wraps it). We instantiate just the net and adapt I/O.

LaRa.forward(imgs, rots, trans, intrins) -> (B,1,200,200) logits. rots/trans follow
the LSS cam->ego convention. Consumes binary seg only.

Requires: pip install fairscale ; env WEIGHTS_PATH (EfficientNet cam-encoder weights).
"""
import os
import torch.nn as nn
from hydra.utils import instantiate

from .geom import add_repo_to_path, rots_trans
from .repo_compose import compose_repo_cfg


class LaRaWrapper(nn.Module):
    def __init__(self, key, repo_root, config_name="train",
                 experiment="LaRa_inCamrays_outCoord"):
        super().__init__()
        self.key = key
        add_repo_to_path(repo_root)
        os.environ.setdefault("WEIGHTS_PATH", "")  # CamEncode reads ${oc.env:WEIGHTS_PATH}

        # the experiment fills mandatory input_embeddings / query_generators.
        cfg = compose_repo_cfg(config_dir=f"{repo_root}/configs", config_name=config_name,
                               overrides=[f"experiment={experiment}"])
        # VERIFY: grid_conf/data_aug_conf in LaRa default to 200x200 / +-50m (matches host).
        self.net = instantiate(cfg.model.net)

    def forward(self, batch):
        imgs = batch["image"]                    # (B,N,3,H,W)
        intrins = batch["intrinsics"]            # (B,N,3,3)
        rots, trans = rots_trans(batch)                   # (B,N,3,3), (B,N,3) cam->ego
        out = self.net(imgs, rots, trans, intrins)        # (B,1,200,200) row=+X col=+Y
        out = out.flip(-2).flip(-1)                        # -> GaussianLSS GT frame (row=-X col=-Y); same as LSS
        return {self.key: out}
