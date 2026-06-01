"""lift-splat-shoot wrapper (plain nn.Module, no hydra in the source repo).

LSS.forward(x, rots, trans, intrins, post_rots, post_trans) -> (B,1,200,200) logits.
We build rots/trans by inverting the GaussianLSS extrinsics (lidar->cam) to cam->ego,
and set post_rots/post_trans to identity because image resize/crop is already baked
into the unified loader's images + intrinsics.

LSS output BEV is (B,C,X,Y): row=+X(forward), col=+Y(left). GaussianLSS GT (V matrix
[[0,-2,100],[-2,0,100],...]) is row=-X, col=-Y. Same axis assignment but BOTH directions
inverted => align with a 180deg rotation: out.flip(-2).flip(-1)  (NOT a transpose).
Extrinsic (cam->egolidarflat via ego_from_cam) is correct as-is. Consumes binary seg only.
"""
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .geom import add_repo_to_path, rots_trans


class LSSWrapper(nn.Module):
    def __init__(self, key, repo_root, grid_conf, data_aug_conf, outC=1):
        super().__init__()
        self.key = key
        add_repo_to_path(repo_root)
        from src.models import LiftSplatShoot  # import the class directly (avoids src.train deps)

        grid_conf = OmegaConf.to_container(grid_conf, resolve=True)
        data_aug_conf = OmegaConf.to_container(data_aug_conf, resolve=True)
        self.net = LiftSplatShoot(grid_conf, data_aug_conf, outC=outC)

    def forward(self, batch):
        x = batch["image"]                       # (B,N,3,H,W)
        intrins = batch["intrinsics"]            # (B,N,3,3)
        rots, trans = rots_trans(batch)                   # (B,N,3,3), (B,N,3) cam->ego
        B, N = x.shape[:2]
        post_rots = torch.eye(3, dtype=x.dtype, device=x.device).expand(B, N, 3, 3)
        post_trans = torch.zeros(B, N, 3, dtype=x.dtype, device=x.device)

        out = self.net(x, rots, trans, intrins, post_rots, post_trans)  # (B,1,200,200) row=+X col=+Y
        out = out.flip(-2).flip(-1)                                     # -> row=-X col=-Y (GaussianLSS GT frame)
        return {self.key: out}
