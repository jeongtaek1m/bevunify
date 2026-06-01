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
    def __init__(self, key, repo_root, grid_conf, data_aug_conf, outC=1,
                 backbone="efficientnet-b0"):
        super().__init__()
        self.key = key
        # Original LSS normalizes images with ImageNet mean/std at the dataloader
        # (third_party/lift-splat-shoot/src/tools.py:167-169); LSS.forward applies none.
        # The unified loader feeds [0,1], so normalize here to feed the pretrained
        # EfficientNet-b0 the distribution it was trained on.
        self.register_buffer("_imnet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("_imnet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
        # LSS's CamEncode hardcodes the trunk AND its channel dims for efficientnet-b0
        # (third_party/lift-splat-shoot/src/models.py:43); other versions don't fit the
        # downstream convs without repo changes. Expose the knob, but pin to native b0.
        if backbone != "efficientnet-b0":
            raise ValueError(
                f"LSS backbone is pinned to 'efficientnet-b0' (repo CamEncode is wired "
                f"to its channel dims); got '{backbone}'. Swapping needs LSS source changes.")
        repo_root = add_repo_to_path(repo_root)
        from src.models import LiftSplatShoot  # import the class directly (avoids src.train deps)

        grid_conf = OmegaConf.to_container(grid_conf, resolve=True)
        data_aug_conf = OmegaConf.to_container(data_aug_conf, resolve=True)
        self.net = LiftSplatShoot(grid_conf, data_aug_conf, outC=outC)

    def forward(self, batch):
        x = batch["image"]                       # (B,N,3,H,W) in [0,1]
        x = (x - self._imnet_mean) / self._imnet_std   # ImageNet norm (matches original loader)
        intrins = batch["intrinsics"]            # (B,N,3,3)
        rots, trans = rots_trans(batch)                   # (B,N,3,3), (B,N,3) cam->ego
        B, N = x.shape[:2]
        post_rots = torch.eye(3, dtype=x.dtype, device=x.device).expand(B, N, 3, 3)
        post_trans = torch.zeros(B, N, 3, dtype=x.dtype, device=x.device)

        out = self.net(x, rots, trans, intrins, post_rots, post_trans)  # (B,1,200,200) row=+X col=+Y
        out = out.flip(-2).flip(-1)                                     # -> row=-X col=-Y (GaussianLSS GT frame)
        return {self.key: out}
