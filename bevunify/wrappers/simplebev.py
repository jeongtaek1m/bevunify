"""simple_bev (Segnet) wrapper (SCAFFOLD — geometry/orientation needs the probe).

Segnet.forward(rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util, rad_occ_mem0=None)
returns (raw_e, feat_e, seg_e, center_e, offset_e), each BEV at (B,C,Z,X) where
Z=forward, X=lateral.

Mapping to the GaussianLSS frame (row=Y=lateral, col=X=forward):
  host(row,col) = (simple_X, simple_Z)  ->  transpose(-1,-2)   [VERIFY with probe]
center_e is post-sigmoid; the shared CenterLoss re-applies sigmoid, so we invert it
back to logits. Consumes seg + center + offset.

Requires the simple_bev repo importable (nets/, utils/).
"""
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .geom import add_repo_to_path, intrinsics_to_4x4


class SimpleBEVWrapper(nn.Module):
    def __init__(self, key, repo_root, encoder_type="res101", Z=200, Y=8, X=200,
                 bounds=(-50.0, 50.0, -5.0, 5.0, -50.0, 50.0), axis_fix="transpose"):
        super().__init__()
        self.key = key
        self.Z, self.Y, self.X = Z, Y, X
        self.bounds = tuple(OmegaConf.to_container(bounds, resolve=True)
                            if OmegaConf.is_config(bounds) else bounds)
        self.axis_fix = axis_fix
        add_repo_to_path(repo_root)
        from nets.segnet import Segnet
        # camera-only (no radar / lidar) — enforced explicitly
        self.net = Segnet(Z=Z, Y=Y, X=X, encoder_type=encoder_type,
                          use_radar=False, use_lidar=False, use_metaradar=False,
                          do_rgbcompress=True, rand_flip=False)

    def _build_vox_util(self, B, device):
        from utils.vox import Vox_util
        scene_centroid = torch.zeros(B, 3, device=device)
        return Vox_util(self.Z, self.Y, self.X, scene_centroid=scene_centroid,
                        bounds=self.bounds, assert_cube=False)

    def _fix(self, t):
        # audit: SimpleBEV BEV is (forward=row, right=col); GaussianLSS GT is row=-X(forward),
        # col=-Y(=right) -> col already matches, only the forward(row) axis is inverted.
        return t.flip(-2)

    def forward(self, batch):
        image = batch["image"]                                   # (B,N,3,H,W) in [0,1]
        B = image.shape[0]
        pix_T_cams = intrinsics_to_4x4(batch["intrinsics"])      # (B,N,4,4)

        # cam0_T_camX = E_0 @ inv(E_X), with E = lidar->cam extrinsics
        E = batch["extrinsics"]                                  # (B,N,4,4)
        E0 = E[:, 0:1]                                           # (B,1,4,4)
        cam0_T_camXs = torch.matmul(E0, torch.inverse(E))        # (B,N,4,4)

        vox_util = self._build_vox_util(B, image.device)
        # VERIFY: simple_bev expects rgb roughly centered; it does (rgb-0.5) internally.
        _, _, seg_e, center_e, offset_e = self.net(
            image, pix_T_cams, cam0_T_camXs, vox_util, rad_occ_mem0=None)

        eps = 1e-6
        # offset: flip(-2) relocates pixels, then negate ch0 (delta-col sign) to match GT; ch1 unchanged
        offset = self._fix(offset_e) * offset_e.new_tensor([-1.0, 1.0]).view(1, 2, 1, 1)
        out = {
            self.key: self._fix(seg_e),                                          # logits
            f"{self.key}_center": self._fix(torch.logit(center_e.clamp(eps, 1 - eps))),
            f"{self.key}_offset": offset,
        }
        return out
