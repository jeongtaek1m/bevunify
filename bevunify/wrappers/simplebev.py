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
        repo_root = add_repo_to_path(repo_root)
        from nets.segnet import Segnet
        # camera-only (no radar / lidar) — enforced explicitly
        self.net = Segnet(Z=Z, Y=Y, X=X, encoder_type=encoder_type,
                          use_radar=False, use_lidar=False, use_metaradar=False,
                          do_rgbcompress=True, rand_flip=False)

    def _build_vox_util(self, B, device):
        from utils.vox import Vox_util
        # original simple_bev shifts the scene centroid 1 m down (train_nuscenes.py:27-34),
        # moving the vertical sampling window to Y in [-4, 6] rather than [-5, 5].
        scene_centroid = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).expand(B, 3)
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

        # cam0_T_camX = E_0 @ inv(E_X), with E = lidar->cam extrinsics.
        # Reference cam = CAM_FRONT (index 1 in the unified order
        # [CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, ...]), matching original
        # refcam_id=1, so the BEV grid is built in the ego-forward frame. Using index 0
        # (CAM_FRONT_LEFT) rotates the whole BEV ~55deg, which a static flip can't fix.
        E = batch["extrinsics"]                                  # (B,N,4,4)
        E0 = E[:, 1:2]                                           # (B,1,4,4) = CAM_FRONT
        cam0_T_camXs = torch.matmul(E0, torch.inverse(E))        # (B,N,4,4)

        vox_util = self._build_vox_util(B, image.device)
        # Segnet does (rgb + 0.5 - imnet_mean)/imnet_std internally (nets/segnet.py:400);
        # the original centers the [0,1] loader image to [-0.5,0.5] (train_nuscenes.py:111).
        # Feed image-0.5 so the model's internal +0.5 cancels -> true ImageNet norm.
        _, _, seg_e, center_e, offset_e = self.net(
            image - 0.5, pix_T_cams, cam0_T_camXs, vox_util, rad_occ_mem0=None)

        eps = 1e-6
        # offset: flip(-2) relocates pixels, then negate ch0 (delta-col sign) to match GT; ch1 unchanged
        offset = self._fix(offset_e) * offset_e.new_tensor([-1.0, 1.0]).view(1, 2, 1, 1)
        out = {
            self.key: self._fix(seg_e),                                          # logits
            f"{self.key}_center": self._fix(torch.logit(center_e.clamp(eps, 1 - eps))),
            f"{self.key}_offset": offset,
        }
        return out
