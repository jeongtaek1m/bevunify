"""BEVFormer (segmentation) wrapper — pure-PyTorch port, no mm* model stack.

Bridges the unified DataModule batch to ``bevformer_native.BEVFormerNative``:
  * ImageNet-normalizes the [0,1] RGB images (the torchvision RN-50 backbone has no
    internal normalization, mirroring the LSS / PointBeV wrappers).
  * builds ``lidar2img = K4x4 @ E`` (ego/lidar 3D -> image pixel) from the loader's
    split intrinsics + extrinsics (needs ``data.split_intrin_extrin=True``).
  * applies the optional ``axis_fix`` to align BEVFormer's BEV frame to the
    GaussianLSS GT frame (confirm with tests/probe_orientation.py before training).

Output: ``{key: (B, 1, bev_h, bev_w)}`` logits — matches the BCE loss / IoU metric.
"""
import torch
import torch.nn as nn

from .geom import intrinsics_to_4x4
from .bevformer_native import BEVFormerNative


class BEVFormerWrapper(nn.Module):
    def __init__(self, key, num_classes=1, embed_dims=256, num_cams=6,
                 bev_h=200, bev_w=200, num_layers=6, num_points_in_pillar=4,
                 feedforward_channels=512, sca_num_points=8, tsa_num_points=4,
                 pc_range=(-50.0, -50.0, -5.0, 50.0, 50.0, 3.0),
                 pretrained_backbone=True, use_checkpoint=True, axis_fix="none",
                 out_h=None, out_w=None):
        super().__init__()
        self.key = key
        self.axis_fix = axis_fix
        # Native seg head is resolution-preserving -> net output is (bev_h, bev_w).
        # For the tiny variant (bev 50x50) the GT BEV is still 200x200, so we
        # bilinear-upsample the logits to (out_h, out_w) = the label resolution
        # (standard semantic-seg practice). Defaults to bev_h/bev_w -> no-op for base.
        self.out_h = out_h if out_h is not None else bev_h
        self.out_w = out_w if out_w is not None else bev_w
        # torchvision RN-50 expects ImageNet-normalized RGB; the unified loader
        # feeds [0,1] (configs/img_params), so normalize here.
        self.register_buffer("_imnet_mean",
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("_imnet_std",
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
        self.net = BEVFormerNative(
            num_classes=num_classes, embed_dims=embed_dims, num_cams=num_cams,
            bev_h=bev_h, bev_w=bev_w, num_layers=num_layers,
            num_points_in_pillar=num_points_in_pillar,
            feedforward_channels=feedforward_channels,
            sca_num_points=sca_num_points, tsa_num_points=tsa_num_points,
            pc_range=pc_range, pretrained_backbone=pretrained_backbone,
            use_checkpoint=use_checkpoint, out_h=self.out_h, out_w=self.out_w)

    def _fix_axes(self, t):
        if self.axis_fix == "flip_x":
            return torch.flip(t, dims=[-2])
        if self.axis_fix == "flip_y":
            return torch.flip(t, dims=[-1])
        if self.axis_fix == "flip_xy":
            return torch.flip(t, dims=[-2, -1])
        if self.axis_fix == "transpose":
            return t.transpose(-1, -2)
        return t

    def forward(self, batch):
        image = batch["image"]                              # (B,N,3,H,W) in [0,1]
        image = (image - self._imnet_mean) / self._imnet_std
        # lidar2img = K4x4 @ E ; extrinsics E: x_cam = E @ x_ego (lidar/ego -> cam)
        K = intrinsics_to_4x4(batch["intrinsics"])          # (B,N,4,4)
        E = batch["extrinsics"]                             # (B,N,4,4)
        lidar2img = torch.matmul(K, E)                      # (B,N,4,4)

        logits = self.net(image, lidar2img)                 # (B,1,out_h,out_w)
        return {self.key: self._fix_axes(logits)}
