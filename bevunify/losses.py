"""BCE-with-pos_weight segmentation loss — the native bev criterion for LaRa / LSS /
PointBeV / simple_bev (GaussianLSS & CVT are natively focal). Same interface and
visibility masking as GaussianLSS.losses.BinarySegmentationLoss, so it drops into
MultipleLoss as the `bev` term.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCESegmentationLoss(nn.Module):
    def __init__(self, min_visibility=0, pos_weight=2.13, key="vehicle"):
        super().__init__()
        self.min_visibility = min_visibility
        self.key = key
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))

    def forward(self, pred_dict, batch, eps=1e-6):
        pred = pred_dict[self.key]
        target = batch[self.key]
        loss = F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=self.pos_weight.to(pred.device), reduction="none")

        mask = torch.ones_like(target, dtype=torch.bool)
        if self.min_visibility > 0:
            vis_mask = batch[f"{self.key}_visibility"] >= self.min_visibility
            mask = mask * vis_mask[:, None]
        return (loss * mask).sum() / (mask.sum() + eps)
