"""BCE-with-pos_weight segmentation loss — the native bev criterion for LaRa / LSS /
PointBeV / simple_bev (GaussianLSS & CVT are natively focal). Same interface and
visibility masking as GaussianLSS.losses.BinarySegmentationLoss, so it drops into
MultipleLoss as the `bev` term.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss

from GaussianLSS.losses import CenterLoss as _HostCenterLoss, OffsetLoss as _HostOffsetLoss


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
        # PointBeV (sparse): restrict the loss to the cells the model actually predicted
        # (coarse u fine sampled mask surfaced by the wrapper). Absent for dense models -> no-op.
        sampled = pred_dict.get("_sampled_mask")
        if sampled is not None:
            mask = mask * (sampled > 0)
        return (loss * mask).sum() / (mask.sum() + eps)


class FocalCenterLoss(nn.Module):
    """CVT-native center loss: SigmoidFocalLoss on RAW center logits vs the center
    heatmap (matches cross_view_transformer.losses.CenterLoss: gamma=2, alpha=-1,
    reduction='none', then masked-mean at visibility>=min_visibility). Replaces the
    host CenterLoss (MSE-on-sigmoid) so CVT's center branch is supervised faithfully.
    """
    def __init__(self, min_visibility=0, alpha=-1.0, gamma=2.0, key="vehicle"):
        super().__init__()
        self.min_visibility = min_visibility
        self.alpha = alpha
        self.gamma = gamma
        self.key = key

    def forward(self, pred_dict, batch, eps=1e-6):
        k = f"{self.key}_center"
        pred = pred_dict[k]                       # raw logits (no sigmoid)
        target = batch[k]                         # center heatmap in [0,1]
        loss = sigmoid_focal_loss(pred, target, self.alpha, self.gamma, reduction="none")
        mask = torch.ones_like(target, dtype=torch.bool)
        if self.min_visibility > 0:
            vis_mask = batch[f"{self.key}_visibility"] >= self.min_visibility
            mask = mask * vis_mask[:, None]
        return (loss * mask).sum() / (mask.sum() + eps)


class BalancedMSECenterLoss(nn.Module):
    """simple_bev-native center loss: balanced MSE on the (sigmoid) center vs the
    heatmap — positive (target>0.5) and negative (target<0.5) MSE are averaged
    separately then *0.5 (matches simple_bev train_nuscenes.balanced_mse_loss,
    valid=None => full map). The simple_bev wrapper feeds center LOGITS, so sigmoid
    here recovers the model's center probability that the original loss operates on.
    """
    def __init__(self, key="vehicle"):
        super().__init__()
        self.key = key

    def forward(self, pred_dict, batch, eps=1e-6):
        k = f"{self.key}_center"
        pred = pred_dict[k].sigmoid()             # wrapper re-logited; sigmoid -> center prob
        target = batch[k]
        mse = F.mse_loss(pred, target, reduction="none")
        pos = (target > 0.5).float()
        neg = (target < 0.5).float()
        pos_loss = (mse * pos).sum() / (pos.sum() + eps)
        neg_loss = (mse * neg).sum() / (neg.sum() + eps)
        return (pos_loss + neg_loss) * 0.5


class FootprintOffsetLoss(nn.Module):
    """simple_bev-native offset loss: L1 (summed over the 2 offset channels) masked by
    the vehicle SEG FOOTPRINT (matches simple_bev: reduce_masked_mean over seg_g*valid)
    — supervised ONLY on vehicle pixels, not on background. Footprint is intersected
    with visibility>=min_visibility to honor the unified vis>=2 ignore policy. This
    replaces the host OffsetLoss, which masked by visibility>=2 over the whole BEV
    (i.e. also supervised offset=0 on background).
    """
    def __init__(self, min_visibility=0, key="vehicle"):
        super().__init__()
        self.min_visibility = min_visibility
        self.key = key

    def forward(self, pred_dict, batch, eps=1e-6):
        pred = pred_dict[f"{self.key}_offset"]
        target = batch[f"{self.key}_offset"]
        l1 = (pred - target).abs().sum(dim=1, keepdim=True)       # (B,1,H,W)
        footprint = batch[self.key] > 0                           # vehicle pixels (B,1,H,W)
        if self.min_visibility > 0:
            vis_mask = batch[f"{self.key}_visibility"] >= self.min_visibility
            footprint = footprint & vis_mask[:, None]
        footprint = footprint.float()
        return (l1 * footprint).sum() / (footprint.sum() + eps)


class _SampledSpatialMixin:
    """Mixin re-implementing the host SpatialRegressionLoss masked-mean (visibility +
    ignore_index) and ADDITIONALLY restricting to the coarse u fine sampled cells PointBeV
    actually predicted -- surfaced by the wrapper as pred['_sampled_mask']. When that key
    is absent the behavior is byte-identical to the host loss, so the host GaussianLSS
    repo stays untouched; only PointBeV's loss config (seg_center_offset_bce) points here.
    ``self.loss_fn`` / ``self.min_visibility`` / ``self.ignore_index`` / ``self.key`` are
    set by the host __init__ this mixin is composed in front of.
    """
    _suffix = ""
    _sigmoid = False

    def forward(self, prediction, batch, eps=1e-6):
        key = f"{self.key}{self._suffix}"
        pred = prediction[key]
        if self._sigmoid:
            pred = pred.sigmoid()
        target = batch[key]
        assert pred.dim() == 4, "Must be a 4D tensor"
        visibility = batch[f"{self.key}_visibility"]

        mask = torch.ones_like(target, dtype=torch.bool)
        if self.min_visibility > 0:
            mask = mask * (visibility >= self.min_visibility)[:, None]
        if self.ignore_index is not None:
            mask = mask * (target != self.ignore_index)
        sampled = prediction.get("_sampled_mask")
        if sampled is not None:
            mask = mask * (sampled > 0)

        loss = self.loss_fn(pred, target, reduction="none")
        return (loss * mask).sum() / (mask.sum() + eps)


class SampledCenterLoss(_SampledSpatialMixin, _HostCenterLoss):
    """Host CenterLoss (MSE on sigmoid center) restricted to PointBeV's sampled cells."""
    _suffix = "_center"
    _sigmoid = True


class SampledOffsetLoss(_SampledSpatialMixin, _HostOffsetLoss):
    """Host OffsetLoss (L1 offset) restricted to PointBeV's sampled cells."""
    _suffix = "_offset"
    _sigmoid = False
