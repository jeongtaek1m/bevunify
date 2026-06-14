"""Pure-PyTorch port of BEVFormer (segmentation) for bevunify.

Faithful re-implementation of the BEVFormer encoder used by
``BEVFormer_segmentation_detection`` with ZERO mm* dependency: the Multi-Scale
Deformable Attention is the vendored pure-PyTorch reference implementation (see
``multi_scale_deformable_attn_pytorch`` below), and everything else — ResNet-50
backbone, a single-level FPN, the BEV query / positional embeddings,
Temporal-Self-Attention (TSA), Spatial-Cross-Attention (SCA), the encoder stack
and the ResNet18-style seg upsampler — is plain ``nn.Module`` code.

Configured for the bevunify benchmark (vs. the stock ``bevformer_small_seg`` config):
  * BEV grid 200x200 over +-50 m (0.5 m / cell) -> matches the GaussianLSS GT frame.
  * ResNet-50 (ImageNet), single-scale C5 feature, single-level FPN.
  * Single-frame: no temporal queue, no CAN-bus, no ego-motion shift
    (TSA degenerates to spatial self-attention over the current BEV, exactly as the
    original first-frame path).
  * Vehicle-binary seg head (SegEncode outC=1, BCE loss owned by the host).

Reference source (ported from, not imported):
  /home/hanyan_arch/git/BEV_seg/BEVFormer_segmentation_detection/
    projects/mmdet3d_plugin/bevformer/modules/{encoder,transformer,
    spatial_cross_attention,temporal_self_attention,seg_subnet}.py
    + dense_heads/bevformer_seg_head.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet50, resnet18

# NOTE: this module is 100% pure PyTorch — NO mm* dependency. The multi-scale
# deformable attention is the reference grid_sample implementation (vendored below
# from mmcv/Deformable-DETR, Apache-2.0), numerically equivalent to the CUDA op.
# This lets BEVFormer train in the same env as the other bevunify models.


# --------------------------------------------------------------------------- #
# init helpers (replicate mmcv.cnn.xavier_init / constant_init exactly)
# --------------------------------------------------------------------------- #
def xavier_init(module, gain=1.0, bias=0.0, distribution="uniform"):
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0.0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                        sampling_locations, attention_weights):
    """Reference (pure-PyTorch) multi-scale deformable attention. Vendored verbatim
    from mmcv.ops.multi_scale_deform_attn (Apache-2.0) so the module carries no mm*
    import. Numerically equivalent to the CUDA op (differs only in fp summation order).

    value             (bs, num_value, num_heads, head_dim)
    spatial_shapes    (num_levels, 2)  rows = (H, W)
    sampling_locations(bs, num_query, num_heads, num_levels, num_points, 2)
    attention_weights (bs, num_query, num_heads, num_levels, num_points)
    """
    bs, _, num_heads, head_dim = value.shape
    _, num_query, _, num_levels, num_points, _ = sampling_locations.shape
    split_sizes = [int(H) * int(W) for H, W in value_spatial_shapes]
    value_list = value.split(split_sizes, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H, W) in enumerate(value_spatial_shapes):
        H, W = int(H), int(W)
        # (bs, H*W, heads, head_dim) -> (bs*heads, head_dim, H, W)
        value_l = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, head_dim, H, W)
        # (bs, num_query, heads, num_points, 2) -> (bs*heads, num_query, num_points, 2)
        sampling_grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l = F.grid_sample(
            value_l, sampling_grid_l, mode="bilinear",
            padding_mode="zeros", align_corners=False)
        sampling_value_list.append(sampling_value_l)
    # (bs, num_query, heads, levels, points) -> (bs*heads, 1, num_query, levels*points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_query, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2)
              * attention_weights).sum(-1).view(
        bs, num_heads * head_dim, num_query)
    return output.transpose(1, 2).contiguous()


def _msda(value, spatial_shapes, level_start_index, sampling_locations,
          attention_weights, im2col_step=64):
    """Pure-PyTorch deformable attention (level_start_index / im2col_step are part of
    the CUDA-op signature and unused here, kept so call sites stay identical)."""
    return multi_scale_deformable_attn_pytorch(
        value, spatial_shapes, sampling_locations, attention_weights)


# --------------------------------------------------------------------------- #
# BEV learned positional encoding (mmdet LearnedPositionalEncoding)
# --------------------------------------------------------------------------- #
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, num_feats, row_num_embed, col_num_embed):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, bs, h, w, device):
        x = torch.arange(w, device=device)
        y = torch.arange(h, device=device)
        x_embed = self.col_embed(x)                       # (w, num_feats)
        y_embed = self.row_embed(y)                       # (h, num_feats)
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1),
             y_embed.unsqueeze(1).repeat(1, w, 1)), dim=-1
        ).permute(2, 0, 1).unsqueeze(0).repeat(bs, 1, 1, 1)
        return pos                                        # (bs, 2*num_feats, h, w)


# --------------------------------------------------------------------------- #
# FFN (mmcv FFN: Linear-ReLU-Drop-Linear-Drop with residual)
# --------------------------------------------------------------------------- #
class FFN(nn.Module):
    def __init__(self, embed_dims, feedforward_channels, ffn_drop=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(ffn_drop),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(ffn_drop),
        )
        # match upstream PerceptionTransformer.init_weights: xavier-uniform on all
        # dim>1 transformer params (FFN weights included; biases keep the default).
        for mod in self.layers:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)

    def forward(self, x, identity=None):
        out = self.layers(x)
        identity = x if identity is None else identity
        return identity + out


# --------------------------------------------------------------------------- #
# Spatial-cross-attention's inner deformable attention (3D refs -> image samples)
# --------------------------------------------------------------------------- #
class MSDeformableAttention3D(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, num_levels=1, num_points=8,
                 im2col_step=64):
        super().__init__()
        assert embed_dims % num_heads == 0
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)

    def forward(self, query, value, reference_points, spatial_shapes,
                level_start_index):
        # query (bs, num_query, c); value (bs, num_value, c) — batch_first.
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)

        # reference_points last dim == 2 (num_Z_anchors reference points / query)
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        bs, num_query, num_Z_anchors, xy = reference_points.shape
        reference_points = reference_points[:, :, None, None, None, :, :]
        sampling_offsets = sampling_offsets / \
            offset_normalizer[None, None, None, :, None, :]
        bs, num_query, num_heads, num_levels, num_all_points, xy = \
            sampling_offsets.shape
        sampling_offsets = sampling_offsets.view(
            bs, num_query, num_heads, num_levels,
            num_all_points // num_Z_anchors, num_Z_anchors, xy)
        sampling_locations = reference_points + sampling_offsets
        bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = \
            sampling_locations.shape
        assert num_all_points == num_points * num_Z_anchors
        sampling_locations = sampling_locations.view(
            bs, num_query, num_heads, num_levels, num_all_points, xy)

        return _msda(value, spatial_shapes, level_start_index,
                     sampling_locations, attention_weights, self.im2col_step)


# --------------------------------------------------------------------------- #
# Spatial Cross Attention (3D BEV pillars -> per-camera image features)
# --------------------------------------------------------------------------- #
class SpatialCrossAttention(nn.Module):
    def __init__(self, embed_dims=256, num_cams=6, num_levels=1, num_points=8,
                 dropout=0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformableAttention3D(
            embed_dims=embed_dims, num_levels=num_levels, num_points=num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(self, query, key, value, reference_points_cam, bev_mask,
                spatial_shapes, level_start_index, indexes, max_len,
                residual=None):
        # query (bs, num_query, c); key/value (num_cam, l, bs, c)
        # indexes[j][i]: visible-query indices of sample j in camera i, computed
        # once per forward in BEVFormerEncoder.forward (upstream recomputed them
        # per layer from sample 0 only — bs==1-only and wasteful).
        inp_residual = query if residual is None else residual
        slots = torch.zeros_like(query)
        bs, num_query, _ = query.size()
        D = reference_points_cam.size(3)

        # rebatch: each camera only attends to the BEV queries that hit it.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[j][i]
                queries_rebatch[j, i, :len(index_query_per_img)] = \
                    query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = \
                    reference_points_per_img[j, index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape
        key = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)

        queries = self.deformable_attention(
            query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
            value=value,
            reference_points=reference_points_rebatch.view(
                bs * self.num_cams, max_len, D, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        ).view(bs, self.num_cams, max_len, self.embed_dims)

        for j in range(bs):
            for i in range(self.num_cams):
                index_query_per_img = indexes[j][i]
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)
        return self.dropout(slots) + inp_residual


# --------------------------------------------------------------------------- #
# Temporal Self Attention (single-frame -> spatial self-attention over BEV)
# --------------------------------------------------------------------------- #
class TemporalSelfAttention(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, num_levels=1, num_points=4,
                 num_bev_queue=2, im2col_step=64, dropout=0.1):
        super().__init__()
        assert embed_dims % num_heads == 0
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.dropout = nn.Dropout(dropout)
        self.sampling_offsets = nn.Linear(
            embed_dims * num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims * num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2).repeat(
            1, self.num_levels * self.num_bev_queue, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(self, query, query_pos, reference_points, spatial_shapes,
                level_start_index, value=None, identity=None):
        # query (bs, len, c); reference_points (bs*num_bev_queue, len, num_levels, 2)
        if value is None:
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs * 2, len_bev, c)
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_bev_queue == 2

        # value rows are sample-interleaved [s0,s0,s1,s1,...] (stack dim=1 above).
        # Upstream's value[:bs] is bs==1-only — at bs>1 it pairs row b with sample
        # b//2's BEV. Take each sample's own history copy (queue slot 0) instead.
        query = torch.cat([value[0::2], query], -1)
        value = self.value_proj(value)
        value = value.reshape(bs * self.num_bev_queue, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue,
            self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue,
            self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_bev_queue,
            self.num_levels, self.num_points)

        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5).reshape(
            bs * self.num_bev_queue, num_query, self.num_heads,
            self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape(
            bs * self.num_bev_queue, num_query, self.num_heads,
            self.num_levels, self.num_points, 2)

        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] \
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        output = _msda(value, spatial_shapes, level_start_index,
                       sampling_locations, attention_weights, self.im2col_step)

        # fuse the (history, current) BEV queue by mean.
        output = output.permute(1, 2, 0).view(num_query, embed_dims, bs,
                                              self.num_bev_queue).mean(-1)
        output = output.permute(2, 0, 1)
        output = self.output_proj(output)
        return self.dropout(output) + identity


# --------------------------------------------------------------------------- #
# One encoder layer: TSA -> norm -> SCA -> norm -> FFN -> norm
# --------------------------------------------------------------------------- #
class BEVFormerLayer(nn.Module):
    def __init__(self, embed_dims=256, num_cams=6, num_levels=1,
                 feedforward_channels=512, ffn_drop=0.1,
                 sca_num_points=8, tsa_num_points=4):
        super().__init__()
        self.tsa = TemporalSelfAttention(
            embed_dims=embed_dims, num_levels=num_levels, num_points=tsa_num_points)
        self.sca = SpatialCrossAttention(
            embed_dims=embed_dims, num_cams=num_cams, num_levels=num_levels,
            num_points=sca_num_points)
        self.ffn = FFN(embed_dims, feedforward_channels, ffn_drop=ffn_drop)
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(3)])

    def forward(self, query, key, value, bev_pos, ref_2d, ref_3d, bev_h, bev_w,
                reference_points_cam, bev_mask, spatial_shapes, level_start_index,
                sca_indexes, sca_max_len):
        tsa_shapes = torch.tensor([[bev_h, bev_w]], device=query.device)
        tsa_lsi = torch.tensor([0], device=query.device)

        query = self.tsa(query, query_pos=bev_pos, reference_points=ref_2d,
                         spatial_shapes=tsa_shapes, level_start_index=tsa_lsi)
        query = self.norms[0](query)
        query = self.sca(query, key, value,
                         reference_points_cam=reference_points_cam, bev_mask=bev_mask,
                         spatial_shapes=spatial_shapes,
                         level_start_index=level_start_index,
                         indexes=sca_indexes, max_len=sca_max_len)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)
        return query


# --------------------------------------------------------------------------- #
# Encoder stack: build reference points, project to cameras, run N layers.
# --------------------------------------------------------------------------- #
class BEVFormerEncoder(nn.Module):
    def __init__(self, num_layers=6, embed_dims=256, num_cams=6, num_levels=1,
                 pc_range=None, num_points_in_pillar=4, feedforward_channels=512,
                 ffn_drop=0.1, sca_num_points=8, tsa_num_points=4,
                 use_checkpoint=True):
        super().__init__()
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([
            BEVFormerLayer(embed_dims=embed_dims, num_cams=num_cams,
                           num_levels=num_levels,
                           feedforward_channels=feedforward_channels,
                           ffn_drop=ffn_drop, sca_num_points=sca_num_points,
                           tsa_num_points=tsa_num_points)
            for _ in range(num_layers)
        ])

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim="3d",
                             bs=1, device="cuda", dtype=torch.float):
        if dim == "3d":
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(
                num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device).view(
                1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device).view(
                1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d
        # dim == '2d'
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device))
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
        return ref_2d

    def point_sampling(self, reference_points, pc_range, lidar2img, img_h, img_w):
        # reference_points (bs, D, num_query, 3) in [0,1]; lidar2img (bs, num_cam, 4, 4)
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]
        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.permute(1, 0, 2, 3)          # (D,B,nq,4)

        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(
            lidar2img.to(torch.float32),
            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5
        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps)
        reference_points_cam[..., 0] /= img_w
        reference_points_cam[..., 1] /= img_h
        bev_mask = (bev_mask
                    & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        bev_mask = torch.nan_to_num(bev_mask)
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
        return reference_points_cam, bev_mask

    def forward(self, bev_query, key, value, bev_h, bev_w, bev_pos,
                spatial_shapes, level_start_index, lidar2img, img_h, img_w):
        # bev_query (num_query, bs, c); bev_pos (num_query, bs, c)
        bs = bev_query.size(1)
        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar, dim="3d", bs=bs,
            device=bev_query.device, dtype=bev_query.dtype)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim="2d", bs=bs,
            device=bev_query.device, dtype=bev_query.dtype)
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, lidar2img, img_h, img_w)

        # Per-sample, per-camera visible-query indices for SCA's rebatch.
        # Upstream derived them from sample 0 alone (bs==1-only) and recomputed
        # them in every layer; they depend only on bev_mask, so compute once here
        # (outside the gradient checkpoint — not recomputed in backward).
        num_cam = bev_mask.size(0)
        sca_indexes = [[bev_mask[i][j].sum(-1).nonzero().squeeze(-1)
                        for i in range(num_cam)] for j in range(bs)]
        # max(…, 1) keeps shapes valid if no query hits any camera
        sca_max_len = max(1, max(idx.numel()
                                 for per_sample in sca_indexes for idx in per_sample))

        bev_query = bev_query.permute(1, 0, 2)            # (bs, num_query, c)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        # single-frame: history == current; duplicate the 2d refs for the queue.
        hybrid_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
            bs * 2, len_bev, num_bev_level, 2)

        output = bev_query
        for layer in self.layers:
            args = (output, key, value, bev_pos, hybrid_ref_2d, ref_3d, bev_h,
                    bev_w, reference_points_cam, bev_mask, spatial_shapes,
                    level_start_index, sca_indexes, sca_max_len)
            # gradient checkpointing (BEVFormer's with_cp) trades compute for memory.
            if self.use_checkpoint and self.training:
                output = checkpoint(layer, *args, use_reentrant=False)
            else:
                output = layer(*args)
        return output                                     # (bs, num_query, c)


# --------------------------------------------------------------------------- #
# ResNet-50 backbone + single-level FPN (C5 -> 256ch)
# --------------------------------------------------------------------------- #
class ResNet50FPN(nn.Module):
    def __init__(self, embed_dims=256, pretrained=True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        trunk = resnet50(weights=weights)
        self.stem = nn.Sequential(trunk.conv1, trunk.bn1, trunk.relu, trunk.maxpool)
        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3
        self.layer4 = trunk.layer4
        # single-level FPN (in_channels=[2048], num_outs=1): lateral 1x1 + 3x3 out
        self.lateral_conv = nn.Conv2d(2048, embed_dims, 1)
        self.fpn_conv = nn.Conv2d(embed_dims, embed_dims, 3, padding=1)
        xavier_init(self.lateral_conv, distribution="uniform", bias=0.0)
        xavier_init(self.fpn_conv, distribution="uniform", bias=0.0)

    def forward(self, x):                                 # x (B*N, 3, H, W)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)                                # (B*N, 2048, H/32, W/32)
        return self.fpn_conv(self.lateral_conv(x))        # (B*N, 256, H/32, W/32)


# --------------------------------------------------------------------------- #
# Seg upsampler (ResNet18-style), output outC channels at BEV resolution.
# --------------------------------------------------------------------------- #
class _Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear",
                              align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        return self.conv(torch.cat([x2, x1], dim=1))


class SegEncode(nn.Module):
    def __init__(self, inC, outC):
        super().__init__()
        trunk = resnet18(weights=None, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu
        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3
        self.up1 = _Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, 1, padding=0))

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x)
        x = self.layer2(x1)
        x2 = self.layer3(x)
        x = self.up1(x2, x1)
        return self.up2(x)


# --------------------------------------------------------------------------- #
# Top module
# --------------------------------------------------------------------------- #
class BEVFormerNative(nn.Module):
    """Single-frame BEVFormer segmentation. forward(image, lidar2img) -> seg logits.

    Args:
        image     (B, N, 3, H, W)   ImageNet-normalized.
        lidar2img (B, N, 4, 4)      ego/lidar 3D -> image pixel (K4x4 @ E).
    Returns:
        seg_logits (B, outC, bev_h, bev_w)
    """

    def __init__(self, num_classes=1, embed_dims=256, num_cams=6,
                 bev_h=200, bev_w=200, num_layers=6, num_levels=1,
                 num_points_in_pillar=4, feedforward_channels=512,
                 sca_num_points=8, tsa_num_points=4,
                 pc_range=(-50.0, -50.0, -5.0, 50.0, 50.0, 3.0),
                 pretrained_backbone=True, use_checkpoint=True):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.num_levels = num_levels
        self.pc_range = list(pc_range)

        self.img_backbone = ResNet50FPN(embed_dims=embed_dims,
                                        pretrained=pretrained_backbone)
        self.positional_encoding = LearnedPositionalEncoding(
            num_feats=embed_dims // 2, row_num_embed=bev_h, col_num_embed=bev_w)
        self.bev_embedding = nn.Embedding(bev_h * bev_w, embed_dims)
        self.level_embeds = nn.Parameter(torch.Tensor(num_levels, embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(num_cams, embed_dims))
        nn.init.normal_(self.level_embeds)
        nn.init.normal_(self.cams_embeds)

        self.encoder = BEVFormerEncoder(
            num_layers=num_layers, embed_dims=embed_dims, num_cams=num_cams,
            num_levels=num_levels, pc_range=self.pc_range,
            num_points_in_pillar=num_points_in_pillar,
            feedforward_channels=feedforward_channels,
            sca_num_points=sca_num_points, tsa_num_points=tsa_num_points,
            use_checkpoint=use_checkpoint)
        self.seg_decoder = SegEncode(inC=embed_dims, outC=num_classes)

    def get_bev_features(self, mlvl_feats):
        """mlvl_feats: list of (B, N, C, h, w). Returns bev_embed (B, H*W, C)."""
        bs = mlvl_feats[0].size(0)
        device = mlvl_feats[0].device
        dtype = mlvl_feats[0].dtype

        bev_queries = self.bev_embedding.weight.to(dtype)          # (H*W, C)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)    # (H*W, bs, C)
        bev_pos = self.positional_encoding(
            bs, self.bev_h, self.bev_w, device).to(dtype)          # (bs, C, H, W)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)              # (H*W, bs, C)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            feat = feat.flatten(3).permute(1, 0, 3, 2)            # (cam, bs, h*w, c)
            feat = feat + self.cams_embeds[:, None, None, :].to(dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(dtype)
            spatial_shapes.append((h, w))
            feat_flatten.append(feat)
        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)           # (cam, h*w, bs, c)
        return bev_queries, bev_pos, feat_flatten, spatial_shapes, level_start_index

    def forward(self, image, lidar2img):
        B, N, _, H, W = image.shape
        feat = self.img_backbone(image.reshape(B * N, *image.shape[2:]))
        feat = feat.reshape(B, N, feat.shape[1], feat.shape[2], feat.shape[3])
        mlvl_feats = [feat]

        bev_queries, bev_pos, feat_flatten, spatial_shapes, level_start_index = \
            self.get_bev_features(mlvl_feats)

        bev_embed = self.encoder(
            bev_queries, feat_flatten, feat_flatten, self.bev_h, self.bev_w,
            bev_pos, spatial_shapes, level_start_index, lidar2img, H, W)

        # (bs, H*W, C) -> (bs, C, H, W). NOTE: upstream BEVFormer uses
        # reshape(bev_h, bev_w, bs, -1) which is ONLY correct for bs==1 (its
        # samples_per_gpu=1); for bs>1 it scrambles batch into the spatial dims.
        # q index is row-major (h*bev_w + w), so reshape per-sample as (B,H,W,C).
        seg_bev = bev_embed.reshape(B, self.bev_h, self.bev_w, -1).permute(0, 3, 1, 2)
        seg_bev = torch.rot90(seg_bev, k=-1, dims=[2, 3])
        seg_bev = torch.flip(seg_bev, dims=[3])
        return self.seg_decoder(seg_bev)                          # (B, outC, H, W)
