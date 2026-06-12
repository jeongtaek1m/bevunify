import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List
from copy import deepcopy

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w (h,w has xy componnet)
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w (homogeneous)
    indices = indices[None]                                                 # 1 3 h w 

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))




class BEVSelfAttnBlock(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x_bev, bev_pos = None, bev_mask = None):
        B, C, H, W = x_bev.shape
        bev = x_bev.flatten(2).transpose(1, 2)  # B, H*W, C
        if bev_pos is not None:
            bev_pos = bev_pos.flatten(2).transpose(1, 2)
            q = k = bev + bev_pos
        else:
            q = k = bev
        
        if bev_mask is not None:
            bev_mask = bev_mask.flatten(1)
        
        attn_output, _ = self.self_attn(q, k, bev, attn_mask=bev_mask)
        bev = bev + self.dropout1(attn_output)
        bev = self.norm1(bev)
        ff = self.linear2(self.dropout(self.activation(self.linear1(bev))))
        bev = bev + self.dropout2(ff)
        bev = self.norm2(bev)
        return bev.transpose(1, 2).reshape(B, C, H, W)
    

class BEVEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        sigma: int,
        bev_height: int,
        bev_width: int,
        h_meters: int,
        w_meters: int,
        offset: int,
        decoder_blocks: list,
    ):
        """
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()

        # each decoder block upsamples the bev embedding by a factor of 2
        h = bev_height // (2 ** len(decoder_blocks))
        w = bev_width // (2 ** len(decoder_blocks))

        # bev coordinates
        grid = generate_grid(h, w).squeeze(0)
        grid[0] = bev_width * grid[0]
        grid[1] = bev_height * grid[1]

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()                                  # 3 3
        grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')                      # 3 (h w)
        grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)                    # 3 h w

        # egocentric frame
        self.register_buffer('grid', grid, persistent=False)                    # 3 h w
        self.learned_features = nn.Parameter(sigma * torch.randn(dim, h, w))    # d h w

    def get_prior(self):
        return self.learned_features


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), 
                                  nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), 
                                  nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), 
                                  nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), 
                                 nn.GELU(), 
                                 nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None):
        """
        q: (b n d H W)
        k: (b n d h w)
        v: (b n d h w)
        """
        _, _, _, H, W = q.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b n d H W -> b n (H W) d')
        k = rearrange(k, 'b n d h w -> b n (h w) d')
        v = rearrange(v, 'b n d h w -> b (n h w) d')

        # Project with multiple heads
        q = self.to_q(q)                                # b (n H W) (heads dim_head)
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q, k)
        dot = rearrange(dot, 'b n Q K -> b Q (n K)')
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z


class CrossViewAttention(nn.Module):
    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        heads: int = 4,
        dim_head: int = 32,
        no_image_features: bool = False,
        skip: bool = True,
        bev_self_num_head: int = 8,
        bev_self_ffn_dim: int = None,
        bev_self_dropout: float = 0.1,
        pe_depth_bins: int = 16,
        pe_depth_min: float = 1.0,
        pe_depth_max: float = 61.2,
    ):
        super().__init__()

        # 1 1 3 h w
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)

        # D depth bins (paper Sec III-B: each 2D coordinate maps to D 3D points c_d).
        # PETR-style linear discretization along the viewing ray.
        self.register_buffer('pe_depths',
                             torch.linspace(float(pe_depth_min), float(pe_depth_max),
                                            int(pe_depth_bins)), persistent=False)

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))
        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.bev_embed = nn.Conv2d(2, dim, 1)

        self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

        # --- Dual-Space Positional Encoding MLPs (paper Sec III-B) ---
        # local  : camera-frame frustum points (K^-1 only — extrinsic-FREE)
        # global : the same points in the reference frame (extrinsics applied)
        # Each half outputs dim/2; the final PE is their CONCAT (paper Eq. 5) so that
        # under extrinsic noise only the global half degrades while the clean local
        # channels survive in their own subspace (no learned fusion mixing them).
        assert dim % 2 == 0, "dim must be even for Concat(local, global) PE halves"
        d_in = int(pe_depth_bins) * 3
        self.local_pe = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
        )

        self.global_pe = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
        )

        # --- Image-Perception Positional Encoding (IPPE, paper Eq. 6) ---
        # gates the LOCAL half: p_local = MLP(LN(c_cam)) ∘ MLP(LN(F_i))
        self.img_feat_pe = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
        )


    def forward(
        self,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
        bev_mask = None,  # Optional mask for BEV self-attention
    ):
        """
        x: (b, c, H, W)                 bev embedding
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)             cam to lidar

        Returns: (b, d, H, W)
        """
        b, n, _, _, _ = feature.shape

        pixel = self.image_plane                                                # 1 1 3 h w
        _, _, _, h, w = pixel.shape

        # --- DSPE: D depth-bin frustum points per pixel (paper Sec III-B) ---
        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # 1 1 3 (h w)
        ray = I_inv @ pixel_flat                                                # b n 3 (h w)  z=1 ray
        D = self.pe_depths.numel()
        cam_pts = ray.unsqueeze(2) * self.pe_depths.view(1, 1, D, 1, 1)         # b n D 3 (h w)

        # LOCAL PE — camera frame, extrinsic-FREE (intrinsics only)
        local_in = rearrange(cam_pts, 'b n D c K -> (b n K) (D c)')            # (b n h w) D*3
        local_pos = self.local_pe(local_in)                                     # (b n h w) d/2

        # GLOBAL PE — same points in the reference frame (extrinsics applied)
        cam_h = F.pad(cam_pts, (0, 0, 0, 1), value=1)                           # b n D 4 (h w)
        glob = torch.einsum('b n i j, b n D j K -> b n D i K', E_inv, cam_h)    # b n D 4 (h w)
        glob_in = rearrange(glob[:, :, :, :3], 'b n D c K -> (b n K) (D c)')   # slice homog. 1
        global_pos = self.global_pe(glob_in)                                    # (b n h w) d/2

        # IPPE gate on the LOCAL half (paper Eq. 6), then CONCAT (paper Eq. 5)
        feat_flat_pe = rearrange(feature, 'b n d h w -> (b n h w) d')
        local_pos = local_pos * self.img_feat_pe(feat_flat_pe)                  # (b n h w) d/2
        p = torch.cat([local_pos, global_pos], dim=-1)                          # (b n h w) d
        p = rearrange(p, '(b n h w) d -> (b n) d h w', b=b, n=n, h=h, w=w)      # (b n) d h w

        # KEY PE: absolute (paper decouples extrinsics into the global half only)
        img_embed = p
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d h w

        # QUERY PE: absolute-world BEV coordinates — SAME frame convention as the key
        # (CVT's camera-relative w_embed - c_embed would re-inject extrinsics into
        # every query, undoing the decoupling the paper's robustness claim rests on).
        world = bev.grid[:2]                                                    # 2 H W
        w_embed = self.bev_embed(world[None])                                   # 1 d H W
        w_embed = w_embed / (w_embed.norm(dim=1, keepdim=True) + 1e-7)          # 1 d H W
        query_pos = repeat(w_embed[0], 'd H W -> b n d H W', b=b, n=n)          # b n d H W

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')               # (b n) d h w   "φ"

        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)              # (b n) d h w
        else:
            key_flat = img_embed                                                # (b n) d h w

        val_flat = self.feature_linear(feature_flat)                            # (b n) d h w

        # Expand + refine the BEV embedding
        query = query_pos + x[:, None]                                          # b n d H W
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w

        return self.cross_attend(query, key, val, skip=x if self.skip else None)

class BEVTransformerLayer(nn.Module):
    """한 트랜스포머 레이어: BEV Self-Attn -> Cross-View Attn."""
    def __init__(self, dim, bev_self_num_head, bev_self_ffn_dim, bev_self_dropout,
                 cross_view_module: CrossViewAttention):
        super().__init__()
        self.self_block = BEVSelfAttnBlock(dim, bev_self_num_head, bev_self_ffn_dim, bev_self_dropout)
        self.cross_block = cross_view_module

    def forward(self, x, bev, feature, I_inv, E_inv, bev_mask=None):
        # BEV pos (global XY) for self-attn
        world_xy = bev.grid[:2]                         # (2,H,W)
        bev_pos = self.cross_block.bev_embed(world_xy[None])  # reuse module's conv
        bev_pos = bev_pos.expand(x.size(0), -1, -1, -1)
        x = self.self_block(x, bev_pos=bev_pos, bev_mask=bev_mask)
        x = self.cross_block(x, bev, feature, I_inv, E_inv)
        return x


class BEVTransformerEncoder(nn.Module):
    """Self→Cross layer stack; layer i attends image-feature scale i (paper Fig. 2:
    three encoder layers paired with the three backbone scales)."""
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
    def forward(self, x, bev, features, I_inv, E_inv, bev_mask=None):
        for layer, feature in zip(self.layers, features):
            x = layer(x, bev, feature, I_inv, E_inv, bev_mask=bev_mask)
        return x
    

class Encoder(nn.Module):
    def __init__(
            self,
            backbone,
            cross_view: dict,
            bev_embedding: dict,
            dim: int = 128,
            middle: List[int] = [2, 2, 2],
            num_layers: int = 3,
            feature_level: int = -1,
            scale: float = 1.0,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone

        if scale < 1.0: #why?
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)
        assert len(self.backbone.output_shapes) == num_layers, \
            "paper pairs each of the 3 encoder layers with one backbone scale"
        self.feature_level = feature_level   # accepted for config compat; multi-scale is used

        # One INDEPENDENTLY-initialized Self->Cross layer per backbone scale
        # (paper Fig. 2: three scales feed the x3 encoder; no weight cloning).
        layers = []
        for feat_shape in self.backbone.output_shapes:
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape
            cross_i = CrossViewAttention(feat_height, feat_width, feat_dim, dim, **cross_view)
            layer = BEVTransformerLayer(
                dim=dim,
                bev_self_num_head=cross_view.get('bev_self_num_head', 8),
                bev_self_ffn_dim=cross_view.get('bev_self_ffn_dim', dim*2),
                bev_self_dropout=cross_view.get('bev_self_dropout', 0.1),
                cross_view_module=cross_i,
            )
            layers.append(layer)
        self.transformer = BEVTransformerEncoder(nn.ModuleList(layers))

        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)            # b n c h w
        I_inv = batch['intrinsics'].inverse()           # b n 3 3
        E_inv = batch['extrinsics'].inverse()           # b n 4 4

        features = [rearrange(self.down(y), '(b n) ... -> b n ...', b=b, n=n)
                    for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()              # d H W
        x = repeat(x, '... -> b ...', b=b)              # b d H W

        x = self.transformer(x, self.bev_embedding, features, I_inv, E_inv)
        return x


        # return x