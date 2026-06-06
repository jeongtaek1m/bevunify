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
    ):
        super().__init__()

        # 1 1 3 h w
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)

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
        # self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        ### self attention
        # if bev_self_ffn_dim is None:
        #     bev_self_ffn_dim = dim * 2
        # self.bev_self_atten = BEVSelfAttnBlock(
        #     d_model=dim,
        #     nhead=bev_self_num_head,
        #     dim_feedforward=bev_self_ffn_dim,
        #     dropout=bev_self_dropout,
        # )


        self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

        # --- Dual‑Space Positional Encoding MLPs ---

        self.local_pe = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3,dim),
            nn.GELU(),
            nn.Linear(dim,dim),
        )

        self.global_pe = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4,dim),
            nn.GELU(),
            nn.Linear(dim,dim),
        )

        # --- Image‑Perception Positional Encoding (IPPE) ---
        self.img_feat_pe = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.pe_fusion = nn.Linear(dim * 2, dim)


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

        # --------------------------------------------------------------
        # [1] BEV Self-Attention (논문: BEV query self-attn 단계)
        # --------------------------------------------------------------
        # world_xy = bev.grid[:2]                                            # 2 H W
        # bev_pos = self.bev_embed(world_xy[None])          # 1 d H W
        # bev_pos = bev_pos.expand(b, -1, -1, -1)                             # b d H W
        # x = self.bev_self_atten(x, bev_pos=bev_pos, bev_mask = bev_mask)  # b d H W


        pixel = self.image_plane                                                # b n 3 h w
        _, _, _, h, w = pixel.shape
        
        c = E_inv[..., -1:]                                                     # b n 4 1       (C 그리고 마지막은 1 : homogeneous) #translation cam to lidar R-1*t
        
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]                # (b n) 4 1 1
        c_embed = self.cam_embed(c_flat)                                        # (b n) d 1 1   "τ"

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # 1 1 3 (h w)
        cam = I_inv @ pixel_flat                                                # b n 3 (h w)   <- [b n 3 3] @ [1 1 3 (h w)] each pixel coordinate multiply inverse intrinsic
        # Local Positioning Embedding
        cam_coords_flat = rearrange(cam, 'b n d (h w) -> (b n h w) d', h=h, w=w)
        local_pos = self.local_pe(cam_coords_flat)                              # (b n h w) d



        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                     # b n 4 (h w)   for homogeneous
        d = E_inv @ cam                                                         # b n 4 (h w)   extrinsic 이랑 intrinsic invere 취한 pixel 값 (world 좌표계 근데 depth가 1인) "δ"       
        d_flat = rearrange(d, 'b n d (h w) -> (b n h w) d', h=h, w=w)           # (b n h w) 4 

        # Global Positioning Embedding
        global_pos = self.global_pe(d_flat)                                     # (b n h w) d                               

        feat_flat = rearrange(feature, 'b n d h w -> (b n h w) d')
        local_pos = local_pos * self.img_feat_pe(feat_flat)                     # (b n h w) d   local position embedding
        pe_cat = torch.cat([local_pos, global_pos], dim=-1)                     # (b n h w) 2d
        p = self.pe_fusion(pe_cat)                                              # (b n h w) d   position embedding
        p = rearrange(p, '(b n h w) d -> (b n) d h w', b=b, n=n, h=h, w=w)      # (b n) d h w   position embedding

        # img_embed = p - c_embed                                                 # (b n) d h w  
        img_embed = p
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d h w

        world = bev.grid[:2]                                                    # 2 H W
        w_embed = self.bev_embed(world[None])                                   # 1 d H W
        # The BEV query keeps CVT's camera-aware embedding (w_embed - c_embed). NOTE: an earlier
        # attempt to make it absolute-world (drop - c_embed, on the reading that "DSPE replaces the
        # camera PE") empirically DESTABILIZED training — fix@lr4e-3 diverged at the OneCycle peak
        # while the original survives (0.328). The paper's DSPE/IPPE replaces only the IMAGE-side
        # (key) positional encoding (Eq 1-6 are all image coords); the BEV query is left as CVT's.
        bev_embed = w_embed - c_embed                                           # (b n) d H W
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d H W
        query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)      # b n d H W

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
    """Self→Cross 레이어 스택."""
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
    def forward(self, x, bev, feature, I_inv, E_inv, bev_mask=None):
        for layer in self.layers:
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

        # cross_views = list()
        # layers = list()

        # for feat_shape, num_layers, layer_name in zip(self.backbone.output_shapes, middle, self.backbone.layer_names):
        #     _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape
        #     print(f'layer_name: {layer_name}, feat_shape: {feat_shape}, feat_dim: {feat_dim}, feat_height: {feat_height}, feat_width: {feat_width}')
        #     cva = CrossViewAttention(feat_height, feat_width, feat_dim, dim, **cross_view)
        #     cross_views.append(cva)

        #     layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)])
        #     layers.append(layer)
        
        # self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        # self.cross_views = nn.ModuleList(cross_views)
        # self.layers = nn.ModuleList(layers)
        self.feature_level = feature_level

        # MULTI-SCALE (restored from CVT; the iros2024 branch had collapsed this to a single
        # scale, making DSPE weaker than the CVT baseline). One block per backbone scale,
        # applied coarse->fine on the shared BEV query:
        #   BEV self-attention  ->  DSPE cross-view attention  ->  ResNet bottleneck refine.
        self_blocks = []
        cross_views = []
        layers = []
        for feat_shape, n_mid, name in zip(self.backbone.output_shapes, middle,
                                           self.backbone.layer_names):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape
            print(f'[Encoder] scale {name} -> (dim={feat_dim}, H={feat_height}, W={feat_width})')
            self_blocks.append(BEVSelfAttnBlock(
                d_model=dim,
                nhead=cross_view.get('bev_self_num_head', 8),
                dim_feedforward=cross_view.get('bev_self_ffn_dim', dim * 2),
                dropout=cross_view.get('bev_self_dropout', 0.1)))
            cross_views.append(CrossViewAttention(feat_height, feat_width, feat_dim, dim, **cross_view))
            layers.append(nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(n_mid)]))
        self.self_blocks = nn.ModuleList(self_blocks)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)



    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)            # b n c h w
        I_inv = batch['intrinsics'].inverse()           # b n 3 3
        E_inv = batch['extrinsics'].inverse()           # b n 4 4

        features = [self.down(y) for y in self.backbone(self.norm(image))]   # one per scale

        x = self.bev_embedding.get_prior()              # d H W
        x = repeat(x, '... -> b ...', b=b)              # b d H W

        world_xy = self.bev_embedding.grid[:2]          # 2 H W (global BEV coords for self-attn pos)
        for self_block, cross_view, layer, feature in zip(
                self.self_blocks, self.cross_views, self.layers, features):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            bev_pos = cross_view.bev_embed(world_xy[None]).expand(b, -1, -1, -1)   # b d H W
            x = self_block(x, bev_pos=bev_pos)          # BEV self-attention
            x = cross_view(x, self.bev_embedding, feature, I_inv, E_inv)          # DSPE cross-view
            x = layer(x)                                # ResNet bottleneck refine
        return x