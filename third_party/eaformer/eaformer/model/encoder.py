import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List


ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


# Finite "-inf" marker for behind-camera (geometrically invisible) keys. Far below any
# reachable band log W (worst case ~ -(0.12·530px)^2 ≈ -4e3) so it is distinguishable,
# yet finite so an all-masked query softmaxes to uniform instead of NaN.
EAF_BEHIND_FILL = -3.0e4


def compute_epipolar_geometry(grid, intrinsics, extrinsics):
    """Scale-independent epipolar geometry, shared by all CrossViewAttention stages.

    grid:       (3, H, W) BEV ego-frame coords (BEVEmbedding.grid).
    intrinsics: (b, n, 3, 3) UN-inverted camera intrinsics.
    extrinsics: (b, n, 4, 4) UN-inverted ego->cam (x_cam = E @ x_ego).

    For each BEV cell at ego (X, Y, 0) the vertical world line {(X, Y, z)} projects to
    the epipolar line l = p0 x p1 (two-point projection, z=0 and z=1). Returns fp32:
      l_hat  (b, n, Q, 3)  epipolar line, normalized so a^2+b^2=1 (x·l_hat = px distance)
      behind (b, n, Q, 1)  True where both points are behind the camera
      d_qi   (b, n, Q)     cell -> camera-center distance in ego meters
    """
    device_type = 'cuda' if intrinsics.is_cuda else 'cpu'
    # Geometry must be fp32 regardless of AMP (projection depth & squared terms).
    with torch.autocast(device_type=device_type, enabled=False):
        K = intrinsics.float()                                             # b n 3 3
        E = extrinsics.float()                                             # b n 4 4
        P = K @ E[..., :3, :]                                              # b n 3 4

        X = grid[0].reshape(-1).float()                                    # Q
        Y = grid[1].reshape(-1).float()                                    # Q
        ones = torch.ones_like(X)
        zeros = torch.zeros_like(X)
        pt0 = torch.stack([X, Y, zeros, ones], dim=0)                      # 4 Q  (z=0)
        pt1 = torch.stack([X, Y, ones, ones], dim=0)                       # 4 Q  (z=1)
        pts = torch.cat([pt0, pt1], dim=1)                                 # 4 (2Q)

        proj = P @ pts                                                     # b n 3 (2Q)
        proj = rearrange(proj, 'b n c (two Q) -> b n two Q c', two=2)      # b n 2 Q 3
        p0 = proj[:, :, 0]                                                 # b n Q 3
        p1 = proj[:, :, 1]                                                 # b n Q 3

        line = torch.linalg.cross(p0, p1, dim=-1)                          # b n Q 3
        ab = torch.linalg.norm(line[..., :2], dim=-1, keepdim=True)        # b n Q 1
        l_hat = line / (ab + 1e-7)                                         # a^2 + b^2 = 1

        eps = 1e-6
        behind = (p0[..., 2:3] <= eps) & (p1[..., 2:3] <= eps)             # b n Q 1

        # camera center in ego frame without inverting E: c = -R^T t
        R = E[..., :3, :3]
        t = E[..., :3, 3]
        cam_xyz = -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)     # b n 3
        world_xyz = torch.stack([X, Y, zeros], dim=-1)                     # Q 3
        d_qi = torch.linalg.norm(world_xyz[None, None] - cam_xyz[:, :, None], dim=-1)  # b n Q

    return {'l_hat': l_hat, 'behind': behind, 'd_qi': d_qi}


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

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None, eaf_bias=None, eaf_mode='multiplicative'):
        """
        q: (b n d H W)
        k: (b n d h w)
        v: (b n d h w)
        eaf_bias: (b n Q K) == log W from the Epipolar Attention Field (geometry only,
                  head-independent), or None to fall back to plain CVT cross-attention.
        eaf_mode: 'multiplicative' -> softmax(W ⊙ QK^T/√d)  (paper Eq. 2)
                  'additive'       -> softmax(QK^T/√d + log W)  (log-bias masking, stable)
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
        dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q, k)    # (b m) n Q K

        if eaf_bias is not None:
            # dot's batch axis is (b m) with b OUTER, m (head) INNER (see the q/k/v
            # rearrange above). View it back to (b, m, n, Q, K) so the geometric
            # eaf_bias (b, n, Q, K) broadcasts over heads WITHOUT materialising a
            # per-head copy.
            bh, ncam, Qd, Kd = dot.shape
            bsz = bh // self.heads
            dotv = dot.view(bsz, self.heads, ncam, Qd, Kd)
            bias = eaf_bias[:, None]                                            # b 1 n Q K
            if eaf_mode == 'additive':
                dotv = dotv + bias.to(dotv.dtype)
            else:
                dotv = dotv * bias.exp().to(dotv.dtype)                         # W ∈ (0, 1]
                # A logit multiplied toward 0 is NOT suppressed by softmax (it still
                # beats negative logits), so geometrically invisible (behind-camera)
                # keys — marked with EAF_BEHIND_FILL in the bias — get a hard additive
                # mask on top of the paper's Eq.2 gate. In-place: dotv is the mul
                # output (not saved for backward), mask broadcasts over heads.
                dotv.masked_fill_(bias <= EAF_BEHIND_FILL * 0.5, EAF_BEHIND_FILL)
            dot = dotv.reshape(bh, ncam, Qd, Kd)

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
        # --- Epipolar Attention Field knobs (EAFormer) ---
        use_eaf: bool = True,
        use_pe: bool = True,
        eaf_mode: str = 'multiplicative',
        eaf_lambda: float = 1.0,
        eaf_learnable_lambda: bool = False,
        eaf_bev_range_m: float = 1000.0,
        eaf_d0: float = 12.0,
        eaf_clamp_qi=(0.01, 0.12),
    ):
        super().__init__()

        # Feature-cell CENTER coordinates in true image pixels (1 1 3 h w). The epipolar
        # line lives in the real-intrinsics pixel frame, so keys must sit where the
        # feature cells actually sample: (i+0.5)*stride. (CVT's corner-aligned
        # linspace(0,1)*W grid is misregistered by up to half a cell at the borders —
        # tolerable for a LEARNED embedding, not for the metric EAF.)
        xs = (torch.arange(feat_width, dtype=torch.float32) + 0.5) * (image_width / feat_width)
        ys = (torch.arange(feat_height, dtype=torch.float32) + 0.5) * (image_height / feat_height)
        image_plane = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)   # 2 h w
        image_plane = F.pad(image_plane, (0, 0, 0, 0, 0, 1), value=1)[None, None]  # 1 1 3 h w

        self.register_buffer('image_plane', image_plane, persistent=False)

        # EAF config
        assert eaf_mode in ('multiplicative', 'additive'), f"unknown eaf_mode: {eaf_mode!r}"
        assert use_eaf or use_pe, \
            "use_eaf=False requires use_pe=True (otherwise no spatial correspondence mechanism)"
        self.use_eaf = use_eaf
        self.use_pe = use_pe
        self.eaf_mode = eaf_mode
        self.eaf_bev_range_m = float(eaf_bev_range_m)
        self.eaf_d0 = float(eaf_d0)
        self.eaf_clamp_qi = (float(eaf_clamp_qi[0]), float(eaf_clamp_qi[1]))
        if eaf_learnable_lambda:
            self.eaf_lambda = nn.Parameter(torch.tensor(float(eaf_lambda)))
        else:
            self.register_buffer('eaf_lambda', torch.tensor(float(eaf_lambda)), persistent=False)

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
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

    def forward(
        self,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
        epi: dict = None,
    ):
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)
        epi: shared epipolar geometry (compute_epipolar_geometry), optional.

        Returns: (b, d, H, W)
        """
        b, n, _, _, _ = feature.shape

        pixel = self.image_plane                                                # 1 1 3 h w
        _, _, _, h, w = pixel.shape

        # --- Epipolar Attention Field (geometry-only log W) ---
        eaf_bias = None
        if self.use_eaf:
            eaf_bias = self._epipolar_bias(bev, I_inv, E_inv, epi=epi)         # b n Q K

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')               # (b n) d h w
        val_flat = self.feature_linear(feature_flat)                            # (b n) d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w

        if self.use_pe:
            # CVT camera-aware positional encoding (query_pos / img_embed).
            c = E_inv[..., -1:]                                                 # b n 4 1
            c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]            # (b n) 4 1 1
            c_embed = self.cam_embed(c_flat)                                    # (b n) d 1 1

            pixel_flat = rearrange(pixel, '... h w -> ... (h w)')              # 1 1 3 (h w)
            cam = I_inv @ pixel_flat                                            # b n 3 (h w)
            cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                # b n 4 (h w)
            d = E_inv @ cam                                                     # b n 4 (h w)
            d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)       # (b n) 4 h w
            d_embed = self.img_embed(d_flat)                                    # (b n) d h w

            img_embed = d_embed - c_embed                                       # (b n) d h w
            img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)

            world = bev.grid[:2]                                               # 2 H W
            w_embed = self.bev_embed(world[None])                             # 1 d H W
            bev_embed = w_embed - c_embed                                       # (b n) d H W
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n) # b n d H W

            if self.feature_proj is not None:
                key_flat = img_embed + self.feature_proj(feature_flat)
            else:
                key_flat = img_embed
            query = query_pos + x[:, None]                                     # b n d H W
        else:
            # No positional encoding: spatial correspondence is carried entirely by
            # the Epipolar Attention Field. Query is the learned BEV prior broadcast
            # over cameras; key is the projected image feature.
            assert self.feature_proj is not None, \
                "use_pe=False requires no_image_features=False (key would be empty otherwise)"
            key_flat = self.feature_proj(feature_flat)                         # (b n) d h w
            query = repeat(x, 'b d H W -> b n d H W', n=n)                      # b n d H W

        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)            # b n d h w

        return self.cross_attend(query, key, val, skip=x if self.skip else None,
                                 eaf_bias=eaf_bias, eaf_mode=self.eaf_mode)

    def _epipolar_bias(self, bev, I_inv=None, E_inv=None, epi=None):
        """log W for every (camera, BEV-query, image-key) triple.

        W is a Gaussian over the signed perpendicular distance (in FULL-IMAGE PIXELS —
        scale-consistent across the 1/4 and 1/16 stages) of each feature-cell center
        x_i to the BEV cell's epipolar line l̂:

            log W = -(λ · λ_qi)^2 · (x_i · l̂)^2,   σ_px = 1/(λ·λ_qi)

        λ_qi = (d_qi + d0) / range — paper footprint logic: a fixed-size BEV cell near
        the camera projects to a WIDER image band, far -> narrower.
        Behind-camera (invisible) keys are filled with EAF_BEHIND_FILL (hard suppression).

        epi: shared geometry from compute_epipolar_geometry (hoisted to Encoder.forward).
        Fallback: pass I_inv/E_inv (e.g. standalone/viz use) and it is derived here.

        Returns (b, n, Q, K) where Q = H*W BEV cells, K = h*w feature pixels.
        """
        if epi is None:
            # standalone fallback: recover originals from the inverses
            epi = compute_epipolar_geometry(
                bev.grid, torch.inverse(I_inv.float()), torch.inverse(E_inv.float()))

        device_type = 'cuda' if epi['l_hat'].is_cuda else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            pix = rearrange(self.image_plane[0, 0].float(), 'c h w -> (h w) c')  # K 3
            dist = torch.einsum('b n Q c, K c -> b n Q K', epi['l_hat'], pix)    # px
            dist.pow_(2)   # in-place: dist has no grad (calib + buffers only); halves peak mem

            lam_qi = (epi['d_qi'] + self.eaf_d0) / self.eaf_bev_range_m       # b n Q
            lam_qi = lam_qi.clamp(self.eaf_clamp_qi[0], self.eaf_clamp_qi[1])

            coeff = (self.eaf_lambda.float() * lam_qi).unsqueeze(-1)          # b n Q 1
            neg_bias = -(coeff ** 2) * dist                                   # b n Q K (= log W)
            # Invisible (behind-camera) keys: hard suppression. W=1 "neutral" would let
            # them compete at full strength in the joint softmax over cameras.
            neg_bias = neg_bias.masked_fill_(epi['behind'], EAF_BEHIND_FILL)

        return neg_bias


class Encoder(nn.Module):
    def __init__(
            self,
            backbone,
            cross_view: dict,
            bev_embedding: dict,
            dim: int = 128,
            middle: List[int] = [2, 2],
            scale: float = 1.0,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone

        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)

        cross_views = list()
        layers = list()

        for feat_shape, num_layers in zip(self.backbone.output_shapes, middle):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            cva = CrossViewAttention(feat_height, feat_width, feat_dim, dim, **cross_view)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)])
            layers.append(layer)

        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)            # b n c h w
        I_inv = batch['intrinsics'].inverse()           # b n 3 3
        E_inv = batch['extrinsics'].inverse()           # b n 4 4

        # Scale-independent epipolar geometry, computed ONCE from the un-inverted batch
        # matrices (avoids per-stage double inversion) and shared by all EAF stages.
        epi = None
        if any(getattr(cv, 'use_eaf', False) for cv in self.cross_views):
            epi = compute_epipolar_geometry(
                self.bev_embedding.grid, batch['intrinsics'], batch['extrinsics'])

        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()              # d H W
        x = repeat(x, '... -> b ...', b=b)              # b d H W

        for cross_view, feature, layer in zip(self.cross_views, features, self.layers):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = cross_view(x, self.bev_embedding, feature, I_inv, E_inv, epi=epi)
            x = layer(x)

        return x
