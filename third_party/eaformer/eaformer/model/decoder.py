import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, skip_dim, residual, factor):
        super().__init__()

        dim = out_channels // factor

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))

        if residual:
            self.up = nn.Conv2d(skip_dim, out_channels, 1)
        else:
            self.up = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.conv(x)

        if self.up is not None:
            up = self.up(skip)
            up = F.interpolate(up, x.shape[-2:])

            x = x + up

        return self.relu(x)


class Decoder(nn.Module):
    def __init__(self, dim, blocks, residual=True, factor=2):
        super().__init__()

        layers = list()
        channels = dim

        for out_channels in blocks:
            layer = DecoderBlock(channels, out_channels, dim, residual, factor)
            layers.append(layer)

            channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        y = x

        for layer in self.layers:
            y = layer(y, x)

        return y


# ---------------------------------------------------------------------------
# ASPP semantic head (EAFormer). DeepLab-style Atrous Spatial Pyramid Pooling
# applied to the BEV encoder output, followed by the CVT upsampling decoder.
# ---------------------------------------------------------------------------
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        y = self.conv(self.gap(x))
        return F.interpolate(y, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=(6, 12, 18), dropout=0.1):
        super().__init__()

        branches = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)),
        ]
        for rate in atrous_rates:
            branches.append(ASPPConv(in_channels, out_channels, rate))
        branches.append(ASPPPooling(in_channels, out_channels))
        self.branches = nn.ModuleList(branches)

        self.project = nn.Sequential(
            nn.Conv2d(len(branches) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout))

    def forward(self, x):
        res = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.project(res)


class ASPPUpBlock(nn.Module):
    """One decoder block: ASPP context -> 2x upsample, plus a CVT-style long-range
    residual skip from the (coarse) encoder BEV output injected at every stage
    (interpolated + 1x1 to match channels). The skip restores the gradient/identity
    path that the original CVT decoder had (DecoderBlock x = x + up)."""
    def __init__(self, in_channels, out_channels, skip_dim, atrous_rates, dropout):
        super().__init__()
        self.aspp = ASPP(in_channels, out_channels, atrous_rates=tuple(atrous_rates), dropout=dropout)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.skip = nn.Conv2d(skip_dim, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x0):
        y = self.up(self.aspp(x))
        s = F.interpolate(self.skip(x0), size=y.shape[-2:], mode='bilinear', align_corners=True)
        return self.relu(y + s)


class ASPPDecoder(nn.Module):
    """EAFormer decoder = three ASPP blocks (supp. Sec. 1: "a decoder with three ASPP
    blocks"). Each block applies ASPP context, upsamples 2x, and adds the CVT-style
    long-range skip from the encoder BEV output; a final 3x3 conv refines the full-res
    map. For blocks=[128,128,64] the BEV grid goes 25 -> 50 -> 100 -> 200. Exposes
    ``out_channels`` / ``forward(x)`` like ``Decoder`` (NOTE: Decoder's residual/factor
    args are intentionally NOT accepted — the skip here is always on).
    Modest atrous rates suit the coarse BEV maps.
    """
    def __init__(self, dim, blocks, atrous_rates=(2, 4, 6), dropout=0.1):
        super().__init__()

        layers = []
        channels = dim
        for out_channels in blocks:
            layers.append(ASPPUpBlock(channels, out_channels, dim, atrous_rates, dropout))
            channels = out_channels
        self.layers = nn.ModuleList(layers)
        # full-resolution refinement (the last upsample previously had no conv at 200x200)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.out_channels = channels

    def forward(self, x):
        x0 = x                              # 25x25 encoder BEV output (skip source)
        for layer in self.layers:
            x = layer(x, x0)
        return self.refine(x)
