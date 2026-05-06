"""
UltraLightBlockNet (Utvaa) — Core Architecture.

Hybrid CNN–Transformer network combining Dilated Bottleneck convolutions with
an AffixAttentionBlock that fuses local linear self-attention and coordinate
attention for efficient image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class ConvLayer(nn.Module):
    """Flexible 2-D convolution with optional BatchNorm and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        use_norm: bool = True,
        use_act: bool = True,
        act_layer: nn.Module = nn.SiLU,
        bias: bool = False,
        groups: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                bias=bias, groups=groups, dilation=dilation,
            )
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_act:
            layers.append(act_layer())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DilatedBottleneck(nn.Module):
    """Residual depthwise bottleneck with optional dilation."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, stride: int = 1, dilation: int = 1):
        super().__init__()

        if stride == 1:
            padding = dilation
        elif stride == 2:
            padding = 1 if dilation == 1 else 2 if dilation == 2 else (2 * dilation - 1) // 2
        else:
            raise ValueError(f"Unsupported stride: {stride}")

        self.use_res = in_ch == out_ch and stride == 1

        self.block = nn.Sequential(
            ConvLayer(in_ch, mid_ch, kernel_size=1),
            ConvLayer(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=padding,
                      groups=mid_ch, dilation=dilation),
            ConvLayer(mid_ch, out_ch, kernel_size=1, use_act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_res and x.shape == out.shape:
            out = out + x
        return out


class LocalRepresentationBlock(nn.Module):
    """Depthwise + pointwise projection from CNN feature space to Transformer dimension."""

    def __init__(self, Cin: int, TransformerDim: int):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(Cin, Cin, kernel_size=3, padding=1, groups=Cin)
        self.pointwise_conv = nn.Conv2d(Cin, TransformerDim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        return self.pointwise_conv(x)


class FeedForward(nn.Module):
    """Standard Transformer FFN with LayerNorm, SiLU, and Dropout."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearSelfAttention(nn.Module):
    """
    Linear-complexity self-attention inspired by MobileViTv2.

    Operates on patch tokens shaped (B, patch_pixels, num_patches, embed_dim).
    """

    def __init__(self, embed_dim: int, attn_dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.qkv_proj = ConvLayer(
            embed_dim, 1 + 2 * embed_dim,
            bias=bias, kernel_size=1, use_norm=False, use_act=False,
        )
        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = ConvLayer(
            embed_dim, embed_dim,
            bias=bias, kernel_size=1, use_norm=False, use_act=False,
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, patch_pixels, num_patches, embed_dim)
        x_conv = rearrange(x, 'b p n d -> b d p n')
        qkv = self.qkv_proj(x_conv)
        query, key, value = torch.split(qkv, [1, self.embed_dim, self.embed_dim], dim=1)

        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        context_vector = (key * context_scores).sum(dim=-1, keepdim=True)  # (B, D, P, 1)
        out = self.out_proj(F.relu(value) * context_vector.expand_as(value))
        return rearrange(out, 'b d p n -> b p n d')


class Transformer(nn.Module):
    """Stack of LinearSelfAttention + FeedForward blocks."""

    def __init__(self, dim: int, depth: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                LinearSelfAttention(embed_dim=dim, attn_dropout=dropout),
                FeedForward(dim, mlp_dim, dropout),
            ])
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class CoordinateAttention(nn.Module):
    """Coordinate Attention for spatial-channel recalibration."""

    def __init__(self, inp: int, oup: int, groups: int = 32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // groups)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1)
        self.relu = h_swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.relu(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        x_h = self.conv2(x_h).sigmoid().expand(-1, -1, h, w)
        x_w = self.conv3(x_w).sigmoid().expand(-1, -1, h, w)
        return identity * x_h * x_w


class AffixAttentionBlock(nn.Module):
    """
    Core hybrid block: local CNN features + global linear attention + coordinate attention.

    The block unfolds spatial features into non-overlapping patches, runs a lightweight
    Transformer over them, then fuses the result with coordinate-attended CNN features.
    """

    def __init__(self, Cin: int, TransformerDim: int, Cout: int, depth: int = 2, patch_size: int = 2):
        super().__init__()
        if Cin >= TransformerDim:
            raise ValueError(f"Cin ({Cin}) must be < TransformerDim ({TransformerDim})")

        self.patch_size = patch_size
        self.TransformerDim = TransformerDim
        self.Cin = Cin

        self.local = LocalRepresentationBlock(Cin, TransformerDim)
        self.transformer = Transformer(
            dim=TransformerDim,
            depth=depth,
            mlp_dim=TransformerDim * 2,
            dropout=0.25,
        )
        self.conv_proj = nn.Conv2d(TransformerDim, Cin, kernel_size=1)
        self.coord_att = CoordinateAttention(inp=Cin, oup=Cin)
        self.fusion = nn.Conv2d(2 * Cin, Cout, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ph = pw = self.patch_size
        h_patches, w_patches = H // ph, W // pw

        # Pad input to TransformerDim channels via zero-padding
        padding = torch.zeros(B, self.TransformerDim - C, H, W, device=x.device, dtype=x.dtype)
        res_padded = torch.cat([x, padding], dim=1)

        local_out = self.local(x)  # (B, TransformerDim, H, W)

        # Unfold into patches
        local_patches = rearrange(local_out, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=ph, pw=pw)
        res_patches = rearrange(res_padded, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=ph, pw=pw)

        seq = self.transformer(local_patches + res_patches)

        # Fold back and project
        fm = rearrange(seq, 'b (ph pw) (h w) d -> b d (h ph) (w pw)',
                       ph=ph, pw=pw, h=h_patches, w=w_patches)
        projected = self.conv_proj(fm)             # (B, Cin, H, W)
        attention = self.coord_att(x)              # (B, Cin, H, W)
        out = self.fusion(torch.cat([attention, projected], dim=1))
        return out + x                              # residual


# ---------------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------------

_VARIANTS = {
    "xlarge": dict(dims=[112, 132], channels=[8, 32, 72, 198, 616]),
    "large":  dict(dims=[112, 132], channels=[8, 32, 72, 128, 416]),
    "medium": dict(dims=[96,  112], channels=[8, 32, 64,  96, 352]),
    "tiny":   dict(dims=[48,   64], channels=[8, 16, 32,  48, 288]),
}


class UltraLightBlockNet_L1(nn.Module):
    """
    UltraLightBlockNet — ultra-lightweight hybrid CNN-Transformer classifier.

    Architecture stages (256×256 input):
      Stem   : 256×256 → 128×128
      Stage 1: 128×128 →  32×32  (Dilated Bottlenecks)
      Stage 2:  32×32 →  16×16  (Dilated Bottleneck + AffixAttentionBlock)
      Stage 3:  16×16 →  16×16  (Dilated Bottlenecks + AffixAttentionBlock)
      Head   :  16×16 →  1×1    (Global AvgPool)
      Classifier: Linear → num_classes
    """

    def __init__(self, num_classes: int, image_size: int, dims: list, channels: list):
        super().__init__()
        expansion = 2

        self.stem = nn.Sequential(
            ConvLayer(3, channels[0], kernel_size=3, stride=2, padding=1)
        )

        self.stage1 = nn.Sequential(
            DilatedBottleneck(channels[0], expansion * channels[0], channels[0], stride=2),
            DilatedBottleneck(channels[0], expansion * channels[0], channels[1], stride=2),
            DilatedBottleneck(channels[1], expansion * channels[1], channels[1], stride=1),
        )

        self.stage2 = nn.Sequential(
            DilatedBottleneck(channels[1], expansion * channels[1], channels[2], stride=2, dilation=2),
            AffixAttentionBlock(Cin=channels[2], TransformerDim=dims[0], Cout=channels[2], depth=2),
        )

        self.stage3 = nn.Sequential(
            DilatedBottleneck(channels[2], expansion * channels[2], channels[3], stride=1, dilation=4),
            DilatedBottleneck(channels[3], expansion * channels[3], channels[3], stride=1),
            AffixAttentionBlock(Cin=channels[3], TransformerDim=dims[1], Cout=channels[3], depth=3),
        )

        self.head = nn.Sequential(
            ConvLayer(channels[3], channels[4], kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(channels[4], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        return self.classifier(x)

    @classmethod
    def from_variant(cls, variant: str, num_classes: int, image_size: int = 256) -> "UltraLightBlockNet_L1":
        """Instantiate a named model variant (tiny / medium / large / xlarge)."""
        if variant not in _VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(_VARIANTS)}")
        cfg = _VARIANTS[variant]
        return cls(num_classes=num_classes, image_size=image_size, **cfg)
