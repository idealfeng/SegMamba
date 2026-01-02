"""
SegMamba for Snow Cover Days (SCD) regression.

Goal: per-pixel regression of Snow Cover Days in a winter season, range 0..cfg.SCD_MAX_DAYS.

This model keeps a SegFormer-like multi-scale encoder/decoder structure, but replaces the Transformer
backbone with lightweight 2D state-space (Mamba-style) mixing blocks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg


class SSMScan1D(nn.Module):
    """
    Simple diagonal state-space scan:
      s[t] = a * s[t-1] + b * x[t]
      y[t] = c * s[t] + d * x[t]

    Parameters are per-channel (diagonal), which keeps it fast and stable for small sequence lengths.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.alpha_logit = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.ones(dim) * 0.1)
        self.gamma = nn.Parameter(torch.ones(dim) * 0.1)
        self.delta = nn.Parameter(torch.ones(dim) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)
        returns: (B, L, C)
        """
        if x.ndim != 3:
            raise ValueError(f"SSMScan1D expects (B,L,C), got {tuple(x.shape)}")
        bsz, length, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"dim mismatch: x={dim} module={self.dim}")

        a = torch.sigmoid(self.alpha_logit).view(1, 1, dim)
        b = self.beta.view(1, 1, dim)
        c = self.gamma.view(1, 1, dim)
        d = self.delta.view(1, 1, dim)

        s = x.new_zeros((bsz, dim))
        ys = []
        for t in range(length):
            xt = x[:, t, :]
            s = s * a[:, 0, :] + xt * b[:, 0, :]
            yt = s * c[:, 0, :] + xt * d[:, 0, :]
            ys.append(yt)
        return torch.stack(ys, dim=1)


class Mamba2DBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        hidden = int(dim * mlp_ratio)

        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.in_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)

        self.scan_h = SSMScan1D(dim)
        self.scan_v = SSMScan1D(dim)

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.mlp = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=dim),
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden, dim, kernel_size=1),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def _scan_rows(self, x: torch.Tensor, scan: SSMScan1D) -> torch.Tensor:
        # x: (B,C,H,W) -> (B*H, W, C)
        bsz, ch, h, w = x.shape
        seq = x.permute(0, 2, 3, 1).contiguous().view(bsz * h, w, ch)
        y_f = scan(seq)
        y_b = torch.flip(scan(torch.flip(seq, dims=[1])), dims=[1])
        y = 0.5 * (y_f + y_b)
        return y.view(bsz, h, w, ch).permute(0, 3, 1, 2).contiguous()

    def _scan_cols(self, x: torch.Tensor, scan: SSMScan1D) -> torch.Tensor:
        # x: (B,C,H,W) -> (B*W, H, C)
        bsz, ch, h, w = x.shape
        seq = x.permute(0, 3, 2, 1).contiguous().view(bsz * w, h, ch)
        y_f = scan(seq)
        y_b = torch.flip(scan(torch.flip(seq, dims=[1])), dims=[1])
        y = 0.5 * (y_f + y_b)
        return y.view(bsz, w, h, ch).permute(0, 3, 2, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        u, gate = self.in_proj(x).chunk(2, dim=1)
        u = self.dwconv(u)

        u = 0.5 * (self._scan_rows(u, self.scan_h) + self._scan_cols(u, self.scan_v))
        u = u * torch.sigmoid(gate)
        u = self.out_proj(u)
        u = self.drop(u)
        x = residual + u
        x = x + self.mlp(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MLPDecoder(nn.Module):
    def __init__(self, in_channels_list: list[int], embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_layers = nn.ModuleList([nn.Conv2d(ch, embed_dim, 1) for ch in in_channels_list])
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels_list), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[2:]
        aligned = []
        for i, feat in enumerate(features):
            feat = self.linear_layers[i](feat)
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            aligned.append(feat)
        fused = torch.cat(aligned, dim=1)
        return self.fusion(fused)


@dataclass(frozen=True)
class SegMambaCfg:
    dims: tuple[int, int, int, int]
    depths: tuple[int, int, int, int]


VARIANTS: dict[str, SegMambaCfg] = {
    "tiny": SegMambaCfg(dims=(64, 128, 256, 512), depths=(2, 2, 4, 2)),
}


class SegMambaSCD(nn.Module):
    def __init__(self, in_ch: int, variant: str = "tiny", dropout: float = 0.0):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Available: {sorted(VARIANTS)}")

        v = VARIANTS[variant]
        self.in_ch = in_ch
        self.variant = variant

        self.stem = PatchEmbed(in_ch, v.dims[0], stride=4)
        self.stage1 = nn.Sequential(*[Mamba2DBlock(v.dims[0], dropout=dropout) for _ in range(v.depths[0])])
        self.down1 = Downsample(v.dims[0], v.dims[1])

        self.stage2 = nn.Sequential(*[Mamba2DBlock(v.dims[1], dropout=dropout) for _ in range(v.depths[1])])
        self.down2 = Downsample(v.dims[1], v.dims[2])

        self.stage3 = nn.Sequential(*[Mamba2DBlock(v.dims[2], dropout=dropout) for _ in range(v.depths[2])])
        self.down3 = Downsample(v.dims[2], v.dims[3])

        self.stage4 = nn.Sequential(*[Mamba2DBlock(v.dims[3], dropout=dropout) for _ in range(v.depths[3])])

        self.decoder = MLPDecoder(list(v.dims), embed_dim=256)
        self.head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (B, C, H, W)
        returns: dict with 'pred' in days, shape (B,1,H,W)
        """
        h, w = x.shape[2], x.shape[3]

        f1 = self.stage1(self.stem(x))          # 1/4
        f2 = self.stage2(self.down1(f1))        # 1/8
        f3 = self.stage3(self.down2(f2))        # 1/16
        f4 = self.stage4(self.down3(f3))        # 1/32

        fused = self.decoder([f1, f2, f3, f4])  # 1/4
        logits = self.head(fused)
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)

        pred = torch.sigmoid(logits) * float(getattr(cfg, "SCD_MAX_DAYS", 181))
        return {"pred": pred}


def build_model(variant: str | None = None, in_ch: int | None = None) -> SegMambaSCD:
    """
    Factory function for training scripts.
    cfg.MODEL_TYPE: e.g. "segmamba_tiny"
    """
    if variant is None:
        mt = str(getattr(cfg, "MODEL_TYPE", "segmamba_tiny"))
        variant = mt.split("_", 1)[-1] if "_" in mt else "tiny"
    if in_ch is None:
        in_ch = int(getattr(cfg, "SCD_IN_CHANNELS", 11))
    return SegMambaSCD(in_ch=in_ch, variant=variant, dropout=0.0)

