"""
Baseline: SegFormer + TemporalConv (TCN/1D Conv) for SCD regression.

Input:
  x: (B, C, H, W), where C = 2*T + 4 if cfg.SCD_INPUT_MODE == "temporal"
     dynamic channels layout: [temp_day1..temp_dayT, prcp_day1..prcp_dayT, elev, slope, north, ndvi]

Output:
  dict with 'pred': (B, 1, H, W) in days (0..cfg.SCD_MAX_DAYS)
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SegformerForSemanticSegmentation, SegformerConfig

from config import cfg


class TemporalConv(nn.Module):
    def __init__(
        self,
        in_vars: int = 2,
        out_ch: int = 32,
        kernels: list[int] | None = None,
        strides: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        kernels = kernels or [5, 5, 3]
        strides = strides or [2, 2, 1]
        if len(strides) != len(kernels):
            raise ValueError("strides length must match kernels length")
        layers = []
        ch = in_vars
        for i, k in enumerate(kernels):
            pad = k // 2
            layers.append(nn.Conv1d(ch, out_ch, kernel_size=k, padding=pad, stride=int(strides[i])))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: (BHW, V, T)
        returns: (BHW, C)
        """
        y = self.net(seq)  # (BHW, C, T)
        return y.mean(dim=-1)  # temporal average


class SegFormerTCNSCD(nn.Module):
    def __init__(self, variant: str = "b1"):
        super().__init__()
        t = int(getattr(cfg, "SCD_T", 181))
        max_days = float(getattr(cfg, "SCD_MAX_DAYS", 181))

        self.t = t
        self.max_days = max_days

        self.tcn_out = int(getattr(cfg, "SCD_TCN_CHANNELS", 32))
        kernels = list(getattr(cfg, "SCD_TCN_KERNELS", [5, 5, 3]))
        strides = list(getattr(cfg, "SCD_TCN_STRIDES", [2, 2, 1]))
        dropout = float(getattr(cfg, "SCD_TCN_DROPOUT", 0.1))
        self.tcn = TemporalConv(in_vars=2, out_ch=self.tcn_out, kernels=kernels, strides=strides, dropout=dropout)
        self.spatial_pool = int(getattr(cfg, "SCD_TCN_SPATIAL_POOL", 4))

        # After TCN, we concat 4 static channels.
        in_ch = int(self.tcn_out + 4)

        local_weights = Path("pretrained_weights") / f"segformer_{variant}"
        model_name = str(local_weights) if local_weights.exists() else ""

        # Build from config (no from_pretrained() to avoid torch>=2.6 restriction in newer Transformers).
        base_cfg = SegformerConfig.from_pretrained(model_name) if model_name else SegformerConfig()
        base_cfg.num_labels = 1
        base_cfg.num_channels = in_ch
        base_cfg.semantic_loss_ignore_index = -100

        self.segformer = SegformerForSemanticSegmentation(base_cfg)

        # Optional: load local pretrained weights if available.
        # We only load keys with matching shapes to avoid size-mismatch errors (num_channels/num_labels changed).
        if bool(getattr(cfg, "PRETRAINED", True)) and model_name:
            bin_path = local_weights / "pytorch_model.bin"
            if bin_path.exists():
                state = torch.load(str(bin_path), map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model_sd = self.segformer.state_dict()
                filtered = {k: v for k, v in state.items() if k in model_sd and hasattr(v, "shape") and v.shape == model_sd[k].shape}
                missing = len(model_sd) - len(filtered)
                self.segformer.load_state_dict(filtered, strict=False)
                if missing > 0:
                    # keep it quiet; users can inspect if needed
                    pass

        # We do regression in [0, max_days] using sigmoid scaling.
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (B, 2*T+4, H, W) from dataset in temporal mode.
        """
        bsz, ch, h, w = x.shape
        t = self.t
        expect = 2 * t + 4
        if ch != expect:
            raise ValueError(f"Expected input channels {expect} (=2*T+4), got {ch}. Check cfg.SCD_INPUT_MODE/SCD_T.")

        temp = x[:, 0:t, :, :]          # (B,T,H,W)
        prcp = x[:, t:2 * t, :, :]      # (B,T,H,W)
        static = x[:, 2 * t:2 * t + 4, :, :]  # (B,4,H,W)

        # The naive per-pixel TCN on full-res (256x256) explodes memory.
        # We pool spatially first (default 4x), run TCN on the pooled grid, then upsample back.
        p = self.spatial_pool
        if p > 1:
            temp_ds = F.avg_pool2d(temp, kernel_size=p, stride=p)  # (B,T,H/p,W/p)
            prcp_ds = F.avg_pool2d(prcp, kernel_size=p, stride=p)
        else:
            temp_ds, prcp_ds = temp, prcp

        h2, w2 = temp_ds.shape[-2], temp_ds.shape[-1]

        # per-(pooled)pixel temporal conv: (B,H2,W2,2,T) -> (B*H2*W2,2,T)
        seq = torch.stack([temp_ds, prcp_ds], dim=1)  # (B,2,T,H2,W2)
        seq = seq.permute(0, 3, 4, 1, 2).contiguous().view(bsz * h2 * w2, 2, t)
        emb = self.tcn(seq)  # (B*H2*W2,C)
        emb = emb.view(bsz, h2, w2, self.tcn_out).permute(0, 3, 1, 2).contiguous()  # (B,C,H2,W2)
        if (h2, w2) != (h, w):
            emb = F.interpolate(emb, size=(h, w), mode="bilinear", align_corners=False)

        feat = torch.cat([emb, static], dim=1)  # (B, C+4, H, W)
        logits = self.segformer(pixel_values=feat).logits  # (B,1,h/4,w/4) in HF
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        pred = self.act(logits) * self.max_days
        return {"pred": pred}


def build_model() -> SegFormerTCNSCD:
    mt = str(getattr(cfg, "MODEL_TYPE", "segformer_tcn_b1"))
    # MODEL_TYPE like: segformer_tcn_b1
    variant = mt.split("_")[-1] if "_" in mt else "b1"
    return SegFormerTCNSCD(variant=variant)
