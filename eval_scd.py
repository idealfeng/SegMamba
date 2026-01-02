"""
Evaluate a trained SCD model on val/test splits.

Run:
  python eval_scd.py --ckpt outputs/checkpoints/best_scd.pth --split test --model segformer_tcn_b1
"""

from __future__ import annotations

import argparse

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from config import cfg
from dataset import create_dataloaders
from models.segmamba_scd import build_model as build_mamba_model
from models.segformer_tcn_scd import build_model as build_tcn_model


def _metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    diff = pred - target
    mae = diff.abs().mean()
    rmse = diff.pow(2).mean().sqrt()
    return {"mae": float(mae.detach().cpu().item()), "rmse": float(rmse.detach().cpu().item())}


def _build_model():
    mt = str(getattr(cfg, "MODEL_TYPE", "segformer_tcn_b1")).lower()
    if mt.startswith("segformer_tcn"):
        return build_tcn_model()
    return build_mamba_model()


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate SCD regression models")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--model", type=str, default=str(getattr(cfg, "MODEL_TYPE", "segformer_tcn_b1")))
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    cfg.TASK = "scd"
    cfg.MODEL_TYPE = args.model

    device = torch.device(getattr(cfg, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    _, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers, crop_size=int(getattr(cfg, "CROP_SIZE", 256))
    )
    loader = val_loader if args.split == "val" else test_loader

    model = _build_model().to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    use_amp = bool(getattr(cfg, "USE_AMP", True)) and device.type == "cuda"

    total_mae = 0.0
    total_rmse = 0.0
    n = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval({args.split})"):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True).float()
            with autocast(enabled=use_amp):
                out = model(x)
            m = _metrics(out["pred"], y)
            total_mae += m["mae"]
            total_rmse += m["rmse"]
            n += 1

    mae = total_mae / max(1, n)
    rmse = total_rmse / max(1, n)
    print(f"[SUMMARY] model={cfg.MODEL_TYPE} split={args.split} batches={n} mae={mae:.3f} rmse={rmse:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

