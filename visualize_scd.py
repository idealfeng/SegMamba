"""
Visualize SCD predictions on a few patches (target | prediction side-by-side).

Run:
  python visualize_scd.py --ckpt outputs/checkpoints/best_scd.pth --split test --n 12 --model segformer_tcn_b1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.cuda.amp import autocast
from tqdm import tqdm

from config import cfg
from dataset import create_dataloaders
from models.segmamba_scd import build_model as build_mamba_model
from models.segformer_tcn_scd import build_model as build_tcn_model


def _to_u8(days: np.ndarray) -> np.ndarray:
    max_days = float(getattr(cfg, "SCD_MAX_DAYS", 181))
    x = np.clip(days, 0.0, max_days) / max_days
    return (x * 255.0 + 0.5).astype(np.uint8)


def _build_model():
    mt = str(getattr(cfg, "MODEL_TYPE", "segformer_tcn_b1")).lower()
    if mt.startswith("segformer_tcn"):
        return build_tcn_model()
    return build_mamba_model()


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize SCD predictions")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--model", type=str, default=str(getattr(cfg, "MODEL_TYPE", "segformer_tcn_b1")))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out-dir", type=str, default=str(cfg.VIS_DIR / "scd"))
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="visualize"):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True).float()
            with autocast(enabled=use_amp):
                out = model(x)
            pred = out["pred"].detach().cpu().numpy()  # (B,1,H,W)
            targ = y.detach().cpu().numpy()

            for i in range(pred.shape[0]):
                p = _to_u8(pred[i, 0])
                t = _to_u8(targ[i, 0])
                side = np.concatenate([t, p], axis=1)
                img = Image.fromarray(side, mode="L")
                img.save(out_dir / f"{args.split}_{cfg.MODEL_TYPE}_{saved:04d}.png")
                saved += 1
                if saved >= args.n:
                    print(f"[WROTE] {saved} images to {out_dir}")
                    return 0

    print(f"[WROTE] {saved} images to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

