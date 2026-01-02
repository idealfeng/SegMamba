"""
Train script for Snow Cover Days (SCD) regression.

Recommended (baseline):
  python train_scd.py --model segformer_tcn_b1 --batch-size 1 --num-workers 2 --epochs 200

Mamba variant:
  python train_scd.py --model segmamba_tiny --batch-size 2 --num-workers 2 --epochs 200
"""

from __future__ import annotations

import argparse
from datetime import datetime

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import cfg
from dataset import create_dataloaders
from losses.losses import build_criterion
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
    ap = argparse.ArgumentParser(description="Train SCD regression models")
    ap.add_argument("--model", type=str, default=str(getattr(cfg, "MODEL_TYPE", "segformer_tcn_b1")))
    ap.add_argument("--epochs", type=int, default=int(getattr(cfg, "NUM_EPOCHS", 200)))
    ap.add_argument("--batch-size", type=int, default=int(getattr(cfg, "BATCH_SIZE", 1)))
    ap.add_argument("--num-workers", type=int, default=int(getattr(cfg, "NUM_WORKERS", 2)))
    ap.add_argument("--lr", type=float, default=float(getattr(cfg, "LEARNING_RATE", 5e-4)))
    ap.add_argument("--grad-accum", type=int, default=int(getattr(cfg, "GRADIENT_ACCUMULATION_STEPS", 1)))
    ap.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs (default: 1).")
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    cfg.TASK = "scd"
    cfg.MODEL_TYPE = args.model

    device = torch.device(getattr(cfg, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers, crop_size=int(getattr(cfg, "CROP_SIZE", 256))
    )

    model = _build_model().to(device)
    criterion = build_criterion().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=float(getattr(cfg, "WEIGHT_DECAY", 0.01)),
    )

    use_amp = bool(getattr(cfg, "USE_AMP", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    log_dir = str(cfg.LOG_DIR / ("scd_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    writer = SummaryWriter(log_dir=log_dir) if bool(getattr(cfg, "USE_TENSORBOARD", True)) else None

    start_epoch = 0
    best_val_mae = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optim"])
        if ckpt.get("scaler") is not None and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_mae = float(ckpt.get("best_val_mae", best_val_mae))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"train {epoch+1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for batch in pbar:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True).float()

            with autocast(enabled=use_amp):
                out = model(x)
                loss, _ = criterion(out, y)
                loss = loss / max(1, args.grad_accum)
            scaler.scale(loss).backward()

            if (n_batches + 1) % max(1, args.grad_accum) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            m = _metrics(out["pred"], y)
            total_loss += float((loss.detach().cpu().item()) * max(1, args.grad_accum))
            total_mae += m["mae"]
            total_rmse += m["rmse"]
            n_batches += 1
            pbar.set_postfix(
                loss=total_loss / n_batches, mae=total_mae / n_batches, rmse=total_rmse / n_batches
            )

        train_loss = total_loss / max(1, n_batches)
        train_mae = total_mae / max(1, n_batches)
        train_rmse = total_rmse / max(1, n_batches)

        do_val = (args.val_every > 0) and ((epoch + 1) % args.val_every == 0)
        if do_val:
            model.eval()
            val_loss = 0.0
            val_mae = 0.0
            val_rmse = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="val", leave=False):
                    x = batch["x"].to(device, non_blocking=True)
                    y = batch["y"].to(device, non_blocking=True).float()
                    with autocast(enabled=use_amp):
                        out = model(x)
                        loss, _ = criterion(out, y)
                    m = _metrics(out["pred"], y)
                    val_loss += float(loss.detach().cpu().item())
                    val_mae += m["mae"]
                    val_rmse += m["rmse"]
                    n_val += 1

            val_loss /= max(1, n_val)
            val_mae /= max(1, n_val)
            val_rmse /= max(1, n_val)
        else:
            val_loss = float("nan")
            val_mae = float("nan")
            val_rmse = float("nan")

        if writer and do_val:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("mae/train", train_mae, epoch)
            writer.add_scalar("rmse/train", train_rmse, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("mae/val", val_mae, epoch)
            writer.add_scalar("rmse/val", val_rmse, epoch)

        ckpt_path = cfg.CHECKPOINT_DIR / "last_scd.pth"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "best_val_mae": best_val_mae,
                "cfg": {"TASK": "scd", "MODEL_TYPE": cfg.MODEL_TYPE, "SCD_INPUT_MODE": cfg.SCD_INPUT_MODE},
            },
            ckpt_path,
        )

        if do_val and (val_mae < best_val_mae):
            best_val_mae = val_mae
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if use_amp else None,
                    "best_val_mae": best_val_mae,
                    "cfg": {"TASK": "scd", "MODEL_TYPE": cfg.MODEL_TYPE, "SCD_INPUT_MODE": cfg.SCD_INPUT_MODE},
                },
                cfg.CHECKPOINT_DIR / "best_scd.pth",
            )

        if do_val:
            print(
                f"[EPOCH {epoch+1}] train loss={train_loss:.4f} mae={train_mae:.3f} rmse={train_rmse:.3f} | "
                f"val loss={val_loss:.4f} mae={val_mae:.3f} rmse={val_rmse:.3f} | best_mae={best_val_mae:.3f}"
            )
        else:
            print(
                f"[EPOCH {epoch+1}] train loss={train_loss:.4f} mae={train_mae:.3f} rmse={train_rmse:.3f} | "
                f"val skipped (every {args.val_every}) | best_mae={best_val_mae:.3f}"
            )

    if writer:
        writer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
