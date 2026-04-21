"""
U-Net Denoising — Training Script
===================================
Train a U-Net model for Gaussian noise removal on DIV2K dataset.

Usage:
    python train.py --mode gray          # Train grayscale model (σ=25)
    python train.py --mode color         # Train color (RGB) model (σ=25)
    python train.py --mode gray --epochs 80 --batch_size 8   # Custom settings
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from model import UNet, count_parameters
from dataset import DIV2KDenoisingDataset, prepare_div2k
from utils import compute_psnr, tensor_to_numpy, plot_training_curves


def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net denoiser")
    p.add_argument("--mode", type=str, default="gray", choices=["gray", "color"],
                   help="Training mode: gray (1-ch) or color (3-ch)")
    p.add_argument("--sigma", type=float, default=25.0,
                   help="Training noise sigma (in [0,255] scale)")
    p.add_argument("--epochs", type=int, default=50,
                   help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Batch size")
    p.add_argument("--patch_size", type=int, default=128,
                   help="Training patch size")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Initial learning rate")
    p.add_argument("--data_root", type=str, default=None,
                   help="Path to dataset root (default: ./data)")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers")
    return p.parse_args()


def validate(model, val_loader, device):
    """Run validation, return average PSNR on val set."""
    model.eval()
    psnr_sum = 0.0
    count = 0
    with torch.no_grad():
        for noisy1, noisy2, clean in val_loader:
            noisy1, clean = noisy1.to(device), clean.to(device)
            output = model(noisy1).clamp(0, 1)
            # Compute per-image PSNR
            for i in range(output.size(0)):
                out_np = tensor_to_numpy(output[i])
                cln_np = tensor_to_numpy(clean[i])
                psnr_sum += compute_psnr(cln_np, out_np)
                count += 1
    return psnr_sum / max(count, 1)


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------- Paths ----------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = args.data_root or os.path.join(script_dir, "data")
    ckpt_dir = os.path.join(script_dir, "checkpoints")
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ---------- Dataset ----------
    print("[1/5] Preparing DIV2K dataset ...")
    train_dir, valid_dir = prepare_div2k(data_root)

    train_ds = DIV2KDenoisingDataset(
        train_dir, sigma=args.sigma, mode=args.mode,
        train=True, patch_size=args.patch_size,
    )
    val_ds = DIV2KDenoisingDataset(
        valid_dir, sigma=args.sigma, mode=args.mode,
        train=True, patch_size=args.patch_size,  # use patches for faster val
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ---------- Model ----------
    in_ch = 1 if args.mode == "gray" else 3
    model = UNet(in_channels=in_ch).to(device)
    print(f"[2/5] U-Net ({args.mode}) created — {count_parameters(model):,} parameters")

    # ---------- Optimizer & Scheduler ----------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    scaler = GradScaler("cuda")  # Mixed precision

    # ---------- Training loop ----------
    print(f"[3/5] Training for {args.epochs} epochs (σ={args.sigma}, batch={args.batch_size}) ...")
    best_psnr = 0.0
    train_losses = []
    val_psnrs = []
    model_name = f"n2n_unet_{args.mode}_sigma{int(args.sigma)}"

    total_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for noisy1, noisy2, clean in train_loader:
            noisy1, noisy2 = noisy1.to(device), noisy2.to(device)

            optimizer.zero_grad()
            with autocast("cuda"):

                output = model(noisy1)
                # Noise2Noise: Compare output of noisy1 with noisy2
                loss = criterion(output, noisy2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validate every 5 epochs or last epoch
        if epoch % 5 == 0 or epoch == args.epochs:
            val_psnr = validate(model, val_loader, device)
            val_psnrs.append(val_psnr)
            dt = time.time() - t0

            # Save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, f"{model_name}_best.pth"))

            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {avg_loss:.6f} | Val PSNR: {val_psnr:.2f} dB | "
                  f"Best: {best_psnr:.2f} dB | LR: {scheduler.get_last_lr()[0]:.6f} | "
                  f"Time: {dt:.1f}s")
        else:
            val_psnrs.append(val_psnrs[-1] if val_psnrs else 0)
            dt = time.time() - t0
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {avg_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f} | Time: {dt:.1f}s")

    total_time = time.time() - total_start
    print(f"\n[4/5] Training complete in {total_time / 60:.1f} min — Best PSNR: {best_psnr:.2f} dB")

    # Save final model
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{model_name}_final.pth"))

    # ---------- Plot curves ----------
    print("[5/5] Saving training curves ...")
    plot_training_curves(
        train_losses, val_psnrs,
        os.path.join(results_dir, f"{model_name}_training_curves.png"),
    )
    print(f"Done! Checkpoints in: {ckpt_dir}")


if __name__ == "__main__":
    train()
