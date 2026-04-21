"""
Self2Self and DIP — Training Scripts
======================================
Train models for Gaussian noise removal using:
  1. Self2Self: Bernoulli sampling + Dropout + ensemble inference
  2. DIP (Deep Image Prior): Fixed random input + network structure prior + early stopping

Usage:
    # Self2Self Training
    python train.py --method self2self --mode gray --sigma 25
    python train.py --method self2self --mode color --sigma 25

    # DIP Training (single image)
    python train.py --method dip --mode gray --sigma 25 --image_path path/to/image.png
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from model import Self2SelfUNet, DIPDecoder, count_parameters
from dataset import Self2SelfDataset, SingleImageDIPDataset, prepare_div2k
from utils import compute_psnr, tensor_to_numpy, plot_training_curves


def parse_args():
    p = argparse.ArgumentParser(description="Train Self2Self or DIP denoiser")
    p.add_argument("--method", type=str, default="self2self", choices=["self2self", "dip"],
                   help="Training method: self2self or dip")
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
    p.add_argument("--mask_ratio", type=float, default=0.3,
                   help="Bernoulli mask ratio for Self2Self")
    p.add_argument("--num_samples", type=int, default=20,
                   help="Number of ensemble samples for Self2Self inference")
    p.add_argument("--image_path", type=str, default=None,
                   help="Path to single image for DIP training")
    p.add_argument("--data_root", type=str, default=None,
                   help="Path to dataset root (default: ./data)")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers")
    return p.parse_args()


def validate_self2self(model, val_loader, device, num_samples: int = 20):
    """Run validation for Self2Self with ensemble denoising."""
    model.eval()
    psnr_sum = 0.0
    count = 0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            # Ensemble denoising
            output = model.ensemble_denoise(noisy, num_samples=num_samples).clamp(0, 1)
            # Compute per-image PSNR
            for i in range(output.size(0)):
                out_np = tensor_to_numpy(output[i])
                cln_np = tensor_to_numpy(clean[i])
                psnr_sum += compute_psnr(cln_np, out_np)
                count += 1
    return psnr_sum / max(count, 1)


def train_self2self(args, device, script_dir, data_root):
    """Train Self2Self model."""
    ckpt_dir = os.path.join(script_dir, "checkpoints")
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ---------- Dataset ----------
    print("[1/5] Preparing DIV2K dataset for Self2Self ...")
    train_dir, valid_dir = prepare_div2k(data_root)

    train_ds = Self2SelfDataset(
        train_dir, sigma=args.sigma, mode=args.mode,
        train=True, patch_size=args.patch_size, mask_ratio=args.mask_ratio,
    )
    val_ds = Self2SelfDataset(
        valid_dir, sigma=args.sigma, mode=args.mode,
        train=False, patch_size=args.patch_size,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ---------- Model ----------
    in_ch = 1 if args.mode == "gray" else 3
    model = Self2SelfUNet(in_channels=in_ch, dropout_rate=0.5).to(device)
    print(f"[2/5] Self2Self U-Net ({args.mode}) created — {count_parameters(model):,} parameters")

    # ---------- Optimizer & Scheduler ----------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda")

    # ---------- Training loop ----------
    print(f"[3/5] Training Self2Self for {args.epochs} epochs "
          f"(σ={args.sigma}, batch={args.batch_size}, mask_ratio={args.mask_ratio}) ...")
    best_psnr = 0.0
    train_losses = []
    val_psnrs = []
    model_name = f"s2s_unet_{args.mode}_sigma{int(args.sigma)}"

    total_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for masked_input, noisy_target, mask in train_loader:
            masked_input = masked_input.to(device)
            noisy_target = noisy_target.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            with autocast("cuda"):
                # Forward pass (Dropout is enabled in training mode)
                output = model(masked_input)

                # Loss only on masked pixels
                diff = (output - noisy_target) * mask
                loss = (diff ** 2).sum() / mask.sum()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validate every 5 epochs or last epoch
        if epoch % 5 == 0 or epoch == args.epochs:
            val_psnr = validate_self2self(model, val_loader, device, num_samples=args.num_samples)
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
        os.path.join(results_dir, f"s2s_unet_{args.mode}_sigma{int(args.sigma)}_training_curves.png"),
        method_name="Self2Self",
    )


def train_dip(args, device, script_dir):
    """Train DIP (Deep Image Prior) on a single image."""
    ckpt_dir = os.path.join(script_dir, "checkpoints")
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ---------- Single Image Dataset ----------
    print("[1/4] Loading single image for DIP ...")
    if args.image_path is None:
        # Default: use first image from DIV2K valid set
        data_root = args.data_root or os.path.join(script_dir, "data")
        _, valid_dir = prepare_div2k(data_root)
        args.image_path = valid_dir

    ds = SingleImageDIPDataset(args.image_path, sigma=args.sigma, mode=args.mode)
    noisy, clean = ds[0]
    noisy = noisy.unsqueeze(0).to(device)
    clean = clean.unsqueeze(0).to(device)

    # ---------- Model ----------
    in_ch = 1 if args.mode == "gray" else 3
    model = DIPDecoder(in_channels=32, out_channels=in_ch, num_upsample=8).to(device)
    # Generate random input with the SAME spatial size as the noisy image
    z = model.generate_random_input(batch_size=1, device=device,
                                    target_height=noisy.shape[2], target_width=noisy.shape[3])
    print(f"[2/4] DIP Decoder ({args.mode}) created — {count_parameters(model):,} parameters")

    # ---------- Optimizer ----------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---------- Training loop ----------
    print(f"[3/4] Training DIP for {args.epochs} iterations (σ={args.sigma}) ...")
    best_psnr = 0.0
    best_iter = 0
    train_losses = []
    val_psnrs = []
    model_name = f"dip_{args.mode}_sigma{int(args.sigma)}"

    total_start = time.time()
    for iteration in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward pass from random input
        output = model(z, target_size=(noisy.shape[2], noisy.shape[3]))

        # MSE loss to noisy image
        loss = nn.functional.mse_loss(output, noisy)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Evaluate every 10 iterations
        if iteration % 10 == 0 or iteration == args.epochs:
            with torch.no_grad():
                output_np = tensor_to_numpy(output[0])
                clean_np = tensor_to_numpy(clean[0])
                psnr = compute_psnr(clean_np, output_np)
            val_psnrs.append(psnr)

            # Save best model (early stopping point)
            if psnr > best_psnr:
                best_psnr = psnr
                best_iter = iteration
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, f"{model_name}_best.pth"))

            dt = time.time() - total_start
            print(f"  Iter {iteration:5d}/{args.epochs} | "
                  f"Loss: {loss.item():.6f} | PSNR: {psnr:.2f} dB | "
                  f"Best: {best_psnr:.2f} dB (iter {best_iter}) | "
                  f"Time: {dt:.1f}s")

    total_time = time.time() - total_start
    print(f"\n[4/4] DIP training complete in {total_time / 60:.1f} min — "
          f"Best PSNR: {best_psnr:.2f} dB at iteration {best_iter}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{model_name}_final.pth"))

    # ---------- Plot curves ----------
    print("Saving training curves ...")
    plot_training_curves(
        train_losses, val_psnrs,
        os.path.join(results_dir, f"dip_{args.mode}_sigma{int(args.sigma)}_training_curves.png"),
        method_name="DIP",
    )

    # Save denoised result
    with torch.no_grad():
        output = model(z, target_size=(noisy.shape[2], noisy.shape[3]))
    from utils import save_comparison_figure
    save_comparison_figure(
        [tensor_to_numpy(clean[0]), tensor_to_numpy(noisy[0]), tensor_to_numpy(output[0])],
        ["Clean", f"Noisy (σ={args.sigma})", f"DIP Denoised"],
        os.path.join(results_dir, f"dip_denoised_result_sigma{int(args.sigma)}.png"),
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Use data from exp9 if available
    default_data_root = os.path.join(os.path.dirname(script_dir), "exp9", "data")
    data_root = args.data_root or default_data_root

    if args.method == "self2self":
        train_self2self(args, device, script_dir, data_root)
    elif args.method == "dip":
        train_dip(args, device, script_dir)


if __name__ == "__main__":
    main()
