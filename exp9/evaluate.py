"""
Neighbor2Neighbor — Testing & Evaluation Script
==========================================
Test the trained Neighbor2Neighbor U-Net model on DIV2K validation set across
multiple noise levels (σ = 15, 25, 35, 50).
Generate PSNR/SSIM tables and visual comparison figures.

Usage:
    python evaluate.py --mode gray
    python evaluate.py --mode color
    python evaluate.py --mode gray --sigmas 15 25 35 50
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from model import UNet
from dataset import prepare_div2k
from utils import (compute_psnr, compute_ssim, add_gaussian_noise,
                   tensor_to_numpy, save_comparison_figure)

# ---------- Optional: BM3D ----------
HAS_BM3D = False


# ---------- Optional: FFDNet from KAIR ----------
HAS_FFDNET = False



def parse_args():
    p = argparse.ArgumentParser(description="Test Neighbor2Neighbor U-Net denoiser")
    p.add_argument("--mode", type=str, default="gray", choices=["gray", "color"])
    p.add_argument("--sigmas", type=int, nargs="+", default=[15, 25, 35, 50],
                   help="Test noise sigma values")
    p.add_argument("--model_path", type=str, default=None,
                   help="Path to model checkpoint (auto-detected if not set)")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--max_images", type=int, default=20,
                   help="Max number of test images to evaluate (for speed)")
    return p.parse_args()


def load_test_images(img_dir: str, mode: str, max_images: int = 20):
    """Load test images as list of numpy arrays in [0,1]."""
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    paths = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ])[:max_images]

    images = []
    names = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if mode == "gray":
            img = img.convert("L")
        img_np = np.array(img, dtype=np.float32) / 255.0
        images.append(img_np)
        names.append(os.path.splitext(os.path.basename(p))[0])
    return images, names


def denoise_n2n(model, noisy_np, device, mode):
    """Denoise a single image using Neighbor2Neighbor (standard forward pass)."""
    if noisy_np.ndim == 2:
        t = torch.from_numpy(noisy_np).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    else:
        t = torch.from_numpy(noisy_np.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,H,W)
    t = t.to(device)

    # Pad to multiple of 16 for U-Net pooling layers
    _, _, h, w = t.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h > 0 or pad_w > 0:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode='reflect')

    try:
        with torch.no_grad():
            out = model(t).clamp(0, 1)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  [OOM Warning] Fallback to CPU for {mode} image ({h}x{w})")
            torch.cuda.empty_cache()
            model_cpu = model.cpu()
            with torch.no_grad():
                out = model_cpu(t.cpu()).clamp(0, 1)
            model.to(device)  # send model back to GPU
            out = out.to(device)
        else:
            raise e

    # Remove padding
    if pad_h > 0 or pad_w > 0:
        out = out[:, :, :h, :w]

    return tensor_to_numpy(out)


def denoise_bm3d(noisy_np, sigma):
    """Denoise using BM3D."""
    sigma_01 = sigma / 255.0
    if noisy_np.ndim == 2:
        return np.clip(bm3d_lib.bm3d(noisy_np, sigma_psd=sigma_01), 0, 1)
    else:
        return np.clip(bm3d_lib.bm3d(noisy_np, sigma_psd=sigma_01), 0, 1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = args.data_root or os.path.join(script_dir, "data")
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ---------- Load model ----------
    in_ch = 1 if args.mode == "gray" else 3
    model = UNet(in_channels=in_ch).to(device)

    if args.model_path:
        ckpt_path = args.model_path
    else:
        ckpt_path = os.path.join(script_dir, "checkpoints",
                                 f"n2n_unet_{args.mode}_sigma25_best.pth")
    print(f"Loading model: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # ---------- Load test images ----------
    _, valid_dir = prepare_div2k(data_root)
    images, names = load_test_images(valid_dir, args.mode, args.max_images)
    print(f"Loaded {len(images)} test images ({args.mode} mode)")

    # ---------- Evaluate ----------
    # results[sigma][method] = {"psnr": [...], "ssim": [...]}
    all_results = {}
    visual_samples = {}  # store a couple images for visual comparison

    for sigma in args.sigmas:
        print(f"\n{'='*60}")
        print(f"  Testing σ = {sigma}")
        print(f"{'='*60}")
        all_results[sigma] = {}

        # Initialize containers for each method
        methods = ["N2N"]
        if HAS_BM3D:
            methods.append("BM3D")
        for m in methods:
            all_results[sigma][m] = {"psnr": [], "ssim": [], "time": 0.0}

        for idx, (clean, name) in enumerate(zip(images, names)):
            # Generate noisy image (fixed seed per image+sigma for reproducibility)
            noisy = add_gaussian_noise(clean, sigma, seed=idx * 100 + sigma)

            # --- Neighbor2Neighbor ---
            t0 = time.time()
            denoised_n2n = denoise_n2n(model, noisy, device, args.mode)
            all_results[sigma]["N2N"]["time"] += time.time() - t0
            all_results[sigma]["N2N"]["psnr"].append(compute_psnr(clean, denoised_n2n))
            all_results[sigma]["N2N"]["ssim"].append(compute_ssim(clean, denoised_n2n))

            # --- BM3D ---
            if HAS_BM3D:
                t0 = time.time()
                denoised_bm3d = denoise_bm3d(noisy, sigma)
                all_results[sigma]["BM3D"]["time"] += time.time() - t0
                all_results[sigma]["BM3D"]["psnr"].append(compute_psnr(clean, denoised_bm3d))
                all_results[sigma]["BM3D"]["ssim"].append(compute_ssim(clean, denoised_bm3d))

            # Save visual samples for first 2 images at each sigma
            if idx < 2:
                key = (sigma, name)
                visual_samples[key] = {
                    "clean": clean, "noisy": noisy,
                    "N2N": denoised_n2n,
                }
                if HAS_BM3D:
                    visual_samples[key]["BM3D"] = denoised_bm3d

        # Print per-sigma summary
        for m in methods:
            r = all_results[sigma][m]
            if r["psnr"]:
                avg_p = np.mean(r["psnr"])
                avg_s = np.mean(r["ssim"])
                print(f"  {m:10s} | PSNR: {avg_p:.2f} dB | SSIM: {avg_s:.4f} | "
                      f"Time: {r['time']:.2f}s")

    # ---------- Output results table ----------
    print("\n\n" + "=" * 70)
    print(f"  FINAL RESULTS TABLE ({args.mode} mode, {len(images)} images)")
    print("=" * 70)
    header = f"{'σ':>4}  {'Method':>10}  {'PSNR (dB)':>10}  {'SSIM':>8}"
    print(header)
    print("-" * 40)

    table_lines = [header, "-" * 40]
    for sigma in args.sigmas:
        for m in all_results[sigma]:
            r = all_results[sigma][m]
            if r["psnr"]:
                avg_p = np.mean(r["psnr"])
                avg_s = np.mean(r["ssim"])
                line = f"{sigma:>4}  {m:>10}  {avg_p:>10.2f}  {avg_s:>8.4f}"
                print(line)
                table_lines.append(line)
        print("-" * 40)
        table_lines.append("-" * 40)

    # Save table to file
    table_path = os.path.join(results_dir, f"results_table_{args.mode}.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines))
    print(f"\n→ Results table saved: {table_path}")

    # ---------- Visual comparisons ----------
    print("\nGenerating visual comparison figures ...")
    for (sigma, name), sample in visual_samples.items():
        imgs = [sample["clean"], sample["noisy"], sample["N2N"]]
        noisy_psnr = compute_psnr(sample["clean"], sample["noisy"])
        n2n_psnr = compute_psnr(sample["clean"], sample["N2N"])
        titles = [
            "Ground Truth",
            f"Noisy (σ={sigma}, {noisy_psnr:.2f}dB)",
            f"N2N ({n2n_psnr:.2f}dB)",
        ]
        if "BM3D" in sample:
            bm3d_psnr = compute_psnr(sample["clean"], sample["BM3D"])
            imgs.append(sample["BM3D"])
            titles.append(f"BM3D ({bm3d_psnr:.2f}dB)")

        cmap = "gray" if args.mode == "gray" else None
        save_path = os.path.join(results_dir,
                                 f"visual_{args.mode}_sigma{sigma}_{name}.png")
        save_comparison_figure(imgs, titles, save_path,
                               suptitle=f"Neighbor2Neighbor Denoising — σ={sigma}, {name}",
                               cmap=cmap)

    # ---------- PSNR vs Sigma plot ----------
    print("Generating PSNR vs σ plot ...")
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"N2N": "o-", "BM3D": "s--"}
    for m in list(all_results[args.sigmas[0]].keys()):
        psnr_vals = []
        for sigma in args.sigmas:
            if all_results[sigma][m]["psnr"]:
                psnr_vals.append(np.mean(all_results[sigma][m]["psnr"]))
            else:
                psnr_vals.append(0)
        marker = markers.get(m, "d:")
        ax.plot(args.sigmas, psnr_vals, marker, label=m, linewidth=2, markersize=8)

    ax.set_xlabel("Noise Level σ", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title(f"PSNR vs Noise Level ({args.mode} mode)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(args.sigmas)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"psnr_vs_sigma_{args.mode}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {plot_path}")

    print("\n✓ All testing complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
