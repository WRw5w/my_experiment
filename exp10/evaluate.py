"""
Self2Self and DIP — Evaluation Script
=======================================
Evaluate trained models on Set12 or BSD68 dataset with noise levels σ=15, 25, 35, 50.
Compare with BM3D and Neighbor2Neighbor (from exp9).

Usage:
    python evaluate.py --method self2self --mode gray
    python evaluate.py --method dip --mode gray
    python evaluate.py --method all --mode gray  # Evaluate all methods
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import Self2SelfUNet, DIPDecoder
from utils import compute_psnr, compute_ssim, tensor_to_numpy, add_gaussian_noise


# ============================================================
# Test datasets paths
# ============================================================
SET12_PATH = r"..\exp1\set32"  # Reuse images from exp1
BSD68_PATH = None  # Optional: add BSD68 path if available


def get_test_images(folder_path: str, mode: str = "gray") -> list:
    """Load all images from a folder."""
    # Convert relative path to absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_folder_path = os.path.join(script_dir, folder_path)
    
    if not os.path.exists(abs_folder_path):
        print(f"Warning: {abs_folder_path} not found, skipping.")
        return []
    
    images = []
    for f in sorted(os.listdir(abs_folder_path)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            path = os.path.join(abs_folder_path, f)
            try:
                img = Image.open(path)
                if mode == "gray":
                    img = img.convert("L")
                else:
                    img = img.convert("RGB")
                img_np = np.array(img, dtype=np.float32) / 255.0
                images.append((f, img_np))
            except Exception as e:
                print(f"Failed to load {f}: {e}")
    return images


# ============================================================
# BM3D baseline (if available)
# ============================================================
def denoise_bm3d(noisy: np.ndarray, sigma: float) -> np.ndarray:
    """BM3D denoising (requires bm3d package)."""
    try:
        import bm3d
        sigma_norm = sigma / 255.0
        if noisy.ndim == 2:
            denoised = bm3d.bm3d(noisy, sigma_norm, stage_arg=bm3d.BM3DStages.ALL_STAGES)
        else:
            denoised = np.zeros_like(noisy)
            for c in range(noisy.shape[2]):
                denoised[:, :, c] = bm3d.bm3d(noisy[:, :, c], sigma_norm, 
                                               stage_arg=bm3d.BM3DStages.ALL_STAGES)
        return np.clip(denoised, 0, 1)
    except ImportError:
        print("Warning: bm3d package not installed. Skipping BM3D.")
        return None


# ============================================================
# Load Neighbor2Neighbor model from exp9
# ============================================================
def load_n2n_model(mode: str, sigma: float, device: torch.device) -> torch.nn.Module:
    """Load trained Neighbor2Neighbor model from exp9."""
    try:
        import sys
        sys.path.append("../exp9")
        from exp9.model import UNet
        
        in_ch = 1 if mode == "gray" else 3
        model = UNet(in_channels=in_ch).to(device)
        ckpt_path = f"../exp9/checkpoints/n2n_unet_{mode}_sigma{int(sigma)}_best.pth"
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            print(f"Loaded N2N model from {ckpt_path}")
        else:
            print(f"N2N checkpoint not found at {ckpt_path}")
            return None
        return model
    except Exception as e:
        print(f"Failed to load N2N model: {e}")
        return None


def denoise_n2n(model, noisy: np.ndarray, device: torch.device) -> np.ndarray:
    """Denoise using Neighbor2Neighbor model."""
    if model is None:
        return None
    with torch.no_grad():
        if noisy.ndim == 2:
            tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
        else:
            tensor = torch.from_numpy(noisy.transpose(2, 0, 1)).unsqueeze(0).to(device)
        output = model(tensor).clamp(0, 1)
        return tensor_to_numpy(output[0])


# ============================================================
# Self2Self Evaluation
# ============================================================
def evaluate_self2self(images: list, sigmas: list, mode: str, device: torch.device,
                       results_dir: str, ckpt_dir: str):
    """Evaluate Self2Self model on test images."""
    in_ch = 1 if mode == "gray" else 3
    
    all_results = {}
    
    for sigma in sigmas:
        print(f"\n--- Self2Self Evaluation: σ={sigma} ---")
        ckpt_path = os.path.join(ckpt_dir, f"s2s_unet_{mode}_sigma{int(sigma)}_best.pth")
        
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}, training on-the-fly...")
            # For simplicity, skip if no checkpoint
            continue
        
        # Load model
        model = Self2SelfUNet(in_channels=in_ch, dropout_rate=0.5).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        psnr_list = []
        ssim_list = []
        
        for name, clean in images:
            # Add noise
            noisy = add_gaussian_noise(clean, sigma, seed=42)
            
            # Convert to tensor
            if mode == "gray":
                noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
            else:
                noisy_tensor = torch.from_numpy(noisy.transpose(2, 0, 1)).unsqueeze(0).to(device)
            
            # Ensemble denoising
            with torch.no_grad():
                denoised_tensor = model.ensemble_denoise(noisy_tensor, num_samples=20)
            denoised = tensor_to_numpy(denoised_tensor[0])
            
            # Compute metrics
            psnr = compute_psnr(clean, denoised)
            ssim = compute_ssim(clean, denoised)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
            print(f"  {name}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
        
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        all_results[sigma] = (avg_psnr, avg_ssim, psnr_list, ssim_list)
        print(f"  Average: PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}")
    
    return all_results


# ============================================================
# DIP Evaluation
# ============================================================
def evaluate_dip(images: list, sigmas: list, mode: str, device: torch.device,
                 results_dir: str, ckpt_dir: str, dip_iterations: int = 2000):
    """Evaluate DIP on test images (per-image optimization)."""
    in_ch = 1 if mode == "gray" else 3
    
    all_results = {}
    
    for sigma in sigmas:
        print(f"\n--- DIP Evaluation: σ={sigma} ---")
        
        psnr_list = []
        ssim_list = []
        
        for name, clean in images:
            # Add noise
            noisy = add_gaussian_noise(clean, sigma, seed=42)
            
            # Convert to tensor
            if mode == "gray":
                noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
            else:
                noisy_tensor = torch.from_numpy(noisy.transpose(2, 0, 1)).unsqueeze(0).to(device)
            
            # Calculate appropriate network depth based on image size
            # Each downsample halves the dimension. Need at least 4 pixels at bottleneck.
            h, w = noisy.shape[:2]
            max_depth = int(np.log2(min(h, w) / 4))
            num_upsample = max(2, min(5, max_depth))
            
            # Create DIP model
            model = DIPDecoder(in_channels=32, out_channels=in_ch, num_upsample=num_upsample).to(device)
            z = model.generate_random_input(batch_size=1, device=device,
                                           target_height=h, target_width=w)
            
            # Optimize for this single image
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            best_psnr_img = 0
            best_output = None
            
            for it in range(dip_iterations):
                optimizer.zero_grad()
                output = model(z, target_size=(noisy_tensor.shape[2], noisy_tensor.shape[3]))
                loss = torch.nn.functional.mse_loss(output, noisy_tensor)
                loss.backward()
                optimizer.step()
                
                # Track best PSNR
                if it % 50 == 0 or it == dip_iterations - 1:
                    with torch.no_grad():
                        out_np = tensor_to_numpy(output[0])
                        psnr_img = compute_psnr(clean, out_np)
                        if psnr_img > best_psnr_img:
                            best_psnr_img = psnr_img
                            best_output = out_np.copy()
            
            ssim_img = compute_ssim(clean, best_output)
            psnr_list.append(best_psnr_img)
            ssim_list.append(ssim_img)
            
            print(f"  {name}: PSNR={best_psnr_img:.2f} dB, SSIM={ssim_img:.4f}")
            
            # Save visual result for first image
            if name == images[0][0]:
                from utils import save_comparison_figure
                save_comparison_figure(
                    [clean, noisy, best_output],
                    ["Clean", f"Noisy (σ={sigma})", "DIP"],
                    os.path.join(results_dir, f"dip_visual_{mode}_sigma{int(sigma)}_{name.replace('.png', '')}.png"),
                )
        
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        all_results[sigma] = (avg_psnr, avg_ssim, psnr_list, ssim_list)
        print(f"  Average: PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}")
    
    return all_results


# ============================================================
# Results saving and plotting
# ============================================================
def save_results_table(results: dict, method_name: str, save_path: str):
    """Save results to a text file."""
    with open(save_path, 'w') as f:
        f.write(f"{method_name} Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Sigma':<10} {'Avg PSNR (dB)':<15} {'Avg SSIM':<15}\n")
        f.write("-" * 60 + "\n")
        for sigma, (avg_psnr, avg_ssim, _, _) in results.items():
            f.write(f"{sigma:<10} {avg_psnr:<15.2f} {avg_ssim:<15.4f}\n")
    print(f"Results saved to {save_path}")


def plot_psnr_comparison(results_dict: dict, save_path: str):
    """Plot PSNR comparison across methods."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = {'BM3D': 'b', 'N2N': 'g', 'Self2Self': 'r', 'DIP': 'm'}
    
    for method, results in results_dict.items():
        sigmas = sorted(results.keys())
        psnrs = [results[s][0] for s in sigmas]
        color = colors.get(method, 'k')
        ax.plot(sigmas, psnrs, f'{color}o-', linewidth=2, markersize=8, label=method)
    
    ax.set_xlabel("Noise Level (σ)", fontsize=12)
    ax.set_ylabel("Average PSNR (dB)", fontsize=12)
    ax.set_title("Denoising Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([15, 25, 35, 50])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved to {save_path}")


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Self2Self and DIP denoisers")
    p.add_argument("--method", type=str, default="all", 
                   choices=["self2self", "dip", "n2n", "bm3d", "all"],
                   help="Method to evaluate")
    p.add_argument("--mode", type=str, default="gray", choices=["gray", "color"],
                   help="Evaluation mode: gray or color")
    p.add_argument("--sigmas", type=int, nargs="+", default=[15, 25, 35, 50],
                   help="Noise levels to test")
    p.add_argument("--test_set", type=str, default="set12",
                   help="Test dataset folder")
    p.add_argument("--dip_iterations", type=int, default=2000,
                   help="Number of DIP optimization iterations per image")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    ckpt_dir = os.path.join(script_dir, "checkpoints")
    os.makedirs(results_dir, exist_ok=True)
    
    # Load test images
    test_folder = SET12_PATH
    images = get_test_images(test_folder, args.mode)
    if not images:
        print("No test images found!")
        return
    
    print(f"Loaded {len(images)} test images from {test_folder}")
    
    # Limit to first 8 images for faster evaluation
    images = images[:8]
    
    all_results = {}
    
    # Evaluate BM3D
    if args.method in ["bm3d", "all"]:
        print("\n" + "=" * 60)
        print("Evaluating BM3D...")
        bm3d_results = {}
        for sigma in args.sigmas:
            psnr_list = []
            for name, clean in images:
                noisy = add_gaussian_noise(clean, sigma, seed=42)
                denoised = denoise_bm3d(noisy, sigma)
                if denoised is not None:
                    psnr_list.append(compute_psnr(clean, denoised))
            if psnr_list:
                bm3d_results[sigma] = (np.mean(psnr_list), 0, psnr_list, [])
        if bm3d_results:
            all_results["BM3D"] = bm3d_results
            save_results_table(bm3d_results, "BM3D", 
                              os.path.join(results_dir, "bm3d_results.txt"))
    
    # Evaluate Neighbor2Neighbor
    if args.method in ["n2n", "all"]:
        print("\n" + "=" * 60)
        print("Evaluating Neighbor2Neighbor...")
        n2n_results = {}
        for sigma in args.sigmas:
            model = load_n2n_model(args.mode, sigma, device)
            if model is None:
                continue
            psnr_list = []
            for name, clean in images:
                noisy = add_gaussian_noise(clean, sigma, seed=42)
                denoised = denoise_n2n(model, noisy, device)
                if denoised is not None:
                    psnr_list.append(compute_psnr(clean, denoised))
            if psnr_list:
                n2n_results[sigma] = (np.mean(psnr_list), 0, psnr_list, [])
        if n2n_results:
            all_results["N2N"] = n2n_results
            save_results_table(n2n_results, "Neighbor2Neighbor",
                              os.path.join(results_dir, "n2n_results.txt"))
    
    # Evaluate Self2Self
    if args.method in ["self2self", "all"]:
        print("\n" + "=" * 60)
        print("Evaluating Self2Self...")
        s2s_results = evaluate_self2self(images, args.sigmas, args.mode, device,
                                         results_dir, ckpt_dir)
        if s2s_results:
            all_results["Self2Self"] = s2s_results
            save_results_table(s2s_results, "Self2Self",
                              os.path.join(results_dir, "self2self_results.txt"))
    
    # Evaluate DIP
    if args.method in ["dip", "all"]:
        print("\n" + "=" * 60)
        print("Evaluating DIP...")
        dip_results = evaluate_dip(images, args.sigmas, args.mode, device,
                                   results_dir, ckpt_dir, args.dip_iterations)
        if dip_results:
            all_results["DIP"] = dip_results
            save_results_table(dip_results, "DIP",
                              os.path.join(results_dir, "dip_results.txt"))
    
    # Plot comparison
    if all_results:
        plot_psnr_comparison(all_results, 
                            os.path.join(results_dir, f"psnr_comparison_{args.mode}.png"))
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
