"""
Plot Combined Results for Self2Self and DIP
=============================================
Generate comprehensive visualizations for the experiment report:
  1. PSNR comparison across methods and noise levels
  2. Visual denoising comparison (Clean, Noisy, BM3D, N2N, Self2Self, DIP)
  3. Training curves comparison
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_psnr_vs_sigma(results: dict, save_path: str, mode: str = "gray"):
    """
    Plot PSNR vs noise level (sigma) for multiple methods.
    
    Args:
        results: Dict of {method_name: {sigma: avg_psnr}}
        save_path: Path to save the figure.
        mode: "gray" or "color"
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = {'BM3D': '#1f77b4', 'N2N': '#2ca02c', 'Self2Self': '#d62728', 'DIP': '#9467bd'}
    markers = {'BM3D': 'o', 'N2N': 's', 'Self2Self': '^', 'DIP': 'D'}
    
    for method, sigma_psnr in results.items():
        sigmas = sorted(sigma_psnr.keys())
        psnrs = [sigma_psnr[s] for s in sigmas]
        color = colors.get(method, 'k')
        marker = markers.get(method, 'o')
        ax.plot(sigmas, psnrs, color=color, marker=marker, linewidth=2, 
                markersize=8, label=method)
    
    ax.set_xlabel("Noise Level (σ)", fontsize=13)
    ax.set_ylabel("Average PSNR (dB)", fontsize=13)
    ax.set_title(f"Denoising Performance Comparison ({mode.upper()})", 
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([15, 25, 35, 50])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_visual_comparison(images: dict, titles: list, save_path: str, 
                           mode: str = "gray", sigma: int = 25):
    """
    Plot visual denoising comparison.
    
    Args:
        images: Dict of {method_name: numpy_array}
        titles: List of method names in order
        save_path: Path to save the figure.
        mode: "gray" or "color"
        sigma: Noise level for title
    """
    n = len(titles)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    
    for idx, (method, ax) in enumerate(zip(titles, axes)):
        img = images.get(method, None)
        if img is not None:
            if img.ndim == 2:
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(np.clip(img, 0, 1))
        ax.set_title(method, fontsize=12, fontweight="bold")
        ax.axis("off")
    
    plt.suptitle(f"Visual Comparison (σ={sigma})", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_training_curves_comparison(train_data: dict, save_path: str):
    """
    Plot training curves for multiple methods.
    
    Args:
        train_data: Dict of {method_name: {"loss": [...], "psnr": [...]}}
        save_path: Path to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'Self2Self': '#d62728', 'DIP': '#9467bd', 'N2N': '#2ca02c'}
    
    for method, data in train_data.items():
        color = colors.get(method, 'k')
        epochs = range(1, len(data["loss"]) + 1)
        ax1.plot(epochs, data["loss"], f'{color}-', linewidth=2, label=method)
        if data.get("psnr"):
            ax2.plot(epochs, data["psnr"], f'{color}-', linewidth=2, label=method)
    
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Training Loss", fontsize=12)
    ax1.set_title("Training Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("PSNR (dB)", fontsize=12)
    ax2.set_title("Validation PSNR", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def generate_results_table(results: dict, save_path: str):
    """Generate a formatted results table."""
    with open(save_path, 'w') as f:
        f.write("Denoising Performance Results\n")
        f.write("=" * 80 + "\n\n")
        
        # Get all methods and sigmas
        methods = list(results.keys())
        sigmas = sorted(list(list(results.values())[0].keys()))
        
        # Header
        header = f"{'Method':<15}"
        for sigma in sigmas:
            header += f"σ={sigma:<10}"
        header += f"{'Avg':<10}"
        f.write(header + "\n")
        f.write("-" * 80 + "\n")
        
        # Data rows
        for method in methods:
            row = f"{method:<15}"
            psnr_sum = 0
            count = 0
            for sigma in sigmas:
                psnr = results[method].get(sigma, 0)
                row += f"{psnr:<12.2f}"
                psnr_sum += psnr
                count += 1
            avg = psnr_sum / max(count, 1)
            row += f"{avg:<10.2f}"
            f.write(row + "\n")
    
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    # Example usage
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Example data (replace with actual results)
    example_results = {
        "BM3D": {15: 32.0, 25: 29.0, 35: 27.0, 50: 25.0},
        "N2N": {15: 31.5, 25: 28.5, 35: 26.5, 50: 24.5},
        "Self2Self": {15: 30.0, 25: 27.5, 35: 25.5, 50: 23.5},
        "DIP": {15: 29.0, 25: 26.5, 35: 24.5, 50: 22.5},
    }
    
    plot_psnr_vs_sigma(example_results, 
                       os.path.join(results_dir, "psnr_vs_sigma_gray.png"),
                       mode="gray")
    generate_results_table(example_results,
                          os.path.join(results_dir, "results_table_gray.txt"))
