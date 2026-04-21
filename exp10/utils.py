"""
Utility Functions for Self2Self and DIP Denoising Experiment
=============================================================
- PSNR / SSIM computation
- Gaussian noise injection
- Tensor to numpy conversion
- Visual comparison figure generation
- Training curve plotting
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


# ============================================================
# Metrics
# ============================================================
def compute_psnr(clean: np.ndarray, denoised: np.ndarray, data_range: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio (dB)."""
    return float(_psnr(clean, denoised, data_range=data_range))


def compute_ssim(clean: np.ndarray, denoised: np.ndarray, data_range: float = 1.0) -> float:
    """Compute Structural Similarity Index."""
    # Handle grayscale (H,W) and color (H,W,C)
    if clean.ndim == 3 and clean.shape[2] == 3:
        return float(_ssim(clean, denoised, data_range=data_range, channel_axis=2))
    return float(_ssim(clean, denoised, data_range=data_range))


# ============================================================
# Noise helpers
# ============================================================
def add_gaussian_noise(image: np.ndarray, sigma: float, seed: int = None) -> np.ndarray:
    """
    Add Gaussian noise to an image.

    Args:
        image:  Clean image in [0, 1] range.
        sigma:  Noise standard deviation in [0, 255] scale.
        seed:   Optional random seed for reproducibility.
    Returns:
        Noisy image clipped to [0, 1].
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    noise = rng.randn(*image.shape).astype(np.float32) * (sigma / 255.0)
    return np.clip(image + noise, 0, 1).astype(np.float32)


# ============================================================
# Tensor <-> numpy helpers
# ============================================================
def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert (B,C,H,W) or (C,H,W) tensor -> (H,W) or (H,W,C) numpy in [0,1]."""
    if t.dim() == 4:
        t = t.squeeze(0)
    t = t.detach().cpu().clamp(0, 1)
    arr = t.numpy()
    if arr.shape[0] == 1:
        return arr[0]          # (H, W)
    return arr.transpose(1, 2, 0)  # (H, W, C)


# ============================================================
# Visualization
# ============================================================
def save_comparison_figure(
    images: list,
    titles: list,
    save_path: str,
    suptitle: str = "",
    cmap: str = "gray",
):
    """
    Save a side-by-side comparison figure.

    Args:
        images: List of numpy arrays (H,W) or (H,W,3).
        titles: List of title strings for each subplot.
        save_path: Path to save the figure.
        suptitle: Optional super-title for the figure.
        cmap: Colormap (used only for grayscale images).
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {save_path}")


def plot_training_curves(
    train_losses: list,
    val_psnrs: list,
    save_path: str,
    method_name: str = "Method",
):
    """Plot and save training loss + validation PSNR curves."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, train_losses, "b-", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss (MSE)")
    ax1.set_title(f"{method_name} Training Loss Curve", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    if val_psnrs:
        ax2.plot(epochs, val_psnrs, "r-", linewidth=1.5)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation PSNR (dB)")
        ax2.set_title(f"{method_name} Validation PSNR Curve", fontweight="bold")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {save_path}")


def plot_combined_curves(
    results: dict,
    save_path: str,
    title: str = "Training Curves Comparison",
):
    """
    Plot training curves for multiple methods on the same figure.
    
    Args:
        results: Dict of {method_name: (losses, psnrs)}
        save_path: Path to save the figure.
        title: Figure title.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['b', 'r', 'g', 'm', 'c']
    
    for idx, (name, (losses, psnrs)) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        epochs = range(1, len(losses) + 1)
        ax1.plot(epochs, losses, f'{color}-', linewidth=1.5, label=name)
        if psnrs:
            ax2.plot(epochs, psnrs, f'{color}-', linewidth=1.5, label=name)
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("PSNR Comparison", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {save_path}")
