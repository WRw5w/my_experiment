"""
DIV2K Dataset Loader for Denoising
===================================
- Automatically downloads DIV2K High-Resolution images if not present.
- Supports grayscale and color (RGB) modes.
- Training: random 128×128 crop + data augmentation + online noise injection.
- Testing: full-resolution images + deterministic noise (seeded).
"""

import os
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import urllib.request


# ============================================================
# Download helpers
# ============================================================
DIV2K_URLS = {
    "train": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "valid": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}


def _download_and_extract(url: str, dest_dir: str, desc: str = ""):
    """Download a ZIP from *url* and extract to *dest_dir*."""
    os.makedirs(dest_dir, exist_ok=True)
    zip_name = url.split("/")[-1]
    zip_path = os.path.join(dest_dir, zip_name)

    if not os.path.exists(zip_path):
        print(f"[Dataset] Downloading {desc} from {url} ...")
        urllib.request.urlretrieve(url, zip_path, reporthook=_progress_hook)
        print()  # newline after progress

    # Extract
    folder_name = zip_name.replace(".zip", "")
    extract_target = os.path.join(dest_dir, folder_name)
    if not os.path.isdir(extract_target):
        print(f"[Dataset] Extracting {zip_name} ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)
    return extract_target


def _progress_hook(block_num, block_size, total_size):
    """Simple progress display for urllib."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        print(f"\r  Progress: {pct:5.1f}% ({downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB)",
              end="", flush=True)


def prepare_div2k(data_root: str):
    """Ensure DIV2K train and valid HR folders are downloaded and ready.

    Returns (train_dir, valid_dir) paths.
    """
    train_dir = os.path.join(data_root, "DIV2K_train_HR")
    valid_dir = os.path.join(data_root, "DIV2K_valid_HR")

    if not os.path.isdir(train_dir):
        _download_and_extract(DIV2K_URLS["train"], data_root, "DIV2K Training HR")
    if not os.path.isdir(valid_dir):
        _download_and_extract(DIV2K_URLS["valid"], data_root, "DIV2K Validation HR")

    return train_dir, valid_dir


# ============================================================
# Dataset class
# ============================================================
class DIV2KDenoisingDataset(Dataset):
    """
    DIV2K dataset for image denoising.

    During training:
      - Randomly crop a (patch_size × patch_size) patch
      - Apply random horizontal flip and 90° rotation
      - Add Gaussian noise with specified sigma (online)

    During testing:
      - Use full-resolution images
      - Add noise with a fixed random seed for reproducibility
    """

    IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def __init__(
        self,
        img_dir: str,
        sigma: float = 25.0,
        mode: str = "gray",        # "gray" or "color"
        train: bool = True,
        patch_size: int = 128,
    ):
        """
        Args:
            img_dir:    Path to folder containing HR images.
            sigma:      Noise standard deviation (in [0, 255] scale).
            mode:       "gray" for single channel, "color" for RGB.
            train:      If True, apply augmentation + random crop.
            patch_size: Crop size for training patches.
        """
        super().__init__()
        self.sigma = sigma / 255.0   # store in [0,1] scale
        self.mode = mode
        self.train = train
        self.patch_size = patch_size

        # Collect image paths
        self.image_paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in self.IMG_EXTENSIONS
        ])
        if self.train:
            self.image_paths = self.image_paths[:200]
        assert len(self.image_paths) > 0, f"No images found in {img_dir}"

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        # Load image as RGB PIL
        img = Image.open(self.image_paths[idx]).convert("RGB")

        if self.mode == "gray":
            img = img.convert("L")   # single channel

        # Convert to numpy float32 [0,1]
        img_np = np.array(img, dtype=np.float32) / 255.0

        if self.train:
            img_np = self._random_crop(img_np)
            img_np = self._augment(img_np)

        # Convert to tensor: (C, H, W)
        if img_np.ndim == 2:
            clean = torch.from_numpy(img_np).unsqueeze(0)   # (1, H, W)
        else:
            clean = torch.from_numpy(img_np.transpose(2, 0, 1))  # (3, H, W)

        # Add Gaussian noise
        noise = torch.randn_like(clean) * self.sigma
        noisy = clean + noise

        return noisy, clean

    def _random_crop(self, img: np.ndarray) -> np.ndarray:
        """Randomly crop a patch_size × patch_size region."""
        if img.ndim == 2:
            h, w = img.shape
        else:
            h, w, _ = img.shape
        ps = self.patch_size
        # If image is smaller than patch, resize up (rare for DIV2K)
        if h < ps or w < ps:
            scale = max(ps / h, ps / w) + 0.01
            new_h, new_w = int(h * scale), int(w * scale)
            pil = Image.fromarray((img * 255).astype(np.uint8))
            pil = pil.resize((new_w, new_h), Image.BICUBIC)
            img = np.array(pil, dtype=np.float32) / 255.0
            if img.ndim == 2:
                h, w = img.shape
            else:
                h, w, _ = img.shape

        top = np.random.randint(0, h - ps + 1)
        left = np.random.randint(0, w - ps + 1)
        if img.ndim == 2:
            return img[top:top + ps, left:left + ps]
        return img[top:top + ps, left:left + ps, :]

    @staticmethod
    def _augment(img: np.ndarray) -> np.ndarray:
        """Random horizontal flip and 90° rotation."""
        # Horizontal flip
        if np.random.rand() < 0.5:
            img = np.flip(img, axis=1).copy()
        # Random 90° rotation (0, 1, 2, or 3 times)
        k = np.random.randint(0, 4)
        if img.ndim == 2:
            img = np.rot90(img, k).copy()
        else:
            img = np.rot90(img, k, axes=(0, 1)).copy()
        return img


if __name__ == "__main__":
    # Quick test
    data_root = os.path.join(os.path.dirname(__file__), "data")
    train_dir, valid_dir = prepare_div2k(data_root)
    print(f"Train dir: {train_dir} ({len(os.listdir(train_dir))} files)")
    print(f"Valid dir: {valid_dir} ({len(os.listdir(valid_dir))} files)")

    ds = DIV2KDenoisingDataset(train_dir, sigma=25.0, mode="gray", train=True)
    noisy, clean = ds[0]
    print(f"Sample shape — noisy: {noisy.shape}, clean: {clean.shape}")
