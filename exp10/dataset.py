"""
DIV2K Dataset Loader for Self2Self and DIP Denoising
=====================================================
- Automatically downloads DIV2K High-Resolution images if not present.
- Supports grayscale and color (RGB) modes.
- Training: random 128x128 crop + data augmentation + online noise injection
            + Bernoulli sampling mask for Self2Self.
- Testing: full-resolution images + deterministic noise (seeded).

Key differences from exp9 (Neighbor2Neighbor):
  - Self2Self: Uses Bernoulli sampling mask on input pixels, loss on masked pixels.
  - DIP: Single-image self-supervised learning, no dataset needed (uses fixed random input).
"""

import os
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
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
# Self2Self Dataset
# ============================================================
class Self2SelfDataset(Dataset):
    """
    DIV2K dataset for Self2Self self-supervised denoising.

    During training:
      - Randomly crop a (patch_size x patch_size) patch
      - Apply random horizontal flip and 90 degree rotation
      - Add single Gaussian noise realization (online)
      - Generate Bernoulli sampling mask: randomly select pixels
        - Input: masked pixels replaced with zeros (or neighbors)
        - Target: original noisy values at all positions
        - Loss: computed only on masked positions

    During validation/testing:
      - Use full-resolution images
      - Add noise with a fixed random seed for reproducibility
      - No masking (standard forward pass for evaluation)

    Returns:
      Training:   (masked_input, noisy_target, mask)
      Validation: (noisy, clean)
    """

    IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def __init__(
        self,
        img_dir: str,
        sigma: float = 25.0,
        mode: str = "gray",        # "gray" or "color"
        train: bool = True,
        patch_size: int = 128,
        mask_ratio: float = 0.3,   # fraction of pixels to mask (Self2Self)
    ):
        """
        Args:
            img_dir:      Path to folder containing HR images.
            sigma:        Noise standard deviation (in [0, 255] scale).
            mode:         "gray" for single channel, "color" for RGB.
            train:        If True, apply augmentation + random crop + masking.
            patch_size:   Crop size for training patches.
            mask_ratio:   Fraction of pixels masked for Self2Self (training only).
        """
        super().__init__()
        self.sigma = sigma / 255.0   # store in [0,1] scale
        self.mode = mode
        self.train = train
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

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
        else:
            img_np = self._center_crop(img_np, 512)

        # Convert to tensor: (C, H, W)
        if img_np.ndim == 2:
            clean = torch.from_numpy(img_np.copy()).unsqueeze(0)   # (1, H, W)
        else:
            clean = torch.from_numpy(img_np.transpose(2, 0, 1).copy())  # (3, H, W)

        # Add single Gaussian noise realization
        noise = torch.randn_like(clean) * self.sigma
        noisy = clean + noise

        if self.train:
            # Generate Bernoulli mask
            mask = torch.rand_like(noisy) < self.mask_ratio
            mask = mask.float()
            
            # Create masked input: replace masked pixels with zeros
            masked_input = noisy * (1 - mask)
            
            return masked_input, noisy, mask
        else:
            return noisy, clean

    def _random_crop(self, img: np.ndarray) -> np.ndarray:
        """Randomly crop a patch_size x patch_size region."""
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
            pil = pil.resize((new_w, new_h), Image.Resampling.BICUBIC)
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

    def _center_crop(self, img: np.ndarray, size: int) -> np.ndarray:
        """Center crop to size x size."""
        if img.ndim == 2:
            h, w = img.shape
            top = (h - size) // 2
            left = (w - size) // 2
            return img[top:top + size, left:left + size]
        else:
            h, w, _ = img.shape
            top = (h - size) // 2
            left = (w - size) // 2
            return img[top:top + size, left:left + size, :]

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Random horizontal flip and 90 degree rotation."""
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1).copy()
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=0).copy()
        k = np.random.randint(0, 4)
        if img.ndim == 2:
            img = np.rot90(img, k)
        else:
            img = np.rot90(img, k, axes=(0, 1))
        return img


# ============================================================
# Single Image Dataset for DIP
# ============================================================
class SingleImageDIPDataset(Dataset):
    """
    Single-image dataset for Deep Image Prior denoising.
    
    DIP does not use a training dataset. Instead, it:
      - Takes a single noisy image
      - Uses a fixed random input vector
      - Optimizes the network to reconstruct the noisy image
    
    This dataset simply returns the single noisy image for training.
    """

    def __init__(
        self,
        image_path: str,
        sigma: float = 25.0,
        mode: str = "gray",
    ):
        """
        Args:
            image_path: Path to the single image (or directory for batch).
            sigma:      Noise standard deviation (in [0, 255] scale).
            mode:       "gray" for single channel, "color" for RGB.
        """
        super().__init__()
        self.sigma = sigma / 255.0
        self.mode = mode
        
        # Load image
        if os.path.isdir(image_path):
            # If directory, use first image
            img_files = [f for f in os.listdir(image_path) 
                         if os.path.splitext(f)[1].lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}]
            if img_files:
                image_path = os.path.join(image_path, sorted(img_files)[0])
            else:
                raise ValueError(f"No images found in {image_path}")
        
        img = Image.open(image_path).convert("RGB")
        if self.mode == "gray":
            img = img.convert("L")
        
        img_np = np.array(img, dtype=np.float32) / 255.0
        
        # Convert to tensor
        if img_np.ndim == 2:
            self.clean = torch.from_numpy(img_np).unsqueeze(0)
        else:
            self.clean = torch.from_numpy(img_np.transpose(2, 0, 1))
        
        # Add noise
        noise = torch.randn_like(self.clean) * self.sigma
        self.noisy = self.clean + noise

    def __len__(self) -> int:
        return 1  # Single image

    def __getitem__(self, idx: int):
        return self.noisy, self.clean
