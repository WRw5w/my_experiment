"""
Model Architectures for Self2Self and DIP Image Denoising (Fixed & Optimized)
=============================================================================
1. Self2SelfUNet: 
   - Fixed: Directly outputs image (no residual x - noise) because input x is masked.
   - Tuned: Dropout rate optimized for Self2Self (0.3).
   
2. DIP_UNet (Deep Image Prior):
   - Fixed: Changed from a blind Upsample-Decoder to the standard DIP Hourglass (U-Net).
   - Input noise 'z' now matches the spatial resolution of the target image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ============================================================
# 1. Self2Self U-Net (Fixed)
# ============================================================
class DoubleConvDropout(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Self2SelfUNet(nn.Module):
    """
    Self2Self Network:
    - Must directly output the predicted image (NOT residual) because training inputs are masked.
    - Uses Dropout for Monte Carlo ensemble inference.
    """
    def __init__(self, in_channels: int = 1, base_features: int = 48, dropout_rate: float = 0.3):
        super().__init__()
        f = base_features

        # Encoder path
        self.enc1 = DoubleConvDropout(in_channels, f, dropout_rate)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConvDropout(f, f * 2, dropout_rate)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConvDropout(f * 2, f * 4, dropout_rate)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConvDropout(f * 4, f * 8, dropout_rate)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConvDropout(f * 8, f * 8, dropout_rate)

        # Decoder path
        self.up4 = nn.ConvTranspose2d(f * 8, f * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConvDropout(f * 16, f * 4, dropout_rate)
        
        self.up3 = nn.ConvTranspose2d(f * 4, f * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConvDropout(f * 8, f * 2, dropout_rate)
        
        self.up2 = nn.ConvTranspose2d(f * 2, f * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConvDropout(f * 4, f, dropout_rate)
        
        self.up1 = nn.ConvTranspose2d(f, f, kernel_size=2, stride=2)
        self.dec1 = DoubleConvDropout(f * 2, f, dropout_rate)

        # Final Convolution
        self.out_conv = nn.Conv2d(f, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with skip connections (handling odd dimensions automatically)
        d4 = self.up4(b)
        d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([e4, d4], dim=1))

        d3 = self.up3(d4)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([e3, d3], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([e2, d2], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([e1, d1], dim=1))

        # DIRECT PREDICTION: output the image directly in [0, 1]
        out = self.out_conv(d1)
        return torch.sigmoid(out)

    def ensemble_denoise(self, x: torch.Tensor, num_samples: int = 50) -> torch.Tensor:
        self.train()  # Force Dropout ON
        outputs =[]
        with torch.no_grad():
            for _ in range(num_samples):
                outputs.append(self(x))
        self.eval()
        return torch.stack(outputs).mean(dim=0)


# ============================================================
# 2. Deep Image Prior (DIP) U-Net (Standard Hourglass)
# ============================================================
class DIPBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DIP_UNet(nn.Module):
    """
    Standard Deep Image Prior (Hourglass/U-Net) Architecture.
    
    Why it's better:
    - Input 'z' has the SAME spatial size as the output image.
    - Uses downsampling & upsampling to capture multi-scale priors.
    - Preserves exact local structural features instead of destroying them via late F.interpolate.
    """
    def __init__(self, in_channels: int = 32, out_channels: int = 1, channels: list =[128, 128, 128, 128, 128]):
        super().__init__()
        
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        # Build Encoder
        prev_c = in_channels
        for c in channels:
            self.down_blocks.append(DIPBlock(prev_c, c))
            self.skip_convs.append(nn.Conv2d(c, 4, kernel_size=1)) # Skip connections dimensionality reduction
            prev_c = c
            
        # Build Decoder
        prev_c = channels[-1]
        for c in reversed(channels[:-1]):
            # input to up_block is: prev_c + 4 (from skip)
            self.up_blocks.append(DIPBlock(prev_c + 4, c))
            prev_c = c
            
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Forward pass is normal U-Net
        skips =[]
        x = z
        
        # Down
        for i, down_block in enumerate(self.down_blocks):
            x = down_block(x)
            skips.append(self.skip_convs[i](x))
            if i < len(self.down_blocks) - 1: # Don't pool the bottleneck
                x = F.max_pool2d(x, 2)
                
        # Up
        for i, up_block in enumerate(self.up_blocks):
            skip = skips[-(i+2)]
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = up_block(x)
            
        return self.final_conv(x)

    @staticmethod
    def get_noise(batch_size: int, in_channels: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """
        DIP requires noise of the SAME spatial dimension as the target image.
        Uses Uniform noise [0, 0.1] as standard practice in DIP.
        """
        # torch.rand gives [0, 1]. Multiply by 0.1 as per standard DIP implementation.
        return torch.rand((batch_size, in_channels, height, width), device=device) * 0.1


# ============================================================
# 3. DIPDecoder - Wrapper for DIP_UNet with compatible interface
# ============================================================
class DIPDecoder(nn.Module):
    """
    DIPDecoder wrapper that provides a compatible interface for train.py.
    Uses DIP_UNet internally but with generate_random_input method.
    """
    def __init__(self, in_channels: int = 32, out_channels: int = 1, num_upsample: int = 5):
        super().__init__()
        # num_upsample determines the depth of the network
        # Each upsample level doubles the resolution
        channels = [128] * (num_upsample + 1)  # +1 for bottleneck
        self.unet = DIP_UNet(in_channels=in_channels, out_channels=out_channels, channels=channels)
        self.in_channels = in_channels
        
    def forward(self, z: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Forward pass. target_size is ignored since DIP_UNet requires
        input noise to already match the target spatial dimensions.
        """
        return self.unet(z)
    
    def generate_random_input(self, batch_size: int, device: torch.device,
                               target_height: Optional[int] = None,
                               target_width: Optional[int] = None) -> torch.Tensor:
        """
        Generate random input noise for DIP.
        If target dimensions not specified, uses 256x256 as default.
        """
        if target_height is None:
            target_height = 256
        if target_width is None:
            target_width = 256
        return DIP_UNet.get_noise(batch_size, self.in_channels, target_height, target_width, device)


# ============================================================
# Parameter counting & Test
# ============================================================
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("Testing Fixed Self2SelfUNet...")
    # S2S Input is the masked image itself
    model_s2s = Self2SelfUNet(in_channels=1, dropout_rate=0.3)
    x = torch.randn(1, 1, 128, 128)
    y = model_s2s(x)
    print(f"  Input: {x.shape} -> Output: {y.shape}")
    print(f"  Parameters: {count_parameters(model_s2s):,}")
    
    print("\nTesting Standard DIP_UNet...")
    # DIP Input noise MUST be the same size as target image
    target_H, target_W = 128, 128
    model_dip = DIP_UNet(in_channels=32, out_channels=1)
    
    # Standard DIP noise setup
    z = DIP_UNet.get_noise(1, 32, target_H, target_W, device=torch.device('cpu'))
    y = model_dip(z)
    print(f"  Input noise z: {z.shape} -> Output: {y.shape}")
    print(f"  Parameters: {count_parameters(model_dip):,}")