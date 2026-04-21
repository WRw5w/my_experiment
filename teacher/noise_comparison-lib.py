import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float, metrics
import bm3d
from skimage.restoration import denoise_tv_chambolle

# --- 1. 准备数据 ---
image = img_as_float(data.camera())
sigma = 0.1  # 噪声强度
noisy = image + sigma * np.random.randn(*image.shape)
noisy = np.clip(noisy, 0, 1)

# --- 2. ISTA (基于库函数 TV 正则化) ---
def ista_tv(noisy, lambd, num_iter=50):
    """
    ISTA 算法 - 使用库函数 TV 近端算子

    参数:
        noisy: 噪声图像
        lambd: TV 正则化参数
        num_iter: 迭代次数
    """
    x = noisy.copy()
    step_size = 0.5  # 步长

    for _ in range(num_iter):
        # 梯度步：对于去噪问题，梯度是 (x - noisy)
        grad = x - noisy
        z = x - step_size * grad

        # 近端映射步：使用库函数 TV 去噪
        x = denoise_tv_chambolle(z, weight=lambd * step_size)

    return np.clip(x, 0, 1)

# --- 3. FISTA (基于库函数 TV 正则化) ---
def fista_tv(noisy, lambd, num_iter=50):
    """FISTA 算法 - 使用库函数 TV 近端算子"""
    x = noisy.copy()
    y = x.copy()
    t = 1.0
    step_size = 0.5

    for _ in range(num_iter):
        x_old = x.copy()

        # 梯度步
        grad = y - noisy
        z = y - step_size * grad

        # 近端映射步：使用库函数 TV 去噪
        x = denoise_tv_chambolle(z, weight=lambd * step_size)

        # FISTA 加速步
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next

    return np.clip(x, 0, 1)

# --- 4. BM3D 去噪（使用库函数）---
print("运行 BM3D 去噪...")
denoised_bm3d = bm3d.bm3d(noisy, sigma_psd=sigma)

# --- 运行对比 ---
lambd_tv = 0.1  # TV 权重

print("运行 ISTA...")
res_ista = ista_tv(noisy, lambd_tv, num_iter=30)

print("运行 FISTA...")
res_fista = fista_tv(noisy, lambd_tv, num_iter=30)

# --- 5. 可视化与评价 ---
def get_psnr(im_true, im_test):
    return metrics.peak_signal_noise_ratio(im_true, im_test)

def get_ssim(im_true, im_test):
    return metrics.structural_similarity(im_true, im_test, data_range=1.0)

methods = [
    ("Noisy", noisy),
    ("ISTA (TV-Lib)", res_ista),
    ("FISTA (TV-Lib)", res_fista),
    ("BM3D", denoised_bm3d)
]

print("\n对比结果:")
print("=" * 60)
for name, img in methods:
    psnr_val = get_psnr(image, img)
    ssim_val = get_ssim(image, img)
    print(f"{name:15s} - PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
print("=" * 60)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, (name, img) in zip(axes, methods):
    ax.imshow(img, cmap='gray')
    psnr_val = get_psnr(image, img)
    ssim_val = get_ssim(image, img)
    ax.set_title(f"{name}\nPSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}", fontsize=11)
    ax.axis('off')

plt.tight_layout()
plt.show()
