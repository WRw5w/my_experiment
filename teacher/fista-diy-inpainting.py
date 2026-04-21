import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, restoration, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter
import bm3d

# ============ 自定义近端算子 ============

def prox_tv_custom(x, lambd, n_iter=50, step_size=0.1):
    """
    自定义 TV 近端算子 - 使用梯度下降法
    求解: argmin_u { (1/2)||u - x||^2 + lambd * TV(u) }
    """
    u = x.copy()
    eps = 1e-8  # 避免除零

    for _ in range(n_iter):
        # 计算 x 方向梯度
        grad_x = np.zeros_like(u)
        grad_x[:, :-1] = u[:, 1:] - u[:, :-1]

        # 计算 y 方向梯度
        grad_y = np.zeros_like(u)
        grad_y[:-1, :] = u[1:, :] - u[:-1, :]

        # 计算梯度的模
        grad_norm = np.sqrt(grad_x**2 + grad_y**2 + eps)

        # 计算 TV 项的梯度
        # div(grad(u) / |grad(u)|)
        tv_grad_x = grad_x / grad_norm
        tv_grad_y = grad_y / grad_norm

        # 计算散度
        div_tv = np.zeros_like(u)

        # x 方向散度
        div_tv[:, 1:] += tv_grad_x[:, 1:] - tv_grad_x[:, :-1]
        div_tv[:, 0] += tv_grad_x[:, 0]
        div_tv[:, -1] -= tv_grad_x[:, -2]

        # y 方向散度
        div_tv[1:, :] += tv_grad_y[1:, :] - tv_grad_y[:-1, :]
        div_tv[0, :] += tv_grad_y[0, :]
        div_tv[-1, :] -= tv_grad_y[-2, :]

        # 梯度下降更新
        # grad = (u - x) - lambd * div(grad(u)/|grad(u)|)
        gradient = (u - x) - lambd * div_tv
        u = u - step_size * gradient

    return u

def prox_tv(x, lambd):
    """TV 近端算子 - 调用自定义实现"""
    return prox_tv_custom(x, lambd, n_iter=100)

def prox_tv_library(x, lambd):
    """TV 近端算子 - 使用库函数实现"""
    return restoration.denoise_tv_chambolle(x, weight=lambd)

def prox_l1_soft_threshold(x, lambd):
    """L1 近端算子 - 软阈值 (Soft Thresholding)"""
    return np.sign(x) * np.maximum(np.abs(x) - lambd, 0)

def prox_gaussian(x, lambd):
    """高斯平滑近端算子"""
    sigma = lambd * 10  # 将 lambd 转换为合适的 sigma 值
    return gaussian_filter(x, sigma=sigma)

def prox_bilateral(x, lambd):
    """双边滤波近端算子 (保边去噪)"""
    from skimage.restoration import denoise_bilateral
    sigma_spatial = lambd * 5
    return denoise_bilateral(x, sigma_color=0.05, sigma_spatial=sigma_spatial, channel_axis=None)

def prox_wavelet(x, lambd):
    """小波软阈值近端算子"""
    from skimage.restoration import denoise_wavelet
    return denoise_wavelet(x, sigma=lambd, mode='soft', rescale_sigma=True)

def prox_bm3d(x, lambd):
    """BM3D 近端算子 - 使用 BM3D 库"""
    # BM3D 需要噪声标准差参数
    sigma_psd = lambd * 5  # 将 lambd 转换为噪声标准差
    return bm3d.bm3d(x, sigma_psd=sigma_psd)

# ==========================================

def fista_tv_inpainting(distorted_img, mask, original_img, lambd=0.01, num_iter=100, prox_operator=prox_tv):
    """FISTA 算法实现图像修复"""
    x = distorted_img.copy()
    y = x.copy()
    t = 1.0

    # 记录每次迭代的 PSNR 和 SSIM
    psnr_history = []
    ssim_history = []

    for i in range(num_iter):
        x_old = x.copy()

        # 1. 梯度下降步: 仅在 mask == 1 的地方计算残差
        # grad = M * (y - original)，但在修复中我们只需更新已知像素
        res = y - mask * (y - distorted_img)

        # 2. 自定义近端算子 (Proximal Operator)
        x = prox_operator(res, lambd)

        # 3. FISTA 加速更新
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next

        # 计算当前迭代的 PSNR 和 SSIM
        x_clipped = np.clip(x, 0, 1)
        psnr_val = psnr(original_img, x_clipped, data_range=1.0)
        ssim_val = ssim(original_img, x_clipped, data_range=1.0)
        psnr_history.append(psnr_val)
        ssim_history.append(ssim_val)

        if (i + 1) % 20 == 0:
            print(f"迭代次数: {i+1}/{num_iter}, PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")

    return np.clip(x, 0, 1), psnr_history, ssim_history

def ista_tv_inpainting(distorted_img, mask, original_img, lambd=0.01, num_iter=100, prox_operator=prox_tv):
    """ISTA 算法实现图像修复（无加速版本）"""
    x = distorted_img.copy()

    # 记录每次迭代的 PSNR 和 SSIM
    psnr_history = []
    ssim_history = []

    for i in range(num_iter):
        # 1. 梯度下降步
        res = x - mask * (x - distorted_img)

        # 2. 近端算子
        x = prox_operator(res, lambd)

        # 计算当前迭代的 PSNR 和 SSIM
        x_clipped = np.clip(x, 0, 1)
        psnr_val = psnr(original_img, x_clipped, data_range=1.0)
        ssim_val = ssim(original_img, x_clipped, data_range=1.0)
        psnr_history.append(psnr_val)
        ssim_history.append(ssim_val)

        if (i + 1) % 20 == 0:
            print(f"迭代次数: {i+1}/{num_iter}, PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")

    return np.clip(x, 0, 1), psnr_history, ssim_history

# 1. 加载内置 Demo 图像 (摄影师)
image = img_as_float(data.camera())

# 2. 手动创建一个复杂的掩码 (Mask)
# 包含随机丢失的像素 + 模拟文字划痕
mask = np.ones_like(image)
# 随机丢失 40% 的像素
rand_mask = np.random.choice([0, 1], size=image.shape, p=[0.4, 0.6])
mask *= rand_mask
# 添加几条粗划痕 (模拟划伤或文字)
mask[100:110, 50:200] = 0
mask[200:350, 400:410] = 0
mask[400:410, 100:450] = 0

# 制作受损图像
distorted = image * mask

# 3. 运行修复算法 - 对比自定义 TV 和库函数 TV
# lambd: 越大图像越平滑，越小保留细节越多但修复能力变弱

print("=" * 60)
print("运行自定义 TV 近端算子...")
print("=" * 60)

reconstructed_custom, psnr_history_custom, ssim_history_custom = fista_tv_inpainting(
    distorted, mask, image,
    lambd=0.04,
    num_iter=60,
    prox_operator=prox_tv
)

print("\n" + "=" * 60)
print("运行库函数 TV 近端算子...")
print("=" * 60)

reconstructed_library, psnr_history_library, ssim_history_library = fista_tv_inpainting(
    distorted, mask, image,
    lambd=0.04,
    num_iter=60,
    prox_operator=prox_tv_library
)

print("\n" + "=" * 60)
print("运行 ISTA + TV 算法...")
print("=" * 60)

reconstructed_ista, psnr_history_ista, ssim_history_ista = ista_tv_inpainting(
    distorted, mask, image,
    lambd=0.04,
    num_iter=60,
    prox_operator=prox_tv_library
)

print("\n" + "=" * 60)
print("运行 FISTA + BM3D 算法...")
print("=" * 60)

reconstructed_bm3d, psnr_history_bm3d, ssim_history_bm3d = fista_tv_inpainting(
    distorted, mask, image,
    lambd=0.04,
    num_iter=60,
    prox_operator=prox_bm3d
)

# 计算损坏图像和修复图像的 PSNR 和 SSIM
distorted_psnr = psnr(image, distorted, data_range=1.0)
distorted_ssim = ssim(image, distorted, data_range=1.0)

reconstructed_custom_psnr = psnr(image, reconstructed_custom, data_range=1.0)
reconstructed_custom_ssim = ssim(image, reconstructed_custom, data_range=1.0)

reconstructed_library_psnr = psnr(image, reconstructed_library, data_range=1.0)
reconstructed_library_ssim = ssim(image, reconstructed_library, data_range=1.0)

reconstructed_ista_psnr = psnr(image, reconstructed_ista, data_range=1.0)
reconstructed_ista_ssim = ssim(image, reconstructed_ista, data_range=1.0)

reconstructed_bm3d_psnr = psnr(image, reconstructed_bm3d, data_range=1.0)
reconstructed_bm3d_ssim = ssim(image, reconstructed_bm3d, data_range=1.0)

print("\n" + "=" * 60)
print("对比结果:")
print("=" * 60)
print(f"损坏图像      - PSNR: {distorted_psnr:.2f} dB, SSIM: {distorted_ssim:.4f}")
print(f"FISTA+自定义TV - PSNR: {reconstructed_custom_psnr:.2f} dB, SSIM: {reconstructed_custom_ssim:.4f}")
print(f"FISTA+库函数TV - PSNR: {reconstructed_library_psnr:.2f} dB, SSIM: {reconstructed_library_ssim:.4f}")
print(f"ISTA+TV       - PSNR: {reconstructed_ista_psnr:.2f} dB, SSIM: {reconstructed_ista_ssim:.4f}")
print(f"FISTA+BM3D     - PSNR: {reconstructed_bm3d_psnr:.2f} dB, SSIM: {reconstructed_bm3d_ssim:.4f}")
print("=" * 60)

# 4. 结果对比展示
fig = plt.figure(figsize=(18, 14))

# 创建子图布局：左边3行2列显示6张图像，右边2张曲线图
ax1 = plt.subplot(3, 4, 1)
ax2 = plt.subplot(3, 4, 2)
ax3 = plt.subplot(3, 4, 5)
ax4 = plt.subplot(3, 4, 6)
ax5 = plt.subplot(3, 4, 9)
ax6 = plt.subplot(3, 4, 10)
ax_psnr = plt.subplot(3, 2, 2)
ax_ssim = plt.subplot(3, 2, 4)

# 显示图像
ax1.imshow(image, cmap='gray')
ax1.set_title("1. Original", fontsize=10)
ax1.axis('off')

ax2.imshow(distorted, cmap='gray')
ax2.set_title(f"2. Corrupted\nPSNR: {distorted_psnr:.2f}\nSSIM: {distorted_ssim:.4f}", fontsize=10)
ax2.axis('off')

ax3.imshow(reconstructed_custom, cmap='gray')
ax3.set_title(f"3. FISTA+Custom TV\nPSNR: {reconstructed_custom_psnr:.2f}\nSSIM: {reconstructed_custom_ssim:.4f}", fontsize=10)
ax3.axis('off')

ax4.imshow(reconstructed_library, cmap='gray')
ax4.set_title(f"4. FISTA+Lib TV\nPSNR: {reconstructed_library_psnr:.2f}\nSSIM: {reconstructed_library_ssim:.4f}", fontsize=10)
ax4.axis('off')

ax5.imshow(reconstructed_ista, cmap='gray')
ax5.set_title(f"5. ISTA+TV\nPSNR: {reconstructed_ista_psnr:.2f}\nSSIM: {reconstructed_ista_ssim:.4f}", fontsize=10)
ax5.axis('off')

ax6.imshow(reconstructed_bm3d, cmap='gray')
ax6.set_title(f"6. FISTA+BM3D\nPSNR: {reconstructed_bm3d_psnr:.2f}\nSSIM: {reconstructed_bm3d_ssim:.4f}", fontsize=10)
ax6.axis('off')

# 绘制 PSNR 对比曲线
iterations = range(1, len(psnr_history_custom) + 1)
ax_psnr.plot(iterations, psnr_history_custom, 'b-', linewidth=1.5, label='FISTA+Custom TV')
ax_psnr.plot(iterations, psnr_history_library, 'r-', linewidth=1.5, label='FISTA+Lib TV')
ax_psnr.plot(iterations, psnr_history_ista, 'g--', linewidth=1.5, label='ISTA+TV')
ax_psnr.plot(iterations, psnr_history_bm3d, 'm-.', linewidth=1.5, label='FISTA+BM3D')
ax_psnr.set_xlabel('Iteration', fontsize=10)
ax_psnr.set_ylabel('PSNR (dB)', fontsize=10)
ax_psnr.set_title('PSNR Comparison', fontsize=11)
ax_psnr.legend(fontsize=8)
ax_psnr.grid(True, alpha=0.3)

# 绘制 SSIM 对比曲线
ax_ssim.plot(iterations, ssim_history_custom, 'b-', linewidth=1.5, label='FISTA+Custom TV')
ax_ssim.plot(iterations, ssim_history_library, 'r-', linewidth=1.5, label='FISTA+Lib TV')
ax_ssim.plot(iterations, ssim_history_ista, 'g--', linewidth=1.5, label='ISTA+TV')
ax_ssim.plot(iterations, ssim_history_bm3d, 'm-.', linewidth=1.5, label='FISTA+BM3D')
ax_ssim.set_xlabel('Iteration', fontsize=10)
ax_ssim.set_ylabel('SSIM', fontsize=10)
ax_ssim.set_title('SSIM Comparison', fontsize=11)
ax_ssim.legend(fontsize=8)
ax_ssim.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()