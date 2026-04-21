import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float, metrics

# --- 1. 准备数据 ---
image = img_as_float(data.camera())
sigma = 0.1  # 噪声强度
noisy = image + sigma * np.random.randn(*image.shape)
noisy = np.clip(noisy, 0, 1)

# --- 自定义 TV 近端算子 ---
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
        gradient = (u - x) - lambd * div_tv
        u = u - step_size * gradient

    return u

# --- 2. ISTA (基于自定义 TV 正则化) ---
def ista_tv(noisy, lambd, num_iter=50):
    """ISTA 算法 - 使用自定义 TV 近端算子"""
    x = noisy.copy()
    step_size = 0.5  # 步长

    for _ in range(num_iter):
        # 梯度步：对于去噪问题，梯度是 (x - noisy)
        grad = x - noisy
        z = x - step_size * grad

        # 近端映射步：TV 去噪
        x = prox_tv_custom(z, lambd * step_size, n_iter=10)

    return np.clip(x, 0, 1)

# --- 3. FISTA (基于自定义 TV 正则化) ---
def fista_tv(noisy, lambd, num_iter=50):
    """FISTA 算法 - 使用自定义 TV 近端算子"""
    x = noisy.copy()
    y = x.copy()
    t = 1.0
    step_size = 0.5

    for _ in range(num_iter):
        x_old = x.copy()

        # 梯度步
        grad = y - noisy
        z = y - step_size * grad

        # 近端映射步
        x = prox_tv_custom(z, lambd * step_size, n_iter=10)

        # FISTA 加速步
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next

    return np.clip(x, 0, 1)

# --- 4. 简化的非局部均值去噪（代替 BM3D）---
def nlm_denoise(noisy, h=0.1, patch_size=7, search_size=21):
    """
    简化的非局部均值去噪算法

    参数:
        noisy: 噪声图像
        h: 滤波参数
        patch_size: 块大小
        search_size: 搜索窗口大小
    """
    height, width = noisy.shape
    denoised = np.zeros_like(noisy)

    pad = patch_size // 2
    search_pad = search_size // 2

    # 填充图像
    padded = np.pad(noisy, pad, mode='reflect')

    # 简化：使用稀疏采样加速
    step = max(1, patch_size // 3)

    for i in range(0, height, step):
        for j in range(0, width, step):
            # 当前像素在填充图像中的位置
            pi, pj = i + pad, j + pad

            # 提取参考块
            ref_patch = padded[pi-pad:pi+pad+1, pj-pad:pj+pad+1]

            # 搜索区域
            si_min = max(pad, pi - search_pad)
            si_max = min(height + pad, pi + search_pad + 1)
            sj_min = max(pad, pj - search_pad)
            sj_max = min(width + pad, pj + search_pad + 1)

            weights_sum = 0
            weighted_sum = 0

            # 在搜索窗口内寻找相似块
            for si in range(si_min, si_max, step):
                for sj in range(sj_min, sj_max, step):
                    # 提取候选块
                    cand_patch = padded[si-pad:si+pad+1, sj-pad:sj+pad+1]

                    # 计算块距离
                    dist = np.sum((ref_patch - cand_patch) ** 2) / (patch_size * patch_size)

                    # 计算权重
                    weight = np.exp(-dist / (h ** 2))

                    weights_sum += weight
                    weighted_sum += weight * padded[si, sj]

            # 归一化
            if weights_sum > 0:
                denoised[i, j] = weighted_sum / weights_sum
            else:
                denoised[i, j] = noisy[i, j]

    # 对未计算的像素进行插值
    if step > 1:
        for i in range(height):
            for j in range(width):
                if (i % step != 0) or (j % step != 0):
                    ni = (i // step) * step
                    nj = (j // step) * step
                    denoised[i, j] = denoised[min(ni, height-1), min(nj, width-1)]

    return np.clip(denoised, 0, 1)

# 运行 NLM 去噪
print("运行 NLM 去噪...")
denoised_nlm = nlm_denoise(noisy, h=sigma*2, patch_size=7, search_size=21)

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
    ("ISTA (TV)", res_ista),
    ("FISTA (TV)", res_fista),
    ("NLM", denoised_nlm)
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
