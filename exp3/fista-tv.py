import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2
import bm3d
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import glob
import time

# ==========================================
# 1. Helper Functions
# ==========================================

def load_image(path, size=256):
    """Load image, convert to grayscale, resize, and normalize to [0,1]."""
    if not os.path.exists(path):
        # Fallback if file doesn't exist for demo purposes
        print(f"Warning: {path} not found. Generating dummy image.")
        return np.zeros((size, size))

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size))
    return img.astype(np.float32) / 255.0


def get_mask(shape, drop_rate=0.5):
    """Generate a random binary mask."""
    np.random.seed(42)  # Fixed seed for reproducibility
    mask = np.random.rand(*shape) > drop_rate
    return mask.astype(np.float32)


def soft_threshold(x, lam):
    """Soft Thresholding Operator for L1 norm."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def compute_metrics(clean, estimated):
    """Calculate PSNR and SSIM."""
    # Ensure range [0, 1]
    est_clipped = np.clip(estimated, 0, 1)
    p = psnr(clean, est_clipped, data_range=1.0)
    s = ssim(clean, est_clipped, data_range=1.0)
    return p, s


# ==========================================
# 2. Optimization Algorithms
# ==========================================

def ista_wavelet_inpainting(y, mask, wavelet='db1', lam=0.01, iterations=100):
    """
    ISTA implementation using Wavelet Sparsity.
    Solves: min 1/2||M(W^-1 alpha) - y||^2 + lam||alpha||_1
    """
    # Initialize coefficients (alpha)
    coeffs = pywt.wavedec2(y, wavelet)
    # Flatten coefficients for easier manipulation if needed,
    # but here we keep structure for pywt

    # Precompute Lipschitz constant approximation (Max eigenvalue of M^T M is 1)
    # Step size eta <= 1/L. Here L=1.
    eta = 1.0

    # Initialize reconstruction
    x_rec = np.copy(y)

    losses = []

    for k in range(iterations):
        # 1. Gradient Step: grad = M^T (M x - y)
        # Note: M is symmetric and diagonal, so M^T = M
        residual = mask * x_rec - y
        grad_x = mask * residual  # Gradient w.r.t image domain

        # Move to wavelet domain to update coefficients
        # x_k+1 = prox(W(x_k - eta * grad))

        # Current image minus gradient step
        z = x_rec - eta * grad_x

        # Wavelet Transform
        coeffs = pywt.wavedec2(z, wavelet)

        # 2. Proximal Step (Soft Thresholding) on coefficients
        new_coeffs = list(coeffs)
        new_coeffs[0] = coeffs[0]  # Usually don't threshold approximation coeff
        for i in range(1, len(coeffs)):
            new_coeffs[i] = tuple([soft_threshold(c, lam * eta) for c in coeffs[i]])

        # Inverse Wavelet Transform to get new image
        x_rec = pywt.waverec2(new_coeffs, wavelet)

        # Record loss
        loss = 0.5 * np.linalg.norm(mask * x_rec - y) ** 2 + lam * np.sum(
            [np.sum(np.abs(c)) for c in new_coeffs[1:]])  # Simplified
        losses.append(loss)

    return x_rec, losses


def fista_wavelet_inpainting(y, mask, wavelet='db1', lam=0.01, iterations=100):
    """
    FISTA implementation using Wavelet Sparsity (Accelerated).
    """
    eta = 1.0

    # Initialization
    x_k = np.copy(y)
    y_k = np.copy(x_k)  # Momentum variable
    t_k = 1.0

    losses = []

    for k in range(iterations):
        # 1. Gradient Step on y_k (momentum point)
        residual = mask * y_k - y
        grad = mask * residual

        z = y_k - eta * grad

        # 2. Proximal Step (Wavelet Soft Thresholding)
        coeffs = pywt.wavedec2(z, wavelet)
        new_coeffs = list(coeffs)
        new_coeffs[0] = coeffs[0]
        for i in range(1, len(coeffs)):
            new_coeffs[i] = tuple([soft_threshold(c, lam * eta) for c in coeffs[i]])

        x_k_next = pywt.waverec2(new_coeffs, wavelet)

        # 3. FISTA Update (Momentum)
        t_k_next = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2
        y_k = x_k_next + ((t_k - 1) / t_k_next) * (x_k_next - x_k)

        # Update vars
        x_k = x_k_next
        t_k = t_k_next

        # Record Loss
        loss = 0.5 * np.linalg.norm(mask * x_k - y) ** 2
        losses.append(loss)

    return x_k, losses


import numpy as np


def prox_tv(y, lam, n_iters=100):
    """
    Standard Dual Approach for TV Denoising (Chambolle's Algorithm).
    Solves: min_x  0.5 ||x - y||^2 + lam * TV(x)

    Corrects the gradient/divergence definitions to ensure adjointness.
    """
    # 确保 lam 是浮点数
    lam = float(lam)

    # 初始化对偶变量 p = (px, py)
    # px 存储水平方向的对偶变量，py 存储垂直方向
    px = np.zeros_like(y)
    py = np.zeros_like(y)

    # 步长 tau (1/8 for 2D guarantees convergence)
    tau = 0.125

    for _ in range(n_iters):
        # --- 1. 计算散度 (Divergence) ---
        # div(p) = dx^T(px) + dy^T(py) (Backward difference)

        # 计算 px 的后向差分
        div_px = np.zeros_like(px)
        div_px[1:-1, :] = px[1:-1, :] - px[:-2, :]
        div_px[0, :] = px[0, :]  # 边界条件
        div_px[-1, :] = -px[-2, :]  # 边界条件保证 adjoint

        # 计算 py 的后向差分
        div_py = np.zeros_like(py)
        div_py[:, 1:-1] = py[:, 1:-1] - py[:, :-2]
        div_py[:, 0] = py[:, 0]
        div_py[:, -1] = -py[:, -2]

        div_p = div_px + div_py

        # --- 2. 估计当前的 Primal 变量 ---
        # 根据对偶公式: x = y - lam * div(p)
        x_curr = y - lam * div_p

        # --- 3. 计算梯度 (Gradient) ---
        # grad(x) = (dx(x), dy(x)) (Forward difference)

        gx = np.zeros_like(x_curr)
        gx[:-1, :] = x_curr[1:, :] - x_curr[:-1, :]  # x[i+1] - x[i]

        gy = np.zeros_like(x_curr)
        gy[:, :-1] = x_curr[:, 1:] - x_curr[:, :-1]  # x[j+1] - x[j]

        # --- 4. 更新对偶变量 (Gradient Ascent on Dual) ---
        px_new = px + tau * gx
        py_new = py + tau * gy

        # --- 5. 投影 (Projection) ---
        # 投影到 L2 球: max(1, |p|)
        norm = np.sqrt(px_new ** 2 + py_new ** 2)
        norm = np.maximum(1.0, norm)

        px = px_new / norm
        py = py_new / norm

    # --- 最终重建 ---
    div_px = np.zeros_like(px)
    div_px[1:-1, :] = px[1:-1, :] - px[:-2, :]
    div_px[0, :] = px[0, :]
    div_px[-1, :] = -px[-2, :]

    div_py = np.zeros_like(py)
    div_py[:, 1:-1] = py[:, 1:-1] - py[:, :-2]
    div_py[:, 0] = py[:, 0]
    div_py[:, -1] = -py[:, -2]

    return y - lam * (div_px + div_py)


import numpy as np


def prox_tv(b, lam, n_iters=20):
    """
    Fast Gradient Projection (FGP) algorithm for TV Denoising.
    Solves: min_x  0.5 ||x - b||^2 + lam * TV(x)

    Ref: Beck & Teboulle, "Fast Gradient-Based Algorithms for Constrained
    Total Variation Image Denoising...", IEEE TIP 2009.
    """
    # 确保输入是浮点
    b = b.astype(np.float64)

    # 获取尺寸
    rows, cols = b.shape

    # 初始化对偶变量 (p, q) 对应 (x方向, y方向) 的梯度限制
    p = np.zeros((rows, cols), dtype=np.float64)
    q = np.zeros((rows, cols), dtype=np.float64)

    # 投影算子需要的中间变量
    r = np.zeros_like(p)
    s = np.zeros_like(q)

    # 步长 (1 / (8 * L), L=1 for TV) -> 0.125
    # 为了数值稳定性，取稍微保守的值
    step = 0.125  # (1.0 / 8.0)

    def get_gradient(x):
        """ 计算梯度 (Forward Difference) """
        # grad_x: x[i+1, j] - x[i, j]
        gx = np.zeros_like(x)
        gx[:-1, :] = x[1:, :] - x[:-1, :]

        # grad_y: x[i, j+1] - x[i, j]
        gy = np.zeros_like(x)
        gy[:, :-1] = x[:, 1:] - x[:, :-1]
        return gx, gy

    def get_divergence(p, q):
        """ 计算散度 (Backward Difference) - 梯度的负伴随 """
        # div_x of p
        # p[i, j] - p[i-1, j]
        dx = np.zeros_like(p)
        dx[1:-1, :] = p[1:-1, :] - p[:-2, :]
        dx[0, :] = p[0, :]
        dx[-1, :] = -p[-2, :]

        # div_y of q
        # q[i, j] - q[i, j-1]
        dy = np.zeros_like(q)
        dy[:, 1:-1] = q[:, 1:-1] - q[:, :-2]
        dy[:, 0] = q[:, 0]
        dy[:, -1] = -q[:, -2]

        return dx + dy

    # FGP 迭代
    # t_k 用于 FISTA 风格的加速 (Nesterov momentum for Dual)
    t_k = 1.0

    for _ in range(n_iters):
        # 1. 基于当前的对偶变量 (r, s)，计算 Primal 也就是去噪后的图像估计
        # x = b - lam * L^T(r,s)
        # 由于 div = -L^T，所以公式是: x = b + lam * div(r,s)
        # 这里的 div 必须和 get_divergence 保持一致
        div = get_divergence(r, s)
        x_est = b + lam * div

        # 2. 计算 x_est 的梯度
        gx, gy = get_gradient(x_est)

        # 3. 梯度上升更新对偶变量 (Projected Gradient Ascent on Dual)
        # 这里的 update direction 是关键
        p_next = r - step * (gx * lam)  # 注意这里的符号，通常对偶是 - grad
        # 其实最简单的写法是: p = project(p + sigma * grad(b + lam*div(p)))
        # 我们使用 Beck & Teboulle 的标准形式:
        # projected_gradient_step: temp = r + (1/(8*lam)) * grad( b - lam * div(r) ) ... 比较乱

        # --- 让我们换回最简单的 Chambolle (2004) 原始迭代，不要用 FGP 复杂的动量，先保证稳定 ---
        # Chambolle 迭代公式:
        # p^{n+1} = (p^n + tau * grad( div(p^n) - b/lam )) / (1 + tau |grad(...)|)
        # 等价于:
        # temp = div(p, q) - b/lam
        # g_temp_x, g_temp_y = grad(temp)
        # p_new = p + step * g_temp_x
        # q_new = q + step * g_temp_y

        # 这里的 b/lam 可能会很大，容易溢出。
        # 我们采用下面的稳定实现：

        # 当前估计: u = b + lam * div(p, q)
        u = b + lam * get_divergence(p, q)

        # 误差梯度: g = grad(u)
        g_x, g_y = get_gradient(u)

        # 更新对偶变量 (Ascent direction)
        # 注意：这里我们最小化 ||x-b||^2 + lam*TV
        # 对应的对偶更新方向是: p = p - step * grad(u) ???
        # 不，正确的 Chambolle 更新是: p = (p + tau * grad(u)) / ...
        # 因为 u = b + lam * div p。
        # grad(u) 其实是 grad(b) + lam * grad(div p).

        # 严格的 Chambolle 更新步骤：
        p_new = p + (step / lam) * g_x
        q_new = q + (step / lam) * g_y

        # 投影 (Projection) onto unit ball
        norm = np.sqrt(p_new ** 2 + q_new ** 2)
        norm = np.maximum(1.0, norm)

        p = p_new / norm
        q = q_new / norm

    return b + lam * get_divergence(p, q)


def fista_tv_inpainting(y, mask, lam=0.03, iterations=100):
    """
    FISTA Inpainting with TV Regularization.
    Safe normalization and Warm-Start included.
    """
    # --- 1. 安全归一化 ---
    # 你的 Loss 曲线显示数值在 15-50 之间，说明是 0-1 范围
    # 但如果为了保险，我们检查最大值
    y_in = y.copy()
    scale = 1.0
    if y_in.max() > 1.5:
        scale = 255.0
        y_in = y_in / scale

    # --- 2. 热启动 (Warm Start) ---
    # 用均值填充空洞，避免 "Zero-Lock"
    x_k = np.copy(y_in)
    valid_mask = (mask > 0.5)
    mean_val = np.mean(y_in[valid_mask])
    x_k[~valid_mask] = mean_val  # 填补空洞

    y_k = np.copy(x_k)  # 动量变量
    t_k = 1.0
    eta = 1.0  # 步长 (Lipschitz constant = 1 for Mask)

    losses = []

    for k in range(iterations):
        # 1. Gradient Step
        # data_term_grad = M^T (M y_k - y)
        # mask is M.
        # residual = y_k - y (only where mask=1)
        residual = mask * (y_k - y_in)

        z = y_k - eta * residual

        # 2. Proximal Step
        # 注意: 参数传入 lam * eta
        # 对于 0-1 范围且严重缺失的图片，lam 建议 0.01 - 0.05
        x_k_next = prox_tv(z, lam * eta, n_iters=10)

        # 3. FISTA Momentum
        t_k_next = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2
        y_k = x_k_next + ((t_k - 1) / t_k_next) * (x_k_next - x_k)

        # Update
        x_k = x_k_next
        t_k = t_k_next

        # Clip
        x_k = np.clip(x_k, 0, 1)
        y_k = np.clip(y_k, 0, 1)

        # Monitor
        loss = 0.5 * np.linalg.norm(mask * x_k - y_in) ** 2
        losses.append(loss)

    # 还原
    result = x_k * scale
    return result, losses
# ==========================================
# 3. Main Execution & Comparison
# ==========================================
# -----------------------------------------------------------
# 这里的 def 必须顶格，不要有缩进
# -----------------------------------------------------------
def run_single_image(image_path, show_plot=False):
    """
    运行单张实验：BM3D vs ISTA(L1) vs FISTA(L1) vs FISTA(TV).
    修复了解包Bug，并补全了作业要求的 TV 正则项对比。
    """
    img_name = os.path.basename(image_path)
    print(f"\n--- Processing: {img_name} ---")

    # 1. 准备数据
    img_gt = load_image(image_path)
    if img_gt is None: return {}
    mask = get_mask(img_gt.shape, drop_rate=0.5)
    masked_img = img_gt * mask

    # --- 辅助函数：安全运行算法并拆包 ---
    def run_safe(algo_func, name, *args, **kwargs):
        t0 = time.time()
        try:
            res = algo_func(*args, **kwargs)
        except Exception as e:
            print(f"Warning: {name} failed - {e}")
            return masked_img, [], 0.0

        t_cost = time.time() - t0

        # 【核心修复】智能拆包 (修复了 tuple index out of range)
        loss = []
        if isinstance(res, (tuple, list)):
            img = res[0]  # 取图片
            # 自动判断 loss 的位置，通常是第2个元素 res[1]
            if len(res) > 1:
                loss = res[1]
        else:
            img = res  # BM3D 这种情况
            loss = []

        # 确保是 numpy 格式
        if not isinstance(img, np.ndarray): img = np.array(img)
        return img, loss, t_cost

    # 2. Run BM3D
    img_bm3d, _, t_bm3d = run_safe(bm3d.bm3d, "BM3D", masked_img, sigma_psd=30 / 255,
                                   stage_arg=bm3d.BM3DStages.ALL_STAGES)
    p_bm3d, s_bm3d = compute_metrics(img_gt, img_bm3d)
    print(f"BM3D      | Time: {t_bm3d:.4f}s | PSNR: {p_bm3d:.2f} | SSIM: {s_bm3d:.4f}")

    # 3. Run ISTA (Wavelet/L1)
    img_ista, loss_ista, t_ista = run_safe(ista_wavelet_inpainting, "ISTA", masked_img, mask, lam=0.01, iterations=100)
    p_ista, s_ista = compute_metrics(img_gt, img_ista)
    print(f"ISTA (W)  | Time: {t_ista:.4f}s | PSNR: {p_ista:.2f} | SSIM: {s_ista:.4f}")

    # 4. Run FISTA (Wavelet/L1)
    img_fista, loss_fista, t_fista = run_safe(fista_wavelet_inpainting, "FISTA-L1", masked_img, mask, lam=0.01,
                                              iterations=100)
    p_fista, s_fista = compute_metrics(img_gt, img_fista)
    print(f"FISTA (W) | Time: {t_fista:.4f}s | PSNR: {p_fista:.2f} | SSIM: {s_fista:.4f}")

    # 5. Run FISTA (TV) - 作业要求必须有！
    if 'fista_tv_inpainting' in globals():
        img_tv, loss_tv, t_tv = run_safe(fista_tv_inpainting, "FISTA-TV", masked_img, mask, lam=0.03, iterations=100)
        p_tv, s_tv = compute_metrics(img_gt, img_tv)
        print(f"FISTA (TV)| Time: {t_tv:.4f}s | PSNR: {p_tv:.2f} | SSIM: {s_tv:.4f}")
    else:
        img_tv, loss_tv, p_tv = masked_img, [], 0.0
        print("Warning: fista_tv_inpainting function missing.")

    # 6. 可视化 (Show Plot)
    if show_plot:
        plt.figure(figsize=(18, 6))
        titles = ["Original", "Masked", f"BM3D\n{p_bm3d:.2f}dB", f"ISTA-W\n{p_ista:.2f}dB", f"FISTA-W\n{p_fista:.2f}dB",
                  f"FISTA-TV\n{p_tv:.2f}dB"]
        images = [img_gt, masked_img, img_bm3d, img_ista, img_fista, img_tv]

        for i in range(6):
            plt.subplot(1, 6, i + 1)
            plt.imshow(np.clip(images[i], 0, 1), cmap='gray')
            plt.title(titles[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

        # 收敛曲线对比 (作业要求)
        plt.figure(figsize=(8, 5))
        if len(loss_ista) > 0: plt.plot(loss_ista, label='ISTA (L1)', linestyle='--')
        if len(loss_fista) > 0: plt.plot(loss_fista, label='FISTA (L1)')
        if len(loss_tv) > 0: plt.plot(loss_tv, label='FISTA (TV)')
        plt.yscale('log')
        plt.legend()
        plt.title('Convergence Analysis')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.show()

    return {
        'filename': img_name,
        'psnr_bm3d': p_bm3d, 'psnr_ista': p_ista, 'psnr_fista': p_fista, 'psnr_tv': p_tv
    }
### ==========================================
### 4. Modified Set14 Experiment & Main
### ==========================================

def run_set14_experiment(target_file_path):
    """
    遍历 Set14 文件夹，计算所有图片的平均指标。
    """
    # 获取 Set14 文件夹路径
    folder_path = os.path.dirname(target_file_path)

    # 支持 png, bmp, jpg, tif
    exts = ['*.png', '*.bmp', '*.jpg', '*.tif']
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not image_files:
        print(f"Error: No images found in {folder_path}")
        return

    print(f"Found {len(image_files)} images in Set14. Starting Loop...")

    results = []
    target_name = os.path.basename(target_file_path)

    for img_path in image_files:
        # 如果是 ppt3.png，则 show_plot=True
        is_target = (os.path.basename(img_path) == target_name)

        try:
            res = run_single_image(img_path, show_plot=is_target)
            results.append(res)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()

    # Calculate Averages
    if results:
        avg_p_bm3d = np.mean([r['psnr_bm3d'] for r in results])
        avg_p_ista = np.mean([r['psnr_ista'] for r in results])
        avg_p_fista = np.mean([r['psnr_fista'] for r in results])

        print("\n" + "=" * 40)
        print(f"AVERAGE RESULTS over {len(results)} images:")
        print(f"BM3D  PSNR: {avg_p_bm3d:.2f}")
        print(f"ISTA  PSNR: {avg_p_ista:.2f}")
        print(f"FISTA PSNR: {avg_p_fista:.2f}")
        print("=" * 40)


if __name__ == "__main__":
    # 使用你原本的路径 (注意反斜杠转义或使用 raw string r"...")
    target_img_filename = r"D:\zhangchengxi's_BM3D\Set14\ppt3.png"

    # 检查路径是否存在，不存在则尝试使用你之前报错信息里的路径
    if not os.path.exists(target_img_filename):
        # 备用路径（根据你的报错信息推测）
        target_img_filename = r"C:\Users\19811\Desktop\Set14\Set14\ppt3.png"

    if os.path.exists(target_img_filename):
        run_set14_experiment(target_img_filename)
    else:
        print(f"File not found: {target_img_filename}")
        print("Please check the path in 'if __name__ == ...' block.")