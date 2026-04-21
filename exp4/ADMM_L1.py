import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg') # 设置为非交互式后端，防止在无界面环境下崩溃
import matplotlib.pyplot as plt
from skimage import data, img_as_float, color, io
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import cv2
import csv

try:
    import bm3d
    HAS_BM3D = True
except ImportError:
    HAS_BM3D = False
    print("Warning: bm3d not found. Please run `pip install bm3d` if you want BM3D comparison.")


# ==========================================
# Logger for Trackable Metrics
# ==========================================
class PerfLogger:
    def __init__(self, algo_name, orig_np, log_iter_freq=100, log_time_freq=0.5, thresholds=[20, 23, 25]):
        self.algo_name = algo_name
        self.orig_np = orig_np
        self.log_iter_freq = log_iter_freq
        self.log_time_freq = log_time_freq
        self.thresholds = sorted(thresholds)
        self.history = []       # list of dictionaries
        self.thresh_hits = {}   # thresh -> (iter, time)
        
        self.start_time = None
        self.next_time_target = log_time_freq
        self.total_algo_time = 0.0

    def start(self):
        self.start_time = time.time()

    def log_iter(self, iteration, x_tensor):
        now = time.time()
        running_time = now - self.start_time + self.total_algo_time
        
        x_np = torch.clamp(x_tensor, 0, 1).squeeze().cpu().numpy()
        p_val = psnr_metric(self.orig_np, x_np, data_range=1.0)
        s_val = ssim_metric(self.orig_np, x_np, data_range=1.0)

        for th in self.thresholds:
            if th not in self.thresh_hits and p_val >= th:
                self.thresh_hits[th] = (iteration + 1, running_time)

        log_it = False
        if (iteration + 1) % self.log_iter_freq == 0:
            log_it = True
        if running_time >= self.next_time_target:
            log_it = True
            while self.next_time_target <= running_time:
                self.next_time_target += self.log_time_freq

        if log_it:
            self.history.append({
                'algo': self.algo_name,
                'iteration': iteration + 1,
                'time_sec': running_time,
                'psnr': p_val,
                'ssim': s_val
            })

        # Pause timer for validation logic overhead
        self.total_algo_time = running_time
        self.start_time = time.time()
        return p_val, s_val


# ==========================================
# 1. 核心算法: AMDD (ADMM) 用于 Total Variation 去噪
# ==========================================
def diff_x(x):
    pad = F.pad(x, (0, 1, 0, 0), mode='replicate')
    return pad[:, :, :, 1:] - pad[:, :, :, :-1]

def diff_y(x):
    pad = F.pad(x, (0, 0, 0, 1), mode='replicate')
    return pad[:, :, 1:, :] - pad[:, :, :-1, :]

def diff_x_T(x):
    res = torch.zeros_like(x)
    res[:, :, :, 0] = -x[:, :, :, 0]
    res[:, :, :, 1:-1] = x[:, :, :, :-2] - x[:, :, :, 1:-1]
    res[:, :, :, -1] = x[:, :, :, -2]
    return res

def diff_y_T(x):
    res = torch.zeros_like(x)
    res[:, :, 0, :] = -x[:, :, 0, :]
    res[:, :, 1:-1, :] = x[:, :, :-2, :] - x[:, :, 1:-1, :]
    res[:, :, -1, :] = x[:, :, -2, :]
    return res

def run_AMDD(y, lam=0.5, rho=5.0, rho_w=10.0, num_iters=300, original_img=None, logger=None):
    x = y.clone()
    z_x = torch.zeros_like(x)
    z_y = torch.zeros_like(x)
    u_x = torch.zeros_like(x)
    u_y = torch.zeros_like(x)
    
    # TV-L1 新增辅助变量
    w = torch.zeros_like(x)
    v = torch.zeros_like(x)
    
    # 严格保证收敛的步长: 1 / L, 其中 L = rho_w + 8 * rho
    lr = 1.0 / (rho_w + 8.0 * rho)
    
    psnr_history, ssim_history = [], []
    
    if original_img is not None and logger is None:
        orig_np = original_img.squeeze().cpu().numpy()
        
    if logger: logger.start()

    for i in range(num_iters):
        # ====================
        # 1. 更新 x (二次惩罚项梯度下降)
        # ====================
        dx = diff_x(x)
        dy = diff_y(x)
        
        # TV部分 + L1保真部分的合成梯度
        grad_x = rho_w * (x - y - w + v) + rho * diff_x_T(dx - z_x + u_x) + rho * diff_y_T(dy - z_y + u_y)
        x = x - lr * grad_x
        
        # ====================
        # 2. 更新 z (TV正则项 软阈值，修改为各向同性以匹配 ISTA)
        # ====================
        dx = diff_x(x)
        dy = diff_y(x)
        
        vx = dx + u_x
        vy = dy + u_y
        v_norm = torch.sqrt(vx**2 + vy**2)
        factor = torch.relu(v_norm - lam / rho) / (v_norm + 1e-8)
        
        z_x = vx * factor
        z_y = vy * factor
        
        # ====================
        # 3. 更新 w (L1数据保真项 软阈值)
        # 用于剥离脉冲噪声和部分高斯噪声
        # ====================
        w_in = x - y + v
        w = torch.sign(w_in) * torch.relu(torch.abs(w_in) - 1.0 / rho_w)
        
        # ====================
        # 4. 更新 u, v (对偶变量)
        # ====================
        u_x = u_x + dx - z_x
        u_y = u_y + dy - z_y
        v = v + x - y - w
        
        if logger is not None:
            p_val, s_val = logger.log_iter(i, x)
            psnr_history.append(p_val)
            ssim_history.append(s_val)
        elif original_img is not None:
            x_clipped = torch.clamp(x, 0, 1)
            x_np = x_clipped.squeeze().cpu().numpy()
            psnr_history.append(psnr_metric(orig_np, x_np, data_range=1.0))
            ssim_history.append(ssim_metric(orig_np, x_np, data_range=1.0))

    if original_img is not None or logger is not None:
        return torch.clamp(x, 0, 1), psnr_history, ssim_history
    return torch.clamp(x, 0, 1)


# ==========================================
# 2. 对比算法: ISTA / FISTA
# ==========================================
def prox_tv_custom_pt(x_tensor, lambd, n_iter=10, step_size=0.1):
    u = x_tensor.clone()
    eps = 1e-8
    for _ in range(n_iter):
        grad_x = torch.zeros_like(u)
        grad_x[:, :, :, :-1] = u[:, :, :, 1:] - u[:, :, :, :-1]
        grad_y = torch.zeros_like(u)
        grad_y[:, :, :-1, :] = u[:, :, 1:, :] - u[:, :, :-1, :]
        
        grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + eps)
        tv_grad_x = grad_x / grad_norm
        tv_grad_y = grad_y / grad_norm
        
        div_tv = torch.zeros_like(u)
        div_tv[:, :, :, 1:] += tv_grad_x[:, :, :, 1:] - tv_grad_x[:, :, :, :-1]
        div_tv[:, :, :, 0] += tv_grad_x[:, :, :, 0]
        div_tv[:, :, :, -1] -= tv_grad_x[:, :, :, -2]
        
        div_tv[:, :, 1:, :] += tv_grad_y[:, :, 1:, :] - tv_grad_y[:, :, :-1, :]
        div_tv[:, :, 0, :] += tv_grad_y[:, :, 0, :]
        div_tv[:, :, -1, :] -= tv_grad_y[:, :, -2, :]
        
        gradient = (u - x_tensor) - lambd * div_tv
        u = u - step_size * gradient
    return u

def ista_tv(noisy, lambd, num_iter=50, original_img=None, logger=None):
    x = noisy.clone()
    step_size = 0.5
    psnr_history, ssim_history = [], []
    
    if original_img is not None and logger is None:
        orig_np = original_img.squeeze().cpu().numpy()

    if logger: logger.start()

    for i in range(num_iter):
        # 使用 Subgradient 进行 L1 保真项的优化
        grad = torch.sign(x - noisy)
        # L1 优化的次梯度容易震荡，需要配合较小的步长
        z = x - 0.05 * grad
        x = prox_tv_custom_pt(z, lambd * 0.05, n_iter=10)
        
        if logger is not None:
            p_val, s_val = logger.log_iter(i, x)
            psnr_history.append(p_val)
            ssim_history.append(s_val)
        elif original_img is not None:
            x_clipped = torch.clamp(x, 0, 1)
            x_np = x_clipped.squeeze().cpu().numpy()
            psnr_history.append(psnr_metric(orig_np, x_np, data_range=1.0))
            ssim_history.append(ssim_metric(orig_np, x_np, data_range=1.0))
            
    if original_img is not None or logger is not None:
        return torch.clamp(x, 0, 1), psnr_history, ssim_history
    return torch.clamp(x, 0, 1)

def fista_tv(noisy, lambd, num_iter=50, original_img=None, logger=None):
    x = noisy.clone()
    y = x.clone()
    t = 1.0
    step_size = 0.5
    psnr_history, ssim_history = [], []
    
    if original_img is not None and logger is None:
        orig_np = original_img.squeeze().cpu().numpy()

    if logger: logger.start()

    for i in range(num_iter):
        x_old = x.clone()
        # 使用 Subgradient 进行 L1 保真项的优化
        grad = torch.sign(y - noisy)
        z = y - 0.05 * grad
        x = prox_tv_custom_pt(z, lambd * 0.05, n_iter=10)
        
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next
        
        if logger is not None:
            p_val, s_val = logger.log_iter(i, x)
            psnr_history.append(p_val)
            ssim_history.append(s_val)
        elif original_img is not None:
            x_clipped = torch.clamp(x, 0, 1)
            x_np = x_clipped.squeeze().cpu().numpy()
            psnr_history.append(psnr_metric(orig_np, x_np, data_range=1.0))
            ssim_history.append(ssim_metric(orig_np, x_np, data_range=1.0))
            
    if original_img is not None or logger is not None:
        return torch.clamp(x, 0, 1), psnr_history, ssim_history
    return torch.clamp(x, 0, 1)


# ==========================================
# 3. Demo 直观对比代码
# ==========================================
def compare_and_visualize_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Comparison Demo on: {device}")
    
    image_np = img_as_float(data.camera())
    sigma = 0.1
    # 1. 添加高斯噪声
    noisy_np = image_np + sigma * np.random.randn(*image_np.shape)
    
    # 2. 添加椒盐噪声 (Salt and Pepper Noise)
    prob_snp = 0.05
    rdn = np.random.rand(*noisy_np.shape)
    noisy_np[rdn < prob_snp / 2] = 0.0
    noisy_np[rdn > 1 - prob_snp / 2] = 1.0
    
    noisy_np = np.clip(noisy_np, 0, 1)
    
    img_clean_t = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    img_noisy_t = torch.tensor(noisy_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # === Hyperparameters ===
    lam_amdd = 0.5 # TV-L1 需要较大的正则化权重
    rho_amdd = 5.0
    rho_w_amdd = 10.0
    lam_tv = 0.5 # ISTA TV-L1
    iters = 300 
    
    log_iter_freq = 100
    log_time_freq = 0.5
    thresholds = [20, 23, 25]
    
    loggers = {}
    
    # 1. AMDD
    print(f"\n[1/4] Running AMDD (Ours, TV-L1), iters={iters}...")
    loggers["AMDD"] = PerfLogger("AMDD", image_np, log_iter_freq, log_time_freq, thresholds)
    res_amdd, psnr_amdd_h, ssim_amdd_h = run_AMDD(
        img_noisy_t, lam=lam_amdd, rho=rho_amdd, rho_w=rho_w_amdd, num_iters=iters, original_img=img_clean_t, logger=loggers["AMDD"]
    )
    
    # 2. ISTA
    print(f"[2/4] Running ISTA (Teacher TV), iters={iters}...")
    loggers["ISTA"] = PerfLogger("ISTA", image_np, log_iter_freq, log_time_freq, thresholds)
    res_ista, psnr_ista_h, ssim_ista_h = ista_tv(
        img_noisy_t, lambd=lam_tv, num_iter=iters, original_img=img_clean_t, logger=loggers["ISTA"]
    )
    
    # 3. FISTA
    print(f"[3/4] Running FISTA (Teacher TV), iters={iters}...")
    loggers["FISTA"] = PerfLogger("FISTA", image_np, log_iter_freq, log_time_freq, thresholds)
    res_fista, psnr_fista_h, ssim_fista_h = fista_tv(
        img_noisy_t, lambd=lam_tv, num_iter=iters, original_img=img_clean_t, logger=loggers["FISTA"]
    )
    
    # 4. BM3D
    print("[4/4] Running BM3D...")
    bm3d_time = 0.0
    res_bm3d_np = np.zeros_like(image_np)
    if HAS_BM3D:
        t0 = time.time()
        res_bm3d_np = bm3d.bm3d(noisy_np, sigma_psd=sigma)
        bm3d_time = time.time() - t0
        res_bm3d_np = np.clip(res_bm3d_np, 0, 1)
    else:
        print("BM3D module missing, skipping BM3D.")
        
    res_amdd_np = res_amdd.squeeze().cpu().numpy()
    res_ista_np = res_ista.squeeze().cpu().numpy()
    res_fista_np = res_fista.squeeze().cpu().numpy()
    
    noisy_psnr = psnr_metric(image_np, noisy_np, data_range=1.0)
    noisy_ssim = ssim_metric(image_np, noisy_np, data_range=1.0)
    
    methods = [
        ("Noisy", noisy_np, noisy_psnr, noisy_ssim),
        ("ISTA", res_ista_np, psnr_ista_h[-1], ssim_ista_h[-1]),
        ("FISTA", res_fista_np, psnr_fista_h[-1], ssim_fista_h[-1]),
        ("AMDD", res_amdd_np, psnr_amdd_h[-1], ssim_amdd_h[-1])
    ]
    
    if HAS_BM3D:
        bm3d_psnr = psnr_metric(image_np, res_bm3d_np, data_range=1.0)
        bm3d_ssim = ssim_metric(image_np, res_bm3d_np, data_range=1.0)
        methods.insert(1, ("BM3D", res_bm3d_np, bm3d_psnr, bm3d_ssim))
    
    print("\n" + "=" * 60)
    print(f"对比结果 (Iter {iters}):")
    print("=" * 60)
    for name, img, p, s in methods:
        print(f"{name:15s} - PSNR: {p:.2f} dB, SSIM: {s:.4f}")
    if HAS_BM3D:
        print(f"BM3D execution time: {bm3d_time:.3f} s")
    print("=" * 60)
    
    # === Output Directory setup ===
    # 获取当前脚本所在目录，确保路径正确
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(script_dir, "outputs", f"exp_{timestamp}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    print(f"\n=> 实验结果将保存至: {output_dir}")

    # === Output CSV Report ===
    csv_filename = os.path.join(output_dir, "denoising_performance_log.csv")
    with open(csv_filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Iteration", "Time_sec", "PSNR", "SSIM"])
        for alg, logger in loggers.items():
            for row in logger.history:
                writer.writerow([row['algo'], row['iteration'], f"{row['time_sec']:.3f}", f"{row['psnr']:.2f}", f"{row['ssim']:.4f}"])
                
        writer.writerow([])
        writer.writerow(["Threshold Hits (PSNR -> Iter, Time)"])
        writer.writerow(["Algorithm", "Threshold(dB)", "Hit_Iteration", "Hit_Time_sec"])
        for alg, logger in loggers.items():
            for th in sorted(thresholds):
                if th in logger.thresh_hits:
                    it, tm = logger.thresh_hits[th]
                    writer.writerow([alg, th, it, f"{tm:.3f}"])
                else:
                    writer.writerow([alg, th, "Did Not Hit", "N/A"])
                    
        if HAS_BM3D:
            writer.writerow([])
            writer.writerow(["BM3D Quick Profile"])
            writer.writerow(["Time_sec", "Final_PSNR", "Final_SSIM"])
            writer.writerow([f"{bm3d_time:.3f}", f"{bm3d_psnr:.2f}", f"{bm3d_ssim:.4f}"])
            
    print(f"\n=> 性能日志已成功导出至: {csv_filename}")
    
    # === 输出 PDF 绘图 ===
    from getimage import save_comparison_plot
    psnr_dict = {
        'ISTA': psnr_ista_h,
        'FISTA': psnr_fista_h,
        'AMDD': psnr_amdd_h
    }
    ssim_dict = {
        'ISTA': ssim_ista_h,
        'FISTA': ssim_fista_h,
        'AMDD': ssim_amdd_h
    }
    
    if HAS_BM3D:
        psnr_dict['BM3D_Baseline'] = [bm3d_psnr]
        ssim_dict['BM3D_Baseline'] = [bm3d_ssim]
        
    save_comparison_plot(methods, image_np, psnr_dict, ssim_dict, filename="denoising_comparison.pdf", save_dir=output_dir)
    
if __name__ == '__main__':
    # 运行对比大 Demo
    compare_and_visualize_demo()
