import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
    print("Warning: bm3d not found. Please run `pip install bm3d` if you want standard BM3D comparison.")

# Add KAIR path for DnCNN
sys.path.append(r"D:\02_Projects\ML\KAIR")
try:
    from models.network_dncnn import DnCNN as net
except ImportError:
    print("Error: Could not load KAIR DnCNN model. Check path.")

# ==========================================
# 0. Utilities
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

# ==========================================
# 1. ADMM
# ==========================================
def run_AMDD(y, lam=0.08, rho=10.0, num_iters=300):
    x = y.clone()
    z_x = torch.zeros_like(x)
    z_y = torch.zeros_like(x)
    u_x = torch.zeros_like(x)
    u_y = torch.zeros_like(x)
    lr = 0.05 
    
    for i in range(num_iters):
        dx = diff_x(x)
        dy = diff_y(x)
        grad_x = (x - y) + rho * diff_x_T(dx - z_x + u_x) + rho * diff_y_T(dy - z_y + u_y)
        x = x - lr * grad_x
        
        dx = diff_x(x)
        dy = diff_y(x)
        vx = dx + u_x
        vy = dy + u_y
        v_norm = torch.sqrt(vx**2 + vy**2)
        factor = torch.relu(v_norm - lam / rho) / (v_norm + 1e-8)
        
        z_x = vx * factor
        z_y = vy * factor
        
        u_x = u_x + dx - z_x
        u_y = u_y + dy - z_y
        
    return torch.clamp(x, 0, 1)

# ==========================================
# 2. ISTA & FISTA
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

def ista_tv(noisy, lambd, num_iter=50):
    x = noisy.clone()
    step_size = 0.5
    for i in range(num_iter):
        grad = x - noisy
        z = x - step_size * grad
        x = prox_tv_custom_pt(z, lambd * step_size, n_iter=10)
    return torch.clamp(x, 0, 1)

def fista_tv(noisy, lambd, num_iter=50):
    x = noisy.clone()
    y = x.clone()
    t = 1.0
    step_size = 0.5
    for i in range(num_iter):
        x_old = x.clone()
        grad = y - noisy
        z = y - step_size * grad
        x = prox_tv_custom_pt(z, lambd * step_size, n_iter=10)
        
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next
    return torch.clamp(x, 0, 1)

# ==========================================
# 3. DnCNN Runner
# ==========================================
def run_DnCNN(img_noisy_t, device, model_pool=r"D:\02_Projects\ML\KAIR\model_zoo", model_name='dncnn_25'):
    n_channels = 1
    nb = 17
    model_path = os.path.join(model_pool, model_name+'.pth')
    
    model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    with torch.no_grad():
        img_E = model(img_noisy_t)
        
    return torch.clamp(img_E, 0, 1)

# ==========================================
# 4. Main Comparison Execution
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    image_np = img_as_float(data.camera())
    sigma = 25.0 / 255.0  # dncnn_25 is trained on noise level 25
    
    np.random.seed(0)
    noisy_np = image_np + sigma * np.random.randn(*image_np.shape)
    noisy_np = np.clip(noisy_np, 0, 1)
    
    img_clean_t = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    img_noisy_t = torch.tensor(noisy_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    results = {}
    
    # 0. BM3D Baseline
    if HAS_BM3D:
        print("Running BM3D...")
        t0 = time.time()
        res_bm3d_np = bm3d.bm3d(noisy_np, sigma_psd=sigma)
        res_bm3d_np = np.clip(res_bm3d_np, 0, 1)
        results['BM3D'] = {'img': res_bm3d_np, 'time': time.time() - t0}
    
    # 1. ISTA
    print("Running ISTA...")
    t0 = time.time()
    res_ista = ista_tv(img_noisy_t, lambd=0.1, num_iter=100)
    results['ISTA'] = {'img': res_ista.squeeze().cpu().numpy(), 'time': time.time() - t0}
    
    # 2. FISTA
    print("Running FISTA...")
    t0 = time.time()
    res_fista = fista_tv(img_noisy_t, lambd=0.1, num_iter=100)
    results['FISTA'] = {'img': res_fista.squeeze().cpu().numpy(), 'time': time.time() - t0}
    
    # 3. ADMM (AMDD)
    print("Running ADMM...")
    t0 = time.time()
    res_admm = run_AMDD(img_noisy_t, lam=0.1, rho=5.0, num_iters=100)
    results['ADMM'] = {'img': res_admm.squeeze().cpu().numpy(), 'time': time.time() - t0}
    
    # 4. DnCNN
    print("Running DnCNN...")
    t0 = time.time()
    res_dncnn = run_DnCNN(img_noisy_t, device)
    results['DnCNN'] = {'img': res_dncnn.squeeze().cpu().numpy(), 'time': time.time() - t0}
    
    # Calculate Metrics and Print
    summary_lines = []
    summary_lines.append("\n" + "="*50)
    summary_lines.append(f"{'Method':<10} | {'PSNR (dB)':<10} | {'SSIM':<10} | {'Time (s)':<10}")
    summary_lines.append("-" * 50)
    
    noisy_psnr = psnr_metric(image_np, noisy_np, data_range=1.0)
    noisy_ssim = ssim_metric(image_np, noisy_np, data_range=1.0)
    summary_lines.append(f"{'Noisy':<10} | {noisy_psnr:<10.2f} | {noisy_ssim:<10.4f} | {'N/A':<10}")
    
    for name, data_dict in results.items():
        img = data_dict['img']
        t = data_dict['time']
        p = psnr_metric(image_np, img, data_range=1.0)
        s = ssim_metric(image_np, img, data_range=1.0)
        summary_lines.append(f"{name:<10} | {p:<10.2f} | {s:<10.4f} | {t:<10.4f}")
        
    summary_lines.append("="*50)
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "metrics_summary.txt"), "w", encoding='utf-8') as f:
        f.write(summary_text)
    
    # Save individual images for paper inclusion (keeping as png for archival, but they are small)
    io.imsave(os.path.join(output_dir, "01_GroundTruth.png"), (image_np * 255).astype(np.uint8))
    io.imsave(os.path.join(output_dir, "02_Noisy.png"), (noisy_np * 255).astype(np.uint8))
    for name in results:
        io.imsave(os.path.join(output_dir, f"result_{name}.png"), (results[name]['img'] * 255).astype(np.uint8))

    # Detailed Visualization for Paper (Grid)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    # Define images to show in order
    plot_data = [
        ("Ground Truth", image_np),
        (f"Noisy ({noisy_psnr:.2f}dB)", noisy_np)
    ]
    for name in ['BM3D', 'ISTA', 'FISTA', 'ADMM', 'DnCNN']:
        if name in results:
            p = psnr_metric(image_np, results[name]['img'], data_range=1.0)
            plot_data.append((f"{name} ({p:.2f}dB)", results[name]['img']))

    for i in range(len(axes)):
        if i < len(plot_data):
            title, img = plot_data[i]
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "denoising_visual_comparison.jpg"), dpi=150, bbox_inches='tight')
    
    # Zoomed-in patch comparison (Bottom right corner/typical texture)
    h, w = image_np.shape
    r_start, c_start = h//2, w//2
    patch_size = 120
    
    fig_patch, axes_p = plt.subplots(1, 4, figsize=(20, 5))
    patch_names = ['Ground Truth', 'BM3D', 'ADMM', 'DnCNN']
    for i, name in enumerate(patch_names):
        if name == 'Ground Truth':
            target = image_np
        elif name in results:
            target = results[name]['img']
        else:
            continue
            
        patch = target[r_start:r_start+patch_size, c_start:c_start+patch_size]
        axes_p[i].imshow(patch, cmap='gray')
        axes_p[i].set_title(f"Zoomed: {name}", fontsize=14)
        axes_p[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "denoising_zoom_comparison.jpg"), dpi=150, bbox_inches='tight')
    
    # === Performance Charts generation ===
    plt.style.use('ggplot')
    fig_charts, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    methods_list = []
    psnr_list = []
    ssim_list = []
    time_list = []
    
    # Use order consistent with table
    for name in ['ADMM', 'ISTA', 'FISTA', 'BM3D', 'DnCNN']:
        if name in results:
            methods_list.append(name)
            psnr_list.append(psnr_metric(image_np, results[name]['img'], data_range=1.0))
            ssim_list.append(ssim_metric(image_np, results[name]['img'], data_range=1.0))
            time_list.append(results[name]['time'])
            
    x_axis = np.arange(len(methods_list))
    width_bar = 0.35
    
    ax1.bar(x_axis - width_bar/2, psnr_list, width_bar, label='PSNR (dB)', color='skyblue', edgecolor='black')
    ax1.set_ylabel('PSNR (dB)', fontweight='bold')
    ax1.set_title('Quality Comparison: PSNR & SSIM', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_axis)
    ax1.set_xticklabels(methods_list)
    ax1.set_ylim(25, 31)
    ax1.legend(loc='upper left')
    
    ax1_twin_p = ax1.twinx()
    ax1_twin_p.bar(x_axis + width_bar/2, ssim_list, width_bar, label='SSIM', color='orange', alpha=0.7, edgecolor='black')
    ax1_twin_p.set_ylabel('SSIM', fontweight='bold')
    ax1_twin_p.set_ylim(0.5, 0.9)
    ax1_twin_p.legend(loc='upper right')
    
    ax2.bar(methods_list, time_list, color='salmon', edgecolor='black')
    ax2.set_yscale('log')
    ax2.set_ylabel('Time (s) - Log Scale', fontweight='bold')
    ax2.set_title('Efficiency Comparison: Execution Time', fontsize=14, fontweight='bold')
    for i, v in enumerate(time_list):
        ax2.text(i, v, f'{v:.2f}s', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_charts.jpg"), dpi=150, bbox_inches='tight')
    
    print(f"\nVisual results and performance charts (JPG) saved to: {output_dir}")

if __name__ == '__main__':
    main()
