import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from skimage import data, img_as_float, restoration
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ============ 老师的方法 (Teacher's Approach) ============
def fista_tv_inpainting_teacher(distorted_img, mask, original_img, lambd=0.04, num_iter=60):
    x = distorted_img.copy()
    y = x.copy()
    t = 1.0
    psnr_hist, ssim_hist = [], []
    for i in range(num_iter):
        x_old = x.copy()
        res = y - mask * (y - distorted_img)
        x = restoration.denoise_tv_chambolle(res, weight=lambd)
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next
        x_clip = np.clip(x, 0, 1)
        psnr_hist.append(psnr(original_img, x_clip, data_range=1.0))
        ssim_hist.append(ssim(original_img, x_clip, data_range=1.0))
    return np.clip(x, 0, 1), psnr_hist, ssim_hist

def prox_tv_custom(x, lambd, n_iter=50, step_size=0.1):
    u = x.copy()
    eps = 1e-8
    for _ in range(n_iter):
        grad_x = np.zeros_like(u)
        grad_x[:, :-1] = u[:, 1:] - u[:, :-1]
        grad_y = np.zeros_like(u)
        grad_y[:-1, :] = u[1:, :] - u[:-1, :]
        grad_norm = np.sqrt(grad_x**2 + grad_y**2 + eps)
        tv_grad_x = grad_x / grad_norm
        tv_grad_y = grad_y / grad_norm
        div_tv = np.zeros_like(u)
        div_tv[:, 1:] += tv_grad_x[:, 1:] - tv_grad_x[:, :-1]
        div_tv[:, 0] += tv_grad_x[:, 0]
        div_tv[:, -1] -= tv_grad_x[:, -2]
        div_tv[1:, :] += tv_grad_y[1:, :] - tv_grad_y[:-1, :]
        div_tv[0, :] += tv_grad_y[0, :]
        div_tv[-1, :] -= tv_grad_y[-2, :]
        gradient = (u - x) - lambd * div_tv
        u = u - step_size * gradient
    return u

def fista_tv_inpainting_teacher_custom(distorted_img, mask, original_img, lambd=0.04, num_iter=60):
    x = distorted_img.copy()
    y = x.copy()
    t = 1.0
    psnr_hist, ssim_hist = [], []
    for i in range(num_iter):
        x_old = x.copy()
        res = y - mask * (y - distorted_img)
        x = prox_tv_custom(res, lambd, n_iter=100)
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next
        x_clip = np.clip(x, 0, 1)
        psnr_hist.append(psnr(original_img, x_clip, data_range=1.0))
        ssim_hist.append(ssim(original_img, x_clip, data_range=1.0))
    return np.clip(x, 0, 1), psnr_hist, ssim_hist

# ============ 你的原始方法 (Your Original Approach Adapted to Inpainting) ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def calc_div_tv(u, eps=1e-5):
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
    
    return div_tv

def my_admm_inpainting(distorted, mask, original_img, num_iters=60, lam=0.04, rho=10.0):
    y = torch.tensor(distorted, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    m = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    x = y.clone()
    z_x = torch.zeros_like(x)
    z_y = torch.zeros_like(x)
    u_x = torch.zeros_like(x)
    u_y = torch.zeros_like(x)
    
    lr = 1.0 / (1.0 + 8.0 * rho)
    psnr_history, ssim_history = [], []

    for i in range(num_iters):
        for _ in range(5):
            dx = diff_x(x)
            dy = diff_y(x)
            grad_x = m * (x - y) + rho * diff_x_T(dx - z_x + u_x) + rho * diff_y_T(dy - z_y + u_y)
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
        
        x_clipped = torch.clamp(x, 0, 1).squeeze().cpu().numpy()
        psnr_history.append(psnr(original_img, x_clipped, data_range=1.0))
        ssim_history.append(ssim(original_img, x_clipped, data_range=1.0))
        
    return x_clipped, psnr_history, ssim_history

def my_fista_inpainting(distorted, mask, original_img, num_iter=60, lambd=0.04):
    y = torch.tensor(distorted, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    m = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    x = y.clone()
    y_look = x.clone()
    t = 1.0
    step_size = 0.05
    psnr_history, ssim_history = [], []

    for i in range(num_iter):
        x_old = x.clone()
        
        div_tv = calc_div_tv(y_look)
        grad = m * (y_look - y) - lambd * div_tv
        x = y_look - step_size * grad
        
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y_look = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next
        
        x_clipped = torch.clamp(x, 0, 1).squeeze().cpu().numpy()
        psnr_history.append(psnr(original_img, x_clipped, data_range=1.0))
        ssim_history.append(ssim(original_img, x_clipped, data_range=1.0))
        
    return x_clipped, psnr_history, ssim_history

if __name__ == '__main__':
    image = img_as_float(data.camera())
    
    mask = np.ones_like(image)
    np.random.seed(42)  
    rand_mask = np.random.choice([0, 1], size=image.shape, p=[0.4, 0.6])
    mask *= rand_mask
    mask[100:110, 50:200] = 0
    mask[200:350, 400:410] = 0
    mask[400:410, 100:450] = 0
    
    distorted = image * mask
    
    lambd = 0.04
    num_iter = 60
    
    print("正在运行 老师的 FISTA+库函数TV...")
    rec_teacher, psnr_teacher, ssim_teacher = fista_tv_inpainting_teacher(
        distorted, mask, image, lambd=lambd, num_iter=num_iter)

    print("正在运行 老师的 FISTA+手写TV...")
    rec_teacher_c, psnr_teacher_c, ssim_teacher_c = fista_tv_inpainting_teacher_custom(
        distorted, mask, image, lambd=lambd, num_iter=num_iter)
        
    print("正在运行 你的旧版 ADMM (5步梯度下降版)...")
    rec_admm, psnr_admm, ssim_admm = my_admm_inpainting(
        distorted, mask, image, num_iters=num_iter, lam=lambd)
        
    print("正在运行 你的旧版 FISTA (次梯度直降法)...")
    rec_fista, psnr_fista, ssim_fista = my_fista_inpainting(
        distorted, mask, image, num_iter=num_iter, lambd=lambd)
        
    dist_psnr = psnr(image, distorted, data_range=1.0)
    dist_ssim = ssim(image, distorted, data_range=1.0)

    print("\n" + "="*60)
    print("最终结果对比 (迭代次数: 60)")
    print("="*60)
    print(f"损坏图像             - PSNR: {dist_psnr:.2f} dB, SSIM: {dist_ssim:.4f}")
    print(f"老师的FISTA(库函数TV) - PSNR: {psnr_teacher[-1]:.2f} dB, SSIM: {ssim_teacher[-1]:.4f}")
    print(f"老师的FISTA(手写TV)   - PSNR: {psnr_teacher_c[-1]:.2f} dB, SSIM: {ssim_teacher_c[-1]:.4f}")
    print(f"我的旧版 ADMM        - PSNR: {psnr_admm[-1]:.2f} dB, SSIM: {ssim_admm[-1]:.4f}")
    print(f"我的旧版 FISTA       - PSNR: {psnr_fista[-1]:.2f} dB, SSIM: {ssim_fista[-1]:.4f}")
    print("="*60)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(distorted, cmap='gray')
    plt.title(f"Corrupted\nPSNR: {dist_psnr:.2f}")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(rec_teacher, cmap='gray')
    plt.title(f"Teacher Library TV\nPSNR: {psnr_teacher[-1]:.2f}")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(rec_teacher_c, cmap='gray')
    plt.title(f"Teacher Custom TV\nPSNR: {psnr_teacher_c[-1]:.2f}")
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(rec_admm, cmap='gray')
    plt.title(f"My Old ADMM\nPSNR: {psnr_admm[-1]:.2f}")
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(rec_fista, cmap='gray')
    plt.title(f"My Old FISTA\nPSNR: {psnr_fista[-1]:.2f}")
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    iters = range(1, num_iter + 1)
    plt.plot(iters, psnr_teacher, 'r-', linewidth=2, label="Teacher Library TV")
    plt.plot(iters, psnr_teacher_c, 'g--', linewidth=2, label="Teacher Custom TV")
    plt.plot(iters, psnr_admm, 'b-', label="My Old ADMM")
    plt.plot(iters, psnr_fista, 'm:', label="My Old FISTA")
    plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Comparison (Inpainting)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('my_baseline_vs_teacher.png', dpi=150)
    print("\n=> 已保存对比图片至: my_baseline_vs_teacher.png")
