import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import cv2
from skimage import io, img_as_float, color
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import time
import numpy as np

# 导入你的算法模块
import BM3D_test
import ISTA_test

# ==========================================
# 1. 配置路径
# ==========================================
input_dir = r"C:\Users\19811\Desktop\set32"  # 输入文件夹
output_dir = r"C:\Users\19811\Desktop\stm32b_SP"  # 输出保存的大文件夹 (建议改个名区分高斯实验)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")


# ==========================================
# 2. 辅助函数
# ==========================================

def add_salt_pepper_noise(x, prob=0.05):
    """
    添加椒盐噪声
    prob: 噪声比例 (0 < prob < 1)
    """
    x_noisy = x.clone()
    noise_tensor = torch.rand_like(x_noisy)

    # 椒 (Pepper) -> 0 (黑色)
    x_noisy[noise_tensor < prob / 2] = 0.0
    # 盐 (Salt) -> 1 (白色)
    x_noisy[noise_tensor > 1 - prob / 2] = 1.0

    return x_noisy


def add_gaussian_noise(x, sigma=0.25):
    """
    添加高斯噪声
    sigma: 噪声强度
    """
    noise = torch.randn_like(x) * sigma
    return x + noise


def save_images(save_root, filename, prob, noisy_t, ista_t, bm3d_t, sigma):
    """
    保存图像
    结构: save_root / SP_0.05 / [Noisy, ISTA, BM3D] / filename
    """
    # 1. 定义当前噪声浓度下的文件夹
    # prob_dir 例如: "SP_0.05"
    prob_dir = os.path.join(save_root, f"SP_{prob}_G{sigma}")

    path_noisy = os.path.join(prob_dir, "Noisy")
    path_ista = os.path.join(prob_dir, "ISTA_Result")
    path_bm3d = os.path.join(prob_dir, "BM3D_Result")

    # 2. 创建文件夹 (一共 3个浓度 * 3种类型 = 9个子文件夹)
    for p in [path_noisy, path_ista, path_bm3d]:
        os.makedirs(p, exist_ok=True)

    # 3. 转换并保存
    def to_uint8(tensor):
        arr = tensor.squeeze().cpu().detach().numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return arr

    cv2.imwrite(os.path.join(path_noisy, filename), to_uint8(noisy_t))
    cv2.imwrite(os.path.join(path_ista, filename), to_uint8(ista_t))
    cv2.imwrite(os.path.join(path_bm3d, filename), to_uint8(bm3d_t))


# ==========================================
# 3. 核心运行逻辑
# ==========================================
def run_compare(image_path, filename, prob=0.05, sigma=0.25):
    # --- 1. 读取图像 (包含之前的 Bug 修复) ---
    if not os.path.exists(image_path):
        return None
    try:
        imag = io.imread(image_path)
    except Exception as e:
        return None

    # 处理通道
    if imag.ndim == 3:
        channels = imag.shape[2]
        if channels == 3:
            imag = color.rgb2gray(imag)
        elif channels == 4:
            imag = color.rgb2gray(imag[:, :, :3])
        elif channels == 2:
            imag = imag[:, :, 0]

    # 转 Tensor
    image = torch.tensor(img_as_float(imag), dtype=torch.float32).to(device)
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        image = image.unsqueeze(0)

    # --- 2. 添加椒盐噪声 ---
    image_noise_sp = add_salt_pepper_noise(image, prob=prob)

    # --- 3. 添加高斯噪声 ---
    image_noise_gaussian = add_gaussian_noise(image, sigma=sigma)

    # --- 4. 计算加噪后的初始 PSNR (Noisy PSNR) ---
    img_np = image.squeeze().cpu().numpy()
    noisy_sp_np = image_noise_sp.squeeze().cpu().numpy()
    noisy_gaussian_np = image_noise_gaussian.squeeze().cpu().numpy()

    # 确保范围在 0-1
    noisy_sp_np = np.clip(noisy_sp_np, 0, 1)
    noisy_gaussian_np = np.clip(noisy_gaussian_np, 0, 1)

    psnr_noisy_sp = psnr_metric(img_np, noisy_sp_np, data_range=1.0)
    psnr_noisy_gaussian = psnr_metric(img_np, noisy_gaussian_np, data_range=1.0)

    # --- 5. 运行算法 ---
    # BM3D
    try:
        bm3d_solver = BM3D_test.BM3D_GPU_Solver(sigma=50)
    except AttributeError:
        bm3d_solver = BM3D_test.BM3D()

    image_clean_BM3D_sp = bm3d_solver.denoise(image_noise_sp)
    image_clean_BM3D_gaussian = bm3d_solver.denoise(image_noise_gaussian)

    # ISTA (增加 lambda 以应对强噪声点)
    image_clean_ISTA_sp = ISTA_test.run_ISTA(image_noise_sp, lam=0.15)
    image_clean_ISTA_gaussian = ISTA_test.run_ISTA(image_noise_gaussian, lam=0.15)

    # --- 6. 保存结果 ---
    save_images(output_dir, filename, prob, image_noise_sp, image_clean_ISTA_sp, image_clean_BM3D_sp, sigma)
    save_images(output_dir, filename, prob, image_noise_gaussian, image_clean_ISTA_gaussian, image_clean_BM3D_gaussian,
                sigma)

    # --- 7. 计算去噪后的指标 ---
    clean_ista_sp_np = np.clip(image_clean_ISTA_sp.squeeze().cpu().numpy(), 0, 1)
    clean_bm3d_sp_np = np.clip(image_clean_BM3D_sp.squeeze().cpu().numpy(), 0, 1)

    clean_ista_gaussian_np = np.clip(image_clean_ISTA_gaussian.squeeze().cpu().numpy(), 0, 1)
    clean_bm3d_gaussian_np = np.clip(image_clean_BM3D_gaussian.squeeze().cpu().numpy(), 0, 1)

    p_ista_sp = psnr_metric(img_np, clean_ista_sp_np, data_range=1.0)
    s_ista_sp = ssim_metric(img_np, clean_ista_sp_np, data_range=1.0)

    p_bm3d_sp = psnr_metric(img_np, clean_bm3d_sp_np, data_range=1.0)
    s_bm3d_sp = ssim_metric(img_np, clean_bm3d_sp_np, data_range=1.0)

    p_ista_gaussian = psnr_metric(img_np, clean_ista_gaussian_np, data_range=1.0)
    s_ista_gaussian = ssim_metric(img_np, clean_ista_gaussian_np, data_range=1.0)

    p_bm3d_gaussian = psnr_metric(img_np, clean_bm3d_gaussian_np, data_range=1.0)
    s_bm3d_gaussian = ssim_metric(img_np, clean_bm3d_gaussian_np, data_range=1.0)

    return psnr_noisy_sp, p_ista_sp, s_ista_sp, p_bm3d_sp, s_bm3d_sp, psnr_noisy_gaussian, p_ista_gaussian, s_ista_gaussian, p_bm3d_gaussian, s_bm3d_gaussian


# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 定义椒盐噪声概率: 5%, 10%, 20%
    sp_probs = [0.02, 0.037, 0.074]
    # 定义高斯噪声强度: 0.25, 0.37929, 0.42
    gaussian_sigmas = [0.25, 0.37929, 0.42]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for prob in sp_probs:
        for sigma in gaussian_sigmas:
            print(f"\n=== Testing S&P Noise Probability: {prob}, Gaussian Noise Sigma: {sigma} ===")
            print(f"Saving to: {os.path.join(output_dir, f'SP_{prob}_G{sigma}')}")

            # 累加器
            sum_psnr_noisy_sp = 0  # 新增: 噪声图的PSNR
            sum_psnr_ista_sp = 0
            sum_ssim_ista_sp = 0
            sum_psnr_bm3d_sp = 0
            sum_ssim_bm3d_sp = 0

            sum_psnr_noisy_gaussian = 0
            sum_psnr_ista_gaussian = 0
            sum_ssim_ista_gaussian = 0
            sum_psnr_bm3d_gaussian = 0
            sum_ssim_bm3d_gaussian = 0
            cnt = 0

            loop_start = time.time()

            file_list = os.listdir(input_dir)
            valid_files = [f for f in file_list if f.endswith(('.png', '.jpg', '.bmp', '.tif'))]

            for filename in valid_files:
                full_path = os.path.join(input_dir, filename)

                res = run_compare(full_path, filename, prob=prob, sigma=sigma)

                if res is not None:
                    # S&P 噪声
                    p_noisy_sp, p_i_sp, s_i_sp, p_b_sp, s_b_sp, _, _, _, _, _ = res
                    # 高斯噪声
                    _, _, _, _, _, p_noisy_gaussian, p_i_gaussian, s_i_gaussian, p_b_gaussian, s_b_gaussian = res

                    sum_psnr_noisy_sp += p_noisy_sp
                    sum_psnr_ista_sp += p_i_sp
                    sum_ssim_ista_sp += s_i_sp
                    sum_psnr_bm3d_sp += p_b_sp
                    sum_ssim_bm3d_sp += s_b_sp

                    sum_psnr_noisy_gaussian += p_noisy_gaussian
                    sum_psnr_ista_gaussian += p_i_gaussian
                    sum_ssim_ista_gaussian += s_i_gaussian
                    sum_psnr_bm3d_gaussian += p_b_gaussian
                    sum_ssim_bm3d_gaussian += s_b_gaussian
                    cnt += 1

                    # 打印单张图片结果: 增加了 Noisy PSNR 的显示
                    print(
                        f"  > {filename}: Noisy(SP)={p_noisy_sp:.2f}dB | ISTA(SP)={p_i_sp:.2f}dB | BM3D(SP)={p_b_sp:.2f}dB")
                    print(
                        f"  > {filename}: Noisy(Gaussian)={p_noisy_gaussian:.2f}dB | ISTA(Gaussian)={p_i_gaussian:.2f}dB | BM3D(Gaussian)={p_b_gaussian:.2f}dB")

            if cnt > 0:
                avg_p_noisy_sp = sum_psnr_noisy_sp / cnt
                avg_p_ista_sp = sum_psnr_ista_sp / cnt
                avg_s_ista_sp = sum_ssim_ista_sp / cnt
                avg_p_bm3d_sp = sum_psnr_bm3d_sp / cnt
                avg_s_bm3d_sp = sum_ssim_bm3d_sp / cnt

                avg_p_noisy_gaussian = sum_psnr_noisy_gaussian / cnt
                avg_p_ista_gaussian = sum_psnr_ista_gaussian / cnt
                avg_s_ista_gaussian = sum_ssim_ista_gaussian / cnt
                avg_p_bm3d_gaussian = sum_psnr_bm3d_gaussian / cnt
                avg_s_bm3d_gaussian = sum_ssim_bm3d_gaussian / cnt

                print("-" * 60)
                print(f"Results for S&P Prob {prob}, Gaussian Sigma {sigma}:")
                print(
                    f"Average Noisy (SP): {avg_p_noisy_sp:.4f} dB | Average Noisy (Gaussian): {avg_p_noisy_gaussian:.4f} dB")
                print(f"ISTA(SP) -> PSNR: {avg_p_ista_sp:.4f}, SSIM: {avg_s_ista_sp:.4f}")
                print(f"BM3D(SP) -> PSNR: {avg_p_bm3d_sp:.4f}, SSIM: {avg_s_bm3d_sp:.4f}")
                print(f"ISTA(Gaussian) -> PSNR: {avg_p_ista_gaussian:.4f}, SSIM: {avg_s_ista_gaussian:.4f}")
                print(f"BM3D(Gaussian) -> PSNR: {avg_p_bm3d_gaussian:.4f}, SSIM: {avg_s_bm3d_gaussian:.4f}")
                print(f"Time: {time.time() - loop_start:.2f}s")
