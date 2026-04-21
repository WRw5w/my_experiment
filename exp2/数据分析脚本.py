import os
import cv2
import numpy as np
import pandas as pd
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric


# ====== 新增：安全图像读取函数 ======
def safe_imread(path):
    """安全读取图像：优先skimage，失败时回退到OpenCV"""
    try:
        return img_as_float(io.imread(path))
    except Exception as e_skimage:
        try:
            img_cv = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img_cv is None:
                raise ValueError("OpenCV无法解码此图像")

            # 归一化
            if img_cv.dtype == np.uint8:
                img = img_cv.astype(np.float32) / 255.0
            elif img_cv.dtype == np.uint16:
                img = img_cv.astype(np.float32) / 65535.0
            else:
                img = img_cv.astype(np.float32)
                if img.max() > 1.0:
                    img = img / img.max()

            # 颜色空间转换
            if img.ndim == 3:
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            return img
        except Exception:
            # 保留原始错误供诊断
            raise RuntimeError(
                f"读取失败: '{os.path.basename(path)}'\n"
                f"  提示: LZW压缩TIFF需安装imagecodecs → pip install imagecodecs"
            )


# ====== 通道处理保持不变（已修复） ======
def convert_to_gray(image):
    if image.ndim == 2:
        return image
    elif image.ndim == 3:
        ch = image.shape[2]
        if ch == 1:
            return image[:, :, 0]
        elif ch == 3:
            return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        elif ch == 4:
            rgb = image[:, :, :3]
            return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        else:
            return image[:, :, 0]
    else:
        raise ValueError(f"不支持的维度: {image.shape}")


def calculate_psnr_ssim(original_image, denoised_image):
    original_gray = convert_to_gray(original_image)
    denoised_gray = convert_to_gray(denoised_image)

    if original_gray.shape != denoised_gray.shape:
        denoised_gray = cv2.resize(
            denoised_gray,
            (original_gray.shape[1], original_gray.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    original_gray = np.clip(original_gray, 0, 1)
    denoised_gray = np.clip(denoised_gray, 0, 1)

    psnr = psnr_metric(original_gray, denoised_gray, data_range=1.0)
    ssim = ssim_metric(original_gray, denoised_gray, data_range=1.0, channel_axis=None)  # 显式指定2D

    return psnr, ssim


# ====== 增强错误处理的主流程 ======
def process_results(input_dir, original_dir):
    results = []
    skipped_images = []  # 记录跳过的图像

    for noise_config in os.listdir(input_dir):
        config_path = os.path.join(input_dir, noise_config)
        if not (os.path.isdir(config_path) and noise_config.startswith("SP_")):
            continue

        try:
            parts = noise_config.split('_')
            salt_pepper_prob = float(parts[1])
            gaussian_sigma = float(parts[2].replace('G', ''))
        except Exception as e:
            print(f"⚠️  无法解析噪声参数 '{noise_config}': {e}")
            continue

        # 收集所有图像文件
        image_files = set()
        for result_type in ['Noisy', 'ISTA_Result', 'BM3D_Result']:
            result_dir = os.path.join(config_path, result_type)
            if os.path.exists(result_dir):
                for f in os.listdir(result_dir):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        image_files.add(f)

        # 处理每张图像
        for filename in sorted(image_files):
            original_path = os.path.join(original_dir, filename)
            if not os.path.exists(original_path):
                skipped_images.append((filename, "原图不存在"))
                continue

            try:
                # 使用安全读取函数
                original_img = safe_imread(original_path)

                record = {
                    'Image': filename,
                    'SaltPepper_Prob': salt_pepper_prob,
                    'Gaussian_Sigma': gaussian_sigma
                }

                for result_type in ['Noisy', 'ISTA_Result', 'BM3D_Result']:
                    result_path = os.path.join(config_path, result_type, filename)
                    if os.path.exists(result_path):
                        denoised_img = safe_imread(result_path)
                        psnr, ssim = calculate_psnr_ssim(original_img, denoised_img)
                        record[f'PSNR_{result_type}'] = psnr
                        record[f'SSIM_{result_type}'] = ssim
                    else:
                        record[f'PSNR_{result_type}'] = np.nan
                        record[f'SSIM_{result_type}'] = np.nan

                results.append(record)

            except Exception as e:
                skipped_images.append((filename, str(e)))
                continue  # 跳过当前图像，继续处理

    # 打印跳过统计
    if skipped_images:
        print(f"\n⚠️  跳过 {len(skipped_images)} 张图像:")
        for i, (name, reason) in enumerate(skipped_images[:5]):  # 只显示前5个
            print(f"   {i + 1}. {name}: {reason[:60]}...")
        if len(skipped_images) > 5:
            print(f"   ... 还有 {len(skipped_images) - 5} 张图像未显示")

    return pd.DataFrame(results)


# ====== 其余函数保持不变（略） ======
# analyze_results() 和 main() 与之前相同，此处省略以节省空间

def main():
    input_dir = r"C:\Users\19811\Desktop\stm32b_SP"
    original_dir = r"C:\Users\19811\Desktop\set32"

    print("🚀 开始处理图像去噪结果...")
    df = process_results(input_dir, original_dir)

    if df.empty:
        print("❌ 错误: 未找到有效图像数据")
        return

    # 保存结果
    output_path = "denoising_results.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 结果已保存: {output_path} ({df.shape[0]} 张图像 × {df.shape[1]} 个指标)")

    # 数据分析
    analyze_results(df)


if __name__ == "__main__":
    main()