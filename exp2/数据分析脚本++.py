import os
import cv2
import numpy as np
import pandas as pd
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric


# ==================== 安全图像读取（兼容LZW压缩） ====================
def safe_imread(path):
    """优先skimage，失败回退OpenCV"""
    try:
        return img_as_float(io.imread(path))
    except Exception:
        img_cv = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_cv is None:
            raise RuntimeError(f"无法读取图像: {os.path.basename(path)}")

        # 归一化
        if img_cv.dtype == np.uint8:
            img = img_cv.astype(np.float32) / 255.0
        elif img_cv.dtype == np.uint16:
            img = img_cv.astype(np.float32) / 65535.0
        else:
            img = img_cv.astype(np.float32)
            if img.max() > 1.0:
                img = img / img.max()

        # BGR→RGB转换
        if img.ndim == 3 and img.shape[2] in (3, 4):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if img.shape[2] == 3 else cv2.COLOR_BGRA2RGBA)
        return img


# ==================== 通道统一处理（灰度化） ====================
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


# ==================== PSNR/SSIM计算 ====================
def calculate_psnr_ssim(original_image, denoised_image):
    orig_gray = convert_to_gray(original_image)
    denoised_gray = convert_to_gray(denoised_image)

    if orig_gray.shape != denoised_gray.shape:
        denoised_gray = cv2.resize(denoised_gray, (orig_gray.shape[1], orig_gray.shape[0]))

    orig_gray = np.clip(orig_gray, 0, 1)
    denoised_gray = np.clip(denoised_gray, 0, 1)

    psnr = psnr_metric(orig_gray, denoised_gray, data_range=1.0)
    ssim = ssim_metric(orig_gray, denoised_gray, data_range=1.0, channel_axis=None)
    return psnr, ssim


# ==================== 图像级数据采集 ====================
def collect_image_metrics(input_dir, original_dir):
    records = []
    skipped = 0

    for noise_config in sorted(os.listdir(input_dir)):
        config_path = os.path.join(input_dir, noise_config)
        if not (os.path.isdir(config_path) and noise_config.startswith("SP_")):
            continue

        # 解析噪声参数: SP_0.05_G0.25 → salt=0.05, gauss=0.25
        try:
            _, sp_str, g_str = noise_config.split('_')
            sp_prob = float(sp_str)
            g_sigma = float(g_str.replace('G', ''))
        except:
            continue

        # 收集该配置下所有图像
        image_files = set()
        for stage in ['Noisy', 'ISTA_Result', 'BM3D_Result']:
            stage_dir = os.path.join(config_path, stage)
            if os.path.exists(stage_dir):
                for f in os.listdir(stage_dir):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        image_files.add(f)

        # 处理每张图像
        for img_name in image_files:
            orig_path = os.path.join(original_dir, img_name)
            if not os.path.exists(orig_path):
                skipped += 1
                continue

            try:
                orig_img = safe_imread(orig_path)
                record = {
                    'Image': img_name,
                    'SaltPepper_Prob': sp_prob,
                    'Gaussian_Sigma': g_sigma
                }

                for stage in ['Noisy', 'ISTA_Result', 'BM3D_Result']:
                    stage_path = os.path.join(config_path, stage, img_name)
                    if os.path.exists(stage_path):
                        denoised_img = safe_imread(stage_path)
                        psnr, ssim = calculate_psnr_ssim(orig_img, denoised_img)
                        record[f'PSNR_{stage}'] = psnr
                        record[f'SSIM_{stage}'] = ssim
                    else:
                        record[f'PSNR_{stage}'] = np.nan
                        record[f'SSIM_{stage}'] = np.nan

                records.append(record)
            except:
                skipped += 1
                continue

    if skipped:
        print(f"⚠️  跳过 {skipped} 张图像（文件缺失或读取失败）")

    return pd.DataFrame(records)


# ==================== 聚合为组平均（核心功能） ====================
def aggregate_by_noise_config(df):
    """将每组噪声配置聚合为单行平均值 + 增益分析"""
    # 分组计算均值
    agg = df.groupby(['SaltPepper_Prob', 'Gaussian_Sigma']).agg({
        'PSNR_Noisy': 'mean',
        'SSIM_Noisy': 'mean',
        'PSNR_ISTA_Result': 'mean',
        'SSIM_ISTA_Result': 'mean',
        'PSNR_BM3D_Result': 'mean',
        'SSIM_BM3D_Result': 'mean'
    }).round(4).reset_index()

    # 计算增益指标（关键分析维度）
    agg['PSNR_ISTA_gain'] = (agg['PSNR_ISTA_Result'] - agg['PSNR_Noisy']).round(4)  # ISTA相对噪声提升
    agg['PSNR_BM3D_gain'] = (agg['PSNR_BM3D_Result'] - agg['PSNR_Noisy']).round(4)  # BM3D相对噪声提升
    agg['PSNR_ISTA_vs_BM3D'] = (agg['PSNR_ISTA_Result'] - agg['PSNR_BM3D_Result']).round(4)  # ISTA相对BM3D优势

    # 按噪声强度排序（便于观察趋势）
    agg = agg.sort_values(['SaltPepper_Prob', 'Gaussian_Sigma']).reset_index(drop=True)

    return agg


# ==================== 主流程 ====================
def main():
    input_dir = r"C:\Users\19811\Desktop\stm32b_SP"  # 去噪结果目录
    original_dir = r"C:\Users\19811\Desktop\set32"  # 原图目录

    print("📊 正在计算各噪声配置下的平均去噪性能...\n")

    # 1. 采集图像级数据
    df_detail = collect_image_metrics(input_dir, original_dir)
    if df_detail.empty:
        print("❌ 未找到有效图像数据")
        return

    # 2. 聚合为组平均（核心输出）
    df_summary = aggregate_by_noise_config(df_detail)

    # 3. 保存聚合结果（主分析文件）
    summary_path = "denoising_summary.csv"
    df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')

    # 4. 控制台输出完整表格（无省略）
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("=" * 100)
    print("✅ 去噪性能汇总表（每行 = 一种噪声配置的平均指标）")
    print("=" * 100)
    print(df_summary.to_string(index=False))
    print("=" * 100)
    print(f"\n💾 已保存至: {os.path.abspath(summary_path)}")
    print(f"   行数: {len(df_summary)} (噪声配置数) × 列数: {len(df_summary.columns)} (指标数)")
    print("\n📌 列说明:")
    print("   • SaltPepper_Prob / Gaussian_Sigma: 噪声强度参数")
    print("   • PSNR_Noisy / SSIM_Noisy: 噪声图像质量")
    print("   • PSNR_ISTA_Result / SSIM_ISTA_Result: ISTA去噪后质量")
    print("   • PSNR_BM3D_Result / SSIM_BM3D_Result: BM3D去噪后质量")
    print("   • PSNR_ISTA_gain: ISTA相对噪声的PSNR提升（越大越好）")
    print("   • PSNR_BM3D_gain: BM3D相对噪声的PSNR提升（越大越好）")
    print("   • PSNR_ISTA_vs_BM3D: ISTA相对BM3D的PSNR差值（>0表示ISTA更优）")

    # 5. 可选：保存详细数据（每张图像一行）
    detail_path = "denoising_detail.csv"
    df_detail.to_csv(detail_path, index=False, encoding='utf-8-sig')
    print(f"   💡 详细数据（每图像一行）已保存至: {detail_path}")


if __name__ == "__main__":
    main()