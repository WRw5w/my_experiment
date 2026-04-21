# 嗯,主函数要有以下几个部分:1.配置,2.初始化图像,3.进行实验,4.输出结果

# ----------------------------配置
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
# science toolket
from skimage import io, img_as_float, color
from skimage.metrics import peak_signal_noise_ratio as psnr
import time
import numpy as np
import BM3D_test
import ISTA_test

BM3D_ = BM3D()
input_dir = r"C:\Users\19811\Desktop\set32"
output_dir = r"C:\Users\19811\Desktop\stm32b"
path = r"C:\Users\19811\Desktop\123"
# raw,不然出错
start_time = time.time()
# woc,原来要在gpu上面1运行要选择device呀
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")


# ---------------------------控制噪声的添加
def add_noisy(x, device="gpu"):
    # 噪声参数
    goss = [25, 37.42929, 42]
    jiaoyan = 0.02
    for i in range(3):
        noise = torch.normal(mean=0, std=goss[i] / 255.0, size=x.shape).to(device)
        x_noisy = x + noise
        prob = torch.rand(x.shape).to(device)
        x_noisy[prob < jiaoyan / 2] = 0
        x_noisy[prob > 1 - jiaoyan / 2] = 1
        yield x_noisy


# ----------------------------bm3d的调用以及计算结果如何
def run_compare(image_path, sigma=25):
    # 简单处理, 先不考虑彩色图了
    imag = io.imread(image_path)
    if imag.ndim == 3:
        imag = color.rgb2gray(imag)
    # 只考虑二维的情况
    image = torch.tensor(img_as_float(imag), dtype=torch.float32).to(device)
    image_noise = image.clone().to(device)

    # 添加噪声并进行去噪
    image_noise = add_noisy(image_noise)

    # 调用 BM3D 和 ISTA 进行去噪
    image_clean_BM3D = BM3D_test.BM3D.denoise(image_noise)
    image_clean_ISTA = ISTA_test.run_ISTA(image_noise)

    # 转回 CPU 进行 PSNR 计算
    img_np = image.cpu().numpy()
    noisy_np = image_noise.cpu().numpy()
    clean_np = image_clean.cpu().numpy()
    clean_np = np.clip(clean_np, 0, 1)
    noisy_np = np.clip(noisy_np, 0, 1)

    return psnr(img_np, noisy_np), psnr(img_np, clean_np)


# 主函数
for i in range(9):
    sumnoise = sumdenoise = sumsmnoise = sumsmdenoise = cnt = 0
    sumnoise_BM3D = sumdenoise_BM3D = sumsmnoise_BM3D = sumsmdenoise_BM3D = cnt = 0

    for filename in os.listdir(input_dir):
        full_path = os.path.join(input_dir, filename)
        if not os.path.isfile(full_path):
            continue

        noisy, denoised, smnoisy, smdenoised, noisy_BM3D, denoised_BM3D, smnoisy_BM3D, smdenoised_BM3D = run_compare(
            os.path.join(input_dir, filename))

        sumnoise += noisy
        sumdenoise += denoised
        sumsmnoise += smnoisy
        sumsmdenoise += smdenoised
        sumnoise_BM3D += noisy_BM3D
        sumsmnoise_BM3D += denoised_BM3D
        sumsmnoise_BM3D += smnoisy_BM3D
        sumsmdenoise_BM3D += smdenoised_BM3D
        cnt += 1

    end_time = time.time()

    print("ISTA", end=':')
    print(f"psnr: {sumnoise / cnt}, {sumdenoise / cnt}")
    print(f"smmi: {sumsmnoise / cnt}, {sumsmdenoise / cnt}")

    print("BM3D", end=':')
    print(f"psnr: {sumnoise_BM3D / cnt}, {sumsmnoise_BM3D / cnt}")
    print(f"smmi: {sumsmnoise_BM3D / cnt}, {sumsmdenoise_BM3D / cnt}")
    print(f"spend time: {end_time - start_time}")
