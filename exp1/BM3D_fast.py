import numpy as np
import os
from os import name
from skimage import io, img_as_float, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr
import bm3d
import matplotlib.pyplot as plt
import os
import requests
import zipfile
import time
from skimage.metrics import structural_similarity as ssim

def run_bm3d(image_path, sigmapsnr_std=37.42929):
    global sumnoise, sumdenoise, sumsmnoise, sumsmdenoise,filename
    img_clean = img_as_float(io.imread(image_path))
    noise_sigma = sigmapsnr_std / 255.0
    noise = np.random.normal(0, noise_sigma, img_clean.shape)
    img_noisy = np.clip(img_clean + noise, 0, 1)
    img_denoised = bm3d.bm3d(img_noisy, sigma_psd=noise_sigma)

    psnr_noisy = psnr(img_clean, img_noisy)
    psnr_denoised = psnr(img_clean, img_denoised)
    axis = -1 if img_clean.ndim == 3 else None
    score_noisy = ssim(img_clean, img_noisy, data_range=1.0, channel_axis=axis)
    score_denoised = ssim(img_clean, img_denoised, data_range=1.0, channel_axis=axis)

    name_base = os.path.splitext(filename)[0]
    noisy_name = f"{name_base}_sigma25_noisy.png"
    denoised_name = f"{name_base}_sigma25_denoised.png"
    img_denoised = np.clip(img_denoised, 0, 1)
    img_n_save = img_as_ubyte(img_noisy)
    img_d_save = img_ubyte = img_as_ubyte(img_denoised)
    io.imsave(os.path.join(output_dir, noisy_name), img_n_save)
    io.imsave(os.path.join(output_dir, denoised_name), img_d_save)
    return psnr_noisy, psnr_denoised ,score_noisy, score_denoised


start_time = time.time()

#---------------------------------------------------------------------

input_dir=r"C:\Users\19811\Desktop\set32"
output_dir = r"C:\Users\19811\Desktop\stm32b"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#---------------------------------------------------------------------
sumnoise=sumdenoise=sumsmnoise=sumsmdenoise=cnt=0

for filename in os.listdir(input_dir):
    full_path = os.path.join(input_dir,filename)
    if not os.path.isfile(full_path):
        continue
    noisy, denoised, smnoisy, smdenoised = run_bm3d(os.path.join(input_dir, filename))
    sumnoise += noisy
    sumdenoise += denoised
    sumsmnoise += smnoisy
    sumsmdenoise += smdenoised
    cnt+=1

end_time = time.time()
print(f"psnr:{sumnoise/cnt}, {sumdenoise/cnt}")
print(f"smmi:{sumsmnoise/cnt}, {sumsmdenoise/cnt}")
print(f"spend time:{end_time - start_time}")