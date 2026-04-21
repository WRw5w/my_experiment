import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sigmas = [15, 25, 35, 50]
gray_psnr = [31.05, 29.60, 25.55, 20.07]
color_psnr = [28.64, 25.86, 23.14, 19.81]

plt.figure(figsize=(8, 5))
plt.plot(sigmas, gray_psnr, 'o-', linewidth=2, label='U-Net (Gray)')
plt.plot(sigmas, color_psnr, 's-', linewidth=2, label='U-Net (Color)')
plt.xlabel(r"Noise Level ($\sigma$)")
plt.ylabel("PSNR (dB)")
plt.title("U-Net Denoising Generalization Capability")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/combined_psnr_vs_sigma.png', dpi=150, bbox_inches='tight')
print("Generated combined plot.")
