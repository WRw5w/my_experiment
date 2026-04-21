import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sigmas = [15, 25, 35, 50]
# Placeholder values — update after running evaluate.py
gray_psnr = [0, 0, 0, 0]
color_psnr = [0, 0, 0, 0]

plt.figure(figsize=(8, 5))
plt.plot(sigmas, gray_psnr, 'o-', linewidth=2, label='N2V (Gray)')
plt.plot(sigmas, color_psnr, 's-', linewidth=2, label='N2V (Color)')
plt.xlabel(r"Noise Level ($\sigma$)")
plt.ylabel("PSNR (dB)")
plt.title("Noise2Void Denoising Generalization Capability")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/combined_psnr_vs_sigma.png', dpi=150, bbox_inches='tight')
print("Generated combined plot.")
