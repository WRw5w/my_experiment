import os
import matplotlib.pyplot as plt
import numpy as np

# 结果数据 (从 metrics_summary.txt 动态读取)
output_dir = r'D:\02_Projects\ML\zhangchengxi_BM3D\exp5\outputs'
summary_path = os.path.join(output_dir, "metrics_summary.txt")

methods = []
psnr_values = []
ssim_values = []
time_values = []

if os.path.exists(summary_path):
    with open(summary_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if '|' in line and 'Method' not in line and 'Noisy' not in line and '---' not in line and '=' not in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 4:
                    methods.append(parts[0])
                    psnr_values.append(float(parts[1]))
                    ssim_values.append(float(parts[2]))
                    time_values.append(float(parts[3]))
else:
    print(f"Warning: {summary_path} not found. Using fallback data.")
    methods = ['ADMM', 'ISTA', 'FISTA', 'BM3D', 'DnCNN']
    psnr_values = [27.75, 28.52, 28.53, 29.70, 29.99]
    ssim_values = [0.6438, 0.7216, 0.7221, 0.7963, 0.8148]
    time_values = [0.2953, 2.1029, 1.5234, 10.9503, 0.9677]

# 设置绘图风格
plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 1. 绘制 PSNR 和 SSIM 的对比 (双Y轴柱状图)
x = np.arange(len(methods))
width = 0.35

rects1 = ax1.bar(x - width/2, psnr_values, width, label='PSNR (dB)', color='skyblue', edgecolor='black')
ax1.set_ylabel('PSNR (dB)', fontweight='bold')
ax1.set_title('Quality Comparison: PSNR & SSIM', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.set_ylim(25, 31) # 聚焦差异
ax1.legend(loc='upper left')

# 创建次坐标轴用于 SSIM
ax1_twin = ax1.twinx()
rects2 = ax1_twin.bar(x + width/2, ssim_values, width, label='SSIM', color='orange', alpha=0.7, edgecolor='black')
ax1_twin.set_ylabel('SSIM', fontweight='bold')
ax1_twin.set_ylim(0.5, 0.9)
ax1_twin.legend(loc='upper right')

# 2. 绘制运行时间对比 (对数坐标柱状图，因为差异巨大)
ax2.bar(methods, time_values, color='salmon', edgecolor='black')
ax2.set_yscale('log') # 使用对数坐标以便观察巨大的时间跨度
ax2.set_ylabel('Time (s) - Log Scale', fontweight='bold')
ax2.set_title('Efficiency Comparison: Execution Time', fontsize=14, fontweight='bold')

# 在柱状图上方标注数值
for i, v in enumerate(time_values):
    ax2.text(i, v, f'{v:.2f}s', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

# 保存图表
output_path = r'D:\02_Projects\ML\zhangchengxi_BM3D\exp5\outputs\performance_charts.jpg'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Charts (JPG) saved to: {output_path}")
plt.show()
