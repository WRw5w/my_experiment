import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl

# 设置专业学术图表风格（兼容所有 matplotlib 版本）
mpl.rcParams['font.family'] = 'Arial, DejaVu Sans, sans-serif'  # 跨平台字体
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.1


def generate_denoising_plots(summary_csv='denoising_summary.csv',
                             output_dir='analysis_plots'):
    """
    Generate publication-ready denoising performance plots with English labels
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取数据
    try:
        df = pd.read_csv(summary_csv)
    except FileNotFoundError:
        print(f"❌ Error: Summary file not found at '{summary_csv}'")
        print("   Please run the denoising analysis script first to generate denoising_summary.csv")
        return

    # 验证必要列
    required_cols = ['SaltPepper_Prob', 'Gaussian_Sigma',
                     'PSNR_ISTA_gain', 'PSNR_BM3D_gain']
    if not all(col in df.columns for col in required_cols):
        print("❌ Error: Missing required columns in summary file")
        print(f"   Required: {required_cols}")
        print(f"   Available: {list(df.columns)}")
        return

    # ====================== 图1: 高斯噪声强度对PSNR增益的影响 ======================
    fig, ax = plt.subplots(figsize=(10, 6))

    # 按SaltPepper_Prob分组绘制
    salt_pepper_probs = sorted(df['SaltPepper_Prob'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(salt_pepper_probs)))  # 避免过浅/过深

    # 绘制数据线
    for i, sp_prob in enumerate(salt_pepper_probs):
        subset = df[df['SaltPepper_Prob'] == sp_prob].sort_values('Gaussian_Sigma')

        # ISTA - 实线圆点
        ax.plot(subset['Gaussian_Sigma'], subset['PSNR_ISTA_gain'],
                marker='o',
                linestyle='-',
                color=colors[i],
                linewidth=2.5,
                markersize=8,
                label=f'ISTA (S&P={sp_prob:.3f})')

        # BM3D - 虚线方块
        ax.plot(subset['Gaussian_Sigma'], subset['PSNR_BM3D_gain'],
                marker='s',
                linestyle='--',
                color=colors[i],
                linewidth=2.0,
                markersize=7,
                label=f'BM3D (S&P={sp_prob:.3f})')

    # 设置坐标轴
    ax.set_xlabel('Gaussian Noise Standard Deviation ($\\sigma$)', labelpad=12, fontweight='bold')
    ax.set_ylabel('PSNR Gain (dB)', labelpad=12, fontweight='bold')
    ax.set_title('Impact of Gaussian Noise Intensity on Denoising Performance',
                 pad=18, fontweight='bold', fontsize=16)

    # 设置坐标轴范围（根据数据自动调整）
    ax.set_xlim(df['Gaussian_Sigma'].min() * 0.95, df['Gaussian_Sigma'].max() * 1.05)
    ax.set_ylim(0, df[['PSNR_ISTA_gain', 'PSNR_BM3D_gain']].max().max() * 1.15)

    # 添加专业网格
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
    ax.set_axisbelow(True)  # 网格线在数据线下方

    # 添加数据标签（仅标注ISTA，避免拥挤）
    for _, row in df.iterrows():
        ax.annotate(f"{row['PSNR_ISTA_gain']:.2f}",
                    (row['Gaussian_Sigma'], row['PSNR_ISTA_gain']),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha='center',
                    fontsize=9,
                    color='darkblue',
                    fontweight='bold')

    # 创建紧凑图例
    ax.legend(loc='upper right',
              frameon=True,
              framealpha=0.95,
              shadow=True,
              ncol=2,  # 两列布局避免过长
              fontsize=10)

    # 优化布局
    plt.tight_layout()

    # 保存高分辨率图像
    plt.savefig(os.path.join(output_dir, 'gaussian_noise_effect.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'gaussian_noise_effect.pdf'))
    plt.close()
    print(f"✅ Plot 1 saved: {output_dir}/gaussian_noise_effect.png (300 DPI)")

    # ====================== 图2: 椒盐噪声概率对PSNR增益的影响 ======================
    fig, ax = plt.subplots(figsize=(10, 6))

    # 按Gaussian_Sigma分组绘制
    gaussian_sigmas = sorted(df['Gaussian_Sigma'].unique())
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(gaussian_sigmas)))

    for i, g_sigma in enumerate(gaussian_sigmas):
        subset = df[df['Gaussian_Sigma'] == g_sigma].sort_values('SaltPepper_Prob')

        # ISTA
        ax.plot(subset['SaltPepper_Prob'], subset['PSNR_ISTA_gain'],
                marker='o',
                linestyle='-',
                color=colors[i],
                linewidth=2.5,
                markersize=8,
                label=f'ISTA ($\\sigma$={g_sigma:.3f})')

        # BM3D
        ax.plot(subset['SaltPepper_Prob'], subset['PSNR_BM3D_gain'],
                marker='s',
                linestyle='--',
                color=colors[i],
                linewidth=2.0,
                markersize=7,
                label=f'BM3D ($\\sigma$={g_sigma:.3f})')

    # 设置坐标轴
    ax.set_xlabel('Salt & Pepper Noise Probability', labelpad=12, fontweight='bold')
    ax.set_ylabel('PSNR Gain (dB)', labelpad=12, fontweight='bold')
    ax.set_title('Impact of Salt & Pepper Noise Probability on Denoising Performance',
                 pad=18, fontweight='bold', fontsize=16)

    # 设置坐标轴范围
    ax.set_xlim(df['SaltPepper_Prob'].min() * 0.9, df['SaltPepper_Prob'].max() * 1.1)
    ax.set_ylim(0, df[['PSNR_ISTA_gain', 'PSNR_BM3D_gain']].max().max() * 1.15)

    # 添加专业网格
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
    ax.set_axisbelow(True)

    # 添加数据标签
    for _, row in df.iterrows():
        ax.annotate(f"{row['PSNR_ISTA_gain']:.2f}",
                    (row['SaltPepper_Prob'], row['PSNR_ISTA_gain']),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha='center',
                    fontsize=9,
                    color='darkblue',
                    fontweight='bold')

    # 创建紧凑图例
    ax.legend(loc='upper right',
              frameon=True,
              framealpha=0.95,
              shadow=True,
              ncol=2,
              fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'salt_pepper_effect.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'salt_pepper_effect.pdf'))
    plt.close()
    print(f"✅ Plot 2 saved: {output_dir}/salt_pepper_effect.png (300 DPI)")

    # ====================== 生成论文用数据表格 ======================
    # 创建格式化表格
    table_df = df[['SaltPepper_Prob', 'Gaussian_Sigma',
                   'PSNR_ISTA_gain', 'PSNR_BM3D_gain']].copy()
    table_df = table_df.sort_values(['SaltPepper_Prob', 'Gaussian_Sigma']).reset_index(drop=True)

    # 保存为CSV（英文表头）
    table_df.columns = ['SaltPepper_Probability', 'Gaussian_Sigma',
                        'ISTA_PSNR_Gain_dB', 'BM3D_PSNR_Gain_dB']
    table_df.to_csv(os.path.join(output_dir, 'denoising_results_table.csv'),
                    index=False, float_format='%.4f')
    print(f"✅ Data table saved: {output_dir}/denoising_results_table.csv")

    # 保存为LaTeX表格
    latex_header = ['S\\&P Prob.', 'Gaussian $\\sigma$', 'ISTA Gain (dB)', 'BM3D Gain (dB)']
    latex_df = table_df.copy()
    latex_df.columns = latex_header

    latex_table = latex_df.to_latex(index=False,
                                    float_format="%.2f",
                                    column_format='cccc',
                                    escape=False)

    # 添加表格标题和标签
    latex_table = latex_table.replace(
        '\\begin{tabular}',
        '\\begin{table}[htbp]\n\\centering\n\\caption{Denoising performance comparison under different noise configurations}\n\\label{tab:denoising_results}\n\\begin{tabular}'
    ).replace(
        '\\end{tabular}',
        '\\bottomrule\n\\end{tabular}\n\\end{table}'
    ).replace(
        '\\midrule', '\\midrule\n\\toprule'
    )

    # 添加booktabs包支持
    latex_table = '\\usepackage{booktabs}\n' + latex_table

    with open(os.path.join(output_dir, 'denoising_results_table.tex'), 'w') as f:
        f.write(latex_table)
    print(f"✅ LaTeX table saved: {output_dir}/denoising_results_table.tex")

    # ====================== 打印分析摘要 ======================
    print("\n" + "=" * 70)
    print("📊 ACADEMIC ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"✓ Generated 2 publication-ready figures (PNG + PDF, 300 DPI)")
    print(f"✓ Created data table for results section (CSV + LaTeX)")
    print(f"✓ All labels and captions in professional English")
    print(f"✓ Output directory: {os.path.abspath(output_dir)}")
    print("\n📌 Recommended usage for academic paper:")
    print("   • Insert PDF figures for vector-quality graphics")
    print("   • Use LaTeX table directly in \\begin{table} environment")
    print("   • Caption suggestion for Fig.1:")
    print("     'PSNR gain versus Gaussian noise intensity for different")
    print("      salt & pepper noise probabilities. Solid lines with circles")
    print("      represent ISTA; dashed lines with squares represent BM3D.'")
    print("=" * 70)


if __name__ == "__main__":
    generate_denoising_plots()