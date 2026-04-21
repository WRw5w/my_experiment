import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_comparison_plot(methods, image_np, psnr_h_dict, ssim_h_dict, filename="denoising_comparison.pdf", save_dir="."):
    """
    绘制所有的去噪结果图像以及 PSNR/SSIM 随迭代次数变化的曲线图，
    并将其保存为 PDF 文件。
    
    参数:
        methods: 列表，格式为 [(名称, 图像numpy数组, 最终PSNR, 最终SSIM), ...]
        image_np: 完美的原始图像(Ground Truth)，用于展示
        psnr_h_dict: 字典，格式为 {算法名称: [PSNR历史列表], ...}
        ssim_h_dict: 字典，格式为 {算法名称: [SSIM历史列表], ...}
        filename: 保存的文件名，默认 'denoising_comparison.pdf'
        save_dir: 保存路径，默认为当前路径
    """
    
    # 确保保存的文件夹存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, filename)
    print(f"正在保存绘图至: {save_path}")
        
    num_methods = len(methods)
    
    fig = plt.figure(figsize=(18, 10))
    
    # === 绘制顶部的去噪效果图对比 (最多展示前5个) ===
    for i in range(min(5, num_methods)):
        ax = plt.subplot(2, 4, i+1)
        name, img, p, s = methods[i]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"{name}\nPSNR: {p:.2f} dB\nSSIM: {s:.4f}", fontsize=11)
        ax.axis('off')
        
    # === 原图 ===
    ax_org = plt.subplot(2, 4, 6)
    ax_org.imshow(image_np, cmap='gray')
    ax_org.set_title("Original (Ground Truth)", fontsize=11)
    ax_org.axis('off')
    
    # 取随算法共同最大迭代长度作为X轴
    max_iters = max([len(h) for h in psnr_h_dict.values()] + [1])
    iteration_range = range(1, max_iters + 1)
    
    # 预设一下颜色和线型
    styles = {
        'ISTA': ('g--', 1.5),
        'FISTA': ('r-', 1.5),
        'AMDD': ('b-', 2.0),
        'BM3D_Baseline': ('k:', 1.5)
    }

    # === 绘制 PSNR 曲线 ===
    ax_psnr = plt.subplot(2, 4, 7)
    for name, history in psnr_h_dict.items():
        if len(history) == 1: # 类似BM3D这种只有一个点画基准线
            ax_psnr.axhline(history[0], color='k', linestyle=':', label=name)
        else:
            style, width = styles.get(name, ('m-.', 1.5)) # 默认给洋红
            ax_psnr.plot(range(1, len(history) + 1), history, style, linewidth=width, label=name)
            
    ax_psnr.set_xscale('log')
    import matplotlib.ticker as ticker
    ax_psnr.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax_psnr.set_xlabel('Iteration (Log Scale)', fontsize=10)
    ax_psnr.set_ylabel('PSNR (dB)', fontsize=10)
    ax_psnr.set_title('PSNR Comparison', fontsize=11)
    ax_psnr.legend(fontsize=9)
    ax_psnr.grid(True, which="both", ls="-", alpha=0.3)
    
    # === 绘制 SSIM 曲线 ===
    ax_ssim = plt.subplot(2, 4, 8)
    for name, history in ssim_h_dict.items():
        if len(history) == 1:
            ax_ssim.axhline(history[0], color='k', linestyle=':', label=name)
        else:
            style, width = styles.get(name, ('m-.', 1.5))
            ax_ssim.plot(range(1, len(history) + 1), history, style, linewidth=width, label=name)
            
    ax_ssim.set_xscale('log')
    ax_ssim.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax_ssim.set_xlabel('Iteration (Log Scale)', fontsize=10)
    ax_ssim.set_ylabel('SSIM', fontsize=10)
    ax_ssim.set_title('SSIM Comparison', fontsize=11)
    ax_ssim.legend(fontsize=9)
    ax_ssim.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches='tight')
    plt.close(fig) # 防止在终端弹出或者占用内存
    print("=> 导出对比图完成！")
