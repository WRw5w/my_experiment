import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch
import torch.nn.functional as F
import numpy as np
import torch_dct as dct
from skimage import io, img_as_float, color
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

# ==========================================
# 1. 配置与工具
# ==========================================
INPUT_DIR = r"C:\Users\19811\Desktop\set32"  # 请修改为你的路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

class BM3D:
    def __init__(self, sigma=25):
        self.sigma = sigma
        self.block_size = 8       # 块大小 (8x8)
        self.search_win = 13      # 搜索窗口大小 (越大越慢，但效果越好，必须是奇数)
        self.match_num = 8        # 每个组保留多少个相似块 (Group Size)
        self.threshold_factor = 2.7 * sigma / 255.0  # 阈值
        self.step = 4             # 滑动步长 (越小越慢，效果越好。PyTorch版建议设为 4 或 8)

    def _block_matching(self, img):
        """
        全并行的块匹配算法。
        策略：不遍历像素，而是遍历"偏移量"。
        """
        b, c, h, w = img.shape
        pad_size = self.search_win // 2
        # 1. 对原图进行 Unfold，得到所有参考块
        # ref_patches: [B, C*kw*kh, L], L 是块的总数
        # 为了显存考虑，我们不完全展开所有东西，而是采用 Shift-Map 策略

        # 计算所有块的基础“自身平方和”比较复杂，我们换一种思路：
        # 计算整张图在不同偏移下的 MSE。

        # 为了处理边界，先 Padding
        img_pad = F.pad(img, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

        # 结果容器：[B, num_shifts, H, W]
        # 我们只计算 step 网格上的点，节省计算量
        # 但为了利用 Fold，最简单的方法是计算全图距离，然后只取 step 点

        num_shifts = self.search_win ** 2
        # 生成所有可能的 (dy, dx) 偏移
        shifts = []
        for dy in range(-pad_size, pad_size + 1):
            for dx in range(-pad_size, pad_size + 1):
                shifts.append((dy, dx))

        # 存储每个像素点、在每个偏移下的 Block Distance
        # 维度: [B, NumShifts, H, W] -> 显存杀手，需要优化
        # 优化：直接 Unfold 原图得到 ref_patches [B, dim, num_blocks]

        # --- 显存优化版匹配策略 ---
        # 1. 提取所有参考块
        patches = F.unfold(img, kernel_size=self.block_size, stride=self.step)
        # patches: [B, C*8*8, Num_Patches]

        B, D, N = patches.shape
        # 初始化距离矩阵，设为无穷大
        distances = torch.full((B, num_shifts, N), float('inf'), device=img.device)

        # 2. 遍历搜索窗口内的每一个偏移
        idx = 0
        for dy in range(-pad_size, pad_size + 1):
            for dx in range(-pad_size, pad_size + 1):
                # 将图像平移
                # 利用切片模拟 roll，比 torch.roll 快
                # 原始中心区域: img
                # 偏移后的图: 取 img_pad 中间对应偏移的区域
                start_y, start_x = pad_size + dy, pad_size + dx
                shifted_img = img_pad[:, :, start_y:start_y+h, start_x:start_x+w]

                # 提取平移后的图的块
                shifted_patches = F.unfold(shifted_img, kernel_size=self.block_size, stride=self.step)

                # 计算距离 (MSE): sum((ref - shift)^2)
                # 维度: [B, NumPatches]
                dist = torch.sum((patches - shifted_patches) ** 2, dim=1)

                distances[:, idx, :] = dist
                idx += 1

        # 3. 排序找最小的 K 个
        # values: [B, K, N], indices: [B, K, N] (indices 是 shift 的索引)
        topk_dists, topk_indices = torch.topk(distances, k=self.match_num, dim=1, largest=False)

        return topk_indices, shifts, patches

    def denoise(self, img):
        """
        img: [B, 1, H, W]
        """
        B, C, H, W = img.shape
        pad = self.search_win // 2
        img_pad = F.pad(img, (pad, pad, pad, pad), mode='reflect')

        # 1. 块匹配
        # topk_indices: [B, K, N_blocks] -> 记录了每个块最相似的 K 个偏移索引
        topk_indices, shifts, ref_patches_flat = self._block_matching(img)

        # 2. 组装 3D 组 (Stacking)
        # 我们需要根据 topk_indices 把对应的块取出来堆叠
        # ref_patches_flat: [B, Dim, N]
        num_blocks = ref_patches_flat.shape[2]
        dim = ref_patches_flat.shape[1] # 64

        # 准备容器: [B, GroupSize, Dim, NumBlocks]
        group_3d = torch.zeros((B, self.match_num, dim, num_blocks), device=img.device)

        # 填入数据
        for k in range(self.match_num):
            # 获取第 k 个最近邻的 shift 索引
            # shift_idx: [B, N]
            shift_idx_batch = topk_indices[:, k, :]

            # 由于 PyTorch 的 gather 比较麻烦，这里用循环处理 Shift (一共只有 GroupSize 次，很快)
            # 更高效的方法是 index_select，但需要展平 shifts

            # 这种写法为了代码可读性，实际上 shift 只有几十种，
            # 更好的方法是先把所有 shifted patches 存起来，但太占显存。
            # 这里我们重新提取一次被选中的块 (牺牲计算换显存)

            # 这里的逻辑稍微复杂：我们需要对每个 block，根据它的 shift_idx 去找数据
            # 为了并行，我们只能遍历 "可能的 Shift" (search_win^2)，掩码选取
            # 但那样太慢。

            # === 快速提取方案 ===
            # 利用所有可能的 shifted patches 已经算过一次，但没存。
            # 我们根据 shift_idx 的值，直接从 padded image 对应位置提取太难向量化。
            # 妥协方案：只针对被选中的 shift 再次 unfold

            # 实际上，我们可以利用 topk_indices 里的索引直接映射回 (dy, dx)
            # 然后针对每个 batch 中的 block 提取。
            # 但这在 Python 里很难写成无循环。

            # 回退一步：为了作业能运行且逻辑清晰，
            # 我们在 block_matching 里其实已经生成了 shifted_patches。
            # 既然不能存所有，我们在这里不得不再次遍历 search window (偏移量)，
            # 仅仅把匹配上的搬运过来。

            pass # 占位

        # --- 重新实现的 Grouping 逻辑 (显存友好的) ---
        # 我们创建一个 [B, MatchNum, Dim, NumBlocks] 的 0 张量
        # 遍历所有可能的 shifts。如果某个 block 的第 k 个邻居指向这个 shift，就填入数据。

        idx = 0
        for dy in range(-pad, pad + 1):
            for dx in range(-pad, pad + 1):
                # 当前 shift 对应的 patches
                start_y, start_x = pad + dy, pad + dx
                shifted_img = img_pad[:, :, start_y:start_y+H, start_x:start_x+W]
                current_patches = F.unfold(shifted_img, kernel_size=self.block_size, stride=self.step)

                # 检查 topk_indices 中哪些位置等于当前 idx
                # topk_indices: [B, MatchNum, NumBlocks]
                mask = (topk_indices == idx) # [B, MatchNum, NumBlocks]

                if mask.any():
                    # 广播复制: current_patches [B, Dim, NumBlocks] -> [B, 1, Dim, NumBlocks]
                    src = current_patches.unsqueeze(1).expand(-1, self.match_num, -1, -1)
                    # 只在 mask 为 True 的地方更新 group_3d
                    # group_3d [B, MatchNum, Dim, NumBlocks]

                    # 这是一个 In-place update
                    group_3d = torch.where(mask.unsqueeze(2), src, group_3d)

                idx += 1

        # 3. 协同滤波 (3D Transform -> Threshold -> Inverse)
        # group_3d shape: [B, GroupSize, Dim(64), NumBlocks]
        # 变换为 [B, NumBlocks, GroupSize, 8, 8] 以便进行 3D DCT

        # Reshape: Dim(64) -> 8x8
        g_data = group_3d.permute(0, 3, 1, 2) # [B, N, K, 64]
        g_data = g_data.view(B, num_blocks, self.match_num, self.block_size, self.block_size)

        # 3D DCT (利用 torch_dct)
        # 输入要求 [..., D, H, W]
        g_dct = dct.dct_3d(g_data, norm='ortho')

        # 硬阈值 (Hard Thresholding)
        mask_t = torch.abs(g_dct) > self.threshold_factor
        cnt_nonzero = torch.sum(mask_t, dim=(2,3,4), keepdim=True) # 统计非零系数用于加权
        g_dct_thresh = g_dct * mask_t.float()

        # 3D Inverse DCT
        g_rec = dct.idct_3d(g_dct_thresh, norm='ortho')

        # 4. 聚合 (Aggregation)
        # 权重计算: Wi = 1 / (N_nonzero + epsilon)
        weights = 1.0 / (cnt_nonzero + 1e-8) # [B, N, 1, 1, 1]

        # 将数据加权
        g_rec_weighted = g_rec * weights

        # 我们只需要把这些块放回原位。
        # g_rec: [B, N, K, 8, 8]。BM3D通常把 K 个块都放回去，或者只放回参考块。
        # 标准做法是把整个 Group 放回去。但为了简化 Fold 操作，
        # 我们这里只取回“参考块位置”的那一份估计值（因为每个块都被作为参考块算了一次，已经覆盖了）
        # 也就是取 k=0 ? 不，BM3D利用了协同滤波，每一块都得到了去噪。
        # 这里的 Group_3D 包含了 shifted 的块。
        # 这是一个难点：Fold 只能还原参考块的位置。
        # **简化策略**：我们只取 Group 中的第一个分量（即参考块自己被去噪后的结果），
        # 加上它在作为别人的邻居时被去噪的结果？
        # 不，最简单的正确 PyTorch 实现是：只保留 Group 中的第 0 个切片 (Reference Block 的估计)。
        # 因为所有块都会轮流做 Reference，所以全图都会被覆盖。

        estimate_patches = g_rec_weighted[:, :, 0, :, :] # [B, N, 8, 8]
        estimate_weights = weights[:, :, 0, :, :]        # [B, N, 1, 1]

        # Reshape 回 unfold 的格式: [B, Dim, NumBlocks]
        estimate_patches = estimate_patches.view(B, num_blocks, -1).permute(0, 2, 1)
        # 权重同理 (扩展到 Dim 大小以便 fold)
        estimate_weights = estimate_weights.view(B, num_blocks, -1).permute(0, 2, 1)
        estimate_weights = estimate_weights.expand_as(estimate_patches)

        # Fold 回去
        # output_size 需要是 (H, W)
        output_size = (H, W)

        # 分子: sum(weight * block)
        numerator = F.fold(estimate_patches, output_size=output_size,
                           kernel_size=self.block_size, stride=self.step)

        # 分母: sum(weight)
        denominator = F.fold(estimate_weights, output_size=output_size,
                             kernel_size=self.block_size, stride=self.step)

        # 最终结果
        result = numerator / (denominator + 1e-8)

        return torch.clamp(result, 0, 1)
