#让我看看,要分为以下几个板块 1,一个大的BM3D的class 2, 一个提取图像并进行操作的read操作  对,后面提交的时候要把文件夹夹在这里面
#3.主函数的运行 4.调用BM3D
#----------------------------配置
import torch_dct as dct
import torch
#science toolket
from skimage import io, img_as_float, color
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import time
import numpy as np


path=r"C:\Users\19811\Desktop\123"
# raw,不然出错
start_time=time.time()
#woc,原来要在gpu上面1运行要选择device呀
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

#----------------------------bm3d的class

#然后就是第二个难点,bm3d的实现有哪些东西,嗯,首先我们要定义L2范数,不过可以直接调库,然后就是搜索的算法,搜索
#的算法包含匹配的臂肘,搜索匹配上之后还要有dct和dct还原,以及将原函数复原,复原里面也还有计数器的设定,嗯\
#先写L2范数,再写dct和ldct,嗯先写匹配再写还原嗯
# 嗯,不过为什么要先用2d的dct,应该是相同的的呀,那么一个可以封装到一个3d的变换里面
#dct具有分离性,直接在一个函数里面就可以实现3d的,哦,由于我想用torch来加加速,所以得下一个dct的库
class BM3D(object):
    def __init__(self):
        self.step=3
        self.long=39
        self.short=8
        self.choose_max=5
        self.sigma=25
        self.step2=3
        self.big=2.7/255*self.sigma#传说中的经验公式

    #哦,单下划线表示类里面的函数,双下划线两端是自用的
    def _dct (self,image):
        image = dct.dct(image, norm='ortho')
        image = image.transpose(-1, -2)
        image = dct.dct(image, norm='ortho')
        image = image.transpose(-1, -2)
        image = image.permute(1, 2, 0)
        image = dct.dct(image, norm='ortho')
        image = image.permute(2, 0, 1)
        return image

    def _idct (self,image):
        image = image.permute(1, 2, 0)
        image = dct.idct(image, norm='ortho')
        image = image.permute(2, 0, 1)
        image = image.transpose(-1, -2)
        image = dct.idct(image, norm='ortho')
        image = image.transpose(-1, -2)
        image = dct.idct(image, norm='ortho')
        return image

    def _choose(self, image, i, j):
        h, w = image.shape[-2:]  # 获取宽高
        # 限制搜索范围，防止越界
        rmin = max(0, i - self.long)
        rmax = min(h - self.short, i + self.long)
        cmin = max(0, j - self.long)
        cmax = min(w - self.short, j + self.long)

        # 提取当前参考块
        orign = image[i:i + self.short, j:j + self.short]

        len2 = []
        find = []
        block = []

        # --- 这里的循环是速度瓶颈，但在不重写为 unfold 的情况下，我们尽量优化内部 ---
        for u in range(rmin, rmax + 1, self.step):  # 注意 python range 是左闭右开，所以 +1
            for v in range(cmin, cmax + 1, self.step):
                # 提取比较块
                ma = image[u:u + self.short, v:v + self.short]
                # L2 范数计算 (直接利用 GPU tensor 计算)
                # 使用 dist = ((orign - ma)**2).sum() 比 linalg.norm 快一点点，因为不需要开根号排序结果是一样的
                templen = torch.sum((orign - ma) ** 2)

                len2.append(templen)
                find.append((u, v))
                block.append(ma)

        # 转换为 Tensor 并在 GPU 上排序
        len2s = torch.tensor(len2, device=image.device)
        blocks = torch.stack(block)  # 这里 stack 会自动把 list 里的 tensor 堆叠

        # 排序
        # argsort 默认是升序，也就是最小距离在前，这是对的
        sort_idx = torch.argsort(len2s).to(device)

        # 截取前 N 个
        # 修正：防止找到的块少于 choose_max 的情况（比如在边缘）
        num_keep = min(self.choose_max, len(sort_idx))
        sort_idx = sort_idx[:num_keep]

        # 返回排序后的块和位置索引（因为 find 是 list，不能直接用 tensor 索引，需要转换一下思路）
        # 方法：把 find 转成 tensor 再索引
        finds = torch.tensor(find, device=image.device)[sort_idx].to(device)
        return blocks[sort_idx], finds

    def denoise(self, img):
        h, w = img.shape[:2]
        s = self.short

        cnt1 = torch.zeros_like(img).to(device)
        cnt2 = torch.zeros_like(img).to(device)

        for r in range(0, h -s + 1, self.step2):
            for c in range(0, w - s + 1, self.step2):
                matched_blocks, positions = self._choose(img, r, c)
                group_3d_dct = self._dct(matched_blocks)
                mask = torch.abs(group_3d_dct).to(device) > self.big
                cnt_nonzero = torch.sum(mask).item()
                group_3d_dct[torch.abs(group_3d_dct).to(device) < self.big] = 0
                group_3d_denoised = self._idct(group_3d_dct)
                weight = 1.0 / (self.sigma ** 2 * cnt_nonzero) if cnt_nonzero >= 1 else 1.0
                for i, (pr, pc) in enumerate(positions):
                    cnt1[pr:pr + s, pc:pc + s] += weight * group_3d_denoised[i]
                    cnt2[pr:pr + s, pc:pc + s] += weight

        return cnt1 / (cnt2 + 1e-10)
#----------------------------bm3d的调用以及计算结果如何
def run_bm3d(image_path,sigma=25):
    #简单处理,先不考虑彩色图了
    imag=io.imread(image_path)
    if imag.ndim==3:
        imag=color.rgb2gray(imag)
    image = torch.tensor(img_as_float(imag), dtype=torch.float32).to(device)
    #只考虑二维的情况
    image_noise=image.clone()
    image_noise+=torch.normal(mean=0,std=sigma/255.0,size=image.shape).to(device)
    image_clean=FBM3D.denoise(image_noise)
    img_np = image.cpu().numpy()
    noisy_np = image_noise.cpu().numpy()
    clean_np = image_clean.cpu().numpy()
    clean_np = np.clip(clean_np, 0, 1)
    noisy_np = np.clip(noisy_np, 0, 1)
    #psnr(a,b)=psnr(b,a)不用管qianhou位置
    return psnr(img_np, noisy_np), psnr(img_np, clean_np)

#----------------------------主函数
sumnoise=sumdenoise=cnt=0
FBM3D=BM3D()
for filename in os.listdir(path):
    full_path = os.path.join(path,filename)
    if not os.path.isfile(full_path):
        continue
    noisy, denoised = run_bm3d(os.path.join(path, filename))
    sumnoise += noisy
    sumdenoise += denoised
    cnt+=1

end_time = time.time()
print(sumnoise/cnt, sumdenoise/cnt)
print(end_time - start_time)