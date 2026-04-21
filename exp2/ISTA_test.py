import torch
import torch.nn.functional as F
def haar_wavelet_decompose(x):
    b, c, h, w = x.shape
    pad_h = h % 2
    pad_w = w % 2
    if pad_h != 0 or pad_w != 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    x00 = x[:, :, 0::2, 0::2]  # 偶行偶列
    x10 = x[:, :, 1::2, 0::2]  # 偶行奇列
    x01 = x[:, :, 0::2, 1::2]  # 奇行偶列
    x11 = x[:, :, 1::2, 1::2]  # 奇行奇列
    ll = (x00 + x10 + x01 + x11) / 2.0
    lh = (x00 - x10 + x01 - x11) / 2.0
    lv = (x00 + x10 - x01 - x11) / 2.0
    ld = (x00 - x10 - x01 + x11) / 2.0
    return torch.cat([ll, lh, lv, ld], dim=1), (pad_h, pad_w)
def haar_wavelet_reconstruct(coeffs, pads):
    b, c_all, h_half, w_half = coeffs.shape
    c = c_all // 4
    ll, lh, lv, ld = torch.split(coeffs, c, dim=1)
    res = torch.zeros((b, c, h_half * 2, w_half * 2), device=coeffs.device)
    res[:, :, 0::2, 0::2] = (ll + lh + lv + ld) / 2.0
    res[:, :, 1::2, 0::2] = (ll - lh + lv - ld) / 2.0
    res[:, :, 0::2, 1::2] = (ll + lh - lv - ld) / 2.0
    res[:, :, 1::2, 1::2] = (ll - lh - lv + ld) / 2.0
    pad_h, pad_w = pads
    if pad_h > 0:
        res = res[:, :, :-pad_h, :]  # 切掉最下面的一行
    if pad_w > 0:
        res = res[:, :, :, :-pad_w]  # 切掉最右边的一列
    return res
def run_ISTA(x, lam=0.01):
    u, pads = haar_wavelet_decompose(x)
    u_prox = torch.sign(u) * torch.relu(torch.abs(u) - lam)
    x_out = haar_wavelet_reconstruct(u_prox, pads)
    return x_out