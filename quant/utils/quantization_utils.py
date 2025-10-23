import torch
import numpy as np 

@torch.no_grad()
def quantize_per_tensor_absmax(x):
    scale = x.abs().max() / 127
    if not x.is_cuda():
        x = x.float()
    qx = x.div(scale).round().clamp(-128, 127)
    x_q = qx.to(torch.int8)
    return x_q, scale

@torch.no_grad()
def quantize_weight_per_channel_absmax(w):
    scales = w.abs().max(dim=1) / 127
    scales = scales.view(-1, 1)

    if not w.is_cuda():
        w = w.float()
    qw = w.div(scales).round().clamp(-128, 127)
    w_q = qw.to(torch.int8)
    return w_q, scales
