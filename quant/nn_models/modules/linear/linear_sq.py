import torch
import warnings
import torch.nn as nn
from torch.autograd import Function
from quant.utils.common_utils import get_best_device
from quant.utils.quantization_utils import (
    quantize_per_tensor_absmax,
    quantize_weight_per_channel_absmax,
)

user_has_been_warned = False
from .linear_base import LinearBase
class SqW8A8BBF16OBF16Linear(LinearBase):
    # For qkv_proj
    def __init__(self, in_features, out_features, bias, weight_scale=1.0, input_scale=1.0, alpha=1.0, beta=1.0, dev="cuda:0"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('qweight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False,
                                                                device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros(
                (self.out_features), dtype=torch.float16, requires_grad=False,device=dev)) # qwen2是bf16,opt是fp16
        else:
            self.bias = None
        self.register_buffer('weight_scale', torch.tensor(weight_scale,device=dev)) 
        self.register_buffer('input_scale', torch.tensor(input_scale,device=dev))

    @torch.no_grad()
    # 如果是上线的话
    # 这里直接使用cutlass in8 精度乘法
    # y = cutlass_int8_gemm(x_i8, self.q_weight, self.bias, input_scale, weight_scale)
    def forward(self, x):
        x_shape = x.shape
        # [batchsize, tokens, hiddendim] => [bs * tokens, hiddendim]
        x = x.view(-1, x_shape[-1]).to(self.qweight.device)
        x_bf16 = x.to(torch.bfloat16) * self.input_scale.item()  # 反量化输入
        weight_bf16 = self.qweight.to(torch.bfloat16) * self.weight_scale.item()  # 反量化权重
        weight_bf16 = weight_bf16.t()
        y = torch.matmul(x_bf16, weight_bf16)  # FP16/BF16 计算
        if self.bias:
            y += self.bias #[xx, out feats] + [1, out feats]
        # [batchsize*tokens, out_feats] => [batchsize, tokens, out_feats]
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_linear(module: torch.nn.Linear, input_scale, dev="cuda:0"):
        int8_module = SqW8A8BBF16OBF16Linear(
            module.in_features, module.out_features, module.bias is not None)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        
        # 这里无论选择PerTensor或者PerChannel，weight scale的shape都要和register buffer的weight scale对的上才行
        int8_module.weight_scale = torch.tensor(weight_scale, device=dev) # scalar
        int8_module.input_scale = torch.tensor(input_scale, device=dev) # scalar
        int8_module.qweight = int8_weight # [out, in] row major
        if module.bias is not None:
            int8_module.bias = module.bias.clone()
        return int8_module
