import torch
import warnings
import torch.nn as nn
from torch.autograd import Function
from quant.utils.common_utils import get_best_device
from quant.utils.packing_utils import dequantize_gemm
from .linear_base import LinearBase

class AWQLinearMMFunction(Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(
        ctx,
        x,
        qweight,
        qzeros,
        scales,
        w_bit,
        group_size=128,
        bias=None,
        out_features=0,
    ):
        # the forward pass can use ctx.
        ctx.save_for_forward(x, qweight, qzeros, scales, bias)
        ctx.out_features = out_features
        # x.shape[4, 5120], w.shape[5120, 64], out.shape = [4, 64]
        out_shape = x.shape[:-1] + (out_features,)
        x = x.to(torch.float16)
        if x.shape[0] == 0:
            return torch.zeros(out_shape, dtype=x.dtype, device=x.device)
        out = dequantize_gemm(qweight, qzeros, scales, w_bit, group_size)
        out = torch.matmul(x, out)

        out = out + bias if bias is not None else out
        out = out.reshape(out_shape)

        if len(out.shape) == 2:
            out = out.unsqueeze(0)
        
        return out
class AWQLinear_GEMM(LinearBase):

    def __init__(
        self, w_bit, group_size, in_features, out_features, bias, dev
    ):
        super(LinearBase, self).__init__()
        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features

        assert self.in_features % self.group_size == 0
        assert out_features % (32 //self.w_bit) == 0

        self.register_buffer(
            "qweight",
            torch.zeros(
                (in_features, out_features // (32 // self.w_bit)),
                dtype = torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "qzeros",# [40, 640], 行上group size为128，列上每8个int4 pack为int
            torch.zeros(
                (in_features // self.group_size, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_scales(
            "scales",
            torch.zeros(
                in_features // self.group_size, out_features,
                dtype=torch.float16,
                device = dev,
            )
        )
        if bias:
            self.register_buffer(
                "bias",# [5120]
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                    device=dev,
                ),
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None
    ):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        # 1. quantize weight to int4
        # qx = clip(round(x / scale + zp)) = clip(round((x + scales * zp) / scale))
        scales_zeros = zeros * scales
        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()
        pack_num = 32 // awq_linear.w_bit

        # scales.shape = [5120//128, 5120]
        # zeros.shape=[5120/128,5120]
        # scale_zeros.shape=[5120/128,5120]
        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append( # group_size 个元素共享一个zeros
                torch.round( #linear.weight.data.shape=[5120,5120]对应[out channels, in channels], scale_zeros.shape=[40,5120], 
                    linear.weight.data[:, idx] + scales_zeros[idx // group_size]
                    / awq_linear.scales[idx // group_size]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1) # [5120, 5120]
        intweight = intweight.t().contiguous() # 由out features, in feat 转到in feat, out feat
        intweight = intweight.to(dtype=torch.int32)

        # 2. pack 8xint4 value weight into 1 int32 type weight
        qweight = torch.zeros( #[5120, 640]
            (intweight.shape[0], intweight.shape[1] // 32 * awq_linear.w_bit),
            dtype = torch.int32,
            device = intweight.device,
        )
        # 将intweight pack为q weight，并且按照order map重排序这pack num个元素
        for col in range(intweight.shape[1] // pack_num): #5120/8 = 640
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]] # 8个int32类型实际int4数据
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit) # for循环中，依次把qweight_col（只有低4bit有数据）移到qweight[:, col]的8个4bit位置
        # 0000 0000 0000 0000 0000 0100 0101 0010 qweight[: col]
        # 0000 0000 0000 0000 0000 0000 0000 0100 i = 2 左移 8位
        
        awq_linear.qweight = qweight
        zeros = zeros.to(dtype=torch.int32)
        # 3.pack 8xint4 value zero into 1 int32 type zero
        qzeros = torch.zeros(  #【40，5120】==pack==>【40，640】
            (zeros.shape[0], zeros.shape[1] // 32 * awq_linear.w_bit),
            dtype=torch.int32,
            device=zeros.device,
        )

        for col in range(zeros.shape[1] // pack_num): # 40 // 8 = 5
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num): # 8
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros
        
        return awq_linear

    def forward(self, x):
        # x.shape=[4,5120],w.shape=[5120,64],out.shape=[4,64]
        input_dtype = x.dtype
        # Todo : awq gemm kernel 支持triton
        # if input_dtype != torch.float16:
        #       x = x.half()

        with torch.no_grad(): 
            out = AWQLinearMMFunction.apply(
                x,
                self.qweight,
                self.qzeros,
                self.scales,
                self.w_bit,
                self.group_size,
                self.bias,
                self.out_features,
            )
        # if input_dtype != torch.float16:
        #     out = out.to(dtype=input_dtype)
        return out
