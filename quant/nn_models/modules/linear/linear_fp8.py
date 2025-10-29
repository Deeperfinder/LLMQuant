import transformers
import torch
from typing import Tuple, Optional
from .linear_base import LinearBase
import triton
import triton.language as tl

def per_tensor_quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    finfo = torch.finfo(torch.float_e4m3fn)
    if tensor.numel() ==0:
        min_val, max_val = (
            torch.tensor(-16.0, dtype=tensor.dtype),
            torch.tensor(16.0, dtype=tensor.dtype),
        )
    else:
        min_val, max_val = tensor.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = amax.clamp(min=1e-12) / finfo.max
    qweight = (tensor / scale).clamp(min=finfo.min, max=finfo.max)
    scale = scale.float()
    return qweight, scale

def per_channel_quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    finfo = torch.finfo(torch.float8_e4m3fn)
    if tensor.numel() == 0:
        # Deal with empty tensors (triggered by empty MoE experts)
        print("[warning] You are experiencing empty MoE experts, tensor numbers = 0")
        qweight = torch.empty_like(tensor, dtype=torch.float8_e4m3fn)
        scales = torch.ones((*tensor.shape[:-1], 1), dtype=torch.float32)
        return qweight, scales
    amax = tensor.abs().max(dim=-1, keepdim=True)
    scale = amax.clamp(min=1e-12) / finfo.max
    qweight = (tensor / scale).clamp(min=finfo.min, max=finfo.max)
    scale = scale.float()
    return qweight, scale

def per_token_group_quant_fp8(x: torch.Tensor,
                              group_size: int, 
                              dtype: torch.float8_e4m3fn,
                              eps=1e-10) -> Tuple[torch.Tensor, float]:
    assert x.shape[-1] % group_size == 0,("The last dimension of 'x' cannot be divided by group_size!")
    assert x.is_contiguous(),("The input tensor must be contiguous!")

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    # x[M, N] ==> x_[M*N // group_size , group_size]
    x_ = x.reshape(x.numel() // group_size, group_size)
    # max 返回最大值和其索引[value, index]
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / fp8_max

    # x_[M*N, group_size] / x_s[M*N, 1]
    x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max).to(dtype)
    # x_q : [M * N, group_size] => [M, group_size *C]
    x_q = x_q.reshape(x.shape)
    # x_s : [M*N // group_size, 1] ==> [M, N // group_size] 
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1]//group_size,))

    return x_q, x_s

def _per_token_group_quant_8bit_raw(x: torch.Tensor, 
                                    group_size: int,
                                    eps: float = 1e-10,
                                    dtype: torch.dtype = torch.float8_e4m3fnm
                                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Function use triton to perform per-token-group quantization on an input tensor 'x'.
        Args:
            x: The input tenosr with ndim >= 2.
            group_size: The group size used for quantization.
            eps: The minimum to avoid dividing zero.
            dtype: The dype of output tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the scaling factor for quantization.
    """
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    info = torch.finfo(dtype)
    bit8_max = info.max
    bit8_min = info.min

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = torch.empty(x.shape[:-1] + (x.shape[-1] // group_size,),
                      device=x.device,
                      dtype=torch.float32)
    M = x.numel() // group_size
    N = group_size
    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK//256, 1), 8)
    num_stages = 1
    _per_token_group_quant_8bit[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        N,
        eps,
        bit8_min=bit8_min,
        bit8_max=bit8_max,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return x_q, x_s

@triton.jit
def _per_token_group_quant_8bit(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    y_stride,
    N,
    eps,
    bit8_min,
    bit8_max,
    BLOCK:tl.constexpr,
):
    """
        A triton-accelerated function to perform per-token-group quantization on a tensor.
        This function converts the tensor values into float8 values.
        notes:
            一个block的大小为y_stride == group_size
    """
    # Map the program id to the row of X and Y it should compute
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK) #N <= BLOCK
    mask = cols < N

    # 加载目标数据
    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / bit8_max
    y_s_inv = 1.0 / y_s
    y_q = tl.clamp(y*y_s_inv, bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)

def static_per_tensor_quantize(tensor: torch.Tensor, static_scale: float) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    qweight = (tensor / static_scale).clamp(min=finfo.min, max=finfo.max)
    return qweight.to(torch.float8_e4m3fn)

def fp8_gemm(A, A_scale, B, B_scale, bias, out_dtype):
    if A.numel() == 0:
        # Deal with empty tensors (triggeted by empty MoE experts)
        return torch.empty(size=(0, B.shape[0]), dtype=out_dtype, device=A.device)

    native_fp8_support=False
    if native_fp8_support:
        need_reshape = A.dim() == 3
        if need_reshape:
            batch_size = A.shape[0]
            A_input = A.reshape(-1, A.shape[-1])
        else:
            batch_size = None
            A_input = A
        output, _ = torch._scaled_mm(
            A_input, # [m, in]
            B.t(),  # [in, out]
            out_dtype = out_dtype,
            scale_a = A_scale,
            scale_b = B_scale,
            bias = bias,
        )
        if need_reshape:
            output = output.reshape(
                batch_size, output.shape[0] // batch_size, output.shape[1]
            )
        else:
            output = torch.nn.functional.linear(
                A.to(out_dtype) * A_scale.to(out_dtype),
                B.to(out_dtype) * B_scale.to(out_dtype),
                bias=bias,
            )
        return output

def replace_module(model, name, new_module: torch.nn.Module):
    if "." in name:
        parent_name = name.rsplit(".", 1)[0] # model.layers.0.self_attn
        child_name = name[len(parent_name) + 1 :] # q_proj
        parent = model.get_submodule(parent_name)
        # Qwen2SdpaAttention(
        # (q_proj): Linear(in_features=5120, out_features=5120, bias=True)
        # (k_proj): Linear(in_features=5120, out_features=1024, bias=True)
        # (v_proj): Linear(in_features=5120, out_features=1024, bias=True)
        # (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
        # (rotary_emb): Qwen2RotaryEmbedding()
        # )
    else:
        parent_name = ""
        parent = model
        child_name = name
    # Qwen2SdpaAttention(
    # (q_proj): FP8DynamicLinear()
    # (k_proj): Linear(in_features=5120, out_features=1024, bias=True)
    # (v_proj): Linear(in_features=5120, out_features=1024, bias=True)
    # (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
    # (rotary_emb): Qwen2RotaryEmbedding()
    # )
    setattr(parent, child_name, new_module) # 把new module替换掉child name，成为parent的新child

class FP8DynamicLinear(LinearBase):
    def __init__(
        self,
        in_features,
        out_features,
        bias: bool,
        dev="cuda:0",
        dtype=torch.bfloat16,
        qdtype=torch.float8_e4m3fn,
        per_tensor=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.qdtype = qdtype
        self.weight = torch.nn.Parameter(torch.randn((self.out_features, self.in_features) ,dtype=dtype, 
                                          device=dev, requires_grad=False))
        self.per_tensor = per_tensor
        if self.per_tensor:
            self.weight_scale = torch.nn.Parameter(torch.randn(
                (1) ,dtype=torch.float32, device=dev, requires_grad=False))
        else: #  per channel
            self.weight_scale = torch.nn.Parameter(torch.randn(
                (self.out_features,1) ,dtype=torch.float32, device=dev, requires_grad=False))
        if bias:
            # 这里类型为bf16还是fp16需要根据模型类型而定,opt为fp16,qwen2为bf16
            self.bias = torch.nn.Parameter(torch.zeros( 
                (self.out_features), dtype=dtype, requires_grad=False, device=dev)) 
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, module: torch.nn.Linear, group_size=0, zeros=None, per_tensor=True):
        assert group_size == 0, "not support group wise fp8 quant yet! pls set group_size = 0"  
        fp8_dynamic_linear = cls(
            module.in_features, module.out_features, module.bias is not None, per_tensor=True)

        # 这里无论选择PerTensor或者PerChannel，weight scale的shape都要和register buffer的weight scale对的上
        if module.bias is not None:
            fp8_dynamic_linear.bias.data = module.bias.clone()
        if per_tensor:
            fp8_weight, weight_scale = per_tensor_quantize(module.weight)
            fp8_dynamic_linear.weight_scale.data = torch.tensor(weight_scale, device="cuda:0").unsqueeze(0) # 加了unsqueeze，scalar => [1]
        else: # per channel
            fp8_weight, weight_scale = per_channel_quantize(module.weight)
            weight_scale = weight_scale.to("cuda:0")
            fp8_dynamic_linear.weight_scale.data = weight_scale.detach().clone() # To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone()
        # fp8 weight在此处还是bf16 type fp8 val,fwd处to为fp8
        fp8_dynamic_linear.weight.data = fp8_weight # [out, in] row major

        return fp8_dynamic_linear
    
    def forward(self, x):
        if per_tensor:
            qinput, x_scale = per_tensor_quantize(x)
        else:
            qinput, x_scale = per_channel_quantize(x)
            self.weight_scale = self.weight_scale.t()
    
        self.weight = self.weight.to(self.qdtype)
        output = fp8_gemm(
            A=qinput,
            A_scale=x_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
        )
        return output
    
#. calib， 已经量化的weight
class FP8StaticLinearQuantizer(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        qdtype,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.nn.Parameter,
        quantize_output: bool = False,
    ):
        super().__init__()
        # 无需用Parameter
        self.qweight = weight #torch.nn.Parameter(weight, requires_grad=False)
        self.qdtype = qdtype
        self.weight_scale = weight_scale #torch.nn.Parameter(weight_scale, requires_grad=False)
        self.in_features = in_features
        self.out_features = out_features
        if bias is not None:
            self.bias = bias #torch.nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None
        # 需要保存calibration得到的act input scale
        self.input_scale = None
        # 需要保存calibration得到的kv scale
        self.output_scale = None
        self.quantize_output = quantize_output

    def forward(self, x):
        print("FP8StaticLinearQuantizer forward !!!!!!!!!!!")

        qinput, x_input_scale = per_tensor_quantize(x) # observer
        self.input_scale = x_input_scale
        # if self.input_scale is None:
        #     self.input_scale = torch.nn.Parameter(x_input_scale, requires_grad=False)
        # elif x_input_scale > self.input_scale:
        #     self.input_scale = torch.nn.Parameter(x_input_scale, requires_grad=False)
        qweight = self.qweight.to(self.qdtype)
        qinput = qinput.to(self.qdtype)
        output = fp8_gemm(
            A=qinput, # bf16
            A_scale=self.input_scale,
            B=qweight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype, # bf16
        )

        # Optionally, quantize output and record scale
        if self.quantize_output: # observer
            qoutput, output_scale = per_tensor_quantize(output)
            self.output_scale = output_scale
            # if self.output_scale is None:
            #     self.output_scale = torch.nn.Parameter(output_scale, requires_grad=False)
            # elif output_scale > self.output_scale:
            #     self.output_scale = torch.nn.Parameter(output_scale, requires_grad=False)
            output = qoutput.to(output.dtype) * output_scale 
        return output #fp16
    
class FP8StaticLinear(LinearBase):
    def __init__(
        self,
        in_features,
        out_features,
        bias,
        dev="cuda:0",
        dtype=torch.bfloat16,
        qdtype=torch.float8_e4m3fn, # TODO: 需要在quant tool中把e4m3或e5m2加入到quant config
        per_tensor=True, # only enable static quant when per tensor
        quantize_output=False
    ):
        super().__init__()        
        self.per_tensor = per_tensor
        self.qdtype = qdtype
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn((self.out_features, self.in_features) ,dtype=dtype, device=dev, requires_grad=False))
        if self.per_tensor:
            self.weight_scale = torch.nn.Parameter(torch.randn(
                (1) ,dtype=torch.float32, device=dev, requires_grad=False))
        else: # never go here
            self.weight_scale = torch.nn.Parameter(torch.randn(
                (self.out_features, 1) ,dtype=torch.float32, device=dev, requires_grad=False))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros( # 对于sq,这里bf16还是fp16需要根据模型类型而定,opt为fp16,qwen2为bf16
                (self.out_features), dtype=dtype, requires_grad=False, device=dev)) 
        else:
            self.bias = None
        # static quant only support per tensor act 
        self.input_scale = torch.nn.Parameter(torch.randn((1) ,dtype=torch.float32, device=dev, requires_grad=False))
        self.quantize_output = quantize_output
        self.output_scale = torch.nn.Parameter(torch.randn((1) ,dtype=torch.float32, device=dev, requires_grad=False))

    @classmethod
    def from_linear(cls, in_features, out_features, fp8_weight, weight_scales, bias, input_scale=None, output_scale=None, group_size=0, zeros=None, quantize_output=False):
        assert group_size == 0, "not support group wise fp8 quant yet! pls set group_size = 0"  
        fp8_static_linear = cls(
            in_features, out_features, bias is not None, per_tensor=True, quantize_output=quantize_output)
        
        if bias is not None:
            fp8_static_linear.bias.data = bias.clone()
        if True:# fp8_static_linear.per_tensor always true
            fp8_static_linear.weight_scale.data = torch.tensor(weight_scales, device=fp8_weight.device) # scalar,这里无需unsqueeze，因为在dyn quant中unsqueeze过了
        fp8_static_linear.weight.data = fp8_weight # [out, in] row major
        fp8_static_linear.input_scale.data = input_scale.unsqueeze(0)
        if quantize_output:
            fp8_static_linear.output_scale.data = output_scale.unsqueeze(0)
        
        return fp8_static_linear
    
    def forward(self, x):
        # scale is known in advance, so naming static
        print("FP8StaticLinear forward !!!!!!!!!!!")
        qinput = static_per_tensor_quantize(x, self.input_scale)
        weight = self.weight.to(self.qdtype)
        output = fp8_gemm( # 这里是tensor wise gemm，不是row wise gemm
            A=qinput,
            A_scale=self.input_scale,
            B=weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
        ) # fp16/fp32

        if self.quantize_output:
            qoutput = static_per_tensor_quantize(output, self.output_scale) # fp16/fp32 output / outputscale => fp8
            output = qoutput.to(output.dtype) * self.output_scale # fp8 * outputscale => fp16/32

        return output # fp16/fp32
