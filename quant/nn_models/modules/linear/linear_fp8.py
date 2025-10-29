import transformers
import torch
from typing import Tuple, Optional
from .linear_base import LinearBase
from quant.utils.fp8_kernel import *

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
