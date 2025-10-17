import torch
import torch.nn as nn
from typing import Tuple, List
from quant.utils.common_utils import get_op_name

from transformers.models.llama.modeling_llama import LlamaRMSNorm

allowed_norms = [nn.LayerNorm, LlamaRMSNorm]

def apply_scale(module: nn.Module, scales_list: List, input_feat_dict: dict):
    best_device = next(module.parameters()).device

    # layer_names: module.mlp.gate_proj, module.mlp.up_proj, module.mlp.down_proj, module.self_attn.o_proj
    # {'self_attn.q_proj': Linear(in_features=5120, out_features=5120, bias=True)} 
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_name(module, prev_op_name)
        layers = [get_op_name(module, name) for name in layer_names]
        
        prev_op.to(best_device)
        scales.to(best_device)

        if(
            isinstance(prev_op, nn.Linear)
            and type(layers) == list
            and isinstance(layers[0], nn.Linear)
        ):
            scale_fc_fcs(prev_op, layers, scales)       # scale_fc_fcs
        
        elif isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)     # scale_fc_fc 

        elif(
            any(isinstance(prev_op, t) for t in allowed_norms)
            or "rmsnorm" in str(prev_op.__class__).lower()
        ):
            scale_ln_fcs(prev_op, layers, scales)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")
    # apply the scaling to input feat if given; prepare it for clipping
    if input_feat_dict is not None:
        for layer_name in layer_names:
            # Skip the modules that are not quantized
            if layer_name in input_feat_dict:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device))
     
    prev_op.cpu()
    for layer in layers:
        layer.cpu()
    scales.cpu()

@torch.no_grad()
def scale_ln_fcs(ln: nn.Linear, fcs: List[nn.Linear], scales: torch.Tensor):
    if not isinstance(fcs, list):
        fcs = [fcs]
    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales) # layernorm/s = gamma/s * (x-mean) / sqrt + beta / s

    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)
    
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
    
    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0

@torch.no_grad()
def scale_fc_fc(fc1: nn.Linear, fc2: nn.Linear, scales: torch.Tensor):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    scales = scales.to(fc1.weight.device)
    # 把fc2 act的s^-1放在fc1
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))
    
    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0

@torch.no_grad()
def scale_fc_fcs(fc1: nn.Linear, fcs: List[nn.Linear], scales: torch.Tensor):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(fc1.weight.device)
    # 把x的s^-1融合进prev_op(这里的fc1), 假设fc1的weight为w1，bais为b1，那么w1=w1/scales, b1=b1/scales
    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))
    # w * s
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
