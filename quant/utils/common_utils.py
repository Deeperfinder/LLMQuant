import torch
import gc
from torch import nn

def get_best_device():
    if torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def exculde_layers_to_not_quantize(linear_layers, modules_to_not_quantize):
    if modules_to_not_quantize is None:
        return linear_layers
    filter_layers = {}
    for name, layer in linear_layers.items():
        if not any(key in name for key in modules_to_not_quantize):
            filter_layers[name] = layer

    return filter_layers

def clear_memory(weight=None):
    if weight is not None:
        del weight
    gc.collect()
    torch.cuda.empty_cache()

def get_op_by_name(module, op_name):
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")

def get_op_name(module, op):
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")

def set_op_name(layer, name, new_module):
    levels = name.split(".") # self_attn.q_proj(torch.nn.Module) => self_attn.q_proj(q_linear)
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)

def append_str_prefix(x, prefix):
    if isinstance(x, str):
        return prefix +x
    elif isinstance(x, tuple):
        return tuple([append_str_prefix(y, prefix) for y in x])
    elif isinstance(x, list):
        return [append_str_prefix(y, prefix) for y in x]
    else:
        return x
