import torch
from transformers import AutoConfig
from quant.nn_models import *

Quant_CAUSAL_LM_MODEL_MAP = {
    "qwen2": Qwen2ModelForCausalLM,
    "llama": LLamaModelForCausalLM,
    "llama4": Llama4ModelForCausalLM,
}

def check_and_get_model_type(model_path, trust_remote_code, **model_init_kwargs):
    # 读取模型的config.json
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, **model_init_kwargs
    )
    print("model type is ", config.model_type)
    if config.model_type not in Quant_CAUSAL_LM_MODEL_MAP.keys():
        raise TypeError(f"{config.model_type} isn't supported yet.")

class AutoQuantForCausalLM:
    
    def __init__(self):
        raise EnvironmentError(
            "you must instantiate AutoQuantForCausalLLM with from_pretrained func."
        )
    @staticmethod
    def from_pretrained(
        self,
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
        safetensors=True,
        device_map=None,
        low_cpu_mem_usage=True,
        use_cache=False,
        **model_init_kwargs,
    ):
        model_type = check_and_get_model_type(
            model_path, trust_remote_code, **model_init_kwargs
        )
        return Quant_CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(
            model_path,
            model_type,
            torch_dtype=torch_dtype,             # 模型权重类型
            trust_remote_code=trust_remote_code, # 相信远端code或者模型
            safetensors=safetensors,             # 模型权重格式
            device_map=device_map,               # 指定模型加载的设备
            low_cpu_mem_usage=low_cpu_mem_usage, # 控制模型加载时是否尽量减少 CPU 内存占用，如果为true，模型会分步加载权重到GPU，避免一次占用大量内存
            use_cache=use_cache,                 # 控制是否使用kv cache 来缓存kv
            **model_init_kwargs,
        )
