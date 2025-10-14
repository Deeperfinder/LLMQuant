import os
import torch
import transformers

from torch import nn
from transformer import (
    PretrainedModel,
    PretrainedConfig,
    AutoConfig
)
from huggingface_hub import snapshot_download, save_torch_state_dict
from .config import QuantConfig
from quant.quantization import get_concrete_quantizer_cls   

TRANSFORMERS_AUTO_MAPPING_DICT={
    "qwen2" : "AutoModelForCausalLM"
}

"""
    抽象类，
    包含quantize方法，和save_quantized方法
"""
class BaseModelForCausalLM(nn.Module):
    def __init__(self,
                 model,        # The pretrained or quantized model 
                 model_type,   # The model type, ep . "qwen2"
                 is_quantized, # 判断该模型是否已经被量化
                 config,       # The config of the model
                 quant_config):
        super().__init__()
        self.model : PretrainedModel = model
        self.model_type : str= model_type
        self.is_quantized : bool = is_quantized
        self.config : PretrainedConfig = config
        self.quant_config : QuantConfig = quant_config
    
    @classmethod
    def from_pretrained(
        self,
        model_path,
        model_type,
        torch_dtype="auto",
        trust_remote_code=True,
        safetensors=True,
        device_map=None,
        low_cpu_mem_usage=True,
        use_cache=False,
        **model_init_kwargs,
    ):
        model_weights_path, config, quant_config = self._load_config(
            self, model_path, "", safetensors, trust_remote_code=trust_remote_code
        )
        target_cls_name = TRANSFORMERS_AUTO_MAPPING_DICT[config.model_type]
        target_cls = getattr(transformers, target_cls_name)

        if model_init_kwargs.get("low_cpu_mem_usage") is None:
            model_init_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
        if model_init_kwargs.get("use_cache") is None:
            model_init_kwargs["use_cache"] = use_cache
        # torch.nn.modules
        model = target_cls.from_pretrained(
            model_weights_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            use_safetensors=safetensors,
            device_map=device_map,
            **model_init_kwargs,    
        )

        model.eval()
        
        return self(
            model,
            model_type,
            is_quantized=False,
            config=config,
            quant_config=quant_config
        )
    
    @torch.no_grad() # 不用为反向传播计算一些东西，这样传输的速率更快
    def quantize(
        self,
        tokenizer=None,
        quant_config={},
        calib_data="pileval",
        duo_scaling=True,                   # 是否在scale的时候使用 w/x 还是 x
        fake_quant=False,
        apply_clip=True,
        n_parallel_calib_samples=None,
        max_calib_samples=128, 
        max_calib_seq_len=512,
        max_chunk_memory=1024 * 1024 * 1024, # 1GB AWQ 全局搜索误差损失，activation均值显存分配
        **kwargs,
    ):
        self.quant_config: QuantConfig = QuantConfig.from_dict(quant_config)
        if hasattr(self, "modules_to_not_convert"):
            self.quant_config.modules_to_not_convert = self.modules_to_not_convert

        # dispatch to dedicated quantizer
        quantizer_cls = get_concrete_quantizer_cls(self.quant_config.quant_method)
        self.quantizer = quantizer_cls(
            self,
            self.model,
            self.model_type,
            tokenizer,
            self.quant_config,
            self.quant_config.quant_method,
            self.quant_config.w_bit,
            self.quant_config.q_group_size,
            self.quant_config.zero_point,
            calib_data,
            duo_scaling, # only for awq
            modules_to_not_convert=self.quant_config.modules_to_not_convert,
            fake_quant=fake_quant, # use for awq and sq, but can change name to fake_quant
            apply_clip=apply_clip, # only for awq
            n_parallel_calib_samples=n_parallel_calib_samples, # only for awq
            max_calib_samples=max_calib_samples,
            max_calib_seq_len=max_calib_seq_len,
            max_chunk_memory=max_chunk_memory, # only for awq
            **kwargs,
        )
        self.quantizer.quantize()
        self.is_quantized = True

    def _load_config(self, 
                     model_path,
                     model_filename,
                     safetensors=True,
                     trust_remote_code=True,
                     max_seq_len=4096,
                     download_kwargs=None,
                     **config_kwargs,
    ):
        # step1: download model if path is not a dir
        if not os.path.isdir(model_path):
            ignore_patterns = ["*msgpack*", "*h5*", "*optimizer.pt", "*.onnx*"]
            if safetensors:
                ignore_patterns.extend(["*.pt*", "*.bin*", "consolidated*"])
            else:
                ignore_patterns.append("*.safetensors*")
            
            # 下载模型到/root/.cache/huggingface/hub
            model_path = snapshot_download(model_path, ignore_patterns = ignore_patterns)

        if model_filename != "":
            model_weights_path = model_path + f"/{model_filename}"
        else:
            model_weights_path = model_path

        # step2 : load config and set seqlen
        quant_config = QuantConfig.from_pretrained(model_path)
        # load model config and set max generation length
        if max_seq_len is None and hasattr(self, "max_seq_len_key"):
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, **config_kwargs
            )
            config.max_seq_len = getattr(config, self.max_seq_len_key, 2048)
            if hasattr(config, "text_config"):
                config.text_config.max_seq_len = getattr(
                    config, self.max_seq_len_key, 2048
                )
        else:
            max_seq_len = 2048 if max_seq_len is None else max_seq_len
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, **config_kwargs
            )
            config.max_seq_len = max_seq_len     
        return model_weights_path, config, quant_config       

    def save_quantized(self, save_dir):
        save_torch_state_dict(
            state_dict = self.model.state_dict(),
            save_directory = save_dir,
        )
