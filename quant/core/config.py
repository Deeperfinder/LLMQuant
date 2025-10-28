# 量化配置 动态 静态 perchannel per-token
import os
import json
from typing import Dict, Optional
from dataclasses import dataclass,field
from transformers.utils.hub import PushToHubMixin

"""
在量化的时候，保存量化配置，
在runtime的时候，读取配置，获取量化的具体信息，从而分发到不同的量化算子(fp8、sq、awq)
1. 用于从config.json读取config，反序列化，用来初始化本类
2. 在保存的时候，将量化配置序列化为Dict
"""

@dataclass
class QuantConfig(PushToHubMixin):# 专门用于将模型、配置或分词器等对象一键上传到 Hugging Face Hub，QuantConfig自动获得上传到 Hub 的能力
    # * 对于 list, dict, set 等可变类型，默认值必须用 field(default_factory=...)
    # * 对于 int, str, bool, None 等不可变类型，可以直接赋值，但用 field(default=...) 也无妨（更统一）
    quant_method: str = field(default = "awq")
    zero_point: bool = field(default = False) # for awq
    q_group_size: int = field(default = 0)    # for awq
    w_bit: int = field(default = 8) # awq is 4
    config_file_name: str = "config.json"
    modules_to_not_convert: list = field(default_factory=lambda: ["lm_head"])
    fp8_static_quant: bool = field(default = False)
    per_tensor: bool = field(default = True)
    kv_cache_quant_layers: list = field(default_factory=list) # when enabled, quantize_output=true


    @classmethod
    def from_dict(cls, quant_config: Dict={}):
        if not quant_config:
            quant_config = cls()
        else:
            quant_config = cls(**quant_config)
        return quant_config
    # 1. 
    @classmethod
    def from_pretrained(cls, save_dir, **kwargs):
        resolved_config_file = os.path.join(save_dir, cls.config_file_name)
        print(f"resolved_config_file:{resolved_config_file}")
        with open(resolved_config_file, "r", encoding="utf-8") as f:
            loaded_config = json.loads(f.read())
        
        quant_config = loaded_config.get("quantization_config")
        if quant_config is not None:
            # **dict 会把字典的 key-value 自动展开为关键字参数（keyword arguments）传给构造函数。 即传给类
            quant_config = cls(**quant_config)
        
        if quant_config is None:
            quant_config = cls()
        return quant_config
    
    def to_transformers_dict(self):
        return {
            "quant_method": self.quant_method,
            "zero_point": self.zero_point,
            "q_group_size": self.q_group_size,
            "w_bit": self.w_bit,
            "modules_to_not_convert": self.modules_to_not_convert,
            "fp8_static_quant": self.fp8_static_quant,
            "per_tensor": self.per_tensor,
            "kv_cache_quant": self.kv_cache_quant_layers
        }
    
