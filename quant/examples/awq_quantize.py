from quant.core.api import AutoQuantFromCausalLLM
from transformers import AutoTokenizer
model_path = "Qwen/Qwen2.5-14B-Instruct"
quant_path = "Qwen2.5-14B-Instruct-awq"

quant_config = {"quant_method" : "awq",
                "w_bit" : 4,
                "zero_point" : True,
                "q_group_size" : 128}

# 类名调用，静态方法
model = AutoQuantFromCausalLLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)

tokenizer.save_quantized(quant_path)
print(f"Model is quantized and saved at {quant_path}")
