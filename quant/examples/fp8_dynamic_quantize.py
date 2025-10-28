from quant.core.api import AutoQuantForCausalLM
from transformers import AutoTokenizer


model_path = "/cfs_cloud_code/fiinnxu/models/modelscope/Qwen2.5-7B-Instruct"
quant_path = "/cfs_cloud_code/fiinnxu/models/modelscope/Qwen2.5-7B-Instruct-fp8-dynamic"

quant_config = {"quant_method": "fp8_dynamic_quant"}

# load model
model = AutoQuantForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f'Model is quantized and saved at "{quant_path}"')
