# 🚀 Advanced Quantization Toolkit for LLMs

A high-performance, extensible quantization framework for Large Language Models (LLMs), supporting **AWQ**, **SmoothQuant**, and **FP8** with flexible granularity options. 

---
| Feature | Status |
|--------|--------|
| **Quantization Methods** | |
| • AWQ (Activation-aware Weight Quantization) | ✅ Supported |
| • SmoothQuant | ✅ Supported |
| • FP8 Quantization (E4M3 / E5M2) | ✅ Supported |
| **Granularity Options** | |
| • Per-tensor quantization | ✅ Supported |
| • Per-channel quantization (weights) | ✅ Supported |
| • Per-group quantization (e.g., group size=128) | ✅ Supported |
| • Per-token quantization (activations) | ✅ Supported |
| **FP8 Capabilities** | |
| • Static FP8 quantization (offline calibration) | ✅ Supported |
| • Dynamic FP8 quantization (runtime scale) | ✅ Supported |
| • Multi-GPU FP8 inference & training | ✅ Supported |
| **High-Performance Kernels (In Progress)** | |
| • Triton: `per_token_group_quant` | 🟨 In Development |
| • Triton: `w8a8_block_fp8_matmul` | 🟨 In Development |
| • CUTE: `per_token_group_quant_8bit` | 🟨 In Development |
| • CUTE: `fp8_gemm_cute` | 🟨 In Development |
| • CUDA: `per_token_group_quant_8bit` (hand-optimized) | 🟨 In Development |
---

## 🚀 Quick Start

### Installation
```bash
# Install dependencies and the quant package
pip uninstall datasets modelscope -y
pip install addict zstandard
pip install "modelscope[dataset]" --upgrade
pip install -e .
```

### Run an Example (AWQ)
Make sure your model is downloaded in advance.
```bash
cd quant/examples
python awq_quantize.py
```

---

## 🧩 Extending the Framework

### Add a New Model
1. Edit `quant/core/api.py`  
   Add your model to `Quant_CAUSAL_LM_MODEL_MAP`:
   ```python
   Quant_CAUSAL_LM_MODEL_MAP = {
       "llama": LlamaQuantizer,
       "qwen": QwenQuantizer,
       "your_model": YourModelQuantizer,  # ← add here
   }
   ```

### Add a New Quantization Method
1. Register the method in `quant/quantization/__init__.py`:
   ```python
   method_to_quantizer = {
       "awq": AWQQuantizer,
       "smoothquant": SmoothQuantQuantizer,
       "your_method": YourMethodQuantizer,  # ← add here
   }
   ```
2. Create a new directory under `quant/quantization/your_method/` with:
   - `__init__.py`
   - `quantizer.py` (implement quantization logic)
3. Replace FP16 linear layers with your quantized version in  
   `quant/nn_models/modules/linear/linear_*.py`

---

## 🔗 Related Projects

- [**AWQ: Activation-aware Weight Quantization**](https://github.com/mit-han-lab/llm-awq)  
- [**SmoothQuant: Accurate and Efficient PTQ for LLMs**](https://github.com/mit-han-lab/smoothquant)  
- [**NVIDIA FP8 for Transformer Engines**](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/)

---

## 📄 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

> 💡 **Contribution Welcome!**  
> We encourage contributions of new quantization methods, model support, and high-performance kernels.

--- 
