# run_triton_fp8_quant.py

import torch
from quant.utils.fp8_kernel import triton_per_token_group_quant_8bit

def main():
    # 配置参数
    num_tokens = 1024
    hidden_dim = 4096
    group_size = 128
    dtype = torch.float8_e4m3fn  # 或 torch.float8_e5m2

    # 创建输入张量（模拟激活值）
    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden_dim, device='cuda', dtype=torch.bfloat16)
    print(f"Input shape: {x.shape}, dtype: {x.dtype}")

    # 调用 Triton 量化 kernel（注意：使用关键字参数避免位置错误！）
    x_quant, scales = triton_per_token_group_quant_8bit(
        x,
        group_size=group_size,
        eps=1e-10,
        dtype=dtype
    )

    # 输出信息
    print(f"Quantized output shape: {x_quant.shape}, dtype: {x_quant.dtype}")
    print(f"Scales shape: {scales.shape}, dtype: {scales.dtype}")
    print(f"Example scale values: {scales[0, :5]}")

    # 简单验证：检查是否为 FP8
    assert x_quant.dtype == dtype, f"Expected {dtype}, got {x_quant.dtype}"
    assert scales.dtype == torch.float32, "Scales should be float32"
    assert x_quant.shape == x.shape, "Output shape mismatch"
    assert scales.shape == (num_tokens, hidden_dim // group_size), "Scales shape mismatch"

    print("✅ Triton per-token group FP8 quantization succeeded!")

if __name__ == "__main__":
    # 确保 CUDA 可用
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script.")
    main()
