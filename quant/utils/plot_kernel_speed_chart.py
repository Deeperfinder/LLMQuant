import matplotlib.pyplot as plt

# 数据准备
data = [
    {"tokens": 128, "per_token_group_quant_fp8": 0.04229, "triton_per_token_group_quant_8bit": 0.00929, "sgl_per_token_group_quant_8bit": 0.00803},
    {"tokens": 256, "per_token_group_quant_fp8": 0.05807, "triton_per_token_group_quant_8bit": 0.01399, "sgl_per_token_group_quant_8bit": 0.01024},
    {"tokens": 512, "per_token_group_quant_fp8": 0.08894, "triton_per_token_group_quant_8bit": 0.02311, "sgl_per_token_group_quant_8bit": 0.01536},
    {"tokens": 1024, "per_token_group_quant_fp8": 0.15779, "triton_per_token_group_quant_8bit": 0.04111, "sgl_per_token_group_quant_8bit": 0.02449},
    {"tokens": 4096, "per_token_group_quant_fp8": 0.58170, "triton_per_token_group_quant_8bit": 0.14843, "sgl_per_token_group_quant_8bit": 0.07882},
    {"tokens": 8192, "per_token_group_quant_fp8": 1.13467, "triton_per_token_group_quant_8bit": 0.29055, "sgl_per_token_group_quant_8bit": 0.15102},
    {"tokens": 16384, "per_token_group_quant_fp8": 2.23874, "triton_per_token_group_quant_8bit": 0.57650, "sgl_per_token_group_quant_8bit": 0.29552},
]

# 分离数据以便绘图
tokens = [item["tokens"] for item in data]
times_fp8 = [item["per_token_group_quant_fp8"] for item in data]
times_triton = [item["triton_per_token_group_quant_8bit"] for item in data]
times_sgl = [item["sgl_per_token_group_quant_8bit"] for item in data]

# 绘制图表
plt.figure(figsize=(12, 8), dpi=300)
plt.plot(tokens, times_fp8, marker='o', label='per_token_group_quant_fp8')
plt.plot(tokens, times_triton, marker='s', label='triton_per_token_group_quant_8bit')
plt.plot(tokens, times_sgl, marker='^', label='sgl_per_token_group_quant_8bit')

plt.title('\nPerformance Comparison of Different Quantization Methods\n (hidden_dim=7168, group_size=128)')
plt.xlabel('Number of Tokens')
plt.ylabel('Mean Time (milliseconds)') # 直接表示时间单位
# plt.xscale('log')
# plt.yscale('log') # 如果需要显示对数刻度可以保留此行，否则可以删除或注释掉这一行
plt.legend()
plt.grid(True)

# 保存图表到当前目录下的jpg文件
plt.savefig('./performance_comparison.jpg')

# 显示图表
plt.show()
