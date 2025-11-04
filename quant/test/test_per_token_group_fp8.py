from quant.utils.fp8_kernel import triton_per_token_group_quant_8bit, per_token_group_quant_fp8
from quant.utils.kernel_binding import try_load_library
import torch
import numpy as np


def create_per_token_group_quant_test_data(num_tokens, hidden_dim, num_ranks, flags):
    device = torch.device("cuda")
    dtype = torch.bfloat16

    seed = num_tokens * 10000 + hidden_dim
    gen_cpu = torch.Generator(device="cpu")
    gen_cpu.manual_seed(seed)
    gen_cuda = torch.Generator(device="cuda")
    gen_cuda.manual_seed(seed)

    if flags["fuse_silu_and_mul"]:
        effective_hidden_dim = hidden_dim * 2
    else:
        effective_hidden_dim = hidden_dim
    del hidden_dim

    if (masked_layout_mode := flags["masked_layout_mode"]) is not None:
        num_max_dispatch_tokens_per_rank = 768
        num_global_experts = 288
        num_local_experts, remainder = divmod(num_global_experts, num_ranks)
        assert remainder == 0

        # mimic DeepEP low_latency_dispatch output
        x = torch.randn(
            num_local_experts,
            num_max_dispatch_tokens_per_rank * num_ranks,
            effective_hidden_dim,
            device=device,
            dtype=dtype,
            generator=gen_cuda,
        )

        if masked_layout_mode == "balanced":
            masked_m = _compute_balanced_split(num_tokens, num_local_experts)
        elif masked_layout_mode == "imbalanced":
            masked_m = _compute_imbalanced_split(
                num_tokens, num_local_experts, gen_cpu=gen_cpu
            )
        elif masked_layout_mode == "extreme":
            masked_m = torch.tensor(
                [num_tokens] + [0] * (num_local_experts - 1), dtype=torch.int
            )
        else:
            raise NotImplementedError
        print(f"{masked_layout_mode=} {masked_m=} {x.shape=}")

        masked_m = masked_m.to(device)

        return x, masked_m
    else:
        x = torch.randn(
            num_tokens,
            effective_hidden_dim,
            device=device,
            dtype=dtype,
            generator=gen_cuda,
        )
        x[torch.randn(x.shape, device=device, generator=gen_cuda) < 0.001] *= 10
        return x, None

def _compute_balanced_split(total: int, arr_len: int):
    base = total // arr_len
    remainder = total % arr_len
    ans = [base + 1 if i < remainder else base for i in range(arr_len)]
    assert sum(ans) == total
    return torch.tensor(ans, dtype=torch.int)


def _compute_imbalanced_split(
    total: int, arr_len: int, gen_cpu, dtype=torch.int
) -> list[int]:
    # can use `rand ** 2`, `rand ** 3`, etc, to change how imbalanced it is
    noise_raw = torch.rand(arr_len, generator=gen_cpu) ** 3

    noise = noise_raw / noise_raw.sum()
    ans = (noise * total).round().to(dtype)

    diff = total - ans.sum().item()
    while diff != 0:
        idx = torch.randint(0, arr_len, (1,), generator=gen_cpu).item()
        if diff > 0:
            ans[idx] += 1
            diff -= 1
        elif diff < 0 and ans[idx] > 0:
            ans[idx] -= 1
            diff += 1

    assert sum(ans) == total
    return ans


def test_per_token_group_quant_kernel(
    fn: callable,
    num_tokens,
    hidden_dim,
    group_size,
    kernel_ops=False,
    flags=None,
    num_ranks=1,
    dst_dtype=torch.float8_e4m3fn,
    num_tests=50,
    num_warmup=20,
):
    flags = {"fuse_silu_and_mul": False, "masked_layout_mode": None}
    x, masked_m = create_per_token_group_quant_test_data(
        num_tokens=num_tokens, hidden_dim=hidden_dim, num_ranks=num_ranks, flags=flags
    )
    # prepare for kernel_ops input
    if kernel_ops:
        out_shape = (*x.shape[:-1], x.shape[-1] )
        x_q = torch.empty(out_shape, dtype=dst_dtype, device=x.device)
        x_s = torch.empty(x.shape[:-1] + (x.shape[-1] // group_size,),
                      device=x.device,
                      dtype=torch.float32)
        info = torch.finfo(dst_dtype)
        bit8_max = info.max
        bit8_min = info.min
        eps=1e-10
        scale_ue8m0=False

    # Testing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]

    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')
    for _ in range(num_warmup):
        if kernel_ops:
            fn(x, x_q, x_s, group_size, eps, bit8_min, bit8_max, scale_ue8m0)
        else:
            fn(x, group_size, dst_dtype)
    torch.cuda.synchronize()

    for i in range(num_tests):
        cache.zero_()
        if kernel_ops:
            start_events[i].record()
            fn(x, x_q, x_s, group_size, eps, bit8_min, bit8_max, scale_ue8m0)
            end_events[i].record()
        else:
            start_events[i].record()
            fn(x, group_size, dst_dtype)
            end_events[i].record()

    torch.cuda.synchronize()
    total_times_msecs = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)])[1:]
    mean_time_msecs = np.average(total_times_msecs)
    print(f"func name: {fn.__name__}, {mean_time_msecs=} mseconds.")
    return mean_time_msecs

if __name__ == "__main__":
    num_tokens = [128, 256, 512, 1024, 4096, 8192, 16384]
    # hidden_dim = [128, 256, 384, 512, 1024, 1536, 1664, 2048, 4096, 7168, 16384]
    hidden_dims = [4096, 7168]
    group_size = 128
    kenrel_ops = try_load_library(verbose=False)
    for num_token in num_tokens:
        for hidden_dim in hidden_dims:
            print(f"num_token: {num_token}, hidden_dim: {hidden_dim}, group_size: {group_size}")
            res1 = test_per_token_group_quant_kernel(per_token_group_quant_fp8, num_token, hidden_dim, group_size)
            res2 = test_per_token_group_quant_kernel(triton_per_token_group_quant_8bit, num_token, hidden_dim, group_size, dst_dtype=torch.float8_e4m3fn)            
            res = test_per_token_group_quant_kernel(kenrel_ops.sgl_per_token_group_quant_8bit, num_token, hidden_dim, group_size, dst_dtype=torch.float8_e4m3fn, kernel_ops=True)
            print("[Triton] improvement: ", res1 / res2)
            print("[cuda] kernel_ops improvement: ", res1 / res)
            print("------------------------------------------------------------")
