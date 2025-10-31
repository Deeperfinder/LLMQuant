import torch
import triton
import triton.language as tl
from typing import List

def per_tensor_quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    finfo = torch.finfo(torch.float_e4m3fn)
    if tensor.numel() ==0:
        min_val, max_val = (
            torch.tensor(-16.0, dtype=tensor.dtype),
            torch.tensor(16.0, dtype=tensor.dtype),
        )
    else:
        min_val, max_val = tensor.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = amax.clamp(min=1e-12) / finfo.max
    qweight = (tensor / scale).clamp(min=finfo.min, max=finfo.max)
    scale = scale.float()
    return qweight, scale

def per_channel_quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    finfo = torch.finfo(torch.float8_e4m3fn)
    if tensor.numel() == 0:
        # Deal with empty tensors (triggered by empty MoE experts)
        print("[warning] You are experiencing empty MoE experts, tensor numbers = 0")
        qweight = torch.empty_like(tensor, dtype=torch.float8_e4m3fn)
        scales = torch.ones((*tensor.shape[:-1], 1), dtype=torch.float32)
        return qweight, scales
    amax = tensor.abs().max(dim=-1, keepdim=True)
    scale = amax.clamp(min=1e-12) / finfo.max
    qweight = (tensor / scale).clamp(min=finfo.min, max=finfo.max)
    scale = scale.float()
    return qweight, scale

def per_token_group_quant_fp8(x: torch.Tensor,
                              group_size: int, 
                              dtype: torch.float8_e4m3fn,
                              eps=1e-10) -> Tuple[torch.Tensor, float]:
    assert x.shape[-1] % group_size == 0,("The last dimension of 'x' cannot be divided by group_size!")
    assert x.is_contiguous(),("The input tensor must be contiguous!")

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    # x[M, N] ==> x_[M*N // group_size , group_size]
    x_ = x.reshape(x.numel() // group_size, group_size)
    # max 返回最大值和其索引[value, index]
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / fp8_max

    # x_[M*N, group_size] / x_s[M*N, 1]
    x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max).to(dtype)
    # x_q : [M * N, group_size] => [M, group_size *C]
    x_q = x_q.reshape(x.shape)
    # x_s : [M*N // group_size, 1] ==> [M, N // group_size] 
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1]//group_size,))

    return x_q, x_s

def _per_token_group_quant_8bit_raw(x: torch.Tensor, 
                                    group_size: int,
                                    eps: float = 1e-10,
                                    dtype: torch.dtype = torch.float8_e4m3fnm
                                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Function use triton to perform per-token-group quantization on an input tensor 'x'.
        Args:
            x: The input tenosr with ndim >= 2.
            group_size: The group size used for quantization.
            eps: The minimum to avoid dividing zero.
            dtype: The dype of output tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the scaling factor for quantization.
    """
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    info = torch.finfo(dtype)
    bit8_max = info.max
    bit8_min = info.min

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = torch.empty(x.shape[:-1] + (x.shape[-1] // group_size,),
                      device=x.device,
                      dtype=torch.float32)
    M = x.numel() // group_size
    N = group_size
    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK//256, 1), 8)
    num_stages = 1
    _per_token_group_quant_8bit[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        N,
        eps,
        bit8_min=bit8_min,
        bit8_max=bit8_max,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return x_q, x_s

@triton.jit
def _per_token_group_quant_8bit(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    y_stride,
    N,
    eps,
    bit8_min,
    bit8_max,
    BLOCK:tl.constexpr,
):
    """
        A triton-accelerated function to perform per-token-group quantization on a tensor.
        This function converts the tensor values into float8 values.
        notes:
            一个block的大小为y_stride == group_size
    """
    # Map the program id to the row of X and Y it should compute
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK) #N <= BLOCK
    mask = cols < N

    # 加载目标数据
    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / bit8_max
    y_s_inv = 1.0 / y_s
    y_q = tl.clamp(y*y_s_inv, bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)

def static_per_tensor_quantize(tensor: torch.Tensor, static_scale: float) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    qweight = (tensor / static_scale).clamp(min=finfo.min, max=finfo.max)
    return qweight.to(torch.float8_e4m3fn)

def fp8_gemm(A, A_scale, B, B_scale, bias, out_dtype):
    if A.numel() == 0:
        # Deal with empty tensors (triggeted by empty MoE experts)
        return torch.empty(size=(0, B.shape[0]), dtype=out_dtype, device=A.device)

    native_fp8_support=False
    if native_fp8_support:
        need_reshape = A.dim() == 3
        if need_reshape:
            batch_size = A.shape[0]
            A_input = A.reshape(-1, A.shape[-1])
        else:
            batch_size = None
            A_input = A
        output, _ = torch._scaled_mm(
            A_input, # [m, in]
            B.t(),  # [in, out]
            out_dtype = out_dtype,
            scale_a = A_scale,
            scale_b = B_scale,
            bias = bias,
        )
        if need_reshape:
            output = output.reshape(
                batch_size, output.shape[0] // batch_size, output.shape[1]
            )
        else:
            output = torch.nn.functional.linear(
                A.to(out_dtype) * A_scale.to(out_dtype),
                B.to(out_dtype) * B_scale.to(out_dtype),
                bias=bias,
            )
        return output

def native_w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
        block_size= [bN, bK]
        A 矩阵切分为[M x bK]
        B 矩阵切分为[bK x bN]
        先把A切为MxbK的tiles， B切成 bKxbN， C切成MxbN
    """
    A = A.to(compute_dtype)
    B = B.to(compute_dtype)

    assert A.shape[-1] == B.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert len(block_size) == 2

    block_n, block_k = block_size[0], block_size[1]
    # 判断最后一个维度相等，即 N / group_size = As.shape[-1] 
    assert (A.shape[-1] + block_k -1) // block_k == As.shape[-1]
    # 判断前面的 M 维度相等
    assert A.shape[:-1] == As.shape[:-1]

    # 将bs+token 维度reshape
    M = A.numel() // A.shape[-1]
    N, K = B.shape

    # N 维度添加到切片末尾
    origin_c_shape = A.shape[:-1] + (N, ) 
    A = A.reshape(M, A.shape[-1])
    As = As.reshape(M, As.shape[-1])
    n_tiles = (N + block_n -1) // block_n
    k_tiles = (K + block_k -1) // block_k

    assert n_tiles == Bs.shape[0], f"{n_tiles} == {Bs.shape[0]}"
    assert k_tiles == Bs.shape[1], f"{k_tiles} == {Bs.shape[1]}"

    C_shape = (M, N)
    C = torch.zeros(C_shape, dtype=compute_dtype, device=A.device)
    # A 按 K 维分块，每个tile是[M, block_k]
    A_tiles = [A[:, i*block_k : min((i+1)*block_k, K)] for i in range(k_tiles)]
    # B 按 N 和 K 维分块：B_tiles[j][i] = [block_n, block_k]
    B_tiles = [
        [
            B[
                j * block_n : min((j+1)*block_n, N),
                i * block_k : min((i+1)*block_k, K),
            ]
            for i in range(k_tiles)
        ]
        for j in range(n_tiles)
    ]
    C_tiles = [C[:, j*block_n : min((j+1)*block_n, N)] for j in range(n_tiles)]
    As_tiles = [As[:, i:i+1] for i in range(k_tiles)]

    for i in range(k_tiles):
        for j in range(n_tiles):
            a = A_tiles[i]
            b = B_tiles[j][i]
            c = C_tiles[j]
            s = As_tiles[i] * Bs[j][i]
            c[:, :] += torch.matmul(a, b.t()) * s
    C = C.reshape(origin_c_shape).to(output_dtype)
    return C

@triton.jit
def _w8a8_block_fp8_matmul(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
        Triton-accelerated function used to perform linear operations (dot
        product) on input tensors `A` and `B` with block-wise quantization, 
        and store the result in output tensor `C`.
    """  
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group)  // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K -k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K -k * BLOCK_SIZE_K, other=0.0)
        k_start = k*BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a,b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def w8a8_block_fp8_matmul_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """This function performs matrix multiplication with block-wise quantization.

    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.

    Args:
        A: The input tensor, e.g., activation.
        B: The input tensor, e.g., weight.
        As: The per-token-group quantization scale for `A`.
        Bs: The per-block quantization scale for `B`. [block_n, block_k]
        block_size: The block size for per-block quantization. It should be 2-dim, e.g., [128, 128].
        output_dytpe: The dtype of the returned tensor.

    Returns:
        torch.Tensor: The result of matmul.
    """
    block_n, block_k = block_size
    M = A.numel() // A.shape(-1)
    assert B.ndim == 2
    assert B.is_contiguous()
    assert Bs.ndim == 2
    N, K = B.shape
    C_shape = A.shape[:-1] + (N, )
    C = A.new_empty(C_shape, dtype=output_dtype)

    # default config
    config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_size[0],
            "BLOCK_SIZE_K": block_size[1],
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 3,
    }
    def grid(META):
        return{
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLCOK_SIZE_N"])
        }

    _w8a8_block_fp8_matmul[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
    )

    return C    
