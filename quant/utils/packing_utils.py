import torch

AWQ_ORDER = [0, 2, 4, 5, 1, 3, 5, 7]
# 相当于把pack阶段reoder的weight再reoder成正常的01234567。 index-> loc
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

def unpack_awq(qweight: torch.Tensor, 
               qzeros: torch.Tensor,
               bits: int):
    shifts = torch.arange(0, 32, bits, device=qzeros.device) # 0, 4, 8, 12 ... 28

    # unpacking clumnwise
    iweights = torch.bitwise_right_shift(qweight[:,:, None], shifts[None, None, :]).to(torch.int8)
    iweights = iweights.view(iweights.shape[0], -1)

    # unpacking columnwise
    if qzeros is not None:
        izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(torch.int8)
        izeros = izeros.view(izeros.shape[0], -1)
    else:
        izeros = qzeros
    return iweights, izeros

def reverse_awq_order(iweights: torch.Tensor, 
                      izeros: torch.Tensor, 
                      bits: int):
    reverse_order_tensor = torch.arange(
        iweights.shape[-1],
        dtype = torch.int32,
        device=izeros.device,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[: AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    if izeros is not None:
        izeros = izeros[:, reverse_order_tensor]
    iweights = iweights[:, reverse_order_tensor]

    return iweights, izeros

# lienar_awq
def dequantize_gemm(
    qweight,
    qzeros,
    scales,
    bits,
    group_size
):
    iweight, izeros = unpack_awq(qweight, qzeros, bits) 
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)

    # overflow checks
    iweight = torch.bitwise_and(iweight, (2**bits) -1)
    izeros = torch.bitwise_and(izeros, (2**bits) -1)

    # fp16 weights
    # scales.shape = [40, 5120] => [5120 = 40 * 128, 5120]
    # weight.shape = [5120, 5120]
    scales = scales.repeat_interleave(group_size, dim=0)
    izeros = izeros.repeat_interleave(group_size, dim=0)
    fp_weight = (iweight - izeros) * scales

    return fp_weight
