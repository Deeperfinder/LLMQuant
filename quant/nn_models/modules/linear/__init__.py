from .linear_base import LinearBase
from .linear_awq import AWQLinear_GEMM
method_to_linear: dict[str, type[LinearBase]] = {
    "awq": AWQLinear_GEMM,
    #TODO: support more grained quantization
}
def get_concrete_linear_module(quant_method):
    return method_to_linear[quant_method]
