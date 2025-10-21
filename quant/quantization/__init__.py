from .base.quantizer import BaseQuantizer
from .awq.quantizer import AwqQuantizer

method_to_quantizer: dict[str, type[BaseQuantizer]] = {
    "awq": AwqQuantizer,
}
def get_concrete_quantizer_cls(quant_method):
    return method_to_quantizer[quant_method]
