from .base.quantizer import BaseQuantizer
from .awq.quantizer import AwqQuantizer
from .sq.quantizer import SqQuantizer
method_to_quantizer: dict[str, type[BaseQuantizer]] = {
    "awq": AwqQuantizer,
    "sq": SqQuantizer,
}
def get_concrete_quantizer_cls(quant_method):
    return method_to_quantizer[quant_method]
