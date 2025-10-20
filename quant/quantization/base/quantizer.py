import torch
class BaseQuantizer(torch.nn.Module):
    """Base class for quantizers, including fp8，awq, sq"""
    def __init__(self):
        super().__init__()
