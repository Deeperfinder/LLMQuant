import os
import torch
from torch.utils.cpp_extension import load

def get_build_cuda_cflags(build_pkg: bool = True):
    extra_cuda_cflags = []
    # extra_cuda_cflags.append("-O3")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF_OPERATORS__")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF_CONVERSIONS__")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF2_OPERATORS__")
    extra_cuda_cflags.append("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
    # extra_cuda_cflags.append("--expt-relaxed-constexpr")
    # extra_cuda_cflags.append("--expt-extended-lambda")
    # extra_cuda_cflags.append("--use_fast_math")
    if not build_pkg:
      # 将-v选项传递给PTX assember
      # 输出内容： 
      #     register使用情况，
      #     每个block共享内存使用、
      #     常量内存使用、
      #     栈溢出：因寄存器不足而spill到local mem的次数
      extra_cuda_cflags.append("-Xptxas -v")
      extra_cuda_cflags.append("-diag-suppress 177")
    else:
      extra_cuda_cflags.append("--ptxas-options=-v")

    extra_cuda_cflags.append(f"-I /cfs_cloud_code/fiinnxu/flashinfer/include")
    extra_cuda_cflags.append(f"-I /cfs_cloud_code/fiinnxu/code/LLMQuant/quant/utils")
    # extra_cuda_cflags.append('-lcublas')
    return extra_cuda_cflags


def get_build_sources():
    build_sources = []
    build_sources.append('../csrc/per_token_group_quant_8bit.cu')
    build_sources.append('../csrc/kernel_ops.cc')


    return build_sources

def get_device_name():
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return device_name
def get_device_capability():
    return torch.cuda.get_device_capability(torch.cuda.current_device())

def build_from_sources(verbose: bool= False):
    torch_arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    # Load the CUDA kernel as a python module
    pretty_print_line(f"Loading lib on device: {get_device_name()}, "
                      f"capability: {get_device_capability()}, "
                      f": {torch_arch_list_env}")
    
    return load(name = 'kernel_ops',
                sources = get_build_sources(),  
                extra_cuda_cflags = get_build_cuda_cflags(False),
                extra_cflags = ["-std=c++17"],
                verbose=verbose
)

def pretty_print_line(m: str = "", sep: str = "-", width: int = 100):
    res_len = width - len(m)
    left_len = int(res_len / 2)
    right_len = res_len - left_len
    pretty_line = sep * left_len + m + sep * right_len
    print(pretty_line)




def try_load_library(verbose: bool = False):
    pretty_print_line("kernel_ops lib build from sources")
    gemm = build_from_sources(verbose=verbose)
    return gemm
