#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/extension.h>

#define STRINGFY(str) #str

void sgl_per_token_group_quant_8bit(
    at::Tensor input,
    at::Tensor output_q,
    at::Tensor output_s,
    int64_t group_size,
    double eps,
    double fp8_min,
    double fp8_max,
    bool scale_ue8m0);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("sgl_per_token_group_quant_8bit", &sgl_per_token_group_quant_8bit, STRINGFY(sgl_per_token_group_quant_8bit));
}
