#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>
#include "pml_cuda.h"

// Loosely based on https://pytorch.org/tutorials/advanced/cpp_extension.html

// CUDA forward declarations

#define CHECK_CUDA(x)                                                          \
    AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor>
pml_step_cuda( cudaStream_t  stream,
    torch::Tensor G,
    float stepRate,
    float stepAlpha,
    torch::Tensor wallMask,
    torch::Tensor Qx2,     
    torch::Tensor Qy2,     
    torch::Tensor Qz2,
    torch::Tensor PMLCoefs);

std::vector<torch::Tensor>
pml_step(
    torch::Tensor G,
    float stepRate,
    float stepAlpha,
    torch::Tensor wallMask,
    torch::Tensor Qx2,     
    torch::Tensor Qy2,     
    torch::Tensor Qz2,
    torch::Tensor PMLCoefs)
{
    CHECK_INPUT(G);
    CHECK_INPUT(wallMask);
    CHECK_INPUT(Qx2);
    CHECK_INPUT(Qy2);
    CHECK_INPUT(Qz2);
    CHECK_INPUT(PMLCoefs);

    c10::DeviceGuard guard(G.device());

    return pml_step_cuda(
        at::cuda::getCurrentCUDAStream(),
        G,
        stepRate,
        stepAlpha,
        wallMask,
        Qx2,     
        Qy2,     
        Qz2,
        PMLCoefs
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("pml_step", &pml_step, "FEM Grid PML (CUDA)");
}
