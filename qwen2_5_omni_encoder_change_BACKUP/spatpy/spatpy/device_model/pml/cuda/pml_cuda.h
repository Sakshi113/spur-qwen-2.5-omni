#ifndef PML_CUDA_H
#define PML_CUDA_H

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor>
pml_step(
    torch::Tensor G,
    float stepRate,
    float stepAlpha,
    torch::Tensor wallMask,
    torch::Tensor Qx2,     
    torch::Tensor Qy2,     
    torch::Tensor Qz2,
    torch::Tensor PMLCoefs);

#endif
