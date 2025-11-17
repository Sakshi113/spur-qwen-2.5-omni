#ifndef TORCH_CUARRAYS_H
#define TORCH_CUARRAYS_H

#include <torch/extension.h>

/* convenience aliases for torch guff for working with cuda kernels */

template <typename T, size_t N>
using CuArray = torch::PackedTensorAccessor32<T, N>;

#define cuscalar_t scalar_t

typedef short cuint_t;
typedef long cutime_t;

template <typename T, size_t N>
CuArray<T, N>
cuarr(torch::Tensor x)
{
    return x.packed_accessor32<T, N>();
}

#define STRINGIFY_BASE(x) #x
#define STRINGIFY(x) STRINGIFY_BASE(x)

#define CUARRAY_KERNEL(FN, T, stream, blocks, threads, ...)                    \
    AT_DISPATCH_FLOATING_TYPES(T, STRINGIFY(fn##_op), ([&] {                   \
                                   FN<cuscalar_t, cuint_t>                     \
                                       <<<blocks, threads, 0, stream>>>(       \
                                           __VA_ARGS__);                       \
                               }));

#endif
