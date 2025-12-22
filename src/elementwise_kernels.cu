//
// Created by Bujor Ionut Raul on 16.11.2025.
//

#include "elementwise_kernels.cuh"

/// Refactored templates for element-wise operations
/// 2 variants: contiguous and strided (views) arrays
template<typename dtype, typename Func>
__global__ void elementWiseKernelContiguous(
    dtype *output, const int out_offset,
    const int size,
    Func func,
    const dtype *input_a = nullptr, const int a_offset = 0,
    const dtype *input_b = nullptr, const int b_offset = 0)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[out_offset + idx] = func(
        input_a ? input_a[a_offset + idx]: dtype(0),
        input_b ? input_b[b_offset + idx]: dtype(0)
    );
}

template<typename dtype, typename Func>
__global__ void elementWiseKernelStrided(
    dtype *output, const int out_offset, const int *out_strides,
    const int size, const int ndim, const int *shape,
    Func func,
    const dtype *input_a = nullptr, const int a_offset = 0, const int *a_strides = nullptr,
    const dtype *input_b = nullptr, const int b_offset = 0, const int *b_strides = nullptr
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    // Convert flat index to multi index
    int multi_idx[33]; // maximum supported is 33 dims (like NumPy/CuPy)
    int remaining = idx;
    for (int i = ndim - 1; i >= 0; i--) {
        multi_idx[i] = remaining % shape[i];
        remaining /= shape[i];
    }
    int out_idx = out_offset;
    int a_idx = a_offset;
    int b_idx = b_offset;
    for (int i = 0; i < ndim; i++) {
        out_idx += multi_idx[i] * out_strides[i];
        a_idx += multi_idx[i] * a_strides[i];
        if (b_strides) b_idx += multi_idx[i] * b_strides[i];
    }
    output[out_idx] = func(
        input_a? input_a[a_idx]: dtype(0),
        input_b? input_b[b_idx]: dtype(0)
    );
}