#include <vector>
#include <algorithm>
#include "utils.h"
#include <cuda_runtime.h>

std::vector<int> getSizeAgnosticKernelConfigParams() {
    std::vector<int> params(2);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    params[0] = prop.multiProcessorCount * 8;
    params[1] = 256;
    return params;
}

int getNBlocks(int n, int threads) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = (n + threads - 1) / threads;
    int maxBlocks = 8 * prop.multiProcessorCount;
    return std::min(blocks, maxBlocks); // hard cap
}

int flatToStridedIndex(const int idx, const int offset, const std::vector<int> &strides,
                       int ndim, const std::vector<int> &shape) {
    int multi_idx[33]; // maximum supported is 33 dims (like NumPy/CuPy)
    int remaining = idx;
    for (int i = ndim - 1; i >= 0; i--) {
        multi_idx[i] = remaining % shape[i];
        remaining /= shape[i];
    }
    int final_idx = offset;
    for (int i = 0; i < ndim; i++) {
        final_idx += multi_idx[i] * strides[i];
    }
    return final_idx;
}

void cudaFreeMulti(const std::vector<void*> &cuda_ptrs) {
    for (auto ptr: cuda_ptrs) {
        cudaFree(ptr);
    }
}

