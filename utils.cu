#include <vector>
using namespace std;

vector<int> getSizeAgnosticKernelConfigParams() {
    vector<int> params(2);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    params[0] = prop.multiProcessorCount * 4;
    params[1] = 256;
    return params;
}

int flatToStridedIndex(const int idx, const int offset, const vector<int> &strides,
                       int ndim, const vector<int> &shape) {
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

void cudaFreeMulti(vector<void*> cuda_ptrs) {
    for (auto ptr: cuda_ptrs) {
        cudaFree(ptr);
    }
}