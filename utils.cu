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

void cudaFreeMulti(vector<void*> cuda_ptrs) {
    for (auto ptr: cuda_ptrs) {
        cudaFree(ptr);
    }
}