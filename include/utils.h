#include <vector>


std::vector<int> getSizeAgnosticKernelConfigParams();

int getNBlocks(int n, int threads);

int flatToStridedIndex(int idx, int offset,
                       const std::vector<int>& strides,
                       int ndim,
                       const std::vector<int>& shape);

void cudaFreeMulti(const std::vector<void*>& cuda_ptrs);