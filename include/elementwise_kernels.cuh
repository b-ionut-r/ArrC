//
// Created by Bujor Ionut Raul on 22.12.2025.
//

#ifndef ARRC_ELEMENTWISE_KERNELS_CUH
#define ARRC_ELEMENTWISE_KERNELS_CUH

/// Refactored templates for element-wise operations
/// 2 variants: contiguous and strided (views) arrays
template<typename dtype, typename Func>
__global__ void elementWiseKernelContiguous(
    dtype *output, const int out_offset,
    const int size,
    Func func,
    const dtype *input_a = nullptr, const int a_offset = 0,
    const dtype *input_b = nullptr, const int b_offset = 0
    );

template<typename dtype, typename Func>
__global__ void elementWiseKernelStrided(
    dtype *output, const int out_offset, const int *out_strides,
    const int size, const int ndim, const int *shape,
    Func func,
    const dtype *input_a = nullptr, const int a_offset = 0, const int *a_strides = nullptr,
    const dtype *input_b = nullptr, const int b_offset = 0, const int *b_strides = nullptr
);

/// FUNCTORS
template <typename dtype>
struct SetConstantOp {
    dtype value;
    __device__ dtype operator()(dtype, dtype) const {
        return value;
    }
};
template <typename dtype>
struct AssignOp {
    __device__ dtype operator()(dtype a, dtype b) const {
        return b;
    }
};

template <typename dtype>
struct ScalarAddOp {
    dtype scalar;
    __device__ dtype operator()(dtype a, dtype) const {
        return a + scalar;
    }
};

template <typename dtype>
struct ScalarMulOp {
    dtype scalar;
    __device__ dtype operator()(dtype a, dtype) const {
        return a * scalar;
    }
};

template <typename dtype>
struct ScalarRSubOp {
    dtype scalar;
    __device__ dtype operator()(dtype a, dtype) const {
        return scalar - a;
    }
};

template <typename dtype>
struct ScalarRDivOp {
    dtype scalar;
    __device__ dtype operator()(dtype a, dtype) const {
        return scalar / a;
    }
};

template <typename dtype>
struct AffineAddOp {
    dtype alpha, beta;
    __device__ dtype operator()(dtype a, dtype b) const {
        return alpha * a + beta * b;
    }
};

template <typename dtype>
struct MulOp {
    dtype scalar = 1;
    __device__ dtype operator()(dtype a, dtype b) const {
        return scalar * a * b;
    }
};

template <typename dtype>
struct DivOp {
    dtype scalar = 1;
    __device__ dtype operator()(dtype a, dtype b) const {
        return scalar * a / b;
    }
};
#endif //ARRC_ELEMENTWISE_KERNELS_CUH