//
// Created by Bujor Ionut Raul on 16.11.2025.
//

// ==================== SET CONSTANT ====================

// Fast version for contiguous arrays
template <typename dtype>
__global__ void setConstant(dtype *data, int offset, dtype value, const int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += gridStride){
        data[offset + i] = value;
    }
}

// Strided version for views
template <typename dtype>
__global__ void setConstantStrided(dtype *data, const int offset, const int* strides,
                                   const int* shape, const int ndim,
                                   dtype value, const int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;

    for (int flat_idx = idx; flat_idx < size; flat_idx += gridStride) {
        // Convert flat index to multi-dimensional indices
        int multi_idx[33]; // hardcoded limit of dims, matches NumPy/CuPy
        int remaining = flat_idx;
        for (int i = ndim - 1; i >= 0; i--) {
            multi_idx[i] = remaining % shape[i];
            remaining /= shape[i];
        }

        // Compute actual memory offset using strides
        int linear_idx = offset;
        for (int i = 0; i < ndim; i++) {
            linear_idx += multi_idx[i] * strides[i];
        }

        data[linear_idx] = value;
    }
}

// ==================== AFFINE ADD ====================

// Fast version for contiguous arrays
template <typename dtype>
__global__ void affineAdd(dtype *c, int c_offset,
                         const dtype *a, int a_offset,
                         const dtype *b, int b_offset,
                         const int size, dtype alpha=1, dtype beta=1)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += gridStride) {
        c[c_offset + i] = alpha * a[a_offset + i] + beta * b[b_offset + i];
    }
}

// Strided version
template <typename dtype>
__global__ void affineAddStrided(
    dtype *c, const int c_offset, const int* c_strides,
    const dtype *a, const int a_offset, const int* a_strides,
    const dtype *b, const int b_offset, const int* b_strides,
    const int* shape, const int ndim, const int size,
    dtype alpha=1, dtype beta=1)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;

    for (int flat_idx = idx; flat_idx < size; flat_idx += gridStride) {
        int multi_idx[33];
        int remaining = flat_idx;
        for (int i = ndim - 1; i >= 0; i--) {
            multi_idx[i] = remaining % shape[i];
            remaining /= shape[i];
        }

        int c_idx = c_offset;
        int a_idx = a_offset;
        int b_idx = b_offset;
        for (int i = 0; i < ndim; i++) {
            c_idx += multi_idx[i] * c_strides[i];
            a_idx += multi_idx[i] * a_strides[i];
            b_idx += multi_idx[i] * b_strides[i];
        }

        c[c_idx] = alpha * a[a_idx] + beta * b[b_idx];
    }
}

// ==================== SCALAR MULTIPLY ====================

// Fast version for contiguous arrays
template <typename dtype>
__global__ void scalarMul(dtype *b, int b_offset,
                         const dtype *a, int a_offset,
                         const int size, dtype alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += gridStride) {
        b[b_offset + i] = alpha * a[a_offset + i];
    }
}

// Strided version
template <typename dtype>
__global__ void scalarMulStrided(
    dtype *b, const int b_offset, const int* b_strides,
    const dtype *a, const int a_offset, const int* a_strides,
    const int* shape, const int ndim, const int size, dtype alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;

    for (int flat_idx = idx; flat_idx < size; flat_idx += gridStride) {
        int multi_idx[33];
        int remaining = flat_idx;
        for (int i = ndim - 1; i >= 0; i--) {
            multi_idx[i] = remaining % shape[i];
            remaining /= shape[i];
        }

        int b_idx = b_offset;
        int a_idx = a_offset;
        for (int i = 0; i < ndim; i++) {
            b_idx += multi_idx[i] * b_strides[i];
            a_idx += multi_idx[i] * a_strides[i];
        }

        b[b_idx] = alpha * a[a_idx];
    }
}

// ==================== ASSIGN ====================
// (Already strided, keep as is)

template <typename dtype>
__global__ void assign(
    dtype* dst, const int dst_offset, const int* dst_strides,
    const dtype* src, const int src_offset, const int* src_strides,
    const int* shape, const int ndim, const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;

    for (int flat_idx = idx; flat_idx < size; flat_idx += gridStride) {
        int multi_idx[33];
        int remaining = flat_idx;
        for (int i = ndim - 1; i >= 0; i--) {
            multi_idx[i] = remaining % shape[i];
            remaining /= shape[i];
        }

        int src_linear_idx = src_offset;
        int dst_linear_idx = dst_offset;
        for (int i = 0; i < ndim; i++) {
            src_linear_idx += multi_idx[i] * src_strides[i];
            dst_linear_idx += multi_idx[i] * dst_strides[i];
        }

        dst[dst_linear_idx] = src[src_linear_idx];
    }
}