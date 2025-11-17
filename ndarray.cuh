//
// Created by Bujor Ionut Raul on 16.11.2025.
//

#ifndef ARRC_NDARRAY_H
#define ARRC_NDARRAY_H

#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include "kernels.cu"
#include "slices.cpp"
using namespace std;

///
inline cudaDeviceProp getDeviceProp() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop;
}

inline cudaDeviceProp dev_prop = getDeviceProp();
inline int N_BLOCKS = dev_prop.multiProcessorCount * 4;
inline int N_THREADS = 256;
///

/// FORWARD DECLARATIONS FOR OSTREAM
template <typename dtype>
class NDArray;
template <typename dtype>
ostream& operator<<(ostream &os, const NDArray<dtype> &arr);
///

template <typename dtype>
class NDArray {
protected:

    dtype *data;
    vector<int> shape; int ndim; int size;
    vector<int> strides;
    int itemBytes;
    int offset; bool ownsData;

    // Helper to allocate device memory for strides/shape
    void allocateDeviceArrays(int** dStrides, int** dShape) const {
        cudaMalloc(dStrides, ndim * sizeof(int));
        cudaMalloc(dShape, ndim * sizeof(int));
        cudaMemcpy(*dStrides, strides.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(*dShape, shape.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);
    }

public:

    /// CONSTRUCTORS and DESTRUCTORS
    NDArray() = default;
    NDArray(const vector<int> &shape);
    void _computeStrides();
    NDArray(dtype *data, const vector<int> &shape, const int &offset,
            const vector<int> &strides);
    ~NDArray();

    /// GETTERS and SETTERS (inline)
    dtype* getData() {return data;}
    void setData(dtype *new_data) {data = new_data;}
    vector<int> getShape() const {return shape;}
    int getNDim() const {return ndim;}
    int getSize() const {return size;}
    vector<int> getStrides() const {return strides;}
    void setStrides(const vector <int> &new_strides) {
        if (strides.size() != new_strides.size()) {
            throw runtime_error("Cannot set strides: new_strides.size() != strides.size()");
        }
        strides = new_strides;
    }
    int getItemBytes() const {return itemBytes;}
    int getOffset() const {return offset;}
    bool getOwnsData() const {return ownsData;}

    /// UTILITY FUNCTIONS
    bool isContiguous() const;

    /// OVERLOADED OPERATORS
    dtype& operator[](const std::vector<int>& idx);
    NDArray operator[](vector<Slice> slices);
    NDArray& operator=(const dtype &value);
    NDArray& operator=(const NDArray &other);
    NDArray operator+(const NDArray &other) const;
    NDArray operator-() const;
    NDArray operator-(const NDArray &other) const;
    friend ostream& operator<< <>(ostream &os, const NDArray<dtype> &arr);
};



/// DEFINITIONS FOR TEMPLATES ALSO NEED TO BE IN HEADER ///
template<typename dtype>
NDArray<dtype>::NDArray(const vector<int> &shape):
    shape(shape),
    ndim(shape.size()),
    strides(shape.size()),
    itemBytes(sizeof(dtype)),
    offset(0),
    ownsData(true) {
    size = shape[0];
    for (int i = 1; i < ndim; i++) {
        size *= shape[i];
    }
    _computeStrides();
    cudaMallocManaged(&data, size * itemBytes);
}

template<typename dtype>
void NDArray<dtype>::_computeStrides() {
    int prod = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides[i] = prod;
        prod *= shape[i];
    }
}

template<typename dtype>
NDArray<dtype>::NDArray(dtype *data, const vector<int> &shape, const int &offset, const vector<int> &strides):
    data(data),
    shape(shape),
    ndim(shape.size()),
    strides(strides),
    itemBytes(sizeof(dtype)),
    offset(offset),
    ownsData(false) {
    size = shape[0];
    for (int i = 1; i < ndim; i++) {
        size *= shape[i];
    }
};

template<typename dtype>
NDArray<dtype>::~NDArray() {
    shape.clear(); strides.clear();
    if (ownsData) {
        cudaFree(data);
    }
}

template <typename dtype>
bool NDArray<dtype>::isContiguous() const {
    if (ndim == 0) return true;
    int expected_stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        if (strides[i] != expected_stride) return false;
        expected_stride *= shape[i];
    }
    return true;
}

template<typename dtype>
dtype& NDArray<dtype>::operator[](const std::vector<int>& idx) {
    if (idx.size() != ndim) {
        throw runtime_error(to_string(ndim) + " indices are needed.");
    }
    int flat_idx = 0;
    for (int i = 0; i < ndim; i++) {
        flat_idx += strides[i] * idx[i];
    }
    return *(data + offset + flat_idx);
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator[](vector<Slice> slices) {
    int n_slices = slices.size();
    if (n_slices > ndim) {
        throw runtime_error("Too many slices. Only " + to_string(ndim) +
            " slices are needed.");
    }
    for (int i = 0; i < n_slices; i++) {
        slices[i].normalizeEnd(shape[i]);
    }
    while (n_slices < ndim) {
        slices.push_back(Slice(0, shape[n_slices], 1));
        n_slices++;
    }
    vector<int> new_shape(ndim);
    vector<int> new_strides(ndim);
    int ptr_offset = offset;
    for (int i = 0; i < ndim; i++) {
        int start = slices[i].getStart();
        int step = slices[i].getStep();
        ptr_offset += start * strides[i];
        new_strides[i] = step * strides[i];
        new_shape[i] = slices[i].size();
    }
    NDArray<dtype> result(data, new_shape, ptr_offset, new_strides);
    return result;
}

template <typename dtype>
NDArray<dtype>& NDArray<dtype>::operator=(const dtype &value) {
    if (isContiguous()) {
        setConstant<<<N_BLOCKS, N_THREADS>>>(data, offset, value, size);
    } else {
        int *dStrides, *dShape;
        allocateDeviceArrays(&dStrides, &dShape);
        setConstantStrided<<<N_BLOCKS, N_THREADS>>>(
            data, offset, dStrides, dShape, ndim, value, size);
        cudaFree(dStrides);
        cudaFree(dShape);
    }
    cudaDeviceSynchronize();
    return *this;
}

template <typename dtype>
NDArray<dtype>& NDArray<dtype>::operator=(const NDArray<dtype> &other) {
    if (shape != other.shape) {
        throw runtime_error("Cannot assign array of shape " + to_string(other.shape[0]) +
        " to array of shape " + to_string(shape[0]));
    }

    if (isContiguous() && other.isContiguous() && offset == 0 && other.offset == 0) {
        cudaMemcpy(data, other.data, size * itemBytes, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        return *this;
    }

    int *dStrides, *dOtherStrides, *dShape;
    allocateDeviceArrays(&dStrides, &dShape);
    cudaMalloc(&dOtherStrides, other.ndim * sizeof(int));
    cudaMemcpy(dOtherStrides, other.strides.data(), other.ndim * sizeof(int), cudaMemcpyHostToDevice);

    assign<<<N_BLOCKS, N_THREADS>>>(
        data, offset, dStrides,
        other.data, other.offset, dOtherStrides,
        dShape, ndim, size
    );
    cudaDeviceSynchronize();
    cudaFree(dStrides);
    cudaFree(dOtherStrides);
    cudaFree(dShape);
    return *this;
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator+(const NDArray<dtype> &other) const {
    if (shape != other.shape) {
        throw runtime_error("Cannot perform elementwise addition on 2 "
                            "arrays with different shapes.");
    }
    NDArray<dtype> result(shape);

    if (isContiguous() && other.isContiguous()) {
        affineAdd<<<N_BLOCKS, N_THREADS>>>(
            result.data, result.offset,
            data, offset,
            other.data, other.offset,
            size);
    } else {
        int *dResultStrides, *dStrides, *dOtherStrides, *dShape;
        result.allocateDeviceArrays(&dResultStrides, &dShape);
        allocateDeviceArrays(&dStrides, &dShape);  // dShape already allocated, just update dStrides
        cudaFree(dShape);  // Free the duplicate
        allocateDeviceArrays(&dStrides, &dShape);  // Reallocate both
        cudaMalloc(&dOtherStrides, other.ndim * sizeof(int));
        cudaMemcpy(dOtherStrides, other.strides.data(), other.ndim * sizeof(int), cudaMemcpyHostToDevice);

        affineAddStrided<<<N_BLOCKS, N_THREADS>>>(
            result.data, result.offset, dResultStrides,
            data, offset, dStrides,
            other.data, other.offset, dOtherStrides,
            dShape, ndim, size);

        cudaFree(dResultStrides);
        cudaFree(dStrides);
        cudaFree(dOtherStrides);
        cudaFree(dShape);
    }
    cudaDeviceSynchronize();
    return result;
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator-() const {
    NDArray<dtype> result(shape);

    if (isContiguous()) {
        scalarMul<<<N_BLOCKS, N_THREADS>>>(
            result.data, result.offset,
            data, offset,
            size, dtype(-1));
    } else {
        int *dResultStrides, *dStrides, *dShape;
        result.allocateDeviceArrays(&dResultStrides, &dShape);
        cudaFree(dShape);  // Free duplicate
        allocateDeviceArrays(&dStrides, &dShape);

        scalarMulStrided<<<N_BLOCKS, N_THREADS>>>(
            result.data, result.offset, dResultStrides,
            data, offset, dStrides,
            dShape, ndim, size, dtype(-1));

        cudaFree(dResultStrides);
        cudaFree(dStrides);
        cudaFree(dShape);
    }
    cudaDeviceSynchronize();
    return result;
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator-(const NDArray<dtype> &other) const {
    if (shape != other.shape) {
        throw runtime_error("Cannot perform elementwise subtraction on 2 "
                            "arrays with different shapes.");
    }
    NDArray<dtype> result(shape);

    if (isContiguous() && other.isContiguous()) {
        affineAdd<<<N_BLOCKS, N_THREADS>>>(
            result.data, result.offset,
            data, offset,
            other.data, other.offset,
            size, dtype(1), dtype(-1));
    } else {
        int *dResultStrides, *dStrides, *dOtherStrides, *dShape;
        result.allocateDeviceArrays(&dResultStrides, &dShape);
        cudaFree(dShape);
        allocateDeviceArrays(&dStrides, &dShape);
        cudaMalloc(&dOtherStrides, other.ndim * sizeof(int));
        cudaMemcpy(dOtherStrides, other.strides.data(), other.ndim * sizeof(int), cudaMemcpyHostToDevice);

        affineAddStrided<<<N_BLOCKS, N_THREADS>>>(
            result.data, result.offset, dResultStrides,
            data, offset, dStrides,
            other.data, other.offset, dOtherStrides,
            dShape, ndim, size, dtype(1), dtype(-1));

        cudaFree(dResultStrides);
        cudaFree(dStrides);
        cudaFree(dOtherStrides);
        cudaFree(dShape);
    }
    cudaDeviceSynchronize();
    return result;
}

template <typename dtype>
ostream& operator<<(ostream &os, const NDArray<dtype> &arr) {
    os << "[";
    for (int i = 0; i < arr.size; i++) {
        for (int j = 0; j < arr.ndim-1; j++) {
            if (arr.strides[j] != 0 && i % arr.strides[j] == 0) {
                os << "[";
            }
        }
        os << arr.data[arr.offset + i];
        bool any_close = false;
        for (int j = arr.ndim - 2; j >= 0; j--) {
            if (arr.strides[j] != 0 && i % arr.strides[j] == arr.strides[j] - 1) {
                os << "]";
                any_close = true;
            }
        }
        if (any_close && i!=arr.size-1) os << endl;
        if (!any_close) os << ", ";
    }
    os << "]";
    return os;
}

#endif //ARRC_NDARRAY_H