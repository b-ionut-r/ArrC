//
// Created by Bujor Ionut Raul on 16.11.2025.
//

#ifndef ARRC_NDARRAY_H
#define ARRC_NDARRAY_H

#include <iostream>
#include <vector>
#include <string>
#include "kernels.cu"
#include "slices.cpp"
#include "utils.cu"
#include "exceptions.h"
using namespace std;


/// FORWARD DECLARATIONS FOR OSTREAM
template <typename dtype>
class NDArray;
template <typename dtype>
ostream& operator<<(ostream &os, const NDArray<dtype> &arr);
template <typename dtype>
istream& operator>>(istream &is, NDArray<dtype> &arr);
///

template <typename dtype>
class NDArray {
protected:
    dtype *data;
    vector<int> shape; int ndim; int size;
    vector<int> strides;
    int itemBytes;
    int offset; bool ownsData;
    int N_BLOCKS; int N_THREADS = 256;
    int id;
    // Static members
    static int idGenerator;
    static size_t totalAllocatedMemory; // GPU Memory
    static size_t getTotalAllocatedMemory(){return totalAllocatedMemory;}
    // Helpers
    void allocateDeviceMetadata(int** dStrides=nullptr,
                                int** dShape=nullptr) const;
    template <typename Op>
    void executeElementWise(Op op, const NDArray *result,
                            const NDArray *other = nullptr) const;
public:
    /// CONSTRUCTORS and DESTRUCTORS
    NDArray() = default;
    NDArray(const vector<int> &shape); // alocator constructor
    void _computeStrides();
    NDArray(dtype *data, const vector<int> &shape, const int &offset,
            const vector<int> &strides); // viewer constructor
    NDArray(const NDArray<dtype> &other); // copy constructor
    NDArray(NDArray<dtype> &&other) noexcept; /* move constructor
    for returned rvalues views of ndarray. */
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
            throw NDimMismatchException("New strides vector must have "
                                        "same size as old strides vector.");
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
    friend istream& operator>> <>(istream &is, NDArray<dtype> &arr);
};

template<typename dtype>
int NDArray<dtype>::idGenerator = 0; // static variable is initialized outside class
template<typename dtype>
size_t NDArray<dtype>::totalAllocatedMemory = 0;


/// DEFINITIONS FOR TEMPLATES ALSO NEED TO BE IN HEADER ///

template<typename dtype>
NDArray<dtype>::NDArray(const vector<int> &shape):
    shape(shape),
    ndim(shape.size()),
    strides(shape.size()),
    itemBytes(sizeof(dtype)),
    offset(0),
    ownsData(true),
    id(++idGenerator)
{
    size = shape[0];
    for (int i = 1; i < ndim; i++) {
        size *= shape[i];
    }
    N_BLOCKS = (size + N_THREADS - 1) / N_THREADS;
    _computeStrides();
    cudaMallocManaged(&data, size * itemBytes);
    totalAllocatedMemory += size * itemBytes;
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
    ownsData(false),
    id(++idGenerator)
{
    size = shape[0];
    for (int i = 1; i < ndim; i++) {
        size *= shape[i];
    }
    N_BLOCKS = (size + N_THREADS - 1) / N_THREADS;
};


template<typename dtype>
NDArray<dtype>::NDArray(const NDArray<dtype> &other):
    shape(other.shape), ndim(other.ndim), size(other.size),
    strides(other.ndim), itemBytes(sizeof(dtype)),
    offset(0), ownsData(true), id(++idGenerator)
{
    N_BLOCKS = (size + N_THREADS - 1) / N_THREADS;
    _computeStrides();
    cudaMallocManaged(&data, size * itemBytes);
    totalAllocatedMemory += size * itemBytes;
    if (other.isContiguous() && other.offset == 0) {
        cudaMemcpy(data, other.data, size * itemBytes, cudaMemcpyDeviceToDevice);
    } else {
        NDArray<dtype> temp(data, shape, 0, strides); // temp view
        temp.executeElementWise(AssignOp<dtype>{}, &temp, &other);
        temp.ownsData = false;
    }
    cudaDeviceSynchronize();
}


template<typename dtype>
NDArray<dtype>::NDArray(NDArray<dtype> &&other) noexcept:
    data(other.data), shape(move(other.shape)),
    ndim(other.ndim), size(other.size),
    strides(move(other.strides)), itemBytes(other.itemBytes),
    offset(other.offset), ownsData(other.ownsData),
    N_BLOCKS(other.N_BLOCKS), id(other.id) // "steals" id of rvalue
{
    other.data = nullptr;
    other.ownsData = false;
}


template<typename dtype>
NDArray<dtype>::~NDArray() {
    shape.clear(); strides.clear();
    if (ownsData) {
        cudaFree(data);
        totalAllocatedMemory -= size * itemBytes;
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
        throw IndexingException(to_string(ndim) + " indices are needed.");
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
        throw IndexingException("Too many slices. Only " + to_string(ndim)
            + " slices are needed.");
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
    NDArray<dtype> result(data, new_shape, ptr_offset, new_strides); // view
    return result;
}



template <typename dtype>
template <typename Op>
void NDArray<dtype>::executeElementWise(
    Op op,
    const NDArray<dtype> *result,
    const NDArray<dtype> *other
    ) const {
    bool allContig = isContiguous();
    if (other) {
        allContig = allContig && other->isContiguous();
    }
    if (allContig) {
        elementWiseKernelContiguous<<<N_BLOCKS, N_THREADS>>>(
            result->data, result->offset, result->size,
            op,
            this->data, this->offset,
            other ? other->data : nullptr, other ? other->offset : 0
        );
        cudaDeviceSynchronize();
    } else {
        int *dResultStrides, *dStrides, *dOtherStrides = nullptr, *dShape;
        result->allocateDeviceMetadata(&dResultStrides, &dShape);
        allocateDeviceMetadata(&dStrides, nullptr);
        if (other) {
            other->allocateDeviceMetadata(&dOtherStrides, nullptr);
        }
        elementWiseKernelStrided<<<N_BLOCKS, N_THREADS>>>(
            result->data, result->offset, dResultStrides,
            result->size, result->ndim, dShape,
            op,
            this->data, this->offset, dStrides,
            other ? other->data : nullptr, other ? other->offset : 0, dOtherStrides
        );
        cudaDeviceSynchronize();
        if (other) {
            cudaFreeMulti({dResultStrides, dStrides, dOtherStrides, dShape});
        } else {
            cudaFreeMulti({dResultStrides, dStrides, dShape});
        }
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw CudaKernelException(cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}


template <typename dtype>
NDArray<dtype>& NDArray<dtype>::operator=(const dtype &value) {
    executeElementWise(SetConstantOp<dtype>{value}, this, nullptr); // inplace execution
    return *this;
}

template <typename dtype>
NDArray<dtype>& NDArray<dtype>::operator=(const NDArray<dtype> &other) {
    if (shape != other.shape) {
        throw ShapeMismatchException("Cannot assign array of shape " + to_string(other.shape[0]) +
        " to array of shape " + to_string(shape[0]));
    }
    if (isContiguous() && other.isContiguous() && offset == 0 && other.offset == 0) {
        cudaMemcpy(data, other.data, size * itemBytes, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        return *this;
    }
    executeElementWise(AssignOp<dtype>{}, this,  &other); // inplace execution
    return *this;
}


template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator+(const NDArray<dtype> &other) const {
    if (shape != other.shape) {
        throw ShapeMismatchException("Cannot perform elementwise addition on 2 "
                            "arrays with different shapes.");
    }
    NDArray<dtype> result(shape);
    executeElementWise(AffineAddOp<dtype>{1, 1}, &result, &other);
    return result;
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator-() const {
    NDArray<dtype> result(shape);
    executeElementWise(ScalarMulOp<dtype>{-1}, &result, nullptr);
    return result;
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator-(const NDArray<dtype> &other) const {
    if (shape != other.shape) {
        throw ShapeMismatchException("Cannot perform elementwise subtraction on 2 "
                            "arrays with different shapes.");
    }
    NDArray<dtype> result(shape);
    executeElementWise(AffineAddOp<dtype>{1, -1}, &result, &other);
    return result;
}


template <typename dtype>
ostream& operator<<(ostream &os, const NDArray<dtype> &arr) {
    if (arr.size == 0) {
        os << "[]";
        return os;
    }
    vector<int> multi_idx(arr.ndim);
    for (int i = 0; i < arr.size; i++) {
        int remaining = i;
        for (int d = arr.ndim - 1; d >= 0; d--) {
            multi_idx[d] = remaining % arr.shape[d];
            remaining /= arr.shape[d];
        }
        if (i == 0) {
            for(int d = 0; d < arr.ndim; ++d) os << "[";
        }
        else {
            for (int d = 0; d < arr.ndim - 1; d++) {
                if (multi_idx[d + 1] == 0) {
                    if (d == arr.ndim - 2) os << endl;
                    os << "[";
                } else {
                    break;
                }
            }
        }
        int data_idx = arr.offset;
        for (int d = 0; d < arr.ndim; d++) {
            data_idx += multi_idx[d] * arr.strides[d];
        }
        os << arr.data[data_idx];
        bool any_close = false;
        for (int d = arr.ndim - 1; d >= 0; d--) {
            if (multi_idx[d] == arr.shape[d] - 1) {
                os << "]";
                any_close = true;
            } else {
                break;
            }
        }
        if (any_close && i != arr.size - 1) {
        }
        if (!any_close) {
            os << ", ";
        }
    }
    return os;
}

template <typename dtype>
istream& operator>>(istream &is, const NDArray<dtype> &arr) {
    if (arr.ownsData) {
        for (int i = 0; i < arr.size; i++) {
            is >> arr.data[i];
        }
    }
    else{
        for (int i = 0; i < arr.size; i++) {
            int real_idx = flatToStridedIndex(i, arr.offset, arr.strides,
                arr.ndim, arr.shape);
            is >> arr.data[real_idx];
        }
    }
    return is;
}


// Helper to allocate device memory for strides/shape
template <typename dtype>
void NDArray<dtype>::allocateDeviceMetadata(int** dStrides, int** dShape) const {
    if (dStrides != nullptr) {
        cudaMalloc(dStrides, ndim * sizeof(int));
        cudaMemcpy(*dStrides, strides.data(),
                ndim * sizeof(int), cudaMemcpyHostToDevice);
    }
    if (dShape != nullptr) {
        cudaMalloc(dShape, ndim * sizeof(int));
        cudaMemcpy(*dShape, shape.data(),
            ndim * sizeof(int), cudaMemcpyHostToDevice);
    }
}



template <typename dtype>
vector<vector<int>> getBroadcastingDims(const NDArray<dtype> &a , const NDArray<dtype> &b) {
    /// Helper function that returns the final shape after broadcasting,
    /// and the axes that need broadcasting in each array for an
    /// elementwise operation.
    int n = a.getNDim(); int f_shape;
    vector<int> final_shape(n); vector<int> a_broadcast(n);  vector<int> b_broadcast(n);
    if (a.getNDim() != b.getNDim()) {
        return vector{final_shape, a_broadcast, b_broadcast};
    }
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            if (a[i] == 1) {
                a_broadcast.push_back(i);
                f_shape = b[i];
            }
            if (b[i] == 1) {
                b_broadcast.push_back(i);
                f_shape = a[i];
            }
        } else {
            f_shape = a[i];
        }
        final_shape.push_back(f_shape);
    }
    return vector{final_shape, a_broadcast, b_broadcast};
}

#endif //ARRC_NDARRAY_H
