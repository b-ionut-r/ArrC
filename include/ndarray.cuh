//
// Created by Bujor Ionut Raul on 16.11.2025.
//

#ifndef ARRC_NDARRAY_H
#define ARRC_NDARRAY_H

#include <iostream>
#include <vector>
#include <string>
#include <list>
#include "ndarray.cuh"
#include "elementwise_kernels.cuh"
#include "slices.h"
#include "utils.h"
#include "exceptions.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <variant>


/// FORWARD DECLARATIONS FOR OSTREAM
template <typename dtype>
class NDArray;
template <typename dtype>
ostream& operator<<(ostream &os, const NDArray<dtype> &arr);
template <typename dtype>
istream& operator>>(istream &is, NDArray<dtype> &arr);
///


template <typename dtype>
class NDArray{
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

    // Helpers
    void allocateDeviceMetadata(int** dStrides=nullptr,
                                int** dShape=nullptr) const;
public:
    using value_type = dtype;
    /// CONSTRUCTORS and DESTRUCTORS
    NDArray(); // default constructor
    NDArray(const vector<int> &shape); // alocator constructor
    void _computeStrides();
    NDArray(dtype *data, const vector<int> &shape, const int &offset,
            const vector<int> &strides); // viewer constructor
    NDArray(const NDArray<dtype> &other); // copy constructor
    NDArray(NDArray<dtype> &&other) noexcept; /* move constructor
    for returned rvalues views of ndarray. */
    NDArray& operator=(NDArray<dtype> &&other) noexcept; // move assignment
    ~NDArray();

    /// GETTERS and SETTERS (inline)
    dtype* getData() {return data;}
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
    static size_t getTotalAllocatedMemory() { return totalAllocatedMemory; }

    /// UTILITY FUNCTIONS
    bool isContiguous() const;

    /// OVERLOADED OPERATORS
    template <typename Op>
    NDArray<dtype> executeElementWise(Op op, const NDArray *other = nullptr,
                                      NDArray *final = nullptr) const;
    dtype& operator[](const std::vector<int>& idx);
    NDArray operator[](vector<Slice> slices);
    NDArray& operator=(const dtype &value);
    NDArray& operator=(const NDArray &other);
    NDArray operator+(const NDArray &other) const;
    NDArray operator+(const dtype &value) const;
    NDArray operator-() const;
    NDArray operator-(const NDArray &other) const;
    NDArray operator-(const dtype &value) const;
    NDArray operator*(const NDArray &other) const;
    NDArray operator*(const dtype &value) const;
    NDArray operator/(const NDArray &other) const;
    NDArray operator/(const dtype &value) const;
    friend ostream& operator<< <>(ostream &os, const NDArray<dtype> &arr);
    friend istream& operator>> <>(istream &is, NDArray<dtype> &arr);

    /// OTHERS
    template <typename newDtype>
    NDArray<newDtype> cast() const;
    NDArray zeros_like() const;
    NDArray ones_like() const;
};

template<typename dtype>
int NDArray<dtype>::idGenerator = 0; // static variable (initialized outside class)
template<typename dtype>
size_t NDArray<dtype>::totalAllocatedMemory = 0;


/// DEFINITIONS FOR TEMPLATES ALSO NEED TO BE IN HEADER ///

template<typename dtype>
NDArray<dtype>::NDArray():
    data(nullptr),
    shape({}),
    ndim(0),
    strides({}),
    itemBytes(sizeof(dtype)),
    offset(0),
    ownsData(false),
    id(++idGenerator),
    size(0),
    N_BLOCKS(0){
};


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
    cudaError_t err = cudaMallocManaged(&data, size * itemBytes);
    if (err != cudaSuccess) {
        throw CudaKernelException("OOM during NDArray allocation");
    }
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
    strides(other.strides), itemBytes(sizeof(dtype)),
    offset(0), ownsData(true), id(++idGenerator)
{
    N_BLOCKS = (size + N_THREADS - 1) / N_THREADS;
    _computeStrides();
    cudaError_t err = cudaMallocManaged(&data, size * itemBytes);
    if (err != cudaSuccess) {
        throw CudaKernelException("OOM during NDArray allocation");
    }
    totalAllocatedMemory += size * itemBytes;
    if (other.isContiguous() && other.offset == 0) {
        cudaMemcpy(data, other.data, size * itemBytes, cudaMemcpyDeviceToDevice);
    } else {
        NDArray<dtype> temp(data, shape, 0, strides); // temp view
        temp.executeElementWise(AssignOp<dtype>{}, &other, &temp);
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
    other.ndim = 0;
    other.size = 0;
    other.offset = 0;
    other.N_BLOCKS = 0;
}

template<typename dtype>
NDArray<dtype>& NDArray<dtype>::operator=(NDArray<dtype> &&other) noexcept {
    if (this != &other) {
        // Free existing resources
        if (ownsData && data != nullptr) {
            cudaFree(data);
            totalAllocatedMemory -= size * itemBytes;
        }
        // Transfer ownership
        data = other.data;
        shape = move(other.shape);
        ndim = other.ndim;
        size = other.size;
        strides = move(other.strides);
        itemBytes = other.itemBytes;
        offset = other.offset;
        ownsData = other.ownsData;
        N_BLOCKS = other.N_BLOCKS;
        id = other.id;
        // Nullify source
        other.data = nullptr;
        other.ownsData = false;
        other.ndim = 0;
        other.size = 0;
        other.offset = 0;
        other.N_BLOCKS = 0;
    }
    return *this;
}

template<typename dtype>
NDArray<dtype>::~NDArray() {
    shape.clear(); strides.clear();
    if (ownsData && data != nullptr) {
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
NDArray<dtype> NDArray<dtype>::executeElementWise(
    Op op,
    const NDArray<dtype> *other,
    NDArray<dtype> *final) const
{
    /* first, second result */
    bool allContig = this->isContiguous() && (other ? other->isContiguous() : true);
    /// HANDLE BROADCASTING
    NDArray<dtype> *result = nullptr;
    const NDArray<dtype> *first = nullptr;
    const NDArray<dtype> *second = nullptr;
    bool delFirst = false, delSecond = false, delResult = false;
    if (other == nullptr || final != nullptr) {
        first = this;
        second = other;
        result = final? final: new NDArray<dtype>(first->shape);
        if(final == nullptr) delResult = true;
    } else {
        auto info = getBroadcastInfo(*this, *other);
        if (info.aBroadcastAxes.empty()) first = this;
        else {
            vector<int> newStrides = this -> strides;
            for (int i = 0; i < info.aBroadcastAxes.size(); i++) {
                newStrides[info.aBroadcastAxes[i]] = 0;
            }
            first = new NDArray<dtype>(this->data, info.finalShape, this->offset, newStrides);
            delFirst = true;
        }
        if (info.bBroadcastAxes.empty()) second = other;
        else{
            vector<int> newStrides = other -> strides;
            for (int i = 0; i < info.bBroadcastAxes.size(); i++) {
                newStrides[info.bBroadcastAxes[i]] = 0;
            }
            second = new NDArray<dtype>(other->data, info.finalShape, other->offset, newStrides);
            delSecond = true;
        }
        result = new NDArray<dtype>(info.finalShape);
        delResult = true;
    }
    cudaError_t err = cudaSuccess;

    try {
        if (allContig) {
            elementWiseKernelContiguous<<<N_BLOCKS, N_THREADS>>>(
                result->data, result->offset, result->size,
                op,
                first->data, first->offset,
                second ? second->data : nullptr, second ? second->offset : 0
            );
            cudaDeviceSynchronize();
        } else {
            int *dResultShape = nullptr;
            int *dResultStrides = nullptr, *dFirstStrides = nullptr, *dSecondStrides = nullptr;

            try {
                result->allocateDeviceMetadata(&dResultStrides, &dResultShape);
                first->allocateDeviceMetadata(&dFirstStrides, nullptr);
                if (second) second->allocateDeviceMetadata(&dSecondStrides, nullptr);

                elementWiseKernelStrided<<<N_BLOCKS, N_THREADS>>>(
                    result->data, result->offset, dResultStrides,
                    result->size, result->ndim, dResultShape,
                    op,
                    first->data, first->offset, dFirstStrides,
                    second ? second->data : nullptr, second ? second->offset : 0, dSecondStrides
                );
                cudaDeviceSynchronize();
            } catch (...) {
                // Clean up device memory on exception
                if (dResultShape) cudaFree(dResultShape);
                if (dResultStrides) cudaFree(dResultStrides);
                if (dFirstStrides) cudaFree(dFirstStrides);
                if (dSecondStrides) cudaFree(dSecondStrides);
                throw;
            }

            cudaFreeMulti({dResultShape, dResultStrides, dFirstStrides});
            if (second) cudaFree(dSecondStrides);
        }

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw CudaKernelException(cudaGetErrorString(err));
        }
    } catch (...) {
        // Clean up temporary objects on any exception
        if (delFirst) delete first;
        if (delSecond) delete second;
        if (delResult) delete result;
        throw;
    }

    // Normal cleanup
    if (delFirst) delete first;
    if (delSecond) delete second;
    if (delResult) {
        NDArray retVal = std::move(*result);
        delete result;
        return retVal;
    } else {
        return *result;
    }
}


template <typename dtype>
NDArray<dtype>& NDArray<dtype>::operator=(const dtype &value) {
    executeElementWise(SetConstantOp<dtype>{value}, nullptr, this); // inplace execution
    return *this;  // Return reference to this, not the temporary from executeElementWise
}

template <typename dtype>
NDArray<dtype>& NDArray<dtype>::operator=(const NDArray<dtype> &other) {
    if (shape != other.shape)
        throw ShapeMismatchException("Cannot assign arrays of different shapes.");
    if (isContiguous() && other.isContiguous() && offset == 0 && other.offset == 0) {
        cudaMemcpy(data, other.data, size * itemBytes, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        return *this;
    }
    executeElementWise(AssignOp<dtype>{}, &other, this); // inplace execution
    return *this;  // Return reference to this, not the temporary from executeElementWise
}


template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator+(const NDArray<dtype> &other) const {
    return executeElementWise(AffineAddOp<dtype>{1, 1}, &other, nullptr);
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator+(const dtype &value) const {
    return executeElementWise(ScalarAddOp<dtype>{value}, nullptr, nullptr);
}

template <typename dtype>
NDArray<dtype> operator+(dtype value, const NDArray<dtype> &arr) {
    return arr + value;
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator-() const {
    return executeElementWise(ScalarMulOp<dtype>{static_cast<dtype>(-1)}, nullptr, nullptr);
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator-(const NDArray<dtype> &other) const {
    return executeElementWise(AffineAddOp<dtype>{static_cast<dtype>(1), static_cast<dtype>(-1)},
                              &other, nullptr);
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator-(const dtype &value) const {
    return executeElementWise(ScalarAddOp<dtype>{-value}, nullptr, nullptr);
}

template <typename dtype>
NDArray<dtype> operator-(const dtype &value, const NDArray<dtype> &arr) {
    return arr.executeElementWise(ScalarRSubOp<dtype>{value}, nullptr, nullptr);
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator*(const NDArray<dtype> &other) const {
    return executeElementWise(MulOp<dtype>{}, &other, nullptr);
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator*(const dtype &value) const {
    return executeElementWise(ScalarMulOp<dtype>{value}, nullptr, nullptr);
}

template <typename dtype>
NDArray<dtype> operator*(const dtype &value, const NDArray<dtype> &arr) {
    return arr * value;
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator/(const NDArray<dtype> &other) const {
    return executeElementWise(DivOp<dtype>{}, &other, nullptr);
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator/(const dtype &value) const {
    return executeElementWise(ScalarMulOp<dtype>{1/value}, nullptr, nullptr);
}

template <typename dtype>
NDArray<dtype> operator/(const dtype &value, const NDArray<dtype> &arr) {
    return arr.executeElementWise(ScalarRDivOp<dtype>{value}, nullptr, nullptr);
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
istream& operator>>(istream &is, NDArray<dtype> &arr) {
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
template <typename newDtype>
NDArray<newDtype> NDArray<dtype>::cast() const {
    NDArray<newDtype> result(shape);

    int* dShape = nullptr;
    int* dSrcStrides = nullptr;
    int* dDstStrides = nullptr;

    try {
        cudaMalloc(&dShape, ndim * sizeof(int));
        cudaMalloc(&dSrcStrides, ndim * sizeof(int));
        cudaMalloc(&dDstStrides, ndim * sizeof(int));

        cudaMemcpy(dShape, shape.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dSrcStrides, strides.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);
        auto resultStrides = result.getStrides();
        cudaMemcpy(dDstStrides, resultStrides.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);

        castKernel<newDtype, dtype><<<N_BLOCKS, N_THREADS>>>(
            result.getData(), 0, dDstStrides,
            data, offset, dSrcStrides,
            size, ndim, dShape
        );
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw CudaKernelException(cudaGetErrorString(err));
        }
    } catch (...) {
        if (dShape) cudaFree(dShape);
        if (dSrcStrides) cudaFree(dSrcStrides);
        if (dDstStrides) cudaFree(dDstStrides);
        throw;
    }

    cudaFree(dShape);
    cudaFree(dSrcStrides);
    cudaFree(dDstStrides);

    return result;
}


template<typename dtype>
NDArray<dtype> NDArray<dtype>::zeros_like() const {
    NDArray<dtype> zeros(shape);
    zeros = 0;
    return zeros;
}

template<typename dtype>
NDArray<dtype> NDArray<dtype>::ones_like() const {
    NDArray<dtype> ones(shape);
    ones = 1;
    return ones;
}


/// BROADCASTING HELPERS ///
template <typename dtype>
struct BroadcastInfo {
    vector<int> finalShape;  // container 1
    vector<int> aBroadcastAxes; // container 2
    vector<int> bBroadcastAxes; // container 2
};

template <typename dtype>
BroadcastInfo<dtype> getBroadcastInfo(const NDArray<dtype> &a, const NDArray<dtype> &b) {
    BroadcastInfo<dtype> out;
    // Quick checks
    const int n = a.getNDim();
    if (n != b.getNDim()) {
        throw NDimMismatchException("Arrays of different ndims. Cannot broadcast.");
    }
    if (a.getShape() == b.getShape()) {
        out.finalShape = a.getShape();
        return out;
    }
    out.finalShape.reserve(n);
    for (int i = 0; i < n; i++) {
        const int da = a.getShape()[i];
        const int db = b.getShape()[i]; 
        if (da == db) {
            out.finalShape.push_back(da);
        } else if (da == 1) {
            out.aBroadcastAxes.push_back(i);
            out.finalShape.push_back(db);
        } else if (db == 1) {
            out.bBroadcastAxes.push_back(i);
            out.finalShape.push_back(da);
        } else {
            throw ShapeMismatchException("Arrays of different shapes. Cannot broadcast");
        }
    }
    return out;
}

/// VARIANTS

namespace arr {
    template <typename dtype>
    using NDArray = NDArray<dtype>;
    using NDArrayVariant = std::variant<
        NDArray<int32_t>,
        NDArray<int64_t>,
        NDArray<size_t>,
        NDArray<float>,
        NDArray<double>,
        NDArray<__half>,
        NDArray<__nv_bfloat16>,
        NDArray<bool>
    >;
    using NDArrayPtrVariant = std::variant<
        NDArray<int32_t>*,
        NDArray<int64_t>*,
        NDArray<size_t>*,
        NDArray<float>*,
        NDArray<double>*,
        NDArray<__half>*,
        NDArray<__nv_bfloat16>*,
        NDArray<bool>*
    >;

    using NDArrayUniquePtrVariant = std::variant<
        std::unique_ptr<NDArray<int32_t>>,
        std::unique_ptr<NDArray<int64_t>>,
        std::unique_ptr<NDArray<size_t>>,
        std::unique_ptr<NDArray<float>>,
        std::unique_ptr<NDArray<double>>,
        std::unique_ptr<NDArray<__half>>,
        std::unique_ptr<NDArray<__nv_bfloat16>>,
        std::unique_ptr<NDArray<bool>>
    >;

    template <typename dtype>
    NDArray<dtype> make_constant(const vector<int> &shape, const dtype &value) {
        NDArray<dtype> array(shape);
        array = (dtype)value;
        return array;
    }

    template <typename dtype>
    NDArray<dtype> make_zeros(const vector<int> &shape) {
        return make_constant(shape, (dtype)0);
    }
    template <typename dtype>
    NDArray<dtype> make_ones(const vector<int> &shape) {
        return make_constant(shape, (dtype)1);
    }
}




#endif //ARRC_NDARRAY_H
