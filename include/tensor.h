//
// Created by Bujor Ionut Raul on 22.12.2025.
//

#ifndef ARRC_TENSOR_H
#define ARRC_TENSOR_H

#include "ndarray.cuh"
#include <string>
#include <span>
#include <unordered_set>


class TensorBase{
public:
    virtual ~TensorBase() = 0;
};

template <typename dtype>
class Tensor: public TensorBase{
private:
    static size_t idGenerator;
    size_t id;
    NDArray<dtype> *data;
    NDArray<dtype> *grad = nullptr;
    auto gradFn = nullptr;
    bool requiresGrad;
public:
    using value_type = dtype;
    Tensor(NDArray<dtype> *data, const bool &requiresGrad = false):
    data(data), requiresGrad(requiresGrad), id(idGenerator++) {
        if (requiresGrad)
            grad = new NDArray<dtype>(data->getShape());
            *grad = (dtype)0;
    }
    ~Tensor(){
        delete data;
        if (requiresGrad) delete grad;
        idGenerator--;
    };
    NDArray<dtype>* getDataPtr() const {return data;}
    NDArray<dtype>* getGradPtr() const {return grad;}
    NDArray<dtype> getData() const {return *data;}
    NDArray<dtype> getGrad() const {return requiresGrad ? *grad : nullptr;}
    std::string getName() const {return "UnnamedTensor_" + std::to_string(id);}
    int getSize() const {return data->getSize();}
    vector<int> getShape() const {return data->getShape();}
    void zeroGrad() {delete grad; grad = nullptr;}
    void backward(NDArray<dtype> *grad = nullptr,
                  const bool retainGraph = false,
                  const int preserveAncestors = 4);
    template <typename newDtype>
    Tensor<newDtype> cast() const;
};

template <typename dtype>
size_t Tensor<dtype>::idGenerator = 0;


template <typename dtype>
void buildTopo(const Tensor<dtype> *tensor, std::vector<Tensor<dtype>*> &topoOrder,
                std::unordered_set<Tensor<dtype>*> &visited) {
    if (visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    if (tensor->gradFn != nullptr) {
        for (auto parent : tensor->gradFn->parents) {
            if (parent -> requiresGrad) {
                buildTopo(parent, topoOrder, visited);
            }
        }
    }
}

template <typename dtype>
void Tensor<dtype>::backward(NDArray<dtype> *grad,
                             const bool retainGraph,
                             const int preserveAncestors) {
    bool deleteGrad = false;
    if (!requiresGrad)
        throw std::runtime_error("Cannot do backprop for a tensor that does not require grad.");
    if (grad == nullptr) {
        if (getSize() != 1) {
            throw std::runtime_error("backward() can only be called for scalar outputs. "
                    "For non-scalar outputs, gradient must be provided.");
        }
        grad = new NDArray<dtype>(arr::make_ones<dtype>(getShape()));
        deleteGrad = true;
    }
    // Build computational graph using topological sort
    auto topoOrder = std::vector<Tensor<dtype>*>();
    auto visited = std::unordered_set<const Tensor<dtype>*>();
    buildTopo(this, topoOrder, visited);

    auto nodesToPreserve = std::vector<Tensor<dtype>*>();
    for (int i = topoOrder.size() - 1; i >= topoOrder.size() - preserveAncestors; i--) {
        nodesToPreserve.push_back(topoOrder[i]);
    };


    for (Tensor *tensor: reverse(topoOrder)) {
        if (tensor->gradFn == nullptr) continue;
        auto gradOutput = tensor->grad;
        vector<NDArray*>parentGrads = tensor->gradFn->backward(*gradOutput);
        for (int i = 0; i < parentGrads.size(); i++) {
           Tensor *parentTensor = tensor->gradFn->parents[i];
           NDArray<dtype> *parentGrad = parentGrads[i];
           if (parentTensor->requiresGrad() && parentGrad != nullptr) {
               if (parentTensor->grad == nullptr) {
                   parentTensor->grad = parentGrad;
               }
               else {
                   parentTensor->grad->executeElementWise(AffineAddOp<dtype>{1, 1},
                       parentGrad, parentTensor->grad); // inplace accumulation
               }
           }
        }
        if (!retainGraph) {
            bool isLeaf = tensor->gradFn == nullptr;
            bool isPreserved = false;
            if (nodesToPreserve.find(tensor) != nodesToPreserve.end()) {
                isPreserved = true;
            }
            if (!isLeaf && !isPreserved) {
                // delete tensor->data;
                // tensor->data = nullptr;
                // delete tensor->grad;
                // tensor->grad = nullptr;
                delete tensor;
            }
        }
    }
    if (deleteGrad) delete grad;
}


template <typename dtype>
template <typename newDtype>
Tensor<newDtype> Tensor<dtype>::cast() const {
    auto t = Tensor<newDtype>(
        new NDArray<newDtype>(data->template cast<newDtype>()),
        requiresGrad
    );
    if (requiresGrad) t.grad = new NDArray<newDtype>(grad->template cast<newDtype>());
    return t;
}


#endif //ARRC_TENSOR_H