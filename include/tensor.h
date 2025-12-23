//
// Created by Bujor Ionut Raul on 22.12.2025.
//

#ifndef ARRC_TENSOR_H
#define ARRC_TENSOR_H

#include "ndarray.cuh"
#include <string>
#include <span>
#include <unordered_set>
#include <algorithm>

// Forward declaration
class Function;


template <typename dtype>
class Tensor{
private:
    static size_t idGenerator;
    size_t id;
    NDArray<dtype> *data;
    NDArray<dtype> *grad = nullptr;
    class Function* gradFn = nullptr;
    bool requiresGrad;
public:
    using value_type = dtype;
    Tensor(NDArray<dtype> *data, const bool &requiresGrad = false, Function* gradFn = nullptr):
    data(data), requiresGrad(requiresGrad), gradFn(gradFn), id(idGenerator++) {
        if (requiresGrad){
            grad = new NDArray<dtype>(data->getShape());
            *grad = (dtype)0;
        }
    }
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor(){
        // Clean up Function from registry if not already cleaned by backward()
        if (gradFn != nullptr) {
            Function::unregister_function(gradFn);
            gradFn = nullptr;
        }
        if (data) delete data;
        if (grad) delete grad;  // Always delete if allocated, regardless of requiresGrad
    };
    NDArray<dtype>* getDataPtr() const {return data;}
    NDArray<dtype>* getGradPtr() const {return grad;}
    NDArray<dtype> getData() const {return *data;}
    const NDArray<dtype>* getGrad() const {return requiresGrad ? grad : nullptr;}
    std::string getName() const {return "UnnamedTensor_" + std::to_string(id);}
    int getSize() const {return data->getSize();}
    vector<int> getShape() const {return data->getShape();}
    void zeroGrad() {
        if (requiresGrad && grad) {
            *grad = (dtype)0;
        }
    }
    void backward(NDArray<dtype> *grad = nullptr,
                  const bool retainGraph = false,
                  const int preserveAncestors = 4);
    template <typename newDtype>
    Tensor<newDtype> cast() const;
};

template <typename dtype>
size_t Tensor<dtype>::idGenerator = 0;

template <typename dtype>
Tensor<dtype>::Tensor(Tensor&& other) noexcept
    : id(other.id),
      data(other.data),
      grad(other.grad),
      gradFn(other.gradFn),
      requiresGrad(other.requiresGrad) {
    other.data = nullptr;
    other.grad = nullptr;
    other.gradFn = nullptr;
    other.requiresGrad = false;
}

template <typename dtype>
Tensor<dtype>& Tensor<dtype>::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        // Clean up existing resources
        if (gradFn != nullptr) {
            Function::unregister_function(gradFn);
        }
        if (data) delete data;
        if (grad) delete grad;
        // Transfer ownership
        data = other.data;
        grad = other.grad;
        gradFn = other.gradFn;
        requiresGrad = other.requiresGrad;
        id = other.id;
        // Nullify source
        other.data = nullptr;
        other.grad = nullptr;
        other.gradFn = nullptr;
        other.requiresGrad = false;
    }
    return *this;
}


template <typename dtype>
void buildTopo(Tensor<dtype> *tensor, std::vector<Tensor<dtype>*> &topoOrder,
                std::unordered_set<Tensor<dtype>*> &visited) {
    if (tensor == nullptr || visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    if (tensor->gradFn != nullptr) {
        for (const auto &parent_variant : tensor->gradFn->parent_tensors) {
            std::visit([&](auto parent) {
                if (parent != nullptr && parent->requiresGrad) {
                    buildTopo(parent, topoOrder, visited);
                }
            }, parent_variant);
        }
    }
    topoOrder.push_back(tensor);
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
    // Initialize this tensor's gradient
    if (this->grad == nullptr) {
        this->grad = new NDArray<dtype>(this->getShape());
    }
    *(this->grad) = *grad;

    // Build computational graph using topological sort
    auto topoOrder = std::vector<Tensor<dtype>*>();
    auto visited = std::unordered_set<Tensor<dtype>*>();
    buildTopo(this, topoOrder, visited);

    // Build set of nodes to preserve (the most recent N in topo order)
    std::unordered_set<Tensor<dtype>*> nodesToPreserve;
    size_t toPreserve = std::min(static_cast<size_t>(preserveAncestors), topoOrder.size());
    for (size_t i = topoOrder.size() - toPreserve; i < topoOrder.size(); i++) {
        nodesToPreserve.insert(topoOrder[i]);
    }

    // Track functions to cleanup after backward pass completes
    std::vector<Function*> functionsToCleanup;


    for (auto it = topoOrder.rbegin(); it != topoOrder.rend(); ++it) {
        Tensor<dtype> *tensor = *it;
        if (tensor->gradFn == nullptr) continue;

        auto gradOutput = tensor->grad;
        if (gradOutput == nullptr) continue;

        // REAL FIX: Pass current parent data directly to backward, no caching needed
        std::vector<arr::NDArrayPtrVariant> current_parent_data;
        for (const auto &parent_variant : tensor->gradFn->parent_tensors) {
            std::visit([&](auto parent) {
                if (parent && parent->getDataPtr()) {
                    current_parent_data.push_back(parent->getDataPtr());
                } else {
                    // Parent was deleted, use null - backward must handle this gracefully
                    current_parent_data.push_back(static_cast<NDArray<float>*>(nullptr));
                }
            }, parent_variant);
        }

        auto parentGrads = tensor->gradFn->backward(gradOutput, current_parent_data);

        for (size_t i = 0; i < parentGrads.size() && i < tensor->gradFn->parent_tensors.size(); i++) {
            std::visit([&](auto parentTensor) {
                std::visit([&](auto parentGrad) {
                    using parent_dtype = typename std::decay_t<decltype(*parentTensor)>::value_type;
                    using grad_dtype = typename std::decay_t<decltype(*parentGrad)>::value_type;

                    if constexpr (std::is_same_v<parent_dtype, grad_dtype>) {
                        if (parentTensor->requiresGrad && parentGrad != nullptr) {
                            if (parentTensor->grad == nullptr) {
                                parentTensor->grad = new NDArray<parent_dtype>(*parentGrad);
                                delete parentGrad;
                            } else {
                                parentTensor->grad->executeElementWise(AffineAddOp<parent_dtype>{1, 1},
                                    parentGrad, parentTensor->grad);
                                delete parentGrad;
                            }
                        } else if (parentGrad != nullptr) {
                            delete parentGrad;
                        }
                    }
                }, parentGrads[i]);
            }, tensor->gradFn->parent_tensors[i]);
        }
        // Cleanup: mark functions for removal if not retaining graph
        if (!retainGraph && tensor->gradFn != nullptr) {
            bool isPreserved = nodesToPreserve.count(tensor) > 0;
            if (!isPreserved) {
                // Schedule function for cleanup after loop completes
                functionsToCleanup.push_back(tensor->gradFn);
                tensor->gradFn = nullptr;
            }
        }
    }

    // Cleanup phase: unregister functions from the global registry
    if (!retainGraph) {
        for (Function* fn : functionsToCleanup) {
            Function::unregister_function(fn);
        }
    }

    if (deleteGrad) delete grad;
}


template <typename dtype>
template <typename newDtype>
Tensor<newDtype> Tensor<dtype>::cast() const {
    auto new_data = new NDArray<newDtype>(data->template cast<newDtype>());
    auto t = Tensor<newDtype>(new_data, requiresGrad);

    // FIXED: Exception-safe cast implementation
    if (requiresGrad && grad) {
        try {
            auto new_grad = new NDArray<newDtype>(grad->template cast<newDtype>());
            delete t.grad;  // Safe because we just created t
            t.grad = new_grad;
        } catch (...) {
            // If grad cast fails, cleanup and rethrow
            delete new_data;
            throw;
        }
    }
    return t;
}



namespace tensor {
    using TensorVariant = std::variant<
        NDArray<int32_t>,
        NDArray<int64_t>,
        NDArray<size_t>,
        NDArray<float>,
        NDArray<double>,
        NDArray<__half>,
        NDArray<__nv_bfloat16>,
        NDArray<bool>
    >;
    using TensorPtrVariant = std::variant<
        Tensor<int>*,
        Tensor<int32_t>*,
        Tensor<int64_t>*,
        Tensor<size_t>*,
        Tensor<float>*,
        Tensor<double>*,
        Tensor<__half>*,
        Tensor<__nv_bfloat16>*,
        Tensor<bool>*
    >;
}

#endif //ARRC_TENSOR_H
