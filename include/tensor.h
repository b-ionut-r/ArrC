//
// Created by Bujor Ionut Raul on 22.12.2025.
//

#ifndef ARRC_TENSOR_H
#define ARRC_TENSOR_H

#include "ndarray.cuh"
#include <string>
#include <unordered_set>
#include <algorithm>
#include <memory>
#include <variant>

// Forward declarations
template <typename dtype> class TensorPtr;
template <typename dtype>
void buildTopo(TensorPtr<dtype> *tensor, std::vector<TensorPtr<dtype>*> &topoOrder,
               std::unordered_set<TensorPtr<dtype>*> &visited);
class Function;

template <typename dtype>
class TensorPtr{
private:
    static size_t idGenerator;
    size_t id;
    std::unique_ptr<NDArray<dtype>> data;
    std::unique_ptr<NDArray<dtype>> grad;
    std::shared_ptr<Function> gradFn = nullptr;
    bool requiresGrad;

    // Friend declarations for autograd system
    friend void buildTopo<dtype>(TensorPtr<dtype>*, std::vector<TensorPtr<dtype>*>&,
                                  std::unordered_set<TensorPtr<dtype>*>&);
    friend class Function;
public:
    using value_type = dtype;
    // Constructor taking unique_ptr (preferred)
    TensorPtr(std::unique_ptr<NDArray<dtype>> data, const bool &requiresGrad = false, std::shared_ptr<Function> gradFn = nullptr):
    data(std::move(data)), requiresGrad(requiresGrad), gradFn(std::move(gradFn)), id(idGenerator++) {
        if (requiresGrad){
            grad = std::make_unique<NDArray<dtype>>(this->data->getShape());
            *grad = (dtype)0;
        }
    }
    TensorPtr(const std::vector<int> &shape, const bool &requiresGrad = false):
    TensorPtr(std::make_unique<NDArray<dtype>>(shape), requiresGrad) {}
    
    TensorPtr(const TensorPtr&) = delete;
    TensorPtr& operator=(const TensorPtr&) = delete;

    TensorPtr(TensorPtr&& other) noexcept
        : id(other.id),
          data(std::move(other.data)),
          grad(std::move(other.grad)),
          gradFn(std::move(other.gradFn)),
          requiresGrad(other.requiresGrad) {
        other.requiresGrad = false;
    }

    TensorPtr& operator=(TensorPtr&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            grad = std::move(other.grad);
            gradFn = std::move(other.gradFn);
            requiresGrad = other.requiresGrad;
            id = other.id;
            other.requiresGrad = false;
        }
        return *this;
    }

    ~TensorPtr() = default;
    
    NDArray<dtype>* getDataPtr() const {return data.get();}
    NDArray<dtype>* getGradPtr() const {return grad.get();}
    NDArray<dtype> getData() const {return *data;}
    const NDArray<dtype>* getGrad() const {return requiresGrad ? grad.get() : nullptr;}
    bool getRequiresGrad() const { return requiresGrad; }
    std::string getName() const {return "UnnamedTensor_" + std::to_string(id);}
    int getSize() const {return data->getSize();}
    vector<int> getShape() const {return data->getShape();}
    
    void zeroGrad() {
        if (requiresGrad && grad) {
            *grad = (dtype)0;
        }
    }
    
    void replaceGrad(std::unique_ptr<NDArray<dtype>> newGrad) {
        grad = std::move(newGrad);
    }
    
    void backward(NDArray<dtype> *grad = nullptr,
                  const bool retainGraph = false,
                  const int preserveAncestors = 4);
                  
    template <typename newDtype>
    TensorPtr<newDtype> cast() const;

private:
    void build_topological_sort(TensorPtr<dtype> *tensor, std::vector<TensorPtr<dtype>*> &topoOrder,
                                std::unordered_set<TensorPtr<dtype>*> &visited) {
        if (tensor == nullptr || visited.count(tensor)) return;
        visited.insert(tensor);
        
        if (tensor->gradFn != nullptr) {
            for (const auto &parent_variant : tensor->gradFn->parent_tensors) {
                std::visit([&](auto weak_parent) {
                    using parent_tensor_t = typename std::decay_t<decltype(weak_parent)>::element_type;
                    using parent_dtype = typename parent_tensor_t::value_type;

                    if constexpr (std::is_same_v<parent_dtype, dtype>) {
                        if (auto parent = weak_parent.lock()) {
                            if (parent->getRequiresGrad()) {
                                build_topological_sort(parent.get(), topoOrder, visited);
                            }
                        }
                    }
                }, parent_variant);
            }
        }
        topoOrder.push_back(tensor);
    }
};

template <typename dtype>
size_t TensorPtr<dtype>::idGenerator = 0;

template <typename dtype>
void TensorPtr<dtype>::backward(NDArray<dtype> *grad,
                             const bool retainGraph,
                             const int preserveAncestors) {
    if (!requiresGrad)
        throw std::runtime_error("Cannot do backprop for a tensor that does not require grad.");
        
    bool deleteGrad = false;
    if (grad == nullptr) {
        if (getSize() != 1) {
            throw std::runtime_error("backward() can only be called for scalar outputs. "
                    "For non-scalar outputs, gradient must be provided.");
        }
        grad = new NDArray<dtype>(arr::make_ones<dtype>(getShape()));
        deleteGrad = true;
    }

    if (!this->grad) {
        this->grad = std::make_unique<NDArray<dtype>>(this->getShape());
    }
    *(this->grad) = *grad;

    std::vector<TensorPtr<dtype>*> topoOrder;
    std::unordered_set<TensorPtr<dtype>*> visited;
    
    // Start topological sort from this node
    build_topological_sort(this, topoOrder, visited);

    std::unordered_set<TensorPtr<dtype>*> nodesToPreserve;
    size_t toPreserve = std::min(static_cast<size_t>(preserveAncestors), topoOrder.size());
    for (size_t i = topoOrder.size() - toPreserve; i < topoOrder.size(); i++) {
        nodesToPreserve.insert(topoOrder[i]);
    }

    for (auto it = topoOrder.rbegin(); it != topoOrder.rend(); ++it) {
        TensorPtr<dtype> *tensor = *it;
        if (!tensor->gradFn) continue;
        
        auto* gradOutput = tensor->grad.get();
        if (!gradOutput) continue;

        std::vector<arr::NDArrayPtrVariant> current_parent_data;
        // Collect parent data safely
        for (const auto &parent_variant : tensor->gradFn->parent_tensors) {
            std::visit([&](auto weak_parent) {
                using shared_type = typename std::decay_t<decltype(weak_parent)>::element_type;
                if (auto parent = weak_parent.lock()) {
                    current_parent_data.push_back(parent->getDataPtr());
                } else {
                    current_parent_data.push_back(static_cast<NDArray<typename shared_type::value_type>*>(nullptr));
                }
            }, parent_variant);
        }

        auto parentGrads = tensor->gradFn->backward(gradOutput, current_parent_data);

        for (size_t i = 0; i < parentGrads.size() && i < tensor->gradFn->parent_tensors.size(); i++) {
            std::visit([&](auto weak_parent) {
                if (auto parentTensor = weak_parent.lock()) {
                    std::visit([&](auto& parentGradPtr) {
                        using parent_dtype = typename std::decay_t<decltype(*parentTensor)>::value_type;
                        using grad_dtype = typename std::decay_t<decltype(*parentGradPtr)>::value_type;

                        if constexpr (std::is_same_v<parent_dtype, grad_dtype>) {
                            if (parentTensor->getRequiresGrad() && parentGradPtr) {
                                auto* parentGrad = parentTensor->getGradPtr();
                                if (!parentGrad) {
                                    parentTensor->replaceGrad(std::move(parentGradPtr));
                                } else {
                                    parentGrad->executeElementWise(AffineAddOp<parent_dtype>{1, 1},
                                        parentGradPtr.get(), parentGrad);
                                }
                            }
                        }
                    }, parentGrads[i]);
                }
            }, tensor->gradFn->parent_tensors[i]);
        }
        
        if (!retainGraph && tensor->gradFn != nullptr && !nodesToPreserve.count(tensor)) {
            tensor->gradFn.reset();
        }
    }
    if (deleteGrad) delete grad;
}


template <typename dtype>
template <typename newDtype>
TensorPtr<newDtype> TensorPtr<dtype>::cast() const {
    // Allocate new data with casted values (unique_ptr for exception safety)
    auto new_data = std::make_unique<NDArray<newDtype>>(data->template cast<newDtype>());

    std::unique_ptr<NDArray<newDtype>> new_grad;
    if (requiresGrad && grad) {
        new_grad = std::make_unique<NDArray<newDtype>>(grad->template cast<newDtype>());
    }

    // Create tensor - it will allocate its own grad if requiresGrad
    auto t = TensorPtr<newDtype>(std::move(new_data), requiresGrad);

    // Replace the auto-allocated grad with our casted one
    if (new_grad) {
        t.replaceGrad(std::move(new_grad));
    }
    return t;
}
template <typename dtype>
class Tensor {
private:
    std::shared_ptr<TensorPtr<dtype>> impl;
public:
    using value_type = dtype;
    Tensor() = default;
    Tensor(std::shared_ptr<TensorPtr<dtype>> impl): impl(std::move(impl)) {}
    Tensor(const std::vector<int> &shape, const bool &requiresGrad = false):
        impl(std::make_shared<TensorPtr<dtype>>(std::make_unique<NDArray<dtype>>(shape), requiresGrad)) {}
    Tensor(std::unique_ptr<NDArray<dtype>> data, const bool &requiresGrad = false):
        impl(std::make_shared<TensorPtr<dtype>>(std::move(data), requiresGrad)) {}

    bool defined() const { return static_cast<bool>(impl); }
    explicit operator bool() const { return static_cast<bool>(impl); }
    const std::shared_ptr<TensorPtr<dtype>>& shared() const { return impl; }
    TensorPtr<dtype>* get() const { return impl.get(); }
    TensorPtr<dtype>* operator->() const { return impl.get(); }

    NDArray<dtype>* getDataPtr() const { return impl->getDataPtr(); }
    NDArray<dtype>* getGradPtr() const { return impl->getGradPtr(); }
    NDArray<dtype> getData() const { return impl->getData(); }
    const NDArray<dtype>* getGrad() const { return impl->getGrad(); }
    bool getRequiresGrad() const { return impl->getRequiresGrad(); }
    std::string getName() const { return impl->getName(); }
    int getSize() const { return impl->getSize(); }
    vector<int> getShape() const { return impl->getShape(); }
    void zeroGrad() { if (impl) impl->zeroGrad(); }
    void backward(NDArray<dtype> *grad = nullptr,
                  const bool retainGraph = false,
                  const int preserveAncestors = 4) {
        impl->backward(grad, retainGraph, preserveAncestors);
    }
    NDArray<dtype>& data() { return *impl->getDataPtr(); }
    const NDArray<dtype>& data() const { return *impl->getDataPtr(); }
    NDArray<dtype>* grad() const { return impl ? impl->getGradPtr() : nullptr; }

    template <typename newDtype>
    Tensor<newDtype> cast() const {
        auto casted = impl->template cast<newDtype>();
        return Tensor<newDtype>(std::make_shared<TensorPtr<newDtype>>(std::move(casted)));
    }
};

namespace tensor {
    // TensorSharedVariant for passing tensor shared_ptr (inputs to functions)
    using TensorSharedVariant = std::variant<
        std::shared_ptr<TensorPtr<int32_t>>,
        std::shared_ptr<TensorPtr<int64_t>>,
        std::shared_ptr<TensorPtr<size_t>>,
        std::shared_ptr<TensorPtr<float>>,
        std::shared_ptr<TensorPtr<double>>,
        std::shared_ptr<TensorPtr<__half>>,
        std::shared_ptr<TensorPtr<__nv_bfloat16>>,
        std::shared_ptr<TensorPtr<bool>>
    >;

    // TensorWeakVariant for storing weak references (in Function::parent_tensors)
    using TensorWeakVariant = std::variant<
        std::weak_ptr<TensorPtr<int32_t>>,
        std::weak_ptr<TensorPtr<int64_t>>,
        std::weak_ptr<TensorPtr<size_t>>,
        std::weak_ptr<TensorPtr<float>>,
        std::weak_ptr<TensorPtr<double>>,
        std::weak_ptr<TensorPtr<__half>>,
        std::weak_ptr<TensorPtr<__nv_bfloat16>>,
        std::weak_ptr<TensorPtr<bool>>
    >;

    // Helper function to create tensors with shared_ptr
    template<typename dtype>
    std::shared_ptr<TensorPtr<dtype>> make_tensor(const std::vector<int>& shape, bool requiresGrad = false) {
        return std::make_shared<TensorPtr<dtype>>(
            std::make_unique<NDArray<dtype>>(shape), requiresGrad);
    }

    // Helper to create tensor from existing NDArray
    template<typename dtype>
    std::shared_ptr<TensorPtr<dtype>> make_tensor(std::unique_ptr<NDArray<dtype>> data, bool requiresGrad = false) {
        return std::make_shared<TensorPtr<dtype>>(std::move(data), requiresGrad);
    }

    template<typename dtype>
    ::Tensor<dtype> zeros(const std::vector<int>& shape, bool requiresGrad = false) {
        auto t = make_tensor<dtype>(shape, requiresGrad);
        *t->getDataPtr() = static_cast<dtype>(0);
        return ::Tensor<dtype>(std::move(t));
    }

    template<typename dtype>
    ::Tensor<dtype> ones(const std::vector<int>& shape, bool requiresGrad = false) {
        auto t = make_tensor<dtype>(shape, requiresGrad);
        *t->getDataPtr() = static_cast<dtype>(1);
        return ::Tensor<dtype>(std::move(t));
    }
}

#endif //ARRC_TENSOR_H
