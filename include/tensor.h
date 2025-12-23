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
#include <memory>

// Forward declarations for friend access
template <typename dtype>
void buildTopo(Tensor<dtype> *tensor, std::vector<Tensor<dtype>*> &topoOrder,
               std::unordered_set<Tensor<dtype>*> &visited);
class Function;

template <typename dtype>
class Tensor{
private:
    static size_t idGenerator;
    size_t id;
    std::unique_ptr<NDArray<dtype>> data;
    std::unique_ptr<NDArray<dtype>> grad;
    std::shared_ptr<Function> gradFn = nullptr;
    bool requiresGrad;

    // Friend declarations for autograd system
    friend void buildTopo<dtype>(Tensor<dtype>*, std::vector<Tensor<dtype>*>&,
                                  std::unordered_set<Tensor<dtype>*>&);
    friend class Function;
public:
    using value_type = dtype;
    // Constructor taking unique_ptr (preferred)
    Tensor(std::unique_ptr<NDArray<dtype>> data, const bool &requiresGrad = false, std::shared_ptr<Function> gradFn = nullptr):
    data(std::move(data)), requiresGrad(requiresGrad), gradFn(std::move(gradFn)), id(idGenerator++) {
        if (requiresGrad){
            grad = std::make_unique<NDArray<dtype>>(this->data->getShape());
            *grad = (dtype)0;
        }
    }
    // Legacy constructor taking raw pointer (wraps in unique_ptr)
    Tensor(NDArray<dtype> *data, const bool &requiresGrad = false, std::shared_ptr<Function> gradFn = nullptr):
    Tensor(std::unique_ptr<NDArray<dtype>>(data), requiresGrad, std::move(gradFn)) {}
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor() = default;  // shared_ptr handles gradFn cleanup automatically
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
    // For internal use by cast() - replaces the gradient buffer
    void replaceGrad(std::unique_ptr<NDArray<dtype>> newGrad) {
        grad = std::move(newGrad);
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
      data(std::move(other.data)),
      grad(std::move(other.grad)),
      gradFn(std::move(other.gradFn)),
      requiresGrad(other.requiresGrad) {
    other.requiresGrad = false;
}

template <typename dtype>
Tensor<dtype>& Tensor<dtype>::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        // Transfer ownership - shared_ptr handles cleanup automatically
        data = std::move(other.data);
        grad = std::move(other.grad);
        gradFn = std::move(other.gradFn);
        requiresGrad = other.requiresGrad;
        id = other.id;
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
            std::visit([&](auto weak_parent) {
                // Lock the weak_ptr to get a shared_ptr
                if (auto parent = weak_parent.lock()) {
                    if (parent->requiresGrad) {
                        buildTopo(parent.get(), topoOrder, visited);
                    }
                }
                // If lock() fails, parent was deleted - skip it
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
    if (!this->grad) {
        this->grad = std::make_unique<NDArray<dtype>>(this->getShape());
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
    // With shared_ptr, we just need to reset the tensor's gradFn to release ownership


    for (auto it = topoOrder.rbegin(); it != topoOrder.rend(); ++it) {
        Tensor<dtype> *tensor = *it;
        if (tensor->gradFn == nullptr) continue;

        auto* gradOutput = tensor->grad.get();
        if (gradOutput == nullptr) continue;

        // Lock weak_ptrs to get parent data safely
        std::vector<arr::NDArrayPtrVariant> current_parent_data;
        for (const auto &parent_variant : tensor->gradFn->parent_tensors) {
            std::visit([&](auto weak_parent) {
                using weak_type = std::decay_t<decltype(weak_parent)>;
                using shared_type = typename weak_type::element_type;
                using value_type = typename shared_type::value_type;

                // Lock the weak_ptr - if parent deleted, we get nullptr
                if (auto parent = weak_parent.lock()) {
                    current_parent_data.push_back(parent->getDataPtr());
                } else {
                    // Parent was deleted, use typed null
                    current_parent_data.push_back(static_cast<NDArray<value_type>*>(nullptr));
                }
            }, parent_variant);
        }

        auto parentGrads = tensor->gradFn->backward(gradOutput, current_parent_data);

        for (size_t i = 0; i < parentGrads.size() && i < tensor->gradFn->parent_tensors.size(); i++) {
            std::visit([&](auto weak_parent) {
                if (auto parentTensor = weak_parent.lock()) {
                    std::visit([&](auto& parentGradPtr) {  // reference to unique_ptr
                        using parent_dtype = typename std::decay_t<decltype(*parentTensor)>::value_type;
                        using grad_dtype = typename std::decay_t<decltype(*parentGradPtr)>::value_type;

                        if constexpr (std::is_same_v<parent_dtype, grad_dtype>) {
                            if (parentTensor->requiresGrad && parentGradPtr) {
                                if (!parentTensor->grad) {
                                    parentTensor->grad = std::move(parentGradPtr);
                                } else {
                                    parentTensor->grad->executeElementWise(AffineAddOp<parent_dtype>{1, 1},
                                        parentGradPtr.get(), parentTensor->grad.get());
                                }
                            }
                        }
                    }, parentGrads[i]);
                }
                // If parent deleted, unique_ptr auto-cleans up - no manual delete needed
            }, tensor->gradFn->parent_tensors[i]);
        }
        // Cleanup: release function ownership if not retaining graph
        if (!retainGraph && tensor->gradFn != nullptr) {
            bool isPreserved = nodesToPreserve.count(tensor) > 0;
            if (!isPreserved) {
                // Release ownership - shared_ptr handles deletion automatically
                tensor->gradFn.reset();
            }
        }
    }

    // No separate cleanup phase needed - shared_ptr handles deletion when reset()

    if (deleteGrad) delete grad;
}


template <typename dtype>
template <typename newDtype>
Tensor<newDtype> Tensor<dtype>::cast() const {
    // Allocate new data with casted values (unique_ptr for exception safety)
    auto new_data = std::make_unique<NDArray<newDtype>>(data->template cast<newDtype>());

    std::unique_ptr<NDArray<newDtype>> new_grad;
    if (requiresGrad && grad) {
        new_grad = std::make_unique<NDArray<newDtype>>(grad->template cast<newDtype>());
    }

    // Create tensor - it will allocate its own grad if requiresGrad
    auto t = Tensor<newDtype>(std::move(new_data), requiresGrad);

    // Replace the auto-allocated grad with our casted one
    if (new_grad) {
        t.replaceGrad(std::move(new_grad));
    }
    return t;
}



namespace tensor {
    // TensorSharedVariant for passing tensor shared_ptr (inputs to functions)
    using TensorSharedVariant = std::variant<
        std::shared_ptr<Tensor<int>>,
        std::shared_ptr<Tensor<int32_t>>,
        std::shared_ptr<Tensor<int64_t>>,
        std::shared_ptr<Tensor<size_t>>,
        std::shared_ptr<Tensor<float>>,
        std::shared_ptr<Tensor<double>>,
        std::shared_ptr<Tensor<__half>>,
        std::shared_ptr<Tensor<__nv_bfloat16>>,
        std::shared_ptr<Tensor<bool>>
    >;

    // TensorWeakVariant for storing weak references (in Function::parent_tensors)
    using TensorWeakVariant = std::variant<
        std::weak_ptr<Tensor<int>>,
        std::weak_ptr<Tensor<int32_t>>,
        std::weak_ptr<Tensor<int64_t>>,
        std::weak_ptr<Tensor<size_t>>,
        std::weak_ptr<Tensor<float>>,
        std::weak_ptr<Tensor<double>>,
        std::weak_ptr<Tensor<__half>>,
        std::weak_ptr<Tensor<__nv_bfloat16>>,
        std::weak_ptr<Tensor<bool>>
    >;

    // Legacy TensorPtrVariant (kept for backward compatibility during transition)
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

    // Helper function to create tensors with shared_ptr
    template<typename dtype>
    std::shared_ptr<Tensor<dtype>> make_tensor(const std::vector<int>& shape, bool requiresGrad = false) {
        return std::make_shared<Tensor<dtype>>(
            std::make_unique<NDArray<dtype>>(shape), requiresGrad);
    }

    // Helper to create tensor from existing NDArray
    template<typename dtype>
    std::shared_ptr<Tensor<dtype>> make_tensor(std::unique_ptr<NDArray<dtype>> data, bool requiresGrad = false) {
        return std::make_shared<Tensor<dtype>>(std::move(data), requiresGrad);
    }
}

#endif //ARRC_TENSOR_H
