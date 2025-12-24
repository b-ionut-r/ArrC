//
// Created by Bujor Ionut Raul on 22.12.2025.
//

#ifndef ARRC_TENSOR_H
#define ARRC_TENSOR_H

#include "ndarray.cuh"
#include <unordered_set>
#include <memory>
#include <variant>

template <typename dtype> class TensorPtr;
class Function;

template <typename dtype>
class TensorPtr {
    std::unique_ptr<NDArray<dtype>> data_;
    std::unique_ptr<NDArray<dtype>> grad_;
    std::shared_ptr<Function> gradFn_;
    bool requiresGrad_;

    friend class Function;
public:
    using value_type = dtype;

    TensorPtr(std::unique_ptr<NDArray<dtype>> data, bool requiresGrad = false,
              std::shared_ptr<Function> gradFn = nullptr)
        : data_(std::move(data)), requiresGrad_(requiresGrad), gradFn_(std::move(gradFn)) {
        if (requiresGrad_) {
            grad_ = std::make_unique<NDArray<dtype>>(data_->getShape());
            *grad_ = static_cast<dtype>(0);
        }
    }

    TensorPtr(const std::vector<int>& shape, bool requiresGrad = false)
        : TensorPtr(std::make_unique<NDArray<dtype>>(shape), requiresGrad) {}

    TensorPtr(const TensorPtr&) = delete;
    TensorPtr& operator=(const TensorPtr&) = delete;
    TensorPtr(TensorPtr&&) noexcept = default;
    TensorPtr& operator=(TensorPtr&&) noexcept = default;
    ~TensorPtr() = default;

    NDArray<dtype>* data() const { return data_.get(); }
    NDArray<dtype>* grad() const { return grad_.get(); }
    bool requiresGrad() const { return requiresGrad_; }
    int size() const { return data_->getSize(); }
    std::vector<int> shape() const { return data_->getShape(); }

    void zeroGrad() {
        if (requiresGrad_ && grad_) *grad_ = static_cast<dtype>(0);
    }

    void replaceGrad(std::unique_ptr<NDArray<dtype>> newGrad) {
        grad_ = std::move(newGrad);
    }

    void backward(NDArray<dtype>* grad = nullptr, bool retainGraph = false, int preserveAncestors = 4);

    template <typename newDtype>
    TensorPtr<newDtype> cast() const {
        auto newData = std::make_unique<NDArray<newDtype>>(data_->template cast<newDtype>());
        auto result = TensorPtr<newDtype>(std::move(newData), requiresGrad_);
        if (requiresGrad_ && grad_)
            result.replaceGrad(std::make_unique<NDArray<newDtype>>(grad_->template cast<newDtype>()));
        return result;
    }

private:
    void buildTopoSort(TensorPtr<dtype>* tensor, std::vector<TensorPtr<dtype>*>& order,
                       std::unordered_set<TensorPtr<dtype>*>& visited);
};

template <typename dtype>
void TensorPtr<dtype>::buildTopoSort(TensorPtr<dtype>* tensor, std::vector<TensorPtr<dtype>*>& order,
                                     std::unordered_set<TensorPtr<dtype>*>& visited) {
    if (!tensor || visited.count(tensor)) return;
    visited.insert(tensor);

    if (tensor->gradFn_) {
        for (const auto& parentVar : tensor->gradFn_->parent_tensors) {
            std::visit([&](auto weakParent) {
                using ParentT = typename std::decay_t<decltype(weakParent)>::element_type;
                if constexpr (std::is_same_v<typename ParentT::value_type, dtype>) {
                    if (auto parent = weakParent.lock(); parent && parent->requiresGrad())
                        buildTopoSort(parent.get(), order, visited);
                }
            }, parentVar);
        }
    }
    order.push_back(tensor);
}

template <typename dtype>
void TensorPtr<dtype>::backward(NDArray<dtype>* grad, bool retainGraph, int preserveAncestors) {
    if (!requiresGrad_)
        throw std::runtime_error("Cannot backprop on tensor that doesn't require grad.");

    std::unique_ptr<NDArray<dtype>> ownedGrad;
    if (!grad) {
        if (size() != 1)
            throw std::runtime_error("backward() requires gradient for non-scalar outputs.");
        ownedGrad = std::make_unique<NDArray<dtype>>(arr::make_ones<dtype>(shape()));
        grad = ownedGrad.get();
    }

    if (!grad_) grad_ = std::make_unique<NDArray<dtype>>(shape());
    *grad_ = *grad;

    std::vector<TensorPtr<dtype>*> topoOrder;
    std::unordered_set<TensorPtr<dtype>*> visited;
    buildTopoSort(this, topoOrder, visited);

    std::unordered_set<TensorPtr<dtype>*> preserve;
    size_t numPreserve = std::min(static_cast<size_t>(preserveAncestors), topoOrder.size());
    for (size_t i = topoOrder.size() - numPreserve; i < topoOrder.size(); i++)
        preserve.insert(topoOrder[i]);

    for (auto it = topoOrder.rbegin(); it != topoOrder.rend(); ++it) {
        TensorPtr<dtype>* tensor = *it;
        if (!tensor->gradFn_ || !tensor->grad_) continue;

        std::vector<arr::NDArrayPtrVariant> parentData;
        for (const auto& parentVar : tensor->gradFn_->parent_tensors) {
            std::visit([&](auto weakParent) {
                using SharedT = typename std::decay_t<decltype(weakParent)>::element_type;
                if (auto parent = weakParent.lock())
                    parentData.push_back(parent->data());
                else
                    parentData.push_back(static_cast<NDArray<typename SharedT::value_type>*>(nullptr));
            }, parentVar);
        }

        auto parentGrads = tensor->gradFn_->backward(tensor->grad_.get(), parentData);

        for (size_t i = 0; i < parentGrads.size() && i < tensor->gradFn_->parent_tensors.size(); i++) {
            std::visit([&](auto weakParent) {
                if (auto parentTensor = weakParent.lock()) {
                    std::visit([&](auto& gradPtr) {
                        using ParentDtype = typename std::decay_t<decltype(*parentTensor)>::value_type;
                        using GradDtype = typename std::decay_t<decltype(*gradPtr)>::value_type;
                        if constexpr (std::is_same_v<ParentDtype, GradDtype>) {
                            if (parentTensor->requiresGrad() && gradPtr) {
                                auto* pGrad = parentTensor->grad();
                                if (!pGrad)
                                    parentTensor->replaceGrad(std::move(gradPtr));
                                else
                                    pGrad->executeElementWise(AffineAddOp<ParentDtype>{1, 1}, gradPtr.get(), pGrad);
                            }
                        }
                    }, parentGrads[i]);
                }
            }, tensor->gradFn_->parent_tensors[i]);
        }

        if (!retainGraph && tensor->gradFn_ && !preserve.count(tensor))
            tensor->gradFn_.reset();
    }
}

template <typename dtype>
class Tensor {
    std::shared_ptr<TensorPtr<dtype>> impl_;
public:
    using value_type = dtype;

    Tensor() = default;
    Tensor(std::shared_ptr<TensorPtr<dtype>> impl) : impl_(std::move(impl)) {}
    Tensor(const std::vector<int>& shape, bool requiresGrad = false)
        : impl_(std::make_shared<TensorPtr<dtype>>(std::make_unique<NDArray<dtype>>(shape), requiresGrad)) {}
    Tensor(std::unique_ptr<NDArray<dtype>> data, bool requiresGrad = false)
        : impl_(std::make_shared<TensorPtr<dtype>>(std::move(data), requiresGrad)) {}

    explicit operator bool() const { return static_cast<bool>(impl_); }
    const std::shared_ptr<TensorPtr<dtype>>& shared() const { return impl_; }
    TensorPtr<dtype>* get() const { return impl_.get(); }
    TensorPtr<dtype>* operator->() const { return impl_.get(); }

    NDArray<dtype>& data() { return *impl_->data(); }
    const NDArray<dtype>& data() const { return *impl_->data(); }
    NDArray<dtype>* grad() const { return impl_ ? impl_->grad() : nullptr; }
    bool requiresGrad() const { return impl_->requiresGrad(); }
    int size() const { return impl_->size(); }
    std::vector<int> shape() const { return impl_->shape(); }

    void zeroGrad() { if (impl_) impl_->zeroGrad(); }
    void backward(NDArray<dtype>* grad = nullptr, bool retainGraph = false, int preserveAncestors = 4) {
        impl_->backward(grad, retainGraph, preserveAncestors);
    }

    template <typename newDtype>
    Tensor<newDtype> cast() const {
        return Tensor<newDtype>(std::make_shared<TensorPtr<newDtype>>(std::move(impl_->template cast<newDtype>())));
    }
};

namespace tensor {
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

    template<typename dtype>
    Tensor<dtype> zeros(const std::vector<int>& shape, bool requiresGrad = false) {
        auto impl = std::make_shared<TensorPtr<dtype>>(std::make_unique<NDArray<dtype>>(shape), requiresGrad);
        *impl->data() = static_cast<dtype>(0);
        return Tensor<dtype>(std::move(impl));
    }

    template<typename dtype>
    Tensor<dtype> ones(const std::vector<int>& shape, bool requiresGrad = false) {
        auto impl = std::make_shared<TensorPtr<dtype>>(std::make_unique<NDArray<dtype>>(shape), requiresGrad);
        *impl->data() = static_cast<dtype>(1);
        return Tensor<dtype>(std::move(impl));
    }
}

#endif //ARRC_TENSOR_H
