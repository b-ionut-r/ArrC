//
// Created by Bujor Ionut Raul on 23.12.2025.
//

#ifndef ARRC_BASE_H
#define ARRC_BASE_H

#include <variant>
#include <vector>
#include <memory>
#include "../ndarray.cuh"
#include "../tensor.h"

class Function {
public:
    std::vector<tensor::TensorWeakVariant> parent_tensors;

    Function() = default;
    virtual ~Function() = default;

    template<typename T>
    std::shared_ptr<T> operator()(const std::vector<tensor::TensorSharedVariant>& inputs,
                                  std::shared_ptr<Function> self);

    virtual arr::NDArrayPtrVariant forward(const std::vector<arr::NDArrayPtrVariant>& inputs) const = 0;
    virtual std::vector<arr::NDArrayUniquePtrVariant> backward(
        const arr::NDArrayPtrVariant& gradOut,
        const std::vector<arr::NDArrayPtrVariant>& parentData) const = 0;
};

template<typename T>
std::shared_ptr<T> Function::operator()(const std::vector<tensor::TensorSharedVariant>& inputs,
                                        std::shared_ptr<Function> self) {
    parent_tensors.clear();
    parent_tensors.reserve(inputs.size());

    std::vector<arr::NDArrayPtrVariant> parentData;
    parentData.reserve(inputs.size());
    bool reqGrad = false;

    for (const auto& input : inputs) {
        std::visit([&](auto shared) {
            parent_tensors.push_back(std::weak_ptr(shared));
            if (shared->requiresGrad()) reqGrad = true;
            parentData.push_back(shared->data());
        }, input);
    }

    return std::visit([&](auto outPtr) -> std::shared_ptr<T> {
        using dtype = typename std::decay_t<decltype(*outPtr)>::value_type;
        if constexpr (std::is_same_v<T, TensorPtr<dtype>>) {
            return std::make_shared<T>(std::unique_ptr<NDArray<dtype>>(outPtr), reqGrad, reqGrad ? self : nullptr);
        } else {
            delete outPtr;
            return nullptr;
        }
    }, forward(parentData));
}

#endif //ARRC_BASE_H
