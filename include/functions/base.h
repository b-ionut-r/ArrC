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
        const arr::NDArrayPtrVariant& grad_output,
        const std::vector<arr::NDArrayPtrVariant>& parent_data) const = 0;
};

template<typename T>
std::shared_ptr<T> Function::operator()(const std::vector<tensor::TensorSharedVariant>& inputs,
                                        std::shared_ptr<Function> self) {
    parent_tensors.clear();
    parent_tensors.reserve(inputs.size());

    std::vector<arr::NDArrayPtrVariant> parents_data;
    parents_data.reserve(inputs.size());
    bool reqGrad = false;

    for (const auto& parent : inputs) {
        std::visit([&](auto parent_shared) {
            parent_tensors.push_back(std::weak_ptr(parent_shared));
            if (parent_shared->getRequiresGrad())
                reqGrad = true;
            parents_data.push_back(parent_shared->getDataPtr());
        }, parent);
    }

    auto output = forward(parents_data);
    return std::visit([&](auto output_ptr) -> std::shared_ptr<T> {
        using dtype = typename std::decay_t<decltype(*output_ptr)>::value_type;
        if constexpr (std::is_same_v<T, TensorPtr<dtype>>) {
            auto output_unique = std::unique_ptr<NDArray<dtype>>(output_ptr);
            return std::make_shared<T>(std::move(output_unique), reqGrad, reqGrad ? self : nullptr);
        } else {
            delete output_ptr;
            return nullptr;
        }
    }, output);
}

#endif //ARRC_BASE_H
