//
// Created by Bujor Ionut Raul on 23.12.2025.
//

#ifndef ARRC_ARITHMETIC_H
#define ARRC_ARITHMETIC_H

#include "base.h"
#include "../ndarray.cuh"
#include <memory>

class AddFunction : public Function {
public:
    std::string getName() const override { return "AddFunction"; }

    arr::NDArrayPtrVariant forward(const std::vector<arr::NDArrayPtrVariant> &inputs) const override {
        if (inputs.size() != 2)
            throw std::runtime_error("AddFunction requires exactly 2 inputs");

        return std::visit([](auto a, auto b) -> arr::NDArrayPtrVariant {
            using dtype = typename std::decay_t<decltype(*a)>::value_type;
            return new NDArray<dtype>(*a + *b);
        }, inputs[0], inputs[1]);
    }

    std::vector<arr::NDArrayUniquePtrVariant> backward(const arr::NDArrayPtrVariant &grad_output,
                                                       const std::vector<arr::NDArrayPtrVariant>&) const override {
        std::vector<arr::NDArrayUniquePtrVariant> grads;
        std::visit([&](auto grad) {
            using dtype = typename std::decay_t<decltype(*grad)>::value_type;
            grads.push_back(std::make_unique<NDArray<dtype>>(*grad));  // d(a+b)/da = 1
            grads.push_back(std::make_unique<NDArray<dtype>>(*grad));  // d(a+b)/db = 1
        }, grad_output);
        return grads;
    }
};

class SubFunction : public Function {
public:
    std::string getName() const override { return "SubFunction"; }

    arr::NDArrayPtrVariant forward(const std::vector<arr::NDArrayPtrVariant> &inputs) const override {
        if (inputs.size() != 2)
            throw std::runtime_error("SubFunction requires exactly 2 inputs");

        return std::visit([](auto a, auto b) -> arr::NDArrayPtrVariant {
            using dtype = typename std::decay_t<decltype(*a)>::value_type;
            return new NDArray<dtype>(*a - *b);
        }, inputs[0], inputs[1]);
    }

    std::vector<arr::NDArrayUniquePtrVariant> backward(const arr::NDArrayPtrVariant &grad_output,
                                                       const std::vector<arr::NDArrayPtrVariant>&) const override {
        std::vector<arr::NDArrayUniquePtrVariant> grads;
        std::visit([&](auto grad) {
            using dtype = typename std::decay_t<decltype(*grad)>::value_type;
            grads.push_back(std::make_unique<NDArray<dtype>>(*grad));   // d(a-b)/da = 1
            grads.push_back(std::make_unique<NDArray<dtype>>(-*grad));  // d(a-b)/db = -1
        }, grad_output);
        return grads;
    }
};

class MulFunction : public Function {
public:
    std::string getName() const override { return "MulFunction"; }

    arr::NDArrayPtrVariant forward(const std::vector<arr::NDArrayPtrVariant> &inputs) const override {
        if (inputs.size() != 2)
            throw std::runtime_error("MulFunction requires exactly 2 inputs");

        return std::visit([](auto a, auto b) -> arr::NDArrayPtrVariant {
            using dtype = typename std::decay_t<decltype(*a)>::value_type;
            return new NDArray<dtype>(*a * *b);
        }, inputs[0], inputs[1]);
    }

    std::vector<arr::NDArrayUniquePtrVariant> backward(const arr::NDArrayPtrVariant &grad_output,
                                                       const std::vector<arr::NDArrayPtrVariant> &parent_data) const override {
        if (parent_data.size() != 2)
            throw std::runtime_error("MulFunction backward requires exactly 2 parent tensors");

        return std::visit([&](auto grad) -> std::vector<arr::NDArrayUniquePtrVariant> {
            using dtype = typename std::decay_t<decltype(*grad)>::value_type;
            std::vector<arr::NDArrayUniquePtrVariant> grads;
            std::visit([&](auto a, auto b) {
                if (a && b) {
                    grads.push_back(std::make_unique<NDArray<dtype>>(*grad * *b));  // d(a*b)/da = b
                    grads.push_back(std::make_unique<NDArray<dtype>>(*grad * *a));  // d(a*b)/db = a
                } else {
                    grads.push_back(std::make_unique<NDArray<dtype>>(grad->zeros_like()));
                    grads.push_back(std::make_unique<NDArray<dtype>>(grad->zeros_like()));
                }
            }, parent_data[0], parent_data[1]);
            return grads;
        }, grad_output);
    }
};

class DivFunction : public Function {
public:
    std::string getName() const override { return "DivFunction"; }

    arr::NDArrayPtrVariant forward(const std::vector<arr::NDArrayPtrVariant> &inputs) const override {
        if (inputs.size() != 2)
            throw std::runtime_error("DivFunction requires exactly 2 inputs");

        return std::visit([](auto a, auto b) -> arr::NDArrayPtrVariant {
            using dtype = typename std::decay_t<decltype(*a)>::value_type;
            return new NDArray<dtype>(*a / *b);
        }, inputs[0], inputs[1]);
    }

    std::vector<arr::NDArrayUniquePtrVariant> backward(const arr::NDArrayPtrVariant &grad_output,
                                                       const std::vector<arr::NDArrayPtrVariant> &parent_data) const override {
        if (parent_data.size() != 2)
            throw std::runtime_error("DivFunction backward requires exactly 2 parent tensors");

        return std::visit([&](auto grad) -> std::vector<arr::NDArrayUniquePtrVariant> {
            using dtype = typename std::decay_t<decltype(*grad)>::value_type;
            std::vector<arr::NDArrayUniquePtrVariant> grads;
            std::visit([&](auto a, auto b) {
                if (a && b) {
                    // d(a/b)/da = 1/b
                    grads.push_back(std::make_unique<NDArray<dtype>>(*grad / *b));
                    // d(a/b)/db = -a/b^2
                    grads.push_back(std::make_unique<NDArray<dtype>>(-*grad * *a / (*b * *b)));
                } else {
                    grads.push_back(std::make_unique<NDArray<dtype>>(grad->zeros_like()));
                    grads.push_back(std::make_unique<NDArray<dtype>>(grad->zeros_like()));
                }
            }, parent_data[0], parent_data[1]);
            return grads;
        }, grad_output);
    }
};

// Factory functions for creating operations
namespace functions {
    inline std::shared_ptr<Function> add() { return std::make_shared<AddFunction>(); }
    inline std::shared_ptr<Function> sub() { return std::make_shared<SubFunction>(); }
    inline std::shared_ptr<Function> mul() { return std::make_shared<MulFunction>(); }
    inline std::shared_ptr<Function> div() { return std::make_shared<DivFunction>(); }
}

#endif //ARRC_ARITHMETIC_H
