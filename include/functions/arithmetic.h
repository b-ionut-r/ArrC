//
// Created by Bujor Ionut Raul on 23.12.2025.
//

#ifndef ARRC_ARITHMETIC_H
#define ARRC_ARITHMETIC_H

#include "base.h"
#include "../ndarray.cuh"
#include "../elementwise_kernels.cuh"

class AddFunction : public Function {
public:
    std::string getName() const override {
        return "AddFunction";
    }

    arr::NDArrayPtrVariant forward(const std::vector<arr::NDArrayPtrVariant> &inputs) const override {
        if (inputs.size() != 2) {
            throw std::runtime_error("AddFunction requires exactly 2 inputs");
        }

        return std::visit([&](auto a, auto b) -> arr::NDArrayPtrVariant {
            using dtype = typename std::decay_t<decltype(*a)>::value_type;
            auto result = new NDArray<dtype>(a->executeElementWise(AffineAddOp<dtype>{1, 1}, b));
            return result;
        }, inputs[0], inputs[1]);
    }

    std::vector<arr::NDArrayPtrVariant> backward(const arr::NDArrayPtrVariant &grad_output,
                                                const std::vector<arr::NDArrayPtrVariant> &parent_data) const override {
        std::vector<arr::NDArrayPtrVariant> grads;

        std::visit([&](auto grad) {
            using dtype = typename std::decay_t<decltype(*grad)>::value_type;

            auto grad_a = new NDArray<dtype>(*grad);
            auto grad_b = new NDArray<dtype>(*grad);

            grads.push_back(grad_a);
            grads.push_back(grad_b);
        }, grad_output);

        return grads;
    }
};

class MulFunction : public Function {
public:
    std::string getName() const override {
        return "MulFunction";
    }

    arr::NDArrayPtrVariant forward(const std::vector<arr::NDArrayPtrVariant> &inputs) const override {
        if (inputs.size() != 2) {
            throw std::runtime_error("MulFunction requires exactly 2 inputs");
        }

        return std::visit([&](auto a, auto b) -> arr::NDArrayPtrVariant {
            using dtype = typename std::decay_t<decltype(*a)>::value_type;
            auto result = new NDArray<dtype>(a->executeElementWise(MulOp<dtype>{}, b));
            return result;
        }, inputs[0], inputs[1]);
    }

    std::vector<arr::NDArrayPtrVariant> backward(const arr::NDArrayPtrVariant &grad_output,
                                                const std::vector<arr::NDArrayPtrVariant> &parent_data) const override {
        if (parent_data.size() != 2) {
            throw std::runtime_error("MulFunction backward requires exactly 2 parent tensors");
        }

        return std::visit([&](auto grad) -> std::vector<arr::NDArrayPtrVariant> {
            std::vector<arr::NDArrayPtrVariant> grads;

            // REAL FIX: Use parent data passed directly from backward pass
            auto a_data = parent_data[0];
            auto b_data = parent_data[1];

            std::visit([&](auto a, auto b) {
                using dtype = typename std::decay_t<decltype(*grad)>::value_type;
                if (a && b) {  // Check for null pointers
                    // grad_a = grad_output * b, grad_b = grad_output * a
                    auto grad_a = new NDArray<dtype>(grad->executeElementWise(MulOp<dtype>{}, b));
                    auto grad_b = new NDArray<dtype>(grad->executeElementWise(MulOp<dtype>{}, a));
                    grads.push_back(grad_a);
                    grads.push_back(grad_b);
                } else {
                    // Parent was deleted, return zero gradients
                    auto zero_grad_a = new NDArray<dtype>(grad->zeros_like());
                    auto zero_grad_b = new NDArray<dtype>(grad->zeros_like());
                    grads.push_back(zero_grad_a);
                    grads.push_back(zero_grad_b);
                }
            }, a_data, b_data);

            return grads;
        }, grad_output);
    }
};

class SubFunction : public Function {
public:
    std::string getName() const override {
        return "SubFunction";
    }

    arr::NDArrayPtrVariant forward(const std::vector<arr::NDArrayPtrVariant> &inputs) const override {
        if (inputs.size() != 2) {
            throw std::runtime_error("SubFunction requires exactly 2 inputs");
        }

        return std::visit([&](auto a, auto b) -> arr::NDArrayPtrVariant {
            using dtype = typename std::decay_t<decltype(*a)>::value_type;
            auto result = new NDArray<dtype>(a->executeElementWise(AffineAddOp<dtype>{1, -1}, b));
            return result;
        }, inputs[0], inputs[1]);
    }

    std::vector<arr::NDArrayPtrVariant> backward(const arr::NDArrayPtrVariant &grad_output,
                                                const std::vector<arr::NDArrayPtrVariant> &parent_data) const override {
        std::vector<arr::NDArrayPtrVariant> grads;

        std::visit([&](auto grad) {
            using dtype = typename std::decay_t<decltype(*grad)>::value_type;

            auto grad_a = new NDArray<dtype>(*grad);
            auto grad_b = new NDArray<dtype>(grad->executeElementWise(ScalarMulOp<dtype>{-1}));

            grads.push_back(grad_a);
            grads.push_back(grad_b);
        }, grad_output);

        return grads;
    }
};

class DivFunction : public Function {
public:
    std::string getName() const override {
        return "DivFunction";
    }

    arr::NDArrayPtrVariant forward(const std::vector<arr::NDArrayPtrVariant> &inputs) const override {
        if (inputs.size() != 2) {
            throw std::runtime_error("DivFunction requires exactly 2 inputs");
        }

        return std::visit([&](auto a, auto b) -> arr::NDArrayPtrVariant {
            using dtype = typename std::decay_t<decltype(*a)>::value_type;
            auto result = new NDArray<dtype>(a->executeElementWise(DivOp<dtype>{}, b));
            return result;
        }, inputs[0], inputs[1]);
    }

    std::vector<arr::NDArrayPtrVariant> backward(const arr::NDArrayPtrVariant &grad_output,
                                                const std::vector<arr::NDArrayPtrVariant> &parent_data) const override {
        if (parent_data.size() != 2) {
            throw std::runtime_error("DivFunction backward requires exactly 2 parent tensors");
        }

        return std::visit([&](auto grad) -> std::vector<arr::NDArrayPtrVariant> {
            std::vector<arr::NDArrayPtrVariant> grads;

            // REAL FIX: Use parent data passed directly from backward pass
            auto a_data = parent_data[0];
            auto b_data = parent_data[1];

            std::visit([&](auto a, auto b) {
                using dtype = typename std::decay_t<decltype(*grad)>::value_type;
                if (a && b) {  // Check for null pointers
                    // grad_a = grad_output / b
                    auto grad_a = new NDArray<dtype>(grad->executeElementWise(DivOp<dtype>{}, b));
                    // grad_b = -grad_output * a / (b * b)
                    auto b_squared = new NDArray<dtype>(b->executeElementWise(MulOp<dtype>{}, b));
                    auto neg_grad_a = new NDArray<dtype>(grad->executeElementWise(ScalarMulOp<dtype>{-1}));
                    auto numerator = new NDArray<dtype>(neg_grad_a->executeElementWise(MulOp<dtype>{}, a));
                    auto grad_b = new NDArray<dtype>(numerator->executeElementWise(DivOp<dtype>{}, b_squared));

                    // Clean up temporaries
                    delete b_squared;
                    delete neg_grad_a;
                    delete numerator;

                    grads.push_back(grad_a);
                    grads.push_back(grad_b);
                } else {
                    // Parent was deleted, return zero gradients
                    auto zero_grad_a = new NDArray<dtype>(grad->zeros_like());
                    auto zero_grad_b = new NDArray<dtype>(grad->zeros_like());
                    grads.push_back(zero_grad_a);
                    grads.push_back(zero_grad_b);
                }
            }, a_data, b_data);

            return grads;
        }, grad_output);
    }
};

// Factory functions for creating operations
namespace functions {
    inline Function* add() {
        return Function::register_function(std::make_unique<AddFunction>());
    }

    inline Function* mul() {
        return Function::register_function(std::make_unique<MulFunction>());
    }

    inline Function* sub() {
        return Function::register_function(std::make_unique<SubFunction>());
    }

    inline Function* div() {
        return Function::register_function(std::make_unique<DivFunction>());
    }
}

#endif //ARRC_ARITHMETIC_H