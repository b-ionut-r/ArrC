//
// Created by Bujor Ionut Raul on 23.12.2025.
//

#ifndef ARRC_BASE_H
#define ARRC_BASE_H

#include <string>
#include <variant>
#include <vector>
#include <memory>
#include "../ndarray.cuh"
#include "../tensor.h"

class Function {
private:
    static std::vector<std::unique_ptr<Function>> function_registry;
    static bool cleanup_registered;
public:
    // REAL FIX: Simple solution - Functions store tensor pointers but DON'T access them in backward
    // Instead, tensors pass their current data directly to backward() calls
    std::vector<tensor::TensorPtrVariant> parent_tensors = {};

    static void ensure_cleanup_registered() {
        if (!cleanup_registered) {
            std::atexit(cleanup_registry);
            cleanup_registered = true;
        }
    }
public:
    Function() = default;
    virtual ~Function() = default;

    static Function* register_function(std::unique_ptr<Function> fn) {
        ensure_cleanup_registered();
        Function* ptr = fn.get();
        function_registry.push_back(std::move(fn));
        return ptr;
    }

    static void cleanup_registry() {
        function_registry.clear();
    }

    // Unregister a specific function from the registry (called after backward)
    static void unregister_function(Function* fn) {
        if (fn == nullptr) return;
        auto it = std::remove_if(function_registry.begin(), function_registry.end(),
            [fn](const std::unique_ptr<Function>& f) {
                return f.get() == fn;
            });
        function_registry.erase(it, function_registry.end());
    }

    // Get current registry size (for debugging/testing)
    static size_t registry_size() {
        return function_registry.size();
    }
    tensor::TensorVariant operator()(const std::vector<tensor::TensorVariant> &inputs);
    virtual std::string getName() const = 0;
    virtual arr::NDArrayPtrVariant forward(const std::vector<arr::NDArrayPtrVariant> &inputs) const = 0;
    // REAL FIX: Pass parent data directly to backward, don't rely on stored pointers
    virtual std::vector<arr::NDArrayPtrVariant> backward(const arr::NDArrayPtrVariant &grad_output,
                                                         const std::vector<arr::NDArrayPtrVariant> &parent_data) const = 0;
};

tensor::TensorVariant Function::operator()(const std::vector<tensor::TensorVariant> &inputs) {
    parent_tensors.clear();
    parent_tensors.reserve(inputs.size());

    std::vector<arr::NDArrayPtrVariant> parents_data;
    parents_data.reserve(inputs.size());
    bool requiresGrad = false;

    for (const auto &parent: inputs) {
        // Store tensor references for backward pass
        std::visit([&](auto parent_ptr) {
            parent_tensors.push_back(parent_ptr);
            if (parent_ptr->requiresGrad)
                requiresGrad = true;
            parents_data.push_back(parent_ptr->getDataPtr());
        }, parent);
    }

    auto output = forward(parents_data);
    return std::visit([&](auto output) {
        using dtype = typename std::decay_t<decltype(*output)>::value_type;
        return Tensor<dtype>(output, requiresGrad, requiresGrad ? this : nullptr);
    }, output);
}

// Static member definition
std::vector<std::unique_ptr<Function>> Function::function_registry;
bool Function::cleanup_registered = false;

#endif //ARRC_BASE_H