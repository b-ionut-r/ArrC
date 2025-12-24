//
// Created by Bujor Ionut Raul on 23.12.2025.
//

#ifndef ARRC_ARITHMETIC_H
#define ARRC_ARITHMETIC_H

#include "base.h"

class AddFunction : public Function {
public:
    arr::NDArrayPtrVariant forward(const std::vector<arr::NDArrayPtrVariant>& inputs) const override {
        if (inputs.size() != 2)
            throw std::runtime_error("AddFunction requires exactly 2 inputs");
        return std::visit([](auto a, auto b) -> arr::NDArrayPtrVariant {
            using Ta = typename std::decay_t<decltype(*a)>::value_type;
            using Tb = typename std::decay_t<decltype(*b)>::value_type;
            if constexpr (std::is_same_v<Ta, Tb>) return new NDArray<Ta>(*a + *b);
            else throw std::runtime_error("AddFunction: type mismatch");
        }, inputs[0], inputs[1]);
    }

    std::vector<arr::NDArrayUniquePtrVariant> backward(
            const arr::NDArrayPtrVariant& gradOut,
            const std::vector<arr::NDArrayPtrVariant>&) const override {
        std::vector<arr::NDArrayUniquePtrVariant> grads;
        std::visit([&](auto g) {
            using T = typename std::decay_t<decltype(*g)>::value_type;
            grads.push_back(std::make_unique<NDArray<T>>(*g));
            grads.push_back(std::make_unique<NDArray<T>>(*g));
        }, gradOut);
        return grads;
    }
};

class SubFunction : public Function {
public:
    arr::NDArrayPtrVariant forward(const std::vector<arr::NDArrayPtrVariant>& inputs) const override {
        if (inputs.size() != 2)
            throw std::runtime_error("SubFunction requires exactly 2 inputs");
        return std::visit([](auto a, auto b) -> arr::NDArrayPtrVariant {
            using Ta = typename std::decay_t<decltype(*a)>::value_type;
            using Tb = typename std::decay_t<decltype(*b)>::value_type;
            if constexpr (std::is_same_v<Ta, Tb>) return new NDArray<Ta>(*a - *b);
            else throw std::runtime_error("SubFunction: type mismatch");
        }, inputs[0], inputs[1]);
    }

    std::vector<arr::NDArrayUniquePtrVariant> backward(
            const arr::NDArrayPtrVariant& gradOut,
            const std::vector<arr::NDArrayPtrVariant>&) const override {
        std::vector<arr::NDArrayUniquePtrVariant> grads;
        std::visit([&](auto g) {
            using T = typename std::decay_t<decltype(*g)>::value_type;
            grads.push_back(std::make_unique<NDArray<T>>(*g));
            grads.push_back(std::make_unique<NDArray<T>>(-*g));
        }, gradOut);
        return grads;
    }
};

class MulFunction : public Function {
public:
    arr::NDArrayPtrVariant forward(const std::vector<arr::NDArrayPtrVariant>& inputs) const override {
        if (inputs.size() != 2)
            throw std::runtime_error("MulFunction requires exactly 2 inputs");
        return std::visit([](auto a, auto b) -> arr::NDArrayPtrVariant {
            using Ta = typename std::decay_t<decltype(*a)>::value_type;
            using Tb = typename std::decay_t<decltype(*b)>::value_type;
            if constexpr (std::is_same_v<Ta, Tb>) return new NDArray<Ta>(*a * *b);
            else throw std::runtime_error("MulFunction: type mismatch");
        }, inputs[0], inputs[1]);
    }

    std::vector<arr::NDArrayUniquePtrVariant> backward(
            const arr::NDArrayPtrVariant& gradOut,
            const std::vector<arr::NDArrayPtrVariant>& parentData) const override {
        if (parentData.size() != 2)
            throw std::runtime_error("MulFunction backward requires 2 parents");
        return std::visit([&](auto g) -> std::vector<arr::NDArrayUniquePtrVariant> {
            using T = typename std::decay_t<decltype(*g)>::value_type;
            std::vector<arr::NDArrayUniquePtrVariant> grads;
            std::visit([&](auto a, auto b) {
                using Ta = typename std::decay_t<decltype(*a)>::value_type;
                using Tb = typename std::decay_t<decltype(*b)>::value_type;
                if constexpr (std::is_same_v<Ta, Tb> && std::is_same_v<Ta, T>) {
                    if (a && b) {
                        grads.push_back(std::make_unique<NDArray<T>>(*g * *b));
                        grads.push_back(std::make_unique<NDArray<T>>(*g * *a));
                    } else {
                        grads.push_back(std::make_unique<NDArray<T>>(g->zeros_like()));
                        grads.push_back(std::make_unique<NDArray<T>>(g->zeros_like()));
                    }
                }
            }, parentData[0], parentData[1]);
            return grads;
        }, gradOut);
    }
};

class DivFunction : public Function {
public:
    arr::NDArrayPtrVariant forward(const std::vector<arr::NDArrayPtrVariant>& inputs) const override {
        if (inputs.size() != 2)
            throw std::runtime_error("DivFunction requires exactly 2 inputs");
        return std::visit([](auto a, auto b) -> arr::NDArrayPtrVariant {
            using Ta = typename std::decay_t<decltype(*a)>::value_type;
            using Tb = typename std::decay_t<decltype(*b)>::value_type;
            if constexpr (std::is_same_v<Ta, Tb>) return new NDArray<Ta>(*a / *b);
            else throw std::runtime_error("DivFunction: type mismatch");
        }, inputs[0], inputs[1]);
    }

    std::vector<arr::NDArrayUniquePtrVariant> backward(
            const arr::NDArrayPtrVariant& gradOut,
            const std::vector<arr::NDArrayPtrVariant>& parentData) const override {
        if (parentData.size() != 2)
            throw std::runtime_error("DivFunction backward requires 2 parents");
        return std::visit([&](auto g) -> std::vector<arr::NDArrayUniquePtrVariant> {
            using T = typename std::decay_t<decltype(*g)>::value_type;
            std::vector<arr::NDArrayUniquePtrVariant> grads;
            std::visit([&](auto a, auto b) {
                using Ta = typename std::decay_t<decltype(*a)>::value_type;
                using Tb = typename std::decay_t<decltype(*b)>::value_type;
                if constexpr (std::is_same_v<Ta, Tb> && std::is_same_v<Ta, T>) {
                    if (a && b) {
                        grads.push_back(std::make_unique<NDArray<T>>(*g / *b));
                        grads.push_back(std::make_unique<NDArray<T>>(-*g * *a / (*b * *b)));
                    } else {
                        grads.push_back(std::make_unique<NDArray<T>>(g->zeros_like()));
                        grads.push_back(std::make_unique<NDArray<T>>(g->zeros_like()));
                    }
                }
            }, parentData[0], parentData[1]);
            return grads;
        }, gradOut);
    }
};

template <typename dtype>
class ScalarAffineFunction : public Function {
    dtype alpha_, beta_;
public:
    ScalarAffineFunction(dtype alpha, dtype beta) : alpha_(alpha), beta_(beta) {}

    arr::NDArrayPtrVariant forward(const std::vector<arr::NDArrayPtrVariant>& inputs) const override {
        if (inputs.size() != 1)
            throw std::runtime_error("ScalarAffineFunction requires exactly 1 input");
        return std::visit([&](auto a) -> arr::NDArrayPtrVariant {
            using T = typename std::decay_t<decltype(*a)>::value_type;
            if constexpr (std::is_same_v<T, dtype>) return new NDArray<dtype>(*a * alpha_ + beta_);
            else throw std::runtime_error("ScalarAffineFunction: type mismatch");
        }, inputs[0]);
    }

    std::vector<arr::NDArrayUniquePtrVariant> backward(
            const arr::NDArrayPtrVariant& gradOut,
            const std::vector<arr::NDArrayPtrVariant>&) const override {
        std::vector<arr::NDArrayUniquePtrVariant> grads;
        std::visit([&](auto g) {
            using T = typename std::decay_t<decltype(*g)>::value_type;
            if constexpr (std::is_same_v<T, dtype>)
                grads.push_back(std::make_unique<NDArray<dtype>>(*g * alpha_));
        }, gradOut);
        return grads;
    }
};

template <typename dtype>
class ScalarRDivFunction : public Function {
    dtype scalar_;
public:
    explicit ScalarRDivFunction(dtype scalar) : scalar_(scalar) {}

    arr::NDArrayPtrVariant forward(const std::vector<arr::NDArrayPtrVariant>& inputs) const override {
        if (inputs.size() != 1)
            throw std::runtime_error("ScalarRDivFunction requires exactly 1 input");
        return std::visit([&](auto a) -> arr::NDArrayPtrVariant {
            using T = typename std::decay_t<decltype(*a)>::value_type;
            if constexpr (std::is_same_v<T, dtype>) return new NDArray<dtype>(scalar_ / *a);
            else throw std::runtime_error("ScalarRDivFunction: type mismatch");
        }, inputs[0]);
    }

    std::vector<arr::NDArrayUniquePtrVariant> backward(
            const arr::NDArrayPtrVariant& gradOut,
            const std::vector<arr::NDArrayPtrVariant>& parentData) const override {
        if (parentData.size() != 1)
            throw std::runtime_error("ScalarRDivFunction backward requires 1 parent");
        return std::visit([&](auto g) -> std::vector<arr::NDArrayUniquePtrVariant> {
            using Tg = typename std::decay_t<decltype(*g)>::value_type;
            std::vector<arr::NDArrayUniquePtrVariant> grads;
            std::visit([&](auto a) {
                using Ta = typename std::decay_t<decltype(*a)>::value_type;
                if constexpr (std::is_same_v<Ta, Tg> && std::is_same_v<Ta, dtype>) {
                    if (a) grads.push_back(std::make_unique<NDArray<dtype>>(-*g * scalar_ / (*a * *a)));
                    else grads.push_back(std::make_unique<NDArray<dtype>>(g->zeros_like()));
                }
            }, parentData[0]);
            return grads;
        }, gradOut);
    }
};

namespace functions {
    inline std::shared_ptr<Function> add() { return std::make_shared<AddFunction>(); }
    inline std::shared_ptr<Function> sub() { return std::make_shared<SubFunction>(); }
    inline std::shared_ptr<Function> mul() { return std::make_shared<MulFunction>(); }
    inline std::shared_ptr<Function> div() { return std::make_shared<DivFunction>(); }

    template <typename dtype>
    std::shared_ptr<Function> affine(dtype alpha, dtype beta) {
        return std::make_shared<ScalarAffineFunction<dtype>>(alpha, beta);
    }

    template <typename dtype>
    std::shared_ptr<Function> rdiv(dtype scalar) {
        return std::make_shared<ScalarRDivFunction<dtype>>(scalar);
    }
}

// Tensor-Tensor operators
template <typename dtype>
Tensor<dtype> operator+(const Tensor<dtype>& a, const Tensor<dtype>& b) {
    auto fn = functions::add();
    return Tensor<dtype>(fn->operator()<TensorPtr<dtype>>({a.shared(), b.shared()}, fn));
}

template <typename dtype>
Tensor<dtype> operator-(const Tensor<dtype>& a, const Tensor<dtype>& b) {
    auto fn = functions::sub();
    return Tensor<dtype>(fn->operator()<TensorPtr<dtype>>({a.shared(), b.shared()}, fn));
}

template <typename dtype>
Tensor<dtype> operator*(const Tensor<dtype>& a, const Tensor<dtype>& b) {
    auto fn = functions::mul();
    return Tensor<dtype>(fn->operator()<TensorPtr<dtype>>({a.shared(), b.shared()}, fn));
}

template <typename dtype>
Tensor<dtype> operator/(const Tensor<dtype>& a, const Tensor<dtype>& b) {
    auto fn = functions::div();
    return Tensor<dtype>(fn->operator()<TensorPtr<dtype>>({a.shared(), b.shared()}, fn));
}

// Unary negation
template <typename dtype>
Tensor<dtype> operator-(const Tensor<dtype>& a) {
    auto fn = functions::affine<dtype>(dtype(-1), dtype(0));
    return Tensor<dtype>(fn->operator()<TensorPtr<dtype>>({a.shared()}, fn));
}

// Tensor-scalar operators
template <typename dtype>
Tensor<dtype> operator+(const Tensor<dtype>& a, dtype v) {
    auto fn = functions::affine<dtype>(dtype(1), v);
    return Tensor<dtype>(fn->operator()<TensorPtr<dtype>>({a.shared()}, fn));
}

template <typename dtype>
Tensor<dtype> operator+(dtype v, const Tensor<dtype>& a) { return a + v; }

template <typename dtype>
Tensor<dtype> operator-(const Tensor<dtype>& a, dtype v) {
    auto fn = functions::affine<dtype>(dtype(1), -v);
    return Tensor<dtype>(fn->operator()<TensorPtr<dtype>>({a.shared()}, fn));
}

template <typename dtype>
Tensor<dtype> operator-(dtype v, const Tensor<dtype>& a) {
    auto fn = functions::affine<dtype>(dtype(-1), v);
    return Tensor<dtype>(fn->operator()<TensorPtr<dtype>>({a.shared()}, fn));
}

template <typename dtype>
Tensor<dtype> operator*(const Tensor<dtype>& a, dtype v) {
    auto fn = functions::affine<dtype>(v, dtype(0));
    return Tensor<dtype>(fn->operator()<TensorPtr<dtype>>({a.shared()}, fn));
}

template <typename dtype>
Tensor<dtype> operator*(dtype v, const Tensor<dtype>& a) { return a * v; }

template <typename dtype>
Tensor<dtype> operator/(const Tensor<dtype>& a, dtype v) {
    auto fn = functions::affine<dtype>(dtype(1) / v, dtype(0));
    return Tensor<dtype>(fn->operator()<TensorPtr<dtype>>({a.shared()}, fn));
}

template <typename dtype>
Tensor<dtype> operator/(dtype v, const Tensor<dtype>& a) {
    auto fn = functions::rdiv<dtype>(v);
    return Tensor<dtype>(fn->operator()<TensorPtr<dtype>>({a.shared()}, fn));
}

#endif //ARRC_ARITHMETIC_H
