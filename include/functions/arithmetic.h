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
            using dtype_a = typename std::decay_t<decltype(*a)>::value_type;
            using dtype_b = typename std::decay_t<decltype(*b)>::value_type;
            if constexpr (std::is_same_v<dtype_a, dtype_b>) {
                return new NDArray<dtype_a>(*a + *b);
            } else {
                throw std::runtime_error("AddFunction: type mismatch between inputs");
            }
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
            using dtype_a = typename std::decay_t<decltype(*a)>::value_type;
            using dtype_b = typename std::decay_t<decltype(*b)>::value_type;
            if constexpr (std::is_same_v<dtype_a, dtype_b>) {
                return new NDArray<dtype_a>(*a - *b);
            } else {
                throw std::runtime_error("SubFunction: type mismatch between inputs");
            }
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
            using dtype_a = typename std::decay_t<decltype(*a)>::value_type;
            using dtype_b = typename std::decay_t<decltype(*b)>::value_type;
            if constexpr (std::is_same_v<dtype_a, dtype_b>) {
                return new NDArray<dtype_a>(*a * *b);
            } else {
                throw std::runtime_error("MulFunction: type mismatch between inputs");
            }
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
                using dtype_a = typename std::decay_t<decltype(*a)>::value_type;
                using dtype_b = typename std::decay_t<decltype(*b)>::value_type;
                if constexpr (std::is_same_v<dtype_a, dtype_b> && std::is_same_v<dtype_a, dtype>) {
                    if (a && b) {
                        grads.push_back(std::make_unique<NDArray<dtype>>(*grad * *b));  // d(a*b)/da = b
                        grads.push_back(std::make_unique<NDArray<dtype>>(*grad * *a));  // d(a*b)/db = a
                    } else {
                        grads.push_back(std::make_unique<NDArray<dtype>>(grad->zeros_like()));
                        grads.push_back(std::make_unique<NDArray<dtype>>(grad->zeros_like()));
                    }
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
            using dtype_a = typename std::decay_t<decltype(*a)>::value_type;
            using dtype_b = typename std::decay_t<decltype(*b)>::value_type;
            if constexpr (std::is_same_v<dtype_a, dtype_b>) {
                return new NDArray<dtype_a>(*a / *b);
            } else {
                throw std::runtime_error("DivFunction: type mismatch between inputs");
            }
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
                using dtype_a = typename std::decay_t<decltype(*a)>::value_type;
                using dtype_b = typename std::decay_t<decltype(*b)>::value_type;
                if constexpr (std::is_same_v<dtype_a, dtype_b> && std::is_same_v<dtype_a, dtype>) {
                    if (a && b) {
                        // d(a/b)/da = 1/b
                        grads.push_back(std::make_unique<NDArray<dtype>>(*grad / *b));
                        // d(a/b)/db = -a/b^2
                        grads.push_back(std::make_unique<NDArray<dtype>>(-*grad * *a / (*b * *b)));
                    } else {
                        grads.push_back(std::make_unique<NDArray<dtype>>(grad->zeros_like()));
                        grads.push_back(std::make_unique<NDArray<dtype>>(grad->zeros_like()));
                    }
                }
            }, parent_data[0], parent_data[1]);
            return grads;
        }, grad_output);
    }
};

template <typename dtype>
class ScalarAffineFunction : public Function {
    dtype alpha, beta;
public:
    ScalarAffineFunction(dtype alpha, dtype beta): alpha(alpha), beta(beta) {}
    std::string getName() const override { return "ScalarAffineFunction"; }

    arr::NDArrayPtrVariant forward(const std::vector<arr::NDArrayPtrVariant> &inputs) const override {
        if (inputs.size() != 1)
            throw std::runtime_error("ScalarAffineFunction requires exactly 1 input");
        return std::visit([&](auto a) -> arr::NDArrayPtrVariant {
            using dtype_a = typename std::decay_t<decltype(*a)>::value_type;
            if constexpr (std::is_same_v<dtype_a, dtype>) {
                return new NDArray<dtype>(*a * alpha + beta);
            } else {
                throw std::runtime_error("ScalarAffineFunction: type mismatch between input and scalar");
            }
        }, inputs[0]);
    }

    std::vector<arr::NDArrayUniquePtrVariant> backward(const arr::NDArrayPtrVariant &grad_output,
                                                       const std::vector<arr::NDArrayPtrVariant>&) const override {
        std::vector<arr::NDArrayUniquePtrVariant> grads;
        std::visit([&](auto grad) {
            using dtype_g = typename std::decay_t<decltype(*grad)>::value_type;
            if constexpr (std::is_same_v<dtype_g, dtype>) {
                grads.push_back(std::make_unique<NDArray<dtype>>(*grad * alpha));
            }
        }, grad_output);
        return grads;
    }
};

template <typename dtype>
class ScalarRDivFunction : public Function {
    dtype scalar;
public:
    explicit ScalarRDivFunction(dtype scalar): scalar(scalar) {}
    std::string getName() const override { return "ScalarRDivFunction"; }

    arr::NDArrayPtrVariant forward(const std::vector<arr::NDArrayPtrVariant> &inputs) const override {
        if (inputs.size() != 1)
            throw std::runtime_error("ScalarRDivFunction requires exactly 1 input");
        return std::visit([&](auto a) -> arr::NDArrayPtrVariant {
            using dtype_a = typename std::decay_t<decltype(*a)>::value_type;
            if constexpr (std::is_same_v<dtype_a, dtype>) {
                return new NDArray<dtype>(scalar / *a);
            } else {
                throw std::runtime_error("ScalarRDivFunction: type mismatch between input and scalar");
            }
        }, inputs[0]);
    }

    std::vector<arr::NDArrayUniquePtrVariant> backward(const arr::NDArrayPtrVariant &grad_output,
                                                       const std::vector<arr::NDArrayPtrVariant> &parent_data) const override {
        if (parent_data.size() != 1)
            throw std::runtime_error("ScalarRDivFunction backward requires exactly 1 parent tensor");
        return std::visit([&](auto grad) -> std::vector<arr::NDArrayUniquePtrVariant> {
            using dtype_g = typename std::decay_t<decltype(*grad)>::value_type;
            std::vector<arr::NDArrayUniquePtrVariant> grads;
            std::visit([&](auto a) {
                using dtype_a = typename std::decay_t<decltype(*a)>::value_type;
                if constexpr (std::is_same_v<dtype_a, dtype_g> && std::is_same_v<dtype_a, dtype>) {
                    if (a) grads.push_back(std::make_unique<NDArray<dtype>>(-*grad * scalar / (*a * *a)));
                    else grads.push_back(std::make_unique<NDArray<dtype>>(grad->zeros_like()));
                }
            }, parent_data[0]);
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
    template <typename dtype>
    inline std::shared_ptr<Function> affine(dtype alpha, dtype beta) {
        return std::make_shared<ScalarAffineFunction<dtype>>(alpha, beta);
    }
    template <typename dtype>
    inline std::shared_ptr<Function> rdiv(dtype scalar) {
        return std::make_shared<ScalarRDivFunction<dtype>>(scalar);
    }
}

template <typename dtype>
inline std::shared_ptr<TensorPtr<dtype>> operator+(const std::shared_ptr<TensorPtr<dtype>> &a,
                                                   const std::shared_ptr<TensorPtr<dtype>> &b) {
    auto fn = functions::add(); return fn->operator()<TensorPtr<dtype>>({a, b}, fn);
}
template <typename dtype>
inline std::shared_ptr<TensorPtr<dtype>> operator-(const std::shared_ptr<TensorPtr<dtype>> &a,
                                                   const std::shared_ptr<TensorPtr<dtype>> &b) {
    auto fn = functions::sub(); return fn->operator()<TensorPtr<dtype>>({a, b}, fn);
}
template <typename dtype>
inline std::shared_ptr<TensorPtr<dtype>> operator*(const std::shared_ptr<TensorPtr<dtype>> &a,
                                                   const std::shared_ptr<TensorPtr<dtype>> &b) {
    auto fn = functions::mul(); return fn->operator()<TensorPtr<dtype>>({a, b}, fn);
}
template <typename dtype>
inline std::shared_ptr<TensorPtr<dtype>> operator/(const std::shared_ptr<TensorPtr<dtype>> &a,
                                                   const std::shared_ptr<TensorPtr<dtype>> &b) {
    auto fn = functions::div(); return fn->operator()<TensorPtr<dtype>>({a, b}, fn);
}
template <typename dtype>
inline std::shared_ptr<TensorPtr<dtype>> operator-(const std::shared_ptr<TensorPtr<dtype>> &a) {
    auto fn = functions::affine<dtype>(static_cast<dtype>(-1), static_cast<dtype>(0));
    return fn->operator()<TensorPtr<dtype>>({a}, fn);
}
template <typename dtype>
inline std::shared_ptr<TensorPtr<dtype>> operator+(const std::shared_ptr<TensorPtr<dtype>> &a, const dtype &value) {
    auto fn = functions::affine<dtype>(static_cast<dtype>(1), value); return fn->operator()<TensorPtr<dtype>>({a}, fn);
}
template <typename dtype>
inline std::shared_ptr<TensorPtr<dtype>> operator+(const dtype &value, const std::shared_ptr<TensorPtr<dtype>> &a) {
    return a + value;
}
template <typename dtype>
inline std::shared_ptr<TensorPtr<dtype>> operator-(const std::shared_ptr<TensorPtr<dtype>> &a, const dtype &value) {
    auto fn = functions::affine<dtype>(static_cast<dtype>(1), static_cast<dtype>(-value));
    return fn->operator()<TensorPtr<dtype>>({a}, fn);
}
template <typename dtype>
inline std::shared_ptr<TensorPtr<dtype>> operator-(const dtype &value, const std::shared_ptr<TensorPtr<dtype>> &a) {
    auto fn = functions::affine<dtype>(static_cast<dtype>(-1), value); return fn->operator()<TensorPtr<dtype>>({a}, fn);
}
template <typename dtype>
inline std::shared_ptr<TensorPtr<dtype>> operator*(const std::shared_ptr<TensorPtr<dtype>> &a, const dtype &value) {
    auto fn = functions::affine<dtype>(value, static_cast<dtype>(0)); return fn->operator()<TensorPtr<dtype>>({a}, fn);
}
template <typename dtype>
inline std::shared_ptr<TensorPtr<dtype>> operator*(const dtype &value, const std::shared_ptr<TensorPtr<dtype>> &a) {
    return a * value;
}
template <typename dtype>
inline std::shared_ptr<TensorPtr<dtype>> operator/(const std::shared_ptr<TensorPtr<dtype>> &a, const dtype &value) {
    auto fn = functions::affine<dtype>(static_cast<dtype>(1) / value, static_cast<dtype>(0));
    return fn->operator()<TensorPtr<dtype>>({a}, fn);
}
template <typename dtype>
inline std::shared_ptr<TensorPtr<dtype>> operator/(const dtype &value, const std::shared_ptr<TensorPtr<dtype>> &a) {
    auto fn = functions::rdiv<dtype>(value); return fn->operator()<TensorPtr<dtype>>({a}, fn);
}

template <typename dtype>
inline Tensor<dtype> operator+(const Tensor<dtype> &a, const Tensor<dtype> &b) {
    return Tensor<dtype>(a.shared() + b.shared());
}
template <typename dtype>
inline Tensor<dtype> operator-(const Tensor<dtype> &a, const Tensor<dtype> &b) {
    return Tensor<dtype>(a.shared() - b.shared());
}
template <typename dtype>
inline Tensor<dtype> operator*(const Tensor<dtype> &a, const Tensor<dtype> &b) {
    return Tensor<dtype>(a.shared() * b.shared());
}
template <typename dtype>
inline Tensor<dtype> operator/(const Tensor<dtype> &a, const Tensor<dtype> &b) {
    return Tensor<dtype>(a.shared() / b.shared());
}
template <typename dtype>
inline Tensor<dtype> operator-(const Tensor<dtype> &a) {
    return Tensor<dtype>(-a.shared());
}
template <typename dtype>
inline Tensor<dtype> operator+(const Tensor<dtype> &a, const dtype &value) {
    return Tensor<dtype>(a.shared() + value);
}
template <typename dtype>
inline Tensor<dtype> operator+(const dtype &value, const Tensor<dtype> &a) {
    return Tensor<dtype>(value + a.shared());
}
template <typename dtype>
inline Tensor<dtype> operator-(const Tensor<dtype> &a, const dtype &value) {
    return Tensor<dtype>(a.shared() - value);
}
template <typename dtype>
inline Tensor<dtype> operator-(const dtype &value, const Tensor<dtype> &a) {
    return Tensor<dtype>(value - a.shared());
}
template <typename dtype>
inline Tensor<dtype> operator*(const Tensor<dtype> &a, const dtype &value) {
    return Tensor<dtype>(a.shared() * value);
}
template <typename dtype>
inline Tensor<dtype> operator*(const dtype &value, const Tensor<dtype> &a) {
    return Tensor<dtype>(value * a.shared());
}
template <typename dtype>
inline Tensor<dtype> operator/(const Tensor<dtype> &a, const dtype &value) {
    return Tensor<dtype>(a.shared() / value);
}
template <typename dtype>
inline Tensor<dtype> operator/(const dtype &value, const Tensor<dtype> &a) {
    return Tensor<dtype>(value / a.shared());
}

template <typename dtype>
inline TensorPtr<dtype> operator+(const TensorPtr<dtype> &a, const TensorPtr<dtype> &b) {
    return TensorPtr<dtype>(std::make_unique<NDArray<dtype>>(*a.getDataPtr() + *b.getDataPtr()), false);
}
template <typename dtype>
inline TensorPtr<dtype> operator-(const TensorPtr<dtype> &a, const TensorPtr<dtype> &b) {
    return TensorPtr<dtype>(std::make_unique<NDArray<dtype>>(*a.getDataPtr() - *b.getDataPtr()), false);
}
template <typename dtype>
inline TensorPtr<dtype> operator*(const TensorPtr<dtype> &a, const TensorPtr<dtype> &b) {
    return TensorPtr<dtype>(std::make_unique<NDArray<dtype>>(*a.getDataPtr() * *b.getDataPtr()), false);
}
template <typename dtype>
inline TensorPtr<dtype> operator/(const TensorPtr<dtype> &a, const TensorPtr<dtype> &b) {
    return TensorPtr<dtype>(std::make_unique<NDArray<dtype>>(*a.getDataPtr() / *b.getDataPtr()), false);
}
template <typename dtype>
inline TensorPtr<dtype> operator-(const TensorPtr<dtype> &a) {
    return TensorPtr<dtype>(std::make_unique<NDArray<dtype>>(-*a.getDataPtr()), false);
}
template <typename dtype>
inline TensorPtr<dtype> operator+(const TensorPtr<dtype> &a, const dtype &value) {
    return TensorPtr<dtype>(std::make_unique<NDArray<dtype>>(*a.getDataPtr() + value), false);
}
template <typename dtype>
inline TensorPtr<dtype> operator+(const dtype &value, const TensorPtr<dtype> &a) { return a + value; }
template <typename dtype>
inline TensorPtr<dtype> operator-(const TensorPtr<dtype> &a, const dtype &value) {
    return TensorPtr<dtype>(std::make_unique<NDArray<dtype>>(*a.getDataPtr() - value), false);
}
template <typename dtype>
inline TensorPtr<dtype> operator-(const dtype &value, const TensorPtr<dtype> &a) {
    return TensorPtr<dtype>(std::make_unique<NDArray<dtype>>(value - *a.getDataPtr()), false);
}
template <typename dtype>
inline TensorPtr<dtype> operator*(const TensorPtr<dtype> &a, const dtype &value) {
    return TensorPtr<dtype>(std::make_unique<NDArray<dtype>>(*a.getDataPtr() * value), false);
}
template <typename dtype>
inline TensorPtr<dtype> operator*(const dtype &value, const TensorPtr<dtype> &a) { return a * value; }
template <typename dtype>
inline TensorPtr<dtype> operator/(const TensorPtr<dtype> &a, const dtype &value) {
    return TensorPtr<dtype>(std::make_unique<NDArray<dtype>>(*a.getDataPtr() / value), false);
}
template <typename dtype>
inline TensorPtr<dtype> operator/(const dtype &value, const TensorPtr<dtype> &a) {
    return TensorPtr<dtype>(std::make_unique<NDArray<dtype>>(value / *a.getDataPtr()), false);
}

#endif //ARRC_ARITHMETIC_H
