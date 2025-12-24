//
// Created by Bujor Ionut Raul on 22.12.2025.
//

#include <iostream>
#include <vector>
#include <string>
#include "optim/adam.cuh"
#include "optim/kernels.cuh"
#include "exceptions.h"
#include "ndarray.cuh"
#include "tensor.h"


Adam::Adam(const std::vector<tensor::TensorSharedVariant> &params, const float &lr,
        const float &weightDecay, const float &beta1, const float &beta2,
        const double &eps, const ComputeDType &dtype,
        const bool &adamW):
       Optimizer(params, lr, weightDecay, dtype),
       beta1(beta1),beta2(beta2), eps(eps), adamW(adamW) {
    try {
        for (const auto &param : params) {
            std::visit([&](auto param_shared) {
                using dtype = typename std::decay_t<decltype(*param_shared)>::value_type;
                auto mom = new NDArray<dtype>(param_shared->shape());
                mom->executeElementWise(SetConstantOp<dtype>{static_cast<dtype>(0)}, nullptr, mom);
                firstMomentum.push_back(mom);
            }, param);
        }
        for (const auto &param : params) {
            std::visit([&](auto param_shared) {
                using dtype = typename std::decay_t<decltype(*param_shared)>::value_type;
                auto mom = new NDArray<dtype>(param_shared->shape());
                mom->executeElementWise(SetConstantOp<dtype>{static_cast<dtype>(0)}, nullptr, mom);
                secondMomentum.push_back(mom);
            }, param);
        }
    } catch (...) {
        for (auto &mom : firstMomentum)
            std::visit([&](auto mom) { delete mom; }, mom);
        for (auto &mom : secondMomentum)
            std::visit([&](auto mom) { delete mom; }, mom);
        firstMomentum.clear();
        secondMomentum.clear();
        throw;
    }
};

Adam::~Adam(){
    for (auto &mom : firstMomentum)
        std::visit([&](auto mom){delete mom;}, mom);
    for (auto &mom : secondMomentum)
        std::visit([&](auto mom) {delete mom;}, mom);
    firstMomentum.clear();
    secondMomentum.clear();
}


void Adam::step() {
    t++;  // Increment BEFORE computing bias correction to avoid division by zero
    double biasCorrection1 = 1.0 - pow(beta1, t);
    double biasCorrection2 = 1.0 - pow(beta2, t);
    for (size_t i = 0; i < params.size(); i++) {
        auto run = [&](auto dummy) {
            using dtype = decltype(dummy);
            std::visit([&](auto weak_param, auto m1, auto m2) {
                using param_tensor = typename std::decay_t<decltype(weak_param)>::element_type;
                using param_dtype = typename param_tensor::value_type;
                using m1_dtype = typename std::decay_t<decltype(*m1)>::value_type;
                using m2_dtype = typename std::decay_t<decltype(*m2)>::value_type;

                if constexpr (std::is_same_v<param_dtype, m1_dtype> &&
                              std::is_same_v<param_dtype, m2_dtype>) {
                    if (auto param = weak_param.lock()) {
                        if (param->requiresGrad() && param->grad() != nullptr) {
                            int NThreads = 256;
                            int NBlocks = getNBlocks(param->size(), NThreads);

                            fusedAdamKernel<dtype, param_dtype, param_dtype, param_dtype><<<NBlocks, NThreads>>>(
                                param->size(),
                                param->data()->getData(),
                                param->grad()->getData(),
                                m1->getData(),
                                m2->getData(),
                                lr,
                                weightDecay,
                                beta1,
                                beta2,
                                biasCorrection1,
                                biasCorrection2,
                                eps,
                                adamW
                            );
                        }
                    }
                }
            }, params[i], firstMomentum[i], secondMomentum[i]);
        };
        switch (dtype) {
            case HALF: run(__half(0)); break;
            case FLOAT: run(float{0}); break;
            case DOUBLE: run(double{0}); break;
        }
    }
    // Synchronize and check errors once per step
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw CudaKernelException(cudaGetErrorString(err));
    }
    // Note: t was already incremented at the start of step()
}

ostream & operator<<(ostream &os, const Adam &adam) {
    switch (adam.adamW) {
        case true: os << "AdamW optimizer: "; break;
        case false: os << "Adam optimizer: "; break;
    }
    os << "LR: " << adam.lr << ", ";
    os << "Weight Decay: " << adam.weightDecay << ", ";
    os << "Beta1: " << adam.beta1 << ", ";
    os << "Beta2: " << adam.beta2 << ", ";
    os << "Eps: " << adam.eps << ", ";
    os << "t: " << adam.t << endl;
    return os;
}









