//
// Created by Bujor Ionut Raul on 22.12.2025.
//
#include <iostream>
#include <vector>
#include <string>
#include "optim/sgd.cuh"
#include "optim/kernels.cuh"
#include "exceptions.h"
#include "ndarray.cuh"
#include "tensor.h"
#include "utils.h"


SGD::SGD(const std::vector<tensor::TensorSharedVariant> &params, const float &lr,
        const float &weightDecay, const float &beta, const ComputeDType &dtype):
        Optimizer(params, lr, weightDecay, dtype), beta(beta) {
    try {
        for (const auto &param : params) {
            std::visit([&](auto param_shared) {
                using param_dtype = typename std::decay_t<decltype(*param_shared)>::value_type;
                auto mom = new NDArray<param_dtype>(param_shared->getShape());
                mom->executeElementWise(SetConstantOp<param_dtype>{static_cast<param_dtype>(0)}, nullptr, mom);
                momentum.push_back(mom);
            }, param);
        }
    } catch (...) {
        for (auto &mom: momentum)
            std::visit([&](auto mom) { delete mom; }, mom);
        momentum.clear();
        throw;
    }
};

SGD::~SGD() {
    for (auto &mom: momentum) {
        std::visit([&](auto mom) {delete mom;}, mom);
    }
    momentum.clear();
};

void SGD::step() {
    for (size_t i = 0; i < params.size(); i++) {
        auto run = [&](auto dummy) {
            using dtype = decltype(dummy);
            std::visit([&](auto weak_param, auto mom) {
                // Lock the weak_ptr to get access to the tensor
                if (auto param = weak_param.lock()) {
                    if (param->getRequiresGrad() && param->getGradPtr() != nullptr) {
                        int NThreads = 256;
                        int NBlocks = getNBlocks(param->getSize(), NThreads);

                        fusedSGDKernel<dtype><<<NBlocks, NThreads>>>(
                            param->getSize(),
                            param->getDataPtr()->getData(),
                            param->getGradPtr()->getData(),
                            mom->getData(),
                            lr,
                            weightDecay,
                            beta
                        );
                    }
                }
                // If lock() fails, parameter was deleted - skip it
            }, params[i], momentum[i]);
        };
        switch (dtype) {
            case HALF: run(__half(0)); break;
            case FLOAT: run (float{0}); break;
            case DOUBLE: run(double{0}); break;
        }
    };
    // Syncronize and check errors once per step
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw CudaKernelException(cudaGetErrorString(err));
    }
    t++;
}

ostream & operator<<(ostream &os, const SGD &sgd) {
    os << "SGD optimizer: " << endl;
    os << "LR: " << sgd.lr << ", ";
    os << "Weight Decay: " << sgd.weightDecay << ", ";
    os << "Beta: " << sgd.beta << ", ";
    os << "t: " << sgd.t << endl;
    return os;
}









