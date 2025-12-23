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


SGD::SGD(const std::vector<tensor::TensorPtrVariant> &params, const float &lr,
        const float &weightDecay, const float &beta, const ComputeDType &dtype):
        Optimizer(params, lr, weightDecay, dtype), beta(beta) {
    try {
        for (const auto &param : params) {
            std::visit([&](auto param) {
                using dtype = decltype(*param->data)::value_type;
                auto mom = new NDArray<dtype>(param->getShape());
                mom->executeElementWise(SetConstantOp<dtype>{static_cast<dtype>(0)}, nullptr, mom);
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

SGD::~SGD() override {
    for (auto &mom: momentum) {
        std::visit([&](auto mom) {delete mom;}, mom);
    }
    momentum.clear();
};

void SGD::step() {
    for (size_t i = 0; i < params.size(); i++) {
        auto run = [&](auto dummy) {
            using dtype = decltype(dummy);
            std::visit([&](auto param, auto mom) {
                if (param->requiresGrad() && param->getGrad() != nullptr) {
                    int NThreads = 256;
                    int NBlocks = getNBlocks(param->getSize(), NThreads);

                    fusedSGDKernel<dtype><<<NBlocks, NThreads>>>(
                        param->getSize(),
                        param->getData()->getData(),
                        param->getGrad()->getData(),
                        mom->getData(),
                        lr,
                        weightDecay,
                        beta
                    );
                };
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









