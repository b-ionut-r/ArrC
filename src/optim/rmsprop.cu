//
// Created by Bujor Ionut Raul on 22.12.2025.
//
#include <iostream>
#include <vector>
#include <string>
#include "optim/rmsprop.cuh"
#include "optim/kernels.cuh"
#include "exceptions.h"
#include "ndarray.cuh"
#include "tensor.h"
#include "utils.h"
#include <cuda_fp16.h>


RMSProp::RMSProp(const std::vector<tensor::TensorPtrVariant> &params, const float &lr,
                 const float &weightDecay, const float &beta,
                 const double &eps, const ComputeDType &dtype):
       Optimizer(params, lr, weightDecay, dtype), beta(beta), eps(eps) {
    try {
        for (const auto &param: params) {
            std::visit([&](auto param) {
                using param_dtype = typename std::decay_t<decltype(*param)>::value_type;
                auto mom = new NDArray<param_dtype>(param->getShape());
                mom->executeElementWise(SetConstantOp<param_dtype>{static_cast<param_dtype>(0)}, nullptr, mom);
                momentum.push_back(mom);
            }, param);
        }
    } catch (...) {
        for (auto &mom: momentum)
            std::visit([&](auto mom){ delete mom; }, mom);
        momentum.clear();
        throw;
    }
};

RMSProp::~RMSProp() {
    for (auto &mom: momentum) {
        std::visit([&](auto mom){delete mom;}, mom);
    }
    momentum.clear();
};

void RMSProp::step() {
    for (size_t i = 0; i < params.size(); i++) {
        auto run = [&](auto dummy) {
            using dtype = decltype(dummy);
            std::visit([&](auto param, auto mom) {
                if (param->requiresGrad && param->getGradPtr() != nullptr) {
                    int NThreads = 256;
                    int NBlocks = getNBlocks(param->getSize(), NThreads);
                    fusedRMSPropKernel<dtype><<<NBlocks, NThreads>>>(
                        param->getSize(),
                        param->getDataPtr()->getData(),
                        param->getGradPtr()->getData(),
                        mom->getData(),
                        lr,
                        weightDecay,
                        beta,
                        eps
                    );
                };
            }, params[i], momentum[i]);
        };
        switch (dtype) {
            case HALF: run(__half(0)); break;
            case FLOAT: run (float{0}); break;
            case DOUBLE: run(double{0}); break;
        }
    }
    // Syncronize and check errors once per step
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw CudaKernelException(cudaGetErrorString(err));
    }
    t++;
}

ostream & operator<<(ostream &os, const RMSProp &rms) {
    os << "RMSProp optimizer: " << endl;
    os << "LR: " << rms.lr << ", ";
    os << "Weight Decay: " << rms.weightDecay << ", ";
    os << "Beta: " << rms.beta << ", ";
    os << "Eps: " << rms.eps << ", ";
    os << "t: " << rms.t << endl;
    return os;
}









