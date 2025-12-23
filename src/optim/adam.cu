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


Adam::Adam(const std::vector<tensor::TensorPtrVariant> &params, const float &lr,
        const float &weightDecay, const float &beta1, const float &beta2,
        const double &eps, const ComputeDType &dtype,
        const bool &adamW):
       Optimizer(params, lr, weightDecay, dtype),
       beta1(beta1),beta2(beta2), eps(eps), adamW(adamW) {
    for (const auto &param : params) {
        std::visit([&](auto param) {
            using dtype = decltype(*param->data)::value_type;
            auto mom = new NDArray<dtype>(param->getShape());
            mom->executeElementWise(SetConstantOp<dtype>{static_cast<dtype>(0)}, nullptr, mom);
            firstMomentum.push_back(mom);
        }, param);
    }
    for (const auto &param : params) {
        std::visit([&](auto param) {
            using dtype = decltype(*param->data)::value_type;
            auto mom = new NDArray<dtype>(param->getShape());
            mom->executeElementWise(SetConstantOp<dtype>{static_cast<dtype>(0)}, nullptr, mom);
            secondMomentum.push_back(mom);
        }, param);
    }
};

Adam::~Adam() override{
    for (auto &mom : firstMomentum)
        std::visit([&](auto mom){delete mom;}, mom);
    for (auto &mom : secondMomentum)
        std::visit([&](auto mom) {delete mom;}, mom);
    firstMomentum.clear();
    secondMomentum.clear();
}


void Adam::step() {
    double biasCorrection1 = 1 - pow(beta1, t);
    double biasCorrection2 = 1 - pow(beta2, t);
    for (size_t i = 0; i < params.size(); i++) {
        auto run = [&](auto dummy) {
            using dtype = decltype(dummy);
            std::visit([&](auto param, auto m1, auto m2) {
                if (param->requiresGrad() && param->getGrad() != nullptr) {
                    int NThreads = 256;
                    int NBlocks = getNBlocks(param->getSize(), NThreads);

                    fusedAdamKernel<dtype><<<NBlocks, NThreads>>>(
                        param->getSize(),
                        param->getData()->getData(),
                        param->getGrad()->getData(),
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
            }, params[i], firstMomentum[i], secondMomentum[i]);
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









