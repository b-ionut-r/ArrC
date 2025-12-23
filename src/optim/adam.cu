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
#include <cuda_fp16.h>

Adam::Adam(const std::vector<Tensor*> &params, const float &lr,
        const float &weightDecay, const float &beta1, const float &beta2,
        const double &eps, const ComputeDType &dtype,
        const bool &adamW):
       Optimizer(params, lr, weightDecay, dtype),
       beta1(beta1),beta2(beta2), eps(eps), adamW(adamW) {
    for (int i = 0; i < params.size(); i++) {
        using dtype = typename decltype(*params[i]->data)::value_type;
        NDArray<dtype> *mom = new NDArray<dtype>(params[i]->getShape());
        mom->executeElementWise(SetConstantOp<dtype>{(dtype)0}, nullptr, mom);
        firstMomentum.push_back(mom);
    }
    for (int i = 0; i < params.size(); i++) {
        using dtype = typename decltype(*params[i]->data)::value_type;
        NDArray<dtype> *mom = new NDArray<dtype>(params[i]->getShape());
        mom->executeElementWise(SetConstantOp<dtype>{(dtype)0}, nullptr, mom);
        secondMomentum.push_back(mom);
    }
};

Adam::~Adam() override{
    for (auto &mom : firstMomentum)
        delete mom;
    for (auto &mom : secondMomentum)
        delete mom;
    firstMomentum.clear();
    secondMomentum.clear();
}


void Adam::step() {
    double biasCorrection1 = 1 - pow(beta1, t);
    double biasCorrection2 = 1 - pow(beta2, t);
    for (size_t i = 0; i < params.size(); i++) {
        auto *param = params[i]; // Tensor<>*
        auto *m1 = firstMomentum[i]; // NDArray<>*
        auto *m2 = secondMomentum[i];
        if (param->requiresGrad() && param->getGrad() != nullptr) {
            int NThreads = 256;
            int NBlocks = getNBlocks(param->getSize(), NThreads);
            auto run = [&](auto dummy) {
                using CompT = decltype(dummy);
                fusedAdamKernel<CompT><<<NBlocks, NThreads>>>(
                    param->getSize(),
                    param->getData()->getData(),
                    param->getGrad()->getData(),
                    m1,
                    m2,
                    lr,
                    weightDecay,
                    beta1,
                    beta2,
                    biasCorrection1,
                    biasCorrection2,
                    eps,
                    adamW
                );
                // Syncronize and check errors
                cudaDeviceSynchronize();
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    throw CudaKernelException(cudaGetErrorString(err));
                }
            };
            switch (dtype) {
                case HALF: run(__half{0}); break;
                case FLOAT: run(float{0}); break;
                case DOUBLE: run(double{0}); break;
            }
        };
    };
    t++;
}

StateDict Adam::getStateDict() const {
    StateDict state;
    state["lr"] = lr;
    state["weight_decay"] = weightDecay;
    state["beta1"] = beta1;
    state["beta2"] = beta2;
    state["t"] = t;
    state["firstMomentum"] = firstMomentum;
    state["secondMomentum"] = secondMomentum;
    state["eps"] = eps;
    state["adamW"] = adamW;
    return state;
}

void Adam::loadStateDict(const StateDict &state) {
    lr = state["lr"];
    weightDecay = state["weightDecay"];
    beta1 = state["beta"];
    beta2 = state["beta2"];
    t = state["t"];
    firstMomentum = state["firstMomentum"];
    secondMomentum = state["secondMomentum"];
    eps = state["eps"];
    adamW = state["adamW"];
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









