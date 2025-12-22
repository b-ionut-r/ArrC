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
#include <cuda_fp16.h>

void SGD::step() {
    for (size_t i = 0; i < params.size(); i++) {
        auto *param = params[i]; // Tensor<>*
        auto *mom = momentum[i]; // NDArray<>*
        if (param->requiresGrad() && param->getGrad() != nullptr) {
            int NThreads = 256;
            int NBlocks = getNBlocks(param->getSize(), NThreads);
            // Lambda function
            auto run = [&](auto dummy) {
                using CompT = decltype(dummy);
                fusedSGDKernel<CompT><<<NBlocks, NThreads>>>(
                    param->getSize(),
                    param->getData()->getData(),
                    param->getGrad()->getData(),
                    mom,
                    lr,
                    weightDecay,
                    beta
                );
                // Syncronize and check errors
                cudaDeviceSynchronize();
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    throw CudaKernelException(cudaGetErrorString(err));
                }
            };
            switch (dtype) {
                case HALF: run(__half(0)); break;
                case FLOAT: run(float(0)); break;
                case DOUBLE: run(double(0)); break;
            }
        };
    };
    t++;
}

StateDict SGD::getStateDict() const {
    StateDict state;
    state["lr"] = lr;
    state["weight_decay"] = weightDecay;
    state["beta"] = beta;
    state["t"] = t;
    state["momentum"] = momentum;
    return state;
}

void SGD::loadStateDict(const StateDict &state) {
    lr = state["lr"];
    weightDecay = state["weightDecay"];
    beta = state["beta"];
    t = state["t"];
    momentum = state["momentum"];
}

ostream & operator<<(ostream &os, const SGD &sgd) {
    os << "SGD optimizer: " << endl;
    os << "LR: " << sgd.lr << ", ";
    os << "Weight Decay: " << sgd.weightDecay << ", ";
    os << "Beta: " << sgd.beta << ", ";
    os << "t: " << sgd.t << endl;
    return os;
}









