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


void RMSProp::step() {
    for (size_t i = 0; i < params.size(); i++) {
        auto *param = params[i]; // Tensor<>*
        auto *mom = momentum[i]; // NDArray<>*
        if (param->requiresGrad() && param->getGrad() != nullptr) {
            int NThreads = 256;
            int NBlocks = getNBlocks(param->getSize(), NThreads);
            auto run = [&](auto dummy) {
                using CompT = decltype(dummy);
                fusedRMSPropKernel<CompT><<<NBlocks, NThreads>>>(
                    param->getSize(),
                    param->getData()->getData(),
                    param->getGrad()->getData(),
                    mom,
                    lr,
                    weightDecay,
                    beta,
                    eps
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

StateDict RMSProp::getStateDict() const {
    StateDict state;
    state["lr"] = lr;
    state["weight_decay"] = weightDecay;
    state["beta"] = beta;
    state["t"] = t;
    state["momentum"] = momentum;
    state["eps"] = eps;
    return state;
}

void RMSProp::loadStateDict(const StateDict &state) {
    lr = state["lr"];
    weightDecay = state["weightDecay"];
    beta = state["beta"];
    t = state["t"];
    momentum = state["momentum"];
    eps = state["eps"];
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









