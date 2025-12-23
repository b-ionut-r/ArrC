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


SGD::SGD(const std::vector<TensorBase*> &params, const float &lr,
        const float &weightDecay, const float &beta, const ComputeDType &dtype):
        Optimizer(params, lr, weightDecay, dtype), beta(beta) {
    for (int i = 0; i < params.size(); i++) {
        using dtype = typename decltype(*params[i]->data)::value_type;
        NDArray<dtype> *mom = new NDArray<dtype>(params[i]->getShape());
        mom->executeElementWise(SetConstantOp<dtype>{(dtype)0}, nullptr, mom);
        momentum.push_back(mom);
    }
};

SGD::~SGD() override {
    for (auto &mom: momentum) {
        delete mom;
    }
    momentum.clear();
};

void SGD::step() {
    for (size_t i = 0; i < params.size(); i++) {
        auto *param = params[i]; // TensorBase*
        auto *mom = momentum[i]; // NDArrayBase*
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

ostream & operator<<(ostream &os, const SGD &sgd) {
    os << "SGD optimizer: " << endl;
    os << "LR: " << sgd.lr << ", ";
    os << "Weight Decay: " << sgd.weightDecay << ", ";
    os << "Beta: " << sgd.beta << ", ";
    os << "t: " << sgd.t << endl;
    return os;
}









