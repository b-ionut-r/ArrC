//
// Created by Bujor Ionut Raul on 22.12.2025.
//

#ifndef ARRC_SGD_H
#define ARRC_SGD_H
#include <iostream>
#include "optimizer.h"
#include "ndarray.cuh"

class SGD: public Optimizer {
private:
    float beta;
    std::vector<arr::NDArrayPtrVariant> momentum;
public:
    SGD(const std::vector<tensor::TensorSharedVariant> &params, const float &lr,
        const float &weightDecay, const float &beta, const ComputeDType &dtype = FLOAT);
    ~SGD() override;
    void step() override;
    friend std::ostream & operator<<(std::ostream &os, const SGD &sgd);
    float getBeta() const {return beta;}
};

#endif //ARRC_SGD_H