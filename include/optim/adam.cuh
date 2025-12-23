//
// Created by Bujor Ionut Raul on 22.12.2025.
//

#ifndef ARRC_ADAM_H
#define ARRC_ADAM_H

#include <iostream>
#include "optimizer.h"
#include "tensor.h"
#include "ndarray.cuh"

class Adam: public Optimizer {
private:
    float beta1, beta2;
    double eps;
    bool adamW;
    std::vector<NDArrayBase*> firstMomentum;
    std::vector<NDArrayBase*> secondMomentum;
public:
    Adam(const std::vector<Tensor*> &params, const float &lr,
         const float &weightDecay, const float &beta1, const float &beta2,
         const double &eps = 1e-8, const ComputeDType &dtype = FLOAT,
         const bool &adamW = false);
    ~Adam() override;
    void step() override;
    StateDict getStateDict() const override;
    void loadStateDict(const StateDict &state) override;
    friend std::ostream & operator<<(std::ostream &os, const Adam &sgd);
    float getBeta1() const {return beta1;}
    float getBeta2() const {return beta2;}
    double getEps() const {return eps;}
};

#endif //ARRC_ADAM_H