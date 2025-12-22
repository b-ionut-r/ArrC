//
// Created by Bujor Ionut Raul on 22.12.2025.
//

#ifndef ARRC_ADAM_H
#define ARRC_ADAM_H

#include <iostream>
#include "optimizer.h"
#include "../ndarray.cuh"
#include "../tensor.h"

class Adam: public Optimizer {
private:
    float beta1, beta2;
    double eps;
    double biasCorrection1, biasCorrection2;
    bool adamW;
    std::vector<NDArray*> firstMomentum;
    std::vector<NDArray*> secondMomentum;
public:
    Adam(const std::vector<Tensor*> &params, const float &lr,
         const float &weightDecay, const float &beta1, const float &beta2,
         const double &eps = 1e-8, const ComputeDType &dtype = FLOAT,
         const bool &adamW = false):
        Optimizer(params, lr, weightDecay, dtype),
        beta1(beta1),beta2(beta2), eps(eps), adamW(adamW){};
    ~Adam() override {
        for (auto &mom: firstMomentum) {
            delete mom;
        }
        firstMomentum.clear();
        for (auto &mom: secondMomentum) {
            delete mom;
        };
        secondMomentum.clear();
    };
    void step() override;
    StateDict getStateDict() const override;
    void loadStateDict(const StateDict &state) override;
    friend std::ostream & operator<<(std::ostream &os, const Adam &sgd);
    float getBeta1() const {return beta1;}
    float getBeta2() const {return beta2;}
    double getEps() const {return eps;}
};

#endif //ARRC_ADAM_H