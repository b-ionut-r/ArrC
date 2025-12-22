//
// Created by Bujor Ionut Raul on 22.12.2025.
//

#ifndef ARRC_SGD_H
#define ARRC_SGD_H
#include <iostream>
#include "optimizer.h"

class SGD: public Optimizer {
private:
    float beta;
    std::vector<NDArray*> momentum;
public:
    SGD(const std::vector<Tensor*> &params, const float &lr,
        const float &weightDecay, const float &beta, const ComputeDType &dtype = FLOAT):
        Optimizer(params, lr, weightDecay, dtype), beta(beta) {};
    ~SGD() override {
        for (auto &mom: momentum) {
            delete mom;
        }
        momentum.clear();
    };
    void step() override;
    StateDict getStateDict() const override;
    void loadStateDict(const StateDict &state) override;
    friend std::ostream & operator<<(std::ostream &os, const SGD &sgd);
    float getBeta() const {return beta;}
};

#endif //ARRC_SGD_H