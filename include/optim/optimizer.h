#ifndef ARRC_OPTIMIZER_H
#define ARRC_OPTIMIZER_H

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <variant>
#include "tensor.h"
#include "ndarray.cuh"

/*
 Abstract Base Class for Deep Learning Optimizers.
 Design Pattern 1: STRATEGY (step method).
*/
enum ComputeDType {
    HALF,
    FLOAT,
    DOUBLE
};

class Optimizer {
protected:
    std::vector<tensor::TensorWeakVariant> params;  // Weak references to parameters
    float lr;
    float weightDecay;
    size_t t = 0;
    ComputeDType dtype = FLOAT;
public:
    // Constructor takes shared_ptr and stores as weak_ptr
    Optimizer(const std::vector<tensor::TensorSharedVariant> &params, const float &lr, const float &weightDecay,
              const ComputeDType &dtype = FLOAT):
              lr(lr), weightDecay(weightDecay), dtype(dtype) {
        // Convert shared_ptr to weak_ptr for storage
        this->params.reserve(params.size());
        for (const auto& p : params) {
            std::visit([this](auto sp) {
                this->params.push_back(std::weak_ptr(sp));
            }, p);
        }
    };
    virtual ~Optimizer() {}
    virtual void step() = 0;
    void zeroGrad() {
        for (auto &weak_param: params) {
            std::visit([](auto wp) {
                if (auto t = wp.lock()) {
                    t->zeroGrad();
                }
            }, weak_param);
        }
    }
    float getLR() const {return lr;}
    float getWeightDecay() const {return weightDecay;}
    size_t getT() const {return t;}
    ComputeDType getDType() const {return dtype;}
    void setLR(const float &lr) {this->lr = lr;}
    void setWeightDecay(const float &weight_decay) {this->weightDecay = weight_decay;}
    // Note: operator<< is defined in each subclass, not as pure virtual in base
};

#endif // ARRC_OPTIMIZER_H
