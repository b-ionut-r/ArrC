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
    std::vector<tensor::TensorPtrVariant>params;
    float lr;
    float weightDecay;
    size_t t = 0;
    ComputeDType dtype = FLOAT;
public:
    Optimizer(const std::vector<tensor::TensorPtrVariant> &params, const float &lr, const float &weightDecay,
              const ComputeDType &dtype = FLOAT):
              params(params), lr(lr), weightDecay(weightDecay), dtype(dtype) {
    };
    virtual ~Optimizer() = 0;
    virtual void step() = 0;
    void zeroGrad() {
        for (auto &param: params)
            std::visit([](auto t){t->zeroGrad();}, param);
    }
    float getLR() const {return lr;}
    float getWeightDecay() const {return weightDecay;}
    size_t getT() const {return t;}
    ComputeDType getDType() const {return dtype;}
    void setLR(const float &lr) {this->lr = lr;}
    void setWeightDecay(const float &weight_decay) {this->weightDecay = weight_decay;}
    friend std::ostream operator<<(std::ostream &os, const Optimizer &opt) = 0;
};

Optimizer::~Optimizer(){};