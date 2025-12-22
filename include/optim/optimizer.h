#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <variant>
#include "tensor.h"
#include "ndarray.cuh"

using StateValue = std::variant<
          size_t,
          float,
          NDArray*,
          std::vector<NDArray*>
>;
using StateDict = std::unordered_map<std::string, StateValue>;

enum ComputeDType {
    HALF,
    FLOAT,
    DOUBLE
};


/*
 Abstract Base Class for Deep Learning Optimizers.
 Design Pattern 1: STRATEGY (step method).
*/
class Optimizer {
protected:
    std::vector<Tensor*>params;
    float lr;
    float weightDecay;
    size_t t = 0;
    ComputeDType dtype = FLOAT;
public:
    Optimizer(const std::vector<Tensor*> &params, const float &lr, const float &weightDecay,
              const ComputeDType &dtype = FLOAT):
              params(params), lr(lr), weightDecay(weightDecay), dtype(dtype) {
    };
    virtual ~Optimizer() = 0;
    virtual void step() = 0;
    void zeroGrad() {
        for (auto &param: params)
            param->zeroGrad();
    }
    virtual StateDict getStateDict() const = 0;
    virtual void loadStateDict(const StateDict &state) = 0;
    float getLR() const {return lr;}
    float getWeightDecay() const {return weightDecay;}
    size_t getT() const {return t;}
    ComputeDType getDType() const {return dtype;}
    void setLR(const float &lr) {this->lr = lr;}
    void setWeightDecay(const float &weight_decay) {this->weightDecay = weight_decay;}
    friend std::ostream operator<<(std::ostream &os, const Optimizer &opt) = 0;
};