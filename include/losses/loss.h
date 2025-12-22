//
// Created by Bujor Ionut Raul on 22.12.2025.
//

#ifndef ARRC_LOSS_H
#define ARRC_LOSS_H
#include "../ndarray.cuh"

class Loss {
protected:
public:
    virtual ~Loss() = default;
    virtual std::string name() const = 0;
    virtual NDArray<float> forward(const NDArray )

};

#endif //ARRC_LOSS_H