//
// Created by Bujor Ionut Raul on 23.12.2025.
//

#ifndef ARRC_LOSSES_H
#define ARRC_LOSSES_H

#include "../ndarray.cuh"

class Loss {
public:
    virtual ~Loss() = default;
    virtual arr::NDArrayPtrVariant forward(const arr::NDArrayPtrVariant& predictions,
                                          const arr::NDArrayPtrVariant& targets) = 0;
    virtual arr::NDArrayPtrVariant backward() = 0;
};


#endif //ARRC_LOSSES_H