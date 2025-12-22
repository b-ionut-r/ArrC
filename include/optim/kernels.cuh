template <typename CompT = float, typename ParamT = float, typename GradT = float, typename MomT = float>
__global__ void fusedSGDKernel(
    const size_t size,
    ParamT *param,
    const GradT *grad,
    MomT *momentum,
    const float lr,
    const float weightDecay,
    const float beta
);


template <typename CompT = float, typename ParamT = float, typename GradT = float, typename MomT = float>
__global__ void fusedRMSPropKernel(
    const size_t size,
    ParamT *param,
    const GradT *grad,
    MomT *momentum,
    const float lr,
    const float weightDecay,
    const float beta,
    const double eps
);

template <typename CompT = float, typename ParamT = float, typename GradT = float, typename MomT = float>
__global__ void fusedAdamKernel(
    const size_t size,
    ParamT *param,
    const GradT *grad,
    MomT *first_momentum,
    MomT *second_momentum,
    const float lr,
    const float weightDecay,
    const float beta1,
    const float beta2,
    const double biasCorrection1,
    const double biasCorrection2,
    const double eps
);


