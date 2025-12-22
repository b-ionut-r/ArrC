cmake_minimum_required(VERSION 4.0)
project(ArrC CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(ArrC
        include/ndarray.cuh
        include/exceptions.h
        src/optim/sgd.cu
        include/tensor.h
        include/serialize.h
        include/optim/sgd.cuh
        src/optim/rmsprop.cu
        src/optim/adam.cu
        src/optim/adam.cu
        include/slices.h
        src/slices.cpp
        include/elementwise_kernels.cuh
        include/losses/loss.h
)

set_target_properties(ArrC PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(ArrC PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(ArrC PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)