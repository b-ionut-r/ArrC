cmake_minimum_required(VERSION 4.0)
project(ArrC CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(ArrC
        kernels.cu
        slices.cpp
        ndarray.cuh
        main.cu)

set_target_properties(ArrC PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
