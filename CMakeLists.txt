cmake_minimum_required(VERSION 3.18)
project(ArrC CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

add_executable(ArrC
        # Main entry point
        main.cu

        # Core source files
        src/optim/sgd.cu
        src/optim/rmsprop.cu
        src/optim/adam.cu
        src/slices.cpp
        src/utils.cpp

        # Headers (for IDE visibility)
        include/ndarray.cuh
        include/tensor.h
        include/exceptions.h
        include/slices.h
        include/elementwise_kernels.cuh
        include/utils.h
        include/functions/base.h
        include/functions/arithmetic.h
        include/optim/optimizer.h
        include/optim/sgd.cuh
        include/optim/adam.cuh
        include/optim/rmsprop.cuh
        include/optim/kernels.cuh
        include/nn/losses.h
)

set_target_properties(ArrC PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(ArrC PRIVATE CUDA::cudart)

target_include_directories(ArrC PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)
