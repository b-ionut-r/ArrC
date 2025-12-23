# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

**Build the project:**
```bash
mkdir -p build && cd build
cmake ..
make
```

**Run the executable:**
```bash
./ArrC
```

Note: This project requires CMake 4.0+ and CUDA compiler support. The CMakeLists.txt configures CUDA separable compilation and C++20 standard.

## Architecture Overview

ArrC is a CUDA-based deep learning framework implementing automatic differentiation (autograd) with the following key components:

### Core Data Structures

**NDArray (`include/ndarray.cuh`)**:
- Template-based multidimensional array with GPU memory management
- Supports strided views, slicing, and broadcasting
- Element-wise operations via CUDA kernels (`include/elementwise_kernels.cuh`)
- Uses unified memory (cudaMallocManaged) for seamless CPU/GPU access
- Tracks memory usage with static `totalAllocatedMemory`

**Tensor (`include/tensor.h`)**:
- Wrapper around NDArray with automatic differentiation capabilities
- Maintains gradient information and computational graph via `Function* gradFn`
- Implements topological sort for backward pass in `backward()` method
- Uses variant types for mixed-precision support (half, float, double, bfloat16)

### Automatic Differentiation System

**Function Base Class (`include/functions/base.h`)**:
- Abstract base for all differentiable operations
- Stores parent tensor references in computational graph
- Implements forward/backward pattern for gradient computation

**Arithmetic Operations (`include/functions/arithmetic.h`)**:
- Concrete Function implementations: AddFunction, MulFunction, SubFunction, DivFunction
- Factory functions in `functions` namespace for creating operations
- Currently has simplified backward implementations (TODO: proper gradient computation with stored inputs)

### Optimization Framework

**Optimizer Base (`include/optim/optimizer.h`)**:
- Abstract Strategy pattern for different optimization algorithms
- Supports SGD (`include/optim/sgd.cuh`), Adam (`include/optim/adam.cuh`), RMSprop (`include/optim/rmsprop.cuh`)
- Uses variant types to handle different tensor precisions
- Includes weight decay and learning rate management

**Neural Network Components**:
- Loss functions framework started in `include/nn/losses.h` (currently incomplete)

### Memory Management

- NDArray manages GPU memory allocation/deallocation
- Tensor destructor handles gradient memory cleanup
- Backward pass includes graph cleanup (nodes deleted after use unless `retainGraph=true`)
- `preserveAncestors` parameter controls how many recent nodes to keep during backward pass

### Key Design Patterns

1. **Template-based generic programming**: NDArray and Tensor are templated for different data types
2. **Strategy pattern**: Optimizer abstract base class with concrete implementations
3. **Variant types**: Support for mixed precision via std::variant
4. **RAII**: Automatic memory management in destructors
5. **Computational graph**: Function-based automatic differentiation system

### File Organization

- `include/`: All headers (.h/.cuh files)
  - Core: `ndarray.cuh`, `tensor.h`, `utils.h`, `exceptions.h`
  - Functions: `functions/base.h`, `functions/arithmetic.h`
  - Optimizers: `optim/optimizer.h`, `optim/[algorithm].cuh`
  - Neural nets: `nn/losses.h`
  - CUDA: `elementwise_kernels.cuh`, `optim/kernels.cuh`
- `src/`: Implementation files (.cpp/.cu)
  - Utilities: `utils.cpp`, `slices.cpp`
  - Optimizers: `optim/[algorithm].cu`

### Development Notes

- The project uses C++20 features and requires CUDA-capable GPU
- All template definitions are in headers due to CUDA compilation requirements
- Memory leaks are carefully managed with proper RAII and explicit cleanup
- Backward pass automatically manages computational graph lifecycle