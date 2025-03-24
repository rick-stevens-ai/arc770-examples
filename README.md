# Intel Arc A770 GPU oneAPI Examples

This repository contains optimized oneAPI examples specifically tuned for the Intel Arc A770 GPU.
All examples have been modified to use single precision floating-point operations and modern SYCL features.

## Examples Included

1. Black-Scholes Option Pricing
   - Single precision implementation
   - Performance: ~41.28 GOptions/s
   - Modern SYCL features with get_multi_ptr()
   - Optimized for Arc A770

2. Binomial Option Pricing
   - Single precision implementation
   - Processes 8.39M options with 2048 time steps
   - Performance: 7.31 seconds execution time
   - Stack-based array allocation in device code

3. Matrix Multiplication
   - Single precision GEMM operations
   - Performance: 16.0685 TF/s
   - Optimized for Arc A770 architecture

4. Computed Tomography
   - Single precision DFT operations
   - Modern SYCL implementation
   - Optimized image processing

## Requirements

- Intel Arc A770 GPU
- Intel oneAPI Base Toolkit 2025.1
- Ubuntu 24.04 (Noble) or compatible Linux distribution
- Intel GPU drivers (minimum version 1.3.29138)

## Building and Running

Each example can be built using the provided Makefile:

```bash
cd src/<example_name>
make clean && make
./<example_name>
```

Or use the master Makefile to build all examples:

```bash
make clean && make build_all
make run_all
```

## Performance Results

All performance measurements were taken on an Intel Arc A770 GPU:

1. Black-Scholes:
   - 41.28 GOptions/s
   - L1 norm: 6.96e-08
   - Test verification: PASSED

2. Binomial:
   - 7.31 seconds for 8.39M options
   - 2048 time steps
   - Test verification: PASSED

3. Matrix Multiplication:
   - 16.0685 TF/s (single precision)
   - Test verification: PASSED

4. Computed Tomography:
   - Successfully processes 400x400 images
   - Test verification: PASSED

## Updates and Optimizations

1. Replaced deprecated SYCL features:
   - Removed intel::reqd_sub_group_size
   - Updated to sycl::reqd_sub_group_size
   - Replaced get_pointer() with get_multi_ptr()

2. Single Precision Optimizations:
   - All examples modified for single precision
   - Optimized for Arc A770 GPU capabilities
   - Verified numerical accuracy

3. Memory Management:
   - Stack-based allocation in device code
   - Proper buffer management
   - Optimized data transfers

## License

This project is licensed under the MIT License - see the LICENSE file for details.
