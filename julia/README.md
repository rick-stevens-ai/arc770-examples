# Julia oneAPI Examples

This directory contains two examples demonstrating the use of oneAPI.jl for GPU acceleration:

## 1. Conjugate Gradient Solver

Located in `conjugate_gradient/`, this example implements:
- CPU and GPU versions of the conjugate gradient method for solving linear systems
- Benchmarking of both implementations for various matrix sizes
- Verification of solution accuracy
- Performance comparison between CPU and GPU implementations

## 2. Student's T-Test

Located in `ttest/`, this example demonstrates:
- Statistical computation acceleration using GPUs
- Implementation of Student's t-test for comparing two sample means
- Benchmarking with different sample sizes
- Verification of statistical accuracy

## Running the Examples

To run all examples:
```bash
make all
```

To run individual examples:
```bash
make run_cg    # Run conjugate gradient example
make run_ttest # Run t-test example
```

To clean built files:
```bash
make clean
```

## Requirements

- Julia 1.6 or higher
- oneAPI.jl
- Intel GPU with proper drivers installed
