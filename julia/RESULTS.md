# oneAPI Julia Examples Results

This document contains the results of running the oneAPI Julia examples on an Intel GPU.

## Available Examples

1. Device Check (`check_device.jl`)
2. Matrix Multiplication (`matrix_multiply/`)
3. Block Cholesky & CT Reconstruction (`block_cholesky/`)
4. Conjugate Gradient Solver (`conjugate_gradient/`)
5. T-test Benchmark (`ttest/`)

## Detailed Results

### 1. Device Check
- Successfully detected Intel GPU
- Supported work-group sizes: 8, 16, 32, 64, 128, 256

### 2. Matrix Multiplication
Performance comparison between CPU and GPU implementations:
| Matrix Size | CPU Time (ms) | GPU Time (ms) | Relative Difference | Speedup |
|------------|---------------|---------------|-------------------|---------|
| 32x32      | 0.012        | 2.979         | 9.57e-8          | 0.004x  |
| 64x64      | 0.044        | 3.039         | 1.23e-7          | 0.014x  |
| 128x128    | 0.068        | 2.714         | 1.70e-7          | 0.025x  |
| 256x256    | 0.131        | 2.663         | 1.28e-7          | 0.049x  |

Note: The GPU performance includes data transfer overhead. For larger matrices, the GPU would show better performance.

### 3. Block Cholesky (CT Reconstruction)
- CPU Radon transform: 0.261 seconds
- GPU Radon transform: 9.623 seconds (including compilation time)
- Image reconstruction: 0.112 seconds
- Mean absolute difference between CPU and GPU: 0.0

### 4. Conjugate Gradient Solver
Test problem characteristics:
- Matrix size: 5x5
- Condition number: 2.280
- Relative error: 1.36e-7
- Residual norm: 1.65e-6

### 5. T-test Benchmark
CPU implementation scaling:
| Sample Size | Time (seconds) | T-statistic | Degrees of Freedom |
|------------|----------------|-------------|-------------------|
| 1,000      | 1.027e-6      | -12.900     | 1,993.150        |
| 10,000     | 4.915e-6      | -35.319     | 19,997.438       |
| 100,000    | 3.224e-5      | -111.294    | 199,991.170      |

## Implementation Status

| Example | Status | Implementation | Performance Notes |
|---------|--------|----------------|------------------|
| Device Check | ✓ | oneAPI | Verified GPU capabilities |
| Matrix Multiply | ✓ | CPU & GPU | Small matrices favor CPU |
| Block Cholesky | ✓ | CPU & GPU | GPU overhead significant |
| Conjugate Gradient | ✓ | CPU | High accuracy achieved |
| T-test | ✓ | CPU | Linear scaling with size |

## Running the Examples

Use the provided Makefile to run all examples:
```bash
make test_all
```

Or run individual examples:
```bash
make check_device
make run_matrix_multiply
make run_block_cholesky
make run_cg
make run_ttest
```

## Future Improvements

1. Optimize GPU implementations for larger datasets
2. Add GPU implementation for T-test
3. Improve memory management for Block Cholesky
4. Add GPU implementation for Conjugate Gradient
5. Implement batched operations for small matrices
