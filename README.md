# Intel Arc A770 oneAPI Examples

This repository contains example code demonstrating the use of Intel oneAPI and OpenCL with the Intel Arc A770 GPU.

## Test Results Summary

### C++ SYCL Test
- Successfully detected Intel Arc A770 GPU
- Executed vector multiplication kernel
- Input: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
- Output: [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
- Environment configured properly for SYCL applications

### Rust OpenCL Test
- Detected correct platform (Intel(R) OpenCL Graphics)
- Successfully identified Intel Arc A770 Graphics
- Executed vector multiplication kernel
- Input: [0.0, 1.0, 2.0, ..., 15.0]
- Output: [0.0, 2.0, 4.0, ..., 30.0]
- Confirmed OpenCL compute capability

## Directory Structure
- `cpp_test/`: SYCL C++ example
- `rust_test/`: Rust OpenCL example

## Requirements
- Intel oneAPI Base Toolkit
- Rust toolchain (for Rust examples)
- Intel GPU drivers
