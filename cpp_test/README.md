# C++ SYCL Test

This directory contains a SYCL test program for the Intel Arc A770 GPU.

## Test Results
- Successfully detected and utilized Intel Arc A770 GPU
- Executed simple computation doubling vector elements
- Input: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
- Output: [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]

Note: Warnings about "ZE_LOADER_DEBUG_TRACE" can be disregarded as they don't affect program operation.

## Requirements
- Intel oneAPI Base Toolkit
- Intel GPU drivers
- C++ compiler with SYCL support

## Building
```bash
source /opt/intel/oneapi/setvars.sh
make
```

## Running
```bash
./sycl_test
```
