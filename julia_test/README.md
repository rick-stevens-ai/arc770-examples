# oneAPI.jl Test with Intel Arc A770

This project demonstrates basic GPU computation using oneAPI.jl with the Intel Arc A770 GPU.

## Test Results

The test program successfully demonstrated the following capabilities:

### 1. Device Detection
- Successfully detected the Intel Arc A770 GPU
- Device info: ZeDevice(GPU, vendor 0x8086, device 0x56a0)

### 2. Computation Test
- Input: Float32 array [1.0, 2.0, ..., 16.0]
- Successfully executed the kernel that doubles each element
- Output: Float32 array [2.0, 4.0, ..., 32.0]

### 3. Key Components
The test confirmed working functionality of:
- oneArray for GPU memory allocation
- Kernel function with oneAPI.get_global_id()
- @oneapi macro for kernel launch
- Proper synchronization
- Successful data transfer back to host

## Conclusions
The test confirms that:
1. The oneAPI.jl package is working correctly
2. The Intel Arc A770 GPU is properly recognized
3. Basic GPGPU computations can be performed

This basic example can serve as a foundation for more complex GPU computations using oneAPI.jl with the Intel Arc A770.

## Project Structure
- `src/OneAPITest.jl`: Main module containing the test implementation
- `run_test.jl`: Script to execute the test

## Dependencies
- Julia
- oneAPI.jl package
