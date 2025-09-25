# Test Results - Intel Arc A770 oneAPI Examples

Test Date: March 24, 2024
Hardware: Intel Arc A770 Graphics

## Test Results Summary

### 1. C++ Test (cpp_test)
- Successfully ran the pre-compiled GPU test
- Demonstrated basic SYCL functionality on the Arc A770
- Test completed successfully

### 2. Julia Test (julia_test)
- Successfully ran the Julia oneAPI test using oneAPI.jl
- Input: `Float32[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]`
- Output: `Float32[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0]`
- GPU Device detected: ZeDevice(GPU, vendor 0x8086, device 0x56a0)

### 3. Rust Test (rust_test)
- Successfully built and ran using Cargo
- Used OpenCL 3.0 for GPU interaction
- Detected Platforms:
  - Intel(R) OpenCL
  - Intel(R) OpenCL Graphics
- GPU Device: Intel(R) Arc(TM) A770 Graphics
- Input: `[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]`
- Output: `[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]`

## Conclusion
All three implementations (C++/SYCL, Julia/oneAPI, and Rust/OpenCL) successfully demonstrated GPU computation on the Intel Arc A770. Each test performed array multiplication (doubling elements) as a verification of GPU functionality. The tests showed consistent results across different programming languages and GPU computing frameworks.

The only minor issue encountered was a warning in the Rust code about an unused variable, which did not affect functionality.
