# Rust OpenCL Test

This directory contains an OpenCL test program written in Rust for the Intel Arc A770 GPU.

## Test Results
- Successfully detected platforms:
  * Intel(R) OpenCL
  * Intel(R) OpenCL Graphics (GPU platform)
- Identified Intel Arc A770 Graphics device
- Executed vector multiplication kernel:
  * Input: [0.0, 1.0, 2.0, ..., 15.0]
  * Output: [0.0, 2.0, 4.0, ..., 30.0]

## Requirements
- Rust toolchain
- Intel oneAPI Base Toolkit
- Intel GPU drivers
- OpenCL runtime

## Dependencies
```toml
[dependencies]
opencl3 = "0.9"
```

## Building
```bash
source /opt/intel/oneapi/setvars.sh
cargo build
```

## Running
```bash
cargo run
```
