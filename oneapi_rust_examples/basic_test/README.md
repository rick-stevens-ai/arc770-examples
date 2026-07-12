# oneAPI SYCL Basic Test (Rust + C++)

This example demonstrates how to use Intel oneAPI SYCL from Rust via FFI bindings.

## Prerequisites

1. **Intel oneAPI Toolkit**: Install the Intel oneAPI Base Toolkit
   ```bash
   # Example for Ubuntu/Debian
   wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
   echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
   sudo apt update
   sudo apt install intel-basekit
   ```

2. **Environment Setup**: Source the oneAPI environment
   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```

3. **Intel MAX GPU Drivers**: Ensure Intel Data Center GPU MAX drivers are installed
   ```bash
   # Check if GPU is detected
   clinfo
   sycl-ls
   ```

## Building

```bash
# Make sure oneAPI environment is sourced
source /opt/intel/oneapi/setvars.sh

# Build the project
cargo build --release
```

## Running

```bash
cargo run --release
```

## What it does

1. **Device Discovery**: Lists all available SYCL devices (CPU, GPU, etc.)
2. **Device Selection**: Automatically selects an Intel GPU device (preferred for MAX GPUs)
3. **Vector Addition**: Performs a simple parallel vector addition: `C = A + B`
4. **Verification**: Verifies the results are correct

## Expected Output

```
oneAPI SYCL Basic Test
======================
Found 2 SYCL device(s)

Device 0: Intel(R) Data Center GPU Max 1550
  Vendor: Intel(R) Corporation
  Type: GPU
  Max work group size: 1024

Device 1: Intel(R) Xeon(R) CPU
  Vendor: Intel(R) Corporation
  Type: CPU/Other
  Max work group size: 8192

Using device 0 for computation
Input data A: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
Input data B: [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]
Using device: Intel(R) Data Center GPU Max 1550
Output data C: [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0, 39.0, 42.0, 45.0]

✓ Vector addition completed successfully!
✓ Results verified correctly!
```

## Architecture

- **Rust Main Application**: Handles device enumeration, data preparation, and result verification
- **C++ SYCL Kernels**: Implements the actual GPU computations using oneAPI SYCL
- **FFI Bridge**: Unsafe Rust extern "C" bindings to call C++ functions
- **Build System**: Custom build.rs script that compiles C++ with the DPC++ compiler

## Troubleshooting

1. **No devices found**: Make sure oneAPI environment is sourced and GPU drivers are installed
2. **Compilation errors**: Ensure DPC++ compiler is in PATH and SYCL headers are available
3. **Runtime errors**: Check that the Intel GPU runtime is properly configured