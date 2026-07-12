# oneAPI SYCL Examples for Rust

This directory contains hybrid Rust+C++ examples that demonstrate GPU computing using Intel oneAPI SYCL. These examples are designed to run on Intel Data Center GPU Max series and other SYCL-compatible devices.

## Overview

Since direct Rust SYCL bindings are not officially available, these examples use a hybrid approach:
- **Rust application code**: Handles data preparation, device enumeration, and result verification
- **C++ SYCL kernels**: Implements GPU computations using Intel oneAPI SYCL
- **FFI bridge**: Unsafe Rust extern "C" bindings to call C++ functions
- **Custom build system**: build.rs scripts compile C++ code with DPC++ compiler

## Examples

### 1. Basic Test (`basic_test/`)
Simple vector addition demonstrating SYCL device discovery and basic GPU computation.
- **What it does**: Adds two vectors element-wise on GPU
- **Key features**: Device enumeration, basic SYCL operations, result verification
- **Best for**: Learning SYCL basics, testing your setup

### 2. Matrix Multiplication (`matrix_multiply/`)
High-performance dense matrix multiplication with optimization techniques.
- **What it does**: Multiplies two matrices using GPU acceleration
- **Key features**: Basic and tiled algorithms, performance comparison with CPU
- **Matrix sizes**: 256x256, 512x512, 1024x1024
- **Best for**: Understanding parallel algorithms and optimization

### 3. Monte Carlo Pi Estimation (`monte_carlo/`)
Statistical computation using random sampling to estimate π.
- **What it does**: Estimates π using Monte Carlo method with millions of samples
- **Key features**: Parallel random number generation, reduction operations
- **Sample counts**: 1M, 10M, 100M samples
- **Best for**: Stochastic algorithms and large-scale parallel computing

### 4. Binomial Options Pricing (`binomial_options/`)
Financial computation for European option pricing using binomial trees.
- **What it does**: Prices stock options using binomial lattice model
- **Key features**: Complex financial algorithms, accuracy vs Black-Scholes comparison
- **Problem sizes**: 1K to 262K options with 256-2048 time steps
- **Best for**: Real-world application of GPU computing in finance

## Prerequisites

### 1. Intel oneAPI Base Toolkit

**Ubuntu/Debian:**
```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-basekit
```

**Red Hat/CentOS:**
```bash
sudo dnf install -y dnf-plugins-core
sudo dnf config-manager --add-repo https://yum.repos.intel.com/oneapi
sudo rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo dnf install intel-basekit
```

### 2. Intel GPU Drivers (for Data Center MAX GPUs)

```bash
# Add Intel GPU repository
wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | sudo apt-key add -
sudo apt-add-repository 'deb [arch=amd64] https://repositories.intel.com/graphics/ubuntu focal main'

# Install drivers
sudo apt update
sudo apt install intel-opencl-icd intel-level-zero-gpu level-zero intel-media-va-driver-non-free libmfx1
```

### 3. Environment Setup

**Every time before building/running:**
```bash
source /opt/intel/oneapi/setvars.sh
```

**Optional - Add to your shell profile:**
```bash
echo "source /opt/intel/oneapi/setvars.sh" >> ~/.bashrc
```

## Building and Running

### Quick Start - Run All Examples

```bash
# Source oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Build and run each example
cd basic_test && cargo run --release && cd ..
cd matrix_multiply && cargo run --release && cd ..
cd monte_carlo && cargo run --release && cd ..
cd binomial_options && cargo run --release && cd ..
```

### Individual Examples

```bash
# Source environment
source /opt/intel/oneapi/setvars.sh

# Navigate to example directory
cd basic_test  # or matrix_multiply, monte_carlo, binomial_options

# Build (first time may take longer due to C++ compilation)
cargo build --release

# Run
cargo run --release
```

## Verification and Testing

### Check Your Setup

```bash
# Verify oneAPI installation
dpcpp --version

# Check for Intel GPUs
sycl-ls
clinfo | grep Intel

# Verify GPU is detected
cd basic_test
cargo run --release
```

### Expected Output

For Intel Data Center GPU Max, you should see output like:
```
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
```

## Performance Expectations

### Intel Data Center GPU Max 1550 (typical results)

- **Matrix Multiply (1024x1024)**: ~10-50x speedup vs CPU
- **Monte Carlo (100M samples)**: ~20-100x speedup vs CPU
- **Binomial Options (262K options)**: ~30-80x speedup vs CPU
- **Memory bandwidth**: ~1000 GB/s theoretical

### Performance varies based on:
- Problem size (larger problems generally show better GPU speedup)
- Memory access patterns (coalesced access is crucial)
- Compute vs memory bound workloads
- Driver version and system configuration

## Troubleshooting

### Common Issues

**1. No SYCL devices found**
```bash
# Check oneAPI environment
source /opt/intel/oneapi/setvars.sh
echo $DPCPP_ROOT

# Verify GPU drivers
sycl-ls
```

**2. Compilation errors**
```bash
# Check DPC++ compiler
which dpcpp
dpcpp --version

# Clean and rebuild
cargo clean
cargo build --release
```

**3. Runtime errors**
```bash
# Check GPU accessibility
sudo dmesg | grep i915
lspci | grep VGA

# Verify permissions
groups $USER  # should include video/render groups
```

**4. Performance issues**
- Ensure GPU frequency scaling is disabled
- Check for thermal throttling
- Verify PCIe link speed
- Monitor GPU utilization with `intel_gpu_top`

### Debug Mode

For more verbose output:
```bash
export SYCL_RT_WARNING_LEVEL=1
export INTEL_DEVICE_SELECTOR=gpu
cargo run
```

## Architecture Details

### Hybrid Rust+C++ Design

Each example follows this pattern:

```
Rust Application Layer
├── Data preparation and validation
├── Device selection and management
├── Performance measurement
└── Result verification

FFI Boundary (unsafe extern "C")
├── Type marshalling (Rust ↔ C++)
├── Memory management
└── Error handling

C++ SYCL Kernel Layer
├── SYCL queue and buffer management
├── Kernel implementation
├── GPU memory optimization
└── Error reporting
```

### Build Process

1. **build.rs** detects oneAPI installation and DPC++ compiler
2. **cc crate** compiles C++ SYCL code with appropriate flags
3. **Static linking** combines Rust application with C++ kernels
4. **Runtime** dynamically loads SYCL runtime and GPU drivers

## Extending the Examples

### Adding New Kernels

1. Add C++ function to appropriate `cpp/*.cpp` file
2. Add corresponding `extern "C"` declaration in Rust
3. Update build.rs if needed for additional includes/flags
4. Implement Rust wrapper with proper error handling

### Optimization Techniques Used

- **Coalesced memory access**: Ensure contiguous GPU memory reads/writes
- **Local memory usage**: Share data within work groups
- **Work group size tuning**: Match GPU architecture characteristics
- **Async execution**: Overlap computation and data transfer
- **Buffer reuse**: Minimize allocation overhead

## License

These examples are provided under the MIT License. See the main repository LICENSE file for details.

## Support and Feedback

For issues with:
- **oneAPI/SYCL**: Intel oneAPI forums and documentation
- **Intel GPU drivers**: Intel Graphics support
- **These examples**: File issues on the repository

## References

- [Intel oneAPI Documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
- [SYCL Specification](https://www.khronos.org/sycl/)
- [Intel Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series.html)
- [DPC++ Compiler Guide](https://intel.github.io/llvm-docs/GetStartedGuide.html)