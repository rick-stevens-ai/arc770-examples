# Quick Start Guide

## Prerequisites
1. Install Intel oneAPI Base Toolkit 2025.1
2. Install Intel GPU drivers (minimum version 1.3.29138)
3. Ensure Intel Arc A770 GPU is recognized by the system

## Installation
```bash
# Clone the repository
git clone <repository_url>
cd arc770_oneapi_examples

# Build all examples
make clean && make build_all

# Run all examples
make run_all
```

## Individual Examples
Each example can be built and run separately:
```bash
cd src/<example_name>
make clean && make
./<example_name>
```

## Verification
Each example includes built-in verification:
- Matrix multiplication checks result accuracy
- Option pricing computes L1 norm
- Image processing verifies reconstruction

## Troubleshooting
1. If build fails, ensure oneAPI environment is properly set up:
   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```
2. Check GPU detection:
   ```bash
   sycl-ls
   ```
3. Verify GPU driver version:
   ```bash
   sudo lspci -v | grep -A 10 VGA
   ```
