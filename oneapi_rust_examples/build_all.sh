#!/bin/bash

# Build script for all oneAPI SYCL Rust examples
# Usage: ./build_all.sh [clean]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}oneAPI SYCL Rust Examples Build Script${NC}"
echo "========================================"

# Check if oneAPI environment is sourced
if [ -z "$DPCPP_ROOT" ] && [ -z "$ONEAPI_ROOT" ]; then
    echo -e "${RED}Error: oneAPI environment not detected${NC}"
    echo "Please run: source /opt/intel/oneapi/setvars.sh"
    exit 1
fi

# Check for DPC++ compiler
if ! command -v dpcpp &> /dev/null; then
    echo -e "${RED}Error: DPC++ compiler not found in PATH${NC}"
    echo "Please ensure oneAPI DPC++/C++ Compiler is installed and sourced"
    exit 1
fi

echo -e "${GREEN}✓ oneAPI environment detected${NC}"
echo "DPC++ version: $(dpcpp --version | head -n1)"

# Examples to build
EXAMPLES=("basic_test" "matrix_multiply" "monte_carlo" "binomial_options")
FAILED_BUILDS=()
SUCCESSFUL_BUILDS=()

# Clean option
if [ "$1" == "clean" ]; then
    echo -e "${YELLOW}Cleaning all examples...${NC}"
    for example in "${EXAMPLES[@]}"; do
        if [ -d "$example" ]; then
            echo "Cleaning $example..."
            cd "$example"
            cargo clean
            cd ..
        fi
    done
    echo -e "${GREEN}✓ All examples cleaned${NC}"
    exit 0
fi

# Build each example
for example in "${EXAMPLES[@]}"; do
    echo -e "\n${BLUE}Building $example...${NC}"
    echo "----------------------------------------"

    if [ ! -d "$example" ]; then
        echo -e "${RED}✗ Directory $example not found${NC}"
        FAILED_BUILDS+=("$example")
        continue
    fi

    cd "$example"

    # Build the example
    if cargo build --release; then
        echo -e "${GREEN}✓ $example built successfully${NC}"
        SUCCESSFUL_BUILDS+=("$example")

        # Quick test run (just device detection)
        echo "Testing device detection..."
        if timeout 30s cargo run --release > /tmp/${example}_output.log 2>&1; then
            if grep -q "Found.*SYCL device" /tmp/${example}_output.log; then
                echo -e "${GREEN}✓ $example device detection successful${NC}"
            else
                echo -e "${YELLOW}⚠ $example ran but no SYCL devices detected${NC}"
            fi
        else
            echo -e "${YELLOW}⚠ $example build succeeded but runtime test failed${NC}"
        fi

    else
        echo -e "${RED}✗ $example build failed${NC}"
        FAILED_BUILDS+=("$example")
    fi

    cd ..
done

# Summary
echo -e "\n${BLUE}Build Summary${NC}"
echo "============="

if [ ${#SUCCESSFUL_BUILDS[@]} -gt 0 ]; then
    echo -e "${GREEN}Successful builds (${#SUCCESSFUL_BUILDS[@]}):${NC}"
    for example in "${SUCCESSFUL_BUILDS[@]}"; do
        echo "  ✓ $example"
    done
fi

if [ ${#FAILED_BUILDS[@]} -gt 0 ]; then
    echo -e "${RED}Failed builds (${#FAILED_BUILDS[@]}):${NC}"
    for example in "${FAILED_BUILDS[@]}"; do
        echo "  ✗ $example"
    done
fi

# Check for SYCL devices
echo -e "\n${BLUE}System Information${NC}"
echo "=================="
echo "SYCL devices:"
if command -v sycl-ls &> /dev/null; then
    sycl-ls 2>/dev/null || echo "No SYCL devices found or error querying devices"
else
    echo "sycl-ls command not found"
fi

# Performance recommendations
echo -e "\n${BLUE}Performance Tips${NC}"
echo "================"
echo "• For best performance, run examples individually with --release flag"
echo "• Monitor GPU usage with: intel_gpu_top (if available)"
echo "• Check GPU frequency: cat /sys/class/drm/card*/gt/gt0/rps_cur_freq_mhz"
echo "• For MAX GPUs, ensure proper cooling and power delivery"

# Exit with appropriate code
if [ ${#FAILED_BUILDS[@]} -gt 0 ]; then
    exit 1
else
    echo -e "\n${GREEN}All examples built successfully!${NC}"
    exit 0
fi