#!/bin/bash

# Build script for all Apple Metal Rust examples
# Usage: ./build_all.sh [clean]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Apple Metal Rust Examples Build Script${NC}"
echo "========================================"

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Error: These examples require macOS with Metal support${NC}"
    echo "Current OS: $OSTYPE"
    exit 1
fi

# Check macOS version (Metal requires 10.11+)
MACOS_VERSION=$(sw_vers -productVersion)
MACOS_MAJOR=$(echo $MACOS_VERSION | cut -d. -f1)
MACOS_MINOR=$(echo $MACOS_VERSION | cut -d. -f2)

if [[ $MACOS_MAJOR -lt 10 ]] || [[ $MACOS_MAJOR -eq 10 && $MACOS_MINOR -lt 11 ]]; then
    echo -e "${RED}Error: Metal requires macOS 10.11 or later${NC}"
    echo "Current version: $MACOS_VERSION"
    exit 1
fi

echo -e "${GREEN}✓ macOS $MACOS_VERSION detected (Metal supported)${NC}"

# Check for Metal support
if ! system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Metal"; then
    echo -e "${YELLOW}⚠ Warning: Metal support not detected in system profile${NC}"
    echo "This may still work on newer systems where Metal is always available"
fi

# Check Rust installation
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}Error: Rust/Cargo not found${NC}"
    echo "Please install Rust: https://rustup.rs/"
    exit 1
fi

echo -e "${GREEN}✓ Rust toolchain detected${NC}"
echo "Rust version: $(rustc --version)"

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
        echo "Testing Metal device detection..."
        if timeout 30s cargo run --release > /tmp/${example}_output.log 2>&1; then
            if grep -q "Metal Device:" /tmp/${example}_output.log; then
                echo -e "${GREEN}✓ $example Metal device detection successful${NC}"
                DEVICE_NAME=$(grep "Metal Device:" /tmp/${example}_output.log | head -n1)
                echo "  $DEVICE_NAME"
            else
                echo -e "${YELLOW}⚠ $example ran but no Metal device detected${NC}"
            fi
        else
            echo -e "${YELLOW}⚠ $example build succeeded but runtime test failed${NC}"
            # Show last few lines of output for debugging
            echo "Last few lines of output:"
            tail -n 5 /tmp/${example}_output.log | sed 's/^/  /'
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

# System information
echo -e "\n${BLUE}System Information${NC}"
echo "=================="
echo "macOS version: $MACOS_VERSION"
echo "Rust version: $(rustc --version)"

# Metal device information
echo -e "\nMetal GPU Information:"
if command -v system_profiler &> /dev/null; then
    system_profiler SPDisplaysDataType 2>/dev/null | grep -A5 -B1 "Metal" | grep -E "(Chipset Model|Metal|VRAM)" | head -10 | sed 's/^/  /'
else
    echo "  system_profiler not available"
fi

# Performance tips
echo -e "\n${BLUE}Performance Tips${NC}"
echo "================"
echo "• For best performance, ensure your Mac isn't thermally throttled"
echo "• Close other GPU-intensive applications before running examples"
echo "• Monitor GPU usage in Activity Monitor while running examples"
echo "• Use 'cargo run --release' for performance testing (not debug builds)"

if [[ $(sysctl -n hw.optional.arm64 2>/dev/null) == "1" ]]; then
    echo "• Apple Silicon detected: unified memory provides excellent GPU bandwidth"
else
    echo "• Intel Mac detected: discrete GPU will show best performance"
fi

# Individual run instructions
echo -e "\n${BLUE}Individual Example Usage${NC}"
echo "========================"
echo "To run examples individually:"
for example in "${SUCCESSFUL_BUILDS[@]}"; do
    echo "  cd $example && cargo run --release && cd .."
done

# Exit with appropriate code
if [ ${#FAILED_BUILDS[@]} -gt 0 ]; then
    echo -e "\n${YELLOW}Some builds failed. Check error messages above.${NC}"
    exit 1
else
    echo -e "\n${GREEN}All examples built successfully!${NC}"
    echo -e "${GREEN}Your system is ready for Metal GPU computing with Rust.${NC}"
    exit 0
fi