# Apple Metal GPU Computing Examples for Rust

This directory contains Rust examples that demonstrate GPU computing using Apple's Metal Performance Shaders framework via the modern `objc2-metal` bindings. These examples are designed to run on macOS systems with Metal-capable GPUs, including Apple Silicon Macs and Intel Macs with dedicated graphics.

## Overview

These examples showcase GPU acceleration using Apple Metal through idiomatic Rust code:
- **Rust application code**: Handles setup, data management, and performance measurement using modern objc2 bindings
- **Metal compute shaders**: GPU kernels written in Metal Shading Language (MSL)
- **Native Metal integration**: Direct use of Metal APIs without intermediate C++ layers
- **Modern Rust ecosystem**: Uses objc2-metal (recommended) instead of deprecated metal-rs

## Examples

### 1. Basic Test (`basic_test/`)
Fundamental Metal operations including vector addition and parallel reduction.
- **What it does**: Vector addition and parallel sum reduction on GPU
- **Key features**: Device discovery, basic compute pipelines, threadgroup memory usage
- **Best for**: Learning Metal basics, verifying your setup works

### 2. Matrix Multiplication (`matrix_multiply/`)
High-performance dense matrix multiplication with multiple optimization strategies.
- **What it does**: Multiplies matrices using basic, tiled, and SIMD-optimized algorithms
- **Key features**: Memory coalescing, threadgroup local memory, performance comparison
- **Matrix sizes**: 256x256, 512x512, 1024x1024
- **Best for**: Understanding GPU memory patterns and optimization techniques

### 3. Monte Carlo Simulation (`monte_carlo/`)
Statistical computing using Monte Carlo methods with high-quality random number generation.
- **What it does**: Estimates π, performs numerical integration, demonstrates parallel reduction
- **Key features**: GPU random number generation (SimpleRNG + Philox), large-scale parallel processing
- **Sample counts**: 1M to 100M samples
- **Best for**: Stochastic algorithms and understanding GPU parallelism

### 4. Binomial Options Pricing (`binomial_options/`)
Financial computation for derivatives pricing using binomial and trinomial lattice models.
- **What it does**: Prices European options using binomial trees, compares with Black-Scholes
- **Key features**: Complex financial algorithms, accuracy validation, multiple pricing models
- **Problem sizes**: 1K to 131K options with up to 2048 time steps
- **Best for**: Real-world financial applications and algorithm convergence studies

## Prerequisites

### 1. macOS with Metal Support

**Minimum Requirements:**
- macOS 10.11 El Capitan or later
- Metal-capable GPU (most Macs since 2012)

**Recommended:**
- macOS 13 Ventura or later
- Apple Silicon (M1/M2/M3) or Intel Mac with dedicated GPU
- 8GB+ RAM for large problem sizes

**Verify Metal Support:**
```bash
system_profiler SPDisplaysDataType | grep Metal
```

### 2. Rust Toolchain

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Ensure you're on a recent version
rustup update
```

### 3. Xcode Command Line Tools

```bash
xcode-select --install
```

## Building and Running

### Quick Start - Test All Examples

```bash
# Clone and navigate to examples
cd metal_rust_examples

# Run all examples in sequence
./build_all.sh
```

### Individual Examples

```bash
# Navigate to specific example
cd basic_test  # or matrix_multiply, monte_carlo, binomial_options

# Build and run
cargo run --release
```

### Development Build (with debug info)

```bash
cargo run  # Debug build for development
```

## Expected Performance

### Apple Silicon M2 Pro (typical results)

- **Matrix Multiply (1024x1024)**: ~150 GFLOPS, 20-50x speedup vs CPU
- **Monte Carlo (100M samples)**: ~100M samples/sec, 15-30x speedup vs CPU
- **Binomial Options (131K options)**: ~1M options/sec, 25-60x speedup vs CPU
- **Memory Bandwidth**: ~200-400 GB/s effective

### Intel Mac with AMD GPU (typical results)

- **Matrix Multiply (1024x1024)**: ~80 GFLOPS, 10-25x speedup vs CPU
- **Monte Carlo (100M samples)**: ~50M samples/sec, 8-20x speedup vs CPU
- **Binomial Options (131K options)**: ~500K options/sec, 15-40x speedup vs CPU

**Performance varies based on:**
- GPU model and compute unit count
- Problem size (larger problems show better GPU utilization)
- Memory access patterns and cache efficiency
- Thermal state and power management settings

## Verification

### Check Your Metal Setup

```bash
# Verify Metal devices are available
cd basic_test
cargo run --release

# Expected output should show your GPU:
# Metal Device: Apple M2 Pro
#   Max threads per threadgroup: 1024
#   Supports unified memory: true
```

### Performance Profiling

**Using Xcode Instruments:**
1. Build with debug symbols: `cargo build --release`
2. Open Instruments and attach to your process
3. Use GPU profiling template to analyze kernel performance

**Using Activity Monitor:**
1. Run examples with `cargo run --release`
2. Monitor GPU usage in Activity Monitor → GPU tab
3. Should show near 100% GPU utilization for compute-heavy kernels

## Architecture Details

### Modern objc2-metal Integration

```rust
// Automatic memory management with Retained<T>
let device = MTLCreateSystemDefaultDevice()
    .context("No Metal device found")?;

// Type-safe buffer creation
let buffer = device.newBufferWithBytes_length_options(
    data.as_ptr() as *const std::ffi::c_void,
    byte_length,
    MTLResourceOptions::StorageModeShared,
);

// Compute pipeline with error handling
let pipeline = device
    .newComputePipelineStateWithFunction_error(&function)
    .context("Failed to create pipeline")?;
```

### Metal Shader Organization

Each example includes optimized Metal kernels:

```metal
#include <metal_stdlib>
using namespace metal;

kernel void compute_kernel(device const float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          uint gid [[thread_position_in_grid]]) {
    // GPU computation here
    output[gid] = process(input[gid]);
}
```

### Build System

The examples use standard Rust tooling:
- **Cargo.toml**: Dependencies on objc2-metal ecosystem
- **No build.rs needed**: Metal shaders compiled at runtime
- **include_str!()**: Embeds shaders directly in binary for portability

## Optimization Techniques Used

### Memory Access Patterns
- **Coalesced access**: Ensure contiguous GPU memory reads/writes
- **Shared memory**: Use threadgroup memory for data reuse
- **Buffer management**: Minimize allocation overhead with proper buffer sizing

### Compute Organization
- **Threadgroup sizing**: Match GPU warp/wavefront size (typically 32)
- **Occupancy optimization**: Balance register usage vs parallelism
- **Kernel fusion**: Combine operations to reduce memory traffic

### Algorithm-Specific Optimizations
- **Matrix multiply**: Tiling for cache efficiency, SIMD instructions
- **Monte Carlo**: High-quality RNG (Philox), parallel reduction
- **Options pricing**: Shared memory for binomial lattices, convergence analysis

## Troubleshooting

### Common Issues

**1. "No Metal device found"**
```bash
# Check if your Mac supports Metal
system_profiler SPDisplaysDataType | grep Metal

# Ensure you're running on macOS 10.11+
sw_vers
```

**2. Compilation errors with objc2-metal**
```bash
# Update Rust toolchain
rustup update

# Clean and rebuild
cargo clean && cargo build --release
```

**3. Poor GPU performance**
```bash
# Check thermal throttling
sudo powermetrics -i 1 -n 5 | grep -E "(GPU|CPU) die temperature"

# Monitor GPU frequency
system_profiler SPDisplaysDataType | grep "GPU Core Speed"
```

**4. Runtime panics or crashes**
```bash
# Run with debug info
RUST_BACKTRACE=1 cargo run

# Check for Metal validation errors
export MTL_DEBUG_LAYER=1
cargo run
```

### Performance Debugging

**GPU underutilization:**
- Increase problem size for better parallelism
- Check threadgroup size alignment with hardware
- Profile with Instruments to identify bottlenecks

**Memory bandwidth bottlenecks:**
- Analyze access patterns in Metal shaders
- Consider data layout transformations
- Use shared/threadgroup memory where appropriate

**Accuracy issues:**
- Check floating-point precision requirements
- Verify algorithm convergence parameters
- Compare against CPU reference implementations

## Best Practices

### GPU Algorithm Design
1. **Maximize parallelism**: Design algorithms with thousands of independent threads
2. **Minimize divergence**: Avoid conditional branches that vary between threads
3. **Optimize memory**: Use coalesced access patterns and shared memory effectively
4. **Consider precision**: Balance float32 vs float16 based on accuracy requirements

### Rust Integration
1. **Use objc2-metal**: Prefer modern objc2 bindings over deprecated metal-rs
2. **Handle errors properly**: Metal operations can fail; use proper error handling
3. **Memory safety**: Be careful with unsafe operations when interfacing with Metal
4. **Resource management**: Let objc2's Retained<T> handle Metal object lifecycle

## Extensions and Modifications

### Adding New Kernels

1. Add Metal shader to `shaders/` directory
2. Load shader using `include_str!()` and compile with `newLibraryWithSource`
3. Create compute pipeline and configure buffers
4. Dispatch work with appropriate threadgroup sizing

### Optimization Opportunities

- **Metal Performance Shaders**: Use MPS for optimized BLAS operations
- **Indirect command buffers**: For GPU-driven rendering/compute
- **Raytracing**: Use Metal raytracing for advanced algorithms
- **Machine learning**: Integrate with Metal Performance Shaders ML

## References

- [Metal Programming Guide](https://developer.apple.com/metal/)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [objc2 Documentation](https://docs.rs/objc2/)
- [objc2-metal API Reference](https://docs.rs/objc2-metal/)
- [Metal Best Practices Guide](https://developer.apple.com/documentation/metal/best_practices_for_metal_apps)

## License

These examples are provided under the MIT License. See the main repository LICENSE file for details.