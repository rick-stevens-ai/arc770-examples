# Intel Arc GPU OpenCL Examples in Rust

This collection demonstrates various OpenCL applications optimized for Intel Arc GPUs, implemented in Rust.
Each example showcases different aspects of GPU computing using OpenCL.

## Prerequisites

- Rust toolchain (install via [rustup](https://rustup.rs/))
- Intel GPU drivers with OpenCL support
- Intel oneAPI Base Toolkit
- Make (for building all examples)

## Examples

1. **Matrix Multiplication**
   - Demonstrates basic matrix operations using OpenCL
   - Compares performance between CPU and GPU implementations
   - Shows device discovery and selection

2. **Monte Carlo Pi Calculation**
   - Calculates π using Monte Carlo method
   - Demonstrates random number generation on GPU
   - Includes CPU validation of results

3. **Binomial Options Pricing**
   - Financial computation example
   - Processes multiple options in parallel
   - Shows high-performance numerical calculations

## Building the Examples

You can build all examples using make:

```bash
# Build all examples
make all

# Clean build artifacts
make clean

# Build and run all examples
make run

# Run tests
make test
```

### Building Individual Examples

Each example can be built separately using Cargo:

```bash
# Navigate to example directory
cd matrix_multiply
# Build with release optimizations
cargo build --release
```

## Running the Examples

### Using Make

```bash
# Run all examples
make run
```

### Running Individual Examples

```bash
# Matrix Multiplication
cd matrix_multiply
cargo run --release

# Monte Carlo Pi Calculation
cd monte_carlo
cargo run --release

# Binomial Options
cd binomial_options
cargo run --release
```

## Performance Results

See [RESULTS.md](RESULTS.md) for detailed performance measurements and analysis.

## System Requirements

- Intel Arc GPU (tested on Intel Arc A770)
- Linux operating system with recent kernel (tested on latest Ubuntu LTS)
- Intel GPU drivers installed and configured
- OpenCL 3.0 support
- Rust 1.70 or newer

## Troubleshooting

1. **OpenCL Platform Not Found**
   - Ensure Intel GPU drivers are properly installed
   - Verify OpenCL installation: `clinfo`
   - Check device visibility: `lspci | grep VGA`

2. **Build Failures**
   - Update Rust toolchain: `rustup update`
   - Verify OpenCL development files are installed
   - Check system OpenCL installation

3. **Performance Issues**
   - Ensure GPU is not being used by display server
   - Check thermal throttling
   - Verify OpenCL device selection in examples

## Contributing

Contributions are welcome! Please feel free to submit pull requests with:
- New examples
- Performance improvements
- Bug fixes
- Documentation improvements

## License

This project is licensed under the MIT License - see the LICENSE file for details.
