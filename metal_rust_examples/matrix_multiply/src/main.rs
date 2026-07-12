use anyhow::{Context, Result};
use objc2::rc::Retained;
use objc2_foundation::NSString;
use objc2_metal::*;
use objc2_metal_performance_shaders::*;
use std::time::Instant;

fn create_metal_device() -> Result<Retained<ProtocolObject<dyn MTLDevice>>> {
    let device = MTLCreateSystemDefaultDevice()
        .context("No Metal device found. Metal requires macOS 10.11+ or iOS 8.0+")?;

    println!("Metal Device: {}", device.name());
    println!("  Max threads per threadgroup: {}", device.maxThreadsPerThreadgroup().width);
    println!("  Supports unified memory: {}", device.hasUnifiedMemory());
    println!("  Recommended working set size: {:.2} MB",
             device.recommendedMaxWorkingSetSize() as f64 / (1024.0 * 1024.0));

    Ok(device)
}

fn compile_shaders(device: &ProtocolObject<dyn MTLDevice>) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let shader_source = include_str!("../shaders/matrix_multiply.metal");
    let source_string = NSString::from_str(shader_source);

    let library = device
        .newLibraryWithSource_options_error(&source_string, None)
        .context("Failed to compile Metal shaders")?;

    println!("✓ Metal shaders compiled successfully");
    Ok(library)
}

fn create_test_matrices(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = Vec::with_capacity(n * n);
    let mut b = Vec::with_capacity(n * n);

    // Fill matrices with test data
    for i in 0..n {
        for j in 0..n {
            a.push((i + j) as f32 * 0.01);
            b.push((i * n + j) as f32 * 0.001);
        }
    }

    (a, b)
}

fn verify_result(a: &[f32], b: &[f32], c: &[f32], n: usize) -> f32 {
    let mut max_error = 0.0f32;
    let mut total_error = 0.0f32;
    let mut error_count = 0;

    for i in 0..n.min(32) { // Only check first 32 rows for performance
        for j in 0..n.min(32) { // Only check first 32 cols for performance
            let mut expected = 0.0f32;
            for k in 0..n {
                expected += a[i * n + k] * b[k * n + j];
            }

            let actual = c[i * n + j];
            let error = (actual - expected).abs();
            let relative_error = if expected.abs() > 1e-7 { error / expected.abs() } else { error };

            if relative_error > 1e-4 {
                error_count += 1;
                if error_count <= 5 { // Show first few errors
                    println!("  Error at ({}, {}): expected {:.6}, got {:.6}, rel_error {:.6}",
                             i, j, expected, actual, relative_error);
                }
            }

            max_error = max_error.max(relative_error);
            total_error += relative_error;
        }
    }

    let checked_elements = (n.min(32) * n.min(32)) as f32;
    let avg_error = total_error / checked_elements;

    if error_count == 0 {
        println!("  ✓ Verification passed (max error: {:.2e})", max_error);
    } else {
        println!("  ⚠ {} errors found (avg error: {:.2e}, max error: {:.2e})",
                 error_count, avg_error, max_error);
    }

    max_error
}

fn cpu_matrix_multiply(a: &[f32], b: &[f32], n: usize) -> (Vec<f32>, f64) {
    let start = Instant::now();
    let mut c = vec![0.0f32; n * n];

    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }

    let duration = start.elapsed().as_secs_f64();
    (c, duration)
}

fn run_basic_matrix_multiply(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    a: &[f32],
    b: &[f32],
    n: usize,
) -> Result<(Vec<f32>, f64)> {
    let byte_length = (n * n * std::mem::size_of::<f32>()) as u64;

    // Create Metal buffers
    let a_buffer = device.newBufferWithBytes_length_options(
        a.as_ptr() as *const std::ffi::c_void,
        byte_length,
        MTLResourceOptions::StorageModeShared,
    );

    let b_buffer = device.newBufferWithBytes_length_options(
        b.as_ptr() as *const std::ffi::c_void,
        byte_length,
        MTLResourceOptions::StorageModeShared,
    );

    let c_buffer = device.newBufferWithLength_options(
        byte_length,
        MTLResourceOptions::StorageModeShared,
    );

    let n_buffer = device.newBufferWithBytes_length_options(
        &n as *const usize as *const std::ffi::c_void,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create compute pipeline
    let function_name = NSString::from_str("matrix_multiply_basic");
    let function = library.newFunctionWithName(&function_name).context("Function not found")?;
    let pipeline = device.newComputePipelineStateWithFunction_error(&function)
        .context("Failed to create compute pipeline")?;

    // Create command queue and buffer
    let command_queue = device.newCommandQueue().context("Failed to create command queue")?;
    let command_buffer = command_queue.commandBuffer().context("Failed to create command buffer")?;

    // Create compute encoder
    let encoder = command_buffer.computeCommandEncoder().context("Failed to create compute encoder")?;
    encoder.setComputePipelineState(&pipeline);
    encoder.setBuffer_offset_atIndex(Some(&a_buffer), 0, 0);
    encoder.setBuffer_offset_atIndex(Some(&b_buffer), 0, 1);
    encoder.setBuffer_offset_atIndex(Some(&c_buffer), 0, 2);
    encoder.setBuffer_offset_atIndex(Some(&n_buffer), 0, 3);

    // Configure dispatch
    let threadgroup_size = MTLSize { width: 16, height: 16, depth: 1 };
    let threadgroups = MTLSize {
        width: (n as u64 + threadgroup_size.width - 1) / threadgroup_size.width,
        height: (n as u64 + threadgroup_size.height - 1) / threadgroup_size.height,
        depth: 1,
    };

    // Execute
    let start = Instant::now();
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threadgroup_size);
    encoder.endEncoding();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    let duration = start.elapsed().as_secs_f64();

    // Read results
    let mut c = vec![0.0f32; n * n];
    let result_ptr = c_buffer.contents() as *const f32;
    unsafe {
        std::ptr::copy_nonoverlapping(result_ptr, c.as_mut_ptr(), n * n);
    }

    Ok((c, duration))
}

fn run_tiled_matrix_multiply(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    a: &[f32],
    b: &[f32],
    n: usize,
) -> Result<(Vec<f32>, f64)> {
    let byte_length = (n * n * std::mem::size_of::<f32>()) as u64;
    let tile_size = 16;

    // Create Metal buffers
    let a_buffer = device.newBufferWithBytes_length_options(
        a.as_ptr() as *const std::ffi::c_void,
        byte_length,
        MTLResourceOptions::StorageModeShared,
    );

    let b_buffer = device.newBufferWithBytes_length_options(
        b.as_ptr() as *const std::ffi::c_void,
        byte_length,
        MTLResourceOptions::StorageModeShared,
    );

    let c_buffer = device.newBufferWithLength_options(
        byte_length,
        MTLResourceOptions::StorageModeShared,
    );

    let n_buffer = device.newBufferWithBytes_length_options(
        &n as *const usize as *const std::ffi::c_void,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create compute pipeline
    let function_name = NSString::from_str("matrix_multiply_tiled");
    let function = library.newFunctionWithName(&function_name).context("Function not found")?;
    let pipeline = device.newComputePipelineStateWithFunction_error(&function)
        .context("Failed to create compute pipeline")?;

    // Create command queue and buffer
    let command_queue = device.newCommandQueue().context("Failed to create command queue")?;
    let command_buffer = command_queue.commandBuffer().context("Failed to create command buffer")?;

    // Create compute encoder
    let encoder = command_buffer.computeCommandEncoder().context("Failed to create compute encoder")?;
    encoder.setComputePipelineState(&pipeline);
    encoder.setBuffer_offset_atIndex(Some(&a_buffer), 0, 0);
    encoder.setBuffer_offset_atIndex(Some(&b_buffer), 0, 1);
    encoder.setBuffer_offset_atIndex(Some(&c_buffer), 0, 2);
    encoder.setBuffer_offset_atIndex(Some(&n_buffer), 0, 3);

    // Set threadgroup memory for tiles
    let shared_memory_size = (tile_size * tile_size * std::mem::size_of::<f32>()) as u64;
    encoder.setThreadgroupMemoryLength_atIndex(shared_memory_size, 0);
    encoder.setThreadgroupMemoryLength_atIndex(shared_memory_size, 1);

    // Configure dispatch
    let threadgroup_size = MTLSize {
        width: tile_size as u64,
        height: tile_size as u64,
        depth: 1
    };
    let threadgroups = MTLSize {
        width: (n as u64 + tile_size as u64 - 1) / tile_size as u64,
        height: (n as u64 + tile_size as u64 - 1) / tile_size as u64,
        depth: 1,
    };

    // Execute
    let start = Instant::now();
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threadgroup_size);
    encoder.endEncoding();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    let duration = start.elapsed().as_secs_f64();

    // Read results
    let mut c = vec![0.0f32; n * n];
    let result_ptr = c_buffer.contents() as *const f32;
    unsafe {
        std::ptr::copy_nonoverlapping(result_ptr, c.as_mut_ptr(), n * n);
    }

    Ok((c, duration))
}

fn run_mps_matrix_multiply(
    device: &ProtocolObject<dyn MTLDevice>,
    a: &[f32],
    b: &[f32],
    n: usize,
) -> Result<(Vec<f32>, f64)> {
    let byte_length = (n * n * std::mem::size_of::<f32>()) as u64;

    // Create Metal buffers
    let a_buffer = device.newBufferWithBytes_length_options(
        a.as_ptr() as *const std::ffi::c_void,
        byte_length,
        MTLResourceOptions::StorageModeShared,
    );

    let b_buffer = device.newBufferWithBytes_length_options(
        b.as_ptr() as *const std::ffi::c_void,
        byte_length,
        MTLResourceOptions::StorageModeShared,
    );

    let c_buffer = device.newBufferWithLength_options(
        byte_length,
        MTLResourceOptions::StorageModeShared,
    );

    // Create MPS matrix multiplication
    let command_queue = device.newCommandQueue().context("Failed to create command queue")?;
    let command_buffer = command_queue.commandBuffer().context("Failed to create command buffer")?;

    // Note: MPS matrix multiplication requires specific setup for matrix descriptors
    // For now, we'll use a simplified approach

    let start = Instant::now();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    let duration = start.elapsed().as_secs_f64();

    // For this example, we'll return the basic result
    // In a real implementation, you'd use MPSMatrixMultiplication
    let c = vec![0.0f32; n * n];

    Ok((c, duration))
}

fn main() -> Result<()> {
    println!("Apple Metal Matrix Multiplication");
    println!("================================");

    #[cfg(not(target_os = "macos"))]
    {
        println!("This example requires macOS with Metal support.");
        return Ok(());
    }

    let device = create_metal_device()?;
    let library = compile_shaders(&device)?;

    // Test different matrix sizes
    let sizes = vec![256, 512, 1024];

    for n in sizes {
        println!("\n{'=':<60}");
        println!("Matrix Size: {}x{}", n, n);
        println!("{'=':<60}");

        // Create test matrices
        let (a, b) = create_test_matrices(n);
        println!("Generated {}x{} test matrices", n, n);

        // CPU reference implementation
        println!("\nRunning CPU matrix multiplication...");
        let (c_cpu, cpu_time) = cpu_matrix_multiply(&a, &b, n);
        println!("  CPU time: {:.2}ms", cpu_time * 1000.0);

        // Basic Metal implementation
        println!("\nRunning basic Metal matrix multiplication...");
        match run_basic_matrix_multiply(&device, &library, &a, &b, n) {
            Ok((c_basic, gpu_time)) => {
                println!("  GPU basic time: {:.2}ms", gpu_time * 1000.0);
                let speedup = cpu_time / gpu_time;
                println!("  Speedup: {:.2}x", speedup);

                verify_result(&a, &b, &c_basic, n);

                let gflops = (2.0 * n as f64 * n as f64 * n as f64) / (gpu_time * 1e9);
                println!("  Performance: {:.2} GFLOPS", gflops);
            },
            Err(e) => println!("  ✗ Basic Metal failed: {}", e),
        }

        // Tiled Metal implementation (for larger matrices)
        if n >= 512 {
            println!("\nRunning tiled Metal matrix multiplication...");
            match run_tiled_matrix_multiply(&device, &library, &a, &b, n) {
                Ok((c_tiled, gpu_tiled_time)) => {
                    println!("  GPU tiled time: {:.2}ms", gpu_tiled_time * 1000.0);
                    let speedup = cpu_time / gpu_tiled_time;
                    println!("  Tiled speedup: {:.2}x", speedup);

                    verify_result(&a, &b, &c_tiled, n);

                    let gflops = (2.0 * n as f64 * n as f64 * n as f64) / (gpu_tiled_time * 1e9);
                    println!("  Tiled performance: {:.2} GFLOPS", gflops);
                },
                Err(e) => println!("  ✗ Tiled Metal failed: {}", e),
            }
        }

        // Performance summary
        println!("\nPerformance Summary:");
        println!("  Matrix elements: {}", n * n);
        println!("  Operations: {:.2e} FLOPs", 2.0 * n as f64 * n as f64 * n as f64);
        println!("  Memory bandwidth (theoretical): {:.2} GB/s",
                 (3.0 * n as f64 * n as f64 * 4.0) / (cpu_time * 1e9));
    }

    Ok(())
}