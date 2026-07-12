use anyhow::{Context, Result};
use objc2::rc::Retained;
use objc2_foundation::NSString;
use objc2_metal::*;
use rand::Rng;
use std::time::Instant;

fn create_metal_device() -> Result<Retained<ProtocolObject<dyn MTLDevice>>> {
    let device = MTLCreateSystemDefaultDevice()
        .context("No Metal device found. Metal requires macOS 10.11+ or iOS 8.0+")?;

    println!("Metal Device: {}", device.name());
    println!("  Max threads per threadgroup: {}", device.maxThreadsPerThreadgroup().width);
    println!("  Supports unified memory: {}", device.hasUnifiedMemory());

    Ok(device)
}

fn compile_shaders(device: &ProtocolObject<dyn MTLDevice>) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let shader_source = include_str!("../shaders/monte_carlo.metal");
    let source_string = NSString::from_str(shader_source);

    let library = device
        .newLibraryWithSource_options_error(&source_string, None)
        .context("Failed to compile Metal shaders")?;

    println!("✓ Metal shaders compiled successfully");
    Ok(library)
}

fn cpu_monte_carlo_pi(samples: u32) -> (f64, f64) {
    let start = Instant::now();
    let mut rng = rand::thread_rng();
    let mut inside_count = 0u32;

    for _ in 0..samples {
        let x: f64 = rng.gen_range(-1.0..1.0);
        let y: f64 = rng.gen_range(-1.0..1.0);

        if x * x + y * y <= 1.0 {
            inside_count += 1;
        }
    }

    let duration = start.elapsed().as_secs_f64();
    let pi_estimate = 4.0 * inside_count as f64 / samples as f64;

    (pi_estimate, duration)
}

fn run_monte_carlo_pi_metal(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    total_samples: u32,
    use_philox: bool,
) -> Result<(f64, f64)> {
    let num_threads = 1024;
    let samples_per_thread = (total_samples + num_threads - 1) / num_threads;
    let actual_samples = samples_per_thread * num_threads;

    // Create buffers
    let results_buffer = device.newBufferWithLength_options(
        (num_threads as usize * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let samples_buffer = device.newBufferWithBytes_length_options(
        &samples_per_thread as *const u32 as *const std::ffi::c_void,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let seed = rand::thread_rng().gen::<u32>();
    let seed_buffer = device.newBufferWithBytes_length_options(
        &seed as *const u32 as *const std::ffi::c_void,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create compute pipeline
    let function_name = if use_philox {
        "monte_carlo_pi_philox"
    } else {
        "monte_carlo_pi_simple"
    };
    let function_name_ns = NSString::from_str(function_name);
    let function = library.newFunctionWithName(&function_name_ns)
        .context("Failed to find compute function")?;
    let pipeline = device.newComputePipelineStateWithFunction_error(&function)
        .context("Failed to create compute pipeline")?;

    // Create command queue and buffer
    let command_queue = device.newCommandQueue().context("Failed to create command queue")?;
    let command_buffer = command_queue.commandBuffer().context("Failed to create command buffer")?;

    // Create compute encoder
    let encoder = command_buffer.computeCommandEncoder().context("Failed to create compute encoder")?;
    encoder.setComputePipelineState(&pipeline);
    encoder.setBuffer_offset_atIndex(Some(&results_buffer), 0, 0);
    encoder.setBuffer_offset_atIndex(Some(&samples_buffer), 0, 1);
    encoder.setBuffer_offset_atIndex(Some(&seed_buffer), 0, 2);

    // Configure dispatch
    let threadgroup_size = MTLSize { width: 256, height: 1, depth: 1 };
    let threadgroups = MTLSize {
        width: (num_threads as u64 + threadgroup_size.width - 1) / threadgroup_size.width,
        height: 1,
        depth: 1,
    };

    // Execute
    let start = Instant::now();
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threadgroup_size);
    encoder.endEncoding();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    let duration = start.elapsed().as_secs_f64();

    // Read results and sum them
    let result_ptr = results_buffer.contents() as *const u32;
    let mut total_inside = 0u64;

    unsafe {
        for i in 0..num_threads as usize {
            total_inside += *result_ptr.add(i) as u64;
        }
    }

    let pi_estimate = 4.0 * total_inside as f64 / actual_samples as f64;

    println!("  Processed {} samples ({} threads × {} samples/thread)",
             actual_samples, num_threads, samples_per_thread);
    println!("  Points inside circle: {}", total_inside);

    Ok((pi_estimate, duration))
}

fn run_parallel_reduction_sum(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    input: &[u32],
) -> Result<(u64, f64)> {
    let n = input.len();
    let threadgroup_size = 256;
    let num_groups = (n + threadgroup_size - 1) / threadgroup_size;

    // Create buffers
    let input_buffer = device.newBufferWithBytes_length_options(
        input.as_ptr() as *const std::ffi::c_void,
        (n * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device.newBufferWithLength_options(
        (num_groups * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create compute pipeline
    let function_name = NSString::from_str("parallel_sum");
    let function = library.newFunctionWithName(&function_name)
        .context("Failed to find reduction function")?;
    let pipeline = device.newComputePipelineStateWithFunction_error(&function)
        .context("Failed to create reduction pipeline")?;

    // Create command queue and buffer
    let command_queue = device.newCommandQueue().context("Failed to create command queue")?;
    let command_buffer = command_queue.commandBuffer().context("Failed to create command buffer")?;

    // Create compute encoder
    let encoder = command_buffer.computeCommandEncoder().context("Failed to create compute encoder")?;
    encoder.setComputePipelineState(&pipeline);
    encoder.setBuffer_offset_atIndex(Some(&input_buffer), 0, 0);
    encoder.setBuffer_offset_atIndex(Some(&output_buffer), 0, 1);
    encoder.setThreadgroupMemoryLength_atIndex((threadgroup_size * std::mem::size_of::<u32>()) as u64, 0);

    let threadgroups = MTLSize { width: num_groups as u64, height: 1, depth: 1 };
    let threads_per_group = MTLSize { width: threadgroup_size as u64, height: 1, depth: 1 };

    // Execute
    let start = Instant::now();
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_group);
    encoder.endEncoding();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    let duration = start.elapsed().as_secs_f64();

    // Read partial results and sum them
    let result_ptr = output_buffer.contents() as *const u32;
    let mut total_sum = 0u64;

    unsafe {
        for i in 0..num_groups {
            total_sum += *result_ptr.add(i) as u64;
        }
    }

    Ok((total_sum, duration))
}

fn run_monte_carlo_integration(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    a: f32,
    b: f32,
    total_samples: u32,
) -> Result<(f64, f64)> {
    let num_threads = 512;
    let samples_per_thread = (total_samples + num_threads - 1) / num_threads;

    // Create buffers
    let results_buffer = device.newBufferWithLength_options(
        (num_threads as usize * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let samples_buffer = device.newBufferWithBytes_length_options(
        &samples_per_thread as *const u32 as *const std::ffi::c_void,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let seed = rand::thread_rng().gen::<u32>();
    let seed_buffer = device.newBufferWithBytes_length_options(
        &seed as *const u32 as *const std::ffi::c_void,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let a_buffer = device.newBufferWithBytes_length_options(
        &a as *const f32 as *const std::ffi::c_void,
        std::mem::size_of::<f32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let b_buffer = device.newBufferWithBytes_length_options(
        &b as *const f32 as *const std::ffi::c_void,
        std::mem::size_of::<f32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create compute pipeline
    let function_name = NSString::from_str("monte_carlo_integrate");
    let function = library.newFunctionWithName(&function_name)
        .context("Failed to find integration function")?;
    let pipeline = device.newComputePipelineStateWithFunction_error(&function)
        .context("Failed to create integration pipeline")?;

    // Create command queue and buffer
    let command_queue = device.newCommandQueue().context("Failed to create command queue")?;
    let command_buffer = command_queue.commandBuffer().context("Failed to create command buffer")?;

    // Create compute encoder
    let encoder = command_buffer.computeCommandEncoder().context("Failed to create compute encoder")?;
    encoder.setComputePipelineState(&pipeline);
    encoder.setBuffer_offset_atIndex(Some(&results_buffer), 0, 0);
    encoder.setBuffer_offset_atIndex(Some(&samples_buffer), 0, 1);
    encoder.setBuffer_offset_atIndex(Some(&seed_buffer), 0, 2);
    encoder.setBuffer_offset_atIndex(Some(&a_buffer), 0, 3);
    encoder.setBuffer_offset_atIndex(Some(&b_buffer), 0, 4);

    // Configure dispatch
    let threadgroup_size = MTLSize { width: 64, height: 1, depth: 1 };
    let threadgroups = MTLSize {
        width: (num_threads as u64 + threadgroup_size.width - 1) / threadgroup_size.width,
        height: 1,
        depth: 1,
    };

    // Execute
    let start = Instant::now();
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threadgroup_size);
    encoder.endEncoding();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    let duration = start.elapsed().as_secs_f64();

    // Read results and sum them
    let result_ptr = results_buffer.contents() as *const f32;
    let mut total_integral = 0.0f64;

    unsafe {
        for i in 0..num_threads as usize {
            total_integral += *result_ptr.add(i) as f64;
        }
    }

    // Average the results
    total_integral /= num_threads as f64;

    Ok((total_integral, duration))
}

fn main() -> Result<()> {
    println!("Apple Metal Monte Carlo Computation");
    println!("==================================");

    #[cfg(not(target_os = "macos"))]
    {
        println!("This example requires macOS with Metal support.");
        return Ok(());
    }

    let device = create_metal_device()?;
    let library = compile_shaders(&device)?;

    // Test different sample counts
    let sample_counts = vec![1_000_000, 10_000_000, 100_000_000];

    for samples in sample_counts {
        println!("\n{'=':<60}");
        println!("Monte Carlo Pi Estimation - {} samples", samples);
        println!("{'=':<60}");

        // CPU reference
        println!("Running CPU Monte Carlo Pi calculation...");
        let (cpu_pi, cpu_time) = cpu_monte_carlo_pi(samples);
        let cpu_error = (cpu_pi - std::f64::consts::PI).abs();
        println!("  CPU Pi estimate: {:.6}", cpu_pi);
        println!("  CPU time: {:.2}ms", cpu_time * 1000.0);
        println!("  CPU error: {:.6}", cpu_error);

        // GPU simple RNG
        println!("\nRunning Metal Monte Carlo Pi (simple RNG)...");
        match run_monte_carlo_pi_metal(&device, &library, samples, false) {
            Ok((gpu_pi, gpu_time)) => {
                let gpu_error = (gpu_pi - std::f64::consts::PI).abs();
                println!("  Metal Pi estimate: {:.6}", gpu_pi);
                println!("  Metal time: {:.2}ms", gpu_time * 1000.0);
                println!("  Metal error: {:.6}", gpu_error);

                let speedup = cpu_time / gpu_time;
                let samples_per_sec = samples as f64 / gpu_time;
                println!("  Speedup vs CPU: {:.2}x", speedup);
                println!("  Samples per second: {:.2e}", samples_per_sec);
            },
            Err(e) => println!("  ✗ Metal simple RNG failed: {}", e),
        }

        // GPU Philox RNG (higher quality)
        println!("\nRunning Metal Monte Carlo Pi (Philox RNG)...");
        match run_monte_carlo_pi_metal(&device, &library, samples, true) {
            Ok((gpu_pi_philox, gpu_time_philox)) => {
                let gpu_error_philox = (gpu_pi_philox - std::f64::consts::PI).abs();
                println!("  Metal Pi estimate (Philox): {:.6}", gpu_pi_philox);
                println!("  Metal time (Philox): {:.2}ms", gpu_time_philox * 1000.0);
                println!("  Metal error (Philox): {:.6}", gpu_error_philox);

                let speedup_philox = cpu_time / gpu_time_philox;
                println!("  Speedup vs CPU (Philox): {:.2}x", speedup_philox);
            },
            Err(e) => println!("  ✗ Metal Philox RNG failed: {}", e),
        }

        // Performance summary
        println!("\nPerformance Summary:");
        println!("  Total samples: {}", samples);
        println!("  Actual Pi: {:.6}", std::f64::consts::PI);
        println!("  Theoretical error scale: {:.6}", 1.0 / (samples as f64).sqrt());
    }

    // Test parallel reduction
    println!("\n{'=':<60}");
    println!("Parallel Reduction Test");
    println!("{'=':<60}");

    let test_data: Vec<u32> = (1..=10000).collect();
    let expected_sum: u64 = test_data.iter().map(|&x| x as u64).sum();

    match run_parallel_reduction_sum(&device, &library, &test_data) {
        Ok((gpu_sum, reduction_time)) => {
            println!("  Expected sum: {}", expected_sum);
            println!("  GPU sum: {}", gpu_sum);
            println!("  Time: {:.2}ms", reduction_time * 1000.0);

            if gpu_sum == expected_sum {
                println!("  ✓ Parallel reduction correct!");
            } else {
                println!("  ✗ Parallel reduction error: {}", (gpu_sum as i64 - expected_sum as i64).abs());
            }
        },
        Err(e) => println!("  ✗ Parallel reduction failed: {}", e),
    }

    // Test Monte Carlo integration
    println!("\n{'=':<60}");
    println!("Monte Carlo Integration Test (∫x² dx from 0 to 1)");
    println!("{'=':<60}");

    let a = 0.0f32;
    let b = 1.0f32;
    let expected_integral = (b.powi(3) - a.powi(3)) / 3.0; // ∫x² dx = x³/3

    match run_monte_carlo_integration(&device, &library, a, b, 10_000_000) {
        Ok((gpu_integral, integration_time)) => {
            let error = (gpu_integral as f32 - expected_integral).abs();
            println!("  Expected integral: {:.6}", expected_integral);
            println!("  GPU integral: {:.6}", gpu_integral);
            println!("  Error: {:.6}", error);
            println!("  Time: {:.2}ms", integration_time * 1000.0);

            if error < 0.001 {
                println!("  ✓ Monte Carlo integration successful!");
            } else {
                println!("  ⚠ Integration error larger than expected");
            }
        },
        Err(e) => println!("  ✗ Monte Carlo integration failed: {}", e),
    }

    println!("\n✓ All Monte Carlo tests completed!");

    Ok(())
}