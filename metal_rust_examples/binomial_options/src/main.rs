use anyhow::{Context, Result};
use objc2::rc::Retained;
use objc2_foundation::NSString;
use objc2_metal::*;
use rand::Rng;
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy)]
struct OptionData {
    spot_price: f32,
    strike_price: f32,
    risk_free_rate: f32,
    volatility: f32,
    time_to_maturity: f32,
    is_call: u32, // 1 for call, 0 for put
}

fn create_metal_device() -> Result<Retained<ProtocolObject<dyn MTLDevice>>> {
    let device = MTLCreateSystemDefaultDevice()
        .context("No Metal device found. Metal requires macOS 10.11+ or iOS 8.0+")?;

    println!("Metal Device: {}", device.name());
    println!("  Max threads per threadgroup: {}", device.maxThreadsPerThreadgroup().width);
    println!("  Supports unified memory: {}", device.hasUnifiedMemory());

    Ok(device)
}

fn compile_shaders(device: &ProtocolObject<dyn MTLDevice>) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let shader_source = include_str!("../shaders/binomial_options.metal");
    let source_string = NSString::from_str(shader_source);

    let library = device
        .newLibraryWithSource_options_error(&source_string, None)
        .context("Failed to compile Metal shaders")?;

    println!("✓ Metal shaders compiled successfully");
    Ok(library)
}

fn generate_test_options(num_options: usize) -> Vec<OptionData> {
    let mut rng = rand::thread_rng();
    let mut options = Vec::with_capacity(num_options);

    for _ in 0..num_options {
        options.push(OptionData {
            spot_price: rng.gen_range(50.0..150.0),
            strike_price: rng.gen_range(50.0..150.0),
            risk_free_rate: rng.gen_range(0.01..0.06),
            volatility: rng.gen_range(0.1..0.4),
            time_to_maturity: rng.gen_range(0.1..2.0),
            is_call: if rng.gen_bool(0.5) { 1 } else { 0 },
        });
    }

    options
}

fn cpu_black_scholes(opt: &OptionData) -> f32 {
    let s = opt.spot_price;
    let k = opt.strike_price;
    let r = opt.risk_free_rate;
    let sigma = opt.volatility;
    let t = opt.time_to_maturity;
    let is_call = opt.is_call == 1;

    if t <= 0.0 {
        return if is_call {
            (s - k).max(0.0)
        } else {
            (k - s).max(0.0)
        };
    }

    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();

    // Simplified normal CDF approximation
    let norm_cdf = |x: f32| -> f32 {
        0.5 * (1.0 + libm::erff(x / std::f32::consts::SQRT_2))
    };

    if is_call {
        s * norm_cdf(d1) - k * (-r * t).exp() * norm_cdf(d2)
    } else {
        k * (-r * t).exp() * norm_cdf(-d2) - s * norm_cdf(-d1)
    }
}

fn cpu_binomial_pricing(opt: &OptionData, num_steps: u32) -> f32 {
    let s = opt.spot_price;
    let k = opt.strike_price;
    let r = opt.risk_free_rate;
    let sigma = opt.volatility;
    let t = opt.time_to_maturity;
    let is_call = opt.is_call == 1;

    let dt = t / num_steps as f32;
    let u = (sigma * dt.sqrt()).exp();
    let d = 1.0 / u;
    let p = ((-r * dt).exp() - d) / (u - d);
    let disc = (-r * dt).exp();

    // Initialize option values at maturity
    let mut values = Vec::with_capacity((num_steps + 1) as usize);
    for i in 0..=num_steps {
        let stock_price = s * u.powi((num_steps - i) as i32) * d.powi(i as i32);
        let payoff = if is_call {
            (stock_price - k).max(0.0)
        } else {
            (k - stock_price).max(0.0)
        };
        values.push(payoff);
    }

    // Backward induction
    for step in (0..num_steps).rev() {
        for i in 0..=step {
            values[i as usize] = disc * (p * values[i as usize] + (1.0 - p) * values[(i + 1) as usize]);
        }
    }

    values[0]
}

fn run_metal_binomial_pricing(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    options: &[OptionData],
    num_steps: u32,
    use_shared_memory: bool,
) -> Result<(Vec<f32>, f64)> {
    let num_options = options.len();

    // Create buffers
    let options_buffer = device.newBufferWithBytes_length_options(
        options.as_ptr() as *const std::ffi::c_void,
        (num_options * std::mem::size_of::<OptionData>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let prices_buffer = device.newBufferWithLength_options(
        (num_options * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let steps_buffer = device.newBufferWithBytes_length_options(
        &num_steps as *const u32 as *const std::ffi::c_void,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create compute pipeline
    let function_name = if use_shared_memory {
        "binomial_option_pricing_shared"
    } else {
        "binomial_option_pricing"
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
    encoder.setBuffer_offset_atIndex(Some(&options_buffer), 0, 0);
    encoder.setBuffer_offset_atIndex(Some(&prices_buffer), 0, 1);
    encoder.setBuffer_offset_atIndex(Some(&steps_buffer), 0, 2);

    if use_shared_memory {
        // Set threadgroup memory for shared version
        let shared_memory_size = ((num_steps + 1) * std::mem::size_of::<f32>() as u32) as u64;
        encoder.setThreadgroupMemoryLength_atIndex(shared_memory_size, 0);
    }

    // Configure dispatch
    let threadgroup_size = if use_shared_memory {
        MTLSize { width: 32, height: 1, depth: 1 }
    } else {
        MTLSize { width: 64, height: 1, depth: 1 }
    };

    let threadgroups = MTLSize {
        width: (num_options as u64 + threadgroup_size.width - 1) / threadgroup_size.width,
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

    // Read results
    let mut prices = vec![0.0f32; num_options];
    let result_ptr = prices_buffer.contents() as *const f32;
    unsafe {
        std::ptr::copy_nonoverlapping(result_ptr, prices.as_mut_ptr(), num_options);
    }

    Ok((prices, duration))
}

fn run_metal_black_scholes(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    options: &[OptionData],
) -> Result<(Vec<f32>, f64)> {
    let num_options = options.len();

    // Create buffers
    let options_buffer = device.newBufferWithBytes_length_options(
        options.as_ptr() as *const std::ffi::c_void,
        (num_options * std::mem::size_of::<OptionData>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let prices_buffer = device.newBufferWithLength_options(
        (num_options * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create compute pipeline
    let function_name = NSString::from_str("black_scholes_pricing");
    let function = library.newFunctionWithName(&function_name)
        .context("Failed to find Black-Scholes function")?;
    let pipeline = device.newComputePipelineStateWithFunction_error(&function)
        .context("Failed to create Black-Scholes pipeline")?;

    // Create command queue and buffer
    let command_queue = device.newCommandQueue().context("Failed to create command queue")?;
    let command_buffer = command_queue.commandBuffer().context("Failed to create command buffer")?;

    // Create compute encoder
    let encoder = command_buffer.computeCommandEncoder().context("Failed to create compute encoder")?;
    encoder.setComputePipelineState(&pipeline);
    encoder.setBuffer_offset_atIndex(Some(&options_buffer), 0, 0);
    encoder.setBuffer_offset_atIndex(Some(&prices_buffer), 0, 1);

    // Configure dispatch
    let threadgroup_size = MTLSize { width: 128, height: 1, depth: 1 };
    let threadgroups = MTLSize {
        width: (num_options as u64 + threadgroup_size.width - 1) / threadgroup_size.width,
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

    // Read results
    let mut prices = vec![0.0f32; num_options];
    let result_ptr = prices_buffer.contents() as *const f32;
    unsafe {
        std::ptr::copy_nonoverlapping(result_ptr, prices.as_mut_ptr(), num_options);
    }

    Ok((prices, duration))
}

fn verify_results(metal_prices: &[f32], reference_prices: &[f32], tolerance: f32) -> (usize, f32, f32) {
    let mut matches = 0;
    let mut max_error = 0.0f32;
    let mut total_error = 0.0f32;

    for (i, (&metal, &reference)) in metal_prices.iter().zip(reference_prices.iter()).enumerate() {
        let error = (metal - reference).abs();
        let relative_error = if reference.abs() > 1e-6 {
            error / reference.abs()
        } else {
            error
        };

        total_error += relative_error;
        max_error = max_error.max(relative_error);

        if relative_error < tolerance {
            matches += 1;
        } else if i < 5 { // Show first few mismatches
            println!("  Option {}: Metal={:.4}, Reference={:.4}, Error={:.4} ({:.2}%)",
                     i, metal, reference, error, relative_error * 100.0);
        }
    }

    (matches, max_error, total_error / metal_prices.len() as f32)
}

fn main() -> Result<()> {
    println!("Apple Metal Binomial Options Pricing");
    println!("====================================");

    #[cfg(not(target_os = "macos"))]
    {
        println!("This example requires macOS with Metal support.");
        return Ok(());
    }

    let device = create_metal_device()?;
    let library = compile_shaders(&device)?;

    // Test different configurations
    let test_configs = vec![
        (1024, 256),    // Small test
        (8192, 512),    // Medium test
        (32768, 1024),  // Large test
        (131072, 2048), // Very large test
    ];

    for (num_options, num_steps) in test_configs {
        println!("\n{'=':<70}");
        println!("Configuration: {} options, {} time steps", num_options, num_steps);
        println!("{'=':<70}");

        // Generate test options
        println!("Generating {} test options...", num_options);
        let options = generate_test_options(num_options);

        // CPU Black-Scholes reference
        println!("Computing CPU Black-Scholes reference...");
        let bs_start = Instant::now();
        let bs_prices: Vec<f32> = options.iter().map(cpu_black_scholes).collect();
        let bs_time = bs_start.elapsed().as_secs_f64();
        println!("  CPU Black-Scholes time: {:.2}ms", bs_time * 1000.0);

        // GPU Black-Scholes
        println!("Running Metal Black-Scholes...");
        match run_metal_black_scholes(&device, &library, &options) {
            Ok((metal_bs_prices, metal_bs_time)) => {
                println!("  Metal Black-Scholes time: {:.2}ms", metal_bs_time * 1000.0);
                let bs_speedup = bs_time / metal_bs_time;
                println!("  Black-Scholes speedup: {:.2}x", bs_speedup);

                let (bs_matches, bs_max_error, bs_avg_error) =
                    verify_results(&metal_bs_prices, &bs_prices, 0.01);
                println!("  Accuracy: {}/{} matches ({:.1}%), max error: {:.2}%, avg error: {:.2}%",
                         bs_matches, num_options,
                         100.0 * bs_matches as f32 / num_options as f32,
                         bs_max_error * 100.0, bs_avg_error * 100.0);
            },
            Err(e) => println!("  ✗ Metal Black-Scholes failed: {}", e),
        }

        // Skip large binomial calculations to avoid memory issues
        if num_steps > 512 {
            println!("  Skipping binomial pricing (num_steps > 512)");
            continue;
        }

        // CPU binomial reference
        println!("Computing CPU binomial reference...");
        let binomial_start = Instant::now();
        let cpu_binomial_prices: Vec<f32> = options.iter()
            .map(|opt| cpu_binomial_pricing(opt, num_steps as u32))
            .collect();
        let binomial_time = binomial_start.elapsed().as_secs_f64();
        println!("  CPU binomial time: {:.2}ms", binomial_time * 1000.0);

        // GPU basic binomial
        println!("Running Metal binomial pricing (basic)...");
        match run_metal_binomial_pricing(&device, &library, &options, num_steps as u32, false) {
            Ok((metal_binomial_prices, metal_binomial_time)) => {
                println!("  Metal binomial time: {:.2}ms", metal_binomial_time * 1000.0);
                let binomial_speedup = binomial_time / metal_binomial_time;
                println!("  Binomial speedup: {:.2}x", binomial_speedup);

                let (binomial_matches, binomial_max_error, binomial_avg_error) =
                    verify_results(&metal_binomial_prices, &cpu_binomial_prices, 0.01);
                println!("  Accuracy vs CPU: {}/{} matches ({:.1}%), max error: {:.2}%, avg error: {:.2}%",
                         binomial_matches, num_options,
                         100.0 * binomial_matches as f32 / num_options as f32,
                         binomial_max_error * 100.0, binomial_avg_error * 100.0);

                // Compare binomial vs Black-Scholes
                let (bs_binomial_matches, bs_binomial_max_error, bs_binomial_avg_error) =
                    verify_results(&metal_binomial_prices, &bs_prices, 0.05);
                println!("  Binomial vs Black-Scholes: {}/{} matches ({:.1}%), max error: {:.2}%",
                         bs_binomial_matches, num_options,
                         100.0 * bs_binomial_matches as f32 / num_options as f32,
                         bs_binomial_max_error * 100.0);

                let options_per_sec = num_options as f64 / metal_binomial_time;
                println!("  Performance: {:.2e} options/second", options_per_sec);
            },
            Err(e) => println!("  ✗ Metal binomial failed: {}", e),
        }

        // GPU shared memory binomial (for smaller problems)
        if num_steps <= 256 && num_options <= 32768 {
            println!("Running Metal binomial pricing (shared memory)...");
            match run_metal_binomial_pricing(&device, &library, &options, num_steps as u32, true) {
                Ok((metal_shared_prices, metal_shared_time)) => {
                    println!("  Metal shared time: {:.2}ms", metal_shared_time * 1000.0);
                    let shared_speedup = binomial_time / metal_shared_time;
                    println!("  Shared speedup: {:.2}x", shared_speedup);

                    let (shared_matches, shared_max_error, shared_avg_error) =
                        verify_results(&metal_shared_prices, &cpu_binomial_prices, 0.01);
                    println!("  Shared accuracy: {}/{} matches ({:.1}%), max error: {:.2}%",
                             shared_matches, num_options,
                             100.0 * shared_matches as f32 / num_options as f32,
                             shared_max_error * 100.0);
                },
                Err(e) => println!("  ✗ Metal shared memory failed: {}", e),
            }
        }

        // Show sample results
        println!("\nSample Results (first 5 options):");
        println!("{:<8} {:<6} {:<8} {:<8} {:<8} {:<8} {:<8} {:<10} {:<10}",
                 "Option", "Type", "Spot", "Strike", "Rate", "Vol", "Time", "Binomial", "Black-Scholes");

        for i in 0..5.min(num_options) {
            let opt = &options[i];
            let binomial_price = if num_steps <= 512 {
                cpu_binomial_pricing(opt, num_steps as u32)
            } else {
                0.0
            };
            let bs_price = cpu_black_scholes(opt);

            println!("{:<8} {:<6} {:<8.2} {:<8.2} {:<8.3} {:<8.3} {:<8.3} {:<10.4} {:<10.4}",
                     i,
                     if opt.is_call == 1 { "Call" } else { "Put" },
                     opt.spot_price,
                     opt.strike_price,
                     opt.risk_free_rate,
                     opt.volatility,
                     opt.time_to_maturity,
                     binomial_price,
                     bs_price
            );
        }
    }

    println!("\n✓ All option pricing tests completed!");
    println!("\nNote: Binomial model converges to Black-Scholes as time steps increase.");
    println!("Small differences between models are expected and indicate proper convergence.");

    Ok(())
}