use anyhow::Result;
use libc::{c_char, c_int};
use rand::Rng;
use std::ffi::CStr;
use std::time::Instant;

#[repr(C)]
struct DeviceInfo {
    name: [c_char; 256],
    vendor: [c_char; 256],
    is_gpu: bool,
    max_work_group_size: c_int,
}

#[repr(C)]
struct OptionData {
    spot_price: f32,
    strike_price: f32,
    risk_free_rate: f32,
    volatility: f32,
    time_to_maturity: f32,
    is_call: c_int,  // 1 for call, 0 for put
}

extern "C" {
    fn get_device_count() -> c_int;
    fn get_device_info(device_id: c_int, info: *mut DeviceInfo) -> c_int;
    fn binomial_option_pricing_sycl(
        options: *mut OptionData,
        option_prices: *mut f32,
        num_options: c_int,
        num_steps: c_int,
        device_id: c_int,
        elapsed_ms: *mut f64,
    ) -> c_int;
    fn binomial_option_pricing_optimized_sycl(
        options: *mut OptionData,
        option_prices: *mut f32,
        num_options: c_int,
        num_steps: c_int,
        device_id: c_int,
        elapsed_ms: *mut f64,
    ) -> c_int;
    fn compute_black_scholes_reference(
        options: *mut OptionData,
        bs_prices: *mut f32,
        num_options: c_int,
    ) -> c_int;
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

fn verify_results(binomial_prices: &[f32], bs_prices: &[f32], tolerance: f32) -> (usize, f32, f32) {
    let mut matches = 0;
    let mut max_error = 0.0f32;
    let mut total_error = 0.0f32;

    for (i, (&binomial, &bs)) in binomial_prices.iter().zip(bs_prices.iter()).enumerate() {
        let error = (binomial - bs).abs();
        let relative_error = if bs.abs() > 1e-6 { error / bs.abs() } else { error };

        total_error += relative_error;
        max_error = max_error.max(relative_error);

        if relative_error < tolerance {
            matches += 1;
        } else if i < 5 {  // Show first few mismatches for debugging
            println!("  Option {}: Binomial={:.4}, BS={:.4}, Error={:.4} ({:.2}%)",
                     i, binomial, bs, error, relative_error * 100.0);
        }
    }

    (matches, max_error, total_error / binomial_prices.len() as f32)
}

fn main() -> Result<()> {
    println!("oneAPI SYCL Binomial Options Pricing");
    println!("====================================");

    // Get available devices
    let device_count = unsafe { get_device_count() };
    println!("Found {} SYCL device(s)", device_count);

    if device_count == 0 {
        println!("No SYCL devices found!");
        return Ok(());
    }

    // List all devices
    for i in 0..device_count {
        let mut device_info = DeviceInfo {
            name: [0; 256],
            vendor: [0; 256],
            is_gpu: false,
            max_work_group_size: 0,
        };

        let result = unsafe { get_device_info(i, &mut device_info) };
        if result == 0 {
            let name = unsafe { CStr::from_ptr(device_info.name.as_ptr()).to_string_lossy() };
            let vendor = unsafe { CStr::from_ptr(device_info.vendor.as_ptr()).to_string_lossy() };

            println!("\nDevice {}: {}", i, name);
            println!("  Vendor: {}", vendor);
            println!("  Type: {}", if device_info.is_gpu { "GPU" } else { "CPU/Other" });
            println!("  Max work group size: {}", device_info.max_work_group_size);
        }
    }

    // Select GPU device (prefer Intel for MAX GPUs)
    let mut selected_device = -1;
    for i in 0..device_count {
        let mut device_info = DeviceInfo {
            name: [0; 256],
            vendor: [0; 256],
            is_gpu: false,
            max_work_group_size: 0,
        };

        let result = unsafe { get_device_info(i, &mut device_info) };
        if result == 0 {
            let vendor = unsafe { CStr::from_ptr(device_info.vendor.as_ptr()).to_string_lossy() };

            if device_info.is_gpu && vendor.to_lowercase().contains("intel") {
                selected_device = i;
                break;
            }
            if device_info.is_gpu && selected_device == -1 {
                selected_device = i;
            }
        }
    }

    if selected_device == -1 {
        selected_device = 0;
    }

    println!("\nUsing device {} for computation", selected_device);

    // Test different configurations
    let test_configs = vec![
        (1024, 256),    // Small test
        (16384, 512),   // Medium test
        (65536, 1024),  // Large test
        (262144, 2048), // Production size
    ];

    for (num_options, num_steps) in test_configs {
        println!("\n{:=<70}", "");
        println!("Configuration: {} options, {} time steps", num_options, num_steps);
        println!("{:=<70}", "");

        // Generate test options
        println!("Generating {} test options...", num_options);
        let mut options = generate_test_options(num_options);

        // Prepare output arrays
        let mut binomial_prices = vec![0.0f32; num_options];
        let mut binomial_prices_opt = vec![0.0f32; num_options];
        let mut bs_prices = vec![0.0f32; num_options];

        // Compute Black-Scholes reference prices
        println!("Computing Black-Scholes reference prices...");
        let bs_start = Instant::now();
        let bs_result = unsafe {
            compute_black_scholes_reference(
                options.as_mut_ptr(),
                bs_prices.as_mut_ptr(),
                num_options as c_int,
            )
        };
        let bs_time = bs_start.elapsed();

        if bs_result != 0 {
            println!("✗ Black-Scholes computation failed!");
            continue;
        }

        println!("Black-Scholes time: {:.2}ms", bs_time.as_secs_f64() * 1000.0);

        // Run basic SYCL binomial pricing
        println!("Running basic SYCL binomial option pricing...");
        let mut sycl_elapsed = 0.0;
        let sycl_result = unsafe {
            binomial_option_pricing_sycl(
                options.as_mut_ptr(),
                binomial_prices.as_mut_ptr(),
                num_options as c_int,
                num_steps as c_int,
                selected_device,
                &mut sycl_elapsed,
            )
        };

        if sycl_result == 0 {
            println!("SYCL basic time: {:.2}ms", sycl_elapsed);

            // Verify against Black-Scholes
            let (matches, max_error, avg_error) = verify_results(&binomial_prices, &bs_prices, 0.05);
            println!("Accuracy vs Black-Scholes:");
            println!("  Matches (within 5%): {}/{} ({:.1}%)",
                     matches, num_options, 100.0 * matches as f32 / num_options as f32);
            println!("  Max relative error: {:.4} ({:.2}%)", max_error, max_error * 100.0);
            println!("  Avg relative error: {:.4} ({:.2}%)", avg_error, avg_error * 100.0);

            // Performance metrics
            let options_per_sec = num_options as f64 / (sycl_elapsed / 1000.0);
            let bs_speedup = (bs_time.as_secs_f64() * 1000.0) / sycl_elapsed;
            println!("Performance:");
            println!("  Options per second: {:.2e}", options_per_sec);
            println!("  Speedup vs Black-Scholes: {:.2}x", bs_speedup);

        } else {
            println!("✗ SYCL basic execution failed!");
        }

        // Run optimized SYCL version for smaller problem sizes
        if num_steps <= 512 && num_options <= 65536 {
            println!("Running optimized SYCL binomial option pricing...");
            let mut sycl_opt_elapsed = 0.0;
            let sycl_opt_result = unsafe {
                binomial_option_pricing_optimized_sycl(
                    options.as_mut_ptr(),
                    binomial_prices_opt.as_mut_ptr(),
                    num_options as c_int,
                    num_steps as c_int,
                    selected_device,
                    &mut sycl_opt_elapsed,
                )
            };

            if sycl_opt_result == 0 {
                println!("SYCL optimized time: {:.2}ms", sycl_opt_elapsed);

                let (matches_opt, max_error_opt, avg_error_opt) =
                    verify_results(&binomial_prices_opt, &bs_prices, 0.05);
                println!("Optimized accuracy vs Black-Scholes:");
                println!("  Matches (within 5%): {}/{} ({:.1}%)",
                         matches_opt, num_options, 100.0 * matches_opt as f32 / num_options as f32);
                println!("  Max relative error: {:.4} ({:.2}%)", max_error_opt, max_error_opt * 100.0);

                if sycl_elapsed > 0.0 {
                    let optimization_speedup = sycl_elapsed / sycl_opt_elapsed;
                    println!("  Optimization improvement: {:.2}x", optimization_speedup);
                }

            } else {
                println!("✗ Optimized SYCL execution failed!");
            }
        }

        // Show sample results
        println!("\nSample Results (first 5 options):");
        println!("{:<8} {:<6} {:<8} {:<8} {:<8} {:<8} {:<8} {:<10} {:<10}",
                 "Option", "Type", "Spot", "Strike", "Rate", "Vol", "Time", "Binomial", "Black-Scholes");

        for i in 0..5.min(num_options) {
            let opt = &options[i];
            println!("{:<8} {:<6} {:<8.2} {:<8.2} {:<8.3} {:<8.3} {:<8.3} {:<10.4} {:<10.4}",
                     i,
                     if opt.is_call == 1 { "Call" } else { "Put" },
                     opt.spot_price,
                     opt.strike_price,
                     opt.risk_free_rate,
                     opt.volatility,
                     opt.time_to_maturity,
                     binomial_prices[i],
                     bs_prices[i]
            );
        }

        // Convergence analysis for large problems
        if num_steps >= 1024 {
            println!("\nConvergence Analysis:");
            println!("  With {} time steps, binomial model should closely approximate Black-Scholes", num_steps);
            println!("  Binomial convergence rate: O(1/n) where n is number of time steps");
            println!("  For accurate results, typically need 100+ time steps");
        }
    }

    println!("\nNote: Binomial model converges to Black-Scholes as number of time steps increases.");
    println!("Small differences between models are expected and normal.");

    Ok(())
}