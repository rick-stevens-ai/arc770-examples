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

extern "C" {
    fn get_device_count() -> c_int;
    fn get_device_info(device_id: c_int, info: *mut DeviceInfo) -> c_int;
    fn monte_carlo_pi_sycl(
        total_samples: c_int,
        device_id: c_int,
        pi_estimate: *mut f64,
        elapsed_ms: *mut f64,
    ) -> c_int;
    fn monte_carlo_pi_philox_sycl(
        total_samples: c_int,
        device_id: c_int,
        pi_estimate: *mut f64,
        elapsed_ms: *mut f64,
    ) -> c_int;
}

fn cpu_monte_carlo_pi(samples: i32) -> (f64, f64) {
    let start = Instant::now();
    let mut rng = rand::thread_rng();
    let mut inside_count = 0;

    for _ in 0..samples {
        let x: f64 = rng.gen_range(-1.0..1.0);
        let y: f64 = rng.gen_range(-1.0..1.0);

        if x * x + y * y <= 1.0 {
            inside_count += 1;
        }
    }

    let duration = start.elapsed();
    let pi_estimate = 4.0 * inside_count as f64 / samples as f64;

    (pi_estimate, duration.as_secs_f64() * 1000.0)
}

fn main() -> Result<()> {
    println!("oneAPI SYCL Monte Carlo Pi Calculation");
    println!("======================================");

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

    // Test different sample counts
    let sample_counts = vec![1_000_000, 10_000_000, 100_000_000];

    for samples in sample_counts {
        println!("\n{'=':<60}");
        println!("Sample count: {}", samples);
        println!("{'=':<60}");

        // Run CPU version for comparison
        println!("Running CPU Monte Carlo Pi calculation...");
        let (cpu_pi, cpu_time) = cpu_monte_carlo_pi(samples);
        let cpu_error = (cpu_pi - std::f64::consts::PI).abs();
        println!("CPU Pi estimate: {:.6}", cpu_pi);
        println!("CPU time: {:.2}ms", cpu_time);
        println!("CPU error: {:.6}", cpu_error);

        // Run GPU version 1 (parallel with local RNG)
        println!("\nRunning SYCL Monte Carlo Pi calculation (parallel)...");
        let mut gpu_pi = 0.0;
        let mut gpu_time = 0.0;
        let result = unsafe {
            monte_carlo_pi_sycl(
                samples,
                selected_device,
                &mut gpu_pi,
                &mut gpu_time,
            )
        };

        if result == 0 {
            let gpu_error = (gpu_pi - std::f64::consts::PI).abs();
            println!("SYCL Pi estimate: {:.6}", gpu_pi);
            println!("SYCL time: {:.2}ms", gpu_time);
            println!("SYCL error: {:.6}", gpu_error);

            let speedup = cpu_time / gpu_time;
            println!("Speedup vs CPU: {:.2}x", speedup);
        } else {
            println!("✗ SYCL execution failed!");
        }

        // Run GPU version 2 (single task with pre-generated random numbers)
        // Only for smaller sample counts to avoid memory issues
        if samples <= 10_000_000 {
            println!("\nRunning SYCL Monte Carlo Pi calculation (single task)...");
            let mut gpu_pi2 = 0.0;
            let mut gpu_time2 = 0.0;
            let result2 = unsafe {
                monte_carlo_pi_philox_sycl(
                    samples,
                    selected_device,
                    &mut gpu_pi2,
                    &mut gpu_time2,
                )
            };

            if result2 == 0 {
                let gpu_error2 = (gpu_pi2 - std::f64::consts::PI).abs();
                println!("SYCL Pi estimate (v2): {:.6}", gpu_pi2);
                println!("SYCL time (v2): {:.2}ms", gpu_time2);
                println!("SYCL error (v2): {:.6}", gpu_error2);

                let speedup2 = cpu_time / gpu_time2;
                println!("Speedup vs CPU (v2): {:.2}x", speedup2);
            } else {
                println!("✗ SYCL v2 execution failed!");
            }
        }

        // Performance and accuracy summary
        println!("\nSummary for {} samples:", samples);
        println!("  Actual Pi: {:.6}", std::f64::consts::PI);
        if cpu_time > 0.0 {
            let samples_per_sec_cpu = samples as f64 / (cpu_time / 1000.0);
            println!("  CPU: {:.2}M samples/sec", samples_per_sec_cpu / 1_000_000.0);
        }
        if gpu_time > 0.0 {
            let samples_per_sec_gpu = samples as f64 / (gpu_time / 1000.0);
            println!("  SYCL: {:.2}M samples/sec", samples_per_sec_gpu / 1_000_000.0);
        }

        // Convergence analysis
        if samples >= 10_000_000 {
            println!("\nConvergence Analysis:");
            println!("  As sample count increases, estimates should converge to π = {:.6}", std::f64::consts::PI);
            println!("  Monte Carlo error typically decreases as 1/√n where n is sample count");
            let theoretical_error = 1.0 / (samples as f64).sqrt();
            println!("  Theoretical error scale: {:.6}", theoretical_error);
        }
    }

    Ok(())
}