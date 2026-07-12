use anyhow::Result;
use libc::{c_char, c_int};
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
    fn matrix_multiply_sycl(
        a: *mut f32,
        b: *mut f32,
        c: *mut f32,
        size: c_int,
        device_id: c_int,
        elapsed_ms: *mut f64,
    ) -> c_int;
    fn matrix_multiply_optimized_sycl(
        a: *mut f32,
        b: *mut f32,
        c: *mut f32,
        size: c_int,
        device_id: c_int,
        elapsed_ms: *mut f64,
    ) -> c_int;
}

fn cpu_matrix_multiply(a: &[f32], b: &[f32], c: &mut [f32], size: usize) {
    for i in 0..size {
        for j in 0..size {
            let mut sum = 0.0;
            for k in 0..size {
                sum += a[i * size + k] * b[k * size + j];
            }
            c[i * size + j] = sum;
        }
    }
}

fn verify_result(gpu_result: &[f32], cpu_result: &[f32], size: usize) -> bool {
    const EPSILON: f32 = 1e-4;

    for i in 0..size * size {
        if (gpu_result[i] - cpu_result[i]).abs() > EPSILON {
            println!("Mismatch at index {}: GPU={}, CPU={}", i, gpu_result[i], cpu_result[i]);
            return false;
        }
    }
    true
}

fn main() -> Result<()> {
    println!("oneAPI SYCL Matrix Multiplication");
    println!("=================================");

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

    // Test different matrix sizes
    let sizes = vec![256, 512, 1024];

    for size in sizes {
        println!("\n{'=':<50}");
        println!("Matrix size: {}x{}", size, size);
        println!("{'=':<50}");

        // Prepare test matrices
        let mut a: Vec<f32> = vec![1.0; size * size];
        let mut b: Vec<f32> = vec![2.0; size * size];
        let mut c_gpu: Vec<f32> = vec![0.0; size * size];
        let mut c_gpu_opt: Vec<f32> = vec![0.0; size * size];
        let mut c_cpu: Vec<f32> = vec![0.0; size * size];

        // Fill matrices with some pattern for better testing
        for i in 0..size {
            for j in 0..size {
                a[i * size + j] = (i + j) as f32 * 0.01;
                b[i * size + j] = (i * size + j) as f32 * 0.001;
            }
        }

        // Run CPU version for verification
        println!("Running CPU matrix multiplication...");
        let cpu_start = Instant::now();
        cpu_matrix_multiply(&a, &b, &mut c_cpu, size);
        let cpu_duration = cpu_start.elapsed();
        println!("CPU time: {:.2}ms", cpu_duration.as_secs_f64() * 1000.0);

        // Run basic SYCL version
        println!("Running basic SYCL matrix multiplication...");
        let mut gpu_elapsed = 0.0;
        let result = unsafe {
            matrix_multiply_sycl(
                a.as_mut_ptr(),
                b.as_mut_ptr(),
                c_gpu.as_mut_ptr(),
                size as c_int,
                selected_device,
                &mut gpu_elapsed,
            )
        };

        if result == 0 {
            println!("SYCL time: {:.2}ms", gpu_elapsed);

            // Verify results
            if verify_result(&c_gpu, &c_cpu, size) {
                println!("✓ Basic SYCL results verified correctly!");
            } else {
                println!("✗ Basic SYCL verification failed!");
            }

            let speedup = (cpu_duration.as_secs_f64() * 1000.0) / gpu_elapsed;
            println!("Speedup vs CPU: {:.2}x", speedup);
        } else {
            println!("✗ Basic SYCL execution failed!");
        }

        // Run optimized SYCL version for larger matrices
        if size >= 512 {
            println!("Running optimized SYCL matrix multiplication...");
            let mut gpu_opt_elapsed = 0.0;
            let result = unsafe {
                matrix_multiply_optimized_sycl(
                    a.as_mut_ptr(),
                    b.as_mut_ptr(),
                    c_gpu_opt.as_mut_ptr(),
                    size as c_int,
                    selected_device,
                    &mut gpu_opt_elapsed,
                )
            };

            if result == 0 {
                println!("Optimized SYCL time: {:.2}ms", gpu_opt_elapsed);

                if verify_result(&c_gpu_opt, &c_cpu, size) {
                    println!("✓ Optimized SYCL results verified correctly!");
                } else {
                    println!("✗ Optimized SYCL verification failed!");
                }

                let speedup_opt = (cpu_duration.as_secs_f64() * 1000.0) / gpu_opt_elapsed;
                println!("Optimized speedup vs CPU: {:.2}x", speedup_opt);

                if gpu_elapsed > 0.0 {
                    let improvement = gpu_elapsed / gpu_opt_elapsed;
                    println!("Optimization improvement: {:.2}x", improvement);
                }
            } else {
                println!("✗ Optimized SYCL execution failed!");
            }
        }

        // Performance summary
        println!("\nPerformance Summary for {}x{}:", size, size);
        if gpu_elapsed > 0.0 {
            let gflops = (2.0 * size as f64 * size as f64 * size as f64) / (gpu_elapsed * 1e6);
            println!("  GFLOPS (basic): {:.2}", gflops);
        }
    }

    Ok(())
}