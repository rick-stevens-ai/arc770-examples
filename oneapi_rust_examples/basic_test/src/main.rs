use anyhow::Result;
use libc::{c_char, c_int};
use std::ffi::CStr;

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
    fn simple_vector_add(
        a: *mut f32,
        b: *mut f32,
        c: *mut f32,
        size: c_int,
        device_id: c_int,
    ) -> c_int;
}

fn main() -> Result<()> {
    println!("oneAPI SYCL Basic Test");
    println!("======================");

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

    // Find a CPU device first to test if this is GPU-specific issue
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

            // Try CPU device temporarily to debug kernel issue
            if !device_info.is_gpu && vendor.to_lowercase().contains("intel") {
                selected_device = i;
                break;
            }
        }
    }

    // Use first device if no GPU found
    if selected_device == -1 {
        selected_device = 0;
    }

    println!("\nUsing device {} for computation", selected_device);

    // Prepare test data
    let size = 16;
    let mut a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let mut b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
    let mut c: Vec<f32> = vec![0.0; size];

    println!("\nInput data A: {:?}", a);
    println!("Input data B: {:?}", b);

    // Perform vector addition using SYCL
    let result = unsafe {
        simple_vector_add(
            a.as_mut_ptr(),
            b.as_mut_ptr(),
            c.as_mut_ptr(),
            size as c_int,
            selected_device,
        )
    };

    if result != 0 {
        println!("Error executing SYCL kernel!");
        return Ok(());
    }

    println!("Output data C: {:?}", c);

    // Verify results
    let mut all_correct = true;
    for i in 0..size {
        let expected = a[i] + b[i];
        if (c[i] - expected).abs() > 1e-6 {
            println!("Verification failed at index {}: expected {}, got {}", i, expected, c[i]);
            all_correct = false;
        }
    }

    if all_correct {
        println!("\n✓ Vector addition completed successfully!");
        println!("✓ Results verified correctly!");
    } else {
        println!("\n✗ Verification failed!");
    }

    Ok(())
}