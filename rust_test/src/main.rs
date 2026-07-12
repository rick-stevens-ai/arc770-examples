use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
    kernel::Kernel,
    memory::{Buffer, CL_MEM_READ_WRITE},
    platform::get_platforms,
    program::Program,
    types::CL_TRUE,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Detecting OpenCL platforms and devices...");

    // Get available platforms
    let platforms = get_platforms()?;
    println!("Found {} platform(s)", platforms.len());

    let mut target_platform = None;
    let mut target_device = None;

    // Find the Intel GPU platform and device
    for platform in platforms.iter() {
        let name = platform.name()?;
        let vendor = platform.vendor()?;
        println!("\nPlatform: {} ({})", name, vendor);

        if name.contains("Graphics") {
            let devices = platform.get_devices(CL_DEVICE_TYPE_GPU)?;
            println!("Found {} GPU device(s)", devices.len());

            for device_id in devices.iter() {
                let device = Device::new(*device_id);
                let device_name = device.name()?;
                println!("  Device: {}", device_name);

                if device_name.contains("Arc") {
                    target_platform = Some(platform);
                    target_device = Some((*device_id, device_name));
                    break;
                }
            }
        }
    }

    // Use the found platform and device
    if let (Some(platform), Some((device_id, device_name))) = (target_platform, target_device) {
        println!("\nTesting computation on: {}", device_name);

        let device = Device::new(device_id);
        let context = Context::from_device(&device)?;
        
        // Create command queue with properties
        let properties = CL_QUEUE_PROFILING_ENABLE;
        let queue = unsafe {
            CommandQueue::create_with_properties(
                &context,
                device_id,
                properties,
                0
            )?
        };

        // Initialize data
        let data_size = 16;
        let src_data: Vec<f32> = (0..data_size).map(|x| x as f32).collect();
        println!("Input data: {:?}", src_data);

        // Create and initialize buffer
        let mut buffer = unsafe {
            Buffer::<f32>::create(
                &context,
                CL_MEM_READ_WRITE,
                data_size,
                std::ptr::null_mut(),
            )?
        };

        // Write initial data
        unsafe {
            queue.enqueue_write_buffer(
                &mut buffer,
                CL_TRUE,
                0,
                &src_data,
                &[],
            )?;
        }

        // Create and build program
        let program_source = r#"
            __kernel void multiply_by_two(__global float* data) {
                int idx = get_global_id(0);
                data[idx] = data[idx] * 2.0f;
            }
        "#;

        let program = Program::create_and_build_from_source(
            &context,
            program_source,
            "",
        )?;

        let kernel = Kernel::create(&program, "multiply_by_two")?;
        
        // Set kernel argument
        unsafe {
            kernel.set_arg(0, &buffer)?;
        }

        // Execute kernel
        unsafe {
            queue.enqueue_nd_range_kernel(
                kernel.get(),
                1,
                std::ptr::null(),
                &data_size,
                std::ptr::null(),
                &[],
            )?;
        }

        // Read results
        let mut result_data = vec![0.0f32; data_size];
        unsafe {
            queue.enqueue_read_buffer(
                &buffer,
                CL_TRUE,
                0,
                &mut result_data,
                &[],
            )?;
        }

        println!("Output data: {:?}", result_data);
    } else {
        println!("No Intel Arc GPU found!");
    }

    Ok(())
}
