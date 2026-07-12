use anyhow::Result;
use ocl::{Platform, Device, Context, Queue, Program, Buffer, Kernel};
use rand::Rng;

const NUM_POINTS: usize = 1_000_000;

fn main() -> Result<()> {
    println!("Monte Carlo Pi Calculation using OpenCL");

    // List available platforms and devices
    let platforms = Platform::list();
    println!("\nFound {} platform(s)", platforms.len());

    for (i, platform) in platforms.iter().enumerate() {
        println!("\nPlatform {}: {}", i, platform.name()?);
        println!("  Vendor: {}", platform.vendor()?);
        println!("  Version: {}", platform.version()?);

        let devices = Device::list_all(platform)?;
        println!("  Found {} device(s)", devices.len());

        for (j, device) in devices.iter().enumerate() {
            println!("  Device {}: {}", j, device.name()?);

            // Use first GPU device found
            let device_type = device.info(ocl::core::DeviceInfo::Type)?;
            if let ocl::core::DeviceInfoResult::Type(dev_type) = device_type {
                if dev_type == ocl::core::DeviceType::GPU {
                    println!("\nUsing device for computation: {}", device.name()?);

                    // Create context
                    let context = Context::builder()
                        .platform(platform.clone())
                        .devices(device.clone())
                        .build()?;

                    // Create queue with explicit properties
                    let queue = Queue::new(&context, device.clone(), None)?;

                    // OpenCL kernel for Monte Carlo Pi calculation
                    let src = r#"
                        __kernel void monte_carlo_pi(
                            __global uint* points_inside
                        ) {
                            uint gid = get_global_id(0);
                            uint points_per_thread = 1000;
                            uint inside = 0;

                            // Use a simple PRNG for demonstration
                            uint seed = gid;
                            for (uint i = 0; i < points_per_thread; i++) {
                                // Generate random x,y between -1 and 1
                                float x = (float)((seed * 1103515245 + 12345) % 0x100000000) / 0x100000000 * 2.0f - 1.0f;
                                seed = (seed * 1103515245 + 12345);
                                float y = (float)((seed * 1103515245 + 12345) % 0x100000000) / 0x100000000 * 2.0f - 1.0f;
                                seed = (seed * 1103515245 + 12345);

                                // Check if point is inside unit circle
                                if (x*x + y*y <= 1.0f) {
                                    inside++;
                                }
                            }
                            points_inside[gid] = inside;
                        }
                    "#;

                    // Create and build program
                    let program = Program::builder()
                        .devices(device)
                        .src(src)
                        .build(&context)?;

                    // Calculate buffer size and create buffer
                    let buffer_len = NUM_POINTS / 1000;

                    // Make sure buffer_len is divisible by the work group size
                    let work_group_size = 64;
                    let adjusted_buffer_len = ((buffer_len + work_group_size - 1) / work_group_size) * work_group_size;

                    let points_inside_buf = Buffer::<u32>::builder()
                        .queue(queue.clone())
                        .flags(ocl::MemFlags::new().read_write())
                        .len(adjusted_buffer_len)
                        .build()?;

                    // Initialize buffer with zeros
                    points_inside_buf.write(&vec![0u32; adjusted_buffer_len]).enq()?;

                    // Create and run kernel
                    let kernel = Kernel::builder()
                        .program(&program)
                        .name("monte_carlo_pi")
                        .queue(queue.clone())
                        .arg(&points_inside_buf)
                        .global_work_size(adjusted_buffer_len)
                        .local_work_size(work_group_size)
                        .build()?;

                    unsafe {
                        kernel.enq()?;
                    }

                    // Read results
                    let mut points_inside = vec![0u32; adjusted_buffer_len];
                    points_inside_buf.read(&mut points_inside).enq()?;

                    // Calculate pi (only use the original buffer_len points)
                    let total_inside: u32 = points_inside[..buffer_len].iter().sum();
                    let total_points = (NUM_POINTS) as f64;
                    let pi_estimate = 4.0 * (total_inside as f64) / total_points;

                    println!("\nResults:");
                    println!("Total points: {}", total_points);
                    println!("Points inside circle: {}", total_inside);
                    println!("Pi estimate: {}", pi_estimate);
                    println!("Actual Pi: {}", std::f64::consts::PI);
                    println!("Error: {:.10}", (pi_estimate - std::f64::consts::PI).abs());

                    // Compare with CPU calculation
                    println!("\nComparing with CPU calculation...");
                    let mut rng = rand::thread_rng();
                    let mut cpu_inside = 0;
                    for _ in 0..NUM_POINTS {
                        let x: f64 = rng.gen_range(-1.0..1.0);
                        let y: f64 = rng.gen_range(-1.0..1.0);
                        if x*x + y*y <= 1.0 {
                            cpu_inside += 1;
                        }
                    }
                    let cpu_pi = 4.0 * (cpu_inside as f64) / total_points;
                    println!("CPU Pi estimate: {}", cpu_pi);
                    println!("CPU Error: {:.10}", (cpu_pi - std::f64::consts::PI).abs());

                    break;
                }
            }
        }
    }

    Ok(())
}
