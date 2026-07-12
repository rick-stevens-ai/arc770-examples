use anyhow::Result;
use ocl::{Platform, Device, Context, Queue, Program, Buffer, ProQue};

fn main() -> Result<()> {
    println!("Detecting OpenCL devices...");
    
    // Get available platforms
    let platforms = Platform::list();
    println!("Found {} platform(s)", platforms.len());
    
    for (i, platform) in platforms.iter().enumerate() {
        println!("\nPlatform {}: {}", i, platform.name()?);
        println!("  Vendor: {}", platform.vendor()?);
        println!("  Version: {}", platform.version()?);
        
        // Get GPU devices for this platform
        let devices = Device::list_all(platform)?;
        println!("  Found {} device(s)", devices.len());
        
        for (j, device) in devices.iter().enumerate() {
            println!("  Device {}: {}", j, device.name()?);
            println!("    Vendor: {}", device.vendor()?);
            println!("    Version: {}", device.version()?);
            
            // Try to create a context and queue for this device
            let context = Context::builder()
                .platform(platform.clone())
                .devices(device.clone())
                .build()?;
            
            let queue = Queue::new(&context, *device, None)?;
            println!("    Successfully created queue");

            // Create test matrices
            let size = 1024;
            let total_elements = size * size;
            let mut a = vec![1.0f32; total_elements];
            let mut b = vec![2.0f32; total_elements];
            let mut c = vec![0.0f32; total_elements];

            // Create program and kernel
            let src = r#"
                __kernel void matrix_multiply(
                    __global const float* a,
                    __global const float* b,
                    __global float* c,
                    const int size)
                {
                    int row = get_global_id(0);
                    int col = get_global_id(1);
                    
                    float sum = 0.0f;
                    for (int k = 0; k < size; k++) {
                        sum += a[row * size + k] * b[k * size + col];
                    }
                    c[row * size + col] = sum;
                }
            "#;

            let pro_que = ProQue::builder()
                .context(context)
                .device(device.clone())
                .src(src)
                .dims([size, size])
                .build()?;

            let a_buf = Buffer::builder()
                .queue(pro_que.queue().clone())
                .flags(ocl::MemFlags::new().read_only())
                .len(total_elements)
                .copy_host_slice(&a)
                .build()?;

            let b_buf = Buffer::builder()
                .queue(pro_que.queue().clone())
                .flags(ocl::MemFlags::new().read_only())
                .len(total_elements)
                .copy_host_slice(&b)
                .build()?;

            let c_buf = Buffer::builder()
                .queue(pro_que.queue().clone())
                .flags(ocl::MemFlags::new().write_only())
                .len(total_elements)
                .build()?;

            println!("Starting matrix multiplication...");
            let start = std::time::Instant::now();

            let kernel = pro_que.kernel_builder("matrix_multiply")
                .arg(&a_buf)
                .arg(&b_buf)
                .arg(&c_buf)
                .arg(size as i32)
                .build()?;

            unsafe {
                kernel.enq()?;
            }

            // Read results
            c_buf.read(&mut c).enq()?;
            pro_que.queue().finish()?;

            let duration = start.elapsed();
            println!("Matrix multiplication completed in {:?}", duration);

            // Verify a few results
            println!("Sample results (first few elements):");
            for i in 0..5 {
                println!("c[{}] = {}", i, c[i]);
            }
        }
    }
    
    Ok(())
}
