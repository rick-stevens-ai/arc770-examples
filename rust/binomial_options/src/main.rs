use ocl::{Buffer, Context, Device, DeviceType, Kernel, Platform, Program, Queue, Result as OclResult};

const NUM_STEPS: usize = 2048;        // Number of time steps
const NUM_OPTIONS: usize = 262144;    // Number of options to price in parallel
const WORK_GROUP_SIZE: usize = 512;   // Work group size optimized for GPU

// Optimized OpenCL kernel for binomial option pricing
const KERNEL_SRC: &str = r#"
#define OPTION_BLOCK_SIZE 32  // Process multiple options per work group

__kernel void binomial_option_pricer(
    __global const float* spot_prices,    // Current stock prices
    __global const float* strikes,        // Strike prices
    __global const float* rates,          // Risk-free rates
    __global const float* volatilities,   // Volatilities
    __global const float* maturities,     // Time to maturity
    __global const int* is_calls,         // Option type (1 for call, 0 for put)
    __global float* option_prices,        // Output option prices
    const int num_steps                   // Number of time steps
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);
    
    // Load parameters for this option
    float S = spot_prices[gid];      // Spot price
    float K = strikes[gid];          // Strike price
    float r = rates[gid];            // Risk-free rate
    float v = volatilities[gid];     // Volatility
    float T = maturities[gid];       // Time to maturity
    int is_call = is_calls[gid];     // Option type
    
    float dt = T / num_steps;
    float u = exp(v * sqrt(dt));     // Up factor
    float d = 1.0f / u;              // Down factor
    float p = (exp(r * dt) - d) / (u - d); // Risk-neutral probability
    float disc = exp(-r * dt);       // Discount factor
    
    // Use local memory for better performance
    __local float shared_prices[(OPTION_BLOCK_SIZE + 1) * 2];
    int shared_idx = lid * 2;
    
    // Initialize values at terminal nodes
    float spot_up = S * pow(u, num_steps - lid);
    float spot_down = spot_up * pow(d, lid);
    
    // Store initial values in shared memory
    shared_prices[shared_idx] = is_call ? max(0.0f, spot_up - K) : max(0.0f, K - spot_up);
    shared_prices[shared_idx + 1] = is_call ? max(0.0f, spot_down - K) : max(0.0f, K - spot_down);
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Backward induction
    for (int step = num_steps - 1; step >= 0; step--) {
        if (lid < step) {
            shared_prices[shared_idx] = disc * (p * shared_prices[shared_idx] + 
                                              (1.0f - p) * shared_prices[shared_idx + 1]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result for this option
    if (lid == 0) {
        option_prices[gid] = shared_prices[0];
    }
}
"#;

fn get_gpu_device() -> OclResult<(Platform, Device)> {
    let platforms = Platform::list();
    
    for platform in platforms {
        println!("Checking platform: {}", platform.name()?);
        
        // Get all GPU devices for this platform
        if let Ok(devices) = Device::list(platform, Some(DeviceType::GPU)) {
            // Return the first GPU device found
            if let Some(device) = devices.first() {
                println!("Selected GPU platform: {}", platform.name()?);
                println!("Selected GPU device: {}", device.name()?);
                println!("Device vendor: {}", device.vendor()?);
                println!("Device version: {}", device.version()?);
                println!("Device max work group size: {}", device.max_wg_size()?);
                
                return Ok((platform, device.clone()));
            }
        }
    }
    
    Err("No GPU device found".into())
}

fn main() -> OclResult<()> {
    // Get GPU platform and device
    let (platform, device) = get_gpu_device()?;

    // Create context and queue
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;

    let queue = Queue::new(&context, device, None)?;

    // Create program with build options for optimization
    let program = Program::builder()
        .src(KERNEL_SRC)
        .devices(device)
        .cmplr_opt("-cl-fast-relaxed-math")  // Enable fast math optimizations
        .build(&context)?;

    // Generate test data
    let spot_prices: Vec<f32> = (0..NUM_OPTIONS).map(|_| rand::random::<f32>() * 100.0 + 50.0).collect();
    let strikes: Vec<f32> = (0..NUM_OPTIONS).map(|_| rand::random::<f32>() * 100.0 + 50.0).collect();
    let rates: Vec<f32> = (0..NUM_OPTIONS).map(|_| rand::random::<f32>() * 0.05 + 0.01).collect();
    let volatilities: Vec<f32> = (0..NUM_OPTIONS).map(|_| rand::random::<f32>() * 0.3 + 0.1).collect();
    let maturities: Vec<f32> = (0..NUM_OPTIONS).map(|_| rand::random::<f32>() * 2.0 + 0.1).collect();
    let is_calls: Vec<i32> = (0..NUM_OPTIONS).map(|_| if rand::random::<bool>() { 1 } else { 0 }).collect();

    // Create buffers with optimal flags for GPU
    let spot_buf = Buffer::builder()
        .queue(queue.clone())
        .flags(ocl::MemFlags::new().read_only().copy_host_ptr())
        .len(NUM_OPTIONS)
        .copy_host_slice(&spot_prices)
        .build()?;

    let strike_buf = Buffer::builder()
        .queue(queue.clone())
        .flags(ocl::MemFlags::new().read_only().copy_host_ptr())
        .len(NUM_OPTIONS)
        .copy_host_slice(&strikes)
        .build()?;

    let rate_buf = Buffer::builder()
        .queue(queue.clone())
        .flags(ocl::MemFlags::new().read_only().copy_host_ptr())
        .len(NUM_OPTIONS)
        .copy_host_slice(&rates)
        .build()?;

    let vol_buf = Buffer::builder()
        .queue(queue.clone())
        .flags(ocl::MemFlags::new().read_only().copy_host_ptr())
        .len(NUM_OPTIONS)
        .copy_host_slice(&volatilities)
        .build()?;

    let maturity_buf = Buffer::builder()
        .queue(queue.clone())
        .flags(ocl::MemFlags::new().read_only().copy_host_ptr())
        .len(NUM_OPTIONS)
        .copy_host_slice(&maturities)
        .build()?;

    let is_call_buf = Buffer::builder()
        .queue(queue.clone())
        .flags(ocl::MemFlags::new().read_only().copy_host_ptr())
        .len(NUM_OPTIONS)
        .copy_host_slice(&is_calls)
        .build()?;

    let price_buf = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(ocl::MemFlags::new().write_only())
        .len(NUM_OPTIONS)
        .build()?;

    // Create kernel with optimized work group size
    let kernel = Kernel::builder()
        .program(&program)
        .name("binomial_option_pricer")
        .queue(queue.clone())
        .global_work_size(NUM_OPTIONS)
        .local_work_size(WORK_GROUP_SIZE)
        .arg(&spot_buf)
        .arg(&strike_buf)
        .arg(&rate_buf)
        .arg(&vol_buf)
        .arg(&maturity_buf)
        .arg(&is_call_buf)
        .arg(&price_buf)
        .arg(NUM_STEPS as i32)
        .build()?;

    // Start timing
    let start = std::time::Instant::now();

    // Execute kernel and wait for completion
    unsafe {
        kernel.enq()?;
    }
    queue.finish()?;

    // Read results
    let mut prices = vec![0.0f32; NUM_OPTIONS];
    price_buf.read(&mut prices).enq()?;
    queue.finish()?;

    let elapsed = start.elapsed();

    // Print sample results and statistics
    println!("\nResults for first 5 options:");
    println!("Option\tType\tSpot\tStrike\tRate\tVol\tMaturity\tPrice");
    for i in 0..5 {
        println!("{}\t{}\t{:.2}\t{:.2}\t{:.3}\t{:.3}\t{:.3}\t{:.3}",
            i,
            if is_calls[i] == 1 { "Call" } else { "Put" },
            spot_prices[i],
            strikes[i],
            rates[i],
            volatilities[i],
            maturities[i],
            prices[i]
        );
    }

    println!("\nPerformance:");
    println!("Total options priced: {}", NUM_OPTIONS);
    println!("Time steps per option: {}", NUM_STEPS);
    println!("Total execution time: {:.3} ms", elapsed.as_secs_f64() * 1000.0);
    println!("Options per second: {:.2e}", NUM_OPTIONS as f64 / elapsed.as_secs_f64());

    Ok(())
}
