use ocl::{Buffer, Context, Device, DeviceType, Kernel, Platform, Program, Queue, Result as OclResult};
use rand::Rng;
use std::time::Instant;

const TILE_SIZE: i32 = 32;
const WORK_GROUP_SIZE: usize = TILE_SIZE as usize;
const MAX_RUNTIME_SECS: f64 = 10.0;

// Test different matrix sizes
const MATRIX_SIZES: &[usize] = &[512, 1024, 2048, 4096, 8192];

// OpenCL kernel for tiled matrix multiplication
const KERNEL_SRC: &str = r#"
__kernel void matrix_multiply(
    const int M,
    const int N,
    const int K,
    __global const float* A,
    __global const float* B,
    __global float* C,
    __local float* A_tile,
    __local float* B_tile
) {
    // Get global work-item ID
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    // Get local work-item ID
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    
    // Initialize accumulator
    float acc = 0.0f;
    
    // Loop over tiles
    const int num_tiles = K / TILE_SIZE;
    for (int t = 0; t < num_tiles; t++) {
        // Load one tile of A and B into local memory
        const int tile_row = row;
        const int tile_col = t * TILE_SIZE + local_col;
        if (tile_row < M && tile_col < K) {
            A_tile[local_row * TILE_SIZE + local_col] = A[tile_row * K + tile_col];
        } else {
            A_tile[local_row * TILE_SIZE + local_col] = 0.0f;
        }
        
        const int b_tile_row = t * TILE_SIZE + local_row;
        const int b_tile_col = col;
        if (b_tile_row < K && b_tile_col < N) {
            B_tile[local_row * TILE_SIZE + local_col] = B[b_tile_row * N + b_tile_col];
        } else {
            B_tile[local_row * TILE_SIZE + local_col] = 0.0f;
        }
        
        // Synchronize to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Multiply the tiles and accumulate
        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += A_tile[local_row * TILE_SIZE + k] * B_tile[k * TILE_SIZE + local_col];
        }
        
        // Synchronize before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Store the final result in C
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
"#;

fn get_gpu_device() -> OclResult<(Platform, Device)> {
    let platforms = Platform::list();
    
    for platform in platforms {
        if let Ok(devices) = Device::list(platform, Some(DeviceType::GPU)) {
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

fn matrix_multiply_cpu(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

struct BenchmarkResult {
    size: usize,
    gpu_time: f64,
    cpu_time: f64,
    max_diff: f32,
    gpu_tflops: f64,
    cpu_gflops: f64,
    speedup: f64,
}

fn run_benchmark(size: usize, context: &Context, device: Device, queue: Queue) -> OclResult<Option<BenchmarkResult>> {
    println!("\nTesting size: {} x {} x {}", size, size, size);
    
    let mut rng = rand::thread_rng();
    let a: Vec<f32> = (0..size*size).map(|_| rng.random::<f32>()).collect();
    let b: Vec<f32> = (0..size*size).map(|_| rng.random::<f32>()).collect();
    let mut c_gpu = vec![0.0f32; size*size];
    let mut c_cpu = vec![0.0f32; size*size];

    // Create program
    let program = Program::builder()
        .src(KERNEL_SRC)
        .devices(device)
        .cmplr_def("TILE_SIZE", TILE_SIZE)
        .cmplr_opt("-cl-fast-relaxed-math -cl-mad-enable -cl-denorms-are-zero")
        .build(&context)?;

    // Create buffers
    let a_buf = Buffer::builder()
        .queue(queue.clone())
        .flags(ocl::MemFlags::new().read_only().copy_host_ptr())
        .len(size*size)
        .copy_host_slice(&a)
        .build()?;

    let b_buf = Buffer::builder()
        .queue(queue.clone())
        .flags(ocl::MemFlags::new().read_only().copy_host_ptr())
        .len(size*size)
        .copy_host_slice(&b)
        .build()?;

    let c_buf = Buffer::builder()
        .queue(queue.clone())
        .flags(ocl::MemFlags::new().write_only())
        .len(size*size)
        .build()?;

    // Create kernel
    let kernel = Kernel::builder()
        .program(&program)
        .name("matrix_multiply")
        .queue(queue.clone())
        .global_work_size([size, size])
        .local_work_size([WORK_GROUP_SIZE, WORK_GROUP_SIZE])
        .arg(size as i32)
        .arg(size as i32)
        .arg(size as i32)
        .arg(&a_buf)
        .arg(&b_buf)
        .arg(&c_buf)
        .arg_local::<f32>(WORK_GROUP_SIZE * WORK_GROUP_SIZE)
        .arg_local::<f32>(WORK_GROUP_SIZE * WORK_GROUP_SIZE)
        .build()?;

    // Run GPU implementation
    println!("Running GPU implementation...");
    let start = Instant::now();
    unsafe {
        kernel.enq()?;
    }
    queue.finish()?;
    c_buf.read(&mut c_gpu).enq()?;
    let gpu_time = start.elapsed().as_secs_f64();
    println!("GPU Time: {:.3} ms", gpu_time * 1000.0);

    if gpu_time > MAX_RUNTIME_SECS {
        println!("GPU time exceeded {} seconds, skipping larger sizes", MAX_RUNTIME_SECS);
        return Ok(None);
    }

    // Run CPU implementation
    println!("Running CPU implementation...");
    let start = Instant::now();
    matrix_multiply_cpu(&a, &b, &mut c_cpu, size, size, size);
    let cpu_time = start.elapsed().as_secs_f64();
    println!("CPU Time: {:.3} ms", cpu_time * 1000.0);

    if cpu_time > MAX_RUNTIME_SECS {
        println!("CPU time exceeded {} seconds, skipping larger sizes", MAX_RUNTIME_SECS);
        return Ok(None);
    }

    // Calculate metrics
    let mut max_diff = 0.0f32;
    for i in 0..size*size {
        max_diff = max_diff.max((c_gpu[i] - c_cpu[i]).abs());
    }

    let flops = 2.0 * (size * size * size) as f64;
    let gpu_tflops = flops / gpu_time / 1e12;
    let cpu_gflops = flops / cpu_time / 1e9;
    let speedup = cpu_time / gpu_time;

    Ok(Some(BenchmarkResult {
        size,
        gpu_time,
        cpu_time,
        max_diff,
        gpu_tflops,
        cpu_gflops,
        speedup,
    }))
}

fn main() -> OclResult<()> {
    // Get GPU device
    let (platform, device) = get_gpu_device()?;
    
    // Create context and queue
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;

    let queue = Queue::new(&context, device, None)?;

    println!("Starting matrix multiplication benchmark");
    println!("Testing sizes: {:?}", MATRIX_SIZES);
    println!("Maximum runtime per test: {} seconds", MAX_RUNTIME_SECS);

    let mut results = Vec::new();

    for &size in MATRIX_SIZES {
        match run_benchmark(size, &context, device, queue.clone())? {
            Some(result) => results.push(result),
            None => break,
        }
    }

    // Print summary
    println!("\nBenchmark Summary:");
    println!("Size\tGPU(ms)\tCPU(ms)\tMax Diff\tGPU TFLOPS\tCPU GFLOPS\tSpeedup");
    println!("--------------------------------------------------------------------");
    for result in results {
        println!("{}x{}\t{:.1}\t{:.1}\t{:.2e}\t{:.2}\t{:.2}\t{:.2}x",
            result.size, result.size,
            result.gpu_time * 1000.0,
            result.cpu_time * 1000.0,
            result.max_diff,
            result.gpu_tflops,
            result.cpu_gflops,
            result.speedup);
    }

    Ok(())
}
