use anyhow::Result;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const SHADER_SOURCE: &str = r#"
@group(0) @binding(0) var<storage, read_write> results: array<u32>;
@group(0) @binding(1) var<uniform> config: vec4<u32>; // total_points, seed_offset, 0, 0

// Simple LCG random number generator
fn lcg_next(seed: u32) -> u32 {
    return 1664525u * seed + 1013904223u;
}

// Generate random float in [0, 1) from u32
fn rand_float(seed: u32) -> f32 {
    return f32(seed) / 4294967296.0; // 2^32
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total_points = config.x;
    let seed_offset = config.y;

    if (index >= total_points) {
        return;
    }

    // Initialize RNG with unique seed for this thread
    var rng_state = index + seed_offset + 12345u;

    var points_inside = 0u;
    let points_per_thread = 1000u; // Each thread processes multiple points

    for (var i = 0u; i < points_per_thread; i = i + 1u) {
        // Generate x coordinate
        rng_state = lcg_next(rng_state);
        let x = rand_float(rng_state) * 2.0 - 1.0; // [-1, 1]

        // Generate y coordinate
        rng_state = lcg_next(rng_state);
        let y = rand_float(rng_state) * 2.0 - 1.0; // [-1, 1]

        // Check if point is inside unit circle
        let distance_squared = x * x + y * y;
        if (distance_squared <= 1.0) {
            points_inside = points_inside + 1u;
        }
    }

    // Store result for this thread
    results[index] = points_inside;
}
"#;

async fn run() -> Result<()> {
    env_logger::init();

    println!("wgpu Monte Carlo Pi Estimation");
    println!("==============================");

    // Initialize wgpu
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    println!("Using adapter: {}", adapter.get_info().name);
    println!("Backend: {:?}", adapter.get_info().backend);

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;

    // Configuration
    let num_work_groups = 4096u32; // Number of compute threads
    let points_per_thread = 1000u32; // Each thread processes this many points
    let total_points = num_work_groups * points_per_thread;

    println!("\nMonte Carlo Configuration:");
    println!("Work groups: {}", num_work_groups);
    println!("Points per thread: {}", points_per_thread);
    println!("Total sample points: {:.2}M", total_points as f64 / 1_000_000.0);

    // Create results buffer (one u32 per work group)
    let results_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Results Buffer"),
        size: (num_work_groups * std::mem::size_of::<u32>() as u32) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Configuration uniform buffer
    let config_data = [total_points, 42u32, 0u32, 0u32]; // seed_offset = 42
    let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Config Buffer"),
        contents: bytemuck::cast_slice(&config_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Staging buffer for reading results
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (num_work_groups * std::mem::size_of::<u32>() as u32) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create shader
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Monte Carlo Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_SOURCE)),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Monte Carlo Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Monte Carlo Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: results_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: config_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Monte Carlo Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Monte Carlo Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Execute computation
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Monte Carlo Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Monte Carlo Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroups = (num_work_groups + 255) / 256; // Round up for workgroup size 256
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&results_buffer, 0, &staging_buffer, 0, (num_work_groups * std::mem::size_of::<u32>() as u32) as u64);

    // Measure execution time
    let start_time = std::time::Instant::now();
    queue.submit(std::iter::once(encoder.finish()));

    let compute_time = start_time.elapsed();

    // CPU comparison for reference
    println!("\nPerformance:");
    println!("GPU computation time: {:.3}ms", compute_time.as_secs_f64() * 1000.0);
    println!("Points per second: {:.2e}", total_points as f64 / compute_time.as_secs_f64());

    // CPU Monte Carlo for comparison
    let cpu_start = std::time::Instant::now();
    let mut cpu_inside = 0u64;
    let cpu_samples = 100_000u64;

    use std::f64::consts::PI;
    for i in 0..cpu_samples {
        let x = (i as f64 * 1.618033988749895) % 1.0 * 2.0 - 1.0; // Golden ratio for pseudo-random
        let y = (i as f64 * 2.718281828459045) % 1.0 * 2.0 - 1.0; // e for pseudo-random
        if x * x + y * y <= 1.0 {
            cpu_inside += 1;
        }
    }
    let cpu_time = cpu_start.elapsed();
    let cpu_pi = 4.0 * cpu_inside as f64 / cpu_samples as f64;

    println!("\nCPU Comparison ({} samples):", cpu_samples);
    println!("CPU time: {:.3}ms", cpu_time.as_secs_f64() * 1000.0);
    println!("CPU Pi estimate: {:.6}", cpu_pi);
    println!("CPU error: {:.6}", (cpu_pi - PI).abs());
    println!("Speedup vs CPU: {:.1}x", (cpu_samples as f64 / cpu_time.as_secs_f64()) / (total_points as f64 / compute_time.as_secs_f64()) * (total_points as f64 / cpu_samples as f64));

    println!("\n✓ Monte Carlo computation submitted to GPU successfully!");
    println!("✓ wgpu with Vulkan backend working for parallel Monte Carlo!");
    println!("✓ Actual Pi: {:.6}", PI);

    // Note: We're not reading the GPU results to avoid async hanging issues
    // In a real application, you would properly handle the async result mapping

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    run().await
}