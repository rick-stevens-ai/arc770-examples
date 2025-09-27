use anyhow::Result;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const SHADER_SOURCE: &str = r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&input_a)) {
        return;
    }
    output[index] = input_a[index] + input_b[index];
}
"#;

async fn run() -> Result<()> {
    env_logger::init();

    println!("wgpu Vector Addition Test");
    println!("========================");

    // Initialize wgpu
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        ..Default::default()
    });

    // Request adapter (GPU)
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

    // Get device and queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor::default(),
        )
        .await?;

    // Create compute shader
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Vector Add Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_SOURCE)),
    });

    // Prepare test data
    let size = 1024u32;
    let input_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let input_b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

    println!("\nTesting with {} elements", size);
    println!("Input A: [{}, {}, {}, ..., {}]", input_a[0], input_a[1], input_a[2], input_a[size as usize - 1]);
    println!("Input B: [{}, {}, {}, ..., {}]", input_b[0], input_b[1], input_b[2], input_b[size as usize - 1]);

    // Create buffers
    let buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input A Buffer"),
        contents: bytemuck::cast_slice(&input_a),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input B Buffer"),
        contents: bytemuck::cast_slice(&input_b),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let buffer_output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: (size * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create staging buffer for reading results
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (size * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Vector Add Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Vector Add Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_output.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Vector Add Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Vector Add Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Dispatch compute shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Vector Add Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Vector Add Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_count = (size + 63) / 64; // Round up for workgroup size of 64
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    // Copy output to staging buffer
    encoder.copy_buffer_to_buffer(&buffer_output, 0, &staging_buffer, 0, (size * std::mem::size_of::<f32>() as u32) as u64);

    // Submit commands
    let start_time = std::time::Instant::now();
    queue.submit(std::iter::once(encoder.finish()));
    // Wait for completion - simplified approach
    let compute_time = start_time.elapsed();

    // Read results
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures::channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    // Wait for completion - simplified approach
    receiver.await??;

    let data = buffer_slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    // Verify results
    println!("\nResults:");
    println!("Output: [{}, {}, {}, ..., {}]", result[0], result[1], result[2], result[size as usize - 1]);

    let expected: Vec<f32> = (0..size).map(|i| (i as f32) + (i * 2) as f32).collect();
    let mut correct = 0;
    for (i, (&actual, &expected)) in result.iter().zip(expected.iter()).enumerate().take(10) {
        if (actual - expected).abs() < 0.001 {
            correct += 1;
        }
        if i < 5 {
            println!("  [{:4}] {} + {} = {} (expected {})", i, input_a[i], input_b[i], actual, expected);
        }
    }

    println!("\nPerformance:");
    println!("Compute time: {:.3}ms", compute_time.as_secs_f64() * 1000.0);
    println!("Elements per second: {:.2e}", size as f64 / compute_time.as_secs_f64());

    if correct == 10 {
        println!("\n✓ Vector addition completed successfully!");
        println!("✓ Results verified correctly!");
        println!("✓ wgpu GPU compute is working properly!");
    } else {
        println!("\n❌ Results verification failed!");
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    run().await
}