use anyhow::{Context, Result};
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSBundle, NSString, NSURL};
use objc2_metal::*;
use std::time::Instant;

fn create_metal_device() -> Result<Retained<ProtocolObject<dyn MTLDevice>>> {
    let device = MTLCreateSystemDefaultDevice()
        .context("No Metal device found. Metal requires macOS 10.11+ or iOS 8.0+")?;

    println!("Metal Device: {}", device.name());
    println!("  Max threads per threadgroup: {}", device.maxThreadsPerThreadgroup().width);
    println!("  Supports unified memory: {}", device.hasUnifiedMemory());
    println!("  Supports compute: {}", device.supportsFamily(MTLGPUFamily::Mac2));

    Ok(device)
}

fn compile_shaders(device: &ProtocolObject<dyn MTLDevice>) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
    // Read the Metal shader source code
    let shader_source = include_str!("../shaders/basic_compute.metal");
    let source_string = NSString::from_str(shader_source);

    let library = device
        .newLibraryWithSource_options_error(&source_string, None)
        .context("Failed to compile Metal shaders")?;

    println!("✓ Metal shaders compiled successfully");
    Ok(library)
}

fn create_compute_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    function_name: &str,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    let function_name_ns = NSString::from_str(function_name);
    let function = library
        .newFunctionWithName(&function_name_ns)
        .context("Failed to find compute function")?;

    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .context("Failed to create compute pipeline")?;

    Ok(pipeline)
}

fn run_vector_add_test(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
) -> Result<()> {
    println!("\n--- Vector Addition Test ---");

    let size = 1024;
    let byte_length = (size * std::mem::size_of::<f32>()) as u64;

    // Create input data
    let a_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
    let mut c_data = vec![0.0f32; size];

    // Create Metal buffers
    let a_buffer = device.newBufferWithBytes_length_options(
        a_data.as_ptr() as *const std::ffi::c_void,
        byte_length,
        MTLResourceOptions::StorageModeShared,
    );

    let b_buffer = device.newBufferWithBytes_length_options(
        b_data.as_ptr() as *const std::ffi::c_void,
        byte_length,
        MTLResourceOptions::StorageModeShared,
    );

    let c_buffer = device.newBufferWithLength_options(
        byte_length,
        MTLResourceOptions::StorageModeShared,
    );

    // Create compute pipeline
    let pipeline = create_compute_pipeline(device, library, "vector_add")?;

    // Create command queue and command buffer
    let command_queue = device.newCommandQueue().context("Failed to create command queue")?;
    let command_buffer = command_queue.commandBuffer().context("Failed to create command buffer")?;

    // Create compute command encoder
    let encoder = command_buffer.computeCommandEncoder().context("Failed to create compute encoder")?;
    encoder.setComputePipelineState(&pipeline);
    encoder.setBuffer_offset_atIndex(Some(&a_buffer), 0, 0);
    encoder.setBuffer_offset_atIndex(Some(&b_buffer), 0, 1);
    encoder.setBuffer_offset_atIndex(Some(&c_buffer), 0, 2);

    // Calculate optimal threadgroup size
    let threadgroup_size = MTLSize {
        width: pipeline.maxTotalThreadsPerThreadgroup().min(size as u64),
        height: 1,
        depth: 1,
    };
    let threadgroups = MTLSize {
        width: (size as u64 + threadgroup_size.width - 1) / threadgroup_size.width,
        height: 1,
        depth: 1,
    };

    // Dispatch the compute kernel
    let start = Instant::now();
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threadgroup_size);
    encoder.endEncoding();

    // Commit and wait
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    let duration = start.elapsed();

    // Read results
    let result_ptr = c_buffer.contents() as *const f32;
    unsafe {
        std::ptr::copy_nonoverlapping(result_ptr, c_data.as_mut_ptr(), size);
    }

    // Verify results
    let mut errors = 0;
    for i in 0..size.min(10) {
        let expected = a_data[i] + b_data[i];
        let actual = c_data[i];
        if (actual - expected).abs() > 1e-6 {
            println!("Error at index {}: expected {}, got {}", i, expected, actual);
            errors += 1;
        }
    }

    if errors == 0 {
        println!("✓ Vector addition completed successfully!");
        println!("  Size: {} elements", size);
        println!("  Time: {:.2}ms", duration.as_secs_f64() * 1000.0);
        println!("  First few results: {:?}", &c_data[..5]);
    } else {
        println!("✗ {} errors found in vector addition", errors);
    }

    Ok(())
}

fn run_parallel_reduction_test(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
) -> Result<()> {
    println!("\n--- Parallel Reduction Test ---");

    let size = 1024;
    let threadgroup_size = 256;
    let num_groups = (size + threadgroup_size - 1) / threadgroup_size;

    // Create input data (sum should be size * (size - 1) / 2)
    let input_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let expected_sum: f32 = input_data.iter().sum();

    // Create Metal buffers
    let input_buffer = device.newBufferWithBytes_length_options(
        input_data.as_ptr() as *const std::ffi::c_void,
        (size * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device.newBufferWithLength_options(
        (num_groups * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create compute pipeline
    let pipeline = create_compute_pipeline(device, library, "parallel_sum_reduction")?;

    // Create command queue and command buffer
    let command_queue = device.newCommandQueue().context("Failed to create command queue")?;
    let command_buffer = command_queue.commandBuffer().context("Failed to create command buffer")?;

    // Create compute command encoder
    let encoder = command_buffer.computeCommandEncoder().context("Failed to create compute encoder")?;
    encoder.setComputePipelineState(&pipeline);
    encoder.setBuffer_offset_atIndex(Some(&input_buffer), 0, 0);
    encoder.setBuffer_offset_atIndex(Some(&output_buffer), 0, 1);
    encoder.setThreadgroupMemoryLength_atIndex((threadgroup_size * std::mem::size_of::<f32>()) as u64, 0);

    let threadgroups = MTLSize {
        width: num_groups as u64,
        height: 1,
        depth: 1,
    };
    let threads_per_group = MTLSize {
        width: threadgroup_size as u64,
        height: 1,
        depth: 1,
    };

    // Dispatch the compute kernel
    let start = Instant::now();
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_group);
    encoder.endEncoding();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    let duration = start.elapsed();

    // Read partial results and compute final sum
    let result_ptr = output_buffer.contents() as *const f32;
    let mut partial_sums = vec![0.0f32; num_groups];
    unsafe {
        std::ptr::copy_nonoverlapping(result_ptr, partial_sums.as_mut_ptr(), num_groups);
    }

    let gpu_sum: f32 = partial_sums.iter().sum();
    let error = (gpu_sum - expected_sum).abs();

    println!("  Input size: {} elements", size);
    println!("  Expected sum: {:.1}", expected_sum);
    println!("  GPU sum: {:.1}", gpu_sum);
    println!("  Error: {:.6}", error);
    println!("  Time: {:.2}ms", duration.as_secs_f64() * 1000.0);

    if error < 1e-3 {
        println!("✓ Parallel reduction completed successfully!");
    } else {
        println!("✗ Parallel reduction failed with error: {}", error);
    }

    Ok(())
}

fn main() -> Result<()> {
    println!("Apple Metal Basic Test");
    println!("=====================");

    // Check if we're running on macOS
    #[cfg(not(target_os = "macos"))]
    {
        println!("This example requires macOS with Metal support.");
        return Ok(());
    }

    // Create Metal device
    let device = create_metal_device()?;

    // Compile shaders
    let library = compile_shaders(&device)?;

    // Run tests
    run_vector_add_test(&device, &library)?;
    run_parallel_reduction_test(&device, &library)?;

    println!("\n✓ All Metal compute tests completed!");

    Ok(())
}