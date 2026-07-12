use anyhow::Result;
use std::fs::File;
use std::io::{Read, Write};
use std::process::Command;

fn main() -> Result<()> {
    println!("oneAPI SYCL Basic Test (Standalone Architecture)");
    println!("================================================");

    // List devices first
    match Command::new("./sycl_device_info").output() {
        Ok(output) if output.status.success() => {
            print!("{}", String::from_utf8_lossy(&output.stdout));
        }
        _ => {
            println!("Detecting SYCL devices using standalone architecture...");
            println!("Available devices will be shown when SYCL runner executes.");
        }
    }

    let selected_device = 0; // Use first device

    // Prepare test data
    let size = 16;
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

    println!("\nInput data A: {:?}", a);
    println!("Input data B: {:?}", b);

    // Write input data to binary file
    let mut input_file = File::create("input.bin")?;
    input_file.write_all(&(size as i32).to_le_bytes())?;
    for val in &a {
        input_file.write_all(&val.to_le_bytes())?;
    }
    for val in &b {
        input_file.write_all(&val.to_le_bytes())?;
    }
    drop(input_file);

    // Run SYCL computation
    let output = Command::new("./sycl_runner")
        .args(["vector_add", &selected_device.to_string(), "input.bin", "output.bin"])
        .output()?;

    if !output.status.success() {
        println!("Error executing SYCL kernel!");
        println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        return Ok(());
    }

    println!("{}", String::from_utf8_lossy(&output.stdout));

    // Read result from binary file
    let mut output_file = File::open("output.bin")?;
    let mut buffer = Vec::new();
    output_file.read_to_end(&mut buffer)?;

    let mut c = Vec::new();
    for chunk in buffer.chunks(4) {
        if chunk.len() == 4 {
            let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
            c.push(f32::from_le_bytes(bytes));
        }
    }

    println!("Output data C: {:?}", c);

    // Verify results
    let mut all_correct = true;
    for i in 0..size {
        let expected = a[i] + b[i];
        if i < c.len() && (c[i] - expected).abs() > 1e-6 {
            println!("Verification failed at index {}: expected {}, got {}", i, expected, c[i]);
            all_correct = false;
        }
    }

    if all_correct && c.len() == size {
        println!("\n✓ Vector addition completed successfully!");
        println!("✓ Results verified correctly!");
        println!("✓ oneAPI SYCL is working properly!");
    } else {
        println!("\n✗ Verification failed!");
    }

    // Cleanup
    std::fs::remove_file("input.bin").ok();
    std::fs::remove_file("output.bin").ok();

    Ok(())
}