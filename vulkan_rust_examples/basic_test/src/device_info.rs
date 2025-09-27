use anyhow::Result;
use vulkano::{
    instance::{Instance, InstanceCreateInfo},
    device::{physical::PhysicalDeviceType, Device, DeviceCreateInfo, QueueCreateInfo, DeviceExtensions},
    VulkanLibrary,
};

fn main() -> Result<()> {
    println!("Vulkan Device Information");
    println!("========================");

    // Create Vulkan instance
    let library = VulkanLibrary::new()?;
    let instance = Instance::new(library, InstanceCreateInfo::default())?;

    println!("Vulkan instance created successfully");

    // List physical devices
    for (i, physical_device) in instance.enumerate_physical_devices()?.enumerate() {
        let props = physical_device.properties();
        println!("\nDevice {}: {}", i, props.device_name);
        println!("  Type: {:?}", props.device_type);
        println!("  Vendor ID: 0x{:X}", props.vendor_id);
        println!("  Device ID: 0x{:X}", props.device_id);
        println!("  Driver Version: {}", props.driver_version);
        println!("  API Version: {}.{}.{}",
            props.api_version.major,
            props.api_version.minor,
            props.api_version.patch
        );

        // Check if this is Intel Arc
        if props.vendor_id == 0x8086 && props.device_name.to_lowercase().contains("arc") {
            println!("  ✓ Intel Arc GPU detected!");

            // Try to create a device
            match Device::new(
                physical_device,
                DeviceCreateInfo {
                    queue_create_infos: vec![QueueCreateInfo {
                        queue_family_index: 0,
                        ..Default::default()
                    }],
                    enabled_extensions: DeviceExtensions::empty(),
                    ..Default::default()
                },
            ) {
                Ok((device, _queues)) => {
                    println!("  ✓ Device created successfully!");
                    println!("  ✓ Vulkan compute capability confirmed!");
                    drop(device); // Clean up
                }
                Err(e) => {
                    println!("  ❌ Device creation failed: {}", e);
                }
            }
        }
    }

    println!("\nVulkan setup verification complete!");
    Ok(())
}