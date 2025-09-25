#[cfg(test)]
mod tests {
    use oneapi::sycl::{device, platform, queue, target};

    #[test]
    fn test_gpu_available() {
        let platform = platform::get_platforms()
            .expect("Failed to get platforms")
            .into_iter()
            .find(|p| p.name().contains("Intel"))
            .expect("No Intel platform found");

        let gpu_device = device::get_devices(&platform, target::gpu)
            .expect("Failed to get devices")
            .into_iter()
            .find(|d| d.name().contains("A770"))
            .expect("No Intel Arc A770 GPU found");

        let queue = queue::Queue::new(&gpu_device)
            .expect("Failed to create queue");

        assert!(queue.is_valid());
    }
}
