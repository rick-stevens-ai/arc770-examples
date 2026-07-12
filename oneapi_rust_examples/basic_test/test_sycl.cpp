#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

int main() {
    try {
        // Get all devices
        auto devices = sycl::device::get_devices();
        std::cout << "Found " << devices.size() << " SYCL devices" << std::endl;

        if (devices.empty()) {
            std::cout << "No SYCL devices found!" << std::endl;
            return 1;
        }

        // Use the first device
        auto device = devices[0];
        sycl::queue q(device);

        std::cout << "Using device: " << device.get_info<sycl::info::device::name>() << std::endl;

        // Simple vector addition test
        const int size = 16;
        std::vector<float> a(size), b(size), c(size);

        // Initialize data
        for (int i = 0; i < size; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(i * 2);
            c[i] = 0.0f;
        }

        // Use USM
        float* usm_a = sycl::malloc_device<float>(size, q);
        float* usm_b = sycl::malloc_device<float>(size, q);
        float* usm_c = sycl::malloc_device<float>(size, q);

        // Copy data to device
        q.memcpy(usm_a, a.data(), size * sizeof(float)).wait();
        q.memcpy(usm_b, b.data(), size * sizeof(float)).wait();

        // Submit kernel
        q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
            usm_c[idx] = usm_a[idx] + usm_b[idx];
        }).wait();

        // Copy result back
        q.memcpy(c.data(), usm_c, size * sizeof(float)).wait();

        // Clean up
        sycl::free(usm_a, q);
        sycl::free(usm_b, q);
        sycl::free(usm_c, q);

        // Verify results
        bool success = true;
        for (int i = 0; i < size; ++i) {
            float expected = a[i] + b[i];
            if (std::abs(c[i] - expected) > 1e-6) {
                std::cout << "Error at index " << i << ": expected " << expected << ", got " << c[i] << std::endl;
                success = false;
            }
        }

        if (success) {
            std::cout << "✓ SYCL kernel executed successfully!" << std::endl;
        } else {
            std::cout << "✗ SYCL kernel verification failed!" << std::endl;
        }

        return success ? 0 : 1;

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL error: " << e.what() << std::endl;
        return 1;
    }
}