#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cstring>

extern "C" {

struct DeviceInfo {
    char name[256];
    char vendor[256];
    bool is_gpu;
    int max_work_group_size;
};

int get_device_count() {
    try {
        auto devices = sycl::device::get_devices();
        return static_cast<int>(devices.size());
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL error in get_device_count: " << e.what() << std::endl;
        return 0;
    }
}

int get_device_info(int device_id, DeviceInfo* info) {
    try {
        auto devices = sycl::device::get_devices();
        if (device_id >= static_cast<int>(devices.size())) {
            return -1;
        }

        auto device = devices[device_id];

        // Copy device name
        std::string name = device.get_info<sycl::info::device::name>();
        strncpy(info->name, name.c_str(), 255);
        info->name[255] = '\0';

        // Copy vendor name
        std::string vendor = device.get_info<sycl::info::device::vendor>();
        strncpy(info->vendor, vendor.c_str(), 255);
        info->vendor[255] = '\0';

        // Check if GPU
        info->is_gpu = device.is_gpu();

        // Get max work group size
        info->max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();

        return 0;
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL error in get_device_info: " << e.what() << std::endl;
        return -1;
    }
}

int simple_vector_add(float* a, float* b, float* c, int size, int device_id) {
    try {
        auto devices = sycl::device::get_devices();
        if (device_id >= static_cast<int>(devices.size())) {
            return -1;
        }

        auto device = devices[device_id];
        sycl::queue q(device);

        std::cout << "Using device: " << device.get_info<sycl::info::device::name>() << std::endl;

        // Copy input data to vectors (simpler approach)
        std::vector<float> vec_a(a, a + size);
        std::vector<float> vec_b(b, b + size);
        std::vector<float> vec_c(size);

        // Use device memory allocation
        float* usm_a = sycl::malloc_device<float>(size, q);
        float* usm_b = sycl::malloc_device<float>(size, q);
        float* usm_c = sycl::malloc_device<float>(size, q);

        // Copy to device
        q.memcpy(usm_a, vec_a.data(), size * sizeof(float)).wait();
        q.memcpy(usm_b, vec_b.data(), size * sizeof(float)).wait();

        // Simple kernel execution (same as working sycl_runner.cpp)
        q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
            usm_c[idx] = usm_a[idx] + usm_b[idx];
        }).wait();

        // Copy result back
        q.memcpy(vec_c.data(), usm_c, size * sizeof(float)).wait();

        // Copy to output array
        for (int i = 0; i < size; i++) {
            c[i] = vec_c[i];
        }

        // Cleanup
        sycl::free(usm_a, q);
        sycl::free(usm_b, q);
        sycl::free(usm_c, q);

        std::cout << "✓ Vector addition completed successfully!" << std::endl;
        return 0;

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL error in simple_vector_add: " << e.what() << std::endl;
        return -1;
    }
}

} // extern "C"