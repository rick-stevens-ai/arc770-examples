#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

// Force SYCL initialization for FFI
__attribute__((constructor))
void init_sycl() {
    try {
        // Initialize SYCL runtime
        auto devices = sycl::device::get_devices();
        if (!devices.empty()) {
            auto device = devices[0];
            sycl::queue q(device);
            // Force initialization of runtime
        }
    } catch (...) {
        // Ignore initialization errors
    }
}

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
        std::strncpy(info->name, name.c_str(), sizeof(info->name) - 1);
        info->name[sizeof(info->name) - 1] = '\0';

        // Copy vendor name
        std::string vendor = device.get_info<sycl::info::device::vendor>();
        std::strncpy(info->vendor, vendor.c_str(), sizeof(info->vendor) - 1);
        info->vendor[sizeof(info->vendor) - 1] = '\0';

        // Check if it's a GPU
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

        // Use USM (Unified Shared Memory) approach instead of buffers/accessors
        float* usm_a = sycl::malloc_device<float>(size, q);
        float* usm_b = sycl::malloc_device<float>(size, q);
        float* usm_c = sycl::malloc_device<float>(size, q);

        // Copy input data to device
        q.memcpy(usm_a, a, size * sizeof(float)).wait();
        q.memcpy(usm_b, b, size * sizeof(float)).wait();

        // Submit kernel using USM pointers without explicit kernel naming
        q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
            usm_c[idx] = usm_a[idx] + usm_b[idx];
        }).wait();

        // Copy result back to host
        q.memcpy(c, usm_c, size * sizeof(float)).wait();

        // Free device memory
        sycl::free(usm_a, q);
        sycl::free(usm_b, q);
        sycl::free(usm_c, q);

        q.wait();
        return 0;

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL error in simple_vector_add: " << e.what() << std::endl;
        return -1;
    }
}

} // extern "C"