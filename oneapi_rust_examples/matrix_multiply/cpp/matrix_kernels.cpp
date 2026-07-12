#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>

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
        std::cerr << "SYCL error: " << e.what() << std::endl;
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

        std::string name = device.get_info<sycl::info::device::name>();
        std::strncpy(info->name, name.c_str(), sizeof(info->name) - 1);
        info->name[sizeof(info->name) - 1] = '\0';

        std::string vendor = device.get_info<sycl::info::device::vendor>();
        std::strncpy(info->vendor, vendor.c_str(), sizeof(info->vendor) - 1);
        info->vendor[sizeof(info->vendor) - 1] = '\0';

        info->is_gpu = device.is_gpu();
        info->max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();

        return 0;
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL error: " << e.what() << std::endl;
        return -1;
    }
}

int matrix_multiply_sycl(
    float* a,
    float* b,
    float* c,
    int size,
    int device_id,
    double* elapsed_ms
) {
    try {
        auto devices = sycl::device::get_devices();
        if (device_id >= static_cast<int>(devices.size())) {
            return -1;
        }

        auto device = devices[device_id];
        sycl::queue q(device);

        std::cout << "Running matrix multiplication on: "
                  << device.get_info<sycl::info::device::name>() << std::endl;

        // Create SYCL buffers
        sycl::buffer<float, 2> buf_a(a, sycl::range<2>(size, size));
        sycl::buffer<float, 2> buf_b(b, sycl::range<2>(size, size));
        sycl::buffer<float, 2> buf_c(c, sycl::range<2>(size, size));

        auto start = std::chrono::high_resolution_clock::now();

        // Submit matrix multiplication kernel
        q.submit([&](sycl::handler& h) {
            auto acc_a = buf_a.get_access<sycl::access::mode::read>(h);
            auto acc_b = buf_b.get_access<sycl::access::mode::read>(h);
            auto acc_c = buf_c.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<2>(size, size), [=](sycl::id<2> idx) {
                int row = idx[0];
                int col = idx[1];

                float sum = 0.0f;
                for (int k = 0; k < size; k++) {
                    sum += acc_a[row][k] * acc_b[k][col];
                }
                acc_c[row][col] = sum;
            });
        });

        q.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        *elapsed_ms = duration.count() / 1000.0;

        return 0;

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL error in matrix_multiply: " << e.what() << std::endl;
        return -1;
    }
}

// Optimized version using local memory (for larger matrices)
int matrix_multiply_optimized_sycl(
    float* a,
    float* b,
    float* c,
    int size,
    int device_id,
    double* elapsed_ms
) {
    try {
        auto devices = sycl::device::get_devices();
        if (device_id >= static_cast<int>(devices.size())) {
            return -1;
        }

        auto device = devices[device_id];
        sycl::queue q(device);

        std::cout << "Running optimized matrix multiplication on: "
                  << device.get_info<sycl::info::device::name>() << std::endl;

        // Create SYCL buffers
        sycl::buffer<float, 2> buf_a(a, sycl::range<2>(size, size));
        sycl::buffer<float, 2> buf_b(b, sycl::range<2>(size, size));
        sycl::buffer<float, 2> buf_c(c, sycl::range<2>(size, size));

        const int TILE_SIZE = 16;  // Tile size for blocking

        auto start = std::chrono::high_resolution_clock::now();

        // Submit optimized matrix multiplication kernel with tiling
        q.submit([&](sycl::handler& h) {
            auto acc_a = buf_a.get_access<sycl::access::mode::read>(h);
            auto acc_b = buf_b.get_access<sycl::access::mode::read>(h);
            auto acc_c = buf_c.get_access<sycl::access::mode::write>(h);

            // Local memory for tiles
            sycl::accessor<float, 2, sycl::access::mode::read_write,
                           sycl::access::target::local> tile_a(
                sycl::range<2>(TILE_SIZE, TILE_SIZE), h);
            sycl::accessor<float, 2, sycl::access::mode::read_write,
                           sycl::access::target::local> tile_b(
                sycl::range<2>(TILE_SIZE, TILE_SIZE), h);

            h.parallel_for(sycl::nd_range<2>(
                sycl::range<2>(size, size),
                sycl::range<2>(TILE_SIZE, TILE_SIZE)
            ), [=](sycl::nd_item<2> item) {
                int row = item.get_global_id(0);
                int col = item.get_global_id(1);
                int local_row = item.get_local_id(0);
                int local_col = item.get_local_id(1);

                float sum = 0.0f;

                for (int tile = 0; tile < (size + TILE_SIZE - 1) / TILE_SIZE; tile++) {
                    // Load tiles into local memory
                    int tile_row = tile * TILE_SIZE + local_row;
                    int tile_col = tile * TILE_SIZE + local_col;

                    if (row < size && tile_col < size) {
                        tile_a[local_row][local_col] = acc_a[row][tile_col];
                    } else {
                        tile_a[local_row][local_col] = 0.0f;
                    }

                    if (tile_row < size && col < size) {
                        tile_b[local_row][local_col] = acc_b[tile_row][col];
                    } else {
                        tile_b[local_row][local_col] = 0.0f;
                    }

                    item.barrier(sycl::access::fence_space::local_space);

                    // Compute partial sum using local memory
                    for (int k = 0; k < TILE_SIZE; k++) {
                        sum += tile_a[local_row][k] * tile_b[k][local_col];
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (row < size && col < size) {
                    acc_c[row][col] = sum;
                }
            });
        });

        q.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        *elapsed_ms = duration.count() / 1000.0;

        return 0;

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL error in matrix_multiply_optimized: " << e.what() << std::endl;
        return -1;
    }
}

} // extern "C"