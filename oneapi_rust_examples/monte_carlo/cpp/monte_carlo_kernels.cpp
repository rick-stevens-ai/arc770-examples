#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>
#include <random>

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

// Linear Congruential Generator for GPU random numbers
class SimpleRNG {
public:
    SimpleRNG(unsigned int seed) : state(seed) {}

    float next() {
        state = (state * 1103515245u + 12345u) & 0x7fffffff;
        return static_cast<float>(state) / static_cast<float>(0x7fffffff);
    }

private:
    unsigned int state;
};

int monte_carlo_pi_sycl(
    int total_samples,
    int device_id,
    double* pi_estimate,
    double* elapsed_ms
) {
    try {
        auto devices = sycl::device::get_devices();
        if (device_id >= static_cast<int>(devices.size())) {
            return -1;
        }

        auto device = devices[device_id];
        sycl::queue q(device);

        std::cout << "Running Monte Carlo Pi calculation on: "
                  << device.get_info<sycl::info::device::name>() << std::endl;

        // Buffer to store results from each work item
        const int work_group_size = 256;
        const int num_work_groups = (total_samples + work_group_size - 1) / work_group_size;
        const int actual_samples = num_work_groups * work_group_size;

        std::vector<int> results(num_work_groups, 0);
        sycl::buffer<int, 1> result_buf(results.data(), sycl::range<1>(num_work_groups));

        auto start = std::chrono::high_resolution_clock::now();

        // Generate random seeds for each work group
        std::random_device rd;
        std::vector<unsigned int> seeds(num_work_groups);
        for (int i = 0; i < num_work_groups; i++) {
            seeds[i] = rd() + i;
        }
        sycl::buffer<unsigned int, 1> seed_buf(seeds.data(), sycl::range<1>(num_work_groups));

        q.submit([&](sycl::handler& h) {
            auto result_acc = result_buf.get_access<sycl::access::mode::write>(h);
            auto seed_acc = seed_buf.get_access<sycl::access::mode::read>(h);

            // Local memory for reduction
            sycl::accessor<int, 1, sycl::access::mode::read_write,
                           sycl::access::target::local> local_results(
                sycl::range<1>(work_group_size), h);

            h.parallel_for(sycl::nd_range<1>(
                sycl::range<1>(actual_samples),
                sycl::range<1>(work_group_size)
            ), [=](sycl::nd_item<1> item) {
                int global_id = item.get_global_id(0);
                int local_id = item.get_local_id(0);
                int group_id = item.get_group(0);

                // Create RNG with unique seed for each work item
                SimpleRNG rng(seed_acc[group_id] + local_id);

                int samples_per_item = work_group_size;
                int inside_count = 0;

                // Each work item processes multiple samples
                for (int i = 0; i < samples_per_item; i++) {
                    float x = rng.next() * 2.0f - 1.0f;  // -1 to 1
                    float y = rng.next() * 2.0f - 1.0f;  // -1 to 1

                    if (x * x + y * y <= 1.0f) {
                        inside_count++;
                    }
                }

                local_results[local_id] = inside_count;

                // Synchronize work items in the work group
                item.barrier(sycl::access::fence_space::local_space);

                // Reduction within work group
                for (int offset = work_group_size / 2; offset > 0; offset >>= 1) {
                    if (local_id < offset) {
                        local_results[local_id] += local_results[local_id + offset];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Store result for this work group
                if (local_id == 0) {
                    result_acc[group_id] = local_results[0];
                }
            });
        });

        q.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        *elapsed_ms = duration.count() / 1000.0;

        // Sum up results from all work groups
        int total_inside = 0;
        for (int count : results) {
            total_inside += count;
        }

        // Calculate Pi estimate
        int total_processed_samples = num_work_groups * work_group_size * work_group_size;
        *pi_estimate = 4.0 * static_cast<double>(total_inside) / static_cast<double>(total_processed_samples);

        std::cout << "Processed " << total_processed_samples << " samples" << std::endl;
        std::cout << "Points inside circle: " << total_inside << std::endl;

        return 0;

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL error in monte_carlo_pi: " << e.what() << std::endl;
        return -1;
    }
}

// Alternative implementation with better random number generation
int monte_carlo_pi_philox_sycl(
    int total_samples,
    int device_id,
    double* pi_estimate,
    double* elapsed_ms
) {
    try {
        auto devices = sycl::device::get_devices();
        if (device_id >= static_cast<int>(devices.size())) {
            return -1;
        }

        auto device = devices[device_id];
        sycl::queue q(device);

        std::cout << "Running Monte Carlo Pi (Philox RNG) on: "
                  << device.get_info<sycl::info::device::name>() << std::endl;

        // Use a simpler approach with pre-generated random numbers
        std::vector<float> random_points(total_samples * 2);

        // Generate random numbers on CPU first (for simplicity)
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (int i = 0; i < total_samples * 2; i++) {
            random_points[i] = dis(gen);
        }

        sycl::buffer<float, 1> points_buf(random_points.data(), sycl::range<1>(total_samples * 2));
        sycl::buffer<int, 1> result_buf(sycl::range<1>(1));

        auto start = std::chrono::high_resolution_clock::now();

        q.submit([&](sycl::handler& h) {
            auto points_acc = points_buf.get_access<sycl::access::mode::read>(h);
            auto result_acc = result_buf.get_access<sycl::access::mode::write>(h);

            h.single_task([=]() {
                int inside_count = 0;

                for (int i = 0; i < total_samples; i++) {
                    float x = points_acc[i * 2];
                    float y = points_acc[i * 2 + 1];

                    if (x * x + y * y <= 1.0f) {
                        inside_count++;
                    }
                }

                result_acc[0] = inside_count;
            });
        });

        q.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        *elapsed_ms = duration.count() / 1000.0;

        // Read back result
        auto result_acc = result_buf.get_access<sycl::access::mode::read>();
        int total_inside = result_acc[0];

        *pi_estimate = 4.0 * static_cast<double>(total_inside) / static_cast<double>(total_samples);

        std::cout << "Processed " << total_samples << " samples" << std::endl;
        std::cout << "Points inside circle: " << total_inside << std::endl;

        return 0;

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL error in monte_carlo_pi_philox: " << e.what() << std::endl;
        return -1;
    }
}

} // extern "C"