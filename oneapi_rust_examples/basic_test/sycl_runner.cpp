#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <operation> <device_id> <input_file> [output_file]" << std::endl;
        return 1;
    }

    std::string operation = argv[1];
    int device_id = std::atoi(argv[2]);
    std::string input_file = argv[3];
    std::string output_file = (argc > 4) ? argv[4] : "output.bin";

    try {
        auto devices = sycl::device::get_devices();
        if (device_id >= static_cast<int>(devices.size())) {
            std::cerr << "Error: Device ID " << device_id << " not available" << std::endl;
            return 1;
        }

        auto device = devices[device_id];
        sycl::queue q(device);

        std::cout << "Using device: " << device.get_info<sycl::info::device::name>() << std::endl;

        if (operation == "vector_add") {
            // Read input data from file
            std::ifstream infile(input_file, std::ios::binary);
            if (!infile) {
                std::cerr << "Error: Cannot open input file " << input_file << std::endl;
                return 1;
            }

            // Read size
            int size;
            infile.read(reinterpret_cast<char*>(&size), sizeof(int));

            // Read arrays A and B
            std::vector<float> a(size), b(size), c(size);
            infile.read(reinterpret_cast<char*>(a.data()), size * sizeof(float));
            infile.read(reinterpret_cast<char*>(b.data()), size * sizeof(float));
            infile.close();

            // Perform vector addition on SYCL device
            float* usm_a = sycl::malloc_device<float>(size, q);
            float* usm_b = sycl::malloc_device<float>(size, q);
            float* usm_c = sycl::malloc_device<float>(size, q);

            q.memcpy(usm_a, a.data(), size * sizeof(float)).wait();
            q.memcpy(usm_b, b.data(), size * sizeof(float)).wait();

            q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
                usm_c[idx] = usm_a[idx] + usm_b[idx];
            }).wait();

            q.memcpy(c.data(), usm_c, size * sizeof(float)).wait();

            sycl::free(usm_a, q);
            sycl::free(usm_b, q);
            sycl::free(usm_c, q);

            // Write result to output file
            std::ofstream outfile(output_file, std::ios::binary);
            outfile.write(reinterpret_cast<const char*>(c.data()), size * sizeof(float));
            outfile.close();

            std::cout << "Vector addition completed successfully!" << std::endl;
            return 0;
        } else {
            std::cerr << "Error: Unknown operation " << operation << std::endl;
            return 1;
        }

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL error: " << e.what() << std::endl;
        return 1;
    }
}