#include <sycl/sycl.hpp>
#include <iostream>

class vector_addition;

int main() {
    constexpr size_t N = 16;
    int data[N];
    
    try {
        // Initialize the input data
        for (size_t i = 0; i < N; i++) {
            data[i] = i;
        }

        // By-default get a GPU device
        sycl::device dev = sycl::device(sycl::gpu_selector_v);
        
        std::cout << "Running on device: " 
                 << dev.get_info<sycl::info::device::name>() << "\n"
                 << "Device vendor: "
                 << dev.get_info<sycl::info::device::vendor>() << "\n";

        // Create a queue to submit work to the device
        sycl::queue queue(dev);

        // Create a buffer for the data
        {
            sycl::buffer buf(data, sycl::range<1>(N));

            queue.submit([&](sycl::handler& h) {
                auto accessor = buf.get_access<sycl::access::mode::read_write>(h);
                
                h.parallel_for<class vector_addition>(
                    sycl::range<1>(N),
                    [=](sycl::id<1> idx) {
                        accessor[idx] *= 2;
                    }
                );
            });
        }

        // Print results
        std::cout << "Results: ";
        for (size_t i = 0; i < N; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;

        return 0;
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception caught: " << e.what() << std::endl;
        return 1;
    }
}
