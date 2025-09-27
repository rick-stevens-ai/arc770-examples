#include <sycl/sycl.hpp>
int main() {
    for (const auto& platform : sycl::platform::get_platforms()) {
        std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>() << std::endl;
        for (const auto& device : platform.get_devices()) {
            std::cout << "  Device: " << device.get_info<sycl::info::device::name>() << std::endl;
            std::cout << "    Vendor: " << device.get_info<sycl::info::device::vendor>() << std::endl;
            std::cout << "    Type: " << (device.is_gpu() ? "GPU" : device.is_cpu() ? "CPU" : "unknown") << std::endl;
        }
    }
    return 0;
}
