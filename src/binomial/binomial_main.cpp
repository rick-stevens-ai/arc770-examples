#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include "binomial.hpp"

int main() {
    const size_t num_elements = 2000000;
    const int blockSize = 256;
    const int StepSize = 64;
    const int sg_size = 32;
    
    // Initialize input data
    stock_data_t stock_data;
    init_data(stock_data, num_elements);
    
    try {
        // Calculate call option (SIGN = 1)
        std::cout << "\nCalculating Call Option Price\n";
        OverallRun(stock_data, 1, blockSize, StepSize, sg_size);
        verify_results(stock_data, 1, StepSize);
        
        // Calculate put option (SIGN = -1)
        std::cout << "\nCalculating Put Option Price\n";
        OverallRun(stock_data, -1, blockSize, StepSize, sg_size);
        verify_results(stock_data, -1, StepSize);
        
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
