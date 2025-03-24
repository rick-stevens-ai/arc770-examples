#include <sycl/sycl.hpp>
#include <iostream>
#include <algorithm>
#include "black_scholes.hpp"
#include "input.hpp"
#include "precision.hpp"

using namespace sycl;

// The OverallRun offloads black scholes calculation to SYCL device
// with single or double precision operations
void OverallRun(stock_data_t stock_data_on_cpu, int SIGN,
              const int blockSize, const int numIterations,
              const int sg_size) {
    std::cout << "Single Precision Black&Scholes Option Pricing"
             << " version 1.6 running on " 
             << queue{sycl::default_selector_v}.get_device().get_info<info::device::name>()
             << " using DPC++, workgroup size " 
             << blockSize << ", sub-group size " << sg_size 
             << ".\n";

    std::cout << "Compiler Version: "
             << __INTEL_LLVM_COMPILER << " " 
             << __INTEL_LLVM_COMPILER_VERSION << ", "
             << "LLVM " << __INTEL_LLVM_VERSION << " based.\n";

    std::cout << "Driver Version  : " 
             << queue{sycl::default_selector_v}.get_device().get_info<info::device::driver_version>()
             << "\n";

    std::cout << "Build Time      : " << __DATE__ << " " << __TIME__ << "\n";

    const size_t input_size = stock_data_on_cpu.size();
    std::cout << "Input Dataset   : " << input_size << "\n";

    constexpr size_t wait_count = 8;
    std::vector<event> AllEvents(wait_count);
    try {
        queue main_queue{default_selector_v};
        buffer option_strike{stock_data_on_cpu.OptionStrike};
        buffer stock_price{stock_data_on_cpu.StockPrice};
        buffer option_years{stock_data_on_cpu.OptionYears};
        buffer output_call{stock_data_on_cpu.OptionCallPrice};

        for (int i = 0; i < numIterations; i++) {
            using Type = PREC_TYPE;
            auto e1 = main_queue.submit([&](handler& h) {
                accessor option_strike_accessor{option_strike, h};
                accessor stock_price_accessor{stock_price, h};
                accessor option_years_accessor{option_years, h};
                accessor output_call_accessor{output_call, h};

                range num_items{input_size};
                h.parallel_for(
                    nd_range<1>{num_items, range<1>(blockSize)},
                    [=](nd_item<1> item) [[sycl::kernel_args_restrict]] [[sycl::reqd_sub_group_size(sg_size)]] {
                        size_t tid = item.get_global_id(0);
                        Type call;
                        BlackScholesBodyPipeline(
                            output_call_accessor.get_pointer(),
                            stock_price_accessor.get_pointer(),
                            option_strike_accessor.get_pointer(),
                            option_years_accessor.get_pointer(),
                            SIGN, tid);
                    });
            });
            AllEvents[i & (wait_count - 1)] = e1;

            if ((i & (wait_count - 1)) == ((wait_count - 1))) {
                for (int j = 0; j < wait_count; j++) {
                    AllEvents[j].wait();
                }
            }
        }
    } catch (sycl::exception const& e) {
        std::cout << "An exception is caught while offloading the work.\n";
        std::terminate();
    }
}
