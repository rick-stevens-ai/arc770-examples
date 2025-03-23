#include <sycl/sycl.hpp>
#include <iostream>
#include <algorithm>
#include "binomial.hpp"

using namespace sycl;

void OverallRun(stock_data_t stock_data_on_cpu, int SIGN,
              const int blockSize, const int StepSize,
              const int sg_size) {
    std::cout << "Single Precision Binomial Option Pricing"
             << " version 1.8 running on " 
             << queue{sycl::default_selector_v}.get_device().get_info<info::device::name>()
             << " using DPC++, workgroup size " 
             << blockSize << ", sub-group size " << sg_size 
             << ".\n";

    const size_t input_size = stock_data_on_cpu.size();
    std::cout << "Input Dataset   : " << input_size << "\n";

    try {
        queue main_queue{default_selector_v};
        buffer<Type> option_strike{stock_data_on_cpu.OptionStrike};
        buffer<Type> stock_price{stock_data_on_cpu.StockPrice};
        buffer<Type> option_years{stock_data_on_cpu.OptionYears};
        buffer<Type> output_call{stock_data_on_cpu.OptionCallPrice};

        main_queue.submit([&](handler& h) {
            accessor option_strike_accessor{option_strike, h};
            accessor stock_price_accessor{stock_price, h};
            accessor option_years_accessor{option_years, h};
            accessor output_call_accessor{output_call, h, write_only};

            range num_items{input_size};
            h.parallel_for(
                nd_range<1>{num_items, range<1>(blockSize)},
                [=](nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    size_t tid = item.get_global_id(0);
                    if (tid < input_size) {
                        BinomialBodyPipeline(
                            output_call_accessor.get_multi_ptr<access::decorated::no>().get(),
                            stock_price_accessor.get_multi_ptr<access::decorated::no>().get(),
                            option_strike_accessor.get_multi_ptr<access::decorated::no>().get(),
                            option_years_accessor.get_multi_ptr<access::decorated::no>().get(),
                            StepSize, SIGN, tid);
                    }
                });
        }).wait();
    } catch (sycl::exception const& e) {
        std::cout << "An exception is caught while offloading the work.\n";
        std::terminate();
    }
}
