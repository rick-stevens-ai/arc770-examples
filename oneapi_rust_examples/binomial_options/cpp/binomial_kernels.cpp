#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

extern "C" {

struct DeviceInfo {
    char name[256];
    char vendor[256];
    bool is_gpu;
    int max_work_group_size;
};

struct OptionData {
    float spot_price;
    float strike_price;
    float risk_free_rate;
    float volatility;
    float time_to_maturity;
    int is_call;  // 1 for call, 0 for put
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

int binomial_option_pricing_sycl(
    OptionData* options,
    float* option_prices,
    int num_options,
    int num_steps,
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

        std::cout << "Running binomial option pricing on: "
                  << device.get_info<sycl::info::device::name>() << std::endl;

        // Create SYCL buffers
        sycl::buffer<OptionData, 1> options_buf(options, sycl::range<1>(num_options));
        sycl::buffer<float, 1> prices_buf(option_prices, sycl::range<1>(num_options));

        auto start = std::chrono::high_resolution_clock::now();

        // Submit binomial option pricing kernel
        q.submit([&](sycl::handler& h) {
            auto options_acc = options_buf.get_access<sycl::access::mode::read>(h);
            auto prices_acc = prices_buf.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(num_options), [=](sycl::id<1> idx) {
                int option_id = idx[0];
                OptionData opt = options_acc[option_id];

                float S = opt.spot_price;
                float K = opt.strike_price;
                float r = opt.risk_free_rate;
                float sigma = opt.volatility;
                float T = opt.time_to_maturity;
                int is_call = opt.is_call;

                float dt = T / num_steps;
                float u = sycl::exp(sigma * sycl::sqrt(dt));
                float d = 1.0f / u;
                float p = (sycl::exp(r * dt) - d) / (u - d);
                float disc = sycl::exp(-r * dt);

                // Create array for option values at each node
                // We need num_steps + 1 values for the final time step
                float option_values[2049];  // Max for 2048 steps + 1

                // Initialize option values at maturity (final time step)
                for (int i = 0; i <= num_steps; i++) {
                    float stock_price = S * sycl::pow(u, num_steps - i) * sycl::pow(d, i);
                    if (is_call) {
                        option_values[i] = sycl::max(0.0f, stock_price - K);
                    } else {
                        option_values[i] = sycl::max(0.0f, K - stock_price);
                    }
                }

                // Backward induction
                for (int step = num_steps - 1; step >= 0; step--) {
                    for (int i = 0; i <= step; i++) {
                        option_values[i] = disc * (p * option_values[i] + (1.0f - p) * option_values[i + 1]);
                    }
                }

                prices_acc[option_id] = option_values[0];
            });
        });

        q.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        *elapsed_ms = duration.count() / 1000.0;

        return 0;

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL error in binomial_option_pricing: " << e.what() << std::endl;
        return -1;
    }
}

// Optimized version using local memory and work groups
int binomial_option_pricing_optimized_sycl(
    OptionData* options,
    float* option_prices,
    int num_options,
    int num_steps,
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

        std::cout << "Running optimized binomial option pricing on: "
                  << device.get_info<sycl::info::device::name>() << std::endl;

        // Create SYCL buffers
        sycl::buffer<OptionData, 1> options_buf(options, sycl::range<1>(num_options));
        sycl::buffer<float, 1> prices_buf(option_prices, sycl::range<1>(num_options));

        const int WORK_GROUP_SIZE = 64;
        const int MAX_STEPS_LOCAL = 512;  // Limit for local memory usage

        auto start = std::chrono::high_resolution_clock::now();

        // If we have too many steps, fall back to global memory version
        if (num_steps > MAX_STEPS_LOCAL) {
            return binomial_option_pricing_sycl(options, option_prices, num_options,
                                              num_steps, device_id, elapsed_ms);
        }

        // Submit optimized kernel with local memory
        q.submit([&](sycl::handler& h) {
            auto options_acc = options_buf.get_access<sycl::access::mode::read>(h);
            auto prices_acc = prices_buf.get_access<sycl::access::mode::write>(h);

            // Local memory for option values
            sycl::accessor<float, 1, sycl::access::mode::read_write,
                           sycl::access::target::local> local_values(
                sycl::range<1>(MAX_STEPS_LOCAL + 1), h);

            h.parallel_for(sycl::nd_range<1>(
                sycl::range<1>((num_options + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE * WORK_GROUP_SIZE),
                sycl::range<1>(WORK_GROUP_SIZE)
            ), [=](sycl::nd_item<1> item) {
                int global_id = item.get_global_id(0);
                int local_id = item.get_local_id(0);

                if (global_id >= num_options) return;

                OptionData opt = options_acc[global_id];

                float S = opt.spot_price;
                float K = opt.strike_price;
                float r = opt.risk_free_rate;
                float sigma = opt.volatility;
                float T = opt.time_to_maturity;
                int is_call = opt.is_call;

                float dt = T / num_steps;
                float u = sycl::exp(sigma * sycl::sqrt(dt));
                float d = 1.0f / u;
                float p = (sycl::exp(r * dt) - d) / (u - d);
                float disc = sycl::exp(-r * dt);

                // Initialize option values at maturity using shared work
                for (int i = local_id; i <= num_steps; i += WORK_GROUP_SIZE) {
                    float stock_price = S * sycl::pow(u, num_steps - i) * sycl::pow(d, i);
                    if (is_call) {
                        local_values[i] = sycl::max(0.0f, stock_price - K);
                    } else {
                        local_values[i] = sycl::max(0.0f, K - stock_price);
                    }
                }

                item.barrier(sycl::access::fence_space::local_space);

                // Backward induction with work sharing
                for (int step = num_steps - 1; step >= 0; step--) {
                    for (int i = local_id; i <= step; i += WORK_GROUP_SIZE) {
                        local_values[i] = disc * (p * local_values[i] + (1.0f - p) * local_values[i + 1]);
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Only one work item writes the result
                if (local_id == 0) {
                    prices_acc[global_id] = local_values[0];
                }
            });
        });

        q.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        *elapsed_ms = duration.count() / 1000.0;

        return 0;

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL error in binomial_option_pricing_optimized: " << e.what() << std::endl;
        return -1;
    }
}

// Black-Scholes formula for comparison
float black_scholes_price(float S, float K, float T, float r, float sigma, bool is_call) {
    if (T <= 0) return is_call ? std::max(0.0f, S - K) : std::max(0.0f, K - S);

    float d1 = (std::log(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * std::sqrt(T));
    float d2 = d1 - sigma * std::sqrt(T);

    // Simplified normal CDF approximation
    auto norm_cdf = [](float x) {
        return 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f)));
    };

    if (is_call) {
        return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
    } else {
        return K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
    }
}

int compute_black_scholes_reference(
    OptionData* options,
    float* bs_prices,
    int num_options
) {
    for (int i = 0; i < num_options; i++) {
        OptionData opt = options[i];
        bs_prices[i] = black_scholes_price(
            opt.spot_price,
            opt.strike_price,
            opt.time_to_maturity,
            opt.risk_free_rate,
            opt.volatility,
            opt.is_call == 1
        );
    }
    return 0;
}

} // extern "C"