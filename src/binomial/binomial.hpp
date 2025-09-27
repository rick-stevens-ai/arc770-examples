#ifndef BINOMIAL_HPP
#define BINOMIAL_HPP

#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include "precision.hpp"

using Type = PREC_TYPE;

struct stock_data_t {
    std::vector<Type> StockPrice;
    std::vector<Type> OptionStrike;
    std::vector<Type> OptionYears;
    std::vector<Type> OptionCallPrice;

    size_t size() const { return StockPrice.size(); }
};

inline void init_data(stock_data_t& data, size_t size) {
    data.StockPrice.resize(size);
    data.OptionStrike.resize(size);
    data.OptionYears.resize(size);
    data.OptionCallPrice.resize(size);

    for (size_t i = 0; i < size; i++) {
        data.StockPrice[i] = static_cast<Type>(100.0 + i);
        data.OptionStrike[i] = static_cast<Type>(98.0 + i);
        data.OptionYears[i] = static_cast<Type>(2.0);
    }
}

inline void BinomialBodyPipeline(Type* call_result,
                             const Type* stock_price,
                             const Type* option_strike,
                             const Type* option_years,
                             const int steps,
                             const int SIGN,
                             const size_t tid) {
    const Type S = stock_price[tid];
    const Type L = option_strike[tid];
    const Type t = option_years[tid];
    const Type r = static_cast<Type>(0.02);
    const Type sigma = static_cast<Type>(0.30);

    const Type dt = t / steps;
    const Type u = sycl::exp(sigma * sycl::sqrt(dt));
    const Type d = 1.0 / u;
    const Type disc = sycl::exp(-r * dt);
    const Type p = (1.0 - d * disc) / (u - d);
    const Type oneMinusP = 1.0 - p;

    // Use stack allocation with fixed size
    constexpr int MAX_STEPS = 256;  // Maximum number of steps
    Type prices[MAX_STEPS];
    Type asset = S * sycl::pow(d, steps);

    // Ensure we don't exceed array bounds
    const int actual_steps = sycl::min(steps, MAX_STEPS - 1);

    for (int i = 0; i <= actual_steps; ++i) {
        prices[i] = sycl::max(static_cast<Type>(0.0),
                            (SIGN > 0) ? asset - L : L - asset);
        asset *= u;
    }

    for (int step = actual_steps - 1; step >= 0; --step) {
        for (int i = 0; i <= step; ++i) {
            prices[i] = (p * prices[i + 1] + oneMinusP * prices[i]) * disc;
        }
    }

    call_result[tid] = prices[0];
}

inline void verify_results(const stock_data_t& stock_data, const int SIGN, const int steps) {
    std::vector<Type> h_call_result = stock_data.OptionCallPrice;
    std::vector<Type> h_CallResultCPU(stock_data.size());

    for (size_t i = 0; i < stock_data.size(); i++) {
        BinomialBodyPipeline(h_CallResultCPU.data(),
                         stock_data.StockPrice.data(),
                         stock_data.OptionStrike.data(),
                         stock_data.OptionYears.data(),
                         steps, SIGN, i);
    }

    Type sum_delta = 0.0;
    Type sum_ref = 0.0;
    Type max_delta = 0.0;
    for (size_t i = 0; i < stock_data.size(); i++) {
        Type ref = sycl::fabs(h_CallResultCPU[i]);
        Type delta = sycl::fabs(h_CallResultCPU[i] - h_call_result[i]);
        if (delta > max_delta) max_delta = delta;
        sum_delta += delta;
        sum_ref += ref;
    }

    Type L1norm = sum_delta / sum_ref;
    std::cout << "L1 norm: " << L1norm << std::endl;
    std::cout << ((L1norm < 1e-6) ? "TEST PASSED\n" : "TEST FAILED\n");
}

void OverallRun(stock_data_t stock_data_on_cpu, int SIGN,
              const int blockSize, const int StepSize,
              const int sg_size);

#endif  // BINOMIAL_HPP
