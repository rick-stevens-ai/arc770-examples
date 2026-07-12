#include <metal_stdlib>
using namespace metal;

struct OptionData {
    float spot_price;
    float strike_price;
    float risk_free_rate;
    float volatility;
    float time_to_maturity;
    uint is_call; // 1 for call, 0 for put
};

// Basic binomial option pricing
kernel void binomial_option_pricing(device const OptionData* options [[buffer(0)]],
                                   device float* option_prices [[buffer(1)]],
                                   constant uint& num_steps [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]]) {

    OptionData opt = options[gid];

    float S = opt.spot_price;
    float K = opt.strike_price;
    float r = opt.risk_free_rate;
    float sigma = opt.volatility;
    float T = opt.time_to_maturity;
    bool is_call = (opt.is_call == 1);

    float dt = T / float(num_steps);
    float u = exp(sigma * sqrt(dt));  // up factor
    float d = 1.0f / u;              // down factor
    float p = (exp(r * dt) - d) / (u - d); // risk-neutral probability
    float disc = exp(-r * dt);       // discount factor

    // Create array for option values at each time step
    // Note: This limits us to smaller num_steps due to local memory constraints
    float option_values[513]; // Max 512 steps + 1

    if (num_steps > 512) {
        option_prices[gid] = 0.0f; // Error case
        return;
    }

    // Initialize option values at maturity (final time step)
    for (uint i = 0; i <= num_steps; i++) {
        float stock_price = S * pow(u, float(num_steps - i)) * pow(d, float(i));
        if (is_call) {
            option_values[i] = max(0.0f, stock_price - K);
        } else {
            option_values[i] = max(0.0f, K - stock_price);
        }
    }

    // Backward induction
    for (int step = int(num_steps) - 1; step >= 0; step--) {
        for (int i = 0; i <= step; i++) {
            option_values[i] = disc * (p * option_values[i] + (1.0f - p) * option_values[i + 1]);
        }
    }

    option_prices[gid] = option_values[0];
}

// Optimized binomial option pricing with shared memory
kernel void binomial_option_pricing_shared(device const OptionData* options [[buffer(0)]],
                                          device float* option_prices [[buffer(1)]],
                                          constant uint& num_steps [[buffer(2)]],
                                          threadgroup float* shared_values [[threadgroup(0)]],
                                          uint gid [[thread_position_in_grid]],
                                          uint lid [[thread_position_in_threadgroup]],
                                          uint group_size [[threads_per_threadgroup]]) {

    if (num_steps > 256) {
        if (lid == 0) option_prices[gid] = 0.0f;
        return;
    }

    OptionData opt = options[gid];

    float S = opt.spot_price;
    float K = opt.strike_price;
    float r = opt.risk_free_rate;
    float sigma = opt.volatility;
    float T = opt.time_to_maturity;
    bool is_call = (opt.is_call == 1);

    float dt = T / float(num_steps);
    float u = exp(sigma * sqrt(dt));
    float d = 1.0f / u;
    float p = (exp(r * dt) - d) / (u - d);
    float disc = exp(-r * dt);

    // Initialize final values using shared work
    for (uint i = lid; i <= num_steps; i += group_size) {
        float stock_price = S * pow(u, float(num_steps - i)) * pow(d, float(i));
        if (is_call) {
            shared_values[i] = max(0.0f, stock_price - K);
        } else {
            shared_values[i] = max(0.0f, K - stock_price);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Backward induction with collaborative work
    for (int step = int(num_steps) - 1; step >= 0; step--) {
        for (int i = int(lid); i <= step; i += int(group_size)) {
            shared_values[i] = disc * (p * shared_values[i] + (1.0f - p) * shared_values[i + 1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Only one thread writes the result
    if (lid == 0) {
        option_prices[gid] = shared_values[0];
    }
}

// Black-Scholes formula for comparison
float normal_cdf_approx(float x) {
    // Abramowitz and Stegun approximation
    const float a1 =  0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 =  1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 =  1.061405429f;
    const float p  =  0.3275911f;

    float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    x = fabs(x) / sqrt(2.0f);

    float t = 1.0f / (1.0f + p * x);
    float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

    return 0.5f * (1.0f + sign * y);
}

kernel void black_scholes_pricing(device const OptionData* options [[buffer(0)]],
                                 device float* option_prices [[buffer(1)]],
                                 uint gid [[thread_position_in_grid]]) {

    OptionData opt = options[gid];

    float S = opt.spot_price;
    float K = opt.strike_price;
    float r = opt.risk_free_rate;
    float sigma = opt.volatility;
    float T = opt.time_to_maturity;
    bool is_call = (opt.is_call == 1);

    if (T <= 0.0f) {
        if (is_call) {
            option_prices[gid] = max(0.0f, S - K);
        } else {
            option_prices[gid] = max(0.0f, K - S);
        }
        return;
    }

    float d1 = (log(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * sqrt(T));
    float d2 = d1 - sigma * sqrt(T);

    if (is_call) {
        option_prices[gid] = S * normal_cdf_approx(d1) - K * exp(-r * T) * normal_cdf_approx(d2);
    } else {
        option_prices[gid] = K * exp(-r * T) * normal_cdf_approx(-d2) - S * normal_cdf_approx(-d1);
    }
}

// Trinomial option pricing (more accurate than binomial)
kernel void trinomial_option_pricing(device const OptionData* options [[buffer(0)]],
                                     device float* option_prices [[buffer(1)]],
                                     constant uint& num_steps [[buffer(2)]],
                                     uint gid [[thread_position_in_grid]]) {

    if (num_steps > 256) {
        option_prices[gid] = 0.0f;
        return;
    }

    OptionData opt = options[gid];

    float S = opt.spot_price;
    float K = opt.strike_price;
    float r = opt.risk_free_rate;
    float sigma = opt.volatility;
    float T = opt.time_to_maturity;
    bool is_call = (opt.is_call == 1);

    float dt = T / float(num_steps);
    float u = exp(sigma * sqrt(3.0f * dt));  // up factor
    float d = 1.0f / u;                      // down factor
    float m = 1.0f;                          // middle factor

    float pu = 0.5f * ((sigma * sigma * dt + (r - 0.5f * sigma * sigma) * (r - 0.5f * sigma * sigma) * dt * dt) / (sigma * sigma * dt) + (r - 0.5f * sigma * sigma) * dt / (sigma * sqrt(dt)));
    float pd = 0.5f * ((sigma * sigma * dt + (r - 0.5f * sigma * sigma) * (r - 0.5f * sigma * sigma) * dt * dt) / (sigma * sigma * dt) - (r - 0.5f * sigma * sigma) * dt / (sigma * sqrt(dt)));
    float pm = 1.0f - pu - pd;

    float disc = exp(-r * dt);

    // Option values array - trinomial requires 2*n+1 values
    float option_values[513]; // Up to 256 steps

    uint max_nodes = 2 * num_steps + 1;

    // Initialize option values at maturity
    for (uint i = 0; i < max_nodes; i++) {
        int j = int(i) - int(num_steps); // j ranges from -num_steps to +num_steps
        float stock_price = S * pow(u, max(j, 0)) * pow(d, max(-j, 0));

        if (is_call) {
            option_values[i] = max(0.0f, stock_price - K);
        } else {
            option_values[i] = max(0.0f, K - stock_price);
        }
    }

    // Backward induction
    for (int step = int(num_steps) - 1; step >= 0; step--) {
        for (int i = 0; i <= 2 * step; i++) {
            option_values[i] = disc * (pu * option_values[i + 2] +
                                     pm * option_values[i + 1] +
                                     pd * option_values[i]);
        }
    }

    option_prices[gid] = option_values[0];
}