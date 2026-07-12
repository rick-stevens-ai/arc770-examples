#include <metal_stdlib>
using namespace metal;

// Linear Congruential Generator for GPU random numbers
struct SimpleRNG {
    uint state;

    SimpleRNG(uint seed) : state(seed) {}

    float next() {
        state = state * 1103515245u + 12345u;
        return float(state & 0x7fffffffu) / float(0x7fffffffu);
    }
};

// Philox RNG - higher quality random number generator
struct PhiloxRNG {
    uint4 counter;
    uint2 key;

    PhiloxRNG(uint seed, uint index) {
        counter = uint4(index, 0, 0, 0);
        key = uint2(seed, seed ^ 0xdeadbeef);
    }

    uint2 philox_round(uint2 counter, uint2 key) {
        const uint M0 = 0xd2511f53u;
        const uint M1 = 0xcd9e8d57u;
        const uint W0 = 0x9e3779b9u;
        const uint W1 = 0xbb67ae85u;

        uint hi0 = mul_hi(M0, counter.x);
        uint hi1 = mul_hi(M1, counter.y);
        uint lo0 = M0 * counter.x;
        uint lo1 = M1 * counter.y;

        return uint2(hi1 ^ counter.x ^ key.x, hi0 ^ counter.y ^ key.y) ^ uint2(lo1, lo0);
    }

    uint2 philox() {
        uint2 cnt = counter.xy;
        uint2 k = key;

        cnt = philox_round(cnt, k); k += uint2(0x9e3779b9u, 0xbb67ae85u);
        cnt = philox_round(cnt, k); k += uint2(0x9e3779b9u, 0xbb67ae85u);
        cnt = philox_round(cnt, k); k += uint2(0x9e3779b9u, 0xbb67ae85u);
        cnt = philox_round(cnt, k); k += uint2(0x9e3779b9u, 0xbb67ae85u);
        cnt = philox_round(cnt, k); k += uint2(0x9e3779b9u, 0xbb67ae85u);
        cnt = philox_round(cnt, k); k += uint2(0x9e3779b9u, 0xbb67ae85u);
        cnt = philox_round(cnt, k); k += uint2(0x9e3779b9u, 0xbb67ae85u);
        cnt = philox_round(cnt, k); k += uint2(0x9e3779b9u, 0xbb67ae85u);
        cnt = philox_round(cnt, k); k += uint2(0x9e3779b9u, 0xbb67ae85u);
        cnt = philox_round(cnt, k);

        counter.x++;
        return cnt;
    }

    float2 next_float2() {
        uint2 r = philox();
        return float2(r) / float(0xffffffffu);
    }
};

// Simple Monte Carlo Pi estimation
kernel void monte_carlo_pi_simple(device uint* results [[buffer(0)]],
                                 constant uint& samples_per_thread [[buffer(1)]],
                                 constant uint& seed_base [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    SimpleRNG rng(seed_base + gid * 12345u);
    uint inside_count = 0;

    for (uint i = 0; i < samples_per_thread; i++) {
        float x = rng.next() * 2.0f - 1.0f; // -1 to 1
        float y = rng.next() * 2.0f - 1.0f; // -1 to 1

        if (x * x + y * y <= 1.0f) {
            inside_count++;
        }
    }

    results[gid] = inside_count;
}

// High-quality Monte Carlo Pi estimation with Philox RNG
kernel void monte_carlo_pi_philox(device uint* results [[buffer(0)]],
                                 constant uint& samples_per_thread [[buffer(1)]],
                                 constant uint& seed_base [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    PhiloxRNG rng(seed_base, gid);
    uint inside_count = 0;

    for (uint i = 0; i < samples_per_thread; i += 2) {
        float2 point = rng.next_float2() * 2.0f - 1.0f; // -1 to 1

        if (point.x * point.x + point.y * point.y <= 1.0f) {
            inside_count++;
        }

        // Process second point if we have samples left
        if (i + 1 < samples_per_thread) {
            float2 point2 = rng.next_float2() * 2.0f - 1.0f;
            if (point2.x * point2.x + point2.y * point2.y <= 1.0f) {
                inside_count++;
            }
        }
    }

    results[gid] = inside_count;
}

// Parallel reduction for summing results
kernel void parallel_sum(device const uint* input [[buffer(0)]],
                        device uint* output [[buffer(1)]],
                        threadgroup uint* shared_data [[threadgroup(0)]],
                        uint gid [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]],
                        uint group_size [[threads_per_threadgroup]],
                        uint total_threads [[threads_per_grid]]) {

    // Load data into shared memory
    uint value = (gid < total_threads) ? input[gid] : 0;
    shared_data[lid] = value;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform reduction in shared memory
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_data[lid] += shared_data[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes the result for this threadgroup
    if (lid == 0) {
        output[gid / group_size] = shared_data[0];
    }
}

// Monte Carlo integration for arbitrary functions
kernel void monte_carlo_integrate(device float* results [[buffer(0)]],
                                 constant uint& samples_per_thread [[buffer(1)]],
                                 constant uint& seed_base [[buffer(2)]],
                                 constant float& a [[buffer(3)]],    // lower bound
                                 constant float& b [[buffer(4)]],    // upper bound
                                 uint gid [[thread_position_in_grid]]) {
    PhiloxRNG rng(seed_base, gid);
    float sum = 0.0f;
    float range = b - a;

    for (uint i = 0; i < samples_per_thread; i++) {
        float2 random = rng.next_float2();
        float x = a + random.x * range;

        // Example: integrate x^2 from a to b
        float f_x = x * x;
        sum += f_x;
    }

    results[gid] = sum * range / float(samples_per_thread);
}