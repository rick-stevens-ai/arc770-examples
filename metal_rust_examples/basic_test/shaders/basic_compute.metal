#include <metal_stdlib>
using namespace metal;

kernel void vector_add(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* c [[buffer(2)]],
                      uint index [[thread_position_in_grid]]) {
    c[index] = a[index] + b[index];
}

kernel void vector_multiply(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* c [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
    c[index] = a[index] * b[index];
}

kernel void parallel_sum_reduction(device const float* input [[buffer(0)]],
                                  device float* output [[buffer(1)]],
                                  threadgroup float* shared_data [[threadgroup(0)]],
                                  uint index [[thread_position_in_grid]],
                                  uint local_index [[thread_position_in_threadgroup]],
                                  uint group_size [[threads_per_threadgroup]]) {
    // Load data into shared memory
    shared_data[local_index] = input[index];

    // Wait for all threads to load data
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform reduction in shared memory
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (local_index < stride) {
            shared_data[local_index] += shared_data[local_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes the result for this threadgroup
    if (local_index == 0) {
        output[index / group_size] = shared_data[0];
    }
}