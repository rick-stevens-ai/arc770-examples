#include <metal_stdlib>
using namespace metal;

// Basic matrix multiplication kernel
kernel void matrix_multiply_basic(device const float* A [[buffer(0)]],
                                 device const float* B [[buffer(1)]],
                                 device float* C [[buffer(2)]],
                                 constant uint& N [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= N || col >= N) return;

    float sum = 0.0;
    for (uint k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Optimized matrix multiplication with tiling and shared memory
kernel void matrix_multiply_tiled(device const float* A [[buffer(0)]],
                                 device const float* B [[buffer(1)]],
                                 device float* C [[buffer(2)]],
                                 constant uint& N [[buffer(3)]],
                                 threadgroup float* shared_A [[threadgroup(0)]],
                                 threadgroup float* shared_B [[threadgroup(1)]],
                                 uint2 gid [[thread_position_in_grid]],
                                 uint2 lid [[thread_position_in_threadgroup]],
                                 uint2 group_size [[threads_per_threadgroup]]) {

    const uint TILE_SIZE = 16; // Must match threadgroup size
    uint row = gid.y;
    uint col = gid.x;

    float sum = 0.0;

    // Process tiles
    for (uint tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile of A into shared memory
        uint a_row = row;
        uint a_col = tile * TILE_SIZE + lid.x;
        if (a_row < N && a_col < N) {
            shared_A[lid.y * TILE_SIZE + lid.x] = A[a_row * N + a_col];
        } else {
            shared_A[lid.y * TILE_SIZE + lid.x] = 0.0;
        }

        // Load tile of B into shared memory
        uint b_row = tile * TILE_SIZE + lid.y;
        uint b_col = col;
        if (b_row < N && b_col < N) {
            shared_B[lid.y * TILE_SIZE + lid.x] = B[b_row * N + b_col];
        } else {
            shared_B[lid.y * TILE_SIZE + lid.x] = 0.0;
        }

        // Wait for all threads to load data
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial sum using shared memory
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += shared_A[lid.y * TILE_SIZE + k] * shared_B[k * TILE_SIZE + lid.x];
        }

        // Wait before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// SIMD-optimized matrix multiplication for smaller matrices
kernel void matrix_multiply_simd(device const float* A [[buffer(0)]],
                                device const float* B [[buffer(1)]],
                                device float* C [[buffer(2)]],
                                constant uint& N [[buffer(3)]],
                                uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= N || col >= N) return;

    float4 sum = float4(0.0);
    uint k;

    // Process 4 elements at a time using SIMD
    for (k = 0; k + 3 < N; k += 4) {
        float4 a_vec = float4(A[row * N + k], A[row * N + k + 1],
                             A[row * N + k + 2], A[row * N + k + 3]);
        float4 b_vec = float4(B[k * N + col], B[(k + 1) * N + col],
                             B[(k + 2) * N + col], B[(k + 3) * N + col]);
        sum += a_vec * b_vec;
    }

    float result = sum.x + sum.y + sum.z + sum.w;

    // Handle remaining elements
    for (; k < N; k++) {
        result += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = result;
}