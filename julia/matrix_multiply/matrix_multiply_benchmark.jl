using oneAPI
using LinearAlgebra
using Random
using BenchmarkTools

function matrix_multiply_cpu(A::Matrix{Float32}, B::Matrix{Float32})
    return A * B
end

function matrix_multiply_gpu(A::Matrix{Float32}, B::Matrix{Float32})
    d_A = oneArray(A)
    d_B = oneArray(B)
    d_C = similar(d_A, size(A,1), size(B,2))
    d_C .= d_A * d_B
    oneAPI.synchronize()
    return Array(d_C)
end

function run_benchmark()
    sizes = [32, 64, 128, 256]
    
    println("\nMatrix Multiplication Benchmark")
    println("============================")
    
    for n in sizes
        println("\nTesting size: $(n)x$(n)")
        
        # Generate random matrices
        A = rand(Float32, n, n)
        B = rand(Float32, n, n)
        
        # Warm-up
        C_cpu = matrix_multiply_cpu(A, B)
        C_gpu = try
            matrix_multiply_gpu(A, B)
        catch e
            println("GPU warm-up failed: ", e)
            continue
        end
        
        # CPU timing
        t_cpu = @elapsed C_cpu = matrix_multiply_cpu(A, B)
        
        # GPU timing
        t_gpu = @elapsed C_gpu = matrix_multiply_gpu(A, B)
        
        # Verify results
        diff_norm = norm(C_cpu - C_gpu) / norm(C_cpu)
        
        println("CPU time: $(t_cpu*1000) ms")
        println("GPU time: $(t_gpu*1000) ms")
        println("Relative difference: $diff_norm")
        println("Speedup: $(t_cpu/t_gpu)x")
    end
end

try
    run_benchmark()
catch e
    println("Benchmark failed: ", e)
    println("Error type: ", typeof(e))
end
