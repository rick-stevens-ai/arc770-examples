using oneAPI
using Statistics
using Random

function student_t_test_cpu(x::Vector{Float32}, y::Vector{Float32})
    nx, ny = length(x), length(y)
    mx, my = mean(x), mean(y)
    vx, vy = var(x), var(y)
    
    # Pooled standard error
    se = sqrt((vx/nx + vy/ny))
    
    # T statistic
    t = (mx - my) / se
    
    # Degrees of freedom (Welch-Satterthwaite approximation)
    df = (vx/nx + vy/ny)^2 / ((vx/nx)^2/(nx-1) + (vy/ny)^2/(ny-1))
    
    return t, df
end

function run_benchmark()
    # Test parameters
    sizes = [1000, 10000, 100000]
    
    for n in sizes
        println("\nTesting with sample size: $n")
        
        # Generate test data
        rng = MersenneTwister(42)
        x = randn(rng, Float32, n)
        y = randn(rng, Float32, n) .+ 0.5f0  # Add effect size
        
        # CPU implementation
        t_cpu = @elapsed t_stat, df = student_t_test_cpu(x, y)
        println("CPU time: ", t_cpu, " seconds")
        println("t-statistic: ", t_stat)
        println("degrees of freedom: ", df)
    end
end

# Run benchmark
try
    run_benchmark()
catch e
    println("Benchmark failed: ", e)
    println("Error type: ", typeof(e))
end
