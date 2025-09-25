using oneAPI
using LinearAlgebra

function safe_gpu_operation(operation, args...)
    try
        result = operation(args...)
        oneAPI.synchronize()
        return result
    catch e
        println("GPU operation failed: ", typeof(e))
        rethrow(e)
    end
end

function conjugate_gradient_cpu(A::Matrix{Float32}, b::Vector{Float32}, max_iter=1000, tol=1e-10)
    x = zeros(Float32, length(b))
    r = b - A * x
    p = copy(r)
    rsold = dot(r, r)

    for i in 1:max_iter
        Ap = A * p
        alpha = rsold / dot(p, Ap)
        x .+= alpha .* p
        r .-= alpha .* Ap
        rsnew = dot(r, r)
        if sqrt(rsnew) < tol
            return x, i
        end
        p .= r .+ (rsnew / rsold) .* p
        rsold = rsnew
    end
    return x, max_iter
end

function conjugate_gradient_gpu(A::Matrix{Float32}, b::Vector{Float32}, max_iter=1000, tol=1e-10)
    n = length(b)
    println("Starting GPU conjugate gradient solver for size: $n")
    
    try
        # Initialize on CPU first
        x = zeros(Float32, n)
        r = copy(b)  # r = b - A*x, but x is zero initially
        p = copy(r)
        
        # Transfer to GPU with explicit synchronization
        println("Transferring data to GPU...")
        d_A = oneArray(A)
        oneAPI.synchronize()
        d_b = oneArray(b)
        oneAPI.synchronize()
        d_x = oneArray(x)
        oneAPI.synchronize()
        d_r = oneArray(r)
        oneAPI.synchronize()
        d_p = oneArray(p)
        oneAPI.synchronize()
        println("Data transfer complete")
        
        # Initial residual norm
        h_r = Array(d_r)
        r_norm = sqrt(dot(h_r, h_r))
        println("Initial residual norm: ", r_norm)
        
        for i in 1:max_iter
            # Matrix-vector product A*p using CPU
            h_p = Array(d_p)
            h_Ap = A * h_p
            d_Ap = oneArray(h_Ap)
            oneAPI.synchronize()
            
            # Compute step size alpha
            h_p = Array(d_p)
            h_Ap = Array(d_Ap)
            h_r = Array(d_r)
            
            p_Ap = dot(h_p, h_Ap)
            r_r = dot(h_r, h_r)
            alpha = r_r / p_Ap
            
            if !isfinite(alpha)
                println("Non-finite alpha detected: ", alpha)
                println("p_Ap = ", p_Ap)
                println("r_r = ", r_r)
                break
            end
            
            # Update solution and residual
            d_x .+= alpha .* d_p
            d_r .-= alpha .* d_Ap
            oneAPI.synchronize()
            
            # Check convergence
            h_r = Array(d_r)
            r_norm_new = sqrt(dot(h_r, h_r))
            
            if i % 10 == 0
                println("Iteration $i, residual: $r_norm_new")
            end
            
            if r_norm_new < tol
                println("Converged after $i iterations")
                return Array(d_x), i
            end
            
            # Update search direction
            beta = dot(h_r, h_r) / r_r
            d_p .= d_r .+ beta .* d_p
            oneAPI.synchronize()
            
            if !isfinite(beta)
                println("Non-finite beta detected: ", beta)
                break
            end
        end
        
        println("Maximum iterations reached")
        return Array(d_x), max_iter
    catch e
        println("GPU computation failed: ", e)
        println("Error type: ", typeof(e))
        rethrow(e)
    end
end

function test_conjugate_gradient()
    # Create a small test problem
    n = 5  # Reduced size for testing
    A = rand(Float32, n, n)
    A = A * A' + n * I  # Make symmetric positive definite
    x_true = ones(Float32, n)
    b = A * x_true
    
    println("\nTesting conjugate gradient solver")
    println("Matrix size: $n x $n")
    println("Condition number: ", cond(A))
    
    # Solve using CPU implementation first
    println("\nSolving with CPU implementation:")
    x_cpu, iters_cpu = conjugate_gradient_cpu(A, b)
    println("CPU iterations: ", iters_cpu)
    println("CPU relative error: ", norm(x_cpu - x_true) / norm(x_true))
    
    # Solve using GPU implementation
    println("\nSolving with GPU implementation:")
    x_gpu, iters_gpu = conjugate_gradient_gpu(A, b)
    
    # Check solution
    println("\nSolution verification:")
    println("True solution: ", x_true)
    println("CPU solution: ", x_cpu)
    println("GPU solution: ", x_gpu)
    println("CPU-GPU difference: ", norm(x_cpu - x_gpu))
    println("GPU relative error: ", norm(x_gpu - x_true) / norm(x_true))
    println("GPU residual norm: ", norm(A * x_gpu - b))
end

# Run test with exception handling
try
    test_conjugate_gradient()
catch e
    println("Test failed: ", e)
    println("Error type: ", typeof(e))
end
