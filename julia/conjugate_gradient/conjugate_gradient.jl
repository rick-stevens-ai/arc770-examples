using oneAPI
using LinearAlgebra

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

function test_conjugate_gradient()
    # Create a small test problem
    n = 5  # Small size for testing
    A = rand(Float32, n, n)
    A = A * A' + n * I  # Make symmetric positive definite
    x_true = ones(Float32, n)
    b = A * x_true
    
    println("\nTesting conjugate gradient solver")
    println("Matrix size: $n x $n")
    println("Condition number: ", cond(A))
    
    # Solve using CPU implementation
    println("\nSolving with CPU implementation:")
    x_cpu, iters = conjugate_gradient_cpu(A, b)
    
    # Check solution
    println("\nSolution verification:")
    println("True solution: ", x_true)
    println("CPU solution: ", x_cpu)
    println("Relative error: ", norm(x_cpu - x_true) / norm(x_true))
    println("Residual norm: ", norm(A * x_cpu - b))
end

# Run test with exception handling
try
    test_conjugate_gradient()
catch e
    println("Test failed: ", e)
    println("Error type: ", typeof(e))
end
