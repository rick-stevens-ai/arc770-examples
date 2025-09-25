using oneAPI

function vector_add(a, b, c)
    i = get_global_id()
    @inbounds c[i] = a[i] + b[i]
    return
end

function main()
    # Initialize data
    N = 1000
    a = ones(Float32, N)
    b = 2ones(Float32, N)
    c = similar(a)

    # Create oneAPI arrays
    d_a = oneArray(a)
    d_b = oneArray(b)
    d_c = similar(d_a)

    # Launch kernel
    @oneapi items=N vector_add(d_a, d_b, d_c)
    
    # Copy result back to host
    c = Array(d_c)
    
    # Verify result
    println("Test passed: ", all(c .== 3.0f0))
end

main()
