using oneAPI

function check_device()
    println("Checking oneAPI Device Properties:")
    println("=================================")
    
    # Create a simple array to force device initialization
    A = oneArray(zeros(Float32, 1))
    
    # Print device info
    println("\nDevice Information:")
    # Try to get work-group size through a simple kernel
    function kernel(A)
        gid = get_global_id(0)
        wgs = get_local_size(0)
        if gid == 1
            A[1] = wgs
        end
        return nothing
    end

    # Try different work-group sizes
    println("\nTesting work-group sizes:")
    test_sizes = [8, 16, 32, 64, 128, 256]
    for size in test_sizes
        try
            @oneapi items=size groups=size kernel(A)
            println("Work-group size $size: Supported")
        catch e
            println("Work-group size $size: Not supported ($e)")
        end
    end
end

check_device()
