module OneAPITest

using oneAPI

# Define kernel outside main function
function kernel!(A)
    i = oneAPI.get_global_id()
    @inbounds A[i] = A[i] * 2
    return nothing
end

function run_test()
    println("Testing oneAPI with Intel Arc A770...")
    
    # List available devices
    println("\nAvailable Devices:")
    for (i, dev) in enumerate(oneAPI.devices())
        println("$i: $dev")
    end

    try
        # Create test data
        data = Float32[i for i in 1:16]
        println("\nInput data: ", data)

        # Create GPU array and copy data
        d_data = oneArray(data)

        # Execute kernel using oneAPI
        @oneapi items=length(data) kernel!(d_data)
        synchronize()
        
        # Get results back
        result = Array(d_data)
        println("\nOutput data: ", result)
    catch e
        println("Error: ", e)
        println(stacktrace(catch_backtrace()))
    end
end

export run_test

end
