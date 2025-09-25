using Test
using oneAPI

@testset "GPU Tests" begin
    # Test GPU availability
    devices = oneapi_get_devices()
    @test !isempty(devices)
    
    # Find Intel Arc A770
    gpu = nothing
    for dev in devices
        if contains(lowercase(oneapi_device_name(dev)), "a770")
            gpu = dev
            break
        end
    end
    
    @test gpu !== nothing
    @test oneapi_device_type(gpu) == oneAPI.CL_DEVICE_TYPE_GPU
end
