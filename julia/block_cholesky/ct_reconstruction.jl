using oneAPI
using Images
using TestImages
using FFTW
using Interpolations
using Plots
using Statistics
using ImagePhantoms: shepp_logan

# Load and prepare test image
function prepare_image(size::Int=256)
    phantom = Float32.(shepp_logan(size))
    return phantom
end

# CPU implementation of Radon transform
function radon_transform_cpu(image::Matrix{Float32}, angles::Vector{Float32})
    height, width = size(image)
    diagonal = ceil(Int, sqrt(2f0 * max(height, width)))
    projections = zeros(Float32, diagonal, length(angles))
    
    center_x = Float32((width - 1) / 2)
    center_y = Float32((height - 1) / 2)
    
    for (angle_idx, θ) in enumerate(angles)
        sin_θ = sin(θ)
        cos_θ = cos(θ)
        
        for r_idx in 1:diagonal
            r = Float32(r_idx) - Float32(diagonal)/2f0
            acc = 0.0f0
            
            for x in 1:width
                tx = Float32(x) - center_x
                for y in 1:height
                    ty = Float32(y) - center_y
                    ray_r = tx * cos_θ + ty * sin_θ
                    if abs(ray_r - r) < 0.5f0
                        acc += image[y, x]
                    end
                end
            end
            projections[r_idx, angle_idx] = acc
        end
    end
    
    return projections
end

# GPU implementation of Radon transform
function radon_transform_gpu(image::Matrix{Float32}, angles::Vector{Float32})
    try
        height, width = size(image)
        diagonal = ceil(Int, sqrt(2f0 * max(height, width)))
        
        # Transfer data to GPU
        image_gpu = oneArray(image)
        projections_gpu = oneArray(zeros(Float32, diagonal, length(angles)))
        
        center_x = Float32((width - 1) / 2)
        center_y = Float32((height - 1) / 2)
        
        # Create kernel for parallel computation
        function kernel(projections, image, θ, angle_idx, center_x, center_y, height, width)
            idx = get_global_id()
            r = Float32(idx) - Float32(size(projections, 1))/2f0
            
            if idx <= size(projections, 1)
                sin_θ = sin(Float32(θ))
                cos_θ = cos(Float32(θ))
                
                acc = 0.0f0
                for x in 1:width
                    tx = Float32(x) - center_x
                    for y in 1:height
                        ty = Float32(y) - center_y
                        ray_r = tx * cos_θ + ty * sin_θ
                        if abs(ray_r - r) < 0.5f0
                            acc += image[y, x]
                        end
                    end
                end
                projections[idx, angle_idx] = acc
            end
            return nothing
        end

        # Execute kernel for each angle
        for (angle_idx, θ) in enumerate(angles)
            @oneapi items=(diagonal) kernel(
                projections_gpu, image_gpu, θ, angle_idx,
                center_x, center_y, height, width
            )
            synchronize()
        end
        
        # Transfer result back to CPU
        projections = Array(projections_gpu)
        return projections
    catch e
        println("GPU computation failed: ", e)
        println("Error type: ", typeof(e))
        return nothing
    end
end

# Filtered backprojection for image reconstruction
function filtered_backprojection(sinogram::Matrix{Float32}, angles::Vector{Float32}, image_size::Int)
    filter = Float32.(abs.(fftfreq(size(sinogram, 1))))
    filtered = zeros(Float32, size(sinogram))
    
    for i in 1:size(sinogram, 2)
        projection = sinogram[:, i]
        filtered[:, i] = real.(ifft(fft(projection) .* filter))
    end
    
    reconstructed = zeros(Float32, image_size, image_size)
    center = Float32((image_size - 1) / 2)
    
    for x in 1:image_size
        for y in 1:image_size
            for (i, θ) in enumerate(angles)
                tx = Float32(x) - center
                ty = Float32(y) - center
                r = tx * cos(θ) + ty * sin(θ)
                r_idx = round(Int, r + size(filtered, 1)/2)
                if 1 <= r_idx <= size(filtered, 1)
                    reconstructed[y, x] += filtered[r_idx, i]
                end
            end
        end
    end
    
    return reconstructed ./ Float32(length(angles))
end

# Main function to run the example
function run_ct_example()
    image_size = 256
    n_angles = 180
    angles = Float32.(range(0, Float32(π), length=n_angles))

    println("Preparing test image...")
    original = prepare_image(image_size)

    println("\nPerforming CPU Radon transform...")
    @time cpu_projections = radon_transform_cpu(original, angles)

    println("\nPerforming GPU Radon transform...")
    @time begin
        gpu_projections = radon_transform_gpu(original, angles)
        if isnothing(gpu_projections)
            println("GPU implementation failed, skipping comparison")
            return
        end
    end

    println("\nPerforming image reconstruction...")
    @time reconstructed = filtered_backprojection(cpu_projections, angles, image_size)

    p1 = heatmap(original, title="Original", color=:grays)
    p2 = heatmap(cpu_projections, title="Sinogram (CPU)", color=:grays)
    p3 = heatmap(gpu_projections, title="Sinogram (GPU)", color=:grays)
    p4 = heatmap(reconstructed, title="Reconstructed", color=:grays)

    plot(p1, p2, p3, p4, layout=(2,2), size=(800,800))
    savefig("ct_results.png")

    error_cpu_gpu = mean(abs.(cpu_projections - gpu_projections))
    println("\nMean absolute difference between CPU and GPU projections: ", error_cpu_gpu)
end

println("Initializing oneAPI...")
try
    devices = oneAPI.devices()
    for (i, dev) in enumerate(devices)
        println("Device $i: ", dev)
    end
    run_ct_example()
catch e
    println("Error initializing oneAPI: ", e)
    rethrow(e)
end
