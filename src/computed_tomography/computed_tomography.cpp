#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
#include "oneapi/mkl.hpp"
#include "bitmap.hpp"

using namespace sycl;
using namespace oneapi::mkl;
using std::cout;
using std::cerr;
using std::vector;

int main(int argc, char **argv) {
    if (argc < 6) {
        cout << "Usage: " << argv[0]
             << " <width> <height> <input.bmp> <radon.bmp> <restored.bmp>\n\n";
        return 0;
    }

    size_t width = atoi(argv[1]);
    size_t height = atoi(argv[2]);
    char *input_file = argv[3];
    char *radon_file = argv[4];
    char *output_file = argv[5];

    constexpr size_t angles = 180;
    constexpr float pi = 3.14159265358979323846f;

    bitmap_image input(input_file);
    if (!input) {
        cout << "Error opening input image\n";
        return 1;
    }

    if (input.width() != width || input.height() != height) {
        cout << "Incorrect image dimensions\n";
        return 1;
    }

    // Create SYCL queue
    queue main_queue{default_selector_v};
    cout << "Running on: " << main_queue.get_device().get_info<info::device::name>() << "\n";

    vector<float> sinogram(width * angles);
    vector<float> restored(width * height);
    vector<float> image(width * height);

    // Copy input image to vector
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            unsigned char r, g, b;
            input.get_pixel(j, i, r, g, b);
            image[i * width + j] = (r + g + b) / 3.0f;
        }
    }

    // Prepare angles
    vector<float> theta(angles);
    for (size_t i = 0; i < angles; i++) {
        theta[i] = i * pi / angles;
    }

    // Create dimensions for transforms
    vector<int64_t> forward_dims = {(int64_t)height, (int64_t)width};
    vector<int64_t> backward_dims = {(int64_t)width, (int64_t)height};

    try {
        // Create and configure DFT descriptors
        auto forward_desc = dft::descriptor<dft::precision::SINGLE, dft::domain::REAL>(forward_dims);
        forward_desc.commit(main_queue);

        auto backward_desc = dft::descriptor<dft::precision::SINGLE, dft::domain::REAL>(backward_dims);
        backward_desc.commit(main_queue);

        // Compute forward transform
        auto forward_event = dft::compute_forward(forward_desc, image.data(), sinogram.data());
        forward_event.wait();

        // Save sinogram to bitmap
        bitmap_image radon(angles, width);
        float max_val = 0;
        for (size_t i = 0; i < width * angles; i++) {
            max_val = std::max(max_val, std::abs(sinogram[i]));
        }
        if (max_val > 0) {
            for (size_t i = 0; i < width; i++) {
                for (size_t j = 0; j < angles; j++) {
                    unsigned char val = (unsigned char)(std::abs(sinogram[j * width + i]) * 255 / max_val);
                    radon.set_pixel(j, i, val, val, val);
                }
            }
        }
        radon.save_image(radon_file);

        // Compute inverse transform
        auto backward_event = dft::compute_backward(backward_desc, sinogram.data(), restored.data());
        backward_event.wait();

        // Save restored image to bitmap
        bitmap_image output(width, height);
        max_val = 0;
        for (size_t i = 0; i < width * height; i++) {
            max_val = std::max(max_val, std::abs(restored[i]));
        }
        if (max_val > 0) {
            for (size_t i = 0; i < height; i++) {
                for (size_t j = 0; j < width; j++) {
                    unsigned char val = (unsigned char)(std::abs(restored[i * width + j]) * 255 / max_val);
                    output.set_pixel(j, i, val, val, val);
                }
            }
        }
        output.save_image(output_file);

        cout << "Finished successfully\n";
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
