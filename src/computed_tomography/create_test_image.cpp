#include "bitmap.hpp"

int main() {
    const int width = 400;
    const int height = 400;
    bitmap_image image(width, height);

    // Create a simple pattern (a white square in the middle)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (x >= width/4 && x < 3*width/4 && 
                y >= height/4 && y < 3*height/4) {
                image.set_pixel(x, y, 255, 255, 255);
            } else {
                image.set_pixel(x, y, 0, 0, 0);
            }
        }
    }

    image.save_image("input.bmp");
    return 0;
}
