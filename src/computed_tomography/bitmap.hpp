#pragma once

#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdint>

#pragma pack(push, 1)
struct BMPHeader {
    uint16_t signature;
    uint32_t fileSize;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t dataOffset;
    uint32_t headerSize;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bitsPerPixel;
    uint32_t compression;
    uint32_t imageSize;
    int32_t xPixelsPerMeter;
    int32_t yPixelsPerMeter;
    uint32_t colorsUsed;
    uint32_t colorsImportant;
};
#pragma pack(pop)

class bitmap_image {
private:
    std::vector<unsigned char> data;
    size_t w, h;
    bool valid;

public:
    bitmap_image() : w(0), h(0), valid(false) {}

    bitmap_image(size_t width, size_t height) 
        : w(width), h(height), valid(true) {
        data.resize(width * height * 3, 0);
    }

    bitmap_image(const char* filename) : valid(false) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) return;

        BMPHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));

        if (header.signature != 0x4D42 || header.compression != 0) return;

        w = header.width;
        h = header.height;
        data.resize(w * h * 3);

        // Skip to pixel data
        file.seekg(header.dataOffset, std::ios::beg);

        // Read pixel data
        size_t padding = (4 - (w * 3) % 4) % 4;
        for (int y = h - 1; y >= 0; --y) {
            file.read(reinterpret_cast<char*>(data.data() + y * w * 3), w * 3);
            file.seekg(padding, std::ios::cur);
        }

        valid = true;
    }

    void save_image(const char* filename) {
        if (!valid) return;

        std::ofstream file(filename, std::ios::binary);
        if (!file) return;

        size_t padding = (4 - (w * 3) % 4) % 4;
        size_t paddedRowSize = w * 3 + padding;
        size_t fileSize = sizeof(BMPHeader) + h * paddedRowSize;

        BMPHeader header = {
            0x4D42,                    // signature
            (uint32_t)fileSize,        // fileSize
            0,                         // reserved1
            0,                         // reserved2
            sizeof(BMPHeader),         // dataOffset
            40,                        // headerSize
            (int32_t)w,               // width
            (int32_t)h,               // height
            1,                         // planes
            24,                        // bitsPerPixel
            0,                         // compression
            (uint32_t)(h * paddedRowSize), // imageSize
            2835,                      // xPixelsPerMeter
            2835,                      // yPixelsPerMeter
            0,                         // colorsUsed
            0                          // colorsImportant
        };

        file.write(reinterpret_cast<char*>(&header), sizeof(header));

        std::vector<char> pad(padding, 0);
        for (int y = h - 1; y >= 0; --y) {
            file.write(reinterpret_cast<char*>(data.data() + y * w * 3), w * 3);
            if (padding > 0) {
                file.write(pad.data(), padding);
            }
        }
    }

    void get_pixel(size_t x, size_t y, unsigned char& r, unsigned char& g, unsigned char& b) const {
        if (!valid || x >= w || y >= h) {
            r = g = b = 0;
            return;
        }
        size_t pos = (y * w + x) * 3;
        b = data[pos];
        g = data[pos + 1];
        r = data[pos + 2];
    }

    void set_pixel(size_t x, size_t y, unsigned char r, unsigned char g, unsigned char b) {
        if (!valid || x >= w || y >= h) return;
        size_t pos = (y * w + x) * 3;
        data[pos] = b;
        data[pos + 1] = g;
        data[pos + 2] = r;
    }

    size_t width() const { return w; }
    size_t height() const { return h; }
    operator bool() const { return valid; }
};
