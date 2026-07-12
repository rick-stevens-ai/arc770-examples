#include <fstream>
#include <vector>

int main() {
    int size = 16;
    std::vector<float> a, b;

    for (int i = 0; i < size; i++) {
        a.push_back(static_cast<float>(i));
        b.push_back(static_cast<float>(i * 2));
    }

    std::ofstream outfile("test_input.bin", std::ios::binary);
    outfile.write(reinterpret_cast<const char*>(&size), sizeof(int));
    outfile.write(reinterpret_cast<const char*>(a.data()), size * sizeof(float));
    outfile.write(reinterpret_cast<const char*>(b.data()), size * sizeof(float));
    outfile.close();

    return 0;
}