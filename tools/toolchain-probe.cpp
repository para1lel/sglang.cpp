#include <cstdio>
#include <cuda_runtime_api.h>

cudaError_t run_device_probe(int input, int* output);

int main() {
    int output = 0;
    const auto status = run_device_probe(41, &output);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA probe failed: %s\n", cudaGetErrorString(status));
        return 1;
    }
    std::printf("GPU returned %d (expected 42)\n", output);
    return output == 42 ? 0 : 1;
}
