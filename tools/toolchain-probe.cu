#include <cuda_runtime.h>

namespace {
__device__ int result;

__global__ void add_one(int input) {
    result = input + 1;
}
}

cudaError_t run_device_probe(int input, int* output) {
    add_one<<<1, 1>>>(input);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    return cudaMemcpyFromSymbol(output, result, sizeof(*output));
}
