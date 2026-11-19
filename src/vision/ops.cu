#include "sglang/vision/detail.cuh"

namespace sglang::vision_detail {
namespace {

constexpr int warp_size = 32;
constexpr int outputs_per_block = threads / warp_size;

int blocks(int count) {
    return (count + threads - 1) / threads;
}

__global__ void cast_kernel(const float* input, Bf16* output, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = __float2bfloat16_rn(input[index]);
    }
}

__global__ void linear_kernel(const Bf16* input, Affine weights, Bf16* output,
                              int input_width, int output_width) {
    const int row = blockIdx.y;
    const int column = blockIdx.x * outputs_per_block + threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    if (column >= output_width) {
        return;
    }
    double dot = 0.0;
    for (int channel = lane; channel < input_width; channel += warp_size) {
        const double activation = __bfloat162float(input[row * input_width + channel]);
        const double weight = __bfloat162float(weights.weight[column * input_width + channel]);
        dot = __fma_rn(activation, weight, dot);
    }
    for (int stride = warp_size / 2; stride; stride /= 2) {
        dot += __shfl_down_sync(0xffffffff, dot, stride);
    }
    if (lane == 0) {
        output[row * output_width + column] =
            __float2bfloat16_rn(static_cast<float>(dot + __bfloat162float(weights.bias[column])));
    }
}

__global__ void add_kernel(Bf16* destination, const Bf16* value, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        destination[index] = __hadd(destination[index], value[index]);
    }
}

template <bool approximate>
__global__ void gelu_kernel(Bf16* data, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = __bfloat162float(data[index]);
        const float gate = approximate
            ? tanhf(0.7978845608028654f * (value + 0.044715f * value * value * value))
            : erff(value * 0.7071067811865475f);
        data[index] = __float2bfloat16_rn(0.5f * value * (1.0f + gate));
    }
}

__device__ double block_sum(double value, double* partial) {
    const int lane = threadIdx.x % warp_size;
    for (int delta = warp_size / 2; delta; delta /= 2) {
        value += __shfl_down_sync(0xffffffff, value, delta);
    }
    if (lane == 0) {
        partial[threadIdx.x / warp_size] = value;
    }
    __syncthreads();
    double total = partial[0];
    for (int warp = 1; warp < outputs_per_block; ++warp) {
        total += partial[warp];
    }
    __syncthreads();
    return total;
}

__global__ void norm_kernel(const Bf16* input, Affine weights, Bf16* output) {
    __shared__ double partial[outputs_per_block];
    const int start = blockIdx.x * hidden;
    double sum = 0.0;
    for (int channel = threadIdx.x; channel < hidden; channel += threads) {
        sum += __bfloat162float(input[start + channel]);
    }
    const double mean = block_sum(sum, partial) / hidden;
    double squares = 0.0;
    for (int channel = threadIdx.x; channel < hidden; channel += threads) {
        const double delta = __bfloat162float(input[start + channel]) - mean;
        squares = __fma_rn(delta, delta, squares);
    }
    const double inverse = rsqrt(block_sum(squares, partial) / hidden + 1.0e-6);
    for (int channel = threadIdx.x; channel < hidden; channel += threads) {
        const double value = (__bfloat162float(input[start + channel]) - mean) * inverse;
        const double normalized = __fma_rn(value, __bfloat162float(weights.weight[channel]),
                                          __bfloat162float(weights.bias[channel]));
        output[start + channel] = __float2bfloat16_rn(static_cast<float>(normalized));
    }
}

}

void cast_patches(const float* input, Bf16* output, int count, cudaStream_t stream) {
    cast_kernel<<<blocks(count), threads, 0, stream>>>(input, output, count);
    check(cudaGetLastError(), "Cast vision patches");
}

void linear(const Bf16* input, Affine weights, Bf16* output, int rows,
            int input_width, int output_width, cudaStream_t stream) {
    const dim3 tiles((output_width + outputs_per_block - 1) / outputs_per_block, rows);
    linear_kernel<<<tiles, threads, 0, stream>>>(input, weights, output, input_width, output_width);
    check(cudaGetLastError(), "Project vision rows");
}

void add(Bf16* destination, const Bf16* value, int count, cudaStream_t stream) {
    add_kernel<<<blocks(count), threads, 0, stream>>>(destination, value, count);
    check(cudaGetLastError(), "Add vision residual");
}

void normalize(const Bf16* input, Affine weights, Bf16* output, int rows, cudaStream_t stream) {
    norm_kernel<<<rows, threads, 0, stream>>>(input, weights, output);
    check(cudaGetLastError(), "Normalize vision rows");
}

void gelu(Bf16* data, int count, bool approximate, cudaStream_t stream) {
    if (approximate) {
        gelu_kernel<true><<<blocks(count), threads, 0, stream>>>(data, count);
    } else {
        gelu_kernel<false><<<blocks(count), threads, 0, stream>>>(data, count);
    }
    check(cudaGetLastError(), "Apply vision GELU");
}

}
