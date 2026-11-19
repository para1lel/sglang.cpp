#include "sglang/vision/detail.cuh"

#include <math_constants.h>

namespace sglang::vision_detail {
namespace {

constexpr int rotary_width = head_width / 2;

int blocks(int count) {
    return (count + threads - 1) / threads;
}

__device__ int2 coordinates(Grid grid, int index) {
    const int patch = index % grid.frame();
    const int block = patch / 4;
    return make_int2(2 * (block / (grid.width / 2)) + (patch / 2) % 2,
                     2 * (block % (grid.width / 2)) + patch % 2);
}

__device__ float position_coordinate(int index, int count) {
    constexpr float end = table_side - 1;
    const float step = end / (count - 1);
    return index < count / 2 ? step * index : fmaf(-step, count - index - 1, end);
}

__global__ void position_kernel(const Bf16* table, Bf16* output, Grid grid) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= grid.rows * hidden) {
        return;
    }
    const int2 point = coordinates(grid, index / hidden);
    const float source_h = position_coordinate(point.x, grid.height);
    const float source_w = position_coordinate(point.y, grid.width);
    const int low_h = static_cast<int>(source_h);
    const int low_w = static_cast<int>(source_w);
    const int high_h = min(low_h + 1, table_side - 1);
    const int high_w = min(low_w + 1, table_side - 1);
    const float dh = source_h - low_h;
    const float dw = source_w - low_w;
    const float bottom_right = dh * dw;
    const float bottom_left = dh - bottom_right;
    const float top_right = dw - bottom_right;
    const float top_left = 1.0f - dh - top_right;
    const int channel = index % hidden;
    const int offsets[] = {low_h * table_side + low_w, low_h * table_side + high_w,
                           high_h * table_side + low_w, high_h * table_side + high_w};
    const float weights[] = {top_left, top_right, bottom_left, bottom_right};
    float value = 0.0f;
    for (int corner = 0; corner < 4; ++corner) {
        const auto product = __hmul(table[offsets[corner] * hidden + channel],
                                   __float2bfloat16_rn(weights[corner]));
        value += __bfloat162float(product);
    }
    output[index] = __float2bfloat16_rn(value);
}

__global__ void rotary_kernel(float2* table, Grid grid) {
    constexpr int frequencies = rotary_width / 2;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= grid.frame() * rotary_width) {
        return;
    }
    const int2 point = coordinates(grid, index / rotary_width);
    const int frequency = index % rotary_width;
    const float fraction = static_cast<float>(frequency % frequencies) / frequencies;
    const float inverse = 1.0f / powf(10000.0f, fraction);
    const float angle = (frequency < frequencies ? point.x : point.y) * inverse;
    sincosf(angle, &table[index].y, &table[index].x);
}

__global__ void rotate_kernel(const Bf16* qkv, const float2* table, Bf16* qk, Grid grid) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= grid.rows * 2 * hidden) {
        return;
    }
    const int row = index / (2 * hidden);
    const int local = index % (2 * hidden);
    const int channel = local % head_width;
    const int source = row * 3 * hidden + local;
    const float2 angle = table[(row % grid.frame()) * rotary_width + channel % rotary_width];
    const float partner = channel < rotary_width ? -__bfloat162float(qkv[source + rotary_width])
                                                : __bfloat162float(qkv[source - rotary_width]);
    qk[index] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(qkv[source]), angle.x) +
                                   __fmul_rn(partner, angle.y));
}

template <bool maximum>
__device__ double reduce(double value, double* partial) {
    const int lane = threadIdx.x % 32;
    for (int delta = 16; delta; delta /= 2) {
        const double peer = __shfl_down_sync(0xffffffff, value, delta);
        value = maximum ? fmax(value, peer) : value + peer;
    }
    if (lane == 0) {
        partial[threadIdx.x / 32] = value;
    }
    __syncthreads();
    double total = partial[0];
    for (int warp = 1; warp < threads / 32; ++warp) {
        total = maximum ? fmax(total, partial[warp]) : total + partial[warp];
    }
    __syncthreads();
    return total;
}

__global__ void attention_kernel(const Bf16* qk, const Bf16* qkv, Bf16* context, int frame) {
    constexpr double scale = 0.11785113019775792;
    __shared__ double probabilities[threads];
    __shared__ double partial[threads / 32];
    const int row = blockIdx.x;
    const int head = blockIdx.y * head_width;
    const int first = row / frame * frame;
    const int channel = head + threadIdx.x;
    const Bf16* query = qk + row * 2 * hidden + head;
    double running_max = -CUDART_INF;
    double denominator = 0.0;
    double numerator = 0.0;

    for (int start = 0; start < frame; start += threads) {
        const int count = min(threads, frame - start);
        double score = -CUDART_INF;
        if (threadIdx.x < count) {
            const Bf16* key = qk + (first + start + threadIdx.x) * 2 * hidden + hidden + head;
            double dot = 0.0;
            for (int inner = 0; inner < head_width; ++inner) {
                dot = __fma_rn(__bfloat162float(query[inner]), __bfloat162float(key[inner]), dot);
            }
            score = dot * scale;
        }
        const double next_max = fmax(running_max, reduce<true>(score, partial));
        const double carry = exp(running_max - next_max);
        const double probability = exp(score - next_max);
        probabilities[threadIdx.x] = probability;
        denominator = __fma_rn(denominator, carry, reduce<false>(probability, partial));
        if (threadIdx.x < head_width) {
            double tile = 0.0;
            for (int key_row = 0; key_row < count; ++key_row) {
                const int value = (first + start + key_row) * 3 * hidden + 2 * hidden + channel;
                tile = __fma_rn(probabilities[key_row], __bfloat162float(qkv[value]), tile);
            }
            numerator = __fma_rn(numerator, carry, tile);
        }
        running_max = next_max;
    }
    if (threadIdx.x < head_width) {
        context[row * hidden + channel] =
            __float2bfloat16_rn(static_cast<float>(numerator / denominator));
    }
}

}

void positions(const Bf16* table, Bf16* output, Grid grid, cudaStream_t stream) {
    position_kernel<<<blocks(grid.rows * hidden), threads, 0, stream>>>(table, output, grid);
    check(cudaGetLastError(), "Interpolate vision positions");
}

void make_rotary(float2* table, Grid grid, cudaStream_t stream) {
    rotary_kernel<<<blocks(grid.frame() * rotary_width), threads, 0, stream>>>(table, grid);
    check(cudaGetLastError(), "Build vision rotary table");
}

void rotate(const Bf16* qkv, const float2* table, Bf16* qk, Grid grid, cudaStream_t stream) {
    rotate_kernel<<<blocks(grid.rows * 2 * hidden), threads, 0, stream>>>(qkv, table, qk, grid);
    check(cudaGetLastError(), "Rotate vision queries and keys");
}

void attend(const Bf16* qk, const Bf16* qkv, Bf16* context, Grid grid, cudaStream_t stream) {
    attention_kernel<<<dim3(grid.rows, heads), threads, 0, stream>>>(
        qk, qkv, context, grid.frame());
    check(cudaGetLastError(), "Attend within vision temporal pairs");
}

}
