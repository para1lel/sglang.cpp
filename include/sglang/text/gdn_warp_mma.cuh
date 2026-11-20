#pragma once

namespace sglang::gdn_chunk::warp_mma {

constexpr int columns = 8;
constexpr int warp_value_tiles = dimension / (columns * warps);

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
struct OperandA {
    unsigned int fragments[4];
};

struct OperandB {
    unsigned int fragments[2];
};

struct Result {
    float fragments[4]{};
};

template<bool Transpose>
__device__ __forceinline__ OperandA load_a(const Bf16* data, int stride, int lane) {
    const int matrix = lane / 8;
    const int row = lane % 8 + 8 * (Transpose ? matrix / 2 : matrix % 2);
    const int column = 8 * (Transpose ? matrix % 2 : matrix / 2);
    const auto address = static_cast<unsigned int>(__cvta_generic_to_shared(data + row * stride +
                                                                           column));
    OperandA operand;
    auto& frag = operand.fragments;
    if constexpr (Transpose) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3]) : "r"(address));
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3]) : "r"(address));
    }
    return operand;
}

__device__ __forceinline__ OperandB load_b(const Bf16* data, int stride, int lane) {
    const auto address = static_cast<unsigned int>(
        __cvta_generic_to_shared(data + (lane % 16) * stride));
    OperandB operand;
    auto& frag = operand.fragments;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(frag[0]), "=r"(frag[1]) : "r"(address));
    return operand;
}

__device__ __forceinline__ void accumulate(Result& result, OperandA lhs, OperandB rhs) {
    auto& frag = result.fragments;
    const auto& first = lhs.fragments;
    const auto& second = rhs.fragments;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(frag[0]), "+f"(frag[1]), "+f"(frag[2]), "+f"(frag[3])
                 : "r"(first[0]), "r"(first[1]), "r"(first[2]), "r"(first[3]),
                   "r"(second[0]), "r"(second[1]));
}

constexpr int key_tiles = dimension / tile;

__device__ __forceinline__ int result_row(int lane, int element) {
    return lane / 4 + (element / 2) * 8;
}

__device__ __forceinline__ int result_column(int lane, int element) {
    return (lane % 4) * 2 + element % 2;
}

__device__ __forceinline__ void store_bf16(Bf16* destination, Result result, int lane) {
    const int column = result_column(lane, 0);
    #pragma unroll
    for (int element = 0; element < 4; element += 2) {
        auto* pair = reinterpret_cast<__nv_bfloat162*>(
            destination + result_row(lane, element) * columns + column);
        *pair = __floats2bfloat162_rn(result.fragments[element], result.fragments[element + 1]);
    }
}

__device__ __forceinline__ void project(const Bf16* qk, const Bf16* state, int lane,
                                        Result (&projection)[2 * time_tiles]) {
    #pragma unroll
    for (int key = 0; key < key_tiles; ++key) {
        const auto rhs = load_b(state + key * tile * columns, columns, lane);
        #pragma unroll
        for (int task = 0; task < 2 * time_tiles; ++task) {
            const auto lhs = load_a<false>(qk + task * tile * key_stride + key * tile,
                                           key_stride, lane);
            accumulate(projection[task], lhs, rhs);
        }
    }
}

__device__ __forceinline__ void solve(const Bf16* matrix, const Bf16* operand, int lane,
                                      Result (&result)[time_tiles]) {
    #pragma unroll
    for (int step = 0; step < time_tiles; ++step) {
        const auto rhs = load_b(operand + step * tile * columns, columns, lane);
        #pragma unroll
        for (int task = step; task < time_tiles; ++task) {
            const auto lhs = load_a<false>(matrix + task * tile * rows + step * tile, rows, lane);
            accumulate(result[task], lhs, rhs);
        }
    }
}

struct Stage {
    alignas(32) Bf16 qk[2 * rows * key_stride];
    alignas(32) Bf16 inverse[rows * rows];
    alignas(32) Bf16 scores[rows * rows];
    float initial[rows];
    float tail[rows];
    float beta[rows];
};

__device__ __forceinline__ void copy_async(Bf16* target, const Bf16* source, int bytes) {
    const auto address = static_cast<unsigned int>(__cvta_generic_to_shared(target));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(address), "l"(source), "r"(bytes) : "memory");
}

__device__ __forceinline__ void prefetch(Stage& target, const Matrices& matrix,
                                         const Bf16* convolved, int first, int valid, int head) {
    constexpr int vectors_per_row = dimension / vector_elements;
    for (int vector = threadIdx.x; vector < 2 * rows * vectors_per_row; vector += threads) {
        const int row = vector / vectors_per_row;
        const int position = row % rows;
        const int column = vector % vectors_per_row * vector_elements;
        const int plane = row / rows;
        const int feature = plane * text_gdn_qk_width + head / group * dimension + column;
        const auto offset = static_cast<std::size_t>(first + min(position, valid - 1));
        const auto* source = convolved + offset * text_gdn_qkv_width + feature;
        copy_async(target.qk + row * key_stride + column, source,
                   position < valid ? sizeof(uint4) : 0);
    }
    for (int index = threadIdx.x * vector_elements; index < rows * rows;
         index += threads * vector_elements) {
        copy_async(target.inverse + index, matrix.inverse + index, sizeof(uint4));
        copy_async(target.scores + index, matrix.scores + index, sizeof(uint4));
    }
    if (threadIdx.x < rows) {
        target.initial[threadIdx.x] = matrix.initial[threadIdx.x];
        target.tail[threadIdx.x] = matrix.tail[threadIdx.x];
        target.beta[threadIdx.x] = matrix.beta[threadIdx.x];
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

constexpr int dynamic_bytes = 2 * sizeof(Stage);

__device__ __forceinline__ void write_observable(const Stage& stage, const Bf16* work,
                                                const Result (&projection)[2 * time_tiles],
                                                int first, int valid, int feature, int lane,
                                                Bf16* observable) {
    Result output[time_tiles];
    solve(stage.scores, work, lane, output);
    #pragma unroll
    for (int task = 0; task < time_tiles; ++task) {
        #pragma unroll
        for (int element = 0; element < 4; ++element) {
            const int position = task * tile + result_row(lane, element);
            if (position < valid) {
                constexpr float query_scale = 0.08838834764831845f;
                const float value = fmaf(stage.initial[position],
                    projection[task].fragments[element], output[task].fragments[element]);
                observable[static_cast<std::size_t>(first + position) * text_gdn_value_width +
                           feature + result_column(lane, element)] =
                    __float2bfloat16(query_scale * value);
            }
        }
    }
}

__device__ __forceinline__ void advance_state(const Bf16* scaled_keys, const Bf16* work,
                                             float retention, int lane,
                                             Result (&accumulator)[key_tiles]) {
    #pragma unroll
    for (int key = 0; key < key_tiles; ++key) {
        #pragma unroll
        for (int element = 0; element < 4; ++element) {
            accumulator[key].fragments[element] *= retention;
        }
    }
    #pragma unroll
    for (int step = 0; step < time_tiles; ++step) {
        const auto rhs = load_b(work + step * tile * columns, columns, lane);
        #pragma unroll
        for (int key = 0; key < key_tiles; ++key) {
            const auto lhs = load_a<true>(scaled_keys + step * tile * key_stride + key * tile,
                                         key_stride, lane);
            accumulate(accumulator[key], lhs, rhs);
        }
    }
}

__global__ void recur_warp(int count, const Bf16* convolved, const Matrices* matrices,
                           float* recurrent, Bf16* observable) {
    alignas(32) extern __shared__ unsigned char dynamic_frames[];
    auto* frames = reinterpret_cast<Stage*>(dynamic_frames);
    __shared__ alignas(32) Bf16 scaled_keys[rows * key_stride];
    __shared__ alignas(32) Bf16 states[warps][dimension * columns];
    __shared__ alignas(32) Bf16 workspace[warps][rows * columns];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int head = blockIdx.x;
    const int feature = head * dimension + (blockIdx.y * warps + warp) * columns;
    auto* carry = recurrent + feature * dimension;
    auto* state = states[warp];
    auto* work = workspace[warp];
    prefetch(frames[0], matrices[head], convolved, 0, min(count, rows), head);
    Result accumulator[key_tiles];
    #pragma unroll
    for (int key = 0; key < key_tiles; ++key) {
        #pragma unroll
        for (int element = 0; element < 4; ++element) {
            accumulator[key].fragments[element] =
                carry[result_column(lane, element) * dimension + key * tile +
                      result_row(lane, element)];
        }
        store_bf16(state + key * tile * columns, accumulator[key], lane);
    }

    for (int first = 0; first < count; first += rows) {
        const int valid = min(count - first, rows);
        const int chunk = first / rows;
        const auto& stage = frames[chunk % 2];
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if (first + rows < count) {
            prefetch(frames[(chunk + 1) % 2], matrices[(chunk + 1) * heads + head], convolved,
                     first + rows, min(count - first - rows, rows), head);
        }

        for (int index = threadIdx.x; index < rows * dimension; index += threads) {
            const int position = index / dimension;
            const int offset = position * key_stride + index % dimension;
            scaled_keys[offset] =
                __float2bfloat16(stage.tail[position] *
                                 __bfloat162float(stage.qk[rows * key_stride + offset]));
        }

        Result projection[2 * time_tiles];
        project(stage.qk, state, lane, projection);
        #pragma unroll
        for (int task = 0; task < time_tiles; ++task) {
            Result residual;
            #pragma unroll
            for (int element = 0; element < 4; ++element) {
                const int position = task * tile + result_row(lane, element);
                if (position < valid) {
                    const int token = first + position;
                    const auto* target = convolved +
                        static_cast<std::size_t>(token) * text_gdn_qkv_width +
                        2 * text_gdn_qk_width + feature;
                    residual.fragments[element] = stage.beta[position] *
                        (__bfloat162float(target[result_column(lane, element)]) -
                         stage.initial[position] *
                         projection[time_tiles + task].fragments[element]);
                }
            }
            store_bf16(work + task * tile * columns, residual, lane);
        }
        __syncwarp();
        Result correction[time_tiles];
        solve(stage.inverse, work, lane, correction);
        __syncwarp();
        #pragma unroll
        for (int task = 0; task < time_tiles; ++task) {
            store_bf16(work + task * tile * columns, correction[task], lane);
        }
        __syncwarp();

        write_observable(stage, work, projection, first, valid, feature, lane, observable);
        __syncthreads();

        advance_state(scaled_keys, work, stage.initial[rows - 1], lane, accumulator);
        if (first + rows < count) {
            #pragma unroll
            for (int key = 0; key < key_tiles; ++key) {
                store_bf16(state + key * tile * columns, accumulator[key], lane);
            }
        }
    }
    #pragma unroll
    for (int key = 0; key < key_tiles; ++key) {
        #pragma unroll
        for (int element = 0; element < 4; ++element) {
            carry[result_column(lane, element) * dimension + key * tile +
                  result_row(lane, element)] = accumulator[key].fragments[element];
        }
    }
}
#else
constexpr int dynamic_bytes = 0;

__global__ void recur_warp(int, const Bf16*, const Matrices*, float*, Bf16*) {
    __trap();
}
#endif

}
