#include "sglang/text/gdn/gdn_chunk.cuh"

#include <algorithm>
#include <mma.h>
#include <type_traits>

namespace sglang::gdn_chunk {

namespace mma = nvcuda::wmma;

constexpr int dimension = text_gdn_head_dim;
constexpr int warps = 4;
constexpr int threads = warps * 32;
constexpr int time_tiles = rows / tile;
constexpr int key_stride = dimension + tile / 2;
constexpr int vector_elements = sizeof(uint4) / sizeof(Bf16);
static_assert(dimension == warps * 2 * tile);

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
using ARow = mma::fragment<mma::matrix_a, tile, tile, tile, Bf16, mma::row_major>;
using BCol = mma::fragment<mma::matrix_b, tile, tile, tile, Bf16, mma::col_major>;
using Acc = mma::fragment<mma::accumulator, tile, tile, tile, float>;

template <typename S, int PrepareThreads>
__device__ __forceinline__ void stage_qk(Bf16* target, const Bf16* convolved,
                                        int first, int valid, int qk_head) {
    constexpr int vectors_per_row = dimension / vector_elements;
    for (int vector = threadIdx.x; vector < 2 * rows * vectors_per_row;
         vector += PrepareThreads) {
        const int row = vector / vectors_per_row;
        const int position = row % rows;
        const int component = vector % vectors_per_row * vector_elements;
        const int plane = row / rows;
        uint4 value{};
        if (position < valid) {
            const auto* source = convolved + static_cast<std::size_t>(first + position) *
                S::qkv_width + plane * S::qk_width + qk_head * dimension;
            value = *reinterpret_cast<const uint4*>(source + component);
        }
        *reinterpret_cast<uint4*>(target + row * key_stride + component) = value;
    }
}

template<int Stride>
__device__ __forceinline__ void store_grouped_matrices(Matrices& matrix, const Bf16* solution,
                                                        const Bf16* scores, int thread) {
    __syncthreads();
    using Pair = __nv_bfloat162;
    static_assert(rows % 2 == 0 && Stride % 2 == 0);
    for (int index = thread * 2; index < rows * rows; index += 2 * threads) {
        const int offset = index / rows * Stride + index % rows;
        *reinterpret_cast<Pair*>(matrix.inverse + index) =
            *reinterpret_cast<const Pair*>(solution + offset);
        *reinterpret_cast<Pair*>(matrix.scores + index) =
            *reinterpret_cast<const Pair*>(scores + offset);
    }
}
template <int SharedHeads>
__device__ __forceinline__ void segment_qk_dots(const Bf16* qk, float* dots, int warp) {
    constexpr int prepare_warps = warps * SharedHeads;
    for (int task = warp; task < 2 * time_tiles * time_tiles; task += prepare_warps) {
        const int plane = task / (time_tiles * time_tiles);
        const int target = task / time_tiles % time_tiles * tile;
        const int source = task % time_tiles * tile;
        if (source > target) {
            continue;
        }
        Acc result;
        mma::fill_fragment(result, 0.0f);
        #pragma unroll
        for (int key = 0; key < dimension; key += tile) {
            ARow lhs;
            BCol rhs;
            mma::load_matrix_sync(lhs, qk + (plane * rows + target) * key_stride + key,
                                  key_stride);
            mma::load_matrix_sync(rhs, qk + (rows + source) * key_stride + key, key_stride);
            mma::mma_sync(result, lhs, rhs, result);
        }
        mma::store_matrix_sync(dots + (plane * rows + target) * rows + source,
                               result, rows, mma::mem_row_major);
    }
}

template <typename S, int SharedHeads>
__device__ __forceinline__ void prepare_segment(int count, const Bf16* convolved,
                                                const float2* gates, Matrices* matrices) {
    const int first = blockIdx.y * rows;
    if (first >= count) {
        return;
    }
    __shared__ alignas(32) Bf16 qk[2 * rows * key_stride];
    __shared__ alignas(32) float dots[2 * rows * rows];
    __shared__ float lower[SharedHeads][rows * rows];
    __shared__ float decay[SharedHeads][rows];
    __shared__ float beta[SharedHeads][rows];
    const int first_head = blockIdx.x * SharedHeads;
    const int valid = min(count - first, rows);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    auto* chunk = matrices + blockIdx.y * S::value_heads + first_head;

    stage_qk<S, threads * SharedHeads>(qk, convolved, first, valid,
                                       first_head / S::heads_per_qk);
    if (warp < SharedHeads) {
        const auto gate = lane < valid
            ? gates[(first + lane) * S::value_heads + first_head + warp]
            : make_float2(1.0f, 0.0f);
        decay[warp][lane] = gate.x;
        beta[warp][lane] = gate.y;
        chunk[warp].beta[lane] = gate.y;
    }
    __syncthreads();

    segment_qk_dots<SharedHeads>(qk, dots, warp);
    __syncthreads();

    constexpr bool grouped = SharedHeads > 1;
    constexpr int stride = rows + (grouped ? 2 : 0);
    constexpr int plane_size = rows * stride;
    static_assert(2 * SharedHeads * plane_size <= 2 * rows * key_stride);

    if (warp < SharedHeads) {
        auto& matrix = chunk[warp];
        auto* scores = grouped ? qk + warp * plane_size : matrix.scores;
        float product = 1.0f;
        for (int source = lane; source >= 0; --source) {
            const int index = lane * rows + source;
            lower[warp][source * rows + lane] = source < lane
                ? product * beta[warp][lane] * dots[rows * rows + index] : 0.0f;
            scores[lane * stride + source] = __float2bfloat16(product * dots[index]);
            if (lane == rows - 1) {
                matrix.tail[source] = product;
            }
            product *= decay[warp][source];
        }
        matrix.initial[lane] = product;
    }
    const int value_head = warp / warps;
    const int head_warp = warp % warps;
    auto& matrix = chunk[value_head];
    auto* scores = grouped ? qk + value_head * plane_size : matrix.scores;
    auto* solution = grouped ? qk + (SharedHeads + value_head) * plane_size : matrix.inverse;
    for (int index = head_warp * 32 + lane; index < rows * rows; index += threads) {
        if (index / rows < index % rows) {
            scores[index / rows * stride + index % rows] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();
    for (int column = head_warp; column < rows; column += warps) {
        float inverse = lane == column ? 1.0f : 0.0f;
        for (int pivot = column; pivot < rows - 1; ++pivot) {
            const float solved = __shfl_sync(0xffffffffu, inverse, pivot);
            if (lane > pivot) {
                inverse = fmaf(-lower[value_head][pivot * rows + lane], solved, inverse);
            }
        }
        solution[lane * stride + column] = __float2bfloat16(inverse);
    }
    if constexpr (grouped) {
        store_grouped_matrices<stride>(matrix, solution, scores, head_warp * 32 + lane);
    }
}

template <typename S = GdnFull, int SharedHeads = 1>
__global__ void prepare(int count, const Bf16* convolved, const float2* gates,
                        Matrices* matrices) {
    prepare_segment<S, SharedHeads>(count, convolved, gates, matrices);
}

template <typename S = GdnFull>
__global__ void prepare_segments(const __grid_constant__ Segments segments) {
    const auto& segment = segments.values[blockIdx.z];
    prepare_segment<S, S::heads_per_qk>(segment.count, segment.convolved, segment.gates,
                                        segment.matrices);
}
#else
template <typename S = GdnFull, int SharedHeads = 1>
__global__ void prepare(int, const Bf16*, const float2*, Matrices*) {
    __trap();
}
template <typename S = GdnFull>
__global__ void prepare_segments(Segments) {
    __trap();
}
#endif

namespace warp_mma {

constexpr int columns = 8;
constexpr int block_warps = 8;
constexpr int block_threads = block_warps * 32;
constexpr int warp_value_tiles = dimension / (columns * block_warps);
constexpr int matrix_stride = rows + vector_elements;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
using Pair = __nv_bfloat162;

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
            const auto lhs = load_a<false>(
                matrix + task * tile * matrix_stride + step * tile, matrix_stride, lane);
            accumulate(result[task], lhs, rhs);
        }
    }
}

struct Stage {
    alignas(32) Bf16 qk[2 * rows * key_stride];
    alignas(32) Bf16 inverse[rows * matrix_stride];
    alignas(32) Bf16 scores[rows * matrix_stride];
    float initial[rows];
    float tail[rows];
    float beta[rows];
};

constexpr int value_stage_warps = 4;

struct ValueStage : Stage {
    alignas(32) Bf16 values[rows * value_stage_warps * columns];
};

struct PipelinedStage : ValueStage {
    alignas(32) Bf16 scaled[rows * key_stride];
};

__device__ __forceinline__ void copy_async(Bf16* target, const Bf16* source, int bytes) {
    const auto address = static_cast<unsigned int>(__cvta_generic_to_shared(target));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(address), "l"(source), "r"(bytes) : "memory");
}

template <typename S, int BlockWarps, typename Frame>
__device__ __forceinline__ void prefetch(Frame& target, const Matrices& matrix,
                                         const Bf16* convolved, int first, int valid, int head,
                                         int first_feature, int thread) {
    constexpr int vectors_per_row = dimension / vector_elements;
    for (int vector = thread; vector < 2 * rows * vectors_per_row; vector += BlockWarps * 32) {
        const int row = vector / vectors_per_row;
        const int position = row % rows;
        const int column = vector % vectors_per_row * vector_elements;
        const int plane = row / rows;
        const int feature = plane * S::qk_width + head / S::heads_per_qk * dimension + column;
        const auto offset = static_cast<std::size_t>(first + min(position, valid - 1));
        const auto* source = convolved + offset * S::qkv_width + feature;
        copy_async(target.qk + row * key_stride + column, source,
                   position < valid ? sizeof(uint4) : 0);
    }
    if constexpr (std::is_base_of_v<ValueStage, Frame>) {
        static_assert(BlockWarps == value_stage_warps);
        constexpr int width = BlockWarps * columns;
        constexpr int vectors = width / vector_elements;
        static_assert(rows * vectors == BlockWarps * 32);
        const int position = thread / vectors;
        const int component = thread % vectors * vector_elements;
        const auto offset = static_cast<std::size_t>(first + min(position, valid - 1));
        const auto* source = convolved + offset * S::qkv_width + 2 * S::qk_width +
                             first_feature + component;
        copy_async(target.values + position * width + component, source,
                   position < valid ? sizeof(uint4) : 0);
    }
    for (int index = thread * vector_elements; index < rows * rows;
         index += BlockWarps * 32 * vector_elements) {
        const int offset = index / rows * matrix_stride + index % rows;
        copy_async(target.inverse + offset, matrix.inverse + index, sizeof(uint4));
        copy_async(target.scores + offset, matrix.scores + index, sizeof(uint4));
    }
    if (thread < rows) {
        target.initial[thread] = matrix.initial[thread];
        target.tail[thread] = matrix.tail[thread];
        target.beta[thread] = matrix.beta[thread];
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

constexpr int dynamic_bytes = 2 * sizeof(Stage);
constexpr int value_dynamic_bytes = 2 * sizeof(ValueStage);
constexpr int two_frame_bytes = 2 * sizeof(PipelinedStage);
constexpr int three_frame_bytes = 3 * sizeof(PipelinedStage);
constexpr int value_warp_static_bytes = sizeof(Bf16) *
    value_stage_warps * (dimension + rows) * columns;
constexpr int local_static_bytes = sizeof(Bf16) * rows * key_stride + value_warp_static_bytes;
constexpr int local_shared_bytes = dynamic_bytes + local_static_bytes;
constexpr int value_shared_bytes = value_dynamic_bytes + local_static_bytes;
constexpr int two_frame_shared_bytes = two_frame_bytes + value_warp_static_bytes;
constexpr int three_frame_shared_bytes = three_frame_bytes + value_warp_static_bytes;

template<bool Grouped, int BlockWarps>
__device__ __forceinline__ void scale_keys(const Stage& stage, Bf16* scaled_keys, int thread) {
    constexpr int elements = Grouped ? 2 : 1;
    for (int index = elements * thread; index < rows * dimension;
         index += elements * BlockWarps * 32) {
        const int position = index / dimension;
        const int offset = position * key_stride + index % dimension;
        const auto* key = stage.qk + rows * key_stride + offset;
        const float tail = stage.tail[position];
        if constexpr (Grouped) {
            const auto keys = __bfloat1622float2(*reinterpret_cast<const Pair*>(key));
            *reinterpret_cast<Pair*>(scaled_keys + offset) =
                __floats2bfloat162_rn(tail * keys.x, tail * keys.y);
        } else {
            scaled_keys[offset] = __float2bfloat16(tail * __bfloat162float(*key));
        }
    }
}

template <typename S, int BlockWarps, int PipelineSlots>
__device__ __forceinline__ void stage_tiles(int count, int head, int first_feature,
                                            const Bf16* convolved, const Matrices* matrices,
                                            PipelinedStage* frames) {
    constexpr int producer_threads = BlockWarps * 32;
    const int thread = threadIdx.x - producer_threads;
    const int chunks = (count + rows - 1) / rows;
    const auto fetch = [&](int chunk) {
        const int first = chunk * rows;
        prefetch<S, BlockWarps>(frames[chunk % PipelineSlots],
                                matrices[chunk * S::value_heads + head], convolved, first,
                                min(count - first, rows), head, first_feature, thread);
    };
    if constexpr (PipelineSlots == 3) {
        fetch(0);
        if (chunks > 1)
            fetch(1);
    }
    for (int chunk = 0; chunk < chunks; ++chunk) {
        if constexpr (PipelineSlots == 3) {
            if (chunk + 1 < chunks) {
                asm volatile("cp.async.wait_group 1;" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;" ::: "memory");
            }
            asm volatile("bar.sync 1, %0;" ::"n"(producer_threads) : "memory");
        } else {
            fetch(chunk);
            asm volatile("cp.async.wait_group 0; bar.sync 1, %0;" ::"n"(producer_threads)
                         : "memory");
        }
        auto& frame = frames[chunk % PipelineSlots];
        scale_keys<true, BlockWarps>(frame, frame.scaled, thread);
        asm volatile("bar.sync 2;" ::: "memory");
        if constexpr (PipelineSlots == 3) {
            if (chunk + 2 < chunks)
                fetch(chunk + 2);
        }
    }
}

template <typename S, bool Grouped, typename Frame>
__device__ __forceinline__ void write_residual(const Frame& stage, const Bf16* convolved,
                                              const Result (&projection)[2 * time_tiles],
                                              int first, int valid, int feature, int lane,
                                              Bf16* work) {
    constexpr int elements = Grouped ? 2 : 1;
    #pragma unroll
    for (int task = 0; task < time_tiles; ++task) {
        Result residual;
        #pragma unroll
        for (int element = 0; element < 4; element += elements) {
            const int position = task * tile + result_row(lane, element);
            if (position < valid) {
                const int token = first + position;
                const auto* target = convolved +
                    static_cast<std::size_t>(token) * S::qkv_width +
                    2 * S::qk_width + feature + result_column(lane, element);
                if constexpr (std::is_base_of_v<ValueStage, Frame>) {
                    constexpr int width = value_stage_warps * columns;
                    target = stage.values + position * width + (threadIdx.x / 32) * columns +
                             result_column(lane, element);
                }
                const float beta = stage.beta[position];
                const float initial = stage.initial[position];
                const auto& projected = projection[time_tiles + task];
                if constexpr (Grouped) {
                    const auto value = __bfloat1622float2(*reinterpret_cast<const Pair*>(target));
                    residual.fragments[element] =
                        beta * (value.x - initial * projected.fragments[element]);
                    residual.fragments[element + 1] =
                        beta * (value.y - initial * projected.fragments[element + 1]);
                } else {
                    residual.fragments[element] = beta *
                        (__bfloat162float(*target) - initial * projected.fragments[element]);
                }
            }
        }
        store_bf16(work + task * tile * columns, residual, lane);
    }
}

template <typename S, bool Grouped>
__device__ __forceinline__ void write_observable(const Stage& stage, const Bf16* work,
                                                const Result (&projection)[2 * time_tiles],
                                                int first, int valid, int feature, int lane,
                                                Bf16* observable) {
    Result output[time_tiles];
    solve(stage.scores, work, lane, output);
    #pragma unroll
    for (int task = 0; task < time_tiles; ++task) {
        #pragma unroll
        for (int element = 0; element < 4; element += Grouped ? 2 : 1) {
            const int position = task * tile + result_row(lane, element);
            if (position < valid) {
                constexpr float query_scale = 0.08838834764831845f;
                const float first_value = fmaf(stage.initial[position],
                    projection[task].fragments[element], output[task].fragments[element]);
                auto* target = observable +
                    static_cast<std::size_t>(first + position) * S::value_width +
                    feature + result_column(lane, element);
                if constexpr (Grouped) {
                    const float second_value = fmaf(stage.initial[position],
                        projection[task].fragments[element + 1],
                        output[task].fragments[element + 1]);
                    *reinterpret_cast<Pair*>(target) = __floats2bfloat162_rn(
                        query_scale * first_value, query_scale * second_value);
                } else {
                    *target = __float2bfloat16(query_scale * first_value);
                }
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

static __device__ __forceinline__ void store_carry(float* target, int lane,
                                                   const Result (&accumulator)[key_tiles]) {
    #pragma unroll
    for (int key = 0; key < key_tiles; ++key) {
    #pragma unroll
        for (int element = 0; element < 4; ++element) {
            target[(result_column(lane, element) * dimension) + (key * tile) +
                   result_row(lane, element)] = accumulator[key].fragments[element];
        }
    }
}

static __device__ __forceinline__ void
store_checkpoints(const gdn_prefill::Checkpoints& checkpoints, int& next, int last, int feature,
                  int lane, const Result (&accumulator)[key_tiles]) {
    while (next < checkpoints.count && checkpoints.values[next].rows <= last) {
        store_carry(checkpoints.values[next].state.recurrent +
                        (static_cast<std::size_t>(feature) * dimension),
                    lane, accumulator);
        ++next;
    }
}

template <typename S, bool Grouped, int BlockWarps, typename Frame>
static __device__ __forceinline__ void
wait_and_prefetch(Frame* frames, Bf16* scaled_keys, int first, int count, int head, int feature,
                  const Bf16* convolved, const Matrices* matrices) {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    const int chunk = first / rows;
    if (first + rows < count) {
        prefetch<S, BlockWarps>(
            frames[(chunk + 1) % 2], matrices[(chunk + 1) * S::value_heads + head], convolved,
            first + rows, min(count - first - rows, rows), head, feature, threadIdx.x);
    }
    scale_keys<Grouped, BlockWarps>(frames[chunk % 2], scaled_keys, threadIdx.x);
}

static __device__ __forceinline__ void
store_state_tiles(Bf16* state, int lane, const Result (&accumulator)[key_tiles], bool next_tile) {
    if (!next_tile) {
        return;
    }
    #pragma unroll
    for (int key = 0; key < key_tiles; ++key) {
        store_bf16(state + (static_cast<std::size_t>(key) * tile * columns), accumulator[key],
                   lane);
    }
}

template <typename S, bool Grouped, int BlockWarps = block_warps, typename Frame = Stage,
          int PipelineSlots = 0, bool Snapshot = false>
static __device__ __forceinline__ void
recur_segment(int count, const Bf16* convolved, const Matrices* matrices, float* recurrent,
              Bf16* observable, const gdn_prefill::Checkpoints* checkpoints = nullptr) {
    alignas(32) extern __shared__ unsigned char dynamic_frames[];
    auto* frames = reinterpret_cast<Frame*>(dynamic_frames);
    __shared__ alignas(32) Bf16 scaled_keys[rows * key_stride];
    __shared__ alignas(32) Bf16 states[BlockWarps][dimension * columns];
    __shared__ alignas(32) Bf16 workspace[BlockWarps][rows * columns];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int head = blockIdx.x;
    constexpr bool pipelined = PipelineSlots != 0;
    const int first_feature = head * dimension + blockIdx.y * BlockWarps * columns;
    if constexpr (pipelined) {
        if (warp >= BlockWarps) {
            stage_tiles<S, BlockWarps, PipelineSlots>(count, head, first_feature, convolved,
                                                      matrices, frames);
            return;
        }
    } else {
        prefetch<S, BlockWarps>(frames[0], matrices[head], convolved, 0, min(count, rows), head,
                                first_feature, threadIdx.x);
    }
    const int feature = first_feature + warp * columns;
    auto* carry = recurrent + feature * dimension;
    auto* state = states[warp];
    auto* work = workspace[warp];
    Result accumulator[key_tiles];
    int next_checkpoint = 0;
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
        const auto& stage = frames[chunk % (pipelined ? PipelineSlots : 2)];
        if constexpr (pipelined) {
            asm volatile("bar.sync 2;" ::: "memory");
        } else {
            wait_and_prefetch<S, Grouped, BlockWarps>(frames, scaled_keys, first, count, head,
                                                      first_feature, convolved, matrices);
        }

        Result projection[2 * time_tiles];
        project(stage.qk, state, lane, projection);
        write_residual<S, Grouped>(stage, convolved, projection, first, valid, feature, lane, work);
        __syncwarp();
        Result correction[time_tiles];
        solve(stage.inverse, work, lane, correction);
        __syncwarp();
        #pragma unroll
        for (int task = 0; task < time_tiles; ++task) {
            store_bf16(work + task * tile * columns, correction[task], lane);
        }
        __syncwarp();

        if (!Snapshot || observable) {
            write_observable<S, Grouped>(stage, work, projection, first, valid, feature, lane,
                                         observable);
        }
        if constexpr (Snapshot) {
            store_checkpoints(*checkpoints, next_checkpoint, first + valid - 1, feature, lane,
                              accumulator);
        }
        if constexpr (pipelined) {
            advance_state(stage.scaled, work, stage.initial[rows - 1], lane, accumulator);
        } else {
            __syncthreads();
            advance_state(scaled_keys, work, stage.initial[rows - 1], lane, accumulator);
        }
        if constexpr (Snapshot) {
            store_checkpoints(*checkpoints, next_checkpoint, first + valid, feature, lane,
                              accumulator);
        }
        store_state_tiles(state, lane, accumulator, first + rows < count);
    }
    store_carry(carry, lane, accumulator);
}

template <typename S = GdnFull, int BlockWarps = block_warps, bool Vector = false,
          bool StageValues = false, int PipelineSlots = 0>
__global__ void recur_warp(int count, const Bf16* convolved, const Matrices* matrices,
                           float* recurrent, Bf16* observable) {
    using Frame = std::conditional_t<PipelineSlots != 0, PipelinedStage,
                                     std::conditional_t<StageValues, ValueStage, Stage>>;
    recur_segment<S, Vector, BlockWarps, Frame, PipelineSlots>(count, convolved, matrices,
                                                               recurrent, observable);
}

template <typename S, int BlockWarps, bool Vector, bool StageValues, int PipelineSlots>
static __global__ void
recur_warp_checkpoints(int count, const Bf16* convolved, const Matrices* matrices, float* recurrent,
                       Bf16* observable, const __grid_constant__ gdn_prefill::Checkpoints points) {
    using Frame = std::conditional_t<PipelineSlots != 0, PipelinedStage,
                                     std::conditional_t<StageValues, ValueStage, Stage>>;
    recur_segment<S, Vector, BlockWarps, Frame, PipelineSlots, true>(
        count, convolved, matrices, recurrent, observable, &points);
}

template <typename S = GdnFull>
__global__ __launch_bounds__(block_threads, 3)
void recur_segments(const __grid_constant__ Segments segments) {
    const auto& segment = segments.values[blockIdx.z];
    recur_segment<S, true>(segment.count, segment.convolved, segment.matrices,
                          segment.recurrent, segment.observable);
}

template <typename S>
static __global__ __launch_bounds__(block_threads, 3) void recur_segment_checkpoints(
    const __grid_constant__ Segments segments, const __grid_constant__ SegmentCheckpoints points) {
    const auto& segment = segments.values[blockIdx.z];
    recur_segment<S, true, block_warps, Stage, 0, true>(
        segment.count, segment.convolved, segment.matrices, segment.recurrent, segment.observable,
        &points.values[blockIdx.z]);
}
#else
constexpr int dynamic_bytes = 0;
constexpr int value_dynamic_bytes = 0;
constexpr int local_shared_bytes = 0;
constexpr int value_shared_bytes = 0;
constexpr int two_frame_bytes = 0;
constexpr int three_frame_bytes = 0;
constexpr int two_frame_shared_bytes = 0;
constexpr int three_frame_shared_bytes = 0;

template <typename S = GdnFull, int BlockWarps = block_warps, bool Vector = false,
          bool StageValues = false, int PipelineSlots = 0>
__global__ void recur_warp(int, const Bf16*, const Matrices*, float*, Bf16*) {
    __trap();
}
template <typename S, int BlockWarps, bool Vector, bool StageValues, int PipelineSlots>
static __global__ void recur_warp_checkpoints(int, const Bf16*, const Matrices*, float*, Bf16*,
                                              gdn_prefill::Checkpoints) {
    __trap();
}
template <typename S = GdnFull>
__global__ void recur_segments(Segments) {
    __trap();
}
template <typename S>
static __global__ void recur_segment_checkpoints(Segments, SegmentCheckpoints) {
    __trap();
}
#endif

}

template <typename S = GdnFull>
inline std::size_t scratch_bytes(int count) {
    return static_cast<std::size_t>((count + rows - 1) / rows) * S::value_heads *
           sizeof(Matrices);
}

template <typename S>
bool eligible(int count, const void* scratch, std::size_t bytes) {
    return count >= min_count && scratch && bytes >= scratch_bytes<S>(count) &&
           reinterpret_cast<std::uintptr_t>(scratch) % alignof(Matrices) == 0;
}

template <typename S>
constexpr int recurrence_warps = std::is_same_v<S, GdnLocal> ? 4 : warp_mma::block_warps;

template <typename S, bool StageValues = false, int PipelineSlots = 0>
static const void* recurrence_kernel(bool checkpoints) {
    constexpr int warps = recurrence_warps<S>;
    constexpr bool local = std::is_same_v<S, GdnLocal>;
    return checkpoints
               ? reinterpret_cast<const void*>(
                     warp_mma::recur_warp_checkpoints<S, warps, local, StageValues, PipelineSlots>)
               : reinterpret_cast<const void*>(
                     warp_mma::recur_warp<S, warps, local, StageValues, PipelineSlots>);
}

template <typename S, bool StageValues = false, int PipelineSlots = 0>
static cudaError_t configure_kernel(bool checkpoints, int bytes) {
    return cudaFuncSetAttribute(recurrence_kernel<S, StageValues, PipelineSlots>(checkpoints),
                                cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

static cudaError_t configure_local(bool& enabled, int device, Recurrence& recurrence,
                                   bool checkpoints) {
    int shared_limit = 0;
    const auto status =
        cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status != cudaSuccess) {
        return status;
    }
    if (shared_limit < warp_mma::local_shared_bytes) {
        enabled = false;
        return cudaSuccess;
    }
    if (recurrence == Recurrence::unstaged || shared_limit < warp_mma::value_shared_bytes) {
        recurrence = Recurrence::unstaged;
        return configure_kernel<GdnLocal>(checkpoints, warp_mma::dynamic_bytes);
    }
    if (shared_limit >= warp_mma::three_frame_shared_bytes) {
        recurrence = Recurrence::three_frames;
        return configure_kernel<GdnLocal, true, 3>(checkpoints, warp_mma::three_frame_bytes);
    }
    if (shared_limit >= warp_mma::two_frame_shared_bytes) {
        recurrence = Recurrence::two_frames;
        return configure_kernel<GdnLocal, true, 2>(checkpoints, warp_mma::two_frame_bytes);
    }
    recurrence = Recurrence::staged;
    return configure_kernel<GdnLocal, true>(checkpoints, warp_mma::value_dynamic_bytes);
}

template <typename S>
cudaError_t configure(bool& enabled, bool grouped, Recurrence* recurrence, bool checkpoints) {
    int device = 0, major = 0, minor = 0;
    auto status = cudaGetDevice(&device);
    if (status == cudaSuccess) {
        status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    }
    if (status == cudaSuccess) {
        status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    }
    if (status != cudaSuccess) {
        return status;
    }
    enabled = major == 10 && minor == 0;
    if (!enabled) {
        return cudaSuccess;
    }
    if (grouped) {
        const void* kernel =
            checkpoints ? reinterpret_cast<const void*>(warp_mma::recur_segment_checkpoints<S>)
                        : reinterpret_cast<const void*>(warp_mma::recur_segments<S>);
        return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    warp_mma::dynamic_bytes);
    }
    if constexpr (std::is_same_v<S, GdnLocal>) {
        if (recurrence) {
            return configure_local(enabled, device, *recurrence, checkpoints);
        }
    }
    return configure_kernel<S>(checkpoints, warp_mma::dynamic_bytes);
}

template <typename S, bool StageValues, int PipelineSlots = 0>
static cudaError_t enqueue_chunk(Segment& segment, cudaStream_t stream,
                                 gdn_prefill::Checkpoints* points) {
    void* prepare_args[] = {
        &segment.count,
        static_cast<void*>(&segment.convolved),
        static_cast<void*>(&segment.gates),
        static_cast<void*>(&segment.matrices),
    };
    constexpr int shared_heads = StageValues ? S::heads_per_qk : 1;
    const auto status =
        cudaLaunchKernel(prepare<S, shared_heads>,
                         dim3(S::value_heads / shared_heads, (segment.count + rows - 1) / rows),
                         dim3(threads * shared_heads), prepare_args, 0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    constexpr int recurrence_threads = recurrence_warps<S> * (PipelineSlots != 0 ? 64 : 32);
    constexpr int frame_bytes[] = {0, 0, warp_mma::two_frame_bytes, warp_mma::three_frame_bytes};
    constexpr int ordinary_bytes =
        StageValues ? warp_mma::value_dynamic_bytes : warp_mma::dynamic_bytes;
    constexpr int shared = PipelineSlots != 0 ? frame_bytes[PipelineSlots] : ordinary_bytes;
    void* recur_args[] = {
        &segment.count,
        static_cast<void*>(&segment.convolved),
        static_cast<void*>(&segment.matrices),
        static_cast<void*>(&segment.recurrent),
        static_cast<void*>(&segment.observable),
        points,
    };
    return cudaLaunchKernel(
        recurrence_kernel<S, StageValues, PipelineSlots>(points != nullptr),
        dim3(S::value_heads, dimension / (warp_mma::columns * recurrence_warps<S>)),
        dim3(recurrence_threads), recur_args, shared, stream);
}

template <typename S>
static Segment interior_prefix(const Segment& segment, TextGdnPrefillCheckpoint point) {
    if (point.rows >= segment.count || point.rows % rows == 0) {
        return {};
    }
    const int first = point.rows / rows * rows;
    return {
        point.rows - first,
        segment.convolved + static_cast<std::size_t>(first) * S::qkv_width,
        segment.gates + first * S::value_heads,
        segment.matrices,
        point.state.recurrent,
        nullptr,
    };
}

template <typename S, bool StageValues, int PipelineSlots = 0>
static cudaError_t launch_chunk(Segment segment, cudaStream_t stream,
                                gdn_prefill::Checkpoints* points) {
    auto status = enqueue_chunk<S, StageValues, PipelineSlots>(segment, stream, points);
    if (status != cudaSuccess || !points) {
        return status;
    }
    gdn_prefill::Checkpoints empty{};
    for (int index = 0; index < points->count; ++index) {
        auto prefix = interior_prefix<S>(segment, points->values[index]);
        if (prefix.count != 0) {
            status = enqueue_chunk<S, StageValues, PipelineSlots>(prefix, stream, &empty);
            if (status != cudaSuccess) {
                return status;
            }
        }
    }
    return cudaSuccess;
}

template <typename S>
cudaError_t launch(Segment segment, cudaStream_t stream, Recurrence recurrence,
                   gdn_prefill::Checkpoints* checkpoints) {
    if constexpr (std::is_same_v<S, GdnLocal>) {
        if (recurrence == Recurrence::three_frames) {
            return launch_chunk<S, true, 3>(segment, stream, checkpoints);
        }
        if (recurrence == Recurrence::two_frames) {
            return launch_chunk<S, true, 2>(segment, stream, checkpoints);
        }
        if (recurrence == Recurrence::staged) {
            return launch_chunk<S, true>(segment, stream, checkpoints);
        }
    }
    return launch_chunk<S, false>(segment, stream, checkpoints);
}

template <typename S>
static cudaError_t enqueue_segments(Segments& segments, gdn_prefill::SegmentExtent extent,
                                    cudaStream_t stream, SegmentCheckpoints* points) {
    void* args[] = {&segments, points};
    const auto chunks = (extent.max_rows + rows - 1) / rows;
    const auto status =
        cudaLaunchKernel(prepare_segments<S>, dim3(S::qk_heads, chunks, extent.count),
                         dim3(threads * S::heads_per_qk), args, 0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    const void* kernel = points != nullptr
                             ? reinterpret_cast<const void*>(warp_mma::recur_segment_checkpoints<S>)
                             : reinterpret_cast<const void*>(warp_mma::recur_segments<S>);
    return cudaLaunchKernel(kernel, dim3(S::value_heads, warp_mma::warp_value_tiles, extent.count),
                            dim3(warp_mma::block_threads), args, warp_mma::dynamic_bytes, stream);
}

template <typename S>
cudaError_t launch_segments(Segments segments, int count, int max_rows, cudaStream_t stream,
                            SegmentCheckpoints* checkpoints) {
    auto status = enqueue_segments<S>(segments, {count, max_rows}, stream, checkpoints);
    if (status != cudaSuccess || !checkpoints) {
        return status;
    }
    SegmentCheckpoints empty{};
    for (int index = 0; index < text_gdn_max_prefill_checkpoints; ++index) {
        Segments prefixes{};
        int pending = 0;
        int longest = 0;
        for (int lane = 0; lane < count; ++lane) {
            const auto& points = checkpoints->values[lane];
            if (index >= points.count) {
                continue;
            }
            const auto prefix = interior_prefix<S>(segments.values[lane], points.values[index]);
            if (prefix.count != 0) {
                prefixes.values[pending++] = prefix;
                longest = std::max(longest, prefix.count);
            }
        }
        if (pending != 0) {
            status = enqueue_segments<S>(prefixes, {pending, longest}, stream, &empty);
            if (status != cudaSuccess) {
                return status;
            }
        }
    }
    return cudaSuccess;
}

template bool eligible<GdnFull>(int, const void*, std::size_t);
template bool eligible<GdnLocal>(int, const void*, std::size_t);
template cudaError_t configure<GdnFull>(bool&, bool, Recurrence*, bool);
template cudaError_t configure<GdnLocal>(bool&, bool, Recurrence*, bool);
template cudaError_t launch<GdnFull>(Segment, cudaStream_t, Recurrence, gdn_prefill::Checkpoints*);
template cudaError_t launch<GdnLocal>(Segment, cudaStream_t, Recurrence, gdn_prefill::Checkpoints*);
template cudaError_t launch_segments<GdnFull>(Segments, int, int, cudaStream_t,
                                              SegmentCheckpoints*);
template cudaError_t launch_segments<GdnLocal>(Segments, int, int, cudaStream_t,
                                               SegmentCheckpoints*);
}

std::size_t sglang::text_gdn_local_chunk_scratch_bytes(int rows) {
    return gdn_chunk::scratch_bytes<GdnLocal>(rows);
}
