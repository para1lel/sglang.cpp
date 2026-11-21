#include "sglang/prefill_fp8.h"

#include <cstddef>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>

namespace sglang {
namespace {

constexpr int group_size = text_fp8_group_size;
constexpr int warp_size = 32;
constexpr int accum_stages = 2;
constexpr int packed_wide_tile_cols = 256;

template <int Rows, int Cols, int Copies, int Batch, int Resident, int CopyBatch = 1,
          bool Compact = false>
struct Tile {
    static constexpr int rows = Rows;
    static constexpr int cols = Cols;
    static constexpr int copies = Copies;
    static constexpr int batch = Batch;
    static constexpr int resident = Resident;
    static constexpr bool wide = Cols == packed_wide_tile_cols;
    static constexpr int warp_cols = wide || Compact ? group_size : (Cols < 64 ? Cols : 64);
    static constexpr int partial_cols = wide || Compact ? 16 : warp_cols;
    static constexpr int consumers = Rows * Cols / (warp_size * warp_cols);
    static constexpr int threads = (consumers + 2) * warp_size;
    static constexpr int stage_bytes = (Rows + Cols) * group_size;
    static constexpr int copy_batch = CopyBatch;
    static constexpr int copy_slots = Copies / CopyBatch;
    static constexpr int accum_cols = Rows == 64 ? Cols / 2 : Cols;
    static constexpr int tmem_cols = accum_stages * Batch * accum_cols;
    static constexpr int store_width = 8;

    static_assert(Rows == 128 || (Rows == 64 && Cols == 128));
    static_assert(group_size % 32 == 0 && (group_size % Cols == 0 || wide));
    static_assert(Copies % CopyBatch == 0);
    static_assert(!Compact || (Rows == 128 && Cols == 128));
    static_assert(tmem_cols >= 32 && tmem_cols <= 512 && (tmem_cols & (tmem_cols - 1)) == 0);
};

template <class Config>
struct Pipeline {
    unsigned long long copy_ready[Config::copy_slots];
    unsigned long long copy_empty[Config::copy_slots];
    unsigned long long accum_ready[accum_stages];
    unsigned long long accum_empty[accum_stages];
    unsigned tmem;
};

__device__ unsigned shared_address(const void* pointer) {
    return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ void init_barrier(unsigned long long* barrier, int arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(shared_address(barrier)), "r"(arrivals) : "memory");
}

__device__ void wait_barrier(const unsigned long long* barrier, unsigned phase) {
    asm volatile("{ .reg .pred ready;\n"
                 "WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%0], %1;\n"
                 "@!ready bra.uni WAIT; }"
                 :: "r"(shared_address(barrier)), "r"(phase) : "memory");
}

__device__ void arrive_barrier(unsigned long long* barrier) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                 :: "r"(shared_address(barrier)) : "memory");
}

__device__ void commit_mma(unsigned long long* barrier) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                 :: "r"(shared_address(barrier)) : "memory");
}

__device__ void copy_tile(const CUtensorMap* map, unsigned shared, unsigned barrier,
                          int col, int row) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :: "r"(shared), "l"(map), "r"(col), "r"(row), "r"(barrier) : "memory");
}

template <class Config>
__device__ void produce_tiles(Pipeline<Config>& pipe, unsigned shared, const CUtensorMap* input,
                              const CUtensorMap* weight, int row_base, int col_base, int groups) {
    for (int group = 0; group < groups; group += Config::copy_batch) {
        const int slot = (group % Config::copies) / Config::copy_batch;
        if (group >= Config::copies) {
            wait_barrier(&pipe.copy_empty[slot], (group / Config::copies - 1) % 2);
        }
        const int remaining = groups - group;
        const int count = remaining < Config::copy_batch ? remaining : Config::copy_batch;
        const unsigned barrier = shared_address(&pipe.copy_ready[slot]);
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                     :: "r"(barrier), "r"(count * Config::stage_bytes) : "memory");
        for (int offset = 0; offset < count; ++offset) {
            const unsigned start =
                shared + (slot * Config::copy_batch + offset) * Config::stage_bytes;
            const int position = (group + offset) * group_size;
            copy_tile(input, start, barrier, position, row_base);
            copy_tile(weight, start + Config::rows * group_size, barrier, position, col_base);
        }
    }
}

__device__ std::uint64_t matrix_desc(unsigned start) {
    constexpr std::uint64_t stride = 8 * group_size / 16;
    constexpr std::uint64_t swizzle_128b = std::uint64_t{2} << 61;
    constexpr std::uint64_t descriptor_version = std::uint64_t{1} << 46;
    constexpr std::uint64_t leading_offset = std::uint64_t{1} << 16;
    return swizzle_128b | descriptor_version | (stride << 32) | leading_offset |
           ((start & 0x3ffffu) >> 4);
}

template <class Config>
__device__ void multiply_tiles(Pipeline<Config>& pipe, unsigned shared, int groups) {
    constexpr unsigned accumulator_f32 = 1 << 4;
    constexpr unsigned instruction = ((Config::rows / 16) << 24) | ((Config::cols / 8) << 17) |
                                     accumulator_f32;
    for (int group = 0; group < groups; ++group) {
        const int stage = group % Config::copies;
        const int copy_slot = stage / Config::copy_batch;
        const int batch = group / Config::batch;
        const int accum = group % (accum_stages * Config::batch);
        if (batch >= accum_stages && group % Config::batch == 0) {
            wait_barrier(&pipe.accum_empty[batch % accum_stages], (batch / accum_stages - 1) % 2);
        }
        if (group % Config::copy_batch == 0) {
            wait_barrier(&pipe.copy_ready[copy_slot], (group / Config::copies) % 2);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        }
        const unsigned start = shared + stage * Config::stage_bytes;
        const unsigned destination = pipe.tmem + accum * Config::accum_cols;
        #pragma unroll
        for (int chunk = 0; chunk < group_size; chunk += 32) {
            const auto input_desc = matrix_desc(start + chunk);
            const auto weight_desc = matrix_desc(start + Config::rows * group_size + chunk);
            if constexpr (Config::rows == 64) {
                asm volatile("{ .reg .pred accumulate;\n"
                             "setp.ne.u32 accumulate, %4, 0;\n"
                             "tcgen05.mma.ws.cta_group::1.kind::f8f6f4 "
                             "[%0], %1, %2, %3, accumulate; }"
                             :: "r"(destination), "l"(input_desc), "l"(weight_desc),
                                "r"(instruction), "r"(chunk) : "memory");
            } else {
                asm volatile("{ .reg .pred accumulate;\n"
                             "setp.ne.u32 accumulate, %4, 0;\n"
                             "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, accumulate; }"
                             :: "r"(destination), "l"(input_desc), "l"(weight_desc),
                                "r"(instruction), "r"(chunk) : "memory");
            }
        }
        if (group % Config::copy_batch == Config::copy_batch - 1 || group == groups - 1) {
            commit_mma(&pipe.copy_empty[copy_slot]);
        }
        if (group % Config::batch == Config::batch - 1 || group == groups - 1) {
            commit_mma(&pipe.accum_ready[batch % accum_stages]);
        }
    }
}

template <int Count>
__device__ void load_partial(unsigned source, unsigned (&out)[Count]) {
    if constexpr (Count == 16) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,"
                     "%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\ntcgen05.wait::ld.sync.aligned;"
                     : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]),
                       "=r"(out[4]), "=r"(out[5]), "=r"(out[6]), "=r"(out[7]),
                       "=r"(out[8]), "=r"(out[9]), "=r"(out[10]), "=r"(out[11]),
                       "=r"(out[12]), "=r"(out[13]), "=r"(out[14]), "=r"(out[15])
                     : "r"(source) : "memory");
    } else if constexpr (Count == 32) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,"
                     "%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,"
                     "%26,%27,%28,%29,%30,%31}, [%32];\ntcgen05.wait::ld.sync.aligned;"
                     : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]), "=r"(out[4]),
                       "=r"(out[5]), "=r"(out[6]), "=r"(out[7]), "=r"(out[8]), "=r"(out[9]),
                       "=r"(out[10]), "=r"(out[11]), "=r"(out[12]), "=r"(out[13]), "=r"(out[14]),
                       "=r"(out[15]), "=r"(out[16]), "=r"(out[17]), "=r"(out[18]), "=r"(out[19]),
                       "=r"(out[20]), "=r"(out[21]), "=r"(out[22]), "=r"(out[23]), "=r"(out[24]),
                       "=r"(out[25]), "=r"(out[26]), "=r"(out[27]), "=r"(out[28]), "=r"(out[29]),
                       "=r"(out[30]), "=r"(out[31])
                     : "r"(source) : "memory");
    } else {
        static_assert(Count == 64);
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,"
                     "%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,"
                     "%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,"
                     "%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63"
                     "}, [%64];\ntcgen05.wait::ld.sync.aligned;"
                     : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]), "=r"(out[4]),
                       "=r"(out[5]), "=r"(out[6]), "=r"(out[7]), "=r"(out[8]), "=r"(out[9]),
                       "=r"(out[10]), "=r"(out[11]), "=r"(out[12]), "=r"(out[13]), "=r"(out[14]),
                       "=r"(out[15]), "=r"(out[16]), "=r"(out[17]), "=r"(out[18]), "=r"(out[19]),
                       "=r"(out[20]), "=r"(out[21]), "=r"(out[22]), "=r"(out[23]), "=r"(out[24]),
                       "=r"(out[25]), "=r"(out[26]), "=r"(out[27]), "=r"(out[28]), "=r"(out[29]),
                       "=r"(out[30]), "=r"(out[31]), "=r"(out[32]), "=r"(out[33]), "=r"(out[34]),
                       "=r"(out[35]), "=r"(out[36]), "=r"(out[37]), "=r"(out[38]), "=r"(out[39]),
                       "=r"(out[40]), "=r"(out[41]), "=r"(out[42]), "=r"(out[43]), "=r"(out[44]),
                       "=r"(out[45]), "=r"(out[46]), "=r"(out[47]), "=r"(out[48]), "=r"(out[49]),
                       "=r"(out[50]), "=r"(out[51]), "=r"(out[52]), "=r"(out[53]), "=r"(out[54]),
                       "=r"(out[55]), "=r"(out[56]), "=r"(out[57]), "=r"(out[58]), "=r"(out[59]),
                       "=r"(out[60]), "=r"(out[61]), "=r"(out[62]), "=r"(out[63])
                     : "r"(source) : "memory");
    }
}

__device__ void scale_pair(float& result0, float& result1, float value0, float value1,
                           float scale) {
    asm("{ .reg .b64 values, scales, sums;\n"
        "mov.b64 values, {%2,%3};\n"
        "mov.b64 scales, {%4,%4};\n"
        "mov.b64 sums, {%0,%1};\n"
        "fma.rn.f32x2 sums, values, scales, sums;\n"
        "mov.b64 {%0,%1}, sums; }"
        : "+f"(result0), "+f"(result1)
        : "f"(value0), "f"(value1), "f"(scale));
}

template <class Config>
__device__ void accumulate(Pipeline<Config>& pipe, const float* activation,
                           const __nv_bfloat16* weight, int groups, unsigned tmem,
                           float (&result)[Config::warp_cols]) {
    for (int group = 0; group < groups; ++group) {
        const int slot = group % (accum_stages * Config::batch);
        const int batch = group / Config::batch;
        const float activation_scale = activation ? activation[group] : 0.0f;
        const float scale = __fmul_rn(activation_scale, __bfloat162float(weight[group]));
        if (group % Config::batch == 0) {
            wait_barrier(&pipe.accum_ready[batch % accum_stages], (batch / accum_stages) % 2);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        }
        #pragma unroll
        for (int base = 0; base < Config::warp_cols; base += Config::partial_cols) {
            unsigned partial[Config::partial_cols];
            load_partial(tmem + slot * Config::accum_cols + base, partial);
            #pragma unroll
            for (int col = 0; col < Config::partial_cols; col += 2) {
                scale_pair(result[base + col], result[base + col + 1],
                           __uint_as_float(partial[col]), __uint_as_float(partial[col + 1]), scale);
            }
        }
        if (group % Config::batch == Config::batch - 1 || group == groups - 1) {
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
            __syncwarp();
            if (threadIdx.x % warp_size == 0) {
                arrive_barrier(&pipe.accum_empty[batch % accum_stages]);
            }
        }
    }
}

struct OutputSlice {
    __nv_bfloat16* values;
    int width;
};

struct SplitOutput {
    OutputSlice slices[3];
};

__device__ OutputSlice locate_output(__nv_bfloat16* output, int width, int&) {
    return {output, width};
}

__device__ OutputSlice locate_output(SplitOutput output, int, int& col) {
    int index = 0;
    while (index < 2 && col >= output.slices[index].width) {
        col -= output.slices[index++].width;
    }
    return output.slices[index];
}

template <class Config, class Output>
__global__ __launch_bounds__(Config::threads, Config::resident)
void fp8_tma(const __grid_constant__ CUtensorMap activation_map,
              const __grid_constant__ CUtensorMap weight_map, const float* activation_scales,
              const __nv_bfloat16* weight_scales, int rows, int cols, int groups,
              int row_tile_group, Output output) {
    alignas(1024) extern __shared__ unsigned char tiles[];
    __shared__ Pipeline<Config> pipe;
    auto* epilogue = reinterpret_cast<__nv_bfloat16 (*)[Config::cols + Config::store_width]>(tiles);
    const int warp = threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    const int row_base = (blockIdx.y * row_tile_group + blockIdx.x % row_tile_group) * Config::rows;
    const int col_base = (blockIdx.x / row_tile_group) * Config::cols;
    if (row_base >= rows) {
        return;
    }
    const unsigned shared = shared_address(tiles);
    if (threadIdx.x == 0) {
        for (int slot = 0; slot < Config::copy_slots; ++slot) {
            init_barrier(&pipe.copy_ready[slot], 1);
            init_barrier(&pipe.copy_empty[slot], 1);
        }
        for (int slot = 0; slot < accum_stages; ++slot) {
            init_barrier(&pipe.accum_ready[slot], 1);
            init_barrier(&pipe.accum_empty[slot], Config::consumers);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == Config::consumers) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(shared_address(&pipe.tmem)), "r"(Config::tmem_cols) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();

    const int row_warps = Config::rows / warp_size;
    const int tmem_row = (warp % row_warps) * warp_size;
    const int local_row = tmem_row + lane;
    const int local_col = (warp / row_warps) * Config::warp_cols;
    const int tmem_col = Config::rows == 64 ? 0 : local_col;
    float result[Config::warp_cols] = {};
    if (warp < Config::consumers) {
        const int row = row_base + local_row;
        const auto* activation = row < rows ? activation_scales + std::size_t(row) * groups
                                            : nullptr;
        const int scale_col = Config::wide ? col_base + local_col : col_base;
        const auto* weight = weight_scales + std::size_t(scale_col / group_size) * groups;
        const unsigned tmem = pipe.tmem + (tmem_row << 16) + tmem_col;
        accumulate(pipe, activation, weight, groups, tmem, result);
    } else if (warp == Config::consumers && lane == 0) {
        multiply_tiles(pipe, shared, groups);
    } else if (warp == Config::consumers + 1 && lane == 0) {
        produce_tiles(pipe, shared, &activation_map, &weight_map, row_base, col_base, groups);
    }
    __syncthreads();

    if (warp < Config::consumers) {
        #pragma unroll
        for (int col = 0; col < Config::warp_cols; col += 2) {
            const auto packed = __floats2bfloat162_rn(result[col], result[col + 1]);
            *reinterpret_cast<__nv_bfloat162*>(&epilogue[local_row][local_col + col]) = packed;
        }
    }
    if (warp == Config::consumers) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(pipe.tmem), "r"(Config::tmem_cols) : "memory");
    }
    __syncthreads();
    int column = col_base;
    const auto [values, width] = locate_output(output, cols, column);
    if (width % Config::store_width == 0 && reinterpret_cast<std::uintptr_t>(values) % 16 == 0) {
        for (int index = threadIdx.x * Config::store_width; index < Config::rows * Config::cols;
             index += Config::threads * Config::store_width) {
            const int row = index / Config::cols;
            const int col = index % Config::cols;
            if (row_base + row < rows && column + col < width) {
                auto* destination = values + std::size_t(row_base + row) * width + column + col;
                *reinterpret_cast<uint4*>(destination) =
                    *reinterpret_cast<const uint4*>(&epilogue[row][col]);
            }
        }
    } else {
        for (int index = threadIdx.x; index < Config::rows * Config::cols;
             index += Config::threads) {
            const int row = index / Config::cols;
            const int col = index % Config::cols;
            if (row_base + row < rows && column + col < width) {
                values[std::size_t(row_base + row) * width + column + col] = epilogue[row][col];
            }
        }
    }
}

CUresult encode_map(PFN_cuTensorMapEncodeTiled_v12000 encode, CUtensorMap& map,
                    const std::uint8_t* data, int rows, int width, int tile) {
    const cuuint64_t dimensions[] = {static_cast<cuuint64_t>(width), static_cast<cuuint64_t>(rows)};
    const cuuint64_t stride[] = {static_cast<cuuint64_t>(width)};
    const cuuint32_t box[] = {group_size, static_cast<cuuint32_t>(tile)};
    const cuuint32_t element_stride[] = {1, 1};
    return encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<std::uint8_t*>(data),
                  dimensions, stride, box, element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                  CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

struct ByteSpan {
    std::uintptr_t begin;
    std::uintptr_t end;

    bool valid() const { return begin != 0 && begin < end; }
    bool overlaps(ByteSpan other) const { return begin < other.end && other.begin < end; }
};

template <typename Element>
ByteSpan byte_span(const Element* pointer, std::size_t count) {
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin % alignof(Element) != 0 ||
        count > (std::numeric_limits<std::uintptr_t>::max() - begin) / sizeof(Element)) {
        return {};
    }
    return {begin, begin + count * sizeof(Element)};
}

bool valid_buffers(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                   TextFp8ActivationView input, int rows, int out_width, int in_width,
                   __nv_bfloat16* output, int destination_width) {
    static_assert(std::numeric_limits<std::size_t>::digits >=
                  2 * std::numeric_limits<int>::digits);
    const auto activations = static_cast<std::size_t>(rows);
    const auto channels = static_cast<std::size_t>(out_width);
    const auto groups = static_cast<std::size_t>(text_fp8_group_count(in_width));
    const ByteSpan destination = byte_span(output, activations * destination_width);
    const ByteSpan sources[]{
        byte_span(weight, channels * in_width),
        byte_span(weight_scales, text_fp8_group_count(out_width) * groups),
        byte_span(input.values, activations * in_width),
        byte_span(input.scales, activations * groups),
    };
    if (!destination.valid()) {
        return false;
    }
    for (const auto source : sources) {
        if (!source.valid() || source.overlaps(destination)) {
            return false;
        }
    }
    return true;
}

cudaError_t current_stream_device(cudaStream_t stream, int& device) {
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }
    int stream_device = 0;
    status = cudaStreamGetDevice(stream, &stream_device);
    if (status != cudaSuccess) {
        return status;
    }
    return stream_device == device ? cudaSuccess : cudaErrorInvalidResourceHandle;
}

std::optional<cudaError_t> tensor_map_failure(CUresult status) {
    switch (status) {
        case CUDA_ERROR_NOT_SUPPORTED: return std::nullopt;
        case CUDA_ERROR_INVALID_VALUE: return cudaErrorInvalidValue;
        case CUDA_ERROR_INVALID_CONTEXT: return cudaErrorIncompatibleDriverContext;
        case CUDA_ERROR_NOT_INITIALIZED: return cudaErrorInitializationError;
        case CUDA_ERROR_OUT_OF_MEMORY: return cudaErrorMemoryAllocation;
        default: return cudaErrorUnknown;
    }
}

template <class Config, class Output>
std::optional<cudaError_t> launch_tma(
    const std::uint8_t* weight, const __nv_bfloat16* weight_scales, TextFp8ActivationView input,
    int rows, int out_width, int in_width, Output output, cudaStream_t stream, int device,
    int row_tile_group = 1) {
    constexpr int shared_bytes = Config::copies * Config::stage_bytes;
    const auto kernel = fp8_tma<Config, Output>;
    cudaFuncAttributes attributes{};
    auto status = cudaFuncGetAttributes(&attributes, kernel);
    if (status == cudaErrorInvalidDeviceFunction || status == cudaErrorNoKernelImageForDevice) {
        return std::nullopt;
    }
    if (status != cudaSuccess) {
        return status;
    }
    int shared_limit = 0;
    status = cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status != cudaSuccess) {
        return status;
    }
    if (static_cast<std::size_t>(shared_limit) < shared_bytes + attributes.sharedSizeBytes) {
        return std::nullopt;
    }
    void* entry = nullptr;
    status = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &entry, 12000,
                                              cudaEnableDefault, nullptr);
    if (status == cudaErrorNotSupported || (status == cudaSuccess && !entry)) {
        return std::nullopt;
    }
    if (status != cudaSuccess) {
        return status;
    }
    const auto encode = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(entry);
    CUtensorMap activation_map, weight_map;
    auto encoded = encode_map(encode, activation_map, input.values, rows, in_width, Config::rows);
    if (encoded == CUDA_SUCCESS) {
        encoded = encode_map(encode, weight_map, weight, out_width, in_width, Config::cols);
    }
    if (encoded != CUDA_SUCCESS) {
        return tensor_map_failure(encoded);
    }
    status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  shared_bytes);
    if (status != cudaSuccess) {
        return status;
    }
    int groups = in_width / group_size;
    void* args[] = {&activation_map, &weight_map, &input.scales, &weight_scales,
                    &rows, &out_width, &groups, &row_tile_group, &output};
    const dim3 grid((1 + (out_width - 1) / Config::cols) * row_tile_group,
                    1 + (rows - 1) / (Config::rows * row_tile_group));
    return cudaLaunchKernel(kernel, grid, Config::threads, args, shared_bytes, stream);
}

std::optional<cudaError_t> tma_device(cudaStream_t stream, int& device) {
    auto status = current_stream_device(stream, device);
    if (status != cudaSuccess) {
        return status;
    }
    int major = 0, minor = 0;
    status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (status == cudaSuccess) {
        status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    }
    if (status != cudaSuccess) {
        return status;
    }
    if (major != 10 || minor != 0) {
        return std::nullopt;
    }
    return cudaSuccess;
}

}

std::optional<cudaError_t> try_text_fp8_gemm_sm100(
    const std::uint8_t* weight, const __nv_bfloat16* weight_scales, TextFp8ActivationView input,
    int rows, int out_width, int in_width, __nv_bfloat16* output, cudaStream_t stream) {
    return try_text_fp8_gemm_sm100(weight, weight_scales, input, rows, out_width, in_width,
                                   output, stream, false);
}

std::optional<cudaError_t> try_text_fp8_gemm_sm100(
    const std::uint8_t* weight, const __nv_bfloat16* weight_scales, TextFp8ActivationView input,
    int rows, int out_width, int in_width, __nv_bfloat16* output, cudaStream_t stream,
    bool allow_large_rows) {
    if (rows <= 0 || out_width <= 0 || in_width <= 0 ||
        !valid_buffers(weight, weight_scales, input, rows, out_width, in_width,
                       output, out_width)) {
        return cudaErrorInvalidValue;
    }
    const int max_rows = allow_large_rows ? serial_prefill_max_rows : text_embedding_max_rows;
    if (rows < 3 || rows > max_rows || in_width % group_size != 0 ||
        reinterpret_cast<std::uintptr_t>(weight) % 16 != 0 ||
        reinterpret_cast<std::uintptr_t>(input.values) % 16 != 0) {
        return std::nullopt;
    }
    int device = 0;
    const auto status = tma_device(stream, device);
    if (status != cudaSuccess) {
        return status;
    }
    if (rows == 1024 && out_width == 5120 && in_width == 17408) {
        const auto paired = launch_tma<Tile<128, 128, 6, 2, 1, 2>>(
            weight, weight_scales, input, rows, out_width, in_width, output, stream, device, 8);
        if (paired) {
            return paired;
        }
    }
    const bool narrow_high_k = in_width >= 8192 && rows < 1024;
    if (rows > 128 && rows < 1024 && (out_width >= 8192 || in_width >= 6144)) {
        const auto paired = launch_tma<Tile<128, 128, 6, 2, 1, 2>>(
            weight, weight_scales, input, rows, out_width, in_width, output, stream, device,
            1 + (rows - 1) / 128);
        if (paired) {
            return paired;
        }
    }
    if ((rows <= 64 && out_width >= 4096) || narrow_high_k) {
        return launch_tma<Tile<64, 128, 6, 2, 1>>(weight, weight_scales, input, rows,
                                                 out_width, in_width, output, stream, device);
    }
    if (out_width >= 8192 || in_width >= 6144) {
        return launch_tma<Tile<128, 128, 6, 2, 1>>(weight, weight_scales, input, rows,
                                                  out_width, in_width, output, stream, device);
    }
    if (out_width < 1024) {
        return launch_tma<Tile<128, 32, 4, 1, 2>>(weight, weight_scales, input, rows,
                                                 out_width, in_width, output, stream, device);
    }
    return launch_tma<Tile<128, 64, 4, 1, 2>>(weight, weight_scales, input, rows,
                                             out_width, in_width, output, stream, device);
}

std::optional<cudaError_t> try_prefill_fp8_packed_sm100(
    TextFp8ActivationView input, int rows, int in_width,
    std::initializer_list<PrefillFp8Projection> projections, cudaStream_t stream) {
    if (rows <= 0 || in_width <= 0 || projections.size() < 2 || projections.size() > 3) {
        return cudaErrorInvalidValue;
    }
    const auto& first = *projections.begin();
    if (rows < 256 || rows > serial_prefill_max_rows || in_width % group_size != 0 ||
        reinterpret_cast<std::uintptr_t>(input.values) % 16 != 0 ||
        reinterpret_cast<std::uintptr_t>(first.weight) % 16 != 0) {
        return std::nullopt;
    }
    const auto scale_width = static_cast<std::size_t>(in_width / group_size);
    int total_width = 0;
    SplitOutput output{};
    int index = 0;
    bool wide_slices = true;
    for (const auto& projection : projections) {
        if (projection.width <= 0 ||
            projection.width > std::numeric_limits<int>::max() - total_width) {
            return cudaErrorInvalidValue;
        }
        const auto weights = byte_span(first.weight, std::size_t(total_width) * in_width);
        const auto scales = byte_span(first.scales,
                                      std::size_t(total_width / group_size) * scale_width);
        if (projection.width % group_size != 0 ||
            reinterpret_cast<std::uintptr_t>(projection.weight) != weights.end ||
            reinterpret_cast<std::uintptr_t>(projection.scales) != scales.end) {
            return std::nullopt;
        }
        wide_slices &= projection.width % packed_wide_tile_cols == 0;
        output.slices[index++] = {projection.output, projection.width};
        total_width += projection.width;
    }
    for (int slice = 0; slice < index; ++slice) {
        const auto [values, width] = output.slices[slice];
        if (!valid_buffers(first.weight, first.scales, input, rows, total_width, in_width,
                           values, width)) {
            return cudaErrorInvalidValue;
        }
        const auto span = byte_span(values, std::size_t(rows) * width);
        for (int previous = 0; previous < slice; ++previous) {
            const auto [other, other_width] = output.slices[previous];
            if (span.overlaps(byte_span(other, std::size_t(rows) * other_width))) {
                return cudaErrorInvalidValue;
            }
        }
    }
    int device = 0;
    const auto status = tma_device(stream, device);
    if (status != cudaSuccess) {
        return status;
    }
    const bool packed_mlp_chunk = projections.size() == 2 &&
                                  output.slices[0].width == 17408 &&
                                  output.slices[1].width == 17408;
    if (rows == 768 && in_width == 5120 && packed_mlp_chunk) {
        using GateUpTile = Tile<128, 128, 3, 1, 2, 1, true>;
        const auto compact = launch_tma<GateUpTile>(
            first.weight, first.scales, input, rows, total_width, in_width, output,
            stream, device, rows / GateUpTile::rows);
        if (compact) {
            return compact;
        }
    }
    if (rows == 1024 && in_width == 5120 && wide_slices &&
        (packed_mlp_chunk || total_width == 16384 || total_width == 14336)) {
        const auto wide = launch_tma<Tile<128, packed_wide_tile_cols, 4, 1, 1>>(
            first.weight, first.scales, input, rows, total_width, in_width, output, stream, device,
            packed_mlp_chunk ? 8 : 1);
        if (wide) {
            return wide;
        }
    }
    if (rows < 1024) {
        const int row_tiles = 1 + (rows - 1) / 128;
        const auto paired = wide_slices
            ? launch_tma<Tile<128, packed_wide_tile_cols, 4, 1, 1, 2>>(
                  first.weight, first.scales, input, rows, total_width, in_width, output,
                  stream, device, row_tiles)
            : launch_tma<Tile<128, 128, 6, 2, 1, 2>>(
                  first.weight, first.scales, input, rows, total_width, in_width, output,
                  stream, device, row_tiles);
        if (paired) {
            return paired;
        }
    }
    return launch_tma<Tile<128, 128, 6, 2, 1>>(first.weight, first.scales, input, rows,
                                              total_width, in_width, output, stream, device);
}

}
