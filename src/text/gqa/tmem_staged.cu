#include "sglang/text/gqa/tmem_ops.cuh"
#include "sglang/runtime/backend/cuda_resources.h"
#include <algorithm>
#include <climits>
#include <cudaTypedefs.h>
#include <memory>

namespace sglang {

using namespace gqa_prefill;
using namespace gqa_prefill::tmem;

namespace gqa_prefill::tmem {

cudaError_t consume_prefill_submission(cudaError_t status) {
    const auto recorded = cudaGetLastError();
    if (status == cudaSuccess ||
        (!decoder::CudaError::terminal(status) && decoder::CudaError::terminal(recorded))) {
        return recorded;
    }
    return status;
}

std::optional<cudaError_t> encode_pair_maps(Mxfp8KvCache cache, PairMaps& maps) {
    void* entry = nullptr;
    cudaDriverEntryPointQueryResult query{};
    const auto status = cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &entry, 12000, cudaEnableDefault, &query);
    if (status == cudaErrorNotSupported || status == cudaErrorSymbolNotFound) {
        return std::nullopt;
    }
    if (status != cudaSuccess) {
        return status;
    }
    if (!entry || query != cudaDriverEntryPointSuccess) {
        return std::nullopt;
    }
    const auto encode = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(entry);
    const cuuint64_t width = static_cast<cuuint64_t>(cache.kv_heads) * channels;
    const cuuint64_t strides[] = {width, width * flash_keys};
    const cuuint64_t shape[] = {width, flash_keys,
                               static_cast<cuuint64_t>(cache.page_count) * pair_page_halves};
    constexpr cuuint32_t elements[] = {1, 1, 1};
    constexpr cuuint32_t key_box[] = {channels, pair_key_rows, pair_page_halves};
    constexpr cuuint32_t value_box[] = {pair_value_channels, flash_keys, pair_page_halves};
    constexpr cuuint64_t scale_stride[] = {pair_scale_bytes};
    const cuuint64_t scale_shape[] = {pair_scale_bytes / 4,
                                    static_cast<cuuint64_t>(cache.page_count) * cache.kv_heads};
    constexpr cuuint32_t scale_box[] = {pair_scale_bytes / 4, 1};
    const auto encode_map = [&](CUtensorMap& map, void* source, CUtensorMapDataType type,
                                unsigned rank, const cuuint64_t* dimensions,
                                const cuuint64_t* steps, const cuuint32_t* box,
                                CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
        return encode(&map, type, rank, source, dimensions, steps, box, elements,
                      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    };
    auto result = encode_map(maps.keys, cache.key_payload, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                             3, shape, strides, key_box);
    if (result == CUDA_SUCCESS) {
        result = encode_map(maps.values, cache.value_payload, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                            3, shape, strides, value_box, CU_TENSOR_MAP_SWIZZLE_128B);
    }
    if (result == CUDA_SUCCESS) {
        result = encode_map(maps.key_scales, cache.key_scales, CU_TENSOR_MAP_DATA_TYPE_UINT32,
                            2, scale_shape, scale_stride, scale_box);
    }
    if (result == CUDA_SUCCESS) {
        result = encode_map(maps.value_scales, cache.value_scales, CU_TENSOR_MAP_DATA_TYPE_UINT32,
                            2, scale_shape, scale_stride, scale_box);
    }
    if (result == CUDA_ERROR_INVALID_VALUE || result == CUDA_ERROR_NOT_SUPPORTED) {
        return std::nullopt;
    }
    return decoder::CudaError::from_driver(result);
}

std::optional<cudaError_t> encode_flash_maps(Mxfp8KvCache cache, CUtensorMap& key_map,
                                             CUtensorMap& value_map, bool automatic) {
    void* entry = nullptr;
    const auto status = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &entry, 12000,
                                                         cudaEnableDefault, nullptr);
    if (status != cudaSuccess) {
        return flash_setup_error(status, automatic && status == cudaErrorNotSupported);
    }
    if (!entry) {
        return std::nullopt;
    }
    const auto encode = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(entry);
    constexpr cuuint32_t tile[] = {channels, flash_keys};
    constexpr cuuint32_t elements[] = {1, 1};
    const cuuint64_t stride[] = {static_cast<cuuint64_t>(cache.kv_heads) * channels};
    const cuuint64_t shape[] = {
        stride[0], static_cast<cuuint64_t>(cache.page_count) * mxfp8_page_size};
    const auto encode_payload = [&](CUtensorMap& map, std::uint8_t* data) {
        return encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, data, shape, stride, tile,
                      elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    };
    auto result = encode_payload(key_map, cache.key_payload);
    if (result == CUDA_SUCCESS) {
        result = encode_payload(value_map, cache.value_payload);
    }
    if (!automatic) {
        return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
    }
    if (result == CUDA_ERROR_INVALID_VALUE || result == CUDA_ERROR_NOT_SUPPORTED) {
        return std::nullopt;
    }
    return decoder::CudaError::from_driver(result);
}

}

namespace {

struct StagedRange {
    std::uintptr_t begin;
    std::size_t bytes;
    std::size_t alignment;

    StagedRange(const void* pointer, std::size_t size, std::size_t align)
        : begin(reinterpret_cast<std::uintptr_t>(pointer)), bytes(size), alignment(align) {}

    bool valid() const {
        return begin && bytes && begin % alignment == 0 && bytes <= UINTPTR_MAX - begin;
    }

    bool overlaps(const StagedRange& other) const {
        return begin < other.begin + other.bytes && other.begin < begin + bytes;
    }
};

template <int Count>
bool valid_staged_ranges(const decoder::PagedPrefillSequence* sequences,
                         StagedRange keys, StagedRange values) {
    constexpr auto head_bytes = channels * sizeof(Bf16);
    const auto row_bytes = static_cast<std::size_t>(
        gqa_query_heads(sequences[0].cache.kv_heads)) * head_bytes;
    static_assert(head_bytes % alignof(uint4) == 0);
    static_assert(mxfp8_payload_bytes_per_page % alignof(uint4) == 0 &&
                  mxfp8_scale_bytes_per_page % alignof(std::uint64_t) == 0);
    static_assert(SIZE_MAX / mxfp8_payload_bytes_per_page >= INT_MAX);
    if (row_bytes == 0) {
        return false;
    }
    if (!keys.valid() || !values.valid() || keys.overlaps(values)) {
        return false;
    }
    for (int lane = 0; lane < Count; ++lane) {
        const auto& sequence = sequences[lane];
        const StagedRange output(sequence.output, sequence.rows * row_bytes,
                                 Count > 1 ? alignof(uint4) : alignof(Bf16));
        if (!output.valid() || output.overlaps(keys) || output.overlaps(values)) {
            return false;
        }
        for (int peer = 0; peer < Count; ++peer) {
            const auto& other = sequences[peer];
            const auto pages = static_cast<std::size_t>(other.cache.page_count);
            const auto logical_pages =
                (other.position + other.rows + mxfp8_page_size - 1) / mxfp8_page_size;
            const StagedRange inputs[] = {
                {other.queries, other.rows * row_bytes, alignof(Bf16)},
                {other.pages.device_pages, logical_pages * sizeof(std::int32_t),
                 alignof(std::int32_t)},
                {other.cache.key_payload,
                 pages * gqa_payload_bytes_per_page(other.cache.kv_heads), alignof(uint4)},
                {other.cache.value_payload,
                 pages * gqa_payload_bytes_per_page(other.cache.kv_heads), alignof(uint4)},
                {other.cache.key_scales,
                 pages * gqa_scale_bytes_per_page(other.cache.kv_heads), alignof(std::uint64_t)},
                {other.cache.value_scales,
                 pages * gqa_scale_bytes_per_page(other.cache.kv_heads),
                 alignof(std::uint64_t)},
            };
            for (const auto& input : inputs) {
                if (!input.valid() || input.overlaps(output) ||
                    input.overlaps(keys) || input.overlaps(values)) {
                    return false;
                }
            }
            if (peer < lane &&
                output.overlaps({other.output, other.rows * row_bytes, output.alignment})) {
                return false;
            }
        }
    }
    return true;
}

template <typename Function>
std::optional<cudaError_t> staged_driver_entry(const char* name, unsigned version,
                                              Function& function) {
    void* address = nullptr;
    cudaDriverEntryPointQueryResult query{};
    const auto status = cudaGetDriverEntryPointByVersion(
        name, &address, version, cudaEnableDefault, &query);
    if (status != cudaSuccess) {
        return flash_setup_error(
            status, status == cudaErrorNotSupported || status == cudaErrorSymbolNotFound);
    }
    if (!address || query != cudaDriverEntryPointSuccess) {
        return std::nullopt;
    }
    function = reinterpret_cast<Function>(address);
    return cudaSuccess;
}

std::optional<cudaError_t> staged_stream_context(cudaStream_t stream) {
    if (!stream || stream == cudaStreamLegacy || stream == cudaStreamPerThread) {
        return std::nullopt;
    }
    PFN_cuCtxGetCurrent_v4000 get_current = nullptr;
    if (const auto status = staged_driver_entry("cuCtxGetCurrent", 4000, get_current);
        !status || *status != cudaSuccess) {
        return status;
    }
    PFN_cuStreamGetCtx_v9020 get_stream = nullptr;
    if (const auto status = staged_driver_entry("cuStreamGetCtx", 9020, get_stream);
        !status || *status != cudaSuccess) {
        return status;
    }
    CUcontext current = nullptr;
    CUcontext context = nullptr;
    auto status = get_current(&current);
    if (status == CUDA_SUCCESS && current) {
        status = get_stream(stream, &context);
    }
    if (status == CUDA_SUCCESS) {
        return current && context == current ? std::optional{cudaSuccess} : std::nullopt;
    }
    if (status == CUDA_ERROR_NOT_SUPPORTED) {
        return std::nullopt;
    }
    return decoder::CudaError::from_driver(status);
}

std::optional<cudaError_t> staged_submission(cudaError_t status) {
    if (status == cudaSuccess) return status;
    const bool unavailable = paired_kernel_unavailable(status) ||
        status == cudaErrorInvalidConfiguration || status == cudaErrorInvalidClusterSize ||
        status == cudaErrorLaunchOutOfResources;
    if (unavailable) return flash_setup_error(status, true);
    return consume_prefill_submission(status);
}

std::optional<cudaError_t> encode_staged_maps(int full_pages, int kv_heads,
                                           std::uint16_t* keys, std::uint16_t* values,
                                           StagedPairMaps& maps,
                                           bool single = false, bool page = false,
                                           bool strict = false) {
    PFN_cuTensorMapEncodeTiled_v12000 encode = nullptr;
    if (const auto status = staged_driver_entry("cuTensorMapEncodeTiled", 12000, encode);
        !status || *status != cudaSuccess) {
        return status;
    }
    constexpr cuuint64_t width_bytes = tmem_swizzle * sizeof(std::uint16_t);
    constexpr cuuint64_t strides[] = {width_bytes, width_bytes * mxfp8_page_size};
    const cuuint64_t shape[] = {tmem_swizzle, mxfp8_page_size,
        static_cast<cuuint64_t>(full_pages) * kv_heads * staged_planes};
    constexpr cuuint32_t elements[] = {1, 1, 1};
    const cuuint32_t key_rows = page ? mxfp8_page_size : (single ? flash_keys : pair_key_rows);
    const cuuint32_t key_box[] = {tmem_swizzle, key_rows, staged_planes};
    const cuuint32_t value_box[] = {
        tmem_swizzle, flash_keys,
        static_cast<cuuint32_t>(single ? staged_planes : staged_planes / 2)};
    const auto encode_map = [&](CUtensorMap& map, void* source, const cuuint32_t* box) {
        return encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT16, 3, source, shape, strides,
                      box, elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    };
    auto result = encode_map(maps.keys, keys, key_box);
    if (result == CUDA_SUCCESS) {
        result = encode_map(maps.values, values, value_box);
    }
    if (result == CUDA_ERROR_NOT_SUPPORTED || (result == CUDA_ERROR_INVALID_VALUE && !strict)) {
        return std::nullopt;
    }
    return decoder::CudaError::from_driver(result);
}

template <bool Strict = false>
std::optional<cudaError_t> prepare_staged_single_kernel(const void* kernel,
                                                       const cudaLaunchConfig_t& config) {
    int device = 0;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) return status;
    if (properties.major != 10 || properties.minor != 0) return std::nullopt;
    const auto prepared = prepare_flash_kernel(
        kernel, config.dynamicSmemBytes, properties, !Strict);
    if (!prepared || *prepared != cudaSuccess) return prepared;
    int blocks = 0;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, config.blockDim.x,
                                                          config.dynamicSmemBytes);
    if (status != cudaSuccess) {
        if constexpr (Strict) return status;
        return flash_setup_error(status, paired_kernel_unavailable(status) ||
            status == cudaErrorInvalidConfiguration || status == cudaErrorInvalidValue);
    }
    return blocks > 0 ? std::optional{cudaSuccess} : std::nullopt;
}

constexpr int staged_causal_group_rows = 4 * flash_chunk;
constexpr int staged_causal_early_context = 32768;
constexpr int staged_rank_page_min_position = 8192 - staged_causal_group_rows;

template <int Count, bool CausalGroup, bool Partial>
std::optional<cudaError_t> prepare_long_page(const StagedRequests<Count>& requests,
                                            std::uint16_t* keys, std::uint16_t* values,
                                            int total_pages, StagedPairMaps& maps,
                                            const void*& selected, cudaLaunchConfig_t& config) {
    for (const auto& request : requests.items) {
        if (reinterpret_cast<std::uintptr_t>(request.output) % alignof(uint4) != 0) {
            return std::nullopt;
        }
    }
    const int kv_heads = requests.items[0].cache.kv_heads;
    auto kernel = reinterpret_cast<const void*>(tmem_staged_page_attention<Count>);
    if constexpr (CausalGroup || Partial) {
        if (kv_heads == gqa_rank_geometry.kv_heads) {
            kernel = reinterpret_cast<const void*>(tmem_staged_page_attention<Count, true>);
        }
    }
    auto local = config;
    local.blockDim.x = staged_compact_threads;
    local.dynamicSmemBytes = sizeof(StagedPageTile);
    const auto prepared = prepare_staged_single_kernel<true>(kernel, local);
    if (const auto status = consume_prefill_submission(prepared.value_or(cudaSuccess));
        status != cudaSuccess) return status;
    if (!prepared) return std::nullopt;
    const auto encoded = encode_staged_maps(total_pages, kv_heads, keys, values, maps,
                                            true, true, true);
    if (const auto status = consume_prefill_submission(encoded.value_or(cudaSuccess));
        status != cudaSuccess) {
        const bool early_causal_group = CausalGroup &&
            requests.items[0].position <= staged_causal_early_context - staged_causal_group_rows;
        if (early_causal_group && encoded == cudaErrorMemoryAllocation &&
            status == cudaErrorMemoryAllocation) return std::nullopt;
        return status;
    }
    if (encoded) {
        selected = kernel;
        config = local;
    }
    return encoded;
}

template <int Count, bool Partial, bool CausalGroup, bool LongPage>
std::optional<cudaError_t> submit_staged_requests(StagedRequests<Count>& requests,
                                                  std::uint16_t* keys, std::uint16_t* values,
                                                  int* invalid, int full_pages, int total_pages,
                                                  cudaStream_t stream) {
    const int position = requests.items[0].position;
    auto paired = tmem_staged_pair_attention<false, Count>;
    if constexpr (Count == 1) {
        if (position % mxfp8_page_size) {
            paired = tmem_staged_pair_attention<true, Count>;
        }
    }
    auto config = paired_config(sizeof(StagedPairTile),
                                gqa_head_pairs(requests.items[0].cache), stream);
    if constexpr (Count > 1) {
        config.gridDim.z = Count;
        config.blockDim.x = staged_compact_threads;
    }
    if constexpr (CausalGroup) config.gridDim.x = requests.rows[0] / flash_rows;
    if constexpr (Partial && LongPage) {
        config.gridDim.x = (requests.rows[0] + flash_rows - 1) / flash_rows;
    }
    StagedPairMaps maps{};
    const void* kernel = reinterpret_cast<const void*>(paired);
    bool single = false;
    if constexpr ((Count == 4 || CausalGroup) && !LongPage) {
        if (Partial || position % flash_chunk == 0) {
            const auto local = Partial
                ? reinterpret_cast<const void*>(tmem_staged_page_attention<Count>)
                : reinterpret_cast<const void*>(tmem_staged_single_attention<Count>);
            auto local_config = config;
            local_config.dynamicSmemBytes = sizeof(StagedSingleTile);
            if constexpr (CausalGroup) {
                local_config.blockDim.x = staged_compact_threads;
            }
            if constexpr (Partial) {
                int rows = 0;
                for (const int active : requests.rows) rows = std::max(rows, active);
                local_config.gridDim.x = (rows + flash_rows - 1) / flash_rows;
            }
            const auto prepared = prepare_staged_single_kernel(local, local_config);
            if (prepared && *prepared != cudaSuccess) return prepared;
            if (prepared) {
                const auto encoded = encode_staged_maps(total_pages,
                                                        requests.items[0].cache.kv_heads,
                                                        keys, values, maps, true, Partial);
                if (encoded && *encoded != cudaSuccess && *encoded != cudaErrorMemoryAllocation) {
                    return encoded;
                }
                if (encoded && *encoded == cudaSuccess) {
                    kernel = local;
                    config = local_config;
                    single = true;
                }
            }
        }
    }
    if constexpr (LongPage) {
        const auto prepared = prepare_long_page<Count, CausalGroup, Partial>(
            requests, keys, values, total_pages, maps, kernel, config);
        if (prepared && *prepared != cudaSuccess) return prepared;
        single = prepared.has_value();
    }
    if constexpr (Partial || CausalGroup) {
        if (!single) return std::nullopt;
    }
    if (!single) {
        if (const auto status = prepare_paired_kernel(kernel, config);
            !status || *status != cudaSuccess) {
            return status;
        }
        const auto encoded = encode_staged_maps(full_pages * Count,
                                                requests.items[0].cache.kv_heads,
                                                keys, values, maps);
        if (!encoded || *encoded == cudaErrorMemoryAllocation) return std::nullopt;
        if (*encoded != cudaSuccess) return encoded;
    }
    void* decode_args[] = {&requests, &keys, &values, &invalid};
    const auto decoded = cudaLaunchKernel(pair_predecode<Count, Partial>,
        dim3(full_pages, requests.items[0].cache.kv_heads, Count),
            channels, decode_args, 0, stream);
    if constexpr (LongPage) {
        if (single) {
            if (const auto status = consume_prefill_submission(decoded); status != cudaSuccess) {
                return status;
            }
            void* args[] = {&requests, &maps, &invalid};
            return consume_prefill_submission(cudaLaunchKernel(kernel, config.gridDim,
                config.blockDim, args, config.dynamicSmemBytes, stream));
        }
    }
    if (decoded != cudaSuccess) {
        return staged_submission(decoded);
    }
    void* args[] = {&requests, &maps, &invalid};
    return staged_submission(cudaLaunchKernel(kernel, config.gridDim, config.blockDim, args,
                                              config.dynamicSmemBytes, stream));
}

template <int Count, bool Partial = false, bool CausalGroup = false, bool LongPage = false>
std::optional<cudaError_t> launch_staged_requests(const decoder::PagedPrefillSequence* sequences,
                                                 cudaStream_t stream,
                                                 decoder::PairedPrefillScratch scratch) {
    const int position = sequences[0].position;
    const int kv_heads = sequences[0].cache.kv_heads;
    constexpr int min_partial_rows = LongPage ? decoder::staged_serial_min_rows : 256;
    constexpr int max_partial_visible = LongPage ? decoder::staged_serial_max_tokens : 32768;
    if (!StagedRange(scratch.keys, scratch.key_bytes, 1).valid() ||
        !StagedRange(scratch.values, scratch.value_bytes, 1).valid()) return std::nullopt;
    if constexpr (!Partial && !CausalGroup) {
        if (position < flash_chunk || position > 131072 / Count - flash_chunk) {
            return std::nullopt;
        }
    }
    StagedRequests<Count> requests{};
    int full_pages = 0;
    int total_pages = 0;
    for (int lane = 0; lane < Count; ++lane) {
        const auto& sequence = sequences[lane];
        if (sequence.cache.kv_heads != kv_heads)
            return std::nullopt;
        const int rows = sequence.rows;
        if constexpr (Partial) {
            if (rows < min_partial_rows || rows > flash_chunk || sequence.position < 0 ||
                sequence.position > max_partial_visible - rows) return std::nullopt;
        } else if constexpr (CausalGroup) {
            constexpr int context = LongPage
                ? decoder::staged_prefill_batch_long_max_position + flash_chunk
                : staged_causal_early_context;
            if (rows != staged_causal_group_rows || position < 0 || position > context - rows ||
                position % flash_chunk != 0) return std::nullopt;
        } else if (rows != flash_chunk || sequence.position != position) {
            return std::nullopt;
        }
        const int visible = sequence.position + rows;
        const int pages = (visible + (Partial ? mxfp8_page_size - 1 : 0)) / mxfp8_page_size;
        requests.rows[lane] = rows;
        requests.page_base[lane] = total_pages;
        total_pages += pages;
        full_pages = std::max(full_pages, pages);
        auto& request = requests.items[lane];
        request = {sequence.cache, sequence.pages.device_pages, position, sequence.queries,
                   sequence.output};
        request.position = sequence.position;
        if (const auto status = validate_paired_request(request, sequence.pages, rows);
            !status || *status != cudaSuccess) return status;
    }
    if (const auto pending = cudaPeekAtLastError(); pending != cudaSuccess) {
        return pending;
    }
    const auto page_heads = static_cast<std::size_t>(total_pages) * kv_heads;
    const auto elements = page_heads * staged_head_elements;
    const auto decoded_bytes = elements * sizeof(std::uint16_t);
    const auto value_bytes = decoded_bytes + page_heads * sizeof(int);
    static_assert(staged_alignment % alignof(int) == 0);
    void* key_storage = scratch.keys;
    void* value_storage = scratch.values;
    auto* keys = static_cast<std::uint16_t*>(
        std::align(staged_alignment, decoded_bytes, key_storage, scratch.key_bytes));
    auto* values = static_cast<std::uint16_t*>(
        std::align(staged_alignment, value_bytes, value_storage, scratch.value_bytes));
    if (!keys || !values) {
        return std::nullopt;
    }
    cudaStreamCaptureStatus capture{};
    const auto capture_status = cudaStreamIsCapturing(stream, &capture);
    if (capture_status != cudaSuccess) {
        return capture_status;
    }
    if (capture != cudaStreamCaptureStatusNone) {
        return std::nullopt;
    }
    if constexpr (Count > 1) {
        if (const auto status = staged_stream_context(stream); !status || *status != cudaSuccess) {
            return status;
        }
    }
    if (!valid_staged_ranges<Count>(sequences, {keys, decoded_bytes, staged_alignment},
                                    {values, value_bytes, staged_alignment})) {
        return std::nullopt;
    }
    return submit_staged_requests<Count, Partial, CausalGroup, LongPage>(requests, keys, values,
        reinterpret_cast<int*>(values + elements), full_pages, total_pages, stream);
}

}

std::optional<cudaError_t> launch_staged_tmem_prefill(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position,
    const Bf16* queries, Bf16* output, cudaStream_t stream,
    decoder::PairedPrefillScratch scratch) {
    if (position > 31744) {
        return std::nullopt;
    }
    const decoder::PagedPrefillSequence sequence{cache, pages, position, flash_chunk, queries,
                                                output};
    return launch_staged_requests<1>(&sequence, stream, scratch);
}

std::optional<cudaError_t> launch_causal_group_staged_tmem_prefill(
    const decoder::PagedPrefillSequence& sequence, cudaStream_t stream,
    decoder::PairedPrefillScratch scratch) {
    constexpr int last_single_position = staged_causal_early_context - staged_causal_group_rows;
    if (sequence.position > last_single_position ||
        (sequence.cache.kv_heads == gqa_rank_geometry.kv_heads &&
         sequence.position >= staged_rank_page_min_position)) {
        const auto status = launch_staged_requests<1, false, true, true>(
            &sequence, stream, scratch);
        if (status) return consume_prefill_submission(*status);
        if (sequence.position > last_single_position) return std::nullopt;
    }
    const auto status = launch_staged_requests<1, false, true>(&sequence, stream, scratch);
    if (status && *status != cudaSuccess) return consume_prefill_submission(*status);
    return status;
}

std::optional<cudaError_t> decoder::try_batched_staged_paged_prefill(
    const PagedPrefillSequence* sequences, int count, cudaStream_t stream,
    PairedPrefillScratch scratch) {
    if (!sequences || count != staged_prefill_batch_size) {
        return std::nullopt;
    }
    const int position = sequences[0].position;
    if (position < staged_prefill_batch_min_position ||
        position > staged_prefill_batch_max_position || position % flash_chunk != 0) {
        return std::nullopt;
    }
    return launch_staged_requests<staged_prefill_batch_size>(sequences, stream, scratch);
}

std::optional<cudaError_t> decoder::try_long_staged_paged_prefill(
    const PagedPrefillSequence* sequences, int count, cudaStream_t stream,
    PairedPrefillScratch scratch) {
    if (!sequences || !(count == 1 || count == 2 || count == staged_prefill_batch_size)) {
        return std::nullopt;
    }
    const int position = sequences[0].position;
    if (position <= staged_prefill_batch_max_position ||
        position > staged_prefill_batch_long_max_position || position % flash_chunk != 0) {
        return std::nullopt;
    }
    if (count == staged_prefill_batch_size) {
        return launch_staged_requests<staged_prefill_batch_size>(sequences, stream, scratch);
    }
    if (count == 2) {
        return launch_staged_requests<2, false, false, true>(sequences, stream, scratch);
    }
    return launch_staged_requests<1, false, false, true>(sequences, stream, scratch);
}

std::optional<cudaError_t> decoder::try_long_serial_staged_paged_prefill(
    const PagedPrefillSequence& sequence, cudaStream_t stream, PairedPrefillScratch scratch) {
    if (sequence.cache.kv_heads != gqa_rank_geometry.kv_heads ||
        !wants_long_serial_staging(sequence.position, sequence.rows)) return std::nullopt;
    return launch_staged_requests<1, true, false, true>(&sequence, stream, scratch);
}

std::optional<cudaError_t> launch_partial_staged_tmem_prefill(
    const decoder::PagedPrefillSequence* sequences, int count, cudaStream_t stream,
    decoder::PairedPrefillScratch scratch) {
    if (!sequences || count != decoder::staged_prefill_batch_size) return std::nullopt;
    return launch_staged_requests<decoder::staged_prefill_batch_size, true>(
        sequences, stream, scratch);
}

}
