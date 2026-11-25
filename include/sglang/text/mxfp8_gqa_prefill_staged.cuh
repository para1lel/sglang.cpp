#pragma once

template <bool Packed = false>
__device__ __forceinline__ void tmem_store_output(
    Word row_address, FlashSoftmax state, Attention request, int row, int head,
    int rows = flash_rows, int member = threadIdx.x) {
    const auto output = row_offset(gqa_query_heads(request.cache.kv_heads),
                                   row + max(0, member) / ordered_heads,
                                   head + max(0, member) % ordered_heads);
    const float inverse = __frcp_rn(state.mass);
    #pragma unroll 1
    for (int column = 0; column < channels; column += 32) {
        float accumulator[32];
        tmem_load(row_address + column, accumulator);
        if (member >= 0 && member / ordered_heads < rows) {
            if constexpr (Packed) {
                #pragma unroll
                for (int index = 0; index < 32; index += 8) {
                    const uint4 packed = make_uint4(
                        tmem_pack(accumulator[index] * inverse,
                                  accumulator[index + 1] * inverse),
                        tmem_pack(accumulator[index + 2] * inverse,
                                  accumulator[index + 3] * inverse),
                        tmem_pack(accumulator[index + 4] * inverse,
                                  accumulator[index + 5] * inverse),
                        tmem_pack(accumulator[index + 6] * inverse,
                                  accumulator[index + 7] * inverse));
                    *reinterpret_cast<uint4*>(request.output + output + column + index) = packed;
                }
            } else {
                #pragma unroll
                for (int index = 0; index < 32; ++index) {
                    request.output[output + column + index] =
                        __float2bfloat16_rn(accumulator[index] * inverse);
                }
            }
        }
    }
}

__device__ __forceinline__ void pair_arrive(Word ready) {
    asm volatile("{ .reg .u32 leader;\n"
                 "mapa.shared::cluster.u32 leader, %0, 0;\n"
                 "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [leader]; }"
                 :: "r"(ready) : "memory");
}

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
    switch (status) {
    case CUDA_SUCCESS:
        return current && context == current ? std::optional{cudaSuccess} : std::nullopt;
    case CUDA_ERROR_NOT_SUPPORTED: return std::nullopt;
    case CUDA_ERROR_INVALID_HANDLE: return cudaErrorInvalidResourceHandle;
    case CUDA_ERROR_NOT_INITIALIZED: return cudaErrorInitializationError;
    case CUDA_ERROR_DEINITIALIZED: return cudaErrorCudartUnloading;
    default: return cudaErrorUnknown;
    }
}

cudaError_t consume_prefill_submission(cudaError_t status) {
    const auto recorded = cudaGetLastError();
    if (status == cudaSuccess ||
        (!decoder::CudaError::terminal(status) && decoder::CudaError::terminal(recorded))) {
        return recorded;
    }
    return status;
}

std::optional<cudaError_t> staged_submission(cudaError_t status) {
    if (status == cudaSuccess) return status;
    const bool unavailable = paired_kernel_unavailable(status) ||
        status == cudaErrorInvalidConfiguration || status == cudaErrorInvalidClusterSize ||
        status == cudaErrorLaunchOutOfResources;
    if (unavailable) return flash_setup_error(status, true);
    return consume_prefill_submission(status);
}
