#pragma once

namespace {

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
    switch (result) {
    case CUDA_SUCCESS: return cudaSuccess;
    case CUDA_ERROR_INVALID_VALUE:
    case CUDA_ERROR_NOT_SUPPORTED: return std::nullopt;
    case CUDA_ERROR_OUT_OF_MEMORY: return cudaErrorMemoryAllocation;
    case CUDA_ERROR_NOT_INITIALIZED: return cudaErrorInitializationError;
    case CUDA_ERROR_DEINITIALIZED: return cudaErrorCudartUnloading;
    default: return cudaErrorUnknown;
    }
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
    switch (result) {
    case CUDA_SUCCESS: return cudaSuccess;
    case CUDA_ERROR_INVALID_VALUE:
    case CUDA_ERROR_NOT_SUPPORTED: return std::nullopt;
    case CUDA_ERROR_INVALID_CONTEXT: return cudaErrorIncompatibleDriverContext;
    case CUDA_ERROR_CONTEXT_IS_DESTROYED: return cudaErrorContextIsDestroyed;
    case CUDA_ERROR_OUT_OF_MEMORY: return cudaErrorMemoryAllocation;
    case CUDA_ERROR_NOT_INITIALIZED: return cudaErrorInitializationError;
    case CUDA_ERROR_DEINITIALIZED: return cudaErrorCudartUnloading;
    case CUDA_ERROR_ILLEGAL_ADDRESS: return cudaErrorIllegalAddress;
    case CUDA_ERROR_LAUNCH_FAILED: return cudaErrorLaunchFailure;
    default: return cudaErrorUnknown;
    }
}

}