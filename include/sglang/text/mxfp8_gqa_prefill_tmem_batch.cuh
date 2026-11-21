#pragma once

namespace sglang {
namespace {

struct TmemAttentionBatch {
    PagedAttentionBatch tiles;
    CUtensorMap key_maps[paged_batch_sequences];
    CUtensorMap value_maps[paged_batch_sequences];
};

__launch_bounds__(tmem_threads, 1)
__global__ void batched_ragged_tmem_attention(const __grid_constant__ TmemAttentionBatch batch) {
    alignas(1024) extern __shared__ unsigned char shared_tmem[];
    const int tile = blockIdx.x;
    const int index = batch.tiles.locate(tile);
    tmem_flash_tile<true, true>(
        *reinterpret_cast<TmemFlashTile*>(shared_tmem), batch.tiles.requests[index],
        batch.key_maps[index], batch.value_maps[index], nullptr, nullptr,
        batch.tiles.rows[index], batch.tiles.row(tile, index, flash_rows));
}

}

std::optional<cudaError_t> decoder::launch_batched_ragged_tmem(
    const PagedPrefillSequence* sequences, int count, cudaStream_t stream) {
    int device = 0;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        return status;
    }
    if (properties.major != 10 || properties.minor != 0 || properties.multiProcessorCount != 148) {
        return std::nullopt;
    }
    if (const auto pending = cudaPeekAtLastError(); pending != cudaSuccess) {
        return pending;
    }
    constexpr auto dynamic_shared = sizeof(TmemFlashTile);
    const auto kernel = batched_ragged_tmem_attention;
    cudaFuncAttributes attributes{};
    status = cudaFuncGetAttributes(&attributes, kernel);
    if (status != cudaSuccess) {
        return flash_setup_error(status, paired_kernel_unavailable(status));
    }
    if (properties.sharedMemPerBlockOptin < dynamic_shared + attributes.sharedSizeBytes) {
        return std::nullopt;
    }
    status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  dynamic_shared);
    if (status != cudaSuccess) {
        return flash_setup_error(status,
            status == cudaErrorInvalidValue || paired_kernel_unavailable(status));
    }
    TmemAttentionBatch batch{};
    batch.tiles.count = count;
    int tiles = 0;
    for (int index = 0; index < count; ++index) {
        const auto& sequence = sequences[index];
        constexpr int max_physical_pages = INT_MAX / mxfp8_page_size + 1;
        if (sequence.cache.page_count > max_physical_pages) {
            return std::nullopt;
        }
        const auto encoded = encode_flash_maps(sequence.cache, batch.key_maps[index],
                                               batch.value_maps[index], true);
        if (!encoded || *encoded != cudaSuccess) {
            return encoded;
        }
        batch.tiles.requests[index] = {sequence.cache, sequence.pages.device_pages,
                                       sequence.position, sequence.queries, sequence.output};
        batch.tiles.rows[index] = sequence.rows;
        tiles += (sequence.rows + flash_rows - 1) / flash_rows;
        batch.tiles.tile_end[index] = tiles;
    }
    void* args[] = {&batch};
    return cudaLaunchKernel(kernel, dim3(tiles, head_pairs), tmem_threads, args,
                            dynamic_shared, stream);
}

}
