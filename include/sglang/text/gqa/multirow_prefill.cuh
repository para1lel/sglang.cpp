#pragma once

cudaError_t decoder::tensor_core_multirow_prefill_gqa(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
    const Bf16* queries, Bf16* output, cudaStream_t stream, GqaGeometry geometry) {
    if (rows == 0) return cudaSuccess;
    if (!gqa_geometry_matches(geometry, cache)) return cudaErrorInvalidValue;
    return tensor_core_multirow_prefill_gqa(cache, pages, position, rows, queries, output,
                                           stream, PairedPrefillScratch{});
}

cudaError_t decoder::tensor_core_multirow_prefill_gqa(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
    const Bf16* queries, Bf16* output, cudaStream_t stream, PairedPrefillScratch scratch) {
    if (rows == 0) return cudaSuccess;
    if (position < 0 || rows < 0 || position > INT_MAX - rows) return cudaErrorInvalidValue;
    Attention request{cache, pages.device_pages, position, queries, output};
    const int visible = position + rows;
    const int flash_limit = rows == flash_chunk ? max_flash_visible_tokens :
                                                  max_multirow_visible_tokens;
    const bool ragged = default_ragged_prefill(cache, position, rows) ||
        default_ragged_prefill_tail({cache, pages, position, rows, queries, output},
                                   max_multirow_visible_tokens);
    if ((ragged || raw_pipeline_before_flash(cache, rows, visible)) && visible <= flash_limit) {
        if (const auto status = launch_tmem_prefill(cache, pages, position, queries, output,
                                                   stream, true, rows, true)) {
            return *status;
        }
    }
    if (rows >= min_flash_rows && rows <= flash_chunk && visible <= flash_limit &&
        (reinterpret_cast<std::uintptr_t>(cache.key_payload) % alignof(uint4) == 0) &&
        (reinterpret_cast<std::uintptr_t>(cache.value_payload) % alignof(uint4) == 0)) {
        if (rows == flash_chunk &&
            (position >= 8192 || position % flash_chunk == 0)) {
            constexpr int max_default_paired_position = 31744;
            const bool default_paired = position <= max_default_paired_position &&
                (position % flash_chunk == 0 || position % mxfp8_page_size != 0);
            if (default_paired && scratch.keys && scratch.values) {
                if (const auto status = launch_staged_tmem_prefill(cache, pages, position,
                                                                  queries, output, stream,
                                                                  scratch)) {
                    return *status;
                }
            }
            if (default_paired || (position % flash_chunk == 0 &&
                                   position >= max_multirow_visible_tokens)) {
                if (const auto status = launch_paired_tmem_prefill(cache, pages, position, queries,
                                                                  output, stream)) {
                    return *status;
                }
            }
        }
        if (const auto status = launch_flash_prefill(request, rows, stream)) {
            return *status;
        }
    }
    void* args[] = {&request, &rows};
    constexpr int min_paired_rows = 64;
    constexpr int min_staged_visible = 8192;
    constexpr int min_wide_tail_rows = 32;
    constexpr int min_paired_tail_rows = 12;
    if (rows >= 2 && rows < min_flash_rows &&
        visible >= min_staged_visible && visible <= max_multirow_visible_tokens &&
        reinterpret_cast<std::uintptr_t>(cache.key_payload) % alignof(uint4) == 0 &&
        reinterpret_cast<std::uintptr_t>(cache.value_payload) % alignof(uint4) == 0 &&
        reinterpret_cast<std::uintptr_t>(cache.key_scales) % alignof(std::uint64_t) == 0 &&
        reinterpret_cast<std::uintptr_t>(cache.value_scales) % alignof(std::uint64_t) == 0) {
        const int tile_rows = rows >= min_wide_tail_rows ? wide_rows :
            rows >= min_paired_tail_rows && rows % paired_rows == 0 ? paired_rows : 1;
        const dim3 blocks(rows / tile_rows + rows % tile_rows, gqa_head_pairs(cache));
        const auto kernel = tile_rows == wide_rows ?
            multirow_prefill_attention<wide_rows, 2, true> : tile_rows == paired_rows ?
            multirow_prefill_attention<paired_rows, 3, true> :
            multirow_prefill_attention<1, 4, true>;
        return cudaLaunchKernel(kernel, blocks, channels, args, 0, stream);
    }
    if (rows < min_paired_rows || visible > max_multirow_visible_tokens) {
        const dim3 blocks(rows, gqa_head_pairs(cache));
        return cudaLaunchKernel(protected_prefill_attention, blocks, channels, args, 0, stream);
    }
    constexpr int min_wide_rows = 128;
    const int tile_rows = rows >= min_wide_rows ? wide_rows : paired_rows;
    const dim3 blocks(rows / tile_rows + rows % tile_rows, gqa_head_pairs(cache));
    const auto kernel = tile_rows == wide_rows ? multirow_prefill_attention<wide_rows, 2> :
                                                multirow_prefill_attention<paired_rows, 3>;
    return cudaLaunchKernel(kernel, blocks, channels, args, 0, stream);
}