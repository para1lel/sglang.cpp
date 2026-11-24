#pragma once

constexpr int pipeline_members = 64;
constexpr int pipeline_rows = pipeline_members / ordered_heads;
constexpr int pipeline_threads = tmem_threads + 32;
constexpr int pipeline_query_rows = 256;

struct alignas(1024) TmemPipelineTile {
    std::uint16_t query[channels / tmem_swizzle][pipeline_members][tmem_swizzle];
    std::uint16_t key[channels / tmem_swizzle][flash_keys][tmem_swizzle];
    std::uint16_t value[2][channels / tmem_swizzle][flash_keys][tmem_swizzle];
    std::uint16_t probability[2][pipeline_members][flash_keys];
    FlashPayload raw[2];
    int bad_query[pipeline_members];
    int bad_key;
    Word tmem;
    alignas(8) unsigned long long copy_ready;
    alignas(8) unsigned long long key_ready;
    alignas(8) unsigned long long scores_ready[2];
    alignas(8) unsigned long long output_ready[2];
    alignas(8) unsigned long long operands_ready[2];
};

static_assert(offsetof(TmemPipelineTile, raw) % alignof(FlashPayload) == 0 &&
              offsetof(TmemPipelineTile, key) % 1024 == 0 &&
              offsetof(TmemPipelineTile, value) % 1024 == 0 &&
              offsetof(TmemPipelineTile, probability) % 1024 == 0);
static_assert(pipeline_query_rows % pipeline_rows == 0 &&
              sizeof(TmemPipelineTile) == sizeof(TmemFlashTile));

template <typename Tile>
__device__ void tmem_repair_ordered(const Tile& tile, Attention request, int row, int head,
                                    int rows) {
    __shared__ OrderedScratch fallback;
    constexpr int members = sizeof(tile.bad_query) / sizeof(int);
    const int position = request.position + row;
    if (!__syncthreads_or(threadIdx.x < members && tile.bad_query[threadIdx.x]) &&
        tile.bad_key >= position + rows) {
        return;
    }
    if (threadIdx.x >= channels) {
        return;
    }
    for (int offset = 0; offset < rows; ++offset) {
        if (position + offset >= tile.bad_key || tile.bad_query[offset * ordered_heads] ||
            tile.bad_query[offset * ordered_heads + 1]) {
            protected_attention(fallback, request, row + offset, head);
        }
    }
}

__device__ __forceinline__ void pipeline_init(unsigned long long& barrier, Word arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(tmem_smem(&barrier)), "r"(arrivals) : "memory");
}

template <int Barrier, int Count, int Leader>
__device__ __forceinline__ void pipeline_publish(unsigned long long& ready) {
    asm volatile("fence.proxy.async.shared::cta;\n"
                 "bar.sync %0, %1;" :: "n"(Barrier), "n"(Count) : "memory");
    if (threadIdx.x == Leader) {
        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                     :: "r"(tmem_smem(&ready)) : "memory");
    }
}

__device__ void pipeline_issue(TmemPipelineTile& tile, int chunks) {
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int stage = chunk & 1;
        const Word phase = (chunk >> 1) & 1;
        if (chunk + 1 < chunks) {
            tmem_copy_wait(tmem_smem(&tile.key_ready), stage ^ 1);
            tmem_query_key<false, false, pipeline_members>(
                tile, tmem_smem(&tile.scores_ready[1 - stage]),
                tmem_scores + (1 - stage) * flash_keys);
        }
        tmem_copy_wait(tmem_smem(&tile.operands_ready[stage]), phase);
        tmem_probability_value<false, pipeline_members>(
            tile.tmem, tile.probability[stage], tile.value[stage],
            tmem_smem(&tile.output_ready[stage]), chunk != 0);
    }
}

template <bool Key>
__device__ void pipeline_decode(TmemPipelineTile& tile, Attention request, int kv_head,
                                int chunk, int end, Word copy) {
    const int first = chunk * flash_keys;
    const int stage = chunk & 1;
    auto& decoded = Key ? tile.key : tile.value[stage];
    if (end - first < flash_keys) {
        tmem_stage_kv<Key, true>(decoded, tile.bad_key, tile.raw[stage], request, first,
                                kv_head, end - first);
        return;
    }
    if constexpr (Key) {
        tmem_copy_wait(copy, stage);
    }
    tmem_stage_kv<Key>(decoded, tile.bad_key, tile.raw[stage], request, first, kv_head, flash_keys);
}

__device__ void pipeline_produce(TmemPipelineTile& tile, Attention request, int kv_head, int end,
                                 const CUtensorMap* keys, const CUtensorMap* values) {
    const Word copy = tmem_smem(&tile.copy_ready);
    flash_tma_prefetch<flash_queries>(tile.raw[0], request, 0, kv_head, keys, values, copy);
    pipeline_decode<true>(tile, request, kv_head, 0, end, copy);
    pipeline_publish<1, tmem_producers, flash_queries>(tile.key_ready);
    if (2 * flash_keys <= end) {
        flash_tma_prefetch<flash_queries>(tile.raw[1], request, flash_keys, kv_head,
                                         keys, values, copy);
    }
    for (int first = 0, chunk = 0; first < end; first += flash_keys, ++chunk) {
        const int stage = chunk & 1;
        const Word phase = (chunk >> 1) & 1;
        if (chunk >= 2) {
            tmem_mma_wait(tmem_smem(&tile.output_ready[stage]), phase ^ 1);
        }
        pipeline_decode<false>(tile, request, kv_head, chunk, end, copy);
        pipeline_publish<1, tmem_producers, flash_queries>(tile.operands_ready[stage]);
        if (first + flash_keys < end) {
            tmem_mma_wait(tmem_smem(&tile.scores_ready[stage]), phase);
            pipeline_decode<true>(tile, request, kv_head, chunk + 1, end, copy);
            pipeline_publish<1, tmem_producers, flash_queries>(tile.key_ready);
            if (first + 3 * flash_keys <= end) {
                flash_tma_prefetch<flash_queries>(tile.raw[stage], request,
                    first + 2 * flash_keys, kv_head, keys, values, copy);
            }
        }
    }
}

__device__ void pipeline_softmax(TmemPipelineTile& tile, Attention request, int row, int head,
                                 int position, int end, int rows, int warp) {
    const Word address = tile.tmem + ((warp * 32) << 16);
    const int lane = threadIdx.x % 32;
    const int member = lane < 16 ? warp * 16 + lane : -1;
    FlashSoftmax state{-CUDART_INF_F, 0.0f};
    for (int first = 0, chunk = 0; first < end; first += flash_keys, ++chunk) {
        const int stage = chunk & 1;
        const Word phase = (chunk >> 1) & 1;
        tmem_mma_wait(tmem_smem(&tile.scores_ready[stage]), phase);
        if (chunk >= 2) {
            tmem_mma_wait(tmem_smem(&tile.output_ready[stage]), phase ^ 1);
        }
        const float history = tmem_softmax_chunk<true>(tile.probability[stage], address,
            position, first, state, tmem_scores + stage * flash_keys, member);
        if (chunk != 0) {
            tmem_mma_wait(tmem_smem(&tile.output_ready[1 - stage]), ((chunk - 1) >> 1) & 1);
            tmem_correct_output(address, history);
        }
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        pipeline_publish<2, flash_queries, 0>(tile.operands_ready[stage]);
    }
    const int last = (end - 1) / flash_keys;
    tmem_mma_wait(tmem_smem(&tile.output_ready[last & 1]), (last >> 1) & 1);
    tmem_store_output(address, state, request, row, head, rows, member);
}

__launch_bounds__(pipeline_threads, 1)
__global__ void tmem_pipeline_attention(Attention request,
                                        const __grid_constant__ CUtensorMap key_map,
                                        const __grid_constant__ CUtensorMap value_map, int rows) {
    extern __shared__ TmemPipelineTile shared_pipeline[];
    auto& tile = shared_pipeline[0];
    const int row = blockIdx.x * pipeline_rows;
    const int head = blockIdx.y * ordered_heads;
    const int position = request.position + row;
    const int active = min(pipeline_rows, rows - row);
    const int end = position + active;
    const int warp = threadIdx.x / 32;
    tmem_initialize(tile, warp, tmem_smem(&tile.copy_ready), tmem_smem(&tile.scores_ready[0]));
    if (threadIdx.x == 0) {
        pipeline_init(tile.key_ready, 1);
        pipeline_init(tile.scores_ready[1], 1);
        for (int stage = 0; stage < 2; ++stage) {
            pipeline_init(tile.output_ready[stage], 1);
            pipeline_init(tile.operands_ready[stage], 2);
        }
    }
    if (threadIdx.x < tmem_threads) {
        tmem_stage_query<pipeline_members>(tile, request, row, head, active);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x < flash_queries) {
        pipeline_softmax(tile, request, row, head, position, end, active, warp);
    } else if (threadIdx.x < tmem_threads) {
        pipeline_produce(tile, request, head / ratio, end, &key_map, &value_map);
    } else if (threadIdx.x == tmem_threads) {
        tmem_copy_wait(tmem_smem(&tile.key_ready), 0);
        tmem_query_key<false, false, pipeline_members>(tile, tmem_smem(&tile.scores_ready[0]));
        pipeline_issue(tile, (end + flash_keys - 1) / flash_keys);
    }
    __syncthreads();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(tile.tmem), "r"(tmem_cols) : "memory");
    }
    tmem_repair_ordered(tile, request, row, head, active);
}

std::optional<cudaError_t> prepare_flash_kernel(
    const void* kernel, std::size_t dynamic_shared, const cudaDeviceProp& properties,
    bool automatic) {
    cudaFuncAttributes attributes{};
    auto status = cudaFuncGetAttributes(&attributes, kernel);
    if (status != cudaSuccess) {
        return flash_setup_error(status, automatic && paired_kernel_unavailable(status));
    }
    if (properties.sharedMemPerBlockOptin < dynamic_shared + attributes.sharedSizeBytes) {
        return std::nullopt;
    }
    status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  dynamic_shared);
    return status == cudaSuccess ? cudaSuccess : flash_setup_error(status, automatic &&
        (status == cudaErrorInvalidValue || paired_kernel_unavailable(status)));
}

cudaError_t launch_pipeline(Attention& request, CUtensorMap& keys, CUtensorMap& values,
                            int rows, cudaStream_t stream) {
    void* args[] = {&request, &keys, &values, &rows};
    const dim3 blocks((rows + pipeline_rows - 1) / pipeline_rows, head_pairs);
    return cudaLaunchKernel(tmem_pipeline_attention, blocks, pipeline_threads, args,
                            sizeof(TmemPipelineTile), stream);
}
