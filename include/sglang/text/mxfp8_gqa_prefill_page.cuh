constexpr int staged_page_keys = mxfp8_page_size;
static_assert(staged_page_keys == 2 * flash_keys &&
              tmem_scores + 2 * staged_page_keys == tmem_cols);

struct alignas(1024) StagedPageTile {
    std::uint16_t query[staged_planes][flash_queries][tmem_swizzle];
    std::uint16_t key[staged_planes][staged_page_keys][tmem_swizzle];
    std::uint16_t value[2][staged_planes][flash_keys][tmem_swizzle];
    int bad_query[flash_queries];
    int bad_key;
    Word tmem;
    unsigned long long key_ready[2];
    unsigned long long value_ready[2];
    unsigned long long scores_ready[2];
    unsigned long long output_ready[2];
    unsigned long long probability_ready[2];
};
static_assert(offsetof(StagedPageTile, key) % 1024 == 0 &&
              offsetof(StagedPageTile, value) % 1024 == 0);
static_assert(sizeof(StagedPageTile) == sizeof(StagedSingleTile));

__device__ void stage_page_keys(StagedPageTile& tile, const StagedPairMaps& maps,
                               const int* invalid, int page_base, int kv_head, int pages,
                               int kv_heads) {
    for (int page = 0; page < pages; ++page) {
        if (page) {
            const int previous = page - 1;
            tmem_copy_wait(tmem_smem(&tile.scores_ready[previous & 1]), (previous >> 1) & 1);
        }
        const int page_head = (page_base + page) * kv_heads + kv_head;
        const int bad = invalid[page_head];
        if (bad != INT_MAX) atomicMin(&tile.bad_key, bad);
        const Word barrier = tmem_smem(&tile.key_ready[page & 1]);
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                     :: "r"(barrier), "n"(sizeof(tile.key)) : "memory");
        pair_copy_decoded(&maps.keys, tile.key, barrier, 0, page_head * staged_planes);
    }
}

template <int N, cute::UMMA::Major B>
__device__ __forceinline__ std::uint64_t page_instruction() {
    return cute::UMMA::make_runtime_instr_desc<cute::bfloat16_t, cute::bfloat16_t, float,
        flash_queries, N, cute::UMMA::Major::K, B>();
}

__device__ void issue_page_scores(StagedPageTile& tile, int page) {
    using Atom = cute::SM100_MMA_F16BF16_SS<cute::bfloat16_t, cute::bfloat16_t, float,
        flash_queries, staged_page_keys, cute::UMMA::Major::K, cute::UMMA::Major::K>;
    const int slot = page & 1;
    const Word phase = (page >> 1) & 1;
    tmem_copy_wait(tmem_smem(&tile.key_ready[slot]), phase);
    const Word score = tile.tmem + tmem_scores + slot * staged_page_keys;
    #pragma unroll
    for (int dimension = 0; dimension < channels; dimension += 16) {
        const int plane = dimension / tmem_swizzle;
        const int offset = dimension % tmem_swizzle;
        const auto query = tmem_descriptor(&tile.query[plane][0][offset]);
        const auto key = tmem_descriptor(&tile.key[plane][0][offset]);
        Atom::fma(query, key, score, dimension != 0,
                  page_instruction<staged_page_keys, cute::UMMA::Major::K>());
    }
    if (cute::elect_one_sync()) tmem_commit(tmem_smem(&tile.scores_ready[slot]));
}

__device__ Word staged_page_score(int chunk) {
    return tmem_scores + ((chunk >> 1) & 1) * staged_page_keys + (chunk & 1) * flash_keys;
}

__device__ void issue_page_values(StagedPageTile& tile, int chunk) {
    using Atom = cute::SM100_MMA_F16BF16_TS<cute::bfloat16_t, cute::bfloat16_t, float,
        flash_queries, channels, cute::UMMA::Major::K, cute::UMMA::Major::MN>;
    const Word probability = staged_page_score(chunk);
    const int stage = chunk & 1;
    #pragma unroll
    for (int token = 0; token < flash_keys; token += 16) {
        const auto value = tmem_descriptor<true>(&tile.value[stage][0][token][0]);
        const Word p = tile.tmem + probability + token / 2;
        Atom::fma(p, value, tile.tmem, chunk != 0 || token != 0,
                  page_instruction<channels, cute::UMMA::Major::MN>());
    }
    if (cute::elect_one_sync()) tmem_commit(tmem_smem(&tile.output_ready[stage]));
}

template <bool ScoreLookahead>
__device__ void issue_page_mmas(StagedPageTile& tile, int chunks) {
    issue_page_scores(tile, 0);
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int stage = chunk & 1;
        const Word phase = (chunk >> 1) & 1;
        if constexpr (ScoreLookahead) {
            if (stage == 0 && chunk + 2 < chunks) issue_page_scores(tile, (chunk >> 1) + 1);
        }
        tmem_copy_wait(tmem_smem(&tile.value_ready[stage]), phase);
        tmem_mma_wait(tmem_smem(&tile.probability_ready[stage]), phase);
        issue_page_values(tile, chunk);
        if constexpr (!ScoreLookahead) {
            if (stage == 0 && chunk + 2 < chunks) issue_page_scores(tile, (chunk >> 1) + 1);
        }
    }
}

__device__ void staged_page_softmax(StagedPageTile& tile, const Attention& request, int row,
                                    int head, int position, int active_rows, int chunks) {
    const int warp = threadIdx.x / 32;
    const Word row_address = tile.tmem + ((warp * 32) << 16);
    FlashSoftmax state{-CUDART_INF_F, 0.0f};
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int stage = chunk & 1;
        const Word phase = (chunk >> 1) & 1;
        if (stage == 0) {
            tmem_mma_wait(tmem_smem(&tile.scores_ready[phase]), (chunk >> 2) & 1);
        }
        const int first = chunk * flash_keys;
        const int visible = position + threadIdx.x / ordered_heads - first + 1;
        const Word score = staged_page_score(chunk);
        const float history = first + flash_keys <= position + 1 ?
            tmem_probabilities<true, true, true>(tile.tmem, row_address, threadIdx.x,
                visible, state, score, score) :
            tmem_probabilities<false, true, true>(tile.tmem, row_address, threadIdx.x,
                visible, state, score, score);
        if (chunk) {
            tmem_mma_wait(tmem_smem(&tile.output_ready[1 - stage]), ((chunk - 1) >> 1) & 1);
            tmem_correct_output(row_address, history);
        }
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        __syncwarp();
        if ((threadIdx.x & 31) == 0) {
            asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                         :: "r"(tmem_smem(&tile.probability_ready[stage])) : "memory");
        }
    }
    const int final = chunks - 1;
    tmem_mma_wait(tmem_smem(&tile.output_ready[final & 1]), (final >> 1) & 1);
    tmem_store_output<true>(row_address, state, request, row, head, active_rows);
}

template <int Count, bool ScoreLookahead = false>
__global__ __launch_bounds__(256, 1) void tmem_staged_page_attention(
    const __grid_constant__ StagedRequests<Count> requests,
    const __grid_constant__ StagedPairMaps maps, const int* invalid) {
    extern __shared__ StagedPageTile page_shared[];
    auto& tile = page_shared[0];
    const int lane = requests.lane();
    const Attention request = requests.items[lane];
    const int row = blockIdx.x * flash_rows;
    const int head = blockIdx.y * ordered_heads;
    const int position = request.position + row;
    const int active_rows = min(flash_rows, requests.rows[lane] - row);
    if (active_rows <= 0) return;
    const int end = position + active_rows;
    const int chunks = (end + flash_keys - 1) / flash_keys;
    const int page_base = requests.page_base[lane];
    const int warp = threadIdx.x / 32;
    if (threadIdx.x == 0) {
        tile.bad_key = INT_MAX;
        for (int stage = 0; stage < 2; ++stage) {
            pipeline_init(tile.key_ready[stage], 1);
            pipeline_init(tile.value_ready[stage], 1);
            pipeline_init(tile.scores_ready[stage], 1);
            pipeline_init(tile.output_ready[stage], 1);
            pipeline_init(tile.probability_ready[stage], flash_queries / 32);
        }
    }
    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                     "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                     :: "r"(tmem_smem(&tile.tmem)), "r"(tmem_cols) : "memory");
    }
    tmem_stage_query<flash_queries, 8>(tile, request, row, head, active_rows);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x < flash_queries) {
        staged_page_softmax(tile, request, row, head, position, active_rows, chunks);
    } else if (threadIdx.x == 128) {
        stage_page_keys(tile, maps, invalid, page_base, head / ratio, (chunks + 1) / 2,
                        request.cache.kv_heads);
    } else if (threadIdx.x == 160) {
        single_stage<false>(tile, maps, invalid, page_base, head / ratio,
                            request.cache.kv_heads, end);
    } else if (warp == 6) {
        issue_page_mmas<ScoreLookahead>(tile, chunks);
    }
    __syncthreads();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(tile.tmem), "r"(tmem_cols) : "memory");
    }
    tmem_repair_ordered(tile, request, row, head, active_rows);
}
