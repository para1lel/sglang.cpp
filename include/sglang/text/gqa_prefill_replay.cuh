constexpr int prefill_replay_threads = 128;

struct PrefillReplayTile {
    alignas(16) std::uint8_t values[replay_tile][replay_channels];
    alignas(8) std::uint8_t exponents[replay_tile][mxfp8_groups];
};

__device__ void load_prefill_replay(PrefillReplayTile& tile, Mxfp8KvCache cache,
                                    const std::int32_t* pages, int first, int last,
                                    int kv_head, int first_channel) {
    constexpr int vectors = replay_channels / 16;
    const int token = threadIdx.x / vectors;
    const int column = threadIdx.x % vectors * 16;
    const int page = pages[first / mxfp8_page_size];
    const int slot = (first + token) % mxfp8_page_size;
    const bool visible = first + token <= last;
    const auto payload =
        payload_offset(cache.kv_heads, page, slot, kv_head) + first_channel + column;
    const auto values =
        static_cast<unsigned>(__cvta_generic_to_shared(&tile.values[token][column]));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(values), "l"(cache.value_payload + payload), "r"(visible ? 16 : 0)
                 : "memory");
    if (column == 0) {
        const auto scale = scale_offset(cache.kv_heads, page, slot, kv_head, 0);
        const auto exponents =
            static_cast<unsigned>(__cvta_generic_to_shared(tile.exponents[token]));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;"
                     :: "r"(exponents), "l"(cache.value_scales + scale), "r"(visible ? 8 : 0)
                     : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__global__ void prefill_replay_kernel(Mxfp8KvCache cache, const std::int32_t* pages, int last,
                                      decoder::ExactGqaWorkspace workspace,
                                      __nv_bfloat16* output) {
    __shared__ PrefillReplayTile tiles[2];
    __shared__ float history_weight[replay_tile], token_weight[replay_tile], denominator;
    __shared__ bool unit_history;
    const int head = blockIdx.x;
    const int thread = threadIdx.x;
    const int first_channel = blockIdx.y * replay_channels;
    const int group = (first_channel + thread) / mxfp8_group_size;
    const int kv_head = head / queries_per_kv_head;
    const int query_heads = gqa_query_heads(cache.kv_heads);
    if (head >= query_heads) {
        return;
    }
    const auto* scores = workspace.scores + head * workspace.capacity;
    float maximum = -CUDART_INF_F;
    float mass = 0.0f;
    float weighted = 0.0f;
    load_prefill_replay(tiles[0], cache, pages, 0, last, kv_head, first_channel);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    for (int first = 0; first <= last; first += replay_tile) {
        const int count = min(replay_tile, last - first + 1);
        const int stage = first / replay_tile % 2;
        if (first + replay_tile <= last) {
            load_prefill_replay(tiles[1 - stage], cache, pages, first + replay_tile,
                                last, kv_head, first_channel);
        }
        if (thread < replay_tile) {
            const float score = thread < count ? scores[first + thread] : -CUDART_INF_F;
            const bool unit = isfinite(maximum) && __all_sync(warp_mask, score <= maximum);
            if (thread == 0) {
                unit_history = unit;
            }
            float prefix = fmaxf(maximum, score);
            if (!unit) {
                #pragma unroll
                for (int offset = 1; offset < replay_tile; offset *= 2) {
                    const float earlier = __shfl_up_sync(warp_mask, prefix, offset);
                    if (thread >= offset) {
                        prefix = fmaxf(earlier, prefix);
                    }
                }
                const float previous = __shfl_up_sync(warp_mask, prefix, 1);
                history_weight[thread] = expf((thread == 0 ? maximum : previous) - prefix);
                maximum = __shfl_sync(warp_mask, prefix, replay_tile - 1);
            }
            token_weight[thread] = expf(score - prefix);
        }
        __syncthreads();
        if (thread < replay_channels) {
            const auto& tile = tiles[stage];
            if (unit_history) {
                #pragma unroll 4
                for (int token = 0; token < count; ++token) {
                    const float value = decode_e8m0(tile.values[token][thread],
                                                   tile.exponents[token][group]);
                    weighted = fmaf(token_weight[token], value, weighted);
                }
            } else {
                #pragma unroll 4
                for (int token = 0; token < count; ++token) {
                    const float value = decode_e8m0(tile.values[token][thread],
                                                   tile.exponents[token][group]);
                    weighted = fmaf(token_weight[token], value, history_weight[token] * weighted);
                }
            }
        }
        if (thread == 0) {
            for (int token = 0; token < count; ++token) {
                mass = unit_history ? mass + token_weight[token] :
                    fmaf(history_weight[token], mass, token_weight[token]);
            }
        }
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }
    if (thread == 0) {
        denominator = mass;
    }
    __syncthreads();
    if (thread < replay_channels) {
        output[row_offset(0, query_heads, head) + first_channel + thread] =
            __float2bfloat16_rn(weighted / denominator);
    }
}
