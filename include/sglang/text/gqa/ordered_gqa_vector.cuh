constexpr int vector_replay_threads = 4 * mxfp8_group_size;
constexpr int vector_replay_channels = mxfp8_group_size;
static_assert(mxfp8_payload_bytes_per_page % alignof(std::uint32_t) == 0 &&
              mxfp8_head_dim % alignof(std::uint32_t) == 0 &&
              vector_replay_channels % alignof(std::uint32_t) == 0);

__device__ bool replay_values_aligned(Mxfp8KvCache cache) {
    return reinterpret_cast<std::uintptr_t>(cache.value_payload) % alignof(std::uint32_t) == 0;
}

__device__ __forceinline__ std::uint32_t replay_value_word(
    const std::uint8_t* source, bool aligned_values) {
    if (aligned_values) {
        return *reinterpret_cast<const std::uint32_t*>(source);
    }
    return static_cast<std::uint32_t>(source[0]) |
           (static_cast<std::uint32_t>(source[1]) << 8) |
           (static_cast<std::uint32_t>(source[2]) << 16) |
           (static_cast<std::uint32_t>(source[3]) << 24);
}

__device__ __forceinline__ void load_replay_values(
    Mxfp8KvCache cache, int page, int token, int kv_head, int group, int count,
    bool aligned_values, float (&values)[replay_tile][vector_replay_channels]) {
    constexpr int vectors = vector_replay_channels / 4;
    const int payload_stride = cache.kv_heads * mxfp8_head_dim;
    constexpr int scale_stride = (mxfp8_page_size / mxfp8_group_size) * mxfp8_groups;
    const auto payload = payload_offset(cache.kv_heads, page, token, kv_head)
                       + group * vector_replay_channels;
    const auto scale = scale_offset(cache.kv_heads, page, token, kv_head, group);
    const int vector = threadIdx.x % vectors;
    #pragma unroll
    for (int index = threadIdx.x / vectors; index < replay_tile;
         index += vector_replay_threads / vectors) {
        if (index < count) {
            const auto* source = cache.value_payload + payload + index * payload_stride +
                                 vector * 4;
            const auto packed = replay_value_word(source, aligned_values);
            const auto exponent = cache.value_scales[scale + index * scale_stride];
            *reinterpret_cast<float4*>(&values[index][vector * 4]) =
                make_float4(decode_e8m0(packed, exponent), decode_e8m0(packed >> 8, exponent),
                            decode_e8m0(packed >> 16, exponent),
                            decode_e8m0(packed >> 24, exponent));
        }
    }
}

template <bool Unit>
__device__ float replay_vector_channel(
                                        const float (&values)[replay_tile][vector_replay_channels],
                                        const float2* weights, int count, int lane,
                                        float weighted) {
    #pragma unroll 8
    for (int index = 0; index < count; ++index) {
        const auto factor = weights[index];
        weighted = fmaf(factor.y, values[index][lane], Unit ? weighted : factor.x * weighted);
    }
    return weighted;
}

/// The replay pass shared by vector_replay_kernel and the batched verify
/// kernels: `head` and `group` name the block coordinates the caller guards.
__device__ __noinline__ void vector_replay_body(Mxfp8KvCache cache, const std::int32_t* pages,
                                                int last, decoder::ExactGqaWorkspace workspace,
                                                __nv_bfloat16* output, int head, int group) {
    __shared__ alignas(16) float values[replay_tile][vector_replay_channels];
    __shared__ float2 weights[replay_tile];
    __shared__ float denominator;
    __shared__ bool unit_history;
    const int query_heads = gqa_query_heads(cache.kv_heads);
    const int kv_head = head / queries_per_kv_head;
    const int thread = threadIdx.x;
    const int lane = thread % mxfp8_group_size;
    const int column = group * vector_replay_channels;
    const auto* scores = workspace.scores + head * workspace.capacity;
    float maximum = -CUDART_INF_F;
    float mass = 0.0f;
    float weighted = 0.0f;
    const bool aligned_values = replay_values_aligned(cache);
    for (int first = 0; first <= last; first += replay_tile) {
        const int count = min(replay_tile, last - first + 1);
        const int physical_page = pages[first / mxfp8_page_size];
        const int token = first % mxfp8_page_size;
        load_replay_values(cache, physical_page, token, kv_head, group, count, aligned_values,
                           values);
        if (thread < mxfp8_group_size) {
            const float score = lane < count ? scores[first + lane] : -CUDART_INF_F;
            const bool unit = isfinite(maximum) && __all_sync(warp_mask, score <= maximum);
            if (lane == 0) {
                unit_history = unit;
            }
            float prefix = fmaxf(maximum, score);
            if (unit) {
                if (lane < count) {
                    weights[lane] = {1.0f, expf(score - prefix)};
                }
            } else {
                #pragma unroll
                for (int offset = 1; offset < replay_tile; offset *= 2) {
                    const float earlier = __shfl_up_sync(warp_mask, prefix, offset);
                    if (lane >= offset) {
                        prefix = fmaxf(earlier, prefix);
                    }
                }
                const float previous = __shfl_up_sync(warp_mask, prefix, 1);
                if (lane < count) {
                    weights[lane] = {expf((lane == 0 ? maximum : previous) - prefix),
                                     expf(score - prefix)};
                }
                maximum = __shfl_sync(warp_mask, prefix, replay_tile - 1);
            }
        }
        __syncthreads();
        if (thread < mxfp8_group_size) {
            weighted = unit_history
                ? replay_vector_channel<true>(values, weights, count, lane, weighted)
                : replay_vector_channel<false>(values, weights, count, lane, weighted);
        }
        if (thread == mxfp8_group_size) {
            #pragma unroll 8
            for (int index = 0; index < count; ++index) {
                const auto factor = weights[index];
                mass = fmaf(factor.x, mass, factor.y);
            }
            denominator = mass;
        }
        __syncthreads();
    }
    if (thread < mxfp8_group_size) {
        output[row_offset(0, query_heads, head) + column + lane] =
            __float2bfloat16_rn(weighted / denominator);
    }
}

template <bool SkipTargetSplit, typename Position, typename Cache, typename Pages, typename Scores>
__global__ void vector_replay_kernel(Cache cache_binding, Pages page_binding, Position position,
                                     Scores score_binding, __nv_bfloat16* output, int split_end) {
    const auto cache = cache_value(cache_binding);
    const auto* pages = page_values(page_binding);
    const int last = decoder::position_value(position);
    if constexpr (SkipTargetSplit) {
        if (decoder::target_split_position(last, split_end)) {
            return;
        }
    }
    const auto workspace = score_workspace(score_binding);
    const int head = blockIdx.x;
    // Bound descriptors launch the replay with the maximum head grid; the
    // lane's own cache decides which blocks carry work.
    if (head >= gqa_query_heads(cache.kv_heads)) {
        return;
    }
    vector_replay_body(cache, pages, last, workspace, output, head, blockIdx.y);
}

template <bool RecomputeScores>
__device__ __noinline__ void ordered_replay(
    Mxfp8KvCache cache, const std::int32_t* pages, int last, const float* scores,
    const __nv_bfloat16* queries, __nv_bfloat16* output) {
    __shared__ float history_weight[replay_tile];
    __shared__ float token_weight[replay_tile];
    __shared__ float denominator;
    __shared__ float values[replay_tile][replay_channels];
    __shared__ float tile_scores[RecomputeScores ? replay_tile : 1];
    const int query_head = blockIdx.x;
    const int kv_head = query_head / queries_per_kv_head;
    const int query_heads = gqa_query_heads(cache.kv_heads);
    // Bound descriptors use the maximum head grid; the lane's cache guards.
    if (query_head >= query_heads) return;
    const int thread = threadIdx.x;
    const int column = thread % replay_channels;
    const int channel = blockIdx.y * replay_channels + column;
    const int group = channel / mxfp8_group_size;
    const int warp = thread / mxfp8_group_size;
    const int lane = thread % mxfp8_group_size;
    float query[mxfp8_groups];
    if constexpr (RecomputeScores)
        load_query(query_heads, queries, query_head, lane, query);
    float maximum = -CUDART_INF_F;
    float weight_sum = 0.0f;
    float weighted_value = 0.0f;
    constexpr int tokens_per_load = replay_threads / replay_channels;
    const int payload_stride = cache.kv_heads * mxfp8_head_dim;
    constexpr int scale_stride = (mxfp8_page_size / mxfp8_group_size) * mxfp8_groups;
    for (int first = 0; first <= last; first += replay_tile) {
        const int count = min(replay_tile, last - first + 1);
        const int physical_page = pages[first / mxfp8_page_size];
        const int token = first % mxfp8_page_size;
        const auto payload =
            payload_offset(cache.kv_heads, physical_page, token, kv_head) + channel;
        const auto scale = scale_offset(cache.kv_heads, physical_page, token, kv_head, group);
        if constexpr (RecomputeScores) {
            for (int index = warp; index < count; index += score_warps) {
                const float score =
                    attention_score(cache, physical_page, token + index, kv_head, lane, query);
                if (lane == 0) {
                    tile_scores[index] = score;
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int index = thread / replay_channels; index < replay_tile; index += tokens_per_load) {
            if (index < count) {
                values[index][column] = decode_e8m0(
                    cache.value_payload[payload + index * payload_stride],
                    cache.value_scales[scale + index * scale_stride]);
            }
        }
        if (thread < mxfp8_group_size) {
            float score = -CUDART_INF_F;
            if (thread < count) {
                if constexpr (RecomputeScores) {
                    score = tile_scores[thread];
                } else {
                    score = scores[first + thread];
                }
            }
            float prefix = fmaxf(maximum, score);
            #pragma unroll
            for (int offset = 1; offset < replay_tile; offset *= 2) {
                const float earlier = __shfl_up_sync(warp_mask, prefix, offset);
                if (thread >= offset) {
                    prefix = fmaxf(earlier, prefix);
                }
            }
            const float previous = __shfl_up_sync(warp_mask, prefix, 1);
            if (thread < count) {
                history_weight[thread] = expf((thread == 0 ? maximum : previous) - prefix);
                token_weight[thread] = expf(score - prefix);
            }
            maximum = __shfl_sync(warp_mask, prefix, replay_tile - 1);
        }
        __syncthreads();

        if (thread < replay_channels) {
            #pragma unroll 4
            for (int index = 0; index < count; ++index) {
                weighted_value = fmaf(token_weight[index], values[index][column],
                                      history_weight[index] * weighted_value);
            }
        }
        if (thread == 0) {
            for (int index = 0; index < count; ++index) {
                weight_sum = fmaf(history_weight[index], weight_sum, token_weight[index]);
            }
            denominator = weight_sum;
        }
        __syncthreads();
    }
    if (thread < replay_channels) {
        output[row_offset(0, query_heads, query_head) + channel] =
            __float2bfloat16_rn(weighted_value / denominator);
    }
}
