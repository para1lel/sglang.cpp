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
    constexpr int payload_stride = mxfp8_kv_heads * mxfp8_head_dim;
    constexpr int scale_stride = (mxfp8_page_size / mxfp8_group_size) * mxfp8_groups;
    const auto payload = payload_offset(page, token, kv_head) + group * vector_replay_channels;
    const auto scale = scale_offset(page, token, kv_head, group);
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

template <bool SkipTargetSplit, typename Position, typename Cache, typename Pages, typename Scores>
__global__ void vector_replay_kernel(Cache cache_binding, Pages page_binding, Position position,
                                     Scores score_binding, __nv_bfloat16* output) {
    __shared__ alignas(16) float values[replay_tile][vector_replay_channels];
    __shared__ float2 weights[replay_tile];
    __shared__ float denominator;
    __shared__ bool unit_history;
    const auto cache = cache_value(cache_binding);
    const auto* pages = page_values(page_binding);
    const int last = decoder::position_value(position);
    if constexpr (SkipTargetSplit) {
        if (decoder::target_split_position(last)) {
            return;
        }
    }
    const auto workspace = score_workspace(score_binding);
    const int head = blockIdx.x;
    const int kv_head = head / queries_per_kv_head;
    const int group = blockIdx.y;
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
        output[row_offset(0, mxfp8_query_heads, head) + column + lane] =
            __float2bfloat16_rn(weighted / denominator);
    }
}
