constexpr int reuse_rows = text_gdn_max_verify_rows;
constexpr int reuse_requests = decoder::target_split_eager_max_rows / reuse_rows;
constexpr int reuse_threads = reuse_rows * threads;
static_assert(reuse_rows == 5 && reuse_requests == 4);

struct ReusePacket {
    decoder::PagedDecodeLane rows[reuse_requests][reuse_rows];
};

struct ReuseStorage {
    union {
        Stage stage[2];
        Merge merge[reuse_rows];
    } data;
    WarpScales scales[reuse_rows][warps];
    int pages[max_split_pages];
    Query queries[reuse_rows];
};

__global__ __launch_bounds__(reuse_threads, 1)
void reuse_split_kernel(const __grid_constant__ ReusePacket input) {
    alignas(16) extern __shared__ unsigned char shared_raw[];
    auto& shared = *reinterpret_cast<ReuseStorage*>(shared_raw);
    const auto& group = input.rows[blockIdx.z];
    const auto& newest = group[reuse_rows - 1];
    const int query_index = threadIdx.x / threads;
    const auto& lane = group[query_index];
    const int position = lane.position;
    const int split = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int warp_thread = threadIdx.x % threads;
    const int warp = warp_thread / 32;
    const int lane_id = warp_thread % 32;
    const int chunks = newest.position / chunk_tokens + 1;
    const int first = split * chunks / max_splits;
    const int last = (split + 1) * chunks / max_splits;
    const int first_page = first / chunks_per_page;
    const int page_count = (last - 1) / chunks_per_page - first_page + 1;
    if (threadIdx.x < page_count) {
        shared.pages[threadIdx.x] = newest.pages.device_pages[first_page + threadIdx.x];
    }
    auto& query = shared.queries[query_index];
    auto& scales = shared.scales[query_index][warp];
    if (warp == 0) {
        load_query(query, lane.buffers.queries, kv_head, lane_id);
    }
    __syncthreads();
    if (query_index == 0) {
        issue_warp_tile(shared.pages, shared.data.stage[0], newest, kv_head, first, first_page,
                        warp, lane_id);
        commit();
        wait_pending<0>();
    }
    __syncthreads();
    Running run{};
    run.maximum = -CUDART_INF_F;
    run.flag = query.flag[lane_id];
    const int tail = position / chunk_tokens;
    for (int chunk = first; chunk < last; ++chunk) {
        const int slot = (chunk - first) % 2;
        if (query_index == 0 && chunk + 1 < last) {
            issue_warp_tile(shared.pages, shared.data.stage[1 - slot], newest, kv_head, chunk + 1,
                            first_page, warp, lane_id);
            commit();
        }
        if (chunk < tail) {
            process_tile(shared.data.stage[slot], scales, query, run, chunk, position, warp,
                         lane_id);
        } else {
            process_tile<true>(shared.data.stage[slot], scales, query, run, chunk, position, warp,
                               lane_id);
        }
        if (query_index == 0) {
            wait_pending<0>();
        }
        __syncthreads();
    }
    auto& merge = shared.data.merge[query_index];
    store_warp(merge, run, warp, lane_id / 4, lane_id % 4);
    __syncthreads();
    write_weighted_partials(merge, lane.buffers.scratch, kv_head, split, warp_thread);
}

bool same_cache(const decoder::PagedDecodeLane& a, const decoder::PagedDecodeLane& b) {
    return a.pages.device_pages == b.pages.device_pages &&
           a.pages.page_count == b.pages.page_count &&
           a.cache.kv_heads == b.cache.kv_heads &&
           a.cache.key_payload == b.cache.key_payload &&
           a.cache.value_payload == b.cache.value_payload &&
           a.cache.key_scales == b.cache.key_scales &&
           a.cache.value_scales == b.cache.value_scales &&
           a.cache.page_count == b.cache.page_count;
}

bool make_reuse_packet(const EagerTargetBatch& batch, int count, ReusePacket& packet) {
    if (count != reuse_requests * reuse_rows || batch.end != text_attention_max_tokens) {
        return false;
    }
    int lengths[reuse_requests]{};
    int groups = 0;
    for (int i = 0; i < count; ++i) {
        const auto& row = batch.lanes[i];
        if (row.position < 32768 || row.position >= 40960 ||
            !readable(row) || !valid_buffers(row.buffers) || !tensor_aligned(row)) {
            return false;
        }
        int group = 0;
        while (group < groups && !same_cache(packet.rows[group][0], row)) {
            ++group;
        }
        if (group == groups) {
            if (groups == reuse_requests) {
                return false;
            }
            ++groups;
        }
        const int index = lengths[group];
        const int start = index == 0 ? row.position : packet.rows[group][0].position;
        if (index >= reuse_rows || row.position != start + index ||
            row.position / chunk_tokens != start / chunk_tokens) {
            return false;
        }
        packet.rows[group][index] = row;
        ++lengths[group];
    }
    return true;
}

bool reuse_kernel_unavailable(cudaError_t status) {
    return status == cudaErrorInvalidDeviceFunction || status == cudaErrorNoKernelImageForDevice ||
           status == cudaErrorNotSupported;
}

std::optional<cudaError_t> reuse_failure(cudaError_t status, bool can_fall_back) {
    const auto pending = cudaPeekAtLastError();
    if (pending == status) {
        (void)cudaGetLastError();
    }
    if (!can_fall_back) {
        return status;
    }
    if (pending != cudaSuccess && pending != status) {
        return pending;
    }
    return std::nullopt;
}

std::optional<cudaError_t> launch_reuse(const EagerTargetBatch& batch, int count,
                                       cudaStream_t stream) {
    ReusePacket packet;
    if (!make_reuse_packet(batch, count, packet)) {
        return std::nullopt;
    }
    if (const auto pending = cudaPeekAtLastError(); pending != cudaSuccess) {
        return pending;
    }
    const auto kernel = reuse_split_kernel;
    auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       sizeof(ReuseStorage));
    if (status != cudaSuccess) {
        return reuse_failure(status, status == cudaErrorInvalidValue ||
                                         reuse_kernel_unavailable(status));
    }
    void* arguments[] = {&packet};
    const dim3 blocks(max_splits, packet.rows[0][0].cache.kv_heads, reuse_requests);
    status = cudaLaunchKernel(kernel, blocks, reuse_threads, arguments, sizeof(ReuseStorage),
                              stream);
    if (status != cudaSuccess) {
        return reuse_failure(status, reuse_kernel_unavailable(status) ||
                                         status == cudaErrorInvalidConfiguration ||
                                         status == cudaErrorLaunchOutOfResources);
    }
    return cudaSuccess;
}
