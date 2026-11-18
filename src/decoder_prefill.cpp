#include "sglang/decoder_prefill.h"
#include "sglang/decoder_mtp.h"
#include "sglang/prefill_fp8.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <mutex>
#include <utility>

namespace sglang::decoder {

struct PrefillPool {
    explicit PrefillPool(int ordinal) : device(ordinal) {}

    ~PrefillPool() {
        if (handle) {
            cudaMemPoolDestroy(handle);
        }
    }

    const int device;
    cudaMemPool_t handle = nullptr;
    void* pending = nullptr;
    bool retains_backing = false;
    PrefillPool* next_retained = nullptr;
    std::unique_ptr<PrefillPool> next;
};

}

namespace sglang {
namespace {

using namespace decoder;

class PrefillPools {
public:
    void configure_retention(PrefillPool& pool) {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            for (auto* owner = retained_; owner; owner = owner->next_retained) {
                if (owner->device == pool.device) {
                    return;
                }
            }
            pool.next_retained = retained_;
            retained_ = &pool;
        }

        std::uint64_t threshold = 32 * 1024 * 1024;
        const auto status =
            cudaMemPoolSetAttribute(pool.handle, cudaMemPoolAttrReleaseThreshold, &threshold);
        if (status == cudaSuccess) {
            pool.retains_backing = true;
            return;
        }

        {
            const std::lock_guard<std::mutex> lock(mutex_);
            auto** link = &retained_;
            while (*link != &pool) {
                link = &(*link)->next_retained;
            }
            *link = pool.next_retained;
            pool.next_retained = nullptr;
        }
        if (status != cudaErrorNotSupported) {
            check(status, "Configure prefill pool retention");
        }
    }

    std::unique_ptr<PrefillPool> take_idle(int device) {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (auto pool = take(idle_, device, true)) {
            return pool;
        }
        return take(idle_, device);
    }

    std::unique_ptr<PrefillPool> take_pending(int device) {
        if (!has_pending_.load(std::memory_order_relaxed)) {
            return {};
        }
        const std::lock_guard<std::mutex> lock(mutex_);
        std::unique_ptr<PrefillPool> pending;
        while (auto pool = take(pending_, device)) {
            pool->next = std::move(pending);
            pending = std::move(pool);
        }
        has_pending_.store(pending_ != nullptr, std::memory_order_relaxed);
        return pending;
    }

    void put(std::unique_ptr<PrefillPool> pool) {
        const std::lock_guard<std::mutex> lock(mutex_);
        auto& list = pool->pending ? pending_ : idle_;
        pool->next = std::move(list);
        list = std::move(pool);
        if (pending_) {
            has_pending_.store(true, std::memory_order_relaxed);
        }
    }

private:
    static std::unique_ptr<PrefillPool> take(std::unique_ptr<PrefillPool>& list, int device,
                                             bool must_retain_backing = false) {
        auto* link = &list;
        while (*link && ((*link)->device != device ||
                        (must_retain_backing && !(*link)->retains_backing))) {
            link = &(*link)->next;
        }
        auto pool = std::move(*link);
        if (pool) {
            *link = std::move(pool->next);
        }
        return pool;
    }

    std::mutex mutex_;
    PrefillPool* retained_ = nullptr;
    std::atomic<bool> has_pending_{false};
    std::unique_ptr<PrefillPool> idle_;
    std::unique_ptr<PrefillPool> pending_;
};

PrefillPools& prefill_pools() {
    static PrefillPools pools;
    return pools;
}

void recover_prefill_pool(PrefillPool& pool) noexcept {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device != pool.device ||
        cudaDeviceSynchronize() != cudaSuccess) {
        return;
    }
    std::uint64_t used = 0;
    if (cudaMemPoolGetAttribute(pool.handle, cudaMemPoolAttrUsedMemCurrent, &used) != cudaSuccess) {
        return;
    }
    if (used == 0 ||
        (cudaFreeAsync(pool.pending, cudaStreamPerThread) == cudaSuccess &&
         cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess)) {
        pool.pending = nullptr;
    }
}

bool prefill_allocation_unavailable(cudaError_t status) {
    return status == cudaErrorMemoryAllocation || status == cudaErrorNotSupported;
}

struct ScratchLayout {
    std::byte* data = nullptr;
    std::size_t bytes = 0;

    template <typename Element>
    Element* take(std::size_t count) {
        constexpr std::size_t alignment = 16;
        const auto offset = (bytes + alignment - 1) / alignment * alignment;
        bytes = offset + count * sizeof(Element);
        return data ? reinterpret_cast<Element*>(data + offset) : nullptr;
    }
};

struct TileBuffers {
    TileBuffers(ScratchLayout& layout, int rows)
        : hidden(layout.take<Bf16>(rows * hidden_size)),
          normalized(layout.take<Bf16>(rows * hidden_size)),
          residual(layout.take<Bf16>(rows * hidden_size)),
          projected_attention(layout.take<Bf16>(rows * hidden_size)),
          attention(layout.take<Bf16>(rows * std::max(text_gdn_value_width, text_gqa_q_width))),
          mlp_gate(layout.take<Bf16>(rows * intermediate_size)),
          mlp_up(layout.take<Bf16>(rows * intermediate_size)),
          q_gate(layout.take<Bf16>(rows * text_gqa_q_gate_width)),
          keys(layout.take<Bf16>(rows * text_gqa_kv_width)),
          values(layout.take<Bf16>(rows * text_gqa_kv_width)),
          rope_queries(layout.take<Bf16>(rows * text_gqa_q_width)),
          rope_keys(layout.take<Bf16>(rows * text_gqa_kv_width)),
          gdn_qkv(layout.take<Bf16>(rows * text_gdn_qkv_width)),
          gdn_z(layout.take<Bf16>(rows * text_gdn_value_width)),
          gdn_b(layout.take<Bf16>(rows * text_gdn_value_heads)),
          gdn_a(layout.take<Bf16>(rows * text_gdn_value_heads)),
          gdn_scratch(layout.take<Bf16>(text_gdn_workspace_bytes(rows) / sizeof(Bf16))),
          activation{
              layout.take<std::uint8_t>(rows * intermediate_size),
              layout.take<float>(rows * text_fp8_group_count(intermediate_size))} {}

    Bf16* hidden;
    Bf16* normalized;
    Bf16* residual;
    Bf16* projected_attention;
    Bf16* attention;
    Bf16* mlp_gate;
    Bf16* mlp_up;
    Bf16* q_gate;
    Bf16* keys;
    Bf16* values;
    Bf16* rope_queries;
    Bf16* rope_keys;
    Bf16* gdn_qkv;
    Bf16* gdn_z;
    Bf16* gdn_b;
    Bf16* gdn_a;
    Bf16* gdn_scratch;
    TextFp8Activation activation;
};

struct Chunk : PrefillProjections {
    TileBuffers work;
    int position;

    void seed_draft(MtpDraftRequest& draft, const int* shifted_tokens, int seed_rows,
                    const Bf16* final_norm) const {
        static_assert(intermediate_size >= 2 * hidden_size);
        static_assert(std::max(text_gdn_value_width, text_gqa_q_width) >= hidden_size);
        check(text_rms_norm_rows(work.hidden, work.residual, final_norm, hidden_size, seed_rows,
                                 work.normalized, nullptr, stream),
              "Normalize target prompt rows for MTP");
        const PrefillProjections projections{activation, seed_rows, stream};
        draft.seed_prefill(position, shifted_tokens, work.normalized, projections,
                           {work.projected_attention, work.mlp_gate, work.attention,
                            work.keys, work.values, work.rope_keys});
    }

    void gdn(const GdnLayer& layer, TextGdnState state) const {
        const auto input = quantize(work.normalized, hidden_size);
        project(layer.qkv, input, work.gdn_qkv);
        project(layer.z, input, work.gdn_z);
        project(layer.b, work.normalized, work.gdn_b);
        project(layer.a, work.normalized, work.gdn_a);
        check(text_gdn_forward(rows, TextGdnMode::prefill, work.gdn_qkv, work.gdn_z, work.gdn_b,
                               work.gdn_a, layer.recurrent, state, work.gdn_scratch,
                               text_gdn_workspace_bytes(rows), work.attention, stream),
              "Advance prefill Gated DeltaNet state");
        project(layer.output, quantize(work.attention, text_gdn_value_width),
                work.projected_attention);
    }

    void gqa(const GqaLayer& layer, Mxfp8KvCache cache, Mxfp8PageTable pages) const {
        const auto input = quantize(work.normalized, hidden_size);
        project(layer.q_gate, input, work.q_gate);
        project(layer.key, input, work.keys);
        project(layer.value, input, work.values);
        check(text_gqa_preprocess(position, rows, work.q_gate, work.keys, layer.query_norm,
                                  layer.key_norm, work.rope_queries, work.rope_keys, stream),
              "Normalize and rotate prefill Q/K");
        check(mxfp8_kv_append(cache, pages, position, rows, work.rope_keys, work.values, stream),
              "Append prefill MXFP8 KV");
        check(mxfp8_gqa_attention(cache, pages, position + rows, position, rows, work.rope_queries,
                                 work.attention, stream), "Read causal prefill MXFP8 attention");
        check(text_gqa_postgate(rows, work.q_gate, work.attention, work.attention, stream),
              "Gate prefill grouped-query attention");
        project(layer.output, quantize(work.attention, text_gqa_q_width), work.projected_attention);
    }

    void mlp(const Layer& layer) const {
        const auto input = quantize(work.normalized, hidden_size);
        project(layer.mlp_gate, input, work.mlp_gate);
        project(layer.mlp_up, input, work.mlp_up);
        check(text_swiglu(work.mlp_gate, work.mlp_up, rows * intermediate_size,
                          work.mlp_gate, stream), "Apply prefill SwiGLU");
        project(layer.mlp_down, quantize(work.mlp_gate, intermediate_size), work.hidden);
    }
};

}

TextFp8ActivationView PrefillProjections::quantize(const Bf16* input, int width) const {
    check(text_quantize_fp8_rows(input, width, rows, activation, stream),
          "Quantize prefill projection input");
    return {activation.values, activation.scales};
}

void PrefillProjections::project(Fp8Weight weight, TextFp8ActivationView input,
                                 Bf16* output) const {
    if (tensor_core_fp8) {
        const auto status = try_text_fp8_gemm_sm100(
            weight.values, weight.scales, input, rows, weight.rows, weight.columns,
            output, stream);
        if (status) {
            check(*status, "Run tensor-core prefill FP8 projection");
            return;
        }
    }
    check(text_fp8_gemv_rows(weight.values, weight.scales, input, weight.rows, weight.columns,
                             rows, output, stream), "Run exact prefill FP8 projection");
}

void PrefillProjections::project(Bf16Weight weight, const Bf16* input, Bf16* output) const {
    check(text_bf16_gemv_rows(weight.values, input, weight.rows, weight.columns, rows, output,
                              stream), "Run exact prefill BF16 projection");
}

void decoder::reclaim_prefill_workspaces(int device) {
    auto& pools = prefill_pools();
    auto pending = pools.take_pending(device);
    while (pending) {
        auto pool = std::move(pending);
        pending = std::move(pool->next);
        recover_prefill_pool(*pool);
        pools.put(std::move(pool));
    }
}

PrefillWorkspace::PrefillWorkspace(cudaStream_t stream, int capacity)
    : stream_(stream), capacity_(capacity) {
    if (!stream || capacity < 1 || capacity > prefill_chunk_rows) {
        throw std::invalid_argument("Invalid decoder prefill workspace");
    }
    if (capacity == 1) {
        return;
    }
    int device = 0;
    check(cudaGetDevice(&device), "Get CUDA device for prefill workspace");
    reclaim_prefill_workspaces(device);
    auto& pools = prefill_pools();
    auto pool = pools.take_idle(device);
    if (!pool) {
        pool = std::make_unique<PrefillPool>(device);
        cudaMemPoolProps props{};
        props.allocType = cudaMemAllocationTypePinned;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id = device;
        const auto created = cudaMemPoolCreate(&pool->handle, &props);
        if (prefill_allocation_unavailable(created)) {
            return;
        }
        check(created, "Create prefill workspace pool");
        int disabled = 0;
        check(cudaMemPoolSetAttribute(pool->handle, cudaMemPoolReuseAllowInternalDependencies,
                                      &disabled), "Configure prefill workspace pool");
        pools.configure_retention(*pool);
    }
    ScratchLayout layout;
    const TileBuffers buffers(layout, capacity);
    void* data = nullptr;
    const auto status = cudaMallocFromPoolAsync(&data, layout.bytes, pool->handle, stream);
    if (status != cudaSuccess) {
        pools.put(std::move(pool));
        if (!prefill_allocation_unavailable(status)) {
            check(status, "Allocate prefill workspace");
        }
        return;
    }
    pool_ = std::move(pool);
    data_ = data;
    bytes_ = layout.bytes;
}

PrefillWorkspace::~PrefillWorkspace() {
    if (release_status_ == cudaSuccess) {
        const auto status = release();
        if (status != cudaSuccess) {
            std::fprintf(stderr, "Release prefill workspace: %s\n", cudaGetErrorString(status));
        }
    }
}

cudaError_t PrefillWorkspace::release() noexcept {
    if (data_ && release_status_ == cudaSuccess) {
        release_status_ = cudaFreeAsync(data_, stream_);
        if (release_status_ != cudaSuccess) {
            pool_->pending = data_;
            recover_prefill_pool(*pool_);
        }
        prefill_pools().put(std::move(pool_));
        data_ = nullptr;
    }
    stream_ = nullptr;
    return release_status_;
}

void PrefillWorkspace::close() {
    check(release(), "Release prefill workspace");
}

void Decoder::Model::prefill_chunk(Request& request, const int* tokens, int rows, int position,
                                   PrefillWorkspace& workspace, MtpDraftRequest* draft,
                                   int prompt_remaining) const {
    const auto stream = workspace.stream();
    if (!stream || !tokens || rows < 1 || rows > workspace.capacity() || position < 0 ||
        position > text_attention_max_tokens - rows ||
        (position + rows - 1) / mxfp8_page_size >= request.pages.page_count ||
        (draft && prompt_remaining < rows)) {
        throw std::invalid_argument("Invalid decoder prefill tile");
    }
    if (rows == 1 || !workspace.data()) {
        if (!std::all_of(tokens, tokens + rows,
                         [](int token) { return token >= 0 && token < vocab_size; })) {
            throw std::invalid_argument("Invalid decoder prefill token");
        }
        for (int row = 0; row < rows; ++row) {
            const ForwardLane lane{&request, tokens[row], position + row, TextGdnMode::prefill};
            forward(&lane, 1, stream);
            if (draft && row + 1 < prompt_remaining) {
                normalize_final(request.work, stream);
                draft->enqueue(position + row, tokens[row + 1], request.work.normalized, false,
                               stream);
            }
        }
        return;
    }
    ScratchLayout layout{static_cast<std::byte*>(workspace.data())};
    const TileBuffers buffers(layout, workspace.capacity());
    const bool tensor_core = !draft && fp8_prefill() == Fp8PrefillPolicy::tensor_core;
    const Chunk chunk{{buffers.activation, rows, stream, tensor_core}, buffers, position};
    const auto& work = chunk.work;
    check(text_embedding_rows(embedding, vocab_size, hidden_size, tokens, rows,
                               work.residual, stream), "Gather prefill embeddings");
    int gqa_index = 0;
    int gdn_index = 0;
    for (int index = 0; index < layer_count; ++index) {
        const auto& layer = layers[index];
        const auto* input = index == 0 ? work.residual : work.hidden;
        auto* residual = index == 0 ? nullptr : work.residual;
        check(text_rms_norm_rows(input, residual, layer.input_norm, hidden_size, rows,
                                 work.normalized, residual, stream),
              "Normalize prefill layer input");
        if (is_full_attention(index)) {
            chunk.gqa(layer.gqa, request.caches[gqa_index++], request.pages);
        } else {
            chunk.gdn(layer.gdn, request.states[gdn_index++]);
        }
        check(text_rms_norm_rows(work.projected_attention, work.residual, layer.attention_norm,
                                 hidden_size, rows, work.normalized, work.residual, stream),
              "Normalize prefill post-attention residual");
        chunk.mlp(layer);
    }
    if (draft && prompt_remaining > 1) {
        chunk.seed_draft(*draft, tokens + 1, std::min(rows, prompt_remaining - 1), final_norm);
    }
    const int last = (rows - 1) * hidden_size;
    const auto bytes = hidden_size * sizeof(Bf16);
    check(cudaMemcpyAsync(request.work.hidden, work.hidden + last, bytes,
                          cudaMemcpyDeviceToDevice, stream), "Stage final prefill hidden row");
    check(cudaMemcpyAsync(request.work.residual, work.residual + last, bytes,
                          cudaMemcpyDeviceToDevice, stream), "Stage final prefill residual row");
}

}
