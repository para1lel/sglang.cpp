#include "sglang/decoder_prefill.h"
#include "sglang/decoder_batch.h"

#include "sglang/decoder_media_internal.h"
#include "sglang/decoder_mtp.h"
#include "sglang/prefill_fp8.h"
#include "sglang/text/decoder_logprobs.h"
#include "sglang/text/prefill_attn_sm100.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string_view>
#include <utility>

namespace sglang::decoder {

struct PrefillPool {
    PrefillPool(int ordinal, bool large) : device(ordinal), large(large) {}

    ~PrefillPool() {
        if (handle) {
            cudaMemPoolDestroy(handle);
        }
    }

    const int device;
    const bool large;
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
                if (owner->device == pool.device && owner->large == pool.large) {
                    return;
                }
            }
            pool.next_retained = retained_;
            retained_ = &pool;
        }

        std::uint64_t threshold = (pool.large ? 1024 : 32) * 1024 * 1024;
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

    std::unique_ptr<PrefillPool> take_idle(int device, bool large) {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (auto pool = take(idle_, device, large, true)) {
            return pool;
        }
        return take(idle_, device, large);
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
                                             std::optional<bool> large = std::nullopt,
                                             bool must_retain_backing = false) {
        auto* link = &list;
        while (*link && ((*link)->device != device ||
                        (large && (*link)->large != *large) ||
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

    template <typename Element, std::size_t alignment = 16>
    Element* take(std::size_t count) {
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
          mlp_gate(layout.take<Bf16, 32>(rows * intermediate_size)),
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

    PairedPrefillScratch gqa_scratch() const {
        const auto mlp_begin = reinterpret_cast<std::uintptr_t>(mlp_gate);
        const auto mlp_end = reinterpret_cast<std::uintptr_t>(q_gate);
        const auto gdn_begin = reinterpret_cast<std::uintptr_t>(gdn_qkv);
        const auto gdn_end = reinterpret_cast<std::uintptr_t>(activation.values);
        return {reinterpret_cast<std::byte*>(mlp_gate), mlp_end - mlp_begin,
                reinterpret_cast<std::byte*>(gdn_qkv), gdn_end - gdn_begin};
    }

    TileBuffers attention_segment(int offset) const {
        static_assert(text_gdn_value_width == text_gqa_q_width);
        auto segment = *this;
        segment.attention += offset * text_gqa_q_width;
        segment.q_gate += offset * text_gqa_q_gate_width;
        segment.keys += offset * text_gqa_kv_width;
        segment.values += offset * text_gqa_kv_width;
        segment.rope_queries += offset * text_gqa_q_width;
        segment.rope_keys += offset * text_gqa_kv_width;
        segment.gdn_qkv += offset * text_gdn_qkv_width;
        segment.gdn_z += offset * text_gdn_value_width;
        segment.gdn_b += offset * text_gdn_value_heads;
        segment.gdn_a += offset * text_gdn_value_heads;
        return segment;
    }

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
    const MropePosition* mrope;

    bool project_packed(TextFp8ActivationView input,
                        std::initializer_list<PrefillFp8Projection> projections) const {
        if (!tensor_core_fp8) {
            return false;
        }
        const auto status = try_prefill_fp8_packed_sm100(input, rows, hidden_size,
                                                         projections, stream, grouped_projection);
        if (!status) {
            return false;
        }
        check(*status, "Run packed FP8 prefill projections");
        return true;
    }

    void seed_draft(MtpDraftRequest& draft, const int* shifted_tokens, int seed_rows,
                    const Bf16* final_norm, bool wmma) const {
        static_assert(intermediate_size >= 2 * hidden_size);
        static_assert(std::max(text_gdn_value_width, text_gqa_q_width) >= hidden_size);
        const PrefillProjections projections{activation, seed_rows, stream, false, false, wmma};
        draft.seed_prefill(position, shifted_tokens, work.hidden, work.residual, final_norm,
                           projections, {work.normalized, work.projected_attention, work.mlp_gate,
                                         work.attention, work.keys, work.values, work.rope_keys});
    }

    void gdn_input(const GdnLayer& layer) const {
        const TextFp8ActivationView input{activation.values, activation.scales};
        if (!project_packed(input, {
                {layer.qkv.values, layer.qkv.scales, work.gdn_qkv, layer.qkv.rows},
                {layer.z.values, layer.z.scales, work.gdn_z, layer.z.rows},
            })) {
            project(layer.qkv, input, work.gdn_qkv);
            project(layer.z, input, work.gdn_z);
        }
        if (tensor_core_fp8 && layer.b.rows == text_gdn_value_heads &&
            layer.a.rows == text_gdn_value_heads && layer.b.columns == hidden_size &&
            layer.a.columns == hidden_size) {
            const auto status = try_prefill_bf16_gdn_gates(
                layer.b.values, layer.a.values, work.normalized, rows, work.gdn_b, work.gdn_a,
                stream);
            if (status) {
                check(*status, "Project BF16 prefill Gated DeltaNet gates");
                return;
            }
        }
        project(layer.b, work.normalized, work.gdn_b);
        project(layer.a, work.normalized, work.gdn_a);
    }

    void gdn_recur(const GdnLayer& layer, TextGdnState state, int bank,
                   TextFp8Activation quantized) const {
        static_assert(text_gqa_kv_width * sizeof(Bf16) >= text_gdn_value_heads * sizeof(float2));
        const bool chunked_recurrence = position != 0 || bank != 0;
        auto* gates = reinterpret_cast<float2*>(work.keys);
        auto* scratch = chunked_recurrence ? work.mlp_gate : nullptr;
        const auto scratch_bytes = rows * intermediate_size * sizeof(Bf16);
        if (quantized.values) {
            check(tensor_core_prefill_gdn(rows, work.gdn_qkv, work.gdn_z, work.gdn_b, work.gdn_a,
                                          layer.recurrent, state, work.gdn_scratch, gates,
                                          quantized, stream, scratch, scratch_bytes),
                  "Advance and quantize prefill Gated DeltaNet");
            return;
        }
        const auto status = tensor_core_fp8
            ? tensor_core_prefill_gdn(rows, work.gdn_qkv, work.gdn_z, work.gdn_b, work.gdn_a,
                                      layer.recurrent, state, work.gdn_scratch, gates,
                                      work.attention, stream, scratch, scratch_bytes)
            : text_gdn_forward(rows, TextGdnMode::prefill, work.gdn_qkv, work.gdn_z, work.gdn_b,
                               work.gdn_a, layer.recurrent, state, work.gdn_scratch,
                               text_gdn_workspace_bytes(rows), work.attention, stream);
        check(status, "Advance prefill Gated DeltaNet state");
    }

    void gdn_attention(const GdnLayer& layer, TextGdnState state, int bank) const {
        const bool quantized = tensor_core_fp8 && !mrope;
        gdn_recur(layer, state, bank, quantized ? activation : TextFp8Activation{});
        const auto input = quantized
            ? TextFp8ActivationView{activation.values, activation.scales}
            : quantize(work.attention, text_gdn_value_width);
        project(layer.output, input, work.projected_attention);
    }

    void gqa_input(const GqaLayer& layer) const {
        const TextFp8ActivationView input{activation.values, activation.scales};
        if (!project_packed(input, {
                {layer.q_gate.values, layer.q_gate.scales, work.q_gate, layer.q_gate.rows},
                {layer.key.values, layer.key.scales, work.keys, layer.key.rows},
                {layer.value.values, layer.value.scales, work.values, layer.value.rows},
            })) {
            project(layer.q_gate, input, work.q_gate);
            project(layer.key, input, work.keys);
            project(layer.value, input, work.values);
        }
    }

    void gqa_append(const GqaLayer& layer, Mxfp8KvCache cache, Mxfp8PageTable pages) const {
        const auto rotation = mrope
            ? mrope_gqa_preprocess(mrope, rows, work.q_gate, work.keys, layer.query_norm,
                                    layer.key_norm, work.rope_queries, work.rope_keys, stream)
            : text_gqa_preprocess(position, rows, work.q_gate, work.keys, layer.query_norm,
                                   layer.key_norm, work.rope_queries, work.rope_keys, stream);
        check(rotation, "Normalize and rotate prefill Q/K");
        check(mxfp8_kv_append(cache, pages, position, rows, work.rope_keys, work.values, stream),
              "Append prefill MXFP8 KV");
    }

    void gqa_gate() const {
        check(text_gqa_postgate(rows, work.q_gate, work.attention, work.attention, stream),
              "Gate prefill grouped-query attention");
    }

    void gqa_attention(const GqaLayer& layer, Mxfp8KvCache cache, Mxfp8PageTable pages,
                       PairedPrefillScratch scratch) const {
        gqa_append(layer, cache, pages);
        const auto status = tensor_core_fp8
            ? tensor_core_model_prefill_gqa(cache, pages, position, rows, work.rope_queries,
                                            work.attention, stream, scratch)
            : mxfp8_gqa_attention(cache, pages, position + rows, position, rows, work.rope_queries,
                                   work.attention, stream);
        check(status, "Read causal prefill MXFP8 attention");
        gqa_gate();
    }

    void attention_output(Fp8Weight output, int width) const {
        project(output, quantize(work.attention, width), work.projected_attention);
    }

    void mlp(const Layer& layer) const {
        const TextFp8ActivationView input{activation.values, activation.scales};
        if (!project_packed(input, {
                {layer.mlp_gate.values, layer.mlp_gate.scales, work.mlp_gate, layer.mlp_gate.rows},
                {layer.mlp_up.values, layer.mlp_up.scales, work.mlp_up, layer.mlp_up.rows},
            })) {
            project(layer.mlp_gate, input, work.mlp_gate);
            project(layer.mlp_up, input, work.mlp_up);
        }
        if (tensor_core_fp8) {
            check(tensor_core_prefill_swiglu_quantize(work.mlp_gate, work.mlp_up,
                                                      intermediate_size, rows, activation, stream),
                  "Apply and quantize prefill SwiGLU");
            project(layer.mlp_down, input, work.hidden);
            return;
        }
        check(text_swiglu(work.mlp_gate, work.mlp_up, rows * intermediate_size,
                          work.mlp_gate, stream), "Apply prefill SwiGLU");
        project(layer.mlp_down, quantize(work.mlp_gate, intermediate_size), work.hidden);
    }

    void normalize(const Bf16* branch, Bf16* residual, const Bf16* weight) const {
        if (tensor_core_fp8) {
            check(tensor_core_prefill_rms_quantize(branch, residual, weight, hidden_size, rows,
                                                   work.normalized, activation, stream),
                  "Normalize and quantize prefill residual");
            return;
        }
        check(text_rms_norm_rows(branch, residual, weight, hidden_size, rows,
                                 work.normalized, residual, stream), "Normalize prefill residual");
        quantize(work.normalized, hidden_size);
    }
};

void gather_embeddings(const Bf16* embedding, const int* tokens, int rows,
                       Bf16* residual, cudaStream_t stream) {
    for (int offset = 0; offset < rows; offset += text_embedding_max_rows) {
        const int count = std::min(text_embedding_max_rows, rows - offset);
        check(text_embedding_rows(embedding, vocab_size, hidden_size, tokens + offset, count,
                                  residual + offset * hidden_size, stream),
              "Gather prefill embeddings");
    }
}

void stage_prefill_row(Request& request, const TileBuffers& work, int row, cudaStream_t stream) {
    const int offset = row * hidden_size;
    const auto bytes = hidden_size * sizeof(Bf16);
    check(cudaMemcpyAsync(request.work.hidden, work.hidden + offset, bytes,
                          cudaMemcpyDeviceToDevice, stream), "Stage final prefill hidden row");
    check(cudaMemcpyAsync(request.work.residual, work.residual + offset, bytes,
                          cudaMemcpyDeviceToDevice, stream), "Stage final prefill residual row");
}

void capture_prefill_scores(const DecoderModel& model, const PrefillGrant& grant,
                            const TileBuffers* buffers, int offset, cudaStream_t stream) {
    if (!grant.scores) return;
    const auto& capture = *grant.scores;
    const int first = std::max(0, capture.start - grant.position);
    const int end = capture.score_end(grant.position, grant.rows) - grant.position;
    for (int start = first; start < end; start += decoder_batch_max_lanes) {
        const int count = std::min(end - start, decoder_batch_max_lanes);
        for (int index = 0; index < count; ++index) {
            const int row = start + index;
            auto work = grant.request->work;
            if (buffers) {
                work.hidden = buffers->hidden + (offset + row) * hidden_size;
                work.residual = buffers->residual + (offset + row) * hidden_size;
            }
            capture.enqueue(model, work, grant.position + row, index, stream);
        }
        capture.collect(count, stream);
    }
}

constexpr int layerwise_row_alignment = 128;
constexpr int max_layerwise_rows = 1023;
static_assert(max_layerwise_rows / 256 <= max_gdn_prefill_segments);
static_assert(text_gqa_kv_width * sizeof(Bf16) >= text_gdn_value_heads * sizeof(float2));

struct PrefillGroup {
    int grants;
    int rows;
};

int aligned_layerwise_rows(int rows) {
    return (rows + layerwise_row_alignment - 1) / layerwise_row_alignment *
        layerwise_row_alignment;
}

PrefillGroup prefill_group(const PrefillGrant* grants, int count, bool mixed) {
    PrefillGroup group{0, 0};
    int aligned_rows = 0;
    for (int index = 0; index < count; ++index) {
        const auto& grant = grants[index];
        const int aligned = aligned_layerwise_rows(grant.rows);
        if (grant.rows < 256 || grant.rows != grant.tile_rows ||
            (!mixed && aligned != grant.rows) || aligned > max_layerwise_rows - aligned_rows) {
            break;
        }
        ++group.grants;
        group.rows += grant.rows;
        aligned_rows += aligned;
    }
    return group.grants > 1 ? group : PrefillGroup{1, 0};
}

template <typename Run>
void each_prefill_segment(const PrefillGrant* grants, int count, int offset, Run run) {
    for (int index = 0; index < count; ++index) {
        const auto& grant = grants[index];
        run(grant, offset);
        offset += grant.rows;
    }
}

template <typename Run>
void each_attention_group(const PrefillGrant* grants, int count, bool mixed, Run run) {
    int offset = 0;
    for (int index = 0; index < count;) {
        const auto group = prefill_group(grants + index, count - index, mixed);
        run(grants + index, group.grants, offset);
        offset += group.grants == 1 ? grants[index].rows : group.rows;
        index += group.grants;
    }
}

void read_paged_sequences(const PagedPrefillSequence* sequences, int count, cudaStream_t stream,
                          PairedPrefillScratch scratch) {
    if (const auto batched = try_batched_paged_prefill(sequences, count, stream)) {
        check(*batched, "Read batched causal prefill MXFP8 attention");
        return;
    }
    for (int index = 0; index < count; ++index) {
        const auto& sequence = sequences[index];
        check(tensor_core_model_prefill_gqa(sequence.cache, sequence.pages, sequence.position,
                                           sequence.rows, sequence.queries, sequence.output,
                                           stream, scratch), "Read causal prefill MXFP8 attention");
    }
}

Chunk prefill_attention_segment(const Chunk& chunk, const PrefillGrant& grant, int offset) {
    return Chunk{{{}, grant.rows, chunk.stream, true}, chunk.work.attention_segment(offset),
                 grant.position, nullptr};
}

void prefill_group_gqa(const Chunk& chunk, const GqaLayer& layer, int bank,
                       const PrefillGrant* grants, int count, int offset,
                       bool staged_batch = false) {
    static_assert(staged_prefill_batch_size >= max_layerwise_rows / 256);
    std::array<PagedPrefillSequence, staged_prefill_batch_size> sequences;
    int index = 0;
    each_prefill_segment(grants, count, offset, [&](const PrefillGrant& grant, int row) {
        const auto segment = prefill_attention_segment(chunk, grant, row);
        const auto cache = grant.request->caches[bank];
        const auto pages = grant.request->pages;
        segment.gqa_append(layer, cache, pages);
        sequences[index++] = {cache, pages, grant.position, grant.rows, segment.work.rope_queries,
                              segment.work.attention};
    });
    if (staged_batch) {
        const auto scratch = chunk.work.gqa_scratch();
        if (const auto staged = try_batched_staged_paged_prefill(sequences.data(), count,
                                                                chunk.stream, scratch)) {
            check(*staged, "Read staged batched causal prefill MXFP8 attention");
            return;
        }
        for (int lane = 0; lane < count; ++lane) {
            read_paged_sequences(sequences.data() + lane, 1, chunk.stream, scratch);
        }
        return;
    }
    read_paged_sequences(sequences.data(), count, chunk.stream,
                         count == 1 ? chunk.work.gqa_scratch() : PairedPrefillScratch{});
}

bool can_batch_staged_gqa(const PrefillGrant* grants, int count) {
    if (count != staged_prefill_batch_size) {
        return false;
    }
    const int position = grants[0].position;
    return position >= staged_prefill_batch_min_position &&
        position <= staged_prefill_batch_max_position && position % batch_prefill_tile_rows == 0 &&
        std::all_of(grants, grants + count, [&](const PrefillGrant& grant) {
            return grant.position == position && grant.rows == batch_prefill_tile_rows;
        });
}

void prefill_group_gdn(const Chunk& chunk, const GdnLayer& layer, int bank,
                       const PrefillGrant* grants, int count, int offset) {
    const auto& buffers = chunk.work;
    GdnPrefillSegment segments[max_gdn_prefill_segments]{};
    bool grouped_recurrence = count > 1;
    int index = 0;
    each_prefill_segment(grants, count, offset, [&](const PrefillGrant& grant, int row) {
        const auto segment = buffers.attention_segment(row);
        const TextFp8Activation fp8{
            buffers.activation.values + std::size_t(row) * text_gdn_value_width,
            buffers.activation.scales + std::size_t(row) * text_gdn_value_heads,
        };
        grouped_recurrence = grouped_recurrence &&
            grant.rows % layerwise_row_alignment == 0 && (grant.position != 0 || bank != 0);
        segments[index++] = {
            grant.rows, segment.gdn_qkv, segment.gdn_z, segment.gdn_b, segment.gdn_a,
            grant.request->states[bank],
            buffers.gdn_scratch + text_gdn_workspace_bytes(row) / sizeof(Bf16),
            reinterpret_cast<float2*>(segment.keys), fp8,
            buffers.mlp_gate + row * intermediate_size,
            std::size_t(grant.rows) * intermediate_size * sizeof(Bf16),
        };
    });
    const auto status = grouped_recurrence
        ? tensor_core_prefill_gdn_group(segments, count, layer.recurrent, chunk.stream)
        : std::nullopt;
    if (status) {
        check(*status, "Advance grouped prefill Gated DeltaNet");
        return;
    }
    index = 0;
    each_prefill_segment(grants, count, offset, [&](const PrefillGrant& grant, int row) {
        const auto& segment = segments[index++];
        prefill_attention_segment(chunk, grant, row).gdn_recur(
            layer, segment.state, bank, segment.output);
    });
}

void prefill_layerwise(const DecoderModel& model, const PrefillGrant* grants, PrefillGroup group,
                       PrefillWorkspace& workspace, bool mixed) {
    const auto stream = workspace.stream();
    ScratchLayout layout{static_cast<std::byte*>(workspace.data())};
    const TileBuffers buffers(layout, workspace.capacity());
    const bool grouped = group.rows > max_layerwise_rows;
    const Chunk chunk{{buffers.activation, group.rows, stream, true, grouped}, buffers, 0, nullptr};
    each_prefill_segment(grants, group.grants, 0, [&](const PrefillGrant& grant, int offset) {
        gather_embeddings(model.embedding, grant.tokens, grant.rows,
                          buffers.residual + offset * hidden_size, stream);
    });
    execute_layers(
        model.layers,
        [&](const Bf16* weight, const char*, const auto& residuals) {
            const auto [branch, residual] = residuals(buffers);
            chunk.normalize(branch, residual, weight);
        },
        [&](const GqaLayer& layer, int bank) {
            chunk.gqa_input(layer);
            if (grouped && can_batch_staged_gqa(grants, group.grants)) {
                prefill_group_gqa(chunk, layer, bank, grants, group.grants, 0, true);
            } else {
                each_attention_group(grants, group.grants, mixed,
                                     [&](const PrefillGrant* first, int count, int offset) {
                    prefill_group_gqa(chunk, layer, bank, first, count, offset);
                });
            }
            chunk.gqa_gate();
            chunk.attention_output(layer.output, text_gqa_q_width);
        },
        [&](const GdnLayer& layer, int bank) {
            chunk.gdn_input(layer);
            if (grouped) {
                prefill_group_gdn(chunk, layer, bank, grants, group.grants, 0);
            } else {
                each_attention_group(grants, group.grants, mixed,
                                     [&](const PrefillGrant* first, int count, int offset) {
                    prefill_group_gdn(chunk, layer, bank, first, count, offset);
                });
            }
            chunk.project(layer.output, {buffers.activation.values, buffers.activation.scales},
                          buffers.projected_attention);
        },
        [&](const Layer& layer) { chunk.mlp(layer); });
    each_prefill_segment(grants, group.grants, 0, [&](const PrefillGrant& grant, int offset) {
        if (grant.draft && grant.prompt_remaining > 1) {
            auto seed = chunk;
            seed.position = grant.position;
            seed.work.hidden += offset * hidden_size;
            seed.work.residual += offset * hidden_size;
            const bool wmma = model.backend.uses_b200_operators &&
                grant.position + grant.prompt_remaining >= wmma_seed_min_prompt_tokens;
            seed.seed_draft(*grant.draft, grant.tokens + 1,
                            std::min(grant.rows, grant.prompt_remaining - 1),
                            model.final_norm, wmma);
        }
        capture_prefill_scores(model, grant, &buffers, offset, stream);
        stage_prefill_row(*grant.request, buffers, offset + grant.rows - 1, stream);
    });
}

bool prefill_projection_tiles(const DecoderModel& model, const PrefillGrant* grants, int count,
                              int lanes, cudaStream_t stream,
                              DecoderBatchStepStats& stats) {
    const int rows = lanes * batch_prefill_tile_rows;
    PrefillWorkspace workspace(stream, rows);
    if (!workspace.data()) {
        return false;
    }
    stats.prefill_workspace_bytes = std::max(stats.prefill_workspace_bytes, workspace.bytes());
    for (int offset = 0; offset < grants[0].rows; offset += batch_prefill_tile_rows) {
        for (int index = 0; index < count; index += lanes) {
            std::array<PrefillGrant, grouped_prefill_max_rows / batch_prefill_tile_rows> group;
            for (int lane = 0; lane < lanes; ++lane) {
                group[lane] = prefill_tile(grants[index + lane], offset);
            }
            prefill_layerwise(model, group.data(), {lanes, rows}, workspace, false);
            ++stats.layerwise_prefill_groups;
            stats.layerwise_prefill_rows += rows;
        }
    }
    workspace.close();
    return true;
}

}

bool decoder::try_prefill_projection_pairs(
    const DecoderModel& model, const PrefillGrant* grants, int count, cudaStream_t stream,
    DecoderBatchStepStats& stats) {
    if (!model.backend.tensor_core_prefill || !pairable_prefill_grants(grants, count)) {
        return false;
    }
    if (count == 4 && prefill_projection_tiles(model, grants, count, 4, stream, stats)) {
        return true;
    }
    return prefill_projection_tiles(model, grants, count, 2, stream, stats);
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
            output, stream, rows > prefill_chunk_rows, grouped_projection);
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
    if (!stream || capacity < 1 || capacity > grouped_prefill_max_rows) {
        throw std::invalid_argument("Invalid decoder prefill workspace");
    }
    if (capacity == 1) {
        return;
    }
    int device = 0;
    check(cudaGetDevice(&device), "Get CUDA device for prefill workspace");
    reclaim_prefill_workspaces(device);
    const bool large = capacity > prefill_chunk_rows;
    auto& pools = prefill_pools();
    auto pool = pools.take_idle(device, large);
    if (!pool) {
        pool = std::make_unique<PrefillPool>(device, large);
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

void Decoder::Model::seed_draft(Workspace& work, MtpDraftRequest& draft, int position,
                                int shifted_token, cudaStream_t stream) const {
    normalize_final(work, stream);
    draft.enqueue(position, shifted_token, work.normalized, false, stream);
}

int PromptLogprobCapture::score_end(int position, int rows) const {
    return std::min(position + rows, static_cast<int>(prompt->size()) - 1);
}

void PromptLogprobCapture::enqueue(const DecoderModel& model, Workspace& work, int position,
                                  int index, cudaStream_t stream) const {
    model.compute_prefill_logits(work, stream);
    const int token = (*prompt)[position + 1];
    check(enqueue_prompt_logprobs(work.logits, vocab_size, token, top, workspace,
                                  readback + index, stream), "Score prompt token");
}

void PromptLogprobCapture::collect(int rows, cudaStream_t stream) const {
    if (!rows) return;
    check(cudaStreamSynchronize(stream), "Collect prompt token scores");
    for (int index = 0; index < rows; ++index) {
        const auto& row = readback[index];
        output->push_back(
            row.selected.token_id < 0 ? nullptr : collect_generated_logprobs(row, top));
    }
}

void decoder::finish_prompt_logprobs(const DecoderOptions& options, DecoderPromptLogprobs& scores,
                                     DecoderBatchStep& result) {
    result.sampling_failed = std::any_of(scores.begin(), scores.end(),
                                        [](const auto& row) { return !row; });
    if (!result.sampling_failed) {
        result.prompt_logprobs = std::make_shared<DecoderPromptLogprobs>(std::move(scores));
    }
    if (options.max_new_tokens == 0 || result.sampling_failed) {
        result.finish_reason = DecoderFinishReason::length;
    }
}

void Decoder::Model::prefill_chunk(Request& request, const int* tokens, int rows, int position,
                                   PrefillWorkspace& workspace, MtpDraftRequest* draft,
                                   int prompt_remaining, const MediaPrefill* media,
                                   const PromptLogprobCapture* scores, bool exact) const {
    const auto stream = workspace.stream();
    if (!stream || !tokens || rows < 1 || rows > workspace.capacity() || position < 0 ||
        position > text_attention_max_tokens - rows ||
        (position + rows - 1) / mxfp8_page_size >= request.pages.page_count ||
        (draft && (prompt_remaining < rows || media))) {
        throw std::invalid_argument("Invalid decoder prefill tile");
    }
    if (rows == 1 || !workspace.data()) {
        if (!std::all_of(tokens, tokens + rows,
                         [](int token) { return token >= 0 && token < vocab_size; })) {
            throw std::invalid_argument("Invalid decoder prefill token");
        }
        int pending = 0;
        const int first = scores ? scores->start : position;
        const int end = scores ? scores->score_end(position, rows) : position;
        for (int row = 0; row < rows; ++row) {
            const int token_position = position + row;
            ForwardLane lane{
                &request, tokens[row], token_position, TextGdnMode::prefill, false,
                media ? media->feature(token_position) : nullptr,
                media ? media->positions + token_position : nullptr};
            lane.exact = exact || media;
            forward(&lane, 1, stream);
            if (token_position >= first && token_position < end) {
                scores->enqueue(*this, request.work, token_position, pending++, stream);
                if (pending == decoder_batch_max_lanes) {
                    scores->collect(std::exchange(pending, 0), stream);
                }
            }
            if (draft && row + 1 < prompt_remaining) {
                seed_draft(request.work, *draft, token_position, tokens[row + 1], stream);
            }
        }
        if (scores) scores->collect(pending, stream);
        return;
    }
    ScratchLayout layout{static_cast<std::byte*>(workspace.data())};
    const TileBuffers buffers(layout, workspace.capacity());
    const bool tensor_core = backend.tensor_core_prefill && !exact && !media;
    const Chunk chunk{{buffers.activation, rows, stream, tensor_core}, buffers, position,
                      media ? media->positions + position : nullptr};
    const auto& work = chunk.work;
    gather_embeddings(embedding, tokens, rows, work.residual, stream);
    if (media) {
        media->substitute(work.residual, position, rows, stream);
    }
    const auto normalize = [&](const Bf16* weight, const char*, const auto& residuals) {
        const auto [branch, residual] = residuals(work);
        chunk.normalize(branch, residual, weight);
    };
    execute_layers(layers, normalize,
                   [&](const GqaLayer& layer, int bank) {
                       chunk.gqa_input(layer);
                       chunk.gqa_attention(layer, request.caches[bank], request.pages,
                                           work.gqa_scratch());
                       chunk.attention_output(layer.output, text_gqa_q_width);
                   },
                   [&](const GdnLayer& layer, int bank) {
                       chunk.gdn_input(layer);
                       chunk.gdn_attention(layer, request.states[bank], bank);
                   },
                   [&](const Layer& layer) { chunk.mlp(layer); });
    if (draft && prompt_remaining > 1) {
        const bool wmma = tensor_core && backend.target_split_end == text_attention_max_tokens &&
                          position + prompt_remaining >= wmma_seed_min_prompt_tokens;
        chunk.seed_draft(*draft, tokens + 1, std::min(rows, prompt_remaining - 1),
                         final_norm, wmma);
    }
    if (scores) {
        const PrefillGrant grant{&request, tokens, position, rows, rows, 0, nullptr, nullptr,
                                 scores};
        capture_prefill_scores(*this, grant, &work, 0, stream);
    }
    stage_prefill_row(request, work, rows - 1, stream);
}

void Decoder::Model::prefill_grants(const PrefillGrant* grants, int count, cudaStream_t stream,
                                   DecoderBatchStepStats& stats, bool prefix_cache,
                                   bool label_scores) const {
    int capacity = 1;
    for (int index = 0; index < count; ++index) {
        capacity = std::max(capacity, grants[index].tile_rows);
    }
    std::optional<PrefillWorkspace> workspace(std::in_place, stream, capacity);
    if (!workspace->data() && capacity > prefill_chunk_rows) {
        workspace.emplace(stream, prefill_chunk_rows);
    }
    stats.prefill_workspace_bytes = std::max(stats.prefill_workspace_bytes, workspace->bytes());
    for (int index = 0; index < count; ++index) {
        const auto& grant = grants[index];
        if (grant.rows == 1 && grant.draft && grant.prompt_remaining > 1) {
            seed_draft(grant.request->work, *grant.draft, grant.position, grant.tokens[1], stream);
        } else if (grant.rows == 1 && !label_scores) {
            capture_prefill_scores(*this, grant, nullptr, 0, stream);
        } else {
            for (int offset = 0; offset < grant.rows;) {
                const int rows = std::min({grant.tile_rows, workspace->capacity(),
                                           grant.rows - offset});
                prefill_chunk(*grant.request, grant.tokens + offset, rows,
                              grant.position + offset, *workspace, grant.draft,
                              grant.prompt_remaining - offset, grant.media, grant.scores,
                              label_scores || prefix_cache || grant.exact);
                offset += rows;
            }
        }
        if (label_scores && grant.rows == grant.prompt_remaining) {
            enqueue_label_logits(*grant.request, stream);
        }
    }
    workspace->close();
}

void PrefillProjections::project_seed_fc(
    Bf16Weight weight, const Bf16* input, Bf16* output) const {
    if (!wmma_draft_seed) {
        project(weight, input, output);
        return;
    }
    check(text_bf16_seed_fc(weight.values, input, weight.rows, weight.columns, rows, output,
                            stream), "Run MTP prefill seed FC");
}

}
