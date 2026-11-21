#include "sglang/decoder_internal.h"

#include "sglang/decoder_cuda.cuh"
#include "sglang/prefill_fp8.h"
#include "sglang/text/gqa_decode_step.h"
#include "sglang/text_ops.h"

#include <algorithm>
#include <cstdio>
#include <utility>

namespace sglang::decoder {
namespace {

constexpr int packed_capacity = packed_decode_max_rows;
static_assert(packed_capacity <= gqa_decode_step_max_lanes);
static_assert(vocab_size * sizeof(Bf16) >= paged_decode_scratch_bytes);
static_assert(vocab_size * sizeof(Bf16) % 16 == 0);

struct PackedScratchOom : CudaError {
    explicit PackedScratchOom(const CudaError& error) : CudaError(error) {}
};

template <typename T>
T* row_of(T* base, int width, int row) {
    return base + static_cast<std::size_t>(width) * row;
}

bool packable(const ForwardLane& lane) {
    return lane.packed && lane.mode == TextGdnMode::decode && !lane.feature && !lane.mrope &&
        lane.rotary_position < 0 && lane.position >= 0 && lane.position < text_attention_max_tokens;
}

TextFp8DecodeTarget target(Fp8Weight weight, Bf16* output) {
    return {weight.values, weight.scales, weight.rows, output};
}

int bucket_for(int active) {
    for (int index = 0; index < static_cast<int>(packed_decode_buckets.size()); ++index) {
        const int rows = packed_decode_buckets[index];
        if (rows >= active && rows <= packed_capacity) {
            return index;
        }
    }
    return -1;
}

}

class PackedDecode {
public:
    struct Lane {
        int position;
        Mxfp8PageTable pages;
        std::array<Mxfp8KvCache, full_attention_layers> caches;
    };

    struct Staging {
        int active;
        std::array<int, packed_capacity> tokens;
        std::array<Lane, packed_capacity> lanes;
        std::array<std::array<TextGdnState, packed_capacity>, gdn_layers> states;
    };

    struct Plan {
        int active = 0;
        int bucket = 0;
        int rows = 0;
        std::array<int, packed_capacity> lane{};
    };

    explicit PackedDecode(const DecoderModel& model);

    static bool select(const ForwardLane* lanes, int count, Plan& plan);
    void bind(const ForwardLane* lanes, const Plan& plan);
    void upload(cudaStream_t stream) const;
    void enqueue(int rows, cudaStream_t stream) const;
    void read_winners(TextLogitCandidate* winners, int active, cudaStream_t stream) const;
    Workspace* view(int row) { return &views_[row]; }

    std::array<bool, packed_decode_buckets.size()> warmed{};

private:
    struct Rows {
        Bf16* hidden;
        Bf16* residual;
        Bf16* normalized;
        Bf16* projected_attention;
        Bf16* attention;
        Bf16* q_gate;
        Bf16* keys;
        Bf16* values;
        Bf16* rope_queries;
        Bf16* rope_keys;
        Bf16* gdn_qkv;
        Bf16* gdn_z;
        Bf16* gdn_b;
        Bf16* gdn_a;
        Bf16* convolved;
        Bf16* observable;
        Bf16* mlp_gate;
        Bf16* mlp_up;
        Bf16* logits;
        TextFp8Activation activation;
        TextLogitCandidate* argmax_scratch;
        TextLogitCandidate* argmax;
    };

    struct Step {
        int rows;
        cudaStream_t stream;
    };

    template <typename Element>
    Element* allocate_scratch(std::size_t count);
    Bf16* allocate_rows(int width);
    TextFp8ActivationView quantize(const Bf16* input, int width, const Step& step) const;
    void normalize(const Bf16* branch, Bf16* residual, const Bf16* weight,
                   const Step& step) const;
    void project(Fp8Weight weight, TextFp8ActivationView input, Bf16* output,
                 const Step& step) const;
    void gdn(const GdnLayer& layer, int bank, const Step& step) const;
    void gqa(const GqaLayer& layer, int bank, const Step& step) const;
    void attend(const GqaLayer& layer, int bank, const Step& step) const;
    void mlp(const Layer& layer, const Step& step) const;
    void head(const Step& step) const;
    void fill_pad(int row);

    const DecoderModel& model_;
    Allocations memory_;
    Rows rows_{};
    Staging* device_ = nullptr;
    std::unique_ptr<Staging> host_;
    std::array<Workspace, packed_capacity> views_{};
};

template <typename Element>
Element* PackedDecode::allocate_scratch(std::size_t count) {
    try {
        return memory_.allocate<Element>(count);
    } catch (const CudaError& error) {
        if (error.status != cudaErrorMemoryAllocation) {
            throw;
        }
        try {
            const auto pending = cudaGetLastError();
            if (pending != cudaErrorMemoryAllocation) {
                check(pending, "Check CUDA error after packed scratch allocation failed");
            }
            memory_.close();
        } catch (...) {
            std::fprintf(stderr, "sglang decoder: packed scratch allocation failed: %s\n",
                         error.what());
            throw;
        }
        throw PackedScratchOom(error);
    }
}

Bf16* PackedDecode::allocate_rows(int width) {
    const auto count = static_cast<std::size_t>(width) * packed_capacity;
    auto* rows = allocate_scratch<Bf16>(count);
    check(cudaMemset(rows, 0, count * sizeof(Bf16)), "Clear packed decode rows");
    return rows;
}

PackedDecode::PackedDecode(const DecoderModel& model)
    : model_(model), memory_(model.device), host_(std::make_unique<Staging>()) {
    rows_.hidden = allocate_rows(hidden_size);
    rows_.residual = allocate_rows(hidden_size);
    rows_.normalized = allocate_rows(hidden_size);
    rows_.projected_attention = allocate_rows(hidden_size);
    rows_.attention = allocate_rows(text_gqa_q_width);
    rows_.q_gate = allocate_rows(text_gqa_q_gate_width);
    rows_.keys = allocate_rows(text_gqa_kv_width);
    rows_.values = allocate_rows(text_gqa_kv_width);
    rows_.rope_queries = allocate_rows(text_gqa_q_width);
    rows_.rope_keys = allocate_rows(text_gqa_kv_width);
    rows_.gdn_qkv = allocate_rows(text_gdn_qkv_width);
    rows_.gdn_z = allocate_rows(text_gdn_value_width);
    rows_.gdn_b = allocate_rows(text_gdn_value_heads);
    rows_.gdn_a = allocate_rows(text_gdn_value_heads);
    rows_.convolved = allocate_rows(text_gdn_qkv_width);
    rows_.observable = allocate_rows(text_gdn_value_width);
    rows_.mlp_gate = allocate_rows(intermediate_size);
    rows_.mlp_up = allocate_rows(intermediate_size);
    rows_.logits = allocate_rows(vocab_size);
    rows_.activation = {
        allocate_scratch<std::uint8_t>(std::size_t{intermediate_size} * packed_capacity),
        allocate_scratch<float>(std::size_t{text_fp8_group_count(intermediate_size)} *
                                packed_capacity)};
    constexpr int scratch = text_argmax_scratch_count(vocab_size);
    rows_.argmax_scratch =
        allocate_scratch<TextLogitCandidate>(std::size_t{scratch} * packed_capacity);
    rows_.argmax = allocate_scratch<TextLogitCandidate>(packed_capacity);
    device_ = allocate_scratch<Staging>(1);
    for (int row = 0; row < packed_capacity; ++row) {
        auto& view = views_[row];
        view.logits = row_of(rows_.logits, vocab_size, row);
        view.argmax_scratch = row_of(rows_.argmax_scratch, scratch, row);
        view.argmax = rows_.argmax + row;
    }
    check(cudaDeviceSynchronize(), "Prepare packed decode rows");
}

void PackedDecode::fill_pad(int row) {
    auto& staging = *host_;
    staging.tokens[row] = 0;
    staging.lanes[row] = {-1, {}, {}};
    for (auto& layer : staging.states) {
        layer[row] = {};
    }
}

bool PackedDecode::select(const ForwardLane* lanes, int count, Plan& plan) {
    plan = {};
    for (int index = 0; index < count && plan.active < packed_capacity; ++index) {
        if (packable(lanes[index])) {
            plan.lane[plan.active++] = index;
        }
    }
    if (plan.active < packed_decode_min_lanes) {
        return false;
    }
    plan.bucket = bucket_for(plan.active);
    if (plan.bucket < 0) {
        return false;
    }
    plan.rows = packed_decode_buckets[plan.bucket];
    return true;
}

void PackedDecode::bind(const ForwardLane* lanes, const Plan& plan) {
    auto& staging = *host_;
    staging.active = plan.active;
    for (int row = 0; row < plan.active; ++row) {
        const auto& lane = lanes[plan.lane[row]];
        const auto& request = *lane.request;
        staging.tokens[row] = lane.token_id;
        staging.lanes[row] = {lane.position, request.pages, request.caches};
        for (int layer = 0; layer < gdn_layers; ++layer) {
            staging.states[layer][row] = request.states[layer];
        }
    }
    for (int row = plan.active; row < plan.rows; ++row) {
        fill_pad(row);
    }
}

void PackedDecode::upload(cudaStream_t stream) const {
    check(cudaMemcpyAsync(device_, host_.get(), sizeof(Staging), cudaMemcpyHostToDevice, stream),
          "Upload packed decode bindings");
}

TextFp8ActivationView PackedDecode::quantize(const Bf16* input, int width,
                                             const Step& step) const {
    check(text_quantize_fp8_rows(input, width, step.rows, rows_.activation, step.stream),
          "Quantize packed decode projection input");
    return {rows_.activation.values, rows_.activation.scales};
}

void PackedDecode::normalize(const Bf16* branch, Bf16* residual, const Bf16* weight,
                             const Step& step) const {
    check(tensor_core_prefill_rms_quantize(branch, residual, weight, hidden_size, step.rows,
                                           rows_.normalized, rows_.activation, step.stream),
          "Normalize and quantize packed decode residual");
}

void PackedDecode::project(Fp8Weight weight, TextFp8ActivationView input, Bf16* output,
                           const Step& step) const {
    check(text_fp8_decode_rows(weight.values, weight.scales, input, weight.rows, weight.columns,
                               step.rows, output, step.stream), "Run packed decode projection");
}

void PackedDecode::gdn(const GdnLayer& layer, int bank, const Step& step) const {
    const TextFp8ActivationView input{rows_.activation.values, rows_.activation.scales};
    const TextFp8DecodeTarget targets[]{target(layer.qkv, rows_.gdn_qkv),
                                        target(layer.z, rows_.gdn_z)};
    check(text_fp8_decode_rows_fused(targets, 2, input, hidden_size, step.rows, step.stream),
          "Run packed Gated DeltaNet input projections");
    check(text_bf16_decode_gates(layer.b.values, layer.a.values, rows_.normalized, step.rows,
                                 rows_.gdn_b, rows_.gdn_a, step.stream),
          "Run packed Gated DeltaNet gate projections");
    const PackedGdnRows rows{rows_.gdn_qkv, rows_.gdn_z, rows_.gdn_b, rows_.gdn_a,
                             rows_.convolved, rows_.observable, rows_.attention};
    check(packed_decode_gdn(step.rows, &device_->active, rows, layer.recurrent,
                            device_->states[bank].data(), step.stream),
          "Advance packed Gated DeltaNet state");
    project(layer.output, quantize(rows_.attention, text_gdn_value_width, step),
            rows_.projected_attention, step);
}

void PackedDecode::attend(const GqaLayer& layer, int bank, const Step& step) const {
    std::array<GqaDecodeStepLane, packed_capacity> reads;
    for (int row = 0; row < step.rows; ++row) {
        const auto* lane = &device_->lanes[row];
        auto* queries = row_of(rows_.rope_queries, text_gqa_q_width, row);
        const PagedDecodeBuffers buffers{queries,
                                         row_of(rows_.attention, text_gqa_q_width, row),
                                         row_of(rows_.logits, vocab_size, row)};
        reads[row] = {{&lane->caches[bank], &lane->pages, &lane->position, nullptr, buffers},
                      queries, row_of(rows_.q_gate, text_gqa_q_gate_width, row),
                      row_of(rows_.keys, text_gqa_kv_width, row),
                      row_of(rows_.values, text_gqa_kv_width, row),
                      row_of(rows_.rope_keys, text_gqa_kv_width, row)};
    }
    check(decode_gqa_step(reads.data(), step.rows, layer.query_norm, layer.key_norm, step.stream),
          "Run packed grouped-query attention step");
}

void PackedDecode::gqa(const GqaLayer& layer, int bank, const Step& step) const {
    const TextFp8ActivationView input{rows_.activation.values, rows_.activation.scales};
    const TextFp8DecodeTarget targets[]{target(layer.q_gate, rows_.q_gate),
                                        target(layer.key, rows_.keys),
                                        target(layer.value, rows_.values)};
    check(text_fp8_decode_rows_fused(targets, 3, input, hidden_size, step.rows, step.stream),
          "Run packed QKV projections");
    attend(layer, bank, step);
    project(layer.output, quantize(rows_.attention, text_gqa_q_width, step),
            rows_.projected_attention, step);
}

void PackedDecode::mlp(const Layer& layer, const Step& step) const {
    const TextFp8ActivationView input{rows_.activation.values, rows_.activation.scales};
    const TextFp8DecodeTarget targets[]{target(layer.mlp_gate, rows_.mlp_gate),
                                        target(layer.mlp_up, rows_.mlp_up)};
    check(text_fp8_decode_rows_fused(targets, 2, input, hidden_size, step.rows, step.stream),
          "Run packed gate and up projections");
    check(tensor_core_prefill_swiglu_quantize(rows_.mlp_gate, rows_.mlp_up, intermediate_size,
                                              step.rows, rows_.activation, step.stream),
          "Apply and quantize packed SwiGLU");
    project(layer.mlp_down, input, rows_.hidden, step);
}

void PackedDecode::head(const Step& step) const {
    const auto& head = model_.lm_head;
    check(text_rms_norm_rows(rows_.hidden, rows_.residual, model_.final_norm, hidden_size,
                             step.rows, rows_.normalized, nullptr, step.stream),
          "Normalize packed final residual");
    check(text_bf16_decode_rows(head.values, rows_.normalized, head.rows, head.columns, step.rows,
                                rows_.logits, step.stream), "Run packed BF16 LM head");
    check(packed_decode_argmax(rows_.logits, vocab_size, step.rows, &device_->active,
                               rows_.argmax_scratch, rows_.argmax, step.stream),
          "Select packed greedy tokens");
}

void PackedDecode::enqueue(int rows, cudaStream_t stream) const {
    const Step step{rows, stream};
    check(packed_decode_embedding(model_.embedding, vocab_size, hidden_size,
                                  device_->tokens.data(), rows, rows_.residual, stream),
          "Look up packed token embeddings");
    const auto normalize_step = [&](const Bf16* weight, const char*, const auto& residuals) {
        const auto [branch, residual] = residuals(rows_);
        normalize(branch, residual, weight, step);
    };
    execute_layers(model_.layers, normalize_step,
                   [&](const GqaLayer& layer, int bank) { gqa(layer, bank, step); },
                   [&](const GdnLayer& layer, int bank) { gdn(layer, bank, step); },
                   [&](const Layer& layer) { mlp(layer, step); });
    head(step);
}

void PackedDecode::read_winners(TextLogitCandidate* winners, int active,
                                cudaStream_t stream) const {
    check(cudaMemcpyAsync(winners, rows_.argmax, active * sizeof(TextLogitCandidate),
                          cudaMemcpyDeviceToHost, stream), "Read packed greedy winners");
}

DecodeGraphs::DecodeGraphs() = default;
DecodeGraphs::~DecodeGraphs() = default;
DecodeGraphs::DecodeGraphs(DecodeGraphs&&) noexcept = default;
DecodeGraphs& DecodeGraphs::operator=(DecodeGraphs&&) noexcept = default;

namespace {

bool replay_narrow(const DecoderModel& model, Allocations& execution, DecodeGraphs& graphs,
                   const ForwardLane* lanes, int count) {
    if (count == 1 && lanes[0].position >=
        lanes[0].request->work.gqa_scores(text_attention_max_tokens).capacity) {
        return false;
    }
    auto& warmed = graphs.narrow_warmed[count - 1];
    if (!std::exchange(warmed, true)) {
        return false;
    }
    return execution.replay_batch_graph(count, [&] {
        if (!graphs.narrow) {
            graphs.narrow = std::make_unique<BatchGraphScratch>(model.device);
        }
    }, [&] {
        auto& scratch = *graphs.narrow;
        std::array<BatchGraphLane, decoder_batch_max_graph_lanes> capture;
        for (int index = 0; index < count; ++index) {
            capture[index] = {&*scratch.work[index], scratch.device_bindings + index};
        }
        model.capture_batch_decode(capture.data(), count, execution.stream());
    }, [&] {
        auto& scratch = *graphs.narrow;
        for (int index = 0; index < count; ++index) {
            const auto& lane = lanes[index];
            auto& request = *lane.request;
            auto scores = scratch.work[index]->gqa_scores(text_attention_max_tokens);
            if (paged_decode_position(lane.position)) {
                scores = {};
            } else if (lane.position >= scores.capacity) {
                scores = request.batch_graph_scores();
                if (lane.position >= scores.capacity) {
                    return false;
                }
            }
            scratch.bindings[index] = {{0.0f, lane.token_id}, lane.position, scores,
                                       request.pages, request.caches, request.states};
        }
        check(cudaMemcpyAsync(scratch.device_bindings, scratch.bindings.data(),
                              count * sizeof(BatchDecodeBinding), cudaMemcpyHostToDevice,
                              execution.stream()), "Upload batch decode bindings");
        return true;
    });
}

DecodeRun run_narrow(const DecoderModel& model, Allocations& execution, DecodeGraphs& graphs,
                     ForwardLane* lanes, int count, DecodeRunOptions options) {
    DecodeRun run;
    const auto stream = execution.stream();
    const bool graph = options.cuda_graphs && options.graph_step &&
        count <= decoder_batch_max_graph_lanes && replay_narrow(model, execution, graphs, lanes,
                                                                count);
    if (graph) {
        for (int index = 0; index < count; ++index) {
            run.ready[index] = &*graphs.narrow->work[index];
            run.graphed |= std::uint64_t{1} << index;
        }
        return run;
    }
    lanes[0].narrow_gqa = count == 1 && options.narrow_allowed && lanes[0].rotary_position == -1;
    model.forward(lanes, count, stream);
    if (options.graph_step && model.compute_eager_decode_logits(lanes, count, stream)) {
        for (int index = 0; index < count; ++index) {
            run.ready[index] = &lanes[index].request->work;
        }
    }
    return run;
}

PackedDecode* packed_decode(const DecoderModel& model, DecodeGraphs& graphs) {
    if (!graphs.packed && !graphs.packed_unavailable) {
        try {
            graphs.packed = std::make_unique<PackedDecode>(model);
        } catch (const PackedScratchOom& error) {
            graphs.packed_unavailable = true;
            std::fprintf(stderr, "sglang decoder: packed decode unavailable: %s\n", error.what());
        }
    }
    return graphs.packed.get();
}

bool run_packed(const DecoderModel& model, Allocations& execution, DecodeGraphs& graphs,
                ForwardLane* lanes, int count, DecodeRunOptions options, DecodeRun& run) {
    PackedDecode::Plan plan;
    if (!PackedDecode::select(lanes, count, plan)) {
        return false;
    }
    auto* packed = packed_decode(model, graphs);
    if (!packed) {
        return false;
    }
    const bool read_winners = options.graph_step &&
        std::all_of(plan.lane.begin(), plan.lane.begin() + plan.active, [&](int index) {
            return (options.greedy_winners >> index & 1) != 0;
        });
    auto* winners = read_winners ? execution.prepare_packed_winners() : nullptr;
    packed->bind(lanes, plan);
    const auto stream = execution.stream();
    const bool graph = options.cuda_graphs && options.graph_step &&
        std::exchange(packed->warmed[plan.bucket], true);
    const auto capture = [&] { packed->enqueue(plan.rows, stream); };
    const auto bind = [&] {
        packed->upload(stream);
        return true;
    };
    const bool replayed =
        graph && execution.replay_packed_graph(plan.bucket, plan.rows, [] {}, capture, bind);
    if (!replayed) {
        packed->upload(stream);
        packed->enqueue(plan.rows, stream);
    }
    if (winners) {
        packed->read_winners(winners, plan.active, stream);
    }
    std::array<bool, decoder_batch_max_lanes> claimed{};
    for (int row = 0; row < plan.active; ++row) {
        const int index = plan.lane[row];
        claimed[index] = true;
        run.ready[index] = packed->view(row);
        if (winners) {
            run.winners[index] = winners + row;
        }
        if (replayed) {
            run.graphed |= std::uint64_t{1} << index;
        }
    }
    std::array<ForwardLane, decoder_batch_max_lanes> rest;
    int remaining = 0;
    for (int index = 0; index < count; ++index) {
        if (!claimed[index]) {
            rest[remaining++] = lanes[index];
        }
    }
    if (remaining != 0) {
        rest[0].narrow_gqa = remaining == 1 && options.narrow_allowed &&
            rest[0].rotary_position == -1;
        model.forward(rest.data(), remaining, stream);
    }
    return true;
}

}

DecodeRun run_decode_lanes(const DecoderModel& model, Allocations& execution,
                           DecodeGraphs& graphs, ForwardLane* lanes, int count,
                           DecodeRunOptions options) {
    DecodeRun run;
    if (count == 0) {
        return run;
    }
    if (options.packed && count >= packed_decode_min_lanes &&
        run_packed(model, execution, graphs, lanes, count, options, run)) {
        return run;
    }
    return run_narrow(model, execution, graphs, lanes, count, options);
}

}
