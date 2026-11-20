#include "sglang/decoder_verify.h"

#include "sglang/decoder_sampling.h"

#include <algorithm>
#include <cstdio>

namespace sglang::decoder {
namespace {

TextGdnReplayRows replay_rows(Bf16* storage, int count) {
    const int qkv = count * text_gdn_qkv_width;
    const int gates = count * text_gdn_value_heads;
    return {storage, storage + qkv, storage + 2 * qkv, storage + 2 * qkv + gates};
}

void free_pinned(void* results) {
    const auto status = cudaFreeHost(results);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "sglang decoder: Free verifier pinned storage: %s\n",
                     cudaGetErrorString(status));
    }
}

}

VerificationWorkspace::VerificationWorkspace(Allocations& memory, int rows) {
    const int layer_elements = 2 * rows * (text_gdn_qkv_width + text_gdn_value_heads);
    auto* storage = memory.allocate<Bf16>(gdn_layers * layer_elements);
    for (auto& layer : replay) {
        layer.inputs = replay_rows(storage, rows);
        storage += layer_elements;
    }
    auto* z = memory.allocate<Bf16>(rows * text_gdn_value_width);
    auto* attention = memory.allocate<Bf16>(rows * text_gdn_value_width);
    observable = memory.allocate<Bf16>(rows * text_gdn_value_width);
    winners = memory.allocate<TextLogitCandidate>(rows);
    work.reserve(rows);
    for (int index = 0; index < rows; ++index) {
        auto& row = work.emplace_back(memory, attention + index * text_gdn_value_width,
                                      winners + index);
        row.gdn_z = z + index * text_gdn_value_width;
    }
}

void VerificationWorkspace::bind_gdn(int layer, int rows) {
    const auto inputs = replay[layer].inputs;
    for (int index = 0; index < rows; ++index) {
        work[index].gdn_qkv = inputs.qkv + index * text_gdn_qkv_width;
        work[index].gdn_b = inputs.b + index * text_gdn_value_heads;
        work[index].gdn_a = inputs.a + index * text_gdn_value_heads;
    }
}

VerificationGraphScratch::VerificationGraphScratch(int device)
    : memory(device), workspace(memory, k1_verification_rows),
      device_binding(memory.allocate<VerificationGraphBinding>(1)) {
    void* pinned = nullptr;
    check(cudaMallocHost(&pinned, sizeof(VerificationGraphBinding)),
          "Allocate K1 verifier graph bindings");
    host_binding = std::shared_ptr<VerificationGraphBinding>(
        static_cast<VerificationGraphBinding*>(pinned), free_pinned);
}

void VerificationGraphScratch::upload(const VerificationSequence& sequence, cudaStream_t stream) {
    auto& binding = *host_binding;
    sequence.verification->bind_graph(binding);
    const int capacity = binding.pages.page_count * mxfp8_page_size;
    for (int row = 0; row < k1_verification_rows; ++row) {
        binding.inputs[row] = {0.0f, sequence.tokens[row]};
        binding.positions[row] = sequence.position + row;
        binding.scores[row] = workspace.work[row].gqa_scores(capacity);
    }
    check(cudaMemcpyAsync(device_binding, host_binding.get(), sizeof(binding),
                          cudaMemcpyHostToDevice, stream), "Upload K1 verifier graph bindings");
}

TargetVerification::TargetVerification(Request& request, int capacity, int prefix, int rows,
                                       bool logits, cudaStream_t stream)
    : request_(request), stream_(stream ? stream : request.memory.stream()),
      capacity_(capacity), valid_prefix_(prefix) {
    auto& memory = request.memory;
    auto& host_owner = memory.transfers().verification_results;
    const int reserved = std::max(request.kv_limit, request.pages.page_count * mxfp8_page_size);
    if (!stream_ || (memory.stream() && memory.stream() != stream_) ||
        memory.graph() || host_owner || rows < 1 ||
        rows > target_verify_max_rows || prefix < 0 || prefix > capacity ||
        capacity > reserved) {
        throw std::invalid_argument("Invalid target verification storage");
    }
    allocate_gdn(memory, tail_);
    eager_workspace_.emplace(memory, rows);

    const auto winner_bytes = rows * sizeof(TextLogitCandidate);
    const auto logits_bytes = logits ? rows * vocab_size * sizeof(Bf16) : 0;
    void* host = nullptr;
    check(cudaMallocHost(&host, winner_bytes + logits_bytes), "Allocate verifier results");
    host_owner = std::shared_ptr<void>(host, free_pinned);
    host_winners_ = static_cast<TextLogitCandidate*>(host);
    host_logits_ = logits ? reinterpret_cast<Bf16*>(host_winners_ + rows) : nullptr;
}

void TargetVerification::begin(int position, int count, cudaStream_t stream) {
    if (stream != stream_ || pending_rows_ != 0 ||
        position != valid_prefix_ || count < 1 ||
        count > static_cast<int>(eager_workspace_->work.size()) || count > capacity_ - position ||
        count > request_.pages.page_count * mxfp8_page_size - position) {
        throw std::invalid_argument("Invalid target verification interval");
    }
    pending_rows_ = count;
    active_workspace_ = &*eager_workspace_;
}

VerificationRow TargetVerification::row(int index) {
    return {&request_, &workspace().work.at(index), valid_prefix_ + index, this, index == 0};
}

bool TargetVerification::graph_eligible(const VerificationSequence& sequence) const {
    const int capacity = request_.pages.page_count * mxfp8_page_size;
    return sequence.count == k1_verification_rows &&
        sequence.position <= eager_workspace_->work.front().gqa_scores(capacity).capacity -
            k1_verification_rows &&
        std::all_of(sequence.tokens, sequence.tokens + k1_verification_rows,
                    [](int token) { return token >= 0 && token < vocab_size; });
}

void TargetVerification::bind_graph(VerificationGraphBinding& binding) const {
    binding.pages = request_.pages;
    binding.caches = request_.caches;
    binding.initial = request_.states;
    binding.tail = tail_;
}

void TargetVerification::use_graph_workspace(VerificationWorkspace& graph_workspace) {
    active_workspace_ = &graph_workspace;
}

void TargetVerification::bind_gdn(int layer) {
    workspace().bind_gdn(layer, pending_rows_);
}

void TargetVerification::advance_gdn(int layer, TextGdnWeights weights) {
    auto& scratch = workspace();
    auto& saved = scratch.replay[layer];
    saved.weights = weights;
    const auto& work = scratch.work.front();
    check(text_gdn_verify_tail(pending_rows_, saved.inputs, work.gdn_z, weights,
                               request_.states[layer], tail_[layer], scratch.observable,
                               work.attention, stream_),
          "Advance target GDN and retain its tail");
}

void TargetVerification::sample(const SamplingParameters& parameters, int first_output,
                                void* scratch) {
    for (int index = 0; index < pending_rows_; ++index) {
        const auto& work = workspace().work[index];
        check(sample_logits(work.logits, vocab_size, &work.argmax->token_id, parameters,
                            first_output + index, scratch, stream_),
              "Sample target verification logits");
    }
}

void TargetVerification::read() {
    auto* verifier = this;
    read_all(&verifier, 1, stream_);
}

void TargetVerification::read_all(TargetVerification* const* verifiers, int count,
                                   cudaStream_t stream) {
    for (int index = 0; index < count; ++index) {
        const auto* verifier = verifiers[index];
        if (!verifier || verifier->stream_ != stream ||
            verifier->pending_rows_ == 0 || verifier->ready_) {
            throw std::logic_error("No target verification results pending on this stream");
        }
    }
    for (int index = 0; index < count; ++index) {
        verifiers[index]->enqueue_results();
    }
    check(cudaStreamSynchronize(stream), "Complete target verification round");
    for (int index = 0; index < count; ++index) {
        verifiers[index]->ready_ = true;
    }
}

void TargetVerification::enqueue_results() {
    const auto& scratch = workspace();
    check(cudaMemcpyAsync(host_winners_, scratch.winners,
                          pending_rows_ * sizeof(TextLogitCandidate), cudaMemcpyDeviceToHost,
                          stream_), "Read target verification winners");
    if (host_logits_) {
        for (int index = 0; index < pending_rows_; ++index) {
            check(cudaMemcpyAsync(host_logits_ + index * vocab_size, scratch.work[index].logits,
                                  vocab_size * sizeof(Bf16), cudaMemcpyDeviceToHost, stream_),
                  "Read target verification logits");
        }
    }
}

void TargetVerification::commit(int count) {
    if (!ready_ || count < 0 || count > pending_rows_) {
        throw std::logic_error("Invalid target verification commit");
    }
    if (count > 0 && count < pending_rows_) {
        for (int layer = 0; layer < gdn_layers; ++layer) {
            const auto& saved = workspace().replay[layer];
            check(text_gdn_replay_prefix(count, saved.inputs, saved.weights,
                                         request_.states[layer], tail_[layer],
                                         stream_),
                  "Replay committed target GDN prefix");
        }
    }
    if (count > 0) {
        request_.states.swap(tail_);
        valid_prefix_ += count;
    }
    ready_ = false;
    pending_rows_ = 0;
    active_workspace_ = nullptr;
}

int TargetVerification::winner(int index, bool sampled) const {
    const int token = host_winners_[index].token_id;
    if (sampled && token == -1) {
        throw DecoderSamplingError();
    }
    if (token < 0 || token >= vocab_size) {
        throw std::runtime_error("Target verification winner falls outside vocabulary");
    }
    return token;
}

const Bf16* TargetVerification::logits(int index) const {
    return host_logits_ ? host_logits_ + index * vocab_size : nullptr;
}

}
