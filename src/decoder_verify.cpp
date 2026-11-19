#include "sglang/decoder_verify.h"

#include <cstdio>

namespace sglang::decoder {
namespace {

TextGdnReplayRows replay_rows(Bf16* storage, int count) {
    const int qkv = count * text_gdn_qkv_width;
    const int gates = count * text_gdn_value_heads;
    return {storage, storage + qkv, storage + 2 * qkv, storage + 2 * qkv + gates};
}

void free_results(void* results) {
    const auto status = cudaFreeHost(results);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "sglang decoder: Free verifier results: %s\n",
                     cudaGetErrorString(status));
    }
}

}

TargetVerification::TargetVerification(Request& request, int capacity, int prefix, int rows,
                                       bool logits, cudaStream_t stream)
    : request_(request), stream_(stream ? stream : request.memory.stream()),
      capacity_(capacity), valid_prefix_(prefix) {
    auto& memory = request.memory;
    auto& host_owner = memory.transfers().verification_results;
    if (!stream_ || (memory.stream() && memory.stream() != stream_) ||
        memory.graph() || host_owner || rows < 1 ||
        rows > target_verify_max_rows || prefix < 0 || prefix > capacity ||
        capacity > request.pages.page_count * mxfp8_page_size) {
        throw std::invalid_argument("Invalid target verification storage");
    }
    allocate_gdn(memory, tail_);
    const int layer_elements = 2 * rows * (text_gdn_qkv_width + text_gdn_value_heads);
    auto* storage = memory.allocate<Bf16>(gdn_layers * layer_elements);
    for (auto& layer : replay_) {
        layer.inputs = replay_rows(storage, rows);
        storage += layer_elements;
    }
    auto* z = memory.allocate<Bf16>(rows * text_gdn_value_width);
    auto* attention = memory.allocate<Bf16>(rows * text_gdn_value_width);
    observable_ = memory.allocate<Bf16>(rows * text_gdn_value_width);
    device_winners_ = memory.allocate<TextLogitCandidate>(rows);
    work_.reserve(rows);
    for (int index = 0; index < rows; ++index) {
        auto& work = work_.emplace_back(memory, attention + index * text_gdn_value_width,
                                        device_winners_ + index);
        work.gdn_z = z + index * text_gdn_value_width;
    }

    const auto winner_bytes = rows * sizeof(TextLogitCandidate);
    const auto logits_bytes = logits ? rows * vocab_size * sizeof(Bf16) : 0;
    void* host = nullptr;
    check(cudaMallocHost(&host, winner_bytes + logits_bytes), "Allocate verifier results");
    host_owner = std::shared_ptr<void>(host, free_results);
    host_winners_ = static_cast<TextLogitCandidate*>(host);
    host_logits_ = logits ? reinterpret_cast<Bf16*>(host_winners_ + rows) : nullptr;
}

void TargetVerification::begin(int position, int count, cudaStream_t stream) {
    if (stream != stream_ || pending_rows_ != 0 ||
        position != valid_prefix_ || count < 1 ||
        count > static_cast<int>(work_.size()) || count > capacity_ - position) {
        throw std::invalid_argument("Invalid target verification interval");
    }
    pending_rows_ = count;
}

VerificationRow TargetVerification::row(int index) {
    return {&request_, &work_.at(index), valid_prefix_ + index, this, index == 0};
}

void TargetVerification::bind_gdn(int layer) {
    const auto inputs = replay_[layer].inputs;
    for (int index = 0; index < pending_rows_; ++index) {
        work_[index].gdn_qkv = inputs.qkv + index * text_gdn_qkv_width;
        work_[index].gdn_b = inputs.b + index * text_gdn_value_heads;
        work_[index].gdn_a = inputs.a + index * text_gdn_value_heads;
    }
}

void TargetVerification::advance_gdn(int layer, TextGdnWeights weights) {
    auto& saved = replay_[layer];
    saved.weights = weights;
    const auto& work = work_.front();
    check(text_gdn_verify_tail(pending_rows_, saved.inputs, work.gdn_z, weights,
                               request_.states[layer], tail_[layer], observable_,
                               work.attention, stream_),
          "Advance target GDN and retain its tail");
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
    check(cudaMemcpyAsync(host_winners_, device_winners_,
                          pending_rows_ * sizeof(TextLogitCandidate), cudaMemcpyDeviceToHost,
                          stream_), "Read target verification winners");
    if (host_logits_) {
        for (int index = 0; index < pending_rows_; ++index) {
            check(cudaMemcpyAsync(host_logits_ + index * vocab_size, work_[index].logits,
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
            const auto& saved = replay_[layer];
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
}

int TargetVerification::winner(int index) const {
    const int token = host_winners_[index].token_id;
    if (token < 0 || token >= vocab_size) {
        throw std::runtime_error("Target verification winner falls outside vocabulary");
    }
    return token;
}

const Bf16* TargetVerification::logits(int index) const {
    return host_logits_ ? host_logits_ + index * vocab_size : nullptr;
}

}
