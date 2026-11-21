#pragma once

#include "sglang/decoder_batch.h"
#include "sglang/decoder_media.h"

#include <atomic>
#include <chrono>

namespace sglang::server {

struct NativeTextStops;

struct BatchMetrics {
    DecoderPrefixCacheStats cache;
    int cache_capacity = 0;
    std::uint64_t prompt_tokens = 0;
    std::uint64_t generation_tokens = 0;
    std::uint64_t prefill_requests = 0;
};

struct BatchProfileStep {
    std::uint64_t elapsed_ns = 0;
    int width = 0;
    int prefill_rows = 0;
    int decode_lanes = 0;
    std::uint32_t graph_slots = 0;
};

struct BatchProfile {
    std::string name;
    std::vector<BatchProfileStep> steps;
};

class BatchRequestId {
public:
    explicit BatchRequestId(std::string value) : value(std::move(value)) {}

    bool dispatch() {
        auto expected = State::waiting;
        return state_.compare_exchange_strong(expected, State::dispatched);
    }

    bool abort(bool dispatched_only = false) {
        auto expected = state_.load();
        while (expected == State::dispatched ||
               (!dispatched_only && expected == State::waiting)) {
            const auto target = expected == State::waiting ? State::aborted_waiting :
                                                             State::aborted_running;
            if (state_.compare_exchange_weak(expected, target)) {
                return true;
            }
        }
        return false;
    }

    bool finish() {
        auto expected = State::dispatched;
        return state_.compare_exchange_strong(expected, State::finished);
    }

    void finish_abort() {
        const auto state = state_.load();
        state_.store(state == State::aborted_waiting ? State::finished_waiting_abort :
                                                       State::finished_running_abort);
    }

    bool finished() const {
        const auto state = state_.load();
        return state == State::finished || state == State::finished_waiting_abort ||
               state == State::finished_running_abort;
    }
    bool aborted() const {
        const auto state = state_.load();
        return state == State::aborted_waiting || state == State::aborted_running ||
               state == State::finished_waiting_abort || state == State::finished_running_abort;
    }
    bool aborted_waiting() const {
        const auto state = state_.load();
        return state == State::aborted_waiting || state == State::finished_waiting_abort;
    }

    const std::string value;

private:
    enum class State {
        waiting, dispatched, aborted_waiting, aborted_running, finished,
        finished_waiting_abort, finished_running_abort,
    };
    std::atomic<State> state_{State::waiting};
};

struct BatchPrompt {
    std::vector<int> prompt;
    DecoderOptions options;
    BatchRequestId* request_id = nullptr;
    std::shared_ptr<const NativeTextStops> text_stops = nullptr;
};

struct BatchCallbacks {
    std::function<bool()> is_cancelled;
    std::function<bool()> on_admitted;
    std::function<bool()> on_stream_poll;
    std::function<bool(std::size_t, const DecoderResult&, bool)> on_output;
    bool buffer_decode_output = false;
};

class BatchService {
public:
    BatchService(const Decoder& decoder, bool trace, bool cuda_graphs = false, int mtp_steps = 0,
                 int max_active = decoder_batch_max_mtp_lanes, int prefix_cache_entries = 0,
                 int max_num_batched_tokens = 0);
    ~BatchService();

    DecoderResult generate(int slot, const std::vector<int>& prompt, const DecoderOptions& options,
                           const DecoderCallbacks& callbacks,
                           std::vector<DecoderMedia> media = std::vector<DecoderMedia>(),
                           std::size_t expanded_tokens = 0,
                           BatchRequestId* request_id = nullptr);
    std::optional<std::vector<DecoderResult>>
    generate_batch(int slot, const std::vector<BatchPrompt>& prompts,
                   const BatchCallbacks& callbacks);
    void notify_abort();
    bool ready() const;
    bool wait_for_generation(std::chrono::steady_clock::time_point started,
                             std::chrono::steady_clock::time_point deadline,
                             const std::function<bool()>& cancelled);
    void rethrow_failure() const;
    BatchMetrics metrics() const;
    bool reset_prefix_cache();
    bool start_profile(std::string name, std::size_t iterations);
    std::optional<BatchProfile> stop_profile();

private:
    struct State;
    std::unique_ptr<State> state_;
};

void trace_batch_step(int width, int decode, std::size_t request_bytes);

}
