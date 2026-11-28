#pragma once

#include "sglang/decoder_media.h"
#include "sglang/runtime/backend/tp_ar_counters.h"
#include "sglang/runtime/server/batch_cache.h"
#include "sglang/runtime/server/batch_executor.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>

namespace sglang::server {

struct NativeTextStops;

struct BatchMetrics {
    DecoderPrefixCacheStats cache;
    int cache_capacity = 0;
    std::uint64_t prompt_tokens = 0;
    std::uint64_t generation_tokens = 0;
    std::uint64_t prefill_requests = 0;
    LmCacheMetrics lmcache;
    decoder::TpArCounters tp_ar;
    bool tp_ar_present = false;
    std::uint64_t resident_admission_retries = 0;
    std::uint64_t mailbox_grace_timeouts = 0;
    int admission_waiting_for_capacity = 0;
    int admission_waiting_for_lookup = 0;
    int admission_waiting_for_restore = 0;
};


struct BatchRuntimeState {
    bool ready;
    int running_requests;
    std::uint64_t generation_tokens;
};

struct BatchProfileStep {
    std::uint64_t elapsed_ns = 0;
    int width = 0;
    int prefill_rows = 0;
    int decode_lanes = 0;
    std::uint32_t graph_slots = 0;
    std::uint32_t verification_graph_slots = 0;
};

struct BatchProfile {
    std::string name;
    std::vector<BatchProfileStep> steps;
};

class BatchRequestId {
public:
    explicit BatchRequestId(std::string value);

    bool dispatch();
    bool abort(bool dispatched_only = false);
    bool claim_completion();
    bool finish();
    void finish_abort();
    bool finished() const;
    bool aborted() const;
    bool aborted_waiting() const;

    const std::string value;

private:
    enum class State {
        waiting, dispatched, finishing, finished, aborted_waiting, aborted_running,
        finished_waiting_abort, finished_running_abort,
    };
    std::atomic<State> state_{State::waiting};
};

struct BatchPrompt {
    std::vector<int> prompt;
    DecoderOptions options;
    BatchRequestId* request_id = nullptr;
    std::shared_ptr<const NativeTextStops> text_stops = nullptr;
    bool return_text_in_logprobs = false;
    bool no_stop_trim = false;
    bool skip_special_tokens = true;
    std::vector<DecoderMedia> media = {};
    std::size_t expanded_tokens = 0;

    std::size_t prompt_tokens() const;
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
    BatchService(std::unique_ptr<BatchExecutor> executor, bool trace, int mtp_steps = 0);
    BatchService(std::unique_ptr<BatchExecutor> executor, bool trace, int mtp_steps,
                 std::unique_ptr<decoder::LmCacheClient> cache, std::string cache_identity);
    BatchService(std::unique_ptr<BatchExecutor> executor, bool trace, int mtp_steps,
                 const decoder::LmCacheConfig& cache,
                 std::vector<std::array<std::uint8_t, 16>> rank_devices,
                 std::string cache_identity);
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
    void stop_cache_restores() noexcept;
    bool ready() const;
    bool wait_for_generation(std::chrono::steady_clock::time_point started,
                             std::chrono::steady_clock::time_point deadline,
                             const std::function<bool()>& cancelled);
    void rethrow_failure() const;
    BatchMetrics metrics() const;
    BatchRuntimeState runtime_state() const;
    bool reset_prefix_cache();
    bool start_profile(std::string name, std::size_t iterations);
    std::optional<BatchProfile> stop_profile();
    BatchTopology topology() const noexcept;
    void set_tp_ar_source(const Decoder* decoder);

private:
    struct State;
    std::unique_ptr<State> state_;
    const Decoder* tp_ar_source_ = nullptr;
};

}
