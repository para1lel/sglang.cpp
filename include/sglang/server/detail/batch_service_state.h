#pragma once

#include "sglang/server/batch_service.h"
#include "sglang/server/detail/output_mailbox.h"
#include "sglang/runtime/execution/batch_admission_order.h"
#include "sglang/server/indexed_admission.h"
#include "sglang/server/text_stop.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

namespace sglang::server {
namespace batch_service_detail {

struct Job {
    std::vector<int> prompt;
    std::size_t prompt_tokens = 0;
    DecoderOptions options;
    std::vector<DecoderMedia> media;
    OutputMailbox pending;
    std::optional<DecoderFinishReason> finish;
    std::exception_ptr error;
    BatchRequestId* request_id = nullptr;
    std::unique_ptr<TokenTextStop> text_stop;
    std::optional<std::size_t> matched_text_stop;
    int cached_tokens = 0;
    bool admitted = false;
    bool cancelled = false;
    bool completion_claimed = false;
    bool prefill_completed = false;
    bool buffer_decode_output = false;
    bool output_started = false;
    bool output_overlapped = false;
    bool health_check = false;
    bool cache_reading = false;
    bool prefix_pending = false;
    bool retiring_prefix = false;
    BatchCache::Cancel cache_cancel;
    std::shared_ptr<BatchCacheRead> cache_read;
    std::vector<std::shared_ptr<decoder::PrefixCacheKeys>> cache_keys;
    std::shared_ptr<BatchRestoreJoin> restore_join;
    std::shared_ptr<BatchStorePairing> store_pairing;
    std::shared_future<bool> prefix_operation;
    std::optional<DecoderFinishReason> deferred_finish;

    ~Job() {
        if (restore_join) restore_join->fail();
    }

    bool aborted() const { return request_id && request_id->aborted(); }
    bool discard_deferred_store() const {
        return prefix_pending && !prefix_operation.valid() && (cancelled || aborted());
    }
    bool has_output() const { return !pending.empty() && !(output_overlapped && aborted()); }
    bool can_overlap_mtp_output() const {
        return options.mtp_steps > 0 && buffer_decode_output && output_started;
    }
    bool ready(bool buffered = false) const {
        return !prefix_pending && (cancelled || aborted() || (!cache_reading &&
            (buffered || can_overlap_mtp_output() ? !pending.full() : pending.empty())));
    }
    bool can_buffer(bool per_token = false) const {
        const bool eligible = options.mtp_steps > 0
            ? per_token && !options.requires_target_only() && !options.label_token_ids
            : prompt.size() >= decoder_batch_max_step_tokens;
        return eligible && media.empty() &&
            !options.samples() && !options.constraint && !options.eager && options.top_logits == 0;
    }
};

using Clock = std::chrono::steady_clock;
struct HealthProbe {
    std::shared_ptr<Job> job;
    Clock::time_point deadline;
    bool progressed = false;
};

using Jobs = std::array<std::shared_ptr<Job>, decoder_batch_max_lanes>;
struct Work {
    Jobs jobs;
    BatchAdmissionOrder admission;
    bool buffered;
};
using Selection = BatchSession::Selection;
using Steps = BatchSession::Steps;
struct BatchJob {
    std::shared_ptr<Job> job;
    std::size_t index = 0;
};
using BatchJobs = std::array<BatchJob, decoder_batch_max_lanes>;
inline constexpr auto disconnect_poll = std::chrono::milliseconds(10);
inline constexpr auto mailbox_grace = std::chrono::milliseconds(1);
inline constexpr int mailbox_min_lanes = 16;
inline constexpr int wide_decode_min_ready = 25;
}

using namespace batch_service_detail;

struct BatchService::State {
    State(std::unique_ptr<BatchExecutor> owner, bool trace, int mtp_steps,
          std::unique_ptr<decoder::LmCacheClient> client, std::string identity);
    State(std::unique_ptr<BatchExecutor> owner, bool trace, int mtp_steps,
          const decoder::LmCacheConfig& cache, std::vector<std::array<std::uint8_t, 16>> devices,
          std::string identity);
    ~State();
    void start();
    void process(BatchSession& batch);
    std::optional<std::uint64_t> begin_iteration(BatchSession& batch);
    std::optional<Work> wait_for_jobs(BatchSession& batch);
    std::uint32_t refresh_prefill_roster(Work& work, int capacity);
    bool idle_for_health() const;
    bool wide_decode_ready(bool buffered) const;
    bool decode_mailboxes_ready(bool require_all) const;
    bool prepare(BatchSession& batch, int slot, const std::shared_ptr<Job>& job, bool buffered);
    bool publish(int slot, Job& job, const DecoderBatchStep& step);
    void publish_steps(const Jobs& snapshot, const Selection& selected, Steps& steps);
    void begin_cache_read(int slot, const std::shared_ptr<Job>& job);
    void cancel_cache_read(Job& job);
    void configure_checkpoint(Job& job);
    bool prefix_ready(const Job& job) const {
        if (!job.prefix_pending) return false;
        if (!job.prefix_operation.valid()) {
            return job.discard_deferred_store() || (lmcache && lmcache->store_capacity());
        }
        return job.prefix_operation.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }
    bool poll_prefix(BatchSession& batch, int slot, const std::shared_ptr<Job>& job);
    void fail_jobs(std::exception_ptr error, bool permanent);
    void cancel(std::unique_lock<std::mutex>& lock, int slot, Job& job);
    DecoderResult consume(std::unique_lock<std::mutex>& lock, int slot, Job& job,
                          const DecoderCallbacks& callbacks);
    void cancel_batch(std::unique_lock<std::mutex>& lock, const BatchJobs& active);
    bool drain_batch(std::unique_lock<std::mutex>& lock, int owner, BatchJobs& active,
                     const BatchCallbacks& callbacks, std::vector<DecoderResult>& results,
                     std::size_t& completed);
    bool consume_batch(std::unique_lock<std::mutex>& lock, int owner,
                       const std::vector<BatchPrompt>& prompts, const BatchCallbacks& callbacks,
                       BatchJobs& active,
                       std::vector<DecoderResult>& results);
    void enqueue_batch(int owner, const std::vector<BatchPrompt>& prompts, BatchJobs& active,
                       std::size_t& next, bool buffered);
    bool drain_waiting_aborts(std::unique_lock<std::mutex>& lock, int owner,
                              const std::vector<BatchPrompt>& prompts,
                              const BatchCallbacks& callbacks, std::vector<DecoderResult>& results,
                              std::size_t& completed);

    const bool trace;
    const bool cuda_graphs;
    const int mtp_steps;
    const int max_active;
    const int prefix_cache_entries;
    const int max_num_batched_tokens;
    std::mutex mutex;
    std::condition_variable changed;
    Jobs jobs;
    BatchAdmissionOrder admission{max_active};
    Selection generating{};
    Selection reserved{};
    IndexedAdmission indexed_admission;
    Clock::time_point last_generation = Clock::time_point::min();
    std::optional<HealthProbe> health_probe;
    BatchMetrics counters;
    std::optional<BatchProfile> profile;
    std::uint64_t profile_epoch = 0;
    std::size_t profile_limit = 0;
    bool reset_pending = false;
    bool closing = false;
    std::exception_ptr failure;
    std::atomic_bool failed{false};
    std::unique_ptr<BatchExecutor> executor;
    std::unique_ptr<BatchCache> lmcache;
};

}
