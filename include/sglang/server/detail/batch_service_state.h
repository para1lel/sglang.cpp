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
    bool output_grace_expired = false;
    bool health_check = false;
    bool cache_reading = false;
    bool prefix_pending = false;
    bool retiring_prefix = false;
    BatchCache::Cancel cache_cancel;
    std::shared_ptr<BatchCacheRead> cache_read;
    std::vector<std::shared_ptr<decoder::PrefixCacheKeys>> cache_keys;
    std::shared_ptr<BatchRestoreJoin> restore_join;
    std::shared_future<bool> prefix_operation;
    std::optional<DecoderFinishReason> deferred_finish;
    std::chrono::steady_clock::time_point admission_retry_at;

    ~Job();

    bool aborted() const;
    bool discard_deferred_store() const;
    bool has_output() const;
    bool awaiting_capacity() const;
    bool can_overlap_mtp_output() const;
    bool ready(bool buffered = false, bool capacity_blocked = false) const;
    bool can_buffer(bool per_token = false) const;
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
    bool has_runnable_job(Clock::time_point& retry_at) const;
    bool decode_cohort_ready(bool buffered, bool require_all = false) const;
    bool prepare(BatchSession& batch, int slot, const std::shared_ptr<Job>& job, bool buffered,
                 bool capacity_blocked);
    bool admit(BatchSession& batch, int slot, Job& job);
    bool defer_capacity_retry(BatchSession& batch, Job& job, std::exception_ptr error);
    bool publish(int slot, Job& job, const DecoderBatchStep& step);
    void publish_steps(const Jobs& snapshot, const Selection& selected, Steps& steps);
    void begin_cache_read(int slot, const std::shared_ptr<Job>& job);
    void cancel_cache_read(Job& job);
    void configure_checkpoint(Job& job);
    bool prefix_ready(const Job& job) const;
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
    const bool resident_prefix_cache;
    const std::string cache_identity;
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
    BatchSession* session = nullptr;
    std::optional<BatchProfile> profile;
    std::uint64_t profile_epoch = 0;
    std::size_t profile_limit = 0;
    std::shared_ptr<std::promise<void>> reset_pending;
    bool closing = false;
    std::exception_ptr failure;
    std::atomic_bool failed{false};
    std::unique_ptr<BatchExecutor> executor;
    std::unique_ptr<BatchCache> lmcache;
};

}
