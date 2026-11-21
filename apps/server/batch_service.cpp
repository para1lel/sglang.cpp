#include "sglang/server/batch_service.h"

#include "sglang/decoder_batch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>

namespace sglang::server {
namespace {

struct Job {
    std::vector<int> prompt;
    std::size_t prompt_tokens = 0;
    DecoderOptions options;
    std::vector<DecoderMedia> media;
    std::optional<DecoderBatchStep> pending;
    std::optional<DecoderFinishReason> finish;
    std::exception_ptr error;
    int cached_tokens = 0;
    bool admitted = false;
    bool cancelled = false;
    bool prefill_completed = false;

    bool ready() const { return cancelled || !pending; }
};

using Jobs = std::array<std::shared_ptr<Job>, decoder_batch_max_lanes>;
using Selection = std::array<bool, decoder_batch_max_lanes>;
using Steps = std::array<DecoderBatchStep, decoder_batch_max_lanes>;
struct BatchJob {
    std::shared_ptr<Job> job;
    std::size_t index = 0;
};
using BatchJobs = std::array<BatchJob, decoder_batch_max_lanes>;
constexpr auto disconnect_poll = std::chrono::milliseconds(10);
constexpr auto mailbox_grace = std::chrono::milliseconds(1);

void trace_prefill_rows(const DecoderBatchStepStats& stats, int max_active) {
    std::string line = "batch prefill_rows=";
    for (int slot = 0; slot < std::max(max_active, decoder_batch_max_mtp_lanes); ++slot) {
        if (slot != 0) {
            line += ',';
        }
        line += std::to_string(stats.prefill_rows[slot]);
    }
    std::cerr << (line + " workspace_bytes=" + std::to_string(stats.prefill_workspace_bytes) +
                  '\n');
}

void trace_verification(const Steps& steps, const DecoderBatchStepStats& stats) {
    std::cerr << ("batch verify lanes=" + std::to_string(stats.verification_lanes) +
                  " rows=" + std::to_string(stats.verification_rows) +
                  " peak_request_bytes=" + std::to_string(stats.peak_request_bytes) + '\n');
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        const auto& step = steps[slot];
        if (step.mtp_verified_rows || step.mtp_boundary) {
            std::cerr << ("batch mtp slot=" + std::to_string(slot) +
                          " drafted=" + std::to_string(step.mtp_drafted_tokens) +
                          " accepted=" + std::to_string(step.mtp_accepted_tokens) +
                          " verified=" + std::to_string(step.mtp_verified_rows) +
                          " replayed=" + std::to_string(step.mtp_replayed_rows) +
                          " boundary=" + std::to_string(step.mtp_boundary) + '\n');
        }
    }
}

bool emit_step(const DecoderBatchStep& step, const DecoderCallbacks& callbacks,
               DecoderResult& result) {
    result.mtp_rounds += step.mtp_drafted_tokens != 0;
    result.mtp_drafted_tokens += step.mtp_drafted_tokens;
    result.output_token_ids.push_back(*step.token);
    result.mtp_accepted_tokens += step.mtp_accepted_tokens > 0;
    if (callbacks.on_token && !callbacks.on_token(*step.token)) {
        return false;
    }
    for (int index = 0; index < step.extra_token_count; ++index) {
        if (callbacks.is_cancelled && callbacks.is_cancelled()) {
            return false;
        }
        result.output_token_ids.push_back(step.extra_tokens[index]);
        result.mtp_accepted_tokens += index + 1 < step.mtp_accepted_tokens;
        if (callbacks.on_token && !callbacks.on_token(step.extra_tokens[index])) {
            return false;
        }
    }
    return true;
}

}

struct BatchService::State {
    State(const Decoder& decoder, bool trace, bool cuda_graphs, int mtp_steps, int max_active,
          int prefix_cache_entries, int max_num_batched_tokens);
    ~State();

    void run(const Decoder& decoder, std::promise<void> started);
    void process(DecoderBatch& batch);
    std::optional<std::uint64_t> begin_iteration(DecoderBatch& batch);
    std::optional<Jobs> wait_for_jobs();
    bool prepare(DecoderBatch& batch, int slot, const std::shared_ptr<Job>& job);
    void publish(int slot, Job& job, const DecoderBatchStep& step);
    void record_profile(std::uint64_t epoch, std::uint64_t elapsed, int width,
                        const Steps& steps, const DecoderBatchStepStats& stats);
    void fail_jobs(std::exception_ptr error, bool permanent);
    void cancel(std::unique_lock<std::mutex>& lock, int slot, Job& job);
    DecoderResult consume(std::unique_lock<std::mutex>& lock, int slot, Job& job,
                          const DecoderCallbacks& callbacks);
    void cancel_batch(std::unique_lock<std::mutex>& lock, const BatchJobs& active);
    bool drain_batch(std::unique_lock<std::mutex>& lock, BatchJobs& active,
                     const BatchCallbacks& callbacks, std::vector<DecoderResult>& results,
                     std::size_t& completed);
    bool consume_batch(std::unique_lock<std::mutex>& lock, int owner,
                       const std::vector<BatchPrompt>& prompts, const BatchCallbacks& callbacks,
                       BatchJobs& active,
                       std::vector<DecoderResult>& results);
    void enqueue_batch(int owner, const std::vector<BatchPrompt>& prompts, BatchJobs& active,
                       std::size_t& next);

    const bool trace;
    const bool cuda_graphs;
    const int mtp_steps;
    const int max_active;
    const int prefix_cache_entries;
    const int max_num_batched_tokens;
    std::mutex mutex;
    std::condition_variable changed;
    Jobs jobs;
    Selection generating{};
    BatchMetrics counters;
    std::optional<BatchProfile> profile;
    std::uint64_t profile_epoch = 0;
    std::size_t profile_limit = 0;
    std::exception_ptr failure;
    std::atomic_bool failed{false};
    bool reset_pending = false;
    bool closing = false;
    std::thread worker;
};

BatchService::State::State(const Decoder& decoder, bool trace, bool cuda_graphs, int mtp_steps,
                           int max_active, int prefix_cache_entries, int max_num_batched_tokens)
    : trace(trace), cuda_graphs(cuda_graphs), mtp_steps(mtp_steps), max_active(max_active),
      prefix_cache_entries(prefix_cache_entries), max_num_batched_tokens(max_num_batched_tokens) {
    if (max_active < 1 || max_active > decoder_batch_max_lanes) {
        throw std::invalid_argument("Invalid decoder batch service capacity");
    }
    counters.cache_capacity = prefix_cache_entries;
    std::promise<void> started;
    auto ready = started.get_future();
    worker = std::thread(&State::run, this, std::cref(decoder), std::move(started));
    try {
        ready.get();
    } catch (...) {
        worker.join();
        throw;
    }
}

BatchService::State::~State() {
    {
        const std::lock_guard lock(mutex);
        closing = true;
    }
    changed.notify_all();
    worker.join();
}

void BatchService::State::run(const Decoder& decoder, std::promise<void> started) {
    std::unique_ptr<DecoderBatch> batch;
    try {
        batch = std::make_unique<DecoderBatch>(decoder,
            DecoderBatchOptions{cuda_graphs, max_active, prefix_cache_entries,
                                max_num_batched_tokens, std::numeric_limits<std::size_t>::max()});
        counters.cache = batch->prefix_cache_stats();
    } catch (...) {
        started.set_exception(std::current_exception());
        return;
    }
    started.set_value();
    try {
        process(*batch);
    } catch (...) {
        fail_jobs(std::current_exception(), true);
    }
}

void BatchService::State::fail_jobs(std::exception_ptr error, bool permanent) {
    const std::lock_guard lock(mutex);
    if (permanent) {
        failure = error;
        failed.store(true);
        reset_pending = false;
    }
    for (auto& job : jobs) {
        if (job) {
            job->error = error;
            job.reset();
        }
    }
    changed.notify_all();
}

std::optional<Jobs> BatchService::State::wait_for_jobs() {
    std::unique_lock lock(mutex);
    const auto end = jobs.begin() + max_active;
    changed.wait(lock, [&] {
        return closing || reset_pending || std::any_of(jobs.begin(), end, [](const auto& job) {
            return job && job->ready();
        });
    });
    const bool admitting = std::any_of(jobs.begin(), end, [](const auto& job) {
        return job && !job->admitted && !job->cancelled;
    });
    changed.wait_for(lock, mailbox_grace, [&] {
        return closing || reset_pending ||
               std::all_of(jobs.begin(), end, [admitting](const auto& job) {
                   return job ? job->ready() : !admitting;
               });
    });
    if (closing) {
        return std::nullopt;
    }
    return jobs;
}

bool BatchService::State::prepare(DecoderBatch& batch, int slot,
                                  const std::shared_ptr<Job>& job) {
    bool cancelled;
    {
        const std::lock_guard lock(mutex);
        if (!job || !job->ready()) {
            return false;
        }
        cancelled = job->cancelled;
    }
    if (cancelled) {
        if (job->admitted) {
            batch.cancel(slot);
        }
        const std::lock_guard lock(mutex);
        publish(slot, *job, {std::nullopt, DecoderFinishReason::cancelled});
        changed.notify_all();
        return false;
    }
    if (!job->admitted) {
        try {
            if (job->media.empty()) {
                job->cached_tokens = batch.admit(slot, std::move(job->prompt), job->options);
            } else {
                batch.admit_multimodal(slot, std::move(job->prompt), job->options, job->media);
                job->media.clear();
            }
            const std::lock_guard lock(mutex);
            job->admitted = true;
            changed.notify_all();
        } catch (...) {
            if (!batch.available()) {
                throw;
            }
            const std::lock_guard lock(mutex);
            job->error = std::current_exception();
            jobs[slot].reset();
            changed.notify_all();
            return false;
        }
    }
    return true;
}

void BatchService::State::publish(int slot, Job& job, const DecoderBatchStep& step) {
    if (step.token && !job.cancelled) {
        if (counters.cache_capacity != 0) {
            if (!job.prefill_completed) {
                job.prefill_completed = true;
                counters.prompt_tokens += job.prompt_tokens;
                ++counters.prefill_requests;
            }
            counters.generation_tokens += 1 + step.extra_token_count;
        }
        job.pending = step;
    }
    if (step.sampling_failed || step.constraint_failed) {
        job.error = step.constraint_failed ? std::make_exception_ptr(DecoderConstraintError())
                                           : std::make_exception_ptr(DecoderSamplingError());
        jobs[slot].reset();
        return;
    }
    if (step.finish_reason) {
        job.finish = step.finish_reason;
        jobs[slot].reset();
    }
}

void BatchService::State::record_profile(std::uint64_t epoch, std::uint64_t elapsed, int width,
                                         const Steps& steps, const DecoderBatchStepStats& stats) {
    if (!profile || epoch != profile_epoch || profile->steps.size() == profile_limit || !width) {
        return;
    }
    const int decode = std::count_if(steps.begin(), steps.end(),
                                    [](const auto& step) { return step.decode; });
    const int rows = std::accumulate(stats.prefill_rows.begin(), stats.prefill_rows.end(), 0);
    profile->steps.push_back({elapsed, width, rows, decode, stats.graph_slots});
}

void trace_batch_step(int width, int decode, std::size_t request_bytes) {
    std::cerr << ("batch width=" + std::to_string(width) + " decode=" + std::to_string(decode) +
                  " request_bytes=" + std::to_string(request_bytes) + '\n');
}

void trace_prefix_cache(const DecoderPrefixCacheStats& stats) {
    std::cerr << ("prefix cache entries=" + std::to_string(stats.entries) +
                  " pages=" + std::to_string(stats.pages) +
                  " device_bytes=" + std::to_string(stats.device_bytes) +
                  " hits=" + std::to_string(stats.hits) +
                  " cached_tokens=" + std::to_string(stats.cached_tokens) +
                  " prefill_tokens=" + std::to_string(stats.prefill_tokens) +
                  " stores=" + std::to_string(stats.stores) +
                  " evictions=" + std::to_string(stats.evictions) + '\n');
}

std::optional<std::uint64_t> BatchService::State::begin_iteration(DecoderBatch& batch) {
    std::unique_lock lock(mutex);
    if (reset_pending) {
        lock.unlock();
        batch.reset_prefix_cache();
        const auto stats = batch.prefix_cache_stats();
        lock.lock();
        counters.cache = stats;
        reset_pending = false;
        changed.notify_all();
        return std::nullopt;
    }
    return profile && profile->steps.size() < profile_limit ? profile_epoch : 0;
}

void BatchService::State::process(DecoderBatch& batch) {
    std::size_t traced_bytes = 0;
    const bool trace_cache = trace && prefix_cache_entries != 0;
    auto traced_cache = batch.prefix_cache_stats();
    if (trace_cache) {
        trace_prefix_cache(traced_cache);
    }
    while (const auto snapshot = wait_for_jobs()) {
        const auto iteration = begin_iteration(batch);
        if (!iteration) {
            continue;
        }
        const auto epoch = *iteration;
        Selection selected{};
        for (int slot = 0; slot < max_active; ++slot) {
            selected[slot] = prepare(batch, slot, (*snapshot)[slot]);
        }
        const int width = std::count(selected.begin(), selected.end(), true);
        Steps steps{};
        DecoderBatchStepStats stats;
        std::uint64_t elapsed = 0;
        if (width) {
            try {
                const auto start = epoch ? std::chrono::steady_clock::now()
                                         : std::chrono::steady_clock::time_point{};
                steps = batch.advance_chunked(selected, trace || epoch ? &stats : nullptr);
                if (epoch) {
                    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - start).count();
                }
            } catch (...) {
                if (!batch.available()) {
                    throw;
                }
                fail_jobs(std::current_exception(), false);
                if (trace) {
                    trace_batch_step(0, 0, batch.request_device_allocation_bytes());
                    traced_bytes = 0;
                }
                continue;
            }
        }
        if (trace) {
            const auto bytes = batch.request_device_allocation_bytes();
            const int decode = std::count_if(steps.begin(), steps.end(),
                                              [](const auto& step) { return step.decode; });
            if (width || bytes != traced_bytes) {
                trace_batch_step(width, decode, bytes);
                if (width) {
                    trace_prefill_rows(stats, max_active);
                    std::cerr << ("batch target_rows=" + std::to_string(stats.target_rows) +
                                  " step_budget=" + std::to_string(max_num_batched_tokens) + '\n');
                    if (stats.graph_slots) {
                        std::cerr << ("batch graph_slots=" +
                                     std::to_string(stats.graph_slots) + '\n');
                    }
                    if (stats.sampling_workspace_bytes) {
                        std::cerr << ("batch sampling_workspace_bytes=" +
                                      std::to_string(stats.sampling_workspace_bytes) + '\n');
                    }
                    if (stats.constraint_workspace_bytes) {
                        std::cerr << ("batch constraint_workspace_bytes=" +
                                      std::to_string(stats.constraint_workspace_bytes) + '\n');
                    }
                    if (mtp_steps != 0) {
                        trace_verification(steps, stats);
                    } else {
                        std::cerr << ("batch peak_request_bytes=" +
                                      std::to_string(stats.peak_request_bytes) + '\n');
                    }
                }
                traced_bytes = bytes;
            }
        }
        const auto cache = prefix_cache_entries ? batch.prefix_cache_stats()
                                                : DecoderPrefixCacheStats{};
        if (trace_cache) {
            const bool finished = std::any_of(steps.begin(), steps.end(),
                [](const auto& step) { return step.finish_reason.has_value(); });
            if (finished || cache.hits != traced_cache.hits ||
                cache.stores != traced_cache.stores) {
                trace_prefix_cache(cache);
                traced_cache = cache;
            }
        }
        {
            const std::lock_guard lock(mutex);
            counters.cache = cache;
            if (epoch) {
                record_profile(epoch, elapsed, width, steps, stats);
            }
            for (int slot = 0; slot < max_active; ++slot) {
                if (selected[slot]) {
                    publish(slot, *(*snapshot)[slot], steps[slot]);
                }
            }
        }
        changed.notify_all();
    }
}

void BatchService::State::cancel(std::unique_lock<std::mutex>& lock, int slot, Job& job) {
    if (jobs[slot].get() != &job) {
        return;
    }
    job.cancelled = true;
    changed.notify_all();
    changed.wait(lock, [&] { return jobs[slot].get() != &job; });
}

DecoderResult BatchService::State::consume(std::unique_lock<std::mutex>& lock, int slot,
                                           Job& job, const DecoderCallbacks& callbacks) {
    DecoderResult result;
    bool announced = false;
    for (;;) {
        const bool announce = job.admitted && !job.error && callbacks.on_admitted && !announced;
        const bool poll_stream = announced && callbacks.on_stream_poll && !job.error &&
            !job.finish;
        lock.unlock();
        const bool connected = (!callbacks.is_cancelled || !callbacks.is_cancelled()) &&
            (!announce || callbacks.on_admitted()) &&
            (!poll_stream || callbacks.on_stream_poll());
        lock.lock();
        announced = announced || announce;
        if (!connected) {
            cancel(lock, slot, job);
            result.finish_reason = DecoderFinishReason::cancelled;
            return result;
        }
        const bool can_emit = announced || !job.admitted || !callbacks.on_admitted;
        if (job.pending && can_emit) {
            const auto step = *job.pending;
            lock.unlock();
            const bool accepted = emit_step(step, callbacks, result);
            lock.lock();
            if (!accepted) {
                cancel(lock, slot, job);
                result.finish_reason = DecoderFinishReason::cancelled;
                return result;
            }
            job.pending.reset();
            changed.notify_all();
        } else if (job.error) {
            std::rethrow_exception(job.error);
        } else if (job.finish && can_emit) {
            result.finish_reason = *job.finish;
            return result;
        } else if (!job.pending && !job.finish) {
            changed.wait_for(lock, disconnect_poll);
        }
    }
}

void BatchService::State::cancel_batch(std::unique_lock<std::mutex>& lock,
                                       const BatchJobs& active) {
    const auto owned = [&](int slot) {
        return active[slot].job && jobs[slot] == active[slot].job;
    };
    for (int slot = 0; slot < max_active; ++slot) {
        if (owned(slot)) {
            jobs[slot]->cancelled = true;
        }
    }
    changed.notify_all();
    changed.wait(lock, [&] {
        for (int slot = 0; slot < max_active; ++slot) {
            if (owned(slot)) {
                return false;
            }
        }
        return true;
    });
}

void BatchService::State::enqueue_batch(int owner, const std::vector<BatchPrompt>& prompts,
                                        BatchJobs& active, std::size_t& next) {
    for (int offset = 0; offset < max_active && next < prompts.size(); ++offset) {
        const int slot = (owner + offset) % max_active;
        if (jobs[slot] || active[slot].job || (slot != owner && generating[slot])) {
            continue;
        }
        auto job = std::make_shared<Job>();
        job->prompt = prompts[next].prompt;
        job->prompt_tokens = job->prompt.size();
        job->options = prompts[next].options;
        job->options.mtp_steps = mtp_steps;
        active[slot] = {job, next++};
        jobs[slot] = std::move(job);
    }
}

bool BatchService::State::drain_batch(std::unique_lock<std::mutex>& lock, BatchJobs& active,
                                      const BatchCallbacks& callbacks,
                                      std::vector<DecoderResult>& results,
                                      std::size_t& completed) {
    const DecoderCallbacks collect;
    for (auto& item : active) {
        if (!item.job) {
            continue;
        }
        auto& job = *item.job;
        if (job.error) {
            std::rethrow_exception(job.error);
        }
        const auto finish = job.finish;
        if (finish == DecoderFinishReason::cancelled) {
            return false;
        }
        if (!job.pending && !finish) {
            continue;
        }
        auto& result = results[item.index];
        if (job.pending) {
            emit_step(*job.pending, collect, result);
        }
        result.cached_prompt_tokens = job.cached_tokens;
        if (finish) {
            result.finish_reason = *finish;
        }
        if (callbacks.on_output) {
            lock.unlock();
            const bool written = callbacks.on_output(item.index, result, finish.has_value());
            lock.lock();
            if (!written) {
                return false;
            }
        }
        if (job.pending) {
            job.pending.reset();
            changed.notify_all();
        }
        if (finish) {
            item.job.reset();
            ++completed;
        }
    }
    return true;
}

bool BatchService::State::consume_batch(std::unique_lock<std::mutex>& lock, int owner,
                                        const std::vector<BatchPrompt>& prompts,
                                        const BatchCallbacks& callbacks, BatchJobs& active,
                                        std::vector<DecoderResult>& results) {
    std::size_t next = 0;
    std::size_t completed = 0;
    bool announced = false;
    while (completed < prompts.size()) {
        lock.unlock();
        const bool connected = (!callbacks.is_cancelled || !callbacks.is_cancelled()) &&
            (!announced || !callbacks.on_stream_poll || callbacks.on_stream_poll());
        lock.lock();
        if (!connected) {
            return false;
        }
        if (failure) {
            std::rethrow_exception(failure);
        }
        if (!announced && callbacks.on_admitted) {
            bool admitted = false;
            for (const auto& item : active) {
                if (item.job && item.job->error) {
                    std::rethrow_exception(item.job->error);
                }
                admitted = admitted || (item.job && item.job->admitted);
            }
            if (admitted) {
                lock.unlock();
                announced = callbacks.on_admitted();
                lock.lock();
                if (!announced) {
                    return false;
                }
            }
        }
        if ((announced || !callbacks.on_admitted) &&
            !drain_batch(lock, active, callbacks, results, completed)) {
            return false;
        }
        const auto submitted = next;
        enqueue_batch(owner, prompts, active, next);
        if (next != submitted) {
            changed.notify_all();
        }
        if (completed < prompts.size()) {
            changed.wait_for(lock, disconnect_poll);
        }
    }
    return true;
}

BatchService::BatchService(const Decoder& decoder, bool trace, bool cuda_graphs, int mtp_steps,
                           int max_active, int prefix_cache_entries, int max_num_batched_tokens)
    : state_(std::make_unique<State>(decoder, trace, cuda_graphs, mtp_steps, max_active,
                                     prefix_cache_entries, max_num_batched_tokens)) {}

BatchService::~BatchService() = default;

bool BatchService::ready() const {
    return !state_->failed.load();
}

void BatchService::rethrow_failure() const {
    if (!ready()) {
        const std::lock_guard lock(state_->mutex);
        std::rethrow_exception(state_->failure);
    }
}

BatchMetrics BatchService::metrics() const {
    const std::lock_guard lock(state_->mutex);
    return state_->counters;
}

bool BatchService::reset_prefix_cache() {
    std::unique_lock lock(state_->mutex);
    if (state_->failure) {
        std::rethrow_exception(state_->failure);
    }
    if (state_->reset_pending ||
        std::any_of(state_->generating.begin(), state_->generating.end(),
                    [](bool active) { return active; })) {
        return false;
    }
    state_->reset_pending = true;
    state_->changed.notify_all();
    state_->changed.wait(lock, [&] { return !state_->reset_pending; });
    if (state_->failure) {
        std::rethrow_exception(state_->failure);
    }
    return true;
}

bool BatchService::start_profile(std::string name, std::size_t iterations) {
    if (iterations == 0 || iterations > 64) {
        throw std::invalid_argument("Native profile iteration limit must be in [1, 64]");
    }
    const std::lock_guard lock(state_->mutex);
    if (state_->profile) {
        return false;
    }
    BatchProfile profile{std::move(name), {}};
    profile.steps.reserve(iterations);
    state_->profile = std::move(profile);
    state_->profile_limit = iterations;
    ++state_->profile_epoch;
    return true;
}

std::optional<BatchProfile> BatchService::stop_profile() {
    const std::lock_guard lock(state_->mutex);
    return std::exchange(state_->profile, std::nullopt);
}

DecoderResult BatchService::generate(int slot, const std::vector<int>& prompt,
                                     const DecoderOptions& options,
                                     const DecoderCallbacks& callbacks,
                                     std::vector<DecoderMedia> media, std::size_t expanded_tokens) {
    if (slot < 0 || slot >= state_->max_active) {
        throw std::out_of_range("Invalid decoder batch service slot");
    }
    auto job = std::make_shared<Job>();
    job->prompt = prompt;
    job->prompt_tokens = media.empty() ? prompt.size() : expanded_tokens;
    job->options = options;
    if (media.empty()) {
        job->options.mtp_steps = state_->mtp_steps;
    }
    job->media = std::move(media);
    std::unique_lock lock(state_->mutex);
    state_->changed.wait(lock, [&] { return !state_->reset_pending; });
    if (state_->failure) {
        std::rethrow_exception(state_->failure);
    }
    auto& queue = state_->jobs.at(slot);
    auto& generating = state_->generating[slot];
    if (generating) {
        throw std::logic_error("Connection already has an active decoder job");
    }
    generating = true;
    try {
        while (queue && !state_->failure) {
            lock.unlock();
            const bool cancelled = callbacks.is_cancelled && callbacks.is_cancelled();
            lock.lock();
            if (cancelled) {
                generating = false;
                state_->changed.notify_all();
                DecoderResult result;
                result.finish_reason = DecoderFinishReason::cancelled;
                return result;
            }
            if (queue) {
                state_->changed.wait_for(lock, disconnect_poll);
            }
        }
        if (state_->failure) {
            std::rethrow_exception(state_->failure);
        }
        queue = job;
        state_->changed.notify_all();
        auto result = state_->consume(lock, slot, *job, callbacks);
        result.cached_prompt_tokens = job->cached_tokens;
        generating = false;
        state_->changed.notify_all();
        return result;
    } catch (...) {
        if (!lock.owns_lock()) {
            lock.lock();
        }
        state_->cancel(lock, slot, *job);
        generating = false;
        state_->changed.notify_all();
        throw;
    }
}

std::optional<std::vector<DecoderResult>>
BatchService::generate_batch(int slot, const std::vector<BatchPrompt>& prompts,
                             const BatchCallbacks& callbacks) {
    if (slot < 0 || slot >= state_->max_active) {
        throw std::out_of_range("Invalid decoder batch service slot");
    }
    std::vector<DecoderResult> results(prompts.size());
    BatchJobs active;
    std::unique_lock lock(state_->mutex);
    state_->changed.wait(lock, [&] { return !state_->reset_pending; });
    auto& generating = state_->generating[slot];
    if (generating) {
        throw std::logic_error("Connection already has an active decoder job");
    }
    generating = true;
    bool completed;
    try {
        completed = state_->consume_batch(lock, slot, prompts, callbacks, active, results);
    } catch (...) {
        if (!lock.owns_lock()) {
            lock.lock();
        }
        state_->cancel_batch(lock, active);
        generating = false;
        state_->changed.notify_all();
        throw;
    }
    if (!completed) {
        state_->cancel_batch(lock, active);
    }
    generating = false;
    state_->changed.notify_all();
    if (completed) {
        return results;
    }
    return std::nullopt;
}

}
