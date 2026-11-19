#include "sglang/server/batch_service.h"

#include "sglang/decoder_batch.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
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
    for (int slot = 0; slot < decoder_batch_max_mtp_lanes; ++slot) {
        const auto& step = steps[slot];
        if (step.mtp_drafted_tokens || step.mtp_boundary) {
            std::cerr << ("batch mtp slot=" + std::to_string(slot) +
                          " drafted=" + std::to_string(step.mtp_drafted_tokens) +
                          " accepted=" + std::to_string(step.mtp_accepted_tokens) +
                          " verified=" + std::to_string(step.mtp_drafted_tokens ?
                              step.mtp_drafted_tokens + 1 : 0) +
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
          int prefix_cache_entries);
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

    const bool trace;
    const bool cuda_graphs;
    const int mtp_steps;
    const int max_active;
    const int prefix_cache_entries;
    std::mutex mutex;
    std::condition_variable changed;
    Jobs jobs;
    Selection connections{};
    Selection generating{};
    BatchMetrics counters;
    std::optional<BatchProfile> profile;
    std::uint64_t profile_epoch = 0;
    std::size_t profile_limit = 0;
    std::exception_ptr failure;
    bool reset_pending = false;
    bool closing = false;
    std::thread worker;
};

BatchService::State::State(const Decoder& decoder, bool trace, bool cuda_graphs, int mtp_steps,
                           int max_active, int prefix_cache_entries)
    : trace(trace), cuda_graphs(cuda_graphs), mtp_steps(mtp_steps), max_active(max_active),
      prefix_cache_entries(prefix_cache_entries) {
    if (max_active < 1 || max_active > decoder_batch_max_lanes) {
        throw std::invalid_argument("Invalid decoder batch service capacity");
    }
    if (mtp_steps != 0 && max_active > decoder_batch_max_mtp_lanes) {
        throw std::invalid_argument("Decoder batch service capacity above 4 requires MTP off");
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
            DecoderBatchOptions{cuda_graphs, max_active, prefix_cache_entries});
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
            job->cached_tokens = batch.admit(slot, std::move(job->prompt), job->options);
            job->admitted = true;
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
    const std::lock_guard lock(mutex);
    if (reset_pending) {
        batch.reset_prefix_cache();
        counters.cache = batch.prefix_cache_stats();
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
    for (;;) {
        lock.unlock();
        const bool abandoned = callbacks.is_cancelled && callbacks.is_cancelled();
        lock.lock();
        if (abandoned) {
            cancel(lock, slot, job);
            result.finish_reason = DecoderFinishReason::cancelled;
            return result;
        }
        if (job.pending) {
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
        } else if (job.finish) {
            result.finish_reason = *job.finish;
            result.cached_prompt_tokens = job.cached_tokens;
            return result;
        } else {
            changed.wait_for(lock, disconnect_poll);
        }
    }
}

BatchService::BatchService(const Decoder& decoder, bool trace, bool cuda_graphs, int mtp_steps,
                           int max_active, int prefix_cache_entries)
    : state_(std::make_unique<State>(decoder, trace, cuda_graphs, mtp_steps, max_active,
                                     prefix_cache_entries)) {}

BatchService::~BatchService() = default;

void BatchService::claim_connection(int slot) {
    const std::lock_guard lock(state_->mutex);
    state_->connections.at(slot) = true;
}

void BatchService::release_connection(int slot) {
    const std::lock_guard lock(state_->mutex);
    state_->connections.at(slot) = false;
}

BatchMetrics BatchService::metrics() const {
    const std::lock_guard lock(state_->mutex);
    auto result = state_->counters;
    for (int slot = 0; slot < state_->max_active; ++slot) {
        result.running += state_->connections[slot] || state_->generating[slot];
    }
    return result;
}

bool BatchService::reset_prefix_cache() {
    std::unique_lock lock(state_->mutex);
    if (state_->failure) {
        std::rethrow_exception(state_->failure);
    }
    for (int slot = 0; slot < state_->max_active; ++slot) {
        if (state_->connections[slot] || state_->generating[slot]) {
            return false;
        }
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
                                     const DecoderCallbacks& callbacks) {
    if (slot < 0 || slot >= state_->max_active) {
        throw std::out_of_range("Invalid decoder batch service slot");
    }
    auto job = std::make_shared<Job>();
    job->prompt = prompt;
    job->prompt_tokens = prompt.size();
    job->options = options;
    if (!job->options.requires_target_only()) {
        job->options.mtp_steps = state_->mtp_steps;
    }
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
    queue = job;
    state_->changed.notify_all();
    try {
        auto result = state_->consume(lock, slot, *job, callbacks);
        generating = false;
        if (result.finish_reason != DecoderFinishReason::cancelled) {
            state_->connections[slot] = false;
        }
        return result;
    } catch (...) {
        if (!lock.owns_lock()) {
            lock.lock();
        }
        state_->cancel(lock, slot, *job);
        generating = false;
        throw;
    }
}

}
