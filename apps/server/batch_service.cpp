#include "sglang/server/batch_service.h"

#include "sglang/decoder_batch.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace sglang::server {
namespace {

struct Job {
    std::vector<int> prompt;
    DecoderOptions options;
    std::optional<DecoderBatchStep> pending;
    std::optional<DecoderFinishReason> finish;
    std::exception_ptr error;
    int cached_tokens = 0;
    bool admitted = false;
    bool cancelled = false;

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
    std::optional<Jobs> wait_for_jobs();
    bool prepare(DecoderBatch& batch, int slot, const std::shared_ptr<Job>& job);
    void publish(int slot, Job& job, const DecoderBatchStep& step);
    void fail_jobs(std::exception_ptr error, bool permanent);
    void cancel(std::unique_lock<std::mutex>& lock, int slot, Job& job);

    const bool trace;
    const bool cuda_graphs;
    const int mtp_steps;
    const int max_active;
    const int prefix_cache_entries;
    std::mutex mutex;
    std::condition_variable changed;
    Jobs jobs;
    std::exception_ptr failure;
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
        return closing || std::any_of(jobs.begin(), end, [](const auto& job) {
            return job && job->ready();
        });
    });
    const bool admitting = std::any_of(jobs.begin(), end, [](const auto& job) {
        return job && !job->admitted && !job->cancelled;
    });
    changed.wait_for(lock, mailbox_grace, [&] {
        return closing || std::all_of(jobs.begin(), end, [admitting](const auto& job) {
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
    if (step.sampling_failed || step.constraint_failed) {
        job.error = step.constraint_failed ? std::make_exception_ptr(DecoderConstraintError())
                                           : std::make_exception_ptr(DecoderSamplingError());
        jobs[slot].reset();
        return;
    }
    if (step.token && !job.cancelled) {
        job.pending = step;
    }
    if (step.finish_reason) {
        job.finish = step.finish_reason;
        jobs[slot].reset();
    }
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

void BatchService::State::process(DecoderBatch& batch) {
    std::size_t traced_bytes = 0;
    const bool trace_cache = trace && prefix_cache_entries != 0;
    auto traced_cache = batch.prefix_cache_stats();
    if (trace_cache) {
        trace_prefix_cache(traced_cache);
    }
    while (const auto snapshot = wait_for_jobs()) {
        Selection selected{};
        for (int slot = 0; slot < max_active; ++slot) {
            selected[slot] = prepare(batch, slot, (*snapshot)[slot]);
        }
        const int width = std::count(selected.begin(), selected.end(), true);
        Steps steps{};
        DecoderBatchStepStats stats;
        if (width) {
            try {
                steps = batch.advance_chunked(selected, trace ? &stats : nullptr);
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
        if (trace_cache) {
            const auto current = batch.prefix_cache_stats();
            const bool finished = std::any_of(steps.begin(), steps.end(),
                [](const auto& step) { return step.finish_reason.has_value(); });
            if (finished || current.hits != traced_cache.hits ||
                current.stores != traced_cache.stores) {
                trace_prefix_cache(current);
                traced_cache = current;
            }
        }
        {
            const std::lock_guard lock(mutex);
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
    job.cancelled = true;
    changed.notify_all();
    changed.wait(lock, [&] { return jobs[slot].get() != &job; });
}

BatchService::BatchService(const Decoder& decoder, bool trace, bool cuda_graphs, int mtp_steps,
                           int max_active, int prefix_cache_entries)
    : state_(std::make_unique<State>(decoder, trace, cuda_graphs, mtp_steps, max_active,
                                     prefix_cache_entries)) {}

BatchService::~BatchService() = default;

DecoderResult BatchService::generate(int slot, const std::vector<int>& prompt,
                                     const DecoderOptions& options,
                                     const DecoderCallbacks& callbacks) {
    if (slot < 0 || slot >= state_->max_active) {
        throw std::out_of_range("Invalid decoder batch service slot");
    }
    auto job = std::make_shared<Job>();
    job->prompt = prompt;
    job->options = options;
    if (!job->options.requires_target_only()) {
        job->options.mtp_steps = state_->mtp_steps;
    }
    DecoderResult result;
    std::unique_lock lock(state_->mutex);
    if (state_->failure) {
        std::rethrow_exception(state_->failure);
    }
    auto& queue = state_->jobs.at(slot);
    if (queue) {
        throw std::logic_error("Connection already has an active decoder job");
    }
    queue = job;
    state_->changed.notify_all();
    try {
        for (;;) {
            lock.unlock();
            const bool abandoned = callbacks.is_cancelled && callbacks.is_cancelled();
            lock.lock();
            if (abandoned) {
                state_->cancel(lock, slot, *job);
                result.finish_reason = DecoderFinishReason::cancelled;
                return result;
            }
            if (job->error) {
                std::rethrow_exception(job->error);
            }
            if (job->pending) {
                const auto step = *job->pending;
                lock.unlock();
                const bool accepted = emit_step(step, callbacks, result);
                lock.lock();
                if (!accepted) {
                    state_->cancel(lock, slot, *job);
                    result.finish_reason = DecoderFinishReason::cancelled;
                    return result;
                }
                job->pending.reset();
                state_->changed.notify_all();
            } else if (job->finish) {
                result.finish_reason = *job->finish;
                result.cached_prompt_tokens = job->cached_tokens;
                return result;
            } else {
                state_->changed.wait_for(lock, disconnect_poll);
            }
        }
    } catch (...) {
        if (!lock.owns_lock()) {
            lock.lock();
        }
        state_->cancel(lock, slot, *job);
        throw;
    }
}

}
