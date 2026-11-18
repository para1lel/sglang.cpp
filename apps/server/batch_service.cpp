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
    std::optional<int> token;
    std::optional<DecoderFinishReason> finish;
    std::exception_ptr error;
    bool admitted = false;
    bool cancelled = false;

    bool ready() const { return cancelled || !token; }
};

using Jobs = std::array<std::shared_ptr<Job>, decoder_batch_max_lanes>;
using Selection = std::array<bool, decoder_batch_max_lanes>;
constexpr auto disconnect_poll = std::chrono::milliseconds(10);
constexpr auto mailbox_grace = std::chrono::milliseconds(1);

void trace_prefill_rows(const DecoderBatchStepStats& stats) {
    const auto& rows = stats.prefill_rows;
    std::cerr << ("batch prefill_rows=" + std::to_string(rows[0]) + ',' + std::to_string(rows[1]) +
                  ',' + std::to_string(rows[2]) + ',' + std::to_string(rows[3]) +
                  " workspace_bytes=" + std::to_string(stats.prefill_workspace_bytes) + '\n');
}

}

struct BatchService::State {
    State(const Decoder& decoder, bool trace);
    ~State();

    void run(const Decoder& decoder, std::promise<void> started);
    void process(DecoderBatch& batch);
    std::optional<Jobs> wait_for_jobs();
    bool prepare(DecoderBatch& batch, int slot, const std::shared_ptr<Job>& job);
    void publish(int slot, Job& job, const DecoderBatchStep& step);
    void cancel(std::unique_lock<std::mutex>& lock, int slot, Job& job);

    const bool trace;
    std::mutex mutex;
    std::condition_variable changed;
    Jobs jobs;
    std::exception_ptr failure;
    bool closing = false;
    std::thread worker;
};

BatchService::State::State(const Decoder& decoder, bool trace) : trace(trace) {
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
        batch = std::make_unique<DecoderBatch>(decoder);
    } catch (...) {
        started.set_exception(std::current_exception());
        return;
    }
    started.set_value();
    try {
        process(*batch);
    } catch (...) {
        const std::lock_guard lock(mutex);
        failure = std::current_exception();
        for (auto& job : jobs) {
            if (job) {
                job->error = failure;
                job.reset();
            }
        }
        changed.notify_all();
    }
}

std::optional<Jobs> BatchService::State::wait_for_jobs() {
    std::unique_lock lock(mutex);
    changed.wait(lock, [&] {
        return closing || std::any_of(jobs.begin(), jobs.end(), [](const auto& job) {
            return job && job->ready();
        });
    });
    const bool admitting = std::any_of(jobs.begin(), jobs.end(), [](const auto& job) {
        return job && !job->admitted && !job->cancelled;
    });
    changed.wait_for(lock, mailbox_grace, [&] {
        return closing || std::all_of(jobs.begin(), jobs.end(), [admitting](const auto& job) {
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
            batch.admit(slot, std::move(job->prompt), job->options);
            job->admitted = true;
        } catch (const std::invalid_argument&) {
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
        job.token = step.token;
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

void BatchService::State::process(DecoderBatch& batch) {
    std::size_t traced_bytes = 0;
    while (const auto snapshot = wait_for_jobs()) {
        Selection selected{};
        for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
            selected[slot] = prepare(batch, slot, (*snapshot)[slot]);
        }
        const int width = std::count(selected.begin(), selected.end(), true);
        std::array<DecoderBatchStep, decoder_batch_max_lanes> steps{};
        DecoderBatchStepStats stats;
        if (width) {
            steps = batch.advance_chunked(selected, trace ? &stats : nullptr);
        }
        if (trace) {
            const auto bytes = batch.request_device_allocation_bytes();
            const int decode = std::count_if(steps.begin(), steps.end(),
                                              [](const auto& step) { return step.decode; });
            if (width || bytes != traced_bytes) {
                trace_batch_step(width, decode, bytes);
                if (width) {
                    trace_prefill_rows(stats);
                }
                traced_bytes = bytes;
            }
        }
        {
            const std::lock_guard lock(mutex);
            for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
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

BatchService::BatchService(const Decoder& decoder, bool trace)
    : state_(std::make_unique<State>(decoder, trace)) {}

BatchService::~BatchService() = default;

DecoderResult BatchService::generate(int slot, const std::vector<int>& prompt,
                                     const DecoderOptions& options,
                                     const DecoderCallbacks& callbacks) {
    auto job = std::make_shared<Job>();
    job->prompt = prompt;
    job->options = options;
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
            if (job->token) {
                const int token = *job->token;
                lock.unlock();
                result.output_token_ids.push_back(token);
                const bool accepted = !callbacks.on_token || callbacks.on_token(token);
                lock.lock();
                if (!accepted) {
                    state_->cancel(lock, slot, *job);
                    result.finish_reason = DecoderFinishReason::cancelled;
                    return result;
                }
                job->token.reset();
                state_->changed.notify_all();
            } else if (job->finish) {
                result.finish_reason = *job->finish;
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
