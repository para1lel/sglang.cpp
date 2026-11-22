#include "sglang/server/batch_service.h"

#include "sglang/server/batch_trace.h"
#include "sglang/server/indexed_admission.h"
#include "sglang/server/output_mailbox.h"
#include "sglang/server/text_stop.h"

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
    OutputMailbox pending;
    std::optional<DecoderFinishReason> finish;
    std::exception_ptr error;
    BatchRequestId* request_id = nullptr;
    std::unique_ptr<NativeTextStop> text_stop;
    int cached_tokens = 0;
    bool admitted = false;
    bool cancelled = false;
    bool prefill_completed = false;
    bool buffer_decode_output = false;
    bool output_started = false;
    bool output_overlapped = false;
    bool health_check = false;

    bool aborted() const { return request_id && request_id->aborted(); }
    bool has_output() const { return !pending.empty() && !(output_overlapped && aborted()); }
    bool ready(bool buffered = false) const {
        return cancelled || aborted() || (buffered ? !pending.full() : pending.empty());
    }
    bool can_buffer() const {
        return media.empty() && prompt.size() >= decoder_batch_max_step_tokens &&
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
    bool buffered;
};
using Selection = std::array<bool, decoder_batch_max_lanes>;
using Steps = std::array<DecoderBatchStep, decoder_batch_max_lanes>;
struct BatchJob {
    std::shared_ptr<Job> job;
    std::size_t index = 0;
};
using BatchJobs = std::array<BatchJob, decoder_batch_max_lanes>;
constexpr auto disconnect_poll = std::chrono::milliseconds(10);
constexpr auto mailbox_grace = std::chrono::milliseconds(1);
constexpr int wide_decode_min_ready = 25;

}

struct BatchService::State {
    State(const Decoder& decoder, bool trace, bool cuda_graphs, int mtp_steps, int max_active,
          int prefix_cache_entries, int max_num_batched_tokens);
    ~State();

    void run(const Decoder& decoder, std::promise<void> started);
    void process(DecoderBatch& batch);
    std::optional<std::uint64_t> begin_iteration(DecoderBatch& batch);
    std::optional<Work> wait_for_jobs();
    bool idle_for_health() const;
    bool wide_decode_ready(bool buffered) const;
    bool prepare(DecoderBatch& batch, int slot, const std::shared_ptr<Job>& job, bool buffered);
    bool publish(int slot, Job& job, const DecoderBatchStep& step);
    void publish_steps(const Jobs& snapshot, const Selection& selected, const Steps& steps);
    void record_profile(std::uint64_t epoch, std::uint64_t elapsed, int width,
                        const Steps& steps, const DecoderBatchStepStats& stats);
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
    Selection generating{};
    Selection reserved{};
    IndexedAdmission indexed_admission;
    Clock::time_point last_generation = Clock::time_point::min();
    std::optional<HealthProbe> health_probe;
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
            if (job->request_id) {
                job->request_id->finish();
            }
            job->error = error;
            job.reset();
        }
    }
    changed.notify_all();
}

bool BatchService::State::wide_decode_ready(bool buffered) const {
    if (!cuda_graphs || mtp_steps != 0 || max_active != decoder_batch_max_lanes) {
        return false;
    }
    int ready = 0;
    for (const auto& job : jobs) {
        if (!job || job->cancelled || job->aborted() || !job->prefill_completed) {
            return false;
        }
        if (buffered && (!job->buffer_decode_output || !job->output_started)) {
            return false;
        }
        ready += job->ready(buffered);
    }
    return ready >= wide_decode_min_ready;
}

bool BatchService::State::idle_for_health() const {
    return health_probe && !reset_pending && Clock::now() < health_probe->deadline &&
           !health_probe->progressed && !health_probe->job->finish && !health_probe->job->error &&
           std::none_of(jobs.begin(), jobs.end(), [](const auto& job) { return bool(job); }) &&
           std::none_of(generating.begin(), generating.end(), [](bool active) { return active; });
}

std::optional<Work> BatchService::State::wait_for_jobs() {
    std::unique_lock lock(mutex);
    const auto end = jobs.begin() + max_active;
    changed.wait(lock, [&] {
        return closing || reset_pending || wide_decode_ready(true) || idle_for_health() ||
               std::any_of(jobs.begin(), end, [](const auto& job) {
                   return job && job->ready();
               });
    });
    const bool admitting = std::any_of(jobs.begin(), end, [](const auto& job) {
        return job && !job->admitted && !job->cancelled && !job->aborted();
    });
    changed.wait_for(lock, mailbox_grace, [&] {
        return closing || reset_pending || wide_decode_ready(true) || wide_decode_ready(false) ||
               std::all_of(jobs.begin(), end, [admitting](const auto& job) {
                   return job ? job->ready() : !admitting;
               });
    });
    if (closing) {
        return std::nullopt;
    }
    if (idle_for_health()) {
        jobs.front() = health_probe->job;
    }
    return Work{jobs, wide_decode_ready(true)};
}

bool BatchService::State::prepare(DecoderBatch& batch, int slot,
                                  const std::shared_ptr<Job>& job, bool buffered) {
    bool cancelled;
    {
        const std::lock_guard lock(mutex);
        if (!job || !job->ready(buffered)) {
            return false;
        }
        cancelled = job->cancelled || job->aborted();
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
            if (!job->request_id || job->request_id->finish()) {
                job->error = std::current_exception();
            } else {
                job->request_id->finish_abort();
                job->finish = DecoderFinishReason::aborted;
            }
            jobs[slot].reset();
            changed.notify_all();
            return false;
        }
    }
    return true;
}

bool BatchService::State::publish(int slot, Job& job, const DecoderBatchStep& step) {
    const bool terminal = step.finish_reason || step.sampling_failed || step.constraint_failed;
    if (terminal && job.request_id && !job.request_id->finish() && !job.cancelled) {
        job.request_id->finish_abort();
        job.finish = DecoderFinishReason::aborted;
        jobs[slot].reset();
        return false;
    }
    const bool published = (step.token || step.prompt_logprobs) &&
        !step.sampling_failed && !job.cancelled && !job.aborted();
    if (published) {
        const bool first_token = !std::exchange(job.prefill_completed, true);
        if (!job.health_check) {
            if (first_token) {
                counters.prompt_tokens += job.prompt_tokens;
                ++counters.prefill_requests;
            }
            counters.generation_tokens += step.token.has_value() + step.extra_token_count;
        }
        job.output_overlapped |= !job.pending.empty();
        job.pending.push(step);
    }
    if (step.sampling_failed || step.constraint_failed) {
        job.error = step.constraint_failed ? std::make_exception_ptr(DecoderConstraintError())
                                           : std::make_exception_ptr(DecoderSamplingError());
        jobs[slot].reset();
        return false;
    }
    if (step.finish_reason) {
        job.finish = step.finish_reason;
        jobs[slot].reset();
    }
    return published;
}

void BatchService::State::publish_steps(const Jobs& snapshot, const Selection& selected,
                                        const Steps& steps) {
    bool generated = false;
    for (int slot = 0; slot < max_active; ++slot) {
        if (selected[slot] && publish(slot, *snapshot[slot], steps[slot])) {
            generated = true;
        }
    }
    if (generated) {
        last_generation = Clock::now();
        if (health_probe && last_generation <= health_probe->deadline) {
            health_probe->progressed = true;
        }
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
        bool match_text = false;
        for (int slot = 0; slot < max_active; ++slot) {
            const auto& job = snapshot->jobs[slot];
            selected[slot] = prepare(batch, slot, job, snapshot->buffered);
            match_text |= selected[slot] && job->text_stop != nullptr;
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
                    trace_batch_details(steps, stats, max_active, max_num_batched_tokens,
                                        mtp_steps);
                }
                traced_bytes = bytes;
            }
        }
        if (match_text) {
            for (int slot = 0; slot < max_active; ++slot) {
                const auto& job = snapshot->jobs[slot];
                if (selected[slot] && job->text_stop && job->text_stop->apply(steps[slot])) {
                    batch.cancel(slot);
                }
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
            publish_steps(snapshot->jobs, selected, steps);
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
        const bool has_output = job.has_output();
        if (has_output && can_emit) {
            const auto step = job.pending.front();
            lock.unlock();
            const bool accepted = emit_step(step, callbacks, result);
            lock.lock();
            if (!accepted) {
                cancel(lock, slot, job);
                result.finish_reason = DecoderFinishReason::cancelled;
                return result;
            }
            job.pending.pop();
            job.output_started = true;
            changed.notify_all();
        } else if (job.error) {
            std::rethrow_exception(job.error);
        } else if (job.finish && can_emit) {
            result.finish_reason = *job.finish;
            return result;
        } else if (!has_output && !job.finish) {
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
                                        BatchJobs& active, std::size_t& next, bool buffered) {
    const bool indexed = prompts.size() > 1;
    if (indexed && next < prompts.size()) {
        std::uint32_t slots = 0;
        for (int slot = 0; slot < max_active; ++slot) {
            if (!active[slot].job) {
                slots |= std::uint32_t{1} << slot;
            }
        }
        indexed_admission.wait(owner, slots);
    }
    const int candidate_count = indexed ? max_active : 1;
    const auto now = indexed ? Clock::now() : Clock::time_point{};
    for (int offset = 0; offset < candidate_count && next < prompts.size(); ++offset) {
        while (next < prompts.size() && prompts[next].request_id &&
               prompts[next].request_id->aborted_waiting()) {
            ++next;
        }
        if (next == prompts.size()) {
            break;
        }
        const int slot = indexed ? offset : owner;
        if (jobs[slot] || active[slot].job || (slot != owner && reserved[slot]) ||
            (indexed && !indexed_admission.can_claim(owner, slot, now))) {
            continue;
        }
        auto job = std::make_shared<Job>();
        job->prompt = prompts[next].prompt;
        job->prompt_tokens = job->prompt.size();
        job->options = prompts[next].options;
        job->options.mtp_steps = mtp_steps;
        job->request_id = prompts[next].request_id;
        job->buffer_decode_output = buffered && job->can_buffer();
        const auto& text_stops = prompts[next].text_stops;
        if (text_stops) {
            job->text_stop = std::make_unique<NativeTextStop>(text_stops);
        }
        if (job->request_id && !job->request_id->dispatch()) {
            break;
        }
        active[slot] = {job, next++};
        jobs[slot] = std::move(job);
        if (indexed) {
            indexed_admission.claimed(owner, slot);
        }
    }
    if (next == prompts.size()) {
        indexed_admission.pause(owner);
    }
}

bool BatchService::State::drain_waiting_aborts(
    std::unique_lock<std::mutex>& lock, int owner, const std::vector<BatchPrompt>& prompts,
    const BatchCallbacks& callbacks, std::vector<DecoderResult>& results,
    std::size_t& completed) {
    for (std::size_t index = 0; index < prompts.size(); ++index) {
        auto* id = prompts[index].request_id;
        auto& result = results[index];
        if (!id || !id->aborted_waiting() || result.finish_reason == DecoderFinishReason::aborted) {
            continue;
        }
        id->finish_abort();
        result.finish_reason = DecoderFinishReason::aborted;
        if (callbacks.on_output) {
            indexed_admission.pause(owner);
            lock.unlock();
            const bool written = callbacks.on_output(index, result, true);
            lock.lock();
            if (!written) {
                return false;
            }
        }
        ++completed;
    }
    return true;
}

bool BatchService::State::drain_batch(std::unique_lock<std::mutex>& lock, int owner,
                                      BatchJobs& active,
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
        if (job.finish == DecoderFinishReason::cancelled) {
            return false;
        }
        const bool has_step = job.has_output();
        const auto finish = has_step && job.pending.size > 1 ? std::nullopt : job.finish;
        if (!has_step && !finish) {
            continue;
        }
        auto& result = results[item.index];
        if (has_step) {
            emit_step(job.pending.front(), collect, result);
        }
        result.cached_prompt_tokens = job.cached_tokens;
        if (finish) {
            result.finish_reason = *finish;
            if (*finish == DecoderFinishReason::stop && job.text_stop) {
                result.matched_text_stop = job.text_stop->matched;
            }
        }
        if (callbacks.on_output) {
            indexed_admission.pause(owner);
            lock.unlock();
            const bool written = callbacks.on_output(item.index, result, finish.has_value());
            lock.lock();
            if (!written) {
                return false;
            }
        }
        if (has_step) {
            job.pending.pop();
            job.output_started = true;
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
    const bool buffered = prompts.size() == 1 && callbacks.buffer_decode_output;
    while (completed < prompts.size()) {
        const bool poll_stream = announced && callbacks.on_stream_poll;
        if (callbacks.is_cancelled || poll_stream) {
            indexed_admission.poll(owner, Clock::now() + disconnect_poll);
            lock.unlock();
            const bool connected = (!callbacks.is_cancelled || !callbacks.is_cancelled()) &&
                (!poll_stream || callbacks.on_stream_poll());
            lock.lock();
            if (!connected) {
                return false;
            }
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
            const bool all_aborted = std::all_of(prompts.begin(), prompts.end(),
                [](const auto& prompt) {
                    return prompt.request_id && prompt.request_id->aborted();
                });
            if (admitted || all_aborted) {
                indexed_admission.pause(owner);
                lock.unlock();
                announced = callbacks.on_admitted();
                lock.lock();
                if (!announced) {
                    return false;
                }
            }
        }
        if (announced || !callbacks.on_admitted) {
            if (!drain_batch(lock, owner, active, callbacks, results, completed) ||
                !drain_waiting_aborts(lock, owner, prompts, callbacks, results, completed)) {
                return false;
            }
        }
        const auto submitted = next;
        enqueue_batch(owner, prompts, active, next, buffered);
        if (next != submitted) {
            changed.notify_all();
        }
        if (completed < prompts.size() && (!buffered || std::none_of(active.begin(), active.end(),
            [](const auto& item) { return item.job && item.job->has_output(); }))) {
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

bool BatchService::ready() const { return !state_->failed.load(); }
bool BatchService::supports_generated_logprobs() const { return state_->mtp_steps == 0; }

bool BatchService::wait_for_generation(Clock::time_point started, Clock::time_point deadline,
                                       const std::function<bool()>& cancelled) {
    auto& state = *state_;
    std::unique_lock lock(state.mutex);
    if (state.closing || state.failure || state.health_probe || cancelled()) {
        return false;
    }
    if (state.last_generation > started && state.last_generation <= deadline) {
        return true;
    }
    if (Clock::now() >= deadline) {
        return false;
    }

    auto job = std::make_shared<Job>();
    job->prompt = {0};
    job->health_check = true;
    state.health_probe.emplace(HealthProbe{job, deadline});
    state.changed.notify_all();
    bool healthy = false;
    try {
        while (!state.closing && !state.failure && !cancelled()) {
            if (state.health_probe->progressed) {
                healthy = true;
                break;
            }
            const auto now = Clock::now();
            if (now >= deadline || job->error || job->finish) {
                break;
            }
            state.changed.wait_until(lock, std::min(deadline, now + disconnect_poll));
        }
    } catch (...) {
        job->cancelled = true;
        state.health_probe.reset();
        state.changed.notify_all();
        throw;
    }
    job->cancelled = true;
    state.health_probe.reset();
    state.changed.notify_all();
    return healthy;
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

BatchRuntimeState BatchService::runtime_state() const {
    const std::lock_guard lock(state_->mutex);
    const int running = std::count_if(state_->jobs.begin(), state_->jobs.end(),
        [](const auto& job) { return job && job->admitted && !job->health_check; });
    return {ready(), running, state_->counters.generation_tokens};
}

bool BatchService::reset_prefix_cache() {
    std::unique_lock lock(state_->mutex);
    if (state_->failure) {
        std::rethrow_exception(state_->failure);
    }
    const auto& first = state_->jobs.front();
    if (state_->reset_pending || (first && first->health_check) ||
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
                                     std::vector<DecoderMedia> media, std::size_t expanded_tokens,
                                     BatchRequestId* request_id) {
    if (slot < 0 || slot >= state_->max_active) {
        throw std::out_of_range("Invalid decoder batch service slot");
    }
    auto job = std::make_shared<Job>();
    job->prompt = prompt;
    job->prompt_tokens = media.empty() ? prompt.size() : expanded_tokens;
    job->options = options;
    job->request_id = request_id;
    if (media.empty()) {
        job->options.mtp_steps = state_->mtp_steps;
    }
    job->media = std::move(media);
    job->buffer_decode_output = callbacks.buffer_decode_output && job->can_buffer();
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
    generating = state_->reserved[slot] = true;
    try {
        while (queue && !state_->failure && !job->aborted()) {
            lock.unlock();
            const bool cancelled = callbacks.is_cancelled && callbacks.is_cancelled();
            lock.lock();
            if (cancelled) {
                generating = state_->reserved[slot] = false;
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
        if (request_id && !request_id->dispatch()) {
            request_id->finish_abort();
            generating = state_->reserved[slot] = false;
            state_->changed.notify_all();
            DecoderResult result;
            result.finish_reason = DecoderFinishReason::aborted;
            return result;
        }
        queue = job;
        state_->changed.notify_all();
        auto result = state_->consume(lock, slot, *job, callbacks);
        result.cached_prompt_tokens = job->cached_tokens;
        generating = state_->reserved[slot] = false;
        state_->changed.notify_all();
        return result;
    } catch (...) {
        if (!lock.owns_lock()) {
            lock.lock();
        }
        state_->cancel(lock, slot, *job);
        generating = state_->reserved[slot] = false;
        state_->changed.notify_all();
        throw;
    }
}

void BatchService::notify_abort() {
    const std::lock_guard lock(state_->mutex);
    state_->changed.notify_all();
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
    state_->reserved[slot] = prompts.size() == 1;
    bool completed;
    try {
        completed = state_->consume_batch(lock, slot, prompts, callbacks, active, results);
    } catch (...) {
        if (!lock.owns_lock()) {
            lock.lock();
        }
        state_->indexed_admission.finish(slot);
        state_->cancel_batch(lock, active);
        generating = state_->reserved[slot] = false;
        state_->changed.notify_all();
        throw;
    }
    state_->indexed_admission.finish(slot);
    if (!completed) {
        state_->cancel_batch(lock, active);
    }
    generating = state_->reserved[slot] = false;
    state_->changed.notify_all();
    if (completed) {
        return results;
    }
    return std::nullopt;
}

}
