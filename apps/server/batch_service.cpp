#include "sglang/server/detail/batch_service_state.h"
#include "sglang/server/detail/batch_trace.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace sglang::server {

BatchService::State::State(std::unique_ptr<BatchExecutor> owner, bool trace, int mtp_steps,
                           std::unique_ptr<decoder::LmCacheClient> client, std::string identity)
    : trace(trace), cuda_graphs(owner->options().cuda_graphs), mtp_steps(mtp_steps),
      max_active(owner->options().max_active),
      prefix_cache_entries(owner->options().prefix_cache_entries),
      max_num_batched_tokens(owner->options().max_num_batched_tokens),
      resident_prefix_cache(owner->options().resident_prefix_cache), cache_identity(identity),
      executor(std::move(owner)) {
    if (max_active < 1 || max_active > decoder_batch_max_lanes) {
        throw std::invalid_argument("Invalid decoder batch service capacity");
    }
    counters.cache_capacity = prefix_cache_entries;
    if (client) lmcache = std::make_unique<BatchCache>(std::move(client), std::move(identity),
                                                      max_active);
}

BatchService::State::State(std::unique_ptr<BatchExecutor> owner, bool trace, int mtp_steps,
                           const decoder::LmCacheConfig& cache,
                           std::vector<std::array<std::uint8_t, 16>> devices, std::string identity)
    : State(std::move(owner), trace, mtp_steps, {}, identity) {
    if (devices.size() < 2) {
        throw std::invalid_argument("Ranked LMCache requires at least two rank devices");
    }
    lmcache = std::make_unique<BatchCache>(cache, std::move(devices), std::move(identity),
                                           max_active);
}

void BatchService::State::start() {
    executor->start(
        [this](BatchSession& batch) {
            const std::scoped_lock lock(mutex);
            counters.cache = batch.prefix_cache_stats();
            session = &batch;
        },
        [this](BatchSession& batch) {
            try {
                process(batch);
            } catch (...) {
                fail_jobs(std::current_exception(), true);
            }
            if (lmcache) {
                lmcache->stop_restores();
                lmcache->drain_transfers();
            }
            try {
                batch.release_retired_allocations();
            } catch (...) {
                fail_jobs(std::current_exception(), true);
            }
            const std::scoped_lock lock(mutex);
            session = nullptr;
            counters.cache = batch.prefix_cache_stats();
        });
}

BatchService::State::~State() {
    {
        const std::lock_guard lock(mutex);
        closing = true;
        if (reset_pending) {
            reset_pending->set_exception(
                std::make_exception_ptr(CacheResetUnavailable("Prefix cache service is closing")));
            reset_pending.reset();
        }
        for (auto& job : jobs) if (job) cancel_cache_read(*job);
    }
    if (lmcache) lmcache->stop_restores();
    changed.notify_all();
    executor->join();
    lmcache.reset();
}

void BatchService::State::fail_jobs(std::exception_ptr error, bool permanent) {
    const std::lock_guard lock(mutex);
    if (permanent) {
        failure = error;
        failed.store(true);
        if (reset_pending) {
            reset_pending->set_exception(error);
            reset_pending.reset();
        }
    }
    for (auto& job : jobs) {
        if (job && (!job->retiring_prefix || permanent)) {
            cancel_cache_read(*job);
            job->deferred_finish.reset();
            if (job->request_id && !job->request_id->finish() && !job->cancelled &&
                job->can_overlap_mtp_output()) {
                job->request_id->finish_abort();
                job->finish = DecoderFinishReason::aborted;
            } else {
                job->error = error;
            }
            if (!permanent && job->prefix_operation.valid()) {
                job->prefix_pending = true;
                job->retiring_prefix = true;
                job->request_id = nullptr;
            } else {
                job.reset();
            }
        }
    }
    changed.notify_all();
}

bool BatchService::State::idle_for_health() const {
    return health_probe && !reset_pending && Clock::now() < health_probe->deadline &&
           !health_probe->progressed && !health_probe->job->finish && !health_probe->job->error &&
           std::none_of(jobs.begin(), jobs.end(), [](const auto& job) { return bool(job); }) &&
           std::none_of(generating.begin(), generating.end(), [](bool active) { return active; });
}

bool BatchService::State::has_runnable_job(Clock::time_point& retry_at) const {
    retry_at = Clock::time_point::max();
    for (const int slot : admission) {
        const auto& job = jobs[slot];
        if (!job)
            continue;
        const bool capacity_blocked = retry_at != Clock::time_point::max();
        if (prefix_ready(*job) || job->ready(false, capacity_blocked))
            return true;
        if (!capacity_blocked && job->awaiting_capacity())
            retry_at = job->admission_retry_at;
    }
    return false;
}

std::optional<Work> BatchService::State::wait_for_jobs(BatchSession& batch) {
    std::unique_lock lock(mutex);
    const auto* end = jobs.data() + max_active;
    Clock::time_point retry_at;
    const auto ready = [&] {
        return closing || reset_pending || decode_cohort_ready(true) || idle_for_health() ||
               (trace && lmcache && batch.prefix_cache_stats().stores != counters.cache.stores) ||
               has_runnable_job(retry_at);
    };
    if (std::none_of(jobs.cbegin(), end, [](const auto& job) { return bool(job); }) &&
        !changed.wait_for(lock, std::chrono::seconds(5), ready)) {
        lock.unlock();
        batch.release_retired_allocations();
        lock.lock();
        counters.cache = batch.prefix_cache_stats();
    }
    while (!ready()) {
        if (retry_at == Clock::time_point::max()) {
            changed.wait(lock);
        } else if (changed.wait_until(lock, retry_at) == std::cv_status::timeout) {
            lock.unlock();
            batch.release_retired_allocations();
            lock.lock();
        }
    }
    bool admitting = false;
    for (const int slot : admission) {
        const auto& job = jobs[slot];
        if (!job)
            continue;
        if (job->awaiting_capacity())
            break;
        admitting = !job->admitted && !job->cancelled && !job->aborted() && job->ready();
        if (admitting)
            break;
    }
    if (!changed.wait_for(lock, mailbox_grace, [&] {
            return closing || reset_pending || decode_cohort_ready(true, true) ||
                   decode_cohort_ready(false) ||
                   std::all_of(jobs.cbegin(), end, [admitting](const auto& job) {
                       return job ? job->pending.empty() || job->output_grace_expired ||
                                        job->ready()
                                  : !admitting;
                   });
        })) {
        for (const auto& job : jobs) {
            if (job && !job->pending.empty() && !job->ready())
                job->output_grace_expired = true;
        }
        ++counters.mailbox_grace_timeouts;
    }
    if (closing) return std::nullopt;
    if (idle_for_health()) jobs.front() = health_probe->job;
    return Work{jobs, admission, decode_cohort_ready(true)};
}

std::uint32_t BatchService::State::refresh_prefill_roster(Work& work, int capacity) {
    const std::unique_lock lock(mutex, std::try_to_lock);
    if (!lock || closing || reset_pending) return 0;
    std::uint32_t added = 0;
    for (const int slot : admission) {
        if (capacity == 0) break;
        const auto& job = jobs[slot];
        if (job && job != work.jobs[slot] && !job->health_check && !job->admitted &&
            !job->cache_reading && !job->cancelled && !job->aborted()) {
            work.jobs[slot] = job;
            added |= std::uint32_t{1} << slot;
            --capacity;
        }
    }
    work.admission = admission;
    return added;
}
bool BatchService::State::prepare(BatchSession& batch, int slot, const std::shared_ptr<Job>& job,
                                  bool buffered, bool capacity_blocked) {
    bool cancelled;
    {
        const std::lock_guard lock(mutex);
        if (!job || !job->ready(buffered, capacity_blocked)) {
            return false;
        }
        cancelled = job->cancelled || job->aborted();
    }
    if (cancelled) {
        if (job->admitted) {
            batch.cancel(slot);
        }
        const std::lock_guard lock(mutex);
        cancel_cache_read(*job);
        job->admission_retry_at = {};
        publish(slot, *job, {std::nullopt, DecoderFinishReason::cancelled});
        changed.notify_all();
        return false;
    }
    return job->admitted || admit(batch, slot, *job);
}

bool BatchService::State::publish(int slot, Job& job, const DecoderBatchStep& step) {
    const bool terminal = step.finish_reason || step.sampling_failed || step.constraint_failed;
    if (terminal && job.request_id && !job.request_id->finish() && !job.cancelled) {
        job.request_id->finish_abort();
        job.finish = DecoderFinishReason::aborted;
        jobs[slot].reset();
        return false;
    }
    const bool published = (step.token || step.prompt_logprobs || step.label_logprobs) &&
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
                                        Steps& steps) {
    bool generated = false;
    for (int slot = 0; slot < max_active; ++slot) {
        if (!selected[slot]) continue;
        auto& job = *snapshot[slot];
        auto& step = steps[slot];
        job.prefix_pending = step.prefix_pending;
        if (job.prefix_pending) {
            job.deferred_finish = std::exchange(step.finish_reason, {});
        }
        if (publish(slot, job, step)) {
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

void BatchService::State::process(BatchSession& batch) {
    const bool cache_enabled =
        lmcache != nullptr || resident_prefix_cache || prefix_cache_entries != 0;
    BatchTrace trace_state(trace && cache_enabled, batch.prefix_cache_stats());
    while (auto snapshot = wait_for_jobs(batch)) {
        const auto iteration = begin_iteration(batch);
        if (!iteration) {
            continue;
        }
        const auto epoch = *iteration;
        Selection selected{};
        bool cold = true, has_empty_slot = false, starts_long_prefill = false;
        bool capacity_blocked = false;
        for (const int slot : snapshot->admission) {
            const auto& job = snapshot->jobs[slot];
            if (!job) {
                has_empty_slot = true;
                continue;
            }
            cold &= !job->admitted;
            if (job->prefix_pending && !poll_prefix(batch, slot, job)) {
                continue;
            }
            selected[slot] = prepare(batch, slot, job, snapshot->buffered, capacity_blocked);
            capacity_blocked |= job->admission_retry_at != Clock::time_point{};
            starts_long_prefill |= cold && selected[slot] && !job->health_check &&
                job->prompt_tokens >=
                    static_cast<std::size_t>(job->cached_tokens) + max_num_batched_tokens;
        }
        int width = std::count(selected.begin(), selected.end(), true);
        const int capacity = std::min(max_active, max_num_batched_tokens) - width;
        if (cold && starts_long_prefill && has_empty_slot && !capacity_blocked && capacity > 0) {
            const auto added = refresh_prefill_roster(*snapshot, capacity);
            for (const int slot : snapshot->admission) {
                if (!(added >> slot & 1U)) continue;
                const auto& job = snapshot->jobs[slot];
                selected[slot] = prepare(batch, slot, job, snapshot->buffered, capacity_blocked);
                capacity_blocked |= job->admission_retry_at != Clock::time_point{};
                width += selected[slot];
            }
        }
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
                if (!batch.available()) throw;
                fail_jobs(std::current_exception(), false);
                if (trace) trace_state.failed(batch.request_device_allocation_bytes());
                continue;
            }
        }
        if (trace) {
            trace_state.step(width, steps, stats, batch.request_device_allocation_bytes(),
                             max_active, max_num_batched_tokens, mtp_steps);
        }
        const auto cache = cache_enabled ? batch.prefix_cache_stats() : DecoderPrefixCacheStats{};
        trace_state.cache(cache, steps);
        {
            const std::lock_guard lock(mutex);
            counters.cache = cache;
            if (epoch && profile && epoch == profile_epoch &&
                profile->steps.size() != profile_limit && width) {
                profile->steps.push_back(batch_profile_step(elapsed, width, steps, stats));
            }
            publish_steps(snapshot->jobs, selected, steps);
        }
        changed.notify_all();
    }
}

void BatchService::State::cancel(std::unique_lock<std::mutex>& lock, int slot, Job& job) {
    if (jobs[slot].get() != &job || job.retiring_prefix) {
        return;
    }
    job.cancelled = !job.completion_claimed;
    if (job.cancelled) cancel_cache_read(job);
    changed.notify_all();
    changed.wait(lock, [&] { return jobs[slot].get() != &job; });
}

DecoderResult BatchService::State::consume(std::unique_lock<std::mutex>& lock, int slot,
                                           Job& job, const DecoderCallbacks& callbacks) {
    DecoderResult result;
    bool announced = false;
    const bool notify_admission = callbacks.on_admitted || callbacks.on_prompt_admitted;
    for (;;) {
        const bool announce = job.admitted && !job.error && notify_admission && !announced;
        const auto cached = announce ? job.cached_tokens : 0;
        const bool poll = announced && callbacks.on_stream_poll && !job.error && !job.finish;
        lock.unlock();
        const bool connected = (!callbacks.is_cancelled || !callbacks.is_cancelled()) &&
            (!announce || (callbacks.on_prompt_admitted ? callbacks.on_prompt_admitted(cached) :
                callbacks.on_admitted())) && (!poll || callbacks.on_stream_poll());
        lock.lock();
        announced = announced || announce;
        if (!connected) {
            cancel(lock, slot, job);
            result.finish_reason = DecoderFinishReason::cancelled;
            return result;
        }
        const bool can_emit = announced || !job.admitted || !notify_admission;
        if (job.has_output() && can_emit) {
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
            job.output_grace_expired = false;
            changed.notify_all();
        } else if (job.error) {
            std::rethrow_exception(job.error);
        } else if (job.finish && can_emit) {
            result.finish_reason = *job.finish;
            return result;
        } else if (!job.has_output() && !job.finish &&
                   (announced || !notify_admission || !job.admitted)) {
            changed.wait_for(lock, disconnect_poll);
        }
    }
}

void BatchService::State::cancel_batch(std::unique_lock<std::mutex>& lock,
                                       const BatchJobs& active) {
    const auto owned = [&](int slot) {
        return active[slot].job && jobs[slot] == active[slot].job &&
            !jobs[slot]->retiring_prefix;
    };
    for (int slot = 0; slot < max_active; ++slot) {
        if (owned(slot)) {
            jobs[slot]->cancelled = !jobs[slot]->completion_claimed;
            if (jobs[slot]->cancelled) cancel_cache_read(*jobs[slot]);
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
    if (reset_pending)
        return;
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
        job->media = prompts[next].media;
        job->prompt_tokens = prompts[next].prompt_tokens();
        job->options = prompts[next].options;
        job->options.mtp_steps = mtp_steps;
        job->request_id = prompts[next].request_id;
        job->buffer_decode_output = buffered && job->can_buffer();
        const auto& text_stops = prompts[next].text_stops;
        if (text_stops) {
            job->text_stop = std::make_unique<TokenTextStop>(
                std::shared_ptr<const TextStops>(text_stops, &text_stops->stops));
            job->options.text_stop = [current = job.get(), tokenizer = &text_stops->tokenizer]
                                     (const int* ids, int count, bool terminal) {
                return current->text_stop->match_native(*tokenizer, ids, count, terminal,
                                                       current->matched_text_stop);
            };
        }
        if (job->request_id && !job->request_id->dispatch()) {
            break;
        }
        active[slot] = {job, next++};
        admission.admitted(slot);
        jobs[slot] = std::move(job);
        begin_cache_read(slot, jobs[slot]);
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
            if (*finish == DecoderFinishReason::stop) {
                result.matched_text_stop = job.matched_text_stop;
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
            job.output_grace_expired = false;
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

BatchService::BatchService(std::unique_ptr<BatchExecutor> executor, bool trace, int mtp_steps)
    : BatchService(std::move(executor), trace, mtp_steps, {}, {}) {}

BatchService::BatchService(std::unique_ptr<BatchExecutor> executor, bool trace, int mtp_steps,
                           std::unique_ptr<decoder::LmCacheClient> cache, std::string identity) {
    if (!executor) throw std::invalid_argument("Missing decoder batch executor");
    state_ = std::make_unique<State>(std::move(executor), trace, mtp_steps, std::move(cache),
                                     std::move(identity));
    state_->start();
}

BatchService::BatchService(std::unique_ptr<BatchExecutor> executor, bool trace, int mtp_steps,
                           const decoder::LmCacheConfig& cache,
                           std::vector<std::array<std::uint8_t, 16>> rank_devices,
                           std::string identity) {
    if (!executor) throw std::invalid_argument("Missing decoder batch executor");
    state_ = std::make_unique<State>(std::move(executor), trace, mtp_steps, cache,
                                     std::move(rank_devices), std::move(identity));
    state_->start();
}

BatchService::~BatchService() = default;

void BatchService::stop_cache_restores() noexcept {
    if (state_->lmcache) state_->lmcache->stop_restores();
}

bool BatchService::ready() const { return !state_->failed.load(); }

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
    auto metrics = state_->counters;
    for (const auto& job : state_->jobs) {
        if (!job || job->admitted || job->cancelled || job->aborted())
            continue;
        metrics.admission_waiting_for_capacity += static_cast<int>(job->awaiting_capacity());
        metrics.admission_waiting_for_lookup += static_cast<int>(job->cache_reading);
        metrics.admission_waiting_for_restore +=
            static_cast<int>(job->prefix_pending && !job->retiring_prefix);
    }
    if (auto* cache = state_->lmcache.get()) {
        const auto* session = state_->session;
        const bool backup_pending = session != nullptr && session->prefix_backup_pending();
        metrics.lmcache = cache->metrics();
        if (backup_pending && metrics.lmcache.pending_stores == 0)
            metrics.lmcache.pending_stores = 1;
        metrics.cache.stores = metrics.lmcache.stores;
    }
    if (tp_ar_source_ != nullptr) {
        metrics.tp_ar = tp_ar_source_->tp_ar_counters();
        metrics.tp_ar_present = true;
    }
    return metrics;
}

void BatchService::set_tp_ar_source(const Decoder* decoder) { tp_ar_source_ = decoder; }

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
    if (state_->closing || state_->reset_pending ||
        std::any_of(state_->jobs.begin(), state_->jobs.end(),
                    [](const auto& job) { return bool(job); }) ||
        std::any_of(state_->generating.begin(), state_->generating.end(),
                    [](bool active) { return active; })) {
        return false;
    }
    state_->reset_pending = std::make_shared<std::promise<void>>();
    auto completion = state_->reset_pending->get_future();
    state_->changed.notify_all();
    lock.unlock();
    completion.get();
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
    job->options.mtp_steps = state_->mtp_steps;
    job->request_id = request_id;
    job->media = std::move(media);
    job->buffer_decode_output = callbacks.buffer_decode_output && job->can_buffer(true);
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
        state_->admission.admitted(slot);
        queue = job;
        state_->begin_cache_read(slot, job);
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

BatchTopology BatchService::topology() const noexcept { return state_->executor->topology(); }
}
