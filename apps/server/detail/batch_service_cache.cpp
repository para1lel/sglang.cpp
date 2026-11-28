#include "sglang/server/detail/batch_service_state.h"

#include "sglang/runtime/cache/prefix_cache_layout.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace sglang::server {
namespace {

std::vector<std::shared_ptr<decoder::PrefixCacheKeys>>
make_keys(const batch_service_detail::Job& job, int world, std::string_view identity) {
    std::string domain = std::to_string(identity.size()) + ':';
    domain.append(identity).append(job.options.prefix_cache_salt);
    std::vector<std::shared_ptr<decoder::PrefixCacheKeys>> keys;
    keys.reserve(static_cast<std::size_t>(world));
    for (int rank = 0; rank < world; ++rank) {
        const auto layout = decoder::tp2_prefix_cache_layout(rank);
        keys.push_back(std::make_shared<decoder::PrefixCacheKeys>(job.prompt, domain, &job.media,
                                                                  world == 1 ? nullptr : &layout));
    }
    return keys;
}

}

void BatchService::State::begin_cache_read(int slot, const std::shared_ptr<Job>& job) {
    if (!lmcache || job->health_check) return;
    const auto& options = job->options;
    const int max_restore = options.score_prompt
        ? std::min(options.prompt_logprob_start, static_cast<int>(job->prompt_tokens) - 1)
        : std::numeric_limits<int>::max();
    const bool draft = options.mtp_steps > 0 && !options.requires_target_only() &&
                       !options.label_token_ids;
    auto cancelled = std::make_shared<std::atomic_bool>(false);
    const std::weak_ptr<Job> weak = job;
    lmcache->lookup([weak, world = lmcache->world_size()](std::string_view identity) {
        const auto current = weak.lock();
        return current ? make_keys(*current, world, identity)
                       : std::vector<std::shared_ptr<decoder::PrefixCacheKeys>>{};
    }, max_restore, draft, cancelled, [this, slot, weak](BatchCacheRead result) {
        const auto job = weak.lock();
        if (!job) return;
        const std::lock_guard lock(mutex);
        if (closing || jobs[slot] != job || job->cancelled || job->aborted()) return;
        job->cache_keys = std::move(result.keys);
        job->cache_read = std::make_shared<BatchCacheRead>(std::move(result));
        job->cache_reading = false;
        changed.notify_all();
    });
    job->cache_cancel = std::move(cancelled);
    job->cache_reading = true;
}

void BatchService::State::cancel_cache_read(Job& job) {
    if (job.cache_cancel) {
        job.cache_cancel->store(true);
        lmcache->wake_readers();
        job.cache_cancel.reset();
    }
}

std::optional<std::uint64_t> BatchService::State::begin_iteration(BatchSession& batch) {
    std::unique_lock lock(mutex);
    if (const auto reset = reset_pending) {
        lock.unlock();
        std::exception_ptr error;
        try {
            if (lmcache)
                lmcache->drain_transfers();
            batch.reset_prefix_cache();
            if (lmcache)
                lmcache->reset();
        } catch (...) {
            error = std::current_exception();
        }
        const auto stats = batch.prefix_cache_stats();
        lock.lock();
        counters.cache = stats;
        if (reset_pending == reset) {
            if (error)
                reset->set_exception(error);
            else
                reset->set_value();
            reset_pending.reset();
        }
        changed.notify_all();
        return std::nullopt;
    }
    return profile && profile->steps.size() < profile_limit ? profile_epoch : 0;
}

bool BatchService::State::admit(BatchSession& batch, int slot, Job& job) {
    try {
        if (lmcache != nullptr || resident_prefix_cache || prefix_cache_entries != 0) {
            configure_checkpoint(job);
            job.options.claim_completion = [this, current = &job] {
                const std::scoped_lock lock(mutex);
                if (current->cancelled || closing)
                    return false;
                current->completion_claimed =
                    current->request_id == nullptr || current->request_id->claim_completion();
                return current->completion_claimed;
            };
        }
        if (job.media.empty()) {
            job.cached_tokens = batch.admit(slot, std::move(job.prompt), job.options);
        } else {
            job.cached_tokens =
                batch.admit_multimodal(slot, std::move(job.prompt), job.options, job.media);
            job.media.clear();
        }
        job.options.prefix_cache_read.reset();
        const std::scoped_lock lock(mutex);
        job.admission_retry_at = {};
        job.prefix_pending = job.prefix_operation.valid();
        job.admitted = !job.prefix_pending;
        if (job.admitted)
            job.cache_read.reset();
        job.cache_cancel.reset();
        changed.notify_all();
        return job.admitted;
    } catch (...) {
        if (!batch.available())
            throw;
        const auto error = std::current_exception();
        if (defer_capacity_retry(batch, job, error))
            return false;
        const std::scoped_lock lock(mutex);
        job.admission_retry_at = {};
        if (job.request_id == nullptr || job.request_id->finish()) {
            job.error = error;
        } else {
            job.request_id->finish_abort();
            job.finish = DecoderFinishReason::aborted;
        }
        jobs[slot].reset();
        changed.notify_all();
        return false;
    }
}

bool BatchService::State::defer_capacity_retry(BatchSession& batch, Job& job,
                                               std::exception_ptr error) {
    if (!resident_prefix_cache)
        return false;
    try {
        std::rethrow_exception(std::move(error));
    } catch (const DecoderCapacityError& capacity) {
        if (!capacity.retryable)
            return false;
    } catch (...) {
        return false;
    }
    const auto cache = batch.prefix_cache_stats();
    const std::scoped_lock lock(mutex);
    const bool active = std::any_of(jobs.begin(), jobs.end(), [](const auto& current) {
        return current != nullptr &&
               (current->admitted || current->prefix_pending || current->retiring_prefix);
    });
    if (!active && cache.pending_backups == 0 && job.admission_retry_at != Clock::time_point{}) {
        return false;
    }
    job.admission_retry_at = Clock::now() + disconnect_poll;
    ++counters.resident_admission_retries;
    return true;
}

void BatchService::State::configure_checkpoint(Job& job) {
    if (job.health_check)
        return;
    if (!lmcache && resident_prefix_cache && job.cache_keys.empty()) {
        job.cache_keys = make_keys(job, executor->topology().tp_size, cache_identity);
    }
    if (job.cache_keys.empty()) return;
    job.options.prefix_cache_keys = job.cache_keys;
    if (!lmcache || !job.cache_read)
        return;
    job.options.prefix_cache_backing = lmcache->backing();
    job.options.prefix_cache_read = std::shared_ptr<const decoder::PrefixCacheRead>(
        job.cache_read, &job.cache_read->match);
    const auto notify = [this] {
        const std::scoped_lock lock(mutex);
        changed.notify_all();
    };
    job.options.restore_prefix = [this, current = &job, notify](
            const decoder::PrefixDeviceTransfer& view) {
        if (!current->restore_join) current->restore_join = lmcache->begin_restore();
        auto completion = lmcache->restore(current->restore_join, current->cache_keys, view,
                                           notify);
        const std::scoped_lock lock(mutex);
        current->prefix_operation = completion;
        return completion;
    };
}

bool BatchService::State::poll_prefix(BatchSession& batch, int slot,
                                      const std::shared_ptr<Job>& job) {
    {
        const std::lock_guard lock(mutex);
        if (job->discard_deferred_store()) {
            job->prefix_pending = false;
            return true;
        }
        if (!prefix_ready(*job)) return false;
    }
    DecoderBatchPrefixUpdate update;
    try {
        update = batch.poll_prefix(slot);
        if (update.pending) return false;
    } catch (...) {
        if (!batch.available()) throw;
        const auto error = std::current_exception();
        batch.cancel(slot);
        const std::lock_guard lock(mutex);
        job->prefix_pending = false;
        job->prefix_operation = {};
        job->deferred_finish.reset();
        if (job->request_id && !job->request_id->finish()) {
            job->request_id->finish_abort();
            job->finish = DecoderFinishReason::aborted;
        } else {
            job->error = error;
        }
        jobs[slot].reset();
        changed.notify_all();
        return false;
    }
    const std::lock_guard lock(mutex);
    job->prefix_pending = false;
    job->prefix_operation = {};
    if (job->retiring_prefix) {
        jobs[slot].reset();
        changed.notify_all();
        return false;
    }
    if (!job->admitted) {
        job->cached_tokens = update.restored.value_or(0);
        job->admitted = true;
        job->cache_read.reset();
        changed.notify_all();
    }
    if (job->deferred_finish) {
        publish(slot, *job, {std::nullopt, std::exchange(job->deferred_finish, {})});
        changed.notify_all();
        return false;
    }
    return true;
}

}
