#include "sglang/runtime/server/batch_service_state.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace sglang::server {

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
    lmcache->lookup([weak](std::string_view identity) {
        const auto current = weak.lock();
        if (!current) return std::shared_ptr<decoder::PrefixCacheKeys>{};
        std::string domain = std::to_string(identity.size()) + ':';
        domain.append(identity).append(current->options.prefix_cache_salt);
        return std::make_shared<decoder::PrefixCacheKeys>(current->prompt, domain,
                                                          &current->media);
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

void BatchService::State::configure_checkpoint(Job& job) {
    if (!job.cache_keys) return;
    job.options.prefix_cache_read = std::shared_ptr<const decoder::PrefixCacheRead>(
        job.cache_read, &job.cache_read->match);
    job.options.publish_prefix_checkpoints =
        job.cache_keys->tokens() == job.cache_keys->cacheable_tokens();
    auto notify = [this] {
        const std::lock_guard lock(mutex);
        changed.notify_all();
    };
    job.options.restore_prefix = [this, current = &job, notify](
            const decoder::PrefixDeviceTransfer& view) {
        auto completion = lmcache->restore(current->cache_keys, view, notify);
        const std::lock_guard lock(mutex);
        current->prefix_operation = completion;
        return completion;
    };
    job.options.reserve_prefix_store = [this](std::shared_ptr<void>& reservation) {
        return lmcache->reserve_store(reservation);
    };
    job.options.store_prefix = [this, current = &job, notify](
            decoder::PrefixDeviceTransfer&& view, const int* generated, int count,
            std::shared_ptr<void> reservation, std::shared_ptr<std::atomic_uint64_t> stores) {
        auto& keys = *current->cache_keys;
        for (int i = 0; i < count; ++i) keys.append(generated[i]);
        lmcache->store(keys, std::move(view), std::move(reservation), std::move(stores), notify);
    };
}

bool BatchService::State::poll_prefix(BatchSession& batch, int slot,
                                      const std::shared_ptr<Job>& job) {
    {
        const std::lock_guard lock(mutex);
        if (job->discard_deferred_store()) {
            job->prefix_pending = false;
            return true;  // Let ordinary cancellation retire the unsubmitted checkpoint.
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
