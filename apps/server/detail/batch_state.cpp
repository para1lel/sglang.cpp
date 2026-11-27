#include "sglang/server/detail/batch_service_state.h"

#include <utility>

namespace sglang::server {

BatchRequestId::BatchRequestId(std::string value) : value(std::move(value)) {}

bool BatchRequestId::dispatch() {
    auto expected = State::waiting;
    return state_.compare_exchange_strong(expected, State::dispatched);
}

bool BatchRequestId::abort(bool dispatched_only) {
    auto expected = state_.load();
    while (expected == State::dispatched ||
           (!dispatched_only && expected == State::waiting)) {
        const auto target = expected == State::waiting ? State::aborted_waiting :
                                                         State::aborted_running;
        if (state_.compare_exchange_weak(expected, target)) {
            return true;
        }
    }
    return false;
}

bool BatchRequestId::claim_completion() {
    auto expected = State::dispatched;
    return state_.compare_exchange_strong(expected, State::finishing);
}

bool BatchRequestId::finish() {
    auto expected = state_.load();
    while (expected == State::dispatched || expected == State::finishing) {
        if (state_.compare_exchange_weak(expected, State::finished)) return true;
    }
    return false;
}

void BatchRequestId::finish_abort() {
    const auto state = state_.load();
    state_.store(state == State::aborted_waiting ? State::finished_waiting_abort :
                                                State::finished_running_abort);
}

bool BatchRequestId::finished() const {
    const auto state = state_.load();
    return state == State::finished || state == State::finished_waiting_abort ||
           state == State::finished_running_abort;
}

bool BatchRequestId::aborted() const {
    const auto state = state_.load();
    return state == State::aborted_waiting || state == State::aborted_running ||
           state == State::finished_waiting_abort || state == State::finished_running_abort;
}

bool BatchRequestId::aborted_waiting() const {
    const auto state = state_.load();
    return state == State::aborted_waiting || state == State::finished_waiting_abort;
}

std::size_t BatchPrompt::prompt_tokens() const {
    return media.empty() ? prompt.size() : expanded_tokens;
}

void IndexedAdmission::wait(int owner, std::uint32_t slots) {
    auto& entry = waiting_[owner];
    if (entry.ticket == 0) {
        entry.ticket = ++last_ticket_;
    }
    entry.slots = slots;
    entry.eligible_until = Clock::time_point::max();
}

bool IndexedAdmission::can_claim(int owner, int slot, Clock::time_point now) const {
    const auto ticket = waiting_[owner].ticket;
    const auto bit = std::uint32_t{1} << slot;
    for (const auto& entry : waiting_) {
        if ((entry.slots & bit) != 0 && entry.ticket < ticket && now < entry.eligible_until) {
            return false;
        }
    }
    return true;
}

void IndexedAdmission::claimed(int owner, int slot) {
    auto& entry = waiting_[owner];
    entry.slots &= ~(std::uint32_t{1} << slot);
    entry.ticket = ++last_ticket_;
}

void IndexedAdmission::poll(int owner, Clock::time_point deadline) {
    waiting_[owner].eligible_until = deadline;
}

void IndexedAdmission::pause(int owner) { waiting_[owner].slots = 0; }

void IndexedAdmission::finish(int owner) { waiting_[owner] = Entry{}; }

bool OutputMailbox::empty() const { return size == 0; }

bool OutputMailbox::full() const { return size == steps.size(); }

const DecoderBatchStep& OutputMailbox::front() const { return steps[head]; }

void OutputMailbox::push(const DecoderBatchStep& step) {
    steps[(head + size) % steps.size()] = step;
    ++size;
}

void OutputMailbox::pop() {
    head = (head + 1) % steps.size();
    --size;
}

bool emit_step(const DecoderBatchStep& step, const DecoderCallbacks& callbacks,
               DecoderResult& result) {
    if (step.prompt_logprobs) {
        result.prompt_logprobs = step.prompt_logprobs;
    }
    if (step.label_logprobs) {
        result.label_logprobs = step.label_logprobs;
    }
    if (!step.token) return true;
    result.mtp_rounds += step.mtp_drafted_tokens != 0;
    result.mtp_drafted_tokens += step.mtp_drafted_tokens;
    const auto emit = [&](int token, const auto& logprobs, bool accepted) {
        result.output_token_ids.push_back(token);
        if (logprobs) {
            result.generated_logprobs.push_back(logprobs);
        }
        result.mtp_accepted_tokens += accepted;
        if (logprobs && callbacks.on_scored_token) {
            return callbacks.on_scored_token(token, logprobs);
        }
        return !callbacks.on_token || callbacks.on_token(token);
    };
    if (!emit(*step.token, step.generated_logprobs, step.mtp_accepted_tokens > 0)) {
        return false;
    }
    for (int index = 0; index < step.extra_token_count; ++index) {
        if (callbacks.is_cancelled && callbacks.is_cancelled()) {
            return false;
        }
        if (!emit(step.extra_tokens[index], step.extra_generated_logprobs[index],
                  index + 1 < step.mtp_accepted_tokens)) {
            return false;
        }
    }
    return true;
}

namespace batch_service_detail {

Job::~Job() {
    if (restore_join) restore_join->fail();
}

bool Job::aborted() const { return request_id && request_id->aborted(); }

bool Job::discard_deferred_store() const {
    return prefix_pending && !prefix_operation.valid() && (cancelled || aborted());
}

bool Job::has_output() const { return !pending.empty() && !(output_overlapped && aborted()); }

bool Job::can_overlap_mtp_output() const {
    return options.mtp_steps > 0 && buffer_decode_output && output_started;
}

bool Job::ready(bool buffered) const {
    return !prefix_pending && (cancelled || aborted() || (!cache_reading &&
        (buffered || can_overlap_mtp_output() ? !pending.full() : pending.empty())));
}

bool Job::can_buffer(bool per_token) const {
    const bool eligible = options.mtp_steps > 0
        ? per_token && !options.requires_target_only() && !options.label_token_ids
        : prompt.size() >= decoder_batch_max_step_tokens;
    return eligible && media.empty() &&
        !options.samples() && !options.constraint && !options.eager && options.top_logits == 0;
}

}

bool BatchService::State::prefix_ready(const Job& job) const {
    if (!job.prefix_pending) return false;
    if (!job.prefix_operation.valid()) {
        return job.discard_deferred_store() || (lmcache && lmcache->store_capacity());
    }
    return job.prefix_operation.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

}
