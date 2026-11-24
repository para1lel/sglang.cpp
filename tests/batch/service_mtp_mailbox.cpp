#include "sglang/runtime/server/single_device_batch_executor.h"
#include "sglang/server/batch_service.h"
#include "sglang/server/text_stop.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {
using namespace sglang;
using namespace sglang::server;
using namespace std::chrono_literals;
using Selection = std::array<bool, decoder_batch_max_lanes>;
using Steps = std::array<DecoderBatchStep, decoder_batch_max_lanes>;
using Finish = DecoderFinishReason;

enum class Outcome { length, abort, disconnect, failure };
enum class Arrival { none, during_admit, during_advance, aborted, disconnected, blocked_writer };

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::abort();
    }
}

struct Barrier {
    bool entered = false;
    bool released = false;
};

struct Fixture {
    std::mutex mutex;
    std::condition_variable changed;
    Barrier seed, writer, peer, fault;
    Barrier before_claim, after_claim;
    Barrier prefill_admit, prefill_advance;
    std::array<Barrier, decoder_batch_max_lanes> consumers{};
    std::array<int, decoder_batch_max_lanes> admissions{};
    std::vector<int> admission_order;
    std::vector<Selection> prefill_grants;
    bool inspect_prefill = false;
    int max_lanes = 4;
    int hold_prefill_slot = 0;
    int fail_admission_slot = -1;
    int primary_steps = 0;
    int fail_at = 0;
    int cancels = 0;
    int terminal_stores = 0;
    bool claim_terminal = false;
    bool fail_after_claim = false;
    bool permanent = false;
    bool unavailable = false;

    template<class Predicate>
    void await(std::unique_lock<std::mutex>& lock, Predicate ready,
               const char* message = "Timed out at a decoder or writer barrier") {
        require(changed.wait_for(lock, 5s, ready), message);
    }

    void hold(std::unique_lock<std::mutex>& lock, Barrier& barrier) {
        barrier.entered = true;
        changed.notify_all();
        await(lock, [&] { return barrier.released; });
    }

    void release(Barrier& barrier) {
        barrier.released = true;
        changed.notify_all();
    }
};

Fixture* fixture;

std::vector<int> prefix(std::size_t length) {
    std::vector<int> ids(length);
    std::iota(ids.begin(), ids.end(), 100);
    return ids;
}

template<class T>
void expect_fault(std::future<T>& result) {
    try {
        static_cast<void>(result.get());
        require(false, "Decoder failure did not reach the request");
    } catch (const std::runtime_error& error) {
        require(std::string_view(error.what()) == "fixture step failure",
                "Unexpected decoder error");
    }
}

void run_arrival(BatchService& service) {
    const auto peer = service.generate(2, {300}, {}, {});
    require(peer.output_token_ids == std::vector{300} && peer.finish_reason == Finish::length,
            "Blocked writer stalled a new arrival");
}

void check_scalar_output(Outcome outcome) {
    Fixture observed;
    fixture = &observed;
    if (outcome == Outcome::failure) {
        observed.fail_at = 3;
        observed.fault.released = true;
    }
    const Decoder decoder({}, 0, true);
    BatchService service(std::make_unique<SingleDeviceBatchExecutor>(decoder,
        BatchExecutor::Options{true, 4, 0, decoder_default_max_num_batched_tokens}), false, 4);
    BatchRequestId id("scalar-mailbox");
    std::vector<int> delivered;
    DecoderOptions options;
    options.max_new_tokens = 16;
    DecoderCallbacks callbacks;
    callbacks.buffer_decode_output = true;
    callbacks.on_token = [&](int token) {
        delivered.push_back(token);
        std::unique_lock lock(observed.mutex);
        if (token == 100) observed.hold(lock, observed.seed);
        if (token == 102) {
            observed.hold(lock, observed.writer);
            return outcome != Outcome::disconnect;
        }
        return true;
    };
    auto request = std::async(std::launch::async, [&] {
        return service.generate(0, {100}, options, callbacks, {}, 0, &id);
    });
    {
        std::unique_lock lock(observed.mutex);
        observed.await(lock, [&] { return observed.seed.entered; });
    }
    run_arrival(service);
    {
        std::unique_lock lock(observed.mutex);
        require(observed.primary_steps == 1, "First acknowledgement did not gate lookahead");
        observed.release(observed.seed);
        observed.await(lock, [&] { return observed.writer.entered && observed.primary_steps >= 3; },
                       "No lookahead behind a claimed MTP round");
    }
    if (outcome != Outcome::failure) run_arrival(service);
    {
        std::unique_lock lock(observed.mutex);
        require(observed.primary_steps == 3, "Full mailbox allowed another lookahead step");
        if (outcome == Outcome::abort) {
            require(id.abort(true), "Named abort did not win active-request arbitration");
            lock.unlock();
            service.notify_abort();
            lock.lock();
            observed.await(lock, [&] { return observed.cancels == 1; });
        }
        observed.release(observed.writer);
    }
    if (outcome == Outcome::failure) {
        expect_fault(request);
        require(delivered == prefix(6), "Worker failure dropped the acknowledged prefix");
    } else {
        const auto result = request.get();
        const auto finish = outcome == Outcome::abort ? Finish::aborted :
            outcome == Outcome::disconnect ? Finish::cancelled : Finish::length;
        const auto count = outcome == Outcome::abort ? 6 : outcome == Outcome::disconnect ? 3 : 16;
        const auto accepted = outcome == Outcome::abort ? 4U :
            outcome == Outcome::disconnect ? 2U : 12U;
        require(result.finish_reason == finish && delivered == prefix(count) &&
                    result.output_token_ids == delivered && result.mtp_accepted_tokens == accepted,
                "Wrong scalar prefix, acceptance count, or terminal reason");
    }
    require(observed.cancels == (outcome == Outcome::abort || outcome == Outcome::disconnect) &&
                service.generate(0, {100}, {}, {}).output_token_ids == prefix(1),
            "Unexpected retirement or slot reuse after scalar output");
}

void check_backpressure(DecoderOptions options, bool opt_in) {
    Fixture observed;
    fixture = &observed;
    const Decoder decoder({}, 0, true);
    BatchService service(std::make_unique<SingleDeviceBatchExecutor>(decoder,
        BatchExecutor::Options{true, 4, 0, decoder_default_max_num_batched_tokens}), false, 4);
    options.max_new_tokens = 16;
    DecoderCallbacks callbacks;
    callbacks.buffer_decode_output = opt_in;
    callbacks.on_token = [&](int token) {
        if (token == 101) {
            std::unique_lock lock(observed.mutex);
            observed.hold(lock, observed.writer);
        }
        return true;
    };
    auto request = std::async(std::launch::async, [&] {
        return service.generate(0, {100}, options, callbacks);
    });
    {
        std::unique_lock lock(observed.mutex);
        observed.await(lock, [&] { return observed.writer.entered; });
    }
    run_arrival(service);
    {
        const std::lock_guard lock(observed.mutex);
        require(observed.primary_steps == 2, "Ineligible scalar request overlapped its writer");
        observed.release(observed.writer);
    }
    const auto result = request.get();
    require(result.output_token_ids == prefix(16) && result.finish_reason == Finish::length &&
                result.mtp_drafted_tokens == (opt_in ? 0U : 12U), "Wrong ineligible scalar result");
}

void check_step_delivery(bool stop, bool permanent) {
    Fixture observed;
    fixture = &observed;
    observed.permanent = permanent;
    observed.fault.released = true;
    const Decoder decoder({}, 0, true);
    const Tokenizer tokenizer("unused");
    BatchService service(std::make_unique<SingleDeviceBatchExecutor>(decoder,
        BatchExecutor::Options{true, 4, 0, decoder_default_max_num_batched_tokens}), false, 4);
    BatchPrompt prompt;
    prompt.prompt = {100};
    prompt.options.max_new_tokens = 16;
    prompt.text_stops = std::make_shared<NativeTextStops>(NativeTextStops{
        tokenizer, TextStops({stop ? "STOP" : "unreachable"})});
    std::vector<DecoderResult> delivered;
    bool terminal = false;
    BatchCallbacks callbacks;
    callbacks.buffer_decode_output = true;
    callbacks.on_admitted = [] { return true; };
    callbacks.on_stream_poll = [&] {
        if (delivered.size() == 2 && !observed.writer.entered) {
            std::unique_lock lock(observed.mutex);
            observed.hold(lock, observed.writer);
        }
        return true;
    };
    callbacks.on_output = [&](std::size_t index, const DecoderResult& result, bool done) {
        require(index == 0 && !terminal, "Bad step stream index or extra terminal frame");
        delivered.push_back(result);
        terminal = done;
        return true;
    };
    auto request = std::async(std::launch::async, [&] {
        return service.generate_batch(0, {prompt}, callbacks);
    });
    {
        std::unique_lock lock(observed.mutex);
        observed.await(lock, [&] {
            return observed.writer.entered && observed.primary_steps >= 3;
        });
    }
    run_arrival(service);
    {
        const std::lock_guard lock(observed.mutex);
        require(observed.primary_steps == 3, "Step stream overlapped an unacknowledged output");
        if (!stop) observed.fail_at = 4;
        observed.release(observed.writer);
    }
    if (stop) {
        const auto result = request.get();
        require(result && result->size() == 1 && terminal &&
                    result->front().finish_reason == Finish::stop &&
                    result->front().matched_text_stop == 0,
                "Missing matched text stop or terminal frame");
    } else {
        expect_fault(request);
        require(!terminal, "Step stream returned a false terminal frame");
    }
    require(delivered.size() == 3 && delivered[0].output_token_ids == prefix(1) &&
                delivered[1].output_token_ids == prefix(6) &&
                delivered[2].output_token_ids == prefix(stop ? 9 : 11) &&
                delivered[2].mtp_drafted_tokens == 8 &&
                delivered[2].mtp_accepted_tokens == (stop ? 7U : 8U),
            "Step stream lost or miscounted the delivered prefix");
}

void check_abort_fault(bool permanent) {
    Fixture observed;
    fixture = &observed;
    observed.fail_at = 3;
    observed.permanent = permanent;
    const Decoder decoder({}, 0, true);
    BatchService service(std::make_unique<SingleDeviceBatchExecutor>(decoder,
        BatchExecutor::Options{true, 4, 0, decoder_default_max_num_batched_tokens}), false, 4);
    BatchRequestId id("abort-during-fault");
    DecoderOptions options;
    options.max_new_tokens = 16;
    DecoderCallbacks peer_callbacks;
    peer_callbacks.on_token = [&](int token) {
        require(token == 200, "Nonaborted peer advanced while its writer was blocked");
        std::unique_lock lock(observed.mutex);
        observed.hold(lock, observed.peer);
        return true;
    };
    auto peer = std::async(std::launch::async, [&] {
        return service.generate(1, {200}, options, peer_callbacks);
    });
    {
        std::unique_lock lock(observed.mutex);
        observed.await(lock, [&] { return observed.peer.entered; });
    }
    DecoderCallbacks callbacks;
    callbacks.buffer_decode_output = true;
    callbacks.on_token = [&](int token) {
        if (token == 102) {
            std::unique_lock lock(observed.mutex);
            observed.hold(lock, observed.writer);
        }
        return true;
    };
    auto request = std::async(std::launch::async, [&] {
        return service.generate(0, {100}, options, callbacks, {}, 0, &id);
    });
    {
        std::unique_lock lock(observed.mutex);
        observed.await(lock, [&] { return observed.writer.entered && observed.fault.entered; });
    }
    require(id.abort(true), "Named abort did not win active-request arbitration");
    service.notify_abort();
    {
        const std::lock_guard lock(observed.mutex);
        observed.release(observed.fault);
        observed.release(observed.peer);
    }
    expect_fault(peer);
    {
        const std::lock_guard lock(observed.mutex);
        require(id.finished() && id.aborted(), "Failure stranded the winning abort ID");
        observed.release(observed.writer);
    }
    const auto result = request.get();
    require(result.finish_reason == Finish::aborted && result.output_token_ids == prefix(6) &&
                result.mtp_accepted_tokens == 4 && result.mtp_drafted_tokens == 4,
            "Failure replaced the abort or dropped the output prefix");
    require(service.runtime_state().running_requests == 0 && service.ready() != permanent,
            "Worker failure left active lanes or wrong health");
    if (permanent) {
        auto rejected = std::async(std::launch::async, [&] {
            return service.generate(0, {100}, {}, {});
        });
        expect_fault(rejected);
    } else {
        require(service.generate(0, {100}, {}, {}).output_token_ids == prefix(1),
                "Recoverable failure prevented slot reuse");
    }
}

void check_completion_claim(bool abort_first, bool fail_after_claim) {
    Fixture observed;
    fixture = &observed;
    observed.claim_terminal = true;
    observed.fail_after_claim = fail_after_claim;
    const Decoder decoder({}, 0, true);
    BatchService service(std::make_unique<SingleDeviceBatchExecutor>(decoder,
        BatchExecutor::Options{true, 4, 1, decoder_default_max_num_batched_tokens}), false, 4);
    BatchRequestId id("terminal-claim");
    auto request = std::async(std::launch::async, [&] {
        return service.generate(0, {100}, {}, {}, {}, 0, &id);
    });
    {
        std::unique_lock lock(observed.mutex);
        observed.await(lock, [&] { return observed.before_claim.entered; });
        if (abort_first) require(id.abort(true), "Abort lost before terminal claim");
        observed.release(observed.before_claim);
        observed.await(lock, [&] { return observed.after_claim.entered; });
        require(!id.finished(), "Request ID became reusable before terminal publication");
        if (!abort_first) require(!id.abort(true), "Abort won after terminal claim");
        require(observed.terminal_stores == 0, "Terminal store preceded the claim barrier");
        observed.release(observed.after_claim);
    }
    if (fail_after_claim) {
        expect_fault(request);
    } else {
        const auto result = request.get();
        require(result.finish_reason == (abort_first ? Finish::aborted : Finish::length),
                "Wrong completion reason after claim arbitration");
        require(result.output_token_ids == (abort_first ? std::vector<int>{} : prefix(1)),
                "Losing terminal output was published");
    }
    require(id.finished() && id.aborted() == abort_first, "Request ID did not settle");
    require(observed.terminal_stores == (!abort_first && !fail_after_claim),
            "Only the successful claimed completion may store the terminal endpoint");
}

std::future<DecoderResult> start_prefill_request(Fixture& observed, BatchService& service,
                                                Arrival arrival, int slot, BatchRequestId* id) {
    return std::async(std::launch::async, [&observed, &service, arrival, slot, id] {
        DecoderOptions options;
        options.max_new_tokens = 1;
        DecoderCallbacks callbacks;
        callbacks.is_cancelled = [&] {
            std::unique_lock lock(observed.mutex);
            observed.hold(lock, observed.consumers[slot]);
            return false;
        };
        if (slot == 1 && arrival == Arrival::disconnected) {
            callbacks.on_admitted = [&] {
                std::unique_lock lock(observed.mutex);
                observed.hold(lock, observed.writer);
                return false;
            };
        }
        if (slot == 1 && arrival == Arrival::blocked_writer) {
            callbacks.on_token = [&](int) {
                std::unique_lock lock(observed.mutex);
                observed.hold(lock, observed.writer);
                return true;
            };
        }
        auto prompt = prefix(decoder_default_max_num_batched_tokens);
        prompt.front() += slot;
        return service.generate(slot, prompt, options, callbacks, {}, 0, id);
    });
}

void check_prefill_output(const DecoderResult& result, int token) {
    require(result.finish_reason == Finish::length &&
            result.output_token_ids == std::vector{token},
            "Prefill owner did not receive exactly one output");
}

void check_prefill_roster(Arrival arrival, bool trace = false) {
    Fixture observed;
    fixture = &observed;
    observed.inspect_prefill = true;
    const bool in_flight = arrival == Arrival::during_advance;
    observed.prefill_admit.released = in_flight || arrival == Arrival::none;
    observed.prefill_advance.released = arrival == Arrival::none;
    if (arrival == Arrival::none) observed.consumers[0].released = true;
    const Decoder decoder({}, 0, true);
    BatchService service(std::make_unique<SingleDeviceBatchExecutor>(decoder,
        BatchExecutor::Options{true, 4, 1, decoder_default_max_num_batched_tokens}), trace, 4);
    BatchRequestId late_id("late-prefill");
    auto first = start_prefill_request(observed, service, arrival, 0, nullptr);
    std::future<DecoderResult> late;
    if (arrival != Arrival::none) {
        {
            std::unique_lock lock(observed.mutex);
            const auto& first_barrier = in_flight ? observed.prefill_advance :
                                                    observed.prefill_admit;
            observed.await(lock, [&] { return first_barrier.entered; });
        }
        late = start_prefill_request(observed, service, arrival, 1, &late_id);
        {
            std::unique_lock lock(observed.mutex);
            observed.await(lock, [&] { return observed.consumers[0].entered &&
                                              observed.consumers[1].entered; });
        }
        if (arrival == Arrival::aborted) {
            require(late_id.abort(true), "Late prefill abort did not win");
            service.notify_abort();
        }
        require(!service.reset_prefix_cache(), "Cache reset took ownership of a live prefill slot");
        {
            std::unique_lock lock(observed.mutex);
            if (!in_flight) {
                observed.release(observed.prefill_admit);
                observed.await(lock, [&] { return observed.prefill_advance.entered; });
            }
            observed.release(observed.consumers[0]);
            observed.release(observed.consumers[1]);
            if (arrival == Arrival::disconnected) {
                observed.await(lock, [&] { return observed.writer.entered; });
                observed.release(observed.writer);
            }
            observed.release(observed.prefill_advance);
        }
    }
    check_prefill_output(first.get(), 100);
    if (arrival == Arrival::blocked_writer) {
        std::unique_lock lock(observed.mutex);
        observed.await(lock, [&] { return observed.writer.entered; });
        require(late.wait_for(0s) == std::future_status::timeout,
                "Blocked writer retired before its output was delivered");
        observed.release(observed.writer);
    }
    if (late.valid()) {
        const auto result = late.get();
        if (arrival == Arrival::aborted || arrival == Arrival::disconnected) {
            const auto reason = arrival == Arrival::aborted ? Finish::aborted : Finish::cancelled;
            require(result.finish_reason == reason && result.output_token_ids.empty(),
                    "Cancelled prefill owner published output or settled incorrectly");
        } else {
            check_prefill_output(result, 101);
        }
        require(late_id.finished(), "Late request ID remained live after retirement");
    }
    const bool joined = arrival != Arrival::none && arrival != Arrival::aborted && !in_flight;
    {
        const std::lock_guard lock(observed.mutex);
        require(observed.prefill_grants.size() == (in_flight ? 2U : 1U) &&
                observed.prefill_grants.front() == Selection{true, joined},
                "A prefill grant selected the wrong roster");
        if (in_flight) require(observed.prefill_grants.back() == Selection{false, true},
                               "In-flight arrival did not select its own next grant");
        require(observed.admissions[0] == 1 && observed.admissions[1] == (joined || in_flight) &&
                observed.cancels == 0, "A roster member was admitted or retired twice");
    }
    if (!trace) {
        require(!service.runtime_state().running_requests && service.reset_prefix_cache(),
                "Retired prefill roster did not release cache ownership");
        DecoderOptions options;
        options.max_new_tokens = 1;
        check_prefill_output(service.generate(1, {200}, options, {}), 200);
    }
}

void check_prefill_publication_order() {
    Fixture observed;
    fixture = &observed;
    observed.inspect_prefill = true;
    observed.max_lanes = 8;
    observed.prefill_admit.released = true;
    observed.consumers[1].released = true;
    const Decoder decoder({}, 0, true);
    BatchService service(std::make_unique<SingleDeviceBatchExecutor>(decoder,
        BatchExecutor::Options{true, 8, 0, decoder_default_max_num_batched_tokens}), false, 4);
    auto blocker = start_prefill_request(observed, service, Arrival::none, 1, nullptr);
    {
        std::unique_lock lock(observed.mutex);
        observed.await(lock, [&] { return observed.prefill_advance.entered; });
    }
    auto older = start_prefill_request(observed, service, Arrival::none, 7, nullptr);
    {
        std::unique_lock lock(observed.mutex);
        observed.await(lock, [&] { return observed.consumers[7].entered; });
    }
    auto newer = start_prefill_request(observed, service, Arrival::none, 0, nullptr);
    {
        std::unique_lock lock(observed.mutex);
        observed.await(lock, [&] { return observed.consumers[0].entered; });
        observed.release(observed.consumers[7]);
        observed.release(observed.consumers[0]);
        observed.release(observed.prefill_advance);
    }
    check_prefill_output(blocker.get(), 101);
    check_prefill_output(older.get(), 107);
    check_prefill_output(newer.get(), 100);
    require(observed.admission_order == std::vector{1, 7, 0},
            "Worker admitted newer slot 0 before published slot 7");
}

void check_prefill_replacement_order(bool fail_admission) {
    Fixture observed;
    fixture = &observed;
    observed.inspect_prefill = true;
    observed.hold_prefill_slot = 2;
    observed.fail_admission_slot = fail_admission ? 0 : -1;
    observed.consumers[1].released = true;
    observed.consumers[2].released = true;
    const Decoder decoder({}, 0, true);
    BatchService service(std::make_unique<SingleDeviceBatchExecutor>(decoder,
        BatchExecutor::Options{true, 4, 0, decoder_default_max_num_batched_tokens}), false, 4);
    auto blocker = start_prefill_request(observed, service, Arrival::none, 1, nullptr);
    {
        std::unique_lock lock(observed.mutex);
        observed.await(lock, [&] { return observed.prefill_advance.entered; });
    }
    BatchRequestId old_id("retiring-cold-slot");
    auto old = start_prefill_request(observed, service, Arrival::none, 0, &old_id);
    {
        std::unique_lock lock(observed.mutex);
        observed.await(lock, [&] { return observed.consumers[0].entered; });
    }
    auto head = start_prefill_request(observed, service, Arrival::none, 2, nullptr);
    if (!fail_admission) {
        require(old_id.abort(true), "Old cold slot did not accept abort");
        service.notify_abort();
    }
    {
        std::unique_lock lock(observed.mutex);
        observed.await(lock, [&] { return observed.consumers[2].entered; });
        observed.release(observed.prefill_advance);
        observed.await(lock, [&] { return observed.prefill_admit.entered; });
        observed.release(observed.consumers[0]);
    }
    if (fail_admission) {
        expect_fault(old);
    } else {
        require(old.get().finish_reason == Finish::aborted, "Old cold slot did not retire");
    }
    {
        std::lock_guard lock(observed.mutex);
        observed.consumers[0] = {};
    }
    auto older = start_prefill_request(observed, service, Arrival::none, 0, nullptr);
    {
        std::unique_lock lock(observed.mutex);
        observed.await(lock, [&] { return observed.consumers[0].entered; });
    }
    auto newer = start_prefill_request(observed, service, Arrival::none, 3, nullptr);
    {
        std::unique_lock lock(observed.mutex);
        observed.await(lock, [&] { return observed.consumers[3].entered; });
        observed.release(observed.consumers[0]);
        observed.release(observed.consumers[3]);
        observed.release(observed.prefill_admit);
    }
    check_prefill_output(blocker.get(), 101);
    check_prefill_output(head.get(), 102);
    check_prefill_output(older.get(), 100);
    check_prefill_output(newer.get(), 103);
    require(observed.admission_order == std::vector{1, 2, 0, 3} &&
                observed.prefill_grants[1] == Selection{true, false, true, true},
            "Cold refresh admitted a newer job before a replacement in the same turn");
}

void check_prefill_trace() {
    std::ostringstream output;
    auto* const previous = std::cerr.rdbuf(output.rdbuf());
    check_prefill_roster(Arrival::none, true);
    std::cerr.rdbuf(previous);
    const std::string cache = "prefix cache entries=0 pages=0 device_bytes=0 hits=0"
        " cached_tokens=0 prefill_tokens=0 stores=0 evictions=0\n";
    std::string rows = "batch prefill_rows=0";
    for (int slot = 1; slot < decoder_batch_max_mtp_lanes; ++slot) rows += ",0";
    require(output.str() == cache + "batch width=1 decode=0 request_bytes=0\n" + rows +
            " workspace_bytes=0\nbatch target_rows=0 step_budget=" +
            std::to_string(decoder_default_max_num_batched_tokens) +
            "\nbatch verify lanes=0 rows=0 peak_request_bytes=0\n" + cache,
            "Prefill trace records or ordering changed");
}
}

namespace sglang {
struct Decoder::Model {};
Decoder::Decoder(const std::filesystem::path&, int, bool) {}
Decoder::~Decoder() = default;
struct Tokenizer::Model {};
Tokenizer::Tokenizer(const std::filesystem::path&) {}
Tokenizer::~Tokenizer() = default;
std::string_view Tokenizer::piece(TokenId id, bool) const { return id == 108 ? "STOP" : "x"; }

struct DecoderBatch::State {
    struct Lane {
        int base = 0;
        int produced = 0;
        int remaining = 0;
        bool mtp = false;
        std::function<int(const int*, int, bool)> text_stop;
        std::function<bool()> claim_completion;
    };
    std::array<Lane, decoder_batch_max_lanes> lanes{};
};

DecoderBatch::DecoderBatch(const Decoder&, DecoderBatchOptions options)
    : state_(std::make_unique<State>()) {
    require(options.max_lanes == fixture->max_lanes && options.cuda_graphs,
            "Unexpected worker configuration");
}
DecoderBatch::~DecoderBatch() = default;
int DecoderBatch::admit(int slot, std::vector<int> prompt, const DecoderOptions& options) {
    auto& lane = state_->lanes[slot];
    require((prompt.size() == 1 || fixture->inspect_prefill) && !lane.remaining,
            "Bad or reused live slot");
    if (slot == fixture->fail_admission_slot) {
        fixture->fail_admission_slot = -1;
        throw std::runtime_error("fixture step failure");
    }
    lane = {prompt.front(), 0, options.max_new_tokens,
            options.mtp_steps > 0 && !options.requires_target_only() && !options.label_token_ids,
            options.text_stop, options.claim_completion};
    if (fixture->inspect_prefill) {
        std::unique_lock lock(fixture->mutex);
        if (slot == fixture->hold_prefill_slot && !fixture->admissions[slot]) {
            fixture->hold(lock, fixture->prefill_admit);
        }
        ++fixture->admissions[slot];
        fixture->admission_order.push_back(slot);
    }
    return 0;
}
int DecoderBatch::admit_multimodal(int, std::vector<int>, const DecoderOptions&,
                                  const std::vector<DecoderMedia>&) { std::abort(); }
void DecoderBatch::cancel(int slot) {
    require(std::exchange(state_->lanes[slot].remaining, 0) > 0, "Cancel of a retired slot");
    const std::lock_guard lock(fixture->mutex);
    ++fixture->cancels;
    fixture->changed.notify_all();
}

void publish_fixture_checkpoint(const std::function<bool()>& claim, DecoderBatchStep& step) {
    std::unique_lock lock(fixture->mutex);
    fixture->hold(lock, fixture->before_claim);
    lock.unlock();
    const bool allowed = claim();
    lock.lock();
    fixture->hold(lock, fixture->after_claim);
    if (!allowed) {
        step.finish_reason = Finish::cancelled;
    } else if (fixture->fail_after_claim) {
        throw std::runtime_error("fixture step failure");
    } else {
        ++fixture->terminal_stores;
    }
}

Steps DecoderBatch::advance_chunked(const Selection& selected, DecoderBatchStepStats*) {
    if (fixture->inspect_prefill) {
        std::unique_lock lock(fixture->mutex);
        fixture->prefill_grants.push_back(selected);
        if (fixture->prefill_grants.size() == 1) fixture->hold(lock, fixture->prefill_advance);
    }
    Steps steps{};
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        if (!selected[slot]) continue;
        auto& lane = state_->lanes[slot];
        require(lane.remaining > 0, "Selected an inactive lane");
        if (slot == 0 && lane.produced + lane.remaining == 16) {
            std::unique_lock lock(fixture->mutex);
            if (++fixture->primary_steps == fixture->fail_at) {
                fixture->hold(lock, fixture->fault);
                fixture->unavailable = fixture->permanent;
                state_->lanes = {};
                throw std::runtime_error("fixture step failure");
            }
            fixture->changed.notify_all();
        }
        auto& step = steps[slot];
        step.decode = lane.produced > 0;
        const bool speculative = step.decode && lane.mtp && lane.remaining >= 5;
        const int available = speculative ? 5 : 1;
        std::array<int, decoder_max_mtp_steps + 1> ids;
        for (int index = 0; index < available; ++index) {
            ids[index] = lane.base + lane.produced + index;
        }
        const bool terminal = lane.remaining == available;
        const int cutoff = lane.text_stop ? lane.text_stop(ids.data(), available, terminal) : 0;
        const int count = cutoff ? cutoff : available;
        step.token = ids.front();
        lane.produced += count;
        lane.remaining -= count;
        if (speculative) {
            step.mtp_drafted_tokens = 4;
            step.extra_token_count = count - 1;
            step.mtp_accepted_tokens = std::min(4, count);
            std::copy_n(ids.begin() + 1, step.extra_token_count, step.extra_tokens.begin());
        }
        if (cutoff) {
            step.finish_reason = Finish::stop;
            lane.remaining = 0;
        } else if (terminal) {
            step.finish_reason = Finish::length;
        }
        if (step.finish_reason && fixture->claim_terminal) {
            publish_fixture_checkpoint(lane.claim_completion, step);
        }
    }
    return steps;
}

bool DecoderBatch::available() const { return !fixture->unavailable; }
std::size_t DecoderBatch::request_device_allocation_bytes() const { return 0; }
DecoderPrefixCacheStats DecoderBatch::prefix_cache_stats() const { return {}; }
void DecoderBatch::reset_prefix_cache() {}
void DecoderBatch::release_retired_allocations() {}
}

int main(int argc, char** argv) {
    const std::string_view mode = argc == 2 ? argv[1] : "all";
    require(argc <= 2 && (mode == "all" || mode == "scalar" || mode == "backpressure" ||
                          mode == "step" || mode == "abort" || mode == "prefill" ||
                          mode == "trace"), "Unknown mailbox scenario");
    if (mode == "all" || mode == "scalar") {
        for (auto outcome : {Outcome::length, Outcome::abort,
                             Outcome::disconnect, Outcome::failure}) {
            check_scalar_output(outcome);
        }
        std::cout << "Scalar mailbox gating, bounded FIFO, and settlement passed\n";
    }
    if (mode == "all" || mode == "backpressure") {
        check_backpressure({}, false);
        DecoderOptions options;
        options.logit_bias = std::make_shared<const std::vector<DecoderLogitBias>>(
            std::vector<DecoderLogitBias>{{101, 1.0F}});
        check_backpressure(options, true);
        options = {};
        options.frequency_penalty = 1.0;
        check_backpressure(options, true);
        options = {};
        options.min_new_tokens = 1;
        check_backpressure(options, true);
        std::cout << "Scalar opt-out and target-only backpressure passed\n";
    }
    if (mode == "all" || mode == "step") {
        check_step_delivery(false, false);
        check_step_delivery(false, true);
        check_step_delivery(true, false);
        std::cout << "Singleton step prefix, text stop, and failure passed\n";
    }
    if (mode == "all" || mode == "abort") {
        check_abort_fault(false);
        check_abort_fault(true);
        check_completion_claim(true, false);
        check_completion_claim(false, false);
        check_completion_claim(false, true);
        std::cout << "Abort ownership across recoverable and permanent failures passed\n";
    }
    if (mode == "all" || mode == "trace") check_prefill_trace();
    if (mode == "all" || mode == "prefill") {
        for (auto arrival : {Arrival::none, Arrival::during_admit, Arrival::during_advance,
                             Arrival::aborted, Arrival::disconnected, Arrival::blocked_writer}) {
            check_prefill_roster(arrival);
        }
        check_prefill_publication_order();
        check_prefill_replacement_order(false);
        check_prefill_replacement_order(true);
        std::cout << "Cold prefill roster, late arrival, and output ownership passed\n";
    }
}
