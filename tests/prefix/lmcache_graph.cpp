#include "sglang/decoder_internal.h"
#include "sglang/decoder_prefix_cache.h"
#include "sglang/testing/prefix_fixture.h"
#include "sglang/server/batch_service.h"

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace {
using namespace sglang;
using namespace sglang::decoder;
using namespace std::chrono_literals;
using Selection = std::array<bool, decoder_batch_max_lanes>;

class StoreGate {
public:
    void arm() {
        std::lock_guard lock(mutex_);
        armed_ = true;
        held_ = false;
    }

    void hold() {
        std::unique_lock lock(mutex_);
        if (!armed_) return;
        held_ = true;
        changed_.notify_all();
        if (!changed_.wait_for(lock, 20s, [this] { return !armed_; }))
            throw std::runtime_error("Checkpoint STORE timed out before explicit release");
    }

    bool await() {
        std::unique_lock lock(mutex_);
        return changed_.wait_for(lock, 20s, [this] { return held_; });
    }

    void release() {
        std::lock_guard lock(mutex_);
        armed_ = false;
        changed_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    bool armed_ = false;
    bool held_ = false;
};
StoreGate gate;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

DecoderBatchStep step(DecoderBatch& batch, int slot, int depth, bool graph, bool decode,
                      bool selected_peer = false) {
    Selection selected{};
    selected[slot] = true;
    if (selected_peer) selected[1 - slot] = true;
    DecoderBatchStepStats stats;
    auto results = batch.advance_chunked(selected, &stats);
    const auto& result = results[slot];
    require(result.token.has_value() && result.decode == decode, "Unexpected decode step");
    const auto actual = depth ? stats.verification_graph_slots : stats.graph_slots;
    require(bool(actual & (1u << slot)) == graph, "Unexpected CUDA graph selection");
    if (selected_peer) require(!results[1 - slot].token, "Work ran on a leased checkpoint lane");
    return result;
}

struct Capture {
    std::vector<int> history;
    int output;
};

Capture capture(DecoderBatch& batch, const LmCacheClient& cache, const std::vector<int>& prompt,
                const std::string& salt, int depth, bool held) {
    DecoderOptions options;
    options.max_new_tokens = depth ? 16 : 2;
    options.ignore_eos = depth == 0;
    options.mtp_steps = depth;
    if (depth) options.additional_stop_token_ids.insert(198);
    const auto acknowledged = batch.prefix_cache_stats().stores;
    const auto stores = prefix_fixture::attach(options, cache, prompt, salt, [] { gate.hold(); });
    require(batch.admit(0, prompt, options) == 0, "Unexpected cold checkpoint admission");
    const auto initial = step(batch, 0, depth, false, false);
    require(!initial.prefix_pending, "Premature checkpoint export");
    if (held) gate.arm();
    try {
        const auto last = step(batch, 0, depth, true, true);
        require(!last.prefix_pending && last.finish_reason && !last.extra_token_count,
                "Terminal single-input graph checkpoint blocked completion");
        if (held) {
            require(gate.await(), "Daemon checkpoint store did not reach the hold");
            require(batch.prefix_cache_stats().stores == acknowledged,
                    "Pending transfer was counted as an acknowledged store");
            DecoderOptions successor{.max_new_tokens = 1, .ignore_eos = true};
            require(batch.admit(0, prompt, successor) == 0,
                    "Detached terminal store retained the producer slot");
            const auto reused = step(batch, 0, 0, false, false);
            require(reused.finish_reason.has_value(), "Reused producer did not finish");
            const auto unrelated = step(batch, 1, depth, true, true, true);
            require(*unrelated.token != *last.token,
                    "Peer fixture did not produce a distinct token");
            gate.release();
            std::cout << "K" << depth << " peer graph replay token " << *unrelated.token << '\n';
        }
        stores->wait();
        require(batch.prefix_cache_stats().stores == acknowledged + 1,
                "Direct decoder did not count the acknowledged store exactly once");
        auto history = prompt;
        history.push_back(*initial.token);
        return {std::move(history), *last.token};
    } catch (...) {
        gate.release();
        throw;
    }
}

auto persisted_state(const LmCacheClient& cache, const std::vector<int>& prompt,
                     const std::string& salt, cudaStream_t stream) {
    const int end = static_cast<int>(prompt.size());
    Request request{0};
    request.initialize(end, stream, end);
    request.reset(stream);
    const auto read = prefix_fixture::read_checkpoint(cache, prompt, salt, end);
    const auto view = restore_prefix_view(request, *read, stream);
    require(cache.retrieve(PrefixCacheKeys(prompt, salt), view) == LmCacheStatus::complete,
            "Cannot retrieve stored native graph state");
    return std::make_pair(prefix_fixture::target_digest(request, end),
                          prefix_fixture::recurrent_digest(request, request.work));
}

struct Score {
    int token;
    float selected;
    std::vector<int> alternatives;
    std::vector<float> logprobs;
    bool operator==(const Score& rhs) const {
        return token == rhs.token && selected == rhs.selected && alternatives == rhs.alternatives &&
               logprobs == rhs.logprobs;
    }
};

std::vector<Score> generate(DecoderBatch& batch, const LmCacheClient& cache,
                            const std::vector<int>& prompt, int count, const std::string& salt,
                            bool warm_graph = true) {
    DecoderOptions options;
    options.max_new_tokens = count;
    options.ignore_eos = true;
    options.generated_top_logprobs = 5;
    const int endpoint = static_cast<int>(prompt.size());
    std::shared_ptr<prefix_fixture::Stores> stores;
    if (!salt.empty()) {
        options.prefix_cache_read = prefix_fixture::read_checkpoint(cache, prompt, salt, endpoint);
        stores = prefix_fixture::attach(options, cache, prompt, salt);
    }
    require(batch.admit(0, prompt, options) == 0, "Unexpected synchronous admission");
    if (!salt.empty()) {
        require(prefix_fixture::wait(batch, 0).restored == endpoint,
                "Warm endpoint did not restore");
    }
    std::vector<Score> output;
    for (int row = 0; row != count; ++row) {
        const auto result = step(batch, 0, 0, row > int(!warm_graph), row != 0);
        require(result.generated_logprobs && result.generated_logprobs->alternatives.size() == 5,
                "Missing continuation score record");
        require(bool(result.finish_reason) == (row + 1 == count), "Unexpected continuation exit");
        const auto& scores = *result.generated_logprobs;
        Score score{*result.token, scores.selected.value, {}, {}};
        require(scores.selected.token_id == score.token, "Score winner differs from emitted ID");
        for (const auto& alternative : scores.alternatives) {
            score.alternatives.push_back(alternative.token_id);
            score.logprobs.push_back(alternative.value);
        }
        output.push_back(std::move(score));
    }
    if (stores) stores->wait();
    return output;
}

void cancelled_boundary(Decoder& model, const LmCacheClient& cache) {
    DecoderBatch batch(model, {.max_lanes = 1});
    const std::vector<int> prompt(decoder_batch_prefix_checkpoint_interval, 198);
    Selection selected{};
    selected[0] = true;
    for (const bool text_stop : {false, true}) {
        const auto salt = prefix_fixture::run_salt(text_stop ? "boundary-stop-abort" :
                                                               "boundary-length-abort");
        server::BatchRequestId id(salt);
        require(id.dispatch(), "Boundary request was not dispatched");
        DecoderOptions options;
        options.max_new_tokens = text_stop ? 4 : 1;
        options.ignore_eos = true;
        int claims = 0;
        options.claim_completion = [&] {
            ++claims;
            if (!text_stop) require(id.abort(), "Length completion already claimed");
            return id.claim_completion();
        };
        if (text_stop) options.text_stop = [&](const int*, int count, bool terminal) {
            require(count == 1 && !terminal && id.abort(), "Text stop already claimed");
            return count;
        };
        const auto stores = prefix_fixture::attach(options, cache, prompt, salt);
        require(batch.admit(0, prompt, options) == 0, "Unexpected boundary admission");
        int rows = 0;
        while (rows < int(prompt.size())) {
            DecoderBatchStepStats stats;
            const auto last = batch.advance_chunked(selected, &stats)[0];
            require(stats.prefill_rows[0] > 0, "Boundary prefill did not advance");
            rows += stats.prefill_rows[0];
            if (rows == int(prompt.size())) {
                require(last.finish_reason == DecoderFinishReason::cancelled &&
                        !last.prefix_pending,
                        "Lost terminal claim published a periodic or deferred checkpoint");
            }
        }
        require(claims == 1 && stores->receipts.empty() && !batch.poll_prefix(0).pending,
                "Cancelled boundary retained a store or claimed completion twice");
        const auto match = cache.lookup_prefix(PrefixCacheKeys(prompt, salt), rows, false);
        require(match.gdn == 0 && match.gqa == 0, "Cancelled endpoint reached the daemon");
        std::cout << salt << " rows " << rows << " claims " << claims << " stores 0\n";
    }
}

void synchronous_mtp(const Decoder& model, const LmCacheClient& cache,
                     const std::vector<int>& prompt, const std::string& salt) {
    DecoderOptions options;
    options.max_new_tokens = 4;
    options.ignore_eos = true;
    options.mtp_steps = 1;
    const auto cold = model.generate(prompt, options);
    const int end = static_cast<int>(prompt.size());
    options.prefix_cache_read = prefix_fixture::read_checkpoint(cache, prompt, salt, end, true);
    const PrefixCacheKeys keys(prompt, salt);
    for (const bool fail_restore : {false, true}) {
        const auto output_salt = prefix_fixture::run_salt(
            fail_restore ? "sync-restart" : "sync-hit");
        const auto stores = prefix_fixture::attach(options, cache, prompt, output_salt);
        int restores = 0, reservations = 0;
        options.restore_prefix = [&](const PrefixDeviceTransfer& view) {
            ++restores;
            require(cache.retrieve(keys, view) == LmCacheStatus::complete,
                    "Could not populate the synchronous MTP restore destination");
            std::promise<bool> completion;
            if (fail_restore) {
                completion.set_exception(std::make_exception_ptr(std::runtime_error("GET failed")));
            } else {
                completion.set_value(true);
            }
            return completion.get_future().share();
        };
        options.reserve_prefix_store = [&](std::shared_ptr<void>&) { return ++reservations >= 3; };
        bool claimed = false;
        options.claim_completion = [&] {
            claimed = true;
            return true;
        };
        DecoderCallbacks callbacks;
        const auto deadline = std::chrono::steady_clock::now() + 20s;
        callbacks.is_cancelled = [&] {
            return claimed || std::chrono::steady_clock::now() >= deadline;
        };
        const auto result = model.generate(prompt, options, callbacks);
        require(claimed && restores == 1 &&
                result.cached_prompt_tokens == (fail_restore ? 0 : end) &&
                result.finish_reason == DecoderFinishReason::length && result.mtp_rounds > 0 &&
                result.output_token_ids == cold.output_token_ids,
                "Synchronous MTP restore or cold restart changed generation");
        require(reservations == 3 && stores->receipts.size() == 1,
                "Synchronous MTP lost the deferred terminal STORE");
        stores->wait();
        auto history = prompt;
        history.insert(history.end(), result.output_token_ids.begin(),
                       result.output_token_ids.end() - 1);
        prefix_fixture::read_checkpoint(cache, history, output_salt, int(history.size()), true);
    }
    DecoderBatch batch(model, {.max_lanes = 1});
    DecoderOptions failed;
    failed.mtp_steps = 1;
    failed.prefix_cache_read = options.prefix_cache_read;
    failed.restore_prefix = [](const PrefixDeviceTransfer&) {
        std::promise<bool> completion;
        completion.set_exception(std::make_exception_ptr(
            CudaError(cudaErrorInvalidValue, "Injected CUDA restore failure")));
        return completion.get_future().share();
    };
    batch.admit(0, prompt, failed);
    bool propagated = false;
    try {
        batch.poll_prefix(0);
    } catch (const CudaError& error) {
        propagated = error.status == cudaErrorInvalidValue;
    }
    require(propagated, "CUDA restore failure did not propagate");
    std::cout << "synchronous K1 hit and exceptional GET restart, exact cold IDs, "
                 "deferred STORE acknowledged\n";
}

void failed_store_receipt(Decoder& model, const std::vector<int>& prompt) {
    DecoderBatch batch(model, {.max_lanes = 1});
    DecoderOptions options{.max_new_tokens = 1, .ignore_eos = true};
    const auto path = std::string(std::getenv("LMCACHE_SOCKET")) + "/unreachable";
    const auto stores = prefix_fixture::attach(options, LmCacheClient({path, "sglang", 1000}),
                                               prompt, "failed-receipt");
    require(batch.admit(0, prompt, options) == 0, "Unexpected failed-store admission");
    const auto result = step(batch, 0, 0, false, false);
    require(result.finish_reason && !result.prefix_pending && stores->receipts.size() == 1,
            "A failed store did not submit and retire its source lane");
    bool failed = false;
    try {
        stores->wait();
    } catch (const std::runtime_error&) {
        failed = true;
    }
    require(failed && batch.prefix_cache_stats().stores == 0,
            "Failed transfer counted as a successful store");
}

}

int main(int argc, char** argv) try {
    require(argc == 2, "Pass the model checkpoint path");
    const std::vector<int> prompt{151644, 872, 198, 1988, 1475, 311, 420, 2317, 25, 220, 16, 488,
                                  16, 284, 30, 151645, 198, 151644, 77091, 198};
    const std::vector<int> peer{151644, 872, 198, 3119, 220, 20, 3805, 389, 279, 3675, 13, 151645,
                                 198, 151644, 77091, 198};
    const auto cache = prefix_fixture::client();
    Decoder model(argv[1], 0, true);
    Allocations memory(0);
    memory.open_stream();
    cancelled_boundary(model, cache);
    failed_store_receipt(model, prompt);
    for (const int depth : {0, 1}) {
        DecoderBatch batch(model, {.cuda_graphs = true, .max_lanes = 2});
        DecoderOptions peer_options;
        peer_options.max_new_tokens = 16;
        peer_options.ignore_eos = true;
        peer_options.mtp_steps = depth;
        require(batch.admit(1, peer, peer_options) == 0, "Unexpected peer admission");
        step(batch, 1, depth, false, false);
        step(batch, 1, depth, false, true);

        const auto unheld_salt = prefix_fixture::run_salt("graph-unheld-K" + std::to_string(depth));
        const auto held_salt = prefix_fixture::run_salt("graph-held-K" + std::to_string(depth));
        const auto baseline = capture(batch, cache, prompt, unheld_salt, depth, false);
        const auto overlap = capture(batch, cache, prompt, held_salt, depth, true);
        batch.cancel(1);
        require(baseline.history == overlap.history && baseline.output == overlap.output,
                "Graph checkpoint token stream changed");
        const auto reference = persisted_state(cache, baseline.history, unheld_salt,
                                               memory.stream());
        const auto actual = persisted_state(cache, overlap.history, held_salt, memory.stream());
        require(reference == actual, "Held graph export corrupted GQA/GDN or hidden/residual");
        if (depth == 1) synchronous_mtp(model, cache, baseline.history, unheld_salt);

        constexpr int count = 5;
        const auto cold = generate(batch, cache, prompt, count + 1, "", depth == 0);
        const auto warm = generate(batch, cache, baseline.history, count, unheld_salt);
        const auto held = generate(batch, cache, overlap.history, count, held_salt);
        require(warm == held, "Held and unheld warm continuations differ");
        std::cout << "K" << depth << " exact native state and continuation";
        for (int index = 0; index != count; ++index) {
            require(cold[index + 1].token == held[index].token,
                    "Cold and warm continuation IDs differ");
            std::cout << ' ' << held[index].token;
        }
        std::cout << " exact warm scores " << count * 6 << '\n';
    }
    std::cout << "PASS\n";
    return 0;
} catch (const std::exception& error) {
    gate.release();
    std::cerr << error.what() << '\n';
    return 1;
}
