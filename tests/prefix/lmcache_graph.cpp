#include "sglang/testing/prefix_fixture.h"
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
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
        changed_.wait_for(lock, 20s, [this] { return !armed_; });
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
    prefix_fixture::attach(options, cache, prompt, salt, [] { gate.hold(); });
    require(batch.admit(0, prompt, options) == 0, "Unexpected cold checkpoint admission");
    const auto initial = step(batch, 0, depth, false, false);
    require(!initial.prefix_pending, "Premature checkpoint export");
    if (held) gate.arm();
    try {
        const auto last = step(batch, 0, depth, true, true);
        require(last.prefix_pending && last.finish_reason && !last.extra_token_count,
                "Missing terminal single-input graph checkpoint export");
        if (held) {
            require(gate.await(), "Daemon checkpoint store did not reach the hold");
            require(batch.poll_prefix(0).pending, "Export did not remain pending at the hold");
            const auto unrelated = step(batch, 1, depth, true, true, true);
            require(*unrelated.token != *last.token,
                    "Peer fixture did not produce a distinct token");
            require(batch.poll_prefix(0).pending, "Export finished while peer replay accessed it");
            gate.release();
            std::cout << "K" << depth << " peer graph replay token " << *unrelated.token << '\n';
        }
        prefix_fixture::wait(batch, 0);
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
    if (!salt.empty()) {
        options.prefix_cache_read = prefix_fixture::read_checkpoint(cache, prompt, salt, endpoint);
        prefix_fixture::attach(options, cache, prompt, salt);
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
    if (!salt.empty()) prefix_fixture::wait(batch, 0);
    return output;
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
    for (const int depth : {0, 1}) {
        DecoderBatch batch(model, {.cuda_graphs = true, .max_lanes = 2});
        DecoderOptions peer_options;
        peer_options.max_new_tokens = 16;
        peer_options.ignore_eos = true;
        peer_options.mtp_steps = depth;
        require(batch.admit(1, peer, peer_options) == 0, "Unexpected peer admission");
        step(batch, 1, depth, false, false);
        step(batch, 1, depth, false, true);

        const auto unheld_salt = "graph-unheld-K" + std::to_string(depth);
        const auto held_salt = "graph-held-K" + std::to_string(depth);
        const auto baseline = capture(batch, cache, prompt, unheld_salt, depth, false);
        const auto overlap = capture(batch, cache, prompt, held_salt, depth, true);
        batch.cancel(1);
        require(baseline.history == overlap.history && baseline.output == overlap.output,
                "Graph checkpoint token stream changed");
        const auto reference = persisted_state(cache, baseline.history, unheld_salt,
                                               memory.stream());
        const auto actual = persisted_state(cache, overlap.history, held_salt, memory.stream());
        require(reference == actual, "Held graph export corrupted GQA/GDN or hidden/residual");

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
