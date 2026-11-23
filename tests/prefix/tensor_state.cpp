#include "decoder_batch.cpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>

namespace {

using namespace sglang;
using namespace sglang::decoder;

constexpr int page = mxfp8_page_size;

void require(bool condition, std::string_view what) {
    if (!condition) {
        throw std::runtime_error(std::string(what));
    }
}

std::vector<int> ids(const char* path, std::size_t expected) {
    std::ifstream input(path);
    std::vector<int> result;
    int token;
    while (input >> token) {
        result.push_back(token);
    }
    require(input.eof() && result.size() == expected, "Unexpected fixture size");
    return result;
}

template<class Element>
std::vector<Element> read_device(const Element* from, std::size_t count) {
    std::vector<Element> data(count);
    check(cudaMemcpy(data.data(), from, count * sizeof(Element), cudaMemcpyDeviceToHost),
          "Read device state");
    return data;
}

template<class Element>
void same(const Element* left, const Element* right, std::size_t count, std::string_view field) {
    const auto left_host = read_device(left, count);
    const auto right_host = read_device(right, count);
    require(std::memcmp(left_host.data(), right_host.data(), count * sizeof(Element)) == 0, field);
}

std::vector<std::int32_t> mapping(const Request& request) {
    return read_device(request.pages.device_pages, request.pages.page_count);
}

void remap(Request& request, bool reverse) {
    auto pages = mapping(request);
    if (reverse) {
        std::reverse(pages.begin(), pages.end());
    } else {
        std::rotate(pages.begin(), pages.begin() + 1, pages.end());
    }
    check(cudaMemcpy(const_cast<std::int32_t*>(request.pages.device_pages), pages.data(),
                     pages.size() * sizeof(pages.front()), cudaMemcpyHostToDevice),
          "Replace test request page map");
}

void compare_scales(const std::uint8_t* expected, const std::uint8_t* hit, int rows) {
    if (rows == page) {
        same(expected, hit, mxfp8_scale_bytes_per_page, "Full scale page");
        return;
    }
    const auto left = read_device(expected, mxfp8_scale_bytes_per_page);
    const auto right = read_device(hit, mxfp8_scale_bytes_per_page);
    for (int row = 0; row < rows; ++row) {
        for (int head = 0; head < mxfp8_kv_heads; ++head) {
            const int lane = head * mxfp8_group_size + row % mxfp8_group_size;
            const int group = lane * (page / mxfp8_group_size) + row / mxfp8_group_size;
            const int offset = group * mxfp8_groups;
            require(std::memcmp(left.data() + offset, right.data() + offset, mxfp8_groups) == 0,
                    "Active tail scale values");
        }
    }
}

void compare_kv(const Request& expected, const Request& hit, int length) {
    const auto left_map = mapping(expected);
    const auto right_map = mapping(hit);
    for (int layer = 0; layer < full_attention_layers; ++layer) {
        const auto& left = expected.caches[layer];
        const auto& right = hit.caches[layer];
        for (int logical = 0; logical * page < length; ++logical) {
            const int rows = std::min(page, length - logical * page);
            const auto left_payload = left_map[logical] * mxfp8_payload_bytes_per_page;
            const auto right_payload = right_map[logical] * mxfp8_payload_bytes_per_page;
            const auto left_scale = left_map[logical] * mxfp8_scale_bytes_per_page;
            const auto right_scale = right_map[logical] * mxfp8_scale_bytes_per_page;
            const auto count = rows * (mxfp8_payload_bytes_per_page / page);
            same(left.key_payload + left_payload, right.key_payload + right_payload, count,
                 "Key payload");
            same(left.value_payload + left_payload, right.value_payload + right_payload, count,
                 "Value payload");
            compare_scales(left.key_scales + left_scale, right.key_scales + right_scale, rows);
            compare_scales(left.value_scales + left_scale, right.value_scales + right_scale, rows);
        }
    }
}

void compare_state(const Request& expected, const Request& hit, int length, bool scored) {
    compare_kv(expected, hit, length);
    same(expected.states.front().history, hit.states.front().history,
         gdn_layers * gdn_history_elements, "GDN convolution history");
    same(expected.states.front().recurrent, hit.states.front().recurrent,
         gdn_layers * gdn_state_elements, "GDN recurrent state");
    if (scored) {
        same(expected.work.hidden, hit.work.hidden, hidden_size, "Last hidden");
        same(expected.work.residual, hit.work.residual, hidden_size, "Last residual");
        same(expected.work.logits, hit.work.logits, vocab_size, "All vocabulary BF16 logits");
    }
}

Selection slots(std::initializer_list<int> ids) {
    Selection result{};
    for (int slot : ids) {
        result.at(slot) = true;
    }
    return result;
}

Request& request(DecoderBatch& batch, int slot) {
    return *batch.state_->storage->slots.at(slot)->request;
}

std::vector<int> head(const std::vector<int>& tokens, int count) {
    return {tokens.begin(), tokens.begin() + count};
}

DecoderOptions options(std::string salt) {
    DecoderOptions result;
    result.prefix_cache_salt = std::move(salt);
    result.max_new_tokens = 16;
    result.temperature = 0.8;
    result.top_k = 20;
    result.top_p = 0.9;
    result.seed = -509;
    return result;
}

DecoderBatchStep first_result(DecoderBatch& batch, int lane, int peer = -1) {
    const auto selected = peer < 0 ? slots({lane}) : slots({lane, peer});
    for (int step = 0; step < 128; ++step) {
        DecoderBatchStepStats stats;
        const auto results = batch.advance_chunked(selected, &stats);
        const auto& result = results[lane];
        require(!result.sampling_failed && !result.constraint_failed, "First token failed");
        if (result.token) return result;
        require(!result.finish_reason && stats.prefill_rows[lane] > 0, "Prefill did not advance");
    }
    throw std::runtime_error("First token deadline");
}

int first_token(DecoderBatch& batch, int lane, int peer = -1) {
    const auto step = first_result(batch, lane, peer);
    require(!step.finish_reason && !step.decode && !step.mtp_drafted_tokens,
            "First token did not come from the prompt score");
    return *step.token;
}

void finish_prompt(DecoderBatch& batch, int lane) {
    const auto step = first_result(batch, lane);
    require(step.finish_reason == DecoderFinishReason::length && !step.decode &&
            !step.mtp_drafted_tokens, "One-token prime did not complete at the prompt boundary");
}

void retry_discarded(DecoderBatch& batch, const std::vector<int>& prompt, int cached) {
    const auto state = options("abandoned");
    require(batch.admit(2, prompt, state) == 0, "Abandoned request started warm");
    if (cached > page) {
        auto first = batch.advance(slots({2}));
        require(!first[2].token, "Abandoned request completed prefill");
    }
    batch.cancel(2);
    require(batch.admit(2, prompt, state) == 0, "Abandoned request published a checkpoint");
    batch.cancel(2);
}

void inspect(DecoderBatch& oracle, DecoderBatch& candidate, const std::vector<int>& prompt,
             int cached, std::ofstream& logits) {
    oracle.reset_prefix_cache();
    candidate.reset_prefix_cache();
    const int length = prompt.size();
    const auto parameters = options("case");
    retry_discarded(candidate, prompt, cached);
    require(oracle.admit(0, prompt, parameters) == 0, "Cache-disabled oracle was warm");
    if (cached) {
        auto prime = parameters;
        prime.max_new_tokens = 1;
        require(candidate.admit(0, head(prompt, cached), prime) == 0, "Unexpected warm prime");
        remap(request(candidate, 0), true);
        finish_prompt(candidate, 0);
        require(candidate.prefix_cache_stats().pages == (cached + page - 1) / page,
                "Wrong checkpoint pages");
    }
    int expected = -1;
    if (length == cached) {
        expected = first_token(oracle, 0);
    } else {
        int processed = 0;
        while (processed < cached) {
            DecoderBatchStepStats stats;
            const auto cold = oracle.advance_chunked(slots({0}), &stats)[0];
            require(!cold.token && !cold.finish_reason && stats.prefill_rows[0] == page,
                    "Cold prefix scoring started before its comparison boundary");
            processed += stats.prefill_rows[0];
        }
        require(processed == cached, "Cold prefill missed the comparison boundary");
    }
    require(candidate.admit(1, prompt, parameters) == cached, "Unexpected restore depth");
    if (cached) {
        auto& warm = request(candidate, 1);
        remap(warm, false);
        auto& cache = *candidate.state_->storage->prefix_cache;
        const auto stream = candidate.state_->execution.stream();
        require(cache.restore(warm, prompt, parameters.prefix_cache_salt, stream) == cached,
                "Restore into permuted destination");
        check(cudaStreamSynchronize(stream), "Complete permuted restore");
        compare_state(request(oracle, 0), warm, cached, false);
    }
    const auto peer = options("peer");
    if (length != cached) {
        require(oracle.admit(3, {25}, peer) == 0, "Unexpected cold peer restore");
        first_token(oracle, 3);
    }
    require(candidate.admit(3, {25}, peer) == 0, "Unexpected warm peer");
    first_token(candidate, 3);
    if (length != cached) {
        for (int processed = cached; processed < length;) {
            DecoderBatchStepStats cold_stats, warm_stats;
            const auto cold = oracle.advance_chunked(slots({0, 3}), &cold_stats)[0];
            const auto warm = candidate.advance_chunked(slots({1, 3}), &warm_stats)[1];
            const int rows = cold_stats.prefill_rows[0];
            require(rows == std::min(page - 1, length - processed) &&
                    rows == warm_stats.prefill_rows[1] &&
                    cold_stats.target_rows == rows + 1 && warm_stats.target_rows == rows + 1 &&
                    cold.token == warm.token && !cold.finish_reason && !warm.finish_reason,
                    "Matched suffix grants or seeded tokens differ");
            std::cout << "Suffix at " << processed << " grants cold " << rows << " warm "
                      << warm_stats.prefill_rows[1] << " plus one decode row each\n";
            processed += rows;
            if (processed == length) {
                require(cold.token.has_value(), "Final suffix grant has no score");
                expected = *cold.token;
            }
        }
        oracle.cancel(3);
    } else {
        require(first_token(candidate, 1, 3) == expected, "Seeded first token differs");
    }
    compare_state(request(oracle, 0), request(candidate, 1), length, true);
    const auto row = read_device(request(candidate, 1).work.logits, vocab_size);
    logits.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(Bf16));
    require(logits.good(), "Write BF16 evidence");
    candidate.cancel(3);
    candidate.cancel(1);
    oracle.cancel(0);
    require(candidate.request_device_allocation_bytes() == 0, "Request storage leak");
    std::cout << "length " << length << " cached " << cached << " token " << expected
              << " KV, GDN, hidden, residual and full BF16 logits match\n";
}

void terminal_history(DecoderBatch& oracle, DecoderBatch& candidate,
                      const std::vector<int>& prompt, bool graph) {
    oracle.reset_prefix_cache();
    candidate.reset_prefix_cache();
    DecoderOptions producer;
    producer.prefix_cache_salt = "w1";
    producer.max_new_tokens = 128;
    producer.ignore_eos = true;
    auto reference = producer;
    ++reference.max_new_tokens;
    const auto before = candidate.prefix_cache_stats();
    require(oracle.admit(0, prompt, reference) == 0 && candidate.admit(0, prompt, producer) == 0,
            "Terminal history fixture started warm");

    auto history = prompt;
    history.push_back(first_token(candidate, 0));
    require(first_token(oracle, 0) == history.back(), "Cold control first token differs");
    DecoderBatchStepStats cold_stats, warm_stats;
    for (int row = 1; row < producer.max_new_tokens; ++row) {
        require(candidate.prefix_cache_stats().stores == before.stores,
                "Published an off-cadence checkpoint before the turn ended");
        const auto expected = oracle.advance_chunked(slots({0}), &cold_stats)[0];
        const auto actual = candidate.advance_chunked(slots({0}), &warm_stats)[0];
        const auto finish = row + 1 == producer.max_new_tokens
            ? std::optional{DecoderFinishReason::length} : std::nullopt;
        require(actual.token && actual.token == expected.token && actual.decode &&
                !expected.finish_reason && actual.finish_reason == finish,
                "Terminal generation disagrees with the cold reference");
        history.push_back(*actual.token);
    }
    const int processed = history.size() - 1;
    const auto saved = candidate.prefix_cache_stats();
    require(processed == 255 && saved.entries == 1 && saved.pages == 2 &&
            saved.stores == before.stores + 1 && !candidate.request_device_allocation_bytes(),
            "Successful W1 turn did not save exactly the 255 processed rows");
    const std::uint32_t graph_slots = graph ? 1 : 0;
    require(cold_stats.graph_slots == graph_slots && warm_stats.graph_slots == graph_slots,
            "The terminal W1 rows did not use the selected execution path");
    auto& expected_state = request(oracle, 0);
    auto& expected_row = graph ? *oracle.state_->storage->decode_graphs.narrow->work[0]
                               : expected_state.work;
    if (graph) {
        const auto& produced = *candidate.state_->storage->decode_graphs.narrow->work[0];
        same(expected_row.hidden, produced.hidden, hidden_size, "Produced graph hidden");
        same(expected_row.residual, produced.residual, hidden_size, "Produced graph residual");
        same(expected_row.logits, produced.logits, vocab_size, "Produced graph BF16 logits");
    }

    const auto key = head(history, processed);
    require(candidate.admit(1, key, producer) == processed, "Terminal checkpoint key differs");
    require(first_token(candidate, 1) == history.back(), "Terminal score differs");
    const auto& restored = request(candidate, 1);
    compare_state(expected_state, restored, processed, false);
    same(expected_row.hidden, restored.work.hidden, hidden_size, "Terminal hidden");
    same(expected_row.residual, restored.work.residual, hidden_size, "Terminal residual");
    const auto stream = oracle.state_->execution.stream();
    oracle.state_->model.compute_prefill_logits(expected_row, stream);
    check(cudaStreamSynchronize(stream), "Score the cold terminal row with the restored head");
    same(expected_row.logits, restored.work.logits, vocab_size, "Terminal BF16 logits");
    candidate.cancel(1);
    auto changed = history;
    changed[processed - 1] = (changed[processed - 1] + 1) % vocab_size;
    require(candidate.admit(1, changed, producer) == 0, "Matched a changed consumed token");
    candidate.cancel(1);
    require(candidate.admit(1, history, producer) == processed, "Last emitted ID was cached");
    DecoderBatchStepStats stats;
    const auto continuation = candidate.advance_chunked(slots({1}), &stats)[1];
    const auto expected = oracle.advance_chunked(slots({0}))[0];
    require(continuation.token && continuation.token == expected.token &&
            stats.prefill_rows[1] == 1, "Last emitted ID did not run as the continuation suffix");
    candidate.cancel(1);
    std::cout << "W1 " << (graph ? "graph" : "eager") << " terminal: one 255-row snapshot, "
              << "exact state/logits, last ID replayed\n";
}

void denied_completion(DecoderBatch& batch, const std::vector<int>& tokens) {
    constexpr int interval = decoder_batch_prefix_checkpoint_interval;
    for (const int length : {interval - 1, interval}) {
        batch.reset_prefix_cache();
        const auto before = batch.prefix_cache_stats();
        const auto prompt = head(tokens, length);
        DecoderOptions denied;
        int claims = 0;
        denied.claim_completion = [&] { ++claims; return false; };
        require(batch.admit(1, prompt, denied) == 0, "Denied claim fixture started warm");
        const auto result = first_result(batch, 1);
        const auto saved = batch.prefix_cache_stats();
        const bool periodic = length == interval;
        const auto finish = periodic ? DecoderFinishReason::length : DecoderFinishReason::cancelled;
        require(result.finish_reason == finish && claims == !periodic &&
                saved.stores == before.stores + periodic && saved.entries == int(periodic),
                "Denied terminal claim did not preserve only the committed cadence snapshot");
        require(batch.admit(1, prompt, denied) == (periodic ? interval : 0),
                "Denied claim yielded the wrong subsequent restore");
        batch.cancel(1);
    }
    std::cout << "Denied completion: no 8191 save, one 8192 cadence save\n";
}

void interrupted_cadence(Decoder& decoder, const std::vector<int>& tokens) {
    constexpr int interval = decoder_batch_prefix_checkpoint_interval;
    auto prompt = tokens;
    prompt.insert(prompt.end(), tokens.begin(), tokens.end());
    prompt.push_back(tokens.front());
    DecoderBatch batch(decoder, {false, 1, 2, 1024});
    DecoderOptions options;
    require(batch.admit(0, prompt, options) == 0, "Cadence fixture started warm");
    int processed = 0;
    while (processed < 2 * interval) {
        DecoderBatchStepStats stats;
        const auto next = batch.advance_chunked(slots({0}), &stats)[0];
        require(stats.prefill_rows[0] > 0 && !next.token && !next.finish_reason,
                "Cadence became a terminal publication");
        processed += stats.prefill_rows[0];
        const auto stored = batch.prefix_cache_stats();
        require(stored.stores == static_cast<unsigned>(processed / interval) &&
                stored.entries == processed / interval, "Unexpected periodic checkpoint count");
    }
    require(processed == 2 * interval, "Prefill crossed the second cadence boundary");
    batch.cancel(0);
    require(batch.admit(0, prompt, options) == 2 * interval,
            "Cancellation lost the cadence checkpoint or cached the unprocessed suffix");
    batch.cancel(0);
    require(!batch.request_device_allocation_bytes(), "Cadence fixture retained request state");
    std::cout << "Nonterminal checkpoints at 8192 and 16384 survived cancellation\n";
}

}

int main(int argc, char** argv) {
    try {
        require(argc == 5 || (argc == 6 && std::string_view(argv[5]) == "--short"),
                "Pass checkpoint, 8192 long IDs, 128 W1 IDs, and BF16 output path");
        const auto lane = ids(argv[2], 8192);
        const auto w1 = ids(argv[3], 128);
        std::ofstream logits(argv[4], std::ios::binary);
        require(logits.good(), "Open BF16 evidence");
        Decoder decoder(argv[1]);
        DecoderBatch oracle(decoder, {true, 4, 0, page});
        DecoderBatch candidate(decoder, {true, 4, 2, page});
        for (const int size : {127, 128, 129, 255, 256, 257, 385}) {
            inspect(oracle, candidate, head(lane, size), size, logits);
        }
        inspect(oracle, candidate, head(lane, 385), 2 * page, logits);
        if (argc == 5) {
            constexpr int interval = decoder_batch_prefix_checkpoint_interval;
            inspect(oracle, candidate, lane, interval, logits);
            auto suffix = lane;
            suffix.push_back(lane.front());
            inspect(oracle, candidate, suffix, interval, logits);
            terminal_history(oracle, candidate, w1, true);
            {
                DecoderBatch cold_eager(decoder, {false, 2, 0, page});
                DecoderBatch warm_eager(decoder, {false, 2, 2, page});
                terminal_history(cold_eager, warm_eager, w1, false);
            }
            denied_completion(candidate, lane);
            interrupted_cadence(decoder, lane);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
