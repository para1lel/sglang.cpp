#include "sglang/testing/prefix_fixture.h"

#include <cuda_runtime.h>

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <vector>

extern "C" cudaError_t __real_cudaLaunchKernel(const void*, dim3, dim3, void**, std::size_t,
                                              cudaStream_t);

namespace {

using namespace sglang;
using Mask = std::array<bool, decoder_batch_max_lanes>;
constexpr int page = 128;
constexpr int banks = 48;

enum class Stage { ordered, prepare, chunked, exact };

std::vector<Stage> launches;
bool record = false;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template<typename Operation>
auto trace(Operation operation) {
    launches.clear();
    record = true;
    auto result = operation();
    record = false;
    return result;
}

void expect_route(std::vector<Stage> expected, const char* label) {
    std::array<int, 4> counts{};
    for (const auto stage : launches) {
        ++counts[static_cast<int>(stage)];
    }
    std::cout << label << " ordered=" << counts[0] << " v9_prepare=" << counts[1]
              << " v9_recur=" << counts[2] << " exact=" << counts[3] << std::endl;
    require(launches == expected, "Unexpected GDN prefill launch order");
}

void expect_tile(bool first, const char* label) {
    std::vector<Stage> expected;
    if (first) {
        expected.push_back(Stage::ordered);
    }
    for (int bank = first ? 1 : 0; bank < banks; ++bank) {
        expected.push_back(Stage::prepare);
        expected.push_back(Stage::chunked);
    }
    expect_route(std::move(expected), label);
}

void check_default(const Decoder& decoder, const std::vector<int>& prompt) {
    const auto result = trace([&] {
        return decoder.generate(prompt, {.max_new_tokens = 4, .top_logits = 16});
    });
    expect_tile(true, "default serial prefill");
    require(result.output_token_ids.size() == 4 && result.step_top_logits.size() == 4 &&
            result.finish_reason == DecoderFinishReason::length, "Default generation incomplete");
    for (std::size_t step = 0; step < result.output_token_ids.size(); ++step) {
        const auto& top = result.step_top_logits[step];
        require(top.size() == 16, "Default generation omitted logits");
        std::cout << "token=" << result.output_token_ids[step];
        for (const auto& entry : top) {
            std::cout << ' ' << entry.token_id << '=' << std::hexfloat << entry.logit;
        }
        std::cout << '\n';
    }
}

struct Step {
    std::array<DecoderBatchStep, decoder_batch_max_lanes> lanes;
    DecoderBatchStepStats stats;
};

Step advance(DecoderBatch& batch, const Mask& selected) {
    Step step;
    step.lanes = trace([&] { return batch.advance_chunked(selected, &step.stats); });
    return step;
}

int token(const Step& step, int slot, bool decode, bool finished) {
    const auto& lane = step.lanes[slot];
    require(lane.token && lane.decode == decode && !lane.sampling_failed &&
            !lane.constraint_failed, "Unexpected generation step");
    require(finished ? lane.finish_reason == DecoderFinishReason::length : !lane.finish_reason,
            "Unexpected lane retirement");
    return *lane.token;
}

void check_prefix_fallback(const Decoder& decoder, const std::vector<int>& prompt) {
    DecoderBatch batch(decoder, {.cuda_graphs = true, .max_lanes = 2});
    const auto cache = prefix_fixture::client();
    constexpr auto salt = "gdn-routes";
    const std::vector<int> prefix(prompt.begin(), prompt.begin() + page);
    DecoderOptions export_options{.max_new_tokens = 1, .ignore_eos = true};
    const auto stores = prefix_fixture::attach(export_options, cache, prefix, salt);
    require(batch.admit(0, prefix, export_options) == 0, "Unexpected prefix on export");
    const auto stored = advance(batch, Mask{true});
    require(stored.lanes[0].finish_reason && !stored.lanes[0].prefix_pending,
            "Terminal prefix export blocked completion");
    stores->wait();
    batch.release_retired_allocations();
    DecoderOptions options{.max_new_tokens = 4, .ignore_eos = true};
    require(batch.admit(0, prefix, options) == 0, "First peer unexpectedly restored");
    const auto prime = advance(batch, Mask{true});
    expect_tile(true, "first absolute tile");
    require(prime.stats.prefill_rows[0] == page && prime.stats.graph_slots == 0 &&
            prime.stats.prefill_workspace_bytes > 0, "Wrong prefix prime scheduler step");
    token(prime, 0, false, false);

    options.max_new_tokens = 5;
    options.prefix_cache_read = prefix_fixture::read_checkpoint(cache, prompt, salt, page);
    const auto restored_stores = prefix_fixture::attach(options, cache, prompt, salt);
    require(batch.admit(1, prompt, options) == 0, "Restore should complete asynchronously");
    require(prefix_fixture::wait(batch, 1).restored == page, "Incorrect restored prefix position");
    const auto live = batch.request_device_allocation_bytes();
    const auto mixed = advance(batch, Mask{true, true});
    expect_tile(false, "restored tile with decode peer");
    require(mixed.stats.prefill_rows[0] == 0 && mixed.stats.prefill_rows[1] == page &&
            mixed.stats.graph_slots == 0 &&
            mixed.stats.prefill_workspace_bytes == prime.stats.prefill_workspace_bytes &&
            mixed.stats.peak_request_bytes == live + mixed.stats.prefill_workspace_bytes &&
            batch.request_device_allocation_bytes() == live, "Wrong mixed scheduler workspace");
    token(mixed, 0, true, false);
    std::array restored{token(mixed, 1, false, false), -1, -1, -1, -1};
    const auto warm = advance(batch, Mask{true, true});
    expect_route({}, "decode-only graph warmup");
    require(warm.stats.graph_slots == 0 && warm.stats.prefill_workspace_bytes == 0,
            "Wrong two-lane graph warmup");
    token(warm, 0, true, false);
    restored[1] = token(warm, 1, true, false);
    const auto decode = advance(batch, Mask{true, true});
    expect_route({}, "decode-only with prefix cache");
    require(decode.stats.graph_slots == 0 && decode.stats.prefill_workspace_bytes == 0,
            "Prefix decode used a graph or prefill scratch");
    token(decode, 0, true, true);
    restored[2] = token(decode, 1, true, false);
    const auto single_warm = advance(batch, Mask{false, true});
    expect_route({}, "single-lane graph replay");
    require(single_warm.stats.graph_slots == (1u << 1) &&
            single_warm.stats.prefill_workspace_bytes == 0, "Wrong singleton graph replay");
    restored[3] = token(single_warm, 1, true, false);
    const auto retire = advance(batch, Mask{false, true});
    expect_route({}, "retiring decode");
    require(retire.stats.graph_slots == (1u << 1) && retire.stats.prefill_workspace_bytes == 0,
            "Retiring prefix lane missed singleton replay or used prefill scratch");
    restored[4] = token(retire, 1, true, true);
    restored_stores->wait();
    require(batch.request_device_allocation_bytes() == 0, "Retired lanes kept request storage");
    options.prefix_cache_read.reset();
    options.restore_prefix = nullptr;
    options.store_prefix = nullptr;
    require(batch.admit(1, prompt, options) == 0, "Fresh lane unexpectedly restored");
    const auto fresh = advance(batch, Mask{false, true});
    expect_tile(true, "reused lane cold prefill");
    require(fresh.stats.prefill_rows[1] == 2 * page, "Wrong cold prefill grant");
    std::array cold{token(fresh, 1, false, false), -1, -1, -1, -1};
    for (std::size_t index = 1; index < cold.size(); ++index) {
        cold[index] = token(advance(batch, Mask{false, true}), 1, true, index + 1 == cold.size());
    }
    require(batch.request_device_allocation_bytes() == 0, "Reused lane retained request storage");
    require(restored == cold, "Prefix restore or mixed step changed greedy output");
    std::cout << "restored=" << page << " mixed_workspace=" << mixed.stats.prefill_workspace_bytes
              << " peak=" << mixed.stats.peak_request_bytes << " tokens";
    for (const auto id : cold) {
        std::cout << ' ' << id;
    }
    std::cout << '\n';
}

}

extern "C" cudaError_t __wrap_cudaLaunchKernel(const void* function, dim3 grid, dim3 block,
                                               void** args, std::size_t shared,
                                               cudaStream_t stream) {
    if (record) {
        const char* name = nullptr;
        const auto status = cudaFuncGetName(&name, function);
        if (status != cudaSuccess) {
            return status;
        }
        const std::string_view kernel(name);
        if (kernel.find("gdn_prefill_recur") != kernel.npos) {
            launches.push_back(Stage::ordered);
        } else if (kernel.find("gdn_chunk") != kernel.npos) {
            if (kernel.find("prepare") != kernel.npos) {
                launches.push_back(Stage::prepare);
            } else if (kernel.find("recur_warp") != kernel.npos) {
                launches.push_back(Stage::chunked);
            }
        } else if (kernel.find("gdn_recurILb1E") != kernel.npos) {
            launches.push_back(Stage::exact);
        }
    }
    return __real_cudaLaunchKernel(function, grid, block, args, shared, stream);
}

int main(int argc, char** argv) try {
    require(argc == 3, "Usage: gdn-routes CHECKPOINT PROMPT_IDS");
    std::ifstream file(argv[2]);
    std::vector<int> prompt{std::istream_iterator<int>(file), std::istream_iterator<int>()};
    require(prompt.size() >= 2 * page, "Expected at least 256 prompt token IDs");
    prompt.resize(2 * page);
    const Decoder decoder(argv[1], 0, false);
    require(decoder.uses_b200_operators(), "GDN route check requires a B200");
    check_default(decoder, prompt);
    check_prefix_fallback(decoder, prompt);
    return 0;
} catch (const std::exception& error) {
    std::cerr << "GDN route check failed: " << error.what() << '\n';
    return 1;
}
