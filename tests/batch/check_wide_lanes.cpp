#include "sglang/decoder_batch.h"
#include "sglang/text_ops.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sglang::decoder {
struct ForwardLane;
}

extern "C" void real_forward(const void*, const sglang::decoder::ForwardLane*, int, cudaStream_t)
    asm("__real__ZNK6sglang7Decoder5Model7forwardEPKNS_7decoder11ForwardLaneEiP11CUstream_st");
extern "C" void wrap_forward(const void*, const sglang::decoder::ForwardLane*, int, cudaStream_t)
    asm("__wrap__ZNK6sglang7Decoder5Model7forwardEPKNS_7decoder11ForwardLaneEiP11CUstream_st");

extern "C" cudaError_t real_gemm(const std::uint8_t*, const __nv_bfloat16*, int, int,
                                 const sglang::TextFp8Projection*, int, cudaStream_t)
    asm("__real__ZN6sglang13text_fp8_gemmEPKhPK13__nv_bfloat16ii"
        "PKNS_17TextFp8ProjectionEiP11CUstream_st");
extern "C" cudaError_t wrap_gemm(const std::uint8_t*, const __nv_bfloat16*, int, int,
                                 const sglang::TextFp8Projection*, int, cudaStream_t)
    asm("__wrap__ZN6sglang13text_fp8_gemmEPKhPK13__nv_bfloat16ii"
        "PKNS_17TextFp8ProjectionEiP11CUstream_st");

namespace {

using namespace sglang;
using Selection = std::array<bool, decoder_batch_max_lanes>;

struct Projection {
    const std::uint8_t* weight;
    int columns;
    int count;
    bool aligned;
};

std::vector<int> forward_widths;
std::vector<Projection> projections;

void require(bool condition, const char* failure) {
    if (!condition) {
        throw std::runtime_error(failure);
    }
}

bool aligned(const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0;
}

void check_packed(int width) {
    require(forward_widths == std::vector{width}, "Expected one wide model forward");
    std::unordered_map<const std::uint8_t*, int> lanes_per_weight;
    for (const auto& projection : projections) {
        require(projection.aligned && projection.columns % 512 == 0 &&
                    projection.count >= 2 && projection.count <= text_fp8_max_batch,
                "Projection did not qualify for a packed FP8 kernel");
        lanes_per_weight[projection.weight] += projection.count;
    }
    require(!lanes_per_weight.empty(), "No packed FP8 projections");
    for (const auto& [weight, lanes] : lanes_per_weight) {
        require(lanes == width, "FP8 projection missed or duplicated lanes");
    }
    std::cout << "packed width=" << width << " weights=" << lanes_per_weight.size()
              << " groups=" << projections.size() << '\n';
}

struct Scenario {
    DecoderOptions options;
    std::array<std::vector<int>, decoder_batch_max_lanes> oracles;
    std::array<int, decoder_batch_max_lanes> cursors{};
    Selection live{};
    DecoderBatch batch;

    Scenario(const Decoder& decoder, int width) : batch(decoder, {true, width}) {
        options.max_new_tokens = 6;
        options.eager = true;
    }

    void admit(int slot, int prompt, const DecoderResult& oracle) {
        require(oracle.finish_reason == DecoderFinishReason::length &&
                    oracle.output_token_ids.size() ==
                        static_cast<std::size_t>(options.max_new_tokens),
                "Invalid serial oracle");
        oracles[slot] = oracle.output_token_ids;
        cursors[slot] = 0;
        live[slot] = true;
        auto batch_options = options;
        batch_options.eager = false;
        batch.admit(slot, {prompt}, batch_options);
    }

    std::uint32_t step(Selection selected, int packed_width = 0) {
        forward_widths.clear();
        projections.clear();
        DecoderBatchStepStats stats;
        const auto steps = batch.advance_chunked(selected, &stats);
        std::uint32_t decoded = 0;
        for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
            const auto& result = steps[slot];
            if (!selected[slot] || !live[slot]) {
                require(!result.decode && !result.token && !result.finish_reason,
                        "Inactive slot advanced");
                continue;
            }
            auto& cursor = cursors[slot];
            require(result.decode == (cursor > 0) && stats.prefill_rows[slot] == (cursor == 0),
                    "Unexpected phase for selected slot");
            require(result.token == oracles[slot].at(cursor++), "Slot diverged from serial");
            require(result.extra_token_count == 0, "K0 emitted speculative tokens");
            if (result.decode) {
                decoded |= std::uint32_t{1} << slot;
            }
            const bool done = cursor == options.max_new_tokens;
            require(done ? result.finish_reason == DecoderFinishReason::length
                         : !result.finish_reason, "Unexpected slot finish");
            live[slot] = !done;
        }
        require(stats.graph_slots == 0 || stats.graph_slots == decoded, "Incorrect graph mask");
        if (packed_width) {
            require(stats.graph_slots == 0, "Unsupported width used a batch graph");
            check_packed(packed_width);
        }
        std::cout << "step decode=" << decoded << " graph=" << stats.graph_slots
                  << " bytes=" << batch.request_device_allocation_bytes() << '\n';
        return stats.graph_slots;
    }
};

void run(const std::filesystem::path& checkpoint, int width) {
    const Decoder decoder(checkpoint);
    Scenario test(decoder, width);
    constexpr std::array prompts{13, 198, 31784};
    std::array<DecoderResult, prompts.size()> serial;
    for (std::size_t index = 0; index < prompts.size(); ++index) {
        serial[index] = decoder.generate({prompts[index]}, test.options);
    }
    require(serial[0].output_token_ids != serial[2].output_token_ids &&
                serial[1].output_token_ids != serial[2].output_token_ids,
            "Replacement fixture does not distinguish request state");
    for (int slot = 0; slot < width; ++slot) {
        const auto index = slot % 2;
        test.admit(slot, prompts[index], serial[index]);
    }
    test.step(test.live, width);
    test.step(test.live, width);

    const int last = width - 1;
    const auto before = test.batch.request_device_allocation_bytes();
    test.batch.cancel(last);
    require(test.batch.request_device_allocation_bytes() < before, "Cancel retained slot memory");
    test.admit(last, prompts[2], serial[2]);
    test.step(test.live, width);

    Selection sparse{};
    sparse[0] = sparse[last] = true;
    require(test.step(sparse) == 0 && test.step(sparse) == 0, "Sparse B200 pair used a graph");
    sparse[0] = false;
    require(test.step(sparse) == 0, "Singleton did not warm eagerly");
    require(test.step(sparse) == (std::uint32_t{1} << last),
            "Singleton graph omitted the highest logical slot");
    while (std::any_of(test.live.begin(), test.live.end(), [](bool active) { return active; })) {
        test.step(test.live);
    }
    require(test.batch.request_device_allocation_bytes() == 0, "Retired slots retained memory");
}

}

extern "C" void wrap_forward(const void* model, const sglang::decoder::ForwardLane* lanes,
                             int count, cudaStream_t stream) {
    forward_widths.push_back(count);
    real_forward(model, lanes, count, stream);
}

extern "C" cudaError_t wrap_gemm(const std::uint8_t* weight, const __nv_bfloat16* scales,
                                 int rows, int columns, const TextFp8Projection* inputs,
                                 int count, cudaStream_t stream) {
    const bool inputs_aligned = std::all_of(inputs, inputs + count, [](const auto& input) {
        return aligned(input.input.values);
    });
    projections.push_back({weight, columns, count, aligned(weight) && inputs_aligned});
    return real_gemm(weight, scales, rows, columns, inputs, count, stream);
}

int main(int argc, char** argv) {
    try {
        require(argc == 3, "Usage: check-wide-lanes CHECKPOINT 16|32");
        const std::string_view width(argv[2]);
        require(width == "16" || width == "32", "Expected width 16 or 32");
        require(decoder_batch_max_lanes >= (width == "16" ? 16 : 32),
                "Decoder capacity is below the requested width");
        run(argv[1], width == "16" ? 16 : 32);
        std::cout << "Wide native lanes passed\n";
    } catch (const std::exception& failure) {
        std::cerr << failure.what() << '\n';
        return 1;
    }
}
