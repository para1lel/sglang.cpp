#include "sglang/decoder_batch.h"
#include "sglang/text_ops.h"
#include "sglang/tokenizer/tokenizer.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using namespace sglang;
using Clock = std::chrono::steady_clock;
constexpr int slots = decoder_batch_max_lanes;
using Selection = std::array<bool, slots>;
constexpr int gqa_layers = 16;
constexpr int vocabulary = model_vocab_size;
constexpr int wide_slots = 5;

struct Routes {
    int split = 0;
    int merge = 0;
    int replay = 0;
    int width = 0;
    bool consistent = true;
};
Routes routes;
bool watch = false;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct LogitCapture {
    explicit LogitCapture(const char* path) {
        if (path) {
            file.open(path, std::ios::binary);
            check(file.good(), "Unable to open logits output");
            check(cudaMallocHost(reinterpret_cast<void**>(&scores),
                                 wide_slots * vocabulary * sizeof(*scores)) == cudaSuccess,
                  "Unable to allocate logits capture");
        }
    }

    ~LogitCapture() {
        if (scores) cudaFreeHost(scores);
    }

    LogitCapture(const LogitCapture&) = delete;
    LogitCapture& operator=(const LogitCapture&) = delete;

    cudaError_t copy(const __nv_bfloat16* source, int count, cudaStream_t stream) {
        check(count == vocabulary && rows < wide_slots, "Unexpected logits shape");
        return cudaMemcpyAsync(scores + rows++ * vocabulary, source, count * sizeof(*scores),
                               cudaMemcpyDeviceToHost, stream);
    }

    void flush() {
        if (!scores) return;
        check(rows == wide_slots, "Missing eager logits rows");
        file.write(reinterpret_cast<const char*>(scores), rows * vocabulary * sizeof(*scores));
        check(file.good(), "Unable to write logits");
        rows = 0;
    }

    std::ofstream file;
    __nv_bfloat16* scores = nullptr;
    int rows = 0;
};

LogitCapture* logit_capture = nullptr;

Selection select(std::initializer_list<int> lanes) {
    Selection selected{};
    for (int lane : lanes) selected[lane] = true;
    return selected;
}

std::vector<int> read_prompt(const char* file) {
    std::ifstream input(file);
    std::vector<int> ids{std::istream_iterator<int>(input), {}};
    check(!ids.empty() && input.eof(), "Expected a whitespace-delimited token ID fixture");
    return ids;
}

void admit(DecoderBatch& batch, int lane, const std::vector<int>& source, int length,
           const DecoderOptions& options) {
    std::vector<int> prompt;
    prompt.reserve(length);
    for (int index = 0; index < length; ++index) {
        prompt.push_back(source[(index + lane * 19) % source.size()]);
    }
    check(batch.admit(lane, std::move(prompt), options) == 0, "Unexpected prefix cache hit");
}

std::array<int, slots> prefill(DecoderBatch& batch, Selection remaining) {
    std::array<int, slots> first;
    first.fill(-1);
    for (int iteration = 0; std::any_of(remaining.begin(), remaining.end(),
                                        [](bool selected) { return selected; }); ++iteration) {
        check(iteration < 12, "Prefill failed to make progress");
        DecoderBatchStepStats stats;
        routes = {};
        watch = true;
        const auto outputs = batch.advance_chunked(remaining, &stats);
        watch = false;
        check(routes.split == 0 && routes.merge == 0 && routes.replay == 0,
              "Prefill invoked long decode");
        for (int slot = 0; slot < slots; ++slot) {
            const auto& output = outputs[slot];
            check(!output.decode && !output.finish_reason && !output.sampling_failed &&
                      !output.constraint_failed && (!output.token || remaining[slot]),
                  "Unexpected prefill response");
            if (output.token) {
                first[slot] = *output.token;
                std::cout << "token prefill " << slot << ' ' << *output.token << '\n';
                remaining[slot] = false;
            }
        }
        std::cout << "prefill_step " << iteration << std::endl;
    }
    return first;
}

std::array<int, slots> step(DecoderBatch& batch, const char* label, const Selection& selected,
                           int expected_width, std::uint32_t graph_mask = 0,
                           LogitCapture* capture = nullptr) {
    DecoderBatchStepStats stats;
    routes = {};
    watch = true;
    logit_capture = capture && capture->scores ? capture : nullptr;
    const auto start = Clock::now();
    const auto outputs = batch.advance_chunked(selected, &stats);
    const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    logit_capture = nullptr;
    watch = false;
    const auto expected_launches = expected_width == 0 ? 0 : gqa_layers;
    std::cout << "route " << label << " split=" << routes.split << " merge=" << routes.merge
              << " replay=" << routes.replay << " width=" << routes.width << " graph_mask="
              << stats.graph_slots << " target_rows=" << stats.target_rows
              << " wall_ms=" << elapsed << std::endl;
    check(routes.consistent && routes.width == expected_width &&
              routes.split == expected_launches && routes.merge == expected_launches &&
              routes.replay == expected_launches,
          "Unexpected long decode launch sequence");
    check(stats.graph_slots == graph_mask, "Unexpected decode graph mask");
    if (capture) capture->flush();
    std::array<int, slots> tokens;
    tokens.fill(-1);
    for (int slot = 0; slot < slots; ++slot) {
        const auto& output = outputs[slot];
        check(stats.prefill_rows[slot] == 0 && output.decode == selected[slot] &&
                  output.token.has_value() == selected[slot] && !output.sampling_failed &&
                  !output.constraint_failed && !output.finish_reason, "Unexpected decode response");
        if (output.token) {
            tokens[slot] = *output.token;
            std::cout << "token " << label << ' ' << slot << ' ' << *output.token << '\n';
        }
    }
    return tokens;
}

}

cudaError_t real_argmax(const __nv_bfloat16*, int, TextLogitCandidate*, TextLogitCandidate*,
                        cudaStream_t)
    asm("__real__ZN6sglang11text_argmaxEPK13__nv_bfloat16i"
        "PNS_18TextLogitCandidateES4_P11CUstream_st");
cudaError_t wrap_argmax(const __nv_bfloat16*, int, TextLogitCandidate*, TextLogitCandidate*,
                        cudaStream_t)
    asm("__wrap__ZN6sglang11text_argmaxEPK13__nv_bfloat16i"
        "PNS_18TextLogitCandidateES4_P11CUstream_st");

cudaError_t wrap_argmax(const __nv_bfloat16* scores, int count, TextLogitCandidate* scratch,
                        TextLogitCandidate* output, cudaStream_t stream) {
    if (logit_capture) {
        const auto status = logit_capture->copy(scores, count, stream);
        if (status != cudaSuccess) return status;
    }
    return real_argmax(scores, count, scratch, output, stream);
}

extern "C" cudaError_t __real_cudaLaunchKernel(const void*, dim3, dim3, void**, std::size_t,
                                               cudaStream_t);
extern "C" cudaError_t __wrap_cudaLaunchKernel(const void* function, dim3 grid, dim3 block,
                                               void** args, std::size_t shared,
                                               cudaStream_t stream) {
    if (watch) {
        const char* name = nullptr;
        check(cudaFuncGetName(&name, function) == cudaSuccess, "Cannot identify decode kernel");
        const std::string_view symbol(name);
        const bool split = symbol.find("split_long_decode_attention") != symbol.npos;
        const bool replay = symbol.find("replay_long_paged_decode") != symbol.npos;
        const bool merge = symbol.find("merge_decode_attention") != symbol.npos &&
                           symbol.find("Li32E") != symbol.npos;
        if (split || merge || replay) {
            if (!routes.width) routes.width = grid.z;
            const bool next = split ? routes.split == routes.replay
                : merge ? routes.merge + 1 == routes.split : routes.replay + 1 == routes.merge;
            routes.consistent &= grid.z == routes.width && block.x == 256 &&
                grid.x == (split ? 16u : 24u) && grid.y == (split ? 12u : 1u) && next;
            routes.split += split;
            routes.merge += merge;
            routes.replay += replay;
        }
    }
    return __real_cudaLaunchKernel(function, grid, block, args, shared, stream);
}

int main(int argc, char** argv) try {
    check(argc == 5 || argc == 6,
          "Usage: long-decode-routes CHECKPOINT PROMPT base|candidate tensor-core|exact [LOGITS]");
    const std::string_view arm(argv[3]);
    const std::string_view policy(argv[4]);
    check(arm == "base" || arm == "candidate", "Unexpected test arm");
    check(policy == "tensor-core" || policy == "exact", "Unexpected precision policy");
    const bool exact = policy == "exact";
    const int grouped = arm == "candidate" && !exact ? 3 : 0;
    const Decoder decoder(argv[1], 0, false,
                          exact ? Fp8PrefillPolicy::exact : Fp8PrefillPolicy::tensor_core);
    DecoderBatch batch(decoder, {.cuda_graphs = true, .max_lanes = slots,
                                 .max_num_batched_tokens = 32768});
    const auto prompt = read_prompt(argv[2]);
    LogitCapture capture(argc == 6 ? argv[5] : nullptr);
    DecoderOptions greedy;
    greedy.max_new_tokens = 24;
    greedy.seed = 9001;
    DecoderOptions sample = greedy;
    sample.temperature = 0.65;
    sample.top_k = 8;
    sample.top_p = 0.9;
    sample.seed = 416;
    admit(batch, 29, prompt, exact ? 131 : 32767, greedy);
    admit(batch, 3, prompt, exact ? 129 : 32769, sample);
    admit(batch, 17, prompt, exact ? 134 : 32773, greedy);
    admit(batch, 8, prompt, 193, sample);
    admit(batch, 24, prompt, 127, greedy);
    const auto all = select({29, 3, 17, 8, 24});
    const auto first = prefill(batch, all);
    const auto wide = step(batch, "wide", all, grouped, 0, &capture);
    const auto retained = select({29, 3, 17, 24});
    step(batch, "four", retained, 0);
    batch.cancel(8);
    admit(batch, 8, prompt, 193, sample);
    const auto reused_first = prefill(batch, select({8}));
    check(reused_first[8] == first[8], "Reused sampled lane changed first ID");
    const auto reused = step(batch, "reused", all, grouped, 0, &capture);
    check(reused[8] == wide[8], "Reused sampled lane changed second ID");
    const auto sparse = select({3, 24});
    step(batch, "sparse", sparse, 0);
    step(batch, "regroup", all, grouped, 0, &capture);
    const auto short_lanes = select({8, 24});
    constexpr std::uint32_t short_mask = (std::uint32_t{1} << 8) | (std::uint32_t{1} << 24);
    step(batch, "short-graph", short_lanes, 0, short_mask);
    step(batch, "short-replay", short_lanes, 0, short_mask);
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
