#include "sglang/decoder_batch.h"
#include "sglang/decoder_internal.h"
#include "sglang/text_ops.h"
#include "sglang/tokenizer/tokenizer.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

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
    int prepare = 0;
    int combine = 0;
    int width = 0;
    int tensor = 0;
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
            if (cudaMallocHost(reinterpret_cast<void**>(&targets),
                               wide_slots * sizeof(*targets)) != cudaSuccess) {
                cudaFreeHost(scores);
                check(false, "Unable to allocate teacher targets");
            }
        }
    }

    ~LogitCapture() {
        if (scores) cudaFreeHost(scores);
        if (targets) cudaFreeHost(targets);
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
        check(rows == wide_slots, "Missing decode logits rows");
        file.write(reinterpret_cast<const char*>(scores), rows * vocabulary * sizeof(*scores));
        check(file.good(), "Unable to write logits");
        rows = 0;
    }

    std::ofstream file;
    __nv_bfloat16* scores = nullptr;
    int* targets = nullptr;
    int rows = 0;
    bool force = false;
    bool force_initial = false;
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
        check(routes.split == 0 && routes.merge == 0 && routes.replay == 0 &&
                  routes.prepare == 0 && routes.combine == 0,
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
                           LogitCapture* capture = nullptr, bool tensor = false,
                           bool fused_candidate = false, bool capture_graph = false) {
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
    const auto legacy_launches = fused_candidate ? 0 : expected_launches;
    const auto combined_launches = fused_candidate ? expected_launches : 0;
    std::cout << "route " << label << " split=" << routes.split << " merge=" << routes.merge
              << " replay=" << routes.replay << " prepare=" << routes.prepare
              << " combine=" << routes.combine << " width=" << routes.width << " graph_mask="
              << stats.graph_slots << " target_rows=" << stats.target_rows
              << " tensor=" << routes.tensor << " wall_ms=" << elapsed << std::endl;
    check(routes.consistent && routes.width == expected_width &&
              routes.split == expected_launches && routes.merge == legacy_launches &&
              routes.replay == legacy_launches && routes.combine == combined_launches &&
              routes.prepare == (capture_graph ? expected_launches : 0) &&
              routes.tensor == (tensor ? legacy_launches : 0),
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

void all_long(DecoderBatch& batch, const std::vector<int>& prompt, DecoderOptions greedy,
              LogitCapture& capture, bool candidate, bool legacy_tensor,
              const std::vector<int>& teacher) {
    constexpr std::array selected_slots{3, 8, 17, 24, 29};
    constexpr std::array labels{"tensor-first", "tensor-second", "tensor-third"};
    const auto all = select({3, 8, 17, 24, 29});
    for (int index = 0; index < wide_slots; ++index) {
        admit(batch, selected_slots[index], prompt, 32767 + index * 2, greedy);
    }
    const bool force_initial = teacher.size() == labels.size() * wide_slots + 1;
    if (force_initial) {
        capture.targets[0] = teacher.front();
        capture.force_initial = true;
        logit_capture = &capture;
    }
    const auto first = prefill(batch, all);
    logit_capture = nullptr;
    capture.force_initial = false;
    for (int lane = 0; force_initial && lane < wide_slots; ++lane) {
        check(first[selected_slots[lane]] == teacher.front(),
              "Initial teacher ID did not reach the corresponding slot");
    }
    capture.force = !teacher.empty();
    for (std::size_t index = 0; index < labels.size(); ++index) {
        if (capture.force) {
            const auto first_target = index * wide_slots + force_initial;
            std::copy_n(teacher.data() + first_target, wide_slots, capture.targets);
        }
        constexpr std::uint32_t all_mask =
            (1u << 3) | (1u << 8) | (1u << 17) | (1u << 24) | (1u << 29);
        const int width = !candidate ? wide_slots : index < 2 ? 8 : 0;
        const auto tokens = step(batch, labels[index], all, width,
                                 candidate && index > 0 ? all_mask : 0, &capture, legacy_tensor,
                                 candidate, candidate);
        for (int lane = 0; capture.force && lane < wide_slots; ++lane) {
            check(tokens[selected_slots[lane]] == capture.targets[lane],
                  "Teacher ID did not reach the corresponding slot");
        }
    }
    capture.force = false;
    constexpr std::uint32_t mask = (1u << 3) | (1u << 8) | (1u << 17) | (1u << 24);
    step(batch, "long-four", select({3, 8, 17, 24}), candidate ? 4 : 0,
         candidate ? 0 : mask, nullptr, false, candidate, candidate);
    step(batch, "long-four-replay", select({3, 8, 17, 24}), candidate ? 4 : 0,
         mask, nullptr, false, candidate, candidate);
}

void mixed_lengths(DecoderBatch& batch, const std::vector<int>& prompt, bool exact,
                   DecoderOptions greedy, LogitCapture& capture, int grouped, bool candidate) {
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
    constexpr std::uint32_t all_mask =
        (1u << 3) | (1u << 8) | (1u << 17) | (1u << 24) | (1u << 29);
    const auto first = prefill(batch, all);
    const auto wide = step(batch, "wide", all, candidate ? 8 : grouped,
                           0, &capture, false, candidate, candidate);
    const auto retained = select({29, 3, 17, 24});
    step(batch, "four", retained, candidate ? 4 : 0, 0, nullptr, false, candidate, candidate);
    batch.cancel(8);
    admit(batch, 8, prompt, 193, sample);
    const auto reused_first = prefill(batch, select({8}));
    check(reused_first[8] == first[8], "Reused sampled lane changed first ID");
    const auto reused = step(batch, "reused", all, candidate ? 8 : grouped,
                             candidate ? all_mask : 0, &capture, false, candidate, candidate);
    check(reused[8] == wide[8], "Reused sampled lane changed second ID");
    const auto sparse = select({3, 24});
    step(batch, "sparse", sparse, candidate ? 2 : 0, 0, nullptr, false, candidate, candidate);
    step(batch, "regroup", all, candidate ? 0 : grouped, candidate ? all_mask : 0,
         &capture, false, candidate);
    const auto short_lanes = select({8, 24});
    constexpr std::uint32_t short_mask = (std::uint32_t{1} << 8) | (std::uint32_t{1} << 24);
    step(batch, "short-graph", short_lanes, candidate ? 2 : 0, short_mask,
         nullptr, false, candidate, candidate);
    step(batch, "short-replay", short_lanes, 0, short_mask);
    for (const int slot : {3, 8, 17, 24, 29}) batch.cancel(slot);
}

void narrow_lengths(DecoderBatch& batch, const std::vector<int>& prompt, DecoderOptions greedy) {
    const auto lanes = select({1, 9, 21});
    constexpr std::uint32_t mask = (1u << 1) | (1u << 9) | (1u << 21);
    admit(batch, 1, prompt, 8192, greedy);
    admit(batch, 9, prompt, 8212, greedy);
    admit(batch, 21, prompt, 100, greedy);
    prefill(batch, lanes);
    step(batch, "narrow-eager", lanes, 4, 0, nullptr, false, true, true);
    step(batch, "narrow-capture", lanes, 4, mask, nullptr, false, true, true);
    step(batch, "narrow-replay", lanes, 0, mask);
    const auto singleton = select({21});
    constexpr std::uint32_t single_mask = 1u << 21;
    step(batch, "single-eager", singleton, 0);
    step(batch, "single-capture", singleton, 0, single_mask);
    step(batch, "single-replay", singleton, 0, single_mask);
    for (const int slot : {1, 9, 21}) batch.cancel(slot);
}

}

using namespace sglang::decoder;

void real_select(RequestProgress*, Request&, Workspace&, const DecoderOptions&,
                 const ConstraintWorkspace&, const SamplingParameters*, void*, cudaStream_t)
    asm("__real__ZN6sglang7decoder15RequestProgress6selectERNS0_7RequestERNS0_9WorkspaceERKNS_"
        "14DecoderOptionsERKNS0_19ConstraintWorkspaceEPKNS0_18SamplingParametersEPvP11CUstream_st");
void wrap_select(RequestProgress*, Request&, Workspace&, const DecoderOptions&,
                 const ConstraintWorkspace&, const SamplingParameters*, void*, cudaStream_t)
    asm("__wrap__ZN6sglang7decoder15RequestProgress6selectERNS0_7RequestERNS0_9WorkspaceERKNS_"
        "14DecoderOptionsERKNS0_19ConstraintWorkspaceEPKNS0_18SamplingParametersEPvP11CUstream_st");

void wrap_select(RequestProgress* progress, Request& request, Workspace& work,
                 const DecoderOptions& options, const ConstraintWorkspace& scratch,
                 const SamplingParameters* sampling, void* sampling_workspace,
                 cudaStream_t stream) {
    const int* teacher = nullptr;
    if (logit_capture) {
        if (logit_capture->force_initial) {
            check(progress->generated == 0, "Unexpected long prefill scoring");
            teacher = logit_capture->targets;
        } else {
            decoder::check(logit_capture->copy(work.logits, vocabulary, stream),
                           "Capture raw logits");
            if (logit_capture->force) {
                teacher = logit_capture->targets + logit_capture->rows - 1;
            }
        }
    }
    if (teacher) {
        check(!sampling && !options.constraint, "Teacher forcing needs greedy scoring");
        decoder::check(cudaMemcpyAsync(&work.argmax->token_id, teacher, sizeof(*teacher),
                                       cudaMemcpyHostToDevice, stream), "Upload forced token");
    }
    real_select(progress, request, work, options, scratch, sampling, sampling_workspace, stream);
}

extern "C" cudaError_t __real_cudaLaunchKernel(const void*, dim3, dim3, void**, std::size_t,
                                               cudaStream_t);

static void record_launch(std::string_view symbol, dim3 grid, dim3 block) {
    const bool prepare = symbol.find("prepare_kernel") != symbol.npos &&
                         symbol.find("gqa_decode") != symbol.npos;
    const bool split = symbol.find("split_kernel") != symbol.npos &&
                       symbol.find("gqa_decode") != symbol.npos;
    const bool combine = symbol.find("combine_replay") != symbol.npos;
    if (!prepare && !split && !combine) return;
    const auto width = split ? grid.z : grid.y;
    const bool prepared = routes.prepare == routes.combine + 1;
    const bool next = prepare ? routes.prepare == routes.combine
        : split ? routes.split == routes.combine && prepared
                : routes.split == routes.combine + 1;
    if (!routes.width) routes.width = width;
    routes.consistent &= width == unsigned(routes.width) && next &&
        block.x == (split ? 128 : 256) && block.y == 1 && block.z == 1 &&
        grid.x == (prepare ? 28 : split ? 16 : 24) && (split ? grid.y == 4 : grid.z == 1);
    routes.prepare += prepare;
    routes.split += split;
    routes.combine += combine;
}

extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 grid, dim3 block,
                                          void** args, std::size_t shared, cudaStream_t stream) {
    static const auto launch = reinterpret_cast<decltype(&__cudaLaunchKernel)>(
        dlsym(RTLD_NEXT, "__cudaLaunchKernel"));
    check(launch != nullptr, "Cannot resolve CUDA kernel launch");
    if (watch) {
        static void* const driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        check(driver != nullptr, "Cannot load the CUDA driver");
        static const auto get_name = reinterpret_cast<decltype(&cuKernelGetName)>(
            dlsym(driver, "cuKernelGetName"));
        const char* name = nullptr;
        check(get_name && get_name(&name, kernel) == CUDA_SUCCESS,
              "Cannot identify compiler-generated CUDA launch");
        record_launch(name, grid, block);
    }
    return launch(kernel, grid, block, args, shared, stream);
}

extern "C" cudaError_t __wrap_cudaLaunchKernel(const void* function, dim3 grid, dim3 block,
                                               void** args, std::size_t shared,
                                               cudaStream_t stream) {
    if (watch) {
        const char* name = nullptr;
        check(cudaFuncGetName(&name, function) == cudaSuccess, "Cannot identify decode kernel");
        const std::string_view symbol(name);
        const bool tensor = symbol.find("split_long_decode_tensor") != symbol.npos;
        const bool split = tensor || symbol.find("split_long_decode_attention") != symbol.npos;
        const bool replay = symbol.find("replay_long_paged_decode") != symbol.npos;
        const bool merge = symbol.find("merge_decode_attention") != symbol.npos &&
                           symbol.find("Li32E") != symbol.npos;
        if (split || merge || replay) {
            if (!routes.width) routes.width = grid.z;
            const bool next = split ? routes.split == routes.replay
                : merge ? routes.merge + 1 == routes.split : routes.replay + 1 == routes.merge;
            routes.consistent &= grid.z == routes.width && block.x == (tensor ? 128 : 256) &&
                grid.x == (split ? 16u : 24u) && grid.y == (tensor ? 4u : split ? 12u : 1u) &&
                block.y == 1 && block.z == 1 && next;
            routes.split += split;
            routes.tensor += tensor;
            routes.merge += merge;
            routes.replay += replay;
        }
    }
    return __real_cudaLaunchKernel(function, grid, block, args, shared, stream);
}

int main(int argc, char** argv) try {
    check(argc >= 5 && argc <= 7,
          "Usage: long-decode-routes CHECKPOINT PROMPT ARM POLICY [LOGITS [TEACHER]]");
    const std::string_view arm(argv[3]);
    const std::string_view policy(argv[4]);
    check(arm == "base" || arm == "paired" || arm == "legacy-tensor" || arm == "candidate",
          "Unexpected test arm");
    check(policy == "tensor-core" || policy == "exact", "Unexpected precision policy");
    const bool exact = policy == "exact";
    const bool candidate = arm == "candidate" && !exact;
    const int grouped = arm != "base" && !exact ? 3 : 0;
    const Decoder decoder(argv[1], 0, false,
                          exact ? Fp8PrefillPolicy::exact : Fp8PrefillPolicy::tensor_core);
    const DecoderBatchOptions options{.cuda_graphs = true, .max_lanes = slots,
                                      .max_num_batched_tokens = 32768};
    const auto prompt = read_prompt(argv[2]);
    const auto teacher = argc == 7 ? read_prompt(argv[6]) : std::vector<int>{};
    check(teacher.empty() || (grouped &&
          (teacher.size() == 3 * wide_slots || teacher.size() == 3 * wide_slots + 1) &&
          std::all_of(teacher.begin(), teacher.end(),
                      [](int token) { return token >= 0 && token < vocabulary; })),
          "Expected 15 in-vocabulary teacher IDs with an optional shared first ID");
    LogitCapture capture(argc >= 6 ? argv[5] : nullptr);
    DecoderOptions greedy;
    greedy.max_new_tokens = 24;
    greedy.seed = 9001;
    {
        DecoderBatch batch(decoder, options);
        mixed_lengths(batch, prompt, exact, greedy, capture, grouped, candidate);
        if (grouped && !candidate) {
            all_long(batch, prompt, greedy, capture, false, arm == "legacy-tensor", teacher);
        }
    }
    if (candidate) {
        {
            DecoderBatch batch(decoder, options);
            all_long(batch, prompt, greedy, capture, true, false, teacher);
        }
        DecoderBatch batch(decoder, options);
        narrow_lengths(batch, prompt, greedy);
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
