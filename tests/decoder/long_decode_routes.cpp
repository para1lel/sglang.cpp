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
#include <sstream>
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
    int combine = 0;
    int width = 0;
    int tensor = 0;
    bool consistent = true;
};
Routes routes;
bool watch = false;

struct MtpRoutes {
    int readers = 0;
    int merges = 0;
    int legacy = 0;
    unsigned requests = slots;
};
MtpRoutes* mtp_routes = nullptr;

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
                  routes.combine == 0,
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
                           int split_lanes, std::uint32_t graph_mask = 0,
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
    const auto expected_launches = split_lanes * gqa_layers;
    std::cout << "route " << label << " split=" << routes.split << " merge=" << routes.merge
              << " replay=" << routes.replay
              << " combine=" << routes.combine << " width=" << routes.width << " graph_mask="
              << stats.graph_slots << " target_rows=" << stats.target_rows
              << " tensor=" << routes.tensor << " wall_ms=" << elapsed << std::endl;
    check(routes.consistent && routes.width == int(split_lanes != 0) &&
              routes.split == expected_launches && routes.combine == expected_launches &&
              routes.merge == 0 && routes.replay == 0 && routes.tensor == 0,
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
              LogitCapture& capture, const std::vector<int>& teacher) {
    constexpr std::array selected_slots{3, 8, 17, 24, 29};
    constexpr std::array labels{"long-first", "long-second", "long-third"};
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
        const auto tokens = step(batch, labels[index], all, wide_slots, 0, &capture);
        for (int lane = 0; capture.force && lane < wide_slots; ++lane) {
            check(tokens[selected_slots[lane]] == capture.targets[lane],
                  "Teacher ID did not reach the corresponding slot");
        }
    }
    capture.force = false;
    step(batch, "long-four", select({3, 8, 17, 24}), 4);
    step(batch, "long-four-repeat", select({3, 8, 17, 24}), 4);
}

void mixed_lengths(DecoderBatch& batch, const std::vector<int>& prompt, DecoderOptions greedy,
                   LogitCapture& capture) {
    DecoderOptions sample = greedy;
    sample.temperature = 0.65;
    sample.top_k = 8;
    sample.top_p = 0.9;
    sample.seed = 416;
    admit(batch, 29, prompt, 32767, greedy);
    admit(batch, 3, prompt, 32769, sample);
    admit(batch, 17, prompt, 32773, greedy);
    admit(batch, 8, prompt, 193, sample);
    admit(batch, 24, prompt, 127, greedy);
    const auto all = select({29, 3, 17, 8, 24});
    const auto first = prefill(batch, all);
    const auto wide = step(batch, "wide", all, 3, 0, &capture);
    const auto retained = select({29, 3, 17, 24});
    step(batch, "four", retained, 3);
    batch.cancel(8);
    admit(batch, 8, prompt, 193, sample);
    const auto reused_first = prefill(batch, select({8}));
    check(reused_first[8] == first[8], "Reused sampled lane changed first ID");
    const auto reused = step(batch, "reused", all, 3, 0, &capture);
    check(reused[8] == wide[8], "Reused sampled lane changed second ID");
    const auto sparse = select({3, 24});
    step(batch, "sparse", sparse, 1);
    step(batch, "regroup", all, 3, 0, &capture);
    const auto short_lanes = select({8, 24});
    step(batch, "short-pair", short_lanes, 0);
    step(batch, "short-pair-repeat", short_lanes, 0);
    for (const int slot : {3, 8, 17, 24, 29}) batch.cancel(slot);
}

void narrow_lengths(DecoderBatch& batch, const std::vector<int>& prompt, DecoderOptions greedy) {
    const auto lanes = select({1, 9, 21});
    admit(batch, 1, prompt, 8192, greedy);
    admit(batch, 9, prompt, 8212, greedy);
    admit(batch, 21, prompt, 100, greedy);
    prefill(batch, lanes);
    step(batch, "mixed-first", lanes, 2);
    step(batch, "mixed-second", lanes, 2);
    step(batch, "mixed-third", lanes, 2);
    const auto singleton = select({21});
    constexpr std::uint32_t single_mask = 1u << 21;
    step(batch, "single-eager", singleton, 0);
    step(batch, "single-capture", singleton, 1, single_mask);
    step(batch, "single-replay", singleton, 0, single_mask);
    for (const int slot : {1, 9, 21}) batch.cancel(slot);
}

void check_crossing_tokens(const DecoderBatchStep& result, int& ordinal) {
    constexpr std::array cycle{760, 45366, 369, 7629, 13, 198};
    check(result.decode && !result.finish_reason && result.mtp_drafted_tokens == 4 &&
              result.mtp_accepted_tokens == 4 && result.mtp_verified_rows == 5 &&
              result.mtp_replayed_rows == 0 && result.extra_token_count == 4,
          "W5 K4 acceptance changed");
    check(result.token == cycle[ordinal++ % cycle.size()], "W5 first ID changed");
    for (int index = 0; index < result.extra_token_count; ++index) {
        check(result.extra_tokens[index] == cycle[ordinal++ % cycle.size()], "W5 draft ID changed");
    }
}

void mtp_crossings(const char* checkpoint, const char* fixture) {
    constexpr int prompt_length = 32768;
    constexpr int quota = 8192;
    constexpr std::array ahead{0, 2, 3, 4, 7, 10, 12, 24, 27, 29, 30, 31};
    const Decoder decoder(checkpoint, 0, true);
    check(slots == 32 && decoder.uses_b200_operators(), "W5 route check requires a B200");
    DecoderBatch batch(decoder, {true, slots, 0, quota});
    DecoderOptions options;
    options.mtp_steps = 4;
    options.max_new_tokens = 192;
    options.ignore_eos = true;
    std::ifstream file(fixture);
    for (int slot = 0; slot < slots; ++slot) {
        std::string line;
        check(bool(std::getline(file, line)), "Missing W5 fixture lane");
        std::istringstream ids(line);
        std::vector<int> prompt{std::istream_iterator<int>(ids), {}};
        check(ids.eof() && prompt.size() == prompt_length, "Invalid W5 fixture lane");
        check(batch.admit(slot, std::move(prompt), options) == 0, "Unexpected W5 prefix hit");
    }
    Selection all;
    all.fill(true);
    DecoderBatchStepStats stats;
    for (int start = 0; start < prompt_length; start += quota / slots) {
        const auto steps = batch.advance_chunked(all, &stats);
        check(stats.target_rows == quota && stats.verification_lanes == 0, "W5 prefill changed");
        if (start + quota / slots == prompt_length) {
            for (const auto& step : steps) check(step.token == 760, "W5 seed changed");
        }
    }
    std::array<int, slots> ordinals;
    ordinals.fill(1);
    Selection lead{};
    for (int slot : ahead) lead[slot] = true;
    const auto first = batch.advance_chunked(lead, &stats);
    check(stats.verification_lanes == int(ahead.size()), "W5 lead roster changed");
    for (int slot : ahead) check_crossing_tokens(first[slot], ordinals[slot]);

    int crossing_waves = 0;
    for (int wave = 0; wave <= 25; ++wave) {
        int crossings = 0;
        for (int ordinal : ordinals) {
            const int position = prompt_length + ordinal - 1;
            crossings += position / 64 != (position + 4) / 64;
        }
        MtpRoutes launches;
        mtp_routes = &launches;
        const auto steps = batch.advance_chunked(all, &stats);
        mtp_routes = nullptr;
        check(stats.verification_lanes == slots && stats.verification_rows == 5 * slots &&
                  std::all_of(stats.prefill_rows.begin(), stats.prefill_rows.end(),
                              [](int rows) { return rows == 0; }), "W5 verify geometry changed");
        check(launches.readers == gqa_layers && launches.merges == gqa_layers && !launches.legacy,
              "W5 verification lost its grouped long reader or merge");
        for (int slot = 0; slot < slots; ++slot) check_crossing_tokens(steps[slot], ordinals[slot]);
        if (crossings) {
            check(crossings == (crossing_waves % 2 == 0 ? 12 : 20), "W5 crossing setup changed");
            ++crossing_waves;
            std::cout << "K4 crossing lanes=" << crossings << " readers=" << launches.readers
                      << " merges=" << launches.merges << " legacy=" << launches.legacy << '\n';
        }
    }
    check(crossing_waves == 4, "Missing mixed K64 and P128 crossing waves");
}

std::vector<std::vector<int>> read_mtp_cohort(const char* fixture) {
    std::ifstream file(fixture);
    check(file.good(), "Cannot open 32K MTP cohort");
    std::vector<std::vector<int>> prompts;
    for (std::string line; std::getline(file, line);) {
        std::istringstream ids(line);
        auto& prompt = prompts.emplace_back();
        prompt.assign(std::istream_iterator<int>(ids), {});
        check(ids.eof() && prompt.size() == 32768, "Invalid 32K MTP fixture lane");
    }
    check(prompts.size() == 16 || prompts.size() == slots, "Expected W4 or W5 MTP cohort");
    return prompts;
}

void mtp_mixed(const char* checkpoint, const char* fixture) {
    auto prompts = read_mtp_cohort(fixture);
    const int count = static_cast<int>(prompts.size());
    const Decoder decoder(checkpoint, 0, true);
    check(decoder.uses_b200_operators(), "Mixed MTP route check requires a B200");
    DecoderBatch batch(decoder, {true, count, 0, 8192});
    DecoderOptions options;
    options.mtp_steps = 4;
    options.max_new_tokens = 96;
    options.ignore_eos = true;
    Selection selected{};
    std::fill_n(selected.begin(), count, true);
    for (int slot = 0; slot < count; ++slot) {
        if (slot % 4 == 0) prompts[slot].resize(32765);
        if (slot % 4 == 1) prompts[slot].resize(32384);
        check(batch.admit(slot, std::move(prompts[slot]), options) == 0, "Unexpected prefix hit");
    }
    DecoderBatchStepStats stats;
    for (int pass = 0; std::any_of(selected.begin(), selected.end(), [](bool x) { return x; });
         ++pass) {
        check(pass < 140, "Mixed MTP prefill did not reach ready width");
        const auto steps = batch.advance_chunked(selected, &stats);
        check(!stats.verification_lanes, "A parked mixed MTP lane entered verification");
        for (int slot = 0; slot < count; ++slot) {
            if (!steps[slot].token) continue;
            if (slot % 4 != 1) {
                check(steps[slot].token == (slot % 4 == 0 ? 7629 : 760), "Mixed MTP seed changed");
            }
            selected[slot] = false;
        }
    }
    std::fill_n(selected.begin(), count, true);
    MtpRoutes launches;
    launches.requests = count / 4 * 3;
    mtp_routes = &launches;
    const auto steps = batch.advance_chunked(selected, &stats);
    mtp_routes = nullptr;
    check(stats.verification_lanes == count && stats.verification_rows == 5 * count,
          "Mixed MTP verification geometry changed");
    check(launches.readers == gqa_layers && launches.merges == gqa_layers && launches.legacy,
          "Eligible partial verification suffix lost its long reader or eager remainder");
    for (int slot = 0; slot < count; ++slot) {
        if (slot % 4 != 1) {
            int ordinal = slot % 4 == 0 ? 4 : 1;
            check_crossing_tokens(steps[slot], ordinal);
        } else {
            check(steps[slot].decode && steps[slot].token && !steps[slot].finish_reason,
                  "Below-threshold verification lane failed");
        }
    }
    std::cout << "Mixed K4 lanes=" << count << " readers=" << launches.readers
              << " merges=" << launches.merges << " legacy=" << launches.legacy << '\n';
}

}

using namespace sglang::decoder;

void real_select(RequestProgress*, Request&, Workspace&, const DecoderOptions&,
                 const ConstraintWorkspace&, const SamplingParameters*, void*, cudaStream_t,
                 const AdjustedLogitsWorkspace*)
    asm("__real__ZN6sglang7decoder15RequestProgress6selectERNS0_7RequestERNS0_9WorkspaceERKNS_"
        "14DecoderOptionsERKNS0_19ConstraintWorkspaceEPKNS0_18SamplingParametersEPvP11CUstream_st"
        "PKNS0_23AdjustedLogitsWorkspaceE");
void wrap_select(RequestProgress*, Request&, Workspace&, const DecoderOptions&,
                 const ConstraintWorkspace&, const SamplingParameters*, void*, cudaStream_t,
                 const AdjustedLogitsWorkspace*)
    asm("__wrap__ZN6sglang7decoder15RequestProgress6selectERNS0_7RequestERNS0_9WorkspaceERKNS_"
        "14DecoderOptionsERKNS0_19ConstraintWorkspaceEPKNS0_18SamplingParametersEPvP11CUstream_st"
        "PKNS0_23AdjustedLogitsWorkspaceE");

void wrap_select(RequestProgress* progress, Request& request, Workspace& work,
                 const DecoderOptions& options, const ConstraintWorkspace& scratch,
                 const SamplingParameters* sampling, void* sampling_workspace,
                 cudaStream_t stream, const AdjustedLogitsWorkspace* adjusted) {
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
    real_select(progress, request, work, options, scratch, sampling, sampling_workspace, stream,
                adjusted);
}

extern "C" cudaError_t __real_cudaLaunchKernel(const void*, dim3, dim3, void**, std::size_t,
                                               cudaStream_t);

static void record_launch(std::string_view symbol, dim3 grid, dim3 block) {
    if (mtp_routes) {
        const bool reader = symbol.find("decode_long") != symbol.npos;
        const bool merge = symbol.find("target_split_combine") != symbol.npos &&
                           grid.z == 5 * mtp_routes->requests;
        const bool legacy = (symbol.find("gqa_decode") != symbol.npos &&
                             symbol.find("split_kernel") != symbol.npos && grid.z != 4) ||
                            symbol.find("reuse_split_kernel") != symbol.npos;
        if (reader) {
            check(grid.x == 16 && grid.y == 4 && grid.z == mtp_routes->requests && block.x == 480,
                  "Unexpected MTP long reader geometry");
            check(mtp_routes->readers == mtp_routes->merges, "Missing previous MTP merge");
            ++mtp_routes->readers;
        }
        if (merge) {
            check(grid.x == 24 && grid.y == 1 && block.x == 256 &&
                      mtp_routes->readers == mtp_routes->merges + 1, "Unexpected MTP merge");
            ++mtp_routes->merges;
        }
        mtp_routes->legacy += legacy;
        return;
    }
    const bool split = symbol.find("split_kernel") != symbol.npos &&
                       symbol.find("gqa_decode") != symbol.npos;
    const bool combine = symbol.find("target_split_combine") != symbol.npos;
    if (!split && !combine) return;
    const bool next = split ? routes.split == routes.combine :
                              routes.split == routes.combine + 1;
    routes.width = grid.z;
    routes.consistent &= grid.z == 1 && next &&
        block.x == (split ? 128 : 256) && block.y == 1 && block.z == 1 &&
        grid.x == (split ? 16 : 24) && grid.y == (split ? 4 : 1);
    routes.split += split;
    routes.combine += combine;
}

extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 grid, dim3 block,
                                          void** args, std::size_t shared, cudaStream_t stream) {
    static const auto launch = reinterpret_cast<decltype(&__cudaLaunchKernel)>(
        dlsym(RTLD_NEXT, "__cudaLaunchKernel"));
    check(launch != nullptr, "Cannot resolve CUDA kernel launch");
    if (watch || mtp_routes) {
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
    if (watch || mtp_routes) {
        const char* name = nullptr;
        check(cudaFuncGetName(&name, function) == cudaSuccess, "Cannot identify decode kernel");
        record_launch(name, grid, block);
        if (mtp_routes) return __real_cudaLaunchKernel(function, grid, block, args, shared, stream);
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
    if (argc == 4 && std::string_view(argv[2]) == "--mtp-crossings") {
        mtp_crossings(argv[1], argv[3]);
        return 0;
    }
    if (argc == 4 && std::string_view(argv[2]) == "--mtp-mixed") {
        mtp_mixed(argv[1], argv[3]);
        return 0;
    }
    check(argc >= 3 && argc <= 5,
          "Usage: long-decode-routes CHECKPOINT "
          "{PROMPT [LOGITS [TEACHER]]|--mtp-crossings W5|--mtp-mixed W4_OR_W5_IDS}");
    const Decoder decoder(argv[1], 0, false);
    check(decoder.uses_b200_operators(), "Long decode route check requires a B200");
    const DecoderBatchOptions options{.cuda_graphs = true, .max_lanes = slots,
                                      .max_num_batched_tokens = 32768};
    const auto prompt = read_prompt(argv[2]);
    const auto teacher = argc == 5 ? read_prompt(argv[4]) : std::vector<int>{};
    check(teacher.empty() || (
          (teacher.size() == 3 * wide_slots || teacher.size() == 3 * wide_slots + 1) &&
          std::all_of(teacher.begin(), teacher.end(),
                      [](int token) { return token >= 0 && token < vocabulary; })),
          "Expected 15 in-vocabulary teacher IDs with an optional shared first ID");
    LogitCapture capture(argc >= 4 ? argv[3] : nullptr);
    DecoderOptions greedy;
    greedy.max_new_tokens = 24;
    greedy.seed = 9001;
    {
        DecoderBatch batch(decoder, options);
        mixed_lengths(batch, prompt, greedy, capture);
    }
    {
        DecoderBatch batch(decoder, options);
        all_long(batch, prompt, greedy, capture, teacher);
    }
    DecoderBatch batch(decoder, options);
    narrow_lengths(batch, prompt, greedy);
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
