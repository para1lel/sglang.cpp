#include "sglang/decoder_batch.h"
#include "sglang/decoder_sampling.h"
#include "sglang/text_ops.h"
#include "sglang/tokenizer/tokenizer.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace sglang;
using decoder::SamplingParameters;
constexpr int slots = decoder_batch_max_lanes;
constexpr int vocabulary = model_vocab_size;
constexpr int hidden = 5120;
constexpr int group_width = 4;
using Selection = std::array<bool, slots>;
using Score = const __nv_bfloat16*;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check(cudaError_t status) {
    require(status == cudaSuccess, cudaGetErrorString(status));
}

struct Scoring {
    Score logits;
    const int* winner;
};

struct Sampling {
    Score logits;
    const int* winner;
    std::uint64_t seed;
    int position;
};

struct Transfer {
    const int* source;
    const int* destination;
};

struct Head {
    int grouped = 0;
    std::vector<Score> projections;
    std::vector<Scoring> argmax;
    std::map<TextLogitCandidate*, Score> scratch;
};

struct Trace {
    bool enabled = false;
    bool capturing = false;
    bool reject_capture = false;
    int rejected = 0;
    int captures = 0;
    int replays = 0;
    Head eager;
    Head captured;
    const Head* replayed = nullptr;
    std::map<cudaGraphExec_t, Head> graphs;
    std::vector<Sampling> samples;
    std::vector<Transfer> copied;

    void reset() {
        eager = {};
        replayed = nullptr;
        samples.clear();
        copied.clear();
        captures = 0;
        replays = 0;
        rejected = 0;
    }
};

Trace trace;

void observe_group(Head& head, dim3 grid, dim3 block, void** arguments) {
    require(*static_cast<const int*>(arguments[1]) == vocabulary &&
                grid.x == vocabulary / 8 && grid.y == 1 && grid.z == 1 &&
                block.x == 128 && block.y == 1 && block.z == 1,
            "Unexpected grouped head launch geometry");
    std::array<TextBf16Projection, group_width> rows;
    std::memcpy(rows.data(), arguments[2], sizeof(rows));
    std::set<Score> inputs;
    for (const auto& row : rows) {
        require(row.input && row.output && inputs.insert(row.input).second,
                "Grouped projections did not receive distinct inputs");
        head.projections.push_back(row.output);
    }
    ++head.grouped;
}

void observe_kernel(std::string_view symbol, dim3 grid, dim3 block, void** arguments) {
    auto& head = trace.capturing ? trace.captured : trace.eager;
    if (symbol.find("bf16_lm_head_batch_kernel") != symbol.npos) {
        observe_group(head, grid, block, arguments);
    } else if (symbol.find("bf16_gemv_kernel") != symbol.npos &&
               *static_cast<const int*>(arguments[2]) == vocabulary &&
               *static_cast<const int*>(arguments[3]) == hidden) {
        head.projections.push_back(*static_cast<__nv_bfloat16**>(arguments[4]));
    } else if (symbol.find("argmax_tiles_kernel") != symbol.npos &&
               *static_cast<const int*>(arguments[1]) == vocabulary) {
        const auto scratch = *static_cast<TextLogitCandidate**>(arguments[2]);
        const auto logits = *static_cast<Score*>(arguments[0]);
        require(head.scratch.emplace(scratch, logits).second, "Duplicate argmax tile launch");
    } else if (symbol.find("argmax_finish_kernel") != symbol.npos) {
        const auto scratch = *static_cast<TextLogitCandidate**>(arguments[0]);
        if (const auto found = head.scratch.find(scratch); found != head.scratch.end()) {
            const auto* winner = *static_cast<TextLogitCandidate**>(arguments[2]);
            head.argmax.push_back({found->second, &winner->token_id});
            head.scratch.erase(found);
        }
    }
}

Selection select(std::initializer_list<int> lanes) {
    Selection selected{};
    for (const int slot : lanes) selected[slot] = true;
    return selected;
}

std::uint32_t mask(const Selection& selected) {
    std::uint32_t bits = 0;
    for (int slot = 0; slot < slots; ++slot) {
        if (selected[slot]) bits |= std::uint32_t{1} << slot;
    }
    return bits;
}

struct Scenario {
    Selection selected;
    Selection prefill{};
    bool graph = false;
    bool capture = false;
    bool reject = false;
    bool group = false;
};

class CheckpointRun {
public:
    CheckpointRun(const Decoder& decoder, bool graph, bool candidate, const char* prompt_path,
                  const char* output_path)
        : batch_(decoder, {.cuda_graphs = graph, .max_lanes = slots,
                           .max_num_batched_tokens = 32768}),
          candidate_(candidate), output_(output_path, std::ios::binary), logits_(vocabulary) {
        std::ifstream input(prompt_path);
        prompt_.assign(std::istream_iterator<int>(input), {});
        require(input.eof() && !prompt_.empty(), "Expected a whitespace-delimited prompt fixture");
        require(output_.good(), "Unable to open logits output");
    }

    void admit(int slot, int length, std::uint64_t seed, bool sampled) {
        std::vector<int> ids;
        for (int index = 0; index < length; ++index) {
            ids.push_back(prompt_[(slot * 19 + index + seed) % prompt_.size()]);
        }
        DecoderOptions options;
        options.max_new_tokens = 24;
        options.seed = seed;
        if (sampled) {
            options.temperature = 0.95;
            options.top_k = 64;
            options.top_p = 0.99;
        }
        require(batch_.admit(slot, std::move(ids), options) == 0, "Unexpected prefix restore");
        lanes_[slot] = {sampled, seed, 0, length};
    }

    void cancel(int slot) { batch_.cancel(slot); }
    void step(const char* label, const Scenario& scenario);
    int non_greedy() const { return non_greedy_; }

private:
    struct Lane {
        bool sampled = false;
        std::uint64_t seed = 0;
        int position = 0;
        int prompt_rows = 0;
    };

    void record(const char* label, int slot, int token, const Scoring& score);

    DecoderBatch batch_;
    bool candidate_;
    std::ofstream output_;
    std::vector<__nv_bfloat16> logits_;
    std::vector<int> prompt_;
    std::array<Lane, slots> lanes_;
    int non_greedy_ = 0;
};

void CheckpointRun::record(const char* label, int slot, int token, const Scoring& score) {
    check(cudaMemcpy(logits_.data(), score.logits, logits_.size() * sizeof(logits_[0]),
                     cudaMemcpyDeviceToHost));
    float maximum = -std::numeric_limits<float>::infinity();
    int greedy = -1;
    for (int index = 0; index < vocabulary; ++index) {
        const float value = __bfloat162float(logits_[index]);
        require(std::isfinite(value), "Nonfinite checkpoint logit");
        if (value > maximum) {
            maximum = value;
            greedy = index;
        }
    }
    auto& lane = lanes_[slot];
    if (lane.sampled) {
        non_greedy_ += greedy != token;
    } else {
        require(greedy == token, "Greedy token did not select the bound logits");
    }
    output_.write(reinterpret_cast<const char*>(logits_.data()),
                  logits_.size() * sizeof(logits_[0]));
    require(output_.good(), "Unable to write logits");
    std::cout << "token " << label << ' ' << slot << ' ' << lane.position << ' ' << token
              << ' ' << greedy << ' ' << lane.sampled << '\n';
    ++lane.position;
}

void CheckpointRun::step(const char* label, const Scenario& scenario) {
    DecoderBatchStepStats stats;
    trace.reset();
    trace.reject_capture = scenario.reject;
    trace.enabled = true;
    const auto output = batch_.advance_chunked(scenario.selected, &stats);
    trace.enabled = false;
    const auto& head = trace.replayed ? *trace.replayed : trace.eager;
    std::cout << "route " << label << " grouped=" << head.grouped
              << " projected=" << head.projections.size() << " argmax=" << head.argmax.size()
              << " sampled=" << trace.samples.size() << " copied=" << trace.copied.size()
              << " captures=" << trace.captures << " replays=" << trace.replays
              << " rejected=" << trace.rejected << " graph_mask=" << stats.graph_slots
              << std::endl;
    require(head.grouped == (candidate_ && scenario.group), "Unexpected grouped LM head dispatch");
    require(trace.captures == scenario.capture && trace.replays == scenario.graph &&
                trace.rejected == scenario.reject && !trace.reject_capture &&
                stats.graph_slots == (scenario.graph ? mask(scenario.selected) : 0),
            "Unexpected graph route");
    require(!trace.replayed || trace.eager.projections.empty(), "Duplicate eager head on replay");
    const auto count = head.argmax.size();
    require(head.scratch.empty() && head.projections.size() == count &&
                trace.copied.size() == count &&
                std::set<Score>(head.projections.begin(), head.projections.end()).size() == count,
            "Unexpected projection or winner bindings");
    std::set<const int*> sources;
    std::set<const int*> destinations;
    std::size_t scored = 0;
    std::size_t sampled = 0;
    for (int slot = 0; slot < slots; ++slot) {
        const auto& result = output[slot];
        const auto& lane = lanes_[slot];
        require(result.decode == (scenario.selected[slot] && !scenario.prefill[slot]) &&
                    result.token.has_value() == scenario.selected[slot] &&
                    stats.prefill_rows[slot] == (scenario.prefill[slot] ? lane.prompt_rows : 0) &&
                    !result.finish_reason && !result.sampling_failed && !result.constraint_failed &&
                    !result.extra_token_count && !result.mtp_drafted_tokens,
                "Unexpected checkpoint output or K0 route");
        if (!result.token) continue;
        require(scored < count, "Missing scored request");
        const auto& score = head.argmax[scored];
        const auto& copy = trace.copied[scored];
        require(score.logits == head.projections[scored] && score.winner == copy.source &&
                    sources.insert(copy.source).second &&
                    destinations.insert(copy.destination).second &&
                    *copy.destination == *result.token,
                "Projection, argmax and winner copy are bound to different requests");
        if (lane.sampled) {
            require(sampled < trace.samples.size(), "Missing sampler launch");
            const auto& sample = trace.samples[sampled++];
            require(sample.logits == score.logits && sample.winner == score.winner &&
                        sample.seed == lane.seed && sample.position == lane.position,
                    "Sampler is bound to the wrong request");
        }
        record(label, slot, *result.token, score);
        ++scored;
    }
    require(scored == count && sampled == trace.samples.size(), "Extra score or sample");
}

void exercise(CheckpointRun& run, bool graph, bool reject) {
    const auto first = select({2, 11, 19, 27});
    run.admit(27, 31, 9001, false);
    run.admit(2, 19, 9002, true);
    run.admit(19, 43, 9003, true);
    run.admit(11, 23, 9004, false);
    run.step("prefill", {first, first});
    run.step("width4-warm", {.selected = first, .group = true});
    run.step("width4-second", {.selected = first, .graph = graph && !reject,
                               .capture = graph && !reject, .reject = reject, .group = true});
    run.step("width4-third", {.selected = first, .graph = graph && !reject, .group = true});
    const auto narrow = select({2, 19, 27});
    run.step("width3-warm", {narrow});
    run.step("width3-second", {.selected = narrow, .graph = graph, .capture = graph});
    const auto new_lane = select({5});
    run.admit(5, 29, 9010, false);
    run.step("mixed", {select({2, 5, 11, 19, 27}), new_lane});
    run.step("width5", {select({2, 5, 11, 19, 27})});
    run.cancel(11);
    const auto packed = select({2, 5, 19, 27});
    run.step("width4-reordered", {.selected = packed, .graph = graph && !reject, .group = true});
    run.cancel(2);
    run.admit(11, 37, 9011, true);
    run.step("reused-slot-prefill", {select({11}), select({11})});
    run.step("width4-reused", {.selected = select({5, 11, 19, 27}), .graph = graph && !reject,
                               .group = true});
    require(run.non_greedy() > 0, "Seeded sampling never selected a non-greedy token");
}

}

extern "C" cudaError_t __real_cudaLaunchKernel(const void*, dim3, dim3, void**, std::size_t,
                                               cudaStream_t);
extern "C" cudaError_t __wrap_cudaLaunchKernel(const void* function, dim3 grid, dim3 block,
                                               void** args, std::size_t shared,
                                               cudaStream_t stream) {
    if (trace.enabled) {
        const char* name = nullptr;
        check(cudaFuncGetName(&name, function));
        observe_kernel(name, grid, block, args);
    }
    return __real_cudaLaunchKernel(function, grid, block, args, shared, stream);
}

extern "C" cudaError_t __real_cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode);
extern "C" cudaError_t __wrap_cudaStreamBeginCapture(cudaStream_t stream,
                                                     cudaStreamCaptureMode mode) {
    if (trace.enabled && trace.reject_capture) {
        trace.reject_capture = false;
        ++trace.rejected;
        return cudaErrorStreamCaptureUnsupported;
    }
    const auto status = __real_cudaStreamBeginCapture(stream, mode);
    if (trace.enabled && status == cudaSuccess) {
        trace.capturing = true;
        trace.captured = {};
        ++trace.captures;
    }
    return status;
}

extern "C" cudaError_t __real_cudaStreamEndCapture(cudaStream_t, cudaGraph_t*);
extern "C" cudaError_t __wrap_cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph) {
    const auto status = __real_cudaStreamEndCapture(stream, graph);
    trace.capturing = false;
    return status;
}

extern "C" cudaError_t __real_cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t,
                                                    unsigned long long);
extern "C" cudaError_t __wrap_cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph,
                                                    unsigned long long flags) {
    const auto status = __real_cudaGraphInstantiate(exec, graph, flags);
    if (trace.enabled && status == cudaSuccess) trace.graphs[*exec] = std::move(trace.captured);
    return status;
}

extern "C" cudaError_t __real_cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
extern "C" cudaError_t __wrap_cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) {
    const auto status = __real_cudaGraphLaunch(exec, stream);
    if (trace.enabled && status == cudaSuccess) {
        trace.replayed = &trace.graphs.at(exec);
        ++trace.replays;
    }
    return status;
}

extern "C" cudaError_t __real_cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind,
                                              cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* destination, const void* source,
                                              std::size_t bytes, cudaMemcpyKind kind,
                                              cudaStream_t stream) {
    if (trace.enabled && kind == cudaMemcpyDeviceToHost && bytes == sizeof(int)) {
        trace.copied.push_back({static_cast<const int*>(source),
                                static_cast<const int*>(destination)});
    }
    return __real_cudaMemcpyAsync(destination, source, bytes, kind, stream);
}

cudaError_t real_sample(Score, int, int*, SamplingParameters, int, void*, cudaStream_t)
    asm("__real__ZN6sglang7decoder13sample_logitsEPK13__nv_bfloat16iPiNS0_18SamplingParametersEiPv"
        "P11CUstream_st");
cudaError_t wrap_sample(Score, int, int*, SamplingParameters, int, void*, cudaStream_t)
    asm("__wrap__ZN6sglang7decoder13sample_logitsEPK13__nv_bfloat16iPiNS0_18SamplingParametersEiPv"
        "P11CUstream_st");

cudaError_t wrap_sample(Score scores, int count, int* token, SamplingParameters parameters,
                        int position, void* workspace, cudaStream_t stream) {
    if (trace.enabled) {
        require(count == vocabulary && !trace.capturing, "Unexpected sampler shape or graph");
        trace.samples.push_back({scores, token, parameters.seed, position});
    }
    return real_sample(scores, count, token, parameters, position, workspace, stream);
}

int main(int argc, char** argv) try {
    require(argc == 6,
            "Usage: grouped-head-routes CHECKPOINT PROMPT base|candidate "
            "eager|graph|rejected LOGITS");
    const std::string_view arm(argv[3]);
    const std::string_view mode(argv[4]);
    require(arm == "base" || arm == "candidate", "Unexpected test arm");
    require(mode == "eager" || mode == "graph" || mode == "rejected", "Unexpected graph mode");
    const bool graph = mode != "eager";
    const Decoder decoder(argv[1]);
    CheckpointRun run(decoder, graph, arm == "candidate", argv[2], argv[5]);
    exercise(run, graph, mode == "rejected");
    std::cout << "Grouped head routes passed with " << run.non_greedy() << " non-greedy tokens\n";
} catch (const std::exception& error) {
    std::cerr << "Grouped head route check failed: " << error.what() << '\n';
    return 1;
}
