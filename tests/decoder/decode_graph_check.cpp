#include "sglang/decoder.h"

#include <cuda_runtime_api.h>

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

constexpr int steps = 4;
constexpr int vocabulary = 248320;
constexpr int chunk_rows = 128;
unsigned graph_launches = 0;
unsigned graph_captures = 0;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void match(const sglang::DecoderResult& graph, const sglang::DecoderResult& eager) {
    require(graph.finish_reason == sglang::DecoderFinishReason::length &&
            eager.finish_reason == graph.finish_reason, "Greedy finish reason differs");
    require(graph.output_token_ids.size() == steps &&
            graph.output_token_ids == eager.output_token_ids, "Greedy token IDs differ");
    require(graph.step_top_logits.size() == steps && eager.step_top_logits.size() == steps,
            "Missing greedy score rows");
    for (int step = 0; step < steps; ++step) {
        const auto& graph_row = graph.step_top_logits[step];
        const auto& eager_row = eager.step_top_logits[step];
        require(graph_row.size() == vocabulary && eager_row.size() == vocabulary,
                "Incomplete greedy vocabulary scores");
        for (int rank = 0; rank < vocabulary; ++rank) {
            require(graph_row[rank].token_id == eager_row[rank].token_id &&
                    std::memcmp(&graph_row[rank].logit, &eager_row[rank].logit, sizeof(float)) == 0,
                    "Greedy vocabulary score bits or ranking differ");
        }
    }
}

}

extern "C" cudaError_t __real_cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
extern "C" cudaError_t __real_cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode);

extern "C" cudaError_t __wrap_cudaGraphLaunch(cudaGraphExec_t graph, cudaStream_t stream) {
    const auto status = __real_cudaGraphLaunch(graph, stream);
    if (status == cudaSuccess) {
        ++graph_launches;
    }
    return status;
}

extern "C" cudaError_t __wrap_cudaStreamBeginCapture(cudaStream_t stream,
                                                      cudaStreamCaptureMode mode) {
    const auto status = __real_cudaStreamBeginCapture(stream, mode);
    if (status == cudaSuccess) {
        ++graph_captures;
    }
    return status;
}

int main(int argc, char** argv) try {
    require(argc == 2, "Usage: decode-graph-check CHECKPOINT");
    const sglang::Decoder decoder(argv[1]);
    std::vector<int> prompt(chunk_rows + 1, 760);
    prompt.back() = 369;
    sglang::DecoderOptions options;
    options.max_new_tokens = steps;
    options.top_logits = vocabulary;
    const auto captures_before = graph_captures;

    const auto graph = decoder.generate(prompt, options);
    require(graph_launches == steps - 1, "Default greedy decode missed CUDA graph launches");
    require(graph_captures == captures_before + 1, "Default greedy decode must capture once");
    const auto reused = decoder.generate(prompt, options);
    require(graph_launches == 2 * (steps - 1), "Reused greedy decode missed CUDA graph launches");
    require(graph_captures == captures_before + 1, "Reused greedy decode recaptured the graph");
    options.eager = true;
    const auto eager = decoder.generate(prompt, options);
    require(graph_launches == 2 * (steps - 1), "Eager greedy decode launched a CUDA graph");
    require(graph_captures == captures_before + 1, "Eager greedy decode captured a CUDA graph");
    match(graph, eager);
    match(reused, eager);
    std::cout << "Greedy CUDA graph and eager score checks passed\n";
} catch (const std::exception& error) {
    std::cerr << "Greedy CUDA graph check failed: " << error.what() << '\n';
    return 1;
}
