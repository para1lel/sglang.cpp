#include "sglang/decoder_internal.h"
#include "sglang/decoder_cuda.cuh"
#include "sglang/text/mtp/tc_vocab.h"
#include "sglang/text/mtp/tc_vocab_certificate.cuh"

#include <cublasLt.h>

#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace sglang::decoder {
namespace {

using namespace tc_vocab;

static_assert(vocab_size == vocabulary && hidden_size == width);

bool setting_enabled(const char* name) {
    const auto* value = std::getenv(name);
    return value && std::string_view(value) == "1";
}

int attribute(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t key) {
    int value = -1;
    if (cublasLtMatmulAlgoConfigGetAttribute(&algo, key, &value, sizeof(value), nullptr) !=
        CUBLAS_STATUS_SUCCESS) {
        return -1;
    }
    return value;
}

struct Matmul {
    cublasLtHandle_t handle{};
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t weights{}, inputs{}, scores{};
    cublasLtMatmulAlgo_t algorithm{};

    ~Matmul() {
        if (scores) cublasLtMatrixLayoutDestroy(scores);
        if (inputs) cublasLtMatrixLayoutDestroy(inputs);
        if (weights) cublasLtMatrixLayoutDestroy(weights);
        if (operation) cublasLtMatmulDescDestroy(operation);
        if (handle) cublasLtDestroy(handle);
    }

    bool prepare();
    cublasStatus_t run(const std::uint8_t* weight, const std::uint8_t* input, int* output,
                       cudaStream_t stream) const;
};

bool Matmul::prepare() {
    constexpr auto ok = CUBLAS_STATUS_SUCCESS;
    const cublasOperation_t transpose = CUBLAS_OP_T;
    if (cublasLtGetVersion() != 130800 ||
        cublasLtCreate(&handle) != ok ||
        cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32I, CUDA_R_32I) != ok ||
        cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                      &transpose, sizeof(transpose)) != ok ||
        cublasLtMatrixLayoutCreate(&weights, CUDA_R_8I, width, vocabulary, width) != ok ||
        cublasLtMatrixLayoutCreate(&inputs, CUDA_R_8I, width, rows, width) != ok ||
        cublasLtMatrixLayoutCreate(&scores, CUDA_R_32I, vocabulary, rows, vocabulary) != ok) {
        return false;
    }
    cublasLtMatmulPreference_t preference{};
    if (cublasLtMatmulPreferenceCreate(&preference) != ok) {
        return false;
    }
    std::array<cublasLtMatmulHeuristicResult_t, 64> options;
    int available = 0;
    constexpr std::uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE |
                                        CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    constexpr std::uint64_t workspace = 64 << 20;
    const bool found =
        cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                            &workspace, sizeof(workspace)) == ok &&
        cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                            &reduction, sizeof(reduction)) == ok &&
        cublasLtMatmulAlgoGetHeuristic(handle, operation, weights, inputs, scores, scores,
                                      preference, options.size(), options.data(), &available) == ok;
    cublasLtMatmulPreferenceDestroy(preference);
    constexpr int profiled_entry = 7;
    if (!found || available <= profiled_entry) {
        return false;
    }
    const auto& entry = options[profiled_entry];
    if (entry.state != ok || entry.workspaceSize != 0 ||
        attribute(entry.algo, CUBLASLT_ALGO_CONFIG_ID) != 71 ||
        attribute(entry.algo, CUBLASLT_ALGO_CONFIG_TILE_ID) != 23 ||
        attribute(entry.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM) != 1 ||
        attribute(entry.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME) !=
            CUBLASLT_REDUCTION_SCHEME_NONE) {
        return false;
    }
    algorithm = entry.algo;
    return true;
}

cublasStatus_t Matmul::run(const std::uint8_t* weight, const std::uint8_t* input, int* output,
                          cudaStream_t stream) const {
    const int alpha = 1, beta = 0;
    return cublasLtMatmul(handle, operation, &alpha, weight, weights, input, inputs, &beta,
                         output, scores, output, scores, &algorithm, nullptr, 0, stream);
}

}

struct TcVocabWeights {
    Matmul matmul;
    const Bf16* original{};
    const std::uint8_t* quantized{};
    Stats* stats{};
    bool trace = false;
};

struct TcVocabScratch {
    explicit TcVocabScratch(Allocations& memory)
        : packed(memory.allocate<Bf16>(rows * width)),
          quantized(memory.allocate<std::uint8_t>(rows * width)),
          maxima(memory.allocate<float>(rows)),
          stats(memory.allocate<Stats>(rows)), scores(memory.allocate<int>(rows * vocabulary)),
          lower(memory.allocate<float>(rows * tiles)), threshold(memory.allocate<float>(rows)),
          count(memory.allocate<int>(rows)), ids(memory.allocate<int>(rows * slots)),
          partial(memory.allocate<Winner>(rows * refine_tiles)) {}

    Bf16* packed;
    std::uint8_t* quantized;
    float* maxima;
    Stats* stats;
    int* scores;
    float* lower;
    float* threshold;
    int* count;
    int* ids;
    Winner* partial;
};

std::shared_ptr<const TcVocabWeights> prepare_tc_vocab(const Bf16* original,
                                                       const std::uint8_t* quantized,
                                                       const float* maxima, Allocations& memory) {
    auto weights = std::make_shared<TcVocabWeights>();
    if (!weights->matmul.prepare()) {
        std::fputs("sglang tc_vocab: approved B200 cuBLASLt algorithm unavailable\n", stderr);
        return {};
    }
    weights->original = original;
    weights->quantized = quantized;
    weights->stats = memory.allocate<Stats>(vocabulary);
    weights->trace = setting_enabled("SGLANG_VERIFY_TC_VOCAB_TRACE");
    shared_stats<<<vocabulary, threads, 0, memory.stream()>>>(
        original, quantized, maxima, weights->stats);
    check(cudaGetLastError(), "Prepare shared INT8 verification weight bounds");
    std::fputs("sglang tc_vocab: enabled B200 shared INT8 cuBLASLt algo=71 tile=23 split=1\n",
               stderr);
    return weights;
}

std::shared_ptr<const TcVocabScratch> prepare_tc_vocab_scratch(Allocations& memory) {
    return std::make_shared<TcVocabScratch>(memory);
}

bool try_tc_vocab(const TcVocabWeights& weights, const TcVocabScratch& scratch,
                  const std::array<TcVocabRow, tc_vocab_rows>& batch, cudaStream_t stream) {
    Inputs inputs;
    Outputs outputs;
    for (int row = 0; row < rows; ++row) {
        inputs.rows[row] = batch[row].input;
        outputs.rows[row] = batch[row].winner;
    }
    pack_inputs<<<rows, threads, 0, stream>>>(inputs, scratch.packed);
    check(cudaGetLastError(), "Pack shared INT8 verification inputs");
    check(quantize_draft_head_int8(scratch.packed, rows, scratch.quantized, scratch.maxima, stream),
          "Quantize shared INT8 verification inputs");
    shared_stats<<<rows, threads, 0, stream>>>(
        scratch.packed, scratch.quantized, scratch.maxima, scratch.stats);
    check(cudaGetLastError(), "Prepare shared INT8 verification input bounds");
    const auto status = weights.matmul.run(weights.quantized, scratch.quantized, scratch.scores,
                                           stream);
    if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
        return false;
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(cublasLtGetStatusString(status));
    }
    lower_bounds<<<dim3(tiles, rows), threads, 0, stream>>>(
        scratch.scores, weights.stats, scratch.stats, scratch.lower);
    thresholds<<<rows, threads, 0, stream>>>(scratch.lower, scratch.threshold);
    check(cudaMemsetAsync(scratch.count, 0, rows * sizeof(int), stream),
          "Clear verification vocabulary candidate counts");
    candidates<<<dim3((vocabulary + threads - 1) / threads, rows), threads, 0, stream>>>(
        scratch.scores, scratch.threshold, scratch.count, scratch.ids);
    refine<<<dim3(refine_tiles, rows), threads, 0, stream>>>(
        weights.original, inputs, scratch.count, scratch.ids, scratch.partial);
    finish<<<rows, threads, 0, stream>>>(scratch.partial, outputs);
    check(cudaGetLastError(), "Certify greedy verification vocabulary winners");
    if (weights.trace) {
        std::array<int, rows> counts{};
        check(cudaMemcpyAsync(counts.data(), scratch.count, sizeof(counts), cudaMemcpyDeviceToHost,
                              stream), "Read verification vocabulary candidate counts");
        check(cudaStreamSynchronize(stream), "Finish verification vocabulary trace");
        std::fprintf(stderr, "sglang tc_vocab: counts=");
        for (int count : counts) {
            std::fprintf(stderr, "%d,", count);
        }
        std::fputc('\n', stderr);
    }
    return true;
}

}
