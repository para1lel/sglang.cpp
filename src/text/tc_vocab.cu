#include "sglang/decoder_internal.h"
#include "sglang/decoder_cuda.cuh"
#include "sglang/text/decode_proj.h"
#include "sglang/text/decode_proj_sm100.cuh"
#include "sglang/text/mtp/tc_vocab.h"
#include "sglang/text/mtp/tc_vocab_certificate.cuh"

#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace sglang::decoder {
namespace {

using namespace tc_vocab;

static_assert(vocab_size == vocabulary && hidden_size == width);

constexpr int score_columns = 32;
constexpr int narrow_columns = 16;
constexpr int score_stages = 4;
constexpr int score_chunk = 128;
template <int Columns>
constexpr int score_shared_bytes = score_stages * decode_proj::stage_bytes<Columns>;

template <int Columns>
constexpr unsigned score_instruction =
    (decode_proj::tile_rows / 16 << 24) | (Columns / 8 << 17) |
    (1u << 10) | (1u << 7) | (2u << 4);

static_assert(vocabulary % decode_proj::tile_rows == 0 && width % score_chunk == 0);
static_assert(rows <= score_columns && rows > 16);
static_assert(tc_vocab_narrow_rows <= 8);

bool setting_enabled(const char* name) {
    const auto* value = std::getenv(name);
    return value && std::string_view(value) == "1";
}

void check_direct_status(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        const auto recorded = cudaGetLastError();
        if (!CudaError::terminal(status) && CudaError::terminal(recorded)) {
            check(recorded, operation);
        }
        check(status, operation);
    }
}

template <int Columns>
__device__ void multiply_scores(decode_proj::DecodePipeline<score_stages>& pipe,
                                unsigned shared) {
    using namespace decode_proj;
    for (int index = 0; index < width / score_chunk; ++index) {
        const int stage = index % score_stages;
        wait_barrier(&pipe.copy_ready[stage], (index / score_stages) & 1);
        fence_after_sync();
        const unsigned start = shared + stage * stage_bytes<Columns>;
        #pragma unroll
        for (int chunk = 0; chunk < score_chunk; chunk += 32) {
            const auto weight = matrix_desc(start + chunk);
            const auto input = matrix_desc(start + weight_bytes + chunk);
            const unsigned accumulate = index | chunk;
            asm volatile("{ .reg .pred accumulate;\n"
                         "setp.ne.u32 accumulate, %4, 0;\n"
                         "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, accumulate; }"
                         :: "r"(pipe.tmem), "l"(weight), "l"(input),
                            "r"(score_instruction<Columns>), "r"(accumulate) : "memory");
        }
        commit_mma(&pipe.copy_empty[stage]);
    }
    commit_mma(&pipe.accum_ready[0]);
}

template <int Columns>
__device__ void store_scores(const decode_proj::DecodePipeline<score_stages>& pipe,
                             int first_token, int* scores, int count) {
    using namespace decode_proj;
    wait_barrier(&pipe.accum_ready[0], 0);
    fence_after_sync();
    const int token = first_token + threadIdx.x;
    const unsigned source = pipe.tmem + (static_cast<unsigned>(threadIdx.x / warp_size *
                                                               warp_size) << 16);
    if constexpr (Columns == narrow_columns) {
        unsigned values[8];
        load_tmem8(source, values);
        for (int row = 0; row < count; ++row) {
            scores[row * vocabulary + token] = static_cast<int>(values[row]);
        }
    } else {
        unsigned values[24];
        load_tmem16(source, values);
        load_tmem8(source + 16, values + 16);
        #pragma unroll
        for (int row = 0; row < rows; ++row) {
            scores[row * vocabulary + token] = static_cast<int>(values[row]);
        }
    }
    fence_before_sync();
}

template <int Columns>
__global__ __launch_bounds__(decode_proj::threads, 2) void vocabulary_scores(
    const __grid_constant__ CUtensorMap weight, const __grid_constant__ CUtensorMap input,
    int* scores, int count) {
    using namespace decode_proj;
    alignas(1024) extern __shared__ unsigned char tiles[];
    __shared__ DecodePipeline<score_stages> pipe;
    const int warp = threadIdx.x / warp_size;
    const int first_token = blockIdx.x * tile_rows;
    if (threadIdx.x == 0) {
        init_pipeline(pipe);
    }
    if (warp == mma_warp) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(shared_address(&pipe.tmem)), "r"(score_columns) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    const auto shared = shared_address(tiles);
    if (warp < consumer_warps) {
        store_scores<Columns>(pipe, first_token, scores, count);
    } else if (warp == mma_warp && threadIdx.x % warp_size == 0) {
        multiply_scores<Columns>(pipe, shared);
    } else if (warp == copy_warp && threadIdx.x % warp_size == 0) {
        produce<score_chunk, Columns>(pipe, shared, &weight, &input, first_token, 0, 0,
                                      width / score_chunk);
    }
    __syncthreads();
    if (warp == mma_warp) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(pipe.tmem), "r"(score_columns) : "memory");
    }
}

}

struct TcVocabWeights {
    CUtensorMap map{};
    const Bf16* original{};
    Stats* stats{};
    bool trace = false;
};

struct TcVocabScratch {
    TcVocabScratch(Allocations& memory, int capacity)
        : packed(memory.allocate<Bf16>(capacity * width)),
          quantized(memory.allocate<std::uint8_t>(capacity * width)),
          maxima(memory.allocate<float>(capacity)),
          stats(memory.allocate<Stats>(capacity)),
          scores(memory.allocate<int>(capacity * vocabulary)),
          lower(memory.allocate<float>(capacity * tiles)),
          threshold(memory.allocate<float>(capacity)),
          count(memory.allocate<int>(capacity)), ids(memory.allocate<int>(capacity * slots)),
          partial(memory.allocate<Winner>(capacity * refine_tiles)) {}

    Bf16* packed;
    std::uint8_t* quantized;
    CUtensorMap map{};
    std::array<CUtensorMap, tc_vocab_narrow_rows> narrow_maps{};
    bool wide = false;
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
    decode_proj::LaunchTarget target{};
    const auto status = decode_proj::prepare_launch(memory.stream(), target);
    if (!status || *status == cudaErrorNotSupported || !decode_proj::aligned16(quantized)) {
        return {};
    }
    check_direct_status(*status, "Prepare native INT8 verification projection");
    auto weights = std::make_shared<TcVocabWeights>();
    if (decode_proj::encode_map(target, weights->map, quantized, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                1, vocabulary, width, decode_proj::tile_rows) != CUDA_SUCCESS) {
        return {};
    }
    check_direct_status(cudaFuncSetAttribute(vocabulary_scores<score_columns>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            score_shared_bytes<score_columns>),
                        "Configure native INT8 verification projection");
    check_direct_status(cudaFuncSetAttribute(vocabulary_scores<narrow_columns>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            score_shared_bytes<narrow_columns>),
                        "Configure narrow INT8 verification projection");
    weights->original = original;
    weights->stats = memory.allocate<Stats>(vocabulary);
    weights->trace = setting_enabled("SGLANG_VERIFY_TC_VOCAB_TRACE");
    shared_stats<<<vocabulary, threads, 0, memory.stream()>>>(
        original, quantized, maxima, weights->stats);
    check(cudaGetLastError(), "Prepare shared INT8 verification weight bounds");
    std::fputs("sglang tc_vocab: enabled B200 native INT8 Tensor Core projection\n", stderr);
    return weights;
}

std::shared_ptr<const TcVocabScratch> prepare_tc_vocab_scratch(Allocations& memory, bool wide) {
    decode_proj::LaunchTarget target{};
    const auto status = decode_proj::prepare_launch(memory.stream(), target);
    if (!status || *status == cudaErrorNotSupported) {
        return {};
    }
    check_direct_status(*status, "Prepare native INT8 verification scratch");
    auto scratch = std::make_shared<TcVocabScratch>(memory, wide ? rows : tc_vocab_narrow_rows);
    if (wide && decode_proj::encode_map(target, scratch->map, scratch->quantized,
                                       CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, rows, width,
                                       score_columns) != CUDA_SUCCESS) {
        return {};
    }
    for (int count = 1; count <= tc_vocab_narrow_rows; ++count) {
        if (decode_proj::encode_map(target, scratch->narrow_maps[count - 1], scratch->quantized,
                                    CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, count, width,
                                    narrow_columns) != CUDA_SUCCESS) {
            return {};
        }
    }
    scratch->wide = wide;
    return scratch;
}

bool try_tc_vocab(const TcVocabWeights& weights, const TcVocabScratch& scratch,
                  const TcVocabRow* batch, int count, cudaStream_t stream) {
    const bool narrow = count >= 1 && count <= tc_vocab_narrow_rows;
    if (!narrow && (count != rows || !scratch.wide)) return false;
    Inputs inputs{};
    Outputs outputs{};
    for (int row = 0; row < count; ++row) {
        inputs.rows[row] = batch[row].input;
        outputs.rows[row] = batch[row].winner;
    }
    pack_inputs<<<count, threads, 0, stream>>>(inputs, scratch.packed);
    check(cudaGetLastError(), "Pack shared INT8 verification inputs");
    check_direct_status(quantize_draft_head_int8(scratch.packed, count, scratch.quantized,
                                                scratch.maxima, stream),
                        "Quantize shared INT8 verification inputs");
    shared_stats<<<count, threads, 0, stream>>>(
        scratch.packed, scratch.quantized, scratch.maxima, scratch.stats);
    check(cudaGetLastError(), "Prepare shared INT8 verification input bounds");
    constexpr int blocks = vocabulary / decode_proj::tile_rows;
    if (narrow) {
        vocabulary_scores<narrow_columns>
            <<<blocks, decode_proj::threads, score_shared_bytes<narrow_columns>, stream>>>(
                weights.map, scratch.narrow_maps[count - 1], scratch.scores, count);
    } else {
        vocabulary_scores<score_columns>
            <<<blocks, decode_proj::threads, score_shared_bytes<score_columns>, stream>>>(
                weights.map, scratch.map, scratch.scores, count);
    }
    check(cudaGetLastError(), "Run native INT8 verification projection");
    lower_bounds<<<dim3(tiles, count), threads, 0, stream>>>(
        scratch.scores, weights.stats, scratch.stats, scratch.lower);
    check(cudaGetLastError(), "Bound shared INT8 verification scores");
    thresholds<<<count, threads, 0, stream>>>(scratch.lower, scratch.threshold);
    check(cudaGetLastError(), "Reduce shared INT8 verification bounds");
    check_direct_status(cudaMemsetAsync(scratch.count, 0, count * sizeof(int), stream),
                        "Clear verification vocabulary candidate counts");
    candidates<<<dim3((vocabulary + threads - 1) / threads, count), threads, 0, stream>>>(
        scratch.scores, scratch.threshold, scratch.count, scratch.ids);
    check(cudaGetLastError(), "Collect shared INT8 verification candidates");
    refine<<<dim3(refine_tiles, count), threads, 0, stream>>>(
        weights.original, inputs, scratch.count, scratch.ids, scratch.partial);
    check(cudaGetLastError(), "Refine shared INT8 verification candidates");
    finish<<<count, threads, 0, stream>>>(scratch.partial, outputs);
    check(cudaGetLastError(), "Certify greedy verification vocabulary winners");
    if (weights.trace) {
        std::array<int, rows> counts{};
        check_direct_status(cudaMemcpyAsync(counts.data(), scratch.count, count * sizeof(int),
                                            cudaMemcpyDeviceToHost, stream),
                            "Read verification vocabulary candidate counts");
        check_direct_status(cudaStreamSynchronize(stream), "Finish verification vocabulary trace");
        std::fprintf(stderr, "sglang tc_vocab: counts=");
        for (int row = 0; row < count; ++row) {
            std::fprintf(stderr, "%d,", counts[row]);
        }
        std::fputc('\n', stderr);
    }
    return true;
}

}
