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
constexpr int wide_columns = 64;
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
static_assert(tc_vocab_rows <= score_columns && tc_vocab_rows > 16);
static_assert(tc_vocab_wide_rows <= wide_columns && tc_vocab_wide_rows > 32);
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
    } else if constexpr (Columns == score_columns) {
        unsigned values[24];
        load_tmem16(source, values);
        load_tmem8(source + 16, values + 16);
        #pragma unroll
        for (int row = 0; row < tc_vocab_rows; ++row) {
            scores[row * vocabulary + token] = static_cast<int>(values[row]);
        }
    } else {
        unsigned values[40];
        load_tmem16(source, values);
        load_tmem16(source + 16, values + 16);
        load_tmem8(source + 32, values + 32);
        #pragma unroll
        for (int row = 0; row < tc_vocab_wide_rows; ++row) {
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
                     :: "r"(shared_address(&pipe.tmem)), "r"(tmem_columns<Columns, 1>) : "memory");
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
                     :: "r"(pipe.tmem), "r"(tmem_columns<Columns, 1>) : "memory");
    }
}

}

struct TcVocabWeights {
    CUtensorMap map{};
    const Bf16* original{};
    Stats* stats{};
    bool trace = false;
};

namespace {

struct ScoreRows {
    Bf16* packed{};
    std::uint8_t* quantized{};
    float* maxima{};
    Stats* stats{};
    int* scores{};
    float* lower{};
    float* threshold{};
    int* count{};
    int* ids{};
    Winner* partial{};
};

struct WideScoreStorage {
    std::uint8_t quantized[rows * width];
    Bf16 packed[rows * width];
    float maxima[rows];
    Stats stats[rows];
    int scores[rows * vocabulary];
    float lower[rows * tiles];
    float threshold[rows];
    int count[rows];
    int ids[rows * slots];
    Winner partial[rows * refine_tiles];
};

struct ScoreScratch {
    ScoreScratch(int device, int capacity)
        : memory(device), capacity(capacity) {
        buffers.quantized = memory.allocate<std::uint8_t>(capacity * width);
    }

    void allocate_rows() {
        buffers.packed = memory.allocate<Bf16>(capacity * width);
        buffers.maxima = memory.allocate<float>(capacity);
        buffers.stats = memory.allocate<Stats>(capacity);
        buffers.scores = memory.allocate<int>(capacity * vocabulary);
        buffers.lower = memory.allocate<float>(capacity * tiles);
        buffers.threshold = memory.allocate<float>(capacity);
        buffers.count = memory.allocate<int>(capacity);
        buffers.ids = memory.allocate<int>(capacity * slots);
        buffers.partial = memory.allocate<Winner>(capacity * refine_tiles);
    }

    Allocations memory;
    int capacity;
    ScoreRows buffers;
    CUtensorMap map{};
    std::array<CUtensorMap, tc_vocab_narrow_rows> narrow_maps{};
};

CUresult prepare_score_maps(const decode_proj::LaunchTarget& target, ScoreScratch& scratch) {
    if (scratch.capacity >= tc_vocab_rows) {
        const auto status = decode_proj::encode_map(target, scratch.map, scratch.buffers.quantized,
            CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, tc_vocab_rows, width, score_columns);
        if (status != CUDA_SUCCESS) return status;
    }
    for (int count = 1; count <= tc_vocab_narrow_rows; ++count) {
        const auto status = decode_proj::encode_map(target, scratch.narrow_maps[count - 1],
            scratch.buffers.quantized, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, count, width,
            narrow_columns);
        if (status != CUDA_SUCCESS) return status;
    }
    return CUDA_SUCCESS;
}

}

struct TcVocabScratch {
    decode_proj::LaunchTarget target;
    bool resident;
    mutable bool try_wide;
    std::unique_ptr<ScoreScratch> score;
    mutable ScoreRows wide;
    mutable CUtensorMap wide_map;
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

std::shared_ptr<const TcVocabScratch> prepare_tc_vocab_scratch(Allocations& memory, bool wide,
                                                               bool resident) {
    decode_proj::LaunchTarget target{};
    const auto status = decode_proj::prepare_launch(memory.stream(), target);
    if (!status || *status == cudaErrorNotSupported) return {};
    check_direct_status(*status, "Prepare native INT8 verification scratch");
    auto score = std::make_unique<ScoreScratch>(target.device,
                                                wide ? tc_vocab_rows : tc_vocab_narrow_rows);
    if (prepare_score_maps(target, *score) != CUDA_SUCCESS) return {};
    score->allocate_rows();
    return std::make_shared<TcVocabScratch>(
        TcVocabScratch{target, resident, wide, std::move(score), {}, {}});
}

namespace {

bool check_wide_map(CUresult status) {
    const auto direct = static_cast<cudaError_t>(status);
    const auto recorded = cudaGetLastError();
    if (CudaError::terminal(direct)) check(direct, "Encode wide INT8 verification tensor map");
    const bool declined = status == CUDA_ERROR_INVALID_VALUE || status == CUDA_ERROR_NOT_SUPPORTED;
    if (!declined || recorded != direct) {
        check(recorded, "Encode wide INT8 verification tensor map");
    }
    if (declined) return false;
    check(status == CUDA_SUCCESS ? cudaSuccess : cudaErrorUnknown,
          "Encode wide INT8 verification tensor map");
    return true;
}

bool prepare_wide_scratch(const TcVocabScratch& scratch) {
    if (scratch.wide.packed) return true;
    if (!scratch.try_wide) return false;
    scratch.try_wide = false;
    check(cudaPeekAtLastError(), "Check CUDA status before preparing M40 vocabulary scratch");
    if (scratch.resident && available_device_bytes() <
            decoder_shared_reserve + sizeof(WideScoreStorage)) {
        return false;
    }
    const auto configured = cudaFuncSetAttribute(vocabulary_scores<wide_columns>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, score_shared_bytes<wide_columns>);
    if (configured == cudaErrorNotSupported || configured == cudaErrorInvalidValue) {
        const auto pending = cudaGetLastError();
        if (pending != configured) check(pending, "Check CUDA status before M20 fallback");
        return false;
    }
    check_direct_status(configured, "Configure wide INT8 verification projection");

    WideScoreStorage* storage;
    try {
        storage = scratch.score->memory.allocate<WideScoreStorage>(1);
    } catch (const CudaError& error) {
        if (error.status != cudaErrorMemoryAllocation) throw;
        const auto pending = cudaGetLastError();
        if (pending != error.status) check(pending, "Check CUDA status before M20 fallback");
        return false;
    }
    // The session owns the slab even if its optional descriptor is unavailable.
    const auto status = decode_proj::encode_map(scratch.target, scratch.wide_map,
        storage->quantized, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, rows, width, wide_columns);
    if (!check_wide_map(status)) return false;
    scratch.wide = {storage->packed, storage->quantized, storage->maxima, storage->stats,
                    storage->scores, storage->lower, storage->threshold, storage->count,
                    storage->ids, storage->partial};
    return true;
}

void score(const TcVocabWeights& weights, const ScoreRows& scratch, const CUtensorMap& input_map,
           const TcVocabRow* batch, int count, cudaStream_t stream, bool packed) {
    Inputs inputs{};
    Outputs outputs{};
    for (int row = 0; row < count; ++row) {
        inputs.rows[row] = batch[row].input;
        outputs.rows[row] = batch[row].winner;
    }
    if (!packed) {
        pack_inputs<<<count, threads, 0, stream>>>(inputs, scratch.packed);
        check(cudaGetLastError(), "Pack shared INT8 verification inputs");
    }
    check_direct_status(quantize_draft_head_int8(scratch.packed, count, scratch.quantized,
                                                scratch.maxima, stream),
                        "Quantize shared INT8 verification inputs");
    shared_stats<<<count, threads, 0, stream>>>(
        scratch.packed, scratch.quantized, scratch.maxima, scratch.stats);
    check(cudaGetLastError(), "Prepare shared INT8 verification input bounds");
    constexpr int blocks = vocabulary / decode_proj::tile_rows;
    if (count <= tc_vocab_narrow_rows) {
        vocabulary_scores<narrow_columns>
            <<<blocks, decode_proj::threads, score_shared_bytes<narrow_columns>, stream>>>(
                weights.map, input_map, scratch.scores, count);
    } else if (count == tc_vocab_wide_rows) {
        vocabulary_scores<wide_columns>
            <<<blocks, decode_proj::threads, score_shared_bytes<wide_columns>, stream>>>(
                weights.map, input_map, scratch.scores, count);
    } else {
        vocabulary_scores<score_columns>
            <<<blocks, decode_proj::threads, score_shared_bytes<score_columns>, stream>>>(
                weights.map, input_map, scratch.scores, count);
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
}

}

bool try_tc_vocab(const TcVocabWeights& weights, const TcVocabScratch& scratch,
                  const TcVocabRow* batch, int count, cudaStream_t stream) {
    if (count < 1 || count > scratch.score->capacity ||
        (count > tc_vocab_narrow_rows && count != tc_vocab_rows)) {
        return false;
    }
    const auto& packet = *scratch.score;
    const auto& map = count <= tc_vocab_narrow_rows ? packet.narrow_maps[count - 1] : packet.map;
    score(weights, packet.buffers, map, batch, count, stream, false);
    return true;
}

bool try_tc_vocab_final(const TcVocabWeights& weights, const TcVocabScratch& scratch,
                        const TcVocabRow* batch, const TcVocabRmsRows& rms,
                        const Bf16* gain, cudaStream_t stream) {
    if ((!scratch.try_wide && !scratch.wide.packed) || !gain || !decode_proj::aligned16(gain)) {
        return false;
    }
    for (const auto& row : rms.rows) {
        if (!row.branch || !row.residual || !row.normalized ||
            !decode_proj::aligned16(row.branch) || !decode_proj::aligned16(row.residual)) {
            return false;
        }
    }
    if (!prepare_wide_scratch(scratch)) return false;
    const auto& buffers = scratch.wide;
    check_direct_status(pack_tc_vocab_rms(rms, gain, buffers.packed, stream),
                        "Normalize and pack shared INT8 verification inputs");
    score(weights, buffers, scratch.wide_map, batch, tc_vocab_wide_rows, stream, true);
    return true;
}

}
