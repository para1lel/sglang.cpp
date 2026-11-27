#include "sglang/decoder_internal.h"
#include "sglang/text/decode_proj.h"
#include "sglang/text/mtp/tc_vocab_launch.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <string_view>

namespace sglang::decoder {
namespace {

using namespace tc_vocab;

static_assert(vocab_size == vocabulary && hidden_size == width);

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

}

struct TcVocabWeights {
    CUtensorMap map{};
    CUtensorMap short_bf16_map{};
    bool short_ready = false;
    const Bf16* original{};
    Stats* stats{};
    bool trace = false;
};

namespace {

struct ShortScoreScratch {
    CUtensorMap quantized_map{};
    CUtensorMap packed_map{};
    float* lower{};
    int* seeds{};
    ShortCandidates* candidates{};
    Winner* partial{};
};

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

template <int Rows> struct WideScoreStorage {
    std::uint8_t quantized[Rows * width];
    Bf16 packed[Rows * width];
    float maxima[Rows];
    Stats stats[Rows];
    int scores[Rows * vocabulary];
    float lower[rows * tiles];
    float threshold[rows];
    int count[rows];
    int ids[rows * slots];
    Winner partial[rows * refine_tiles];
};

struct WideScores {
    bool prepare;
    ScoreRows rows{};
    CUtensorMap map{};
};

struct ScoreScratch {
    ScoreScratch(int device, int capacity)
        : memory(device), capacity(capacity) {
        buffers.quantized = memory.allocate<std::uint8_t>(capacity * width);
    }

    void allocate_rows() {
        buffers.packed = memory.allocate<Bf16>(capacity * width);
        buffers.maxima = memory.allocate<float>(capacity);
        buffers.stats = static_cast<Stats*>(memory.allocate_elements(capacity, sizeof(Stats)));
        buffers.scores = memory.allocate<int>(capacity * vocabulary);
        buffers.lower = memory.allocate<float>(capacity * tiles);
        buffers.threshold = memory.allocate<float>(capacity);
        buffers.count = memory.allocate<int>(capacity);
        buffers.ids = memory.allocate<int>(capacity * slots);
        buffers.partial = static_cast<Winner*>(
            memory.allocate_elements(capacity * refine_tiles, sizeof(Winner)));
    }

    Allocations memory;
    int capacity;
    ScoreRows buffers;
    CUtensorMap map{};
    std::array<CUtensorMap, tc_vocab_narrow_rows> narrow_maps{};
    ShortScoreScratch short_score;
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

void prepare_short_weights(const decode_proj::LaunchTarget& target, TcVocabWeights& weights) {
    if (!decode_proj::aligned16(weights.original)) return;
    const auto status = decode_proj::prepare_bf16_single_split(
        target, short_tiles, width / decode_proj::bf16_chunk_elements);
    if (!status) return;
    check_direct_status(*status, "Prepare packed BF16 verification replay");
    weights.short_ready = decode_proj::encode_map(target, weights.short_bf16_map, weights.original,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, vocabulary, width, decode_proj::tile_rows) ==
        CUDA_SUCCESS;
}

void prepare_short_scratch(const decode_proj::LaunchTarget& target, ScoreScratch& score) {
    auto& scratch = score.short_score;
    if (decode_proj::encode_map(target, scratch.quantized_map, score.buffers.quantized,
            CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, short_rows, width, short_columns) != CUDA_SUCCESS ||
        decode_proj::encode_map(target, scratch.packed_map, score.buffers.packed,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, short_rows, width, short_columns) !=
            CUDA_SUCCESS) {
        return;
    }
    check_direct_status(configure_short_screen(),
                        "Configure ten-row INT8 verification projection");
    check_direct_status(configure_short_seeds(),
                        "Configure ten-row packed BF16 verification seeds");
    check_direct_status(configure_short_replay(),
                        "Configure ten-row packed BF16 verification replay");
    scratch.lower = score.memory.allocate<float>(short_rows * short_tiles);
    scratch.seeds = score.memory.allocate<int>(short_rows);
    scratch.candidates = static_cast<ShortCandidates*>(
        score.memory.allocate_elements(1, sizeof(ShortCandidates)));
    scratch.partial = static_cast<Winner*>(
        score.memory.allocate_elements(short_rows * short_tiles, sizeof(Winner)));
}

}

struct TcVocabScratch {
    decode_proj::LaunchTarget target;
    bool resident;
    std::unique_ptr<ScoreScratch> score;
    mutable WideScores wide;
    mutable WideScores paired;
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
    check_direct_status(configure_packet_scores(tc_vocab_rows),
                        "Configure native INT8 verification projection");
    check_direct_status(configure_packet_scores(tc_vocab_narrow_rows),
                        "Configure narrow INT8 verification projection");
    weights->original = original;
    prepare_short_weights(target, *weights);
    weights->stats = static_cast<Stats*>(memory.allocate_elements(vocabulary, sizeof(Stats)));
    weights->trace = setting_enabled("SGLANG_VERIFY_TC_VOCAB_TRACE");
    check(launch_shared_stats(vocabulary, original, quantized, maxima, weights->stats,
                              memory.stream()), "Prepare shared INT8 verification weight bounds");
    std::fputs("sglang tc_vocab: enabled B200 native INT8 Tensor Core projection\n", stderr);
    return weights;
}

std::shared_ptr<const TcVocabScratch> prepare_tc_vocab_scratch(Allocations& memory, bool wide,
                                                               bool resident, bool short_packet) {
    decode_proj::LaunchTarget target{};
    const auto status = decode_proj::prepare_launch(memory.stream(), target);
    if (!status || *status == cudaErrorNotSupported) return {};
    check_direct_status(*status, "Prepare native INT8 verification scratch");
    const int capacity = wide ? tc_vocab_rows : (short_packet ? short_rows : tc_vocab_narrow_rows);
    auto score = std::make_unique<ScoreScratch>(target.device, capacity);
    if (prepare_score_maps(target, *score) != CUDA_SUCCESS) return {};
    score->allocate_rows();
    if (short_packet) prepare_short_scratch(target, *score);
    return std::make_shared<TcVocabScratch>(
        TcVocabScratch{target, resident, std::move(score), WideScores{wide}, WideScores{wide}});
}

namespace {

bool check_wide_map(CUresult status) {
    const auto direct = CudaError::from_driver(status);
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

template <int Rows, int Columns>
bool prepare_wide_scratch(const TcVocabScratch& scratch, WideScores& scores,
                          const ScoreRows& reusable = ScoreRows{}) {
    if (scores.rows.packed)
        return true;
    if (!scores.prepare)
        return false;
    scores.prepare = false;
    check(cudaPeekAtLastError(), "Check CUDA status before preparing wide vocabulary scratch");
    const DeviceScope device(scratch.target.device);
    using Storage = WideScoreStorage<Rows>;
    if (!reusable.packed && scratch.resident &&
        available_device_bytes() < decoder_shared_reserve + sizeof(Storage)) {
        return false;
    }
    const auto configured =
        Rows == paired_columns ? configure_paired_scores() : configure_packet_scores(Rows);
    if (configured == cudaErrorNotSupported || configured == cudaErrorInvalidValue) {
        const auto pending = cudaGetLastError();
        if (pending != configured)
            check(pending, "Check CUDA status before vocabulary fallback");
        return false;
    }
    check_direct_status(configured, "Configure wide INT8 verification projection");

    auto buffers = reusable;
    Storage* storage = nullptr;
    if (!buffers.packed) {
        try {
            storage =
                static_cast<Storage*>(scratch.score->memory.allocate_elements(1, sizeof(Storage)));
        } catch (const CudaError& error) {
            if (error.status != cudaErrorMemoryAllocation)
                throw;
            const auto pending = cudaGetLastError();
            if (pending != error.status) {
                check(pending, "Check CUDA status before vocabulary fallback");
            }
            return false;
        }
        buffers = {
            storage->packed, storage->quantized, storage->maxima, storage->stats, storage->scores,
            storage->lower,  storage->threshold, storage->count,  storage->ids,   storage->partial,
        };
    }
    const auto status =
        decode_proj::encode_map(scratch.target, scores.map, buffers.quantized,
                                CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, Rows, width, Columns);
    if (!check_wide_map(status)) {
        if (storage != nullptr)
            scratch.score->memory.free_block(storage, sizeof(Storage));
        return false;
    }
    scores.rows = buffers;
    return true;
}

void quantize(const ScoreRows& scratch, int count, cudaStream_t stream) {
    check_direct_status(
        quantize_draft_head_int8(scratch.packed, count, scratch.quantized, scratch.maxima, stream),
        "Quantize shared INT8 verification inputs");
    check(launch_shared_stats(count, scratch.packed, scratch.quantized, scratch.maxima,
                              scratch.stats, stream),
          "Prepare shared INT8 verification input bounds");
}

void certify(const TcVocabWeights& weights, const ScoreRows& scratch, const TcVocabRow* batch,
             int count, cudaStream_t stream) {
    Inputs inputs{};
    Outputs outputs{};
    for (int row = 0; row < count; ++row) {
        inputs.rows[row] = batch[row].input;
        outputs.rows[row] = batch[row].winner;
    }
    check(launch_lower_bounds(count, scratch.scores, weights.stats, scratch.stats, scratch.lower,
                              stream), "Bound shared INT8 verification scores");
    check(launch_thresholds(count, scratch.lower, scratch.threshold, stream),
          "Reduce shared INT8 verification bounds");
    check_direct_status(cudaMemsetAsync(scratch.count, 0, count * sizeof(int), stream),
                        "Clear verification vocabulary candidate counts");
    check(launch_candidates(count, scratch.scores, scratch.threshold, scratch.count, scratch.ids,
                            stream), "Collect shared INT8 verification candidates");
    check(launch_refine(count, weights.original, inputs, scratch.count, scratch.ids,
                        scratch.partial, stream), "Refine shared INT8 verification candidates");
    check(launch_finish(count, scratch.partial, outputs, stream),
          "Certify greedy verification vocabulary winners");
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

void score(const TcVocabWeights& weights, const ScoreRows& scratch, const CUtensorMap& input_map,
           const TcVocabRow* batch, int count, cudaStream_t stream, bool packed) {
    if (!packed) {
        Inputs inputs{};
        for (int row = 0; row < count; ++row)
            inputs.rows[row] = batch[row].input;
        check(launch_pack_inputs(count, inputs, scratch.packed, stream),
              "Pack shared INT8 verification inputs");
    }
    quantize(scratch, count, stream);
    check(launch_vocabulary_scores(weights.map, input_map, scratch.scores, count, stream),
          "Run native INT8 verification projection");
    certify(weights, scratch, batch, count, stream);
}

bool can_pack(const TcVocabRmsRows& rms, const Bf16* gain) {
    return gain != nullptr && decode_proj::aligned16(gain) &&
           std::all_of(std::begin(rms.rows), std::end(rms.rows),
                       [](const TcVocabRmsRows::Row& row) {
                           return row.branch != nullptr && row.residual != nullptr &&
                                  row.normalized != nullptr && decode_proj::aligned16(row.branch) &&
                                  decode_proj::aligned16(row.residual);
                       });
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

bool try_tc_vocab_short(const TcVocabWeights& weights, const TcVocabScratch& scratch,
                        const TcVocabRow* batch, cudaStream_t stream) {
    const auto& packet = *scratch.score;
    const auto& small = packet.short_score;
    const auto& buffers = packet.buffers;
    if (!weights.short_ready || !small.partial) return false;

    Inputs inputs{};
    Outputs outputs{};
    for (int row = 0; row < short_rows; ++row) {
        inputs.rows[row] = batch[row].input;
        outputs.rows[row] = batch[row].winner;
    }
    check(launch_pack_inputs(short_rows, inputs, buffers.packed, stream),
          "Pack ten-row verification inputs");
    check_direct_status(quantize_draft_head_int8(buffers.packed, short_rows, buffers.quantized,
                                                buffers.maxima, stream),
                        "Quantize ten-row verification inputs");
    check(launch_shared_stats(short_rows, buffers.packed, buffers.quantized, buffers.maxima,
                              buffers.stats, stream), "Prepare ten-row verification input bounds");
    check(launch_screen_short(weights.map, small.quantized_map, weights.stats, buffers.stats,
                               buffers.scores, small.lower, stream),
          "Screen ten-row verification scores");
    check(launch_select_short(small.lower, buffers.threshold, small.seeds, stream),
          "Select ten-row verification seed tiles");
    check(launch_improve_short(weights.short_bf16_map, small.packed_map, small.seeds,
                               buffers.threshold, stream), "Score ten-row verification seed tiles");
    check_direct_status(cudaMemsetAsync(small.candidates, 0, offsetof(ShortCandidates, tiles),
                                        stream), "Clear ten-row verification candidate counts");
    check(launch_collect_short(buffers.scores, buffers.threshold, small.candidates, stream),
          "Collect ten-row verification candidates");
    check(launch_replay_short(weights.short_bf16_map, small.packed_map, buffers.scores,
                              buffers.threshold, small.candidates, small.partial, stream),
          "Replay ten-row verification BF16 tiles");
    check(launch_finish_short(small.candidates, small.partial, outputs, stream),
          "Select ten-row verification BF16 winners");
    return true;
}

bool try_tc_vocab_final(const TcVocabWeights& weights, const TcVocabScratch& scratch,
                        const TcVocabRow* batch, const TcVocabRmsRows& rms,
                        const Bf16* gain, cudaStream_t stream) {
    if (!can_pack(rms, gain) ||
        !prepare_wide_scratch<rows, wide_columns>(scratch, scratch.wide, scratch.paired.rows)) {
        return false;
    }
    const auto& buffers = scratch.wide.rows;
    check_direct_status(pack_tc_vocab_rms(rms, gain, buffers.packed, stream),
                        "Normalize and pack shared INT8 verification inputs");
    score(weights, buffers, scratch.wide.map, batch, rows, stream, true);
    return true;
}

bool try_tc_vocab_final_pair(const TcVocabWeights& weights, const TcVocabScratch& scratch,
                             const TcVocabRow* batch, const TcVocabRmsRows (&rms)[2],
                             const Bf16* gain, cudaStream_t stream) {
    if (!can_pack(rms[0], gain) || !can_pack(rms[1], gain))
        return false;
    auto& paired = scratch.paired;
    if (paired.rows.packed == nullptr) {
        if (!paired.prepare)
            return false;
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        check(cudaStreamIsCapturing(stream, &capture), "Check vocabulary stream capture");
        if (capture != cudaStreamCaptureStatusNone ||
            !prepare_wide_scratch<paired_columns, paired_columns>(scratch, paired)) {
            return false;
        }
    }

    const auto& buffers = paired.rows;
    for (int half = 0; half < 2; ++half) {
        check_direct_status(
            pack_tc_vocab_rms(rms[half], gain,
                              buffers.packed + (static_cast<std::ptrdiff_t>(half) * rows * width),
                              stream),
            "Normalize and pack shared INT8 verification inputs");
    }
    quantize(buffers, paired_columns, stream);
    check(launch_paired_scores(weights.map, paired.map, buffers.scores, stream),
          "Run paired INT8 verification projection");
    certify(weights, buffers, batch, rows, stream);
    auto second = buffers;
    second.stats += rows;
    second.scores += static_cast<std::ptrdiff_t>(rows) * vocabulary;
    certify(weights, second, batch + rows, rows, stream);
    return true;
}
}
