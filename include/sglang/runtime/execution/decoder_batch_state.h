#pragma once

#include "sglang/runtime/execution/decoder_tp.h"
#include "sglang/decoder_internal.h"
#include "sglang/decoder_batch.h"
#include "sglang/text/decoder_batch_plan.h"
#include "sglang/text/label_logprobs.h"
#include "sglang/decoder_mtp.h"
#include "sglang/decoder_verify.h"
#include "sglang/decoder_prefill.h"
#include "sglang/runtime/model/decoder_batch_lane.h"
#include "sglang/runtime/model/decoder_batch_checkpoint.h"
#include "sglang/text/mtp/draft_depth.h"
#include "sglang/text/mtp/tc_vocab.h"
#include "sglang/mxfp8_gqa.h"
#include "sglang/decoder_sampling.h"
#include "sglang/runtime/backend/cuda_resources.h"
#include <cstdlib>
#include <array>
#include <memory>
#include <optional>

namespace sglang {
namespace decoder {
namespace batch_detail {

struct BatchStorage {
    BatchStorage(int device, std::shared_ptr<const Allocations> memory, bool resident, int lanes);
    std::shared_ptr<const Allocations> weights;
    RequestBlockCache retired;
    std::optional<RequestBlockCache> peer_retired;
    std::array<std::optional<Lane>, decoder_batch_max_lanes> slots;
    BatchAdmissionOrder admission;
    DecodeGraphs decode_graphs;
    std::optional<VerificationGraph> verification_graph;
    std::shared_ptr<const TcVocabScratch> tc_vocab;
    std::shared_ptr<TextLogitCandidate> verification_winners;
};

std::size_t request_bytes(const BatchStorage& storage);

using Selection = std::array<bool, decoder_batch_max_lanes>;
using Steps = std::array<DecoderBatchStep, decoder_batch_max_lanes>;
using CheckpointRows = std::array<const Workspace*, decoder_batch_max_lanes>;
static_assert(decoder_batch_max_lanes <= prefill_chunk_rows);
static_assert(prefill_chunk_rows == mxfp8_page_size);

StepPlan plan_step(const BatchStorage& storage, const Selection& selected, int lanes,
                          int budget, bool chunked, WideMtpCandidates* wide,
                          bool tensor_parallel = false);

}  
}  

using namespace decoder;
using namespace decoder::batch_detail;

struct DecoderBatch::State {
    State(const Decoder::Model& owner, DecoderBatchOptions options,
          Decoder::TpRuntime* parallel = nullptr);
    ~State();

    struct PrefillCheckpointBatch;

    struct StepLanes {
        std::array<ForwardLane, decoder_batch_max_lanes> work;
        std::array<ForwardLane, decoder_batch_max_lanes> single_prefill;
        std::array<ForwardLane, decoder_batch_max_lanes> peer_work;
        std::array<ForwardLane, decoder_batch_max_lanes> peer_single;
        std::array<Request*, decoder_batch_max_lanes> scoring{};
        std::array<int, decoder_batch_max_lanes> packed_slots;
        PrefillRows speculative{};
        int count = 0;
        int single_count = 0;
        bool contains_mtp = false;
    };

    Steps advance(const Selection& selected, bool chunked, DecoderBatchStepStats* stats);
    void submit(const Selection& selected, const PrefillRows& decode, Steps& results,
                DecoderBatchStepStats& stats, bool& contains_mtp, CheckpointRows* checkpoint_rows,
                const WideMtpGroups& wide_groups);
    void collect_step_lanes(const Selection& selected, const PrefillRows& rows,
                            const PrefillRows& decode, Steps& results,
                            CheckpointRows* checkpoint_rows, StepLanes& lanes);
    void dispatch_step_lanes(StepLanes& lanes, int max_rows, Steps& results,
                             DecoderBatchStepStats& stats, CheckpointRows* checkpoint_rows,
                             const WideMtpGroups& wide_groups,
                             const PrefillCheckpointBatch& checkpoints);
    void prefill(const PrefillRows& rows, DecoderBatchStepStats& stats,
                 const PrefillCheckpointBatch& checkpoints);
    void publish_checkpoint(Lane& lane, DecoderBatchStep& result, const Workspace* row,
                            bool& publishing);
    void retire(std::optional<Lane>& lane) const;
    void finish_lane(std::optional<Lane>& lane, DecoderBatchStep& result, const Workspace* row,
                     bool& publishing);
    bool submit_checkpoint(Lane& lane);
    void store_checkpoint(Lane& lane, int end, const Workspace* row, bool terminal,
                          std::shared_ptr<void> reservation,
                          const PrefillCheckpoint* local = nullptr,
                          const PrefillCheckpoint* peer = nullptr);
    void plan_prefill_checkpoints(Lane& lane);
    void publish_prefill_checkpoint(Lane& lane, const PrefillCheckpoint& local,
                                    const PrefillCheckpoint* peer);
    ResidentPrefixRead prepare_resident_read(const Lane& lane) const;
    void apply_resident_read(Lane& lane, const ResidentPrefixRead& hit);
    bool stage_resident_restore(Lane& lane, ResidentPrefixRead& resident,
                                const std::shared_ptr<const PrefixCacheRead>& cpu) const;
    void finish_resident_restore(Lane& lane, bool transferred);
    void repair_resident_restore(Lane& lane, const PrefixDeviceLayout& layout,
                                 bool target_transferred, bool draft_transferred) const;
    void restore_resident_checkpoint(Lane& lane, const PrefixSnapshots& snapshots) const;
    void publish_resident_pages(Lane& lane, int target_rows, int draft_rows, bool include_tail,
                                const PrefixRestoreExtent* backed = nullptr) const;
    void publish_resident_terminal(Lane& lane, const Workspace* row) const;
    void publish_prefill_checkpoint_at(Lane& lane, const PrefillCheckpointBatch& batch, int slot,
                                       int end);
    struct PrefillCheckpointBatch {
        std::array<PrefillCheckpointList, decoder_batch_max_lanes> local;
        std::array<PrefillCheckpointList, decoder_batch_max_lanes> peer;
        std::array<int, decoder_batch_max_lanes> counts{};
        std::array<int, decoder_batch_max_lanes> retained{};
        const PrefixPrefillPool* blocked_pool = nullptr;
        std::size_t blocked_capacity = 0;
        std::size_t blocked_count = 0;
        bool blocked_resident = false;
        bool require(const PrefixPrefillPool& pool, std::size_t capacity, std::size_t count);
    };
    void publish_resident_prefill(const PrefillRows& rows, PrefillCheckpointBatch& entries);
    PrefillCheckpointBatch prepare_prefill_checkpoints(PrefillRows& rows);
    bool prepare_lane_checkpoints(const Lane& lane, int slot, PrefillCheckpointBatch& batch,
                                  int& rows);
    void acquire_prefill_checkpoint(PrefillCheckpointBatch& batch, int slot, int end,
                                    std::size_t capacity);
    void finish_restore(Lane& lane, int restored);
    void score(int slot, Lane& lane, Workspace& work, bool logits_ready, bool decode);
    void speculate_group(const PrefillRows& verification_rows, int& next_slot, Steps& results,
                         DecoderBatchStepStats& stats, const WideMtpGroups& wide_groups,
                         CheckpointRows* checkpoint_rows);
    void commit_verification(MtpRound* rounds, const int* slots, MtpRound* peer_rounds, int size,
                             Steps& results, DecoderBatchStepStats& stats,
                             CheckpointRows* checkpoint_rows, cudaStream_t stream,
                             const MtpRound::Cancel& cancel);
    void repair_verification_group(const DraftRepairs* repairs, int count, cudaStream_t stream);
    int collect_rounds(const PrefillRows& verification_rows, int& next_slot,
                       std::array<MtpRound, max_wide_mtp_lanes>& rounds,
                       std::array<int, max_wide_mtp_lanes>& slots,
                       std::array<MtpRound, max_wide_mtp_lanes>& peer_rounds,
                       const WideMtpGroups& wide_groups);
    bool prepare_wide_mtp(int lanes);
    WideMtpGroups prepare_wide_groups(const PrefillRows& decode,
                                     const WideMtpCandidates& candidates, bool& contains_mtp);
    void collect_scores(const std::array<Request*, decoder_batch_max_lanes>& scoring,
                        Steps& results);
    std::array<std::size_t, 2> reserve(int prompt, DecoderOptions& request,
                                       std::size_t extra_bytes) const;
    RequestBlockCache* request_cache(const DecoderOptions& request);
    RequestBlockCache* peer_request_cache(const Lane& lane);
    void validate_admission(DecoderOptions& options, const std::vector<int>& prompt,
                            bool multimodal) const;
    void prepare_admission(Lane& lane, int slot);
    [[noreturn]] void fail_admission(const CudaError& error, std::optional<Lane>& lane);
    void initialize(Lane& lane, int slot, ResidentPrefixRead& reuse);
    void initialize_request_workspace(Lane& lane, int slot);
    void initialize_prefix(Lane& lane, ResidentPrefixRead& reuse);
    void drain(const char* operation, std::optional<Lane>* lane = nullptr);
    void fail_step(bool recoverable);
    std::optional<Lane>& slot_at(int slot);

    void require_ready();

    void disable() noexcept;

    const Decoder::Model& model;
    const DecoderBatchOptions options;
    KvResidencyPolicy residency;
    std::optional<KvResidencyPolicy> peer_residency;
    std::unique_ptr<PrefixResidency> prefix_pool;
    std::shared_ptr<BatchStorage> storage;
    Allocations execution;
    Decoder::TpRuntime* tp = nullptr;
    std::unique_ptr<TpDriver> tp_driver;
    PrefixPrefillPool prefill_checkpoints;
    PrefixPrefillPool peer_prefill_checkpoints;
    std::uint8_t* wide_mtp_activation = nullptr;
    DraftDepthScratch draft_depth;
    DecoderPrefixCacheStats prefix_stats;
    std::shared_ptr<std::atomic_uint64_t> acknowledged_stores;
    int wide_mtp_capacity = 0;
    int wide_mtp_limit = 0;
    std::byte* sampling_workspace = nullptr;
    std::byte* logprob_workspace = nullptr;
    GeneratedLogprobReadback* logprob_readback = nullptr;
    ConstraintWorkspace constraint_workspace;
    AdjustedLogitsWorkspace adjusted_workspace;
    bool failed = false;
    SerialOutput* output = nullptr;
    MtpRound::Cancel no_cancel;
    const MtpRound::Cancel& cancel_probe() const;
    bool cancelled() const;
};

}
