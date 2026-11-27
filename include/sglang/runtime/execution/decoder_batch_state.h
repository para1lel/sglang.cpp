#pragma once
#include "sglang/text/gdn/gdn_replay_batch.h"

#include "sglang/runtime/execution/decoder_tp.h"
#include "sglang/decoder_internal.h"
#include "sglang/decoder_batch.h"
#include "sglang/text/decoder_batch_plan.h"
#include "sglang/text/label_logprobs.h"
#include "sglang/decoder_mtp.h"
#include "sglang/decoder_verify.h"
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
                             const WideMtpGroups& wide_groups);
    void prefill(const PrefillRows& rows, DecoderBatchStepStats& stats);
    void publish_checkpoint(Lane& lane, DecoderBatchStep& result, const Workspace* row,
                            bool& publishing);
    static void retire(std::optional<Lane>& lane, const TpDriver* tp_driver = nullptr);
    bool submit_checkpoint(Lane& lane);
    void finish_restore(Lane& lane, int restored);
    void score(int slot, Lane& lane, Workspace& work, bool logits_ready, bool decode);
    void speculate_group(const PrefillRows& verification_rows, int& next_slot, Steps& results,
                         DecoderBatchStepStats& stats, const WideMtpGroups& wide_groups,
                         CheckpointRows* checkpoint_rows);
    bool prepare_peer_repair(const MtpRound* rounds, int count);
    void repair_verification_group(const DraftRepairs* repairs, int count, cudaStream_t stream);
    int collect_rounds(const PrefillRows& verification_rows, int& next_slot,
                       const WideMtpGroups& wide_groups,
                       std::array<MtpRound, max_wide_mtp_lanes>& rounds,
                       std::array<MtpRound, max_wide_mtp_lanes>& peer_rounds,
                       std::array<int, max_wide_mtp_lanes>& slots);
    bool prepare_wide_mtp(int lanes);
    WideMtpGroups prepare_wide_groups(const PrefillRows& decode,
                                     const WideMtpCandidates& candidates, bool& contains_mtp);
    void collect_scores(const std::array<Request*, decoder_batch_max_lanes>& scoring,
                        Steps& results);
    std::size_t reserve(DecoderOptions& request, int prompt, std::size_t extra_bytes) const;
    void initialize(Lane& lane, int slot);
    void drain(const char* operation, std::optional<Lane>* lane = nullptr);
    void fail_step(bool recoverable);
    std::optional<Lane>& slot_at(int slot);

    void require_ready() const;

    void disable() noexcept;

    
    
    
    
    
    
    
    
    TextGdnReplayBatch* prepare_gdn_replays();

    const Decoder::Model& model;
    const DecoderBatchOptions options;
    KvResidencyPolicy residency;
    std::shared_ptr<BatchStorage> storage;
    Allocations execution;
    Decoder::TpRuntime* tp = nullptr;
    std::unique_ptr<TpDriver> tp_driver;
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
    TextGdnReplayBatch gdn_replays;
    bool gdn_replays_broken = false;
    bool failed = false;
    SerialOutput* output = nullptr;
    MtpRound::Cancel no_cancel;
    const MtpRound::Cancel& cancel_probe() const;
    bool cancelled() const;
};

}
