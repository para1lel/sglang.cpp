#pragma once

#include "sglang/decoder.h"
#include "sglang/decoder_batch.h"
#include "sglang/decoder_internal.h"
#include "sglang/decoder_mtp.h"
#include "sglang/decoder_prefill.h"
#include "sglang/runtime/cache/prefix_cache_layout.h"
#include "sglang/text/gdn/gdn_replay_batch.h"
#include "sglang/runtime/model/decoder_batch_checkpoint.h"
#include "sglang/runtime/model/decoder_batch_lane.h"

#include <atomic>
#include <cuda_runtime.h>
#include <memory>

namespace sglang {
namespace decoder {

const PrefixCacheLayout* tp2_prefix_layouts();

}

struct DecoderBatch::TpDriver {
    TpDriver(const Decoder::Model& owner, Decoder::TpRuntime& runtime);

    void create_peer_request(decoder::Lane& lane) const;
    void initialize_peer(decoder::Lane& lane, int request_capacity, int resident_limit) const;
    void grow_peer_kv(decoder::Lane& lane, int positions, int written) const;
    void sync_peer_stream() const noexcept;
    void drain_peer(const decoder::Lane& lane) const;
    void reset_peer_cold(decoder::Lane& lane) const;

    void register_prefix_restore(
        decoder::Lane& lane, const std::shared_ptr<const decoder::PrefixCacheRead>& read,
        cudaStream_t stream) const;
    void finish_peer_restore(decoder::Lane& lane) const;

    static decoder::ForwardLane mirror_lane(decoder::ForwardLane lane,
                                            decoder::Request* peer_request);
    void forward_decode(decoder::ForwardLane* work, decoder::ForwardLane* peer_work, int count,
                        cudaStream_t stream, bool regular_graph,
                        decoder::DecodeRun& run) const;
    void forward_single(decoder::ForwardLane* single, decoder::ForwardLane* peer_single,
                        int count, cudaStream_t stream) const;
    void prefill(const decoder::PrefillGrant* grants, const decoder::PrefillGrant* peer_grants,
                 int count, cudaStream_t stream, DecoderBatchStepStats& stats) const;

    void prepare_peer_verification(decoder::Lane& lane, int capacity) const;
    static decoder::MtpRound peer_round(const decoder::Lane& lane, int rows);
    bool prepare_peer_repair(const decoder::MtpRound* rounds, int count);
    void verify(decoder::MtpRound* rounds, decoder::MtpRound* peer_rounds, int count,
                cudaStream_t stream, decoder::ConstraintWorkspace& constraints,
                std::byte* sampling_workspace, std::byte* logprob_workspace,
                const decoder::TcVocabScratch* vocabulary, bool capture,
                bool allow_peer_repair) const;
    void commit_peer_round(decoder::Lane& lane, const decoder::MtpRound& round,
                           const decoder::Workspace* const* saved,
                           TextGdnReplayBatch* gdn_batch = nullptr) const;
    TextGdnReplayBatch* prepare_peer_gdn_replays() const;
    void finish_peer_gdn_replays() const;

    void store_peer_checkpoint(decoder::Lane& lane, int end, int generated, int count,
                               bool terminal, const std::shared_ptr<void>& reservation,
                               const std::shared_ptr<std::atomic_uint64_t>& acknowledged) const;

    const Decoder::Model& model;
    Decoder::TpRuntime& tp;
    mutable TextGdnReplayBatch peer_replays;
    mutable bool peer_replays_broken = false;
    std::unique_ptr<decoder::PeerDraftRepair> peer_repair;
    bool peer_repair_prepared = false;
};

}
