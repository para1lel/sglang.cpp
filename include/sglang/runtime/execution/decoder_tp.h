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

/// Rank-local prefix cache layouts for the two-rank restore and store views.
const PrefixCacheLayout* tp2_prefix_layouts();

}

/// Tensor-parallel batch driver (world size 2). The owning DecoderBatch runs on
/// rank 0; the driver mirrors every rank-0 step on the peer rank: request
/// lifetime, KV growth, joint prefix restore and store, decode and prefill
/// forwards, speculative verification, and commit/rewind. Logits, sampling, and
/// speculative decisions stay on rank 0; rank 1 consumes committed counts.
/// Projections and attention fan out through the Model staged-projection trio
/// (forward_tp, prefill_grants_tp, verify_mtp_tp) and the LayerBatch
/// rank-local geometry carried by the WeightSharding helpers.
struct DecoderBatch::TpDriver {
    TpDriver(const Decoder::Model& owner, Decoder::TpRuntime& runtime);

    /// Allocate the rank-1 request for a freshly admitted lane.
    void create_peer_request(decoder::Lane& lane) const;
    /// Initialize the rank-1 request and wait for it before admission returns.
    void initialize_peer(decoder::Lane& lane, int request_capacity, int resident_limit) const;
    void grow_peer_kv(decoder::Lane& lane, int positions, int written) const;
    /// Best-effort peer sync for teardown paths that must not throw.
    void sync_peer_stream() const noexcept;
    /// Wait for in-flight peer work before the lane is released.
    void drain_peer(const decoder::Lane& lane) const;
    /// A failed joint restore may have dirtied either rank; both reset cold.
    void reset_peer_cold(decoder::Lane& lane) const;

    /// Register both ranks' restore views before either command is queued so a
    /// throw during rank-1 view construction cannot leave rank 0 waiting on a
    /// joint future that no rank-1 registration will settle. Rank 1 restores the
    /// same prefix without the draft tail.
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
    void verify(decoder::MtpRound* rounds, decoder::MtpRound* peer_rounds, int count,
                cudaStream_t stream, decoder::ConstraintWorkspace& constraints,
                std::byte* sampling_workspace, std::byte* logprob_workspace,
                const decoder::TcVocabScratch* vocabulary) const;
    /// Rank 1 rewinds its verification state by rank 0's accepted count and
    /// mirrors any committed checkpoint row so exported planes stay current.
    /// gdn_batch, when set, collects the rank-1 GDN prefix replays for one
    /// address-table dispatch by finish_peer_gdn_replays().
    void commit_peer_round(decoder::Lane& lane, const decoder::MtpRound& round,
                           const decoder::Workspace* const* saved,
                           TextGdnReplayBatch* gdn_batch = nullptr) const;
    /// Reset the rank-1 replay collector for a commit group (lazily allocating
    /// under the peer device scope); null on allocation failure.
    TextGdnReplayBatch* prepare_peer_gdn_replays() const;
    /// Dispatch the collected rank-1 replays on the peer stream.
    void finish_peer_gdn_replays() const;

    /// Export and store rank 1's copy of a checkpoint through its own layout.
    void store_peer_checkpoint(decoder::Lane& lane, int end, int generated, int count,
                               bool terminal, const std::shared_ptr<void>& reservation,
                               const std::shared_ptr<std::atomic_uint64_t>& acknowledged) const;

    const Decoder::Model& model;
    Decoder::TpRuntime& tp;
    mutable TextGdnReplayBatch peer_replays;
    mutable bool peer_replays_broken = false;
};

}
