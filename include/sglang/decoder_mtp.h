#pragma once

#include "sglang/decoder_internal.h"

namespace sglang::decoder {

struct DraftOperatorCapabilities;
struct DraftDepthScratch;
struct PrefillProjections;
struct PrefixCacheRead;
struct PrefixCacheLayout;
class PrefixPrefillSnapshot;
struct PrefixDeviceTransfer;
struct PrefixRestoreExtent;

inline constexpr int draft_projection_rows = 16;

struct MtpPrefillBuffers {
    Bf16* normalized_hidden;
    Bf16* embedding;
    Bf16* concatenated;
    Bf16* layer_input;
    Bf16* keys;
    Bf16* values;
    Bf16* rope_keys;
};

struct MtpDraftInput {
    MtpDraftRequest* request;
    int position;
    int token;
    const Bf16* hidden;
    const MropePosition* mrope = nullptr;
    int rotary_position = -1;
};

using DraftRepairs = std::array<MtpDraftInput, decoder_batch_max_mtp_steps>;

class MtpDraftWeights {
public:
  MtpDraftWeights(const std::filesystem::path& checkpoint, int device,
                  std::shared_ptr<const Allocations> target_weights, const Bf16* embedding,
                  Bf16Weight lm_head, DraftOperatorCapabilities capabilities,
                  WeightSharding shard = WeightSharding());
  ~MtpDraftWeights();

  std::size_t device_allocation_bytes() const;
  int vocab_begin() const;
  int vocab_rows() const;
  int kv_heads() const;
  std::shared_ptr<const TcVocabWeights> shared_vocab() const;

private:
    struct State;
    std::unique_ptr<const State> state_;
    friend class MtpDraftRequest;
};

class MtpDraftRequest {
public:
  struct PrefixStorage {
      const PrefixSlab* slab = nullptr;
      PrefixPages pages;
      const ResidentPrefixRead* reuse = nullptr;
      int slab_rank = 0;
  };

    MtpDraftRequest(std::shared_ptr<const MtpDraftWeights> weights, int capacity,
                    RequestBlockCache* cache = nullptr);
    MtpDraftRequest(std::shared_ptr<const MtpDraftWeights> weights, int capacity,
                    RequestBlockCache* cache, PrefixStorage storage);
    ~MtpDraftRequest();

    const Bf16* enqueue(int position, int shifted_token, const Bf16* hidden, bool score,
                        cudaStream_t stream, const ConstraintWorkspace* scratch = nullptr,
                        const ConstraintCandidates* allowed = nullptr,
                        const MropePosition* mrope = nullptr, int rotary_position = -1,
                        const Bf16* feature = nullptr);
    static void enqueue_wave(MtpDraftInput* rows, int count, bool score, cudaStream_t stream);
    static bool prepare_depth(MtpDraftRequest& lane, DraftDepthScratch& scratch,
                              std::uint8_t* activation, bool tensor_parallel, cudaStream_t stream);
    static void enqueue_depth(MtpDraftInput* rows, int count, const DraftDepthScratch& scratch,
                              cudaStream_t stream);
    static void repair_wave(const DraftRepairs* repairs, int count, Bf16* packed,
                            cudaStream_t stream, bool exact = false);
    static void enqueue_tp(MtpDraftInput* const (&rows)[2], int count, bool score,
                           const cudaStream_t (&streams)[2], const TpPartials (&partials)[2],
                           std::uint8_t* const (&packed)[2], TensorParallel& group,
                           const ConstraintWorkspace* scratch = nullptr,
                           const ConstraintCandidates* const* allowed = nullptr);
    static int read_tp_token(MtpDraftRequest& primary, MtpDraftRequest& peer);
    static bool retry_invalid_tp(MtpDraftRequest& primary, MtpDraftRequest& peer,
                                 const cudaStream_t (&streams)[2]);
    static void seed_tp(MtpDraftInput primary, MtpDraftInput peer, TensorParallel& group,
                        cudaStream_t stream, cudaStream_t peer_stream);

    void seed_prefill(int position, const int* shifted_tokens, const Bf16* hidden,
                      const Bf16* residual, const Bf16* final_norm,
                      const PrefillProjections& projections, MtpPrefillBuffers buffers,
                      const MediaPrefill* media = nullptr);
    static void seed_prefill_tp(MtpDraftRequest& primary, MtpDraftRequest& peer, int position,
                                const int* shifted_tokens, const Bf16* (&hidden)[2],
                                const Bf16* (&residual)[2], const Bf16* (&final_norm)[2],
                                const PrefillProjections (&projections)[2],
                                MtpPrefillBuffers (&buffers)[2], TensorParallel& group);

    int read_token() const;
    bool retry_invalid_int8(cudaStream_t stream);

    int valid_prefix() const;
    int kv_elements_per_row() const;
    void rewind(int valid_tokens);
    std::size_t device_allocation_bytes() const;
    void recycle_after_completion();
    const PrefixPages& resident_prefix_pages() const;
    const std::vector<std::int32_t>& prefix_physical_pages() const;
    void commit_restored_prefix(int rows);

  private:
    const Bf16* enqueue_impl(int position, int shifted_token, const Bf16* hidden, bool score,
                            cudaStream_t stream, const ConstraintWorkspace* scratch,
                            const ConstraintCandidates* allowed, bool defer_score,
                            const MropePosition* mrope, int rotary_position, const Bf16* feature);
    static void enqueue_wave_impl(MtpDraftInput* rows, int count, bool score,
                                  cudaStream_t stream, bool defer_score, Bf16* packed = nullptr,
                                  bool exact = false);
    friend PrefixDeviceTransfer
    restore_prefix_view(Request& request, const PrefixCacheRead& read, cudaStream_t initialized,
                        MtpDraftRequest* draft, const PrefixCacheLayout* layout,
                        const std::shared_ptr<PrefixPrefillSnapshot>& snapshot,
                        const PrefixRestoreExtent* extent);
    friend PrefixDeviceTransfer export_prefix_view(const Request&, int, int, int, cudaStream_t,
                                                   const MtpDraftRequest*, const Workspace*, bool,
                                                   const PrefixCacheLayout*,
                                                   const std::shared_ptr<PrefixPrefillSnapshot>&);
    friend int commit_restored_prefix(Request& request, const PrefixDeviceTransfer& view,
                                      MtpDraftRequest* draft, cudaStream_t stream);
    Mxfp8KvCache prefix_cache_view(int rows, bool storing) const;
    const Allocations& prefix_cache_memory() const;

    struct State;
    static void score_tp(State* const* const (&states)[2], int count,
                         const cudaStream_t (&streams)[2], TensorParallel& group,
                         const ConstraintWorkspace* scratch,
                         const ConstraintCandidates* const* allowed);
    std::unique_ptr<State> state_;
};

}
