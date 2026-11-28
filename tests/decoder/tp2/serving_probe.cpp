#include "sglang/checkpoint.h"
#include "sglang/decoder_internal.h"
#include "sglang/decoder_mtp.h"
#include "sglang/runtime/backend/tensor_parallel.h"
#include "sglang/runtime/backend/tp_ar_counters.h"

#include <cudaTypedefs.h>
#include <nccl.h>
#include <unistd.h>

#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
using sglang::TextLogitCandidate;
using namespace sglang::decoder;
constexpr std::size_t draft_bytes = sizeof(Bf16) * hidden_size * 2 * hidden_size / 2;
constexpr std::size_t head_bytes = sizeof(Bf16) * vocab_size * hidden_size / 2;

struct Stripe {
    std::size_t bytes = 0;
    std::array<std::array<std::byte, 32>, 3> samples{};
    const void* device = nullptr;

    std::array<std::size_t, 3> offsets() const {
        return {0, bytes / 2, bytes - 32};
    }
};

struct ServingProbe {
    const void* partials[2][2]{};
    std::size_t calls = 0;
    std::size_t last_count = 0;
    std::size_t target_rows = 0;
    int world_size = 0;
    const void* traversal_buffers[2][2]{};
    std::size_t traversal_count = 0;
    std::size_t tiled_passes = 0;
    std::size_t decode_passes = 0;
    Stripe draft_fc[2]{{draft_bytes}, {draft_bytes}};
    Stripe draft_head{head_bytes};
    bool track_upload = false;
    bool draft_wave = false;
    int observed_draft_rows = 0;
    unsigned int draft_gather_ranks = 0;
    std::size_t draft_phase_rows[2][2]{};
    bool track_decode = false;
    const TensorParallel* serving_group = nullptr;
    bool vendor_transport = false;
    const void* draft_sources[2]{};
    void* draft_destinations[2]{};
    TpArCounters target_origin;
    std::array<std::uint64_t, 2> draft_collectives{};
    std::array<std::uint64_t, 2> draft_payload{};
    bool reading_draft = false;
    int collected_scores = 0;
    std::array<TextLogitCandidate, 2> rank_candidates{};
    std::array<std::size_t, 2> score_wins{};
    std::size_t scored_proposals = 0;
};

ServingProbe& serving_probe() {
    static ServingProbe probe;
    return probe;
}

void need(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void pointer(int rank, const char* name, const void* data, std::size_t bytes) {
    cudaPointerAttributes attributes{};
    need(data != nullptr && bytes != 0 &&
             cudaPointerGetAttributes(&attributes, data) == cudaSuccess &&
             attributes.type == cudaMemoryTypeDevice && attributes.device == rank,
         "Weight or partial buffer belongs to the wrong device");
    void* entry = nullptr;
    need(cudaGetDriverEntryPointByVersion("cuMemGetAddressRange", &entry, 3020, cudaEnableDefault,
                                          nullptr) == cudaSuccess &&
             entry != nullptr,
         "No CUDA allocation-range query");
    const auto locate = reinterpret_cast<PFN_cuMemGetAddressRange_v3020>(entry);
    const auto address = reinterpret_cast<CUdeviceptr>(data);
    CUdeviceptr base = 0;
    std::size_t backing = 0;
    need(locate(&base, &backing, address) == CUDA_SUCCESS && address >= base &&
             address - base <= backing && bytes <= backing - (address - base),
         "Tensor span exceeds its CUDA allocation");
    std::cout << "rank=" << rank << " allocation=" << name << " address=" << data
              << " bytes=" << bytes << " base=" << reinterpret_cast<const void*>(base)
              << " backing_bytes=" << backing << '\n';
}

void remember(Stripe& stripe, const std::byte* data) {
    const auto offsets = stripe.offsets();
    for (std::size_t index = 0; index < offsets.size(); ++index) {
        std::memcpy(stripe.samples[index].data(), data + offsets[index], 32);
    }
}

bool matches(const Stripe& stripe, const void* data) {
    const auto offsets = stripe.offsets();
    for (std::size_t index = 0; index < offsets.size(); ++index) {
        if (std::memcmp(static_cast<const std::byte*>(data) + offsets[index],
                        stripe.samples[index].data(), 32) != 0)
            return false;
    }
    return true;
}

void check_stripe(int rank, const char* name, const Stripe& stripe) {
    pointer(rank, name, stripe.device, stripe.bytes);
    const auto offsets = stripe.offsets();
    const auto* base = static_cast<const std::byte*>(stripe.device);
    for (std::size_t index = 0; index < offsets.size(); ++index) {
        std::array<std::byte, 32> sample{};
        need(cudaMemcpy(sample.data(), base + offsets[index], sample.size(),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
                 sample == stripe.samples[index],
             "Rank-local draft checkpoint stripe mismatch");
    }
}
}

void watch_draft_upload(const std::filesystem::path& directory) {
    auto& probe = serving_probe();
    sglang::Checkpoint checkpoint(directory);
    const auto& fc = checkpoint.tensor("mtp.fc.weight", sglang::TensorDtype::BF16,
                                       {hidden_size, 2ULL * hidden_size});
    const auto& head =
        checkpoint.tensor("lm_head.weight", sglang::TensorDtype::BF16, {vocab_size, hidden_size});
    need(fc.size_bytes == 2 * draft_bytes && head.size_bytes == 2 * head_bytes,
         "Invalid MTP checkpoint extent");
    for (int rank = 0; rank < 2; ++rank) {
        remember(probe.draft_fc[rank], fc.data + (rank * draft_bytes));
    }
    remember(probe.draft_head, head.data + head_bytes);
    probe.track_upload = true;
}

extern "C" cudaError_t real_copy_async(void* dest, const void* source, std::size_t bytes,
                                       cudaMemcpyKind kind,
                                       cudaStream_t stream) asm("__real_cudaMemcpyAsync");
extern "C" cudaError_t wrap_copy_async(void* dest, const void* source, std::size_t bytes,
                                       cudaMemcpyKind kind,
                                       cudaStream_t stream) asm("__wrap_cudaMemcpyAsync");
extern "C" cudaError_t wrap_copy_async(void* dest, const void* source, std::size_t bytes,
                                       cudaMemcpyKind kind, cudaStream_t stream) {
    auto& probe = serving_probe();
    if (probe.track_upload && kind == cudaMemcpyHostToDevice) {
        if (bytes == 2 * draft_bytes && matches(probe.draft_fc[0], source)) {
            throw std::runtime_error("Replicated MTP FC upload");
        }
        if (bytes == draft_bytes || bytes == head_bytes || bytes == 2 * head_bytes) {
            int rank = -1;
            need(cudaGetDevice(&rank) == cudaSuccess && rank >= 0 && rank < 2,
                 "Misplaced draft checkpoint upload");
            auto& stripe = bytes == draft_bytes ? probe.draft_fc[rank] : probe.draft_head;
            if (bytes == 2 * head_bytes && rank == 1 &&
                matches(probe.draft_head, static_cast<const std::byte*>(source) + head_bytes)) {
                throw std::runtime_error("Peer uploaded the full draft vocabulary");
            }
            if (bytes == stripe.bytes && matches(stripe, source)) {
                need(stripe.device == nullptr && (bytes != head_bytes || rank == 1),
                     "Duplicate or misplaced sharded draft upload");
                stripe.device = dest;
            }
        }
    }
    return real_copy_async(dest, source, bytes, kind, stream);
}

namespace sglang {
class MediaDecoderCheck {
  public:
    static void inspect(const Decoder& decoder, int world) {
        auto& probe = serving_probe();
        probe.world_size = world;
        probe.track_upload = false;
        const auto* ranks = decoder.tensor_parallel_info();
        need((ranks != nullptr) == (world == 2), "Unexpected public TP receipt");
        if (world == 1)
            return;
        need(ranks != nullptr && ranks->world_size == 2 && decoder.tp_ != nullptr &&
                 decoder.tp_->group != nullptr,
             "Missing TP group");
        need(std::memcmp(ranks->ranks[0].uuid, ranks->ranks[1].uuid, 16) != 0,
             "Duplicate rank UUID");
        probe.serving_group = decoder.tp_->group.get();
        probe.serving_group->snapshot_transport(probe.target_origin);
        probe.vendor_transport = probe.target_origin.mnnvl_active;
        for (int rank = 0; rank < world; ++rank) {
            decoder::DeviceScope selected(rank);
            const auto& model = rank == 0 ? *decoder.model_ : *decoder.tp_->peer;
            const auto& actual = ranks->ranks[rank];
            cudaDeviceProp device{};
            need(cudaGetDeviceProperties(&device, rank) == cudaSuccess && model.device == rank &&
                     actual.cuda_device == rank && actual.local_rank == rank &&
                     actual.pid == getpid() &&
                     std::memcmp(device.uuid.bytes, actual.uuid, 16) == 0 &&
                     actual.nccl_established,
                 "Invalid rank identity");
            const auto draft =
                model.mtp_weights != nullptr ? model.mtp_weights->device_allocation_bytes() : 0;
            const bool has_head = model.lm_head.values != nullptr;
            need(has_head == (rank == 0) && actual.lm_head == has_head && draft > 0 && actual.mtp,
                 "Head/draft ownership");
            need(actual.weight_bytes == model.memory->bytes() + draft &&
                     actual.draft_weight_bytes == draft && actual.draft_kv_heads == 2 &&
                     actual.draft_vocab_begin == rank * (vocab_size / 2) &&
                     actual.draft_vocab_rows == vocab_size / 2,
                 "Public draft shard receipt mismatch");
            need(model.mtp_weights->vocab_begin() == rank * (vocab_size / 2) &&
                     model.mtp_weights->vocab_rows() == vocab_size / 2,
                 "Draft vocabulary ownership");
            check_stripe(rank, "mtp_fc", probe.draft_fc[rank]);
            if (rank == 1)
                check_stripe(rank, "mtp_vocabulary", probe.draft_head);
            std::cout << "rank=" << rank << " device=" << actual.cuda_device
                      << " pid=" << actual.pid << " weight_bytes=" << actual.weight_bytes
                      << " draft_bytes=" << draft
                      << " draft_vocab_begin=" << actual.draft_vocab_begin
                      << " draft_vocab_rows=" << actual.draft_vocab_rows
                      << " draft_kv_heads=" << actual.draft_kv_heads << " uuid=";
            std::cout << std::hex << std::setfill('0');
            for (const unsigned char byte : actual.uuid)
                std::cout << std::setw(2) << static_cast<unsigned>(byte);
            std::cout << std::dec << std::setfill(' ') << '\n';
            if (has_head) {
                need(model.lm_head.rows == vocab_size && model.lm_head.columns == hidden_size,
                     "Unexpected rank-0 vocabulary head dimensions");
                pointer(rank, "lm_head", model.lm_head.values,
                        sizeof(Bf16) * vocab_size * hidden_size);
            }
            pointer(rank, "embedding", model.embedding, sizeof(Bf16) * vocab_size * hidden_size);
            const auto& gate = model.layers[0].mlp_gate;
            need(gate.rows == intermediate_size / 2 && gate.columns == hidden_size,
                 "Unexpected local MLP shard dimensions");
            pointer(rank, "layer0_mlp_gate", gate.values,
                    static_cast<std::size_t>(gate.rows) * gate.columns);
            const auto& slab = decoder.tp_->partials[rank];
            const auto bytes = sizeof(Bf16) * hidden_size * tp_max_forward_lanes;
            pointer(rank, "attention_partial", slab.attention, bytes);
            pointer(rank, "mlp_partial", slab.hidden, bytes);
            probe.partials[rank][0] = slab.attention;
            probe.partials[rank][1] = slab.hidden;
        }
    }
};
}

void inspect_serving(const sglang::Decoder& decoder, int world) {
    sglang::MediaDecoderCheck::inspect(decoder, world);
}

extern "C" void watch_serving_draft(bool decode) {
    serving_probe().track_decode = decode;
}

extern "C" ncclResult_t real_all_gather(const void* source, void* dest, std::size_t count,
                                        ncclDataType_t type, ncclComm_t comm,
                                        cudaStream_t stream) asm("__real_ncclAllGather");
extern "C" ncclResult_t wrap_all_gather(const void* source, void* dest, std::size_t count,
                                        ncclDataType_t type, ncclComm_t comm,
                                        cudaStream_t stream) asm("__wrap_ncclAllGather");

extern "C" ncclResult_t wrap_all_gather(const void* source, void* dest, std::size_t count,
                                        ncclDataType_t type, ncclComm_t comm, cudaStream_t stream) {
    auto& probe = serving_probe();
    if (probe.draft_wave) {
        int rank = -1;
        int device = -1;
        cudaPointerAttributes source_attr{};
        cudaPointerAttributes dest_attr{};
        need(type == ncclBfloat16 &&
                 count == static_cast<std::size_t>(probe.observed_draft_rows) * hidden_size / 2 &&
                 cudaGetDevice(&rank) == cudaSuccess && rank >= 0 && rank < 2 &&
                 cudaStreamGetDevice(stream, &device) == cudaSuccess && device == rank &&
                 cudaPointerGetAttributes(&source_attr, source) == cudaSuccess &&
                 cudaPointerGetAttributes(&dest_attr, dest) == cudaSuccess &&
                 source_attr.device == rank && dest_attr.device == rank && source != dest &&
                 source == probe.draft_sources[rank] && dest == probe.draft_destinations[rank] &&
                 (probe.draft_gather_ranks & (1U << static_cast<unsigned>(rank))) == 0,
             "Invalid draft FC gather shard");
        probe.draft_gather_ranks |= 1U << static_cast<unsigned>(rank);
    }
    return real_all_gather(source, dest, count, type, comm, stream);
}

using DraftWave = void(MtpDraftInput* const (&)[2], int, bool, const cudaStream_t (&)[2],
                       const TpPartials (&)[2], std::uint8_t* const (&)[2], TensorParallel&,
                       const ConstraintWorkspace*, const ConstraintCandidates* const*);

extern "C" DraftWave SGLANG_REAL_DRAFT_WAVE;
extern "C" DraftWave SGLANG_WRAP_DRAFT_WAVE;

extern "C" void SGLANG_WRAP_DRAFT_WAVE(MtpDraftInput* const (&rows)[2], int count, bool score,
                                       const cudaStream_t (&streams)[2],
                                       const TpPartials (&buffers)[2],
                                       std::uint8_t* const (&packed)[2], TensorParallel& group,
                                       const ConstraintWorkspace* scratch,
                                       const ConstraintCandidates* const* allowed) {
    auto& probe = serving_probe();
    need(!probe.draft_wave && count > 0, "Invalid draft wave fixture call");
    for (int lane = 0; lane < count; ++lane) {
        const auto& root = rows[0][lane];
        const auto& peer = rows[1][lane];
        need(root.request != nullptr && peer.request != nullptr && root.request != peer.request &&
                 peer.position == root.position && peer.token == root.token,
             "Draft ranks did not process the same causal row");
        for (const auto* row : {&root, &peer}) {
            need(row->request->kv_elements_per_row() == 2 * 256 &&
                     row->request->valid_prefix() == row->position,
                 "Invalid local draft KV input");
        }
    }
    probe.draft_wave = true;
    probe.observed_draft_rows = count;
    for (int rank = 0; rank < 2; ++rank) {
        probe.draft_sources[rank] = buffers[rank].attention;
        probe.draft_destinations[rank] = buffers[rank].hidden;
    }
    TpArCounters before;
    TpArCounters after;
    group.snapshot_transport(before);
    probe.draft_gather_ranks = 0;
    SGLANG_REAL_DRAFT_WAVE(rows, count, score, streams, buffers, packed, group, scratch, allowed);
    probe.draft_wave = false;
    need(probe.draft_gather_ranks == 3, "Draft FC gather did not use both physical ranks");
    group.snapshot_transport(after);
    for (int rank = 0; rank < 2; ++rank) {
        const auto& previous = before.per_rank[rank];
        const auto& current = after.per_rank[rank];
        probe.draft_collectives[rank] += current.ar_calls_total - previous.ar_calls_total;
        probe.draft_payload[rank] +=
            current.ar_bytes_payload_total - previous.ar_bytes_payload_total;
    }
    for (int rank = 0; rank < 2; ++rank) {
        for (int lane = 0; lane < count; ++lane) {
            const auto& row = rows[rank][lane];
            need(row.request->valid_prefix() == row.position + 1 &&
                     (row.hidden != nullptr) == score,
                 "Draft sharded append or score did not commit");
            if (score) {
                cudaPointerAttributes local{};
                need(cudaPointerGetAttributes(&local, row.hidden) == cudaSuccess &&
                         local.device == rank,
                     "Draft score activated on the wrong rank");
            }
        }
        if (probe.track_decode && (score || scratch == nullptr))
            probe.draft_phase_rows[rank][score ? 0 : 1] += count;
    }
}

extern "C" TextLogitCandidate
real_draft_score(void*) asm("__real__ZN6sglang7decoder15MtpDraftRequest5State9tp_winnerEv");
extern "C" TextLogitCandidate
wrap_draft_score(void* state) asm("__wrap__ZN6sglang7decoder15MtpDraftRequest5State9tp_winnerEv");

extern "C" TextLogitCandidate wrap_draft_score(void* state) {
    auto& probe = serving_probe();
    const auto candidate = real_draft_score(state);
    if (probe.reading_draft) {
        need(probe.collected_scores < 2, "Draft verifier read too many vocabulary shards");
        const int begin = probe.collected_scores * (vocab_size / 2);
        need(candidate.token_id >= begin && candidate.token_id < begin + (vocab_size / 2) &&
                 std::isfinite(candidate.score),
             "Invalid vocabulary-shard score or token ID");
        probe.rank_candidates[probe.collected_scores++] = candidate;
    }
    return candidate;
}

extern "C" int real_draft_token(MtpDraftRequest&, MtpDraftRequest&) asm(
    "__real__ZN6sglang7decoder15MtpDraftRequest13read_tp_tokenERS1_S2_");
extern "C" int wrap_draft_token(MtpDraftRequest& root, MtpDraftRequest& peer) asm(
    "__wrap__ZN6sglang7decoder15MtpDraftRequest13read_tp_tokenERS1_S2_");

extern "C" int wrap_draft_token(MtpDraftRequest& root, MtpDraftRequest& peer) {
    auto& probe = serving_probe();
    probe.reading_draft = probe.track_decode;
    probe.collected_scores = 0;
    const int token = real_draft_token(root, peer);
    if (probe.reading_draft) {
        need(probe.collected_scores == 2, "Draft selector did not read both device score shards");
        const int rank = probe.rank_candidates[1].score > probe.rank_candidates[0].score ? 1 : 0;
        need(token == probe.rank_candidates[rank].token_id, "Draft selector chose the wrong shard");
        const auto& root_score = probe.rank_candidates[0];
        const auto& peer_score = probe.rank_candidates[1];
        std::cout << "draft_scores root=" << root_score.token_id << ':' << std::hexfloat
                  << root_score.score << " peer=" << peer_score.token_id << ':' << peer_score.score
                  << std::defaultfloat << " selected=" << token << '\n';
        ++probe.score_wins[rank];
        ++probe.scored_proposals;
    }
    probe.reading_draft = false;
    return token;
}

extern "C" ncclResult_t real_all_reduce(const void* source, void* dest, std::size_t count,
                                        ncclDataType_t type, ncclRedOp_t op, ncclComm_t comm,
                                        cudaStream_t stream) asm("__real_ncclAllReduce");
extern "C" ncclResult_t wrap_all_reduce(const void* source, void* dest, std::size_t count,
                                        ncclDataType_t type, ncclRedOp_t op, ncclComm_t comm,
                                        cudaStream_t stream) asm("__wrap_ncclAllReduce");

extern "C" ncclResult_t wrap_all_reduce(const void* source, void* dest, std::size_t count,
                                        ncclDataType_t type, ncclRedOp_t op, ncclComm_t comm,
                                        cudaStream_t stream) {
    auto& probe = serving_probe();
    if (probe.draft_wave || probe.world_size == 0 || probe.vendor_transport) {
        return real_all_reduce(source, dest, count, type, op, comm, stream);
    }
    const int rank = static_cast<int>(probe.calls % 2);
    const int boundary = static_cast<int>((probe.calls / 2) % 2);
    int device = -1;
    int stream_device = -1;
    need(probe.world_size == 2 && type == ncclBfloat16 && op == ncclSum && comm != nullptr &&
             stream != nullptr && cudaGetDevice(&device) == cudaSuccess && device == rank &&
             cudaStreamGetDevice(stream, &stream_device) == cudaSuccess && stream_device == rank &&
             count != 0 && count % hidden_size == 0 && (rank == 0 || probe.last_count == count) &&
             source == dest,
         "Unexpected target reduction");
    const auto index = probe.calls % (4 * static_cast<std::size_t>(layer_count));
    if (index == 0)
        probe.traversal_count = count;
    need(count == probe.traversal_count, "Row count changed within target traversal");
    if (index < 4) {
        probe.traversal_buffers[rank][boundary] = dest;
        const bool tiled = dest != probe.partials[rank][boundary];
        if (tiled) {
            need(count > hidden_size, "Single-row reduction missed registered decode slab");
            pointer(rank, boundary != 0 ? "prefill_mlp" : "prefill_attention", dest,
                    sizeof(Bf16) * count);
        }
        if (rank != 0) {
            need(tiled == (probe.traversal_buffers[0][boundary] != probe.partials[0][boundary]),
                 "Ranks dispatched different target paths");
            need(dest != probe.traversal_buffers[0][boundary], "Ranks alias the same reduction");
        }
        if (index == 3) {
            need(probe.traversal_buffers[0][0] != probe.traversal_buffers[0][1] &&
                     probe.traversal_buffers[1][0] != probe.traversal_buffers[1][1],
                 "Reduction phases alias");
            if (tiled)
                ++probe.tiled_passes;
            else
                ++probe.decode_passes;
        }
    } else {
        need(dest == probe.traversal_buffers[rank][boundary], "Unexpected per-layer partial slab");
    }
    probe.last_count = count;
    probe.target_rows += count / hidden_size;
    ++probe.calls;
    return real_all_reduce(source, dest, count, type, op, comm, stream);
}

void check_serving_reductions(const char* label, std::size_t minimum_rows) {
    auto& probe = serving_probe();
    const bool speculative = std::strncmp(label, "batch_mtp4_", 11) == 0;
    need((probe.world_size == 2 && speculative) ? probe.scored_proposals >= 8
                                                : probe.scored_proposals == 0,
         "Incomplete draft vocabulary scoring receipt");
    std::cout << "case=" << label << " scored_proposals=" << probe.scored_proposals
              << " rank0_wins=" << probe.score_wins[0] << " rank1_wins=" << probe.score_wins[1]
              << '\n';
    probe.scored_proposals = 0;
    probe.score_wins.fill(0);
    int draft_rank = 0;
    for (auto& rows : probe.draft_phase_rows) {
        const bool expect = probe.world_size == 2 && speculative;
        need(expect ? rows[0] >= 8 && rows[1] > 0 : rows[0] == 0 && rows[1] == 0,
             "Missing physical rank proposal or post-verification repair");
        std::cout << "case=" << label << " rank=" << draft_rank
                  << " draft_proposal_rows=" << rows[0] << " draft_repair_rows=" << rows[1] << '\n';
        rows[0] = rows[1] = 0;
        ++draft_rank;
    }
    if (probe.serving_group != nullptr) {
        TpArCounters current;
        probe.serving_group->snapshot_transport(current);
        constexpr auto boundaries = 2 * layer_count;
        constexpr auto row_bytes = sizeof(Bf16) * hidden_size;
        std::array<std::uint64_t, 2> target_calls{};
        std::array<std::uint64_t, 2> target_payload{};
        for (int rank = 0; rank < 2; ++rank) {
            const auto& now = current.per_rank[rank];
            const auto& previous = probe.target_origin.per_rank[rank];
            target_calls[rank] =
                now.ar_calls_total - previous.ar_calls_total - probe.draft_collectives[rank];
            target_payload[rank] = now.ar_bytes_payload_total - previous.ar_bytes_payload_total -
                                   probe.draft_payload[rank];
            need(target_calls[rank] != 0 && target_calls[rank] % boundaries == 0 &&
                     target_payload[rank] % (boundaries * row_bytes) == 0 &&
                     target_payload[rank] / (boundaries * row_bytes) >= minimum_rows,
                 "Rank target transport omitted a full model traversal");
            std::cout << "case=" << label << " rank=" << rank
                      << " target_collectives=" << target_calls[rank]
                      << " target_payload_bytes=" << target_payload[rank] << '\n';
        }
        need(target_calls[0] == target_calls[1] && target_payload[0] == target_payload[1],
             "Target transport ranks disagreed");
        probe.target_origin = std::move(current);
        probe.draft_collectives = {};
        probe.draft_payload = {};
    }
    if (probe.vendor_transport)
        return;
    constexpr int halves_per_pass = 2 * 2 * layer_count;
    const auto traversals = probe.target_rows / halves_per_pass;
    std::cout << "case=" << label << " nccl_allreduce_calls=" << probe.calls
              << " passes=" << probe.calls / halves_per_pass << " target_rows=" << traversals
              << " tiled=" << probe.tiled_passes << " decode=" << probe.decode_passes << '\n';
    const bool complete = probe.calls != 0 && probe.calls % halves_per_pass == 0 &&
                          probe.target_rows % halves_per_pass == 0 && traversals >= minimum_rows &&
                          probe.tiled_passes != 0 && probe.decode_passes != 0;
    need(probe.world_size == 2 ? complete : probe.calls == 0,
         "Incomplete 64-layer target traversal");
    probe.target_rows = 0;
    probe.calls = 0;
    probe.tiled_passes = probe.decode_passes = 0;
}
