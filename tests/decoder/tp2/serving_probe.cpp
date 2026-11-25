#include "sglang/checkpoint.h"
#include "sglang/decoder_internal.h"
#include "sglang/decoder_mtp.h"

#include <cudaTypedefs.h>
#include <nccl.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
using namespace sglang::decoder;
const void* partials[2][2]{};
std::size_t calls = 0, last_count = 0, target_rows = 0;
int world_size = 0;
const void* traversal_buffers[2][2]{};
std::size_t traversal_count = 0;
std::size_t tiled_passes = 0, decode_passes = 0;
constexpr std::size_t draft_bytes = sizeof(Bf16) * hidden_size * 2 * hidden_size;
using DraftSample = std::array<std::array<std::byte, 32>, 3>;
constexpr std::array<std::size_t, 3> offsets{0, draft_bytes / 2, draft_bytes - 32};
DraftSample expected_fc{};
const void* draft_fc[2]{};
bool track_upload = false;

void need(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void pointer(int rank, const char* name, const void* data, std::size_t bytes) {
    cudaPointerAttributes attributes{};
    need(data && bytes && cudaPointerGetAttributes(&attributes, data) == cudaSuccess &&
         attributes.type == cudaMemoryTypeDevice && attributes.device == rank,
         "Weight or partial buffer belongs to the wrong device");
    void* entry = nullptr;
    need(cudaGetDriverEntryPointByVersion("cuMemGetAddressRange", &entry, 3020,
                                          cudaEnableDefault, nullptr) == cudaSuccess && entry,
         "No CUDA allocation-range query");
    const auto locate = reinterpret_cast<PFN_cuMemGetAddressRange_v3020>(entry);
    const auto address = reinterpret_cast<CUdeviceptr>(data);
    CUdeviceptr base = 0;
    std::size_t backing = 0;
    need(locate(&base, &backing, address) == CUDA_SUCCESS && address >= base &&
         address - base <= backing && bytes <= backing - (address - base),
         "Tensor span exceeds its CUDA allocation");
    std::printf("rank=%d allocation=%s address=%p bytes=%zu base=%p backing_bytes=%zu\n",
                rank, name, data, bytes, reinterpret_cast<void*>(base), backing);
}
}

void watch_draft_upload(const std::filesystem::path& directory) {
    sglang::Checkpoint checkpoint(directory);
    const auto& tensor = checkpoint.tensor("mtp.fc.weight", sglang::TensorDtype::BF16,
                                           {hidden_size, 2 * hidden_size});
    need(tensor.size_bytes == draft_bytes, "Invalid MTP FC extent");
    for (std::size_t i = 0; i < offsets.size(); ++i) {
        std::memcpy(expected_fc[i].data(), tensor.data + offsets[i], expected_fc[i].size());
    }
    track_upload = true;
}

extern "C" cudaError_t __real_cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind,
                                               cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* dest, const void* source, std::size_t bytes,
                                               cudaMemcpyKind kind, cudaStream_t stream) {
    const auto status = __real_cudaMemcpyAsync(dest, source, bytes, kind, stream);
    if (status == cudaSuccess && track_upload && bytes == draft_bytes &&
        kind == cudaMemcpyHostToDevice) {
        const auto* data = static_cast<const std::byte*>(source);
        for (std::size_t i = 0; i < offsets.size(); ++i) {
            if (std::memcmp(data + offsets[i], expected_fc[i].data(), expected_fc[i].size())) {
                return status;
            }
        }
        int rank = -1;
        need(cudaGetDevice(&rank) == cudaSuccess && rank >= 0 && rank < 2 && !draft_fc[rank],
             "Duplicate or misplaced MTP FC upload");
        draft_fc[rank] = dest;
    }
    return status;
}

namespace sglang {
class MediaDecoderCheck {
public:
    static void inspect(const Decoder& decoder, int world) {
        world_size = world;
        track_upload = false;
        const auto* ranks = decoder.tensor_parallel_info();
        need((ranks != nullptr) == (world == 2), "Unexpected public TP receipt");
        if (world == 1) return;
        need(ranks->world_size == 2 && decoder.tp_ && decoder.tp_->group, "Missing TP group");
        need(std::memcmp(ranks->ranks[0].uuid, ranks->ranks[1].uuid, 16), "Duplicate rank UUID");
        for (int rank = 0; rank < world; ++rank) {
            decoder::DeviceScope selected(rank);
            const auto& model = rank == 0 ? *decoder.model_ : *decoder.tp_->peer;
            const auto& actual = ranks->ranks[rank];
            cudaDeviceProp device{};
            need(cudaGetDeviceProperties(&device, rank) == cudaSuccess && model.device == rank &&
                 actual.cuda_device == rank && actual.local_rank == rank &&
                 actual.pid == getpid() && std::memcmp(device.uuid.bytes, actual.uuid, 16) == 0 &&
                 actual.nccl_established, "Invalid rank identity");
            const auto draft = model.mtp_weights ? model.mtp_weights->device_allocation_bytes() : 0;
            const bool has_head = model.lm_head.values != nullptr;
            need(has_head == (rank == 0) && actual.lm_head == has_head &&
                 (draft > 0) == (rank == 0) && actual.mtp == (draft > 0), "Head/draft ownership");
            need(actual.weight_bytes == model.memory->bytes() + draft, "Rank byte mismatch");
            need((draft_fc[rank] != nullptr) == (rank == 0), "Draft FC allocation ownership");
            if (draft_fc[rank]) {
                pointer(rank, "mtp_fc", draft_fc[rank], draft_bytes);
                const auto* base = static_cast<const std::byte*>(draft_fc[rank]);
                DraftSample readback{};
                for (std::size_t i = 0; i < offsets.size(); ++i) {
                    need(cudaMemcpy(readback[i].data(), base + offsets[i], readback[i].size(),
                                    cudaMemcpyDeviceToHost) == cudaSuccess, "Read draft FC");
                }
                need(readback == expected_fc, "Draft FC checkpoint byte mismatch");
            }
            using namespace sglang::decoder;
            std::printf("rank=%d device=%d pid=%ld weight_bytes=%zu draft_bytes=%zu uuid=",
                        rank, actual.cuda_device, actual.pid, actual.weight_bytes, draft);
            for (unsigned char byte : actual.uuid) std::printf("%02x", byte);
            std::puts("");
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
            pointer(rank, "layer0_mlp_gate", gate.values, std::size_t(gate.rows) * gate.columns);
            const auto& slab = decoder.tp_->partials[rank];
            const auto bytes = sizeof(Bf16) * hidden_size * tp_max_forward_lanes;
            pointer(rank, "attention_partial", slab.attention, bytes);
            pointer(rank, "mlp_partial", slab.hidden, bytes);
            partials[rank][0] = slab.attention;
            partials[rank][1] = slab.hidden;
        }
    }
};
}

void inspect_serving(const sglang::Decoder& decoder, int world) {
    sglang::MediaDecoderCheck::inspect(decoder, world);
}

extern "C" ncclResult_t __real_ncclAllReduce(const void*, void*, std::size_t, ncclDataType_t,
                                              ncclRedOp_t, ncclComm_t, cudaStream_t);

extern "C" ncclResult_t __wrap_ncclAllReduce(const void* source, void* dest, std::size_t count,
                                              ncclDataType_t type, ncclRedOp_t op,
                                              ncclComm_t comm, cudaStream_t stream) {
    const int rank = int(calls % 2), boundary = int((calls / 2) % 2);
    int device = -1, stream_device = -1;
    need(world_size == 2 && type == ncclBfloat16 && op == ncclSum && comm && stream &&
         cudaGetDevice(&device) == cudaSuccess && device == rank &&
         cudaStreamGetDevice(stream, &stream_device) == cudaSuccess && stream_device == rank &&
         count && count % hidden_size == 0 && (!rank || last_count == count) && source == dest,
         "Unexpected target reduction");
    const auto index = calls % (4 * layer_count);
    if (!index) traversal_count = count;
    need(count == traversal_count, "Row count changed within target traversal");
    if (index < 4) {
        traversal_buffers[rank][boundary] = dest;
        const bool tiled = dest != partials[rank][boundary];
        if (tiled) {
            need(count > hidden_size, "Single-row reduction missed registered decode slab");
            pointer(rank, boundary ? "prefill_mlp" : "prefill_attention", dest,
                    sizeof(Bf16) * count);
        }
        if (rank) {
            need(tiled == (traversal_buffers[0][boundary] != partials[0][boundary]),
                 "Ranks dispatched different target paths");
            need(dest != traversal_buffers[0][boundary], "Ranks alias the same reduction");
        }
        if (index == 3) {
            need(traversal_buffers[0][0] != traversal_buffers[0][1] &&
                 traversal_buffers[1][0] != traversal_buffers[1][1], "Reduction phases alias");
            if (tiled) ++tiled_passes;
            else ++decode_passes;
        }
    } else {
        need(dest == traversal_buffers[rank][boundary], "Unexpected per-layer partial slab");
    }
    last_count = count;
    target_rows += count / hidden_size;
    ++calls;
    return __real_ncclAllReduce(source, dest, count, type, op, comm, stream);
}

void check_serving_reductions(const char* label, std::size_t minimum_rows) {
    constexpr int halves_per_pass = 2 * 2 * layer_count;
    const auto traversals = target_rows / halves_per_pass;
    std::printf("case=%s nccl_allreduce_calls=%zu passes=%zu target_rows=%zu tiled=%zu "
                "decode=%zu\n", label, calls, calls / halves_per_pass, traversals, tiled_passes,
                decode_passes);
    const bool complete = calls && calls % halves_per_pass == 0 &&
        target_rows % halves_per_pass == 0 && traversals >= minimum_rows && tiled_passes &&
        decode_passes;
    need(world_size == 2 ? complete : calls == 0, "Incomplete 64-layer target traversal");
    target_rows = 0;
    calls = 0;
    tiled_passes = decode_passes = 0;
}
