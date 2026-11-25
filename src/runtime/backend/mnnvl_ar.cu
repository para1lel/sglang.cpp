// MNNVL one-shot allreduce transport: FlashInfer TRT-LLM MNNVL kernel
// embedded from source (Apache-2.0, flashinfer 0.7.4,
// trtllm_mnnvl_allreduce.cuh). Workspace mirrors
// flashinfer/comm/mnnvl.py McastGPUBuffer and
// flashinfer/comm/trtllm_mnnvl_ar.py MNNVLAllReduceFusionWorkspace: per-rank
// pinned allocations, a two-device multicast object, and triple Lamport
// buffers. Single process, one host thread enqueues both rank halves in rank
// order (the TensorParallel::allreduce_bf16 contract).
//
// Build: this TU must be compiled by nvcc (the embedded header carries device
// templates) with the flashinfer wheel headers on the include path, e.g.
//   -I<flashinfer>/data/include -I<flashinfer>/data/spdlog/include
// and linked against the CUDA driver (-lcuda).

#include "sglang/runtime/backend/mnnvl_ar.h"

#include <cuda.h>

#include <flashinfer/comm/trtllm_mnnvl_allreduce.cuh>
#include <cstdint>
#include <cstdio>

#include <stdexcept>
#include <string>
#include <utility>

namespace sglang::decoder {
namespace {

using Bf16 = __nv_bfloat16;

constexpr int kRanks = 2;
constexpr int kTokenDim = 5120;  // model hidden width; counts are whole rows
// Upstream MNNVL one-shot threshold (MNNVL_ONE_SHOT_THRESHOLD,
// trtllm_mnnvl_ar.py): tokens * hidden * ranks * element_size bytes.
constexpr std::uint64_t kOneshotMaxBytes = 1024 * 1024;
constexpr std::size_t kElementSize = sizeof(Bf16);
constexpr std::uint32_t kSignalPad = 2048;    // SIGNAL_PAD_SIZE
constexpr int kNumLamportBuffers = 3;         // NUM_LAMPORT_BUFFERS
constexpr std::uint32_t kLamportSentinel = 0x80000000u;  // fp32 -0.0 bits
// TWOSHOT_NUM_STAGES * PACKED_ALIGNMENT_BYTES from the upstream layout.
constexpr std::uint32_t kBufferAlignment = 2 * 16;

std::uint64_t round_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// Restores the caller's current CUDA device on scope exit.
class CallerDevice {
public:
    CallerDevice() { cudaGetDevice(&saved_); }
    ~CallerDevice() { cudaSetDevice(saved_); }
    void select(int device, const char* what) {
        cudaError_t status = cudaSetDevice(device);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
        }
    }

private:
    int saved_ = 0;
};

__global__ void fill_sentinel_kernel(std::uint32_t* target, std::size_t count) {
    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t stride = gridDim.x * blockDim.x;
    for (std::size_t i = index; i < count; i += stride) target[i] = kLamportSentinel;
}

std::string describe(CUresult status, const char* what) {
    const char* text = nullptr;
    cuGetErrorString(status, &text);
    return std::string(what) + ": " + (text ? text : "unknown driver error");
}

void require_cu(CUresult status, const char* what) {
    if (status != CUDA_SUCCESS) {
        throw std::runtime_error(describe(status, what));
    }
}

}  // namespace

struct MnnvlAllReduce::State {
    std::vector<int> devices;  // rank r -> CUDA device ordinal
    bool failed = false;

    // Multicast workspace (all zero/empty until created).
    CUmemGenericAllocationHandle physical[kRanks]{};
    CUmemGenericAllocationHandle multicast{};
    CUdeviceptr uc_base = 0;          // one VA range holding both ranks' buffers
    std::uint64_t uc_ptrs[kRanks]{};  // rank r buffer VA, readable from both devices
    CUdeviceptr mc_va = 0;            // multicast VA; writes land in every bound buffer
    std::uint64_t allocation_size = 0;
    std::uint32_t buffer_size_bytes = 0;  // usable bytes per Lamport buffer
    std::uint64_t max_elements = 0;       // oneshot envelope in elements

    // Per-rank device side of the Lamport protocol.
    std::uint32_t* flags[kRanks]{};  // 9-word Lamport flag state per rank
    void** ptrs_dev[kRanks]{};       // device array of both ranks' buffer VAs

    bool created_physical[kRanks] = {false, false};
    bool created_multicast = false;
    bool bound_multicast[kRanks] = {false, false};
    bool reserved_uc = false;
    bool mapped_uc[kRanks] = {false, false};
    bool reserved_mc = false;
    bool mapped_mc = false;

    // RAII: every workspace resource is released when the state dies, so a
    // failed create() (including an outer allocation failure after setup)
    // cannot leak driver handles.
    ~State();

    // Releases every workspace resource this state created; safe on partial
    // state and idempotent (defined below; driver handles are freed even
    // when an unmap fails, because we are already tearing down).
    void release();
};

MnnvlArError::MnnvlArError(const std::string& what, bool peer_enqueued_arg)
    : std::runtime_error(what), peer_enqueued(peer_enqueued_arg) {}

MnnvlAllReduce::MnnvlAllReduce(std::unique_ptr<State> state) : state_(std::move(state)) {}

int MnnvlAllReduce::rank_count() const noexcept { return kRanks; }

int MnnvlAllReduce::device(int rank) const { return state_->devices.at(rank); }

std::size_t MnnvlAllReduce::max_count() const noexcept { return state_->max_elements; }

bool MnnvlAllReduce::healthy() const noexcept { return !state_->failed; }

void MnnvlAllReduce::ensure_healthy() const {
    if (state_->failed) {
        throw std::runtime_error("An MNNVL allreduce failed earlier.");
    }
}

MnnvlAllReduce::State::~State() {
    release();
}

void MnnvlAllReduce::State::release() {
    // Flags and pointer arrays are ordinary per-rank allocations; the driver
    // workspace resources are freed even if the cudaFree calls fail (we are
    // already tearing down).
    CallerDevice device_scope;
    for (int r = 0; r < kRanks; ++r) {
        if (flags[r] != nullptr) {
            cudaSetDevice(devices[r]);
            cudaFree(flags[r]);
            flags[r] = nullptr;
        }
        if (ptrs_dev[r] != nullptr) {
            cudaSetDevice(devices[r]);
            cudaFree(ptrs_dev[r]);
            ptrs_dev[r] = nullptr;
        }
    }
    if (mapped_mc) {
        (void)cuMemUnmap(mc_va, allocation_size);
        mapped_mc = false;
    }
    if (reserved_mc) {
        (void)cuMemAddressFree(mc_va, allocation_size);
        reserved_mc = false;
    }
    for (int r = kRanks - 1; r >= 0; --r) {
        if (mapped_uc[r]) {
            (void)cuMemUnmap(uc_ptrs[r], allocation_size);
            mapped_uc[r] = false;
        }
    }
    if (reserved_uc) {
        (void)cuMemAddressFree(uc_base, allocation_size * kRanks);
        reserved_uc = false;
    }
    for (int r = kRanks - 1; r >= 0; --r) {
        if (bound_multicast[r]) {
            (void)cuMulticastUnbind(multicast, devices[r], 0, allocation_size);
            bound_multicast[r] = false;
        }
    }
    for (int r = kRanks - 1; r >= 0; --r) {
        if (created_physical[r]) {
            (void)cuMemRelease(physical[r]);
            created_physical[r] = false;
        }
    }
    if (created_multicast) {
        (void)cuMemRelease(multicast);
        created_multicast = false;
    }
}

MnnvlAllReduce::~MnnvlAllReduce() = default;

// Probe both devices before anything is set up: this TU carries an sm_100a
// image, so a non-SM100 device would only fail at the first enqueue. Reject
// up front and let the dispatcher fall back to NCCL.
void MnnvlAllReduce::probe_devices(State& state) {
    for (int r = 0; r < kRanks; ++r) {
        int major = 0;
        int minor = 0;
        cudaError_t status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                                    state.devices[r]);
        if (status == cudaSuccess) {
            status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                            state.devices[r]);
        }
        if (status != cudaSuccess || major != 10 || minor != 0) {
            throw std::runtime_error("The MNNVL transport requires SM100a devices.");
        }
    }
}

// Round the requested allocation up to the allocation and multicast
// granularities. The multicast object must cover the final rounded
// allocation; its size field is refreshed after every round-up (a stale,
// smaller value is only latent while the Lamport data fits under it).
// Returns the multicast granularity, which also paces VA reservations.
std::uint64_t MnnvlAllReduce::size_workspace(State& state, std::uint64_t requested) {
    CUmemAllocationProp alloc_prop{};
    alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    alloc_prop.location.id = state.devices[0];
    alloc_prop.allocFlags.gpuDirectRDMACapable = 1;
    std::size_t granularity = 0;
    require_cu(cuMemGetAllocationGranularity(&granularity, &alloc_prop,
                                             CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
               "Query allocation granularity");
    state.allocation_size = round_up(requested + kSignalPad, granularity);

    CUmulticastObjectProp mc_prop{};
    mc_prop.numDevices = kRanks;
    mc_prop.size = state.allocation_size;
    mc_prop.handleTypes = 0;  // single process: no shareable handle
    mc_prop.flags = 0;
    std::size_t mc_granularity = 0;
    require_cu(cuMulticastGetGranularity(&mc_granularity, &mc_prop,
                                         CU_MULTICAST_GRANULARITY_RECOMMENDED),
               "Query multicast granularity");
    state.allocation_size = round_up(state.allocation_size, mc_granularity);
    mc_prop.size = state.allocation_size;
    require_cu(cuMulticastGetGranularity(&mc_granularity, &mc_prop,
                                         CU_MULTICAST_GRANULARITY_RECOMMENDED),
               "Re-query multicast granularity");
    state.allocation_size = round_up(state.allocation_size, mc_granularity);
    mc_prop.size = state.allocation_size;
    return mc_granularity;
}

// Allocate the per-rank pinned buffers and the multicast object, then add
// both devices and bind each rank buffer into it.
void MnnvlAllReduce::create_physical(State& state) {
    CallerDevice device_scope;
    CUmemAllocationProp alloc_prop{};
    alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    alloc_prop.allocFlags.gpuDirectRDMACapable = 1;

    CUmulticastObjectProp mc_prop{};
    mc_prop.numDevices = kRanks;
    mc_prop.size = state.allocation_size;
    mc_prop.handleTypes = 0;  // single process: no shareable handle
    mc_prop.flags = 0;

    for (int r = 0; r < kRanks; ++r) {
        device_scope.select(state.devices[r], "Select a rank device");
        alloc_prop.location.id = state.devices[r];
        require_cu(cuMemCreate(&state.physical[r], state.allocation_size, &alloc_prop, 0),
                   "Create the rank buffer");
        state.created_physical[r] = true;
    }
    require_cu(cuMulticastCreate(&state.multicast, &mc_prop), "Create the multicast object");
    state.created_multicast = true;
    for (int r = 0; r < kRanks; ++r) {
        require_cu(cuMulticastAddDevice(state.multicast, state.devices[r]),
                   "Add a device to the multicast object");
    }
    for (int r = 0; r < kRanks; ++r) {
        require_cu(cuMulticastBindMem(state.multicast, 0, state.physical[r], 0,
                                      state.allocation_size, 0),
                   "Bind a rank buffer to the multicast object");
        state.bound_multicast[r] = true;
    }
}

// Map the unicast VA range (both ranks' buffers, accessible from both
// devices) and the multicast VA range.
void MnnvlAllReduce::map_workspace(State& state, std::uint64_t mc_granularity) {
    require_cu(cuMemAddressReserve(&state.uc_base, state.allocation_size * kRanks,
                                   mc_granularity, 0, 0),
               "Reserve the unicast range");
    state.reserved_uc = true;
    for (int r = 0; r < kRanks; ++r) {
        state.uc_ptrs[r] =
            state.uc_base + static_cast<std::uint64_t>(r) * state.allocation_size;
        require_cu(cuMemMap(state.uc_ptrs[r], state.allocation_size, 0, state.physical[r], 0),
                   "Map a rank buffer");
        state.mapped_uc[r] = true;
    }
    CUmemAccessDesc access[kRanks];
    for (int r = 0; r < kRanks; ++r) {
        access[r].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access[r].location.id = state.devices[r];
        access[r].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    require_cu(cuMemSetAccess(state.uc_base, state.allocation_size * kRanks, access, kRanks),
               "Grant unicast access");

    // Multicast VA, accessible from both devices.
    require_cu(cuMemAddressReserve(&state.mc_va, state.allocation_size, mc_granularity, 0, 0),
               "Reserve the multicast range");
    state.reserved_mc = true;
    require_cu(cuMemMap(state.mc_va, state.allocation_size, 0, state.multicast, 0),
               "Map the multicast range");
    state.mapped_mc = true;
    require_cu(cuMemSetAccess(state.mc_va, state.allocation_size, access, kRanks),
               "Grant multicast access");
}

// Per-Lamport-buffer bytes, mirroring upstream exactly: usable size is the
// allocation minus the signal pad (McastGPUBuffer::get_size, mnnvl.py), and
// each buffer is a third of that, aligned down to 32 bytes
// (trtllm_mnnvl_ar.py). The uint32 bound is checked where the flag word is
// derived. Then prime the per-rank Lamport flag state and buffer-pointer
// arrays, and pre-dirty the data area with the sentinel.
void MnnvlAllReduce::init_lamport(State& state) {
    const std::uint64_t usable = state.allocation_size - kSignalPad;
    const std::uint64_t per_buffer =
        usable / kNumLamportBuffers / kBufferAlignment * kBufferAlignment;
    if (per_buffer > 0xFFFFFFFFull) {
        throw std::runtime_error("The multicast workspace exceeds the Lamport flag width.");
    }
    state.buffer_size_bytes = static_cast<std::uint32_t>(per_buffer);

    CallerDevice device_scope;
    for (int r = 0; r < kRanks; ++r) {
        device_scope.select(state.devices[r], "Select a rank device");
        cudaError_t status = cudaMalloc(&state.flags[r], sizeof(std::uint32_t) * 9);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("Allocate the Lamport flags: ") +
                                     cudaGetErrorString(status));
        }
        status = cudaMalloc(&state.ptrs_dev[r], sizeof(void*) * kRanks);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("Allocate the buffer pointers: ") +
                                     cudaGetErrorString(status));
        }
        // [cur, dirty, bytesPerBuffer, dirtyNumStages, numBytesToClear x4,
        //  accessCountPtr]
        const std::uint32_t flag_host[9] = {0, 2, state.buffer_size_bytes, 0, 0, 0, 0, 0, 0};
        status = cudaMemcpy(state.flags[r], flag_host, sizeof(flag_host), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("Initialize the Lamport flags: ") +
                                     cudaGetErrorString(status));
        }
        void* ptr_host[kRanks] = {reinterpret_cast<void*>(state.uc_ptrs[0]),
                                  reinterpret_cast<void*>(state.uc_ptrs[1])};
        status = cudaMemcpy(state.ptrs_dev[r], ptr_host, sizeof(ptr_host), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("Initialize the buffer pointers: ") +
                                     cudaGetErrorString(status));
        }
        fill_sentinel_kernel<<<256, 256, 0, 0>>>(
            reinterpret_cast<std::uint32_t*>(state.uc_ptrs[r]),
            (state.allocation_size - kSignalPad) / sizeof(std::uint32_t));
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("Pre-dirty the Lamport buffers: ") +
                                     cudaGetErrorString(status));
        }
    }
    for (int r = 0; r < kRanks; ++r) {
        device_scope.select(state.devices[r], "Select a rank device");
        cudaError_t status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("Finish the Lamport initialization: ") +
                                     cudaGetErrorString(status));
        }
    }
}

std::unique_ptr<MnnvlAllReduce> MnnvlAllReduce::create(std::vector<int> devices) {
    if (devices.size() != kRanks) {
        throw std::invalid_argument("MnnvlAllReduce requires exactly two devices.");
    }
    if (devices[0] == devices[1]) {
        throw std::invalid_argument("MnnvlAllReduce requires distinct devices.");
    }

    auto state = std::make_unique<State>();
    state->devices = std::move(devices);
    // Coarse failure category for the setup line (mirrors the dispatch
    // layer's fallback reasons); refined as setup progresses. Declared before
    // the try so the catch handler can report it.
    const char* reason = "no_vmm";
    try {
        reason = "device_unsupported";
        probe_devices(*state);

        reason = "multicast_create_failed";
        // Size the workspace to the one-shot envelope so any count up to
        // max_count() dispatches without reallocation.
        state->max_elements = kOneshotMaxBytes / (kElementSize * kRanks);
        const std::uint64_t oneshot_bytes = state->max_elements * kElementSize * kRanks;
        const std::uint64_t buffer_size = round_up(oneshot_bytes, kBufferAlignment);
        const std::uint64_t requested = buffer_size * kNumLamportBuffers;
        const std::uint64_t mc_granularity = size_workspace(*state, requested);
        create_physical(*state);

        reason = "no_vmm";
        map_workspace(*state, mc_granularity);
        init_lamport(*state);
    } catch (...) {
        std::printf(
            "tp2_transport status=nccl ranks=%d devices=%d,%d oneshot_max_bytes=%llu "
            "workspace_bytes_per_rank=%llu reason=%s\n",
            kRanks, state->devices[0], state->devices[1],
            static_cast<unsigned long long>(kOneshotMaxBytes),
            static_cast<unsigned long long>(state->allocation_size), reason);
        return nullptr;
    }
    std::printf(
        "tp2_transport status=mnnvl ranks=%d devices=%d,%d oneshot_max_bytes=%llu "
        "workspace_bytes_per_rank=%llu reason=ok\n",
        kRanks, state->devices[0], state->devices[1],
        static_cast<unsigned long long>(kOneshotMaxBytes),
        static_cast<unsigned long long>(state->allocation_size));
    return std::unique_ptr<MnnvlAllReduce>(new MnnvlAllReduce(std::move(state)));
}

// Preflight before the first launch: a stream on the wrong device, a stream
// that already carries an error, or a half-capturing rank pair would leave a
// kernel spinning on Lamport flags (a graph-recorded kernel cannot
// synchronize with an eager peer), so both ranks are checked before anything
// is enqueued. All checks are cheap host-side calls that alter no pending
// error state; cudaStreamGetDevice is plain metadata and runs in every mode,
// while the pending-error probe via cudaStreamQuery is only legal on a
// stream that is not capturing. The capture-state query itself fails
// closed: an error return or an invalidated capture rejects the call
// instead of guessing.
void MnnvlAllReduce::preflight(State& state, __nv_bfloat16* const buffers[],
                               const cudaStream_t streams[]) {
    cudaStreamCaptureStatus capture_status[kRanks];
    for (int r = 0; r < kRanks; ++r) {
        int stream_device = -1;
        if (cudaStreamGetDevice(streams[r], &stream_device) != cudaSuccess ||
            stream_device != state.devices[r]) {
            throw std::invalid_argument(
                "MnnvlAllReduce::allreduce_bf16 requires each stream to belong to its "
                "rank's device.");
        }
        const cudaError_t capture_error = cudaStreamIsCapturing(streams[r], &capture_status[r]);
        if (capture_error != cudaSuccess) {
            state.failed = true;
            throw MnnvlArError(
                std::string("Query the allreduce stream capture state: ") +
                    cudaGetErrorString(capture_error),
                false);
        }
        if (capture_status[r] == cudaStreamCaptureStatusNone) {
            const cudaError_t pending = cudaStreamQuery(streams[r]);
            if (pending != cudaSuccess && pending != cudaErrorNotReady) {
                state.failed = true;
                throw MnnvlArError(
                    std::string("The allreduce stream already carries an error: ") +
                        cudaGetErrorString(pending),
                    false);
            }
        } else if (capture_status[r] != cudaStreamCaptureStatusActive) {
            throw std::invalid_argument(
                "MnnvlAllReduce::allreduce_bf16 requires a stream whose capture is "
                "active, not invalidated.");
        }
        if (reinterpret_cast<std::uintptr_t>(buffers[r]) % sizeof(float4) != 0) {
            throw std::invalid_argument(
                "MnnvlAllReduce::allreduce_bf16 requires 16-byte aligned buffers.");
        }
    }
    if (capture_status[0] != capture_status[1]) {
        throw std::invalid_argument(
            "MnnvlAllReduce::allreduce_bf16 requires both ranks to share one capture "
            "state; a graph-recorded kernel cannot Lamport-synchronize with an "
            "eager peer.");
    }
}

void MnnvlAllReduce::allreduce_bf16(__nv_bfloat16* const buffers[], const cudaStream_t streams[],
                                    std::size_t count) {
    if (count == 0) {
        throw std::invalid_argument(
            "MnnvlAllReduce::allreduce_bf16 requires a positive element count.");
    }
    for (int r = 0; r < kRanks; ++r) {
        if (buffers[r] == nullptr || streams[r] == nullptr) {
            throw std::invalid_argument(
                "MnnvlAllReduce::allreduce_bf16 requires a buffer and a stream for every "
                "rank.");
        }
    }
    if (count % kTokenDim != 0) {
        throw std::invalid_argument(
            "MnnvlAllReduce::allreduce_bf16 requires a whole number of hidden-width rows.");
    }
    if (count > state_->max_elements) {
        throw std::invalid_argument(
            "MnnvlAllReduce::allreduce_bf16 requires a count within the one-shot envelope.");
    }
    ensure_healthy();
    preflight(*state_, buffers, streams);

    const int tokens = static_cast<int>(count / kTokenDim);
    CallerDevice device_scope;
    for (int r = 0; r < kRanks; ++r) {
        // Both ranks' decisions were made before any launch; a failure here
        // poisons the group and the operation surfaces an error rather than
        // retrying the collective through NCCL.
        device_scope.select(state_->devices[r], "Select a rank device");
        flashinfer::trtllm_mnnvl_allreduce::AllReduceFusionParams params{};
        params.nRanks = kRanks;
        params.rank = r;
        params.numTokens = tokens;
        params.tokenDim = kTokenDim;
        params.bufferPtrsDev = state_->ptrs_dev[r];
        params.bufferPtrLocal = reinterpret_cast<void*>(state_->uc_ptrs[r]);
        params.multicastPtr = reinterpret_cast<void*>(state_->mc_va);
        params.bufferFlags = state_->flags[r];
        params.rmsNormFusion = false;
        params.launchWithPdl = false;
        params.input = buffers[r];
        params.residualIn = nullptr;
        params.gamma = nullptr;
        params.epsilon = 0.0;
        params.weightBias = 0.0f;
        params.quantType = flashinfer::trtllm_mnnvl_allreduce::QuantType::kNone;
        params.residualOut = nullptr;
        params.output = buffers[r];  // in-place: input == output
        params.quantOut = nullptr;
        params.scalingFactorOut = nullptr;
        params.stream = streams[r];
        const cudaError_t status =
            flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionDispatch<Bf16>(params);
        if (status != cudaSuccess) {
            state_->failed = true;
            throw MnnvlArError(std::string("Enqueue the MNNVL BF16 allreduce: ") +
                                   cudaGetErrorString(status),
                               r > 0);
        }
    }
}

}  // namespace sglang::decoder
