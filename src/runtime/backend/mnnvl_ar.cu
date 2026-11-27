#include "sglang/runtime/backend/mnnvl_ar.h"
#include "sglang/runtime/backend/mnnvl_rms_quantize.h"
#include "sglang/text/quantization/rms_quantize.cuh"
#include "sglang/runtime/execution/tp_verify_graph.h"

#include <cuda.h>
#include <cuda/atomic>

#include <array>
#include <cstdint>
#include <cstdio>
#include <new>

#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace sglang::decoder {
namespace {

using Bf16 = __nv_bfloat16;

constexpr int kRanks = 2;
constexpr int kTokenDim = 5120;   
constexpr std::uint64_t kOneshotMaxBytes = 1024 * 1024;
constexpr int kTwoshotRows = 80;
constexpr int kTwoshotThreads = 128;
constexpr std::size_t kElementSize = sizeof(Bf16);
constexpr std::uint32_t kSignalPad = 2048;     
constexpr int kNumLamportBuffers = 3;          
constexpr std::uint32_t kLamportSentinel = 0x80000000u;   
constexpr std::uint32_t kBufferAlignment = 2 * 16;
constexpr std::size_t kTwoshotCount = kTwoshotRows * kTokenDim;
constexpr std::size_t kTwoshotBufferBytes = 2 * kTwoshotCount * kElementSize;
constexpr int kBf16PerPack = sizeof(uint4) / kElementSize;
constexpr unsigned kRowBytes = kTokenDim * kElementSize;
constexpr int packs_per_row = kTokenDim / kBf16PerPack;
constexpr int kTwoshotBlocksPerRow = packs_per_row / kTwoshotThreads;
constexpr int kTwoshotGrid = kTwoshotRows * kTwoshotBlocksPerRow;
static_assert(kTwoshotRows % kRanks == 0 && packs_per_row % kTwoshotThreads == 0);

struct alignas(16) LamportState {
    uint4 layout;
    uint2 dirty_bytes;
    std::uint32_t arrivals;
};

template <int Stages>
struct LamportBuffer {
    LamportState* state;
    uint4 layout;
    uint2 dirty_bytes;

    __device__ explicit LamportBuffer(LamportState* flags)
        : state(flags), layout(flags->layout), dirty_bytes(flags->dirty_bytes) {}

    __device__ uint4* current(void* base, int stage = 0) const {
        const unsigned capacity = layout.z / sizeof(uint4);
        return static_cast<uint4*>(base) + layout.x * capacity + stage * (capacity / Stages);
    }

    __device__ void clear(void* base, unsigned stage) const {
        if (stage >= layout.w) return;
        constexpr uint4 sentinel{kLamportSentinel, kLamportSentinel,
                                 kLamportSentinel, kLamportSentinel};
        const unsigned capacity = layout.z / sizeof(uint4);
        auto* dirty = static_cast<uint4*>(base) + layout.y * capacity + stage * (capacity / 2);
        const unsigned packs = (stage == 0 ? dirty_bytes.x : dirty_bytes.y) / sizeof(uint4);
        const unsigned first = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
        const unsigned stride = gridDim.x * gridDim.y * blockDim.x;
        for (unsigned index = first; index < packs; index += stride) dirty[index] = sentinel;
    }

    __device__ void arrive() const {
        const unsigned threads = (blockDim.x + 31) & ~31u;
        if (threadIdx.x < 32) {
            asm volatile("barrier.cta.sync 1, %0;" :: "r"(threads) : "memory");
            if (threadIdx.x == 0) count_arrival();
        } else {
            asm volatile("barrier.cta.arrive 1, %0;" :: "r"(threads) : "memory");
        }
    }

    __device__ void arrive_cluster() const {
        __cluster_barrier_arrive();
        __cluster_barrier_wait();
        if (blockIdx.y == 0 && threadIdx.x == 0) count_arrival();
    }

    __device__ void rotate(unsigned groups, uint2 next_dirty_bytes) const {
        if (blockIdx.x != 0 || blockIdx.y != 0 || threadIdx.x != 0) return;
        unsigned arrivals;
        do {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(arrivals)
                         : "l"(&state->arrivals) : "memory");
        } while (arrivals < groups);
        state->layout = {(layout.x + 1) % kNumLamportBuffers, layout.x, layout.z, Stages};
        state->dirty_bytes = next_dirty_bytes;
        state->arrivals = 0;
    }

private:
    __device__ void count_arrival() const {
        asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" ::
                     "l"(&state->arrivals), "r"(1) : "memory");
    }
};

union PackedBf16 {
    uint4 packed;
    Bf16 elements[kBf16PerPack];
};

__device__ void sanitize_payload(PackedBf16& value) {
#pragma unroll
    for (int index = 0; index < kBf16PerPack; ++index) {
        if (__bfloat16_as_ushort(value.elements[index]) == 0x8000u) {
            value.elements[index] = __float2bfloat16_rn(0.f);
        }
    }
}

__device__ void store_multicast_pack(uint4* address, uint4 pack) {
    asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
                 :: "l"(address), "f"(__uint_as_float(pack.x)), "f"(__uint_as_float(pack.y)),
                    "f"(__uint_as_float(pack.z)), "f"(__uint_as_float(pack.w)) : "memory");
}

__device__ bool poll_pack(const uint4* address, PackedBf16& value) {
    auto& pack = value.packed;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(pack.x), "=r"(pack.y), "=r"(pack.z), "=r"(pack.w)
                 : "l"(address) : "memory");
    return pack.x != kLamportSentinel && pack.y != kLamportSentinel &&
           pack.z != kLamportSentinel && pack.w != kLamportSentinel;
}

__device__ PackedBf16 reduce_packs(const PackedBf16& rank0, const PackedBf16& rank1) {
    PackedBf16 reduced;
#pragma unroll
    for (int index = 0; index < kBf16PerPack; ++index) {
        const float first = 0.f + __bfloat162float(rank0.elements[index]);
        const float sum = first + __bfloat162float(rank1.elements[index]);
        reduced.elements[index] = __float2bfloat16_rn(sum);
    }
    return reduced;
}

template <int Rank>
__device__ PackedBf16 reduce_local_remote(const PackedBf16& local, const PackedBf16& remote) {
    if constexpr (Rank == 0) return reduce_packs(local, remote);
    return reduce_packs(remote, local);
}

struct AllReduceArgs {
    uint4* values;
    void* local;
    void* peer;
    void* multicast;
    LamportState* flags;
};

template <int Rank>
__global__ __launch_bounds__(1024) void oneshot_allreduce(AllReduceArgs args) {
    const unsigned row = blockIdx.x;
    const unsigned offset = blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned packed = row * packs_per_row + offset;
    LamportBuffer<1> flags(args.flags);
    PackedBf16 value{args.values[packed]};
    sanitize_payload(value);
    auto* multicast = flags.current(args.multicast);
    auto* local = flags.current(args.local);
    store_multicast_pack(multicast + (row * kRanks + Rank) * packs_per_row + offset, value.packed);
    flags.arrive_cluster();
    flags.clear(args.local, 0);
    flags.clear(args.local, 1);
    PackedBf16 remote;
    while (!poll_pack(local + (row * kRanks + 1 - Rank) * packs_per_row + offset, remote)) {}
    args.values[packed] = reduce_local_remote<Rank>(value, remote).packed;
    flags.rotate(gridDim.x, {gridDim.x * kRanks * kRowBytes, 0});
}

__global__ __launch_bounds__(kTwoshotThreads) void twoshot_allreduce(AllReduceArgs args, int rank) {
    const int row = blockIdx.x;
    const int offset = blockIdx.y * kTwoshotThreads + threadIdx.x;
    const int packed = row * packs_per_row + offset;
    const bool owner = row % kRanks == rank;
    const int scatter_offset = row / kRanks * kRanks * packs_per_row + offset;
    LamportBuffer<2> flags(args.flags);
    auto* scatter_local = flags.current(args.local);
    auto* scatter_dest = owner ? scatter_local : flags.current(args.peer);
    PackedBf16 value{args.values[packed]};
    sanitize_payload(value);
    scatter_dest[scatter_offset + rank * packs_per_row] = value.packed;
    flags.clear(args.local, 0);
    if (owner) {
        PackedBf16 rank0;
        PackedBf16 rank1;
        bool ready;
        do {
            const bool rank0_ready = poll_pack(scatter_local + scatter_offset, rank0);
            const bool rank1_ready = poll_pack(scatter_local + scatter_offset + packs_per_row,
                                               rank1);
            ready = rank0_ready && rank1_ready;
        } while (!ready);
        auto reduced = reduce_packs(rank0, rank1);
        sanitize_payload(reduced);
        store_multicast_pack(flags.current(args.multicast, 1) + packed, reduced.packed);
    }
    flags.clear(args.local, 1);
    flags.arrive();
    PackedBf16 result;
    while (!poll_pack(flags.current(args.local, 1) + packed, result)) {}
    args.values[packed] = result.packed;
    flags.rotate(kTwoshotGrid, {kTwoshotRows * kRowBytes, kTwoshotRows * kRowBytes});
}

bool twoshot_grid_fits(int multiprocessors) {
    int active_blocks = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, twoshot_allreduce, kTwoshotThreads, 0) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return active_blocks * multiprocessors >= kTwoshotGrid;
}

cudaError_t launch_oneshot(AllReduceArgs args, int rank, int rows, int multiprocessors,
                           cudaStream_t stream) {
    int cluster = 8;
    while (rows * cluster > multiprocessors && cluster > 1) cluster /= 2;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim = {1, static_cast<unsigned>(cluster), 1};
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(rows, cluster);
    config.blockDim = dim3(packs_per_row / cluster);
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;
    return rank == 0 ? cudaLaunchKernelEx(&config, oneshot_allreduce<0>, args)
                     : cudaLaunchKernelEx(&config, oneshot_allreduce<1>, args);
}

cudaError_t launch_twoshot(AllReduceArgs args, int rank, cudaStream_t stream) {
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(kTwoshotRows, kTwoshotBlocksPerRow);
    config.blockDim = dim3(kTwoshotThreads);
    config.stream = stream;
    return cudaLaunchKernelEx(&config, twoshot_allreduce, args, rank);
}

std::uint64_t round_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

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

using namespace text_kernel;

static_assert(kTokenDim == rms_hidden_width);
constexpr int packs_per_thread = (packs_per_row + block_threads - 1) / block_threads;

template <typename Rows>
struct ReduceNormArgs {
    Bf16* branch;
    const Bf16* gain;
    void* local;
    void* multicast;
    LamportState* flags;
    Rows rows;
};

template <typename Rows>
struct ReduceNorm {
    ReduceNormArgs<Rows> args;
};

using PairAtomic = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system>;

template <int Width> struct ReduceNorm<MnnvlWideRmsQuantizeRows<Width>> {
    ReduceNormArgs<MnnvlWideRmsQuantizeRows<Width>> args;
    std::uint64_t* rejected;
    std::uint64_t epoch;
};

__device__ bool poll_wide_rank0(const uint4* address, PackedBf16& remote, std::uint64_t* rejected,
                                std::uint64_t epoch) {
    unsigned misses = 0;
    while (!poll_pack(address, remote)) {
        if ((++misses & 255) == 0 &&
            PairAtomic(*rejected).load(cuda::memory_order_acquire) == epoch) return false;
    }
    return true;
}

template <int Rank, typename Rows>
__global__ __launch_bounds__(block_threads) void reduce_norm_rows(
    const __grid_constant__ ReduceNorm<Rows> params) {
    constexpr bool can_cancel = Rank == 0 && !std::is_same_v<Rows, MnnvlRmsQuantizeRows>;
    const auto& args = params.args;
    const int row = blockIdx.x;
    const auto& output = args.rows.entries[row];
    auto* branch = args.branch + row * kTokenDim;
    __shared__ alignas(16) Bf16 branch_values[kTokenDim];
    __shared__ alignas(16) Bf16 residual_values[kTokenDim];
    __shared__ alignas(16) Bf16 gains[kTokenDim];

    LamportBuffer<1> flags(args.flags);
    auto* const multicast = flags.current(args.multicast);
    const auto* const peer = flags.current(args.local) + (row * kRanks + 1 - Rank) * packs_per_row;
    PackedBf16 values[packs_per_thread];
#pragma unroll
    for (int index = 0; index < packs_per_thread; ++index) {
        const int packed = threadIdx.x + index * block_threads;
        if (packed < packs_per_row) {
            auto& value = values[index];
            value.packed = reinterpret_cast<const uint4*>(branch)[packed];
            sanitize_payload(value);
            store_multicast_pack(multicast + (row * kRanks + Rank) * packs_per_row + packed,
                                 value.packed);
            stage_rms_vector_async(output.residual + packed * 8, residual_values + packed * 8);
            stage_rms_vector_async(args.gain + packed * 8, gains + packed * 8);
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    flags.arrive();
    flags.clear(args.local, 0);
    flags.clear(args.local, 1);
    bool cancelled = false;
#pragma unroll
    for (int index = 0; index < packs_per_thread; ++index) {
        const int packed = threadIdx.x + index * block_threads;
        if (packed < packs_per_row) {
            PackedBf16 remote;
            if constexpr (can_cancel) {
                if (cancelled ||
                    !poll_wide_rank0(peer + packed, remote, params.rejected, params.epoch)) {
                    cancelled = true;
                    continue;
                }
            } else {
                while (!poll_pack(peer + packed, remote)) {}
            }
            const auto reduced = reduce_local_remote<Rank>(values[index], remote);
            reinterpret_cast<uint4*>(branch)[packed] = reduced.packed;
            reinterpret_cast<uint4*>(branch_values)[packed] = reduced.packed;
        }
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    if constexpr (can_cancel) {
        if (__syncthreads_or(cancelled)) return;
    } else {
        __syncthreads();
    }
    rms_quantize_staged<true, true>(branch_values, residual_values, gains, output.residual,
                                    output.normalized, output.activation);
    flags.rotate(gridDim.x, {gridDim.x * kRanks * kRowBytes, 0});
}

template <typename Rows>
constexpr auto reduce_kernels = std::array{reduce_norm_rows<0, Rows>, reduce_norm_rows<1, Rows>};

template <typename Rows>
bool reduce_norm_ready(int device, int rank) {
    cudaFuncAttributes attributes{};
    int resident = 0;
    int multiprocessors = 0;
    const auto kernel = reduce_kernels<Rows>[rank];
    if (cudaFuncGetAttributes(&attributes, kernel) != cudaSuccess ||
        attributes.maxThreadsPerBlock < block_threads ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, block_threads, 0) !=
            cudaSuccess ||
        cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device) !=
            cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return resident * multiprocessors >= Rows::size;
}

}   

struct MnnvlAllReduce::State {
    std::vector<int> devices;   
    bool failed = false;
    bool twoshot_supported = false;
    bool reduce_norm_supported = true;
    bool reduce_norm_m40_supported = true;
    bool reduce_norm_m80_supported = true;
    std::uint64_t* wide_rejected = nullptr;
    std::uint64_t* wide_device_word = nullptr;
    std::uint64_t wide_epoch = 0;

    CUmemGenericAllocationHandle physical[kRanks]{};
    CUmemGenericAllocationHandle multicast{};
    CUdeviceptr uc_base = 0;           
    std::uint64_t uc_ptrs[kRanks]{};   
    CUdeviceptr mc_va = 0;             
    std::uint64_t allocation_size = 0;
    std::uint32_t buffer_size_bytes = 0;   
    std::uint64_t max_elements = 0;        

    LamportState* flags[kRanks]{};
    int multiprocessors[kRanks]{};

    bool created_physical[kRanks] = {false, false};
    bool created_multicast = false;
    bool bound_multicast[kRanks] = {false, false};
    bool reserved_uc = false;
    bool mapped_uc[kRanks] = {false, false};
    bool reserved_mc = false;
    bool mapped_mc = false;

    bool map_wide_submission();
    ~State();

    void release();
};

MnnvlArError::MnnvlArError(const std::string& what, bool peer_enqueued_arg)
    : std::runtime_error(what), peer_enqueued(peer_enqueued_arg) {}

MnnvlAllReduce::MnnvlAllReduce(std::unique_ptr<State> state) : state_(std::move(state)) {}

int MnnvlAllReduce::rank_count() const noexcept { return kRanks; }

int MnnvlAllReduce::device(int rank) const { return state_->devices.at(rank); }

std::size_t MnnvlAllReduce::max_count() const noexcept { return state_->max_elements; }

bool MnnvlAllReduce::supports_twoshot(std::size_t count) const noexcept {
    return count == kTwoshotCount && state_->twoshot_supported;
}

bool MnnvlAllReduce::healthy() const noexcept { return !state_->failed; }

void MnnvlAllReduce::ensure_healthy() const {
    if (state_->failed) {
        throw std::runtime_error("An MNNVL allreduce failed earlier.");
    }
}

MnnvlAllReduce::State::~State() {
    release();
}

bool MnnvlAllReduce::State::map_wide_submission() {
    void* memory = nullptr;
    constexpr unsigned flags = cudaHostAllocMapped | cudaHostAllocPortable;
    if (cudaHostAlloc(&memory, sizeof(std::uint64_t), flags) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    wide_rejected = new (memory) std::uint64_t{0};
    CallerDevice device_scope;
    void* device_word = nullptr;
    if (cudaSetDevice(devices[0]) != cudaSuccess ||
        cudaHostGetDevicePointer(&device_word, memory, 0) != cudaSuccess) {
        (void)cudaGetLastError();
        (void)cudaFreeHost(memory);
        wide_rejected = nullptr;
        return false;
    }
    wide_device_word = static_cast<std::uint64_t*>(device_word);
    return true;
}

void MnnvlAllReduce::State::release() {
    CallerDevice device_scope;
    for (int r = 0; r < kRanks; ++r) {
        if (flags[r] != nullptr) {
            cudaSetDevice(devices[r]);
            cudaFree(flags[r]);
            flags[r] = nullptr;
        }
    }
    if (wide_rejected != nullptr) {
        (void)cudaFreeHost(wide_rejected);
        wide_rejected = nullptr;
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
    mc_prop.handleTypes = 0;   
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

void MnnvlAllReduce::create_physical(State& state) {
    CallerDevice device_scope;
    CUmemAllocationProp alloc_prop{};
    alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    alloc_prop.allocFlags.gpuDirectRDMACapable = 1;

    CUmulticastObjectProp mc_prop{};
    mc_prop.numDevices = kRanks;
    mc_prop.size = state.allocation_size;
    mc_prop.handleTypes = 0;   
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

    require_cu(cuMemAddressReserve(&state.mc_va, state.allocation_size, mc_granularity, 0, 0),
               "Reserve the multicast range");
    state.reserved_mc = true;
    require_cu(cuMemMap(state.mc_va, state.allocation_size, 0, state.multicast, 0),
               "Map the multicast range");
    state.mapped_mc = true;
    require_cu(cuMemSetAccess(state.mc_va, state.allocation_size, access, kRanks),
               "Grant multicast access");
}

void MnnvlAllReduce::init_lamport(State& state) {
    const std::uint64_t usable = state.allocation_size - kSignalPad;
    const std::uint64_t per_buffer =
        usable / kNumLamportBuffers / kBufferAlignment * kBufferAlignment;
    if (per_buffer > 0xFFFFFFFFull) {
        throw std::runtime_error("The multicast workspace exceeds the Lamport flag width.");
    }
    state.buffer_size_bytes = static_cast<std::uint32_t>(per_buffer);
    state.twoshot_supported = per_buffer >= kTwoshotBufferBytes;

    CallerDevice device_scope;
    for (int r = 0; r < kRanks; ++r) {
        device_scope.select(state.devices[r], "Select a rank device");
        cudaError_t status = cudaMalloc(&state.flags[r], sizeof(LamportState));
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("Allocate the Lamport flags: ") +
                                     cudaGetErrorString(status));
        }
        const LamportState flag_host{{0, 2, state.buffer_size_bytes, 0}, {0, 0}, 0};
        status = cudaMemcpy(state.flags[r], &flag_host, sizeof(flag_host), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("Initialize the Lamport flags: ") +
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
        status = cudaDeviceGetAttribute(&state.multiprocessors[r],
                                        cudaDevAttrMultiProcessorCount, state.devices[r]);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("Query the MNNVL device size: ") +
                                     cudaGetErrorString(status));
        }
        state.twoshot_supported &= twoshot_grid_fits(state.multiprocessors[r]);
        state.reduce_norm_supported &= reduce_norm_ready<MnnvlRmsQuantizeRows>(state.devices[r], r);
        state.reduce_norm_m40_supported &=
            reduce_norm_ready<MnnvlWideRmsQuantizeRows<40>>(state.devices[r], r);
        state.reduce_norm_m80_supported &=
            reduce_norm_ready<MnnvlWideRmsQuantizeRows<80>>(state.devices[r], r);
    }
    if ((state.reduce_norm_m40_supported || state.reduce_norm_m80_supported) &&
        !state.map_wide_submission()) {
        state.reduce_norm_m40_supported = false;
        state.reduce_norm_m80_supported = false;
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
    const char* reason = "no_vmm";
    try {
        reason = "device_unsupported";
        probe_devices(*state);

        reason = "multicast_create_failed";
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

cudaStreamCaptureStatus MnnvlAllReduce::preflight(State& state, __nv_bfloat16* const buffers[],
                                                  const cudaStream_t streams[], bool two_shot) {
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
    if (two_shot && capture_status[0] != cudaStreamCaptureStatusNone) {
        throw std::invalid_argument("The two-shot MNNVL allreduce requires eager streams.");
    }
    return capture_status[0];
}

void MnnvlAllReduce::allreduce_bf16(__nv_bfloat16* const buffers[], const cudaStream_t streams[],
                                    std::size_t count, bool two_shot) {
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
    if (!two_shot && count > state_->max_elements) {
        throw std::invalid_argument(
            "MnnvlAllReduce::allreduce_bf16 requires a count within the one-shot envelope.");
    }
    if (two_shot && !supports_twoshot(count)) {
        throw std::invalid_argument("Unsupported two-shot MNNVL allreduce count or device.");
    }
    ensure_healthy();
    preflight(*state_, buffers, streams, two_shot);

    const int tokens = static_cast<int>(count / kTokenDim);
    CallerDevice device_scope;
    for (int r = 0; r < kRanks; ++r) {
        device_scope.select(state_->devices[r], "Select a rank device");
        const AllReduceArgs args{reinterpret_cast<uint4*>(buffers[r]),
                                 reinterpret_cast<void*>(state_->uc_ptrs[r]),
                                 reinterpret_cast<void*>(state_->uc_ptrs[1 - r]),
                                 reinterpret_cast<void*>(state_->mc_va), state_->flags[r]};
        const cudaError_t status = two_shot
            ? launch_twoshot(args, r, streams[r])
            : launch_oneshot(args, r, tokens, state_->multiprocessors[r], streams[r]);
        if (status != cudaSuccess) {
            state_->failed = true;
            throw MnnvlArError(std::string("Enqueue the MNNVL BF16 allreduce: ") +
                                   cudaGetErrorString(status),
                               r > 0);
        }
    }
}

bool MnnvlAllReduce::try_allreduce_rms_quantize(
    Bf16* const buffers[], const cudaStream_t streams[],
    const MnnvlRmsQuantizeRows rows[], const Bf16* const gains[]) {
    const int width = rows[0].count;
    if (!MnnvlRmsQuantizeRows::supports(width) || rows[1].count != width) return false;
    return try_reduce_norm(buffers, streams, rows, gains, width);
}

template <int Width>
bool MnnvlAllReduce::try_allreduce_rms_quantize_wide(Bf16* const buffers[],
                                                     const cudaStream_t streams[],
                                                     const MnnvlWideRmsQuantizeRows<Width> rows[],
                                                     const Bf16* const gains[]) {
    return try_reduce_norm(buffers, streams, rows, gains, Width);
}

template bool MnnvlAllReduce::try_allreduce_rms_quantize_wide<40>(
    Bf16* const[], const cudaStream_t[], const MnnvlWideRmsQuantizeRows<40>[], const Bf16* const[]);
template bool MnnvlAllReduce::try_allreduce_rms_quantize_wide<80>(
    Bf16* const[], const cudaStream_t[], const MnnvlWideRmsQuantizeRows<80>[], const Bf16* const[]);

template <typename Rows>
bool MnnvlAllReduce::try_reduce_norm(Bf16* const buffers[], const cudaStream_t streams[],
                                    const Rows rows[], const Bf16* const gains[], int width) {
    constexpr bool narrow = std::is_same_v<Rows, MnnvlRmsQuantizeRows>;
    if constexpr (Rows::size == kTwoshotRows) {
        if (!state_->reduce_norm_m80_supported || state_->buffer_size_bytes < kTwoshotBufferBytes) {
            return false;
        }
    } else {
        const bool supported =
            narrow ? state_->reduce_norm_supported : state_->reduce_norm_m40_supported;
        if (!supported || static_cast<std::size_t>(width) * kTokenDim > state_->max_elements) {
            return false;
        }
    }
    for (int rank = 0; rank < kRanks; ++rank) {
        if (!buffers[rank] || !streams[rank] || !gains[rank] ||
            !aligned(buffers[rank], sizeof(float4)) || !aligned(gains[rank], sizeof(float4))) {
            return false;
        }
        for (int index = 0; index < width; ++index) {
            const auto& row = rows[rank].entries[index];
            if (!row.residual || !row.normalized || !row.activation.values ||
                !row.activation.scales ||
                !aligned(row.residual, sizeof(float4)) || !aligned(row.normalized, alignof(Bf16)) ||
                !aligned(row.activation.values, alignof(uint2)) ||
                !aligned(row.activation.scales, alignof(float))) {
                return false;
            }
        }
    }
    ensure_healthy();
    if (preflight(*state_, buffers, streams, false) != cudaStreamCaptureStatusNone) {
        if constexpr (!narrow) return false;
        if (width != tp_verify_graph::rows || !tp_verify_graph::binding(streams[0]) ||
            !tp_verify_graph::binding(streams[1])) return false;
    }
    std::uint64_t epoch = 0;
    if constexpr (!narrow) {
        if (state_->wide_epoch == UINT64_MAX)
            return false;
        epoch = ++state_->wide_epoch;
    }

    ReduceNorm<Rows> launch[kRanks];
    for (int rank = 0; rank < kRanks; ++rank) {
        launch[rank].args = {buffers[rank], gains[rank],
                             reinterpret_cast<void*>(state_->uc_ptrs[rank]),
                             reinterpret_cast<void*>(state_->mc_va), state_->flags[rank],
                             rows[rank]};
        if constexpr (!narrow) {
            launch[rank].rejected = state_->wide_device_word;
            launch[rank].epoch = epoch;
        }
    }

    CallerDevice device_scope;
    for (int rank = 0; rank < kRanks; ++rank) {
        cudaError_t status = cudaSetDevice(state_->devices[rank]);
        if (status == cudaSuccess) {
            void* args[]{&launch[rank]};
            status = cudaLaunchKernel(reduce_kernels<Rows>[rank], width, block_threads, args, 0,
                                      streams[rank]);
        }
        if (status != cudaSuccess) {
            state_->failed = true;
            if constexpr (!narrow) {
                if (rank > 0) {
                    PairAtomic(*state_->wide_rejected).store(epoch, cuda::memory_order_release);
                    for (int peer = 0; peer < kRanks; ++peer) {
                        (void)cudaStreamSynchronize(streams[peer]);
                    }
                }
            }
            throw MnnvlArError(
                std::string("Enqueue the MNNVL BF16 reduction and RMS quantization: ") +
                    cudaGetErrorString(status), rank > 0);
        }
    }
    return true;
}

}   

namespace sglang::decoder {

bool MnnvlRmsQuantizeRows::supports(int count) {
    return count == size || (count >= 3 && count <= 5);
}

}
