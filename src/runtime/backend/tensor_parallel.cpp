#include "sglang/runtime/backend/tensor_parallel.h"

#include <nccl.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sglang/runtime/backend/cuda_resources.h"

namespace sglang::decoder {
namespace {

void check_nccl(ncclResult_t status, const char* operation) {
    if (status != ncclSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + ncclGetErrorString(status));
    }
}

// Preserves the caller's current CUDA device across the scope. Selection
// failures throw; restoration is best effort because the scope may unwind.
class CallerDevice {
public:
    CallerDevice() { check(cudaGetDevice(&previous_), "Read the current CUDA device"); }
    ~CallerDevice() { (void)cudaSetDevice(previous_); }
    CallerDevice(const CallerDevice&) = delete;
    CallerDevice& operator=(const CallerDevice&) = delete;

    void select(int device, const char* operation) { check(cudaSetDevice(device), operation); }

private:
    int previous_ = 0;
};

// NCCL 2.9 added CUDA-graph capture support for collectives.
#if defined(NCCL_MAJOR) && defined(NCCL_MINOR)
constexpr bool nccl_supports_cuda_graphs = NCCL_MAJOR > 2 || (NCCL_MAJOR == 2 && NCCL_MINOR >= 9);
#else
constexpr bool nccl_supports_cuda_graphs = false;
#endif

}  // namespace

struct TensorParallel::State {
    std::vector<int> devices;
    std::vector<ncclComm_t> comms;
    // A host-side enqueue failed or a communicator reported an asynchronous
    // error: the communicators must abort, not destroy, on shutdown.
    bool failed = false;

    // Best effort and never throws: this runs while the owning thread may be
    // unwinding. The caller has already fenced all enqueued collectives.
    ~State() {
        int restore = -1;
        if (cudaGetDevice(&restore) != cudaSuccess) restore = -1;
        for (std::size_t rank = 0; rank < comms.size(); ++rank) {
            if (comms[rank] == nullptr) continue;
            (void)cudaSetDevice(devices[rank]);
            ncclResult_t asynchronous = ncclSuccess;
            const ncclResult_t polled = ncclCommGetAsyncError(comms[rank], &asynchronous);
            if (failed || polled != ncclSuccess || asynchronous != ncclSuccess) {
                (void)ncclCommAbort(comms[rank]);
            } else {
                (void)ncclCommDestroy(comms[rank]);
            }
        }
        if (restore >= 0) (void)cudaSetDevice(restore);
    }
};

std::unique_ptr<TensorParallel> TensorParallel::create(std::vector<int> devices) {
    return std::unique_ptr<TensorParallel>(new TensorParallel(std::move(devices)));
}

TensorParallel::TensorParallel(std::vector<int> devices) : state_(new State) {
    state_->devices = std::move(devices);
    const std::size_t ranks = state_->devices.size();
    if (ranks == 0) {
        throw std::invalid_argument("TensorParallel requires at least one device.");
    }
    for (std::size_t left = 0; left < ranks; ++left) {
        for (std::size_t right = left + 1; right < ranks; ++right) {
            if (state_->devices[left] == state_->devices[right]) {
                throw std::invalid_argument("TensorParallel requires distinct devices.");
            }
        }
    }

    // Validate every ordinal before entering the NCCL rendezvous: a rank
    // rejected inside the group would leave the other ranks waiting for it in
    // ncclGroupEnd.
    {
        CallerDevice validation_scope;
        for (const int device : state_->devices) {
            validation_scope.select(device, "Select a tensor-parallel device");
        }
    }

    ncclUniqueId id{};
    check_nccl(ncclGetUniqueId(&id), "Create an NCCL communicator id");
    state_->comms.assign(ranks, nullptr);

    // All ranks live in this process, so every communicator must be created
    // inside one NCCL group; a grouped init from a single host thread is the
    // documented rendezvous for multiple per-device communicators.
    check_nccl(ncclGroupStart(), "Open the NCCL communicator group");
    ncclResult_t first = ncclSuccess;
    try {
        CallerDevice device_scope;
        for (std::size_t rank = 0; rank < ranks; ++rank) {
            device_scope.select(state_->devices[rank], "Select a tensor-parallel device");
            const ncclResult_t status =
                ncclCommInitRank(&state_->comms[rank], static_cast<int>(ranks), id,
                                 static_cast<int>(rank));
            if (status != ncclSuccess && first == ncclSuccess) first = status;
        }
    } catch (...) {
        (void)ncclGroupEnd();
        state_->failed = true;
        throw;
    }
    const ncclResult_t grouped = ncclGroupEnd();
    if (first != ncclSuccess || grouped != ncclSuccess) {
        state_->failed = true;
        throw std::runtime_error(std::string("Initialize NCCL tensor-parallel communicators: ") +
                                 ncclGetErrorString(first != ncclSuccess ? first : grouped));
    }
}

TensorParallel::~TensorParallel() = default;

int TensorParallel::rank_count() const noexcept { return static_cast<int>(state_->devices.size()); }

int TensorParallel::device(int rank) const {
    if (rank < 0 || static_cast<std::size_t>(rank) >= state_->devices.size()) {
        throw std::invalid_argument("TensorParallel::device requires a rank of the group.");
    }
    return state_->devices[static_cast<std::size_t>(rank)];
}

void TensorParallel::ensure_healthy() {
    if (state_->failed) {
        throw std::runtime_error("A tensor-parallel collective failed earlier.");
    }
    for (ncclComm_t comm : state_->comms) {
        ncclResult_t asynchronous = ncclSuccess;
        const ncclResult_t polled = ncclCommGetAsyncError(comm, &asynchronous);
        if (polled != ncclSuccess || asynchronous != ncclSuccess) {
            state_->failed = true;
            throw std::runtime_error(
                std::string("A tensor-parallel collective failed asynchronously: ") +
                ncclGetErrorString(polled != ncclSuccess ? polled : asynchronous));
        }
    }
}

void TensorParallel::allreduce_bf16(__nv_bfloat16* const buffers[], const cudaStream_t streams[],
                                    std::size_t count) {
    const std::size_t ranks = state_->devices.size();
    if (count == 0) {
        throw std::invalid_argument(
            "TensorParallel::allreduce_bf16 requires a positive element count.");
    }
    for (std::size_t rank = 0; rank < ranks; ++rank) {
        if (buffers[rank] == nullptr || streams[rank] == nullptr) {
            throw std::invalid_argument(
                "TensorParallel::allreduce_bf16 requires a buffer and a stream for every rank.");
        }
    }
    ensure_healthy();

    // Group the per-rank enqueues: collectives on multiple communicators from
    // one host thread must be grouped or the thread can block inside an
    // enqueue waiting for the other communicator's progress.
    check_nccl(ncclGroupStart(), "Open the NCCL operation group");
    try {
        CallerDevice device_scope;
        for (std::size_t rank = 0; rank < ranks; ++rank) {
            device_scope.select(state_->devices[rank], "Select a tensor-parallel device");
            const ncclResult_t status =
                ncclAllReduce(buffers[rank], buffers[rank], count, ncclBfloat16, ncclSum,
                              state_->comms[rank], streams[rank]);
            if (status != ncclSuccess) {
                throw std::runtime_error(std::string("Enqueue the NCCL BF16 allreduce: ") +
                                         ncclGetErrorString(status));
            }
        }
    } catch (...) {
        (void)ncclGroupEnd();
        state_->failed = true;
        throw;
    }
    const ncclResult_t grouped = ncclGroupEnd();
    if (grouped != ncclSuccess) {
        state_->failed = true;
        throw std::runtime_error(std::string("Close the NCCL operation group: ") +
                                 ncclGetErrorString(grouped));
    }
}

void TensorParallel::broadcast_int32(std::int32_t* const buffers[], const cudaStream_t streams[],
                                     std::size_t count, int root) {
    const std::size_t ranks = state_->devices.size();
    if (count == 0) {
        throw std::invalid_argument(
            "TensorParallel::broadcast_int32 requires a positive element count.");
    }
    if (root < 0 || static_cast<std::size_t>(root) >= ranks) {
        throw std::invalid_argument(
            "TensorParallel::broadcast_int32 requires a rank of the group as root.");
    }
    for (std::size_t rank = 0; rank < ranks; ++rank) {
        if (buffers[rank] == nullptr || streams[rank] == nullptr) {
            throw std::invalid_argument(
                "TensorParallel::broadcast_int32 requires a buffer and a stream for every rank.");
        }
    }
    ensure_healthy();

    // Same grouping rule as allreduce_bf16: multiple communicators, one thread.
    check_nccl(ncclGroupStart(), "Open the NCCL operation group");
    try {
        CallerDevice device_scope;
        for (std::size_t rank = 0; rank < ranks; ++rank) {
            device_scope.select(state_->devices[rank], "Select a tensor-parallel device");
            const ncclResult_t status =
                ncclBroadcast(buffers[rank], buffers[rank], count, ncclInt32, root,
                              state_->comms[rank], streams[rank]);
            if (status != ncclSuccess) {
                throw std::runtime_error(std::string("Enqueue the NCCL int32 broadcast: ") +
                                         ncclGetErrorString(status));
            }
        }
    } catch (...) {
        (void)ncclGroupEnd();
        state_->failed = true;
        throw;
    }
    const ncclResult_t grouped = ncclGroupEnd();
    if (grouped != ncclSuccess) {
        state_->failed = true;
        throw std::runtime_error(std::string("Close the NCCL operation group: ") +
                                 ncclGetErrorString(grouped));
    }
}

bool TensorParallel::graph_eligible() const noexcept { return nccl_supports_cuda_graphs; }

}  // namespace sglang::decoder
