#include "sglang/runtime/backend/tensor_parallel.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace {

using sglang::decoder::TensorParallel;
using sglang::decoder::TpArCounters;
using sglang::decoder::TpArRankCounters;
using RankCounter = decltype(&TpArRankCounters::ar_calls_total);
constexpr int ranks = 2;
constexpr int row = 5120;
constexpr int wide = 80 * row;
constexpr int capacity = 81 * row + 8;
constexpr auto bytes = capacity * sizeof(__nv_bfloat16);

bool& nccl_only() {
    static bool enabled = false;
    return enabled;
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

void require(bool condition, const char* operation) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", operation);
        std::exit(1);
    }
}

void check_caller(int caller, const char* operation) {
    int actual = -1;
    check(cudaGetDevice(&actual), "cudaGetDevice");
    require(actual == caller, operation);
}

float input(int rank, int index, int generation) {
    if (index % 29 == 0) return -0.0f;
    if (index % 29 == 1) return rank == 0 ? 256.0f : 1.0f;
    if (rank == 0) return static_cast<float>(index % 23 - 13 + generation);
    return static_cast<float>(2 * (index % 17) + 35 - 3 * generation);
}

std::uint16_t bits(__nv_bfloat16 value) {
    std::uint16_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

TpArCounters snapshot(const TensorParallel& group) {
    TpArCounters result;
    group.snapshot_transport(result);
    require(result.ranks == ranks, "unexpected counter rank count");
    return result;
}

void expect_route(const TensorParallel& group, const TpArCounters& before, RankCounter transport,
                  RankCounter fallback = nullptr) {
    const auto after = snapshot(group);
    for (int r = 0; r < ranks; ++r) {
        const auto& old = before.per_rank[r];
        const auto& now = after.per_rank[r];
        require(now.ar_calls_total == old.ar_calls_total + 1 &&
                    now.*transport == old.*transport + 1 &&
                    now.ar_calls_total == now.ar_nccl_total + now.ar_mnnvl_oneshot_total +
                                              now.ar_mnnvl_twoshot_total,
                "unexpected collective transport");
        require(!fallback || now.*fallback == old.*fallback + 1, "unexpected fallback reason");
    }
}

constexpr auto one_shot = &TpArRankCounters::ar_mnnvl_oneshot_total;
constexpr auto two_shot = &TpArRankCounters::ar_mnnvl_twoshot_total;
constexpr auto nccl = &TpArRankCounters::ar_nccl_total;
constexpr auto oversize = &TpArRankCounters::ar_fallback_oversize_total;

struct Fixture {
    const std::array<int, ranks> devices;
    cudaStream_t streams[ranks]{};
    __nv_bfloat16* buffers[ranks]{};
    __nv_bfloat16* host[ranks]{};
    cudaGraph_t graphs[ranks]{};
    cudaGraphExec_t execs[ranks]{};
    cudaEvent_t fork{};
    cudaEvent_t join{};

    explicit Fixture(const TensorParallel& group) : devices{group.device(0), group.device(1)} {
        for (int r = 0; r < ranks; ++r) {
            check(cudaSetDevice(devices[r]), "select allocation device");
            check(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking), "create stream");
            check(cudaMalloc(&buffers[r], bytes), "allocate device rows");
            check(cudaMallocHost(&host[r], bytes), "allocate pinned staging");
        }
        check(cudaSetDevice(devices[0]), "select fork device");
        check(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming), "create capture fork");
        check(cudaSetDevice(devices[1]), "select join device");
        check(cudaEventCreateWithFlags(&join, cudaEventDisableTiming), "create capture join");
    }

    void upload(int generation) {
        for (int r = 0; r < ranks; ++r) {
            for (int i = 0; i < capacity; ++i) {
                host[r][i] = __float2bfloat16_rn(input(r, i, generation));
            }
            check(cudaSetDevice(devices[r]), "select input device");
            check(cudaMemcpyAsync(buffers[r], host[r], bytes, cudaMemcpyHostToDevice, streams[r]),
                  "upload rank input");
        }
    }

    void verify(int generation, int count, RankCounter transport, int offset = 0) {
        for (int r = 0; r < ranks; ++r) {
            check(cudaSetDevice(devices[r]), "select result device");
            check(cudaMemcpyAsync(host[r], buffers[r], bytes, cudaMemcpyDeviceToHost, streams[r]),
                  "download rank result");
        }
        for (int r = 0; r < ranks; ++r) {
            check(cudaSetDevice(devices[r]), "select completion device");
            check(cudaStreamSynchronize(streams[r]), "complete result copy");
            for (int i = 0; i < capacity; ++i) {
                const bool active = i >= offset && i < offset + count;
                const float sum = input(0, i, generation) + input(1, i, generation);
                const float expected = active ? sum : input(r, i, generation);
                const auto want = bits(__float2bfloat16_rn(expected));
                const auto actual = bits(host[r][i]);
                if (active && expected == 0.0f) {
                    if (actual == 0 || (transport == nccl && actual == 0x8000)) continue;
                } else if (actual == want) {
                    continue;
                }
                std::fprintf(stderr, "generation %d rank %d index %d: expected %04x, got %04x\n",
                             generation, r, i, static_cast<unsigned>(want),
                             static_cast<unsigned>(actual));
                std::exit(1);
            }
        }
    }

    bool reduce(TensorParallel& group, int count, int offset = 0) {
        check(cudaSetDevice(devices[0]), "select collective caller");
        __nv_bfloat16* shifted[ranks]{buffers[0] + offset, buffers[1] + offset};
        const auto route = group.allreduce_bf16(shifted, streams, count);
        check_caller(devices[0], "allreduce changed the caller device");
        return route == sglang::decoder::TpAllReduceRoute::mnnvl_two_shot;
    }

    void capture(TensorParallel& group, int count) {
        for (int r = 0; r < ranks; ++r) {
            check(cudaSetDevice(devices[r]), "select capture device");
            check(cudaStreamBeginCapture(streams[r], cudaStreamCaptureModeThreadLocal),
                  "begin rank stream capture");
        }
        require(!reduce(group, count), "captured split submission certified eager two-shot");
        for (int r = ranks - 1; r >= 0; --r) {
            check(cudaSetDevice(devices[r]), "select graph device");
            check(cudaStreamEndCapture(streams[r], &graphs[r]), "end rank stream capture");
        }
        for (int r = 0; r < ranks; ++r) {
            check(cudaSetDevice(devices[r]), "select graph execution device");
            check(cudaGraphInstantiateWithFlags(&execs[r], graphs[r], 0), "instantiate rank graph");
        }
    }

    void capture_joined(TensorParallel& group, int count) {
        check(cudaSetDevice(devices[0]), "select capture root");
        check(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeThreadLocal),
              "begin joined capture");
        check(cudaEventRecord(fork, streams[0]), "record capture fork");
        check(cudaSetDevice(devices[1]), "select capture peer");
        check(cudaStreamWaitEvent(streams[1], fork), "join peer to capture");
        require(!reduce(group, count), "captured joined submission certified eager two-shot");
        check(cudaSetDevice(devices[1]), "select capture peer after reduce");
        check(cudaEventRecord(join, streams[1]), "record capture join");
        check(cudaSetDevice(devices[0]), "select capture root after reduce");
        check(cudaStreamWaitEvent(streams[0], join), "finish peer capture");
        check(cudaStreamEndCapture(streams[0], &graphs[0]), "end joined capture");
        check(cudaGraphInstantiateWithFlags(&execs[0], graphs[0], 0), "instantiate joined graph");
    }

    void reject_mixed_capture(TensorParallel& group) {
        const auto before = snapshot(group);
        check(cudaSetDevice(devices[0]), "select mixed capture rank");
        check(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeThreadLocal),
              "begin capture on one rank");
        try {
            reduce(group, wide);
            require(false, "accepted mixed eager and captured streams");
        } catch (const std::invalid_argument&) {
            check_caller(devices[0], "mixed capture rejection changed the caller device");
        }
        cudaGraph_t empty{};
        check(cudaStreamEndCapture(streams[0], &empty), "end rejected capture");
        check(cudaGraphDestroy(empty), "destroy rejected graph");
        const auto after = snapshot(group);
        for (int r = 0; r < ranks; ++r) {
            require(after.per_rank[r].ar_calls_total == before.per_rank[r].ar_calls_total,
                    "mixed capture enqueued a collective");
        }
    }

    void replay() {
        for (int r = 0; r < ranks; ++r) {
            if (!execs[r]) continue;
            check(cudaSetDevice(devices[r]), "select replay device");
            check(cudaGraphLaunch(execs[r], streams[r]), "launch rank graph");
        }
    }

    ~Fixture() {
        check(cudaSetDevice(devices[0]), "select fork teardown device");
        check(cudaEventDestroy(fork), "destroy capture fork");
        check(cudaSetDevice(devices[1]), "select join teardown device");
        check(cudaEventDestroy(join), "destroy capture join");
        for (int r = 0; r < ranks; ++r) {
            check(cudaSetDevice(devices[r]), "select teardown device");
            if (execs[r]) check(cudaGraphExecDestroy(execs[r]), "destroy executable graph");
            if (graphs[r]) check(cudaGraphDestroy(graphs[r]), "destroy captured graph");
            check(cudaStreamDestroy(streams[r]), "destroy rank stream");
            check(cudaFree(buffers[r]), "free rank buffer");
            check(cudaFreeHost(host[r]), "free pinned staging");
        }
    }
};

void check_eager(TensorParallel& group, bool vendor, RankCounter unavailable) {
    struct Case { int count; int offset; RankCounter transport; RankCounter fallback; };
    const std::array<Case, 9> cases{{
        {row, 0, one_shot, nullptr}, {51 * row, 0, one_shot, nullptr},
        {52 * row, 0, nccl, oversize}, {wide, 0, two_shot, nullptr},
        {wide, 0, two_shot, nullptr}, {79 * row, 0, nccl, oversize},
        {81 * row, 0, nccl, oversize}, {wide - 1, 0, nccl, oversize},
        {wide, 1, nccl, &TpArRankCounters::ar_fallback_unaligned_total},
    }};
    Fixture fixture(group);
    int generation = 0;
    for (int epoch = 0; epoch < 3; ++epoch) {
        for (const auto& test : cases) {
            const auto transport = vendor ? test.transport : nccl;
            const auto before = snapshot(group);
            fixture.upload(generation);
            require(fixture.reduce(group, test.count, test.offset) == (transport == two_shot),
                    "eager two-shot certificate disagrees with submitted transport");
            expect_route(group, before, transport, vendor ? test.fallback : unavailable);
            fixture.verify(generation++, test.count, transport, test.offset);
        }
    }
    fixture.reject_mixed_capture(group);
    std::puts("eager: 27 exact BF16 sums, signed zero, guard regions and width routes");
}

void check_capture(TensorParallel& group, bool joined, int count, RankCounter transport,
                   RankCounter fallback) {
    Fixture fixture(group);
    const auto before = snapshot(group);
    if (joined) fixture.capture_joined(group, count);
    else fixture.capture(group, count);
    expect_route(group, before, transport, fallback);
    for (int generation = 1; generation <= 4; ++generation) {
        fixture.upload(generation);
        if (joined) {
            for (int r = 0; r < ranks; ++r) {
                check(cudaSetDevice(fixture.devices[r]), "select joined replay input device");
                check(cudaStreamSynchronize(fixture.streams[r]), "finish joined replay input");
            }
        }
        fixture.replay();
        if (joined) {
            check(cudaStreamSynchronize(fixture.streams[0]), "finish joined replay");
        }
        fixture.verify(generation, count, transport);
    }
    expect_route(group, before, transport, fallback);
}

}

extern "C" decltype(cuMulticastGetGranularity)
    real_multicast_granularity asm("__real_cuMulticastGetGranularity");
extern "C" decltype(cuMulticastGetGranularity)
    wrapped_multicast_granularity asm("__wrap_cuMulticastGetGranularity");

CUresult wrapped_multicast_granularity(std::size_t* granularity,
                                       const CUmulticastObjectProp* properties,
                                       CUmulticastGranularity_flags option) {
    if (nccl_only())
        return CUDA_ERROR_NOT_SUPPORTED;
    return real_multicast_granularity(granularity, properties, option);
}

int main(int argc, char** argv) try {
    const std::string_view mode = argc == 2 ? argv[1] : "";
    const bool require_mnnvl = mode == "--require-mnnvl";
    const bool require_nccl = mode == "--require-nccl";
    nccl_only() = require_nccl;
    require(argc == 1 || require_mnnvl || require_nccl, "invalid collective check arguments");
    const int caller = require_mnnvl ? 0 : 1;
    const std::array<int, ranks> devices{caller, 1 - caller};
    int visible = 0;
    check(cudaGetDeviceCount(&visible), "count visible GPUs");
    require(visible == ranks, "expose exactly two reserved GPUs through CUDA_VISIBLE_DEVICES");
    check(cudaSetDevice(caller), "select initial caller");
    auto group = TensorParallel::create({devices[0], devices[1]});
    check_caller(caller, "create changed the caller device");
    require(group->rank_count() == ranks, "unexpected rank count");
    for (int r = 0; r < ranks; ++r) require(group->device(r) == devices[r], "wrong rank device");
    const auto counters = snapshot(*group);
    const bool vendor = counters.mnnvl_active;
    require(!require_mnnvl || vendor, "MNNVL is required for this run");
    require(!require_nccl || !vendor, "NCCL is required for this run");
    const auto unavailable = counters.mnnvl_built
                                 ? &TpArRankCounters::ar_fallback_setup_unavailable_total
                                 : &TpArRankCounters::ar_fallback_disabled_total;
    check_eager(*group, vendor, unavailable);
    require(group->graph_eligible(), "paired capture unavailable on the managed NCCL runtime");
    check_capture(*group, false, row, nccl,
                  vendor ? &TpArRankCounters::ar_fallback_capture_mismatch_total : unavailable);
    check_capture(*group, false, wide, nccl, vendor ? oversize : unavailable);
    check_capture(*group, true, 40 * row, vendor ? one_shot : nccl,
                  vendor ? nullptr : unavailable);
    check_capture(*group, true, wide, nccl, vendor ? oversize : unavailable);
    std::puts("capture: joined and independent routes, four fresh-input replays each");
    check(cudaSetDevice(caller), "select destruction caller");
    group.reset();
    check_caller(caller, "destruction changed the caller device");
    try {
        TensorParallel::create({devices.back(), visible});
        require(false, "create accepted an invalid second device");
    } catch (const std::exception&) {
        check_caller(caller, "failed create changed the caller device");
    }
    std::puts("caller device preserved across create, enqueue, capture and destruction");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
