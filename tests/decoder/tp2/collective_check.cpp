#include "sglang/runtime/backend/tensor_parallel.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace {

using sglang::decoder::TensorParallel;
constexpr int ranks = 2;
constexpr std::array<int, ranks> devices{1, 0};
constexpr int caller = devices.front();
constexpr int count = 5120;
constexpr auto bytes = (count + 1) * sizeof(__nv_bfloat16);

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

void check_caller(const char* operation) {
    int actual = -1;
    check(cudaGetDevice(&actual), "cudaGetDevice");
    require(actual == caller, operation);
}

int input(int rank, int index, int generation) {
    if (index == count) return 100 + rank;
    if (rank == 0) return (index % 23) - 13 + generation;
    return 2 * (index % 17) + 35 - 3 * generation;
}

struct Fixture {
    cudaStream_t streams[ranks]{};
    __nv_bfloat16* buffers[ranks]{};
    __nv_bfloat16* host[ranks]{};
    cudaGraph_t graphs[ranks]{};
    cudaGraphExec_t execs[ranks]{};

    Fixture() {
        for (int r = 0; r < ranks; ++r) {
            check(cudaSetDevice(devices[r]), "select allocation device");
            check(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking), "create stream");
            check(cudaMalloc(&buffers[r], bytes), "allocate device rows");
            check(cudaMallocHost(&host[r], bytes), "allocate pinned staging");
        }
    }

    void upload(int generation) {
        for (int r = 0; r < ranks; ++r) {
            for (int i = 0; i <= count; ++i) {
                host[r][i] = __float2bfloat16(static_cast<float>(input(r, i, generation)));
            }
            check(cudaSetDevice(devices[r]), "select input device");
            check(cudaMemcpyAsync(buffers[r], host[r], bytes, cudaMemcpyHostToDevice, streams[r]),
                  "upload rank input");
        }
    }

    void verify(int generation) {
        for (int r = 0; r < ranks; ++r) {
            check(cudaSetDevice(devices[r]), "select result device");
            check(cudaMemcpyAsync(host[r], buffers[r], bytes, cudaMemcpyDeviceToHost, streams[r]),
                  "download rank result");
        }
        for (int r = 0; r < ranks; ++r) {
            check(cudaSetDevice(devices[r]), "select completion device");
            check(cudaStreamSynchronize(streams[r]), "complete result copy");
            for (int i = 0; i <= count; ++i) {
                int expected = input(r, i, generation);
                if (i < count) expected = input(0, i, generation) + input(1, i, generation);
                const float actual = __bfloat162float(host[r][i]);
                if (actual != expected) {
                    std::fprintf(stderr, "generation %d rank %d index %d: expected %d, got %g\n",
                                 generation, r, i, expected, actual);
                    std::exit(1);
                }
            }
        }
    }

    void reduce(TensorParallel& group) {
        check(cudaSetDevice(caller), "select collective caller");
        group.allreduce_bf16(buffers, streams, count);
        check_caller("allreduce changed the caller device");
    }

    void capture(TensorParallel& group) {
        for (int r = 0; r < ranks; ++r) {
            check(cudaSetDevice(devices[r]), "select capture device");
            check(cudaStreamBeginCapture(streams[r], cudaStreamCaptureModeThreadLocal),
                  "begin rank stream capture");
        }
        reduce(group);
        for (int r = ranks - 1; r >= 0; --r) {
            check(cudaSetDevice(devices[r]), "select graph device");
            check(cudaStreamEndCapture(streams[r], &graphs[r]), "end rank stream capture");
        }
        for (int r = 0; r < ranks; ++r) {
            check(cudaSetDevice(devices[r]), "select graph execution device");
            check(cudaGraphInstantiateWithFlags(&execs[r], graphs[r], 0), "instantiate rank graph");
        }
    }

    void replay() {
        for (int r = 0; r < ranks; ++r) {
            check(cudaSetDevice(devices[r]), "select replay device");
            check(cudaGraphLaunch(execs[r], streams[r]), "launch rank graph");
        }
    }

    ~Fixture() {
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

}

int main() try {
    int visible = 0;
    check(cudaGetDeviceCount(&visible), "count visible GPUs");
    require(visible == ranks, "expose exactly two reserved GPUs through CUDA_VISIBLE_DEVICES");
    check(cudaSetDevice(caller), "select initial caller");
    try {
        TensorParallel::create({devices.back(), visible});
        require(false, "create accepted an invalid second device");
    } catch (const std::exception&) {
        check_caller("failed create changed the caller device");
    }
    auto group = TensorParallel::create({devices[0], devices[1]});
    check_caller("create changed the caller device");
    require(group->rank_count() == ranks, "unexpected rank count");
    for (int r = 0; r < ranks; ++r) require(group->device(r) == devices[r], "wrong rank device");
    {
        Fixture fixture;
        fixture.upload(0);
        fixture.reduce(*group);
        fixture.verify(0);
        std::puts("eager: exact BF16 sum and tail guard on both rank streams");
        require(group->graph_eligible(), "paired capture unavailable on the managed NCCL runtime");
        fixture.capture(*group);
        for (int generation : {1, 2}) {
            fixture.upload(generation);
            fixture.replay();
            fixture.verify(generation);
        }
        std::puts("thread-local capture: two replays with fresh inputs on both rank streams");
    }
    check(cudaSetDevice(caller), "select destruction caller");
    group.reset();
    check_caller("destruction changed the caller device");
    std::puts("caller device preserved across create, enqueue, capture and destruction");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
