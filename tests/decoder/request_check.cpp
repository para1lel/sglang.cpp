#include "sglang/decoder.h"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

extern "C" cudaError_t __real_cudaMalloc(void** storage, std::size_t bytes);
extern "C" cudaError_t __real_cudaFree(void* storage);
extern "C" cudaError_t __real_cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned flags);
extern "C" cudaError_t __real_cudaStreamDestroy(cudaStream_t stream);
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t stream);
extern "C" cudaError_t __real_cudaMemcpyAsync(void* destination, const void* source,
                                             std::size_t bytes, cudaMemcpyKind kind,
                                             cudaStream_t stream);

namespace {

enum class Fault { none, allocation, synchronize };

struct WeightRead;

struct CudaCalls {
    std::unordered_set<void*> allocations;
    std::unordered_set<cudaStream_t> streams;
    Fault pending = Fault::none;
    int allocation_countdown = 0;
    std::size_t completed_reads = 0;
    bool read_pending = false;
    int injections = 0;
    bool bad_handle = false;
    WeightRead* weight_read = nullptr;

    bool consume(Fault fault) {
        if (pending != fault) {
            return false;
        }
        pending = Fault::none;
        ++injections;
        return true;
    }
};

CudaCalls calls;
constexpr int vocabulary = 248320;
const std::vector<int> prompt{760, 6511, 314, 9338, 369};
const std::vector<int> two_ids{11751, 13};

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct WeightRead {
    explicit WeightRead(std::unordered_set<void*> weights) : model(std::move(weights)) {
        require(!model.empty(), "No model allocation for the pending read");
        early_releases.reserve(model.size());
        void* pinned = nullptr;
        auto status = cudaMallocHost(&pinned, sizeof(*result));
        require(status == cudaSuccess, "Cannot allocate pinned weight readback");
        result = static_cast<unsigned char*>(pinned);
        status = cudaMemcpy(&expected, *model.begin(), sizeof(expected), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            cudaFreeHost(result);
            throw std::runtime_error("Cannot read initial model byte");
        }
        *result = static_cast<unsigned char>(expected ^ 0xff);
        calls.weight_read = this;
    }

    ~WeightRead() {
        release();
        calls.weight_read = nullptr;
        if (calls.streams.count(stream) && __real_cudaStreamSynchronize(stream) != cudaSuccess) {
            return;
        }
        for (auto* storage : early_releases) {
            if (__real_cudaFree(storage) == cudaSuccess) {
                calls.allocations.erase(storage);
            }
        }
        cudaFreeHost(result);
    }

    static void wait(void* opaque) {
        auto& read = *static_cast<WeightRead*>(opaque);
        std::unique_lock lock(read.mutex);
        read.timed_out = !read.ready.wait_for(lock, std::chrono::seconds(20), [&read] {
            return read.released;
        });
    }

    cudaError_t queue(cudaStream_t request) {
        stream = request;
        const auto status = cudaLaunchHostFunc(stream, wait, this);
        if (status != cudaSuccess) {
            return status;
        }
        return __real_cudaMemcpyAsync(result, *model.begin(), sizeof(*result),
                                      cudaMemcpyDeviceToHost, stream);
    }

    void release() {
        const std::lock_guard lock(mutex);
        released = true;
        ready.notify_one();
    }

    void check_pending() const {
        require(calls.streams.count(stream) == 1, "Deferred request stream was lost");
        require(cudaStreamQuery(stream) == cudaErrorNotReady, "Weight read is no longer pending");
    }

    bool finished() {
        const std::lock_guard lock(mutex);
        return drained && !timed_out && *result == expected;
    }

    std::unordered_set<void*> model;
    std::unordered_set<void*> early_releases;
    cudaStream_t stream = nullptr;
    unsigned char* result = nullptr;
    unsigned char expected = 0;
    std::mutex mutex;
    std::condition_variable ready;
    int sync_failures = 0;
    bool released = false;
    bool timed_out = false;
    bool drained = false;
};

void check_ids(const sglang::DecoderResult& result) {
    require(result.output_token_ids == two_ids, "Unexpected greedy token IDs");
    require(result.finish_reason == sglang::DecoderFinishReason::length, "Wrong finish reason");
}

void check_order(const std::vector<sglang::DecoderLogit>& row) {
    require(row.size() == vocabulary, "Full ranking lacks vocabulary entries");
    std::vector<bool> seen(vocabulary);
    bool saw_tie = false;
    for (std::size_t index = 0; index < row.size(); ++index) {
        const auto& entry = row[index];
        require(entry.token_id >= 0 && entry.token_id < vocabulary, "Token outside vocabulary");
        require(!seen[entry.token_id] && std::isfinite(entry.logit),
                "Invalid ranked token or logit");
        seen[entry.token_id] = true;
        if (index == 0) {
            continue;
        }
        const auto& previous = row[index - 1];
        if (previous.logit == entry.logit) {
            saw_tie = true;
            require(previous.token_id < entry.token_id, "Tie is not ordered by ascending token ID");
        } else {
            require(previous.logit > entry.logit, "Ranking is not ordered by descending logit");
        }
    }
    require(saw_tie, "Full ranking did not exercise ties");
}

void check_ranking(const sglang::Decoder& decoder) {
    sglang::DecoderOptions options;
    options.max_new_tokens = 2;
    options.top_logits = 1;
    const auto top_one = decoder.generate(prompt, options);
    options.top_logits = vocabulary;
    const auto full = decoder.generate(prompt, options);
    check_ids(top_one);
    check_ids(full);
    require(top_one.step_top_logits.size() == two_ids.size() &&
                full.step_top_logits.size() == two_ids.size(),
            "Expected a ranking for each generation step");
    for (std::size_t step = 0; step < two_ids.size(); ++step) {
        const auto& small = top_one.step_top_logits[step];
        const auto& all = full.step_top_logits[step];
        check_order(all);
        require(small.size() == 1, "Top-one request returned the wrong count");
        require(small.front().token_id == two_ids[step] &&
                    small.front().token_id == all.front().token_id &&
                    small.front().logit == all.front().logit,
                "Top one differs from the full ranking");
        std::cout << "Step " << step << " top-one capacity " << small.capacity() << '\n';
        require(small.capacity() <= 32, "Top-one result retains oversized capacity");
    }
}

void check_live(std::size_t model_allocations) {
    require(!calls.bad_handle, "CUDA release attempted on a non-live handle");
    require(calls.streams.empty(), "CUDA stream remains live outside the request");
    require(calls.allocations.size() == model_allocations,
            "CUDA allocations remain live outside their owner");
}

void expect_failure(const sglang::Decoder& decoder, const sglang::DecoderOptions& options,
                    Fault fault) {
    calls.pending = fault;
    calls.allocation_countdown = 3;
    const std::string_view expected = fault == Fault::allocation ? "Allocate CUDA memory" :
                                                                  "Synchronize decoder stream";
    std::string error;
    try {
        decoder.generate(prompt, options);
    } catch (const std::exception& failure) {
        error = failure.what();
    }
    std::cout << "Injected failure reported by generate: " << error << '\n';
    if (fault == Fault::synchronize) {
        require(calls.completed_reads == static_cast<std::size_t>(options.max_new_tokens),
                "The synchronization check did not complete both greedy reads");
    }
    require(calls.injections == 1 && calls.pending == Fault::none,
            "The selected CUDA fault was not injected");
    require(error.find(expected) != std::string::npos,
            "Generate did not report the injected failure at the expected operation");
}

void check_cleanup(const sglang::Decoder& decoder, Fault fault) {
    const auto model_allocations = calls.allocations.size();
    require(model_allocations != 0, "No model allocations reached the CUDA observer");
    check_live(model_allocations);
    sglang::DecoderOptions options;
    options.max_new_tokens = 2;
    if (fault != Fault::none) {
        expect_failure(decoder, options, fault);
    }
    check_ids(decoder.generate(prompt, options));
    check_live(model_allocations);
    check_ids(decoder.generate(prompt, options));
    check_live(model_allocations);
}

void check_weight_lifetime(const char* checkpoint) {
    const sglang::Decoder standby(checkpoint);
    const auto baseline = calls.allocations;
    auto victim = std::make_unique<sglang::Decoder>(checkpoint);
    auto weights = calls.allocations;
    for (auto* allocation : baseline) {
        weights.erase(allocation);
    }
    WeightRead read(std::move(weights));
    sglang::DecoderOptions options;
    options.max_new_tokens = 2;
    std::string error;
    try {
        victim->generate(prompt, options);
    } catch (const std::exception& failure) {
        error = failure.what();
    }
    require(error.find("Complete decoder step") != std::string::npos,
            "Generation did not surface the pending read synchronization failure");
    require(read.sync_failures >= 2, "Fault did not prevent both completion checks");
    read.check_pending();
    victim.reset();
    read.check_pending();
    std::cout << "Failed completion checks " << read.sync_failures
              << ", early model releases " << read.early_releases.size() << '\n';
    require(read.early_releases.empty(), "Model release attempted while weight read was pending");
    read.release();
    check_ids(standby.generate(prompt, options));
    require(read.finished(), "Deferred request did not complete its real CUDA weight read");
    require(calls.allocations == baseline, "Deferred model storage was not reclaimed");
    check_live(baseline.size());
}

}

extern "C" cudaError_t __wrap_cudaMalloc(void** storage, std::size_t bytes) {
    if (calls.pending == Fault::allocation && --calls.allocation_countdown == 0 &&
        calls.consume(Fault::allocation)) {
        *storage = nullptr;
        return cudaErrorMemoryAllocation;
    }
    const auto status = __real_cudaMalloc(storage, bytes);
    if (status == cudaSuccess && !calls.allocations.insert(*storage).second) {
        calls.bad_handle = true;
    }
    return status;
}

extern "C" cudaError_t __wrap_cudaFree(void* storage) {
    if (storage != nullptr && calls.allocations.count(storage) == 0) {
        calls.bad_handle = true;
        return cudaErrorInvalidValue;
    }
    auto* read = calls.weight_read;
    if (read && !read->drained && read->model.count(storage)) {
        if (!read->early_releases.insert(storage).second) {
            calls.bad_handle = true;
            return cudaErrorInvalidValue;
        }
        return cudaSuccess;
    }
    const auto status = __real_cudaFree(storage);
    if (status == cudaSuccess) {
        calls.allocations.erase(storage);
    }
    return status;
}

extern "C" cudaError_t __wrap_cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned flags) {
    const auto status = __real_cudaStreamCreateWithFlags(stream, flags);
    if (status == cudaSuccess && !calls.streams.insert(*stream).second) {
        calls.bad_handle = true;
    }
    return status;
}

extern "C" cudaError_t __wrap_cudaStreamDestroy(cudaStream_t stream) {
    if (calls.streams.count(stream) == 0) {
        calls.bad_handle = true;
        return cudaErrorInvalidResourceHandle;
    }
    const auto status = __real_cudaStreamDestroy(stream);
    if (status == cudaSuccess) {
        calls.streams.erase(stream);
    }
    return status;
}

extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* destination, const void* source,
                                             std::size_t bytes, cudaMemcpyKind kind,
                                             cudaStream_t stream) {
    const auto status = __real_cudaMemcpyAsync(destination, source, bytes, kind, stream);
    auto* read = calls.weight_read;
    if (status == cudaSuccess && read && !read->stream && kind == cudaMemcpyDeviceToHost) {
        return read->queue(stream);
    }
    if (status == cudaSuccess && calls.pending == Fault::synchronize &&
        kind == cudaMemcpyDeviceToHost) {
        calls.read_pending = true;
    }
    return status;
}

extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream) {
    if (calls.streams.count(stream) == 0) {
        calls.bad_handle = true;
        return cudaErrorInvalidResourceHandle;
    }
    auto* read = calls.weight_read;
    if (read && read->stream == stream && !read->released) {
        ++read->sync_failures;
        return cudaErrorLaunchOutOfResources;
    }
    if (calls.completed_reads == two_ids.size() && calls.consume(Fault::synchronize)) {
        return cudaErrorLaunchOutOfResources;
    }
    const auto status = __real_cudaStreamSynchronize(stream);
    if (status == cudaSuccess && read && read->stream == stream) {
        read->drained = true;
    }
    if (status == cudaSuccess && calls.read_pending) {
        calls.read_pending = false;
        ++calls.completed_reads;
    }
    return status;
}

int main(int argc, const char** argv) {
    try {
        require(argc == 3, "Usage: request_check CHECKPOINT MODE");
        const std::string_view mode = argv[2];
        if (mode == "weight-lifetime") {
            check_weight_lifetime(argv[1]);
        } else {
            const sglang::Decoder decoder(argv[1]);
            if (mode == "ranking") {
                check_ranking(decoder);
            } else if (mode == "cleanup") {
                check_cleanup(decoder, Fault::none);
            } else if (mode == "allocation") {
                check_cleanup(decoder, Fault::allocation);
            } else if (mode == "synchronize") {
                check_cleanup(decoder, Fault::synchronize);
            } else {
                throw std::invalid_argument("Unknown request check");
            }
        }
        check_live(0);
        std::cout << mode << " request check passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Request check failed: " << error.what() << '\n';
        return 1;
    }
}
