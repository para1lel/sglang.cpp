#include "sglang/decoder_internal.h"
#include "sglang/decoder_mtp.h"
#include "sglang/decoder_prefill.h"
#include "sglang/decoder_prefix_cache.h"
#include "sglang/decoder_verify.h"
#include "sglang/runtime/backend/device_arena.h"
#include "sglang/runtime/cache/lmcache_group.h"
#include "sglang/runtime/server/batch_cache.h"
#include "sglang/text/decoder_logprobs.h"
#include "sglang/testing/prefix_fixture.h"
#include "sglang/tokenizer/tokenizer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <tuple>
#include <stdexcept>
#include <thread>
#include <utility>

using namespace sglang;
using namespace sglang::decoder;

namespace {
constexpr int checkpoint = 1279;
constexpr int imported = 1391;
constexpr int prompt_end = 1429;
constexpr int capacity = 1536;
std::vector<std::pair<int, int>> appends;
const std::int32_t* measured_pages = nullptr;
std::optional<std::size_t> checkpoint_copy_bytes;

void require(bool predicate, const char* message) {
    if (!predicate) throw std::runtime_error(message);
}

class CopyGate {
public:
    ~CopyGate() {
        open();
        std::unique_lock lock(mutex_);
        if (entered_) changed_.wait(lock, [this] { return finished_; });
    }

    void wait() {
        std::unique_lock lock(mutex_);
        entered_ = true;
        changed_.notify_all();
        passed_ = changed_.wait_for(lock, std::chrono::seconds(15), [this] { return opened_; });
        finished_ = true;
        changed_.notify_all();
    }

    void wait_until_entered() {
        std::unique_lock lock(mutex_);
        require(changed_.wait_for(lock, std::chrono::seconds(10), [this] { return entered_; }),
                "A checkpoint gate did not reach its CUDA call");
    }

    void open() {
        const std::lock_guard lock(mutex_);
        opened_ = true;
        changed_.notify_all();
    }

    bool held() {
        const std::lock_guard lock(mutex_);
        return entered_ && !finished_;
    }

    void expect_released() {
        std::unique_lock lock(mutex_);
        require(changed_.wait_for(lock, std::chrono::seconds(10), [this] { return finished_; }) &&
                passed_, "A checkpoint gate timed out before explicit release");
    }

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    bool entered_ = false;
    bool opened_ = false;
    bool finished_ = false;
    bool passed_ = false;
};

CopyGate* copy_gate = nullptr;
struct PhysicalStorageCalls {
    std::atomic<bool> watching{false};
    std::atomic<int> count{0};
};

PhysicalStorageCalls& physical_storage_calls() {
    static PhysicalStorageCalls calls;
    return calls;
}
enum class CheckpointFault { none, second_copy, event };
CheckpointFault checkpoint_fault = CheckpointFault::none;
std::size_t checkpoint_copies = 0;

void store(const LmCacheClient& cache, const PrefixCacheKeys& keys,
           const PrefixDeviceTransfer& command) {
    require(cache.store(keys, command) == LmCacheStatus::complete, "Cache store failed");
}

int restore(const LmCacheClient& cache, const PrefixCacheKeys& keys, Request& request,
            const PrefixCacheRead& read, cudaStream_t stream, MtpDraftRequest* draft) {
    const auto command = restore_prefix_view(request, read, stream, draft);
    require(cache.retrieve(keys, command) == LmCacheStatus::complete, "Cache retrieve failed");
    return commit_restored_prefix(request, command, draft);
}

void expect_appends(const char* label, int position, int rows) {
    require(appends.size() == static_cast<std::size_t>(rows ? full_attention_layers : 0),
            "Unexpected count of MXFP8 append submissions");
    for (auto actual : appends) {
        require(actual == std::pair<int, int>{position, rows}, "Unexpected MXFP8 append extent");
    }
    std::printf("writes %s %d %d %zu\n", label, position, rows, appends.size());
    appends.clear();
}

struct Lane {
    Request request;
    MtpDraftRequest draft;
    DecoderPromptLogprobs scores;
    explicit Lane(const DecoderModel& model, cudaStream_t stream, int initial = capacity,
                  RequestBlockCache* retired = nullptr)
        : request(0, false, retired), draft(model.mtp_weights, capacity, retired) {
        request.initialize(initial, stream, capacity);
        request.reset(stream);
    }
};

struct Generation {
    std::vector<int> ids;
    std::vector<int> proposals;
    std::vector<int> accepted;
    std::vector<std::uint16_t> logits;
    std::string target;
    std::array<std::string, 4> recurrent;
};

Generation generate(const DecoderModel& model, Lane& lane, int position, cudaStream_t stream) {
    constexpr int rounds = 3;
    Generation result;
    model.compute_prefill_logits(lane.request.work, stream);
    check(text_argmax(lane.request.work.logits, vocab_size, lane.request.work.argmax_scratch,
                      lane.request.work.argmax, stream), "Select initial target winner");
    TextLogitCandidate first{};
    copy_cuda_async(&first, lane.request.work.argmax, sizeof(first), cudaMemcpyDeviceToHost,
                    stream, "Read first target winner");
    synchronize_cuda(stream, "Complete initial target scores");
    result.ids.push_back(first.token_id);
    ConstraintWorkspace constraint;
    TargetVerification target(lane.request, capacity, position, target_verify_max_rows, true,
                              stream);
    const Workspace* final_row = nullptr;
    for (int index = 0; index < rounds; ++index) {
        MtpRound round{&lane.draft, &target, &lane.request, nullptr, nullptr,
                       {result.ids.back()}, position, static_cast<int>(result.ids.size()),
                       target_verify_max_rows, decoder_max_mtp_steps};
        require(MtpRound::propose_all(&round, 1, constraint, stream, {}), "K4 stopped");
        require(round.drafted == decoder_max_mtp_steps, "Missing all four K4 proposals");
        result.proposals.insert(result.proposals.end(), round.tokens.begin() + 1,
                                round.tokens.begin() + round.rows());
        model.verify_mtp(&round, 1, constraint, nullptr, stream, nullptr, nullptr);
        for (int row = 0; row < round.rows(); ++row) {
            const auto size = result.logits.size();
            result.logits.resize(size + vocab_size);
            std::memcpy(result.logits.data() + size, target.logits(row), vocab_size * sizeof(Bf16));
        }
        int accepted = 0;
        require(round.commit([&](int token, const Bf16*, bool matched, int row) {
            result.ids.push_back(token);
            final_row = target.row(row).work;
            accepted += matched;
            return true;
        }, stream, {}, nullptr, nullptr), "K4 commit stopped");
        result.accepted.push_back(accepted);
        position += round.consumed;
    }
    synchronize_cuda(stream, "Complete test generation");
    require(final_row, "Missing committed test verification row");
    result.target = prefix_fixture::target_digest(lane.request, position);
    result.recurrent = prefix_fixture::recurrent_digest(lane.request, *final_row);
    return result;
}

void compare_scores(const DecoderPromptLogprobs& cold, const DecoderPromptLogprobs& warm) {
    require(cold.size() == warm.size() && !cold.empty(), "Sparse score row count differs");
    double worst = 0;
    const auto compare = [&](float a, float b) {
        require(std::isfinite(a) && std::isfinite(b), "Non-finite sparse teacher score");
        worst = std::max(worst, std::abs(double(a) - double(b)));
    };
    int changed_top = 0;
    for (std::size_t i = 0; i < cold.size(); ++i) {
        require(cold[i] && warm[i] && cold[i]->selected.token_id == warm[i]->selected.token_id,
                "Sparse teacher token differs");
        compare(cold[i]->selected.value, warm[i]->selected.value);
        const auto& a = cold[i]->alternatives;
        const auto& b = warm[i]->alternatives;
        require(a.size() == b.size(), "Sparse top alternatives count differs");
        for (std::size_t j = 0; j < a.size(); ++j) {
            changed_top += a[j].token_id != b[j].token_id;
            compare(a[j].value, b[j].value);
        }
    }
    std::printf("teacher %zu max_logprob_difference %.9g changed_top_ids %d\n",
                cold.size(), worst, changed_top);
    require(changed_top == 0, "Sparse teacher top alternative IDs differ");
    require(worst <= 0.05, "Sparse teacher scores exceeded tolerance");
}

void compare_generation(const Generation& cold, const Generation& warm) {
    require(cold.ids == warm.ids && cold.proposals == warm.proposals &&
            cold.accepted == warm.accepted, "Actual K4 outputs, proposals or acceptance differ");
    require(cold.logits == warm.logits, "Actual K4 full logits differ");
    require(cold.target == warm.target, "K4 committed target state differs");
    require(cold.recurrent == warm.recurrent, "K4 committed recurrent/observable differs");
    std::printf("k4 ids %zu proposals %zu logits %zu acceptance", cold.ids.size(),
                cold.proposals.size(), cold.logits.size());
    for (int n : cold.accepted) std::printf(" %d", n);
    std::printf("\n");
}

void check_history_gap(const LmCacheClient& cache, const std::vector<int>& prompt,
                       const DecoderModel& model, cudaStream_t stream, Lane& cold) {
    const PrefixCacheKeys keys(prompt, prefix_fixture::run_salt("oracle-gap"));
    auto first = export_prefix_view(cold.request, 0, 0, 512, stream);
    first.layout.gdn = {};
    store(cache, keys, first);
    auto later = export_prefix_view(cold.request, 768, 0, imported, stream, &cold.draft);
    store(cache, keys, later);
    const auto read = LmCacheGroup(cache).lookup_prefix({&keys}, prompt_end, true);
    require(read.usable_gqa == 512 && read.matched_gdn == imported &&
                read.matched_gdn_endpoints == std::vector<int>{imported} && read.usable_gdn == 0 &&
                read.usable_draft == 0,
            "Daemon crossed a GQA history gap");
    Lane gap(model, stream, imported);
    require(restore(cache, keys, gap.request, read, stream, &gap.draft) == 0 &&
            gap.request.cached_gqa_tokens == 512 && gap.draft.valid_prefix() == 0,
            "GDN was exposed across a historical GQA hole");
    const auto before = prefix_fixture::target_digest(gap.request, 512);
    gap.request.grow_kv(1600 - 64, 0, stream);
    require(before == prefix_fixture::target_digest(gap.request, 512),
            "Growth lost imported rows with zero coherent checkpoint");
    std::printf("hole usable_gqa 512 matched_gdn %d coherent_gdn 0 draft_rows 0\n",
                read.matched_gdn);
}

class GuardedMxfp8 {
public:
    GuardedMxfp8(Allocations& memory, int pages, unsigned seed) : pages_(pages) {
        std::size_t span = 67;
        for (int plane = 0; plane < 8; ++plane) {
            offsets_[plane] = span;
            span += pages * page_bytes(plane) * layers(plane) + 89 + plane;
        }
        bytes.resize(span);
        for (auto& value : bytes) {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            value = seed >> 24;
        }
        device = memory.allocate<std::uint8_t>(span);
    }

    PrefixDeviceMxfp8 view(int family, int begin, int end) const {
        PrefixDeviceMxfp8 result{begin, end, layers(family * 4), pages_, mxfp8_kv_heads, {}};
        for (int plane = 0; plane < 4; ++plane) {
            result.planes[plane] = {0, offsets_[family * 4 + plane]};
        }
        return result;
    }

    std::size_t offset(int plane, int layer, int row, int head, int element) const {
        const auto base = offsets_[plane] + page_bytes(plane) *
            (layer * pages_ + row / mxfp8_page_size);
        const int token = row % mxfp8_page_size;
        if (plane % 4 < 2) {
            return base + (token * mxfp8_kv_heads + head) * mxfp8_head_dim + element;
        }
        return base + head * mxfp8_page_size * mxfp8_groups +
            (token % 32) * 32 + (token / 32) * 8 + element;
    }

    static int layers(int plane) { return plane < 4 ? full_attention_layers : 1; }
    static int elements(int plane) { return plane % 4 < 2 ? mxfp8_head_dim : mxfp8_groups; }

    std::uint8_t* device;
    std::vector<std::uint8_t> bytes;

private:
    static std::size_t page_bytes(int plane) {
        return plane % 4 < 2 ? mxfp8_payload_bytes_per_page : mxfp8_scale_bytes_per_page;
    }
    int pages_;
    std::array<std::size_t, 8> offsets_;
};

void check_guarded_ranges(const LmCacheClient& cache, const std::vector<int>& prompt,
                          Allocations& memory) {
    GuardedMxfp8 source(memory, 4, 0xe457abf3), destination(memory, 5, 0x117ca31b);
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "Read test CUDA device UUID");
    cudaEvent_t done;
    check(cudaEventCreateWithFlags(&done, cudaEventInterprocess | cudaEventDisableTiming),
          "Create guarded range ready event");
    const std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, decltype(&cudaEventDestroy)>
        event(done, cudaEventDestroy);

    const auto transfer = [&](const GuardedMxfp8& buffer, int begin, int end) {
        cudaIpcEventHandle_t ready;
        cudaIpcMemHandle_t address;
        check(cudaIpcGetEventHandle(&ready, done), "Export guarded range ready event");
        check(cudaIpcGetMemHandle(&address, buffer.device), "Export guarded range allocation");
        PrefixDeviceTransfer result;
        auto& command = result.transfer;
        std::memcpy(command.device.data(), properties.uuid.bytes, command.device.size());
        std::memcpy(command.ready_event.data(), &ready, command.ready_event.size());
        auto& allocation = command.allocations.emplace_back();
        std::memcpy(allocation.handle.data(), &address, allocation.handle.size());
        allocation.bytes = buffer.bytes.size();
        result.layout.gqa = buffer.view(0, begin, end);
        result.layout.draft = buffer.view(1, begin, end);
        return result;
    };

    check(cudaMemcpy(source.device, source.bytes.data(), source.bytes.size(),
                     cudaMemcpyHostToDevice), "Initialize guarded range source");
    check(cudaEventRecord(done, memory.stream()), "Record guarded range ready event");
    for (const auto [begin, end] : {std::pair{0, 31}, {0, 33}, {0, 127}, {0, 129},
                                  {0, 255}, {0, 257}, {256, 257}}) {
        const PrefixCacheKeys keys(prompt, prefix_fixture::run_salt(
            "guarded-" + std::to_string(begin) + "-" + std::to_string(end)));
        store(cache, keys, transfer(source, begin, end));
        check(cudaMemcpy(destination.device, destination.bytes.data(), destination.bytes.size(),
                         cudaMemcpyHostToDevice), "Initialize guarded range destination");
        LmCacheReceipt receipt;
        require(cache.retrieve(keys, transfer(destination, begin, end), &receipt) ==
                LmCacheStatus::complete, "Guarded range retrieve failed");
        require(receipt.bytes[0] == std::size_t(end - begin) * 33792 &&
                receipt.bytes[2] == std::size_t(end - begin) * 2112,
                "Guarded range logical byte count differs");
        auto expected = destination.bytes;
        for (int plane = 0; plane < 8; ++plane) {
            for (int layer = 0; layer < GuardedMxfp8::layers(plane); ++layer) {
                for (int row = begin; row < end; ++row) {
                    for (int head = 0; head < mxfp8_kv_heads; ++head) {
                        for (int i = 0; i < GuardedMxfp8::elements(plane); ++i) {
                            expected[destination.offset(plane, layer, row, head, i)] =
                                source.bytes[source.offset(plane, layer, row, head, i)];
                        }
                    }
                }
            }
        }
        std::vector<std::uint8_t> actual(expected.size());
        check(cudaMemcpy(actual.data(), destination.device, actual.size(), cudaMemcpyDeviceToHost),
              "Read guarded destination");
        require(expected == actual, "Guarded range data, stride, or untouched padding differs");
        std::printf("guarded range [%d,%d) GQA %lu draft %lu\n", begin, end,
                    receipt.bytes[0], receipt.bytes[2]);
    }
}

void check_key_provenance(const std::vector<int>& prompt) {
    PrefixCacheKeys keys(prompt, "oracle");
    auto changed = prompt;
    ++changed[checkpoint - 1];
    PrefixCacheKeys changed_keys(changed, "oracle");
    require(keys.key(checkpoint - 1) == changed_keys.key(checkpoint - 1) &&
        keys.key(checkpoint) != changed_keys.key(checkpoint), "Shifted draft provenance differs");
    PrefixCacheKeys incremental({prompt.begin(), prompt.end() - 1}, "oracle");
    incremental.append(prompt.back());
    require(incremental.key(prompt_end) == keys.key(prompt_end), "Incremental text key differs");
}
}

extern "C" cudaError_t real_append(Mxfp8KvCache, Mxfp8PageTable, int, int, const Bf16*,
    const Bf16*, cudaStream_t)
    asm("__real__ZN6sglang15mxfp8_kv_appendENS_12Mxfp8KvCacheENS_14Mxfp8PageTableEii"
        "PK13__nv_bfloat16S4_P11CUstream_st");
extern "C" cudaError_t wrap_append(Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
    const Bf16* keys, const Bf16* values, cudaStream_t stream)
    asm("__wrap__ZN6sglang15mxfp8_kv_appendENS_12Mxfp8KvCacheENS_14Mxfp8PageTableEii"
        "PK13__nv_bfloat16S4_P11CUstream_st");
extern "C" cudaError_t wrap_append(Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
    const Bf16* keys, const Bf16* values, cudaStream_t stream) {
    if (pages.device_pages == measured_pages) appends.emplace_back(position, rows);
    return real_append(cache, pages, position, rows, keys, values, stream);
}

extern "C" cudaError_t __real_cudaMemcpyAsync(void*, const void*, std::size_t,
                                             cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* dst, const void* src, std::size_t size,
                                             cudaMemcpyKind kind, cudaStream_t stream) {
    if (checkpoint_copy_bytes && kind == cudaMemcpyDeviceToDevice) *checkpoint_copy_bytes += size;
    if (copy_gate && kind == cudaMemcpyDeviceToDevice) {
        const auto status = cudaLaunchHostFunc(stream, [](void* gate) {
            static_cast<CopyGate*>(gate)->wait();
        }, std::exchange(copy_gate, nullptr));
        if (status != cudaSuccess) return status;
    }
    if (kind == cudaMemcpyDeviceToDevice && checkpoint_fault == CheckpointFault::second_copy &&
        ++checkpoint_copies == 2) {
        checkpoint_fault = CheckpointFault::none;
        return __real_cudaMemcpyAsync(nullptr, src, size, kind, stream);
    }
    return __real_cudaMemcpyAsync(dst, src, size, kind, stream);
}

extern "C" cudaError_t __real_cudaFree(void*);
extern "C" cudaError_t __wrap_cudaFree(void* memory) {
    auto& calls = physical_storage_calls();
    if (memory != nullptr && calls.watching)
        ++calls.count;
    return __real_cudaFree(memory);
}

extern "C" cudaError_t __real_cudaMalloc(void**, std::size_t);
extern "C" cudaError_t __wrap_cudaMalloc(void** memory, std::size_t size) {
    auto& calls = physical_storage_calls();
    if (calls.watching)
        ++calls.count;
    return __real_cudaMalloc(memory, size);
}

extern "C" cudaError_t real_event_record(cudaEvent_t event,
                                         cudaStream_t stream) asm("__real_cudaEventRecord");
extern "C" cudaError_t wrap_event_record(cudaEvent_t event,
                                         cudaStream_t stream) asm("__wrap_cudaEventRecord");
extern "C" cudaError_t wrap_event_record(cudaEvent_t event, cudaStream_t stream) {
    if (checkpoint_fault == CheckpointFault::event) {
        checkpoint_fault = CheckpointFault::none;
        return real_event_record(nullptr, stream);
    }
    return real_event_record(event, stream);
}

struct GraphCheckpointOracle {
    const DecoderModel& model;
    Allocations& execution;
    PrefillWorkspace& workspace;
    const LmCacheClient& cache;
    std::vector<int> history{1000, 1012, 1271, 1137};
    static constexpr int prefix = 4;

    void prime(Request& request) const {
        request.initialize(128, execution.stream());
        request.reset(execution.stream());
        request.prefix_checkpoint_outputs = true;
        model.prefill_chunk(request, history.data(), prefix, 0, workspace);
    }

    void check_export(Request& owner, const Workspace& owned_row, const Workspace& shared_row,
                      const std::array<std::string, 4>& expected, int end, bool verifier) const {
        const auto stream = execution.stream();
        synchronize_cuda(stream, "Complete peer replay before exporting owner state");
        require(prefix_fixture::recurrent_digest(owner, shared_row) != expected,
                "Peer replay did not overwrite shared graph state");
        require(prefix_fixture::recurrent_digest(owner, owned_row) == expected,
                "Peer replay overwrote request-owned graph state");
        const PrefixCacheKeys keys(history, prefix_fixture::run_salt(
            verifier ? "pre-export-k1" : "pre-export-k0"));
        store(cache, keys, export_prefix_view(owner, 0, 0, end, stream, nullptr, &owned_row));
        Request restored{model.device};
        restored.initialize(128, stream);
        restored.reset(stream);
        require(restore(cache, keys, restored, {end, end, end, end, 0}, stream, nullptr) == end,
                "Graph checkpoint did not restore");
        require(prefix_fixture::recurrent_digest(restored, restored.work) == expected,
                "Graph checkpoint restored stale raw operands");
        std::printf("K%d shared graph row overwritten before export; owner state exact\n",
                    int(verifier));
    }
};

void check_replay_before_export(GraphCheckpointOracle oracle, bool verifier) {
    const auto stream = oracle.execution.stream();
    Request owner{oracle.model.device}, peer{oracle.model.device};
    oracle.prime(owner);
    oracle.prime(peer);
    const int prefix = oracle.prefix;
    if (!verifier) {
        DecodeGraphs graph;
        const auto replay = [&](Request& request, int token, int position) {
            ForwardLane lane{&request, token, position, TextGdnMode::decode};
            return run_decode_lanes(oracle.model, oracle.execution, graph, &lane, 1,
                                    {true, true, true});
        };
        require(!replay(peer, 1033, prefix).graphed, "K0 warmup unexpectedly replayed");
        const auto a = replay(owner, 1147, prefix);
        require(a.graphed == 1 && a.ready[0], "Owner K0 did not replay");
        synchronize_cuda(stream, "Observe owner K0 raw operands");
        const auto expected = prefix_fixture::recurrent_digest(owner, *a.ready[0]);
        const auto b = replay(peer, 1437, prefix + 1);
        require(b.graphed == 1 && a.ready[0] == b.ready[0], "Peer K0 did not share graph row");
        oracle.history.push_back(1147);
        oracle.check_export(owner, owner.work, *a.ready[0], expected, prefix + 1, false);
        return;
    }

    constexpr int verification_row_count = 2;
    VerificationGraph graph(oracle.execution);
    TargetVerification a(owner, 128, prefix, verification_row_count, false, stream);
    TargetVerification b(peer, 128, prefix, verification_row_count, false, stream);
    const auto replay = [&](TargetVerification& verification, int position,
                            const std::array<int, verification_row_count>& tokens) {
        const VerificationSequence sequence{
            &verification,
            position,
            tokens.data(),
            verification_row_count,
        };
        return oracle.model.forward_verify(&sequence, 1, stream, &graph, nullptr);
    };
    require(!replay(b, prefix, {1033, 1341}), "K1 warmup unexpectedly replayed");
    b.read();
    b.commit(verification_row_count);
    require(replay(a, prefix, {1147, 1299}), "Owner K1 did not replay");
    const auto* shared_row = a.row(0).work;
    a.read();
    a.commit(1);
    synchronize_cuda(stream, "Commit owner K1 raw operands");
    const auto expected = prefix_fixture::recurrent_digest(owner, *shared_row);
    require(replay(b, prefix + verification_row_count, {1437, 1567}) && shared_row == b.row(0).work,
            "Peer K1 did not share graph row");
    b.read();
    b.commit(verification_row_count);
    oracle.history.push_back(1147);
    oracle.check_export(owner, *a.checkpoint_row(0), *shared_row, expected, prefix + 1, true);
}

constexpr std::size_t old_slab_bytes = 1280ULL * 33792;

void check_detached_growth(Lane& lane, RequestBlockCache& retired, const LmCacheClient& cache,
                           const PrefixCacheKeys& keys, PrefixDeviceTransfer alias,
                           cudaStream_t stream) {
    const auto before = cache.lookup_prefix(keys, true);
    require(before.gqa == 0 && before.endpoints.empty(),
            "Unsubmitted detached checkpoint was already visible");
    lane.request.grow_kv(capacity, checkpoint, stream);
    require(retired.unavailable_bytes() == old_slab_bytes,
            "An exported slab was charged incorrectly across two views");
    alias.transfer.device = {};
    bool failed = false;
    try {
        store(cache, keys, alias);
    } catch (const std::runtime_error&) {
        failed = true;
    }
    const auto after = cache.lookup_prefix(keys, true);
    require(failed && after.gqa == 0 && after.endpoints.empty(),
            "Invalid device STORE was published");
    alias = {};
    require(retired.unavailable_bytes() == old_slab_bytes,
            "A failed STORE released another view's source");
}

template <typename Extend>
void check_terminal_detach(const DecoderModel& model, cudaStream_t stream,
                           const LmCacheClient& cache, const PrefixCacheKeys& keys,
                           const PrefixCacheKeys& terminal_keys, const PrefixCacheRead& read,
                           const Lane& cold, RequestBlockCache& retired, const Extend& extend) {
    PrefixDeviceTransfer terminal;
    const auto target = prefix_fixture::target_digest(cold.request, prompt_end);
    const auto gdn = prefix_fixture::recurrent_digest(cold.request, cold.request.work);
    {
        Lane finished(model, stream, capacity, &retired);
        require(restore(cache, keys, finished.request, read, stream, &finished.draft) == checkpoint,
                "Missing source for terminal checkpoint oracle");
        extend(finished, false);
        checkpoint_copy_bytes = 0;
        terminal = export_prefix_view(finished.request, 0, 0, prompt_end, stream,
                                      &finished.draft, nullptr, true);
        require(std::exchange(checkpoint_copy_bytes, std::nullopt) == 0,
                "Terminal checkpoint converted an existing native allocation");
        synchronize_cuda(stream, "Complete direct terminal checkpoint source");
        finished.request.memory.recycle_after_completion();
        finished.draft.recycle_after_completion();
    }
    retired.clear();
    std::size_t expected_bytes = 0;
    for (const auto& allocation : terminal.transfer.allocations) expected_bytes += allocation.bytes;
    require(retired.unavailable_bytes() == expected_bytes,
            "Terminal release incorrectly charged detached source allocations");
    Lane successor(model, stream, capacity, &retired);
    synchronize_cuda(stream, "Initialize independent request over retired cache pool");
    store(cache, terminal_keys, terminal);
    terminal = {};
    const auto terminal_read =
        LmCacheGroup(cache).lookup_prefix({&terminal_keys}, prompt_end, true);
    require(restore(cache, terminal_keys, successor.request, terminal_read, stream,
                    &successor.draft) == prompt_end &&
                terminal_read.usable_draft == prompt_end - 1,
            "Terminal object retirement lost a stored cache family");
    require(prefix_fixture::target_digest(successor.request, prompt_end) == target &&
            prefix_fixture::recurrent_digest(successor.request, successor.request.work) == gdn,
            "Terminal source was reused before STORE completed");
    retired.clear();
    require(retired.unavailable_bytes() == 0, "Terminal STORE left detached bytes charged");
    std::printf("terminal D2D 0 retained %zu after retirement and successor reuse\n",
                expected_bytes);
}

std::array<const void*, 4> checkpoint_sources(const Request& request) {
    return {request.states.front().history, request.states.front().recurrent,
            request.work.hidden, request.work.residual};
}

static void check_delayed_checkpoint(const DecoderModel& model, const LmCacheClient& cache,
                                     const PrefixCacheKeys& keys, const PrefixCacheKeys& saved_keys,
                                     cudaStream_t stream, const PrefixCacheRead& read) {
    RequestBlockCache retired(0, true);
    Allocations peer_execution(0);
    peer_execution.open_stream();
    const auto peer_stream = peer_execution.stream();
    auto source = std::make_unique<Lane>(model, stream, capacity, &retired);
    require(restore(cache, keys, source->request, read, stream, &source->draft) == checkpoint,
            "Missing source for delayed checkpoint");
    const auto target = prefix_fixture::target_digest(source->request, checkpoint);
    const auto gdn = prefix_fixture::recurrent_digest(source->request, source->request.work);
    const auto operands = checkpoint_sources(source->request);
    {
        Lane spare(model, peer_stream, capacity, &retired);
        synchronize_cuda(peer_stream, "Warm independently recyclable request blocks");
        spare.request.memory.recycle_after_completion();
        spare.draft.recycle_after_completion();
    }
    PrefixDeviceTransfer pending;
    CopyGate gate;
    copy_gate = &gate;
    pending = export_prefix_view(source->request, 0, 0, checkpoint, stream, &source->draft);
    gate.wait_until_entered();
    source->request.memory.recycle_after_completion();
    source->draft.recycle_after_completion();
    source.reset();
    const auto cached = retired.bytes();
    Lane reused(model, peer_stream, capacity, &retired);
    require(retired.bytes() < cached, "Successor did not exercise the resident recycler");
    const auto follower = checkpoint_sources(reused.request);
    for (std::size_t index = 0; index < operands.size(); ++index) {
        require(operands[index] != follower[index], "Retirement recycled a pending D2D source");
    }
    synchronize_cuda(peer_stream, "Advance recycled request while checkpoint producer is held");
    require(gate.held(), "Checkpoint copy was already complete at retirement or physical reuse");
    gate.open();
    gate.expect_released();
    store(cache, saved_keys, pending);
    pending = {};
    const auto saved = LmCacheGroup(cache).lookup_prefix({&saved_keys}, prompt_end, true);
    require(restore(cache, saved_keys, reused.request, saved, peer_stream, &reused.draft) ==
                    checkpoint &&
                saved.usable_draft == checkpoint - 1,
            "Delayed STORE did not restore each state family");
    require(prefix_fixture::target_digest(reused.request, checkpoint) == target &&
            prefix_fixture::recurrent_digest(reused.request, reused.request.work) == gdn,
            "Delayed producer read a successor's state after cancellation");
    retired.clear();
    require(retired.unavailable_bytes() == 0, "Delayed STORE left detached bytes");
    std::puts("held CUDA D2D through source cancellation and physical reuse; restored exact state");
}

void check_checkpoint_faults(const Lane& lane, const LmCacheClient& cache,
                             const PrefixCacheKeys& keys, cudaStream_t stream) {
    const auto arena = DeviceArena::current(0);
    const auto available = arena->cache_available_bytes();
    Allocations held(0);
    std::size_t block = 256;
    while (block <= available / 2)
        block *= 2;
    for (; block >= 256; block /= 2) {
        while (arena->cache_available_bytes() >= block) {
            try {
                held.allocate_cache<std::byte>(block);
            } catch (const CudaError& error) {
                if (error.status != cudaErrorMemoryAllocation)
                    throw;
                break;
            }
        }
    }
    require(arena->cache_available_bytes() == 0,
            "Checkpoint fixture did not exhaust its cache slices");
    bool exhausted = false;
    try {
        export_prefix_view(lane.request, 0, 0, prompt_end, stream, &lane.draft);
    } catch (const CudaError& error) {
        exhausted = error.status == cudaErrorMemoryAllocation;
    }
    held.close();
    require(exhausted && arena->cache_available_bytes() == available,
            "Exhausted checkpoint lost its error or leaked cache storage");
    struct Case { CheckpointFault fault; cudaError_t status; std::size_t submitted; };
    for (const auto test : {Case{CheckpointFault::second_copy, cudaErrorInvalidValue, 2},
                            Case{CheckpointFault::event, cudaErrorInvalidResourceHandle, 0}}) {
        checkpoint_copies = 0;
        checkpoint_fault = test.fault;
        bool failed = false;
        try {
            export_prefix_view(lane.request, 0, 0, prompt_end, stream, &lane.draft);
        } catch (const CudaError& error) {
            failed = error.status == test.status;
        }
        require(failed && checkpoint_fault == CheckpointFault::none &&
                    checkpoint_copies == test.submitted && cudaPeekAtLastError() == cudaSuccess,
                "A checkpoint CUDA submission error was lost or left stale state");
        require(arena->cache_available_bytes() == available,
                "Failed checkpoint leaked cache storage");
    }
    const auto unmatched = cache.lookup_prefix(keys, true);
    require(unmatched.gqa == 0 && unmatched.endpoints.empty(),
            "Failed checkpoint unexpectedly reached LMCache");
    auto version = export_prefix_view(lane.request, 0, 0, prompt_end, stream, &lane.draft);
    store(cache, keys, version);
    version = {};
    synchronize_cuda(stream, "Confirm successful checkpoint version release");
    require(arena->cache_available_bytes() == available, "Completed checkpoint retained its slice");
    require(LmCacheGroup(cache).lookup_prefix({&keys}, prompt_end, true).usable_gdn == prompt_end,
            "Same-thread publication failed after recoverable CUDA errors");
    std::puts("actual checkpoint allocation, second-copy and event failures drain; retry succeeds");
}

void check_worker_release(const Lane& lane, const LmCacheClient& cache,
                          const PrefixCacheKeys& keys, cudaStream_t stream) {
    server::BatchCache worker(std::make_unique<LmCacheClient>(cache), "", 1);
    auto pairing = std::make_shared<server::BatchStorePairing>();
    const std::vector rank_keys{std::make_shared<PrefixCacheKeys>(keys)};
    const auto arena = DeviceArena::current(0);
    const auto available = arena->cache_available_bytes();
    for (const bool retain : {false, true}) {
        auto view = export_prefix_view(lane.request, prompt_end, prompt_end - 1, prompt_end,
                                       stream, &lane.draft);
        const auto leased = arena->cache_available_bytes();
        require(leased < available && !view.transfer.allocations.empty(),
                "Checkpoint state version did not lease cache storage");
        auto alias = retain ? view.store_owner : nullptr;
        std::shared_ptr<void> reservation;
        require(worker.reserve_store(reservation), "Checkpoint worker declined reservation");
        auto stores = std::make_shared<std::atomic_uint64_t>(0);
        const auto caller = std::this_thread::get_id();
        std::promise<bool> done;
        auto& calls = physical_storage_calls();
        calls.watching = true;
        worker.store(
            pairing, rank_keys, nullptr, 0, std::move(view), std::move(reservation), stores, [&] {
                done.set_value(std::this_thread::get_id() != caller &&
                               arena->cache_available_bytes() == (retain ? leased : available) &&
                               cudaPeekAtLastError() == cudaSuccess);
            });
        const bool completed = done.get_future().get();
        calls.watching = false;
        require(completed && *stores == 1 && !worker.stores_pending() && calls.count == 0,
                "STORE worker acknowledged before returning its lease or changed physical storage");
        alias.reset();
        synchronize_cuda(stream, "Confirm checkpoint alias release");
        require(arena->cache_available_bytes() == available, "Last alias retained its cache slice");
    }
    std::puts("STORE worker returns its lease before acknowledging completion; aliases stay valid");
}

void check_continued_prefill(const Lane& cold, const Lane& warm,
                             const std::string& imported_target) {
    require(imported_target == prefix_fixture::target_digest(warm.request, imported),
            "Replayed GQA prefix was overwritten");
    compare_scores(cold.scores, warm.scores);
    require(prefix_fixture::target_digest(cold.request, prompt_end) ==
            prefix_fixture::target_digest(warm.request, prompt_end), "Final target state differs");
    require(prefix_fixture::recurrent_digest(cold.request, cold.request.work) ==
            prefix_fixture::recurrent_digest(warm.request, warm.request.work),
            "Prefill GDN/terminal state differs");
}

auto allocate_score_readbacks() {
    void* host = nullptr;
    check(cudaMallocHost(&host, sizeof(GeneratedLogprobReadback) * decoder_batch_max_lanes),
          "Allocate prompt score readbacks");
    return std::unique_ptr<GeneratedLogprobReadback, decltype(&cudaFreeHost)>(
        static_cast<GeneratedLogprobReadback*>(host), cudaFreeHost);
}

void check_model_state(const DecoderModel& model) {
    Allocations execution(0);
    execution.open_stream();
    const auto stream = execution.stream();
    PrefillWorkspace workspace(stream, batch_prefill_tile_rows);
    require(workspace.data(), "Native tile arena unavailable");
    auto* score_scratch = execution.allocate<std::byte>(generated_logprob_workspace_bytes());
    const auto score_host = allocate_score_readbacks();
    std::vector<int> prompt(prompt_end);
    for (int i = 0; i < prompt_end; ++i) prompt[i] = 1000 + (i * 77 + i / 11) % 610;
    check_key_provenance(prompt);
    const auto cache = prefix_fixture::client();
    check_guarded_ranges(cache, prompt, execution);
    const PrefixCacheKeys keys(prompt, prefix_fixture::run_salt("oracle-split"));
    const auto tile = [&](Lane& lane, int first, int rows, int terminal, int score = -1) {
        PromptLogprobCapture capture{&prompt, &lane.scores, score_host.get(), score_scratch,
                                     3, score};
        model.prefill_chunk(lane.request, prompt.data() + first, rows, first, workspace,
                            &lane.draft, terminal - first, nullptr,
                            score >= 0 ? &capture : nullptr);
    };
    RequestBlockCache retired(0, true);
    Lane cold(model, stream, 1280, &retired);
    tile(cold, 0, 1024, checkpoint);
    tile(cold, 1024, checkpoint - 1024, checkpoint);
    checkpoint_copy_bytes = 0;
    auto prime = export_prefix_view(cold.request, 0, 0, checkpoint, stream, &cold.draft);
    require(std::exchange(checkpoint_copy_bytes, std::nullopt) == 153964544,
            "Prefill snapshot copied outside the four mutable state planes");
    auto alias = export_prefix_view(cold.request, 0, 0, checkpoint, stream, &cold.draft);
    const auto prime_target = prefix_fixture::target_digest(cold.request, checkpoint);
    const auto prime_gdn = prefix_fixture::recurrent_digest(cold.request, cold.request.work);
    require(cold.draft.valid_prefix() == checkpoint - 1, "Wrong prime shifted extent");
    const auto extend = [&](Lane& lane, bool warm) {
        model.seed_draft(lane.request.work, lane.draft, checkpoint - 1, prompt[checkpoint], stream);
        require(lane.draft.valid_prefix() == checkpoint, "Wrong predecessor seam after reseed");
        measured_pages = warm ? lane.request.pages.device_pages : nullptr;
        tile(lane, checkpoint, 1, prompt_end, checkpoint);
        if (warm) expect_appends("cached_scalar", 0, 0);
        tile(lane, checkpoint + 1, 36, prompt_end, 1313);
        if (warm) expect_appends("cached_tile", 0, 0);
        tile(lane, 1316, 112, prompt_end, 1417);
        if (warm) expect_appends("crossing_tile", imported, 1428 - imported);
        tile(lane, 1428, 1, prompt_end);
        if (warm) expect_appends("uncovered_scalar", 1428, 1);
        measured_pages = nullptr;
        require(lane.draft.valid_prefix() == prompt_end - 1, "Wrong continuation shifted extent");
    };
    check_detached_growth(cold, retired, cache, keys, std::move(alias), stream);
    extend(cold, false);
    require(prime_gdn != prefix_fixture::recurrent_digest(cold.request, cold.request.work),
            "Continuation did not overwrite the checkpoint source");
    store(cache, keys, prime);
    prime = {};
    retired.clear();
    require(retired.unavailable_bytes() == 0, "Acknowledged STORE left detached bytes charged");
    std::cout << "prefill D2D 153964544 target retained " << old_slab_bytes
              << " across two views and failed STORE\n";
    auto available = export_prefix_view(cold.request, 0, imported - 1, imported, stream,
                                        &cold.draft);
    available.layout.gdn = {};
    store(cache, keys, available);
    const auto read = LmCacheGroup(cache).lookup_prefix({&keys}, prompt_end, true);
    require(read.usable_gqa == imported && read.matched_gdn == checkpoint &&
                read.usable_gdn == checkpoint && read.usable_draft == checkpoint - 1,
            "Unexpected split-family daemon lookup");
    Lane warm(model, stream);
    require(restore(cache, keys, warm.request, read, stream, &warm.draft) == checkpoint,
            "Mixed family coherent endpoint differs");
    require(warm.request.cached_gqa_tokens == imported &&
            warm.draft.valid_prefix() == checkpoint - 1, "Independent restore endpoints differ");
    require(prime_target == prefix_fixture::target_digest(warm.request, checkpoint) &&
            prime_gdn == prefix_fixture::recurrent_digest(warm.request, warm.request.work),
            "Restored device checkpoint differs");
    const auto imported_target = prefix_fixture::target_digest(warm.request, imported);
    require(imported_target == prefix_fixture::target_digest(cold.request, imported),
            "Imported target prefix differs");
    extend(warm, true);
    synchronize_cuda(stream, "Complete warm split-family prefill");
    check_continued_prefill(cold, warm, imported_target);
    check_terminal_detach(model, stream, cache, keys,
        PrefixCacheKeys(prompt, prefix_fixture::run_salt("oracle-detached-terminal")), read,
        cold, retired, extend);
    const PrefixCacheKeys delayed_keys(prompt,
                                       prefix_fixture::run_salt("oracle-delayed-checkpoint"));
    check_delayed_checkpoint(model, cache, keys, delayed_keys, stream, read);
    check_checkpoint_faults(cold, cache,
        PrefixCacheKeys(prompt, prefix_fixture::run_salt("oracle-copy-faults")), stream);
    check_worker_release(cold, cache,
        PrefixCacheKeys(prompt, prefix_fixture::run_salt("oracle-worker-release")), stream);
    check_history_gap(cache, prompt, model, stream, cold);
    std::cout << "split gqa " << read.usable_gqa << " gdn " << read.usable_gdn << " draft "
              << read.usable_draft << '\n';

    compare_generation(generate(model, cold, prompt_end, stream),
                       generate(model, warm, prompt_end, stream));
    check_replay_before_export({model, execution, workspace, cache}, false);
    check_replay_before_export({model, execution, workspace, cache}, true);
}

int main(int argc, char** argv) try {
    require(argc == 2, "checkpoint path required");
    require(tokenizer::sha256("abc") ==
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA-256 differs");
    check(cudaSetDevice(0), "Select leased device");
    const auto arena = DeviceArena::start(0, default_gpu_memory_utilization);
    DecoderModel model(argv[1], 0);
    model.mtp_weights = std::make_shared<MtpDraftWeights>(
        argv[1], 0, model.memory, model.embedding, model.lm_head, model.backend.draft);
    model.tc_vocab = model.mtp_weights->shared_vocab();
    arena->commit(arena->startup_remaining_bytes() / 2);
    require(model.is_b200() && model.mtp_weights, "Requires real B200 and K4 weights");
    check_model_state(model);
    std::puts("PASS");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "model probe: %s\n", error.what());
    return 1;
}
