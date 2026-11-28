#include "sglang/decoder_batch.h"
#include "sglang/runtime/backend/tensor_parallel.h"
#include "sglang/runtime/cache/prefix_payload.h"

#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace sglang;
using namespace sglang::decoder;
using Clock = std::chrono::steady_clock;
using Selection = std::array<bool, decoder_batch_max_lanes>;

struct Export {
    cudaIpcMemHandle_t handle;
    void* base;
};
struct ReplayCounts {
    int submissions = 0;
    int rows = 0;
    int collectives = 0;
    int two_shot = 0;
};

struct GraphProbe {
    std::vector<Export> exports;
    bool record_exports = false;
    ReplayCounts counts;
};

GraphProbe& graph_probe() {
    static GraphProbe probe;
    return probe;
}

struct ReplayCase {
    int world;
    int width;
    int depth;
    bool sample;
    bool graphs;
    std::string_view shape;
};

struct Admission {
    int slot;
    int length;
};

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

unsigned char* plane_address(const PrefixDeviceTransfer& view, PrefixDeviceAddress span,
                             std::size_t bytes) {
    const auto& allocation = view.transfer.allocations.at(span.allocation);
    require(span.offset <= allocation.bytes && bytes <= allocation.bytes - span.offset,
            "Synthetic restore exceeds its exported allocation");
    const auto& exports = graph_probe().exports;
    const auto address = std::find_if(exports.begin(), exports.end(), [&](const Export& item) {
        return std::memcmp(&item.handle, allocation.handle.data(), sizeof(item.handle)) == 0;
    });
    require(address != exports.end(), "Synthetic restore did not export this allocation");
    return static_cast<unsigned char*>(address->base) + span.offset;
}

void fill_kv(const PrefixDeviceTransfer& view, const PrefixDeviceMxfp8& kv, int slot, int rank,
             bool draft) {
    if (kv.end == 0)
        return;
    require(kv.begin == 0 && kv.physical_pages.empty(), "Unexpected synthetic prefix page map");
    const auto pages = static_cast<std::size_t>(kv.layers) * kv.page_count;
    const auto page_elements = static_cast<std::size_t>(kv.kv_heads) * mxfp8_page_size;
    const auto payload = page_elements * mxfp8_head_dim;
    const auto scales = page_elements * mxfp8_groups;
    const int variation = (slot + (draft ? 0 : rank)) % 4;
    for (int plane = 0; plane < 2; ++plane) {
        auto* data = plane_address(view, kv.planes[plane], pages * payload);
        auto* scale = plane_address(view, kv.planes[plane + 2], pages * scales);
        const auto value = 0x18U + (static_cast<unsigned>(variation + plane) % 4U);
        cuda_check(cudaMemset(data, static_cast<int>(value), pages * payload),
                   "Fill synthetic KV pages");
        if (pages > 1) {
            cuda_check(cudaMemset2D(data + payload, 2 * payload, static_cast<int>(value | 0x80U),
                                    payload, pages / 2),
                       "Distinguish alternating physical KV pages");
        }
        cuda_check(cudaMemset(scale, 0x7f, pages * scales), "Fill synthetic KV scales");
    }
    std::cout << "plane\t" << slot << '\t' << (draft ? "draft" : "target") << '\t' << rank << '\t'
              << pages << '\t' << payload << '\t' << scales << '\n';
}

std::shared_future<bool> restore(const PrefixDeviceTransfer& view, int slot,
                                 const ReplayCase& config, int end) {
    int caller = -1;
    cuda_check(cudaGetDevice(&caller), "Identify synthetic restore caller");
    for (int rank = 0; rank < config.world; ++rank) {
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, rank), "Identify synthetic restore rank");
        if (std::memcmp(view.transfer.device.data(), properties.uuid.bytes, 16) != 0)
            continue;
        require(view.layout.gqa.end == end && view.layout.gdn.end == end,
                "Synthetic restore lost its target checkpoint");
        cuda_check(cudaSetDevice(rank), "Select synthetic restore rank");
        cuda_check(cudaEventSynchronize(static_cast<cudaEvent_t>(view.ready.get())),
                   "Wait for synthetic prefix storage");
        fill_kv(view, view.layout.gqa, slot, rank, false);
        fill_kv(view, view.layout.draft, slot, rank, true);
        for (const auto& span : view.layout.gdn.planes) {
            cuda_check(cudaMemset(plane_address(view, span.address, span.bytes), 0, span.bytes),
                       "Fill synthetic recurrent snapshot");
        }
        cuda_check(cudaDeviceSynchronize(), "Commit synthetic prefix storage");
        cuda_check(cudaSetDevice(caller), "Restore synthetic caller device");
        std::cout << "restore\t" << slot << '\t' << rank << '\t' << end << '\t'
                  << view.layout.gqa.kv_heads << '\t' << view.layout.draft.end << '\t'
                  << view.layout.draft.kv_heads << '\n';
        std::promise<bool> result;
        result.set_value(true);
        return result.get_future().share();
    }
    throw std::runtime_error("Synthetic restore was requested on an unknown rank");
}

std::uint64_t device_clock() {
    using Timestamp = std::uint64_t (*)();
    static const auto timestamp =
        reinterpret_cast<Timestamp>(dlsym(RTLD_DEFAULT, "batch_graph_activity_timestamp"));
    return timestamp != nullptr ? timestamp() : 0;
}

void trace_devices(int world) {
    using Identify = std::uint32_t (*)();
    const auto identify =
        reinterpret_cast<Identify>(dlsym(RTLD_DEFAULT, "batch_graph_activity_device"));
    if (identify == nullptr)
        return;
    int caller = -1;
    cuda_check(cudaGetDevice(&caller), "Identify trace caller");
    for (int rank = 0; rank < world; ++rank) {
        cuda_check(cudaSetDevice(rank), "Select trace rank");
        std::cout << "device\t" << rank << '\t' << identify() << '\n';
    }
    cuda_check(cudaSetDevice(caller), "Restore trace caller device");
}

void memory(const char* stage, int index, const ReplayCase& config, std::size_t requests) {
    int caller = -1;
    cuda_check(cudaGetDevice(&caller), "Identify memory snapshot caller");
    for (int rank = 0; rank < config.world; ++rank) {
        cuda_check(cudaSetDevice(rank), "Select memory snapshot rank");
        std::size_t free = 0;
        std::size_t total = 0;
        cuda_check(cudaMemGetInfo(&free, &total), "Read rank memory snapshot");
        std::cout << "memory\t" << stage << '\t' << index << '\t' << rank << '\t' << requests
                  << '\t' << free << '\t' << total << '\n';
    }
    cuda_check(cudaSetDevice(caller), "Restore memory snapshot caller");
}

int prompt_length(std::string_view shape, int ordinal) {
    if (shape == "short")
        return 127 + (ordinal % 3);
    if (shape == "split")
        return 2039 + (ordinal % 3);
    if (shape == "edge")
        return 32761 + (ordinal % 3);
    if (shape == "max")
        return 262144 - 26;
    if (shape == "mixed-max") {
        if (ordinal == 0)
            return 262144 - 26;
        if (ordinal <= 2)
            return 32760 + ordinal;
        return 127 + (ordinal % 3);
    }
    constexpr std::array lengths{32761, 127, 32771, 128, 16382};
    return lengths[ordinal % lengths.size()];
}

std::uint32_t bits(const Selection& selection) {
    std::uint32_t result = 0;
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        if (selection[slot])
            result |= std::uint32_t{1} << static_cast<unsigned>(slot);
    }
    return result;
}

constexpr int output_budget = 26;

void admit(DecoderBatch& runner, Admission admission, const ReplayCase& config) {
    const int slot = admission.slot;
    const int length = admission.length;
    constexpr std::array pool{
        13,   198, 279,  826,   760, 6511, 314, 9338, 369,
        1570, 315, 4950, 31784, 303, 806,  352, 952,  969,
    };
    std::vector<int> tokens;
    tokens.reserve(length);
    for (int row = 0; row < length; ++row) {
        tokens.push_back(pool[(row + (slot * 3)) % pool.size()]);
    }
    DecoderOptions options;
    options.max_new_tokens = output_budget;
    options.mtp_steps = config.depth;
    options.ignore_eos = true;
    options.publish_prefix_checkpoints = false;
    if (config.sample) {
        options.temperature = 0.8;
        options.top_k = 12;
        options.seed = 7103 + slot;
    }
    const int end = length - 2;
    auto read = std::make_shared<PrefixCacheRead>();
    read->matched_gqa = read->matched_gdn = read->usable_gqa = read->usable_gdn = end;
    read->matched_gdn_endpoints.push_back(end);
    if (config.depth != 0) {
        read->usable_draft = read->matched_draft = end - 1;
        read->draft_ready_gdn_endpoints.push_back(end);
    }
    options.prefix_cache_read = std::move(read);
    options.restore_prefix = [slot, config, end](const PrefixDeviceTransfer& view) {
        return restore(view, slot, config, end);
    };
    auto& probe = graph_probe();
    probe.exports.clear();
    probe.record_exports = true;
    runner.admit(slot, std::move(tokens), options);
    probe.record_exports = false;
    const auto update = runner.poll_prefix(slot);
    require(!update.pending && update.restored == end, "Synthetic checkpoint was not restored");
    std::cout << "admit\t" << slot << '\t' << length << '\t' << output_budget << '\n';
}

void exercise(const Decoder& decoder, const ReplayCase& config) {
    DecoderBatch runner(decoder, {
                                     .cuda_graphs = config.graphs,
                                     .max_lanes = decoder_batch_max_lanes,
                                     .max_num_batched_tokens = decoder_batch_max_step_tokens,
                                     .tp_verification_graphs = config.graphs,
                                 });
    Selection active{};
    std::array<int, decoder_batch_max_lanes> lengths{};
    std::array<int, decoder_batch_max_lanes> emitted{};
    for (int ordinal = 0; ordinal < config.width; ++ordinal) {
        const int slot = ((ordinal * 7) + 3) % decoder_batch_max_lanes;
        active[slot] = true;
        lengths[slot] = prompt_length(config.shape, ordinal);
        admit(runner, {.slot = slot, .length = lengths[slot]}, config);
    }
    const auto& counts = graph_probe().counts;
    memory("admitted", -1, config, runner.request_device_allocation_bytes());
    for (int step = 0; std::any_of(active.begin(), active.end(), [](bool live) { return live; });
         ++step) {
        require(step <= output_budget, "Synthetic requests did not make progress");
        const auto begin = device_clock();
        const auto start = Clock::now();
        const auto before = counts;
        DecoderBatchStepStats stats;
        const auto result = runner.advance_chunked(active, &stats);
        const auto elapsed = std::chrono::duration<double, std::micro>(Clock::now() - start);
        const auto end = device_clock();
        const auto selected = bits(active);
        std::cout << "step\t" << step << '\t' << selected << '\t' << stats.graph_slots << '\t'
                  << stats.verification_graph_slots << '\t' << stats.target_rows << '\t'
                  << stats.verification_lanes << '\t' << stats.verification_rows << '\t' << begin
                  << '\t' << end << '\t' << elapsed.count() << '\t' << stats.peak_request_bytes
                  << '\t' << stats.prefill_workspace_bytes << '\t'
                  << (counts.submissions - before.submissions) << '\t'
                  << (counts.rows - before.rows) << '\t'
                  << (counts.collectives - before.collectives) << '\t'
                  << (counts.two_shot - before.two_shot) << '\n';
        require(((stats.graph_slots | stats.verification_graph_slots) & ~selected) == 0,
                "Graph replay included an inactive slot");
        for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
            const auto& output = result[slot];
            require(!output.sampling_failed && !output.constraint_failed && !output.prefix_pending,
                    "Synthetic request failed");
            if (!active[slot]) {
                require(!output.token && !output.decode && !output.finish_reason,
                        "Inactive synthetic request advanced");
                continue;
            }
            require(output.token.has_value() && output.decode == (step != 0),
                    "Unexpected synthetic step output");
            require(stats.prefill_rows[slot] == (step == 0 ? 2 : 0),
                    "Synthetic fixture unexpectedly recomputed prefix history");
            require(output.extra_token_count >= 0 && output.extra_token_count <= config.depth,
                    "Invalid synthetic speculative output width");
            const int input_position = lengths[slot] + emitted[slot] - 1;
            std::cout << "lane\t" << step << '\t' << slot << '\t' << input_position << '\t'
                      << output.mtp_drafted_tokens << '\t' << output.mtp_accepted_tokens << '\t'
                      << output.mtp_verified_rows << '\t' << output.mtp_replayed_rows << '\t'
                      << output.mtp_boundary << '\t' << output.finish_reason.has_value() << '\t'
                      << output.token.value_or(-1);
            for (int index = 0; index < output.extra_token_count; ++index) {
                std::cout << '\t' << output.extra_tokens[index];
            }
            std::cout << '\n';
            emitted[slot] += 1 + output.extra_token_count;
            require(emitted[slot] <= output_budget, "Synthetic request exceeded its output budget");
            if (output.finish_reason) {
                require(*output.finish_reason == DecoderFinishReason::length &&
                            emitted[slot] == output_budget,
                        "Synthetic request stopped incorrectly");
                active[slot] = false;
            }
        }
        memory("step", step, config, runner.request_device_allocation_bytes());
    }
    runner.release_retired_allocations();
    require(runner.request_device_allocation_bytes() == 0, "Retired lanes retained request state");
    memory("retired", -1, config, runner.request_device_allocation_bytes());
}

}

extern "C" cudaError_t real_ipc_handle(cudaIpcMemHandle_t*,
                                       void*) asm("__real_cudaIpcGetMemHandle");
extern "C" cudaError_t wrap_ipc_handle(cudaIpcMemHandle_t* handle,
                                       void* base) asm("__wrap_cudaIpcGetMemHandle");
extern "C" cudaError_t wrap_ipc_handle(cudaIpcMemHandle_t* handle, void* base) {
    const auto status = real_ipc_handle(handle, base);
    auto& probe = graph_probe();
    if (probe.record_exports && status == cudaSuccess)
        probe.exports.push_back({*handle, base});
    return status;
}

extern "C" void real_replay(TensorParallel* group) asm(
    "__real__ZN6sglang7decoder14TensorParallel25record_fused_graph_replayEv") __attribute__((weak));
extern "C" void wrap_replay(TensorParallel* group) asm(
    "__wrap__ZN6sglang7decoder14TensorParallel25record_fused_graph_replayEv");
extern "C" void wrap_replay(TensorParallel* group) {
    ++graph_probe().counts.submissions;
    real_replay(group);
}

extern "C" void real_replay_rows(TensorParallel* group, int rows, int collectives) asm(
    "__real__ZN6sglang7decoder14TensorParallel25record_fused_graph_replayEii")
    __attribute__((weak));
extern "C" void wrap_replay_rows(TensorParallel* group, int rows, int collectives) asm(
    "__wrap__ZN6sglang7decoder14TensorParallel25record_fused_graph_replayEii");
extern "C" void wrap_replay_rows(TensorParallel* group, int rows, int collectives) {
    auto& counts = graph_probe().counts;
    ++counts.submissions;
    counts.rows += rows;
    counts.collectives += collectives;
    real_replay_rows(group, rows, collectives);
}

extern "C" void real_replay_transport(
    TensorParallel* group, int rows, int collectives,
    int two_shot) asm("__real__ZN6sglang7decoder14TensorParallel25record_fused_graph_replayEiii")
    __attribute__((weak));
extern "C" void wrap_replay_transport(
    TensorParallel* group, int rows, int collectives,
    int two_shot) asm("__wrap__ZN6sglang7decoder14TensorParallel25record_fused_graph_replayEiii");
extern "C" void wrap_replay_transport(TensorParallel* group, int rows, int collectives,
                                      int two_shot) {
    auto& counts = graph_probe().counts;
    ++counts.submissions;
    counts.rows += rows;
    counts.collectives += collectives;
    counts.two_shot += two_shot;
    real_replay_transport(group, rows, collectives, two_shot);
}

int main(int argc, const char* const* argv) try {
    require(argc == 8, "Expected CHECKPOINT TP WIDTH DEPTH eager|graph greedy|sample SHAPE");
    const int world = std::stoi(argv[2]);
    const int width = std::stoi(argv[3]);
    const int depth = std::stoi(argv[4]);
    const std::string_view path = argv[5];
    const std::string_view mode = argv[6];
    const std::string_view shape = argv[7];
    require((world == 1 || world == 2) && width >= 1 && width <= decoder_batch_max_lanes &&
                (depth == 0 || depth == 1 || depth == 4) && (path == "graph" || path == "eager") &&
                (mode == "sample" || mode == "greedy") &&
                (shape == "short" || shape == "split" || shape == "edge" || shape == "max" ||
                 shape == "mixed" || shape == "mixed-max"),
            "Invalid synthetic replay case");
    std::cout << std::unitbuf;
    std::cout << "case\t" << world << '\t' << width << '\t' << depth << '\t' << path << '\t' << mode
              << '\t' << shape << '\n';
    const Decoder decoder(argv[1], 0, depth != 0, world);
    trace_devices(world);
    require(decoder.uses_b200_operators(), "Expected native Blackwell operators");
    const ReplayCase config{world, width, depth, mode == "sample", path == "graph", shape};
    memory("model", -1, config, 0);
    exercise(decoder, config);
    memory("released", -1, config, 0);
} catch (const std::exception& error) {
    graph_probe().record_exports = false;
    std::cerr << "graph-generalization: " << error.what() << '\n';
    return 1;
}
