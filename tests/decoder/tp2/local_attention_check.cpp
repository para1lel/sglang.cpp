#include "sglang/text/gdn_local.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace sglang;
namespace dec = sglang::decoder;
using Bf16 = __nv_bfloat16;

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class Array {
    static constexpr std::size_t guard = 256;
    unsigned char* storage_ = nullptr;
    std::size_t count_;

public:
    explicit Array(std::size_t count, int poison = 0xff) : count_(count) {
        check(cudaMallocManaged(&storage_, count * sizeof(T) + 2 * guard), "allocate");
        std::memset(storage_, 0xa5, guard);
        std::memset(data(), poison, count * sizeof(T));
        std::memset(storage_ + guard + count * sizeof(T), 0xa5, guard);
    }
    ~Array() { cudaFree(storage_); }
    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;
    T* data() { return reinterpret_cast<T*>(storage_ + guard); }
    void check_guards() const {
        for (std::size_t i = 0; i < guard; ++i) {
            if (storage_[i] != 0xa5 || storage_[guard + count_ * sizeof(T) + i] != 0xa5) {
                throw std::runtime_error("allocation guard changed");
            }
        }
    }
    std::vector<T> snapshot() {
        check_guards();
        return {data(), data() + count_};
    }
};

int full_channel(int channel, int rank, bool qkv, int local_width) {
    if (rank < 0) return channel;
    if (!qkv) return rank * local_width + channel;
    if (channel < 1024) return rank * 1024 + channel;
    if (channel < 2048) return 2048 + rank * 1024 + channel - 1024;
    return 4096 + rank * 3072 + channel - 2048;
}

float sample(std::uint32_t index, std::uint32_t salt) {
    auto x = (index + 1) * 0x9e3779b9u + salt * 0x85ebca6bu;
    x ^= x >> 16;
    x *= 0x21f0aaadu;
    x ^= x >> 15;
    return (static_cast<int>(x % 251) - 125) / 256.0f;
}

template <typename T>
void fill(Array<T>& values, int rows, int width, int inner, int rank, int salt,
          bool qkv = false) {
    const int full_width = rank < 0 ? width : width * 2;
    for (int row = 0; row < rows; ++row) {
        for (int ch = 0; ch < width; ++ch) {
            const auto full = row * full_width + full_channel(ch, rank, qkv, width);
            for (int i = 0; i < inner; ++i) {
                values.data()[(row * width + ch) * inner + i] = T(sample(full * inner + i, salt));
            }
        }
    }
}

template <typename T>
void compare(const char* field, const std::vector<T>& full, const std::vector<T>& local,
             int rows, int width, int inner, int rank, bool qkv = false) {
    if (rows * std::size_t(width) * inner != local.size() || local.size() * 2 != full.size()) {
        throw std::runtime_error(std::string(field) + " invalid snapshot dimensions");
    }
    for (int row = 0; row < rows; ++row) {
        for (int ch = 0; ch < width; ++ch) {
            const auto local_offset = (row * width + ch) * inner;
            const auto full_offset = (row * width * 2 + full_channel(ch, rank, qkv, width)) * inner;
            if (std::memcmp(full.data() + full_offset, local.data() + local_offset,
                            inner * sizeof(T)) != 0) {
                for (int i = 0; i < inner; ++i) {
                    if (std::memcmp(&full[full_offset + i], &local[local_offset + i], sizeof(T))) {
                        std::fprintf(stderr,
                            "%s rank=%d row=%d channel=%d inner=%d full=%g local=%g\n",
                            field, rank, row, ch, i, double(float(full[full_offset + i])),
                            double(float(local[local_offset + i])));
                        throw std::runtime_error("rank slice differs");
                    }
                }
            }
        }
    }
}

template <typename T>
void finite_nonzero(const std::vector<T>& values) {
    bool nonzero = false;
    for (const auto& value : values) {
        const auto number = float(value);
        if (!std::isfinite(number)) throw std::runtime_error("nonfinite result");
        nonzero |= number != 0;
    }
    if (!nonzero) throw std::runtime_error("all-zero result");
}

struct GdnResult {
    std::vector<Bf16> output, history, prefix_history;
    std::vector<float> recurrent, prefix_recurrent;
};

void verify_batch(bool local, int features, int heads, dec::GdnVerificationLane base,
                  TextGdnWeights weights) {
    dec::GdnVerificationBatch batch;
    const int width = heads * 128;
    for (int lane = 0; lane < batch.lanes; ++lane) {
        const int row = lane * batch.rows;
        const int hist = lane * features * 3;
        const int recur = lane * heads * 128 * 128;
        batch.entries[lane] = {
            {base.inputs.qkv + row * features, base.inputs.convolved + row * features,
             base.inputs.b + row * heads, base.inputs.a + row * heads}, base.z + row * width,
            {base.initial.history + hist, base.initial.recurrent + recur},
            {base.tail.history + hist, base.tail.recurrent + recur},
            base.observable + row * width, base.output + row * width};
    }
    const auto verify = local ? dec::decode_verify_gdn_batch_local : dec::decode_verify_gdn_batch;
    check(verify(batch, weights, nullptr), "GDN batched verification");
}

GdnResult gdn(int rows, int rank) {
    const bool local = rank >= 0;
    const int poison = local ? 0x7f : 0xff;
    const int features = local ? 5120 : 10240;
    const int heads = local ? 24 : 48;
    const int width = heads * 128;
    const int lanes = rows == 20 ? dec::GdnVerificationBatch::lanes : 1;
    Array<Bf16> qkv(rows * features), z(rows * width), b(rows * heads), a(rows * heads);
    Array<Bf16> conv(features * 4), decay(heads), bias(heads), norm(128);
    Array<Bf16> history(lanes * features * 3), tail_history(lanes * features * 3, poison);
    Array<float> state(lanes * heads * 128 * 128), tail_state(lanes * heads * 128 * 128, poison);
    Array<Bf16> workspace(rows * (features + width), poison), output(rows * width, poison);
    fill(qkv, rows, features, 1, rank, 1, true);
    fill(z, rows, width, 1, rank, 2);
    fill(b, rows, heads, 1, rank, 3);
    fill(a, rows, heads, 1, rank, 4);
    fill(conv, 1, features, 4, rank, 5, true);
    fill(decay, 1, heads, 1, rank, 6);
    fill(bias, 1, heads, 1, rank, 7);
    fill(norm, 1, 128, 1, -1, 8);
    fill(history, lanes, features, 3, rank, 9, true);
    fill(state, lanes, heads, 128 * 128, rank, 10);
    TextGdnWeights weights{conv.data(), decay.data(), bias.data(), norm.data()};
    Array<TextGdnState> bindings(2);
    bindings.data()[0] = {history.data(), state.data()};
    bindings.data()[1] = {tail_history.data(), tail_state.data()};
    GdnResult result;
    if (rows == 1) {
        const auto decode = local ? dec::decode_gdn_local : dec::decode_gdn;
        check(decode(qkv.data(), z.data(), b.data(), a.data(), weights, bindings.data(),
                     workspace.data(), output.data(), nullptr), "GDN decode");
    } else if (rows == 5) {
        const auto initial_h = history.snapshot();
        const auto initial_s = state.snapshot();
        TextGdnReplayRows replay{qkv.data(), workspace.data(), b.data(), a.data()};
        const auto verify = local ? dec::decode_verify_gdn_local : dec::decode_verify_gdn;
        check(verify(rows, replay, z.data(), weights, bindings.data(), bindings.data() + 1,
                     workspace.data() + rows * features, output.data(), nullptr), "GDN verify");
        check(cudaDeviceSynchronize(), "GDN verify completion");
        result.history = tail_history.snapshot();
        result.recurrent = tail_state.snapshot();
        Array<Bf16> prefix_h(features * 3, poison);
        Array<float> prefix_s(heads * 128 * 128, poison);
        const auto prefix = local ? text_gdn_replay_prefix_local : text_gdn_replay_prefix;
        check(prefix(2, replay, weights, bindings.data()[0], {prefix_h.data(), prefix_s.data()},
                     nullptr), "GDN prefix replay");
        check(cudaDeviceSynchronize(), "GDN replay completion");
        result.prefix_history = prefix_h.snapshot();
        result.prefix_recurrent = prefix_s.snapshot();
        if (std::memcmp(history.data(), initial_h.data(), initial_h.size() * sizeof(Bf16)) ||
            std::memcmp(state.data(), initial_s.data(), initial_s.size() * sizeof(float))) {
            throw std::runtime_error("GDN verification/replay mutated committed state");
        }
    } else if (lanes > 1) {
        const auto initial_h = history.snapshot();
        const auto initial_s = state.snapshot();
        verify_batch(local, features, heads,
                     {{qkv.data(), workspace.data(), b.data(), a.data()}, z.data(),
                      bindings.data()[0], bindings.data()[1], workspace.data() + rows * features,
                      output.data()}, weights);
        check(cudaDeviceSynchronize(), "GDN batched verification completion");
        result.history = tail_history.snapshot();
        result.recurrent = tail_state.snapshot();
        if (std::memcmp(history.data(), initial_h.data(), initial_h.size() * sizeof(Bf16)) ||
            std::memcmp(state.data(), initial_s.data(), initial_s.size() * sizeof(float))) {
            throw std::runtime_error("GDN batched verification mutated committed state");
        }
    } else {
        Array<float2> gates(rows * heads, poison);
        const auto scratch_bytes = ((rows + 31) / 32) * heads * (2 * 32 * 32 * 2 + 3 * 32 * 4);
        Array<std::uint8_t> scratch(scratch_bytes, poison);
        using Prefill = cudaError_t (*)(int, const Bf16*, const Bf16*, const Bf16*, const Bf16*,
            TextGdnWeights, TextGdnState, Bf16*, float2*, Bf16*, cudaStream_t, void*, std::size_t);
        Prefill prefill = local ? static_cast<Prefill>(dec::tensor_core_prefill_gdn_local) :
                                  static_cast<Prefill>(dec::tensor_core_prefill_gdn);
        check(prefill(rows, qkv.data(), z.data(), b.data(), a.data(), weights, bindings.data()[0],
                      workspace.data(), gates.data(), output.data(), nullptr, scratch.data(),
                      scratch_bytes), "GDN prefill");
        check(cudaDeviceSynchronize(), "GDN prefill completion");
        if (std::all_of(scratch.data(), scratch.data() + scratch_bytes,
                        [poison](auto byte) { return byte == poison; })) {
            throw std::runtime_error("GDN chunk scratch was unused");
        }
        gates.check_guards();
        scratch.check_guards();
    }
    check(cudaDeviceSynchronize(), "GDN completion");
    result.output = output.snapshot();
    finite_nonzero(result.output);
    if (result.history.empty()) {
        result.history = history.snapshot();
        result.recurrent = state.snapshot();
    }
    finite_nonzero(result.recurrent);
    workspace.check_guards();
    return result;
}

void gdn_check() {
    for (const int rows : {1, 5, 20, 128}) {
        const int lanes = rows == 20 ? dec::GdnVerificationBatch::lanes : 1;
        const auto full = gdn(rows, -1);
        for (int rank = 0; rank < 2; ++rank) {
            const auto local = gdn(rows, rank);
            compare("GDN output", full.output, local.output, rows, 3072, 1, rank);
            compare("GDN recurrence", full.recurrent, local.recurrent, lanes, 24, 128 * 128, rank);
            compare("GDN history", full.history, local.history, lanes, 5120, 3, rank, true);
            if (rows == 5) {
                compare("replayed recurrence", full.prefix_recurrent, local.prefix_recurrent,
                        1, 24, 128 * 128, rank);
                compare("replayed history", full.prefix_history, local.prefix_history,
                        1, 5120, 3, rank, true);
            }
            std::printf("GDN rows=%d rank=%d output=%zu history=%zu "
                        "recurrence=%zu prefix=%zu exact\n", rows, rank, local.output.size(),
                        local.history.size(), local.recurrent.size(),
                        local.prefix_history.size() + local.prefix_recurrent.size());
        }
    }
}

constexpr int kv_rows = 2052;
constexpr int pages = (kv_rows + 127) / 128 + 1;
constexpr int prefill_rows = 128;
constexpr int verify_rows = 5;

struct GqaResult {
    std::array<std::vector<std::uint8_t>, 4> cache;
    std::vector<Bf16> rope_queries, rope_keys, decode, split, verify, prefill, gated, bound;
};

std::vector<Bf16> bound_decode(Mxfp8KvCache cache, Mxfp8PageTable table, const Bf16* query,
                               int width, int poison, void* scratch) {
    struct Bindings {
        Mxfp8KvCache cache;
        Mxfp8PageTable table;
        int position;
        dec::ExactGqaWorkspace workspace;
    };
    Array<float> scores(kv_rows * width / 256, poison);
    Array<Bf16> output(2 * width, poison);
    Bindings* raw = nullptr;
    check(cudaMalloc(&raw, sizeof(*raw)), "allocate device-only GQA descriptors");
    std::unique_ptr<Bindings, decltype(&cudaFree)> device(raw, cudaFree);
    for (int row = 0; row < 2; ++row) {
        const Bindings host{cache, table, row == 0 ? 130 : 2047, {scores.data(), kv_rows}};
        check(cudaMemcpy(device.get(), &host, sizeof(host), cudaMemcpyHostToDevice),
              "bind GQA descriptors");
        dec::BoundPagedDecodeLane lane{
            &raw->cache, &raw->table, &raw->position, &raw->workspace,
            {query, output.data() + row * width, scratch}};
        check(dec::target_split_gqa(lane, nullptr), "GQA bound decode");
        check(cudaDeviceSynchronize(), "GQA bound completion");
    }
    scores.check_guards();
    return output.snapshot();
}

GqaResult gqa(int rank) {
    const bool local = rank >= 0;
    const int poison = local ? 0x7f : 0xff;
    const int heads = local ? 2 : 4;
    const int queries = heads * 6;
    const int width = queries * 256;
    const auto geometry = local ? gqa_rank_geometry : gqa_full_geometry;
    Array<Bf16> k(kv_rows * heads * 256), v(kv_rows * heads * 256), q(prefill_rows * width);
    fill(k, kv_rows, heads, 256, rank, 11);
    fill(v, kv_rows, heads, 256, rank, 12);
    Array<Bf16> q_gate(prefill_rows * queries * 512), q_norm(256), k_norm(256);
    Array<Bf16> rope_keys(kv_rows * heads * 256, poison);
    Array<Bf16> prefill_keys(prefill_rows * heads * 256, poison);
    fill(q_gate, prefill_rows, queries, 512, rank, 13);
    fill(q_norm, 1, 256, 1, -1, 14);
    fill(k_norm, 1, 256, 1, -1, 15);
    check(text_gqa_preprocess_keys(0, kv_rows, k.data(), k_norm.data(), rope_keys.data(),
                                   nullptr, geometry), "GQA key RoPE");
    check(text_gqa_preprocess(61, prefill_rows, q_gate.data(), k.data() + 61 * heads * 256,
                             q_norm.data(), k_norm.data(), q.data(), prefill_keys.data(),
                             nullptr, geometry), "GQA RoPE");
    Array<int> map(pages - 1);
    for (int i = 0; i < pages - 1; ++i) map.data()[i] = pages - 1 - i;
    const Mxfp8PageTable table{map.data(), pages - 1};
    Array<std::uint8_t> kp(pages * 128 * heads * 256, 0xa5), vp(pages * 128 * heads * 256, 0xa5);
    Array<std::uint8_t> ks(pages * heads * 128 * 8, 0xa5), vs(pages * heads * 128 * 8, 0xa5);
    const Mxfp8KvCache cache{kp.data(), vp.data(), ks.data(), vs.data(), pages, heads};
    for (int start : {0, 127}) {
        const int count = start == 0 ? 127 : kv_rows - start;
        check(mxfp8_kv_append(cache, table, start, count, rope_keys.data() + start * heads * 256,
                              v.data() + start * heads * 256, nullptr, geometry), "KV append");
    }
    Array<Bf16> decode(width, poison), split(width, poison), verify(verify_rows * width, poison);
    Array<Bf16> prefill(prefill_rows * width, poison), gated(prefill_rows * width, poison);
    Array<int> position(1);
    *position.data() = 130;
    check(dec::decode_gqa_attention(cache, table, position.data(), q.data(), decode.data(),
                                    nullptr), "GQA decode");
    Array<std::uint8_t> scratch(verify_rows * dec::paged_decode_scratch_bytes, poison);
    dec::PagedDecodeLane lane{cache, table, 2047, {q.data(), split.data(), scratch.data()}};
    check(dec::target_split_gqa(lane, nullptr), "GQA split decode");
    std::array<dec::PagedDecodeLane, verify_rows> lanes;
    for (int i = 0; i < verify_rows; ++i) {
        lanes[i] = {cache, table, 2047 + i, {q.data() + i * width, verify.data() + i * width,
                    scratch.data() + i * dec::paged_decode_scratch_bytes}};
    }
    check(dec::target_split_gqa(lanes.data(), verify_rows, dec::target_split_end_position, nullptr),
          "GQA split verification");
    check(dec::tensor_core_multirow_prefill_gqa(cache, table, 61, prefill_rows, q.data(),
                                               prefill.data(), nullptr, geometry), "GQA prefill");
    check(text_gqa_postgate(prefill_rows, q_gate.data(), prefill.data(), gated.data(),
                            nullptr, geometry), "GQA postgate");
    check(cudaDeviceSynchronize(), "GQA completion");
    if (std::memcmp(rope_keys.data() + 61 * heads * 256, prefill_keys.data(),
                    std::size_t(prefill_rows) * heads * 256 * sizeof(Bf16))) {
        throw std::runtime_error("combined preprocessing disagrees with key-only RoPE");
    }
    GqaResult result{{kp.snapshot(), vp.snapshot(), ks.snapshot(), vs.snapshot()},
                     q.snapshot(), rope_keys.snapshot(), decode.snapshot(), split.snapshot(),
                     verify.snapshot(), prefill.snapshot(), gated.snapshot(),
                     bound_decode(cache, table, q.data(), width, poison, scratch.data())};
    finite_nonzero(result.bound);
    for (int plane = 0; plane < 4; ++plane) {
        const int page_bytes = heads * 128 * (plane < 2 ? 256 : 8);
        const auto& bytes = result.cache[plane];
        const auto untouched = [](auto b) { return b == 0xa5; };
        if (!std::all_of(bytes.begin(), bytes.begin() + page_bytes, untouched) ||
            std::all_of(bytes.begin() + 2 * page_bytes, bytes.end(), untouched)) {
            throw std::runtime_error("KV physical page mapping");
        }
    }
    scratch.check_guards();
    finite_nonzero(result.decode);
    finite_nonzero(result.split);
    finite_nonzero(result.verify);
    finite_nonzero(result.prefill);
    finite_nonzero(result.gated);
    return result;
}

void gqa_check() {
    const auto full = gqa(-1);
    for (int rank = 0; rank < 2; ++rank) {
        const auto local = gqa(rank);
        for (int plane = 0; plane < 4; ++plane) {
            compare("GQA cache", full.cache[plane], local.cache[plane],
                    plane < 2 ? pages * 128 : pages, 2, plane < 2 ? 256 : 128 * 8, rank);
        }
        compare("GQA query RoPE", full.rope_queries, local.rope_queries, prefill_rows, 12,
                256, rank);
        compare("GQA key RoPE", full.rope_keys, local.rope_keys, kv_rows, 2, 256, rank);
        compare("GQA decode", full.decode, local.decode, 1, 12, 256, rank);
        compare("GQA split decode", full.split, local.split, 1, 12, 256, rank);
        compare("GQA bound exact and split", full.bound, local.bound, 2, 12, 256, rank);
        compare("GQA split verification", full.verify, local.verify, verify_rows, 12, 256, rank);
        compare("GQA prefill", full.prefill, local.prefill, prefill_rows, 12, 256, rank);
        compare("GQA postgate", full.gated, local.gated, prefill_rows, 12, 256, rank);
        std::printf("GQA rank=%d appended=%d physical_pages=%d payload_bytes=%zu scale_bytes=%zu "
                    "decode=%zu split=%zu verify=%zu prefill=%zu exact\n", rank, kv_rows, pages,
                    2 * local.cache[0].size(), 2 * local.cache[2].size(), local.decode.size(),
                    local.split.size(), local.verify.size(), local.prefill.size());
    }
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("supply gqa or gdn");
        if (std::string(argv[1]) == "gqa") gqa_check();
        else if (std::string(argv[1]) == "gdn") gdn_check();
        else throw std::runtime_error("unknown check");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }
}
