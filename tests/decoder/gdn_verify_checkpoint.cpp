#include "sglang/decoder_prefill.h"
#include "sglang/decoder_verify.h"
#include "sglang/runtime/backend/device_arena.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>

using namespace sglang;
using namespace sglang::decoder;

namespace {

constexpr int lanes = decoder_batch_max_mtp_lanes;
constexpr int rows = target_verify_max_rows;
constexpr int capacity = 256;

using Tokens = std::array<std::vector<int>, lanes>;

std::vector<int> ids(const char* path) {
    std::ifstream file(path);
    return {std::istream_iterator<int>(file), std::istream_iterator<int>()};
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Element>
std::vector<Element> read_device(const Element* pointer, std::size_t count) {
    std::vector<Element> values(count);
    check(cudaMemcpy(values.data(), pointer, count * sizeof(Element), cudaMemcpyDeviceToHost),
          "Read checkpoint evidence");
    return values;
}

template <class Element>
std::size_t equal_device(const Element* left, const Element* right, std::size_t count) {
    const auto lhs = read_device(left, count);
    const auto rhs = read_device(right, count);
    require(std::memcmp(lhs.data(), rhs.data(), count * sizeof(Element)) == 0,
            "committed state mismatch");
    return count * sizeof(Element);
}

std::size_t equal_state(Request& lhs, Request& rhs, int used, cudaStream_t stream) {
    lhs.materialize_gdn(stream);
    rhs.materialize_gdn(stream);
    check(cudaStreamSynchronize(stream), "Complete committed GDN materialization");
    std::size_t compared = 0;
    const auto left_pages = read_device(lhs.pages.device_pages, lhs.pages.page_count);
    const auto right_pages = read_device(rhs.pages.device_pages, rhs.pages.page_count);
    constexpr int page_size = mxfp8_page_size;
    constexpr int payload_per_row = mxfp8_payload_bytes_per_page / page_size;
    for (int logical = 0; logical * page_size < used; ++logical) {
        const int active = std::min(page_size, used - logical * page_size);
        for (int layer = 0; layer < full_attention_layers; ++layer) {
            const auto& left = lhs.caches[layer];
            const auto& right = rhs.caches[layer];
            for (const auto field : {&Mxfp8KvCache::key_payload, &Mxfp8KvCache::value_payload}) {
                compared += equal_device(
                    left.*field + left_pages[logical] * mxfp8_payload_bytes_per_page,
                    right.*field + right_pages[logical] * mxfp8_payload_bytes_per_page,
                    active * payload_per_row);
            }
            for (const auto field : {&Mxfp8KvCache::key_scales, &Mxfp8KvCache::value_scales}) {
                const auto left_scale = read_device(
                    left.*field + left_pages[logical] * mxfp8_scale_bytes_per_page,
                    mxfp8_scale_bytes_per_page);
                const auto right_scale = read_device(
                    right.*field + right_pages[logical] * mxfp8_scale_bytes_per_page,
                    mxfp8_scale_bytes_per_page);
                for (int row = 0; row < active; ++row) {
                    for (int head = 0; head < mxfp8_kv_heads; ++head) {
                        const int lane = head * mxfp8_group_size + row % mxfp8_group_size;
                        const int group = lane * (page_size / mxfp8_group_size) +
                                          row / mxfp8_group_size;
                        const int offset = group * mxfp8_groups;
                        require(std::memcmp(left_scale.data() + offset,
                                            right_scale.data() + offset, mxfp8_groups) == 0,
                                "committed cache scale mismatch");
                        compared += mxfp8_groups;
                    }
                }
            }
        }
    }
    compared += equal_device(lhs.states.front().history, rhs.states.front().history,
                             gdn_layers * gdn_history_elements);
    compared += equal_device(lhs.states.front().recurrent, rhs.states.front().recurrent,
                             gdn_layers * gdn_state_elements);
    return compared;
}

void equal_scores(const DecoderModel& model, TargetVerification& verifier, int row,
                  const Request& request, cudaStream_t stream) {
    const auto& work = *verifier.row(row).work;
    equal_device(work.normalized, request.work.normalized, hidden_size);
    const auto& head = model.lm_head;
    check(decode_bf16_lm_head(head.values, work.normalized, head.rows, head.columns, work.logits,
                              stream),
          "Score verifier row with scalar oracle head");
    check(cudaStreamSynchronize(stream), "Complete scalar oracle head");
    equal_device(work.logits, request.work.logits, vocab_size);
}

void seed(const DecoderModel& model, Request& request, const std::vector<int>& prompt,
          int prompt_length, cudaStream_t stream) {
    request.reset(stream);
    PrefillWorkspace work(stream, prompt_length);
    model.prefill_chunk(request, prompt.data(), prompt_length, 0, work);
    work.close();
}

void advance(const DecoderModel& model, Request& request, int token, int position,
             cudaStream_t stream) {
    const ForwardLane lane{&request, token, position, TextGdnMode::decode};
    model.forward(&lane, 1, stream);
    model.compute_logits(request.work, stream, true);
    check(cudaStreamSynchronize(stream), "Finish scalar oracle decode");
}

void test(const DecoderModel& model, const Tokens& prompts, int base_cut, bool permute) {
    Allocations execution(0);
    execution.open_stream(model.memory);
    const auto stream = execution.stream();
    std::array<std::unique_ptr<Request>, lanes> candidate, oracle;
    std::array<std::unique_ptr<TargetVerification>, lanes> target;
    std::array<VerificationSequence, lanes> sequences;
    for (int lane = 0; lane < lanes; ++lane) {
        const int prompt_length = 126 + lane;
        candidate[lane] = std::make_unique<Request>(0);
        oracle[lane] = std::make_unique<Request>(0);
        candidate[lane]->initialize(capacity, stream);
        oracle[lane]->initialize(capacity, stream);
        seed(model, *candidate[lane], prompts[lane], prompt_length, stream);
        seed(model, *oracle[lane], prompts[lane], prompt_length, stream);
        target[lane] = std::make_unique<TargetVerification>(
            *candidate[lane], capacity, prompt_length, rows, true, stream);
        const int sequence = permute ? lanes - 1 - lane : lane;
        sequences[sequence] = {target[lane].get(), prompt_length,
                               prompts[lane].data() + prompt_length, rows};
    }
    require(!model.forward_verify(sequences.data(), lanes, stream, nullptr, nullptr),
            "expected eager four-lane verification");
    std::size_t compared = 0;
    for (int lane = 0; lane < lanes; ++lane) {
        const int prompt_length = 126 + lane;
        auto& verifier = *target[lane];
        verifier.read();
        const auto* tokens = prompts[lane].data() + prompt_length;
        for (int row = 0; row < rows; ++row) {
            advance(model, *oracle[lane], tokens[row], prompt_length + row, stream);
            equal_scores(model, verifier, row, *oracle[lane], stream);
        }
        const int cut = (base_cut + lane) % (rows + 1);
        verifier.commit(cut);
        if (cut != rows) {
            seed(model, *oracle[lane], prompts[lane], prompt_length, stream);
            for (int row = 0; row < cut; ++row) {
                advance(model, *oracle[lane], tokens[row], prompt_length + row, stream);
            }
        }
        compared += equal_state(*oracle[lane], *candidate[lane], prompt_length + cut, stream);

        const auto& neighbor = prompts[(lane + 1) % lanes];
        const auto different = std::find_if(neighbor.begin() + prompt_length + rows + base_cut,
                                            neighbor.end(), [&](int token) {
                                                return cut == rows || token != tokens[cut];
                                            });
        require(different != neighbor.end(), "correction token must differ");
        const int correction = *different;
        const VerificationSequence replacement{&verifier, prompt_length + cut, &correction, 1};
        require(!model.forward_verify(&replacement, 1, stream, nullptr, nullptr),
                "replacement path");
        verifier.read();
        advance(model, *oracle[lane], correction, prompt_length + cut, stream);
        equal_scores(model, verifier, 0, *oracle[lane], stream);
        verifier.commit(1);
        compared += equal_state(*oracle[lane], *candidate[lane], prompt_length + cut + 1, stream);
    }
    std::cout << "case=" << base_cut << " permuted=" << permute << " score_rows="
              << lanes * (rows + 1) << " state_bytes=" << compared << std::endl;
}

void test_graph(const DecoderModel& model, const Tokens& prompts) {
    Allocations execution(0);
    execution.open_stream(model.memory);
    const auto stream = execution.stream();
    VerificationGraph graph(execution);
    constexpr int prompt_length = 127;
    constexpr int graph_rows = target_verify_max_rows;
    constexpr std::array<int, 3> cuts{graph_rows, 2, 0};
    for (int attempt = 0; attempt < 3; ++attempt) {
        Request candidate(0), oracle(0);
        candidate.initialize(capacity, stream);
        oracle.initialize(capacity, stream);
        const auto& prompt = prompts[attempt];
        seed(model, candidate, prompt, prompt_length, stream);
        seed(model, oracle, prompt, prompt_length, stream);
        TargetVerification verifier(candidate, capacity, prompt_length, graph_rows, true, stream);
        const auto* tokens = prompt.data() + prompt_length;
        const VerificationSequence sequence{&verifier, prompt_length, tokens, graph_rows};
        const bool replayed = model.forward_verify(&sequence, 1, stream, &graph, nullptr);
        require(replayed == (attempt != 0), "expected graph capture and reuse after eager warmup");
        verifier.read();
        for (int row = 0; row < graph_rows; ++row) {
            advance(model, oracle, tokens[row], prompt_length + row, stream);
            equal_scores(model, verifier, row, oracle, stream);
        }
        const int cut = cuts[attempt];
        verifier.commit(cut);
        if (cut != graph_rows) {
            seed(model, oracle, prompt, prompt_length, stream);
            for (int row = 0; row < cut; ++row) {
                advance(model, oracle, tokens[row], prompt_length + row, stream);
            }
        }
        equal_state(oracle, candidate, prompt_length + cut, stream);
    }
    std::cout << "five-row graph: eager warmup, capture, and reuse at cuts 5, 2, and 0 passed"
              << '\n';
}

}

int main(int argc, char** argv) {
    try {
        require(argc == lanes + 2, "expected checkpoint and four frozen token files");
        Tokens prompts;
        for (int lane = 0; lane < lanes; ++lane) {
            prompts[lane] = ids(argv[lane + 2]);
            require(prompts[lane].size() >= 145, "short prompt fixture");
        }
        const auto arena = DeviceArena::start(0, default_gpu_memory_utilization);
        const DecoderModel model(argv[1], 0);
        arena->commit(arena->startup_remaining_bytes() / 2);
        for (int cut = 0; cut <= rows; ++cut) {
            test(model, prompts, cut, cut % 2 != 0);
        }
        test_graph(model, prompts);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
