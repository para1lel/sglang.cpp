#include "decoder_batch.cpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>

namespace {

using namespace sglang;
using namespace sglang::decoder;

constexpr int page = mxfp8_page_size;

void require(bool condition, std::string_view what) {
    if (!condition) {
        throw std::runtime_error(std::string(what));
    }
}

std::vector<int> ids(const char* path, std::size_t expected) {
    std::ifstream input(path);
    std::vector<int> result;
    int token;
    while (input >> token) {
        result.push_back(token);
    }
    require(input.eof() && result.size() == expected, "Unexpected fixture size");
    return result;
}

template<class Element>
std::vector<Element> read_device(const Element* from, std::size_t count) {
    std::vector<Element> data(count);
    check(cudaMemcpy(data.data(), from, count * sizeof(Element), cudaMemcpyDeviceToHost),
          "Read device state");
    return data;
}

template<class Element>
void same(const Element* left, const Element* right, std::size_t count, std::string_view field) {
    const auto left_host = read_device(left, count);
    const auto right_host = read_device(right, count);
    require(std::memcmp(left_host.data(), right_host.data(), count * sizeof(Element)) == 0, field);
}

std::vector<std::int32_t> mapping(const Request& request) {
    return read_device(request.pages.device_pages, request.pages.page_count);
}

void remap(Request& request, bool reverse) {
    auto pages = mapping(request);
    if (reverse) {
        std::reverse(pages.begin(), pages.end());
    } else {
        std::rotate(pages.begin(), pages.begin() + 1, pages.end());
    }
    check(cudaMemcpy(const_cast<std::int32_t*>(request.pages.device_pages), pages.data(),
                     pages.size() * sizeof(pages.front()), cudaMemcpyHostToDevice),
          "Replace test request page map");
}

void compare_scales(const std::uint8_t* expected, const std::uint8_t* hit, int rows) {
    if (rows == page) {
        same(expected, hit, mxfp8_scale_bytes_per_page, "Full scale page");
        return;
    }
    const auto left = read_device(expected, mxfp8_scale_bytes_per_page);
    const auto right = read_device(hit, mxfp8_scale_bytes_per_page);
    for (int row = 0; row < rows; ++row) {
        for (int head = 0; head < mxfp8_kv_heads; ++head) {
            const int lane = head * mxfp8_group_size + row % mxfp8_group_size;
            const int group = lane * (page / mxfp8_group_size) + row / mxfp8_group_size;
            const int offset = group * mxfp8_groups;
            require(std::memcmp(left.data() + offset, right.data() + offset, mxfp8_groups) == 0,
                    "Active tail scale values");
        }
    }
}

void compare_kv(const Request& expected, const Request& hit, int length) {
    const auto left_map = mapping(expected);
    const auto right_map = mapping(hit);
    for (int layer = 0; layer < full_attention_layers; ++layer) {
        const auto& left = expected.caches[layer];
        const auto& right = hit.caches[layer];
        for (int logical = 0; logical * page < length; ++logical) {
            const int rows = std::min(page, length - logical * page);
            const auto left_payload = left_map[logical] * mxfp8_payload_bytes_per_page;
            const auto right_payload = right_map[logical] * mxfp8_payload_bytes_per_page;
            const auto left_scale = left_map[logical] * mxfp8_scale_bytes_per_page;
            const auto right_scale = right_map[logical] * mxfp8_scale_bytes_per_page;
            const auto count = rows * (mxfp8_payload_bytes_per_page / page);
            same(left.key_payload + left_payload, right.key_payload + right_payload, count,
                 "Key payload");
            same(left.value_payload + left_payload, right.value_payload + right_payload, count,
                 "Value payload");
            compare_scales(left.key_scales + left_scale, right.key_scales + right_scale, rows);
            compare_scales(left.value_scales + left_scale, right.value_scales + right_scale, rows);
        }
    }
}

void compare_state(const Request& expected, const Request& hit, int length, bool scored) {
    compare_kv(expected, hit, length);
    same(expected.states.front().history, hit.states.front().history,
         gdn_layers * gdn_history_elements, "GDN convolution history");
    same(expected.states.front().recurrent, hit.states.front().recurrent,
         gdn_layers * gdn_state_elements, "GDN recurrent state");
    if (scored) {
        same(expected.work.hidden, hit.work.hidden, hidden_size, "Last hidden");
        same(expected.work.residual, hit.work.residual, hidden_size, "Last residual");
        same(expected.work.logits, hit.work.logits, vocab_size, "All vocabulary BF16 logits");
    }
}

Selection slots(std::initializer_list<int> ids) {
    Selection result{};
    for (int slot : ids) {
        result.at(slot) = true;
    }
    return result;
}

Request& request(DecoderBatch& batch, int slot) {
    return *batch.state_->storage->slots.at(slot)->request;
}

std::vector<int> head(const std::vector<int>& tokens, int count) {
    return {tokens.begin(), tokens.begin() + count};
}

DecoderOptions options(std::string salt) {
    DecoderOptions result;
    result.prefix_cache_salt = std::move(salt);
    result.max_new_tokens = 16;
    result.temperature = 0.8;
    result.top_k = 20;
    result.top_p = 0.9;
    result.seed = -509;
    return result;
}

int first_token(DecoderBatch& batch, int lane, int peer = -1) {
    const auto selected = peer < 0 ? slots({lane}) : slots({lane, peer});
    for (int step = 0; step < 128; ++step) {
        DecoderBatchStepStats stats;
        const auto results = batch.advance_chunked(selected, &stats);
        require(!results[lane].finish_reason, "First token request retired");
        if (results[lane].token) {
            return *results[lane].token;
        }
        require(stats.prefill_rows[lane] == page, "Incomplete non-final cold prefill tile");
    }
    throw std::runtime_error("First token deadline");
}

void retry_discarded(DecoderBatch& batch, const std::vector<int>& prompt, int cached) {
    const auto state = options("abandoned");
    require(batch.admit(2, prompt, state) == 0, "Abandoned request started warm");
    if (cached > page) {
        auto first = batch.advance(slots({2}));
        require(!first[2].token, "Abandoned request completed prefill");
    }
    batch.cancel(2);
    require(batch.admit(2, prompt, state) == 0, "Abandoned request published a checkpoint");
    batch.cancel(2);
}

void inspect(DecoderBatch& oracle, DecoderBatch& candidate, const std::vector<int>& lane,
             std::vector<int> prime, int length, int cached, std::ofstream& logits) {
    oracle.reset_prefix_cache();
    candidate.reset_prefix_cache();
    const auto prompt = head(lane, length);
    const auto parameters = options("case");
    retry_discarded(candidate, prompt, cached);
    require(oracle.admit(0, prompt, parameters) == 0, "Cache-disabled oracle was warm");
    if (cached) {
        require(candidate.admit(0, prime, parameters) == 0, "Unexpected warm prime");
        remap(request(candidate, 0), true);
        first_token(candidate, 0);
        candidate.cancel(0);
        require(candidate.prefix_cache_stats().pages == cached / page, "Wrong checkpoint pages");
    }
    const int expected = first_token(oracle, 0);
    require(candidate.admit(1, prompt, parameters) == cached, "Unexpected restore depth");
    if (cached) {
        auto& warm = request(candidate, 1);
        remap(warm, false);
        auto& cache = *candidate.state_->storage->prefix_cache;
        const auto stream = candidate.state_->execution.stream();
        require(cache.restore(warm, prompt, parameters.prefix_cache_salt, stream) == cached,
                "Restore into permuted destination");
        check(cudaStreamSynchronize(stream), "Complete permuted restore");
        if (length == cached) {
            compare_state(request(oracle, 0), warm, length, false);
        }
    }
    const auto peer = options("peer");
    require(candidate.admit(3, {25}, peer) == 0, "Unexpected warm peer");
    first_token(candidate, 3);
    require(first_token(candidate, 1, 3) == expected, "Seeded first token differs");
    compare_state(request(oracle, 0), request(candidate, 1), length, true);
    const auto row = read_device(request(candidate, 1).work.logits, vocab_size);
    logits.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(Bf16));
    require(logits.good(), "Write BF16 evidence");
    candidate.cancel(3);
    candidate.cancel(1);
    oracle.cancel(0);
    require(candidate.request_device_allocation_bytes() == 0, "Request storage leak");
    std::cout << "length " << length << " cached " << cached << " token " << expected
              << " KV, GDN, hidden, residual and full BF16 logits match\n";
}

}

int main(int argc, char** argv) {
    try {
        require(argc == 5 || (argc == 6 && std::string_view(argv[5]) == "--short"),
                "Pass checkpoint, W3 lane IDs, W3 prime IDs and BF16 output path");
        const auto lane = ids(argv[2], 8192);
        const auto w3_prime = ids(argv[3], 6401);
        std::ofstream logits(argv[4], std::ios::binary);
        require(logits.good(), "Open BF16 evidence");
        Decoder decoder(argv[1], 0, false, Fp8PrefillPolicy::tensor_core);
        DecoderBatch oracle(decoder, {false, 1, 0});
        DecoderBatch candidate(decoder, {true, 4, 2});
        for (const int size : {127, 128, 129, 255, 256, 257, 385}) {
            const int cached = size / page * page;
            auto prime = head(lane, cached + (size != cached));
            if (size != cached) {
                prime.back() = (prime.back() + 1) % vocab_size;
            }
            inspect(oracle, candidate, lane, std::move(prime), size, cached, logits);
        }
        if (argc == 5) {
            inspect(oracle, candidate, lane, head(lane, 6400), 6400, 6400, logits);
            inspect(oracle, candidate, lane, w3_prime, 8192, 6400, logits);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
