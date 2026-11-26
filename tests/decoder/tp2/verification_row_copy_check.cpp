#include "sglang/runtime/execution/verification_row_copy.h"
#include "sglang/runtime/model/qwen_text_model.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace sglang::decoder;
using Bits = std::uint16_t;
using DeviceElement = __nv_bfloat16;

constexpr int width = qwen_text_model.hidden_size;
constexpr int capacity = VerificationCopyRows::capacity;
constexpr int vocabulary = 193;
constexpr int guard = 8;
constexpr int stride = width + 3 * guard;
constexpr Bits canary = 0xa5a5;
static_assert(sizeof(Bits) == sizeof(DeviceElement));

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class Stream {
public:
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "Create stream"); }
    ~Stream() { cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    cudaStream_t value;
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), elements_ * sizeof(Bits)),
              "Allocate CUDA buffer");
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void upload(const std::vector<Bits>& host, cudaStream_t stream) {
        require(host.size() == elements_, "Incorrect upload size");
        check(cudaMemcpyAsync(data, host.data(), elements_ * sizeof(Bits), cudaMemcpyHostToDevice,
                              stream), "Upload input bits");
    }

    void clear(cudaStream_t stream) {
        check(cudaMemsetAsync(data, canary & 255, elements_ * sizeof(Bits), stream),
              "Set guard bytes");
    }

    std::vector<Bits> download(cudaStream_t stream) const {
        std::vector<Bits> host(elements_);
        check(cudaMemcpyAsync(host.data(), data, elements_ * sizeof(Bits), cudaMemcpyDeviceToHost,
                              stream), "Read output bits");
        check(cudaStreamSynchronize(stream), "Finish copy check");
        return host;
    }

    DeviceElement* data;

private:
    std::size_t elements_;
};

std::vector<Bits> source_data(int rows, unsigned salt) {
    std::vector<Bits> data(rows * width);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < width; ++column) {
            data[row * width + column] = static_cast<Bits>((row * 257 + column * 13) ^ salt);
        }
    }
    return data;
}

int permuted_slot(int row) { return (row * 37 + 11) % capacity; }

VerificationEmbeddingTokens embedding_tokens() {
    VerificationEmbeddingTokens tokens{};
    tokens.id[0] = 0;
    tokens.id[1] = vocabulary - 1;
    for (int row = 2; row < capacity; ++row) {
        tokens.id[row] = (row * 31) % (vocabulary - 2) + 1;
    }
    tokens.id[3] = tokens.id[0];
    return tokens;
}

VerificationCopyRows destinations(DeviceBuffer& output) {
    VerificationCopyRows rows{};
    for (int row = 0; row < capacity; ++row) {
        rows.destination[row] = output.data + permuted_slot(row) * stride + guard;
    }
    return rows;
}

int check_rows(const DeviceBuffer& output, const std::vector<Bits>& source,
               const int* source_rows, int count, cudaStream_t stream) {
    const auto actual = output.download(stream);
    std::array<int, capacity> expected{};
    expected.fill(-1);
    for (int row = 0; row < count; ++row) {
        expected[permuted_slot(row)] = source_rows ? source_rows[row] : row;
    }
    const auto guard_bits = [](Bits bits) { return bits == canary; };
    int mismatches = 0;
    for (int slot = 0; slot < capacity; ++slot) {
        const auto begin = actual.begin() + slot * stride;
        const int row = expected[slot];
        if (row < 0) {
            require(std::all_of(begin, begin + stride, guard_bits), "Unused destination changed");
            continue;
        }
        require(std::all_of(begin, begin + guard, guard_bits), "Prefix guard changed");
        require(std::all_of(begin + guard + width, begin + stride, guard_bits),
                "Suffix guard changed");
        mismatches += !std::equal(begin + guard, begin + guard + width,
                                  source.begin() + row * width);
    }
    return mismatches;
}

void check_scatter(const DeviceBuffer& slab, DeviceBuffer& output, const std::vector<Bits>& input,
                   const VerificationCopyRows& rows, int count, cudaStream_t stream) {
    output.clear(stream);
    check(scatter_verification_hidden(slab.data, rows, count, stream), "Scatter hidden rows");
    require(check_rows(output, input, nullptr, count, stream) == 0, "Scatter bits differ");
    if (count > 1) {
        auto wrong = rows;
        std::rotate(wrong.destination, wrong.destination + 1, wrong.destination + count);
        output.clear(stream);
        check(scatter_verification_hidden(slab.data, wrong, count, stream), "Scatter control");
        require(check_rows(output, input, nullptr, count, stream) == count,
                "Scatter did not detect every permuted row");
    }
}

void check_gather(const DeviceBuffer& weights, DeviceBuffer& output, const std::vector<Bits>& input,
                  const VerificationCopyRows& rows, const VerificationEmbeddingTokens& tokens,
                  int count, cudaStream_t stream) {
    output.clear(stream);
    check(gather_verification_embeddings(weights.data, vocabulary, tokens, rows, count, stream),
          "Gather embedding rows");
    require(check_rows(output, input, tokens.id, count, stream) == 0, "Gather bits differ");
    if (count > 1) {
        auto wrong = rows;
        std::rotate(wrong.destination, wrong.destination + 1, wrong.destination + count);
        output.clear(stream);
        check(gather_verification_embeddings(weights.data, vocabulary, tokens, wrong, count,
                                             stream),
              "Gather control");
        require(check_rows(output, input, tokens.id, count, stream) == count,
                "Gather did not detect every permuted row");
    }
}

void check_rejected(const DeviceBuffer& slab, const DeviceBuffer& weights,
                    const VerificationCopyRows& rows, VerificationEmbeddingTokens tokens,
                    cudaStream_t stream) {
    for (int count : {0, capacity + 1}) {
        require(scatter_verification_hidden(slab.data, rows, count, stream) ==
                    cudaErrorInvalidValue,
                "Scatter accepted an invalid count");
        require(gather_verification_embeddings(weights.data, vocabulary, tokens, rows, count,
                                               stream) == cudaErrorInvalidValue,
                "Gather accepted an invalid count");
    }
    for (int token : {-1, vocabulary}) {
        tokens.id[1] = token;
        require(gather_verification_embeddings(weights.data, vocabulary, tokens, rows, 2, stream) ==
                    cudaErrorInvalidValue,
                "Gather accepted an invalid token");
    }
}

void check_checkpoint(DeviceBuffer& slab, DeviceBuffer& output, const std::vector<Bits>& input,
                      const VerificationCopyRows& rows, cudaStream_t stream) {
    DeviceBuffer checkpoint(width);
    constexpr int row = 7;
    check(scatter_verification_hidden(slab.data, rows, 10, stream), "Scatter checkpoint rows");
    check(cudaMemcpyAsync(checkpoint.data, rows.destination[row], width * sizeof(Bits),
                          cudaMemcpyDeviceToDevice, stream), "Copy selected hidden row");
    slab.clear(stream);
    const auto saved = checkpoint.download(stream);
    require(std::equal(saved.begin(), saved.end(), input.begin() + row * width),
            "Checkpoint changed after slab reuse");
    const auto owned = output.download(stream);
    const auto begin = owned.begin() + permuted_slot(row) * stride + guard;
    require(std::equal(begin, begin + width, saved.begin()), "Owned hidden row changed");
}

}

int main() try {
    Stream stream;
    DeviceBuffer slab(capacity * width);
    DeviceBuffer weights(vocabulary * width);
    DeviceBuffer output(capacity * stride);
    const auto slab_input = source_data(capacity, 0x753c);
    const auto weight_input = source_data(vocabulary, 0x9341);
    const auto tokens = embedding_tokens();
    const auto rows = destinations(output);
    slab.upload(slab_input, stream.value);
    weights.upload(weight_input, stream.value);
    for (int count : {1, 5, 10, 40, 80, capacity}) {
        check_scatter(slab, output, slab_input, rows, count, stream.value);
        check_gather(weights, output, weight_input, rows, tokens, count, stream.value);
    }
    output.clear(stream.value);
    check_rejected(slab, weights, rows, tokens, stream.value);
    const auto cleared = output.download(stream.value);
    require(std::all_of(cleared.begin(), cleared.end(), [](Bits bits) { return bits == canary; }),
            "A rejected copy modified a destination");
    require(slab.download(stream.value) == slab_input, "Scatter modified its input");
    require(weights.download(stream.value) == weight_input, "Gather modified its input");
    check_checkpoint(slab, output, slab_input, rows, stream.value);
    std::cout << "TP2 verification row copy checks passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
