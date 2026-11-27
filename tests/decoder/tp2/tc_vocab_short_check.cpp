#include "sglang/checkpoint.h"
#include "sglang/decoder_internal.h"
#include "sglang/runtime/backend/cuda_resources.h"
#include "sglang/text/bf16_head_projection.h"
#include "sglang/text/mtp/tc_vocab.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using namespace sglang;
using namespace sglang::decoder;

constexpr int rows = tc_vocab_short_rows;
constexpr int lane_rows = rows / 2;
constexpr int elements = rows * hidden_size;
constexpr auto matrix_size = static_cast<std::size_t>(vocab_size) * hidden_size;
using RowStorage = std::array<Bf16, elements>;
using Winner = TextLogitCandidate;

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void transfer(void* dest, const void* src, std::size_t size, cudaMemcpyKind kind,
              cudaStream_t stream) {
    check(cudaMemcpyAsync(dest, src, size, kind, stream), "Copy operator fixture data");
    check(cudaStreamSynchronize(stream), "Finish operator fixture data transfer");
}

struct PinnedWinners {
    PinnedWinners() {
        check(cudaMallocHost(reinterpret_cast<void**>(&data), rows * sizeof(Winner)),
              "Allocate pinned verification winners");
    }
    ~PinnedWinners() { cudaFreeHost(data); }
    PinnedWinners(const PinnedWinners&) = delete;
    PinnedWinners& operator=(const PinnedWinners&) = delete;
    Winner* data = nullptr;
};

struct Inputs {
    RowStorage branch;
    RowStorage residual;
    RowStorage normalized;
};

Inputs read_capture(const char* path) {
    Inputs inputs;
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(&inputs), sizeof(inputs));
    expect(stream.good() && stream.peek() == std::ifstream::traits_type::eof(),
           "Expected exactly three ten-row BF16 matrices");
    return inputs;
}

class OperatorTest {
public:
    explicit OperatorTest(const Checkpoint& checkpoint) {
        device.open_stream();
        stream = device.stream();
        head = device.allocate<Bf16>(matrix_size);
        gain = device.allocate<Bf16>(hidden_size);
        branch = device.allocate<Bf16>(elements);
        residual = device.allocate<Bf16>(elements);
        normalized = device.allocate<Bf16>(elements);
        packed = device.allocate<Bf16>(elements);
        packed_logits = device.allocate<Bf16>(rows * vocab_size);
        logits = device.allocate<Bf16>(rows * vocab_size);
        winners = device.allocate<Winner>(rows);
        argmax = device.allocate<Winner>(rows * text_argmax_scratch_count(vocab_size));

        const auto& w = checkpoint.tensor("lm_head.weight", TensorDtype::BF16,
                                           {vocab_size, hidden_size});
        const auto& g = checkpoint.tensor("model.language_model.norm.weight", TensorDtype::BF16,
                                           {hidden_size});
        transfer(head, w.data, w.size_bytes, cudaMemcpyHostToDevice, stream);
        transfer(gain, g.data, g.size_bytes, cudaMemcpyHostToDevice, stream);
        auto* quantized = device.allocate<std::uint8_t>(matrix_size);
        auto* scales = device.allocate<float>(vocab_size);
        check(quantize_draft_head_int8(head, vocab_size, quantized, scales, stream),
              "Quantize checkpoint head for operator fixture");
        weights = prepare_tc_vocab(head, quantized, scales, device);
        scratch = prepare_tc_vocab_scratch(device, true, true, true);
        expect(weights && scratch, "Production ten-row scorer is unavailable");
        check(cudaStreamSynchronize(stream), "Prepare fixture weight statistics");

        for (int row = 0; row < rows; ++row) {
            const int lane_slot = row % 2 * lane_rows + row / 2;
            const auto* input = normalized + row * hidden_size;
            packed_rows[row] = {input, logits + row * vocab_size};
            greedy_rows[row] = {input, winners + lane_slot};
        }
    }

    void normalize() const {
        for (int row = 0; row < rows; ++row) {
            const int offset = row * hidden_size;
            check(text_rms_norm(branch + offset, residual + offset, gain, hidden_size,
                                normalized + offset, nullptr, stream),
                  "Run production final norm");
        }
    }

    void direct() const {
        const auto status = project_bf16_head(head, vocab_size, hidden_size, packed_rows.data(),
                                              rows, packed, packed_logits, stream);
        expect(status.has_value(), "Production packed head declined ten rows");
        check(*status, "Run production packed BF16 head");
        for (int row = 0; row < rows; ++row) {
            check(text_argmax(packed_rows[row].output, vocab_size,
                argmax + row * text_argmax_scratch_count(vocab_size), greedy_rows[row].winner,
                stream), "Run production greedy argmax");
        }
    }

    void certified() const {
        expect(try_tc_vocab_short(*weights, *scratch, greedy_rows.data(), stream),
               "Production ten-row scorer declined");
    }

    void read(Winner* pinned) const {
        for (int lane = 0; lane < 2; ++lane) {
            const int offset = lane * lane_rows;
            check(cudaMemcpyAsync(pinned + offset, winners + offset, lane_rows * sizeof(Winner),
                                  cudaMemcpyDeviceToHost, stream), "Read target winners per lane");
        }
    }

    void compare(const char* name) {
        direct();
        read(reference.data);
        certified();
        read(observed.data);
        check(cudaStreamSynchronize(stream), "Finish focused comparison");
        for (int row = 0; row < rows; ++row) {
            const auto a = reference.data[row];
            const auto b = observed.data[row];
            if (a.token_id != b.token_id || std::bit_cast<unsigned>(a.score) !=
                                            std::bit_cast<unsigned>(b.score)) {
                std::cerr << name << " winner " << row << " expected " << a.token_id << ' '
                          << a.score << " observed " << b.token_id << ' ' << b.score << '\n';
                throw std::runtime_error("Packed BF16 winner mismatch");
            }
        }
        std::cout << name << " ten BF16 winners match\n";
    }

    void capture(const char* name) {
        const auto captured = read_capture(name);
        transfer(branch, captured.branch.data(), sizeof(RowStorage), cudaMemcpyHostToDevice,
                 stream);
        transfer(residual, captured.residual.data(), sizeof(RowStorage), cudaMemcpyHostToDevice,
                 stream);
        normalize();
        RowStorage current;
        transfer(current.data(), normalized, sizeof(current), cudaMemcpyDeviceToHost, stream);
        expect(std::memcmp(current.data(), captured.normalized.data(), sizeof(current)) == 0,
               "Captured final norm differs at BF16 precision");
        compare(name);
        benchmark(name);
    }

    void stress() {
        RowStorage input{};
        for (int row = 0; row < rows; ++row) input[row * hidden_size + 17] = Bf16(1.0f);
        const auto ordinary = input;
        transfer(normalized, input.data(), sizeof(input), cudaMemcpyHostToDevice, stream);
        compare("sparse BF16 tie");
        expect(reference.data[0].token_id == 354, "Positive lowest-token tie was not exercised");

        std::fill_n(input.begin(), hidden_size, Bf16(0.0f));
        transfer(normalized, input.data(), sizeof(input), cudaMemcpyHostToDevice, stream);
        compare("one overflowing zero row");
        expect(reference.data[0].token_id == 0 && reference.data[0].score == 0,
               "Full-vocabulary zero tie was not exercised");

        input = ordinary;
        input[hidden_size] = Bf16(std::numeric_limits<float>::quiet_NaN());
        input[2 * hidden_size] = Bf16(std::numeric_limits<float>::infinity());
        input[3 * hidden_size] = Bf16(0x1p90f);
        transfer(normalized, input.data(), sizeof(input), cudaMemcpyHostToDevice, stream);
        compare("mixed finite, NaN, infinity, and out-of-domain rows");
    }

private:
    float measure(bool use_certificate) {
        constexpr int repeats = 12;
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < repeats; ++iteration) {
            normalize();
            if (use_certificate) certified();
            else direct();
            read(observed.data);
            check(cudaStreamSynchronize(stream), "Complete verification winner readback");
        }
        return std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - start)
                   .count() / repeats;
    }

    void benchmark(const char* name) {
        std::array<float, 9> direct_times{}, certified_times{};
        measure(false);
        measure(true);
        for (std::size_t trial = 0; trial < direct_times.size(); ++trial) {
            if (trial % 2 == 0) {
                direct_times[trial] = measure(false);
                certified_times[trial] = measure(true);
            } else {
                certified_times[trial] = measure(true);
                direct_times[trial] = measure(false);
            }
        }
        std::sort(direct_times.begin(), direct_times.end());
        std::sort(certified_times.begin(), certified_times.end());
        std::cout << name << " host-visible median_us direct=" << std::fixed << std::setprecision(3)
                  << direct_times[4] << " certified=" << certified_times[4] << '\n';
    }

    Allocations device{0};
    cudaStream_t stream{};
    Bf16 *head{}, *gain{}, *branch{}, *residual{}, *normalized{}, *packed{}, *packed_logits{};
    Bf16* logits{};
    Winner *winners{}, *argmax{};
    PinnedWinners reference, observed;
    std::shared_ptr<const TcVocabWeights> weights;
    std::shared_ptr<const TcVocabScratch> scratch;
    std::array<Bf16HeadRow, rows> packed_rows;
    std::array<TcVocabRow, rows> greedy_rows;
};

}

int main(int argc, char** argv) {
    try {
        expect(argc >= 2, "Supply the checkpoint and optional captured W1 packets");
        int devices = 0;
        check(cudaGetDeviceCount(&devices), "Locate isolated accelerator");
        expect(devices == 1, "Run with only the reserved GPU visible");
        check(cudaSetDevice(0), "Select isolated accelerator");
        const Checkpoint checkpoint(argv[1]);
        OperatorTest test(checkpoint);
        for (int i = 2; i < argc; ++i) test.capture(argv[i]);
        test.stress();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
