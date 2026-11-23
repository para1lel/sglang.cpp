#include "sglang/decoder_internal.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using namespace sglang;
using namespace sglang::decoder;

class Capture {
public:
    explicit Capture(const char* path) : file_(path, std::ios::binary) {
        if (!file_) {
            throw std::runtime_error("Could not open capture");
        }
    }

    template <typename Value>
    void device(const Value* data, std::size_t count) {
        const auto bytes = count * sizeof(Value);
        std::vector<Value> host(count);
        check(cudaMemcpy(host.data(), data, bytes, cudaMemcpyDeviceToHost), "Read checkpoint");
        file_.write(reinterpret_cast<const char*>(host.data()), bytes);
        if (!file_) {
            throw std::runtime_error("Could not write capture");
        }
    }

    void state(const Request& request) {
        check(cudaStreamSynchronize(request.memory.stream()), "Finish recurrent state");
        for (const auto& layer : request.states) {
            device(layer.history, gdn_history_elements);
            device(layer.recurrent, gdn_state_elements);
        }
    }

    void logits(const Request& request) {
        check(cudaStreamSynchronize(request.memory.stream()), "Finish logits");
        device(request.work.logits, vocab_size);
        device(&request.work.argmax->token_id, 1);
    }

    void close() {
        file_.close();
        if (!file_) {
            throw std::runtime_error("Could not close capture");
        }
    }

private:
    std::ofstream file_;
};

std::vector<int> ids(const char* path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Could not open token fixture");
    }
    return {std::istream_iterator<int>(file), std::istream_iterator<int>()};
}

void teacher_force(const Decoder::Model& model, const std::vector<int>& prompt,
                   const std::vector<int>& teachers, Capture& capture) {
    int position = static_cast<int>(prompt.size());
    Request request(0, position + static_cast<int>(teachers.size()), model.memory);
    request.reset();
    DecoderResult result;
    int chunks = 0;
    const DecoderCallbacks callbacks{.is_cancelled = [&] {
        capture.state(request);
        ++chunks;
        return false;
    }};
    RequestProgress progress;
    const SerialOutput output(callbacks, 0, result);
    if (!model.prefill_serial(request, progress, prompt, output, result, nullptr, 0) ||
        chunks != 2 || result.peak_device_allocation_bytes != 255557632) {
        throw std::runtime_error("Expected the 1024-row automatic serial prefill route");
    }
    model.compute_logits(request);
    capture.logits(request);
    for (const int token : teachers) {
        model.forward(request, token, position++, TextGdnMode::decode);
        model.compute_logits(request, true);
        capture.logits(request);
    }
    capture.state(request);
    std::cout << "prompt=" << prompt.size() << " chunks=" << chunks
              << " workspace=" << result.peak_device_allocation_bytes
              << " teacher_steps=" << teachers.size() << " state_snapshots=" << chunks + 1
              << " gdn_layers=" << gdn_layers << " logits=" << vocab_size << '\n';
}

}

int main(int argc, char** argv) try {
    if (argc != 5) {
        throw std::invalid_argument("Usage: gdn-checkpoint CHECKPOINT PROMPT TEACHERS CAPTURE");
    }
    const auto prompt = ids(argv[2]);
    const auto teachers = ids(argv[3]);
    if (prompt.size() != 2048 || teachers.size() != 8) {
        throw std::invalid_argument("Expected 2048 prompt IDs and eight teacher tokens");
    }
    const Decoder::Model model(argv[1], 0);
    Capture capture(argv[4]);
    teacher_force(model, prompt, teachers, capture);
    capture.close();
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
