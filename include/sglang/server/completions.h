#pragma once

#include "sglang/tokenizer/tokenizer.h"

#include <exception>
#include <mutex>
#include <string_view>

namespace sglang {

class Decoder;
struct DecoderOptions;

namespace server {

class BatchService;
class ChatTemplate;
class Socket;
class VisionCompletion;

enum class SamplingDefaults { model, openai };

class CompletionTokenizer {
public:
    explicit CompletionTokenizer(const std::filesystem::path& checkpoint,
                                 SamplingDefaults sampling = SamplingDefaults::model);
    ~CompletionTokenizer();
    const Tokenizer& get() const;
    const ChatTemplate& chat_template() const;
    const DecoderOptions& chat_sampling_defaults() const;

private:
    const std::filesystem::path checkpoint_;
    const SamplingDefaults sampling_mode_;
    mutable std::once_flag loaded_;
    mutable std::unique_ptr<const Tokenizer> tokenizer_;
    mutable std::once_flag chat_loaded_;
    mutable std::unique_ptr<const ChatTemplate> chat_template_;
    mutable std::once_flag sampling_loaded_;
    mutable std::unique_ptr<const DecoderOptions> sampling_defaults_;
};

void complete(const Socket& connection, const Decoder& decoder, std::string_view body,
              bool trace = false, int mtp_steps = 0);
void complete(const Socket& connection, BatchService& service, int slot, std::string_view body);
void complete(const Socket& connection, const Decoder& decoder,
              const CompletionTokenizer& tokenizer, std::string_view body,
              bool trace = false, int mtp_steps = 0, bool eager = false, int target_rows = 0,
              std::exception_ptr* failure = nullptr);
void complete(const Socket& connection, BatchService& service, int slot,
              const CompletionTokenizer& tokenizer, std::string_view body);
void complete_chat(const Socket& connection, const Decoder& decoder,
                   const CompletionTokenizer& tokenizer, std::string_view body,
                   bool trace, int mtp_steps, bool eager, VisionCompletion* vision = nullptr,
                   int target_rows = 0, std::exception_ptr* failure = nullptr);
void complete_chat(const Socket& connection, BatchService& service, int slot,
                   const CompletionTokenizer& tokenizer, std::string_view body,
                   VisionCompletion* vision = nullptr);
void write_error(const Socket& connection, std::string_view status, const char* message);

}
}
