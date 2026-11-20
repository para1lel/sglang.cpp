#pragma once

#include "sglang/tokenizer/tokenizer.h"

#include <mutex>
#include <string_view>

namespace sglang {

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

void complete(const Socket& connection, BatchService& service, int slot, std::string_view body);
void complete(const Socket& connection, BatchService& service, int slot,
              const CompletionTokenizer& tokenizer, std::string_view body);
void complete_chat(const Socket& connection, BatchService& service, int slot,
                   const CompletionTokenizer& tokenizer, std::string_view body,
                   VisionCompletion* vision = nullptr);
void write_error(const Socket& connection, std::string_view status, const char* message);

}
}
