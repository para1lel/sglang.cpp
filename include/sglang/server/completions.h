#pragma once

#include "sglang/tokenizer/tokenizer.h"

#include <mutex>
#include <string_view>

namespace sglang {

class Decoder;

namespace server {

class BatchService;
class ChatTemplate;
class Socket;

class CompletionTokenizer {
public:
    explicit CompletionTokenizer(const std::filesystem::path& checkpoint);
    ~CompletionTokenizer();
    const Tokenizer& get() const;
    const ChatTemplate& chat_template() const;

private:
    const std::filesystem::path checkpoint_;
    mutable std::once_flag loaded_;
    mutable std::unique_ptr<const Tokenizer> tokenizer_;
    mutable std::once_flag chat_loaded_;
    mutable std::unique_ptr<const ChatTemplate> chat_template_;
};

void complete(const Socket& connection, const Decoder& decoder, std::string_view body,
              bool trace = false, int mtp_steps = 0);
void complete(const Socket& connection, BatchService& service, int slot, std::string_view body);
void complete(const Socket& connection, const Decoder& decoder,
              const CompletionTokenizer& tokenizer, std::string_view body,
              bool trace = false, int mtp_steps = 0, bool eager = false);
void complete(const Socket& connection, BatchService& service, int slot,
              const CompletionTokenizer& tokenizer, std::string_view body);
void complete_chat(const Socket& connection, const Decoder& decoder,
                   const CompletionTokenizer& tokenizer, std::string_view body,
                   bool trace, int mtp_steps, bool eager);
void complete_chat(const Socket& connection, BatchService& service, int slot,
                   const CompletionTokenizer& tokenizer, std::string_view body);
void write_error(const Socket& connection, std::string_view status, const char* message);

}
}
