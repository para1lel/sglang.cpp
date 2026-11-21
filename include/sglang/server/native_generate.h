#pragma once

#include "sglang/json.h"
#include "sglang/server/batch_service.h"
#include "sglang/server/prefix_controls.h"

#include <map>
#include <memory>
#include <mutex>

namespace sglang::server {

class CompletionTokenizer;
class Socket;

struct GenerateInput {
    json::Object fields;
    std::size_t count = 1;
    bool stream = false;
    bool batch = false;
    bool text = false;
};

GenerateInput parse_generate_input(std::string_view body);
void set_generate_constraint(const json::Object& sampling, const Tokenizer& tokenizer,
                             DecoderOptions& options);
std::vector<BatchPrompt> parse_generate_prompts(const GenerateInput& input,
                                               const CompletionTokenizer& tokenizer);

class NativeGenerate {
public:
    NativeGenerate(BatchService& service, const CompletionTokenizer& tokenizer)
        : service_(service), tokenizer_(tokenizer) {}

    QueuedGenerate prepare(std::string_view body);
    void abort(const Socket& connection, std::string_view body);

private:
    struct Registration;
    std::shared_ptr<Registration> register_ids(std::vector<std::optional<std::string>> ids);

    BatchService& service_;
    const CompletionTokenizer& tokenizer_;
    std::mutex mutex_;
    std::map<std::string, BatchRequestId*, std::less<>> ids_;
    std::uint64_t next_id_ = 1;
};

}
