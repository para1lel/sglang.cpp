#pragma once

#include "sglang/json.h"
#include "sglang/server/batch_service.h"
#include "sglang/server/prefix_controls.h"

#include <map>
#include <memory>
#include <mutex>
#include <random>

namespace sglang {
class Tokenizer;
}

namespace sglang::server {

class CompletionTokenizer;
class Socket;
struct HttpRequest;
enum class OllamaCompletion;

struct GenerateInput {
    json::Object fields;
    std::size_t count = 1;
    std::size_t samples = 1;
    bool stream = false;
    bool batch = false;
    bool text = false;

    std::size_t item_count() const;
    bool indexed() const;
};

GenerateInput parse_generate_input(std::string_view body);
const json::Array* generate_parameter_list(const json::Value* value, std::size_t count,
                                           bool strings);
std::size_t generate_sample_count(const GenerateInput& input, const json::Array* params);
void set_generate_constraint(const json::Object& sampling, const Tokenizer& tokenizer,
                             DecoderOptions& options);
std::vector<BatchPrompt> parse_generate_prompts(const GenerateInput& input,
                                               const CompletionTokenizer& tokenizer,
                                               const std::vector<int>* history = nullptr);

bool session_control_route(std::string_view path);

class NativeGenerate {
public:
    NativeGenerate(BatchService& service, const CompletionTokenizer& tokenizer);

    QueuedGenerate prepare(std::string_view body);
    QueuedGenerate prepare_ollama(const Socket& connection, std::string_view body, bool json_body,
                                  OllamaCompletion completion, bool anthropic_errors);
    void abort(const Socket& connection, std::string_view body);
    void serve_session(const Socket& connection, const HttpRequest& request);

private:
    struct Registration;
    struct Session;
    struct SessionTurn {
        SessionTurn(NativeGenerate& owner, std::shared_ptr<Session> session, std::string id);
        ~SessionTurn();
        void reserve(const BatchPrompt& item);
        void finish(const DecoderResult& result);

        NativeGenerate& owner;
        std::shared_ptr<Session> session;
        std::string id;
        std::vector<int> history;
        bool pending = false;
    };

    static std::string random_id(std::random_device& random);
    std::shared_ptr<Registration> register_ids(const GenerateInput& input);
    std::shared_ptr<Session> bind_session(const GenerateInput& input);
    std::unique_ptr<SessionTurn> begin_session(const GenerateInput& input, std::string_view rid,
                                               const std::shared_ptr<Session>& session);
    void close_session(Session& session);
    void expire_sessions();

    BatchService& service_;
    const CompletionTokenizer& tokenizer_;
    std::mutex mutex_;
    std::map<std::string, BatchRequestId*, std::less<>> ids_;
    std::uint64_t next_id_ = 1;
    std::mutex sessions_mutex_;
    std::map<std::string, std::shared_ptr<Session>, std::less<>> sessions_;
    std::size_t session_tokens_ = 0;
};

}
