#include "sglang/runtime/server/batch_session.h"
#include "sglang/runtime/server/batch_executor.h"

namespace sglang::server {

int BatchTopology::device_count() const noexcept { return dp_size * tp_size * pp_size; }

DecoderBatchPrefixUpdate BatchSession::poll_prefix(int) { return {}; }

BatchSession::~BatchSession() = default;

BatchExecutor::~BatchExecutor() = default;

const BatchExecutor::Options& BatchExecutor::options() const noexcept { return options_; }

BatchExecutor::BatchExecutor(Options options) : options_(options) {}

}
