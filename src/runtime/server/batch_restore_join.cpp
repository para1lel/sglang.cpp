#include "sglang/runtime/server/batch_cache.h"

namespace sglang::server {

BatchRestoreJoin::BatchRestoreJoin(std::shared_ptr<State> state) : state_(std::move(state)) {}

}
