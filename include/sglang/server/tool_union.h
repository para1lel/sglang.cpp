#pragma once

#include "sglang/json.h"

namespace sglang::server {

bool matches_tool_union(const json::Value& value, const json::Value& schema,
                        const json::Value& root);

}
