#pragma once

#include "sglang/runtime/model/qwen_layers.h"

namespace sglang::decoder {

struct VerificationWorkspace;

TextGdnReplayRows replay_rows(Bf16* storage, int count);

TextGdnReplayRows replay_rows_local(Bf16* storage, int count);

void relayout_verification_workspace_local(VerificationWorkspace& scratch, int rows);

void free_pinned(void* results);

}

