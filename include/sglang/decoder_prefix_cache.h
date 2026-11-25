#pragma once

#include "sglang/runtime/cache/prefix_payload.h"
#include <cuda_runtime_api.h>

namespace sglang::decoder {

struct Request;
struct Workspace;
class MtpDraftRequest;

PrefixDeviceTransfer restore_prefix_view(Request& request, const PrefixCacheRead& read,
                                         cudaStream_t initialized,
                                         MtpDraftRequest* draft = nullptr);
PrefixDeviceTransfer export_prefix_view(const Request& request, int gqa_begin, int draft_begin,
                                        int end, cudaStream_t source_ready,
                                        const MtpDraftRequest* draft = nullptr,
                                        const Workspace* row = nullptr, bool terminal = false);
int commit_restored_prefix(Request& request, const PrefixDeviceTransfer& view,
                           MtpDraftRequest* draft = nullptr);

}
