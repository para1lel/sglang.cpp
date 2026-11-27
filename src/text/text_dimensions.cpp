#include "sglang/text/gdn_local.h"
#include "sglang/text_ops.h"

namespace sglang {
namespace {

constexpr int argmax_scratch_count(int vocab_size) {
    return vocab_size / text_argmax_tile + (vocab_size % text_argmax_tile != 0);
}

static_assert(argmax_scratch_count(text_argmax_tile) == 1);
static_assert(argmax_scratch_count(text_argmax_tile + 1) == 2);

template <typename Shape>
std::size_t gdn_workspace_bytes(int rows) {
    return rows > 0 && rows <= text_attention_max_tokens
               ? sizeof(__nv_bfloat16) * static_cast<std::size_t>(rows) *
                     (Shape::qkv_width + Shape::value_width)
               : 0;
}

}

int text_fp8_group_count(int width) {
    return width / text_fp8_group_size + (width % text_fp8_group_size != 0);
}

int text_argmax_scratch_count(int vocab_size) {
    return argmax_scratch_count(vocab_size);
}

std::size_t text_gdn_workspace_bytes(int rows) {
    return gdn_workspace_bytes<GdnFull>(rows);
}

std::size_t text_gdn_local_workspace_bytes(int rows) {
    return gdn_workspace_bytes<GdnLocal>(rows);
}

}
