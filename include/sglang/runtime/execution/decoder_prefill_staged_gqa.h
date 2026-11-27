#pragma once

#include "sglang/runtime/execution/decoder_prefill_tiles.h"

namespace sglang::decoder::prefill_detail {

void prefill_group_gqa(const Chunk& chunk, const GqaLayer& layer, int bank,
                      const PrefillGrant* grants, int count, int offset,
                      bool staged_batch = false);

using PrefillPartners = std::array<int, staged_prefill_batch_size>;

std::optional<PrefillPartners> mixed_prefill_partners(const PrefillGrant* grants, int count);

void prefill_mixed_gqa(const Chunk& chunk, const GqaLayer& layer, int bank,
                       const PrefillGrant* grants, const PrefillPartners& partners);

bool can_batch_staged_gqa(const PrefillGrant* grants, int count);

}
