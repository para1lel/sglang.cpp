#pragma once

// Private to decoder_prefill.cpp, where Chunk and its packed workspace are defined.
PagedPrefillSequence append_prefill_sequence(const Chunk& chunk, const GqaLayer& layer, int bank,
                                            const PrefillGrant& grant, int row) {
    const auto segment = prefill_attention_segment(chunk, grant, row);
    const auto cache = grant.request->caches[bank];
    const auto pages = grant.request->pages;
    segment.gqa_append(layer, cache, pages);
    return {cache, pages, grant.position, grant.rows, segment.work.rope_queries,
            segment.work.attention};
}

void read_prefill_group(const Chunk& chunk, const PagedPrefillSequence* sequences, int count,
                        bool staged_batch = false) {
    const int position = sequences[0].position;
    const bool long_lane = count == 1 && chunk.rows == grouped_prefill_max_rows &&
        position >= 32768 && position <= staged_prefill_batch_long_max_position;
    if (staged_batch || long_lane) {
        const auto scratch = chunk.work.gqa_scratch();
        if (position <= staged_prefill_batch_max_position) {
            if (const auto staged = try_batched_staged_paged_prefill(sequences, count,
                                                                    chunk.stream, scratch)) {
                check(*staged, "Read staged batched causal prefill MXFP8 attention");
                return;
            }
            for (int lane = 0; lane < count; ++lane) {
                read_paged_sequences(sequences + lane, 1, chunk.stream, scratch);
            }
            return;
        }
        const int visible = position + batch_prefill_tile_rows;
        const int subgroup = std::min(count, visible <= 32768 ? 4 : visible <= 65536 ? 2 : 1);
        for (int lane = 0; lane < count; lane += subgroup) {
            const auto* first = sequences + lane;
            if (const auto staged = try_long_staged_paged_prefill(first, subgroup, chunk.stream,
                                                                  scratch)) {
                check(*staged, "Read long staged causal prefill MXFP8 attention");
                continue;
            }
            for (int index = 0; index < subgroup; ++index) {
                read_paged_sequences(first + index, 1, chunk.stream, scratch);
            }
        }
        return;
    }
    read_paged_sequences(sequences, count, chunk.stream,
                         count == 1 ? chunk.work.gqa_scratch() : PairedPrefillScratch{});
}

void prefill_group_gqa(const Chunk& chunk, const GqaLayer& layer, int bank,
                      const PrefillGrant* grants, int count, int offset,
                      bool staged_batch = false) {
    static_assert(staged_prefill_batch_size >= max_layerwise_rows / 256);
    std::array<PagedPrefillSequence, staged_prefill_batch_size> sequences;
    int index = 0;
    each_prefill_segment(grants, count, offset, [&](const PrefillGrant& grant, int row) {
        sequences[index++] = append_prefill_sequence(chunk, layer, bank, grant, row);
    });
    read_prefill_group(chunk, sequences.data(), count, staged_batch);
}

using PrefillPartners = std::array<int, staged_prefill_batch_size>;

std::optional<PrefillPartners> mixed_prefill_partners(const PrefillGrant* grants, int count) {
    if (count != staged_prefill_batch_size) return std::nullopt;
    PrefillPartners partners;
    for (int lane = 0; lane < count; ++lane) {
        const auto& grant = grants[lane];
        if (grant.media || grant.rows != batch_prefill_tile_rows) return std::nullopt;
        for (int prior = 0; prior < lane; ++prior) {
            if (grants[prior].request == grant.request) return std::nullopt;
        }
        partners[lane] = lane;
    }
    bool paired = false;
    for (int lane = 0; lane < count; ++lane) {
        const int position = grants[lane].position;
        if (partners[lane] != lane || position < 8192 ||
            position > 65536 - batch_prefill_tile_rows || position % batch_prefill_tile_rows) {
            continue;
        }
        for (int next = lane + 1; next < count; ++next) {
            if (partners[next] != next || grants[next].position != position) continue;
            partners[lane] = next;
            partners[next] = lane;
            paired = true;
            break;
        }
    }
    return paired ? std::optional{partners} : std::nullopt;
}

void prefill_mixed_gqa(const Chunk& chunk, const GqaLayer& layer, int bank,
                       const PrefillGrant* grants, const PrefillPartners& partners) {
    for (int lane = 0; lane < staged_prefill_batch_size; ++lane) {
        const int other = partners[lane];
        if (other < lane) continue;
        const auto first = append_prefill_sequence(chunk, layer, bank, grants[lane],
                                                  lane * batch_prefill_tile_rows);
        if (other == lane) {
            read_prefill_group(chunk, &first, 1);
            continue;
        }
        const PagedPrefillSequence pair[]{first, append_prefill_sequence(
            chunk, layer, bank, grants[other], other * batch_prefill_tile_rows)};
        if (const auto staged = try_long_staged_paged_prefill(pair, 2, chunk.stream,
                                                              chunk.work.gqa_scratch())) {
            check(*staged, "Read equal-position staged causal prefill MXFP8 attention");
        } else {
            for (const auto& sequence : pair) read_prefill_group(chunk, &sequence, 1);
        }
    }
}

bool can_batch_staged_gqa(const PrefillGrant* grants, int count) {
    if (count != staged_prefill_batch_size) {
        return false;
    }
    const int position = grants[0].position;
    return position >= staged_prefill_batch_min_position &&
        position <= staged_prefill_batch_long_max_position &&
        position % batch_prefill_tile_rows == 0 &&
        std::all_of(grants, grants + count, [&](const PrefillGrant& grant) {
            return grant.position == position && grant.rows == batch_prefill_tile_rows;
        });
}
