#include "sglang/runtime/execution/decoder_prefill_staged_gqa.h"
#include "sglang/text/gdn_local.h"
#include <algorithm>
#include <stdexcept>

namespace sglang::decoder::prefill_detail {

namespace {

PagedPrefillSequence append_prefill_sequence(const Chunk& chunk, const GqaLayer& layer, int bank,
                                            const PrefillGrant& grant, int row) {
    const auto segment = prefill_attention_segment(chunk, grant, row);
    const auto cache = grant.request->caches[bank];
    const auto pages = grant.request->pages;
    segment.gqa_append(layer, cache, pages);
    return {cache, pages, grant.position, grant.rows, segment.work.rope_queries,
            segment.work.attention};
}

bool read_staged_group(const Chunk& chunk, const PagedPrefillSequence* sequences, int count,
                       PairedPrefillScratch scratch) {
    const int position = sequences[0].position;
    if (position <= staged_prefill_batch_max_position) {
        const auto staged =
            try_batched_staged_paged_prefill(sequences, count, chunk.stream, scratch);
        if (staged)
            check(*staged, "Read staged batched causal prefill MXFP8 attention");
        return staged.has_value();
    }
    const auto staged = try_long_staged_paged_prefill(sequences, count, chunk.stream, scratch);
    if (staged)
        check(*staged, "Read long staged causal prefill MXFP8 attention");
    return staged.has_value();
}

bool prefill_grouped_partial_gqa(const Chunk& chunk, const GqaLayer& layer, int bank,
                                        const PrefillGrant* grants, int count,
                                        PairedPrefillScratch scratch, PairedPrefillScratch staging,
                                        bool postgate) {
    constexpr int width = staged_prefill_batch_size;
    if (count < width) return false;
    const int rows = grants[0].rows;
    const bool ordered = count % width == 0 && (rows == 256 || rows == 512) &&
        std::all_of(grants, grants + count, [rows](const PrefillGrant& grant) {
            return grant.rows == rows && grant.position >= 8192 && grant.position <= 32768 - rows;
        });
    int offset = 0;
    int first = 0;
    for (; first + width <= count; first += width) {
        std::array<PagedPrefillSequence, width> sequences;
        int packet_end = offset;
        for (int lane = 0; lane < width; ++lane) {
            const auto& grant = grants[first + lane];
            const auto segment = prefill_attention_segment(chunk, grant, packet_end);
            const auto cache = grant.request->caches[bank];
            const auto pages = grant.request->pages;
            segment.gqa_append(layer, cache, pages);
            sequences[lane] = {cache, pages, grant.position, grant.rows, segment.work.rope_queries,
                               segment.work.attention};
            packet_end += grant.rows;
        }
        auto submitted = launch_partial_staged_tmem_prefill(sequences.data(), width,
                                                            chunk.stream, staging);
        if (!submitted && ordered) {
            submitted = try_batched_partial_paged_prefill(sequences.data(), width, chunk.stream);
        }
        if (submitted) check(*submitted, "Read grouped ordered partial prefill attention");
        for (int lane = 0; lane < width; ++lane) {
            const auto& grant = grants[first + lane];
            const auto segment = prefill_attention_segment(chunk, grant, offset);
            const auto& sequence = sequences[lane];
            if (!submitted) segment.gqa_read(sequence.cache, sequence.pages, scratch);
            if (postgate) segment.gqa_gate();
            offset += grant.rows;
        }
    }
    for (; first < count; ++first) {
        const auto& grant = grants[first];
        const auto segment = prefill_attention_segment(chunk, grant, offset);
        segment.gqa_attention(layer, grant.request->caches[bank], grant.request->pages,
                              scratch, postgate);
        offset += grant.rows;
    }
    return true;
}

bool prefill_causal_group_gqa(const Chunk& chunk, const GqaLayer& layer, int bank,
                                     const PrefillGrant* grants, int count,
                                     PairedPrefillScratch serial_scratch, bool postgate) {
    constexpr int tile_rows = batch_prefill_tile_rows;
    constexpr int group_rows = grouped_prefill_max_rows;
    constexpr int long_context = staged_prefill_batch_long_max_position + tile_rows;
    const auto& first = grants[0];
    const bool resident = chunk.long_staging.resident && first.media == nullptr &&
                          wants_resident_gqa_staging(first.request->cached_gqa_tokens,
                                                     first.position, count * tile_rows);
    const int rows = count * tile_rows;
    if (resident) {
        if (rows != group_rows && rows != group_rows - tile_rows) {
            return false;
        }
    } else if (rows != group_rows || first.position < 0 ||
               first.position > long_context - group_rows || first.position % tile_rows != 0) {
        return false;
    }
    for (int index = 0; index < count; ++index) {
        const auto& grant = grants[index];
        if (grant.request != first.request || grant.media || grant.rows != tile_rows ||
            grant.position != first.position + index * tile_rows) return false;
    }
    const auto cache = first.request->caches[bank];
    const auto pages = first.request->pages;
    auto combined = first;
    combined.rows = rows;
    const auto segment = prefill_attention_segment(chunk, combined, 0);
    segment.gqa_append(layer, cache, pages);
    const PagedPrefillSequence sequence{
        cache, pages, first.position, rows, chunk.work.rope_queries, chunk.work.attention,
    };
    const auto submitted =
        resident
            ? try_resident_staged_paged_prefill(sequence, chunk.stream, chunk.long_staging.scratch)
            : launch_causal_group_staged_tmem_prefill(sequence, chunk.stream,
                                                      chunk.work.gqa_scratch());
    if (submitted) {
        check(*submitted, "Read staged single-request causal prefill group");
    } else {
        for (int lane = 0, offset = 0; lane < count; offset += grants[lane++].rows) {
            const auto& grant = grants[lane];
            const auto tile = prefill_attention_segment(chunk, grant, offset);
            check(tensor_core_multirow_prefill_gqa(cache, pages, grant.position, grant.rows,
                                                   tile.work.rope_queries, tile.work.attention,
                                                   chunk.stream, serial_scratch),
                  "Read causal prefill MXFP8 attention");
        }
    }
    if (postgate) segment.gqa_gate();
    return true;
}

PairedPrefillScratch serial_gqa_scratch(const TileBuffers& buffers, int serial_capacity) {
    ScratchLayout serial_layout{reinterpret_cast<std::byte*>(buffers.hidden)};
    const auto sizes = TileBuffers(serial_layout, serial_capacity, buffers.dims).gqa_scratch();
    auto scratch = buffers.gqa_scratch();
    scratch.key_bytes = sizes.key_bytes;
    scratch.value_bytes = sizes.value_bytes;
    return scratch;
}

}

void prefill_group_gdn(const Chunk& chunk, const GdnLayer& layer, int bank,
                              const PrefillGrant* grants, int count, int offset) {
    const auto& buffers = chunk.work;
    const auto& dims = buffers.dims;
    GdnPrefillSegment segments[max_gdn_prefill_segments]{};
    std::array<GdnPrefillCheckpoints, max_gdn_prefill_segments> captures;
    bool grouped_recurrence = count > 1;
    int index = 0;
    for (int lane = 0, row = offset; lane < count; row += grants[lane++].rows) {
        const auto& grant = grants[lane];
        const auto segment = buffers.attention_segment(row);
        const TextFp8Activation fp8{
            buffers.activation.values + std::size_t(row) * dims.gdn_value_width,
            buffers.activation.scales + std::size_t(row) * dims.gdn_heads,
        };
        grouped_recurrence = grouped_recurrence && grant.rows % prefill_chunk_rows == 0 &&
                             (grant.position != 0 || bank != 0);
        captures[index] = GdnPrefillCheckpoints(grant, bank);
        const auto scratch_offset = dims.is_local_gdn() ? text_gdn_local_workspace_bytes(row)
                                                        : text_gdn_workspace_bytes(row);
        segments[index] = {
            grant.rows,
            segment.gdn_qkv,
            segment.gdn_z,
            segment.gdn_b,
            segment.gdn_a,
            grant.request->states[bank],
            reinterpret_cast<Bf16*>(reinterpret_cast<std::byte*>(buffers.gdn_scratch) +
                                    scratch_offset),
            reinterpret_cast<float2*>(segment.keys),
            fp8,
            buffers.mlp_gate + row * dims.intermediate,
            std::size_t(grant.rows) * dims.intermediate * sizeof(Bf16),
            captures[index].view(),
        };
        ++index;
    }
    const auto status = grouped_recurrence
        ? (dims.is_local_gdn()
               ? tensor_core_prefill_gdn_group_local(segments, count, layer.recurrent,
                                                     chunk.stream)
               : tensor_core_prefill_gdn_group(segments, count, layer.recurrent, chunk.stream))
        : std::nullopt;
    if (status) {
        if (*status != cudaSuccess && cudaPeekAtLastError() == *status) {
            (void)cudaGetLastError();
        }
        check(*status, "Advance grouped prefill Gated DeltaNet");
        return;
    }
    index = 0;
    for (int lane = 0, row = offset; lane < count; row += grants[lane++].rows) {
        const auto& grant = grants[lane];
        const auto& segment = segments[index++];
        prefill_attention_segment(chunk, grant, row)
            .gdn_recur(layer, segment.state, bank, segment.output, segment.checkpoints);
    }
}

bool is_full_partial_prefill_group(const PrefillGrant* grants, PrefillGroup group) {
    const int rows = grants[0].rows;
    if (group.rows != grouped_prefill_max_rows || (rows != 256 && rows != 512) ||
        group.grants % max_gdn_prefill_segments != 0) {
        return false;
    }
    for (int index = 1; index < group.grants; ++index) {
        if (grants[index].rows != rows) return false;
    }
    return true;
}

void prefill_partial_gdn(const Chunk& chunk, const GdnLayer& layer, int bank,
                                const PrefillGrant* grants, int count, bool grouped) {
    const auto& activation = chunk.work.activation;
    const auto& dims = chunk.work.dims;
    if (grouped) {
        int offset = 0;
        for (int index = 0; index < count; index += max_gdn_prefill_segments) {
            prefill_group_gdn(chunk, layer, bank, grants + index, max_gdn_prefill_segments, offset);
            offset += max_gdn_prefill_segments * grants[index].rows;
        }
        return;
    }
    if (std::size_t(dims.kv_width) * sizeof(Bf16) <
        std::size_t(dims.gdn_heads) * sizeof(float2)) {
        throw std::logic_error("Prefill key buffer too narrow for Gated DeltaNet gates");
    }
    int index = 0;
    int row = 0;
    while (index < count) {
        const PrefillGrant& first = grants[index];
        int run = 1;
        while (index + run < count && grants[index + run].request == first.request &&
               !grants[index + run].media && grants[index + run].rows == first.rows &&
               grants[index + run].position == first.position + run * first.rows) {
            ++run;
        }
        const int rows = run * first.rows;
        if (activation.values && chunk.tensor_core_fp8 && dims.is_local_gdn() &&
            count == 4 && run == 4 && rows == grouped_prefill_max_rows &&
            first.rows == batch_prefill_tile_rows && !first.media &&
            first.position % batch_prefill_tile_rows == 0) {
            const bool ordered_first = bank == 0 && first.position == 0;
            if (ordered_first) {
                const auto segment = prefill_attention_segment(chunk, first, row);
                const GdnPrefillCheckpoints captures(first, bank);
                segment.gdn_recur(layer, first.request->states[bank], bank, activation,
                                  captures.view());
            }
            const int start = ordered_first ? 1 : 0;
            const int offset = row + start * batch_prefill_tile_rows;
            auto combined = grants[index + start];
            combined.rows = rows - start * batch_prefill_tile_rows;
            const auto segment = prefill_attention_segment(chunk, combined, offset);
            const TextFp8Activation quantized{
                activation.values + std::size_t(offset) * dims.gdn_value_width,
                activation.scales + std::size_t(offset) * dims.gdn_heads,
            };
            const GdnPrefillCheckpoints captures(combined, bank);
            segment.gdn_recur(layer, first.request->states[bank], bank, quantized, captures.view());
        } else {
            for (int step = 0; step < run; ++step) {
                const auto segment =
                    prefill_attention_segment(chunk, grants[index + step], row);
                const TextFp8Activation quantized{
                    activation.values + std::size_t(row) * dims.gdn_value_width,
                    activation.scales + std::size_t(row) * dims.gdn_heads,
                };
                const auto& grant = grants[index + step];
                const GdnPrefillCheckpoints captures(grant, bank);
                segment.gdn_recur(layer, grant.request->states[bank], bank, quantized,
                                  captures.view());
                row += grants[index + step].rows;
            }
            index += run;
            continue;
        }
        row += rows;
        index += run;
    }
}

void prefill_serial_gqa(const Chunk& chunk, const GqaLayer& layer, int bank,
                               const PrefillGrant* grants, int count, int serial_capacity,
                               bool causal, bool postgate) {
    const auto scratch = serial_gqa_scratch(chunk.work, serial_capacity);
    if (causal) {
        if (prefill_causal_group_gqa(chunk, layer, bank, grants, count, scratch, postgate)) return;
    } else if (prefill_grouped_partial_gqa(
                   chunk, layer, bank, grants, count, scratch, chunk.work.gqa_scratch(),
                   postgate)) {
        return;
    }
    for (int lane = 0, offset = 0; lane < count; offset += grants[lane++].rows) {
        const auto& grant = grants[lane];
        const auto segment = prefill_attention_segment(chunk, grant, offset);
        segment.gqa_attention(layer, grant.request->caches[bank], grant.request->pages,
                              scratch, postgate);
    }
}

void prefill_group_gqa(const Chunk& chunk, const GqaLayer& layer, int bank,
                      const PrefillGrant* grants, int count, int offset,
                      bool staged_batch) {
    std::array<PagedPrefillSequence, staged_prefill_batch_size> sequences;
    int index = 0;
    for (int lane = 0, row = offset; lane < count; row += grants[lane++].rows) {
        const auto& grant = grants[lane];
        sequences[index++] = append_prefill_sequence(chunk, layer, bank, grant, row);
    }
    const auto scratch = chunk.work.gqa_scratch();
    if (!staged_batch) {
        if (const auto batched = try_batched_paged_prefill(sequences.data(), count, chunk.stream)) {
            check(*batched, "Read batched causal prefill MXFP8 attention");
            return;
        }
    }
    const int visible = grants[0].position + batch_prefill_tile_rows;
    int row = offset;
    for (int lane = 0; lane < count;) {
        const int left = count - lane;
        int width = 1;
        if (left == 4 && visible <= 32768) {
            width = 4;
        } else if (left >= 2 && visible <= 65536) {
            width = 2;
        }
        const bool staged =
            staged_batch && read_staged_group(chunk, sequences.data() + lane, width, scratch);
        for (const int end = lane + width; lane < end; ++lane) {
            if (!staged) {
                const auto& sequence = sequences[lane];
                prefill_attention_segment(chunk, grants[lane], row)
                    .gqa_read(sequence.cache, sequence.pages, scratch);
            }
            row += grants[lane].rows;
        }
    }
}

std::optional<PrefillPartners> mixed_prefill_partners(const PrefillGrant* grants, int count) {
    if (count < 2 || count > staged_prefill_batch_size)
        return std::nullopt;
    PrefillPartners partners;
    for (int lane = 0; lane < count; ++lane) {
        const auto& grant = grants[lane];
        if (grant.rows != batch_prefill_tile_rows)
            return std::nullopt;
        for (int prior = 0; prior < lane; ++prior) {
            if (grants[prior].request == grant.request) return std::nullopt;
        }
        partners[lane] = lane;
    }
    bool paired = false;
    for (int lane = 0; lane < count; ++lane) {
        const int position = grants[lane].position;
        if (grants[lane].media != nullptr || partners[lane] != lane ||
            position <= staged_prefill_batch_max_position ||
            position > 65536 - batch_prefill_tile_rows || position % batch_prefill_tile_rows) {
            continue;
        }
        for (int next = lane + 1; next < count; ++next) {
            if (grants[next].media != nullptr || partners[next] != next ||
                grants[next].position != position)
                continue;
            partners[lane] = next;
            partners[next] = lane;
            paired = true;
            break;
        }
    }
    return paired ? std::optional{partners} : std::nullopt;
}

void prefill_mixed_gqa(const Chunk& chunk, const GqaLayer& layer, int bank,
                       const PrefillGrant* grants, int count, const PrefillPartners& partners) {
    const auto scratch = chunk.work.gqa_scratch();
    const auto read = [&](int lane, const PagedPrefillSequence& sequence) {
        prefill_attention_segment(chunk, grants[lane], lane * batch_prefill_tile_rows)
            .gqa_read(sequence.cache, sequence.pages, scratch);
    };
    for (int lane = 0; lane < count; ++lane) {
        const int other = partners[lane];
        if (other < lane) continue;
        const auto first = append_prefill_sequence(chunk, layer, bank, grants[lane],
                                                  lane * batch_prefill_tile_rows);
        if (other == lane) {
            const bool long_lane = grants[lane].media == nullptr &&
                                   chunk.rows == grouped_prefill_max_rows &&
                                   first.position >= staged_serial_min_position;
            if (!long_lane || !read_staged_group(chunk, &first, 1, scratch))
                read(lane, first);
            continue;
        }
        const PagedPrefillSequence pair[]{first, append_prefill_sequence(
            chunk, layer, bank, grants[other], other * batch_prefill_tile_rows)};
        if (!read_staged_group(chunk, pair, 2, scratch)) {
            read(lane, pair[0]);
            read(other, pair[1]);
        }
    }
}

bool can_batch_staged_gqa(const PrefillGrant* grants, int count) {
    if (count != staged_prefill_batch_size && count != 2)
        return false;
    const int position = grants[0].position;
    const int minimum =
        count == 2 ? staged_prefill_batch_max_position + 1 : staged_prefill_batch_min_position;
    return position >= minimum &&
           position <= (count == 2 ? 65536 - batch_prefill_tile_rows
                                   : staged_prefill_batch_long_max_position) &&
           position % batch_prefill_tile_rows == 0 &&
           std::all_of(grants, grants + count, [&](const PrefillGrant& grant) {
               return !grant.media && grant.position == position &&
                      grant.rows == batch_prefill_tile_rows;
           });
}

}
