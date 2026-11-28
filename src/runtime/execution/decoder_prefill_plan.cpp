#include "sglang/runtime/execution/decoder_prefill_plan.h"
#include <algorithm>
#include <bitset>
#include <optional>
#include <stdexcept>

namespace sglang::decoder {

PrefillGrant prefill_tile(PrefillGrant grant, PrefillTileSpan span) {
    grant.tokens += span.offset;
    grant.position += span.offset;
    grant.prompt_remaining -= span.offset;
    grant.rows = span.rows;
    return grant;
}

}

namespace sglang::decoder::prefill_detail {

namespace {

struct ProjectionStep {
    PrefillGroup group{};
    PrefillAttentionMode attention = PrefillAttentionMode::serial_segments;
    std::array<PrefillSegment, decoder_batch_max_lanes> segments{};
};

struct ProjectionCursor {
    const PrefillGrant* grants = nullptr;
    int count = 0;
    int min_rows = 0;
    int max_rows = 0;
    bool whole_tiles = false;
    bool ranked = false;
    bool resident = false;
    std::array<int, decoder_batch_max_lanes> offsets{};

    int remaining(int source) const {
        return grants[source].rows - offsets[source];
    }
    int position(int source) const {
        return grants[source].position + offsets[source];
    }

    bool has_tile(int source) const {
        return grants[source].tile_rows == batch_prefill_tile_rows &&
               remaining(source) >= batch_prefill_tile_rows;
    }

    void add(ProjectionStep& step, int source, int rows) {
        step.segments[step.group.grants++] = {source, offsets[source]};
        step.group.rows += rows;
        offsets[source] += rows;
    }

    bool cold_pair(ProjectionStep& step, int first) {
        constexpr int tile = batch_prefill_tile_rows;
        const auto cold = [&](int source) {
            const auto& grant = grants[source];
            return has_tile(source) && remaining(source) >= 2 * tile && position(source) == 0 &&
                   grant.media == nullptr && grant.request != nullptr &&
                   grant.request->cached_gqa_tokens == 0;
        };
        if (!ranked || max_rows < 4 * tile || !cold(first))
            return false;
        for (int peer = first + 1; peer < count; ++peer) {
            if (!cold(peer))
                continue;
            for (const int source : {first, peer}) {
                add(step, source, tile);
                add(step, source, tile);
            }
            return true;
        }
        return false;
    }

    void select_warm_pairs(std::bitset<decoder_batch_max_lanes>& sources, int first,
                           int& lanes) const {
        constexpr int tile = batch_prefill_tile_rows;
        const int available = max_rows / tile;
        for (int next = first + 1; lanes + 2 <= available && next + 1 < count; ++next) {
            const int start = position(next);
            if (sources[next] || !has_tile(next) || grants[next].media != nullptr ||
                start % tile != 0 || start <= staged_prefill_batch_max_position ||
                start > 65536 - tile)
                continue;
            for (int partner = next + 1; partner < count; ++partner) {
                if (sources[partner] || !has_tile(partner) || grants[partner].media != nullptr ||
                    position(partner) != start)
                    continue;
                sources.set(next);
                sources.set(partner);
                lanes += 2;
                break;
            }
        }
    }

    bool full_tiles(ProjectionStep& step, int first) {
        constexpr int tile = batch_prefill_tile_rows;
        if (!whole_tiles || !has_tile(first) || max_rows < 2 * tile)
            return false;
        std::bitset<decoder_batch_max_lanes> sources;
        sources.set(first);
        int lanes = 1;
        const int available = max_rows / tile;
        const int start = position(first);
        const bool prefer_equal = grants[first].media == nullptr &&
                                  start >= staged_prefill_batch_min_position &&
                                  start <= 65536 - tile && start % tile == 0 &&
                                  (start > staged_prefill_batch_max_position || available == 4);
        if (prefer_equal) {
            for (int next = first + 1; next < count && lanes < available; ++next) {
                if (has_tile(next) && grants[next].media == nullptr && position(next) == start) {
                    sources.set(next);
                    ++lanes;
                }
            }
        }
        select_warm_pairs(sources, first, lanes);
        for (int next = first + 1; next < count && lanes < available; ++next) {
            if (has_tile(next) && !sources[next]) {
                sources.set(next);
                ++lanes;
            }
        }
        if (lanes < 2)
            return false;
        for (int source = first; source < count; ++source) {
            if (sources[source])
                add(step, source, tile);
        }
        step.attention = step.group.rows == grouped_prefill_max_rows
                             ? PrefillAttentionMode::full_group
                             : PrefillAttentionMode::batched;
        return true;
    }

    void causal_group(ProjectionStep& step, int source) {
        const int tiles = std::min(remaining(source), max_rows) / batch_prefill_tile_rows;
        step.attention = PrefillAttentionMode::causal_segments;
        for (int lane = 0; lane < tiles; ++lane) {
            add(step, source, batch_prefill_tile_rows);
        }
    }

    bool prefer_resident_group(int source, int tiles) const {
        const auto& grant = grants[source];
        const int start = position(source);
        return resident && tiles >= 3 && has_tile(source) && grant.media == nullptr &&
               grant.request != nullptr && start % batch_prefill_tile_rows == 0 &&
               wants_resident_gqa_staging(grant.request->cached_gqa_tokens, start,
                                          tiles * batch_prefill_tile_rows);
    }

    bool next(ProjectionStep& step) {
        step.group = {};
        step.attention = PrefillAttentionMode::serial_segments;
        int first = 0;
        while (first < count && remaining(first) == 0)
            ++first;
        if (first == count)
            return false;
        const int first_rows = remaining(first);
        const int causal_tiles = std::min(first_rows, max_rows) / batch_prefill_tile_rows;
        if (prefer_resident_group(first, causal_tiles)) {
            causal_group(step, first);
            return true;
        }
        if (cold_pair(step, first) || full_tiles(step, first))
            return true;
        if (has_tile(first) && causal_tiles >= 3) {
            causal_group(step, first);
            return true;
        }
        add(step, first, first_rows);
        if (first_rows < min_rows || first_rows > grants[first].tile_rows ||
            first_rows > batch_prefill_tile_rows)
            return true;
        for (int next = first + 1; next < count; ++next) {
            const int rows = remaining(next);
            if (rows >= min_rows && rows <= grants[next].tile_rows &&
                rows <= batch_prefill_tile_rows && rows <= max_rows - step.group.rows) {
                add(step, next, rows);
            }
        }
        return true;
    }
};

int projection_capacity(ProjectionCursor cursor) {
    ProjectionStep step;
    int capacity = 0;
    while (cursor.next(step)) {
        if (step.group.grants > 1)
            capacity = std::max(capacity, step.group.rows);
    }
    return capacity;
}

void dispatch_projection_groups(ProjectionCursor& cursor, const PartialProjectionSchedule& schedule,
                                PrefillWorkspace& workspace, int serial_capacity,
                                DecoderBatchStepStats& stats) {
    ProjectionStep step;
    std::array<PrefillGrant, decoder_batch_max_lanes> tiles;
    while (cursor.next(step)) {
        const auto [source, offset] = step.segments[0];
        if (step.group.grants == 1) {
            schedule.native(schedule.state, source, cursor.grants + source, workspace,
                            serial_capacity, offset);
            continue;
        }
        for (int lane = 0; lane < step.group.grants; ++lane) {
            const auto [owner, begin] = step.segments[lane];
            const auto& grant = cursor.grants[owner];
            const int rows = std::min(grant.tile_rows, grant.rows - begin);
            tiles[lane] = prefill_tile(grant, {.offset = begin, .rows = rows});
            validate_prefill_tile(tiles[lane], workspace, serial_capacity);
        }
        schedule.group(schedule.state, step.segments.data(), tiles.data(), step.group, workspace,
                       step.attention, serial_capacity);
        ++stats.layerwise_prefill_groups;
        stats.layerwise_prefill_rows += step.group.rows;
    }
}

}

int long_serial_visible(const PrefillGrant* grants, int count, bool grouped) {
    int visible = 0;
    for (int index = 0; index < count; ++index) {
        const auto& grant = grants[index];
        if (grant.tile_rows < staged_serial_min_rows) continue;
        const int grouped_rows = grouped && grant.tile_rows == batch_prefill_tile_rows &&
                grant.position % batch_prefill_tile_rows == 0
            ? grant.rows / grouped_prefill_max_rows * grouped_prefill_max_rows : 0;
        for (int offset = grouped_rows; offset < grant.rows; offset += grant.tile_rows) {
            const int rows = std::min(grant.tile_rows, grant.rows - offset);
            const int position = grant.position + offset;
            if (wants_long_serial_staging(position, rows)) {
                visible = std::max(visible, position + rows);
            }
        }
    }
    return visible;
}

bool wants_resident_gqa_staging(int cached_tokens, int position, int rows) {
    return cached_tokens >= staged_serial_min_position && position >= staged_serial_min_position &&
           rows >= staged_serial_min_rows && position <= text_attention_max_tokens - rows;
}

int resident_gqa_staging_visible(const PrefillGrant* grants, int count) {
    int visible = 0;
    for (int index = 0; index < count; ++index) {
        const auto& grant = grants[index];
        if (grant.request == nullptr || grant.media != nullptr ||
            grant.tile_rows < staged_serial_min_rows)
            continue;
        const int tail = grant.rows % grant.tile_rows;
        const int rows = tail >= staged_serial_min_rows ? tail : grant.tile_rows;
        const int end = grant.position + grant.rows - (tail < staged_serial_min_rows ? tail : 0);
        if (end - grant.position >= rows &&
            wants_resident_gqa_staging(grant.request->cached_gqa_tokens, end - rows, rows)) {
            visible = std::max(visible, end);
        }
    }
    return visible;
}

void validate_prefill_tile(const PrefillGrant& grant, const PrefillWorkspace& workspace,
                                  int capacity) {
    const auto rows = grant.rows;
    const auto position = grant.position;
    if (workspace.stream() == nullptr || grant.request == nullptr || grant.tokens == nullptr ||
        rows < 1 || rows > capacity || capacity > workspace.capacity() || position < 0 ||
        position > text_attention_max_tokens - rows ||
        (position + rows - 1) / mxfp8_page_size >= grant.request->pages.page_count ||
        (grant.draft != nullptr && grant.prompt_remaining < rows)) {
        throw std::invalid_argument("Invalid decoder prefill tile");
    }
}

void validate_ranked_prefill_grants(std::array<const PrefillGrant*, 2> ranks, int count) {
    for (int index = 0; index < count; ++index) {
        const auto& grant = ranks[0][index];
        const auto& mirror = ranks[1][index];
        if (grant.tokens != mirror.tokens || grant.position != mirror.position ||
            grant.rows != mirror.rows || grant.tile_rows != mirror.tile_rows ||
            grant.prompt_remaining != mirror.prompt_remaining ||
            grant.request->cached_gqa_tokens != mirror.request->cached_gqa_tokens) {
            throw std::invalid_argument("Mismatched prefill grants across tensor-parallel ranks");
        }
        if ((grant.draft != nullptr) != (mirror.draft != nullptr) || mirror.scores != nullptr ||
            mirror.media != nullptr) {
            throw std::invalid_argument("Invalid peer prefill draft, media, or score capture");
        }
        if (grant.media != nullptr) {
            throw std::invalid_argument(
                "Multimodal prefill is unsupported under tensor parallelism");
        }
    }
}

bool execute_partial_projection_groups(const PrefillGrant* grants, int count, cudaStream_t stream,
                                       DecoderBatchStepStats& stats,
                                       const PartialProjectionSchedule& schedule, int min_rows,
                                       GqaGeometry staged_geometry, int staged_visible,
                                       bool resident_staging, bool whole_tiles, bool ranked) {
    if (count == 0)
        return false;
    if (count < 0 || count > decoder_batch_max_lanes) {
        throw std::invalid_argument("Invalid number of prefill grants");
    }
    int serial_capacity = 1;
    for (int index = 0; index < count; ++index) {
        serial_capacity = std::max(serial_capacity, grants[index].tile_rows);
        for (int prior = 0; prior < index; ++prior) {
            if (grants[prior].request == grants[index].request)
                return false;
        }
    }
    int max_rows = grouped_prefill_max_rows;
    bool resident = false;
    std::optional<PrefillWorkspace> workspace;
    const auto prepare = [&](int capacity, bool require_resident) {
        workspace.emplace(stream, capacity, staged_geometry, staged_visible, resident_staging);
        if (workspace->data() == nullptr)
            return false;
        resident = workspace->resident_gqa_staging();
        if ((!require_resident || resident) &&
            (schedule.prepare == nullptr || schedule.prepare(schedule.state, capacity, resident)) &&
            (!require_resident || resident))
            return true;
        workspace->close();
        check(cudaStreamSynchronize(stream), "Release provisional prefill workspace");
        return false;
    };
    for (;;) {
        ProjectionCursor cursor{grants, count, min_rows, max_rows, whole_tiles, ranked};
        const int grouped_capacity = projection_capacity(cursor);
        if (grouped_capacity == 0)
            return false;
        const int capacity = std::max(grouped_capacity, serial_capacity);
        int resident_capacity = 0;
        if (resident_staging && max_rows >= 3 * batch_prefill_tile_rows) {
            cursor.resident = true;
            resident_capacity = projection_capacity(cursor);
            if (resident_capacity > capacity && prepare(resident_capacity, true))
                break;
        }
        if (prepare(capacity, false)) {
            resident = resident && resident_capacity <= capacity;
            break;
        }
        constexpr int pair_rows = 2 * batch_prefill_tile_rows;
        if (capacity <= std::max(pair_rows, serial_capacity))
            return false;
        max_rows = pair_rows;
    }

    stats.prefill_workspace_bytes = std::max(stats.prefill_workspace_bytes, workspace->bytes());
    ProjectionCursor cursor{grants, count, min_rows, max_rows, whole_tiles, ranked, resident};
    dispatch_projection_groups(cursor, schedule, *workspace, serial_capacity, stats);
    workspace->close();
    return true;
}
}
