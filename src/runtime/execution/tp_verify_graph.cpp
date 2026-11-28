#include "sglang/runtime/execution/tp_verify_graph.h"
#include "sglang/runtime/backend/tensor_parallel.h"
#include "sglang/runtime/execution/decoder_forward_layers.h"
#include "sglang/runtime/execution/verification_row_copy.h"
#include "sglang/text/gdn_local.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <new>
#include <tuple>

namespace sglang::decoder {

template <>
tp_verify_graph::DeviceBinding*
Allocations::allocate<tp_verify_graph::DeviceBinding>(std::size_t count) {
    return static_cast<tp_verify_graph::DeviceBinding*>(
        allocate_elements(count, sizeof(tp_verify_graph::DeviceBinding)));
}

}

namespace sglang::decoder::tp_verify_graph {

namespace {
std::array<HostBinding*, 2>& active_bindings() {
    thread_local std::array<HostBinding*, 2> active{};
    return active;
}

bool capture_rejected(cudaError_t status) {
    return status == cudaErrorNotSupported || status == cudaErrorMemoryAllocation ||
        status == cudaErrorStreamCaptureUnsupported ||
        status == cudaErrorStreamCaptureInvalidated ||
        status == cudaErrorStreamCaptureUnjoined;
}

void cleanup(cudaError_t status, const char* operation) noexcept {
    if (status != cudaSuccess) {
        static_cast<void>(std::fputs("sglang decoder: ", stderr));
        static_cast<void>(std::fputs(operation, stderr));
        static_cast<void>(std::fputs(": ", stderr));
        static_cast<void>(std::fputs(cudaGetErrorString(status), stderr));
        static_cast<void>(std::fputc('\n', stderr));
    }
}

bool streams_available(const std::array<int, 2>& devices,
                       const std::array<cudaStream_t, 2>& streams) {
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceScope scope(devices[rank]);
        auto status = cudaStreamCaptureStatusNone;
        check(cudaStreamIsCapturing(streams[rank], &status), "Check TP verifier stream capture");
        if (status != cudaStreamCaptureStatusNone) return false;
    }
    return true;
}

constexpr int graph_slots = 4;
constexpr int collectives_per_forward = 2 * layer_count;

struct LaneLayout {
    std::array<int, target_verify_max_rows> rows{};
    int depth = 0;
    int long_start = 0;
    bool operator==(const LaneLayout& other) const;
};

bool LaneLayout::operator==(const LaneLayout& other) const {
    return rows == other.rows && depth == other.depth && long_start == other.long_start;
}
}

int packet_rows(const VerificationRow* rows, int remaining) {
    constexpr int lanes = decoder_batch_max_mtp_lanes;
    const int packet = rows[0].binding_lane / lanes;
    const int limit = std::min(remaining, verification_packet_rows);
    int count = 1;
    while (count < limit && rows[count].binding_lane / lanes == packet)
        ++count;
    return count;
}

bool SplitGroup::operator==(const SplitGroup& other) const {
    return first == other.first && count == other.count && variant == other.variant;
}

namespace {

using Input = std::array<const VerificationRow*, 2>;
using DecodeInput = std::array<const ForwardLane*, 2>;

struct Layout {
    std::array<LaneLayout, max_lanes> lanes{};
    std::array<int, max_rows> short_rows{};
    std::array<int, max_rows> split_rows{};
    int row_count = 0;
    int lane_count = 0;
    int max_depth = 0;
    int short_count = 0;
    int split_count = 0;
    int long_count = 0;
    bool singleton_fused = false;
    std::array<SplitGroup, max_rows / verification_packet_rows> splits{};
    int split_groups = 0;
    gqa_decode::LongDecodeShape long_shape{};
    std::uint32_t decode_split = 0;
    bool ordinary = false;
    bool operator==(const Layout& other) const;
};

bool Layout::operator==(const Layout& other) const {
    return lanes == other.lanes && short_rows == other.short_rows &&
           split_rows == other.split_rows && row_count == other.row_count &&
           lane_count == other.lane_count && max_depth == other.max_depth &&
           short_count == other.short_count && split_count == other.split_count &&
           long_count == other.long_count && singleton_fused == other.singleton_fused &&
           splits == other.splits && split_groups == other.split_groups &&
           long_shape == other.long_shape && decode_split == other.decode_split &&
           ordinary == other.ordinary;
}

bool distinct_seed(const VerificationRow* packet, const VerificationRow& seed, const Layout& layout,
                   int lane) {
    for (int prior = 0; prior < lane; ++prior) {
        const auto& existing = packet[layout.lanes[prior].rows[0]];
        if (seed.request->pages.device_pages == existing.request->pages.device_pages ||
            seed.verification == existing.verification)
            return false;
    }
    return true;
}

int seed_depth(const VerificationRow* packet, const VerificationRow& seed, cudaStream_t stream,
               const Layout& layout, int lane) {
    if (!seed.first || seed.request == nullptr || seed.verification == nullptr ||
        seed.position < 0 || !seed.verification->rank_local_gdn() ||
        seed.verification->stream() != stream)
        return 0;
    const int depth = seed.verification->pending_rows();
    if (depth < 1 || depth > target_verify_max_rows ||
        seed.position > text_attention_max_tokens - depth ||
        !distinct_seed(packet, seed, layout, lane))
        return 0;
    for (const auto& cache : seed.request->caches) {
        if (cache.kv_heads != gqa_rank_geometry.kv_heads ||
            !cache_covers(cache, seed.request->pages, seed.position + depth))
            return 0;
    }
    return depth;
}

bool pack_rows(const VerificationRow* packet, const Layout& layout, LaneLayout* begin,
               const LaneLayout* end, int longest, int& index) {
    for (int depth = 0; depth < longest; ++depth) {
        for (auto* lane = begin; lane != end; ++lane) {
            auto& entry = *lane;
            if (depth >= entry.depth)
                continue;
            if (index == layout.row_count)
                return false;
            const auto& seed = packet[entry.rows[0]];
            const auto& row = packet[index];
            if (row.binding != nullptr || row.constrained || row.work == nullptr ||
                row.first != (depth == 0) || row.verification != seed.verification ||
                row.request != seed.request || row.position != seed.position + depth)
                return false;
            entry.rows[depth] = index++;
        }
    }
    return true;
}

bool plan_lanes(const VerificationRow* packet, cudaStream_t stream, Layout& layout) {
    int first = 0;
    for (int lane = 0; lane < layout.lane_count;) {
        const int width = std::min(decoder_batch_max_mtp_lanes, layout.lane_count - lane);
        int longest = 0;
        for (int offset = 0; offset < width; ++offset) {
            if (first + offset >= layout.row_count)
                return false;
            const auto& seed = packet[first + offset];
            const int depth = seed_depth(packet, seed, stream, layout, lane + offset);
            if (depth == 0)
                return false;
            auto& entry = layout.lanes[lane + offset];
            entry.depth = depth;
            entry.rows[0] = first + offset;
            longest = std::max(longest, depth);
        }
        layout.max_depth = std::max(layout.max_depth, longest);
        auto* begin = layout.lanes.data() + lane;
        if (!pack_rows(packet, layout, begin, begin + width, longest, first))
            return false;
        lane += width;
    }
    return first == layout.row_count;
}

void plan_attention(Layout& layout, const VerificationRow* packet, bool packed) {
    gqa_decode::LongBatch long_batch;
    GqaVerifySelection selected{};
    layout.long_count = plan_gqa_verify_long(packet, layout.row_count, 0, long_batch, selected);
    if (layout.long_count != 0) {
        layout.long_shape = gqa_decode::long_decode_shape(long_batch, layout.long_count);
    }
    for (int lane = 0; lane < layout.lane_count; ++lane) {
        auto& entry = layout.lanes[lane];
        while (entry.long_start < entry.depth &&
               (layout.long_count == 0 || !selected[entry.rows[entry.long_start]])) {
            ++entry.long_start;
        }
    }
    for (int first = 0; first < layout.row_count;) {
        const int width =
            packed ? packet_rows(packet + first, layout.row_count - first) : layout.row_count;
        std::array<PagedDecodeLane, verification_packet_rows> reads;
        const int start = layout.split_count;
        for (int index = first; index < first + width; ++index) {
            if (layout.long_count != 0 && selected[index])
                continue;
            const auto& row = packet[index];
            if (row.position < target_split_first_position) {
                layout.short_rows[layout.short_count++] = index;
            } else {
                reads[layout.split_count - start] = {
                    row.request->caches[0],
                    row.request->pages,
                    row.position,
                    paged_gqa_buffers(*row.work),
                };
                layout.split_rows[layout.split_count++] = index;
            }
        }
        const int count = layout.split_count - start;
        if (count != 0) {
            const auto variant =
                gqa_decode::split_variant(reads.data(), count, text_attention_max_tokens);
            layout.splits[layout.split_groups++] = {start, count, variant};
        }
        first += width;
    }
}

std::optional<Layout> plan(const Input& input, int lanes, int rows,
                           const std::array<cudaStream_t, 2>& streams, bool singleton_fused,
                           bool packed) {
    if (lanes < 1 || lanes > max_lanes || rows < lanes || rows > max_rows ||
        (!packed && lanes > decoder_batch_max_mtp_lanes))
        return std::nullopt;
    Layout layout{};
    Layout peer{};
    for (auto* key : {&layout, &peer}) {
        key->lane_count = lanes;
        key->row_count = rows;
        key->singleton_fused = singleton_fused;
    }
    if (!plan_lanes(input[0], streams[0], layout) || !plan_lanes(input[1], streams[1], peer))
        return std::nullopt;
    for (int row = 0; row < rows; ++row) {
        if (input[0][row].position != input[1][row].position)
            return std::nullopt;
    }
    plan_attention(layout, input[0], packed);
    plan_attention(peer, input[1], packed);
    if (!(layout == peer))
        return std::nullopt;
    return layout;
}

std::optional<Layout> plan_decode(const DecodeInput& input, int count) {
    static_assert(decoder_batch_max_lanes <= 32 && decoder_batch_max_lanes <= max_rows);
    if (count < 1 || count > decoder_batch_max_lanes)
        return std::nullopt;
    Layout layout{};
    layout.ordinary = true;
    layout.row_count = layout.lane_count = count;
    for (int index = 0; index < count; ++index) {
        const auto& primary = input[0][index];
        const auto& peer = input[1][index];
        if (primary.position != peer.position || primary.token_id != peer.token_id) {
            return std::nullopt;
        }
        for (const auto* lane : {&primary, &peer}) {
            if (lane->request == nullptr || lane->mode != TextGdnMode::decode ||
                lane->feature != nullptr || lane->mrope != nullptr ||
                lane->singleton_scores.scores != nullptr || lane->token_id < 0 ||
                lane->token_id >= vocab_size || lane->position < 0 ||
                lane->position >= text_attention_max_tokens)
                return std::nullopt;
            for (const auto& cache : lane->request->caches) {
                if (cache.kv_heads != gqa_rank_geometry.kv_heads ||
                    !cache_covers(cache, lane->request->pages, lane->position + 1)) {
                    return std::nullopt;
                }
            }
        }
        if (target_split_position(primary.position, text_attention_max_tokens)) {
            layout.decode_split |= std::uint32_t{1} << static_cast<unsigned>(index);
        }
    }
    return layout;
}

bool same_map(const Mxfp8KvCache& a, const Mxfp8KvCache& b) {
    return a.key_payload == b.key_payload && a.value_payload == b.value_payload &&
           a.key_scales == b.key_scales && a.value_scales == b.value_scales &&
           a.page_count == b.page_count && a.kv_heads == b.kv_heads;
}

struct Rank {
    Rank(int id, cudaStream_t lane_stream)
        : device(id), stream(lane_stream), memory(id), table(memory.allocate<DeviceBinding>(1)),
          attention(memory.allocate<Bf16>(value_elements)),
          z(memory.allocate<Bf16>(value_elements)), observed(memory.allocate<Bf16>(value_elements)),
          packed(memory.allocate<Bf16>(2 * (qkv_elements + gate_elements))) {
        try {
            constexpr auto alignment = alignof(DeviceBinding);
            auto bytes = sizeof(DeviceBinding) + alignment - 1;
            check(cudaMallocHost(&host, bytes), "Allocate pinned TP verifier binding");
            auto* aligned = host;
            staging =
                new (std::align(alignment, sizeof(DeviceBinding), aligned, bytes)) DeviceBinding{};
            check(cudaEventCreateWithFlags(&edge, cudaEventDisableTiming),
                  "Create TP verifier edge");
        } catch (...) {
            if (host != nullptr)
                cleanup(cudaFreeHost(host), "Free failed TP verifier binding");
            throw;
        }
    }

    ~Rank() {
        if (edge != nullptr)
            cleanup(cudaEventDestroy(edge), "Destroy TP verifier edge");
        if (host != nullptr)
            cleanup(cudaFreeHost(host), "Free TP verifier binding");
    }

    Rank(const Rank&) = delete;
    Rank& operator=(const Rank&) = delete;
    Rank(Rank&&) = delete;
    Rank& operator=(Rank&&) = delete;

    void allocate_rows(int rows) {
        for (; allocated < rows; ++allocated) {
            work[allocated] = Workspace(memory, attention);
        }
    }

    void arrange(const Layout& layout) {
        auto* convolved = packed + qkv_elements;
        auto* gate_b = convolved + qkv_elements;
        auto* gate_a = gate_b + gate_elements;
        std::size_t first = 0;
        for (int lane = 0; lane < layout.lane_count; ++lane) {
            const auto& entry = layout.lanes[lane];
            const auto values = first * GdnLocal::value_width;
            const auto qkv = first * GdnLocal::qkv_width;
            const auto gates = first * GdnLocal::value_heads;
            gdn[lane] = {
                {},
                {packed + qkv, convolved + qkv, gate_b + gates, gate_a + gates},
                z + values,
                observed + values,
                attention + values,
                entry.depth,
            };
            for (int depth = 0; depth < entry.depth; ++depth) {
                const auto row = first + static_cast<std::size_t>(depth);
                auto& scratch = work[entry.rows[depth]];
                scratch.attention = attention + (row * GdnLocal::value_width);
                scratch.gdn_z = z + (row * GdnLocal::value_width);
                scratch.gdn_qkv = packed + (row * GdnLocal::qkv_width);
                scratch.gdn_b = gate_b + (row * GdnLocal::value_heads);
                scratch.gdn_a = gate_a + (row * GdnLocal::value_heads);
            }
            first += entry.depth;
        }
    }

    bool prepare(const VerificationRow* packet, const Layout& layout) {
        arrange(layout);
        for (int index = 0; index < layout.row_count; ++index) {
            const auto& row = packet[index];
            auto& fixed = work[index];
            staging->request[index] = {row.work->hidden, row.work->residual};
            staging->resident[index] = {fixed.hidden, fixed.residual};
            staging->rotary[index] = row.rotary_position < 0 ? row.position : row.rotary_position;
            for (int layer = 0; layer < full_attention_layers; ++layer) {
                staging->gqa[layer].reads[index] = {
                    row.request->caches[layer],
                    row.request->pages,
                    row.position,
                    paged_gqa_buffers(fixed),
                };
            }
        }
        for (int layer = 0; layer < gdn_layers; ++layer) {
            for (int lane = 0; lane < layout.lane_count; ++lane) {
                auto job = gdn[lane];
                job.state = packet[layout.lanes[lane].rows[0]].request->gdn_replay[layer];
                staging->gdn[layer][lane] = job;
            }
        }
        if (layout.long_count != 0) {
            for (auto& layer : staging->gqa) {
                if (!prepare_long(layer, layout))
                    return false;
            }
        }
        return true;
    }

    bool prepare(const ForwardLane* packet, const Layout& layout) {
        constexpr std::size_t stride = GdnLocal::qkv_width + GdnLocal::value_width;
        static_assert(decoder_batch_max_lanes * stride <= qkv_elements);
        auto* decode_scratch = packed + qkv_elements;
        auto* gate_b = decode_scratch + qkv_elements;
        auto* gate_a = gate_b + gate_elements;
        for (int index = 0; index < layout.row_count; ++index) {
            const auto& lane = packet[index];
            const auto& request = *lane.request;
            auto& fixed = work[index];
            const auto row = static_cast<std::size_t>(index);
            fixed.attention = attention + (row * GdnLocal::value_width);
            fixed.gdn_z = z + (row * GdnLocal::value_width);
            fixed.gdn_qkv = packed + (row * GdnLocal::qkv_width);
            fixed.gdn_b = gate_b + (row * GdnLocal::value_heads);
            fixed.gdn_a = gate_a + (row * GdnLocal::value_heads);
            fixed.gdn_scratch = decode_scratch + (row * stride);
            staging->tokens[index] = lane.token_id;
            staging->request[index] = {request.work.hidden, request.work.residual};
            staging->resident[index] = {fixed.hidden, fixed.residual};
            const int rotary = lane.rotary_position < 0 ? lane.position : lane.rotary_position;
            const auto scores = fixed.gqa_scores(request.pages.page_count * mxfp8_page_size);
            const TextModelStateRow final_state{};
            staging->decode[index] = {
                {},
                lane.position,
                rotary,
                scores,
                request.pages,
                request.caches,
                request.gdn_replay,
                final_state,
            };
        }
        return true;
    }

    static bool prepare_long(GqaBinding& layer, const Layout& layout) {
        int long_lane = 0;
        for (int lane = 0; lane < layout.lane_count; ++lane) {
            const auto& entry = layout.lanes[lane];
            if (entry.long_start == entry.depth)
                continue;
            auto& binding = layer.long_reads[long_lane++];
            auto& request = binding.request;
            const auto& seed = layer.reads[entry.rows[entry.long_start]];
            const bool encode = !same_map(request.cache, seed.cache);
            request.pages = seed.pages;
            request.first = seed.position;
            request.count = entry.depth - entry.long_start;
            for (int depth = 0; depth < request.count; ++depth) {
                request.rows[depth] = layer.reads[entry.rows[entry.long_start + depth]].buffers;
            }
            if (encode) {
                request.cache = seed.cache;
                const auto status = gqa_decode::prepare_long_decode_binding(binding);
                if (!status || *status != cudaSuccess)
                    request.cache = {};
                if (!status)
                    return false;
                check(*status, "Encode TP verifier KV tensor maps");
            }
        }
        return true;
    }

    static constexpr auto value_elements = std::size_t{max_rows} * GdnLocal::value_width;
    static constexpr auto qkv_elements = std::size_t{max_rows} * GdnLocal::qkv_width;
    static constexpr auto gate_elements = std::size_t{max_rows} * GdnLocal::value_heads;

    int device;
    cudaStream_t stream;
    Allocations memory;
    DeviceBinding* table;
    DeviceBinding* staging = nullptr;
    void* host = nullptr;
    cudaEvent_t edge = nullptr;
    Bf16* attention;
    Bf16* z;
    Bf16* observed;
    Bf16* packed;
    int allocated = 0;
    Workspace work[max_rows];
    gdn_replay_ssm::Lane gdn[max_lanes];
};
}

class Graph {
public:
    Graph(const std::array<int, 2>& devices, const std::array<cudaStream_t, 2>& streams) {
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceScope scope(devices[rank]);
            ranks[rank] = std::make_unique<Rank>(devices[rank], streams[rank]);
        }
    }

    ~Graph() {
        if (!shutdown()) {
            for (auto& rank : ranks)
                std::ignore = rank.release();
        }
    }

    bool matches(const std::array<int, 2>& devices,
                 const std::array<cudaStream_t, 2>& streams) const {
        for (int rank = 0; rank < 2; ++rank) {
            if (ranks[rank]->device != devices[rank] || ranks[rank]->stream != streams[rank]) {
                return false;
            }
        }
        return true;
    }

    bool shutdown() noexcept {
        if (closed)
            return true;
        try {
            if (in_flight) {
                for (const auto& rank : ranks) {
                    const DeviceScope scope(rank->device);
                    check(cudaStreamSynchronize(rank->stream), "Fence TP verifier replay");
                }
                in_flight = false;
            }
            const DeviceScope scope(ranks[0]->device);
            for (auto& entry : entries)
                destroy(entry);
            closed = true;
            return true;
        } catch (const CudaError& error) {
            cleanup(error.status, error.what());
            return false;
        }
    }

    template <typename Packet, typename Forward>
    int replay(const Packet& input, const Layout& layout, const Forward& forward,
               bool& replicated) {
        if (in_flight) {
            const DeviceScope scope(ranks[0]->device);
            const auto completed = cudaEventQuery(ranks[0]->edge);
            if (completed == cudaErrorNotReady)
                return -1;
            check(completed, "Check previous TP graph replay");
            in_flight = false;
        }
        auto& entry = entry_for(layout);
        if (entry.rejected)
            return -1;
        try {
            for (int rank = 0; rank < 2; ++rank) {
                auto& state = *ranks[rank];
                const DeviceScope scope(state.device);
                state.allocate_rows(layout.row_count);
                if (!state.prepare(input[rank], layout))
                    return -1;
            }
            if (!entry.executable && !capture(entry, input, layout, forward)) {
                reject(entry, "incomplete capture");
                return -1;
            }
        } catch (const CudaError& error) {
            if (!capture_rejected(error.status))
                throw;
            const auto pending = cudaGetLastError();
            if (pending != error.status && !capture_rejected(pending)) {
                check(pending, "Check CUDA status before eager TP verification");
            }
            reject(entry, error.what());
            return -1;
        }
        submit(entry.executable, layout);
        replicated = entry.replicated;
        return entry.two_shot;
    }

    void complete() noexcept {
        in_flight = false;
    }
    bool pending() const noexcept {
        return in_flight;
    }

  private:
    struct Entry {
        Layout layout{};
        cudaGraphExec_t executable = nullptr;
        std::uint64_t used = 0;
        bool rejected = false;
        int two_shot = 0;
        bool replicated = false;
    };

    static void reject(Entry& entry, const char* reason) {
        entry.rejected = true;
        std::array<char, 4> lanes{};
        std::array<char, 4> rows{};
        std::ignore =
            std::to_chars(lanes.data(), lanes.data() + lanes.size() - 1, entry.layout.lane_count);
        std::ignore =
            std::to_chars(rows.data(), rows.data() + rows.size() - 1, entry.layout.row_count);
        static_cast<void>(std::fputs("sglang decoder: TP verifier graph lanes=", stderr));
        static_cast<void>(std::fputs(lanes.data(), stderr));
        static_cast<void>(std::fputs(" rows=", stderr));
        static_cast<void>(std::fputs(rows.data(), stderr));
        static_cast<void>(std::fputs(" declined: ", stderr));
        static_cast<void>(std::fputs(reason, stderr));
        static_cast<void>(std::fputc('\n', stderr));
    }

    static void destroy(Entry& entry) {
        if (entry.executable != nullptr) {
            check(cudaGraphExecDestroy(entry.executable), "Destroy TP verifier executable");
            entry.executable = nullptr;
        }
    }

    Entry& entry_for(const Layout& layout) {
        auto* replacement = entries.begin();
        for (auto* it = entries.begin(); it != entries.end(); ++it) {
            if (it->used != 0 && it->layout == layout) {
                it->used = ++clock;
                return *it;
            }
            if (it->used < replacement->used)
                replacement = it;
        }
        const DeviceScope scope(ranks[0]->device);
        destroy(*replacement);
        *replacement = {layout, nullptr, ++clock};
        return *replacement;
    }

    static HostBinding make_binding(const Rank& rank, const Layout& layout) {
        return {
            rank.stream,         rank.table,          layout.row_count,   layout.lane_count,
            layout.max_depth,    layout.short_rows,   layout.short_count, layout.split_rows,
            layout.splits,       layout.split_groups, layout.long_count,  layout.long_shape,
            layout.decode_split,
        };
    }

    template <typename Forward>
    bool enqueue_capture(const Input& input, const Layout& layout, const Forward& forward) {
        std::array<std::array<VerificationRow, max_rows>, 2> resident;
        for (int rank = 0; rank < 2; ++rank) {
            for (int index = 0; index < layout.row_count; ++index) {
                resident[rank][index] = input[rank][index];
                resident[rank][index].work = &ranks[rank]->work[index];
            }
            copy_rows(*ranks[rank], layout.row_count, false);
        }
        const bool replicated = forward(Input{resident[0].data(), resident[1].data()});
        for (const auto& rank : ranks)
            copy_rows(*rank, layout.row_count, true);
        return replicated;
    }

    template <typename Forward>
    bool enqueue_capture([[maybe_unused]] const DecodeInput& input, const Layout& layout,
                         const Forward& forward) {
        std::array<std::array<BatchGraphLane, decoder_batch_max_lanes>, 2> resident;
        for (int rank = 0; rank < 2; ++rank) {
            for (int index = 0; index < layout.row_count; ++index) {
                resident[rank][index] = {
                    &ranks[rank]->work[index],
                    &ranks[rank]->table->decode[index],
                };
            }
        }
        const bool replicated = forward(resident[0].data(), resident[1].data());
        for (const auto& rank : ranks)
            copy_rows(*rank, layout.row_count, true);
        return replicated;
    }

    static void copy_rows(const Rank& rank, int rows, bool output) {
        const DeviceScope scope(rank.device);
        check(exchange_rows(rank.table, rows, output, rank.stream), "Capture TP verifier row copy");
    }

    template <typename Packet, typename Forward>
    bool capture(Entry& entry, const Packet& input, const Layout& layout, const Forward& forward) {
        const auto& root = *ranks[0];
        const auto& peer = *ranks[1];
        const DeviceScope root_scope(root.device);
        check(cudaGetLastError(), "Check CUDA status before TP verifier capture");
        check(cudaStreamBeginCapture(root.stream, cudaStreamCaptureModeThreadLocal),
              "Begin TP verifier capture");
        HostBinding bindings[]{make_binding(root, layout), make_binding(peer, layout)};
        bool joined = false;
        std::exception_ptr error;
        try {
            check(cudaEventRecord(root.edge, root.stream), "Fork TP verifier capture");
            {
                const DeviceScope scope(peer.device);
                check(cudaStreamWaitEvent(peer.stream, root.edge), "Join TP verifier capture");
                joined = true;
            }
            active_bindings() = {&bindings[0], &bindings[1]};
            entry.replicated = enqueue_capture(input, layout, forward);
        } catch (...) {
            error = std::current_exception();
        }
        active_bindings() = {};
        if (joined) {
            try {
                {
                    const DeviceScope scope(peer.device);
                    check(cudaEventRecord(peer.edge, peer.stream),
                          "Record TP verifier capture tail");
                }
                check(cudaStreamWaitEvent(root.stream, peer.edge), "Merge TP verifier capture");
            } catch (...) {
                if (!error)
                    error = std::current_exception();
            }
        }
        cudaGraph_t source = nullptr;
        const auto ended = cudaStreamEndCapture(root.stream, &source);
        if (!streams_available({root.device, peer.device}, {root.stream, peer.stream})) {
            if (source)
                cleanup(cudaGraphDestroy(source), "Destroy unfinished TP verifier graph");
            throw std::runtime_error("TP verifier stream remains in capture");
        }
        const bool complete =
            std::all_of(std::begin(bindings), std::end(bindings), [&](const HostBinding& binding) {
                return binding.gqa_calls == full_attention_layers &&
                       binding.gdn_calls == gdn_layers &&
                       binding.collectives == collectives_per_forward;
            });
        if (ended != cudaSuccess || error || !complete) {
            if (source)
                check(cudaGraphDestroy(source), "Destroy rejected TP verifier graph");
            if (CudaError::terminal(ended))
                check(ended, "End TP verifier capture");
            if (error) std::rethrow_exception(error);
            check(ended, "End TP verifier capture");
            return false;
        }
        const auto instantiated = cudaGraphInstantiateWithFlags(&entry.executable, source, 0);
        check(cudaGraphDestroy(source), "Destroy TP verifier source graph");
        if (instantiated == cudaErrorInvalidValue) {
            const auto pending = cudaGetLastError();
            if (pending != instantiated)
                check(pending, "Check rejected TP verifier executable");
            return false;
        }
        entry.two_shot = bindings[0].two_shot;
        check(instantiated, "Instantiate TP verifier graph");
        return true;
    }

    void submit(cudaGraphExec_t executable, const Layout& layout) {
        const auto& root = *ranks[0];
        const auto& peer = *ranks[1];
        const std::size_t bytes =
            layout.ordinary
                ? offsetof(DeviceBinding, decode) + (layout.row_count * sizeof(BatchDecodeBinding))
                : sizeof(DeviceBinding);
        in_flight = true;
        for (auto& rank : ranks) {
            const DeviceScope scope(rank->device);
            check(cudaMemcpyAsync(rank->table, rank->staging, bytes, cudaMemcpyHostToDevice,
                                  rank->stream),
                  "Upload TP verifier binding");
        }
        {
            const DeviceScope scope(peer.device);
            check(cudaEventRecord(peer.edge, peer.stream), "Publish peer TP verifier inputs");
        }
        {
            const DeviceScope scope(root.device);
            check(cudaStreamWaitEvent(root.stream, peer.edge), "Wait for peer TP verifier inputs");
            check(cudaGraphLaunch(executable, root.stream), "Replay TP verifier graph");
            check(cudaEventRecord(root.edge, root.stream), "Publish TP verifier completion");
        }
        {
            const DeviceScope scope(peer.device);
            check(cudaStreamWaitEvent(peer.stream, root.edge), "Order peer TP verifier completion");
        }
    }

    std::array<std::unique_ptr<Rank>, 2> ranks;
    std::array<Entry, graph_slots> entries{};
    std::uint64_t clock = 0;
    bool in_flight = false;
    bool closed = false;
};

namespace {

template <typename Packet, typename Forward>
bool replay(std::shared_ptr<Graph>& owner, bool& disabled, const Packet& input,
            const Layout& layout, const std::array<int, 2>& devices,
            const std::array<cudaStream_t, 2>& streams, TensorParallel& group,
            const Forward& forward, bool& replicated) {
    const auto count = static_cast<std::size_t>(layout.row_count) * hidden_size;
    if (disabled || !group.fused_graph_eligible() ||
        (!group.supports_mnnvl_oneshot(count) && !group.supports_mnnvl_twoshot(count)) ||
        !streams_available(devices, streams))
        return false;
    if (!owner) {
        check(cudaGetLastError(), "Check CUDA status before TP verifier allocation");
        try {
            owner = std::make_shared<Graph>(devices, streams);
        } catch (const CudaError& error) {
            if (!capture_rejected(error.status)) throw;
            const auto pending = cudaGetLastError();
            if (pending != error.status && !capture_rejected(pending)) {
                check(pending, "Check CUDA status after TP verifier allocation");
            }
            disabled = true;
            cleanup(error.status, "Disable TP verifier graph scratch allocation");
            return false;
        }
    }
    if (!owner->matches(devices, streams)) return false;
    try {
        const int two_shot = owner->replay(input, layout, forward, replicated);
        if (two_shot < 0)
            return false;
        group.record_fused_graph_replay(layout.row_count, collectives_per_forward, two_shot);
    } catch (...) {
        if (owner->pending())
            group.fail_fused_graph_replay();
        throw;
    }
    return true;
}

}

Submission::Submission(TensorParallel& transport, Graph* graph) noexcept
    : group(transport), owner(graph), uncaught(std::uncaught_exceptions()) {}

Submission::~Submission() {
    if (owner == nullptr)
        return;
    if (std::uncaught_exceptions() != uncaught) group.fail_fused_graph_replay();
    else
        owner->complete();
}

bool close(std::shared_ptr<Graph>& owner, TensorParallel& group) noexcept {
    if (owner && !owner->shutdown()) {
        group.fail_fused_graph_replay();
        return false;
    }
    owner.reset();
    return true;
}

bool owns_collective(const cudaStream_t streams[], int rows) {
    const auto& active = active_bindings();
    return active[0] != nullptr && active[1] != nullptr && active[0]->stream == streams[0] &&
           active[1]->stream == streams[1] && active[0]->rows == rows && active[1]->rows == rows;
}

void record_collective(cudaStream_t stream, bool two_shot) {
    const auto& active = active_bindings();
    if (active[0] != nullptr && active[1] != nullptr && active[0]->stream == stream) {
        for (auto* rank : active) {
            ++rank->collectives;
            rank->two_shot += static_cast<int>(two_shot);
        }
    }
}

HostBinding* binding(cudaStream_t stream) {
    for (auto* rank : active_bindings()) {
        if (rank != nullptr && rank->stream == stream)
            return rank;
    }
    return nullptr;
}
}

namespace sglang {
using namespace decoder;

tp_verify_graph::Graph* Decoder::Model::forward_verification_tp(
    const Model& peer, const VerificationRow* rows, const VerificationRow* peer_rows,
    int count, int total, cudaStream_t stream, cudaStream_t peer_stream, TensorParallel& group,
    const TpPartials& primary_partials, const TpPartials& peer_partials,
    std::uint8_t* primary_packed, std::uint8_t* peer_packed, bool greedy, bool capture,
    bool& eager_replicated) const {
    const tp_verify_graph::Input input{rows, peer_rows};
    const auto forward = [&](const tp_verify_graph::Input& live) {
        return forward_layers_tp(peer, live[0], live[1], total, stream, peer_stream, group,
                                 primary_partials, peer_partials, primary_packed, peer_packed,
                                 count == 1 && greedy);
    };
    if (capture && tp != nullptr && backend.vector_decode_attention &&
        peer.backend.vector_decode_attention &&
        backend.target_split_end == text_attention_max_tokens &&
        peer.backend.target_split_end == text_attention_max_tokens &&
        (tp->verification_capture_stream == nullptr || tp->verification_capture_stream == stream)) {
        const auto layout =
            tp_verify_graph::plan(input, count, total, {stream, peer_stream},
                                  count == 1 && greedy && total >= 3 && total <= fused_verify_rows,
                                  primary_packed != nullptr && peer_packed != nullptr);
        if (layout)
            tp->verification_capture_stream = stream;
        if (layout &&
            tp_verify_graph::replay(tp->verification_capture, tp->verification_capture_disabled,
                                    input, *layout, {device, peer.device}, {stream, peer_stream},
                                    group, forward, eager_replicated)) {
            return tp->verification_capture.get();
        }
    }
    eager_replicated = forward(input);
    return nullptr;
}

bool Decoder::Model::forward_tp(const Model& peer, const ForwardLane* primary_lanes,
                                const ForwardLane* peer_lanes, int count, cudaStream_t stream,
                                cudaStream_t peer_stream, TensorParallel& group,
                                const TpPartials& primary_partials, const TpPartials& peer_partials,
                                bool capture) const {
    if (primary_lanes == nullptr || peer_lanes == nullptr || count < 1 ||
        count > decoder_batch_max_lanes) {
        throw std::invalid_argument("Invalid decoder batch width");
    }
    const tp_verify_graph::DecodeInput input{primary_lanes, peer_lanes};
    if (capture && tp != nullptr && backend.ordered_attention && peer.backend.ordered_attention &&
        backend.vector_decode_attention && peer.backend.vector_decode_attention &&
        backend.target_split_end == text_attention_max_tokens &&
        peer.backend.target_split_end == text_attention_max_tokens &&
        (tp->verification_capture_stream == nullptr || tp->verification_capture_stream == stream)) {
        const auto layout = tp_verify_graph::plan_decode(input, count);
        const auto forward = [&](const BatchGraphLane* primary, const BatchGraphLane* second) {
            const Bf16* embeddings[] = {embedding, peer.embedding};
            const BatchGraphLane* lanes[] = {primary, second};
            const int devices[] = {device, peer.device};
            const cudaStream_t streams[] = {stream, peer_stream};
            for (int rank = 0; rank < 2; ++rank) {
                const DeviceScope scope(devices[rank]);
                VerificationCopyRows outputs{};
                for (int index = 0; index < count; ++index) {
                    outputs.destination[index] = lanes[rank][index].work->residual;
                }
                const auto* table = tp_verify_graph::binding(streams[rank])->device;
                check(gather_verification_embeddings(embeddings[rank], table->tokens, outputs,
                                                     count, streams[rank]),
                      "Capture TP decode embeddings");
            }
            return forward_layers_tp(peer, primary, second, count, stream, peer_stream, group,
                                     primary_partials, peer_partials);
        };
        bool replicated = false;
        if (layout)
            tp->verification_capture_stream = stream;
        if (layout &&
            tp_verify_graph::replay(tp->verification_capture, tp->verification_capture_disabled,
                                    input, *layout, {device, peer.device}, {stream, peer_stream},
                                    group, forward, replicated))
            return true;
    }
    using forward_detail::embed_forward_lanes;
    embed_forward_lanes(embedding, primary_lanes, count, stream, "Look up token embedding");
    {
        const DeviceScope scope(peer.device);
        embed_forward_lanes(peer.embedding, peer_lanes, count, peer_stream,
                            "Look up peer token embedding");
    }
    forward_layers_tp(peer, primary_lanes, peer_lanes, count, stream, peer_stream, group,
                      primary_partials, peer_partials);
    return false;
}
}
