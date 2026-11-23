#define main tensor_prefix_state_main
#include "tensor_state.cpp"
#undef main

#include "sglang/decoder_constraint.h"

namespace {

bool fail_draft_copy = false;
const MtpDraftRequest* observed_draft = nullptr;
std::optional<int> boundary_input;

const Lane& active_lane(const DecoderBatch& batch, int slot) {
    return *batch.state_->storage->slots.at(slot);
}

DecoderOptions mtp_options(int depth, std::string salt = "shared") {
    auto result = options(std::move(salt));
    result.mtp_steps = depth;
    result.max_new_tokens = 13;
    return result;
}

void retire(DecoderBatch& batch, int slot) {
    batch.cancel(slot);
    require(batch.request_device_allocation_bytes() == 0, "Request allocations remain");
}

int score_prompt(DecoderBatch& batch, int slot) {
    for (int step = 0; step < 128; ++step) {
        DecoderBatchStepStats stats;
        const auto result = batch.advance_chunked(slots({slot}), &stats)[slot];
        require(!result.finish_reason && !result.sampling_failed && !result.constraint_failed,
                "Prompt did not reach its first score");
        if (result.token) {
            require(!result.decode && result.mtp_drafted_tokens == 0, "First score used a draft");
            return *result.token;
        }
        require(stats.prefill_rows[slot] > 0, "Prompt did not advance");
    }
    throw std::runtime_error("First score deadline");
}

void compare_counters(const DecoderBatchStep& expected, const DecoderBatchStep& actual) {
    require(expected.token == actual.token && expected.extra_tokens == actual.extra_tokens &&
            expected.extra_token_count == actual.extra_token_count &&
            expected.finish_reason == actual.finish_reason &&
            expected.mtp_drafted_tokens == actual.mtp_drafted_tokens &&
            expected.mtp_accepted_tokens == actual.mtp_accepted_tokens &&
            expected.mtp_verified_rows == actual.mtp_verified_rows &&
            expected.mtp_replayed_rows == actual.mtp_replayed_rows,
            "Warm continuation differs from the cold oracle");
}

void check_charge(const DecoderBatch& warm, int capacity) {
    const auto stats = warm.prefix_cache_stats();
    constexpr std::size_t target_entry = 370233344;
    constexpr std::size_t draft_entry = 13516800;
    require(stats.draft_device_bytes == draft_entry * capacity &&
            stats.device_bytes == (target_entry + draft_entry) * capacity,
            "Combined cache does not charge both preallocated fragments");
}

void check_pair(DecoderBatch& cold, DecoderBatch& warm, const std::vector<int>& prompt,
                int depth, int restored, std::string salt = "shared") {
    const auto consume = mtp_options(depth, std::move(salt));
    // Enable the same exact prefill route as the warm batch, with no reusable entry.
    cold.reset_prefix_cache();
    const auto cold_before = cold.prefix_cache_stats();
    require(cold.admit(0, prompt, consume) == 0, "Cold oracle unexpectedly restored");
    const auto before = warm.prefix_cache_stats();
    require(warm.admit(1, prompt, consume) == restored, "Warm target restore length");
    const auto& warm_lane = active_lane(warm, 1);
    if (depth) {
        require(warm_lane.draft && warm_lane.draft->valid_prefix() == restored - 1,
                "Restore retained the prompt-dependent last draft row");
    } else {
        require(!warm_lane.draft, "K0 restored a draft request");
    }
    const int expected = score_prompt(cold, 0);
    const auto cold_after = cold.prefix_cache_stats();
    require(cold_after.hits == cold_before.hits &&
            cold_after.cached_tokens == cold_before.cached_tokens &&
            cold_after.draft_cached_rows == cold_before.draft_cached_rows &&
            cold_after.prefill_tokens - cold_before.prefill_tokens == prompt.size(),
            "Cold oracle did not compute the entire prompt without reuse");
    require(score_prompt(warm, 1) == expected, "Warm first selected token differs");
    compare_state(request(cold, 0), request(warm, 1), prompt.size(), true);
    const auto after = warm.prefix_cache_stats();
    const auto target_cached = static_cast<std::uint64_t>(restored);
    const auto target_computed = prompt.size() - target_cached;
    require(after.cached_tokens - before.cached_tokens == target_cached &&
            after.prefill_tokens - before.prefill_tokens == target_computed &&
            after.draft_cached_rows - before.draft_cached_rows == (depth ? target_cached - 1 : 0) &&
            after.draft_prefill_rows - before.draft_prefill_rows == (depth ? target_computed : 0),
            "Prompt and draft counters disagree");
    if (depth) {
        require(warm_lane.draft->valid_prefix() == int(prompt.size()) - 1,
                "First target score appended an extra draft row");
    }
    bool completed = false;
    for (int step = 0; step < 13 && !completed; ++step) {
        const auto a = cold.advance_chunked(slots({0}))[0];
        const auto b = warm.advance_chunked(slots({1}))[1];
        compare_counters(a, b);
        if (step == 0 && depth) {
            require(b.mtp_drafted_tokens == depth && b.mtp_verified_rows == depth + 1,
                    "First warm continuation did not verify a real draft");
        }
        if (b.finish_reason) {
            require(b.finish_reason == DecoderFinishReason::length, "Unexpected retirement");
            completed = true;
        }
    }
    require(completed && cold.request_device_allocation_bytes() == 0 &&
            warm.request_device_allocation_bytes() == 0, "Continuation retained allocations");
}

void enrichment(Decoder& decoder, const std::vector<int>& tokens) {
    DecoderBatch cold(decoder, {false, 2, 1, page});
    DecoderBatch warm(decoder, {false, 2, 1, page});
    check_charge(warm, 1);
    const auto shared = head(tokens, page);
    require(warm.admit(0, shared, mtp_options(0)) == 0, "Unexpected prime hit");
    first_token(warm, 0);
    retire(warm, 0);
    const auto original = warm.prefix_cache_stats();
    require(original.entries == 1 && original.pages == 1 && original.stores == 1,
            "Target-only prime did not publish exactly once");
    require(warm.admit(0, shared, mtp_options(4)) == 0, "K4 reused a target-only entry");
    first_token(warm, 0);
    retire(warm, 0);
    const auto enriched = warm.prefix_cache_stats();
    require(enriched.entries == 1 && enriched.pages == 1 && enriched.stores == 2 &&
            enriched.evictions == 0 && enriched.draft_misses == 1 && enriched.hits == 0,
            "Same-key enrichment changed occupancy or counted a false hit");
    check_charge(warm, 1);
    check_pair(cold, warm, shared, 1, page);
    check_pair(cold, warm, shared, 0, page);
    const auto suffix = head(tokens, page + 1);
    check_pair(cold, warm, suffix, 4, page);
    check_pair(cold, warm, suffix, 0, page);
    warm.reset_prefix_cache();
    require(warm.prefix_cache_stats().entries == 0 && warm.prefix_cache_stats().pages == 0,
            "Reset retained a target or draft entry");
    check_charge(warm, 1);
}

void changed_tiles(Decoder& decoder, const std::vector<int>& tokens) {
    auto prompt = head(tokens, 541);
    const int boundary = 3 * page;
    auto producer = head(prompt, boundary + 1);
    producer.back() = (producer.back() + 1) % vocab_size;
    DecoderBatch cold(decoder, {false, 2, 1, 1024});
    DecoderBatch warm(decoder, {false, 2, 1, 1024});
    require(warm.admit(0, producer, mtp_options(3)) == 0, "Unexpected larger prime hit");
    score_prompt(warm, 0);
    retire(warm, 0);
    check_pair(cold, warm, head(prompt, boundary), 0, boundary);
    check_pair(cold, warm, prompt, 0, boundary);
    // A K0 publication of the longer prefix leaves the capacity-one entry target-only.
    require(warm.admit(0, producer, mtp_options(3)) == 0, "Unexpected stale draft hit");
    score_prompt(warm, 0);
    retire(warm, 0);
    check_pair(cold, warm, prompt, 3, boundary);
    std::cout << "K3-to-K0 exact full/partial and changed-tile K3 continuation passed\n";
}

void check_observed_input(DecoderBatch& batch, int slot, int first, bool generated) {
    observed_draft = active_lane(batch, slot).draft.get();
    boundary_input.reset();
    if (generated) {
        const auto step = batch.advance_chunked(slots({slot}))[slot];
        require(step.mtp_drafted_tokens > 0 && step.mtp_verified_rows > 1,
                "Generated boundary did not run verification");
    } else {
        score_prompt(batch, slot);
    }
    observed_draft = nullptr;
    require(boundary_input == first, "Draft used the previous request's shifted boundary token");
}

void changed_first_token(Decoder& decoder, const Tokenizer& tokenizer,
                         const std::vector<int>& tokens) {
    DecoderBatch warm(decoder, {false, 2, 1, page});
    const auto prompt = head(tokens, page);
    const auto suffix = head(tokens, page + 1);
    auto original = suffix;
    original.back() = (original.back() + 1) % vocab_size;
    auto producer = mtp_options(4);
    producer.max_new_tokens = 32;
    producer.constraint = TokenConstraint::compile(tokenizer,
        R"(root ::= "The colors are red, orange, yellow, green, blue, and violet.")");
    require(warm.admit(0, original, producer) == 0, "Unexpected constrained prime hit");
    const int original_y0 = score_prompt(warm, 0);
    retire(warm, 0);
    require(warm.admit(0, suffix, mtp_options(4)) == page, "Changed suffix did not hit");
    check_observed_input(warm, 0, suffix[page], false);
    retire(warm, 0);

    auto consume = mtp_options(2);
    consume.max_new_tokens = 32;
    consume.constraint = TokenConstraint::compile(tokenizer,
        R"(root ::= "Please write yes, no, and maybe on the next line.")");
    require(warm.admit(1, prompt, consume) == page, "Changed grammar did not hit");
    const int selected_y0 = score_prompt(warm, 1);
    require(selected_y0 != original_y0, "Fixture did not change first selected token");
    check_observed_input(warm, 1, selected_y0, true);
    retire(warm, 1);
    std::cout << "Deferred row used x[P] for partial hit and changed y0 for full hit\n";
}

void check_fault_trigger(std::string_view operation) {
    require(!fail_draft_copy && operation.find("MTP prefix tail scales") != operation.npos,
            "Failure did not come from the draft-side transfer");
}

void interrupted_store(DecoderBatch& warm, const std::vector<int>& prompt) {
    require(warm.admit(0, prompt, mtp_options(0)) == 0, "Unexpected fault prime hit");
    first_token(warm, 0);
    retire(warm, 0);
    require(warm.admit(0, prompt, mtp_options(4)) == 0, "Unexpected fault enrichment hit");
    fail_draft_copy = true;
    try {
        first_token(warm, 0);
        throw std::runtime_error("Draft-side store fault did not throw");
    } catch (const std::runtime_error& error) {
        check_fault_trigger(error.what());
    }
    require(warm.available() && warm.request_device_allocation_bytes() == 0,
            "Failed store did not drain and retire the request");
    const auto failed = warm.prefix_cache_stats();
    require(failed.entries == 0 && failed.pages == 0 && failed.stores == 1 && failed.hits == 0,
            "Draft-side store failure exposed a half snapshot");
    require(warm.admit(0, prompt, mtp_options(4)) == 0, "Failed store counted as hit");
    first_token(warm, 0);
    retire(warm, 0);
}

void interrupted_restore(DecoderBatch& cold, DecoderBatch& warm, const std::vector<int>& prompt) {
    const auto before = warm.prefix_cache_stats();
    fail_draft_copy = true;
    try {
        warm.admit(1, prompt, mtp_options(2));
        throw std::runtime_error("Draft-side restore fault did not throw");
    } catch (const std::runtime_error& error) {
        check_fault_trigger(error.what());
    }
    require(warm.available() && warm.request_device_allocation_bytes() == 0,
            "Failed restore retained a lane or made the worker unavailable");
    const auto failed = warm.prefix_cache_stats();
    require(failed.entries == before.entries && failed.hits == before.hits &&
            failed.cached_tokens == before.cached_tokens &&
            failed.draft_cached_rows == before.draft_cached_rows,
            "Failed restore altered source or counted a hit");
    check_pair(cold, warm, prompt, 2, page);
}

}  // namespace

const Bf16* actual_enqueue(MtpDraftRequest*, int, int, const Bf16*, bool, cudaStream_t,
                           const ConstraintWorkspace*, const ConstraintCandidates*)
    asm("__real__ZN6sglang7decoder15MtpDraftRequest7enqueueEiiPK13__nv_bfloat16b"
        "P11CUstream_stPKNS0_19ConstraintWorkspaceEPKNS0_20ConstraintCandidatesE");
const Bf16* observed_enqueue(MtpDraftRequest*, int, int, const Bf16*, bool, cudaStream_t,
                             const ConstraintWorkspace*, const ConstraintCandidates*)
    asm("__wrap__ZN6sglang7decoder15MtpDraftRequest7enqueueEiiPK13__nv_bfloat16b"
        "P11CUstream_stPKNS0_19ConstraintWorkspaceEPKNS0_20ConstraintCandidatesE");

const Bf16* observed_enqueue(MtpDraftRequest* draft, int position, int shifted_token,
                             const Bf16* hidden, bool score, cudaStream_t stream,
                             const ConstraintWorkspace* scratch,
                             const ConstraintCandidates* allowed) {
    if (draft == observed_draft && !boundary_input) {
        require(position == page - 1, "First draft rebuild did not start at P - 1");
        boundary_input = shifted_token;
    }
    return actual_enqueue(draft, position, shifted_token, hidden, score, stream, scratch, allowed);
}

extern "C" cudaError_t __real_cudaMemcpy2DAsync(void*, std::size_t, const void*, std::size_t,
                                               std::size_t, std::size_t, cudaMemcpyKind,
                                               cudaStream_t);

extern "C" cudaError_t __wrap_cudaMemcpy2DAsync(void* to, std::size_t to_pitch, const void* from,
                                               std::size_t from_pitch, std::size_t width,
                                               std::size_t height, cudaMemcpyKind kind,
                                               cudaStream_t stream) {
    constexpr std::size_t head_scales = page * mxfp8_groups;
    if (fail_draft_copy && kind == cudaMemcpyDeviceToDevice && height == mxfp8_kv_heads &&
        from_pitch == head_scales && to_pitch == head_scales &&
        width == head_scales - mxfp8_groups) {
        fail_draft_copy = false;
        return cudaErrorInvalidValue;
    }
    return __real_cudaMemcpy2DAsync(to, to_pitch, from, from_pitch, width, height, kind, stream);
}

int main(int argc, char** argv) {
    try {
        require(argc == 4, "Pass the checkpoint, 257 short IDs and 8192 long IDs");
        const auto tokens = ids(argv[2], 257);
        const auto long_tokens = ids(argv[3], 8192);
        const Tokenizer tokenizer(std::string(argv[1]) + "/tokenizer.json");
        Decoder decoder(argv[1], 0, true);
        enrichment(decoder, tokens);
        changed_tiles(decoder, long_tokens);
        changed_first_token(decoder, tokenizer, tokens);
        DecoderBatch cold(decoder, {false, 2, 1, page});
        DecoderBatch warm(decoder, {false, 2, 4, page});
        check_charge(warm, 4);
        const auto prompt = head(tokens, page);
        interrupted_store(warm, prompt);
        interrupted_restore(cold, warm, prompt);
        std::cout << "Prefix MTP enrichment, accounting, continuation and fault retry passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
