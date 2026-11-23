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

void prime_prompt(DecoderBatch& batch, const std::vector<int>& prompt, int depth) {
    auto prime = mtp_options(depth);
    prime.max_new_tokens = 1;
    require(batch.admit(0, prompt, prime) == 0, "Unexpected prime hit");
    finish_prompt(batch, 0);
    require(!batch.request_device_allocation_bytes(), "Prompt prime retained memory");
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

void check_charge(const DecoderBatch& batch, int target_pages, int draft_pages) {
    const auto stats = batch.prefix_cache_stats();
    constexpr auto kv_bytes = 2 * (mxfp8_payload_bytes_per_page + mxfp8_scale_bytes_per_page);
    constexpr auto state = gdn_layers * (gdn_history_elements * sizeof(Bf16) +
        gdn_state_elements * sizeof(float)) + 2 * hidden_size * sizeof(Bf16);
    const auto target = target_pages ? target_pages * full_attention_layers * kv_bytes + state : 0;
    const auto draft = draft_pages * kv_bytes;
    require(stats.entries == int(target_pages > 0) && stats.pages == target_pages &&
            stats.draft_device_bytes == draft && stats.device_bytes == target + draft,
            "Unexpected lazy target or draft snapshot charge");
}

void check_pair(DecoderBatch& cold, DecoderBatch& warm, const std::vector<int>& prompt,
                int depth, int restored, std::string salt = "shared") {
    auto consume = mtp_options(depth, std::move(salt));
    consume.claim_completion = [] { return false; };
    // Match the warm batch's native tile boundaries and clear the reference entries.
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
    const int expected = first_token(cold, 0);
    const auto cold_after = cold.prefix_cache_stats();
    require(cold_after.hits == cold_before.hits &&
            cold_after.cached_tokens == cold_before.cached_tokens &&
            cold_after.draft_cached_rows == cold_before.draft_cached_rows &&
            cold_after.prefill_tokens - cold_before.prefill_tokens == prompt.size(),
            "Cold oracle did not compute the entire prompt without reuse");
    require(first_token(warm, 1) == expected, "Warm first selected token differs");
    compare_state(request(cold, 0), request(warm, 1), prompt.size(), true);
    const auto after = warm.prefix_cache_stats();
    const auto target_cached = static_cast<std::uint64_t>(restored);
    const auto target_computed = prompt.size() - target_cached;
    require(after.hits == before.hits + 1 && after.cached_tokens - before.cached_tokens ==
                target_cached &&
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
            require(b.finish_reason == DecoderFinishReason::cancelled, "Unexpected retirement");
            completed = true;
        }
    }
    require(completed && cold.request_device_allocation_bytes() == 0 &&
            warm.request_device_allocation_bytes() == 0, "Continuation retained allocations");
}

void enrichment(Decoder& decoder, const std::vector<int>& tokens, int length) {
    const int budget = std::max(length, page);
    DecoderBatch cold(decoder, {false, 2, 1, budget});
    DecoderBatch warm(decoder, {false, 2, 1, budget});
    check_charge(warm, 0, 0);
    const int target_pages = (length + page - 1) / page;
    const int draft_pages = (length + page - 2) / page;
    const auto shared = head(tokens, length);
    prime_prompt(warm, shared, 0);
    const auto original = warm.prefix_cache_stats();
    require(original.entries == 1 && original.pages == target_pages && original.stores == 1,
            "Target-only prime did not publish exactly once");
    check_charge(warm, target_pages, 0);
    prime_prompt(warm, shared, 4);
    const auto enriched = warm.prefix_cache_stats();
    require(enriched.entries == 1 && enriched.pages == target_pages && enriched.stores == 2 &&
            enriched.evictions == 0 && enriched.draft_misses == 1 && enriched.hits == 0,
            "Same-key enrichment changed occupancy or counted a false hit");
    check_charge(warm, target_pages, draft_pages);
    check_pair(cold, warm, shared, 1, length);
    check_pair(cold, warm, shared, 0, length);
    if (length == page) {
        const auto suffix = head(tokens, page + 1);
        check_pair(cold, warm, suffix, 4, page);
        check_pair(cold, warm, suffix, 0, page);
    }
    warm.reset_prefix_cache();
    require(warm.prefix_cache_stats().entries == 0 && warm.prefix_cache_stats().pages == 0,
            "Reset retained a target or draft entry");
    check_charge(warm, 0, 0);
}

void changed_tiles(Decoder& decoder, const std::vector<int>& tokens) {
    constexpr int boundary = 1024;
    const auto prompt = head(tokens, boundary + 541);
    const auto producer = head(prompt, boundary);
    DecoderBatch cold(decoder, {false, 2, 1, 1024});
    DecoderBatch warm(decoder, {false, 2, 1, 1024});
    prime_prompt(warm, producer, 3);
    check_pair(cold, warm, producer, 0, boundary);
    check_pair(cold, warm, prompt, 0, boundary);
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
        first_token(batch, slot);
    }
    observed_draft = nullptr;
    require(boundary_input == first, "Draft used the previous request's shifted boundary token");
}

void changed_first_token(Decoder& decoder, const Tokenizer& tokenizer,
                         const std::vector<int>& tokens) {
    DecoderBatch warm(decoder, {false, 2, 2, page});
    const auto prompt = head(tokens, page);
    const auto suffix = head(tokens, page + 1);
    auto producer = mtp_options(4);
    producer.max_new_tokens = 32;
    producer.constraint = TokenConstraint::compile(tokenizer,
        R"(root ::= "The colors are red, orange, yellow, green, blue, and violet.")");
    require(warm.admit(0, prompt, producer) == 0, "Unexpected constrained cold hit");
    const int original_y0 = first_token(warm, 0);
    retire(warm, 0);
    require(warm.prefix_cache_stats().entries == 0, "Cancelled prompt published a checkpoint");
    prime_prompt(warm, prompt, 4);
    require(warm.admit(0, suffix, mtp_options(4)) == page, "Changed suffix did not hit");
    check_observed_input(warm, 0, suffix[page], false);
    retire(warm, 0);

    auto consume = mtp_options(2);
    consume.max_new_tokens = 32;
    consume.constraint = TokenConstraint::compile(tokenizer,
        R"(root ::= "Please write yes, no, and maybe on the next line.")");
    require(warm.admit(1, prompt, consume) == page, "Changed grammar did not hit");
    const int selected_y0 = first_token(warm, 1);
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
    prime_prompt(warm, prompt, 0);
    auto producer = mtp_options(4);
    producer.max_new_tokens = 1;
    require(warm.admit(0, prompt, producer) == 0, "Unexpected fault enrichment hit");
    fail_draft_copy = true;
    try {
        finish_prompt(warm, 0);
        throw std::runtime_error("Draft-side store fault did not throw");
    } catch (const std::runtime_error& error) {
        check_fault_trigger(error.what());
    }
    require(warm.available() && warm.request_device_allocation_bytes() == 0,
            "Failed store did not drain and retire the request");
    const auto failed = warm.prefix_cache_stats();
    require(failed.entries == 0 && failed.pages == 0 && failed.stores == 1 && failed.hits == 0,
            "Draft-side store failure exposed a half snapshot");
    check_charge(warm, 0, 0);
    prime_prompt(warm, prompt, 4);
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
            failed.draft_cached_rows == before.draft_cached_rows &&
            failed.device_bytes == before.device_bytes &&
            failed.draft_device_bytes == before.draft_device_bytes,
            "Failed restore altered source or counted a hit");
    check_pair(cold, warm, prompt, 2, page);
}

void compare_draft(DecoderBatch& batch, const MtpDraftRequest& cold,
                   const MtpDraftRequest& warm, int rows) {
    auto& state = *batch.state_;
    Allocations memory(state.model.device);
    const int pages = (rows + page - 1) / page;
    const auto allocate = [&] {
        return Mxfp8KvCache{memory.allocate<std::uint8_t>(pages * mxfp8_payload_bytes_per_page),
                           memory.allocate<std::uint8_t>(pages * mxfp8_payload_bytes_per_page),
                           memory.allocate<std::uint8_t>(pages * mxfp8_scale_bytes_per_page),
                           memory.allocate<std::uint8_t>(pages * mxfp8_scale_bytes_per_page),
                           pages};
    };
    const auto left = allocate();
    const auto right = allocate();
    const auto stream = state.execution.stream();
    cold.snapshot_prefix(left, rows, stream);
    warm.snapshot_prefix(right, rows, stream);
    check(cudaStreamSynchronize(stream), "Read draft checkpoints");
    const auto payload = rows * (mxfp8_payload_bytes_per_page / page);
    same(left.key_payload, right.key_payload, payload, "Terminal draft key payload");
    same(left.value_payload, right.value_payload, payload, "Terminal draft value payload");
    for (int pg = 0; pg < pages; ++pg) {
        const auto offset = pg * mxfp8_scale_bytes_per_page;
        const auto count = std::min(page, rows - pg * page);
        compare_scales(left.key_scales + offset, right.key_scales + offset, count);
        compare_scales(left.value_scales + offset, right.value_scales + offset, count);
    }
}

void compare_verified_checkpoint(DecoderBatch& oracle, DecoderBatch& candidate,
                                 const std::vector<int>& history, const DecoderOptions& options,
                                 int committed_rows) {
    const int processed = history.size() - 1;
    const auto key = head(history, processed);
    const auto& source = active_lane(oracle, 0);
    auto& final_row = source.verification->eager_workspace_->work.at(committed_rows - 1);
    require(candidate.admit(1, key, options) == processed, "Terminal K4 key differs");
    const auto& restored = active_lane(candidate, 1);
    require(restored.draft->valid_prefix() == processed - 1, "Terminal draft cursor differs");
    compare_state(*source.request, *restored.request, processed, false);
    same(final_row.hidden, restored.request->work.hidden, hidden_size, "Terminal verifier hidden");
    same(final_row.residual, restored.request->work.residual, hidden_size,
         "Terminal verifier residual");
    compare_draft(candidate, *source.draft, *restored.draft, processed - 1);
    require(first_token(candidate, 1) == history.back(), "Terminal K4 first score differs");
    const auto stream = oracle.state_->execution.stream();
    oracle.state_->model.compute_prefill_logits(final_row, stream);
    check(cudaStreamSynchronize(stream), "Score the cold verifier row with the restored head");
    same(final_row.logits, restored.request->work.logits, vocab_size,
         "Terminal verifier BF16 logits");
    retire(candidate, 1);
}

void committed_k4(Decoder& decoder, const std::vector<int>& tokens,
                  const std::vector<int>& suffix) {
    constexpr int interval = decoder_batch_prefix_checkpoint_interval;
    auto prompt = head(tokens, interval - 9 - suffix.size());
    prompt.insert(prompt.end(), suffix.begin(), suffix.end());
    DecoderBatch oracle(decoder, {false, 2, 2, 1024});
    DecoderBatch candidate(decoder, {false, 2, 2, 1024});
    DecoderOptions options;
    options.mtp_steps = 4;
    options.ignore_eos = true;
    options.max_new_tokens = 96;
    require(oracle.admit(0, prompt, options) == 0, "K4 oracle started warm");
    int emitted = 0;
    int stop_at = -1;
    auto producer = options;
    producer.text_stop = [&](const int*, int count, bool) {
        const int cutoff = stop_at - emitted;
        if (cutoff > 0 && cutoff <= count) {
            emitted += cutoff;
            return cutoff;
        }
        emitted += count;
        return 0;
    };
    require(candidate.admit(1, prompt, producer) == 0, "K4 producer started warm");
    auto history = prompt;
    history.push_back(first_token(oracle, 0));
    require(first_token(candidate, 1) == history.back() && emitted == 1 &&
            candidate.prefix_cache_stats().stores == 0,
            "K4 published at the off-cadence prompt end");

    bool accepted_proposal = false;
    bool crossed = false;
    int verified_rows = 0;
    int committed_rows = 0;
    while (emitted < options.max_new_tokens - 5) {
        const int start = history.size() - 1;
        const auto expected = oracle.advance_chunked(slots({0}))[0];
        require(expected.token && !expected.finish_reason, "K4 oracle did not advance");
        const int count = expected.extra_token_count + 1;
        const int end = start + count;
        std::cout << "K4 position " << start << " drafted " << expected.mtp_drafted_tokens
                  << " accepted " << expected.mtp_accepted_tokens << " verified "
                  << expected.mtp_verified_rows << " committed " << count << '\n';
        const bool rejected = expected.mtp_drafted_tokens == 4 && expected.mtp_replayed_rows > 0;
        accepted_proposal |= expected.mtp_drafted_tokens == 4 && expected.mtp_accepted_tokens > 0;
        if (start >= interval && rejected && accepted_proposal) {
            stop_at = emitted + count;
            verified_rows = expected.mtp_verified_rows;
            committed_rows = count;
        }
        const auto actual = candidate.advance_chunked(slots({1}))[1];
        auto reference = expected;
        if (stop_at != -1) reference.finish_reason = DecoderFinishReason::stop;
        compare_counters(reference, actual);
        history.push_back(*actual.token);
        history.insert(history.end(), actual.extra_tokens.begin(),
                       actual.extra_tokens.begin() + actual.extra_token_count);
        const auto saved = candidate.prefix_cache_stats();
        if (start < interval && end >= interval) {
            require(end == interval && expected.mtp_verified_rows <= interval - start &&
                    !actual.finish_reason, "K4 verifier crossed an unpublished cadence checkpoint");
            crossed = true;
            require(candidate.admit(0, history, options) == interval, "K4 cadence key differs");
            require(active_lane(candidate, 0).draft->valid_prefix() == interval - 1,
                    "Cadence draft cursor differs");
            candidate.cancel(0);
        }
        require(saved.entries == int(crossed) + int(actual.finish_reason.has_value()) &&
                saved.stores == static_cast<unsigned>(saved.entries),
                "Unexpected K4 checkpoint publication");
        if (actual.finish_reason) break;
    }
    const int processed = history.size() - 1;
    require(crossed && verified_rows > committed_rows && committed_rows > 0 &&
            processed > interval && !candidate.request_device_allocation_bytes(),
            "Fixture lacked a terminal K4 rejection");
    compare_verified_checkpoint(oracle, candidate, history, options, committed_rows);

    auto branch = history;
    branch[processed - 1] = (branch[processed - 1] + 1) % vocab_size;
    require(candidate.admit(1, branch, options) == interval,
            "Rejected suffix matched terminal key");
    retire(candidate, 1);
    require(candidate.admit(1, history, options) == processed, "Last emitted K4 token was cached");
    DecoderBatchStepStats stats;
    const auto continuation = candidate.advance_chunked(slots({1}), &stats)[1];
    const auto expected = oracle.advance_chunked(slots({0}))[0];
    require(continuation.token && continuation.token == expected.token &&
            stats.prefill_rows[1] == 1, "Last emitted K4 ID did not run as a normal suffix");
    retire(candidate, 1);
    retire(oracle, 0);
    std::cout << "K4: cadence 8192, terminal " << processed << " after " << committed_rows
              << " of " << verified_rows << " verified rows, exact target/draft/logits and suffix"
              << '\n';
}

}  // namespace

const Bf16* actual_enqueue(MtpDraftRequest*, int, int, const Bf16*, bool, cudaStream_t,
                          const ConstraintWorkspace*, const ConstraintCandidates*,
                          const MropePosition*, int, const Bf16*)
    asm("__real__ZN6sglang7decoder15MtpDraftRequest7enqueueEiiPK13__nv_bfloat16b"
        "P11CUstream_stPKNS0_19ConstraintWorkspaceEPKNS0_20ConstraintCandidatesE"
        "PKNS_13MropePositionEiS4_");
const Bf16* observed_enqueue(MtpDraftRequest*, int, int, const Bf16*, bool, cudaStream_t,
                            const ConstraintWorkspace*, const ConstraintCandidates*,
                            const MropePosition*, int, const Bf16*)
    asm("__wrap__ZN6sglang7decoder15MtpDraftRequest7enqueueEiiPK13__nv_bfloat16b"
        "P11CUstream_stPKNS0_19ConstraintWorkspaceEPKNS0_20ConstraintCandidatesE"
        "PKNS_13MropePositionEiS4_");

const Bf16* observed_enqueue(MtpDraftRequest* draft, int position, int shifted_token,
                             const Bf16* hidden, bool score, cudaStream_t stream,
                             const ConstraintWorkspace* scratch,
                             const ConstraintCandidates* allowed, const MropePosition* mrope,
                             int rotary_position, const Bf16* feature) {
    if (draft == observed_draft && !boundary_input) {
        require(position == page - 1, "First draft rebuild did not start at P - 1");
        boundary_input = shifted_token;
    }
    return actual_enqueue(draft, position, shifted_token, hidden, score, stream, scratch, allowed,
                          mrope, rotary_position, feature);
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
        require(argc == 5, "Pass checkpoint, 257 short IDs, 8192 long IDs, and 128 W1 IDs");
        const auto tokens = ids(argv[2], 257);
        const auto long_tokens = ids(argv[3], 8192);
        const auto w1 = ids(argv[4], 128);
        const Tokenizer tokenizer(std::string(argv[1]) + "/tokenizer.json");
        Decoder decoder(argv[1], 0, true);
        for (const int length : {1, page, page + 1}) enrichment(decoder, tokens, length);
        changed_tiles(decoder, long_tokens);
        changed_first_token(decoder, tokenizer, tokens);
        DecoderBatch cold(decoder, {false, 2, 1, page});
        DecoderBatch warm(decoder, {false, 2, 4, page});
        check_charge(warm, 0, 0);
        const auto prompt = head(tokens, page);
        interrupted_store(warm, prompt);
        interrupted_restore(cold, warm, prompt);
        check_charge(warm, 1, 1);
        committed_k4(decoder, long_tokens, w1);
        std::cout << "Prefix MTP enrichment, accounting, continuation and fault retry passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
