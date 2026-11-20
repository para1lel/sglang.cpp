#include "sglang/decoder_internal.h"

#include "sglang/decoder_constraint_cuda.h"
#include "sglang/decoder_sampling.h"

namespace sglang::decoder {

void ConstraintWorkspace::allocate(Allocations& memory, bool wide) {
    if (!ids) {
        logits = memory.allocate<Bf16>(constraint_max_candidates);
        ids = memory.allocate<int>(constraint_max_candidates);
    }
    if (wide && !masked_logits) {
        masked_logits = memory.allocate<Bf16>(vocab_size);
    }
    if (wide && !mask) {
        mask = memory.allocate<std::uint32_t>(constraint_mask_words);
    }
}

std::size_t ConstraintWorkspace::bytes() const {
    const auto sparse = ids ? constraint_max_candidates * (sizeof(int) + sizeof(Bf16)) : 0;
    const auto mask_bytes = mask ? constraint_mask_words * sizeof(std::uint32_t) : 0;
    const auto masked_bytes = masked_logits ? vocab_size * sizeof(Bf16) : 0;
    return sparse + mask_bytes + masked_bytes;
}

void select_constrained(LogitSelection work, const ConstraintWorkspace& scratch,
                        ConstraintCandidates allowed, const SamplingParameters* sampling,
                        int position, void* sampling_workspace, cudaStream_t stream) {
    if (allowed.count == 0) {
        throw std::logic_error("Output constraint has no remaining continuation");
    }
    const bool wide = allowed.mask != nullptr;
    const int count = wide ? vocab_size : allowed.count;
    const auto* logits = wide ? scratch.masked_logits : scratch.logits;
    if (wide) {
        check(cudaMemcpyAsync(scratch.mask, allowed.mask,
                              constraint_mask_words * sizeof(std::uint32_t),
                              cudaMemcpyHostToDevice, stream), "Upload allowed token mask");
        check(mask_constrained_logits(work.logits, scratch.mask, scratch.masked_logits,
                                      count, stream), "Mask disallowed logits");
    } else {
        check(cudaMemcpyAsync(scratch.ids, allowed.ids, count * sizeof(int),
                              cudaMemcpyHostToDevice, stream), "Upload allowed token IDs");
        check(gather_constrained_logits(work.logits, scratch.ids, scratch.logits, count, stream),
              "Gather allowed logits");
    }
    if (sampling) {
        auto parameters = *sampling;
        parameters.top_k = std::min(parameters.top_k, allowed.count);
        check(sample_logits(logits, count, &work.argmax->token_id, parameters,
                            position, sampling_workspace, stream), "Sample allowed logits");
    } else {
        check(text_argmax(logits, count, work.argmax_scratch, work.argmax, stream),
              "Choose highest allowed logit");
    }
    const auto* ids = wide ? nullptr : scratch.ids;
    check(remap_constrained_token(logits, ids, count, &work.argmax->token_id, stream),
          "Resolve allowed token ID");
}

void select_constrained(Workspace& work, const ConstraintWorkspace& scratch,
                        const ConstraintCursor& cursor, const SamplingParameters* sampling,
                        int position, void* sampling_workspace, cudaStream_t stream) {
    const ConstraintCandidates allowed{static_cast<int>(cursor.ids().size()), cursor.ids().data(),
                                       cursor.mask().empty() ? nullptr : cursor.mask().data()};
    select_constrained({work.logits, work.argmax_scratch, work.argmax}, scratch, allowed, sampling,
                       position, sampling_workspace, stream);
}

}
