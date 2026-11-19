#include "sglang/decoder_internal.h"

#include "sglang/decoder_constraint_cuda.h"
#include "sglang/decoder_sampling.h"

namespace sglang::decoder {

void ConstraintWorkspace::allocate(Allocations& memory) {
    if (!ids) {
        logits = memory.allocate<Bf16>(constraint_max_candidates);
        ids = memory.allocate<int>(constraint_max_candidates);
    }
}

std::size_t ConstraintWorkspace::bytes() const {
    return ids ? constraint_max_candidates * (sizeof(int) + sizeof(Bf16)) : 0;
}

void select_constrained(Workspace& work, const ConstraintWorkspace& scratch,
                        const ConstraintCursor& cursor, const SamplingParameters* sampling,
                        int position, void* sampling_workspace, cudaStream_t stream) {
    const auto& allowed = cursor.ids();
    const int count = static_cast<int>(allowed.size());
    if (count == 0) {
        throw std::logic_error("Output constraint has no remaining continuation");
    }
    check(cudaMemcpyAsync(scratch.ids, allowed.data(), count * sizeof(int),
                          cudaMemcpyHostToDevice, stream), "Upload allowed token IDs");
    check(gather_constrained_logits(work.logits, scratch.ids, scratch.logits, count, stream),
          "Gather allowed logits");
    if (sampling) {
        auto parameters = *sampling;
        parameters.top_k = std::min(parameters.top_k, count);
        check(sample_logits(scratch.logits, count, &work.argmax->token_id, parameters,
                            position, sampling_workspace, stream), "Sample allowed logits");
    } else {
        check(text_argmax(scratch.logits, count, work.argmax_scratch, work.argmax, stream),
              "Choose highest allowed logit");
    }
    check(remap_constrained_token(scratch.logits, scratch.ids, count, &work.argmax->token_id,
                                  stream), "Resolve allowed token ID");
}

}
