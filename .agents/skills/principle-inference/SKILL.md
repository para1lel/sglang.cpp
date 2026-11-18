---
name: principle-inference
description: Rules for model state, serving semantics, prefix reuse, and speculative inference.
disable-model-invocation: true
---

# Inference engineering principles

Use this domain with the project's adopted general rules.
Apply the relevant rules to the model and execution paths that the project supports.
These rules do not require a particular framework, model family, or device topology.

## Define the model and runtime contracts

Pin the model configuration, weights, tokenizer, numerical policy, and reference revision.
Verify model-specific behavior against those sources.
A similar model name or an earlier design discussion does not establish that behavior.

Separate ownership of model mathematics, kernel execution, request scheduling, logical cache state,
physical storage, and speculative verification.
Define the interfaces and invariants between these responsibilities.
This separation does not require a particular class hierarchy or virtual dispatch.
Batch formation and device partitioning must preserve each request's positions and state ownership.
Publish the resolved numerical and dispatch policies used by the running engine.
Compare those policies before routing compatible requests or sharing state between replicas.
Keep permitted worker-capacity differences separate from semantic compatibility.

Establish a correct execution path before combining optimizations that change several contracts.
Use that path to isolate a changed mechanism while preserving unrelated numerical settings.
Do not replace an existing verified path solely to impose a new development sequence.

## Describe state by meaning and coverage

Define token positions, counts, and interval endpoints before using them in cache interfaces.
For each state component, identify its owner, valid token coverage, dependencies, and recovery method.
Distinguish token-indexed KV entries, mutable recurrent or convolution state, immutable checkpoints,
and temporary replay state.
Their retention and recovery rules can differ.

Keep processed target positions, accepted draft tokens, and emitted output tokens distinct.
A correction or bonus token can be emitted before the model processes it.
Do not use one accepted-token count as the valid length of every state component.
Define the coherent request state at each commit boundary.

Share immutable state between requests.
Use private mutable state or copy-on-write before a fork changes shared storage.
Reuse a prefix only when its state, numerical policy, and model identity are compatible.
A token-prefix match can extend beyond the nearest usable recurrent checkpoint.
Restore a valid ancestor and replay the missing interval when the state cannot be sliced.

## Commit speculative work coherently

Keep tentative execution private until target verification establishes the accepted work.
Physical writes can occur before acceptance.
Publication for shared prefix reuse must wait for valid state and completed dependencies.

Define how each state component commits, restores, or replays after rejection.
Commit the accepted prefix and remove rejected effects from the request's observable state.
Update token history, component coverage, and storage ownership as one coherent transition.
Publish reusable state only after that transition is valid.

Examine zero, partial, and full acceptance, including correction and bonus-token handling.
When a chunk crosses a checkpoint boundary, capture the state at the required position.
A test of only the chunk's final position can miss that boundary.
Choose checkpoint spacing and tail retention from recovery cost and memory constraints.
Do not require a fixed interval or a snapshot for every speculative token.

For stochastic decoding, preserve the target distribution required by the sampling contract.
Equal seeds alone do not require equal output sequences when random-number consumption differs.
Use the [speculative decoding paper](https://proceedings.mlr.press/v202/leviathan23a.html)
for the distribution-preserving verification method and its assumptions.

## Complete snapshots before reuse

Protect snapshot sources until all asynchronous readers finish.
Use a dependency, immutable staging, or another mechanism that preserves the captured state.
A destination allocation, launch, or recorded event does not establish completion.
Do not mutate or recycle the source while its snapshot copy still reads it.

Carry ownership through the actual stream, graph, transfer, and request lifetimes.
Cancellation and eviction must retire pending users before storage reuse.
Restore only a completed checkpoint whose model identity, component coverage, and format match.
Use the target version's [CUDA guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
or the execution backend's primary documentation for completion and lifetime requirements.

## Specify numerical storage

Treat weight, KV, active-state, and checkpoint precision as separate decisions.
For compressed storage, define values, scales, scale grouping, layout, padding, and tail handling.
Include this format in cache compatibility and restore behavior.
An allocation size or a quantization name does not fully describe the representation.

Lossless storage preserves saved bits.
It does not by itself prove bitwise equality after execution with a different operation order.
Use exact comparisons for discrete metadata and justified tolerances for numerical results.
Assess compression error separately from changes to cache reuse or speculative execution.
Keep arithmetic and reduction policies consistent across equivalent decode, prefill, and speculative paths.
Select canonical partitions from each query's visible prefix when equivalence requires them.
Locate the first divergent state or logit row before selecting a numerical correction.
Plausible output text and wider tolerances do not establish the required equivalence.

## Preserve serving and sampling semantics

Preserve contract-significant schema order through every parser and ABI boundary.
For tool-call responses, preserve explicit termination causes, parsed content, and usage.
Examine blocking and streaming paths with the same request and cutoff conditions.
Do not restore raw syntax that a parser has already consumed.
An ordered type at one boundary cannot recover order lost at an earlier boundary.

Keep probability normalization consistent with the sampler's precision and mass requirements.
Exercise a dominant head with small positive tails as well as ordinary distributions.
Do not discard required probability mass or weaken tolerances to hide an incompatible normalizer.

## Verify transitions with independent references

Use references for distinct questions:

- An independent mathematical implementation checks an operation's numerical meaning.
- A pinned model implementation with matching weights and tokenizer checks model behavior.
- A simple verified runtime path checks cache, replay, batching, and speculation equivalence.

A runtime comparison must keep precision and other unrelated policies fixed.
A model reference that shares the affected kernel cannot independently establish that kernel's math.
State what each reference can establish.
Capture emitted native token IDs directly.
Re-encoding output text does not establish the exact IDs emitted by the runtime.
A linked checkpoint implementation does not establish that a probe enabled checkpoints.

Select transitions that can expose the changed contract.
Examples include cold versus reused prefixes, whole versus chunked execution, save and restore,
fork and eviction, request reordering, and rejection across a checkpoint boundary.
Check hidden recurrent and convolution state as well as the next-token logits when relevant.
Test logical contents, coverage, ownership, and reference counts.
Equivalent runs need not allocate the same physical page identifiers.
Use synchronization or memory checks when numerical comparisons cannot expose a lifetime defect.

Keep distinct transition coverage without a full product of every model, length, and batch size.
For stochastic behavior, use the algorithm's distribution argument and suitable statistical checks.
Finite samples alone do not prove exact distribution equivalence.

## Measure useful inference work

Measure time to first token, token latency, useful output throughput, and memory capacity separately.
Count actual useful output tokens, not rejected proposals or replayed positions, in output throughput.
Include draft execution, target verification, replay, checkpoint transfer, and restore costs in
the end-to-end metric that claims their benefit.

Compare matching workloads, numerical policies, hardware, and cache conditions.
Report cold and reused-prefix paths separately when they differ.
Acceptance rate and cache hit rate explain activity; neither alone establishes a speedup.
Keep per-request or per-case failures visible when the claim applies to every case.
