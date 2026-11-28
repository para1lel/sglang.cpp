# Native MTP

Multi-token prediction (MTP) verifies draft candidates in order against the
target model. It supports greedy decoding and sampling.

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Generator and server | Set the proposal limit and pass request options to the decoder. | [Generator options](../apps/sglang-generate.cpp#L59)<!-- prefix: "void usage(s" --> · [Server dispatch](../apps/sglang-server.cpp#L464)<!-- prefix: "BatchService" --> |
| MTP round | Draft proposals and commit target outputs in order. | [Proposals](../src/decoder_verify.cpp#L171)<!-- prefix: "bool MtpRoun" --> · [Commit](../src/decoder_verify.cpp#L293)<!-- prefix: "bool MtpRoun" --> |
| Target verifier | Process causal rows by layer and commit each request's consumed prefix. | [Forward](../src/decoder_verify.cpp#L855)<!-- prefix: "bool Decoder" --> · [State commit](../src/runtime/execution/decoder_verify_workspace.cpp#L48)<!-- prefix: "void TargetV" --> |

## Runtime path

`sglang-server` and `sglang-generate` default to MTP depth four (K4). Use `--mtp-steps 0` to disable MTP (K0). Values `1`–`4` limit draft proposals per round. Both commands accept `--mtp-k` as an alias. C++ `DecoderOptions` defaults to K0. [Depth option](../apps/sglang-server.cpp#L168)<!-- prefix: "} else if (o" --> · [Server default](../apps/sglang-server.cpp#L56)<!-- prefix: "int mtp_step" --> · [Generator default](../apps/sglang-generate.cpp#L153)<!-- prefix: "arguments.op" --> · [C++ options](../include/sglang/decoder.h#L64)<!-- prefix: "int mtp_step" -->

At capacities 1–32, text, media, and native generation use configured depth. Greedy [token logprobs](native-generate.md#token-logprobs) can score committed MTP rows. [Sampling](#sampled-mtp) and [constraints](#constrained-mtp) use their own verification paths. Pointwise scores and target-only options use K0. `Decoder::generate` uses one batch lane for MTP. Draft weights load once with shared target embeddings and head, including [B200 INT8 support](native-model-info.md#int8-draft-head). Each request owns target state, draft MXFP8 KV, and verifier storage. C++ selects depth per lane. Positive-depth lanes can [restore paired prefixes](prefix-cache.md#checkpoint-compatibility-and-resume). [Effective depth](../src/decoder_batch.cpp#L217)<!-- prefix: "if (options." --> · [Verifier scores](../src/decoder_verify.cpp#L452)<!-- prefix: "void TargetV" --> · [Weights](../src/decoder.cpp#L488)<!-- prefix: "Decoder::Dec" --> · [Admission](../src/decoder_batch.cpp#L264)<!-- prefix: "int DecoderB" -->

Eligible B200 prompt tiles hold up to 1,024 rows. Other devices cap at 128. Draft seeding joins each final-normalized target row with the next prompt embedding, including media coordinates. After two normalizations and FC/KV projection it rotates keys and appends MXFP8. It skips query projection, attention, MLP, and scoring. Single rows or insufficient scratch use scalar prefill. [Tile cap](../src/runtime/execution/decoder_execution_policy.cpp#L25)<!-- prefix: "int DecoderE" --> · [Draft input](../src/decoder_prefill.cpp#L204)<!-- prefix: "seed.seed_dr" --> · [Draft seed](../src/decoder_mtp.cpp#L714)<!-- prefix: "void MtpDraf" -->

For prompts ≥4,096 tokens, eligible B200 Tensor Core seed FC can use TMA for 64–128 rows at 10,240×5,120 channels. Input and weights need 16-byte alignment. TMA also needs 4-byte output alignment, the image, tensor-map support and shared memory. Preflight decline uses 32×64 WMMA. Ineligible FC shapes or inputs use SIMT, which can change draft KV and proposals. CUDA errors propagate. [Prefill selection](../src/decoder_prefill.cpp#L202)<!-- prefix: "const bool w" --> · [FC dispatch](../src/text/draft_seed_fc.cu#L124)<!-- prefix: "cudaError_t " --> · [TMA preflight](../src/text/draft_seed_fc_sm100.cu#L196)<!-- prefix: "std::optiona" --> · [Submission](../src/text/draft_seed_fc_sm100.cu#L239)<!-- prefix: "return submi" -->

B200 multi-row seed prepares FP8 in slices of at most 128 rows, then projects K and V in separate grid banks and issues one key normalization/rotation and one append. Each grouped request seeds separately. Other devices and single rows process each slice separately. [existing prefill workspace](native-prefill-workspace.md#seed-scratch) · [Paired projection](../src/text/fp8_projection.cu#L701)<!-- prefix: "cudaError_t " -->

The target scores the last prompt row and selects the first token. If generation continues, that token and the last normalized target row start the draft round at `L-1`. Later proposals use the previous draft's final-normalized row and prediction. The proposal loop synchronizes the stream before reading winners and again before reading BF16 retry results. [Proposal loop](../src/decoder_verify.cpp#L171)<!-- prefix: "bool MtpRoun" -->

Eligible B200 draft lanes select [grouped online attention](native-model-info.md#draft-attention) automatically.

### Draft winner readback

On B200, two to four unconstrained lanes with mapped pinned storage share two argmax launches. The first reduces tiles. The second writes each winner directly to the mapped record. Both follow per-lane tie and nonfinite rules. [Dispatch](../src/decoder_mtp.cpp#L312)<!-- prefix: "void MtpDraf" --> · [Launcher](../src/text_ops.cu#L539)<!-- prefix: "cudaError_t " --> · [Selection](../src/text_ops.cu#L221)<!-- prefix: "__device__ T" -->

Capture can reuse mappings. Eager requests can allocate them on demand. Singletons, constrained lanes, and other devices use per-lane selection and copy. If any mapping allocation fails or capture prevents it, the group uses per-lane selection. Other CUDA errors propagate. [Mapping](../src/decoder_mtp.cpp#L292)<!-- prefix: "bool MtpDraf" --> · [Allocation](../src/decoder_verify.cpp#L512)<!-- prefix: "std::shared_" --> · [Per-lane selection](../src/decoder_mtp.cpp#L312)<!-- prefix: "void MtpDraf" -->

### Target round

Each target round starts with the latest committed token. Proposals supply subsequent inputs. The batch limits proposals to `K`, remaining outputs minus one, and assigned target rows minus one. A one-row verification makes no proposals. One remaining output uses ordinary eager decode. Zero budget without scoring allocates no request state. [Batch assignment](../src/decoder_batch.cpp#L441)<!-- prefix: "if (results[" --> · [Zero-budget finish](../src/decoder_batch.cpp#L275)<!-- prefix: "if (options." -->

Ordinary packets interleave causal rows by position across up to four slots and 20 rows. Each lane commits independently. Packets of 2–20 rows can share BF16 head projection, with grouped fallback on clean decline and CUDA error propagation. Serial routes can request raw BF16 logits. Eligible greedy rounds use the [certified head](native-model-info.md#certified-tensor-core-target-head).

B200 wide rounds require loaded draft weights, certified scratch, capacity ≥8, and a step cap ≥40. Combine adjacent eager packets into 8/16/32 lanes and 40/80/160 rows, with TP2 capped at 16 lanes. Each lane needs five granted and remaining rows, greedy unconstrained K4, and no raw logits. Sparse slots preserve order. [Wide FP8](fp8-prefill-projections.md#multi-lane-target-verification), [certified vocabulary](native-model-info.md#certified-tensor-core-target-head), [draft repair](native-model-info.md#grouped-draft-projections), [long GQA](singleton-gqa-decode.md#tensor-core-reads-across-long-k4-packets), and [compact GDN](gdn-prefill.md#grouped-target-verification) describe grouped operators. Short drafts use eligible aligned groups of eight or ordinary packets. [Target round](../src/decoder_verify.cpp#L644)<!-- prefix: "bool Decoder" --> · [Packed head](../include/sglang/text/bf16_head_projection.h#L15)<!-- prefix: "std::optiona" --> · [Planner](../src/text/decoder_batch_plan.cpp#L27)<!-- prefix: "WideMtpGroup" -->

The controller commits matching proposals in order. The first mismatch commits a target correction and ends the round. It leaves later proposals unconsumed. Full acceptance can commit a target bonus within the stop rule and output budget. A stop match takes precedence over length. A correction or bonus becomes the next target input if generation continues. Draft KV rewinds to the first target position of the round. Continuing lanes rebuild only the accepted draft KV rows from target-normalized rows. [Round commit](../src/decoder_verify.cpp#L293)<!-- prefix: "bool MtpRoun" -->

Verification appends tentative GDN rows to the request's [ReplaySSM log](gdn-decode.md#runtime-path). Commit keeps the consumed prefix and clears the pending count. The target KV prefix advances by the same number of rows. Later forwards bound GQA reads by position and overwrite rejected slots before they become visible. [Causal attention](../src/runtime/execution/decoder_forward_attention.cpp#L358)<!-- prefix: "void attend_" --> · [State commit](../src/runtime/execution/decoder_verify_workspace.cpp#L48)<!-- prefix: "void TargetV" -->

C++ callers consume `token` followed by the first `extra_token_count` entries of `extra_tokens`; `extra_token` mirrors the first extra. HTTP publishes only after batch commit. Per-token callbacks check cancellation between IDs. See [native step delivery](native-generate.md#text-stops) and [bounded output](http-server.md#bounded-decode-output). [Batch API](../include/sglang/decoder_batch.h#L115)<!-- prefix: "class Decode" --> · [Token delivery](../apps/server/detail/batch_state.cpp#L117)<!-- prefix: "bool emit_st" --> · [Step delivery](../apps/server/batch_service.cpp#L543)<!-- prefix: "bool BatchSe" -->

### Target verification graph

TP1 [replay](batch-graphs.md) supports one unconstrained text lane with two target rows. Both positions must fit inline GQA scores. The last eligible pair is 5,118 and 5,119 (zero-based). Replay and eager verification share two-row projection grouping. Replay uses bound GQA dispatch, including the [split-KV position check](singleton-gqa-decode.md#split-kv-target-attention) when the split policy is active. The first eligible group runs eagerly. The next captures the verifier on the batch stream. [Group dispatch](../src/decoder_verify.cpp#L879)<!-- prefix: "if (graph &&" --> · [Row eligibility](../src/decoder_verify.cpp#L397)<!-- prefix: "bool TargetV" --> · [Capture](../src/decoder_verify.cpp#L880)<!-- prefix: "std::exchang" -->

Each launch refreshes tokens, positions, KV page tables and pointers, ReplaySSM state, and GQA score bindings. The executable reuses fixed batch-owned scratch. Draft proposals, sampling, result readback, acceptance, and state commit run outside it. C++ reports replayed slots in `verification_graph_slots`. The HTTP `graph_slots` trace reports K0 replay. [Bindings](../src/decoder_verify.cpp#L52)<!-- prefix: "void Verific" --> · [Slot mask](../include/sglang/decoder_batch.h#L52)<!-- prefix: "std::uint32_" --> · [HTTP trace](../apps/server/detail/batch_trace.cpp#L46)<!-- prefix: "if (stats.gr" -->

TP1 K4 full rounds, direct `Decoder::generate`, and ineligible groups run eagerly. One output remaining uses ordinary decode. Other groups can still qualify. [TP2 verification replay](batch-graphs.md#tp2-target-verification-graph) has separate eligibility. `--eager` disables both.

### Sampled MTP

An item with active [output penalties](output-penalties.md), [logit bias](serving/openai-logprobs.md#logit-bias), or a positive [output minimum](output-penalties.md#minimum-output) uses K0. Eligible peers in the same batch keep their configured depth. [Target-only selection](../src/runtime/execution/decoder_options.cpp#L18)<!-- prefix: "bool Decoder" -->

The draft proposes greedy IDs. On each causal row the [target sampler](text-serving.md#sampling) draws `T` from its filtered law `P`, including the first output. Each draw uses the resolved seed and prospective output index. It emits `T`, accepting draft `d` if `T == d` (probability `P(d)`). The target draw supplies the correction directly. Full acceptance permits a bonus under the budget and stop rules.

Discarded rows leave their output indices available on the corrected history. Identical effective rows, sampler settings, seeds, and output indices match K0. Prefill, forward precision, and batch grouping can change logits and draws. [Verifier sampler](../src/decoder_verify.cpp#L430)<!-- prefix: "void TargetV" -->

### Constrained MTP

The first target output uses the [ordinary constraint selection](output-constraints.md#selection-and-finish). Each round starts from the committed cursor. The draft chooses the highest-scoring allowed ID and advances a private cursor. Each row uses its own remaining output budget. If a proposal completes the constraint, the target verifies it and skips the bonus row. [Round cursor](../src/decoder_verify.cpp#L81)<!-- prefix: "void MtpCons" -->

The target filters each verified row with the allowed IDs for that candidate prefix. It uses the ordinary greedy rule or samples with the prospective output index. Only emitted target IDs advance the live cursor. A mismatch or finish discards later proposals and their cursor states. A correction or bonus can also complete the constraint. Identical score rows, cursors, sampler options, and output indices give identical target decisions. [Target selection](../src/decoder_verify.cpp#L440)<!-- prefix: "void TargetV" -->

## Constraints and failure behavior

The verifier reserves up to K+1 rows, limited by the initial budget minus one and the step cap. Resident batches reserve on admission. Other paths reserve before first verification. The request owns the [ReplaySSM log](gdn-decode.md#runtime-path) independently of verifier workspace. The batch API rejects top logits. Constraints add up to 16,384 pinned bytes per row for IDs or 31,040 for a wide mask. [Resident batch path](batch-graphs.md#resident-memory-admission) · [Verifier workspace](../src/runtime/execution/decoder_verify_workspace.cpp#L6)<!-- prefix: "Verification" --> · [Constraint staging](../src/decoder_verify.cpp#L66)<!-- prefix: "MtpConstrain" -->

Batch steps release prefill scratch. Check cancellation during steps and [prefix operations](prefix-cache.md#constraints-and-failure-behavior). On recoverable failure, fail current jobs and drain; outstanding restores retire their slots when complete while peers can admit replacements. Uncertain GPU completion disables the batch and retains storage. [Step recovery](../src/decoder_batch.cpp#L918)<!-- prefix: "} catch (con" --> · [Cleanup](../src/runtime/backend/cuda_resources.cpp#L234)<!-- prefix: "CleanupError" -->

Draft prefix advances after successful append submission. On a later preparation failure the paired path attempts to append prepared rows before reporting it. Draft preparation, projection, and append failures mark the draft failed. A target normalization failure leaves its flag clear if the flush succeeds. [Append](../src/decoder_mtp.cpp#L748)<!-- prefix: "check(mxfp8_" --> · [Target normalization failure](../src/decoder_mtp.cpp#L781)<!-- prefix: "append_prepa" --> · [Draft preparation failure](../src/decoder_mtp.cpp#L806)<!-- prefix: "append_prepa" -->

For recoverable TP1 verifier setup failures, eager fallback requires ended capture, a drained stream, and successful handle cleanup. This disables only the verifier executable. A binding upload failure follows speculative step recovery without replaying the failed round. A verification launch error or unconfirmed capture cleanup disables the batch without reevaluating the target. Resources stay live until safe cleanup. [Graph recovery](../src/runtime/backend/cuda_resources.cpp#L550)<!-- prefix: "const auto f" --> · [Launch](../src/runtime/backend/cuda_resources.cpp#L627)<!-- prefix: "const auto l" -->

If a constrained draft row has no valid winner, drafting stops and the target can still select an output. For an invalid winner on a consumed constrained row, direct generation throws `DecoderConstraintError`. The batch reports `constraint_failed` and retires the lane. For an invalid unconstrained sample, direct generation throws `DecoderSamplingError`. The batch reports `sampling_failed`. HTTP drops the failed lane's unpublished step tokens. Rows discarded after a mismatch or stop have no effect. [Batch consumption](../include/sglang/runtime/model/decoder_batch_lane.h#L55)<!-- prefix: "void finish_" --> · [HTTP failure](../apps/server/batch_service.cpp#L260)<!-- prefix: "if (step.sam" -->

## Verification

Run the [MTP acceptance procedure](../tests/mtp/README.md) with both release binaries, the pinned checkpoint, and a reserved B200. Require exit zero. Compare default against explicit K4 and compare K0 on matched target and prompt tiles. Check IDs, finishes, usage, and wide SSE. For seed parity pass a row-wise seed generator with the same target verifier to `--reference-generate`.

For grouped winners compare full BF16 rows, IDs and scores at widths 2–4, including ties and invalid inputs. Trace mapped writes and lane fallback in eager and graph. Seed probes cover positions 1/127/128/129/257, long, partial, media and restored tiles and scores. Compare paired projection/rotary against per-slice controls and TMA FC against WMMA, and exercise SIMT. Match target state, draft KV and later logits when arithmetic is preserved. Require one paired projection and append per tile. Inject a second-slice failure and require first-slice-only advance and lane reuse.

Compare target state with row-wise controls across zero, partial and full acceptance, rewind, KV crossings, mixed depths and score capacity. At widths 8/16/32 match grants, target dispatch, and fixed histories with four-lane drafting. Cover sparse K1 wider than four, one-row no-proposal grants, final eager token, zero budget, decline, hard errors, reset and zero retired bytes at capacity 32. Match IDs, finish and usage, distinguishing draft drift from committed state. [Wide projection controls](fp8-prefill-projections.md#wide-verifier-checks) · [Native MTP diagnostics](performance.md#native-mtp)

### Verification graph checks

Use the [batch graph runbook](../tests/batch/README.md) on a reserved B200 for BF16 row and ID parity, sparse replacement, mixed K0/K1 steps, eager exclusions, capture decline and terminal launch or cleanup faults. [Runner](../tests/mtp/run_native.py#L1)<!-- prefix: "\"\"\"Opt-in re" -->

The [HTTP probe](../tests/mtp/graph_http.py#L213)<!-- prefix: "def main():" --> checks text and chat SSE IDs, finishes, usage and verifier counters against eager, with CUPTI request intervals. Cover positions 127/128, the last eligible pair, 5,119/5,120 and larger positions. Require warmup, acceptance, sampled rejection, stop/EOS, final one-output round and reset/replacement. Also limit a higher depth to two target rows. Follow the [constraint gates](#constrained-verification).

### Sampled verification

Follow [sampled MTP](../tests/mtp/README.md#sampled-mtp) for the CPU target-law oracle and live HTTP commands. Cover first rejection, partial/full acceptance, stop, and budget at depths 0–4. On a reserved B200, compare fixed seeds with matched prefill, prompt and budget. Compare [sampler filters](../tests/text/README.md#sampling) with a CPU row oracle. For a differing ID inspect ordered scores and margins with `--top-logits 32` and compare full BF16 rows on shared histories.

With `CHECKPOINT` set to the pinned model directory and the reserved GPU visible as device zero, run each depth with the same options:

```sh
for depth in 0 1 2 3 4; do
  sglang-generate --checkpoint "$CHECKPOINT" --device 0 \
    --prompt-ids 48,25,220,16,10,16,28 --max-new-tokens 21 \
    --temperature 1.8 --top-p 0.89 --top-k 56 --min-p 0.04 \
    --seed -9223372036854775808 --top-logits 32 --mtp-steps "$depth" \
    > "sampled-k${depth}.json" || break
done
```

Require 21 outputs and `length` at each depth. Compare `output_token_ids` and `step_top_logits`. Check K0 has no draft counters and positive depths record drafting and rejection.

Run both HTTP text routes at configured capacities one and four. Match JSON and SSE IDs, usage, and finish reasons for the same seeded cases. Exercise different seeds, greedy peers, rejection, EOS, and a one-token budget. Trace draft counts and dispatch separately from timing.

### Constrained verification

Follow the [positive-depth constraint checks](output-constraints.md#positive-depth-checks) for the HTTP and C++ routes. Check target state and continuation as for other verifier changes.

### Measured trade-offs

Early rejection can waste precomputed rows and require GDN replay. Check the intended depth and seed policy on fixed prompts. Follow [AgentX](performance.md#agentx-performance-standard) for performance evaluation. A fixed-schedule TP2 state fixture matched all 80 terminal payloads byte for byte, along with output IDs and acceptance.
