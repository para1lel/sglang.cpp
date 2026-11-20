# Development handoff

The last published implementation is `c6b421091d24a7d1d79dc7c99d2df3cddbbbb76b`. It supports text prefix reuse with exact prefill at MTP depths 0–4. Tensor-core prefill supports prefix reuse at MTP depth 0. Build the published tree with the normal [XMake release procedure](../docs/development.md). The patches here are inputs for separate worktrees. The main build uses the ordinary source tree.

## Required and named tool selection

The published server rejects `required` and named `tool_choice` with HTTP 400 and `Unsupported tool_choice. Expected auto, none, or null.` It also rejects strict `auto` and `parallel_tool_calls:false`. These results matched in JSON and SSE. The source patch adds selection and constraint compilation. In `apps/server/text_request.cpp`, `read_chat_tools` still expects a JSON pointer where `chat_tools` now returns `ChatTools`. Resolve that type mismatch, then wire the compiled tool constraint into request admission. No candidate executable was tested.

Apply the [source patch](tool-choice-source.patch) and [test patch](tool-choice-tests.patch) to the same clean base. From the root of the checkout containing this handoff, run the following in the managed Pixi environment. Choose fresh sibling paths for the worktree and build.

```bash
repo=$(pwd -P)
worktree="$repo/../sglang-tool-choice"
build="$repo/../sglang-tool-choice-build"
git worktree add --detach "$worktree" c6b421091d24a7d1d79dc7c99d2df3cddbbbb76b
git -C "$worktree" apply --check "$repo/handoff/tool-choice-source.patch" \
    "$repo/handoff/tool-choice-tests.patch"
git -C "$worktree" apply "$repo/handoff/tool-choice-source.patch" \
    "$repo/handoff/tool-choice-tests.patch"
cd "$worktree"
```

Fix the type mismatch before building. The probe aborts on the parent rejection because `parse_chat_request` throws `server::HttpError`. Include `sglang/server/http.h` in the probe. Handle `HttpError` alongside the existing handler for `std::invalid_argument`. Run XMake from within the worktree. Build the probe and run the portable cases.

```bash
xmake f -P . -o "$build" -p linux -a x86_64 -m release
xmake build -P . -j 8 sglang-server
bash tests/constraints/build_tool_probe.sh "$worktree" "$build" "$build/tool-probe"
python3 -B tests/constraints/tool_mask.py "$build/tool-probe" "$build/tool-mask-portable"
```

For comparison with the saved [reference fixture](tool-choice-fixtures.json.gz), decompress it in the isolated build directory and verify its SHA-256. Run the mask runner with `--reference` after fixing the probe.

```bash
gzip -dc "$repo/handoff/tool-choice-fixtures.json.gz" > "$build/tool-choice-fixtures.json"
(cd "$build" && sha256sum -c <<'EOF'
97eb43d8cd7169ec2e44ace8fc9d141abbf9c3bef2cfb4fb519740913ae7c152  tool-choice-fixtures.json
EOF
) && python3 -B tests/constraints/tool_mask.py "$build/tool-probe" "$build/tool-mask-reference" \
    --reference "$build/tool-choice-fixtures.json"
```

To regenerate the oracle, use SGLang `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`, XGrammar `0.2.7`, and the local Qwen3.8-27B-FP8 checkpoint. Use the checkpoint's tokenizer bytes and both EOS IDs. Replay Qwen XML wrappers, required and named selection, ordered strict fields, explicit reasoning close and implicit tool open, and exact-budget completion. Record the allowed bits at each token and the typed output. Pass the regenerated fixture to the mask runner with `--reference`. Keep the completed-call behavior and real `length` in JSON and SSE. The [pinned selector](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/function_call/function_call_parser.py#L321-L442) specifies the tool grammar entry point.

## Kernel work

All six kernel patches apply independently to `09c762a37f253ce645baf2aa2e3498d862096348`. Use a fresh worktree at that revision for each patch. Repeat native and vLLM comparisons with the fixed [workloads and criteria](../docs/performance-comparison.md). Keep operator, forced-token, and complete HTTP measurements distinct.

- **Paired FP8 gate/up and batched SwiGLU quantization.** The [FFN patch](kernels/ffn-candidate.patch) enables the batched epilogue. The compile-time `SGLANG_EXPERIMENT_FFN_PAIR` macro also enables paired gate/up. W3 measured 141.062 → 143.940 → 149.446 output IDs/s for stock, epilogue, and pair. The pair gained 5.94% over stock and improved all three p95 latencies. The first W4 pass measured 88.145 → 91.466 → 93.172 IDs/s with three measured cohorts per arm. The pair gained 5.70% over stock. The reverse-order runs for the two candidates did not finish. Finish W4 reversal on the same card before deciding on adoption. Forced-token logits and state, SSE IDs, and the tested graph, eager, and MTP paths matched.
- **BF16 GDN projection and grouped LM head.** [GDN Pack1](kernels/gdn-pack1.patch) improved W3 throughput by 4.20% and 4.22% in the two orders and passed each stock-relative gate. W3 exercised the existing width-four LM head route. The [head patch](kernels/head-compose.patch) changed W3 pooled throughput by −0.035%. Both candidates matched the observed full logits and state. W4 was stopped before a complete comparison.
- **Staged long-context decode KV.** The [KV patch](kernels/long-kv.patch) uses `SGLANG_EXPERIMENTAL_LONG_KV=serial` or `overlap`. An unset variable selects stock. State, full logits, mixed-length fallback, graph reuse, and short HTTP checks passed. Complete the full-output same-card W4 timing in both directions before evaluating adoption. W3 is a dispatch control for the long tensor path.
- **FP8 prefill issue order.** [Grouped issue](kernels/issue-grouped.patch) and [grouped issue with prewait](kernels/issue-prewait.patch) are alternative W2 probes. They replace the two production `try_text_fp8_gemm_sm100` overloads with experimental entry points. Restore the production overloads when adapting either patch to a full engine build. The isolated three-arm comparison reached the build stage. Whole-model W1/W2 quality and serving measurements remain open.

K128 prefill ticket batching improved isolated kernels but missed the whole-model throughput gates. The FP16 activation sidecar preserved bits but increased quantization-plus-projection time. The fast small-M MMA projection improved W3/W4 throughput by about 23%/18% but failed whole-model quality. Its FP64 outer-scale follow-up matched the independent reference's top choice on 77/84 rows versus 80/84 for stock and was also declined. The [performance comparison](../docs/performance-comparison.md#experiments-declined-for-adoption) records the K128 and sidecar results.

## Remaining vLLM gap

The fixed same-card comparison measured native/vLLM throughput of 139.186/376.948 output IDs/s on W3 and 86.999/328.267 on W4. FP8 projection and paged GQA were the largest measured decode kernel gaps. The native FFN and GDN improvements are incremental against that baseline. Repeat projections and GQA timing after any accepted change.

In the formal W5 and W6 comparison, all native lanes timed out before headers and vLLM completed both rows. A separate GPU4 stock W5 diagnostic on `6b4f734c9c1a90ce16ff3b61ed9d3f0ad128c811` had 30 of 32 lanes time out on the first read. Two streams started at about 58.6 seconds and yielded 2,863 output IDs matching the golden prefixes before the cohort deadline. No cohort completed. GPU4 stock W6 timed out before headers on all four lanes.

The short W5 GPU4 trace sampled one 1,024-row tile at each prefix position 0, 10,240, and 19,456. Flash attention grew from 3.041 to 45.972 to 85.481 ms while FP8 projections stayed near 39 ms. The sample adds synchronization and covers different lane mixes. W6 instrumentation was canceled before readiness. There is no matched vLLM trace for these samples. Compare the engines at aligned prompt histories and useful causal work before selecting the next long-prefill change.

The abnormal CUPTI sampling loop was fixed by starting its cursor once per completed buffer, retaining the last returned record, and aborting on nonprogress. The bounded collector used three windows, a 4,096-record and 2 MiB cap per window, a fixed eight-buffer pool of 256 KiB each, and a watchdog. Keep these safeguards when rebuilding the diagnostic.
