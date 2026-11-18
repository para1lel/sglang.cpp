# Batch CUDA Graph checks

Use the managed Pixi `default` environment and reserve an idle B200 for the entire
run. Set `GPU_UUID` to its device UUID while holding the shared UUID lock. Use the
pinned Qwen3.8-27B-FP8 checkpoint.

From the repository root, build the normal release server and the two opt-in checks.

```bash
out="$PWD/build/batch-graph-check"
checkpoint=/data0/shared/Qwen3.8-27B-FP8
xmake f -P "$PWD" -o "$out" -p linux -a x86_64 -m release \
    --mxfp8-gqa-check=y --batch-graph-lanes=y
xmake build -P "$PWD" sglang-server
xmake build -P "$PWD" mxfp8-gqa-check
xmake build -P "$PWD" batch-graph-lanes
GPU_UUID="${GPU_UUID:?Set the reserved B200 UUID}" \
    bash tests/batch/run_batch_graph.sh "$out" "$checkpoint" "$out/evidence"
```

The runner uses fresh processes for eager and graph under both exact and tensor-core
prefill. It leaves the command inputs, HTTP SSE, full BF16 scores, scheduler traces,
and CUPTI runtime/device records in the evidence directory.

- The CUDA operator probe rebinds distinct MXFP8 cache and page-table addresses,
  crosses positions 127 and 128, and checks KV bytes, inactive storage, and BF16
  attention. It rebinds separate GDN history and recurrence banks and compares
  output, BF16 history, and FP32 state against ordinary decode.
- The model probe changes sparse active sets at widths two, three, and four.
  It checks cancellation, zero-budget retirement, positive slot reuse, greedy IDs,
  and complete BF16 scores against eager. Each graph mask must equal the slots
  with a decode result. Within this short scenario, once a width has replayed,
  each subsequent decode-only step at that width must replay. CUPTI must record
  one actual CUDA graph per width through capture, launch, and destruction.
- Fault probes inject unsupported capture, failed instantiation, replay rejection,
  scratch and instantiation allocation failures, and an uncertain post-launch
  synchronization. They check eager fallback, graph cleanup, continued replay at
  other widths, and fail-closed behavior after uncertain completion.
- The public HTTP probe sends distinct prompts with all three widths and checks
  independent EOS and explicit stop, cancellation, overlapping replacement, and
  recovery. Both server routes must return identical SSE and release request state.

Expect `Batch CUDA Graph operator, lane, fault and HTTP checks passed`. The
[evidence checker](check_graph_evidence.py#L1)<!-- prefix: "\"\"\"Check rea" -->
can be run again against an existing evidence directory.

Run the slot-mask regression without a checkpoint or GPU.

```bash
python3 -IB tests/batch/check_graph_evidence.py --check-slot-mask
```
