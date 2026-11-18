# Native inference checkpoint · November 18, 2026

The patches are independent candidates unless an ordering is specified. The common published
base is `98b5b35916b7fcadd36859c30abf069e03def377`. Replays use the managed
`/data1/shared/dongwu.chen/pixi.toml` default environment, XMake release, CUDA 13.4.92,
and `/data0/shared/Qwen3.8-27B-FP8` on a reserved B200.

## Source candidates

| Patch | Fixed base and order | SHA-256 |
| --- | --- | --- |
| `native-batch-serving.patch` | Common base | `c6ccdb91f1f71709c2b21c9f766aad7486881b8508d4e416aee4b8b2a3af1163` |
| `native-batch-fixtures.patch` | Data only, apply in the batch replay tree | `c71f261f8418f4b329ee0e7e6025ddc1717f4a10ac67272923e6cc82d8924ab9` |
| `native-decode-bandwidth.patch` | `20656335c614747c85157146543a99b5c2b0b95c` | `0e291bfa5e513cc839c2a09a623f8877c0a206a22566487823d9c9300ed0a75a` |
| `native-batched-prefill.patch` | Common base | `b9bc48eccf0efdaf4c28314429b0899a122fd3537d4f556a85a67635b22818b4` |
| `optional-chunked-gdn-wip.patch` | After `native-batched-prefill.patch` | `20c546f709aa7ff6b9fd915bce4e7743aad097d73bcc2b86ddd5692724b53710` |
| `comparison-mode.patch` | Common base, `benchmarks/streaming/client.py` only | `3e20e93444f2f080f17136c514140b7a64fea0c4a5d6c592e37250f4bcbc4bd2` |

- **Batch serving.** An archived-source release build passed the repository rules and native
  B200 checks. Two and four concurrent sockets matched standalone token IDs, stop/length
  reasons and SSE usage. Decode reached widths two and four. A fifth socket waited.
  Mixed stops, page crossing, resets during prefill and decode, slot reuse, and six
  serial streaming tests passed. Idle request allocations returned to zero. The mixed-page
  peak was 640,395,796 request bytes and resident memory stayed at 28,944 MiB after reuse.
  The wide, packed fallback and scalar fallback paths each matched 1,370 standalone BF16
  results bitwise. All eight standalone CLI cases passed. The final GPU script stopped
  after these checks on `hon3: command not found`, before the MXFP8 operator and sanitizer
  stages. W3–W6 timing is pending. The fixture patch preserves the exact six input and
  serial-oracle records, including two configured stops, without raw run output.
- **Decode bandwidth.** The packed FP8 GEMV change passed four real native cases,
  projection comparisons and synthetic memcheck. W1/W2 matched all 640 native output IDs.
  B200 diagnostic median GPU step was 11.355 → 10.263 ms, and 5/64 median CLI throughput
  was 78.41 → 87.88 tok/s. Formal W1/W2 serving timing is pending.
- **Batched prefill.** The normalized patch includes the matrix release and FP8 group
  scale-product correction. CPU matrix/quantizer, GDN, MXFP8 append/mask/reuse, and
  full-checkpoint native winner, stop, length and cancellation checks passed. On the
  mixed-tail comparison, eager and batched scores differed by up to 8.8125 despite equal
  winners. The endpoint run reached W1/W2 warmups. Timing and integration with the selected
  graph remain open.
- **Forward-ported GDN WIP.** The overlay preserves the recurrence and call wrapper.
  XMake linkage, prefill selection and validation on this base are pending. The older
  tested replay below also carries distinct BF16-rounded convolution tap arithmetic.
- **Comparison-mode WIP.** This is Dirac's captured client diff. Replaying it produces Git
  blob `259500f27870dab0b215f067b5453b58b2305f1c` from
  `ec08d7e9d9e09ad3275b792b5760fdd11f99b797`. Capture and replay only.
  The comparison mode is untested and has no formal native W1/W2 timing. Use the normal
  tracked `benchmarks/streaming/client.py` for the validated harness.

## Historical chunked GDN replay

Start from published main ancestor `eb50f596d194502e4357282639955efcbe26c52a`.
The archive exclusions below drop paths absent from the historical tree. The first patch
reconstructs the eight files whose contents differ. The result matched historical tree
`c1dd46508ae749513f2211acda2ed300d2a698af` in paths, modes, and file contents.
The dependency patches restore the historical build and call-site wiring and BF16 tap
behavior. The resulting base matched the three saved pre-chunk files byte for byte,
and the full replay matched the original validation source tree. The replay requires
only the published commit and the patches in this checkpoint.

| Order | Patch | SHA-256 |
| --- | --- | --- |
| 1 | `chunked-gdn-dependencies/historical-base.patch` | `0bdf9bce1cb8a7e7c58bcacaefb3e865bc1478723f5122415985a6f9e413cd8c` |
| 2 | `chunked-gdn-dependencies/integration.patch` | `fe3551931d13b578a8cc0fa4c31f6a722586478604237d390f22b35411707a14` |
| 3 | `chunked-gdn-dependencies/text-prefill-ops.patch` | `926723a06bc415de0e0e8bad2a7f7597190aa2ae1a4e4128c4082d75300f5957` |
| 4 | `chunked-gdn-dependencies/fp8-scale-product.patch` | `9afb2b5aa3f96f34e2e9560a059d51fa947b313b376b322e465176192f7b140f` |
| 5 | `chunked-gdn-dependencies/bf16-tap-products.patch` | `ac742b467f0d30834865a04e3b8d397ad01e571bd2d7e9632dd53dd2a8ea6e2c` |
| 6 | `chunked-gdn.patch` | `055a78d3b8b68fdc31b7ccb30599c295ddc76c22eee6779dd7d928293c27c089` |

Run from the repository root in Bash. Use the exact revision for the zero-context patch.

```bash
set -euo pipefail
checkpoint_dir="$PWD/checkpoints/2026-11-18"
revision=eb50f596d194502e4357282639955efcbe26c52a
replay=$(mktemp -d /tmp/sglang-gdn-replay.XXXXXX)
git archive "$revision" -- . \
  ':(exclude)benchmarks/streaming' \
  ':(exclude)tests/decoder/cli_delimiters.py' \
  ':(exclude)tests/decoder/request_check.cpp' \
  ':(exclude)tests/test_streaming_client.py' \
  ':(exclude)tests/test_streaming_dispatch.py' | tar -x -C "$replay"
(
  cd "$replay"
  base_patch="$checkpoint_dir/chunked-gdn-dependencies/historical-base.patch"
  git apply --check --unidiff-zero --whitespace=error-all "$base_patch"
  git apply --unidiff-zero --whitespace=error-all "$base_patch"
  for patch in integration text-prefill-ops fp8-scale-product bf16-tap-products; do
    patch_file="$checkpoint_dir/chunked-gdn-dependencies/$patch.patch"
    git apply --check --whitespace=error-all "$patch_file"
    git apply --whitespace=error-all "$patch_file"
  done
  git apply --check --whitespace=error-all "$checkpoint_dir/chunked-gdn.patch"
  git apply --whitespace=error-all "$checkpoint_dir/chunked-gdn.patch"
  bash tools/build.sh --build-dir "$replay/out" --mxfp8-check
)
```

The original component run matched six FlashInfer layer captures bitwise across 40,108,032
BF16 output values and 4,718,592 FP32 state values, and passed 28 synthetic comparisons.
Both native arms passed the four public CLI cases. The historical cross-engine diagnostic
still forked at W1 step 20 and W2 step 64, with more W2 score violations than its control.
There was no formal W1/W2 timing.

## Selected CUDA graph

The authoritative graph is in `src/decoder.cpp`, `src/decoder_forward.cpp`,
`src/decoder_resources.cpp` and `include/sglang/decoder_cuda.cuh`, with CUDA ops in
`src/{mxfp8_gqa,text_attention_gqa,text_ops}.cu`. The selected implementation matched
the tested graph source byte for byte. The maintained `tests/decoder/request_check.cpp`
also checks callback publication after a failed replay and allocation reuse. The graph
captures 64 decode layers, the LM head and device greedy selection after the eager first
token. It reuses captured storage until growth or error and retains weights through drain.

The published eager build and graph build matched every generated ID, callback, finish
reason, top-64 entry and raw BF16 logit for W1 and W2 (128 and 512 output rows).
The 640 dense rows per build had SHA-256
`a7a19ea8bdfaec50bead1af7a33a8f6c5fa4bec8d96244b0ec1fc4171591ce3f`.
Another 149 debug rows matched bitwise across page growth, reuse, stops, cancellation and
reentry. Both servers passed six streaming tests. The separate graph lifetime run
exercised deferred weight reads and injected synchronization errors.

The 5/8 B200 diagnostic reported median host intertoken latency of 11.352 → 10.798 ms,
TTFT of 55.022 → 54.567 ms, and 56.328 → 61.478 tok/s. Its warm requests had zero graph
instantiations. The first cold decode transition incurred capture and instantiation.
Formal W1/W2 serving throughput and TTFT targets remain open.

## Resume

Set `checkpoint_dir` to this directory, then choose `revision` and `patch` from the tables.
From the repository, create a scratch replay and apply the relevant patches in order.
All series passed strict Git replay. An outer `git diff --check` flags the single-space
prefix on blank unified-diff context lines. Keep those bytes to preserve the recorded hashes.

```bash
replay=$(mktemp -d /tmp/sglang-replay.XXXXXX)
git archive "$revision" | tar -x -C "$replay"
(cd "$replay" && git apply --check --whitespace=error-all "$checkpoint_dir/$patch" &&
  git apply --whitespace=error-all "$checkpoint_dir/$patch")
(cd "$replay" && bash tools/build.sh --build-dir "$replay/out" --mxfp8-check)
```

For a server, configure XMake with `-P "$replay" -o "$replay/server-build" -m release`
in the managed environment, then run `xmake build -P "$replay" sglang-server`.
Use `--max-active 4 --trace-batches` on the batch candidate, replay the fixture prompts
against `/v1/completions`, and compare tokens, usage and finish reasons with the serial
oracle. Run the tracked `tests/test_native_streaming.py --port PORT` with the server in
its default single-active mode.

For graph correctness, build `sglang-generate` from the common eager base and from the
checkpoint commit. With `GPU_UUID` set to a reserved idle B200, run the commands below
for each generator. Compare only output IDs, finish reasons and top logits. The short
CLI command can use `--top-logits 248320` for a full-vocabulary comparison.

```bash
for workload in W1 W2; do
  fixture="benchmarks/streaming/$workload.json"
  jq -r '.prompt_token_ids | join(" ")' "$fixture" > "$replay/$workload.ids"
  CUDA_VISIBLE_DEVICES="$GPU_UUID" "$GENERATOR" --device 0 \
    --prompt-file "$replay/$workload.ids" --max-new-tokens "$(jq -r .output_tokens "$fixture")" \
    --top-logits 64 | jq '{output_token_ids,finish_reason,step_top_logits}'
done
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$GENERATOR" --device 0 \
  --prompt-ids 760,6511,314,9338,369 --max-new-tokens 8 --warmup 2 --repeats 9
```

The CLI reruns generation and the short wall-time diagnostic. Callback-level timing,
fault injection and raw BF16 capture in the original evidence used a separate native
driver.
