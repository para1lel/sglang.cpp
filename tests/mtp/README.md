# Native MTP acceptance

Run the live check with release binaries, the pinned Qwen3.8-27B-FP8 checkpoint, and a reserved B200. Select the GPU before starting the runner. This check is opt-in and requires no GPU access from the pre-commit gate.

```sh
python tests/mtp/run_native.py \
  --generate build/bin/sglang-generate \
  --server build/bin/sglang-server \
  --evidence artifacts/mtp-native/run
```

The runner uses explicit `--mtp-steps 0` as its target-only control and compares
the default generator run with explicit K4. Native generation checks greedy IDs, ordered top-64 BF16 logits,
stop and length trimming, the 128-token KV page boundary, and MTP counters.
Single-request HTTP compares token streams. Batch bursts check exact token events,
finish reasons, usage, and joint target rows at widths 2 through 4 and depths 1
through 4.

Add `--batch-spot-check` for a short qualification run. It compares generator JSON, ordered top-64 logits, and MTP counters at depths 0 through 4 on a 127-token prefix. It checks joint HTTP verifier rows at width 2 for all four depths and at width 4 for depth 4. Each generator run records stdout JSON and stderr separately, and each HTTP run records its server trace.

Both modes check excessive MTP depth and CLI sampling validation, then check the width-4,
depth-4 server's sampling policy. The full run also checks the default server and depth-4
serving with `--max-active 1`. The frozen benchmark ID route still requires temperature and seed zero. Native chat
and text accept per-request sampling controls, including on a positive-depth server.

The draft seed check uses fixed 129- and 257-token prefill prefixes at steps 0, 1, and 4. These prefixes require the last row of each full prefill tile to use the next tile’s first token as its shifted draft input. The runner checks ordered output and logits against target-only generation and checks draft, acceptance, and round counts against the pinned row-wise results. At step 4 it disconnects after the first verification output for both `silver-key` and the 257-token prefix, waits for request retirement, and submits a continuation with the observed tokens on the same server.

Pass `--reference-generate /path/to/previous/sglang-generate` to compare the seed cases and the `silver-key` route with the previous release generator. For the seed change use a generator built from `e574e54720e0aa9ba08ed95a21e5af1756956310`, which seeds the draft row by row. The comparison checks exact greedy IDs, finish reason, ordered top-64 logits, and MTP counters. Each invocation keeps its commands and outputs under a new evidence directory.

`reference.json.fixture` preserves the pinned reference’s target inputs, proposals, acceptance, correction, bonus, and trimming records. The capital trace also records the shifted draft inputs across prefill and a partial match. The integration runner uses native target-only output for exact comparisons because the reference and native implementations use different numerical paths, including different KV storage formats.

## Sampled MTP

Run the finite-vocabulary oracle on the CPU and the HTTP regression with a release
server on a reserved B200.

```sh
python tests/mtp/sampled_law.py
python tests/mtp/sampled_http.py \
  --server build/bin/sglang-server \
  --evidence artifacts/mtp-sampled/run
```

The oracle compares complete sequence probabilities at depths 1 through 4 with direct
target sampling under five proposal laws. It also checks that reusing an output
ordinal after a rejected suffix preserves the target-only draw sequence.

The HTTP check compares seeded JSON and SSE with depth-0 controls on the same
runtime. At capacity one it covers depths 0 through 4 and eager depths 0 and 4. At
capacity four it covers graph serving at depths 0 through 4 and eager serving
at depths 0, 1, and 4. The sampling and grammar cases check the resulting token
sequences against depth-0 output.
The check covers long prompts, concurrent seeds, grammar masks, sampling filters,
EOS, stop at the output limit, and replay after a TCP reset.
Server traces verify graph replay at depth 0, joint verification at positive
depths, and replay of discarded verifier state.

Decoded RGB8 chat uses the configured depth. The check compares media output with the
depth-0 control, verifies draft activity at positive depths, and checks rejection of a
per-request depth. The runner saves requests, transcripts, effective caps, startup results, and server traces.

Each invocation checks startup options with CUDA hidden. Select the live checks as follows.

- The example above runs the depth-comparison matrix with prefix reuse disabled.
- Add `--lmcache-socket "$LMCACHE_SOCKET"` to run that matrix and both cache checks. Use an isolated, ready [daemon on the same GPU](../../docs/serving/lmcache.md#start-the-daemon).
- Add both `--lmcache-socket "$LMCACHE_SOCKET"` and `--eager-cache-only` to run just the eager K4 live check.

The width-two K0 cache check covers active and queued cancellation, HTTP 409 on busy reset, and seeded output after idle reset. The eager K4 check requires an 8,192-token warm hit, identical seeded output, draft activity on both arms, and zero graph replays.
