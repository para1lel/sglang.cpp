# Native MTP acceptance

Run the live check with release binaries, the pinned Qwen3.8-27B-FP8 checkpoint, and a reserved B200. Select the GPU before starting the runner. This check is opt-in and requires no GPU access from the pre-commit gate.

```sh
python tests/mtp/run_native.py \
  --generate build/bin/sglang-generate \
  --server build/bin/sglang-server \
  --evidence artifacts/mtp-native/run
```

The runner compares explicit MTP steps 0 through 4 with the same binary’s default target-only path. It checks greedy IDs, ordered top-64 BF16 logits, stop and length trimming, the 128-token KV page boundary, MTP counters, and serial HTTP token streams. Batch bursts check exact token events, finish reasons, usage, and joint target rows at widths 2 through 4 and depths 1 through 4.

Add `--batch-spot-check` for a short qualification run. It compares generator JSON, ordered top-64 logits, and MTP counters at depths 0 through 4 on a 127-token prefix. It checks joint HTTP verifier rows at width 2 for all four depths and at width 4 for depth 4. Each generator run records stdout JSON and stderr separately, and each HTTP run records its server trace.

Both modes check excessive MTP depth and CLI sampling validation, then check the width-4,
depth-4 server's sampling policy. The full run also checks the default and serial depth-4
servers. The frozen benchmark ID route still requires temperature and seed zero. Native chat
and text accept per-request sampling controls, including on a positive-depth server.

The draft seed check uses 129- and 257-token W2 prefixes at steps 0, 1, and 4. These prefixes require the last row of each full prefill tile to use the next tile’s first token as its shifted draft input. The runner checks ordered output and logits against target-only generation and checks draft, acceptance, and round counts against the pinned row-wise results. At step 4 it disconnects after the first verification output for both `silver-key` and the 257-token prefix, waits for request retirement, and submits a continuation with the observed tokens on the same server.

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

The HTTP check compares seeded JSON and SSE with exact depth-0 controls. It
covers serial exact prefill at depths 0 through 4, serial eager at depth 0, and
width-four graph and eager serving with both prefill policies at depths 0, 1,
and 4. It also checks width-four exact graph serving at depths 2 and 3.
Positive-depth text uses exact prefill under either policy. Masked requests
use the configured policy at depth 0 and retain their own control.
The width-four cases include a long prompt, concurrent seeds, grammar masks,
sampling filters, EOS, stop at the output limit, and replay after a TCP reset.
Server traces verify graph replay at depth 0, joint verification at positive
depths, and replay of discarded verifier state.

Serial and batched decoded RGB8 chat use depth 0 even when the server is
configured for text speculation. The check compares media output with the
depth-0 control and verifies rejection of a per-request depth. The width-two
sampled depth-0 case checks active and queued cancellation, rejection of a busy
prefix-cache reset, and unchanged seeded output after an idle reset. The runner
saves requests, transcripts, effective caps, startup results, and server traces.
The startup matrix checks positive-depth admission through width 32 and
preserves the prefix-cache, eager, and graph restrictions.
