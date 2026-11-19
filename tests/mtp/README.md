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

Both modes check that the CLI rejects excessive MTP depth and RNG flags and that the width-4, depth-4 server accepts greedy requests while rejecting unsupported sampling options. The full run also checks the HTTP policy on the default and serial depth-4 servers. The raw-ID route requires temperature and seed zero. Chat accepts temperature zero and rejects seed and `top_p`. Text completion rejects temperature and seed options.

The draft seed check uses 129- and 257-token W2 prefixes at steps 0, 1, and 4. These prefixes require the last row of each full prefill tile to use the next tile’s first token as its shifted draft input. The runner checks ordered output and logits against target-only generation and checks draft, acceptance, and round counts against the pinned row-wise results. At step 4 it disconnects after the first verification output for both `silver-key` and the 257-token prefix, waits for request retirement, and submits a continuation with the observed tokens on the same server.

Pass `--reference-generate /path/to/previous/sglang-generate` to compare the seed cases and the `silver-key` route with the previous release generator. For the seed change use a generator built from `e574e54720e0aa9ba08ed95a21e5af1756956310`, which seeds the draft row by row. The comparison checks exact greedy IDs, finish reason, ordered top-64 logits, and MTP counters. Each invocation keeps its commands and outputs under a new evidence directory.

`reference.json.fixture` preserves the pinned reference’s target inputs, proposals, acceptance, correction, bonus, and trimming records. The capital trace also records the shifted draft inputs across prefill and a partial match. The integration runner uses native target-only output for exact comparisons because the reference and native implementations use different numerical paths, including different KV storage formats.
