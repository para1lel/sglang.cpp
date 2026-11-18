# Native MTP acceptance

Run the live check with release binaries, the pinned Qwen3.8-27B-FP8 checkpoint, and a reserved B200. Select the GPU before starting the runner. This check is opt-in and requires no GPU access from the pre-commit gate.

```sh
python tests/mtp/run_native.py \
  --generate build/bin/sglang-generate \
  --server build/bin/sglang-server \
  --evidence artifacts/mtp-native/run
```

The runner compares explicit MTP steps 0 through 4 with the same binary’s default target-only path. It checks greedy IDs, ordered top-64 BF16 logits, stop and length trimming, the 128-token KV page boundary, MTP counters, and serial HTTP token streams. At step 4 it disconnects after the first verification output, waits for request retirement, and submits a continuation with the observed tokens on the same server.

Pass `--reference-generate /path/to/previous/sglang-generate` to also compare the `silver-key` route with the per-token native verifier at steps 1 through 4. This checks exact draft, acceptance, and round counts as well as ordered outputs and logits. Each invocation keeps its commands and outputs under a new evidence directory.

`reference.json.fixture` preserves the pinned reference’s target inputs, proposals, acceptance, correction, bonus, and trimming records. The capital trace also records the shifted draft inputs across prefill and a partial match. The integration runner uses native target-only output for exact comparisons because the reference and native implementations use different numerical paths, including different KV storage formats.
