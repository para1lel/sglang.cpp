# Native source checkpoint · November 19, 2026

This checkpoint preserves unmerged native implementations for Qwen3.8-27B-FP8 on one B200. The publication anchor is `723eb810d18336baf5d5e21bb5423119cf6e655a`. Each series starts from its own published base in `manifest.json`. Reconstruct each series in isolation and apply its patches in the listed order. Device results describe historical prototype runs. Production adoption remains open.

## Reconstruct and build

Use Git, `tar`, and Python 3.11 or newer. Run from a clone containing the publication anchor and its history. Set `bundle` to this directory, and set `CHECKPOINT` to the independently obtained checkpoint at revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`.

```sh
repo=$PWD
bundle="$repo/checkpoints/2026-11-19"
(cd "$bundle" && sha256sum --strict -c SHA256SUMS)
python3 "$bundle/verify.py" --repo "$repo"

work=$(mktemp -d)
python3 "$bundle/verify.py" --repo "$repo" --output "$work/trees" chat
```

The verifier exports each Git base and applies its patches. It checks SHA-256 values and inventories for `apps/`, `include/`, `src/`, `xmake.lua`, and the remaining patched paths. It reads Git objects without using the checkout or index. Use `--output` to keep the reconstructed trees in a new directory. `manifest.json` lists patch paths and digests. `sources/` lists reconstructed source files and digests. All 14 series replayed from a clone without a checkout. All listed release targets built with CUDA devices hidden.

The independent build review verified loader dependencies for all 19 targets. It used the CUDA toolkit driver stub for the vision encoder and decoder smoke builds and observed no device opens.

Build in the repository’s default Pixi environment with XMake 3.1.2 or newer, a CUDA 13.4 toolchain targeting SM100, and LLVM 23 for series that link it. After activating the environment:

```sh
cd "$work/trees/chat"
xmake f -P . -m release -o "$work/build-chat" -y
xmake -P . -j 4 sglang-server
```

Build the targets listed for your series in the manifest. For model runs, expose the leased B200, configure the host’s CUDA driver compatibility library as needed, and give `sglang-server` or `sglang-generate` the path with `--checkpoint "$CHECKPOINT"`.

## Historical checks and limits

| Series | Entry point | Historical evidence and limits |
| --- | --- | --- |
| `raw-text` | `/v1/completions` with `model: "Qwen3.8-27B-FP8"` and a UTF-8 string `prompt` | 40,794 tokenizer differential cases, 86 fixtures and 93,610 NFC invariants passed. Real B200 serving matched direct-ID responses in default, two-lane and two-step MTP modes, with raw Unicode exercised in both response modes. Greedy text completion with a 1 MiB prompt limit. |
| `chat` | Apply `raw-text` first, then use `/v1/chat/completions` | 136 exact HF prompt-byte and ID matches, 135 native HTTP mock token-input matches and 156 parent response matches. GPU inference through this route was outside the check. Historical tool calls render, while forced tool calls and external media are unsupported. |
| `mtp-one-sync` | `sglang-generate --mtp-steps K --mtp-verify parallel` | Reconstructs the tested candidate-04 foundation before applying the original parallel verifier. Passed 24 state trials and 96 generation trials. Isolated verifier speedup was 1.42–1.92× over serial on the reported short cases. |
| `mtp-state-replay` | `sglang-generate --mtp-steps K --mtp-verify parallel` or `--mtp-verify replay` | Includes the parallel path and its GQA and workspace prerequisites. Passed 90 short and 90 long exact state/logit trials plus two partial commits across 5,120 tokens. Replay used 151.93–159.60 MiB of incremental state allocation versus 295.05–737.63 MiB for checkpoints. The timings cover 33 paired cases. |
| `mtp-continuous-batch` | `sglang-server --max-active 2` or `4` with `--mtp-steps 0` through `4` | Independent variant on the publication anchor. The short HTTP comparisons matched with widths 2 and 4 across all five MTP depths. Throughput varied by depth and scheduling. |
| `multi-lane-graphs` | `sglang-server --max-active 4 --batch-cuda-graphs` | Graph decode widths 2–4 matched all 39 greedy IDs and their top-16 scores. CUPTI recorded eight replays. Memcheck, slot reuse, and an injected launch error were checked. Prefill, mixed steps and width-one batch steps use eager execution. |
| `wide-batch` | `sglang-server --max-active 8`, `16` or `32` | Actual HTTP widths reached 8, 16 and 32. All BF16 vocabulary scores matched over 1,073 lane steps. Repeated-KV 32K capsules measured 25.62 and 44.72 output IDs/s at 16 and 32 lanes. Full W4/W5 serving runs remain open. |
| `sampling` | CLI `--temperature`, `--top-k`, `--top-p`, `--seed` or corresponding completion fields | CPU distribution, seeded repeatability, serial/batch selection, greedy fallback and HTTP validation checked. Stochastic selection transfers BF16 logits to the host and uses a native RNG. Positive-temperature sampling with MTP is rejected. |
| `gqa-flash` | `sglang-generate --prefill mxfp8-flash` or `--prefill exact-tiles` | Standalone compensated MXFP8 reader and single-request integration. Passed the FP64 error budget at real 128, 2,048 and 8,192-token layer-3 shapes. Synthetic long cases had budget misses and model token differences were observed. The exact path stays the default. |
| `gqa-long-decode` | `sglang-server` on the publication anchor | Guarded split/merge GQA with exact fallback. Fast scratch is bounded to 512–65,536 tokens of allocated capacity with at most 32 partitions. Clean source build and W2 CUDA profiling were recorded. Full long-context serving parity remains unverified. |
| `fast-w3` | `sglang-server --max-active 4 --fast-prefill-rows 1024 --fast-gdn` | Recovered exact prerequisite, then the measured `combined-02` source. Two warmups and five measured W3 cohorts each matched 4,096 vLLM IDs. It measured 25.955 versus 379.524 output IDs/s, 21.367 s p95 TTFT and 133.717 ms p95 TPOT, missing the adoption gates. |
| `w3-bounded-decode` | Add `--flash-prefill --split-decode` to the fast W3 build | Includes both `fast-w3` patches and the measured Flash/bounded decode overlay. The 1- and 16-output checks matched the reference. Five full W3 measured cohorts matched 4,096 IDs each at 94.58–105.80 output IDs/s, below the 383.32 output IDs/s adoption threshold. |
| `chunked-prefix-cache` | `sglang-server --max-active 4 --prefix-cache-pages 256 --prefix-cache-checkpoints 5` | Uses the published exact chunked-prefill tree. Warm W3 matched all 4,096 fresh IDs, with 25,600 cached and 7,168 computed prefill rows. State, active-borrower eviction and cancellation probes passed. Timing was affected by interference. The final patch adjusts cache-disabled telemetry. The cache requires MTP off. |
| `vision` | [Regenerate the fixtures and build the offline encoder and decoder](vision/README.md) | Eight CPU cases matched 2,347,008 patch elements per dtype exactly. Historical B200 merged worst-row error was 19.98% for image and 7.87% for video, above the 5% envelope. This slice takes prepared tensors for offline inference. |

Historical W3 results use the published direct-ID benchmark fixtures and vLLM `81a8cb063e6247fe254d7caf7e98a6691d71022a`. See `benchmarks/concurrent/` in the clone for the workload and reference. The W3 patches use fresh Git diff headers. `w3-combined-02` excludes three inactive `.orig` backups. The replayed build inputs match the frozen measured sources.

The tokenizer checks the checkpoint’s `tokenizer.json` SHA-256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`. Chat additionally checks `tokenizer_config.json` (`207c03c41a06a9c3b0cc527d8a56115159c1eac6b59a4ffae85b646e773d074e`) and `chat_template.jinja` (`af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375`). The raw-text patch retains the Unicode data license at `tools/unicode/LICENSE`.
