# Candidate B quality against pinned vLLM

Independent fixed-history quality experiment, November 21, 2026.

The 352 width-four decode rows show a mean Candidate B change of +0.000228 nats in KL to pinned vLLM and +0.014366 nats/token in teacher NLL. Both paired 95% intervals include zero. All 32 native BF16 logit rows at prefill are byte-identical across the builds, including the nine prompts at 4,096–32,768 tokens. Keep case 2, decode step 2, as the first replay target for further drift investigation. It had the largest per-row increase in KL to vLLM.

## Prior evidence

- The native drift report, `artifacts/resume-drift/report.md`, compared Candidate B with HEAD on 32 prompts and 352 width-four decode rows. The 32 native BF16 logit rows at prefill were byte-identical. Decode had 332/352 greedy agreements, mean `KL(HEAD || B)` of 0.010905, and teacher NLL change of +0.014366 nats/token with a 95% interval of [−0.003239, +0.033153].
- The native `exact` run and FP64 operator replays are numerical controls. The existing pinned-vLLM decoder fixtures retain 64 scores per row on separate histories. The W4 serving artifacts retain generated IDs. These records cannot supply full-vocabulary KL on the width-four histories.
- The independent long-prefill review, `artifacts/resume-merge-accept/handoff.md`, checks the experimental SM100 split merge on captured W4 tiles. The separate author report, `artifacts/resume-merge-accept/report.md`, records full native logits for one W4 lane using native Flash and ordered attention as controls. This experiment uses the default native attention route.

## Inputs and settings

Candidate B is tree `e459be98a510d3235c81c94ef5210ef76a622e75`. All 127 files under `apps/`, `include/`, `src/` and `xmake.lua` match the reconstructed Release source. The native probe source receipt covers the 112 build and library files. The archived HEAD capture used the baseline fixture built from `d6e6eb41cb4ac96ed9f992aee908ac2ea850dca4`. Those inference sources also match `44486cc0388c5dd315706888a559598bb1e9ae26` and `aa4ec04b63c381cf11fdb4a3801fb925bacb5135`. The intake HEAD `dc1adecded6a8af8658d3410307144892c680b89` differs from `aa4ec04b` only in batch tests and their documentation. See `artifacts/test-accept-a/report.md` and `artifacts/coding-fix-a/build-quality-probes.log` for the fixture and link records.

The checkpoint is `/data0/shared/Qwen3.8-27B-FP8`. The experiment reuses all 32 cases from `artifacts/test-accept-a/corpus.tsv` in their original order. Prompts cover 128–32,768 tokens across fiction, technical text and multilingual text, with plain-text and transcription-style chat inputs. Nine prompts have at least 4,096 tokens, including seven with at least 8,192. The native probe's corpus reader regenerates the prompt IDs and next 12 source-text IDs. The input digest includes the manifest, source texts and tokenizer.

Both native captures use Release builds, tensor-core FP8 prefill and enabled decode graphs. The probe prefills each lane separately and forces the teacher token at commit. It retains all 248,320 BF16 logits per row. Step 0 is the distribution at the last prompt position. Steps 1–11 use the actual committed teacher prefix in width-four decode. `artifacts/resume-drift/probe.sh` records the archived Candidate B capture command. The reproduction below reuses both native captures.

The reference is vLLM `81a8cb063e6247fe254d7caf7e98a6691d71022a` in `/data1/shared/dongwu.chen/environments/vllm-81a8cb063e62`, with PyTorch 2.14.0+cu132 and FlashInfer 0.7.4. It uses physical B200 GPU 3 (`GPU-ceeff2f7-3f39-50b4-8d29-576159781b9d`), driver `590.48.01`, eager execution, seed 0, BF16 model dtype, FP8 checkpoint weights, and tensor parallelism of one. It uses FLASHINFER with BF16 queries and 16 GiB of per-tensor `fp8_e4m3` KV cache with unit K and V scales. Native uses its own MXFP8 KV cache and kernels.

The reference disables prefix caching, speculative decoding and asynchronous scheduling. It submits the same four cases together, with a batch token budget of 65,536 and a sequence limit of 32,780. Built-in trace replay commits the 12 teacher IDs. The capture copies the full raw vocabulary tensor before vLLM sampling and records an FP32 representation. It verifies the selected input position, input ID and final output IDs. The analysis joins all three arms by case and step and checks the target, prompt length and source label.

## Metrics

The analysis normalizes the full logits with FP64 log-softmax. KL is `KL(vLLM || native)` in nats. Teacher NLL is the negative log probability of the next source-text ID, in nats/token. Deltas are Candidate B minus HEAD. The means weight each prompt equally within a group. Paired 95% percentile intervals use 10,000 prompt-level bootstrap draws with seed `20261121`. Repeated source texts and the small long-context subset limit generalization. The run scores fixed histories and only one distribution at the end of each prefill.

## Results

All 384 rows matched the three-arm input manifest. Each of the eight vLLM cohorts started with four prefills and then executed 11 pure width-four decode steps. Both native builds report 348 graph-backed decode rows out of 352.

| Phase and prompt length | Rows | HEAD KL | B KL | Δ KL [95% interval] | Δ NLL [95% interval] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill, all | 32 | 0.007734 | 0.007734 | 0 [0, 0] | 0 [0, 0] |
| Prefill, ≥4,096 | 9 | 0.007518 | 0.007518 | 0 [0, 0] | 0 [0, 0] |
| Prefill, ≥8,192 | 7 | 0.009656 | 0.009656 | 0 [0, 0] | 0 [0, 0] |
| Decode, all | 352 | 0.019699 | 0.019927 | +0.000228 [−0.002376, +0.003167] | +0.014366 [−0.003159, +0.033216] |
| Decode, ≥4,096 | 99 | 0.014719 | 0.014555 | −0.000164 [−0.002296, +0.001589] | +0.016785 [−0.021986, +0.069142] |
| Decode, ≥8,192 | 77 | 0.017442 | 0.017023 | −0.000420 [−0.002983, +0.001864] | +0.013517 [−0.031893, +0.081639] |

| Phase and prompt length | vLLM NLL | HEAD NLL | B NLL | HEAD and B top-1 agreements with vLLM |
| --- | ---: | ---: | ---: | ---: |
| Prefill, all | 10.450406 | 10.304335 | 10.304335 | 31/32, 31/32 |
| Prefill, ≥4,096 | 11.582352 | 11.385097 | 11.385097 | 9/9, 9/9 |
| Prefill, ≥8,192 | 11.846925 | 11.843329 | 11.843329 | 7/7, 7/7 |
| Decode, all | 1.857584 | 1.828588 | 1.842954 | 331/352, 332/352 |
| Decode, ≥4,096 | 2.087654 | 1.987738 | 2.004524 | 94/99, 92/99 |
| Decode, ≥8,192 | 2.270916 | 2.148528 | 2.162046 | 73/77, 71/77 |

At decode step 2 of case 2 (`2600.txt:text`, 2,048 prompt tokens), KL rose from 0.093618 to 0.318272 nats. Case 19 (`1342.txt:text`, 16,384 prompt tokens) had the largest KL increase among the long cases, from 0.090077 to 0.221613 nats at step 2. The mean decode differences and their intervals summarize this corpus. Individual positions can differ more, and the seven ≥8,192-token prompts leave substantial uncertainty in the mean NLL change.

## Reproduction

Run these commands from the repository root. The captured run used `reference-v2` and `analysis-v2` under `artifacts/resume-quality-ref`. The vLLM runner takes the GPU 3 lock and checks occupancy before starting. Exit code 75 indicates the lock or occupancy check prevented the run.

```bash
(
  set -euo pipefail
  out=artifacts/resume-quality-ref
  bash "$out/inputs.sh"
  sha256sum -c "$out/input-sha256.txt"
  bash "$out/run.sh" reference-replay 65536
  OPENBLAS_NUM_THREADS=1 /data1/shared/dongwu.chen/.pixi/envs/default/bin/python \
    -B "$out/compare.py" "$out/inputs.tsv" \
    artifacts/coding-fix-a/quality-baseline-tensor-graphs \
    artifacts/resume-drift/quality-release \
    "$out/reference-replay" "$out/analysis-replay"
)
```

The private directory holds `reference-v2.command.txt`, `reference-v2.log`, `reference-v2.gpu.csv`, `reference-v2.compute-before.csv`, `native-source-mapping.txt`, `input-sha256.txt` and `result-sha256.txt`. `reference-v2/` holds `config.json`, `rows.tsv`, `logits.f32` and `completed.json`. `analysis-v2/` holds per-row `rows.csv` and the grouped `summary.json`.
