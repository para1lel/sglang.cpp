# November 19, 2026 performance source checkpoint

This checkpoint preserves eight measured source states for Qwen3.8-27B-FP8 on one NVIDIA B200. Each state replays independently from history at `e574e54720e0aa9ba08ed95a21e5af1756956310`. The W3 and vision chains reuse exact patch blobs from `fa46857586102c97a3f7483e1452c9437728e569:checkpoints/2026-11-19`.

| Series | Preserved source |
| --- | --- |
| `serial-vector` | Fast-native serial graph, private pool threshold, paired BF16 projections and vector KV loads |
| `serial-quant` | `serial-vector` plus warp activation quantization |
| `serial-cta` | Separate integration on the published request lease and private pool, with paired BF16, vector KV, warp quantization and CTA GQA prefill |
| `w3-flash8192` | Prior bounded W3 Flash chain plus the fused 8,192-row prefill donor from `fast-w3-attention` |
| `w3-fused` | `w3-flash8192` plus layer-family projection fusion on the native weight layout |
| `wide-fast` | Native wide projections, fast prefill, request reservations and the graph binding used by the measured service |
| `wide-gqa` | `wide-fast` plus guarded split/merge GQA, ordered fallback and graph binding for the split path |
| `vision-rounding` | Measured patch bias, BF16 position interpolation and online-softmax rounding, with the corresponding encoder and decoder replay inputs |

## Replay and build

Set `REPO_URL` to the repository URL. Set `BUNDLE` to this directory in the published checkout or to a separate copy. Set `SEAL` to the SHA-256 of `SHA256SUMS` from the trusted publication handoff.

Use a full clone that contains the pinned history.

Run the replay on Linux x86-64 with Python 3.9 or newer, Git, GNU tar and GNU coreutils.

The verifier checks the seal, the complete bundle file set and every listed digest before it loads the frozen ordered manifest. It reads Git objects and verifies the base trees. It checks the hashes and paths of the patches. It compares the full inventory of `apps/`, `include/`, `src/` and `xmake.lua` against the frozen release input, including Git executable modes. Vision also verifies `tools/vision-language-check.cu` and `vision/encoder/`. The two W3 inventories omit three named inactive `.orig` files. The requested output appears together with the verification records after every selected series passes.

```bash
work=$(mktemp -d)
git clone --no-checkout "$REPO_URL" "$work/repo"
python3 -B "$BUNDLE/verify.py" --seal "$SEAL" --repo "$work/repo" --output "$work/trees"
```

Append one or more series names to verify a subset. Build with XMake 3.1.2 or newer, GCC/G++ with C++20 support, LLVM 23 development headers and libraries, and a CUDA Toolkit with SM100 support. Activate a Conda or Pixi environment with these paths under `CONDA_PREFIX`:

- `bin/nvcc`
- `targets/x86_64-linux/include` and `targets/x86_64-linux/lib` for CUDA headers and libraries
- `include/llvm` and `lib/libLLVM-23.so` for LLVM

Set `CC` and `CXX` explicitly to the compatible `gcc` and `g++` executables. The recorded toolchain used CUDA 13.4.92 and GCC 15.3.0. To run the executables, retain the toolchain's runtime libraries on the loader search path. The recorded environment supplies `libcudart.so.13` from the CUDA target library directory and `libLLVM.so.23.1` from the prefix library directory. GPU inference also requires the NVIDIA driver or compatible driver libraries.

The helper verifies the replay receipt and source tree before building from a separate copy. It hides GPUs and writes the product checksums after every target in the arm builds successfully.

```bash
(
    set -e
    for arm in serial-vector serial-quant serial-cta w3-flash8192 w3-fused wide-fast wide-gqa vision-rounding; do
        bash "$BUNDLE/build-release.sh" "$SEAL" "$work/trees/$arm" "$work/build-$arm"
    done
)
```

The vision build uses `vision/checkpoint_manifest.inc`, the 333-entry C++ tensor-offset table whose digest the earlier checkpoint records. The earlier checkpoint provides the generator.

To audit or regenerate the table, set `CHECKPOINT` to the pinned model directory. Set `VLLM_ENV` to the pinned vLLM environment in the earlier checkpoint's `vision/README.md`. Set `SGLANG_REPO` to a clone that contains the pinned SGLang revision.

```bash
revision=f04d886bf79aa3d6237cbb12a3311838eb9f9bd8
git -C "$work/repo" checkout --detach e574e54720e0aa9ba08ed95a21e5af1756956310
git -C "$SGLANG_REPO" archive --format=tar.gz --prefix="sglang-$revision/" \
    "$revision" > "$work/sglang.tar.gz"
bash "$work/repo/checkpoints/2026-11-19/vision/replay-preprocess.sh" \
    "$work/repo" "$CHECKPOINT" "$VLLM_ENV" "$work/sglang.tar.gz" "$work/vision-inputs"
cmp "$BUNDLE/vision/checkpoint_manifest.inc" "$work/vision-inputs/manifest/checkpoint_manifest.inc"
```

`build-receipt.json` preserves the original fresh-clone CPU replay and release build receipt. Its source inventories, patches and vision build input retain their exact bytes. The measured and rebuilt binary hashes identify their respective binaries. Compiling uses this bundle, Git history and the build toolchain. Running the server or regenerating the vision manifest requires the pinned checkpoint and the corresponding runtime dependencies.

## Publication format

This revision reformats the independently replayed source bundle with Git tree `8b2daab3d41895b7a6aad8c7d87a245ab28d59ed`. Obtain this revision's seal from its own publication handoff. The earlier bundle's `SHA256SUMS` has SHA-256 `7ecb0634ddebb4e00998a90b3784b9855d846da8d57032a00da9542eb01f0c7e`.

The ordered manifest has the same parsed values and replay order. Eight prose fields in the receipt and results now use arrays of string fragments. Concatenate each array with an empty separator to recover the original field, including spaces. In the receipt these fields are `clone` and `build_notes`. In the results they are `serial.protocol`, `w3.protocol`, `wide_gqa.synthetic_operator.controls`, `wide_gqa.http_protocol`, `vision.coverage` and `vision.provenance.donors`. Numeric measurements and digest strings retain their values.

The original `build-receipt.json` has SHA-256 `20751fc10bf1ad5c251dfcc3599cf5a2dd26814aba08c81306ba0d59a03cd10b`. The original `results.json` for that independently replayed bundle has SHA-256 `2c77fa4d93e9736a90943269ee8a885641d0ca3c47256ec910f5849b3872d3e9`. The vision evidence references `vision-parity/serving-stage-comparison.json` with `native_prior_vs_sglang` and `vision-parity/online-stage-comparison.json` with `native_online_vs_sglang`. Their individual SHA-256 values are in `results.json`.

## Historical measurements

`results.json` records the measured protocols, key metrics, constraints and evidence digests. All GPU results predate this package's CPU validation.

- **Serial W1/W2.** Vector and warp quantization reached 130.40/121.41 and 130.84/121.73 output IDs/s against 115.32/114.03 for the GPU1 vLLM pair. The CTA integration reached 130.02/121.18 against 116.34/115.09 on GPU4. Vector and quantization agree bitwise with the preceding fast-native path on 158,924,800 BF16 logit words and all 640 teacher-forced winners. CTA changes the first W2 token relative to warp prefill. Quant W2 median TTFT was 358.56 ms versus 79.80 ms for its pin, and CTA missed the p95 TTFT gate on both workloads and the W1 p95 end-to-end gate.
- **W3.** Layer fusion reached 144.84 output IDs/s on four lanes with 8,192 input and 1,024 output IDs per lane, from 131.63 for the preceding native donor. All 28,672 streamed IDs, including warm-up, agreed with the pin. The pinned vLLM reference reached 379.52 output IDs/s. The fused native arm missed all four performance gates.
- **Wide capacity.** The native service admitted the full W4/W5 reservations at widths 16 and 32. Both cohorts were cancelled after 2,048 total prefill rows and emitted zero IDs. W5 peak sampled NVML memory was 71,786 MiB. A separate real 32K lane produced eight IDs after 64.746 s TTFT. Exact eager and graph output agreement used short requests at four widths.
- **Wide split GQA.** Synthetic 32K operator timings improved at widths 16 and 32. Shortened real 8,192+16 HTTP cohorts improved steady full-width decode and request TPOT, with eager control versus split graph. They matched 214/256 and 379/512 output IDs. On W5 the retirement step raised all-step p95 from 1,067.840 to 1,951.210 ms. Real 32K HTTP coverage is one lane. Full-length, full-width W4/W5 completion and adoption gates remain unmeasured.
- **Vision.** Worst projected-row relative RMSE against the pinned SGLang tower fell from 21.00% to 5.11% on image and 6.70% to 3.32% on video. Image remains above the 5% criterion. Teacher-forced video logit relative RMSE against the same decoder with SGLang rows increased from 18.92% to 19.65%.

## Run the published HTTP protocols

Check out `e574e54720e0aa9ba08ed95a21e5af1756956310` in the disposable clone for the benchmark clients. Expose one B200. Start one server at a time, with `ARM` set to the chosen replay series. The serial client uses the pinned W1/W2 fixture bytes. Run the pinned vLLM comparator separately using `benchmarks/streaming/README.md` at the anchor.

```bash
"$work/build-$ARM/engine/bin/sglang-server" --checkpoint "$CHECKPOINT" --port 17384 \
    --max-active 1 --mtp-steps 0 --fp8-prefill fast-native
```

From a second shell with the same variables:

```bash
python3 -B "$work/repo/benchmarks/streaming/client.py" \
    --url http://127.0.0.1:17384 --output "$work/results-$ARM"
```

For W3, set `ARM` to `w3-flash8192` or `w3-fused`.

Install `tokenizers==0.23.3` in the Python environment that freezes the W3 prompts. The version is pinned in `benchmarks/concurrent/gpu2-reference.json`.

Start the server with the following flags. The published client writes its summary even when timing gates fail.

```bash
"$work/build-$ARM/engine/bin/sglang-server" --checkpoint "$CHECKPOINT" --port 17382 \
    --max-active 4 --fast-prefill-rows 8192 --fast-gdn --flash-prefill --split-decode --graph-decode
```

```bash
cd "$work/repo"
python3 -B -m benchmarks.concurrent.freeze_prompts "$CHECKPOINT/tokenizer.json" "$work/fixtures"
python3 -B -m benchmarks.concurrent.client W3 http://127.0.0.1:17382 "$work/fixtures" "$work/results-$ARM"
```

The wide server flags were `--max-active 32 --fast-prefill-rows 1024 --fast-gdn --batch-graphs --max-batch-context 132096 --trace-batches`, with `--split-gqa-decode` for the split policy. The shortened W4/W5 study used the first 8,192 IDs of each published lane fixture and 16 output IDs per request. The published client runs the full workload when passed `W4` or `W5`. The capacity cancellations, shortened timing pairs and vision parity summaries describe historical runs. Their raw logs, traces and media are outside the bundle.

The bundle leaves out the single-page serial split, packed W3 weight layouts, standalone MTP seed, standalone batch-graph unit, sampling and chat. The measured wide source includes the graph integration it used. `manifest.json` records the patch and source identities, with donor and measurement provenance in `results.json`.
