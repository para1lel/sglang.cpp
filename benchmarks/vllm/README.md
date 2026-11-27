# Independent vLLM baseline on one B200

Run these commands from the repository root on the benchmark host. `GPU` accepts a physical
`nvidia-smi` index or UUID and defaults to `1`. The runner resolves the index to a UUID, locks
that device, checks available memory and active compute processes, and exposes it to vLLM as
logical CUDA device 0. Both engines must run sequentially on the same physical UUID for a pair.

```bash
bash benchmarks/vllm/setup.sh
GPU=1 bash benchmarks/vllm/run.sh benchmarks/vllm/small-fp8.json
```

## Environment

`setup.sh` provisions `/data1/shared/dongwu.chen/environments/vllm-81a8cb063e62` with a detached source checkout and CPython 3.12 venv. It checks the commit-specific CUDA 13.2 index for the vLLM wheel, fetches that commit wheel from `wheels.vllm.ai`, and fetches the PyTorch wheel separately from `download-r2.pytorch.org`. It verifies both digests, the installed vLLM version and dependency compatibility. `constraints.txt` and `wheel.sha256` pin dependencies and direct wheels. The environment keeps installed package versions and shared-object hashes in `freeze.txt` and `native.sha256`.

| Component | Identity |
| --- | --- |
| vLLM checkout | `81a8cb063e6247fe254d7caf7e98a6691d71022a` |
| Official vLLM wheel | `0.34.0rc1.dev787+g81a8cb063`, `cp38-abi3-manylinux_2_28_x86_64` |
| PyTorch wheel | `2.14.0+cu132`, `cp312-cp312-manylinux_2_28_x86_64` |
| Host tools | `/usr/bin/python3.12`, `/usr/local/cuda-13.2`, user CUDA 13.4 compatibility library |
| Local checkpoint | `/data0/shared/Qwen3.8-27B-FP8`, `Qwen3_5ForConditionalGeneration` |
| Checkpoint manifest SHA-256 | `b54fe44144f1701a9369e8b912997ee5cab77cd176c8f97ad3fe63e78cf3c303` |

Setup uses host `git`, `curl`, `jq`, `flock` and `uv` with the isolated interpreter. Proxy access defaults to `http://127.0.0.1:7890` and accepts HTTP, HTTPS and all-proxy environment overrides. The runner verifies wheel, native binary and checkpoint hashes before loading, sets Hugging Face offline mode and keeps generated kernels, autotuning and temporary files in the run directory. It seeds from `/data1/shared/dongwu.chen/cache/vllm-81a8cb063e62` when present. `VLLM_RUNS_DIR` overrides the environment's `runs/` directory.

## Small workload

[`small.json`](small.json#L2)<!-- prefix: "\"prompt_toke" --> sends five direct token IDs and returns IDs without detokenization. The eager TP1 FP8-weight/BF16-activation engine uses a 256-token context and 1 GiB KV cache. One warmup and one measured request must each return the same eight greedy IDs and finish at length, with EOS ignored, seed 0 and prefix caching off.

Measure scheduling, prefill and decode inside `LLM.generate` after loading and warmup. Throughput is eight tokens divided by the measured request duration. The ten-minute timeout preserves logs and exit status on failure. Run evidence includes settings, IDs, timing, logs and GPU snapshots.

## FP8 comparator

Run `small-fp8.json` with the same launcher after setup. The fixture pins the
eight greedy IDs and requests `fp8_e4m3` GQA KV with FlashInfer and BF16 queries.
During the warmup, the runner observes FlashInfer `fa2` paged prefill and
`trtllm-gen` decode, their E4M3 KV inputs, the LBHNC cache layout, and BF16
convolution and FP32 recurrent GDN state.
It checks the checkpoint's scale inventory and verifies unit per-tensor K and V scales
on all 16 GQA layers. The FP8 subprocess enables vLLM callable serialization for the
local worker probe. It writes `fp8-dispatch.json` with the observations and succeeds only after
dispatch, scale, and token checks pass.

The observer checks the phases each request actually exercises. A one-token prompt uses
TRTLLM-GEN for its first GQA step and prefill for its first GDN step. With a multi-token
prompt and one output token, GQA and GDN only exercise prefill. Unused GQA phases report
zero calls with `null` for the entry and backend.

[FP8 result](small-fp8-result.json#L1)<!-- prefix: "{" --> preserves `run.cctk0V`, including the historical two-warmup, seven-measurement result and raw-evidence hashes. [Frozen two-warmup, seven-measurement revision](https://github.com/para1lel/sglang.cpp/blob/e681f90867aa9c59bb0c11bf0e796557f36ce088/benchmarks/vllm/README.md#fp8-comparator)

## Historical BF16 result

Run the BF16 diagnostic with `GPU=1 bash benchmarks/vllm/run.sh benchmarks/vllm/small.json`.

[BF16 result](small-result.json#L1)<!-- prefix: "{" --> preserves the successful `run.GeXtAT` samples, IDs and runtime. [Frozen run and failure notes](https://github.com/para1lel/sglang.cpp/blob/e681f90867aa9c59bb0c11bf0e796557f36ce088/benchmarks/vllm/README.md#historical-bf16-result)

## Pairing with sglang.cpp

1. Reuse the exact checkpoint manifest, prompt IDs, greedy sampling settings, and output length.
   Match TP, prefix cache, eager execution, and speculation settings. The
   [benchmark agreement](../../docs/performance.md#gqa-cache-comparison) permits measured vLLM
   `fp8_e4m3` GQA KV against the candidate's native MXFP8 GQA KV. Verify and record both engines'
   effective cache formats and scale policies. The historical BF16 result is diagnostic only.
2. Run each implementation with exclusive access to the same UUID. Preserve both hardware
   snapshots, client and runtime versions, engine logs, and warmup policy. Record differences
   in kernel dispatch and environment.
3. Verify actual prompt and output counts, exact greedy output IDs, and completion reason before
   reporting a speed ratio. Compare the same call boundary and use the same rate definition.
4. For the serving milestone, follow `docs/performance.md`. Replay the frozen W1–W6 workloads
   through one streaming client for both engines. Preserve request arrival, first-token, last-token,
   and terminal timestamps for TTFT, TPOT, end-to-end latency, and cohort throughput. Offline
   samples here do not provide stream timestamps.
