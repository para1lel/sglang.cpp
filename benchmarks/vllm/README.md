# Independent vLLM baseline on one B200

Run these commands from the repository root on the benchmark host. `GPU` accepts a physical
`nvidia-smi` index or UUID and defaults to `1`. The runner resolves the index to a UUID, locks
that device, checks available memory and active compute processes, and exposes it to vLLM as
logical CUDA device 0. Both engines must run sequentially on the same physical UUID for a pair.

```bash
bash benchmarks/vllm/setup.sh
GPU=1 bash benchmarks/vllm/run.sh benchmarks/vllm/small.json
```

## Environment

`setup.sh` provisions `/data1/shared/dongwu.chen/environments/vllm-81a8cb063e62` with a detached
source checkout, a CPython 3.12 virtual environment, and official Linux x86-64 wheels. It checks
the upstream commit-specific CUDA 13.2 index, artifact digests, installed vLLM version, and
dependency compatibility. `constraints.txt` pins the resolved transitive package versions.
`wheel.sha256` pins the two directly installed wheels. `freeze.txt` and `native.sha256` in the
environment record the installed packages and the hashes of vLLM shared objects.

| Component | Identity |
| --- | --- |
| vLLM checkout | `81a8cb063e6247fe254d7caf7e98a6691d71022a` |
| Official vLLM wheel | `0.34.0rc1.dev787+g81a8cb063`, `cp38-abi3-manylinux_2_28_x86_64` |
| PyTorch wheel | `2.14.0+cu132`, `cp312-cp312-manylinux_2_28_x86_64` |
| Host tools | `/usr/bin/python3.12`, `/usr/local/cuda-13.2`, user CUDA 13.4 compatibility library |
| Local checkpoint | `/data0/shared/Qwen3.8-27B-FP8`, `Qwen3_5ForConditionalGeneration` |
| Checkpoint manifest SHA-256 | `b54fe44144f1701a9369e8b912997ee5cab77cd176c8f97ad3fe63e78cf3c303` |

Setup uses `git`, `curl`, `jq`, `flock`, and `uv` from the host. Every package installation names
the isolated interpreter explicitly. Network access uses Mihomo at `127.0.0.1:7890`. The runner
requires the local checkpoint and hashes the weight, configuration, tokenizer, and template files
before loading. It sets Hugging Face offline mode. vLLM and kernel libraries can have additional
first-run initialization or compilation costs.

## Small fixture

`small.json` supplies the five token IDs for `The capital of France is`. The engine accepts IDs
directly and returns greedy token IDs without detokenization. One resident eager engine serves
two warmups and seven measured serial requests, each with eight output tokens, EOS ignored,
prefix caching disabled, and seed 0. Each call must complete at the requested length and
produce the same token IDs. The text-only path uses checkpoint FP8 weights, BF16 activation
dtype, tensor parallelism 1, a 256-token context, and a 1 GiB KV cache allocation.

Latency starts immediately before `LLM.generate` and ends when it returns the completed request.
It includes scheduling, prefill, and decode. Model loading, checkpoint hashing, and the two
warmups occur outside the samples. The offline serial smoke rate, `output_tokens_per_second`,
divides the 56 measured output tokens by the sum of the seven completed request durations in
seconds. `output_tokens_per_median_request_second` divides eight tokens by median request latency
in seconds. The stored individual samples allow both rates to be recalculated. The runner enforces a
ten-minute engine timeout. Its printed run directory contains per-trial samples, output IDs,
the engine log, settings, and before/after device snapshots. A failed run preserves its log
and exit status.

## Pairing with sglang.cpp

1. Reuse the exact checkpoint manifest, prompt IDs, greedy sampling settings, and output length.
   Match TP, effective cache precision, prefix cache, eager execution, and speculation settings.
2. Run each implementation with exclusive access to the same UUID. Preserve both hardware
   snapshots, client and runtime versions, engine logs, and warmup policy. Record differences
   in kernel dispatch and environment.
3. Verify actual prompt and output counts, exact greedy output IDs, and completion reason before
   reporting a speed ratio. Compare the same call boundary and use the same rate definition.
4. For the serving milestone, follow `docs/performance.md`. Replay the frozen W1–W6 workloads
   through one streaming client for both engines. Preserve request arrival, first-token, last-token,
   and terminal timestamps for TTFT, TPOT, end-to-end latency, and cohort throughput. Offline
   samples here do not provide stream timestamps.
