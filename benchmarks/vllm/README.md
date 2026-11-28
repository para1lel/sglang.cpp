# Pinned vLLM environment

Provision the vLLM environment from the repository root on the benchmark host. The
[AgentX runbook](../../docs/agentx-256k.md) defines the serving launch and replay.

```bash
bash benchmarks/vllm/setup.sh
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

Setup uses host `git`, `curl`, `jq`, `flock` and `uv` with the isolated interpreter. Proxy access defaults to `http://127.0.0.1:7890` and accepts HTTP, HTTPS and all-proxy environment overrides.
