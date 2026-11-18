# sglang.cpp

A C++ and CUDA C++ inference engine in development for **Qwen3.8-27B-FP8 on one NVIDIA B200**.
The native `sglang_cuda` library implements paged MXFP8 KV-cache writes and single-request
causal grouped-query attention for the checkpoint's full-attention shape. Model loading,
token generation, and serving are subsequent milestones.

The public C++ API is in
[`include/sglang/mxfp8_gqa.h`](include/sglang/mxfp8_gqa.h#L48)<!-- prefix: "cudaError_t " -->.
The CUDA implementation is in
[`src/mxfp8_gqa.cu`](src/mxfp8_gqa.cu#L32)<!-- prefix: "__device__ i" -->.

## Build on the project host

Run these commands from this checkout. XMake uses the CUDA Toolkit and C++ compiler from the
`default` environment in `/data1/shared/dongwu.chen/pixi.toml`.

```bash
bash tools/toolchain-xmake.sh --install
bash tools/build.sh --build-dir build
nvidia-smi --query-gpu=index,name,uuid,memory.free --format=csv
bash tools/build.sh --build-dir build --gpu 4
```

Choose an available B200 index in the last command. The build compiles `sglang_cuda` and
both toolchain probes. It verifies the AOT probe's native cubin and resolved CUDA runtime.
With `--gpu`, the AOT probe launches one kernel and checks its return value.
Without `--build-dir`, the script uses a temporary build directory.

To run the P1 NVRTC probe, use the physical index of a reserved B200.

```bash
bash tools/build.sh --build-dir build --nvrtc --gpu "${GPU_INDEX:?set the reserved B200 index}"
```

The probe compiles a native `sm_100` kernel and caches the loaded function in its CUDA
context for the second launch. Both launches verify all 257 integers. The output records
the device UUID and properties, allocation size, cold and warm cache results,
NVRTC compilation time, and completed kernel time from CUDA events.

Build the candidate MXFP8 correctness check without using a GPU.

```bash
bash tools/build.sh --build-dir build --mxfp8-check
```

`mxfp8-gqa-check` links `sglang_cuda`. Run the reported executable on a reserved B200
to check cache bytes and attention outputs. This build command only compiles and links the check.

## Continue development

The checkpoint lives at `/data0/shared/Qwen3.8-27B-FP8`. Keep its weights and generated
artifacts outside version control. Use SGLang as the architecture reference and the pinned
vLLM installation as the independent performance comparator.

- [Isolated vLLM setup and single-B200 baseline](benchmarks/vllm/README.md) reproduces
  the reference token IDs for the first candidate decoder. Baseline vLLM `81a8cb063e62` reached
  **25.89 output tokens/s** and **309.06 ms** median request latency on this serial fixture
  (five prompt tokens, eight generated tokens, two warmups, seven measured requests).
  The W1–W6 serving comparison remains open.
- [Checkpoint and SGLang source map](docs/research/checkpoint.md) identifies the upstream
  revision and records content verification for all 81 files and 1,606 indexed tensors.
- [Development milestones](docs/development.md) define the path from the runtime probe to
  a decoder that matches reference tokens, then to serving and single-card optimization.
- [Benchmark agreement](docs/performance.md) specifies the vLLM commit, the small greedy
  fixture, and the throughput, latency, and correctness evidence for W1–W6.

Start a maintenance session by explicitly invoking `.agents/skills/main-mode/SKILL.md`
with the milestone you want to advance. Main assigns the repository's local research,
coding, test, and documentation roles. See the [repository instructions](AGENTS.md).

For a fresh checkout on this host, install the gate dependencies and enable the local hooks.
Run commits from the managed default environment so the hooks use its Python.

```bash
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890
pixi shell --manifest-path /data1/shared/dongwu.chen/pixi.toml --environment default
python -m pip install -r .agents/checks/requirements.txt
git config --local core.hooksPath .githooks
git config --local http.https://github.com/.proxy http://127.0.0.1:7890
python -IB .agents/checks/check_rules.py check .
python -IB .agents/checks/check_rules.py package .agents/skills --policy-root .
git hook run pre-commit
```

To verify the handoff independently, run the installed hooks in disposable clones and save
the observations. The proof checks staged and unstaged disagreements, an alternate index,
the frozen size limits, test failures, and the XMake compile gate.

```bash
mkdir -p artifacts/handoff
python -B tests/prove_handoff.py --revision HEAD > artifacts/handoff/verification.jsonl
tail -1 artifacts/handoff/verification.jsonl
```

Push each verified commit to the private `origin` and compare `git rev-parse HEAD` with
`git ls-remote origin refs/heads/main`. The hardware run and performance measurements are
separate from the staged build and host checks.
