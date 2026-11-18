# sglang.cpp

A C++ and CUDA C++ inference engine in development for **Qwen3.8-27B-FP8 on one NVIDIA B200**.
The first build target checks that the managed toolchain compiles, links, and runs a native
`sm_100` kernel. Model loading, token generation, and serving are subsequent milestones.

## Build on the project host

Run these commands from this checkout. XMake uses the CUDA Toolkit and C++ compiler from the
`default` environment in `/data1/shared/dongwu.chen/pixi.toml`.

```bash
bash tools/toolchain-xmake.sh --install
bash tools/build.sh --build-dir build
nvidia-smi --query-gpu=index,name,uuid,memory.free --format=csv
bash tools/build.sh --build-dir build --gpu 4
```

Choose an available B200 index in the last command. The build verifies the native cubin and
resolved CUDA runtime. The device probe launches one kernel and checks its return value.
Without `--build-dir`, the script uses a temporary build directory.

## Continue development

The checkpoint lives at `/data0/shared/Qwen3.8-27B-FP8`. Keep its weights and generated
artifacts outside version control. Use SGLang as the architecture reference and the pinned
vLLM installation as the independent performance comparator.

- [Isolated vLLM setup and single-B200 smoke run](benchmarks/vllm/README.md) gives the
  commands to generate the first reference token IDs and measure serial request latency.
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

Push each verified commit to the private `origin` and compare `git rev-parse HEAD` with
`git ls-remote origin refs/heads/main`. The hardware run and performance measurements are
separate from the staged build and host checks.
