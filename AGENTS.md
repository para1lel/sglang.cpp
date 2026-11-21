# Maintaining sglang.cpp

Build a C++ and CUDA C++ inference engine for the local Qwen3.8-27B-FP8 checkpoint on one B200.
Use SGLang as the architecture and behavior reference. Compare end-to-end performance against
the pinned vLLM revision in its separate environment. Verify checkpoint semantics before coding.

## Continuing development

Invoke [main-mode](.agents/skills/main-mode/SKILL.md) explicitly for ongoing work.
Main owns the development branch and index. Start its session review, give workers a fixed
input and disjoint write scopes, and pass each worker the path to its local `SKILL.md`.
The following roles and their resources are bundled with this checkout.

- [Research](.agents/skills/research-mode/SKILL.md) verifies unknowns under `docs/research/`.
- [Coding](.agents/skills/coding-mode/SKILL.md) owns implementation, builds and root files.
- [Test](.agents/skills/test-mode/SKILL.md) independently reviews, tests and preserves evidence.
- [Doc](.agents/skills/doc-mode/SKILL.md) maintains the other project documents under `docs/`.

For kernel work pass [Blackwell](.agents/skills/principle-blackwell/SKILL.md).
For model or serving behavior pass [inference](.agents/skills/principle-inference/SKILL.md).
For benchmarking pass [performance](.agents/skills/principle-performance/SKILL.md).
When a `.codegraph/` index exists, use `codegraph_explore` or `codegraph explore` first to
locate code. The user decides whether to create an index.

## Working on this server

Use XMake and the default environment from `/data1/shared/dongwu.chen/pixi.toml`.
Keep Python packages in Pixi, executables under the user's `bin`, runtimes in `software`,
configuration in `config` and disposable output in `cache` or ignored project `artifacts/`.
Use the configured centralized directory for additional Pixi environments and the existing npm
prefix for global Node tools. Use the host driver and user CUDA 13.4 compatibility libraries.
For network operations use the local Mihomo mixed proxy at `127.0.0.1:7890` with its existing
rule-mode routing. Keep weights at `/data0/shared/Qwen3.8-27B-FP8`.

## Integrating changes

Install local hooks with `git config --local core.hooksPath .githooks` and install
`.agents/checks/requirements.txt` with the default Pixi Python. Hooks check the selected
index and commit message. Perform semantic and English review separately. Keep raw results
in `artifacts/`.

Run tests only when concrete evidence strongly supports the expected outcome.
Prioritize operator and framework tuning and focused local measurements.
Start E2E only when prior evidence makes success highly likely, and stop at the first missed
gate (W5 before W6).

Commit each independently verified change using Conventional Commits. Push each commit to
the private `origin` before starting the next change. Check the remote branch SHA against
local `HEAD` and resolve a rejected push before continuing. Keep the frozen adoption baseline
and limits from `.agents/gates.json` fixed through ordinary changes.
