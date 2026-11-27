# Maintaining sglang.cpp

Build a C++ and CUDA C++ inference engine for the local Qwen3.8-27B-FP8 checkpoint on one B200.
Use SGLang as the architecture and behavior reference. Compare end-to-end performance against
the pinned vLLM revision in its separate environment. Verify checkpoint semantics before coding.

## Continuing development

Invoke [main-mode](.agents/skills/main-mode/SKILL.md) explicitly for ongoing work.
Main owns the development branch and index. Start its session review, give workers a fixed
input and disjoint write scopes, and pass each worker the path to its local `SKILL.md`.
Return all worker reports to Main through native agent results using the worker's final response.
Never use `codex queue --message` or any user-input queue to contact Main.
Main's input is reserved for the user.

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
Use the local HTTP proxy at `127.0.0.1:7890` for network operations.
Keep weights at `/data0/shared/Qwen3.8-27B-FP8`.

## Integrating changes

Install local hooks with `git config --local core.hooksPath .githooks` and install
`.agents/checks/requirements.txt` with the default Pixi Python. Hooks check the selected
index and commit message. Keep comments out of project code files: the pre-commit hook
rejects any added or changed code file whose staged content contains a comment, including
comments that predate the change. Shebangs, preprocessor directives, quoted literals, and
here-documents are not comments. Remove comments only from code files you actually touch;
do not clean up untouched legacy files. Perform semantic and English review separately.
Keep raw results in `artifacts/`.

The declaration-only function rule applies to every tracked project header, including
private headers and `.ipp` and `.tpp` files. Put function bodies, lambdas, and `= default`
definitions in compiled `.cpp` or `.cu` files. The sole exception is a deleted function
written `= delete` on its first declaration.

Run tests only when concrete evidence strongly supports the expected outcome.
Prioritize operator and framework tuning and focused local measurements.
Start E2E only when prior evidence makes success highly likely, and stop at the first missed
gate (W5 before W6).

Develop each feature on a feature branch and open a pull request to `main`.
Commit each independently verified change using Conventional Commits. Push each commit to
the feature branch on `origin` before starting the next change. Check the remote branch SHA
against local `HEAD` and resolve a rejected push before continuing. Squash-merge the pull
request into `main` only when the user confirms the feature is complete. Keep the frozen
adoption baseline and limits from `.agents/gates.json` fixed through ordinary changes.
