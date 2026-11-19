# Native text serving checks

## Sampling

In the project Pixi environment, configure the optional check and build it
with the native CUDA targets.

```bash
xmake f -m release --sampling-check=y
xmake build sampling-check
xmake run sampling-check /data0/shared/Qwen3.8-27B-FP8
```

Reserve a B200 and select it with `CUDA_VISIBLE_DEVICES` before running.
Omit the checkpoint path to run only the synthetic BF16 logit cases. The
checkpoint run also checks the direct decoder with serial and packed graphs,
eager fallback, mixed MTP, and slot reuse. The text and chat HTTP tests below
exercise the sampled JSON and event-stream routes.

## Chat Layer 1

With the native server running against the pinned Qwen3.8-27B-FP8 checkpoint
at `--max-active 1`, run the chat tests through its HTTP listener.

```bash
python3 -B tests/text/test_native_chat.py --port 17384 -v
```

The test uses the Python standard library. It compares the prompt IDs and
usage returned by the endpoint with the frozen checkpoint template fixtures
and checks the frozen rendered text through the raw tokenizer route. Use
`--checkpoint` to override the default `/data0/shared/Qwen3.8-27B-FP8`.

The rendered fixture text comes from Qwen's `Qwen/Qwen3.8-27B-FP8`, revision
`c3a462cc3d38cf351f3b26cbd763469eeb74ef81`, licensed under Apache-2.0.
The fixture records the asset hashes and uses Hugging Face Transformers 5.23.0,
Tokenizers 0.23.3, and Jinja 3.1.6.
[Checkpoint provenance](../../docs/research/checkpoint.md#checkpoint-evidence)

To regenerate a row, decode assistant function-argument strings as JSON
objects. Copy the first non-null value from `reasoning_content`, `thinking`,
or `reasoning` into `reasoning_content`, retaining empty strings. Pass the
messages and template options to `AutoTokenizer.apply_chat_template` with
`tokenize=False` and encode the result with `add_special_tokens=False`. Set
`add_generation_prompt=True` when the request omits it. Concatenate stored
rendered text and argument fragments without adding separators.

The wire checks cover JSON and SSE, usage, explicit token stops, malformed
requests, raw text parity, and replay after a TCP reset. For public JSON and
SSE with `return_token_ids` disabled, the test decodes the raw generation IDs
using the pinned tokenizer bytes and compares the full content at exhausted
budgets of one, two, and seven tokens. The CPU fixture forces an incomplete
UTF-8 scalar at both short budgets and requires U+FFFD on both public routes.

## Normal server entrypoint

The integration runner starts the candidate and the Unit 18 reference on
loopback. Use commit `7a3b85164588173c0b9ba3aa97acf887b360c862` for the
reference. Its tree is `d4c12ddb4f2f04d4ef0dd32cabedff486ff73502`, and it
contains the LLVM-linked CPU helper used by the control.

Run from the candidate source root in the managed Pixi `default` environment
with Python 3.11 or newer. Point `parent` to an existing checkout of the
reference commit. Build both Release servers and choose a fresh evidence
directory for each run.

```bash
candidate="$PWD"
parent="${UNIT18_SOURCE:?Set UNIT18_SOURCE to the absolute Unit 18 source path}"
test "$(git -C "$parent" rev-parse HEAD)" = 7a3b85164588173c0b9ba3aa97acf887b360c862
candidate_build="$candidate/build/chat-release"
parent_build="$parent/build/chat-release"
xmake f -P "$candidate" -o "$candidate_build" -p linux -a x86_64 -m release
xmake build -P "$candidate" sglang-server
xmake f -P "$parent" -o "$parent_build" -p linux -a x86_64 -m release
xmake build -P "$parent" sglang-server
checkpoint=/data0/shared/Qwen3.8-27B-FP8
out="$candidate_build/chat-evidence"
mkdir -p "$out"
bash tests/text/build_chat_cpu.sh "$candidate" "$candidate_build" "$out/candidate-cpu"
bash "$parent/tests/text/build_chat_cpu.sh" "$parent" "$parent_build" "$out/parent-cpu"
CUDA_VISIBLE_DEVICES='' python3 -B tests/text/run_chat_layer1.py --mode cpu \
    --binary "$out/candidate-cpu" --control "$out/parent-cpu" \
    --checkpoint "$checkpoint" --output "$out/cpu"
```

The CPU link uses the production HTTP entrypoint, scheduler, JSON, and
tokenizer. It replaces the decoder with deterministic token pieces and emits
a link map. Build each control with its own checkout's helper so it uses the
JSON dependency of that revision. The reset case checks an incomplete UTF-8
scalar, cancellation before reuse of the same slot, and three peers that
finish without changes to their streams.

For the GPU run, hold the shared UUID lock on an idle B200. Build the graph
observer from `tests/batch/graph_activity.cpp` with the CUDA toolkit and
`libcupti`, as in `tests/batch/run_batch_graph.sh`. Set `observer` to the
resulting shared library and `GPU_UUID` to the reserved device.

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?Set the reserved GPU UUID}" \
    python3 -B tests/text/run_chat_layer1.py --mode gpu \
    --binary "$candidate_build/bin/sglang-server" \
    --control "$parent_build/bin/sglang-server" \
    --observer "$observer" --checkpoint "$checkpoint" --output "$out/gpu"
```

Both runs check full prompt IDs, zero-token and nonempty responses, and raw
and legacy response bodies before and after slot reuse and against Unit 18.
The GPU run compares mixed widths two through four against isolated
requests and requires CUPTI device graph replays at each width. It checks
continued replay when a cancelled chat is replaced while three legacy streams
stay active. The same candidate binary serves serial MTP at depth four for
greedy parity, including the explicit legacy stop fixture.
