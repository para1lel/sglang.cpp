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
checkpoint run checks singleton graph replay at a high slot, eager fallback,
mixed MTP, and slot reuse. It expects eager decode for mixed lanes on the B200
ordered-attention path and packed graph replay when that path is disabled.
The text and chat HTTP tests below exercise the sampled JSON and event-stream routes.

## Finite-choice constraints

From the project source directory in the managed Pixi environment, build
the optional direct decoder check and the native server.

```bash
xmake f -m release --constraint-check=y
xmake build constraint-check
xmake build sglang-server
checkpoint=/data0/shared/Qwen3.8-27B-FP8
xmake run constraint-check "$checkpoint"
python3 -B tests/text/test_native_constraint.py \
  --server "$PWD/build/bin/sglang-server" \
  --control "${UNCONSTRAINED_SERVER:?Set this to a pre-constraint Release server with sampling}" \
  --checkpoint "$checkpoint" \
  --evidence "$PWD/artifacts/constraint-http"
```

Reserve a B200 and select it with `CUDA_VISIBLE_DEVICES`. The direct test
compares allowed tokens to a checkpoint-byte oracle, forces greedy and
sampled selection at an accepting prefix, and injects a bad allowed row
through the real selector. Batch checks cover eager and graph replay,
peer completion, cancellation, and slot reuse. Pass `--cursor-only` after
the checkpoint path to run the byte oracle and cursor checks without a GPU.

The HTTP runner starts serial and continuous servers with MTP disabled
and at depth four. It checks both native routes for complete literal
outputs, split UTF-8, usage and finish reasons, invalid grammar handling,
and unchanged unconstrained greedy and sampled output. Use a fresh
evidence directory for each run.

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

The loopback runner compares the candidate with Unit 18, commit `7a3b85164588173c0b9ba3aa97acf887b360c862`, tree `d4c12ddb4f2f04d4ef0dd32cabedff486ff73502`, which provides the LLVM-linked CPU helper. From the candidate root in managed Pixi `default` with Python 3.11+, set `UNIT18_SOURCE` to that checkout. Build both Release servers and use fresh evidence.

```bash
candidate="$PWD"
parent="${UNIT18_SOURCE:?Set UNIT18_SOURCE to the absolute Unit 18 source path}"
test "$(git -C "$parent" rev-parse HEAD)" = 7a3b85164588173c0b9ba3aa97acf887b360c862
candidate_build="$candidate/build/chat-release"
parent_build="$parent/build/chat-release"
for src in "$candidate" "$parent"; do
    xmake f -P "$src" -o "$src/build/chat-release" -p linux -a x86_64 -m release
    xmake build -P "$src" sglang-server
done
checkpoint=/data0/shared/Qwen3.8-27B-FP8
out="$candidate_build/chat-evidence"
mkdir -p "$out"
bash tests/text/build_chat_cpu.sh "$candidate" "$candidate_build" "$out/candidate-cpu"
bash "$parent/tests/text/build_chat_cpu.sh" "$parent" "$parent_build" "$out/parent-cpu"
CUDA_VISIBLE_DEVICES='' python3 -B tests/text/run_chat_layer1.py --mode cpu \
    --binary "$out/candidate-cpu" --control "$out/parent-cpu" \
    --checkpoint "$checkpoint" --output "$out/cpu"
```

Each checkout's helper links that revision's production HTTP entrypoint, scheduler, JSON, and tokenizer with deterministic decoding and emits a link map. The reset case checks an incomplete UTF-8 scalar, cancellation before same-slot reuse, and three peers with unchanged streams.

For GPU, hold the shared UUID lock on an idle B200. Set `observer` to the shared CUPTI observer built from `tests/batch/graph_activity.cpp` with the toolkit and `libcupti`. [Observer build](../batch/README.md#wide-k0-serving)

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
stay active. The same candidate binary runs depth-four MTP with one scheduler
slot and checks greedy parity, including the explicit legacy stop fixture.

## Qwen chat output

Use the Release build directory and managed Pixi with Python 3.11+. Use fresh evidence.

```bash
build="$PWD/build"
checkpoint=/data0/shared/Qwen3.8-27B-FP8
out="$PWD/artifacts/chat-output"
mkdir -p "$out"
bash tests/text/build_chat_cpu.sh "$PWD" "$build" "$out/chat-cpu"
CUDA_VISIBLE_DEVICES='' python3 -B tests/text/check_chat_output_cpu.py \
    --server "$out/chat-cpu" --checkpoint "$checkpoint" --evidence "$out/cpu"
```

The CPU HTTP check runs the server with deterministic checkpoint token IDs.
It splits markers and UTF-8 scalars across tokens and checks how JSON and SSE
carry the same output. It checks independent flags, tool selection, typed
arguments, and replay after a connection reset. Recovery cases distinguish
closed parameters containing literal wrappers from unclosed parameters
followed by another call. They cover invalid sibling fields, JSON quotes,
stray closing tags, the token limit, and explicit stops.

The direct parser probe uses the same tokenizer and JSON libraries. Check whole input, single bytes, tokenizer pieces, every two-way split, seeded chunks, and cancellation.

```bash
"${CXX:-c++}" -std=c++17 -O2 -pthread -Iinclude \
    tests/text/chat_output_chunks.cpp apps/server/qwen/{qwen_chat_output,qwen_tool_syntax}.cpp \
    "$build/lib/libsglang_tokenizer.a" "$build/lib/libsglang_json.a" \
    -o "$out/chat-output-chunks"
"$out/chat-output-chunks" "$checkpoint/tokenizer.json"
```

Hold the shared UUID lock on an idle B200 for the native check.

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?Set the reserved GPU UUID}" \
    python3 -B tests/text/test_chat_output.py --server "$build/bin/sglang-server" \
    --checkpoint "$checkpoint" --evidence "$out/gpu"
```

The native check compares parser-off and parser-on generation at width one.
It repeats the parsed responses with four concurrent requests and checks the
server trace for overlapping decode lanes. The reasoning and tool examples
also run without token diagnostics or an SSE usage frame.

## Native generate on CPU

Run from the repository root in the managed Pixi default environment. The
scripted decoder uses the checkpoint tokenizer and does not load weights.

```bash
out="$PWD/artifacts/native-generate-cpu"
checkpoint=/data0/shared/Qwen3.8-27B-FP8
mkdir -p "$out"
bash tests/text/build_chat_cpu.sh "$PWD" - "$out/server"
CUDA_VISIBLE_DEVICES='' python3 -B tests/text/test_native_generate_cpu.py \
    --binary "$out/server" --checkpoint "$checkpoint" --output "$out/evidence"
CUDA_VISIBLE_DEVICES='' python3 -B tests/text/test_native_parallel_sampling_cpu.py \
    --binary "$out/server" --checkpoint "$checkpoint" --output "$out/parallel"
```

The suite checks raw text and token ID inputs, cumulative SSE, sampling defaults,
validation, model discovery, and shared queue overflow. It disconnects a queued
client and an active stream over TCP and checks that a waiting request and a
subsequent request reuse the released slot.

The parallel suite uses a step gate to hold two active samples and one waiting
sample from one request. It checks indexed output, admission limits, seed
forwarding, fresh IDs and pre-admission duplicate checks. It aborts an active
sample using its observed SSE ID and checks disconnect cleanup with a sample
still waiting.

## Sparse logit bias

Run from the repository root in the managed Pixi default environment. Set
`build` to the XMake Release output directory for the same source tree, with
`sglang-server` and its libraries built. Set `GPU_UUID` to the reserved B200.

```bash
build="$PWD/build"
checkpoint=/data0/shared/Qwen3.8-27B-FP8
out="$PWD/artifacts/logit-bias-check"
mkdir -p "$out"
bash tests/text/build_chat_cpu.sh "$PWD" "$build" "$out/server"
CUDA_VISIBLE_DEVICES='' python3 -B tests/text/test_sampling_routes_cpu.py \
    --server "$out/server" --checkpoint "$checkpoint" --output "$out/cpu"
bash tests/text/run_logit_bias_check.sh "$build" "$checkpoint" \
    "${GPU_UUID:?Set the reserved B200 UUID}" "$out/gpu"
```

The CPU HTTP suite checks token ID aliases in wire order, empty bias, and
validation before streaming. The GPU runner takes both UUID locks and runs
the checkpoint against a host oracle for FP32 score addition, log-softmax,
and seeded selection. It also checks constraint masking, packed eager and
graph parity, and a target-only biased lane beside an MTP lane.
The minimum-output checks cover all three request routes, full-vocabulary
stop masking, seeded and greedy selection, and different minima in packed lanes.
