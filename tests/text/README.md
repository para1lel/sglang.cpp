# Native chat Layer 1 checks

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

The integration runner starts the candidate and the graph parent on loopback.
Build both Release trees with the managed Pixi environment. Set `candidate`
and `parent` to their absolute source paths. Set `candidate_build` and
`parent_build` to the corresponding directories containing `bin/` and `lib/`.
Use Python 3.11 or newer and choose a fresh evidence directory for each run.

```bash
checkpoint=/data0/shared/Qwen3.8-27B-FP8
out="$candidate_build/chat-evidence"
mkdir -p "$out"
bash tests/text/build_chat_cpu.sh "$candidate" "$candidate_build" "$out/candidate-cpu"
bash tests/text/build_chat_cpu.sh "$parent" "$parent_build" "$out/parent-cpu"
CUDA_VISIBLE_DEVICES='' python3 -B tests/text/run_chat_layer1.py --mode cpu \
    --binary "$out/candidate-cpu" --control "$out/parent-cpu" \
    --checkpoint "$checkpoint" --output "$out/cpu"
```

The CPU link uses the production HTTP entrypoint, scheduler, JSON, and
tokenizer. It replaces the decoder with deterministic token pieces and emits
a link map. The reset case checks an incomplete UTF-8 scalar, cancellation
before reuse of the same slot, and three peers that finish without changes
to their streams.

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
and legacy response bodies before and after slot reuse and against the graph
parent. The GPU run compares mixed widths two through four against isolated
requests and requires CUPTI device graph replays at each width. It checks
continued replay when a cancelled chat is replaced while three legacy streams
stay active. The same candidate binary serves serial MTP at depth four for
greedy parity, including the explicit legacy stop fixture.
