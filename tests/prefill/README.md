# Native prefill acceptance

Run the opt-in checkpoint check with a release `sglang-generate`, release `sglang-server`,
the local Qwen3.8-27B-FP8 checkpoint, and one exclusively reserved B200. Set
`CUDA_VISIBLE_DEVICES` to the reserved device UUID before starting the runner. Use the default
Pixi environment and a fresh evidence directory.

```sh
python -B tests/prefill/run_native.py --check-fixtures
python -B tests/prefill/run_native.py \
    --generate build/bin/sglang-generate \
    --server build/bin/sglang-server \
    --evidence artifacts/prefill-check
```

Pass either executable to run its checks on seven fixed W2 prefixes from 1 to 511 rows.
The CLI uses `--mtp-steps 0` and compares five greedy IDs and the ordered top 64 scores
across two invocations. Each score digest covers the little-endian 32-bit token ID and
IEEE 754 binary32 logit in step and rank order.

The HTTP checks use `--mtp-steps 0`, compare complete SSE against serial SSE under the
same server configuration, and retain the serial responses as evidence. The first
server checks the default eight active requests and 32,768-token budget. A separate
`--max-active 1 --eager --max-num-batched-tokens 8192` launch checks the explicit override.

The width-two cohort uses 129- and 257-token prompts. The width-four cohort uses 2-, 127-,
129-, and 257-token prompts. Both use the same five-token generation budget and an explicit
scheduler budget of 128 positions per step. The trace must show multiple chunked lanes
and decode alongside prefill within that budget. The width-four check resets a
connection after a chunk is traced and before any completion arrives, waits for request
allocations to return to zero, then checks a replacement stream and its own allocation-to-zero
transition. The reset evidence records the trace offsets for both transitions. Each launch
records its command and `/health` response. Require `passed` in the resulting `status.json`.

The CPU-only CLI check hides CUDA devices and verifies help and rejection of the
retired FP8 prefill selector before model load.

```sh
python -B tests/prefill/run_native.py --check-policy \
    --generate build/bin/sglang-generate --server build/bin/sglang-server
```

Build the tensor-core operator check against the release CUDA libraries in the default Pixi
environment. Run it with a reserved SM100 device visible.

```sh
toolkit="$CONDA_PREFIX/targets/x86_64-linux"
c++ -std=c++17 -O3 -I include -isystem "$toolkit/include" \
    tests/prefill/tensor_core_check.cpp \
    -Wl,--start-group build/lib/libsglang_prefill_fp8_sm100.a \
    build/lib/libsglang_cuda.a build/lib/libsglang_cuda_support.a -Wl,--end-group \
    -L "$toolkit/lib" -Wl,-rpath,"$toolkit/lib" -lcudart -o build/tensor-core-check
build/tensor-core-check
```

The operator check verifies FP64-derived BF16 output and output guards for 197 channels
and 1,024 inputs. It covers tensor-core launches at 3, 127, and 128 rows and exact fallback
at 2 and 129 rows. It also checks alignment, odd input widths, and invalid arguments.

Configure with `--mxfp8-gqa-check=y` and build `mxfp8-gqa-check` to exercise the
serving attention routes. The CUDA check covers raw and staged paired kernels,
Flash and protected paths, causal and nonfinite recovery, alignment and context
boundaries, and graph replay. It checks that exact-fit staged scratch matches raw
output bit for bit and that one-byte-short scratch and capture use the raw route.
