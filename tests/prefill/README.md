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

Pass either executable to run only its checks. The CLI compares five greedy tokens and the
ordered top 64 scores at prompt lengths 1, 2, 127, 128, 129, and 257. The oracle uses prefixes of
the fixed W2 prompt and the native row-wise decoder at
`dc087912a794372485e78c0a6de5140ac43746e7`. Each score digest covers the little-endian
32-bit token ID and IEEE 754 binary32 logit for each ranked score, in output step and rank order.

The HTTP check compares complete streams at width one and overlapping cohorts at widths two
and four. The batch trace must show multiple chunked lanes, decode alongside prefill, and
at most 128 positions per step. The width-four check resets a connection after a chunk is traced
and before any completion arrives, waits for request allocations to return to zero, then
checks a replacement stream. Require `passed` in the resulting `status.json`.

For the FP8 prefill policy options, run the CPU-only check against the release binaries.
It hides all CUDA devices and verifies help and argument rejection before model load.

```sh
python -B tests/prefill/run_native.py --check-policy \
    --generate build/bin/sglang-generate --server build/bin/sglang-server
```

Build the tensor-core operator check against the release CUDA libraries in the default Pixi
environment. Run it with a reserved SM100 device visible.

```sh
toolkit="$CONDA_PREFIX/targets/x86_64-linux"
c++ -std=c++17 -O3 -I include -isystem "$toolkit/include" \
    tests/prefill/tensor_core_check.cpp build/lib/libsglang_prefill_fp8_sm100.a \
    build/lib/libsglang_cuda.a \
    -L "$toolkit/lib" -Wl,-rpath,"$toolkit/lib" -lcudart -o build/tensor-core-check
build/tensor-core-check
```

The operator check verifies FP64-derived BF16 output and output guards for 197 channels
and 1,024 inputs. It covers tensor-core launches at 3, 127, and 128 rows and exact fallback
at 2 and 129 rows. It also checks alignment, odd input widths, and invalid arguments.
