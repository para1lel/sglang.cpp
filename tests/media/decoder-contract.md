# Mixed media decoder contract

Run the test in the project Pixi environment with an assigned SM100 GPU.
Configure and build the `sglang-generate` release target to obtain its static libraries.

```bash
checkpoint=/data0/shared/Qwen3.8-27B-FP8
bash tests/media/run_decoder_contract.sh build /tmp/media-contract "$checkpoint"
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 71 \
    /tmp/media-contract "$checkpoint" --embedding-only
```

The test uses two rectangular images and a two-plane video with unequal timestamp lengths.
Distinct synthetic BF16 rows check substitution for every channel, untouched text rows and
guard rows. Two image spans cross prefill tile boundaries. The wrapper traces arguments to the
real CUDA operators and compares each layer's physical MXFP8 append/read positions with
independent expected mRoPE axes through prefill and the first three decode calls.

The second generation injects `cudaErrorMemoryAllocation` for the prefill workspace pool.
It exercises the one-row fallback with the same inputs and checks that rotary and physical
positions, output callbacks, and cache exclusions still hold. Reported logits diagnose normal
and fallback computation. The test does not require their arithmetic to be bitwise identical.
Swapping equally shaped feature owners under identical text, seed, and salt changes scores.
Replaying the original owners restores the original scores bitwise.
The public-entry test also covers rejected layout, timestamp, ownership address, CUDA storage,
placeholder, MTP, and context-budget inputs, and cancellation before launching a request.

The smaller-budget run compares greedy IDs across grants and compares score bits on a
repeat with the same grant. The K4 fixture traces the BF16 features at shifted token
positions and the original prompt M-RoPE axes for draft seed keys. It checks the first
draft query's axes and verifies a speculative round across an MXFP8 page boundary,
including rejected rows and the output length limit.
