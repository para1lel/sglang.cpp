# Qwen3.8-27B-FP8 checkpoint contract

## Checkpoint evidence

On 2026-11-18 the local `/data0/shared/Qwen3.8-27B-FP8` matched all published file names, sizes, and content hashes for `Qwen/Qwen3.8-27B-FP8` revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`. [Revision manifest](https://huggingface.co/api/models/Qwen/Qwen3.8-27B-FP8/revision/c3a462cc3d38cf351f3b26cbd763469eeb74ef81?blobs=true)

| Check | Result |
| --- | --- |
| File count | 81 |
| LFS SHA-256 matches | 67 / 67 |
| Non-LFS Git blob SHA-1 matches | 14 / 14 |
| Local bytes read | 30,890,041,705 |
| Tensor bytes | 30,866,663,264 |
| Safe-tensor header and length-prefix bytes | 203,664 |
| Safe-tensor index entries | 1,606 |

The safe-tensor index has 1,606 entries across 64 layer shards, `outside.safetensors`, and `mtp.safetensors`. Every entry maps to exactly one shard header. Shape, dtype, and offsets account for all payload bytes. [Weight index](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/c3a462cc3d38cf351f3b26cbd763469eeb74ef81/model.safetensors.index.json)

## Model contract

| Surface | Contract |
| --- | --- |
| Architecture | `Qwen3_5ForConditionalGeneration`, `qwen3_5` |
| Hidden width | 5,120 |
| FFN | SiLU gated, width 17,408 |
| RMS epsilon | `1e-6` |
| Embedding and LM head | Separate BF16 `[248320,5120]` |
| GQA layers | Zero-based `3,7,…,63` |
| GQA heads | 24 query, 4 KV, head width 256 |
| GDN layers | Remaining 48 |
| GDN heads | 16 Q/K, 48 V, V width 128 |
| Context | 262,144 |
| MRoPE | Sections `[11,11,10]`, theta 10,000,000, 64 rotary dimensions per GQA head |
| EOS | `248046`, `248044` |
| Sampling defaults | Temperature 1, top-k 20, top-p 0.95 |

The GQA packed projection shapes are `[12288,5120]`, `[1024,5120]`, and `[5120,6144]`. The GDN packed QKV, Z, and depthwise convolution shapes are `[10240,5120]`, `[6144,5120]`, and `[10240,1,4]`. `A_log` and `dt_bias` each have 48 elements.

The vision encoder has 27 layers, width 1,152, 16 heads, FFN width 4,304, and output width 5,120. RGB patches span two frames and 16×16 pixels, followed by 2×2 spatial merging. `outside.safetensors` contains 333 BF16 vision tensors, both vocabulary matrices, and the final text norm. `Qwen3VLProcessor` selects `Qwen2VLImageProcessorFast` and `Qwen3VLVideoProcessor`; RGB means and standard deviations are all `0.5`. Vision start/end IDs are `248053` and `248054`, and image/video placeholder IDs are `248056` and `248057`.

`mtp.safetensors` has 22 tensors: two pre-projection norms, a BF16 `[5120,10240]` projection, one full-attention decoder layer, and a final norm. The configuration selects one draft layer and shared target embeddings.

All 407 `F8_E4M3` weight matrices use BF16 `weight_scale_inv` blocks of shape `[ceil(out/128),ceil(in/128)]`. Decode weight element `[r,c]` as `fp8_value × scale[r//128,c//128]`; storage order is output row then input column. Of 1,199 BF16 tensors, 407 are scales totaling 3,015,040 bytes. [Reference dequantization](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/layers/quantization/fp8_utils.py#L2512)<!-- prefix: "return (x_q_" -->

## SGLang reference contract

The archived comparison uses SGLang `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8` and records the GDN recurrence, draft architecture, and MXFP8 layout and dispatch limits at that revision. [Archived SGLang reference](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/docs/research/checkpoint.md#sglang-reference-contract)

## State capacity

One complete target GDN snapshot holds 146.8125 MiB of history and recurrent state. Ideal target GQA K/V storage is 33,792 bytes per token, or 8.25 GiB at the 262,144-token limit, excluding allocation overhead, draft state, scratch, and graph pools. [Archived sizing and quality evidence](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/docs/research/checkpoint.md#state-capacity)
