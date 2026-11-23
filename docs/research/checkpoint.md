# Qwen3.8-27B-FP8 checkpoint contract

## Checkpoint evidence

The local copy at `/data0/shared/Qwen3.8-27B-FP8` exactly matches `Qwen/Qwen3.8-27B-FP8` revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`. On 2026-11-18, all 81 file names and sizes matched. All 67 LFS objects matched their published SHA-256 values, and the other 14 files matched their Git blob SHA-1 values. The full local read covered 30,890,041,705 bytes. [Revision manifest](https://huggingface.co/api/models/Qwen/Qwen3.8-27B-FP8/revision/c3a462cc3d38cf351f3b26cbd763469eeb74ef81?blobs=true)

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

The pinned SGLang source is commit `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`. It loads the Qwen3-VL wrapper around the dense Qwen3.5 text model and packs Q/K/V, FFN gate/up, GDN QKV/Z, and B/A. GQA applies a sigmoid output gate. For one GDN head:

- `g = -exp(A_log) × softplus(a + dt_bias)`
- `β = sigmoid(b)`
- `H = exp(g) × S`
- `S′ = H + β(v − Hk)kᵀ`
- Output is `S′q` before output normalization and gating

Prefix reuse must copy both convolution history `[10240,3]` and recurrent state `[48,128,128]` per GDN layer. A radix hit can resume only after the last reusable recurrent boundary. MTP concatenates normalized embeddings and hidden states, projects them, and reuses one decoder. For top-k 1, verification width is `steps + 1`; three and four proposals use widths four and five.

MXFP8 KV storage uses one unsigned exponent byte per 32 channels. The writer computes `ceil(log2(max(amax,1e-30)/448)) + 127`, clamps to `[0,254]`, and stores E4M3 payloads. Scale bytes use page/head/token-within-32/token-group/channel-group order at page size 128. One FA4 scale-scatter helper requires four groups, while this model’s head width 256 needs eight. The pinned prefix-valid commit path raises for this model, so upstream D256 MXFP8 reachability remains unresolved.

## State capacity

- GDN history and state: `48 × (10240×3×2 + 48×128×128×4)` = 146.8125 MiB per complete target snapshot
- GQA ideal K/V: `16 × 2 × 4 × (256 + 8)` = 33,792 bytes/token
- Native-context GQA ideal K/V: 8.25 GiB

These figures exclude padding, allocator overhead, draft KV, live and speculative state, workspaces, and graph pools.

## Provenance

- Checkpoint revision: `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`
- SGLang revision: `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`
- Verification date: 2026-11-18
- Evidence basis: authenticated local files, remote manifest, safe-tensor headers, and pinned source reads

Historical Candidate B quality evidence is recoverable from tree `e459be98a510d3235c81c94ef5210ef76a622e75`; its baseline capture used fixture source `d6e6eb41cb4ac96ed9f992aee908ac2ea850dca4`. Broader historical measurements are tied to native revisions `6b4f734c9c1a90ce16ff3b61ed9d3f0ad128c811`, `bc07a3b56dc18a9bc32986d76c057a23953d8800`, and `18b9f47ca5dc6ab32c7d51958559407a2f5f85d1`.
