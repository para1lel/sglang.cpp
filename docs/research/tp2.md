# Qwen3.8-27B-FP8 on two B200s

This investigation maps Qwen3.8-27B-FP8 checkpoint tensors, FP8 scale blocks, and LMCache state to two B200 ranks. See the [checkpoint contract and SHA-256 audit](checkpoint.md#checkpoint-evidence), [TP2 runtime](../serving/tp2.md), and [collective transport](../serving/tp2.md#two-rank-allreduce-transport).
