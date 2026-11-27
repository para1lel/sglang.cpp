# Qwen3.8-27B-FP8 on two B200s

This investigation maps Qwen3.8-27B-FP8 checkpoint tensors, FP8 scale blocks, and LMCache state to two B200 ranks. It records the checkpoint SHA-256 audit, the original rank layout, and NVIDIA collective and GDN source provenance. [Archived rank and operator investigation](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/docs/research/tp2.md) · [Checkpoint contract](checkpoint.md)

## Workload evidence

Release `e912b39bec428ce93728d62b4cae18899d7d3462` passed W1–W3. Native W4 took 29,046.274476 ms against vLLM's 25,335.314580 ms and missed the gate by 1,177.428438 ms. W5 was skipped. The archived investigation records the M80 fusion candidate. [Current TP2 acceptance](../serving/tp2.md#acceptance-target) · [Separate W5 trace](tp2-w5-prefill.md)
