# Why long, partial prefill tiles cost time in TP2 W5

In the eight-request, 160-output trace of native `337edce79e46222c90ad700c468b0c5468729463`, eight long partial prefill tiles spent 12.890763 s on rank 0 and 12.897292 s on rank 1 in protected GQA. The separate official 1,024-output cohort took 66.235146 s native and 39.459475 s vLLM and recorded client clocks only. [Archived runs, dispatch, and evidence hashes](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/docs/research/tp2-w5-prefill.md)

## Subsequent coverage

Staged long prefill subsequently matched every BF16 result for the captured 1,004-row tile. Ordered staging also covers eligible aligned 20-row tails. The replay did not measure full W5 makespan. [Current operator verification](../ordered-gqa-prefill.md#verification) · [Current TP2 acceptance](../serving/tp2.md#acceptance-target)
