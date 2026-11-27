# Why long, partial prefill tiles cost time in TP2 W5

Long partial tiles exceeded the former fast GQA path's 32,768-token visibility limit and ran protected attention. The eight-request trace used 160 outputs per request on native `337edce79e46222c90ad700c468b0c5468729463`. Eight such tiles spent 12.890763 s on rank 0 and 12.897292 s on rank 1 in protected GQA, with the ranks running concurrently. The separate official cohort used 1,024 outputs per request and took 66.235146 s native and 39.459475 s vLLM. Its client clocks alone cannot establish how much protected attention contributed to that gap. [Archived runs, dispatch, and evidence hashes](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/docs/research/tp2-w5-prefill.md)

## Subsequent coverage

Staged long prefill subsequently matched every BF16 result for the captured 1,004-row tile. Ordered staging also covers eligible aligned 20-row tails. The replay did not measure full W5 makespan. [Current operator verification](../ordered-gqa-prefill.md#verification) · [Current TP2 acceptance](../serving/tp2.md#acceptance-target)
