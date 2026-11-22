# Matched W1 and W2 K4 with split target attention

On November 22, 2026 UTC, Release `96aff6ab1796b71f07e11ad529e4f77837fde4ff`
ran the [opt-in split profile](../../benchmarks/paired/README.md#ordered-tensor-core-with-split-target-attention-and-int8-draft).
Exact FP8 remains the default. Both arms used B200 GPU3
(`GPU-ceeff2f7-3f39-50b4-8d29-576159781b9d`), both GPU UUID locks, local
NUMA allocation, server CPUs 24–33, client CPUs 34–35, 32 slots and an 8,192-token cap.
Each timed arm ran three warmups and nine measured cohorts.

| Row (input/output IDs) | Native median (range), ms | Pinned vLLM median (range), ms | Strict median gate |
| --- | ---: | ---: | --- |
| W1 (128/128) | 613.519116 (612.100262–663.536757) | 650.140507 (649.687019–650.901273) | Pass, 5.633% lower |
| W2 (2,048/512) | 1,542.896320 (1,523.720252–2,212.537332) | 1,742.236508 (1,740.036383–1,744.450085) | Pass, 11.442% lower |

Each native range includes one sample above the vLLM maximum. Native median
time from first output to `[DONE]` was 3.660% lower in W1 and 16.125% lower
in W2. Replay verified 53 streams and native same-policy K0 parity.
The native W1 trace reached depth four in 49 of 50 rounds. vLLM supplied timing.

An independent capped W3 diagnostic on a private tree measured 11.561 s of
target verification attention across four 8,192-token lanes at 32 outputs each.
This blocked the formal W3 launch pending long-context runtime qualification.
W3–W6 remain unmeasured.

Raw comparisons, audits, lock receipts and teardown are under
`artifacts/continue-split-campaign-test/campaign-published-96aff6a-gpu3/`.
See `W2/independent-live-locks.txt` for lock paths. The separate W3 diagnostic
is `artifacts/continue-w3-short-diagnostic/REPORT.md`.
