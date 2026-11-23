# Serving session handoff

## Formal result

On November 23, 2026 UTC, the latest completed matched native-defaults versus
pinned vLLM MTP4 campaign passed W1 and W2 on B200 GPU3. Each timed arm used
three warmups and nine cohorts. Both rows exited zero.

| Row | Native median | vLLM MTP4 median | Gate |
| --- | ---: | ---: | --- |
| W1, 128 input and 128 output IDs | 613.877 ms | 648.982 ms | Pass, 5.409% lower |
| W2, 2,048 input and 512 output IDs | 1,407.243 ms | 1,952.643 ms | Pass, 27.931% lower |

This run measured native `8b4e3f62e0a7a053ffdefe85da99a841fabeb5d3` with the
argv receipt repair in the benchmark launcher. The vLLM pin is
`81a8cb063e6247fe254d7caf7e98a6691d71022a`. The native release predates the
subsequent online draft, early causal prefill, four-grant projection, and
grouped GDN changes. See the [published result](paired-native-defaults-results.md)
for the recorded row hashes, checkpoint identity, and launch settings.

The later literal-default comparison on `c018977` completed W1 and W2
against vLLM with speculation disabled. Medians were 619.607 vs 1,217.040 ms
and 1,404.959 vs 4,389.295 ms. Its continuation was canceled before W3.
The matched MTP4 ledger above remains the formal comparison.

## Current source and W3

- **Published** `932a77d5dbac11016dc96ce31e7cdf879d0a13f1`, tree
  `28f844b36bdd7a5a896ef9ad8a3135b0f053b463`. This includes four-grant
  projection (`061d627`) and grouped GDN prefill. The GDN gate passed both
  hooks, native HTTP and scoring checks, state comparison, and bounded timing.
- **Fused M20 pending bound paged replay**. The runtime composition is
  `ebd78bf9fe73728d60bcb22507315811d14d3ebb`. The documentation bridge
  selects `77318cbc87e448d722e39cb2c156931a5dd2b451`. The joined runtime
  passed independent dispatch and state checks. Both hooks passed on the
  independent anchor variant `94a16ba1ce3ff089472f1895453d8f825ec1b3e1`.
  The final docs tree passed exact replay and repository rule checks.
- **Projection prototypes passed build only**. The M20 activation prototype is
  `05170fd06822552bd35a63518297244dad3f806b`. The combined prototype is
  `cd22e49e22dbc38f215249e1639dcd4f87575fc6`. Both use Bohr's standalone
  anchor tree `ce149737118030fbdfc09e9a50b4c9291cac0c70` as their parent.
  The prefill-layout work remains exploratory.

On `77318cb`, the latest capped diagnostic used four 8,192-ID lanes on GPU3
and the saved same-device MTP4 comparator. Native versus vLLM median
makespan was 2,262.376 vs 1,945.789 ms at 256 output IDs per lane and
3,053.781 vs 2,622.102 ms at 512. The 512-cap native sample range reached
5,090.793 ms. The linear 1,024-ID projection was 4,636.591 vs 3,974.729 ms,
about 16.7% slower. Full W3 remains unmeasured. Keep the formal W3 gate
closed and W4–W6 pending.

## Recovery after cleanup

Restore the flat handoff files from tag `handoff-20261123-mtp4`
(`42a9f7fcf04e78ec40bfafe790849f4afbaed542`) into an existing empty directory.
This annotated tree tag may need an explicit fetch.

```sh
git fetch origin tag handoff-20261123-mtp4
git archive handoff-20261123-mtp4 | tar -x -C /path/to/empty-directory
```

The tag retains summaries and selected sources only. Rerun the workloads
to capture fresh raw SSE and rebuild to produce fresh binaries.

- `m20-on-932a77d.patch` and `m20-on-061d627.patch` apply to their
  respective bases. Bound paged replay remains pending for `932a77d`.
- `activation.patch` and `combined.patch` are build-only prototypes for
  repaired base `ce149737118030fbdfc09e9a50b4c9291cac0c70`.
- `seed_tc.cu`, `seed.cu`, and `seed-receipt.txt` cover the operator-only
  seed probes.
- `w3-run.sh`, `composed.patch`, `w3-source-identity.txt`, `w3-replay.json`,
  and `m20-replay.json` cover the capped diagnostic and M20 replay.
- `W1-comparison.json` and `W2-comparison.json` contain the formal
  comparison summaries.
- Reconstruct the measured formal source from `8b4e3f6` and replace
  `benchmarks/paired/run_k4.sh` with the blob from `c018977`. The measured
  `benchmarks` subtree is `d09757c21332ab018cb436673ed6363a28a3b5b0`.
  Follow the [build instructions](../../README.md#build-on-the-project-host)
  and [paired protocol](../../benchmarks/paired/README.md#matched-k4-timing-protocol)
  with the checkpoint manifest and pinned environment.
- Verify the archived `composed.patch` SHA-256
  `1f7da6d772b62ca6e539e5f1bca3062da95514f015f917abe8093f2751f10fb9`.
  Apply it with `git apply --index` in an isolated checkout of `932a77d`.
  `git write-tree` must return `77318cbc87e448d722e39cb2c156931a5dd2b451`.
