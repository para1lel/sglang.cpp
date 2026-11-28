# Scheduler load

`GET /v1/loads` reports the local HTTP scheduler load on one native worker.
It works with prefix reuse off or on. The HTTP reader serves the request
without claiming a completion slot or querying CUDA. [Dispatch](../../apps/server/prefix_controls.cpp#L600)<!-- prefix: "if (path == " --> · [Pinned Python route](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/v1_loads.py#L96)<!-- prefix: "@router.get(" -->

## Filters and formats

`dp_rank` is optional. Rank zero selects the local worker. Other integer
values return an empty `loads` array, or a single line feed in Prometheus
format. A malformed rank returns HTTP 400 with the native error envelope.
The reader form-decodes names and values and uses the last value for repeated
parameters. It ignores unrecognized query parameters and a framed GET body.
Other methods return HTTP 405 with `Allow: GET`, and HEAD omits the body.

`include` accepts comma-separated section names with surrounding ASCII
whitespace. Core fields always appear for a selected rank. An omitted or
empty `include`, or the section `all`, includes every available section.
`queues` includes the queue breakdown. `core`, `memory`, `spec`, `lora`, and
`disagg` are valid names. Sections without native fields are omitted. An
unknown or empty section in a nonempty list yields the pinned HTTP 500 plain
text response when a rank is selected, unless the list includes `all`.

JSON is the default. Only the exact value `format=prometheus` selects
`text/plain; version=0.0.4; charset=utf-8`. The text format emits numeric
fields as `sglang_` gauges with the label `dp_rank="0"`. Queue fields use the
`sglang_queues_` prefix. [Filter and response writer](../../apps/server/loads.cpp#L177)<!-- prefix: "void write_l" -->

## Native counters

The JSON envelope contains a UTC timestamp, `version: null`, the accelerator
name from the decoder's startup CUDA device properties, `num_accelerators: 1`,
and `loads`. Its single possible row has `dp_rank: 0` and a Unix timestamp.

| Field | Native meaning |
| --- | --- |
| `num_running_reqs` | Claimed HTTP completion-queue requests until their response sockets close. |
| `num_waiting_reqs` | Requests waiting in that queue after removal of disconnected clients. |
| `max_running_requests` | Configured `--max-active` HTTP admission slots. |
| `queues.waiting` | The same waiting count, included when the queue section is selected. |

Both counts come from one lock-protected snapshot. Each queued HTTP request
counts once, including a batch of prompts. Health probes and reader-only
controls use separate paths. Consult [`/ready`](../http-server.md#health-and-readiness)
for readiness. Native omits upstream token-pool, throughput, utilization,
cache-hit-rate, memory, speculation, adapter, and disaggregation measures
that this endpoint cannot derive with the same definitions. Prefix cache
metrics remain available separately through [`/metrics`](../prefix-cache.md#native-http-controls).
[Queue snapshot](../../apps/server/prefix_controls.cpp#L603)<!-- prefix: "const LoadSn" --> · [CUDA device properties](../../src/runtime/backend/cuda_backend_policy.cpp#L39)<!-- prefix: "check(cudaGe" -->
