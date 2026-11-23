# Native and SGLang HTTP compatibility

## Scope

This note records the compatibility boundary between the native service and pinned SGLang. It is based on the default Python autoregressive application; Ray reuses that application. `SGLANG_RUST_SERVER=1` and `sgl-router` use separate route registries.

## Route families

Native implements the core serving and operational families needed by the frozen client:

- Health, readiness, and generation progress
- Model and server information
- `/generate`
- `/v1/chat/completions`
- `/v1/completions`
- Tokenization and Anthropic count-tokens
- OpenAI Responses
- Abort
- Prefix-cache controls, including reset, metrics, profiling, and flush
- Scheduler-load and Prometheus metrics

Native Responses and chat support text, image, video, reasoning, tool-call, and JSON-schema paths. They do not claim SGLang’s full administrative, distributed, adapter, retrieval, or hosted-tool surface.

The historical comparison counted 34 of 92 default SGLang paths with native defaults, 36 with prefix reuse or the Responses store, and 38 with both. Those counts are fixed to the source revisions below and must not be cited as a live parity claim without a fresh audit.

## Behavioral constraints

- Native `/metrics` requires prefix reuse; SGLang exposes metrics without that requirement.
- Native `/reset_prefix_cache`, `/start_profile`, and `/stop_profile` are tied to the native prefix-cache implementation.
- Native `/flush_cache` works with reuse on or off and waits for idle with a FIFO timeout policy.
- Native abort handles `/generate`; SGLang also aborts OpenAI generation paths.
- Native server information reports the native schema, not SGLang’s distributed configuration layout.
- Native media, schema, and hosted-tool coverage is narrower than upstream.

## Provenance

- SGLang revision: `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`
- Native route baseline: `bd55e77edd926e4db9a8e2a115b54b6d773eb879`
- Native smoke revision: `18b9f47ca5dc6ab32c7d51958559407a2f5f85d1`
- Audit date: 2026-11-22
- Evidence basis: pinned source trees and native HTTP smoke artifacts
