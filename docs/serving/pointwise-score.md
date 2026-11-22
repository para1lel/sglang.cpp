# Native pointwise scoring

`POST /v1/score` scores candidate token IDs after each prompt on Qwen3.8-27B-FP8. It returns one JSON response and generates zero tokens.

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| HTTP reader | Require the `application/json` media type for score `POST` requests. | [Media type](../../apps/server/http.cpp#L209)<!-- prefix: "(request.met" --> |
| Route dispatcher | Require `POST` before queue admission. | [Method](../../apps/server/prefix_controls.cpp#L665)<!-- prefix: "const bool s" --> |
| Score worker | Parse the request, build prompts, submit the batch, and write the response. | [Worker](../../apps/server/score.cpp#L363)<!-- prefix: "void serve_s" --> · [Parser](../../apps/server/score.cpp#L264)<!-- prefix: "ScoreRequest" --> · [Response](../../apps/server/score.cpp#L329)<!-- prefix: "std::string " --> |
| Decoder | Project the last prompt row and calculate label log probabilities over the model vocabulary. | [Readback](../../src/decoder_forward.cpp#L668)<!-- prefix: "void Decoder" --> · [Normalization](../../include/sglang/text/label_logprobs.h#L12)<!-- prefix: "std::vector\u003c" --> |
| Reference | Define causal pointwise scoring with upstream `--enable-mis` off. | [Pinned schema](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/entrypoints/openai/protocol.py#L1619)<!-- prefix: "class Scorin" --> · [Label validation](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/tokenizer_manager_score_mixin.py#L920)<!-- prefix: "is_generatio" --> · [Pointwise prefill](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/tokenizer_manager_score_mixin.py#L1173)<!-- prefix: "batch_reques" --> |

## Runtime path

One HTTP request enters the [shared admission queue](../http-server.md#http-admission-and-shutdown). The worker submits one zero-output decoder job per item. If the client disconnects, the worker cancels unfinished work. A request with zero items returns without decoder work. [Dispatch](../../apps/server/prefix_controls.cpp#L771)<!-- prefix: "} else if (s" --> · [Worker](../../apps/server/score.cpp#L363)<!-- prefix: "void serve_s" -->

The decoder processes score grants separately from generation grants. It uses exact target prefill with MTP depth zero, even when generation uses tensor-core prefill or MTP. Score jobs skip prefix lookup and restore. They leave cache counters and entries untouched. At the end of the final prefill grant, the decoder queues a host copy of the full BF16 next-token logits row. It returns the selected log probabilities in label order. [MTP depth](../../src/decoder_batch.cpp#L467)<!-- prefix: "if (options." --> · [Cache restore guard](../../src/decoder_batch.cpp#L428)<!-- prefix: "} else if (a" --> · [Score grants](../../src/decoder_batch.cpp#L765)<!-- prefix: "model.prefil" --> · [Cache accounting guard](../../src/decoder_batch.cpp#L917)<!-- prefix: "cache \u0026\u0026 row" --> · [Final grant](../../src/decoder_prefill.cpp#L932)<!-- prefix: "if (label_sc" -->

## Constraints and failure behavior

Send a JSON object with `Content-Type: application/json`. The route works with native prefix reuse off or on.

| Field | Native behavior |
| --- | --- |
| `items` | Required. Accepts one string, an array of strings, or an array of integer token-ID rows. A bare empty string or empty array returns zero rows. |
| `query` | Optional text or one integer token-ID array. Omission or null supplies an empty query of the item's type. |
| `item_first` | Defaults to `false`. Set it to `true` to put the item before the query. The worker inserts no separator. |
| `label_token_ids` | Required. Use one shared ID array or one ID array per item. For nonempty items, each label row contains one or more distinct integers in `[0, 248044)`. |
| `apply_softmax`, `temperature` | Softmax defaults to `false`. Temperature defaults to `1` and must be finite and positive. A temperature other than `1` requires softmax. |
| `return_token_logprobs` | Defaults to `false`. When enabled, returns raw full-vocabulary log probabilities for the selected IDs in item and label order. |
| `model` | Defaults to `default`. The server echoes the supplied string and uses the loaded checkpoint. |

The worker concatenates text before tokenization and uses no chat template. With a text query and token-ID items, it renders each row as bracketed text such as `[12, 34]` before concatenation. An ID query requires ID rows when items are nonempty. Direct prompt IDs must be in `[0, 248320)`. A request with zero items still requires the label field. The parser checks its type but skips label-row count, length, range, and uniqueness checks. [Prompt construction](../../apps/server/score.cpp#L210)<!-- prefix: "std::vector<" --> · [Empty items](../../apps/server/score.cpp#L294)<!-- prefix: "if (!items.s" -->

The HTTP reader limits the request body to 2 MiB. The score parser applies these additional limits to each request with nonempty items. [Body limit](../../apps/server/http.cpp#L31)<!-- prefix: "constexpr st" --> · [Score limits](../../apps/server/score.cpp#L20)<!-- prefix: "constexpr st" -->

| Resource | Maximum | Counting rule |
| --- | ---: | --- |
| Items | 1,024 | One item produces one prompt and one score row. |
| Tokens per prompt | 262,144 | Each prompt needs at least one token and must fit decoder capacity. |
| Aggregate prompt IDs | 1,048,576 | Count all composed prompts, including each copy of the query. |
| Aggregate composed text | 8 MiB | Count UTF-8 bytes before tokenization, including repeats of the query and rendered ID rows. |
| Returned label values | 262,144 | Count each label once per item in `scores`. If requested, count the same labels again in `token_logprobs`. |

The parser checks item count, direct prompt ID counts, composed text bytes, and label values before building the prompt vectors. Text tokenization uses the smaller of the per-prompt maximum and the remaining aggregate token budget. Requests that exceed a score limit return HTTP 400 with type `BadRequestError`. [Budget checks](../../apps/server/score.cpp#L311)<!-- prefix: "check_label_" --> · [Bounded tokenization](../../apps/server/score.cpp#L223)<!-- prefix: "const auto p" -->

`scores` keeps item order and label order. Let `l` be the full-vocabulary log probability for a label. Without softmax, the score is `exp(l)`. With softmax, let `m` be the maximum `l` among that item's labels. The worker calculates `exp((l - m) / temperature)` for each label. It divides each result by the sum for that item. [Probability conversion](../../apps/server/score.cpp#L233)<!-- prefix: "std::vector<" -->

Temperature and softmax leave the raw log probabilities unchanged. If `return_token_logprobs` is missing or `false`, `token_logprobs` is `null`. If it is `true` and there are zero items, `token_logprobs` is `[]`. `pooled_hidden_states` is always `null`. `object` is `"scoring"`. Usage counts each composed prompt in full. `total_tokens` equals `prompt_tokens`. Completion and reasoning counts are zero. Both token-detail fields are `null`. [Response fields](../../apps/server/score.cpp#L329)<!-- prefix: "std::string " -->

Native checks JSON types directly. The pinned schema also permits Pydantic coercions. For nonempty items, native rejects extraction anchors, embedding overrides, custom `rid`, and `return_pooled_hidden_states: true`. The route has no media, streaming, fused scoring, or classification readouts. [Native subset](../../apps/server/score.cpp#L295)<!-- prefix: "if (optional" --> · [Pinned options](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/tokenizer_manager_score_mixin.py#L1015)<!-- prefix: "use_multi_it" -->

Native converts BF16 logits to float and uses host FP64 for log normalization. It floors selected values at the negative finite FP32 limit. Pinned pointwise scoring reads `output_token_ids_logprobs[0]` after FP32 next-token `log_softmax` and label gathering. The two normalization paths can produce different raw values from identical logits. [Native normalization](../../include/sglang/text/label_logprobs.h#L12)<!-- prefix: "std::vector<" --> · [Pinned FP32 conversion](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/layers/logits_processor.py#L1596)<!-- prefix: "def _copy_lo" --> · [Pinned log-softmax](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/layers/logprob_processor.py#L893)<!-- prefix: "logprobs = t" --> · [Pinned pointwise readout](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/managers/tokenizer_manager_score_mixin.py#L406)<!-- prefix: "# Pointwise:" -->

Invalid media type, malformed JSON, and invalid request shapes return HTTP 400 with type `Bad Request`. Invalid scoring arguments return HTTP 400 with type `BadRequestError`. Both include `object="error"`, `code=400`, `param=null`, and a `message`. A full HTTP queue returns 503. Unexpected worker failures return 500. Other methods return HTTP 405 with `Allow: POST` and `{"detail":"Method Not Allowed"}`. `HEAD` advertises 31 bytes and sends no body. Recognized trailing-slash targets follow the native redirect rules. [Errors](../../apps/server/score.cpp#L372)<!-- prefix: "} catch (con" --> · [Method writer](../../apps/server/http.cpp#L551)<!-- prefix: "void write_m" --> · [Redirects](../http-server.md#request-targets-and-redirects)

## Verification

From the repository root, activate the locked default Pixi environment. Use the local checkpoint for the CPU contract and HTTP fixture.

```bash
checkpoint=/data0/shared/Qwen3.8-27B-FP8
output=artifacts/score-cpu
bash tests/decoder/run_score_check.sh "$checkpoint/tokenizer.json" "$output"
bash tests/text/build_chat_cpu.sh . - "$output/sglang-server-cpu"
/data1/shared/dongwu.chen/.pixi/envs/default/bin/python \
    tests/text/test_native_tokenization_cpu.py --binary "$output/sglang-server-cpu" \
    --checkpoint "$checkpoint" --output "$output"
```

Require exit status zero for each command. The score contract checks budget arithmetic, prompt composition, ordered labels, empty items, normalization, and response fields. The tokenization suite includes a score fixture for methods, `HEAD` body suppression, and an empty score `POST`. [Budget arithmetic](../../tests/decoder/score_check.cpp#L26)<!-- prefix: "void check_b" --> · [HTTP fixture](../../tests/text/test_native_tokenization_cpu.py#L106)<!-- prefix: "def test_sco" -->

When scoring changes, use a reserved B200 and the same checkpoint. Compare raw label log probabilities with appended-label prompt log probabilities from exact `/generate`. Compare concurrent scoring and greedy generation with their serial controls. Compare score responses with prefix reuse off, enabled with a cold cache, and enabled with a warm cache. Use generation to confirm that the cache can store and restore checkpoints. Verify that score requests leave cache counters unchanged.

An independent full-model replay checked 14 requests per cache configuration against pinned SGLang with MIS off. Both engines returned HTTP 200 and agreed on response shape, item and label order, and usage. Raw label log probabilities differed by up to 0.277 nats. In one row, both engines ranked the same label first, and two labels exchanged second and third place. This contract gives no cross-engine guarantee of exact scores or rankings.

Native scores agreed with eight appended-label exact `/generate` readouts within 4.153 × 10⁻⁷ nats. Native score responses matched bit for bit with prefix reuse off and with cold or warm caches. Score requests bypassed prefix reuse and left the counters unchanged. Generation confirmed that the primed prefix was reusable.
