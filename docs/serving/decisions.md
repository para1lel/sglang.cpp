# Decisions

Send `POST /v1/decisions` with `Content-Type: application/json` and a nonblank string or a
nonempty JSON object or array as `input`. With the loaded `Qwen3.8-27B-FP8` checkpoint,
`prompt_format_version` may be omitted, null, or a value that converts to integer `1`.
Other values return HTTP 400. The adapter uses the checkpoint's chat
template with thinking disabled and scores answer labels through the shared
[pointwise scoring pipeline](pointwise-score.md).

```json
{
  "input": "My laptop will not start.",
  "prompt_format_version": 1,
  "questions": [{
    "id": "category",
    "type": "choice",
    "question": "What kind of issue is this?",
    "options": [{"name": "technical"}, {"name": "billing"}]
  }],
  "return_prompt_token_ids": true
}
```

Each unique question `id` keys an answer. A `choice` has 2–26 options with unique names and
optional descriptions and returns the highest-probability option. A `score` has 2–10 ordered
levels and returns the probability-weighted zero-based level. A `yes_no` question returns
probabilities for `yes` and `no`, with optional descriptions for either answer.

`temperature` is finite and positive and scales the distribution over the admitted labels.
`label_mass` sums their probabilities in the full model vocabulary at temperature 1.
`return_prompt_token_ids` adds the rendered prompt and answer label IDs. The response
reports prompt usage with zero output tokens. Every answer label must append one distinct
token to the unchanged rendered prompt.

An `images` array can contain up to four inline images placed before the text in each
question's prompt. Supply `data:image/png;base64,...` as a string or an object with `url`.
PNG decoding requires an inline media codec build. Native RGB8 image objects can also
specify `url`, `height`, and `width` as in [chat media serving](../chat-serving.md).
Image objects accept automatic detail. URL fetching, bare base64, image processing
overrides, and visual marker text in the input or questions are unsupported.
With images, usage counts expanded visual tokens and `return_prompt_token_ids: true`
returns 400 because text-only `/v1/score` cannot replay the images.
The request may supply a model name for the response metadata. The scoring worker always
uses the loaded checkpoint. The native pointwise score item, text, prompt, and label budgets
also bound Decisions requests.

## System One

Send `POST /v1/systemone` with a `model` string, a `state` string, object or array, and a
`questions` object keyed by question ID. The generic text adapter reports the path of the
loaded checkpoint in `model`. The `state` can be blank. Each question specifies a `type`
and optional `instructions`. A `choice` takes `criteria` as a mapping of option names to
descriptions or null. A `score` takes `criteria` as an ordered list of level descriptions.
A `noul` takes `instructions` or nonblank descriptions in `criteria.true` and
`criteria.false`.

Choice and score answers include probabilities and `confidence`. Scores also include a
legend that preserves the supplied levels. A `noul` answer reports the probability of the
true condition. The `x_label_mass` field reports the total probability of allowed labels
in the full vocabulary. Usage counts input tokens and reports zero output tokens.

Choices support 1–255 entries, subject to the distinct single-token label budget of the
loaded tokenizer and the shared pointwise budgets. Scores support 1–10 levels. Invalid
request fields return a 422 `detail` array. Text inputs support an empty `images` array.
Nonempty images return 400. The route uses generic checkpoint scoring. An Ollama route
configured to the same path takes precedence.
