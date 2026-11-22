# Decisions

Send `POST /v1/decisions` with `Content-Type: application/json` and a nonblank string or a
nonempty JSON object or array as `input`. With the loaded `Qwen3.8-27B-FP8` checkpoint,
`prompt_format_version` may be omitted, null, or a value that converts to integer `1`.
Other values return HTTP 400. This text adapter uses the checkpoint's chat
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

This text route supports an empty `images` array. A request with image inputs returns 400.
The request may supply a model name for the response metadata. The scoring worker always
uses the loaded checkpoint. The native pointwise score item, text, prompt, and label budgets
also bound Decisions requests.
