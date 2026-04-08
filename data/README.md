# Data Layout

- `task/`: official Touché dataset files plus `preprocessed/` outputs from
  `touche-preprocess`
- `generated/gemini/`: Gemini-generated neutral-response files
- `generated/qwen/`: Qwen-generated neutral-response files
- `generated/chatgpt/`: reserved for a future hosted OpenAI-style provider

The upstream dataset card is in `task/README.md`.

## How The Setup Matrix Uses These Files

- `setup6`, `setup8`, `setup9`, `setup10`, `setup11`, and `setup12` use the
  plain `query_response` prompt, so they only consume `query` and `response`
  even when the JSONL file also contains a neutral field.
- `setup6-qwen` still uses `query_response`; it mainly exists so training and
  evaluation are aligned on the Qwen-enriched file family.
- `setup7` and `setup7-qwen` use the neutral field directly as part of the
  prompt.
- `setup4` uses the Gemini neutral field as an explicitly labeled reference.
- `setup100` to `setup105_1`, plus `setup113`, `setup114`, `setup117`,
  `setup118`, and `setup119`, are neutral-vs-response experiments; they depend
  on having a valid neutral field such as `gemini25flashlite` or `qwen`.
- `setup110` and `setup111` are the scalar multi-anchor baselines. They depend
  on both the Gemini and Qwen generated files and merge them by `id`.
- `setup106` is still documented, but its sentence-delta backend is not
  currently wired into the CLI.

See `../setup.md` for the full per-setup explanation.
