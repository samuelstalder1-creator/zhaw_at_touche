# Implementation Details

## Module Mapping

- Neutral-response generation is implemented in:
  - `src/zhaw_at_touche/generation_utils.py`
  - `src/zhaw_at_touche/cli/generate_neutral.py`
- Generated-data token analysis is implemented in:
  - `src/zhaw_at_touche/generated_stats.py`
  - `src/zhaw_at_touche/cli/generated_stats.py`
- Split-overlap checks are implemented in:
  - `src/zhaw_at_touche/overlap_utils.py`
  - `src/zhaw_at_touche/cli/check_overlap.py`
- Training, validation, and manual inference are implemented in:
  - `src/zhaw_at_touche/modeling.py`
  - `src/zhaw_at_touche/cli/train_model.py`
  - `src/zhaw_at_touche/cli/validate_model.py`
  - `src/zhaw_at_touche/embedding_divergence.py`
  - `src/zhaw_at_touche/cli/embedding_divergence.py`
  - `src/zhaw_at_touche/cli/manual_inference.py`
- Evaluation summaries are implemented in:
  - `src/zhaw_at_touche/evaluation_utils.py`
  - `src/zhaw_at_touche/cli/evaluation_matrix.py`
- Only the current package layout is kept in the repository; imported standalone folders are no longer present.

## Folder Decisions

- `data/task/` is the canonical home for official dataset files.
- `data/task/preprocessed/` contains merged response + label files created by `touche-preprocess`.
- `data/generated/gemini/` stores generated neutral-response datasets.
- `data/generated/qwen/` stores generated neutral-response datasets from the self-hosted Qwen backend.
- `data/generated/chatgpt/` is intentionally present even though no hosted OpenAI generation backend is implemented yet.
- `train_model/<setup-name>.json` stores reusable defaults for named training experiments such as `setup4`, `setup6`, `setup7`, `setup8`, `setup9`, `setup10`, `setup11`, `setup12`, and the embedding-divergence `setup100`.
- The current setup family covers RoBERTa, Longformer, DeBERTa-v3, ALBERT, ELECTRA, and DistilRoBERTa variants.
- `validate_model/<setup-name>.json` stores reusable defaults for evaluation-only experiments such as `teamCMU`.
- `validate_model/setup4.json` and `validate_model/setup7.json` mirror the reference-aware local training setups.
- `validate_model/setup9.json`, `validate_model/setup10.json`, `validate_model/setup11.json`, and `validate_model/setup12.json` provide explicit local validation presets for the newer query-response baselines.
- `models/<setup-name>/` stores saved Hugging Face model bundles plus `training_summary.json`.
- `results/<setup-name>/` stores validation artifacts and prediction exports.

## CLI Defaults

- Training defaults to `data/generated/gemini/responses-train-with-neutral_gemini.jsonl` if present.
- If generated files are missing, training falls back to `data/task/preprocessed/responses-train-merged.jsonl`.
- Training uses the full training file by default, with optional row limiting through `--max-train-rows`.
- Validation defaults to the generated Gemini `test` file only, with fallback to the preprocessed merged `test` file.
- Validation can include both `validation` and `test` through `--eval-splits validation test`.
- Validation can load either a local saved model directory or a remote Hugging Face model name.
- Input formatting is configurable; `setup7` uses a long-context prompt with `gemini25flashlite` as a neutral reference, `setup4` uses an unbiased-reference / RAG-response prompt, and `setup6`, `setup8`, `setup9`, `setup10`, `setup11`, and `setup12` stay on the default query-response prompt.
- The embedding-divergence baseline in `setup100` defaults to fitting its threshold during `touche-train` and reusing the saved state during `touche-embed-divergence`, with validation-based calibration as a fallback.
- Device resolution order is `cuda -> mps -> cpu`, unless the user explicitly forces a device.

## Output Contracts

### `touche-preprocess`

- Input: `data/task/responses-<split>.jsonl` + `data/task/responses-<split>-labels.jsonl`
- Output: `data/task/preprocessed/responses-<split>-merged.jsonl`

### `touche-generate-neutral`

- Adds a generated text field named from the model alias, for example `gemini25flashlite` or `qwen`.
- Preserves base fields and merged label metadata when available.
- Appends incrementally so interrupted runs can resume.

### `touche-train`

- Loads optional defaults from `train_model/<setup-name>.json` before applying CLI overrides.
- Saves either a Hugging Face model/tokenizer bundle or an embedding-divergence state bundle to `models/<setup-name>/`, depending on `trainer_type`.
- Writes `training_summary.json` next to the model bundle.
- Writes step/epoch monitoring events to `training_metrics.jsonl` for classifier training.
- Uses the full training split by default, with optional subset training through `--max-train-rows`.
- Supports alternative input formats such as the `setup7` long-context neutral-reference prompt and the `setup4` unbiased-reference / RAG-response prompt.
- Supports `none`, `linear`, and `cosine_with_warmup` schedulers plus optimizer controls such as weight decay, layerwise LR decay, and temporary embedding freezing.
- Supports W&B online logging and optional epoch-end validation monitoring for classifier training.

### `touche-validate`

- Loads optional defaults from `validate_model/<setup-name>.json` before applying CLI overrides.
- Delegates `setup100`-style embedding-divergence configs to the matching backend instead of forcing them through the classifier validator.
- Writes one `*-predictions.jsonl` file per evaluated dataset.
- Uses the `test` split by default unless explicit input files or `--eval-splits` are passed.
- Evaluates the main `response` field against `gold_label`.
- If a generated text field is present, also scores it and reports its positive-rate as a false-positive monitoring signal.
- Supports reference-aware validation presets such as `setup7` and `setup4`.

### `touche-embed-divergence`

- Loads optional defaults from `validate_model/setup100.json`.
- Embeds the neutral reference and the response separately with a sentence-embedding model.
- Computes a drift score either at the full-response level or by greedy sentence alignment.
- Uses a saved threshold from `models/<setup-name>/embedding_state.json` when available, and calibrates on validation data only when no saved threshold or manual `--threshold` is supplied.
- Writes the same family of metrics, confusion-matrix, prediction, and misclassification artifacts as the classifier validator.

## Current Assumptions

- The classification task remains binary: `0 = no ad`, `1 = ad`.
- The default training text format remains `Query: ... Response: ... Answer:`. `setup7` uses a long-context neutral-reference format, `setup4` uses an unbiased-reference / RAG-response format, and the other current local setups stay on the default prompt.
- The local setup matrix currently spans RoBERTa, Longformer, DeBERTa-v3 variants, ALBERT, ELECTRA, and DistilRoBERTa.
- `setup100` is a saved-state semantic-drift baseline rather than a fine-tuned classifier setup.
- Neutral-response generation currently supports Gemini plus a self-hosted OpenAI-compatible Qwen backend.

## Known Gaps

- No hosted OpenAI generation client is implemented yet for `data/generated/chatgpt/`.
- GPU-specific PyTorch wheel selection is documented, but not hardcoded into the project because that depends on the remote machine.
- End-to-end training was not run as part of this migration; lightweight utility tests cover the pure Python pieces.
