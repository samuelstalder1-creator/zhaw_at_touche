# Implementation Details

## Consolidation

- `neutral_response_generator/main.py` was migrated into:
  - `src/zhaw_at_touche/generation_utils.py`
  - `src/zhaw_at_touche/cli/generate_neutral.py`
- `neutral_response_generator/analyze_token_counts.py` was migrated into:
  - `src/zhaw_at_touche/generated_stats.py`
  - `src/zhaw_at_touche/cli/generated_stats.py`
- `neutral_response_generator/check_dataset_overlap.py` was migrated into:
  - `src/zhaw_at_touche/overlap_utils.py`
  - `src/zhaw_at_touche/cli/check_overlap.py`
- `dagmar/setup6/main.py` was split into:
  - `src/zhaw_at_touche/modeling.py`
  - `src/zhaw_at_touche/cli/train_model.py`
  - `src/zhaw_at_touche/cli/validate_model.py`
  - `src/zhaw_at_touche/cli/manual_inference.py`
- `run_evaluation/main.py` was migrated into:
  - `src/zhaw_at_touche/evaluation_utils.py`
  - `src/zhaw_at_touche/cli/evaluation_matrix.py`
- The legacy `run_evaluation/` folder was removed because its evaluation tooling now lives under `src/zhaw_at_touche/`.
- The legacy `neutral_response_generator/` folder was removed because its generator, stats, and overlap tooling now live under `src/zhaw_at_touche/`.

## Folder Decisions

- `data/task/` is the canonical home for official dataset files.
- `data/task/preprocessed/` contains merged response + label files created by `touche-preprocess`.
- `data/generated/gemini/` stores generated neutral-response datasets.
- `data/generated/chatgpt/` is intentionally present even though no OpenAI generation backend is implemented yet.
- `train_model/<setup-name>.json` stores reusable defaults for named training experiments such as `setup6`.
- `models/<setup-name>/` stores saved Hugging Face model bundles plus `training_summary.json`.
- `results/<setup-name>/` stores validation artifacts and prediction exports.

## CLI Defaults

- Training defaults to `data/generated/gemini/responses-train-with-neutral_gemini.jsonl` if present.
- If generated files are missing, training falls back to `data/task/preprocessed/responses-train-merged.jsonl`.
- Validation defaults to the generated Gemini `test` and `validation` files, with the same fallback to preprocessed merged files.
- Device resolution order is `cuda -> mps -> cpu`, unless the user explicitly forces a device.

## Output Contracts

### `touche-preprocess`

- Input: `data/task/responses-<split>.jsonl` + `data/task/responses-<split>-labels.jsonl`
- Output: `data/task/preprocessed/responses-<split>-merged.jsonl`

### `touche-generate-neutral`

- Adds a generated text field named from the model alias, for example `gemini25flashlite`.
- Preserves base fields and merged label metadata when available.
- Appends incrementally so interrupted runs can resume.

### `touche-train`

- Loads optional defaults from `train_model/<setup-name>.json` before applying CLI overrides.
- Saves the Hugging Face model/tokenizer bundle to `models/<setup-name>/`.
- Writes `training_summary.json` next to the model bundle.

### `touche-validate`

- Writes one `*-predictions.jsonl` file per evaluated dataset.
- Evaluates the main `response` field against `gold_label`.
- If a generated text field is present, also scores it and reports its positive-rate as a false-positive monitoring signal.

## Current Assumptions

- The classification task remains binary: `0 = no ad`, `1 = ad`.
- The training text format remains `Query: ... Response: ... Answer:`.
- Gemini is the only implemented neutral-generation backend in this migration.

## Known Gaps

- No OpenAI generation client is implemented yet for `data/generated/chatgpt/`.
- GPU-specific PyTorch wheel selection is documented, but not hardcoded into the project because that depends on the remote machine.
- End-to-end training was not run as part of this migration; lightweight utility tests cover the pure Python pieces.
