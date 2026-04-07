# Implementation Details

## Module Mapping

- Neutral-response generation:
  - `src/zhaw_at_touche/generation_utils.py`
  - `src/zhaw_at_touche/cli/generate_neutral.py`
- Classifier training and inference:
  - `src/zhaw_at_touche/modeling.py`
  - `src/zhaw_at_touche/cli/train_model.py`
  - `src/zhaw_at_touche/cli/validate_model.py`
  - `src/zhaw_at_touche/cli/manual_inference.py`
- Embedding-divergence baselines:
  - `src/zhaw_at_touche/embedding_divergence.py`
  - `src/zhaw_at_touche/embedding_setups.py`
  - `src/zhaw_at_touche/cli/embedding_divergence.py`
- Pairwise distance analysis:
  - `src/zhaw_at_touche/pairwise_distance.py`
  - `src/zhaw_at_touche/cli/pairwise_distances.py`
- Evaluation summaries:
  - `src/zhaw_at_touche/evaluation_utils.py`
  - `src/zhaw_at_touche/cli/evaluation_matrix.py`
- Supporting analysis:
  - `src/zhaw_at_touche/generated_stats.py`
  - `src/zhaw_at_touche/overlap_utils.py`
  - `src/zhaw_at_touche/cli/generated_stats.py`
  - `src/zhaw_at_touche/cli/check_overlap.py`

## Folder Decisions

- `data/task/` stores the official Touché files.
- `data/task/preprocessed/` stores merged response-plus-label JSONL files.
- `data/generated/gemini/` stores Gemini neutral rewrites.
- `data/generated/qwen/` stores Qwen neutral rewrites.
- `train_model/<setup-name>.json` stores named training defaults.
- `validate_model/<setup-name>.json` stores named evaluation defaults.
- `models/<setup-name>/` stores trained bundles or saved state.
- `results/<setup-name>/` stores evaluation artifacts.

## Current Setup Coverage

### Fully supported by the current CLI

- Classifier setups: `setup4`, `setup6`, `setup6-qwen`, `setup7`,
  `setup7-qwen`, `setup8`, `setup9`, `setup10`, `setup11`, `setup12`
- Embedding-divergence setups: `setup100`, `setup101`, `setup102`

### Documented but archived

- `setup103`, `setup104`, `setup105`, `setup106`

Those archived setup descriptors remain in `train_model/` because they explain
historical experiments and some committed result directories, but the current
`touche-train` parser only accepts `trainer_type=classifier` and
`trainer_type=embedding_divergence`.

## CLI Defaults

- Training defaults to the Gemini-enriched train split when that file exists,
  otherwise it falls back to the merged task file.
- Validation defaults to the Gemini-enriched test split when that file exists,
  otherwise it falls back to the merged task file.
- Device resolution order is `cuda -> mps -> cpu` unless the user forces a
  device.
- `--generated-provider qwen` switches the default evaluation files to
  `data/generated/qwen/` and, for reference-aware setups, also switches the
  default reference field and label.

## Output Contracts

### `touche-train`

- Classifier runs write a Hugging Face bundle plus `training_summary.json`,
  `training_metrics.jsonl`, and optional W&B run files.
- Embedding-divergence runs write `embedding_state.json` plus
  `training_summary.json`.

### `touche-validate`

- Writes prediction JSONL files, metrics summaries, confusion matrices,
  standardized CSV exports, and misclassification exports.
- Delegates `setup100` to `setup102` to the embedding-divergence backend when
  the validation preset sets `scoring_backend=embedding_divergence`.

### `touche-pairwise-distances`

- Merges one or more JSONL files by `id`.
- Computes response-level or sentence-level embedding distances for explicit
  field pairs.
- Writes JSONL, CSV, and summary JSON outputs under the chosen results
  directory.

## Key Setup Concepts

- Prompt shape: `query_response`, `query_neutral_response`,
  `query_reference_rag_response`
- Provider-specific neutral field: `gemini25flashlite` versus `qwen`
- Optimization controls: scheduler, warmup, weight decay, gradient clipping,
  layerwise LR decay, temporary embedding freezing
- Embedding scoring controls: distance metric, score granularity, sentence
  aggregation, threshold metric

`setup.md` is the canonical explanation of how those concepts map onto each
named setup.
