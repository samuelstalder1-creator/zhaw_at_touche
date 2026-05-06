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
- Anchor-distance baseline:
  - `src/zhaw_at_touche/anchor_distance_classifier.py`
  - `src/zhaw_at_touche/cli/anchor_distance_classifier.py`
  - `src/zhaw_at_touche/anchor_distance_threshold.py`
  - `src/zhaw_at_touche/cli/anchor_distance_threshold.py`
- Learned embedding-feature backends:
  - `src/zhaw_at_touche/embedding_lr_classifier.py`
  - `src/zhaw_at_touche/cli/embedding_lr_classifier.py`
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
  `setup7-qwen`, `setup8`, `setup9`, `setup10`, `setup11`, `setup12`,
  `setup115`, `setup116`
- Cross-encoder setups: `setup105`, `setup105_1`
- Learned embedding-feature setups: `setup103`, `setup103-qwen`,
  `setup103-gemma`, `setup104-base`, `setup104`, `setup104-qwen`,
  `setup113`, `setup114`, `setup117`, `setup118`, `setup119`,
  `setup120-qwen`, `setup120-gemma`, `setup121-qwen`, `setup121-gemma`
- Embedding-divergence setups: `setup100`, `setup101`, `setup102`
- Scalar anchor setups: `setup110`, `setup111`

### Documented but not currently wired end-to-end

- `setup106`

That sentence-delta descriptor remains in `train_model/` because it explains a
historical experiment and committed result directory, but the current
`touche-train` parser does not expose a sentence-delta backend.

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
- Learned embedding-feature runs write `embedding_lr_classifier.pkl`,
  `embedding_state.json`, and `training_summary.json`.
- Anchor-distance runs write `anchor_distance_classifier.pkl`,
  `embedding_state.json`, and `training_summary.json`.
- Anchor-distance threshold runs write `embedding_state.json` and
  `training_summary.json`.

### `touche-validate`

- Writes prediction JSONL files, metrics summaries, confusion matrices,
  standardized CSV exports, and misclassification exports.
- Delegates `setup100` to `setup102` to the embedding-divergence backend when
  the validation preset sets `scoring_backend=embedding_divergence`.
- Delegates `setup103`, `setup103-qwen`, `setup103-gemma`, `setup104-base`,
  `setup104`, `setup104-qwen`, `setup113`, `setup114`, `setup117`,
  `setup118`, `setup119`, `setup120-qwen`, `setup120-gemma`,
  `setup121-qwen`, and `setup121-gemma` to the embedding-LR backend when the
  validation preset sets the matching learned-feature scoring backend.
- Delegates `setup110` to the anchor-distance backend when the validation
  preset sets `scoring_backend=anchor_distance_classifier`.
- Delegates `setup111` to the handcrafted anchor-distance backend when the
  validation preset sets `scoring_backend=anchor_distance_threshold`.

## Key Setup Concepts

- Prompt shape: `response_only`, `query_response`,
  `query_neutral_response`, `query_dual_neutral_response`,
  `query_reference_rag_response`, `cross_encoder`
- Provider-specific neutral field: `gemini25flashlite` versus `qwen`
- Optimization controls: scheduler, warmup, weight decay, gradient clipping,
  layerwise LR decay, temporary embedding freezing
- Embedding scoring controls: distance metric, score granularity, sentence
  aggregation, threshold metric
- Multi-anchor distance features: query-vs-Gemini, query-vs-Qwen,
  Gemini-vs-Qwen, and response-vs-anchor distances used by `setup110` and
  `setup111`

`setup.md` is the canonical explanation of how those concepts map onto each
named setup.
