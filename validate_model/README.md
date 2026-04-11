# Validation Setup Files

This directory stores reusable defaults for evaluation-only runs.

Use these files when you want `touche-validate` to know where the model lives,
which files to evaluate, and whether the run should stay on the normal
classifier path or delegate to one of the saved-state embedding backends.

The deep explanation of each experiment lives in `../setup.md`.

## How Loading Works

Each file is named `<setup-name>.json`.

```bash
uv run touche-validate --setup-name teamCMU
```

The CLI loads the matching JSON file first and then applies explicit CLI
overrides.

## Supported JSON Fields

- `model_name`
- `model_dir`
- `results_dir`
- `eval_splits`
- `input_files`
- `aux_input_files`
- `calibration_input_files`
- `aux_calibration_input_files`
- `generated_provider`
- `generated_field`
- `text_field`
- `input_format`
- `reference_field`
- `reference_label`
- `aux_reference_field`
- `aux_reference_label`
- `reference_label_1`
- `reference_label_2`
- `pad_to_max_length`
- `max_length`
- `batch_size`
- `threshold`
- `threshold_metric`
- `device`
- `scoring_backend`
- `embedding_model_name`
- `query_field`
- `response_field`
- `neutral_field`
- `aux_neutral_field`
- `distance_metric`
- `score_granularity`
- `sentence_agg`

## Current Preset Inventory

| Preset | Purpose | Notes |
| --- | --- | --- |
| `teamCMU` | remote evaluation-only preset | loads a published Hugging Face model |
| `setup4` | reference-aware classifier validation | mirrors the DeBERTa reference + RAG prompt |
| `setup6` | classifier validation | plain classifier path using default Gemini-backed files |
| `setup6-qwen` | provider-specific classifier validation | defaults to the Qwen-enriched test file |
| `setup7` | reference-aware long-context validation | uses Gemini neutral context |
| `setup7-qwen` | provider-specific long-context validation | uses Qwen neutral context |
| `setup8` | classifier validation | plain classifier path |
| `setup9` | stabilized DeBERTa validation | plain classifier path |
| `setup10` | ALBERT validation | plain classifier path |
| `setup11` | ELECTRA validation | plain classifier path |
| `setup12` | DistilRoBERTa validation | plain classifier path |
| `setup105_1` | cross-encoder validation | stable RoBERTa retry over response + neutral |
| `setup115` | response-only classifier validation | plain classifier path with no query and no neutral |
| `setup116` | dual-neutral prompted classifier validation | plain classifier path with paired Gemini + Qwen files |
| `setup100` | embedding-divergence validation | delegates to `touche-embed-divergence` backend |
| `setup101` | embedding-divergence validation | delegates to `touche-embed-divergence` backend |
| `setup102` | embedding-divergence validation | delegates to `touche-embed-divergence` backend |
| `setup103`, `setup104`, `setup117`, `setup119` | learned embedding-feature validation | delegates to the embedding-LR backend |
| `setup113`, `setup114`, `setup118` | dual-file learned embedding-feature validation | delegates to the embedding-LR backend and merges Gemini + Qwen rows by `id` |
| `setup110` | anchor-distance validation | delegates to the anchor-distance backend and merges Gemini + Qwen rows by `id` |
| `setup111` | anchor-distance threshold validation | delegates to the handcrafted anchor-distance backend and merges Gemini + Qwen rows by `id` |

Historical descriptor still present in `validate_model/`:

- `setup106`

There is currently no dedicated `validate_model/setup105.json` preset. When
`setup105` is evaluated, it relies on the default model/results path
resolution or explicit CLI arguments instead.

## Common Commands

### Local classifier evaluation

```bash
uv run touche-validate --setup-name setup6
uv run touche-validate --setup-name setup12
```

### Reference-aware evaluation

```bash
uv run touche-validate --setup-name setup4
uv run touche-validate --setup-name setup7
uv run touche-validate --setup-name setup7-qwen
uv run touche-validate --setup-name setup115
uv run touche-validate --setup-name setup116
```

### Provider-specific evaluation on Qwen-generated files

```bash
uv run touche-validate --setup-name setup6 --generated-provider qwen
uv run touche-validate --setup-name setup6-qwen
```

When `--generated-provider qwen` is used and `--results-dir` is omitted, the
validator writes to `results/<setup-name>-qwen/` so the Qwen-backed evaluation
does not overwrite the default Gemini-backed artifacts.

For reference-aware setups such as `setup7`, the validator also switches the
default `reference_field` and `reference_label` from Gemini to Qwen unless you
override them manually.

### Embedding-divergence validation

```bash
uv run touche-train --setup-name setup100
uv run touche-validate --setup-name setup100

uv run touche-train --setup-name setup101
uv run touche-validate --setup-name setup101

uv run touche-train --setup-name setup102
uv run touche-validate --setup-name setup102
```

These presets delegate to the embedding-divergence backend automatically. The
validator reuses the saved `embedding_state.json` threshold when it exists and
only recalibrates on validation data if no saved or manual threshold is
available.

### Learned embedding-feature validation

```bash
uv run touche-train --setup-name setup103
uv run touche-validate --setup-name setup103

uv run touche-train --setup-name setup113
uv run touche-validate --setup-name setup113

uv run touche-train --setup-name setup118
uv run touche-validate --setup-name setup118
```

These presets delegate to the embedding-LR backend automatically. Single-file
variants read one generated JSONL source. Dual-file variants merge paired
Gemini and Qwen files by `id` before embedding and scoring.

### Anchor-distance validation

```bash
uv run touche-train --setup-name setup110
uv run touche-validate --setup-name setup110

uv run touche-train --setup-name setup111
uv run touche-validate --setup-name setup111
```

These presets merge the Gemini and Qwen files by `id` and compute the same six
response-level cosine distances. `setup110` applies the saved
logistic-regression bundle plus calibrated threshold from `models/setup110/`.
`setup111` uses the handcrafted score `response_drift - anchor_cohesion` and
reuses the calibrated threshold from `models/setup111/`.

Unlike `--generated-provider qwen` on the classifier path, `setup110` and
`setup111` do not swap between providers. They need both paired file families
at the same time.

## Output Contracts

Validation writes to `results/<setup-name>/` unless a preset or CLI override
changes the directory. Typical outputs are:

- `metrics_summary.json`
- `response_metrics.txt`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `standardized_results.csv`
- `misclassified_analysis.csv`
- `*-predictions.jsonl`
