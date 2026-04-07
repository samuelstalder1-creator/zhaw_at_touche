# Training Setup Files

This directory stores reusable JSON defaults for named training experiments.

The deep explanation of every setup and every concept lives in `../setup.md`.
This file focuses on how the JSON files interact with `touche-train`.

## How Loading Works

Each file is named `<setup-name>.json`.

```bash
uv run touche-train --setup-name setup6
```

The CLI first loads `train_model/setup6.json`, then applies any explicit CLI
overrides on top of it.

## Current Backend Support

### Supported By The Current CLI

| Trainer type | Setup names | Notes |
| --- | --- | --- |
| `classifier` | `setup4`, `setup6`, `setup6-qwen`, `setup7`, `setup7-qwen`, `setup8`, `setup9`, `setup10`, `setup11`, `setup12` | fine-tuned transformer classifiers |
| `embedding_divergence` | `setup100`, `setup101`, `setup102` | saved-state semantic-drift baselines |
| `anchor_distance_classifier` | `setup110` | saved-state logistic regression over six pairwise anchor distances |
| `anchor_distance_threshold` | `setup111` | saved-state handcrafted multi-anchor score with calibrated threshold |

### Archived JSON Descriptors

`setup103`, `setup104`, `setup105`, and `setup106` are documented in
`../setup.md`, but the current `touche-train` parser does not expose their
trainer backends. Their JSON files remain useful as research notes and for
explaining the committed `results/setup103/` and `results/setup104/` artifacts.

## Supported JSON Fields

The loader accepts these fields:

- `trainer_type`
- `train_file`
- `aux_train_file`
- `validation_file`
- `aux_validation_file`
- `model_name`
- `model_dir`
- `max_length`
- `epochs`
- `batch_size`
- `grad_accum`
- `learning_rate`
- `optimizer_eps`
- `weight_decay`
- `lr_scheduler`
- `warmup_ratio`
- `max_grad_norm`
- `gradient_checkpointing`
- `layerwise_lr_decay`
- `freeze_embeddings_epochs`
- `device`
- `max_train_rows`
- `input_format`
- `reference_field`
- `reference_label`
- `pad_to_max_length`
- `positive_class_weight_scale`
- `query_field`
- `response_field`
- `neutral_field`
- `aux_neutral_field`
- `distance_metric`
- `score_granularity`
- `sentence_agg`
- `sentence_delta_agg`
- `threshold_metric`
- `wandb_enabled`
- `wandb_project`
- `wandb_dir`
- `wandb_run_name`

Not every field is used by every backend.

## Common Commands

### Plain classifier baselines

```bash
uv run touche-train --setup-name setup6
uv run touche-train --setup-name setup10
uv run touche-train --setup-name setup12
```

### Reference-aware classifiers

```bash
uv run touche-train --setup-name setup4
uv run touche-train --setup-name setup7
uv run touche-train --setup-name setup7-qwen
```

### DeBERTa stabilization experiments

```bash
uv run touche-train --setup-name setup8
uv run touche-train --setup-name setup9
```

### Embedding-divergence baselines

```bash
uv run touche-train --setup-name setup100
uv run touche-train --setup-name setup101
uv run touche-train --setup-name setup102
```

### Anchor-distance baseline

```bash
uv run touche-train --setup-name setup110
uv run touche-train --setup-name setup111
```

These setups merge the Gemini and Qwen training files by `id` and report
separate progress bars for the `query`, `response`, Gemini-neutral, and
Qwen-neutral embedding passes. `setup110` learns a logistic regression over
the six cosine-distance features, while `setup111` uses a handcrafted score
and calibrates only the threshold.

### Subset training

```bash
uv run touche-train --setup-name setup6 --max-train-rows 1000
```

### Disable W&B

```bash
uv run touche-train --setup-name setup7 --no-wandb
```

## Output Contracts

### Classifier setups

Classifier runs write to `models/<setup-name>/`:

- Hugging Face model/tokenizer bundle
- `training_summary.json`
- `training_metrics.jsonl`
- `wandb/` run files by default

### Embedding-divergence setups

Embedding-divergence runs write to `models/<setup-name>/`:

- `embedding_state.json`
- `training_summary.json`

They do not write a Hugging Face classifier bundle, `training_metrics.jsonl`,
or a W&B run.

### Anchor-distance setups

`setup110` writes to `models/setup110/`:

- `anchor_distance_classifier.pkl`
- `embedding_state.json`
- `training_summary.json`

`setup111` writes to `models/setup111/`:

- `embedding_state.json`
- `training_summary.json`

## Notes About Providers

- Gemini-backed files are the default source for most setups.
- `setup6-qwen` and `setup7-qwen` switch the training and validation files to
  `data/generated/qwen/`.
- Only the reference-aware and embedding-based setups actually consume the
  neutral field as part of the model logic. Plain `query_response` classifiers
  still train on just the query and the response text.
