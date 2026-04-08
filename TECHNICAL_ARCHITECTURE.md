# Technical Architecture

This document explains how the repository is structured at the code level and
how the setup system fits into the package.

The canonical per-setup explanation is in `setup.md`. This file focuses on how
the code implements those ideas.

## Architecture Layers

1. Storage layer
   Inputs and outputs live in stable repository-relative folders such as
   `data/`, `train_model/`, `validate_model/`, `models/`, and `results/`.
2. Domain logic layer
   Shared Python logic lives in `src/zhaw_at_touche/`.
3. CLI layer
   Thin wrappers in `src/zhaw_at_touche/cli/` parse arguments, resolve setup
   defaults, and call shared functions.
4. Test layer
   Unit tests under `tests/` cover the pure Python pieces such as setup
   loading, metrics, overlap logic, and the saved-state embedding backends.

## Repository Structure

```text
.
├── data/
│   ├── task/
│   └── generated/
├── train_model/
├── validate_model/
├── models/
├── results/
├── src/zhaw_at_touche/
│   └── cli/
└── tests/
```

## Core Modules

### `constants.py`

Defines shared default directories, provider names, default model names, and
base schema fields.

### `jsonl.py`

Provides the low-level JSONL read/append/write helpers used across the entire
pipeline.

### `datasets.py`

Handles Touché-specific data preparation and prompt rendering.

The prompt layer currently supports:

- `response_only`
- `query_response`
- `query_neutral_response`
- `query_dual_neutral_response`
- `cross_encoder`
- `query_reference_rag_response`

Those formats map directly to the setup families documented in `setup.md`.

### `training_setups.py`

Loads `train_model/<setup-name>.json` and validates allowed training fields.

### `validation_setups.py`

Loads `validate_model/<setup-name>.json` and validates allowed evaluation
fields.

### `embedding_setups.py`

Loads defaults specific to embedding-divergence evaluation.

### `modeling.py`

Implements the fine-tuned classifier runtime:

- device resolution
- dataset loading and tokenization
- class weighting
- optimizer and scheduler construction
- training loop execution
- model loading and batch prediction

This is the runtime used by the classifier and cross-encoder setups:

- `setup4`
- `setup6`
- `setup6-qwen`
- `setup7`
- `setup7-qwen`
- `setup8`
- `setup9`
- `setup10`
- `setup11`
- `setup12`
- `setup105`
- `setup105_1`
- `setup115`
- `setup116`

### `embedding_divergence.py`

Implements the semantic-drift baselines:

- sentence splitting
- embedding extraction
- greedy sentence alignment
- cosine-distance scoring
- threshold calibration
- saved-state serialization

This module powers `setup100`, `setup101`, and `setup102`.

### `anchor_distance_classifier.py`

Implements the multi-anchor embedding baseline:

- merge Gemini and Qwen JSONL rows by `id`
- embed `query`, `response`, Gemini neutral, and Qwen neutral
- derive six response-level cosine-distance features
- fit a logistic regression over those distances
- save a reusable classifier bundle plus threshold/state metadata

This module powers `setup110`.

It also owns the shared JSONL row-merging helper used by the multi-file
embedding backends.

### `embedding_lr_classifier.py`

Implements the learned embedding-feature family:

- embed the configured text fields once with a frozen encoder
- build residual or stacked feature matrices such as
  `response_emb - neutral_emb` or `[query_emb | delta_gemini | delta_qwen]`
- fit a logistic regression over those features
- calibrate and save a threshold alongside the classifier bundle

This module powers `setup103`, `setup104`, `setup113`, `setup114`, `setup117`,
`setup118`, and `setup119`.

### `anchor_distance_threshold.py`

Implements the no-classifier multi-anchor baseline:

- merge Gemini and Qwen JSONL rows by `id`
- embed `query`, `response`, Gemini neutral, and Qwen neutral
- derive the same six response-level cosine-distance features as `setup110`
- compute `response_drift - anchor_cohesion`
- calibrate and save only a threshold/state bundle

This module powers `setup111`.

## CLI Entry Points

The root `pyproject.toml` exposes these command-line tools:

- `touche-preprocess`
- `touche-generate-neutral`
- `touche-train`
- `touche-validate`
- `touche-embed-divergence`
- `touche-predict`
- `touche-stats-data`
- `touche-stats-generated`
- `touche-check-overlap`
- `touche-eval-matrix`

## Setup-System Boundaries

### Supported by the current training CLI

- `trainer_type=classifier`
- `trainer_type=cross_encoder`
- `trainer_type=embedding_divergence`
- `trainer_type=embedding_residual_classifier`
- `trainer_type=embedding_classifier`
- `trainer_type=query_residual_classifier`
- `trainer_type=dual_residual_classifier`
- `trainer_type=dual_embedding_classifier`
- `trainer_type=query_dual_residual_classifier`
- `trainer_type=anchor_distance_classifier`
- `trainer_type=anchor_distance_threshold`

### Documented but not currently wired end-to-end

`train_model/setup106.json` describes the sentence-delta experiment, but its
trainer backend is not currently exposed by `touche-train`.

That is why the repo can simultaneously contain:

- committed result directories such as `results/setup106/`
- a setup JSON file for `setup106`
- no current CLI path that would retrain that idea from scratch

## End-To-End Flow

### Classifier path

1. preprocess or load generated JSONL data
2. resolve a training setup
3. render the chosen prompt format
4. fine-tune the transformer classifier
5. save the bundle under `models/<setup-name>/`
6. evaluate with `touche-validate`
7. write metrics and prediction artifacts under `results/<setup-name>/`

### Embedding-divergence path

1. load generated JSONL data with a neutral field
2. resolve an embedding-divergence setup
3. embed the response and neutral text
4. score response-level or sentence-level drift
5. fit a threshold and save `embedding_state.json`
6. evaluate with `touche-validate` or `touche-embed-divergence`

### Learned embedding-feature path

1. load one or two generated JSONL sources
2. optionally merge Gemini and Qwen rows by `id`
3. embed the configured text fields with a frozen encoder
4. build residual or stacked feature matrices
5. fit logistic regression and calibrate the threshold
6. evaluate with `touche-validate`

### Scalar anchor path

1. load paired Gemini and Qwen generated JSONL data
2. merge rows by `id`
3. embed `query`, `response`, Gemini neutral, and Qwen neutral
4. derive the six anchor-distance features
5. either fit a logistic regression (`setup110`) or use the handcrafted score
   `response_drift - anchor_cohesion` (`setup111`)
6. fit or reuse the calibrated threshold
7. evaluate with `touche-validate`

## Design Intent

- Keep all setup defaults in JSON rather than introducing a heavier config
  framework.
- Keep CLI adapters thin so the domain logic stays reusable.
- Keep artifacts explicit and file-based for easy inspection and archival.
- Separate canonical setup explanation (`setup.md`) from code-level architecture
  explanation (this file).
