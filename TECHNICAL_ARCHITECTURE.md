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
   Unit tests under `tests/` cover the pure Python pieces such as setup loading,
   metrics, overlap logic, and pairwise-distance helpers.

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

- `query_response`
- `query_neutral_response`
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

This is the runtime used by the classifier setups:

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

### `anchor_distance_threshold.py`

Implements the no-classifier multi-anchor baseline:

- merge Gemini and Qwen JSONL rows by `id`
- embed `query`, `response`, Gemini neutral, and Qwen neutral
- derive the same six response-level cosine-distance features as `setup110`
- compute `response_drift - anchor_cohesion`
- calibrate and save only a threshold/state bundle

This module powers `setup111`.

### `pairwise_distance.py`

Implements field-pair distance analysis:

- merge multiple JSONL files by `id`
- compare arbitrary text fields such as `response`, `gemini25flashlite`, and
  `qwen`
- compute response-level or sentence-level distances
- summarize pairwise score distributions

Its row-merging helper is also reused by `setup110` and `setup111`.

## CLI Entry Points

The root `pyproject.toml` exposes these command-line tools:

- `touche-preprocess`
- `touche-generate-neutral`
- `touche-train`
- `touche-validate`
- `touche-embed-divergence`
- `touche-pairwise-distances`
- `touche-predict`
- `touche-stats-data`
- `touche-stats-generated`
- `touche-check-overlap`
- `touche-eval-matrix`

## Setup-System Boundaries

### Supported by the current training CLI

- `trainer_type=classifier`
- `trainer_type=embedding_divergence`
- `trainer_type=anchor_distance_classifier`
- `trainer_type=anchor_distance_threshold`

### Present as archived setup descriptors

`train_model/setup103.json` to `train_model/setup106.json` describe additional
research ideas, but their trainer backends are not currently exposed by
`touche-train`.

That is why the repo can simultaneously contain:

- committed result directories such as `results/setup103/` and `results/setup104/`
- setup JSON files for `setup103` to `setup106`
- no current CLI path that would retrain those ideas from scratch

The docs now call those setups archived rather than pretending they are
first-class runtime options.

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

### Anchor-distance path

1. load paired Gemini and Qwen generated JSONL data
2. merge rows by `id`
3. embed `query`, `response`, Gemini neutral, and Qwen neutral
4. derive the six pairwise anchor-distance features
5. either fit a logistic regression (`setup110`) or use the handcrafted score
   `response_drift - anchor_cohesion` (`setup111`)
6. fit or reuse the calibrated threshold
7. evaluate with `touche-validate`

### Pairwise-analysis path

1. load one or more JSONL files
2. merge them by `id`
3. compare explicit field pairs such as `response:qwen`
4. write per-record and aggregate distance summaries

## Design Intent

- Keep all setup defaults in JSON rather than introducing a heavier config
  framework.
- Keep CLI adapters thin so the domain logic stays reusable.
- Keep artifacts explicit and file-based for easy inspection and archival.
- Separate canonical setup explanation (`setup.md`) from code-level architecture
  explanation (this file).
