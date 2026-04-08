# Models Directory

This directory stores training outputs.

## Output Shapes

### Classifier and cross-encoder setups

Classifier and cross-encoder runs usually create `models/<setup-name>/`
containing:

- the Hugging Face model bundle
- the tokenizer files
- `training_summary.json`
- `training_metrics.jsonl`
- optional `wandb/` run files

### Embedding-divergence setups

Embedding-divergence runs usually create `models/<setup-name>/` containing:

- `embedding_state.json`
- `training_summary.json`

### Learned embedding-feature setups

These runs usually create `models/<setup-name>/` containing:

- `embedding_lr_classifier.pkl`
- `embedding_state.json`
- `training_summary.json`

### Scalar anchor setups

`setup110` usually creates `models/setup110/` containing:

- `anchor_distance_classifier.pkl`
- `embedding_state.json`
- `training_summary.json`

`setup111` usually creates `models/setup111/` containing:

- `embedding_state.json`
- `training_summary.json`

### Descriptor-only historical setup

`setup106` is still documented in `../setup.md`, but its sentence-delta
backend is not currently wired into the CLI. Historical model directories for
that setup are therefore not guaranteed to exist locally.

## Current Repository State

The repository only commits placeholder or scaffold directories here:

- `models/setup6/`
- `models/setupX/`
- `models/setupY/`

Most real model bundles are expected to be produced locally and kept out of
version control.
