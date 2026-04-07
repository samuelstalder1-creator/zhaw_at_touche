# Models Directory

This directory stores training outputs.

## Output Shapes

### Classifier setups

Classifier runs usually create `models/<setup-name>/` containing:

- the Hugging Face model bundle
- the tokenizer files
- `training_summary.json`
- `training_metrics.jsonl`
- optional `wandb/` run files

### Embedding-divergence setups

Embedding-divergence runs usually create `models/<setup-name>/` containing:

- `embedding_state.json`
- `training_summary.json`

### Anchor-distance setup

The active anchor-distance baseline `setup110` usually creates
`models/setup110/` containing:

- `anchor_distance_classifier.pkl`
- `embedding_state.json`
- `training_summary.json`

### Archived experimental setups

Some documented setups such as `setup103` to `setup106` are currently archived
rather than first-class runnable backends. Their concepts are still documented
in `../setup.md`, but the corresponding model directories are not guaranteed to
exist locally.

## Current Repository State

The repository only commits placeholder or scaffold directories here:

- `models/setup6/`
- `models/setupX/`
- `models/setupY/`

Most real model bundles are expected to be produced locally and kept out of
version control.
