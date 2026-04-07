# Implementation Overview

- The repository uses one root `uv` project with shared CLI entry points.
- Shared logic lives in `src/zhaw_at_touche/`.
- Named experiment defaults live in `train_model/` and `validate_model/`.
- Trained outputs live in `models/`.
- Evaluation outputs live in `results/`.

## Workflow

1. `touche-preprocess` merges the raw response and label files.
2. `touche-generate-neutral` creates neutral rewrites with Gemini or a locally
   loaded Qwen backend.
3. `touche-train` runs either a fine-tuned classifier setup or an
   embedding-based setup such as embedding divergence or anchor distance.
4. `touche-validate` evaluates a saved local model bundle, a remote preset such
   as `teamCMU`, or delegates embedding-divergence and anchor-distance setups
   to their dedicated backends.
5. `touche-embed-divergence` runs the semantic-drift validator directly.
6. `touche-pairwise-distances` computes explicit embedding distances between
   chosen text fields such as `response`, `gemini25flashlite`, and `qwen`.
7. `touche-predict` supports manual inference for one-off examples.
8. `touche-stats-data`, `touche-stats-generated`, `touche-check-overlap`, and
   `touche-eval-matrix` provide supporting analysis utilities.

## Setup Families

- Fine-tuned classifier setups: `setup4`, `setup6`, `setup6-qwen`, `setup7`,
  `setup7-qwen`, `setup8`, `setup9`, `setup10`, `setup11`, `setup12`
- Embedding-divergence setups: `setup100`, `setup101`, `setup102`
- Anchor-distance setup: `setup110`
- Archived experimental descriptors kept for documentation and historical
  results: `setup103`, `setup104`, `setup105`, `setup106`

The canonical setup explanation is in `setup.md`.
