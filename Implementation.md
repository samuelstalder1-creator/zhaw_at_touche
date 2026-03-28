# Implementation Overview

- The repository now uses one root `uv` project instead of three standalone Python repos.
- Shared logic lives in `src/zhaw_at_touche/` and is exposed through CLI entry points.
- Raw task data lives in `data/task/`.
- Generated neutral responses live in `data/generated/<provider>/`.
- Named training defaults live in `train_model/<setup-name>.json`.
- Evaluation-only defaults live in `validate_model/<setup-name>.json`.
- Trained models live in `models/<setup-name>/`.
- Validation artifacts live in `results/<setup-name>/`.

## Workflow

1. `touche-preprocess` merges response rows with label rows.
2. `touche-generate-neutral` creates neutral responses with Gemini.
3. `touche-train` trains a binary classifier on the full training split by default and supports optional subset training.
4. `touche-validate` evaluates either a saved local model bundle or an evaluation-only remote model setup, using the test split by default and optionally validation plus test.
5. `touche-predict` supports manual inference for custom text.
6. `touche-stats-data` and `touche-stats-generated` provide dataset summaries, token analysis, and histograms.
7. `touche-check-overlap` reports split leakage across train, validation, and test response files.
8. `touche-eval-matrix` preserves the old `run_evaluation` summary flow.
