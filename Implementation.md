# Implementation Overview

- The repository uses one root `uv` project with shared code and CLI entry points.
- Shared logic lives in `src/zhaw_at_touche/` and is exposed through CLI entry points.
- Raw task data lives in `data/task/`.
- Generated neutral responses live in `data/generated/<provider>/`.
- Named training defaults live in `train_model/<setup-name>.json`.
- Evaluation-only defaults live in `validate_model/<setup-name>.json`.
- Trained models live in `models/<setup-name>/`.
- Validation artifacts live in `results/<setup-name>/`.

## Workflow

1. `touche-preprocess` merges response rows with label rows.
2. `touche-generate-neutral` creates neutral responses with either Gemini or a self-hosted OpenAI-compatible Qwen backend.
3. `touche-train` trains a binary classifier on the full training split by default, supports optional subset training, can switch input formats for setups such as `setup7` and `setup4`, exposes optimizer controls such as weight decay and scheduler selection, and writes local monitoring logs.
4. `touche-validate` evaluates either a saved local model bundle, an evaluation-only remote model setup, or backend-specific setups such as `setup100` that delegate to the embedding-divergence validator.
5. `touche-embed-divergence` evaluates the embedding-divergence baseline directly, normally using the threshold/state first fit and saved by `touche-train --setup-name setup100`, and falls back to validation calibration when needed.
6. `touche-predict` supports manual inference for custom text.
7. `touche-stats-data` and `touche-stats-generated` provide dataset summaries, token analysis, and histograms.
8. `touche-check-overlap` reports split leakage across train, validation, and test response files.
9. `touche-eval-matrix` generates a confusion-matrix summary from existing prediction files.

The current named experiment family covers RoBERTa, Longformer, several DeBERTa-v3 variants, ALBERT, ELECTRA, DistilRoBERTa, and the embedding-divergence `setup100`.
