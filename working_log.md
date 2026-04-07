# Working Log

## Completed Work

- Reviewed the Touché 2025 task framing and the repository experiment history.
- Unified the workflow into one `uv`-managed Python package with shared CLI
  tooling for preprocessing, generation, training, validation, inference,
  overlap checks, results summaries, and pairwise field-distance analysis.
- Standardized the repository structure around `data/`, `train_model/`,
  `validate_model/`, `models/`, `results/`, and `src/zhaw_at_touche/`.
- Implemented preprocessing plus neutral-response generation for Gemini and a
  local Qwen backend.
- Implemented reusable setup loading for training and validation.
- Expanded the classifier family across RoBERTa, Longformer, DeBERTa-v3,
  ALBERT, ELECTRA, and DistilRoBERTa experiments.
- Added semantic-drift baselines `setup100`, `setup101`, and `setup102`.
- Preserved archived experiment descriptors `setup103` to `setup106` plus the
  committed `setup103` and `setup104` result artifacts.
- Added pairwise distance tooling to compare fields such as `response`,
  `gemini25flashlite`, and `qwen`.
- Rewrote the Markdown documentation so `setup.md` is now the canonical
  concepts-first setup reference and the other docs point back to it
  consistently.

## Future TODOs

- Train final candidate models on the target GPU environment and compare the
  remaining uncommitted setups such as `setup4`, `setup7`, `setup9`, and
  `setup11`.
- Decide whether to restore first-class training support for archived setup
  ideas `setup103` to `setup106` or remove those JSON descriptors from the
  active experiment surface.
- Retrain the DeBERTa-v3 variants with the stabilized settings and compare them
  against the strongest committed classifier baselines.
- Archive final submission-ready model bundles and validation artifacts.
- Build and smoke-test the final container or deployment packaging required for
  the Touché hand-in.
