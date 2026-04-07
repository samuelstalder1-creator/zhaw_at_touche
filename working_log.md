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
- Added multi-anchor embedding baseline `setup110` (logistic regression on 6
  cosine scalars from Gemini + Qwen neutrals).
- Preserved archived experiment descriptors `setup103` to `setup106` plus the
  committed `setup103` and `setup104` result artifacts.
- Added pairwise distance tooling to compare fields such as `response`,
  `gemini25flashlite`, and `qwen`.
- Analysed `setup105` collapse: all-positive predictions caused by DeBERTa-v3
  instability and a missing `validate_model/setup105.json` that caused the
  cross-encoder to be evaluated with the wrong input format.
- Added `setup105_1`: cross-encoder retry with RoBERTa-base,
  `positive_class_weight_scale=1.5`, linear scheduler, and a correct
  `validate_model/setup105_1.json`.
- Analysed `setup110` failure: 6 cosine scalars discard the directional
  information in the delta vector that makes `setup103` work.
- Added `setup113` (dual residual: `[delta_gemini | delta_qwen]`) and
  `setup114` (full dual stack: `[R | G | Q | delta_G | delta_Q]`) as
  dual-provider extensions of `setup103` and `setup104`.
- Simplified `results.md` to Macro F1 + confusion matrix only; added newly
  committed results for `setup7-qwen`, `setup106`, `setup110`, and `setup105`.
- Rewrote `setup.md` around experiment families and the core research questions
  rather than per-setup descriptions with embedded result snapshots.

## Open Questions

- Does injecting the neutral reference into the prompt (`setup7`) beat plain
  `query_response` classifiers (`setup6`)? Setup7 has no committed results.
- Does the cross-encoder (`setup105_1`) outperform bi-encoder family 3 setups
  (`setup103`, `setup104`) when trained on a stable backbone?
- Do dual-provider neutrals (`setup113`, `setup114`) improve over single-provider
  residuals (`setup103`, `setup104`)?
- Is DeBERTa-v3 actually better than RoBERTa for this task once training is
  stabilized (`setup8`, `setup9`)?

## Future TODOs

- Implement training backends for `dual_residual_classifier` (setup113),
  `dual_embedding_classifier` (setup114), and `cross_encoder` (setup105_1).
- Run `setup105_1`, `setup113`, `setup114` and commit results.
- Run remaining uncommitted classifier setups: `setup4`, `setup7`, `setup9`, `setup11`.
- Archive final submission-ready model bundles and validation artifacts.
- Build and smoke-test the final container or deployment packaging for the
  Touché hand-in.
