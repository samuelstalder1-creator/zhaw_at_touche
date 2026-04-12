# Working Log

## Completed Work

- Reviewed the Touché 2025 task framing and the repository experiment history.
- Unified the workflow into one `uv`-managed Python package with shared CLI
  tooling for preprocessing, generation, training, validation, inference,
  overlap checks, and results summaries.
- Standardized the repository structure around `data/`, `train_model/`,
  `validate_model/`, `models/`, `results/`, and `src/zhaw_at_touche/`.
- Implemented preprocessing plus neutral-response generation for Gemini and a
  local Qwen backend.
- Implemented reusable setup loading for training and validation.
- Expanded the classifier family across RoBERTa, Longformer, DeBERTa-v3,
  ALBERT, ELECTRA, and DistilRoBERTa experiments.
- Added `setup115` (response-only RoBERTa) and `setup116` (dual-neutral
  Longformer) to complete the classifier-side ablation matrix.
- Added semantic-drift baselines `setup100`, `setup101`, and `setup102`.
- Activated learned embedding-feature backends for `setup103`, `setup104`,
  `setup113`, `setup114`, `setup117`, `setup118`, and `setup119`.
- Added scalar multi-anchor baseline `setup110` (logistic regression on 6
  cosine scalars from Gemini + Qwen neutrals).
- Kept `setup106` as the remaining descriptor-only historical experiment.
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
- Added `setup117` and `setup118` to test whether query embeddings sharpen the
  learned delta signal, plus `setup119` as the Qwen-only residual counterpart
  to `setup103`.
- Removed the standalone pairwise-distance tooling and kept only the shared
  JSONL merge helper used by the multi-file embedding backends.
- Simplified `results.md` to Macro F1 + confusion matrix only; added newly
  committed results for `setup7-qwen`, `setup106`, `setup110`, and `setup105`.
- Added and documented later committed results for `setup7`, `setup114`,
  `setup115`, `setup116`, `setup117`, `setup118`, and `setup119`, then
  refreshed the repository docs to align the inventories, result summaries,
  and research-progress tracking.
- Rewrote `setup.md` around experiment families and the core research questions
  rather than per-setup descriptions with embedded result snapshots.
