# Working Log

Last updated: 2026-03-28

## Project Context

This repository contains the Touché ad-detection workflow in one shared Python
3.12 project managed with `uv`. The codebase is organized around CLI commands
for preprocessing, neutral-response generation, model training, validation,
manual inference, overlap checks, and dataset statistics.

## Completed Work

- Consolidated the project into a single package-oriented repository with shared
  logic under `src/zhaw_at_touche/` and thin CLI entrypoints.
- Standardized the folder layout for raw data, generated data, training setups,
  validation setups, trained models, and validation results.
- Implemented preprocessing to merge Touché response files with label files into
  easier-to-consume JSONL datasets.
- Implemented neutral-response generation for Gemini-based reference data.
- Implemented the classifier training workflow with Hugging Face
  sequence-classification models, a PyTorch training loop, automatic device
  selection (`cuda -> mps -> cpu`), class weighting for label imbalance, and
  support for alternative input formats such as the `setup7`
  neutral-reference prompt.
- Implemented model validation and result export, including confusion matrices,
  metrics summaries, standardized CSV output, and misclassification analysis.
- Implemented manual inference tooling for quick local testing of trained
  classifiers.
- Implemented utility tooling for overlap checks, evaluation-matrix generation,
  and generated-data statistics.
- Added reusable JSON setup files for training and validation presets such as
  `setup6`, `setup7`, and `teamCMU`.
- Added lightweight unit tests for pure utility code and configuration loading.
- Standardized online monitoring with Weights & Biases during training.
- Configured training to evaluate the validation dataset at the end of every
  epoch when a validation file is present.
- Extended W&B monitoring to include validation loss and accuracy,
  positive-class precision/recall/F1, macro and weighted aggregate metrics,
  confusion-count monitoring (`tn`, `fp`, `fn`, `tp`), positive-rate
  monitoring, and best-validation summary tracking.
- Added repository ignore rules so derived model artifacts under `models/` are
  not picked up unintentionally.

## Recent Practical Decisions

- Confirmed that the ML stack in this repository is PyTorch-based, not
  TensorFlow-based.
- Removed the TensorBoard path from the training workflow and kept W&B online as
  the primary training monitor.
- Kept validation as an in-training evaluation mechanism rather than a separate
  manual step only after training.
- Kept model artifacts out of version control and left the repository focused on
  code, configs, and reproducible workflow definitions.

## Current Repository State

- The repository worktree is currently clean.
- There is no committed trained model bundle in `models/setup6/` at the moment;
  the directory currently contains only the placeholder file.
- The project documentation covers setup, training, validation, and monitoring,
  but the final challenge hand-in packaging is still open work.

## Known Gaps / Open Items

- No OpenAI-based neutral-response generation backend is implemented yet for the
  reserved `data/generated/chatgpt/` path.
- End-to-end final-model training and final submission packaging are not yet
  captured in a reproducible hand-in workflow inside the repository.
- Containerization for challenge delivery is not implemented yet.

## Future TODOs

- Train final candidate models on the intended remote GPU environment and record
  the decisive W&B runs.
- Compare the main training setups (`setup6`, `setup7`, and any follow-up
  experiments) using validation and test metrics.
- Decide which checkpoint should be treated as the final submission model.
- Re-run validation on the final checkpoint and archive the final result files
  needed for submission.
- Build a Docker container that packages the trained model and the required
  inference entrypoint for the Touché challenge hand-in.
- Define exactly how the container should be started in the challenge
  environment, including the input format, model loading path, and prediction
  command or service contract.
- Add a container-focused smoke test on a clean machine to verify that the
  model, tokenizer, and runtime dependencies load correctly without local repo
  assumptions.
- Freeze the final runtime dependency set for the hand-in container and verify
  reproducibility.
- Write a short submission-oriented README describing how to build the Docker
  image, how to run predictions, and what files are expected at runtime.
- Confirm the exact Touché submission requirements and map them against the
  container output contract before hand-in.
- Prepare a final checklist covering trained model availability, validation
  evidence, container build success, local inference smoke test, and submission
  archive contents.

## Suggested Next Step

The next high-value task is to define the Docker hand-in contract clearly, then
package one trained model bundle plus a minimal inference entrypoint into a
reproducible container image.
