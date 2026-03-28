# Working Log

## Completed Work

- Unified the Touché workflow into one `uv`-managed Python package with shared
  CLI tooling for preprocessing, generation, training, validation, inference,
  overlap checks, and stats.
- Standardized the repository structure around `data/`, `train_model/`,
  `validate_model/`, `models/`, `results/`, and `src/zhaw_at_touche/`.
- Implemented preprocessing and Gemini neutral-response generation for the
  dataset pipeline.
- Implemented PyTorch-based classifier training with reusable setup presets,
  configurable prompt formats, class weighting, validation during training, and
  W&B monitoring.
- Implemented validation/export tooling for metrics, confusion matrices,
  prediction files, and misclassification analysis.
- Added utility/test coverage for configuration loading, metrics, overlap
  checks, generated-data stats, and dataset helpers.

## Future TODOs

- Train final candidate models on the target GPU environment and compare the key
  setups on validation and test performance.
- Select the final submission checkpoint and rerun validation to archive the
  final results.
- Build a Docker container with the trained model for the Touché challenge
  hand-in.
- Define the container inference contract: runtime inputs, model path, and
  prediction command/service behavior.
- Add a clean-machine smoke test for the container and freeze the final runtime
  dependencies.
- Write a short submission README covering image build, prediction usage, and
  required runtime files.
- Verify the exact Touché submission requirements and check the final hand-in
  package against them.
