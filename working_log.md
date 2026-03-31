# Working Log

## Completed Work

- Read the papers from the 2025 Touché challenge and reviewed participant
  approaches, including teams such as `teamCMU`.
- Built up project experimentation across multiple environments, including
  earlier work in Google Colab and later training-oriented setup on ZHAW
  OpenShift sources.
- Unified the Touché workflow into one `uv`-managed Python package with shared
  CLI tooling for preprocessing, generation, training, validation, inference,
  overlap checks, and stats.
- Standardized the repository structure around `data/`, `train_model/`,
  `validate_model/`, `models/`, `results/`, and `src/zhaw_at_touche/`.
- Merged the previously separate files and workstreams into this single
  repository, `zhaw_at_touche`, to form one full ML development pipeline from
  preprocessing to training, evaluation, and planned Docker deployment.
- Implemented preprocessing and a Gemini-based neutral-response generation
  pipeline for the dataset workflow.
- Extended neutral-response generation to support a self-hosted Qwen backend
  through an OpenAI-compatible local API.
- Built and compared different model setups, including `roberta-base` and
  DeBERTa-based configurations.
- Analyzed token distributions in the data and generated-response outputs.
- Developed a setup using neutral padding / padding-focused configuration for
  model-input experiments.
- Implemented PyTorch-based classifier training with reusable setup presets,
  configurable prompt formats, class weighting, validation during training, and
  W&B monitoring.
- Expanded the local experiment matrix with stabilized DeBERTa-v3, ALBERT,
  ELECTRA, and DistilRoBERTa presets plus matching validation presets.
- Extended the training loop with linear scheduling, explicit weight decay,
  layerwise LR decay, and optional embedding freezing for architecture-specific
  fine-tuning experiments.
- Added `setup100`, an embedding-divergence baseline that fits and saves a
  threshold/state bundle during training and reuses it during evaluation
  instead of fine-tuning a classifier.
- Implemented validation/export tooling for metrics, confusion matrices,
  prediction files, and misclassification analysis.
- Added utility/test coverage for configuration loading, metrics, overlap
  checks, generated-data stats, and dataset helpers.

## Future TODOs

- Train final candidate models on the target GPU environment and compare the key
  setups on validation and test performance.
- Analyze whether `setup9` actually fixes the DeBERTa-v3 instability seen in
  `setup4` and `setup8`.
- Train and compare the newly added ALBERT, ELECTRA, and DistilRoBERTa setup
  alternatives against the RoBERTa and Longformer baselines.
- Compare `setup100` against the trained classifier baselines to see whether
  semantic drift scoring can surface ad insertions with fewer model-specific
  training assumptions.
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
