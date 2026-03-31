# Technical Architecture

## Purpose

This document explains the technical implementation of the `zhaw_at_touche` repository, how the code is structured, and how the main training and evaluation workflow is executed.

The project is organized as one Python package with multiple CLI entry points:

- preprocessing and dataset utilities
- neutral-response generation
- model training, validation, inference, and evaluation summaries

The package is built as a single `uv` project and exposes command-line tools through the root [`pyproject.toml`](pyproject.toml).

## System Overview

At a high level, the repository is a file-oriented machine learning workflow for binary ad detection in Touché response data.

The architecture has four layers:

1. Storage layer
   Raw, generated, trained, and evaluated artifacts live in predictable folders under `data/`, `models/`, `results/`, `train_model/`, and `validate_model/`.
2. Domain logic layer
   Shared logic lives in `src/zhaw_at_touche/` and implements JSONL I/O, dataset merging, generation helpers, training/inference, metrics, and analysis utilities.
3. CLI layer
   Thin wrappers in `src/zhaw_at_touche/cli/` parse arguments, resolve defaults, call shared functions, and write artifacts.
4. Test layer
   Lightweight unit tests under `tests/` cover the pure Python parts of the system such as config loading, metrics, overlap analysis, and generated-stats behavior.

## Repository Structure

```text
.
├── data/
│   ├── task/                # canonical Touché datasets and labels
│   └── generated/           # neutral-response outputs by provider
├── train_model/             # reusable JSON defaults for named experiments
├── validate_model/          # reusable JSON defaults for evaluation-only experiments
├── models/                  # saved Hugging Face model bundles
├── results/                 # validation outputs and evaluation artifacts
├── src/zhaw_at_touche/      # shared package code
│   └── cli/                 # command-line entrypoints
└── tests/                   # utility-focused unit tests
```

This layout deliberately separates immutable inputs from derived artifacts:

- `data/task/` contains source datasets and labels.
- `data/task/preprocessed/` contains merged response-plus-label files created from raw inputs.
- `data/generated/<provider>/` contains neutral-response files enriched with generated text fields.
- `validate_model/<setup-name>.json` contains evaluation-only presets for already-trained models.
- `models/<setup-name>/` contains trained classifier bundles and a `training_summary.json`.
- `results/<setup-name>/` contains prediction exports, confusion matrices, metrics, and analysis CSVs.

## Packaging And Entry Points

The root [`pyproject.toml`](pyproject.toml) defines the project and exposes these console scripts:

- `touche-preprocess`
- `touche-generate-neutral`
- `touche-train`
- `touche-validate`
- `touche-embed-divergence`
- `touche-predict`
- `touche-stats-data`
- `touche-stats-generated`
- `touche-check-overlap`
- `touche-eval-matrix`

Each CLI module is intentionally small. The pattern is consistent across the codebase:

1. parse CLI arguments
2. resolve default paths and runtime settings
3. call shared utility functions from `src/zhaw_at_touche/`
4. write files and print a compact summary

This keeps business logic reusable and testable instead of embedding it inside argument-parsing code.

## Core Code Modules

### `constants.py`

[`src/zhaw_at_touche/constants.py`](src/zhaw_at_touche/constants.py) defines shared defaults:

- repository-relative directories such as `data/task`, `models`, and `results`
- default dataset splits and task categories
- default setup and provider names
- the default Gemini model name
- the set of base fields used to distinguish generated text fields from existing schema fields

This file is the central place for directory conventions and default naming.

### `jsonl.py`

[`src/zhaw_at_touche/jsonl.py`](src/zhaw_at_touche/jsonl.py) provides the low-level file abstraction used throughout the repository:

- `read_jsonl`
- `append_jsonl`
- `write_jsonl`
- `count_jsonl_rows`

The entire pipeline uses JSONL as the exchange format between steps. That keeps each stage simple, streaming-friendly, and easy to inspect manually.

### `datasets.py`

[`src/zhaw_at_touche/datasets.py`](src/zhaw_at_touche/datasets.py) contains dataset-specific transformations:

- loading label maps from label JSONL files
- merging response rows with matching label rows
- auto-detecting generated-response fields in enriched files
- normalizing text and counting words
- building the classifier prompt format

Three input formats are currently supported:

- the default `Query: ... Response: ... Answer:` prompt, used by setups such as `setup6`, `setup8`, `setup9`, `setup10`, `setup11`, and `setup12`
- a neutral-reference format used by `setup7`, where query, Gemini neutral reference, and target response are combined into one long context
- a reference-plus-RAG format used by `setup4`, where query, unbiased reference, and RAG response are rendered with an explicit advertisement-labeling instruction

The classifier input is intentionally standardized as:

```text
Query: <query>
Response: <response>
Answer:
```

That formatting is reused in both training and inference so the model sees the same input structure everywhere.

### `training_setups.py`

[`src/zhaw_at_touche/training_setups.py`](src/zhaw_at_touche/training_setups.py) loads optional JSON defaults from `train_model/<setup-name>.json`.

This gives the project a simple experiment-configuration mechanism without introducing a separate config framework. The implementation validates supported fields, including scheduler and optimizer-tuning fields such as `weight_decay`, `layerwise_lr_decay`, and `freeze_embeddings_epochs`, and lets the CLI override any of them later.

### `validation_setups.py`

[`src/zhaw_at_touche/validation_setups.py`](src/zhaw_at_touche/validation_setups.py) loads optional JSON defaults from `validate_model/<setup-name>.json`.

This is separate from `training_setups.py` on purpose. It allows evaluation-only presets for already-trained local or remote models, such as the `teamCMU` Hugging Face model, without treating them as trainable experiments.

### `embedding_setups.py`

[`src/zhaw_at_touche/embedding_setups.py`](src/zhaw_at_touche/embedding_setups.py) loads optional defaults for the embedding-divergence baseline from `validate_model/setup100.json`.

It keeps the experiment idea lightweight: a named JSON setup can define the embedding model, neutral-reference field, score granularity, thresholding behavior, and output paths without introducing another configuration system.

### `modeling.py`

[`src/zhaw_at_touche/modeling.py`](src/zhaw_at_touche/modeling.py) is the main ML runtime module. It is responsible for:

- device selection
- dataset loading
- tokenization and collation
- class weighting
- training loop execution
- model bundle loading
- batch prediction for saved models

Important implementation details:

- `resolve_device` chooses `cuda`, then `mps`, then `cpu`, unless the user explicitly forces a device.
- `InstructionCollator` converts raw records into the standardized prompt format before tokenization.
- `build_class_weights` compensates for class imbalance by up-weighting the positive class.
- `train_model` runs a manual PyTorch training loop with `AdamW`, gradient accumulation, optional gradient checkpointing, optional gradient clipping, `none`/`linear`/`cosine_with_warmup` schedulers, optional layerwise LR decay, temporary embedding freezing, and optional autocast for supported CUDA hardware.
- training uses the full training file by default, but `max_train_rows` can restrict it to a subset.
- trained model and tokenizer files are saved directly to `models/<setup-name>/`.
- `load_model_reference` can load either a local bundle path or a remote Hugging Face model reference.
- training can log step and epoch metrics to local files and to W&B online.
- predictions are returned as a small `Prediction` dataclass containing the binary label and the positive-class probability.

The training code intentionally avoids a larger trainer abstraction. The advantage is that the execution path remains explicit and easy to adapt for this binary classification task.

### `embedding_divergence.py`

[`src/zhaw_at_touche/embedding_divergence.py`](src/zhaw_at_touche/embedding_divergence.py) implements the evaluation-only semantic-drift baseline used by `setup100`.

It is responsible for:

- loading a sentence-embedding model
- mean-pooling token embeddings into normalized sentence or passage vectors
- splitting responses into sentences
- greedy sentence alignment between the neutral reference and the response
- cosine-distance scoring
- threshold calibration on labeled validation data

### `generation_utils.py`

[`src/zhaw_at_touche/generation_utils.py`](src/zhaw_at_touche/generation_utils.py) implements the neutral-response generation support logic:

- the system prompt used to constrain generated text
- stable model-to-field aliasing
- response cleanup and formatting normalization
- usage-token extraction across response schema variants
- retry logic with exponential backoff

The cleaning step is important. Model output is normalized into one continuous paragraph, bullet markers are stripped, escaped unicode is decoded, and repeated whitespace is collapsed. This reduces downstream schema drift and keeps generated rows consistent.

### `generated_stats.py`

[`src/zhaw_at_touche/generated_stats.py`](src/zhaw_at_touche/generated_stats.py) provides analytics for generated-response datasets:

- row loading and generated-field detection
- basic word and character statistics
- tokenizer-based token counts using `google.genai.local_tokenizer.LocalTokenizer`
- histogram binning
- SVG histogram rendering

This module is intentionally self-contained: it computes summaries and directly renders static SVGs without requiring a notebook or plotting server.

### `overlap_utils.py`

[`src/zhaw_at_touche/overlap_utils.py`](src/zhaw_at_touche/overlap_utils.py) checks split leakage across train, validation, and test files.

It supports overlap checks by:

- `id`
- `query`
- `response`
- `query+response`

The implementation builds indexes per split, intersects key sets, and returns small sample payloads for reporting.

### `evaluation_utils.py`

[`src/zhaw_at_touche/evaluation_utils.py`](src/zhaw_at_touche/evaluation_utils.py) contains the pure metric and confusion-matrix logic shared by validation and summary reporting:

- JSONL discovery
- confusion-matrix counting
- per-label, macro, and weighted metrics
- matrix and metrics rendering
- CSV export
- compact metrics payload generation

This module is deliberately independent from the training code so evaluation summaries can be generated from existing prediction files without loading a model.

## CLI Modules

The CLI package is in [`src/zhaw_at_touche/cli/`](src/zhaw_at_touche/cli/). Each file is a thin adapter over one domain area.

### `preprocess_data.py`

[`src/zhaw_at_touche/cli/preprocess_data.py`](src/zhaw_at_touche/cli/preprocess_data.py) merges `responses-<split>.jsonl` with `responses-<split>-labels.jsonl` and writes `responses-<split>-merged.jsonl`.

This step turns the raw paired-file format into a more convenient single-file format for later stages.

### `generate_neutral.py`

[`src/zhaw_at_touche/cli/generate_neutral.py`](src/zhaw_at_touche/cli/generate_neutral.py) drives the neutral-response generation workflow.

Implementation characteristics:

- resolves default paths from the requested split
- supports Gemini plus self-hosted OpenAI-compatible generation backends such as Qwen
- loads labels so generated rows can preserve label metadata
- supports resumable generation by skipping IDs already written to the output file
- uses `ThreadPoolExecutor` for parallel API calls
- appends rows incrementally instead of buffering all output in memory

This is designed for long-running generation jobs that may be interrupted and resumed.

### `train_model.py`

[`src/zhaw_at_touche/cli/train_model.py`](src/zhaw_at_touche/cli/train_model.py) resolves experiment defaults and launches training.

The CLI uses a two-stage parse:

1. parse `--setup-name` and `--setups-dir`
2. load JSON defaults for that setup
3. build the final parser with those defaults
4. allow explicit CLI flags to override them

That is a pragmatic alternative to a heavier configuration system.

The command uses the full training split by default. For quicker experiments it
also supports subset training through `--max-train-rows`.

It also supports setup-specific prompt structures. `setup7`, for example, uses
Longformer with a 1024-token context that includes the `gemini25flashlite`
neutral response as a reference segment.

Other current presets such as `setup6`, `setup8`, `setup9`, `setup10`,
`setup11`, and `setup12` stay on the default query-response prompt, while
`setup9` additionally exercises the newer optimizer controls for stabilized
DeBERTa-v3 fine-tuning and `setup10` to `setup12` use linear warmup/decay
scheduling for ALBERT, ELECTRA, and DistilRoBERTa baselines.

For monitoring, the command writes `training_metrics.jsonl` locally and can log
the same metrics to W&B online.

### `validate_model.py`

[`src/zhaw_at_touche/cli/validate_model.py`](src/zhaw_at_touche/cli/validate_model.py) is the central evaluation entrypoint.

It performs the following:

1. load a saved model bundle
2. or load a remote Hugging Face model reference from an evaluation setup
3. resolve input files
4. run batch prediction on the main response text
5. optionally detect and score a generated text field
6. write prediction JSONL files
7. aggregate file-level and overall metrics
8. export CSV, JSON, TXT, and PNG artifacts

Its default evaluation scope is the `test` split. Validation can be added by
passing `--eval-splits validation test`, or bypassed entirely with explicit
`--input-files`.

The same module also supports reference-aware validation, which matters for
models like `setup7` whose prompt structure differs from the default
query-response format.

Matching local validation presets are currently defined for `setup4`, `setup7`,
`setup9`, `setup10`, `setup11`, and `setup12`. Setups such as `setup6` and
`setup8` still work through the default `models/<setup-name>/` and
`results/<setup-name>/` resolution path without a dedicated validation JSON.

The validation step does more than print metrics. It also standardizes outputs into reusable artifact files:

- `metrics_summary.json`
- `response_metrics.txt`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `standardized_results.csv`
- `misclassified_analysis.csv`
- `*-predictions.jsonl`

### `embedding_divergence.py`

[`src/zhaw_at_touche/cli/embedding_divergence.py`](src/zhaw_at_touche/cli/embedding_divergence.py) runs the setup100 baseline end to end.

It performs the following:

1. load the embedding-divergence setup defaults
2. load the embedding model
3. score validation records to calibrate a threshold when needed
4. score the requested evaluation files
5. write prediction JSONL files
6. aggregate file-level and overall metrics
7. export CSV, JSON, TXT, and PNG artifacts

### `manual_inference.py`

[`src/zhaw_at_touche/cli/manual_inference.py`](src/zhaw_at_touche/cli/manual_inference.py) reuses the same prediction path for one-off examples. It supports both:

- direct single-example invocation with `--query` and `--response`
- an interactive prompt loop

This is useful for quick sanity checks against a trained model bundle without creating a JSONL file first.

### `data_stats.py`

[`src/zhaw_at_touche/cli/data_stats.py`](src/zhaw_at_touche/cli/data_stats.py) prints descriptive statistics for the official task files. The implementation changes summary logic by category:

- `responses`: query/response word lengths, topic distribution, search engine distribution
- `sentence-pairs`: combined pair length
- `tokens`: token list lengths

### `generated_stats.py`

[`src/zhaw_at_touche/cli/generated_stats.py`](src/zhaw_at_touche/cli/generated_stats.py) is the front-end for generated-response analytics. It can run basic summaries only or add tokenizer-based token metrics plus SVG histograms.

Legacy aliases are preserved for compatibility:

- `--model` as an alias for `--tokenizer-model`
- `--neutral-field` as an alias for `--generated-field`

### `check_overlap.py`

[`src/zhaw_at_touche/cli/check_overlap.py`](src/zhaw_at_touche/cli/check_overlap.py) prints split sizes and overlap samples using `overlap_utils.py`.

### `evaluation_matrix.py`

[`src/zhaw_at_touche/cli/evaluation_matrix.py`](src/zhaw_at_touche/cli/evaluation_matrix.py) builds a confusion-matrix summary from prediction JSONL files without requiring model inference.

## End-To-End Data Flow

The main workflow is file-based and staged.

### 1. Raw task inputs

Input files originate in `data/task/`, for example:

- `responses-train.jsonl`
- `responses-train-labels.jsonl`

### 2. Preprocessing

`touche-preprocess` merges response rows with label rows and writes:

- `data/task/preprocessed/responses-train-merged.jsonl`
- `data/task/preprocessed/responses-validation-merged.jsonl`
- `data/task/preprocessed/responses-test-merged.jsonl`

### 3. Neutral-response generation

`touche-generate-neutral` enriches response rows with a generated field derived from the model alias, for example:

- `gemini25flashlite`
- `qwen`

The output is written under `data/generated/<provider>/`, for example
`data/generated/gemini/` or `data/generated/qwen/`.

### 4. Training

`touche-train` reads one JSONL dataset, tokenizes `Query + Response`, trains a binary classifier, and writes:

- model weights
- tokenizer files
- `training_summary.json`

### 5. Validation

`touche-validate` loads either a saved local bundle or a remote evaluation-only model reference, predicts labels for one or more evaluation files, and writes:

- per-file prediction JSONL outputs
- aggregate metrics
- confusion matrices
- misclassification exports

### 6. Evaluation reporting

`touche-eval-matrix` can later consume those prediction JSONL outputs and regenerate matrix summaries without rerunning the model.

## Architecture Diagram

```text
                +----------------------+
                |   data/task/*.jsonl  |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | touche-preprocess    |
                | merge responses+labels
                +----------+-----------+
                           |
                           v
         +-----------------------------------------+
         | data/task/preprocessed/*.jsonl          |
         | or data/generated/<provider>/*.jsonl    |
         +----------------+------------------------+
                          |
             +------------+-------------+
             |                          |
             v                          v
   +--------------------+      +------------------------+
   | touche-train       |      | touche-generate-neutral|
   | modeling.py        |      | generation_utils.py    |
   +---------+----------+      +-----------+------------+
             |                             |
             v                             v
   +--------------------+      +------------------------+
   | models/<setup>/    |      | generated JSONL files  |
   +---------+----------+      +-----------+------------+
             |                             |
             +-------------+---------------+
                           |
                           v
                +----------------------+
                | touche-validate      |
                | predictions + metrics|
                +----------+-----------+
                           |
                           v
                +----------------------+
                | results/<setup>/     |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | touche-eval-matrix   |
                +----------------------+
```

## Design Choices

### Why JSONL everywhere

JSONL keeps the pipeline easy to stream, append, inspect, and diff. It also avoids coupling the pipeline to a database or binary dataset format.

### Why thin CLIs and shared modules

The codebase separates orchestration from logic. That makes utilities reusable between commands and keeps most of the important code importable and testable.

### Why explicit artifact directories

The repository uses conventional directories instead of hidden cache-style outputs. This makes the training and evaluation process easier to audit and reproduce.

### Why a manual training loop

The classifier training path is simple enough that a custom loop is clearer than introducing a higher-level trainer abstraction. The current implementation makes device logic, class weights, tokenization, and saved outputs easy to inspect.

## Testing Strategy

The current tests focus on deterministic utility behavior rather than full end-to-end model execution.

Examples:

- [`tests/test_training_setups.py`](tests/test_training_setups.py) verifies setup loading and CLI override precedence.
- [`tests/test_evaluation_utils.py`](tests/test_evaluation_utils.py) verifies confusion-matrix counting and aggregate metrics.
- [`tests/test_generated_stats.py`](tests/test_generated_stats.py) verifies histogram behavior, tokenizer metric naming, and legacy CLI aliases.

This gives quick coverage over the logic most likely to regress while avoiding heavyweight training jobs in unit tests.

## Current Limitations

- Neutral-response generation is implemented for Gemini plus self-hosted OpenAI-compatible providers such as Qwen.
- The repository is strongly file-based and does not include experiment tracking, dataset versioning, or a service layer.
- Validation and reporting are batch-oriented rather than online or interactive.
- Large dataset files are stored directly in the repository, which creates repository-size pressure and GitHub large-file warnings.

## Summary

The repository is structured as a single Python package that orchestrates a staged ML workflow through small CLIs and reusable modules. The code favors explicit file conventions, predictable artifact generation, and lightweight abstractions over framework-heavy architecture. That is a good fit for a research-oriented workflow where inspectability and reproducibility matter more than service orchestration.
