# zhaw_at_touche

Unified `uv`-managed tooling for the Touché ad-detection workflow in one shared Python 3.12 project with CLI entry points.

## Repository Layout

```text
.
├── data
│   ├── task/                     # official Touché task files
│   └── generated/
│       ├── gemini/              # generated neutral responses from Gemini
│       ├── qwen/                # generated neutral responses from a self-hosted Qwen backend
│       └── chatgpt/             # reserved for future hosted OpenAI outputs
├── train_model/                 # named training setup defaults
├── validate_model/              # evaluation-only setup defaults
├── models/
│   ├── setupX/                  # saved model bundle for one experiment
│   ├── setupY/                  # saved model bundle for another experiment
│   └── setup6/                  # saved model bundle for the merged Dagmar setup
├── results/
│   ├── setupX/                  # validation outputs, confusion matrix, prediction files
│   └── setup6/                  # validation outputs for the merged Dagmar setup
├── src/zhaw_at_touche/          # shared package code and CLIs
├── tests/                       # lightweight unit tests for pure utility code
├── Implementation.md
├── Implementation_details.md
└── QA_checklist.md
```

## Environment

- Python: `3.12`
- Package manager: `uv`
- Local validation target: Apple Silicon / M2 Mac
- Remote training + inference target: Ubuntu host with L4 GPU

`uv sync` is the default setup command. The project auto-selects `cuda`, then `mps`, then `cpu`. If the Ubuntu host needs a CUDA-specific PyTorch wheel, install the current PyTorch build that matches that machine before training.

## Setup

```bash
uv sync
```

## Main Commands

### 1. Preprocess the raw task data

Merges `responses-<split>.jsonl` with the corresponding label file and writes easier-to-consume JSONL files into `data/task/preprocessed/`.

```bash
uv run touche-preprocess
```

### 2. Generate neutral responses

Gemini generation requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

```bash
export GEMINI_API_KEY="..."
uv run touche-generate-neutral --split train
uv run touche-generate-neutral --split validation
uv run touche-generate-neutral --split test
```

Default outputs:

- `data/generated/gemini/responses-train-with-neutral_gemini.jsonl`
- `data/generated/gemini/responses-validation-with-neutral_gemini.jsonl`
- `data/generated/gemini/responses-test-with-neutral_gemini.jsonl`

Self-hosted Qwen generation uses an OpenAI-compatible local endpoint. Example for the validation split:

```bash
export QWEN_API_BASE="http://127.0.0.1:8000/v1"
export QWEN_API_KEY="EMPTY"
uv run touche-generate-neutral \
  --provider qwen \
  --split validation \
  --model Qwen/Qwen2.5-1.5B-Instruct
```

This writes:

- `data/generated/qwen/responses-validation-with-neutral_qwen.jsonl`

### 3. Train a model

```bash
uv run touche-train --setup-name setupX
```

The CLI also reads optional defaults from `train_model/<setup-name>.json`.

Default training output directory:

- `models/setupX/`

You can override the dataset or model name:

```bash
uv run touche-train \
  --setup-name setupY \
  --train-file data/task/preprocessed/responses-train-merged.jsonl \
  --model-name FacebookAI/roberta-base
```

Merged Dagmar setup:

```bash
uv run touche-train --setup-name setup6
```

DeBERTa-v3 variant of setup6:

```bash
uv run touche-train --setup-name setup8
```

Longformer setup with Gemini neutral-reference context:

```bash
uv run touche-train --setup-name setup7
```

DeBERTa-v3 setup with the unbiased-reference / RAG-response prompt:

```bash
uv run touche-train --setup-name setup4
```

Stabilized DeBERTa-v3 setup with lower LR and optimizer safeguards:

```bash
uv run touche-train --setup-name setup9
```

ALBERT-base-v2 setup with linear warmup/decay scheduling:

```bash
uv run touche-train --setup-name setup10
```

ELECTRA-base discriminator setup with linear warmup/decay scheduling:

```bash
uv run touche-train --setup-name setup11
```

DistilRoBERTa setup with linear warmup/decay scheduling:

```bash
uv run touche-train --setup-name setup12
```

Embedding-divergence setup that fits and saves a threshold/state bundle instead
of a classifier checkpoint:

```bash
uv run touche-train --setup-name setup100
```

Classifier training writes local monitoring artifacts next to the model bundle:

- `training_summary.json`
- `training_metrics.jsonl`
- W&B run files in `models/<setup-name>/wandb/` by default

`setup100` instead writes `embedding_state.json` plus `training_summary.json`.

For online monitoring, log in to W&B first:

```bash
uv run wandb login
```

Then train as usual and open the run in the configured W&B project.
When a validation file is available, training evaluates it at the end of each epoch and logs validation loss, accuracy, precision, recall, F1, and confusion-count monitoring to W&B.
You can disable W&B logging with `--no-wandb`.

By default training uses the full training dataset. To train on a subset:

```bash
uv run touche-train --setup-name setup6 --max-train-rows 1000
```

### 4. Validate a trained model

```bash
uv run touche-validate --setup-name setupX
```

By default validation evaluates only the `test` split. To evaluate both
validation and test data:

```bash
uv run touche-validate --setup-name setupX --eval-splits validation test
```

The validator also supports evaluation-only presets for already-trained remote models:

```bash
uv run touche-validate --setup-name teamCMU
```

`teamCMU` is defined in `validate_model/teamCMU.json` and evaluates the published Hugging Face model `teknology/ad-classifier-v0.4` without using `train_model/`. It also defaults to test-only evaluation unless you pass `--eval-splits validation test`.

The local Longformer training preset also has a matching validation preset:

```bash
uv run touche-validate --setup-name setup7
```

Additional local validation presets are available for `setup4`, `setup9`,
`setup10`, `setup11`, and `setup12`. Setups such as `setup6` and `setup8`
still validate through the default `models/<setup-name>/` and
`results/<setup-name>/` path resolution.

Default validation artifacts:

- `results/setupX/metrics_summary.json`
- `results/setupX/response_metrics.txt`
- `results/setupX/confusion_matrix.csv`
- `results/setupX/confusion_matrix.png`
- `results/setupX/standardized_results.csv`
- `results/setupX/misclassified_analysis.csv`
- `results/setupX/*-predictions.jsonl`

### 4b. Run the embedding-divergence baseline

`setup100` is a two-stage experiment. `touche-train --setup-name setup100`
fits and saves an embedding-divergence threshold/state bundle in
`models/setup100/embedding_state.json`. You can then validate it either with
`touche-validate --setup-name setup100` or directly with
`touche-embed-divergence --setup-name setup100`. Both routes load the saved
threshold by default and fall back to validation calibration only if no saved
state or manual threshold is available.

```bash
uv run touche-train --setup-name setup100
uv run touche-validate --setup-name setup100
uv run touche-embed-divergence --setup-name setup100
```

Default artifacts:

- `models/setup100/embedding_state.json`
- `models/setup100/training_summary.json`
- `results/setup100/metrics_summary.json`
- `results/setup100/response_metrics.txt`
- `results/setup100/confusion_matrix.csv`
- `results/setup100/confusion_matrix.png`
- `results/setup100/standardized_results.csv`
- `results/setup100/misclassified_analysis.csv`
- `results/setup100/*-predictions.jsonl`

### 5. Manually test a model with custom text

Single example:

```bash
uv run touche-predict --setup-name setupX --query "..." --response "..."
```

Interactive mode:

```bash
uv run touche-predict --setup-name setupX
```

### 6. Get statistics from the task data

```bash
uv run touche-stats-data
```

Optional JSON export:

```bash
uv run touche-stats-data --json-out results/setupX/data_stats.json
```

### 7. Get statistics from generated neutral responses

Basic length statistics:

```bash
uv run touche-stats-generated
```

With Gemini tokenizer-based counts and SVG histograms:

```bash
uv run touche-stats-generated \
  --tokenizer-model gemini-2.5-flash-lite \
  --histogram-dir results/generated_stats/histograms \
  --json-out results/setupX/generated_stats.json
```

Default histogram output:

- `results/generated_stats/histograms/<input-stem>-token-histogram.svg`

Legacy aliases are also accepted:

- `--model` for `--tokenizer-model`
- `--neutral-field` for `--generated-field`

### 8. Check split overlap before training

```bash
uv run touche-check-overlap
```

Useful override:

- `uv run touche-check-overlap --sample-limit 5`

### 9. Generate a confusion-matrix summary from prediction files

```bash
uv run touche-eval-matrix results/setupX
```

## Notes

- `data/task/README.md` contains the original dataset description.
- `data/generated/qwen/` is used for self-hosted Qwen neutral-response runs.
- `data/generated/chatgpt/` is included so future hosted OpenAI-generated files can follow the same layout later.
- `validate_model/` stores evaluation-only presets for already-trained models such as `teamCMU`.
- Use the root CLI entry points in `src/zhaw_at_touche/` for generation, training, validation, and analysis tasks.
