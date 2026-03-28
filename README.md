# zhaw_at_touche

Unified `uv`-managed tooling for the Touché ad-detection workflow. The old `dagmar`, `neutral_response_generator`, and `run_evaluation` repos have been consolidated into one shared Python 3.12 project with CLI entry points.

## Repository Layout

```text
.
├── data
│   ├── task/                     # official Touché task files
│   └── generated/
│       ├── gemini/              # generated neutral responses from Gemini
│       └── chatgpt/             # reserved for generated OpenAI outputs
├── train_model/                 # named training setup defaults
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

Requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

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

### 3. Train a model

```bash
uv run touche-train --setup-name setupX
```

The CLI also reads optional defaults from `train_model/<setup-name>.json`.

Default model output:

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

### 4. Validate a trained model

```bash
uv run touche-validate --setup-name setupX
```

Default validation artifacts:

- `results/setupX/metrics_summary.json`
- `results/setupX/response_metrics.txt`
- `results/setupX/confusion_matrix.csv`
- `results/setupX/confusion_matrix.png`
- `results/setupX/standardized_results.csv`
- `results/setupX/misclassified_analysis.csv`
- `results/setupX/*-predictions.jsonl`

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

### 9. Recreate the old run_evaluation summary

```bash
uv run touche-eval-matrix results/setupX
```

## Notes

- `data/task/README.md` contains the original dataset description.
- `data/generated/chatgpt/` is included so OpenAI-generated files can follow the same layout later.
- The neutral-response generation tooling now lives in `src/zhaw_at_touche/`; use the root CLI entry points instead of the old standalone scripts.
