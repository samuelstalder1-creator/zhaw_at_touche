# zhaw_at_touche

Unified `uv`-managed tooling for the Touché ad-detection workflow.

The repository is organized around one Python package, reusable experiment
setups, and file-based artifacts for training and evaluation.

## Repository Layout

```text
.
├── data/
│   ├── task/                  # official Touché files and labels
│   └── generated/             # neutral-response files by provider
├── train_model/               # named training setup defaults
├── validate_model/            # named validation setup defaults
├── models/                    # trained bundles and saved state
├── results/                   # evaluation artifacts
├── src/zhaw_at_touche/        # package code and CLI entry points
├── tests/                     # lightweight unit tests
├── setup.md                   # canonical setup reference
├── results.md                 # committed result summary
├── TECHNICAL_ARCHITECTURE.md  # code-level architecture
├── Implementation.md
├── Implementation_details.md
└── QA_checklist.md
```

## Environment

- Python: `3.12`
- Package manager: `uv`
- Device resolution order: `cuda -> mps -> cpu`
- Typical local validation target: Apple Silicon / M2 Mac
- Typical remote training target: Ubuntu host with an L4 GPU

Setup:

```bash
uv sync
```

## Workflow

### 1. Preprocess the raw task data

```bash
uv run touche-preprocess
```

This merges the raw response files with their label files and writes easier to
consume JSONL files into `data/task/preprocessed/`.

### 2. Generate neutral responses

Gemini:

```bash
export GEMINI_API_KEY="..."
uv run touche-generate-neutral --split train
uv run touche-generate-neutral --split validation
uv run touche-generate-neutral --split test
```

Qwen:

```bash
uv run touche-generate-neutral \
  --provider qwen \
  --split validation \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda
```

Gemini writes `data/generated/gemini/*.jsonl`. Qwen writes
`data/generated/qwen/*.jsonl`.

### 3. Train a setup

Classifier examples:

```bash
uv run touche-train --setup-name setup6
uv run touche-train --setup-name setup12
uv run touche-train --setup-name setup7
```

Embedding-based examples:

```bash
uv run touche-train --setup-name setup100
uv run touche-train --setup-name setup101
uv run touche-train --setup-name setup102
```

Anchor-distance example:

```bash
uv run touche-train --setup-name setup110
```

### 4. Validate a setup

```bash
uv run touche-validate --setup-name setup6
uv run touche-validate --setup-name setup12
uv run touche-validate --setup-name setup7
uv run touche-validate --setup-name setup110
```

Provider-specific evaluation:

```bash
uv run touche-validate --setup-name setup6 --generated-provider qwen
uv run touche-validate --setup-name setup7-qwen
```

### 5. Inspect pairwise distances between response fields

```bash
uv run touche-pairwise-distances \
  --input-files \
    data/generated/gemini/responses-test-with-neutral_gemini.jsonl \
    data/generated/qwen/responses-test-with-neutral_qwen.jsonl \
  --pair gemini25flashlite:qwen \
  --pair response:gemini25flashlite \
  --pair response:qwen \
  --results-dir results/pairwise-test
```

This merges rows by `id` and writes pairwise distance summaries for the
requested field pairs.

## Setup Families

### Supported by the current training CLI

| Family | Setup names | Summary |
| --- | --- | --- |
| Fine-tuned classifier | `setup4`, `setup6`, `setup6-qwen`, `setup7`, `setup7-qwen`, `setup8`, `setup9`, `setup10`, `setup11`, `setup12` | transformer classifiers over prompt-formatted inputs |
| Embedding divergence | `setup100`, `setup101`, `setup102` | saved-state semantic-drift baselines over response vs neutral embeddings |
| Anchor distance | `setup110` | saved-state logistic regression over six pairwise query/Gemini/Qwen/response embedding distances |

### Present as archived descriptors

`setup103`, `setup104`, `setup105`, and `setup106` are still documented in
`setup.md` because their JSON descriptors and some committed results remain in
the repository, but the current `touche-train` parser does not expose their
trainer backends.

## Where To Read More

- `setup.md`: canonical explanation of every setup and every concept used
- `train_model/README.md`: how training setup JSON files work
- `validate_model/README.md`: how validation setup JSON files work
- `results.md`: committed metrics and selection guidance
- `TECHNICAL_ARCHITECTURE.md`: package and module architecture

## Current Result Headlines

- Best committed Gemini-backed classifier: `setup12`
- Best committed Qwen-backed classifier: `setup6-qwen`
- Best committed archived embedding-feature idea: `setup104`
- New runnable multi-anchor embedding baseline: `setup110`
- Current semantic-drift baselines (`setup100` to `setup102`) are clearly below
  the classifier family

See `results.md` for the exact numbers and caveats.
