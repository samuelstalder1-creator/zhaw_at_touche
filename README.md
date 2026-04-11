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
uv run touche-train --setup-name setup103
uv run touche-train --setup-name setup105_1
uv run touche-train --setup-name setup116
uv run touche-train --setup-name setup100
uv run touche-train --setup-name setup101
uv run touche-train --setup-name setup102
```

Anchor-distance example:

```bash
uv run touche-train --setup-name setup110
uv run touche-train --setup-name setup111
```

### 4. Validate a setup

```bash
uv run touche-validate --setup-name setup6
uv run touche-validate --setup-name setup12
uv run touche-validate --setup-name setup7
uv run touche-validate --setup-name setup110
uv run touche-validate --setup-name setup111
```

Provider-specific evaluation:

```bash
uv run touche-validate --setup-name setup6 --generated-provider qwen
uv run touche-validate --setup-name setup7-qwen
```

## Setup Families

### Supported by the current training CLI

| Family | Setup names | Summary |
| --- | --- | --- |
| Fine-tuned classifier | `setup4`, `setup6`, `setup6-qwen`, `setup7`, `setup7-qwen`, `setup8`, `setup9`, `setup10`, `setup11`, `setup12`, `setup115`, `setup116` | transformer classifiers over prompt-formatted inputs |
| Cross-encoder | `setup105`, `setup105_1` | jointly encodes response and neutral reference in one sequence |
| Learned embedding features | `setup103`, `setup104`, `setup113`, `setup114`, `setup117`, `setup118`, `setup119` | frozen encoder plus learned logistic regression over delta or stacked embedding features |
| Embedding divergence | `setup100`, `setup101`, `setup102` | saved-state semantic-drift baselines over response vs neutral embeddings |
| Scalar anchor baseline | `setup110`, `setup111` | multi-anchor Gemini+Qwen baselines over six query/Gemini/Qwen/response cosine distances; `setup110` learns weights, `setup111` uses a handcrafted score |

### Descriptor Only

`setup106` remains documented in `setup.md` because its JSON descriptors and a
historical committed result remain in the repository, but the current
`touche-train` and `touche-validate` flows do not expose a sentence-delta
backend.

## Where To Read More

- `setup.md`: canonical explanation of every setup and every concept used
- `train_model/README.md`: how training setup JSON files work
- `validate_model/README.md`: how validation setup JSON files work
- `results.md`: committed metrics and selection guidance
- `TECHNICAL_ARCHITECTURE.md`: package and module architecture

## Current Result Headlines

- Best committed overall classifier: `setup115`
- Best committed Gemini-backed query-aware classifier: `setup12`
- Best committed Qwen-backed classifier: `setup6-qwen`
- Best committed cross-encoder retry: `setup105_1`
- Best committed learned embedding-feature setup: `setup104`
- Best committed dual-neutral delta setup: `setup113`
- Scalar anchor baselines `setup110` and `setup111` remain far below the
  vector-delta family
- Current semantic-drift baselines (`setup100` to `setup102`) are clearly below
  the classifier family

See `results.md` for the exact numbers and caveats.
