# Setup 8 Summary

`setup8` is the DeBERTa-v3 version of `setup6`.

## What It Is

- Setup name: `setup8`
- Purpose: keep the `setup6` recipe and replace the backbone with `microsoft/deberta-v3-base`
- Train command: `uv run touche-train --setup-name setup8`
- Validate command: `uv run touche-validate --setup-name setup8`

## Key Configuration

- Model: `microsoft/deberta-v3-base`
- Training file: `data/generated/gemini/responses-train-with-neutral_gemini.jsonl`
- Validation file during training: `data/generated/gemini/responses-validation-with-neutral_gemini.jsonl`
- Input format: `query_response`
- Max length: `512`
- Epochs: `3`
- Batch size: `16`
- Gradient accumulation: `4`
- Effective batch size per optimizer step: `64`
- Learning rate: `2e-05`
- Optimizer epsilon: `1e-08`
- Scheduler: `none`
- Gradient checkpointing: `false`
- Positive class weight scale: `2.0`
- W&B logging: enabled by default, project `zhaw-at-touche-training`
- Output model directory: `models/setup8/`
- Output results directory: `results/setup8/`

## Status

As of 2026-03-31, the preset exists, but there are no committed `setup8` training or validation artifacts in the repository yet.

- No committed `models/setup8/`
- No committed `results/setup8/`

## Closest Baseline Results

The closest available reference is `setup6`, because `setup8` mirrors it except for the backbone model.

- Overall accuracy: `0.3071`
- Positive-class precision: `0.3071`
- Positive-class recall: `1.0000`
- Positive-class F1: `0.4699`
- Observed behavior: `setup6` predicted label `1` for every row

Confusion counts from `setup6` overall:

- True negatives: `0`
- False positives: `8315`
- False negatives: `0`
- True positives: `3685`

## Example Input

Example training row:

- `query`: `Can the Dyson air purifier help with allergies and asthma?`
- `label`: `1`
- `item`: `Dyson Purifier Hot+Cool Formaldehyde Purifying Fan Heater`

Formatted model input for `setup8`:

```text
Query: Can the Dyson air purifier help with allergies and asthma?
Response: The Dyson air purifier can help reduce allergens and improve air quality ... If you have severe allergies or asthma, it's advisable to consult a healthcare provider for personalized advice and treatment.
Answer:
```

## Example Output

Expected prediction record shape:

```json
{
  "source_file": "responses-test-with-neutral_gemini.jsonl",
  "id": "4SFIB7N0-6313-ZTXU",
  "query": "What are the benefits of using water coolers instead of bottled water?",
  "gold_label": 1,
  "response_label": 1,
  "response_ad_prob": 1.0,
  "gemini25flashlite_label": 1,
  "gemini25flashlite_ad_prob": 1.0
}
```

`1` means ad and `0` means neutral.
