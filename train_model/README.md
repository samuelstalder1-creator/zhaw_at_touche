# Training Setups

This directory stores reusable training defaults for named experiment setups.

Each setup is a JSON file named `<setup-name>.json`. The `touche-train` CLI
loads the matching file automatically when you pass `--setup-name <setup-name>`.

Supported JSON fields:

- `train_file`
- `model_name`
- `model_dir`
- `max_length`
- `epochs`
- `batch_size`
- `grad_accum`
- `learning_rate`
- `device`
- `max_train_rows`
- `input_format`
- `reference_field`
- `reference_label`
- `pad_to_max_length`
- `positive_class_weight_scale`
- `validation_file`
- `wandb_enabled`
- `wandb_project`
- `wandb_dir`
- `wandb_run_name`

Example:

```bash
uv run touche-train --setup-name setup6
```

Longformer neutral-reference setup:

```bash
uv run touche-train --setup-name setup7
```

By default `touche-train` uses the full training file. To train on only a
subset, pass for example:

```bash
uv run touche-train --setup-name setup6 --max-train-rows 1000
```

Training also writes local monitoring artifacts:

- `training_summary.json`
- `training_metrics.jsonl`
- offline W&B files under `<model-dir>/wandb/` by default

W&B logging runs in offline mode and stores files locally. You can disable it:

```bash
uv run touche-train --setup-name setup7 --no-wandb
```
