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
- `optimizer_eps`
- `weight_decay`
- `lr_scheduler`
- `warmup_ratio`
- `max_grad_norm`
- `gradient_checkpointing`
- `layerwise_lr_decay`
- `freeze_embeddings_epochs`
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

DeBERTa-v3 variant of setup6:

```bash
uv run touche-train --setup-name setup8
```

Longformer neutral-reference setup:

```bash
uv run touche-train --setup-name setup7
```

DeBERTa-v3 setup with unbiased reference + RAG-response prompt:

```bash
uv run touche-train --setup-name setup4
```

Stabilized DeBERTa-v3 setup with lower LR, more warmup, weight decay, layerwise LR decay, and one frozen-embedding epoch:

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

By default `touche-train` uses the full training file. To train on only a
subset, pass for example:

```bash
uv run touche-train --setup-name setup6 --max-train-rows 1000
```

Training also writes local monitoring artifacts:

- `training_summary.json`
- `training_metrics.jsonl`
- W&B run files under `<model-dir>/wandb/` by default

W&B logging uses the online service. Authenticate first:

```bash
uv run wandb login
```

You can disable it:

```bash
uv run touche-train --setup-name setup7 --no-wandb
```

If a validation file is configured, `touche-train` evaluates that split at the
end of every epoch and logs the validation metrics plus confusion-count
monitoring to W&B.
