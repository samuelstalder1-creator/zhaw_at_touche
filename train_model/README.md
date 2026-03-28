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

Example:

```bash
uv run touche-train --setup-name setup6
```

By default `touche-train` uses the full training file. To train on only a
subset, pass for example:

```bash
uv run touche-train --setup-name setup6 --max-train-rows 1000
```
