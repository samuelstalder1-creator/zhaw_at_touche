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

Example:

```bash
uv run touche-train --setup-name setup6
```
