# Validation Setups

This directory stores reusable defaults for evaluation-only runs.

Use it for already-trained models that should be evaluated through
`touche-validate`, including remote Hugging Face models that are not part of
the local training pipeline.

Each setup is a JSON file named `<setup-name>.json`. The `touche-validate` CLI
loads the matching file automatically when you pass `--setup-name <setup-name>`.

Supported JSON fields:

- `model_name`
- `model_dir`
- `results_dir`
- `eval_splits`
- `input_files`
- `text_field`
- `input_format`
- `reference_field`
- `reference_label`
- `pad_to_max_length`
- `generated_field`
- `max_length`
- `batch_size`
- `threshold`
- `device`

Example:

```bash
uv run touche-validate --setup-name teamCMU
```

Local validation preset for the Longformer training setup:

```bash
uv run touche-validate --setup-name setup7
```

Local validation preset for the DeBERTa-v3 `setup4` training setup:

```bash
uv run touche-validate --setup-name setup4
```

Additional local validation presets for the newer query-response baselines:

```bash
uv run touche-validate --setup-name setup9
uv run touche-validate --setup-name setup10
uv run touche-validate --setup-name setup11
uv run touche-validate --setup-name setup12
```

Setups such as `setup6` and `setup8` do not need a dedicated validation JSON.
They fall back to the default local paths `models/<setup-name>/` and
`results/<setup-name>/` when you pass `--setup-name`.

Embedding-divergence baseline:

```bash
uv run touche-embed-divergence --setup-name setup100
```

`setup100` is evaluation-only and uses `validate_model/setup100.json` with the
new `touche-embed-divergence` CLI. It embeds the neutral reference and the
response separately, computes a drift score, calibrates a threshold on the
validation split, and evaluates on the test split.

By default the validator evaluates only the `test` split. To evaluate both
validation and test data, either set `eval_splits` in the setup JSON or pass:

```bash
uv run touche-validate --setup-name teamCMU --eval-splits validation test
```
