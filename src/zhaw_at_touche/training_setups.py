from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_TRAINING_SETUPS_DIR = Path("train_model")

ALLOWED_SETUP_FIELDS = {
    "aux_neutral_field",
    "aux_reference_label",
    "aux_train_file",
    "aux_validation_file",
    "batch_size",
    "device",
    "distance_metric",
    "epochs",
    "freeze_embeddings_epochs",
    "gradient_checkpointing",
    "grad_accum",
    "input_format",
    "layerwise_lr_decay",
    "learning_rate",
    "lr_scheduler",
    "max_length",
    "max_grad_norm",
    "max_train_rows",
    "model_dir",
    "model_name",
    "neutral_field",
    "optimizer_eps",
    "pad_to_max_length",
    "positive_class_weight_scale",
    "query_field",
    "reference_field",
    "reference_label",
    "reference_label_1",
    "reference_label_2",
    "response_field",
    "score_granularity",
    "sentence_agg",
    "sentence_delta_agg",
    "threshold_metric",
    "train_file",
    "trainer_type",
    "validation_file",
    "wandb_dir",
    "wandb_enabled",
    "wandb_project",
    "wandb_run_name",
    "weight_decay",
    "warmup_ratio",
}


def setup_config_path(setup_name: str, setups_dir: Path = DEFAULT_TRAINING_SETUPS_DIR) -> Path:
    return setups_dir / f"{setup_name}.json"


def load_setup_defaults(
    setup_name: str,
    setups_dir: Path = DEFAULT_TRAINING_SETUPS_DIR,
) -> dict[str, Any]:
    config_path = setup_config_path(setup_name, setups_dir)
    if not config_path.exists():
        return {}

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in training setup config: {config_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Training setup config must be a JSON object: {config_path}")

    unknown_fields = sorted(set(payload) - ALLOWED_SETUP_FIELDS - {"description"})
    if unknown_fields:
        joined = ", ".join(unknown_fields)
        raise ValueError(f"Unsupported fields in training setup config {config_path}: {joined}")

    defaults: dict[str, Any] = {}
    for field in ALLOWED_SETUP_FIELDS:
        if field in payload:
            defaults[field] = payload[field]
    return defaults
