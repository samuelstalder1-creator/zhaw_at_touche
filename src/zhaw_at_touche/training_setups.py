from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_TRAINING_SETUPS_DIR = Path("train_model")

ALLOWED_SETUP_FIELDS = {
    "batch_size",
    "device",
    "epochs",
    "grad_accum",
    "learning_rate",
    "max_length",
    "model_dir",
    "model_name",
    "train_file",
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
        value = payload.get(field)
        if value is not None:
            defaults[field] = value
    return defaults
