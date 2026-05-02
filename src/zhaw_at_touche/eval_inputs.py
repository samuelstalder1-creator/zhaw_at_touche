from __future__ import annotations

from pathlib import Path
from typing import Sequence

SUPPORTED_GENERATED_PROVIDERS = ("gemini", "qwen", "gemma426b")
PROVIDER_GENERATED_FIELDS = {
    "gemini": "gemini25flashlite",
    "qwen": "qwen",
    "gemma426b": "gemma4_26b",
}
PROVIDER_REFERENCE_LABELS = {
    "gemini": "GEMINI",
    "qwen": "QWEN",
    "gemma426b": "GEMMA4-26B",
}


def generated_response_paths(provider: str, splits: Sequence[str]) -> list[Path]:
    return [
        Path(f"data/generated/{provider}/responses-{split}-with-neutral_{provider}.jsonl")
        for split in splits
    ]


def task_response_paths(splits: Sequence[str]) -> list[Path]:
    return [Path(f"data/task/preprocessed/responses-{split}-merged.jsonl") for split in splits]


def resolve_default_eval_paths(
    eval_splits: Sequence[str] | None = None,
    generated_provider: str | None = None,
) -> list[Path]:
    splits = list(eval_splits) if eval_splits else ["test"]
    if generated_provider is not None:
        return generated_response_paths(generated_provider, splits)

    gemini_paths = generated_response_paths("gemini", splits)
    if all(path.exists() for path in gemini_paths):
        return gemini_paths

    return task_response_paths(splits)


def resolve_default_calibration_paths(generated_provider: str | None = None) -> list[Path]:
    if generated_provider is not None:
        return generated_response_paths(generated_provider, ["validation"])

    gemini_validation = generated_response_paths("gemini", ["validation"])[0]
    if gemini_validation.exists():
        return [gemini_validation]

    return task_response_paths(["validation"])


def generated_field_for_provider(provider: str) -> str:
    return PROVIDER_GENERATED_FIELDS[provider]


def reference_label_for_provider(provider: str) -> str:
    return PROVIDER_REFERENCE_LABELS[provider]


def with_provider_results_dir(base_dir: Path, generated_provider: str | None) -> Path:
    if generated_provider in (None, "gemini"):
        return base_dir

    suffix = f"-{generated_provider}"
    if base_dir.name.endswith(suffix):
        return base_dir

    return base_dir.parent / f"{base_dir.name}{suffix}"
