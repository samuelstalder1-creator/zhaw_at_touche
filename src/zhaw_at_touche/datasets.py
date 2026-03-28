from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .constants import BASE_GENERATED_FIELDS
from .jsonl import read_jsonl

LABEL_FIELDS = (
    "label",
    "item",
    "advertiser",
    "ad_num",
    "spans",
    "sentence_spans",
    "bio_tags",
)

WHITESPACE_RE = re.compile(r"\s+")
DEFAULT_INPUT_FORMAT = "query_response"
NEUTRAL_REFERENCE_INPUT_FORMAT = "query_neutral_response"
RAG_REFERENCE_INPUT_FORMAT = "query_reference_rag_response"
SUPPORTED_INPUT_FORMATS = (
    DEFAULT_INPUT_FORMAT,
    NEUTRAL_REFERENCE_INPUT_FORMAT,
    RAG_REFERENCE_INPUT_FORMAT,
)


def load_label_map(label_path: Path) -> dict[str, dict[str, Any]]:
    label_map: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(label_path):
        row_id = row.get("id")
        if not isinstance(row_id, str):
            continue
        label_map[row_id] = row
    return label_map


def merge_response_row(
    response_row: dict[str, Any],
    label_row: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(response_row)
    if not label_row:
        return merged

    for key in LABEL_FIELDS:
        if key in label_row:
            merged[key] = label_row[key]
    return merged


def merge_response_split(response_path: Path, label_path: Path) -> list[dict[str, Any]]:
    label_map = load_label_map(label_path)
    merged_rows: list[dict[str, Any]] = []
    missing_labels: list[str] = []

    for row in read_jsonl(response_path):
        row_id = row.get("id")
        if not isinstance(row_id, str):
            continue
        label_row = label_map.get(row_id)
        if label_row is None:
            missing_labels.append(row_id)
        merged_rows.append(merge_response_row(row, label_row))

    if missing_labels:
        sample = ", ".join(missing_labels[:5])
        raise ValueError(
            f"Missing label rows for {len(missing_labels)} response rows in {response_path}. "
            f"Examples: {sample}"
        )

    return merged_rows


def detect_generated_text_field(first_row: dict[str, Any], explicit_field: str | None) -> str:
    if explicit_field:
        if explicit_field not in first_row:
            raise ValueError(f"--generated-field '{explicit_field}' not found in the input rows.")
        return explicit_field

    candidates = [
        key
        for key, value in first_row.items()
        if key not in BASE_GENERATED_FIELDS and isinstance(value, str)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            "Could not auto-detect the generated response field. Pass --generated-field explicitly."
        )
    joined = ", ".join(sorted(candidates))
    raise ValueError(
        f"Multiple candidate generated response fields found: {joined}. "
        "Pass --generated-field explicitly."
    )


def build_model_input(
    query: str,
    response: str,
    *,
    input_format: str = DEFAULT_INPUT_FORMAT,
    reference_response: str | None = None,
    reference_label: str = "GEMINI",
) -> str:
    if input_format == DEFAULT_INPUT_FORMAT:
        return f"Query: {query}\nResponse: {response}\nAnswer:"
    if input_format == NEUTRAL_REFERENCE_INPUT_FORMAT:
        neutral_reference = reference_response or ""
        return (
            f"USER QUERY: {query}\n\n"
            f"NEUTRAL REFERENCE ({reference_label}): {neutral_reference}\n\n"
            f"RESPONSE TO CLASSIFY: {response}\n\n"
            "LABEL THIS AS AD OR NEUTRAL:"
        )
    if input_format == RAG_REFERENCE_INPUT_FORMAT:
        neutral_reference = reference_response or ""
        return (
            f"Query: {query}\n"
            f"{reference_label}: {neutral_reference}\n"
            f"RAG Response: {response}\n"
            "Task: Is the RAG Response an advertisement? Answer:"
        )
    joined = ", ".join(SUPPORTED_INPUT_FORMATS)
    raise ValueError(f"Unsupported input format '{input_format}'. Expected one of: {joined}")


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip())


def word_count(text: str) -> int:
    normalized = normalize_text(text)
    if not normalized:
        return 0
    return len(normalized.split(" "))
