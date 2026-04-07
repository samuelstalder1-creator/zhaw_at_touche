from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from .embedding_divergence import (
    aggregate_sentence_distances,
    cosine_distance,
    embed_texts,
    greedy_sentence_alignment,
    split_sentences,
    top_sentence_candidates,
)
from .jsonl import read_jsonl


@dataclass(frozen=True)
class FieldPair:
    left_field: str
    right_field: str

    @property
    def key(self) -> str:
        return f"{self.left_field}__{self.right_field}"

    @property
    def label(self) -> str:
        return f"{self.left_field}:{self.right_field}"


def parse_field_pair(raw_value: str) -> FieldPair:
    left_field, separator, right_field = raw_value.partition(":")
    if separator != ":" or not left_field.strip() or not right_field.strip():
        raise ValueError(
            "Field pairs must use the form 'left_field:right_field', "
            f"received '{raw_value}'."
        )
    return FieldPair(left_field=left_field.strip(), right_field=right_field.strip())


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, float):
        return math.isnan(value)
    return False


def merge_jsonl_records_by_id(input_paths: Sequence[Path]) -> list[dict[str, Any]]:
    merged_by_id: dict[str, dict[str, Any]] = {}
    record_order: list[str] = []

    for path in input_paths:
        for row in read_jsonl(path):
            record_id = row.get("id")
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(f"Every input row must contain a non-empty string 'id' field: {path}")

            if record_id not in merged_by_id:
                merged_by_id[record_id] = dict(row)
                record_order.append(record_id)
                continue

            merged_row = merged_by_id[record_id]
            for key, incoming_value in row.items():
                if key not in merged_row or _is_missing(merged_row[key]):
                    merged_row[key] = incoming_value
                    continue
                if _is_missing(incoming_value):
                    continue
                if merged_row[key] != incoming_value:
                    raise ValueError(
                        f"Conflicting values for id='{record_id}', field='{key}' while merging {path}."
                    )

    return [merged_by_id[record_id] for record_id in record_order]


def _require_text_field(record: dict[str, Any], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Record {record.get('id', '<unknown>')} is missing a valid '{field_name}' field."
        )
    return value


def score_record_pair(
    *,
    tokenizer,
    model,
    record: dict[str, Any],
    left_field: str,
    right_field: str,
    score_granularity: str,
    sentence_agg: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> tuple[float, list[dict[str, Any]] | None]:
    left_text = _require_text_field(record, left_field)
    right_text = _require_text_field(record, right_field)

    if score_granularity == "response":
        embeddings = embed_texts(
            tokenizer=tokenizer,
            model=model,
            texts=[left_text, right_text],
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        return cosine_distance(embeddings[0], embeddings[1]), None

    if score_granularity == "sentence":
        left_sentences = split_sentences(left_text)
        right_sentences = split_sentences(right_text)
        left_embeddings = embed_texts(
            tokenizer=tokenizer,
            model=model,
            texts=left_sentences,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        right_embeddings = embed_texts(
            tokenizer=tokenizer,
            model=model,
            texts=right_sentences,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        alignment = greedy_sentence_alignment(
            response_sentences=left_sentences,
            neutral_sentences=right_sentences,
            response_embeddings=left_embeddings,
            neutral_embeddings=right_embeddings,
        )
        candidates = [
            {
                "left_sentence": candidate["response_sentence"],
                "matched_right_sentence": candidate["matched_neutral_sentence"],
                "distance": candidate["distance"],
                "matched": candidate["matched"],
            }
            for candidate in top_sentence_candidates(alignment)
        ]
        return aggregate_sentence_distances(alignment, sentence_agg), candidates

    raise ValueError(f"Unsupported score granularity '{score_granularity}'.")


def summarize_pairwise_scores(
    rows: Sequence[dict[str, Any]],
    pairs: Sequence[FieldPair],
) -> dict[str, Any]:
    pair_summaries: dict[str, Any] = {}
    available_labels = sorted(
        {
            int(row["label"])
            for row in rows
            if row.get("label") is not None
        }
    )

    for pair in pairs:
        values = [
            float(row["pairwise_scores"][pair.key])
            for row in rows
            if pair.key in row.get("pairwise_scores", {})
        ]
        if not values:
            continue

        pair_summary: dict[str, Any] = {
            "left_field": pair.left_field,
            "right_field": pair.right_field,
            "count": len(values),
            "mean": sum(values) / len(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
        }

        if available_labels:
            by_label: dict[str, Any] = {}
            for label in available_labels:
                label_values = [
                    float(row["pairwise_scores"][pair.key])
                    for row in rows
                    if row.get("label") is not None
                    and int(row["label"]) == label
                    and pair.key in row.get("pairwise_scores", {})
                ]
                if not label_values:
                    continue
                by_label[str(label)] = {
                    "count": len(label_values),
                    "mean": sum(label_values) / len(label_values),
                    "median": median(label_values),
                    "min": min(label_values),
                    "max": max(label_values),
                }
            if by_label:
                pair_summary["by_label"] = by_label

        pair_summaries[pair.key] = pair_summary

    return {
        "records": len(rows),
        "pairs": pair_summaries,
    }
