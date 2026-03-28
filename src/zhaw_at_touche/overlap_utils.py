from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Hashable

from .jsonl import read_jsonl

DEFAULT_OVERLAP_FIELDS = ("id", "query", "response", "query+response")


@dataclass(frozen=True)
class RowRef:
    row_id: str
    query: str
    response: str


@dataclass(frozen=True)
class OverlapSample:
    key_text: str
    split_ids: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class OverlapComparison:
    split_names: tuple[str, ...]
    overlap_count: int
    samples: list[OverlapSample]

    @property
    def label(self) -> str:
        return " vs ".join(self.split_names)


def load_split(path: Path) -> list[RowRef]:
    rows: list[RowRef] = []
    for row in read_jsonl(path):
        row_id = row.get("id")
        query = row.get("query")
        response = row.get("response")
        if not isinstance(row_id, str):
            continue
        if not isinstance(query, str) or not isinstance(response, str):
            continue
        rows.append(RowRef(row_id=row_id, query=query, response=response))
    return rows


def shorten(text: str, width: int = 120) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= width:
        return one_line
    return one_line[: width - 3] + "..."


def format_key(field_name: str, key: object) -> str:
    if field_name == "id":
        return str(key)
    if field_name in {"query", "response"}:
        return shorten(str(key))
    if field_name == "query+response":
        query, response = key
        return f"query={shorten(str(query), 80)} | response={shorten(str(response), 80)}"
    raise ValueError(f"Unsupported overlap field: {field_name}")


def overlap_key_fn(field_name: str) -> Callable[[RowRef], Hashable]:
    key_fn_map: dict[str, Callable[[RowRef], Hashable]] = {
        "id": lambda row: row.row_id,
        "query": lambda row: row.query,
        "response": lambda row: row.response,
        "query+response": lambda row: (row.query, row.response),
    }
    try:
        return key_fn_map[field_name]
    except KeyError as exc:
        joined = ", ".join(DEFAULT_OVERLAP_FIELDS)
        raise ValueError(f"Unsupported overlap field '{field_name}'. Expected one of: {joined}") from exc


def build_index(
    rows: list[RowRef],
    key_fn: Callable[[RowRef], Hashable],
) -> dict[Hashable, list[RowRef]]:
    index: dict[Hashable, list[RowRef]] = {}
    for row in rows:
        key = key_fn(row)
        index.setdefault(key, []).append(row)
    return index


def dataset_sizes(split_rows: dict[str, list[RowRef]]) -> dict[str, int]:
    return {split_name: len(rows) for split_name, rows in split_rows.items()}


def collect_overlap_report(
    field_name: str,
    split_rows: dict[str, list[RowRef]],
    sample_limit: int,
) -> list[OverlapComparison]:
    if sample_limit < 0:
        raise ValueError("--sample-limit must be >= 0.")

    key_fn = overlap_key_fn(field_name)
    split_indexes = {
        split_name: build_index(rows, key_fn)
        for split_name, rows in split_rows.items()
    }
    split_keys = {
        split_name: set(index.keys())
        for split_name, index in split_indexes.items()
    }

    comparisons: list[OverlapComparison] = []
    comparison_sets = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
        ("train", "validation", "test"),
    ]
    for split_names in comparison_sets:
        overlap_keys = set.intersection(*(split_keys[name] for name in split_names))
        samples: list[OverlapSample] = []
        for overlap_key in sorted(overlap_keys, key=lambda value: format_key(field_name, value))[:sample_limit]:
            samples.append(
                OverlapSample(
                    key_text=format_key(field_name, overlap_key),
                    split_ids={
                        split_name: tuple(row.row_id for row in split_indexes[split_name][overlap_key][:3])
                        for split_name in split_names
                    },
                )
            )
        comparisons.append(
            OverlapComparison(
                split_names=split_names,
                overlap_count=len(overlap_keys),
                samples=samples,
            )
        )
    return comparisons
