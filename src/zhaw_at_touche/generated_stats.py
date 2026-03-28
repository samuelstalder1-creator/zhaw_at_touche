from __future__ import annotations

import html
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .datasets import detect_generated_text_field, word_count
from .jsonl import read_jsonl

NEUTRAL_RESPONSE_TOKENS_METRIC = "neutral_response_tokens"
ALL_TOGETHER_TOKENS_METRIC = "all_together_tokens"


@dataclass(frozen=True)
class GeneratedRow:
    row_id: str
    query: str
    response: str
    generated_response: str


@dataclass(frozen=True)
class MetricSummary:
    name: str
    total: int
    average: float
    median: float
    min_value: int
    min_id: str
    max_value: int
    max_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricAnalysis:
    summaries: list[MetricSummary]
    metric_values: dict[str, list[int]]


def load_generated_rows(
    path: Path,
    generated_field: str | None,
    limit: int = 0,
) -> tuple[str, list[GeneratedRow]]:
    iterator = read_jsonl(path)
    try:
        first_row = next(iterator)
    except StopIteration as exc:
        raise ValueError(f"Input file is empty: {path}") from exc

    resolved_generated_field = detect_generated_text_field(first_row, generated_field)
    rows: list[GeneratedRow] = []
    skipped = 0

    for raw_row in read_jsonl(path):
        if limit and len(rows) >= limit:
            break

        row_id = raw_row.get("id")
        query = raw_row.get("query")
        response = raw_row.get("response")
        generated_response = raw_row.get(resolved_generated_field)

        if not isinstance(row_id, str):
            skipped += 1
            continue
        if not isinstance(query, str) or not isinstance(response, str):
            skipped += 1
            continue
        if not isinstance(generated_response, str):
            skipped += 1
            continue

        rows.append(
            GeneratedRow(
                row_id=row_id,
                query=query,
                response=response,
                generated_response=generated_response,
            )
        )

    if skipped:
        print(f"[warn] Skipped {skipped} malformed rows from {path}")

    return resolved_generated_field, rows


def summarize_metric(name: str, values: list[int], row_ids: list[str]) -> MetricSummary:
    if not values:
        raise ValueError(f"No values available for metric '{name}'.")
    min_index = min(range(len(values)), key=values.__getitem__)
    max_index = max(range(len(values)), key=values.__getitem__)
    return MetricSummary(
        name=name,
        total=sum(values),
        average=sum(values) / len(values),
        median=float(statistics.median(values)),
        min_value=values[min_index],
        min_id=row_ids[min_index],
        max_value=values[max_index],
        max_id=row_ids[max_index],
    )


def basic_length_summaries(rows: list[GeneratedRow]) -> list[MetricSummary]:
    row_ids = [row.row_id for row in rows]
    query_words = [word_count(row.query) for row in rows]
    response_words = [word_count(row.response) for row in rows]
    generated_words = [word_count(row.generated_response) for row in rows]
    combined_words = [
        query + response + generated
        for query, response, generated in zip(query_words, response_words, generated_words)
    ]
    query_chars = [len(row.query) for row in rows]
    response_chars = [len(row.response) for row in rows]
    generated_chars = [len(row.generated_response) for row in rows]

    return [
        summarize_metric("query_words", query_words, row_ids),
        summarize_metric("response_words", response_words, row_ids),
        summarize_metric("generated_words", generated_words, row_ids),
        summarize_metric("combined_words", combined_words, row_ids),
        summarize_metric("query_chars", query_chars, row_ids),
        summarize_metric("response_chars", response_chars, row_ids),
        summarize_metric("generated_chars", generated_chars, row_ids),
    ]


def build_histogram(values: list[int], requested_bins: int) -> tuple[list[int], list[tuple[float, float]], int, int]:
    if not values:
        raise ValueError("Histogram values must not be empty.")
    if requested_bins < 1:
        raise ValueError("Histogram bin count must be >= 1.")

    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        return [len(values)], [(float(min_value), float(max_value))], min_value, max_value

    value_range = max_value - min_value
    bin_count = min(requested_bins, value_range + 1)
    bin_width = value_range / bin_count

    counts = [0] * bin_count
    ranges: list[tuple[float, float]] = []
    for index in range(bin_count):
        start = min_value + index * bin_width
        end = min_value + (index + 1) * bin_width if index < bin_count - 1 else float(max_value)
        ranges.append((start, end))

    for value in values:
        normalized = (value - min_value) / value_range
        bin_index = min(int(normalized * bin_count), bin_count - 1)
        counts[bin_index] += 1

    return counts, ranges, min_value, max_value


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.1f}"


def _value_to_x(value: float, plot_x: float, plot_width: float, min_value: float, max_value: float) -> float:
    if max_value == min_value:
        return plot_x + plot_width / 2
    return plot_x + ((value - min_value) / (max_value - min_value)) * plot_width


def render_histogram_panel(
    metric_name: str,
    values: list[int],
    summary: MetricSummary,
    panel_x: int,
    panel_y: int,
    panel_width: int,
    panel_height: int,
    bins: int,
) -> str:
    outer_margin = 18
    plot_left = panel_x + 56
    plot_top = panel_y + 52
    plot_width = panel_width - 78
    plot_height = panel_height - 114
    plot_bottom = plot_top + plot_height
    plot_right = plot_left + plot_width

    counts, ranges, min_value, max_value = build_histogram(values, bins)
    max_count = max(counts) if counts else 1
    bar_width = plot_width / max(len(counts), 1)

    parts = [
        f'<rect x="{panel_x}" y="{panel_y}" width="{panel_width}" height="{panel_height}" rx="12" fill="#ffffff" stroke="#d0d7de" />',
        f'<text x="{panel_x + outer_margin}" y="{panel_y + 28}" font-size="18" font-weight="700" fill="#0f172a">{html.escape(metric_name)}</text>',
        (
            f'<text x="{panel_x + outer_margin}" y="{panel_y + 46}" font-size="12" fill="#475569">'
            f"avg {_format_number(summary.average)} | median {_format_number(summary.median)} | "
            f"min {summary.min_value:,} | max {summary.max_value:,}</text>"
        ),
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#334155" stroke-width="1" />',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#334155" stroke-width="1" />',
    ]

    for grid_index in range(5):
        y = plot_top + (plot_height / 4) * grid_index
        grid_value = round(max_count * (1 - grid_index / 4))
        parts.append(
            f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" stroke="#e2e8f0" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{plot_left - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#64748b">{grid_value}</text>'
        )

    for index, count in enumerate(counts):
        bar_height = 0 if max_count == 0 else (count / max_count) * plot_height
        x = plot_left + index * bar_width
        y = plot_bottom - bar_height
        width = max(bar_width - 1, 1)
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{bar_height:.2f}" fill="#3b82f6" opacity="0.85" />'
        )

    min_label = _format_number(min_value)
    mid_label = _format_number((min_value + max_value) / 2)
    max_label = _format_number(max_value)
    for label, x, anchor in (
        (min_label, plot_left, "start"),
        (mid_label, plot_left + plot_width / 2, "middle"),
        (max_label, plot_right, "end"),
    ):
        parts.append(
            f'<text x="{x:.2f}" y="{plot_bottom + 22}" text-anchor="{anchor}" font-size="11" fill="#64748b">{label}</text>'
        )

    mean_x = _value_to_x(summary.average, plot_left, plot_width, min_value, max_value)
    median_x = _value_to_x(summary.median, plot_left, plot_width, min_value, max_value)
    parts.append(
        f'<line x1="{mean_x:.2f}" y1="{plot_top}" x2="{mean_x:.2f}" y2="{plot_bottom}" stroke="#ef4444" stroke-width="2" stroke-dasharray="6 4" />'
    )
    parts.append(
        f'<line x1="{median_x:.2f}" y1="{plot_top}" x2="{median_x:.2f}" y2="{plot_bottom}" stroke="#16a34a" stroke-width="2" stroke-dasharray="2 3" />'
    )
    parts.append(
        f'<text x="{plot_right}" y="{plot_top - 10}" text-anchor="end" font-size="11" fill="#64748b">red=avg, green=median</text>'
    )

    first_range = ranges[0]
    last_range = ranges[-1]
    range_label = f"bins={len(counts)} | from {_format_number(first_range[0])} to {_format_number(last_range[1])}"
    parts.append(
        f'<text x="{panel_x + outer_margin}" y="{panel_y + panel_height - 16}" font-size="11" fill="#64748b">{range_label}</text>'
    )
    return "\n".join(parts)


def histogram_output_path(input_path: Path, histogram_dir: Path) -> Path:
    return histogram_dir / f"{input_path.stem}-token-histogram.svg"


def write_histogram_svg(
    path: Path,
    input_path: Path,
    metric_values: dict[str, list[int]],
    summaries: list[MetricSummary],
    row_count: int,
    bins: int,
) -> None:
    panel_width = 640
    panel_height = 320
    gap_x = 24
    gap_y = 24
    margin_x = 28
    margin_y = 96
    width = margin_x * 2 + panel_width * 2 + gap_x
    height = margin_y + panel_height * 2 + gap_y + 28

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc" />',
        f'<text x="{margin_x}" y="38" font-size="28" font-weight="700" fill="#0f172a">Token Count Distribution</text>',
        (
            f'<text x="{margin_x}" y="62" font-size="14" fill="#475569">'
            f"{html.escape(input_path.name)} | rows={row_count:,}</text>"
        ),
    ]

    metric_order = [
        "query_tokens",
        "response_tokens",
        NEUTRAL_RESPONSE_TOKENS_METRIC,
        ALL_TOGETHER_TOKENS_METRIC,
    ]
    summary_map = {summary.name: summary for summary in summaries}

    for index, metric_name in enumerate(metric_order):
        col = index % 2
        row = index // 2
        panel_x = margin_x + col * (panel_width + gap_x)
        panel_y = margin_y + row * (panel_height + gap_y)
        parts.append(
            render_histogram_panel(
                metric_name=metric_name,
                values=metric_values[metric_name],
                summary=summary_map[metric_name],
                panel_x=panel_x,
                panel_y=panel_y,
                panel_width=panel_width,
                panel_height=panel_height,
                bins=bins,
            )
        )

    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _count_texts(tokenizer: Any, texts: list[str]) -> list[int]:
    result = tokenizer.compute_tokens(texts)
    tokens_info = getattr(result, "tokens_info", None) or []
    if len(tokens_info) != len(texts):
        raise RuntimeError(
            "Tokenizer returned an unexpected number of token groups. "
            "Try lowering --batch-size."
        )

    counts: list[int] = []
    for tokens in tokens_info:
        token_ids = getattr(tokens, "token_ids", None)
        if token_ids is not None:
            counts.append(len(token_ids))
            continue
        raw_tokens = getattr(tokens, "tokens", None)
        if raw_tokens is not None:
            counts.append(len(raw_tokens))
            continue
        raise RuntimeError("Tokenizer output did not contain token ids or token strings.")
    return counts


def token_length_analysis(
    rows: list[GeneratedRow],
    model_name: str,
    batch_size: int,
) -> MetricAnalysis:
    from google.genai.local_tokenizer import LocalTokenizer

    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    tokenizer = LocalTokenizer(model_name)
    row_ids = [row.row_id for row in rows]
    query_tokens: list[int] = []
    response_tokens: list[int] = []
    neutral_response_tokens: list[int] = []
    all_together_tokens: list[int] = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts: list[str] = []
        for row in batch:
            texts.extend([row.query, row.response, row.generated_response])
        counts = _count_texts(tokenizer, texts)
        for index in range(0, len(counts), 3):
            query_count = counts[index]
            response_count = counts[index + 1]
            generated_count = counts[index + 2]
            query_tokens.append(query_count)
            response_tokens.append(response_count)
            neutral_response_tokens.append(generated_count)
            all_together_tokens.append(query_count + response_count + generated_count)

    metric_values = {
        "query_tokens": query_tokens,
        "response_tokens": response_tokens,
        NEUTRAL_RESPONSE_TOKENS_METRIC: neutral_response_tokens,
        ALL_TOGETHER_TOKENS_METRIC: all_together_tokens,
    }
    return MetricAnalysis(
        summaries=[
            summarize_metric("query_tokens", query_tokens, row_ids),
            summarize_metric("response_tokens", response_tokens, row_ids),
            # Keep legacy metric names stable so old reports and histograms remain comparable.
            summarize_metric(NEUTRAL_RESPONSE_TOKENS_METRIC, neutral_response_tokens, row_ids),
            summarize_metric(ALL_TOGETHER_TOKENS_METRIC, all_together_tokens, row_ids),
        ],
        metric_values=metric_values,
    )


def token_length_summaries(
    rows: list[GeneratedRow],
    model_name: str,
    batch_size: int,
) -> list[MetricSummary]:
    return token_length_analysis(rows, model_name, batch_size).summaries


def summaries_to_dict(summaries: list[MetricSummary]) -> list[dict[str, Any]]:
    return [summary.to_dict() for summary in summaries]
