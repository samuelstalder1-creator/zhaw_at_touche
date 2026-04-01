from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel

from .evaluation_utils import metrics_dict
from .modeling import autocast_context, load_jsonl_rows, load_tokenizer_from_pretrained

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class DivergencePrediction:
    label: int
    score: float
    sentence_candidates: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class EmbeddingDivergenceTrainingConfig:
    embedding_model_name: str
    train_path: Path
    output_dir: Path
    max_length: int
    batch_size: int
    device: str
    neutral_field: str
    distance_metric: str
    score_granularity: str
    sentence_agg: str
    threshold_metric: str
    validation_path: Path | None = None


def embedding_state_path(model_dir: Path) -> Path:
    return model_dir / "embedding_state.json"


def emit_progress(message: str) -> None:
    print(message, flush=True)


def split_sentences(text: str) -> list[str]:
    chunks = [chunk.strip() for chunk in SENTENCE_SPLIT_RE.split(text) if chunk.strip()]
    return chunks or [text.strip()] if text.strip() else []


def mean_pool_embeddings(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape).float()
    masked = last_hidden_state * mask
    token_counts = mask.sum(dim=1).clamp(min=1e-9)
    pooled = masked.sum(dim=1) / token_counts
    return F.normalize(pooled, p=2, dim=1)


def load_embedding_model(model_name: str, device: str):
    tokenizer = load_tokenizer_from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def embed_texts(
    *,
    tokenizer,
    model,
    texts: Sequence[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    if not texts:
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not determine embedding dimension from the embedding model.")
        return torch.empty((0, hidden_size), dtype=torch.float32)

    embeddings: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            tokenized = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            attention_mask = tokenized["attention_mask"]
            inputs = {key: value.to(device) for key, value in tokenized.items()}
            with autocast_context(device):
                outputs = model(**inputs)
            pooled = mean_pool_embeddings(outputs.last_hidden_state, attention_mask.to(device))
            embeddings.append(pooled.detach().cpu())
    return torch.cat(embeddings, dim=0)


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 or right.numel() == 0:
        return 1.0
    similarity = torch.sum(left * right).clamp(min=-1.0, max=1.0)
    return float(1.0 - similarity.item())


def greedy_sentence_alignment(
    *,
    response_sentences: Sequence[str],
    neutral_sentences: Sequence[str],
    response_embeddings: torch.Tensor,
    neutral_embeddings: torch.Tensor,
) -> list[dict[str, Any]]:
    if not response_sentences:
        return []
    if not neutral_sentences or neutral_embeddings.numel() == 0:
        return [
            {
                "response_sentence": sentence,
                "matched_neutral_sentence": None,
                "distance": 1.0,
                "matched": False,
            }
            for sentence in response_sentences
        ]

    distance_matrix = 1.0 - response_embeddings @ neutral_embeddings.T
    unmatched_responses = set(range(len(response_sentences)))
    unmatched_neutrals = set(range(len(neutral_sentences)))
    assignments: dict[int, dict[str, Any]] = {}

    while unmatched_responses and unmatched_neutrals:
        best_pair: tuple[int, int] | None = None
        best_distance = float("inf")
        for response_index in unmatched_responses:
            for neutral_index in unmatched_neutrals:
                distance = float(distance_matrix[response_index, neutral_index].item())
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (response_index, neutral_index)

        if best_pair is None:
            break

        response_index, neutral_index = best_pair
        assignments[response_index] = {
            "response_sentence": response_sentences[response_index],
            "matched_neutral_sentence": neutral_sentences[neutral_index],
            "distance": best_distance,
            "matched": True,
        }
        unmatched_responses.remove(response_index)
        unmatched_neutrals.remove(neutral_index)

    for response_index in unmatched_responses:
        assignments[response_index] = {
            "response_sentence": response_sentences[response_index],
            "matched_neutral_sentence": None,
            "distance": 1.0,
            "matched": False,
        }

    return [assignments[index] for index in range(len(response_sentences))]


def aggregate_sentence_distances(alignment: Sequence[dict[str, Any]], sentence_agg: str) -> float:
    if not alignment:
        return 0.0
    distances = [float(candidate["distance"]) for candidate in alignment]
    if sentence_agg == "max":
        return max(distances)
    if sentence_agg == "mean":
        return sum(distances) / len(distances)
    if sentence_agg == "top2_mean":
        top_distances = sorted(distances, reverse=True)[:2]
        return sum(top_distances) / len(top_distances)
    if sentence_agg == "top3_mean":
        top_distances = sorted(distances, reverse=True)[:3]
        return sum(top_distances) / len(top_distances)
    raise ValueError(f"Unsupported sentence aggregation '{sentence_agg}'.")


def top_sentence_candidates(
    alignment: Sequence[dict[str, Any]],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    ranked = sorted(alignment, key=lambda item: float(item["distance"]), reverse=True)
    return list(ranked[:limit])


def calibrate_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    threshold_metric: str,
) -> tuple[float, dict[str, Any]]:
    if len(scores) != len(labels):
        raise ValueError("Scores and labels must have the same length for calibration.")
    if not scores:
        raise ValueError("At least one calibration score is required.")

    unique_scores = sorted(set(float(score) for score in scores))
    candidate_thresholds = [unique_scores[0] - 1e-6]
    candidate_thresholds.extend(
        (left + right) / 2.0
        for left, right in zip(unique_scores, unique_scores[1:])
    )
    candidate_thresholds.append(unique_scores[-1] + 1e-6)

    best_threshold = candidate_thresholds[0]
    best_summary: dict[str, Any] | None = None
    best_metric = float("-inf")
    best_accuracy = float("-inf")

    for threshold in candidate_thresholds:
        predictions = [1 if score >= threshold else 0 for score in scores]
        summary = metrics_dict(labels, predictions)
        positive_label = summary.get("positive_label")
        macro = summary.get("macro")
        positive_f1 = 0.0
        if isinstance(positive_label, dict):
            positive_f1 = float(positive_label["f1"])
        macro_f1 = 0.0
        if isinstance(macro, dict):
            macro_f1 = float(macro["f1"])

        if threshold_metric == "positive_f1":
            metric_value = positive_f1
        elif threshold_metric == "macro_f1":
            metric_value = macro_f1
        elif threshold_metric == "accuracy":
            metric_value = float(summary["accuracy"])
        else:
            raise ValueError(f"Unsupported threshold metric '{threshold_metric}'.")

        accuracy_value = float(summary["accuracy"])
        if metric_value > best_metric or (
            metric_value == best_metric and accuracy_value > best_accuracy
        ):
            best_threshold = threshold
            best_summary = summary
            best_metric = metric_value
            best_accuracy = accuracy_value

    if best_summary is None:
        raise ValueError("Could not calibrate a threshold.")
    return best_threshold, best_summary


def record_scores(
    *,
    tokenizer,
    model,
    records: Sequence[dict[str, Any]],
    neutral_field: str,
    score_granularity: str,
    sentence_agg: str,
    device: str,
    batch_size: int,
    max_length: int,
    progress_label: str | None = None,
) -> list[float]:
    scores: list[float] = []
    iterator = records
    if progress_label:
        iterator = tqdm(records, desc=progress_label, unit="record")
    for record in iterator:
        score, _ = score_record(
            tokenizer=tokenizer,
            model=model,
            record=record,
            neutral_field=neutral_field,
            score_granularity=score_granularity,
            sentence_agg=sentence_agg,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        scores.append(score)
    return scores


def load_embedding_state(model_dir: Path) -> dict[str, Any] | None:
    state_path = embedding_state_path(model_dir)
    if not state_path.exists():
        return None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Embedding state must be a JSON object: {state_path}")
    return payload


def train_embedding_divergence(config: EmbeddingDivergenceTrainingConfig) -> dict[str, Any]:
    if config.distance_metric != "cosine":
        raise ValueError(f"Unsupported distance metric '{config.distance_metric}'.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(f"loading embedding model {config.embedding_model_name}")
    tokenizer, model = load_embedding_model(config.embedding_model_name, config.device)
    emit_progress(f"loading training rows from {config.train_path}")
    train_records = load_jsonl_rows(config.train_path)
    emit_progress(
        f"scoring {len(train_records)} training rows "
        f"({config.score_granularity}, neutral_field={config.neutral_field})"
    )
    train_scores = record_scores(
        tokenizer=tokenizer,
        model=model,
        records=train_records,
        neutral_field=config.neutral_field,
        score_granularity=config.score_granularity,
        sentence_agg=config.sentence_agg,
        device=config.device,
        batch_size=config.batch_size,
        max_length=config.max_length,
        progress_label="setup100 train",
    )
    train_labels = [int(record["label"]) for record in train_records]

    threshold_records = train_records
    threshold_scores = train_scores
    threshold_labels = train_labels
    threshold_source = "train"

    validation_summary = None
    validation_scores: list[float] | None = None
    validation_labels: list[int] | None = None
    if config.validation_path is not None:
        emit_progress(f"loading calibration rows from {config.validation_path}")
        validation_records = load_jsonl_rows(config.validation_path)
        emit_progress(
            f"scoring {len(validation_records)} validation rows for threshold fitting"
        )
        validation_scores = record_scores(
            tokenizer=tokenizer,
            model=model,
            records=validation_records,
            neutral_field=config.neutral_field,
            score_granularity=config.score_granularity,
            sentence_agg=config.sentence_agg,
            device=config.device,
            batch_size=config.batch_size,
            max_length=config.max_length,
            progress_label="setup100 validation",
        )
        validation_labels = [int(record["label"]) for record in validation_records]
        threshold_records = validation_records
        threshold_scores = validation_scores
        threshold_labels = validation_labels
        threshold_source = "validation"

    emit_progress(
        f"fitting threshold on {len(threshold_records)} {threshold_source} rows "
        f"using metric={config.threshold_metric}"
    )
    threshold, threshold_summary = calibrate_threshold(
        threshold_scores,
        threshold_labels,
        threshold_metric=config.threshold_metric,
    )
    emit_progress(f"selected threshold={threshold:.6f} ({threshold_source})")

    train_predictions = [1 if score >= threshold else 0 for score in train_scores]
    train_summary = metrics_dict(train_labels, train_predictions)
    if validation_scores is not None and validation_labels is not None:
        validation_predictions = [1 if score >= threshold else 0 for score in validation_scores]
        validation_summary = metrics_dict(validation_labels, validation_predictions)

    summary = {
        "trainer_type": "embedding_divergence",
        "embedding_model_name": config.embedding_model_name,
        "train_path": str(config.train_path),
        "validation_path": str(config.validation_path) if config.validation_path else None,
        "output_dir": str(config.output_dir),
        "device": config.device,
        "neutral_field": config.neutral_field,
        "distance_metric": config.distance_metric,
        "score_granularity": config.score_granularity,
        "sentence_agg": config.sentence_agg,
        "threshold_metric": config.threshold_metric,
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "threshold_source": threshold_source,
        "threshold": threshold,
        "train_rows": len(train_records),
        "threshold_rows": len(threshold_records),
        "train_summary": train_summary,
        "threshold_summary": threshold_summary,
        "validation_summary": validation_summary,
    }
    emit_progress(f"writing embedding-divergence state to {config.output_dir}")
    state_path = embedding_state_path(config.output_dir)
    state_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_path = config.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def score_record(
    *,
    tokenizer,
    model,
    record: dict[str, Any],
    neutral_field: str,
    score_granularity: str,
    sentence_agg: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> tuple[float, list[dict[str, Any]] | None]:
    response = record.get("response")
    neutral = record.get(neutral_field)
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"Record {record.get('id', '<unknown>')} is missing a valid 'response' field.")
    if not isinstance(neutral, str) or not neutral.strip():
        raise ValueError(
            f"Record {record.get('id', '<unknown>')} is missing a valid '{neutral_field}' field."
        )

    if score_granularity == "response":
        embeddings = embed_texts(
            tokenizer=tokenizer,
            model=model,
            texts=[neutral, response],
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        return cosine_distance(embeddings[0], embeddings[1]), None

    if score_granularity == "sentence":
        neutral_sentences = split_sentences(neutral)
        response_sentences = split_sentences(response)
        neutral_embeddings = embed_texts(
            tokenizer=tokenizer,
            model=model,
            texts=neutral_sentences,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        response_embeddings = embed_texts(
            tokenizer=tokenizer,
            model=model,
            texts=response_sentences,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        alignment = greedy_sentence_alignment(
            response_sentences=response_sentences,
            neutral_sentences=neutral_sentences,
            response_embeddings=response_embeddings,
            neutral_embeddings=neutral_embeddings,
        )
        return aggregate_sentence_distances(alignment, sentence_agg), top_sentence_candidates(alignment)

    raise ValueError(f"Unsupported score granularity '{score_granularity}'.")


def score_records(
    *,
    tokenizer,
    model,
    records: Sequence[dict[str, Any]],
    neutral_field: str,
    score_granularity: str,
    sentence_agg: str,
    threshold: float,
    device: str,
    batch_size: int,
    max_length: int,
) -> list[DivergencePrediction]:
    predictions: list[DivergencePrediction] = []
    for record in records:
        score, candidates = score_record(
            tokenizer=tokenizer,
            model=model,
            record=record,
            neutral_field=neutral_field,
            score_granularity=score_granularity,
            sentence_agg=sentence_agg,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        label = 1 if score >= threshold else 0
        predictions.append(
            DivergencePrediction(
                label=label,
                score=score,
                sentence_candidates=candidates,
            )
        )
    return predictions
