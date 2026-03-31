from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel

from .evaluation_utils import metrics_dict
from .modeling import autocast_context, load_tokenizer_from_pretrained

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class DivergencePrediction:
    label: int
    score: float
    sentence_candidates: list[dict[str, Any]] | None = None


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
        positive_f1 = 0.0
        if isinstance(positive_label, dict):
            positive_f1 = float(positive_label["f1"])

        if threshold_metric == "positive_f1":
            metric_value = positive_f1
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
