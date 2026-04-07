from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .embedding_divergence import (
    calibrate_threshold,
    embed_texts,
    embedding_state_path,
    emit_progress,
    load_embedding_model,
)
from .evaluation_utils import metrics_dict
from .pairwise_distance import merge_jsonl_records_by_id

CLASSIFIER_BUNDLE_FILENAME = "anchor_distance_classifier.pkl"


@dataclass(frozen=True)
class AnchorDistancePrediction:
    label: int
    score: float
    feature_scores: dict[str, float]


@dataclass(frozen=True)
class AnchorDistanceTrainingConfig:
    embedding_model_name: str
    train_path: Path
    aux_train_path: Path
    output_dir: Path
    max_length: int
    batch_size: int
    device: str
    query_field: str
    response_field: str
    neutral_field: str
    aux_neutral_field: str
    threshold_metric: str
    score_granularity: str = "response"
    validation_path: Path | None = None
    aux_validation_path: Path | None = None


def classifier_bundle_path(model_dir: Path) -> Path:
    return model_dir / CLASSIFIER_BUNDLE_FILENAME


def load_classifier_bundle(model_dir: Path) -> Any:
    bundle_path = classifier_bundle_path(model_dir)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Anchor-distance classifier bundle not found: {bundle_path}")
    with bundle_path.open("rb") as handle:
        return pickle.load(handle)


def load_anchor_distance_state(model_dir: Path) -> dict[str, Any] | None:
    state_path = embedding_state_path(model_dir)
    if not state_path.exists():
        return None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Anchor-distance state must be a JSON object: {state_path}")
    return payload


def feature_pairs(
    *,
    query_field: str,
    response_field: str,
    neutral_field: str,
    aux_neutral_field: str,
) -> list[tuple[str, str]]:
    return [
        (query_field, neutral_field),
        (query_field, aux_neutral_field),
        (neutral_field, aux_neutral_field),
        (query_field, response_field),
        (neutral_field, response_field),
        (aux_neutral_field, response_field),
    ]


def pair_feature_name(left_field: str, right_field: str) -> str:
    return f"{left_field}__{right_field}_distance"


def _require_text_field(record: dict[str, Any], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Record {record.get('id', '<unknown>')} is missing a valid '{field_name}' field."
        )
    return value


def load_merged_records(primary_path: Path, secondary_path: Path) -> list[dict[str, Any]]:
    return merge_jsonl_records_by_id([primary_path, secondary_path])


def embed_record_fields(
    *,
    tokenizer,
    model,
    records: Sequence[dict[str, Any]],
    fields: Sequence[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> dict[str, torch.Tensor]:
    embeddings_by_field: dict[str, torch.Tensor] = {}
    for field_name in fields:
        texts = [_require_text_field(record, field_name) for record in records]
        embeddings_by_field[field_name] = embed_texts(
            tokenizer=tokenizer,
            model=model,
            texts=texts,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
    return embeddings_by_field


def batch_cosine_distances(left_embeddings: torch.Tensor, right_embeddings: torch.Tensor) -> torch.Tensor:
    similarities = torch.sum(left_embeddings * right_embeddings, dim=1).clamp(min=-1.0, max=1.0)
    return (1.0 - similarities).to(dtype=torch.float32)


def build_feature_columns(
    *,
    embeddings_by_field: dict[str, torch.Tensor],
    pairs: Sequence[tuple[str, str]],
) -> dict[str, torch.Tensor]:
    feature_columns: dict[str, torch.Tensor] = {}
    for left_field, right_field in pairs:
        feature_columns[pair_feature_name(left_field, right_field)] = batch_cosine_distances(
            embeddings_by_field[left_field],
            embeddings_by_field[right_field],
        )
    return feature_columns


def build_feature_rows(
    *,
    feature_columns: dict[str, torch.Tensor],
    feature_names: Sequence[str],
) -> list[list[float]]:
    if not feature_names:
        return []
    num_rows = len(feature_columns[feature_names[0]])
    column_values = {name: feature_columns[name].tolist() for name in feature_names}
    return [
        [float(column_values[name][row_index]) for name in feature_names]
        for row_index in range(num_rows)
    ]


def summarize_feature_columns(
    *,
    feature_columns: dict[str, torch.Tensor],
    labels: Sequence[int],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    available_labels = sorted(set(int(label) for label in labels))
    for feature_name, values_tensor in feature_columns.items():
        values = [float(value) for value in values_tensor.tolist()]
        feature_summary: dict[str, Any] = {
            "count": len(values),
            "mean": sum(values) / len(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
        }
        by_label: dict[str, Any] = {}
        for label in available_labels:
            label_values = [value for value, value_label in zip(values, labels) if int(value_label) == label]
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
            feature_summary["by_label"] = by_label
        summaries[feature_name] = feature_summary
    return summaries


def build_feature_dataset(
    *,
    tokenizer,
    model,
    records: Sequence[dict[str, Any]],
    query_field: str,
    response_field: str,
    neutral_field: str,
    aux_neutral_field: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> tuple[list[str], list[list[float]], dict[str, torch.Tensor]]:
    fields = [query_field, response_field, neutral_field, aux_neutral_field]
    embeddings_by_field = embed_record_fields(
        tokenizer=tokenizer,
        model=model,
        records=records,
        fields=fields,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    pairs = feature_pairs(
        query_field=query_field,
        response_field=response_field,
        neutral_field=neutral_field,
        aux_neutral_field=aux_neutral_field,
    )
    feature_names = [pair_feature_name(left_field, right_field) for left_field, right_field in pairs]
    feature_columns = build_feature_columns(
        embeddings_by_field=embeddings_by_field,
        pairs=pairs,
    )
    feature_rows = build_feature_rows(
        feature_columns=feature_columns,
        feature_names=feature_names,
    )
    return feature_names, feature_rows, feature_columns


def train_anchor_distance_classifier(config: AnchorDistanceTrainingConfig) -> dict[str, Any]:
    if config.score_granularity != "response":
        raise ValueError(
            "Anchor-distance classifier currently supports only score_granularity='response'."
        )
    if config.validation_path is not None and config.aux_validation_path is None:
        raise ValueError("Anchor-distance validation requires both validation_path and aux_validation_path.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(f"loading embedding model {config.embedding_model_name}")
    tokenizer, model = load_embedding_model(config.embedding_model_name, config.device)

    emit_progress(f"loading training rows from {config.train_path} + {config.aux_train_path}")
    train_records = load_merged_records(config.train_path, config.aux_train_path)
    emit_progress(f"embedding {len(train_records)} training rows for anchor-distance features")
    feature_names, train_features, train_feature_columns = build_feature_dataset(
        tokenizer=tokenizer,
        model=model,
        records=train_records,
        query_field=config.query_field,
        response_field=config.response_field,
        neutral_field=config.neutral_field,
        aux_neutral_field=config.aux_neutral_field,
        device=config.device,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )
    train_labels = [int(record["label"]) for record in train_records]

    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    emit_progress("fitting logistic regression over anchor-distance features")
    classifier.fit(train_features, train_labels)
    train_scores = [float(score) for score in classifier.predict_proba(train_features)[:, 1]]

    threshold_records = train_records
    threshold_scores = train_scores
    threshold_labels = train_labels
    threshold_source = "train"
    validation_summary = None
    validation_feature_summary = None

    if config.validation_path is not None and config.aux_validation_path is not None:
        emit_progress(
            f"loading calibration rows from {config.validation_path} + {config.aux_validation_path}"
        )
        validation_records = load_merged_records(config.validation_path, config.aux_validation_path)
        emit_progress(f"embedding {len(validation_records)} validation rows for threshold fitting")
        _, validation_features, validation_feature_columns = build_feature_dataset(
            tokenizer=tokenizer,
            model=model,
            records=validation_records,
            query_field=config.query_field,
            response_field=config.response_field,
            neutral_field=config.neutral_field,
            aux_neutral_field=config.aux_neutral_field,
            device=config.device,
            batch_size=config.batch_size,
            max_length=config.max_length,
        )
        validation_labels = [int(record["label"]) for record in validation_records]
        validation_scores = [
            float(score) for score in classifier.predict_proba(validation_features)[:, 1]
        ]
        threshold_records = validation_records
        threshold_scores = validation_scores
        threshold_labels = validation_labels
        threshold_source = "validation"
        validation_feature_summary = summarize_feature_columns(
            feature_columns=validation_feature_columns,
            labels=validation_labels,
        )
    else:
        validation_labels = None
        validation_scores = None

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

    classifier_step = classifier.named_steps["classifier"]
    coefficients = {
        feature_name: float(weight)
        for feature_name, weight in zip(feature_names, classifier_step.coef_[0], strict=True)
    }
    summary = {
        "trainer_type": "anchor_distance_classifier",
        "embedding_model_name": config.embedding_model_name,
        "train_path": str(config.train_path),
        "aux_train_path": str(config.aux_train_path),
        "validation_path": str(config.validation_path) if config.validation_path else None,
        "aux_validation_path": str(config.aux_validation_path) if config.aux_validation_path else None,
        "output_dir": str(config.output_dir),
        "device": config.device,
        "query_field": config.query_field,
        "response_field": config.response_field,
        "neutral_field": config.neutral_field,
        "aux_neutral_field": config.aux_neutral_field,
        "score_granularity": config.score_granularity,
        "threshold_metric": config.threshold_metric,
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "threshold_source": threshold_source,
        "threshold": threshold,
        "train_rows": len(train_records),
        "threshold_rows": len(threshold_records),
        "feature_names": feature_names,
        "feature_coefficients": coefficients,
        "intercept": float(classifier_step.intercept_[0]),
        "train_feature_summary": summarize_feature_columns(
            feature_columns=train_feature_columns,
            labels=train_labels,
        ),
        "validation_feature_summary": validation_feature_summary,
        "train_summary": train_summary,
        "threshold_summary": threshold_summary,
        "validation_summary": validation_summary,
    }

    emit_progress(f"writing anchor-distance classifier state to {config.output_dir}")
    with classifier_bundle_path(config.output_dir).open("wb") as handle:
        pickle.dump(classifier, handle)
    state_path = embedding_state_path(config.output_dir)
    state_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_path = config.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def score_records(
    *,
    classifier,
    tokenizer,
    model,
    records: Sequence[dict[str, Any]],
    query_field: str,
    response_field: str,
    neutral_field: str,
    aux_neutral_field: str,
    threshold: float,
    device: str,
    batch_size: int,
    max_length: int,
) -> list[AnchorDistancePrediction]:
    feature_names, feature_rows, feature_columns = build_feature_dataset(
        tokenizer=tokenizer,
        model=model,
        records=records,
        query_field=query_field,
        response_field=response_field,
        neutral_field=neutral_field,
        aux_neutral_field=aux_neutral_field,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    scores = [float(score) for score in classifier.predict_proba(feature_rows)[:, 1]]
    column_values = {feature_name: feature_columns[feature_name].tolist() for feature_name in feature_names}
    predictions: list[AnchorDistancePrediction] = []
    for index, score in enumerate(scores):
        feature_scores = {
            feature_name: float(column_values[feature_name][index])
            for feature_name in feature_names
        }
        predictions.append(
            AnchorDistancePrediction(
                label=1 if score >= threshold else 0,
                score=score,
                feature_scores=feature_scores,
            )
        )
    return predictions
