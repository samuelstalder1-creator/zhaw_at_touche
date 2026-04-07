from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from .anchor_distance_classifier import (
    build_feature_dataset,
    load_merged_records,
    pair_feature_name,
    summarize_feature_columns,
)
from .embedding_divergence import (
    calibrate_threshold,
    embedding_state_path,
    emit_progress,
    load_embedding_model,
)
from .evaluation_utils import metrics_dict


@dataclass(frozen=True)
class AnchorDistanceThresholdPrediction:
    label: int
    score: float
    anchor_cohesion: float
    response_drift: float
    feature_scores: dict[str, float]


@dataclass(frozen=True)
class AnchorDistanceThresholdTrainingConfig:
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


def load_anchor_distance_threshold_state(model_dir: Path) -> dict[str, Any] | None:
    state_path = embedding_state_path(model_dir)
    if not state_path.exists():
        return None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Anchor-distance threshold state must be a JSON object: {state_path}")
    return payload


def score_formula(
    *,
    query_field: str,
    response_field: str,
    neutral_field: str,
    aux_neutral_field: str,
) -> dict[str, Any]:
    return {
        "anchor_cohesion_mean": [
            pair_feature_name(query_field, neutral_field),
            pair_feature_name(query_field, aux_neutral_field),
            pair_feature_name(neutral_field, aux_neutral_field),
        ],
        "response_drift_mean": [
            pair_feature_name(query_field, response_field),
            pair_feature_name(neutral_field, response_field),
            pair_feature_name(aux_neutral_field, response_field),
        ],
        "final_score": "response_drift - anchor_cohesion",
    }


def derive_score_columns(
    *,
    feature_columns: dict[str, torch.Tensor],
    query_field: str,
    response_field: str,
    neutral_field: str,
    aux_neutral_field: str,
) -> dict[str, torch.Tensor]:
    formula = score_formula(
        query_field=query_field,
        response_field=response_field,
        neutral_field=neutral_field,
        aux_neutral_field=aux_neutral_field,
    )
    anchor_cohesion = torch.stack(
        [feature_columns[feature_name] for feature_name in formula["anchor_cohesion_mean"]],
        dim=1,
    ).mean(dim=1)
    response_drift = torch.stack(
        [feature_columns[feature_name] for feature_name in formula["response_drift_mean"]],
        dim=1,
    ).mean(dim=1)
    return {
        "anchor_cohesion": anchor_cohesion,
        "response_drift": response_drift,
        "anchor_distance_score": response_drift - anchor_cohesion,
    }


def score_records(
    *,
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
    progress_prefix: str | None = None,
) -> list[AnchorDistanceThresholdPrediction]:
    feature_names, _, feature_columns = build_feature_dataset(
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
        progress_prefix=progress_prefix,
    )
    derived_columns = derive_score_columns(
        feature_columns=feature_columns,
        query_field=query_field,
        response_field=response_field,
        neutral_field=neutral_field,
        aux_neutral_field=aux_neutral_field,
    )

    column_values = {feature_name: feature_columns[feature_name].tolist() for feature_name in feature_names}
    derived_values = {name: tensor.tolist() for name, tensor in derived_columns.items()}
    predictions: list[AnchorDistanceThresholdPrediction] = []
    for index, score in enumerate(derived_values["anchor_distance_score"]):
        feature_scores = {
            feature_name: float(column_values[feature_name][index])
            for feature_name in feature_names
        }
        predictions.append(
            AnchorDistanceThresholdPrediction(
                label=1 if float(score) >= threshold else 0,
                score=float(score),
                anchor_cohesion=float(derived_values["anchor_cohesion"][index]),
                response_drift=float(derived_values["response_drift"][index]),
                feature_scores=feature_scores,
            )
        )
    return predictions


def train_anchor_distance_threshold(config: AnchorDistanceThresholdTrainingConfig) -> dict[str, Any]:
    if config.score_granularity != "response":
        raise ValueError(
            "Anchor-distance threshold currently supports only score_granularity='response'."
        )
    if config.validation_path is not None and config.aux_validation_path is None:
        raise ValueError("Anchor-distance threshold validation requires both validation_path and aux_validation_path.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(f"loading embedding model {config.embedding_model_name}")
    tokenizer, model = load_embedding_model(config.embedding_model_name, config.device)

    emit_progress(f"loading training rows from {config.train_path} + {config.aux_train_path}")
    train_records = load_merged_records(config.train_path, config.aux_train_path)
    emit_progress(f"embedding {len(train_records)} training rows for anchor-distance threshold features")
    feature_names, _, train_feature_columns = build_feature_dataset(
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
        progress_prefix="setup111 train",
    )
    train_derived_columns = derive_score_columns(
        feature_columns=train_feature_columns,
        query_field=config.query_field,
        response_field=config.response_field,
        neutral_field=config.neutral_field,
        aux_neutral_field=config.aux_neutral_field,
    )
    train_labels = [int(record["label"]) for record in train_records]
    train_scores = [float(score) for score in train_derived_columns["anchor_distance_score"].tolist()]

    threshold_records = train_records
    threshold_scores = train_scores
    threshold_labels = train_labels
    threshold_source = "train"
    validation_summary = None
    validation_feature_summary = None
    validation_score_summary = None

    if config.validation_path is not None and config.aux_validation_path is not None:
        emit_progress(
            f"loading calibration rows from {config.validation_path} + {config.aux_validation_path}"
        )
        validation_records = load_merged_records(config.validation_path, config.aux_validation_path)
        emit_progress(f"embedding {len(validation_records)} validation rows for threshold fitting")
        _, _, validation_feature_columns = build_feature_dataset(
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
            progress_prefix="setup111 validation",
        )
        validation_derived_columns = derive_score_columns(
            feature_columns=validation_feature_columns,
            query_field=config.query_field,
            response_field=config.response_field,
            neutral_field=config.neutral_field,
            aux_neutral_field=config.aux_neutral_field,
        )
        validation_labels = [int(record["label"]) for record in validation_records]
        validation_scores = [
            float(score)
            for score in validation_derived_columns["anchor_distance_score"].tolist()
        ]
        threshold_records = validation_records
        threshold_scores = validation_scores
        threshold_labels = validation_labels
        threshold_source = "validation"
        validation_feature_summary = summarize_feature_columns(
            feature_columns=validation_feature_columns,
            labels=validation_labels,
        )
        validation_score_summary = summarize_feature_columns(
            feature_columns=validation_derived_columns,
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

    summary = {
        "trainer_type": "anchor_distance_threshold",
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
        "score_formula": score_formula(
            query_field=config.query_field,
            response_field=config.response_field,
            neutral_field=config.neutral_field,
            aux_neutral_field=config.aux_neutral_field,
        ),
        "train_feature_summary": summarize_feature_columns(
            feature_columns=train_feature_columns,
            labels=train_labels,
        ),
        "train_score_summary": summarize_feature_columns(
            feature_columns=train_derived_columns,
            labels=train_labels,
        ),
        "validation_feature_summary": validation_feature_summary,
        "validation_score_summary": validation_score_summary,
        "train_summary": train_summary,
        "threshold_summary": threshold_summary,
        "validation_summary": validation_summary,
    }

    emit_progress(f"writing anchor-distance threshold state to {config.output_dir}")
    state_path = embedding_state_path(config.output_dir)
    state_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_path = config.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
