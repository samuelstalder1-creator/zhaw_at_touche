from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .anchor_distance_classifier import embed_record_fields, load_merged_records
from .embedding_divergence import (
    calibrate_threshold,
    emit_progress,
    embedding_state_path,
    load_embedding_model,
)
from .evaluation_utils import metrics_dict
from .jsonl import read_jsonl

BUNDLE_FILENAME = "embedding_lr_classifier.pkl"

SINGLE_FILE_TRAINERS = frozenset({
    "embedding_residual_classifier",
    "embedding_classifier",
    "query_residual_classifier",
})

DUAL_FILE_TRAINERS = frozenset({
    "dual_residual_classifier",
    "dual_embedding_classifier",
    "query_dual_residual_classifier",
})

ALL_TRAINER_TYPES = SINGLE_FILE_TRAINERS | DUAL_FILE_TRAINERS


@dataclass(frozen=True)
class EmbeddingLRPrediction:
    label: int
    score: float


@dataclass(frozen=True)
class EmbeddingLRConfig:
    trainer_type: str
    embedding_model_name: str
    train_path: Path
    output_dir: Path
    max_length: int
    batch_size: int
    device: str
    response_field: str
    neutral_field: str
    threshold_metric: str
    query_field: str = "query"
    aux_neutral_field: str | None = None
    aux_train_path: Path | None = None
    validation_path: Path | None = None
    aux_validation_path: Path | None = None


def bundle_path(model_dir: Path) -> Path:
    return model_dir / BUNDLE_FILENAME


def load_bundle(model_dir: Path) -> Any:
    p = bundle_path(model_dir)
    if not p.exists():
        raise FileNotFoundError(f"Embedding LR classifier bundle not found: {p}")
    with p.open("rb") as handle:
        return pickle.load(handle)


def load_state(model_dir: Path) -> dict[str, Any] | None:
    state_path = embedding_state_path(model_dir)
    if not state_path.exists():
        return None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Embedding LR state must be a JSON object: {state_path}")
    return payload


def _required_fields(
    trainer_type: str,
    response_field: str,
    neutral_field: str,
    aux_neutral_field: str | None,
    query_field: str,
) -> list[str]:
    fields = [response_field, neutral_field]
    if trainer_type in DUAL_FILE_TRAINERS:
        if not aux_neutral_field:
            raise ValueError(f"{trainer_type} requires aux_neutral_field.")
        fields.append(aux_neutral_field)
    if trainer_type in {"query_residual_classifier", "query_dual_residual_classifier"}:
        fields.append(query_field)
    return fields


def _build_feature_matrix(
    trainer_type: str,
    embeddings: dict[str, torch.Tensor],
    response_field: str,
    neutral_field: str,
    aux_neutral_field: str | None,
    query_field: str,
) -> tuple[list[str], np.ndarray]:
    R = embeddings[response_field].cpu().numpy()
    N = embeddings[neutral_field].cpu().numpy()
    delta = R - N

    if trainer_type == "embedding_residual_classifier":
        return [f"delta_{response_field}_{neutral_field}"], delta

    if trainer_type == "embedding_classifier":
        names = [response_field, neutral_field, f"delta_{response_field}_{neutral_field}"]
        return names, np.concatenate([R, N, delta], axis=1)

    if trainer_type == "query_residual_classifier":
        Q = embeddings[query_field].cpu().numpy()
        names = [query_field, f"delta_{response_field}_{neutral_field}"]
        return names, np.concatenate([Q, delta], axis=1)

    if aux_neutral_field is None:
        raise ValueError(f"{trainer_type} requires aux_neutral_field.")
    N2 = embeddings[aux_neutral_field].cpu().numpy()
    delta2 = R - N2

    if trainer_type == "dual_residual_classifier":
        names = [
            f"delta_{response_field}_{neutral_field}",
            f"delta_{response_field}_{aux_neutral_field}",
        ]
        return names, np.concatenate([delta, delta2], axis=1)

    if trainer_type == "dual_embedding_classifier":
        names = [
            response_field, neutral_field, aux_neutral_field,
            f"delta_{response_field}_{neutral_field}",
            f"delta_{response_field}_{aux_neutral_field}",
        ]
        return names, np.concatenate([R, N, N2, delta, delta2], axis=1)

    if trainer_type == "query_dual_residual_classifier":
        Q = embeddings[query_field].cpu().numpy()
        names = [
            query_field,
            f"delta_{response_field}_{neutral_field}",
            f"delta_{response_field}_{aux_neutral_field}",
        ]
        return names, np.concatenate([Q, delta, delta2], axis=1)

    raise ValueError(f"Unknown trainer_type: {trainer_type}")


def _load_records(
    trainer_type: str,
    primary_path: Path,
    aux_path: Path | None,
) -> list[dict[str, Any]]:
    if trainer_type in DUAL_FILE_TRAINERS:
        if not aux_path:
            raise ValueError(f"{trainer_type} requires aux_train_path.")
        return load_merged_records(primary_path, aux_path)
    return list(read_jsonl(primary_path))


def _embed_and_build(
    *,
    trainer_type: str,
    tokenizer,
    model,
    records: Sequence[dict[str, Any]],
    response_field: str,
    neutral_field: str,
    aux_neutral_field: str | None,
    query_field: str,
    device: str,
    batch_size: int,
    max_length: int,
    progress_prefix: str | None = None,
) -> tuple[list[str], np.ndarray]:
    fields = _required_fields(trainer_type, response_field, neutral_field, aux_neutral_field, query_field)
    embeddings = embed_record_fields(
        tokenizer=tokenizer,
        model=model,
        records=records,
        fields=fields,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        progress_prefix=progress_prefix,
    )
    return _build_feature_matrix(
        trainer_type=trainer_type,
        embeddings=embeddings,
        response_field=response_field,
        neutral_field=neutral_field,
        aux_neutral_field=aux_neutral_field,
        query_field=query_field,
    )


def train_embedding_lr_classifier(config: EmbeddingLRConfig) -> dict[str, Any]:
    if config.trainer_type not in ALL_TRAINER_TYPES:
        raise ValueError(f"Unknown trainer_type: {config.trainer_type}. Expected one of: {sorted(ALL_TRAINER_TYPES)}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(f"loading embedding model {config.embedding_model_name}")
    tokenizer, model = load_embedding_model(config.embedding_model_name, config.device)

    emit_progress(f"loading training rows from {config.train_path}")
    train_records = _load_records(config.trainer_type, config.train_path, config.aux_train_path)
    emit_progress(f"embedding {len(train_records)} training rows ({config.trainer_type})")
    feature_names, train_features = _embed_and_build(
        trainer_type=config.trainer_type,
        tokenizer=tokenizer,
        model=model,
        records=train_records,
        response_field=config.response_field,
        neutral_field=config.neutral_field,
        aux_neutral_field=config.aux_neutral_field,
        query_field=config.query_field,
        device=config.device,
        batch_size=config.batch_size,
        max_length=config.max_length,
        progress_prefix=f"{config.trainer_type} train",
    )
    train_labels = [int(record["label"]) for record in train_records]

    classifier = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")),
    ])
    emit_progress("fitting logistic regression")
    classifier.fit(train_features, train_labels)
    train_scores = [float(s) for s in classifier.predict_proba(train_features)[:, 1]]

    threshold_records = train_records
    threshold_scores = train_scores
    threshold_labels = train_labels
    threshold_source = "train"
    validation_summary = None

    if config.validation_path is not None:
        emit_progress(f"loading calibration rows from {config.validation_path}")
        val_records = _load_records(config.trainer_type, config.validation_path, config.aux_validation_path)
        emit_progress(f"embedding {len(val_records)} validation rows for threshold calibration")
        _, val_features = _embed_and_build(
            trainer_type=config.trainer_type,
            tokenizer=tokenizer,
            model=model,
            records=val_records,
            response_field=config.response_field,
            neutral_field=config.neutral_field,
            aux_neutral_field=config.aux_neutral_field,
            query_field=config.query_field,
            device=config.device,
            batch_size=config.batch_size,
            max_length=config.max_length,
            progress_prefix=f"{config.trainer_type} validation",
        )
        val_labels = [int(record["label"]) for record in val_records]
        val_scores = [float(s) for s in classifier.predict_proba(val_features)[:, 1]]
        threshold_records = val_records
        threshold_scores = val_scores
        threshold_labels = val_labels
        threshold_source = "validation"

    emit_progress(f"fitting threshold on {len(threshold_records)} {threshold_source} rows")
    threshold, threshold_summary = calibrate_threshold(
        threshold_scores, threshold_labels, threshold_metric=config.threshold_metric
    )
    emit_progress(f"selected threshold={threshold:.6f}")

    train_predictions = [1 if s >= threshold else 0 for s in train_scores]
    train_summary = metrics_dict(train_labels, train_predictions)
    if config.validation_path is not None:
        val_predictions = [1 if s >= threshold else 0 for s in threshold_scores]
        validation_summary = metrics_dict(threshold_labels, val_predictions)

    lr_step = classifier.named_steps["classifier"]
    feature_dim = train_features.shape[1]
    summary = {
        "trainer_type": config.trainer_type,
        "embedding_model_name": config.embedding_model_name,
        "train_path": str(config.train_path),
        "aux_train_path": str(config.aux_train_path) if config.aux_train_path else None,
        "validation_path": str(config.validation_path) if config.validation_path else None,
        "output_dir": str(config.output_dir),
        "device": config.device,
        "response_field": config.response_field,
        "neutral_field": config.neutral_field,
        "aux_neutral_field": config.aux_neutral_field,
        "query_field": config.query_field,
        "threshold_metric": config.threshold_metric,
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "feature_names": feature_names,
        "feature_dim": feature_dim,
        "threshold_source": threshold_source,
        "threshold": threshold,
        "intercept": float(lr_step.intercept_[0]),
        "train_rows": len(train_records),
        "threshold_rows": len(threshold_records),
        "threshold_summary": threshold_summary,
        "train_summary": train_summary,
        "validation_summary": validation_summary,
    }

    emit_progress(f"writing {config.trainer_type} state to {config.output_dir}")
    with bundle_path(config.output_dir).open("wb") as handle:
        pickle.dump(classifier, handle)
    state_path = embedding_state_path(config.output_dir)
    state_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (config.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def score_records(
    *,
    classifier,
    tokenizer,
    model,
    records: Sequence[dict[str, Any]],
    trainer_type: str,
    response_field: str,
    neutral_field: str,
    aux_neutral_field: str | None,
    query_field: str,
    threshold: float,
    device: str,
    batch_size: int,
    max_length: int,
    progress_prefix: str | None = None,
) -> list[EmbeddingLRPrediction]:
    _, feature_matrix = _embed_and_build(
        trainer_type=trainer_type,
        tokenizer=tokenizer,
        model=model,
        records=records,
        response_field=response_field,
        neutral_field=neutral_field,
        aux_neutral_field=aux_neutral_field,
        query_field=query_field,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        progress_prefix=progress_prefix,
    )
    scores = [float(s) for s in classifier.predict_proba(feature_matrix)[:, 1]]
    return [
        EmbeddingLRPrediction(label=1 if score >= threshold else 0, score=score)
        for score in scores
    ]
