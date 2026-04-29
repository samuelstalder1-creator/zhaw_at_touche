from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

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
DEFAULT_LR_C_VALUES = (1.0,)
DEFAULT_LR_CLASS_WEIGHT_OPTIONS = ("balanced",)

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
class EmbeddingLRFeatureConfig:
    delta_centering: str = "none"
    append_delta_abs: bool = False
    append_pairwise_cosine: bool = False
    append_delta_norm: bool = False


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
    delta_centering: str = "none"
    append_delta_abs: bool = False
    append_pairwise_cosine: bool = False
    append_delta_norm: bool = False
    lr_c_values: tuple[float, ...] = field(default_factory=lambda: DEFAULT_LR_C_VALUES)
    lr_class_weight_options: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_LR_CLASS_WEIGHT_OPTIONS
    )


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


def feature_config_from_state(state: Mapping[str, Any] | None) -> EmbeddingLRFeatureConfig:
    if not state:
        return EmbeddingLRFeatureConfig()
    return EmbeddingLRFeatureConfig(
        delta_centering=str(state.get("delta_centering", "none")),
        append_delta_abs=bool(state.get("append_delta_abs", False)),
        append_pairwise_cosine=bool(state.get("append_pairwise_cosine", False)),
        append_delta_norm=bool(state.get("append_delta_norm", False)),
    )


def delta_centers_from_state(state: Mapping[str, Any] | None) -> dict[str, np.ndarray]:
    if not state:
        return {}
    raw_centers = state.get("delta_center_vectors")
    if raw_centers is None:
        return {}
    if not isinstance(raw_centers, dict):
        raise ValueError("delta_center_vectors in saved state must be a JSON object.")
    centers: dict[str, np.ndarray] = {}
    for name, values in raw_centers.items():
        if not isinstance(name, str):
            raise ValueError("delta_center_vectors keys must be strings.")
        centers[name] = np.asarray(values, dtype=np.float32)
    return centers


def _serialize_delta_centers(delta_centers: Mapping[str, np.ndarray]) -> dict[str, list[float]]:
    return {
        name: [float(value) for value in center.tolist()]
        for name, center in delta_centers.items()
    }


def _metric_value_from_summary(summary: Mapping[str, Any], threshold_metric: str) -> float:
    if threshold_metric == "accuracy":
        return float(summary["accuracy"])
    if threshold_metric == "positive_f1":
        positive = summary.get("positive_label")
        if not isinstance(positive, dict):
            raise ValueError("positive_label summary missing while extracting positive_f1.")
        return float(positive["f1"])
    if threshold_metric == "macro_f1":
        macro = summary.get("macro")
        if not isinstance(macro, dict):
            raise ValueError("macro summary missing while extracting macro_f1.")
        return float(macro["f1"])
    raise ValueError(f"Unsupported threshold metric '{threshold_metric}'.")


def _normalize_lr_class_weight_option(raw_option: str) -> str | None:
    if raw_option == "balanced":
        return "balanced"
    if raw_option == "none":
        return None
    raise ValueError(f"Unsupported lr_class_weight option '{raw_option}'.")


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


def _rowwise_cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerators = np.sum(left * right, axis=1)
    denominators = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    safe_denominators = np.clip(denominators, a_min=1e-12, a_max=None)
    return (numerators / safe_denominators).astype(np.float32)


def _resolve_delta_centers(
    *,
    feature_config: EmbeddingLRFeatureConfig,
    delta_vectors: Mapping[str, np.ndarray],
    labels_for_centering: Sequence[int] | None,
    provided_delta_centers: Mapping[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    if feature_config.delta_centering == "none":
        return {}
    if feature_config.delta_centering != "negative_mean":
        raise ValueError(f"Unsupported delta centering strategy '{feature_config.delta_centering}'.")

    if provided_delta_centers:
        centers = {
            name: np.asarray(vector, dtype=np.float32)
            for name, vector in provided_delta_centers.items()
            if name in delta_vectors
        }
        missing = sorted(set(delta_vectors) - set(centers))
        if missing:
            raise ValueError(
                "Missing delta center vectors for: " + ", ".join(missing)
            )
        return centers

    if labels_for_centering is None:
        raise ValueError(
            "labels_for_centering are required when fitting negative-mean delta centers."
        )

    label_array = np.asarray(labels_for_centering, dtype=int)
    negative_mask = label_array == 0
    if not np.any(negative_mask):
        raise ValueError("negative_mean delta centering requires at least one negative-label row.")

    return {
        name: delta[negative_mask].mean(axis=0).astype(np.float32)
        for name, delta in delta_vectors.items()
    }


def _build_feature_matrix(
    trainer_type: str,
    embeddings: dict[str, torch.Tensor],
    response_field: str,
    neutral_field: str,
    aux_neutral_field: str | None,
    query_field: str,
    feature_config: EmbeddingLRFeatureConfig,
    delta_center_vectors: Mapping[str, np.ndarray] | None = None,
    labels_for_centering: Sequence[int] | None = None,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    R = embeddings[response_field].cpu().numpy().astype(np.float32, copy=False)
    N = embeddings[neutral_field].cpu().numpy().astype(np.float32, copy=False)

    delta_name = f"delta_{response_field}_{neutral_field}"
    raw_deltas: dict[str, np.ndarray] = {
        delta_name: (R - N).astype(np.float32, copy=False),
    }
    delta_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {
        delta_name: (R, N),
    }

    Q: np.ndarray | None = None
    if trainer_type in {"query_residual_classifier", "query_dual_residual_classifier"}:
        Q = embeddings[query_field].cpu().numpy().astype(np.float32, copy=False)

    N2: np.ndarray | None = None
    delta_name_2: str | None = None
    if trainer_type in DUAL_FILE_TRAINERS:
        if aux_neutral_field is None:
            raise ValueError(f"{trainer_type} requires aux_neutral_field.")
        N2 = embeddings[aux_neutral_field].cpu().numpy().astype(np.float32, copy=False)
        delta_name_2 = f"delta_{response_field}_{aux_neutral_field}"
        raw_deltas[delta_name_2] = (R - N2).astype(np.float32, copy=False)
        delta_pairs[delta_name_2] = (R, N2)

    resolved_delta_centers = _resolve_delta_centers(
        feature_config=feature_config,
        delta_vectors=raw_deltas,
        labels_for_centering=labels_for_centering,
        provided_delta_centers=delta_center_vectors,
    )
    centered_deltas = {
        name: delta - resolved_delta_centers.get(name, 0.0)
        for name, delta in raw_deltas.items()
    }

    feature_names: list[str] = []
    feature_blocks: list[np.ndarray] = []

    def add_block(name: str, values: np.ndarray) -> None:
        feature_names.append(name)
        feature_blocks.append(values.astype(np.float32, copy=False))

    def add_delta_block(name: str) -> None:
        centered_delta = centered_deltas[name]
        left_embeddings, right_embeddings = delta_pairs[name]
        add_block(name, centered_delta)
        if feature_config.append_delta_abs:
            add_block(f"abs_{name}", np.abs(centered_delta))
        if feature_config.append_pairwise_cosine:
            cosine_values = _rowwise_cosine_similarity(left_embeddings, right_embeddings).reshape(-1, 1)
            add_block(f"cosine_{name}", cosine_values)
        if feature_config.append_delta_norm:
            norm_values = np.linalg.norm(centered_delta, axis=1, keepdims=True).astype(np.float32)
            add_block(f"norm_{name}", norm_values)

    if trainer_type == "embedding_residual_classifier":
        add_delta_block(delta_name)
    elif trainer_type == "embedding_classifier":
        add_block(response_field, R)
        add_block(neutral_field, N)
        add_delta_block(delta_name)
    elif trainer_type == "query_residual_classifier":
        if Q is None:
            raise ValueError("query_residual_classifier requires query embeddings.")
        add_block(query_field, Q)
        add_delta_block(delta_name)
    elif trainer_type == "dual_residual_classifier":
        if delta_name_2 is None:
            raise ValueError("dual_residual_classifier requires auxiliary residual features.")
        add_delta_block(delta_name)
        add_delta_block(delta_name_2)
    elif trainer_type == "dual_embedding_classifier":
        if N2 is None or delta_name_2 is None or aux_neutral_field is None:
            raise ValueError("dual_embedding_classifier requires auxiliary neutral features.")
        add_block(response_field, R)
        add_block(neutral_field, N)
        add_block(aux_neutral_field, N2)
        add_delta_block(delta_name)
        add_delta_block(delta_name_2)
    elif trainer_type == "query_dual_residual_classifier":
        if Q is None or delta_name_2 is None:
            raise ValueError("query_dual_residual_classifier requires query and auxiliary residual features.")
        add_block(query_field, Q)
        add_delta_block(delta_name)
        add_delta_block(delta_name_2)
    else:
        raise ValueError(f"Unknown trainer_type: {trainer_type}")

    if not feature_blocks:
        raise ValueError(f"No feature blocks were built for trainer_type '{trainer_type}'.")

    return feature_names, np.concatenate(feature_blocks, axis=1), resolved_delta_centers


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
    feature_config: EmbeddingLRFeatureConfig,
    delta_center_vectors: Mapping[str, np.ndarray] | None = None,
    labels_for_centering: Sequence[int] | None = None,
    progress_prefix: str | None = None,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
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
        feature_config=feature_config,
        delta_center_vectors=delta_center_vectors,
        labels_for_centering=labels_for_centering,
    )


def train_embedding_lr_classifier(config: EmbeddingLRConfig) -> dict[str, Any]:
    if config.trainer_type not in ALL_TRAINER_TYPES:
        raise ValueError(
            f"Unknown trainer_type: {config.trainer_type}. Expected one of: {sorted(ALL_TRAINER_TYPES)}"
        )
    if not config.lr_c_values:
        raise ValueError("lr_c_values must contain at least one candidate.")
    if not config.lr_class_weight_options:
        raise ValueError("lr_class_weight_options must contain at least one candidate.")

    feature_config = EmbeddingLRFeatureConfig(
        delta_centering=config.delta_centering,
        append_delta_abs=config.append_delta_abs,
        append_pairwise_cosine=config.append_pairwise_cosine,
        append_delta_norm=config.append_delta_norm,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(f"loading embedding model {config.embedding_model_name}")
    tokenizer, model = load_embedding_model(config.embedding_model_name, config.device)

    emit_progress(f"loading training rows from {config.train_path}")
    train_records = _load_records(config.trainer_type, config.train_path, config.aux_train_path)
    train_labels = [int(record["label"]) for record in train_records]
    emit_progress(f"embedding {len(train_records)} training rows ({config.trainer_type})")
    feature_names, train_features, delta_centers = _embed_and_build(
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
        feature_config=feature_config,
        labels_for_centering=train_labels,
        progress_prefix=f"{config.trainer_type} train",
    )

    threshold_source = "train"
    val_labels: list[int] | None = None
    val_scores: list[float] | None = None
    val_features: np.ndarray | None = None
    if config.validation_path is not None:
        emit_progress(f"loading calibration rows from {config.validation_path}")
        val_records = _load_records(config.trainer_type, config.validation_path, config.aux_validation_path)
        emit_progress(f"embedding {len(val_records)} validation rows for threshold calibration")
        _, val_features, _ = _embed_and_build(
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
            feature_config=feature_config,
            delta_center_vectors=delta_centers,
            progress_prefix=f"{config.trainer_type} validation",
        )
        val_labels = [int(record["label"]) for record in val_records]
        threshold_source = "validation"

    candidate_results: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None
    best_metric_value = float("-inf")
    best_accuracy = float("-inf")

    for class_weight_option in config.lr_class_weight_options:
        resolved_class_weight = _normalize_lr_class_weight_option(class_weight_option)
        for c_value in config.lr_c_values:
            emit_progress(
                "fitting logistic regression "
                f"(C={c_value:g}, class_weight={class_weight_option})"
            )
            classifier = Pipeline([
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight=resolved_class_weight,
                        C=float(c_value),
                        max_iter=1000,
                        solver="lbfgs",
                    ),
                ),
            ])
            classifier.fit(train_features, train_labels)
            current_train_scores = [
                float(score) for score in classifier.predict_proba(train_features)[:, 1]
            ]

            if val_features is not None and val_labels is not None:
                current_threshold_scores = [
                    float(score) for score in classifier.predict_proba(val_features)[:, 1]
                ]
                current_threshold_labels = val_labels
                current_threshold_rows = len(val_labels)
            else:
                current_threshold_scores = current_train_scores
                current_threshold_labels = train_labels
                current_threshold_rows = len(train_labels)

            threshold, threshold_summary = calibrate_threshold(
                current_threshold_scores,
                current_threshold_labels,
                threshold_metric=config.threshold_metric,
            )
            metric_value = _metric_value_from_summary(threshold_summary, config.threshold_metric)
            accuracy_value = float(threshold_summary["accuracy"])
            candidate_summary = {
                "c": float(c_value),
                "class_weight": class_weight_option,
                "threshold": float(threshold),
                "selection_metric": metric_value,
                "selection_accuracy": accuracy_value,
                "threshold_rows": current_threshold_rows,
            }
            candidate_results.append(candidate_summary)

            if metric_value > best_metric_value or (
                metric_value == best_metric_value and accuracy_value > best_accuracy
            ):
                best_metric_value = metric_value
                best_accuracy = accuracy_value
                best_candidate = {
                    "classifier": classifier,
                    "train_scores": current_train_scores,
                    "threshold_scores": current_threshold_scores,
                    "threshold_labels": current_threshold_labels,
                    "threshold": threshold,
                    "threshold_summary": threshold_summary,
                    "class_weight": class_weight_option,
                    "c": float(c_value),
                }

    if best_candidate is None:
        raise ValueError("Could not fit any logistic-regression candidate.")

    classifier = best_candidate["classifier"]
    threshold = float(best_candidate["threshold"])
    threshold_summary = best_candidate["threshold_summary"]
    train_scores = best_candidate["train_scores"]
    threshold_scores = best_candidate["threshold_scores"]
    threshold_labels = best_candidate["threshold_labels"]

    emit_progress(
        "selected logistic regression "
        f"(C={best_candidate['c']:g}, class_weight={best_candidate['class_weight']})"
    )
    emit_progress(f"selected threshold={threshold:.6f}")

    train_predictions = [1 if score >= threshold else 0 for score in train_scores]
    train_summary = metrics_dict(train_labels, train_predictions)

    validation_summary = None
    if val_labels is not None:
        val_predictions = [1 if score >= threshold else 0 for score in threshold_scores]
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
        "threshold_rows": len(threshold_labels),
        "threshold_summary": threshold_summary,
        "train_summary": train_summary,
        "validation_summary": validation_summary,
        "delta_centering": feature_config.delta_centering,
        "append_delta_abs": feature_config.append_delta_abs,
        "append_pairwise_cosine": feature_config.append_pairwise_cosine,
        "append_delta_norm": feature_config.append_delta_norm,
        "delta_center_vectors": _serialize_delta_centers(delta_centers),
        "lr_c_values": [float(value) for value in config.lr_c_values],
        "lr_class_weight_options": list(config.lr_class_weight_options),
        "selected_lr_candidate": {
            "c": best_candidate["c"],
            "class_weight": best_candidate["class_weight"],
            "selection_metric": best_metric_value,
            "selection_accuracy": best_accuracy,
        },
        "lr_candidate_results": candidate_results,
    }

    emit_progress(f"writing {config.trainer_type} state to {config.output_dir}")
    with bundle_path(config.output_dir).open("wb") as handle:
        pickle.dump(classifier, handle)
    state_path = embedding_state_path(config.output_dir)
    state_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (config.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
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
    saved_state: Mapping[str, Any] | None = None,
    progress_prefix: str | None = None,
) -> list[EmbeddingLRPrediction]:
    feature_config = feature_config_from_state(saved_state)
    delta_center_vectors = delta_centers_from_state(saved_state)
    _, feature_matrix, _ = _embed_and_build(
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
        feature_config=feature_config,
        delta_center_vectors=delta_center_vectors,
        progress_prefix=progress_prefix,
    )
    scores = [float(score) for score in classifier.predict_proba(feature_matrix)[:, 1]]
    return [
        EmbeddingLRPrediction(label=1 if score >= threshold else 0, score=score)
        for score in scores
    ]
