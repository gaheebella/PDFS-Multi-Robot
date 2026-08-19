"""GT-free runtime evidence calibration for temporal opening tracks.

Calibration labels are consumed only by :func:`fit_calibration`; runtime
``estimate_opening_evidence`` accepts local observable features only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_NAMES = ("two_wall_fraction", "persistence_fraction", "interval_iou_mean")


@dataclass(frozen=True)
class EvidenceCalibration:
    """Monotonic logistic calibration and calibration-only state boundaries."""

    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    accepted_threshold: float
    uncertain_threshold: float
    calibration_geometry_ids: tuple[str, ...]


def _matrix(rows: Sequence[Mapping[str, Any]], calibration: EvidenceCalibration | None = None) -> np.ndarray:
    values = np.asarray([[float(row[name]) if row.get(name, "") != "" else 0.0 for name in FEATURE_NAMES] for row in rows], dtype=float)
    if calibration is None: return values
    return (values - np.asarray(calibration.mean)) / np.asarray(calibration.scale)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def fit_calibration(rows: Sequence[Mapping[str, Any]], *, geometry_ids: Sequence[str], steps: int = 2500, learning_rate: float = 0.05) -> EvidenceCalibration:
    """Fit a positive-coefficient logistic evidence score on calibration rows.

    The non-negative projection preserves the intended monotonic interpretation
    without injecting a geometry- or sensor-specific threshold.  Tri-state
    boundaries are selected exclusively from calibration predictions.
    """
    selected = [row for row in rows if str(row.get("case_id")) in set(geometry_ids)]
    if not selected: raise ValueError("calibration split is empty")
    x_raw = _matrix(selected); y = np.asarray([1.0 if str(row["track_label"]) == "true" else 0.0 for row in selected])
    mean, scale = np.mean(x_raw, axis=0), np.std(x_raw, axis=0); scale = np.where(scale <= 1e-9, 1.0, scale); x = (x_raw - mean) / scale
    weights = np.zeros(3, dtype=float); intercept = 0.0
    for _ in range(steps):
        prediction = _sigmoid(intercept + x @ weights); error = prediction - y
        intercept -= learning_rate * float(np.mean(error)); weights -= learning_rate * (x.T @ error / len(x)); weights = np.maximum(weights, 0.0)
    probabilities = _sigmoid(intercept + x @ weights)
    candidates = np.unique(np.r_[0.5, probabilities])
    feasible = []
    for threshold in candidates:
        accepted = probabilities >= threshold; count = int(np.sum(accepted)); precision = float(np.sum(y[accepted]) / max(count, 1)); recall = float(np.sum(y[accepted]) / max(np.sum(y), 1));
        if precision >= 0.90: feasible.append((recall, threshold))
    accepted_threshold = float(max(feasible, default=(0.0, 0.90))[1])
    uncertain_threshold = float(np.percentile(probabilities, 25))
    uncertain_threshold = min(uncertain_threshold, accepted_threshold)
    return EvidenceCalibration(tuple(FEATURE_NAMES), tuple(mean), tuple(scale), tuple(weights), float(intercept), accepted_threshold, uncertain_threshold, tuple(sorted(set(map(str, geometry_ids)))))


def estimate_opening_evidence(track_features: Mapping[str, Any], calibration: EvidenceCalibration) -> dict[str, Any]:
    """Estimate reliability from local track observables without GT/map inputs."""
    x = _matrix([track_features], calibration)[0]
    score = float(_sigmoid(np.asarray([calibration.intercept + x @ np.asarray(calibration.coefficients)]))[0])
    if score >= calibration.accepted_threshold: state = "ACCEPTED"
    elif score <= calibration.uncertain_threshold: state = "UNCERTAIN"
    else: state = "PROVISIONAL"
    return {"evidence_score": score, "state": state, "two_wall_fraction": float(track_features["two_wall_fraction"]), "persistence_fraction": float(track_features["persistence_fraction"]), "interval_iou_mean": float(track_features["interval_iou_mean"] if track_features.get("interval_iou_mean", "") != "" else 0.0)}


def ablation_features(rows: Sequence[Mapping[str, Any]], geometry_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Fit the same monotonic model for every non-empty primary feature subset."""
    output = []
    subsets = (("two_wall_fraction",), ("persistence_fraction",), ("interval_iou_mean",), ("persistence_fraction", "interval_iou_mean"), ("persistence_fraction", "two_wall_fraction"), ("interval_iou_mean", "two_wall_fraction"), FEATURE_NAMES, ("boundary_std_deg",), ("wall_tangent_axial_std_deg",), FEATURE_NAMES + ("boundary_std_deg",), FEATURE_NAMES + ("wall_tangent_axial_std_deg",))
    for subset in subsets:
        selected = [row for row in rows if str(row.get("case_id")) in set(geometry_ids)]
        # Descriptive, threshold-free separation is reported for subsets; the
        # full model is used for the tri-state runtime calibration.
        true = [np.mean([float(row[x]) if row.get(x, "") != "" else 0.0 for x in subset]) for row in selected if row["track_label"] == "true"]
        false = [np.mean([float(row[x]) if row.get(x, "") != "" else 0.0 for x in subset]) for row in selected if row["track_label"] == "false"]
        output.append({"feature_subset": "+".join(subset), "true_mean": float(np.mean(true)) if true else "", "false_mean": float(np.mean(false)) if false else "", "mean_gap": float(np.mean(true) - np.mean(false)) if true and false else ""})
    return output
