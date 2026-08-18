"""General fusion of wall-geometry and Stable-motion Branch tangents.

This evaluation-only module reuses existing physical trajectories, sensor
models, wall estimation, and wall reliability.  It adds a minimal grouped
calibration for motion uncertainty and a GT-free circular fusion API.

The two predicted P90 errors are treated as comparable angular *scales*, not
as guaranteed Gaussian confidence intervals.  Their inverse squares are used
only when their circular uncertainty intervals are mutually consistent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from junction_detection.integration.branch_orientation_geometry_motion_comparison import (
    EVALUATION_CORRECT_DEG,
    _representative_cases,
    geometry_wall_segments,
)
from junction_detection.integration.pointcloud_wall_orientation_reliability import (
    _fit_linear_quantile,
    _group_folds,
    _spearman,
    estimate_wall_reliability,
)
from junction_detection.integration.pointcloud_wall_orientation_sensor_robustness import (
    SENSOR_CONDITIONS,
    _candidate_diagnostics,
)
from junction_detection.integration.pointcloud_wall_parallel_orientation import (
    WallEstimate,
    estimate_wall_parallel_tangent,
)
from junction_detection.integration.trajectory_stability_diagnostics import (
    angular_error_deg,
    normalize_angle_deg,
)
from junction_detection.pointcloud.pointcloud_junction_detector import detect_openings
from junction_detection.pointcloud.pointcloud_junction_detector_local_topology import (
    _match_openings,
    ground_truth_openings_from_geometry,
)
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import (
    simulate_lidar_scan,
)


MOTION_QUANTILE = 0.90
MOTION_MODEL_VERSION = "stable_motion_linear_quantile_v1"
EVALUATION_SENSOR_IDS = (
    "clean",
    "noise_0.08",
    "occlusion_0.80",
    "visibility_0.70",
    "resolution_4deg",
)
FORBIDDEN_RUNTIME_TERMS = (
    "gt", "actual_error", "case_id", "turn", "width", "length",
    "anchor", "branch_angle", "sensor", "noise", "occlusion", "visibility",
)


@dataclass(frozen=True)
class MotionReliability:
    """Runtime Stable-motion availability and predicted P90 angular error."""

    predicted_p90_error_deg: float | None
    available: bool
    model_version: str
    reason: str


@dataclass(frozen=True)
class BranchOrientationEvidence:
    """One runtime orientation estimate with a coarse angular uncertainty."""

    tangent_deg: float | None
    uncertainty_deg: float | None
    available: bool
    source: str


@dataclass(frozen=True)
class FusedBranchOrientation:
    """Traceable final Branch orientation result."""

    tangent_deg: float | None
    uncertainty_deg: float | None
    status: str
    geometry_used: bool
    motion_used: bool
    disagreement_deg: float | None


def _short_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    """Convert NumPy scalar outputs to JSON-native Python scalars."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical_json(value: Any) -> str:
    """Serialize evaluation outputs deterministically for replay checks."""
    return json.dumps(value, sort_keys=True, default=_json_default)


def _motion_training_rows(
    comparison_csv: Path, stability_csv: Path
) -> list[dict[str, Any]]:
    """Join existing Stable-motion outputs with runtime stability observables."""
    comparisons = _read_rows(comparison_csv)
    stable_phase = {
        (row["case_id"], row["branch_id"]): row
        for row in _read_rows(stability_csv)
        if row["phase"] == "STABLE_GATE"
    }
    rows = []
    for comparison in comparisons:
        key = (comparison["case_id"], comparison["branch_id"])
        if comparison["estimator_available"] != "True" or key not in stable_phase:
            continue
        phase = stable_phase[key]
        rows.append({
            "case_id": comparison["case_id"],
            "branch_id": comparison["branch_id"],
            "stable_sample_count": float(comparison["stable_segment_count"]),
            "stable_robot_count": float(comparison["stable_robot_count"]),
            "stable_dispersion_deg": float(phase["circular_dispersion_deg"]),
            "stable_resultant": float(phase["resultant_length"]),
            "actual_motion_error_deg": float(comparison["stable_error_deg"]),
        })
    if not rows:
        raise ValueError("no existing motion reliability rows could be joined")
    return rows


def _motion_raw_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([
        [
            math.log1p(float(row["stable_sample_count"])),
            math.log1p(float(row["stable_robot_count"])),
            float(row["stable_dispersion_deg"]),
            float(row["stable_resultant"]),
        ]
        for row in rows
    ])


def _fit_motion_calibration(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw = _motion_raw_matrix(rows)
    mean = np.mean(raw, axis=0)
    scale = np.std(raw, axis=0)
    scale = np.where(scale <= np.finfo(float).eps, 1.0, scale)
    matrix = np.column_stack((np.ones(len(raw)), (raw - mean) / scale))
    target = np.asarray([float(row["actual_motion_error_deg"]) for row in rows])
    coefficients = _fit_linear_quantile(
        matrix, target, quantile=MOTION_QUANTILE
    )
    return {
        "model_version": MOTION_MODEL_VERSION,
        "quantile": MOTION_QUANTILE,
        "feature_names": [
            "intercept", "z(log1p(stable_sample_count))",
            "z(log1p(stable_robot_count))", "z(stable_dispersion_deg)",
            "z(stable_resultant)",
        ],
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "training_row_count": len(rows),
        "training_case_count": len({row["case_id"] for row in rows}),
    }


def estimate_motion_reliability(
    stable_sample_count: float,
    stable_robot_count: float,
    stable_dispersion_deg: float | None,
    stable_resultant: float | None,
    calibration: Mapping[str, Any],
) -> MotionReliability:
    """Predict Stable-motion P90 error using runtime-only observables."""
    values = (stable_sample_count, stable_robot_count, stable_dispersion_deg, stable_resultant)
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return MotionReliability(None, False, str(calibration.get("model_version", MOTION_MODEL_VERSION)), "motion_observables_unavailable")
    if stable_sample_count <= 0.0 or stable_robot_count <= 0.0:
        return MotionReliability(None, False, str(calibration.get("model_version", MOTION_MODEL_VERSION)), "motion_samples_unavailable")
    raw = np.asarray([
        math.log1p(float(stable_sample_count)),
        math.log1p(float(stable_robot_count)),
        float(stable_dispersion_deg),
        float(stable_resultant),
    ])
    mean = np.asarray(calibration["mean"], dtype=float)
    scale = np.asarray(calibration["scale"], dtype=float)
    vector = np.r_[1.0, (raw - mean) / scale]
    prediction = float(vector @ np.asarray(calibration["coefficients"], dtype=float))
    # Directional angular error is physically bounded by 180 degrees.
    prediction = min(180.0, max(0.0, prediction))
    return MotionReliability(prediction, True, str(calibration["model_version"]), "available")


def _validated_evidence(evidence: BranchOrientationEvidence) -> BranchOrientationEvidence:
    if not evidence.available or evidence.tangent_deg is None or evidence.uncertainty_deg is None:
        return BranchOrientationEvidence(None, None, False, evidence.source)
    if not math.isfinite(evidence.tangent_deg) or not math.isfinite(evidence.uncertainty_deg):
        return BranchOrientationEvidence(None, None, False, evidence.source)
    uncertainty = min(180.0, max(0.0, float(evidence.uncertainty_deg)))
    return BranchOrientationEvidence(
        normalize_angle_deg(evidence.tangent_deg), uncertainty, True, evidence.source
    )


def _inverse_scale_circular_mean(
    first: BranchOrientationEvidence, second: BranchOrientationEvidence
) -> tuple[float | None, float | None]:
    """Fuse consistent circular directions using inverse-P90-scale squares."""
    assert first.uncertainty_deg is not None and second.uncertainty_deg is not None
    if first.uncertainty_deg == 0.0 or second.uncertainty_deg == 0.0:
        if first.uncertainty_deg == second.uncertainty_deg == 0.0:
            if angular_error_deg(first.tangent_deg, second.tangent_deg) > 0.0:
                return None, None
            return first.tangent_deg, 0.0
        exact = first if first.uncertainty_deg == 0.0 else second
        return exact.tangent_deg, 0.0
    weights = np.asarray([
        1.0 / first.uncertainty_deg**2,
        1.0 / second.uncertainty_deg**2,
    ])
    angles = np.radians([first.tangent_deg, second.tangent_deg])
    x = float(np.sum(weights * np.cos(angles)))
    y = float(np.sum(weights * np.sin(angles)))
    if math.hypot(x, y) <= np.finfo(float).eps:
        return None, None
    tangent = normalize_angle_deg(math.degrees(math.atan2(y, x)))
    uncertainty = float(1.0 / math.sqrt(np.sum(weights)))
    return tangent, uncertainty


def fuse_branch_orientation(
    geometry_evidence: BranchOrientationEvidence,
    motion_evidence: BranchOrientationEvidence,
) -> FusedBranchOrientation:
    """Fuse two GT-free circular orientation evidences or safely abstain.

    Conflict is determined by non-overlap of the two predicted P90 angular
    intervals: ``disagreement > uncertainty_g + uncertainty_m``.  Dominance is
    declared only when the stronger source's center lies within the weaker
    interval while the reverse is false.  No fixed degree threshold is used.
    """
    geometry = _validated_evidence(geometry_evidence)
    motion = _validated_evidence(motion_evidence)
    if not geometry.available and not motion.available:
        return FusedBranchOrientation(None, None, "unavailable", False, False, None)
    if geometry.available and not motion.available:
        return FusedBranchOrientation(geometry.tangent_deg, geometry.uncertainty_deg, "geometry_only", True, False, None)
    if motion.available and not geometry.available:
        return FusedBranchOrientation(motion.tangent_deg, motion.uncertainty_deg, "motion_only", False, True, None)

    assert geometry.tangent_deg is not None and motion.tangent_deg is not None
    assert geometry.uncertainty_deg is not None and motion.uncertainty_deg is not None
    disagreement = angular_error_deg(geometry.tangent_deg, motion.tangent_deg)
    if disagreement > geometry.uncertainty_deg + motion.uncertainty_deg:
        return FusedBranchOrientation(None, None, "conflict", False, False, disagreement)
    smaller = min(geometry.uncertainty_deg, motion.uncertainty_deg)
    larger = max(geometry.uncertainty_deg, motion.uncertainty_deg)
    if disagreement > smaller and disagreement <= larger and geometry.uncertainty_deg != motion.uncertainty_deg:
        dominant = geometry if geometry.uncertainty_deg < motion.uncertainty_deg else motion
        status = "geometry_dominant" if dominant.source == "geometry" else "motion_dominant"
        return FusedBranchOrientation(
            dominant.tangent_deg,
            dominant.uncertainty_deg,
            status,
            dominant.source == "geometry",
            dominant.source == "motion",
            disagreement,
        )
    tangent, uncertainty = _inverse_scale_circular_mean(geometry, motion)
    if tangent is None:
        return FusedBranchOrientation(None, None, "conflict", False, False, disagreement)
    return FusedBranchOrientation(tangent, uncertainty, "agreement_fused", True, True, disagreement)


def _motion_oof(
    rows: Sequence[Mapping[str, Any]], folds: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_ids = _group_folds(rows, folds, seed)
    predictions = np.zeros(len(rows), dtype=float)
    for fold in range(folds):
        train_indices = np.flatnonzero(fold_ids != fold)
        test_indices = np.flatnonzero(fold_ids == fold)
        training = [rows[index] for index in train_indices]
        testing = [rows[index] for index in test_indices]
        calibration = _fit_motion_calibration(training)
        for index, row in zip(test_indices, testing):
            reliability = estimate_motion_reliability(
                row["stable_sample_count"], row["stable_robot_count"],
                row["stable_dispersion_deg"], row["stable_resultant"], calibration,
            )
            predictions[index] = float(reliability.predicted_p90_error_deg)
    target = np.asarray([float(row["actual_motion_error_deg"]) for row in rows])
    output = [
        {
            **dict(row),
            "predicted_p90_error_deg": predictions[index],
            "covered": target[index] <= predictions[index],
            "fold": int(fold_ids[index]),
        }
        for index, row in enumerate(rows)
    ]
    metrics = {
        "row_count": len(rows),
        "empirical_coverage": float(np.mean(target <= predictions)),
        "mean_predicted_bound_deg": float(np.mean(predictions)),
        "median_predicted_bound_deg": float(np.median(predictions)),
        "p90_predicted_bound_deg": float(np.percentile(predictions, 90)),
        "spearman_predicted_vs_actual": _spearman(predictions, target),
    }
    return output, metrics


def _geometry_evidence(
    case: Any,
    condition: Any,
    wall_calibration: Mapping[str, Any],
) -> tuple[BranchOrientationEvidence, dict[str, Any]]:
    geometry = case.geometry
    walls = geometry_wall_segments(geometry)
    maximum_extent = geometry.central_radius + max(branch.length for branch in geometry.branches)
    scan = simulate_lidar_scan(
        walls, geometry.center,
        angle_step_deg=condition.angle_step_deg,
        max_range_m=maximum_extent * 1.25,
        noise_std_m=condition.noise_std_m,
        dropout_probability=condition.dropout_probability,
        occlusion_probability=condition.occlusion_probability,
        visible_boundary_ratio=condition.visible_boundary_ratio,
        seed=geometry.seed,
    )
    openings = detect_openings(*scan.detector_input())
    gt_openings = ground_truth_openings_from_geometry(
        [branch.angle_deg for branch in geometry.branches],
        anchor_xy=geometry.center,
        anchor_yaw_deg=0.0,
        corridor_width_m=geometry.branches[0].width,
        central_radius_m=geometry.central_radius,
    )
    matches = _match_openings(gt_openings, openings)
    target = case.outgoing_branches[0]
    target_gt = min(
        range(len(gt_openings)),
        key=lambda index: angular_error_deg(gt_openings[index]["center_angle"], target.angle_deg),
    )
    match = next((item for item in matches if item[0] == target_gt), None)
    if match is None:
        return BranchOrientationEvidence(None, None, False, "geometry"), {
            "wall_estimate_mode": "opening_unmatched",
            "geometry_point_count": 0,
            "geometry_wall_span_m": "",
        }
    opening = openings[match[1]]
    estimate = estimate_wall_parallel_tangent(
        scan.angle_deg, scan.range_m, scan.max_range_m, opening
    )
    diagnostics = _candidate_diagnostics(
        scan.angle_deg, scan.range_m, scan.max_range_m, opening, estimate.estimate_mode
    )
    span = diagnostics["selected_wall_span_m"]
    reliability = estimate_wall_reliability(
        estimate, None if span == "" else float(span), wall_calibration
    )
    evidence = BranchOrientationEvidence(
        estimate.tangent_deg,
        reliability.predicted_p90_error_deg,
        reliability.available,
        "geometry",
    )
    return evidence, {
        "wall_estimate_mode": estimate.estimate_mode,
        "geometry_point_count": estimate.fitted_point_count,
        "geometry_wall_span_m": span,
        "geometry_raw_disagreement_deg": "" if estimate.wall_disagreement_deg is None else estimate.wall_disagreement_deg,
    }


def _evaluation_rows(
    cases: Sequence[Any],
    motion_rows: Mapping[str, Mapping[str, str]],
    robot_rows: Mapping[str, Mapping[str, str]],
    conditions: Sequence[Any],
    wall_calibration: Mapping[str, Any],
    motion_calibration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for case in cases:
        case_id = case.geometry.case_id
        target = case.outgoing_branches[0]
        motion_row = motion_rows[case_id]
        motion_available = motion_row["stable_tangent_deg"] != ""
        if motion_available:
            motion_reliability = estimate_motion_reliability(
                float(motion_row["stable_sample_count"]),
                float(robot_rows[case_id]["stable_robot_count"]),
                float(motion_row["stable_dispersion_deg"]),
                float(motion_row["stable_resultant"]),
                motion_calibration,
            )
            motion_evidence = BranchOrientationEvidence(
                float(motion_row["stable_tangent_deg"]),
                motion_reliability.predicted_p90_error_deg,
                motion_reliability.available,
                "motion",
            )
        else:
            motion_reliability = MotionReliability(None, False, MOTION_MODEL_VERSION, "motion_unavailable")
            motion_evidence = BranchOrientationEvidence(None, None, False, "motion")
        for condition in conditions:
            geometry_evidence, geometry_diag = _geometry_evidence(
                case, condition, wall_calibration
            )
            fused = fuse_branch_orientation(geometry_evidence, motion_evidence)
            geometry_error = "" if not geometry_evidence.available else angular_error_deg(geometry_evidence.tangent_deg, target.angle_deg)
            motion_error = "" if not motion_evidence.available else angular_error_deg(motion_evidence.tangent_deg, target.angle_deg)
            fusion_error = "" if fused.tangent_deg is None else angular_error_deg(fused.tangent_deg, target.angle_deg)
            geometry_good = geometry_error != "" and float(geometry_error) < EVALUATION_CORRECT_DEG
            motion_good = motion_error != "" and float(motion_error) < EVALUATION_CORRECT_DEG
            category = (
                "both_good" if geometry_good and motion_good
                else "geometry_good_motion_bad" if geometry_good
                else "motion_good_geometry_bad" if motion_good
                else "both_bad_or_unavailable"
            )
            output.append({
                "case_id": case_id,
                "branch_id": target.branch_id,
                "seed": case.geometry.seed,
                "sensor_condition": condition.condition_id,
                "turn_severity": case.turn_severity(target),
                "turn_angle_deg": case.turn_angle_deg(target),
                "width": target.width,
                "branch_length": target.length,
                "gt_tangent_deg": target.angle_deg,
                "geometry_tangent_deg": "" if not geometry_evidence.available else geometry_evidence.tangent_deg,
                "geometry_available": geometry_evidence.available,
                "geometry_uncertainty_deg": "" if not geometry_evidence.available else geometry_evidence.uncertainty_deg,
                **geometry_diag,
                "motion_tangent_deg": "" if not motion_evidence.available else motion_evidence.tangent_deg,
                "motion_available": motion_evidence.available,
                "motion_uncertainty_deg": "" if not motion_evidence.available else motion_evidence.uncertainty_deg,
                "stable_sample_count": motion_row["stable_sample_count"],
                "stable_robot_count": robot_rows[case_id]["stable_robot_count"],
                "stable_dispersion_deg": motion_row["stable_dispersion_deg"],
                "stable_resultant": motion_row["stable_resultant"],
                "source_disagreement_deg": "" if fused.disagreement_deg is None else fused.disagreement_deg,
                "fusion_status": fused.status,
                "final_tangent_deg": "" if fused.tangent_deg is None else fused.tangent_deg,
                "final_uncertainty_deg": "" if fused.uncertainty_deg is None else fused.uncertainty_deg,
                "geometry_used": fused.geometry_used,
                "motion_used": fused.motion_used,
                "complementarity_category": category,
                "geometry_error_deg": geometry_error,
                "motion_error_deg": motion_error,
                "fusion_error_deg": fusion_error,
                "geometry_bound_exceeded": geometry_error != "" and float(geometry_error) > float(geometry_evidence.uncertainty_deg),
                "motion_bound_exceeded": motion_error != "" and float(motion_error) > float(motion_evidence.uncertainty_deg),
                "fusion_bound_exceeded": fusion_error != "" and float(fusion_error) > float(fused.uncertainty_deg),
            })
    return output


def _stats(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows if row.get(key, "") != ""]
    return {
        "row_count": len(rows),
        "available_count": len(values),
        "coverage": len(values) / max(len(rows), 1),
        "mean_error_deg": "" if not values else float(np.mean(values)),
        "median_error_deg": "" if not values else float(np.median(values)),
        "p90_error_deg": "" if not values else float(np.percentile(values, 90)),
        "max_error_deg": "" if not values else float(np.max(values)),
    }


def _summaries(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    summary = []
    for estimator, key in (
        ("geometry", "geometry_error_deg"),
        ("motion", "motion_error_deg"),
        ("fusion", "fusion_error_deg"),
    ):
        summary.append({"group": "overall", "estimator": estimator, **_stats(rows, key)})
    statuses = sorted({str(row["fusion_status"]) for row in rows})
    by_status = []
    for status in statuses:
        selected = [row for row in rows if row["fusion_status"] == status]
        by_status.append({
            "fusion_status": status,
            "count": len(selected),
            **_stats(selected, "fusion_error_deg"),
        })
    categories = sorted({str(row["complementarity_category"]) for row in rows})
    unsafe = 0
    for category in categories:
        selected = [row for row in rows if row["complementarity_category"] == category]
        numeric_wrong = sum(
            row["fusion_error_deg"] != "" and float(row["fusion_error_deg"]) >= EVALUATION_CORRECT_DEG
            for row in selected
        )
        abstained = sum(row["fusion_error_deg"] == "" for row in selected)
        unsafe += numeric_wrong
        summary.append({
            "group": category,
            "estimator": "fusion",
            **_stats(selected, "fusion_error_deg"),
            "abstained_count": abstained,
            "numeric_wrong_at_existing_evaluation_boundary": numeric_wrong,
        })
    fusion_stats = _stats(rows, "fusion_error_deg")
    geometry_stats = _stats(rows, "geometry_error_deg")
    motion_stats = _stats(rows, "motion_error_deg")
    classification = (
        "A"
        if unsafe == 0
        and fusion_stats["p90_error_deg"] <= min(geometry_stats["p90_error_deg"], motion_stats["p90_error_deg"])
        else "B"
        if fusion_stats["p90_error_deg"] <= max(geometry_stats["p90_error_deg"], motion_stats["p90_error_deg"])
        or any(row["fusion_status"] == "conflict" for row in rows)
        else "C"
    )
    summary.extend([
        {"group": "safety", "estimator": "fusion", "metric": "conflict_count", "value": sum(row["fusion_status"] == "conflict" for row in rows)},
        {"group": "safety", "estimator": "fusion", "metric": "conflict_rate", "value": sum(row["fusion_status"] == "conflict" for row in rows) / max(len(rows), 1)},
        {"group": "safety", "estimator": "fusion", "metric": "unavailable_count", "value": sum(row["fusion_status"] == "unavailable" for row in rows)},
        {"group": "safety", "estimator": "fusion", "metric": "unavailable_rate", "value": sum(row["fusion_status"] == "unavailable" for row in rows) / max(len(rows), 1)},
        {"group": "usage", "estimator": "fusion", "metric": "geometry_used_count", "value": sum(bool(row["geometry_used"]) for row in rows)},
        {"group": "usage", "estimator": "fusion", "metric": "motion_used_count", "value": sum(bool(row["motion_used"]) for row in rows)},
        {"group": "safety", "estimator": "fusion", "metric": "uncertainty_bound_exceedance_count", "value": sum(bool(row["fusion_bound_exceeded"]) for row in rows)},
        {"group": "classification", "estimator": "fusion", "metric": "case", "value": classification},
    ])
    return summary, by_status, classification


def _save_plots(directory: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    labels = [f"{row['case_id'].replace('boundary_', '')}\n{row['sensor_condition']}" for row in rows]
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(16, 6), constrained_layout=True)
    axis.scatter(x, [np.nan if row["geometry_error_deg"] == "" else float(row["geometry_error_deg"]) for row in rows], s=16, label="geometry")
    axis.scatter(x, [np.nan if row["motion_error_deg"] == "" else float(row["motion_error_deg"]) for row in rows], s=16, label="motion")
    axis.scatter(x, [np.nan if row["fusion_error_deg"] == "" else float(row["fusion_error_deg"]) for row in rows], s=22, marker="x", label="fusion")
    axis.set_xticks(x, labels, rotation=90, fontsize=5)
    axis.set(ylabel="angular error [deg]", title="Geometry, motion, and safe fusion")
    axis.legend()
    figure.savefig(directory / "geometry_motion_fusion_error_comparison.png", dpi=160)
    plt.close(figure)

    paired = [row for row in rows if row["source_disagreement_deg"] != ""]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for status in sorted({row["fusion_status"] for row in paired}):
        selected = [row for row in paired if row["fusion_status"] == status]
        axis.scatter(
            [float(row["source_disagreement_deg"]) for row in selected],
            [np.nan if row["fusion_error_deg"] == "" else float(row["fusion_error_deg"]) for row in selected],
            label=status, alpha=0.65,
        )
    axis.set(xlabel="geometry-motion circular disagreement [deg]", ylabel="fusion error [deg]", title="Fusion behavior vs source disagreement")
    axis.legend(fontsize=8)
    figure.savefig(directory / "fusion_error_vs_source_disagreement.png", dpi=160)
    plt.close(figure)

    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["fusion_status"])] = counts.get(str(row["fusion_status"]), 0) + 1
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.bar(list(counts), list(counts.values()))
    axis.set(ylabel="row count", title="Traceable fusion status overview")
    axis.tick_params(axis="x", rotation=35)
    figure.savefig(directory / "fusion_status_overview.png", dpi=160)
    plt.close(figure)


def run_sanity_tests() -> None:
    """Audit circular wrap, 180° opposition, dominance, and rotation invariance."""
    zero_wrap = fuse_branch_orientation(
        BranchOrientationEvidence(359.0, 4.0, True, "geometry"),
        BranchOrientationEvidence(1.0, 4.0, True, "motion"),
    )
    assert zero_wrap.status == "agreement_fused"
    assert angular_error_deg(zero_wrap.tangent_deg, 0.0) < 1.0e-9
    first = BranchOrientationEvidence(179.0, 4.0, True, "geometry")
    second = BranchOrientationEvidence(-179.0, 4.0, True, "motion")
    wrapped = fuse_branch_orientation(first, second)
    assert wrapped.status == "agreement_fused"
    assert angular_error_deg(wrapped.tangent_deg, 180.0) < 1.0e-9
    opposed = fuse_branch_orientation(
        BranchOrientationEvidence(0.0, 2.0, True, "geometry"),
        BranchOrientationEvidence(180.0, 2.0, True, "motion"),
    )
    assert opposed.status == "conflict" and opposed.tangent_deg is None
    dominant = fuse_branch_orientation(
        BranchOrientationEvidence(10.0, 1.0, True, "geometry"),
        BranchOrientationEvidence(20.0, 15.0, True, "motion"),
    )
    assert dominant.status == "geometry_dominant" and dominant.tangent_deg == 10.0
    rotated = fuse_branch_orientation(
        BranchOrientationEvidence(79.0, 4.0, True, "geometry"),
        BranchOrientationEvidence(81.0, 4.0, True, "motion"),
    )
    assert angular_error_deg(rotated.tangent_deg, normalize_angle_deg(wrapped.tangent_deg - 100.0)) < 1.0e-9


def _audit_runtime_apis() -> None:
    for function in (estimate_motion_reliability, fuse_branch_orientation):
        names = " ".join(inspect.signature(function).parameters).lower()
        forbidden = [term for term in FORBIDDEN_RUNTIME_TERMS if term in names]
        if forbidden:
            raise AssertionError(f"{function.__name__} exposes forbidden runtime fields: {forbidden}")


def run_benchmark(
    output_dir: Path,
    *,
    seed: int,
    motion_comparison_csv: Path,
    motion_stability_csv: Path,
    paired_comparison_csv: Path,
    boundary_csv: Path,
    wall_calibration_json: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    motion_training = _motion_training_rows(motion_comparison_csv, motion_stability_csv)
    motion_oof, motion_metrics = _motion_oof(motion_training, 5, seed)
    replay_oof, replay_metrics = _motion_oof(motion_training, 5, seed)
    if _canonical_json(motion_oof) != _canonical_json(replay_oof):
        raise AssertionError("motion calibration deterministic replay mismatch")
    motion_calibration = _fit_motion_calibration(motion_training)
    motion_calibration.update({
        "source_comparison_csv": str(motion_comparison_csv),
        "source_stability_csv": str(motion_stability_csv),
        "source_head": _short_head(),
        "grouped_cv_folds": 5,
        "grouped_cv_metrics": motion_metrics,
        "runtime_forbidden_features": list(FORBIDDEN_RUNTIME_TERMS),
    })
    wall_calibration = json.loads(wall_calibration_json.read_text(encoding="utf-8"))
    motion_rows = {row["case_id"]: row for row in _read_rows(paired_comparison_csv)}
    robot_rows = {row["case_id"]: row for row in _read_rows(boundary_csv)}
    cases = _representative_cases(seed)
    conditions_by_id = {condition.condition_id: condition for condition in SENSOR_CONDITIONS}
    conditions = [conditions_by_id[condition_id] for condition_id in EVALUATION_SENSOR_IDS]
    rows = _evaluation_rows(
        cases, motion_rows, robot_rows, conditions, wall_calibration, motion_calibration
    )
    replay_rows = _evaluation_rows(
        cases, motion_rows, robot_rows, conditions, wall_calibration, motion_calibration
    )
    if _canonical_json(rows) != _canonical_json(replay_rows):
        raise AssertionError("fusion deterministic replay mismatch")
    summary, by_status, classification = _summaries(rows)
    failures = sorted(
        [
            {
                "case_id": row["case_id"],
                "sensor_condition": row["sensor_condition"],
                "fusion_status": row["fusion_status"],
                "geometry_error_deg": row["geometry_error_deg"],
                "motion_error_deg": row["motion_error_deg"],
                "fusion_error_deg": row["fusion_error_deg"],
                "source_disagreement_deg": row["source_disagreement_deg"],
                "reason": "safe_abstention" if row["fusion_error_deg"] == "" else "numeric_output",
            }
            for row in rows
            if row["fusion_error_deg"] == ""
            or bool(row["fusion_bound_exceeded"])
        ],
        key=lambda row: -1.0 if row["fusion_error_deg"] == "" else float(row["fusion_error_deg"]),
        reverse=True,
    )
    _write_rows(output_dir / "branch_orientation_fusion_results.csv", rows)
    _write_rows(output_dir / "branch_orientation_fusion_summary.csv", summary)
    _write_rows(output_dir / "branch_orientation_fusion_by_status.csv", by_status)
    _write_rows(output_dir / "branch_orientation_fusion_failure_cases.csv", failures)
    _write_rows(output_dir / "motion_reliability_oof.csv", motion_oof)
    (output_dir / "motion_reliability_calibration.json").write_text(
        json.dumps(motion_calibration, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    _save_plots(output_dir, rows)
    result = {
        "classification": classification,
        "physical_case_count": len(cases),
        "sensor_condition_count": len(conditions),
        "paired_row_count": len(rows),
        "motion_oof": motion_metrics,
        "geometry": _stats(rows, "geometry_error_deg"),
        "motion": _stats(rows, "motion_error_deg"),
        "fusion": _stats(rows, "fusion_error_deg"),
        "status_counts": {
            status: sum(row["fusion_status"] == status for row in rows)
            for status in sorted({row["fusion_status"] for row in rows})
        },
        "source_usage_counts": {
            "geometry": sum(bool(row["geometry_used"]) for row in rows),
            "motion": sum(bool(row["motion_used"]) for row in rows),
        },
        "conflict_rate": sum(row["fusion_status"] == "conflict" for row in rows) / max(len(rows), 1),
        "unavailable_rate": sum(row["fusion_status"] == "unavailable" for row in rows) / max(len(rows), 1),
        "deterministic_replay": True,
        "source_sha256": {
            "wall_calibration": hashlib.sha256(wall_calibration_json.read_bytes()).hexdigest(),
            "motion_comparison": hashlib.sha256(motion_comparison_csv.read_bytes()).hexdigest(),
        },
        "output_dir": str(output_dir),
    }
    (output_dir / "branch_orientation_fusion_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output-dir", type=Path, default=Path(f"/tmp/pdfs_branch_orientation_fusion_{_short_head()}"))
    parser.add_argument("--motion-comparison-csv", type=Path, default=Path("/tmp/pdfs_general_sph_inlet_tangent_3b37ddf/sph_inlet_tangent_comparison.csv"))
    parser.add_argument("--motion-stability-csv", type=Path, default=Path("/tmp/pdfs_general_sph_inlet_tangent_3b37ddf/sph_inlet_turning_stability_summary.csv"))
    parser.add_argument("--paired-comparison-csv", type=Path, default=Path("/tmp/pdfs_geometry_motion_comparison_3b37ddf/geometry_motion_comparison.csv"))
    parser.add_argument("--boundary-csv", type=Path, default=Path("/tmp/pdfs_stable_tangent_boundary_3b37ddf/failure_boundary_results.csv"))
    parser.add_argument("--wall-calibration-json", type=Path, default=Path("/tmp/pdfs_wall_reliability_3ff9e0b/wall_reliability_calibration.json"))
    parser.add_argument("--sanity-test", action="store_true")
    args = parser.parse_args()
    if args.sanity_test:
        run_sanity_tests()
        print("fusion circular/ambiguity/rotation sanity: PASS")
    _audit_runtime_apis()
    result = run_benchmark(
        args.output_dir,
        seed=args.seed,
        motion_comparison_csv=args.motion_comparison_csv,
        motion_stability_csv=args.motion_stability_csv,
        paired_comparison_csv=args.paired_comparison_csv,
        boundary_csv=args.boundary_csv,
        wall_calibration_json=args.wall_calibration_json,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
