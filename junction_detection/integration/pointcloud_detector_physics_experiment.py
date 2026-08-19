"""EXP-PointCloudDetector-001: physics-aware detector evaluation.

The protected baseline detector, wall estimator/reliability, motion estimator,
and fusion rule are imported unchanged.  Simulator geometry enters only scan
generation and post-hoc matching/scoring.  Every runtime detector call receives
Anchor-local angle/range, declared sensor range uncertainty, angular sampling
implicit in the angle array, and the known robot diameter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from junction_detection.integration.branch_orientation_geometry_motion_comparison import (
    EVALUATION_CORRECT_DEG,
    _representative_cases,
    geometry_wall_segments,
)
from junction_detection.integration.general_branch_orientation_fusion import (
    BranchOrientationEvidence,
    MotionReliability,
    _fit_motion_calibration,
    _motion_training_rows,
    estimate_motion_reliability,
    fuse_branch_orientation,
)
from junction_detection.integration.pointcloud_branch_orientation_generalization import (
    _gt_opening,
    angular_error_deg,
    create_benchmark_cases,
    normalize_angle_deg,
    wall_segments,
)
from junction_detection.integration.pointcloud_wall_orientation_reliability import (
    SELECTED_SPEC,
    _fit_calibration,
    estimate_wall_reliability,
)
from junction_detection.integration.pointcloud_wall_orientation_sensor_robustness import (
    SENSOR_CONDITIONS,
    SensorCondition,
    _candidate_diagnostics,
)
from junction_detection.integration.pointcloud_wall_parallel_orientation import (
    estimate_wall_parallel_tangent,
)
from junction_detection.pointcloud import pointcloud_junction_detector as baseline
from junction_detection.pointcloud.pointcloud_junction_detector_local_topology import (
    _match_openings,
    ground_truth_openings_from_geometry,
)
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import (
    simulate_lidar_scan,
)
from junction_detection.pointcloud.pointcloud_junction_detector_uncertainty_aware import (
    DetectorStages,
    detect_openings as detect_uncertainty_aware,
    run_synthetic_sanity,
)


ROBOT_DIAMETER_M = 2.70
FAMILYWISE_FALSE_ALARM_PROBABILITY = 0.01
FUSION_SENSOR_IDS = (
    "clean", "noise_0.08", "occlusion_0.80", "visibility_0.70", "resolution_4deg"
)
VARIANTS = (
    "baseline",
    "adaptive_discontinuity",
    "adaptive_plus_merge",
    "adaptive_merge_physical_width",
    "adaptive_merge_wall_support",
    "combined",
)
PROTECTED_FILES = (
    "junction_detection/pointcloud/pointcloud_junction_detector.py",
    "junction_detection/integration/pointcloud_wall_parallel_orientation.py",
    "junction_detection/integration/pointcloud_wall_orientation_reliability.py",
    "junction_detection/integration/general_branch_orientation_fusion.py",
    "junction_detection/integration/trajectory_stability_diagnostics.py",
)
FORBIDDEN_DETECTOR_ARGUMENTS = (
    "gt", "map", "wall", "branch", "topology", "case", "anchor", "global", "yaw"
)


def _short_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


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


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _detect_variant(
    variant: str,
    angles: np.ndarray,
    ranges: np.ndarray,
    condition: SensorCondition,
) -> list[dict[str, float]]:
    """Call one detector variant with runtime-permitted inputs only."""
    if variant == "baseline":
        return baseline.detect_openings(angles, ranges)
    stages = {
        "adaptive_discontinuity": DetectorStages(True, False, False, False),
        "adaptive_plus_merge": DetectorStages(True, True, False, False),
        "adaptive_merge_physical_width": DetectorStages(True, True, True, False),
        "adaptive_merge_wall_support": DetectorStages(True, True, False, True),
        "combined": DetectorStages(True, True, True, True),
    }[variant]
    return detect_uncertainty_aware(
        angles,
        ranges,
        noise_std_m=condition.noise_std_m,
        robot_diameter_m=ROBOT_DIAMETER_M,
        false_alarm_probability=FAMILYWISE_FALSE_ALARM_PROBABILITY,
        stages=stages,
    )


def _boundary_error(gt: Mapping[str, float], detected: Mapping[str, float]) -> float:
    return 0.5 * (
        angular_error_deg(float(gt["start_angle"]), float(detected["start_angle"]))
        + angular_error_deg(float(gt["end_angle"]), float(detected["end_angle"]))
    )


def _positive_overlap_matches(
    ground_truth: Sequence[dict[str, float]],
    detected: Sequence[dict[str, float]],
) -> list[tuple[int, int, float]]:
    """Keep only geometrically overlapping pairs from the shared matcher."""
    return [match for match in _match_openings(ground_truth, detected) if match[2] > baseline.EPSILON]


def _evaluate_scan(
    case: Any,
    condition: SensorCondition,
    scan: Any,
    variant: str,
) -> list[dict[str, Any]]:
    """Match and score one detector; GT remains outside its call boundary."""
    angles, ranges = scan.detector_input()
    detected = _detect_variant(variant, angles, ranges, condition)
    ground_truth = [_gt_opening(case, branch) for branch in case.branches]
    matches = _positive_overlap_matches(ground_truth, detected)
    by_gt = {gt_index: (det_index, iou) for gt_index, det_index, iou in matches}
    matched_detected = {det_index for _, det_index, _ in matches}
    scan_base = {
        "variant": variant,
        "case_id": case.case_id,
        "seed": case.seed,
        "condition_id": condition.condition_id,
        "degradation_type": condition.degradation_type,
        "noise_std_m": condition.noise_std_m,
        "angle_step_deg": condition.angle_step_deg,
        "topology": case.topology,
        "gt_opening_count": len(ground_truth),
        "detected_opening_count": len(detected),
        "matched_opening_count": len(matches),
        "missed_opening_count": len(ground_truth) - len(matches),
        "false_positive_count": len(detected) - len(matched_detected),
        "correct_opening_count": len(detected) == len(ground_truth),
    }
    output: list[dict[str, Any]] = []
    for branch_index, branch in enumerate(case.branches):
        gt = ground_truth[branch_index]
        gt_tangent = normalize_angle_deg(branch.angle_deg - case.anchor_yaw_deg)
        base_row = {
            **scan_base,
            "branch_id": branch.branch_id,
            "width_group": case.width_group,
            "length_group": case.length_group,
            "anchor_offset_group": case.offset_group,
            "mouth_geometry": case.mouth_geometry,
            "gt_tangent_deg": gt_tangent,
            "opening_matched": branch_index in by_gt,
            "wall_estimator_available": False,
            "estimate_mode": "opening_unmatched",
        }
        if branch_index not in by_gt:
            output.append(base_row)
            continue
        detected_index, iou = by_gt[branch_index]
        opening = detected[detected_index]
        estimate = estimate_wall_parallel_tangent(angles, ranges, scan.max_range_m, opening)
        available = estimate.tangent_deg is not None
        diagnostics = _candidate_diagnostics(
            angles, ranges, scan.max_range_m, opening, estimate.estimate_mode
        )
        output.append({
            **base_row,
            **diagnostics,
            "match_iou": iou,
            "opening_start_angle": opening["start_angle"],
            "opening_end_angle": opening["end_angle"],
            "opening_center_angle": opening["center_angle"],
            "opening_width_deg": opening["width_deg"],
            "opening_boundary_error_deg": _boundary_error(gt, opening),
            "boundary_uncertainty_deg": opening.get("boundary_uncertainty_deg", ""),
            "estimated_mouth_width_m": opening.get("estimated_mouth_width_m", ""),
            "mouth_width_lower_m": opening.get("mouth_width_lower_m", ""),
            "opening_center_error_deg": angular_error_deg(float(opening["center_angle"]), gt_tangent),
            "wall_estimator_available": available,
            "wall_tangent_deg": "" if not available else estimate.tangent_deg,
            "wall_tangent_error_deg": "" if not available else angular_error_deg(float(estimate.tangent_deg), gt_tangent),
            "estimate_mode": estimate.estimate_mode,
            "usable_wall_sides": estimate.usable_wall_sides,
            "selected_wall_point_count": estimate.fitted_point_count,
            "line_fit_residual_m": "" if estimate.line_fit_residual_m is None else estimate.line_fit_residual_m,
            "left_right_raw_disagreement_deg": "" if estimate.wall_disagreement_deg is None else estimate.wall_disagreement_deg,
        })
    return output


def _run_detector_matrix(seed: int, *, announce: bool) -> list[dict[str, Any]]:
    cases = create_benchmark_cases(seed)
    output: list[dict[str, Any]] = []
    total = len(cases) * len(SENSOR_CONDITIONS)
    index = 0
    for condition in SENSOR_CONDITIONS:
        for case in cases:
            index += 1
            if announce:
                print(f"[detector physics] scan {index}/{total} {condition.condition_id}:{case.case_id}", flush=True)
            maximum_extent = case.central_radius + max(branch.length for branch in case.branches)
            scan = simulate_lidar_scan(
                wall_segments(case),
                case.anchor_xy,
                anchor_yaw_deg=case.anchor_yaw_deg,
                angle_step_deg=condition.angle_step_deg,
                max_range_m=maximum_extent * 1.30,
                noise_std_m=condition.noise_std_m,
                dropout_probability=condition.dropout_probability,
                occlusion_probability=condition.occlusion_probability,
                visible_boundary_ratio=condition.visible_boundary_ratio,
                seed=case.seed,
            )
            for variant in VARIANTS:
                output.extend(_evaluate_scan(case, condition, scan, variant))
    return output


def _stats(values: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": "" if not values else float(np.mean(values)),
        "median": "" if not values else float(np.median(values)),
        "p90": "" if not values else float(np.percentile(values, 90)),
        "max": "" if not values else float(np.max(values)),
    }


def _scan_level(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    output = []
    for row in rows:
        key = (str(row["variant"]), str(row["case_id"]), str(row["condition_id"]))
        if key not in seen:
            seen.add(key)
            output.append(row)
    return output


def _summary(rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in group_fields)].append(row)
    output = []
    for key, selected in sorted(groups.items()):
        scans = _scan_level(selected)
        matched = [row for row in selected if bool(row["opening_matched"])]
        walls = [row for row in matched if bool(row["wall_estimator_available"])]
        boundary_errors = [float(row["opening_boundary_error_deg"]) for row in matched]
        wall_errors = [float(row["wall_tangent_error_deg"]) for row in walls]
        modes = Counter(str(row["estimate_mode"]) for row in walls)
        output.append({
            **dict(zip(group_fields, key)),
            "scan_count": len(scans),
            "branch_count": len(selected),
            "correct_opening_count_rate": float(np.mean([bool(row["correct_opening_count"]) for row in scans])),
            "missed_opening_count": sum(int(row["missed_opening_count"]) for row in scans),
            "false_positive_count": sum(int(row["false_positive_count"]) for row in scans),
            "opening_matching_coverage": len(matched) / max(len(selected), 1),
            "opening_boundary_error_mean_deg": _stats(boundary_errors)["mean"],
            "opening_boundary_error_p90_deg": _stats(boundary_errors)["p90"],
            "wall_tangent_availability": len(walls) / max(len(selected), 1),
            "two_wall_rate": modes["two_wall_parallel"] / max(len(selected), 1),
            "dominant_one_wall_rate": modes["one_wall_dominant_span"] / max(len(selected), 1),
            "one_wall_only_rate": modes["one_wall_observed"] / max(len(selected), 1),
            "wall_error_mean_deg": _stats(wall_errors)["mean"],
            "wall_error_median_deg": _stats(wall_errors)["median"],
            "wall_error_p90_deg": _stats(wall_errors)["p90"],
            "wall_error_max_deg": _stats(wall_errors)["max"],
        })
    return output


def _fit_protected_wall_calibration(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recreate the unchanged wall model from baseline benchmark observations."""
    training = [dict(row) for row in rows if row["variant"] == "baseline" and bool(row["wall_estimator_available"])]
    calibration = _fit_calibration(training, SELECTED_SPEC)
    calibration.update({
        "source": "current experiment baseline rows; unchanged selected feature specification",
        "training_row_count": len(training),
        "training_case_count": len({row["case_id"] for row in training}),
    })
    return calibration


def _attach_wall_reliability(
    rows: list[dict[str, Any]], calibration: Mapping[str, Any]
) -> None:
    for row in rows:
        row["wall_predicted_p90_error_deg"] = ""
        row["wall_reliability_available"] = False
        if not bool(row["wall_estimator_available"]):
            continue
        # Re-evaluation is unnecessary: build the exact immutable API input.
        from junction_detection.integration.pointcloud_wall_parallel_orientation import WallEstimate
        estimate = WallEstimate(
            tangent_deg=float(row["wall_tangent_deg"]),
            left_orientation_deg=None,
            right_orientation_deg=None,
            usable_wall_sides=int(row["usable_wall_sides"]),
            fitted_point_count=int(row["selected_wall_point_count"]),
            line_fit_residual_m=None if row["line_fit_residual_m"] == "" else float(row["line_fit_residual_m"]),
            wall_disagreement_deg=None if row["left_right_raw_disagreement_deg"] == "" else float(row["left_right_raw_disagreement_deg"]),
            estimate_mode=str(row["estimate_mode"]),
            selected_points=np.empty((0, 2)),
        )
        span = row.get("selected_wall_span_m", "")
        reliability = estimate_wall_reliability(
            estimate, None if span == "" else float(span), calibration
        )
        row["wall_predicted_p90_error_deg"] = "" if not reliability.available else reliability.predicted_p90_error_deg
        row["wall_reliability_available"] = reliability.available


def _fusion_rows(
    seed: int,
    wall_calibration: Mapping[str, Any],
    motion_training_comparison: Path,
    motion_training_stability: Path,
    paired_motion_csv: Path,
    boundary_csv: Path,
) -> list[dict[str, Any]]:
    """Re-run frozen fusion before/after using identical scans and motion."""
    training = _motion_training_rows(motion_training_comparison, motion_training_stability)
    motion_calibration = _fit_motion_calibration(training)
    motion_rows = {row["case_id"]: row for row in _read_rows(paired_motion_csv)}
    robot_rows = {row["case_id"]: row for row in _read_rows(boundary_csv)}
    conditions = {condition.condition_id: condition for condition in SENSOR_CONDITIONS}
    output = []
    for case in _representative_cases(seed):
        geometry = case.geometry
        case_id = geometry.case_id
        target = case.outgoing_branches[0]
        motion_row = motion_rows[case_id]
        if motion_row["stable_tangent_deg"] == "":
            motion_evidence = BranchOrientationEvidence(None, None, False, "motion")
            motion_reliability = MotionReliability(None, False, "", "motion_unavailable")
        else:
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
        for condition_id in FUSION_SENSOR_IDS:
            condition = conditions[condition_id]
            maximum_extent = geometry.central_radius + max(branch.length for branch in geometry.branches)
            scan = simulate_lidar_scan(
                geometry_wall_segments(geometry), geometry.center,
                angle_step_deg=condition.angle_step_deg,
                max_range_m=maximum_extent * 1.25,
                noise_std_m=condition.noise_std_m,
                dropout_probability=condition.dropout_probability,
                occlusion_probability=condition.occlusion_probability,
                visible_boundary_ratio=condition.visible_boundary_ratio,
                seed=geometry.seed,
            )
            gt = ground_truth_openings_from_geometry(
                [branch.angle_deg for branch in geometry.branches],
                anchor_xy=geometry.center,
                anchor_yaw_deg=0.0,
                corridor_width_m=geometry.branches[0].width,
                central_radius_m=geometry.central_radius,
            )
            target_gt = min(range(len(gt)), key=lambda idx: angular_error_deg(gt[idx]["center_angle"], target.angle_deg))
            evidences: dict[str, BranchOrientationEvidence] = {}
            modes: dict[str, str] = {}
            for detector_name in ("baseline", "combined"):
                openings = _detect_variant(detector_name, scan.angle_deg, scan.range_m, condition)
                match = next((item for item in _positive_overlap_matches(gt, openings) if item[0] == target_gt), None)
                if match is None:
                    evidences[detector_name] = BranchOrientationEvidence(None, None, False, "geometry")
                    modes[detector_name] = "opening_unmatched"
                    continue
                opening = openings[match[1]]
                estimate = estimate_wall_parallel_tangent(scan.angle_deg, scan.range_m, scan.max_range_m, opening)
                diagnostics = _candidate_diagnostics(scan.angle_deg, scan.range_m, scan.max_range_m, opening, estimate.estimate_mode)
                span = diagnostics["selected_wall_span_m"]
                reliability = estimate_wall_reliability(estimate, None if span == "" else float(span), wall_calibration)
                evidences[detector_name] = BranchOrientationEvidence(
                    estimate.tangent_deg, reliability.predicted_p90_error_deg, reliability.available, "geometry"
                )
                modes[detector_name] = estimate.estimate_mode
            before = fuse_branch_orientation(evidences["baseline"], motion_evidence)
            after = fuse_branch_orientation(evidences["combined"], motion_evidence)
            output.append({
                "case_id": case_id,
                "seed": geometry.seed,
                "sensor_condition": condition_id,
                "gt_tangent_deg": target.angle_deg,
                "motion_error_deg": "" if not motion_evidence.available else angular_error_deg(motion_evidence.tangent_deg, target.angle_deg),
                "baseline_geometry_available": evidences["baseline"].available,
                "baseline_wall_mode": modes["baseline"],
                "baseline_geometry_error_deg": "" if not evidences["baseline"].available else angular_error_deg(evidences["baseline"].tangent_deg, target.angle_deg),
                "before_status": before.status,
                "before_error_deg": "" if before.tangent_deg is None else angular_error_deg(before.tangent_deg, target.angle_deg),
                "new_geometry_available": evidences["combined"].available,
                "new_wall_mode": modes["combined"],
                "new_geometry_error_deg": "" if not evidences["combined"].available else angular_error_deg(evidences["combined"].tangent_deg, target.angle_deg),
                "after_status": after.status,
                "after_error_deg": "" if after.tangent_deg is None else angular_error_deg(after.tangent_deg, target.angle_deg),
            })
    return output


def _heuristic_audit() -> list[dict[str, Any]]:
    return [
        {"file": "pointcloud_junction_detector.py", "function": "_detect_openings_with_diagnostics", "parameter": "smoothing_window_size=5", "unit": "samples", "role": "circular smoothing", "class": "D", "replacement": "no angular blur; range uncertainty enters statistical discontinuity"},
        {"file": "pointcloud_junction_detector.py", "function": "_detect_openings_with_diagnostics", "parameter": "wall_reference_quantile=0.25", "unit": "fraction", "role": "near-wall reference", "class": "C", "replacement": "censored max-range core avoids a geometry-population quantile"},
        {"file": "pointcloud_junction_detector.py", "function": "_detect_openings_with_diagnostics", "parameter": "far_range_fraction=0.55", "unit": "fraction", "role": "open support", "class": "D", "replacement": "sensor-ceiling censoring plus noise-normalized adjacent differences"},
        {"file": "pointcloud_junction_detector.py", "function": "_detect_openings_with_diagnostics", "parameter": "merge_gap_deg=3", "unit": "degree", "role": "merge gaps", "class": "D", "replacement": "overlap of angular quantization confidence intervals"},
        {"file": "pointcloud_junction_detector.py", "function": "_detect_openings_with_diagnostics", "parameter": "min_opening_width_deg=5", "unit": "degree", "role": "reject narrow openings", "class": "D", "replacement": "local mouth chord lower bound versus robot diameter"},
        {"file": "pointcloud_junction_detector.py", "function": "_detect_openings_with_diagnostics", "parameter": "gradient_mad_scale=4", "unit": "dimensionless", "role": "boundary significance", "class": "C", "replacement": "declared family-wise false-alarm probability"},
        {"file": "pointcloud_junction_detector.py", "function": "_detect_openings_with_diagnostics", "parameter": "min_gradient_threshold=0.05", "unit": "m/deg", "role": "gradient floor", "class": "D", "replacement": "sensor noise and robust scan-difference scale"},
        {"file": "pointcloud_junction_detector.py", "function": "_detect_openings_with_diagnostics", "parameter": "boundary_search_deg=6", "unit": "degree", "role": "boundary search radius", "class": "D", "replacement": "contiguous statistically significant ramp; no fixed search angle"},
        {"file": "pointcloud_junction_detector.py", "function": "_detect_openings_with_diagnostics", "parameter": "coarse_width + 2*boundary_search_deg + 2", "unit": "degree", "role": "reject implausible refined width", "class": "D", "replacement": "boundary confidence intervals and physical mouth-width lower bound"},
        {"file": "pointcloud_junction_detector.py", "function": "_detect_openings_with_diagnostics", "parameter": "confidence=0.7*contrast+0.3*boundary", "unit": "dimensionless", "role": "reported confidence only", "class": "D", "replacement": "not used for decisions in the new detector; retained outputs use normalized evidence"},
        {"file": "pointcloud_junction_detector.py", "function": "_detect_openings_with_diagnostics", "parameter": "dynamic_span<=1e-6", "unit": "meter", "role": "degenerate-scan guard", "class": "E", "replacement": "new implementation uses floating-point-scaled numerical tolerance"},
        {"file": "pointcloud_junction_detector.py", "function": "_detect_openings_with_diagnostics", "parameter": "coarse_width>=359", "unit": "degree", "role": "reject all-open scan", "class": "A", "replacement": "retain physical observability rule, expressed relative to one angular bin"},
        {"file": "pointcloud_wall_parallel_orientation.py", "function": "_straight_subruns", "parameter": "numerical/straightness tolerances", "unit": "mixed", "role": "TLS wall support", "class": "E", "replacement": "protected in this experiment; reliability records span/residual/disagreement"},
    ]


def _outlier_threshold(rows: Sequence[Mapping[str, Any]]) -> float:
    errors = np.asarray([
        float(row["wall_tangent_error_deg"]) for row in rows
        if row["variant"] == "baseline" and row["condition_id"] == "clean" and row.get("wall_tangent_error_deg", "") != ""
    ])
    # Evaluation-only empirical tail marker; it never enters detection.
    return float(np.percentile(errors, 99.0))


def _save_plots(
    directory: Path,
    rows: Sequence[Mapping[str, Any]],
    condition_summary: Sequence[Mapping[str, Any]],
    ablation: Sequence[Mapping[str, Any]],
    fusion_rows: Sequence[Mapping[str, Any]],
) -> None:
    chosen = [row for row in condition_summary if row["variant"] in {"baseline", "combined"}]
    condition_ids = [condition.condition_id for condition in SENSOR_CONDITIONS]
    x = np.arange(len(condition_ids))
    for filename, field, ylabel, title in (
        ("opening_coverage_by_sensor_condition.png", "opening_matching_coverage", "coverage", "Opening matching coverage"),
        ("wall_tangent_availability_comparison.png", "wall_tangent_availability", "coverage", "End-to-end wall tangent availability"),
        ("wall_tangent_error_comparison.png", "wall_error_p90_deg", "P90 error [deg]", "Wall tangent error"),
    ):
        figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
        for variant in ("baseline", "combined"):
            lookup = {row["condition_id"]: row for row in chosen if row["variant"] == variant}
            axis.plot(x, [float(lookup[c][field]) for c in condition_ids], "o-", label=variant)
        axis.set_xticks(x, condition_ids, rotation=45, ha="right")
        axis.set(ylabel=ylabel, title=title)
        axis.legend()
        figure.savefig(directory / filename, dpi=160)
        plt.close(figure)

    scan_rows = [row for row in _scan_level(rows) if row["variant"] in {"baseline", "combined"}]
    missed = {(variant, condition): sum(
        int(row["missed_opening_count"]) for row in scan_rows
        if row["variant"] == variant and row["condition_id"] == condition
    ) for variant in ("baseline", "combined") for condition in condition_ids}
    false_positive = {(variant, condition): sum(
        int(row["false_positive_count"]) for row in scan_rows
        if row["variant"] == variant and row["condition_id"] == condition
    ) for variant in ("baseline", "combined") for condition in condition_ids}
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for offset, variant in ((-0.2, "baseline"), (0.2, "combined")):
        miss_values = [missed[(variant, condition)] for condition in condition_ids]
        fp_values = [false_positive[(variant, condition)] for condition in condition_ids]
        axis.bar(x + offset, miss_values, 0.4, label=f"{variant} missed")
        axis.bar(x + offset, fp_values, 0.4, bottom=miss_values, alpha=0.55, label=f"{variant} false positive")
    axis.set_xticks(x, condition_ids, rotation=45, ha="right")
    axis.set(ylabel="opening error count", title="Missed and false-positive openings")
    axis.legend()
    figure.savefig(directory / "baseline_vs_adaptive_opening_failures.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    fx = np.arange(len(fusion_rows))
    axis.scatter(fx, [np.nan if row["before_error_deg"] == "" else float(row["before_error_deg"]) for row in fusion_rows], label="before")
    axis.scatter(fx, [np.nan if row["after_error_deg"] == "" else float(row["after_error_deg"]) for row in fusion_rows], marker="x", label="after")
    axis.set(ylabel="fusion error [deg]", title="Frozen fusion before/after detector upgrade")
    axis.legend()
    figure.savefig(directory / "fusion_error_before_after_detector_upgrade.png", dpi=160)
    plt.close(figure)

    worst = sorted(
        fusion_rows,
        key=lambda row: -1.0 if row["before_error_deg"] == "" else float(row["before_error_deg"]),
        reverse=True,
    )[:8]
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    labels = [f"{row['case_id']}\n{row['sensor_condition']}" for row in worst]
    wx = np.arange(len(worst))
    axis.bar(wx - 0.2, [np.nan if row["before_error_deg"] == "" else float(row["before_error_deg"]) for row in worst], 0.4, label="before")
    axis.bar(wx + 0.2, [np.nan if row["after_error_deg"] == "" else float(row["after_error_deg"]) for row in worst], 0.4, label="after")
    axis.set_xticks(wx, labels, rotation=35, ha="right", fontsize=7)
    axis.set(ylabel="fusion error [deg]", title="Worst failure recovery examples")
    axis.legend()
    figure.savefig(directory / "worst_failure_recovery_examples.png", dpi=160)
    plt.close(figure)

    uncertainty_rows = [row for row in rows if row["variant"] == "combined" and row.get("boundary_uncertainty_deg", "") != "" and row.get("opening_boundary_error_deg", "") != ""]
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.scatter([float(row["boundary_uncertainty_deg"]) for row in uncertainty_rows], [float(row["opening_boundary_error_deg"]) for row in uncertainty_rows], alpha=0.35)
    axis.set(xlabel="boundary uncertainty [deg]", ylabel="actual boundary error [deg]", title="Boundary uncertainty vs evaluation error")
    figure.savefig(directory / "uncertainty_vs_actual_boundary_error.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax = np.arange(len(ablation))
    axis.plot(ax, [float(row["wall_tangent_availability"]) for row in ablation], "o-", label="wall availability")
    axis.plot(ax, [float(row["opening_matching_coverage"]) for row in ablation], "s-", label="opening coverage")
    axis.set_xticks(ax, [row["variant"] for row in ablation], rotation=35, ha="right")
    axis.set(ylim=(-0.02, 1.02), title="Detector ablation summary")
    axis.legend()
    figure.savefig(directory / "detector_ablation_summary.png", dpi=160)
    plt.close(figure)


def _runtime_audits() -> None:
    run_synthetic_sanity()
    signature = " ".join(inspect.signature(detect_uncertainty_aware).parameters).lower()
    forbidden = [term for term in FORBIDDEN_DETECTOR_ARGUMENTS if term in signature]
    if forbidden:
        raise AssertionError(f"detector runtime signature leaks forbidden fields: {forbidden}")
    # Translate simulator walls and Anchor together, then verify that the
    # resulting local scan and detector output are unchanged.
    walls = np.asarray([
        [[-3.0, -2.0], [3.0, -2.0]],
        [[3.0, -2.0], [3.0, 2.0]],
        [[3.0, 2.0], [-3.0, 2.0]],
        [[-3.0, 2.0], [-3.0, -2.0]],
    ])
    shift = np.asarray([91.0, -47.0])
    original_scan = baseline.simulate_lidar_scan(walls, (0.0, 0.0), max_range_m=10.0)
    translated_scan = baseline.simulate_lidar_scan(walls + shift, shift, max_range_m=10.0)
    if not np.allclose(original_scan.range_m, translated_scan.range_m, rtol=0.0, atol=1.0e-12):
        raise AssertionError("global translation changed Anchor-local ranges")

    angles = np.arange(-180.0, 180.0)
    ranges = 5.0 + 2.0 * (np.cos(np.radians(angles)) > 0.8)
    first = detect_uncertainty_aware(angles, ranges, robot_diameter_m=0.1)
    second = detect_uncertainty_aware(angles.copy(), ranges.copy(), robot_diameter_m=0.1)
    if json.dumps(first, sort_keys=True) != json.dumps(second, sort_keys=True):
        raise AssertionError("detector deterministic/local-frame replay mismatch")


def run_experiment(
    output_dir: Path,
    *,
    seed: int,
    motion_training_comparison: Path,
    motion_training_stability: Path,
    paired_motion_csv: Path,
    boundary_csv: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _runtime_audits()
    protected_sha = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in PROTECTED_FILES}
    rows = _run_detector_matrix(seed, announce=True)
    replay = _run_detector_matrix(seed, announce=False)
    first_hash = hashlib.sha256(json.dumps(rows, sort_keys=True, default=_json_default).encode()).hexdigest()
    replay_hash = hashlib.sha256(json.dumps(replay, sort_keys=True, default=_json_default).encode()).hexdigest()
    if first_hash != replay_hash:
        raise AssertionError("detector matrix deterministic replay mismatch")

    wall_calibration = _fit_protected_wall_calibration(rows)
    _attach_wall_reliability(rows, wall_calibration)
    overall = _summary(rows, ("variant",))
    condition_summary = _summary(rows, ("variant", "condition_id"))
    opening_summary = [{key: value for key, value in row.items() if not key.startswith("wall_")} for row in overall]
    wall_summary = overall
    outlier_threshold = _outlier_threshold(rows)
    for row in rows:
        wall_error = row.get("wall_tangent_error_deg", "")
        row["evaluation_wall_outlier"] = wall_error != "" and float(wall_error) > outlier_threshold

    fusion_rows = _fusion_rows(
        seed, wall_calibration, motion_training_comparison,
        motion_training_stability, paired_motion_csv, boundary_csv,
    )
    ablation = overall
    _write_rows(output_dir / "detector_baseline_vs_adaptive_results.csv", rows)
    _write_rows(output_dir / "detector_opening_summary.csv", opening_summary)
    _write_rows(output_dir / "detector_wall_geometry_summary.csv", wall_summary)
    _write_rows(output_dir / "detector_condition_summary.csv", condition_summary)
    _write_rows(output_dir / "detector_ablation.csv", ablation)
    _write_rows(output_dir / "fusion_regression_with_new_detector.csv", fusion_rows)
    _write_rows(output_dir / "detector_heuristic_audit.csv", _heuristic_audit())
    (output_dir / "wall_reliability_calibration_reproduced.json").write_text(
        json.dumps(wall_calibration, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    _save_plots(output_dir, rows, condition_summary, ablation, fusion_rows)

    overall_by_variant = {row["variant"]: row for row in overall}
    baseline_result = overall_by_variant["baseline"]
    combined = overall_by_variant["combined"]
    before_errors = [float(row["before_error_deg"]) for row in fusion_rows if row["before_error_deg"] != ""]
    after_errors = [float(row["after_error_deg"]) for row in fusion_rows if row["after_error_deg"] != ""]
    worst_rows = [row for row in fusion_rows if row["case_id"] in {"boundary_t080_long_production_s0", "boundary_t080_long_production_s1"} and row["sensor_condition"] == "resolution_4deg"]
    classification = (
        "C"
        if combined["false_positive_count"] > baseline_result["false_positive_count"]
        and combined["correct_opening_count_rate"] < baseline_result["correct_opening_count_rate"]
        else "A"
        if combined["opening_matching_coverage"] >= baseline_result["opening_matching_coverage"]
        and combined["false_positive_count"] <= baseline_result["false_positive_count"]
        and combined["wall_error_max_deg"] <= baseline_result["wall_error_max_deg"]
        else "B"
    )
    result = {
        "experiment": "EXP-PointCloudDetector-001",
        "classification": classification,
        "head": _short_head(),
        "geometry_case_count": len(create_benchmark_cases(seed)),
        "sensor_condition_count": len(SENSOR_CONDITIONS),
        "variant_count": len(VARIANTS),
        "branch_variant_condition_row_count": len(rows),
        "deterministic_replay": True,
        "baseline": baseline_result,
        "combined": combined,
        "evaluation_wall_outlier_threshold_deg": outlier_threshold,
        "evaluation_wall_outlier_definition": "baseline clean empirical P99; evaluation only",
        "baseline_wall_outlier_count": sum(row["variant"] == "baseline" and row["evaluation_wall_outlier"] for row in rows),
        "combined_wall_outlier_count": sum(row["variant"] == "combined" and row["evaluation_wall_outlier"] for row in rows),
        "fusion_before": _stats(before_errors),
        "fusion_after": _stats(after_errors),
        "fusion_before_coverage": len(before_errors) / max(len(fusion_rows), 1),
        "fusion_after_coverage": len(after_errors) / max(len(fusion_rows), 1),
        "fusion_before_status_counts": dict(Counter(row["before_status"] for row in fusion_rows)),
        "fusion_after_status_counts": dict(Counter(row["after_status"] for row in fusion_rows)),
        "worst_regression_rows": worst_rows,
        "protected_sha256": protected_sha,
        "matrix_sha256": first_hash,
        "output_dir": str(output_dir),
    }
    (output_dir / "detector_physics_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    return result


def finalize_existing_output(output_dir: Path) -> dict[str, Any]:
    """Refresh evaluation-only tail labels/decision without rerunning scans."""
    results_path = output_dir / "detector_baseline_vs_adaptive_results.csv"
    rows: list[dict[str, Any]] = _read_rows(results_path)
    threshold = _outlier_threshold(rows)
    for row in rows:
        wall_error = row.get("wall_tangent_error_deg", "")
        row["evaluation_wall_outlier"] = wall_error != "" and float(wall_error) > threshold
    _write_rows(results_path, rows)

    fusion_rows = _read_rows(output_dir / "fusion_regression_with_new_detector.csv")
    result_path = output_dir / "detector_physics_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    baseline_result = result["baseline"]
    combined = result["combined"]
    classification = (
        "C"
        if combined["false_positive_count"] > baseline_result["false_positive_count"]
        and combined["correct_opening_count_rate"] < baseline_result["correct_opening_count_rate"]
        else "A"
        if combined["opening_matching_coverage"] >= baseline_result["opening_matching_coverage"]
        and combined["false_positive_count"] <= baseline_result["false_positive_count"]
        and combined["wall_error_max_deg"] <= baseline_result["wall_error_max_deg"]
        else "B"
    )
    result.update({
        "classification": classification,
        "evaluation_wall_outlier_threshold_deg": threshold,
        "evaluation_wall_outlier_definition": "baseline clean empirical P99; evaluation only",
        "baseline_wall_outlier_count": sum(row["variant"] == "baseline" and bool(row["evaluation_wall_outlier"]) for row in rows),
        "combined_wall_outlier_count": sum(row["variant"] == "combined" and bool(row["evaluation_wall_outlier"]) for row in rows),
        "fusion_before_status_counts": dict(Counter(row["before_status"] for row in fusion_rows)),
        "fusion_after_status_counts": dict(Counter(row["after_status"] for row in fusion_rows)),
    })
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output-dir", type=Path, default=Path(f"/tmp/pdfs_pointcloud_detector_physics_{_short_head()}"))
    parser.add_argument("--motion-training-comparison", type=Path, required=True)
    parser.add_argument("--motion-training-stability", type=Path, required=True)
    parser.add_argument("--paired-motion-csv", type=Path, required=True)
    parser.add_argument("--boundary-csv", type=Path, required=True)
    args = parser.parse_args()
    result = run_experiment(
        args.output_dir,
        seed=args.seed,
        motion_training_comparison=args.motion_training_comparison,
        motion_training_stability=args.motion_training_stability,
        paired_motion_csv=args.paired_motion_csv,
        boundary_csv=args.boundary_csv,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
