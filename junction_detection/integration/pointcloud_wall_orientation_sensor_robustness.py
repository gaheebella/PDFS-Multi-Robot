"""Sensor-degradation robustness evaluation for the unchanged wall estimator.

This diagnostics-only benchmark reuses the repository's sensor-enhanced LiDAR
simulation, opening detector, 35-case geometry generator, and wall-parallel
tangent estimator.  Simulator geometry and GT are confined to post-hoc
matching/scoring.  No estimator, detector, threshold, or production code is
modified or tuned for any sensor condition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from junction_detection.integration.pointcloud_branch_orientation_generalization import (
    EPSILON,
    OrientationCase,
    _gt_opening,
    angular_error_deg,
    create_benchmark_cases,
    normalize_angle_deg,
    wall_segments,
)
from junction_detection.integration.pointcloud_wall_parallel_orientation import (
    NUMERICAL_ANGLE_TOL_DEG,
    STAGE_ONE_CASE_IDS,
    _best_line_in_component,
    _evaluate_case as evaluate_noiseless_reference,
    _opening_hit_components,
    estimate_wall_parallel_tangent,
    run_synthetic_sanity,
)
from junction_detection.pointcloud.pointcloud_junction_detector import detect_openings
from junction_detection.pointcloud.pointcloud_junction_detector_local_topology import (
    _match_openings,
)
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import (
    simulate_lidar_scan,
)


@dataclass(frozen=True)
class SensorCondition:
    """One independent, repository-defined sensor degradation condition."""

    condition_id: str
    degradation_type: str
    degradation_level: str
    noise_std_m: float = 0.0
    dropout_probability: float = 0.0
    occlusion_probability: float = 0.0
    visible_boundary_ratio: float = 1.0
    angle_step_deg: float = 1.0


# Values through visibility_0.70 are copied from the existing local-topology
# failure sweep.  Two/four-degree scans are exact 1/2 and 1/4 density versions
# of the native 1-degree scan; no estimator setting changes with resolution.
SENSOR_CONDITIONS = (
    SensorCondition("clean", "clean", "native"),
    SensorCondition("noise_0.03", "range_noise", "mild", noise_std_m=0.03),
    SensorCondition("noise_0.08", "range_noise", "moderate", noise_std_m=0.08),
    SensorCondition("dropout_0.05", "dropout", "mild", dropout_probability=0.05),
    SensorCondition("dropout_0.15", "dropout", "moderate", dropout_probability=0.15),
    SensorCondition("occlusion_0.40", "occlusion", "mild", occlusion_probability=0.40),
    SensorCondition("occlusion_0.80", "occlusion", "moderate", occlusion_probability=0.80),
    SensorCondition("visibility_0.90", "partial_visibility", "mild", visible_boundary_ratio=0.90),
    SensorCondition("visibility_0.70", "partial_visibility", "moderate", visible_boundary_ratio=0.70),
    SensorCondition("resolution_2deg", "angular_resolution", "moderate", angle_step_deg=2.0),
    SensorCondition("resolution_4deg", "angular_resolution", "coarse", angle_step_deg=4.0),
)


def _short_head() -> str:
    """Return current short Git identity without changing repository state."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _candidate_diagnostics(
    angles_deg: np.ndarray,
    ranges: np.ndarray,
    max_range_m: float,
    opening: Mapping[str, float],
    estimate_mode: str,
) -> dict[str, Any]:
    """Observe the unchanged estimator's raw boundary candidates.

    The same private extraction helpers called by the estimator are reused;
    this function records support only and does not select or refit anything.
    """
    components = _opening_hit_components(
        angles_deg, ranges, max_range_m, opening
    )
    boundary_components = components[:1]
    if len(components) > 1:
        boundary_components.append(components[-1])
    candidates = [_best_line_in_component(component) for component in boundary_components]
    candidates = [candidate for candidate in candidates if candidate is not None]
    right = candidates[0] if candidates else None
    left = candidates[-1] if len(candidates) > 1 else None

    if not candidates or estimate_mode.startswith("unavailable"):
        selected_span = ""
    elif estimate_mode == "two_wall_parallel":
        selected_span = float(sum(candidate.span_m for candidate in candidates))
    elif estimate_mode == "one_wall_dominant_span":
        selected_span = float(max(candidate.span_m for candidate in candidates))
    else:
        selected_span = float(candidates[0].span_m)
    return {
        "left_fitted_point_count": 0 if left is None else left.point_count,
        "right_fitted_point_count": 0 if right is None else right.point_count,
        "left_wall_span_m": "" if left is None else left.span_m,
        "right_wall_span_m": "" if right is None else right.span_m,
        "left_line_residual_m": "" if left is None else left.residual_m,
        "right_line_residual_m": "" if right is None else right.residual_m,
        "selected_wall_span_m": selected_span,
    }


def _evaluate_case_condition(
    case: OrientationCase,
    condition: SensorCondition,
) -> list[dict[str, Any]]:
    """Evaluate one geometry under one independent sensor condition."""
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
    angles_deg, ranges = scan.detector_input()
    detected = detect_openings(angles_deg, ranges)
    ground_truth = [_gt_opening(case, branch) for branch in case.branches]
    match_by_gt = {
        gt_index: (detected_index, iou)
        for gt_index, detected_index, iou in _match_openings(ground_truth, detected)
    }
    offset = case.anchor_offset
    offset_magnitude = float(np.linalg.norm(offset))
    rows: list[dict[str, Any]] = []
    for branch_index, branch in enumerate(case.branches):
        gt_tangent = normalize_angle_deg(branch.angle_deg - case.anchor_yaw_deg)
        relative_offset_direction: float | str = ""
        if offset_magnitude > EPSILON:
            offset_world = math.degrees(math.atan2(offset[1], offset[0]))
            relative_offset_direction = normalize_angle_deg(offset_world - branch.angle_deg)
        base = {
            "case_id": case.case_id,
            "seed": case.seed,
            "condition_id": condition.condition_id,
            "degradation_type": condition.degradation_type,
            "degradation_level": condition.degradation_level,
            "noise_std_m": condition.noise_std_m,
            "dropout_probability": condition.dropout_probability,
            "occlusion_probability": condition.occlusion_probability,
            "visible_boundary_ratio": condition.visible_boundary_ratio,
            "angle_step_deg": condition.angle_step_deg,
            "topology": case.topology,
            "branch_id": branch.branch_id,
            "branch_angle_world_deg": branch.angle_deg,
            "gt_tangent_deg": gt_tangent,
            "corridor_width": branch.width,
            "width_group": case.width_group,
            "branch_length": branch.length,
            "anchor_offset_magnitude": offset_magnitude,
            "anchor_offset_normalized": offset_magnitude / case.central_radius,
            "anchor_offset_group": case.offset_group,
            "anchor_offset_direction_group": case.offset_direction_group,
            "anchor_offset_direction_relative_gt_deg": relative_offset_direction,
            "global_rotation_deg": case.global_rotation_deg,
            "mouth_geometry": case.mouth_geometry,
            "opening_matched": branch_index in match_by_gt,
            "detected_opening_count": len(detected),
            "wall_estimator_available": False,
            "estimate_mode": "opening_unmatched",
        }
        if branch_index not in match_by_gt:
            rows.append(base)
            continue

        detected_index, iou = match_by_gt[branch_index]
        opening = detected[detected_index]
        center_error = angular_error_deg(float(opening["center_angle"]), gt_tangent)
        estimate = estimate_wall_parallel_tangent(
            angles_deg, ranges, scan.max_range_m, opening
        )
        available = estimate.tangent_deg is not None
        diagnostics = _candidate_diagnostics(
            angles_deg, ranges, scan.max_range_m, opening, estimate.estimate_mode
        )
        wall_error = "" if not available else angular_error_deg(
            float(estimate.tangent_deg), gt_tangent
        )
        rows.append({
            **base,
            **diagnostics,
            "opening_confidence": opening["confidence"],
            "opening_start_angle": opening["start_angle"],
            "opening_end_angle": opening["end_angle"],
            "opening_center_angle": opening["center_angle"],
            "opening_center_error_deg": center_error,
            "match_iou": iou,
            "wall_estimator_available": available,
            "wall_tangent_deg": "" if not available else estimate.tangent_deg,
            "wall_tangent_error_deg": wall_error,
            "wall_minus_center_error_deg": "" if not available else float(wall_error) - center_error,
            "estimate_mode": estimate.estimate_mode,
            "left_wall_orientation_deg": "" if estimate.left_orientation_deg is None else estimate.left_orientation_deg,
            "right_wall_orientation_deg": "" if estimate.right_orientation_deg is None else estimate.right_orientation_deg,
            "usable_wall_sides": estimate.usable_wall_sides,
            "selected_wall_point_count": estimate.fitted_point_count,
            "line_fit_residual_m": "" if estimate.line_fit_residual_m is None else estimate.line_fit_residual_m,
            "left_right_raw_disagreement_deg": "" if estimate.wall_disagreement_deg is None else estimate.wall_disagreement_deg,
        })
    return rows


def _run_matrix(
    cases: Sequence[OrientationCase],
    conditions: Sequence[SensorCondition],
    *,
    announce: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(cases) * len(conditions)
    run_index = 0
    for condition in conditions:
        for case in cases:
            run_index += 1
            if announce:
                print(
                    f"[wall sensor] {run_index}/{total} "
                    f"{condition.condition_id}:{case.case_id}",
                    flush=True,
                )
            rows.extend(_evaluate_case_condition(case, condition))
    return rows


def _errors(rows: Sequence[Mapping[str, Any]], estimator: str) -> list[float]:
    key = "opening_center_error_deg" if estimator == "opening_center" else "wall_tangent_error_deg"
    return [float(row[key]) for row in rows if row.get(key, "") != ""]


def _statistics(rows: Sequence[Mapping[str, Any]], estimator: str) -> dict[str, Any]:
    """Return separate opening and conditional-wall coverage plus errors."""
    total = len(rows)
    matched = [row for row in rows if bool(row.get("opening_matched", False))]
    available = [row for row in matched if bool(row.get("wall_estimator_available", False))]
    errors = _errors(rows, estimator)
    return {
        "branch_count": total,
        "opening_matched_count": len(matched),
        "opening_detector_coverage": len(matched) / max(total, 1),
        "wall_available_count": len(available),
        "conditional_wall_coverage": len(available) / max(len(matched), 1),
        "end_to_end_wall_coverage": len(available) / max(total, 1),
        "mean_error_deg": "" if not errors else float(np.mean(errors)),
        "median_error_deg": "" if not errors else float(np.median(errors)),
        "p90_error_deg": "" if not errors else float(np.percentile(errors, 90)),
        "max_error_deg": "" if not errors else float(np.max(errors)),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks, including ties, without SciPy."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _correlations(
    rows: Sequence[Mapping[str, Any]], x_key: str, y_key: str = "wall_tangent_error_deg"
) -> tuple[float, float, int]:
    selected = [row for row in rows if row.get(x_key, "") != "" and row.get(y_key, "") != ""]
    if len(selected) < 2:
        return float("nan"), float("nan"), len(selected)
    x = np.asarray([float(row[x_key]) for row in selected])
    y = np.asarray([float(row[y_key]) for row in selected])
    if np.std(x) <= EPSILON or np.std(y) <= EPSILON:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(x, y)[0, 1])
    rx, ry = _rankdata(x), _rankdata(y)
    spearman = float("nan") if np.std(rx) <= EPSILON or np.std(ry) <= EPSILON else float(np.corrcoef(rx, ry)[0, 1])
    return pearson, spearman, len(selected)


def _clean_reproduced(
    clean_rows: Sequence[Mapping[str, Any]], cases: Sequence[OrientationCase]
) -> bool:
    """Compare sensor-enhanced clean output with the prior noiseless evaluator."""
    reference: dict[tuple[str, str], Mapping[str, Any]] = {}
    for case in cases:
        case_rows, _ = evaluate_noiseless_reference(case)
        for row in case_rows:
            reference[(str(row["case_id"]), str(row["branch_id"]))] = row
    for row in clean_rows:
        other = reference[(str(row["case_id"]), str(row["branch_id"]))]
        if bool(row["opening_matched"]) != bool(other["opening_matched"]):
            return False
        if bool(row["wall_estimator_available"]) != bool(other["wall_estimator_available"]):
            return False
        for key in ("opening_center_error_deg", "wall_tangent_error_deg"):
            first, second = row.get(key, ""), other.get(key, "")
            if first == "" or second == "":
                if first != second:
                    return False
            elif not math.isclose(float(first), float(second), abs_tol=NUMERICAL_ANGLE_TOL_DEG):
                return False
    return True


def _stage_one_gate(
    rows: Sequence[Mapping[str, Any]], clean_reproduced: bool
) -> tuple[bool, dict[str, bool]]:
    """Gate Stage 2 using comparisons, not a tuned accuracy target."""
    mild_ids = {condition.condition_id for condition in SENSOR_CONDITIONS if condition.degradation_level == "mild"}
    mild = [row for row in rows if row["condition_id"] in mild_ids]
    mild_by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in mild:
        mild_by_condition[str(row["condition_id"])].append(row)

    # A >1-native-bin regression on a Branch where wall is already worse than
    # its paired center estimate is treated as a catastrophic relative outlier.
    # The one-degree value is the native sensor resolution, not a tuned error.
    relative_outliers = [
        row for row in mild
        if row.get("wall_tangent_error_deg", "") != ""
        and float(row["wall_tangent_error_deg"]) > float(row["opening_center_error_deg"])
        and float(row["wall_tangent_error_deg"]) > 1.0
    ]
    family_available = True
    for condition_id in mild_ids:
        selected = mild_by_condition[condition_id]
        for family in {str(row["topology"]) for row in selected}:
            family_rows = [row for row in selected if str(row["topology"]) == family]
            family_available &= any(bool(row.get("wall_estimator_available")) for row in family_rows)
    checks = {
        "clean_baseline_reproduced": clean_reproduced,
        "mild_mean_advantage_each_condition": all(
            float(_statistics(selected, "wall")["mean_error_deg"])
            < float(_statistics(selected, "opening_center")["mean_error_deg"])
            for selected in mild_by_condition.values()
        ),
        "mild_general_geometry_available": family_available,
        "mild_no_relative_catastrophic_outlier": not relative_outliers,
        "failure_stage_accounting_complete": all(
            bool(row.get("opening_matched"))
            or row.get("estimate_mode") == "opening_unmatched"
            for row in rows
        ),
    }
    return all(checks.values()), checks


def _classify(rows: Sequence[Mapping[str, Any]]) -> str:
    """Classify robustness using condition-level relative behavior."""
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition_id"])].append(row)
    nonclean = [selected for key, selected in by_condition.items() if key != "clean"]
    wall_advantage = [
        float(_statistics(selected, "wall")["mean_error_deg"])
        < float(_statistics(selected, "opening_center")["mean_error_deg"])
        for selected in nonclean
        if _statistics(selected, "wall")["mean_error_deg"] != ""
    ]
    availability_loss = any(
        _statistics(selected, "wall")["conditional_wall_coverage"]
        < _statistics(by_condition["clean"], "wall")["conditional_wall_coverage"]
        for selected in nonclean
    )
    if wall_advantage and all(wall_advantage) and not availability_loss:
        return "A"
    mild = [selected for selected in nonclean if selected[0]["degradation_level"] == "mild"]
    mild_advantage = all(
        float(_statistics(selected, "wall")["mean_error_deg"])
        < float(_statistics(selected, "opening_center")["mean_error_deg"])
        for selected in mild
    )
    return "B" if mild_advantage else "C"


def _summary_by_degradation(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["condition_id"])].append(row)
    output: list[dict[str, Any]] = []
    condition_order = {condition.condition_id: index for index, condition in enumerate(SENSOR_CONDITIONS)}
    for condition_id, selected in sorted(groups.items(), key=lambda item: condition_order[item[0]]):
        for estimator in ("opening_center", "wall"):
            output.append({
                "condition_id": condition_id,
                "degradation_type": selected[0]["degradation_type"],
                "degradation_level": selected[0]["degradation_level"],
                "estimator": estimator,
                **_statistics(selected, estimator),
            })
    return output


def _summary_by_mode(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["condition_id"]), str(row["estimate_mode"]))].append(row)
    output = []
    for (condition_id, mode), selected in sorted(groups.items()):
        errors = _errors(selected, "wall")
        output.append({
            "condition_id": condition_id,
            "degradation_type": selected[0]["degradation_type"],
            "estimate_mode": mode,
            "count": len(selected),
            "mean_error_deg": "" if not errors else float(np.mean(errors)),
            "p90_error_deg": "" if not errors else float(np.percentile(errors, 90)),
            "max_error_deg": "" if not errors else float(np.max(errors)),
        })
    return output


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("case_id,branch_id,condition_id,failure_stage\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_plots(directory: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    by_degradation = _summary_by_degradation(rows)
    conditions = [condition.condition_id for condition in SENSOR_CONDITIONS if any(row["condition_id"] == condition.condition_id for row in rows)]
    center = {row["condition_id"]: row for row in by_degradation if row["estimator"] == "opening_center"}
    wall = {row["condition_id"]: row for row in by_degradation if row["estimator"] == "wall"}
    x = np.arange(len(conditions))

    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    axis.plot(x, [float(center[key]["mean_error_deg"]) for key in conditions], "o-", label="opening center mean")
    axis.plot(x, [float(wall[key]["mean_error_deg"]) for key in conditions], "o-", label="wall mean")
    axis.plot(x, [float(wall[key]["p90_error_deg"]) for key in conditions], "s--", label="wall P90")
    axis.set_xticks(x, conditions, rotation=45, ha="right")
    axis.set(ylabel="orientation error [deg]", title="Wall error under independent sensor degradation")
    axis.legend()
    figure.savefig(directory / "wall_error_vs_sensor_degradation.png", dpi=160)
    plt.close(figure)

    noise_rows = [row for row in rows if row["degradation_type"] in {"clean", "range_noise"} and row.get("wall_tangent_error_deg", "") != ""]
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for condition_id in ("clean", "noise_0.03", "noise_0.08"):
        selected = [row for row in noise_rows if row["condition_id"] == condition_id]
        axis.scatter([float(row["opening_center_error_deg"]) for row in selected], [float(row["wall_tangent_error_deg"]) for row in selected], alpha=0.55, label=condition_id)
    limit = max([1.0] + [float(row["opening_center_error_deg"]) for row in noise_rows] + [float(row["wall_tangent_error_deg"]) for row in noise_rows])
    axis.plot([0, limit], [0, limit], "--", color="0.5")
    axis.set(xlabel="opening-center error [deg]", ylabel="wall error [deg]", title="Opening center vs wall under range noise")
    axis.legend()
    figure.savefig(directory / "opening_center_vs_wall_under_noise.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    axis.plot(x, [float(wall[key]["opening_detector_coverage"]) for key in conditions], "o-", label="opening coverage")
    axis.plot(x, [float(wall[key]["conditional_wall_coverage"]) for key in conditions], "s-", label="wall | matched opening")
    axis.plot(x, [float(wall[key]["end_to_end_wall_coverage"]) for key in conditions], "^-", label="end-to-end wall")
    axis.set_xticks(x, conditions, rotation=45, ha="right")
    axis.set(ylim=(-0.02, 1.02), ylabel="coverage", title="Opening and conditional wall coverage")
    axis.legend()
    figure.savefig(directory / "wall_coverage_vs_sensor_degradation.png", dpi=160)
    plt.close(figure)

    available = [row for row in rows if row.get("wall_tangent_error_deg", "") != ""]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.scatter([float(row["selected_wall_point_count"]) for row in available], [float(row["wall_tangent_error_deg"]) for row in available], c=[float(row["visible_boundary_ratio"]) for row in available], cmap="viridis", alpha=0.55)
    axis.set(xlabel="selected fitted point count", ylabel="wall error [deg]", title="Point support vs wall error")
    figure.savefig(directory / "wall_error_vs_point_count.png", dpi=160)
    plt.close(figure)

    residual_rows = [row for row in available if row.get("line_fit_residual_m", "") != ""]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.scatter([float(row["line_fit_residual_m"]) for row in residual_rows], [float(row["wall_tangent_error_deg"]) for row in residual_rows], c=[float(row["noise_std_m"]) for row in residual_rows], cmap="plasma", alpha=0.55)
    axis.set(xlabel="TLS residual [m]", ylabel="wall error [deg]", title="Line-fit residual vs wall error")
    figure.savefig(directory / "wall_error_vs_fit_residual.png", dpi=160)
    plt.close(figure)


def run_benchmark(output_dir: Path, seed: int) -> dict[str, Any]:
    """Run deterministic Stage 1 and expand only if its strict gate passes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_cases = create_benchmark_cases(seed)
    cases_by_id = {case.case_id: case for case in all_cases}
    stage_one_cases = [cases_by_id[case_id] for case_id in STAGE_ONE_CASE_IDS]
    stage_one_rows = _run_matrix(stage_one_cases, SENSOR_CONDITIONS, announce=True)
    replay_rows = _run_matrix(stage_one_cases, SENSOR_CONDITIONS, announce=False)
    first_hash = hashlib.sha256(json.dumps(stage_one_rows, sort_keys=True).encode()).hexdigest()
    replay_hash = hashlib.sha256(json.dumps(replay_rows, sort_keys=True).encode()).hexdigest()
    if first_hash != replay_hash:
        raise AssertionError("sensor robustness deterministic replay mismatch")

    clean_rows = [row for row in stage_one_rows if row["condition_id"] == "clean"]
    clean_reproduced = _clean_reproduced(clean_rows, stage_one_cases)
    stage_one_passed, stage_checks = _stage_one_gate(stage_one_rows, clean_reproduced)
    if stage_one_passed:
        print("[wall sensor] Stage 1 PASS; expanding to 35 cases", flush=True)
        rows = _run_matrix(all_cases, SENSOR_CONDITIONS, announce=True)
        stage = "stage2_full"
    else:
        print("[wall sensor] Stage 1 FAIL; stopping before full benchmark", flush=True)
        rows = stage_one_rows
        stage = "stage1_only"

    classification = _classify(rows)
    failures: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("opening_matched"):
            failure_stage = "opening_detection_failure"
        elif not row.get("wall_estimator_available"):
            failure_stage = "wall_extraction_failure"
        elif float(row["wall_tangent_error_deg"]) > float(row["opening_center_error_deg"]):
            failure_stage = "line_orientation_worse_than_center"
        else:
            continue
        failures.append({
            "case_id": row["case_id"],
            "branch_id": row["branch_id"],
            "seed": row["seed"],
            "condition_id": row["condition_id"],
            "failure_stage": failure_stage,
            "estimate_mode": row["estimate_mode"],
            "opening_center_error_deg": row.get("opening_center_error_deg", ""),
            "wall_tangent_error_deg": row.get("wall_tangent_error_deg", ""),
            "selected_wall_point_count": row.get("selected_wall_point_count", ""),
            "line_fit_residual_m": row.get("line_fit_residual_m", ""),
            "left_right_raw_disagreement_deg": row.get("left_right_raw_disagreement_deg", ""),
        })

    by_degradation = _summary_by_degradation(rows)
    by_mode = _summary_by_mode(rows)
    summary: list[dict[str, Any]] = [
        {"metric": "stage", "value": stage},
        {"metric": "classification", "value": classification},
        {"metric": "geometry_case_count", "value": len({row["case_id"] for row in rows})},
        {"metric": "sensor_condition_count", "value": len({row["condition_id"] for row in rows})},
        {"metric": "branch_condition_row_count", "value": len(rows)},
        {"metric": "deterministic_replay_sha256", "value": first_hash},
        {"metric": "clean_baseline_reproduced", "value": clean_reproduced},
    ]
    for check, passed in stage_checks.items():
        summary.append({"metric": f"stage1_{check}", "value": passed})
    for x_key in ("selected_wall_point_count", "line_fit_residual_m", "left_right_raw_disagreement_deg", "selected_wall_span_m"):
        pearson, spearman, count = _correlations(rows, x_key)
        summary.extend([
            {"metric": f"{x_key}_vs_error_pearson", "value": pearson},
            {"metric": f"{x_key}_vs_error_spearman", "value": spearman},
            {"metric": f"{x_key}_vs_error_sample_count", "value": count},
        ])

    _write_rows(output_dir / "wall_sensor_robustness_results.csv", rows)
    _write_rows(output_dir / "wall_sensor_robustness_summary.csv", summary)
    _write_rows(output_dir / "wall_sensor_robustness_by_degradation.csv", by_degradation)
    _write_rows(output_dir / "wall_sensor_robustness_by_mode.csv", by_mode)
    _write_rows(output_dir / "wall_sensor_failure_cases.csv", failures)
    _save_plots(output_dir, rows)
    result = {
        "stage": stage,
        "classification": classification,
        "stage_one_case_count": len(stage_one_cases),
        "executed_case_count": len({row["case_id"] for row in rows}),
        "sensor_condition_count": len(SENSOR_CONDITIONS),
        "branch_condition_row_count": len(rows),
        "clean_baseline_reproduced": clean_reproduced,
        "deterministic_replay": first_hash == replay_hash,
        "stage_one_checks": stage_checks,
        "output_dir": str(output_dir),
    }
    (output_dir / "wall_sensor_robustness_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"/tmp/pdfs_wall_sensor_robustness_{_short_head()}"),
    )
    parser.add_argument("--sanity-test", action="store_true")
    args = parser.parse_args()
    if args.sanity_test:
        run_synthetic_sanity()
        print("inherited wall TLS translation/rotation sanity: PASS")
    result = run_benchmark(args.output_dir, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
