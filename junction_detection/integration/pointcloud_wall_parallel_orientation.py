"""Evaluation-only wall-parallel Point Cloud Branch orientation benchmark.

The estimator consumes only an Anchor-local angle/range scan, its measured
maximum range, and one opening returned by the existing ``detect_openings``.
Simulator geometry and GT tangents are used only after estimation for scoring.

This is intentionally an ideal, noiseless geometry experiment.  It does not
modify the detector, production controller, or Stable-motion estimator, and it
does not implement geometry/motion fusion.
"""

from __future__ import annotations

import argparse
import csv
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
    signed_error_deg,
    wall_segments,
)
from junction_detection.pointcloud.pointcloud_junction_detector import (
    detect_openings,
    simulate_lidar_scan,
)
from junction_detection.pointcloud.pointcloud_junction_detector_local_topology import (
    _match_openings,
)


NUMERICAL_ANGLE_TOL_DEG = math.degrees(math.sqrt(np.finfo(float).eps))


@dataclass(frozen=True)
class LineCandidate:
    """A straight, contiguous Point Cloud surface candidate."""

    orientation_deg: float
    point_count: int
    span_m: float
    residual_m: float
    points: np.ndarray


@dataclass(frozen=True)
class WallEstimate:
    """Wall-parallel tangent estimate and observable diagnostics."""

    tangent_deg: float | None
    left_orientation_deg: float | None
    right_orientation_deg: float | None
    usable_wall_sides: int
    fitted_point_count: int
    line_fit_residual_m: float | None
    wall_disagreement_deg: float | None
    estimate_mode: str
    selected_points: np.ndarray


def _short_head() -> str:
    """Return the current short Git identity without changing repository state."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _axial_difference_deg(first: float, second: float) -> float:
    """Smallest angular difference between unoriented lines (modulo 180°)."""
    return abs(float((first - second + 90.0) % 180.0 - 90.0))


def _signed_from_opening(line_orientation_deg: float, opening_center_deg: float) -> float:
    """Resolve a line's 180° ambiguity using only the observed opening direction."""
    first = normalize_angle_deg(line_orientation_deg)
    second = normalize_angle_deg(line_orientation_deg + 180.0)
    if angular_error_deg(first, opening_center_deg) <= angular_error_deg(
        second, opening_center_deg
    ):
        return first
    return second


def fit_tls_line(points: Sequence[Sequence[float]]) -> LineCandidate:
    """Fit an arbitrary 2D line by total least squares/PCA.

    The reported residual is RMS orthogonal point-to-line distance.  No world
    axis or expected Branch direction enters the fit.
    """
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 2:
        raise ValueError("a line fit needs at least two finite 2D points")
    if not np.all(np.isfinite(array)):
        raise ValueError("line-fit points must be finite")
    centered = array - np.mean(array, axis=0)
    _, singular_values, vectors = np.linalg.svd(centered, full_matrices=False)
    if not singular_values.size or singular_values[0] <= np.finfo(float).eps:
        raise ValueError("line-fit points must have non-zero spatial extent")
    direction = vectors[0]
    normal = np.asarray([-direction[1], direction[0]])
    distances = centered @ normal
    orientation = math.degrees(math.atan2(direction[1], direction[0])) % 180.0
    return LineCandidate(
        orientation_deg=float(orientation),
        point_count=len(array),
        span_m=float(np.linalg.norm(array[-1] - array[0])),
        residual_m=float(np.sqrt(np.mean(distances**2))),
        points=array.copy(),
    )


def _straight_subruns(points: np.ndarray) -> list[np.ndarray]:
    """Split an ordered hit component at non-collinear surface transitions.

    In this noiseless experiment, ray intersections with one straight wall are
    collinear to floating-point precision.  Therefore the only tolerance here
    is derived from machine precision, not fitted to benchmark performance.
    A polygonal central arc and a straight corridor wall naturally separate
    when their consecutive chord orientations change.
    """
    if len(points) < 2:
        return []
    edges = np.diff(points, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    valid = lengths > np.finfo(float).eps
    runs: list[np.ndarray] = []
    start_edge = 0
    orientations = np.degrees(np.arctan2(edges[:, 1], edges[:, 0])) % 180.0
    for edge_index in range(1, len(edges)):
        split = (
            not valid[edge_index - 1]
            or not valid[edge_index]
            or _axial_difference_deg(
                float(orientations[edge_index - 1]),
                float(orientations[edge_index]),
            )
            > NUMERICAL_ANGLE_TOL_DEG
        )
        if split:
            if valid[start_edge:edge_index].all():
                runs.append(points[start_edge : edge_index + 1])
            start_edge = edge_index
    if valid[start_edge:].all():
        runs.append(points[start_edge : len(edges) + 1])
    return [run for run in runs if len(run) >= 2]


def _opening_hit_components(
    angles_deg: np.ndarray,
    ranges: np.ndarray,
    max_range_m: float,
    opening: Mapping[str, float],
) -> list[np.ndarray]:
    """Return circularly ordered physical-hit components inside an opening.

    Component continuity is defined by the scan's own angular resolution.  No
    fixed boundary window or fixed number of rays is introduced.
    """
    angles = np.asarray(angles_deg, dtype=float)
    values = np.asarray(ranges, dtype=float)
    if angles.shape != values.shape or angles.ndim != 1 or len(angles) < 2:
        raise ValueError("angles_deg and ranges must be equal non-trivial 1D arrays")
    start = float(opening["start_angle"])
    width = float(opening["width_deg"])
    relative = (angles - start) % 360.0
    hit = values < float(max_range_m) - np.finfo(float).eps * max(1.0, max_range_m)
    selected = np.flatnonzero((relative <= width + EPSILON) & hit)
    if not selected.size:
        return []
    order = selected[np.argsort(relative[selected])]
    sorted_angles = np.sort(angles % 360.0)
    angular_steps = np.diff(np.r_[sorted_angles, sorted_angles[0] + 360.0])
    positive_steps = angular_steps[angular_steps > NUMERICAL_ANGLE_TOL_DEG]
    scan_step = float(np.median(positive_steps))
    tolerance = NUMERICAL_ANGLE_TOL_DEG * max(1.0, scan_step)
    groups: list[list[int]] = [[int(order[0])]]
    for previous, current in zip(order[:-1], order[1:]):
        separation = float(relative[current] - relative[previous])
        if abs(separation - scan_step) <= tolerance:
            groups[-1].append(int(current))
        else:
            groups.append([int(current)])
    theta = np.deg2rad(angles)
    points = np.column_stack((values * np.cos(theta), values * np.sin(theta)))
    return [points[group] for group in groups if len(group) >= 2]


def _best_line_in_component(points: np.ndarray) -> LineCandidate | None:
    """Return the greatest observed straight span in one hit component."""
    candidates: list[LineCandidate] = []
    for run in _straight_subruns(points):
        try:
            candidates.append(fit_tls_line(run))
        except ValueError:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.span_m)


def estimate_wall_parallel_tangent(
    angles_deg: Sequence[float],
    ranges: Sequence[float],
    max_range_m: float,
    opening: Mapping[str, float],
) -> WallEstimate:
    """Estimate a Branch tangent from opening-adjacent straight wall returns.

    At most the first and last hit components in the detected opening sector
    are side-wall candidates.  Parallel candidates are combined axially.  If
    they disagree, the candidate with greater physically observed line span is
    retained; an exact span tie is reported unavailable rather than guessed.
    The opening center resolves sign only and never changes the fitted axis.
    """
    components = _opening_hit_components(
        np.asarray(angles_deg, dtype=float),
        np.asarray(ranges, dtype=float),
        float(max_range_m),
        opening,
    )
    boundary_components = components[:1]
    if len(components) > 1:
        boundary_components.append(components[-1])
    candidates = [_best_line_in_component(component) for component in boundary_components]
    candidates = [candidate for candidate in candidates if candidate is not None]
    right = candidates[0] if candidates else None
    left = candidates[-1] if len(candidates) > 1 else None
    if not candidates:
        return WallEstimate(None, None, None, 0, 0, None, None, "unavailable", np.empty((0, 2)))

    disagreement = None
    selected: list[LineCandidate]
    mode: str
    if len(candidates) == 1:
        selected = candidates
        mode = "one_wall_observed"
    else:
        disagreement = _axial_difference_deg(
            candidates[0].orientation_deg, candidates[1].orientation_deg
        )
        if disagreement <= NUMERICAL_ANGLE_TOL_DEG:
            selected = candidates
            mode = "two_wall_parallel"
        else:
            span_difference = abs(candidates[0].span_m - candidates[1].span_m)
            span_tolerance = math.sqrt(np.finfo(float).eps) * max(
                candidates[0].span_m, candidates[1].span_m, 1.0
            )
            if span_difference <= span_tolerance:
                return WallEstimate(
                    None,
                    left.orientation_deg if left else None,
                    right.orientation_deg if right else None,
                    0,
                    0,
                    None,
                    disagreement,
                    "unavailable_ambiguous_surfaces",
                    np.empty((0, 2)),
                )
            selected = [max(candidates, key=lambda candidate: candidate.span_m)]
            mode = "one_wall_dominant_span"

    # Axial circular mean: double angles to remove the 180° ambiguity.
    doubled = np.deg2rad([2.0 * item.orientation_deg for item in selected])
    weights = np.asarray([item.span_m for item in selected])
    axis_angle = 0.5 * math.degrees(
        math.atan2(float(np.sum(weights * np.sin(doubled))), float(np.sum(weights * np.cos(doubled))))
    ) % 180.0
    tangent = _signed_from_opening(axis_angle, float(opening["center_angle"]))
    point_count = sum(item.point_count for item in selected)
    residual = math.sqrt(
        sum(item.point_count * item.residual_m**2 for item in selected) / point_count
    )
    selected_points = np.vstack([item.points for item in selected])
    return WallEstimate(
        tangent_deg=tangent,
        left_orientation_deg=None if left is None else left.orientation_deg,
        right_orientation_deg=None if right is None else right.orientation_deg,
        usable_wall_sides=len(selected),
        fitted_point_count=point_count,
        line_fit_residual_m=residual,
        wall_disagreement_deg=disagreement,
        estimate_mode=mode,
        selected_points=selected_points,
    )


def _evaluate_case(case: OrientationCase) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run ideal LiDAR/detection/estimation, then score against simulator GT."""
    walls = wall_segments(case)
    maximum_extent = case.central_radius + max(branch.length for branch in case.branches)
    scan = simulate_lidar_scan(
        walls,
        case.anchor_xy,
        anchor_yaw_deg=case.anchor_yaw_deg,
        angle_step_deg=1.0,
        max_range_m=maximum_extent * 1.30,
        noise_std_m=0.0,
        dropout_probability=0.0,
        seed=case.seed,
    )
    angles_deg, ranges = scan.detector_input()
    detected = detect_openings(angles_deg, ranges)
    ground_truth = [_gt_opening(case, branch) for branch in case.branches]
    matches = _match_openings(ground_truth, detected)
    match_by_gt = {gt_index: (det_index, iou) for gt_index, det_index, iou in matches}
    offset = case.anchor_offset
    offset_magnitude = float(np.linalg.norm(offset))
    rows: list[dict[str, Any]] = []
    estimates: dict[str, WallEstimate] = {}
    for branch_index, branch in enumerate(case.branches):
        gt_tangent = normalize_angle_deg(branch.angle_deg - case.anchor_yaw_deg)
        match = match_by_gt.get(branch_index)
        relative_offset = ""
        if offset_magnitude > EPSILON:
            offset_angle = math.degrees(math.atan2(offset[1], offset[0]))
            relative_offset = normalize_angle_deg(offset_angle - branch.angle_deg)
        base = {
            "case_id": case.case_id,
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
            "anchor_offset_direction_relative_gt_deg": relative_offset,
            "global_rotation_deg": case.global_rotation_deg,
            "mouth_geometry": case.mouth_geometry,
            "opening_matched": match is not None,
            "wall_estimator_available": False,
        }
        if match is None:
            rows.append({**base, "estimate_mode": "opening_unmatched"})
            continue
        detected_index, iou = match
        opening = detected[detected_index]
        estimate = estimate_wall_parallel_tangent(
            angles_deg, ranges, scan.max_range_m, opening
        )
        estimates[branch.branch_id] = estimate
        start_delta = signed_error_deg(float(opening["start_angle"]), gt_tangent)
        end_delta = signed_error_deg(float(opening["end_angle"]), gt_tangent)
        wall_available = estimate.tangent_deg is not None
        rows.append({
            **base,
            "wall_estimator_available": wall_available,
            "opening_start_angle": opening["start_angle"],
            "opening_end_angle": opening["end_angle"],
            "opening_center_angle": opening["center_angle"],
            "opening_center_error_deg": angular_error_deg(float(opening["center_angle"]), gt_tangent),
            "wall_tangent_deg": "" if not wall_available else estimate.tangent_deg,
            "wall_tangent_error_deg": "" if not wall_available else angular_error_deg(float(estimate.tangent_deg), gt_tangent),
            "left_wall_orientation_deg": "" if estimate.left_orientation_deg is None else estimate.left_orientation_deg,
            "right_wall_orientation_deg": "" if estimate.right_orientation_deg is None else estimate.right_orientation_deg,
            "usable_wall_sides": estimate.usable_wall_sides,
            "left_right_disagreement_deg": "" if estimate.wall_disagreement_deg is None else estimate.wall_disagreement_deg,
            "fitted_point_count": estimate.fitted_point_count,
            "line_fit_residual_m": "" if estimate.line_fit_residual_m is None else estimate.line_fit_residual_m,
            "estimate_mode": estimate.estimate_mode,
            "opening_confidence": opening["confidence"],
            "boundary_asymmetry_deg": abs(abs(end_delta) - abs(start_delta)),
            "match_iou": iou,
        })
    return rows, {
        "case": case,
        "walls": walls,
        "scan": scan,
        "detected": detected,
        "estimates": estimates,
    }


def _metric(rows: Sequence[Mapping[str, Any]], estimator: str) -> dict[str, Any]:
    """Compute coverage and error statistics for one estimator."""
    key = "opening_center_error_deg" if estimator == "opening_center" else "wall_tangent_error_deg"
    available_key = "opening_matched" if estimator == "opening_center" else "wall_estimator_available"
    available = [row for row in rows if bool(row.get(available_key, False))]
    errors = [float(row[key]) for row in available]
    return {
        "branch_count": len(rows),
        "available_count": len(available),
        "coverage": len(available) / max(len(rows), 1),
        "mean_error_deg": "" if not errors else float(np.mean(errors)),
        "median_error_deg": "" if not errors else float(np.median(errors)),
        "p90_error_deg": "" if not errors else float(np.percentile(errors, 90)),
        "max_error_deg": "" if not errors else float(np.max(errors)),
    }


def _paired(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("opening_matched") and row.get("wall_estimator_available")]


def _correlation(rows: Sequence[Mapping[str, Any]], error_key: str) -> float:
    selected = [row for row in rows if row.get(error_key, "") != ""]
    if len(selected) < 2:
        return float("nan")
    x = np.asarray([float(row["boundary_asymmetry_deg"]) for row in selected])
    y = np.asarray([float(row[error_key]) for row in selected])
    if np.std(x) <= EPSILON or np.std(y) <= EPSILON:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _stage_one_pass(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, dict[str, bool]]:
    """Apply predeclared comparative checks before expanding the benchmark."""
    centered = [row for row in rows if row["anchor_offset_group"] == "centered"]
    off_center = [row for row in rows if row["anchor_offset_group"] != "centered"]
    large = [row for row in rows if row["anchor_offset_group"] == "large"]
    asymmetric = [row for row in rows if row["mouth_geometry"] == "asymmetric"]
    rotation = [row for row in centered if str(row["case_id"]).startswith("centered_")]
    center_c = _metric(centered, "opening_center")
    wall_c = _metric(centered, "wall")
    center_o = _metric(off_center, "opening_center")
    wall_o = _metric(off_center, "wall")
    rotation_errors = [float(row["wall_tangent_error_deg"]) for row in _paired(rotation)]
    checks = {
        # One 1° LiDAR bin is the only physical tolerance in the stage gate.
        "centered_no_regression": float(wall_c["p90_error_deg"]) <= float(center_c["p90_error_deg"]) + 1.0,
        "off_center_mean_improves": float(wall_o["mean_error_deg"]) < float(center_o["mean_error_deg"]),
        "off_center_p90_improves": float(wall_o["p90_error_deg"]) < float(center_o["p90_error_deg"]),
        "large_offset_worst_improves": _metric(large, "wall")["max_error_deg"] < _metric(large, "opening_center")["max_error_deg"],
        "asymmetric_mean_improves": _metric(asymmetric, "wall")["mean_error_deg"] < _metric(asymmetric, "opening_center")["mean_error_deg"],
        "rotation_within_one_scan_bin": bool(rotation_errors) and max(rotation_errors) - min(rotation_errors) <= 1.0,
        # Stage 2 is allowed only when wall estimation loses no matched opening.
        "complete_on_matched_openings": sum(bool(row.get("wall_estimator_available")) for row in rows) == sum(bool(row.get("opening_matched")) for row in rows),
    }
    return all(checks.values()), checks


def _summary_rows(rows: Sequence[Mapping[str, Any]], stage: str, classification: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    selections = {
        "overall": list(rows),
        "centered": [row for row in rows if row["anchor_offset_group"] == "centered"],
        "off_center": [row for row in rows if row["anchor_offset_group"] != "centered"],
        "large_offset": [row for row in rows if row["anchor_offset_group"] == "large"],
        "asymmetric": [row for row in rows if row["mouth_geometry"] == "asymmetric"],
        "production_scale": [row for row in rows if row["width_group"] == "production-scale"],
    }
    for group, selected in selections.items():
        for estimator in ("opening_center", "wall"):
            output.append({"stage": stage, "group": group, "estimator": estimator, **_metric(selected, estimator)})
    output.extend([
        {"stage": stage, "group": "correlation", "estimator": "opening_center", "metric": "boundary_asymmetry_pearson", "value": _correlation(rows, "opening_center_error_deg")},
        {"stage": stage, "group": "correlation", "estimator": "wall", "metric": "boundary_asymmetry_pearson", "value": _correlation(rows, "wall_tangent_error_deg")},
        {"stage": stage, "group": "classification", "estimator": "wall", "metric": "case", "value": classification},
    ])
    return output


def _group_rows(rows: Sequence[Mapping[str, Any]], dimension: str) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[dimension])].append(row)
    output = []
    for value, selected in sorted(groups.items()):
        for estimator in ("opening_center", "wall"):
            output.append({"group_dimension": dimension, "group_value": value, "estimator": estimator, **_metric(selected, estimator)})
    return output


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write CSV rows, retaining a valid file even for an empty failure set."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("case_id,branch_id,failure_reason\n", encoding="utf-8")
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


def _save_plots(directory: Path, rows: Sequence[Mapping[str, Any]], visuals: Mapping[str, Mapping[str, Any]]) -> None:
    paired = _paired(rows)
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.scatter([float(row["opening_center_error_deg"]) for row in paired], [float(row["wall_tangent_error_deg"]) for row in paired], alpha=0.65)
    limit = max([8.5] + [float(row["opening_center_error_deg"]) for row in paired])
    axis.plot([0, limit], [0, limit], "--", color="0.5", label="equal error")
    axis.set(xlabel="opening-center error [deg]", ylabel="wall-tangent error [deg]", title="Opening center vs wall-parallel tangent")
    axis.legend()
    figure.savefig(directory / "opening_center_vs_wall_tangent_error.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x = [float(row["anchor_offset_normalized"]) for row in paired]
    axis.scatter(x, [float(row["opening_center_error_deg"]) for row in paired], alpha=0.45, label="opening center")
    axis.scatter(x, [float(row["wall_tangent_error_deg"]) for row in paired], alpha=0.65, label="wall tangent")
    axis.set(xlabel="Anchor offset / central radius", ylabel="error [deg]", title="Anchor-offset sensitivity")
    axis.legend()
    figure.savefig(directory / "wall_tangent_error_vs_anchor_offset.png", dpi=160)
    plt.close(figure)

    worst = sorted(paired, key=lambda row: float(row["wall_tangent_error_deg"]), reverse=True)[:3]
    figure, axes = plt.subplots(1, max(1, len(worst)), figsize=(5 * max(1, len(worst)), 5), constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    for axis, row in zip(axes_array, worst):
        visual = visuals[str(row["case_id"])]
        case = visual["case"]
        scan = visual["scan"]
        axis.scatter(scan.local_x[scan.hit], scan.local_y[scan.hit], s=4, color="0.55")
        estimate = visual["estimates"][str(row["branch_id"])]
        if len(estimate.selected_points):
            axis.scatter(estimate.selected_points[:, 0], estimate.selected_points[:, 1], s=28, color="tab:blue")
        scale = case.central_radius * 0.75
        for angle, color, label in (
            (float(row["gt_tangent_deg"]), "green", "GT"),
            (float(row["opening_center_angle"]), "red", "center"),
            (float(row["wall_tangent_deg"]), "tab:blue", "wall"),
        ):
            theta = math.radians(angle)
            axis.arrow(0.0, 0.0, scale * math.cos(theta), scale * math.sin(theta), color=color, width=0.35, length_includes_head=True, label=label)
        axis.set_aspect("equal")
        axis.set_title(f"{row['case_id']}:{row['branch_id']}\nwall error={float(row['wall_tangent_error_deg']):.3g}°")
    if worst:
        axes_array[0].legend()
    figure.savefig(directory / "wall_orientation_failure_examples.png", dpi=160)
    plt.close(figure)


def run_synthetic_sanity() -> None:
    """Verify TLS line fit plus translation and rigid-rotation invariance."""
    parameter = np.linspace(-7.0, 9.0, 23)
    angle = 37.0
    direction = np.asarray([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
    base = np.asarray([2.5, -4.0]) + parameter[:, None] * direction
    fitted = fit_tls_line(base)
    assert _axial_difference_deg(fitted.orientation_deg, angle) < NUMERICAL_ANGLE_TOL_DEG
    translated = fit_tls_line(base + np.asarray([123.0, -81.0]))
    assert _axial_difference_deg(fitted.orientation_deg, translated.orientation_deg) < NUMERICAL_ANGLE_TOL_DEG
    rotation = 63.0
    matrix = np.asarray([
        [math.cos(math.radians(rotation)), -math.sin(math.radians(rotation))],
        [math.sin(math.radians(rotation)), math.cos(math.radians(rotation))],
    ])
    rotated = fit_tls_line(base @ matrix.T)
    assert _axial_difference_deg(rotated.orientation_deg, angle + rotation) < NUMERICAL_ANGLE_TOL_DEG
    assert fitted.residual_m < math.sqrt(np.finfo(float).eps)


STAGE_ONE_CASE_IDS = (
    "centered_3way_r000", "centered_3way_r060", "centered_4way_r030", "centered_5way_r120",
    "offset_small_tangent", "offset_small_normal", "offset_medium_normal", "offset_medium_diagonal",
    "offset_large_tangent", "offset_large_normal", "offset_large_diagonal",
    "width_production-scale_centered", "width_production-scale_medium",
    "asymmetric_3way", "asymmetric_5way",
)


def _run_cases(cases: Sequence[OrientationCase]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    visuals: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases, 1):
        print(f"[wall orientation] {index}/{len(cases)} {case.case_id}", flush=True)
        case_rows, visual = _evaluate_case(case)
        rows.extend(case_rows)
        visuals[case.case_id] = visual
    return rows, visuals


def run_benchmark(output_dir: Path, seed: int) -> dict[str, Any]:
    """Run Stage 1 and expand to all 35 cases only if its checks pass."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_cases = create_benchmark_cases(seed)
    by_id = {case.case_id: case for case in all_cases}
    stage_one_cases = [by_id[case_id] for case_id in STAGE_ONE_CASE_IDS]
    stage_one_rows, stage_one_visuals = _run_cases(stage_one_cases)
    stage_one_passed, checks = _stage_one_pass(stage_one_rows)
    if stage_one_passed:
        print("[wall orientation] Stage 1 PASS; expanding to 35 cases", flush=True)
        rows, visuals = _run_cases(all_cases)
        stage = "stage2_full"
        classification = "A"
    else:
        print("[wall orientation] Stage 1 FAIL; stopping before full benchmark", flush=True)
        rows, visuals = stage_one_rows, stage_one_visuals
        stage = "stage1_only"
        # Failure to isolate walls is C; comparative/coverage limitations are B.
        classification = "C" if not checks["complete_on_matched_openings"] else "B"

    failures = []
    for row in rows:
        if not row.get("opening_matched"):
            failures.append({"case_id": row["case_id"], "branch_id": row["branch_id"], "failure_reason": "opening_unmatched"})
        elif not row.get("wall_estimator_available"):
            failures.append({"case_id": row["case_id"], "branch_id": row["branch_id"], "failure_reason": row["estimate_mode"]})
        elif float(row["wall_tangent_error_deg"]) >= float(row["opening_center_error_deg"]) and float(row["wall_tangent_error_deg"]) > NUMERICAL_ANGLE_TOL_DEG:
            failures.append({"case_id": row["case_id"], "branch_id": row["branch_id"], "failure_reason": "wall_not_better", "wall_tangent_error_deg": row["wall_tangent_error_deg"], "opening_center_error_deg": row["opening_center_error_deg"]})

    summary = _summary_rows(rows, stage, classification)
    for name, passed in checks.items():
        summary.append({"stage": "stage1_gate", "group": name, "estimator": "wall", "metric": "passed", "value": passed})
    by_offset = _group_rows(rows, "anchor_offset_group") + _group_rows(rows, "anchor_offset_direction_group")
    _write_rows(output_dir / "wall_orientation_results.csv", rows)
    _write_rows(output_dir / "wall_orientation_summary.csv", summary)
    _write_rows(output_dir / "wall_orientation_by_offset.csv", by_offset)
    _write_rows(output_dir / "wall_orientation_failure_cases.csv", failures)
    _save_plots(output_dir, rows, visuals)
    result = {
        "classification": classification,
        "stage": stage,
        "stage_one_case_count": len(stage_one_cases),
        "executed_case_count": len({row["case_id"] for row in rows}),
        "branch_count": len(rows),
        "stage_one_checks": checks,
        "opening_center": _metric(rows, "opening_center"),
        "wall": _metric(rows, "wall"),
        "output_dir": str(output_dir),
    }
    (output_dir / "wall_orientation_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output-dir", type=Path, default=Path(f"/tmp/pdfs_pointcloud_wall_orientation_{_short_head()}"))
    parser.add_argument("--sanity-test", action="store_true")
    args = parser.parse_args()
    if args.sanity_test:
        run_synthetic_sanity()
        print("wall-parallel line-fit synthetic sanity: PASS")
    result = run_benchmark(args.output_dir, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
