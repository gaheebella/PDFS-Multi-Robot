"""Geometry-general evaluation of opening-center Branch orientation.

The estimator under test is exactly ``opening["center_angle"]`` from the
existing Point Cloud detector.  The detector receives only Anchor-local LiDAR
angle/range arrays.  Geometry, Branch identity, mouth endpoints, and tangent
directions are confined to environment construction and post-hoc evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
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

from junction_detection.pointcloud.pointcloud_junction_detector import (
    _merge_linear_intervals,
    _normalize_angles,
    _split_wrapped_interval,
    detect_openings,
    simulate_lidar_scan,
)
from junction_detection.pointcloud.pointcloud_junction_detector_local_topology import (
    _match_openings,
)


EPSILON = 1.0e-10


def normalize_angle_deg(angle: float) -> float:
    """Normalize one angle to [-180, 180)."""
    return float((float(angle) + 180.0) % 360.0 - 180.0)


def angular_error_deg(first: float, second: float) -> float:
    """Return the smallest unsigned circular angular separation."""
    return abs(normalize_angle_deg(first - second))


def signed_error_deg(estimate: float, truth: float) -> float:
    """Return signed estimate-minus-truth circular error."""
    return normalize_angle_deg(estimate - truth)


def unit(angle_deg: float) -> np.ndarray:
    """Return a unit vector at the supplied world angle."""
    angle = math.radians(angle_deg)
    return np.asarray([math.cos(angle), math.sin(angle)], dtype=float)


@dataclass(frozen=True)
class BranchSpec:
    """One evaluation-only oriented corridor."""

    branch_id: str
    angle_deg: float
    width: float
    length: float
    mouth_lateral_offset: float = 0.0

    @property
    def tangent(self) -> np.ndarray:
        return unit(self.angle_deg)

    @property
    def normal(self) -> np.ndarray:
        tangent = self.tangent
        return np.asarray([-tangent[1], tangent[0]], dtype=float)


@dataclass(frozen=True)
class OrientationCase:
    """One simulator-side geometry and Anchor pose."""

    case_id: str
    seed: int
    topology: str
    center: tuple[float, float]
    central_radius: float
    branches: tuple[BranchSpec, ...]
    anchor_xy: tuple[float, float]
    anchor_yaw_deg: float
    global_rotation_deg: float
    width_group: str
    length_group: str
    offset_group: str
    offset_direction_group: str
    mouth_geometry: str
    family: str

    @property
    def center_array(self) -> np.ndarray:
        return np.asarray(self.center, dtype=float)

    @property
    def anchor_array(self) -> np.ndarray:
        return np.asarray(self.anchor_xy, dtype=float)

    @property
    def anchor_offset(self) -> np.ndarray:
        return self.anchor_array - self.center_array


def _short_head() -> str:
    """Read the current short Git identity without mutation."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write dictionary rows using their union of columns."""
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


def _mouth_endpoints(
    case: OrientationCase,
    branch: BranchSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return left/right corridor-side intersections with the central circle."""
    half_width = branch.width * 0.5
    left_lateral = branch.mouth_lateral_offset + half_width
    right_lateral = branch.mouth_lateral_offset - half_width
    if max(abs(left_lateral), abs(right_lateral)) >= case.central_radius:
        raise ValueError(f"{case.case_id}:{branch.branch_id}: mouth outside circle")
    left_axial = math.sqrt(case.central_radius**2 - left_lateral**2)
    right_axial = math.sqrt(case.central_radius**2 - right_lateral**2)
    left = (
        case.center_array
        + left_axial * branch.tangent
        + left_lateral * branch.normal
    )
    right = (
        case.center_array
        + right_axial * branch.tangent
        + right_lateral * branch.normal
    )
    return left, right


def _local_angle(case: OrientationCase, point: np.ndarray) -> float:
    delta = np.asarray(point, dtype=float) - case.anchor_array
    return normalize_angle_deg(
        math.degrees(math.atan2(delta[1], delta[0])) - case.anchor_yaw_deg
    )


def _gt_opening(case: OrientationCase, branch: BranchSpec) -> dict[str, float]:
    """Build an evaluation-only mouth sector from its physical endpoints."""
    left, right = _mouth_endpoints(case, branch)
    start = _local_angle(case, right)
    end = _local_angle(case, left)
    width = float((end - start) % 360.0)
    if width > 180.0:
        start, end = end, start
        width = 360.0 - width
    return {
        "start_angle": start,
        "end_angle": end,
        "center_angle": normalize_angle_deg(start + width * 0.5),
        "width_deg": width,
    }


def _append_arc(
    walls: list[list[list[float]]],
    center: np.ndarray,
    radius: float,
    start_deg: float,
    end_deg: float,
) -> None:
    """Approximate one visible central-wall arc with 2-degree segments."""
    count = max(1, int(math.ceil((end_deg - start_deg) / 2.0)))
    angles = np.linspace(start_deg, end_deg, count + 1)
    points = center + np.column_stack((
        radius * np.cos(np.radians(angles)),
        radius * np.sin(np.radians(angles)),
    ))
    walls.extend([[p0.tolist(), p1.tolist()] for p0, p1 in zip(points[:-1], points[1:])])


def wall_segments(case: OrientationCase) -> np.ndarray:
    """Create wall segments for unequal widths and laterally shifted mouths."""
    validate_case(case)
    walls: list[list[list[float]]] = []
    openings: list[tuple[float, float]] = []
    for branch in case.branches:
        left, right = _mouth_endpoints(case, branch)
        left_lateral = branch.mouth_lateral_offset + branch.width * 0.5
        right_lateral = branch.mouth_lateral_offset - branch.width * 0.5
        end_axial = case.central_radius + branch.length
        left_end = case.center_array + end_axial * branch.tangent + left_lateral * branch.normal
        right_end = case.center_array + end_axial * branch.tangent + right_lateral * branch.normal
        walls.extend([
            [left.tolist(), left_end.tolist()],
            [right.tolist(), right_end.tolist()],
        ])
        left_polar = math.degrees(math.atan2(
            left[1] - case.center[1], left[0] - case.center[0]
        ))
        right_polar = math.degrees(math.atan2(
            right[1] - case.center[1], right[0] - case.center[0]
        ))
        width = (left_polar - right_polar) % 360.0
        if width > 180.0:
            right_polar, left_polar = left_polar, right_polar
        openings.extend(_split_wrapped_interval(right_polar, left_polar))

    cursor = 0.0
    for start, end in _merge_linear_intervals(openings):
        if start > cursor + EPSILON:
            _append_arc(walls, case.center_array, case.central_radius, cursor, start)
        cursor = max(cursor, end)
    if cursor < 360.0 - EPSILON:
        _append_arc(walls, case.center_array, case.central_radius, cursor, 360.0)
    return np.asarray(walls, dtype=float)


def validate_case(case: OrientationCase) -> None:
    """Reject invalid Anchor placement, duplicate directions, and mouth overlap."""
    if len(case.branches) < 3:
        raise ValueError("benchmark topology must have at least three Branches")
    if np.linalg.norm(case.anchor_offset) >= case.central_radius * 0.80:
        raise ValueError("Anchor must remain safely inside the central region")
    normalized = sorted(branch.angle_deg % 360.0 for branch in case.branches)
    gaps = np.diff(np.r_[normalized, normalized[0] + 360.0])
    if float(np.min(gaps)) <= 1.0e-6:
        raise ValueError("duplicate Branch direction")
    intervals: list[tuple[float, float]] = []
    total_width = 0.0
    for branch in case.branches:
        if branch.width <= 0.0 or branch.length <= 0.0:
            raise ValueError("Branch width and length must be positive")
        left, right = _mouth_endpoints(case, branch)
        left_angle = math.degrees(math.atan2(
            left[1] - case.center[1], left[0] - case.center[0]
        ))
        right_angle = math.degrees(math.atan2(
            right[1] - case.center[1], right[0] - case.center[0]
        ))
        width = (left_angle - right_angle) % 360.0
        if width > 180.0:
            right_angle, left_angle = left_angle, right_angle
            width = 360.0 - width
        intervals.extend(_split_wrapped_interval(right_angle, left_angle))
        total_width += width
    merged = _merge_linear_intervals(intervals)
    merged_width = sum(end - start for start, end in merged)
    if merged_width < total_width - 1.0e-6:
        raise ValueError("Branch mouth overlap")


def _branch_angles(topology: int, rotation: float) -> tuple[float, ...]:
    bases = {
        3: (-110.0, 10.0, 125.0),
        4: (-135.0, -45.0, 40.0, 135.0),
        5: (-150.0, -75.0, -5.0, 65.0, 140.0),
    }
    return tuple(normalize_angle_deg(angle + rotation) for angle in bases[topology])


def _make_case(
    *,
    case_id: str,
    seed: int,
    topology: int,
    rotation: float,
    radius: float,
    widths: Sequence[float],
    lengths: Sequence[float],
    offset_fraction: float,
    offset_direction_deg: float,
    offset_group: str,
    offset_direction_group: str,
    mouth_offsets: Sequence[float] | None = None,
    mouth_geometry: str = "symmetric",
    width_group: str = "nominal",
    length_group: str = "nominal",
    family: str = "representative",
    center: tuple[float, float] = (0.0, 0.0),
) -> OrientationCase:
    angles = _branch_angles(topology, rotation)
    if len(widths) == 1:
        widths = tuple(widths) * topology
    if len(lengths) == 1:
        lengths = tuple(lengths) * topology
    if mouth_offsets is None:
        mouth_offsets = (0.0,) * topology
    branches = tuple(
        BranchSpec(f"B{index}", angle, float(width), float(length), float(mouth_offset))
        for index, (angle, width, length, mouth_offset) in enumerate(
            zip(angles, widths, lengths, mouth_offsets)
        )
    )
    offset = radius * offset_fraction * unit(offset_direction_deg + rotation)
    center_array = np.asarray(center, dtype=float)
    anchor = center_array + offset
    case = OrientationCase(
        case_id=case_id,
        seed=seed,
        topology=f"{topology}-way",
        center=center,
        central_radius=radius,
        branches=branches,
        anchor_xy=(float(anchor[0]), float(anchor[1])),
        anchor_yaw_deg=rotation,
        global_rotation_deg=rotation,
        width_group=width_group,
        length_group=length_group,
        offset_group=offset_group,
        offset_direction_group=offset_direction_group,
        mouth_geometry=mouth_geometry,
        family=family,
    )
    validate_case(case)
    return case


def create_benchmark_cases(seed: int = 20260818) -> list[OrientationCase]:
    """Create a 35-case balanced representative geometry design."""
    cases: list[OrientationCase] = []
    index = 0

    # Topology plus rigid world rotation, with a centered Anchor.
    for topology in (3, 4, 5):
        for rotation in (0.0, 30.0, 60.0, 120.0):
            cases.append(_make_case(
                case_id=f"centered_{topology}way_r{int(rotation):03d}",
                seed=seed + index,
                topology=topology,
                rotation=rotation,
                radius=48.0,
                widths=(24.0,),
                lengths=(105.0,),
                offset_fraction=0.0,
                offset_direction_deg=0.0,
                offset_group="centered",
                offset_direction_group="none",
                family="topology_rotation",
            ))
            index += 1

    # Most important offset design: magnitude crossed with three directions.
    for offset_group, fraction in (("small", 0.12), ("medium", 0.28), ("large", 0.45)):
        for direction_group, direction in (("tangent", -135.0), ("normal", -45.0), ("diagonal", -90.0)):
            cases.append(_make_case(
                case_id=f"offset_{offset_group}_{direction_group}",
                seed=seed + index,
                topology=4,
                rotation=30.0,
                radius=48.0,
                widths=(24.0,),
                lengths=(105.0,),
                offset_fraction=fraction,
                offset_direction_deg=direction,
                offset_group=offset_group,
                offset_direction_group=direction_group,
                family="anchor_offset",
            ))
            index += 1

    # Width checks at centered and medium normal offsets.
    width_specs = (
        ("narrow", 16.0, 42.0),
        ("nominal", 24.0, 48.0),
        ("wide", 32.0, 58.0),
        ("production-scale", 84.0, 120.0),
    )
    for width_group, width, radius in width_specs:
        for offset_group, fraction in (("centered", 0.0), ("medium", 0.28)):
            cases.append(_make_case(
                case_id=f"width_{width_group}_{offset_group}",
                seed=seed + index,
                topology=4,
                rotation=25.0,
                radius=radius,
                widths=(width,),
                lengths=(105.0,),
                offset_fraction=fraction,
                offset_direction_deg=-45.0,
                offset_group=offset_group,
                offset_direction_group="none" if fraction == 0.0 else "normal",
                width_group=width_group,
                family="width",
            ))
            index += 1

    # Length is sampled without a full factorial expansion.
    for length_group, length in (("short", 70.0), ("nominal", 105.0), ("long", 256.0)):
        cases.append(_make_case(
            case_id=f"length_{length_group}_medium_diagonal",
            seed=seed + index,
            topology=4,
            rotation=60.0,
            radius=48.0,
            widths=(24.0,),
            lengths=(length,),
            offset_fraction=0.28,
            offset_direction_deg=-90.0,
            offset_group="medium",
            offset_direction_group="diagonal",
            length_group=length_group,
            family="length",
        ))
        index += 1

    # Unequal widths and laterally shifted corridor mouths.
    for topology, rotation in ((3, 20.0), (4, 55.0), (5, 100.0)):
        widths = tuple(20.0 + 4.0 * (branch % 3) for branch in range(topology))
        mouth_offsets = tuple(
            (0.14 if branch % 2 == 0 else -0.12) * widths[branch]
            for branch in range(topology)
        )
        cases.append(_make_case(
            case_id=f"asymmetric_{topology}way",
            seed=seed + index,
            topology=topology,
            rotation=rotation,
            radius=58.0,
            widths=widths,
            lengths=tuple(90.0 + 18.0 * branch for branch in range(topology)),
            offset_fraction=0.28,
            offset_direction_deg=-90.0,
            offset_group="medium",
            offset_direction_group="diagonal",
            mouth_offsets=mouth_offsets,
            mouth_geometry="asymmetric",
            width_group="mixed",
            length_group="mixed",
            family="asymmetric_mouth",
        ))
        index += 1
    return cases


def _evaluate_case(case: OrientationCase) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ray-cast and call the unchanged angle/range-only opening detector."""
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
    # Information boundary: only these two arrays enter the estimator.
    angles_deg, ranges = scan.detector_input()
    detected = detect_openings(angles_deg, ranges)
    ground_truth = [_gt_opening(case, branch) for branch in case.branches]
    matches = _match_openings(ground_truth, detected)
    match_by_gt = {gt_index: (det_index, iou) for gt_index, det_index, iou in matches}
    offset = case.anchor_offset
    offset_magnitude = float(np.linalg.norm(offset))
    rows = []
    for branch_index, branch in enumerate(case.branches):
        match = match_by_gt.get(branch_index)
        gt_tangent_local = normalize_angle_deg(branch.angle_deg - case.anchor_yaw_deg)
        if offset_magnitude <= EPSILON:
            relative_offset_direction = ""
            relative_offset_angle = ""
        else:
            offset_world_angle = math.degrees(math.atan2(offset[1], offset[0]))
            relative_offset_angle = normalize_angle_deg(offset_world_angle - branch.angle_deg)
            relative_offset_direction = relative_offset_angle
        base = {
            "case_id": case.case_id,
            "seed": case.seed,
            "topology": case.topology,
            "branch_id": branch.branch_id,
            "gt_tangent_deg": gt_tangent_local,
            "anchor_offset_x": float(offset[0]),
            "anchor_offset_y": float(offset[1]),
            "anchor_offset_magnitude": offset_magnitude,
            "anchor_offset_normalized": offset_magnitude / case.central_radius,
            "anchor_offset_group": case.offset_group,
            "anchor_offset_direction_group": case.offset_direction_group,
            "anchor_offset_direction_relative_gt_deg": relative_offset_direction,
            "corridor_width": branch.width,
            "width_group": case.width_group,
            "branch_length": branch.length,
            "length_group": case.length_group,
            "global_rotation_deg": case.global_rotation_deg,
            "mouth_geometry": case.mouth_geometry,
            "mouth_lateral_offset": branch.mouth_lateral_offset,
            "matched": match is not None,
            "opening_detection_failure": match is None,
            "detected_opening_count": len(detected),
            "expected_opening_count": len(case.branches),
        }
        if match is None:
            rows.append({
                **base,
                "opening_start_angle": "",
                "opening_end_angle": "",
                "opening_center_angle": "",
                "angular_error_deg": "",
                "signed_angular_bias_deg": "",
                "opening_confidence": "",
                "angular_opening_width_deg": "",
                "left_boundary_offset_deg": "",
                "right_boundary_offset_deg": "",
                "boundary_asymmetry_deg": "",
                "match_iou": "",
            })
            continue
        detected_index, iou = match
        opening = detected[detected_index]
        center_angle = float(opening["center_angle"])
        start_delta = signed_error_deg(float(opening["start_angle"]), gt_tangent_local)
        end_delta = signed_error_deg(float(opening["end_angle"]), gt_tangent_local)
        rows.append({
            **base,
            "opening_start_angle": opening["start_angle"],
            "opening_end_angle": opening["end_angle"],
            "opening_center_angle": center_angle,
            "angular_error_deg": angular_error_deg(center_angle, gt_tangent_local),
            "signed_angular_bias_deg": signed_error_deg(center_angle, gt_tangent_local),
            "opening_confidence": opening["confidence"],
            "angular_opening_width_deg": opening["width_deg"],
            "left_boundary_offset_deg": end_delta,
            "right_boundary_offset_deg": start_delta,
            "boundary_asymmetry_deg": abs(abs(end_delta) - abs(start_delta)),
            "match_iou": iou,
        })
    visual = {
        "case": case,
        "walls": walls,
        "scan": scan,
        "detected": detected,
        "ground_truth": ground_truth,
    }
    return rows, visual


def _statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute coverage and circular-error distribution metrics."""
    matched = [row for row in rows if bool(row["matched"])]
    errors = [float(row["angular_error_deg"]) for row in matched]
    biases = [float(row["signed_angular_bias_deg"]) for row in matched]
    return {
        "branch_count": len(rows),
        "matched_count": len(matched),
        "opening_coverage": len(matched) / max(len(rows), 1),
        "mean_error_deg": "" if not errors else float(np.mean(errors)),
        "median_error_deg": "" if not errors else float(np.median(errors)),
        "p90_error_deg": "" if not errors else float(np.percentile(errors, 90)),
        "max_error_deg": "" if not errors else float(np.max(errors)),
        "mean_signed_bias_deg": "" if not biases else float(np.mean(biases)),
        "median_signed_bias_deg": "" if not biases else float(np.median(biases)),
    }


def _group_summary(
    rows: Sequence[Mapping[str, Any]],
    dimension: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[dimension])].append(row)
    return [
        {"group_dimension": dimension, "group_value": value, **_statistics(group)}
        for value, group in sorted(groups.items())
    ]


def _correlation(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or np.std(x) <= EPSILON or np.std(y) <= EPSILON:
        return float("nan")
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def _analysis_summary(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    centered = [row for row in rows if row["anchor_offset_group"] == "centered"]
    off_center = [row for row in rows if row["anchor_offset_group"] != "centered"]
    centered_stats = _statistics(centered)
    off_stats = _statistics(off_center)
    matched = [row for row in rows if row["matched"]]
    asymmetry = [float(row["boundary_asymmetry_deg"]) for row in matched]
    errors = [float(row["angular_error_deg"]) for row in matched]
    asymmetry_correlation = _correlation(asymmetry, errors)

    def numeric(stats: Mapping[str, Any], key: str) -> float:
        value = stats[key]
        return float("nan") if value == "" else float(value)

    systematic_offset_growth = all(
        numeric(off_stats, metric) > numeric(centered_stats, metric)
        for metric in ("mean_error_deg", "median_error_deg", "p90_error_deg", "max_error_deg")
    )
    if centered_stats["opening_coverage"] < off_stats["opening_coverage"]:
        classification = "C"
    elif systematic_offset_growth and asymmetry_correlation > 0.0:
        classification = "B"
    else:
        classification = "A"
    rotation_rows = _group_summary(
        [row for row in rows if str(row["case_id"]).startswith("centered_")],
        "global_rotation_deg",
    )
    rotation_means = [float(row["mean_error_deg"]) for row in rotation_rows if row["mean_error_deg"] != ""]
    summary = [
        {"metric": "case_count", "value": len({row["case_id"] for row in rows})},
        {"metric": "branch_count", "value": len(rows)},
        {"metric": "overall_opening_coverage", "value": _statistics(rows)["opening_coverage"]},
        {"metric": "overall_mean_error_deg", "value": _statistics(rows)["mean_error_deg"]},
        {"metric": "overall_median_error_deg", "value": _statistics(rows)["median_error_deg"]},
        {"metric": "overall_p90_error_deg", "value": _statistics(rows)["p90_error_deg"]},
        {"metric": "overall_max_error_deg", "value": _statistics(rows)["max_error_deg"]},
        {"metric": "centered_coverage", "value": centered_stats["opening_coverage"]},
        {"metric": "centered_mean_error_deg", "value": centered_stats["mean_error_deg"]},
        {"metric": "centered_median_error_deg", "value": centered_stats["median_error_deg"]},
        {"metric": "centered_p90_error_deg", "value": centered_stats["p90_error_deg"]},
        {"metric": "centered_max_error_deg", "value": centered_stats["max_error_deg"]},
        {"metric": "off_center_coverage", "value": off_stats["opening_coverage"]},
        {"metric": "off_center_mean_error_deg", "value": off_stats["mean_error_deg"]},
        {"metric": "off_center_median_error_deg", "value": off_stats["median_error_deg"]},
        {"metric": "off_center_p90_error_deg", "value": off_stats["p90_error_deg"]},
        {"metric": "off_center_max_error_deg", "value": off_stats["max_error_deg"]},
        {"metric": "asymmetry_error_pearson", "value": asymmetry_correlation},
        {"metric": "rotation_mean_error_span_deg", "value": "" if not rotation_means else max(rotation_means) - min(rotation_means)},
        {"metric": "classification", "value": classification},
    ]
    return summary, classification


def _bar_plot(path: Path, rows: Sequence[Mapping[str, Any]], dimension: str, title: str) -> None:
    grouped = _group_summary(rows, dimension)
    labels = [row["group_value"] for row in grouped]
    means = [float(row["mean_error_deg"]) if row["mean_error_deg"] != "" else np.nan for row in grouped]
    p90 = [float(row["p90_error_deg"]) if row["p90_error_deg"] != "" else np.nan for row in grouped]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.bar(x - 0.18, means, 0.36, label="mean")
    axis.bar(x + 0.18, p90, 0.36, label="P90")
    axis.set_xticks(x, labels, rotation=45, ha="right")
    axis.set(ylabel="orientation error [deg]", title=title)
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_plots(
    directory: Path,
    rows: Sequence[Mapping[str, Any]],
    visuals: Mapping[str, Mapping[str, Any]],
) -> None:
    matched = [row for row in rows if row["matched"]]
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for direction in sorted({str(row["anchor_offset_direction_group"]) for row in matched}):
        selected = [row for row in matched if str(row["anchor_offset_direction_group"]) == direction]
        axis.scatter(
            [float(row["anchor_offset_normalized"]) for row in selected],
            [float(row["angular_error_deg"]) for row in selected],
            label=direction,
            alpha=0.65,
        )
    axis.set(xlabel="Anchor offset / central radius", ylabel="orientation error [deg]", title="Opening-center error vs Anchor offset")
    axis.legend()
    figure.savefig(directory / "pointcloud_orientation_error_vs_anchor_offset.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.scatter(
        [float(row["gt_tangent_deg"]) for row in matched],
        [float(row["angular_error_deg"]) for row in matched],
        c=[float(row["anchor_offset_normalized"]) for row in matched],
        cmap="viridis", alpha=0.7,
    )
    axis.set(xlabel="GT Branch tangent [deg]", ylabel="orientation error [deg]", title="Error vs arbitrary Branch angle")
    figure.savefig(directory / "pointcloud_orientation_error_vs_branch_angle.png", dpi=160)
    plt.close(figure)

    _bar_plot(directory / "pointcloud_orientation_error_vs_width.png", rows, "width_group", "Error vs corridor width")
    _bar_plot(directory / "pointcloud_orientation_error_vs_topology.png", rows, "topology", "Error vs topology")
    _bar_plot(directory / "pointcloud_orientation_rotation_invariance.png", [row for row in rows if str(row["case_id"]).startswith("centered_")], "global_rotation_deg", "Rigid-rotation invariance")

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.scatter(
        [float(row["boundary_asymmetry_deg"]) for row in matched],
        [float(row["angular_error_deg"]) for row in matched],
        c=[float(row["anchor_offset_normalized"]) for row in matched],
        cmap="plasma", alpha=0.75,
    )
    axis.set(xlabel="boundary asymmetry [deg]", ylabel="orientation error [deg]", title="Angular boundary asymmetry explains center-angle bias")
    figure.savefig(directory / "pointcloud_boundary_asymmetry_vs_orientation_error.png", dpi=160)
    plt.close(figure)

    worst = sorted(
        matched,
        key=lambda row: float(row["angular_error_deg"]),
        reverse=True,
    )[:4]
    figure, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
    for axis, row in zip(axes.flat, worst):
        visual = visuals[str(row["case_id"])]
        case = visual["case"]
        for wall in visual["walls"]:
            axis.plot([wall[0, 0], wall[1, 0]], [wall[0, 1], wall[1, 1]], color="0.55", linewidth=0.8)
        axis.scatter(*case.anchor_xy, marker="x", color="black", s=55)
        gt_world = math.radians(float(row["gt_tangent_deg"]) + case.anchor_yaw_deg)
        estimate_world = math.radians(float(row["opening_center_angle"]) + case.anchor_yaw_deg)
        scale = case.central_radius * 0.8
        axis.arrow(*case.anchor_xy, scale * math.cos(gt_world), scale * math.sin(gt_world), color="green", width=0.5, length_includes_head=True)
        axis.arrow(*case.anchor_xy, scale * math.cos(estimate_world), scale * math.sin(estimate_world), color="red", width=0.35, length_includes_head=True)
        for key, color in (("opening_start_angle", "tab:blue"), ("opening_end_angle", "tab:blue")):
            angle = math.radians(float(row[key]) + case.anchor_yaw_deg)
            axis.plot([case.anchor_xy[0], case.anchor_xy[0] + scale * math.cos(angle)], [case.anchor_xy[1], case.anchor_xy[1] + scale * math.sin(angle)], color=color, linestyle="--")
        axis.set_aspect("equal")
        axis.set_title(f"{row['case_id']}:{row['branch_id']}  error={float(row['angular_error_deg']):.2f}°")
    figure.savefig(directory / "pointcloud_orientation_failure_examples.png", dpi=160)
    plt.close(figure)

    examples = []
    for topology in ("3-way", "4-way", "5-way"):
        example = next(
            visual for visual in visuals.values()
            if visual["case"].topology == topology
            and visual["case"].offset_group != "centered"
        )
        examples.append(example)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, visual in zip(axes, examples):
        case = visual["case"]
        for wall in visual["walls"]:
            axis.plot([wall[0, 0], wall[1, 0]], [wall[0, 1], wall[1, 1]], color="0.4", linewidth=0.8)
        axis.scatter(*case.anchor_xy, marker="x", color="black", s=60)
        for opening in visual["detected"]:
            angle = math.radians(float(opening["center_angle"]) + case.anchor_yaw_deg)
            axis.arrow(*case.anchor_xy, case.central_radius * 0.65 * math.cos(angle), case.central_radius * 0.65 * math.sin(angle), width=0.3, color="tab:red", length_includes_head=True)
        axis.set_aspect("equal")
        axis.set_title(f"{case.topology}: {case.case_id}")
    figure.savefig(directory / "pointcloud_orientation_geometry_examples.png", dpi=160)
    plt.close(figure)


def run_synthetic_sanity() -> None:
    """Check invalid rejection, information boundary, translation, and rotation."""
    cases = create_benchmark_cases(17)
    assert len(cases) == 35
    assert {case.topology for case in cases} == {"3-way", "4-way", "5-way"}
    parameters = set(inspect.signature(detect_openings).parameters)
    assert parameters == {"angles_deg", "ranges", "kwargs"}

    base = _make_case(
        case_id="sanity_base", seed=1, topology=3, rotation=0.0,
        radius=48.0, widths=(24.0,), lengths=(105.0,),
        offset_fraction=0.28, offset_direction_deg=-90.0,
        offset_group="medium", offset_direction_group="diagonal",
    )
    translated = _make_case(
        case_id="sanity_translated", seed=1, topology=3, rotation=0.0,
        radius=48.0, widths=(24.0,), lengths=(105.0,),
        offset_fraction=0.28, offset_direction_deg=-90.0,
        offset_group="medium", offset_direction_group="diagonal",
        center=(217.0, -131.0),
    )
    rotated = _make_case(
        case_id="sanity_rotated", seed=1, topology=3, rotation=60.0,
        radius=48.0, widths=(24.0,), lengths=(105.0,),
        offset_fraction=0.28, offset_direction_deg=-90.0,
        offset_group="medium", offset_direction_group="diagonal",
    )
    base_rows, _ = _evaluate_case(base)
    translated_rows, _ = _evaluate_case(translated)
    rotated_rows, _ = _evaluate_case(rotated)
    base_errors = [row["angular_error_deg"] for row in base_rows]
    assert np.allclose(base_errors, [row["angular_error_deg"] for row in translated_rows])
    assert np.allclose(base_errors, [row["angular_error_deg"] for row in rotated_rows])

    invalid = OrientationCase(
        case_id="invalid_overlap", seed=0, topology="3-way", center=(0.0, 0.0),
        central_radius=30.0,
        branches=(
            BranchSpec("B0", 0.0, 24.0, 80.0),
            BranchSpec("B1", 5.0, 24.0, 80.0),
            BranchSpec("B2", 180.0, 24.0, 80.0),
        ),
        anchor_xy=(0.0, 0.0), anchor_yaw_deg=0.0,
        global_rotation_deg=0.0, width_group="invalid", length_group="invalid",
        offset_group="centered", offset_direction_group="none",
        mouth_geometry="symmetric", family="invalid",
    )
    try:
        validate_case(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid overlapping mouths were not rejected")


def run_benchmark(output_dir: Path, seed: int) -> dict[str, Any]:
    """Evaluate all cases twice, assert determinism, then save artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = create_benchmark_cases(seed)
    rows: list[dict[str, Any]] = []
    visuals: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases, 1):
        print(f"[pointcloud orientation] {index}/{len(cases)} {case.case_id}", flush=True)
        case_rows, visual = _evaluate_case(case)
        rows.extend(case_rows)
        visuals[case.case_id] = visual
    replay_rows = []
    for case in cases:
        case_rows, _ = _evaluate_case(case)
        replay_rows.extend(case_rows)
    first_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    replay_hash = hashlib.sha256(json.dumps(replay_rows, sort_keys=True).encode()).hexdigest()
    if first_hash != replay_hash:
        raise AssertionError("deterministic replay mismatch")

    summary, classification = _analysis_summary(rows)
    by_offset = _group_summary(rows, "anchor_offset_group") + _group_summary(rows, "anchor_offset_direction_group")
    by_width = _group_summary(rows, "width_group")
    by_topology = _group_summary(rows, "topology")
    by_angle = _group_summary(rows, "gt_tangent_deg")
    rotation = _group_summary(
        [row for row in rows if str(row["case_id"]).startswith("centered_")],
        "global_rotation_deg",
    )
    asymmetry_rows = [
        {
            "case_id": row["case_id"],
            "branch_id": row["branch_id"],
            "anchor_offset_normalized": row["anchor_offset_normalized"],
            "mouth_geometry": row["mouth_geometry"],
            "boundary_asymmetry_deg": row["boundary_asymmetry_deg"],
            "angular_error_deg": row["angular_error_deg"],
        }
        for row in rows if row["matched"]
    ]
    _write_rows(output_dir / "pointcloud_orientation_results.csv", rows)
    _write_rows(output_dir / "pointcloud_orientation_summary.csv", summary)
    _write_rows(output_dir / "pointcloud_orientation_by_offset.csv", by_offset)
    _write_rows(output_dir / "pointcloud_orientation_by_width.csv", by_width)
    _write_rows(output_dir / "pointcloud_orientation_by_topology.csv", by_topology)
    _write_rows(output_dir / "pointcloud_orientation_by_angle.csv", by_angle)
    _write_rows(output_dir / "pointcloud_orientation_rotation.csv", rotation)
    _write_rows(output_dir / "pointcloud_orientation_asymmetry.csv", asymmetry_rows)
    _save_plots(output_dir, rows, visuals)
    result = {
        "classification": classification,
        "case_count": len(cases),
        "branch_count": len(rows),
        "deterministic_sha256": first_hash,
    }
    (output_dir / "pointcloud_orientation_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(f"/tmp/pdfs_pointcloud_orientation_{_short_head()}"),
    )
    parser.add_argument("--sanity-test", action="store_true")
    args = parser.parse_args()
    if args.sanity_test:
        run_synthetic_sanity()
        print("pointcloud orientation synthetic sanity: PASS")
    result = run_benchmark(args.output_dir, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"artifacts={args.output_dir}")


if __name__ == "__main__":
    main()
