"""EXP-030: map 2D ghost-viewpoint raw and detector visibility frontiers.

The validated A0 state is acquired once. All other poses are evaluation-only
LiDAR origins in the frozen Anchor-local corridor frame. Geometry is used for
ray casting, robot-footprint validity, raw GT LOS/mouth exposure, and post-hoc
matching only; it never changes detector output or production behavior.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.integration.run_local_asymmetric_viewpoint_geometry_diagnostic import (
    _acquire_m0_snapshot,
    _acquire_m1_anchor,
)
from junction_detection.integration.run_nonforward_viewpoint_magnitude_boundary import (
    _opening_evaluation,
    _point_segment_distance,
)
from junction_detection.integration.run_provisional_anchor_pointcloud_observation import (
    _gt_directions_eval_only,
    _normalize,
    evaluate_snapshot,
)
from junction_detection.integration.run_sensor_rotation_multiview_geometry import (
    _branch_label,
    _detector_branch_rows,
    _gt_visibility_rows,
)
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import (
    detect_openings,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import ROBOT_RADIUS

EXPERIMENT_ID = "EXP-030"
DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/2d_viewpoint_visibility_frontier"
FORWARD_LIMITS = (-0.10, 0.70)
LATERAL_LIMITS = (-0.70, 0.70)
COARSE_STEP = 0.10
FINE_STEP = 0.05
FINAL_STEP = 0.025
MOUTH_SAMPLE_COUNT = 21
M0_POINTS = (
    (0.0, 0.0),
    (0.0, 0.30),
    (0.0, -0.30),
    (0.30, 0.30),
    (0.30, -0.30),
    (0.50, 0.40),
    (0.50, -0.40),
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write heterogeneous dictionaries using a stable field union."""
    if not rows:
        return
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _axis_frame(anchor: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return frozen unit forward and left vectors from the A0 local model."""
    forward = np.asarray(anchor["corridor_forward"], dtype=float)
    forward /= np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0]])
    return forward, left


def _local_to_world(
    anchor: dict[str, Any], forward_ratio: float, lateral_ratio: float
) -> np.ndarray:
    """Convert width-normalized local coordinates to a ghost world point."""
    forward, left = _axis_frame(anchor)
    width = float(anchor["estimated_corridor_width"])
    return (
        np.asarray(anchor["position_eval"], dtype=float)
        + forward * forward_ratio * width
        + left * lateral_ratio * width
    )


def _world_to_local(anchor: dict[str, Any], point: np.ndarray) -> tuple[float, float]:
    """Invert the Anchor-local transform for regression auditing."""
    forward, left = _axis_frame(anchor)
    relative = np.asarray(point, dtype=float) - np.asarray(anchor["position_eval"], dtype=float)
    width = float(anchor["estimated_corridor_width"])
    return float(relative @ forward / width), float(relative @ left / width)


def _candidate_id(forward_ratio: float, lateral_ratio: float) -> str:
    """Create a deterministic coordinate-derived ghost candidate ID."""
    return f"F{forward_ratio:+.3f}_L{lateral_ratio:+.3f}"


def _grid_values(lower: float, upper: float, step: float) -> list[float]:
    """Return inclusive decimal grid values without binary drift."""
    count = int(round((upper - lower) / step))
    return [round(lower + index * step, 6) for index in range(count + 1)]


def _path_clear(
    runner: Any, anchor: dict[str, Any], endpoint: np.ndarray
) -> bool:
    """Audit a straight footprint-valid segment; it is not a success gate."""
    origin = np.asarray(anchor["position_eval"], dtype=float)
    distance = float(np.linalg.norm(endpoint - origin))
    samples = max(2, int(math.ceil(distance / max(ROBOT_RADIUS * 0.5, 1.0e-9))) + 1)
    return all(
        runner.geometry.walkable(origin + fraction * (endpoint - origin), ROBOT_RADIUS)
        for fraction in np.linspace(0.0, 1.0, samples)
    )


def _branch_mouth_points(runner: Any, branch_id: int) -> np.ndarray:
    """Sample cell centers across the existing GT branch-mouth segment."""
    branch = runner.geometry.branches[branch_id]
    radians = math.radians(float(branch.angle_deg))
    direction = np.array([math.sin(radians), math.cos(radians)])
    lateral = np.array([-direction[1], direction[0]])
    center = direction * (float(runner.geometry.junction_size) / 2.0 - 2.0)
    fractions = (np.arange(MOUTH_SAMPLE_COUNT, dtype=float) + 0.5) / MOUTH_SAMPLE_COUNT
    offsets = (fractions - 0.5) * float(branch.width)
    return center + offsets[:, None] * lateral


def _mouth_visible_fraction(
    runner: Any, origin: np.ndarray, branch_id: int, max_range: float
) -> float:
    """Return the fraction of branch-mouth samples with unobstructed LOS."""
    visible = 0
    for target in _branch_mouth_points(runner, branch_id):
        relative = target - origin
        distance = float(np.linalg.norm(relative))
        if distance <= np.finfo(float).eps or distance > max_range:
            continue
        direction = relative / distance
        hits = [
            hit
            for wall in runner.geometry.walls
            if (hit := runner.world.sensor._ray_hit(origin, direction, wall)) is not None
        ]
        nearest = min(hits, default=math.inf)
        if nearest >= distance - 1.0e-7:
            visible += 1
    return visible / MOUTH_SAMPLE_COUNT


def _probe(
    runner: Any,
    anchor: dict[str, Any],
    case: str,
    forward_ratio: float,
    lateral_ratio: float,
    stage: str,
) -> dict[str, Any]:
    """Ray cast one ghost pose, run detector, then compute GT evidence."""
    width = float(anchor["estimated_corridor_width"])
    position = _local_to_world(anchor, forward_ratio, lateral_ratio)
    clearance = min(
        _point_segment_distance(position, wall) for wall in runner.geometry.walls
    )
    inside = bool(runner.geometry.contains(position))
    footprint_walkable = bool(runner.geometry.walkable(position, ROBOT_RADIUS))
    valid = inside and footprint_walkable and clearance >= ROBOT_RADIUS - 1.0e-9
    scan = runner.world.sensor.scan(
        runner.geometry, position, float(anchor["yaw_eval"])
    )
    margin = np.finfo(float).eps * max(1.0, scan.max_range) * 64.0
    snapshot = {
        "context": _candidate_id(forward_ratio, lateral_ratio),
        "angles": scan.angles_deg.copy(),
        "ranges": scan.ranges.copy(),
        "hit": scan.ranges < scan.max_range - margin,
        "max_range": scan.max_range,
        "position_eval": position.copy(),
        "yaw_eval": float(anchor["yaw_eval"]),
        "frame": anchor["frame"],
        "time": anchor["timestamp"],
        "orientation_deg": 0,
        "orientation_valid": True,
        "orientation_validity_reason": "fixed_validated_360_heading",
    }

    # Detector inference is complete before any branch GT/mouth query below.
    openings = list(
        detect_openings(snapshot["angles"].copy(), snapshot["ranges"].copy())
    )
    detector_summary, raw_opening_rows = evaluate_snapshot(runner, snapshot, openings)
    opening_rows, mean_error, mean_iou, matched = _opening_evaluation(
        runner, snapshot, raw_opening_rows
    )
    gt_axis_rows = _gt_visibility_rows(runner, snapshot)
    detector_branch_rows = _detector_branch_rows(runner, snapshot, opening_rows)
    mouth_fractions = {
        _branch_label(runner, branch_id): _mouth_visible_fraction(
            runner, position, branch_id, float(scan.max_range)
        )
        for branch_id in range(len(runner.geometry.branches))
    }
    axis = {row["branch_label_eval"]: row for row in gt_axis_rows}
    detected = {
        row["branch_label_eval"]: bool(row["detected_eval"])
        for row in detector_branch_rows
        if row["branch_id_eval"] != "UNMATCHED"
    }
    corridor_forward = np.asarray(anchor["corridor_forward"], dtype=float)
    corridor_world_deg = math.degrees(
        math.atan2(float(corridor_forward[1]), float(corridor_forward[0]))
    )
    corridor_body_deg = _normalize(corridor_world_deg - float(anchor["yaw_eval"]))
    unmatched = [
        row
        for row in opening_rows
        if row["matched_GT_branch_eval_only"] == ""
    ]
    lateral_false = sum(
        45.0
        <= abs(_normalize(float(row["center_angle_deg"]) - corridor_body_deg))
        <= 135.0
        for row in unmatched
    )
    distance_ratio = math.hypot(forward_ratio, lateral_ratio)
    return {
        "case": case,
        "candidate_id": snapshot["context"],
        "grid_stage": stage,
        "forward_offset": forward_ratio * width,
        "lateral_offset": lateral_ratio * width,
        "forward_ratio_W": forward_ratio,
        "lateral_ratio_W": lateral_ratio,
        "distance_ratio_W": distance_ratio,
        "candidate_valid": valid,
        "candidate_inside_free_space_eval": inside,
        "robot_footprint_walkable_eval": footprint_walkable,
        "robot_radius": ROBOT_RADIUS,
        "wall_clearance_eval": clearance,
        "straight_line_path_clear_eval": _path_clear(runner, anchor, position),
        "valid_lidar_hits": int(detector_summary["valid_lidar_point_count"]),
        "max_range_count": int(detector_summary["max_range_no_return_count"]),
        "plus90_axis_los_eval": bool(axis.get("PLUS90", {}).get("gt_visible_eval", False)),
        "minus90_axis_los_eval": bool(axis.get("MINUS90", {}).get("gt_visible_eval", False)),
        "forward_axis_los_eval": bool(axis.get("AXIAL_FORWARD", {}).get("gt_visible_eval", False)),
        "incoming_axis_los_eval": bool(axis["INCOMING"]["gt_visible_eval"]),
        "plus90_mouth_visible_fraction_eval": mouth_fractions.get("PLUS90", 0.0),
        "minus90_mouth_visible_fraction_eval": mouth_fractions.get("MINUS90", 0.0),
        "forward_mouth_visible_fraction_eval": mouth_fractions.get("AXIAL_FORWARD", 0.0),
        "plus90_detected_eval": detected.get("PLUS90", False),
        "minus90_detected_eval": detected.get("MINUS90", False),
        "forward_detected_eval": detected.get("AXIAL_FORWARD", False),
        "incoming_detected_eval": detected["INCOMING"],
        "opening_count": int(detector_summary["opening_count"]),
        "outgoing_match_count_eval": int(
            detector_summary["matched_outgoing_count_eval_only"]
        ),
        "outgoing_total_eval": int(
            detector_summary["GT_outgoing_branch_count_eval_only"]
        ),
        "false_opening_count_eval": int(
            detector_summary["false_opening_count_eval_only"]
        ),
        "lateral_false_opening_count_eval": int(lateral_false),
        "axial_false_opening_count_eval": int(len(unmatched) - lateral_false),
        "opening_center_error_eval": mean_error,
        "opening_IoU_eval": mean_iou,
        "wall_support_count": int(
            detector_summary["total_fitted_wall_point_count"]
        ),
        "tangent_support_count": int(
            detector_summary["wall_tangent_available_count"]
        ),
        "matched_branch_ids_eval": matched,
        "snapshot": snapshot,
        "opening_rows": opening_rows,
        "detector_openings": openings,
    }


def _state(row: dict[str, Any]) -> tuple[bool, ...]:
    """Return transition-relevant raw and detector visibility booleans."""
    return (
        bool(row["plus90_axis_los_eval"]),
        bool(row["minus90_axis_los_eval"]),
        float(row["plus90_mouth_visible_fraction_eval"]) > 0.0,
        float(row["minus90_mouth_visible_fraction_eval"]) > 0.0,
        bool(row["plus90_detected_eval"]),
        bool(row["minus90_detected_eval"]),
    )


def _transition_midpoints(
    rows: dict[tuple[float, float], dict[str, Any]], spacing: float
) -> set[tuple[float, float]]:
    """Return midpoints only where adjacent valid cells change visibility."""
    additions: set[tuple[float, float]] = set()
    for (forward, lateral), row in list(rows.items()):
        if not row["candidate_valid"]:
            continue
        for delta_forward, delta_lateral in ((spacing, 0.0), (0.0, spacing)):
            neighbor_key = (
                round(forward + delta_forward, 6),
                round(lateral + delta_lateral, 6),
            )
            neighbor = rows.get(neighbor_key)
            if neighbor is None or not neighbor["candidate_valid"]:
                continue
            if _state(row) != _state(neighbor):
                additions.add(
                    (
                        round(forward + delta_forward / 2.0, 6),
                        round(lateral + delta_lateral / 2.0, 6),
                    )
                )
    return additions - set(rows)


def _run_grid(runner: Any, anchor: dict[str, Any]) -> list[dict[str, Any]]:
    """Run coarse grid and refine only observed raw/detector transitions."""
    rows: dict[tuple[float, float], dict[str, Any]] = {}
    for forward in _grid_values(*FORWARD_LIMITS, COARSE_STEP):
        for lateral in _grid_values(*LATERAL_LIMITS, COARSE_STEP):
            rows[(forward, lateral)] = _probe(
                runner,
                anchor,
                "M1_CROSS_BASELINE",
                forward,
                lateral,
                "COARSE_0.10W",
            )
    fine = _transition_midpoints(rows, COARSE_STEP)
    for forward, lateral in sorted(fine):
        rows[(forward, lateral)] = _probe(
            runner,
            anchor,
            "M1_CROSS_BASELINE",
            forward,
            lateral,
            "REFINE_0.05W",
        )
    final = _transition_midpoints(rows, FINE_STEP)
    for forward, lateral in sorted(final):
        rows[(forward, lateral)] = _probe(
            runner,
            anchor,
            "M1_CROSS_BASELINE",
            forward,
            lateral,
            "REFINE_0.025W",
        )
    return sorted(
        rows.values(),
        key=lambda row: (row["forward_ratio_W"], row["lateral_ratio_W"]),
    )


def _nearest(
    rows: Iterable[dict[str, Any]], predicate: Any
) -> dict[str, Any] | None:
    """Return the nearest valid candidate satisfying an evidence predicate."""
    candidates = [row for row in rows if row["candidate_valid"] and predicate(row)]
    return min(
        candidates,
        key=lambda row: (
            row["distance_ratio_W"],
            row["forward_ratio_W"],
            abs(row["lateral_ratio_W"]),
        ),
        default=None,
    )


def _frontier_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize nearest raw-axis, raw-mouth, and detector-visible poses."""
    results = []
    for label, prefix in (("PLUS90", "plus90"), ("MINUS90", "minus90")):
        valid = [row for row in rows if row["candidate_valid"]]
        axis = _nearest(rows, lambda row, p=prefix: row[f"{p}_axis_los_eval"])
        mouth = _nearest(
            rows,
            lambda row, p=prefix: row[f"{p}_mouth_visible_fraction_eval"] > 0.0,
        )
        detector = _nearest(rows, lambda row, p=prefix: row[f"{p}_detected_eval"])
        raw_distance = (
            mouth["distance_ratio_W"]
            if mouth is not None
            else axis["distance_ratio_W"]
            if axis is not None
            else None
        )
        detector_distance = (
            detector["distance_ratio_W"] if detector is not None else None
        )
        visible_mouth = [
            row
            for row in valid
            if row[f"{prefix}_mouth_visible_fraction_eval"] > 0.0
        ]
        hidden_mouth = [
            row
            for row in valid
            if row[f"{prefix}_mouth_visible_fraction_eval"] <= 0.0
        ]
        maximum_mouth = max(
            visible_mouth,
            key=lambda row: row[f"{prefix}_mouth_visible_fraction_eval"],
            default=None,
        )
        results.append(
            {
                "branch": label,
                "nearest_raw_axis_los_candidate": "" if axis is None else axis["candidate_id"],
                "nearest_raw_axis_los_distance_W": "" if axis is None else axis["distance_ratio_W"],
                "nearest_raw_mouth_visible_candidate": "" if mouth is None else mouth["candidate_id"],
                "nearest_raw_mouth_visible_distance_W": "" if mouth is None else mouth["distance_ratio_W"],
                "nearest_raw_mouth_visible_fraction_eval": "" if mouth is None else mouth[f"{prefix}_mouth_visible_fraction_eval"],
                "maximum_raw_mouth_visible_candidate": "" if maximum_mouth is None else maximum_mouth["candidate_id"],
                "maximum_raw_mouth_visible_fraction_eval": "" if maximum_mouth is None else maximum_mouth[f"{prefix}_mouth_visible_fraction_eval"],
                "raw_mouth_visible_valid_candidate_count": len(visible_mouth),
                "raw_mouth_hidden_valid_candidate_count": len(hidden_mouth),
                "raw_mouth_exposure_preexists_at_A0": bool(
                    mouth is not None and math.isclose(mouth["distance_ratio_W"], 0.0)
                ),
                "raw_mouth_transition_frontier_found": bool(visible_mouth and hidden_mouth),
                "nearest_detector_visible_candidate": "" if detector is None else detector["candidate_id"],
                "nearest_detector_visible_distance_W": "" if detector is None else detector_distance,
                "raw_detector_gap_W": (
                    ""
                    if raw_distance is None or detector_distance is None
                    else detector_distance - raw_distance
                ),
                "raw_axis_frontier_found": axis is not None,
                "raw_frontier_found": axis is not None or bool(visible_mouth and hidden_mouth),
                "raw_mouth_exposure_exists": bool(visible_mouth),
                "detector_frontier_found": detector is not None,
            }
        )
    return results


def _pair_rows(
    rows: list[dict[str, Any]], anchor_ids: set[int]
) -> list[dict[str, Any]]:
    """Evaluate branch-set unions and two explicit evaluation-only costs."""
    valid = [row for row in rows if row["candidate_valid"]]
    origin = np.zeros(2)
    results = []
    for first, second in itertools.combinations(valid, 2):
        first_xy = np.array([first["forward_ratio_W"], first["lateral_ratio_W"]])
        second_xy = np.array([second["forward_ratio_W"], second["lateral_ratio_W"]])
        union = set(first["matched_branch_ids_eval"]) | set(second["matched_branch_ids_eval"])
        route_cost = float(np.linalg.norm(first_xy - origin) + np.linalg.norm(second_xy - first_xy))
        star_cost = float(np.linalg.norm(first_xy - origin) + np.linalg.norm(second_xy - origin))
        results.append(
            {
                "candidate_1": first["candidate_id"],
                "candidate_2": second["candidate_id"],
                "candidate_1_forward_ratio_W": first["forward_ratio_W"],
                "candidate_1_lateral_ratio_W": first["lateral_ratio_W"],
                "candidate_2_forward_ratio_W": second["forward_ratio_W"],
                "candidate_2_lateral_ratio_W": second["lateral_ratio_W"],
                "combined_detected_branch_ids_eval": json.dumps(sorted(union)),
                "combined_detected_branch_count_eval": len(union),
                "two_view_3of3_eval": len(union) == 3,
                "route_cost_A0_to_P1_to_P2_W_eval": route_cost,
                "star_cost_A0_to_P1_plus_A0_to_P2_W_eval": star_cost,
                "evaluation_cost": route_cost,
                "cost_definition": "distance(A0,P1)+distance(P1,P2)",
                "combined_false_opening_count_eval": first["false_opening_count_eval"] + second["false_opening_count_eval"],
                "A0_union_count_eval": len(anchor_ids | union),
            }
        )
    return results


def _nearest_rows(
    rows: list[dict[str, Any]], pairs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Create compact single- and two-view useful-pose results."""
    plus = _nearest(rows, lambda row: row["plus90_detected_eval"])
    minus = _nearest(rows, lambda row: row["minus90_detected_eval"])
    single = _nearest(rows, lambda row: row["outgoing_match_count_eval"] == 3)
    successful_pairs = [row for row in pairs if row["two_view_3of3_eval"]]
    best_pair = min(successful_pairs, key=lambda row: row["evaluation_cost"], default=None)
    return [
        {
            "result_type": "NEAREST_PLUS90_DETECTOR",
            "candidate_1": "" if plus is None else plus["candidate_id"],
            "candidate_2": "",
            "distance_or_cost_W": "" if plus is None else plus["distance_ratio_W"],
            "exists": plus is not None,
        },
        {
            "result_type": "NEAREST_MINUS90_DETECTOR",
            "candidate_1": "" if minus is None else minus["candidate_id"],
            "candidate_2": "",
            "distance_or_cost_W": "" if minus is None else minus["distance_ratio_W"],
            "exists": minus is not None,
        },
        {
            "result_type": "NEAREST_SINGLE_VIEW_3OF3",
            "candidate_1": "" if single is None else single["candidate_id"],
            "candidate_2": "",
            "distance_or_cost_W": "" if single is None else single["distance_ratio_W"],
            "exists": single is not None,
        },
        {
            "result_type": "BEST_TWO_VIEW_3OF3",
            "candidate_1": "" if best_pair is None else best_pair["candidate_1"],
            "candidate_2": "" if best_pair is None else best_pair["candidate_2"],
            "distance_or_cost_W": "" if best_pair is None else best_pair["evaluation_cost"],
            "exists": best_pair is not None,
        },
    ]


def _opening_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten detected opening groups with post-hoc match/IoU fields."""
    results = []
    for candidate in rows:
        for opening in candidate["opening_rows"]:
            results.append(
                {
                    "case": candidate["case"],
                    "candidate_id": candidate["candidate_id"],
                    "candidate_valid": candidate["candidate_valid"],
                    "forward_ratio_W": candidate["forward_ratio_W"],
                    "lateral_ratio_W": candidate["lateral_ratio_W"],
                    "opening_id": opening["opening_id"],
                    "start_angle_deg": opening["start_angle_deg"],
                    "end_angle_deg": opening["end_angle_deg"],
                    "center_angle_deg": opening["center_angle_deg"],
                    "angular_width_deg": opening["angular_width_deg"],
                    "confidence": opening["confidence"],
                    "matched_GT_branch_eval_only": opening[
                        "matched_GT_branch_eval_only"
                    ],
                    "center_error_deg_eval_only": opening[
                        "center_error_deg_eval_only"
                    ],
                    "GT_mouth_IoU_eval_only": opening["GT_mouth_IoU_eval_only"],
                    "wall_support": opening["fitted_wall_point_count"],
                    "tangent_support": opening["wall_tangent_deg"] != "",
                }
            )
    return results


def _public_grid(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip large arrays and internal branch sets from candidate CSV rows."""
    excluded = {"matched_branch_ids_eval", "snapshot", "opening_rows", "detector_openings"}
    return [{key: value for key, value in row.items() if key not in excluded} for row in rows]


def _branch_grid(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select branch-specific raw/detector evidence requested by EXP-030."""
    fields = (
        "case",
        "candidate_id",
        "grid_stage",
        "forward_ratio_W",
        "lateral_ratio_W",
        "distance_ratio_W",
        "candidate_valid",
        "plus90_axis_los_eval",
        "minus90_axis_los_eval",
        "plus90_mouth_visible_fraction_eval",
        "minus90_mouth_visible_fraction_eval",
        "plus90_detected_eval",
        "minus90_detected_eval",
        "forward_detected_eval",
        "incoming_detected_eval",
        "opening_count",
        "outgoing_match_count_eval",
        "outgoing_total_eval",
        "false_opening_count_eval",
        "axial_false_opening_count_eval",
        "lateral_false_opening_count_eval",
    )
    return [{field: row[field] for field in fields} for row in rows]


def _m0_rows(runner: Any, anchor: dict[str, Any]) -> list[dict[str, Any]]:
    """Probe only representative straight-corridor negative-control poses."""
    return [
        _probe(runner, anchor, "M0_STRAIGHT", forward, lateral, "M0_REPRESENTATIVE")
        for forward, lateral in M0_POINTS
    ]


def _verdict(
    rows: list[dict[str, Any]], pairs: list[dict[str, Any]], baseline_false: int
) -> tuple[str, list[str]]:
    """Apply EXP-030 frontier verdicts with false-gain protection."""
    valid = [row for row in rows if row["candidate_valid"]]
    invalid = [row for row in rows if not row["candidate_valid"]]
    valid_raw = [
        row
        for row in valid
        if row["plus90_axis_los_eval"]
        or row["minus90_axis_los_eval"]
        or row["plus90_mouth_visible_fraction_eval"] > 0.0
        or row["minus90_mouth_visible_fraction_eval"] > 0.0
    ]
    invalid_raw = [
        row
        for row in invalid
        if row["plus90_axis_los_eval"]
        or row["minus90_axis_los_eval"]
        or row["plus90_mouth_visible_fraction_eval"] > 0.0
        or row["minus90_mouth_visible_fraction_eval"] > 0.0
    ]
    false_dominated = any(
        row["outgoing_match_count_eval"] > 1
        and row["false_opening_count_eval"] > baseline_false
        for row in valid
    )
    clean_single = any(
        row["outgoing_match_count_eval"] == 3
        and row["false_opening_count_eval"] <= baseline_false
        for row in valid
    )
    clean_pair = any(
        row["two_view_3of3_eval"]
        and row["combined_false_opening_count_eval"] <= baseline_false * 2
        for row in pairs
    )
    side_detector = any(
        row["plus90_detected_eval"] or row["minus90_detected_eval"] for row in valid
    )
    if false_dominated:
        primary = "F_FALSE_OPENING_DOMINATED_GAIN"
    elif clean_single:
        primary = "A_SINGLE_VIEW_3OF3_FRONTIER_FOUND"
    elif clean_pair:
        primary = "B_TWO_SIDED_MULTIVIEW_FRONTIERS_FOUND"
    elif valid_raw and not side_detector:
        primary = "C_RAW_VISIBILITY_FRONTIER_FOUND_DETECTOR_NOT_YET_VISIBLE"
    elif not valid_raw and invalid_raw:
        primary = "E_VISIBILITY_EXISTS_ONLY_AT_INVALID_POSE"
    else:
        primary = "D_NO_VISIBILITY_FRONTIER_IN_SEARCH_REGION"
    secondary = []
    if valid_raw:
        secondary.append("VALID_RAW_SIDE_EXPOSURE_EXISTS")
    a0 = next(
        (
            row
            for row in valid
            if math.isclose(row["distance_ratio_W"], 0.0)
        ),
        None,
    )
    if a0 is not None and (
        a0["plus90_mouth_visible_fraction_eval"] > 0.0
        or a0["minus90_mouth_visible_fraction_eval"] > 0.0
    ):
        secondary.append("RAW_MOUTH_EXPOSURE_PREEXISTS_AT_A0")
    if not any(
        row["plus90_axis_los_eval"] or row["minus90_axis_los_eval"] for row in valid
    ):
        secondary.append("NO_RAW_SIDE_AXIS_FRONTIER")
    if side_detector:
        secondary.append("VALID_SIDE_DETECTOR_REGION_EXISTS")
    if invalid_raw:
        secondary.append("INVALID_POSES_HAVE_RAW_EXPOSURE")
    if not secondary:
        secondary.append("NO_SIDE_EXPOSURE_OBSERVED")
    return primary, secondary


def _plot(
    path: Path,
    rows: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
) -> None:
    """Render the 2D sensing frontier in Anchor-local normalized coordinates."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    valid = [row for row in rows if row["candidate_valid"]]
    invalid = [row for row in rows if not row["candidate_valid"]]
    scatter = axes[0, 0].scatter(
        [row["lateral_ratio_W"] for row in valid],
        [row["forward_ratio_W"] for row in valid],
        c=[row["outgoing_match_count_eval"] for row in valid],
        vmin=0,
        vmax=3,
        cmap="viridis",
        s=38,
        label="valid ghost pose",
    )
    axes[0, 0].scatter(
        [row["lateral_ratio_W"] for row in invalid],
        [row["forward_ratio_W"] for row in invalid],
        marker="x",
        color="lightgray",
        s=25,
        label="invalid footprint",
    )
    for prefix, color, marker in (
        ("plus90", "tab:red", "^"),
        ("minus90", "tab:blue", "v"),
    ):
        raw = [row for row in valid if row[f"{prefix}_axis_los_eval"]]
        detected = [row for row in valid if row[f"{prefix}_detected_eval"]]
        axes[0, 0].scatter(
            [row["lateral_ratio_W"] for row in raw],
            [row["forward_ratio_W"] for row in raw],
            facecolors="none",
            edgecolors=color,
            marker=marker,
            s=85,
            label=f"{prefix} axis LOS",
        )
        axes[0, 0].scatter(
            [row["lateral_ratio_W"] for row in detected],
            [row["forward_ratio_W"] for row in detected],
            color=color,
            marker="*",
            s=120,
            label=f"{prefix} detector",
        )
    axes[0, 0].scatter([0], [0], marker="P", color="black", s=100, label="A0")
    axes[0, 0].set(
        title="EXP-030 2D outgoing/side visibility frontier",
        xlabel="lateral / W",
        ylabel="forward / W",
        xlim=(LATERAL_LIMITS[0] - 0.04, LATERAL_LIMITS[1] + 0.04),
        ylim=(FORWARD_LIMITS[0] - 0.04, FORWARD_LIMITS[1] + 0.04),
        aspect="equal",
    )
    fig.colorbar(scatter, ax=axes[0, 0], label="detected outgoing count")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    axes[0, 0].grid(alpha=0.2)

    for axis, prefix, title in (
        (axes[0, 1], "plus90", "+90 raw mouth exposure"),
        (axes[1, 0], "minus90", "-90 raw mouth exposure"),
    ):
        values = axis.scatter(
            [row["lateral_ratio_W"] for row in valid],
            [row["forward_ratio_W"] for row in valid],
            c=[row[f"{prefix}_mouth_visible_fraction_eval"] for row in valid],
            vmin=0,
            vmax=1,
            cmap="magma",
            s=42,
        )
        axis.scatter(
            [row["lateral_ratio_W"] for row in invalid],
            [row["forward_ratio_W"] for row in invalid],
            marker="x",
            color="lightgray",
            s=25,
        )
        axis.set(
            title=title,
            xlabel="lateral / W",
            ylabel="forward / W",
            aspect="equal",
        )
        axis.grid(alpha=0.2)
        fig.colorbar(values, ax=axis, label="visible mouth fraction")

    axes[1, 1].axis("off")
    text = [
        "Frontier summary",
        f"robot radius = {ROBOT_RADIUS:.3f}",
        f"coarse/fine/final = {COARSE_STEP}/{FINE_STEP}/{FINAL_STEP} W",
    ]
    for item in frontier:
        text.append(
            f"{item['branch']}: mouth={item['nearest_raw_mouth_visible_candidate'] or 'none'}, "
            f"axis={item['nearest_raw_axis_los_candidate'] or 'none'}, "
            f"detector={item['nearest_detector_visible_candidate'] or 'none'}"
        )
    axes[1, 1].text(0.02, 0.96, "\n".join(text), va="top", family="monospace")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _deterministic_replay(
    runner: Any, anchor: dict[str, Any], rows: list[dict[str, Any]]
) -> bool:
    """Replay every chosen ghost pose and require exact scan/results."""
    for row in rows:
        repeated = _probe(
            runner,
            anchor,
            row["case"],
            float(row["forward_ratio_W"]),
            float(row["lateral_ratio_W"]),
            row["grid_stage"],
        )
        if not np.array_equal(
            row["snapshot"]["ranges"], repeated["snapshot"]["ranges"]
        ):
            return False
        keys = (
            "candidate_valid",
            "plus90_axis_los_eval",
            "minus90_axis_los_eval",
            "plus90_mouth_visible_fraction_eval",
            "minus90_mouth_visible_fraction_eval",
            "plus90_detected_eval",
            "minus90_detected_eval",
            "outgoing_match_count_eval",
            "false_opening_count_eval",
        )
        if any(row[key] != repeated[key] for key in keys):
            return False
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-anchor-frames", type=int, default=120)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    m1, anchor = _acquire_m1_anchor(args.max_anchor_frames)
    anchor["corridor_forward"] = m1.world.trusted_corridor_forward.copy()
    rows = _run_grid(m1, anchor)
    a0 = next(
        row
        for row in rows
        if math.isclose(row["forward_ratio_W"], 0.0)
        and math.isclose(row["lateral_ratio_W"], 0.0)
    )
    actual_a0_ranges = np.asarray(anchor["ranges"], dtype=float)
    a0_scan_exact = np.array_equal(a0["snapshot"]["ranges"], actual_a0_ranges)
    actual_a0_openings = list(
        detect_openings(np.asarray(anchor["angles_deg"]), actual_a0_ranges)
    )
    a0_detector_exact = a0["detector_openings"] == actual_a0_openings

    frontier = _frontier_summary(rows)
    pairs = _pair_rows(rows, set(a0["matched_branch_ids_eval"]))
    nearest = _nearest_rows(rows, pairs)
    m0, m0_anchor = _acquire_m0_snapshot(int(anchor["frame"]) + 1)
    m0_rows = _m0_rows(m0, m0_anchor)
    baseline_false = a0["false_opening_count_eval"]
    primary, secondary = _verdict(rows, pairs, baseline_false)
    deterministic = _deterministic_replay(m1, anchor, rows) and _deterministic_replay(
        m0, m0_anchor, m0_rows
    )

    transform_errors = []
    normalization_errors = []
    for row in rows:
        recovered = _world_to_local(anchor, row["snapshot"]["position_eval"])
        transform_errors.append(
            max(
                abs(recovered[0] - row["forward_ratio_W"]),
                abs(recovered[1] - row["lateral_ratio_W"]),
            )
        )
        expected = math.hypot(row["forward_offset"], row["lateral_offset"]) / float(
            anchor["estimated_corridor_width"]
        )
        normalization_errors.append(abs(expected - row["distance_ratio_W"]))

    valid_rows = [row for row in rows if row["candidate_valid"]]
    invalid_rows = [row for row in rows if not row["candidate_valid"]]
    clean_single = _nearest(
        rows,
        lambda row: row["outgoing_match_count_eval"] == 3
        and row["false_opening_count_eval"] <= baseline_false,
    )
    successful_pairs = [row for row in pairs if row["two_view_3of3_eval"]]
    best_pair = min(successful_pairs, key=lambda row: row["evaluation_cost"], default=None)
    best_single_score = max(
        (row["outgoing_match_count_eval"] for row in valid_rows), default=0
    )
    best_single = min(
        (
            row
            for row in valid_rows
            if row["outgoing_match_count_eval"] == best_single_score
        ),
        key=lambda row: row["distance_ratio_W"],
    )
    m0_baseline = next(
        row
        for row in m0_rows
        if math.isclose(row["forward_ratio_W"], 0.0)
        and math.isclose(row["lateral_ratio_W"], 0.0)
    )
    m0_false_regression = any(
        row["candidate_valid"]
        and row["false_opening_count_eval"] > m0_baseline["false_opening_count_eval"]
        for row in m0_rows
    )
    m0_lateral_false = any(
        row["candidate_valid"]
        and row["lateral_false_opening_count_eval"] > 0
        for row in m0_rows
    )
    verdict_row = {
        "experiment_id": EXPERIMENT_ID,
        "primary_verdict": primary,
        "secondary_findings": json.dumps(secondary),
        "search_forward_min_W": FORWARD_LIMITS[0],
        "search_forward_max_W": FORWARD_LIMITS[1],
        "search_lateral_min_W": LATERAL_LIMITS[0],
        "search_lateral_max_W": LATERAL_LIMITS[1],
        "coarse_step_W": COARSE_STEP,
        "fine_step_W": FINE_STEP,
        "final_step_W": FINAL_STEP,
        "estimated_corridor_width": anchor["estimated_corridor_width"],
        "robot_radius": ROBOT_RADIUS,
        "candidate_count": len(rows),
        "valid_candidate_count": len(valid_rows),
        "invalid_candidate_count": len(invalid_rows),
        "single_view_3of3_exists": clean_single is not None,
        "nearest_single_view_3of3_candidate": "" if clean_single is None else clean_single["candidate_id"],
        "nearest_single_view_3of3_distance_W": "" if clean_single is None else clean_single["distance_ratio_W"],
        "best_single_view_candidate": best_single["candidate_id"],
        "best_single_view_outgoing_count_eval": best_single_score,
        "two_view_3of3_exists": best_pair is not None,
        "best_two_view_candidate_1": "" if best_pair is None else best_pair["candidate_1"],
        "best_two_view_candidate_2": "" if best_pair is None else best_pair["candidate_2"],
        "best_two_view_evaluation_cost_W": "" if best_pair is None else best_pair["evaluation_cost"],
        "M0_false_opening_regression": m0_false_regression,
        "M0_lateral_false_branch": m0_lateral_false,
        "A0_ghost_range_exact": a0_scan_exact,
        "A0_detector_opening_count_exact": a0_detector_exact,
        "deterministic_replay": deterministic,
        "max_local_transform_error": max(transform_errors, default=0.0),
        "max_width_normalization_error": max(normalization_errors, default=0.0),
        "actual_swarm_movement_performed": False,
        "detector_threshold_changed": False,
        "GT_used_for_candidate_generation": False,
        "GT_map_used_for_evaluation_only": True,
    }

    _write_csv(args.output / "viewpoint_grid.csv", _public_grid(rows))
    _write_csv(args.output / "branch_visibility_grid.csv", _branch_grid(rows))
    _write_csv(args.output / "viewpoint_openings.csv", _opening_rows([*rows, *m0_rows]))
    _write_csv(args.output / "visibility_frontier_summary.csv", frontier)
    _write_csv(args.output / "nearest_useful_viewpoints.csv", nearest)
    _write_csv(args.output / "multiview_pair_eval.csv", pairs)
    _write_csv(args.output / "visibility_frontier_verdict.csv", [verdict_row])
    _write_csv(args.output / "m0_negative_control.csv", _branch_grid(m0_rows))
    _plot(args.output / "visibility_frontier_map.png", rows, frontier)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "primary_verdict": primary,
                "secondary_findings": secondary,
                "width": anchor["estimated_corridor_width"],
                "candidate_count": len(rows),
                "valid": len(valid_rows),
                "invalid": len(invalid_rows),
                "frontier": frontier,
                "best_single_outgoing": best_single_score,
                "single_3of3": clean_single is not None,
                "two_view_3of3": best_pair is not None,
                "M0_false_regression": m0_false_regression,
                "A0_scan_exact": a0_scan_exact,
                "deterministic": deterministic,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
