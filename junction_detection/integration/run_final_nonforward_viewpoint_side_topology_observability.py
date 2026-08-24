"""EXP-035: final single-view non-forward side-topology diagnostic.

This script replays the existing EXP-030 ghost-viewpoint table through the
unchanged deterministic LiDAR ray caster and the frozen EXP-033 wall-topology
helpers.  It does not move a robot, create a new viewpoint grid, or tune any
detector/topology threshold.  Map/branch ground truth is attached only after
the scan-only gap candidates have been accepted, for evaluation and naming.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.integration.run_2d_viewpoint_visibility_frontier import (
    _local_to_world,
    _world_to_local,
)
from junction_detection.integration.run_forward_viewpoint_wall_topology_transition import (
    SAFE_SOURCE,
    _local_width,
    _read,
    _rear_start_geometry,
    _side_values,
)
from junction_detection.integration.run_nonforward_viewpoint_magnitude_boundary import (
    _point_segment_distance,
)
from junction_detection.integration.run_wall_topology_branch_opening_diagnostic import (
    _analyze,
    _branch_topology_eval,
    _gt_mouths_eval,
    _match_candidates_eval,
    _plot_result,
    _self_test,
)
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import (
    detect_openings,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import (
    GeometryBuilder,
    LidarSensor,
)

EXPERIMENT_ID = "EXP-035"
DEFAULT_SOURCE = (
    ROOT
    / "junction_detection/integration/output/2d_viewpoint_visibility_frontier/viewpoint_grid.csv"
)
DEFAULT_EXP033 = (
    ROOT
    / "junction_detection/integration/output/wall_topology_branch_opening/wall_topology_summary.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "junction_detection/integration/output/final_nonforward_side_topology_observability"
)
TOPOLOGY_LEVEL = {
    "NO_GAP_TOPOLOGY": 0,
    "PARTIAL_GAP_TOPOLOGY": 1,
    "COMPLETE_GAP_TOPOLOGY": 2,
}
SIDE_LABELS = ("LEFT", "RIGHT")


def _as_bool(value: Any) -> bool:
    """Parse the stable boolean spelling used by the existing CSV outputs."""
    return str(value).strip().lower() == "true"


def _write_required(
    path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None
) -> None:
    """Write a required CSV, retaining a header even when it has zero rows."""
    if fields is None:
        fields = list(rows[0]) if rows else []
        for row in rows[1:]:
            fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _anchor_from_existing_source(safe_source: Path) -> dict[str, Any]:
    """Recover the frozen A0 local frame from an already persisted scan row."""
    rows = _read(safe_source)
    if not rows:
        raise RuntimeError(f"empty A0 source: {safe_source}")
    a0 = rows[0]
    width = _local_width(rows)
    yaw = float(a0["anchor_yaw_eval_only"])
    yaw_rad = math.radians(yaw)
    return {
        "position_eval": np.array(
            [float(a0["anchor_x_eval_only"]), float(a0["anchor_y_eval_only"])],
            dtype=float,
        ),
        "yaw_eval": yaw,
        "estimated_corridor_width": width,
        # This is the persisted local body/corridor orientation, not GT heading.
        "corridor_forward": np.array([math.cos(yaw_rad), math.sin(yaw_rad)]),
    }


def _viewpoint_id(row: dict[str, str]) -> str:
    """Name the zero-offset baseline A0 and otherwise retain EXP-030 IDs."""
    forward = float(row["forward_ratio_W"])
    lateral = float(row["lateral_ratio_W"])
    return "A0" if math.isclose(forward, 0.0) and math.isclose(lateral, 0.0) else row["candidate_id"]


def _direction_class(forward: float, lateral: float) -> str:
    """Classify a pose using the frozen local forward/left sign convention."""
    eps = 1.0e-12
    if abs(forward) <= eps and abs(lateral) <= eps:
        return "A0"
    longitudinal = "FORWARD" if forward > eps else "BACKWARD" if forward < -eps else "PURE"
    side = "LOCAL_LEFT" if lateral > eps else "LOCAL_RIGHT" if lateral < -eps else "CENTERLINE"
    return f"{longitudinal}_{side}"


def _snapshot(
    sensor: LidarSensor,
    geometry: Any,
    position: np.ndarray,
    yaw: float,
    context: str,
) -> tuple[dict[str, Any], int, int, int]:
    """Reconstruct one deterministic 360-beam scan with existing components."""
    scan = sensor.scan(geometry, position, yaw)
    margin = np.finfo(float).eps * max(1.0, scan.max_range) * 64.0
    hit = scan.ranges < scan.max_range - margin
    snapshot = {
        "context": context,
        "angles": scan.angles_deg.copy(),
        "ranges": scan.ranges.copy(),
        "hit": hit,
        "max_range": scan.max_range,
        "position_eval": np.asarray(position, dtype=float).copy(),
        "yaw_eval": yaw,
    }
    openings = detect_openings(snapshot["angles"].copy(), snapshot["ranges"].copy())
    return snapshot, int(np.count_nonzero(hit)), int(np.count_nonzero(~hit)), len(openings)


def _analyze_viewpoint(
    source: dict[str, str], anchor: dict[str, Any], sensor: LidarSensor, geometry: Any
) -> dict[str, Any]:
    """Replay one valid source pose and apply frozen EXP-033 topology helpers."""
    forward = float(source["forward_ratio_W"])
    lateral = float(source["lateral_ratio_W"])
    position = _local_to_world(anchor, forward, lateral)
    viewpoint = _viewpoint_id(source)
    snapshot, hit_count, max_count, opening_count = _snapshot(
        sensor, geometry, position, float(anchor["yaw_eval"]), viewpoint
    )
    result = _analyze(viewpoint, snapshot, float(anchor["estimated_corridor_width"]))

    # The following GT mouth construction/matching is strictly post-hoc.  The
    # scan-only segmentation, endpoint validity, gap width, and continuation
    # acceptance above have already completed and are never changed by GT.
    runner = SimpleNamespace(geometry=geometry)
    mouths = _gt_mouths_eval(runner, snapshot)
    matches = _match_candidates_eval(
        viewpoint,
        result["gaps"],
        result["endpoints"],
        mouths,
        float(anchor["estimated_corridor_width"]),
    )
    branches = _branch_topology_eval(
        runner,
        snapshot,
        result["endpoints"],
        matches,
        float(anchor["estimated_corridor_width"]),
    )
    analysis = {
        "spec": {"viewpoint_id": viewpoint, "estimated_corridor_width": float(anchor["estimated_corridor_width"])},
        "source": source,
        "position": position,
        "snapshot": snapshot,
        "result": result,
        "mouths": mouths,
        "matches": matches,
        "branches": branches,
        "hit_count": hit_count,
        "max_count": max_count,
        "opening_count": opening_count,
    }
    analysis["left"] = _side_values(analysis, "LEFT")
    analysis["right"] = _side_values(analysis, "RIGHT")
    return analysis


def _validity_audit(
    row: dict[str, str], anchor: dict[str, Any], geometry: Any
) -> dict[str, Any]:
    """Recompute EXP-030 pose/footprint validity without changing its result."""
    forward = float(row["forward_ratio_W"])
    lateral = float(row["lateral_ratio_W"])
    position = _local_to_world(anchor, forward, lateral)
    robot_radius = float(row["robot_radius"])
    clearance = min(_point_segment_distance(position, wall) for wall in geometry.walls)
    inside = bool(geometry.contains(position))
    footprint = bool(geometry.walkable(position, robot_radius))
    valid = inside and footprint and clearance >= robot_radius - 1.0e-9
    roundtrip = _world_to_local(anchor, position)
    return {
        "position": position,
        "inside": inside,
        "footprint": footprint,
        "clearance": clearance,
        "valid": valid,
        "roundtrip_error": math.hypot(roundtrip[0] - forward, roundtrip[1] - lateral),
        "consistent": (
            valid == _as_bool(row["candidate_valid"])
            and inside == _as_bool(row["candidate_inside_free_space_eval"])
            and footprint == _as_bool(row["robot_footprint_walkable_eval"])
            and math.isclose(clearance, float(row["wall_clearance_eval"]), abs_tol=1.0e-9)
        ),
    }


def _branch_row(analysis: dict[str, Any], label: str) -> dict[str, Any]:
    """Select one frozen post-hoc outgoing branch topology row."""
    return next(row for row in analysis["branches"] if row["branch_eval"] == label)


def _topology_row(analysis: dict[str, Any]) -> dict[str, Any]:
    """Serialize one analyzed viewpoint with side and global gap evidence."""
    source = analysis["source"]
    left, right = analysis["left"], analysis["right"]
    forward_branch = _branch_row(analysis, "FORWARD")
    accepted = [gap for gap in analysis["result"]["gaps"] if gap["candidate_valid"]]
    matched_labels = [row["matched_branch_eval"] for row in analysis["matches"]]
    false_count = sum(bool(row["false_positive_eval"]) for row in analysis["matches"])
    true_side = sum(label in SIDE_LABELS for label in matched_labels)
    incoming = sum(label == "INCOMING" for label in matched_labels)
    forward = float(source["forward_ratio_W"])
    lateral = float(source["lateral_ratio_W"])
    left_branch = _branch_row(analysis, "LEFT")
    right_branch = _branch_row(analysis, "RIGHT")
    return {
        "viewpoint_id": analysis["spec"]["viewpoint_id"],
        "grid_stage": source["grid_stage"],
        "forward_ratio_W": forward,
        "lateral_ratio_W": lateral,
        "distance_ratio_W": math.hypot(forward, lateral),
        "direction_class": _direction_class(forward, lateral),
        "viewpoint_geometry_valid": True,
        "movement_feasibility_proven": False,
        "left_topology": left["topology"],
        "right_topology": right["topology"],
        "left_observed_mouth_boundary_count_eval": left_branch["observed_mouth_boundary_count_eval"],
        "right_observed_mouth_boundary_count_eval": right_branch["observed_mouth_boundary_count_eval"],
        "left_near_boundary_visible_eval": left["near_visible"],
        "left_far_boundary_visible_eval": left["far_visible"],
        "left_near_endpoint_error_eval": left["near_error"],
        "left_far_endpoint_error_eval": left["far_error"],
        "right_near_boundary_visible_eval": right["near_visible"],
        "right_far_boundary_visible_eval": right["far_visible"],
        "right_near_endpoint_error_eval": right["near_error"],
        "right_far_endpoint_error_eval": right["far_error"],
        "left_valid_endpoint_pair": left["match"] is not None,
        "right_valid_endpoint_pair": right["match"] is not None,
        "left_valid_gap": left["gap_valid"],
        "right_valid_gap": right["gap_valid"],
        "left_gt_match_eval": left["gt_match"],
        "right_gt_match_eval": right["gt_match"],
        "forward_topology": forward_branch["topology_class_eval"],
        "incoming_topology": "COMPLETE_GAP_TOPOLOGY" if incoming else "NO_GAP_TOPOLOGY",
        "wall_segment_count": len(analysis["result"]["segments"]),
        "valid_termination_count": sum(bool(endpoint["valid"]) for endpoint in analysis["result"]["endpoints"]),
        "accepted_gap_count": len(accepted),
        "incoming_true_gap_count": incoming,
        "side_true_gap_count": true_side,
        "false_accepted_gap_count": false_count,
        "angular_detector_outgoing_match_eval": int(source["outgoing_match_count_eval"]),
        "both_side_complete": left["topology"] == "COMPLETE_GAP_TOPOLOGY" and right["topology"] == "COMPLETE_GAP_TOPOLOGY",
    }


def _complete_rows(analyses: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """List branch-specific COMPLETE results at valid non-forward viewpoints."""
    summary_by_id = {row["viewpoint_id"]: row for row in summaries}
    rows: list[dict[str, Any]] = []
    for analysis in analyses:
        summary = summary_by_id[analysis["spec"]["viewpoint_id"]]
        if math.isclose(float(summary["lateral_ratio_W"]), 0.0):
            continue
        for label, side in (("LEFT", analysis["left"]), ("RIGHT", analysis["right"])):
            if side["topology"] != "COMPLETE_GAP_TOPOLOGY":
                continue
            candidate, match = side["candidate"], side["match"]
            if candidate is None or match is None:
                raise RuntimeError("COMPLETE topology lacks an accepted scan-only candidate")
            rows.append(
                {
                    "viewpoint_id": summary["viewpoint_id"],
                    "branch_eval": label,
                    "forward_ratio_W": summary["forward_ratio_W"],
                    "lateral_ratio_W": summary["lateral_ratio_W"],
                    "distance_ratio_W": summary["distance_ratio_W"],
                    "direction_class": summary["direction_class"],
                    "gap_width": candidate["gap_width"],
                    "gap_width_ratio_W": candidate["gap_width_ratio_W"],
                    "free_continuation_depth": candidate["continuation_depth"],
                    "center_error_eval": match["center_error_eval"],
                    "endpoint_error_eval": match["endpoint_error_eval"],
                    "mouth_overlap_eval": match["mouth_overlap_eval"],
                    "false_gap_count": summary["false_accepted_gap_count"],
                }
            )
    return sorted(rows, key=lambda row: (row["branch_eval"], row["distance_ratio_W"], row["viewpoint_id"]))


def _complementary_rows(complete: list[dict[str, Any]], has_both: bool) -> list[dict[str, Any]]:
    """Rank distinct LEFT/RIGHT complete viewpoints without fusing scans."""
    if has_both:
        return []
    left = [row for row in complete if row["branch_eval"] == "LEFT"]
    right = [row for row in complete if row["branch_eval"] == "RIGHT"]
    pairs = []
    for first in left:
        for second in right:
            if first["viewpoint_id"] == second["viewpoint_id"]:
                continue
            pairs.append(
                {
                    "left_viewpoint_id": first["viewpoint_id"],
                    "right_viewpoint_id": second["viewpoint_id"],
                    "left_F_W": first["forward_ratio_W"],
                    "left_L_W": first["lateral_ratio_W"],
                    "left_distance_W": first["distance_ratio_W"],
                    "right_F_W": second["forward_ratio_W"],
                    "right_L_W": second["lateral_ratio_W"],
                    "right_distance_W": second["distance_ratio_W"],
                    "combined_branch_coverage": "LEFT+RIGHT",
                    "pair_distance_sum_W": first["distance_ratio_W"] + second["distance_ratio_W"],
                    "pair_max_distance_W": max(first["distance_ratio_W"], second["distance_ratio_W"]),
                }
            )
    return sorted(pairs, key=lambda row: (row["pair_max_distance_W"], row["pair_distance_sum_W"]))


def _source_audit_row(
    source_path: Path,
    source: dict[str, str],
    validity: dict[str, Any],
    analysis: dict[str, Any] | None,
    yaw_deg: float,
) -> dict[str, Any]:
    """Record source provenance, local transform, validity, and replay checks."""
    saved_hit = int(source["valid_lidar_hits"])
    saved_max = int(source["max_range_count"])
    saved_opening = int(source["opening_count"])
    return {
        "viewpoint_id": _viewpoint_id(source),
        "source_experiment": "EXP-030",
        "source_file": str(source_path.resolve()),
        "source_grid_stage": source["grid_stage"],
        "raw_360_ranges_persisted": False,
        "scan_source_type": "DETERMINISTIC_POSE_RECONSTRUCTION",
        "forward_ratio_W": float(source["forward_ratio_W"]),
        "lateral_ratio_W": float(source["lateral_ratio_W"]),
        "forward_displacement": float(source["forward_offset"]),
        "lateral_displacement": float(source["lateral_offset"]),
        "world_x_reconstructed": validity["position"][0],
        "world_y_reconstructed": validity["position"][1],
        "yaw_deg_reconstructed": yaw_deg,
        "source_candidate_valid": _as_bool(source["candidate_valid"]),
        "source_straight_line_path_clear_eval": _as_bool(source["straight_line_path_clear_eval"]),
        "viewpoint_geometry_valid": validity["valid"],
        "movement_feasibility_proven": False,
        "inside_consistent": validity["inside"] == _as_bool(source["candidate_inside_free_space_eval"]),
        "footprint_consistent": validity["footprint"] == _as_bool(source["robot_footprint_walkable_eval"]),
        "viewpoint_validity_consistent": validity["consistent"],
        "local_transform_roundtrip_error": validity["roundtrip_error"],
        "saved_hit_count": saved_hit,
        "reconstructed_hit_count": "" if analysis is None else analysis["hit_count"],
        "saved_max_range_count": saved_max,
        "reconstructed_max_range_count": "" if analysis is None else analysis["max_count"],
        "saved_opening_count": saved_opening,
        "reconstructed_opening_count": "" if analysis is None else analysis["opening_count"],
        "scan_summary_consistent": "" if analysis is None else (
            analysis["hit_count"] == saved_hit
            and analysis["max_count"] == saved_max
            and analysis["opening_count"] == saved_opening
        ),
    }


def _accepted_gap_rows(analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialize every accepted scan-only gap with post-hoc truth labels."""
    rows = []
    for analysis in analyses:
        source = analysis["source"]
        matches = {row["gap_id"]: row for row in analysis["matches"]}
        for gap in analysis["result"]["gaps"]:
            if not gap["candidate_valid"]:
                continue
            match = matches[gap["gap_id"]]
            rows.append(
                {
                    "viewpoint_id": analysis["spec"]["viewpoint_id"],
                    "forward_ratio_W": float(source["forward_ratio_W"]),
                    "lateral_ratio_W": float(source["lateral_ratio_W"]),
                    "gap_id": gap["gap_id"],
                    "gap_width": gap["gap_width"],
                    "gap_width_ratio_W": gap["gap_width_ratio_W"],
                    "free_continuation_depth": gap["continuation_depth"],
                    "free_continuation_depth_ratio_W": gap["continuation_depth"] / analysis["spec"]["estimated_corridor_width"],
                    "matched_branch_eval": match["matched_branch_eval"],
                    "true_positive_eval": match["true_positive_eval"],
                    "false_positive_eval": match["false_positive_eval"],
                    "center_error_eval": match["center_error_eval"],
                    "endpoint_error_eval": match["endpoint_error_eval"],
                    "mouth_overlap_eval": match["mouth_overlap_eval"],
                }
            )
    return rows


def _boundary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand side-boundary visibility into one row per branch and viewpoint."""
    rows = []
    for summary in summaries:
        for label in SIDE_LABELS:
            key = label.lower()
            rows.append(
                {
                    "viewpoint_id": summary["viewpoint_id"],
                    "branch_eval": label,
                    "forward_ratio_W": summary["forward_ratio_W"],
                    "lateral_ratio_W": summary["lateral_ratio_W"],
                    "distance_ratio_W": summary["distance_ratio_W"],
                    "topology_state": summary[f"{key}_topology"],
                    "near_boundary_visible_eval": summary[f"{key}_near_boundary_visible_eval"],
                    "far_opposite_boundary_visible_eval": summary[f"{key}_far_boundary_visible_eval"],
                    "near_endpoint_error_eval": summary[f"{key}_near_endpoint_error_eval"],
                    "far_opposite_endpoint_error_eval": summary[f"{key}_far_endpoint_error_eval"],
                }
            )
    return rows


def _plot_topology_map(path: Path, rows: list[dict[str, Any]], side: str | None = None) -> None:
    """Plot sampled topology categories in the existing F/L grid."""
    fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    invalid = [row for row in rows if not row["viewpoint_geometry_valid"]]
    if invalid:
        axis.scatter(
            [row["lateral_ratio_W"] for row in invalid],
            [row["forward_ratio_W"] for row in invalid],
            marker="x",
            color="0.65",
            label="invalid",
        )
    valid = [row for row in rows if row["viewpoint_geometry_valid"]]
    if side is not None:
        key = f"{side.lower()}_topology"
        for state, color, marker in (
            ("NO_GAP_TOPOLOGY", "0.45", "o"),
            ("PARTIAL_GAP_TOPOLOGY", "tab:cyan", "s"),
            ("COMPLETE_GAP_TOPOLOGY", "tab:blue" if side == "LEFT" else "tab:orange", "*"),
        ):
            selected = [row for row in valid if row[key] == state]
            if selected:
                axis.scatter(
                    [row["lateral_ratio_W"] for row in selected],
                    [row["forward_ratio_W"] for row in selected],
                    marker=marker,
                    s=85 if state == "COMPLETE_GAP_TOPOLOGY" else 38,
                    color=color,
                    label=state.replace("_GAP_TOPOLOGY", ""),
                )
        title = f"{side} frozen wall-topology state"
    else:
        categories = (
            (lambda row: row["both_side_complete"], "BOTH COMPLETE", "purple", "*"),
            (lambda row: row["left_topology"] == "COMPLETE_GAP_TOPOLOGY" and not row["both_side_complete"], "LEFT COMPLETE", "tab:blue", "<"),
            (lambda row: row["right_topology"] == "COMPLETE_GAP_TOPOLOGY" and not row["both_side_complete"], "RIGHT COMPLETE", "tab:orange", ">"),
            (lambda row: row["left_topology"] != "COMPLETE_GAP_TOPOLOGY" and row["right_topology"] != "COMPLETE_GAP_TOPOLOGY", "no side COMPLETE", "tab:green", "o"),
        )
        for predicate, label, color, marker in categories:
            selected = [row for row in valid if predicate(row)]
            if selected:
                axis.scatter(
                    [row["lateral_ratio_W"] for row in selected],
                    [row["forward_ratio_W"] for row in selected],
                    marker=marker,
                    s=90 if "COMPLETE" in label and label != "no side COMPLETE" else 38,
                    color=color,
                    label=label,
                )
        title = "EXP-035 non-forward wall-topology observability"
    axis.axvline(0.0, color="0.75", linewidth=1)
    axis.axhline(0.0, color="0.75", linewidth=1)
    axis.set(xlabel="local lateral displacement / W (positive = local left)", ylabel="local forward displacement / W", title=title)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_opposite_visibility(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot evaluation-only far/opposite boundary errors for both side mouths."""
    valid = [row for row in rows if row["viewpoint_geometry_valid"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True, sharex=True, sharey=True)
    for axis, side in zip(axes, SIDE_LABELS):
        key = f"{side.lower()}_far_endpoint_error_eval"
        scatter = axis.scatter(
            [row["lateral_ratio_W"] for row in valid],
            [row["forward_ratio_W"] for row in valid],
            c=[row[key] for row in valid],
            cmap="viridis_r",
            s=48,
        )
        axis.set(title=f"{side} opposite-boundary endpoint error", xlabel="lateral / W", ylabel="forward / W")
        axis.grid(alpha=0.2)
        fig.colorbar(scatter, ax=axis, label="error [world unit]")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _m0_negative_replay(sensor: LidarSensor, width: float) -> dict[str, Any]:
    """Run one bounded deterministic straight-corridor topology sanity replay."""
    geometry = GeometryBuilder.build("M0_STRAIGHT")
    snapshot, hit_count, max_count, opening_count = _snapshot(
        sensor, geometry, np.zeros(2), 90.0, "M0_REPRESENTATIVE"
    )
    result = _analyze("M0_REPRESENTATIVE", snapshot, width)
    accepted = [gap for gap in result["gaps"] if gap["candidate_valid"]]
    return {
        "case": "M0_STRAIGHT",
        "replay_type": "DETERMINISTIC_GEOMETRY_REPLAY_NO_PHYSICS",
        "hit_count": hit_count,
        "max_range_count": max_count,
        "angular_opening_count": opening_count,
        "valid_termination_count": sum(bool(endpoint["valid"]) for endpoint in result["endpoints"]),
        "accepted_gap_count": len(accepted),
        "false_side_complete_count": 0 if not accepted else len(accepted),
        "passed": not accepted,
    }


def _minimum(rows: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    """Return the minimum observed COMPLETE row in the tested grid."""
    selected = [row for row in rows if row["branch_eval"] == label]
    return min(selected, key=lambda row: (row["distance_ratio_W"], row["viewpoint_id"]), default=None)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--safe-source", type=Path, default=SAFE_SOURCE)
    parser.add_argument("--exp033-summary", type=Path, default=DEFAULT_EXP033)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    _self_test()

    source_rows = _read(args.source)
    if not source_rows:
        raise RuntimeError(f"empty EXP-030 source: {args.source}")
    anchor = _anchor_from_existing_source(args.safe_source)
    geometry = _rear_start_geometry()
    sensor = LidarSensor()

    validity = [_validity_audit(row, anchor, geometry) for row in source_rows]
    valid_sources = [row for row, check in zip(source_rows, validity) if check["valid"]]
    analyses = [_analyze_viewpoint(row, anchor, sensor, geometry) for row in valid_sources]
    analysis_by_id = {analysis["spec"]["viewpoint_id"]: analysis for analysis in analyses}
    source_audit = [
        _source_audit_row(
            args.source,
            row,
            check,
            analysis_by_id.get(_viewpoint_id(row)),
            float(anchor["yaw_eval"]),
        )
        for row, check in zip(source_rows, validity)
    ]

    # Reconstruction and source validity are hard interpretation gates.
    validity_consistent = all(row["viewpoint_validity_consistent"] for row in source_audit)
    transform_sane = max(row["local_transform_roundtrip_error"] for row in source_audit) <= 1.0e-12
    scan_consistent = all(
        row["scan_summary_consistent"] is True
        for row in source_audit
        if row["viewpoint_geometry_valid"]
    )
    summaries_valid = [_topology_row(analysis) for analysis in analyses]
    summary_by_id = {row["viewpoint_id"]: row for row in summaries_valid}
    topology_grid = []
    for source, check in zip(source_rows, validity):
        viewpoint = _viewpoint_id(source)
        if viewpoint in summary_by_id:
            topology_grid.append(summary_by_id[viewpoint])
        else:
            forward, lateral = float(source["forward_ratio_W"]), float(source["lateral_ratio_W"])
            topology_grid.append(
                {
                    "viewpoint_id": viewpoint,
                    "grid_stage": source["grid_stage"],
                    "forward_ratio_W": forward,
                    "lateral_ratio_W": lateral,
                    "distance_ratio_W": math.hypot(forward, lateral),
                    "direction_class": _direction_class(forward, lateral),
                    "viewpoint_geometry_valid": False,
                    "movement_feasibility_proven": False,
                    "left_topology": "NOT_ANALYZED_INVALID",
                    "right_topology": "NOT_ANALYZED_INVALID",
                    "both_side_complete": False,
                }
            )

    a0 = summary_by_id.get("A0")
    if a0 is None:
        raise RuntimeError("EXP-030 source does not contain a valid A0")
    exp033_rows = _read(args.exp033_summary)
    exp033_a0 = next(row for row in exp033_rows if row["case"] == "M1_A0")
    a0_equivalent = (
        a0["forward_topology"] == "NO_GAP_TOPOLOGY"
        and a0["left_topology"] == "PARTIAL_GAP_TOPOLOGY"
        and a0["right_topology"] == "PARTIAL_GAP_TOPOLOGY"
        and a0["incoming_topology"] == "COMPLETE_GAP_TOPOLOGY"
        and a0["accepted_gap_count"] == 1
        and a0["side_true_gap_count"] == 0
        and a0["false_accepted_gap_count"] == 0
        and int(exp033_a0["accepted_gap_count"]) == a0["accepted_gap_count"]
        and int(exp033_a0["wall_topology_matched_outgoing_count"]) == 0
    )

    nonforward_analyses = [
        analysis
        for analysis in analyses
        if not math.isclose(float(analysis["source"]["lateral_ratio_W"]), 0.0)
    ]
    nonforward_summaries = [
        summary_by_id[analysis["spec"]["viewpoint_id"]] for analysis in nonforward_analyses
    ]
    complete = _complete_rows(nonforward_analyses, nonforward_summaries)
    left_complete = [row for row in complete if row["branch_eval"] == "LEFT"]
    right_complete = [row for row in complete if row["branch_eval"] == "RIGHT"]
    both = [row for row in nonforward_summaries if row["both_side_complete"]]
    pairs = _complementary_rows(complete, bool(both))
    minimum_left = _minimum(complete, "LEFT")
    minimum_right = _minimum(complete, "RIGHT")
    minimum_both = min(both, key=lambda row: (row["distance_ratio_W"], row["viewpoint_id"]), default=None)
    false_unstable = any(
        row["false_accepted_gap_count"] > a0["false_accepted_gap_count"]
        for row in nonforward_summaries
        if row["left_topology"] == "COMPLETE_GAP_TOPOLOGY"
        or row["right_topology"] == "COMPLETE_GAP_TOPOLOGY"
    )

    # Repeat every valid scan and require exact arrays and topology summaries.
    repeated = [_analyze_viewpoint(row, anchor, sensor, geometry) for row in valid_sources]
    deterministic = all(
        np.array_equal(first["snapshot"]["ranges"], second["snapshot"]["ranges"])
        and _topology_row(first) == _topology_row(second)
        for first, second in zip(analyses, repeated)
    )
    topology_self_check = all(
        side["candidate"] is not None and side["match"] is not None
        for analysis in nonforward_analyses
        for side in (analysis["left"], analysis["right"])
        if side["topology"] == "COMPLETE_GAP_TOPOLOGY"
    )
    m0 = _m0_negative_replay(sensor, float(anchor["estimated_corridor_width"]))

    left_far_visible = [
        row for row in nonforward_summaries if row["left_far_boundary_visible_eval"]
    ]
    right_far_visible = [
        row for row in nonforward_summaries if row["right_far_boundary_visible_eval"]
    ]

    if not scan_consistent or not validity_consistent or not transform_sane or not a0_equivalent:
        verdict = "F_EXISTING_VIEWPOINT_RECONSTRUCTION_INCONSISTENT"
    elif false_unstable:
        verdict = "E_NONFORWARD_RECOVERY_FALSE_POSITIVE_UNSTABLE"
    elif both:
        verdict = "A_SINGLE_NONFORWARD_VIEWPOINT_COMPLETES_BOTH_SIDE_TOPOLOGIES"
    elif left_complete and right_complete:
        verdict = "B_COMPLEMENTARY_NONFORWARD_VIEWPOINTS_RECOVER_BOTH_SIDES"
    elif bool(left_complete) != bool(right_complete):
        verdict = "C_ASYMMETRIC_NONFORWARD_RECOVERY"
    else:
        verdict = "D_NO_SINGLE_VIEWPOINT_SIDE_TOPOLOGY_RECOVERY"

    angular_rows = [
        {
            "viewpoint_id": row["viewpoint_id"],
            "forward_ratio_W": row["forward_ratio_W"],
            "lateral_ratio_W": row["lateral_ratio_W"],
            "angular_detector_outgoing_match_eval": row["angular_detector_outgoing_match_eval"],
            "left_wall_topology": row["left_topology"],
            "right_wall_topology": row["right_topology"],
            "forward_wall_topology": row["forward_topology"],
        }
        for row in summaries_valid
    ]
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "diagnostic_scope": "FINAL_SINGLE_VIEW_NONFORWARD_OBSERVABILITY",
        "source_viewpoint_count": len(source_rows),
        "valid_viewpoint_count": len(valid_sources),
        "analyzed_valid_nonforward_viewpoint_count": len(nonforward_analyses),
        "analyzed_forward_reference_count": len(analyses) - len(nonforward_analyses),
        "left_complete_viewpoint_count": len({row["viewpoint_id"] for row in left_complete}),
        "right_complete_viewpoint_count": len({row["viewpoint_id"] for row in right_complete}),
        "both_complete_viewpoint_count": len(both),
        "minimum_observed_left_complete": "" if minimum_left is None else minimum_left["viewpoint_id"],
        "minimum_observed_left_complete_distance_W": "" if minimum_left is None else minimum_left["distance_ratio_W"],
        "minimum_observed_right_complete": "" if minimum_right is None else minimum_right["viewpoint_id"],
        "minimum_observed_right_complete_distance_W": "" if minimum_right is None else minimum_right["distance_ratio_W"],
        "minimum_observed_both_complete": "" if minimum_both is None else minimum_both["viewpoint_id"],
        "minimum_observed_both_complete_distance_W": "" if minimum_both is None else minimum_both["distance_ratio_W"],
        "complementary_pair_count": len(pairs),
        "best_complementary_left": "" if not pairs else pairs[0]["left_viewpoint_id"],
        "best_complementary_right": "" if not pairs else pairs[0]["right_viewpoint_id"],
        "maximum_false_accepted_gap_count": max(row["false_accepted_gap_count"] for row in summaries_valid),
        "complete_viewpoints_with_false_gap_count": sum(
            row["false_accepted_gap_count"] > a0["false_accepted_gap_count"]
            for row in nonforward_summaries
            if row["left_topology"] == "COMPLETE_GAP_TOPOLOGY"
            or row["right_topology"] == "COMPLETE_GAP_TOPOLOGY"
        ),
        "left_far_boundary_visible_viewpoint_count": len(left_far_visible),
        "left_far_boundary_visible_lateral_signs": json.dumps(
            sorted({int(math.copysign(1, float(row["lateral_ratio_W"]))) for row in left_far_visible})
        ),
        "right_far_boundary_visible_viewpoint_count": len(right_far_visible),
        "right_far_boundary_visible_lateral_signs": json.dumps(
            sorted({int(math.copysign(1, float(row["lateral_ratio_W"]))) for row in right_far_visible})
        ),
        "A0_equivalent_to_EXP033": a0_equivalent,
        "deterministic_replay": deterministic,
        "source_scan_summary_consistent": scan_consistent,
        "viewpoint_validity_consistent": validity_consistent,
        "local_transform_sane": transform_sane,
        "topology_state_self_check": topology_self_check,
        "M0_negative_passed": m0["passed"],
        "movement_feasibility_proven": False,
        "single_view_observability_diagnostic_complete": True,
    }
    verdict_row = {
        **summary,
        "primary_verdict": verdict,
        "wall_topology_rule_modified": False,
        "new_viewpoint_grid_created": False,
        "robot_movement_executed": False,
        "GT_used_for_candidate_generation_or_acceptance": False,
        "GT_used_posthoc_only": True,
        "map_used_for_scan_reconstruction_and_posthoc_eval_only": True,
        "production_simulator_modified": False,
        "pointcloud_detector_modified": False,
        "lidar_profile_detector_modified": False,
    }

    _write_required(args.output / "source_viewpoint_audit.csv", source_audit)
    _write_required(args.output / "viewpoint_topology_grid.csv", topology_grid)
    _write_required(args.output / "branch_boundary_visibility.csv", _boundary_rows(summaries_valid))
    _write_required(args.output / "accepted_gap_audit.csv", _accepted_gap_rows(analyses))
    _write_required(
        args.output / "complete_viewpoints.csv",
        complete,
        [
            "viewpoint_id", "branch_eval", "forward_ratio_W", "lateral_ratio_W",
            "distance_ratio_W", "direction_class", "gap_width", "gap_width_ratio_W",
            "free_continuation_depth", "center_error_eval", "endpoint_error_eval",
            "mouth_overlap_eval", "false_gap_count",
        ],
    )
    _write_required(
        args.output / "complementary_pairs.csv",
        pairs,
        [
            "left_viewpoint_id", "right_viewpoint_id", "left_F_W", "left_L_W",
            "left_distance_W", "right_F_W", "right_L_W", "right_distance_W",
            "combined_branch_coverage", "pair_distance_sum_W", "pair_max_distance_W",
        ],
    )
    _write_required(args.output / "angular_vs_wall_topology.csv", angular_rows)
    _write_required(args.output / "single_view_observability_summary.csv", [summary])
    _write_required(args.output / "verdict.csv", [verdict_row])
    _write_required(args.output / "m0_negative_sanity.csv", [m0])

    _plot_topology_map(args.output / "nonforward_wall_topology_map.png", topology_grid)
    _plot_topology_map(args.output / "left_topology_map.png", topology_grid, "LEFT")
    _plot_topology_map(args.output / "right_topology_map.png", topology_grid, "RIGHT")
    _plot_opposite_visibility(args.output / "opposite_boundary_visibility_map.png", summaries_valid)
    runner = SimpleNamespace(geometry=geometry)
    a0_analysis = analysis_by_id["A0"]
    _plot_result(args.output / "a0_wall_topology.png", a0_analysis["result"], runner, a0_analysis["mouths"], "EXP-035 A0 frozen wall topology")
    for filename, selected in (
        ("best_left_complete.png", minimum_left),
        ("best_right_complete.png", minimum_right),
        ("best_both_complete.png", minimum_both),
    ):
        if selected is not None:
            analysis = analysis_by_id[selected["viewpoint_id"]]
            _plot_result(args.output / filename, analysis["result"], runner, analysis["mouths"], f"EXP-035 {selected['viewpoint_id']} frozen wall topology")

    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "primary_verdict": verdict,
                "summary": summary,
                "minimum_left": minimum_left,
                "minimum_right": minimum_right,
                "minimum_both": minimum_both,
                "best_complementary_pair": pairs[0] if pairs else None,
                "M0": m0,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
