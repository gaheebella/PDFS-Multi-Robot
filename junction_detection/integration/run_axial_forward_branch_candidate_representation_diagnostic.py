"""EXP-037 bounded diagnostic for the missing axial/forward branch candidate.

The frozen EXP-033 extractor and EXP-034 runtime candidate constructor are run
unchanged on M1.  GT mouths and branch labels are attached only after each
runtime-local update, to explain where the already-observed evidence ceases to
be represented as a candidate.  This file diagnoses the failure; it does not
add an axial candidate path or tune any threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.integration.run_provisional_anchor_pointcloud_observation import (
    _contains,
    _gt_directions_eval_only,
)
from junction_detection.integration.run_wall_topology_branch_opening_diagnostic import (
    _beam_range,
    _gt_mouths_eval,
    _self_test as exp033_self_test,
)
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import (
    detect_openings,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import DT
from pygame_simulator.pre_exploration_wall_topology_sph_validation import (
    WallTopologySPHRunner,
    _axis_frame,
    _candidate_state_self_test,
)

EXPERIMENT_ID = "EXP-037"
CASE_ID = "M1_CROSS_BASELINE"
BRANCHES = ("LEFT", "FORWARD", "RIGHT")
DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/axial_forward_branch_candidate_representation_diagnostic"
DEFAULT_FRAMES = 181
MIN_FRAMES = 180
MAX_FRAMES = 240


def _write(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    """Write a diagnostic table, preserving an empty-table header when supplied."""
    if fields is None:
        if not rows:
            return
        fields = list(rows[0])
        for row in rows[1:]:
            fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _normalize(angle: float) -> float:
    """Normalize an angle in degrees to [-180, 180)."""
    return float((angle + 180.0) % 360.0 - 180.0)


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """Return Euclidean distance from a point to a finite segment."""
    edge = end - start
    denominator = float(np.dot(edge, edge))
    ratio = 0.0 if denominator <= 1.0e-12 else float(np.clip(np.dot(point - start, edge) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + ratio * edge)))


def _mouth_assignment(
    first: np.ndarray,
    second: np.ndarray,
    mouths: list[dict[str, Any]],
) -> tuple[str, float, float]:
    """Match an endpoint pair to the nearest GT mouth after extraction."""
    best_label, best_endpoint, best_center = "NONE", math.inf, math.inf
    for mouth in mouths:
        direct = 0.5 * (np.linalg.norm(first - mouth["a"]) + np.linalg.norm(second - mouth["b"]))
        reverse = 0.5 * (np.linalg.norm(first - mouth["b"]) + np.linalg.norm(second - mouth["a"]))
        endpoint_error = float(min(direct, reverse))
        center_error = float(np.linalg.norm(0.5 * (first + second) - 0.5 * (mouth["a"] + mouth["b"])))
        if (endpoint_error, center_error) < (best_endpoint, best_center):
            best_label, best_endpoint, best_center = mouth["label"], endpoint_error, center_error
    return best_label, best_endpoint, best_center


def _endpoint_branch(
    point: np.ndarray,
    mouths: list[dict[str, Any]],
    tolerance: float,
) -> tuple[str, float]:
    """Associate an extracted endpoint with the nearest GT mouth boundary post-hoc."""
    distances = [
        (min(float(np.linalg.norm(point - mouth["a"])), float(np.linalg.norm(point - mouth["b"]))), mouth["label"])
        for mouth in mouths
    ]
    distance, label = min(distances, default=(math.inf, "NONE"))
    return (label if distance <= tolerance else "NONE"), distance


def _angular_match(
    runner: WallTopologySPHRunner,
    snapshot: dict[str, Any],
    openings: list[dict[str, float]],
) -> dict[str, dict[str, Any]]:
    """Match frozen angular openings to branch axes strictly for evaluation."""
    directions, _ = _gt_directions_eval_only(runner, snapshot)
    labels = {
        index: {0.0: "FORWARD", -90.0: "RIGHT", 90.0: "LEFT"}.get(float(branch.angle_deg), f"BRANCH_{index}")
        for index, branch in enumerate(runner.geometry.branches)
    }
    matches: dict[str, dict[str, Any]] = {}
    for branch in directions:
        label = labels[int(branch["branch_id"])]
        containing = [opening for opening in openings if _contains(opening, branch["local_angle_deg"])]
        if containing:
            opening = min(containing, key=lambda row: abs(_normalize(float(row["center_angle"]) - branch["local_angle_deg"])))
            matches[label] = {
                "present": True,
                "center": float(opening["center_angle"]),
                "width": float(opening["width_deg"]),
                "direction": float(branch["local_angle_deg"]),
            }
        else:
            matches[label] = {"present": False, "center": math.nan, "width": math.nan, "direction": float(branch["local_angle_deg"])}
    return matches


def _runtime_candidate_labels(runner: WallTopologySPHRunner) -> dict[str, list[Any]]:
    """Group already-created runtime candidates by their evaluation-only label."""
    grouped = {label: [] for label in BRANCHES}
    for candidate in runner.candidates:
        if candidate.matched_branch_eval_only in grouped:
            grouped[candidate.matched_branch_eval_only].append(candidate)
    return grouped


def _analyze_frame(runner: WallTopologySPHRunner, frame: int) -> dict[str, Any] | None:
    """Attach post-hoc branch labels to one completed runtime-local scan update."""
    result, snapshot = runner.last_result, runner.last_snapshot
    profile = runner.last_profile_result
    if result is None or snapshot is None or profile is None:
        return None
    width = float(profile["estimated_corridor_width"])
    orientation = float(profile["stable_corridor_orientation_deg"])
    if not math.isfinite(width) or width <= 0.0 or not math.isfinite(orientation):
        return None
    mouths = [mouth for mouth in _gt_mouths_eval(runner, snapshot) if mouth["branch_type"] == "OUTGOING"]
    mouth_by_label = {mouth["label"]: mouth for mouth in mouths}
    tolerance = 0.12 * width  # Frozen EXP-033 evaluation tolerance, never a runtime input.
    openings = list(detect_openings(snapshot["angles"], snapshot["ranges"]))
    angular = _angular_match(runner, snapshot, openings)
    candidates = _runtime_candidate_labels(runner)
    _, left_axis = _axis_frame(orientation)
    endpoint_rows = []
    endpoint_branch: dict[int, str] = {}
    segment_by_id = {row["segment_id"]: row for row in result["segments"]}
    for endpoint in result["endpoints"]:
        label, distance = _endpoint_branch(endpoint["point"], mouths, tolerance)
        endpoint_branch[endpoint["endpoint_id"]] = label
        tangents = [segment_by_id[index]["orientation_deg"] for index in endpoint["segment_ids"]]
        endpoint_rows.append({
            "frame": frame,
            "timestamp": snapshot["time"],
            "endpoint_id": endpoint["endpoint_id"],
            "segment_id": json.dumps(endpoint["segment_ids"]),
            "x_local": float(endpoint["point"][0]),
            "y_local": float(endpoint["point"][1]),
            "endpoint_type": endpoint["endpoint_type"],
            "wall_tangent_local": json.dumps(tangents),
            "range": float(np.linalg.norm(endpoint["point"])),
            "associated_branch_eval_only": label,
            "nearest_mouth_boundary_error_eval_only": distance,
        })
    endpoint_by_id = {row["endpoint_id"]: row for row in result["endpoints"]}
    gap_rows = []
    for gap in result["gaps"]:
        first = endpoint_by_id[gap["endpoint_a"]]
        second = endpoint_by_id[gap["endpoint_b"]]
        label, endpoint_error, center_error = _mouth_assignment(first["point"], second["point"], mouths)
        matched = endpoint_error <= tolerance and center_error <= tolerance
        matched_label = label if matched else "NONE"
        side_a = math.copysign(1.0, float(np.dot(first["point"], left_axis)))
        side_b = math.copysign(1.0, float(np.dot(second["point"], left_axis)))
        same_side = side_a == side_b
        runtime_candidate_created = bool(gap["candidate_valid"] and same_side)
        if not gap["candidate_valid"]:
            runtime_reason = "GAP_REJECTED_UPSTREAM"
        elif not same_side:
            runtime_reason = "ENDPOINTS_OPPOSITE_CORRIDOR_SIDES"
        else:
            runtime_reason = "NONE"
        gap_rows.append({
            "frame": frame,
            "timestamp": snapshot["time"],
            "gap_id": gap["gap_id"],
            "endpoint_a": gap["endpoint_a"],
            "endpoint_b": gap["endpoint_b"],
            "gap_center_local": json.dumps([float(value) for value in gap["gap_center"]]),
            "gap_width": gap["gap_width"],
            "gap_width_over_W_hat": gap["gap_width_ratio_W"],
            "free_continuation": gap["continuation_depth"],
            "accepted": gap["candidate_valid"],
            "rejection_reason": gap["rejection_reason"],
            "matched_branch_eval_only": matched_label,
            "endpoint_error_eval_only": endpoint_error,
            "center_error_eval_only": center_error,
            "same_lateral_side_runtime": same_side,
            "runtime_candidate_created": runtime_candidate_created,
            "runtime_rejection_reason": runtime_reason,
        })
    stage_rows = []
    for label in BRANCHES:
        mouth = mouth_by_label[label]
        center = 0.5 * (mouth["a"] + mouth["b"])
        direction = angular[label]["direction"]
        continuation_depth = _beam_range(snapshot, direction) - float(np.linalg.norm(center))
        lidar_free = continuation_depth >= 0.20 * width
        segment_support_count = sum(
            min(_point_segment_distance(target, segment["start"], segment["end"]) for segment in result["segments"]) <= tolerance
            for target in (mouth["a"], mouth["b"])
        ) if result["segments"] else 0
        associated = [endpoint for endpoint in result["endpoints"] if endpoint_branch[endpoint["endpoint_id"]] == label]
        wall_terms = [endpoint for endpoint in associated if endpoint["endpoint_type"] == "WALL_TERMINATION"]
        valid = [endpoint for endpoint in associated if endpoint["valid"]]
        branch_gaps = [row for row in gap_rows if row["matched_branch_eval_only"] == label]
        accepted = [row for row in branch_gaps if row["accepted"]]
        # Use the same free-continuation evidence consumed by the frozen gap
        # extractor.  A GT-mouth center beam alone can be occluded even when
        # the accepted scan-derived gap has valid continuation.
        continuation_depth = max(
            [continuation_depth] + [float(row["free_continuation"]) for row in branch_gaps]
        )
        lidar_free = continuation_depth >= 0.20 * width
        runtime = candidates[label]
        partial = any(
            event["candidate_id"] == candidate.candidate_id and event["event"] == "PARTIAL_BRANCH_CANDIDATE"
            for candidate in runtime for event in runner.events
        )
        complete = any(candidate.topology_type == "COMPLETE" for candidate in runtime)
        if runtime:
            first_failure = "NONE"
        elif not lidar_free:
            first_failure = "LIDAR_FREE_EVIDENCE"
        elif segment_support_count == 0:
            first_failure = "WALL_SEGMENT_SUPPORT"
        elif not valid:
            first_failure = "VALID_ENDPOINT_SUPPORT"
        elif not branch_gaps:
            first_failure = "ENDPOINT_PAIR_OR_GAP_CONSTRUCTION"
        elif not accepted:
            reasons = sorted({str(row["rejection_reason"]) for row in branch_gaps})
            first_failure = "GAP_ACCEPTANCE:" + "+".join(reasons)
        elif any(row["runtime_rejection_reason"] == "ENDPOINTS_OPPOSITE_CORRIDOR_SIDES" for row in accepted) and not runtime:
            first_failure = "CANDIDATE_CONSTRUCTION_SAME_LATERAL_SIDE_GATE"
        elif not runtime:
            first_failure = "RUNTIME_TOPOLOGY_CONSTRUCTION"
        else:
            first_failure = "NONE"
        stage_rows.append({
            "frame": frame,
            "timestamp": snapshot["time"],
            "branch_eval_only": label,
            "lidar_free_evidence": lidar_free,
            "lidar_free_continuation_depth": continuation_depth,
            "angular_opening_present_eval_only": angular[label]["present"],
            "angular_opening_center_eval_only": angular[label]["center"],
            "angular_opening_width_eval_only": angular[label]["width"],
            "wall_segment_support": segment_support_count > 0,
            "wall_segment_boundary_support_count_eval_only": segment_support_count,
            "wall_termination_count": len(wall_terms),
            "valid_termination_count": len(valid),
            "pair_count": len(branch_gaps),
            "gap_count": len(branch_gaps),
            "accepted_gap_count": len(accepted),
            "partial_present": partial,
            "complete_present": complete,
            "candidate_present": bool(runtime),
            "first_failure_stage": first_failure,
        })
    return {
        "frame": frame,
        "timestamp": snapshot["time"],
        "result": result,
        "snapshot": snapshot,
        "mouths": mouths,
        "openings": openings,
        "orientation": orientation,
        "width": width,
        "stage_rows": stage_rows,
        "endpoint_rows": endpoint_rows,
        "gap_rows": gap_rows,
        "candidate_ids": [candidate.candidate_id for candidate in runner.candidates],
    }


def _run_once(frames: int) -> tuple[WallTopologySPHRunner, list[dict[str, Any]], int]:
    """Run one fixed-length bounded M1 diagnostic without GT-based stopping."""
    runner = WallTopologySPHRunner(CASE_ID, rear_start=True)
    captures = []
    for frame in range(frames):
        row = runner.step(frame)
        if row is None:
            continue
        capture = _analyze_frame(runner, frame)
        if capture is not None:
            captures.append(capture)
    return runner, captures, frames


def _signature(captures: list[dict[str, Any]]) -> list[Any]:
    """Return an exact deterministic signature without plot-only arrays."""
    signature = []
    for capture in captures:
        signature.append((
            capture["frame"],
            capture["timestamp"],
            capture["candidate_ids"],
            capture["stage_rows"],
            capture["endpoint_rows"],
            capture["gap_rows"],
        ))
    return signature


def _select_best(captures: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Select the latest frame with maximum observable pipeline progress."""
    def score(capture: dict[str, Any]) -> tuple[int, ...]:
        row = next(item for item in capture["stage_rows"] if item["branch_eval_only"] == label)
        return (
            int(row["candidate_present"]), int(row["complete_present"]), int(row["partial_present"]),
            int(row["accepted_gap_count"] > 0), int(row["gap_count"] > 0),
            int(row["valid_termination_count"] > 0), int(row["wall_segment_support"]),
            int(row["lidar_free_evidence"]), int(row["angular_opening_present_eval_only"]),
            capture["frame"],
        )
    return max(captures, key=score)


def _summary_rows(captures: list[dict[str, Any]], deterministic: bool) -> list[dict[str, Any]]:
    """Build the requested LEFT/FORWARD/RIGHT comparison table."""
    best = {label: next(row for row in _select_best(captures, label)["stage_rows"] if row["branch_eval_only"] == label) for label in BRANCHES}
    metrics = [
        ("LiDAR free evidence", "lidar_free_evidence"),
        ("Angular opening shadow", "angular_opening_present_eval_only"),
        ("Wall segment support", "wall_segment_support"),
        ("WALL_TERMINATION count", "wall_termination_count"),
        ("Valid termination count", "valid_termination_count"),
        ("Second termination", "valid_termination_count"),
        ("Endpoint pair", "pair_count"),
        ("Gap constructed", "gap_count"),
        ("Gap accepted", "accepted_gap_count"),
        ("PARTIAL topology", "partial_present"),
        ("COMPLETE topology", "complete_present"),
        ("Candidate created", "candidate_present"),
        ("First failure stage", "first_failure_stage"),
    ]
    rows = []
    for metric, key in metrics:
        values = {}
        for label in BRANCHES:
            value = best[label][key]
            if metric == "Second termination":
                value = int(value) >= 2
            elif metric in {"Endpoint pair", "Gap constructed", "Gap accepted"}:
                value = int(value) > 0
            values[label.lower()] = value
        rows.append({"metric": metric, **values})
    rows.append({"metric": "Deterministic replay", "left": deterministic, "forward": deterministic, "right": deterministic})
    return rows


def _plot_capture(path: Path, capture: dict[str, Any], runner: WallTopologySPHRunner, title: str) -> None:
    """Plot scan, extracted topology, local axes, openings, and eval-only GT."""
    result, snapshot = capture["result"], capture["snapshot"]
    fig, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    points = np.asarray(result["points"])
    hit = np.asarray(snapshot["hit"], dtype=bool)
    axis.scatter(points[hit, 0], points[hit, 1], s=8, color="0.72", label="LiDAR hits")
    hit_indices = np.flatnonzero(hit)[::15]
    for index in hit_indices:
        axis.plot([0.0, points[index, 0]], [0.0, points[index, 1]], color="0.88", linewidth=0.4)
    for segment in result["segments"]:
        axis.plot([segment["start"][0], segment["end"][0]], [segment["start"][1], segment["end"][1]], linewidth=2)
        axis.text(*(0.5 * (segment["start"] + segment["end"])), f"S{segment['segment_id']}", fontsize=7)
    marker = {"WALL_TERMINATION": "o", "CORNER": "s", "SCAN_LIMIT": "^", "UNSTABLE_ENDPOINT": "x"}
    color = {"WALL_TERMINATION": "tab:red", "CORNER": "tab:orange", "SCAN_LIMIT": "tab:blue", "UNSTABLE_ENDPOINT": "0.3"}
    endpoints = {row["endpoint_id"]: row for row in result["endpoints"]}
    for endpoint in result["endpoints"]:
        axis.scatter(*endpoint["point"], marker=marker[endpoint["endpoint_type"]], color=color[endpoint["endpoint_type"]], s=60)
        axis.text(*endpoint["point"], f"E{endpoint['endpoint_id']}:{endpoint['endpoint_type']}", fontsize=6)
    for gap in result["gaps"]:
        first, second = endpoints[gap["endpoint_a"]]["point"], endpoints[gap["endpoint_b"]]["point"]
        axis.plot([first[0], second[0]], [first[1], second[1]], color="tab:green" if gap["candidate_valid"] else "tab:red", alpha=0.85 if gap["candidate_valid"] else 0.12, linewidth=3 if gap["candidate_valid"] else 1)
    forward, left = _axis_frame(capture["orientation"])
    axis.arrow(0, 0, *(forward * 35), color="black", width=0.7, length_includes_head=True, label="local corridor forward")
    axis.arrow(0, 0, *(left * 25), color="0.35", width=0.5, length_includes_head=True, label="local left")
    directions, _ = _gt_directions_eval_only(runner, snapshot)
    for index, branch in enumerate(runner.geometry.branches):
        label = {0.0: "FORWARD", -90.0: "RIGHT", 90.0: "LEFT"}.get(float(branch.angle_deg), f"BRANCH_{index}")
        mouth = next(row for row in capture["mouths"] if row["label"] == label)
        axis.plot([mouth["a"][0], mouth["b"][0]], [mouth["a"][1], mouth["b"][1]], "--", linewidth=2.5, label=f"GT {label} (EVAL ONLY)")
        angle = math.radians(directions[index]["local_angle_deg"])
        center = 0.5 * (mouth["a"] + mouth["b"])
        axis.arrow(center[0], center[1], 20 * math.cos(angle), 20 * math.sin(angle), color="tab:purple", width=0.35, length_includes_head=True)
    for opening in capture["openings"]:
        angle = math.radians(float(opening["center_angle"]))
        axis.plot([0, 90 * math.cos(angle)], [0, 90 * math.sin(angle)], color="magenta", linestyle=":", linewidth=1.5)
    for candidate in runner.candidates:
        if candidate.candidate_id not in capture["candidate_ids"]:
            continue
        axis.scatter(*candidate.endpoint_local, marker="X", color="purple", s=90, label=f"runtime {candidate.candidate_id}")
    axis.scatter(0, 0, marker="*", color="black", s=140, label="LiDAR")
    axis.set(title=f"{title}\nframe={capture['frame']} t={capture['timestamp']:.6f}s", xlabel="LiDAR-local x", ylabel="LiDAR-local y", aspect="equal", xlim=(-160, 160), ylim=(-160, 160))
    axis.grid(alpha=0.2)
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), fontsize=7, loc="upper left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(output: Path, frames: int) -> dict[str, Any]:
    """Execute bounded main/replay diagnostics and save the requested artifacts."""
    if not (MIN_FRAMES <= frames <= MAX_FRAMES):
        raise ValueError(f"Require {MIN_FRAMES} <= frames <= {MAX_FRAMES}")
    output.mkdir(parents=True, exist_ok=True)
    main_runner, captures, main_frames = _run_once(frames)
    replay_runner, replay_captures, replay_frames = _run_once(frames)
    deterministic = main_frames == replay_frames and _signature(captures) == _signature(replay_captures)
    stage_rows = [row for capture in captures for row in capture["stage_rows"]]
    endpoint_rows = [row for capture in captures for row in capture["endpoint_rows"]]
    gap_rows = [row for capture in captures for row in capture["gap_rows"]]
    summary = _summary_rows(captures, deterministic)
    _write(output / "forward_branch_stage_audit.csv", stage_rows)
    _write(output / "forward_branch_endpoint_audit.csv", endpoint_rows)
    _write(output / "forward_branch_gap_audit.csv", gap_rows)
    _write(output / "forward_vs_side_summary.csv", summary)
    evidence = next((capture for capture in captures if any(row["lidar_free_evidence"] for row in capture["stage_rows"])), captures[0])
    side = next((capture for capture in captures if {"LEFT", "RIGHT"}.issubset({row["branch_eval_only"] for row in capture["stage_rows"] if row["candidate_present"]}) and not next(row for row in capture["stage_rows"] if row["branch_eval_only"] == "FORWARD")["candidate_present"]), _select_best(captures, "LEFT"))
    angular = next((capture for capture in captures if next(row for row in capture["stage_rows"] if row["branch_eval_only"] == "FORWARD")["angular_opening_present_eval_only"]), _select_best(captures, "FORWARD"))
    _plot_capture(output / "initial_branch_evidence_frame.png", evidence, main_runner, "Initial branch evidence")
    _plot_capture(output / "side_candidate_without_forward_frame.png", side, main_runner, "Side candidates present; FORWARD absent")
    _plot_capture(output / "angular_forward_evidence_frame.png", angular, main_runner, "Angular FORWARD evidence shadow")
    forward = next(row for row in _select_best(captures, "FORWARD")["stage_rows"] if row["branch_eval_only"] == "FORWARD")
    baseline = {
        event: sorted(float(row["timestamp"]) for row in main_runner.events if row["event"] == event)
        for event in ("PARTIAL_BRANCH_CANDIDATE", "COMPLETE_BRANCH_CANDIDATE")
    }
    return {
        "main_frames": main_frames,
        "replay_frames": replay_frames,
        "time_seconds": main_frames * DT,
        "deterministic": deterministic,
        "forward": forward,
        "baseline": baseline,
        "candidate_labels": sorted(candidate.matched_branch_eval_only for candidate in main_runner.candidates),
        "output": str(output.resolve()),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    exp033_self_test()
    _candidate_state_self_test()
    result = run(args.output, args.frames)
    print(json.dumps({"experiment_id": EXPERIMENT_ID, **result}, indent=2, default=str))


if __name__ == "__main__":
    main()
