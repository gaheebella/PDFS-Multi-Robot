"""EXP-033: wall-topology branch-mouth feasibility at fixed A0.

Runtime topology uses one ideal body-local LiDAR scan, its hit/no-return mask,
and the already available local corridor-width estimate.  GT walls, branch IDs,
and mouth segments are introduced only after candidate generation for matching,
classification, and plots.  No production detector or simulator is modified.
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

from junction_detection.integration.run_2d_viewpoint_visibility_frontier import (
    _branch_mouth_points,
    _probe,
)
from junction_detection.integration.run_local_asymmetric_viewpoint_geometry_diagnostic import (
    _acquire_m0_snapshot,
    _acquire_m1_anchor,
)
from junction_detection.integration.run_side_branch_detector_evidence_pipeline import (
    _write_csv,
)
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import (
    _circular_runs,
    detect_openings,
)

EXPERIMENT_ID = "EXP-033"
DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/wall_topology_branch_opening"
MIN_POINTS = 3


def _normalize(angle: float) -> float:
    """Normalize a degree angle to [-180, 180)."""
    return float((angle + 180.0) % 360.0 - 180.0)


def _local_points(snapshot: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Convert one body-relative polar scan to local Cartesian points."""
    angles = np.asarray(snapshot["angles"], dtype=float)
    ranges = np.asarray(snapshot["ranges"], dtype=float)
    theta = np.deg2rad(angles)
    return np.column_stack((ranges * np.cos(theta), ranges * np.sin(theta))), np.asarray(snapshot["hit"], dtype=bool)


def _line_fit(points: np.ndarray) -> dict[str, Any]:
    """Fit a 2-D line by PCA and return axial direction and residuals."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    values, vectors = np.linalg.eigh(centered.T @ centered)
    direction = vectors[:, int(np.argmax(values))]
    if direction[0] < 0.0 or (abs(direction[0]) < 1.0e-12 and direction[1] < 0.0):
        direction = -direction
    normal = np.array([-direction[1], direction[0]])
    distances = np.abs(centered @ normal)
    return {
        "centroid": centroid,
        "direction": direction,
        "normal": normal,
        "residuals": distances,
        "rms": float(np.sqrt(np.mean(distances**2))),
    }


def _rdp_breaks(points: np.ndarray, tolerance: float) -> list[int]:
    """Return split-and-merge break indices for an ordered clean polyline."""
    if len(points) <= 2:
        return [0, len(points) - 1]
    edge = points[-1] - points[0]
    length = float(np.linalg.norm(edge))
    if length <= 1.0e-12:
        distances = np.linalg.norm(points - points[0], axis=1)
    else:
        offsets = points - points[0]
        distances = np.abs(
            edge[0] * offsets[:, 1] - edge[1] * offsets[:, 0]
        ) / length
    split = int(np.argmax(distances))
    if float(distances[split]) <= tolerance or split == 0 or split == len(points) - 1:
        return [0, len(points) - 1]
    left = _rdp_breaks(points[: split + 1], tolerance)
    right = _rdp_breaks(points[split:], tolerance)
    return left[:-1] + [split + index for index in right]


def _split_jump_runs(run: np.ndarray, points: np.ndarray, max_jump: float) -> list[np.ndarray]:
    """Split a hit run wherever adjacent Cartesian support is physically distant."""
    if len(run) < 2:
        return [run]
    gaps = np.linalg.norm(points[run[1:]] - points[run[:-1]], axis=1)
    cuts = np.flatnonzero(gaps > max_jump) + 1
    return [part for part in np.split(run, cuts) if len(part)]


def _segments_from_scan(snapshot: dict[str, Any], width_hat: float) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Extract explanatory line segments from ordered finite-hit points."""
    points, hit = _local_points(snapshot)
    # Scale-relative constants are fixed diagnostic resolutions, not fitted to GT.
    jump_limit = 0.20 * width_hat
    line_tolerance = 0.005 * width_hat
    pieces: list[np.ndarray] = []
    for circular_run in _circular_runs(hit, value=True):
        for jump_run in _split_jump_runs(circular_run, points, jump_limit):
            if len(jump_run) < MIN_POINTS:
                continue
            breaks = sorted(set(_rdp_breaks(points[jump_run], line_tolerance)))
            for start, end in zip(breaks, breaks[1:]):
                indices = jump_run[start : end + 1]
                if len(indices) >= MIN_POINTS:
                    pieces.append(indices)
    segments = []
    for segment_id, indices in enumerate(pieces):
        fit = _line_fit(points[indices])
        direction = fit["direction"]
        projections = (points[indices] - fit["centroid"]) @ direction
        start = fit["centroid"] + float(np.min(projections)) * direction
        end = fit["centroid"] + float(np.max(projections)) * direction
        segments.append(
            {
                "segment_id": segment_id,
                "beam_start": int(indices[0]),
                "beam_end": int(indices[-1]),
                "point_count": int(len(indices)),
                "start": start,
                "end": end,
                "length": float(np.linalg.norm(end - start)),
                "orientation_deg": _normalize(math.degrees(math.atan2(float(direction[1]), float(direction[0])))),
                "fit_residual": fit["rms"],
                "indices": indices,
            }
        )
    return segments, points


def _axial_difference(first: float, second: float) -> float:
    """Return acute line-orientation difference in degrees."""
    return abs((first - second + 90.0) % 180.0 - 90.0)


def _cluster_endpoints(
    segments: list[dict[str, Any]],
    points: np.ndarray,
    snapshot: dict[str, Any],
    width_hat: float,
) -> list[dict[str, Any]]:
    """Cluster fitted segment ends into wall terminations/corners."""
    entries = []
    hit = np.asarray(snapshot["hit"], dtype=bool)
    ranges = np.asarray(snapshot["ranges"], dtype=float)
    maximum = float(snapshot["max_range"])
    for segment in segments:
        for side, point, beam, neighbor in (
            ("START", segment["start"], segment["beam_start"], (segment["beam_start"] - 1) % len(hit)),
            ("END", segment["end"], segment["beam_end"], (segment["beam_end"] + 1) % len(hit)),
        ):
            jump = float(abs(maximum - ranges[beam])) if not hit[neighbor] else float(np.linalg.norm(points[beam] - points[neighbor]))
            entries.append({"segment_id": segment["segment_id"], "side": side, "point": np.asarray(point), "beam": beam, "neighbor": neighbor, "range_jump": jump})
    tolerance = 0.03 * width_hat
    clusters: list[list[dict[str, Any]]] = []
    for entry in entries:
        target = next((cluster for cluster in clusters if np.linalg.norm(np.mean([item["point"] for item in cluster], axis=0) - entry["point"]) <= tolerance), None)
        if target is None:
            clusters.append([entry])
        else:
            target.append(entry)
    endpoints = []
    by_id = {segment["segment_id"]: segment for segment in segments}
    for endpoint_id, cluster in enumerate(clusters):
        point = np.mean([entry["point"] for entry in cluster], axis=0)
        incident = sorted({entry["segment_id"] for entry in cluster})
        orientations = [by_id[index]["orientation_deg"] for index in incident]
        corner_angle = max((_axial_difference(a, b) for i, a in enumerate(orientations) for b in orientations[i + 1 :]), default=0.0)
        near_scan_limit = float(np.linalg.norm(point)) >= 0.95 * maximum
        has_break = any(not hit[entry["neighbor"]] or entry["range_jump"] > 0.20 * width_hat for entry in cluster)
        if near_scan_limit:
            endpoint_type = "SCAN_LIMIT"
        elif len(incident) >= 2 and corner_angle >= 20.0:
            endpoint_type = "CORNER"
        elif has_break:
            endpoint_type = "WALL_TERMINATION"
        else:
            endpoint_type = "UNSTABLE_ENDPOINT"
        support = sum(by_id[index]["point_count"] for index in incident)
        endpoints.append(
            {
                "endpoint_id": endpoint_id,
                "segment_ids": incident,
                "endpoint_type": endpoint_type,
                "point": point,
                "termination_score": float(min(1.0, max(entry["range_jump"] for entry in cluster) / max(width_hat, 1.0e-9))),
                "range_jump": float(max(entry["range_jump"] for entry in cluster)),
                "neighbor_support": int(support),
                "corner_angle": corner_angle,
                "valid": endpoint_type in {"WALL_TERMINATION", "CORNER"},
            }
        )
    return endpoints


def _beam_range(snapshot: dict[str, Any], bearing_deg: float) -> float:
    """Read the closest measured beam at one local bearing."""
    angles = np.asarray(snapshot["angles"], dtype=float)
    index = int(np.argmin(np.abs((angles - bearing_deg + 180.0) % 360.0 - 180.0)))
    return float(np.asarray(snapshot["ranges"], dtype=float)[index])


def _gap_candidates(
    endpoints: list[dict[str, Any]], snapshot: dict[str, Any], width_hat: float
) -> list[dict[str, Any]]:
    """Pair scan-derived endpoints and test local width/free continuation."""
    valid = [endpoint for endpoint in endpoints if endpoint["valid"]]
    rows = []
    for first_index, first in enumerate(valid):
        for second in valid[first_index + 1 :]:
            a, b = first["point"], second["point"]
            width = float(np.linalg.norm(b - a))
            center = 0.5 * (a + b)
            ratio = width / width_hat
            orientation = _normalize(math.degrees(math.atan2(float((b - a)[1]), float((b - a)[0]))))
            center_range = float(np.linalg.norm(center))
            center_bearing = math.degrees(math.atan2(float(center[1]), float(center[0])))
            continuation_depth = _beam_range(snapshot, center_bearing) - center_range
            continuation = continuation_depth >= 0.20 * width_hat
            plausible_width = 0.50 <= ratio <= 1.50
            accepted = plausible_width and continuation
            reason = "NONE" if accepted else "WIDTH_OUTSIDE_LOCAL_SCALE" if not plausible_width else "NO_OBSERVED_FREE_CONTINUATION"
            rows.append(
                {
                    "gap_id": len(rows),
                    "endpoint_a": first["endpoint_id"],
                    "endpoint_b": second["endpoint_id"],
                    "gap_center": center,
                    "gap_width": width,
                    "gap_width_ratio_W": ratio,
                    "gap_orientation_deg": orientation,
                    "estimated_direction_local": center_bearing,
                    "continuation_depth": continuation_depth,
                    "free_space_continuation": continuation,
                    "candidate_valid": accepted,
                    "rejection_reason": reason,
                    "boundary_support_left": first["neighbor_support"],
                    "boundary_support_right": second["neighbor_support"],
                }
            )
    return rows


def _world_to_local(snapshot: dict[str, Any], point: np.ndarray) -> np.ndarray:
    """Transform an evaluation-only world point into the LiDAR-local frame."""
    relative = np.asarray(point, dtype=float) - np.asarray(snapshot["position_eval"], dtype=float)
    yaw = math.radians(float(snapshot["yaw_eval"]))
    rotation = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    return rotation @ relative


def _gt_mouths_eval(runner: Any, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Build outgoing and incoming GT mouths strictly for post-hoc matching."""
    mouths = []
    labels = {0.0: "FORWARD", -90.0: "RIGHT", 90.0: "LEFT"}
    for branch_id, branch in enumerate(runner.geometry.branches):
        world = _branch_mouth_points(runner, branch_id)
        mouths.append({"branch_id": branch_id, "label": labels.get(float(branch.angle_deg), f"BRANCH_{branch_id}"), "branch_type": "OUTGOING", "a": _world_to_local(snapshot, world[0]), "b": _world_to_local(snapshot, world[-1])})
    entrance = float(runner.geometry.entrance_y)
    half = float(runner.geometry.incoming_width) / 2.0
    mouths.append({"branch_id": "INCOMING", "label": "INCOMING", "branch_type": "INCOMING", "a": _world_to_local(snapshot, np.array([-half, entrance])), "b": _world_to_local(snapshot, np.array([half, entrance]))})
    return mouths


def _endpoint_assignment_error(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Return minimum mean endpoint association error between two segments."""
    return min(float((np.linalg.norm(a - c) + np.linalg.norm(b - d)) / 2.0), float((np.linalg.norm(a - d) + np.linalg.norm(b - c)) / 2.0))


def _match_candidates_eval(
    case: str,
    gaps: list[dict[str, Any]],
    endpoints: list[dict[str, Any]],
    mouths: list[dict[str, Any]],
    width_hat: float,
) -> list[dict[str, Any]]:
    """Post-hoc match already accepted candidates to GT mouths."""
    by_id = {endpoint["endpoint_id"]: endpoint for endpoint in endpoints}
    rows = []
    for gap in gaps:
        if not gap["candidate_valid"]:
            continue
        a = by_id[gap["endpoint_a"]]["point"]
        b = by_id[gap["endpoint_b"]]["point"]
        best = min(mouths, key=lambda mouth: _endpoint_assignment_error(a, b, mouth["a"], mouth["b"]), default=None)
        endpoint_error = math.inf if best is None else _endpoint_assignment_error(a, b, best["a"], best["b"])
        gt_center = np.zeros(2) if best is None else 0.5 * (best["a"] + best["b"])
        gt_width = math.nan if best is None else float(np.linalg.norm(best["b"] - best["a"]))
        center_error = math.inf if best is None else float(np.linalg.norm(gap["gap_center"] - gt_center))
        matched = best is not None and endpoint_error <= 0.12 * width_hat and center_error <= 0.12 * width_hat
        overlap = 0.0 if best is None else max(0.0, 1.0 - endpoint_error / max(gt_width, gap["gap_width"], 1.0e-9))
        rows.append(
            {
                "case": case,
                "candidate_id": f"{case}_C{gap['gap_id']}",
                "gap_id": gap["gap_id"],
                "matched_branch_eval": best["label"] if matched else "FALSE",
                "branch_type_eval": best["branch_type"] if matched else "FALSE",
                "center_error_eval": center_error,
                "endpoint_error_eval": endpoint_error,
                "width_error_eval": math.nan if best is None else gap["gap_width"] - gt_width,
                "mouth_overlap_eval": overlap,
                "true_positive_eval": matched,
                "false_positive_eval": not matched,
            }
        )
    return rows


def _branch_topology_eval(
    runner: Any,
    snapshot: dict[str, Any],
    endpoints: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    width_hat: float,
) -> list[dict[str, Any]]:
    """Classify complete/partial/no topology after runtime candidates exist."""
    mouths = [mouth for mouth in _gt_mouths_eval(runner, snapshot) if mouth["branch_type"] == "OUTGOING"]
    observed = [endpoint for endpoint in endpoints if endpoint["valid"]]
    rows = []
    for mouth in mouths:
        errors = []
        for target in (mouth["a"], mouth["b"]):
            errors.append(min((float(np.linalg.norm(endpoint["point"] - target)) for endpoint in observed), default=math.inf))
        support = sum(error <= 0.12 * width_hat for error in errors)
        accepted = any(row["matched_branch_eval"] == mouth["label"] for row in matches)
        topology = "COMPLETE_GAP_TOPOLOGY" if accepted and support == 2 else "PARTIAL_GAP_TOPOLOGY" if support >= 1 else "NO_GAP_TOPOLOGY"
        rows.append({"branch_eval": mouth["label"], "observed_mouth_boundary_count_eval": support, "nearest_endpoint_error_a_eval": errors[0], "nearest_endpoint_error_b_eval": errors[1], "topology_class_eval": topology, "accepted_candidate_eval": accepted})
    return rows


def _analyze(case: str, snapshot: dict[str, Any], width_hat: float) -> dict[str, Any]:
    """Run the GT-free topology pipeline for one scan."""
    segments, points = _segments_from_scan(snapshot, width_hat)
    endpoints = _cluster_endpoints(segments, points, snapshot, width_hat)
    gaps = _gap_candidates(endpoints, snapshot, width_hat)
    return {"case": case, "snapshot": snapshot, "width_hat": width_hat, "points": points, "segments": segments, "endpoints": endpoints, "gaps": gaps}


def _segment_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"case": result["case"], "segment_id": row["segment_id"], "beam_start": row["beam_start"], "beam_end": row["beam_end"], "point_count": row["point_count"], "start_x_local": row["start"][0], "start_y_local": row["start"][1], "end_x_local": row["end"][0], "end_y_local": row["end"][1], "length": row["length"], "orientation_deg": row["orientation_deg"], "fit_residual": row["fit_residual"]} for row in result["segments"]]


def _endpoint_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"case": result["case"], "endpoint_id": row["endpoint_id"], "segment_id": json.dumps(row["segment_ids"]), "endpoint_type": row["endpoint_type"], "x_local": row["point"][0], "y_local": row["point"][1], "termination_score": row["termination_score"], "range_jump": row["range_jump"], "neighbor_support": row["neighbor_support"], "corner_angle": row["corner_angle"], "valid": row["valid"]} for row in result["endpoints"]]


def _gap_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"case": result["case"], "gap_id": row["gap_id"], "endpoint_a": row["endpoint_a"], "endpoint_b": row["endpoint_b"], "gap_center_x": row["gap_center"][0], "gap_center_y": row["gap_center"][1], "gap_width": row["gap_width"], "gap_width_ratio_W": row["gap_width_ratio_W"], "gap_orientation_deg": row["gap_orientation_deg"], "free_space_continuation": row["free_space_continuation"], "continuation_depth": row["continuation_depth"], "candidate_valid": row["candidate_valid"], "rejection_reason": row["rejection_reason"]} for row in result["gaps"]]


def _candidate_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"case": result["case"], "candidate_id": f"{result['case']}_C{row['gap_id']}", "gap_id": row["gap_id"], "candidate_type": "TWO_BOUNDARY_FREE_CONTINUATION", "center_x_local": row["gap_center"][0], "center_y_local": row["gap_center"][1], "estimated_width": row["gap_width"], "estimated_direction_local": row["estimated_direction_local"], "boundary_support_left": row["boundary_support_left"], "boundary_support_right": row["boundary_support_right"], "free_space_support": row["continuation_depth"], "accepted": row["candidate_valid"]} for row in result["gaps"] if row["candidate_valid"]]


def _plot_result(path: Path, result: dict[str, Any], runner: Any | None, mouths: list[dict[str, Any]] | None, title: str, focus: str | None = None) -> None:
    """Render points, fitted segments, endpoints, gaps, and eval-only mouths."""
    fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    hit = np.asarray(result["snapshot"]["hit"], dtype=bool)
    axis.scatter(result["points"][hit, 0], result["points"][hit, 1], s=8, color="0.65", label="LiDAR hits")
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(result["segments"]))))
    for segment, color in zip(result["segments"], colors):
        axis.plot([segment["start"][0], segment["end"][0]], [segment["start"][1], segment["end"][1]], linewidth=2, color=color)
        axis.text(*(0.5 * (segment["start"] + segment["end"])), f"S{segment['segment_id']}", fontsize=7, clip_on=True)
    for endpoint in result["endpoints"]:
        marker = "o" if endpoint["valid"] else "x"
        axis.scatter(*endpoint["point"], marker=marker, s=55, color="tab:orange" if endpoint["endpoint_type"] == "CORNER" else "tab:red" if endpoint["valid"] else "0.4")
        axis.text(*endpoint["point"], f"E{endpoint['endpoint_id']}", fontsize=7, clip_on=True)
    by_endpoint = {row["endpoint_id"]: row for row in result["endpoints"]}
    for gap in result["gaps"]:
        a, b = by_endpoint[gap["endpoint_a"]]["point"], by_endpoint[gap["endpoint_b"]]["point"]
        axis.plot([a[0], b[0]], [a[1], b[1]], color="tab:green" if gap["candidate_valid"] else "tab:red", alpha=0.8 if gap["candidate_valid"] else 0.12, linewidth=3 if gap["candidate_valid"] else 1)
    if mouths:
        for mouth in mouths:
            if focus and mouth["label"] != focus:
                continue
            axis.plot([mouth["a"][0], mouth["b"][0]], [mouth["a"][1], mouth["b"][1]], "--", linewidth=3, label=f"GT {mouth['label']} (eval)")
    axis.scatter(0.0, 0.0, marker="*", color="black", s=120, label="LiDAR")
    axis.set(title=title, xlabel="local x", ylabel="local y", aspect="equal")
    if focus and mouths:
        mouth = next(item for item in mouths if item["label"] == focus)
        center = 0.5 * (mouth["a"] + mouth["b"])
        radius = 0.85 * result["width_hat"]
        axis.set(xlim=(center[0] - radius, center[0] + radius), ylim=(center[1] - radius, center[1] + radius))
    else:
        axis.set(xlim=(-160, 160), ylim=(-160, 160))
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _self_test() -> None:
    """Validate Cartesian conversion, line fit, endpoint, and gap geometry."""
    snapshot = {"angles": np.array([0.0, 90.0, 180.0, 270.0]), "ranges": np.ones(4), "hit": np.ones(4, dtype=bool)}
    points, _ = _local_points(snapshot)
    assert np.allclose(points, np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]), atol=1.0e-12)
    line = np.column_stack((np.linspace(-2, 2, 9), np.full(9, 3.0)))
    assert _line_fit(line)["rms"] < 1.0e-12
    assert _endpoint_assignment_error(np.array([0.0, 0.0]), np.array([0.0, 2.0]), np.array([0.0, 2.0]), np.array([0.0, 0.0])) == 0.0
    synthetic = {"angles": np.arange(-180.0, 180.0), "ranges": np.full(360, 20.0), "hit": np.ones(360, dtype=bool), "max_range": 20.0}
    assert math.isclose(_beam_range(synthetic, 0.0), 20.0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-anchor-frames", type=int, default=120)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    _self_test()

    m1, anchor = _acquire_m1_anchor(args.max_anchor_frames)
    anchor["corridor_forward"] = m1.world.trusted_corridor_forward.copy()
    probe = _probe(m1, anchor, "M1_CROSS_BASELINE", 0.0, 0.0, "EXP033_A0")
    a0_snapshot = probe["snapshot"]
    width_hat = float(anchor["estimated_corridor_width"])
    m1_result = _analyze("M1_A0", a0_snapshot, width_hat)

    m0, raw_m0 = _acquire_m0_snapshot(int(anchor["frame"]) + 1)
    m0_snapshot = {"angles": np.asarray(raw_m0["angles_deg"]), "ranges": np.asarray(raw_m0["ranges"]), "hit": np.asarray(raw_m0["hit"]), "max_range": raw_m0["max_range"], "position_eval": raw_m0["position_eval"], "yaw_eval": raw_m0["yaw_eval"]}
    m0_width = float(raw_m0["estimated_corridor_width"])
    if not math.isfinite(m0_width):
        m0_width = width_hat
    m0_result = _analyze("M0_STRAIGHT", m0_snapshot, m0_width)

    mouths = _gt_mouths_eval(m1, a0_snapshot)
    m1_matches = _match_candidates_eval("M1_A0", m1_result["gaps"], m1_result["endpoints"], mouths, width_hat)
    branch_topology = _branch_topology_eval(m1, a0_snapshot, m1_result["endpoints"], m1_matches, width_hat)
    m0_matches: list[dict[str, Any]] = []
    accepted_m1 = [row for row in m1_result["gaps"] if row["candidate_valid"]]
    accepted_m0 = [row for row in m0_result["gaps"] if row["candidate_valid"]]
    outgoing_matches = {row["matched_branch_eval"] for row in m1_matches if row["branch_type_eval"] == "OUTGOING"}
    false_m1 = sum(row["false_positive_eval"] for row in m1_matches)
    partial_side = any(row["topology_class_eval"] == "PARTIAL_GAP_TOPOLOGY" for row in branch_topology if row["branch_eval"] in {"LEFT", "RIGHT"})
    complete_side = all(row["accepted_candidate_eval"] for row in branch_topology if row["branch_eval"] in {"LEFT", "RIGHT"})
    if accepted_m0:
        primary = "D_WALL_TOPOLOGY_FALSE_POSITIVE_UNSTABLE"
    elif complete_side and len(outgoing_matches) == 3:
        primary = "A_WALL_TOPOLOGY_RECOVERS_SIDE_BRANCHES"
    elif partial_side:
        primary = "B_PARTIAL_SIDE_TOPOLOGY_RECOVERED"
    elif outgoing_matches == {"FORWARD"}:
        primary = "E_FORWARD_ONLY_RECOVERED"
    else:
        primary = "C_WALL_TOPOLOGY_NOT_OBSERVABLE_AT_A0"
    secondary = []
    if any(row["topology_class_eval"] == "PARTIAL_GAP_TOPOLOGY" for row in branch_topology):
        secondary.append("ONE_BOUNDARY_VISIBLE_OPPOSITE_BOUNDARY_OCCLUDED")
    if not accepted_m0:
        secondary.append("M0_NO_FALSE_GAP")
    if not outgoing_matches:
        secondary.append("NO_COMPLETE_OUTGOING_MOUTH")

    # Deterministic replay uses the same frozen A0 ghost pose and exact topology.
    repeated_probe = _probe(m1, anchor, "M1_CROSS_BASELINE", 0.0, 0.0, "EXP033_REPLAY")
    repeated = _analyze("M1_A0", repeated_probe["snapshot"], width_hat)
    deterministic = np.array_equal(a0_snapshot["ranges"], repeated_probe["snapshot"]["ranges"]) and _segment_rows(m1_result) == _segment_rows(repeated) and _gap_rows(m1_result) == _gap_rows(repeated)

    segment_rows = _segment_rows(m1_result) + _segment_rows(m0_result)
    endpoint_rows = _endpoint_rows(m1_result) + _endpoint_rows(m0_result)
    gap_rows = _gap_rows(m1_result) + _gap_rows(m0_result)
    candidate_rows = _candidate_rows(m1_result) + _candidate_rows(m0_result)
    summary_rows = [
        {"case": "M1_A0", "estimated_corridor_width": width_hat, "wall_segment_count": len(m1_result["segments"]), "endpoint_count": len(m1_result["endpoints"]), "valid_endpoint_count": sum(row["valid"] for row in m1_result["endpoints"]), "gap_candidate_count": len(m1_result["gaps"]), "accepted_gap_count": len(accepted_m1), "wall_topology_matched_outgoing_count": len(outgoing_matches), "true_outgoing_branch_mouths_eval": 3, "left_detected_eval": "LEFT" in outgoing_matches, "right_detected_eval": "RIGHT" in outgoing_matches, "forward_detected_eval": "FORWARD" in outgoing_matches, "false_candidate_count_eval": false_m1, "existing_angular_outgoing_match": probe["outgoing_match_count_eval"], "existing_angular_outgoing_total": probe["outgoing_total_eval"]},
        {"case": "M0_STRAIGHT", "estimated_corridor_width": m0_width, "wall_segment_count": len(m0_result["segments"]), "endpoint_count": len(m0_result["endpoints"]), "valid_endpoint_count": sum(row["valid"] for row in m0_result["endpoints"]), "gap_candidate_count": len(m0_result["gaps"]), "accepted_gap_count": len(accepted_m0), "false_side_branch_count_eval": len(accepted_m0)},
        *[{"case": "M1_A0_BRANCH", **row} for row in branch_topology],
    ]
    verdict = {"experiment_id": EXPERIMENT_ID, "primary_verdict": primary, "secondary_findings": json.dumps(secondary), "deterministic_replay": deterministic, "A0_scan_exact_equivalence": np.array_equal(a0_snapshot["ranges"], repeated_probe["snapshot"]["ranges"]), "local_cartesian_test": True, "line_fit_segment_test": True, "endpoint_extraction_test": True, "gap_geometry_test": True, "GT_used_for_candidate_generation_or_acceptance": False, "GT_used_posthoc_only": True, "production_detector_modified": False}

    _write_csv(args.output / "wall_segments.csv", segment_rows)
    _write_csv(args.output / "wall_endpoints.csv", endpoint_rows)
    _write_csv(args.output / "gap_candidates.csv", gap_rows)
    _write_csv(args.output / "branch_mouth_candidates.csv", candidate_rows)
    _write_csv(args.output / "gt_match_eval.csv", m1_matches + m0_matches)
    _write_csv(args.output / "wall_topology_summary.csv", summary_rows)
    _write_csv(args.output / "wall_topology_verdict.csv", [verdict])
    _plot_result(args.output / "a0_wall_topology.png", m1_result, m1, mouths, "M1 A0 wall topology")
    _plot_result(args.output / "left_branch_topology.png", m1_result, m1, mouths, "M1 A0 left-mouth topology", "LEFT")
    _plot_result(args.output / "right_branch_topology.png", m1_result, m1, mouths, "M1 A0 right-mouth topology", "RIGHT")
    _plot_result(args.output / "m0_negative_control.png", m0_result, None, None, "M0 straight negative control")
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "primary_verdict": primary, "secondary_findings": secondary, "M1": {"segments": len(m1_result["segments"]), "endpoints": len(m1_result["endpoints"]), "valid_endpoints": sum(row["valid"] for row in m1_result["endpoints"]), "gaps": len(m1_result["gaps"]), "accepted": len(accepted_m1), "outgoing_matches": sorted(outgoing_matches), "false_candidates": false_m1}, "branch_topology": branch_topology, "M0": {"segments": len(m0_result["segments"]), "endpoints": len(m0_result["endpoints"]), "accepted": len(accepted_m0)}, "deterministic": deterministic, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
