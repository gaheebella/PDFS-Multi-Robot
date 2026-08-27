"""EXP-036 complementary multi-view branch-evidence fusion diagnostic.

Exactly three viewpoints selected by EXP-035 are deterministically replayed.
Their frozen EXP-033 wall terminations and accepted gaps are transformed into
the A0 Anchor-local frame and associated using local geometry only.  Ground
truth is attached after clustering solely for evaluation; this is not a
production fusion implementation or a movement experiment.
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
from types import SimpleNamespace
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.integration.run_final_nonforward_viewpoint_side_topology_observability import (
    DEFAULT_SOURCE,
    SAFE_SOURCE,
    _analyze_viewpoint,
    _anchor_from_existing_source,
    _as_bool,
    _m0_negative_replay,
    _read,
    _rear_start_geometry,
    _topology_row,
    _viewpoint_id,
    _write_required,
)
from junction_detection.integration.run_wall_topology_branch_opening_diagnostic import (
    _endpoint_assignment_error,
    _self_test,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import LidarSensor

EXPERIMENT_ID = "EXP-036"
DEFAULT_EXP035_OUTPUT = (
    ROOT
    / "junction_detection/integration/output/final_nonforward_side_topology_observability"
)
DEFAULT_OUTPUT = (
    ROOT
    / "junction_detection/integration/output/complementary_multiview_branch_evidence_fusion"
)
VIEW_SPECS = (
    ("A0", "A0"),
    ("F+0.700_L-0.100", "LEFT_COMPLETE"),
    ("F+0.700_L+0.100", "RIGHT_COMPLETE"),
)
NOMINAL_DISTANCE_TOLERANCE_W = 0.075
SENSITIVITY_TOLERANCES_W = (0.050, 0.075, 0.100)
WIDTH_DIFFERENCE_TOLERANCE_W = 0.100
ORIENTATION_TOLERANCE_DEG = 12.0
POSTHOC_MATCH_TOLERANCE_W = 0.120  # Frozen EXP-033 evaluation tolerance.


def _axial_orientation(a: np.ndarray, b: np.ndarray) -> float:
    """Return an undirected line orientation in [0, 180) degrees."""
    delta = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 180.0


def _axial_difference(first: float, second: float) -> float:
    """Return the smallest difference between two undirected orientations."""
    difference = abs((first - second) % 180.0)
    return min(difference, 180.0 - difference)


def _axial_mean(values: list[float]) -> float:
    """Aggregate axial angles with a doubled-angle circular mean."""
    radians = np.radians(np.asarray(values, dtype=float) * 2.0)
    angle = 0.5 * math.degrees(math.atan2(float(np.sin(radians).mean()), float(np.cos(radians).mean())))
    return angle % 180.0


def _sensor_to_world(snapshot: dict[str, Any], point: np.ndarray) -> np.ndarray:
    """Transform one sensor-local point into the ray-casting world frame."""
    yaw = math.radians(float(snapshot["yaw_eval"]))
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    )
    return np.asarray(snapshot["position_eval"], dtype=float) + rotation @ np.asarray(point, dtype=float)


def _world_to_common(anchor: dict[str, Any], point: np.ndarray) -> np.ndarray:
    """Express a world point as [Anchor-forward, Anchor-left] in world units."""
    forward = np.asarray(anchor["corridor_forward"], dtype=float)
    forward /= np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0]])
    relative = np.asarray(point, dtype=float) - np.asarray(anchor["position_eval"], dtype=float)
    return np.array([float(relative @ forward), float(relative @ left)])


def _sensor_to_common(
    anchor: dict[str, Any], snapshot: dict[str, Any], point: np.ndarray
) -> np.ndarray:
    """Transform sensor-local evidence directly into the A0 common frame."""
    return _world_to_common(anchor, _sensor_to_world(snapshot, point))


def _common_to_sensor(
    anchor: dict[str, Any], snapshot: dict[str, Any], point: np.ndarray
) -> np.ndarray:
    """Inverse transform used only by the common-frame round-trip test."""
    forward = np.asarray(anchor["corridor_forward"], dtype=float)
    forward /= np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0]])
    world = (
        np.asarray(anchor["position_eval"], dtype=float)
        + forward * float(point[0])
        + left * float(point[1])
    )
    yaw = math.radians(float(snapshot["yaw_eval"]))
    inverse_rotation = np.array(
        [[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]]
    )
    return inverse_rotation @ (world - np.asarray(snapshot["position_eval"], dtype=float))


def _connected_components(
    records: list[dict[str, Any]], compatible: Callable[[dict[str, Any], dict[str, Any]], bool]
) -> list[list[int]]:
    """Return deterministic connected components under a symmetric relation."""
    adjacency = [set() for _ in records]
    for first, second in itertools.combinations(range(len(records)), 2):
        if compatible(records[first], records[second]):
            adjacency[first].add(second)
            adjacency[second].add(first)
    components = []
    unseen = set(range(len(records)))
    while unseen:
        root = min(unseen)
        stack, component = [root], []
        unseen.remove(root)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _wall_orientation(result: dict[str, Any], endpoint: dict[str, Any]) -> float:
    """Aggregate orientations of the fitted walls supporting one termination."""
    by_id = {segment["segment_id"]: segment for segment in result["segments"]}
    return _axial_mean([float(by_id[index]["orientation_deg"]) % 180.0 for index in endpoint["segment_ids"]])


def _extract_evidence(
    analysis: dict[str, Any], anchor: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract common-frame wall segments, valid terminations, and accepted gaps."""
    viewpoint = analysis["spec"]["viewpoint_id"]
    result = analysis["result"]
    segments = []
    for segment in result["segments"]:
        start = _sensor_to_common(anchor, analysis["snapshot"], segment["start"])
        end = _sensor_to_common(anchor, analysis["snapshot"], segment["end"])
        segments.append(
            {
                "view_id": viewpoint,
                "segment_id": segment["segment_id"],
                "start": start,
                "end": end,
                "orientation": _axial_orientation(start, end),
            }
        )
    terminations = []
    by_endpoint = {row["endpoint_id"]: row for row in result["endpoints"]}
    for endpoint in result["endpoints"]:
        if not endpoint["valid"]:
            continue
        point = _sensor_to_common(anchor, analysis["snapshot"], endpoint["point"])
        terminations.append(
            {
                "view_id": viewpoint,
                "termination_id": f"{viewpoint}_E{endpoint['endpoint_id']}",
                "source_endpoint_id": endpoint["endpoint_id"],
                "point": point,
                "endpoint_type": endpoint["endpoint_type"],
                "wall_orientation": _wall_orientation(result, endpoint),
            }
        )
    gaps = []
    match_by_gap = {row["gap_id"]: row for row in analysis["matches"]}
    for gap in result["gaps"]:
        if not gap["candidate_valid"]:
            continue
        endpoint_a = _sensor_to_common(
            anchor, analysis["snapshot"], by_endpoint[gap["endpoint_a"]]["point"]
        )
        endpoint_b = _sensor_to_common(
            anchor, analysis["snapshot"], by_endpoint[gap["endpoint_b"]]["point"]
        )
        match = match_by_gap[gap["gap_id"]]
        gaps.append(
            {
                "view_id": viewpoint,
                "gap_id": f"{viewpoint}_G{gap['gap_id']}",
                "source_gap_id": gap["gap_id"],
                "endpoint_a": endpoint_a,
                "endpoint_b": endpoint_b,
                "center": 0.5 * (endpoint_a + endpoint_b),
                "width": float(np.linalg.norm(endpoint_b - endpoint_a)),
                "orientation": _axial_orientation(endpoint_a, endpoint_b),
                "continuation_depth": float(gap["continuation_depth"]),
                # This post-hoc field is never read by association functions.
                "source_gt_class_eval": match["matched_branch_eval"],
            }
        )
    return segments, terminations, gaps


def _termination_association(
    terminations: list[dict[str, Any]], width: float, tolerance_w: float
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Cluster same-type, same-wall-direction terminations by common position."""
    distance_tolerance = tolerance_w * width

    def compatible(first: dict[str, Any], second: dict[str, Any]) -> bool:
        return (
            first["view_id"] != second["view_id"]
            and first["endpoint_type"] == second["endpoint_type"]
            and float(np.linalg.norm(first["point"] - second["point"])) <= distance_tolerance
            and _axial_difference(first["wall_orientation"], second["wall_orientation"])
            <= ORIENTATION_TOLERANCE_DEG
        )

    rows, membership = [], {}
    for index, indices in enumerate(_connected_components(terminations, compatible)):
        members = [terminations[item] for item in indices]
        points = np.asarray([item["point"] for item in members])
        center = np.median(points, axis=0)
        cluster_id = f"T{index}"
        for item in members:
            membership[item["termination_id"]] = cluster_id
        rows.append(
            {
                "termination_cluster_id": cluster_id,
                "supporting_view_count": len({item["view_id"] for item in members}),
                "supporting_views": json.dumps(sorted({item["view_id"] for item in members})),
                "member_count": len(members),
                "member_ids": json.dumps([item["termination_id"] for item in members]),
                "center_x": center[0],
                "center_y": center[1],
                "representative_wall_orientation": _axial_mean([item["wall_orientation"] for item in members]),
                "endpoint_type": members[0]["endpoint_type"],
                "position_spread": max(float(np.linalg.norm(item["point"] - center)) for item in members),
                "position_spread_W": max(float(np.linalg.norm(item["point"] - center)) for item in members) / width,
            }
        )
    return rows, membership


def _endpoint_pair_error(first: dict[str, Any], second: dict[str, Any]) -> float:
    """Return minimum mean endpoint correspondence error for two gaps."""
    return _endpoint_assignment_error(
        first["endpoint_a"], first["endpoint_b"], second["endpoint_a"], second["endpoint_b"]
    )


def _gap_pair_metrics(
    first: dict[str, Any], second: dict[str, Any], width: float, tolerance_w: float
) -> dict[str, Any]:
    """Compute GT-free pairwise geometry metrics and association decision."""
    center_distance = float(np.linalg.norm(first["center"] - second["center"]))
    width_difference = abs(first["width"] - second["width"])
    orientation_difference = _axial_difference(first["orientation"], second["orientation"])
    endpoint_error = _endpoint_pair_error(first, second)
    different_views = first["view_id"] != second["view_id"]
    associated = (
        different_views
        and center_distance <= tolerance_w * width
        and endpoint_error <= tolerance_w * width
        and width_difference <= WIDTH_DIFFERENCE_TOLERANCE_W * width
        and orientation_difference <= ORIENTATION_TOLERANCE_DEG
    )
    return {
        "gap_1": first["gap_id"],
        "gap_2": second["gap_id"],
        "view_1": first["view_id"],
        "view_2": second["view_id"],
        "center_distance": center_distance,
        "center_distance_W": center_distance / width,
        "endpoint_correspondence_error": endpoint_error,
        "endpoint_correspondence_error_W": endpoint_error / width,
        "width_difference": width_difference,
        "width_difference_W": width_difference / width,
        "orientation_difference_deg": orientation_difference,
        "distance_tolerance_W": tolerance_w,
        "width_tolerance_W": WIDTH_DIFFERENCE_TOLERANCE_W,
        "orientation_tolerance_deg": ORIENTATION_TOLERANCE_DEG,
        "associated": associated,
    }


def _aligned_endpoints(members: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Align gap endpoint order to the first member, then return component medians."""
    reference_a, reference_b = members[0]["endpoint_a"], members[0]["endpoint_b"]
    first, second = [], []
    for member in members:
        direct = np.linalg.norm(member["endpoint_a"] - reference_a) + np.linalg.norm(member["endpoint_b"] - reference_b)
        swapped = np.linalg.norm(member["endpoint_b"] - reference_a) + np.linalg.norm(member["endpoint_a"] - reference_b)
        if direct <= swapped:
            first.append(member["endpoint_a"])
            second.append(member["endpoint_b"])
        else:
            first.append(member["endpoint_b"])
            second.append(member["endpoint_a"])
    return np.median(np.asarray(first), axis=0), np.median(np.asarray(second), axis=0)


def _gap_association(
    gaps: list[dict[str, Any]], width: float, tolerance_w: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Associate accepted gaps and aggregate each component with coordinate medians."""
    pair_rows = [
        _gap_pair_metrics(first, second, width, tolerance_w)
        for first, second in itertools.combinations(gaps, 2)
        if first["view_id"] != second["view_id"]
    ]
    associated_pairs = {frozenset((row["gap_1"], row["gap_2"])) for row in pair_rows if row["associated"]}

    def compatible(first: dict[str, Any], second: dict[str, Any]) -> bool:
        return frozenset((first["gap_id"], second["gap_id"])) in associated_pairs

    fused = []
    for index, indices in enumerate(_connected_components(gaps, compatible)):
        members = [gaps[item] for item in indices]
        endpoint_a, endpoint_b = _aligned_endpoints(members)
        centers = np.asarray([item["center"] for item in members])
        center = np.median(centers, axis=0)
        views = sorted({item["view_id"] for item in members})
        fused.append(
            {
                "fusion_candidate_id": f"F{index}",
                "member_gap_ids": json.dumps([item["gap_id"] for item in members]),
                "complete_support_count": len(views),
                "complete_supporting_views": json.dumps(views),
                "representative_center": center,
                "representative_endpoint_a": endpoint_a,
                "representative_endpoint_b": endpoint_b,
                "representative_width": float(np.median([item["width"] for item in members])),
                "representative_orientation": _axial_mean([item["orientation"] for item in members]),
                "association_spread": max(float(np.linalg.norm(item["center"] - center)) for item in members),
                "association_spread_W": max(float(np.linalg.norm(item["center"] - center)) for item in members) / width,
                "source_members": members,
            }
        )
    return pair_rows, fused


def _attach_termination_support(
    fused: list[dict[str, Any]],
    termination_clusters: list[dict[str, Any]],
    width: float,
    tolerance_w: float,
) -> None:
    """Attach partial/termination support without reading post-hoc branch labels."""
    for candidate in fused:
        endpoints = (candidate["representative_endpoint_a"], candidate["representative_endpoint_b"])
        attached = []
        for endpoint in endpoints:
            cluster = min(
                termination_clusters,
                key=lambda row: np.linalg.norm(endpoint - np.array([row["center_x"], row["center_y"]])),
                default=None,
            )
            if cluster is not None:
                distance = float(np.linalg.norm(endpoint - np.array([cluster["center_x"], cluster["center_y"]])))
                if distance <= tolerance_w * width:
                    attached.append(cluster)
        # A cluster can support both endpoints only in degenerate geometry; count once.
        attached = list({row["termination_cluster_id"]: row for row in attached}.values())
        complete_views = set(json.loads(candidate["complete_supporting_views"]))
        termination_views = set()
        termination_observations = 0
        for cluster in attached:
            termination_views.update(json.loads(cluster["supporting_views"]))
            termination_observations += int(cluster["member_count"])
        partial_views = termination_views - complete_views
        all_views = complete_views | termination_views
        candidate["endpoint_cluster_a"] = attached[0]["termination_cluster_id"] if attached else ""
        candidate["endpoint_cluster_b"] = attached[1]["termination_cluster_id"] if len(attached) > 1 else ""
        candidate["termination_support_count"] = termination_observations
        candidate["partial_support_count"] = len(partial_views)
        candidate["partial_supporting_views"] = json.dumps(sorted(partial_views))
        candidate["supporting_view_count"] = len(all_views)
        candidate["supporting_views"] = json.dumps(sorted(all_views))


def _common_gt_mouths_eval(
    analysis: dict[str, Any], anchor: dict[str, Any]
) -> list[dict[str, Any]]:
    """Transform A0 post-hoc mouths after fusion has completed."""
    return [
        {
            "label": mouth["label"],
            "branch_type": mouth["branch_type"],
            "a": _sensor_to_common(anchor, analysis["snapshot"], mouth["a"]),
            "b": _sensor_to_common(anchor, analysis["snapshot"], mouth["b"]),
        }
        for mouth in analysis["mouths"]
    ]


def _evaluate_fused_posthoc(
    fused: list[dict[str, Any]], mouths: list[dict[str, Any]], width: float
) -> list[dict[str, Any]]:
    """Attach GT labels only after all geometry association and support counts."""
    rows = []
    false_index = 0
    for candidate in fused:
        a, b = candidate["representative_endpoint_a"], candidate["representative_endpoint_b"]
        best = min(mouths, key=lambda mouth: _endpoint_assignment_error(a, b, mouth["a"], mouth["b"]))
        endpoint_error = _endpoint_assignment_error(a, b, best["a"], best["b"])
        center = candidate["representative_center"]
        gt_center = 0.5 * (best["a"] + best["b"])
        center_error = float(np.linalg.norm(center - gt_center))
        gt_width = float(np.linalg.norm(best["b"] - best["a"]))
        matched = endpoint_error <= POSTHOC_MATCH_TOLERANCE_W * width and center_error <= POSTHOC_MATCH_TOLERANCE_W * width
        if matched:
            label = best["label"]
        else:
            label = f"FALSE_{false_index}"
            false_index += 1
        overlap = max(0.0, 1.0 - endpoint_error / max(gt_width, candidate["representative_width"], 1.0e-9))
        rows.append(
            {
                "fusion_candidate_id": candidate["fusion_candidate_id"],
                "GT_class_eval": label,
                "nearest_GT_mouth_eval": best["label"],
                "center_error_eval": center_error,
                "endpoint_error_eval": endpoint_error,
                "width_error_eval": candidate["representative_width"] - gt_width,
                "mouth_overlap_eval": overlap,
                "GT_match_eval": matched,
            }
        )
    return rows


def _source_equivalence(
    analyses: list[dict[str, Any]], exp035_output: Path
) -> tuple[list[dict[str, Any]], bool]:
    """Compare all three replays with the persisted EXP-035 source/topology rows."""
    audit = {row["viewpoint_id"]: row for row in _read(exp035_output / "source_viewpoint_audit.csv")}
    topology = {row["viewpoint_id"]: row for row in _read(exp035_output / "viewpoint_topology_grid.csv")}
    common_yaw = float(analyses[0]["snapshot"]["yaw_eval"])
    rows = []
    for analysis, (_, role) in zip(analyses, VIEW_SPECS):
        view_id = analysis["spec"]["viewpoint_id"]
        current = _topology_row(analysis)
        source, expected = audit[view_id], topology[view_id]
        false_count = sum(bool(row["false_positive_eval"]) for row in analysis["matches"])
        pass_scan = (
            analysis["hit_count"] == int(source["reconstructed_hit_count"])
            and analysis["max_count"] == int(source["reconstructed_max_range_count"])
            and analysis["opening_count"] == int(source["reconstructed_opening_count"])
        )
        pass_topology = (
            current["left_topology"] == expected["left_topology"]
            and current["right_topology"] == expected["right_topology"]
            and current["incoming_topology"] == expected["incoming_topology"]
            and current["false_accepted_gap_count"] == int(expected["false_accepted_gap_count"])
        )
        role_pass = (
            (role == "A0" and current["left_topology"] == current["right_topology"] == "PARTIAL_GAP_TOPOLOGY" and false_count == 0)
            or (role == "LEFT_COMPLETE" and current["left_topology"] == "COMPLETE_GAP_TOPOLOGY" and false_count == 1)
            or (role == "RIGHT_COMPLETE" and current["right_topology"] == "COMPLETE_GAP_TOPOLOGY" and false_count == 1)
        )
        rows.append(
            {
                "view_id": view_id,
                "view_role": role,
                "forward_ratio_W": float(analysis["source"]["forward_ratio_W"]),
                "lateral_ratio_W": float(analysis["source"]["lateral_ratio_W"]),
                "pose_x_local": float(analysis["source"]["forward_ratio_W"]) * analysis["spec"]["estimated_corridor_width"],
                "pose_y_local": float(analysis["source"]["lateral_ratio_W"]) * analysis["spec"]["estimated_corridor_width"],
                "yaw_local": float(analysis["snapshot"]["yaw_eval"]) - common_yaw,
                "viewpoint_geometry_valid": _as_bool(source["viewpoint_geometry_valid"]),
                "movement_feasibility_proven": False,
                "scan_source": source["scan_source_type"],
                "hit_count": analysis["hit_count"],
                "max_range_count": analysis["max_count"],
                "opening_count": analysis["opening_count"],
                "left_topology": current["left_topology"],
                "right_topology": current["right_topology"],
                "incoming_topology": current["incoming_topology"],
                "false_accepted_gap_count": false_count,
                "scan_summary_equivalent": pass_scan,
                "topology_equivalent": pass_topology,
                "role_expectation_equivalent": role_pass,
                "source_equivalence_pass": pass_scan and pass_topology and role_pass,
            }
        )
    return rows, all(row["source_equivalence_pass"] for row in rows)


def _view_evidence_rows(
    analyses: list[dict[str, Any]], terminations: list[dict[str, Any]], gaps: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build a unified evidence table; runtime and post-hoc types stay separate."""
    topology = {analysis["spec"]["viewpoint_id"]: _topology_row(analysis) for analysis in analyses}
    rows = []
    for item in terminations:
        rows.append(
            {
                "view_id": item["view_id"],
                "evidence_id": item["termination_id"],
                "runtime_evidence_type": "WALL_TERMINATION",
                "evidence_type": "WALL_TERMINATION",
                "center_x_common": item["point"][0],
                "center_y_common": item["point"][1],
                "endpoint_a_x": item["point"][0],
                "endpoint_a_y": item["point"][1],
                "endpoint_b_x": "",
                "endpoint_b_y": "",
                "width": 0.0,
                "orientation": item["wall_orientation"],
                "topology_state": "PARTIAL_GAP_TOPOLOGY",
                "GT_class_eval": "",
            }
        )
    for item in gaps:
        posthoc = item["source_gt_class_eval"]
        rows.append(
            {
                "view_id": item["view_id"],
                "evidence_id": item["gap_id"],
                "runtime_evidence_type": "ACCEPTED_GAP",
                "evidence_type": "OTHER_GAP" if posthoc == "FALSE" else "COMPLETE_GAP",
                "center_x_common": item["center"][0],
                "center_y_common": item["center"][1],
                "endpoint_a_x": item["endpoint_a"][0],
                "endpoint_a_y": item["endpoint_a"][1],
                "endpoint_b_x": item["endpoint_b"][0],
                "endpoint_b_y": item["endpoint_b"][1],
                "width": item["width"],
                "orientation": item["orientation"],
                "topology_state": "COMPLETE_GAP_TOPOLOGY",
                "GT_class_eval": posthoc,
            }
        )
    # Source branch topology is included for audit only, never association.
    for analysis in analyses:
        view_id = analysis["spec"]["viewpoint_id"]
        for side in ("left", "right"):
            if topology[view_id][f"{side}_topology"] != "PARTIAL_GAP_TOPOLOGY":
                continue
            rows.append(
                {
                    "view_id": view_id,
                    "evidence_id": f"{view_id}_{side.upper()}_PARTIAL_EVAL",
                    "runtime_evidence_type": "PARTIAL_TERMINATION_SET",
                    "evidence_type": "PARTIAL_BRANCH",
                    "center_x_common": "",
                    "center_y_common": "",
                    "endpoint_a_x": "",
                    "endpoint_a_y": "",
                    "endpoint_b_x": "",
                    "endpoint_b_y": "",
                    "width": "",
                    "orientation": "",
                    "topology_state": "PARTIAL_GAP_TOPOLOGY",
                    "GT_class_eval": side.upper(),
                }
            )
    return rows


def _fused_rows(fused: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove in-memory source objects and serialize fused runtime geometry."""
    return [
        {
            "fusion_candidate_id": row["fusion_candidate_id"],
            "supporting_view_count": row["supporting_view_count"],
            "supporting_views": row["supporting_views"],
            "complete_support_count": row["complete_support_count"],
            "complete_supporting_views": row["complete_supporting_views"],
            "partial_support_count": row["partial_support_count"],
            "partial_supporting_views": row["partial_supporting_views"],
            "termination_support_count": row["termination_support_count"],
            "center_x": row["representative_center"][0],
            "center_y": row["representative_center"][1],
            "width": row["representative_width"],
            "orientation": row["representative_orientation"],
            "endpoint_cluster_a": row["endpoint_cluster_a"],
            "endpoint_cluster_b": row["endpoint_cluster_b"],
            "association_spread": row["association_spread"],
            "association_spread_W": row["association_spread_W"],
            "member_gap_ids": row["member_gap_ids"],
        }
        for row in fused
    ]


def _comparison_rows(
    fused_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare single-view accepted gaps with descriptive multi-view support."""
    evaluation = {row["fusion_candidate_id"]: row for row in eval_rows}
    rows = []
    for candidate in fused_rows:
        current = evaluation[candidate["fusion_candidate_id"]]
        consistent = candidate["supporting_view_count"] >= 2
        rows.append(
            {
                "candidate_eval": candidate["fusion_candidate_id"],
                "single_view_support_pattern": candidate["complete_supporting_views"],
                "multiview_support_count": candidate["supporting_view_count"],
                "complete_support_count": candidate["complete_support_count"],
                "partial_support_count": candidate["partial_support_count"],
                "termination_support_count": candidate["termination_support_count"],
                "multiview_consistent": consistent,
                "GT_class_eval": current["GT_class_eval"],
                "false_positive_single_view": current["GT_class_eval"].startswith("FALSE"),
                # Evaluation-only hypothetical support rule; not production logic.
                "false_positive_after_consistency_rule_eval": current["GT_class_eval"].startswith("FALSE") and consistent,
            }
        )
    return rows


def _plot_common_evidence(
    path: Path,
    all_segments: list[dict[str, Any]],
    terminations: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    width: float,
) -> None:
    """Overlay all scan-only evidence in the A0 common frame."""
    colors = {"A0": "black", "F+0.700_L-0.100": "tab:blue", "F+0.700_L+0.100": "tab:orange"}
    fig, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    for view_id, color in colors.items():
        segments = [row for row in all_segments if row["view_id"] == view_id]
        for index, segment in enumerate(segments):
            axis.plot(
                [segment["start"][1] / width, segment["end"][1] / width],
                [segment["start"][0] / width, segment["end"][0] / width],
                color=color,
                alpha=0.55,
                linewidth=2,
                label=f"{view_id} walls" if index == 0 else None,
            )
        points = [row for row in terminations if row["view_id"] == view_id]
        axis.scatter(
            [row["point"][1] / width for row in points],
            [row["point"][0] / width for row in points],
            color=color,
            marker="o",
            s=55,
            label=f"{view_id} terminations",
        )
        for gap in [row for row in gaps if row["view_id"] == view_id]:
            axis.plot(
                [gap["endpoint_a"][1] / width, gap["endpoint_b"][1] / width],
                [gap["endpoint_a"][0] / width, gap["endpoint_b"][0] / width],
                color=color,
                linewidth=3,
                linestyle="--",
            )
    axis.set(
        xlabel="common lateral / W (positive = Anchor-left)",
        ylabel="common forward / W",
        title="EXP-036 three-view evidence in the A0 common frame",
        aspect="equal",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_fused(
    path: Path, fused: list[dict[str, Any]], eval_rows: list[dict[str, Any]], width: float
) -> None:
    """Render representative gap geometry and support counts after association."""
    evaluation = {row["fusion_candidate_id"]: row for row in eval_rows}
    fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    label_offsets = {
        "INCOMING": (0.0, -0.10),
        "LEFT": (-0.18, 0.02),
        "RIGHT": (0.18, 0.02),
        "FALSE_0": (-0.22, -0.12),
        "FALSE_1": (0.22, 0.12),
    }
    for candidate in fused:
        label = evaluation[candidate["fusion_candidate_id"]]["GT_class_eval"]
        a, b = candidate["representative_endpoint_a"] / width, candidate["representative_endpoint_b"] / width
        color = "tab:red" if label.startswith("FALSE") else "tab:green"
        axis.plot([a[1], b[1]], [a[0], b[0]], color=color, linewidth=4)
        center = candidate["representative_center"] / width
        offset_x, offset_y = label_offsets.get(label, (0.0, 0.0))
        axis.text(
            center[1] + offset_x,
            center[0] + offset_y,
            f"{candidate['fusion_candidate_id']} {label} eval\nviews={candidate['supporting_view_count']} complete={candidate['complete_support_count']}",
            ha="center",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
        )
    axis.scatter(0.0, 0.0, marker="*", color="black", s=120, label="A0")
    axis.set(xlabel="common lateral / W", ylabel="common forward / W", title="Fused candidate clusters (GT labels are post-hoc)", aspect="equal")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_support(
    path: Path, fused_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]
) -> None:
    """Compare complete, partial, and termination support with post-hoc labels."""
    evaluation = {row["fusion_candidate_id"]: row for row in eval_rows}
    labels = [f"{row['fusion_candidate_id']}\n{evaluation[row['fusion_candidate_id']]['GT_class_eval']}" for row in fused_rows]
    positions = np.arange(len(fused_rows))
    fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    axis.bar(positions - 0.25, [row["complete_support_count"] for row in fused_rows], 0.25, label="complete gap views")
    axis.bar(positions, [row["partial_support_count"] for row in fused_rows], 0.25, label="partial views")
    axis.bar(positions + 0.25, [row["termination_support_count"] for row in fused_rows], 0.25, label="termination observations")
    axis.set(xticks=positions, xticklabels=labels, ylabel="support count", title="True vs false multi-view support (labels post-hoc)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_branch(
    path: Path,
    candidate: dict[str, Any],
    termination_clusters: list[dict[str, Any]],
    width: float,
    title: str,
) -> None:
    """Show one side gap and its associated partial termination clusters."""
    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    a, b = candidate["representative_endpoint_a"] / width, candidate["representative_endpoint_b"] / width
    axis.plot([a[1], b[1]], [a[0], b[0]], color="tab:green", linewidth=4, label="complete gap representative")
    attached = {candidate["endpoint_cluster_a"], candidate["endpoint_cluster_b"]} - {""}
    for cluster in termination_clusters:
        if cluster["termination_cluster_id"] not in attached:
            continue
        axis.scatter(cluster["center_y"] / width, cluster["center_x"] / width, s=90, label=f"{cluster['termination_cluster_id']} views={cluster['supporting_view_count']}")
    axis.set(xlabel="common lateral / W", ylabel="common forward / W", title=title, aspect="equal")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _association_self_test() -> None:
    """Check orientation-aware gap association and common cluster mechanics."""
    width = 10.0
    base = {"view_id": "A", "gap_id": "A0", "endpoint_a": np.array([0.0, -5.0]), "endpoint_b": np.array([0.0, 5.0]), "center": np.zeros(2), "width": 10.0, "orientation": 90.0}
    same = {**base, "view_id": "B", "gap_id": "B0", "endpoint_a": np.array([0.1, -5.0]), "endpoint_b": np.array([0.1, 5.0]), "center": np.array([0.1, 0.0]), "orientation": 90.0}
    orthogonal = {**base, "view_id": "C", "gap_id": "C0", "endpoint_a": np.array([-5.0, 0.0]), "endpoint_b": np.array([5.0, 0.0]), "center": np.zeros(2), "orientation": 0.0}
    assert _gap_pair_metrics(base, same, width, 0.075)["associated"]
    assert not _gap_pair_metrics(base, orthogonal, width, 0.075)["associated"]
    assert math.isclose(_axial_difference(179.0, 1.0), 2.0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--safe-source", type=Path, default=SAFE_SOURCE)
    parser.add_argument("--exp035-output", type=Path, default=DEFAULT_EXP035_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    _self_test()
    _association_self_test()

    source = _read(args.source)
    by_id = {_viewpoint_id(row): row for row in source}
    selected = [by_id[view_id] for view_id, _ in VIEW_SPECS]
    anchor = _anchor_from_existing_source(args.safe_source)
    geometry = _rear_start_geometry()
    sensor = LidarSensor()
    analyses = [_analyze_viewpoint(row, anchor, sensor, geometry) for row in selected]
    source_rows, source_equivalent = _source_equivalence(analyses, args.exp035_output)

    # Exact replay is checked independently of the persisted summary counts.
    repeated = [_analyze_viewpoint(row, anchor, sensor, geometry) for row in selected]
    deterministic = all(
        np.array_equal(first["snapshot"]["ranges"], second["snapshot"]["ranges"])
        and _topology_row(first) == _topology_row(second)
        for first, second in zip(analyses, repeated)
    )
    width = float(anchor["estimated_corridor_width"])

    all_segments, terminations, gaps = [], [], []
    for analysis in analyses:
        segments, endpoint_rows, gap_rows = _extract_evidence(analysis, anchor)
        all_segments.extend(segments)
        terminations.extend(endpoint_rows)
        gaps.extend(gap_rows)

    termination_clusters, _ = _termination_association(
        terminations, width, NOMINAL_DISTANCE_TOLERANCE_W
    )
    pair_rows, fused = _gap_association(gaps, width, NOMINAL_DISTANCE_TOLERANCE_W)
    _attach_termination_support(fused, termination_clusters, width, NOMINAL_DISTANCE_TOLERANCE_W)

    # Frame validation: represent one physical point in every sensor frame and
    # recover a single common point. No map/branch information is involved.
    test_common = np.array([1.2 * width, 0.4 * width])
    transform_errors = []
    for analysis in analyses:
        sensor_point = _common_to_sensor(anchor, analysis["snapshot"], test_common)
        recovered = _sensor_to_common(anchor, analysis["snapshot"], sensor_point)
        transform_errors.append(float(np.linalg.norm(recovered - test_common)))
    transform_max_error = max(transform_errors)

    # Association is now finished. GT evaluation begins only below this line.
    mouths = _common_gt_mouths_eval(analyses[0], anchor)
    eval_rows = _evaluate_fused_posthoc(fused, mouths, width)
    fused_rows = _fused_rows(fused)
    evaluation = {row["fusion_candidate_id"]: row for row in eval_rows}
    support_rows = [
        {**row, **evaluation[row["fusion_candidate_id"]]}
        for row in fused_rows
    ]
    comparison = _comparison_rows(fused_rows, eval_rows)

    eval_by_label = {row["GT_class_eval"]: row for row in support_rows}
    left = eval_by_label.get("LEFT")
    right = eval_by_label.get("RIGHT")
    incoming = eval_by_label.get("INCOMING")
    false_rows = sorted(
        (row for row in support_rows if row["GT_class_eval"].startswith("FALSE")),
        key=lambda row: row["GT_class_eval"],
    )

    # Partial-to-complete means at least one non-complete view observes an
    # endpoint cluster belonging to the side candidate; no GT drove attachment.
    left_partial_complete = left is not None and left["partial_support_count"] > 0
    right_partial_complete = right is not None and right["partial_support_count"] > 0
    false_pair = None
    if len(false_rows) >= 2:
        first = next(item for item in fused if item["fusion_candidate_id"] == false_rows[0]["fusion_candidate_id"])
        second = next(item for item in fused if item["fusion_candidate_id"] == false_rows[1]["fusion_candidate_id"])
        false_pair = _gap_pair_metrics(first["source_members"][0], second["source_members"][0], width, NOMINAL_DISTANCE_TOLERANCE_W)

    sensitivity = []
    for tolerance in SENSITIVITY_TOLERANCES_W:
        term_clusters, _ = _termination_association(terminations, width, tolerance)
        _, candidates = _gap_association(gaps, width, tolerance)
        _attach_termination_support(candidates, term_clusters, width, tolerance)
        candidate_eval = _evaluate_fused_posthoc(candidates, mouths, width)
        joined = [
            {**row, **next(item for item in candidate_eval if item["fusion_candidate_id"] == row["fusion_candidate_id"])}
            for row in _fused_rows(candidates)
        ]
        sensitivity.append(
            {
                "distance_tolerance_W": tolerance,
                "termination_cluster_count": len(term_clusters),
                "fused_candidate_count": len(candidates),
                "left_supporting_view_count": next((row["supporting_view_count"] for row in joined if row["GT_class_eval"] == "LEFT"), 0),
                "right_supporting_view_count": next((row["supporting_view_count"] for row in joined if row["GT_class_eval"] == "RIGHT"), 0),
                "incoming_complete_support_count": next((row["complete_support_count"] for row in joined if row["GT_class_eval"] == "INCOMING"), 0),
                "false_candidate_count": sum(row["GT_class_eval"].startswith("FALSE") for row in joined),
                "maximum_false_supporting_view_count": max((row["supporting_view_count"] for row in joined if row["GT_class_eval"].startswith("FALSE")), default=0),
            }
        )
    sensitivity_stable = len({tuple(row.values())[1:] for row in sensitivity}) == 1

    m0 = _m0_negative_replay(sensor, width)
    true_side_support = [row["supporting_view_count"] for row in (left, right) if row is not None]
    false_support = [row["supporting_view_count"] for row in false_rows]
    clear_separation = bool(true_side_support and false_support and min(true_side_support) > max(false_support))
    false_ambiguous = bool(false_support and true_side_support and max(false_support) >= min(true_side_support))

    if not source_equivalent or not deterministic or transform_max_error > 1.0e-10:
        verdict = "E_SOURCE_RECONSTRUCTION_OR_FRAME_ALIGNMENT_INCONSISTENT"
    elif not left_partial_complete or not right_partial_complete:
        verdict = "C_PARTIAL_COMPLETE_ASSOCIATION_FAILS"
    elif left is not None and right is not None and clear_separation:
        verdict = "A_MULTIVIEW_CONSISTENCY_SEPARATES_TRUE_AND_FALSE_BRANCH_EVIDENCE"
    elif left is not None and right is not None and false_ambiguous:
        verdict = "B_MULTIVIEW_RECOVERS_TRUE_BRANCHES_BUT_FALSE_GAPS_REMAIN_AMBIGUOUS"
    else:
        verdict = "D_MULTIVIEW_FUSION_DOES_NOT_IMPROVE_BRANCH_DISAMBIGUATION"

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "source_equivalence_pass": source_equivalent,
        "deterministic_replay": deterministic,
        "common_frame_transform_max_error": transform_max_error,
        "common_frame_definition": "x=Anchor-forward,y=Anchor-left,origin=A0",
        "association_aggregation": "coordinate_median_and_axial_circular_mean",
        "distance_tolerance_W": NOMINAL_DISTANCE_TOLERANCE_W,
        "width_difference_tolerance_W": WIDTH_DIFFERENCE_TOLERANCE_W,
        "orientation_tolerance_deg": ORIENTATION_TOLERANCE_DEG,
        "termination_cluster_count": len(termination_clusters),
        "accepted_source_gap_count": len(gaps),
        "fused_candidate_count": len(fused),
        "left_partial_complete_associated": left_partial_complete,
        "right_partial_complete_associated": right_partial_complete,
        "left_supporting_view_count": 0 if left is None else left["supporting_view_count"],
        "left_complete_support_count": 0 if left is None else left["complete_support_count"],
        "left_partial_support_count": 0 if left is None else left["partial_support_count"],
        "right_supporting_view_count": 0 if right is None else right["supporting_view_count"],
        "right_complete_support_count": 0 if right is None else right["complete_support_count"],
        "right_partial_support_count": 0 if right is None else right["partial_support_count"],
        "incoming_supporting_view_count": 0 if incoming is None else incoming["supporting_view_count"],
        "incoming_complete_support_count": 0 if incoming is None else incoming["complete_support_count"],
        "false_candidate_count": len(false_rows),
        "false_supporting_view_counts": json.dumps(false_support),
        "false_gap_pair_associated": "" if false_pair is None else false_pair["associated"],
        "false_gap_center_separation_W": "" if false_pair is None else false_pair["center_distance_W"],
        "false_gap_orientation_difference_deg": "" if false_pair is None else false_pair["orientation_difference_deg"],
        "true_false_support_clear_separation": clear_separation,
        "multiview_support_alone_insufficient": false_ambiguous,
        "sensitivity_stable": sensitivity_stable,
        "M0_negative_passed": m0["passed"],
        "GT_used_for_association_or_support": False,
        "GT_used_posthoc_only": True,
        "movement_executed": False,
    }
    verdict_row = {
        **summary,
        "primary_verdict": verdict,
        "secondary_finding": "MULTIVIEW_SUPPORT_ALONE_INSUFFICIENT" if false_ambiguous else "",
        "production_fusion_implemented": False,
        "new_viewpoint_search_performed": False,
        "wall_topology_threshold_modified": False,
        "detector_threshold_modified": False,
    }

    _write_required(args.output / "source_viewpoints.csv", source_rows)
    _write_required(args.output / "view_evidence.csv", _view_evidence_rows(analyses, terminations, gaps))
    _write_required(args.output / "termination_associations.csv", termination_clusters)
    _write_required(args.output / "gap_associations.csv", pair_rows)
    _write_required(args.output / "fused_candidates.csv", fused_rows)
    _write_required(args.output / "candidate_support_summary.csv", support_rows)
    _write_required(args.output / "gt_match_eval.csv", eval_rows)
    _write_required(args.output / "single_vs_multiview_comparison.csv", comparison)
    _write_required(args.output / "association_sensitivity.csv", sensitivity)
    _write_required(args.output / "m0_negative_sanity.csv", [m0])
    _write_required(args.output / "verdict.csv", [verdict_row])

    _plot_common_evidence(args.output / "multiview_common_frame_evidence.png", all_segments, terminations, gaps, width)
    _plot_fused(args.output / "fused_candidate_clusters.png", fused, eval_rows, width)
    _plot_support(args.output / "true_vs_false_multiview_support.png", fused_rows, eval_rows)
    if left is not None:
        candidate = next(row for row in fused if row["fusion_candidate_id"] == left["fusion_candidate_id"])
        _plot_branch(args.output / "left_branch_multiview_evidence.png", candidate, termination_clusters, width, "LEFT partial-to-complete evidence (GT label post-hoc)")
    if right is not None:
        candidate = next(row for row in fused if row["fusion_candidate_id"] == right["fusion_candidate_id"])
        _plot_branch(args.output / "right_branch_multiview_evidence.png", candidate, termination_clusters, width, "RIGHT partial-to-complete evidence (GT label post-hoc)")

    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "primary_verdict": verdict,
                "summary": summary,
                "source_viewpoints": source_rows,
                "candidates": support_rows,
                "sensitivity": sensitivity,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
