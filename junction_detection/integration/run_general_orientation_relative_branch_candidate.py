"""EXP-038 targeted shadow evaluation of general branch gap representation.

M1 reuses the persisted EXP-037 accepted-gap audit.  M0 uses one static
deterministic straight-corridor scan.  One known EXP-035 non-forward viewpoint
is reconstructed with its original deterministic ray caster.  No physics,
SPH, detector threshold, production candidate state, or map-specific runtime
rule is modified.
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

from junction_detection.integration.run_final_nonforward_viewpoint_side_topology_observability import (
    DEFAULT_SOURCE,
    SAFE_SOURCE,
    _analyze_viewpoint,
    _anchor_from_existing_source,
    _read,
    _rear_start_geometry,
)
from junction_detection.integration.run_wall_topology_branch_opening_diagnostic import _analyze
from junction_detection.pointcloud.general_branch_candidate import (
    GeneralBranchCandidate,
    OrientationRelativeGapDescriptor,
    build_general_branch_candidate,
    self_test as general_candidate_self_test,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import (
    BASELINE_CORRIDOR_WIDTH,
    GeometryBuilder,
    LidarSensor,
)

EXPERIMENT_ID = "EXP-038"
M1_AUDIT = ROOT / "junction_detection/integration/output/axial_forward_branch_candidate_representation_diagnostic"
EXP035_AUDIT = ROOT / "junction_detection/integration/output/final_nonforward_side_topology_observability/accepted_gap_audit.csv"
DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/general_orientation_relative_branch_candidate"
FALSE_VIEWPOINT_EVAL_ONLY = "F+0.700_L+0.100"


def _write(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    """Write a required result table and retain an empty header if supplied."""
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


def _csv(path: Path) -> list[dict[str, str]]:
    """Read one persisted CSV without mutating its source."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _m1_persisted_input(audit: Path) -> dict[str, Any]:
    """Rebuild the first accepted axial-gap frame from EXP-037 CSV rows."""
    stages = _csv(audit / "forward_branch_stage_audit.csv")
    forward = next(
        row for row in stages
        if row["branch_eval_only"] == "FORWARD" and int(row["accepted_gap_count"]) > 0
    )
    frame = int(forward["frame"])
    endpoint_rows = [row for row in _csv(audit / "forward_branch_endpoint_audit.csv") if int(row["frame"]) == frame]
    gap_rows = [row for row in _csv(audit / "forward_branch_gap_audit.csv") if int(row["frame"]) == frame and _bool(row["accepted"])]
    segments: dict[int, dict[str, Any]] = {}
    endpoints: dict[int, dict[str, Any]] = {}
    for row in endpoint_rows:
        segment_ids = [int(value) for value in json.loads(row["segment_id"])]
        tangent_angles = [float(value) for value in json.loads(row["wall_tangent_local"])]
        for segment_id, angle in zip(segment_ids, tangent_angles):
            segments.setdefault(segment_id, {"segment_id": segment_id, "orientation_deg": angle})
        endpoints[int(row["endpoint_id"])] = {
            "endpoint_id": int(row["endpoint_id"]),
            "point": np.array([float(row["x_local"]), float(row["y_local"])]),
            "endpoint_type": row["endpoint_type"],
            "segment_ids": segment_ids,
        }
    gaps = []
    evaluations = {}
    for row in gap_rows:
        gap_id = int(row["gap_id"])
        center = np.asarray(json.loads(row["gap_center_local"]), dtype=float)
        gaps.append({
            "gap_id": gap_id,
            "endpoint_a": int(row["endpoint_a"]),
            "endpoint_b": int(row["endpoint_b"]),
            "gap_center": center,
            "gap_width": float(row["gap_width"]),
            "gap_width_ratio_W": float(row["gap_width_over_W_hat"]),
            "continuation_depth": float(row["free_continuation"]),
            "estimated_direction_local": math.degrees(math.atan2(float(center[1]), float(center[0]))),
            "candidate_valid": True,
        })
        evaluations[gap_id] = {
            "matched_branch_eval_only": row["matched_branch_eval_only"],
            "is_false_gap_eval_only": row["matched_branch_eval_only"] == "NONE",
            "old_candidate_created": _bool(row["runtime_candidate_created"]),
        }
    return {
        "case": "M1_CROSS_BASELINE",
        "frame": frame,
        "timestamp": float(forward["timestamp"]),
        "corridor_axis_local": np.array([1.0, 0.0]),
        "endpoints": endpoints,
        "segments": segments,
        "gaps": gaps,
        "evaluations": evaluations,
        "points": None,
    }


def _m0_static_input() -> dict[str, Any]:
    """Create one deterministic no-physics straight-corridor negative scan."""
    sensor = LidarSensor()
    geometry = GeometryBuilder.build("M0_STRAIGHT")
    scan = sensor.scan(geometry, np.zeros(2), 90.0)
    margin = np.finfo(float).eps * scan.max_range * 64.0
    snapshot = {
        "context": "M0_STATIC",
        "angles": scan.angles_deg,
        "ranges": scan.ranges,
        "hit": scan.ranges < scan.max_range - margin,
        "max_range": scan.max_range,
    }
    result = _analyze("M0_STATIC", snapshot, BASELINE_CORRIDOR_WIDTH)
    return {
        "case": "M0_STRAIGHT",
        "frame": 0,
        "timestamp": 0.0,
        "corridor_axis_local": np.array([1.0, 0.0]),
        "endpoints": {row["endpoint_id"]: row for row in result["endpoints"]},
        "segments": {row["segment_id"]: row for row in result["segments"]},
        "gaps": [row for row in result["gaps"] if row["candidate_valid"]],
        "evaluations": {row["gap_id"]: {"matched_branch_eval_only": "NONE", "is_false_gap_eval_only": True, "old_candidate_created": False} for row in result["gaps"] if row["candidate_valid"]},
        "points": result["points"],
        "snapshot": snapshot,
    }


def _exp035_representative_input(source: Path, safe_source: Path) -> dict[str, Any]:
    """Reconstruct only the persisted representative false-diagonal viewpoint."""
    source_row = next(row for row in _read(source) if row["candidate_id"] == FALSE_VIEWPOINT_EVAL_ONLY)
    anchor = _anchor_from_existing_source(safe_source)
    analysis = _analyze_viewpoint(source_row, anchor, LidarSensor(), _rear_start_geometry())
    matches = {int(row["gap_id"]): row for row in analysis["matches"]}
    endpoints = {row["endpoint_id"]: row for row in analysis["result"]["endpoints"]}
    gaps = [row for row in analysis["result"]["gaps"] if row["candidate_valid"]]
    evaluations = {}
    for gap in gaps:
        match = matches[int(gap["gap_id"])]
        first, second = endpoints[gap["endpoint_a"]], endpoints[gap["endpoint_b"]]
        same_side = math.copysign(1.0, float(first["point"][1])) == math.copysign(1.0, float(second["point"][1]))
        evaluations[int(gap["gap_id"])] = {
            "matched_branch_eval_only": match["matched_branch_eval"],
            "is_false_gap_eval_only": bool(match["false_positive_eval"]),
            "old_candidate_created": same_side,
        }
    return {
        "case": f"EXP035_{FALSE_VIEWPOINT_EVAL_ONLY}",
        "frame": -1,
        "timestamp": math.nan,
        "corridor_axis_local": np.array([1.0, 0.0]),
        "endpoints": endpoints,
        "segments": {row["segment_id"]: row for row in analysis["result"]["segments"]},
        "gaps": gaps,
        "evaluations": evaluations,
        "points": analysis["result"]["points"],
        "snapshot": analysis["snapshot"],
    }


def _evaluate_case(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, tuple[OrientationRelativeGapDescriptor, GeneralBranchCandidate | None]]]:
    """Apply one label-free constructor, then attach evaluation columns."""
    rows, features, decisions = [], [], {}
    for gap in data["gaps"]:
        gap_id = int(gap["gap_id"])
        descriptor, candidate = build_general_branch_candidate(
            candidate_id=f"{data['case']}_G{gap_id}",
            timestamp=data["timestamp"],
            topology_type="COMPLETE",
            gap=gap,
            endpoints=data["endpoints"],
            segments=data["segments"],
            corridor_axis_local=data["corridor_axis_local"],
        )
        decisions[gap_id] = (descriptor, candidate)
        evaluation = data["evaluations"][gap_id]
        rows.append({
            "case": data["case"],
            "timestamp": data["timestamp"],
            "frame": data["frame"],
            "source_gap_id": gap_id,
            "old_candidate_created": evaluation["old_candidate_created"],
            "new_candidate_created": candidate is not None,
            "topology_type": "COMPLETE",
            "gap_width": descriptor.gap_width,
            "gap_width_over_W_hat": descriptor.gap_width_over_W_hat,
            "free_continuation": descriptor.free_continuation,
            "gap_tangent_deg_local": descriptor.gap_tangent_deg_local,
            "opening_normal_deg_local": descriptor.opening_normal_deg_local,
            "free_space_direction_deg_local": descriptor.free_space_direction_deg_local,
            "relative_opening_angle_deg": descriptor.relative_opening_angle_deg,
            "endpoint_a_type": descriptor.endpoint_a_type,
            "endpoint_b_type": descriptor.endpoint_b_type,
            "geometry_support": descriptor.geometry_support,
            "candidate_reliability": math.nan if candidate is None else candidate.candidate_reliability,
            "normal_free_alignment": descriptor.normal_free_alignment,
            "gap_boundary_wall_alignment_error_deg": descriptor.gap_boundary_wall_alignment_error_deg,
            "rejection_reason": descriptor.rejection_reason,
            "matched_branch_eval_only": evaluation["matched_branch_eval_only"],
            "is_false_gap_eval_only": evaluation["is_false_gap_eval_only"],
        })
        feature_values = {
            "gap_width_over_W_hat": descriptor.gap_width_over_W_hat,
            "free_continuation": descriptor.free_continuation,
            "gap_tangent_deg_local": descriptor.gap_tangent_deg_local,
            "opening_normal_deg_local": descriptor.opening_normal_deg_local,
            "free_space_direction_deg_local": descriptor.free_space_direction_deg_local,
            "normal_free_alignment": descriptor.normal_free_alignment,
            "relative_opening_angle_deg": descriptor.relative_opening_angle_deg,
            "endpoint_a_wall_tangent_deg": json.dumps(descriptor.endpoint_a_wall_tangent_deg),
            "endpoint_b_wall_tangent_deg": json.dumps(descriptor.endpoint_b_wall_tangent_deg),
            "gap_boundary_wall_alignment_error_deg": descriptor.gap_boundary_wall_alignment_error_deg,
            "endpoint_type_combination": f"{descriptor.endpoint_a_type}+{descriptor.endpoint_b_type}",
            "geometry_support": descriptor.geometry_support,
        }
        for feature, value in feature_values.items():
            features.append({
                "case": data["case"],
                "gap_id": gap_id,
                "feature": feature,
                "value": value,
                "matched_branch_eval_only": evaluation["matched_branch_eval_only"],
                "is_false_gap_eval_only": evaluation["is_false_gap_eval_only"],
            })
    return rows, features, decisions


def _summary(case: str, rows: list[dict[str, Any]], true_branch_count: int) -> dict[str, Any]:
    """Aggregate candidate recovery after post-hoc association."""
    new = [row for row in rows if row["new_candidate_created"]]
    true = [row for row in new if row["matched_branch_eval_only"] not in {"NONE", "FALSE", "INCOMING"}]
    false = [row for row in new if row["is_false_gap_eval_only"]]
    return {
        "case": case,
        "true_branch_count_eval_only": true_branch_count,
        "old_candidate_count": sum(bool(row["old_candidate_created"]) for row in rows),
        "new_candidate_count": len(new),
        "true_candidate_count_eval_only": len(true),
        "false_candidate_count_eval_only": len(false),
        "axial_recovered_eval_only": any(row["new_candidate_created"] and row["matched_branch_eval_only"] == "FORWARD" for row in rows),
    }


def _plot_gap_frame(path: Path, data: dict[str, Any], decisions: dict[int, tuple[OrientationRelativeGapDescriptor, GeneralBranchCandidate | None]], title: str, candidates_only: bool = False) -> None:
    """Plot endpoints and orientation-relative frames for accepted gaps."""
    fig, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    points = data.get("points")
    snapshot = data.get("snapshot")
    if points is not None and snapshot is not None:
        hit = np.asarray(snapshot["hit"], dtype=bool)
        axis.scatter(points[hit, 0], points[hit, 1], s=7, color="0.75", label="LiDAR hits")
    for segment in data["segments"].values():
        if "start" in segment and "end" in segment:
            axis.plot([segment["start"][0], segment["end"][0]], [segment["start"][1], segment["end"][1]], color="0.45", linewidth=1.5)
    for endpoint in data["endpoints"].values():
        axis.scatter(*endpoint["point"], color="tab:orange" if endpoint["endpoint_type"] == "CORNER" else "tab:red", s=45)
    for gap in data["gaps"]:
        descriptor, candidate = decisions[int(gap["gap_id"])]
        if candidates_only and candidate is None:
            continue
        a, b, center = descriptor.endpoint_a_local, descriptor.endpoint_b_local, descriptor.gap_center_local
        color = "tab:green" if candidate is not None else "tab:red"
        axis.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=3 if candidate is not None else 1.5, alpha=0.9)
        scale = 18.0
        axis.arrow(center[0], center[1], *(descriptor.gap_tangent_local * scale), color="tab:blue", width=0.35, length_includes_head=True)
        axis.arrow(center[0], center[1], *(descriptor.opening_normal_local * scale), color="tab:green", width=0.35, length_includes_head=True)
        axis.arrow(center[0], center[1], *(descriptor.free_space_direction_local * scale), color="magenta", width=0.25, length_includes_head=True)
        evaluation = data["evaluations"][int(gap["gap_id"])]
        axis.annotate(
            f"G{gap['gap_id']} {evaluation['matched_branch_eval_only']} (EVAL ONLY)\n{descriptor.rejection_reason}",
            xy=center,
            xytext=(5, 7 + 11 * (int(gap["gap_id"]) % 3)),
            textcoords="offset points",
            fontsize=6,
        )
    axis.scatter(0, 0, marker="*", color="black", s=120, label="LiDAR")
    axis.arrow(0, 0, 28, 0, color="black", width=0.5, length_includes_head=True, label="corridor axis")
    axis.set(title=title, xlabel="local x", ylabel="local y", aspect="equal")
    axis.autoscale(); axis.margins(0.15); axis.grid(alpha=0.2)
    axis.plot([], [], color="tab:blue", label="gap tangent")
    axis.plot([], [], color="tab:green", label="opening normal")
    axis.plot([], [], color="magenta", label="free-space direction")
    axis.legend(fontsize=7)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_local_frame(path: Path) -> None:
    """Illustrate one constructor on horizontal, vertical, and angled gaps."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, angle_deg in zip(axes, (0.0, 90.0, 37.0)):
        angle = math.radians(angle_deg)
        tangent = np.array([math.cos(angle), math.sin(angle)])
        center = np.array([0.0, 0.0]); first, second = center - 1.5 * tangent, center + 1.5 * tangent
        endpoints = {0: {"point": first, "endpoint_type": "CORNER", "segment_ids": [0]}, 1: {"point": second, "endpoint_type": "WALL_TERMINATION", "segment_ids": [1]}}
        segments = {0: {"orientation_deg": angle_deg}, 1: {"orientation_deg": angle_deg}}
        gap = {"endpoint_a": 0, "endpoint_b": 1, "gap_center": center + np.array([3.0, 1.0]), "gap_width": 3.0, "gap_width_ratio_W": 1.0, "continuation_depth": 2.0, "candidate_valid": True}
        descriptor, candidate = build_general_branch_candidate("S", 0.0, "COMPLETE", gap, endpoints, segments, np.array([1.0, 0.0]))
        axis.plot([first[0], second[0]], [first[1], second[1]], color="black", linewidth=3)
        axis.arrow(0, 0, *descriptor.gap_tangent_local, color="tab:blue", width=0.03, length_includes_head=True)
        axis.arrow(0, 0, *descriptor.opening_normal_local, color="tab:green", width=0.03, length_includes_head=True)
        axis.arrow(0, 0, *descriptor.free_space_direction_local, color="magenta", width=0.02, length_includes_head=True)
        axis.set(title=f"gap tangent={angle_deg:.0f}°\nGENERAL candidate={candidate is not None}", aspect="equal", xlim=(-2, 2), ylim=(-2, 2))
        axis.grid(alpha=0.2)
    axes[0].plot([], [], color="tab:blue", label="gap tangent")
    axes[0].plot([], [], color="tab:green", label="opening normal")
    axes[0].plot([], [], color="magenta", label="free direction")
    axes[0].legend(fontsize=7)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _canonical(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize NaNs for exact deterministic comparisons."""
    return [{key: "NaN" if isinstance(value, float) and math.isnan(value) else value for key, value in row.items()} for row in rows]


def run(output: Path, m1_audit: Path, source: Path, safe_source: Path) -> dict[str, Any]:
    """Run static targeted cases and save the EXP-038 shadow comparison."""
    output.mkdir(parents=True, exist_ok=True)
    m0 = _m0_static_input()
    m1 = _m1_persisted_input(m1_audit)
    false_case = _exp035_representative_input(source, safe_source)
    all_rows, all_features, decisions = [], [], {}
    for data in (m0, m1, false_case):
        rows, features, case_decisions = _evaluate_case(data)
        all_rows.extend(rows); all_features.extend(features); decisions[data["case"]] = case_decisions
    # Re-read/reconstruct the exact same inputs; no physics or map sweep occurs.
    repeated_inputs = (_m0_static_input(), _m1_persisted_input(m1_audit), _exp035_representative_input(source, safe_source))
    repeated_rows = []
    for data in repeated_inputs:
        rows, _, _ = _evaluate_case(data); repeated_rows.extend(rows)
    deterministic = _canonical(all_rows) == _canonical(repeated_rows)
    by_case = {case: [row for row in all_rows if row["case"] == case] for case in (m0["case"], m1["case"], false_case["case"])}
    summaries = [
        _summary(m0["case"], by_case[m0["case"]], 0),
        _summary(m1["case"], by_case[m1["case"]], 3),
        _summary(false_case["case"], by_case[false_case["case"]], 1),
    ]
    for row in summaries: row["deterministic_replay"] = deterministic
    m1_rows, false_rows = by_case[m1["case"]], by_case[false_case["case"]]
    side_preserved = all(any(row["new_candidate_created"] and row["matched_branch_eval_only"] == label for row in m1_rows) for label in ("LEFT", "RIGHT"))
    axial = any(row["new_candidate_created"] and row["matched_branch_eval_only"] == "FORWARD" for row in m1_rows)
    false_reintroduced = any(row["new_candidate_created"] and row["is_false_gap_eval_only"] for row in false_rows)
    m0_clean = not by_case[m0["case"]]
    if m0_clean and side_preserved and axial and not false_reintroduced:
        verdict = "A_GENERAL_ORIENTATION_RELATIVE_CANDIDATE_RECOVERS_AXIAL_WITHOUT_REGRESSION"
    elif axial and false_reintroduced:
        verdict = "B_AXIAL_RECOVERED_BUT_FALSE_DIAGONAL_REINTRODUCED"
    elif not side_preserved:
        verdict = "C_GENERAL_REPRESENTATION_BREAKS_EXISTING_SIDE_CANDIDATES"
    else:
        verdict = "D_NO_GENERAL_LOCAL_GEOMETRIC_DISAMBIGUATOR_FOUND"
    _write(output / "general_candidate_comparison.csv", all_rows, ["case", "timestamp", "frame", "source_gap_id", "old_candidate_created", "new_candidate_created", "topology_type", "gap_width", "gap_width_over_W_hat", "free_continuation", "gap_tangent_deg_local", "opening_normal_deg_local", "free_space_direction_deg_local", "relative_opening_angle_deg", "endpoint_a_type", "endpoint_b_type", "geometry_support", "candidate_reliability", "normal_free_alignment", "gap_boundary_wall_alignment_error_deg", "rejection_reason", "matched_branch_eval_only", "is_false_gap_eval_only"])
    _write(output / "orientation_relative_feature_audit.csv", all_features)
    _write(output / "general_candidate_summary.csv", summaries)
    _plot_gap_frame(output / "m1_side_axial_accepted_gaps.png", m1, decisions[m1["case"]], "M1 accepted gaps: side, axial, and rejected arbitrary pairs")
    _plot_gap_frame(output / "m1_new_general_candidates.png", m1, decisions[m1["case"]], "M1 new GENERAL_BRANCH_CANDIDATE hypotheses", candidates_only=True)
    false_plot = {
        **false_case,
        "gaps": [
            gap for gap in false_case["gaps"]
            if false_case["evaluations"][int(gap["gap_id"])]["matched_branch_eval_only"] != "INCOMING"
        ],
    }
    _plot_gap_frame(output / "exp035_false_diagonal_comparison.png", false_plot, decisions[false_case["case"]], "EXP-035 true side vs known false diagonal (GT EVAL ONLY)")
    _plot_local_frame(output / "local_geometric_frame_illustration.png")
    return {"verdict": verdict, "deterministic": deterministic, "summaries": summaries, "output": str(output.resolve())}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--m1-audit", type=Path, default=M1_AUDIT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--safe-source", type=Path, default=SAFE_SOURCE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    general_candidate_self_test()
    result = run(args.output, args.m1_audit, args.source, args.safe_source)
    print(json.dumps({"experiment_id": EXPERIMENT_ID, **result}, indent=2, default=str))


if __name__ == "__main__":
    main()
