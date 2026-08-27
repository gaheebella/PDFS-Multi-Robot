"""EXP-034: replay saved forward poses through frozen EXP-033 wall topology.

The prior safe/historical CSVs did not persist raw 360-beam ranges.  They did
persist exact pose, yaw, displacement, hit count, and opening count, so this
diagnostic reconstructs scans with the unchanged deterministic sensor.  No
movement simulation is rerun.  EXP-033 topology helpers are imported without
changing their rules; GT mouths are used only after topology generation.
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

from junction_detection.integration.run_side_branch_detector_evidence_pipeline import _write_csv
from junction_detection.integration.run_lidar_local_corridor_estimation import (
    REAR_START_SHIFT,
)
from junction_detection.integration.run_wall_topology_branch_opening_diagnostic import (
    _analyze,
    _branch_topology_eval,
    _candidate_rows,
    _gap_rows,
    _gt_mouths_eval,
    _match_candidates_eval,
    _plot_result,
    _self_test,
)
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import detect_openings
from pygame_simulator.pre_exploration_general_pipeline_simulator import (
    GeometryCase,
    GeometryBuilder,
    LidarSensor,
    _rect,
    _union_boundary,
)

EXPERIMENT_ID = "EXP-034"
DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/forward_viewpoint_wall_topology_transition"
SAFE_SOURCE = ROOT / "junction_detection/integration/output/safe_active_viewpoint_visibility/safe_viewpoint_scans.csv"
HISTORICAL_SOURCE = ROOT / "junction_detection/integration/output/active_viewpoint_acquisition/active_viewpoint_scans.csv"
EXP033_SUMMARY = ROOT / "junction_detection/integration/output/wall_topology_branch_opening/wall_topology_summary.csv"
SIDE_LABELS = ("LEFT", "RIGHT")


def _rear_start_geometry() -> GeometryCase:
    """Rebuild the exact persisted evaluation geometry without moving robots."""
    original = GeometryBuilder.build("M1_CROSS_BASELINE")
    entrance = float(original.entrance_y)
    length = original.incoming_length + REAR_START_SHIFT
    incoming = _rect(
        np.array([0.0, entrance - 0.5 * length]),
        np.array([0.0, 1.0]),
        original.incoming_width,
        length,
    )
    rects = (incoming,) + original.free_rects[1:]
    return GeometryCase(
        original.case_id,
        original.incoming_width,
        length,
        original.junction_size,
        original.branches,
        rects,
        _union_boundary(rects),
        original.entrance_y,
    )


def _read(path: Path) -> list[dict[str, str]]:
    """Read one existing viewpoint summary CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _local_width(rows: list[dict[str, str]]) -> float:
    """Recover the persisted local width scale from displacement/ratio pairs."""
    estimates = []
    for row in rows:
        advance_key = "cumulative_local_advance"
        ratio_key = "cumulative_width_ratio" if "cumulative_width_ratio" in row else "cumulative_advance_width_ratio"
        advance, ratio = float(row[advance_key]), float(row[ratio_key])
        if ratio > 0.0:
            estimates.append(advance / ratio)
        elif row.get("estimated_corridor_width"):
            estimates.append(float(row["estimated_corridor_width"]))
    if not estimates or max(estimates) - min(estimates) > 1.0e-9:
        raise RuntimeError("persisted viewpoint rows do not define one stable local width")
    return float(np.median(estimates))


def _source_specs(
    path: Path,
    source_experiment: str,
    safety_class: str,
    prefix: str,
) -> list[dict[str, Any]]:
    """Normalize saved pose/displacement metadata without reconstructing scans."""
    rows = _read(path)
    width = _local_width(rows)
    result = []
    for index, row in enumerate(rows):
        ratio_key = "cumulative_width_ratio" if "cumulative_width_ratio" in row else "cumulative_advance_width_ratio"
        hit_key = "valid_hit_count" if "valid_hit_count" in row else "valid_lidar_point_count"
        opening_match_key = "matched_outgoing_eval_only" if "matched_outgoing_eval_only" in row else "matched_outgoing_count_eval_only"
        result.append(
            {
                "viewpoint_id": "V0" if prefix == "V" and index == 0 else f"{prefix}{index}",
                "source_experiment": source_experiment,
                "source_file": str(path.resolve()),
                "scan_source_type": "DETERMINISTIC_REPLAY",
                "raw_scan_persisted": False,
                "movement_safety_class": safety_class,
                "actual_forward_displacement": float(row["cumulative_local_advance"]),
                "actual_forward_displacement_over_W": float(row[ratio_key]),
                "estimated_corridor_width": width,
                "position": np.array([float(row["anchor_x_eval_only"]), float(row["anchor_y_eval_only"])]),
                "yaw_deg": float(row["anchor_yaw_eval_only"]),
                "scan_stationary": math.isclose(float(row["leader_speed_at_scan"]), 0.0, abs_tol=1.0e-12),
                "saved_hit_count": int(row[hit_key]),
                "saved_opening_count": int(row["opening_count"]),
                "saved_angular_outgoing_match": int(row[opening_match_key]),
            }
        )
    return result


def _reconstruct(spec: dict[str, Any], sensor: LidarSensor, geometry: Any) -> dict[str, Any]:
    """Reconstruct an exact saved-pose scan with the unchanged ray caster."""
    scan = sensor.scan(geometry, spec["position"], spec["yaw_deg"])
    margin = np.finfo(float).eps * max(1.0, scan.max_range) * 64.0
    snapshot = {
        "context": spec["viewpoint_id"],
        "angles": scan.angles_deg.copy(),
        "ranges": scan.ranges.copy(),
        "hit": scan.ranges < scan.max_range - margin,
        "max_range": scan.max_range,
        "position_eval": spec["position"].copy(),
        "yaw_eval": spec["yaw_deg"],
    }
    openings = detect_openings(snapshot["angles"], snapshot["ranges"])
    hit_count = int(np.count_nonzero(snapshot["hit"]))
    return {
        "snapshot": snapshot,
        "openings": openings,
        "reconstructed_hit_count": hit_count,
        "reconstructed_opening_count": len(openings),
        "scan_valid": hit_count == spec["saved_hit_count"] and len(openings) == spec["saved_opening_count"],
    }


def _branch_row(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Select one EXP-033 outgoing post-hoc topology row."""
    return next(row for row in rows if row["branch_eval"] == label)


def _analyze_spec(spec: dict[str, Any], sensor: LidarSensor, geometry: Any) -> dict[str, Any]:
    """Apply frozen scan-only topology, then add evaluation-only mouth labels."""
    replay = _reconstruct(spec, sensor, geometry)
    result = _analyze(spec["viewpoint_id"], replay["snapshot"], spec["estimated_corridor_width"])
    runner = SimpleNamespace(geometry=geometry)
    mouths = _gt_mouths_eval(runner, replay["snapshot"])
    matches = _match_candidates_eval(
        spec["viewpoint_id"], result["gaps"], result["endpoints"], mouths, spec["estimated_corridor_width"]
    )
    branches = _branch_topology_eval(
        runner, replay["snapshot"], result["endpoints"], matches, spec["estimated_corridor_width"]
    )
    return {"spec": spec, "replay": replay, "result": result, "mouths": mouths, "matches": matches, "branches": branches}


def _side_values(analysis: dict[str, Any], label: str) -> dict[str, Any]:
    """Extract near/far visibility and accepted GT-matched gap for one side."""
    branch = _branch_row(analysis["branches"], label)
    near_error = min(branch["nearest_endpoint_error_a_eval"], branch["nearest_endpoint_error_b_eval"])
    far_error = max(branch["nearest_endpoint_error_a_eval"], branch["nearest_endpoint_error_b_eval"])
    width = analysis["spec"]["estimated_corridor_width"]
    match = next((row for row in analysis["matches"] if row["matched_branch_eval"] == label), None)
    candidate = None
    if match is not None:
        candidate = next(row for row in analysis["result"]["gaps"] if row["gap_id"] == match["gap_id"])
    return {
        "topology": branch["topology_class_eval"],
        "near_visible": near_error <= 0.12 * width,
        "far_visible": far_error <= 0.12 * width,
        "near_error": near_error,
        "far_error": far_error,
        "gap_valid": candidate is not None,
        "gt_match": match is not None,
        "candidate": candidate,
        "match": match,
    }


def _summary_row(analysis: dict[str, Any]) -> dict[str, Any]:
    """Build the requested one-row-per-viewpoint transition table."""
    spec, result = analysis["spec"], analysis["result"]
    left, right = _side_values(analysis, "LEFT"), _side_values(analysis, "RIGHT")
    forward = _branch_row(analysis["branches"], "FORWARD")
    incoming = any(row["matched_branch_eval"] == "INCOMING" for row in analysis["matches"])
    false_count = sum(bool(row["false_positive_eval"]) for row in analysis["matches"])
    return {
        "viewpoint_id": spec["viewpoint_id"],
        "movement_safety_class": spec["movement_safety_class"],
        "forward_displacement": spec["actual_forward_displacement"],
        "forward_displacement_over_W": spec["actual_forward_displacement_over_W"],
        "left_topology": left["topology"],
        "left_near_boundary_visible_eval": left["near_visible"],
        "left_far_boundary_visible_eval": left["far_visible"],
        "left_near_endpoint_error_eval": left["near_error"],
        "left_far_endpoint_error_eval": left["far_error"],
        "left_gap_valid": left["gap_valid"],
        "left_gt_match_eval": left["gt_match"],
        "right_topology": right["topology"],
        "right_near_boundary_visible_eval": right["near_visible"],
        "right_far_boundary_visible_eval": right["far_visible"],
        "right_near_endpoint_error_eval": right["near_error"],
        "right_far_endpoint_error_eval": right["far_error"],
        "right_gap_valid": right["gap_valid"],
        "right_gt_match_eval": right["gt_match"],
        "forward_topology": forward["topology_class_eval"],
        "incoming_topology": "COMPLETE_GAP_TOPOLOGY" if incoming else "NO_GAP_TOPOLOGY",
        "wall_segment_count": len(result["segments"]),
        "valid_termination_count": sum(endpoint["valid"] for endpoint in result["endpoints"]),
        "scan_limit_endpoint_count": sum(endpoint["endpoint_type"] == "SCAN_LIMIT" for endpoint in result["endpoints"]),
        "gap_candidate_count": len(result["gaps"]),
        "accepted_gap_count": sum(gap["candidate_valid"] for gap in result["gaps"]),
        "false_gap_count": false_count,
        "angular_detector_outgoing_match_eval": spec["saved_angular_outgoing_match"],
        "source_scan_valid": analysis["replay"]["scan_valid"],
    }


def _source_audit(analysis: dict[str, Any]) -> dict[str, Any]:
    """Record why deterministic pose replay is considered equivalent."""
    spec, replay = analysis["spec"], analysis["replay"]
    return {
        "viewpoint_id": spec["viewpoint_id"],
        "source_experiment": spec["source_experiment"],
        "source_file": spec["source_file"],
        "scan_source_type": spec["scan_source_type"],
        "raw_scan_persisted": spec["raw_scan_persisted"],
        "movement_safety_class": spec["movement_safety_class"],
        "actual_forward_displacement": spec["actual_forward_displacement"],
        "actual_forward_displacement_over_W": spec["actual_forward_displacement_over_W"],
        "scan_stationary": spec["scan_stationary"],
        "saved_hit_count": spec["saved_hit_count"],
        "reconstructed_hit_count": replay["reconstructed_hit_count"],
        "saved_opening_count": spec["saved_opening_count"],
        "reconstructed_opening_count": replay["reconstructed_opening_count"],
        "scan_valid": replay["scan_valid"],
    }


def _boundary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand near/far errors into one branch row per viewpoint."""
    rows = []
    for summary in summaries:
        for label in SIDE_LABELS:
            key = label.lower()
            rows.append({
                "viewpoint_id": summary["viewpoint_id"],
                "movement_safety_class": summary["movement_safety_class"],
                "forward_displacement_over_W": summary["forward_displacement_over_W"],
                "branch_eval": label,
                "topology_state": summary[f"{key}_topology"],
                "near_boundary_visible_eval": summary[f"{key}_near_boundary_visible_eval"],
                "far_boundary_visible_eval": summary[f"{key}_far_boundary_visible_eval"],
                "near_endpoint_error_eval": summary[f"{key}_near_endpoint_error_eval"],
                "far_endpoint_error_eval": summary[f"{key}_far_endpoint_error_eval"],
            })
    return rows


def _gap_rows_by_viewpoint(analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialize frozen gaps and attach post-hoc match diagnostics."""
    rows = []
    for analysis in analyses:
        spec = analysis["spec"]
        matches = {row["gap_id"]: row for row in analysis["matches"]}
        for row in _gap_rows(analysis["result"]):
            match = matches.get(row["gap_id"])
            rows.append({
                "viewpoint_id": spec["viewpoint_id"],
                "movement_safety_class": spec["movement_safety_class"],
                "forward_displacement_over_W": spec["actual_forward_displacement_over_W"],
                **{key: value for key, value in row.items() if key != "case"},
                "matched_branch_eval": "" if match is None else match["matched_branch_eval"],
                "center_error_eval": "" if match is None else match["center_error_eval"],
                "endpoint_error_eval": "" if match is None else match["endpoint_error_eval"],
                "width_error_eval": "" if match is None else match["width_error_eval"],
                "mouth_overlap_eval": "" if match is None else match["mouth_overlap_eval"],
                "false_positive_eval": "" if match is None else match["false_positive_eval"],
            })
    return rows


def _transition_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report first observed COMPLETE and the preceding sampled bracket."""
    ordered = sorted(summaries, key=lambda row: row["forward_displacement_over_W"])
    rows = []
    for label in SIDE_LABELS:
        key = label.lower()
        first_index = next((index for index, row in enumerate(ordered) if row[f"{key}_topology"] == "COMPLETE_GAP_TOPOLOGY"), None)
        previous = None if first_index in (None, 0) else ordered[first_index - 1]
        first = None if first_index is None else ordered[first_index]
        rows.append({
            "branch_eval": label,
            "initial_state": ordered[0][f"{key}_topology"],
            "first_observed_complete_viewpoint": "" if first is None else first["viewpoint_id"],
            "first_observed_complete_displacement": "" if first is None else first["forward_displacement"],
            "first_observed_complete_ratio_W": "" if first is None else first["forward_displacement_over_W"],
            "previous_sample_viewpoint": "" if previous is None else previous["viewpoint_id"],
            "previous_sample_ratio_W": "" if previous is None else previous["forward_displacement_over_W"],
            "transition_bracket_low_W": "" if previous is None else previous["forward_displacement_over_W"],
            "transition_bracket_high_W": "" if first is None else first["forward_displacement_over_W"],
            "transition_found": first is not None,
            "movement_safety_class": "" if first is None else first["movement_safety_class"],
        })
    return rows


def _plot_transition(path: Path, summaries: list[dict[str, Any]]) -> None:
    """Plot categorical side topology only at sampled viewpoints."""
    levels = {"NO_GAP_TOPOLOGY": 0, "PARTIAL_GAP_TOPOLOGY": 1, "COMPLETE_GAP_TOPOLOGY": 2}
    fig, axis = plt.subplots(figsize=(9, 5))
    for label, color in (("LEFT", "tab:blue"), ("RIGHT", "tab:orange")):
        key = label.lower()
        display_offset = -0.035 if label == "LEFT" else 0.035
        for safety, marker in (("SAFE_EXISTING", "o"), ("HISTORICAL_GEOMETRY_ONLY", "s")):
            rows = sorted((row for row in summaries if row["movement_safety_class"] == safety), key=lambda row: row["forward_displacement_over_W"])
            if rows:
                axis.plot([row["forward_displacement_over_W"] for row in rows], [levels[row[f"{key}_topology"]] + display_offset for row in rows], marker=marker, color=color, linestyle="-" if safety == "SAFE_EXISTING" else "--", label=f"{label} {safety}")
    axis.set(yticks=(0, 1, 2), yticklabels=("NO", "PARTIAL", "COMPLETE"), xlabel="forward displacement / W_hat", title="Sampled wall-topology transition")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_errors(path: Path, summaries: list[dict[str, Any]]) -> None:
    """Plot near/far endpoint errors without interpolation between safety classes."""
    fig, axis = plt.subplots(figsize=(9, 5))
    styles = (("left", "near", "tab:blue", "o"), ("left", "far", "tab:blue", "s"), ("right", "near", "tab:orange", "o"), ("right", "far", "tab:orange", "s"))
    for branch, boundary, color, marker in styles:
        rows = sorted(summaries, key=lambda row: row["forward_displacement_over_W"])
        axis.plot([row["forward_displacement_over_W"] for row in rows], [row[f"{branch}_{boundary}_endpoint_error_eval"] for row in rows], marker=marker, color=color, linestyle="-" if boundary == "near" else "--", label=f"{branch} {boundary}")
    axis.set(xlabel="forward displacement / W_hat", ylabel="nearest topology endpoint error", title="Side-boundary observability at saved viewpoints")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--safe-source", type=Path, default=SAFE_SOURCE)
    parser.add_argument("--historical-source", type=Path, default=HISTORICAL_SOURCE)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    _self_test()

    geometry = _rear_start_geometry()
    sensor = LidarSensor()
    safe_specs = _source_specs(args.safe_source, "SAFE_ACTIVE_VIEWPOINT_VISIBILITY", "SAFE_EXISTING", "V")
    safe = [_analyze_spec(spec, sensor, geometry) for spec in safe_specs]
    safe_summaries = [_summary_row(item) for item in safe]
    safe_complete = any(row["left_topology"] == "COMPLETE_GAP_TOPOLOGY" or row["right_topology"] == "COMPLETE_GAP_TOPOLOGY" for row in safe_summaries)

    historical: list[dict[str, Any]] = []
    if not safe_complete and args.historical_source.exists():
        historical_specs = _source_specs(args.historical_source, "ACTIVE_VIEWPOINT_ACQUISITION", "HISTORICAL_GEOMETRY_ONLY", "H")
        historical_specs = [spec for spec in historical_specs if spec["actual_forward_displacement_over_W"] > 0.0]
        for index, spec in enumerate(historical_specs, start=1):
            spec["viewpoint_id"] = f"H{index}"
        historical = [_analyze_spec(spec, sensor, geometry) for spec in historical_specs]
    analyses = safe + historical
    summaries = safe_summaries + [_summary_row(item) for item in historical]
    if any(not item["replay"]["scan_valid"] for item in analyses):
        raise RuntimeError("saved-pose replay did not reproduce persisted scan summary")
    ratios = [row["forward_displacement_over_W"] for row in safe_summaries]
    if any(high <= low for low, high in zip(ratios, ratios[1:])):
        raise RuntimeError("safe viewpoint displacement order is not strictly increasing")

    # EXP-033 A0 frozen-rule equivalence gate.
    a0 = safe_summaries[0]
    a0_equivalent = (
        a0["forward_topology"] == "NO_GAP_TOPOLOGY"
        and a0["left_topology"] == "PARTIAL_GAP_TOPOLOGY"
        and a0["right_topology"] == "PARTIAL_GAP_TOPOLOGY"
        and a0["accepted_gap_count"] == 1
        and a0["false_gap_count"] == 0
        and safe[0]["matches"][0]["matched_branch_eval"] == "INCOMING"
    )
    if not a0_equivalent:
        raise RuntimeError("A0 does not reproduce EXP-033 frozen result")

    transitions = _transition_rows(summaries)
    left_transition = next(row for row in transitions if row["branch_eval"] == "LEFT")
    right_transition = next(row for row in transitions if row["branch_eval"] == "RIGHT")
    safe_left = any(row["left_topology"] == "COMPLETE_GAP_TOPOLOGY" for row in safe_summaries)
    safe_right = any(row["right_topology"] == "COMPLETE_GAP_TOPOLOGY" for row in safe_summaries)
    historical_left = any(row["left_topology"] == "COMPLETE_GAP_TOPOLOGY" for row in summaries if row["movement_safety_class"] == "HISTORICAL_GEOMETRY_ONLY")
    historical_right = any(row["right_topology"] == "COMPLETE_GAP_TOPOLOGY" for row in summaries if row["movement_safety_class"] == "HISTORICAL_GEOMETRY_ONLY")
    false_unstable = any(row["false_gap_count"] > a0["false_gap_count"] for row in summaries)
    if false_unstable:
        primary = "F_FORWARD_VIEWPOINT_TOPOLOGY_UNSTABLE"
    elif safe_left and safe_right:
        primary = "A_SAFE_FORWARD_VIEWPOINT_COMPLETES_SIDE_TOPOLOGY"
    elif safe_left != safe_right:
        primary = "B_ASYMMETRIC_FORWARD_TOPOLOGY_RECOVERY"
    elif historical_left or historical_right:
        primary = "D_COMPLETE_TOPOLOGY_ONLY_BEYOND_TESTED_SAFE_RANGE"
    elif historical:
        primary = "E_FORWARD_TRANSLATION_DOES_NOT_COMPLETE_SIDE_TOPOLOGY"
    else:
        primary = "C_NO_SAFE_FORWARD_PARTIAL_TO_COMPLETE_TRANSITION"
    secondary = ["EXISTING_RAW_SCAN_NOT_PERSISTED", "SAVED_POSE_DETERMINISTIC_REPLAY"]
    if not safe_left and not safe_right:
        secondary.append("NO_SAFE_COMPLETE_SIDE_MOUTH")

    # Reconstruct every pose twice and require exact scans/topology summaries.
    repeated = [_analyze_spec(item["spec"], sensor, geometry) for item in analyses]
    deterministic = all(
        np.array_equal(first["replay"]["snapshot"]["ranges"], second["replay"]["snapshot"]["ranges"])
        and _summary_row(first) == _summary_row(second)
        for first, second in zip(analyses, repeated)
    )

    source_rows = [_source_audit(item) for item in analyses]
    gap_rows = _gap_rows_by_viewpoint(analyses)
    verdict = {
        "experiment_id": EXPERIMENT_ID,
        "primary_verdict": primary,
        "secondary_findings": json.dumps(secondary),
        "existing_forward_scan_metadata_found": True,
        "raw_ranges_persisted": False,
        "scan_reconstruction_type": "DETERMINISTIC_SAVED_POSE_REPLAY",
        "safe_viewpoint_count": len(safe),
        "historical_viewpoint_count": len(historical),
        "A0_EXP033_equivalent": a0_equivalent,
        "wall_topology_rule_modified": False,
        "deterministic_replay": deterministic,
        "GT_used_for_wall_segment_endpoint_gap_acceptance": False,
        "GT_used_posthoc_only": True,
        "production_detector_modified": False,
    }
    angular_rows = [{"viewpoint_id": row["viewpoint_id"], "movement_safety_class": row["movement_safety_class"], "forward_displacement_over_W": row["forward_displacement_over_W"], "angular_detector_outgoing_match_eval": row["angular_detector_outgoing_match_eval"], "wall_left_topology": row["left_topology"], "wall_right_topology": row["right_topology"], "wall_forward_topology": row["forward_topology"]} for row in summaries]
    _write_csv(args.output / "source_viewpoint_audit.csv", source_rows)
    _write_csv(args.output / "viewpoint_topology_summary.csv", summaries)
    _write_csv(args.output / "branch_boundary_visibility.csv", _boundary_rows(summaries))
    _write_csv(args.output / "gap_candidates_by_viewpoint.csv", gap_rows)
    _write_csv(args.output / "transition_summary.csv", transitions)
    _write_csv(args.output / "angular_vs_wall_topology.csv", angular_rows)
    _write_csv(args.output / "verdict.csv", [verdict])
    _plot_transition(args.output / "forward_topology_transition.png", summaries)
    _plot_errors(args.output / "side_boundary_visibility_vs_displacement.png", summaries)
    runner = SimpleNamespace(geometry=geometry)
    _plot_result(args.output / "a0_wall_topology.png", safe[0]["result"], runner, safe[0]["mouths"], "EXP-034 A0 frozen wall topology")
    first_complete = next((item for item in analyses if any(row["topology_class_eval"] == "COMPLETE_GAP_TOPOLOGY" and row["branch_eval"] in SIDE_LABELS for row in item["branches"])), None)
    if first_complete is not None:
        _plot_result(args.output / "first_complete_wall_topology.png", first_complete["result"], runner, first_complete["mouths"], f"First observed complete side topology: {first_complete['spec']['viewpoint_id']}")
    print(json.dumps({
        "experiment_id": EXPERIMENT_ID,
        "primary_verdict": primary,
        "secondary_findings": secondary,
        "viewpoints": [(row["viewpoint_id"], row["movement_safety_class"], row["forward_displacement_over_W"], row["left_topology"], row["right_topology"], row["forward_topology"], row["incoming_topology"], row["false_gap_count"]) for row in summaries],
        "transitions": transitions,
        "A0_equivalent": a0_equivalent,
        "deterministic": deterministic,
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
