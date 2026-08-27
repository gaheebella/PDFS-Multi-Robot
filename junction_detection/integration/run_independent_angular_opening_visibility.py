"""EXP-032: validate independent angular opening visibility without tuning.

The frozen Point Cloud detector supplies the smoothed scan, adaptive threshold,
raw candidate mask, gap-filled mask, and final openings.  Candidate intervals
are extracted globally from the LiDAR scan before any GT branch ROI is built.
GT mouth/axis geometry is used only afterwards to label interval support and
compare evaluation semantics; it never creates or modifies scan topology.
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
from matplotlib.patches import Wedge
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.integration.run_2d_viewpoint_visibility_frontier import _probe
from junction_detection.integration.run_active_viewpoint_acquisition import (
    _gt_mouth_interval_eval_only,
)
from junction_detection.integration.run_local_asymmetric_viewpoint_geometry_diagnostic import (
    _acquire_m1_anchor,
)
from junction_detection.integration.run_provisional_anchor_pointcloud_observation import (
    _gt_directions_eval_only,
    _normalize,
)
from junction_detection.integration.run_side_branch_detector_evidence_pipeline import (
    SIDE_LABELS,
    _branch_id,
    _interval_mask,
    _mouth_sample_audit,
    _read_exp030_rows,
    _select_representatives,
    _write_csv,
)
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import (
    _circular_runs,
    _detect_openings_with_diagnostics,
    _run_width_deg,
    detect_openings,
)

EXPERIMENT_ID = "EXP-032"
DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/independent_angular_opening_visibility"
EXP030_OUTPUT = ROOT / "junction_detection/integration/output/2d_viewpoint_visibility_frontier"
BRANCH_LABELS = ("PLUS90", "MINUS90", "AXIAL_FORWARD")
MIN_OPENING_WIDTH_DEG = 5.0


def _positive_width(start: float, end: float) -> float:
    """Return the positive circular width from start to end."""
    return float((end - start) % 360.0)


def _angle_inside(start: float, width: float, angle: float) -> bool:
    """Test exact circular interval inclusion without an added tolerance band."""
    return bool((angle - start) % 360.0 <= width + 1.0e-9)


def _interval_gap(first: dict[str, Any], second: dict[str, Any]) -> float:
    """Return circular edge-to-edge gap for two intervals of width below 180°."""
    if _angle_inside(first["start_angle_deg"], first["width_deg"], second["center_angle_deg"]):
        return 0.0
    if _angle_inside(second["start_angle_deg"], second["width_deg"], first["center_angle_deg"]):
        return 0.0
    center_distance = abs(_normalize(first["center_angle_deg"] - second["center_angle_deg"]))
    return max(0.0, center_distance - 0.5 * (first["width_deg"] + second["width_deg"]))


def _run_record(
    run: np.ndarray,
    interval_id: int,
    angles: np.ndarray,
    angular_steps: np.ndarray,
    kind: str,
) -> dict[str, Any]:
    """Describe one global circular run using beam-boundary angles."""
    start = _normalize(float(angles[int(run[0])] - 0.5 * angular_steps[(int(run[0]) - 1) % len(angles)]))
    width = _run_width_deg(run, angular_steps)
    end = _normalize(start + width)
    return {
        "interval_id": interval_id,
        "interval_kind": kind,
        "start_angle_deg": start,
        "end_angle_deg": end,
        "center_angle_deg": _normalize(start + 0.5 * width),
        "width_deg": width,
        "beam_count": int(len(run)),
        "indices": run,
    }


def _extract_topology(angles: np.ndarray, ranges: np.ndarray) -> dict[str, Any]:
    """Extract scan-only raw intervals and frozen gap-filled detector groups."""
    openings, diagnostics = _detect_openings_with_diagnostics(angles, ranges)
    if openings != detect_openings(angles, ranges):
        raise RuntimeError("private diagnostic helper diverged from public detector")
    smoothed = np.asarray(diagnostics["smoothed_ranges"], dtype=float)
    raw_mask = smoothed >= float(diagnostics["open_threshold"])
    filled_mask = np.asarray(diagnostics["open_support_mask"], dtype=bool)
    angular_steps = np.diff(np.r_[angles, angles[0] + 360.0])
    raw = [
        _run_record(run, index, angles, angular_steps, "RAW")
        for index, run in enumerate(_circular_runs(raw_mask, value=True))
    ]
    groups = [
        _run_record(run, index, angles, angular_steps, "DETECTOR_GROUP")
        for index, run in enumerate(_circular_runs(filled_mask, value=True))
        if MIN_OPENING_WIDTH_DEG <= _run_width_deg(run, angular_steps) < 359.0
    ]
    for group in groups:
        members = [
            interval["interval_id"]
            for interval in raw
            if np.intersect1d(interval["indices"], group["indices"]).size
        ]
        group["raw_interval_ids"] = members
        group["merged_by_gap_rule"] = len(members) > 1
        group["matching_final_opening"] = min(
            openings,
            key=lambda opening: abs(_normalize(opening["center_angle"] - group["center_angle_deg"])),
            default=None,
        )
    return {
        "openings": openings,
        "diagnostics": diagnostics,
        "smoothed": smoothed,
        "raw_mask": raw_mask,
        "filled_mask": filled_mask,
        "raw_intervals": raw,
        "groups": groups,
        "angular_steps": angular_steps,
    }


def _branch_eval_geometry(
    runner: Any, snapshot: dict[str, Any], branch_label: str
) -> dict[str, Any]:
    """Build evaluation-only mouth ROI and branch progression-axis bearing."""
    branch_id = _branch_id(runner, branch_label)
    mouth = _gt_mouth_interval_eval_only(runner, snapshot, branch_id)
    start = float(mouth["start_angle"])
    width = _positive_width(start, float(mouth["end_angle"]))
    axis = next(
        float(item["local_angle_deg"])
        for item in _gt_directions_eval_only(runner, snapshot)[0]
        if int(item["branch_id"]) == branch_id
    )
    samples = _mouth_sample_audit(runner, snapshot, branch_id)
    return {
        "branch_id": branch_id,
        "mouth_start": start,
        "mouth_width": width,
        "mouth_center": _normalize(start + width / 2.0),
        "axis_angle": axis,
        "mouth_fraction": sum(bool(row["visible_eval"]) for row in samples) / len(samples),
        "mouth_samples": samples,
    }


def _overlap_beams(interval: dict[str, Any], roi_mask: np.ndarray) -> int:
    """Count scan beams shared by a global interval and evaluation ROI."""
    return int(np.count_nonzero(roi_mask[np.asarray(interval["indices"], dtype=int)]))


def _evaluate_branch(
    runner: Any,
    probe: dict[str, Any],
    role: str,
    branch_label: str,
    topology: dict[str, Any],
) -> dict[str, Any]:
    """Label already-extracted scan topology with post-hoc branch geometry."""
    snapshot = probe["snapshot"]
    angles = np.asarray(snapshot["angles"], dtype=float)
    geometry = _branch_eval_geometry(runner, snapshot, branch_label)
    roi = _interval_mask(angles, geometry["mouth_start"], geometry["mouth_width"])
    raw_matches = [item for item in topology["raw_intervals"] if _overlap_beams(item, roi)]
    group_matches = [item for item in topology["groups"] if _overlap_beams(item, roi)]

    # A branch-associated independent interval must be a complete global raw
    # component whose center lies in that branch's evaluation ROI.  Merely
    # intersecting the tail of an axial component is not a separate interval.
    independent_raw = [
        item
        for item in raw_matches
        if _angle_inside(geometry["mouth_start"], geometry["mouth_width"], item["center_angle_deg"])
    ]
    independent_groups = [
        item
        for item in group_matches
        if _angle_inside(geometry["mouth_start"], geometry["mouth_width"], item["center_angle_deg"])
    ]
    axis_raw = [
        item
        for item in raw_matches
        if _angle_inside(item["start_angle_deg"], item["width_deg"], geometry["axis_angle"])
    ]
    axis_groups = [
        item
        for item in group_matches
        if _angle_inside(item["start_angle_deg"], item["width_deg"], geometry["axis_angle"])
    ]
    distances = [
        max(
            0.0,
            abs(_normalize(geometry["axis_angle"] - item["center_angle_deg"]))
            - 0.5 * item["width_deg"],
        )
        for item in raw_matches
    ]
    detected_key = (
        f"{branch_label.lower()}_detected_eval"
        if branch_label in SIDE_LABELS
        else "forward_detected_eval"
    )
    detected = bool(probe[detected_key])
    if not raw_matches:
        failure = "ANGULAR_FREE_INTERVAL"
    elif not independent_raw:
        failure = "RAW_INTERVAL_INDEPENDENCE"
    elif not axis_raw:
        failure = "RAW_AXIS_SUPPORT"
    elif not independent_groups:
        failure = "DETECTOR_GROUP_INDEPENDENCE"
    elif not axis_groups:
        failure = "DETECTOR_GROUP_AXIS_SUPPORT"
    elif not detected:
        failure = "FINAL_OPENING"
    else:
        failure = "NONE_PASS"

    candidate_count = int(np.count_nonzero(topology["raw_mask"] & roi))
    row = {
        "viewpoint_id": probe["candidate_id"],
        "viewpoint_role": role,
        "branch_eval": branch_label,
        "mouth_point_los_fraction_eval": geometry["mouth_fraction"],
        "mouth_roi_start_angle_eval": geometry["mouth_start"],
        "mouth_roi_end_angle_eval": _normalize(geometry["mouth_start"] + geometry["mouth_width"]),
        "mouth_roi_width_deg_eval": geometry["mouth_width"],
        "branch_axis_angle_eval": geometry["axis_angle"],
        "angular_interval_present": bool(raw_matches),
        "raw_independent_interval_present": bool(independent_raw),
        "axis_supported_by_raw_interval_eval": bool(axis_raw),
        "detector_independent_group_present": bool(independent_groups),
        "axis_supported_by_detector_group_eval": bool(axis_groups),
        "final_opening_detected": detected,
        "first_failure_stage": failure,
        "candidate_beam_count_in_mouth_roi": candidate_count,
        "candidate_ratio_in_mouth_roi": candidate_count / max(1, int(np.count_nonzero(roi))),
        "overlapping_raw_interval_ids": json.dumps([item["interval_id"] for item in raw_matches]),
        "independent_raw_interval_ids": json.dumps([item["interval_id"] for item in independent_raw]),
        "overlapping_detector_group_ids": json.dumps([item["interval_id"] for item in group_matches]),
        "independent_detector_group_ids": json.dumps([item["interval_id"] for item in independent_groups]),
        "axis_to_nearest_raw_interval_distance_deg_eval": min(distances, default=180.0),
        "open_threshold": topology["diagnostics"]["open_threshold"],
        "gradient_threshold": topology["diagnostics"]["gradient_threshold"],
        "max_positive_discontinuity_in_roi": float(np.max(topology["diagnostics"]["gradient"][_interval_mask(np.asarray(topology["diagnostics"]["boundary_angles"]), geometry["mouth_start"], geometry["mouth_width"])])),
        "max_negative_discontinuity_in_roi": float(np.min(topology["diagnostics"]["gradient"][_interval_mask(np.asarray(topology["diagnostics"]["boundary_angles"]), geometry["mouth_start"], geometry["mouth_width"])])),
    }
    return {"row": row, "geometry": geometry, "roi": roi, "raw_matches": raw_matches, "group_matches": group_matches}


def _interval_rows(
    role: str,
    probe: dict[str, Any],
    topology: dict[str, Any],
    branch_geometries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Serialize global intervals and their nearest frozen detector groups."""
    rows = []
    for interval in topology["raw_intervals"]:
        nearest = min(topology["groups"], key=lambda group: _interval_gap(interval, group), default=None)
        containing = [
            group
            for group in topology["groups"]
            if np.intersect1d(interval["indices"], group["indices"]).size
        ]
        rows.append(
            {
                "viewpoint_id": probe["candidate_id"],
                "viewpoint_role": role,
                "interval_id": interval["interval_id"],
                "start_angle_deg": interval["start_angle_deg"],
                "end_angle_deg": interval["end_angle_deg"],
                "center_angle_deg": interval["center_angle_deg"],
                "width_deg": interval["width_deg"],
                "beam_count": interval["beam_count"],
                "raw_independent": True,
                "nearest_detector_group_id": "" if nearest is None else nearest["interval_id"],
                "angular_gap_to_nearest_group_deg": "" if nearest is None else _interval_gap(interval, nearest),
                "merged_by_gap_rule": bool(containing and containing[0]["merged_by_gap_rule"]),
                "plus90_axis_inside_eval": _angle_inside(interval["start_angle_deg"], interval["width_deg"], branch_geometries["PLUS90"]["axis_angle"]),
                "minus90_axis_inside_eval": _angle_inside(interval["start_angle_deg"], interval["width_deg"], branch_geometries["MINUS90"]["axis_angle"]),
                "forward_axis_inside_eval": _angle_inside(interval["start_angle_deg"], interval["width_deg"], branch_geometries["AXIAL_FORWARD"]["axis_angle"]),
            }
        )
    return rows


def _relationship_rows(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Quantify how each branch ROI intersects global raw/group topology."""
    rows = []
    for evaluation in evaluations:
        base = evaluation["row"]
        for interval in evaluation["raw_matches"]:
            group = min(evaluation["group_matches"], key=lambda item: _interval_gap(interval, item), default=None)
            rows.append(
                {
                    "viewpoint_id": base["viewpoint_id"],
                    "viewpoint_role": base["viewpoint_role"],
                    "branch_eval": base["branch_eval"],
                    "raw_interval_id": interval["interval_id"],
                    "raw_interval_center_deg": interval["center_angle_deg"],
                    "raw_interval_width_deg": interval["width_deg"],
                    "raw_center_inside_branch_roi_eval": _angle_inside(base["mouth_roi_start_angle_eval"], base["mouth_roi_width_deg_eval"], interval["center_angle_deg"]),
                    "nearest_detector_group_id": "" if group is None else group["interval_id"],
                    "detector_group_center_deg": "" if group is None else group["center_angle_deg"],
                    "detector_group_width_deg": "" if group is None else group["width_deg"],
                    "angular_overlap_beams_in_branch_roi_eval": _overlap_beams(interval, evaluation["roi"]),
                    "angular_gap_to_group_deg": "" if group is None else _interval_gap(interval, group),
                    "center_separation_to_group_deg": "" if group is None else abs(_normalize(interval["center_angle_deg"] - group["center_angle_deg"])),
                    "same_contiguous_run": True,
                    "merged_by_gap_rule": False if group is None else group["merged_by_gap_rule"],
                    "minimum_width_pass": interval["width_deg"] >= MIN_OPENING_WIDTH_DEG,
                }
            )
    return rows


def _metric_rows() -> list[dict[str, Any]]:
    """Document the semantic boundary between scan topology and GT labels."""
    return [
        {"metric": "MOUTH_POINT_LOS_FRACTION", "definition": "fraction of 21 physical mouth-segment samples with unobstructed LOS", "source": "GT_POSTHOC", "used_by_detector": False},
        {"metric": "ANGULAR_FREE_INTERVAL", "definition": "global contiguous run in the frozen pre-gap candidate mask", "source": "LIDAR_ONLY", "used_by_detector": False},
        {"metric": "RAW_INDEPENDENT_INTERVAL", "definition": "global raw run whose center lies inside the post-hoc branch mouth angular ROI", "source": "SCAN_TOPOLOGY_THEN_GT_LABEL", "used_by_detector": False},
        {"metric": "BRANCH_AXIS_SUPPORT", "definition": "exact circular inclusion of evaluation-only GT branch axis in a precomputed interval", "source": "GT_POSTHOC", "used_by_detector": False},
        {"metric": "DETECTOR_INDEPENDENT_GROUP", "definition": "frozen accepted gap-filled group whose center lies in the post-hoc branch ROI", "source": "DETECTOR_TOPOLOGY_THEN_GT_LABEL", "used_by_detector": True},
        {"metric": "FINAL_DETECTOR_OPENING", "definition": "unchanged public Point Cloud detector output", "source": "LIDAR_ONLY", "used_by_detector": True},
    ]


def _verdict(side_rows: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Classify full-mouth side topology using the requested EXP-032 cases."""
    full = [row for row in side_rows if "FULL_MOUTH" in row["viewpoint_role"]]
    if full and all(row["mouth_point_los_fraction_eval"] == 1.0 for row in full):
        if all(row["angular_interval_present"] and not row["raw_independent_interval_present"] for row in full):
            primary = "A_SIDE_HAS_NO_INDEPENDENT_ANGULAR_INTERVAL"
        elif all(row["raw_independent_interval_present"] and not row["axis_supported_by_raw_interval_eval"] for row in full):
            primary = "B_SIDE_RAW_INTERVAL_EXISTS_BUT_AXIS_UNSUPPORTED"
        elif all(row["axis_supported_by_raw_interval_eval"] and not row["detector_independent_group_present"] for row in full):
            primary = "C_SIDE_RAW_AXIS_OPENING_MERGED_BY_DETECTOR"
        elif all(row["axis_supported_by_raw_interval_eval"] and not row["final_opening_detected"] for row in full):
            primary = "D_SIDE_INDEPENDENT_AXIS_OPENING_EXISTS_DETECTOR_MISSES"
        else:
            primary = "E_MOUTH_LOS_ONLY_GEOMETRY_CONFIRMED"
    else:
        primary = "F_METRIC_INCONSISTENCY"
    secondary = ["MOUTH_POINT_LOS_NOT_OPENING_VISIBILITY"]
    if any(row["angular_interval_present"] for row in full):
        secondary.append("ANGULAR_CANDIDATE_TAIL_PRESENT")
    if all(not row["axis_supported_by_raw_interval_eval"] for row in full):
        secondary.append("BRANCH_AXIS_UNSUPPORTED")
    return primary, secondary


def _plot_topology(path: Path, scans: dict[str, dict[str, Any]]) -> None:
    """Plot full angular profiles, raw intervals, groups, and GT axes."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for axis, role in zip(axes, ("A0", "FULL_MOUTH")):
        item = scans[role]
        snapshot, topology = item["probe"]["snapshot"], item["topology"]
        angles = np.asarray(snapshot["angles"])
        axis.plot(angles, snapshot["ranges"], color="0.65", linewidth=1, label="raw range")
        axis.plot(angles, topology["smoothed"], color="black", linewidth=1, label="smoothed")
        axis.axhline(topology["diagnostics"]["open_threshold"], linestyle="--", color="tab:orange", label="frozen threshold")
        ymin, ymax = axis.get_ylim()
        for interval in topology["raw_intervals"]:
            mask = _interval_mask(angles, interval["start_angle_deg"], interval["width_deg"])
            axis.fill_between(angles, ymin, ymax, where=mask, alpha=0.12, color="tab:green")
        for group in topology["groups"]:
            axis.axvline(group["center_angle_deg"], color="tab:purple", linewidth=2, alpha=0.7)
        colors = {"PLUS90": "tab:red", "MINUS90": "tab:blue", "AXIAL_FORWARD": "tab:green"}
        for label, geometry in item["geometries"].items():
            axis.axvline(geometry["axis_angle"], linestyle=":", color=colors[label], label=f"{label} axis")
        axis.set(title=f"{role}: global raw intervals (green) and detector group centers (purple)", ylabel="range")
        axis.grid(alpha=0.2)
        axis.legend(ncol=4, fontsize=7)
    axes[-1].set_xlabel("body-local LiDAR angle [deg]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_hierarchy(path: Path, rows: list[dict[str, Any]]) -> None:
    """Visualize the requested visibility hierarchy as pass/fail."""
    fields = [
        "mouth_point_los_fraction_eval",
        "angular_interval_present",
        "raw_independent_interval_present",
        "axis_supported_by_raw_interval_eval",
        "detector_independent_group_present",
        "axis_supported_by_detector_group_eval",
        "final_opening_detected",
    ]
    labels = [f"{row['viewpoint_role']}\n{row['branch_eval']}" for row in rows]
    matrix = []
    for row in rows:
        matrix.append([row[field] > 0.0 if field == "mouth_point_los_fraction_eval" else bool(row[field]) for field in fields])
    fig, axis = plt.subplots(figsize=(13, 6))
    axis.imshow(np.asarray(matrix, dtype=int), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axis.set(xticks=np.arange(len(fields)), xticklabels=["MOUTH_LOS", "ANGULAR", "RAW_INDEP", "RAW_AXIS", "GROUP_INDEP", "GROUP_AXIS", "FINAL"], yticks=np.arange(len(rows)), yticklabels=labels, title="EXP-032 visibility hierarchy")
    for y, values in enumerate(matrix):
        for x, value in enumerate(values):
            axis.text(x, y, "PASS" if value else "FAIL", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_geometry(path: Path, runner: Any, scans: dict[str, dict[str, Any]]) -> None:
    """Show mouth LOS rays, blocked branch axes, and global interval wedges."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    colors = {"PLUS90": "tab:red", "MINUS90": "tab:blue"}
    for axis, role in zip(axes, ("A0", "FULL_MOUTH")):
        item = scans[role]
        snapshot, topology = item["probe"]["snapshot"], item["topology"]
        origin = np.asarray(snapshot["position_eval"], dtype=float)
        for wall in runner.geometry.walls:
            axis.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], color="black", linewidth=1)
        for interval in topology["raw_intervals"]:
            start_world = float(snapshot["yaw_eval"]) + interval["start_angle_deg"]
            axis.add_patch(Wedge(origin, 45.0, start_world, start_world + interval["width_deg"], color="tab:green", alpha=0.10))
        for label in SIDE_LABELS:
            geometry = item["geometries"][label]
            for sample in geometry["mouth_samples"]:
                target = np.array([sample["target_world_x_eval"], sample["target_world_y_eval"]])
                axis.plot([origin[0], target[0]], [origin[1], target[1]], color=colors[label] if sample["visible_eval"] else "0.7", alpha=0.12)
            world_axis = math.radians(float(snapshot["yaw_eval"]) + geometry["axis_angle"])
            target = origin + 70.0 * np.array([math.cos(world_axis), math.sin(world_axis)])
            axis.plot([origin[0], target[0]], [origin[1], target[1]], linestyle="--", color=colors[label], label=f"{label} axis")
        axis.scatter(*origin, marker="*", s=110, color="black", label="LiDAR")
        axis.set(xlim=(-80, 80), ylim=(-180, 65), aspect="equal", title=role, xlabel="world x", ylabel="world y")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _self_test() -> None:
    """Exercise circular extraction, topology separation, and axis inclusion."""
    mask = np.zeros(360, dtype=bool)
    mask[[358, 359, 0, 1]] = True
    runs = _circular_runs(mask, value=True)
    assert len(runs) == 1 and len(runs[0]) == 4
    assert _angle_inside(358.0, 4.0, 1.0)
    assert not _angle_inside(358.0, 4.0, 10.0)
    separated = np.zeros(12, dtype=bool)
    separated[1:4] = True
    separated[7:10] = True
    assert len(_circular_runs(separated, value=True)) == 2
    joined = separated.copy()
    joined[4:7] = True
    assert len(_circular_runs(joined, value=True)) == 1


def _deterministic_replay(runner: Any, anchor: dict[str, Any], scans: dict[str, dict[str, Any]]) -> bool:
    """Require exact range and public detector replay for both unique poses."""
    for item in scans.values():
        probe = item["probe"]
        replay = _probe(runner, anchor, "M1_CROSS_BASELINE", float(probe["forward_ratio_W"]), float(probe["lateral_ratio_W"]), "EXP032_REPLAY")
        if not np.array_equal(probe["snapshot"]["ranges"], replay["snapshot"]["ranges"]):
            return False
        if probe["detector_openings"] != replay["detector_openings"]:
            return False
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exp030-output", type=Path, default=EXP030_OUTPUT)
    parser.add_argument("--max-anchor-frames", type=int, default=120)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    _self_test()

    selected = _select_representatives(_read_exp030_rows(args.exp030_output / "branch_visibility_grid.csv"))
    a0_spec = next(item for item in selected if item["viewpoint_role"] == "A0")
    full_spec = next(item for item in selected if item["viewpoint_role"] == "PLUS90_FULL_MOUTH")
    if not math.isclose(float(full_spec["plus90_mouth_visible_fraction_eval"]), 1.0) or not math.isclose(float(full_spec["minus90_mouth_visible_fraction_eval"]), 1.0):
        raise RuntimeError("selected full-mouth pose is not 1.0 for both side branches")

    runner, anchor = _acquire_m1_anchor(args.max_anchor_frames)
    anchor["corridor_forward"] = runner.world.trusted_corridor_forward.copy()
    scans: dict[str, dict[str, Any]] = {}
    for role, spec in (("A0", a0_spec), ("FULL_MOUTH", full_spec)):
        probe = _probe(runner, anchor, "M1_CROSS_BASELINE", float(spec["forward_ratio_W"]), float(spec["lateral_ratio_W"]), "EXP032_REPRESENTATIVE")
        topology = _extract_topology(np.asarray(probe["snapshot"]["angles"], dtype=float), np.asarray(probe["snapshot"]["ranges"], dtype=float))
        geometries = {label: _branch_eval_geometry(runner, probe["snapshot"], label) for label in BRANCH_LABELS}
        scans[role] = {"probe": probe, "topology": topology, "geometries": geometries, "spec": spec}

    evaluations = []
    for role, item in scans.items():
        for label in SIDE_LABELS:
            evaluations.append(_evaluate_branch(runner, item["probe"], role, label, item["topology"]))
    evaluations.append(_evaluate_branch(runner, scans["A0"]["probe"], "A0", "AXIAL_FORWARD", scans["A0"]["topology"]))
    hierarchy = [item["row"] for item in evaluations]
    side_rows = [row for row in hierarchy if row["branch_eval"] in SIDE_LABELS]
    forward = next(row for row in hierarchy if row["branch_eval"] == "AXIAL_FORWARD")
    primary, secondary = _verdict(side_rows)
    side_full = [row for row in side_rows if row["viewpoint_role"] == "FULL_MOUTH"]
    first_divergence = next(
        (
            name
            for name, forward_value, side_values in (
                ("ANGULAR_FREE_INTERVAL", forward["angular_interval_present"], [row["angular_interval_present"] for row in side_full]),
                ("RAW_INTERVAL_INDEPENDENCE", forward["raw_independent_interval_present"], [row["raw_independent_interval_present"] for row in side_full]),
                ("RAW_AXIS_SUPPORT", forward["axis_supported_by_raw_interval_eval"], [row["axis_supported_by_raw_interval_eval"] for row in side_full]),
                ("DETECTOR_GROUP_INDEPENDENCE", forward["detector_independent_group_present"], [row["detector_independent_group_present"] for row in side_full]),
                ("DETECTOR_GROUP_AXIS_SUPPORT", forward["axis_supported_by_detector_group_eval"], [row["axis_supported_by_detector_group_eval"] for row in side_full]),
                ("FINAL_OPENING", forward["final_opening_detected"], [row["final_opening_detected"] for row in side_full]),
            )
            if forward_value and not all(side_values)
        ),
        "NO_DIVERGENCE",
    )
    deterministic = _deterministic_replay(runner, anchor, scans)

    representative_rows = []
    for role, item in scans.items():
        probe, spec = item["probe"], item["spec"]
        representative_rows.append({
            "viewpoint_role": role,
            "candidate_id": probe["candidate_id"],
            "forward_ratio_W": probe["forward_ratio_W"],
            "lateral_ratio_W": probe["lateral_ratio_W"],
            "candidate_valid": probe["candidate_valid"],
            "plus90_mouth_point_los_fraction_eval": spec["plus90_mouth_visible_fraction_eval"],
            "minus90_mouth_point_los_fraction_eval": spec["minus90_mouth_visible_fraction_eval"],
            "plus90_axis_los_eval": probe["plus90_axis_los_eval"],
            "minus90_axis_los_eval": probe["minus90_axis_los_eval"],
            "plus90_detected_eval": probe["plus90_detected_eval"],
            "minus90_detected_eval": probe["minus90_detected_eval"],
        })
    interval_rows = []
    for role, item in scans.items():
        interval_rows.extend(_interval_rows(role, item["probe"], item["topology"], item["geometries"]))
    verdict = {
        "experiment_id": EXPERIMENT_ID,
        "primary_verdict": primary,
        "secondary_findings": json.dumps(secondary),
        "first_divergence_stage": first_divergence,
        "representative_scan_count": len(scans),
        "deterministic_replay": deterministic,
        "A0_scan_public_detector_equivalent": scans["A0"]["topology"]["openings"] == scans["A0"]["probe"]["detector_openings"],
        "full_mouth_scan_public_detector_equivalent": scans["FULL_MOUTH"]["topology"]["openings"] == scans["FULL_MOUTH"]["probe"]["detector_openings"],
        "positive_forward_all_scan_topology_stages_pass": all(forward[field] for field in ("angular_interval_present", "raw_independent_interval_present", "axis_supported_by_raw_interval_eval", "detector_independent_group_present", "axis_supported_by_detector_group_eval", "final_opening_detected")),
        "circular_interval_self_test": True,
        "independence_topology_self_test": True,
        "branch_axis_inclusion_self_test": True,
        "GT_used_to_create_candidate_or_interval_topology": False,
        "production_detector_modified": False,
    }
    _write_csv(args.output / "representative_viewpoints.csv", representative_rows)
    _write_csv(args.output / "angular_intervals.csv", interval_rows)
    _write_csv(args.output / "interval_group_relationship.csv", _relationship_rows(evaluations))
    _write_csv(args.output / "branch_visibility_hierarchy.csv", hierarchy)
    _write_csv(args.output / "positive_control_comparison.csv", [{"control_type": "FORWARD_POSITIVE", **forward}, *[{"control_type": "SIDE_DIAGNOSTIC", **row} for row in side_rows]])
    _write_csv(args.output / "metric_definition_audit.csv", _metric_rows())
    _write_csv(args.output / "independent_opening_verdict.csv", [verdict])
    _plot_topology(args.output / "angular_interval_topology.png", scans)
    _plot_hierarchy(args.output / "visibility_hierarchy.png", hierarchy)
    _plot_geometry(args.output / "angular_opening_geometry.png", runner, scans)
    print(json.dumps({
        "experiment_id": EXPERIMENT_ID,
        "primary_verdict": primary,
        "secondary_findings": secondary,
        "first_divergence_stage": first_divergence,
        "hierarchy": [{key: row[key] for key in ("viewpoint_role", "branch_eval", "mouth_point_los_fraction_eval", "angular_interval_present", "raw_independent_interval_present", "axis_supported_by_raw_interval_eval", "detector_independent_group_present", "axis_supported_by_detector_group_eval", "final_opening_detected", "first_failure_stage")} for row in hierarchy],
        "deterministic_replay": deterministic,
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
