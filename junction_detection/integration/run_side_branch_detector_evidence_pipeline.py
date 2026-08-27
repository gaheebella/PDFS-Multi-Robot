"""EXP-031: locate where side-mouth evidence leaves the frozen detector pipeline.

This targeted diagnostic reuses EXP-030 representative ghost poses and the
production detector's private diagnostics without modifying its source. GT
branch mouths define evaluation-only ROIs after detector inference. The real
detector has no wall/tangent acceptance stages; optional ROI wall estimates
are therefore explicitly post-hoc diagnostics, never rejection conditions.
"""

from __future__ import annotations

import argparse
import csv
import inspect
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

from junction_detection.integration.pointcloud_wall_parallel_orientation import (
    estimate_wall_parallel_tangent,
)
from junction_detection.integration.run_2d_viewpoint_visibility_frontier import (
    MOUTH_SAMPLE_COUNT,
    _branch_mouth_points,
    _local_to_world,
    _probe,
)
from junction_detection.integration.run_active_viewpoint_acquisition import (
    _gt_mouth_interval_eval_only,
)
from junction_detection.integration.run_local_asymmetric_viewpoint_geometry_diagnostic import (
    _acquire_m0_snapshot,
    _acquire_m1_anchor,
)
from junction_detection.integration.run_provisional_anchor_pointcloud_observation import (
    _gt_directions_eval_only,
    _normalize,
)
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import (
    _circular_runs,
    _detect_openings_with_diagnostics,
    _run_width_deg,
    detect_openings,
)

EXPERIMENT_ID = "EXP-031"
DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/side_branch_detector_evidence_pipeline"
EXP030_OUTPUT = ROOT / "junction_detection/integration/output/2d_viewpoint_visibility_frontier"
SIDE_LABELS = ("PLUS90", "MINUS90")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write heterogeneous dictionaries using their stable field union."""
    if not rows:
        return
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_exp030_rows(path: Path) -> list[dict[str, Any]]:
    """Read the completed EXP-030 branch grid instead of repeating its sweep."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _select_representatives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select A0 and nearest valid maximum-mouth poses from actual EXP-030 data."""
    valid = [row for row in rows if row["candidate_valid"] == "True"]
    a0 = next(
        row
        for row in valid
        if math.isclose(float(row["forward_ratio_W"]), 0.0)
        and math.isclose(float(row["lateral_ratio_W"]), 0.0)
    )
    selected = [{"viewpoint_role": "A0", **a0}]
    for label, prefix in (("PLUS90_FULL_MOUTH", "plus90"), ("MINUS90_FULL_MOUTH", "minus90")):
        maximum = max(float(row[f"{prefix}_mouth_visible_fraction_eval"]) for row in valid)
        candidates = [
            row
            for row in valid
            if math.isclose(float(row[f"{prefix}_mouth_visible_fraction_eval"]), maximum)
        ]
        best = min(
            candidates,
            key=lambda row: (
                float(row["distance_ratio_W"]),
                float(row["forward_ratio_W"]),
                abs(float(row["lateral_ratio_W"])),
            ),
        )
        selected.append({"viewpoint_role": label, **best})
    return selected


def _branch_id(runner: Any, label: str) -> int:
    """Resolve an evaluation-only GT label to its simulator branch ID."""
    target = {"AXIAL_FORWARD": 0.0, "PLUS90": 90.0, "MINUS90": -90.0}[label]
    return next(
        index
        for index, branch in enumerate(runner.geometry.branches)
        if math.isclose(float(branch.angle_deg), target)
    )


def _interval_mask(
    angles: np.ndarray, start_angle: float, width_deg: float
) -> np.ndarray:
    """Return a circular angular interval mask."""
    return (angles - start_angle) % 360.0 <= width_deg + 1.0e-9


def _run_ids(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Assign circular run IDs, leaving non-members as -1."""
    runs = _circular_runs(mask, value=True)
    identifiers = np.full(mask.size, -1, dtype=int)
    for run_id, run in enumerate(runs):
        identifiers[run] = run_id
    return identifiers, runs


def _ordered_roi_runs(mask: np.ndarray, roi: np.ndarray, angles: np.ndarray, start: float) -> list[np.ndarray]:
    """Return contiguous true runs after ordering beams inside a non-wrapped ROI."""
    indices = np.flatnonzero(roi)
    if not len(indices):
        return []
    order = indices[np.argsort((angles[indices] - start) % 360.0)]
    runs: list[list[int]] = []
    current: list[int] = []
    for index in order:
        if mask[index]:
            current.append(int(index))
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return [np.asarray(run, dtype=int) for run in runs]


def _mouth_sample_audit(
    runner: Any, snapshot: dict[str, Any], branch_id: int
) -> list[dict[str, Any]]:
    """Record all EXP-030 mouth samples, target ranges, and first occluders."""
    origin = np.asarray(snapshot["position_eval"], dtype=float)
    maximum = float(snapshot["max_range"])
    rows = []
    for sample_id, target in enumerate(_branch_mouth_points(runner, branch_id)):
        relative = target - origin
        distance = float(np.linalg.norm(relative))
        direction = relative / max(distance, np.finfo(float).eps)
        hits = [
            (wall_id, hit)
            for wall_id, wall in enumerate(runner.geometry.walls)
            if (hit := runner.world.sensor._ray_hit(origin, direction, wall)) is not None
        ]
        wall_id, nearest = min(hits, key=lambda item: item[1], default=(-1, math.inf))
        visible = distance <= maximum and nearest >= distance - 1.0e-7
        world_bearing = math.degrees(math.atan2(float(relative[1]), float(relative[0])))
        rows.append(
            {
                "sample_id": sample_id,
                "target_world_x_eval": float(target[0]),
                "target_world_y_eval": float(target[1]),
                "target_range_eval": distance,
                "bearing_body_deg_eval": _normalize(world_bearing - float(snapshot["yaw_eval"])),
                "visible_eval": visible,
                "first_wall_id_eval": wall_id,
                "first_wall_distance_eval": "" if not math.isfinite(nearest) else nearest,
            }
        )
    return rows


def _nearest_wall_for_beam(
    runner: Any, snapshot: dict[str, Any], beam_angle: float
) -> tuple[int, float]:
    """Identify the first evaluation-only wall intersected by one LiDAR beam."""
    world_angle = math.radians(float(snapshot["yaw_eval"]) + beam_angle)
    direction = np.array([math.cos(world_angle), math.sin(world_angle)])
    origin = np.asarray(snapshot["position_eval"], dtype=float)
    hits = [
        (wall_id, hit)
        for wall_id, wall in enumerate(runner.geometry.walls)
        if (hit := runner.world.sensor._ray_hit(origin, direction, wall)) is not None
    ]
    return min(hits, key=lambda item: item[1], default=(-1, math.inf))


def _pseudo_roi_wall_estimate(
    snapshot: dict[str, Any], start: float, width: float
) -> Any:
    """Run the existing post-hoc wall estimator on a GT ROI, never as acceptance."""
    opening = {
        "start_angle": start,
        "end_angle": _normalize(start + width),
        "center_angle": _normalize(start + width / 2.0),
        "width_deg": width,
    }
    return estimate_wall_parallel_tangent(
        snapshot["angles"], snapshot["ranges"], snapshot["max_range"], opening
    )


def _analyze_branch(
    runner: Any,
    probe: dict[str, Any],
    viewpoint_role: str,
    branch_label: str,
) -> dict[str, Any]:
    """Trace one GT/eval ROI through the exact range-mask/group pipeline."""
    snapshot = probe["snapshot"]
    angles = np.asarray(snapshot["angles"], dtype=float)
    ranges = np.asarray(snapshot["ranges"], dtype=float)

    # Run the frozen production detector before constructing any GT/evaluation
    # ROI.  The ROI below is used only to label and summarize detector evidence.
    final_openings, diagnostics = _detect_openings_with_diagnostics(angles, ranges)
    direct_openings = detect_openings(angles, ranges)
    if final_openings != direct_openings:
        raise RuntimeError("diagnostic helper and public detector diverged")

    branch_id = _branch_id(runner, branch_label)
    gt_interval = _gt_mouth_interval_eval_only(runner, snapshot, branch_id)
    start = float(gt_interval["start_angle"])
    width = float((float(gt_interval["end_angle"]) - start) % 360.0)
    roi = _interval_mask(angles, start, width)

    smoothed = np.asarray(diagnostics["smoothed_ranges"], dtype=float)
    filled_mask = np.asarray(diagnostics["open_support_mask"], dtype=bool)
    raw_mask = smoothed >= float(diagnostics["open_threshold"])
    angular_steps = np.diff(np.r_[angles, angles[0] + 360.0])
    raw_ids, raw_runs = _run_ids(raw_mask)
    group_ids, groups = _run_ids(filled_mask)
    group_widths = [_run_width_deg(run, angular_steps) for run in groups]
    accepted_groups = {
        index
        for index, group_width in enumerate(group_widths)
        if group_width >= 5.0 and group_width < 359.0
    }
    roi_runs = _ordered_roi_runs(raw_mask, roi, angles, start)
    roi_run_widths = [_run_width_deg(run, angular_steps) for run in roi_runs]
    axis_local = next(
        item["local_angle_deg"]
        for item in _gt_directions_eval_only(runner, snapshot)[0]
        if item["branch_id"] == branch_id
    )
    axis_index = int(
        np.argmin(np.abs(((angles - float(axis_local) + 180.0) % 360.0) - 180.0))
    )
    group_id_at_axis = int(group_ids[axis_index])
    group_pass = group_id_at_axis in accepted_groups
    detected = bool(probe[f"{branch_label.lower()}_detected_eval"] if branch_label in SIDE_LABELS else probe["forward_detected_eval"])
    candidate_count = int(np.count_nonzero(raw_mask & roi))
    longest_run = max((len(run) for run in roi_runs), default=0)
    longest_width = max(roi_run_widths, default=0.0)
    roi_peak_above_wall = float(np.max(smoothed[roi]) - diagnostics["wall_reference"])
    range_evidence = roi_peak_above_wall > np.finfo(float).eps * max(1.0, float(np.max(ranges))) * 128.0
    overlapping_groups = {
        int(group_id)
        for group_id in np.unique(group_ids[roi])
        if int(group_id) >= 0 and int(group_id) in accepted_groups
    }
    mouth_samples = _mouth_sample_audit(runner, snapshot, branch_id)
    visible_samples = sum(row["visible_eval"] for row in mouth_samples)
    mouth_fraction = visible_samples / MOUTH_SAMPLE_COUNT

    gradient = np.asarray(diagnostics["gradient"], dtype=float)
    boundary_angles = np.asarray(diagnostics["boundary_angles"], dtype=float)
    boundary_roi = _interval_mask(boundary_angles, start, width)
    strong = boundary_roi & (np.abs(gradient) >= float(diagnostics["gradient_threshold"]))
    positive = gradient[boundary_roi]
    max_positive = float(np.max(positive)) if len(positive) else 0.0
    max_negative = float(np.min(positive)) if len(positive) else 0.0

    if detected:
        first_failure = "NONE_FINAL_OPENING_PASS"
    elif not range_evidence:
        first_failure = "NO_RANGE_EVIDENCE"
    elif candidate_count == 0:
        first_failure = "RANGE_EVIDENCE_NOT_CANDIDATE"
    elif not group_pass:
        if len(roi_runs) > 1:
            first_failure = "CANDIDATE_FRAGMENTED"
        elif group_id_at_axis < 0:
            first_failure = "AXIS_NOT_SUPPORTED_BY_ACCEPTED_GROUP"
        else:
            first_failure = "GROUP_BELOW_MIN_WIDTH"
    else:
        first_failure = "UNEXPECTED_PIPELINE_INCONSISTENCY"

    posthoc_wall = _pseudo_roi_wall_estimate(snapshot, start, width)
    final_masks = [
        _interval_mask(
            angles,
            float(opening["start_angle"]),
            float(opening["width_deg"]),
        )
        for opening in final_openings
    ]
    final_member = np.any(final_masks, axis=0) if final_masks else np.zeros(len(angles), dtype=bool)
    corridor_forward = np.asarray(probe["anchor_corridor_forward"], dtype=float)
    corridor_world = math.degrees(math.atan2(float(corridor_forward[1]), float(corridor_forward[0])))
    corridor_body = _normalize(corridor_world - float(snapshot["yaw_eval"]))

    ray_rows = []
    selected_points = np.asarray(posthoc_wall.selected_points)
    for index in np.flatnonzero(roi):
        wall_id, wall_distance = _nearest_wall_for_beam(runner, snapshot, float(angles[index]))
        local_point = np.array(
            [
                ranges[index] * math.cos(math.radians(float(angles[index]))),
                ranges[index] * math.sin(math.radians(float(angles[index]))),
            ]
        )
        posthoc_contribution = bool(
            len(selected_points)
            and np.min(np.linalg.norm(selected_points - local_point, axis=1)) <= 1.0e-7
        )
        ray_rows.append(
            {
                "viewpoint_id": probe["candidate_id"],
                "viewpoint_role": viewpoint_role,
                "branch_eval": branch_label,
                "beam_index": int(index),
                "angle_body_deg": float(angles[index]),
                "angle_corridor_deg": _normalize(float(angles[index]) - corridor_body),
                "measured_range": float(ranges[index]),
                "smoothed_range": float(smoothed[index]),
                "is_max_range": bool(ranges[index] >= snapshot["max_range"] - np.finfo(float).eps * snapshot["max_range"] * 64.0),
                "hit": bool(snapshot["hit"][index]),
                "hit_wall_id_eval": wall_id,
                "first_wall_distance_eval": "" if not math.isfinite(wall_distance) else wall_distance,
                "raw_mouth_ray_eval": True,
                "axis_related_eval": int(index) == axis_index,
                "range_discontinuity": float(gradient[index]),
                "range_evidence": bool(smoothed[index] > diagnostics["wall_reference"]),
                "candidate_before_grouping": bool(raw_mask[index]),
                "candidate_mask": bool(filled_mask[index]),
                "candidate_run_id": int(raw_ids[index]),
                "candidate_group_id": int(group_ids[index]),
                "group_accepted": int(group_ids[index]) in accepted_groups,
                "wall_support": "N/A_NOT_IN_DETECTOR_PIPELINE",
                "tangent_support": "N/A_NOT_IN_DETECTOR_PIPELINE",
                "posthoc_wall_support_contribution_eval": posthoc_contribution,
                "final_opening_member": bool(final_member[index]),
                "rejection_stage": first_failure,
            }
        )

    stage = {
        "viewpoint_id": probe["candidate_id"],
        "viewpoint_role": viewpoint_role,
        "branch_eval": branch_label,
        "mouth_visible_fraction_eval": mouth_fraction,
        "mouth_sample_count_eval": MOUTH_SAMPLE_COUNT,
        "mouth_visible_sample_count_eval": visible_samples,
        "axis_los_eval": bool(probe[f"{branch_label.lower()}_axis_los_eval"] if branch_label in SIDE_LABELS else probe["forward_axis_los_eval"]),
        "ROI_start_angle_eval": start,
        "ROI_end_angle_eval": _normalize(start + width),
        "ROI_width_deg_eval": width,
        "ROI_beam_count": int(np.count_nonzero(roi)),
        "range_evidence_present": range_evidence,
        "ROI_peak_above_wall_reference": roi_peak_above_wall,
        "max_positive_jump": max_positive,
        "max_negative_jump": max_negative,
        "strong_discontinuity_count": int(np.count_nonzero(strong)),
        "strong_discontinuity_angles": json.dumps(boundary_angles[strong].tolist()),
        "candidate_beam_count": candidate_count,
        "longest_candidate_run": longest_run,
        "longest_candidate_run_width_deg": longest_width,
        "candidate_run_lengths": json.dumps([len(run) for run in roi_runs]),
        "candidate_group_count": len(overlapping_groups),
        "candidate_group_ids": json.dumps(sorted(overlapping_groups)),
        "axis_candidate_before_grouping": bool(raw_mask[axis_index]),
        "axis_candidate_after_gap_fill": bool(filled_mask[axis_index]),
        "axis_group_id": group_id_at_axis,
        "group_pass": group_pass,
        "wall_support_pass": "N/A_NOT_IN_DETECTOR_PIPELINE",
        "tangent_support_pass": "N/A_NOT_IN_DETECTOR_PIPELINE",
        "final_opening_detected": detected,
        "first_failure_stage": first_failure,
        "public_detector_equivalent": final_openings == direct_openings,
        "open_threshold": diagnostics["open_threshold"],
        "wall_reference": diagnostics["wall_reference"],
        "range_ceiling": diagnostics["range_ceiling"],
        "gradient_threshold": diagnostics["gradient_threshold"],
    }
    wall = {
        "viewpoint_id": probe["candidate_id"],
        "viewpoint_role": viewpoint_role,
        "branch_eval": branch_label,
        "detector_wall_support_stage_present": False,
        "detector_tangent_stage_present": False,
        "detector_rejection_can_be_wall_or_tangent": False,
        "posthoc_GT_ROI_estimate_mode_eval": posthoc_wall.estimate_mode,
        "posthoc_GT_ROI_usable_wall_sides_eval": posthoc_wall.usable_wall_sides,
        "posthoc_GT_ROI_fitted_point_count_eval": posthoc_wall.fitted_point_count,
        "posthoc_GT_ROI_line_residual_eval": "" if posthoc_wall.line_fit_residual_m is None else posthoc_wall.line_fit_residual_m,
        "posthoc_GT_ROI_tangent_available_eval": posthoc_wall.tangent_deg is not None,
        "posthoc_GT_ROI_tangent_deg_eval": "" if posthoc_wall.tangent_deg is None else posthoc_wall.tangent_deg,
        "note": "Post-hoc ROI estimator is not an opening acceptance stage",
    }
    group = {
        "viewpoint_id": probe["candidate_id"],
        "viewpoint_role": viewpoint_role,
        "branch_eval": branch_label,
        "raw_candidate_run_count_in_ROI": len(roi_runs),
        "raw_candidate_run_lengths": json.dumps([len(run) for run in roi_runs]),
        "filled_group_count_overlapping_ROI": len(overlapping_groups),
        "filled_group_ids_overlapping_ROI": json.dumps(sorted(overlapping_groups)),
        "all_filled_group_widths_deg": json.dumps(group_widths),
        "axis_group_id": group_id_at_axis,
        "axis_group_accepted": group_pass,
        "final_opening_count": len(final_openings),
        "side_final_opening_detected": detected,
        "rejection_reason": first_failure,
    }
    return {
        "stage": stage,
        "rays": ray_rows,
        "wall": wall,
        "group": group,
        "mouth_samples": mouth_samples,
        "diagnostics": diagnostics,
        "final_openings": final_openings,
        "roi_mask": roi,
        "raw_mask": raw_mask,
        "filled_mask": filled_mask,
    }


def _threshold_rows(analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report frozen default parameters and per-scan adaptive values."""
    signature = inspect.signature(_detect_openings_with_diagnostics)
    defaults = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    results = []
    for analysis in analyses:
        stage = analysis["stage"]
        results.append(
            {
                "viewpoint_id": stage["viewpoint_id"],
                "viewpoint_role": stage["viewpoint_role"],
                "branch_eval": stage["branch_eval"],
                "smoothing_window_size": defaults["smoothing_window_size"],
                "wall_reference_quantile": defaults["wall_reference_quantile"],
                "far_range_fraction": defaults["far_range_fraction"],
                "merge_gap_deg": defaults["merge_gap_deg"],
                "min_opening_width_deg": defaults["min_opening_width_deg"],
                "minimum_beam_count_equivalent_at_1deg": int(math.ceil(defaults["min_opening_width_deg"])),
                "gradient_threshold_setting": "AUTO" if defaults["gradient_threshold"] is None else defaults["gradient_threshold"],
                "gradient_mad_scale": defaults["gradient_mad_scale"],
                "min_gradient_threshold": defaults["min_gradient_threshold"],
                "boundary_search_deg": defaults["boundary_search_deg"],
                "computed_open_threshold": stage["open_threshold"],
                "computed_wall_reference": stage["wall_reference"],
                "computed_range_ceiling": stage["range_ceiling"],
                "computed_gradient_threshold": stage["gradient_threshold"],
                "wall_support_acceptance_condition": "NONE_IN_DETECTOR",
                "tangent_acceptance_condition": "NONE_IN_DETECTOR",
            }
        )
    return results


def _representative_rows(
    selected: list[dict[str, Any]], probes: dict[str, dict[str, Any]], analyses: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Record selected EXP-030 candidates and audited mouth sample geometry."""
    results = []
    for item in selected:
        role = item["viewpoint_role"]
        probe = probes[role]
        role_analyses = [analysis for analysis in analyses if analysis["stage"]["viewpoint_role"] == role]
        results.append(
            {
                "viewpoint_role": role,
                "candidate_id": probe["candidate_id"],
                "forward_ratio_W": probe["forward_ratio_W"],
                "lateral_ratio_W": probe["lateral_ratio_W"],
                "distance_ratio_W": probe["distance_ratio_W"],
                "candidate_valid": probe["candidate_valid"],
                "plus90_mouth_fraction_EXP030": probe["plus90_mouth_visible_fraction_eval"],
                "minus90_mouth_fraction_EXP030": probe["minus90_mouth_visible_fraction_eval"],
                "plus90_axis_los_eval": probe["plus90_axis_los_eval"],
                "minus90_axis_los_eval": probe["minus90_axis_los_eval"],
                "plus90_detected_eval": probe["plus90_detected_eval"],
                "minus90_detected_eval": probe["minus90_detected_eval"],
                "mouth_sample_count_per_branch": MOUTH_SAMPLE_COUNT,
                "mouth_sample_locations_eval": json.dumps(
                    {
                        analysis["stage"]["branch_eval"]: [
                            [row["target_world_x_eval"], row["target_world_y_eval"]]
                            for row in analysis["mouth_samples"]
                        ]
                        for analysis in role_analyses
                    }
                ),
                "mouth_sample_visibility_eval": json.dumps(
                    {
                        analysis["stage"]["branch_eval"]: [
                            row["visible_eval"] for row in analysis["mouth_samples"]
                        ]
                        for analysis in role_analyses
                    }
                ),
            }
        )
    return results


def _positive_comparison(analyses: list[dict[str, Any]], m0_probe: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare A0 forward positive control, side failures, and M0 stability."""
    results = []
    for analysis in analyses:
        stage = analysis["stage"]
        if stage["viewpoint_role"] == "A0" or stage["branch_eval"] in SIDE_LABELS:
            results.append(
                {
                    "control_type": "POSITIVE_FORWARD" if stage["branch_eval"] == "AXIAL_FORWARD" else "SIDE_DIAGNOSTIC",
                    **stage,
                }
            )
    results.append(
        {
            "control_type": "M0_NEGATIVE_CONTROL",
            "viewpoint_id": m0_probe["candidate_id"],
            "viewpoint_role": "M0_A0",
            "branch_eval": "M0_AXIS_ARTIFACT",
            "mouth_visible_fraction_eval": "N/A",
            "axis_los_eval": m0_probe["incoming_axis_los_eval"],
            "final_opening_detected": m0_probe["opening_count"] > 0,
            "first_failure_stage": "EXISTING_AXIS_ARTIFACT_UNCHANGED",
            "opening_count": m0_probe["opening_count"],
            "false_opening_count_eval": m0_probe["false_opening_count_eval"],
        }
    )
    return results


def _verdict(side_analyses: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Classify the first actual loss while preserving mouth-metric semantics."""
    stages = [analysis["stage"] for analysis in side_analyses]
    full_mouth = [stage for stage in stages if math.isclose(stage["mouth_visible_fraction_eval"], 1.0)]
    failures = {stage["first_failure_stage"] for stage in stages}
    if full_mouth and all(not stage["axis_los_eval"] for stage in full_mouth) and all(
        not stage["final_opening_detected"] for stage in full_mouth
    ):
        primary = "G_MOUTH_VISIBILITY_METRIC_NOT_EQUIVALENT_TO_OPENING_EVIDENCE"
    elif failures == {"NO_RANGE_EVIDENCE"}:
        primary = "A_SIDE_EVIDENCE_LOST_AT_RANGE_STAGE"
    elif "RANGE_EVIDENCE_NOT_CANDIDATE" in failures:
        primary = "B_SIDE_EVIDENCE_LOST_AT_CANDIDATE_STAGE"
    elif failures & {
        "CANDIDATE_FRAGMENTED",
        "AXIS_NOT_SUPPORTED_BY_ACCEPTED_GROUP",
        "GROUP_BELOW_MIN_WIDTH",
    }:
        primary = "C_SIDE_EVIDENCE_LOST_AT_GROUPING_STAGE"
    elif "UNEXPECTED_PIPELINE_INCONSISTENCY" in failures:
        primary = "H_PIPELINE_IMPLEMENTATION_INCONSISTENCY"
    else:
        primary = "F_SIDE_EVIDENCE_LOST_AT_FINAL_VALIDATION"
    secondary = ["MOUTH_LOS_PRESENT"]
    if all(not stage["axis_los_eval"] for stage in stages):
        secondary.append("AXIS_LOS_BLOCKED")
    if any(stage["candidate_beam_count"] > 0 for stage in stages):
        secondary.append("CANDIDATE_BEAMS_PRESENT")
    if any(stage["first_failure_stage"] == "AXIS_NOT_SUPPORTED_BY_ACCEPTED_GROUP" for stage in stages):
        secondary.append("AXIS_NOT_SUPPORTED_BY_ACCEPTED_GROUP")
    if any(stage["first_failure_stage"] == "CANDIDATE_FRAGMENTED" for stage in stages):
        secondary.append("CANDIDATE_FRAGMENTED")
    secondary.append("WALL_TANGENT_NOT_DETECTOR_STAGES")
    return primary, secondary


def _plot_profiles(path: Path, analyses: list[dict[str, Any]]) -> None:
    """Compare A0/full-mouth side ROIs with masks and final openings."""
    side = [analysis for analysis in analyses if analysis["stage"]["branch_eval"] in SIDE_LABELS]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    chosen = []
    for label in SIDE_LABELS:
        chosen.append(next(a for a in side if a["stage"]["viewpoint_role"] == "A0" and a["stage"]["branch_eval"] == label))
        full_mouth_role = f"{label}_FULL_MOUTH"
        chosen.append(next(a for a in side if a["stage"]["viewpoint_role"] == full_mouth_role and a["stage"]["branch_eval"] == label))
    for axis, analysis in zip(axes.flat, chosen):
        stage = analysis["stage"]
        rays = analysis["rays"]
        angles = np.array([row["angle_corridor_deg"] for row in rays])
        raw = np.array([row["measured_range"] for row in rays])
        smooth = np.array([row["smoothed_range"] for row in rays])
        candidate = np.array([row["candidate_mask"] for row in rays], dtype=bool)
        axis.plot(angles, raw, "o-", label="raw", markersize=3)
        axis.plot(angles, smooth, "-", label="smoothed")
        axis.axhline(stage["open_threshold"], linestyle="--", label="open threshold")
        axis.scatter(angles[candidate], smooth[candidate], marker="s", label="candidate")
        axis.set(
            title=f"{stage['viewpoint_role']} / {stage['branch_eval']}\n"
            f"mouth={stage['mouth_visible_fraction_eval']:.3f}, failure={stage['first_failure_stage']}",
            xlabel="corridor-relative angle [deg]",
            ylabel="range",
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_pipeline(path: Path, analyses: list[dict[str, Any]]) -> None:
    """Show pass/fail rows from mouth evidence through final opening."""
    selected = [
        analysis
        for analysis in analyses
        if analysis["stage"]["viewpoint_role"] == "A0"
        or "FULL_MOUTH" in analysis["stage"]["viewpoint_role"]
    ]
    labels = [f"{a['stage']['viewpoint_role']}\n{a['stage']['branch_eval']}" for a in selected]
    stages = ("MOUTH", "RANGE", "CANDIDATE", "AXIS_GROUP", "FINAL")
    matrix = []
    for analysis in selected:
        row = analysis["stage"]
        matrix.append(
            [
                row["mouth_visible_fraction_eval"] > 0.0,
                row["range_evidence_present"],
                row["candidate_beam_count"] > 0,
                row["group_pass"],
                row["final_opening_detected"],
            ]
        )
    fig, axis = plt.subplots(figsize=(11, 5.5))
    image = axis.imshow(np.asarray(matrix, dtype=int), aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    axis.set(xticks=np.arange(len(stages)), xticklabels=stages, yticks=np.arange(len(labels)), yticklabels=labels, title="EXP-031 pipeline stage pass/fail")
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            axis.text(x, y, "PASS" if value else "FAIL", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, ticks=(0, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_geometry(path: Path, runner: Any, probes: dict[str, dict[str, Any]]) -> None:
    """Show mouth sample LOS and blocked side axes at representative poses."""
    roles = ("A0", "PLUS90_FULL_MOUTH", "MINUS90_FULL_MOUTH")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, role in zip(axes, roles):
        probe = probes[role]
        origin = np.asarray(probe["snapshot"]["position_eval"])
        for wall in runner.geometry.walls:
            axis.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], color="black", linewidth=1)
        for label, color in (("PLUS90", "tab:red"), ("MINUS90", "tab:blue")):
            branch_id = _branch_id(runner, label)
            samples = _mouth_sample_audit(runner, probe["snapshot"], branch_id)
            points = _branch_mouth_points(runner, branch_id)
            axis.plot(points[:, 0], points[:, 1], color=color, linewidth=3, label=f"{label} mouth")
            for sample, target in zip(samples, points):
                axis.plot([origin[0], target[0]], [origin[1], target[1]], color=color if sample["visible_eval"] else "gray", alpha=0.12)
            branch = runner.geometry.branches[branch_id]
            radians = math.radians(float(branch.angle_deg))
            direction = np.array([math.sin(radians), math.cos(radians)])
            axis.plot([origin[0], origin[0] + direction[0] * probe["snapshot"]["max_range"]], [origin[1], origin[1] + direction[1] * probe["snapshot"]["max_range"]], "--", color=color, alpha=0.7)
        axis.scatter(*origin, marker="*", s=100, color="black", label="LiDAR")
        axis.set(xlim=(-75, 75), ylim=(-180, 60), aspect="equal", title=role, xlabel="world x", ylabel="world y")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _deterministic_replay(runner: Any, anchor: dict[str, Any], probes: dict[str, dict[str, Any]]) -> bool:
    """Replay each unique representative and require exact scan/detector output."""
    seen = set()
    for probe in probes.values():
        key = (probe["forward_ratio_W"], probe["lateral_ratio_W"])
        if key in seen:
            continue
        seen.add(key)
        repeated = _probe(runner, anchor, probe["case"], float(key[0]), float(key[1]), "EXP031_REPLAY")
        if not np.array_equal(probe["snapshot"]["ranges"], repeated["snapshot"]["ranges"]):
            return False
        if probe["detector_openings"] != repeated["detector_openings"]:
            return False
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exp030-output", type=Path, default=EXP030_OUTPUT)
    parser.add_argument("--max-anchor-frames", type=int, default=120)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    selected = _select_representatives(
        _read_exp030_rows(args.exp030_output / "branch_visibility_grid.csv")
    )
    runner, anchor = _acquire_m1_anchor(args.max_anchor_frames)
    anchor["corridor_forward"] = runner.world.trusted_corridor_forward.copy()
    probes: dict[str, dict[str, Any]] = {}
    coordinate_cache: dict[tuple[float, float], dict[str, Any]] = {}
    for item in selected:
        key = (float(item["forward_ratio_W"]), float(item["lateral_ratio_W"]))
        if key not in coordinate_cache:
            coordinate_cache[key] = _probe(runner, anchor, "M1_CROSS_BASELINE", key[0], key[1], "EXP031_REPRESENTATIVE")
            coordinate_cache[key]["anchor_corridor_forward"] = anchor["corridor_forward"].copy()
        probes[item["viewpoint_role"]] = coordinate_cache[key]

    analyses = []
    for role, probe in probes.items():
        for branch_label in SIDE_LABELS:
            analyses.append(_analyze_branch(runner, probe, role, branch_label))
    analyses.append(_analyze_branch(runner, probes["A0"], "A0", "AXIAL_FORWARD"))
    side_analyses = [analysis for analysis in analyses if analysis["stage"]["branch_eval"] in SIDE_LABELS]
    primary, secondary = _verdict(side_analyses)

    m0, m0_anchor = _acquire_m0_snapshot(int(anchor["frame"]) + 1)
    m0_anchor["corridor_forward"] = m0_anchor.get("corridor_forward", np.array([0.0, 1.0]))
    m0_probe = _probe(m0, m0_anchor, "M0_STRAIGHT", 0.0, 0.0, "EXP031_M0")
    deterministic = _deterministic_replay(runner, anchor, probes)
    public_equivalence = all(analysis["stage"]["public_detector_equivalent"] for analysis in analyses)
    full_mouth_equivalence = all(
        analysis["stage"]["public_detector_equivalent"]
        for analysis in analyses
        if "FULL_MOUTH" in analysis["stage"]["viewpoint_role"]
    )
    forward = next(analysis["stage"] for analysis in analyses if analysis["stage"]["branch_eval"] == "AXIAL_FORWARD")
    side_stages = [analysis["stage"] for analysis in side_analyses]
    first_divergence = next(
        (
            stage_name
            for stage_name, forward_value, side_values in (
                ("RANGE_EVIDENCE", forward["range_evidence_present"], [row["range_evidence_present"] for row in side_stages]),
                ("CANDIDATE_BEAMS", forward["candidate_beam_count"] > 0, [row["candidate_beam_count"] > 0 for row in side_stages]),
                ("AXIS_GROUP", forward["group_pass"], [row["group_pass"] for row in side_stages]),
                ("FINAL_OPENING", forward["final_opening_detected"], [row["final_opening_detected"] for row in side_stages]),
            )
            if forward_value and not all(side_values)
        ),
        "NO_DIVERGENCE",
    )
    verdict_row = {
        "experiment_id": EXPERIMENT_ID,
        "primary_verdict": primary,
        "secondary_findings": json.dumps(secondary),
        "first_forward_side_divergence_stage": first_divergence,
        "representative_unique_pose_count": len(coordinate_cache),
        "side_stage_rows": len(side_stages),
        "detector_wall_support_stage_present": False,
        "detector_tangent_stage_present": False,
        "A0_public_detector_equivalence": public_equivalence,
        "full_mouth_public_detector_equivalence": full_mouth_equivalence,
        "M0_opening_count": m0_probe["opening_count"],
        "M0_false_opening_count_eval": m0_probe["false_opening_count_eval"],
        "deterministic_replay": deterministic,
        "production_detector_modified": False,
        "GT_used_for_detector_candidate_or_grouping": False,
        "GT_map_used_for_ROI_and_posthoc_evaluation_only": True,
    }

    _write_csv(args.output / "representative_viewpoints.csv", _representative_rows(selected, probes, analyses))
    _write_csv(args.output / "ray_level_evidence.csv", [row for analysis in analyses for row in analysis["rays"]])
    _write_csv(args.output / "stage_summary.csv", [analysis["stage"] for analysis in analyses])
    _write_csv(args.output / "group_rejection_summary.csv", [analysis["group"] for analysis in analyses])
    _write_csv(args.output / "wall_tangent_diagnostics.csv", [analysis["wall"] for analysis in analyses])
    _write_csv(args.output / "positive_control_comparison.csv", _positive_comparison(analyses, m0_probe))
    _write_csv(args.output / "detector_threshold_audit.csv", _threshold_rows(analyses))
    _write_csv(args.output / "side_branch_pipeline_verdict.csv", [verdict_row])
    _plot_profiles(args.output / "range_profile_comparison.png", analyses)
    _plot_geometry(args.output / "side_branch_los_geometry.png", runner, probes)
    _plot_pipeline(args.output / "pipeline_stage_audit.png", analyses)
    print(json.dumps({
        "experiment_id": EXPERIMENT_ID,
        "primary_verdict": primary,
        "secondary_findings": secondary,
        "representatives": [(row["viewpoint_role"], row["candidate_id"]) for row in selected],
        "side_stages": [(row["viewpoint_role"], row["branch_eval"], row["mouth_visible_fraction_eval"], row["candidate_beam_count"], row["longest_candidate_run"], row["candidate_group_count"], row["group_pass"], row["final_opening_detected"], row["first_failure_stage"]) for row in side_stages],
        "forward_positive": (forward["candidate_beam_count"], forward["group_pass"], forward["final_opening_detected"]),
        "first_divergence": first_divergence,
        "M0_openings": m0_probe["opening_count"],
        "deterministic": deterministic,
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
