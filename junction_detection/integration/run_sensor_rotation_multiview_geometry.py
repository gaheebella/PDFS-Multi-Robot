"""EXP-029: separate GT line-of-sight from detector response under rotation.

The fixed A0 360-degree LiDAR profile is circularly rotated in exact integer
beam increments. No robot, Anchor, map, detector, or physics state is changed.
Map/GT data is consulted only after detector inference for line-of-sight truth,
branch matching, validity auditing, and evaluation-only multi-view unions.
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

from junction_detection.integration.run_local_asymmetric_viewpoint_geometry_diagnostic import (
    _acquire_m0_snapshot,
    _acquire_m1_anchor,
)
from junction_detection.integration.run_provisional_anchor_pointcloud_observation import (
    _gt_directions_eval_only,
    _normalize,
    evaluate_snapshot,
)
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import (
    detect_openings,
)

EXPERIMENT_ID = "EXP-029"
DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/sensor_rotation_multiview_geometry"
ORIENTATIONS = tuple(range(-90, 91, 15))
M0_ORIENTATIONS = (-90, -60, -30, 0, 30, 60, 90)
VIEW_SETS = (
    (0,),
    (0, 30),
    (0, -30),
    (0, 60),
    (0, -60),
    (60, -60),
    (0, 60, -60),
    (90, -90),
    (0, 90, -90),
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write heterogeneous dictionaries using the union of their fields."""
    if not rows:
        return
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _branch_label(runner: Any, branch_id: int | str) -> str:
    """Return evaluation labels without feeding them to detector inference."""
    if branch_id == "INCOMING":
        return "INCOMING"
    angle = float(runner.geometry.branches[int(branch_id)].angle_deg)
    if math.isclose(angle, 0.0):
        return "AXIAL_FORWARD"
    if math.isclose(angle, 90.0):
        return "PLUS90"
    if math.isclose(angle, -90.0):
        return "MINUS90"
    return f"OUTGOING_{int(branch_id)}"


def _rotate_snapshot(
    anchor: dict[str, Any], orientation_deg: int, context: str
) -> dict[str, Any]:
    """Circularly rotate a fixed 360-degree scan by exact integer beams."""
    angles = np.asarray(anchor["angles_deg"], dtype=float)
    ranges = np.asarray(anchor["ranges"], dtype=float)
    step = float(np.median(np.diff(angles)))
    beam_shift_float = float(orientation_deg) / step
    beam_shift = int(round(beam_shift_float))
    if not math.isclose(beam_shift_float, beam_shift, abs_tol=1.0e-12):
        raise ValueError("orientation must be an integer multiple of angular resolution")
    maximum = float(anchor["max_range"])
    margin = np.finfo(float).eps * max(1.0, maximum) * 64.0
    rotated = np.roll(ranges, -beam_shift)
    return {
        "context": context,
        "angles": angles.copy(),
        "ranges": rotated.copy(),
        "hit": rotated < maximum - margin,
        "max_range": maximum,
        "position_eval": np.asarray(anchor["position_eval"], dtype=float).copy(),
        "yaw_eval": float(anchor["yaw_eval"]) + float(orientation_deg),
        "frame": anchor["frame"],
        "time": anchor["timestamp"],
        "orientation_deg": orientation_deg,
        "beam_shift": beam_shift,
        "orientation_valid": len(angles) > 0 and math.isclose(len(angles) * step, 360.0),
        "orientation_validity_reason": "valid_full_360_circular_scan",
    }


def _nearest_beam(angles: np.ndarray, target_deg: float) -> tuple[int, float]:
    """Find the circularly nearest beam and its angular mismatch."""
    differences = np.abs(((angles - target_deg + 180.0) % 360.0) - 180.0)
    index = int(np.argmin(differences))
    return index, float(differences[index])


def _gt_visibility_rows(
    runner: Any, snapshot: dict[str, Any]
) -> list[dict[str, Any]]:
    """Evaluate physical branch-axis LOS from the frozen ray-cast scan.

    A branch is GT-visible when its exact world direction has no wall return
    before sensor max range. This uses no learned or detector threshold.
    """
    outgoing, incoming = _gt_directions_eval_only(runner, snapshot)
    truths = [*outgoing, incoming]
    angles = np.asarray(snapshot["angles"], dtype=float)
    ranges = np.asarray(snapshot["ranges"], dtype=float)
    maximum = float(snapshot["max_range"])
    margin = np.finfo(float).eps * max(1.0, maximum) * 64.0
    rows = []
    for truth in truths:
        branch_id = truth["branch_id"]
        local_angle = float(truth["local_angle_deg"])
        index, mismatch = _nearest_beam(angles, local_angle)
        visible = bool(ranges[index] >= maximum - margin)
        rows.append(
            {
                "orientation_deg": snapshot["orientation_deg"],
                "branch_id_eval": branch_id,
                "branch_label_eval": _branch_label(runner, branch_id),
                "is_outgoing_eval": branch_id != "INCOMING",
                "branch_local_angle_deg_eval": local_angle,
                "gt_visible_eval": visible,
                "gt_visibility_reason_eval": (
                    "clear_line_of_sight" if visible else "wall_occluded_before_max_range"
                ),
                "axis_ray_range_eval": float(ranges[index]),
                "axis_ray_max_range_eval": maximum,
                "nearest_beam_error_deg_eval": mismatch,
                "orientation_valid": snapshot["orientation_valid"],
                "orientation_validity_reason": snapshot["orientation_validity_reason"],
            }
        )
    return rows


def _detector_branch_rows(
    runner: Any,
    snapshot: dict[str, Any],
    opening_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Describe detector matches for every GT branch and unmatched opening."""
    outgoing, incoming = _gt_directions_eval_only(runner, snapshot)
    truths = [*outgoing, incoming]
    results = []
    for truth in truths:
        branch_id = truth["branch_id"]
        match = next(
            (
                row
                for row in opening_rows
                if row["matched_GT_branch_eval_only"] == branch_id
            ),
            None,
        )
        results.append(
            {
                "orientation_deg": snapshot["orientation_deg"],
                "branch_id_eval": branch_id,
                "branch_label_eval": _branch_label(runner, branch_id),
                "is_outgoing_eval": branch_id != "INCOMING",
                "detected_eval": match is not None,
                "opening_group_id": "" if match is None else match["opening_id"],
                "opening_center_deg": "" if match is None else match["center_angle_deg"],
                "opening_width_deg": "" if match is None else match["angular_width_deg"],
                "detector_confidence": "" if match is None else match["confidence"],
                "opening_group_count": len(opening_rows),
                "center_error_deg_eval": (
                    "" if match is None else match["center_error_deg_eval_only"]
                ),
                "orientation_valid": snapshot["orientation_valid"],
            }
        )
    for opening in opening_rows:
        if opening["matched_GT_branch_eval_only"] != "":
            continue
        results.append(
            {
                "orientation_deg": snapshot["orientation_deg"],
                "branch_id_eval": "UNMATCHED",
                "branch_label_eval": "FALSE_OPENING",
                "is_outgoing_eval": False,
                "detected_eval": True,
                "opening_group_id": opening["opening_id"],
                "opening_center_deg": opening["center_angle_deg"],
                "opening_width_deg": opening["angular_width_deg"],
                "detector_confidence": opening["confidence"],
                "opening_group_count": len(opening_rows),
                "center_error_deg_eval": "",
                "orientation_valid": snapshot["orientation_valid"],
            }
        )
    return results


def _classify(
    gt_rows: list[dict[str, Any]], detector_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Join truth/detection rows into the requested four-way taxonomy."""
    detector = {
        (row["orientation_deg"], str(row["branch_id_eval"])): row
        for row in detector_rows
        if row["branch_id_eval"] != "UNMATCHED"
    }
    results = []
    for truth in gt_rows:
        key = (truth["orientation_deg"], str(truth["branch_id_eval"]))
        detected = bool(detector[key]["detected_eval"])
        visible = bool(truth["gt_visible_eval"])
        if not visible and not detected:
            classification = "CASE_1_GEOMETRY_LIMITATION"
        elif visible and not detected:
            classification = "CASE_2_DETECTOR_MISS"
        elif visible and detected:
            classification = "CASE_3_CORRECT_DETECTION"
        else:
            classification = "CASE_4_FALSE_POSITIVE_OR_GT_MISMATCH"
        results.append(
            {
                **truth,
                "detected_eval": detected,
                "classification": classification,
            }
        )
    return results


def _evaluate_orientation(
    runner: Any, anchor: dict[str, Any], orientation: int, case: str
) -> dict[str, Any]:
    """Run the frozen detector first, then perform all GT evaluation."""
    snapshot = _rotate_snapshot(anchor, orientation, f"{case}_{orientation:+d}DEG")
    openings = list(
        detect_openings(snapshot["angles"].copy(), snapshot["ranges"].copy())
    )
    summary, opening_rows = evaluate_snapshot(runner, snapshot, openings)
    gt_rows = _gt_visibility_rows(runner, snapshot)
    detector_rows = _detector_branch_rows(runner, snapshot, opening_rows)
    classified = _classify(gt_rows, detector_rows)
    gt_outgoing = {
        int(row["branch_id_eval"])
        for row in gt_rows
        if row["is_outgoing_eval"] and row["gt_visible_eval"]
    }
    detected_outgoing = {
        int(row["branch_id_eval"])
        for row in detector_rows
        if row["is_outgoing_eval"] and row["detected_eval"]
    }
    return {
        "case": case,
        "orientation_deg": orientation,
        "snapshot": snapshot,
        "openings": openings,
        "opening_rows": opening_rows,
        "gt_rows": gt_rows,
        "detector_rows": detector_rows,
        "classified_rows": classified,
        "gt_outgoing_ids_eval": gt_outgoing,
        "detected_outgoing_ids_eval": detected_outgoing,
        "gt_visible_outgoing_count_eval": len(gt_outgoing),
        "detected_outgoing_count_eval": len(detected_outgoing),
        "opening_group_count": len(openings),
        "false_opening_count_eval": int(summary["false_opening_count_eval_only"]),
        "incoming_detected_eval": bool(summary["incoming_opening_count_eval_only"]),
        "valid_lidar_hits": int(summary["valid_lidar_point_count"]),
        "max_range_count": int(summary["max_range_no_return_count"]),
        "orientation_valid": snapshot["orientation_valid"],
        "orientation_validity_reason": snapshot["orientation_validity_reason"],
    }


def _rotation_sweep_rows(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten per-orientation count metrics."""
    return [
        {
            "experiment_id": EXPERIMENT_ID,
            "case": item["case"],
            "orientation_deg": item["orientation_deg"],
            "orientation_valid": item["orientation_valid"],
            "orientation_validity_reason": item["orientation_validity_reason"],
            "gt_visible_outgoing_count_eval": item["gt_visible_outgoing_count_eval"],
            "detected_outgoing_count_eval": item["detected_outgoing_count_eval"],
            "opening_group_count": item["opening_group_count"],
            "false_opening_count_eval": item["false_opening_count_eval"],
            "incoming_detected_eval": item["incoming_detected_eval"],
            "valid_lidar_hits": item["valid_lidar_hits"],
            "max_range_count": item["max_range_count"],
        }
        for item in evaluations
    ]


def _multiview_union(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union post-hoc branch ID sets; this is not a runtime fusion algorithm."""
    by_orientation = {item["orientation_deg"]: item for item in evaluations}
    outgoing_total = len(next(iter(evaluations))["gt_outgoing_ids_eval"] | {
        int(row["branch_id_eval"])
        for row in next(iter(evaluations))["gt_rows"]
        if row["is_outgoing_eval"]
    })
    results = []
    for view_set in VIEW_SETS:
        selected = [by_orientation[value] for value in view_set]
        gt_union = set().union(*(item["gt_outgoing_ids_eval"] for item in selected))
        detector_union = set().union(
            *(item["detected_outgoing_ids_eval"] for item in selected)
        )
        results.append(
            {
                "view_set": json.dumps(view_set),
                "all_orientations_valid": all(item["orientation_valid"] for item in selected),
                "gt_union_count_eval": len(gt_union),
                "detector_union_count_eval": len(detector_union),
                "gt_union_branches_eval": json.dumps(sorted(gt_union)),
                "detector_union_branches_eval": json.dumps(sorted(detector_union)),
                "outgoing_total_eval": outgoing_total,
                "gt_full_union_eval": len(gt_union) == outgoing_total,
                "detector_full_union_eval": len(detector_union) == outgoing_total,
                "note": "GT/detector branch-set union only; no point-cloud fusion",
            }
        )
    return results


def _summary_rows(classified: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count the four GT-vs-detector cases per orientation and globally."""
    labels = (
        "CASE_1_GEOMETRY_LIMITATION",
        "CASE_2_DETECTOR_MISS",
        "CASE_3_CORRECT_DETECTION",
        "CASE_4_FALSE_POSITIVE_OR_GT_MISMATCH",
    )
    results = []
    orientations = sorted({int(row["orientation_deg"]) for row in classified})
    for scope in [*orientations, "ALL"]:
        selected = (
            classified
            if scope == "ALL"
            else [row for row in classified if row["orientation_deg"] == scope]
        )
        results.append(
            {
                "scope": scope,
                **{
                    label: sum(row["classification"] == label for row in selected)
                    for label in labels
                },
                "branch_truth_row_count": len(selected),
            }
        )
    return results


def _negative_control_rows(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Separate persistent axial artifacts from lateral false openings in M0."""
    results = []
    for item in evaluations:
        orientation = float(item["orientation_deg"])
        false_openings = [
            row
            for row in item["opening_rows"]
            if row["matched_GT_branch_eval_only"] == ""
        ]
        axial = 0
        lateral = 0
        for row in false_openings:
            corridor_angle = _normalize(float(row["center_angle_deg"]) + orientation)
            if 45.0 <= abs(corridor_angle) <= 135.0:
                lateral += 1
            else:
                axial += 1
        results.append(
            {
                "case": item["case"],
                "orientation_deg": item["orientation_deg"],
                "orientation_valid": item["orientation_valid"],
                "opening_group_count": item["opening_group_count"],
                "false_opening_count_eval": len(false_openings),
                "axial_false_opening_count_eval": axial,
                "lateral_false_opening_count_eval": lateral,
                "rotation_false_positive_regression_eval": False,
            }
        )
    baseline = next(row for row in results if row["orientation_deg"] == 0)
    for row in results:
        row["rotation_false_positive_regression_eval"] = (
            row["orientation_valid"]
            and row["false_opening_count_eval"] > baseline["false_opening_count_eval"]
        )
    return results


def _verdict(
    evaluations: list[dict[str, Any]], unions: list[dict[str, Any]]
) -> tuple[str, list[str]]:
    """Apply EXP-029 primary/secondary result ordering."""
    valid = [item for item in evaluations if item["orientation_valid"]]
    baseline = next(item for item in valid if item["orientation_deg"] == 0)
    best_detector = max(item["detected_outgoing_count_eval"] for item in valid)
    best_gt = max(item["gt_visible_outgoing_count_eval"] for item in valid)
    best_union = max(
        row["detector_union_count_eval"]
        for row in unions
        if row["all_orientations_valid"]
    )
    findings = []
    if best_detector == 3:
        primary = "E_FULL_SINGLE_VIEW_RECOVERY"
    elif best_detector > baseline["detected_outgoing_count_eval"]:
        primary = "A_ROTATION_GAIN_DETECTED"
    elif best_union > baseline["detected_outgoing_count_eval"]:
        primary = "D_MULTIVIEW_GAIN_ONLY"
    elif best_gt > baseline["gt_visible_outgoing_count_eval"]:
        primary = "B_GT_GAIN_BUT_DETECTOR_MISS"
    else:
        primary = "C_NO_GT_VISIBILITY_GAIN"
    if best_gt > baseline["gt_visible_outgoing_count_eval"]:
        findings.append("GT_VISIBILITY_GAIN_EXISTS")
    if best_detector > baseline["detected_outgoing_count_eval"]:
        findings.append("SINGLE_VIEW_DETECTOR_GAIN_EXISTS")
    if best_union > baseline["detected_outgoing_count_eval"]:
        findings.append("MULTIVIEW_DETECTOR_GAIN_EXISTS")
    if not findings:
        findings.append("ROTATION_EQUIVALENT_NO_GAIN")
    return primary, findings


def _first_angle(
    evaluations: list[dict[str, Any]], label: str, field: str
) -> int | str:
    """Return the first angle in the declared ascending diagnostic sweep."""
    for item in sorted(evaluations, key=lambda row: row["orientation_deg"]):
        rows = item["gt_rows"] if field == "gt_visible_eval" else item["detector_rows"]
        match = next(
            (row for row in rows if row["branch_label_eval"] == label), None
        )
        if match is not None and bool(match[field]):
            return int(item["orientation_deg"])
    return "NOT_VISIBLE"


def _plot(
    path: Path,
    evaluations: list[dict[str, Any]],
    unions: list[dict[str, Any]],
) -> None:
    """Render count curves, separate heatmaps, unions, and profiles."""
    ordered = sorted(evaluations, key=lambda row: row["orientation_deg"])
    orientations = [item["orientation_deg"] for item in ordered]
    labels = ("AXIAL_FORWARD", "PLUS90", "MINUS90")
    gt_matrix = np.array(
        [
            [
                int(
                    next(
                        row["gt_visible_eval"]
                        for row in item["gt_rows"]
                        if row["branch_label_eval"] == label
                    )
                )
                for label in labels
            ]
            for item in ordered
        ]
    ).T
    detector_matrix = np.array(
        [
            [
                int(
                    next(
                        row["detected_eval"]
                        for row in item["detector_rows"]
                        if row["branch_label_eval"] == label
                    )
                )
                for label in labels
            ]
            for item in ordered
        ]
    ).T
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    axes[0, 0].plot(
        orientations,
        [item["gt_visible_outgoing_count_eval"] for item in ordered],
        "o-",
        label="GT visible",
    )
    axes[0, 0].plot(
        orientations,
        [item["detected_outgoing_count_eval"] for item in ordered],
        "s--",
        label="detector",
    )
    axes[0, 0].set(
        title="Single-view outgoing count",
        xlabel="sensor rotation [deg]",
        ylabel="outgoing branches",
        yticks=(0, 1, 2, 3),
        ylim=(-0.1, 3.1),
    )
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)
    for axis, matrix, title in (
        (axes[0, 1], gt_matrix, "GT LOS visibility (eval only)"),
        (axes[1, 0], detector_matrix, "Detector branch matches"),
    ):
        axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="Blues")
        axis.set(
            title=title,
            xticks=np.arange(len(orientations)),
            xticklabels=orientations,
            yticks=np.arange(len(labels)),
            yticklabels=labels,
            xlabel="sensor rotation [deg]",
        )
        axis.tick_params(axis="x", rotation=45)
    view_labels = [row["view_set"].replace(" ", "") for row in unions]
    x = np.arange(len(unions))
    axes[1, 1].bar(
        x - 0.18,
        [row["gt_union_count_eval"] for row in unions],
        0.36,
        label="GT union",
    )
    axes[1, 1].bar(
        x + 0.18,
        [row["detector_union_count_eval"] for row in unions],
        0.36,
        label="detector union",
    )
    axes[1, 1].set(
        title="Evaluation-only multi-view branch-set unions",
        xticks=x,
        xticklabels=view_labels,
        ylabel="outgoing branches",
        yticks=(0, 1, 2, 3),
        ylim=(0, 3.2),
    )
    axes[1, 1].tick_params(axis="x", rotation=40)
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.25)
    for angle in (-90, 0, 90):
        item = next(row for row in ordered if row["orientation_deg"] == angle)
        axes[2, 0].plot(
            item["snapshot"]["angles"],
            item["snapshot"]["ranges"],
            label=f"{angle:+d}°",
        )
        for opening in item["openings"]:
            axes[2, 0].axvline(
                opening["center_angle"], color="black", alpha=0.12, linewidth=0.8
            )
    axes[2, 0].set(
        title="Representative rotated profiles and opening centers",
        xlabel="rotated sensor-local angle [deg]",
        ylabel="range",
    )
    axes[2, 0].legend()
    axes[2, 0].grid(alpha=0.25)
    axes[2, 1].axis("off")
    axes[2, 1].text(
        0.03,
        0.95,
        "Fixed A0, circular 360° scan rotation\n"
        "GT visibility = branch-axis ray reaches max range\n"
        "GT/map is evaluation only\n"
        "No movement, detector tuning, or runtime fusion",
        va="top",
        family="monospace",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _deterministic_replay(
    runner: Any,
    anchor: dict[str, Any],
    reference: list[dict[str, Any]],
) -> bool:
    """Require exact circular profiles and detector outputs on replay."""
    for original in reference:
        repeated = _evaluate_orientation(
            runner,
            anchor,
            int(original["orientation_deg"]),
            original["case"],
        )
        if not np.array_equal(
            original["snapshot"]["ranges"], repeated["snapshot"]["ranges"]
        ):
            return False
        keys = (
            "gt_visible_outgoing_count_eval",
            "detected_outgoing_count_eval",
            "opening_group_count",
            "false_opening_count_eval",
        )
        if any(original[key] != repeated[key] for key in keys):
            return False
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-anchor-frames", type=int, default=120)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    m1, anchor = _acquire_m1_anchor(args.max_anchor_frames)
    evaluations = [
        _evaluate_orientation(m1, anchor, angle, "M1_CROSS_BASELINE")
        for angle in ORIENTATIONS
    ]
    unions = _multiview_union(evaluations)
    classified = [row for item in evaluations for row in item["classified_rows"]]
    detector_rows = [row for item in evaluations for row in item["detector_rows"]]
    gt_rows = [row for item in evaluations for row in item["gt_rows"]]

    m0, m0_anchor = _acquire_m0_snapshot(int(anchor["frame"]) + 1)
    m0_evaluations = [
        _evaluate_orientation(m0, m0_anchor, angle, "M0_STRAIGHT")
        for angle in M0_ORIENTATIONS
    ]
    negative = _negative_control_rows(m0_evaluations)
    primary, secondary = _verdict(evaluations, unions)
    deterministic = _deterministic_replay(m1, anchor, evaluations) and _deterministic_replay(
        m0, m0_anchor, m0_evaluations
    )
    summary = _summary_rows(classified)
    overall = next(row for row in summary if row["scope"] == "ALL")
    valid = [item for item in evaluations if item["orientation_valid"]]
    baseline_view = next(item for item in valid if item["orientation_deg"] == 0)
    best_single_score = max(item["detected_outgoing_count_eval"] for item in valid)
    best_single = (
        baseline_view
        if best_single_score == baseline_view["detected_outgoing_count_eval"]
        else next(
            item
            for item in valid
            if item["detected_outgoing_count_eval"] == best_single_score
        )
    )
    baseline_union = next(row for row in unions if row["view_set"] == "[0]")
    best_union_score = max(row["detector_union_count_eval"] for row in unions)
    best_union = (
        baseline_union
        if best_union_score == baseline_union["detector_union_count_eval"]
        else next(
            row
            for row in unions
            if row["detector_union_count_eval"] == best_union_score
        )
    )
    verdict_row = {
        "experiment_id": EXPERIMENT_ID,
        "primary_verdict": primary,
        "secondary_findings": json.dumps(secondary),
        "baseline_GT_visible_outgoing_eval": next(
            item["gt_visible_outgoing_count_eval"]
            for item in valid
            if item["orientation_deg"] == 0
        ),
        "baseline_detector_outgoing_eval": next(
            item["detected_outgoing_count_eval"]
            for item in valid
            if item["orientation_deg"] == 0
        ),
        "best_single_orientation_deg": best_single["orientation_deg"],
        "best_single_detector_outgoing_eval": best_single[
            "detected_outgoing_count_eval"
        ],
        "best_multiview_set": best_union["view_set"],
        "best_multiview_detector_union_eval": best_union[
            "detector_union_count_eval"
        ],
        "plus90_first_GT_visible_angle_eval": _first_angle(
            evaluations, "PLUS90", "gt_visible_eval"
        ),
        "plus90_first_detected_angle_eval": _first_angle(
            evaluations, "PLUS90", "detected_eval"
        ),
        "minus90_first_GT_visible_angle_eval": _first_angle(
            evaluations, "MINUS90", "gt_visible_eval"
        ),
        "minus90_first_detected_angle_eval": _first_angle(
            evaluations, "MINUS90", "detected_eval"
        ),
        "GT_visible_detector_miss_count": overall["CASE_2_DETECTOR_MISS"],
        "GT_invisible_detector_positive_count": overall[
            "CASE_4_FALSE_POSITIVE_OR_GT_MISMATCH"
        ],
        "unmatched_false_opening_count": sum(
            item["false_opening_count_eval"] for item in evaluations
        ),
        "M0_false_positive_regression": any(
            row["rotation_false_positive_regression_eval"] for row in negative
        ),
        "M0_lateral_false_opening": any(
            row["lateral_false_opening_count_eval"] > 0 for row in negative
        ),
        "deterministic_replay": deterministic,
        "actual_swarm_movement_performed": False,
        "detector_threshold_changed": False,
        "GT_used_for_orientation_selection": False,
        "GT_map_used_for_posthoc_evaluation_only": True,
    }

    _write_csv(args.output / "rotation_sweep.csv", _rotation_sweep_rows(evaluations))
    _write_csv(args.output / "rotation_branch_visibility.csv", gt_rows)
    _write_csv(args.output / "rotation_detector_results.csv", detector_rows)
    _write_csv(args.output / "rotation_gt_vs_detector.csv", classified)
    _write_csv(args.output / "rotation_multiview_union.csv", unions)
    _write_csv(args.output / "rotation_negative_control.csv", negative)
    _write_csv(args.output / "rotation_summary.csv", summary)
    _write_csv(args.output / "rotation_verdict.csv", [verdict_row])
    _plot(args.output / "rotation_audit.png", evaluations, unions)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "primary_verdict": primary,
                "secondary_findings": secondary,
                "orientation_counts": [
                    (
                        item["orientation_deg"],
                        item["gt_visible_outgoing_count_eval"],
                        item["detected_outgoing_count_eval"],
                    )
                    for item in evaluations
                ],
                "plus90_first_GT": verdict_row["plus90_first_GT_visible_angle_eval"],
                "plus90_first_detected": verdict_row[
                    "plus90_first_detected_angle_eval"
                ],
                "minus90_first_GT": verdict_row[
                    "minus90_first_GT_visible_angle_eval"
                ],
                "minus90_first_detected": verdict_row[
                    "minus90_first_detected_angle_eval"
                ],
                "case_counts": overall,
                "best_multiview_detector_union": best_union[
                    "detector_union_count_eval"
                ],
                "M0_false_regression": verdict_row["M0_false_positive_regression"],
                "deterministic": deterministic,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
