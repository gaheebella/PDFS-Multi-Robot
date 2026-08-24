"""Find non-forward ghost-viewpoint sensing boundaries without swarm motion.

Candidate directions and magnitudes use only the fixed Anchor's stable local
corridor frame and estimated width. Simulator geometry is used only to ray
cast the virtual sensor, audit candidate validity, and score detector output
afterward. This is an evaluation diagnostic, not a movement or fusion policy.
"""

from __future__ import annotations

import argparse
import csv
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

from junction_detection.integration.pointcloud_temporal_opening_persistence import (
    circular_interval_iou,
)
from junction_detection.integration.run_active_viewpoint_acquisition import (
    _gt_mouth_interval_eval_only,
)
from junction_detection.integration.run_local_asymmetric_viewpoint_geometry_diagnostic import (
    SAFE_FORWARD,
    UNSAFE_FORWARD,
    _acquire_m0_snapshot,
    _acquire_m1_anchor,
    local_visibility_features,
)
from junction_detection.integration.run_provisional_anchor_pointcloud_observation import (
    _gt_directions_eval_only,
    _normalize,
    evaluate_snapshot,
)
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import (
    detect_openings,
)

DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/nonforward_viewpoint_magnitude_boundary"
DIRECTIONS = {
    "L": (0.0, 1.0),
    "R": (0.0, -1.0),
    "FL": (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
    "FR": (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)),
}
COARSE_RATIOS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
M0_RATIOS = (0.10, 0.30, 0.50)
REFINEMENT_STEP = 0.025


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write heterogeneous records with a stable union of their fields."""
    if not rows:
        return
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _point_segment_distance(point: np.ndarray, wall: Any) -> float:
    """Return Euclidean distance from a ghost origin to one wall segment."""
    start = np.asarray(wall[0], dtype=float)
    end = np.asarray(wall[1], dtype=float)
    delta = end - start
    denominator = float(delta @ delta)
    if denominator <= np.finfo(float).eps:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(((point - start) @ delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * delta)))


def _candidate_pose(
    runner: Any,
    anchor: dict[str, Any],
    direction: str,
    ratio: float,
) -> tuple[np.ndarray, float, float, float]:
    """Create a width-normalized pose in the frozen Anchor-local frame."""
    width = float(anchor["estimated_corridor_width"])
    magnitude = ratio * width
    forward = np.asarray(anchor["corridor_forward"], dtype=float)
    forward /= np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0]])
    forward_scale, lateral_scale = DIRECTIONS[direction]
    forward_offset = magnitude * forward_scale
    lateral_offset = magnitude * lateral_scale
    position = (
        np.asarray(anchor["position_eval"], dtype=float)
        + forward * forward_offset
        + left * lateral_offset
    )
    return position, magnitude, forward_offset, lateral_offset


def _opening_evaluation(
    runner: Any,
    snapshot: dict[str, Any],
    opening_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float, float, set[int]]:
    """Add evaluation-only mouth IoU and return aggregate matched evidence."""
    results: list[dict[str, Any]] = []
    ious: list[float] = []
    errors: list[float] = []
    matched: set[int] = set()
    for row in opening_rows:
        item = dict(row)
        branch = row["matched_GT_branch_eval_only"]
        if isinstance(branch, int):
            matched.add(branch)
            gt_interval = _gt_mouth_interval_eval_only(runner, snapshot, branch)
            detected = {
                "start_angle": float(row["start_angle_deg"]),
                "end_angle": float(row["end_angle_deg"]),
            }
            iou = circular_interval_iou(detected, gt_interval)
            item["GT_mouth_IoU_eval_only"] = iou
            ious.append(iou)
            errors.append(float(row["center_error_deg_eval_only"]))
        else:
            item["GT_mouth_IoU_eval_only"] = ""
        results.append(item)
    return (
        results,
        float(np.mean(errors)) if errors else math.nan,
        float(np.mean(ious)) if ious else math.nan,
        matched,
    )


def _probe(
    runner: Any,
    anchor: dict[str, Any],
    case: str,
    direction: str,
    ratio: float,
    baseline_outgoing: int,
) -> dict[str, Any]:
    """Ray cast and score one virtual viewpoint with frozen sensor/detector."""
    position, magnitude, forward_offset, lateral_offset = _candidate_pose(
        runner, anchor, direction, ratio
    )
    clearance = min(_point_segment_distance(position, wall) for wall in runner.geometry.walls)
    inside = bool(runner.geometry.contains(position))
    walkable = bool(runner.geometry.walkable(position))
    valid = inside and walkable and clearance > 1.0e-9

    scan = runner.world.sensor.scan(runner.geometry, position, float(anchor["yaw_eval"]))
    margin = np.finfo(float).eps * max(1.0, scan.max_range) * 64.0
    openings = list(detect_openings(scan.angles_deg.copy(), scan.ranges.copy()))
    snapshot = {
        "context": f"{direction}_{ratio:.3f}W",
        "angles": scan.angles_deg.copy(),
        "ranges": scan.ranges.copy(),
        "hit": scan.ranges < scan.max_range - margin,
        "max_range": scan.max_range,
        "position_eval": position.copy(),
        "yaw_eval": float(anchor["yaw_eval"]),
        "frame": anchor["frame"],
        "time": anchor["timestamp"],
    }
    summary, opening_rows = evaluate_snapshot(runner, snapshot, openings)
    evaluated, mean_error, mean_iou, matched = _opening_evaluation(
        runner, snapshot, opening_rows
    )
    outgoing, _ = _gt_directions_eval_only(runner, snapshot)
    plus_id = next(
        (
            item["branch_id"]
            for item in outgoing
            if math.isclose(
                float(runner.geometry.branches[item["branch_id"]].angle_deg), 90.0
            )
        ),
        None,
    )
    minus_id = next(
        (
            item["branch_id"]
            for item in outgoing
            if math.isclose(
                float(runner.geometry.branches[item["branch_id"]].angle_deg), -90.0
            )
        ),
        None,
    )
    plus_visible = plus_id in matched if plus_id is not None else False
    minus_visible = minus_id in matched if minus_id is not None else False
    return {
        "case": case,
        "direction": direction,
        "magnitude_ratio": round(float(ratio), 6),
        "magnitude_absolute": magnitude,
        "forward_offset": forward_offset,
        "lateral_offset": lateral_offset,
        "candidate_valid": valid,
        "candidate_inside_free_space_eval": inside,
        "candidate_walkable_eval": walkable,
        "wall_clearance_eval": clearance,
        "opening_count": int(summary["opening_count"]),
        "outgoing_match_count_eval": int(summary["matched_outgoing_count_eval_only"]),
        "outgoing_total_eval": int(summary["GT_outgoing_branch_count_eval_only"]),
        "plus90_visible_eval": plus_visible,
        "minus90_visible_eval": minus_visible,
        "new_side_branch_count_eval": int(plus_visible) + int(minus_visible),
        "incoming_match_eval": int(summary["incoming_opening_count_eval_only"]),
        "false_opening_count_eval": int(summary["false_opening_count_eval_only"]),
        "valid_lidar_hits": int(summary["valid_lidar_point_count"]),
        "max_range_count": int(summary["max_range_no_return_count"]),
        "wall_support_count": int(summary["total_fitted_wall_point_count"]),
        "tangent_support_count": int(summary["wall_tangent_available_count"]),
        "opening_center_error_eval": mean_error,
        "opening_IoU_eval": mean_iou,
        "visibility_gain_vs_A0_eval": int(summary["matched_outgoing_count_eval_only"])
        - baseline_outgoing,
        "matched_branch_ids_eval": matched,
        "snapshot": snapshot,
        "openings_eval": evaluated,
    }


def _baseline_probe(runner: Any, anchor: dict[str, Any], case: str) -> dict[str, Any]:
    """Evaluate A0 once without treating it as a displacement candidate."""
    position = np.asarray(anchor["position_eval"], dtype=float)
    scan = runner.world.sensor.scan(runner.geometry, position, float(anchor["yaw_eval"]))
    margin = np.finfo(float).eps * max(1.0, scan.max_range) * 64.0
    openings = list(detect_openings(scan.angles_deg.copy(), scan.ranges.copy()))
    snapshot = {
        "context": "A0",
        "angles": scan.angles_deg.copy(),
        "ranges": scan.ranges.copy(),
        "hit": scan.ranges < scan.max_range - margin,
        "max_range": scan.max_range,
        "position_eval": position,
        "yaw_eval": float(anchor["yaw_eval"]),
        "frame": anchor["frame"],
        "time": anchor["timestamp"],
    }
    summary, opening_rows = evaluate_snapshot(runner, snapshot, openings)
    _, _, _, matched = _opening_evaluation(runner, snapshot, opening_rows)
    return {
        "outgoing_match_count_eval": int(summary["matched_outgoing_count_eval_only"]),
        "false_opening_count_eval": int(summary["false_opening_count_eval_only"]),
        "matched_branch_ids_eval": matched,
        "ranges": scan.ranges.copy(),
    }


def _ratios_for_direction(
    runner: Any,
    anchor: dict[str, Any],
    case: str,
    direction: str,
    baseline: int,
) -> list[dict[str, Any]]:
    """Run bounded coarse probes and refine only a valid first transition."""
    rows = [_probe(runner, anchor, case, direction, ratio, baseline) for ratio in COARSE_RATIOS]
    valid_rows = [row for row in rows if row["candidate_valid"]]
    has_gain = any(row["outgoing_match_count_eval"] > baseline for row in valid_rows)
    if not has_gain and rows[-1]["candidate_valid"]:
        rows.append(_probe(runner, anchor, case, direction, 0.70, baseline))

    valid_sorted = sorted(
        (row for row in rows if row["candidate_valid"]),
        key=lambda row: row["magnitude_ratio"],
    )
    first_gain = next(
        (row for row in valid_sorted if row["outgoing_match_count_eval"] > baseline),
        None,
    )
    if first_gain is not None:
        upper = float(first_gain["magnitude_ratio"])
        lower = max(
            (
                float(row["magnitude_ratio"])
                for row in valid_sorted
                if row["magnitude_ratio"] < upper
                and row["outgoing_match_count_eval"] <= baseline
            ),
            default=0.0,
        )
        ratio = lower + REFINEMENT_STEP
        while ratio < upper - 1.0e-9:
            rows.append(_probe(runner, anchor, case, direction, ratio, baseline))
            ratio += REFINEMENT_STEP
    return sorted(rows, key=lambda row: row["magnitude_ratio"])


def _boundary_summary(
    rows: list[dict[str, Any]], baseline: int, width: float
) -> list[dict[str, Any]]:
    """Summarize the first valid sensing transition for every direction."""
    results = []
    for direction in DIRECTIONS:
        selected = sorted(
            (row for row in rows if row["direction"] == direction),
            key=lambda row: row["magnitude_ratio"],
        )
        valid = [row for row in selected if row["candidate_valid"]]
        first_gain = next(
            (row for row in valid if row["outgoing_match_count_eval"] > baseline), None
        )
        first_plus = next((row for row in valid if row["plus90_visible_eval"]), None)
        first_minus = next((row for row in valid if row["minus90_visible_eval"]), None)
        maximum = max((row["outgoing_match_count_eval"] for row in valid), default=baseline)
        best = (
            next(
                (
                    row
                    for row in valid
                    if row["outgoing_match_count_eval"] == maximum
                ),
                None,
            )
            if maximum > baseline
            else None
        )
        results.append(
            {
                "direction": direction,
                "first_gain_ratio": "" if first_gain is None else first_gain["magnitude_ratio"],
                "first_gain_absolute": "" if first_gain is None else float(first_gain["magnitude_ratio"]) * width,
                "first_plus90_ratio_eval": "" if first_plus is None else first_plus["magnitude_ratio"],
                "first_minus90_ratio_eval": "" if first_minus is None else first_minus["magnitude_ratio"],
                "max_outgoing_match_eval": maximum,
                "best_ratio": "" if best is None else best["magnitude_ratio"],
                "single_view_3of3_possible_eval": any(
                    row["outgoing_match_count_eval"] == 3 for row in valid
                ),
                "tested_max_ratio": max((row["magnitude_ratio"] for row in selected), default=0.0),
                "valid_max_ratio": max((row["magnitude_ratio"] for row in valid), default=0.0),
                "status": (
                    "GAIN_FOUND"
                    if first_gain is not None
                    else "NO_GAIN_WITHIN_VALID_TESTED_RANGE"
                ),
            }
        )
    return results


def _multiview_rows(
    rows: list[dict[str, Any]], baseline_ids: set[int], outgoing_total: int
) -> list[dict[str, Any]]:
    """Compute post-hoc unions of per-scan GT matches; this is not fusion."""
    pairs = (("A0", "L"), ("A0", "R"), ("L", "R"), ("FL", "FR"))
    by_key = {
        (row["direction"], float(row["magnitude_ratio"])): row for row in rows
    }
    ratios = sorted({float(row["magnitude_ratio"]) for row in rows})
    results = []
    for ratio in ratios:
        for first, second in pairs:
            first_row = None if first == "A0" else by_key.get((first, ratio))
            second_row = None if second == "A0" else by_key.get((second, ratio))
            if (first != "A0" and first_row is None) or (
                second != "A0" and second_row is None
            ):
                continue
            first_ids = baseline_ids if first == "A0" else first_row["matched_branch_ids_eval"]
            second_ids = baseline_ids if second == "A0" else second_row["matched_branch_ids_eval"]
            valid = (first == "A0" or first_row["candidate_valid"]) and (
                second == "A0" or second_row["candidate_valid"]
            )
            union = set(first_ids) | set(second_ids)
            results.append(
                {
                    "magnitude_ratio": ratio,
                    "view_union_eval": f"{first}+{second}",
                    "both_candidates_valid_eval": valid,
                    "matched_branch_ids_union_eval": json.dumps(sorted(union)),
                    "outgoing_union_count_eval": len(union),
                    "outgoing_total_eval": outgoing_total,
                    "union_3of3_eval": len(union) == outgoing_total,
                    "note": "GT-match set union only; not runtime point-cloud fusion",
                }
            )
    return results


def _opening_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten candidate openings and keep GT-dependent fields explicit."""
    result = []
    for candidate in rows:
        for opening in candidate["openings_eval"]:
            match = opening["matched_GT_branch_eval_only"]
            result.append(
                {
                    "case": candidate["case"],
                    "direction": candidate["direction"],
                    "magnitude_ratio": candidate["magnitude_ratio"],
                    "candidate_valid": candidate["candidate_valid"],
                    "opening_id": opening["opening_id"],
                    "start_angle_deg": opening["start_angle_deg"],
                    "end_angle_deg": opening["end_angle_deg"],
                    "center_angle_deg": opening["center_angle_deg"],
                    "angular_width_deg": opening["angular_width_deg"],
                    "confidence": opening["confidence"],
                    "matched_GT_branch_eval_only": match,
                    "center_error_deg_eval_only": opening["center_error_deg_eval_only"],
                    "GT_mouth_IoU_eval_only": opening["GT_mouth_IoU_eval_only"],
                    "wall_support": opening["fitted_wall_point_count"],
                    "tangent_support": opening["wall_tangent_deg"] != "",
                }
            )
    return result


def _public_sweep_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove arrays and internal evaluation sets from the sweep CSV."""
    excluded = {"matched_branch_ids_eval", "snapshot", "openings_eval"}
    return [{key: value for key, value in row.items() if key not in excluded} for row in rows]


def _forward_reference() -> list[dict[str, float | str]]:
    """Load prior safe/unsafe forward results without rerunning them."""
    result = []
    for source, path in (("SAFE_FORWARD", SAFE_FORWARD), ("UNSAFE_FORWARD", UNSAFE_FORWARD)):
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                result.append(
                    {
                        "source": source,
                        "ratio": float(
                            row.get(
                                "cumulative_width_ratio",
                                row.get("cumulative_advance_width_ratio", 0.0),
                            )
                        ),
                        "outgoing": float(
                            row.get(
                                "matched_outgoing_eval_only",
                                row.get("matched_outgoing_count_eval_only", 0.0),
                            )
                        ),
                    }
                )
    return result


def _verdict(rows: list[dict[str, Any]], baseline: int) -> str:
    """Classify the bounded valid single-view sensing result."""
    valid = [row for row in rows if row["candidate_valid"]]
    if any(
        row["direction"] in ("FL", "FR")
        and row["outgoing_match_count_eval"] == 3
        for row in valid
    ):
        return "B_DIAGONAL_SINGLE_VIEW_3OF3"
    left_plus = any(row["direction"] == "L" and row["plus90_visible_eval"] for row in valid)
    right_minus = any(row["direction"] == "R" and row["minus90_visible_eval"] for row in valid)
    if left_plus and right_minus and not any(
        row["outgoing_match_count_eval"] == 3 for row in valid
    ):
        return "A_OPPOSITE_SIDES_REQUIRE_TWO_SIDED_VIEWS"
    maximum = max((row["outgoing_match_count_eval"] for row in valid), default=baseline)
    if maximum == 2:
        return "C_SINGLE_TRANSLATION_MAXIMUM_2OF3"
    return "D_NO_TRANSLATION_GAIN_WITHIN_VALID_TESTED_RANGE"


def _plot(
    path: Path,
    rows: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    forward: list[dict[str, Any]],
) -> None:
    """Plot valid visibility curves, geometry clearance, and boundaries."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for direction in DIRECTIONS:
        selected = sorted(
            (row for row in rows if row["direction"] == direction),
            key=lambda row: row["magnitude_ratio"],
        )
        valid = [row for row in selected if row["candidate_valid"]]
        invalid = [row for row in selected if not row["candidate_valid"]]
        axes[0, 0].plot(
            [row["magnitude_ratio"] for row in valid],
            [row["outgoing_match_count_eval"] for row in valid],
            "o-",
            label=direction,
        )
        if invalid:
            axes[0, 0].scatter(
                [row["magnitude_ratio"] for row in invalid],
                [row["outgoing_match_count_eval"] for row in invalid],
                marker="x",
                color="black",
                s=70,
                linewidths=1.8,
                zorder=10,
            )
        axes[0, 1].plot(
            [row["magnitude_ratio"] for row in selected],
            [row["wall_clearance_eval"] for row in selected],
            "o-",
            label=direction,
        )
        axes[1, 0].plot(
            [row["magnitude_ratio"] for row in valid],
            [int(row["plus90_visible_eval"]) - int(row["minus90_visible_eval"]) for row in valid],
            "o-",
            label=direction,
        )
    for source in ("SAFE_FORWARD", "UNSAFE_FORWARD"):
        selected = [row for row in forward if row["source"] == source]
        axes[0, 0].plot(
            [row["ratio"] for row in selected],
            [row["outgoing"] for row in selected],
            "--",
            alpha=0.65,
            label=source,
        )
    axes[0, 0].set(
        title="Single-view visibility boundary (x = invalid geometry)",
        xlabel="displacement / estimated width",
        ylabel="outgoing GT match (eval only)",
        yticks=(0, 1, 2, 3),
        ylim=(-0.15, 3.15),
    )
    axes[0, 0].legend(ncol=2)
    axes[0, 0].grid(alpha=0.25)
    axes[0, 1].set(
        title="Ghost-origin wall clearance (eval only)",
        xlabel="displacement / estimated width",
        ylabel="minimum wall clearance",
    )
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set(
        title="Side visibility: +1 = +90, -1 = -90",
        xlabel="displacement / estimated width",
        ylabel="side visibility indicator",
        yticks=(-1, 0, 1),
    )
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)
    axes[1, 1].axis("off")
    text = ["Direction boundaries (valid candidates only)"]
    for row in boundaries:
        text.append(
            f"{row['direction']}: first={row['first_gain_ratio'] or 'none'}, "
            f"max={row['max_outgoing_match_eval']}/3, valid<= {row['valid_max_ratio']}W"
        )
    axes[1, 1].text(0.03, 0.95, "\n".join(text), va="top", family="monospace")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _deterministic_replay(
    runner: Any,
    anchor: dict[str, Any],
    reference: list[dict[str, Any]],
    baseline: int,
) -> bool:
    """Replay every selected ghost scan and require exact range/results."""
    for row in reference:
        repeated = _probe(
            runner,
            anchor,
            row["case"],
            row["direction"],
            float(row["magnitude_ratio"]),
            baseline,
        )
        if not np.array_equal(row["snapshot"]["ranges"], repeated["snapshot"]["ranges"]):
            return False
        keys = (
            "candidate_valid",
            "opening_count",
            "outgoing_match_count_eval",
            "plus90_visible_eval",
            "minus90_visible_eval",
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
    width = float(anchor["estimated_corridor_width"])
    local_features = local_visibility_features(anchor, "M1_CROSS_BASELINE")
    baseline = _baseline_probe(m1, anchor, "M1_CROSS_BASELINE")
    m1_rows: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        m1_rows.extend(
            _ratios_for_direction(
                m1,
                anchor,
                "M1_CROSS_BASELINE",
                direction,
                baseline["outgoing_match_count_eval"],
            )
        )

    m0, m0_anchor = _acquire_m0_snapshot(int(anchor["frame"]) + 1)
    m0_baseline = _baseline_probe(m0, m0_anchor, "M0_STRAIGHT")
    m0_rows = [
        _probe(
            m0,
            m0_anchor,
            "M0_STRAIGHT",
            direction,
            ratio,
            m0_baseline["outgoing_match_count_eval"],
        )
        for direction in DIRECTIONS
        for ratio in M0_RATIOS
    ]

    boundaries = _boundary_summary(
        m1_rows, baseline["outgoing_match_count_eval"], width
    )
    unions = _multiview_rows(
        m1_rows,
        baseline["matched_branch_ids_eval"],
        int(m1_rows[0]["outgoing_total_eval"]),
    )
    verdict = _verdict(m1_rows, baseline["outgoing_match_count_eval"])
    deterministic = _deterministic_replay(
        m1, anchor, m1_rows, baseline["outgoing_match_count_eval"]
    ) and _deterministic_replay(
        m0, m0_anchor, m0_rows, m0_baseline["outgoing_match_count_eval"]
    )
    m0_false_regression = any(
        row["candidate_valid"]
        and row["false_opening_count_eval"]
        > m0_baseline["false_opening_count_eval"]
        for row in m0_rows
    )
    m0_lateral_false = any(
        opening["matched_GT_branch_eval_only"] == ""
        and abs(float(opening["center_angle_deg"])) >= 45.0
        for row in m0_rows
        if row["candidate_valid"]
        for opening in row["openings_eval"]
    )
    forward = _forward_reference()
    max_valid = max(
        (
            row["outgoing_match_count_eval"]
            for row in m1_rows
            if row["candidate_valid"]
        ),
        default=baseline["outgoing_match_count_eval"],
    )
    verdict_row = {
        "verdict": verdict,
        "A0_outgoing_match_eval": baseline["outgoing_match_count_eval"],
        "max_valid_single_view_outgoing_match_eval": max_valid,
        "single_view_3of3_possible_eval": max_valid == 3,
        "any_multiview_union_3of3_eval": any(
            row["both_candidates_valid_eval"] and row["union_3of3_eval"]
            for row in unions
        ),
        "M0_existing_axis_false_opening_count_eval": m0_baseline[
            "false_opening_count_eval"
        ],
        "M0_false_opening_regression_eval": m0_false_regression,
        "M0_lateral_false_opening_eval": m0_lateral_false,
        "deterministic_replay": deterministic,
        "actual_swarm_movement_performed": False,
        "detector_thresholds_changed": False,
        "GT_used_for_candidate_direction": False,
        "GT_map_used_for_validity_and_posthoc_evaluation_only": True,
        "A0_left_right_asymmetry": local_features["left_right_asymmetry"],
        "A0_local_preferred_side_diagnostic_only": local_features[
            "local_preferred_side_by_mean_range"
        ],
    }

    _write_csv(
        args.output / "viewpoint_magnitude_sweep.csv",
        _public_sweep_rows(m1_rows + m0_rows),
    )
    _write_csv(args.output / "viewpoint_boundary_summary.csv", boundaries)
    _write_csv(args.output / "viewpoint_openings.csv", _opening_rows(m1_rows + m0_rows))
    _write_csv(args.output / "multiview_union_eval.csv", unions)
    _write_csv(args.output / "viewpoint_magnitude_verdict.csv", [verdict_row])
    _plot(
        args.output / "viewpoint_magnitude_audit.png",
        m1_rows,
        boundaries,
        forward,
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "width": width,
                "A0_outgoing": baseline["outgoing_match_count_eval"],
                "boundaries": boundaries,
                "M0_false_regression": m0_false_regression,
                "M0_lateral_false": m0_lateral_false,
                "deterministic": deterministic,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
