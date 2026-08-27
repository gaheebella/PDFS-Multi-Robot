"""Compare threshold and geometric visibility at M1 frames 120 and 126.

Detector arrays and decisions are consumed unchanged. Map geometry is consulted
only after each detector call to append explicitly evaluation-only line-of-sight
labels; those labels never enter the detector or simulation controller.
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

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.pointcloud.lidar_profile_junction_detector import (  # noqa: E402
    GeometryProfileConfig,
    LidarProfileJunctionDetector,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import (  # noqa: E402
    LIDAR_MAX_RANGE,
    SimulationRunner,
)


MAP_CASE = "M1_CROSS_BASELINE"
FRAMES = (120, 126)
DEFAULT_OUTPUT = (
    ROOT
    / "junction_detection/integration/output/opening_threshold_geometry_diagnostic"
)
BEAM_FIELDS = (
    "frame",
    "time",
    "theta_deg",
    "measured_range",
    "expected_range",
    "margin_boundary",
    "delta_range",
    "open_candidate",
    "opening_group_id",
    "in_frame126_opening_sector_eval_only",
    "valid_expected_wall_angle",
    "physical_wall_hit_eval_only",
    "open_space_to_sensor_limit_eval_only",
    "gt_side_opening_los_eval_only",
)
SUMMARY_FIELDS = (
    "frame",
    "time",
    "opening_sector_beam_count",
    "sector_open_candidate_count",
    "opening_group_count",
    "sector_margin_blocked_count",
    "sector_margin_blocked_with_side_los_count",
    "sector_measured_le_expected_count",
    "sector_measured_gt_expected_count",
    "sector_physical_wall_hit_count",
    "sector_open_space_to_sensor_limit_count",
    "sector_gt_side_opening_los_count",
    "sector_expected_model_inconsistent_with_los_count",
    "sector_delta_min",
    "sector_delta_mean",
    "sector_delta_max",
    "sector_expected_min",
    "sector_expected_max",
    "sector_expected_at_sensor_max_count",
    "side_walls_valid",
    "corridor_model_initialized",
    "width_observation",
    "estimated_corridor_width",
    "current_corridor_orientation_deg",
    "stable_corridor_orientation_deg",
    "junction_detected",
    "group_start_deg",
    "group_end_deg",
    "group_width_deg",
)


def _capture_frames() -> tuple[SimulationRunner, dict[int, dict[str, Any]]]:
    detector = LidarProfileJunctionDetector(GeometryProfileConfig(LIDAR_MAX_RANGE))
    runner = SimulationRunner(
        MAP_CASE,
        "local_forward",
        profile_detector=detector,
        hold_on_profile_detection=False,
    )
    snapshots: dict[int, dict[str, Any]] = {}
    for frame in range(max(FRAMES) + 1):
        row = runner.step(frame)
        if frame not in FRAMES:
            continue
        if row is None:
            raise RuntimeError(f"requested frame {frame} is not a sampled LiDAR frame")
        scan = runner.last_visual[0].lidar_scan
        result = runner.last_profile_result
        leader = next(
            robot for robot in runner.world.robots
            if robot.robot_id == runner.world.lidar_robot_id
        )
        snapshots[frame] = {
            "frame": frame,
            "time": float(row["timestamp"]),
            "angles": scan.angles_deg.copy(),
            "measured": scan.ranges.copy(),
            "expected": result["expected_ranges"].copy(),
            "delta": result["delta_ranges"].copy(),
            "candidate": result["open_candidate_mask"].copy(),
            "confirmed": result["confirmed_opening_mask"].copy(),
            "valid": result["valid_angle_mask"].copy(),
            "groups": tuple(dict(group) for group in result["opening_groups"]),
            "detected": bool(result["profile_junction_detected"]),
            "margin": float(result["profile_numerical_margin"]),
            "max_range": float(result["profile_max_range"]),
            "side_walls_valid": bool(result["side_walls_valid"]),
            "corridor_model_initialized": bool(result["corridor_model_initialized"]),
            "width_observation": float(result["width_observation"]),
            "estimated_corridor_width": float(result["estimated_corridor_width"]),
            "current_corridor_orientation_deg": float(result["current_corridor_orientation_deg"]),
            "stable_corridor_orientation_deg": float(result["stable_corridor_orientation_deg"]),
            "leader_position_eval_only": leader.position.copy(),
            "lidar_yaw_deg_eval_only": float(runner.world.lidar_yaw_deg),
        }
    return runner, snapshots


def _circular_difference_deg(first: np.ndarray, second: float) -> np.ndarray:
    return (first - second + 180.0) % 360.0 - 180.0


def _group_ids(snapshot: dict[str, Any]) -> np.ndarray:
    """Map confirmed beams to detector-returned group IDs without regrouping."""
    ids = np.full(len(snapshot["angles"]), -1, dtype=int)
    confirmed = snapshot["confirmed"]
    for group in snapshot["groups"]:
        distance = np.abs(
            _circular_difference_deg(
                snapshot["angles"], float(group["center_body_angle_deg"])
            )
        )
        belongs = confirmed & (distance <= 0.5 * float(group["angular_width_deg"]))
        ids[belongs] = int(group["group_id"])
    if np.any(confirmed & (ids < 0)):
        raise RuntimeError("a confirmed beam did not map to a detector-returned group")
    return ids


def _frame126_sector(angles: np.ndarray, groups: tuple[dict[str, Any], ...]) -> np.ndarray:
    """Use the detector-returned frame-126 intervals as the comparison sector."""
    sector = np.zeros(len(angles), dtype=bool)
    for group in groups:
        distance = np.abs(
            _circular_difference_deg(angles, float(group["center_body_angle_deg"]))
        )
        sector |= distance <= 0.5 * float(group["angular_width_deg"])
    return sector


def _side_opening_los_eval_only(
    runner: SimulationRunner,
    snapshot: dict[str, Any],
) -> np.ndarray:
    """Post-hoc ray/side-mouth intersection labels for M1 evaluation only."""
    geometry = runner.geometry
    half_junction = 0.5 * float(geometry.junction_size)
    side_mouths: list[tuple[float, float]] = []
    for branch in geometry.branches:
        direction_x = math.sin(math.radians(float(branch.angle_deg)))
        direction_y = math.cos(math.radians(float(branch.angle_deg)))
        if abs(direction_x) < 1.0 - 1.0e-12 or abs(direction_y) > 1.0e-12:
            continue
        side_mouths.append((math.copysign(half_junction, direction_x), 0.5 * float(branch.width)))

    origin = snapshot["leader_position_eval_only"]
    output = np.zeros(len(snapshot["angles"]), dtype=bool)
    for index, theta in enumerate(snapshot["angles"]):
        world_angle = math.radians(snapshot["lidar_yaw_deg_eval_only"] + float(theta))
        direction = np.array([math.cos(world_angle), math.sin(world_angle)])
        if abs(float(direction[0])) <= np.finfo(float).eps * 32.0:
            continue
        for mouth_x, half_width in side_mouths:
            distance = (mouth_x - float(origin[0])) / float(direction[0])
            if distance < 0.0 or distance > snapshot["max_range"]:
                continue
            crossing_y = float(origin[1]) + distance * float(direction[1])
            reaches_mouth_before_hit = distance <= float(snapshot["measured"][index]) + 1.0e-9
            if abs(crossing_y) <= half_width + 1.0e-9 and reaches_mouth_before_hit:
                output[index] = True
                break
    return output


def _beam_rows(
    runner: SimulationRunner,
    snapshots: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    sector = _frame126_sector(snapshots[126]["angles"], snapshots[126]["groups"])
    rows: list[dict[str, Any]] = []
    evaluations: dict[int, dict[str, Any]] = {}
    for frame in FRAMES:
        snapshot = snapshots[frame]
        group_ids = _group_ids(snapshot)
        wall_hit = snapshot["measured"] < snapshot["max_range"] - snapshot["margin"]
        side_los = _side_opening_los_eval_only(runner, snapshot)
        evaluations[frame] = {
            "sector": sector.copy(),
            "group_ids": group_ids,
            "wall_hit": wall_hit,
            "side_los": side_los,
        }
        boundary = snapshot["expected"] + snapshot["margin"]
        for index, theta in enumerate(snapshot["angles"]):
            rows.append(
                {
                    "frame": frame,
                    "time": snapshot["time"],
                    "theta_deg": float(theta),
                    "measured_range": float(snapshot["measured"][index]),
                    "expected_range": float(snapshot["expected"][index]),
                    "margin_boundary": float(boundary[index]),
                    "delta_range": float(snapshot["delta"][index]),
                    "open_candidate": bool(snapshot["candidate"][index]),
                    "opening_group_id": int(group_ids[index]),
                    "in_frame126_opening_sector_eval_only": bool(sector[index]),
                    "valid_expected_wall_angle": bool(snapshot["valid"][index]),
                    "physical_wall_hit_eval_only": bool(wall_hit[index]),
                    "open_space_to_sensor_limit_eval_only": bool(not wall_hit[index]),
                    "gt_side_opening_los_eval_only": bool(side_los[index]),
                }
            )
    return rows, evaluations


def _json_group_values(groups: tuple[dict[str, Any], ...], key: str) -> str:
    return json.dumps([float(group[key]) for group in groups], separators=(",", ":"))


def _summary_rows(
    snapshots: dict[int, dict[str, Any]],
    evaluations: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for frame in FRAMES:
        snapshot = snapshots[frame]
        evaluation = evaluations[frame]
        sector = evaluation["sector"]
        delta = snapshot["delta"]
        margin_blocked = sector & snapshot["valid"] & (delta > 0.0) & (delta <= snapshot["margin"])
        model_inconsistent = sector & evaluation["side_los"] & ~snapshot["valid"]
        rows.append(
            {
                "frame": frame,
                "time": snapshot["time"],
                "opening_sector_beam_count": int(np.count_nonzero(sector)),
                "sector_open_candidate_count": int(np.count_nonzero(sector & snapshot["candidate"])),
                "opening_group_count": len(snapshot["groups"]),
                "sector_margin_blocked_count": int(np.count_nonzero(margin_blocked)),
                "sector_margin_blocked_with_side_los_count": int(np.count_nonzero(margin_blocked & evaluation["side_los"])),
                "sector_measured_le_expected_count": int(np.count_nonzero(sector & (snapshot["measured"] <= snapshot["expected"]))),
                "sector_measured_gt_expected_count": int(np.count_nonzero(sector & (snapshot["measured"] > snapshot["expected"]))),
                "sector_physical_wall_hit_count": int(np.count_nonzero(sector & evaluation["wall_hit"])),
                "sector_open_space_to_sensor_limit_count": int(np.count_nonzero(sector & ~evaluation["wall_hit"])),
                "sector_gt_side_opening_los_count": int(np.count_nonzero(sector & evaluation["side_los"])),
                "sector_expected_model_inconsistent_with_los_count": int(np.count_nonzero(model_inconsistent)),
                "sector_delta_min": float(np.min(delta[sector])),
                "sector_delta_mean": float(np.mean(delta[sector])),
                "sector_delta_max": float(np.max(delta[sector])),
                "sector_expected_min": float(np.min(snapshot["expected"][sector])),
                "sector_expected_max": float(np.max(snapshot["expected"][sector])),
                "sector_expected_at_sensor_max_count": int(
                    np.count_nonzero(
                        sector
                        & (
                            snapshot["expected"]
                            >= snapshot["max_range"] - snapshot["margin"]
                        )
                    )
                ),
                "side_walls_valid": snapshot["side_walls_valid"],
                "corridor_model_initialized": snapshot["corridor_model_initialized"],
                "width_observation": snapshot["width_observation"],
                "estimated_corridor_width": snapshot["estimated_corridor_width"],
                "current_corridor_orientation_deg": snapshot["current_corridor_orientation_deg"],
                "stable_corridor_orientation_deg": snapshot["stable_corridor_orientation_deg"],
                "junction_detected": snapshot["detected"],
                "group_start_deg": _json_group_values(snapshot["groups"], "start_angle_deg"),
                "group_end_deg": _json_group_values(snapshot["groups"], "end_angle_deg"),
                "group_width_deg": _json_group_values(snapshot["groups"], "angular_width_deg"),
            }
        )
    return rows


def _verdict(summary: list[dict[str, Any]]) -> dict[str, Any]:
    frame120, frame126 = summary
    threshold_blocked_visible_opening = frame120["sector_margin_blocked_with_side_los_count"] > 0
    expected_model_issue = (
        frame120["sector_expected_model_inconsistent_with_los_count"] > 0
        and frame120["sector_expected_at_sensor_max_count"] > 0
    )
    visibility_arrives_at_126 = (
        frame120["sector_gt_side_opening_los_count"] == 0
        and frame126["sector_gt_side_opening_los_count"] > 0
        and frame120["sector_open_candidate_count"] == 0
        and frame126["sector_open_candidate_count"] > 0
    )
    if expected_model_issue:
        case = "CASE_C_EXPECTED_PROFILE_MODEL_ISSUE"
        cause = (
            "frame 120 expected ranges saturate at sensor max because no valid "
            "side-wall/corridor model is available, although GT side-mouth LOS already exists"
        )
    elif threshold_blocked_visible_opening:
        case = "CASE_A_THRESHOLD_MARGIN_BLOCKS_VISIBLE_OPENING"
        cause = "a physically visible side-opening beam exceeds expected range but not the detector margin"
    elif visibility_arrives_at_126:
        case = "CASE_B_GEOMETRY_VISIBILITY"
        cause = "frame 120 rays hit walls before the side mouths; side-opening LOS and large positive delta first coexist at frame 126"
    else:
        case = "INCONCLUSIVE"
        cause = "observed relationships do not uniquely match CASE A, B, or C"
    return {
        "verdict": case,
        "direct_cause": cause,
        "threshold_problem": bool(case.startswith("CASE_A")),
        "geometry_visibility_problem": bool(case.startswith("CASE_B")),
        "expected_profile_problem": bool(case.startswith("CASE_C")),
        "detector_modified": False,
        "gt_used_for_detector_decision": False,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, snapshots: dict[int, dict[str, Any]], sector: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, sharey=True)
    for axis, frame in zip(axes, FRAMES):
        snapshot = snapshots[frame]
        angles = snapshot["angles"]
        boundary = snapshot["expected"] + snapshot["margin"]
        axis.plot(angles, snapshot["measured"], color="black", linewidth=1.2, label="measured")
        axis.plot(angles, snapshot["expected"], color="tab:blue", linestyle="--", label="expected")
        axis.plot(angles, boundary, color="tab:orange", linestyle=":", label="expected + margin")
        axis.fill_between(angles, 0.0, LIDAR_MAX_RANGE, where=sector, color="gray", alpha=0.10, label="frame-126 opening sectors")
        axis.fill_between(angles, 0.0, snapshot["measured"], where=snapshot["candidate"], color="magenta", alpha=0.22, label="OPEN candidate")
        axis.set(title=f"frame {frame} | t={snapshot['time']:.6f} s", ylabel="range", ylim=(0.0, LIDAR_MAX_RANGE * 1.03))
        axis.grid(alpha=0.22)
        axis.legend(loc="lower center", ncol=5)
    axes[-1].set(xlabel="body-relative LiDAR angle theta [deg]", xlim=(-180.0, 179.0))
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    runner, snapshots = _capture_frames()
    beam_rows, evaluations = _beam_rows(runner, snapshots)
    summaries = _summary_rows(snapshots, evaluations)
    verdict = _verdict(summaries)
    _write_csv(args.output / "frame120_vs_126_beams.csv", beam_rows, BEAM_FIELDS)
    _write_csv(args.output / "opening_threshold_geometry_summary.csv", summaries, SUMMARY_FIELDS)
    _write_csv(args.output / "verdict.csv", [verdict])
    if not args.no_plot:
        _plot(args.output / "frame120_vs_126_profile.png", snapshots, evaluations[120]["sector"])
    print(
        f"verdict={verdict['verdict']} output={args.output.resolve()} "
        f"frame120_candidates={summaries[0]['sector_open_candidate_count']} "
        f"frame126_candidates={summaries[1]['sector_open_candidate_count']}"
    )


if __name__ == "__main__":
    main()
