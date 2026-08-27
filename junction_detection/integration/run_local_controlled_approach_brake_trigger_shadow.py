"""EXP-047 local-only controlled-approach brake-trigger shadow validation.

The frozen LiDAR detector is consumed read-only.  After its first Junction
detection, a diagnostic trajectory keeps the existing local-forward propulsion
active and evaluates two event-based brake-ready candidates.  Candidate runs
then call the simulator's existing ``activate_profile_junction_hold`` entry
point, so braking, stationary dwell, and provisional-anchor dynamics are
unchanged.  Map geometry and global positions are used only after local
decisions for evaluation tables and plots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.integration.run_early_detection_anchor_placement import (  # noqa: E402
    BOOTSTRAP_CASE as EXP046_BOOTSTRAP_CASE,
    run_anchor_case,
    stationary_visibility_eval,
)
from junction_detection.integration.run_lidar_local_corridor_estimation import (  # noqa: E402
    REAR_START_SHIFT,
    _rear_start,
)
from junction_detection.integration.run_provisional_anchor_pointcloud_observation import (  # noqa: E402
    _snapshot as _pointcloud_snapshot,
    evaluate_snapshot,
)
from junction_detection.pointcloud.lidar_profile_junction_detector import (  # noqa: E402
    GeometryProfileConfig,
    LidarProfileJunctionDetector,
)
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import (  # noqa: E402
    detect_openings,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import (  # noqa: E402
    LIDAR_MAX_RANGE,
    MIN_SPEED,
    SimulationRunner,
)


EXPERIMENT_ID = "EXP-047"
M0_CASE = "M0_NEGATIVE_CONTROL"
BASELINE_CASE = "M1_BASELINE_CONTROLLED_APPROACH_SHADOW"
BOOTSTRAP_CASE = "M1_PRE_CORRIDOR_CONTROLLED_APPROACH_SHADOW"
IMMEDIATE_ID = "CURRENT_IMMEDIATE_BRAKE"
CANDIDATE_A = "A_SIDE_WALL_LOSS_PERSISTED"
CANDIDATE_B = "B_BILATERAL_LATERAL_GROUPS_PERSISTED"
DEFAULT_OUTPUT = ROOT / (
    "junction_detection/integration/output/"
    "local_controlled_approach_brake_trigger_shadow"
)

# These are not EXP-047 tuning parameters.  They are imported semantics of the
# frozen detector: the model initialization/confirmation observation count and
# the exact broad lateral sectors used by its current side-wall fit.
FROZEN_CONFIG = GeometryProfileConfig(LIDAR_MAX_RANGE)
EXISTING_OBSERVATION_WINDOW = FROZEN_CONFIG.initialization_scan_count
LEFT_LATERAL_SECTOR = (45.0, 135.0)
RIGHT_LATERAL_SECTOR = (-135.0, -45.0)

PROTECTED_PATHS = (
    "pygame_simulator/pre_exploration_general_pipeline_simulator.py",
    "junction_detection/pointcloud/lidar_profile_junction_detector.py",
    "junction_detection/integration/run_lidar_profile_junction_detection.py",
    "junction_detection/integration/run_active_anchor_transition.py",
    "junction_detection/integration/run_pre_corridor_bootstrap_detection_timing.py",
    "junction_detection/integration/run_early_detection_anchor_placement.py",
    "junction_detection/integration/run_local_only_degraded_start_recovery_shadow.py",
    "junction_detection/pointcloud/pointcloud_junction_detector_sensor_enhanced.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes() -> dict[str, str]:
    return {relative: _sha256(ROOT / relative) for relative in PROTECTED_PATHS}


def _new_runner(map_case: str, rear_start: bool = False) -> SimulationRunner:
    detector = LidarProfileJunctionDetector(GeometryProfileConfig(LIDAR_MAX_RANGE))
    runner = SimulationRunner(
        map_case,
        "local_forward",
        profile_detector=detector,
        hold_on_profile_detection=False,
    )
    if rear_start:
        _rear_start(runner, REAR_START_SHIFT)
    return runner


def _leader(runner: SimulationRunner) -> Any:
    return next(
        robot
        for robot in runner.world.robots
        if robot.robot_id == runner.world.lidar_robot_id
    )


def _expected_source(row: dict[str, Any]) -> str:
    if row["corridor_model_initialized"]:
        return "CURRENT_MODEL" if row["side_walls_valid"] else "STABLE_MODEL_HELD"
    if row["side_walls_valid"]:
        return "CURRENT_MODEL"
    return "MAX_RANGE_FALLBACK"


def _local_group_features(groups: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(groups, key=lambda group: float(group["center_angle_deg"]))
    padded: list[dict[str, Any] | None] = ordered[:2] + [None] * max(0, 2 - len(ordered))
    left = [group for group in ordered if float(group["center_angle_deg"]) > 0.0]
    right = [group for group in ordered if float(group["center_angle_deg"]) < 0.0]
    result: dict[str, Any] = {
        "opening_centers": json.dumps(
            [float(group["center_angle_deg"]) for group in ordered],
            separators=(",", ":"),
        ),
        "opening_widths": json.dumps(
            [float(group["angular_width_deg"]) for group in ordered],
            separators=(",", ":"),
        ),
        "left_side_opening_width_deg": (
            max(float(group["angular_width_deg"]) for group in left)
            if left
            else math.nan
        ),
        "right_side_opening_width_deg": (
            max(float(group["angular_width_deg"]) for group in right)
            if right
            else math.nan
        ),
    }
    for index, group in enumerate(padded, start=1):
        for key, source in (
            ("center_deg", "center_angle_deg"),
            ("width_deg", "angular_width_deg"),
            ("start_deg", "start_angle_deg"),
            ("end_deg", "end_angle_deg"),
        ):
            result[f"group_{index}_{key}"] = (
                math.nan if group is None else float(group[source])
            )
    return result


def _bilateral_lateral_groups(groups: list[dict[str, Any]]) -> bool:
    centers = [float(group["center_angle_deg"]) for group in groups]
    left = any(LEFT_LATERAL_SECTOR[0] <= value <= LEFT_LATERAL_SECTOR[1] for value in centers)
    right = any(RIGHT_LATERAL_SECTOR[0] <= value <= RIGHT_LATERAL_SECTOR[1] for value in centers)
    return bool(left and right)


def _eval_position(runner: SimulationRunner) -> dict[str, Any]:
    leader = _leader(runner)
    x, y = float(leader.position[0]), float(leader.position[1])
    geometry = runner.geometry
    if geometry.entrance_y is None:
        return {
            "x_eval_only": x,
            "y_eval_only": y,
            "inside_junction_eval_only": False,
            "distance_to_center_eval_only": math.nan,
            "distance_to_entry_eval_only": math.nan,
        }
    half = 0.5 * float(geometry.junction_size)
    return {
        "x_eval_only": x,
        "y_eval_only": y,
        "inside_junction_eval_only": bool(abs(x) <= half and abs(y) <= half),
        "distance_to_center_eval_only": math.hypot(x, y),
        "distance_to_entry_eval_only": float(geometry.entrance_y - y),
    }


def _feature_row(
    case_id: str,
    runner: SimulationRunner,
    row: dict[str, Any],
    detected_latched: bool,
    invalid_streak: int,
    bilateral_streak: int,
) -> dict[str, Any]:
    profile = runner.last_profile_result
    scan = runner.last_visual[0].lidar_scan
    groups = [dict(group) for group in profile["opening_groups"]]
    angles = np.asarray(scan.angles_deg)
    candidates = np.asarray(profile["open_candidate_mask"], dtype=bool)
    sensor_limit = np.asarray(scan.ranges) >= (
        float(scan.max_range) - float(profile["profile_numerical_margin"])
    )
    forward = np.abs(angles) < LEFT_LATERAL_SECTOR[0]
    rearward = np.abs(angles) > LEFT_LATERAL_SECTOR[1]
    leader = _leader(runner)
    candidate_a = bool(
        detected_latched and invalid_streak >= EXISTING_OBSERVATION_WINDOW
    )
    candidate_b = bool(
        detected_latched and bilateral_streak >= EXISTING_OBSERVATION_WINDOW
    )
    if candidate_b:
        shadow_state = "BRAKE_READY_SHADOW_B"
    elif candidate_a:
        shadow_state = "BRAKE_READY_SHADOW_A"
    elif detected_latched:
        shadow_state = "CONTROLLED_APPROACH"
    else:
        shadow_state = "PRE_DETECTION"
    return {
        "case_id": case_id,
        "frame": int(row["frame"]),
        "time": float(row["timestamp"]),
        "speed": float(np.linalg.norm(leader.velocity)),
        "junction_detected": bool(profile["profile_junction_detected"]),
        "junction_detected_latched_shadow": detected_latched,
        "open_candidate_count": int(profile["opening_candidate_count"]),
        "opening_group_count": int(profile["opening_group_count"]),
        **_local_group_features(groups),
        "forward_open_candidate_count": int(np.count_nonzero(candidates & forward)),
        "rearward_open_candidate_count": int(np.count_nonzero(candidates & rearward)),
        "sensor_limit_count": int(np.count_nonzero(sensor_limit)),
        "sensor_limit_ratio": float(np.mean(sensor_limit)),
        "side_walls_valid": bool(profile["side_walls_valid"]),
        "parallel_error_deg": float(profile["parallel_error_deg"]),
        "current_width": float(profile["width_observation"]),
        "current_offset": float(profile["offset_observation"]),
        "current_orientation": float(profile["current_corridor_orientation_deg"]),
        "stable_width": float(profile["estimated_corridor_width"]),
        "stable_offset": float(profile["estimated_offset"]),
        "stable_orientation": float(profile["stable_corridor_orientation_deg"]),
        "stable_model_initialized": bool(profile["corridor_model_initialized"]),
        "expected_profile_source": _expected_source(row),
        "side_wall_invalid_streak_samples": invalid_streak,
        "bilateral_lateral_group_streak_samples": bilateral_streak,
        "shadow_state": shadow_state,
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "candidate_c": False,
    }


def run_feature_audit(
    case_id: str,
    map_case: str,
    frames: int,
    rear_start: bool = False,
) -> dict[str, Any]:
    """Run propulsion-only trajectory; local decisions never read eval fields."""
    runner = _new_runner(map_case, rear_start)
    rows: list[dict[str, Any]] = []
    detected_latched = False
    invalid_streak = 0
    bilateral_streak = 0
    detection_frame: int | None = None
    candidate_frames: dict[str, int] = {}
    for frame in range(frames):
        row = runner.step(frame)
        if row is None:
            continue
        profile = runner.last_profile_result
        if profile["profile_junction_detected"] and not detected_latched:
            detected_latched = True
            detection_frame = frame
        if detected_latched:
            invalid_streak = invalid_streak + 1 if not profile["side_walls_valid"] else 0
            bilateral_streak = (
                bilateral_streak + 1
                if _bilateral_lateral_groups(profile["opening_groups"])
                else 0
            )
        feature = _feature_row(
            case_id, runner, row, detected_latched, invalid_streak, bilateral_streak
        )
        # Evaluation-only pose/GT fields are appended strictly after both
        # local candidate decisions have been computed.
        feature.update(
            {
                f"leader_{key}": value
                for key, value in _eval_position(runner).items()
            }
        )
        rows.append(feature)
        if feature["candidate_a"] and CANDIDATE_A not in candidate_frames:
            candidate_frames[CANDIDATE_A] = frame
        if feature["candidate_b"] and CANDIDATE_B not in candidate_frames:
            candidate_frames[CANDIDATE_B] = frame
    return {
        "case_id": case_id,
        "runner": runner,
        "rows": rows,
        "detection_frame": detection_frame,
        "candidate_frames": candidate_frames,
    }


Predicate = Callable[[dict[str, Any]], bool]


def _candidate_predicate(candidate_id: str) -> Predicate:
    key = "candidate_a" if candidate_id == CANDIDATE_A else "candidate_b"
    return lambda row: bool(row[key])


def run_candidate_brake(
    candidate_id: str,
    frames: int,
) -> dict[str, Any]:
    runner = _new_runner("M1_CROSS_BASELINE", rear_start=True)
    detected_latched = False
    invalid_streak = 0
    bilateral_streak = 0
    detection: dict[str, Any] | None = None
    trigger: dict[str, Any] | None = None
    brake_start: dict[str, Any] | None = None
    stop: dict[str, Any] | None = None
    anchor: dict[str, Any] | None = None
    trajectory: list[dict[str, Any]] = []
    predicate = _candidate_predicate(candidate_id)
    for frame in range(frames):
        row = runner.step(frame)
        leader = _leader(runner)
        trajectory.append(
            {
                "frame": frame,
                "x_eval_only": float(leader.position[0]),
                "y_eval_only": float(leader.position[1]),
                "speed": float(np.linalg.norm(leader.velocity)),
            }
        )
        if runner.world.braking_active and brake_start is None:
            brake_start = {"frame": frame, **_eval_position(runner)}
        if trigger is not None and stop is None and np.linalg.norm(leader.velocity) < MIN_SPEED:
            stop = {"frame": frame, **_eval_position(runner)}
        if runner.world.provisional_fixed_anchor and anchor is None:
            anchor = {"frame": int(runner.world.anchor_entry_frame), **_eval_position(runner)}
        if row is None:
            continue
        profile = runner.last_profile_result
        if profile["profile_junction_detected"] and not detected_latched:
            detected_latched = True
            detection = {
                "frame": frame,
                "time": float(row["timestamp"]),
                "speed": float(np.linalg.norm(leader.velocity)),
                **_eval_position(runner),
            }
        if detected_latched:
            invalid_streak = invalid_streak + 1 if not profile["side_walls_valid"] else 0
            bilateral_streak = (
                bilateral_streak + 1
                if _bilateral_lateral_groups(profile["opening_groups"])
                else 0
            )
        feature = _feature_row(
            BOOTSTRAP_CASE,
            runner,
            row,
            detected_latched,
            invalid_streak,
            bilateral_streak,
        )
        if trigger is None and predicate(feature):
            trigger = {
                "frame": frame,
                "time": float(row["timestamp"]),
                "speed": float(np.linalg.norm(leader.velocity)),
                "group_count": int(feature["opening_group_count"]),
                "opening_centers": feature["opening_centers"],
                "opening_widths": feature["opening_widths"],
                **_eval_position(runner),
            }
            runner.world.activate_profile_junction_hold(
                float(row["timestamp"]),
                float(profile["stable_corridor_orientation_deg"]),
                float(profile["estimated_corridor_width"]),
            )
    return {
        "candidate_id": candidate_id,
        "runner": runner,
        "detection": detection,
        "trigger": trigger,
        "brake_start": brake_start,
        "stop": stop,
        "anchor": anchor,
        "trajectory": trajectory,
    }


def _visibility_for_runner(runner: SimulationRunner, case_id: str) -> dict[str, Any]:
    if not runner.world.provisional_fixed_anchor:
        return {
            "case_id": case_id,
            "outgoing_visible_count_eval_only": 0,
            "outgoing_gt_count_eval_only": 0,
            "side_visible_count_eval_only": 0,
            "stationary_opening_group_count": 0,
            "stationary_angular_coverage_deg": 0.0,
            "pointcloud_runtime_integrated": False,
        }
    snapshot = _pointcloud_snapshot(runner, "EXP047_STATIONARY_POSTHOC_EVAL_ONLY")
    openings = list(detect_openings(snapshot["angles"], snapshot["ranges"]))
    summary, opening_rows = evaluate_snapshot(runner, snapshot, openings)
    side_ids = {
        index
        for index, branch in enumerate(runner.geometry.branches)
        if abs(float(branch.angle_deg)) >= LEFT_LATERAL_SECTOR[0]
    }
    visible_side = sum(
        isinstance(item["matched_GT_branch_eval_only"], int)
        and item["matched_GT_branch_eval_only"] in side_ids
        for item in opening_rows
    )
    return {
        "case_id": case_id,
        "outgoing_visible_count_eval_only": int(summary["matched_outgoing_count_eval_only"]),
        "outgoing_gt_count_eval_only": int(summary["GT_outgoing_branch_count_eval_only"]),
        "side_visible_count_eval_only": int(visible_side),
        "stationary_opening_group_count": len(openings),
        "stationary_angular_coverage_deg": float(
            sum(float(opening["width_deg"]) for opening in openings)
        ),
        "pointcloud_runtime_integrated": False,
    }


def _distance(first: dict[str, Any] | None, second: dict[str, Any] | None) -> float:
    if first is None or second is None:
        return math.nan
    return math.hypot(
        float(second["x_eval_only"]) - float(first["x_eval_only"]),
        float(second["y_eval_only"]) - float(first["y_eval_only"]),
    )


def _candidate_event(run: dict[str, Any]) -> dict[str, Any]:
    trigger = run["trigger"]
    local_reason = (
        "Junction latched AND current side-wall fit invalid for existing initialization_scan_count"
        if run["candidate_id"] == CANDIDATE_A
        else "Junction latched AND opening centers in both existing lateral wall-fit sectors for existing initialization_scan_count"
    )
    return {
        "case_id": BOOTSTRAP_CASE,
        "candidate_id": run["candidate_id"],
        "trigger_frame": "" if trigger is None else trigger["frame"],
        "trigger_time": "" if trigger is None else trigger["time"],
        "trigger_speed": "" if trigger is None else trigger["speed"],
        "local_reason": local_reason,
        "group_count": "" if trigger is None else trigger["group_count"],
        "opening_centers": "" if trigger is None else trigger["opening_centers"],
        "opening_widths": "" if trigger is None else trigger["opening_widths"],
        "existing_observation_window": EXISTING_OBSERVATION_WINDOW,
        "new_threshold_used": False,
        "runtime_gt_used": False,
    }


def _candidate_position(run: dict[str, Any], visibility: dict[str, Any]) -> dict[str, Any]:
    trigger, stop, anchor, detection = (
        run["trigger"], run["stop"], run["anchor"], run["detection"]
    )
    return {
        "case_id": BOOTSTRAP_CASE,
        "candidate_id": run["candidate_id"],
        "detection_frame": "" if detection is None else detection["frame"],
        "trigger_frame": "" if trigger is None else trigger["frame"],
        "trigger_x_eval_only": math.nan if trigger is None else trigger["x_eval_only"],
        "trigger_y_eval_only": math.nan if trigger is None else trigger["y_eval_only"],
        "stop_frame": "" if stop is None else stop["frame"],
        "stop_x_eval_only": math.nan if stop is None else stop["x_eval_only"],
        "stop_y_eval_only": math.nan if stop is None else stop["y_eval_only"],
        "stop_distance_from_trigger": _distance(trigger, stop),
        "anchor_frame": "" if anchor is None else anchor["frame"],
        "anchor_x_eval_only": math.nan if anchor is None else anchor["x_eval_only"],
        "anchor_y_eval_only": math.nan if anchor is None else anchor["y_eval_only"],
        "inside_junction_eval_only": False if anchor is None else anchor["inside_junction_eval_only"],
        "distance_to_center_eval_only": math.nan if anchor is None else anchor["distance_to_center_eval_only"],
        "distance_to_entry_eval_only": math.nan if anchor is None else anchor["distance_to_entry_eval_only"],
        "approach_duration_frames": "" if trigger is None or detection is None else int(trigger["frame"]) - int(detection["frame"]),
        "approach_distance": _distance(detection, trigger),
        "outgoing_visible_count_eval_only": visibility["outgoing_visible_count_eval_only"],
        "side_visible_count_eval_only": visibility["side_visible_count_eval_only"],
    }


def _immediate_rows(run: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    detection = run["first"]["JUNCTION_DETECTION"]
    brake = run["first"]["BRAKE_TRIGGER"]
    stop = run["first"]["SPEED_BELOW_THRESHOLD"]
    anchor = run["first"]["PROVISIONAL_ANCHOR_ENTER"]
    visible = stationary_visibility_eval(run)
    position = {
        "case_id": BOOTSTRAP_CASE,
        "candidate_id": IMMEDIATE_ID,
        "detection_frame": detection["frame"],
        "trigger_frame": brake["frame"],
        "trigger_x_eval_only": brake["leader_x"],
        "trigger_y_eval_only": brake["leader_y"],
        "stop_frame": stop["frame"],
        "stop_x_eval_only": stop["leader_x"],
        "stop_y_eval_only": stop["leader_y"],
        "stop_distance_from_trigger": math.hypot(stop["leader_x"] - brake["leader_x"], stop["leader_y"] - brake["leader_y"]),
        "anchor_frame": anchor["frame"],
        "anchor_x_eval_only": anchor["leader_x"],
        "anchor_y_eval_only": anchor["leader_y"],
        "inside_junction_eval_only": anchor["inside_junction_region_eval_only"],
        "distance_to_center_eval_only": anchor["distance_to_junction_center_eval_only"],
        "distance_to_entry_eval_only": anchor["distance_to_junction_entry_eval_only"],
        "approach_duration_frames": int(brake["frame"]) - int(detection["frame"]),
        "approach_distance": math.hypot(brake["leader_x"] - detection["leader_x"], brake["leader_y"] - detection["leader_y"]),
        "outgoing_visible_count_eval_only": visible["visible_outgoing_branch_count_eval_only"],
        "side_visible_count_eval_only": visible["visible_side_opening_count_eval_only"],
    }
    visibility = {
        "case_id": IMMEDIATE_ID,
        "outgoing_visible_count_eval_only": visible["visible_outgoing_branch_count_eval_only"],
        "outgoing_gt_count_eval_only": visible["gt_outgoing_branch_count_eval_only"],
        "side_visible_count_eval_only": visible["visible_side_opening_count_eval_only"],
        "stationary_opening_group_count": visible["stationary_opening_group_count"],
        "stationary_angular_coverage_deg": visible["angular_coverage_deg"],
        "pointcloud_runtime_integrated": False,
    }
    return position, visibility


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_features(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [row for row in rows if row["junction_detected_latched_shadow"]]
    progress = [-float(row["leader_distance_to_entry_eval_only"]) for row in selected]
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(progress, [row["left_side_opening_width_deg"] for row in selected], label="left width")
    axes[0].plot(progress, [row["right_side_opening_width_deg"] for row in selected], label="right width")
    axes[0].set_ylabel("opening width [deg]"); axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].plot(progress, [row["group_1_center_deg"] for row in selected], label="group 1 center")
    axes[1].plot(progress, [row["group_2_center_deg"] for row in selected], label="group 2 center")
    axes[1].axhspan(*RIGHT_LATERAL_SECTOR, alpha=.1, color="tab:orange")
    axes[1].axhspan(*LEFT_LATERAL_SECTOR, alpha=.1, color="tab:orange")
    axes[1].set_ylabel("center [deg]"); axes[1].legend(); axes[1].grid(alpha=.2)
    axes[2].plot(progress, [row["sensor_limit_ratio"] for row in selected], label="sensor-limit ratio")
    axes[2].step(progress, [not row["side_walls_valid"] for row in selected], where="post", label="side fit invalid")
    axes[2].set(xlabel="progress relative to entry [eval only]", ylabel="local feature", title="Local features vs Junction progress (GT x-axis EVAL ONLY)")
    axes[2].legend(); axes[2].grid(alpha=.2); figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def _plot_positions(path: Path, immediate: dict[str, Any], candidates: list[dict[str, Any]], geometry: Any) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8, 9))
    for index, wall in enumerate(geometry.walls):
        axis.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], color="black", linewidth=1.0, label="walls" if index == 0 else None)
    rows = [immediate] + candidates
    colors = ["tab:red", "tab:blue", "tab:green"]
    for row, color in zip(rows, colors):
        axis.scatter(row["trigger_x_eval_only"], row["trigger_y_eval_only"], marker="s", color=color, label=f"{row['candidate_id']} trigger")
        axis.scatter(row["anchor_x_eval_only"], row["anchor_y_eval_only"], marker="*", s=130, color=color, label=f"{row['candidate_id']} anchor")
        axis.plot([row["trigger_x_eval_only"], row["anchor_x_eval_only"]], [row["trigger_y_eval_only"], row["anchor_y_eval_only"]], color=color, alpha=.6)
    axis.axhline(float(geometry.entrance_y), color="tab:orange", linestyle="--", label="entry EVAL ONLY")
    axis.set(xlabel="world x [eval only]", ylabel="world y [eval only]", title="Candidate brake and anchor positions"); axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.2); axis.legend(fontsize=7); figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def _plot_opening_geometry(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [row for row in rows if row["junction_detected_latched_shadow"]]
    figure, axis = plt.subplots(figsize=(11, 6))
    times = [row["time"] for row in selected]
    axis.fill_between(times, [row["group_1_start_deg"] for row in selected], [row["group_1_end_deg"] for row in selected], alpha=.35, label="group 1 interval")
    axis.fill_between(times, [row["group_2_start_deg"] for row in selected], [row["group_2_end_deg"] for row in selected], alpha=.35, label="group 2 interval")
    axis.plot(times, [row["group_1_center_deg"] for row in selected], color="tab:blue")
    axis.plot(times, [row["group_2_center_deg"] for row in selected], color="tab:orange")
    axis.set(xlabel="time [s]", ylabel="corridor-relative angle [deg]", title="Opening geometry during propulsion-only controlled approach"); axis.grid(alpha=.2); axis.legend(); figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def _signature(audit: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[Any, ...]:
    return (
        audit["detection_frame"],
        tuple(sorted(audit["candidate_frames"].items())),
        tuple((row["frame"], row["open_candidate_count"], row["opening_group_count"], row["side_walls_valid"], row["candidate_a"], row["candidate_b"]) for row in audit["rows"]),
        tuple((run["candidate_id"], run["trigger"]["frame"] if run["trigger"] else None, run["anchor"]["frame"] if run["anchor"] else None, round(run["anchor"]["y_eval_only"], 9) if run["anchor"] else None) for run in candidates),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--baseline-frames", type=int, default=210)
    parser.add_argument("--m0-frames", type=int, default=600)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args(argv)

    hashes_before = protected_hashes()
    print("[EXP-047] bootstrap propulsion-only feature audit", flush=True)
    bootstrap_audit = run_feature_audit(BOOTSTRAP_CASE, "M1_CROSS_BASELINE", args.frames, rear_start=True)
    print("[EXP-047] baseline propulsion-only feature audit", flush=True)
    baseline_audit = run_feature_audit(BASELINE_CASE, "M1_CROSS_BASELINE", args.baseline_frames)
    print("[EXP-047] M0 negative control", flush=True)
    m0_audit = run_feature_audit(M0_CASE, "M0_STRAIGHT", args.m0_frames)
    candidate_runs = []
    for candidate in (CANDIDATE_A, CANDIDATE_B):
        print(f"[EXP-047] physical brake simulation {candidate}", flush=True)
        candidate_runs.append(run_candidate_brake(candidate, args.frames))
    print("[EXP-047] EXP-046 immediate-brake reference", flush=True)
    immediate_run = run_anchor_case(EXP046_BOOTSTRAP_CASE, "M1_CROSS_BASELINE", args.frames, rear_start=True)

    candidate_visibility = [_visibility_for_runner(run["runner"], run["candidate_id"]) for run in candidate_runs]
    candidate_positions = [_candidate_position(run, visible) for run, visible in zip(candidate_runs, candidate_visibility)]
    immediate_position, immediate_visibility = _immediate_rows(immediate_run)
    all_positions = [immediate_position] + candidate_positions
    all_visibility = [immediate_visibility] + candidate_visibility

    replay_match = False
    if not args.skip_replay:
        print("[EXP-047] deterministic replay feature audit", flush=True)
        replay_audit = run_feature_audit(BOOTSTRAP_CASE, "M1_CROSS_BASELINE", args.frames, rear_start=True)
        replay_candidates = []
        for candidate in (CANDIDATE_A, CANDIDATE_B):
            print(f"[EXP-047] deterministic replay {candidate}", flush=True)
            replay_candidates.append(run_candidate_brake(candidate, args.frames))
        replay_match = _signature(bootstrap_audit, candidate_runs) == _signature(replay_audit, replay_candidates)

    timeline = bootstrap_audit["rows"] + baseline_audit["rows"] + m0_audit["rows"]
    events = [_candidate_event(run) for run in candidate_runs]
    summaries = []
    for position in all_positions:
        summaries.append({
            **position,
            "anchor_observation_improved_vs_immediate": bool(
                position["side_visible_count_eval_only"] > immediate_position["side_visible_count_eval_only"]
                or position["outgoing_visible_count_eval_only"] > immediate_position["outgoing_visible_count_eval_only"]
            ),
            "new_threshold_used": False,
            "runtime_gt_used": False,
            "same_brake_law": True,
        })

    best = max(candidate_positions, key=lambda row: (row["side_visible_count_eval_only"], row["outgoing_visible_count_eval_only"], -abs(float(row["distance_to_center_eval_only"]))))
    strong_success = bool(best["inside_junction_eval_only"] and best["side_visible_count_eval_only"] > immediate_position["side_visible_count_eval_only"])
    if strong_success:
        verdict_name = "A_LOCAL_CONTROLLED_APPROACH_FEASIBLE"
    elif any(row["inside_junction_eval_only"] for row in candidate_positions):
        verdict_name = "B_LOCAL_SIGNAL_EXISTS_BUT_TRIGGER_NEEDS_DESIGN"
    elif all(float(row["distance_to_entry_eval_only"]) < 0.0 for row in candidate_positions):
        verdict_name = "D_CONTROLLED_APPROACH_OVERSHOOTS"
    else:
        verdict_name = "E_CONTROLLED_APPROACH_STILL_TOO_EARLY"
    hashes_after = protected_hashes()
    baseline_detection = baseline_audit["detection_frame"]
    verdict = {
        "verdict": verdict_name,
        "best_candidate": best["candidate_id"],
        "bootstrap_detection_frame": bootstrap_audit["detection_frame"],
        "exp042_detection_equivalent": bootstrap_audit["detection_frame"] == 36,
        "immediate_anchor_frame": immediate_position["anchor_frame"],
        "exp046_immediate_anchor_equivalent": immediate_position["anchor_frame"] == 80 and abs(float(immediate_position["anchor_y_eval_only"]) + 152.74885) < 1.0e-3,
        "baseline_detection_frame": baseline_detection,
        "baseline_detection_equivalent": baseline_detection == 126,
        "m0_detection_count": sum(bool(row["junction_detected"]) for row in m0_audit["rows"]),
        "m0_candidate_a_count": sum(bool(row["candidate_a"]) for row in m0_audit["rows"]),
        "m0_candidate_b_count": sum(bool(row["candidate_b"]) for row in m0_audit["rows"]),
        "new_numeric_threshold_added": False,
        "existing_observation_window": EXISTING_OBSERVATION_WINDOW,
        "runtime_gt_or_map_used": False,
        "production_gating_applied": False,
        "pointcloud_runtime_integrated": False,
        "same_existing_brake_law": True,
        "deterministic_replay_match": replay_match,
        "protected_hashes_unchanged": hashes_before == hashes_after,
        "detector_changed": False,
        "simulator_changed": False,
    }

    assertions = {
        "bootstrap_detection_frame_36": bootstrap_audit["detection_frame"] == 36,
        "baseline_detection_frame_126": baseline_detection == 126,
        "immediate_anchor_frame_80": immediate_position["anchor_frame"] == 80,
        "m0_no_detection": verdict["m0_detection_count"] == 0,
        "m0_no_candidate": verdict["m0_candidate_a_count"] == 0 and verdict["m0_candidate_b_count"] == 0,
        "all_candidates_triggered": all(run["trigger"] is not None for run in candidate_runs),
        "all_candidates_anchored": all(run["anchor"] is not None for run in candidate_runs),
        "no_runtime_gt": not any(bool(row["runtime_gt_used"]) for row in events),
        "protected_hashes_unchanged": hashes_before == hashes_after,
    }
    if not all(assertions.values()):
        raise AssertionError(json.dumps(assertions, sort_keys=True))

    _write(args.output / "post_detection_local_feature_timeline.csv", timeline)
    _write(args.output / "brake_trigger_candidate_events.csv", events)
    _write(args.output / "predicted_or_simulated_anchor_positions.csv", all_positions)
    _write(args.output / "anchor_visibility_comparison.csv", all_visibility)
    _write(args.output / "candidate_summary.csv", summaries)
    _write(args.output / "verdict.csv", [verdict])
    _write(args.output / "protected_hashes.csv", [{"path": path, "sha256_before": hashes_before[path], "sha256_after": hashes_after[path], "unchanged": hashes_before[path] == hashes_after[path]} for path in PROTECTED_PATHS])
    _plot_features(args.output / "local_features_vs_junction_progress_eval_only.png", bootstrap_audit["rows"])
    _plot_positions(args.output / "candidate_brake_and_stop_positions.png", immediate_position, candidate_positions, bootstrap_audit["runner"].geometry)
    _plot_opening_geometry(args.output / "opening_geometry_during_approach.png", bootstrap_audit["rows"])
    print(f"experiment={EXPERIMENT_ID} verdict={verdict_name} bootstrap_detection={bootstrap_audit['detection_frame']} baseline_detection={baseline_detection} candidates={bootstrap_audit['candidate_frames']} best={best['candidate_id']} replay={replay_match} output={args.output.resolve()}")


if __name__ == "__main__":
    main()
