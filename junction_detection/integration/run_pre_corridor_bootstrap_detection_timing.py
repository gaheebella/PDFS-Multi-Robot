"""EXP-042: pre-corridor bootstrap effect on frozen LiDAR detection timing.

Only the existing evaluation rear-start initial condition is varied. Detector,
threshold, grouping, Junction geometry, simulator physics, and runtime inputs
remain unchanged. Global geometry and poses are consumed only after detector
decisions for evaluation tables and plots.
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

from junction_detection.integration.run_lidar_local_corridor_estimation import (  # noqa: E402
    REAR_START_SHIFT,
    _rear_start,
)
from junction_detection.integration.run_opening_threshold_geometry_diagnostic import (  # noqa: E402
    _side_opening_los_eval_only,
)
from junction_detection.pointcloud.lidar_profile_junction_detector import (  # noqa: E402
    GeometryProfileConfig,
    LidarProfileJunctionDetector,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import (  # noqa: E402
    BASELINE_CORRIDOR_WIDTH,
    LIDAR_MAX_RANGE,
    SimulationRunner,
)


EXPERIMENT_ID = "EXP-042"
SEED = 0
BASELINE_CASE = "BASELINE_M1"
BOOTSTRAP_CASE = "BOOTSTRAP_M1"
M0_CASE = "M0_NEGATIVE_CONTROL"
DEFAULT_OUTPUT = (
    ROOT
    / "junction_detection/integration/output/pre_corridor_bootstrap_detection_timing"
)
LIFECYCLE_FIELDS = (
    "case_id",
    "map_case",
    "physics_frame",
    "time",
    "leader_progress_eval_only",
    "leader_x_eval_only",
    "leader_y_eval_only",
    "side_walls_valid",
    "parallel_error_deg",
    "current_width",
    "current_offset",
    "current_orientation",
    "stable_width",
    "stable_offset",
    "stable_orientation",
    "stable_model_update_count",
    "stable_model_initialized",
    "expected_profile_source",
    "open_candidate_count",
    "opening_group_count",
    "junction_detected",
    "gt_side_opening_in_fit_sector_eval_only",
    "distance_to_junction_entry_eval_only",
    "progress_relative_to_junction_entry_eval_only",
)


def _new_runner(map_case: str) -> SimulationRunner:
    detector = LidarProfileJunctionDetector(GeometryProfileConfig(LIDAR_MAX_RANGE))
    return SimulationRunner(
        map_case,
        "local_forward",
        profile_detector=detector,
        hold_on_profile_detection=False,
    )


def _expected_source(row: dict[str, Any]) -> str:
    if row["corridor_model_initialized"]:
        return "CURRENT_MODEL" if row["side_walls_valid"] else "STABLE_MODEL_HELD"
    if row["side_walls_valid"]:
        return "CURRENT_MODEL"
    return "MAX_RANGE_FALLBACK"


def _fit_sector_side_los_count_eval_only(runner: SimulationRunner) -> int:
    scan = runner.last_visual[0].lidar_scan
    leader = next(
        robot for robot in runner.world.robots
        if robot.robot_id == runner.world.lidar_robot_id
    )
    snapshot = {
        "angles": scan.angles_deg,
        "measured": scan.ranges,
        "max_range": scan.max_range,
        "leader_position_eval_only": leader.position,
        "lidar_yaw_deg_eval_only": runner.world.lidar_yaw_deg,
    }
    side_los = _side_opening_los_eval_only(runner, snapshot)
    angles = scan.angles_deg
    fit_sector = ((angles >= 45.0) & (angles <= 135.0)) | (
        (angles >= -135.0) & (angles <= -45.0)
    )
    return int(np.count_nonzero(side_los & fit_sector))


def run_case(
    case_id: str,
    map_case: str,
    frames: int,
    rear_shift: float = 0.0,
) -> tuple[SimulationRunner, list[dict[str, Any]]]:
    runner = _new_runner(map_case)
    if rear_shift:
        _rear_start(runner, rear_shift)
    initial_leader_y = float(runner.world.initial_lidar_position[1])
    lifecycle: list[dict[str, Any]] = []
    for frame in range(frames):
        row = runner.step(frame)
        if row is None:
            continue
        leader = next(
            robot for robot in runner.world.robots
            if robot.robot_id == runner.world.lidar_robot_id
        )
        entry = runner.geometry.entrance_y
        distance_to_entry = (
            float(entry - leader.position[1]) if entry is not None else math.nan
        )
        lifecycle.append(
            {
                "case_id": case_id,
                "map_case": map_case,
                "physics_frame": frame,
                "time": float(row["timestamp"]),
                "leader_progress_eval_only": float(leader.position[1])
                - initial_leader_y,
                "leader_x_eval_only": float(leader.position[0]),
                "leader_y_eval_only": float(leader.position[1]),
                "side_walls_valid": bool(row["side_walls_valid"]),
                "parallel_error_deg": row["parallel_error_deg"],
                "current_width": row["width_observation"],
                "current_offset": row["offset_observation"],
                "current_orientation": row["current_corridor_orientation_deg"],
                "stable_width": row["estimated_corridor_width"],
                "stable_offset": row["estimated_offset"],
                "stable_orientation": row["stable_corridor_orientation_deg"],
                "stable_model_update_count": int(
                    row["corridor_model_update_count"]
                ),
                "stable_model_initialized": bool(
                    row["corridor_model_initialized"]
                ),
                "expected_profile_source": _expected_source(row),
                "open_candidate_count": int(row["opening_candidate_count"]),
                "opening_group_count": int(row["opening_group_count"]),
                "junction_detected": bool(row["profile_junction_detected"]),
                "gt_side_opening_in_fit_sector_eval_only": (
                    _fit_sector_side_los_count_eval_only(runner)
                    if entry is not None
                    else 0
                ),
                "distance_to_junction_entry_eval_only": distance_to_entry,
                "progress_relative_to_junction_entry_eval_only": (
                    -distance_to_entry if math.isfinite(distance_to_entry) else math.nan
                ),
            }
        )
    return runner, lifecycle


def _first(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    return next((row for row in rows if row[key]), None)


def _timing_summary(case_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = _first(rows, "side_walls_valid")
    stable = _first(rows, "stable_model_initialized")
    candidate = _first(rows, "open_candidate_count")
    group = _first(rows, "opening_group_count")
    detection = _first(rows, "junction_detected")
    return {
        "case_id": case_id,
        "first_valid_model_frame": _value(valid, "physics_frame"),
        "first_valid_model_time": _value(valid, "time"),
        "stable_model_init_frame": _value(stable, "physics_frame"),
        "stable_model_init_time": _value(stable, "time"),
        "first_open_frame": _value(candidate, "physics_frame"),
        "first_open_time": _value(candidate, "time"),
        "first_open_group_frame": _value(group, "physics_frame"),
        "first_open_group_time": _value(group, "time"),
        "first_detection_frame": _value(detection, "physics_frame"),
        "first_detection_time": _value(detection, "time"),
        "opening_group_count_at_detection": _value(
            detection, "opening_group_count", 0
        ),
        "expected_source_at_detection": _value(
            detection, "expected_profile_source", "NONE"
        ),
        "stable_width_at_detection": _value(detection, "stable_width"),
        "stable_offset_at_detection": _value(detection, "stable_offset"),
        "stable_orientation_at_detection": _value(
            detection, "stable_orientation"
        ),
    }


def _value(row: dict[str, Any] | None, key: str, default: Any = "") -> Any:
    return default if row is None else row[key]


def _position_summary(case_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    detection = _first(rows, "junction_detected")
    return {
        "case_id": case_id,
        "detection_frame": _value(detection, "physics_frame"),
        "detection_time": _value(detection, "time"),
        "leader_x_eval_only": _value(detection, "leader_x_eval_only"),
        "leader_y_eval_only": _value(detection, "leader_y_eval_only"),
        "distance_to_junction_entry_eval_only": _value(
            detection, "distance_to_junction_entry_eval_only"
        ),
        "progress_relative_to_junction_entry_eval_only": _value(
            detection, "progress_relative_to_junction_entry_eval_only"
        ),
    }


def _bootstrap_cases() -> list[dict[str, Any]]:
    return [
        {
            "case_id": BASELINE_CASE,
            "map_case": "M1_CROSS_BASELINE",
            "initial_offset_description": "repository default initial pose",
            "pre_corridor_length_world": 0.0,
            "pre_corridor_length_over_W": 0.0,
            "pre_corridor_length_over_lidar_range": 0.0,
            "junction_geometry_changed": False,
            "detector_changed": False,
            "seed": SEED,
        },
        {
            "case_id": BOOTSTRAP_CASE,
            "map_case": "M1_CROSS_BASELINE",
            "initial_offset_description": "existing REAR_START_SHIFT and extended incoming straight corridor",
            "pre_corridor_length_world": REAR_START_SHIFT,
            "pre_corridor_length_over_W": REAR_START_SHIFT
            / BASELINE_CORRIDOR_WIDTH,
            "pre_corridor_length_over_lidar_range": REAR_START_SHIFT
            / LIDAR_MAX_RANGE,
            "junction_geometry_changed": False,
            "detector_changed": False,
            "seed": SEED,
        },
        {
            "case_id": M0_CASE,
            "map_case": "M0_STRAIGHT",
            "initial_offset_description": "repository default long straight corridor",
            "pre_corridor_length_world": 0.0,
            "pre_corridor_length_over_W": 0.0,
            "pre_corridor_length_over_lidar_range": 0.0,
            "junction_geometry_changed": False,
            "detector_changed": False,
            "seed": SEED,
        },
    ]


def _replay_signature(rows: list[dict[str, Any]]) -> tuple[Any, ...]:
    summary = _timing_summary(BOOTSTRAP_CASE, rows)
    return tuple(summary.values()) + tuple(
        (
            row["physics_frame"],
            row["side_walls_valid"],
            row["stable_model_initialized"],
            row["expected_profile_source"],
            row["open_candidate_count"],
            row["opening_group_count"],
            row["junction_detected"],
        )
        for row in rows
    )


def _verdict(
    baseline: list[dict[str, Any]],
    bootstrap: list[dict[str, Any]],
    m0: list[dict[str, Any]],
    replay_match: bool,
) -> dict[str, Any]:
    old = _timing_summary(BASELINE_CASE, baseline)
    new = _timing_summary(BOOTSTRAP_CASE, bootstrap)
    bootstrap_detection = _first(bootstrap, "junction_detected")
    stable = _first(bootstrap, "stable_model_initialized")
    clean_bootstrap = bool(
        stable is not None
        and stable["gt_side_opening_in_fit_sector_eval_only"] == 0
    )
    m0_clear = not any(row["open_candidate_count"] for row in m0)
    stable_before_detection = bool(
        stable is not None
        and bootstrap_detection is not None
        and stable["physics_frame"] < bootstrap_detection["physics_frame"]
    )
    earlier = bool(
        old["first_detection_frame"] != ""
        and new["first_detection_frame"] != ""
        and new["first_detection_frame"] < old["first_detection_frame"]
    )
    baseline_reproduced = bool(
        old["first_valid_model_frame"] == 126
        and old["first_open_frame"] == 126
        and old["first_detection_frame"] == 126
        and math.isclose(
            float(old["first_detection_time"]),
            2.116666666666664,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    )
    if not baseline_reproduced:
        name = "BASELINE_REPRODUCTION_FAILED"
    elif not m0_clear:
        name = "D_BOOTSTRAP_CAUSES_FALSE_REGRESSION"
    elif stable_before_detection and clean_bootstrap and earlier:
        name = "A_BOOTSTRAP_FAILURE_CONFIRMED_AND_RECOVERED"
    elif stable_before_detection and clean_bootstrap:
        name = "B_BOOTSTRAP_VALID_BUT_DETECTION_STILL_LATE"
    else:
        name = "C_SIDE_WALL_MODEL_BOOTSTRAP_MECHANISM_INADEQUATE"
    return {
        "verdict": name,
        "baseline_reproduced": baseline_reproduced,
        "clean_bootstrap_before_side_opening_fit_sector_los": clean_bootstrap,
        "stable_model_before_detection": stable_before_detection,
        "timing_improvement_frames": int(old["first_detection_frame"])
        - int(new["first_detection_frame"]),
        "timing_improvement_seconds": float(old["first_detection_time"])
        - float(new["first_detection_time"]),
        "spatial_improvement_world": float(
            _first(bootstrap, "junction_detected")[
                "distance_to_junction_entry_eval_only"
            ]
        )
        - float(
            _first(baseline, "junction_detected")[
                "distance_to_junction_entry_eval_only"
            ]
        ),
        "m0_open_candidate_count": sum(
            int(row["open_candidate_count"]) for row in m0
        ),
        "m0_opening_group_count": sum(
            int(row["opening_group_count"]) for row in m0
        ),
        "m0_detection_count": sum(bool(row["junction_detected"]) for row in m0),
        "deterministic_replay_match": replay_match,
        "threshold_changed": False,
        "margin_changed": False,
        "grouping_changed": False,
        "detector_changed": False,
        "simulator_physics_changed": False,
        "gt_or_map_used_for_runtime_decision": False,
    }


def _write(path: Path, rows: list[dict[str, Any]], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _plot_positions(
    path: Path,
    runner: SimulationRunner,
    baseline: list[dict[str, Any]],
    bootstrap: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    figure, axis = plt.subplots(figsize=(8, 8))
    for rect in runner.geometry.free_rects:
        axis.add_patch(
            Polygon(rect.vertices, closed=True, facecolor="#e8edf2", edgecolor="#66717e")
        )
    points = [
        (baseline[0], "baseline start", "tab:gray", "o"),
        (bootstrap[0], "bootstrap start", "tab:blue", "o"),
        (_first(baseline, "junction_detected"), "baseline detection", "tab:red", "X"),
        (_first(bootstrap, "junction_detected"), "bootstrap detection", "tab:green", "X"),
    ]
    for row, label, color, marker in points:
        axis.scatter(
            row["leader_x_eval_only"],
            row["leader_y_eval_only"],
            color=color,
            marker=marker,
            s=90,
            label=label,
            zorder=4,
        )
    entry = runner.geometry.entrance_y
    axis.axhline(entry, color="tab:orange", linestyle="--", label="Junction entry eval-only")
    axis.set(aspect="equal", xlabel="x eval-only", ylabel="y eval-only", title="EXP-042 bootstrap vs baseline detection position")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_timeline(
    path: Path,
    baseline: list[dict[str, Any]],
    bootstrap: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    source_level = {
        "MAX_RANGE_FALLBACK": 0.0,
        "CURRENT_MODEL": 1.0,
        "STABLE_MODEL_HELD": 2.0,
    }
    for axis, rows, title in zip(
        axes, (baseline, bootstrap), (BASELINE_CASE, BOOTSTRAP_CASE)
    ):
        frames = [row["physics_frame"] for row in rows]
        axis.step(frames, [source_level[row["expected_profile_source"]] for row in rows], where="post", label="expected source 0/1/2")
        axis.step(frames, [1.2 if row["stable_model_initialized"] else 0.0 for row in rows], where="post", label="stable initialized")
        axis.plot(frames, [min(3.0, row["open_candidate_count"] / 2.0) for row in rows], label="OPEN candidates /2 capped")
        axis.step(frames, [2.8 if row["junction_detected"] else 0.0 for row in rows], where="post", label="Junction detected")
        axis.set(title=title, ylabel="state level")
        axis.grid(alpha=0.22)
        axis.legend(loc="best", ncol=4)
    axes[-1].set_xlabel("physics frame")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-frames", type=int, default=127)
    parser.add_argument("--bootstrap-frames", type=int, default=127)
    parser.add_argument("--m0-frames", type=int, default=600)
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args(argv)

    baseline_runner, baseline = run_case(
        BASELINE_CASE, "M1_CROSS_BASELINE", args.baseline_frames
    )
    bootstrap_runner, bootstrap = run_case(
        BOOTSTRAP_CASE,
        "M1_CROSS_BASELINE",
        args.bootstrap_frames,
        REAR_START_SHIFT,
    )
    _, m0 = run_case(M0_CASE, "M0_STRAIGHT", args.m0_frames)
    if args.skip_replay:
        replay_match = False
    else:
        _, replay = run_case(
            BOOTSTRAP_CASE,
            "M1_CROSS_BASELINE",
            args.bootstrap_frames,
            REAR_START_SHIFT,
        )
        replay_match = _replay_signature(bootstrap) == _replay_signature(replay)

    lifecycle = baseline + bootstrap + m0
    timings = [
        _timing_summary(BASELINE_CASE, baseline),
        _timing_summary(BOOTSTRAP_CASE, bootstrap),
        _timing_summary(M0_CASE, m0),
    ]
    positions = [
        _position_summary(BASELINE_CASE, baseline),
        _position_summary(BOOTSTRAP_CASE, bootstrap),
        _position_summary(M0_CASE, m0),
    ]
    verdict = _verdict(baseline, bootstrap, m0, replay_match)
    _write(args.output / "bootstrap_cases.csv", _bootstrap_cases())
    _write(args.output / "corridor_model_lifecycle.csv", lifecycle, LIFECYCLE_FIELDS)
    _write(args.output / "detection_timing_comparison.csv", timings)
    _write(args.output / "detection_position_comparison.csv", positions)
    _write(
        args.output / "m0_negative_control.csv",
        [row for row in timings if row["case_id"] == M0_CASE],
    )
    _write(args.output / "verdict.csv", [verdict])
    _plot_positions(
        args.output / "bootstrap_vs_baseline_detection_position.png",
        bootstrap_runner,
        baseline,
        bootstrap,
    )
    _plot_timeline(
        args.output / "corridor_model_bootstrap_timeline.png",
        baseline,
        bootstrap,
    )
    print(
        f"verdict={verdict['verdict']} output={args.output.resolve()} "
        f"baseline_detection={timings[0]['first_detection_frame']} "
        f"bootstrap_detection={timings[1]['first_detection_frame']} "
        f"replay={replay_match}"
    )


if __name__ == "__main__":
    main()
