"""EXP-045 degraded-start generality and false-suspicion validation.

The EXP-044 advisory is imported unchanged and evaluated over representative
deterministic initial conditions.  Case identity and GT geometry are used only
to construct benchmarks and append post-hoc labels; they never enter the
runtime degraded-state helper or the frozen detector call boundary.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.integration.run_lidar_local_corridor_estimation import (  # noqa: E402
    _rear_start,
)
from junction_detection.integration.run_local_corridor_model_readiness import (  # noqa: E402
    _opening_visible_eval_only,
    add_shadow_readiness,
)
from junction_detection.integration.run_local_only_degraded_start_recovery_shadow import (  # noqa: E402
    add_degraded_start_shadow,
)
from junction_detection.integration.run_pre_corridor_bootstrap_detection_timing import (  # noqa: E402
    _expected_source,
    _fit_sector_side_los_count_eval_only,
    _new_runner,
    run_case,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import (  # noqa: E402
    BASELINE_CORRIDOR_WIDTH,
    LIDAR_MAX_RANGE,
)


EXPERIMENT_ID = "EXP-045"
SEED = 0
OFFSET_MAGNITUDE = 0.1 * BASELINE_CORRIDOR_WIDTH
YAW_MAGNITUDE_DEG = 5.0
DEFAULT_OUTPUT = (
    ROOT
    / "junction_detection/integration/output/degraded_start_generality_validation"
)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    category: str
    map_geometry: str
    start_condition: str
    frames: int = 127
    rear_shift: float = 0.0
    lateral_offset: float = 0.0
    yaw_deg: float = 0.0


def benchmark_cases() -> tuple[CaseSpec, ...]:
    """Return the fixed representative suite; no result-driven sweep."""
    width = BASELINE_CORRIDOR_WIDTH
    return (
        CaseSpec(
            "A_M0_CENTERED",
            "CLEAN_STRAIGHT_START",
            "M0_STRAIGHT",
            "repository centered start",
            frames=600,
        ),
        CaseSpec(
            "A_M0_OFFSET_LEFT",
            "CLEAN_STRAIGHT_START",
            "M0_STRAIGHT",
            "whole swarm shifted left by 0.1W",
            lateral_offset=-OFFSET_MAGNITUDE,
        ),
        CaseSpec(
            "A_M0_OFFSET_RIGHT",
            "CLEAN_STRAIGHT_START",
            "M0_STRAIGHT",
            "whole swarm shifted right by 0.1W",
            lateral_offset=OFFSET_MAGNITUDE,
        ),
        CaseSpec(
            "A_M0_YAW_POS5",
            "CLEAN_STRAIGHT_START",
            "M0_STRAIGHT",
            "existing orientation benchmark +5 deg",
            yaw_deg=YAW_MAGNITUDE_DEG,
        ),
        CaseSpec(
            "A_M0_YAW_NEG5",
            "CLEAN_STRAIGHT_START",
            "M0_STRAIGHT",
            "existing orientation benchmark -5 deg",
            yaw_deg=-YAW_MAGNITUDE_DEG,
        ),
        CaseSpec(
            "B_M1_VERY_NEAR_BASELINE",
            "JUNCTION_DISTANCE_VARIATION",
            "M1_CROSS_BASELINE",
            "repository baseline start",
        ),
        CaseSpec(
            "B_M1_NEAR_HALF_W",
            "JUNCTION_DISTANCE_VARIATION",
            "M1_CROSS_BASELINE",
            "rear shift 0.5W",
            rear_shift=0.5 * width,
        ),
        CaseSpec(
            "B_M1_MEDIUM_ONE_W",
            "JUNCTION_DISTANCE_VARIATION",
            "M1_CROSS_BASELINE",
            "rear shift 1.0W",
            rear_shift=width,
        ),
        CaseSpec(
            "B_M1_CLEAN_FAR_EXP042",
            "JUNCTION_DISTANCE_VARIATION",
            "M1_CROSS_BASELINE",
            "EXP-042 rear shift 160 units",
            rear_shift=160.0,
        ),
        CaseSpec(
            "C_M2_DEFAULT",
            "REPRESENTATIVE_M2_M5",
            "M2_T_JUNCTION",
            "repository default start",
            frames=180,
        ),
        CaseSpec(
            "C_M3_DEFAULT",
            "REPRESENTATIVE_M2_M5",
            "M3_ANGLED_Y",
            "repository default start",
            frames=180,
        ),
        CaseSpec(
            "C_M4_DEFAULT",
            "REPRESENTATIVE_M2_M5",
            "M4_ASYMMETRIC_CROSS",
            "repository default start",
            frames=180,
        ),
        CaseSpec(
            "C_M5_DEFAULT",
            "REPRESENTATIVE_M2_M5",
            "M5_UNEQUAL_WIDTH",
            "repository default start",
            frames=180,
        ),
        CaseSpec(
            "D_M0_LEFT_POS5",
            "NON_JUNCTION_NON_IDEAL_POSE",
            "M0_STRAIGHT",
            "whole swarm -0.1W and body +5 deg",
            lateral_offset=-OFFSET_MAGNITUDE,
            yaw_deg=YAW_MAGNITUDE_DEG,
        ),
        CaseSpec(
            "D_M0_RIGHT_NEG5",
            "NON_JUNCTION_NON_IDEAL_POSE",
            "M0_STRAIGHT",
            "whole swarm +0.1W and body -5 deg",
            lateral_offset=OFFSET_MAGNITUDE,
            yaw_deg=-YAW_MAGNITUDE_DEG,
        ),
    )


def _apply_initial_pose(
    runner: Any, lateral_offset: float, yaw_deg: float
) -> None:
    """Apply benchmark-only initial pose changes without changing control."""
    if lateral_offset:
        for robot in runner.world.robots:
            robot.position[0] += lateral_offset
        if hasattr(runner.world, "initial_front_center_x"):
            runner.world.initial_front_center_x += lateral_offset
    body_yaw = math.radians(90.0 - yaw_deg)
    for robot in runner.world.robots:
        robot.body_yaw_rad = body_yaw
    runner.world.lidar_yaw_deg = math.degrees(body_yaw)
    leader = next(
        robot
        for robot in runner.world.robots
        if robot.robot_id == runner.world.lidar_robot_id
    )
    runner.world.initial_lidar_position = leader.position.copy()


def _run_pose_variant(spec: CaseSpec) -> tuple[Any, list[dict[str, Any]]]:
    """Run the frozen detector with benchmark-only initial pose variation."""
    runner = _new_runner(spec.map_geometry)
    if spec.rear_shift:
        _rear_start(runner, spec.rear_shift)
    _apply_initial_pose(runner, spec.lateral_offset, spec.yaw_deg)
    initial_leader_y = float(runner.world.initial_lidar_position[1])
    lifecycle: list[dict[str, Any]] = []
    for frame in range(spec.frames):
        row = runner.step(frame)
        if row is None:
            continue
        leader = next(
            robot
            for robot in runner.world.robots
            if robot.robot_id == runner.world.lidar_robot_id
        )
        entry = runner.geometry.entrance_y
        distance_to_entry = (
            float(entry - leader.position[1]) if entry is not None else math.nan
        )
        lifecycle.append(
            {
                "case_id": spec.case_id,
                "map_case": spec.map_geometry,
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
                "current_orientation": row[
                    "current_corridor_orientation_deg"
                ],
                "stable_width": row["estimated_corridor_width"],
                "stable_offset": row["estimated_offset"],
                "stable_orientation": row[
                    "stable_corridor_orientation_deg"
                ],
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
                    -distance_to_entry
                    if math.isfinite(distance_to_entry)
                    else math.nan
                ),
            }
        )
    return runner, lifecycle


def run_spec(spec: CaseSpec) -> tuple[Any, list[dict[str, Any]]]:
    """Prefer the exact EXP-042 runner unless pose variation requires a hook."""
    if not spec.lateral_offset and not spec.yaw_deg:
        return run_case(
            spec.case_id,
            spec.map_geometry,
            spec.frames,
            spec.rear_shift,
        )
    return _run_pose_variant(spec)


def _expected_class(
    spec: CaseSpec, lifecycle: list[dict[str, Any]]
) -> str:
    """Assign a post-hoc label; never feed it back to the shadow rule."""
    if spec.map_geometry == "M0_STRAIGHT":
        return "EXPECTED_CLEAN_BOOTSTRAP"
    if _opening_visible_eval_only(lifecycle[0]):
        return "EXPECTED_DEGRADED_BOOTSTRAP"
    if any(row["stable_model_initialized"] for row in lifecycle):
        return "EXPECTED_CLEAN_BOOTSTRAP"
    return "AMBIGUOUS"


def _first(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    return next((row for row in rows if row[key]), None)


def _value(row: dict[str, Any] | None, key: str, default: Any = "") -> Any:
    return default if row is None else row[key]


def _initial_distance(runner: Any) -> float:
    if runner.geometry.entrance_y is None:
        return math.nan
    return float(
        runner.geometry.entrance_y - runner.world.initial_lidar_position[1]
    )


def build_case_rows(
    spec: CaseSpec,
    runner: Any,
    lifecycle: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    readiness = add_shadow_readiness(spec.case_id, lifecycle)
    shadow = add_degraded_start_shadow(spec.case_id, readiness)
    expected = _expected_class(spec, lifecycle)
    distance = _initial_distance(runner)
    case_row = {
        "case_id": spec.case_id,
        "category": spec.category,
        "map_geometry": spec.map_geometry,
        "start_condition": spec.start_condition,
        "start_distance_over_W": (
            distance / BASELINE_CORRIDOR_WIDTH
            if math.isfinite(distance)
            else math.nan
        ),
        "start_distance_over_lidar_range": (
            distance / LIDAR_MAX_RANGE if math.isfinite(distance) else math.nan
        ),
        "lateral_offset_over_W": (
            spec.lateral_offset / BASELINE_CORRIDOR_WIDTH
        ),
        "yaw_deg": spec.yaw_deg,
        "expected_posthoc_class": expected,
        "initial_side_opening_los_beams_eval_only": int(
            lifecycle[0]["gt_side_opening_in_fit_sector_eval_only"]
        ),
        "initial_opening_within_lidar_range_eval_only": (
            _opening_visible_eval_only(lifecycle[0])
        ),
        "seed": SEED,
        "frames": spec.frames,
    }
    consecutive_uninitialized = 0
    timeline: list[dict[str, Any]] = []
    for row in shadow:
        consecutive_uninitialized = (
            0
            if row["stable_model_initialized"]
            else consecutive_uninitialized + 1
        )
        timeline.append(
            {
                "case_id": spec.case_id,
                "category": spec.category,
                "frame": row["frame"],
                "time": row["time"],
                "side_walls_valid": row["side_walls_valid"],
                "parallel_error_deg": row["parallel_error_deg"],
                "current_width": row["current_width"],
                "current_offset": row["current_offset"],
                "current_orientation": row["current_orientation"],
                "stable_width": row["stable_width"],
                "stable_offset": row["stable_offset"],
                "stable_orientation": row["stable_orientation"],
                "stable_update_count": row["stable_update_count"],
                "corridor_model_initialized": row[
                    "stable_model_initialized"
                ],
                "valid_current_model_history_count": row[
                    "valid_current_model_history_count"
                ],
                "consecutive_uninitialized_samples": consecutive_uninitialized,
                "expected_profile_source": row["expected_profile_source"],
                "raw_open_candidate_count": row["open_candidate_count"],
                "raw_open_group_count": row["opening_group_count"],
                "raw_junction_detected": row["junction_detected_raw"],
                "degraded_start_suspected": row[
                    "degraded_start_suspected"
                ],
                "published_junction_detected": row[
                    "junction_detected_published_shadow"
                ],
                "recovery_shadow_state": row["recovery_shadow_state"],
                "runtime_expected_class_available": False,
            }
        )
    return case_row, timeline


def summarize_case(
    case_row: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    valid = _first(rows, "side_walls_valid")
    stable = _first(rows, "corridor_model_initialized")
    degraded = _first(rows, "degraded_start_suspected")
    opening = _first(rows, "raw_open_candidate_count")
    raw = _first(rows, "raw_junction_detected")
    published = _first(rows, "published_junction_detected")
    expected = case_row["expected_posthoc_class"]
    suspected = degraded is not None
    expected_clean = expected == "EXPECTED_CLEAN_BOOTSTRAP"
    expected_degraded = expected == "EXPECTED_DEGRADED_BOOTSTRAP"
    altered = any(
        row["raw_junction_detected"] != row["published_junction_detected"]
        for row in rows
    )
    return {
        "case_id": case_row["case_id"],
        "category": case_row["category"],
        "expected_posthoc_class": expected,
        "stable_model_initialized": stable is not None,
        "first_valid_model_frame": _value(valid, "frame"),
        "stable_init_frame": _value(stable, "frame"),
        "degraded_suspected": suspected,
        "first_degraded_frame": _value(degraded, "frame"),
        "first_open_frame": _value(opening, "frame"),
        "raw_detection_frame": _value(raw, "frame"),
        "published_detection_frame": _value(published, "frame"),
        "false_degraded_eval": bool(expected_clean and suspected),
        "missed_degraded_eval": bool(expected_degraded and not suspected),
        "raw_detection_altered": altered,
        "blocked_raw_detection_samples": sum(
            row["raw_junction_detected"]
            and not row["published_junction_detected"]
            for row in rows
        ),
        "altered_detector_output_samples": sum(
            row["raw_junction_detected"]
            != row["published_junction_detected"]
            for row in rows
        ),
        "final_recovery_shadow_state": rows[-1]["recovery_shadow_state"],
    }


def confusion_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [
        row
        for row in summaries
        if row["expected_posthoc_class"] == "EXPECTED_CLEAN_BOOTSTRAP"
    ]
    degraded = [
        row
        for row in summaries
        if row["expected_posthoc_class"]
        == "EXPECTED_DEGRADED_BOOTSTRAP"
    ]
    ambiguous = [
        row
        for row in summaries
        if row["expected_posthoc_class"] == "AMBIGUOUS"
    ]
    true_degraded = sum(
        row["degraded_suspected"] and not row["missed_degraded_eval"]
        for row in degraded
    )
    false_degraded = sum(row["false_degraded_eval"] for row in clean)
    missed_degraded = sum(row["missed_degraded_eval"] for row in degraded)
    precision_denominator = true_degraded + false_degraded
    recall_denominator = true_degraded + missed_degraded
    return {
        "clean_cases": len(clean),
        "expected_degraded_cases": len(degraded),
        "ambiguous_cases": len(ambiguous),
        "true_degraded": true_degraded,
        "false_degraded": false_degraded,
        "missed_degraded": missed_degraded,
        "precision": (
            true_degraded / precision_denominator
            if precision_denominator
            else math.nan
        ),
        "recall": (
            true_degraded / recall_denominator
            if recall_denominator
            else math.nan
        ),
    }


def preservation_rows(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "case_id": row["case_id"],
            "raw_detection_frame": row["raw_detection_frame"],
            "published_detection_frame": row["published_detection_frame"],
            "blocked_raw_detection_samples": row[
                "blocked_raw_detection_samples"
            ],
            "altered_detector_output_samples": row[
                "altered_detector_output_samples"
            ],
            "raw_detection_altered": row["raw_detection_altered"],
        }
        for row in summaries
    ]


def _signature(rows: list[dict[str, Any]]) -> tuple[Any, ...]:
    return tuple(
        (
            row["frame"],
            row["corridor_model_initialized"],
            row["degraded_start_suspected"],
            row["raw_open_candidate_count"],
            row["raw_open_group_count"],
            row["raw_junction_detected"],
            row["published_junction_detected"],
        )
        for row in rows
    )


def build_verdict(
    summaries: list[dict[str, Any]],
    confusion: dict[str, Any],
    replay_match: bool,
) -> dict[str, Any]:
    by_case = {row["case_id"]: row for row in summaries}
    m0 = by_case["A_M0_CENTERED"]
    baseline = by_case["B_M1_VERY_NEAR_BASELINE"]
    bootstrap = by_case["B_M1_CLEAN_FAR_EXP042"]
    blocked = sum(row["blocked_raw_detection_samples"] for row in summaries)
    altered = sum(row["altered_detector_output_samples"] for row in summaries)
    regressions_pass = bool(
        m0["stable_init_frame"] == 6
        and not m0["degraded_suspected"]
        and m0["raw_detection_frame"] == ""
        and baseline["first_degraded_frame"] == 6
        and baseline["raw_detection_frame"] == 126
        and baseline["published_detection_frame"] == 126
        and bootstrap["stable_init_frame"] == 6
        and bootstrap["first_open_frame"] == 30
        and bootstrap["raw_detection_frame"] == 36
        and not bootstrap["degraded_suspected"]
    )
    if altered or blocked:
        verdict = "E_RAW_DETECTION_REGRESSION"
    elif confusion["false_degraded"]:
        verdict = "B_FALSE_DEGRADED_ON_CLEAN_STARTS"
    elif confusion["missed_degraded"]:
        verdict = "C_MISSED_DEGRADED_STARTS"
    elif regressions_pass and replay_match:
        verdict = "A_DEGRADED_START_GENERALITY_SUPPORTED"
    else:
        verdict = "D_NONBLOCKING_SEMANTICS_GENERAL_BUT_CLASSIFIER_LIMITED"
    return {
        "verdict": verdict,
        "benchmark_case_count": len(summaries),
        "regression_timings_pass": regressions_pass,
        "representative_deterministic_replay_match": replay_match,
        "clean_cases": confusion["clean_cases"],
        "expected_degraded_cases": confusion["expected_degraded_cases"],
        "ambiguous_cases": confusion["ambiguous_cases"],
        "false_degraded": confusion["false_degraded"],
        "missed_degraded": confusion["missed_degraded"],
        "precision": confusion["precision"],
        "recall": confusion["recall"],
        "blocked_raw_detection_samples": blocked,
        "altered_detector_output_samples": altered,
        "exp044_rule_imported_unchanged": True,
        "new_numeric_threshold_added": False,
        "runtime_gt_or_map_used": False,
        "production_changed": False,
        "detector_changed": False,
        "simulator_changed": False,
    }


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_overview(
    path: Path, summaries: list[dict[str, Any]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(12, 8))
    labels = [row["case_id"] for row in summaries]
    y_positions = np.arange(len(labels))
    events = (
        ("stable_init_frame", "READY", "tab:blue", "o"),
        ("first_degraded_frame", "DEGRADED", "tab:orange", "s"),
        ("first_open_frame", "OPEN", "tab:purple", "^"),
        ("raw_detection_frame", "DETECTION", "tab:green", "D"),
    )
    for key, label, color, marker in events:
        x_values = []
        y_values = []
        for y, row in zip(y_positions, summaries):
            value = row[key]
            if value != "":
                x_values.append(float(value))
                y_values.append(y)
        axis.scatter(x_values, y_values, label=label, color=color, marker=marker)
    axis.set(
        xlabel="first event frame",
        ylabel="case",
        yticks=y_positions,
        yticklabels=labels,
        title="EXP-045 degraded-start generality overview",
    )
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_ready_vs_degraded(
    path: Path, summaries: list[dict[str, Any]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["case_id"] for row in summaries]
    ready = [
        float(row["stable_init_frame"])
        if row["stable_init_frame"] != ""
        else math.nan
        for row in summaries
    ]
    degraded = [
        float(row["first_degraded_frame"])
        if row["first_degraded_frame"] != ""
        else math.nan
        for row in summaries
    ]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(13, 6))
    axis.scatter(x, ready, label="READY", color="tab:blue", marker="o")
    axis.scatter(
        x,
        degraded,
        label="DEGRADED suspected",
        color="tab:orange",
        marker="s",
    )
    axis.set(
        xlabel="case",
        ylabel="first event frame",
        xticks=x,
        xticklabels=labels,
        title="READY vs DEGRADED timing",
    )
    axis.tick_params(axis="x", rotation=65)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--limit-cases", type=int, default=0)
    args = parser.parse_args(argv)

    specs = list(benchmark_cases())
    if args.limit_cases:
        specs = specs[: args.limit_cases]
    case_rows: list[dict[str, Any]] = []
    timelines: dict[str, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    for spec in specs:
        runner, lifecycle = run_spec(spec)
        case_row, timeline = build_case_rows(spec, runner, lifecycle)
        case_rows.append(case_row)
        timelines[spec.case_id] = timeline
        summaries.append(summarize_case(case_row, timeline))

    replay_match = False
    if not args.skip_replay and len(specs) == len(benchmark_cases()):
        replay_ids = (
            "B_M1_VERY_NEAR_BASELINE",
            "A_M0_YAW_POS5",
            "C_M3_DEFAULT",
        )
        replay_matches = []
        by_spec = {spec.case_id: spec for spec in specs}
        for case_id in replay_ids:
            spec = by_spec[case_id]
            runner, lifecycle = run_spec(spec)
            _, replay = build_case_rows(spec, runner, lifecycle)
            replay_matches.append(_signature(timelines[case_id]) == _signature(replay))
        replay_match = all(replay_matches)

    confusion = confusion_summary(summaries)
    preservation = preservation_rows(summaries)
    final_verdict = build_verdict(summaries, confusion, replay_match)
    all_timeline_rows = [
        row for spec in specs for row in timelines[spec.case_id]
    ]
    _write(args.output / "generality_cases.csv", case_rows)
    _write(args.output / "degraded_start_timeline.csv", all_timeline_rows)
    _write(args.output / "generality_case_summary.csv", summaries)
    _write(args.output / "confusion_summary.csv", [confusion])
    _write(args.output / "raw_detection_preservation.csv", preservation)
    _write(args.output / "verdict.csv", [final_verdict])
    _plot_overview(
        args.output / "degraded_start_generality_overview.png", summaries
    )
    _plot_ready_vs_degraded(
        args.output / "degraded_vs_ready_timing.png", summaries
    )
    print(
        f"verdict={final_verdict['verdict']} cases={len(specs)} "
        f"clean={confusion['clean_cases']} "
        f"degraded={confusion['expected_degraded_cases']} "
        f"false={confusion['false_degraded']} "
        f"missed={confusion['missed_degraded']} "
        f"blocked={final_verdict['blocked_raw_detection_samples']} "
        f"altered={final_verdict['altered_detector_output_samples']} "
        f"replay={replay_match} output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
