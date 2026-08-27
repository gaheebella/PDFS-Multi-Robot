"""EXP-046 early Junction detection to provisional-anchor placement audit.

The frozen EXP-041 AnchorSession supplies the existing detection, propulsion
gate, braking, stationary dwell, and provisional-anchor semantics.  EXP-042's
rear-start helper changes only the bootstrap initial observation condition.
Map/GT geometry and the stationary Point Cloud call are post-hoc evaluation
only and never affect detector or control transitions.
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
from junction_detection.integration.run_provisional_anchor_pointcloud_observation import (  # noqa: E402
    _snapshot as _pointcloud_snapshot,
    evaluate_snapshot,
)
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import (  # noqa: E402
    detect_openings,
)
from pygame_simulator.lidar_junction_provisional_anchor_visualizer import (  # noqa: E402
    AnchorSession,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import (  # noqa: E402
    MIN_SPEED,
)


EXPERIMENT_ID = "EXP-046"
BASELINE_CASE = "M1_BASELINE"
BOOTSTRAP_CASE = "M1_PRE_CORRIDOR_BOOTSTRAP"
M0_CASE = "M0_NEGATIVE_CONTROL"
DEFAULT_OUTPUT = (
    ROOT / "junction_detection/integration/output/early_detection_anchor_placement"
)


def _leader(session: AnchorSession) -> Any:
    return next(
        robot
        for robot in session.runner.world.robots
        if robot.robot_id == session.runner.world.lidar_robot_id
    )


def _corridor_model_state(session: AnchorSession) -> str:
    profile = session.runner.last_profile_result
    if profile is None:
        return "MODEL_UNINITIALIZED"
    if profile["corridor_model_initialized"]:
        return "MODEL_READY"
    if profile["side_walls_valid"]:
        return "MODEL_BOOTSTRAPPING"
    return "MODEL_UNINITIALIZED"


def _profile_counts(session: AnchorSession) -> tuple[int, int, bool]:
    profile = session.runner.last_profile_result
    if profile is None:
        return 0, 0, False
    return (
        int(np.count_nonzero(profile["open_candidate_mask"])),
        int(profile["opening_group_count"]),
        bool(profile["profile_junction_detected"]),
    )


def _eval_position(session: AnchorSession) -> dict[str, float | bool]:
    leader = _leader(session)
    x, y = float(leader.position[0]), float(leader.position[1])
    geometry = session.runner.geometry
    if geometry.entrance_y is None:
        return {
            "distance_to_junction_entry_eval_only": math.nan,
            "distance_to_junction_center_eval_only": math.nan,
            "longitudinal_offset_from_center_eval_only": math.nan,
            "lateral_offset_from_center_eval_only": math.nan,
            "inside_junction_region_eval_only": False,
        }
    half = 0.5 * float(geometry.junction_size)
    return {
        "distance_to_junction_entry_eval_only": float(
            geometry.entrance_y - y
        ),
        "distance_to_junction_center_eval_only": math.hypot(x, y),
        "longitudinal_offset_from_center_eval_only": y,
        "lateral_offset_from_center_eval_only": x,
        "inside_junction_region_eval_only": bool(
            abs(x) <= half + 1.0e-9 and abs(y) <= half + 1.0e-9
        ),
    }


def _event_row(
    case_id: str,
    session: AnchorSession,
    frame: int,
    event: str,
) -> dict[str, Any]:
    leader = _leader(session)
    candidates, groups, detected = _profile_counts(session)
    return {
        "case_id": case_id,
        "frame": frame,
        "time": float(session.runner.world.time),
        "event": event,
        "leader_x": float(leader.position[0]),
        "leader_y": float(leader.position[1]),
        "speed": float(np.linalg.norm(leader.velocity)),
        "corridor_model_state": _corridor_model_state(session),
        "open_candidate_count": candidates,
        "opening_group_count": groups,
        "junction_detected": detected,
        "braking_active": bool(session.runner.world.braking_active),
        "provisional_anchor": bool(
            session.runner.world.provisional_fixed_anchor
        ),
        **_eval_position(session),
    }


def _trace_row(case_id: str, session: AnchorSession, frame: int) -> dict[str, Any]:
    leader = _leader(session)
    candidates, groups, detected = _profile_counts(session)
    evaluation = _eval_position(session)
    return {
        "case_id": case_id,
        "frame": frame,
        "time": float(session.runner.world.time),
        "leader_x_eval_only": float(leader.position[0]),
        "leader_y_eval_only": float(leader.position[1]),
        "leader_speed": float(np.linalg.norm(leader.velocity)),
        "corridor_model_state": _corridor_model_state(session),
        "open_candidate_count": candidates,
        "opening_group_count": groups,
        "junction_detected": detected,
        "junction_detection_latched": bool(
            session.runner.world.junction_detection_latched
        ),
        "braking_active": bool(session.runner.world.braking_active),
        "leader_stationary": bool(
            np.linalg.norm(leader.velocity) < MIN_SPEED
        ),
        "stationary_dwell_steps": int(
            session.runner.world.stationary_dwell_steps
        ),
        "provisional_anchor": bool(
            session.runner.world.provisional_fixed_anchor
        ),
        **evaluation,
    }


def run_anchor_case(
    case_id: str,
    map_case: str,
    frames: int,
    rear_start: bool = False,
) -> dict[str, Any]:
    """Run one frozen active-anchor session and record physical-frame edges."""
    session = AnchorSession(map_case)
    if rear_start:
        _rear_start(session.runner, REAR_START_SHIFT)
    events: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    first: dict[str, dict[str, Any]] = {}
    braking_seen = False
    anchor_seen = False
    hold_confirmed_seen = False
    previous_session_event_count = 0

    for frame in range(frames):
        session.advance_physics_frame()
        trace.append(_trace_row(case_id, session, frame))
        if frame == 0:
            event = _event_row(case_id, session, frame, "START")
            events.append(event)
            first["START"] = event

        sampled = bool(
            session.runner.rows and session.runner.rows[-1]["frame"] == frame
        )
        if sampled:
            row = session.runner.rows[-1]
            sampled_events = (
                (
                    "MODEL_READY",
                    bool(row["corridor_model_initialized"]),
                ),
                ("FIRST_OPEN", int(row["opening_candidate_count"]) > 0),
                (
                    "JUNCTION_DETECTION",
                    bool(row["profile_junction_detected"]),
                ),
            )
            for event_name, condition in sampled_events:
                if condition and event_name not in first:
                    event = _event_row(case_id, session, frame, event_name)
                    events.append(event)
                    first[event_name] = event

        world = session.runner.world
        if world.braking_active and not braking_seen:
            event = _event_row(case_id, session, frame, "BRAKE_TRIGGER")
            events.append(event)
            first["BRAKE_TRIGGER"] = event
            braking_seen = True
        if (
            braking_seen
            and "SPEED_BELOW_THRESHOLD" not in first
            and float(np.linalg.norm(_leader(session).velocity)) < MIN_SPEED
        ):
            event = _event_row(
                case_id, session, frame, "SPEED_BELOW_THRESHOLD"
            )
            events.append(event)
            first["SPEED_BELOW_THRESHOLD"] = event
        if world.provisional_fixed_anchor and not anchor_seen:
            event = _event_row(
                case_id, session, int(world.anchor_entry_frame),
                "PROVISIONAL_ANCHOR_ENTER",
            )
            event["time"] = float(world.anchor_entry_time)
            events.append(event)
            first["PROVISIONAL_ANCHOR_ENTER"] = event
            anchor_seen = True

        new_session_events = session.events[previous_session_event_count:]
        previous_session_event_count = len(session.events)
        if (
            not hold_confirmed_seen
            and any(
                row["event"] == "ANCHOR_HOLD_CONFIRMED"
                for row in new_session_events
            )
        ):
            event = _event_row(
                case_id, session, frame, "ANCHOR_HOLD_CONFIRMED"
            )
            events.append(event)
            first["ANCHOR_HOLD_CONFIRMED"] = event
            hold_confirmed_seen = True

    return {
        "case_id": case_id,
        "session": session,
        "events": events,
        "trace": trace,
        "first": first,
    }


def _event_value(run: dict[str, Any], event: str, key: str) -> Any:
    row = run["first"].get(event)
    return "" if row is None else row[key]


def _distance_between(
    run: dict[str, Any], first_event: str, second_event: str
) -> float:
    first = run["first"].get(first_event)
    second = run["first"].get(second_event)
    if first is None or second is None:
        return math.nan
    return math.hypot(
        float(second["leader_x"]) - float(first["leader_x"]),
        float(second["leader_y"]) - float(first["leader_y"]),
    )


def case_summary(run: dict[str, Any]) -> dict[str, Any]:
    session = run["session"]
    anchor = run["first"].get("PROVISIONAL_ANCHOR_ENTER")
    detection = run["first"].get("JUNCTION_DETECTION")
    return {
        "case_id": run["case_id"],
        "frame_first_model_ready": _event_value(
            run, "MODEL_READY", "frame"
        ),
        "time_first_model_ready": _event_value(run, "MODEL_READY", "time"),
        "frame_first_open": _event_value(run, "FIRST_OPEN", "frame"),
        "time_first_open": _event_value(run, "FIRST_OPEN", "time"),
        "frame_first_detection": _event_value(
            run, "JUNCTION_DETECTION", "frame"
        ),
        "time_first_detection": _event_value(
            run, "JUNCTION_DETECTION", "time"
        ),
        "frame_brake_trigger": _event_value(run, "BRAKE_TRIGGER", "frame"),
        "time_brake_trigger": _event_value(run, "BRAKE_TRIGGER", "time"),
        "frame_speed_below_threshold": _event_value(
            run, "SPEED_BELOW_THRESHOLD", "frame"
        ),
        "time_speed_below_threshold": _event_value(
            run, "SPEED_BELOW_THRESHOLD", "time"
        ),
        "frame_provisional_anchor_enter": _event_value(
            run, "PROVISIONAL_ANCHOR_ENTER", "frame"
        ),
        "time_provisional_anchor_enter": _event_value(
            run, "PROVISIONAL_ANCHOR_ENTER", "time"
        ),
        "frame_anchor_hold_confirmed": _event_value(
            run, "ANCHOR_HOLD_CONFIRMED", "frame"
        ),
        "time_anchor_hold_confirmed": _event_value(
            run, "ANCHOR_HOLD_CONFIRMED", "time"
        ),
        "detection_x_eval_only": (
            "" if detection is None else detection["leader_x"]
        ),
        "detection_y_eval_only": (
            "" if detection is None else detection["leader_y"]
        ),
        "anchor_x_eval_only": "" if anchor is None else anchor["leader_x"],
        "anchor_y_eval_only": "" if anchor is None else anchor["leader_y"],
        "anchor_inside_junction_eval_only": (
            False
            if anchor is None
            else anchor["inside_junction_region_eval_only"]
        ),
        "anchor_distance_to_entry_eval_only": (
            math.nan
            if anchor is None
            else anchor["distance_to_junction_entry_eval_only"]
        ),
        "anchor_distance_to_center_eval_only": (
            math.nan
            if anchor is None
            else anchor["distance_to_junction_center_eval_only"]
        ),
        "anchor_transition_count": session.anchor_transition_count,
        "pointcloud_detector_integrated_runtime": False,
    }


def braking_metrics(run: dict[str, Any]) -> dict[str, Any]:
    detection_frame = _event_value(run, "JUNCTION_DETECTION", "frame")
    stop_frame = _event_value(run, "SPEED_BELOW_THRESHOLD", "frame")
    detection_time = _event_value(run, "JUNCTION_DETECTION", "time")
    stop_time = _event_value(run, "SPEED_BELOW_THRESHOLD", "time")
    return {
        "case_id": run["case_id"],
        "detection_frame": detection_frame,
        "detection_time": detection_time,
        "detection_speed": _event_value(run, "JUNCTION_DETECTION", "speed"),
        "brake_start_frame": _event_value(run, "BRAKE_TRIGGER", "frame"),
        "brake_start_time": _event_value(run, "BRAKE_TRIGGER", "time"),
        "brake_start_speed": _event_value(run, "BRAKE_TRIGGER", "speed"),
        "stop_frame": stop_frame,
        "stop_time": stop_time,
        "stop_distance_from_detection": _distance_between(
            run, "JUNCTION_DETECTION", "SPEED_BELOW_THRESHOLD"
        ),
        "stop_distance_from_brake": _distance_between(
            run, "BRAKE_TRIGGER", "SPEED_BELOW_THRESHOLD"
        ),
        "stop_latency_frames": (
            ""
            if detection_frame == "" or stop_frame == ""
            else int(stop_frame) - int(detection_frame)
        ),
        "stop_latency_sec": (
            ""
            if detection_time == "" or stop_time == ""
            else float(stop_time) - float(detection_time)
        ),
        "detection_to_anchor_distance": _distance_between(
            run, "JUNCTION_DETECTION", "PROVISIONAL_ANCHOR_ENTER"
        ),
        "detection_to_anchor_latency_frames": (
            ""
            if detection_frame == ""
            or _event_value(run, "PROVISIONAL_ANCHOR_ENTER", "frame") == ""
            else int(_event_value(run, "PROVISIONAL_ANCHOR_ENTER", "frame"))
            - int(detection_frame)
        ),
    }


def anchor_position_eval(run: dict[str, Any]) -> dict[str, Any]:
    anchor = run["first"].get("PROVISIONAL_ANCHOR_ENTER")
    if anchor is None:
        return {
            "case_id": run["case_id"],
            "anchor_x_eval_only": math.nan,
            "anchor_y_eval_only": math.nan,
            "inside_junction_region_eval_only": False,
            "distance_to_entry_eval_only": math.nan,
            "distance_to_center_eval_only": math.nan,
            "longitudinal_offset_from_center_eval_only": math.nan,
            "lateral_offset_from_center_eval_only": math.nan,
        }
    return {
        "case_id": run["case_id"],
        "anchor_x_eval_only": anchor["leader_x"],
        "anchor_y_eval_only": anchor["leader_y"],
        "inside_junction_region_eval_only": anchor[
            "inside_junction_region_eval_only"
        ],
        "distance_to_entry_eval_only": anchor[
            "distance_to_junction_entry_eval_only"
        ],
        "distance_to_center_eval_only": anchor[
            "distance_to_junction_center_eval_only"
        ],
        "longitudinal_offset_from_center_eval_only": anchor[
            "longitudinal_offset_from_center_eval_only"
        ],
        "lateral_offset_from_center_eval_only": anchor[
            "lateral_offset_from_center_eval_only"
        ],
    }


def stationary_visibility_eval(run: dict[str, Any]) -> dict[str, Any]:
    session = run["session"]
    if not session.runner.world.provisional_fixed_anchor:
        return {
            "case_id": run["case_id"],
            "visible_outgoing_branch_count_eval_only": 0,
            "gt_outgoing_branch_count_eval_only": 0,
            "visible_side_opening_count_eval_only": 0,
            "stationary_opening_group_count": 0,
            "angular_coverage_deg": 0.0,
            "occlusion_notes": "NO_PROVISIONAL_ANCHOR",
            "pointcloud_runtime_integrated": False,
        }
    snapshot = _pointcloud_snapshot(
        session.runner, "STATIONARY_ANCHOR_POSTHOC_EVAL_ONLY"
    )
    openings = list(detect_openings(snapshot["angles"], snapshot["ranges"]))
    summary, opening_rows = evaluate_snapshot(
        session.runner, snapshot, openings
    )
    side_ids = {
        index
        for index, branch in enumerate(session.runner.geometry.branches)
        if abs(float(branch.angle_deg)) >= 45.0
    }
    visible_side = sum(
        isinstance(row["matched_GT_branch_eval_only"], int)
        and row["matched_GT_branch_eval_only"] in side_ids
        for row in opening_rows
    )
    missed = int(summary["missed_outgoing_count_eval_only"])
    return {
        "case_id": run["case_id"],
        "visible_outgoing_branch_count_eval_only": int(
            summary["matched_outgoing_count_eval_only"]
        ),
        "gt_outgoing_branch_count_eval_only": int(
            summary["GT_outgoing_branch_count_eval_only"]
        ),
        "visible_side_opening_count_eval_only": visible_side,
        "stationary_opening_group_count": len(openings),
        "angular_coverage_deg": float(
            sum(float(opening["width_deg"]) for opening in openings)
        ),
        "occlusion_notes": (
            "ALL_OUTGOING_VISIBLE"
            if missed == 0
            else f"PARTIAL_{missed}_OUTGOING_MISSED"
        ),
        "pointcloud_runtime_integrated": False,
    }


def _signature(run: dict[str, Any]) -> tuple[Any, ...]:
    summary = case_summary(run)
    braking = braking_metrics(run)
    return (
        tuple(summary.values()),
        tuple(braking.values()),
        tuple(
            (
                row["frame"],
                row["junction_detection_latched"],
                row["braking_active"],
                row["provisional_anchor"],
            )
            for row in run["trace"]
        ),
    )


def build_verdict(
    summaries: list[dict[str, Any]],
    visibility: list[dict[str, Any]],
    replay_match: bool,
) -> dict[str, Any]:
    summary = {row["case_id"]: row for row in summaries}
    visible = {row["case_id"]: row for row in visibility}
    baseline = summary[BASELINE_CASE]
    bootstrap = summary[BOOTSTRAP_CASE]
    m0 = summary[M0_CASE]
    regression = bool(
        baseline["frame_first_detection"] == 126
        and baseline["frame_provisional_anchor_enter"] == 168
        and bootstrap["frame_first_model_ready"] == 6
        and bootstrap["frame_first_open"] == 30
        and bootstrap["frame_first_detection"] == 36
        and not m0["anchor_transition_count"]
        and m0["frame_first_detection"] == ""
    )
    bootstrap_before_entry = bool(
        float(bootstrap["anchor_distance_to_entry_eval_only"]) > 0.0
    )
    bootstrap_inside = bool(bootstrap["anchor_inside_junction_eval_only"])
    all_visible = bool(
        visible[BOOTSTRAP_CASE]["visible_outgoing_branch_count_eval_only"]
        == visible[BOOTSTRAP_CASE]["gt_outgoing_branch_count_eval_only"]
    )
    if bootstrap_before_entry:
        verdict = "B_EARLY_DETECTION_STOPS_TOO_EARLY"
    elif not bootstrap_inside and float(
        bootstrap["anchor_y_eval_only"]
    ) > 0.5 * 84.0:
        verdict = "C_EARLY_DETECTION_STILL_OVERSHOOTS"
    elif bootstrap_inside and not all_visible:
        verdict = "D_ANCHOR_POSITION_NOT_OBSERVATION_USABLE"
    elif bootstrap_inside and all_visible:
        verdict = "E_BASELINE_LATE_BUT_BOOTSTRAP_RECOVERS"
    else:
        verdict = "A_EARLY_DETECTION_YIELDS_USABLE_ANCHOR"
    return {
        "verdict": verdict,
        "baseline_visualizer_equivalent": bool(
            baseline["frame_first_detection"] == 126
            and baseline["frame_provisional_anchor_enter"] == 168
        ),
        "exp042_bootstrap_timing_equivalent": bool(
            bootstrap["frame_first_model_ready"] == 6
            and bootstrap["frame_first_open"] == 30
            and bootstrap["frame_first_detection"] == 36
        ),
        "m0_negative_control_pass": bool(
            m0["frame_first_detection"] == ""
            and not m0["anchor_transition_count"]
        ),
        "all_regressions_pass": regression,
        "deterministic_replay_match": replay_match,
        "bootstrap_anchor_before_entry": bootstrap_before_entry,
        "bootstrap_anchor_inside_junction": bootstrap_inside,
        "bootstrap_all_outgoing_visible_eval_only": all_visible,
        "runtime_gt_or_map_used": False,
        "stationary_pointcloud_runtime_integrated": False,
        "detector_changed": False,
        "braking_changed": False,
        "simulator_changed": False,
    }


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_trajectory(
    path: Path,
    baseline: dict[str, Any],
    bootstrap: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 9))
    geometry = baseline["session"].runner.geometry
    for wall_index, wall in enumerate(geometry.walls):
        start, end = wall
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="black",
            linewidth=1.2,
            label="walls" if wall_index == 0 else None,
        )
    colors = {BASELINE_CASE: "tab:red", BOOTSTRAP_CASE: "tab:blue"}
    for run in (baseline, bootstrap):
        trace = run["trace"]
        axis.plot(
            [row["leader_x_eval_only"] for row in trace],
            [row["leader_y_eval_only"] for row in trace],
            color=colors[run["case_id"]],
            label=run["case_id"],
        )
        for event_name, marker in (
            ("JUNCTION_DETECTION", "o"),
            ("BRAKE_TRIGGER", "s"),
            ("PROVISIONAL_ANCHOR_ENTER", "*"),
        ):
            event = run["first"].get(event_name)
            if event is not None:
                axis.scatter(
                    event["leader_x"],
                    event["leader_y"],
                    color=colors[run["case_id"]],
                    marker=marker,
                    s=100,
                )
    entrance = float(geometry.entrance_y)
    axis.axhline(
        entrance, color="tab:orange", linestyle="--", label="entry EVAL ONLY"
    )
    axis.scatter([0.0], [0.0], marker="x", s=90, label="center EVAL ONLY")
    axis.set(
        xlabel="world x [eval only]",
        ylabel="world y [eval only]",
        title="Baseline vs bootstrap stopping trajectory — GT overlay EVAL ONLY",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.22)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_event_positions(
    path: Path,
    baseline: dict[str, Any],
    bootstrap: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    events = (
        "JUNCTION_DETECTION",
        "BRAKE_TRIGGER",
        "SPEED_BELOW_THRESHOLD",
        "PROVISIONAL_ANCHOR_ENTER",
    )
    figure, axis = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(events))
    for offset, run, color in (
        (-0.12, baseline, "tab:red"),
        (0.12, bootstrap, "tab:blue"),
    ):
        values = [
            float(run["first"][event]["leader_y"])
            if event in run["first"]
            else math.nan
            for event in events
        ]
        axis.scatter(x + offset, values, label=run["case_id"], color=color)
        axis.plot(x + offset, values, color=color, alpha=0.7)
    axis.axhline(-42.0, linestyle="--", color="tab:orange", label="entry eval")
    axis.axhline(0.0, linestyle=":", color="black", label="center eval")
    axis.set(
        xticks=x,
        xticklabels=events,
        ylabel="leader y [eval only]",
        title="Detection → brake → stop → anchor positions",
    )
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_speed(
    path: Path,
    baseline: dict[str, Any],
    bootstrap: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5.5))
    colors = {BASELINE_CASE: "tab:red", BOOTSTRAP_CASE: "tab:blue"}
    for run in (baseline, bootstrap):
        trace = run["trace"]
        axis.plot(
            [
                -float(row["distance_to_junction_entry_eval_only"])
                for row in trace
            ],
            [row["leader_speed"] for row in trace],
            color=colors[run["case_id"]],
            label=run["case_id"],
        )
        for event_name, marker in (
            ("JUNCTION_DETECTION", "o"),
            ("BRAKE_TRIGGER", "s"),
            ("SPEED_BELOW_THRESHOLD", "x"),
        ):
            event = run["first"].get(event_name)
            if event is not None:
                axis.scatter(
                    -float(event["distance_to_junction_entry_eval_only"]),
                    event["speed"],
                    color=colors[run["case_id"]],
                    marker=marker,
                    s=80,
                )
    axis.axvline(0.0, color="tab:orange", linestyle="--", label="entry eval")
    axis.set(
        xlabel="longitudinal progress relative to entry [eval only]",
        ylabel="leader speed",
        title="Speed vs Junction-relative progress",
    )
    axis.grid(alpha=0.22)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--m0-frames", type=int, default=600)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args(argv)

    baseline = run_anchor_case(
        BASELINE_CASE, "M1_CROSS_BASELINE", args.frames
    )
    bootstrap = run_anchor_case(
        BOOTSTRAP_CASE,
        "M1_CROSS_BASELINE",
        args.frames,
        rear_start=True,
    )
    m0 = run_anchor_case(M0_CASE, "M0_STRAIGHT", args.m0_frames)
    runs = [baseline, bootstrap, m0]
    summaries = [case_summary(run) for run in runs]
    braking = [braking_metrics(run) for run in runs]
    positions = [anchor_position_eval(run) for run in runs]
    visibility = [stationary_visibility_eval(run) for run in runs]

    replay_match = False
    if not args.skip_replay:
        baseline_replay = run_anchor_case(
            BASELINE_CASE, "M1_CROSS_BASELINE", args.frames
        )
        bootstrap_replay = run_anchor_case(
            BOOTSTRAP_CASE,
            "M1_CROSS_BASELINE",
            args.frames,
            rear_start=True,
        )
        replay_match = bool(
            _signature(baseline) == _signature(baseline_replay)
            and _signature(bootstrap) == _signature(bootstrap_replay)
        )

    final_verdict = build_verdict(summaries, visibility, replay_match)
    _write(args.output / "anchor_placement_case_summary.csv", summaries)
    _write(
        args.output / "event_timeline.csv",
        [event for run in runs for event in run["events"]],
    )
    _write(args.output / "braking_metrics.csv", braking)
    _write(args.output / "anchor_position_eval.csv", positions)
    _write(args.output / "stationary_visibility_eval.csv", visibility)
    _write(args.output / "verdict.csv", [final_verdict])
    _plot_trajectory(
        args.output / "baseline_vs_bootstrap_stop_trajectory.png",
        baseline,
        bootstrap,
    )
    _plot_event_positions(
        args.output / "detection_brake_anchor_positions.png",
        baseline,
        bootstrap,
    )
    _plot_speed(
        args.output / "speed_vs_progress.png", baseline, bootstrap
    )
    print(
        f"verdict={final_verdict['verdict']} "
        f"baseline_detection={summaries[0]['frame_first_detection']} "
        f"baseline_anchor={summaries[0]['frame_provisional_anchor_enter']} "
        f"bootstrap_detection={summaries[1]['frame_first_detection']} "
        f"bootstrap_anchor={summaries[1]['frame_provisional_anchor_enter']} "
        f"replay={replay_match} output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
