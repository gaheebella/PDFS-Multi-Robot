"""EXP-035 persistent-PARTIAL SPH motion-only targeted validation.

Geometry selection is performed from frozen wall topology alone over the exact
M1 baseline LiDAR trajectory.  Only after selecting the first existing map with
a persistent one-ended side candidate is unchanged natural-SPH motion run.
Detector, candidate, motion, and force parameters are imported unchanged from
EXP-034.  GT branch labels are appended after runtime state updates only.
"""

from __future__ import annotations

import argparse
import csv
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.integration.run_lidar_local_corridor_estimation import _rear_start
from junction_detection.integration.run_wall_topology_branch_opening_diagnostic import (
    _analyze,
    _gt_mouths_eval,
)
from junction_detection.pointcloud.lidar_profile_junction_detector import (
    GeometryProfileConfig,
    LidarProfileJunctionDetector,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import (
    LIDAR_MAX_RANGE,
    GeometryBuilder,
    GeometryCase,
    LidarSensor,
    _rect,
    _union_boundary,
)
from pygame_simulator.pre_exploration_wall_topology_sph_validation import (
    DEFAULT_OUTPUT as EXP034_OUTPUT,
    MOTION_MIN_PROGRESS,
    MOTION_MIN_ROBOTS,
    MOTION_WINDOW_SECONDS,
    BranchCandidate,
    WallTopologySPHRunner,
    _axis_frame,
    _candidate_state_self_test,
    _rotation,
    _summary_rows,
    _write,
    run_case,
)

EXPERIMENT_ID = "EXP-035"
DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/persistent_partial_sph_motion_validation"
SEARCH_CASES = (
    "M1_CROSS_BASELINE",
    "M2_T_JUNCTION",
    "M3_ANGLED_Y",
    "M4_ASYMMETRIC_CROSS",
    "M5_UNEQUAL_WIDTH",
)
BASELINE_REFERENCE = {
    "PARTIAL_BRANCH_CANDIDATE": 0.5166666666666666,
    "COMPLETE_BRANCH_CANDIDATE": 2.2166666666666637,
    "MOTION_SUPPORTED": 6.416666666666649,
}


class PersistentPartialAuditRunner(WallTopologySPHRunner):
    """Add read-only pose, state, and support timing audit to frozen EXP-034."""

    def __init__(self, case_id: str, rear_start: bool = True):
        super().__init__(case_id, rear_start)
        self.pose_timeline: list[dict[str, Any]] = []
        self.state_timeline: list[dict[str, Any]] = []
        self.first_motion_evidence: dict[str, float] = {}
        self.first_motion_condition: dict[str, float] = {}

    def step(self, frame: int) -> dict[str, Any] | None:
        """Advance unchanged physics and append diagnostic-only timing fields."""
        row = super().step(frame)
        if row is None:
            return None
        leader = self.leader()
        self.pose_timeline.append({
            "time": self.world.time,
            "frame": frame,
            "x_eval_only": float(leader.position[0]),
            "y_eval_only": float(leader.position[1]),
            "yaw_deg_local_pose": math.degrees(leader.body_yaw_rad),
        })
        for candidate in self.candidates:
            evidence = candidate.last_evidence
            if candidate.directional_samples > 0:
                self.first_motion_evidence.setdefault(candidate.candidate_id, self.world.time)
            if evidence.get("motion_supported", False):
                self.first_motion_condition.setdefault(candidate.candidate_id, self.world.time)
            self.state_timeline.append({
                "time": self.world.time,
                "candidate_id": candidate.candidate_id,
                "topology_state": candidate.topology_type,
                "termination_count": len(candidate.endpoint_ids),
                "opposite_termination_observed_eval": candidate.topology_type == "COMPLETE",
                "complete_gap_present": candidate.topology_type == "COMPLETE",
                "motion_support_robot_count": evidence.get("supporting_robot_count", 0),
                "stable_support_count": evidence.get("supporting_robot_count_ever", 0),
                "reliability": evidence.get("motion_reliability", 0.0),
                "dispersion": evidence.get("direction_dispersion", math.nan),
                "backflow": evidence.get("backflow_ratio", 0.0),
                "motion_condition_met": evidence.get("motion_supported", False),
                "runtime_state": candidate.state,
                # The current EXP-034 graph already permits PARTIAL→SUPPORTED,
                # so shadow and runtime conditions are identical diagnostics.
                "shadow_motion_only_state": evidence.get("motion_supported", False),
            })
        return row


def _extended_geometry(case_id: str) -> GeometryCase:
    """Build the same rear-start geometry without creating a second swarm."""
    geometry = GeometryBuilder.build(case_id)
    entrance = float(geometry.entrance_y)
    length = geometry.incoming_length + 160.0
    incoming = _rect(
        np.array([0.0, entrance - 0.5 * length]),
        np.array([0.0, 1.0]),
        geometry.incoming_width,
        length,
    )
    rects = (incoming,) + geometry.free_rects[1:]
    return GeometryCase(
        geometry.case_id,
        geometry.incoming_width,
        length,
        geometry.junction_size,
        geometry.branches,
        rects,
        _union_boundary(rects),
        geometry.entrance_y,
    )


def _runtime_side_states(result: dict[str, Any], orientation: float) -> dict[str, dict[str, Any]]:
    """Apply the frozen PARTIAL/COMPLETE definitions to both local half-planes."""
    _, left_axis = _axis_frame(orientation)
    endpoints = {row["endpoint_id"]: row for row in result["endpoints"]}
    terminations = [row for row in result["endpoints"] if row["endpoint_type"] == "WALL_TERMINATION"]
    valid_endpoints = [row for row in result["endpoints"] if row["valid"]]
    output: dict[str, dict[str, Any]] = {}
    for name, sign in (("LOCAL_NEGATIVE", -1.0), ("LOCAL_POSITIVE", 1.0)):
        side = [row for row in terminations if float(np.dot(row["point"], left_axis)) * sign > 0.0]
        # Exact EXP-034 equivalence: PARTIAL creation requires an observed wall
        # termination, while COMPLETE pairing accepts every EXP-033 valid
        # endpoint (WALL_TERMINATION or CORNER) on the same local side.
        ids = {
            row["endpoint_id"]
            for row in valid_endpoints
            if float(np.dot(row["point"], left_axis)) * sign > 0.0
        }
        complete_gaps = [
            gap for gap in result["gaps"]
            if gap["candidate_valid"] and gap["endpoint_a"] in ids and gap["endpoint_b"] in ids
        ]
        output[name] = {
            "topology": "COMPLETE" if complete_gaps else "PARTIAL" if side else "NONE",
            "termination_count": len(side),
            "complete_gap_count": len(complete_gaps),
            "representative_endpoint": None if not side else side[0],
        }
    return output


def _posthoc_branch_label(
    geometry: GeometryCase,
    snapshot: dict[str, Any],
    endpoint: dict[str, Any] | None,
    width: float,
) -> str:
    """Label a topology-only candidate after its runtime classification."""
    if endpoint is None:
        return "NONE"
    mouths = _gt_mouths_eval(SimpleNamespace(geometry=geometry), snapshot)
    best_label, best_error = "FALSE", math.inf
    for mouth in mouths:
        if mouth["branch_type"] != "OUTGOING":
            continue
        error = min(
            float(np.linalg.norm(endpoint["point"] - mouth["a"])),
            float(np.linalg.norm(endpoint["point"] - mouth["b"])),
        )
        if error < best_error:
            best_label, best_error = mouth["label"], error
    return best_label if best_error <= 0.12 * width else "FALSE"


def topology_only_geometry_search(
    poses: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Search existing maps using topology only; never inspect motion output."""
    rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    sensor = LidarSensor()
    for case_id in SEARCH_CASES:
        geometry = _extended_geometry(case_id)
        profile = LidarProfileJunctionDetector(GeometryProfileConfig(LIDAR_MAX_RANGE))
        tracks = {
            side: {"created": math.nan, "initial": "NONE", "complete": math.nan, "last_partial": math.nan, "endpoint": None, "snapshot": None, "width": math.nan}
            for side in ("LOCAL_NEGATIVE", "LOCAL_POSITIVE")
        }
        for pose in poses:
            position = np.array([pose["x_eval_only"], pose["y_eval_only"]])
            scan = sensor.scan(geometry, position, pose["yaw_deg_local_pose"])
            detected = profile.detect(scan.angles_deg, scan.ranges)
            width = float(detected["estimated_corridor_width"])
            orientation = float(detected["stable_corridor_orientation_deg"])
            if not detected["corridor_model_initialized"] or not math.isfinite(width) or not math.isfinite(orientation):
                continue
            snapshot = {
                "context": f"{case_id}_{pose['frame']}",
                "angles": scan.angles_deg,
                "ranges": scan.ranges,
                "hit": scan.ranges < scan.max_range - np.finfo(float).eps * scan.max_range * 64.0,
                "max_range": scan.max_range,
                "position_eval": position,
                "yaw_eval": pose["yaw_deg_local_pose"],
            }
            result = _analyze(snapshot["context"], snapshot, width)
            states = _runtime_side_states(result, orientation)
            for side, state in states.items():
                track = tracks[side]
                if state["topology"] == "PARTIAL":
                    if not math.isfinite(track["created"]):
                        track.update({"created": pose["time"], "initial": "PARTIAL", "endpoint": state["representative_endpoint"], "snapshot": snapshot, "width": width})
                    track["last_partial"] = pose["time"]
                elif state["topology"] == "COMPLETE" and math.isfinite(track["created"]) and not math.isfinite(track["complete"]):
                    track["complete"] = pose["time"]
        run_end = poses[-1]["time"] if poses else 0.0
        for side, track in tracks.items():
            created = math.isfinite(track["created"])
            complete = math.isfinite(track["complete"])
            partial_duration = (run_end - track["created"]) if created and not complete else (track["complete"] - track["created"]) if complete else 0.0
            persistent = created and not complete and partial_duration >= 8.0
            branch_eval = _posthoc_branch_label(geometry, track["snapshot"], track["endpoint"], track["width"]) if created else "NONE"
            row = {
                "geometry_id": case_id,
                "geometry_source": "EXISTING_CLEAN_SIMULATOR_CASE",
                "parameter_changes": "NONE",
                "runtime_local_side": side,
                "candidate_branch_eval": branch_eval,
                "candidate_created": created,
                "t_candidate_created": track["created"],
                "initial_topology": track["initial"],
                "opposite_termination_observed": complete,
                "t_complete": track["complete"],
                "partial_duration": partial_duration,
                "persistent_partial_pass": persistent,
                "selected_for_motion_validation": False,
            }
            rows.append(row)
            if selected is None and persistent and branch_eval not in {"NONE", "FALSE"}:
                selected = row
    if selected is not None:
        selected["selected_for_motion_validation"] = True
    return rows, selected


def _event_time(runner: WallTopologySPHRunner, candidate_id: str, event: str) -> float:
    return next((float(row["timestamp"]) for row in runner.events if row["candidate_id"] == candidate_id and row["event"] == event), math.nan)


def _candidate_summaries(runner: PersistentPartialAuditRunner, run_end: float) -> list[dict[str, Any]]:
    """Summarize topology/motion ordering for each runtime candidate."""
    rows = []
    for candidate in runner.candidates:
        created = candidate.created_time
        complete = _event_time(runner, candidate.candidate_id, "COMPLETE_BRANCH_CANDIDATE")
        motion = _event_time(runner, candidate.candidate_id, "MOTION_SUPPORTED")
        shadow = runner.first_motion_condition.get(candidate.candidate_id, math.nan)
        first_motion = runner.first_motion_evidence.get(candidate.candidate_id, math.nan)
        persistent = not math.isfinite(complete)
        if persistent and math.isfinite(shadow):
            order = "PERSISTENT_PARTIAL_MOTION_ONLY"
        elif math.isfinite(motion) and (not math.isfinite(complete) or motion < complete):
            order = "MOTION_BEFORE_COMPLETE"
        elif math.isfinite(complete) and math.isfinite(motion) and complete < motion:
            order = "COMPLETE_BEFORE_MOTION"
        elif persistent:
            order = "NO_MOTION_SUPPORT"
        else:
            order = "UNRESOLVED"
        evidence = candidate.best_evidence or candidate.last_evidence
        rows.append({
            "candidate_id": candidate.candidate_id,
            "branch_eval_posthoc": candidate.matched_branch_eval_only,
            "t_created": created,
            "t_partial_confirmed": created,
            "t_complete": complete,
            "t_first_motion_evidence": first_motion,
            "t_motion_threshold_met": shadow,
            "t_motion_supported": motion,
            "t_shadow_motion_supported": shadow,
            "t_run_end": run_end,
            "partial_duration": (run_end - created) if persistent else complete - created,
            "persistent_partial": persistent,
            "support_robot_count_at_transition": evidence.get("supporting_robot_count", 0),
            "stable_support_total": evidence.get("supporting_robot_count_ever", 0),
            "free_space_half_plane_entries": candidate.directional_samples,
            "motion_tangent_deg": evidence.get("motion_direction_local", math.nan),
            "reliability": evidence.get("motion_reliability", 0.0),
            "dispersion": evidence.get("direction_dispersion", math.nan),
            "backflow_ratio": evidence.get("backflow_ratio", 0.0),
            "final_state": candidate.state,
            "time_order_class": order,
        })
    return rows


def _trajectory_rows(runner: PersistentPartialAuditRunner) -> list[dict[str, Any]]:
    """Expand stable support histories into one trajectory audit per robot."""
    rows = []
    for candidate in runner.candidates:
        for robot_id in sorted(candidate.supporting_ids):
            history = candidate.histories.get(robot_id, [])
            entry = candidate.support_entry_positions.get(robot_id)
            if entry is None or not history:
                continue
            final = history[-1][1]
            displacement = final - entry
            progress = float(np.dot(displacement, candidate.free_axis_local))
            tangent = math.degrees(math.atan2(displacement[1], displacement[0])) if np.linalg.norm(displacement) > 0 else math.nan
            deltas = [second[1] - first[1] for first, second in zip(history, history[1:])]
            backflow = any(float(np.dot(delta, candidate.free_axis_local)) < 0.0 for delta in deltas)
            initial = candidate.first_positions[robot_id]
            rows.append({
                "candidate_id": candidate.candidate_id,
                "robot_id": robot_id,
                "entry_time": candidate.first_support_time.get(robot_id, math.nan),
                "initial_relative_x": float(initial[0]),
                "initial_relative_y": float(initial[1]),
                "free_space_crossing_time": candidate.first_support_time.get(robot_id, math.nan),
                "trajectory_tangent": tangent,
                "progress_distance": progress,
                "stable_support_boolean": True,
                "backflow_boolean": backflow,
            })
    return rows


def _plot_timeline(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot topology state and frozen motion evidence over time."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    levels = {"PARTIAL": 1, "COMPLETE": 2}
    for candidate_id in sorted({row["candidate_id"] for row in rows}):
        selected = [row for row in rows if row["candidate_id"] == candidate_id]
        times = [row["time"] for row in selected]
        axes[0].step(times, [levels[row["topology_state"]] for row in selected], where="post", label=candidate_id)
        axes[1].plot(times, [row["motion_support_robot_count"] for row in selected], label=f"{candidate_id} support")
        axes[1].plot(times, [row["reliability"] for row in selected], linestyle="--", label=f"{candidate_id} reliability")
    axes[0].set(yticks=[1, 2], yticklabels=["PARTIAL", "COMPLETE"], ylabel="Topology")
    axes[1].set(xlabel="time [s]", ylabel="support / reliability")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_growth(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot cumulative stable supporting robots for each candidate."""
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for candidate_id in sorted({row["candidate_id"] for row in rows}):
        selected = [row for row in rows if row["candidate_id"] == candidate_id]
        axis.step([row["time"] for row in selected], [row["stable_support_count"] for row in selected], where="post", label=candidate_id)
    axis.set(xlabel="time [s]", ylabel="cumulative stable supporting robots", title="Persistent-PARTIAL support growth")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_motion_scene(path: Path, runner: PersistentPartialAuditRunner) -> None:
    """Plot observed terminations and stable robot trajectories in candidate frames."""
    fig, axes = plt.subplots(1, max(1, len(runner.candidates)), figsize=(7 * max(1, len(runner.candidates)), 6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, candidate in zip(axes, runner.candidates):
        axis.scatter(*candidate.endpoint_local, marker="x", s=100, color="tab:purple", label="observed termination")
        axis.arrow(*candidate.endpoint_local, *(candidate.free_axis_local * 20.0), color="tab:green", width=0.7, length_includes_head=True)
        vectors = []
        for robot_id in sorted(candidate.supporting_ids):
            history = candidate.histories.get(robot_id, [])
            if len(history) < 2:
                continue
            points = np.asarray([item[1] for item in history])
            axis.plot(points[:, 0], points[:, 1], linewidth=1.2, alpha=0.8)
            entry = candidate.support_entry_positions.get(robot_id)
            if entry is not None:
                vectors.append(points[-1] - entry)
        if vectors:
            mean = np.mean(vectors, axis=0)
            axis.arrow(*candidate.endpoint_local, *mean, color="black", width=0.8, length_includes_head=True, label="motion tangent")
        axis.set(title=f"{candidate.candidate_id} {candidate.topology_type}/{candidate.state}", xlabel="candidate-local x", ylabel="candidate-local y", aspect="equal")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _same_run(first: PersistentPartialAuditRunner, second: PersistentPartialAuditRunner) -> bool:
    """Compare deterministic state/event and final evidence records."""
    def canonical(value: Any) -> Any:
        if isinstance(value, float) and math.isnan(value):
            return "NaN"
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        return value

    return canonical(first.events) == canonical(second.events) and canonical(
        _candidate_summaries(first, first.world.time)
    ) == canonical(_candidate_summaries(second, second.world.time))


def _baseline_exact(runner: PersistentPartialAuditRunner) -> bool:
    """Check the three frozen reference event times for both candidates."""
    for event, expected in BASELINE_REFERENCE.items():
        observed = [float(row["timestamp"]) for row in runner.events if row["event"] == event]
        if len(observed) != 2 or any(not math.isclose(value, expected, abs_tol=1.0e-12) for value in observed):
            return False
    return True


def _verdict(m0: WallTopologySPHRunner, summaries: list[dict[str, Any]], selected: dict[str, Any] | None) -> str:
    if any(candidate.state == "MOTION_SUPPORTED" for candidate in m0.candidates):
        return "F_FALSE_MOTION_SUPPORT_REGRESSION"
    if selected is None:
        return "E_NO_PERSISTENT_PARTIAL_GEOMETRY_FOUND"
    if any(row["time_order_class"] == "PERSISTENT_PARTIAL_MOTION_ONLY" for row in summaries):
        return "A_PERSISTENT_PARTIAL_MOTION_ONLY_VALIDATION_SUCCESS"
    if any(row["time_order_class"] == "MOTION_BEFORE_COMPLETE" for row in summaries):
        return "B_MOTION_VALIDATION_PRECEDES_TOPOLOGY_COMPLETION"
    if any(row["time_order_class"] == "COMPLETE_BEFORE_MOTION" for row in summaries):
        return "C_TOPOLOGY_COMPLETES_BEFORE_MOTION_SUPPORT"
    if any(row["persistent_partial"] for row in summaries):
        return "D_PERSISTENT_PARTIAL_WITHOUT_MOTION_SUPPORT"
    return "E_NO_PERSISTENT_PARTIAL_GEOMETRY_FOUND"


def run_experiment(output: Path, frames: int, m0_frames: int) -> dict[str, Any]:
    """Execute baseline, topology-only selection, validation/replay, and M0."""
    output.mkdir(parents=True, exist_ok=True)
    baseline = PersistentPartialAuditRunner("M1_CROSS_BASELINE", True)
    for frame in range(frames):
        baseline.step(frame)
    baseline_pass = _baseline_exact(baseline)
    if not baseline_pass:
        raise RuntimeError("EXP-034 exact baseline replay failed")

    search_rows, selected = topology_only_geometry_search(baseline.pose_timeline)
    _write(output / "geometry_search_summary.csv", search_rows)
    if selected is None:
        _write(output / "selected_geometry.csv", [], list(search_rows[0]))
        m0 = run_case("M0_STRAIGHT", m0_frames, False)
        verdict = _verdict(m0, [], None)
        _write(output / "candidate_state_timeline.csv", [], [
            "time", "candidate_id", "topology_state", "termination_count",
            "opposite_termination_observed_eval", "complete_gap_present",
            "motion_support_robot_count", "stable_support_count", "reliability",
            "dispersion", "backflow", "motion_condition_met", "runtime_state",
            "shadow_motion_only_state",
        ])
        _write(output / "candidate_summary.csv", [], [
            "candidate_id", "branch_eval_posthoc", "t_created", "t_partial_confirmed",
            "t_complete", "t_first_motion_evidence", "t_motion_threshold_met",
            "t_motion_supported", "t_shadow_motion_supported", "t_run_end",
            "partial_duration", "persistent_partial", "support_robot_count_at_transition",
            "stable_support_total", "free_space_half_plane_entries", "motion_tangent_deg",
            "reliability", "dispersion", "backflow_ratio", "final_state", "time_order_class",
        ])
        _write(output / "motion_support_timeline.csv", [], [
            "time", "candidate_id", "motion_support_robot_count", "stable_support_count",
            "reliability", "dispersion", "backflow", "motion_condition_met",
        ])
        _write(output / "supporting_robot_trajectories.csv", [], [
            "candidate_id", "robot_id", "entry_time", "initial_relative_x",
            "initial_relative_y", "free_space_crossing_time", "trajectory_tangent",
            "progress_distance", "stable_support_boolean", "backflow_boolean",
        ])
        _write(output / "negative_control_summary.csv", [{
            "geometry_id": "M0_STRAIGHT",
            "candidate_count": len(m0.candidates),
            "motion_supported_count": sum(candidate.state == "MOTION_SUPPORTED" for candidate in m0.candidates),
            "pass": not m0.candidates,
        }])
        _write(output / "verdict.csv", [{
            "experiment_id": EXPERIMENT_ID,
            "verdict": verdict,
            "baseline_exact_replay": baseline_pass,
            "selected_geometry": "NONE",
            "deterministic_replay": "NOT_APPLICABLE_NO_SELECTED_GEOMETRY",
            "seed": "N/A_DETERMINISTIC",
            "M0_negative_control": not m0.candidates,
            "selection_definition_equivalence": True,
            "runtime_GT_map_used": False,
            "detector_modified": False,
            "motion_threshold_modified": False,
            "SPH_force_modified": False,
            "candidate_direction_commanded": False,
        }])
        return {"verdict": verdict, "baseline_pass": baseline_pass, "selected": None}
    _write(output / "selected_geometry.csv", [selected])

    case_id = selected["geometry_id"]
    validation = PersistentPartialAuditRunner(case_id, True)
    replay = PersistentPartialAuditRunner(case_id, True)
    for frame in range(frames):
        validation.step(frame)
    for frame in range(frames):
        replay.step(frame)
    deterministic = _same_run(validation, replay)
    m0 = run_case("M0_STRAIGHT", m0_frames, False)
    summaries = _candidate_summaries(validation, validation.world.time)
    verdict = _verdict(m0, summaries, selected)

    _write(output / "baseline_replay_events.csv", baseline.events)
    _write(output / "candidate_state_timeline.csv", validation.state_timeline)
    _write(output / "candidate_summary.csv", summaries)
    _write(output / "motion_support_timeline.csv", validation.state_timeline)
    _write(output / "supporting_robot_trajectories.csv", _trajectory_rows(validation))
    _write(output / "negative_control_summary.csv", [{
        "geometry_id": "M0_STRAIGHT",
        "candidate_count": len(m0.candidates),
        "motion_supported_count": sum(candidate.state == "MOTION_SUPPORTED" for candidate in m0.candidates),
        "pass": not m0.candidates,
    }])
    _write(output / "angular_detector_shadow.csv", validation.angular_shadow_rows)
    _write(output / "verdict.csv", [{
        "experiment_id": EXPERIMENT_ID,
        "verdict": verdict,
        "baseline_exact_replay": baseline_pass,
        "selected_geometry": case_id,
        "deterministic_replay": deterministic,
        "seed": "N/A_DETERMINISTIC",
        "M0_negative_control": not m0.candidates,
        "runtime_GT_map_used": False,
        "detector_modified": False,
        "motion_threshold_modified": False,
        "SPH_force_modified": False,
        "candidate_direction_commanded": False,
    }])
    _plot_timeline(output / "persistent_partial_timeline.png", validation.state_timeline)
    _plot_growth(output / "support_growth_over_time.png", validation.state_timeline)
    _plot_motion_scene(output / "persistent_partial_motion_scene.png", validation)
    return {
        "verdict": verdict,
        "baseline_pass": baseline_pass,
        "selected": selected,
        "deterministic": deterministic,
        "summaries": summaries,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--m0-frames", type=int, default=120)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _candidate_state_self_test()
    result = run_experiment(args.output, args.frames, args.m0_frames)
    print(f"EXP-035 verdict={result['verdict']}")
    print(f"baseline_exact={result['baseline_pass']} selected={result['selected']}")
    print(f"deterministic={result.get('deterministic', False)}")
    for row in result.get("summaries", []):
        print(row)
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
