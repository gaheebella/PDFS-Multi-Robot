"""EXP-034: partial wall topology plus physical SPH motion validation.

This opt-in integration simulator leaves the clean pre-exploration physics and
the frozen EXP-033 wall-topology extractor unchanged.  A single observed wall
termination is retained as a geometric hypothesis.  Natural local-forward SPH
motion is then measured in the termination-local free-space half-plane; robots
are never assigned or commanded toward a candidate.

Runtime decisions use only one local LiDAR scan, the locally estimated corridor
frame/width, observed wall topology, and robot positions/motion relative to the
LiDAR pose at candidate creation.  Map geometry and branch identity are added
only to the post-hoc evaluation columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
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
from junction_detection.integration.run_provisional_anchor_pointcloud_observation import (
    evaluate_snapshot,
)
from junction_detection.integration.run_wall_topology_branch_opening_diagnostic import (
    _analyze,
    _gt_mouths_eval,
    _self_test as exp033_self_test,
)
from junction_detection.pointcloud.lidar_profile_junction_detector import (
    GeometryProfileConfig,
    LidarProfileJunctionDetector,
)
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import (
    detect_openings,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import (
    DT,
    LIDAR_MAX_RANGE,
    LOCAL_COMMUNICATION_RANGE,
    LOCAL_FORWARD_REFERENCE_SPEED,
    MIN_SPEED,
    ROBOT_RADIUS,
    SAMPLE_PERIOD,
    VISCOELASTIC_REST_MIN,
    PygameRenderer,
    SimulationRunner,
)

EXPERIMENT_ID = "EXP-034"
DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/wall_topology_sph_validation"
MOTION_WINDOW_SECONDS = LOCAL_COMMUNICATION_RANGE / LOCAL_FORWARD_REFERENCE_SPEED
MOTION_MIN_PROGRESS = VISCOELASTIC_REST_MIN
MOTION_MIN_ROBOTS = 2  # A physical flow must contain more than one robot.


def _write(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    """Write heterogeneous diagnostic rows and retain headers for empty controls."""
    if fields is None:
        if not rows:
            return
        fields = list(rows[0])
        for row in rows[1:]:
            fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rotation(yaw_rad: float) -> np.ndarray:
    """Return the local-to-world rotation for one LiDAR pose."""
    return np.array(
        [[math.cos(yaw_rad), -math.sin(yaw_rad)],
         [math.sin(yaw_rad), math.cos(yaw_rad)]],
        dtype=float,
    )


def _axis_frame(orientation_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return sensor-local corridor-forward and local-left unit vectors."""
    angle = math.radians(orientation_deg)
    forward = np.array([math.cos(angle), math.sin(angle)])
    return forward, np.array([-forward[1], forward[0]])


def _snapshot(runner: "WallTopologySPHRunner", frame: int, row: dict[str, Any]) -> dict[str, Any]:
    """Copy a scan for frozen EXP-033 without passing map data to the detector."""
    scan = runner.last_visual[0].lidar_scan
    leader = runner.leader()
    margin = np.finfo(float).eps * max(1.0, scan.max_range) * 64.0
    return {
        "context": f"FRAME_{frame}",
        "angles": scan.angles_deg.copy(),
        "ranges": scan.ranges.copy(),
        "hit": scan.ranges < scan.max_range - margin,
        "max_range": scan.max_range,
        # These fields are not consumed by _analyze. They are retained for
        # plotting and are passed to GT helpers only after runtime updates.
        "position_eval": leader.position.copy(),
        "yaw_eval": math.degrees(leader.body_yaw_rad),
        "frame": frame,
        "time": float(row["timestamp"]),
    }


def _segment_tangent(result: dict[str, Any], endpoint: dict[str, Any]) -> np.ndarray:
    """Recover an axial wall tangent from the endpoint's fitted segment."""
    segment_id = endpoint["segment_ids"][0]
    segment = next(row for row in result["segments"] if row["segment_id"] == segment_id)
    angle = math.radians(float(segment["orientation_deg"]))
    return np.array([math.cos(angle), math.sin(angle)])


def _endpoint_free_continuation(snapshot: dict[str, Any], endpoint: dict[str, Any]) -> float:
    """Measure the scan's local no-wall continuation adjacent to a termination."""
    beam = int(np.argmin(np.abs(
        (np.asarray(snapshot["angles"]) - math.degrees(math.atan2(endpoint["point"][1], endpoint["point"][0])) + 180.0)
        % 360.0 - 180.0
    )))
    ranges = np.asarray(snapshot["ranges"], dtype=float)
    return float(max(ranges[(beam - 1) % len(ranges)], ranges[(beam + 1) % len(ranges)]) - np.linalg.norm(endpoint["point"]))


@dataclass
class BranchCandidate:
    """A persistent local wall hypothesis and its physical motion evidence."""

    candidate_id: str
    topology_type: str
    created_time: float
    created_frame: int
    anchor_position_eval: np.ndarray
    anchor_yaw_rad: float
    endpoint_local: np.ndarray
    endpoint_ids: tuple[int, ...]
    wall_segment_ids: tuple[int, ...]
    wall_tangent_local: np.ndarray
    free_axis_local: np.ndarray
    estimated_width: float
    free_space_evidence: float
    gap_center_local: np.ndarray | None = None
    gap_width: float = math.nan
    state: str = "PARTIAL_UNVALIDATED"
    first_positions: dict[int, np.ndarray] = field(default_factory=dict)
    first_support_time: dict[int, float] = field(default_factory=dict)
    support_entry_positions: dict[int, np.ndarray] = field(default_factory=dict)
    histories: dict[int, list[tuple[float, np.ndarray]]] = field(default_factory=dict)
    supporting_ids: set[int] = field(default_factory=set)
    observed_ids: set[int] = field(default_factory=set)
    backflow_samples: int = 0
    directional_samples: int = 0
    last_evidence: dict[str, Any] = field(default_factory=dict)
    best_evidence: dict[str, Any] = field(default_factory=dict)
    matched_branch_eval_only: str = "UNAVAILABLE"

    def world_to_local(self, point: np.ndarray) -> np.ndarray:
        """Express a simulator position relative to the candidate's LiDAR pose."""
        return _rotation(self.anchor_yaw_rad).T @ (np.asarray(point) - self.anchor_position_eval)

    def update_motion(self, robots: list[Any], timestamp: float) -> list[dict[str, Any]]:
        """Accumulate stable motion in the observed free-space half-plane.

        Association uses the existing communication radius around the observed
        endpoint. Support requires outward crossing, a physical link-length of
        net progress, and one communication-transit time of persistence.
        """
        motion_rows: list[dict[str, Any]] = []
        stable_vectors: list[np.ndarray] = []
        progress_values: list[float] = []
        durations: list[float] = []
        current_support: set[int] = set()
        for robot in robots:
            local = self.world_to_local(robot.position)
            # Preserve the candidate-creation displacement baseline even while
            # a robot is still outside the endpoint's local association range.
            self.first_positions.setdefault(robot.robot_id, local.copy())
            relative = local - self.endpoint_local
            tangent_offset = abs(float(np.dot(relative, self.wall_tangent_local)))
            outward = float(np.dot(relative, self.free_axis_local))
            near = float(np.linalg.norm(relative)) <= LOCAL_COMMUNICATION_RANGE
            if not near:
                continue
            self.observed_ids.add(robot.robot_id)
            history = self.histories.setdefault(robot.robot_id, [])
            history.append((timestamp, local.copy()))
            velocity_local = _rotation(self.anchor_yaw_rad).T @ robot.observed_velocity
            displacement = local - self.first_positions[robot.robot_id]
            outward_speed = float(np.dot(velocity_local, self.free_axis_local))
            if outward > 0.0:
                self.first_support_time.setdefault(robot.robot_id, timestamp)
                self.support_entry_positions.setdefault(robot.robot_id, local.copy())
                duration = timestamp - self.first_support_time[robot.robot_id]
                support_displacement = local - self.support_entry_positions[robot.robot_id]
                outward_progress = float(np.dot(support_displacement, self.free_axis_local))
                self.directional_samples += 1
                if outward_speed < 0.0:
                    self.backflow_samples += 1
                stable = (
                    duration >= MOTION_WINDOW_SECONDS
                    and outward_progress >= MOTION_MIN_PROGRESS
                    and float(np.linalg.norm(support_displacement)) >= MOTION_MIN_PROGRESS
                )
                if stable:
                    current_support.add(robot.robot_id)
                    self.supporting_ids.add(robot.robot_id)
                    stable_vectors.append(support_displacement)
                    progress_values.append(outward_progress)
                    durations.append(duration)
            else:
                stable = False
                duration = 0.0
                outward_progress = float(np.dot(displacement, self.free_axis_local))
            motion_rows.append({
                "timestamp": timestamp,
                "candidate_id": self.candidate_id,
                "robot_id": robot.robot_id,
                "relative_x": float(local[0]),
                "relative_y": float(local[1]),
                "velocity_x_local": float(velocity_local[0]),
                "velocity_y_local": float(velocity_local[1]),
                "displacement": float(np.linalg.norm(displacement)),
                "outward_progress": outward_progress,
                "outward_speed": outward_speed,
                "tangent_offset": tangent_offset,
                "candidate_support": stable,
            })

        if stable_vectors:
            unit = np.asarray(stable_vectors) / np.maximum(
                np.linalg.norm(stable_vectors, axis=1)[:, None], 1.0e-12
            )
            resultant = np.mean(unit, axis=0)
            resultant_norm = float(np.linalg.norm(resultant))
            tangent = math.degrees(math.atan2(resultant[1], resultant[0]))
            dispersion = 1.0 - resultant_norm
        else:
            tangent, dispersion, resultant_norm = math.nan, math.nan, 0.0
        backflow = self.backflow_samples / max(1, self.directional_samples)
        supported = (
            len(current_support) >= MOTION_MIN_ROBOTS
            and backflow <= 0.5
            and bool(progress_values)
        )
        reliability = (
            resultant_norm
            * (1.0 - backflow)
            * min(1.0, len(current_support) / MOTION_MIN_ROBOTS)
            if stable_vectors else 0.0
        )
        if supported:
            self.state = "MOTION_SUPPORTED"
        elif self.state != "MOTION_SUPPORTED" and timestamp - self.created_time >= MOTION_WINDOW_SECONDS:
            self.state = "UNCERTAIN"
        evidence = {
            "supporting_robot_count": len(current_support),
            "supporting_robot_count_ever": len(self.supporting_ids),
            "observed_robot_count": len(self.observed_ids),
            "mean_displacement": float(np.mean([np.linalg.norm(v) for v in stable_vectors])) if stable_vectors else 0.0,
            "median_displacement": float(np.median([np.linalg.norm(v) for v in stable_vectors])) if stable_vectors else 0.0,
            "motion_direction_local": tangent,
            "direction_dispersion": dispersion,
            "forward_progress": float(np.mean(progress_values)) if progress_values else 0.0,
            "backflow_ratio": backflow,
            "motion_duration": float(max(durations, default=0.0)),
            "motion_supported": supported,
            "motion_reliability": reliability,
        }
        self.last_evidence = evidence
        if supported and (
            not self.best_evidence
            or evidence["supporting_robot_count"] > self.best_evidence["supporting_robot_count"]
        ):
            self.best_evidence = dict(evidence)
        return motion_rows


class WallTopologySPHRunner(SimulationRunner):
    """Run frozen wall topology beside unchanged natural SPH dynamics."""

    def __init__(self, case_id: str, rear_start: bool = False):
        profile = LidarProfileJunctionDetector(GeometryProfileConfig(LIDAR_MAX_RANGE))
        super().__init__(
            case_id,
            "local_forward",
            profile_detector=profile,
            hold_on_profile_detection=False,
        )
        if rear_start:
            _rear_start(self)
        self.candidates: list[BranchCandidate] = []
        self.candidate_rows: list[dict[str, Any]] = []
        self.motion_rows: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.event_keys: set[tuple[str, str]] = set()
        self.last_result: dict[str, Any] | None = None
        self.last_snapshot: dict[str, Any] | None = None
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.angular_shadow_rows: list[dict[str, Any]] = []

    def leader(self) -> Any:
        return next(robot for robot in self.world.robots if robot.robot_id == self.world.lidar_robot_id)

    def _event(self, event: str, candidate: BranchCandidate, before: str, after: str) -> None:
        key = (event, candidate.candidate_id)
        if key in self.event_keys:
            return
        self.event_keys.add(key)
        self.events.append({
            "timestamp": self.world.time,
            "event": event,
            "candidate_id": candidate.candidate_id,
            "state_before": before,
            "state_after": after,
        })

    def _existing_side_candidate(self, free_axis: np.ndarray) -> BranchCandidate | None:
        return next(
            (candidate for candidate in self.candidates if float(np.dot(candidate.free_axis_local, free_axis)) > 0.5),
            None,
        )

    def _make_candidates(
        self,
        result: dict[str, Any],
        snapshot: dict[str, Any],
        orientation: float,
        width: float,
    ) -> None:
        """Persist COMPLETE or one-ended PARTIAL candidates without reconstruction."""
        _, left_axis = _axis_frame(orientation)
        endpoints = {row["endpoint_id"]: row for row in result["endpoints"]}
        used: set[int] = set()
        for gap in result["gaps"]:
            if not gap["candidate_valid"]:
                continue
            first, second = endpoints[gap["endpoint_a"]], endpoints[gap["endpoint_b"]]
            side_a = math.copysign(1.0, float(np.dot(first["point"], left_axis)))
            side_b = math.copysign(1.0, float(np.dot(second["point"], left_axis)))
            if side_a != side_b:
                continue
            free_axis = left_axis * side_a
            existing = self._existing_side_candidate(free_axis)
            if existing is None:
                endpoint = first
                existing = self._new_candidate(
                    "COMPLETE", endpoint, result, snapshot, width, free_axis,
                    (first["endpoint_id"], second["endpoint_id"]), gap,
                )
            elif existing.topology_type == "PARTIAL":
                before = existing.state
                existing.topology_type = "COMPLETE"
                existing.endpoint_ids = (first["endpoint_id"], second["endpoint_id"])
                existing.gap_center_local = np.asarray(gap["gap_center"]).copy()
                existing.gap_width = float(gap["gap_width"])
                if before == "PARTIAL_UNVALIDATED":
                    existing.state = "COMPLETE_UNVALIDATED"
                self._event("COMPLETE_BRANCH_CANDIDATE", existing, before, existing.state)
            used.update((first["endpoint_id"], second["endpoint_id"]))

        for endpoint in result["endpoints"]:
            if endpoint["endpoint_id"] in used or endpoint["endpoint_type"] != "WALL_TERMINATION":
                continue
            lateral = float(np.dot(endpoint["point"], left_axis))
            if abs(lateral) <= np.finfo(float).eps:
                continue
            free_axis = left_axis * math.copysign(1.0, lateral)
            if self._existing_side_candidate(free_axis) is None:
                self._new_candidate(
                    "PARTIAL", endpoint, result, snapshot, width, free_axis,
                    (endpoint["endpoint_id"],), None,
                )

    def _new_candidate(
        self,
        topology: str,
        endpoint: dict[str, Any],
        result: dict[str, Any],
        snapshot: dict[str, Any],
        width: float,
        free_axis: np.ndarray,
        endpoint_ids: tuple[int, ...],
        gap: dict[str, Any] | None,
    ) -> BranchCandidate:
        candidate = BranchCandidate(
            candidate_id=f"C{len(self.candidates)}",
            topology_type=topology,
            created_time=self.world.time,
            created_frame=int(snapshot["frame"]),
            anchor_position_eval=self.leader().position.copy(),
            anchor_yaw_rad=self.leader().body_yaw_rad,
            endpoint_local=np.asarray(endpoint["point"]).copy(),
            endpoint_ids=endpoint_ids,
            wall_segment_ids=tuple(endpoint["segment_ids"]),
            wall_tangent_local=_segment_tangent(result, endpoint),
            free_axis_local=np.asarray(free_axis).copy(),
            estimated_width=width,
            free_space_evidence=_endpoint_free_continuation(snapshot, endpoint),
            gap_center_local=None if gap is None else np.asarray(gap["gap_center"]).copy(),
            gap_width=math.nan if gap is None else float(gap["gap_width"]),
            state=f"{topology}_UNVALIDATED",
        )
        self.candidates.append(candidate)
        self._event(f"{topology}_BRANCH_CANDIDATE", candidate, "NONE", candidate.state)
        return candidate

    def _posthoc_label(self, candidate: BranchCandidate, snapshot: dict[str, Any]) -> str:
        """Match a completed runtime candidate to the nearest GT mouth boundary."""
        if self.geometry.entrance_y is None:
            return "NONE"
        mouths = _gt_mouths_eval(SimpleNamespace(geometry=self.geometry), snapshot)
        best_label, best_error = "FALSE", math.inf
        for mouth in mouths:
            if mouth["branch_type"] != "OUTGOING":
                continue
            error = min(
                float(np.linalg.norm(candidate.endpoint_local - mouth["a"])),
                float(np.linalg.norm(candidate.endpoint_local - mouth["b"])),
            )
            if error < best_error:
                best_label, best_error = mouth["label"], error
        return best_label if best_error <= 0.12 * candidate.estimated_width else "FALSE"

    def step(self, frame: int) -> dict[str, Any] | None:
        row = super().step(frame)
        if row is None:
            return None
        snapshot = _snapshot(self, frame, row)
        profile = self.last_profile_result
        width = float(profile["estimated_corridor_width"])
        orientation = float(profile["stable_corridor_orientation_deg"])
        available = bool(
            profile["corridor_model_initialized"] and math.isfinite(width)
            and width > 0.0 and math.isfinite(orientation)
        )
        result = None
        if available:
            result = _analyze(snapshot["context"], snapshot, width)
            self._make_candidates(result, snapshot, orientation, width)
            self.last_result, self.last_snapshot = result, snapshot
        openings = list(detect_openings(snapshot["angles"], snapshot["ranges"]))

        for candidate in self.candidates:
            before = candidate.state
            self.motion_rows.extend(candidate.update_motion(self.world.robots, self.world.time))
            if candidate.state != before:
                event = "MOTION_SUPPORTED" if candidate.state == "MOTION_SUPPORTED" else candidate.state
                self._event(event, candidate, before, candidate.state)

        # Everything above this line is runtime-local. GT branch identity and
        # angular matching are attached only after all state updates finish.
        angular_summary, _ = evaluate_snapshot(self, snapshot, openings)
        self.angular_shadow_rows.append({
            "timestamp": self.world.time,
            "angular_opening_count": len(openings),
            "angular_outgoing_count_eval_only": angular_summary["matched_outgoing_count_eval_only"],
            "angular_false_opening_count_eval_only": angular_summary["false_opening_count_eval_only"],
        })
        for candidate in self.candidates:
            if candidate.matched_branch_eval_only == "UNAVAILABLE":
                # Freeze matching in the candidate-creation scan frame. The
                # candidate itself never consumes this evaluation label.
                candidate.matched_branch_eval_only = self._posthoc_label(candidate, snapshot)
            evidence = candidate.last_evidence
            self.candidate_rows.append({
                "timestamp": self.world.time,
                "candidate_id": candidate.candidate_id,
                "topology_type": candidate.topology_type,
                "termination_count": len(candidate.endpoint_ids),
                "gap_width": candidate.gap_width,
                "free_continuation": candidate.free_space_evidence,
                "candidate_state": candidate.state,
                "observed_termination_x_local": float(candidate.endpoint_local[0]),
                "observed_termination_y_local": float(candidate.endpoint_local[1]),
                "wall_segment_id": json.dumps(candidate.wall_segment_ids),
                "wall_tangent_local_deg": math.degrees(math.atan2(candidate.wall_tangent_local[1], candidate.wall_tangent_local[0])),
                "estimated_width": candidate.estimated_width,
                "supporting_robot_count": evidence.get("supporting_robot_count", 0),
                "motion_reliability": evidence.get("motion_reliability", 0.0),
                "matched_branch_eval_only": candidate.matched_branch_eval_only,
            })

        if not self.candidates:
            self.snapshots["normal_corridor"] = self._capture(frame)
        else:
            self.snapshots.setdefault("first_termination", self._capture(frame))
            self.snapshots.setdefault("partial_candidate", self._capture(frame))
            if any(candidate.last_evidence.get("observed_robot_count", 0) for candidate in self.candidates):
                self.snapshots.setdefault("sph_spread", self._capture(frame))
            if any(candidate.last_evidence.get("motion_duration", 0.0) > 0.0 for candidate in self.candidates):
                self.snapshots.setdefault("stable_accumulation", self._capture(frame))
            if any(candidate.state == "MOTION_SUPPORTED" for candidate in self.candidates):
                self.snapshots.setdefault("motion_supported", self._capture(frame))
        self.snapshots["final"] = self._capture(frame)
        row.update({
            "wall_topology_candidate_count": len(self.candidates),
            "motion_supported_candidate_count": sum(c.state == "MOTION_SUPPORTED" for c in self.candidates),
            "candidate_states": json.dumps({c.candidate_id: c.state for c in self.candidates}, sort_keys=True),
            "angular_opening_count_shadow": len(openings),
        })
        return row

    def _capture(self, frame: int) -> dict[str, Any]:
        """Capture candidate-local robot positions for a representative plot."""
        if self.candidates:
            candidate = self.candidates[0]
            robots = np.asarray([candidate.world_to_local(r.position) for r in self.world.robots])
        else:
            candidate = None
            robots = np.asarray([r.position - self.leader().position for r in self.world.robots])
        return {
            "frame": frame,
            "time": self.world.time,
            "result": self.last_result,
            "snapshot": self.last_snapshot,
            "robots": robots,
            "robot_ids": [robot.robot_id for robot in self.world.robots],
            "candidate_states": {c.candidate_id: c.state for c in self.candidates},
            "supporting": {c.candidate_id: sorted(c.supporting_ids) for c in self.candidates},
            "candidates": list(self.candidates),
        }


class WallTopologySPHRenderer(PygameRenderer):
    """Pygame overlay for wall candidates and their motion-support robots."""

    def draw(self, runner: WallTopologySPHRunner, frame: int) -> None:
        super().draw(runner, frame)
        pygame = self.pygame
        if self.show_diagnostics and runner.last_result is not None and runner.last_snapshot is not None:
            snapshot, result = runner.last_snapshot, runner.last_result
            rotation = _rotation(math.radians(float(snapshot["yaw_eval"])))
            origin = np.asarray(snapshot["position_eval"])

            def world(point: np.ndarray) -> np.ndarray:
                return origin + rotation @ np.asarray(point)

            for segment in result["segments"]:
                pygame.draw.line(self.screen, (255, 180, 55), self.world_to_screen(world(segment["start"])), self.world_to_screen(world(segment["end"])), 2)
            for endpoint in result["endpoints"]:
                if endpoint["endpoint_type"] == "WALL_TERMINATION":
                    pygame.draw.circle(self.screen, (255, 70, 70), self.world_to_screen(world(endpoint["point"])), 6, 2)
        by_id = {robot.robot_id: robot for robot in runner.world.robots}
        supported = set().union(*(candidate.supporting_ids for candidate in runner.candidates)) if runner.candidates else set()
        for robot_id in supported:
            pygame.draw.circle(self.screen, (80, 255, 120), self.world_to_screen(by_id[robot_id].position), 6, 2)
        lines = ["EXP-034 Wall topology + natural SPH"]
        lines.extend(
            f"{c.candidate_id}: {c.topology_type} / {c.state} robots={c.last_evidence.get('supporting_robot_count', 0)}"
            for c in runner.candidates[:5]
        )
        if len(lines) == 1:
            lines.append("Candidates: none")
        for index, text in enumerate(lines):
            self.screen.blit(self.font.render(text, True, (255, 245, 170)), (700, 12 + 22 * index))
        pygame.display.flip()


def _plot_capture(path: Path, capture: dict[str, Any], title: str) -> None:
    """Save one candidate-local robot/topology scene."""
    fig, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
    robots = capture["robots"]
    axis.scatter(robots[:, 0], robots[:, 1], s=8, color="0.65", label="SPH robots")
    support_ids = set().union(*capture["supporting"].values()) if capture["supporting"] else set()
    support_mask = np.asarray([robot_id in support_ids for robot_id in capture["robot_ids"]])
    if np.any(support_mask):
        axis.scatter(robots[support_mask, 0], robots[support_mask, 1], s=28, color="tab:green", label="motion support")
    result = capture["result"]
    candidate_frame = capture["candidates"][0] if capture["candidates"] else None
    snapshot = capture["snapshot"]

    def plot_local(point: np.ndarray) -> np.ndarray:
        if candidate_frame is None or snapshot is None:
            return np.asarray(point)
        world = np.asarray(snapshot["position_eval"]) + _rotation(math.radians(float(snapshot["yaw_eval"]))) @ np.asarray(point)
        return candidate_frame.world_to_local(world)

    if result is not None:
        for segment in result["segments"]:
            start, end = plot_local(segment["start"]), plot_local(segment["end"])
            axis.plot([start[0], end[0]], [start[1], end[1]], color="tab:orange", linewidth=2)
        for endpoint in result["endpoints"]:
            if endpoint["endpoint_type"] == "WALL_TERMINATION":
                axis.scatter(*plot_local(endpoint["point"]), color="tab:red", s=65)
    for candidate in capture["candidates"]:
        axis.scatter(*candidate.endpoint_local, marker="x", s=100, color="tab:purple")
        axis.arrow(candidate.endpoint_local[0], candidate.endpoint_local[1], *(candidate.free_axis_local * 18.0), color="tab:green", width=0.8, length_includes_head=True)
    states = ", ".join(f"{key}:{value}" for key, value in capture["candidate_states"].items()) or "none"
    axis.set(title=f"{title} | frame={capture['frame']} t={capture['time']:.2f}\n{states}", xlabel="candidate-local x", ylabel="candidate-local y", aspect="equal", xlim=(-160, 160), ylim=(-160, 160))
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left", fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_case(case_id: str, frames: int, rear_start: bool = False) -> WallTopologySPHRunner:
    """Run one deterministic bounded physical validation case."""
    runner = WallTopologySPHRunner(case_id, rear_start)
    for frame in range(frames):
        runner.step(frame)
    return runner


def _summary_rows(runner: WallTopologySPHRunner) -> list[dict[str, Any]]:
    """Return one final summary row per persistent candidate."""
    rows = []
    snapshot = runner.last_snapshot
    for candidate in runner.candidates:
        evidence = candidate.best_evidence or candidate.last_evidence
        rows.append({
            "candidate_id": candidate.candidate_id,
            "topology_type": candidate.topology_type,
            "supporting_robot_count": evidence.get("supporting_robot_count_ever", 0),
            "observed_robot_count": evidence.get("observed_robot_count", 0),
            "motion_duration": evidence.get("motion_duration", 0.0),
            "mean_displacement": evidence.get("mean_displacement", 0.0),
            "median_displacement": evidence.get("median_displacement", 0.0),
            "motion_direction_local": evidence.get("motion_direction_local", math.nan),
            "direction_dispersion": evidence.get("direction_dispersion", math.nan),
            "forward_progress": evidence.get("forward_progress", 0.0),
            "backflow_ratio": evidence.get("backflow_ratio", 0.0),
            "motion_supported": candidate.state == "MOTION_SUPPORTED",
            "motion_reliability": evidence.get("motion_reliability", 0.0),
            "final_state": candidate.state,
            "matched_branch_eval_only": candidate.matched_branch_eval_only,
        })
    return rows


def _verdict(m0: WallTopologySPHRunner, m1: WallTopologySPHRunner) -> str:
    """Apply the requested A-E experiment verdicts without parameter tuning."""
    if any(candidate.state == "MOTION_SUPPORTED" for candidate in m0.candidates):
        return "CASE_D_FALSE_MOTION_SUPPORT_IN_CORRIDOR"
    supported = [candidate for candidate in m1.candidates if candidate.state == "MOTION_SUPPORTED"]
    if any(candidate.topology_type == "COMPLETE" for candidate in supported):
        return "CASE_E_COMPLETE_TOPOLOGY_AND_MOTION_SUPPORT"
    if any(candidate.topology_type == "PARTIAL" for candidate in supported):
        return "CASE_A_PARTIAL_TOPOLOGY_WITH_STABLE_MOTION_SUPPORT"
    if m1.candidates:
        return "CASE_B_PARTIAL_TOPOLOGY_WITHOUT_MOTION_SUPPORT"
    return "CASE_C_MOTION_PRESENT_BUT_NOT_ASSOCIABLE_TO_PARTIAL_TOPOLOGY"


def save_outputs(output: Path, m0: WallTopologySPHRunner, m1: WallTopologySPHRunner) -> str:
    """Save required candidate, motion, summary, event, shadow, and scene artifacts."""
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "wall_topology_sph_candidates.csv", m1.candidate_rows)
    _write(output / "wall_topology_sph_motion.csv", m1.motion_rows)
    summary = _summary_rows(m1)
    _write(output / "wall_topology_sph_summary.csv", summary)
    _write(output / "wall_topology_sph_events.csv", m1.events, ["timestamp", "event", "candidate_id", "state_before", "state_after"])
    _write(output / "wall_topology_sph_angular_shadow.csv", m1.angular_shadow_rows)
    _write(output / "wall_topology_sph_m0_summary.csv", _summary_rows(m0), [
        "candidate_id", "topology_type", "supporting_robot_count", "observed_robot_count", "motion_duration", "mean_displacement", "median_displacement", "motion_direction_local", "direction_dispersion", "forward_progress", "backflow_ratio", "motion_supported", "motion_reliability", "final_state", "matched_branch_eval_only",
    ])
    verdict = _verdict(m0, m1)
    _write(output / "wall_topology_sph_verdict.csv", [{
        "experiment_id": EXPERIMENT_ID,
        "verdict": verdict,
        "m0_candidate_count": len(m0.candidates),
        "m0_motion_supported_count": sum(c.state == "MOTION_SUPPORTED" for c in m0.candidates),
        "m1_candidate_count": len(m1.candidates),
        "m1_motion_supported_count": sum(c.state == "MOTION_SUPPORTED" for c in m1.candidates),
        "runtime_inputs": "local_scan,W_hat,local_corridor_axis,wall_segments,wall_terminations,relative_robot_motion",
        "GT_map_used_for_runtime": False,
        "SPH_force_modified": False,
        "detector_threshold_modified": False,
    }])
    captures = [
        (m0.snapshots.get("normal_corridor"), "normal_corridor.png", "M0 normal corridor"),
        (m1.snapshots.get("first_termination"), "first_wall_termination.png", "First WALL_TERMINATION"),
        (m1.snapshots.get("partial_candidate"), "partial_branch_candidate.png", "PARTIAL branch candidate"),
        (m1.snapshots.get("sph_spread"), "sph_candidate_spread.png", "Natural SPH spread"),
        (m1.snapshots.get("stable_accumulation"), "stable_motion_accumulation.png", "Stable motion accumulation"),
        (m1.snapshots.get("motion_supported") or m1.snapshots.get("final"), "motion_support_final.png", "Motion support final"),
    ]
    for capture, name, title in captures:
        if capture is not None:
            _plot_capture(output / name, capture, title)
    return verdict


def _candidate_state_self_test() -> None:
    """Check one-ended candidate state transition and a no-motion negative."""
    candidate = BranchCandidate(
        "C0", "PARTIAL", 0.0, 0, np.zeros(2), 0.0,
        np.zeros(2), (0,), (0,), np.array([1.0, 0.0]), np.array([0.0, 1.0]),
        84.0, 20.0,
    )
    robots = [SimpleNamespace(robot_id=index, position=np.array([0.0, 2.0 + index]), observed_velocity=np.array([0.0, 5.0])) for index in range(2)]
    candidate.update_motion(robots, 0.0)
    for robot in robots:
        robot.position[1] += MOTION_MIN_PROGRESS + 1.0
    candidate.update_motion(robots, MOTION_WINDOW_SECONDS + SAMPLE_PERIOD)
    assert candidate.state == "MOTION_SUPPORTED"
    still = BranchCandidate(
        "C1", "PARTIAL", 0.0, 0, np.zeros(2), 0.0,
        np.zeros(2), (0,), (0,), np.array([1.0, 0.0]), np.array([0.0, 1.0]),
        84.0, 20.0,
    )
    stationary = [SimpleNamespace(robot_id=0, position=np.array([0.0, -1.0]), observed_velocity=np.zeros(2))]
    still.update_motion(stationary, MOTION_WINDOW_SECONDS + SAMPLE_PERIOD)
    assert still.state == "UNCERTAIN"


def run_gui(case_id: str, frames: int, rear_start: bool, output: Path) -> None:
    """Run the actual Pygame pipeline with pause/reset and diagnostic overlay."""
    import pygame

    runner = WallTopologySPHRunner(case_id, rear_start)
    renderer = WallTopologySPHRenderer(runner.geometry, gui_scale=0.75)
    frame, running = 0, True
    while running and (frames <= 0 or frame < frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    renderer.paused = not renderer.paused
                elif event.key == pygame.K_d:
                    renderer.show_diagnostics = not renderer.show_diagnostics
                elif event.key == pygame.K_l:
                    renderer.show_lidar = not renderer.show_lidar
                elif event.key == pygame.K_c:
                    renderer.show_communication_links = not renderer.show_communication_links
                elif event.key == pygame.K_n:
                    renderer.show_support_links = not renderer.show_support_links
                elif event.key == pygame.K_r:
                    runner = WallTopologySPHRunner(case_id, rear_start)
                    renderer.configure_camera(runner.geometry)
                    frame = 0
        if not renderer.paused:
            runner.step(frame)
            frame += 1
        renderer.draw(runner, frame)
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "wall_topology_sph_gui_candidates.csv", runner.candidate_rows)
    pygame.quit()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-case", choices=("M0_STRAIGHT", "M1_CROSS_BASELINE"))
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--m0-frames", type=int, default=120)
    parser.add_argument("--rear-start", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    exp033_self_test()
    _candidate_state_self_test()
    if args.gui:
        run_gui(args.map_case or "M1_CROSS_BASELINE", args.frames, args.rear_start, args.output)
        return
    if args.map_case:
        runner = run_case(args.map_case, args.frames, args.rear_start)
        args.output.mkdir(parents=True, exist_ok=True)
        _write(args.output / f"wall_topology_sph_{args.map_case.lower()}_summary.csv", _summary_rows(runner))
        print(f"case={args.map_case} candidates={len(runner.candidates)} output={args.output.resolve()}")
        return
    m0 = run_case("M0_STRAIGHT", args.m0_frames, False)
    m1 = run_case("M1_CROSS_BASELINE", args.frames, True)
    verdict = save_outputs(args.output, m0, m1)
    print(f"EXP-034 verdict={verdict}")
    for row in _summary_rows(m1):
        print(row)
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
