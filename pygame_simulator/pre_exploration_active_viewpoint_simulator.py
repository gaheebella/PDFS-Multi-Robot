"""EXP-034 observation-driven partial-topology active viewpoint integration.

This opt-in integration simulator reuses the clean general pipeline physics,
local LiDAR/profile model, frozen EXP-033 wall topology, braking, and Anchor
latch.  It adds no fixed displacement target: a locally observed wall
termination keeps the existing supported local-forward swarm motion active and
a same-side accepted gap stops the LiDAR leader. Geometry and global pose are
appended only for post-hoc audit.
"""

from __future__ import annotations

import argparse
import csv
import json
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
from junction_detection.integration.run_nonforward_viewpoint_magnitude_boundary import (
    _point_segment_distance,
)
from junction_detection.integration.run_provisional_anchor_pointcloud_observation import (
    evaluate_snapshot,
)
from junction_detection.integration.run_wall_topology_branch_opening_diagnostic import (
    _analyze,
    _branch_topology_eval,
    _gt_mouths_eval,
    _match_candidates_eval,
    _self_test,
)
from junction_detection.pointcloud.lidar_profile_junction_detector import (
    GeometryProfileConfig,
    LidarProfileJunctionDetector,
)
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import (
    detect_openings,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import (
    ANCHOR_STATIONARY_DWELL_STEPS,
    DT,
    LIDAR_MAX_RANGE,
    MIN_SPEED,
    ROBOT_RADIUS,
    SAMPLE_PERIOD,
    ActiveViewpointConfig,
    PygameRenderer,
    SimulationRunner,
)

EXPERIMENT_ID = "EXP-034"
DEFAULT_OUTPUT = (
    ROOT
    / "junction_detection/integration/output/partial_topology_active_viewpoint_integration"
)
PIPELINE_STATES = (
    "MOBILE_LIDAR_LEADER",
    "JUNCTION_SUSPECTED",
    "ACTIVE_VIEWPOINT_ACQUISITION",
    "VIEWPOINT_READY",
    "FIXED_ANCHOR",
    "VIEWPOINT_NOT_RESOLVED",
)


def _write(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str] | None = None,
) -> None:
    """Write heterogeneous rows while retaining an empty-file header guard."""
    if not rows and fields is None:
        return
    if fields is None:
        fields = list(rows[0])
        for row in rows[1:]:
            fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _topology_text(topology: dict[str, Any] | None) -> str:
    """Return a compact transition-safe topology description."""
    if topology is None:
        return "UNAVAILABLE"
    return (
        f"L={topology['left_topology']};R={topology['right_topology']};"
        f"AXIAL={topology['forward_topology']}"
    )


def _axis_frame(orientation_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return sensor-local corridor forward and left axes."""
    radians = math.radians(orientation_deg)
    forward = np.array([math.cos(radians), math.sin(radians)])
    left = np.array([-forward[1], forward[0]])
    return forward, left


def _runtime_topology(result: dict[str, Any], orientation_deg: float) -> dict[str, Any]:
    """Classify local side topology using only frozen EXP-033 evidence.

    A valid termination on a local side is PARTIAL.  An accepted gap whose two
    endpoints lie in the same local lateral half-plane is COMPLETE on that
    side.  A straddling accepted gap remains an unlabeled axial structure; GT
    is neither queried nor needed by this runtime classification.
    """
    _, left_axis = _axis_frame(orientation_deg)
    endpoints = {row["endpoint_id"]: row for row in result["endpoints"]}
    valid = [row for row in result["endpoints"] if row["valid"]]
    positive = [row for row in valid if float(np.dot(row["point"], left_axis)) > 0.0]
    negative = [row for row in valid if float(np.dot(row["point"], left_axis)) < 0.0]
    accepted = [row for row in result["gaps"] if row["candidate_valid"]]
    left_complete = []
    right_complete = []
    axial = []
    for gap in accepted:
        first = float(np.dot(endpoints[gap["endpoint_a"]]["point"], left_axis))
        second = float(np.dot(endpoints[gap["endpoint_b"]]["point"], left_axis))
        if first > 0.0 and second > 0.0:
            left_complete.append(gap)
        elif first < 0.0 and second < 0.0:
            right_complete.append(gap)
        else:
            axial.append(gap)
    return {
        "left_topology": (
            "COMPLETE_GAP_TOPOLOGY"
            if left_complete
            else "PARTIAL_GAP_TOPOLOGY"
            if positive
            else "NO_GAP_TOPOLOGY"
        ),
        "right_topology": (
            "COMPLETE_GAP_TOPOLOGY"
            if right_complete
            else "PARTIAL_GAP_TOPOLOGY"
            if negative
            else "NO_GAP_TOPOLOGY"
        ),
        "forward_topology": (
            "COMPLETE_GAP_TOPOLOGY" if axial else "NO_GAP_TOPOLOGY"
        ),
        "left_termination_count": len(positive),
        "right_termination_count": len(negative),
        "left_complete_gap_count": len(left_complete),
        "right_complete_gap_count": len(right_complete),
        "axial_gap_count": len(axial),
        "wall_segment_count": len(result["segments"]),
        "valid_termination_count": len(valid),
        "gap_candidate_count": len(result["gaps"]),
        "accepted_gap_count": len(accepted),
        "viewpoint_ready_evidence": bool(left_complete or right_complete),
    }


def _snapshot(
    runner: "ActiveTopologyRunner", frame: int, row: dict[str, Any]
) -> dict[str, Any]:
    """Build the frozen helper's local scan input without geometry metadata."""
    observation = runner.last_visual[0]
    scan = observation.lidar_scan
    margin = np.finfo(float).eps * max(1.0, scan.max_range) * 64.0
    leader = next(
        robot for robot in runner.world.robots if robot.robot_id == runner.world.lidar_robot_id
    )
    return {
        "context": f"FRAME_{frame}",
        "angles": scan.angles_deg.copy(),
        "ranges": scan.ranges.copy(),
        "hit": scan.ranges < scan.max_range - margin,
        "max_range": scan.max_range,
        # The pose fields below are ignored by runtime topology and are used
        # only after state decisions for post-hoc mouth/visibility evaluation.
        "position_eval": leader.position.copy(),
        "yaw_eval": math.degrees(leader.body_yaw_rad),
        "frame": frame,
        "time": float(row["timestamp"]),
    }


def _posthoc_visibility(
    runner: "ActiveTopologyRunner",
    result: dict[str, Any],
    snapshot: dict[str, Any],
    width: float,
    openings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attach GT mouth visibility and angular matches after control decisions."""
    if runner.geometry.entrance_y is None:
        angular_summary, _ = evaluate_snapshot(runner, snapshot, openings)
        return {
            "left_near_boundary_visible_eval_only": False,
            "left_far_boundary_visible_eval_only": False,
            "right_near_boundary_visible_eval_only": False,
            "right_far_boundary_visible_eval_only": False,
            "left_topology_eval_only": "NO_GAP_TOPOLOGY",
            "right_topology_eval_only": "NO_GAP_TOPOLOGY",
            "forward_topology_eval_only": "NO_GAP_TOPOLOGY",
            "angular_outgoing_count_eval_only": angular_summary[
                "matched_outgoing_count_eval_only"
            ],
            "angular_false_opening_count_eval_only": angular_summary[
                "false_opening_count_eval_only"
            ],
        }
    context = SimpleNamespace(geometry=runner.geometry)
    mouths = _gt_mouths_eval(context, snapshot)
    matches = _match_candidates_eval(
        snapshot["context"], result["gaps"], result["endpoints"], mouths, width
    )
    branches = _branch_topology_eval(
        context, snapshot, result["endpoints"], matches, width
    )
    by_label = {row["branch_eval"]: row for row in branches}

    def boundary(label: str) -> tuple[bool, bool]:
        branch = by_label[label]
        errors = sorted(
            (
                float(branch["nearest_endpoint_error_a_eval"]),
                float(branch["nearest_endpoint_error_b_eval"]),
            )
        )
        return errors[0] <= 0.12 * width, errors[1] <= 0.12 * width

    left_near, left_far = boundary("LEFT")
    right_near, right_far = boundary("RIGHT")
    angular_summary, _ = evaluate_snapshot(runner, snapshot, openings)
    return {
        "left_near_boundary_visible_eval_only": left_near,
        "left_far_boundary_visible_eval_only": left_far,
        "right_near_boundary_visible_eval_only": right_near,
        "right_far_boundary_visible_eval_only": right_far,
        "left_topology_eval_only": by_label["LEFT"]["topology_class_eval"],
        "right_topology_eval_only": by_label["RIGHT"]["topology_class_eval"],
        "forward_topology_eval_only": by_label["FORWARD"]["topology_class_eval"],
        "angular_outgoing_count_eval_only": angular_summary[
            "matched_outgoing_count_eval_only"
        ],
        "angular_false_opening_count_eval_only": angular_summary[
            "false_opening_count_eval_only"
        ],
    }


class ActiveTopologyRunner(SimulationRunner):
    """Observation-driven topology state machine on unchanged general physics."""

    def __init__(self, case_id: str, rear_start: bool = False):
        profile = LidarProfileJunctionDetector(GeometryProfileConfig(LIDAR_MAX_RANGE))
        # Infinite target explicitly disables the legacy fixed-distance stop;
        # topology readiness is the sole acquisition stop decision.
        super().__init__(
            case_id,
            "local_forward",
            profile_detector=profile,
            hold_on_profile_detection=False,
            active_viewpoint_config=ActiveViewpointConfig(math.inf, 0),
        )
        if rear_start:
            _rear_start(self)
        self.pipeline_state = "MOBILE_LIDAR_LEADER"
        self.pending_active_start = False
        self.topology_timeline: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.last_runtime_topology: dict[str, Any] | None = None
        self.last_topology_result: dict[str, Any] | None = None
        self.last_topology_snapshot: dict[str, Any] | None = None
        self.suspect_position_eval: np.ndarray | None = None
        self.suspect_time = math.nan
        self.suspect_frame = -1
        self.ready_time = math.nan
        self.ready_frame = -1
        self.fixed_time = math.nan
        self.fixed_frame = -1
        self.active_start_time = math.nan
        self.active_start_frame = -1
        self.event_once: set[str] = set()
        self.collision_count_at_active_start = 0
        self.snapshots: dict[str, dict[str, Any]] = {}

    def _leader(self) -> Any:
        return next(
            robot
            for robot in self.world.robots
            if robot.robot_id == self.world.lidar_robot_id
        )

    def _displacement(self) -> tuple[float, float]:
        """Return post-hoc forward/lateral displacement from suspicion."""
        if self.suspect_position_eval is None or not np.all(
            np.isfinite(self.world.trusted_corridor_forward)
        ):
            return 0.0, 0.0
        relative = self._leader().position - self.suspect_position_eval
        forward = self.world.trusted_corridor_forward
        lateral = np.array([-forward[1], forward[0]])
        return float(relative @ forward), float(relative @ lateral)

    def _event(
        self,
        name: str,
        frame: int,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
    ) -> None:
        """Append an event once using local displacement and topology only."""
        if name in self.event_once:
            return
        self.event_once.add(name)
        forward, _ = self._displacement()
        self.events.append(
            {
                "event": name,
                "timestamp": self.world.time,
                "frame": frame,
                "local_forward_displacement_from_suspect": forward,
                "topology_before": _topology_text(before),
                "topology_after": _topology_text(after),
            }
        )

    def _start_active_viewpoint(self, frame: int) -> None:
        """Continue the existing swarm local-forward drive without a target."""
        if not self.pending_active_start:
            return
        self.pending_active_start = False
        self.pipeline_state = "ACTIVE_VIEWPOINT_ACQUISITION"
        self.active_start_time = self.world.time
        self.active_start_frame = frame
        self.collision_count_at_active_start = self.world.wall_contacts
        # Do not enter the legacy VIEWPOINT_ADVANCE mode: that opt-in mode
        # intentionally drives only the LiDAR robot for fixed-step rescans.
        # Observation-driven acquisition retains the normal local-forward SPH
        # path so the front leader remains supported by the moving swarm.
        self.world.active_viewpoint_state = "OBSERVATION_DRIVEN_ADVANCE"
        self.world.viewpoint_step_target = math.inf
        self.world.viewpoint_step_progress = 0.0
        self.world.viewpoint_step_actual_progress = 0.0
        self.world.viewpoint_cumulative_advance = 0.0
        self._event(
            "ACTIVE_VIEWPOINT_START",
            frame,
            self.last_runtime_topology,
            self.last_runtime_topology,
        )

    def _stop_on_topology(self, frame: int, topology: dict[str, Any]) -> None:
        """Cut leader drive and reuse existing local braking on READY evidence."""
        self.pipeline_state = "VIEWPOINT_READY"
        self.ready_time = self.world.time
        self.ready_frame = frame
        self._event("VIEWPOINT_READY", frame, self.last_runtime_topology, topology)
        predicted = self.world._predict_viewpoint_braking_distance(self._leader())
        support_gap = float(
            self.world.last_connectivity.get("leader_to_front_pack_gap", math.inf)
        )
        self.world._trigger_viewpoint_brake(
            "WALL_TOPOLOGY_COMPLETE", predicted, support_gap
        )

    def _finish_fixed_anchor(self, frame: int) -> None:
        """Expose the base world's latched stationary pose as FIXED_ANCHOR."""
        if self.pipeline_state == "FIXED_ANCHOR":
            return
        self.pipeline_state = "FIXED_ANCHOR"
        self.fixed_time = self.world.time
        self.fixed_frame = frame
        self.world.active_viewpoint_state = "PROVISIONAL_FIXED_ANCHOR"
        if (
            self.last_topology_snapshot is not None
            and self.last_topology_result is not None
            and self.last_runtime_topology is not None
        ):
            self.snapshots.setdefault(
                "fixed_anchor",
                self._capture(
                    frame,
                    self.last_topology_snapshot,
                    self.last_topology_result,
                    self.last_runtime_topology,
                ),
            )
        self._event(
            "FIXED_ANCHOR",
            frame,
            self.last_runtime_topology,
            self.last_runtime_topology,
        )

    def step(self, frame: int) -> dict[str, Any] | None:
        """Advance physics, then evaluate topology at the existing 0.1 s cadence."""
        self._start_active_viewpoint(frame)
        row = super().step(frame)
        if self.world.provisional_fixed_anchor:
            self._finish_fixed_anchor(frame)
        if row is None:
            return None

        snapshot = _snapshot(self, frame, row)
        profile = self.last_profile_result
        width = float(profile["estimated_corridor_width"])
        orientation = float(profile["stable_corridor_orientation_deg"])
        topology_available = bool(
            profile["corridor_model_initialized"]
            and math.isfinite(width)
            and width > 0.0
            and math.isfinite(orientation)
        )
        result = None
        topology = None
        openings = list(
            detect_openings(snapshot["angles"].copy(), snapshot["ranges"].copy())
        )
        if topology_available:
            result = _analyze(snapshot["context"], snapshot, width)
            topology = _runtime_topology(result, orientation)

            # The control transition consumes only `topology` above. All
            # geometry/GT evaluation occurs later when building the CSV row.
            previous = self.last_runtime_topology
            if self.pipeline_state == "MOBILE_LIDAR_LEADER" and (
                topology["left_topology"] == "PARTIAL_GAP_TOPOLOGY"
                or topology["right_topology"] == "PARTIAL_GAP_TOPOLOGY"
                or topology["viewpoint_ready_evidence"]
            ):
                self.pipeline_state = "JUNCTION_SUSPECTED"
                self.suspect_position_eval = self._leader().position.copy()
                self.suspect_time = self.world.time
                self.suspect_frame = frame
                self.world.freeze_trusted_corridor_frame(orientation, width)
                self.world.junction_detection_latched = True
                self.world.junction_detection_source = "WALL_TOPOLOGY_PARTIAL"
                self.world.detection_position_eval = self._leader().position.copy()
                self._event("JUNCTION_SUSPECTED", frame, previous, topology)
                if topology["left_topology"] == "PARTIAL_GAP_TOPOLOGY":
                    self._event("LEFT_PARTIAL", frame, previous, topology)
                if topology["right_topology"] == "PARTIAL_GAP_TOPOLOGY":
                    self._event("RIGHT_PARTIAL", frame, previous, topology)
                self.pending_active_start = True
                self.snapshots.setdefault(
                    "junction_suspected", self._capture(frame, snapshot, result, topology)
                )
            elif self.pipeline_state == "ACTIVE_VIEWPOINT_ACQUISITION" and topology[
                "viewpoint_ready_evidence"
            ]:
                if topology["left_topology"] == "COMPLETE_GAP_TOPOLOGY":
                    self._event("LEFT_COMPLETE", frame, previous, topology)
                if topology["right_topology"] == "COMPLETE_GAP_TOPOLOGY":
                    self._event("RIGHT_COMPLETE", frame, previous, topology)
                self._stop_on_topology(frame, topology)
                self.snapshots.setdefault(
                    "topology_change", self._capture(frame, snapshot, result, topology)
                )
                self.snapshots.setdefault(
                    "viewpoint_ready", self._capture(frame, snapshot, result, topology)
                )

            if self.pipeline_state == "MOBILE_LIDAR_LEADER":
                self.snapshots["normal_corridor"] = self._capture(
                    frame, snapshot, result, topology
                )
            if self.pipeline_state == "ACTIVE_VIEWPOINT_ACQUISITION":
                self.snapshots.setdefault(
                    "active_viewpoint_moving",
                    self._capture(frame, snapshot, result, topology),
                )
            self.snapshots["last_valid_frame"] = self._capture(
                frame, snapshot, result, topology
            )
            self.last_runtime_topology = topology
            self.last_topology_result = result
            self.last_topology_snapshot = snapshot

        # Post-hoc fields are deliberately computed only after state decisions.
        eval_fields = (
            _posthoc_visibility(self, result, snapshot, width, openings)
            if result is not None
            else {
                "left_near_boundary_visible_eval_only": False,
                "left_far_boundary_visible_eval_only": False,
                "right_near_boundary_visible_eval_only": False,
                "right_far_boundary_visible_eval_only": False,
                "left_topology_eval_only": "UNAVAILABLE",
                "right_topology_eval_only": "UNAVAILABLE",
                "forward_topology_eval_only": "UNAVAILABLE",
                "angular_outgoing_count_eval_only": 0,
                "angular_false_opening_count_eval_only": 0,
            }
        )
        forward_displacement, lateral_drift = self._displacement()
        leader = self._leader()
        minimum_clearance = min(
            _point_segment_distance(leader.position, wall) for wall in self.geometry.walls
        )
        topology_fields = topology or {
            "left_topology": "UNAVAILABLE",
            "right_topology": "UNAVAILABLE",
            "forward_topology": "UNAVAILABLE",
            "wall_segment_count": 0,
            "valid_termination_count": 0,
            "gap_candidate_count": 0,
            "accepted_gap_count": 0,
            "viewpoint_ready_evidence": False,
        }
        timeline_row = {
            "timestamp": float(row["timestamp"]),
            "frame": frame,
            "pipeline_state": self.pipeline_state,
            "lidar_x_eval_only": float(leader.position[0]),
            "lidar_y_eval_only": float(leader.position[1]),
            "local_corridor_orientation_deg": orientation,
            "estimated_corridor_width": width,
            "wall_segment_count": topology_fields["wall_segment_count"],
            "valid_termination_count": topology_fields["valid_termination_count"],
            "gap_candidate_count": topology_fields["gap_candidate_count"],
            "accepted_gap_count": topology_fields["accepted_gap_count"],
            "left_topology": topology_fields["left_topology"],
            "right_topology": topology_fields["right_topology"],
            "forward_topology": topology_fields["forward_topology"],
            **eval_fields,
            "angular_opening_count": len(openings),
            "current_speed": float(np.linalg.norm(leader.velocity)),
            "movement_active": self.pipeline_state
            in {"MOBILE_LIDAR_LEADER", "ACTIVE_VIEWPOINT_ACQUISITION"},
            "junction_suspected": self.suspect_position_eval is not None,
            "viewpoint_ready": math.isfinite(self.ready_time),
            "fixed_anchor": self.pipeline_state == "FIXED_ANCHOR",
            "local_forward_displacement_from_suspect_eval_only": forward_displacement,
            "lateral_drift_from_suspect_eval_only": lateral_drift,
            "minimum_wall_clearance_eval_only": minimum_clearance,
            "leader_footprint_valid_eval_only": bool(
                self.geometry.walkable(leader.position, ROBOT_RADIUS)
            ),
            "wall_contact_count_all_robots_eval_only": int(row["wall_contact_count"]),
            "outside_free_space_robot_count_eval_only": int(
                row["outside_free_space_robot_count"]
            ),
            "nan_inf_state_count": int(row["nan_inf_state_count"]),
            "leader_connected": bool(row["leader_connected"]),
            "leader_communication_connected": bool(
                row["leader_communication_connected"]
            ),
        }
        self.topology_timeline.append(timeline_row)
        row.update(timeline_row)
        return row

    @staticmethod
    def _capture(
        frame: int,
        snapshot: dict[str, Any],
        result: dict[str, Any],
        topology: dict[str, Any],
    ) -> dict[str, Any]:
        """Retain a lightweight immutable diagnostic frame."""
        return {
            "frame": frame,
            "snapshot": {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in snapshot.items()
            },
            "result": result,
            "topology": dict(topology),
        }

    def finalize(self, frame_limit: int) -> None:
        """Record the bounded diagnostic outcome without a runtime distance stop."""
        if self.pipeline_state in {
            "JUNCTION_SUSPECTED",
            "ACTIVE_VIEWPOINT_ACQUISITION",
            "VIEWPOINT_READY",
        }:
            before = self.last_runtime_topology
            self.pipeline_state = "VIEWPOINT_NOT_RESOLVED"
            self._event("VIEWPOINT_NOT_RESOLVED", frame_limit, before, before)
            if self.topology_timeline:
                final_row = dict(self.topology_timeline[-1])
                final_row.update(
                    {
                        "timestamp": self.world.time,
                        "frame": frame_limit,
                        "pipeline_state": "VIEWPOINT_NOT_RESOLVED",
                        "movement_active": False,
                    }
                )
                self.topology_timeline.append(final_row)


class ActiveTopologyRenderer(PygameRenderer):
    """Minimal GUI overlay for the new state machine and frozen topology."""

    def draw(self, runner: ActiveTopologyRunner, frame: int) -> None:
        super().draw(runner, frame)
        pygame = self.pygame
        leader = runner._leader()
        colors = {
            "MOBILE_LIDAR_LEADER": (255, 230, 40),
            "JUNCTION_SUSPECTED": (255, 160, 40),
            "ACTIVE_VIEWPOINT_ACQUISITION": (70, 210, 255),
            "VIEWPOINT_READY": (90, 255, 130),
            "FIXED_ANCHOR": (255, 80, 220),
            "VIEWPOINT_NOT_RESOLVED": (255, 90, 90),
        }
        pygame.draw.circle(
            self.screen,
            colors.get(runner.pipeline_state, (255, 255, 255)),
            self.world_to_screen(leader.position),
            8,
            2,
        )
        if self.show_diagnostics and runner.last_topology_result is not None:
            snapshot = runner.last_topology_snapshot
            result = runner.last_topology_result
            yaw = math.radians(float(snapshot["yaw_eval"]))
            rotation = np.array(
                [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
            )

            def world(point: np.ndarray) -> np.ndarray:
                return np.asarray(snapshot["position_eval"]) + rotation @ np.asarray(point)

            for segment in result["segments"]:
                pygame.draw.line(
                    self.screen,
                    (255, 185, 65),
                    self.world_to_screen(world(segment["start"])),
                    self.world_to_screen(world(segment["end"])),
                    2,
                )
            endpoints = {row["endpoint_id"]: row for row in result["endpoints"]}
            for endpoint in result["endpoints"]:
                if endpoint["valid"]:
                    pygame.draw.circle(
                        self.screen,
                        (255, 80, 70),
                        self.world_to_screen(world(endpoint["point"])),
                        5,
                        2,
                    )
            for gap in result["gaps"]:
                if not gap["candidate_valid"]:
                    continue
                pygame.draw.line(
                    self.screen,
                    (90, 255, 120),
                    self.world_to_screen(world(endpoints[gap["endpoint_a"]]["point"])),
                    self.world_to_screen(world(endpoints[gap["endpoint_b"]]["point"])),
                    3,
                )
        topology = runner.last_runtime_topology or {
            "left_topology": "UNAVAILABLE",
            "right_topology": "UNAVAILABLE",
        }
        lines = (
            f"EXP-034 State: {runner.pipeline_state}",
            f"LEFT: {topology['left_topology']}",
            f"RIGHT: {topology['right_topology']}",
            "D toggles wall-topology overlay",
        )
        for index, text in enumerate(lines):
            surface = self.font.render(text, True, (255, 245, 180))
            self.screen.blit(surface, (720, 12 + index * 22))
        pygame.display.flip()


def _plot_frame(path: Path, capture: dict[str, Any], title: str) -> None:
    """Save one local Point Cloud/topology frame without a GUI dependency."""
    snapshot, result, topology = (
        capture["snapshot"],
        capture["result"],
        capture["topology"],
    )
    theta = np.deg2rad(snapshot["angles"])
    points = np.column_stack(
        (snapshot["ranges"] * np.cos(theta), snapshot["ranges"] * np.sin(theta))
    )
    fig, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
    axis.scatter(points[snapshot["hit"], 0], points[snapshot["hit"], 1], s=8, color="0.65")
    for segment in result["segments"]:
        axis.plot(
            [segment["start"][0], segment["end"][0]],
            [segment["start"][1], segment["end"][1]],
            color="tab:blue",
            linewidth=2,
        )
    endpoints = {row["endpoint_id"]: row for row in result["endpoints"]}
    for endpoint in result["endpoints"]:
        if endpoint["valid"]:
            axis.scatter(*endpoint["point"], color="tab:orange", s=55, marker="o")
    for gap in result["gaps"]:
        if not gap["candidate_valid"]:
            continue
        first = endpoints[gap["endpoint_a"]]["point"]
        second = endpoints[gap["endpoint_b"]]["point"]
        axis.plot(
            [first[0], second[0]],
            [first[1], second[1]],
            color="tab:green",
            linewidth=3,
        )
    axis.scatter(0.0, 0.0, marker="*", s=120, color="black")
    short = lambda value: value.replace("_GAP_TOPOLOGY", "")
    axis.set(
        xlabel="LiDAR-local x",
        ylabel="LiDAR-local y",
        aspect="equal",
        xlim=(-160, 160),
        ylim=(-160, 160),
    )
    axis.set_title(
        f"{title} | frame={capture['frame']}\n"
        f"LEFT={short(topology['left_topology'])}  "
        f"RIGHT={short(topology['right_topology'])}  "
        f"AXIAL={short(topology['forward_topology'])}",
        fontsize=11,
    )
    axis.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_case(case_id: str, frames: int, rear_start: bool = False) -> ActiveTopologyRunner:
    """Run one bounded deterministic state-machine case."""
    runner = ActiveTopologyRunner(case_id, rear_start)
    for frame in range(frames):
        runner.step(frame)
        if runner.pipeline_state == "FIXED_ANCHOR":
            # Continue briefly to verify the latched robot remains stopped.
            if frame - runner.fixed_frame >= ANCHOR_STATIONARY_DWELL_STEPS:
                break
    runner.finalize(frames)
    return runner


def _summary(runner: ActiveTopologyRunner, frame_limit: int) -> dict[str, Any]:
    """Summarize state timing, actual motion, and post-hoc physical validity."""
    active = [
        row
        for row in runner.topology_timeline
        if row["pipeline_state"]
        in {"ACTIVE_VIEWPOINT_ACQUISITION", "VIEWPOINT_READY", "FIXED_ANCHOR", "VIEWPOINT_NOT_RESOLVED"}
    ]
    forward, lateral = runner._displacement()
    collision_delta = max(
        0, runner.world.wall_contacts - runner.collision_count_at_active_start
    )
    partial_after_suspect = [
        row
        for row in active
        if row["left_topology"] == "PARTIAL_GAP_TOPOLOGY"
        or row["right_topology"] == "PARTIAL_GAP_TOPOLOGY"
    ]
    minimum_clearance = min(
        (row["minimum_wall_clearance_eval_only"] for row in active),
        default=math.inf,
    )
    footprint_valid = all(
        row["leader_footprint_valid_eval_only"] for row in active
    )
    leader = runner._leader()
    fixed_position_error = (
        float(np.linalg.norm(leader.position - runner.world.anchor_position))
        if runner.world.anchor_position is not None
        else math.nan
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "map_case": runner.geometry.case_id,
        "frame_limit": frame_limit,
        "final_pipeline_state": runner.pipeline_state,
        "suspected": runner.suspect_position_eval is not None,
        "suspect_frame": runner.suspect_frame,
        "suspect_time": runner.suspect_time,
        "active_start_frame": runner.active_start_frame,
        "active_start_time": runner.active_start_time,
        "left_complete_observed": "LEFT_COMPLETE" in runner.event_once,
        "right_complete_observed": "RIGHT_COMPLETE" in runner.event_once,
        "viewpoint_ready": math.isfinite(runner.ready_time),
        "viewpoint_ready_frame": runner.ready_frame,
        "viewpoint_ready_time": runner.ready_time,
        "fixed_anchor": math.isfinite(runner.fixed_time),
        "fixed_anchor_frame": runner.fixed_frame,
        "fixed_anchor_time": runner.fixed_time,
        "fixed_anchor_final_position_error_eval_only": fixed_position_error,
        "fixed_anchor_final_speed": float(np.linalg.norm(leader.velocity)),
        "forward_displacement_from_suspect_eval_only": forward,
        "lateral_drift_from_suspect_eval_only": lateral,
        "elapsed_from_suspect": (
            runner.world.time - runner.suspect_time
            if math.isfinite(runner.suspect_time)
            else 0.0
        ),
        "active_sample_count": len(active),
        "partial_active_sample_count": len(partial_after_suspect),
        "minimum_wall_clearance_eval_only": minimum_clearance,
        "leader_footprint_valid_all_active_eval_only": footprint_valid,
        "leader_collision_or_invalid_movement_eval_only": (
            not footprint_valid or minimum_clearance < ROBOT_RADIUS - 1.0e-9
        ),
        "all_robot_wall_contact_delta_during_active_eval_only": collision_delta,
        "maximum_outside_free_space_robot_count_eval_only": max(
            (row["outside_free_space_robot_count_eval_only"] for row in active),
            default=0,
        ),
        "maximum_nan_inf": max(
            (row["nan_inf_state_count"] for row in active), default=0
        ),
        "leader_connected_all_active": all(
            row["leader_connected"] for row in active
        ),
        "leader_communication_connected_all_active": all(
            row["leader_communication_connected"] for row in active
        ),
        "distance_threshold_used_for_stop": False,
        "GT_map_used_for_control": False,
    }


def _verdict(m0: dict[str, Any], m1: dict[str, Any]) -> str:
    """Return one requested A-E outcome without changing any threshold."""
    if m0["suspected"]:
        return "D_NORMAL_CORRIDOR_FALSE_SUSPECT"
    unsafe = (
        m1["leader_collision_or_invalid_movement_eval_only"]
        or m1["maximum_outside_free_space_robot_count_eval_only"] > 0
        or m1["maximum_nan_inf"] > 0
    )
    if unsafe and not m1["viewpoint_ready"]:
        return "E_VIEWPOINT_ACQUISITION_MOTION_INVALID"
    if m1["viewpoint_ready"] and m1["fixed_anchor"]:
        return "A_PARTIAL_TO_COMPLETE_ACTIVE_VIEWPOINT_SUCCESS"
    if m1["partial_active_sample_count"] > 0:
        return "B_PARTIAL_PERSISTS_WITHOUT_COMPLETE"
    if m1["suspected"]:
        return "C_PARTIAL_EVIDENCE_DISAPPEARS_DURING_MOTION"
    return "M1_JUNCTION_NOT_SUSPECTED_WITHIN_BOUND"


def save_outputs(
    output: Path,
    m0: ActiveTopologyRunner,
    m1: ActiveTopologyRunner,
    m0_frames: int,
    m1_frames: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Write lifecycle CSVs, representative frames, summary, and verdict."""
    output.mkdir(parents=True, exist_ok=True)
    m0_summary, m1_summary = _summary(m0, m0_frames), _summary(m1, m1_frames)
    verdict = _verdict(m0_summary, m1_summary)
    _write(output / "active_viewpoint_timeline_m0.csv", m0.topology_timeline)
    _write(output / "active_viewpoint_timeline_m1.csv", m1.topology_timeline)
    _write(output / "active_viewpoint_timeline.csv", m1.topology_timeline)
    event_fields = [
        "event",
        "timestamp",
        "frame",
        "local_forward_displacement_from_suspect",
        "topology_before",
        "topology_after",
    ]
    _write(output / "active_viewpoint_events_m0.csv", m0.events, event_fields)
    _write(output / "active_viewpoint_events_m1.csv", m1.events)
    _write(output / "active_viewpoint_events.csv", m1.events)
    _write(output / "active_viewpoint_summary.csv", [m0_summary, m1_summary])
    _write(
        output / "active_viewpoint_verdict.csv",
        [
            {
                "experiment_id": EXPERIMENT_ID,
                "verdict": verdict,
                "runtime_inputs": "local_angles_ranges,corridor_orientation,W_hat,wall_segments,terminations,gaps,local_motion",
                "GT_map_used_for_control": False,
                "fixed_displacement_stop_used": False,
                "detector_threshold_modified": False,
                "wall_topology_threshold_modified": False,
                "production_simulator_modified": False,
            }
        ],
    )
    for name, title in (
        ("normal_corridor", "Normal corridor"),
        ("junction_suspected", "First PARTIAL / JUNCTION_SUSPECTED"),
        ("active_viewpoint_moving", "ACTIVE_VIEWPOINT movement"),
        ("topology_change", "Topology change"),
        ("viewpoint_ready", "VIEWPOINT_READY"),
        ("fixed_anchor", "FIXED_ANCHOR"),
    ):
        capture = m1.snapshots.get(name)
        if capture is not None:
            _plot_frame(output / f"{name}.png", capture, title)
    if not m1_summary["viewpoint_ready"] and "last_valid_frame" in m1.snapshots:
        _plot_frame(
            output / "last_valid_frame.png",
            m1.snapshots["last_valid_frame"],
            "Last valid frame / VIEWPOINT_NOT_RESOLVED",
        )
    return m0_summary, m1_summary, verdict


def _state_machine_self_test() -> None:
    """Validate NO→PARTIAL and same-side COMPLETE classification mechanics."""
    endpoints = [
        {"endpoint_id": 0, "point": np.array([10.0, 4.0]), "valid": True},
        {"endpoint_id": 1, "point": np.array([20.0, 5.0]), "valid": True},
    ]
    result = {"segments": [], "endpoints": endpoints, "gaps": []}
    partial = _runtime_topology(result, 0.0)
    assert partial["left_topology"] == "PARTIAL_GAP_TOPOLOGY"
    result["gaps"] = [
        {"endpoint_a": 0, "endpoint_b": 1, "candidate_valid": True}
    ]
    complete = _runtime_topology(result, 0.0)
    assert complete["left_topology"] == "COMPLETE_GAP_TOPOLOGY"
    assert complete["viewpoint_ready_evidence"]


def run_gui(case_id: str, frames: int, rear_start: bool, output: Path) -> None:
    """Run the actual Pygame integration with pause/reset/diagnostic controls."""
    import pygame

    runner = ActiveTopologyRunner(case_id, rear_start)
    renderer = ActiveTopologyRenderer(runner.geometry, gui_scale=0.75)
    frame = 0
    running = True
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
                    runner = ActiveTopologyRunner(case_id, rear_start)
                    renderer.configure_camera(runner.geometry)
                    frame = 0
        if not renderer.paused:
            runner.step(frame)
            frame += 1
        renderer.draw(runner, frame)
    runner.finalize(frames if frames > 0 else frame)
    # GUI output uses an empty M0 placeholder only when explicitly requested;
    # headless validation remains the authoritative M0 negative control.
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "active_viewpoint_timeline_gui.csv", runner.topology_timeline)
    _write(output / "active_viewpoint_events_gui.csv", runner.events)
    pygame.quit()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-case", default="M1_CROSS_BASELINE", choices=("M0_STRAIGHT", "M1_CROSS_BASELINE"))
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--m0-frames", type=int, default=300)
    parser.add_argument("--rear-start", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _self_test()
    _state_machine_self_test()
    if args.gui:
        run_gui(args.map_case, args.frames, args.rear_start, args.output)
        return
    m0 = run_case("M0_STRAIGHT", args.m0_frames)
    m1 = run_case("M1_CROSS_BASELINE", args.frames, rear_start=True)
    m0_summary, m1_summary, verdict = save_outputs(
        args.output, m0, m1, args.m0_frames, args.frames
    )
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "verdict": verdict,
                "m0": m0_summary,
                "m1": m1_summary,
                "events": m1.events,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
