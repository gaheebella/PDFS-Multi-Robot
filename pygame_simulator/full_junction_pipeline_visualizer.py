"""Read-only GUI for the frozen full Junction pipeline.

The EXP-048 front end and EXP-049 stationary pipeline own every algorithmic
decision.  This module records local motion for the frozen EXP-040 identity
helpers, copies completed results, and renders them.  No rendered value feeds
the detector, controller, topology, candidate, or identity layers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.integration.branch_candidate_identity import (  # noqa: E402
    LocalMotionHistory,
    PersistentCandidateIdentity,
    best_existing_match,
    identity_from_observation,
    incoming_features,
    incoming_path_match,
    make_observation,
    pairwise_features,
    rotation,
)
from junction_detection.integration.run_provisional_anchor_stationary_pointcloud_integration import (  # noqa: E402
    BOOTSTRAP_ALIAS,
    M0_ALIAS,
    IntegrationRun,
    _run_signature,
)
from junction_detection.integration.run_wall_topology_branch_opening_diagnostic import (  # noqa: E402
    _gt_mouths_eval,
)
from junction_detection.pointcloud.general_branch_candidate import (  # noqa: E402
    describe_accepted_gap,
)
from pygame_simulator.lidar_junction_controlled_approach_visualizer import (  # noqa: E402
    BASELINE_ALIAS,
    IntegratedRenderer,
    IntegratedSnapshot,
    _empty,
    _finite,
)
from pygame_simulator.lidar_junction_detection_visualizer import (  # noqa: E402
    COLORS,
    MAIN_RECT,
    PROFILE_RECT,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import (  # noqa: E402
    DT,
    SAMPLE_PERIOD,
)
from pygame_simulator.pre_exploration_wall_topology_sph_validation import (  # noqa: E402
    MOTION_WINDOW_SECONDS,
    _axis_frame,
)


EXPERIMENT_NAME = "Full Junction Pipeline Read-Only GUI Integration"
MAP_CASES = (BOOTSTRAP_ALIAS, M0_ALIAS, BASELINE_ALIAS)
DEFAULT_OUTPUT = (
    ROOT
    / "junction_detection/integration/output/full_junction_pipeline_visualizer"
)
PROTECTED_PATHS = (
    "pygame_simulator/pre_exploration_general_pipeline_simulator.py",
    "junction_detection/pointcloud/lidar_profile_junction_detector.py",
    "junction_detection/integration/run_lidar_profile_junction_detection.py",
    "pygame_simulator/lidar_junction_controlled_approach_visualizer.py",
    "junction_detection/integration/run_provisional_anchor_stationary_pointcloud_integration.py",
    "junction_detection/integration/run_wall_topology_branch_opening_diagnostic.py",
    "junction_detection/pointcloud/general_branch_candidate.py",
    "junction_detection/integration/branch_candidate_identity.py",
    "junction_detection/integration/run_candidate_identity_history_benchmark.py",
    "pygame_simulator/pre_exploration_wall_topology_sph_validation.py",
    "junction_detection/pointcloud/pointcloud_junction_detector_sensor_enhanced.py",
)

OVERLAY_COLORS = {
    "point": (164, 174, 184),
    "segment": (72, 202, 228),
    "corner": (255, 184, 76),
    "termination": (255, 83, 92),
    "scan_limit": (145, 155, 168),
    "unstable": (170, 120, 220),
    "gap_accepted": (69, 214, 126),
    "gap_rejected": (255, 153, 43),
    "candidate": (80, 245, 225),
    "history": (180, 125, 255),
    "gt": (255, 210, 70),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_hashes() -> dict[str, str]:
    return {path: _sha256(ROOT / path) for path in PROTECTED_PATHS}


@dataclass(frozen=True)
class IdentityDisplayDecision:
    display_id: str
    source_candidate_id: str
    decision: str
    matched_id: str
    reason: str
    incoming_score: float
    duplicate_score: float


class FullPipelineSession:
    """Drive EXP-049 and attach frozen identity results without feedback."""

    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self.integration = IntegrationRun(case_id)
        self.local_history = LocalMotionHistory(MOTION_WINDOW_SECONDS)
        self.identity_records: dict[str, PersistentCandidateIdentity] = {}
        self.identity_decisions: list[IdentityDisplayDecision] = []
        self.display_events: list[dict[str, Any]] = []
        self._pipeline_event_index = 0
        self._identity_connected = False
        self._analysis_anchor_position: np.ndarray | None = None
        self._analysis_yaw_rad: float | None = None
        self._history_local: tuple[np.ndarray, ...] = ()
        self._parent_axis_local = np.zeros(2)
        self.eval_mouths: list[dict[str, Any]] = []
        self.gap_general_status: dict[int, str] = {}

    @property
    def frontend(self) -> Any:
        return self.integration.frontend

    @property
    def current(self) -> IntegratedSnapshot | None:
        return self.frontend.current

    @property
    def runtime_result(self) -> Any | None:
        return self.integration.runtime_result

    @property
    def topology_visible(self) -> bool:
        return bool(
            self.current is not None
            and self.runtime_result is not None
            and self.current.physics_frame >= self.runtime_result.analysis_frame
        )

    @property
    def identity_status(self) -> str:
        if self.case_id == M0_ALIAS:
            return "NOT INVOKED"
        if self.runtime_result is None:
            return "WAITING_FOR_GENERAL_CANDIDATES"
        if not self._identity_connected:
            return "WAITING_FOR_HISTORY"
        return "AVAILABLE"

    @property
    def sph_status(self) -> str:
        return "NOT CONNECTED"

    def restart(self) -> None:
        self.__init__(self.case_id)

    def _observe_local_motion(self) -> None:
        leader = self.frontend._leader()
        self.local_history.observe(
            float(self.frontend.runner.world.time),
            float(leader.body_yaw_rad),
            np.asarray(leader.observed_velocity, dtype=float),
        )

    def _copy_pipeline_events(self) -> None:
        new_rows = self.integration.pipeline_rows[self._pipeline_event_index :]
        self._pipeline_event_index = len(self.integration.pipeline_rows)
        for row in new_rows:
            self.display_events.append(dict(row))

    def _add_display_event(self, event: str) -> None:
        runtime = self.runtime_result
        if runtime is None:
            return
        self.display_events.append(
            {
                "frame": runtime.analysis_frame,
                "time": runtime.analysis_time,
                "event": event,
                "pipeline_state": "CANDIDATE_IDENTITY_READY",
            }
        )

    def _connect_identity(self) -> None:
        """Invoke only EXP-040 helper APIs over live local inputs."""
        runtime = self.runtime_result
        snapshot = self.current
        if runtime is None or snapshot is None or self._identity_connected:
            return
        leader = self.frontend._leader()
        self._analysis_anchor_position = np.asarray(leader.position, dtype=float).copy()
        self._analysis_yaw_rad = math.radians(float(snapshot.lidar_yaw_deg))

        yaw_odom = self.local_history.current_yaw_odom
        to_current_local = rotation(yaw_odom).T
        current_odom = self.local_history.odom_position.copy()
        self._history_local = tuple(
            to_current_local @ (np.asarray(row[1]) - current_odom)
            for row in self.local_history.samples
        )
        history_snapshot = self.local_history.snapshot()
        self._parent_axis_local = np.asarray(
            history_snapshot["parent_axis_local"], dtype=float
        ).copy()

        topology = runtime.topology_result
        endpoints = {
            int(row["endpoint_id"]): row for row in topology["endpoints"]
        }
        segments = {
            int(row["segment_id"]): row for row in topology["segments"]
        }
        corridor_axis, _ = _axis_frame(float(snapshot.stable_orientation_deg))
        candidate_gap_ids = {
            int(row["source_gap_id"]) for row in runtime.candidate_rows
        }
        for gap in topology["gaps"]:
            gap_id = int(gap["gap_id"])
            topology_state = (
                "TOPOLOGY_ACCEPTED" if gap["candidate_valid"] else "TOPOLOGY_REJECTED"
            )
            self._add_display_event(f"GAP_CREATED G{gap_id} {topology_state}")
            if gap_id in candidate_gap_ids:
                self.gap_general_status[gap_id] = "CANDIDATE_CREATED"
            else:
                descriptor = describe_accepted_gap(
                    gap, endpoints, segments, corridor_axis
                )
                self.gap_general_status[gap_id] = descriptor.rejection_reason
                self._add_display_event(
                    f"GENERAL_CANDIDATE_REJECTED G{gap_id} {descriptor.rejection_reason}"
                )

        width = float(snapshot.estimated_corridor_width)
        for index, (source_id, internal) in enumerate(
            runtime.candidate_internal.items()
        ):
            display_id = f"C{index}"
            descriptor = internal["descriptor"]
            gap = internal["gap"]
            observation = make_observation(
                f"GUI_O{index}",
                runtime.analysis_time,
                runtime.analysis_frame,
                int(gap["gap_id"]),
                descriptor,
                self.local_history,
            )
            incoming = incoming_features(observation, self.local_history, width)
            pairs = [
                (candidate_id, pairwise_features(observation, identity, width))
                for candidate_id, identity in self.identity_records.items()
            ]
            duplicate_score = max(
                (
                    max(float(features["axis_dot"]), float(features["spatial_overlap"]))
                    for _, features in pairs
                ),
                default=0.0,
            )
            if incoming_path_match(incoming):
                decision = "KNOWN_PARENT_PATH"
                matched_id = "KNOWN_PARENT_PATH"
                reason = "DIRECTED_LOCAL_HISTORY_PARENT_MATCH"
            else:
                matched_id, _, reason = best_existing_match(pairs)
                if matched_id is not None:
                    decision = "MERGE_EXISTING"
                    self.identity_records[matched_id].append(observation)
                else:
                    decision = "NEW_OUTGOING"
                    matched_id = display_id
                    reason = "NO_PARENT_OR_EXISTING_IDENTITY_MATCH"
                    self.identity_records[display_id] = identity_from_observation(
                        display_id, observation
                    )
            self.identity_decisions.append(
                IdentityDisplayDecision(
                    display_id=display_id,
                    source_candidate_id=source_id,
                    decision=decision,
                    matched_id=str(matched_id),
                    reason=reason,
                    incoming_score=float(incoming["incoming_axis_dot"]),
                    duplicate_score=float(duplicate_score),
                )
            )
            self._add_display_event(
                f"GENERAL_CANDIDATE_CREATED {display_id} ({source_id})"
            )
            identity_event = {
                "NEW_OUTGOING": "IDENTITY_NEW_OUTGOING",
                "KNOWN_PARENT_PATH": "IDENTITY_PARENT",
                "MERGE_EXISTING": "IDENTITY_MERGE",
            }[decision]
            suffix = (
                f"->{matched_id}" if decision == "MERGE_EXISTING" else ""
            )
            self._add_display_event(f"{identity_event} {display_id}{suffix}")

        self._identity_connected = True
        # Evaluation is deliberately attached only after runtime topology,
        # General Candidate, and identity decisions are final.
        self.integration._posthoc_evaluate()
        eval_snapshot = self.integration._eval_snapshot()
        self.eval_mouths = list(
            _gt_mouths_eval(
                SimpleNamespace(geometry=self.frontend.runner.geometry),
                eval_snapshot,
            )
        )

    def step(self) -> None:
        self._observe_local_motion()
        self.integration.step()
        self._copy_pipeline_events()
        if self.runtime_result is not None and not self._identity_connected:
            self._connect_identity()

    def run(self, frames: int) -> "FullPipelineSession":
        for _ in range(frames):
            self.step()
        return self

    def step_sample(self, direction: int) -> None:
        if direction < 0:
            self.frontend.view_index = max(0, self.frontend.view_index - 1)
            return
        if self.frontend.view_index + 1 < len(self.frontend.snapshots):
            self.frontend.view_index += 1
            return
        sample_count = len(self.frontend.snapshots)
        for _ in range(max(1, round(SAMPLE_PERIOD / DT)) + 1):
            self.step()
            if len(self.frontend.snapshots) > sample_count:
                break

    def pipeline_state(self) -> str:
        if self.topology_visible and self._identity_connected:
            return "CANDIDATE_IDENTITY_READY"
        if self.topology_visible:
            return "BRANCH_CANDIDATES_READY"
        return "WAITING" if self.current is None else self.current.pipeline_state

    def summary(self) -> dict[str, Any]:
        front = self.frontend.summary()
        runtime = self.runtime_result
        topology = None if runtime is None else runtime.topology_result
        return {
            "case_id": self.case_id,
            "ready_frame": front["ready_frame"],
            "first_open_frame": front["first_open_frame"],
            "detection_frame": front["first_detection_frame"],
            "bilateral_frame": front["bilateral_entry_frame"],
            "brake_ready_frame": front["candidate_b_trigger_frame"],
            "braking_frame": front["braking_start_frame"],
            "stop_frame": front["stop_frame"],
            "anchor_frame": front["anchor_frame"],
            "analysis_frame": "" if runtime is None else runtime.analysis_frame,
            "wall_segments": 0 if topology is None else len(topology["segments"]),
            "valid_endpoints": 0 if topology is None else sum(bool(row["valid"]) for row in topology["endpoints"]),
            "physical_gaps": 0 if topology is None else len(topology["gaps"]),
            "general_candidates": 0 if runtime is None else len(runtime.candidate_rows),
            "true_outgoing_recovered": sum(
                bool(row["is_true_outgoing_eval_only"])
                for row in self.integration.branch_eval_rows
            ),
            "false_candidates": sum(
                bool(row["is_false_candidate_eval_only"])
                for row in self.integration.branch_eval_rows
            ),
            "identity_status": self.identity_status,
            "identity_decisions": [row.decision for row in self.identity_decisions],
            "sph_status": self.sph_status,
            "pointcloud_invocations": self.integration.pointcloud_invocation_count,
            "pointcloud_before_hold": self.integration.pointcloud_invoked_before_hold_count,
            "runtime_gt_used": False,
            "frontend_altered_samples": 0,
        }


class FullPipelineRenderer(IntegratedRenderer):
    """Render detached EXP-048/049/040 outputs with independently toggled layers."""

    def __init__(self, pygame: Any, geometry: Any, show_profile: bool) -> None:
        super().__init__(pygame, geometry, show_profile)
        pygame.display.set_caption(EXPERIMENT_NAME)
        self.tiny_font = pygame.font.Font(None, 17)
        self.show_lidar = True
        self.show_segments = True
        self.show_endpoints = True
        self.show_gaps = True
        self.show_candidates = True
        self.show_identity = True
        self.show_sph = True
        self.show_gt = False

    def toggle(self, key: int) -> bool:
        pygame = self.pygame
        mapping = {
            pygame.K_1: "show_lidar",
            pygame.K_2: "show_segments",
            pygame.K_3: "show_endpoints",
            pygame.K_4: "show_gaps",
            pygame.K_5: "show_candidates",
            pygame.K_6: "show_identity",
            pygame.K_7: "show_sph",
            pygame.K_g: "show_gt",
        }
        name = mapping.get(key)
        if name is None:
            return False
        setattr(self, name, not bool(getattr(self, name)))
        return True

    def _draw_world_without_lidar(self, snapshot: IntegratedSnapshot) -> None:
        pygame = self.pygame
        previous_clip = self.screen.get_clip()
        pygame.draw.rect(self.screen, (21, 27, 35), MAIN_RECT, border_radius=6)
        self.screen.set_clip(MAIN_RECT)
        for rect in self.geometry.free_rects:
            pygame.draw.polygon(
                self.screen,
                COLORS["floor"],
                [self.world_to_screen(point, snapshot) for point in rect.vertices],
            )
        for start, end in self.geometry.walls:
            pygame.draw.line(
                self.screen,
                COLORS["wall"],
                self.world_to_screen(start, snapshot),
                self.world_to_screen(end, snapshot),
                3,
            )
        for position in snapshot.robot_positions:
            pygame.draw.circle(
                self.screen, COLORS["robot"], self.world_to_screen(position, snapshot), 2
            )
        pygame.draw.circle(
            self.screen,
            COLORS["leader"],
            self.world_to_screen(snapshot.leader_position, snapshot),
            7,
        )
        if len(snapshot.trajectory_eval_only) > 1:
            points = [
                self.world_to_screen(np.asarray(point), snapshot)
                for point in snapshot.trajectory_eval_only
            ]
            pygame.draw.lines(self.screen, COLORS["group_center"], False, points, 1)
        self.screen.set_clip(previous_clip)

    @staticmethod
    def _arrow_points(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        direction = end - start
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-12:
            return end.copy(), end.copy()
        axis = direction / norm
        lateral = np.array([-axis[1], axis[0]])
        return end - axis * 6.0 + lateral * 3.0, end - axis * 6.0 - lateral * 3.0

    def _local_to_world(self, session: FullPipelineSession, point: Any) -> np.ndarray:
        if session._analysis_anchor_position is None or session._analysis_yaw_rad is None:
            return np.asarray(point, dtype=float)
        return session._analysis_anchor_position + rotation(session._analysis_yaw_rad) @ np.asarray(point, dtype=float)

    def _world_point(self, session: FullPipelineSession, snapshot: IntegratedSnapshot, point: Any) -> tuple[int, int]:
        return self.world_to_screen(self._local_to_world(session, point), snapshot)

    def _draw_topology(self, session: FullPipelineSession, snapshot: IntegratedSnapshot) -> None:
        if not session.topology_visible:
            return
        runtime = session.runtime_result
        result = runtime.topology_result
        pygame = self.pygame
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(MAIN_RECT)

        if self.show_lidar:
            hit = np.asarray(runtime.source_snapshot["hit"], dtype=bool)
            for point in np.asarray(result["points"])[hit][::2]:
                pygame.draw.circle(
                    self.screen,
                    OVERLAY_COLORS["point"],
                    self._world_point(session, snapshot, point),
                    2,
                )

        if self.show_segments:
            for segment in result["segments"]:
                start = np.asarray(segment["start"])
                end = np.asarray(segment["end"])
                pygame.draw.line(
                    self.screen,
                    OVERLAY_COLORS["segment"],
                    self._world_point(session, snapshot, start),
                    self._world_point(session, snapshot, end),
                    4,
                )
                center = 0.5 * (start + end)
                self.text(
                    f"W{segment['segment_id']}",
                    self._world_point(session, snapshot, center),
                    OVERLAY_COLORS["segment"],
                    self.tiny_font,
                )

        endpoints = {int(row["endpoint_id"]): row for row in result["endpoints"]}
        if self.show_endpoints:
            for endpoint in result["endpoints"]:
                point = self._world_point(session, snapshot, endpoint["point"])
                endpoint_type = str(endpoint["endpoint_type"])
                if not endpoint["valid"] or endpoint_type == "SCAN_LIMIT":
                    color = OVERLAY_COLORS["scan_limit"]
                    pygame.draw.line(self.screen, color, (point[0] - 5, point[1] - 5), (point[0] + 5, point[1] + 5), 2)
                    pygame.draw.line(self.screen, color, (point[0] - 5, point[1] + 5), (point[0] + 5, point[1] - 5), 2)
                elif endpoint_type == "WALL_TERMINATION":
                    color = OVERLAY_COLORS["termination"]
                    pygame.draw.circle(self.screen, color, point, 6)
                elif endpoint_type == "CORNER":
                    color = OVERLAY_COLORS["corner"]
                    pygame.draw.circle(self.screen, color, point, 6, 2)
                else:
                    color = OVERLAY_COLORS["unstable"]
                    polygon = [(point[0], point[1] - 6), (point[0] + 6, point[1]), (point[0], point[1] + 6), (point[0] - 6, point[1])]
                    pygame.draw.polygon(self.screen, color, polygon, 2)
                self.text(f"E{endpoint['endpoint_id']}", (point[0] + 5, point[1] - 15), color, self.tiny_font)

        candidate_gap_ids = {
            int(row["source_gap_id"]) for row in runtime.candidate_rows
        }
        if self.show_gaps:
            for gap in result["gaps"]:
                first = endpoints[int(gap["endpoint_a"])]["point"]
                second = endpoints[int(gap["endpoint_b"])]["point"]
                accepted = int(gap["gap_id"]) in candidate_gap_ids
                color = OVERLAY_COLORS["gap_accepted" if accepted else "gap_rejected"]
                pygame.draw.line(
                    self.screen,
                    color,
                    self._world_point(session, snapshot, first),
                    self._world_point(session, snapshot, second),
                    5 if accepted else 3,
                )
                center = np.asarray(gap["gap_center"])
                self.text(f"G{gap['gap_id']}", self._world_point(session, snapshot, center), color, self.tiny_font)

        if self.show_candidates:
            decision_by_source = {
                row.source_candidate_id: row for row in session.identity_decisions
            }
            for row in runtime.candidate_rows:
                source_id = str(row["candidate_id"])
                display = decision_by_source[source_id].display_id
                center = np.array([row["center_x_local"], row["center_y_local"]])
                normal = np.array([row["opening_normal_x_local"], row["opening_normal_y_local"]])
                tangent = np.array([row["opening_tangent_x_local"], row["opening_tangent_y_local"]])
                half_width = 0.5 * float(row["opening_width"])
                tangent_start = center - tangent * half_width
                tangent_end = center + tangent * half_width
                normal_end = center + normal * 25.0
                pygame.draw.line(self.screen, OVERLAY_COLORS["candidate"], self._world_point(session, snapshot, tangent_start), self._world_point(session, snapshot, tangent_end), 3)
                pygame.draw.line(self.screen, OVERLAY_COLORS["candidate"], self._world_point(session, snapshot, center), self._world_point(session, snapshot, normal_end), 4)
                arrow_a, arrow_b = self._arrow_points(center, normal_end)
                pygame.draw.polygon(self.screen, OVERLAY_COLORS["candidate"], [self._world_point(session, snapshot, normal_end), self._world_point(session, snapshot, arrow_a), self._world_point(session, snapshot, arrow_b)])
                self.text(display, self._world_point(session, snapshot, center + normal * 7.0), OVERLAY_COLORS["candidate"], self.small_font)

        if self.show_identity and len(session._history_local) > 1:
            history_world = [self._world_point(session, snapshot, point) for point in session._history_local[::3]]
            if len(history_world) > 1:
                pygame.draw.lines(self.screen, OVERLAY_COLORS["history"], False, history_world, 3)
            parent_end = session._parent_axis_local * 30.0
            pygame.draw.line(self.screen, OVERLAY_COLORS["history"], self._world_point(session, snapshot, np.zeros(2)), self._world_point(session, snapshot, parent_end), 4)
            self.text("PARENT MOTION", self._world_point(session, snapshot, parent_end), OVERLAY_COLORS["history"], self.tiny_font)

        if self.show_gt:
            for mouth in session.eval_mouths:
                first, second = np.asarray(mouth["a"]), np.asarray(mouth["b"])
                pygame.draw.line(self.screen, OVERLAY_COLORS["gt"], self._world_point(session, snapshot, first), self._world_point(session, snapshot, second), 3)
                center = 0.5 * (first + second)
                self.text(f"EVAL {mouth['label']}", self._world_point(session, snapshot, center), OVERLAY_COLORS["gt"], self.tiny_font)
            self.text("GT EVAL ONLY", (MAIN_RECT[0] + MAIN_RECT[2] - 112, MAIN_RECT[1] + 10), OVERLAY_COLORS["gt"], self.small_font)
        self.screen.set_clip(previous_clip)

    def _gap_rejection_lines(self, session: FullPipelineSession) -> list[str]:
        return [
            f"G{gap_id} -> no candidate: {status}"
            for gap_id, status in session.gap_general_status.items()
            if status != "CANDIDATE_CREATED"
        ]

    def draw(self, session: FullPipelineSession, paused: bool) -> None:
        snapshot = session.current
        self.screen.fill(COLORS["background"])
        self.text("Full Junction Pipeline — read-only frozen outputs", (18, 15), font=self.title_font)
        if snapshot is None:
            self.text("Waiting for first sampled LiDAR scan...", (44, 90))
            self.pygame.display.flip()
            return
        if self.show_lidar:
            super()._draw_world(snapshot)
        else:
            self._draw_world_without_lidar(snapshot)
        self._draw_topology(session, snapshot)
        if self.show_profile:
            self._draw_profile(snapshot)

        runtime = session.runtime_result if session.topology_visible else None
        result = None if runtime is None else runtime.topology_result
        panel_x = 870
        y = 530 if self.show_profile else 70
        step = 16
        front_lines = [
            f"{'PAUSED' if paused else 'RUNNING'} | {session.case_id} | f{snapshot.physics_frame}",
            f"pipeline: {session.pipeline_state()} | event: {snapshot.latest_event}",
            f"t={snapshot.timestamp:.6f}s speed={snapshot.speed:.4f}",
            f"corridor={snapshot.corridor_state} W_hat={_finite(snapshot.estimated_corridor_width)}",
            f"expected={snapshot.expected_profile_source}",
            f"OPEN={snapshot.opening_candidate_count} groups={len(snapshot.opening_groups)} JUNCTION={snapshot.junction_detected_latched}",
            f"bilateral={snapshot.bilateral_streak} brake_ready={snapshot.brake_ready} braking={snapshot.braking_active}",
            f"anchor={_empty(snapshot.anchor_enter_frame)} hold={snapshot.anchor_hold_duration:.3f}s",
        ]
        topology_lines = ["POINT CLOUD: WAITING FOR ANCHOR HOLD"] if result is None else [
            f"POINT CLOUD: analysis f{runtime.analysis_frame} single scan",
            f"segments={len(result['segments'])} valid endpoints={sum(bool(row['valid']) for row in result['endpoints'])}",
            f"physical gaps={len(result['gaps'])} general candidates={len(runtime.candidate_rows)}",
        ]
        candidate_lines = []
        if runtime is not None:
            decision_by_source = {row.source_candidate_id: row for row in session.identity_decisions}
            for row in runtime.candidate_rows:
                decision = decision_by_source[str(row["candidate_id"])]
                candidate_lines.append(
                    f"{decision.display_id} {row['topology_state']} angle={row['opening_normal_deg_local']:+.1f} {decision.decision}"
                )
            candidate_lines.extend(self._gap_rejection_lines(session))
        status_lines = [
            f"IDENTITY: {session.identity_status}",
            f"SPH validation: {session.sph_status if self.show_sph else 'OVERLAY OFF'}",
            "LEFT miss preserved: no synthetic candidate" if runtime is not None else "",
        ]
        all_lines = front_lines + topology_lines + candidate_lines + status_lines
        for index, value in enumerate(value for value in all_lines if value):
            color = COLORS["text"]
            if value.startswith(("pipeline", "POINT CLOUD", "IDENTITY")):
                color = COLORS["group_center"]
            if "no candidate" in value or "LEFT miss" in value:
                color = OVERLAY_COLORS["gap_rejected"]
            self.text(value, (panel_x, y + index * step), color, self.tiny_font)

        legend_y = min(842, y + len([row for row in all_lines if row]) * step + 5)
        legend = "Endpoints: CORNER o | WALL_TERMINATION solid | SCAN_LIMIT x | UNSTABLE_ENDPOINT diamond"
        self.text(legend, (panel_x, legend_y), COLORS["muted"], self.tiny_font)

        visible_events = [
            row for row in session.display_events
            if int(row["frame"]) <= snapshot.physics_frame
        ]
        overlay = self.pygame.Surface((410, 194), self.pygame.SRCALPHA)
        overlay.fill((9, 12, 18, 205))
        self.screen.blit(overlay, (22, 58))
        self.text("EVENT LOG", (30, 65), COLORS["group_center"], self.small_font)
        for index, row in enumerate(visible_events[-9:]):
            event_text = str(row["event"])
            if len(event_text) > 52:
                event_text = event_text[:49] + "..."
            self.text(f"f{int(row['frame']):>3} {event_text}", (30, 85 + index * 18), COLORS["text"], self.tiny_font)

        self.text("1 rays  2 walls  3 endpoints  4 gaps  5 candidates  6 identity/history  7 SPH  G EVAL", (20, 858), COLORS["muted"], self.tiny_font)
        self.text("SPACE pause  R restart  LEFT/RIGHT sampled frame  P profile  ESC quit", (20, 878), COLORS["muted"], self.tiny_font)
        self.pygame.display.flip()


def run_case(case_id: str, frames: int) -> FullPipelineSession:
    return FullPipelineSession(case_id).run(frames)


def _assert_equivalence(session: FullPipelineSession) -> None:
    summary = session.summary()
    if session.case_id == BOOTSTRAP_ALIAS and session.integration.frontend.next_physics_frame > 222:
        expected = {
            "ready_frame": 6,
            "first_open_frame": 30,
            "detection_frame": 36,
            "bilateral_frame": 174,
            "brake_ready_frame": 180,
            "braking_frame": 181,
            "stop_frame": 216,
            "anchor_frame": 221,
            "analysis_frame": 222,
            "wall_segments": 6,
            "valid_endpoints": 3,
            "physical_gaps": 3,
            "general_candidates": 2,
            "true_outgoing_recovered": 2,
            "false_candidates": 0,
        }
        checks = {key: summary[key] == value for key, value in expected.items()}
        checks.update({
            "no_pre_hold_invocation": summary["pointcloud_before_hold"] == 0,
            "single_invocation": summary["pointcloud_invocations"] == 1,
            "identity_available": summary["identity_status"] == "AVAILABLE",
            "sph_not_connected": summary["sph_status"] == "NOT CONNECTED",
            "runtime_no_gt": not summary["runtime_gt_used"],
            "frontend_unchanged": summary["frontend_altered_samples"] == 0,
            "left_rejection_preserved": session.gap_general_status.get(1) == "NO_INCIDENT_WALL_SUPPORT_ALONG_GAP_TANGENT",
        })
        if not all(checks.values()):
            raise AssertionError(json.dumps(checks, sort_keys=True))
    if session.case_id == M0_ALIAS:
        checks = {
            "no_detection": summary["detection_frame"] == "",
            "no_anchor": summary["anchor_frame"] == "",
            "no_pointcloud": summary["pointcloud_invocations"] == 0,
            "no_candidates": summary["general_candidates"] == 0,
        }
        if not all(checks.values()):
            raise AssertionError(json.dumps(checks, sort_keys=True))


def _display_signature(session: FullPipelineSession) -> tuple[Any, ...]:
    return (
        _run_signature(session.integration),
        tuple(
            (row.display_id, row.source_candidate_id, row.decision, row.matched_id, row.reason)
            for row in session.identity_decisions
        ),
        tuple((int(row["frame"]), str(row["event"])) for row in session.display_events),
    )


def run_gui(args: argparse.Namespace) -> FullPipelineSession:
    import pygame

    pygame.init()
    session = FullPipelineSession(args.map_case)
    renderer = FullPipelineRenderer(pygame, session.frontend.runner.geometry, args.show_profile)
    clock = pygame.time.Clock()
    paused = bool(args.start_paused)
    running = True
    session.step()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    if not paused:
                        session.frontend.view_index = len(session.frontend.snapshots) - 1
                elif event.key == pygame.K_r:
                    session.restart()
                    renderer.geometry = session.frontend.runner.geometry
                    renderer._configure_camera()
                    paused = bool(args.start_paused)
                    session.step()
                elif event.key == pygame.K_LEFT and paused:
                    session.step_sample(-1)
                elif event.key == pygame.K_RIGHT and paused:
                    session.step_sample(1)
                elif event.key == pygame.K_p:
                    renderer.show_profile = not renderer.show_profile
                else:
                    renderer.toggle(event.key)
        if not paused:
            session.step()
        renderer.draw(session, paused)
        if args.frames > 0 and session.frontend.next_physics_frame >= args.frames:
            running = False
        clock.tick(args.fps)
    pygame.quit()
    return session


def render_keyframes(session: FullPipelineSession, output: Path) -> None:
    import pygame

    pygame.init()
    renderer = FullPipelineRenderer(pygame, session.frontend.runner.geometry, True)
    frames = {
        "provisional_anchor": session.frontend.anchor_enter_frame,
        "stationary_analysis": None if session.runtime_result is None else session.runtime_result.analysis_frame,
        "identity_decision": None if not session.identity_decisions else session.runtime_result.analysis_frame,
        "post_anchor_sensing": session.current.physics_frame if session.current is not None else None,
    }
    output.mkdir(parents=True, exist_ok=True)
    for label, frame in frames.items():
        if frame is None:
            continue
        indices = [index for index, item in enumerate(session.frontend.snapshots) if item.physics_frame >= frame]
        if not indices:
            continue
        session.frontend.view_index = indices[0]
        renderer.draw(session, True)
        pygame.image.save(renderer.screen, output / f"{label}_frame_{frame}.png")
    pygame.quit()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-case", choices=MAP_CASES, default=BOOTSTRAP_ALIAS)
    parser.add_argument("--frames", type=int, default=0, help="0: GUI until ESC; headless uses case default")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--show-profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--deterministic-replay", action="store_true")
    parser.add_argument("--render-keyframes", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error("--frames must be non-negative")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.headless and args.frames == 0:
        args.frames = 600 if args.map_case == M0_ALIAS else 240
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    hashes_before = protected_hashes()
    if args.headless:
        session = run_case(args.map_case, args.frames)
    else:
        session = run_gui(args)
    _assert_equivalence(session)
    replay_match: bool | None = None
    if args.deterministic_replay:
        replay = run_case(args.map_case, args.frames)
        _assert_equivalence(replay)
        replay_match = _display_signature(session) == _display_signature(replay)
        if not replay_match:
            raise AssertionError("full-pipeline deterministic replay mismatch")
    if args.render_keyframes:
        render_keyframes(session, args.output_dir / "keyframes")
    hashes_after = protected_hashes()
    if hashes_before != hashes_after:
        raise AssertionError("protected source hash changed during visualization")
    print(json.dumps({**session.summary(), "deterministic_replay": replay_match, "protected_hashes_unchanged": True}, sort_keys=True))


if __name__ == "__main__":
    main()
