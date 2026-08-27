"""EXP-048 integrated early-detection controlled-approach GUI visualizer.

This file adds visualization and integration only.  It consumes the frozen
LiDAR profile detector, EXP-042 rear-start setup, EXP-047 Candidate-B helpers,
and the simulator's existing braking/provisional-anchor entry point.  Detector
profiles, OPEN masks, group geometry, braking, dwell, and anchor semantics are
never recomputed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.integration.run_lidar_local_corridor_estimation import (  # noqa: E402
    REAR_START_SHIFT,
    _rear_start,
)
from junction_detection.integration.run_local_controlled_approach_brake_trigger_shadow import (  # noqa: E402
    BOOTSTRAP_CASE as EXP047_BOOTSTRAP_CASE,
    CANDIDATE_B,
    EXISTING_OBSERVATION_WINDOW,
    LEFT_LATERAL_SECTOR,
    RIGHT_LATERAL_SECTOR,
    _bilateral_lateral_groups,
    _candidate_predicate,
    _expected_source,
    _feature_row,
    _new_runner,
    _visibility_for_runner,
    protected_hashes,
)
from pygame_simulator.lidar_junction_detection_visualizer import (  # noqa: E402
    COLORS,
    MAIN_RECT,
    PROFILE_RECT,
    DisplaySnapshot,
    Renderer as ProfileRenderer,
    _copy_snapshot,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import (  # noqa: E402
    DT,
    LIDAR_MAX_RANGE,
    MIN_SPEED,
    SAMPLE_PERIOD,
)


EXPERIMENT_ID = "EXP-048"
BOOTSTRAP_ALIAS = "M1_PRE_CORRIDOR_BOOTSTRAP"
BASELINE_ALIAS = "M1_CROSS_BASELINE"
M0_ALIAS = "M0_STRAIGHT"
MAP_CASES = (BOOTSTRAP_ALIAS, BASELINE_ALIAS, M0_ALIAS)
DEFAULT_OUTPUT = (
    ROOT
    / "junction_detection/integration/output/"
    / "integrated_early_detection_controlled_approach_gui"
)

TIMELINE_FIELDS = (
    "case_id", "frame", "time", "pipeline_state", "corridor_state",
    "side_walls_valid", "current_width", "stable_width", "offset",
    "orientation", "stable_model_initialized", "stable_update_count",
    "expected_profile_source", "open_candidate_count", "opening_group_count",
    "left_group_center", "right_group_center", "left_in_lateral_sector",
    "right_in_lateral_sector", "bilateral_streak", "junction_detected",
    "junction_detected_latched", "brake_ready", "braking_active", "speed",
    "stationary_dwell_steps", "provisional_anchor", "anchor_hold_duration",
    "post_anchor_lidar_sample_count", "x_eval_only", "y_eval_only",
    "inside_junction_eval_only",
)
EVENT_FIELDS = (
    "case_id", "frame", "time", "event", "pipeline_state", "corridor_state",
    "speed", "junction_detected", "opening_group_count", "bilateral_streak",
    "x_eval_only", "y_eval_only", "runtime_gt_used",
)


@dataclass(frozen=True)
class IntegratedSnapshot(DisplaySnapshot):
    """Detached visual copy of detector/controller outputs."""

    pipeline_state: str
    corridor_state: str
    latest_event: str
    speed: float
    side_walls_valid: bool
    current_width: float
    current_offset: float
    current_orientation: float
    stable_model_initialized: bool
    stable_update_count: int
    expected_profile_source: str
    left_group_center: float
    right_group_center: float
    left_in_lateral_sector: bool
    right_in_lateral_sector: bool
    bilateral_streak: int
    junction_detected_latched: bool
    brake_ready: bool
    braking_active: bool
    brake_trigger_frame: int | None
    brake_trigger_time: float | None
    brake_distance_eval_only: float
    anchor_enter_frame: int | None
    anchor_enter_time: float | None
    anchor_hold_duration: float
    post_anchor_lidar_samples: int
    outgoing_visible_eval_only: int | None
    outgoing_total_eval_only: int | None
    side_visible_eval_only: int | None
    trajectory_eval_only: tuple[tuple[float, float], ...]


def _actual_map_case(case_id: str) -> str:
    return M0_ALIAS if case_id == M0_ALIAS else BASELINE_ALIAS


def _is_bootstrap(case_id: str) -> bool:
    return case_id == BOOTSTRAP_ALIAS


def _finite(value: float, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "UNINITIALIZED"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


class IntegratedSession:
    """Run the verified EXP-042 → EXP-047 → existing-anchor pipeline."""

    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self.runner = _new_runner(
            _actual_map_case(case_id), rear_start=_is_bootstrap(case_id)
        )
        self.next_physics_frame = 0
        self.snapshots: list[IntegratedSnapshot] = []
        self.view_index = -1
        self.timeline: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.latest_event = "PRE_CORRIDOR_START" if _is_bootstrap(case_id) else "START"
        self.detected_latched = False
        self.invalid_streak = 0
        self.bilateral_streak = 0
        self.first_ready_frame: int | None = None
        self.first_open_frame: int | None = None
        self.first_detection_frame: int | None = None
        self.bilateral_entry_frame: int | None = None
        self.brake_trigger_frame: int | None = None
        self.brake_trigger_time: float | None = None
        self.brake_trigger_position_eval: np.ndarray | None = None
        self.braking_start_frame: int | None = None
        self.stop_frame: int | None = None
        self.anchor_enter_frame: int | None = None
        self.anchor_enter_time: float | None = None
        self.anchor_position_eval: np.ndarray | None = None
        self.anchor_hold_confirmed_frame: int | None = None
        self.post_anchor_lidar_samples = 0
        self.visibility_eval: dict[str, Any] | None = None
        self.trajectory_eval: list[tuple[float, float]] = []
        self._candidate_b = _candidate_predicate(CANDIDATE_B)
        self._start_recorded = False

    def restart(self) -> None:
        self.__init__(self.case_id)

    def _leader(self) -> Any:
        return next(
            robot
            for robot in self.runner.world.robots
            if robot.robot_id == self.runner.world.lidar_robot_id
        )

    def corridor_state(self, profile: dict[str, Any] | None = None) -> str:
        profile = self.runner.last_profile_result if profile is None else profile
        if profile is None:
            return "UNINITIALIZED"
        if profile["corridor_model_initialized"]:
            return "READY"
        if profile["side_walls_valid"]:
            return "BOOTSTRAPPING"
        return "UNINITIALIZED"

    def pipeline_state(self) -> str:
        world = self.runner.world
        if world.provisional_fixed_anchor:
            return "PROVISIONAL_ANCHOR"
        if world.suspect_hold_active or world.braking_active:
            return "BRAKING"
        if self.detected_latched:
            return "CONTROLLED_APPROACH"
        return "MOVING_LIDAR"

    def _eval_pose(self) -> dict[str, Any]:
        leader = self._leader()
        x, y = float(leader.position[0]), float(leader.position[1])
        geometry = self.runner.geometry
        inside = False
        if geometry.entrance_y is not None:
            half = 0.5 * float(geometry.junction_size)
            inside = bool(abs(x) <= half and abs(y) <= half)
        return {"x_eval_only": x, "y_eval_only": y, "inside_junction_eval_only": inside}

    def _record_event(
        self,
        event: str,
        frame: int,
        timestamp: float,
        profile: dict[str, Any] | None,
    ) -> None:
        self.latest_event = event
        leader = self._leader()
        self.events.append(
            {
                "case_id": self.case_id,
                "frame": frame,
                "time": timestamp,
                "event": event,
                "pipeline_state": self.pipeline_state(),
                "corridor_state": self.corridor_state(profile),
                "speed": float(np.linalg.norm(leader.velocity)),
                "junction_detected": bool(
                    profile and profile["profile_junction_detected"]
                ),
                "opening_group_count": int(
                    profile["opening_group_count"] if profile else 0
                ),
                "bilateral_streak": self.bilateral_streak,
                **self._eval_pose(),
                "runtime_gt_used": False,
            }
        )

    @staticmethod
    def _display_centers(
        groups: Iterable[dict[str, Any]],
    ) -> tuple[float, float, bool, bool]:
        centers = [float(group["center_angle_deg"]) for group in groups]
        negative = [value for value in centers if value < 0.0]
        positive = [value for value in centers if value > 0.0]
        left = min(negative) if negative else math.nan
        right = max(positive) if positive else math.nan
        left_in = bool(
            math.isfinite(left)
            and RIGHT_LATERAL_SECTOR[0] <= left <= RIGHT_LATERAL_SECTOR[1]
        )
        right_in = bool(
            math.isfinite(right)
            and LEFT_LATERAL_SECTOR[0] <= right <= LEFT_LATERAL_SECTOR[1]
        )
        return left, right, left_in, right_in

    def _record_physics_edges(self, frame: int) -> None:
        world = self.runner.world
        profile = self.runner.last_profile_result
        timestamp = float(world.time)
        if not self._start_recorded:
            self._record_event(self.latest_event, frame, timestamp, profile)
            self._start_recorded = True
        if world.braking_active and self.braking_start_frame is None:
            self.braking_start_frame = frame
            self._record_event("BRAKING_STARTED", frame, timestamp, profile)
        if (
            self.brake_trigger_frame is not None
            and self.stop_frame is None
            and float(np.linalg.norm(self._leader().velocity)) < MIN_SPEED
        ):
            self.stop_frame = frame
            self._record_event("STOP_SPEED_REACHED", frame, timestamp, profile)
        if world.provisional_fixed_anchor and self.anchor_enter_frame is None:
            self.anchor_enter_frame = int(world.anchor_entry_frame)
            self.anchor_enter_time = float(world.anchor_entry_time)
            self.anchor_position_eval = self._leader().position.copy()
            self._record_event(
                "PROVISIONAL_ANCHOR", self.anchor_enter_frame,
                self.anchor_enter_time, profile,
            )

    def advance_physics_frame(self) -> IntegratedSnapshot | None:
        frame = self.next_physics_frame
        row = self.runner.step(frame)
        self.next_physics_frame += 1
        leader = self._leader()
        self.trajectory_eval.append(
            (float(leader.position[0]), float(leader.position[1]))
        )
        self._record_physics_edges(frame)
        if row is None:
            return None

        profile = self.runner.last_profile_result
        groups = list(profile["opening_groups"])
        if (
            profile["corridor_model_initialized"]
            and self.first_ready_frame is None
        ):
            self.first_ready_frame = frame
            self._record_event("MODEL_READY", frame, float(row["timestamp"]), profile)
        if profile["opening_candidate_count"] > 0 and self.first_open_frame is None:
            self.first_open_frame = frame
            self._record_event("FIRST_OPEN", frame, float(row["timestamp"]), profile)
        if profile["profile_junction_detected"] and not self.detected_latched:
            self.detected_latched = True
            self.first_detection_frame = frame
            self._record_event(
                "JUNCTION_DETECTED_EARLY", frame, float(row["timestamp"]), profile
            )

        if self.detected_latched:
            self.invalid_streak = (
                self.invalid_streak + 1 if not profile["side_walls_valid"] else 0
            )
            bilateral_now = _bilateral_lateral_groups(groups)
            self.bilateral_streak = self.bilateral_streak + 1 if bilateral_now else 0
            if bilateral_now and self.bilateral_entry_frame is None:
                self.bilateral_entry_frame = frame
                self._record_event(
                    "BILATERAL_LATERAL_ENTRY", frame,
                    float(row["timestamp"]), profile,
                )

        feature = _feature_row(
            EXP047_BOOTSTRAP_CASE,
            self.runner,
            row,
            self.detected_latched,
            self.invalid_streak,
            self.bilateral_streak,
        )
        brake_ready = bool(self._candidate_b(feature))
        if brake_ready and self.brake_trigger_frame is None:
            self.brake_trigger_frame = frame
            self.brake_trigger_time = float(row["timestamp"])
            self.brake_trigger_position_eval = leader.position.copy()
            self._record_event("BRAKE_READY", frame, float(row["timestamp"]), profile)
            self.runner.world.activate_profile_junction_hold(
                float(row["timestamp"]),
                float(profile["stable_corridor_orientation_deg"]),
                float(profile["estimated_corridor_width"]),
            )
            self._record_event(
                "BRAKE_TRIGGERED", frame, float(row["timestamp"]), profile
            )

        if self.anchor_enter_frame is not None and frame > self.anchor_enter_frame:
            self.post_anchor_lidar_samples += 1
            if self.anchor_hold_confirmed_frame is None:
                self.anchor_hold_confirmed_frame = frame
                self._record_event(
                    "ANCHOR_HOLD_CONFIRMED", frame,
                    float(row["timestamp"]), profile,
                )
            if self.visibility_eval is None:
                # Post-anchor evaluation only; never feeds any transition.
                self.visibility_eval = _visibility_for_runner(
                    self.runner, "EXP048_GUI_POSTHOC_EVAL_ONLY"
                )

        left_center, right_center, left_in, right_in = self._display_centers(groups)
        base = _copy_snapshot(self.runner, frame, row)
        brake_distance = (
            float(np.linalg.norm(leader.position - self.brake_trigger_position_eval))
            if self.brake_trigger_position_eval is not None
            else 0.0
        )
        hold_duration = (
            max(0.0, float(row["timestamp"]) - self.anchor_enter_time)
            if self.anchor_enter_time is not None
            else 0.0
        )
        snapshot = IntegratedSnapshot(
            **base.__dict__,
            pipeline_state=self.pipeline_state(),
            corridor_state=self.corridor_state(profile),
            latest_event=self.latest_event,
            speed=float(np.linalg.norm(leader.velocity)),
            side_walls_valid=bool(profile["side_walls_valid"]),
            current_width=float(profile["width_observation"]),
            current_offset=float(profile["offset_observation"]),
            current_orientation=float(profile["current_corridor_orientation_deg"]),
            stable_model_initialized=bool(profile["corridor_model_initialized"]),
            stable_update_count=int(profile["corridor_model_update_count"]),
            expected_profile_source=_expected_source(row),
            left_group_center=left_center,
            right_group_center=right_center,
            left_in_lateral_sector=left_in,
            right_in_lateral_sector=right_in,
            bilateral_streak=self.bilateral_streak,
            junction_detected_latched=self.detected_latched,
            brake_ready=brake_ready,
            braking_active=bool(self.runner.world.braking_active),
            brake_trigger_frame=self.brake_trigger_frame,
            brake_trigger_time=self.brake_trigger_time,
            brake_distance_eval_only=brake_distance,
            anchor_enter_frame=self.anchor_enter_frame,
            anchor_enter_time=self.anchor_enter_time,
            anchor_hold_duration=hold_duration,
            post_anchor_lidar_samples=self.post_anchor_lidar_samples,
            outgoing_visible_eval_only=(
                None
                if self.visibility_eval is None
                else int(self.visibility_eval["outgoing_visible_count_eval_only"])
            ),
            outgoing_total_eval_only=(
                None
                if self.visibility_eval is None
                else int(self.visibility_eval["outgoing_gt_count_eval_only"])
            ),
            side_visible_eval_only=(
                None
                if self.visibility_eval is None
                else int(self.visibility_eval["side_visible_count_eval_only"])
            ),
            trajectory_eval_only=tuple(self.trajectory_eval[::3]),
        )
        self.snapshots.append(snapshot)
        self.view_index = len(self.snapshots) - 1
        self.timeline.append(self._timeline_row(snapshot))
        return snapshot

    def _timeline_row(self, snapshot: IntegratedSnapshot) -> dict[str, Any]:
        pose = self._eval_pose()
        return {
            "case_id": self.case_id,
            "frame": snapshot.physics_frame,
            "time": snapshot.timestamp,
            "pipeline_state": snapshot.pipeline_state,
            "corridor_state": snapshot.corridor_state,
            "side_walls_valid": snapshot.side_walls_valid,
            "current_width": snapshot.current_width,
            "stable_width": snapshot.estimated_corridor_width,
            "offset": snapshot.estimated_offset,
            "orientation": snapshot.stable_orientation_deg,
            "stable_model_initialized": snapshot.stable_model_initialized,
            "stable_update_count": snapshot.stable_update_count,
            "expected_profile_source": snapshot.expected_profile_source,
            "open_candidate_count": snapshot.opening_candidate_count,
            "opening_group_count": len(snapshot.opening_groups),
            "left_group_center": snapshot.left_group_center,
            "right_group_center": snapshot.right_group_center,
            "left_in_lateral_sector": snapshot.left_in_lateral_sector,
            "right_in_lateral_sector": snapshot.right_in_lateral_sector,
            "bilateral_streak": snapshot.bilateral_streak,
            "junction_detected": snapshot.junction_detected,
            "junction_detected_latched": snapshot.junction_detected_latched,
            "brake_ready": snapshot.brake_ready,
            "braking_active": snapshot.braking_active,
            "speed": snapshot.speed,
            "stationary_dwell_steps": int(self.runner.world.stationary_dwell_steps),
            "provisional_anchor": bool(self.runner.world.provisional_fixed_anchor),
            "anchor_hold_duration": snapshot.anchor_hold_duration,
            "post_anchor_lidar_sample_count": snapshot.post_anchor_lidar_samples,
            **pose,
        }

    @property
    def current(self) -> IntegratedSnapshot | None:
        return None if self.view_index < 0 else self.snapshots[self.view_index]

    def step_sample(self, direction: int) -> None:
        if direction < 0:
            self.view_index = max(0, self.view_index - 1)
            return
        if self.view_index + 1 < len(self.snapshots):
            self.view_index += 1
            return
        count = len(self.snapshots)
        for _ in range(max(1, round(SAMPLE_PERIOD / DT)) + 1):
            self.advance_physics_frame()
            if len(self.snapshots) > count:
                break

    def summary(self, deterministic_replay: bool | None = None) -> dict[str, Any]:
        anchor_x = (
            math.nan if self.anchor_position_eval is None
            else float(self.anchor_position_eval[0])
        )
        anchor_y = (
            math.nan if self.anchor_position_eval is None
            else float(self.anchor_position_eval[1])
        )
        geometry = self.runner.geometry
        inside = False
        if self.anchor_position_eval is not None and geometry.entrance_y is not None:
            half = 0.5 * float(geometry.junction_size)
            inside = bool(abs(anchor_x) <= half and abs(anchor_y) <= half)
        return {
            "experiment_id": EXPERIMENT_ID,
            "case_id": self.case_id,
            "ready_frame": _empty(self.first_ready_frame),
            "first_open_frame": _empty(self.first_open_frame),
            "first_detection_frame": _empty(self.first_detection_frame),
            "bilateral_entry_frame": _empty(self.bilateral_entry_frame),
            "candidate_b_trigger_frame": _empty(self.brake_trigger_frame),
            "braking_start_frame": _empty(self.braking_start_frame),
            "stop_frame": _empty(self.stop_frame),
            "anchor_frame": _empty(self.anchor_enter_frame),
            "anchor_x_eval_only": anchor_x,
            "anchor_y_eval_only": anchor_y,
            "inside_junction_eval_only": inside,
            "outgoing_visible_eval_only": (
                "" if self.visibility_eval is None
                else self.visibility_eval["outgoing_visible_count_eval_only"]
            ),
            "outgoing_total_eval_only": (
                "" if self.visibility_eval is None
                else self.visibility_eval["outgoing_gt_count_eval_only"]
            ),
            "side_visible_eval_only": (
                "" if self.visibility_eval is None
                else self.visibility_eval["side_visible_count_eval_only"]
            ),
            "final_pipeline_state": self.pipeline_state(),
            "post_anchor_lidar_sample_count": self.post_anchor_lidar_samples,
            "exp042_equivalent": bool(
                self.case_id == BOOTSTRAP_ALIAS
                and self.first_ready_frame == 6
                and self.first_open_frame == 30
                and self.first_detection_frame == 36
            ),
            "exp047_equivalent": bool(
                self.case_id == BOOTSTRAP_ALIAS
                and self.brake_trigger_frame == 180
                and self.anchor_enter_frame == 221
                and math.isfinite(anchor_y)
                and abs(anchor_y + 40.329315371814026) <= 1.0e-6
            ),
            "deterministic_replay": _empty(deterministic_replay),
            "runtime_gt_or_map_used_for_transition": False,
            "candidate_b_reimplemented": False,
            "detector_output_altered": False,
            "brake_law_altered": False,
            "pointcloud_runtime_trigger_used": False,
        }


class IntegratedRenderer(ProfileRenderer):
    """Reuse frozen-output renderer and add EXP-048 state overlays."""

    def __init__(self, pygame: Any, geometry: Any, show_profile: bool) -> None:
        super().__init__(pygame, geometry, show_profile)
        pygame.display.set_caption(
            "EXP-048 — Early Detection → Controlled Approach → Provisional Anchor"
        )

    def _draw_world(self, snapshot: IntegratedSnapshot) -> None:
        super()._draw_world(snapshot)
        pygame = self.pygame
        clip = self.screen.get_clip()
        self.screen.set_clip(MAIN_RECT)
        if len(snapshot.trajectory_eval_only) > 1:
            points = [
                self.world_to_screen(np.asarray(point), snapshot)
                for point in snapshot.trajectory_eval_only
            ]
            pygame.draw.lines(self.screen, COLORS["group_center"], False, points, 1)
        if snapshot.pipeline_state == "PROVISIONAL_ANCHOR":
            pygame.draw.circle(
                self.screen, COLORS["detected"],
                self.world_to_screen(snapshot.leader_position, snapshot), 12, 3,
            )
        self.screen.set_clip(clip)

    def _draw_profile(self, snapshot: IntegratedSnapshot) -> None:
        super()._draw_profile(snapshot)
        pygame = self.pygame
        x, y, width, height = PROFILE_RECT
        sector_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        for low, high in (RIGHT_LATERAL_SECTOR, LEFT_LATERAL_SECTOR):
            left = self._plot_point(PROFILE_RECT, low, 0.0)[0] - x
            right = self._plot_point(PROFILE_RECT, high, 0.0)[0] - x
            pygame.draw.rect(
                sector_surface, (44, 189, 176, 24),
                (left, 0, max(1, right - left), height),
            )
        self.screen.blit(sector_surface, (x, y))
        for index in np.flatnonzero(snapshot.candidate_mask):
            px = self._plot_point(
                PROFILE_RECT, float(snapshot.angles_deg[index]), 0.0
            )[0]
            pygame.draw.line(
                self.screen, COLORS["candidate_beam"],
                (px, y + height - 9), (px, y + height), 2,
            )
        for group in snapshot.opening_groups:
            px = self._plot_point(
                PROFILE_RECT, float(group["center_angle_deg"]), 0.0
            )[0]
            pygame.draw.line(
                self.screen, COLORS["group_center"], (px, y), (px, y + height), 2
            )
            pygame.draw.circle(
                self.screen, COLORS["group_center"],
                self._plot_point(
                    PROFILE_RECT,
                    float(group["center_angle_deg"]),
                    float(group["mean_range"]),
                ),
                5,
            )
        legend_rows = (
            (
                ("MEASURED", COLORS["measured"]),
                ("EXPECTED", COLORS["expected"]),
                ("EXPECTED + MARGIN", COLORS["threshold"]),
            ),
            (
                ("OPEN CANDIDATE", COLORS["candidate_beam"]),
                ("CONFIRMED OPENING", COLORS["open_beam"]),
            ),
        )
        for row_index, legend in enumerate(legend_rows):
            legend_y = y + 5 + row_index * 18
            cursor = x + 7
            for label, color in legend:
                pygame.draw.line(
                    self.screen, color,
                    (cursor, legend_y + 8), (cursor + 14, legend_y + 8), 3,
                )
                self.text(label, (cursor + 18, legend_y), color, self.small_font)
                cursor += 28 + self.small_font.size(label)[0]

    def draw(self, session: IntegratedSession, paused: bool) -> None:
        snapshot = session.current
        self.screen.fill(COLORS["background"])
        self.text(
            "EXP-048 - Moving LiDAR -> Controlled Approach -> Provisional Anchor",
            (18, 15), font=self.title_font,
        )
        if snapshot is None:
            self.text("Waiting for first sampled LiDAR scan...", (44, 90))
            self.pygame.display.flip()
            return
        self._draw_world(snapshot)
        if self.show_profile:
            self._draw_profile(snapshot)

        panel_x = 870
        y = 505 if self.show_profile else 70
        step = 18
        streak = min(snapshot.bilateral_streak, EXISTING_OBSERVATION_WINDOW)
        visibility = (
            "pending"
            if snapshot.outgoing_visible_eval_only is None
            else f"outgoing {snapshot.outgoing_visible_eval_only}/{snapshot.outgoing_total_eval_only}, side {snapshot.side_visible_eval_only}/2"
        )
        lines = [
            f"{'PAUSED' if paused else 'RUNNING'} | case: {session.case_id}",
            f"frame={snapshot.physics_frame}  t={snapshot.timestamp:.6f}s  speed={snapshot.speed:.4f}",
            f"pipeline state: {snapshot.pipeline_state}",
            f"corridor state: {snapshot.corridor_state}",
            f"event: {snapshot.latest_event}",
            f"side_walls_valid={snapshot.side_walls_valid}",
            f"current width={_finite(snapshot.current_width)}  stable width={_finite(snapshot.estimated_corridor_width)}",
            f"offset={_finite(snapshot.estimated_offset, True)}  orientation={_finite(snapshot.stable_orientation_deg, True)} deg",
            f"model initialized={snapshot.stable_model_initialized}  updates={snapshot.stable_update_count}",
            f"EXPECTED SOURCE: {snapshot.expected_profile_source}",
            "Expected/margin clipped at sensor max" if np.any(snapshot.expected_ranges + snapshot.numerical_margin >= LIDAR_MAX_RANGE) else "Expected/margin inside sensor range",
            f"OPEN candidates={snapshot.opening_candidate_count}  groups={len(snapshot.opening_groups)}",
            f"JUNCTION DETECTED EARLY={snapshot.junction_detected_latched}",
            f"Left center={_finite(snapshot.left_group_center, True)}  in [-135,-45]={snapshot.left_in_lateral_sector}",
            f"Right center={_finite(snapshot.right_group_center, True)}  in [45,135]={snapshot.right_in_lateral_sector}",
            f"Bilateral streak={streak}/{EXISTING_OBSERVATION_WINDOW}  BRAKE_READY={snapshot.brake_ready}",
            f"braking_active={snapshot.braking_active}  trigger={_empty(snapshot.brake_trigger_frame)}",
            f"distance after trigger={snapshot.brake_distance_eval_only:.3f} (EVAL ONLY)",
            f"anchor frame={_empty(snapshot.anchor_enter_frame)}  hold={snapshot.anchor_hold_duration:.3f}s",
            f"post-anchor LiDAR samples={snapshot.post_anchor_lidar_samples}",
            f"visibility EVAL ONLY: {visibility}",
        ]
        for index, value in enumerate(lines):
            color = COLORS["text"]
            if value.startswith("pipeline") or value.startswith("event"):
                color = COLORS["group_center"]
            if "BRAKE_READY=True" in value or "PROVISIONAL_ANCHOR" in value:
                color = COLORS["detected"]
            self.text(value, (panel_x, y + index * step), color, self.small_font)

        log_x, log_y = 30, 65
        overlay = self.pygame.Surface((330, 156), self.pygame.SRCALPHA)
        overlay.fill((9, 12, 18, 188))
        self.screen.blit(overlay, (log_x - 8, log_y - 8))
        self.text("EVENT LOG", (log_x, log_y), COLORS["group_center"], self.small_font)
        visible_events = [
            event for event in session.events
            if int(event["frame"]) <= snapshot.physics_frame
        ]
        for index, event in enumerate(visible_events[-7:]):
            self.text(
                f"f{event['frame']:>3}  {event['event']}",
                (log_x, log_y + 20 + index * 18), COLORS["text"], self.small_font,
            )
        self.text(
            "SPACE pause/resume   R restart   LEFT/RIGHT sampled frame   P profile   ESC quit",
            (20, 878), COLORS["muted"], self.small_font,
        )
        self.pygame.display.flip()


def _empty(value: Any) -> Any:
    return "" if value is None else value


def _write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_case(case_id: str, frames: int) -> IntegratedSession:
    session = IntegratedSession(case_id)
    for _ in range(frames):
        session.advance_physics_frame()
    return session


def replay_signature(session: IntegratedSession) -> tuple[Any, ...]:
    summary = session.summary()
    return (
        summary["ready_frame"], summary["first_open_frame"],
        summary["first_detection_frame"], summary["bilateral_entry_frame"],
        summary["candidate_b_trigger_frame"], summary["braking_start_frame"],
        summary["stop_frame"], summary["anchor_frame"],
        round(float(summary["anchor_x_eval_only"]), 9)
        if math.isfinite(float(summary["anchor_x_eval_only"])) else None,
        round(float(summary["anchor_y_eval_only"]), 9)
        if math.isfinite(float(summary["anchor_y_eval_only"])) else None,
        tuple((row["frame"], row["event"], row["pipeline_state"]) for row in session.events),
    )


def _assert_case(session: IntegratedSession) -> None:
    if session.case_id == BOOTSTRAP_ALIAS and session.next_physics_frame > 222:
        summary = session.summary()
        checks = {
            "ready_6": summary["ready_frame"] == 6,
            "open_30": summary["first_open_frame"] == 30,
            "detection_36": summary["first_detection_frame"] == 36,
            "bilateral_entry_174": summary["bilateral_entry_frame"] == 174,
            "candidate_b_180": summary["candidate_b_trigger_frame"] == 180,
            "braking_181": summary["braking_start_frame"] == 181,
            "anchor_221": summary["anchor_frame"] == 221,
            "anchor_x": abs(float(summary["anchor_x_eval_only"]) + 1.777661529818736) <= 1.0e-6,
            "anchor_y": abs(float(summary["anchor_y_eval_only"]) + 40.329315371814026) <= 1.0e-6,
            "inside": bool(summary["inside_junction_eval_only"]),
        }
        if not all(checks.values()):
            raise AssertionError(json.dumps(checks, sort_keys=True))
    if session.case_id == M0_ALIAS:
        checks = {
            "no_detection": session.first_detection_frame is None,
            "no_brake": session.brake_trigger_frame is None,
            "no_anchor": session.anchor_enter_frame is None,
        }
        if not all(checks.values()):
            raise AssertionError(json.dumps(checks, sort_keys=True))


def write_reports(
    output: Path,
    sessions: Iterable[IntegratedSession],
    replay: dict[str, bool | None],
    hashes_before: dict[str, str],
    hashes_after: dict[str, str],
) -> None:
    sessions = list(sessions)
    _write(output / "integrated_gui_timeline.csv", (row for session in sessions for row in session.timeline))
    _write(output / "integrated_gui_events.csv", (row for session in sessions for row in session.events))
    _write(output / "integrated_gui_summary.csv", (session.summary(replay.get(session.case_id)) for session in sessions))
    _write(
        output / "integrated_gui_protected_hashes.csv",
        (
            {
                "path": path,
                "sha256_before": hashes_before[path],
                "sha256_after": hashes_after[path],
                "unchanged": hashes_before[path] == hashes_after[path],
            }
            for path in hashes_before
        ),
    )


def render_keyframes(session: IntegratedSession, output: Path) -> None:
    import pygame
    pygame.init()
    renderer = IntegratedRenderer(pygame, session.runner.geometry, True)
    frames = {
        "A_pre_corridor_start": 0,
        "B_model_ready": session.first_ready_frame,
        "C_first_open": session.first_open_frame,
        "D_early_detection": session.first_detection_frame,
        "E_bilateral_streak_1": session.bilateral_entry_frame,
        "F_brake_ready": session.brake_trigger_frame,
        "G_provisional_anchor": session.anchor_enter_frame,
    }
    output.mkdir(parents=True, exist_ok=True)
    for label, frame in frames.items():
        if frame is None:
            continue
        indices = [
            index for index, item in enumerate(session.snapshots)
            if item.physics_frame >= frame
        ]
        if not indices:
            continue
        session.view_index = indices[0]
        renderer.draw(session, True)
        pygame.image.save(renderer.screen, output / f"{label}_frame_{frame}.png")
    pygame.quit()


def run_gui(args: argparse.Namespace) -> IntegratedSession:
    import pygame
    pygame.init()
    session = IntegratedSession(args.map_case)
    renderer = IntegratedRenderer(pygame, session.runner.geometry, args.show_profile)
    clock = pygame.time.Clock()
    paused = bool(args.start_paused)
    running = True
    session.advance_physics_frame()
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
                        session.view_index = len(session.snapshots) - 1
                elif event.key == pygame.K_r:
                    session.restart()
                    renderer.geometry = session.runner.geometry
                    renderer._configure_camera()
                    paused = bool(args.start_paused)
                    session.advance_physics_frame()
                elif event.key == pygame.K_LEFT and paused:
                    session.step_sample(-1)
                elif event.key == pygame.K_RIGHT and paused:
                    session.step_sample(1)
                elif event.key == pygame.K_p:
                    renderer.show_profile = not renderer.show_profile
        if not paused:
            session.advance_physics_frame()
        renderer.draw(session, paused)
        if args.frames > 0 and session.next_physics_frame >= args.frames:
            running = False
        clock.tick(args.fps)
    pygame.quit()
    return session


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-case", choices=MAP_CASES, default=BOOTSTRAP_ALIAS)
    parser.add_argument("--frames", type=int, default=0, help="0: GUI until ESC; headless default 230")
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
        args.frames = 230
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    hashes_before = protected_hashes()
    replay: dict[str, bool | None] = {args.map_case: None}
    if args.headless:
        session = run_case(args.map_case, args.frames)
        _assert_case(session)
        if args.deterministic_replay:
            replay_session = run_case(args.map_case, args.frames)
            replay[args.map_case] = replay_signature(session) == replay_signature(replay_session)
            if not replay[args.map_case]:
                raise AssertionError("deterministic replay mismatch")
        if args.render_keyframes:
            render_keyframes(session, args.output_dir / "keyframes")
    else:
        session = run_gui(args)
        _assert_case(session)
    hashes_after = protected_hashes()
    if hashes_before != hashes_after:
        raise AssertionError("protected source hash changed during EXP-048")
    write_reports(args.output_dir, [session], replay, hashes_before, hashes_after)
    print(" ".join(f"{key}={value}" for key, value in session.summary(replay[args.map_case]).items()))


if __name__ == "__main__":
    main()
