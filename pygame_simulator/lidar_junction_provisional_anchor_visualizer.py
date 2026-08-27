"""EXP-041: frozen moving-LiDAR detection to Provisional Anchor integration.

This experiment reuses ``SimulationRunner(..., hold_on_profile_detection=True)``
and the existing local controller.  It does not implement a detector, braking
law, pose threshold, Point Cloud stage, or map/GT-triggered transition.
"""

from __future__ import annotations

import argparse
import csv
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

from junction_detection.pointcloud.lidar_profile_junction_detector import (  # noqa: E402
    GeometryProfileConfig,
    LidarProfileJunctionDetector,
)
from pygame_simulator.lidar_junction_detection_visualizer import (  # noqa: E402
    COLORS,
    DisplaySnapshot,
    Renderer as ProfileRenderer,
    _copy_snapshot,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import (  # noqa: E402
    DT,
    LIDAR_MAX_RANGE,
    SAMPLE_PERIOD,
    SimulationRunner,
)


EXPERIMENT_ID = "EXP-041"
MAP_CASES = ("M0_STRAIGHT", "M1_CROSS_BASELINE")
DEFAULT_OUTPUT = (
    ROOT / "junction_detection/integration/output/exp_041_provisional_anchor_stop"
)
EVENT_FIELDS = (
    "case",
    "frame",
    "timestamp",
    "event",
    "pipeline_state",
    "junction_detected",
    "opening_group_count",
    "speed",
    "x_eval_only",
    "y_eval_only",
    "yaw_eval_only",
)
SUMMARY_FIELDS = (
    "case",
    "first_detection_frame",
    "first_detection_time",
    "anchor_enter_frame",
    "anchor_enter_time",
    "stop_latency_frames",
    "stop_latency_seconds",
    "max_speed_after_anchor",
    "position_drift_after_anchor",
    "yaw_drift_after_anchor",
    "opening_group_count_at_detection",
    "detection_count",
    "anchor_transition_count",
    "lidar_samples_after_anchor",
    "profile_updates_after_anchor",
    "final_pipeline_state",
    "gt_or_map_used_for_transition",
    "pointcloud_detector_executed",
)


@dataclass(frozen=True)
class AnchorSnapshot(DisplaySnapshot):
    """Display copy of frozen detector output plus existing controller state."""

    pipeline_state: str
    latest_event: str
    speed: float
    first_detection_frame: int | None
    first_detection_time: float | None
    anchor_enter_frame: int | None
    anchor_enter_time: float | None
    anchor_hold_duration: float
    position_drift_eval_only: float
    yaw_drift_eval_only: float


class AnchorSession:
    """Run the frozen detector and consume the existing active-hold API."""

    def __init__(self, map_case: str) -> None:
        self.map_case = map_case
        detector = LidarProfileJunctionDetector(GeometryProfileConfig(LIDAR_MAX_RANGE))
        self.runner = SimulationRunner(
            map_case,
            "local_forward",
            profile_detector=detector,
            hold_on_profile_detection=True,
            pointcloud_detector=None,
        )
        self.next_physics_frame = 0
        self.snapshots: list[AnchorSnapshot] = []
        self.view_index = -1
        self.events: list[dict[str, Any]] = []
        self.first_detection_frame: int | None = None
        self.first_detection_time: float | None = None
        self.detection_opening_group_count = 0
        self.anchor_enter_frame: int | None = None
        self.anchor_enter_time: float | None = None
        self.anchor_transition_count = 0
        self.detection_count = 0
        self.lidar_samples_after_anchor = 0
        self.profile_updates_after_anchor = 0
        self.max_speed_after_anchor = 0.0
        self.position_drift_after_anchor = 0.0
        self.yaw_drift_after_anchor = 0.0
        self.latest_event = "START_MOVING"
        self._start_recorded = False
        self._hold_confirmed_recorded = False

    def restart(self) -> None:
        self.__init__(self.map_case)

    def _leader(self) -> Any:
        return next(
            robot for robot in self.runner.world.robots
            if robot.robot_id == self.runner.world.lidar_robot_id
        )

    def pipeline_state(self) -> str:
        world = self.runner.world
        if world.provisional_fixed_anchor:
            # Existing internal name denotes a stationary provisional
            # observation pose; EXP-041 does not promote a final FIXED_ANCHOR.
            return "PROVISIONAL_ANCHOR"
        if world.suspect_hold_active:
            return "JUNCTION_SUSPECTED_STOPPING"
        return "MOVING_LIDAR"

    def _event_row(
        self,
        event: str,
        frame: int,
        timestamp: float,
        junction_detected: bool,
        opening_group_count: int,
    ) -> dict[str, Any]:
        leader = self._leader()
        return {
            "case": self.map_case,
            "frame": frame,
            "timestamp": timestamp,
            "event": event,
            "pipeline_state": self.pipeline_state(),
            "junction_detected": junction_detected,
            "opening_group_count": opening_group_count,
            "speed": float(np.linalg.norm(leader.velocity)),
            "x_eval_only": float(leader.position[0]),
            "y_eval_only": float(leader.position[1]),
            "yaw_eval_only": math.degrees(float(leader.body_yaw_rad)),
        }

    def _record_event(
        self,
        event: str,
        frame: int,
        timestamp: float,
        junction_detected: bool,
        opening_group_count: int,
    ) -> None:
        self.latest_event = event
        self.events.append(
            self._event_row(
                event,
                frame,
                timestamp,
                junction_detected,
                opening_group_count,
            )
        )

    def _update_anchor_metrics(self) -> None:
        if self.anchor_enter_frame is None:
            return
        leader = self._leader()
        world = self.runner.world
        speed = float(np.linalg.norm(leader.velocity))
        self.max_speed_after_anchor = max(self.max_speed_after_anchor, speed)
        if world.anchor_position is not None:
            self.position_drift_after_anchor = max(
                self.position_drift_after_anchor,
                float(np.linalg.norm(leader.position - world.anchor_position)),
            )
        if math.isfinite(world.anchor_heading_rad):
            yaw_delta = abs(
                math.degrees(
                    (leader.body_yaw_rad - world.anchor_heading_rad + math.pi)
                    % (2.0 * math.pi)
                    - math.pi
                )
            )
            self.yaw_drift_after_anchor = max(self.yaw_drift_after_anchor, yaw_delta)

    def advance_physics_frame(self) -> AnchorSnapshot | None:
        frame = self.next_physics_frame
        row = self.runner.step(frame)
        self.next_physics_frame += 1
        world = self.runner.world
        profile = self.runner.last_profile_result
        current_detected = bool(profile and profile["profile_junction_detected"])
        group_count = int(profile["opening_group_count"]) if profile else 0

        if not self._start_recorded:
            self._record_event("START_MOVING", frame, float(world.time), False, 0)
            self._start_recorded = True

        if row is not None and bool(row["profile_junction_detected"]):
            self.detection_count += 1
            if self.first_detection_frame is None:
                self.first_detection_frame = frame
                self.first_detection_time = float(row["timestamp"])
                self.detection_opening_group_count = int(row["opening_group_count"])
                self._record_event(
                    "JUNCTION_SUSPECTED",
                    frame,
                    float(row["timestamp"]),
                    True,
                    int(row["opening_group_count"]),
                )

        if world.provisional_fixed_anchor and self.anchor_enter_frame is None:
            self.anchor_enter_frame = int(world.anchor_entry_frame)
            self.anchor_enter_time = float(world.anchor_entry_time)
            self.anchor_transition_count += 1
            self._record_event(
                "PROVISIONAL_ANCHOR_ENTER",
                self.anchor_enter_frame,
                self.anchor_enter_time,
                current_detected,
                group_count,
            )

        self._update_anchor_metrics()

        if row is None:
            return None

        if self.anchor_enter_frame is not None and frame > self.anchor_enter_frame:
            self.lidar_samples_after_anchor += 1
            self.profile_updates_after_anchor += int(profile is not None)
            if not self._hold_confirmed_recorded:
                self._record_event(
                    "ANCHOR_HOLD_CONFIRMED",
                    frame,
                    float(row["timestamp"]),
                    bool(row["profile_junction_detected"]),
                    int(row["opening_group_count"]),
                )
                self._hold_confirmed_recorded = True

        base = _copy_snapshot(self.runner, frame, row)
        hold_duration = (
            max(0.0, float(row["timestamp"]) - self.anchor_enter_time)
            if self.anchor_enter_time is not None
            else 0.0
        )
        snapshot = AnchorSnapshot(
            **base.__dict__,
            pipeline_state=self.pipeline_state(),
            latest_event=self.latest_event,
            speed=float(row["leader_speed"]),
            first_detection_frame=self.first_detection_frame,
            first_detection_time=self.first_detection_time,
            anchor_enter_frame=self.anchor_enter_frame,
            anchor_enter_time=self.anchor_enter_time,
            anchor_hold_duration=hold_duration,
            position_drift_eval_only=self.position_drift_after_anchor,
            yaw_drift_eval_only=self.yaw_drift_after_anchor,
        )
        self.snapshots.append(snapshot)
        self.view_index = len(self.snapshots) - 1
        return snapshot

    @property
    def current(self) -> AnchorSnapshot | None:
        if self.view_index < 0:
            return None
        return self.snapshots[self.view_index]

    def step_sample(self, direction: int) -> None:
        if direction < 0:
            self.view_index = max(0, self.view_index - 1)
            return
        if self.view_index + 1 < len(self.snapshots):
            self.view_index += 1
            return
        initial_count = len(self.snapshots)
        stride = max(1, round(SAMPLE_PERIOD / DT))
        for _ in range(stride + 1):
            self.advance_physics_frame()
            if len(self.snapshots) > initial_count:
                break

    def summary(self) -> dict[str, Any]:
        latency_frames = (
            self.anchor_enter_frame - self.first_detection_frame
            if self.anchor_enter_frame is not None and self.first_detection_frame is not None
            else ""
        )
        latency_seconds = (
            self.anchor_enter_time - self.first_detection_time
            if self.anchor_enter_time is not None and self.first_detection_time is not None
            else ""
        )
        return {
            "case": self.map_case,
            "first_detection_frame": _empty_if_none(self.first_detection_frame),
            "first_detection_time": _empty_if_none(self.first_detection_time),
            "anchor_enter_frame": _empty_if_none(self.anchor_enter_frame),
            "anchor_enter_time": _empty_if_none(self.anchor_enter_time),
            "stop_latency_frames": latency_frames,
            "stop_latency_seconds": latency_seconds,
            "max_speed_after_anchor": self.max_speed_after_anchor,
            "position_drift_after_anchor": self.position_drift_after_anchor,
            "yaw_drift_after_anchor": self.yaw_drift_after_anchor,
            "opening_group_count_at_detection": self.detection_opening_group_count,
            "detection_count": self.detection_count,
            "anchor_transition_count": self.anchor_transition_count,
            "lidar_samples_after_anchor": self.lidar_samples_after_anchor,
            "profile_updates_after_anchor": self.profile_updates_after_anchor,
            "final_pipeline_state": self.pipeline_state(),
            "gt_or_map_used_for_transition": False,
            "pointcloud_detector_executed": False,
        }


class AnchorRenderer(ProfileRenderer):
    """Retain profile rendering and add EXP-041 controller-state diagnostics."""

    def draw(self, session: AnchorSession, gui_paused: bool) -> None:
        snapshot = session.current
        self.screen.fill(COLORS["background"])
        self.text(
            "EXP-041 — LiDAR Junction Detection → Provisional Anchor",
            (18, 15),
            font=self.title_font,
        )
        if snapshot is None:
            self.text("Waiting for first sampled LiDAR scan...", (44, 90))
            self.pygame.display.flip()
            return

        self._draw_world(snapshot)
        if snapshot.pipeline_state == "PROVISIONAL_ANCHOR":
            self.pygame.draw.circle(
                self.screen,
                COLORS["detected"],
                self.world_to_screen(snapshot.leader_position, snapshot),
                11,
                3,
            )
        if self.show_profile:
            self._draw_profile(snapshot)

        panel_x = 872
        info_y = 530 if self.show_profile else 82
        step = 21
        gui_state = "GUI PAUSED" if gui_paused else "GUI RUNNING"
        pipeline_color = (
            COLORS["detected"]
            if snapshot.pipeline_state != "MOVING_LIDAR"
            else COLORS["clear"]
        )
        lines = [
            (f"{gui_state} | map case: {session.map_case}", COLORS["text"]),
            (f"pipeline state: {snapshot.pipeline_state}", pipeline_color),
            (f"event: {snapshot.latest_event}", COLORS["group_center"]),
            (f"frame={snapshot.physics_frame}  t={snapshot.timestamp:.6f} s", COLORS["text"]),
            (f"current speed={snapshot.speed:.9f}", COLORS["text"]),
            (f"JUNCTION_DETECTED current={snapshot.junction_detected}", COLORS["text"]),
            (f"opening groups={len(snapshot.opening_groups)}", COLORS["text"]),
            (_first_detection_label(snapshot), COLORS["text"]),
            (_anchor_label(snapshot), COLORS["text"]),
            (f"anchor hold duration={snapshot.anchor_hold_duration:.6f} s", COLORS["text"]),
            (f"position drift eval-only={snapshot.position_drift_eval_only:.3g}", COLORS["muted"]),
            (f"yaw drift eval-only={snapshot.yaw_drift_eval_only:.3g} deg", COLORS["muted"]),
            (f"post-anchor LiDAR samples={session.lidar_samples_after_anchor}", COLORS["muted"]),
        ]
        for index, (value, color) in enumerate(lines):
            self.text(value, (panel_x, info_y + index * step), color)

        group_y = info_y + len(lines) * step + 5
        self.text("Opening groups (frozen detector output)", (panel_x, group_y), COLORS["group_center"])
        if not snapshot.opening_groups:
            self.text("none", (panel_x, group_y + 22), COLORS["muted"], self.small_font)
        for index, group in enumerate(snapshot.opening_groups[:2]):
            self.text(
                f"#{group['group_id'] + 1} [{group['start_angle_deg']:+.1f}, {group['end_angle_deg']:+.1f}] "
                f"center={group['center_angle_deg']:+.1f} width={group['angular_width_deg']:.1f} deg",
                (panel_x, group_y + 22 + index * 20),
                COLORS["text"],
                self.small_font,
            )

        self.text(
            "SPACE GUI pause/resume   R restart   LEFT/RIGHT sampled frame   P profile   ESC quit",
            (20, 878),
            COLORS["muted"],
            self.small_font,
        )
        self.pygame.display.flip()


def _empty_if_none(value: Any) -> Any:
    return "" if value is None else value


def _first_detection_label(snapshot: AnchorSnapshot) -> str:
    if snapshot.first_detection_frame is None:
        return "first detection: none"
    return (
        f"first detection: frame {snapshot.first_detection_frame}, "
        f"t={snapshot.first_detection_time:.6f} s"
    )


def _anchor_label(snapshot: AnchorSnapshot) -> str:
    if snapshot.anchor_enter_frame is None:
        return "anchor enter: none"
    return (
        f"anchor enter: frame {snapshot.anchor_enter_frame}, "
        f"t={snapshot.anchor_enter_time:.6f} s"
    )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(output: Path, sessions: Iterable[AnchorSession]) -> None:
    sessions = list(sessions)
    _write_csv(
        output / "provisional_anchor_events.csv",
        (event for session in sessions for event in session.events),
        EVENT_FIELDS,
    )
    _write_csv(
        output / "provisional_anchor_summary.csv",
        (session.summary() for session in sessions),
        SUMMARY_FIELDS,
    )


def run_case(map_case: str, frames: int) -> AnchorSession:
    session = AnchorSession(map_case)
    for _ in range(frames):
        session.advance_physics_frame()
    return session


def replay_signature(session: AnchorSession) -> tuple[Any, ...]:
    """Use runtime outputs only; pose fields are evaluation-only comparison data."""
    summary = session.summary()
    return (
        summary["first_detection_frame"],
        summary["first_detection_time"],
        summary["anchor_enter_frame"],
        summary["anchor_enter_time"],
        summary["stop_latency_frames"],
        summary["opening_group_count_at_detection"],
        summary["max_speed_after_anchor"],
        summary["position_drift_after_anchor"],
        summary["yaw_drift_after_anchor"],
        tuple((row["frame"], row["event"], row["pipeline_state"]) for row in session.events),
    )


def run_gui(args: argparse.Namespace) -> AnchorSession:
    import pygame

    pygame.init()
    session = AnchorSession(args.map_case)
    renderer = AnchorRenderer(pygame, session.runner.geometry, args.show_profile)
    clock = pygame.time.Clock()
    gui_paused = bool(args.start_paused)
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
                    gui_paused = not gui_paused
                    if not gui_paused:
                        session.view_index = len(session.snapshots) - 1
                elif event.key == pygame.K_r:
                    session.restart()
                    renderer.geometry = session.runner.geometry
                    renderer._configure_camera()
                    gui_paused = bool(args.start_paused)
                    session.advance_physics_frame()
                elif event.key == pygame.K_LEFT and gui_paused:
                    session.step_sample(-1)
                elif event.key == pygame.K_RIGHT and gui_paused:
                    session.step_sample(1)
                elif event.key == pygame.K_p:
                    renderer.show_profile = not renderer.show_profile

        if not gui_paused:
            session.advance_physics_frame()
        renderer.draw(session, gui_paused)
        if args.frames > 0 and session.next_physics_frame >= args.frames:
            running = False
        clock.tick(args.fps)

    pygame.quit()
    return session


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-case", choices=(*MAP_CASES, "BOTH"), default="M1_CROSS_BASELINE")
    parser.add_argument("--frames", type=int, default=0, help="0 runs GUI until ESC; headless default is 600")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--deterministic-replay", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--show-profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error("--frames must be non-negative")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if not args.headless and args.map_case == "BOTH":
        parser.error("--map-case BOTH is headless-only")
    if args.headless and args.frames == 0:
        args.frames = 600
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.headless:
        cases = MAP_CASES if args.map_case == "BOTH" else (args.map_case,)
        sessions = [run_case(case, args.frames) for case in cases]
        replay_ok: bool | None = None
        if args.deterministic_replay:
            replays = [run_case(case, args.frames) for case in cases]
            replay_ok = all(
                replay_signature(first) == replay_signature(second)
                for first, second in zip(sessions, replays)
            )
        write_reports(args.output_dir, sessions)
        for session in sessions:
            print(" ".join(f"{key}={value}" for key, value in session.summary().items()))
        if replay_ok is not None:
            print(f"deterministic_replay={replay_ok}")
        return

    session = run_gui(args)
    write_reports(args.output_dir, [session])
    print(" ".join(f"{key}={value}" for key, value in session.summary().items()))


if __name__ == "__main__":
    main()
