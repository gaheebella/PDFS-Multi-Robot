"""Read-only Moving LiDAR GUI using the frozen 55% adaptive detector.

The existing visualizer's SimulationRunner factory owns motion, physics, map,
and LiDAR generation.  Each sampled body-local angle/range scan is passed to
the existing Point Cloud detector diagnostics API.  Adaptive output is copied
for display and never feeds the simulator or the old profile detector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
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

from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import (  # noqa: E402
    _detect_openings_with_diagnostics,
)
from pygame_simulator.lidar_junction_detection_visualizer import (  # noqa: E402
    COLORS,
    MAIN_RECT,
    MAP_CASES,
    PROFILE_RECT,
    Renderer as BaseRenderer,
    _new_runner,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import (  # noqa: E402
    DT,
    LIDAR_MAX_RANGE,
    SAMPLE_PERIOD,
)


EXPERIMENT_NAME = "Moving LiDAR Adaptive 55% Junction Detection GUI"
DETECTOR_NAME = "ADAPTIVE_RANGE_55_PERCENT"
DEFAULT_OUTPUT = (
    ROOT / "junction_detection/integration/output/moving_lidar_55pct_detector"
)
PROTECTED_PATHS = (
    "pygame_simulator/pre_exploration_general_pipeline_simulator.py",
    "junction_detection/pointcloud/lidar_profile_junction_detector.py",
    "junction_detection/pointcloud/pointcloud_junction_detector_sensor_enhanced.py",
    "pygame_simulator/lidar_junction_detection_visualizer.py",
)
PARAMETERS = {
    "smoothing_window_size": 5,
    "wall_reference_quantile": 0.25,
    "far_range_fraction": 0.55,
    "merge_gap_deg": 3.0,
    "min_opening_width_deg": 5.0,
    "gradient_threshold": None,
    "gradient_mad_scale": 4.0,
    "min_gradient_threshold": 0.05,
    "boundary_search_deg": 6.0,
}
TIMELINE_FIELDS = (
    "map_case", "frame", "timestamp", "wall_reference", "range_ceiling",
    "dynamic_span", "open_threshold", "open_support_count",
    "opening_group_count", "junction_detected", "leader_x_eval_only",
    "leader_y_eval_only", "runtime_gt_map_used",
)
COMPARISON_FIELDS = (
    "map_case", "frame", "timestamp", "old_profile_detected",
    "new_55pct_detected", "old_open_count", "new_open_support_count",
    "new_opening_count", "comparison_only",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_hashes() -> dict[str, str]:
    return {path: _sha256(ROOT / path) for path in PROTECTED_PATHS}


def _audit_detector_defaults() -> dict[str, Any]:
    """Fail closed if the imported frozen detector defaults ever drift."""
    signature = inspect.signature(_detect_openings_with_diagnostics)
    actual = {
        name: signature.parameters[name].default
        for name in PARAMETERS
    }
    if actual != PARAMETERS:
        raise AssertionError(
            f"frozen detector default mismatch: expected={PARAMETERS!r} actual={actual!r}"
        )
    return actual


def _write(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class AdaptiveSnapshot:
    physics_frame: int
    timestamp: float
    robot_positions: np.ndarray
    leader_position: np.ndarray
    leader_velocity: np.ndarray
    lidar_yaw_deg: float
    angles_deg: np.ndarray
    raw_ranges: np.ndarray
    smoothed_ranges: np.ndarray
    open_support_mask: np.ndarray
    open_threshold: float
    wall_reference: float
    range_ceiling: float
    dynamic_span: float
    gradient_threshold: float
    opening_groups: tuple[dict[str, float], ...]
    junction_detected: bool
    old_profile_detected_comparison_only: bool
    old_open_count_comparison_only: int


def _adaptive_detector(
    angles_deg: np.ndarray, ranges: np.ndarray
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """The only detector boundary: body-local angle and range arrays."""
    return _detect_openings_with_diagnostics(
        np.asarray(angles_deg, dtype=float).copy(),
        np.asarray(ranges, dtype=float).copy(),
    )


class AdaptiveSession:
    """Run unchanged local-forward physics and consume adaptive output."""

    def __init__(self, map_case: str) -> None:
        self.map_case = map_case
        self.runner = _new_runner(map_case)
        self.next_physics_frame = 0
        self.snapshots: list[AdaptiveSnapshot] = []
        self.view_index = -1
        self.timeline: list[dict[str, Any]] = []
        self.comparison: list[dict[str, Any]] = []
        self.physics_trajectory: list[tuple[float, float]] = []
        self.first_open_support_frame: int | None = None
        self.first_open_support_time: float | None = None
        self.first_opening_frame: int | None = None
        self.first_opening_time: float | None = None
        self.first_detection_frame: int | None = None
        self.first_detection_time: float | None = None
        self.old_first_detection_frame: int | None = None

    def restart(self) -> None:
        self.__init__(self.map_case)

    def _leader(self) -> Any:
        return next(
            robot
            for robot in self.runner.world.robots
            if robot.robot_id == self.runner.world.lidar_robot_id
        )

    def advance_physics_frame(self) -> AdaptiveSnapshot | None:
        frame = self.next_physics_frame
        row = self.runner.step(frame)
        self.next_physics_frame += 1
        leader = self._leader()
        self.physics_trajectory.append(
            (float(leader.position[0]), float(leader.position[1]))
        )
        if row is None:
            return None

        observation = self.runner.last_visual[0]
        scan = observation.lidar_scan
        openings, diagnostics = _adaptive_detector(
            scan.angles_deg, scan.ranges
        )
        raw = np.asarray(scan.ranges, dtype=float)
        smoothed = np.asarray(diagnostics["smoothed_ranges"], dtype=float)
        support = np.asarray(diagnostics["open_support_mask"], dtype=bool)
        wall_reference = float(diagnostics["wall_reference"])
        range_ceiling = float(diagnostics["range_ceiling"])
        dynamic_span = max(0.0, range_ceiling - wall_reference)
        old = self.runner.last_profile_result
        old_detected = bool(old["profile_junction_detected"])
        old_count = int(old["opening_group_count"])
        junction_detected = len(openings) > 0

        snapshot = AdaptiveSnapshot(
            physics_frame=frame,
            timestamp=float(row["timestamp"]),
            robot_positions=np.array(
                [robot.position.copy() for robot in self.runner.world.robots]
            ),
            leader_position=leader.position.copy(),
            leader_velocity=leader.observed_velocity.copy(),
            lidar_yaw_deg=float(self.runner.world.lidar_yaw_deg),
            angles_deg=np.asarray(scan.angles_deg, dtype=float).copy(),
            raw_ranges=raw.copy(),
            smoothed_ranges=smoothed.copy(),
            open_support_mask=support.copy(),
            open_threshold=float(diagnostics["open_threshold"]),
            wall_reference=wall_reference,
            range_ceiling=range_ceiling,
            dynamic_span=dynamic_span,
            gradient_threshold=float(diagnostics["gradient_threshold"]),
            opening_groups=tuple(dict(opening) for opening in openings),
            junction_detected=junction_detected,
            old_profile_detected_comparison_only=old_detected,
            old_open_count_comparison_only=old_count,
        )
        self.snapshots.append(snapshot)
        self.view_index = len(self.snapshots) - 1

        support_count = int(np.count_nonzero(support))
        if support_count > 0 and self.first_open_support_frame is None:
            self.first_open_support_frame = frame
            self.first_open_support_time = snapshot.timestamp
        if openings and self.first_opening_frame is None:
            self.first_opening_frame = frame
            self.first_opening_time = snapshot.timestamp
        if junction_detected and self.first_detection_frame is None:
            self.first_detection_frame = frame
            self.first_detection_time = snapshot.timestamp
        if old_detected and self.old_first_detection_frame is None:
            self.old_first_detection_frame = frame

        self.timeline.append(
            {
                "map_case": self.map_case,
                "frame": frame,
                "timestamp": snapshot.timestamp,
                "wall_reference": wall_reference,
                "range_ceiling": range_ceiling,
                "dynamic_span": dynamic_span,
                "open_threshold": snapshot.open_threshold,
                "open_support_count": support_count,
                "opening_group_count": len(openings),
                "junction_detected": junction_detected,
                "leader_x_eval_only": float(leader.position[0]),
                "leader_y_eval_only": float(leader.position[1]),
                "runtime_gt_map_used": False,
            }
        )
        self.comparison.append(
            {
                "map_case": self.map_case,
                "frame": frame,
                "timestamp": snapshot.timestamp,
                "old_profile_detected": old_detected,
                "new_55pct_detected": junction_detected,
                "old_open_count": old_count,
                "new_open_support_count": support_count,
                "new_opening_count": len(openings),
                "comparison_only": True,
            }
        )
        return snapshot

    @property
    def current(self) -> AdaptiveSnapshot | None:
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

    def run(self, frames: int) -> "AdaptiveSession":
        for _ in range(frames):
            self.advance_physics_frame()
        return self

    def first_detection_snapshot(self) -> AdaptiveSnapshot | None:
        return next(
            (
                snapshot
                for snapshot in self.snapshots
                if snapshot.physics_frame == self.first_detection_frame
            ),
            None,
        )

    def summary(
        self,
        *,
        deterministic_replay: bool | None = None,
        movement_equivalent: bool | None = None,
        lidar_equivalent: bool | None = None,
    ) -> dict[str, Any]:
        detected = self.first_detection_snapshot()
        openings = [] if detected is None else list(detected.opening_groups)
        return {
            "map_case": self.map_case,
            "physics_frames": self.next_physics_frame,
            "sample_count": len(self.snapshots),
            "first_open_support_frame": _empty(self.first_open_support_frame),
            "first_open_support_time": _empty(self.first_open_support_time),
            "first_opening_frame": _empty(self.first_opening_frame),
            "first_opening_time": _empty(self.first_opening_time),
            "first_detection_frame": _empty(self.first_detection_frame),
            "first_detection_time": _empty(self.first_detection_time),
            "max_opening_count": max(
                (len(snapshot.opening_groups) for snapshot in self.snapshots),
                default=0,
            ),
            "old_profile_first_detection_frame_comparison_only": _empty(
                self.old_first_detection_frame
            ),
            "first_detection_wall_reference": "" if detected is None else detected.wall_reference,
            "first_detection_range_ceiling": "" if detected is None else detected.range_ceiling,
            "first_detection_dynamic_span": "" if detected is None else detected.dynamic_span,
            "first_detection_open_threshold": "" if detected is None else detected.open_threshold,
            "first_detection_openings": json.dumps(openings, sort_keys=True),
            "deterministic_replay": _empty(deterministic_replay),
            "movement_trajectory_equivalent": _empty(movement_equivalent),
            "lidar_scan_equivalent": _empty(lidar_equivalent),
            "movement_altered": False,
            "adaptive_output_fed_back": False,
            "expected_profile_used_for_gui_decision": False,
            "detector_input_fields": "angles_deg,ranges",
            "runtime_gt_map_used": False,
        }


def _empty(value: Any) -> Any:
    return "" if value is None else value


class AdaptiveRenderer(BaseRenderer):
    """Reuse camera/text setup and render only adaptive detector semantics."""

    def __init__(self, pygame: Any, geometry: Any, show_profile: bool) -> None:
        super().__init__(pygame, geometry, show_profile)
        pygame.display.set_caption(EXPERIMENT_NAME)

    def _draw_world(self, snapshot: AdaptiveSnapshot) -> None:
        pygame = self.pygame
        clip = self.screen.get_clip()
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
        origin = snapshot.leader_position
        for index in range(len(snapshot.angles_deg)):
            angle = snapshot.lidar_yaw_deg + float(snapshot.angles_deg[index])
            direction = np.array(
                [math.cos(math.radians(angle)), math.sin(math.radians(angle))]
            )
            endpoint = origin + direction * float(snapshot.raw_ranges[index])
            color = (
                COLORS["open_beam"]
                if snapshot.open_support_mask[index]
                else COLORS["normal_beam"]
            )
            pygame.draw.line(
                self.screen,
                color,
                self.world_to_screen(origin, snapshot),
                self.world_to_screen(endpoint, snapshot),
                2 if snapshot.open_support_mask[index] else 1,
            )
        for opening in snapshot.opening_groups:
            for key, color, width in (
                ("start_angle", COLORS["group_edge"], 3),
                ("end_angle", COLORS["group_edge"], 3),
                ("center_angle", COLORS["group_center"], 4),
            ):
                angle = snapshot.lidar_yaw_deg + float(opening[key])
                direction = np.array(
                    [math.cos(math.radians(angle)), math.sin(math.radians(angle))]
                )
                endpoint = origin + direction * LIDAR_MAX_RANGE
                pygame.draw.line(
                    self.screen,
                    color,
                    self.world_to_screen(origin, snapshot),
                    self.world_to_screen(endpoint, snapshot),
                    width,
                )
        for position in snapshot.robot_positions:
            pygame.draw.circle(
                self.screen,
                COLORS["robot"],
                self.world_to_screen(position, snapshot),
                2,
            )
        pygame.draw.circle(
            self.screen,
            COLORS["leader"],
            self.world_to_screen(origin, snapshot),
            7,
        )
        speed = float(np.linalg.norm(snapshot.leader_velocity))
        if speed > 1.0e-9:
            arrow = origin + snapshot.leader_velocity / speed * 28.0
            pygame.draw.line(
                self.screen,
                COLORS["group_center"],
                self.world_to_screen(origin, snapshot),
                self.world_to_screen(arrow, snapshot),
                5,
            )
        self.screen.set_clip(clip)

    def _draw_profile(self, snapshot: AdaptiveSnapshot) -> None:
        pygame = self.pygame
        x, y, width, height = PROFILE_RECT
        pygame.draw.rect(self.screen, (25, 31, 40), PROFILE_RECT, border_radius=5)
        for angle in (-180, -90, 0, 90, 180):
            px, _ = self._plot_point(PROFILE_RECT, angle, 0.0)
            pygame.draw.line(self.screen, (55, 64, 76), (px, y), (px, y + height), 1)
            self.text(str(angle), (px - 14, y + height + 7), COLORS["muted"], self.small_font)
        for value in (0, 50, 100, 150):
            _, py = self._plot_point(PROFILE_RECT, -180.0, value)
            pygame.draw.line(self.screen, (55, 64, 76), (x, py), (x + width, py), 1)
            self.text(str(value), (x - 35, py - 8), COLORS["muted"], self.small_font)
        for index in np.flatnonzero(snapshot.open_support_mask):
            left = self._plot_point(PROFILE_RECT, float(snapshot.angles_deg[index]) - 0.5, 0.0)[0]
            right = self._plot_point(PROFILE_RECT, float(snapshot.angles_deg[index]) + 0.5, 0.0)[0]
            overlay = pygame.Surface((max(1, right - left + 1), height), pygame.SRCALPHA)
            overlay.fill((*COLORS["open_fill"], 62))
            self.screen.blit(overlay, (left, y))
        raw_points = [
            self._plot_point(PROFILE_RECT, float(angle), float(value))
            for angle, value in zip(snapshot.angles_deg, snapshot.raw_ranges)
        ]
        smooth_points = [
            self._plot_point(PROFILE_RECT, float(angle), float(value))
            for angle, value in zip(snapshot.angles_deg, snapshot.smoothed_ranges)
        ]
        threshold_y = self._plot_point(
            PROFILE_RECT, 0.0, snapshot.open_threshold
        )[1]
        pygame.draw.lines(self.screen, COLORS["measured"], False, raw_points, 2)
        pygame.draw.lines(self.screen, COLORS["expected"], False, smooth_points, 2)
        pygame.draw.line(
            self.screen,
            COLORS["threshold"],
            (x, threshold_y),
            (x + width, threshold_y),
            2,
        )
        for opening in snapshot.opening_groups:
            for key, color in (
                ("start_angle", COLORS["group_edge"]),
                ("end_angle", COLORS["group_edge"]),
                ("center_angle", COLORS["group_center"]),
            ):
                px = self._plot_point(PROFILE_RECT, float(opening[key]), 0.0)[0]
                pygame.draw.line(self.screen, color, (px, y), (px, y + height), 2)
        pygame.draw.rect(self.screen, COLORS["muted"], PROFILE_RECT, 1, border_radius=5)
        legend = (
            ("RAW", COLORS["measured"]),
            ("SMOOTHED", COLORS["expected"]),
            ("55% OPEN THRESHOLD", COLORS["threshold"]),
            ("OPEN SUPPORT", COLORS["open_beam"]),
        )
        cursor = x + 7
        for label, color in legend:
            pygame.draw.line(self.screen, color, (cursor, y + 13), (cursor + 13, y + 13), 3)
            self.text(label, (cursor + 17, y + 5), color, self.small_font)
            cursor += 28 + self.small_font.size(label)[0]
        self.text("range", (x - 35, y - 24), COLORS["muted"], self.small_font)
        self.text("LiDAR angle theta [deg]", (x + width // 2 - 76, y + height + 28), COLORS["muted"], self.small_font)

    def draw(self, session: AdaptiveSession, paused: bool) -> None:
        snapshot = session.current
        self.screen.fill(COLORS["background"])
        self.text(
            "Moving LiDAR Junction Detection — adaptive 55% frozen detector",
            (18, 15),
            font=self.title_font,
        )
        if snapshot is None:
            self.text("Waiting for first sampled LiDAR scan...", (44, 90))
            self.pygame.display.flip()
            return
        self._draw_world(snapshot)
        if self.show_profile:
            self._draw_profile(snapshot)
        panel_x = 870
        y = 500 if self.show_profile else 72
        step = 18
        lines = [
            f"{'PAUSED' if paused else 'RUNNING'} | map case: {session.map_case}",
            f"frame={snapshot.physics_frame} t={snapshot.timestamp:.6f}s",
            f"DETECTOR: {DETECTOR_NAME}",
            f"far_range_fraction = {PARAMETERS['far_range_fraction']:.2f}",
            f"wall_reference = {snapshot.wall_reference:.6f}",
            f"range_ceiling = {snapshot.range_ceiling:.6f}",
            f"dynamic_span = {snapshot.dynamic_span:.6f}",
            f"open_threshold = {snapshot.open_threshold:.6f}",
            f"OPEN support count = {int(np.count_nonzero(snapshot.open_support_mask))}",
            f"opening group count = {len(snapshot.opening_groups)}",
            f"JUNCTION_DETECTED = {snapshot.junction_detected}",
            f"first detection = {_detection_text(session)}",
            "PURPLE = 55% OPEN SUPPORT",
        ]
        for index, value in enumerate(lines):
            color = COLORS["text"]
            if value.startswith("DETECTOR") or value.startswith("PURPLE"):
                color = COLORS["group_center"]
            if "JUNCTION_DETECTED = True" in value:
                color = COLORS["detected"]
            self.text(value, (panel_x, y + index * step), color, self.small_font)
        groups_y = y + len(lines) * step + 6
        self.text("Opening groups (final detector output)", (panel_x, groups_y), COLORS["group_center"], self.small_font)
        if not snapshot.opening_groups:
            self.text("none", (panel_x, groups_y + 20), COLORS["muted"], self.small_font)
        for index, opening in enumerate(snapshot.opening_groups):
            base = groups_y + 20 + index * 18
            self.text(
                f"#{index} start={opening['start_angle']:+.1f} end={opening['end_angle']:+.1f} "
                f"center={opening['center_angle']:+.1f} width={opening['width_deg']:.1f} "
                f"confidence={opening['confidence']:.3f}",
                (panel_x, base),
                COLORS["text"],
                self.small_font,
            )
        formula_y = groups_y + 25 + max(1, len(snapshot.opening_groups)) * 18
        self.text("open_threshold = wall_reference", (panel_x, formula_y), COLORS["muted"], self.small_font)
        self.text("  + 0.55 x (range_ceiling - wall_reference)", (panel_x, formula_y + 18), COLORS["muted"], self.small_font)
        self.text(
            "SPACE pause/resume   R restart   LEFT/RIGHT sampled frame   P profile   ESC quit",
            (20, 878),
            COLORS["muted"],
            self.small_font,
        )
        self.pygame.display.flip()


def _detection_text(session: AdaptiveSession) -> str:
    if session.first_detection_frame is None:
        return "none"
    return f"frame {session.first_detection_frame}, t={session.first_detection_time:.6f}s"


def _session_signature(session: AdaptiveSession) -> tuple[Any, ...]:
    return (
        session.first_open_support_frame,
        session.first_opening_frame,
        session.first_detection_frame,
        session.old_first_detection_frame,
        tuple(
            (
                snapshot.physics_frame,
                round(snapshot.wall_reference, 9),
                round(snapshot.range_ceiling, 9),
                round(snapshot.open_threshold, 9),
                tuple(np.flatnonzero(snapshot.open_support_mask)),
                tuple(
                    (
                        round(opening["start_angle"], 9),
                        round(opening["end_angle"], 9),
                        round(opening["center_angle"], 9),
                        round(opening["width_deg"], 9),
                        round(opening["confidence"], 9),
                    )
                    for opening in snapshot.opening_groups
                ),
            )
            for snapshot in session.snapshots
        ),
    )


def _movement_equivalent(
    first: AdaptiveSession, second: AdaptiveSession
) -> tuple[bool, bool]:
    movement = bool(
        len(first.physics_trajectory) == len(second.physics_trajectory)
        and np.allclose(
            np.asarray(first.physics_trajectory),
            np.asarray(second.physics_trajectory),
            atol=0.0,
            rtol=0.0,
        )
    )
    lidar = bool(
        len(first.snapshots) == len(second.snapshots)
        and all(
            np.array_equal(left.angles_deg, right.angles_deg)
            and np.array_equal(left.raw_ranges, right.raw_ranges)
            for left, right in zip(first.snapshots, second.snapshots)
        )
    )
    return movement, lidar


def run_gui(args: argparse.Namespace) -> AdaptiveSession:
    import pygame

    pygame.init()
    session = AdaptiveSession(args.map_case)
    renderer = AdaptiveRenderer(pygame, session.runner.geometry, args.show_profile)
    clock = pygame.time.Clock()
    paused = bool(args.start_paused)
    pause_consumed = False
    running = True
    first = session.advance_physics_frame()
    if args.pause_on_detect and first is not None and first.junction_detected:
        paused = True
        pause_consumed = True
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
                    pause_consumed = False
                    session.advance_physics_frame()
                elif event.key == pygame.K_LEFT and paused:
                    session.step_sample(-1)
                elif event.key == pygame.K_RIGHT and paused:
                    session.step_sample(1)
                elif event.key == pygame.K_p:
                    renderer.show_profile = not renderer.show_profile
        if not paused:
            snapshot = session.advance_physics_frame()
            if (
                args.pause_on_detect
                and not pause_consumed
                and snapshot is not None
                and snapshot.junction_detected
            ):
                paused = True
                pause_consumed = True
        renderer.draw(session, paused)
        if args.frames > 0 and session.next_physics_frame >= args.frames:
            running = False
        clock.tick(args.fps)
    if args.screenshot is not None:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(renderer.screen, args.screenshot)
    pygame.quit()
    return session


def _run_replayed_case(
    map_case: str, frames: int
) -> tuple[AdaptiveSession, bool, bool, bool]:
    primary = AdaptiveSession(map_case).run(frames)
    replay = AdaptiveSession(map_case).run(frames)
    movement, lidar = _movement_equivalent(primary, replay)
    deterministic = _session_signature(primary) == _session_signature(replay)
    if not movement or not lidar or not deterministic:
        raise AssertionError(
            json.dumps(
                {
                    "map_case": map_case,
                    "movement_equivalent": movement,
                    "lidar_equivalent": lidar,
                    "deterministic_replay": deterministic,
                },
                sort_keys=True,
            )
        )
    return primary, deterministic, movement, lidar


def write_reports(
    output: Path,
    sessions: list[AdaptiveSession],
    replay: dict[str, bool],
    movement: dict[str, bool],
    lidar: dict[str, bool],
    hashes_before: dict[str, str],
    hashes_after: dict[str, str],
) -> None:
    _write(
        output / "moving_lidar_55pct_timeline.csv",
        (row for session in sessions for row in session.timeline),
        TIMELINE_FIELDS,
    )
    _write(
        output / "detector_comparison.csv",
        (row for session in sessions for row in session.comparison),
        COMPARISON_FIELDS,
    )
    summaries = [
        session.summary(
            deterministic_replay=replay[session.map_case],
            movement_equivalent=movement[session.map_case],
            lidar_equivalent=lidar[session.map_case],
        )
        for session in sessions
    ]
    _write(output / "case_summary.csv", summaries, tuple(summaries[0]))
    hash_rows = [
        {
            "path": path,
            "sha256_before": hashes_before[path],
            "sha256_after": hashes_after[path],
            "unchanged": hashes_before[path] == hashes_after[path],
        }
        for path in hashes_before
    ]
    _write(
        output / "protected_hashes.csv",
        hash_rows,
        ("path", "sha256_before", "sha256_after", "unchanged"),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-case", choices=MAP_CASES, default="M1_CROSS_BASELINE")
    parser.add_argument("--frames", type=int, default=0, help="0: GUI until ESC; headless/validate default 240")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--pause-on-detect", action="store_true")
    parser.add_argument("--show-profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--validate", action="store_true", help="headless M0/M1 plus deterministic replay")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error("--frames must be non-negative")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if (args.headless or args.validate) and args.frames == 0:
        args.frames = 240
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _audit_detector_defaults()
    hashes_before = protected_hashes()
    sessions: list[AdaptiveSession]
    replay: dict[str, bool] = {}
    movement: dict[str, bool] = {}
    lidar: dict[str, bool] = {}
    if args.validate:
        sessions = []
        for map_case in MAP_CASES:
            session, replay[map_case], movement[map_case], lidar[map_case] = (
                _run_replayed_case(map_case, args.frames)
            )
            sessions.append(session)
    elif args.headless:
        session, replay[args.map_case], movement[args.map_case], lidar[args.map_case] = (
            _run_replayed_case(args.map_case, args.frames)
        )
        sessions = [session]
    else:
        session = run_gui(args)
        sessions = [session]
        replay[args.map_case] = False
        movement[args.map_case] = False
        lidar[args.map_case] = False
    hashes_after = protected_hashes()
    if hashes_before != hashes_after:
        raise AssertionError("protected source hash changed during adaptive GUI run")
    write_reports(
        args.output_dir,
        sessions,
        replay,
        movement,
        lidar,
        hashes_before,
        hashes_after,
    )
    for session in sessions:
        print(
            json.dumps(
                session.summary(
                    deterministic_replay=replay[session.map_case],
                    movement_equivalent=movement[session.map_case],
                    lidar_equivalent=lidar[session.map_case],
                ),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
