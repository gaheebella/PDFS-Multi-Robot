"""Read-only Pygame visualizer for the frozen moving-LiDAR Junction detector.

The simulation and detector live in their production/frozen modules.  This
module only copies their outputs into immutable display snapshots and renders
them; it contains no Junction-detection implementation.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.pointcloud.lidar_profile_junction_detector import (  # noqa: E402
    GeometryProfileConfig,
    LidarProfileJunctionDetector,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import (  # noqa: E402
    DT,
    LIDAR_MAX_RANGE,
    SAMPLE_PERIOD,
    SimulationRunner,
)


MAP_CASES = ("M0_STRAIGHT", "M1_CROSS_BASELINE")
WINDOW_SIZE = (1440, 900)
MAIN_RECT = (16, 54, 824, 830)
PROFILE_RECT = (870, 72, 540, 410)
COLORS = {
    "background": (15, 19, 26),
    "floor": (42, 50, 61),
    "wall": (225, 230, 235),
    "robot": (61, 104, 157),
    "leader": (255, 225, 55),
    "normal_beam": (88, 112, 125),
    "candidate_beam": (255, 153, 43),
    "open_beam": (238, 65, 214),
    "group_edge": (255, 184, 76),
    "group_center": (80, 245, 225),
    "measured": (242, 242, 242),
    "expected": (72, 156, 255),
    "threshold": (255, 192, 66),
    "open_fill": (132, 43, 122),
    "text": (235, 239, 244),
    "muted": (155, 169, 184),
    "detected": (255, 83, 92),
    "clear": (69, 214, 126),
}


@dataclass(frozen=True)
class DisplaySnapshot:
    """Detached display data copied after the detector has returned."""

    physics_frame: int
    timestamp: float
    robot_positions: np.ndarray
    leader_position: np.ndarray
    leader_velocity: np.ndarray
    lidar_yaw_deg: float
    angles_deg: np.ndarray
    measured_ranges: np.ndarray
    expected_ranges: np.ndarray
    valid_mask: np.ndarray
    candidate_mask: np.ndarray
    confirmed_mask: np.ndarray
    opening_groups: tuple[dict[str, Any], ...]
    detector_state: str
    junction_detected: bool
    opening_candidate_count: int
    estimated_corridor_width: float
    estimated_offset: float
    stable_orientation_deg: float
    numerical_margin: float


def _new_runner(map_case: str) -> SimulationRunner:
    """Use exactly the detector configuration from the deterministic runner."""
    detector = LidarProfileJunctionDetector(GeometryProfileConfig(LIDAR_MAX_RANGE))
    return SimulationRunner(
        map_case,
        "local_forward",
        profile_detector=detector,
        hold_on_profile_detection=False,
    )


def _copy_snapshot(runner: SimulationRunner, frame: int, row: dict[str, Any]) -> DisplaySnapshot:
    """Copy detector output without deriving or changing any decision field."""
    observation = runner.last_visual[0]
    scan = observation.lidar_scan
    result = runner.last_profile_result
    leader = next(
        robot for robot in runner.world.robots
        if robot.robot_id == runner.world.lidar_robot_id
    )
    return DisplaySnapshot(
        physics_frame=frame,
        timestamp=float(row["timestamp"]),
        robot_positions=np.array([robot.position.copy() for robot in runner.world.robots]),
        leader_position=leader.position.copy(),
        leader_velocity=leader.observed_velocity.copy(),
        lidar_yaw_deg=float(runner.world.lidar_yaw_deg),
        angles_deg=scan.angles_deg.copy(),
        measured_ranges=scan.ranges.copy(),
        expected_ranges=result["expected_ranges"].copy(),
        valid_mask=result["valid_angle_mask"].copy(),
        candidate_mask=result["open_candidate_mask"].copy(),
        confirmed_mask=result["confirmed_opening_mask"].copy(),
        opening_groups=tuple(dict(group) for group in result["opening_groups"]),
        detector_state=str(result["profile_detector_state"]),
        junction_detected=bool(result["profile_junction_detected"]),
        opening_candidate_count=int(result["opening_candidate_count"]),
        estimated_corridor_width=float(result["estimated_corridor_width"]),
        estimated_offset=float(result["estimated_offset"]),
        stable_orientation_deg=float(result["stable_corridor_orientation_deg"]),
        numerical_margin=float(result["profile_numerical_margin"]),
    )


class VisualizerSession:
    """Own simulation time while exposing only copied snapshots to the GUI."""

    def __init__(self, map_case: str) -> None:
        self.map_case = map_case
        self.runner = _new_runner(map_case)
        self.next_physics_frame = 0
        self.snapshots: list[DisplaySnapshot] = []
        self.view_index = -1
        self.first_detection_frame: int | None = None
        self.first_detection_time: float | None = None

    def restart(self) -> None:
        self.__init__(self.map_case)

    def advance_physics_frame(self) -> DisplaySnapshot | None:
        frame = self.next_physics_frame
        row = self.runner.step(frame)
        self.next_physics_frame += 1
        if row is None:
            return None
        snapshot = _copy_snapshot(self.runner, frame, row)
        self.snapshots.append(snapshot)
        self.view_index = len(self.snapshots) - 1
        if snapshot.junction_detected and self.first_detection_frame is None:
            self.first_detection_frame = frame
            self.first_detection_time = snapshot.timestamp
        return snapshot

    @property
    def current(self) -> DisplaySnapshot | None:
        if self.view_index < 0:
            return None
        return self.snapshots[self.view_index]

    def step_sample(self, direction: int) -> None:
        """Browse cached samples, or advance to the next detector sample."""
        if direction < 0:
            self.view_index = max(0, self.view_index - 1)
            return
        if self.view_index + 1 < len(self.snapshots):
            self.view_index += 1
            return
        start_count = len(self.snapshots)
        sample_stride = max(1, round(SAMPLE_PERIOD / DT))
        for _ in range(sample_stride + 1):
            self.advance_physics_frame()
            if len(self.snapshots) > start_count:
                break


class Renderer:
    def __init__(self, pygame: Any, geometry: Any, show_profile: bool) -> None:
        self.pygame = pygame
        self.geometry = geometry
        self.show_profile = show_profile
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Moving LiDAR Junction Detection — frozen output visualizer")
        self.font = pygame.font.Font(None, 22)
        self.small_font = pygame.font.Font(None, 19)
        self.title_font = pygame.font.Font(None, 30)
        self._configure_camera()

    def _configure_camera(self) -> None:
        vertices = np.asarray(
            [point for rect in self.geometry.free_rects for point in rect.vertices],
            dtype=float,
        )
        minimum = np.min(vertices, axis=0)
        maximum = np.max(vertices, axis=0)
        span = np.maximum(maximum - minimum, 1.0)
        x, y, width, height = MAIN_RECT
        self.camera_center = 0.5 * (minimum + maximum)
        if self.geometry.case_id == "M0_STRAIGHT":
            # Preserve useful beam scale in the long straight map.
            span[1] = min(span[1], 440.0)
        self.pixels_per_world = min((width - 40) / (span[0] + 35), (height - 40) / (span[1] + 35))
        self.main_center = np.array([x + width / 2, y + height / 2])

    def world_to_screen(self, point: np.ndarray | tuple[float, float], snapshot: DisplaySnapshot) -> tuple[int, int]:
        center = self.camera_center.copy()
        if self.geometry.case_id == "M0_STRAIGHT":
            center[1] = snapshot.leader_position[1]
        relative = np.asarray(point, dtype=float) - center
        screen = self.main_center + np.array([relative[0], -relative[1]]) * self.pixels_per_world
        return int(screen[0]), int(screen[1])

    def text(self, value: str, position: tuple[int, int], color=None, font=None) -> None:
        self.screen.blit((font or self.font).render(value, True, color or COLORS["text"]), position)

    def _draw_world(self, snapshot: DisplaySnapshot) -> None:
        pygame = self.pygame
        clip_before = self.screen.get_clip()
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
        for index in range(0, len(snapshot.angles_deg), 2):
            angle = snapshot.lidar_yaw_deg + float(snapshot.angles_deg[index])
            direction = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
            endpoint = origin + direction * float(snapshot.measured_ranges[index])
            if snapshot.confirmed_mask[index]:
                color, width = COLORS["open_beam"], 2
            elif snapshot.candidate_mask[index]:
                color, width = COLORS["candidate_beam"], 2
            else:
                color, width = COLORS["normal_beam"], 1
            pygame.draw.line(
                self.screen,
                color,
                self.world_to_screen(origin, snapshot),
                self.world_to_screen(endpoint, snapshot),
                width,
            )

        for group in snapshot.opening_groups:
            for key, color, width in (
                ("start_body_angle_deg", COLORS["group_edge"], 3),
                ("end_body_angle_deg", COLORS["group_edge"], 3),
                ("center_body_angle_deg", COLORS["group_center"], 4),
            ):
                angle = snapshot.lidar_yaw_deg + float(group[key])
                direction = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
                endpoint = origin + direction * LIDAR_MAX_RANGE
                pygame.draw.line(
                    self.screen,
                    color,
                    self.world_to_screen(origin, snapshot),
                    self.world_to_screen(endpoint, snapshot),
                    width,
                )

        for position in snapshot.robot_positions:
            pygame.draw.circle(self.screen, COLORS["robot"], self.world_to_screen(position, snapshot), 2)
        pygame.draw.circle(self.screen, COLORS["leader"], self.world_to_screen(origin, snapshot), 7)

        velocity = snapshot.leader_velocity
        speed = float(np.linalg.norm(velocity))
        if speed > 1e-9:
            arrow = origin + velocity / speed * 28.0
            pygame.draw.line(
                self.screen,
                COLORS["group_center"],
                self.world_to_screen(origin, snapshot),
                self.world_to_screen(arrow, snapshot),
                5,
            )
        self.screen.set_clip(clip_before)

    @staticmethod
    def _plot_point(rect: tuple[int, int, int, int], angle: float, value: float) -> tuple[int, int]:
        x, y, width, height = rect
        px = x + int((angle + 180.0) / 360.0 * width)
        py = y + height - int(np.clip(value / LIDAR_MAX_RANGE, 0.0, 1.0) * height)
        return px, py

    def _draw_profile(self, snapshot: DisplaySnapshot) -> None:
        pygame = self.pygame
        x, y, width, height = PROFILE_RECT
        pygame.draw.rect(self.screen, (25, 31, 40), PROFILE_RECT, border_radius=5)
        for angle in (-180, -90, 0, 90, 180):
            px, _ = self._plot_point(PROFILE_RECT, angle, 0.0)
            pygame.draw.line(self.screen, (55, 64, 76), (px, y), (px, y + height), 1)
            self.text(str(angle), (px - 14, y + height + 7), COLORS["muted"], self.small_font)
        for range_value in (0, 50, 100, 150):
            _, py = self._plot_point(PROFILE_RECT, -180.0, range_value)
            pygame.draw.line(self.screen, (55, 64, 76), (x, py), (x + width, py), 1)
            self.text(str(range_value), (x - 35, py - 8), COLORS["muted"], self.small_font)

        for index in np.flatnonzero(snapshot.confirmed_mask):
            left = self._plot_point(PROFILE_RECT, float(snapshot.angles_deg[index]) - 0.5, 0.0)[0]
            right = self._plot_point(PROFILE_RECT, float(snapshot.angles_deg[index]) + 0.5, 0.0)[0]
            overlay = pygame.Surface((max(1, right - left + 1), height), pygame.SRCALPHA)
            overlay.fill((*COLORS["open_fill"], 72))
            self.screen.blit(overlay, (left, y))

        measured_points = [
            self._plot_point(PROFILE_RECT, float(angle), float(value))
            for angle, value in zip(snapshot.angles_deg, snapshot.measured_ranges)
        ]
        expected_points = [
            self._plot_point(PROFILE_RECT, float(angle), float(value))
            for angle, value in zip(snapshot.angles_deg, snapshot.expected_ranges)
        ]
        # This is only the rendered boundary. The detector's candidate mask is
        # copied above and remains the sole source of every OPEN decision.
        threshold_points = [
            self._plot_point(
                PROFILE_RECT,
                float(angle),
                float(min(LIDAR_MAX_RANGE, expected + snapshot.numerical_margin)),
            )
            for angle, expected in zip(snapshot.angles_deg, snapshot.expected_ranges)
        ]
        pygame.draw.lines(self.screen, COLORS["measured"], False, measured_points, 2)
        pygame.draw.lines(self.screen, COLORS["expected"], False, expected_points, 2)
        pygame.draw.lines(self.screen, COLORS["threshold"], False, threshold_points, 1)
        pygame.draw.rect(self.screen, COLORS["muted"], PROFILE_RECT, 1, border_radius=5)
        self.text("range", (x - 35, y - 24), COLORS["muted"], self.small_font)
        self.text("LiDAR angle theta [deg]", (x + width // 2 - 76, y + height + 28), COLORS["muted"], self.small_font)

    def draw(
        self,
        session: VisualizerSession,
        paused: bool,
        auto_pause: bool,
    ) -> None:
        snapshot = session.current
        self.screen.fill(COLORS["background"])
        self.text("Moving LiDAR Junction Detection", (18, 15), font=self.title_font)
        self.text("read-only consumer of frozen detector output", (365, 22), COLORS["muted"], self.small_font)
        if snapshot is None:
            self.text("Waiting for first sampled LiDAR scan...", (44, 90))
            self.pygame.display.flip()
            return

        self._draw_world(snapshot)
        if self.show_profile:
            self._draw_profile(snapshot)

        panel_x = 872
        info_y = 530 if self.show_profile else 82
        info_step = 20
        state_color = COLORS["detected"] if snapshot.junction_detected else COLORS["clear"]
        run_state = "PAUSED" if paused else "RUNNING"
        lines = [
            (f"{run_state} | map case: {session.map_case}", COLORS["text"]),
            (f"physics frame: {snapshot.physics_frame}   t: {snapshot.timestamp:.6f} s", COLORS["text"]),
            (f"W_hat: {_finite(snapshot.estimated_corridor_width)}   offset: {_finite(snapshot.estimated_offset, signed=True)}", COLORS["text"]),
            (f"corridor orientation: {_finite(snapshot.stable_orientation_deg, signed=True)} deg", COLORS["text"]),
            (f"state: {snapshot.detector_state}", state_color),
            (f"JUNCTION_DETECTED = {snapshot.junction_detected}", state_color),
            (f"OPEN candidates: {snapshot.opening_candidate_count}", COLORS["text"]),
            (f"opening group count: {len(snapshot.opening_groups)}", COLORS["text"]),
            (f"first detection: {_detection_text(session)}", COLORS["text"]),
        ]
        for index, (value, color) in enumerate(lines):
            self.text(value, (panel_x, info_y + index * info_step), color)

        group_y = info_y + len(lines) * info_step + 7
        self.text("Opening groups (detector output)", (panel_x, group_y), COLORS["group_center"])
        if not snapshot.opening_groups:
            self.text("none", (panel_x, group_y + 24), COLORS["muted"])
        for index, group in enumerate(snapshot.opening_groups[:4]):
            base = group_y + 24 + index * 43
            self.text(
                f"#{group['group_id'] + 1} start={group['start_angle_deg']:+.1f}  end={group['end_angle_deg']:+.1f} deg",
                (panel_x, base),
                COLORS["text"],
                self.small_font,
            )
            self.text(
                f"center={group['center_angle_deg']:+.1f}  angular width={group['angular_width_deg']:.1f} deg",
                (panel_x, base + 19),
                COLORS["muted"],
                self.small_font,
            )

        formula_y = min(817, group_y + 24 + max(1, len(snapshot.opening_groups[:4])) * 43 + 8)
        self.text("simple baseline reference only:", (panel_x, formula_y), COLORS["muted"], self.small_font)
        self.text("expected_wall(theta) = (l/2) / |sin(theta)|", (panel_x, formula_y + 19), COLORS["muted"], self.small_font)
        self.text(
            "Rendered profile uses detector expected_ranges (offset/orientation corrected).",
            (panel_x, formula_y + 38),
            COLORS["muted"],
            self.small_font,
        )

        footer = "SPACE pause/resume   R restart   LEFT/RIGHT sampled-frame step   P profile   ESC quit"
        if auto_pause:
            footer += "   pause-on-detect enabled"
        self.text(footer, (22, 875), COLORS["muted"], self.small_font)
        self.pygame.display.flip()


def _finite(value: float, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "UNINITIALIZED"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _detection_text(session: VisualizerSession) -> str:
    if session.first_detection_frame is None:
        return "none"
    return f"frame {session.first_detection_frame}, t={session.first_detection_time:.6f} s"


def run_headless(map_case: str, frames: int) -> dict[str, Any]:
    """Exercise the visualizer data path without opening a window or writing files."""
    session = VisualizerSession(map_case)
    for _ in range(frames):
        session.advance_physics_frame()
    detections = [snapshot for snapshot in session.snapshots if snapshot.junction_detected]
    return {
        "map_case": map_case,
        "physics_frames": frames,
        "sample_count": len(session.snapshots),
        "detection_count": len(detections),
        "first_detection_frame": session.first_detection_frame,
        "first_detection_time": session.first_detection_time,
        "max_opening_group_count": max((len(item.opening_groups) for item in session.snapshots), default=0),
    }


def run_gui(args: argparse.Namespace) -> None:
    import pygame

    pygame.init()
    session = VisualizerSession(args.map_case)
    renderer = Renderer(pygame, session.runner.geometry, args.show_profile)
    clock = pygame.time.Clock()
    paused = bool(args.start_paused)
    detection_pause_consumed = False
    running = True

    # Give a start-paused session a real first detector sample to inspect.
    first = session.advance_physics_frame()
    if args.pause_on_detect and first is not None and first.junction_detected:
        paused = True
        detection_pause_consumed = True

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
                    detection_pause_consumed = False
                    first = session.advance_physics_frame()
                    if args.pause_on_detect and first is not None and first.junction_detected:
                        paused = True
                        detection_pause_consumed = True
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
                and not detection_pause_consumed
                and snapshot is not None
                and snapshot.junction_detected
            ):
                paused = True
                detection_pause_consumed = True
        renderer.draw(session, paused, args.pause_on_detect)
        if args.frames > 0 and session.next_physics_frame >= args.frames:
            running = False
        clock.tick(args.fps)
    pygame.quit()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-case", choices=MAP_CASES, default="M1_CROSS_BASELINE")
    parser.add_argument("--pause-on-detect", action="store_true")
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--show-profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frames", type=int, default=0, help="auto-exit after N physics frames; 0 runs until ESC")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--headless", action="store_true", help="validate the visualizer data path without opening Pygame")
    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error("--frames must be non-negative")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.headless and args.frames == 0:
        args.frames = 600
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.headless:
        result = run_headless(args.map_case, args.frames)
        print(" ".join(f"{key}={value}" for key, value in result.items()))
        return
    run_gui(args)


if __name__ == "__main__":
    main()
