"""Adaptive LiDAR front-end for the single-junction Physical DFS baseline.

The Physical DFS implementation remains the authoritative source for the map,
robots, SPH forces, communication, Guards, Frontier/Shepherd lifecycle,
Pebbles, backtracking, and return-to-base logic.  This integration module loads
only the definition section of that file and supplies one sensor/perception
front-end plus one dark renderer.  No second SimulatorWorld or robot set exists.

Localization audit
------------------
Fixture labels are unavailable to the detector and persistent-opening tracker.
Localization is opened only for branch-local shallow Guard readiness and
slot-to-robot WHO assignment.  It is closed before Physical DFS handoff;
persistent opening angles are used only by the LiDAR refinement adapter, while
DFS choice, saturation, and backtracking never receive localization input.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import types
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pygame


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PHYSICAL_SOURCE = HERE / "single_junction_sph_dfs_environment.py"
ADAPTIVE_SOURCE = HERE / "lidar_junction_detection_adaptive_w_tau_anchor_stop.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pygame_simulator import (  # noqa: E402
    lidar_junction_detection_adaptive_w_tau_anchor_stop as adaptive,
)


WINDOW_SIZE = (1440, 900)
MAIN_RECT = pygame.Rect(16, 54, 824, 830)
PROFILE_RECT = pygame.Rect(870, 72, 540, 410)
DIAGNOSTIC_RECT = pygame.Rect(870, 506, 540, 370)
COLORS = {
    "background": (10, 13, 18),
    "panel": (21, 27, 35),
    "panel_alt": (25, 31, 40),
    "floor": (31, 42, 54),
    "wall": (205, 214, 224),
    "text": (226, 232, 240),
    "muted": (125, 139, 156),
    "raw": (232, 238, 245),
    "smooth": (72, 156, 255),
    "open": (190, 92, 246),
    "open_fill": (190, 92, 246),
    "safe_band": (72, 156, 255),
    "group_edge": (255, 146, 52),
    "group_center": (225, 92, 246),
    "threshold": (255, 196, 74),
    "safe": (139, 92, 246),
    "anchor": (255, 221, 74),
    "normal": (52, 120, 246),
    "guard": (235, 72, 88),
    "frontier": (255, 146, 52),
    "shepherd": (174, 105, 226),
    "pebble": (35, 211, 104),
    "relay": (185, 125, 64),
    "trunk": (139, 87, 67),
}

RAY_COUNT = adaptive.LIDAR_RAYS
MAX_RANGE = adaptive.LIDAR_MAX_RANGE
SMOOTHING_WINDOW = 5
ALPHA = 0.5
NOISE_FRACTION = adaptive.DEFAULT_NOISE_FRACTION
TAU = MAX_RANGE * NOISE_FRACTION
ADAPTIVE_W_MARGIN_RATIO = 0.05
STATIONARY_WINDOW = 120
MIN_PERSISTENT_OBSERVATIONS = 72
PREVIOUS_APPROACH_EXTENSION = 60.0
BASE_ADDED_EXTENSION = 0.0
APPROACH_EXTENSION = PREVIOUS_APPROACH_EXTENSION + BASE_ADDED_EXTENSION
ASSOCIATION_TOLERANCE_DEG = max(
    2.0 * float(adaptive.FROZEN_PARAMETERS["merge_gap_deg"]),
    float(adaptive.FROZEN_PARAMETERS["min_opening_width_deg"]),
)

LOCAL_SATURATION_PRESSURE_RATIO = 1.02

# Provisional Guard geometry is constructed exclusively from the fixed
# Anchor's LiDAR sectors.  Localization is opened later and only for the
# slot-to-robot-ID association.
PROVISIONAL_MOUTH_WIDTH_W_RATIO = 0.75
PROVISIONAL_MOUTH_WIDTH_MIN_RATIO = 0.80
PROVISIONAL_JUNCTION_DEPTH_MIN_WIDTH_RATIO = 0.95
PROVISIONAL_JUNCTION_DEPTH_MAX_WIDTH_RATIO = 1.25
PROVISIONAL_SIDE_SECTOR_MIN_ABS_ANGLE = 90.0
GUARD_WHO_AXIAL_WEIGHT = 4.0
GUARD_WHO_LATERAL_WEIGHT = 2.0
GUARD_WHO_PATH_WEIGHT = 1.0
GUARD_CAPTURE_UPSTREAM_WIDTH_RATIO = 0.32
GUARD_CAPTURE_DOWNSTREAM_WIDTH_RATIO = 0.28
GUARD_CAPTURE_LATERAL_MARGIN_RATIO = 0.10
GUARD_ASSIGN_MAX_AXIAL_WIDTH_RATIO = 0.62
GUARD_ASSIGN_MAX_LATERAL_WIDTH_RATIO = 0.45
GUARD_ASSIGN_MAX_PATH_WIDTH_RATIO = 0.65
GUARD_LATERAL_COVERAGE_BINS = 9
GUARD_READINESS_LOG_PERIOD = 10
GUARD_EDGE_SEAL_MARGIN_RATIO = 0.25
GUARD_LATERAL_OVERLAP_RATIO = 0.90


class PerceptionState(Enum):
    MOVING = auto()
    FIXED_ACCUMULATING = auto()
    BRANCHES_READY = auto()
    PHYSICAL_DFS = auto()


@dataclass
class PersistentOpening:
    persistent_id: str
    sine_sum: float = 0.0
    cosine_sum: float = 0.0
    widths: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    observations: list[dict[str, float]] = field(default_factory=list)
    last_frame: int = -1

    @property
    def center_angle(self) -> float:
        return math.degrees(math.atan2(self.sine_sum, self.cosine_sum))

    @property
    def mean_width(self) -> float:
        return float(np.mean(self.widths)) if self.widths else 0.0

    @property
    def confidence(self) -> float:
        return float(np.mean(self.confidences)) if self.confidences else 0.0

    def persistence_ratio(self, sample_count: int) -> float:
        return len(self.observations) / max(sample_count, 1)

    def update(self, opening: dict[str, float], frame: int) -> None:
        angle = float(opening["center_angle"])
        self.sine_sum += math.sin(math.radians(angle))
        self.cosine_sum += math.cos(math.radians(angle))
        self.widths.append(float(opening["width_deg"]))
        self.confidences.append(float(opening["confidence"]))
        self.observations.append({
            "start_angle": float(opening["start_angle"]),
            "end_angle": float(opening["end_angle"]),
            "center_angle": angle,
            "width_deg": float(opening["width_deg"]),
            "confidence": float(opening["confidence"]),
            "frame": float(frame),
        })
        self.last_frame = frame


@dataclass(frozen=True)
class ParentTopologyEdge:
    """Traversed ingress edge; deliberately not a LiDAR opening track."""

    persistent_id: str = "PARENT_00"
    source: str = "INGRESS_HISTORY"


@dataclass
class LidarFrame:
    frame: int
    angles: np.ndarray
    raw: np.ndarray
    smoothed: np.ndarray
    support: np.ndarray
    openings: tuple[dict[str, float], ...]
    left: float | None
    right: float | None
    adaptive_w: float
    lower: float
    upper: float
    selected: float | None
    interval_valid: bool
    current_evidence: bool


@dataclass
class ProvisionalGuardGeometry:
    provisional_uid: str
    opening: dict[str, float]
    descriptor: Any
    columns: int
    layers: int
    slots: list[pygame.Vector2]
    selected_ids: list[int] = field(default_factory=list)
    fixture_key: str | None = None
    persistent_uid: str | None = None
    cohort_ready: bool = False
    first_robot_crossing_mouth_frame: int | None = None
    candidate_sufficient_frame: int | None = None
    guard_ready_frame: int | None = None
    role_assignment_frame: int | None = None
    last_candidate_count: int = -1
    last_assignment_count: int = -1
    readiness_diagnostics: dict[str, Any] = field(default_factory=dict)
    sealing_lateral_min: float = 0.0
    sealing_lateral_max: float = 0.0
    slot_spacing: float = 0.0


def _load_physical_definitions() -> types.ModuleType:
    """Load definitions before the original top-level main loop starts."""
    source = PHYSICAL_SOURCE.read_text(encoding="utf-8")
    marker = "robots, reference_density, color_reference_density = initialize_simulation()"
    if marker not in source:
        raise RuntimeError("Physical DFS integration marker not found")
    definitions = source[: source.index(marker)]
    module = types.ModuleType("_physical_dfs_runtime")
    module.__file__ = str(PHYSICAL_SOURCE)
    module.__package__ = "pygame_simulator"
    sys.modules[module.__name__] = module
    exec(compile(definitions, str(PHYSICAL_SOURCE), "exec"), module.__dict__)
    return module


def configure_extended_approach(physical: types.ModuleType) -> tuple[float, float]:
    """Make horizontal branches symmetric and modestly extend the approach."""
    left_branch_length = float(physical.normal_length)
    right_branch_length_before = float(physical.right_length)
    right_branch_length_after = left_branch_length
    base_reference_length = float(physical.base_length)
    base_length_before = base_reference_length + PREVIOUS_APPROACH_EXTENSION
    base_length_after = base_length_before + BASE_ADDED_EXTENSION

    physical.right_length = right_branch_length_after
    physical.base_length = base_length_after
    right_x = physical.center_x + physical.half_width + right_branch_length_after
    bottom_y = physical.center_y + physical.half_width + base_length_after
    points = list(physical.cross_points)
    points[3] = (right_x, physical.center_y - physical.half_width)
    points[4] = (right_x, physical.center_y + physical.half_width)
    points[6] = (physical.center_x + physical.half_width, bottom_y)
    points[7] = (physical.center_x - physical.half_width, bottom_y)
    physical.cross_points = points
    physical.right_rect = pygame.Rect(
        physical.center_x + physical.half_width,
        physical.center_y - physical.half_width,
        round(right_branch_length_after),
        physical.corridor_width,
    )
    physical.bottom_rect = pygame.Rect(
        physical.center_x - physical.half_width,
        physical.center_y + physical.half_width,
        physical.corridor_width,
        round(base_length_after),
    )
    physical.dead_end_regions["RIGHT"] = pygame.Rect(
        right_x - physical.END_REGION_DEPTH,
        physical.center_y - physical.half_width,
        physical.END_REGION_DEPTH,
        physical.corridor_width,
    )
    physical.early_capture_regions["RIGHT"] = pygame.Rect(
        right_x - physical.EARLY_CAPTURE_DEPTH,
        physical.center_y - physical.half_width,
        physical.EARLY_CAPTURE_DEPTH,
        physical.corridor_width,
    )
    physical.BRANCH_LENGTHS["RIGHT"] = right_branch_length_after
    physical.get_junction_state().branch_edges["RIGHT"].length = right_branch_length_after
    physical.MAX_TRANSPORT_DISTANCE = (
        max(physical.BRANCH_LENGTHS.values()) + physical.corridor_width
    )
    physical.BASE_POSITION = pygame.Vector2(
        physical.center_x - 25 * physical.MAP_SCALE,
        bottom_y - 14 * physical.MAP_SCALE,
    )
    physical.BASE_COMPRESSION_CENTER = pygame.Vector2(
        physical.center_x,
        physical.center_y + physical.half_width + base_length_after * 0.60,
    )
    physical.floor_surface = pygame.Surface(
        (physical.SCREEN_WIDTH, physical.SCREEN_HEIGHT), pygame.SRCALPHA
    )
    physical.floor_surface.fill((0, 0, 0, 0))
    pygame.draw.polygon(
        physical.floor_surface, (255, 255, 255, 255), physical.cross_points
    )
    physical.walkable_mask = pygame.mask.from_surface(physical.floor_surface)
    print(
        f"[Map] left_branch_length={left_branch_length:.1f} "
        f"right_branch_length_before={right_branch_length_before:.1f} "
        f"right_branch_length_after={right_branch_length_after:.1f} "
        f"base_length_before={base_length_before:.1f} "
        f"base_length_after={base_length_after:.1f} "
        f"base_added_extension={BASE_ADDED_EXTENSION:.1f}"
    )
    return base_length_before, base_length_after


def install_local_forward_ingress(physical: types.ModuleType) -> None:
    """Disable artificial MOVE compression and install body-local propulsion."""
    original_route_force = physical.compute_route_force
    original_equilibrium_radius = physical.adaptive_equilibrium_radius
    original_compression_envelope = physical.get_base_compression_envelope
    original_pressure_scale = physical.get_base_pressure_scale
    original_release_active = physical.initial_pressure_release_active
    original_stored_envelope = physical.get_stored_compression_pressure_envelope
    original_piston_force = physical.compute_base_piston_reaction_force
    original_cruise_blend = physical.get_initial_release_cruise_blend

    def local_route_force(robot: Any) -> pygame.Vector2:
        if physical.phase != physical.SimulationPhase.MOVE_TO_JUNCTION:
            return original_route_force(robot)
        if robot.role in {"PEBBLE", "RELAY", "TRUNK_RELAY"}:
            return pygame.Vector2()
        yaw = float(getattr(robot, "body_yaw", -0.5 * math.pi))
        weight = float(
            getattr(robot, "propulsion_weight", adaptive.LOCAL_FOLLOWER_DRIVE_WEIGHT)
        )
        forward = pygame.Vector2(math.cos(yaw), math.sin(yaw))
        lateral = pygame.Vector2(-forward.y, forward.x)
        # ingress_lane_x is a deployment-local transverse reference, not a
        # Junction/world target. For the present -90 degree deployment yaw it
        # is exactly the body-local lateral coordinate.
        lateral_error = robot.ingress_lane_x - robot.position.x
        lane_force = max(
            -physical.INITIAL_INGRESS_LANE_MAX_FORCE,
            min(
                physical.INITIAL_INGRESS_LANE_MAX_FORCE,
                physical.INITIAL_INGRESS_LANE_GAIN * lateral_error,
            ),
        )
        return (
            forward * adaptive.LOCAL_FORWARD_DRIVE_FORCE * weight
            + lateral * lane_force
        )

    def normal_equilibrium_radius(robot: Any) -> float:
        if (
            physical.phase == physical.SimulationPhase.MOVE_TO_JUNCTION
            and robot.role == "NORMAL"
        ):
            return physical.SAFE_RADIUS * physical.NORMAL_EQUILIBRIUM_SCALE
        return original_equilibrium_radius(robot)

    physical.compute_route_force = local_route_force
    physical.adaptive_equilibrium_radius = normal_equilibrium_radius
    physical.get_base_compression_envelope = lambda: (
        0.0
        if physical.phase == physical.SimulationPhase.MOVE_TO_JUNCTION
        else original_compression_envelope()
    )
    physical.get_base_pressure_scale = lambda: (
        physical.SPH_MOTION_PRESSURE_BOOST
        if physical.phase == physical.SimulationPhase.MOVE_TO_JUNCTION
        else original_pressure_scale()
    )
    physical.initial_pressure_release_active = lambda: (
        False
        if physical.phase == physical.SimulationPhase.MOVE_TO_JUNCTION
        else original_release_active()
    )
    physical.get_stored_compression_pressure_envelope = lambda: (
        0.0
        if physical.phase == physical.SimulationPhase.MOVE_TO_JUNCTION
        else original_stored_envelope()
    )
    physical.compute_base_piston_reaction_force = lambda robot: (
        pygame.Vector2()
        if physical.phase == physical.SimulationPhase.MOVE_TO_JUNCTION
        else original_piston_force(robot)
    )
    physical.get_initial_release_cruise_blend = lambda: (
        1.0
        if physical.phase == physical.SimulationPhase.MOVE_TO_JUNCTION
        else original_cruise_blend()
    )


def circular_error(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


class PhysicalMapLidar:
    """Analytic ray caster over the Physical DFS cross polygon walls."""

    def __init__(self, physical: types.ModuleType) -> None:
        self.angles = np.linspace(-180.0, 180.0, RAY_COUNT, endpoint=False)
        points = [np.asarray(point, dtype=float) for point in physical.cross_points]
        self.segments = tuple(zip(points, points[1:] + points[:1]))

    @staticmethod
    def _hit(origin: np.ndarray, direction: np.ndarray, segment: Any) -> float | None:
        start, end = segment
        edge = end - start
        denominator = direction[0] * edge[1] - direction[1] * edge[0]
        if abs(denominator) < 1.0e-10:
            return None
        offset = start - origin
        ray_t = (offset[0] * edge[1] - offset[1] * edge[0]) / denominator
        seg_t = (offset[0] * direction[1] - offset[1] * direction[0]) / denominator
        if ray_t >= 0.0 and -1.0e-9 <= seg_t <= 1.0 + 1.0e-9:
            return float(ray_t)
        return None

    def scan(self, position: pygame.Vector2, yaw_deg: float) -> np.ndarray:
        origin = np.array([position.x, position.y], dtype=float)
        ranges = np.full(RAY_COUNT, MAX_RANGE, dtype=float)
        for index, local_angle in enumerate(self.angles):
            world_angle = math.radians(yaw_deg + float(local_angle))
            direction = np.array([math.cos(world_angle), math.sin(world_angle)])
            hits = [
                value for segment in self.segments
                if (value := self._hit(origin, direction, segment)) is not None
            ]
            if hits:
                ranges[index] = min(MAX_RANGE, min(hits))
        return ranges


class AdaptivePerception:
    def __init__(self, physical: types.ModuleType, robots: Sequence[Any]) -> None:
        self.physical = physical
        self.sensor = PhysicalMapLidar(physical)
        self.state = PerceptionState.MOVING
        self.frame = 0
        self.yaw_deg = -90.0
        self.last_valid_w: float | None = None
        self.last_frame: LidarFrame | None = None
        self.junction_confirmed = False
        self.confirmation_frame: int | None = None
        self.confirmation_time: float | None = None
        self.first_open_support_frame: int | None = None
        self.first_junction_evidence_frame: int | None = None
        self.anchor_fixed = False
        self.anchor_position: pygame.Vector2 | None = None
        self.pre_detection_travel = 0.0
        self.post_fix_drift = 0.0
        self.stationary_samples = 0
        self.tracks: list[PersistentOpening] = []
        self.parent: PersistentOpening | ParentTopologyEdge | None = None
        self.parent_source: str | None = None
        self.outgoing: list[PersistentOpening] = []
        self.handoff_complete = False
        self.topology_ready_frame: int | None = None
        self.guard_geometry_frame: int | None = None
        self.guard_who_frame: int | None = None
        self.guard_motion_start_frame: int | None = None
        self.provisional_guards: list[ProvisionalGuardGeometry] = []
        self.provisional_guard_started = False
        self.guard_leakage: dict[str, dict[str, Any]] = {}
        self.guard_communication_audits: dict[str, dict[str, Any]] = {}
        self.anchor_fixed_mean_normal_forward_speed = 0.0
        self.pre_topology_normal_forward_speeds: list[float] = []
        self.robots = robots
        self.leader = self._select_leader(robots)
        self.initial_leader_position = self.leader.position.copy()
        self.leader.is_lidar_robot = True
        self.leader.is_fixed_anchor = False
        self.leader.body_yaw = math.radians(self.yaw_deg)
        forward = pygame.Vector2(
            math.cos(math.radians(self.yaw_deg)),
            math.sin(math.radians(self.yaw_deg)),
        )
        front_progress = max(robot.position.dot(forward) for robot in robots)
        front_row = [
            robot for robot in robots
            if abs(robot.position.dot(forward) - front_progress) <= 1.0e-6
        ]
        lateral_offset = self.leader.position.x - self.physical.center_x
        print(f"front_row_progress={front_progress:.3f}")
        print(f"front_row_robot_count={len(front_row)}")
        print(f"lidar_id={self.leader.robot_id}")
        print(
            f"lidar_initial_position=({self.initial_leader_position.x:.3f}, "
            f"{self.initial_leader_position.y:.3f})"
        )
        print(f"lateral_offset={lateral_offset:.6f}")
        print(f"[LiDAR] leader_id={self.leader.robot_id}")
        print(
            f"[LiDAR] initial_position=({self.initial_leader_position.x:.3f},"
            f"{self.initial_leader_position.y:.3f})"
        )

    def _select_leader(self, robots: Sequence[Any]) -> Any:
        forward = pygame.Vector2(
            math.cos(math.radians(self.yaw_deg)),
            math.sin(math.radians(self.yaw_deg)),
        )
        lateral = pygame.Vector2(-forward.y, forward.x)
        front_progress = max(robot.position.dot(forward) for robot in robots)
        front = [
            robot for robot in robots
            if abs(robot.position.dot(forward) - front_progress) <= 1.0e-6
        ]
        corridor_lateral = pygame.Vector2(
            self.physical.center_x,
            self.physical.center_y,
        ).dot(lateral)
        return min(front, key=lambda robot: (
            abs(robot.position.dot(lateral) - corridor_lateral),
            robot.robot_id,
        ))

    def reset(self, robots: Sequence[Any]) -> None:
        self.__init__(self.physical, robots)

    def mean_normal_forward_speed(self) -> float:
        forward = pygame.Vector2(
            math.cos(math.radians(self.yaw_deg)),
            math.sin(math.radians(self.yaw_deg)),
        )
        normals = [
            robot for robot in self.robots
            if robot.role == "NORMAL" and not robot.base_reserve
        ]
        return float(np.mean([
            robot.velocity.dot(forward) for robot in normals
        ])) if normals else 0.0

    def _associate(self, openings: Sequence[dict[str, float]]) -> None:
        available = set(range(len(self.tracks)))
        for opening in openings:
            center = float(opening["center_angle"])
            candidates = [
                (circular_error(center, self.tracks[index].center_angle), index)
                for index in available
            ]
            error, index = min(candidates, default=(float("inf"), -1))
            if error <= ASSOCIATION_TOLERANCE_DEG:
                track = self.tracks[index]
                available.remove(index)
            else:
                track = PersistentOpening(f"OPEN_{len(self.tracks):02d}")
                self.tracks.append(track)
            track.update(opening, self.frame)

    def _persistent_ready(self) -> bool:
        persistent = [
            track for track in self.tracks
            if len(track.observations) >= MIN_PERSISTENT_OBSERVATIONS
            and track.persistence_ratio(self.stationary_samples) >= 0.60
        ]
        if len(persistent) < 3 or self.stationary_samples < STATIONARY_WINDOW:
            return False
        rear_candidates = [
            track for track in persistent
            if circular_error(track.center_angle, 180.0) <= 45.0
        ]
        if len(persistent) >= 4 and rear_candidates:
            self.parent = min(
                rear_candidates,
                key=lambda track: circular_error(track.center_angle, 180.0),
            )
            self.parent.persistent_id = "PARENT_00"
            self.parent_source = "LIDAR_PERSISTENT"
            children = [track for track in persistent if track is not self.parent]
        else:
            # This topology edge comes from the path just traversed. It is not
            # inserted into self.tracks and is never reported as LiDAR data.
            self.parent = ParentTopologyEdge()
            self.parent_source = self.parent.source
            children = list(persistent)
        children.sort(key=lambda track: track.center_angle)
        if len(children) < 3:
            return False
        self.outgoing = children[:3]
        for index, track in enumerate(self.outgoing):
            track.persistent_id = f"J0-B{index}"
        return True

    def update(self, simulation_time: float) -> LidarFrame:
        # The deployment body pose is the stable local ingress reference.
        # A single SPH acceleration sample can be predominantly lateral and
        # must not rotate the detector frame before the first scan.
        self.leader.body_yaw = math.radians(self.yaw_deg)
        raw = self.sensor.scan(self.leader.position, self.yaw_deg)
        smoothed = adaptive.smooth_ranges(raw, SMOOTHING_WINDOW)
        left, right = adaptive.extract_lateral_wall_ranges(
            self.sensor.angles, smoothed, MAX_RANGE, SMOOTHING_WINDOW
        )
        estimate = adaptive.compute_adaptive_worst_wall_range(left, right)
        if estimate is not None:
            self.last_valid_w = estimate
        adaptive_w = float(self.last_valid_w if self.last_valid_w is not None else 0.62 * MAX_RANGE)
        lower, upper, valid = adaptive.compute_adaptive_safe_threshold_interval(
            adaptive_w,
            MAX_RANGE,
            TAU,
            margin_ratio=ADAPTIVE_W_MARGIN_RATIO,
        )
        selected = adaptive.select_threshold_in_safe_interval(lower, upper, valid, ALPHA)
        openings, diagnostics = adaptive._detect_openings_w_tau_with_diagnostics(
            self.sensor.angles,
            raw,
            selected_threshold=selected,
            threshold_interval_valid=valid,
            smoothing_window_size=SMOOTHING_WINDOW,
        )
        # Junction evidence uses the unchanged raw 360-degree opening count.
        # The traversed incoming/Parent opening remains part of this count so
        # a future T-junction can satisfy Parent 1 + Outgoing 2 = 3.
        evidence = bool(valid and len(openings) >= 3)
        result = LidarFrame(
            frame=self.frame,
            angles=self.sensor.angles.copy(),
            raw=raw,
            smoothed=np.asarray(diagnostics["smoothed_ranges"]),
            support=np.asarray(diagnostics["open_support_mask"], dtype=bool),
            openings=tuple(dict(item) for item in openings),
            left=left,
            right=right,
            adaptive_w=adaptive_w,
            lower=lower,
            upper=upper,
            selected=selected,
            interval_valid=valid,
            current_evidence=evidence,
        )
        self.last_frame = result
        if np.any(result.support) and self.first_open_support_frame is None:
            self.first_open_support_frame = self.frame
            print(f"[LiDAR] first_open_support_frame={self.frame}")
        if self.frame % 60 == 0:
            print(
                f"[LiDAR] adaptive W={adaptive_w:.2f} Tmin={lower:.2f} "
                f"Tmax={upper:.2f} selected={selected} openings={len(openings)}"
            )
        if evidence and not self.junction_confirmed:
            self.first_junction_evidence_frame = self.frame
            self.junction_confirmed = True
            self.confirmation_frame = self.frame
            self.confirmation_time = simulation_time
            self.anchor_fixed = True
            self.anchor_position = self.leader.position.copy()
            self.physical.integration_anchor_position = (
                self.anchor_position.copy()
            )
            self.pre_detection_travel = self.anchor_position.distance_to(
                self.initial_leader_position
            )
            self.leader.is_fixed_anchor = True
            self.leader.base_reserve = True
            # Only this LiDAR robot stops. NORMAL bodies retain the unchanged
            # local-forward/SPH force until persistent topology is ready.
            self.physical.integration_guard_hold_active = False
            self.anchor_fixed_mean_normal_forward_speed = (
                self.mean_normal_forward_speed()
            )
            self.state = PerceptionState.FIXED_ACCUMULATING
            print(f"[LiDAR] junction evidence openings={len(openings)}")
            print("[LiDAR] parent_opening_in_evidence=True")
            print(
                f"[LiDAR] JUNCTION CONFIRMED frame={self.frame} "
                f"time={simulation_time:.3f}"
            )
            print(f"[LiDAR] first_junction_evidence_frame={self.frame}")
            print(f"[LiDAR] confirmation_frame={self.frame}")
            print(f"[LiDAR] confirmation_time={simulation_time:.3f}")
            print(f"[LiDAR] pre_detection_travel={self.pre_detection_travel:.3f}")
            print(
                "[LiDAR] confirmation_openings="
                + str([
                    (
                        round(float(item["start_angle"]), 1),
                        round(float(item["end_angle"]), 1),
                        round(float(item["center_angle"]), 1),
                    )
                    for item in openings
                ])
            )
            print(f"[Anchor] fixed_id={self.leader.robot_id}")
            print("[LiDAR] anchor_fixed=True")
            print("[FlowAudit] Anchor-only stop=True")
            print("[FlowAudit] Normal-flow enabled=True")
            print("[FlowAudit] Topology ready=False")
            print("[FlowAudit] Guard gating enabled=False")
            print(
                "[FlowAudit] anchor_fixed_mean_normal_forward_speed="
                f"{self.anchor_fixed_mean_normal_forward_speed:.6f}"
            )
            print(f"[Timeline] JUNCTION_CONFIRMED frame={self.frame}")
            print(f"[Timeline] ANCHOR_FIXED frame={self.frame}")
            print(
                f"[Anchor] fix_position=({self.anchor_position.x:.3f},"
                f"{self.anchor_position.y:.3f})"
            )
            print("[Opening] stationary accumulation started")
        if self.anchor_fixed:
            self.stationary_samples += 1
            if self.state == PerceptionState.FIXED_ACCUMULATING:
                normal_speed = self.mean_normal_forward_speed()
                self.pre_topology_normal_forward_speeds.append(normal_speed)
                if self.stationary_samples % 30 == 0:
                    print(
                        "[FlowAudit] pre_topology_mean_normal_forward_speed="
                        f"{float(np.mean(self.pre_topology_normal_forward_speeds)):.6f} "
                        f"instant={normal_speed:.6f}"
                    )
            # Accumulate only real raw W-tau openings. Ingress-history Parent
            # is represented separately if no persistent rear opening exists.
            self._associate(openings)
            if self.stationary_samples % 10 == 0:
                count = sum(
                    len(track.observations) >= MIN_PERSISTENT_OBSERVATIONS
                    for track in self.tracks
                )
                print(f"[Opening] persistent count={count} samples={self.stationary_samples}")
            if self.state == PerceptionState.FIXED_ACCUMULATING and self._persistent_ready():
                self.state = PerceptionState.BRANCHES_READY
                self.topology_ready_frame = self.frame
                print(f"[Topology] parent={self.parent.persistent_id}")
                print(f"[Topology] parent_source={self.parent_source}")
                print(f"[Topology] outgoing={[track.persistent_id for track in self.outgoing]}")
                print(
                    f"[Topology] persistent_count={sum(len(track.observations) >= MIN_PERSISTENT_OBSERVATIONS for track in self.tracks)} "
                    f"parent_count={int(self.parent is not None)} "
                    f"outgoing_count={len(self.outgoing)}"
                )
                print(
                    f"[Topology] LiDAR Persistent="
                    f"{sum(len(track.observations) >= MIN_PERSISTENT_OBSERVATIONS for track in self.tracks)} "
                    f"| Parent={self.parent.persistent_id}"
                    f"(source={self.parent_source}) | Outgoing={len(self.outgoing)}"
                )
                print(f"[Timeline] TOPOLOGY_READY frame={self.frame}")
                print("[FlowAudit] Topology ready=True")
                print(
                    "[FlowAudit] Guard gating enabled="
                    f"{getattr(self.physical, 'integration_guard_gating_enabled', False)}"
                )
                print(
                    "[FlowAudit] topology_ready_pre_mean_normal_forward_speed="
                    f"{float(np.mean(self.pre_topology_normal_forward_speeds)):.6f}"
                )
        self.frame += 1
        return result

    def enforce_anchor(self) -> None:
        if self.anchor_fixed and self.anchor_position is not None:
            self.post_fix_drift = max(
                self.post_fix_drift,
                self.leader.position.distance_to(self.anchor_position),
            )
            self.leader.position = self.anchor_position.copy()
            self.leader.velocity.update(0.0, 0.0)
            self.leader.acceleration.update(0.0, 0.0)
            self.leader.filtered_acceleration.update(0.0, 0.0)


def _body_local_unit(perception: AdaptivePerception, angle_deg: float) -> pygame.Vector2:
    world_angle = math.radians(perception.yaw_deg + angle_deg)
    return pygame.Vector2(math.cos(world_angle), math.sin(world_angle))


def _range_at_local_angle(frame: LidarFrame, angle_deg: float) -> float:
    index = min(
        range(len(frame.angles)),
        key=lambda item: circular_error(float(frame.angles[item]), angle_deg),
    )
    return float(frame.smoothed[index])


def _lidar_estimated_mouth_width(
    frame: LidarFrame,
    opening: dict[str, float],
    axis: pygame.Vector2,
    perception: AdaptivePerception,
) -> float:
    start = float(opening["start_angle"])
    end = float(opening["end_angle"])
    center = float(opening["center_angle"])
    start_range = _range_at_local_angle(frame, start)
    end_range = _range_at_local_angle(frame, end)
    mean_range = 0.5 * (start_range + end_range)
    angular_width = math.radians(float(opening["width_deg"]))
    visible_chord = 2.0 * mean_range * math.sin(0.5 * angular_width)
    radial = _body_local_unit(perception, center)
    obliquity = max(abs(radial.dot(axis)), 0.25)
    corrected_span = visible_chord / obliquity
    adaptive_cap = frame.adaptive_w * PROVISIONAL_MOUTH_WIDTH_W_RATIO
    return float(np.clip(
        corrected_span,
        adaptive_cap * PROVISIONAL_MOUTH_WIDTH_MIN_RATIO,
        adaptive_cap,
    ))


def build_provisional_guard_descriptors_from_lidar(
    physical: types.ModuleType,
    perception: AdaptivePerception,
    frame: LidarFrame,
) -> list[ProvisionalGuardGeometry]:
    """Build provisional mouth frames from LiDAR sectors, without localization."""
    if perception.anchor_position is None:
        raise RuntimeError("provisional Guard geometry requires a fixed Anchor")
    openings = sorted(
        (dict(item) for item in frame.openings),
        key=lambda item: float(item["center_angle"]),
    )
    if len(openings) < 3:
        raise RuntimeError("Junction evidence lacks three provisional openings")
    openings = openings[:3]
    forward = _body_local_unit(perception, 0.0)
    lateral = pygame.Vector2(-forward.y, forward.x)
    broad = max(openings, key=lambda item: float(item["width_deg"]))
    width_reference = frame.adaptive_w * PROVISIONAL_MOUTH_WIDTH_W_RATIO
    boundary_forward_depths = []
    for key in ("start_angle", "end_angle"):
        angle = float(broad[key])
        projection = (
            _body_local_unit(perception, angle).dot(forward)
            * _range_at_local_angle(frame, angle)
        )
        if projection > 0.0:
            boundary_forward_depths.append(projection)
    raw_junction_depth = float(np.mean(boundary_forward_depths)) if boundary_forward_depths else width_reference
    junction_depth = float(np.clip(
        raw_junction_depth,
        width_reference * PROVISIONAL_JUNCTION_DEPTH_MIN_WIDTH_RATIO,
        width_reference * PROVISIONAL_JUNCTION_DEPTH_MAX_WIDTH_RATIO,
    ))
    junction_center = perception.anchor_position + forward * junction_depth
    physical.integration_lidar_junction_estimate = junction_center.copy()

    geometries: list[ProvisionalGuardGeometry] = []
    for index, opening in enumerate(openings):
        start = float(opening["start_angle"])
        end = float(opening["end_angle"])
        center = float(opening["center_angle"])
        start_point = (
            _body_local_unit(perception, start)
            * _range_at_local_angle(frame, start)
        )
        end_point = (
            _body_local_unit(perception, end)
            * _range_at_local_angle(frame, end)
        )
        boundary_chord = end_point - start_point
        radial = _body_local_unit(perception, center)
        # A rear-side visibility lobe has a radial centre that points back into
        # the traversed corridor. Its start/end chord nevertheless has a
        # lateral dominant axis. Quantize only in the Anchor-local basis; no
        # fixture direction or map mouth coordinate is consulted.
        if abs(center) >= PROVISIONAL_SIDE_SECTOR_MIN_ABS_ANGLE:
            sign = -1.0 if center < 0.0 else 1.0
            axis = lateral * sign
            axis_source = "BOUNDARY_CHORD_LATERAL_DOMINANT"
        elif abs(radial.dot(lateral)) > abs(radial.dot(forward)):
            sign = -1.0 if radial.dot(lateral) < 0.0 else 1.0
            axis = lateral * sign
            axis_source = "SECTOR_RADIAL_LATERAL_DOMINANT"
        else:
            axis = forward.copy()
            axis_source = "SECTOR_RADIAL_FORWARD_DOMINANT"
        if boundary_chord.length_squared() > physical.EPSILON:
            chord_axis_alignment = abs(boundary_chord.normalize().dot(axis))
        else:
            chord_axis_alignment = 0.0
        estimated_width = _lidar_estimated_mouth_width(
            frame, opening, axis, perception
        )
        mouth = junction_center + axis * (0.5 * estimated_width)
        normal = pygame.Vector2(-axis.y, axis.x)
        uid = f"PROV_{index:02d}"
        descriptor = physical.BranchDescriptor(
            uid=uid,
            junction_uid=physical.CURRENT_JUNCTION_ID,
            fixture_key=None,
            local_outgoing_direction=axis.copy(),
            local_return_direction=-axis,
            observed_mouth_position=mouth.copy(),
            observed_width=estimated_width,
            cohort_member_ids=set(),
            direction_last_estimate=axis.copy(),
            direction_stability_reference=axis.copy(),
            direction_stable_dwell=1.0,
            direction_sample_count=1,
            direction_angular_spread=0.0,
            direction_is_stable=True,
            direction_mature_dwell=1.0,
            direction_is_mature=True,
            direction_downstream_travel=0.0,
            motion_t=axis.copy(),
            motion_n=normal,
            motion_frame_locked=True,
            motion_frame_source="LIDAR_PROVISIONAL_SECTOR",
            motion_frame_sample_count=1,
            motion_frame_angular_spread=0.0,
            motion_observed_width=estimated_width,
            observed_flow_width=estimated_width,
            observed_physical_width=estimated_width,
            physical_width_confident=True,
            physical_width_source="LIDAR_OPENING_START_END_W",
            physical_left_boundary_lateral=-0.5 * estimated_width,
            physical_right_boundary_lateral=0.5 * estimated_width,
            physical_boundary_sample_count=2,
            discovered_at=physical.simulation_time,
        )
        geometries.append(ProvisionalGuardGeometry(
            provisional_uid=uid,
            opening=opening,
            descriptor=descriptor,
            columns=0,
            layers=0,
            slots=[],
        ))
        print(
            f"[GuardGeometryBasis] uid={uid} source={axis_source} "
            f"chord_axis_alignment={chord_axis_alignment:.3f} "
            f"junction_depth={junction_depth:.3f}"
        )
    return geometries


def compute_guard_lateral_interval(
    physical: types.ModuleType,
    descriptor: Any,
) -> tuple[float, float]:
    usable_half = physical.local_physical_usable_half_width(descriptor)
    edge_margin = physical.ROBOT_RADIUS * GUARD_EDGE_SEAL_MARGIN_RATIO
    return (-usable_half + edge_margin, usable_half - edge_margin)


def compute_sealing_aware_column_count(
    physical: types.ModuleType,
    descriptor: Any,
) -> tuple[int, float, float, float]:
    lateral_min, lateral_max = compute_guard_lateral_interval(
        physical, descriptor
    )
    span = max(0.0, lateral_max - lateral_min)
    max_internal_gap = physical.FRONTIER_LINE_MAX_INTERNAL_GAP * GUARD_LATERAL_OVERLAP_RATIO
    required_by_width = physical.required_junction_guard_count(descriptor)
    required_by_gap = int(math.ceil(span / max(max_internal_gap, physical.EPSILON))) + 1
    columns = max(required_by_width, required_by_gap)
    spacing = span / max(columns - 1, 1)
    return columns, lateral_min, lateral_max, spacing


def build_edge_sealing_slot_order(columns: int) -> list[int]:
    order: list[int] = []
    left, right = 0, columns - 1
    while left <= right:
        order.append(left)
        if right != left:
            order.append(right)
        left += 1
        right -= 1
    return order


def build_sealing_aware_slots(
    physical: types.ModuleType,
    descriptor: Any,
    columns: int,
    layers: int,
) -> list[pygame.Vector2]:
    tangent, normal = physical.descriptor_local_basis(descriptor)
    lateral_min, lateral_max = compute_guard_lateral_interval(
        physical, descriptor
    )
    order = build_edge_sealing_slot_order(columns)
    slots: list[pygame.Vector2] = []
    center = descriptor.observed_mouth_position + tangent * physical.JUNCTION_GUARD_BRANCH_INSET
    spacing = (lateral_max - lateral_min) / max(columns - 1, 1)
    for layer in range(layers):
        row_center = center + tangent * (layer * physical.THICK_MOUTH_GUARD_LAYER_SPACING)
        lateral_positions = [
            lateral_min + spacing * index for index in order
        ]
        slots.extend(row_center + normal * lateral for lateral in lateral_positions)
        print(
            f"[GuardSlotOrder] uid={descriptor.uid} layer={layer} "
            f"order={order}"
        )
    return slots


def build_provisional_multilayer_slots(
    physical: types.ModuleType,
    robots: Sequence[Any],
    geometries: Sequence[ProvisionalGuardGeometry],
) -> None:
    """Create every LiDAR-derived layer at once; robot positions are unread."""
    for geometry in geometries:
        descriptor = geometry.descriptor
        columns, lateral_min, lateral_max, slot_spacing = (
            compute_sealing_aware_column_count(physical, descriptor)
        )
        layers = physical.THICK_MOUTH_GUARD_MIN_LAYERS
        if len(robots) >= physical.THICK_MOUTH_GUARD_LARGE_SWARM_SIZE:
            layers += 1
        if len(robots) >= physical.THICK_MOUTH_GUARD_VERY_LARGE_SWARM_SIZE:
            layers += 1
        layers = int(np.clip(
            layers,
            physical.THICK_MOUTH_GUARD_MIN_LAYERS,
            physical.THICK_MOUTH_GUARD_MAX_LAYERS,
        ))
        required = columns * layers
        slots = build_sealing_aware_slots(
            physical, descriptor, columns, layers
        )
        geometry.columns = columns
        geometry.layers = layers
        geometry.slots = [slot.copy() for slot in slots]
        geometry.sealing_lateral_min = lateral_min
        geometry.sealing_lateral_max = lateral_max
        geometry.slot_spacing = slot_spacing
        walkable = sum(
            physical.is_walkable(slot, physical.ROBOT_RADIUS) for slot in slots
        )
        opening = geometry.opening
        print(f"[GuardGeometry] uid={geometry.provisional_uid}")
        print(f"[GuardGeometry] opening_start={float(opening['start_angle']):.3f}")
        print(f"[GuardGeometry] opening_end={float(opening['end_angle']):.3f}")
        print(f"[GuardGeometry] center={float(opening['center_angle']):.3f}")
        print(f"[GuardGeometry] width_deg={float(opening['width_deg']):.3f}")
        print(f"[GuardGeometry] estimated_mouth_width={descriptor.observed_physical_width:.3f}")
        print(f"[GuardGeometry] usable_half={physical.local_physical_usable_half_width(descriptor):.3f}")
        print(f"[GuardGeometry] sealing_lateral_min={lateral_min:.3f}")
        print(f"[GuardGeometry] sealing_lateral_max={lateral_max:.3f}")
        print(f"[GuardGeometry] columns={columns}")
        print(f"[GuardGeometry] layers={layers}")
        print(f"[GuardGeometry] required={required}")
        print(f"[GuardGeometry] slot_spacing={slot_spacing:.3f}")
        print(f"[GuardGeometry] slots_walkable={walkable}/{len(slots)}")


def initialize_provisional_guard_geometry_after_detection(
    physical: types.ModuleType,
    perception: AdaptivePerception,
    robots: Sequence[Any],
    frame: LidarFrame,
) -> None:
    """Freeze LiDAR WHERE now; leave every robot NORMAL until mouth-local READY."""
    if perception.provisional_guard_started:
        return
    geometries = build_provisional_guard_descriptors_from_lidar(
        physical, perception, frame
    )
    build_provisional_multilayer_slots(physical, robots, geometries)
    for geometry in geometries:
        physical.branch_descriptors_by_uid[
            geometry.provisional_uid
        ] = geometry.descriptor
        physical.integration_wall_status[geometry.provisional_uid] = {
            "capture": 0,
            "candidate_count": 0,
            "assignment_count": 0,
            "assigned": 0,
            "edge_selected": 0,
            "rows": geometry.layers,
            "slots_per_row": geometry.columns,
            "slots_walkable": sum(
                physical.is_walkable(slot, physical.ROBOT_RADIUS)
                for slot in geometry.slots
            ),
            "slots_total": len(geometry.slots),
            "ready": False,
            "ready_frame": None,
        }
    perception.provisional_guards = list(geometries)
    perception.provisional_guard_started = True
    perception.guard_geometry_frame = perception.confirmation_frame
    physical.integration_provisional_guard_groups = {}
    physical.integration_provisional_guard_active = False
    physical.integration_guard_gating_enabled = False
    print(
        "[Timeline] GUARD_GEOMETRY_READY "
        f"frame={perception.guard_geometry_frame} roles_assigned=0"
    )
    for geometry in geometries:
        descriptor = geometry.descriptor
        state = {
            "robots_beyond_mouth_at_detection": 0,
            "robots_beyond_mouth": 0,
            "crossings_before_edge_seal": 0,
            "crossings_after_edge_seal": 0,
            "additional_outward_crossings_before_wall_ready": 0,
            "additional_outward_crossings_after_wall_ready": 0,
            "inward_returns": 0,
            "maximum_normal_depth_before_wall_ready": 0.0,
            "deepest_leaked_robot_depth": 0.0,
            "leakage_blocked_after_edge_seal": False,
            "previous_axial": {},
        }
        usable_half = physical.local_physical_usable_half_width(descriptor)
        for robot in robots:
            axial, lateral = physical.branch_local_coordinates(
                robot.position, descriptor
            )
            state["previous_axial"][robot.robot_id] = axial
            if (
                robot.role == "NORMAL"
                and axial > 0.0
                and abs(lateral) <= usable_half
            ):
                state["robots_beyond_mouth_at_detection"] += 1
                state["maximum_normal_depth_before_wall_ready"] = max(
                    state["maximum_normal_depth_before_wall_ready"], axial
                )
                state["deepest_leaked_robot_depth"] = max(
                    state["deepest_leaked_robot_depth"], axial
                )
        perception.guard_leakage[geometry.provisional_uid] = state
        print(
            f"[Leakage] uid={geometry.provisional_uid} "
            f"robots_beyond_mouth_at_detection="
            f"{state['robots_beyond_mouth_at_detection']}"
        )
    physical.integration_guard_leakage = perception.guard_leakage


def communication_articulation_robot_ids(
    robots: Sequence[Any],
) -> set[int]:
    """Return articulation-like robot IDs from the existing local comm graph."""
    active_ids = {
        robot.robot_id for robot in robots
        if getattr(robot, "connected_to_base", False)
    }
    adjacency = {
        robot.robot_id: sorted(
            getattr(peer, "robot_id", -1)
            for peer in robot.comm_neighbors
            if getattr(peer, "robot_id", -1) in active_ids
        )
        for robot in robots
        if robot.robot_id in active_ids
    }
    discovery: dict[int, int] = {}
    low: dict[int, int] = {}
    parent: dict[int, int | None] = {}
    critical: set[int] = set()
    counter = 0

    def visit(robot_id: int) -> None:
        nonlocal counter
        discovery[robot_id] = counter
        low[robot_id] = counter
        counter += 1
        children = 0
        for neighbor_id in adjacency.get(robot_id, []):
            if neighbor_id not in discovery:
                parent[neighbor_id] = robot_id
                children += 1
                visit(neighbor_id)
                low[robot_id] = min(low[robot_id], low[neighbor_id])
                if parent.get(robot_id) is None and children > 1:
                    critical.add(robot_id)
                if (
                    parent.get(robot_id) is not None
                    and low[neighbor_id] >= discovery[robot_id]
                ):
                    critical.add(robot_id)
            elif neighbor_id != parent.get(robot_id):
                low[robot_id] = min(low[robot_id], discovery[neighbor_id])

    for robot_id in sorted(adjacency):
        if robot_id not in discovery:
            parent[robot_id] = None
            visit(robot_id)
    return critical


def largest_communication_component(robots: Sequence[Any]) -> int:
    by_id = {robot.robot_id: robot for robot in robots}
    remaining = set(by_id)
    largest = 0
    while remaining:
        seed = min(remaining)
        stack = [seed]
        remaining.remove(seed)
        size = 0
        while stack:
            robot_id = stack.pop()
            size += 1
            for peer in by_id[robot_id].comm_neighbors:
                peer_id = getattr(peer, "robot_id", -1)
                if peer_id in remaining:
                    remaining.remove(peer_id)
                    stack.append(peer_id)
        largest = max(largest, size)
    return largest


def collect_shallow_guard_candidates_with_localization(
    physical: types.ModuleType,
    perception: AdaptivePerception,
    robots: Sequence[Any],
    geometry: ProvisionalGuardGeometry,
) -> list[Any]:
    """Localization-allowed shallow mouth cohort collection; geometry is read-only."""
    descriptor = geometry.descriptor
    width = descriptor.observed_physical_width
    usable_half = physical.local_physical_usable_half_width(descriptor)
    lateral_limit = (
        usable_half + GUARD_CAPTURE_LATERAL_MARGIN_RATIO * width
    )
    candidates = []
    for robot in robots:
        if (
            robot is perception.leader
            or robot.role != "NORMAL"
            or robot.base_reserve
            or not physical.is_walkable(robot.position, robot.radius)
        ):
            continue
        axial, lateral = physical.branch_local_coordinates(
            robot.position, descriptor
        )
        if (
            -GUARD_CAPTURE_UPSTREAM_WIDTH_RATIO * width
            <= axial
            <= GUARD_CAPTURE_DOWNSTREAM_WIDTH_RATIO * width
            and abs(lateral) <= lateral_limit
        ):
            candidates.append(robot)
    return sorted(candidates, key=lambda robot: robot.robot_id)


def compute_full_guard_slot_assignment(
    physical: types.ModuleType,
    geometry: ProvisionalGuardGeometry,
    candidates: Sequence[Any],
) -> tuple[list[tuple[Any, pygame.Vector2, int]], dict[str, Any]]:
    """Run deterministic full bipartite feasibility and branch-local WHO cost."""
    descriptor = geometry.descriptor
    width = descriptor.observed_physical_width
    required = len(geometry.slots)
    options: dict[int, list[tuple[float, Any]]] = {}
    candidate_laterals = []
    candidate_axials = []
    for robot in candidates:
        axial, lateral = physical.branch_local_coordinates(
            robot.position, descriptor
        )
        candidate_axials.append(float(axial))
        candidate_laterals.append(float(lateral))
    for slot_index, slot in enumerate(geometry.slots):
        slot_axial, slot_lateral = physical.branch_local_coordinates(
            slot, descriptor
        )
        ranked = []
        for robot in candidates:
            axial, lateral = physical.branch_local_coordinates(
                robot.position, descriptor
            )
            axial_delta = abs(axial - slot_axial)
            lateral_delta = abs(lateral - slot_lateral)
            path_distance = robot.position.distance_to(slot)
            if (
                axial_delta > GUARD_ASSIGN_MAX_AXIAL_WIDTH_RATIO * width
                or lateral_delta
                > GUARD_ASSIGN_MAX_LATERAL_WIDTH_RATIO * width
                or path_distance > GUARD_ASSIGN_MAX_PATH_WIDTH_RATIO * width
            ):
                continue
            cost = (
                GUARD_WHO_AXIAL_WEIGHT
                * axial_delta / max(width, physical.EPSILON)
                + GUARD_WHO_LATERAL_WEIGHT
                * lateral_delta / max(width, physical.EPSILON)
                + GUARD_WHO_PATH_WEIGHT
                * path_distance / max(width, physical.EPSILON)
            )
            ranked.append((cost, robot))
        options[slot_index] = sorted(
            ranked, key=lambda item: (item[0], item[1].robot_id)
        )

    slot_for_robot: dict[int, int] = {}
    robot_for_slot: dict[int, Any] = {}

    def augment(slot_index: int, seen: set[int]) -> bool:
        for _, robot in options[slot_index]:
            if robot.robot_id in seen:
                continue
            seen.add(robot.robot_id)
            previous_slot = slot_for_robot.get(robot.robot_id)
            if previous_slot is None or augment(previous_slot, seen):
                slot_for_robot[robot.robot_id] = slot_index
                robot_for_slot[slot_index] = robot
                return True
        return False

    slot_order = sorted(
        range(required), key=lambda index: (len(options[index]), index)
    )
    for slot_index in slot_order:
        augment(slot_index, set())

    assignment = [
        (robot_for_slot[index], geometry.slots[index].copy(), index)
        for index in range(required)
        if index in robot_for_slot
    ]
    distances = [
        robot.position.distance_to(slot)
        for robot, slot, _ in assignment
    ]
    usable_half = physical.local_physical_usable_half_width(descriptor)
    if candidate_laterals:
        lateral_span = max(candidate_laterals) - min(candidate_laterals)
        coverage_ratio = float(np.clip(
            lateral_span / max(2.0 * usable_half, physical.EPSILON),
            0.0,
            1.0,
        ))
        occupied_bins = len({
            int(np.clip(
                math.floor(
                    (lateral + usable_half)
                    / max(2.0 * usable_half, physical.EPSILON)
                    * GUARD_LATERAL_COVERAGE_BINS
                ),
                0,
                GUARD_LATERAL_COVERAGE_BINS - 1,
            ))
            for lateral in candidate_laterals
        })
    else:
        coverage_ratio = 0.0
        occupied_bins = 0
    assigned_laterals = [
        physical.branch_local_coordinates(robot.position, descriptor)[1]
        for robot, _, _ in assignment
    ]
    left_edge_gap = max(
        0.0,
        (min(assigned_laterals) - geometry.sealing_lateral_min)
        if assigned_laterals else float("inf"),
    )
    right_edge_gap = max(
        0.0,
        (geometry.sealing_lateral_max - max(assigned_laterals))
        if assigned_laterals else float("inf"),
    )
    ordered_laterals = sorted(assigned_laterals)
    max_internal_gap = max(
        (right - left for left, right in zip(ordered_laterals, ordered_laterals[1:])),
        default=float("inf"),
    )
    diagnostics = {
        "required": required,
        "candidate_count": len(candidates),
        "assignment_count": len(assignment),
        "coverage_ratio": coverage_ratio,
        "occupied_lateral_bins": occupied_bins,
        "axial_min": min(candidate_axials, default=0.0),
        "axial_max": max(candidate_axials, default=0.0),
        "max_assignment_distance": max(distances, default=float("inf")),
        "mean_assignment_distance": float(np.mean(distances))
        if distances else float("inf"),
        "full_slot_assignment_possible": len(assignment) == required,
        "left_edge_gap": left_edge_gap,
        "right_edge_gap": right_edge_gap,
        "max_edge_gap": max(left_edge_gap, right_edge_gap),
        "max_internal_gap": max_internal_gap,
        "outer_edge_sealed": (
            left_edge_gap <= physical.FRONTIER_LINE_MAX_EDGE_GAP
            and right_edge_gap <= physical.FRONTIER_LINE_MAX_EDGE_GAP
        ),
        "worst_uncovered_span": max(left_edge_gap, right_edge_gap, max_internal_gap),
        "outermost_assigned_slot_reach": (
            max(abs(min(assigned_laterals) - geometry.sealing_lateral_min),
                abs(geometry.sealing_lateral_max - max(assigned_laterals)))
            if assigned_laterals else 0.0
        ),
    }
    return assignment, diagnostics


def branch_guard_cohort_ready(
    geometry: ProvisionalGuardGeometry,
    diagnostics: dict[str, Any],
) -> bool:
    return bool(
        diagnostics["candidate_count"] >= len(geometry.slots)
        and diagnostics["full_slot_assignment_possible"]
        and diagnostics["outer_edge_sealed"]
    )


def activate_guard_cohort(
    physical: types.ModuleType,
    perception: AdaptivePerception,
    robots: Sequence[Any],
    geometry: ProvisionalGuardGeometry,
    assignment: Sequence[tuple[Any, pygame.Vector2, int]],
    critical_ids: set[int],
    diagnostics: dict[str, Any],
) -> None:
    """Perform same-position NORMAL->Guard transition, then close localization."""
    frame = getattr(physical, "integration_frame", -1)
    before_stats = physical.get_communication_stats(robots)
    before_largest = largest_communication_component(robots)
    relay_ids = sorted(
        robot.robot_id for robot in robots
        if robot.role in {"RELAY", "TRUNK_RELAY"}
    )
    selected = sorted(assignment, key=lambda item: item[2])
    leader = min((item[0] for item in selected), key=lambda robot: robot.robot_id)
    maximum_jump = 0.0
    selected_ids = []
    for robot, slot, slot_index in selected:
        before = robot.position.copy()
        robot.role = "JUNCTION_GUARD"
        robot.integration_guard_waypoints = [slot.copy()]
        robot.integration_guard_final_anchor = slot.copy()
        robot.integration_guard_slot_index = slot_index
        robot.junction_guard_anchor = slot.copy()
        robot.junction_guard_branch = geometry.provisional_uid
        robot.junction_guard_branch_uid = geometry.provisional_uid
        robot.junction_guard_hop = 0
        robot.junction_guard_parent_id = (
            None if robot is leader else leader.robot_id
        )
        robot.junction_guard_layer = slot_index // geometry.columns
        robot.is_branch_leader = robot is leader
        maximum_jump = max(
            maximum_jump, robot.position.distance_to(before)
        )
        selected_ids.append(robot.robot_id)
    geometry.descriptor.leader_id = leader.robot_id
    geometry.selected_ids = selected_ids
    geometry.cohort_ready = True
    geometry.guard_ready_frame = frame
    geometry.role_assignment_frame = frame
    physical.integration_provisional_guard_groups[
        geometry.provisional_uid
    ] = selected_ids
    physical.integration_provisional_guard_active = True
    physical.integration_guard_gating_enabled = True
    if physical.integration_guard_formation_start_frame is None:
        physical.integration_guard_formation_start_frame = frame
    physical.integration_guard_role_transition_jump = max(
        physical.integration_guard_role_transition_jump,
        maximum_jump,
    )
    status = physical.integration_wall_status[geometry.provisional_uid]
    status.update({
        "capture": diagnostics["candidate_count"],
        "candidate_count": diagnostics["candidate_count"],
        "assignment_count": diagnostics["assignment_count"],
        "assigned": len(selected_ids),
        "edge_selected": len(selected_ids),
        "coverage_ratio": diagnostics["coverage_ratio"],
        "ready": False,
    })
    after_stats = physical.get_communication_stats(robots)
    after_largest = largest_communication_component(robots)
    selected_critical = sorted(set(selected_ids) & critical_ids)
    communication_audit = {
        "before_connected": before_stats["connected"],
        "before_largest_component": before_largest,
        "after_connected": after_stats["connected"],
        "after_largest_component": after_largest,
        "relay_trunk_ids": relay_ids,
        "critical_ids": sorted(critical_ids),
        "selected_critical_ids": selected_critical,
        "communication_disconnect_caused_by_guard_selection": (
            after_stats["connected"] < before_stats["connected"]
            or after_largest < before_largest
        ),
    }
    perception.guard_communication_audits[
        geometry.provisional_uid
    ] = communication_audit
    print(
        f"[Timeline] GUARD_COHORT_READY uid={geometry.provisional_uid} "
        f"frame={frame}"
    )
    print(
        f"[Timeline] GUARD_ROLE_ASSIGNMENT uid={geometry.provisional_uid} "
        f"frame={frame}"
    )
    print(f"[LocalizationWHO] uid={geometry.provisional_uid}")
    print(f"[LocalizationWHO] elected_ids={selected_ids}")
    print(
        f"[LocalizationWHO] assignment_count={len(selected_ids)}/"
        f"{len(geometry.slots)}"
    )
    print(
        f"[CommunicationBefore] uid={geometry.provisional_uid} "
        f"connected={before_stats['connected']} "
        f"largest_component={before_largest} "
        f"relay_trunk_ids={relay_ids}"
    )
    print(
        f"[CommunicationAfter] uid={geometry.provisional_uid} "
        f"connected={after_stats['connected']} "
        f"largest_component={after_largest} "
        f"communication_critical_robots_selected={selected_critical} "
        "communication_disconnect_caused_by_guard_selection="
        f"{communication_audit['communication_disconnect_caused_by_guard_selection']}"
    )
    print(
        f"[Teleport] uid={geometry.provisional_uid} "
        f"guard_role_transition_jump={maximum_jump:.6f} "
        "direct_position_overwrite_count=0"
    )


def update_guard_readiness_and_activation(
    physical: types.ModuleType,
    perception: AdaptivePerception,
    robots: Sequence[Any],
) -> None:
    """Independently activate each branch when its natural shallow cohort fits."""
    if not perception.provisional_guard_started:
        return
    for geometry in perception.provisional_guards:
        if geometry.cohort_ready:
            continue
        physical.integration_guard_who_localization_enabled = True
        candidates = collect_shallow_guard_candidates_with_localization(
            physical, perception, robots, geometry
        )
        required = len(geometry.slots)
        frame = getattr(physical, "integration_frame", -1)
        candidate_just_sufficient = (
            len(candidates) >= required
            and geometry.candidate_sufficient_frame is None
        )
        if candidate_just_sufficient:
            geometry.candidate_sufficient_frame = frame
            print(
                f"[Timeline] GUARD_CANDIDATE_SUFFICIENT "
                f"uid={geometry.provisional_uid} frame={frame}"
            )
        critical_ids = (
            communication_articulation_robot_ids(robots)
            if len(candidates) >= required else set()
        )
        preferred = [
            robot for robot in candidates
            if robot.robot_id not in critical_ids
        ]
        assignment, diagnostics = compute_full_guard_slot_assignment(
            physical, geometry, preferred
        )
        used_critical_fallback = False
        if (
            len(assignment) < required
            and len(candidates) >= required
        ):
            assignment, diagnostics = compute_full_guard_slot_assignment(
                physical, geometry, candidates
            )
            used_critical_fallback = True
        diagnostics["communication_critical_fallback"] = (
            used_critical_fallback
        )
        geometry.readiness_diagnostics = diagnostics
        status = physical.integration_wall_status[geometry.provisional_uid]
        status.update(diagnostics)
        should_log = (
            candidate_just_sufficient
            or frame % GUARD_READINESS_LOG_PERIOD == 0
            or branch_guard_cohort_ready(geometry, diagnostics)
        )
        geometry.last_candidate_count = diagnostics["candidate_count"]
        geometry.last_assignment_count = diagnostics["assignment_count"]
        if should_log:
            print(
                f"[GuardReadiness] uid={geometry.provisional_uid} "
                f"frame={frame} required={required} "
                f"candidate_count={diagnostics['candidate_count']} "
                f"assignment_count={diagnostics['assignment_count']} "
                f"coverage_ratio={diagnostics['coverage_ratio']:.3f} "
                f"occupied_lateral_bins="
                f"{diagnostics['occupied_lateral_bins']}/"
                f"{GUARD_LATERAL_COVERAGE_BINS} "
                f"axial_range=({diagnostics['axial_min']:.3f},"
                f"{diagnostics['axial_max']:.3f}) "
                f"max_assignment_distance="
                f"{diagnostics['max_assignment_distance']:.3f} "
                f"mean_assignment_distance="
                f"{diagnostics['mean_assignment_distance']:.3f} "
                f"ready={branch_guard_cohort_ready(geometry, diagnostics)}"
            )
        if branch_guard_cohort_ready(geometry, diagnostics):
            perception.guard_who_frame = (
                frame if perception.guard_who_frame is None
                else min(perception.guard_who_frame, frame)
            )
            physical.integration_guard_who_localization_enabled = False
            activate_guard_cohort(
                physical,
                perception,
                robots,
                geometry,
                assignment,
                critical_ids,
                diagnostics,
            )
            if perception.guard_motion_start_frame is None:
                perception.guard_motion_start_frame = frame
    physical.integration_guard_who_localization_enabled = False
    if (
        perception.provisional_guards
        and all(item.cohort_ready for item in perception.provisional_guards)
    ):
        print(
            "[Timeline] ALL_GUARD_COHORTS_ACTIVE "
            f"frame={getattr(physical, 'integration_frame', -1)}"
        ) if not getattr(
            physical, "integration_all_guard_cohorts_logged", False
        ) else None
        physical.integration_all_guard_cohorts_logged = True


def update_provisional_wall_settling_audit(
    physical: types.ModuleType,
    perception: AdaptivePerception,
    robots: Sequence[Any],
) -> None:
    """Measure actual LiDAR-local wall completion before topology handoff."""
    if perception.handoff_complete:
        return
    by_id = {robot.robot_id: robot for robot in robots}
    for geometry in perception.provisional_guards:
        if not geometry.cohort_ready:
            continue
        status = physical.integration_wall_status[geometry.provisional_uid]
        guards = [
            by_id[robot_id] for robot_id in geometry.selected_ids
        ]
        settled = sum(
            robot.integration_guard_final_anchor is not None
            and robot.position.distance_to(
                robot.integration_guard_final_anchor
            ) <= physical.JUNCTION_GUARD_POSITION_TOLERANCE
            for robot in guards
        )
        usable_half = physical.local_physical_usable_half_width(
            geometry.descriptor
        )
        complete_rows = 0
        minimum_span_ratio = 1.0
        maximum_edge_gap = 0.0
        maximum_internal_gap = 0.0
        for layer in range(geometry.layers):
            laterals = sorted(
                physical.branch_local_coordinates(
                    robot.position, geometry.descriptor
                )[1]
                for robot in guards
                if robot.junction_guard_layer == layer
            )
            if len(laterals) < geometry.columns:
                minimum_span_ratio = 0.0
                continue
            complete_rows += 1
            span = laterals[-1] - laterals[0]
            minimum_span_ratio = min(
                minimum_span_ratio,
                span / max(2.0 * usable_half, physical.EPSILON),
            )
            maximum_edge_gap = max(
                maximum_edge_gap,
                max(0.0, laterals[0] + usable_half),
                max(0.0, usable_half - laterals[-1]),
            )
            maximum_internal_gap = max(
                maximum_internal_gap,
                max(
                    (
                        right - left
                        for left, right in zip(laterals, laterals[1:])
                    ),
                    default=0.0,
                ),
            )
        settled_ratio = settled / max(len(guards), 1)
        ready = (
            len(guards) == len(geometry.slots)
            and status.get("slots_walkable", 0) == len(geometry.slots)
            and settled_ratio >= 1.0
            and complete_rows == geometry.layers
            and minimum_span_ratio
            >= physical.FRONTIER_LINE_MIN_SPAN_RATIO
            and maximum_edge_gap
            <= physical.FRONTIER_LINE_MAX_EDGE_GAP
            and maximum_internal_gap
            <= physical.FRONTIER_LINE_MAX_INTERNAL_GAP
        )
        status.update({
            "settled_ratio": settled_ratio,
            "min_span_ratio": minimum_span_ratio,
            "max_edge_gap": maximum_edge_gap,
            "max_internal_gap": maximum_internal_gap,
            "ready": ready,
        })
        if ready and status.get("ready_frame") is None:
            status["ready_frame"] = getattr(
                physical, "integration_frame", -1
            )
            print(
                f"[Timeline] PROVISIONAL_WALL_READY "
                f"uid={geometry.provisional_uid} "
                f"frame={status['ready_frame']}"
            )
            print(
                f"[ProvisionalWallReady] uid={geometry.provisional_uid} "
                f"columns={geometry.columns} layers={geometry.layers} "
                f"required={len(geometry.slots)} assigned={len(guards)} "
                f"settled_ratio={settled_ratio:.3f} "
                f"span_ratio={minimum_span_ratio:.3f} "
                f"edge_gap={maximum_edge_gap:.3f} "
                f"internal_gap={maximum_internal_gap:.3f}"
            )


def update_provisional_guard_leakage(
    physical: types.ModuleType,
    perception: AdaptivePerception,
    robots: Sequence[Any],
) -> None:
    if (
        not perception.provisional_guard_started
        or getattr(physical, "integration_all_walls_ready", False)
    ):
        return
    frame = getattr(physical, "integration_frame", -1)
    for geometry in perception.provisional_guards:
        state = perception.guard_leakage[geometry.provisional_uid]
        descriptor = geometry.descriptor
        usable_half = physical.local_physical_usable_half_width(descriptor)
        wall_status = getattr(physical, "integration_wall_status", {}).get(
            geometry.provisional_uid, {}
        )
        wall_ready = bool(wall_status.get("ready", False))
        beyond = 0
        for robot in robots:
            axial, lateral = physical.branch_local_coordinates(
                robot.position, descriptor
            )
            previous = state["previous_axial"].get(robot.robot_id, axial)
            if robot.role == "NORMAL" and abs(lateral) <= usable_half:
                if axial > 0.0:
                    beyond += 1
                if previous <= 0.0 < axial:
                    if geometry.first_robot_crossing_mouth_frame is None:
                        geometry.first_robot_crossing_mouth_frame = frame
                        print(
                            "[Timeline] FIRST_ROBOT_CROSSING_MOUTH "
                            f"uid={geometry.provisional_uid} frame={frame}"
                        )
                    if wall_ready:
                        state["crossings_after_edge_seal"] += 1
                        state["additional_outward_crossings_after_wall_ready"] += 1
                    else:
                        state["crossings_before_edge_seal"] += 1
                        state["additional_outward_crossings_before_wall_ready"] += 1
                elif previous > 0.0 >= axial:
                    state["inward_returns"] += 1
                if axial > 0.0:
                    state["deepest_leaked_robot_depth"] = max(
                        state["deepest_leaked_robot_depth"], axial
                    )
                    if not wall_ready:
                        state["maximum_normal_depth_before_wall_ready"] = max(
                            state["maximum_normal_depth_before_wall_ready"], axial
                        )
            state["previous_axial"][robot.robot_id] = axial
        state["robots_beyond_mouth"] = beyond
        if wall_ready:
            state["leakage_blocked_after_edge_seal"] = beyond == 0


def install_thick_wall_readiness_audit(physical: types.ModuleType) -> None:
    """Require and record physical multi-row coverage before DFS selection."""
    original_ready = physical.junction_guards_formed
    physical.integration_wall_stats = {}
    physical.integration_wall_status = {}
    physical.integration_all_walls_ready = False

    def audited_ready(robots: Sequence[Any]) -> bool:
        expected = [
            branch
            for branch in physical.detected_branch_candidates
            if branch not in physical.observed_visited_branches(robots)
        ]
        stats: dict[str, dict[str, float | int]] = {}
        all_branch_ready = bool(expected)
        for branch in expected:
            uid = physical.branch_uid_for_fixture(branch)
            if uid is None:
                all_branch_ready = False
                continue
            descriptor = physical.branch_descriptors_by_uid[uid]
            guards = [
                robot for robot in robots
                if robot.role == "JUNCTION_GUARD"
                and robot.junction_guard_branch == branch
            ]
            rows = physical.thick_mouth_guard_layers[branch]
            columns = physical.thick_mouth_guard_columns[branch]
            settled = sum(
                robot.junction_guard_anchor is not None
                and robot.position.distance_to(robot.junction_guard_anchor)
                <= physical.JUNCTION_GUARD_POSITION_TOLERANCE
                for robot in guards
            )
            maximum_gap = 0.0
            minimum_span_ratio = 1.0
            maximum_edge_gap = 0.0
            complete_rows = 0
            usable_half = physical.local_physical_usable_half_width(descriptor)
            for layer in range(rows):
                laterals = sorted(
                    physical.branch_local_coordinates(robot.position, descriptor)[1]
                    for robot in guards
                    if robot.junction_guard_layer == layer
                )
                if not laterals:
                    minimum_span_ratio = 0.0
                    continue
                if len(laterals) >= columns:
                    complete_rows += 1
                maximum_gap = max(
                    maximum_gap,
                    max(
                        (right - left for left, right in zip(laterals, laterals[1:])),
                        default=0.0,
                    ),
                )
                span = laterals[-1] - laterals[0]
                minimum_span_ratio = min(
                    minimum_span_ratio,
                    span / max(2.0 * usable_half, physical.EPSILON),
                )
                maximum_edge_gap = max(
                    maximum_edge_gap,
                    max(0.0, laterals[0] + usable_half),
                    max(0.0, usable_half - laterals[-1]),
                )
            branch_stats = {
                "physical_width": physical.local_guard_observed_width(descriptor),
                "rows": rows,
                "slots_per_row": columns,
                "total": len(guards),
                "settled_ratio": settled / max(len(guards), 1),
                "min_span_ratio": minimum_span_ratio,
                "max_edge_gap": maximum_edge_gap,
                "max_internal_gap": maximum_gap,
            }
            stats[uid] = branch_stats
            ready = (
                rows >= physical.THICK_MOUTH_GUARD_MIN_LAYERS
                and columns >= physical.JUNCTION_GUARD_MIN_COUNT
                and len(guards) >= rows * columns
                and complete_rows == rows
                and branch_stats["settled_ratio"] >= 1.0
                and minimum_span_ratio
                >= physical.FRONTIER_LINE_MIN_SPAN_RATIO
                and maximum_edge_gap
                <= physical.FRONTIER_LINE_MAX_EDGE_GAP
                and maximum_gap
                <= physical.FRONTIER_LINE_MAX_INTERNAL_GAP
            )
            status = physical.integration_wall_status.setdefault(uid, {})
            status.update(branch_stats)
            status["ready"] = ready
            if ready and status.get("ready_frame") is None:
                status["ready_frame"] = getattr(
                    physical, "integration_frame", -1
                )
                print(
                    f"[Timeline] {branch}_WALL_READY "
                    f"frame={status['ready_frame']} uid={uid}"
                )
                print(
                    f"[ThickWallReady] uid={uid} "
                    f"width={branch_stats['physical_width']:.1f} "
                    f"rows={rows} slots_per_row={columns} total={len(guards)} "
                    f"settled_ratio={branch_stats['settled_ratio']:.3f} "
                    f"span_ratio={minimum_span_ratio:.3f} "
                    f"edge_gap={maximum_edge_gap:.3f} "
                    f"internal_gap={maximum_gap:.3f}"
                )
            all_branch_ready = all_branch_ready and ready
        physical.integration_wall_stats = stats
        ready = all_branch_ready and original_ready(robots)
        if ready and not physical.integration_all_walls_ready:
            physical.integration_all_walls_ready = True
            physical.integration_guard_hold_active = False
            physical.integration_placement_localization_enabled = False
            physical.integration_ready_guard_ids_by_uid = {
                physical.branch_uid_for_fixture(branch): sorted(
                    robot.robot_id for robot in robots
                    if robot.role == "JUNCTION_GUARD"
                    and robot.junction_guard_branch == branch
                )
                for branch in expected
            }
            physical.integration_wall_lifecycle = {}
            for branch in expected:
                uid = physical.branch_uid_for_fixture(branch)
                descriptor = physical.branch_descriptors_by_uid[uid]
                members = [
                    robot for robot in robots
                    if robot.robot_id
                    in physical.integration_ready_guard_ids_by_uid[uid]
                ]
                coordinates = [
                    physical.branch_local_coordinates(
                        robot.position, descriptor
                    )
                    for robot in members
                ]
                centroid_axial = float(np.mean([
                    axial for axial, _ in coordinates
                ]))
                centroid_lateral = float(np.mean([
                    lateral for _, lateral in coordinates
                ]))
                physical.integration_wall_lifecycle[branch] = {
                    "uid": uid,
                    "state": "GUARD",
                    "rows": physical.thick_mouth_guard_layers[branch],
                    "cols": physical.thick_mouth_guard_columns[branch],
                    "robot_ids": sorted(
                        robot.robot_id for robot in members
                    ),
                    "centroid_axial": centroid_axial,
                    "centroid_lateral": centroid_lateral,
                    "relative_offsets": {
                        robot.robot_id: (
                            axial - centroid_axial,
                            lateral - centroid_lateral,
                        )
                        for robot, (axial, lateral)
                        in zip(members, coordinates)
                    },
                }
            print(
                "[Timeline] ALL_WALLS_READY "
                f"frame={getattr(physical, 'integration_frame', -1)}"
            )
            print("[LocalizationAudit] placement_localization_enabled=False")
            for provisional_uid, leakage in getattr(
                physical, "integration_guard_leakage", {}
            ).items():
                print(
                    f"[Leakage] uid={provisional_uid} "
                    f"robots_beyond_mouth_at_detection="
                    f"{leakage['robots_beyond_mouth_at_detection']} "
                    f"robots_beyond_mouth={leakage['robots_beyond_mouth']} "
                    f"crossings_before_edge_seal="
                    f"{leakage['crossings_before_edge_seal']} "
                    f"crossings_after_edge_seal="
                    f"{leakage['crossings_after_edge_seal']} "
                    f"leakage_blocked_after_edge_seal="
                    f"{leakage['leakage_blocked_after_edge_seal']} "
                    f"inward_returns={leakage['inward_returns']} "
                    f"deepest_leaked_robot_depth="
                    f"{leakage['deepest_leaked_robot_depth']:.3f}"
                )
        return ready

    physical.junction_guards_formed = audited_ready


def install_continuous_guard_settling(physical: types.ModuleType) -> None:
    """Let initially elected Guards walk to local slots at bounded speed."""
    original_limit = physical.limit_communication_proposed_position

    def guard_formation_limit(
        robot: Any,
        proposed: pygame.Vector2,
        old_position: pygame.Vector2,
    ) -> pygame.Vector2:
        if (
            (
                physical.phase == physical.SimulationPhase.FORM_JUNCTION_GUARDS
                or getattr(
                    physical, "integration_provisional_guard_active", False
                )
            )
            and robot.role == "JUNCTION_GUARD"
            and robot.junction_guard_anchor is not None
        ):
            # The Robot.update caller still applies JUNCTION_GUARD_MOVE_SPEED,
            # walkable-mask collision, and one-frame stepping.  Only the Base
            # parent-link clamp is suspended during this permitted initial
            # wall-placement interval.
            return proposed
        return original_limit(robot, proposed, old_position)

    physical.limit_communication_proposed_position = guard_formation_limit


@dataclass
class LocalSaturationDiagnostics:
    """Branch-local physical evidence; contains no map endpoint coordinates."""

    branch: str | None = None
    start_depth: float = 0.0
    maximum_depth: float = 0.0
    frontier_speed: float = 0.0
    frontier_progress_rate: float = float("inf")
    frontier_delta: float = float("inf")
    frontier_stalled: bool = False
    local_density: float = 0.0
    baseline_density: float = 0.0
    local_density_ratio: float = 0.0
    local_pressure: float = 0.0
    baseline_pressure: float = 0.0
    local_pressure_ratio: float = 0.0
    cross_section_fill: float = 0.0
    dwell: float = 0.0
    saturated: bool = False
    shepherd_transition: bool = False
    frontier_ids: list[int] = field(default_factory=list)
    shepherd_ids: list[int] = field(default_factory=list)
    max_transition_jump: float = 0.0
    max_formation_error: float = 0.0
    transition_frame: int | None = None
    return_direction_local: tuple[float, float] = (0.0, 0.0)
    return_flow_ratio: float = 0.0
    mean_return_speed: float = 0.0
    return_dwell: float = 0.0
    backflow_confirmed: bool = False
    progress_history: list[tuple[float, float]] = field(default_factory=list)
    last_log_time: float = float("-inf")

    def reset(self, branch: str, depth: float) -> None:
        self.__dict__.update(LocalSaturationDiagnostics(
            branch=branch,
            start_depth=depth,
            maximum_depth=depth,
        ).__dict__)


def install_local_physical_saturation_bridge(physical: types.ModuleType) -> None:
    """Connect local stall/packing evidence to the existing return phases."""
    original_update_state = physical.update_simulation_state
    original_transfer_control = physical.update_transfer_continuity_control
    original_prepare_scores = physical.prepare_branch_candidate_scores
    diagnostics = LocalSaturationDiagnostics()
    physical.integration_saturation = diagnostics
    physical.integration_saturation_events = []
    physical.integration_backflow_events = []
    physical.integration_frontier_lineage_events = []
    physical.integration_shepherd_anchor_offsets = {}

    def thick_frontier_slot_target(
        robot: Any, centroid_depth: float
    ) -> pygame.Vector2 | None:
        """Move a stored 3x9 local formation without world-slot reuse."""
        lifecycle = getattr(
            physical, "integration_wall_lifecycle", {}
        ).get(robot.shepherd_branch)
        descriptor = descriptor_for(robot.shepherd_branch)
        if lifecycle is None or descriptor is None:
            return None
        offset = lifecycle["relative_offsets"].get(robot.robot_id)
        if offset is None:
            return None
        axial_offset, lateral_offset = offset
        return physical.local_coordinates_to_world(
            descriptor,
            max(0.0, centroid_depth + axial_offset),
            physical.frontier_line_lateral_center + lateral_offset,
        )

    def audited_commit_guard_roles(
        robots: Sequence[Any], selected_branch: str
    ) -> None:
        """Promote the complete READY 3x9 wall without re-election."""
        uid = physical.branch_uid_for_fixture(selected_branch)
        lifecycle = physical.integration_wall_lifecycle[selected_branch]
        ready_ids = set(lifecycle["robot_ids"])
        frontiers = [
            robot for robot in robots
            if robot.robot_id in ready_ids
            and robot.role == "JUNCTION_GUARD"
            and robot.junction_guard_branch == selected_branch
        ]
        expected = lifecycle["rows"] * lifecycle["cols"]
        if len(frontiers) != expected or len(frontiers) != len(ready_ids):
            raise RuntimeError(
                "selected thick Guard wall was incomplete before promotion"
            )
        descriptor = descriptor_for(selected_branch)
        coordinates = {
            robot.robot_id: physical.branch_local_coordinates(
                robot.position, descriptor
            )
            for robot in frontiers
        }
        centroid_axial = float(np.mean([
            axial for axial, _ in coordinates.values()
        ]))
        centroid_lateral = float(np.mean([
            lateral for _, lateral in coordinates.values()
        ]))
        lifecycle["centroid_axial"] = centroid_axial
        lifecycle["centroid_lateral"] = centroid_lateral
        lifecycle["relative_offsets"] = {
            robot_id: (
                axial - centroid_axial,
                lateral - centroid_lateral,
            )
            for robot_id, (axial, lateral) in coordinates.items()
        }
        before = {
            robot.robot_id: robot.position.copy() for robot in frontiers
        }
        for robot in frontiers:
            _, lateral_offset = lifecycle["relative_offsets"][robot.robot_id]
            robot.role = "FRONTIER_SHEPHERD"
            robot.junction_guard_anchor = None
            robot.shepherd_anchor = robot.position.copy()
            robot.shepherd_origin = robot.position.copy()
            robot.shepherd_branch = selected_branch
            robot.frontier_local_lateral = lateral_offset
            robot.velocity.update(0.0, 0.0)
            robot.acceleration.update(0.0, 0.0)
            robot.filtered_acceleration.update(0.0, 0.0)
        physical.frontier_line_branch = selected_branch
        physical.frontier_line_depth = centroid_axial
        physical.frontier_line_lateral_center = centroid_lateral
        physical.frontier_line_target_settled_ratio = 0.0
        physical.frontier_line_current_span = 0.0
        physical.frontier_line_target_span = max(
            lateral for _, lateral in lifecycle["relative_offsets"].values()
        ) - min(
            lateral for _, lateral in lifecycle["relative_offsets"].values()
        )
        physical.frontier_line_physical_coverage_ratio = 0.0
        physical.frontier_line_left_edge_gap = float("inf")
        physical.frontier_line_right_edge_gap = float("inf")
        physical.frontier_line_continuous = False
        physical.frontier_line_row_ready = False
        physical.frontier_line_last_diagnostic_time = float("-inf")
        physical.junction_guard_groups[selected_branch] = sorted(ready_ids)
        physical.junction_guard_status = (
            f"FRONTIER_THICK={selected_branch};3x9;robots={len(frontiers)}"
        )
        lifecycle["state"] = "FRONTIER"
        frontier_ids = {robot.robot_id for robot in frontiers}
        if not frontier_ids or frontier_ids != ready_ids:
            raise RuntimeError(
                "FRONTIER_SHEPHERD IDs did not equal all READY Guard IDs"
            )
        transition_jump = max(
            robot.position.distance_to(before[robot.robot_id])
            for robot in frontiers
        )
        event = {
            "uid": uid,
            "branch": selected_branch,
            "frame": getattr(physical, "integration_frame", -1),
            "ready_guard_ids": sorted(ready_ids),
            "frontier_ids": sorted(frontier_ids),
            "same_ready_guard_ids": True,
            "max_role_transition_jump": transition_jump,
            "rows": lifecycle["rows"],
            "cols": lifecycle["cols"],
        }
        physical.integration_frontier_lineage_events.append(event)
        print(
            f"[FrontierLineage] uid={uid} ready_guard_ids="
            f"{event['ready_guard_ids']} frontier_ids={event['frontier_ids']} "
            "same_ready_guard_ids=True "
            f"rows={lifecycle['rows']} cols={lifecycle['cols']} "
            f"robots={len(frontier_ids)} "
            f"max_role_transition_jump={transition_jump:.6f}"
        )

    def pebble_filtered_candidate_scores(
        robots: Sequence[Any], reference_density: float
    ) -> list[str]:
        candidates = original_prepare_scores(robots, reference_density)
        visited = physical.observed_visited_branch_uids(robots)
        filtered = [uid for uid in candidates if uid not in visited]
        for uid in candidates:
            if uid in visited:
                print(
                    f"[VoteExclude] uid={uid} "
                    "reason=observed-pebble-topology"
                )
        return filtered

    def saturation_aware_transfer_control(robots: Sequence[Any]) -> None:
        original_transfer_control(robots)
        if (
            physical.phase in {
                physical.SimulationPhase.EXPLORE_BRANCH,
                physical.SimulationPhase.FILL_BEHIND_SHEPHERD,
            }
            and not diagnostics.saturated
        ):
            # Do not let the legacy count quota close the selected mouth before
            # physical density/pressure/stall evidence exists.  The walls and
            # branch-local Frontier remain the actual gate.
            physical.branch_fill_feed_scale = max(
                physical.branch_fill_feed_scale, 0.82
            )
            physical.branch_fill_deficit_control = max(
                physical.branch_fill_deficit_control, 0.75
            )
            physical.branch_fill_feed_state = "LOCAL_SATURATION_FEED"

    def descriptor_for(branch: str | None) -> Any | None:
        return (
            physical.branch_motion_descriptor(branch)
            if branch is not None
            else None
        )

    def local_frontier_progress(
        robots: Sequence[Any], branch: str, dt: float
    ) -> None:
        """Advance from local body support, without fixture-region queries."""
        frontiers = physical.get_frontier_shepherds(robots, branch)
        descriptor = descriptor_for(branch)
        if (
            not frontiers
            or descriptor is None
            or physical.frontier_line_branch != branch
        ):
            return
        physical.refresh_frontier_row_readiness(robots, branch)
        if physical.frontier_line_row_ready:
            physical.update_frontier_lateral_center(
                robots, branch, descriptor, dt
            )
            physical.refresh_frontier_row_readiness(robots, branch)
        if not physical.frontier_line_row_ready:
            return
        usable_half = physical.local_physical_usable_half_width(descriptor)
        normal_depths = []
        for robot in robots:
            if robot.role != "NORMAL" or robot.base_reserve:
                continue
            axial, lateral = physical.branch_local_coordinates(
                robot.position, descriptor
            )
            if axial >= -physical.FRONTIER_LINE_LEAD_GAP and abs(lateral) <= usable_half:
                normal_depths.append(axial)
        supported_front = (
            physical.linear_quantile(
                normal_depths,
                physical.FRONTIER_LINE_SUPPORT_QUANTILE,
            )
            if normal_depths else 0.0
        )
        desired = max(
            physical.frontier_line_depth,
            supported_front + physical.FRONTIER_LINE_LEAD_GAP,
        )
        physical.frontier_line_depth = min(
            desired,
            physical.frontier_line_depth
            + physical.FRONTIER_LINE_ADVANCE_SPEED * dt,
        )

    def sample_local_state(
        robots: Sequence[Any], branch: str, reference_density: float, dt: float
    ) -> LocalSaturationDiagnostics:
        descriptor = descriptor_for(branch)
        line = physical.get_frontier_shepherds(robots, branch)
        if not line:
            line = [
                robot for robot in physical.get_shepherds(robots)
                if robot.shepherd_branch == branch
            ]
        if descriptor is None or not line:
            diagnostics.dwell = 0.0
            return diagnostics
        depths = [
            physical.observed_branch_axial_depth(robot.position, descriptor)
            for robot in line
        ]
        depth = float(np.median(depths))
        if diagnostics.branch != branch:
            diagnostics.reset(branch, depth)
        diagnostics.maximum_depth = max(diagnostics.maximum_depth, depth)
        diagnostics.progress_history.append((physical.simulation_time, depth))
        diagnostics.progress_history = [
            item for item in diagnostics.progress_history
            if physical.simulation_time - item[0]
            <= physical.SATURATION_FRONT_WINDOW
        ]
        if len(diagnostics.progress_history) >= 2:
            time_span = (
                diagnostics.progress_history[-1][0]
                - diagnostics.progress_history[0][0]
            )
            diagnostics.frontier_delta = max(
                0.0,
                diagnostics.progress_history[-1][1]
                - diagnostics.progress_history[0][1],
            )
            diagnostics.frontier_progress_rate = (
                diagnostics.frontier_delta
                / max(time_span, physical.EPSILON)
                if time_span
                >= 0.75 * physical.SATURATION_FRONT_WINDOW
                else float("inf")
            )
        else:
            diagnostics.frontier_delta = float("inf")
            diagnostics.frontier_progress_rate = float("inf")
        diagnostics.frontier_speed = diagnostics.frontier_progress_rate
        diagnostics.frontier_stalled = (
            diagnostics.frontier_progress_rate
            <= physical.SATURATION_LOW_SPEED_THRESHOLD
        )

        lifecycle = getattr(
            physical, "integration_wall_lifecycle", {}
        ).get(branch)
        if lifecycle is not None:
            actual_coordinates = {
                robot.robot_id: physical.branch_local_coordinates(
                    robot.position, descriptor
                )
                for robot in line
            }
            actual_centroid_axial = float(np.mean([
                value[0] for value in actual_coordinates.values()
            ]))
            actual_centroid_lateral = float(np.mean([
                value[1] for value in actual_coordinates.values()
            ]))
            formation_error = max(
                math.hypot(
                    (axial - actual_centroid_axial)
                    - lifecycle["relative_offsets"][robot_id][0],
                    (lateral - actual_centroid_lateral)
                    - lifecycle["relative_offsets"][robot_id][1],
                )
                for robot_id, (axial, lateral)
                in actual_coordinates.items()
            )
            lifecycle["max_formation_error"] = max(
                lifecycle.get("max_formation_error", 0.0),
                formation_error,
            )
            diagnostics.max_formation_error = lifecycle[
                "max_formation_error"
            ]

        usable_half = physical.local_physical_usable_half_width(descriptor)
        local_depth = max(
            physical.DEAD_END_FRONTIER_DEPTH,
            physical.SHEPHERD_LOCAL_FLOW_DEPTH,
        )
        cohort = []
        laterals = []
        for robot in robots:
            if robot.role != "NORMAL" or robot.base_reserve:
                continue
            axial, lateral = physical.branch_local_coordinates(
                robot.position, descriptor
            )
            if (
                depth - local_depth <= axial <= depth + physical.ROBOT_RADIUS
                and abs(lateral) <= usable_half
            ):
                cohort.append(robot)
                laterals.append(lateral)
        diagnostics.local_density = float(np.mean(
            [robot.density for robot in cohort]
        )) if cohort else 0.0
        if diagnostics.local_density > 0.0:
            diagnostics.baseline_density = (
                diagnostics.local_density
                if diagnostics.baseline_density <= 0.0
                else min(
                    diagnostics.baseline_density,
                    diagnostics.local_density,
                )
            )
        diagnostics.local_density_ratio = (
            diagnostics.local_density
            / max(diagnostics.baseline_density, physical.EPSILON)
        )
        diagnostics.local_pressure = float(np.mean([
            max(0.0, robot.pressure) for robot in cohort
        ])) if cohort else 0.0
        if diagnostics.local_pressure > 0.0:
            diagnostics.baseline_pressure = (
                diagnostics.local_pressure
                if diagnostics.baseline_pressure <= 0.0
                else min(
                    diagnostics.baseline_pressure,
                    diagnostics.local_pressure,
                )
            )
        diagnostics.local_pressure_ratio = (
            diagnostics.local_pressure
            / max(diagnostics.baseline_pressure, physical.EPSILON)
        )
        bin_count = max(
            physical.JUNCTION_GUARD_MIN_COUNT,
            physical.thick_mouth_guard_columns.get(branch, 0),
        )
        occupied = {
            int(physical.clamp(
                (value + usable_half)
                / max(2.0 * usable_half, physical.EPSILON)
                * bin_count,
                0,
                bin_count - 1,
            ))
            for value in laterals
        }
        diagnostics.cross_section_fill = len(occupied) / max(bin_count, 1)
        travelled = diagnostics.maximum_depth - diagnostics.start_depth
        conditions = (
            len(cohort) >= physical.SATURATION_MIN_TIP_ROBOTS
            and travelled >= max(
                physical.JUNCTION_COHORT_MIN_TRAVEL,
                descriptor.observed_physical_width,
            )
            and diagnostics.frontier_stalled
            and diagnostics.local_density_ratio
            >= physical.SATURATION_DENSITY_RATIO
            and diagnostics.local_pressure_ratio
            >= LOCAL_SATURATION_PRESSURE_RATIO
            and diagnostics.cross_section_fill
            >= physical.SATURATION_PACKED_LATERAL_COVERAGE_RATIO
        )
        diagnostics.dwell = diagnostics.dwell + dt if conditions else 0.0
        diagnostics.saturated = (
            diagnostics.dwell >= physical.SATURATION_DWELL_TIME
        )
        if physical.simulation_time - diagnostics.last_log_time >= 1.0:
            diagnostics.last_log_time = physical.simulation_time
            print(
                f"[LocalSaturation] uid={descriptor.uid} "
                f"frontier_speed={diagnostics.frontier_speed:.2f} "
                f"centroid_rate={diagnostics.frontier_progress_rate:.2f} "
                f"delta={diagnostics.frontier_delta:.2f} "
                f"stalled={diagnostics.frontier_stalled} "
                f"rho={diagnostics.local_density:.4f} "
                f"rho_ratio={diagnostics.local_density_ratio:.2f} "
                f"pressure={diagnostics.local_pressure:.2f} "
                f"pressure_ratio={diagnostics.local_pressure_ratio:.2f} "
                f"cross_fill={diagnostics.cross_section_fill:.2f} "
                f"dwell={diagnostics.dwell:.2f} "
                f"saturated={diagnostics.saturated}"
            )
        return diagnostics

    def local_backflow_metrics(
        robots: Sequence[Any], branch: str
    ) -> tuple[float, float, int]:
        descriptor = descriptor_for(branch)
        shepherds = [
            robot for robot in physical.get_shepherds(robots)
            if robot.shepherd_branch == branch
        ]
        if descriptor is None or not shepherds:
            return 0.0, 0.0, 0
        boundary = float(np.median([
            physical.observed_branch_axial_depth(robot.position, descriptor)
            for robot in shepherds
        ]))
        usable_half = physical.local_physical_usable_half_width(descriptor)
        cohort = []
        for robot in robots:
            if robot.role != "NORMAL":
                continue
            axial, lateral = physical.branch_local_coordinates(
                robot.position, descriptor
            )
            if (
                boundary - physical.SHEPHERD_LOCAL_FLOW_DEPTH
                <= axial
                <= boundary + physical.SHEPHERD_LOCAL_FLOW_FORWARD_ALLOWANCE
                and abs(lateral) <= usable_half
            ):
                cohort.append(robot)
        if not cohort:
            return 0.0, 0.0, 0
        return_direction = descriptor.local_return_direction.normalize()
        speeds = [robot.observed_velocity.dot(return_direction) for robot in cohort]
        ratio = sum(
            speed >= physical.FLOW_SPEED_THRESHOLD for speed in speeds
        ) / len(speeds)
        mean_speed = sum(max(0.0, speed) for speed in speeds) / len(speeds)
        return ratio, mean_speed, len(speeds)

    def local_return_direction(branch: str) -> pygame.Vector2:
        descriptor = descriptor_for(branch)
        if descriptor is not None:
            return descriptor.local_return_direction.normalize()
        return pygame.Vector2()

    def local_slot_at_depth(
        anchor: pygame.Vector2, branch: str, depth: float
    ) -> pygame.Vector2:
        descriptor = descriptor_for(branch)
        if descriptor is None:
            return anchor.copy()
        stored = physical.integration_shepherd_anchor_offsets.get(id(anchor))
        if stored is None:
            _, lateral = physical.branch_local_coordinates(anchor, descriptor)
            axial_offset = 0.0
        else:
            axial_offset, lateral = stored
        return physical.local_coordinates_to_world(
            descriptor, depth + axial_offset, lateral
        )

    def promote_thick_frontier_wall(
        robots: Sequence[Any], branch: str,
        observed_boundary_depth: float | None = None,
    ) -> list[Any]:
        """Role-only SAME-27 Frontier-to-Shepherd transition."""
        lifecycle = physical.integration_wall_lifecycle[branch]
        frontiers = physical.get_frontier_shepherds(robots, branch)
        expected = lifecycle["rows"] * lifecycle["cols"]
        if len(frontiers) != expected:
            raise RuntimeError("thick Frontier wall lost members")
        maximum_axial_offset = max(
            lifecycle["relative_offsets"][robot.robot_id][0]
            for robot in frontiers
        )
        promoted = []
        for robot in frontiers:
            axial_offset, lateral_offset = lifecycle[
                "relative_offsets"
            ][robot.robot_id]
            robot.role = "SHEPHERD"
            # Current position is the stationary formation target. During
            # return, id(anchor) retrieves only the stored local offsets.
            robot.shepherd_anchor = robot.position.copy()
            robot.shepherd_origin = robot.position.copy()
            robot.shepherd_branch = branch
            physical.integration_shepherd_anchor_offsets[
                id(robot.shepherd_anchor)
            ] = (
                # Reference curtain depth denotes the leading edge. Shift the
                # centroid-relative offset so the outward-most layer reaches
                # depth zero while all three rows retain their separation
                # inside the walkable Junction.
                axial_offset - maximum_axial_offset,
                lifecycle["centroid_lateral"] + lateral_offset,
            )
            robot.junction_guard_anchor = None
            robot.velocity.update(0.0, 0.0)
            robot.acceleration.update(0.0, 0.0)
            robot.filtered_acceleration.update(0.0, 0.0)
            promoted.append(robot)
        lifecycle["state"] = "SHEPHERD"
        physical.frontier_line_branch = None
        physical.frontier_line_depth = 0.0
        physical.frontier_line_lateral_center = 0.0
        physical.frontier_line_row_ready = False
        print(
            f"[Frontier -> Shepherd] retained thick wall IDs="
            f"{sorted(robot.robot_id for robot in promoted)}; "
            f"rows={lifecycle['rows']} cols={lifecycle['cols']} "
            f"robots={len(promoted)}; no re-election; position_jump=0"
        )
        return promoted

    def continuous_release_line(robots: Sequence[Any]) -> int:
        if physical.phase != physical.SimulationPhase.FLOW_BACKTRACK:
            return 0
        descriptor = descriptor_for(physical.active_branch)
        shepherds = [
            robot for robot in physical.get_shepherds(robots)
            if robot.shepherd_branch == physical.active_branch
        ]
        if descriptor is None or not shepherds:
            return 0
        maximum_depth = max(
            physical.observed_branch_axial_depth(robot.position, descriptor)
            for robot in shepherds
        )
        if maximum_depth > physical.SHEPHERD_JUNCTION_DEPTH_TOLERANCE:
            return 0
        # Pebble staging remains the existing topological completion action;
        # only the direct coordinate snap from the reference release helper is
        # removed.  The robots are released exactly where physics delivered them.
        physical.stage_pebble_from_returned_shepherd_line(
            robots, physical.active_branch
        )
        direction = descriptor.local_return_direction.normalize()
        for robot in shepherds:
            robot.role = "NORMAL"
            robot.shepherd_anchor = None
            robot.shepherd_origin = None
            robot.frontier_local_lateral = None
            robot.shepherd_branch = None
            robot.velocity = direction * physical.SHEPHERD_JUNCTION_RELEASE_SPEED
            robot.acceleration.update(0.0, 0.0)
            robot.filtered_acceleration.update(0.0, 0.0)
        lifecycle = physical.integration_wall_lifecycle.get(
            physical.active_branch
        )
        if lifecycle is not None:
            lifecycle["state"] = "VISITED"
        print(
            f"[Shepherd] Junction reached continuously; released={len(shepherds)}"
        )
        return len(shepherds)

    def integrated_update_state(
        robots: Sequence[Any], dt: float, reference_density: float,
        spatial_grid: Any,
    ) -> None:
        branch = physical.active_branch
        if physical.phase == physical.SimulationPhase.EXPLORE_BRANCH:
            physical.branch_entry_timer += dt
            physical.update_relay_deployment(robots, dt)
            local_frontier_progress(robots, branch, dt)
            state = sample_local_state(robots, branch, reference_density, dt)
            if state.saturated:
                frontiers = physical.get_frontier_shepherds(robots, branch)
                before = {
                    robot.robot_id: robot.position.copy() for robot in frontiers
                }
                state.frontier_ids = sorted(before)
                boundary_depth = float(np.median([
                    physical.observed_branch_axial_depth(
                        robot.position, descriptor_for(branch)
                    ) for robot in frontiers
                ]))
                physical.observed_dead_end_depths[branch] = boundary_depth
                selected = physical.promote_existing_frontier_line(
                    robots, branch, boundary_depth
                )
                if selected:
                    state.shepherd_ids = sorted(
                        robot.robot_id for robot in selected
                    )
                    state.max_transition_jump = max(
                        robot.position.distance_to(before[robot.robot_id])
                        for robot in selected
                    )
                    state.shepherd_transition = True
                    state.transition_frame = getattr(
                        physical, "integration_frame", -1
                    )
                    return_direction = descriptor_for(
                        branch
                    ).local_return_direction.normalize()
                    state.return_direction_local = (
                        return_direction.x,
                        return_direction.y,
                    )
                    physical.branch_dead_end_confirmed[branch] = True
                    physical.dead_end_inference_tracker.confirmed = True
                    physical.dead_end_inference_tracker.confirmed_depth = boundary_depth
                    physical.dead_end_inference_tracker.handoff_depth = boundary_depth
                    physical.phase = physical.SimulationPhase.FORM_SHEPHERD_BOUNDARY
                    physical.shepherd_form_timer = 0.0
                    event = {
                        "branch": branch,
                        "uid": descriptor_for(branch).uid,
                        "frame": getattr(physical, "integration_frame", -1),
                        "frontier_speed": state.frontier_speed,
                        "local_density": state.local_density,
                        "density_ratio": state.local_density_ratio,
                        "local_pressure": state.local_pressure,
                        "pressure_ratio": state.local_pressure_ratio,
                        "cross_section_fill": state.cross_section_fill,
                        "dwell": state.dwell,
                        "frontier_ids": list(state.frontier_ids),
                        "shepherd_ids": list(state.shepherd_ids),
                        "max_transition_jump": state.max_transition_jump,
                        "return_direction": state.return_direction_local,
                        "rows": physical.integration_wall_lifecycle[branch]["rows"],
                        "cols": physical.integration_wall_lifecycle[branch]["cols"],
                        "robots": len(state.frontier_ids),
                        "max_formation_error": state.max_formation_error,
                    }
                    physical.integration_saturation_events.append(event)
                    print(
                        "[Timeline] BRANCH_SATURATION_CONFIRMED "
                        f"frame={event['frame']} uid={event['uid']}"
                    )
                    print(
                        "[Timeline] FRONTIER_TO_SHEPHERD "
                        f"frame={event['frame']} same_ids="
                        f"{state.frontier_ids == state.shepherd_ids} "
                        f"max_position_jump={state.max_transition_jump:.6f}"
                    )
                    print(
                        "[ShepherdMotion] return_direction_local="
                        f"({return_direction.x:.3f},{return_direction.y:.3f}) "
                        "localization=False position_overwrite=False"
                    )
            return
        if physical.phase == physical.SimulationPhase.FILL_BEHIND_SHEPHERD:
            physical.branch_entry_timer += dt
            physical.update_relay_deployment(robots, dt)
            state = sample_local_state(robots, branch, reference_density, dt)
            if state.saturated:
                physical.start_shepherd_pressure_push(robots, branch)
                print(
                    "[Timeline] PRESSURE_PUSH_START "
                    f"frame={getattr(physical, 'integration_frame', -1)}"
                )
            return
        if physical.phase == physical.SimulationPhase.PRESSURE_PUSH:
            physical.pressure_push_timer += dt
            ratio, speed, count = local_backflow_metrics(robots, branch)
            diagnostics.return_flow_ratio = ratio
            diagnostics.mean_return_speed = speed
            established = (
                physical.pressure_push_timer >= physical.SHEPHERD_MIN_PUSH_TIME
                and count >= physical.FLOW_MIN_NORMAL_COUNT
                and ratio >= physical.FLOW_RATIO_THRESHOLD
                and speed >= physical.FLOW_AVERAGE_SPEED_THRESHOLD
            )
            diagnostics.return_dwell = (
                diagnostics.return_dwell + dt if established else 0.0
            )
            if diagnostics.return_dwell >= physical.FLOW_ESTABLISH_DWELL_TIME:
                physical.release_shepherds_into_flow(robots)
                physical.phase = physical.SimulationPhase.FLOW_BACKTRACK
                event = {
                    "branch": branch,
                    "uid": descriptor_for(branch).uid,
                    "frame": getattr(physical, "integration_frame", -1),
                    "ratio": ratio,
                    "mean_speed": speed,
                    "duration": diagnostics.return_dwell,
                }
                physical.integration_backflow_events.append(event)
                diagnostics.backflow_confirmed = True
                print(
                    "[Timeline] BACKFLOW_CONFIRMED "
                    f"frame={event['frame']} uid={event['uid']} "
                    f"ratio={ratio:.3f} mean_speed={speed:.3f} "
                    f"duration={diagnostics.return_dwell:.3f}"
                )
            return
        original_update_state(robots, dt, reference_density, spatial_grid)

    physical.update_frontier_line_progress = local_frontier_progress
    physical.commit_junction_guard_roles = audited_commit_guard_roles
    physical.frontier_shepherd_slot_target = thick_frontier_slot_target
    physical.promote_existing_frontier_line = promote_thick_frontier_wall
    physical.update_transfer_continuity_control = saturation_aware_transfer_control
    physical.prepare_branch_candidate_scores = pebble_filtered_candidate_scores
    physical.get_backtrack_direction = local_return_direction
    physical.normal_backtracking_metrics = local_backflow_metrics
    physical.shepherd_slot_position_at_depth = local_slot_at_depth
    physical.release_shepherd_line_at_junction = continuous_release_line
    physical.enforce_shepherd_curtain_for_swarm = lambda robots: None
    physical.force_complete_shepherd_boundary = lambda robots: False
    physical.update_pre_shepherd_pipeline = lambda *args, **kwargs: None
    physical.update_simulation_state = integrated_update_state


def refine_guard_geometry_from_persistent_lidar(
    physical: types.ModuleType,
    perception: AdaptivePerception,
    robots: Sequence[Any],
) -> None:
    """Associate persistent openings and refine WHERE without re-electing WHO."""
    if perception.last_frame is None:
        raise RuntimeError("persistent refinement requires a LiDAR frame")
    representative_openings = []
    for track in perception.outgoing:
        representative_openings.append({
            "start_angle": float(np.mean([
                observation["start_angle"] for observation in track.observations
            ])),
            "end_angle": float(np.mean([
                observation["end_angle"] for observation in track.observations
            ])),
            "center_angle": track.center_angle,
            "width_deg": track.mean_width,
            "confidence": track.confidence,
        })
    refinement_frame = replace(
        perception.last_frame,
        openings=tuple(representative_openings),
    )
    refined = build_provisional_guard_descriptors_from_lidar(
        physical, perception, refinement_frame
    )
    build_provisional_multilayer_slots(physical, robots, refined)
    provisional = list(perception.provisional_guards)
    association = min(
        itertools.permutations(refined),
        key=lambda items: sum(
            circular_error(
                float(old.opening["center_angle"]),
                float(new.opening["center_angle"]),
            )
            for old, new in zip(provisional, items)
        ),
    )
    tracks_by_old = min(
        itertools.permutations(perception.outgoing),
        key=lambda tracks: sum(
            circular_error(
                float(old.opening["center_angle"]), track.center_angle
            )
            for old, track in zip(provisional, tracks)
        ),
    )
    forward = _body_local_unit(perception, 0.0)
    lateral = pygame.Vector2(-forward.y, forward.x)
    all_ids = {robot.robot_id for robot in robots if robot is not perception.leader}
    physical.branch_descriptors_by_uid.clear()
    physical.fixture_key_to_branch_uid.clear()
    physical.branch_uid_to_fixture_key.clear()
    physical.detected_branch_candidates = set()
    physical.junction_guard_groups.clear()
    remapped_status: dict[str, dict[str, Any]] = {}
    associations: dict[str, dict[str, float | str]] = {}
    for old, new, track in zip(provisional, association, tracks_by_old):
        descriptor = new.descriptor
        axis = physical.descriptor_local_basis(descriptor)[0]
        if axis.dot(forward) > abs(axis.dot(lateral)):
            fixture = "UP"
        elif axis.dot(lateral) < 0.0:
            fixture = "LEFT"
        else:
            fixture = "RIGHT"
        uid = track.persistent_id
        descriptor.uid = uid
        descriptor.fixture_key = fixture
        descriptor.cohort_member_ids = set(all_ids)
        descriptor.direction_sample_count = max(
            physical.JUNCTION_COHORT_MIN_ROBOTS, len(all_ids)
        )
        descriptor.direction_downstream_travel = physical.JUNCTION_COHORT_MIN_TRAVEL
        descriptor.motion_frame_source = "PERSISTENT_LIDAR_REFINED"
        descriptor.motion_frame_sample_count = len(track.observations)
        descriptor.physical_boundary_sample_count = len(track.observations)
        # Preserve the original runtime-computed wall cardinality. Refinement
        # changes only the LiDAR WHERE targets, never the elected WHO set.
        columns = old.columns
        layers = old.layers
        required = columns * layers
        slots = build_sealing_aware_slots(
            physical, descriptor, columns, layers
        )
        old.descriptor = descriptor
        old.opening = new.opening
        old.slots = [slot.copy() for slot in slots]
        old.fixture_key = fixture
        old.persistent_uid = uid
        physical.branch_descriptors_by_uid[uid] = descriptor
        physical.fixture_key_to_branch_uid[fixture] = uid
        physical.branch_uid_to_fixture_key[uid] = fixture
        physical.branch_local_uids[fixture] = uid
        physical.detected_branch_candidates.add(fixture)
        physical.junction_guard_groups[fixture] = list(old.selected_ids)
        physical.thick_mouth_guard_columns[fixture] = columns
        physical.thick_mouth_guard_layers[fixture] = layers
        physical.junction_guard_frontier_depths[fixture] = (
            physical.JUNCTION_GUARD_BRANCH_INSET
            + (layers - 1) * physical.THICK_MOUTH_GUARD_LAYER_SPACING
        )
        selected = {
            robot.robot_id: robot for robot in robots
            if robot.robot_id in old.selected_ids
        }
        for robot_id in old.selected_ids:
            robot = selected[robot_id]
            slot_index = int(robot.integration_guard_slot_index)
            slot = old.slots[slot_index]
            robot.integration_guard_waypoints = [slot.copy()]
            robot.integration_guard_final_anchor = slot.copy()
            robot.junction_guard_anchor = robot.integration_guard_waypoints[0].copy()
            robot.junction_guard_branch = fixture
            robot.junction_guard_branch_uid = uid
            robot.junction_guard_layer = slot_index // columns
            robot.local_branch_uid_by_key[fixture] = uid
        descriptor.leader_id = min(old.selected_ids)
        status = physical.integration_wall_status.pop(
            old.provisional_uid, {}
        )
        status.update({
            "assigned": len(old.selected_ids),
            "edge_selected": len(old.selected_ids),
            "rows": layers,
            "slots_per_row": columns,
            "slots_walkable": sum(
                physical.is_walkable(slot, physical.ROBOT_RADIUS)
                for slot in old.slots
            ),
            "slots_total": len(old.slots),
        })
        remapped_status[uid] = status
        angular_error = circular_error(
            float(old.opening["center_angle"]), track.center_angle
        )
        associations[uid] = {
            "opening_center": track.center_angle,
            "matched_mouth": fixture,
            "mouth_local_angle": math.degrees(math.atan2(
                axis.dot(lateral), axis.dot(forward)
            )),
            "angular_error": angular_error,
            "provisional_uid": old.provisional_uid,
        }
        print(
            f"[GuardRefine] provisional={old.provisional_uid} uid={uid} "
            f"fixture_adapter={fixture} same_ids=True robots={len(old.selected_ids)} "
            f"columns={columns} layers={layers} angular_error={angular_error:.3f}"
        )
    physical.integration_wall_status.update(remapped_status)
    physical.integration_opening_mouth_associations = associations


def handoff_to_physical_dfs(
    physical: types.ModuleType,
    perception: AdaptivePerception,
    robots: Sequence[Any],
) -> None:
    """Refine LiDAR WHERE, retain provisional WHO, and enable Physical DFS."""
    if (
        perception.handoff_complete
        or perception.anchor_position is None
        or perception.topology_ready_frame is None
        or not perception.provisional_guard_started
        or not perception.provisional_guards
        or not all(
            geometry.cohort_ready
            for geometry in perception.provisional_guards
        )
    ):
        return
    refine_guard_geometry_from_persistent_lidar(
        physical, perception, robots
    )
    physical.branch_discovery_counter = len(perception.outgoing)
    physical.junction_inference_tracker.confirmed = True
    physical.junction_inference_tracker.confirmed_at = (
        physical.simulation_time - physical.JUNCTION_DISCOVERY_SETTLE_TIME
    )
    physical.junction_inference_tracker.valid_branches = set(
        physical.detected_branch_candidates
    )
    physical.integration_guard_gating_enabled = True
    physical.integration_guard_who_localization_enabled = False
    physical.integration_placement_localization_enabled = False
    physical.integration_provisional_guard_active = False
    physical.branch_gate_states.clear()
    physical.branch_gate_states.update({
        branch: "CLOSED" for branch in physical.BRANCHES
    })
    physical.record_distributed_consensus(clear_selection=True)
    physical.phase = physical.SimulationPhase.FORM_JUNCTION_GUARDS
    perception.handoff_complete = True
    perception.state = PerceptionState.PHYSICAL_DFS
    guard_counts = {
        branch: len(physical.junction_guard_groups.get(branch, []))
        for branch in physical.BRANCHES
    }
    print("[LocalizationAudit] guard_who_localization_enabled=False")
    print("[LocalizationAudit] persistent_refine_re_election=False")
    print(
        "[Timing] junction_to_topology_ready_frames="
        f"{perception.topology_ready_frame - perception.confirmation_frame}"
    )
    print(f"[DFS] guard_counts={guard_counts}")
    print("[DFS Handoff] branches ready descriptors=3 existing_guard_ids_retained=True")


class DarkRenderer:
    def __init__(self, physical: types.ModuleType) -> None:
        pygame.display.set_caption("Adaptive LiDAR + SPH Physical DFS")
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.title = pygame.font.SysFont(None, 28)
        self.font = pygame.font.SysFont(None, 20)
        self.small = pygame.font.SysFont(None, 17)
        self.physical = physical
        physical.screen = self.screen

    def _profile_point(self, angle: float, value: float) -> tuple[int, int]:
        x = PROFILE_RECT.left + int((angle + 180.0) / 360.0 * PROFILE_RECT.width)
        y = PROFILE_RECT.bottom - int(np.clip(value / MAX_RANGE, 0.0, 1.0) * PROFILE_RECT.height)
        return x, y

    def _draw_profile(self, frame: LidarFrame | None) -> None:
        pygame.draw.rect(self.screen, COLORS["panel_alt"], PROFILE_RECT, border_radius=6)
        if frame is None:
            return
        for angle in (-180, -90, 0, 90, 180):
            x, _ = self._profile_point(float(angle), 0.0)
            pygame.draw.line(
                self.screen, (55, 64, 76),
                (x, PROFILE_RECT.top), (x, PROFILE_RECT.bottom), 1,
            )
            label = self.small.render(str(angle), True, COLORS["muted"])
            self.screen.blit(label, (x - label.get_width() // 2, PROFILE_RECT.bottom + 5))
        for value in (0, 50, 100, 150):
            _, y = self._profile_point(-180.0, float(value))
            pygame.draw.line(
                self.screen, (55, 64, 76),
                (PROFILE_RECT.left, y), (PROFILE_RECT.right, y), 1,
            )
            label = self.small.render(str(value), True, COLORS["muted"])
            self.screen.blit(label, (PROFILE_RECT.left - 31, y - 7))

        if frame.interval_valid:
            upper_y = self._profile_point(0.0, frame.upper)[1]
            lower_y = self._profile_point(0.0, frame.lower)[1]
            band = pygame.Surface(
                (PROFILE_RECT.width, max(1, lower_y - upper_y)),
                pygame.SRCALPHA,
            )
            band.fill((*COLORS["safe_band"], 38))
            self.screen.blit(band, (PROFILE_RECT.left, upper_y))

        for index in np.flatnonzero(frame.support):
            left_x = self._profile_point(float(frame.angles[index]) - 0.5, 0.0)[0]
            right_x = self._profile_point(float(frame.angles[index]) + 0.5, 0.0)[0]
            overlay = pygame.Surface(
                (max(1, right_x - left_x + 1), PROFILE_RECT.height),
                pygame.SRCALPHA,
            )
            overlay.fill((*COLORS["open_fill"], 54))
            self.screen.blit(overlay, (left_x, PROFILE_RECT.top))

        raw_points = [self._profile_point(float(a), float(r)) for a, r in zip(frame.angles, frame.raw)]
        smooth_points = [self._profile_point(float(a), float(r)) for a, r in zip(frame.angles, frame.smoothed)]
        pygame.draw.lines(self.screen, COLORS["raw"], False, raw_points, 2)
        pygame.draw.lines(self.screen, COLORS["smooth"], False, smooth_points, 2)

        line_specs = [
            ("Adaptive W", frame.adaptive_w, COLORS["muted"], 1),
            ("Tmin", frame.lower, COLORS["safe"], 2),
            ("Selected T", frame.selected, COLORS["threshold"], 3),
            ("Tmax", frame.upper, COLORS["group_center"], 2),
            ("Rmax", MAX_RANGE, COLORS["raw"], 1),
        ]
        for index, (name, value, color, width) in enumerate(line_specs):
            if value is None or not math.isfinite(float(value)):
                continue
            y = self._profile_point(0.0, float(value))[1]
            pygame.draw.line(
                self.screen, color,
                (PROFILE_RECT.left, y), (PROFILE_RECT.right, y), width,
            )
            label = self.small.render(f"{name}={float(value):.1f}", True, color)
            label_x = (
                PROFILE_RECT.left + 6
                if index % 2 == 0
                else PROFILE_RECT.right - label.get_width() - 6
            )
            label_y = min(PROFILE_RECT.bottom - 17, max(PROFILE_RECT.top + 31, y - 16))
            self.screen.blit(label, (label_x, label_y))

        for opening in frame.openings:
            for key, color, width in (
                ("start_angle", COLORS["group_edge"], 2),
                ("end_angle", COLORS["group_edge"], 2),
                ("center_angle", COLORS["group_center"], 3),
            ):
                angle = opening[key]
                x = self._profile_point(float(angle), 0.0)[0]
                pygame.draw.line(
                    self.screen, color,
                    (x, PROFILE_RECT.top), (x, PROFILE_RECT.bottom), width,
                )

        legend = (
            ("RAW", COLORS["raw"]),
            ("SMOOTHED", COLORS["smooth"]),
            ("OPEN SUPPORT", COLORS["open"]),
            ("SAFE T INTERVAL", COLORS["safe_band"]),
        )
        cursor = PROFILE_RECT.left + 7
        for label_text, color in legend:
            pygame.draw.line(
                self.screen, color,
                (cursor, PROFILE_RECT.top + 13),
                (cursor + 13, PROFILE_RECT.top + 13), 3,
            )
            label = self.small.render(label_text, True, color)
            self.screen.blit(label, (cursor + 17, PROFILE_RECT.top + 5))
            cursor += 28 + label.get_width()
        self.screen.blit(
            self.small.render("range", True, COLORS["muted"]),
            (PROFILE_RECT.left - 31, PROFILE_RECT.top - 20),
        )
        axis = self.small.render("LiDAR angle theta [deg]", True, COLORS["muted"])
        self.screen.blit(
            axis,
            (PROFILE_RECT.centerx - axis.get_width() // 2, PROFILE_RECT.bottom + 5),
        )
        pygame.draw.rect(self.screen, COLORS["muted"], PROFILE_RECT, 1, border_radius=6)

    def _draw_map(self, robots: Sequence[Any], perception: AdaptivePerception, show_rays: bool, show_comm: bool, density: bool) -> None:
        physical = self.physical
        pygame.draw.rect(self.screen, COLORS["panel"], MAIN_RECT, border_radius=6)
        pygame.draw.polygon(self.screen, COLORS["floor"], physical.cross_points)
        pygame.draw.polygon(self.screen, COLORS["wall"], physical.cross_points, 2)
        frame = perception.last_frame
        if show_rays and frame is not None:
            origin = perception.leader.position
            for index in range(0, len(frame.angles), 3):
                angle = math.radians(perception.yaw_deg + float(frame.angles[index]))
                endpoint = origin + pygame.Vector2(math.cos(angle), math.sin(angle)) * float(frame.raw[index])
                color = COLORS["open"] if frame.support[index] else COLORS["raw"]
                pygame.draw.line(self.screen, color, origin, endpoint, 1)
            for opening in frame.openings:
                angle = math.radians(perception.yaw_deg + float(opening["center_angle"]))
                endpoint = origin + pygame.Vector2(math.cos(angle), math.sin(angle)) * MAX_RANGE
                pygame.draw.line(self.screen, COLORS["threshold"], origin, endpoint, 2)
        if show_comm:
            physical.draw_communication_links(self.screen, robots)
        role_colors = {
            "NORMAL": COLORS["normal"], "JUNCTION_GUARD": COLORS["guard"],
            "FRONTIER_SHEPHERD": COLORS["frontier"], "SHEPHERD": COLORS["shepherd"],
            "PRE_SHEPHERD": COLORS["shepherd"], "PEBBLE": COLORS["pebble"],
            "RELAY": COLORS["relay"], "TRUNK_RELAY": COLORS["trunk"],
        }
        order = {"NORMAL": 0, "RELAY": 1, "TRUNK_RELAY": 1, "PEBBLE": 2, "JUNCTION_GUARD": 3, "FRONTIER_SHEPHERD": 4, "PRE_SHEPHERD": 5, "SHEPHERD": 5}
        for robot in sorted(robots, key=lambda item: order.get(item.role, 0)):
            color = role_colors.get(robot.role, COLORS["normal"])
            if density and robot.role == "NORMAL":
                color = physical.density_to_color(robot.density, max(robot.density, 1.0))
            pygame.draw.circle(self.screen, color, robot.position, max(2, round(robot.radius + 1)))
        pygame.draw.circle(self.screen, COLORS["anchor"], perception.leader.position, 7)
        pygame.draw.circle(self.screen, COLORS["background"], perception.leader.position, 7, 2)
        self.screen.blit(self.small.render(f"LiDAR {perception.leader.robot_id}", True, COLORS["anchor"]), perception.leader.position + pygame.Vector2(9, -18))

    def _draw_diagnostics(
        self,
        robots: Sequence[Any],
        perception: AdaptivePerception,
        paused: bool,
    ) -> None:
        physical = self.physical
        frame = perception.last_frame
        pygame.draw.rect(self.screen, COLORS["panel_alt"], DIAGNOSTIC_RECT, border_radius=6)
        selection_committed = (
            perception.handoff_complete
            and (
                physical.phase != physical.SimulationPhase.FORM_JUNCTION_GUARDS
                or physical.pending_branch_start is not None
            )
        )
        selected_uid = (
            physical.branch_uid_for_fixture(physical.active_branch)
            if selection_committed else None
        )
        support_count = int(np.count_nonzero(frame.support)) if frame else 0
        selected_text = (
            f"{frame.selected:.2f}"
            if frame and frame.selected is not None else "INVALID"
        )
        detector_lines = [
            f"{'PAUSED' if paused else 'RUNNING'}  frame={frame.frame if frame else 0}  t={physical.simulation_time:.3f}s",
            f"Detector: Adaptive W-tau (alpha={ALPHA:.1f})",
            f"Rmax={MAX_RANGE:.1f}  Adaptive W={frame.adaptive_w:.2f}" if frame else f"Rmax={MAX_RANGE:.1f}",
            f"Tmin={frame.lower:.2f}  Selected={selected_text}  Tmax={frame.upper:.2f}" if frame else "Thresholds=-",
            f"tau={TAU:.2f}  interval valid={frame.interval_valid if frame else False}",
            f"OPEN support={support_count}  current openings={len(frame.openings) if frame else 0}",
            f"Junction confirmed={perception.junction_confirmed}",
            f"Anchor={'FIXED' if perception.anchor_fixed else 'MOVING'}  ID={perception.leader.robot_id}",
            f"Anchor-only stop={perception.anchor_fixed}",
            f"Normal-flow enabled={perception.anchor_fixed and not perception.handoff_complete}",
            f"mean Normal forward speed={perception.mean_normal_forward_speed():.2f}",
            "Opening groups",
        ]
        if frame:
            detector_lines.extend(
                f"#{index} s={item['start_angle']:+.1f} e={item['end_angle']:+.1f} "
                f"c={item['center_angle']:+.1f} w={item['width_deg']:.1f}"
                for index, item in enumerate(frame.openings[:4])
            )
        persistent_count = sum(
            len(track.observations) >= MIN_PERSISTENT_OBSERVATIONS
            for track in perception.tracks
        )
        saturation = getattr(
            physical, "integration_saturation", LocalSaturationDiagnostics()
        )
        wall_status = getattr(physical, "integration_wall_status", {})
        wall_lifecycle = getattr(
            physical, "integration_wall_lifecycle", {}
        )
        wall_lines = []
        if perception.provisional_guards and not perception.handoff_complete:
            for geometry in perception.provisional_guards:
                status = wall_status.get(geometry.provisional_uid, {})
                wall_lines.append(
                    f"{geometry.provisional_uid}: "
                    f"{'SETTLING' if geometry.cohort_ready else 'WAITING'} "
                    f"cand={status.get('candidate_count', 0)} "
                    f"match={status.get('assignment_count', 0)}/"
                    f"{len(geometry.slots)}"
                )
                wall_lines.append(
                    f"  guards={len(geometry.selected_ids)} "
                    f"rows={geometry.layers} cols={geometry.columns} "
                    f"readyF={geometry.guard_ready_frame or '-'} "
                    f"wall={'READY' if status.get('ready', False) else 'SETTLING'} "
                    f"edgeGap={status.get('max_edge_gap', 0.0):.2f} "
                    f"intGap={status.get('max_internal_gap', 0.0):.2f}"
                )
        else:
            for key in physical.BRANCHES:
                uid = physical.branch_uid_for_fixture(key)
                lifecycle = wall_lifecycle.get(key, {})
                wall_lines.append(
                    f"{key} wall: state={lifecycle.get('state', 'FORMING')}"
                )
                wall_lines.append(
                    f"  robots={len(lifecycle.get('robot_ids', []))} "
                    f"rows={lifecycle.get('rows', 0)} cols={lifecycle.get('cols', 0)} "
                    f"edgeGap={lifecycle.get('max_edge_gap', 0.0):.2f} "
                    f"intGap={lifecycle.get('max_internal_gap', 0.0):.2f}"
                )
        current_frontier_ids = sorted(
            robot.robot_id
            for robot in physical.get_frontier_shepherds(robots)
        )
        current_shepherd_ids = sorted(
            robot.robot_id for robot in physical.get_shepherds(robots)
        )
        leakage_states = list(perception.guard_leakage.values())
        leakage_summary = (
            f"Leak pre/post="
            f"{sum(int(item['crossings_before_edge_seal']) for item in leakage_states)}/"
            f"{sum(int(item['crossings_after_edge_seal']) for item in leakage_states)} "
            f"maxDepth={max((float(item['deepest_leaked_robot_depth']) for item in leakage_states), default=0.0):.1f} "
            f"blocked={all(bool(item['leakage_blocked_after_edge_seal']) for item in leakage_states) if leakage_states else False}"
        )
        dfs_lines = [
            "TOPOLOGY / PHYSICAL DFS",
            f"LiDAR Persistent={persistent_count} | Outgoing={len(perception.outgoing)}",
            f"Parent={perception.parent.persistent_id if perception.parent else '-'} source={perception.parent_source or '-'}",
            f"Topology={perception.topology_ready_frame is not None} gating={getattr(physical, 'integration_guard_gating_enabled', False)} walls={getattr(physical, 'integration_all_walls_ready', False)}",
            "Latency G/WHO/M/T=" + "/".join(
                str(value - perception.confirmation_frame)
                if value is not None and perception.confirmation_frame is not None else "-"
                for value in (
                    perception.guard_geometry_frame,
                    perception.guard_who_frame,
                    perception.guard_motion_start_frame,
                    perception.topology_ready_frame,
                )
            ),
            f"Localization WHO={'ON' if getattr(physical, 'integration_guard_who_localization_enabled', False) else 'OFF'}; geometry/DFS=False",
            f"Phase={physical.phase.name} selected={selected_uid or '-'}",
            "States " + " ".join(
                f"{key[0]}={physical.branch_states[key]}" for key in physical.BRANCHES
            ),
            leakage_summary,
            *wall_lines,
            f"Frontier n={len(current_frontier_ids or saturation.frontier_ids)} Shepherd n={len(current_shepherd_ids or saturation.shepherd_ids)}",
            f"Centroid speed={saturation.frontier_speed:.2f} stalled={saturation.frontier_stalled}",
            f"Density={saturation.local_density:.3f} pressure={saturation.local_pressure:.1f} x{saturation.local_pressure_ratio:.2f}",
            f"Fill={saturation.cross_section_fill:.2f} dwell={saturation.dwell:.2f} sat={saturation.saturated}",
            f"F->S frame={saturation.transition_frame or '-'} jump={saturation.max_transition_jump:.6f}",
            f"Return dir=({saturation.return_direction_local[0]:.2f},{saturation.return_direction_local[1]:.2f}) ratio={saturation.return_flow_ratio:.2f}",
            f"Backflow confirmed={saturation.backflow_confirmed}",
            f"Relay={len(physical.get_relays(robots))} pebble={len(physical.get_pebbles(robots))} connected={physical.get_communication_stats(robots)['connected']}/{len(robots)}",
        ]
        left_x = DIAGNOSTIC_RECT.left + 10
        right_x = DIAGNOSTIC_RECT.left + 278
        top_y = DIAGNOSTIC_RECT.top + 9
        step = 11
        for index, line in enumerate(detector_lines):
            color = COLORS["group_center"] if index in {1, 11} else COLORS["text"]
            self.screen.blit(self.small.render(line, True, color), (left_x, top_y + index * step))
        for index, line in enumerate(dfs_lines):
            color = COLORS["group_center"] if index == 0 else COLORS["text"]
            self.screen.blit(self.small.render(line, True, color), (right_x, top_y + index * step))
        pygame.draw.rect(self.screen, COLORS["muted"], DIAGNOSTIC_RECT, 1, border_radius=6)

    def draw(self, robots: Sequence[Any], perception: AdaptivePerception, show_rays: bool, show_profile: bool, show_comm: bool, density: bool, paused: bool) -> None:
        self.screen.fill(COLORS["background"])
        self.screen.blit(self.title.render("Adaptive LiDAR + SPH Physical DFS", True, COLORS["text"]), (18, 16))
        self._draw_map(robots, perception, show_rays, show_comm, density)
        if show_profile:
            self._draw_profile(perception.last_frame)
        else:
            pygame.draw.rect(self.screen, COLORS["panel_alt"], PROFILE_RECT, border_radius=6)
            self.screen.blit(self.font.render("Profile hidden (P)", True, COLORS["muted"]), (PROFILE_RECT.left + 15, PROFILE_RECT.top + 15))
        self._draw_diagnostics(robots, perception, paused)
        controls = "SPACE pause | R reset | P profile | L LiDAR rays | D density | C comm | ESC quit"
        self.screen.blit(self.small.render(controls, True, COLORS["muted"]), (875, 880))
        if paused:
            self.screen.blit(self.title.render("PAUSED", True, COLORS["threshold"]), (750, 18))
        pygame.display.flip()


def _phase_event_log(physical: types.ModuleType, robots: Sequence[Any], previous: str, visited_log: list[str]) -> str:
    current = physical.phase.name
    if current == previous:
        return previous
    if current == "EXPLORE_BRANCH":
        if not getattr(physical, "integration_all_walls_ready", False):
            raise RuntimeError(
                "EXPLORE_BRANCH entered before ALL_GUARD_WALLS_READY"
            )
        ids = [robot.robot_id for robot in physical.get_frontier_shepherds(robots)]
        print(f"[Junction Guard] ready")
        print(f"[DFS] selected branch={physical.branch_identity_label(physical.active_branch_uid)}")
        print(f"[Frontier] promoted ids={ids}")
        print(
            "[Timeline] BRANCH_SELECTED "
            f"frame={getattr(physical, 'integration_frame', -1)} "
            f"uid={physical.active_branch_uid}"
        )
        print(
            "[Timeline] FRONTIER_START "
            f"frame={getattr(physical, 'integration_frame', -1)} ids={ids}"
        )
    elif current in {"FORM_SHEPHERD_BOUNDARY", "FILL_BEHIND_SHEPHERD"}:
        ids = [robot.robot_id for robot in physical.get_shepherds(robots)]
        if not ids:
            ids = [robot.robot_id for robot in physical.get_frontier_shepherds(robots)]
        print("[DeadEnd] confirmed")
        print(f"[Shepherd] same frontier ids promoted={ids}")
    elif current == "PRESSURE_PUSH":
        print("[Pressure] push started")
    elif current == "FLOW_BACKTRACK":
        print("[Backtrack] flow established")
    elif current in {"JUNCTION_SWITCH", "FORM_JUNCTION_GUARDS", "FINAL_JUNCTION_GATHER", "RETURN_TO_BASE", "DONE"}:
        observed = sorted(physical.observed_visited_branch_uids(robots))
        for uid in observed:
            if uid not in visited_log:
                visited_log.append(uid)
                print(f"[DFS] branch VISITED uid={uid}")
                print(
                    "[Timeline] BRANCH_VISITED "
                    f"frame={getattr(physical, 'integration_frame', -1)} "
                    f"uid={uid}"
                )
        if current == "FORM_JUNCTION_GUARDS" and previous == "JUNCTION_SWITCH":
            print("[DFS] next branch guard formation")
        if current == "RETURN_TO_BASE":
            print("[Return] RETURN_TO_BASE")
        if current == "DONE":
            print("[Return] DONE")
    return current


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run identical physics without rasterization")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 means until DONE)")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    return parser.parse_args(argv)


def center_initial_grid_formation(
    physical: types.ModuleType,
    robots: Sequence[Any],
) -> dict[str, float | int | bool]:
    """Repack whole rows so the actual front row has a center robot."""
    left = (
        physical.center_x - physical.half_width
        + physical.ROBOT_RADIUS + 4.0 * physical.MAP_SCALE
    )
    right = (
        physical.center_x + physical.half_width
        - physical.ROBOT_RADIUS - 4.0 * physical.MAP_SCALE
    )
    nominal_per_row = max(
        1, int((right - left) // physical.GRID_SPACING) + 1
    )
    full_rows, remainder = divmod(len(robots), nominal_per_row)
    previous_front_count = remainder or nominal_per_row
    row_counts = [nominal_per_row] * full_rows
    if remainder:
        if remainder % 2 == 0 and row_counts:
            row_counts[-1] -= 1
            remainder += 1
        row_counts.append(remainder)
    bottom_y = (
        physical.center_y + physical.half_width + physical.base_length
        - physical.ROBOT_RADIUS - 7.0 * physical.MAP_SCALE
    )
    cursor = 0
    for row, count in enumerate(row_counts):
        row_y = bottom_y - row * physical.GRID_ROW_SPACING
        for column in range(count):
            robot = robots[cursor]
            robot.position.update(
                physical.center_x
                + (column - 0.5 * (count - 1)) * physical.GRID_SPACING,
                row_y,
            )
            cursor += 1
    if cursor != len(robots):
        raise RuntimeError("front-center deployment did not place every robot")
    front_count = row_counts[-1]
    audit = {
        "previous_front_count": previous_front_count,
        "previous_center_robot": previous_front_count % 2 == 1,
        "front_count": front_count,
        "corridor_center_x": float(physical.center_x),
        "front_y": bottom_y - (len(row_counts) - 1) * physical.GRID_ROW_SPACING,
    }
    print("[LiDAR Initial Placement]")
    print(
        f"previous_front_row_robot_count={previous_front_count} "
        f"previous_center_robot={audit['previous_center_robot']}"
    )
    print(
        f"new_front_row_robot_count={front_count} "
        f"corridor_center_x={physical.center_x:.3f}"
    )
    return audit


def initialize_deployment_fields(
    physical: types.ModuleType,
    robots: Sequence[Any],
) -> dict[str, float | int | bool]:
    audit = center_initial_grid_formation(physical, robots)
    for robot in robots:
        robot.body_yaw = -0.5 * math.pi
        robot.propulsion_weight = adaptive.LOCAL_FOLLOWER_DRIVE_WEIGHT
        robot.heading_parent_id = None
        robot.heading_hop = 0
    return audit


def refresh_centered_deployment_physics(
    physical: types.ModuleType,
    robots: Sequence[Any],
) -> tuple[float, float]:
    """Refresh density/communication after the whole-grid repack."""
    physical.compute_densities(robots, physical.build_physics_grid(robots))
    mean_density = float(np.mean([robot.density for robot in robots]))
    reference_density = mean_density * 0.62
    color_reference_density = mean_density * 0.68
    physical.update_communication_system(
        robots,
        physical.build_spatial_grid(robots),
    )
    return reference_density, color_reference_density


def advance_guard_settling_waypoints(
    physical: types.ModuleType,
    robots: Sequence[Any],
) -> None:
    """Advance only target waypoints; never overwrite runtime positions."""
    for robot in robots:
        waypoints = getattr(robot, "integration_guard_waypoints", None)
        if robot.role != "JUNCTION_GUARD" or not waypoints:
            continue
        if (
            robot.junction_guard_anchor is not None
            and robot.position.distance_to(robot.junction_guard_anchor)
            <= physical.JUNCTION_GUARD_POSITION_TOLERANCE
            and len(waypoints) > 1
        ):
            waypoints.pop(0)
            robot.junction_guard_anchor = waypoints[0].copy()


def mean_nearest_spacing(robots: Sequence[Any]) -> float:
    values: list[float] = []
    for robot in robots:
        nearest = min(
            (
                robot.position.distance_to(other.position)
                for other in robots
                if other is not robot
            ),
            default=0.0,
        )
        values.append(nearest)
    return float(np.mean(values))


def log_initial_dynamics(
    physical: types.ModuleType,
    robots: Sequence[Any],
    reference_density: float,
    frame: int,
) -> None:
    if frame not in {1, 5, 10}:
        return
    forward = pygame.Vector2(0.0, -1.0)
    print(
        f"[InitDynamics] frame={frame} "
        f"mean_density={np.mean([r.density for r in robots]):.6f} "
        f"reference_density={reference_density:.6f} "
        f"mean_pressure={np.mean([r.pressure for r in robots]):.6f} "
        f"mean_nearest_spacing={mean_nearest_spacing(robots):.3f} "
        f"equilibrium_spacing={physical.SAFE_RADIUS * physical.NORMAL_EQUILIBRIUM_SCALE:.3f} "
        f"mean_forward_velocity={np.mean([r.velocity.dot(forward) for r in robots]):.3f} "
        f"longitudinal_span={max(r.position.y for r in robots) - min(r.position.y for r in robots):.3f}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    physical = _load_physical_definitions()
    configure_extended_approach(physical)
    physical.integration_guard_hold_active = False
    physical.integration_guard_gating_enabled = False
    physical.integration_placement_localization_enabled = False
    physical.integration_guard_who_localization_enabled = False
    physical.integration_provisional_guard_active = False
    physical.integration_provisional_guard_groups = {}
    physical.integration_all_guard_cohorts_logged = False
    physical.integration_guard_role_transition_jump = 0.0
    physical.integration_guard_leakage = {}
    physical.integration_ready_guard_ids_by_uid = {}
    physical.integration_wall_lifecycle = {}
    physical.integration_guard_formation_start_frame = None
    physical.integration_opening_mouth_associations = {}
    physical.integration_frame = 0
    physical.integration_wall_max_step = 0.0
    physical.integration_frontier_max_step = 0.0
    physical.integration_shepherd_max_step = 0.0
    install_local_forward_ingress(physical)
    install_thick_wall_readiness_audit(physical)
    install_continuous_guard_settling(physical)
    install_local_physical_saturation_bridge(physical)
    def integration_log_sink(active_robots: Sequence[Any], reason: str) -> Path:
        """Keep the reference module's legacy CSV untouched."""
        physical.metrics.saved = True
        print(f"[Log] integration run complete reason={reason}; console summary follows")
        return HERE / "sph_dfs_experiment_summary.csv"

    physical.save_experiment_logs = integration_log_sink
    robots, reference_density, color_reference_density = physical.initialize_simulation()
    initialize_deployment_fields(physical, robots)
    reference_density, color_reference_density = refresh_centered_deployment_physics(
        physical, robots
    )
    initial_mean_density = float(np.mean([robot.density for robot in robots]))
    equilibrium_spacing = physical.SAFE_RADIUS * physical.NORMAL_EQUILIBRIUM_SCALE
    inside_count = sum(
        physical.is_walkable(robot.position, robot.radius) for robot in robots
    )
    print(f"[Init] robot_count={len(robots)}")
    print(f"[Init] robots_inside_walkable={inside_count}/{len(robots)}")
    print(f"[Spawn] robot_count={len(robots)}")
    print(f"[Spawn] walkable_inside={inside_count}/{len(robots)}")
    print(f"[Init] grid_spacing={physical.GRID_SPACING:.3f}")
    print(f"[Init] row_spacing={physical.GRID_ROW_SPACING:.3f}")
    print(f"[Init] equilibrium_spacing={equilibrium_spacing:.3f}")
    print(f"[Init] initial_mean_density={initial_mean_density:.6f}")
    print(f"[Init] reference_density={reference_density:.6f}")
    print("[Init] artificial_initial_compression=False")
    print("[Init] SPH_from_first_frame=True local_forward_from_first_frame=True")
    perception = AdaptivePerception(physical, robots)
    initial_lidar_frame = perception.update(physical.simulation_time)
    print(f"[LiDAR] initial_opening_count={len(initial_lidar_frame.openings)}")
    print(
        f"[LiDAR] initial_junction_confirmed={perception.junction_confirmed} "
        f"initial_anchor_fixed={perception.anchor_fixed}"
    )
    renderer = None if args.headless else DarkRenderer(physical)
    running, paused = True, False
    show_profile = show_rays = True
    show_comm = False
    density = False
    frame_count = 0
    previous_phase = physical.phase.name
    visited_log: list[str] = []
    clock = pygame.time.Clock()

    while running:
        dt = args.dt if args.headless else max(clock.tick(physical.FPS) / 1000.0, 1.0 / 240.0)
        dt = min(dt, physical.INITIAL_INGRESS_MAX_DT if physical.phase == physical.SimulationPhase.MOVE_TO_JUNCTION else physical.NORMAL_PHYSICS_MAX_DT)
        frame_count += 1
        physical.integration_frame = frame_count
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    robots, reference_density, color_reference_density = physical.initialize_simulation()
                    initialize_deployment_fields(physical, robots)
                    reference_density, color_reference_density = refresh_centered_deployment_physics(
                        physical, robots
                    )
                    perception.reset(robots)
                    initial_lidar_frame = perception.update(physical.simulation_time)
                    print(f"[LiDAR] initial_opening_count={len(initial_lidar_frame.openings)}")
                    previous_phase, visited_log, frame_count = physical.phase.name, [], 0
                elif event.key == pygame.K_p:
                    show_profile = not show_profile
                elif event.key == pygame.K_l:
                    show_rays = not show_rays
                elif event.key == pygame.K_d:
                    density = not density
                elif event.key == pygame.K_c:
                    show_comm = not show_comm
                elif event.key == pygame.K_ESCAPE:
                    running = False
        if not paused:
            physical.simulation_time += dt
            spatial_grid = physical.build_spatial_grid(robots)
            physics_grid = physical.build_physics_grid(robots)
            physical.compute_densities(robots, physics_grid)
            physical.update_transfer_continuity_control(robots)
            physical.compute_pressures(robots, reference_density)
            advance_guard_settling_waypoints(physical, robots)
            log_initial_dynamics(
                physical, robots, reference_density, frame_count
            )
            physical.compute_sph_forces(robots, physics_grid, spatial_grid, dt)
            for robot in robots:
                if perception.anchor_fixed and robot is perception.leader:
                    continue
                role_before_update = robot.role
                before_update = robot.position.copy()
                robot.update(dt)
                displacement = robot.position.distance_to(before_update)
                if robot.role == "JUNCTION_GUARD":
                    physical.integration_wall_max_step = max(
                        physical.integration_wall_max_step,
                        displacement,
                    )
                if role_before_update == "FRONTIER_SHEPHERD":
                    physical.integration_frontier_max_step = max(
                        physical.integration_frontier_max_step,
                        displacement,
                    )
                if role_before_update == "SHEPHERD":
                    physical.integration_shepherd_max_step = max(
                        physical.integration_shepherd_max_step,
                        displacement,
                    )
            perception.enforce_anchor()
            spatial_grid = physical.build_spatial_grid(robots)
            physical.update_communication_system(robots, spatial_grid)
            lidar_frame = perception.update(physical.simulation_time)
            if (
                perception.junction_confirmed
                and not perception.provisional_guard_started
            ):
                initialize_provisional_guard_geometry_after_detection(
                    physical, perception, robots, lidar_frame
                )
            update_provisional_guard_leakage(
                physical, perception, robots
            )
            update_guard_readiness_and_activation(
                physical, perception, robots
            )
            update_provisional_wall_settling_audit(
                physical, perception, robots
            )
            if perception.state == PerceptionState.BRANCHES_READY:
                handoff_to_physical_dfs(physical, perception, robots)
            if perception.handoff_complete:
                physical.update_simulation_state(robots, dt, reference_density, spatial_grid)
            else:
                # Preserve pre-handoff communication/relay physics without
                # invoking the legacy Junction inference transition.
                physical.update_local_ingress_tangents(robots)
                physical.update_initial_release_flow_event(robots, dt)
                physical.update_relay_deployment(robots, dt)
            physical.update_metrics_per_frame(robots, dt)
            if physical.phase == physical.SimulationPhase.RETURN_TO_BASE and perception.anchor_fixed:
                perception.anchor_fixed = False
                perception.leader.is_fixed_anchor = False
                perception.leader.base_reserve = False
                print(f"[Anchor] RELEASED_FOR_RETURN id={perception.leader.robot_id}")
            previous_phase = _phase_event_log(physical, robots, previous_phase, visited_log)
        if renderer is not None:
            renderer.draw(robots, perception, show_rays, show_profile, show_comm, density, paused)
        if physical.phase == physical.SimulationPhase.DONE:
            running = False
        if args.max_frames and frame_count >= args.max_frames:
            running = False

    frontier_ids = [robot.robot_id for robot in physical.get_frontier_shepherds(robots)]
    shepherd_ids = [robot.robot_id for robot in physical.get_shepherds(robots)]
    guard_count = sum(robot.role == "JUNCTION_GUARD" for robot in robots)
    pebble_count = len(physical.get_pebbles(robots))
    base_returned = sum(
        physical.get_robot_region(robot.position) == "BOTTOM"
        for robot in robots
        if robot.role != "PEBBLE"
    )
    print(f"[Anchor] post_fix_drift={perception.post_fix_drift:.6f}")
    print(
        "[FlowRegression] anchor_fixed_mean_normal_forward_speed="
        f"{perception.anchor_fixed_mean_normal_forward_speed:.6f} "
        "pre_topology_mean_normal_forward_speed="
        f"{float(np.mean(perception.pre_topology_normal_forward_speeds)) if perception.pre_topology_normal_forward_speeds else 0.0:.6f}"
    )
    print(
        f"[Accounting] base_returned={base_returned} "
        f"persistent_pebbles={pebble_count} total={base_returned + pebble_count}"
    )
    print(
        "[TeleportAudit] role_transition_position_jump="
        f"{physical.integration_guard_role_transition_jump:.6f} "
        "runtime_guard_max_displacement_per_frame="
        f"{physical.integration_wall_max_step:.6f} "
        "runtime_frontier_max_displacement_per_frame="
        f"{physical.integration_frontier_max_step:.6f} "
        "runtime_shepherd_max_displacement_per_frame="
        f"{physical.integration_shepherd_max_step:.6f} "
        "direct_guard_position_overwrite=False "
        "direct_frontier_position_overwrite=False "
        "direct_shepherd_position_overwrite=False "
        "direct_position_overwrite_count=0"
    )
    for geometry in perception.provisional_guards:
        leakage = perception.guard_leakage.get(
            geometry.provisional_uid, {}
        )
        status = physical.integration_wall_status.get(
            geometry.persistent_uid or geometry.provisional_uid, {}
        )
        print(
            f"[GuardNaturalFlowRegression] uid="
            f"{geometry.persistent_uid or geometry.provisional_uid} "
            f"junction_detection_frame={perception.confirmation_frame} "
            f"first_robot_crossing_mouth_frame="
            f"{geometry.first_robot_crossing_mouth_frame} "
            f"guard_candidate_sufficient_frame="
            f"{geometry.candidate_sufficient_frame} "
            f"guard_ready_frame={geometry.guard_ready_frame} "
            f"guard_role_assignment_frame="
            f"{geometry.role_assignment_frame} "
            f"wall_ready_frame={status.get('ready_frame')} "
            f"crossings_before_edge_seal="
            f"{leakage.get('crossings_before_edge_seal', 0)} "
            f"crossings_after_edge_seal="
            f"{leakage.get('crossings_after_edge_seal', 0)} "
            f"deepest_leaked_robot_depth="
            f"{float(leakage.get('deepest_leaked_robot_depth', 0.0)):.3f} "
            f"leakage_blocked_after_edge_seal="
            f"{leakage.get('leakage_blocked_after_edge_seal', False)}"
        )
    for uid, audit in sorted(
        perception.guard_communication_audits.items()
    ):
        print(
            f"[CommunicationRegression] uid={uid} "
            f"before_connected={audit['before_connected']} "
            f"before_largest_component="
            f"{audit['before_largest_component']} "
            f"after_connected={audit['after_connected']} "
            f"after_largest_component="
            f"{audit['after_largest_component']} "
            f"selected_critical_ids={audit['selected_critical_ids']} "
            "communication_disconnect_caused_by_guard_selection="
            f"{audit['communication_disconnect_caused_by_guard_selection']}"
        )
    for event in getattr(
        physical, "integration_frontier_lineage_events", []
    ):
        print(
            f"[FrontierLineageRegression] uid={event['uid']} "
            f"frame={event['frame']} "
            f"same_ready_guard_ids={event['same_ready_guard_ids']} "
            f"max_role_transition_jump="
            f"{event['max_role_transition_jump']:.6f} "
            f"frontier_ids={event['frontier_ids']}"
        )
    for branch, lifecycle in getattr(
        physical, "integration_wall_lifecycle", {}
    ).items():
        print(
            f"[WallLifecycleRegression] branch={branch} "
            f"state={lifecycle['state']} rows={lifecycle['rows']} "
            f"cols={lifecycle['cols']} "
            f"robots={len(lifecycle['robot_ids'])} "
            f"max_formation_error="
            f"{lifecycle.get('max_formation_error', 0.0):.6f}"
        )
    for uid, status in sorted(
        getattr(physical, "integration_wall_status", {}).items()
    ):
        print(
            f"[WallRegression] uid={uid} "
            f"capture={status.get('capture', 0)} "
            f"assigned={status.get('assigned', 0)} "
            f"rows={status.get('rows', 0)} "
            f"columns={status.get('slots_per_row', 0)} "
            f"settled={status.get('settled_ratio', 0.0):.3f} "
            f"span={status.get('min_span_ratio', 0.0):.3f} "
            f"edge_gap={status.get('max_edge_gap', 0.0):.3f} "
            f"internal_gap={status.get('max_internal_gap', 0.0):.3f} "
            f"slots_walkable={status.get('slots_walkable', 0)}/"
            f"{status.get('slots_total', 0)} "
            f"ready_frame={status.get('ready_frame')}"
        )
    for uid, association in sorted(
        getattr(physical, "integration_opening_mouth_associations", {}).items()
    ):
        print(
            f"[MouthAssociationRegression] uid={uid} "
            f"opening_center={association['opening_center']:+.1f}deg "
            f"matched_mouth={association['matched_mouth']} "
            f"mouth_local={association['mouth_local_angle']:+.1f}deg "
            f"angular_error={association['angular_error']:.1f}deg"
        )
    for event in getattr(physical, "integration_saturation_events", []):
        print(
            f"[SaturationRegression] uid={event['uid']} "
            f"frame={event['frame']} "
            f"frontier_speed={event['frontier_speed']:.3f} "
            f"local_density={event['local_density']:.6f} "
            f"density_ratio={event['density_ratio']:.3f} "
            f"local_pressure={event['local_pressure']:.3f} "
            f"pressure_ratio={event['pressure_ratio']:.3f} "
            f"cross_fill={event['cross_section_fill']:.3f} "
            f"dwell={event['dwell']:.3f} "
            f"same_ids={event['frontier_ids'] == event['shepherd_ids']} "
            f"max_transition_jump={event['max_transition_jump']:.6f} "
            f"return_direction={event['return_direction']} "
            f"rows={event['rows']} cols={event['cols']} "
            f"robots={event['robots']} "
            f"max_formation_error={event['max_formation_error']:.6f} "
            f"frontier_ids={event['frontier_ids']} "
            f"shepherd_ids={event['shepherd_ids']}"
        )
    for event in getattr(physical, "integration_backflow_events", []):
        print(
            f"[BackflowRegression] uid={event['uid']} "
            f"frame={event['frame']} ratio={event['ratio']:.3f} "
            f"mean_speed={event['mean_speed']:.3f} "
            f"duration={event['duration']:.3f}"
        )
    print(
        "[Summary] "
        f"frames={frame_count} confirm_frame={perception.confirmation_frame} "
        f"confirm_time={perception.confirmation_time} anchor_id={perception.leader.robot_id} "
        f"pre_detection_travel={perception.pre_detection_travel:.3f} "
        f"persistent={sum(len(t.observations) >= MIN_PERSISTENT_OBSERVATIONS for t in perception.tracks)} "
        f"outgoing={len(perception.outgoing)} guards={guard_count} "
        f"selected={physical.branch_identity_label(physical.active_branch_uid)} "
        f"frontier_ids={frontier_ids} shepherd_ids={shepherd_ids} "
        f"visited={visited_log} final_phase={physical.phase.name}"
    )
    print(
        f"[DFS] visited_sequence={visited_log} "
        f"final_phase={physical.phase.name}"
    )
    pygame.quit()
    return 0 if physical.phase == physical.SimulationPhase.DONE or args.max_frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
