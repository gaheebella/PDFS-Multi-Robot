"""Adaptive LiDAR front-end for the single-junction Physical DFS baseline.

The Physical DFS implementation remains the authoritative source for the map,
robots, SPH forces, communication, Guards, Frontier/Shepherd lifecycle,
Pebbles, backtracking, and return-to-base logic.  This integration module loads
only the definition section of that file and supplies one sensor/perception
front-end plus one dark renderer.  No second SimulatorWorld or robot set exists.

Localization audit
------------------
Fixture labels are unavailable to the detector and persistent-opening tracker.
They are used once, in ``handoff_to_physical_dfs``, as the explicitly permitted
single-cross adapter needed by the legacy physics functions.  Guard WHO
selection remains inside Physical DFS; mouth position, width, and motion frame
come from the fixed Anchor's local LiDAR observation.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import types
from dataclasses import dataclass, field
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
    "raw": (55, 119, 172),
    "smooth": (64, 207, 160),
    "open": (46, 213, 115),
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
STATIONARY_WINDOW = 120
MIN_PERSISTENT_OBSERVATIONS = 72
PREVIOUS_APPROACH_EXTENSION = 60.0
BASE_ADDED_EXTENSION = 0.0
APPROACH_EXTENSION = PREVIOUS_APPROACH_EXTENSION + BASE_ADDED_EXTENSION
ASSOCIATION_TOLERANCE_DEG = max(
    2.0 * float(adaptive.FROZEN_PARAMETERS["merge_gap_deg"]),
    float(adaptive.FROZEN_PARAMETERS["min_opening_width_deg"]),
)


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
        self.parent: PersistentOpening | None = None
        self.outgoing: list[PersistentOpening] = []
        self.handoff_complete = False
        self.leader = self._select_leader(robots)
        self.initial_leader_position = self.leader.position.copy()
        self.leader.is_lidar_robot = True
        self.leader.is_fixed_anchor = False
        self.leader.body_yaw = math.radians(self.yaw_deg)
        print(f"[LiDAR] leader_id={self.leader.robot_id}")
        print(
            f"[LiDAR] initial_position=({self.initial_leader_position.x:.3f},"
            f"{self.initial_leader_position.y:.3f})"
        )

    def _select_leader(self, robots: Sequence[Any]) -> Any:
        front_y = min(robot.position.y for robot in robots)
        front = [robot for robot in robots if abs(robot.position.y - front_y) <= 1.0e-6]
        return min(front, key=lambda robot: (
            abs(robot.position.x - self.physical.center_x), robot.robot_id
        ))

    def reset(self, robots: Sequence[Any]) -> None:
        self.__init__(self.physical, robots)

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
        if len(persistent) < 4 or self.stationary_samples < STATIONARY_WINDOW:
            return False
        # Incoming motion is local angle zero, so the parent opening is the
        # direction opposite recent ingress: local +/-180 degrees.
        self.parent = min(persistent, key=lambda track: circular_error(track.center_angle, 180.0))
        self.parent.persistent_id = "PARENT_00"
        children = [track for track in persistent if track is not self.parent]
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
            adaptive_w, MAX_RANGE, TAU
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
            self.pre_detection_travel = self.anchor_position.distance_to(
                self.initial_leader_position
            )
            self.leader.is_fixed_anchor = True
            self.leader.base_reserve = True
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
            print(
                f"[Anchor] fix_position=({self.anchor_position.x:.3f},"
                f"{self.anchor_position.y:.3f})"
            )
            print("[Opening] stationary accumulation started")
        if self.anchor_fixed:
            self.stationary_samples += 1
            # Once fixed, accumulate every raw opening again. The actual rear
            # track becomes PARENT_00 in _persistent_ready().
            self._associate(openings)
            if self.stationary_samples % 10 == 0:
                count = sum(
                    len(track.observations) >= MIN_PERSISTENT_OBSERVATIONS
                    for track in self.tracks
                )
                print(f"[Opening] persistent count={count} samples={self.stationary_samples}")
            if self.state == PerceptionState.FIXED_ACCUMULATING and self._persistent_ready():
                self.state = PerceptionState.BRANCHES_READY
                print(f"[Topology] parent={self.parent.persistent_id}")
                print(f"[Topology] outgoing={[track.persistent_id for track in self.outgoing]}")
                print(
                    f"[Topology] persistent_count={sum(len(track.observations) >= MIN_PERSISTENT_OBSERVATIONS for track in self.tracks)} "
                    f"parent_count={int(self.parent is not None)} "
                    f"outgoing_count={len(self.outgoing)}"
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


def _elect_missing_guard_ids_with_localization(
    physical: types.ModuleType,
    robots: Sequence[Any],
    excluded_robot_id: int,
) -> None:
    """Permitted baseline localization: choose WHO, then retain IDs only.

    Slot geometry is LiDAR descriptor-local. Absolute robot positions are read
    only for this one-shot nearest-candidate election. No position, fixture
    centreline, terminal coordinate, or slot is passed into DFS memory.
    """
    claimed = {
        robot_id
        for ids in physical.junction_guard_groups.values()
        for robot_id in ids
    }
    for uid in physical.ordered_discovered_branch_uids():
        descriptor = physical.branch_descriptors_by_uid[uid]
        branch = descriptor.fixture_key
        existing_ids = list(physical.junction_guard_groups.get(branch, []))
        required = physical.required_junction_guard_count(descriptor)
        if len(existing_ids) >= required:
            continue
        for robot in robots:
            if robot.robot_id not in existing_ids:
                continue
            robot.role = "NORMAL"
            robot.junction_guard_anchor = None
            robot.junction_guard_branch = None
            robot.junction_guard_branch_uid = None
            robot.junction_guard_hop = -1
            robot.junction_guard_parent_id = None
            robot.junction_guard_layer = -1
            robot.is_branch_leader = False
            claimed.discard(robot.robot_id)
        physical.junction_guard_groups.pop(branch, None)
        slots = physical.build_local_junction_guard_slots(
            descriptor,
            required,
            physical.JUNCTION_GUARD_BRANCH_INSET,
            fit_walkable=True,
        )
        candidates = [
            robot for robot in robots
            if robot.robot_id not in claimed
            and robot.robot_id != excluded_robot_id
            and robot.role == "NORMAL"
            and not robot.base_reserve
        ]
        # Deterministic minimum-distance WHO election with robot-id tie break.
        selected: list[Any] = []
        available = list(candidates)
        for slot in slots:
            if not available:
                break
            robot = min(available, key=lambda item: (item.position.distance_squared_to(slot), item.robot_id))
            available.remove(robot)
            selected.append(robot)
        assignment = physical.assign_shepherd_slots(selected, slots)
        if len(assignment) < physical.JUNCTION_GUARD_MIN_COUNT:
            continue
        leader = min((item[0] for item in assignment), key=lambda item: item.robot_id)
        ids: list[int] = []
        for robot, slot, _ in assignment:
            robot.role = "JUNCTION_GUARD"
            robot.junction_guard_anchor = slot.copy()
            robot.junction_guard_branch = branch
            robot.junction_guard_branch_uid = uid
            robot.junction_guard_hop = 0
            robot.junction_guard_parent_id = None if robot is leader else leader.robot_id
            robot.junction_guard_layer = 0
            robot.is_branch_leader = robot is leader
            robot.velocity.update(0.0, 0.0)
            robot.acceleration.update(0.0, 0.0)
            robot.filtered_acceleration.update(0.0, 0.0)
            ids.append(robot.robot_id)
            claimed.add(robot.robot_id)
        descriptor.leader_id = leader.robot_id
        physical.junction_guard_groups[branch] = ids
        physical.junction_guard_frontier_depths[branch] = physical.JUNCTION_GUARD_BRANCH_INSET
        print(f"[Localization WHO] uid={uid} elected_guard_ids={ids}")


def handoff_to_physical_dfs(physical: types.ModuleType, perception: AdaptivePerception, robots: Sequence[Any]) -> None:
    """Create local descriptors, then invoke the existing Guard controller.

    The fixture mapping below is the only localization-assisted integration
    adapter. It answers WHO/legacy-key compatibility only; the detector never
    receives labels, target coordinates, dead-end coordinates, or slots.
    """
    if perception.handoff_complete or perception.anchor_position is None:
        return
    physical.detected_branch_candidates = set()
    all_ids = {robot.robot_id for robot in robots if robot is not perception.leader}
    used_fixtures: set[str] = set()
    frame = perception.last_frame
    measured_wall_width = max(
        physical.ROBOT_RADIUS * 6.0,
        float((frame.left or 0.0) + (frame.right or 0.0)),
    )
    # Keep the descriptor as a physical robot-centre width, with a symmetric
    # local LiDAR clearance reserve for boundary uncertainty.
    observed_width = max(
        physical.ROBOT_RADIUS * 6.0,
        measured_wall_width - 12.0 * physical.ROBOT_RADIUS,
    )
    # A cross's child axes are expressed only relative to recent ingress.
    # Snapping each observed gap to a unique local {-90, 0, +90} axis removes
    # corner-visibility bias without consulting a global branch centreline.
    local_axes = (-90.0, 0.0, 90.0)
    assigned: list[tuple[Any, ...]] = []
    ordered_tracks = sorted(perception.outgoing, key=lambda item: item.center_angle)
    axis_assignment = min(
        itertools.permutations(local_axes),
        key=lambda axes: sum(
            circular_error(track.center_angle, axis)
            for track, axis in zip(ordered_tracks, axes)
        ),
    )
    for track, local_axis in zip(ordered_tracks, axis_assignment):
        world_angle = math.radians(perception.yaw_deg + local_axis)
        direction = pygame.Vector2(math.cos(world_angle), math.sin(world_angle)).normalize()
        starts = [item["start_angle"] for item in track.observations]
        ends = [item["end_angle"] for item in track.observations]
        boundary_points = []
        for local_angle in (float(np.median(starts)), float(np.median(ends))):
            errors = np.abs((frame.angles - local_angle + 180.0) % 360.0 - 180.0)
            ray_index = int(np.argmin(errors))
            ray_angle = math.radians(perception.yaw_deg + float(frame.angles[ray_index]))
            boundary_points.append(
                perception.anchor_position
                + pygame.Vector2(math.cos(ray_angle), math.sin(ray_angle)) * float(frame.raw[ray_index])
            )
        provisional_mouth = 0.5 * (boundary_points[0] + boundary_points[1])
        provisional_center = provisional_mouth - direction * (0.5 * observed_width)
        assigned.append(
            (
                track,
                local_axis,
                direction,
                provisional_center,
                (boundary_points[0], boundary_points[1]),
            )
        )
    # The two lateral openings each expose the near bottom corner of the
    # Junction. Their closest boundary returns define the incoming mouth row;
    # half the locally measured corridor width forward gives the Junction
    # centre. This remains entirely Anchor-local and avoids far-wall parallax.
    side_near_corners = [
        min(
            item[4],
            key=lambda point: point.distance_squared_to(perception.anchor_position),
        )
        for item in assigned
        if abs(item[1]) == 90.0
    ]
    if len(side_near_corners) == 2:
        ingress_forward = pygame.Vector2(
            math.cos(math.radians(perception.yaw_deg)),
            math.sin(math.radians(perception.yaw_deg)),
        )
        junction_estimate = (
            0.5 * (side_near_corners[0] + side_near_corners[1])
            + ingress_forward * (0.5 * measured_wall_width)
        )
    else:
        junction_estimate = pygame.Vector2(
            float(np.median([item[3].x for item in assigned])),
            float(np.median([item[3].y for item in assigned])),
        )

    for track, local_axis, direction, _, _ in assigned:
        fixture = max(
            (key for key in physical.BRANCHES if key not in used_fixtures),
            key=lambda key: direction.dot(physical.BRANCH_DIRECTIONS[key]),
        )
        used_fixtures.add(fixture)
        uid = track.persistent_id
        mouth = junction_estimate + direction * (0.5 * observed_width)
        normal = pygame.Vector2(-direction.y, direction.x)
        descriptor = physical.BranchDescriptor(
            uid=uid,
            junction_uid=physical.CURRENT_JUNCTION_ID,
            fixture_key=fixture,
            local_outgoing_direction=direction.copy(),
            local_return_direction=-direction,
            observed_mouth_position=mouth,
            observed_width=observed_width,
            cohort_member_ids=set(all_ids),
            direction_last_estimate=direction.copy(),
            direction_stability_reference=direction.copy(),
            direction_stable_dwell=1.0,
            direction_sample_count=max(physical.JUNCTION_COHORT_MIN_ROBOTS, len(all_ids)),
            direction_angular_spread=0.0,
            direction_is_stable=True,
            direction_mature_dwell=1.0,
            direction_is_mature=True,
            direction_downstream_travel=physical.JUNCTION_COHORT_MIN_TRAVEL,
            motion_t=direction.copy(),
            motion_n=normal,
            motion_frame_locked=True,
            motion_frame_source="FIXED_ANCHOR_PERSISTENT_OPENING",
            motion_frame_sample_count=len(track.observations),
            motion_frame_angular_spread=0.0,
            motion_observed_width=observed_width,
            observed_flow_width=observed_width,
            observed_physical_width=observed_width,
            physical_width_confident=True,
            physical_width_source="FIXED_ANCHOR_LOCAL_LIDAR",
            physical_left_boundary_lateral=-observed_width * 0.5,
            physical_right_boundary_lateral=observed_width * 0.5,
            physical_boundary_sample_count=len(track.observations),
            discovered_at=physical.simulation_time,
        )
        physical.branch_descriptors_by_uid[uid] = descriptor
        physical.fixture_key_to_branch_uid[fixture] = uid
        physical.branch_uid_to_fixture_key[uid] = fixture
        physical.branch_local_uids[fixture] = uid
        physical.detected_branch_candidates.add(fixture)
        for robot in robots:
            robot.local_branch_uid_by_key[fixture] = uid
        print(
            f"[BranchDescriptor] uid={uid} local_axis={local_axis:+.0f}deg "
            f"mouth=({mouth.x:.1f},{mouth.y:.1f}) width={observed_width:.1f} "
            f"fixture-adapter={fixture}"
        )
    physical.branch_discovery_counter = len(perception.outgoing)
    # Compatibility mirror only: downstream distributed voting has a legacy
    # settle gate on this field. The MOVE->GUARD transition above is already
    # and exclusively authorized by Adaptive LiDAR + fixed Anchor + topology.
    physical.junction_inference_tracker.confirmed = True
    physical.junction_inference_tracker.confirmed_at = (
        physical.simulation_time - physical.JUNCTION_DISCOVERY_SETTLE_TIME
    )
    physical.junction_inference_tracker.valid_branches = set(
        physical.detected_branch_candidates
    )
    physical.begin_junction_guard_formation(robots)
    _elect_missing_guard_ids_with_localization(
        physical, robots, perception.leader.robot_id
    )
    physical.phase = physical.SimulationPhase.FORM_JUNCTION_GUARDS
    perception.handoff_complete = True
    perception.state = PerceptionState.PHYSICAL_DFS
    guard_counts = {
        branch: len(physical.junction_guard_groups.get(branch, []))
        for branch in physical.BRANCHES
    }
    print(f"[DFS] guard_counts={guard_counts}")
    print("[DFS Handoff] branches ready descriptors=3")
    print("[Junction Guard] formation started")


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
        for bound, color in ((frame.lower, COLORS["safe"]), (frame.upper, COLORS["safe"]), (frame.selected, COLORS["threshold"])):
            if bound is not None and math.isfinite(bound):
                y = self._profile_point(0.0, float(bound))[1]
                pygame.draw.line(self.screen, color, (PROFILE_RECT.left, y), (PROFILE_RECT.right, y), 1)
        raw_points = [self._profile_point(float(a), float(r)) for a, r in zip(frame.angles, frame.raw)]
        smooth_points = [self._profile_point(float(a), float(r)) for a, r in zip(frame.angles, frame.smoothed)]
        pygame.draw.lines(self.screen, COLORS["raw"], False, raw_points, 1)
        pygame.draw.lines(self.screen, COLORS["smooth"], False, smooth_points, 2)
        for index in np.flatnonzero(frame.support)[::2]:
            point = self._profile_point(float(frame.angles[index]), float(frame.smoothed[index]))
            pygame.draw.circle(self.screen, COLORS["open"], point, 2)
        for opening in frame.openings:
            for angle in (opening["start_angle"], opening["end_angle"]):
                x = self._profile_point(float(angle), 0.0)[0]
                pygame.draw.line(self.screen, COLORS["threshold"], (x, PROFILE_RECT.top), (x, PROFILE_RECT.bottom), 1)
        self.screen.blit(self.small.render("theta [deg]   RAW / SMOOTHED / OPEN SUPPORT", True, COLORS["text"]), (PROFILE_RECT.left + 10, PROFILE_RECT.top + 8))
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

    def _draw_diagnostics(self, robots: Sequence[Any], perception: AdaptivePerception) -> None:
        physical = self.physical
        frame = perception.last_frame
        pygame.draw.rect(self.screen, COLORS["panel_alt"], DIAGNOSTIC_RECT, border_radius=6)
        selected_uid = physical.branch_uid_for_fixture(physical.active_branch) if perception.handoff_complete else None
        lines = [
            f"Detector: Adaptive W-tau (alpha={ALPHA:.1f})",
            f"Anchor: {'FIXED' if perception.anchor_fixed else 'MOVING'}  id={perception.leader.robot_id}",
            f"Junction confirmed: {perception.junction_confirmed}",
            f"Adaptive W={frame.adaptive_w:.2f}  Tmin={frame.lower:.2f}  Tmax={frame.upper:.2f}" if frame else "Adaptive W=-",
            f"Selected T={frame.selected:.2f}  openings={len(frame.openings)}" if frame and frame.selected is not None else "Selected T=-",
            f"Persistent={sum(len(t.observations) >= MIN_PERSISTENT_OBSERVATIONS for t in perception.tracks)}  Parent={perception.parent.persistent_id if perception.parent else '-'}  Outgoing={len(perception.outgoing)}",
            "----------------------------------------------",
            f"Physical DFS phase={physical.phase.name}",
            f"Selected branch={selected_uid or '-'} ({physical.active_branch if perception.handoff_complete else '-'})",
            "States: " + " | ".join(f"{physical.branch_uid_for_fixture(key) or key}={physical.branch_states[key]}" for key in physical.BRANCHES),
            "Guards: " + " | ".join(f"{key}={sum(r.role == 'JUNCTION_GUARD' and r.junction_guard_branch == key for r in robots)}" for key in physical.BRANCHES),
            f"Frontier={len(physical.get_frontier_shepherds(robots))}  Shepherds={len(physical.get_shepherds(robots))}",
            f"Relays={len(physical.get_relays(robots))}  Pebbles={len(physical.get_pebbles(robots))}",
            f"Dead-end={physical.dead_end_inference_tracker.confirmed}  Backflow={physical.phase == physical.SimulationPhase.FLOW_BACKTRACK}",
            f"Connected={physical.get_communication_stats(robots)['connected']}/{len(robots)}",
        ]
        x, y = DIAGNOSTIC_RECT.left + 12, DIAGNOSTIC_RECT.top + 12
        for line in lines:
            self.screen.blit(self.small.render(line, True, COLORS["text"]), (x, y))
            y += 22
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
        self._draw_diagnostics(robots, perception)
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
        ids = [robot.robot_id for robot in physical.get_frontier_shepherds(robots)]
        print(f"[Junction Guard] ready")
        print(f"[DFS] selected branch={physical.branch_identity_label(physical.active_branch_uid)}")
        print(f"[Frontier] promoted ids={ids}")
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


def initialize_deployment_fields(robots: Sequence[Any]) -> None:
    for robot in robots:
        robot.body_yaw = -0.5 * math.pi
        robot.propulsion_weight = adaptive.LOCAL_FOLLOWER_DRIVE_WEIGHT
        robot.heading_parent_id = None
        robot.heading_hop = 0


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
    install_local_forward_ingress(physical)
    def integration_log_sink(active_robots: Sequence[Any], reason: str) -> Path:
        """Keep the reference module's legacy CSV untouched."""
        physical.metrics.saved = True
        print(f"[Log] integration run complete reason={reason}; console summary follows")
        return HERE / "sph_dfs_experiment_summary.csv"

    physical.save_experiment_logs = integration_log_sink
    robots, reference_density, color_reference_density = physical.initialize_simulation()
    initialize_deployment_fields(robots)
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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    robots, reference_density, color_reference_density = physical.initialize_simulation()
                    initialize_deployment_fields(robots)
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
            log_initial_dynamics(
                physical, robots, reference_density, frame_count
            )
            physical.compute_sph_forces(robots, physics_grid, spatial_grid, dt)
            for robot in robots:
                if perception.anchor_fixed and robot is perception.leader:
                    continue
                robot.update(dt)
            perception.enforce_anchor()
            spatial_grid = physical.build_spatial_grid(robots)
            physical.update_communication_system(robots, spatial_grid)
            lidar_frame = perception.update(physical.simulation_time)
            if perception.state == PerceptionState.BRANCHES_READY:
                handoff_to_physical_dfs(physical, perception, robots)
            elif perception.handoff_complete:
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
        f"[Accounting] base_returned={base_returned} "
        f"persistent_pebbles={pebble_count} total={base_returned + pebble_count}"
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
