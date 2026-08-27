"""EXP-046 Pre-Junction Corridor-Based W Calibration.

This self-contained copy preserves EXP-045's active manual-W detector and
anchor semantics.  A local angle/range-history corridor calibrator runs in
shadow mode; map/pose and noise-free scan data are evaluation-only and never
feed threshold, confirmation, anchor control, motion, or calibration gates.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np


# =============================================================================
# 1. Constants, map geometry, robot state, local-forward physics, LiDAR raycast
# Extracted from pre_exploration_general_pipeline_simulator.py.
# =============================================================================
MAP_SCALE = 0.70
# Production count is required to preserve longitudinal packing, density, and
# support-neighborhood regime; the fidelity audit demonstrated that 240 does not.
ROBOT_COUNT = int(os.environ.get("PRE_EXPLORATION_ROBOT_COUNT", "680"))
ROBOT_RADIUS = 1.60 * MAP_SCALE
GRID_SPACING = 4.00 * MAP_SCALE
GRID_ROW_SPACING = 3.80 * MAP_SCALE
SMOOTHING_LENGTH = 22.0 * MAP_SCALE
PRESSURE_GAIN = 2800.0
SPH_PRESSURE_SCALE = 3.0
STIFFNESS_EXPONENT = 0.5
VISCOSITY_XI1, VISCOSITY_XI2 = 0.9, 1.2
DAMPING = 4.0
CORRIDOR_LATERAL_VELOCITY_DAMPING = 12.0
SAFE_RADIUS = 7.5 * MAP_SCALE
REPULSION_GAIN = 260.0
NORMAL_EQUILIBRIUM_SCALE = 1.48
ACCELERATION_FILTER_ALPHA = 0.18
PRESSURE_FORCE_LIMIT = 420.0 * 3.0
VISCOSITY_FORCE_LIMIT = 150.0 * 3.0
VELOCITY_CONSENSUS_GAIN = 6.0
VISCOELASTIC_LINK_RADIUS = SAFE_RADIUS * 1.45
VISCOELASTIC_REST_MIN = ROBOT_RADIUS * 2.05
VISCOELASTIC_REST_MAX = SAFE_RADIUS * 1.65
VISCOELASTIC_ELASTIC_GAIN = 42.0
VISCOELASTIC_DASHPOT_GAIN = 8.0
VISCOELASTIC_FORCE_LIMIT = 75.0 * 3.0
ROUTE_FORCE = 22.0 * 3.0
MAX_SPEED = 36.0 * 3.0
MAX_ACCELERATION = 200.0 * 3.0
DT = 1.0 / 60.0
SAMPLE_PERIOD = 0.10
SENSOR_PERIOD = 0.10
REFERENCE_FRONT_QUANTILE = 0.68
BOUNDARY_GAP_DEG = 120.0
EPSILON = 1e-8
LIDAR_RAYS = 360
LIDAR_MAX_RANGE = 150.0
BASELINE_CORRIDOR_WIDTH = 84.0
MIN_SPEED = 1.2
ANCHOR_STATIONARY_DWELL_STEPS = max(1, round(SAMPLE_PERIOD / DT))


# Adapted from production's MOVE_TO_JUNCTION initialization protocol
# (get_base_pressure_scale, adaptive_equilibrium_radius, compute_route_force,
# and initial_pressure_release_active). Branch/event-specific terms are omitted.
BASE_COMPRESSION_DURATION = 0.65
BASE_COMPRESSION_FORCE = 80.0 * 3.0
BASE_COMPRESSION_RISE_FRACTION = 0.20
BASE_COMPRESSION_FALL_START_FRACTION = 0.80
BASE_COMPRESSION_PRESSURE_SCALE = 0.35
BASE_EXPANSION_PRESSURE_SCALE = 5.20
BASE_EXPANSION_BOOST_DURATION = 3.20
BASE_EXPANSION_RAMP_FRACTION = 0.22
BASE_PACKED_EQUILIBRIUM_SCALE = 0.60
BASE_EQUILIBRIUM_RELEASE_DURATION = 0.40
BASE_STORED_PRESSURE_FLOOR = 0.85
BASE_STORED_PRESSURE_RISE_TIME = 0.12
BASE_STORED_PRESSURE_DECAY_START = 2.80
BASE_STORED_PRESSURE_DURATION = 6.00
INITIAL_INGRESS_FORCE = 2.5 * 3.0
INITIAL_INGRESS_BRAKE_DISTANCE = 52.0 * MAP_SCALE
INITIAL_INGRESS_MIN_FORCE_SCALE = 0.10
INITIAL_INGRESS_LANE_GAIN = 0.25
INITIAL_INGRESS_LANE_MAX_FORCE = 5.0
INITIAL_RELEASE_PRESSURE_FORCE_LIMIT = 260.0 * 3.0
INITIAL_RELEASE_VISCOSITY_MULTIPLIER = 1.35
INITIAL_RELEASE_EXTRA_DAMPING = 6.0
INITIAL_RELEASE_ACCELERATION_FILTER_ALPHA = 0.12
INITIAL_WALL_RESTITUTION = 0.18
PRODUCTION_BASE_LENGTH = 84.0
BASE_PISTON_REACTION_GAIN = 260.0 * 3.0
BASE_PISTON_REACTION_FORCE_LIMIT = 260.0 * 3.0
BASE_PISTON_REACTION_RISE_TIME = 0.25
BASE_PISTON_REACTION_DURATION = 5.0
# Evaluation-derived target scale: the production M1 pre-entrance mean speed
# is about 65 world units/s.  Multiplying by the unchanged viscous damping
# gives the smallest constant body-forward force with that terminal scale.
LOCAL_FORWARD_REFERENCE_SPEED = 65.0
LOCAL_FORWARD_DRIVE_FORCE = DAMPING * LOCAL_FORWARD_REFERENCE_SPEED
LOCAL_FOLLOWER_DRIVE_WEIGHT = 0.50
# Production uses 54 world units before MAP_SCALE for local communication.
# Keep communication/propagation distinct from the shorter SPH support graph.
LOCAL_COMMUNICATION_RANGE = 54.0 * MAP_SCALE
DEPLOYMENT_BODY_YAW_RAD = math.pi / 2.0
DEFAULT_GUI_SCALE = 0.75
LOCAL_WALL_CONFINEMENT_MAX_WIDTH = SMOOTHING_LENGTH * 6.0

Point = tuple[float, float]
Segment = tuple[Point, Point]


def _limit(vector: np.ndarray, maximum: float) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    return vector * (maximum / length) if length > maximum else vector


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _compression_envelope(time_s: float) -> float:
    progress = min(1.0, max(0.0, time_s / BASE_COMPRESSION_DURATION))
    rise = _smoothstep(progress / BASE_COMPRESSION_RISE_FRACTION)
    fall = 1.0 - _smoothstep((progress - BASE_COMPRESSION_FALL_START_FRACTION) / (1.0 - BASE_COMPRESSION_FALL_START_FRACTION))
    return rise * fall


def _pressure_scale(time_s: float) -> float:
    if time_s < BASE_COMPRESSION_DURATION:
        return BASE_COMPRESSION_PRESSURE_SCALE
    elapsed = time_s - BASE_COMPRESSION_DURATION
    if elapsed >= BASE_EXPANSION_BOOST_DURATION:
        return SPH_PRESSURE_SCALE
    progress = min(1.0, max(0.0, elapsed / BASE_EXPANSION_BOOST_DURATION))
    rise = _smoothstep(progress / BASE_EXPANSION_RAMP_FRACTION)
    fall = _smoothstep((progress - (1.0 - BASE_EXPANSION_RAMP_FRACTION)) / BASE_EXPANSION_RAMP_FRACTION)
    peak = BASE_COMPRESSION_PRESSURE_SCALE + (BASE_EXPANSION_PRESSURE_SCALE - BASE_COMPRESSION_PRESSURE_SCALE) * rise
    return peak + (SPH_PRESSURE_SCALE - peak) * fall


def _stored_pressure_envelope(time_s: float) -> float:
    elapsed = time_s - BASE_COMPRESSION_DURATION
    if elapsed <= 0.0 or elapsed >= BASE_STORED_PRESSURE_DURATION:
        return 0.0
    rise = _smoothstep(elapsed / BASE_STORED_PRESSURE_RISE_TIME)
    decay = 1.0 - _smoothstep((elapsed - BASE_STORED_PRESSURE_DECAY_START) / (BASE_STORED_PRESSURE_DURATION - BASE_STORED_PRESSURE_DECAY_START))
    return rise * decay


def _base_piston_envelope(time_s: float) -> float:
    """Return the production Base-wall release envelope without event coupling."""
    elapsed = time_s - BASE_COMPRESSION_DURATION
    if elapsed <= 0.0 or elapsed >= BASE_PISTON_REACTION_DURATION:
        return 0.0
    rise = _smoothstep(elapsed / BASE_PISTON_REACTION_RISE_TIME)
    decay = 1.0 - _smoothstep(
        (elapsed - BASE_PISTON_REACTION_RISE_TIME)
        / (BASE_PISTON_REACTION_DURATION - BASE_PISTON_REACTION_RISE_TIME)
    )
    return rise * decay


def _equilibrium_radius(time_s: float) -> float:
    progress = _smoothstep((time_s - BASE_COMPRESSION_DURATION) / BASE_EQUILIBRIUM_RELEASE_DURATION)
    scale = BASE_PACKED_EQUILIBRIUM_SCALE + (NORMAL_EQUILIBRIUM_SCALE - BASE_PACKED_EQUILIBRIUM_SCALE) * progress
    return max(ROBOT_RADIUS * 2.05, SAFE_RADIUS * scale)


def _kernel(distance: float) -> float:
    if distance < 0.0 or distance > SMOOTHING_LENGTH:
        return 0.0
    return 10.0 / (math.pi * SMOOTHING_LENGTH**2) * (1.0 - distance / SMOOTHING_LENGTH) ** 3


def _gradient(offset: np.ndarray) -> np.ndarray:
    distance = float(np.linalg.norm(offset))
    if distance <= EPSILON or distance > SMOOTHING_LENGTH:
        return np.zeros(2)
    magnitude = -30.0 / (math.pi * SMOOTHING_LENGTH**3) * (1.0 - distance / SMOOTHING_LENGTH) ** 2
    return magnitude * offset / distance


@dataclass(frozen=True)
class BranchSpec:
    angle_deg: float
    width: float
    length: float


@dataclass(frozen=True)
class FreeRect:
    vertices: tuple[Point, Point, Point, Point]

    def contains(self, point: np.ndarray, tolerance: float = 1e-8) -> bool:
        vertices = [np.asarray(value) for value in self.vertices]
        signs = []
        for start, end in zip(vertices, vertices[1:] + vertices[:1]):
            edge = end - start
            signs.append(float(edge[0] * (point[1] - start[1]) - edge[1] * (point[0] - start[0])))
        return min(signs) >= -tolerance or max(signs) <= tolerance


@dataclass(frozen=True)
class GeometryCase:
    case_id: str
    incoming_width: float
    incoming_length: float
    junction_size: float
    branches: tuple[BranchSpec, ...]
    free_rects: tuple[FreeRect, ...]
    walls: tuple[Segment, ...]
    entrance_y: float | None

    def contains(self, point: np.ndarray) -> bool:
        return any(rect.contains(point) for rect in self.free_rects)

    def walkable(self, point: np.ndarray, radius: float = ROBOT_RADIUS) -> bool:
        diagonal = radius / math.sqrt(2.0)
        probes = ((0.,0.),(radius,0.),(-radius,0.),(0.,radius),(0.,-radius),
                  (diagonal,diagonal),(diagonal,-diagonal),(-diagonal,diagonal),(-diagonal,-diagonal))
        return all(self.contains(point + np.asarray(offset)) for offset in probes)


def _rect(center: np.ndarray, direction: np.ndarray, width: float, length: float) -> FreeRect:
    direction = direction / np.linalg.norm(direction)
    lateral = np.array([-direction[1], direction[0]])
    start, end = center - direction * length / 2, center + direction * length / 2
    return FreeRect(tuple(tuple(value) for value in (start-lateral*width/2, end-lateral*width/2, end+lateral*width/2, start+lateral*width/2)))


def _intersection_t(a, b, c, d):
    r, s = b-a, d-c
    denominator = float(r[0]*s[1]-r[1]*s[0])
    if abs(denominator) < 1e-10:
        return None
    q = c-a
    t = float((q[0]*s[1]-q[1]*s[0])/denominator)
    u = float((q[0]*r[1]-q[1]*r[0])/denominator)
    return min(1.0, max(0.0, t)) if -1e-9 <= t <= 1+1e-9 and -1e-9 <= u <= 1+1e-9 else None


def _union_boundary(rects: tuple[FreeRect, ...]) -> tuple[Segment, ...]:
    """Build external walls by splitting and classifying primitive edges."""
    edges = []
    for rect in rects:
        vertices = [np.asarray(value) for value in rect.vertices]
        edges.extend(zip(vertices, vertices[1:] + vertices[:1]))
    inside = lambda point: any(rect.contains(point, 1e-7) for rect in rects)
    pieces = []
    for a, b in edges:
        cuts = [0.0, 1.0]
        for c, d in edges:
            value = _intersection_t(a, b, c, d)
            if value is not None:
                cuts.append(value)
        cuts = sorted(set(round(value, 10) for value in cuts))
        edge = b-a
        normal = np.array([-edge[1], edge[0]]) / max(np.linalg.norm(edge), EPSILON)
        for low, high in zip(cuts, cuts[1:]):
            if high-low <= 1e-8:
                continue
            start, end = a+edge*low, a+edge*high
            mid = (start+end)/2
            if inside(mid+normal*1e-4) != inside(mid-normal*1e-4):
                pieces.append((tuple(start), tuple(end)))
    unique = {}
    for start, end in pieces:
        key = tuple(sorted((tuple(round(value, 7) for value in start), tuple(round(value, 7) for value in end))))
        unique[key] = (start, end)
    return tuple(unique.values())


class GeometryBuilder:
    """Owns map metadata; no runtime diagnostic receives this object."""
    @staticmethod
    def build(case_id: str) -> GeometryCase:
        width, incoming, junction, length = BASELINE_CORRIDOR_WIDTH, 190.0, 84.0, 150.0
        definitions = {
            "M0_STRAIGHT": (),
            "M1_CROSS_BASELINE": (BranchSpec(0, width, length), BranchSpec(-90, width, length), BranchSpec(90, width, length)),
            "M2_T_JUNCTION": (BranchSpec(-90, width, length), BranchSpec(90, width, length)),
            "M3_ANGLED_Y": (BranchSpec(-60, width, length), BranchSpec(60, width, length)),
            "M4_ASYMMETRIC_CROSS": (BranchSpec(0, width, length), BranchSpec(-90, width*.75, length), BranchSpec(90, width, length)),
            "M5_UNEQUAL_WIDTH": (BranchSpec(0, width*.65, length), BranchSpec(-90, width, length), BranchSpec(90, width*1.35, length)),
        }
        if case_id not in definitions:
            raise ValueError(case_id)
        if case_id == "M0_STRAIGHT":
            rects, entrance = (_rect(np.array([0., 280.]), np.array([0., 1.]), width, 1040.),), None
        else:
            entrance = -junction/2
            rects_list = [_rect(np.array([0., entrance-incoming/2]), np.array([0., 1.]), width, incoming), _rect(np.zeros(2), np.array([0., 1.]), junction, junction)]
            for branch in definitions[case_id]:
                radians = math.radians(branch.angle_deg)
                direction = np.array([math.sin(radians), math.cos(radians)])
                start = junction/2-2
                rects_list.append(_rect(direction*(start+branch.length/2), direction, branch.width, branch.length+4))
            rects = tuple(rects_list)
        return GeometryCase(case_id, width, incoming, junction, definitions[case_id], rects, _union_boundary(rects), entrance)


@dataclass
class RobotState:
    robot_id: int
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    observed_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(2))
    density: float = 0.0
    pressure: float = 0.0
    ingress_lane_x: float = 0.0
    # Explicit deployment-time body pose. Runtime control reads this
    # proprioceptive yaw; it never queries map/GT Junction geometry.
    body_yaw_rad: float = DEPLOYMENT_BODY_YAW_RAD
    heading_parent_id: int | None = None
    heading_hop: int = -1
    propulsion_weight: float = 0.0


@dataclass(frozen=True)
class LidarScan:
    angles_deg: np.ndarray
    ranges: np.ndarray
    max_range: float


@dataclass(frozen=True)
class LocalObservation:
    """Runtime input: relative positions/motion and one local LiDAR scan."""
    timestamp: float
    robot_ids: np.ndarray
    relative_positions: np.ndarray
    velocities: np.ndarray
    lidar_robot_id: int
    lidar_scan: LidarScan


def _nearest_point(point: np.ndarray, segment: Segment):
    start, end = np.asarray(segment[0]), np.asarray(segment[1])
    edge = end-start
    ratio = float(np.clip(np.dot(point-start, edge)/max(np.dot(edge, edge), EPSILON), 0, 1))
    nearest = start+edge*ratio
    return nearest, float(np.linalg.norm(point-nearest))


def _inward(geometry: GeometryCase, point: np.ndarray, wall: Segment) -> np.ndarray:
    start, end = np.asarray(wall[0]), np.asarray(wall[1])
    edge = end-start
    normal = np.array([-edge[1], edge[0]])/max(np.linalg.norm(edge), EPSILON)
    return normal if geometry.contains(point+normal*1e-3) else -normal


class LidarSensor:
    """World adapter producing only local angle/range observations."""
    def __init__(self, ray_count=LIDAR_RAYS, max_range=LIDAR_MAX_RANGE):
        self.angles = np.linspace(-180., 180., ray_count, endpoint=False)
        self.max_range = max_range

    @staticmethod
    def _ray_hit(origin, direction, segment):
        start, end = np.asarray(segment[0]), np.asarray(segment[1])
        edge = end-start
        denominator = direction[0]*edge[1]-direction[1]*edge[0]
        if abs(denominator) < 1e-10:
            return None
        offset = start-origin
        ray_t = (offset[0]*edge[1]-offset[1]*edge[0])/denominator
        segment_t = (offset[0]*direction[1]-offset[1]*direction[0])/denominator
        return float(ray_t) if ray_t >= 0 and -1e-9 <= segment_t <= 1+1e-9 else None

    def scan(self, geometry: GeometryCase, position: np.ndarray, yaw_deg: float) -> LidarScan:
        ranges = np.full(len(self.angles), self.max_range)
        for index, angle in enumerate(self.angles):
            radians = math.radians(yaw_deg+angle)
            direction = np.array([math.cos(radians), math.sin(radians)])
            hits = [hit for wall in geometry.walls if (hit := self._ray_hit(position, direction, wall)) is not None]
            if hits:
                ranges[index] = min(self.max_range, min(hits))
        return LidarScan(self.angles.copy(), ranges, self.max_range)


class SimulatorWorld:
    """Exact local-forward specialization of the modular physics world."""

    def __init__(
        self,
        geometry: GeometryCase,
        propulsion_mode: str = "local_forward",
    ) -> None:
        if propulsion_mode != "local_forward":
            raise ValueError("standalone GUI supports local_forward only")
        self.geometry = geometry
        self.propulsion_mode = propulsion_mode
        self.robots = self._create_robots()
        self.initial_mean_y = float(
            np.mean([robot.position[1] for robot in self.robots])
        )
        self.initial_front_y = float(
            max(robot.position[1] for robot in self.robots)
        )
        self.time = 0.0
        self.wall_contacts = 0
        self.wall_corrections = 0
        self.rest_lengths: dict[tuple[int, int], float] = {}
        self.lidar_robot_id = self._select_initial_lidar_leader()
        lidar = next(
            robot
            for robot in self.robots
            if robot.robot_id == self.lidar_robot_id
        )
        self.initial_lidar_position = lidar.position.copy()
        self.sensor = LidarSensor()
        self.lidar_yaw_deg = 90.0
        neighbors = self._neighbors()
        self.last_communication_edges: list[tuple[int, int]] = []
        self.last_support_debug_edges: list[tuple[int, int]] = []
        self.last_connectivity: dict[str, float | int | bool] = {}
        self.local_graph_update_index = 0
        self._update_local_heading_propagation(neighbors)
        self._densities(neighbors)
        initial_mean_density = float(
            np.mean([robot.density for robot in self.robots])
        )
        self.reference_density = initial_mean_density * 0.62
        self.last_mean_pressure_force = 0.0
        self.last_mean_repulsion_force = 0.0
        self.last_mean_lateral_sph_force = 0.0
        self.physics_frame_index = 0
        # EXP-044-only control latch.  Empty in open-loop mode.  Fixed robots
        # remain physical obstacles/neighbors, but receive no force or
        # integration update.
        self.fixed_robot_ids: set[int] = set()

    def fix_robot(self, robot_id: int) -> None:
        """Latch one robot as a fixed anchor without stopping the swarm."""
        robot = next(item for item in self.robots if item.robot_id == robot_id)
        self.fixed_robot_ids.add(robot_id)
        robot.propulsion_weight = 0.0
        robot.acceleration[:] = 0.0
        robot.velocity[:] = 0.0
        robot.observed_velocity[:] = 0.0

    def _create_robots(self):
        half = self.geometry.incoming_width/2
        left, right = -half+ROBOT_RADIUS+4*MAP_SCALE, half-ROBOT_RADIUS-4*MAP_SCALE
        per_row = int((right-left)//GRID_SPACING)+1
        entrance = self.geometry.entrance_y if self.geometry.entrance_y is not None else -42.0
        # Adapted from production create_grid_robots: the rear row is one Base
        # length minus the radius/clearance margin behind the entrance.
        bottom = entrance - (PRODUCTION_BASE_LENGTH - ROBOT_RADIUS - 7 * MAP_SCALE)
        robots = []
        for robot_id in range(ROBOT_COUNT):
            row, column = divmod(robot_id, per_row)
            if self.propulsion_mode == "local_forward" and row == ROBOT_COUNT // per_row and ROBOT_COUNT % per_row:
                partial_count = ROBOT_COUNT % per_row
                x = (column - 0.5 * (partial_count - 1)) * GRID_SPACING
            else:
                x = left + column * GRID_SPACING
            position = np.array([x, bottom+row*GRID_ROW_SPACING])
            robots.append(RobotState(robot_id, position, ingress_lane_x=float(position[0])))
        return robots

    def _select_initial_lidar_leader(self) -> int:
        """Select the fixed front-row robot nearest deployment lateral center."""
        front_y = max(robot.position[1] for robot in self.robots)
        front = [robot for robot in self.robots if abs(robot.position[1]-front_y) <= EPSILON]
        xs = [robot.position[0] for robot in self.robots]
        self.initial_front_center_x = 0.5 * (min(xs) + max(xs))
        return min(front,key=lambda robot:(abs(robot.position[0]-self.initial_front_center_x),robot.robot_id)).robot_id

    def _update_local_heading_propagation(self, support_neighbors=None) -> None:
        """Propagate leader heading over the current local communication graph.

        The leader/front-pack gap uses only rear-half-space relative positions.
        It never uses a global centroid, progress rank, map or Junction phase.
        """
        by_id = {robot.robot_id: robot for robot in self.robots}
        support_neighbors = support_neighbors or self._neighbors()
        communication_neighbors = self._range_neighbors(LOCAL_COMMUNICATION_RANGE)
        leader = by_id[self.lidar_robot_id]
        for robot in self.robots:
            robot.heading_parent_id = None
            robot.heading_hop = -1
            robot.propulsion_weight = 0.0
        leader.heading_hop = 0
        leader.propulsion_weight = 1.0
        queue = [leader.robot_id]
        propagation_edges = []
        while queue:
            parent_id = queue.pop(0)
            parent = by_id[parent_id]
            for peer in sorted(communication_neighbors[parent_id],key=lambda robot: robot.robot_id):
                if peer.heading_hop >= 0:
                    continue
                peer.heading_parent_id = parent_id
                peer.heading_hop = parent.heading_hop + 1
                peer.body_yaw_rad = parent.body_yaw_rad
                # All connected followers receive the same weak relayed drive.
                # A hop gradient stretched the long body despite having the
                # same mean weight; uniform 0.5 preserves that measured scale.
                peer.propulsion_weight = LOCAL_FOLLOWER_DRIVE_WEIGHT
                propagation_edges.append((parent_id, peer.robot_id))
                queue.append(peer.robot_id)

        component_size = sum(robot.heading_hop >= 0 for robot in self.robots)
        support_edge_count = sum(len(peers) for peers in support_neighbors.values()) // 2
        communication_edge_count = sum(len(peers) for peers in communication_neighbors.values()) // 2
        support_debug_edges = []
        for robot_id in sorted(support_neighbors):
            for peer in sorted(support_neighbors[robot_id], key=lambda item: item.robot_id):
                if robot_id < peer.robot_id and len(support_debug_edges) < 800:
                    support_debug_edges.append((robot_id, peer.robot_id))
        self.last_communication_edges = propagation_edges
        self.last_support_debug_edges = support_debug_edges
        self.last_connectivity = {
            "communication_range": LOCAL_COMMUNICATION_RANGE,
            "support_range": SMOOTHING_LENGTH,
            "leader_connected_component_size": component_size,
            "connected_to_leader_count": max(0, component_size - 1),
            "disconnected_count": len(self.robots) - component_size,
            "leader_max_hop": max((robot.heading_hop for robot in self.robots), default=-1),
            "communication_edge_count": communication_edge_count,
            "support_edge_count": support_edge_count,
        }
        self._update_leader_gap(support_neighbors)

    def _update_leader_gap(self, support_neighbors) -> None:
        """Update gap-aware leader drive from the current one-hop support set."""
        leader = next(robot for robot in self.robots if robot.robot_id == self.lidar_robot_id)
        forward = np.array([math.cos(leader.body_yaw_rad), math.sin(leader.body_yaw_rad)])
        direct_followers = support_neighbors[leader.robot_id]
        rear_followers = [
            peer for peer in direct_followers
            if float(np.dot(peer.position - leader.position, forward)) <= 0.0
        ]
        nearest_all = min(
            (float(np.linalg.norm(peer.position - leader.position)) for peer in direct_followers),
            default=math.inf,
        )
        front_gap = min(
            (float(np.linalg.norm(peer.position - leader.position)) for peer in rear_followers),
            default=math.inf,
        )
        if front_gap <= VISCOELASTIC_LINK_RADIUS:
            leader_drive_scale = 1.0
        elif front_gap < SMOOTHING_LENGTH:
            leader_drive_scale = max(
                LOCAL_FOLLOWER_DRIVE_WEIGHT,
                _smoothstep(
                    (SMOOTHING_LENGTH - front_gap)
                    / (SMOOTHING_LENGTH - VISCOELASTIC_LINK_RADIUS)
                ),
            )
        else:
            # Matching follower drive avoids a rear pack piling into a stopped
            # leader while still removing the leader's 2:1 force advantage.
            leader_drive_scale = LOCAL_FOLLOWER_DRIVE_WEIGHT
        leader.propulsion_weight = leader_drive_scale
        self.last_connectivity.update({
            "leader_to_nearest_follower_distance": nearest_all,
            "leader_to_front_pack_gap": front_gap,
            "normalized_leader_gap": front_gap / LOCAL_COMMUNICATION_RANGE,
            "leader_connected": bool(rear_followers),
            "leader_communication_connected": self.last_connectivity.get("leader_connected_component_size", 0) > 1,
            "leader_drive_scale": leader_drive_scale,
            "leader_forward_speed": float(np.dot(leader.velocity, forward)),
        })

    def _grid(self):
        grid = {}
        for robot in self.robots:
            key = tuple(np.floor(robot.position/SMOOTHING_LENGTH).astype(int))
            grid.setdefault(key, []).append(robot)
        return grid

    def _range_neighbors(self, radius: float):
        """Return the current distance graph at a local physical radius."""
        cell_size = radius
        grid = {}
        for robot in self.robots:
            key = tuple(np.floor(robot.position / cell_size).astype(int))
            grid.setdefault(key, []).append(robot)
        result = {}
        for robot in self.robots:
            key = tuple(np.floor(robot.position / cell_size).astype(int)); nearby=[]
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    for peer in grid.get((key[0]+dx,key[1]+dy),()):
                        if peer is not robot and np.linalg.norm(peer.position-robot.position) <= radius:
                            nearby.append(peer)
            result[robot.robot_id] = nearby
        return result

    def _neighbors(self):
        """Return the SPH support graph (not the communication graph)."""
        return self._range_neighbors(SMOOTHING_LENGTH)

    def _densities(self, neighbors):
        self_value = _kernel(0.0)
        for robot in self.robots:
            robot.density = max(EPSILON, self_value+sum(_kernel(float(np.linalg.norm(peer.position-robot.position))) for peer in neighbors[robot.robot_id]))

    def _local_wall_confinement(self, robot: RobotState) -> bool:
        """Detect a corridor from two body-lateral wall ranges only."""
        lateral = np.array([-math.sin(robot.body_yaw_rad),math.cos(robot.body_yaw_rad)])
        ranges=[]
        for direction in (lateral,-lateral):
            hits=[hit for wall in self.geometry.walls if (hit:=LidarSensor._ray_hit(robot.position,direction,wall)) is not None]
            ranges.append(min(hits) if hits else math.inf)
        return all(math.isfinite(value) for value in ranges) and sum(ranges) <= LOCAL_WALL_CONFINEMENT_MAX_WIDTH


    def step(self) -> None:
        """Unchanged local-forward branch of the modular physics integrator."""
        neighbors = self._neighbors()
        self._densities(neighbors)
        if (
            self.local_graph_update_index
            % max(1, round(SAMPLE_PERIOD / DT))
            == 0
        ):
            self._update_local_heading_propagation(neighbors)
        else:
            self._update_leader_gap(neighbors)
        self.local_graph_update_index += 1

        for robot in self.robots:
            ratio = robot.density / max(self.reference_density, EPSILON)
            raw_pressure = max(
                0.0,
                PRESSURE_GAIN
                * robot.density
                * (ratio**STIFFNESS_EXPONENT - 1.0),
            )
            robot.pressure = raw_pressure * SPH_PRESSURE_SCALE

        accelerations: dict[int, np.ndarray] = {}
        equilibrium = SAFE_RADIUS * NORMAL_EQUILIBRIUM_SCALE
        pressure_magnitudes: list[float] = []
        repulsion_magnitudes: list[float] = []
        lateral_sph_magnitudes: list[float] = []

        for robot in self.robots:
            if robot.robot_id in self.fixed_robot_ids:
                accelerations[robot.robot_id] = np.zeros(2)
                continue
            pressure = np.zeros(2)
            viscosity = np.zeros(2)
            elastic = np.zeros(2)
            repulsion = np.zeros(2)
            for peer in neighbors[robot.robot_id]:
                offset = robot.position - peer.position
                distance = float(np.linalg.norm(offset))
                if distance <= EPSILON:
                    continue
                gradient = _gradient(offset)
                coefficient = (
                    robot.pressure / max(robot.density**2, EPSILON)
                    + peer.pressure / max(peer.density**2, EPSILON)
                )
                pressure += -coefficient * gradient
                weight = _kernel(distance) / max(_kernel(0.0), EPSILON)
                viscosity += (
                    peer.velocity - robot.velocity
                ) * (VELOCITY_CONSENSUS_GAIN * weight)

                relative_velocity = robot.velocity - peer.velocity
                approach = float(np.dot(relative_velocity, offset))
                if approach < 0.0:
                    distance_sq = distance * distance
                    mu = (
                        SMOOTHING_LENGTH
                        * approach
                        / (
                            distance_sq
                            + 0.01 * SMOOTHING_LENGTH**2
                        )
                    )
                    sound_i_sq = (
                        robot.pressure + PRESSURE_GAIN * robot.density
                    ) / max(robot.density, EPSILON)
                    sound_j_sq = (
                        peer.pressure + PRESSURE_GAIN * peer.density
                    ) / max(peer.density, EPSILON)
                    sound = 0.5 * (
                        math.sqrt(max(sound_i_sq, 0.0))
                        + math.sqrt(max(sound_j_sq, 0.0))
                    )
                    mean_density = 0.5 * (
                        robot.density + peer.density
                    )
                    artificial = (
                        -VISCOSITY_XI1 * sound * mu
                        + VISCOSITY_XI2 * mu**2
                    ) / max(mean_density, EPSILON)
                    viscosity += -artificial * gradient

                if distance <= VISCOELASTIC_LINK_RADIUS:
                    pair = tuple(
                        sorted((robot.robot_id, peer.robot_id))
                    )
                    rest = self.rest_lengths.setdefault(
                        pair,
                        float(
                            np.clip(
                                equilibrium,
                                VISCOELASTIC_REST_MIN,
                                VISCOELASTIC_REST_MAX,
                            )
                        ),
                    )
                    if robot.robot_id < peer.robot_id:
                        target_rest = float(
                            np.clip(
                                equilibrium,
                                VISCOELASTIC_REST_MIN,
                                VISCOELASTIC_REST_MAX,
                            )
                        )
                        rest += (
                            target_rest - rest
                        ) * min(1.0, 4.0 * DT)
                        rest += (
                            distance - rest
                        ) * min(1.0, 0.85 * DT)
                        self.rest_lengths[pair] = rest
                    radial = float(
                        np.dot(
                            robot.velocity - peer.velocity,
                            offset / distance,
                        )
                    )
                    elastic += (
                        offset
                        / distance
                        * (
                            -VISCOELASTIC_ELASTIC_GAIN
                            * (distance - rest)
                            - VISCOELASTIC_DASHPOT_GAIN * radial
                        )
                    )
                if distance < equilibrium:
                    repulsion += (
                        REPULSION_GAIN
                        * (equilibrium - distance)
                        / equilibrium
                        * offset
                        / distance
                    )

            pressure = _limit(pressure, PRESSURE_FORCE_LIMIT)
            viscosity = _limit(viscosity, VISCOSITY_FORCE_LIMIT)
            elastic = _limit(elastic, VISCOELASTIC_FORCE_LIMIT)
            sph_force = pressure + viscosity + elastic + repulsion
            pressure_magnitudes.append(float(np.linalg.norm(pressure)))
            repulsion_magnitudes.append(float(np.linalg.norm(repulsion)))
            lateral_sph_magnitudes.append(abs(float(sph_force[0])))
            route = np.array(
                [
                    math.cos(robot.body_yaw_rad),
                    math.sin(robot.body_yaw_rad),
                ]
            ) * LOCAL_FORWARD_DRIVE_FORCE * robot.propulsion_weight
            total = (
                pressure
                + viscosity
                + elastic
                + repulsion
                + route
                - DAMPING * robot.velocity
            )
            raw = _limit(total, MAX_ACCELERATION)
            accelerations[robot.robot_id] = (
                (1.0 - ACCELERATION_FILTER_ALPHA)
                * robot.acceleration
                + ACCELERATION_FILTER_ALPHA * raw
            )

        self.last_mean_pressure_force = float(
            np.mean(pressure_magnitudes)
        )
        self.last_mean_repulsion_force = float(
            np.mean(repulsion_magnitudes)
        )
        self.last_mean_lateral_sph_force = float(
            np.mean(lateral_sph_magnitudes)
        )
        for robot in self.robots:
            old = robot.position.copy()
            if robot.robot_id in self.fixed_robot_ids:
                robot.propulsion_weight = 0.0
                robot.acceleration[:] = 0.0
                robot.velocity[:] = 0.0
                robot.observed_velocity[:] = 0.0
                continue
            robot.acceleration = accelerations[robot.robot_id]
            robot.velocity += robot.acceleration * DT
            if self._local_wall_confinement(robot):
                robot.velocity[0] *= math.exp(
                    -CORRIDOR_LATERAL_VELOCITY_DAMPING * DT
                )
            robot.velocity = _limit(robot.velocity, MAX_SPEED)

            x_candidate = np.array(
                [
                    robot.position[0] + robot.velocity[0] * DT,
                    robot.position[1],
                ]
            )
            if self.geometry.walkable(x_candidate):
                robot.position[0] = x_candidate[0]
            else:
                self.wall_contacts += 1
                robot.velocity[0] = (
                    -robot.velocity[0] * INITIAL_WALL_RESTITUTION
                    if self.time >= BASE_COMPRESSION_DURATION
                    else 0.0
                )

            y_candidate = np.array(
                [
                    robot.position[0],
                    robot.position[1] + robot.velocity[1] * DT,
                ]
            )
            if self.geometry.walkable(y_candidate):
                robot.position[1] = y_candidate[1]
            else:
                self.wall_contacts += 1
                robot.velocity[1] = (
                    -robot.velocity[1] * INITIAL_WALL_RESTITUTION
                    if self.time >= BASE_COMPRESSION_DURATION
                    else 0.0
                )
            if not self.geometry.walkable(robot.position):
                self.wall_corrections += 1
            robot.observed_velocity = (robot.position - old) / DT

        self.time += DT
        self.physics_frame_index += 1

    def local_observation(self) -> LocalObservation:
        lidar=next(robot for robot in self.robots if robot.robot_id==self.lidar_robot_id)
        if self.propulsion_mode == "local_forward":
            # Profile angles are defined in the proprioceptive body frame.
            # Observed velocity can contain lateral SPH motion and therefore
            # must not rotate the LiDAR reference axis.
            self.lidar_yaw_deg=math.degrees(lidar.body_yaw_rad)
        elif np.linalg.norm(lidar.observed_velocity) > MIN_SPEED:
            candidate=math.degrees(math.atan2(lidar.observed_velocity[1],lidar.observed_velocity[0]))
            delta=(candidate-self.lidar_yaw_deg+180)%360-180
            if abs(delta)<90: self.lidar_yaw_deg=candidate
        scan=self.sensor.scan(self.geometry,lidar.position,self.lidar_yaw_deg)
        positions=np.array([robot.position-lidar.position for robot in self.robots])
        velocities=np.array([robot.observed_velocity for robot in self.robots])
        return LocalObservation(self.time,np.array([robot.robot_id for robot in self.robots]),positions,velocities,self.lidar_robot_id,scan)




class LocalObservationBuilder:
    """Explicit world-to-runtime boundary."""
    @staticmethod
    def build(world: SimulatorWorld) -> LocalObservation:
        return world.local_observation()
# =============================================================================
# 2. Frozen adaptive 55% opening detector
# Extracted from pointcloud_junction_detector_sensor_enhanced.py.
# =============================================================================

DETECTOR_EPSILON = 1.0e-10

def _normalize_angles(angles_deg: Any) -> Any:
    normalized = (np.asarray(angles_deg) + 180.0) % 360.0 - 180.0
    if np.ndim(angles_deg) == 0:
        return float(normalized)
    return normalized


def _validate_circular_scan(
    angles_deg: Sequence[float], ranges: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.asarray(angles_deg, dtype=float)
    values = np.asarray(ranges, dtype=float)
    if angles.ndim != 1 or values.ndim != 1 or angles.shape != values.shape:
        raise ValueError("angles_deg and ranges must be equal-length 1D arrays")
    if angles.size < 8:
        raise ValueError("at least 8 LiDAR rays are required")
    if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(values)):
        raise ValueError("angles/ranges must contain finite values")
    if np.any(values < 0.0):
        raise ValueError("ranges cannot be negative")
    if np.any(np.diff(angles) <= 0.0):
        raise ValueError("angles_deg must be strictly increasing")

    steps = np.diff(np.r_[angles, angles[0] + 360.0])
    if np.any(steps <= 0.0):
        raise ValueError("angles must describe one circular revolution")
    if abs(float(np.sum(steps)) - 360.0) > max(1.0, 2.0 * float(np.median(steps))):
        raise ValueError("detector expects an approximately 360-degree scan")
    return angles, values, steps


def smooth_ranges(ranges: Sequence[float], window_size: int = 5) -> np.ndarray:
    """Centered circular moving average."""
    values = np.asarray(ranges, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("ranges must be a non-empty 1D sequence")
    if not isinstance(window_size, (int, np.integer)) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if window_size > values.size:
        raise ValueError("window_size cannot exceed scan length")
    half = window_size // 2
    return np.mean([np.roll(values, s) for s in range(-half, half + 1)], axis=0)


def circular_range_gradient(
    angles_deg: Sequence[float], ranges: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Forward circular range gradient in m/degree at ray boundaries."""
    angles, values, steps = _validate_circular_scan(angles_deg, ranges)
    gradient = (np.roll(values, -1) - values) / steps
    boundary_angles = _normalize_angles(angles + 0.5 * steps)
    return boundary_angles, gradient


def _automatic_gradient_threshold(
    gradient: np.ndarray,
    mad_scale: float,
    minimum: float,
) -> float:
    magnitudes = np.abs(gradient)
    median = float(np.median(magnitudes))
    mad = float(np.median(np.abs(magnitudes - median)))
    robust_sigma = 1.4826 * mad
    return max(float(minimum), median + mad_scale * robust_sigma)


def _circular_runs(mask: np.ndarray, value: bool = True) -> list[np.ndarray]:
    """Return circular connected components of a boolean mask as index arrays."""
    mask = np.asarray(mask, dtype=bool)
    n = mask.size
    if n == 0:
        return []
    target = mask == value
    if not np.any(target):
        return []
    if np.all(target):
        return [np.arange(n, dtype=int)]

    starts = np.flatnonzero(target & ~np.roll(target, 1))
    runs: list[np.ndarray] = []
    for start in starts:
        indices = [int(start)]
        cursor = (int(start) + 1) % n
        while target[cursor] and cursor != start:
            indices.append(cursor)
            cursor = (cursor + 1) % n
        runs.append(np.asarray(indices, dtype=int))
    return runs


def _run_width_deg(run: np.ndarray, angular_steps: np.ndarray) -> float:
    return float(np.sum(angular_steps[run]))


def _fill_short_circular_gaps(
    mask: np.ndarray,
    angular_steps: np.ndarray,
    max_gap_deg: float,
) -> np.ndarray:
    """Fill short false runs surrounded by open samples."""
    if max_gap_deg <= 0.0:
        return mask.copy()
    result = np.asarray(mask, dtype=bool).copy()
    if np.all(result) or not np.any(result):
        return result
    for run in _circular_runs(result, value=False):
        if _run_width_deg(run, angular_steps) <= max_gap_deg:
            before = (int(run[0]) - 1) % result.size
            after = (int(run[-1]) + 1) % result.size
            if result[before] and result[after]:
                result[run] = True
    return result


def _boundary_angle_before_ray(
    ray_index: int,
    boundary_angles: np.ndarray,
) -> float:
    return float(boundary_angles[(ray_index - 1) % boundary_angles.size])


def _boundary_angle_after_ray(
    ray_index: int,
    boundary_angles: np.ndarray,
) -> float:
    return float(boundary_angles[ray_index % boundary_angles.size])


def _circular_index_window(center: int, radius: int, n: int) -> np.ndarray:
    offsets = np.arange(-radius, radius + 1, dtype=int)
    return (center + offsets) % n


def _refine_boundary_from_gradient(
    target_gradient_index: int,
    gradient: np.ndarray,
    boundary_angles: np.ndarray,
    *,
    positive: bool,
    search_radius_samples: int,
    minimum_strength: float,
    fallback_angle: float,
) -> tuple[float, float, bool]:
    candidates = _circular_index_window(
        target_gradient_index, search_radius_samples, gradient.size
    )
    local = gradient[candidates]
    if positive:
        best_local = int(np.argmax(local))
        strength = float(local[best_local])
        valid = strength >= minimum_strength
    else:
        best_local = int(np.argmin(local))
        strength = float(-local[best_local])
        valid = strength >= minimum_strength
    if not valid:
        return float(fallback_angle), max(0.0, strength), False
    idx = int(candidates[best_local])
    return float(boundary_angles[idx]), strength, True


def _positive_ccw_width(start_angle: float, end_angle: float) -> float:
    return float((end_angle - start_angle) % 360.0)


def _detect_openings_with_diagnostics(
    angles_deg: Sequence[float],
    ranges: Sequence[float],
    *,
    smoothing_window_size: int = 5,
    wall_reference_quantile: float = 0.25,
    far_range_fraction: float = 0.55,
    merge_gap_deg: float = 3.0,
    min_opening_width_deg: float = 5.0,
    gradient_threshold: Optional[float] = None,
    gradient_mad_scale: float = 4.0,
    min_gradient_threshold: float = 0.05,
    boundary_search_deg: float = 6.0,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Detect an arbitrary number of openings using only local angle/range.

    No expected way count or expected direction appears anywhere in this
    function. The number of outputs is the number of connected open angular
    components found in the current scan.

    Method
    ------
    1) circular moving-average smoothing
    2) infer near-wall reference and far-range ceiling from the scan itself
    3) classify angular samples that are sufficiently far as "open support"
    4) merge only short internal gaps
    5) each remaining circular connected component is one opening candidate
    6) refine candidate start/end with local positive/negative range gradients
    """
    angles, raw, angular_steps = _validate_circular_scan(angles_deg, ranges)

    if smoothing_window_size <= 0 or smoothing_window_size % 2 == 0:
        raise ValueError("smoothing_window_size must be a positive odd integer")
    if not 0.0 <= wall_reference_quantile < 1.0:
        raise ValueError("wall_reference_quantile must be in [0,1)")
    if not 0.0 < far_range_fraction < 1.0:
        raise ValueError("far_range_fraction must be in (0,1)")
    if merge_gap_deg < 0.0:
        raise ValueError("merge_gap_deg must be non-negative")
    if not 0.0 < min_opening_width_deg < 360.0:
        raise ValueError("min_opening_width_deg must be in (0,360)")
    if boundary_search_deg < 0.0:
        raise ValueError("boundary_search_deg must be non-negative")

    smoothed = smooth_ranges(raw, smoothing_window_size)

    # These are inferred from the scan; the detector is not passed sensor/map metadata.
    wall_reference = float(np.quantile(smoothed, wall_reference_quantile))
    range_ceiling = float(np.max(raw))
    dynamic_span = max(0.0, range_ceiling - wall_reference)

    # If there is no meaningful contrast, no opening can be supported by this baseline.
    if dynamic_span <= 1.0e-6:
        diagnostics = {
            "smoothed_ranges": smoothed,
            "open_support_mask": np.zeros(raw.size, dtype=bool),
            "open_threshold": range_ceiling,
            "wall_reference": wall_reference,
            "range_ceiling": range_ceiling,
            "boundary_angles": np.array([], dtype=float),
            "gradient": np.array([], dtype=float),
            "gradient_threshold": 0.0,
            "start_angles": [],
            "end_angles": [],
        }
        return [], diagnostics

    open_threshold = wall_reference + far_range_fraction * dynamic_span
    open_support = smoothed >= open_threshold
    open_support = _fill_short_circular_gaps(
        open_support, angular_steps, merge_gap_deg
    )

    boundary_angles, gradient = circular_range_gradient(angles, smoothed)
    grad_threshold = (
        float(gradient_threshold)
        if gradient_threshold is not None
        else _automatic_gradient_threshold(
            gradient, gradient_mad_scale, min_gradient_threshold
        )
    )

    median_step = float(np.median(angular_steps))
    search_radius_samples = int(np.ceil(boundary_search_deg / median_step))

    openings: list[dict[str, float]] = []
    for run in _circular_runs(open_support, value=True):
        coarse_width = _run_width_deg(run, angular_steps)
        if coarse_width < min_opening_width_deg:
            continue
        if coarse_width >= 359.0:
            # Entire scan is far: there is no observable wall/opening separation.
            continue

        start_ray = int(run[0])
        end_ray = int(run[-1])
        coarse_start = _boundary_angle_before_ray(start_ray, boundary_angles)
        coarse_end = _boundary_angle_after_ray(end_ray, boundary_angles)

        start_grad_index = (start_ray - 1) % gradient.size
        end_grad_index = end_ray % gradient.size

        start_angle, start_strength, start_refined = _refine_boundary_from_gradient(
            start_grad_index,
            gradient,
            boundary_angles,
            positive=True,
            search_radius_samples=search_radius_samples,
            minimum_strength=grad_threshold,
            fallback_angle=coarse_start,
        )
        end_angle, end_strength, end_refined = _refine_boundary_from_gradient(
            end_grad_index,
            gradient,
            boundary_angles,
            positive=False,
            search_radius_samples=search_radius_samples,
            minimum_strength=grad_threshold,
            fallback_angle=coarse_end,
        )

        width = _positive_ccw_width(start_angle, end_angle)
        # Gradient refinement can jump to a neighboring lobe under heavy noise.
        # Fall back to the connected-component boundaries if that becomes implausible.
        if width < min_opening_width_deg or width > min(359.0, coarse_width + 2.0 * boundary_search_deg + 2.0):
            start_angle = coarse_start
            end_angle = coarse_end
            width = _positive_ccw_width(start_angle, end_angle)
            start_refined = False
            end_refined = False

        if width < min_opening_width_deg:
            continue

        center_angle = float(_normalize_angles(start_angle + width / 2.0))
        mean_range = float(np.mean(smoothed[run]))
        peak_range = float(np.max(smoothed[run]))
        contrast_score = float(
            np.clip((mean_range - wall_reference) / max(dynamic_span, DETECTOR_EPSILON), 0.0, 1.0)
        )
        boundary_score = float(
            np.clip(
                min(start_strength, end_strength) / max(grad_threshold, DETECTOR_EPSILON),
                0.0,
                1.0,
            )
        )
        confidence = float(0.7 * contrast_score + 0.3 * boundary_score)

        openings.append(
            {
                "start_angle": float(_normalize_angles(start_angle)),
                "end_angle": float(_normalize_angles(end_angle)),
                "center_angle": center_angle,
                "width_deg": float(width),
                "mean_range_m": mean_range,
                "peak_range_m": peak_range,
                "confidence": confidence,
                "start_refined": float(start_refined),
                "end_refined": float(end_refined),
            }
        )

    openings.sort(key=lambda item: item["center_angle"])
    diagnostics: dict[str, Any] = {
        "smoothed_ranges": smoothed,
        "open_support_mask": open_support,
        "open_threshold": float(open_threshold),
        "wall_reference": float(wall_reference),
        "range_ceiling": float(range_ceiling),
        "boundary_angles": boundary_angles,
        "gradient": gradient,
        "gradient_threshold": float(grad_threshold),
        "start_angles": [opening["start_angle"] for opening in openings],
        "end_angles": [opening["end_angle"] for opening in openings],
    }
    return openings, diagnostics


DEFAULT_NOISE_FRACTION = 0.05
DEFAULT_NOISE_SEED = 17
DEFAULT_THRESHOLD_ALPHA = 0.5
DEFAULT_NOISE_SIGMA = 2.5
DEFAULT_SWEEP_ALPHAS = tuple(index / 20.0 for index in range(1, 20))
THRESHOLD_MODES = ("w-tau", "frozen-55")
NOISE_MODELS = ("none", "truncated-gaussian")
SYSTEM_STATE_MOVING = "MOVING"
SYSTEM_STATE_JUNCTION_CONFIRMED = "JUNCTION_CONFIRMED"
SYSTEM_STATE_FIXED_ANCHOR = "FIXED_ANCHOR"
TRANSITION_CONFIRM_AND_FIX = "CONFIRM_AND_FIX"
TRANSITION_CONFIRM_OPEN_LOOP = "CONFIRM_ONLY_OPEN_LOOP"


def compute_safe_threshold_interval(
    worst_wall_range: float,
    max_range: float,
    tau: float,
) -> tuple[float, float, bool]:
    """Return the strict W–tau interval.

    W is a calibration parameter representing the worst-case maximum
    wall-return distance before a ray becomes an Opening.  It is not estimated
    online and no map/pose information enters this function.
    """
    lower_bound = float(worst_wall_range + tau)
    upper_bound = float(max_range - tau)
    return lower_bound, upper_bound, lower_bound < upper_bound


def select_threshold_in_safe_interval(
    lower_bound: float,
    upper_bound: float,
    interval_valid: bool,
    threshold_alpha: float,
) -> float | None:
    """Select an experimental point; alpha=0.5 is not claimed optimal."""
    if not 0.0 < threshold_alpha < 1.0:
        raise ValueError("threshold_alpha must be strictly between 0 and 1")
    if not interval_valid:
        return None
    return float(
        lower_bound
        + threshold_alpha * (upper_bound - lower_bound)
    )


def _detect_openings_w_tau_with_diagnostics(
    angles_deg: Sequence[float],
    ranges: Sequence[float],
    *,
    selected_threshold: float | None,
    threshold_interval_valid: bool,
    smoothing_window_size: int = 5,
    merge_gap_deg: float = 3.0,
    min_opening_width_deg: float = 5.0,
    gradient_threshold: Optional[float] = None,
    gradient_mad_scale: float = 4.0,
    min_gradient_threshold: float = 0.05,
    boundary_search_deg: float = 6.0,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Apply W–tau support and the frozen opening/group/refinement pipeline."""
    angles, raw, angular_steps = _validate_circular_scan(angles_deg, ranges)
    if smoothing_window_size <= 0 or smoothing_window_size % 2 == 0:
        raise ValueError("smoothing_window_size must be a positive odd integer")
    if merge_gap_deg < 0.0:
        raise ValueError("merge_gap_deg must be non-negative")
    if not 0.0 < min_opening_width_deg < 360.0:
        raise ValueError("min_opening_width_deg must be in (0,360)")
    if boundary_search_deg < 0.0:
        raise ValueError("boundary_search_deg must be non-negative")

    smoothed = smooth_ranges(raw, smoothing_window_size)
    # These scan statistics are diagnostics/confidence normalization only.
    # They do not select the W–tau threshold.
    wall_reference = float(np.quantile(smoothed, 0.25))
    range_ceiling = float(np.max(raw))
    dynamic_span = max(0.0, range_ceiling - wall_reference)
    if not threshold_interval_valid or selected_threshold is None:
        return [], {
            "smoothed_ranges": smoothed,
            "open_support_mask": np.zeros(raw.size, dtype=bool),
            "open_threshold": math.nan,
            "wall_reference": wall_reference,
            "range_ceiling": range_ceiling,
            "boundary_angles": np.array([], dtype=float),
            "gradient": np.array([], dtype=float),
            "gradient_threshold": 0.0,
            "start_angles": [],
            "end_angles": [],
        }

    open_support = smoothed >= float(selected_threshold)
    open_support = _fill_short_circular_gaps(
        open_support,
        angular_steps,
        merge_gap_deg,
    )
    boundary_angles, gradient = circular_range_gradient(angles, smoothed)
    grad_threshold = (
        float(gradient_threshold)
        if gradient_threshold is not None
        else _automatic_gradient_threshold(
            gradient,
            gradient_mad_scale,
            min_gradient_threshold,
        )
    )
    median_step = float(np.median(angular_steps))
    search_radius_samples = int(np.ceil(boundary_search_deg / median_step))

    openings: list[dict[str, float]] = []
    for run in _circular_runs(open_support, value=True):
        coarse_width = _run_width_deg(run, angular_steps)
        if coarse_width < min_opening_width_deg or coarse_width >= 359.0:
            continue
        start_ray = int(run[0])
        end_ray = int(run[-1])
        coarse_start = _boundary_angle_before_ray(start_ray, boundary_angles)
        coarse_end = _boundary_angle_after_ray(end_ray, boundary_angles)
        start_angle, start_strength, start_refined = (
            _refine_boundary_from_gradient(
                (start_ray - 1) % gradient.size,
                gradient,
                boundary_angles,
                positive=True,
                search_radius_samples=search_radius_samples,
                minimum_strength=grad_threshold,
                fallback_angle=coarse_start,
            )
        )
        end_angle, end_strength, end_refined = (
            _refine_boundary_from_gradient(
                end_ray % gradient.size,
                gradient,
                boundary_angles,
                positive=False,
                search_radius_samples=search_radius_samples,
                minimum_strength=grad_threshold,
                fallback_angle=coarse_end,
            )
        )
        width = _positive_ccw_width(start_angle, end_angle)
        if (
            width < min_opening_width_deg
            or width
            > min(
                359.0,
                coarse_width + 2.0 * boundary_search_deg + 2.0,
            )
        ):
            start_angle = coarse_start
            end_angle = coarse_end
            width = _positive_ccw_width(start_angle, end_angle)
            start_refined = False
            end_refined = False
        if width < min_opening_width_deg:
            continue

        center_angle = float(_normalize_angles(start_angle + width / 2.0))
        mean_range = float(np.mean(smoothed[run]))
        peak_range = float(np.max(smoothed[run]))
        contrast_score = float(
            np.clip(
                (mean_range - wall_reference)
                / max(dynamic_span, DETECTOR_EPSILON),
                0.0,
                1.0,
            )
        )
        boundary_score = float(
            np.clip(
                min(start_strength, end_strength)
                / max(grad_threshold, DETECTOR_EPSILON),
                0.0,
                1.0,
            )
        )
        openings.append(
            {
                "start_angle": float(_normalize_angles(start_angle)),
                "end_angle": float(_normalize_angles(end_angle)),
                "center_angle": center_angle,
                "width_deg": float(width),
                "mean_range_m": mean_range,
                "peak_range_m": peak_range,
                "confidence": float(
                    0.7 * contrast_score + 0.3 * boundary_score
                ),
                "start_refined": float(start_refined),
                "end_refined": float(end_refined),
            }
        )

    openings.sort(key=lambda item: item["center_angle"])
    return openings, {
        "smoothed_ranges": smoothed,
        "open_support_mask": open_support,
        "open_threshold": float(selected_threshold),
        "wall_reference": wall_reference,
        "range_ceiling": range_ceiling,
        "boundary_angles": boundary_angles,
        "gradient": gradient,
        "gradient_threshold": float(grad_threshold),
        "start_angles": [opening["start_angle"] for opening in openings],
        "end_angles": [opening["end_angle"] for opening in openings],
    }


# =============================================================================
# 3. Standalone sampled runner and validated EXP-042 rear-start
# =============================================================================

M0_CASE = "M0_STRAIGHT"
M1_BASELINE_CASE = "M1_CROSS_BASELINE"
M1_PRE_CORRIDOR_CASE = "M1_PRE_CORRIDOR_55PCT"
M2_CASE = "M2_T_JUNCTION"
M3_CASE = "M3_ANGLED_Y"
M4_CASE = "M4_ASYMMETRIC_CROSS"
M5_CASE = "M5_UNEQUAL_WIDTH"
MAP_CASES = (
    M0_CASE,
    M1_BASELINE_CASE,
    M1_PRE_CORRIDOR_CASE,
    M2_CASE,
    M3_CASE,
    M4_CASE,
    M5_CASE,
)
REAR_START_SHIFT = 160.0


class SimulationRunner:
    """Minimal sampled adapter around the unchanged SimulatorWorld."""

    def __init__(self, case_id: str) -> None:
        geometry_case = (
            M1_BASELINE_CASE
            if case_id == M1_PRE_CORRIDOR_CASE
            else case_id
        )
        self.geometry = GeometryBuilder.build(geometry_case)
        self.world = SimulatorWorld(self.geometry, "local_forward")
        self.last_visual: tuple[LocalObservation, dict[str, Any]] | None = None

    def step(self, frame: int) -> dict[str, Any] | None:
        self.world.step()
        if frame % max(1, round(SAMPLE_PERIOD / DT)):
            return None
        observation = LocalObservationBuilder.build(self.world)
        self.last_visual = (observation, {})
        return {
            "map_case": self.geometry.case_id,
            "frame": frame,
            "timestamp": self.world.time,
        }


def _rear_start(
    runner: SimulationRunner,
    shift: float = REAR_START_SHIFT,
) -> None:
    """Exact EXP-042 rear-start geometry and swarm-state semantics."""
    original = runner.geometry
    entrance = float(original.entrance_y)
    length = original.incoming_length + shift
    incoming = _rect(
        np.array([0.0, entrance - 0.5 * length]),
        np.array([0.0, 1.0]),
        original.incoming_width,
        length,
    )
    rects = (incoming,) + original.free_rects[1:]
    geometry = GeometryCase(
        original.case_id,
        original.incoming_width,
        length,
        original.junction_size,
        original.branches,
        rects,
        _union_boundary(rects),
        original.entrance_y,
    )
    runner.geometry = geometry
    runner.world.geometry = geometry
    for robot in runner.world.robots:
        robot.position[1] -= shift
    runner.world.initial_mean_y = float(
        np.mean([robot.position[1] for robot in runner.world.robots])
    )
    runner.world.initial_front_y = float(
        max(robot.position[1] for robot in runner.world.robots)
    )
    leader = next(
        robot
        for robot in runner.world.robots
        if robot.robot_id == runner.world.lidar_robot_id
    )
    runner.world.initial_lidar_position = leader.position.copy()


def _new_runner(map_case: str) -> SimulationRunner:
    runner = SimulationRunner(map_case)
    if map_case == M1_PRE_CORRIDOR_CASE:
        _rear_start(runner, REAR_START_SHIFT)
    return runner


# =============================================================================
# 4. Display constants and renderer base
# =============================================================================

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


class BaseRenderer:
    def __init__(self, pygame: Any, geometry: GeometryCase, show_profile: bool) -> None:
        self.pygame = pygame
        self.geometry = geometry
        self.show_profile = show_profile
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.font = pygame.font.Font(None, 22)
        self.small_font = pygame.font.Font(None, 19)
        self.title_font = pygame.font.Font(None, 30)
        self._configure_camera()

    def _configure_camera(self) -> None:
        vertices = np.asarray(
            [
                point
                for rect in self.geometry.free_rects
                for point in rect.vertices
            ],
            dtype=float,
        )
        minimum = np.min(vertices, axis=0)
        maximum = np.max(vertices, axis=0)
        span = np.maximum(maximum - minimum, 1.0)
        x, y, width, height = MAIN_RECT
        self.camera_center = 0.5 * (minimum + maximum)
        if self.geometry.case_id == M0_CASE:
            span[1] = min(span[1], 440.0)
        self.pixels_per_world = min(
            (width - 40) / (span[0] + 35),
            (height - 40) / (span[1] + 35),
        )
        self.main_center = np.array([x + width / 2, y + height / 2])

    def world_to_screen(
        self,
        point: np.ndarray | tuple[float, float],
        snapshot: Any,
    ) -> tuple[int, int]:
        center = self.camera_center.copy()
        if self.geometry.case_id == M0_CASE:
            center[1] = snapshot.leader_position[1]
        relative = np.asarray(point, dtype=float) - center
        screen = self.main_center + np.array(
            [relative[0], -relative[1]]
        ) * self.pixels_per_world
        return int(screen[0]), int(screen[1])

    def text(
        self,
        value: str,
        position: tuple[int, int],
        color: tuple[int, int, int] | None = None,
        font: Any = None,
    ) -> None:
        rendered = (font or self.font).render(
            value,
            True,
            color or COLORS["text"],
        )
        self.screen.blit(rendered, position)

    @staticmethod
    def _plot_point(
        rect: tuple[int, int, int, int],
        angle: float,
        value: float,
    ) -> tuple[int, int]:
        x, y, width, height = rect
        px = x + int((angle + 180.0) / 360.0 * width)
        py = y + height - int(
            np.clip(value / LIDAR_MAX_RANGE, 0.0, 1.0) * height
        )
        return px, py


# =============================================================================
# 5. Read-only adaptive session and Pygame GUI
# Extracted from lidar_junction_detection_threshold_visualizer.py.
# =============================================================================

ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME = "EXP-046 Pre-Junction Corridor-Based W Calibration"
DEFAULT_OUTPUT = ROOT / "lidar_junction_detection_corridor_w_calibration_output"
FROZEN_PARAMETERS = {
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
    "map_case", "frame", "timestamp", "threshold_mode", "noise_model",
    "noise_fraction", "noise_sigma", "noise_seed", "tau",
    "worst_wall_range", "w_tau_lower_bound", "w_tau_upper_bound",
    "threshold_interval_valid", "threshold_alpha", "selected_threshold",
    "frozen55_threshold", "frozen55_opening_count",
    "frozen55_junction_detected", "w_tau_opening_count",
    "w_tau_junction_detected", "active_opening_count",
    "active_opening_present", "active_junction_detected",
    "current_junction_evidence", "junction_confirmed",
    "confirmation_frame", "confirmation_time", "system_state",
    "state_transition",
    "active_open_support_count", "smoothing_window",
    "interval_threshold_count", "interval_junction_stable",
    "interval_opening_count_stable", "interval_junction_consistency",
    "interval_opening_count_consistency", "anchor_stop_on_detect",
    "anchor_fixed", "anchor_fix_frame", "anchor_fix_time",
    "post_fix_position_drift", "post_fix_max_position_drift",
    "post_fix_max_normal_motion",
    "leader_x_eval_only", "leader_y_eval_only", "runtime_gt_map_used",
)
COMPARISON_FIELDS = (
    "map_case", "frame", "timestamp", "noise_model",
    "frozen55_threshold", "w_tau_lower_bound", "w_tau_upper_bound",
    "w_tau_selected_threshold", "threshold_interval_valid",
    "frozen55_opening_count", "w_tau_opening_count",
    "frozen55_junction_detected", "w_tau_junction_detected",
    "active_threshold_mode", "active_opening_count",
    "active_junction_detected", "current_junction_evidence",
    "junction_confirmed", "confirmation_frame", "anchor_fixed",
)
BEAM_FIELDS = (
    "map_case", "frame", "timestamp", "theta_deg", "true_range",
    "noise_value", "measured_range", "smoothed_range",
    "active_open_support",
)
THRESHOLD_SWEEP_FRAME_FIELDS = (
    "map_case", "noise_model", "frame", "timestamp", "alpha",
    "threshold", "opening_count", "junction_detected", "anchor_mode",
)
LOCAL_W_CANDIDATE_FIELDS = (
    "map_case", "frame", "timestamp", "boundary_id", "boundary_side",
    "boundary_angle_deg", "far_run_start_angle", "far_run_end_angle",
    "immediate_wall_range", "local_wall_max", "local_wall_median",
    "local_wall_p90", "noise_model", "tau", "noise_seed",
    "estimator_input", "w_gt_eval_only", "evaluation_gt_used",
    "runtime_gt_map_used",
)
LOCAL_W_TIMELINE_FIELDS = (
    "map_case", "frame", "timestamp", "noise_model", "noise_sigma",
    "noise_seed", "tau", "w_est_available",
    "far_run_count", "w_candidate_count", "far_support_threshold",
    "w_est_max", "w_est_median", "w_est_p90", "w_history_median",
    "w_history_p90", "w_gt_eval_only", "w_error_max",
    "w_error_median", "w_error_p90", "active_w_source",
    "active_manual_w", "shadow_w_feedback_used", "estimator_input",
    "current_junction_evidence", "junction_confirmed", "anchor_fixed",
    "runtime_gt_map_used", "evaluation_gt_used",
)

CALIBRATION_STATES = (
    "W_CALIBRATION_SEARCH",
    "W_CALIBRATING",
    "W_READY_SHADOW",
    "W_FROZEN_SHADOW",
)
W_STATISTICS = ("max", "p90", "p95", "p99")
WALL_SUPPORT_DEFINITIONS = ("NON_FAR", "RUN_BOUNDARY_EXCLUDED")
POSE_CONDITIONS = ("center", "left", "right", "yaw-left", "yaw-right")


@dataclass(frozen=True)
class DetectorExperimentConfig:
    """Calibration and measurement-layer inputs; no runtime map/pose fields."""

    threshold_mode: str
    worst_wall_range: float
    noise_model: str = "truncated-gaussian"
    noise_fraction: float = DEFAULT_NOISE_FRACTION
    noise_sigma: float | None = DEFAULT_NOISE_SIGMA
    noise_seed: int = DEFAULT_NOISE_SEED
    threshold_alpha: float = DEFAULT_THRESHOLD_ALPHA
    smoothing_window: int = 5
    dump_beams: bool = False
    anchor_stop_on_detect: bool = True
    evaluate_interval: bool = True
    w_source: str = "manual"
    w_boundary_window_rays: int = 3
    w_history_scans: int = 5
    w_estimator_use_smoothed: bool = False
    w_calibration_history_scans: int = 10
    w_calibration_use_smoothed: bool = False
    corridor_dominant_min_width_deg: float = 5.0
    corridor_opposition_tolerance_deg: float = 35.0
    corridor_max_far_fraction: float = 0.70
    boundary_exclusion_rays: int = 3
    wall_support_definition: str = "RUN_BOUNDARY_EXCLUDED"
    pose_condition: str = "center"
    pose_lateral_offset: float = 4.0
    pose_yaw_offset_deg: float = 3.0
    pre_corridor_calibration_start: bool = True

    def __post_init__(self) -> None:
        if self.threshold_mode not in THRESHOLD_MODES:
            raise ValueError(f"unknown threshold mode: {self.threshold_mode}")
        if self.noise_model not in NOISE_MODELS:
            raise ValueError(f"unknown noise model: {self.noise_model}")
        if self.w_source != "manual":
            raise ValueError("EXP-046 active w_source must remain manual")
        if self.w_boundary_window_rays <= 0:
            raise ValueError("w_boundary_window_rays must be positive")
        if self.w_history_scans <= 0:
            raise ValueError("w_history_scans must be positive")
        if self.w_calibration_history_scans <= 0:
            raise ValueError("w_calibration_history_scans must be positive")
        if not 0.0 < self.corridor_dominant_min_width_deg < 180.0:
            raise ValueError("corridor dominant width must be in (0,180)")
        if not 0.0 <= self.corridor_opposition_tolerance_deg <= 90.0:
            raise ValueError("corridor opposition tolerance must be in [0,90]")
        if not 0.0 < self.corridor_max_far_fraction < 1.0:
            raise ValueError("corridor max far fraction must be in (0,1)")
        if self.boundary_exclusion_rays < 0:
            raise ValueError("boundary exclusion rays must be non-negative")
        if self.wall_support_definition not in WALL_SUPPORT_DEFINITIONS:
            raise ValueError("unknown wall support definition")
        if self.pose_condition not in POSE_CONDITIONS:
            raise ValueError("unknown pose condition")
        if self.pose_lateral_offset < 0.0 or self.pose_yaw_offset_deg < 0.0:
            raise ValueError("pose perturbation magnitudes must be non-negative")
        if not 0.0 <= self.worst_wall_range <= LIDAR_MAX_RANGE:
            raise ValueError("worst_wall_range must be within sensor range")
        if not 0.0 <= self.noise_fraction < 1.0:
            raise ValueError("noise_fraction must be in [0,1)")
        if not 0.0 < self.threshold_alpha < 1.0:
            raise ValueError("threshold_alpha must be strictly between 0 and 1")
        if (
            self.smoothing_window <= 0
            or self.smoothing_window % 2 == 0
        ):
            raise ValueError("smoothing_window must be a positive odd integer")
        if self.noise_model == "truncated-gaussian":
            if self.noise_sigma is None or self.noise_sigma <= 0.0:
                raise ValueError(
                    "truncated-gaussian noise requires positive noise_sigma"
                )

    @property
    def tau(self) -> float:
        return float(LIDAR_MAX_RANGE * self.noise_fraction)

    @property
    def safe_interval(self) -> tuple[float, float, bool]:
        return compute_safe_threshold_interval(
            self.worst_wall_range,
            LIDAR_MAX_RANGE,
            self.tau,
        )

    @property
    def selected_w_tau_threshold(self) -> float | None:
        lower, upper, valid = self.safe_interval
        return select_threshold_in_safe_interval(
            lower,
            upper,
            valid,
            self.threshold_alpha,
        )


def _audit_frozen_detector_defaults() -> dict[str, Any]:
    signature = inspect.signature(_detect_openings_with_diagnostics)
    actual = {
        name: signature.parameters[name].default
        for name in FROZEN_PARAMETERS
    }
    if actual != FROZEN_PARAMETERS:
        raise AssertionError(
            f"frozen detector defaults changed: "
            f"expected={FROZEN_PARAMETERS!r} actual={actual!r}"
        )
    return actual


def _audit_corridor_calibrator_local_only() -> None:
    actual = set(inspect.signature(calibrate_corridor_w_scan_shadow).parameters)
    allowed = {
        "angles_deg", "measured_ranges", "max_range", "tau",
        "dominant_min_width_deg", "opposition_tolerance_deg",
        "max_far_fraction", "boundary_exclusion_rays",
        "wall_support_definition",
    }
    if actual != allowed:
        raise AssertionError(
            f"corridor calibrator API must remain local-only: {actual!r}"
        )


def _write(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fields: Iterable[str],
) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(materialized)


def _inject_measurement_noise(
    true_ranges: np.ndarray,
    config: DetectorExperimentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if config.noise_model == "none":
        noise = np.zeros_like(true_ranges, dtype=float)
    else:
        # Rejection-sampled Gaussian conditioned on |noise| <= tau.  Sigma is
        # an experimental parameter (default 2.5), not a professor-specified
        # theoretical constant.  Unlike EXP-043's unbounded Gaussian option,
        # this EXP guarantees the meeting-defined hard +/-tau support.
        noise = np.empty_like(true_ranges, dtype=float)
        pending = np.arange(noise.size)
        flat = noise.reshape(-1)
        while pending.size:
            candidates = rng.normal(
                0.0,
                float(config.noise_sigma),
                size=pending.size,
            )
            accepted = np.abs(candidates) <= config.tau
            flat[pending[accepted]] = candidates[accepted]
            pending = pending[~accepted]
    measured = np.clip(
        np.asarray(true_ranges, dtype=float) + noise,
        0.0,
        LIDAR_MAX_RANGE,
    )
    return noise, measured


@dataclass(frozen=True)
class LocalWBoundaryCandidate:
    """One wall-side boundary observation adjacent to a local far run."""

    boundary_id: int
    boundary_side: str
    boundary_ray_index: int
    boundary_angle_deg: float
    far_run_start_angle: float
    far_run_end_angle: float
    immediate_wall_range: float
    local_wall_max: float
    local_wall_median: float
    local_wall_p90: float


@dataclass(frozen=True)
class LocalWEstimate:
    """Shadow W result; never consumed by the active Junction detector."""

    available: bool
    far_support_threshold: float
    far_mask: np.ndarray
    far_runs: tuple[tuple[int, ...], ...]
    candidates: tuple[LocalWBoundaryCandidate, ...]
    w_est_max: float | None
    w_est_median: float | None
    w_est_p90: float | None


def _extract_far_boundary_candidates(
    angles_deg: Sequence[float],
    ranges: Sequence[float],
    *,
    far_support_threshold: float,
    boundary_window_rays: int,
    min_far_width_deg: float,
) -> LocalWEstimate:
    """Extract wall-side candidates using only circular angle/range adjacency.

    This low-level function has no map, case, pose, Junction, or branch input.
    """
    angles, values, angular_steps = _validate_circular_scan(
        angles_deg, ranges
    )
    if boundary_window_rays <= 0:
        raise ValueError("boundary_window_rays must be positive")
    if not 0.0 < min_far_width_deg < 360.0:
        raise ValueError("min_far_width_deg must be in (0,360)")

    far_mask = values >= float(far_support_threshold)
    if np.all(far_mask):
        return LocalWEstimate(
            False,
            float(far_support_threshold),
            far_mask,
            (),
            (),
            None,
            None,
            None,
        )
    far_runs = tuple(
        tuple(int(index) for index in run)
        for run in _circular_runs(far_mask, value=True)
        if _run_width_deg(run, angular_steps) >= min_far_width_deg
    )
    candidates: list[LocalWBoundaryCandidate] = []
    n = values.size
    for run_id, run_tuple in enumerate(far_runs):
        run = np.asarray(run_tuple, dtype=int)
        for side, direction, edge in (
            ("START_PRECEDING", -1, int(run[0])),
            ("END_FOLLOWING", 1, int(run[-1])),
        ):
            indices = np.asarray(
                [
                    (edge + direction * offset) % n
                    for offset in range(1, boundary_window_rays + 1)
                ],
                dtype=int,
            )
            wall_indices = indices[~far_mask[indices]]
            if wall_indices.size == 0:
                continue
            immediate_index = int(wall_indices[0])
            local = values[wall_indices]
            candidates.append(
                LocalWBoundaryCandidate(
                    boundary_id=2 * run_id + (1 if direction > 0 else 0),
                    boundary_side=side,
                    boundary_ray_index=immediate_index,
                    boundary_angle_deg=float(angles[immediate_index]),
                    far_run_start_angle=float(angles[int(run[0])]),
                    far_run_end_angle=float(angles[int(run[-1])]),
                    immediate_wall_range=float(values[immediate_index]),
                    local_wall_max=float(np.max(local)),
                    local_wall_median=float(np.median(local)),
                    local_wall_p90=float(np.quantile(local, 0.90)),
                )
            )

    if not candidates:
        return LocalWEstimate(
            False,
            float(far_support_threshold),
            far_mask,
            far_runs,
            (),
            None,
            None,
            None,
        )
    immediate = np.asarray(
        [candidate.immediate_wall_range for candidate in candidates],
        dtype=float,
    )
    return LocalWEstimate(
        True,
        float(far_support_threshold),
        far_mask,
        far_runs,
        tuple(candidates),
        float(np.max(immediate)),
        float(np.median(immediate)),
        float(np.quantile(immediate, 0.90)),
    )


def estimate_local_w_shadow(
    angles_deg: Sequence[float],
    measured_ranges: Sequence[float],
    *,
    max_range: float,
    tau: float,
    boundary_window_rays: int,
    min_far_width_deg: float = 5.0,
) -> LocalWEstimate:
    """Estimate W from local LiDAR only; active detector feedback is forbidden."""
    return _extract_far_boundary_candidates(
        angles_deg,
        measured_ranges,
        far_support_threshold=float(max_range - tau),
        boundary_window_rays=boundary_window_rays,
        min_far_width_deg=min_far_width_deg,
    )


def evaluate_w_gt_from_true_scan_eval_only(
    angles_deg: Sequence[float],
    true_ranges: Sequence[float],
    *,
    max_range: float,
    min_far_width_deg: float = 5.0,
) -> LocalWEstimate:
    """Noise-free scan oracle for error reporting only, never decision input."""
    return _extract_far_boundary_candidates(
        angles_deg,
        true_ranges,
        far_support_threshold=float(max_range - DETECTOR_EPSILON),
        boundary_window_rays=1,
        min_far_width_deg=min_far_width_deg,
    )


def _circular_distance_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _robust_wall_statistics(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {name: None for name in W_STATISTICS}
    return {
        "max": float(np.max(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }


def calibrate_corridor_w_scan_shadow(
    angles_deg: Sequence[float],
    measured_ranges: Sequence[float],
    *,
    max_range: float,
    tau: float,
    dominant_min_width_deg: float,
    opposition_tolerance_deg: float,
    max_far_fraction: float,
    boundary_exclusion_rays: int,
    wall_support_definition: str,
) -> dict[str, Any]:
    """Classify/calibrate one scan from local angles and ranges only.

    There is intentionally no map case, global pose, GT, active threshold, or
    active Junction state parameter.  The returned values are shadow-only.
    """
    angles, values, angular_steps = _validate_circular_scan(
        angles_deg, measured_ranges
    )
    far_threshold = float(max_range - tau)
    far_mask = values >= far_threshold
    all_far_runs = tuple(
        tuple(int(i) for i in run)
        for run in _circular_runs(far_mask, value=True)
    )
    dominant_runs = tuple(
        run for run in all_far_runs
        if _run_width_deg(np.asarray(run), angular_steps)
        >= dominant_min_width_deg
    )
    centers = []
    widths = []
    for run in dominant_runs:
        run_array = np.asarray(run, dtype=int)
        radians = np.deg2rad(angles[run_array])
        centers.append(float(np.rad2deg(math.atan2(
            float(np.mean(np.sin(radians))),
            float(np.mean(np.cos(radians))),
        )) % 360.0))
        widths.append(float(_run_width_deg(run_array, angular_steps)))
    separation = (
        _circular_distance_deg(centers[0], centers[1])
        if len(centers) == 2 else None
    )
    far_fraction = float(np.mean(far_mask))
    corridor_like = bool(
        len(dominant_runs) == 2
        and separation is not None
        and abs(180.0 - separation) <= opposition_tolerance_deg
        and far_fraction <= max_far_fraction
    )

    non_far_mask = ~far_mask
    excluded_mask = np.zeros(values.size, dtype=bool)
    if boundary_exclusion_rays:
        for run in dominant_runs:
            for edge in (int(run[0]), int(run[-1])):
                for offset in range(1, boundary_exclusion_rays + 1):
                    excluded_mask[(edge - offset) % values.size] = True
                    excluded_mask[(edge + offset) % values.size] = True
    support_masks = {
        "NON_FAR": non_far_mask,
        "RUN_BOUNDARY_EXCLUDED": non_far_mask & ~excluded_mask,
    }
    selected_mask = support_masks[wall_support_definition]
    return {
        "corridor_like": corridor_like,
        "far_support_threshold": far_threshold,
        "far_mask": far_mask,
        "far_runs": all_far_runs,
        "dominant_far_runs": dominant_runs,
        "dominant_centers_deg": tuple(centers),
        "dominant_widths_deg": tuple(widths),
        "two_run_separation_deg": separation,
        "far_fraction": far_fraction,
        "excluded_boundary_mask": excluded_mask,
        "support_masks": support_masks,
        "wall_support_mask": selected_mask,
        "wall_statistics": _robust_wall_statistics(values[selected_mask]),
        "all_definition_statistics": {
            name: _robust_wall_statistics(values[mask])
            for name, mask in support_masks.items()
        },
    }


def _history_summary(history: Sequence[float]) -> dict[str, float | None]:
    values = np.asarray(history, dtype=float)
    if values.size == 0:
        return {"median": None, "p90": None, "mad": None, "span": None}
    median = float(np.median(values))
    return {
        "median": median,
        "p90": float(np.quantile(values, 0.90)),
        "mad": float(np.median(np.abs(values - median))),
        "span": float(np.max(values) - np.min(values)),
    }


def _evaluate_safe_interval(
    angles: np.ndarray,
    measured_ranges: np.ndarray,
    config: DetectorExperimentConfig,
) -> list[dict[str, Any]]:
    """Evaluate scalar detectors across strict interior alphas on one scan."""
    lower, upper, valid = config.safe_interval
    if not valid or not config.evaluate_interval:
        return []
    rows: list[dict[str, Any]] = []
    for alpha in DEFAULT_SWEEP_ALPHAS:
        threshold = select_threshold_in_safe_interval(
            lower,
            upper,
            valid,
            alpha,
        )
        openings, _ = _detect_openings_w_tau_with_diagnostics(
            angles.copy(),
            measured_ranges.copy(),
            selected_threshold=threshold,
            threshold_interval_valid=True,
            smoothing_window_size=config.smoothing_window,
        )
        rows.append(
            {
                "alpha": alpha,
                "threshold": float(threshold),
                "opening_count": len(openings),
                "junction_detected": len(openings) >= 3,
            }
        )
    return rows


def _interval_stability(
    rows: Sequence[dict[str, Any]],
) -> tuple[bool, bool, float, float, int]:
    if not rows:
        return False, False, 0.0, 0.0, 0
    junctions = [bool(row["junction_detected"]) for row in rows]
    counts = [int(row["opening_count"]) for row in rows]
    total = len(rows)
    junction_consistency = max(Counter(junctions).values()) / total
    opening_consistency = max(Counter(counts).values()) / total
    return (
        len(set(junctions)) == 1,
        len(set(counts)) == 1,
        junction_consistency,
        opening_consistency,
        total,
    )


@dataclass(frozen=True)
class AdaptiveSnapshot:
    physics_frame: int
    timestamp: float
    robot_positions: np.ndarray
    leader_position: np.ndarray
    leader_velocity: np.ndarray
    lidar_yaw_deg: float
    angles_deg: np.ndarray
    true_ranges: np.ndarray
    noise_values: np.ndarray
    raw_ranges: np.ndarray
    smoothed_ranges: np.ndarray
    open_support_mask: np.ndarray
    open_threshold: float
    wall_reference: float
    range_ceiling: float
    dynamic_span: float
    gradient_threshold: float
    opening_groups: tuple[dict[str, float], ...]
    opening_present: bool
    current_junction_evidence: bool
    junction_confirmed: bool
    confirmation_frame: int | None
    confirmation_time: float | None
    system_state: str
    state_transition: str
    junction_detected: bool
    frozen55_openings: tuple[dict[str, float], ...]
    frozen55_threshold: float
    frozen55_junction_detected: bool
    w_tau_openings: tuple[dict[str, float], ...]
    w_tau_junction_detected: bool
    w_tau_lower_bound: float
    w_tau_upper_bound: float
    threshold_interval_valid: bool
    w_tau_selected_threshold: float | None
    interval_threshold_count: int
    interval_junction_stable: bool
    interval_opening_count_stable: bool
    interval_junction_consistency: float
    interval_opening_count_consistency: float
    anchor_fixed: bool
    anchor_fix_frame: int | None
    anchor_fix_time: float | None
    local_w_estimate: LocalWEstimate
    local_w_gt_eval_only: LocalWEstimate
    w_gt_eval_only: float | None
    w_error_max: float | None
    w_error_median: float | None
    w_error_p90: float | None
    w_history_median: float | None
    w_history_p90: float | None
    corridor_calibration: dict[str, Any]


class AdaptiveSession:
    """Run EXP-043 physics with optional LiDAR-only fixed-anchor feedback."""

    def __init__(
        self,
        map_case: str,
        config: DetectorExperimentConfig,
    ) -> None:
        self.map_case = map_case
        self.config = config
        self.runner = _new_runner(map_case)
        if (
            self.config.pre_corridor_calibration_start
            and map_case not in (M0_CASE, M1_PRE_CORRIDOR_CASE)
        ):
            _rear_start(self.runner, REAR_START_SHIFT)
        self._apply_pose_condition_eval_setup()
        self.rng = np.random.default_rng(config.noise_seed)
        self.next_physics_frame = 0
        self.snapshots: list[AdaptiveSnapshot] = []
        self.view_index = -1
        self.timeline: list[dict[str, Any]] = []
        self.comparison: list[dict[str, Any]] = []
        self.beam_rows: list[dict[str, Any]] = []
        self.threshold_sweep_frames: list[dict[str, Any]] = []
        self.local_w_candidate_rows: list[dict[str, Any]] = []
        self.local_w_timeline: list[dict[str, Any]] = []
        self.corridor_w_timeline: list[dict[str, Any]] = []
        self.corridor_w_calibration_samples: list[dict[str, Any]] = []
        self.corridor_w_definition_rows: list[dict[str, Any]] = []
        self.corridor_w_histories: dict[str, list[float]] = {
            name: [] for name in W_STATISTICS
        }
        self.calibration_state = "W_CALIBRATION_SEARCH"
        self.eligible_scan_count = 0
        self.first_w_ready_shadow_frame: int | None = None
        self.first_w_frozen_shadow_frame: int | None = None
        self.w_frozen_shadow = False
        self.frozen_w_history_median: dict[str, float | None] = {
            name: None for name in W_STATISTICS
        }
        self.w_est_max_history: list[float] = []
        self.sweep_first_detection: dict[float, tuple[int, float]] = {}
        self.sweep_max_openings: dict[float, int] = {
            alpha: 0 for alpha in DEFAULT_SWEEP_ALPHAS
        }
        self.physics_trajectory: list[tuple[float, float]] = []
        self.first_open_support_frame: int | None = None
        self.first_open_support_time: float | None = None
        self.first_opening_frame: int | None = None
        self.first_opening_time: float | None = None
        self.first_detection_frame: int | None = None
        self.first_detection_time: float | None = None
        self.first_evidence_frame: int | None = None
        self.first_evidence_time: float | None = None
        self.junction_confirmed = False
        self.confirmation_frame: int | None = None
        self.confirmation_time: float | None = None
        self.frozen55_first_detection_frame: int | None = None
        self.w_tau_first_detection_frame: int | None = None
        self.anchor_fixed = False
        self.anchor_fix_frame: int | None = None
        self.anchor_fix_time: float | None = None
        self.anchor_fix_position: np.ndarray | None = None
        self.post_fix_max_position_drift = 0.0
        self.normal_positions_at_fix: dict[int, np.ndarray] = {}
        self.post_fix_max_normal_motion = 0.0

    def _apply_pose_condition_eval_setup(self) -> None:
        """Apply a declared initial-condition perturbation, not a gate input."""
        condition = self.config.pose_condition
        if condition == "center":
            return
        leader = next(
            robot for robot in self.runner.world.robots
            if robot.robot_id == self.runner.world.lidar_robot_id
        )
        if condition in ("left", "right"):
            heading = np.array([
                math.cos(leader.body_yaw_rad),
                math.sin(leader.body_yaw_rad),
            ])
            lateral = np.array([-heading[1], heading[0]])
            sign = 1.0 if condition == "left" else -1.0
            delta = sign * self.config.pose_lateral_offset * lateral
            for robot in self.runner.world.robots:
                robot.position = robot.position + delta
                robot.ingress_lane_x += float(delta[0])
            return
        sign = 1.0 if condition == "yaw-left" else -1.0
        radians = math.radians(sign * self.config.pose_yaw_offset_deg)
        rotation = np.array([
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ])
        origin = leader.position.copy()
        for robot in self.runner.world.robots:
            robot.position = origin + rotation @ (robot.position - origin)
            robot.velocity = rotation @ robot.velocity
            robot.observed_velocity = rotation @ robot.observed_velocity
            robot.body_yaw_rad += radians

    def restart(self) -> None:
        self.__init__(self.map_case, self.config)

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
        angles = np.asarray(scan.angles_deg, dtype=float).copy()
        true_ranges = np.asarray(scan.ranges, dtype=float).copy()
        noise_values, measured_ranges = _inject_measurement_noise(
            true_ranges,
            self.config,
            self.rng,
        )
        estimator_ranges = (
            smooth_ranges(measured_ranges, self.config.smoothing_window)
            if self.config.w_estimator_use_smoothed
            else measured_ranges
        )
        local_w_estimate = estimate_local_w_shadow(
            angles.copy(),
            np.asarray(estimator_ranges, dtype=float).copy(),
            max_range=LIDAR_MAX_RANGE,
            tau=self.config.tau,
            boundary_window_rays=self.config.w_boundary_window_rays,
            min_far_width_deg=FROZEN_PARAMETERS["min_opening_width_deg"],
        )
        local_w_gt = evaluate_w_gt_from_true_scan_eval_only(
            angles.copy(),
            true_ranges.copy(),
            max_range=LIDAR_MAX_RANGE,
            min_far_width_deg=FROZEN_PARAMETERS["min_opening_width_deg"],
        )
        w_gt_eval_only = local_w_gt.w_est_max

        def estimation_error(value: float | None) -> float | None:
            if value is None or w_gt_eval_only is None:
                return None
            return abs(float(value) - float(w_gt_eval_only))

        w_error_max = estimation_error(local_w_estimate.w_est_max)
        w_error_median = estimation_error(local_w_estimate.w_est_median)
        w_error_p90 = estimation_error(local_w_estimate.w_est_p90)
        if local_w_estimate.w_est_max is not None:
            self.w_est_max_history.append(local_w_estimate.w_est_max)
            self.w_est_max_history = self.w_est_max_history[
                -self.config.w_history_scans:
            ]
        w_history_median = (
            None
            if not self.w_est_max_history
            else float(np.median(self.w_est_max_history))
        )
        w_history_p90 = (
            None
            if not self.w_est_max_history
            else float(np.quantile(self.w_est_max_history, 0.90))
        )

        calibration_ranges = (
            smooth_ranges(measured_ranges, self.config.smoothing_window)
            if self.config.w_calibration_use_smoothed
            else measured_ranges
        )
        corridor_calibration = calibrate_corridor_w_scan_shadow(
            angles.copy(),
            np.asarray(calibration_ranges, dtype=float).copy(),
            max_range=LIDAR_MAX_RANGE,
            tau=self.config.tau,
            dominant_min_width_deg=(
                self.config.corridor_dominant_min_width_deg
            ),
            opposition_tolerance_deg=(
                self.config.corridor_opposition_tolerance_deg
            ),
            max_far_fraction=self.config.corridor_max_far_fraction,
            boundary_exclusion_rays=self.config.boundary_exclusion_rays,
            wall_support_definition=self.config.wall_support_definition,
        )
        if corridor_calibration["corridor_like"] and not self.w_frozen_shadow:
            self.eligible_scan_count += 1
            for statistic in W_STATISTICS:
                value = corridor_calibration["wall_statistics"][statistic]
                if value is not None:
                    history = self.corridor_w_histories[statistic]
                    history.append(float(value))
                    del history[:-self.config.w_calibration_history_scans]
            if (
                min(len(values) for values in self.corridor_w_histories.values())
                >= self.config.w_calibration_history_scans
            ):
                self.calibration_state = "W_READY_SHADOW"
                if self.first_w_ready_shadow_frame is None:
                    self.first_w_ready_shadow_frame = frame
            else:
                self.calibration_state = "W_CALIBRATING"
        elif self.calibration_state == "W_READY_SHADOW":
            # Shadow-only local structure freeze: the active detector is not
            # consulted and no value from this branch feeds back into motion.
            self.w_frozen_shadow = True
            self.first_w_frozen_shadow_frame = frame
            self.calibration_state = "W_FROZEN_SHADOW"
            self.frozen_w_history_median = {
                name: _history_summary(values)["median"]
                for name, values in self.corridor_w_histories.items()
            }
        elif self.w_frozen_shadow:
            self.calibration_state = "W_FROZEN_SHADOW"
        else:
            self.calibration_state = "W_CALIBRATION_SEARCH"
        history_summaries = {
            name: _history_summary(values)
            for name, values in self.corridor_w_histories.items()
        }
        corridor_calibration["calibration_state"] = self.calibration_state
        corridor_calibration["eligible_scan_count"] = self.eligible_scan_count
        corridor_calibration["history_count"] = min(
            len(values) for values in self.corridor_w_histories.values()
        )
        corridor_calibration["history_summaries"] = history_summaries
        corridor_calibration["w_ready_shadow"] = (
            self.first_w_ready_shadow_frame is not None
        )
        corridor_calibration["w_frozen_shadow"] = self.w_frozen_shadow

        # Evaluation oracle: a true range strictly below Rmax hit geometry.
        # It is intentionally computed after the runtime decision and is only
        # copied to *_eval_only report fields.
        true_wall_mask_eval_only = true_ranges < (LIDAR_MAX_RANGE - DETECTOR_EPSILON)
        true_wall_stats_eval_only = _robust_wall_statistics(
            true_ranges[true_wall_mask_eval_only]
        )
        near_counts_eval_only = {
            threshold: int(np.count_nonzero(
                true_wall_mask_eval_only & (true_ranges >= threshold)
            ))
            for threshold in (135.0, 140.0, 142.5, 145.0)
        }
        true_wall_count_eval_only = int(np.count_nonzero(true_wall_mask_eval_only))
        near_rmax_wall_fraction_eval_only = (
            near_counts_eval_only[135.0] / true_wall_count_eval_only
            if true_wall_count_eval_only else 0.0
        )
        for candidate in local_w_estimate.candidates:
            self.local_w_candidate_rows.append(
                {
                    "map_case": self.map_case,
                    "frame": frame,
                    "timestamp": float(row["timestamp"]),
                    "boundary_id": candidate.boundary_id,
                    "boundary_side": candidate.boundary_side,
                    "boundary_angle_deg": candidate.boundary_angle_deg,
                    "far_run_start_angle": candidate.far_run_start_angle,
                    "far_run_end_angle": candidate.far_run_end_angle,
                    "immediate_wall_range": candidate.immediate_wall_range,
                    "local_wall_max": candidate.local_wall_max,
                    "local_wall_median": candidate.local_wall_median,
                    "local_wall_p90": candidate.local_wall_p90,
                    "noise_model": self.config.noise_model,
                    "tau": self.config.tau,
                    "noise_seed": self.config.noise_seed,
                    "estimator_input": (
                        "smoothed_measured_range"
                        if self.config.w_estimator_use_smoothed
                        else "raw_measured_range"
                    ),
                    "w_gt_eval_only": _empty(w_gt_eval_only),
                    "evaluation_gt_used": True,
                    "runtime_gt_map_used": False,
                }
            )
        frozen_openings, frozen_diag = _detect_openings_with_diagnostics(
            angles.copy(),
            measured_ranges.copy(),
            smoothing_window_size=self.config.smoothing_window,
        )
        lower, upper, interval_valid = self.config.safe_interval
        selected_w_tau = self.config.selected_w_tau_threshold
        w_tau_openings, w_tau_diag = (
            _detect_openings_w_tau_with_diagnostics(
                angles.copy(),
                measured_ranges.copy(),
                selected_threshold=selected_w_tau,
                threshold_interval_valid=interval_valid,
                smoothing_window_size=self.config.smoothing_window,
            )
        )

        frozen_junction = len(frozen_openings) >= 3
        w_tau_junction = len(w_tau_openings) >= 3
        if self.config.threshold_mode == "w-tau":
            active_openings = w_tau_openings
            active_diag = w_tau_diag
            active_junction = w_tau_junction
        else:
            active_openings = frozen_openings
            active_diag = frozen_diag
            active_junction = frozen_junction

        # Detector observation and system-confirmed state are deliberately
        # separate.  Evidence can toggle on every sampled scan; confirmation
        # is a one-way latch.  An invalid W–tau interval can never confirm.
        current_junction_evidence = bool(
            interval_valid and active_junction
        )

        interval_rows = _evaluate_safe_interval(
            angles,
            measured_ranges,
            self.config,
        )
        if (
            interval_valid
            and not self.config.evaluate_interval
            and selected_w_tau is not None
        ):
            interval_rows = [
                {
                    "alpha": self.config.threshold_alpha,
                    "threshold": selected_w_tau,
                    "opening_count": len(w_tau_openings),
                    "junction_detected": w_tau_junction,
                }
            ]
        (
            interval_junction_stable,
            interval_opening_stable,
            interval_junction_consistency,
            interval_opening_consistency,
            interval_threshold_count,
        ) = _interval_stability(interval_rows)
        for interval_row in interval_rows:
            alpha = float(interval_row["alpha"])
            opening_count = int(interval_row["opening_count"])
            self.sweep_max_openings[alpha] = max(
                self.sweep_max_openings.get(alpha, 0),
                opening_count,
            )
            if (
                bool(interval_row["junction_detected"])
                and alpha not in self.sweep_first_detection
            ):
                self.sweep_first_detection[alpha] = (
                    frame,
                    float(row["timestamp"]),
                )
            self.threshold_sweep_frames.append(
                {
                    "map_case": self.map_case,
                    "noise_model": self.config.noise_model,
                    "frame": frame,
                    "timestamp": float(row["timestamp"]),
                    **interval_row,
                    "anchor_mode": (
                        "CLOSED_LOOP_ANCHOR_STOP"
                        if self.config.anchor_stop_on_detect
                        else "OPEN_LOOP_DIAGNOSTIC"
                    ),
                }
            )

        if current_junction_evidence and not self.junction_confirmed:
            self.first_evidence_frame = frame
            self.first_evidence_time = float(row["timestamp"])
            # Backward-compatible columns retain the precise meaning "first
            # current Junction evidence", not the later diagnostic values.
            self.first_detection_frame = frame
            self.first_detection_time = float(row["timestamp"])
            self.junction_confirmed = True
            self.confirmation_frame = frame
            self.confirmation_time = float(row["timestamp"])
            if self.config.anchor_stop_on_detect:
                self.anchor_fixed = True
                self.anchor_fix_frame = frame
                self.anchor_fix_time = float(row["timestamp"])
                self.anchor_fix_position = leader.position.copy()
                self.normal_positions_at_fix = {
                    robot.robot_id: robot.position.copy()
                    for robot in self.runner.world.robots
                    if robot.robot_id != self.runner.world.lidar_robot_id
                }
                self.runner.world.fix_robot(
                    self.runner.world.lidar_robot_id
                )
                state_transition = TRANSITION_CONFIRM_AND_FIX
            else:
                state_transition = TRANSITION_CONFIRM_OPEN_LOOP
        elif self.anchor_fixed:
            state_transition = SYSTEM_STATE_FIXED_ANCHOR
        elif self.junction_confirmed:
            state_transition = SYSTEM_STATE_JUNCTION_CONFIRMED
        else:
            state_transition = SYSTEM_STATE_MOVING

        system_state = (
            SYSTEM_STATE_FIXED_ANCHOR
            if self.anchor_fixed
            else (
                SYSTEM_STATE_JUNCTION_CONFIRMED
                if self.junction_confirmed
                else SYSTEM_STATE_MOVING
            )
        )

        if self.anchor_fixed and self.anchor_fix_position is not None:
            self.post_fix_max_position_drift = max(
                self.post_fix_max_position_drift,
                float(np.linalg.norm(leader.position - self.anchor_fix_position)),
            )
            self.post_fix_max_normal_motion = max(
                self.post_fix_max_normal_motion,
                max(
                    (
                        float(np.linalg.norm(
                            robot.position
                            - self.normal_positions_at_fix[robot.robot_id]
                        ))
                        for robot in self.runner.world.robots
                        if robot.robot_id != self.runner.world.lidar_robot_id
                    ),
                    default=0.0,
                ),
            )

        smoothed = np.asarray(active_diag["smoothed_ranges"], dtype=float)
        support = np.asarray(active_diag["open_support_mask"], dtype=bool)
        wall_reference = float(active_diag["wall_reference"])
        range_ceiling = float(active_diag["range_ceiling"])
        dynamic_span = max(0.0, range_ceiling - wall_reference)
        active_threshold = float(active_diag["open_threshold"])
        snapshot = AdaptiveSnapshot(
            physics_frame=frame,
            timestamp=float(row["timestamp"]),
            robot_positions=np.array(
                [robot.position.copy() for robot in self.runner.world.robots]
            ),
            leader_position=leader.position.copy(),
            leader_velocity=leader.observed_velocity.copy(),
            lidar_yaw_deg=float(self.runner.world.lidar_yaw_deg),
            angles_deg=angles,
            true_ranges=true_ranges,
            noise_values=noise_values.copy(),
            raw_ranges=measured_ranges.copy(),
            smoothed_ranges=smoothed.copy(),
            open_support_mask=support.copy(),
            open_threshold=active_threshold,
            wall_reference=wall_reference,
            range_ceiling=range_ceiling,
            dynamic_span=dynamic_span,
            gradient_threshold=float(active_diag["gradient_threshold"]),
            opening_groups=tuple(dict(opening) for opening in active_openings),
            opening_present=len(active_openings) > 0,
            current_junction_evidence=current_junction_evidence,
            junction_confirmed=self.junction_confirmed,
            confirmation_frame=self.confirmation_frame,
            confirmation_time=self.confirmation_time,
            system_state=system_state,
            state_transition=state_transition,
            junction_detected=current_junction_evidence,
            frozen55_openings=tuple(
                dict(opening) for opening in frozen_openings
            ),
            frozen55_threshold=float(frozen_diag["open_threshold"]),
            frozen55_junction_detected=frozen_junction,
            w_tau_openings=tuple(
                dict(opening) for opening in w_tau_openings
            ),
            w_tau_junction_detected=w_tau_junction,
            w_tau_lower_bound=lower,
            w_tau_upper_bound=upper,
            threshold_interval_valid=interval_valid,
            w_tau_selected_threshold=selected_w_tau,
            interval_threshold_count=interval_threshold_count,
            interval_junction_stable=interval_junction_stable,
            interval_opening_count_stable=interval_opening_stable,
            interval_junction_consistency=interval_junction_consistency,
            interval_opening_count_consistency=interval_opening_consistency,
            anchor_fixed=self.anchor_fixed,
            anchor_fix_frame=self.anchor_fix_frame,
            anchor_fix_time=self.anchor_fix_time,
            local_w_estimate=local_w_estimate,
            local_w_gt_eval_only=local_w_gt,
            w_gt_eval_only=w_gt_eval_only,
            w_error_max=w_error_max,
            w_error_median=w_error_median,
            w_error_p90=w_error_p90,
            w_history_median=w_history_median,
            w_history_p90=w_history_p90,
            corridor_calibration=corridor_calibration,
        )
        self.snapshots.append(snapshot)
        self.view_index = len(self.snapshots) - 1

        support_count = int(np.count_nonzero(support))
        if support_count > 0 and self.first_open_support_frame is None:
            self.first_open_support_frame = frame
            self.first_open_support_time = snapshot.timestamp
        if active_openings and self.first_opening_frame is None:
            self.first_opening_frame = frame
            self.first_opening_time = snapshot.timestamp
        if frozen_junction and self.frozen55_first_detection_frame is None:
            self.frozen55_first_detection_frame = frame
        if w_tau_junction and self.w_tau_first_detection_frame is None:
            self.w_tau_first_detection_frame = frame

        selected_value = (
            math.nan if selected_w_tau is None else selected_w_tau
        )
        self.local_w_timeline.append(
            {
                "map_case": self.map_case,
                "frame": frame,
                "timestamp": snapshot.timestamp,
                "noise_model": self.config.noise_model,
                "noise_sigma": _empty(self.config.noise_sigma),
                "noise_seed": self.config.noise_seed,
                "tau": self.config.tau,
                "w_est_available": local_w_estimate.available,
                "far_run_count": len(local_w_estimate.far_runs),
                "w_candidate_count": len(local_w_estimate.candidates),
                "far_support_threshold": (
                    local_w_estimate.far_support_threshold
                ),
                "w_est_max": _empty(local_w_estimate.w_est_max),
                "w_est_median": _empty(local_w_estimate.w_est_median),
                "w_est_p90": _empty(local_w_estimate.w_est_p90),
                "w_history_median": _empty(w_history_median),
                "w_history_p90": _empty(w_history_p90),
                "w_gt_eval_only": _empty(w_gt_eval_only),
                "w_error_max": _empty(w_error_max),
                "w_error_median": _empty(w_error_median),
                "w_error_p90": _empty(w_error_p90),
                "active_w_source": "MANUAL",
                "active_manual_w": self.config.worst_wall_range,
                "shadow_w_feedback_used": False,
                "estimator_input": (
                    "smoothed_measured_range"
                    if self.config.w_estimator_use_smoothed
                    else "raw_measured_range"
                ),
                "current_junction_evidence": (
                    current_junction_evidence
                ),
                "junction_confirmed": self.junction_confirmed,
                "anchor_fixed": self.anchor_fixed,
                "runtime_gt_map_used": False,
                "evaluation_gt_used": True,
            }
        )
        safe_w_upper_limit = LIDAR_MAX_RANGE - 2.0 * self.config.tau
        calibration_row: dict[str, Any] = {
            "map_case": self.map_case,
            "frame": frame,
            "timestamp": snapshot.timestamp,
            "noise_model": self.config.noise_model,
            "noise_seed": self.config.noise_seed,
            "corridor_like": corridor_calibration["corridor_like"],
            "calibration_state": self.calibration_state,
            "eligible_scan_count": self.eligible_scan_count,
            "history_count": min(
                len(values) for values in self.corridor_w_histories.values()
            ),
            "far_run_count": len(corridor_calibration["far_runs"]),
            "dominant_far_run_count": len(
                corridor_calibration["dominant_far_runs"]
            ),
            "far_angular_widths_deg": "|".join(
                f"{value:.6f}"
                for value in corridor_calibration["dominant_widths_deg"]
            ),
            "two_run_separation_deg": _empty(
                corridor_calibration["two_run_separation_deg"]
            ),
            "far_fraction": corridor_calibration["far_fraction"],
            "far_support_threshold": corridor_calibration[
                "far_support_threshold"
            ],
            "wall_support_definition": self.config.wall_support_definition,
            "wall_support_count": int(np.count_nonzero(
                corridor_calibration["wall_support_mask"]
            )),
            "excluded_boundary_ray_count": int(np.count_nonzero(
                corridor_calibration["excluded_boundary_mask"]
            )),
            "safe_w_upper_limit": safe_w_upper_limit,
            "near_rmax_wall_ge_135_count_eval_only": near_counts_eval_only[135.0],
            "near_rmax_wall_ge_140_count_eval_only": near_counts_eval_only[140.0],
            "near_rmax_wall_ge_142_5_count_eval_only": near_counts_eval_only[142.5],
            "near_rmax_wall_ge_145_count_eval_only": near_counts_eval_only[145.0],
            "near_rmax_wall_fraction_eval_only": near_rmax_wall_fraction_eval_only,
            "w_ready_shadow": self.first_w_ready_shadow_frame is not None,
            "w_frozen_shadow": self.w_frozen_shadow,
            "first_w_frozen_shadow_frame": _empty(
                self.first_w_frozen_shadow_frame
            ),
            "active_w_source": "MANUAL",
            "active_manual_w": self.config.worst_wall_range,
            "shadow_w_feedback_used": False,
            "current_junction_evidence": current_junction_evidence,
            "junction_confirmed": self.junction_confirmed,
            "anchor_fixed": self.anchor_fixed,
            "runtime_gt_map_used": False,
            "evaluation_gt_used": True,
            "pre_corridor_start_eval_only": (
                self.config.pre_corridor_calibration_start
            ),
            "calibration_input": (
                "smoothed_measured_range"
                if self.config.w_calibration_use_smoothed
                else "raw_measured_range"
            ),
            "raw_vs_smoothed_mean_abs_difference": float(np.mean(np.abs(
                measured_ranges
                - smooth_ranges(measured_ranges, self.config.smoothing_window)
            ))),
        }
        for statistic in W_STATISTICS:
            current = corridor_calibration["wall_statistics"][statistic]
            history = history_summaries[statistic]
            calibration_row[f"w_{statistic}_current"] = _empty(current)
            calibration_row[f"w_{statistic}_history_median"] = _empty(
                history["median"]
            )
            calibration_row[f"w_{statistic}_history_p90"] = _empty(
                history["p90"]
            )
            calibration_row[f"w_{statistic}_history_mad"] = _empty(
                history["mad"]
            )
            calibration_row[f"w_{statistic}_history_span"] = _empty(
                history["span"]
            )
            calibration_row[f"w_safe_interval_valid_{statistic}"] = bool(
                current is not None and current < safe_w_upper_limit
            )
            calibration_row[f"true_wall_{statistic}_eval_only"] = _empty(
                true_wall_stats_eval_only[statistic]
            )
        self.corridor_w_timeline.append(calibration_row)

        for definition, statistics in corridor_calibration[
            "all_definition_statistics"
        ].items():
            definition_mask = corridor_calibration["support_masks"][definition]
            self.corridor_w_definition_rows.append({
                "map_case": self.map_case,
                "frame": frame,
                "timestamp": snapshot.timestamp,
                "noise_model": self.config.noise_model,
                "wall_support_definition": definition,
                "corridor_like": corridor_calibration["corridor_like"],
                "wall_support_count": int(np.count_nonzero(definition_mask)),
                **{
                    f"w_{name}_current": _empty(statistics[name])
                    for name in W_STATISTICS
                },
                **{
                    f"w_safe_interval_valid_{name}": bool(
                        statistics[name] is not None
                        and statistics[name] < safe_w_upper_limit
                    )
                    for name in W_STATISTICS
                },
                "runtime_gt_map_used": False,
                "evaluation_gt_used": False,
            })

        selected_support = corridor_calibration["wall_support_mask"]
        for index in np.flatnonzero(
            selected_support | corridor_calibration["excluded_boundary_mask"]
        ):
            self.corridor_w_calibration_samples.append({
                "map_case": self.map_case,
                "frame": frame,
                "timestamp": snapshot.timestamp,
                "noise_model": self.config.noise_model,
                "noise_seed": self.config.noise_seed,
                "theta_deg": float(angles[index]),
                "measured_range": float(calibration_ranges[index]),
                "true_range_eval_only": float(true_ranges[index]),
                "far_support": bool(corridor_calibration["far_mask"][index]),
                "wall_support": bool(selected_support[index]),
                "boundary_excluded": bool(
                    corridor_calibration["excluded_boundary_mask"][index]
                ),
                "true_wall_hit_eval_only": bool(
                    true_wall_mask_eval_only[index]
                ),
                "near_rmax_wall_eval_only": bool(
                    true_wall_mask_eval_only[index]
                    and true_ranges[index] >= 135.0
                ),
                "runtime_gt_map_used": False,
                "evaluation_gt_used": True,
            })
        self.timeline.append(
            {
                "map_case": self.map_case,
                "frame": frame,
                "timestamp": snapshot.timestamp,
                "threshold_mode": self.config.threshold_mode,
                "noise_model": self.config.noise_model,
                "noise_fraction": self.config.noise_fraction,
                "noise_sigma": (
                    math.nan
                    if self.config.noise_sigma is None
                    else self.config.noise_sigma
                ),
                "noise_seed": self.config.noise_seed,
                "tau": self.config.tau,
                "worst_wall_range": self.config.worst_wall_range,
                "w_tau_lower_bound": lower,
                "w_tau_upper_bound": upper,
                "threshold_interval_valid": interval_valid,
                "threshold_alpha": self.config.threshold_alpha,
                "selected_threshold": active_threshold,
                "frozen55_threshold": float(frozen_diag["open_threshold"]),
                "frozen55_opening_count": len(frozen_openings),
                "frozen55_junction_detected": frozen_junction,
                "w_tau_opening_count": len(w_tau_openings),
                "w_tau_junction_detected": w_tau_junction,
                "active_opening_count": len(active_openings),
                "active_opening_present": len(active_openings) > 0,
                "active_junction_detected": active_junction,
                "current_junction_evidence": current_junction_evidence,
                "junction_confirmed": self.junction_confirmed,
                "confirmation_frame": _empty(self.confirmation_frame),
                "confirmation_time": _empty(self.confirmation_time),
                "system_state": system_state,
                "state_transition": state_transition,
                "active_open_support_count": support_count,
                "smoothing_window": self.config.smoothing_window,
                "interval_threshold_count": interval_threshold_count,
                "interval_junction_stable": interval_junction_stable,
                "interval_opening_count_stable": interval_opening_stable,
                "interval_junction_consistency": (
                    interval_junction_consistency
                ),
                "interval_opening_count_consistency": (
                    interval_opening_consistency
                ),
                "anchor_stop_on_detect": (
                    self.config.anchor_stop_on_detect
                ),
                "anchor_fixed": self.anchor_fixed,
                "anchor_fix_frame": _empty(self.anchor_fix_frame),
                "anchor_fix_time": _empty(self.anchor_fix_time),
                "post_fix_position_drift": (
                    0.0
                    if self.anchor_fix_position is None
                    else float(np.linalg.norm(
                        leader.position - self.anchor_fix_position
                    ))
                ),
                "post_fix_max_position_drift": (
                    self.post_fix_max_position_drift
                ),
                "post_fix_max_normal_motion": (
                    self.post_fix_max_normal_motion
                ),
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
                "noise_model": self.config.noise_model,
                "frozen55_threshold": float(frozen_diag["open_threshold"]),
                "w_tau_lower_bound": lower,
                "w_tau_upper_bound": upper,
                "w_tau_selected_threshold": selected_value,
                "threshold_interval_valid": interval_valid,
                "frozen55_opening_count": len(frozen_openings),
                "w_tau_opening_count": len(w_tau_openings),
                "frozen55_junction_detected": frozen_junction,
                "w_tau_junction_detected": w_tau_junction,
                "active_threshold_mode": self.config.threshold_mode,
                "active_opening_count": len(active_openings),
                "active_junction_detected": active_junction,
                "current_junction_evidence": current_junction_evidence,
                "junction_confirmed": self.junction_confirmed,
                "confirmation_frame": _empty(self.confirmation_frame),
                "anchor_fixed": self.anchor_fixed,
            }
        )
        if self.config.dump_beams:
            for index, theta in enumerate(angles):
                self.beam_rows.append(
                    {
                        "map_case": self.map_case,
                        "frame": frame,
                        "timestamp": snapshot.timestamp,
                        "theta_deg": float(theta),
                        "true_range": float(true_ranges[index]),
                        "noise_value": float(noise_values[index]),
                        "measured_range": float(measured_ranges[index]),
                        "smoothed_range": float(smoothed[index]),
                        "active_open_support": bool(support[index]),
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

    def threshold_sweep_summary_rows(self) -> list[dict[str, Any]]:
        """Summarize the per-scan sweep (fair timing in open-loop mode)."""
        lower, upper, valid = self.config.safe_interval
        if not valid:
            return []
        rows = []
        alphas = (
            DEFAULT_SWEEP_ALPHAS
            if self.config.evaluate_interval
            else (self.config.threshold_alpha,)
        )
        for alpha in alphas:
            first = self.sweep_first_detection.get(alpha)
            rows.append(
                {
                    "case": self.map_case,
                    "noise": self.config.noise_model,
                    "detector": "W_TAU_SAFE_INTERVAL",
                    "alpha": alpha,
                    "threshold": select_threshold_in_safe_interval(
                        lower, upper, True, alpha
                    ),
                    "first_detection_frame": (
                        "" if first is None else first[0]
                    ),
                    "first_detection_time": (
                        "" if first is None else first[1]
                    ),
                    "max_opening_count": self.sweep_max_openings.get(alpha, 0),
                    "junction_detected": first is not None,
                    "M0_false_positive": bool(
                        self.map_case == M0_CASE and first is not None
                    ),
                    "anchor_mode": (
                        "CLOSED_LOOP_SHARED_TRAJECTORY_DIAGNOSTIC"
                        if self.config.anchor_stop_on_detect
                        else "OPEN_LOOP_DIAGNOSTIC"
                    ),
                }
            )
        return rows

    def interval_outcome_stability(self) -> tuple[bool, bool]:
        rows = self.threshold_sweep_summary_rows()
        if not rows:
            return False, False
        return (
            len({bool(row["junction_detected"]) for row in rows}) == 1,
            len({int(row["max_opening_count"]) for row in rows}) == 1,
        )

    def summary(
        self,
        *,
        deterministic_replay: bool | None = None,
        movement_equivalent: bool | None = None,
        lidar_equivalent: bool | None = None,
    ) -> dict[str, Any]:
        lower, upper, valid = self.config.safe_interval
        selected = self.config.selected_w_tau_threshold
        available_count = sum(
            bool(row["w_est_available"])
            for row in self.local_w_timeline
        )
        outcome_junction_stable, outcome_opening_stable = (
            self.interval_outcome_stability()
        )
        return {
            "map_case": self.map_case,
            "noise_model": self.config.noise_model,
            "threshold_mode": self.config.threshold_mode,
            "physics_frames": self.next_physics_frame,
            "sample_count": len(self.snapshots),
            "worst_wall_range": self.config.worst_wall_range,
            "tau": self.config.tau,
            "w_tau_lower_bound": lower,
            "w_tau_upper_bound": upper,
            "threshold_interval_valid": valid,
            "threshold_alpha": self.config.threshold_alpha,
            "w_tau_selected_threshold": _empty(selected),
            "active_w_source": "MANUAL",
            "active_manual_w": self.config.worst_wall_range,
            "shadow_w_feedback_used": False,
            "w_estimator_input": (
                "smoothed_measured_range"
                if self.config.w_estimator_use_smoothed
                else "raw_measured_range"
            ),
            "w_estimator_availability_rate": (
                available_count / len(self.local_w_timeline)
                if self.local_w_timeline else 0.0
            ),
            "corridor_calibration_input": (
                "smoothed_measured_range"
                if self.config.w_calibration_use_smoothed
                else "raw_measured_range"
            ),
            "corridor_calibration_eligible_scan_count": (
                self.eligible_scan_count
            ),
            "first_w_ready_shadow_frame": _empty(
                self.first_w_ready_shadow_frame
            ),
            "first_w_frozen_shadow_frame": _empty(
                self.first_w_frozen_shadow_frame
            ),
            "w_frozen_shadow": self.w_frozen_shadow,
            "pose_condition": self.config.pose_condition,
            "pre_corridor_start_eval_only": (
                self.config.pre_corridor_calibration_start
            ),
            "representative_threshold_only": True,
            "interval_threshold_count": (
                (
                    len(DEFAULT_SWEEP_ALPHAS)
                    if self.config.evaluate_interval
                    else 1
                )
                if valid else 0
            ),
            "interval_junction_outcome_stable": outcome_junction_stable,
            "interval_opening_count_outcome_stable": outcome_opening_stable,
            "interval_junction_stable_frame_fraction": (
                float(np.mean([
                    snapshot.interval_junction_stable
                    for snapshot in self.snapshots
                ]))
                if self.snapshots else 0.0
            ),
            "interval_opening_stable_frame_fraction": (
                float(np.mean([
                    snapshot.interval_opening_count_stable
                    for snapshot in self.snapshots
                ]))
                if self.snapshots else 0.0
            ),
            "first_open_support_frame": _empty(
                self.first_open_support_frame
            ),
            "first_opening_frame": _empty(self.first_opening_frame),
            "first_detection_frame": _empty(self.first_detection_frame),
            "first_detection_time": _empty(self.first_detection_time),
            "first_evidence_frame": _empty(self.first_evidence_frame),
            "first_evidence_time": _empty(self.first_evidence_time),
            "junction_confirmed": self.junction_confirmed,
            "confirmation_frame": _empty(self.confirmation_frame),
            "confirmation_time": _empty(self.confirmation_time),
            "system_state": (
                SYSTEM_STATE_FIXED_ANCHOR
                if self.anchor_fixed
                else (
                    SYSTEM_STATE_JUNCTION_CONFIRMED
                    if self.junction_confirmed
                    else SYSTEM_STATE_MOVING
                )
            ),
            "frozen55_first_detection_frame": _empty(
                self.frozen55_first_detection_frame
            ),
            "w_tau_first_detection_frame": _empty(
                self.w_tau_first_detection_frame
            ),
            "max_opening_count": max(
                (len(snapshot.opening_groups) for snapshot in self.snapshots),
                default=0,
            ),
            "max_frozen55_openings": max(
                (len(snapshot.frozen55_openings) for snapshot in self.snapshots),
                default=0,
            ),
            "max_w_tau_openings": max(
                (len(snapshot.w_tau_openings) for snapshot in self.snapshots),
                default=0,
            ),
            "false_positive": bool(
                self.map_case == M0_CASE
                and self.first_detection_frame is not None
            ),
            "anchor_stop_on_detect": self.config.anchor_stop_on_detect,
            "anchor_fixed": self.anchor_fixed,
            "anchor_fix_frame": _empty(self.anchor_fix_frame),
            "anchor_fix_time": _empty(self.anchor_fix_time),
            "anchor_fix_x_eval_only": (
                "" if self.anchor_fix_position is None
                else float(self.anchor_fix_position[0])
            ),
            "anchor_fix_y_eval_only": (
                "" if self.anchor_fix_position is None
                else float(self.anchor_fix_position[1])
            ),
            "post_fix_max_position_drift": (
                self.post_fix_max_position_drift
            ),
            "post_fix_max_normal_motion": self.post_fix_max_normal_motion,
            "deterministic_replay": _empty(deterministic_replay),
            "movement_trajectory_equivalent": _empty(movement_equivalent),
            "lidar_scan_equivalent": _empty(lidar_equivalent),
            "movement_altered": bool(
                self.config.anchor_stop_on_detect and self.anchor_fixed
            ),
            "adaptive_output_fed_back": bool(
                self.config.anchor_stop_on_detect and self.anchor_fixed
            ),
            "detector_input_fields": (
                "angles_deg,measured_ranges,Rmax,calibrated_W,"
                "tau,smoothing_window"
            ),
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
        if snapshot.anchor_fixed:
            pygame.draw.circle(
                self.screen,
                COLORS["detected"],
                self.world_to_screen(origin, snapshot),
                13,
                3,
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

    def _draw_profile(
        self,
        snapshot: AdaptiveSnapshot,
        config: DetectorExperimentConfig,
    ) -> None:
        pygame = self.pygame
        x, y, width, height = PROFILE_RECT
        pygame.draw.rect(self.screen, (25, 31, 40), PROFILE_RECT, border_radius=5)
        for angle in (-180, -90, 0, 90, 180):
            px, _ = self._plot_point(PROFILE_RECT, angle, 0.0)
            pygame.draw.line(
                self.screen, (55, 64, 76), (px, y), (px, y + height), 1
            )
            self.text(
                str(angle),
                (px - 14, y + height + 7),
                COLORS["muted"],
                self.small_font,
            )
        for value in (0, 50, 100, 150):
            _, py = self._plot_point(PROFILE_RECT, -180.0, value)
            pygame.draw.line(
                self.screen, (55, 64, 76), (x, py), (x + width, py), 1
            )
            self.text(
                str(value), (x - 35, py - 8), COLORS["muted"], self.small_font
            )

        for index in np.flatnonzero(snapshot.local_w_estimate.far_mask):
            left = self._plot_point(
                PROFILE_RECT, float(snapshot.angles_deg[index]) - 0.5, 0.0
            )[0]
            right = self._plot_point(
                PROFILE_RECT, float(snapshot.angles_deg[index]) + 0.5, 0.0
            )[0]
            overlay = pygame.Surface(
                (max(1, right - left + 1), height), pygame.SRCALPHA
            )
            overlay.fill((*COLORS["group_center"], 24))
            self.screen.blit(overlay, (left, y))

        calibration = snapshot.corridor_calibration
        for index in np.flatnonzero(calibration["wall_support_mask"]):
            point = self._plot_point(
                PROFILE_RECT,
                float(snapshot.angles_deg[index]),
                float(snapshot.raw_ranges[index]),
            )
            pygame.draw.circle(self.screen, (80, 210, 130), point, 2)
        for index in np.flatnonzero(calibration["excluded_boundary_mask"]):
            point = self._plot_point(
                PROFILE_RECT,
                float(snapshot.angles_deg[index]),
                float(snapshot.raw_ranges[index]),
            )
            pygame.draw.line(
                self.screen, (255, 170, 55),
                (point[0] - 2, point[1] - 2),
                (point[0] + 2, point[1] + 2), 1,
            )

        if (
            config.threshold_mode == "w-tau"
            and snapshot.threshold_interval_valid
        ):
            upper_y = self._plot_point(
                PROFILE_RECT, 0.0, snapshot.w_tau_upper_bound
            )[1]
            lower_y = self._plot_point(
                PROFILE_RECT, 0.0, snapshot.w_tau_lower_bound
            )[1]
            band = pygame.Surface(
                (width, max(1, lower_y - upper_y)),
                pygame.SRCALPHA,
            )
            band.fill((72, 156, 255, 36))
            self.screen.blit(band, (x, upper_y))

        for index in np.flatnonzero(snapshot.open_support_mask):
            left = self._plot_point(
                PROFILE_RECT, float(snapshot.angles_deg[index]) - 0.5, 0.0
            )[0]
            right = self._plot_point(
                PROFILE_RECT, float(snapshot.angles_deg[index]) + 0.5, 0.0
            )[0]
            overlay = pygame.Surface(
                (max(1, right - left + 1), height),
                pygame.SRCALPHA,
            )
            overlay.fill((*COLORS["open_fill"], 62))
            self.screen.blit(overlay, (left, y))

        raw_points = [
            self._plot_point(PROFILE_RECT, float(angle), float(value))
            for angle, value in zip(snapshot.angles_deg, snapshot.raw_ranges)
        ]
        smooth_points = [
            self._plot_point(PROFILE_RECT, float(angle), float(value))
            for angle, value in zip(
                snapshot.angles_deg,
                snapshot.smoothed_ranges,
            )
        ]
        pygame.draw.lines(
            self.screen, COLORS["measured"], False, raw_points, 2
        )
        pygame.draw.lines(
            self.screen, COLORS["expected"], False, smooth_points, 2
        )

        if config.threshold_mode == "w-tau":
            line_specs: list[tuple[str, float, tuple[int, int, int], int]] = [
                ("W", config.worst_wall_range, COLORS["muted"], 1),
                ("W+tau", snapshot.w_tau_lower_bound, COLORS["group_edge"], 2),
                ("far/Rmax-tau", snapshot.w_tau_upper_bound, COLORS["group_center"], 2),
                ("Rmax", LIDAR_MAX_RANGE, COLORS["measured"], 1),
            ]
            if snapshot.w_tau_selected_threshold is not None:
                line_specs.insert(
                    2,
                    (
                        "selected T",
                        snapshot.w_tau_selected_threshold,
                        COLORS["threshold"],
                        3,
                    ),
                )
            if snapshot.local_w_estimate.w_est_max is not None:
                line_specs.insert(
                    1,
                    (
                        "shadow Wmax",
                        snapshot.local_w_estimate.w_est_max,
                        COLORS["clear"],
                        2,
                    ),
                )
        else:
            line_specs = [
                (
                    "frozen 55% T",
                    snapshot.frozen55_threshold,
                    COLORS["threshold"],
                    3,
                )
            ]
        label_y_by_side: dict[int, list[int]] = {0: [], 1: []}
        for index, (label, value, color, line_width) in enumerate(line_specs):
            py = self._plot_point(PROFILE_RECT, 0.0, value)[1]
            pygame.draw.line(
                self.screen,
                color,
                (x, py),
                (x + width, py),
                line_width,
            )
            side = index % 2
            label_x = x + 5 if side == 0 else x + width - 105
            label_y = max(y + 36, py - 16)
            while any(
                abs(label_y - occupied) < 15
                for occupied in label_y_by_side[side]
            ):
                label_y += 15
            label_y_by_side[side].append(label_y)
            self.text(
                f"{label}={value:.1f}",
                (label_x, label_y),
                color,
                self.small_font,
            )

        for opening in snapshot.opening_groups:
            for key, color in (
                ("start_angle", COLORS["group_edge"]),
                ("end_angle", COLORS["group_edge"]),
                ("center_angle", COLORS["group_center"]),
            ):
                px = self._plot_point(
                    PROFILE_RECT,
                    float(opening[key]),
                    0.0,
                )[0]
                pygame.draw.line(
                    self.screen, color, (px, y), (px, y + height), 2
                )
        for candidate in snapshot.local_w_estimate.candidates:
            point = self._plot_point(
                PROFILE_RECT,
                candidate.boundary_angle_deg,
                candidate.immediate_wall_range,
            )
            pygame.draw.circle(self.screen, COLORS["clear"], point, 5, 2)
        for candidate in snapshot.local_w_gt_eval_only.candidates:
            point = self._plot_point(
                PROFILE_RECT,
                candidate.boundary_angle_deg,
                candidate.immediate_wall_range,
            )
            pygame.draw.line(
                self.screen,
                COLORS["detected"],
                (point[0] - 4, point[1] - 4),
                (point[0] + 4, point[1] + 4),
                2,
            )
            pygame.draw.line(
                self.screen,
                COLORS["detected"],
                (point[0] - 4, point[1] + 4),
                (point[0] + 4, point[1] - 4),
                2,
            )
        pygame.draw.rect(
            self.screen, COLORS["muted"], PROFILE_RECT, 1, border_radius=5
        )
        legend = (
            ("RAW", COLORS["measured"]),
            ("SMOOTHED", COLORS["expected"]),
            ("OPEN SUPPORT", COLORS["open_beam"]),
            ("SAFE T INTERVAL", COLORS["expected"]),
        )
        cursor = x + 7
        for label, color in legend:
            pygame.draw.line(
                self.screen,
                color,
                (cursor, y + 13),
                (cursor + 13, y + 13),
                3,
            )
            self.text(
                label, (cursor + 17, y + 5), color, self.small_font
            )
            cursor += 28 + self.small_font.size(label)[0]
        self.text("range", (x - 35, y - 24), COLORS["muted"], self.small_font)
        self.text(
            "LiDAR angle theta [deg]",
            (x + width // 2 - 76, y + height + 28),
            COLORS["muted"],
            self.small_font,
        )

    def _draw_w_shadow_panel(
        self,
        snapshot: AdaptiveSnapshot,
        config: DetectorExperimentConfig,
    ) -> None:
        pygame = self.pygame
        panel = pygame.Surface((430, 280), pygame.SRCALPHA)
        panel.fill((10, 15, 22, 220))
        self.screen.blit(panel, (28, 68))

        def fmt(value: float | None) -> str:
            return "UNAVAILABLE" if value is None else f"{value:.3f}"

        calibration = snapshot.corridor_calibration
        current = calibration["wall_statistics"]
        safe_limit = LIDAR_MAX_RANGE - 2.0 * config.tau
        history = calibration["history_summaries"]
        lines = (
            f"ACTIVE W SOURCE = MANUAL; W={config.worst_wall_range:.3f}",
            "CORRIDOR W CALIBRATION = SHADOW (NO FEEDBACK)",
            "input = " + (
                "SMOOTHED measured"
                if config.w_calibration_use_smoothed
                else "RAW measured"
            ),
            f"corridor_like={calibration['corridor_like']}",
            f"state={calibration['calibration_state']}",
            f"eligible/history={calibration['eligible_scan_count']}/{calibration['history_count']}",
            f"far threshold={calibration['far_support_threshold']:.3f}",
            f"far runs={len(calibration['far_runs'])} dominant={len(calibration['dominant_far_runs'])}",
            f"far fraction={calibration['far_fraction']:.3f}",
            f"safe W upper limit={safe_limit:.3f}",
            f"W max={fmt(current['max'])} [{'VALID' if current['max'] is not None and current['max'] < safe_limit else 'INVALID'}]",
            f"W p90={fmt(current['p90'])} [{'VALID' if current['p90'] is not None and current['p90'] < safe_limit else 'INVALID'}]",
            f"W p95={fmt(current['p95'])} [{'VALID' if current['p95'] is not None and current['p95'] < safe_limit else 'INVALID'}]",
            f"W p99={fmt(current['p99'])} [{'VALID' if current['p99'] is not None and current['p99'] < safe_limit else 'INVALID'}]",
            "history median max/p90="
            f"{fmt(history['max']['median'])}/{fmt(history['p90']['median'])}",
            "history median p95/p99="
            f"{fmt(history['p95']['median'])}/{fmt(history['p99']['median'])}",
            f"READY={calibration['w_ready_shadow']} FROZEN={calibration['w_frozen_shadow']}",
            f"support={config.wall_support_definition}",
            "GREEN=wall support; ORANGE=boundary excluded",
        )
        for index, line in enumerate(lines):
            color = COLORS["text"]
            if "SHADOW" in line or "GREEN" in line:
                color = COLORS["clear"]
            if "INVALID" in line:
                color = COLORS["detected"]
            self.text(
                line,
                (38, 77 + 15 * index),
                color,
                self.small_font,
            )

    def draw(self, session: AdaptiveSession, paused: bool) -> None:
        snapshot = session.current
        config = session.config
        self.screen.fill(COLORS["background"])
        detector_title = (
            "W–tau noise-aware threshold"
            if config.threshold_mode == "w-tau"
            else "frozen adaptive 55% baseline"
        )
        self.text(
            f"EXP-046 Corridor W Calibration — {detector_title}",
            (18, 15),
            font=self.title_font,
        )
        if snapshot is None:
            self.text("Waiting for first sampled LiDAR scan...", (44, 90))
            self.pygame.display.flip()
            return
        self._draw_world(snapshot)
        if self.show_profile:
            self._draw_profile(snapshot, config)
        self._draw_w_shadow_panel(snapshot, config)

        panel_x = 870
        y = 530 if self.show_profile else 72
        step = 14
        selected_text = (
            "INVALID"
            if snapshot.w_tau_selected_threshold is None
            else f"{snapshot.w_tau_selected_threshold:.3f}"
        )
        lines = [
            f"{'PAUSED' if paused else 'RUNNING'} | {session.map_case}",
            f"frame={snapshot.physics_frame} t={snapshot.timestamp:.6f}s",
            f"Detector: {config.threshold_mode}",
            f"Rmax={LIDAR_MAX_RANGE:.3f}  W={config.worst_wall_range:.3f}",
            f"noise fraction={100.0 * config.noise_fraction:.2f}%  tau={config.tau:.3f}",
            f"noise={config.noise_model} seed={config.noise_seed} sigma={config.noise_sigma}",
            f"SAFE T INTERVAL: {snapshot.w_tau_lower_bound:.3f} < T < {snapshot.w_tau_upper_bound:.3f}",
            f"interval valid={snapshot.threshold_interval_valid} thresholds={snapshot.interval_threshold_count}",
            f"representative alpha={config.threshold_alpha:.3f} T={selected_text} (example only)",
            f"interval Junction stable={snapshot.interval_junction_stable} consistency={snapshot.interval_junction_consistency:.3f}",
            f"interval opening stable={snapshot.interval_opening_count_stable} consistency={snapshot.interval_opening_count_consistency:.3f}",
            f"smoothing={config.smoothing_window} OPEN support={int(np.count_nonzero(snapshot.open_support_mask))}",
            f"current openings={len(snapshot.opening_groups)} present={snapshot.opening_present}",
            f"CURRENT JUNCTION EVIDENCE = {snapshot.current_junction_evidence}",
            f"JUNCTION CONFIRMED = {snapshot.junction_confirmed} frame={_empty(snapshot.confirmation_frame)}",
            f"ANCHOR = {'FIXED' if snapshot.anchor_fixed else 'MOVING'} fix frame={_empty(snapshot.anchor_fix_frame)}",
            f"post-fix drift={session.post_fix_max_position_drift:.3e} normal motion={session.post_fix_max_normal_motion:.3e}",
            f"frozen55: T={snapshot.frozen55_threshold:.3f} groups={len(snapshot.frozen55_openings)}",
            "BLUE BAND = SAFE T INTERVAL (not OPEN range)",
            "PURPLE = ACTIVE OPEN SUPPORT",
        ]
        for index, value in enumerate(lines):
            color = COLORS["text"]
            if value.startswith("Detector") or value.startswith("PURPLE"):
                color = COLORS["group_center"]
            if (
                "EVIDENCE = True" in value
                or "CONFIRMED = True" in value
                or "ANCHOR = FIXED" in value
            ):
                color = COLORS["detected"]
            if value.endswith("False") and value.startswith("interval"):
                color = COLORS["detected"]
            self.text(
                value,
                (panel_x, y + index * step),
                color,
                self.small_font,
            )

        groups_y = y + len(lines) * step + 5
        self.text(
            "Opening groups (active threshold components)",
            (panel_x, groups_y),
            COLORS["group_center"],
            self.small_font,
        )
        if not snapshot.opening_groups:
            self.text(
                "none",
                (panel_x, groups_y + 18),
                COLORS["muted"],
                self.small_font,
            )
        for index, opening in enumerate(snapshot.opening_groups[:4]):
            base_y = groups_y + 18 + index * 17
            self.text(
                f"#{index} s={opening['start_angle']:+.1f} "
                f"e={opening['end_angle']:+.1f} "
                f"c={opening['center_angle']:+.1f} "
                f"w={opening['width_deg']:.1f} "
                f"conf={opening['confidence']:.3f}",
                (panel_x, base_y),
                COLORS["text"],
                self.small_font,
            )
        self.text(
            "SPACE pause/resume   R restart   LEFT/RIGHT sample   "
            "P profile   ESC quit",
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
        session.config,
        session.first_open_support_frame,
        session.first_opening_frame,
        session.first_detection_frame,
        session.first_evidence_frame,
        session.first_evidence_time,
        session.junction_confirmed,
        session.confirmation_frame,
        session.confirmation_time,
        session.frozen55_first_detection_frame,
        session.w_tau_first_detection_frame,
        session.anchor_fixed,
        session.anchor_fix_frame,
        session.anchor_fix_time,
        session.post_fix_max_position_drift,
        session.post_fix_max_normal_motion,
        tuple(sorted(session.sweep_first_detection.items())),
        tuple(sorted(session.sweep_max_openings.items())),
        tuple(
            json.dumps(row, sort_keys=True)
            for row in session.local_w_candidate_rows
        ),
        tuple(
            json.dumps(row, sort_keys=True)
            for row in session.local_w_timeline
        ),
        tuple(
            json.dumps(row, sort_keys=True)
            for row in session.corridor_w_timeline
        ),
        tuple(
            json.dumps(row, sort_keys=True)
            for row in session.corridor_w_calibration_samples
        ),
        tuple(
            (
                snapshot.physics_frame,
                tuple(snapshot.true_ranges),
                tuple(snapshot.noise_values),
                tuple(snapshot.raw_ranges),
                tuple(snapshot.smoothed_ranges),
                tuple(snapshot.open_support_mask),
                snapshot.open_threshold,
                snapshot.frozen55_threshold,
                snapshot.w_tau_selected_threshold,
                snapshot.threshold_interval_valid,
                snapshot.opening_groups,
                snapshot.frozen55_openings,
                snapshot.w_tau_openings,
                snapshot.junction_detected,
                snapshot.current_junction_evidence,
                snapshot.junction_confirmed,
                snapshot.confirmation_frame,
                snapshot.confirmation_time,
                snapshot.system_state,
                snapshot.state_transition,
                snapshot.frozen55_junction_detected,
                snapshot.w_tau_junction_detected,
                snapshot.interval_threshold_count,
                snapshot.interval_junction_stable,
                snapshot.interval_opening_count_stable,
                snapshot.interval_junction_consistency,
                snapshot.interval_opening_count_consistency,
                snapshot.anchor_fixed,
                snapshot.anchor_fix_frame,
                snapshot.anchor_fix_time,
                tuple(snapshot.local_w_estimate.far_mask),
                snapshot.local_w_estimate.far_runs,
                snapshot.local_w_estimate.candidates,
                snapshot.local_w_estimate.w_est_max,
                snapshot.local_w_estimate.w_est_median,
                snapshot.local_w_estimate.w_est_p90,
                snapshot.local_w_gt_eval_only.candidates,
                snapshot.w_gt_eval_only,
                snapshot.w_error_max,
                snapshot.w_error_median,
                snapshot.w_error_p90,
                snapshot.w_history_median,
                snapshot.w_history_p90,
            )
            for snapshot in session.snapshots
        ),
    )


def _movement_equivalent(
    first: AdaptiveSession,
    second: AdaptiveSession,
) -> tuple[bool, bool]:
    movement = bool(
        len(first.physics_trajectory) == len(second.physics_trajectory)
        and np.array_equal(
            np.asarray(first.physics_trajectory),
            np.asarray(second.physics_trajectory),
        )
    )
    lidar = bool(
        len(first.snapshots) == len(second.snapshots)
        and all(
            np.array_equal(left.angles_deg, right.angles_deg)
            and np.array_equal(left.true_ranges, right.true_ranges)
            and np.array_equal(left.noise_values, right.noise_values)
            and np.array_equal(left.raw_ranges, right.raw_ranges)
            for left, right in zip(first.snapshots, second.snapshots)
        )
    )
    return movement, lidar


def _config_from_args(
    args: argparse.Namespace,
    *,
    noise_model: str | None = None,
) -> DetectorExperimentConfig:
    return DetectorExperimentConfig(
        threshold_mode=args.threshold_mode,
        worst_wall_range=args.worst_wall_range,
        noise_model=args.noise_model if noise_model is None else noise_model,
        noise_fraction=args.noise_fraction,
        noise_sigma=args.noise_sigma,
        noise_seed=args.noise_seed,
        threshold_alpha=args.threshold_alpha,
        smoothing_window=args.smoothing_window,
        dump_beams=args.dump_beams,
        anchor_stop_on_detect=args.anchor_stop_on_detect,
        evaluate_interval=True,
        w_source=args.w_source,
        w_boundary_window_rays=args.w_boundary_window_rays,
        w_history_scans=args.w_history_scans,
        w_estimator_use_smoothed=args.w_estimator_use_smoothed,
        w_calibration_history_scans=args.w_calibration_history_scans,
        w_calibration_use_smoothed=args.w_calibration_use_smoothed,
        corridor_dominant_min_width_deg=(
            args.corridor_dominant_min_width_deg
        ),
        corridor_opposition_tolerance_deg=(
            args.corridor_opposition_tolerance_deg
        ),
        corridor_max_far_fraction=args.corridor_max_far_fraction,
        boundary_exclusion_rays=args.boundary_exclusion_rays,
        wall_support_definition=args.wall_support_definition,
        pose_condition=args.pose_condition,
        pose_lateral_offset=args.pose_lateral_offset,
        pose_yaw_offset_deg=args.pose_yaw_offset_deg,
        pre_corridor_calibration_start=args.pre_corridor_calibration_start,
    )


def run_gui(
    args: argparse.Namespace,
    config: DetectorExperimentConfig,
) -> AdaptiveSession:
    import pygame

    pygame.init()
    session = AdaptiveSession(args.map_case, config)
    renderer = AdaptiveRenderer(
        pygame,
        session.runner.geometry,
        args.show_profile,
    )
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
    map_case: str,
    frames: int,
    config: DetectorExperimentConfig,
) -> tuple[AdaptiveSession, bool, bool, bool]:
    primary = AdaptiveSession(map_case, config).run(frames)
    replay = AdaptiveSession(map_case, config).run(frames)
    movement, lidar = _movement_equivalent(primary, replay)
    deterministic = _session_signature(primary) == _session_signature(replay)
    if not movement or not lidar or not deterministic:
        raise AssertionError(
            json.dumps(
                {
                    "map_case": map_case,
                    "noise_model": config.noise_model,
                    "movement_equivalent": movement,
                    "lidar_equivalent": lidar,
                    "deterministic_replay": deterministic,
                },
                sort_keys=True,
            )
        )
    return primary, deterministic, movement, lidar


# =============================================================================
# 6. Reports, validation matrix, CLI
# =============================================================================

def _run_synthetic_corridor_regression(
    config: DetectorExperimentConfig,
) -> list[dict[str, Any]]:
    angles = np.arange(-180.0, 180.0, 1.0)

    def base_corridor() -> np.ndarray:
        values = np.full(angles.size, 80.0, dtype=float)
        values[np.abs(angles) <= 12.0] = LIDAR_MAX_RANGE
        values[np.abs(np.abs(angles) - 180.0) <= 12.0] = LIDAR_MAX_RANGE
        return values

    scans: list[tuple[str, np.ndarray, bool]] = []
    normal = base_corridor()
    scans.append(("A_NORMAL_CORRIDOR", normal, True))
    three_way = normal.copy()
    three_way[np.abs(angles - 90.0) <= 12.0] = LIDAR_MAX_RANGE
    scans.append(("B_THREE_WAY_OPENING", three_way, False))
    isolated = normal.copy()
    isolated[np.argmin(np.abs(angles - 90.0))] = LIDAR_MAX_RANGE
    scans.append(("C_ISOLATED_FAR_RAY", isolated, True))
    scans.append(("D_ALMOST_ALL_FAR", np.full(angles.size, 150.0), False))
    grazing = normal.copy()
    grazing[np.argmin(np.abs(angles - 60.0))] = 149.0
    scans.append(("E_GRAZING_149", grazing, True))
    scans.append(("F_NOISE_FREE_REPEAT", normal.copy(), True))

    rows: list[dict[str, Any]] = []
    prior_normal: dict[str, Any] | None = None
    for name, ranges, expected in scans:
        result = calibrate_corridor_w_scan_shadow(
            angles, ranges,
            max_range=LIDAR_MAX_RANGE,
            tau=config.tau,
            dominant_min_width_deg=config.corridor_dominant_min_width_deg,
            opposition_tolerance_deg=config.corridor_opposition_tolerance_deg,
            max_far_fraction=config.corridor_max_far_fraction,
            boundary_exclusion_rays=config.boundary_exclusion_rays,
            wall_support_definition=config.wall_support_definition,
        )
        passed = bool(result["corridor_like"] == expected)
        if name == "F_NOISE_FREE_REPEAT":
            passed = bool(
                passed
                and prior_normal is not None
                and np.array_equal(result["far_mask"], prior_normal["far_mask"])
                and result["wall_statistics"] == prior_normal["wall_statistics"]
            )
        if name == "A_NORMAL_CORRIDOR":
            prior_normal = result
        contaminated_stats = _robust_wall_statistics(
            ranges[ranges < LIDAR_MAX_RANGE]
        )
        rows.append({
            "case": name,
            "expected_corridor_like": expected,
            "actual_corridor_like": result["corridor_like"],
            "dominant_far_run_count": len(result["dominant_far_runs"]),
            "far_fraction": result["far_fraction"],
            **{
                f"w_{statistic}": _empty(
                    result["wall_statistics"][statistic]
                )
                for statistic in W_STATISTICS
            },
            **{
                f"all_physical_returns_{statistic}_eval_only": _empty(
                    contaminated_stats[statistic]
                )
                for statistic in W_STATISTICS
            },
            "pass": passed,
        })
    if not all(bool(row["pass"]) for row in rows):
        raise AssertionError(f"synthetic corridor regression failed: {rows!r}")
    return rows

def _detector_validation_rows(
    sessions: list[AdaptiveSession],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session in sessions:
        for detector in ("FROZEN_55PCT", "W_TAU_NOISE_AWARE"):
            if detector == "FROZEN_55PCT":
                first = session.frozen55_first_detection_frame
                maximum = max(
                    (
                        len(snapshot.frozen55_openings)
                        for snapshot in session.snapshots
                    ),
                    default=0,
                )
                threshold = (
                    session.snapshots[0].frozen55_threshold
                    if session.snapshots
                    else math.nan
                )
            else:
                first = session.w_tau_first_detection_frame
                maximum = max(
                    (
                        len(snapshot.w_tau_openings)
                        for snapshot in session.snapshots
                    ),
                    default=0,
                )
                selected = session.config.selected_w_tau_threshold
                threshold = math.nan if selected is None else selected
            rows.append(
                {
                    "case": session.map_case,
                    "noise": session.config.noise_model,
                    "detector": detector,
                    "first_detection_frame": _empty(first),
                    "max_openings": maximum,
                    "false_positive": bool(
                        session.map_case == M0_CASE and first is not None
                    ),
                    "threshold": threshold,
                }
            )
    return rows


SWEEP_SUMMARY_FIELDS = (
    "case", "noise", "detector", "alpha", "threshold",
    "first_detection_frame", "first_detection_time", "max_opening_count",
    "junction_detected", "M0_false_positive", "anchor_mode",
)


def _threshold_sweep_output_rows(
    sessions: Sequence[AdaptiveSession],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session in sessions:
        if session.config.evaluate_interval:
            rows.extend(session.threshold_sweep_summary_rows())
        else:
            first = session.w_tau_first_detection_frame
            first_snapshot = next(
                (
                    snapshot for snapshot in session.snapshots
                    if snapshot.physics_frame == first
                ),
                None,
            )
            rows.append(
                {
                    "case": session.map_case,
                    "noise": session.config.noise_model,
                    "detector": "W_TAU_SAFE_INTERVAL",
                    "alpha": session.config.threshold_alpha,
                    "threshold": session.config.selected_w_tau_threshold,
                    "first_detection_frame": _empty(first),
                    "first_detection_time": (
                        "" if first_snapshot is None
                        else first_snapshot.timestamp
                    ),
                    "max_opening_count": max(
                        (len(item.w_tau_openings) for item in session.snapshots),
                        default=0,
                    ),
                    "junction_detected": first is not None,
                    "M0_false_positive": bool(
                        session.map_case == M0_CASE and first is not None
                    ),
                    "anchor_mode": "CLOSED_LOOP_ANCHOR_STOP",
                }
            )

        frozen_first = session.frozen55_first_detection_frame
        frozen_snapshot = next(
            (
                snapshot for snapshot in session.snapshots
                if snapshot.physics_frame == frozen_first
            ),
            None,
        )
        rows.append(
            {
                "case": session.map_case,
                "noise": session.config.noise_model,
                "detector": "FROZEN_55PCT",
                "alpha": "",
                "threshold": (
                    "" if not session.snapshots
                    else session.snapshots[0].frozen55_threshold
                ),
                "first_detection_frame": _empty(frozen_first),
                "first_detection_time": (
                    "" if frozen_snapshot is None
                    else frozen_snapshot.timestamp
                ),
                "max_opening_count": max(
                    (len(item.frozen55_openings) for item in session.snapshots),
                    default=0,
                ),
                "junction_detected": frozen_first is not None,
                "M0_false_positive": bool(
                    session.map_case == M0_CASE
                    and frozen_first is not None
                ),
                "anchor_mode": (
                    "CLOSED_LOOP_ANCHOR_STOP"
                    if session.config.anchor_stop_on_detect
                    else "OPEN_LOOP_DIAGNOSTIC"
                ),
            }
        )
    return rows


def _local_w_error_statistics(
    rows: Sequence[dict[str, Any]],
    field_name: str,
) -> dict[str, Any]:
    values = np.asarray(
        [
            float(row[field_name])
            for row in rows
            if row[field_name] != ""
        ],
        dtype=float,
    )
    if values.size == 0:
        return {"mae": "", "median": "", "p95": "", "maximum": ""}
    return {
        "mae": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def _local_w_case_summary_rows(
    sessions: Sequence[AdaptiveSession],
) -> list[dict[str, Any]]:
    summaries = []
    for session in sessions:
        rows = session.local_w_timeline
        max_stats = _local_w_error_statistics(rows, "w_error_max")
        median_stats = _local_w_error_statistics(rows, "w_error_median")
        p90_stats = _local_w_error_statistics(rows, "w_error_p90")
        available = sum(bool(row["w_est_available"]) for row in rows)
        summaries.append(
            {
                "map_case": session.map_case,
                "noise_model": session.config.noise_model,
                "estimator_input": (
                    "smoothed_measured_range"
                    if session.config.w_estimator_use_smoothed
                    else "raw_measured_range"
                ),
                "sample_count": len(rows),
                "w_estimator_availability_rate": (
                    available / len(rows) if rows else 0.0
                ),
                "w_estimator_unavailable_rate": (
                    1.0 - available / len(rows) if rows else 1.0
                ),
                "w_max_mae": max_stats["mae"],
                "w_max_median_abs_error": max_stats["median"],
                "w_max_p95_error": max_stats["p95"],
                "w_max_max_error": max_stats["maximum"],
                "w_median_mae": median_stats["mae"],
                "w_median_median_abs_error": median_stats["median"],
                "w_median_p95_error": median_stats["p95"],
                "w_median_max_error": median_stats["maximum"],
                "w_p90_mae": p90_stats["mae"],
                "w_p90_median_abs_error": p90_stats["median"],
                "w_p90_p95_error": p90_stats["p95"],
                "w_p90_max_error": p90_stats["maximum"],
                "first_evidence_frame": _empty(
                    session.first_evidence_frame
                ),
                "confirmation_frame": _empty(session.confirmation_frame),
                "anchor_fix_frame": _empty(session.anchor_fix_frame),
                "active_w_source": "MANUAL",
                "active_manual_w": session.config.worst_wall_range,
                "shadow_w_feedback_used": False,
                "runtime_gt_map_used": False,
                "evaluation_gt_used": True,
            }
        )
    return summaries


def _corridor_w_case_summary_rows(
    sessions: Sequence[AdaptiveSession],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for session in sessions:
        rows = session.corridor_w_timeline
        eligible = [row for row in rows if bool(row["corridor_like"])]
        row: dict[str, Any] = {
            "map_case": session.map_case,
            "noise_model": session.config.noise_model,
            "pose_condition": session.config.pose_condition,
            "calibration_input": (
                "smoothed_measured_range"
                if session.config.w_calibration_use_smoothed
                else "raw_measured_range"
            ),
            "sample_count": len(rows),
            "corridor_calibration_availability": bool(eligible),
            "eligible_scan_rate": len(eligible) / len(rows) if rows else 0.0,
            "near_rmax_wall_contamination_rate_eval_only": (
                float(np.mean([
                    float(item["near_rmax_wall_fraction_eval_only"])
                    for item in eligible
                ])) if eligible else 0.0
            ),
            "raw_vs_smoothed_mean_abs_difference": (
                float(np.mean([
                    float(item["raw_vs_smoothed_mean_abs_difference"])
                    for item in rows
                ])) if rows else 0.0
            ),
            "first_w_ready_shadow_frame": _empty(
                session.first_w_ready_shadow_frame
            ),
            "w_frozen_shadow": session.w_frozen_shadow,
            "first_w_frozen_shadow_frame": _empty(
                session.first_w_frozen_shadow_frame
            ),
            "active_junction_evidence_frame": _empty(
                session.first_evidence_frame
            ),
            "anchor_fix_frame": _empty(session.anchor_fix_frame),
            "post_fix_max_position_drift": session.post_fix_max_position_drift,
            "active_w_source": "MANUAL",
            "active_manual_w": session.config.worst_wall_range,
            "shadow_w_feedback_used": False,
            "runtime_gt_map_used": False,
            "evaluation_gt_used": True,
            "pre_corridor_start_eval_only": (
                session.config.pre_corridor_calibration_start
            ),
        }
        for statistic in W_STATISTICS:
            current = np.asarray([
                float(item[f"w_{statistic}_current"])
                for item in eligible
                if item[f"w_{statistic}_current"] != ""
            ], dtype=float)
            history = np.asarray([
                float(item[f"w_{statistic}_history_median"])
                for item in eligible
                if item[f"w_{statistic}_history_median"] != ""
            ], dtype=float)
            row[f"w_{statistic}_mean"] = (
                float(np.mean(current)) if current.size else ""
            )
            row[f"w_{statistic}_median"] = (
                float(np.median(current)) if current.size else ""
            )
            row[f"w_{statistic}_mad"] = (
                float(np.median(np.abs(current - np.median(current))))
                if current.size else ""
            )
            row[f"w_{statistic}_temporal_variation_p95"] = (
                float(np.quantile(np.abs(np.diff(current)), 0.95))
                if current.size > 1 else 0.0 if current.size else ""
            )
            row[f"w_{statistic}_history_temporal_variation_p95"] = (
                float(np.quantile(np.abs(np.diff(history)), 0.95))
                if history.size > 1 else 0.0 if history.size else ""
            )
            valid_key = f"w_safe_interval_valid_{statistic}"
            row[f"w_{statistic}_safe_interval_validity_rate"] = (
                sum(bool(item[valid_key]) for item in eligible) / len(eligible)
                if eligible else 0.0
            )
        summaries.append(row)
    return summaries


def _corridor_w_statistic_comparison_rows(
    sessions: Sequence[AdaptiveSession],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in _corridor_w_case_summary_rows(sessions):
        for statistic in W_STATISTICS:
            rows.append({
                "map_case": summary["map_case"],
                "noise_model": summary["noise_model"],
                "pose_condition": summary["pose_condition"],
                "statistic": statistic,
                "current_mean": summary[f"w_{statistic}_mean"],
                "current_median": summary[f"w_{statistic}_median"],
                "current_mad": summary[f"w_{statistic}_mad"],
                "current_temporal_variation_p95": summary[
                    f"w_{statistic}_temporal_variation_p95"
                ],
                "history_temporal_variation_p95": summary[
                    f"w_{statistic}_history_temporal_variation_p95"
                ],
                "safe_interval_validity_rate": summary[
                    f"w_{statistic}_safe_interval_validity_rate"
                ],
                "safe_w_upper_limit": (
                    LIDAR_MAX_RANGE - 2.0 * sessions[0].config.tau
                ),
                "shadow_w_feedback_used": False,
            })
    return rows


def write_reports(
    output: Path,
    results: list[tuple[AdaptiveSession, bool, bool, bool]],
) -> None:
    sessions = [result[0] for result in results]
    _write(
        output / "w_tau_interval_timeline.csv",
        (row for session in sessions for row in session.timeline),
        TIMELINE_FIELDS,
    )
    _write(
        output / "detector_comparison.csv",
        (row for session in sessions for row in session.comparison),
        COMPARISON_FIELDS,
    )
    _write(
        output / "local_w_candidates.csv",
        (
            row
            for session in sessions
            for row in session.local_w_candidate_rows
        ),
        LOCAL_W_CANDIDATE_FIELDS,
    )
    _write(
        output / "local_w_timeline.csv",
        (
            row
            for session in sessions
            for row in session.local_w_timeline
        ),
        LOCAL_W_TIMELINE_FIELDS,
    )
    local_w_summaries = _local_w_case_summary_rows(sessions)
    _write(
        output / "local_w_case_summary.csv",
        local_w_summaries,
        tuple(local_w_summaries[0]),
    )
    _write(
        output / "corridor_w_timeline.csv",
        (row for session in sessions for row in session.corridor_w_timeline),
        tuple(sessions[0].corridor_w_timeline[0]),
    )
    calibration_samples = [
        row for session in sessions
        for row in session.corridor_w_calibration_samples
    ]
    _write(
        output / "corridor_w_calibration_samples.csv",
        calibration_samples,
        tuple(calibration_samples[0]) if calibration_samples else (
            "map_case", "frame", "timestamp"
        ),
    )
    corridor_summaries = _corridor_w_case_summary_rows(sessions)
    _write(
        output / "corridor_w_case_summary.csv",
        corridor_summaries,
        tuple(corridor_summaries[0]),
    )
    statistic_rows = _corridor_w_statistic_comparison_rows(sessions)
    _write(
        output / "corridor_w_statistic_comparison.csv",
        statistic_rows,
        tuple(statistic_rows[0]),
    )
    definition_rows = [
        row for session in sessions
        for row in session.corridor_w_definition_rows
    ]
    _write(
        output / "corridor_w_wall_support_comparison.csv",
        definition_rows,
        tuple(definition_rows[0]),
    )
    summaries = [
        session.summary(
            deterministic_replay=deterministic,
            movement_equivalent=movement,
            lidar_equivalent=lidar,
        )
        for session, deterministic, movement, lidar in results
    ]
    _write(output / "case_summary.csv", summaries, tuple(summaries[0]))
    _write(
        output / "threshold_sweep_summary.csv",
        _threshold_sweep_output_rows(sessions),
        SWEEP_SUMMARY_FIELDS,
    )
    _write(
        output / "threshold_sweep_frames.csv",
        (
            row
            for session in sessions
            for row in session.threshold_sweep_frames
        ),
        THRESHOLD_SWEEP_FRAME_FIELDS,
    )
    _write(
        output / "anchor_stop_diagnostics.csv",
        [
            {
                "case": session.map_case,
                "noise_model": session.config.noise_model,
                "threshold_alpha": session.config.threshold_alpha,
                "threshold": _empty(
                    session.config.selected_w_tau_threshold
                ),
                "junction_detected": (
                    session.first_detection_frame is not None
                ),
                "current_junction_evidence": (
                    False if not session.snapshots
                    else session.snapshots[-1].current_junction_evidence
                ),
                "junction_confirmed": session.junction_confirmed,
                "confirmation_frame": _empty(session.confirmation_frame),
                "confirmation_time": _empty(session.confirmation_time),
                "system_state": (
                    SYSTEM_STATE_FIXED_ANCHOR
                    if session.anchor_fixed
                    else (
                        SYSTEM_STATE_JUNCTION_CONFIRMED
                        if session.junction_confirmed
                        else SYSTEM_STATE_MOVING
                    )
                ),
                "state_transition": (
                    SYSTEM_STATE_MOVING
                    if not session.snapshots
                    else session.snapshots[-1].state_transition
                ),
                "anchor_stop_on_detect": (
                    session.config.anchor_stop_on_detect
                ),
                "anchor_fixed": session.anchor_fixed,
                "anchor_fix_frame": _empty(session.anchor_fix_frame),
                "anchor_fix_time": _empty(session.anchor_fix_time),
                "anchor_fix_x_eval_only": (
                    "" if session.anchor_fix_position is None
                    else float(session.anchor_fix_position[0])
                ),
                "anchor_fix_y_eval_only": (
                    "" if session.anchor_fix_position is None
                    else float(session.anchor_fix_position[1])
                ),
                "post_fix_max_position_drift": (
                    session.post_fix_max_position_drift
                ),
                "post_fix_max_normal_motion": (
                    session.post_fix_max_normal_motion
                ),
            }
            for session in sessions
        ],
        (
            "case", "noise_model", "threshold_alpha", "threshold",
            "junction_detected", "current_junction_evidence",
            "junction_confirmed", "confirmation_frame",
            "confirmation_time", "system_state", "state_transition",
            "anchor_stop_on_detect", "anchor_fixed",
            "anchor_fix_frame", "anchor_fix_time",
            "anchor_fix_x_eval_only", "anchor_fix_y_eval_only",
            "post_fix_max_position_drift", "post_fix_max_normal_motion",
        ),
    )
    first_config = sessions[0].config
    lower, upper, valid = first_config.safe_interval
    selected = first_config.selected_w_tau_threshold
    _write(
        output / "w_tau_interval.csv",
        [
            {
                "W": first_config.worst_wall_range,
                "tau": first_config.tau,
                "lower": lower,
                "upper": upper,
                "valid": valid,
                "selected_T": math.nan if selected is None else selected,
                "threshold_alpha": first_config.threshold_alpha,
            }
        ],
        (
            "W", "tau", "lower", "upper", "valid", "selected_T",
            "threshold_alpha",
        ),
    )
    beam_rows = [
        row
        for session in sessions
        for row in session.beam_rows
    ]
    if beam_rows:
        _write(output / "beam_diagnostics.csv", beam_rows, BEAM_FIELDS)


def _print_validation_tables(sessions: list[AdaptiveSession]) -> None:
    print("case | noise | T | alpha | first detection | max openings | false positive")
    for row in _threshold_sweep_output_rows(sessions):
        print(
            f"{row['case']} | {row['noise']} | {row['threshold']} | "
            f"{row['alpha']} | {row['first_detection_frame']} | "
            f"{row['max_opening_count']} | {row['M0_false_positive']}"
        )
    print("case | noise | T_low | T_high | tested thresholds | Junction stable | Opening-count stable")
    for session in sessions:
        lower, upper, valid = session.config.safe_interval
        junction_stable, opening_stable = session.interval_outcome_stability()
        print(
            f"{session.map_case} | {session.config.noise_model} | "
            f"{lower} | {upper} | "
            f"{len(session.threshold_sweep_summary_rows()) if valid else 0} | "
            f"{junction_stable} | {opening_stable}"
        )
    print("case | first detection frame | anchor fix frame | post-fix drift")
    for session in sessions:
        print(
            f"{session.map_case} | {_empty(session.first_detection_frame)} | "
            f"{_empty(session.anchor_fix_frame)} | "
            f"{session.post_fix_max_position_drift}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map-case",
        choices=MAP_CASES,
        default=M1_PRE_CORRIDOR_CASE,
    )
    parser.add_argument(
        "--threshold-mode",
        choices=THRESHOLD_MODES,
        default="w-tau",
    )
    parser.add_argument(
        "--worst-wall-range",
        type=float,
        required=True,
        help="calibrated W; never estimated from runtime map/scan",
    )
    parser.add_argument(
        "--w-source",
        choices=("manual",),
        default="manual",
        help="active detector source; EXP-046 intentionally permits manual only",
    )
    parser.add_argument(
        "--w-boundary-window-rays",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--w-history-scans",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--w-estimator-use-smoothed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="optional shadow diagnostic; raw measured range is the default",
    )
    parser.add_argument("--w-calibration-history-scans", type=int, default=10)
    parser.add_argument(
        "--w-calibration-use-smoothed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="shadow calibration input; raw measured range is the default",
    )
    parser.add_argument(
        "--corridor-dominant-min-width-deg", type=float, default=5.0
    )
    parser.add_argument(
        "--corridor-opposition-tolerance-deg", type=float, default=35.0,
        help="CLI-exposed tolerance around opposite (180-degree) far runs",
    )
    parser.add_argument(
        "--corridor-max-far-fraction", type=float, default=0.70
    )
    parser.add_argument("--boundary-exclusion-rays", type=int, default=3)
    parser.add_argument(
        "--wall-support-definition",
        choices=WALL_SUPPORT_DEFINITIONS,
        default="RUN_BOUNDARY_EXCLUDED",
    )
    parser.add_argument("--pose-condition", choices=POSE_CONDITIONS, default="center")
    parser.add_argument("--pose-lateral-offset", type=float, default=4.0)
    parser.add_argument("--pose-yaw-offset-deg", type=float, default=3.0)
    parser.add_argument(
        "--pre-corridor-calibration-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "evaluation initial condition: reuse EXP-042 rear start on "
            "Junction maps; never enters the runtime calibrator"
        ),
    )
    parser.add_argument(
        "--noise-model",
        choices=NOISE_MODELS,
        default="truncated-gaussian",
    )
    parser.add_argument(
        "--noise-fraction",
        type=float,
        default=DEFAULT_NOISE_FRACTION,
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=DEFAULT_NOISE_SIGMA,
        help=(
            "experimental truncated-Gaussian sigma; default 2.5 is not "
            "a theoretically established value"
        ),
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=DEFAULT_NOISE_SEED,
    )
    parser.add_argument(
        "--threshold-alpha",
        type=float,
        default=DEFAULT_THRESHOLD_ALPHA,
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
    )
    parser.add_argument("--dump-beams", action="store_true")
    parser.add_argument(
        "--anchor-stop-on-detect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="latch only the LiDAR robot as fixed at first Junction detection",
    )
    parser.add_argument(
        "--threshold-sweep",
        action="store_true",
        help=(
            "evaluate alpha=0.05..0.95; closed-loop mode independently "
            "replays every threshold"
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="0: GUI until ESC; headless/validate default 240",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--pause-on-detect", action="store_true")
    parser.add_argument(
        "--show-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "A-D none/truncated-Gaussian open-loop sweep replay plus a "
            "closed-loop truncated-Gaussian anchor replay"
        ),
    )
    parser.add_argument(
        "--validate-w-estimator",
        action="store_true",
        help="replayed none/truncated-Gaussian W-estimator matrix on all maps",
    )
    parser.add_argument(
        "--validate-corridor-calibration",
        action="store_true",
        help="EXP-046 all-map/noise deterministic matrix plus synthetic tests",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error("--frames must be non-negative")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if (
        args.headless
        or args.validate
        or args.validate_w_estimator
        or args.validate_corridor_calibration
        or args.threshold_sweep
    ) and args.frames == 0:
        args.frames = 240
    try:
        _config_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _audit_frozen_detector_defaults()
    _audit_corridor_calibrator_local_only()
    results: list[tuple[AdaptiveSession, bool, bool, bool]] = []
    synthetic_rows: list[dict[str, Any]] = []
    if args.validate_corridor_calibration:
        config = _config_from_args(args)
        synthetic_rows = _run_synthetic_corridor_regression(config)
        for map_case in MAP_CASES:
            for noise_model in ("none", "truncated-gaussian"):
                case_config = _config_from_args(args, noise_model=noise_model)
                results.append(
                    _run_replayed_case(map_case, args.frames, case_config)
                )
        if args.pose_condition == "center":
            for pose_condition in POSE_CONDITIONS[1:]:
                for noise_model in ("none", "truncated-gaussian"):
                    pose_config = replace(
                        _config_from_args(args, noise_model=noise_model),
                        pose_condition=pose_condition,
                    )
                    results.append(_run_replayed_case(
                        M1_PRE_CORRIDOR_CASE, args.frames, pose_config
                    ))
    elif args.validate_w_estimator:
        for map_case in MAP_CASES:
            for noise_model in ("none", "truncated-gaussian"):
                config = _config_from_args(args, noise_model=noise_model)
                results.append(
                    _run_replayed_case(map_case, args.frames, config)
                )
    elif args.threshold_sweep:
        config = _config_from_args(args)
        _, _, valid = config.safe_interval
        if not valid:
            results.append(
                _run_replayed_case(args.map_case, args.frames, config)
            )
        elif config.anchor_stop_on_detect:
            # Closed-loop comparison must not share a stop event: each scalar
            # T receives a fresh world and the same seed/initial state.
            for alpha in DEFAULT_SWEEP_ALPHAS:
                independent = replace(
                    config,
                    threshold_alpha=alpha,
                    evaluate_interval=False,
                )
                session = AdaptiveSession(
                    args.map_case,
                    independent,
                ).run(args.frames)
                results.append((session, False, False, False))
        else:
            # In open loop, all thresholds see the exact same scan history, so
            # one replay is both fair and much cheaper than 19 physics runs.
            results.append(
                _run_replayed_case(args.map_case, args.frames, config)
            )
    elif args.validate:
        scenarios = (
            (M0_CASE, "none"),
            (M1_PRE_CORRIDOR_CASE, "none"),
            (M0_CASE, "truncated-gaussian"),
            (M1_PRE_CORRIDOR_CASE, "truncated-gaussian"),
        )
        for map_case, noise_model in scenarios:
            # Fair interval timing is an open-loop diagnostic; all 19 scalar
            # thresholds consume the identical trajectory and noise sequence.
            config = replace(
                _config_from_args(args, noise_model=noise_model),
                anchor_stop_on_detect=False,
            )
            results.append(
                _run_replayed_case(map_case, args.frames, config)
            )
        # Separately validate the intended closed-loop latch on the principal
        # noisy M1 condition without contaminating sweep timing.
        anchor_config = replace(
            _config_from_args(args, noise_model="truncated-gaussian"),
            anchor_stop_on_detect=True,
        )
        results.append(
            _run_replayed_case(
                M1_PRE_CORRIDOR_CASE,
                args.frames,
                anchor_config,
            )
        )
        _print_validation_tables([result[0] for result in results])
    elif args.headless:
        config = _config_from_args(args)
        results.append(
            _run_replayed_case(args.map_case, args.frames, config)
        )
    else:
        config = _config_from_args(args)
        session = run_gui(args, config)
        results.append((session, False, False, False))
    write_reports(args.output_dir, results)
    if synthetic_rows:
        _write(
            args.output_dir / "corridor_w_synthetic_regression.csv",
            synthetic_rows,
            tuple(synthetic_rows[0]),
        )
    for session, deterministic, movement, lidar in results:
        print(
            json.dumps(
                session.summary(
                    deterministic_replay=deterministic,
                    movement_equivalent=movement,
                    lidar_equivalent=lidar,
                ),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
