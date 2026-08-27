
from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np



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


# =============================================================================
# 3. Standalone sampled runner and validated EXP-042 rear-start
# =============================================================================

M0_CASE = "M0_STRAIGHT"
M1_BASELINE_CASE = "M1_CROSS_BASELINE"
M1_PRE_CORRIDOR_CASE = "M1_PRE_CORRIDOR_55PCT"
MAP_CASES = (M0_CASE, M1_BASELINE_CASE, M1_PRE_CORRIDOR_CASE)
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
EXPERIMENT_NAME = "Moving LiDAR Adaptive 55% Junction Detection — standalone"
DETECTOR_NAME = "ADAPTIVE_RANGE_55_PERCENT"
DEFAULT_OUTPUT = ROOT / "lidar_junction_detection_55pct_pre_corridor_output"
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


def _audit_detector_defaults() -> dict[str, Any]:
    actual = {
        name: inspect.signature(
            _detect_openings_with_diagnostics
        ).parameters[name].default
        for name in PARAMETERS
    }
    if actual != PARAMETERS:
        raise AssertionError(
            f"detector defaults changed: expected={PARAMETERS!r} actual={actual!r}"
        )
    return actual


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
        # The previous profile detector was comparison-only and never fed back.
        # Standalone execution keeps the adaptive decision path only.
        old_detected = False
        old_count = 0
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

# =============================================================================
# 6. Reports, deterministic validation, CLI
# =============================================================================

def write_reports(
    output: Path,
    sessions: list[AdaptiveSession],
    replay: dict[str, bool],
    movement: dict[str, bool],
    lidar: dict[str, bool],
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map-case",
        choices=MAP_CASES,
        default=M1_PRE_CORRIDOR_CASE,
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
        help="headless three-case deterministic replay",
    )
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
    write_reports(args.output_dir, sessions, replay, movement, lidar)
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
