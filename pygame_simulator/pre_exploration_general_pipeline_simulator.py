"""Clean evaluation/research simulator for the pre-exploration pipeline.

This module does not import the legacy multi-geometry evaluator.  It implements
a small, general world around the production pre-Junction SPH equations while
excluding DFS, Guards, Shepherds, and branch routing. Its optional clean-GUI
path can latch a shadow suspicion into a provisional fixed observation Anchor.

The runtime diagnostics consume :class:`LocalObservation` only. They require
neither a global map, global localization, GPS, nor SLAM-based poses. Global
coordinates and branch metadata remain confined to the world and GT evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/pre_exploration_general_pipeline"
CASES = ("M0_STRAIGHT", "M1_CROSS_BASELINE", "M2_T_JUNCTION", "M3_ANGLED_Y", "M4_ASYMMETRIC_CROSS", "M5_UNEQUAL_WIDTH")
PROPULSION_MODES = ("production_compression", "local_forward")

# Copied from the production pre-Junction physical parameter block. The
# physical initialization pulse is retained; branch/DFS forces are excluded.
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
    """Global physics/world owner; branch metadata never leaves this class."""
    def __init__(self, geometry: GeometryCase, propulsion_mode: str = "production_compression"):
        if propulsion_mode not in PROPULSION_MODES:
            raise ValueError(f"unknown propulsion mode: {propulsion_mode}")
        self.geometry = geometry
        self.propulsion_mode = propulsion_mode
        self.robots = self._create_robots()
        self.initial_mean_y = float(np.mean([robot.position[1] for robot in self.robots]))
        self.initial_front_y = float(max(robot.position[1] for robot in self.robots))
        self.time = 0.0
        self.wall_contacts = self.wall_corrections = 0
        self.rest_lengths: dict[tuple[int, int], float] = {}
        self.lidar_robot_id = self._select_initial_lidar_leader()
        lidar = next(robot for robot in self.robots if robot.robot_id == self.lidar_robot_id)
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
        initial_mean_density = float(np.mean([robot.density for robot in self.robots]))
        # Both modes use the production normal-SPH reference density. The
        # local-forward variant skips only artificial energy storage/release.
        self.reference_density = initial_mean_density * 0.62
        entrance = geometry.entrance_y if geometry.entrance_y is not None else -42.0
        # Production BASE_COMPRESSION_CENTER is 0.60 Base lengths behind the
        # Junction entrance. The sign is converted to this world's +Y ingress.
        self.compression_center = np.array([0.0, entrance - 0.60 * PRODUCTION_BASE_LENGTH])
        self.ingress_target_y = entrance + 96.6
        self.last_mean_pressure_force = 0.0
        self.last_mean_repulsion_force = 0.0
        self.last_mean_lateral_sph_force = 0.0
        self.suspect_hold_active = False
        self.braking_active = False
        self.stationary_dwell_steps = 0
        self.provisional_fixed_anchor = False
        self.pointcloud_ready = False
        self.physics_frame_index = 0
        self.suspect_entry_position: np.ndarray | None = None
        self.suspect_entry_time = math.nan
        self.anchor_entry_frame = -1
        self.anchor_entry_time = math.nan
        self.anchor_position: np.ndarray | None = None
        self.anchor_heading_rad = math.nan

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

    def step(self):
        neighbors = self._neighbors(); self._densities(neighbors)
        production_mode = self.propulsion_mode == "production_compression"
        if not production_mode:
            # Communication propagation runs at the existing 0.1 s local
            # sensor cadence; gap-aware drive still updates every physics step.
            if self.local_graph_update_index % max(1, round(SAMPLE_PERIOD / DT)) == 0:
                self._update_local_heading_propagation(neighbors)
            else:
                self._update_leader_gap(neighbors)
            self.local_graph_update_index += 1
        pressure_scale = _pressure_scale(self.time) if production_mode else SPH_PRESSURE_SCALE
        for robot in self.robots:
            ratio = robot.density/max(self.reference_density,EPSILON)
            raw_pressure = max(0.0, PRESSURE_GAIN*robot.density*(ratio**STIFFNESS_EXPONENT-1.0))
            robot.pressure = raw_pressure * pressure_scale
            stored_floor = PRESSURE_GAIN * robot.density * BASE_STORED_PRESSURE_FLOOR * (_stored_pressure_envelope(self.time) if production_mode else 0.0)
            robot.pressure = max(robot.pressure, stored_floor)
        accelerations = {}
        equilibrium = _equilibrium_radius(self.time) if production_mode else SAFE_RADIUS * NORMAL_EQUILIBRIUM_SCALE
        release_active = production_mode and BASE_COMPRESSION_DURATION <= self.time < BASE_COMPRESSION_DURATION + BASE_EXPANSION_BOOST_DURATION
        pressure_magnitudes=[]; repulsion_magnitudes=[]; lateral_sph_magnitudes=[]
        for robot in self.robots:
            pressure=np.zeros(2); viscosity=np.zeros(2); elastic=np.zeros(2); repulsion=np.zeros(2)
            for peer in neighbors[robot.robot_id]:
                offset=robot.position-peer.position; distance=float(np.linalg.norm(offset))
                if distance <= EPSILON:
                    continue
                gradient=_gradient(offset)
                coefficient=robot.pressure/max(robot.density**2,EPSILON)+peer.pressure/max(peer.density**2,EPSILON)
                pressure += -coefficient*gradient
                weight=_kernel(distance)/max(_kernel(0.0),EPSILON)
                viscosity += (peer.velocity-robot.velocity)*(VELOCITY_CONSENSUS_GAIN*weight)
                # Adapted from production compute_sph_forces. This Monaghan
                # term damps approaching pairs and is independent of DFS.
                relative_velocity = robot.velocity-peer.velocity
                approach = float(np.dot(relative_velocity,offset))
                if approach < 0.0:
                    distance_sq=distance*distance
                    mu=SMOOTHING_LENGTH*approach/(distance_sq+0.01*SMOOTHING_LENGTH**2)
                    sound_i_sq=(robot.pressure+PRESSURE_GAIN*robot.density)/max(robot.density,EPSILON)
                    sound_j_sq=(peer.pressure+PRESSURE_GAIN*peer.density)/max(peer.density,EPSILON)
                    sound=0.5*(math.sqrt(max(sound_i_sq,0.0))+math.sqrt(max(sound_j_sq,0.0)))
                    mean_density=0.5*(robot.density+peer.density)
                    artificial=(-VISCOSITY_XI1*sound*mu+VISCOSITY_XI2*mu**2)/max(mean_density,EPSILON)
                    viscosity += -artificial*gradient
                if distance <= VISCOELASTIC_LINK_RADIUS:
                    pair=tuple(sorted((robot.robot_id,peer.robot_id)))
                    rest=self.rest_lengths.setdefault(pair,float(np.clip(equilibrium,VISCOELASTIC_REST_MIN,VISCOELASTIC_REST_MAX)))
                    if robot.robot_id < peer.robot_id:
                        rest += (float(np.clip(equilibrium,VISCOELASTIC_REST_MIN,VISCOELASTIC_REST_MAX))-rest) * min(1.0,4.0*DT)
                        rest += (distance-rest) * min(1.0,0.85*DT)
                        self.rest_lengths[pair]=rest
                    radial=float(np.dot(robot.velocity-peer.velocity,offset/distance))
                    elastic += offset/distance*(-VISCOELASTIC_ELASTIC_GAIN*(distance-rest)-VISCOELASTIC_DASHPOT_GAIN*radial)
                if distance < equilibrium:
                    repulsion += REPULSION_GAIN*(equilibrium-distance)/equilibrium*offset/distance
            pressure=_limit(pressure,INITIAL_RELEASE_PRESSURE_FORCE_LIMIT if release_active else PRESSURE_FORCE_LIMIT)
            if release_active: viscosity *= INITIAL_RELEASE_VISCOSITY_MULTIPLIER
            viscosity=_limit(viscosity,VISCOSITY_FORCE_LIMIT); elastic=_limit(elastic,VISCOELASTIC_FORCE_LIMIT)
            sph_force=pressure+viscosity+elastic+repulsion
            pressure_magnitudes.append(float(np.linalg.norm(pressure)))
            repulsion_magnitudes.append(float(np.linalg.norm(repulsion)))
            lateral_sph_magnitudes.append(abs(float(sph_force[0])))
            if not production_mode:
                # The command is [forward, lateral]=[F, 0] in the robot body
                # frame, transformed solely for physics integration.
                route = np.zeros(2) if self.suspect_hold_active else np.array([
                    math.cos(robot.body_yaw_rad), math.sin(robot.body_yaw_rad)
                ]) * LOCAL_FORWARD_DRIVE_FORCE * robot.propulsion_weight
            elif self.time < BASE_COMPRESSION_DURATION:
                compression = self.compression_center - robot.position
                distance = float(np.linalg.norm(compression))
                scale = min(1.0, max(0.15, distance / max(self.geometry.incoming_width * 0.55 / 2.0, EPSILON)))
                route = compression / max(distance, EPSILON) * BASE_COMPRESSION_FORCE * scale * _compression_envelope(self.time)
            else:
                distance_to_target = self.ingress_target_y - robot.position[1]
                forward_scale = max(INITIAL_INGRESS_MIN_FORCE_SCALE, min(1.0, distance_to_target / INITIAL_INGRESS_BRAKE_DISTANCE)) if distance_to_target > 0 else 0.0
                route = np.array([float(np.clip(INITIAL_INGRESS_LANE_GAIN*(robot.ingress_lane_x-robot.position[0]),-INITIAL_INGRESS_LANE_MAX_FORCE,INITIAL_INGRESS_LANE_MAX_FORCE)), INITIAL_INGRESS_FORCE*forward_scale])
            # Adapted from production compute_base_piston_reaction_force. The
            # physical Base release uses only initial-corridor depth here.
            entrance=self.geometry.entrance_y if self.geometry.entrance_y is not None else -42.0
            base_depth=float(np.clip((entrance-robot.position[1])/PRODUCTION_BASE_LENGTH,0.0,1.0))
            normalized_pressure=robot.pressure/max(PRESSURE_GAIN*robot.density,EPSILON)
            piston_magnitude=(min(BASE_PISTON_REACTION_FORCE_LIMIT,BASE_PISTON_REACTION_GAIN*normalized_pressure*_smoothstep(base_depth)*_base_piston_envelope(self.time)) if production_mode else 0.0)
            piston=np.array([0.0,piston_magnitude])
            braking_leader=not production_mode and self.suspect_hold_active and not self.provisional_fixed_anchor and robot.robot_id==self.lidar_robot_id
            total=pressure+viscosity+elastic+repulsion+route+piston-(DAMPING+(INITIAL_RELEASE_EXTRA_DAMPING if release_active else 0.0))*robot.velocity
            raw=_limit(total,MAX_ACCELERATION)
            if braking_leader:
                # Local velocity cancellation, bounded by the existing
                # acceleration scale. Bypassing the propulsion EMA prevents a
                # non-zero equilibrium with persistent SPH contact forces.
                raw=_limit(-robot.velocity/DT,MAX_ACCELERATION)
            alpha = 1.0 if braking_leader else (INITIAL_RELEASE_ACCELERATION_FILTER_ALPHA if release_active else ACCELERATION_FILTER_ALPHA)
            accelerations[robot.robot_id]=(1-alpha)*robot.acceleration+alpha*raw
        self.last_mean_pressure_force=float(np.mean(pressure_magnitudes))
        self.last_mean_repulsion_force=float(np.mean(repulsion_magnitudes))
        self.last_mean_lateral_sph_force=float(np.mean(lateral_sph_magnitudes))
        for robot in self.robots:
            if self.provisional_fixed_anchor and robot.robot_id==self.lidar_robot_id:
                robot.position=self.anchor_position.copy(); robot.velocity[:]=0.; robot.acceleration[:]=0.; robot.observed_velocity[:]=0.
                continue
            old=robot.position.copy(); robot.acceleration=accelerations[robot.robot_id]; robot.velocity += robot.acceleration*DT
            # LOCAL_FORWARD uses only two body-lateral wall ranges: damping is
            # active in a bounded corridor and disappears naturally when an
            # opening makes either side unbounded. No GT phase is consulted.
            if production_mode or self._local_wall_confinement(robot):
                robot.velocity[0] *= math.exp(-CORRIDOR_LATERAL_VELOCITY_DAMPING*DT)
            robot.velocity=_limit(robot.velocity,MAX_SPEED)
            # Production Robot.update performs axis-separated is_walkable
            # checks and damps/rebounds only the blocked velocity component.
            x_candidate=np.array([robot.position[0]+robot.velocity[0]*DT,robot.position[1]])
            if self.geometry.walkable(x_candidate): robot.position[0]=x_candidate[0]
            else:
                self.wall_contacts += 1; robot.velocity[0]=(-robot.velocity[0]*INITIAL_WALL_RESTITUTION if self.time>=BASE_COMPRESSION_DURATION else 0.0)
            y_candidate=np.array([robot.position[0],robot.position[1]+robot.velocity[1]*DT])
            if self.geometry.walkable(y_candidate): robot.position[1]=y_candidate[1]
            else:
                self.wall_contacts += 1; robot.velocity[1]=(-robot.velocity[1]*INITIAL_WALL_RESTITUTION if self.time>=BASE_COMPRESSION_DURATION else 0.0)
            if not self.geometry.walkable(robot.position):
                # Safety accounting only: normal dynamics should never need a
                # geometric projection, matching production's behavior.
                self.wall_corrections += 1
            robot.observed_velocity=(robot.position-old)/DT
        self._update_anchor_transition()
        self.time += DT
        self.physics_frame_index += 1

    def _update_anchor_transition(self) -> None:
        """Confirm local stationarity, then latch the current leader pose."""
        leader=next(robot for robot in self.robots if robot.robot_id==self.lidar_robot_id)
        if not self.suspect_hold_active or self.provisional_fixed_anchor:
            self.braking_active=False
            return
        self.braking_active=True
        if float(np.linalg.norm(leader.velocity)) < MIN_SPEED:
            self.stationary_dwell_steps += 1
        else:
            self.stationary_dwell_steps = 0
        if self.stationary_dwell_steps < ANCHOR_STATIONARY_DWELL_STEPS:
            return
        self.provisional_fixed_anchor=True; self.braking_active=False; self.pointcloud_ready=True
        self.anchor_entry_frame=self.physics_frame_index; self.anchor_entry_time=self.time+DT
        self.anchor_position=leader.position.copy(); self.anchor_heading_rad=leader.body_yaw_rad
        leader.velocity[:]=0.; leader.acceleration[:]=0.; leader.observed_velocity[:]=0.

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

    def sanity(self):
        positions=np.array([robot.position for robot in self.robots]); velocities=np.array([robot.velocity for robot in self.robots])
        minimum=float("inf"); nearest_all=[]; overlap_pairs=0; maximum_penetration=0.0
        for i in range(len(positions)):
            distances=np.linalg.norm(positions[i+1:]-positions[i],axis=1)
            if len(distances):
                minimum=min(minimum,float(np.min(distances))); overlap_pairs += int(np.sum(distances < 2*ROBOT_RADIUS))
                maximum_penetration=max(maximum_penetration,float(max(0.0,2*ROBOT_RADIUS-np.min(distances))))
            all_distances=np.linalg.norm(positions-positions[i],axis=1); all_distances[i]=np.inf; nearest_all.append(float(np.min(all_distances)))
        return {"outside_free_space_robot_count":sum(not self.geometry.contains(robot.position) for robot in self.robots),
            "wall_contact_count":self.wall_contacts,"wall_projection_correction_count":self.wall_corrections,
            "nan_inf_state_count":int(np.size(positions)-np.isfinite(positions).sum()+np.size(velocities)-np.isfinite(velocities).sum()),
            "max_speed":float(np.max(np.linalg.norm(velocities,axis=1))),"mean_speed_sanity":float(np.mean(np.linalg.norm(velocities,axis=1))),
            "min_inter_robot_distance":minimum,"mean_nearest_neighbor_distance_sanity":float(np.mean(nearest_all)),
            "overlap_pair_count":overlap_pairs,"maximum_pair_penetration":maximum_penetration,
            "swarm_lateral_span_sanity":float(np.ptp(positions[:,0])),"swarm_longitudinal_span_sanity":float(np.ptp(positions[:,1])),
            "mean_density":float(np.mean([robot.density for robot in self.robots])),"density_std":float(np.std([robot.density for robot in self.robots])),
            "mean_pressure":float(np.mean([robot.pressure for robot in self.robots])),"mean_pressure_force":self.last_mean_pressure_force,
            "mean_repulsion_force":self.last_mean_repulsion_force,"mean_lateral_sph_force":self.last_mean_lateral_sph_force}

    def connectivity_diagnostics(self):
        """Return local leader/graph diagnostics without world/GT fields."""
        leader=next(robot for robot in self.robots if robot.robot_id==self.lidar_robot_id)
        followers=[robot for robot in self.robots if robot.robot_id!=self.lidar_robot_id]
        leader_speed=float(np.linalg.norm(leader.velocity))
        anchor_displacement=(float(np.linalg.norm(leader.position-self.anchor_position)) if self.anchor_position is not None else 0.0)
        heading_change=(abs(math.degrees((leader.body_yaw_rad-self.anchor_heading_rad+math.pi)%(2*math.pi)-math.pi)) if math.isfinite(self.anchor_heading_rad) else 0.0)
        hold_displacement=(float(np.linalg.norm(leader.position-self.suspect_entry_position)) if self.suspect_entry_position is not None else 0.0)
        return {**self.last_connectivity,"suspect_hold_active":self.suspect_hold_active,"local_forward_propulsion_active":self.propulsion_mode=="local_forward" and not self.suspect_hold_active,
            "leader_braking_active":self.braking_active,"leader_speed":leader_speed,"leader_stationary":leader_speed<MIN_SPEED,"stationary_speed_threshold":MIN_SPEED,
            "follower_mean_speed":float(np.mean([np.linalg.norm(robot.velocity) for robot in followers])),"follower_moving_fraction":float(np.mean([np.linalg.norm(robot.velocity)>=MIN_SPEED for robot in followers])),
            "stationary_dwell_steps":self.stationary_dwell_steps,"stationary_dwell_target":ANCHOR_STATIONARY_DWELL_STEPS,"provisional_fixed_anchor":self.provisional_fixed_anchor,
            "pointcloud_ready":self.pointcloud_ready,"suspect_entry_time":self.suspect_entry_time,"anchor_entry_frame":self.anchor_entry_frame,"anchor_entry_time":self.anchor_entry_time,
            "anchor_x_eval":float(self.anchor_position[0]) if self.anchor_position is not None else math.nan,"anchor_y_eval":float(self.anchor_position[1]) if self.anchor_position is not None else math.nan,
            "leader_displacement_since_hold":hold_displacement,"leader_displacement_since_anchor":anchor_displacement,"leader_heading_change_since_anchor":heading_change}

    def activate_suspect_hold(self) -> None:
        """Latch LOCAL_FORWARD route suppression until the world is reset."""
        if self.propulsion_mode == "local_forward":
            if not self.suspect_hold_active:
                leader=next(robot for robot in self.robots if robot.robot_id==self.lidar_robot_id)
                self.suspect_entry_position=leader.position.copy(); self.suspect_entry_time=self.time
            self.suspect_hold_active = True

    def initialization_phase(self) -> str:
        if self.propulsion_mode == "local_forward" and self.provisional_fixed_anchor:
            return "PROVISIONAL_FIXED_ANCHOR"
        if self.propulsion_mode == "local_forward" and self.suspect_hold_active:
            return "SUSPECT_HOLD"
        if self.propulsion_mode == "local_forward":
            return "LOCAL_FORWARD"
        if self.time < BASE_COMPRESSION_DURATION:
            return "INITIAL_COMPRESSION"
        if self.time < BASE_COMPRESSION_DURATION + BASE_EXPANSION_BOOST_DURATION:
            return "INITIAL_RELEASE"
        return "NORMAL_INGRESS"


class LocalObservationBuilder:
    """Explicit world-to-runtime boundary."""
    @staticmethod
    def build(world: SimulatorWorld) -> LocalObservation:
        return world.local_observation()


class SwarmDiagnostics:
    """Map-independent continuous swarm diagnostics."""
    def __init__(self):
        self.previous_forward=None; self.previous_boundary=set()

    def analyze(self, observation: LocalObservation):
        positions=observation.relative_positions; velocities=observation.velocities; ids=observation.robot_ids
        moving=np.linalg.norm(velocities,axis=1)>=MIN_SPEED
        forward=np.sum(velocities[moving],axis=0) if np.any(moving) else np.array([0.,1.])
        if np.linalg.norm(forward)<EPSILON: forward=self.previous_forward.copy() if self.previous_forward is not None else np.array([0.,1.])
        forward=forward/max(np.linalg.norm(forward),EPSILON)
        if self.previous_forward is not None and np.dot(forward,self.previous_forward)<0: forward=-forward
        self.previous_forward=forward.copy(); lateral=np.array([-forward[1],forward[0]])
        center=np.mean(positions,axis=0); projections=(positions-center)@forward
        reference_mask=projections>=np.quantile(projections,REFERENCE_FRONT_QUANTILE)
        neighbors=[]
        for i in range(len(ids)):
            neighbors.append(np.where((np.linalg.norm(positions-positions[i],axis=1)<=SMOOTHING_LENGTH)&(np.arange(len(ids))!=i))[0])
        local_surface=[]
        for i in np.where(moving)[0]:
            if not any(np.dot(positions[j]-positions[i],velocities[i]/max(np.linalg.norm(velocities[i]),EPSILON))>0 for j in neighbors[i]): local_surface.append(i)
        local_indices=set(local_surface)
        for i in local_surface: local_indices.update(neighbors[i].tolist())
        local_mask=np.array([i in local_indices for i in range(len(ids))])
        def cohort(prefix,mask):
            selected=positions[mask]; selected_v=velocities[mask]
            if not len(selected): return {f"{prefix}_size":0,f"{prefix}_lateral_variance":0.,f"{prefix}_lateral_span":0.,f"{prefix}_longitudinal_span":0.,f"{prefix}_mean_speed":0.,f"{prefix}_mean_abs_lateral_velocity":0.,f"{prefix}_lateral_velocity_variance":0.}
            relative=selected-np.mean(selected,axis=0); lat=relative@lateral; lon=relative@forward; lat_v=selected_v@lateral
            return {f"{prefix}_size":len(selected),f"{prefix}_lateral_variance":float(np.var(lat)),f"{prefix}_lateral_span":float(np.ptp(lat)),f"{prefix}_longitudinal_span":float(np.ptp(lon)),f"{prefix}_mean_speed":float(np.mean(np.linalg.norm(selected_v,axis=1))),f"{prefix}_mean_abs_lateral_velocity":float(np.mean(np.abs(lat_v))),f"{prefix}_lateral_velocity_variance":float(np.var(lat_v))}
        boundary=[]
        for i in range(len(ids)):
            bearings=sorted(math.atan2(positions[j,1]-positions[i,1],positions[j,0]-positions[i,0])%(2*math.pi) for j in neighbors[i])
            gaps=([b-a for a,b in zip(bearings,bearings[1:])]+[bearings[0]+2*math.pi-bearings[-1]]) if len(bearings)>=2 else [2*math.pi]
            if math.degrees(max(gaps))>=BOUNDARY_GAP_DEG: boundary.append(i)
        remaining=set(boundary); components=[]
        while remaining:
            group={remaining.pop()}; queue=list(group)
            while queue:
                current=queue.pop()
                for candidate in tuple(remaining):
                    if np.linalg.norm(positions[current]-positions[candidate])<=SMOOTHING_LENGTH: remaining.remove(candidate); group.add(candidate); queue.append(candidate)
            components.append(group)
        sizes=sorted(map(len,components),reverse=True); ratios=[size/max(len(boundary),1) for size in sizes]; boundary_ids={int(ids[i]) for i in boundary}
        retention=len(boundary_ids&self.previous_boundary)/len(self.previous_boundary) if self.previous_boundary else 1.; self.previous_boundary=boundary_ids
        boundary_positions=positions[boundary] if boundary else np.empty((0,2)); relative=boundary_positions-center if boundary else boundary_positions
        covariance=np.cov(relative.T,bias=True) if len(relative)>1 else np.zeros((2,2)); eigen=np.linalg.eigvalsh(covariance)
        boundary_lat=relative@lateral if len(relative) else np.array([]); boundary_lon=relative@forward if len(relative) else np.array([])
        angles=np.unwrap(np.arctan2(boundary_lat,boundary_lon)) if len(relative) else np.array([])
        degrees=np.array([len(value) for value in neighbors],dtype=float); nearest=[]
        for i in range(len(ids)):
            distances=np.linalg.norm(positions-positions[i],axis=1); distances[i]=np.inf; nearest.append(float(np.min(distances)))
        graph_remaining=set(range(len(ids))); graph_components=[]
        while graph_remaining:
            group={graph_remaining.pop()}; queue=list(group)
            while queue:
                current=queue.pop()
                for peer in neighbors[current]:
                    if int(peer) in graph_remaining: graph_remaining.remove(int(peer)); group.add(int(peer)); queue.append(int(peer))
            graph_components.append(group)
        deviations=[]
        for velocity in velocities[moving]: deviations.append(math.atan2(float(np.dot(velocity,lateral)),float(np.dot(velocity,forward))))
        resultant_length = abs(np.mean(np.exp(1j*np.asarray(deviations)))) if deviations else 1.0
        circular_spread = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(max(float(resultant_length), 1e-12))))) if deviations else 0.0
        result={**cohort("reference_front",reference_mask),**cohort("local_front",local_mask),"local_front_surface_size":len(local_surface),
            "lidar_in_reference_front":bool(reference_mask[int(np.where(ids==observation.lidar_robot_id)[0][0])]),
            "lidar_in_local_front":bool(local_mask[int(np.where(ids==observation.lidar_robot_id)[0][0])]),
            "lidar_longitudinal_front_gap":float(np.max(projections)-projections[int(np.where(ids==observation.lidar_robot_id)[0][0])]),
            "common_forward_x":float(forward[0]),"common_forward_y":float(forward[1]),"motion_bearing_spread":circular_spread,
            "mean_neighbor_degree":float(np.mean(degrees)),"median_neighbor_degree":float(np.median(degrees)),"neighbor_degree_std":float(np.std(degrees)),"mean_nearest_neighbor_distance":float(np.mean(nearest)),
            "neighbor_graph_component_count":len(graph_components),"largest_neighbor_component_fraction":max(map(len,graph_components))/len(ids),
            "boundary_count":len(boundary),"boundary_fraction":len(boundary)/len(ids),"boundary_component_count":len(components),"boundary_largest_component_fraction":sizes[0]/max(len(boundary),1) if sizes else 0.,"boundary_second_component_fraction":sizes[1]/max(len(boundary),1) if len(sizes)>1 else 0.,"boundary_membership_retention":retention,
            "boundary_component_count_vector":sizes,"boundary_component_ratio_vector":ratios,
            "boundary_lateral_span":float(np.ptp(boundary_lat)) if len(boundary_lat) else 0.,"boundary_longitudinal_span":float(np.ptp(boundary_lon)) if len(boundary_lon) else 0.,"boundary_angular_spread":math.degrees(float(np.ptp(angles))) if len(angles) else 0.,"boundary_covariance_trace":float(np.trace(covariance)),"boundary_anisotropy":float((eigen[-1]-eigen[0])/max(eigen.sum(),EPSILON))}
        return result,{"reference_mask":reference_mask,"local_mask":local_mask,"boundary_indices":boundary,"forward":forward}


class CheapLidarDiagnostics:
    """Threshold-free scan summaries; consumes angle/range only."""
    def __init__(self): self.previous=None
    def analyze(self, scan: LidarScan):
        angles,ranges=scan.angles_deg,scan.ranges; left=(angles>=20)&(angles<=160); right=(angles<=-20)&(angles>=-160); forward=np.abs(angles)<=15
        hit=ranges<scan.max_range-1e-9
        discontinuity=np.abs(np.roll(ranges,-1)-ranges); free=~hit
        doubled=np.concatenate([free,free]); longest=current=0
        for value in doubled:
            current=current+1 if value else 0; longest=max(longest,current)
        longest=min(longest,len(free))*360/len(free)
        change=0. if self.previous is None else float(np.mean(np.abs(ranges-self.previous)))
        self.previous=ranges.copy()
        return {"lidar_left_wall_support":float(np.mean(hit[left])),"lidar_right_wall_support":float(np.mean(hit[right])),"lidar_forward_range":float(np.median(ranges[forward])),"lidar_range_discontinuity":float(np.max(discontinuity)),"lidar_range_total_variation":float(np.mean(discontinuity)),"lidar_free_space_angular_span":float(longest),"lidar_range_profile_change":change}


def _circular_mask_sectors(angles_deg: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """Merge adjacent body-relative scan bins, including the -180/180 seam."""
    indices=np.flatnonzero(mask)
    if not len(indices): return []
    if len(indices)==len(mask): return [(float(angles_deg[0]),float(angles_deg[-1]))]
    # Rotate after an unchanged bin, so a seam-crossing run becomes one run.
    start=(int(indices[0])-1)%len(mask)
    while mask[start]: start=(start-1)%len(mask)
    ordered=(np.arange(len(mask))+start+1)%len(mask); sectors=[]; run=[]
    for index in ordered:
        if mask[index]: run.append(int(index))
        elif run:
            sectors.append((float(angles_deg[run[0]]),float(angles_deg[run[-1]]))); run=[]
    if run: sectors.append((float(angles_deg[run[0]]),float(angles_deg[run[-1]])))
    return sectors


class LidarGeometryReasonDiagnostics:
    """Explain frozen detector evidence without changing its runtime contract."""
    def __init__(self, lidar_threshold: float):
        self.bootstrap=[]; self.baseline_profile=None
        # Evaluation-only ray threshold derived from the frozen normalized
        # LiDAR score and sensor range; it does not feed detector decisions.
        self.range_delta_threshold=float(lidar_threshold*LIDAR_MAX_RANGE)

    @staticmethod
    def _format_sectors(sectors: list[tuple[float,float]]) -> str:
        return "["+",".join(f"({start:.0f},{end:.0f})" for start,end in sectors)+"]"

    def analyze(self, scan: LidarScan, row: dict) -> tuple[dict,dict]:
        """Return scalar reasons and masks from body-relative range changes."""
        if self.baseline_profile is None:
            self.bootstrap.append(scan.ranges.copy())
            if len(self.bootstrap)>=3: self.baseline_profile=np.median(np.stack(self.bootstrap),axis=0)
        baseline=(self.baseline_profile if self.baseline_profile is not None else np.median(np.stack(self.bootstrap),axis=0))
        angles=scan.angles_deg; current=scan.ranges; delta=current-baseline
        left=(angles>=20)&(angles<=160); right=(angles<=-20)&(angles>=-160); forward=np.abs(angles)<=15
        base_hit=baseline<scan.max_range-1e-9; now_hit=current<scan.max_range-1e-9
        base_left=float(np.mean(base_hit[left])); base_right=float(np.mean(base_hit[right])); base_forward=float(np.median(baseline[forward]))
        wall_loss=base_hit&(~now_hit)&(delta>self.range_delta_threshold)
        free_increase=(delta>self.range_delta_threshold)&(~wall_loss)
        wall_sectors=_circular_mask_sectors(angles,wall_loss); free_sectors=_circular_mask_sectors(angles,free_increase)
        positive_sectors=_circular_mask_sectors(angles,delta>self.range_delta_threshold)
        largest=max(positive_sectors,key=lambda pair: ((pair[1]-pair[0])%360),default=None)
        free_base=float(row.get("baseline_lidar_free_space_angular_span",row["lidar_free_space_angular_span"]))
        free_now=float(row["lidar_free_space_angular_span"])
        left_now=float(row["lidar_left_wall_support"]); right_now=float(row["lidar_right_wall_support"])
        # L/R and forward baselines below are debug summaries. The frozen
        # detector itself uses current side-wall loss, free-span growth from
        # its baseline, and scan-profile change in one aggregate score.
        result={
            "lidar_left_support_baseline":base_left,"lidar_left_support_current":left_now,"lidar_left_support_delta":left_now-base_left,
            "lidar_right_support_baseline":base_right,"lidar_right_support_current":right_now,"lidar_right_support_delta":right_now-base_right,
            "lidar_free_span_baseline":free_base,"lidar_free_span_current":free_now,"lidar_free_span_delta":free_now-free_base,
            "lidar_forward_baseline":base_forward,"lidar_forward_current":float(row["lidar_forward_range"]),"lidar_forward_delta":float(row["lidar_forward_range"])-base_forward,
            "lidar_left_support_changed":bool(left_now<base_left),"lidar_right_support_changed":bool(right_now<base_right),"lidar_free_span_changed":bool(free_now>free_base),
            "lidar_profile_change_score":float(row.get("lidar_evidence_smoothed",0.0)),"lidar_profile_change_threshold":float(row.get("lidar_threshold",0.0)),"lidar_profile_changed":bool(row.get("shadow_lidar_geometry_change",False)),
            "lidar_range_delta_diagnostic_threshold":self.range_delta_threshold,
            "wall_loss_sectors":self._format_sectors(wall_sectors),"free_space_increase_sectors":self._format_sectors(free_sectors),
            "largest_range_increase_sector":self._format_sectors([largest] if largest is not None else []),
            "lidar_geometry_reason_summary":f"wall_loss={len(wall_sectors)} free_increase={len(free_sectors)}",
        }
        return result,{"lidar_wall_loss_mask":wall_loss,"lidar_free_increase_mask":free_increase}


class GroundTruthEvaluator:
    """Evaluation-only world/geometry consumer."""
    def __init__(self, geometry): self.geometry=geometry
    def evaluate(self, world, runtime, local_mask):
        progress={"gt_mean_forward_progress":float(np.mean([robot.position[1] for robot in world.robots])-world.initial_mean_y),"gt_frontmost_forward_progress":float(max(robot.position[1] for robot in world.robots)-world.initial_front_y)}
        if self.geometry.entrance_y is None: return {"gt_phase":"CORRIDOR_ONLY","gt_frontmost_crossed":False,"gt_reference_front_crossed":False,"gt_local_front_crossed":False,**progress}
        entrance=self.geometry.entrance_y; frontmost=max(robot.position[1] for robot in world.robots)
        ids=np.array([robot.robot_id for robot in world.robots]); positions=np.array([robot.position for robot in world.robots]); forward=np.array([runtime["common_forward_x"],runtime["common_forward_y"]]); center=np.mean(positions,axis=0); projections=(positions-center)@forward; ref=positions[projections>=np.quantile(projections,REFERENCE_FRONT_QUANTILE)]
        local_positions=positions[local_mask]
        local_center_y=float(np.mean(local_positions[:,1])) if len(local_positions) else float("nan")
        ref_y=float(np.mean(ref[:,1]))
        approach=self.geometry.incoming_width/2
        if frontmost<entrance-approach: phase="CORRIDOR"
        elif frontmost<entrance: phase="OPENING_APPROACH"
        elif ref_y<entrance: phase="BOUNDARY_CROSSING"
        else: phase="JUNCTION_REGION"
        return {"gt_phase":phase,"gt_frontmost_crossed":frontmost>=entrance,"gt_reference_front_crossed":ref_y>=entrance,"gt_local_front_crossed":bool(math.isfinite(local_center_y) and local_center_y>=entrance),**progress}


class SimulationRunner:
    def __init__(self, case_id, propulsion_mode="production_compression", shadow_detector=None, hold_on_suspect=False, profile_detector=None):
        self.geometry=GeometryBuilder.build(case_id); self.world=SimulatorWorld(self.geometry,propulsion_mode); self.swarm=SwarmDiagnostics(); self.lidar=CheapLidarDiagnostics(); self.gt=GroundTruthEvaluator(self.geometry); self.rows=[]; self.last_visual=None; self.shadow_detector=shadow_detector; self.hold_on_suspect=hold_on_suspect
        self.lidar_reason=(LidarGeometryReasonDiagnostics(shadow_detector.thresholds.lidar) if shadow_detector is not None else None)
        self.profile_detector=profile_detector; self.last_profile_result=None
    def step(self, frame):
        self.world.step()
        if frame%max(1,round(SAMPLE_PERIOD/DT)): return None
        observation=LocalObservationBuilder.build(self.world); swarm,visual=self.swarm.analyze(observation); lidar=self.lidar.analyze(observation.lidar_scan)
        row={"map_case":self.geometry.case_id,"propulsion_mode":self.world.propulsion_mode,"frame":frame,"timestamp":self.world.time,"initialization_phase":self.world.initialization_phase(),"lidar_robot_id":self.world.lidar_robot_id,"lidar_initial_x":float(self.world.initial_lidar_position[0]),"lidar_initial_y":float(self.world.initial_lidar_position[1]),"lidar_initial_front_center_offset":float(self.world.initial_lidar_position[0]-self.world.initial_front_center_x),**swarm,**lidar,**self.world.sanity(),**self.world.connectivity_diagnostics()}
        if self.profile_detector is not None:
            profile=self.profile_detector.detect(observation.lidar_scan.angles_deg,observation.lidar_scan.ranges); self.last_profile_result=profile
            groups=profile["opening_groups"]
            row.update({key:profile[key] for key in ("profile_detector_state","profile_junction_detected","opening_group_count","corridor_width","profile_max_range","profile_lateral_offset","profile_numerical_margin","profile_max_abs_valid_delta","measured_profile_min","measured_profile_mean","measured_profile_max","expected_profile_min","expected_profile_mean","expected_profile_max")})
            row["opening_groups_json"]=json.dumps(groups,separators=(",",":")); visual.update({"profile_open_candidate_mask":profile["open_candidate_mask"],"profile_confirmed_opening_mask":profile["confirmed_opening_mask"],"profile_expected_ranges":profile["expected_ranges"]})
        # Everything above this line is runtime-local. GT is appended only
        # after the profile detector has completed its decision.
        row.update(self.gt.evaluate(self.world,row,visual["local_mask"]))
        if self.shadow_detector is not None:
            from junction_detection.integration.run_junction_shadow_detection import runtime_features
            detected=self.shadow_detector.update(float(row["timestamp"]),runtime_features(row)); row.update(detected)
            if self.hold_on_suspect and detected["shadow_junction_suspected"]: self.world.activate_suspect_hold()
            row["suspect_hold_active"]=self.world.suspect_hold_active; row["local_forward_propulsion_active"]=self.world.propulsion_mode=="local_forward" and not self.world.suspect_hold_active; row["initialization_phase"]=self.world.initialization_phase()
            reasons,masks=self.lidar_reason.analyze(observation.lidar_scan,row); row.update(reasons); visual.update(masks)
        self.rows.append(row); self.last_visual=(observation,visual); return row


def run_headless(case_id,frames,propulsion_mode="production_compression"):
    runner=SimulationRunner(case_id,propulsion_mode)
    for frame in range(frames): runner.step(frame)
    return runner


def save_case(runner,output):
    if not runner.rows:
        return
    folder=output/runner.geometry.case_id
    if runner.world.propulsion_mode != "production_compression":
        folder=output/runner.world.propulsion_mode/runner.geometry.case_id
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/"pre_exploration_timeline.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(runner.rows[0])); writer.writeheader(); writer.writerows(runner.rows)


class PygameRenderer:
    def __init__(self, geometry, gui_scale=DEFAULT_GUI_SCALE, show_trails=False, show_gt=False):
        import pygame
        pygame.init(); self.pygame=pygame; self.screen=pygame.display.set_mode((1100,800)); self.clock=pygame.time.Clock(); self.font=pygame.font.Font(None,22)
        self.paused=False; self.show_gt=show_gt; self.show_diagnostics=True; self.show_lidar=False; self.show_trails=show_trails; self.show_communication_links=True; self.show_support_links=True; self.trails=[]; self.gui_scale=gui_scale
        self.configure_camera(geometry)
    def configure_camera(self,geometry):
        vertices=np.array([point for rect in geometry.free_rects for point in rect.vertices],dtype=float)
        minimum=np.min(vertices,axis=0); maximum=np.max(vertices,axis=0); span=np.maximum(maximum-minimum,1.0)
        self.camera_center=0.5*(minimum+maximum)
        self.pixels_per_world=self.gui_scale*min(1020.0/(span[0]+40.0),700.0/(span[1]+40.0))
    def world_to_screen(self,point):
        relative=np.asarray(point)-self.camera_center
        return int(550+relative[0]*self.pixels_per_world),int(430-relative[1]*self.pixels_per_world)
    def draw(self,runner,frame):
        pygame=self.pygame; self.screen.fill((18,22,28))
        for rect in runner.geometry.free_rects: pygame.draw.polygon(self.screen,(48,56,65),[self.world_to_screen(point) for point in rect.vertices])
        for a,b in runner.geometry.walls: pygame.draw.line(self.screen,(230,230,230),self.world_to_screen(a),self.world_to_screen(b),3)
        if self.show_gt and runner.geometry.entrance_y is not None:
            y=runner.geometry.entrance_y; pygame.draw.line(self.screen,(255,190,50),self.world_to_screen((-42,y)),self.world_to_screen((42,y)),2)
        visual=runner.last_visual; reference=local=boundary=set(); forward=np.array([0.,1.])
        if visual:
            observation,data=visual; reference=set(observation.robot_ids[data["reference_mask"]]); local=set(observation.robot_ids[data["local_mask"]]); boundary={int(observation.robot_ids[i]) for i in data["boundary_indices"]}; forward=data["forward"]
        if runner.world.propulsion_mode == "local_forward":
            # Quantile/reference cohorts are retained in CSV only as historical
            # diagnostics; LOCAL_FORWARD GUI uses local graph evidence instead.
            reference=local=set()
        lidar=next(robot for robot in runner.world.robots if robot.robot_id==runner.world.lidar_robot_id)
        by_id={robot.robot_id:robot for robot in runner.world.robots}
        if self.show_support_links:
            for start_id,end_id in runner.world.last_support_debug_edges:
                pygame.draw.line(self.screen,(75,115,165),self.world_to_screen(by_id[start_id].position),self.world_to_screen(by_id[end_id].position),1)
        if self.show_communication_links:
            for start_id,end_id in runner.world.last_communication_edges:
                pygame.draw.line(self.screen,(100,210,115),self.world_to_screen(by_id[start_id].position),self.world_to_screen(by_id[end_id].position),1)
        if self.show_trails:
            self.trails.append(lidar.position.copy()); self.trails[:]=self.trails[-240:]
            if len(self.trails)>1: pygame.draw.lines(self.screen,(110,180,255),False,[self.world_to_screen(point) for point in self.trails],2)
        for robot in runner.world.robots:
            color=(75,135,225)
            if robot.robot_id in reference: color=(80,200,115)
            if robot.robot_id in local: color=(75,230,210)
            if robot.robot_id in boundary: color=(255,100,100)
            if robot.robot_id==runner.world.lidar_robot_id: color=(255,230,40)
            pygame.draw.circle(self.screen,color,self.world_to_screen(robot.position),3)
        pygame.draw.line(self.screen,(80,230,255),self.world_to_screen(lidar.position),self.world_to_screen(lidar.position+forward*18),3)
        if self.show_lidar and visual:
            scan=visual[0].lidar_scan
            wall_loss=visual[1].get("lidar_wall_loss_mask",np.zeros(len(scan.ranges),dtype=bool)); free_increase=visual[1].get("lidar_free_increase_mask",np.zeros(len(scan.ranges),dtype=bool))
            profile_candidate=visual[1].get("profile_open_candidate_mask",np.zeros(len(scan.ranges),dtype=bool)); profile_confirmed=visual[1].get("profile_confirmed_opening_mask",np.zeros(len(scan.ranges),dtype=bool))
            for index in range(0,len(scan.ranges),4):
                angle=scan.angles_deg[index]; range_value=scan.ranges[index]
                radians=math.radians(runner.world.lidar_yaw_deg+angle); endpoint=lidar.position+np.array([math.cos(radians),math.sin(radians)])*range_value
                color=((235,70,235) if profile_confirmed[index] else ((245,165,45) if profile_candidate[index] else ((235,90,55) if wall_loss[index] else ((55,225,235) if free_increase[index] else (90,90,55)))))
                emphasized=profile_candidate[index] or wall_loss[index] or free_increase[index]
                pygame.draw.line(self.screen,color,self.world_to_screen(lidar.position),self.world_to_screen(endpoint),2 if emphasized else 1)
        run_state="PAUSED" if self.paused else "RUNNING"
        latest=runner.rows[-1] if runner.rows else {}; hud=[f"{run_state} | Map {runner.geometry.case_id} mode={runner.world.propulsion_mode} frame={frame} t={runner.world.time:.2f}",f"GT phase={latest.get('gt_phase','-')} EVAL ONLY",f"LiDAR robot={runner.world.lidar_robot_id} initial offset={runner.world.initial_lidar_position[0]-runner.world.initial_front_center_x:.2f}",f"GUI scale={self.gui_scale:.2f} comm_links={self.show_communication_links} support_links={self.show_support_links}","SPACE pause/resume R reset 1-6 map G GT D diagnostics L rays T trails C comm N support ESC quit"]
        if "profile_detector_state" in latest:
            hud += [f"[LiDAR Profile Detector] mode=GEOMETRY_BASED state={latest['profile_detector_state']}",f"width={latest['corridor_width']:.1f} local offset={latest['profile_lateral_offset']:+.2f} max_range={latest['profile_max_range']:.1f} groups={latest['opening_group_count']} Junction={latest['profile_junction_detected']}"]
        if self.show_diagnostics and latest:
            second_size=round(latest['boundary_count']*latest['boundary_second_component_fraction'])
            counts=list(latest.get('boundary_component_count_vector',[])); ratios=list(latest.get('boundary_component_ratio_vector',[])); suffix="..." if len(counts)>5 else ""
            counts_text=f"{counts[:5]}{suffix}"; ratios_text="["+",".join(f"{value:.3f}" for value in ratios[:5])+"]"+suffix
            spread_label=(f"local surface-pack span={latest['local_front_lateral_span']:.2f}" if runner.world.propulsion_mode=="local_forward" else f"local/reference front={latest['local_front_size']}/{latest['reference_front_size']} lateral span={latest['local_front_lateral_span']:.2f}")
            hud += [f"init phase={latest['initialization_phase']}",f"min distance={latest['min_inter_robot_distance']:.3f} overlap pairs={latest['overlap_pair_count']} max speed={latest['max_speed']:.2f}",f"wall contacts={latest['wall_contact_count']} projection corrections={latest['wall_projection_correction_count']}",f"[SPH] {spread_label} motion spread={latest['motion_bearing_spread']:.2f}",f"[Boundary] count={latest['boundary_count']} comps={latest['boundary_component_count']} largest={latest['boundary_largest_component_fraction']:.2f} second={second_size}",f"B-counts={counts_text} B-ratios={ratios_text}"]
            if runner.world.propulsion_mode=="local_forward":
                hud += [f"Leader connected={latest['leader_connected']} comp size={latest['leader_connected_component_size']} gap={latest['leader_to_front_pack_gap']:.2f}",f"Nearest follower={latest['leader_to_nearest_follower_distance']:.2f} disconnected={latest['disconnected_count']} comm edges={latest['communication_edge_count']}"]
        if self.show_diagnostics and "shadow_junction_suspected" in latest:
            detector_state="JUNCTION_SUSPECTED" if latest.get("suspect_hold_active",False) or latest["shadow_junction_suspected"] else "NO_JUNCTION"
            hud += [f"[Legacy Shadow EVAL ONLY] {detector_state} Fusion={int(latest['shadow_fusion_trigger'])} SPH={int(latest['shadow_sph_expansion'])} Boundary={int(latest['shadow_boundary_change'])}",
                f"[LiDAR Geometry] Trigger={int(latest['shadow_lidar_geometry_change'])} {latest.get('lidar_geometry_reason_summary','-')}",
                f"L support {latest.get('lidar_left_support_baseline',0):.2f}->{latest.get('lidar_left_support_current',0):.2f} d={latest.get('lidar_left_support_delta',0):+.2f} changed={latest.get('lidar_left_support_changed',False)}",
                f"R support {latest.get('lidar_right_support_baseline',0):.2f}->{latest.get('lidar_right_support_current',0):.2f} d={latest.get('lidar_right_support_delta',0):+.2f} changed={latest.get('lidar_right_support_changed',False)}",
                f"Free span {latest.get('lidar_free_span_baseline',0):.1f}->{latest.get('lidar_free_span_current',0):.1f} d={latest.get('lidar_free_span_delta',0):+.1f} changed={latest.get('lidar_free_span_changed',False)}",
                f"Forward {latest.get('lidar_forward_baseline',0):.1f}->{latest.get('lidar_forward_current',0):.1f} d={latest.get('lidar_forward_delta',0):+.1f}",
                f"LiDAR evidence={latest.get('lidar_profile_change_score',0):.4f} threshold={latest.get('lidar_profile_change_threshold',0):.4f} changed={latest.get('lidar_profile_changed',False)}",
                f"Sectors wall={latest.get('wall_loss_sectors','[]')} free={latest.get('free_space_increase_sectors','[]')}",
                f"[Anchor] phase={latest['initialization_phase']} hold={latest.get('suspect_hold_active',False)} propulsion={latest.get('local_forward_propulsion_active',False)} braking={latest.get('leader_braking_active',False)}",
                f"leader speed={latest.get('leader_speed',0):.3f} stationary={latest.get('leader_stationary',False)} dwell={latest.get('stationary_dwell_steps',0)}/{latest.get('stationary_dwell_target',0)}",
                f"Anchor={latest.get('provisional_fixed_anchor',False)} PointCloud ready={latest.get('pointcloud_ready',False)}"]
        if self.show_diagnostics and "opening_groups_json" in latest:
            groups=json.loads(latest["opening_groups_json"])
            hud += ["[Profile groups] "+("none" if not groups else " | ".join(f"G{group['group_id']}=[{group['start_angle_deg']:.1f},{group['end_angle_deg']:.1f}] c={group['center_angle_deg']:.1f} w={group['angular_width_deg']:.1f}" for group in groups[:4]))]
        for index,text in enumerate(hud): self.screen.blit(self.font.render(text,True,(245,245,245)),(12,10+index*22))
        pygame.display.flip(); self.clock.tick(60)


def _new_gui_runner(case_id, propulsion_mode):
    """Create a GUI runner with the moving LiDAR-profile detector."""
    if propulsion_mode == "local_forward":
        from junction_detection.integration.run_junction_shadow_detection import FROZEN_GUI_THRESHOLDS, ShadowJunctionDetector
        from junction_detection.pointcloud.lidar_profile_junction_detector import GeometryProfileConfig, LidarProfileJunctionDetector
        profile=LidarProfileJunctionDetector(GeometryProfileConfig(BASELINE_CORRIDOR_WIDTH,LIDAR_MAX_RANGE))
        # Legacy fusion remains visible for evaluation, but no longer actuates
        # hold/braking or supplies the new detector's decision.
        return SimulationRunner(case_id,propulsion_mode,ShadowJunctionDetector(FROZEN_GUI_THRESHOLDS),hold_on_suspect=False,profile_detector=profile)
    return SimulationRunner(case_id,propulsion_mode)


def _advance_gui_frame(runner, renderer, frame):
    """Advance exactly one physics frame unless SPACE has paused the GUI."""
    if renderer.paused:
        return frame
    runner.step(frame)
    return frame+1


def run_gui(case_id,frames,show_trails=False,show_gt=False,propulsion_mode="production_compression",gui_scale=DEFAULT_GUI_SCALE):
    import pygame
    cases=list(CASES); current=cases.index(case_id); runner=_new_gui_runner(case_id,propulsion_mode); renderer=PygameRenderer(runner.geometry,gui_scale,show_trails,show_gt); frame=0; running=True
    while running and (frames<=0 or frame<frames):
        for event in pygame.event.get():
            if event.type==pygame.QUIT: running=False
            elif event.type==pygame.KEYDOWN:
                if event.key==pygame.K_ESCAPE: running=False
                elif event.key==pygame.K_SPACE: renderer.paused=not renderer.paused
                elif event.key==pygame.K_r: runner=_new_gui_runner(cases[current],propulsion_mode); frame=0; renderer.trails=[]
                elif event.key==pygame.K_g: renderer.show_gt=not renderer.show_gt
                elif event.key==pygame.K_d: renderer.show_diagnostics=not renderer.show_diagnostics
                elif event.key==pygame.K_l: renderer.show_lidar=not renderer.show_lidar
                elif event.key==pygame.K_t: renderer.show_trails=not renderer.show_trails
                elif event.key==pygame.K_c: renderer.show_communication_links=not renderer.show_communication_links
                elif event.key==pygame.K_n: renderer.show_support_links=not renderer.show_support_links
                elif pygame.K_1<=event.key<=pygame.K_6: current=event.key-pygame.K_1; runner=_new_gui_runner(cases[current],propulsion_mode); renderer.configure_camera(runner.geometry); frame=0; renderer.trails=[]
        frame=_advance_gui_frame(runner,renderer,frame)
        renderer.draw(runner,frame)
    pygame.quit(); return runner


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--map-case",choices=CASES); parser.add_argument("--frames",type=int,default=600)
    parser.add_argument("--propulsion-mode",choices=PROPULSION_MODES,default="production_compression")
    parser.add_argument("--gui-scale",type=float,default=DEFAULT_GUI_SCALE)
    modes=parser.add_mutually_exclusive_group(); modes.add_argument("--gui",action="store_true"); modes.add_argument("--headless",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=Path(os.environ.get("PRE_EXPLORATION_OUTPUT",DEFAULT_OUTPUT))); parser.add_argument("--show-trails",action="store_true"); parser.add_argument("--show-gt",action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    if args.gui:
        runner=run_gui(args.map_case or "M1_CROSS_BASELINE",args.frames,args.show_trails,args.show_gt,args.propulsion_mode,args.gui_scale); save_case(runner,args.output_dir)
    else:
        case=args.map_case or "M1_CROSS_BASELINE"; runner=run_headless(case,args.frames,args.propulsion_mode); save_case(runner,args.output_dir); print(f"case={case} propulsion_mode={args.propulsion_mode} rows={len(runner.rows)} output={args.output_dir.resolve()}")


if __name__=="__main__": main()
