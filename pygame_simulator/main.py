"""SPH-based single-junction Physical DFS prototype.

Implemented research components
-------------------------------
1. Multi-criteria Junction Anchor election.
2. Cost-guided DFS child-branch ordering for the current junction.
3. Dead-end saturation detection using speed, density, occupancy,
   front stagnation, and dwell time.
4. Width-adaptive Shepherd count, scored candidate election, and
   minimum-cost candidate-to-slot assignment.
5. Moving piston-style Shepherd boundary plus weak directional body force.
6. Fixed Base-rooted LOS communication with permanent trunk relays.
7. The original dead-end first-arrival Shepherd selection timing is retained,
   but the Shepherd count is computed from corridor width.
8. Pressure starts only after the ordinary robots saturate behind the formed
   Shepherd boundary; branch relays, release logic, and final gathering remain.

Scope limitation
----------------
The map contains one T/cross junction. The code therefore implements DFS
child ordering at one junction, not recursive multi-junction DFS-tree repair.
The data structures are intentionally separated so they can be extended to a
multi-junction topological graph later.
"""

from __future__ import annotations

import csv
import heapq
import math
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import pygame


pygame.init()

# =========================================================
# 1. Display
# =========================================================

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
FPS = 60
SUBSTEPS = 1

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(
    "Base-Connected SPH DFS | Adaptive Shepherd + Saturation Piston"
)
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)
small_font = pygame.font.SysFont(None, 19)

BACKGROUND_COLOR = (248, 249, 252)
FLOOR_COLOR = (235, 239, 246)
WALL_COLOR = (96, 106, 124)
TEXT_COLOR = (58, 67, 82)
ROBOT_BASE_COLOR = (115, 165, 208)
SHEPHERD_COLOR = (171, 145, 205)
ANCHOR_COLOR = (102, 174, 137)
BASE_COLOR = (73, 94, 128)
TRUNK_RELAY_COLOR = (201, 123, 85)
RELAY_COLOR = (226, 151, 82)
RELAY_SLOT_COLOR = (196, 129, 65)
JUNCTION_COLOR = (153, 164, 181)
END_REGION_COLOR = (224, 171, 115)
COMM_LINK_SAFE_COLOR = (132, 190, 158)
COMM_LINK_WARNING_COLOR = (226, 177, 96)
COMM_LINK_DANGER_COLOR = (214, 103, 103)
DISCONNECTED_COLOR = (205, 96, 96)

# =========================================================
# 2. Cross map
# =========================================================

center_x = 400
center_y = 350
corridor_width = 120
half_width = corridor_width // 2
normal_length = 180
right_length = normal_length * 2

cross_points = [
    (center_x - half_width, center_y - half_width - normal_length),
    (center_x + half_width, center_y - half_width - normal_length),
    (center_x + half_width, center_y - half_width),
    (center_x + half_width + right_length, center_y - half_width),
    (center_x + half_width + right_length, center_y + half_width),
    (center_x + half_width, center_y + half_width),
    (center_x + half_width, center_y + half_width + normal_length),
    (center_x - half_width, center_y + half_width + normal_length),
    (center_x - half_width, center_y + half_width),
    (center_x - half_width - normal_length, center_y + half_width),
    (center_x - half_width - normal_length, center_y - half_width),
    (center_x - half_width, center_y - half_width),
]

junction_rect = pygame.Rect(
    center_x - half_width,
    center_y - half_width,
    corridor_width,
    corridor_width,
)

ANCHOR_REGION_SIZE = 70
anchor_election_rect = pygame.Rect(
    center_x - ANCHOR_REGION_SIZE // 2,
    center_y - ANCHOR_REGION_SIZE // 2,
    ANCHOR_REGION_SIZE,
    ANCHOR_REGION_SIZE,
)

ANCHOR_PARK_POSITION = pygame.Vector2(center_x - 25, center_y - 25)
# Fixed communication root. It remains at the lower entrance throughout exploration.
BASE_POSITION = pygame.Vector2(
    center_x - 25,
    center_y + half_width + normal_length - 14,
)
JUNCTION_STAGING_POSITION = pygame.Vector2(center_x + 10, center_y + 10)

up_rect = pygame.Rect(
    center_x - half_width,
    center_y - half_width - normal_length,
    corridor_width,
    normal_length,
)
left_rect = pygame.Rect(
    center_x - half_width - normal_length,
    center_y - half_width,
    normal_length,
    corridor_width,
)
right_rect = pygame.Rect(
    center_x + half_width,
    center_y - half_width,
    right_length,
    corridor_width,
)
bottom_rect = pygame.Rect(
    center_x - half_width,
    center_y + half_width,
    corridor_width,
    normal_length,
)

END_REGION_DEPTH = 48

dead_end_regions = {
    "UP": pygame.Rect(
        center_x - half_width,
        center_y - half_width - normal_length,
        corridor_width,
        END_REGION_DEPTH,
    ),
    "LEFT": pygame.Rect(
        center_x - half_width - normal_length,
        center_y - half_width,
        END_REGION_DEPTH,
        corridor_width,
    ),
    "RIGHT": pygame.Rect(
        center_x + half_width + right_length - END_REGION_DEPTH,
        center_y - half_width,
        END_REGION_DEPTH,
        corridor_width,
    ),
}

# Keep the original Shepherd election timing: Shepherds are selected only after
# enough leading NORMAL robots enter this dead-end capture region.
EARLY_CAPTURE_DEPTH = 34

early_capture_regions = {
    "UP": pygame.Rect(
        center_x - half_width,
        center_y - half_width - normal_length,
        corridor_width,
        EARLY_CAPTURE_DEPTH,
    ),
    "LEFT": pygame.Rect(
        center_x - half_width - normal_length,
        center_y - half_width,
        EARLY_CAPTURE_DEPTH,
        corridor_width,
    ),
    "RIGHT": pygame.Rect(
        center_x + half_width + right_length - EARLY_CAPTURE_DEPTH,
        center_y - half_width,
        EARLY_CAPTURE_DEPTH,
        corridor_width,
    ),
}

# =========================================================
# 3. State and branch metadata
# =========================================================


class SimulationPhase(Enum):
    MOVE_TO_JUNCTION = auto()
    EXPLORE_BRANCH = auto()
    FORM_SHEPHERD_BOUNDARY = auto()
    FILL_BEHIND_SHEPHERD = auto()
    PRESSURE_PUSH = auto()
    FLOW_BACKTRACK = auto()
    JUNCTION_SWITCH = auto()
    FINAL_JUNCTION_GATHER = auto()
    RETURN_TO_BASE = auto()
    DONE = auto()


BRANCHES = ("UP", "LEFT", "RIGHT")
BRANCH_DIRECTIONS = {
    "UP": pygame.Vector2(0.0, -1.0),
    "LEFT": pygame.Vector2(-1.0, 0.0),
    "RIGHT": pygame.Vector2(1.0, 0.0),
}
BRANCH_LENGTHS = {
    "UP": float(normal_length),
    "LEFT": float(normal_length),
    "RIGHT": float(right_length),
}

# Current cross map has one terminal target per branch. The function that
# computes C_loss still removes the branch edge and counts unreachable targets,
# so the same code can be reused when the graph is expanded.
TOPOLOGY_ADJACENCY = {
    "BASE": {"JUNCTION"},
    "JUNCTION": {"BASE", "UP_TARGET", "LEFT_TARGET", "RIGHT_TARGET"},
    "UP_TARGET": {"JUNCTION"},
    "LEFT_TARGET": {"JUNCTION"},
    "RIGHT_TARGET": {"JUNCTION"},
}
BRANCH_TARGET_NODE = {
    "UP": "UP_TARGET",
    "LEFT": "LEFT_TARGET",
    "RIGHT": "RIGHT_TARGET",
}

phase = SimulationPhase.MOVE_TO_JUNCTION
active_branch = "UP"
branch_states = {branch: "UNVISITED" for branch in BRANCHES}
branch_order_plan: list[str] = []
previous_branch_direction = pygame.Vector2(0.0, -1.0)  # incoming from BASE

junction_anchor: Optional["Robot"] = None
simulation_time = 0.0
junction_switch_timer = 0.0
final_gather_timer = 0.0
shepherd_form_timer = 0.0
pressure_push_timer = 0.0
flow_establish_timer = 0.0

communication_sequence = 0
last_message_signature = None
relay_slots: list[dict] = []
relay_deploy_cooldown = 0.0
relay_retract_cooldown = 0.0
relay_retract_clear_timer = 0.0
relay_motion_scale = 1.0
trunk_relay_slots: list[dict] = []
trunk_relay_deploy_cooldown = 0.0
base_station: Optional["BaseStation"] = None

# =========================================================
# 4. Physics and control parameters
# =========================================================

ROBOT_COUNT = 220
SPAWN_MODE = "grid"
ROBOT_RADIUS = 2
GRID_SPACING = 7

SMOOTHING_LENGTH = 28.0
PRESSURE_GAIN = 1650.0
STIFFNESS_EXPONENT = 0.5
VISCOSITY_XI1 = 0.9
VISCOSITY_XI2 = 1.2
MOTION_SPEED_MULTIPLIER = 2.0
DAMPING = 2.3
SAFE_RADIUS = 7.5
REPULSION_GAIN = 260.0

ROUTE_FORCE = 52.0 * MOTION_SPEED_MULTIPLIER
OUTLET_FORCE = 44.0 * MOTION_SPEED_MULTIPLIER
FLOW_BACKTRACK_FORCE = 46.0 * MOTION_SPEED_MULTIPLIER
FINAL_GATHER_FORCE = 58.0 * MOTION_SPEED_MULTIPLIER
PRESSURE_BACKTRACK_BODY_FORCE = 12.0 * MOTION_SPEED_MULTIPLIER
CENTERING_GAIN = 1.2

MAX_SPEED = 78.0 * MOTION_SPEED_MULTIPLIER
MAX_ACCELERATION = 520.0 * MOTION_SPEED_MULTIPLIER
EPSILON = 1e-8

INITIAL_INGRESS_FORCE = 44.0 * MOTION_SPEED_MULTIPLIER
INITIAL_INGRESS_LANE_GAIN = 1.0
INITIAL_INGRESS_LANE_MAX_FORCE = 20.0
INITIAL_INGRESS_TARGET_Y = center_y + 10.0
INITIAL_INGRESS_BRAKE_DISTANCE = 34.0
INITIAL_INGRESS_MIN_FORCE_SCALE = 0.18
INITIAL_INGRESS_MAX_DT = 0.04
NORMAL_PHYSICS_MAX_DT = 0.05

ISOLATION_NEIGHBOR_THRESHOLD = 4
ISOLATION_ROUTE_BOOST = 1.1
LOCAL_COHESION_GAIN = 20.0

JUNCTION_ENTRY_COUNT = 18
ANCHOR_MOVE_SPEED = 42.0
ANCHOR_POSITION_TOLERANCE = 2.5
JUNCTION_SWITCH_COUNT = 18
JUNCTION_SWITCH_DWELL_TIME = 0.25
RETURN_BOTTOM_TARGET_COUNT = ROBOT_COUNT
FINAL_GATHER_DWELL_TIME = 0.55

# Multi-criteria Anchor election
ANCHOR_ELECTION_MIN_CANDIDATES = 4
ANCHOR_ELECTION_WAIT_TIME = 0.22
ANCHOR_WEIGHT_ARRIVAL = 0.30
ANCHOR_WEIGHT_PARKING = 0.20
ANCHOR_WEIGHT_DIRECTION = 0.15
ANCHOR_WEIGHT_COMMUNICATION = 0.35
ANCHOR_LOCAL_COMM_RANGE = 56.0

# Dead-end saturation
SATURATION_MIN_TIP_ROBOTS = 18
SATURATION_LOW_SPEED_THRESHOLD = 4.0
SATURATION_LOW_SPEED_RATIO = 0.65
SATURATION_DENSITY_RATIO = 1.02
SATURATION_OCCUPANCY_RATIO = 0.16
SATURATION_FRONT_WINDOW = 0.35
SATURATION_FRONT_PROGRESS_EPSILON = 2.2
SATURATION_DWELL_TIME = 0.32
SATURATION_CELL_SIZE = 8.0

# Width-adaptive Shepherd count. Selection timing remains the same as the
# original code: leading robots are selected inside early_capture_regions.
SHEPHERD_BOUNDARY_WALL_CLEARANCE = 14.0
SHEPHERD_FILL_GAP = 38.0
SHEPHERD_FILL_REGION_DEPTH = 62.0
SHEPHERD_MIN_COUNT = 5
SHEPHERD_MAX_COUNT = 14
SHEPHERD_EDGE_MARGIN = 12.0
SHEPHERD_TARGET_SLOT_SPACING = 12.5

SHEPHERD_FORM_SPEED = 110.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_RELEASE_SPEED = 18.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_FORM_TOLERANCE = 3.0
SHEPHERD_FORM_TIMEOUT = 2.4

# Piston motion: Shepherd boundary advances toward the parent junction.
SHEPHERD_PISTON_SPEED = 10.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_PISTON_MAX_TRAVEL = 24.0
SHEPHERD_PRESSURE_FACTOR = 5.2
VIRTUAL_PRESSURE_RADIUS = 60.0
VIRTUAL_PRESSURE_FORCE = 135.0
PRESSURE_RAMP_TIME = 0.8

SHEPHERD_LOCAL_FLOW_DEPTH = 58.0
SHEPHERD_LOCAL_FLOW_FORWARD_ALLOWANCE = 6.0
SHEPHERD_MIN_PUSH_TIME = 0.20
FLOW_SPEED_THRESHOLD = 1.5
FLOW_RATIO_THRESHOLD = 0.45
FLOW_AVERAGE_SPEED_THRESHOLD = 1.8
FLOW_ESTABLISH_DWELL_TIME = 0.12
FLOW_MIN_NORMAL_COUNT = 6
FLOW_FALLBACK_TIME = 1.25
BRANCH_CLEAR_LIMIT = 1

# Communication
COMM_RANGE = 46.0
COMM_LOS_SAMPLE_SPACING = 6.0
COMM_LOS_CLEARANCE = 0.0
COMM_SAFE_DISTANCE = 34.0
COMM_BARRIER_START = COMM_RANGE * 0.84
COMM_RECOVERY_RANGE = 84.0
COMM_RECOVERY_GAIN = 2.2
SHOW_COMM_LINKS_DEFAULT = True
COMM_UPDATE_INTERVAL_FRAMES = 3
ANCHOR_LINK_WARNING_DISTANCE = COMM_SAFE_DISTANCE * 0.82
ANCHOR_LINK_STOP_DISTANCE = COMM_RANGE * 0.90
ANCHOR_MIN_DIRECT_NEIGHBORS = 1
ANCHOR_READY_DIRECT_NEIGHBORS = 1

# Permanent Base-to-Junction trunk relays
TRUNK_RELAY_SPACING = 30.0
TRUNK_RELAY_SELECTION_RADIUS = 50.0
TRUNK_RELAY_DEPLOY_LOOKAHEAD = 12.0
TRUNK_RELAY_DEPLOY_COOLDOWN = 0.08

# Branch relay
RELAY_SPACING = 30.0
RELAY_DEPLOY_LOOKAHEAD = 12.0
RELAY_SELECTION_RADIUS = 52.0
RELAY_END_CLEARANCE = 24.0
RELAY_LANE_MARGIN = 22.0
RELAY_MOVE_SPEED = 125.0 * MOTION_SPEED_MULTIPLIER
RELAY_POSITION_TOLERANCE = 2.5
RELAY_DEPLOY_COOLDOWN = 0.10
RELAY_PASS_MARGIN = 6.0
RELAY_RETRACT_DWELL_TIME = 0.40
RELAY_RETRACT_COOLDOWN = 0.45
RELAY_RELEASE_SPEED = 16.0 * MOTION_SPEED_MULTIPLIER
RELAY_FORMING_SPEED_SCALE = 0.40
RELAY_WAIT_SPEED_SCALE = 0.18
RELAY_DEPLOY_MARGIN = 5.0
RELAY_FRONT_FRACTION = 0.20
RELAY_FRONT_MIN_COUNT = 10
RELAY_FRONT_REQUIRED_CONNECTED_RATIO = 0.90

# Cost-guided branch ordering
BRANCH_COST_LENGTH_WEIGHT = 0.22
BRANCH_COST_RELAY_WEIGHT = 0.20
BRANCH_COST_BACKTRACK_WEIGHT = 0.22
BRANCH_COST_COMM_WEIGHT = 0.16
BRANCH_COST_SWITCH_WEIGHT = 0.10
BRANCH_COST_LOSS_PRIORITY_WEIGHT = 0.10

CELL_SIZE = max(SMOOTHING_LENGTH, VIRTUAL_PRESSURE_RADIUS, COMM_RANGE)

# =========================================================
# 5. Map mask and region checks
# =========================================================

floor_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
floor_surface.fill((0, 0, 0, 0))
pygame.draw.polygon(floor_surface, (255, 255, 255, 255), cross_points)
walkable_mask = pygame.mask.from_surface(floor_surface)


def get_robot_region(position: pygame.Vector2) -> str:
    point = (int(position.x), int(position.y))
    if junction_rect.collidepoint(point):
        return "JUNCTION"
    if up_rect.collidepoint(point):
        return "UP"
    if left_rect.collidepoint(point):
        return "LEFT"
    if right_rect.collidepoint(point):
        return "RIGHT"
    if bottom_rect.collidepoint(point):
        return "BOTTOM"
    return "OUTSIDE"


def is_region_allowed(position: pygame.Vector2) -> bool:
    region = get_robot_region(position)
    if phase == SimulationPhase.MOVE_TO_JUNCTION:
        return region in {"BOTTOM", "JUNCTION"}
    if phase in {
        SimulationPhase.EXPLORE_BRANCH,
        SimulationPhase.FORM_SHEPHERD_BOUNDARY,
        SimulationPhase.FILL_BEHIND_SHEPHERD,
        SimulationPhase.PRESSURE_PUSH,
        SimulationPhase.FLOW_BACKTRACK,
        SimulationPhase.JUNCTION_SWITCH,
    }:
        return region in {"BOTTOM", "JUNCTION", active_branch}
    if phase in {
        SimulationPhase.FINAL_JUNCTION_GATHER,
        SimulationPhase.RETURN_TO_BASE,
    }:
        return region in {"BOTTOM", "JUNCTION", "UP", "LEFT", "RIGHT"}
    if phase == SimulationPhase.DONE:
        return region in {"BOTTOM", "JUNCTION"}
    return region != "OUTSIDE"


def is_walkable(position: pygame.Vector2, radius: float) -> bool:
    x = int(round(position.x))
    y = int(round(position.y))
    diagonal = int(round(radius / math.sqrt(2.0)))
    test_points = [
        (x, y),
        (x + int(radius), y),
        (x - int(radius), y),
        (x, y + int(radius)),
        (x, y - int(radius)),
        (x + diagonal, y + diagonal),
        (x + diagonal, y - diagonal),
        (x - diagonal, y + diagonal),
        (x - diagonal, y - diagonal),
    ]
    for px, py in test_points:
        if not (0 <= px < SCREEN_WIDTH and 0 <= py < SCREEN_HEIGHT):
            return False
        if walkable_mask.get_at((px, py)) == 0:
            return False
    return is_region_allowed(position)


def is_mask_clear_at(position: pygame.Vector2, clearance: float = 0.0) -> bool:
    x = int(round(position.x))
    y = int(round(position.y))
    radius = max(0, int(math.ceil(clearance)))
    offsets = [(0, 0)]
    if radius > 0:
        offsets += [(radius, 0), (-radius, 0), (0, radius), (0, -radius)]
    for dx, dy in offsets:
        px, py = x + dx, y + dy
        if not (0 <= px < SCREEN_WIDTH and 0 <= py < SCREEN_HEIGHT):
            return False
        if walkable_mask.get_at((px, py)) == 0:
            return False
    return True


def has_line_of_sight(a: pygame.Vector2, b: pygame.Vector2) -> bool:
    delta = b - a
    distance = delta.length()
    if distance <= EPSILON:
        return True
    region_a = get_robot_region(a)
    region_b = get_robot_region(b)
    if region_a != "OUTSIDE" and region_a == region_b:
        return True
    sample_count = max(1, int(math.ceil(distance / COMM_LOS_SAMPLE_SPACING)))
    for index in range(1, sample_count):
        sample = a + delta * (index / sample_count)
        if not is_mask_clear_at(sample, COMM_LOS_CLEARANCE):
            return False
    return True

# =========================================================
# 6. General utilities
# =========================================================


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def limit_vector(vector: pygame.Vector2, maximum_length: float) -> pygame.Vector2:
    if vector.length_squared() > maximum_length * maximum_length:
        vector.scale_to_length(maximum_length)
    return vector


def spiky_kernel(distance: float, h: float) -> float:
    if distance < 0.0 or distance > h:
        return 0.0
    q = 1.0 - distance / h
    return 10.0 / (math.pi * h * h) * q**3


def spiky_gradient(r_ij: pygame.Vector2, h: float) -> pygame.Vector2:
    distance = r_ij.length()
    if distance <= EPSILON or distance > h:
        return pygame.Vector2()
    q = 1.0 - distance / h
    magnitude = -30.0 / (math.pi * h**3) * q**2
    return magnitude * (r_ij / distance)


def interpolate_color(a, b, ratio):
    ratio = clamp(ratio, 0.0, 1.0)
    return tuple(int(a[i] + (b[i] - a[i]) * ratio) for i in range(3))


def density_to_color(density: float, reference_density: float):
    ratio = density / max(reference_density, EPSILON)
    if ratio <= 1.0:
        return interpolate_color((151, 190, 226), (142, 204, 190), ratio)
    return interpolate_color(
        (142, 204, 190),
        (242, 187, 126),
        min((ratio - 1.0) / 0.75, 1.0),
    )


def normalized_direction_toward(source, target):
    delta = target - source
    return delta.normalize() if delta.length_squared() > EPSILON else pygame.Vector2()


def get_bottom_hold_point():
    return pygame.Vector2(center_x, center_y + half_width + normal_length - 18)


def get_branch_tip_target(branch: str):
    if branch == "UP":
        return pygame.Vector2(center_x, center_y - half_width - normal_length + 18)
    if branch == "LEFT":
        return pygame.Vector2(center_x - half_width - normal_length + 18, center_y)
    return pygame.Vector2(center_x + half_width + right_length - 18, center_y)


def get_backtrack_direction(branch: str):
    return -BRANCH_DIRECTIONS[branch]


def branch_progress_position(position: pygame.Vector2, branch: str) -> float:
    if branch == "UP":
        return -position.y
    if branch == "LEFT":
        return -position.x
    if branch == "RIGHT":
        return position.x
    return 0.0


def branch_progress(robot: "Robot", branch: str) -> float:
    return branch_progress_position(robot.position, branch)


def branch_depth_from_junction(position: pygame.Vector2, branch: str) -> float:
    """Distance travelled inside a branch, measured from its junction mouth."""
    if branch == "UP":
        return clamp((center_y - half_width) - position.y, 0.0, BRANCH_LENGTHS[branch])
    if branch == "LEFT":
        return clamp((center_x - half_width) - position.x, 0.0, BRANCH_LENGTHS[branch])
    if branch == "RIGHT":
        return clamp(position.x - (center_x + half_width), 0.0, BRANCH_LENGTHS[branch])
    return 0.0


def branch_point_at_depth(branch: str, depth: float) -> pygame.Vector2:
    depth = clamp(depth, 0.0, BRANCH_LENGTHS[branch])
    if branch == "UP":
        return pygame.Vector2(center_x, center_y - half_width - depth)
    if branch == "LEFT":
        return pygame.Vector2(center_x - half_width - depth, center_y)
    return pygame.Vector2(center_x + half_width + depth, center_y)


def get_shepherd_boundary_depth(branch: str) -> float:
    return max(0.0, BRANCH_LENGTHS[branch] - SHEPHERD_BOUNDARY_WALL_CLEARANCE)


def get_shepherd_fill_target(branch: str) -> pygame.Vector2:
    return branch_point_at_depth(
        branch,
        max(0.0, get_shepherd_boundary_depth(branch) - SHEPHERD_FILL_GAP),
    )


def get_saturation_rect(branch: str) -> pygame.Rect:
    boundary_depth = get_shepherd_boundary_depth(branch)
    near_depth = max(0.0, boundary_depth - SHEPHERD_FILL_REGION_DEPTH)
    far_depth = max(near_depth + 1.0, boundary_depth - ROBOT_RADIUS * 2.0)
    if branch == "UP":
        top = center_y - half_width - far_depth
        bottom = center_y - half_width - near_depth
        return pygame.Rect(center_x - half_width, int(top), corridor_width, int(bottom - top))
    if branch == "LEFT":
        left = center_x - half_width - far_depth
        right = center_x - half_width - near_depth
        return pygame.Rect(int(left), center_y - half_width, int(right - left), corridor_width)
    left = center_x + half_width + near_depth
    right = center_x + half_width + far_depth
    return pygame.Rect(int(left), center_y - half_width, int(right - left), corridor_width)


def angle_between(a: pygame.Vector2, b: pygame.Vector2) -> float:
    if a.length_squared() <= EPSILON or b.length_squared() <= EPSILON:
        return 0.0
    dot = clamp(a.normalize().dot(b.normalize()), -1.0, 1.0)
    return math.acos(dot)

# =========================================================
# 7. Experiment metrics
# =========================================================


@dataclass
class ExperimentMetrics:
    start_time: float = 0.0
    completion_time: Optional[float] = None
    disconnected_robot_seconds: float = 0.0
    minimum_pair_distance: float = float("inf")
    safety_violations: int = 0
    saturation_events: list[dict] = field(default_factory=list)
    branch_events: list[dict] = field(default_factory=list)
    pressure_events: list[dict] = field(default_factory=list)
    saved: bool = False


metrics = ExperimentMetrics()


def save_experiment_logs(robots: list["Robot"], reason: str) -> Path:
    if metrics.saved:
        return Path("sph_dfs_experiment_summary.csv")
    metrics.saved = True
    output = Path(__file__).resolve().with_name("sph_dfs_experiment_summary.csv")
    total_distance = sum(robot.total_distance for robot in robots)
    normal_distance = sum(robot.distance_by_role.get("NORMAL", 0.0) for robot in robots)
    relay_distance = sum(robot.distance_by_role.get("RELAY", 0.0) for robot in robots)
    trunk_relay_distance = sum(robot.distance_by_role.get("TRUNK_RELAY", 0.0) for robot in robots)
    shepherd_distance = sum(robot.distance_by_role.get("SHEPHERD", 0.0) for robot in robots)
    anchor_distance = sum(robot.distance_by_role.get("ANCHOR", 0.0) for robot in robots)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["reason", reason])
        writer.writerow(["robot_count", len(robots)])
        writer.writerow(["simulation_time", f"{simulation_time:.6f}"])
        writer.writerow(["total_robot_distance", f"{total_distance:.6f}"])
        writer.writerow(["normal_distance", f"{normal_distance:.6f}"])
        writer.writerow(["relay_distance", f"{relay_distance:.6f}"])
        writer.writerow(["trunk_relay_distance", f"{trunk_relay_distance:.6f}"])
        writer.writerow(["shepherd_distance", f"{shepherd_distance:.6f}"])
        writer.writerow(["anchor_distance", f"{anchor_distance:.6f}"])
        writer.writerow([
            "disconnected_robot_seconds",
            f"{metrics.disconnected_robot_seconds:.6f}",
        ])
        writer.writerow([
            "minimum_pair_distance",
            "" if not math.isfinite(metrics.minimum_pair_distance)
            else f"{metrics.minimum_pair_distance:.6f}",
        ])
        writer.writerow(["safety_violations", metrics.safety_violations])
        writer.writerow(["branch_order", " > ".join(branch_order_plan)])
        writer.writerow(["saturation_event_count", len(metrics.saturation_events)])
        writer.writerow(["pressure_event_count", len(metrics.pressure_events)])
    print(f"[Log] saved: {output}")
    return output

# =========================================================
# 8. Base station and Robot
# =========================================================


class BaseStation:
    """Virtual fixed communication node at the deployment entrance."""

    def __init__(self, position: pygame.Vector2):
        self.robot_id = -1
        self.position = position.copy()
        self.role = "BASE"
        self.comm_neighbors: list[object] = []
        self.connected_to_base = True
        self.comm_hop = 0
        self.comm_parent = None
        self.comm_path_margin = float("inf")
        self.received_branch = None
        self.received_command = None
        self.received_sequence = -1


class Robot:
    def __init__(self, x: float, y: float, robot_id: int):
        self.robot_id = robot_id
        self.position = pygame.Vector2(x, y)
        self.previous_position = self.position.copy()
        self.ingress_lane_x = float(x)
        self.velocity = pygame.Vector2()
        self.acceleration = pygame.Vector2()
        self.radius = ROBOT_RADIUS
        self.density = 0.0
        self.pressure = 0.0
        self.role = "NORMAL"

        self.shepherd_anchor: Optional[pygame.Vector2] = None
        self.shepherd_origin: Optional[pygame.Vector2] = None
        self.relay_anchor: Optional[pygame.Vector2] = None
        self.relay_index = -1
        self.anchor_position: Optional[pygame.Vector2] = None
        self.local_branch_states = None
        self.selected_branch = None
        self.parent_branch = "BOTTOM"

        self.anchor_region_entry_time: Optional[float] = None
        self.was_in_anchor_region = anchor_election_rect.collidepoint(x, y)
        self.anchor_election_score = 0.0

        self.comm_neighbors: list[object] = []
        self.connected_to_base = False
        self.comm_hop = -1
        self.comm_parent: Optional[object] = None
        self.comm_path_margin = float("-inf")
        self.received_branch = None
        self.received_command = None
        self.received_sequence = -1

        self.total_distance = 0.0
        self.distance_by_role = {
            "NORMAL": 0.0,
            "ANCHOR": 0.0,
            "RELAY": 0.0,
            "TRUNK_RELAY": 0.0,
            "SHEPHERD": 0.0,
        }

    def _record_motion(self):
        distance = self.position.distance_to(self.previous_position)
        self.total_distance += distance
        self.distance_by_role[self.role] = self.distance_by_role.get(self.role, 0.0) + distance
        self.previous_position = self.position.copy()

    def update(self, dt: float):
        old_position = self.position.copy()

        if self.role == "ANCHOR" and self.anchor_position is not None:
            error = self.anchor_position - self.position
            if error.length_squared() > ANCHOR_POSITION_TOLERANCE**2:
                scale = get_anchor_deployment_motion_scale(self)
                step = ANCHOR_MOVE_SPEED * scale * dt
                if step > 0.0:
                    next_position = (
                        self.anchor_position.copy()
                        if error.length() <= step
                        else self.position + error.normalize() * step
                    )
                    if is_walkable(next_position, self.radius):
                        self.position = next_position
            else:
                self.position = self.anchor_position.copy()
            self.velocity.update(0.0, 0.0)
            self.acceleration.update(0.0, 0.0)
            self.previous_position = old_position
            self._record_motion()
            return

        if self.role in {"RELAY", "TRUNK_RELAY"} and self.relay_anchor is not None:
            error = self.relay_anchor - self.position
            if error.length_squared() > RELAY_POSITION_TOLERANCE**2:
                step = RELAY_MOVE_SPEED * dt
                next_position = (
                    self.relay_anchor.copy()
                    if error.length() <= step
                    else self.position + error.normalize() * step
                )
                if is_walkable(next_position, self.radius):
                    self.position = next_position
            else:
                self.position = self.relay_anchor.copy()
            self.velocity.update(0.0, 0.0)
            self.acceleration.update(0.0, 0.0)
            self.previous_position = old_position
            self._record_motion()
            return

        if self.role == "SHEPHERD" and self.shepherd_anchor is not None:
            if phase == SimulationPhase.FORM_SHEPHERD_BOUNDARY:
                target = self.shepherd_anchor
            elif phase == SimulationPhase.FILL_BEHIND_SHEPHERD:
                target = self.shepherd_anchor
            elif phase == SimulationPhase.PRESSURE_PUSH:
                travel = min(
                    SHEPHERD_PISTON_MAX_TRAVEL,
                    pressure_push_timer * SHEPHERD_PISTON_SPEED,
                )
                target = self.shepherd_anchor + get_backtrack_direction(active_branch) * travel
            else:
                target = None

            if target is not None:
                error = target - self.position
                # Shepherd motion is limited by the Base-rooted communication
                # path while the boundary is being formed.
                if phase == SimulationPhase.FORM_SHEPHERD_BOUNDARY:
                    if not self.connected_to_base:
                        motion_scale = 0.0
                    elif self.comm_path_margin < RELAY_DEPLOY_MARGIN:
                        motion_scale = 0.35
                    else:
                        motion_scale = 1.0
                else:
                    motion_scale = 1.0
                step = SHEPHERD_FORM_SPEED * motion_scale * dt
                next_position = (
                    target.copy()
                    if error.length() <= step
                    else self.position + error.normalize() * step
                ) if error.length_squared() > EPSILON and step > 0.0 else self.position.copy()
                if is_walkable(next_position, self.radius):
                    self.position = next_position
                self.velocity.update(0.0, 0.0)
                self.acceleration.update(0.0, 0.0)
                self.previous_position = old_position
                self._record_motion()
                return

        self.velocity += self.acceleration * dt
        limit_vector(self.velocity, MAX_SPEED)
        x_position = pygame.Vector2(self.position.x + self.velocity.x * dt, self.position.y)
        if is_walkable(x_position, self.radius):
            self.position.x = x_position.x
        else:
            self.velocity.x = 0.0
        y_position = pygame.Vector2(self.position.x, self.position.y + self.velocity.y * dt)
        if is_walkable(y_position, self.radius):
            self.position.y = y_position.y
        else:
            self.velocity.y = 0.0
        self.acceleration.update(0.0, 0.0)
        self.previous_position = old_position
        self._record_motion()

    def draw(self, surface, color_reference_density, show_density_color):
        x, y = round(self.position.x), round(self.position.y)
        if self.role == "ANCHOR":
            color = ANCHOR_COLOR
        elif self.role == "TRUNK_RELAY":
            color = TRUNK_RELAY_COLOR
        elif self.role == "RELAY":
            color = RELAY_COLOR
        elif self.role == "SHEPHERD":
            color = SHEPHERD_COLOR
        elif show_density_color:
            color = density_to_color(self.density, color_reference_density)
        else:
            color = ROBOT_BASE_COLOR
        marker = pygame.Rect(
            x - self.radius,
            y - self.radius,
            self.radius * 2 + 1,
            self.radius * 2 + 1,
        )
        pygame.draw.rect(surface, color, marker, border_radius=self.radius)
        if self.role in {"RELAY", "TRUNK_RELAY"}:
            ring_color = TRUNK_RELAY_COLOR if self.role == "TRUNK_RELAY" else RELAY_COLOR
            pygame.draw.circle(surface, ring_color, (x, y), self.radius + 3, width=1)
        if base_station is not None and self.role != "BASE" and not self.connected_to_base:
            pygame.draw.rect(
                surface,
                DISCONNECTED_COLOR,
                marker.inflate(2, 2),
                width=1,
                border_radius=self.radius + 1,
            )

# =========================================================
# 9. Robot creation and spatial hash
# =========================================================


def create_grid_robots(robot_count: int):
    robots = []
    left = center_x - half_width + ROBOT_RADIUS + 4
    right = center_x + half_width - ROBOT_RADIUS - 4
    top = center_y + half_width + 12
    bottom = center_y + half_width + normal_length - ROBOT_RADIUS - 7
    per_row = max(1, int((right - left) // GRID_SPACING) + 1)
    for robot_id in range(robot_count):
        row, column = divmod(robot_id, per_row)
        x = left + column * GRID_SPACING
        y = bottom - row * GRID_SPACING
        if y < top:
            print(f"Warning: only {len(robots)} robots fit in the entrance.")
            break
        robots.append(Robot(x, y, robot_id))
    return robots


def create_random_robots(robot_count: int):
    robots = []
    minimum_distance = ROBOT_RADIUS * 2 + 1
    left = center_x - half_width + ROBOT_RADIUS + 4
    right = center_x + half_width - ROBOT_RADIUS - 4
    top = center_y + half_width + 12
    bottom = center_y + half_width + normal_length - ROBOT_RADIUS - 7
    attempts = 0
    while len(robots) < robot_count and attempts < robot_count * 400:
        attempts += 1
        candidate = pygame.Vector2(random.uniform(left, right), random.uniform(top, bottom))
        if all(candidate.distance_to(robot.position) >= minimum_distance for robot in robots):
            robots.append(Robot(candidate.x, candidate.y, len(robots)))
    return robots


def cell_key(position):
    return int(position.x // CELL_SIZE), int(position.y // CELL_SIZE)


def build_spatial_grid(robots):
    grid = {}
    for robot in robots:
        grid.setdefault(cell_key(robot.position), []).append(robot)
    return grid


def iter_neighbor_candidates(robot, grid):
    cx, cy = cell_key(robot.position)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            yield from grid.get((cx + dx, cy + dy), [])

# =========================================================
# 10. Base-rooted communication
# =========================================================


def update_communication_neighbors(robots, grid):
    """Build robot links and explicit Base-to-robot links with LOS checks."""
    range_sq = COMM_RANGE**2
    for robot in robots:
        robot.comm_neighbors = []
    if base_station is None:
        return
    base_station.comm_neighbors = []

    for robot in robots:
        for other in iter_neighbor_candidates(robot, grid):
            if other.robot_id <= robot.robot_id:
                continue
            if robot.position.distance_squared_to(other.position) > range_sq:
                continue
            if not has_line_of_sight(robot.position, other.position):
                continue
            robot.comm_neighbors.append(other)
            other.comm_neighbors.append(robot)

    for robot in robots:
        if base_station.position.distance_squared_to(robot.position) > range_sq:
            continue
        if not has_line_of_sight(base_station.position, robot.position):
            continue
        base_station.comm_neighbors.append(robot)
        robot.comm_neighbors.append(base_station)


def get_anchor_deployment_motion_scale(anchor):
    if anchor is None or anchor.role != "ANCHOR":
        return 0.0
    # Anchor may only move while it remains connected to the fixed Base.
    if not anchor.connected_to_base:
        return 0.0
    neighbors = [
        neighbor for neighbor in anchor.comm_neighbors
        if getattr(neighbor, "role", None) not in {"ANCHOR", "BASE"}
    ]
    if len(neighbors) < ANCHOR_MIN_DIRECT_NEIGHBORS:
        return 0.0
    nearest = min(anchor.position.distance_to(neighbor.position) for neighbor in neighbors)
    if nearest <= ANCHOR_LINK_WARNING_DISTANCE:
        return 1.0
    if nearest >= ANCHOR_LINK_STOP_DISTANCE:
        return 0.0
    return clamp(
        (ANCHOR_LINK_STOP_DISTANCE - nearest)
        / (ANCHOR_LINK_STOP_DISTANCE - ANCHOR_LINK_WARNING_DISTANCE),
        0.0,
        1.0,
    )


def anchor_deployment_ready(anchor, robots):
    if anchor is None or anchor.anchor_position is None:
        return False
    if anchor.position.distance_to(anchor.anchor_position) > ANCHOR_POSITION_TOLERANCE:
        return False
    if not anchor.connected_to_base or not trunk_plan_ready(robots):
        return False
    usable = [
        neighbor
        for neighbor in anchor.comm_neighbors
        if getattr(neighbor, "role", None) not in {"ANCHOR", "BASE"}
        and anchor.position.distance_to(neighbor.position) <= COMM_RANGE * 0.92
    ]
    return len(usable) >= ANCHOR_READY_DIRECT_NEIGHBORS


def get_anchor_message(anchor):
    if anchor is None or not anchor.connected_to_base:
        return None, "WAIT_FOR_BASE_LINK"
    return anchor.selected_branch, phase.name


def propagate_base_message(robots, anchor):
    """Compute widest paths from the fixed Base, then distribute Anchor's command.

    Connectivity is rooted at Base. The Anchor only supplies the DFS command; the
    command is accepted and propagated when the Anchor itself has a Base path.
    """
    global communication_sequence, last_message_signature
    if base_station is None:
        return

    for robot in robots:
        robot.connected_to_base = False
        robot.comm_hop = -1
        robot.comm_parent = None
        robot.comm_path_margin = float("-inf")
        robot.received_branch = None
        robot.received_command = None
        robot.received_sequence = -1

    base_station.connected_to_base = True
    base_station.comm_hop = 0
    base_station.comm_parent = None
    base_station.comm_path_margin = float("inf")

    heap = [(-1.0e12, base_station.robot_id, base_station)]
    while heap:
        _, _, current = heapq.heappop(heap)
        for neighbor in current.comm_neighbors:
            edge_margin = COMM_RANGE - current.position.distance_to(neighbor.position)
            candidate_margin = min(current.comm_path_margin, edge_margin)
            if candidate_margin <= neighbor.comm_path_margin + EPSILON:
                continue
            neighbor.connected_to_base = True
            neighbor.comm_parent = current
            neighbor.comm_hop = current.comm_hop + 1
            neighbor.comm_path_margin = candidate_margin
            heapq.heappush(heap, (-candidate_margin, neighbor.robot_id, neighbor))

    selected_branch, command = get_anchor_message(anchor)
    signature = (selected_branch, command)
    if signature != last_message_signature:
        communication_sequence += 1
        last_message_signature = signature
        print(
            f"[Base Communication] seq={communication_sequence}, "
            f"command={command}, branch={selected_branch}"
        )

    base_station.received_branch = selected_branch
    base_station.received_command = command
    base_station.received_sequence = communication_sequence
    for robot in robots:
        if not robot.connected_to_base:
            continue
        robot.received_branch = selected_branch
        robot.received_command = command
        robot.received_sequence = communication_sequence


def update_communication_system(robots, grid):
    update_communication_neighbors(robots, grid)
    propagate_base_message(robots, junction_anchor)


def find_nearest_connected_robot(robot, grid):
    nearest = None
    best_sq = COMM_RECOVERY_RANGE**2
    for candidate in iter_neighbor_candidates(robot, grid):
        if candidate is robot or not candidate.connected_to_base:
            continue
        distance_sq = robot.position.distance_squared_to(candidate.position)
        if distance_sq >= best_sq:
            continue
        if not has_line_of_sight(robot.position, candidate.position):
            continue
        nearest, best_sq = candidate, distance_sq
    if base_station is not None:
        distance_sq = robot.position.distance_squared_to(base_station.position)
        if (
            distance_sq < best_sq
            and distance_sq <= COMM_RECOVERY_RANGE**2
            and has_line_of_sight(robot.position, base_station.position)
        ):
            nearest = base_station
    return nearest


def compute_connectivity_force(robot, grid):
    if base_station is None or robot.role in {"ANCHOR", "RELAY", "TRUNK_RELAY"}:
        return pygame.Vector2()
    if robot.connected_to_base:
        return pygame.Vector2()
    target = find_nearest_connected_robot(robot, grid)
    if target is None:
        return pygame.Vector2()
    delta = target.position - robot.position
    distance = delta.length()
    return (
        COMM_RECOVERY_GAIN * min(distance, COMM_RECOVERY_RANGE) * delta / distance
        if distance > EPSILON
        else pygame.Vector2()
    )


def get_communication_stats(robots):
    if base_station is None:
        return {
            "connected": 0,
            "disconnected": len(robots),
            "max_hop": 0,
            "direct": 0,
            "margin": 0.0,
            "anchor_connected": False,
        }
    connected = [robot for robot in robots if robot.connected_to_base]
    margins = [
        robot.comm_path_margin
        for robot in connected
        if math.isfinite(robot.comm_path_margin)
    ]
    return {
        "connected": len(connected),
        "disconnected": len(robots) - len(connected),
        "max_hop": max((robot.comm_hop for robot in connected), default=0),
        "direct": len(base_station.comm_neighbors),
        "margin": min(margins) if margins else COMM_RANGE,
        "anchor_connected": bool(junction_anchor and junction_anchor.connected_to_base),
    }


def draw_communication_links(surface, robots):
    if base_station is None:
        return
    for robot in robots:
        parent = robot.comm_parent
        if not robot.connected_to_base or parent is None:
            continue
        distance = robot.position.distance_to(parent.position)
        color = (
            COMM_LINK_SAFE_COLOR
            if distance <= COMM_SAFE_DISTANCE
            else COMM_LINK_WARNING_COLOR
            if distance <= COMM_BARRIER_START
            else COMM_LINK_DANGER_COLOR
        )
        pygame.draw.line(surface, color, robot.position, parent.position, width=1)

# =========================================================
# 11. Permanent Base trunk and adaptive branch relays
# =========================================================


def initialize_trunk_relay_plan():
    global trunk_relay_slots, trunk_relay_deploy_cooldown
    vector = ANCHOR_PARK_POSITION - BASE_POSITION
    length = vector.length()
    trunk_relay_slots = []
    trunk_relay_deploy_cooldown = 0.0
    if length <= EPSILON:
        return
    direction = vector / length
    distance = TRUNK_RELAY_SPACING
    index = 0
    while distance < length - COMM_SAFE_DISTANCE * 0.35:
        trunk_relay_slots.append({
            "index": index,
            "position": BASE_POSITION + direction * distance,
            "path_distance": distance,
        })
        index += 1
        distance += TRUNK_RELAY_SPACING
    print(f"[Base Trunk] slots={len(trunk_relay_slots)}")


def trunk_path_progress(position):
    vector = ANCHOR_PARK_POSITION - BASE_POSITION
    length = vector.length()
    if length <= EPSILON:
        return 0.0
    return clamp((position - BASE_POSITION).dot(vector / length), 0.0, length)


def get_trunk_relays(robots):
    return sorted(
        [robot for robot in robots if robot.role == "TRUNK_RELAY"],
        key=lambda robot: robot.relay_index,
    )


def get_next_undeployed_trunk_slot(robots):
    deployed = {robot.relay_index for robot in get_trunk_relays(robots)}
    return next((slot for slot in trunk_relay_slots if slot["index"] not in deployed), None)


def trunk_relay_at_slot_is_settled(robot):
    return (
        robot.role == "TRUNK_RELAY"
        and robot.relay_anchor is not None
        and robot.position.distance_to(robot.relay_anchor) <= RELAY_POSITION_TOLERANCE
    )


def select_trunk_relay_candidate(robots, slot):
    candidates = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and robot.connected_to_base
        and get_robot_region(robot.position) in {"BOTTOM", "JUNCTION"}
        and robot.position.distance_to(slot["position"]) <= TRUNK_RELAY_SELECTION_RADIUS
    ]
    return min(
        candidates,
        key=lambda robot: (
            robot.position.distance_squared_to(slot["position"]),
            robot.velocity.length_squared(),
            robot.robot_id,
        ),
        default=None,
    )


def update_trunk_relay_deployment(robots, dt):
    global trunk_relay_deploy_cooldown
    trunk_relay_deploy_cooldown = max(0.0, trunk_relay_deploy_cooldown - dt)
    if phase != SimulationPhase.MOVE_TO_JUNCTION:
        return
    next_slot = get_next_undeployed_trunk_slot(robots)
    if next_slot is None:
        return
    front_progress = max(
        (
            trunk_path_progress(robot.position)
            for robot in robots
            if robot.role == "NORMAL"
            and get_robot_region(robot.position) in {"BOTTOM", "JUNCTION"}
        ),
        default=0.0,
    )
    if front_progress < next_slot["path_distance"] - TRUNK_RELAY_DEPLOY_LOOKAHEAD:
        return
    if trunk_relay_deploy_cooldown > 0.0:
        return
    candidate = select_trunk_relay_candidate(robots, next_slot)
    if candidate is None:
        return
    candidate.role = "TRUNK_RELAY"
    candidate.relay_anchor = next_slot["position"].copy()
    candidate.relay_index = next_slot["index"]
    candidate.velocity.update(0.0, 0.0)
    candidate.acceleration.update(0.0, 0.0)
    trunk_relay_deploy_cooldown = TRUNK_RELAY_DEPLOY_COOLDOWN
    print(
        f"[Base Trunk] deployed robot={candidate.robot_id}, "
        f"index={candidate.relay_index}"
    )


def trunk_plan_ready(robots):
    if not trunk_relay_slots:
        return True
    relays = {robot.relay_index: robot for robot in get_trunk_relays(robots)}
    return all(
        slot["index"] in relays and trunk_relay_at_slot_is_settled(relays[slot["index"]])
        for slot in trunk_relay_slots
    )


def release_trunk_relays_for_return(robots):
    released = 0
    for robot in robots:
        if robot.role != "TRUNK_RELAY":
            continue
        robot.role = "NORMAL"
        robot.relay_anchor = None
        robot.relay_index = -1
        robot.velocity.update(0.0, 0.0)
        released += 1
    print(f"[Base Trunk] released for final return={released}")


def draw_trunk_relay_plan(surface, robots):
    for slot in trunk_relay_slots:
        pygame.draw.circle(surface, TRUNK_RELAY_COLOR, slot["position"], 3, width=1)
    nodes = [base_station] if base_station is not None else []
    nodes.extend(get_trunk_relays(robots))
    if junction_anchor is not None:
        nodes.append(junction_anchor)
    for first, second in zip(nodes, nodes[1:]):
        pygame.draw.line(surface, TRUNK_RELAY_COLOR, first.position, second.position, width=2)


def get_relay_path_endpoint(branch):
    if branch == "UP":
        return pygame.Vector2(
            center_x - half_width + RELAY_LANE_MARGIN,
            center_y - half_width - normal_length + RELAY_END_CLEARANCE,
        )
    if branch == "LEFT":
        return pygame.Vector2(
            center_x - half_width - normal_length + RELAY_END_CLEARANCE,
            center_y - half_width + RELAY_LANE_MARGIN,
        )
    return pygame.Vector2(
        center_x + half_width + right_length - RELAY_END_CLEARANCE,
        center_y - half_width + RELAY_LANE_MARGIN,
    )


def initialize_relay_plan(branch):
    global relay_slots, relay_deploy_cooldown, relay_retract_cooldown
    global relay_retract_clear_timer, relay_motion_scale
    start = ANCHOR_PARK_POSITION.copy()
    end = get_relay_path_endpoint(branch)
    vector = end - start
    length = vector.length()
    relay_slots = []
    relay_deploy_cooldown = relay_retract_cooldown = relay_retract_clear_timer = 0.0
    relay_motion_scale = 1.0
    if length <= EPSILON:
        return
    direction = vector / length
    distance = RELAY_SPACING
    index = 0
    while distance <= length:
        relay_slots.append({
            "index": index,
            "position": start + direction * distance,
            "path_distance": distance,
        })
        index += 1
        distance += RELAY_SPACING
    print(f"[Relay] plan branch={branch}, slots={len(relay_slots)}")


def relay_path_progress(position, branch):
    start = ANCHOR_PARK_POSITION
    vector = get_relay_path_endpoint(branch) - start
    length = vector.length()
    if length <= EPSILON:
        return 0.0
    return clamp((position - start).dot(vector / length), 0.0, length)


def get_relays(robots):
    return [robot for robot in robots if robot.role == "RELAY"]


def get_active_branch_relays(robots):
    return sorted(
        [robot for robot in robots if robot.role == "RELAY" and robot.relay_index >= 0],
        key=lambda robot: robot.relay_index,
    )


def relay_at_slot_is_settled(robot):
    return (
        robot.role == "RELAY"
        and robot.relay_anchor is not None
        and robot.position.distance_to(robot.relay_anchor) <= RELAY_POSITION_TOLERANCE
    )


def get_exploration_front_progress(robots, branch):
    candidates = [
        robot
        for robot in robots
        if robot.role in {"NORMAL", "SHEPHERD"}
        and get_robot_region(robot.position) in {"JUNCTION", branch}
    ]
    return max((relay_path_progress(robot.position, branch) for robot in candidates), default=0.0)


def get_next_undeployed_slot(robots):
    deployed = {robot.relay_index for robot in get_active_branch_relays(robots)}
    return next((slot for slot in relay_slots if slot["index"] not in deployed), None)


def select_relay_candidate(robots, slot):
    candidates = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and robot.connected_to_base
        and get_robot_region(robot.position) in {"JUNCTION", active_branch}
        and robot.position.distance_to(slot["position"]) <= RELAY_SELECTION_RADIUS
    ]
    return min(
        candidates,
        key=lambda robot: (
            robot.position.distance_squared_to(slot["position"]),
            robot.velocity.length_squared(),
            robot.robot_id,
        ),
        default=None,
    )


def deploy_relay_for_slot(robots, slot):
    candidate = select_relay_candidate(robots, slot)
    if candidate is None:
        return False
    candidate.role = "RELAY"
    candidate.relay_anchor = slot["position"].copy()
    candidate.relay_index = slot["index"]
    candidate.velocity.update(0.0, 0.0)
    print(f"[Relay] deploy robot={candidate.robot_id}, index={candidate.relay_index}")
    return True


def get_front_communication_status(robots, branch):
    branch_robots = [
        robot
        for robot in robots
        if robot.role in {"NORMAL", "SHEPHERD"}
        and get_robot_region(robot.position) == branch
    ]
    if not branch_robots:
        return {"count": 0, "connected_ratio": 1.0, "minimum_margin": COMM_RANGE, "needs_relay": False}
    branch_robots.sort(key=lambda robot: relay_path_progress(robot.position, branch), reverse=True)
    front_count = min(
        len(branch_robots),
        max(RELAY_FRONT_MIN_COUNT, int(math.ceil(len(branch_robots) * RELAY_FRONT_FRACTION))),
    )
    front = branch_robots[:front_count]
    connected = [robot for robot in front if robot.connected_to_base]
    ratio = len(connected) / max(1, len(front))
    margins = sorted(
        robot.comm_path_margin
        for robot in connected
        if math.isfinite(robot.comm_path_margin)
    )
    robust_margin = margins[min(len(margins) - 1, int(0.20 * len(margins)))] if margins else -COMM_RANGE
    return {
        "count": len(front),
        "connected_ratio": ratio,
        "minimum_margin": robust_margin,
        "needs_relay": ratio < RELAY_FRONT_REQUIRED_CONNECTED_RATIO or robust_margin < RELAY_DEPLOY_MARGIN,
    }


def update_relay_deployment(robots, dt):
    global relay_deploy_cooldown, relay_motion_scale
    relay_deploy_cooldown = max(0.0, relay_deploy_cooldown - dt)
    relay_motion_scale = 1.0
    if phase not in {SimulationPhase.EXPLORE_BRANCH, SimulationPhase.FORM_SHEPHERD_BOUNDARY, SimulationPhase.FILL_BEHIND_SHEPHERD} or junction_anchor is None:
        return
    front_progress = get_exploration_front_progress(robots, active_branch)
    status = get_front_communication_status(robots, active_branch)
    next_slot = get_next_undeployed_slot(robots)
    if status["needs_relay"]:
        relay_motion_scale = RELAY_WAIT_SPEED_SCALE
        if next_slot is not None:
            reached = front_progress >= next_slot["path_distance"] - RELAY_DEPLOY_LOOKAHEAD
            if reached and relay_deploy_cooldown <= 0.0 and deploy_relay_for_slot(robots, next_slot):
                relay_deploy_cooldown = RELAY_DEPLOY_COOLDOWN
                relay_motion_scale = RELAY_FORMING_SPEED_SCALE
    if any(not relay_at_slot_is_settled(relay) for relay in get_active_branch_relays(robots)):
        relay_motion_scale = min(relay_motion_scale, RELAY_FORMING_SPEED_SCALE)
    if status["connected_ratio"] < 0.55:
        relay_motion_scale = 0.0


def release_relay_into_backtracking(robot):
    index = robot.relay_index
    robot.role = "NORMAL"
    robot.relay_anchor = None
    robot.relay_index = -1
    robot.velocity = get_backtrack_direction(active_branch) * RELAY_RELEASE_SPEED
    print(f"[Relay] retract robot={robot.robot_id}, index={index}")


def update_relay_retraction(robots, dt):
    global relay_retract_cooldown, relay_retract_clear_timer
    relay_retract_cooldown = max(0.0, relay_retract_cooldown - dt)
    if phase != SimulationPhase.FLOW_BACKTRACK:
        relay_retract_clear_timer = 0.0
        return
    relays = get_active_branch_relays(robots)
    if not relays:
        relay_retract_clear_timer = 0.0
        return
    farthest = relays[-1]
    progress = relay_path_progress(farthest.position, active_branch)
    mobile = [
        robot
        for robot in robots
        if robot.role in {"NORMAL", "SHEPHERD"}
        and get_robot_region(robot.position) in {active_branch, "JUNCTION"}
    ]
    beyond = [
        robot
        for robot in mobile
        if get_robot_region(robot.position) == active_branch
        and relay_path_progress(robot.position, active_branch) > progress + RELAY_PASS_MARGIN
    ]
    clear = not beyond and all(robot.connected_to_base for robot in mobile)
    relay_retract_clear_timer = relay_retract_clear_timer + dt if clear else 0.0
    if relay_retract_clear_timer >= RELAY_RETRACT_DWELL_TIME and relay_retract_cooldown <= 0.0:
        release_relay_into_backtracking(farthest)
        relay_retract_cooldown = RELAY_RETRACT_COOLDOWN
        relay_retract_clear_timer = 0.0


def draw_relay_plan(surface, robots):
    for slot in relay_slots:
        pygame.draw.circle(surface, RELAY_SLOT_COLOR, slot["position"], 3, width=1)
    nodes = ([junction_anchor] if junction_anchor is not None else []) + get_active_branch_relays(robots)
    for first, second in zip(nodes, nodes[1:]):
        pygame.draw.line(surface, RELAY_COLOR, first.position, second.position, width=2)

# =========================================================
# 12. Anchor election and cost-guided branch ordering
# =========================================================


def update_anchor_entry_records(robots, current_time):
    for robot in robots:
        inside = anchor_election_rect.collidepoint(robot.position.x, robot.position.y)
        if inside and not robot.was_in_anchor_region and robot.anchor_region_entry_time is None and robot.role == "NORMAL":
            robot.anchor_region_entry_time = current_time
        robot.was_in_anchor_region = inside


def local_visible_neighbor_count(robot, robots):
    count = 0
    margin_sum = 0.0
    for other in robots:
        if other is robot or other.role == "ANCHOR":
            continue
        distance = robot.position.distance_to(other.position)
        if distance <= ANCHOR_LOCAL_COMM_RANGE and has_line_of_sight(robot.position, other.position):
            count += 1
            margin_sum += ANCHOR_LOCAL_COMM_RANGE - distance
    average_margin = margin_sum / count if count else 0.0
    return count, average_margin


def compute_anchor_candidate_scores(candidates, robots):
    if not candidates:
        return
    entry_times = [candidate.anchor_region_entry_time for candidate in candidates]
    min_time, max_time = min(entry_times), max(entry_times)
    max_parking_distance = max(
        (candidate.position.distance_to(ANCHOR_PARK_POSITION) for candidate in candidates),
        default=1.0,
    )
    neighbor_info = {candidate.robot_id: local_visible_neighbor_count(candidate, robots) for candidate in candidates}
    max_neighbors = max((info[0] for info in neighbor_info.values()), default=1)
    max_margin = max((info[1] for info in neighbor_info.values()), default=1.0)
    parking_direction = normalized_direction_toward(pygame.Vector2(center_x, center_y), ANCHOR_PARK_POSITION)
    for candidate in candidates:
        arrival = 1.0 if max_time - min_time <= EPSILON else 1.0 - (candidate.anchor_region_entry_time - min_time) / (max_time - min_time)
        parking = 1.0 - candidate.position.distance_to(ANCHOR_PARK_POSITION) / max(max_parking_distance, EPSILON)
        velocity_direction = candidate.velocity.normalize() if candidate.velocity.length_squared() > EPSILON else parking_direction
        direction = 0.5 * (velocity_direction.dot(parking_direction) + 1.0)
        neighbor_count, average_margin = neighbor_info[candidate.robot_id]
        communication = 0.6 * neighbor_count / max(max_neighbors, 1) + 0.4 * average_margin / max(max_margin, EPSILON)
        candidate.anchor_election_score = (
            ANCHOR_WEIGHT_ARRIVAL * arrival
            + ANCHOR_WEIGHT_PARKING * parking
            + ANCHOR_WEIGHT_DIRECTION * direction
            + ANCHOR_WEIGHT_COMMUNICATION * communication
        )


def elect_junction_anchor(robots):
    global junction_anchor
    if junction_anchor is not None:
        return junction_anchor
    candidates = [
        robot
        for robot in robots
        if robot.role == "NORMAL" and robot.anchor_region_entry_time is not None
    ]
    if not candidates:
        return None
    first_entry = min(candidate.anchor_region_entry_time for candidate in candidates)
    if len(candidates) < ANCHOR_ELECTION_MIN_CANDIDATES and simulation_time - first_entry < ANCHOR_ELECTION_WAIT_TIME:
        return None
    compute_anchor_candidate_scores(candidates, robots)
    junction_anchor = max(
        candidates,
        key=lambda robot: (
            robot.anchor_election_score,
            -robot.anchor_region_entry_time,
            -robot.robot_id,
        ),
    )
    junction_anchor.role = "ANCHOR"
    junction_anchor.anchor_position = ANCHOR_PARK_POSITION.copy()
    junction_anchor.local_branch_states = branch_states
    junction_anchor.selected_branch = None
    print(
        f"[Anchor] robot={junction_anchor.robot_id}, score={junction_anchor.anchor_election_score:.3f}, "
        f"entry={junction_anchor.anchor_region_entry_time:.3f}"
    )
    return junction_anchor


def reachable_nodes_without_branch(branch: str) -> set[str]:
    blocked_target = BRANCH_TARGET_NODE[branch]
    visited = {"BASE"}
    queue = deque(["BASE"])
    while queue:
        node = queue.popleft()
        for neighbor in TOPOLOGY_ADJACENCY[node]:
            if {node, neighbor} == {"JUNCTION", blocked_target}:
                continue
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def structural_loss(branch: str) -> int:
    reachable = reachable_nodes_without_branch(branch)
    loss = 0
    for candidate in BRANCHES:
        if branch_states[candidate] == "VISITED":
            continue
        if BRANCH_TARGET_NODE[candidate] not in reachable:
            loss += 1
    return loss


def estimate_branch_comm_risk(branch: str) -> float:
    required_links = max(1, math.ceil(BRANCH_LENGTHS[branch] / COMM_SAFE_DISTANCE))
    required_relays = max(0, required_links - 1)
    relay_capacity = max(1, math.floor(BRANCH_LENGTHS[branch] / RELAY_SPACING))
    return clamp(required_relays / max(relay_capacity, 1), 0.0, 1.5)


def branch_cost(branch: str, incoming_direction: pygame.Vector2):
    length_norm = BRANCH_LENGTHS[branch] / max(BRANCH_LENGTHS.values())
    relay_count = max(0, math.ceil(BRANCH_LENGTHS[branch] / RELAY_SPACING) - 1)
    max_relays = max(1, max(math.ceil(length / RELAY_SPACING) - 1 for length in BRANCH_LENGTHS.values()))
    relay_norm = relay_count / max_relays
    backtrack_norm = length_norm
    comm_risk = estimate_branch_comm_risk(branch)
    switch_norm = angle_between(incoming_direction, BRANCH_DIRECTIONS[branch]) / math.pi
    loss = structural_loss(branch)
    max_loss = max(1, sum(branch_states[candidate] != "VISITED" for candidate in BRANCHES))
    loss_priority = loss / max_loss
    total = (
        BRANCH_COST_LENGTH_WEIGHT * length_norm
        + BRANCH_COST_RELAY_WEIGHT * relay_norm
        + BRANCH_COST_BACKTRACK_WEIGHT * backtrack_norm
        + BRANCH_COST_COMM_WEIGHT * comm_risk
        + BRANCH_COST_SWITCH_WEIGHT * switch_norm
        - BRANCH_COST_LOSS_PRIORITY_WEIGHT * loss_priority
    )
    return total, {
        "length": length_norm,
        "relay": relay_norm,
        "backtrack": backtrack_norm,
        "comm": comm_risk,
        "switch": switch_norm,
        "loss": loss,
    }


def choose_next_branch(anchor):
    global active_branch, previous_branch_direction, branch_order_plan
    if anchor is None or anchor.local_branch_states is None:
        return None
    candidates = [branch for branch in BRANCHES if anchor.local_branch_states[branch] == "UNVISITED"]
    if not candidates:
        anchor.selected_branch = None
        return None
    scored = []
    for branch in candidates:
        cost, components = branch_cost(branch, previous_branch_direction)
        scored.append((cost, branch, components))
    scored.sort(key=lambda item: (item[0], item[1]))
    cost, selected, components = scored[0]
    anchor.local_branch_states[selected] = "ACTIVE"
    anchor.selected_branch = selected
    active_branch = selected
    branch_order_plan.append(selected)
    initialize_relay_plan(selected)
    print(f"[DFS] selected={selected}, cost={cost:.3f}, components={components}")
    return selected


def complete_active_branch(anchor, branch):
    global previous_branch_direction
    if anchor is None or anchor.local_branch_states is None:
        return
    anchor.local_branch_states[branch] = "VISITED"
    anchor.selected_branch = None
    previous_branch_direction = get_backtrack_direction(branch)
    metrics.branch_events.append({"branch": branch, "completed_at": simulation_time})
    print(f"[DFS] completed={branch}")


def release_anchor_for_final_return(anchor):
    if anchor is None:
        return
    anchor.role = "NORMAL"
    anchor.anchor_position = None
    anchor.selected_branch = None
    anchor.velocity.update(0.0, 0.0)


def begin_final_gather():
    global phase, relay_slots, relay_motion_scale, final_gather_timer
    relay_slots = []
    relay_motion_scale = 1.0
    final_gather_timer = 0.0
    phase = SimulationPhase.FINAL_JUNCTION_GATHER
    print("[DFS] final gather")


def begin_final_return(anchor, robots):
    global phase, relay_slots, relay_motion_scale
    relay_slots = []
    relay_motion_scale = 1.0
    release_anchor_for_final_return(anchor)
    release_trunk_relays_for_return(robots)
    phase = SimulationPhase.RETURN_TO_BASE
    print("[DFS] return to base")

# =========================================================
# 13. Saturation detector
# =========================================================


@dataclass
class SaturationTracker:
    branch: Optional[str] = None
    dwell: float = 0.0
    progress_history: deque = field(default_factory=deque)
    low_speed_ratio: float = 0.0
    average_density_ratio: float = 0.0
    occupancy_ratio: float = 0.0
    front_delta: float = float("inf")
    tip_count: int = 0
    saturated: bool = False

    def reset(self, branch: Optional[str] = None):
        self.branch = branch
        self.dwell = 0.0
        self.progress_history.clear()
        self.low_speed_ratio = 0.0
        self.average_density_ratio = 0.0
        self.occupancy_ratio = 0.0
        self.front_delta = float("inf")
        self.tip_count = 0
        self.saturated = False


saturation_tracker = SaturationTracker()


def tip_robots(robots, branch):
    rect = get_saturation_rect(branch)
    return [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and get_robot_region(robot.position) == branch
        and rect.collidepoint(robot.position.x, robot.position.y)
    ]


def tip_occupancy_ratio(robots_at_tip, branch):
    if not robots_at_tip:
        return 0.0
    rect = get_saturation_rect(branch)
    cols = max(1, int(math.ceil(rect.width / SATURATION_CELL_SIZE)))
    rows = max(1, int(math.ceil(rect.height / SATURATION_CELL_SIZE)))
    occupied = set()
    for robot in robots_at_tip:
        col = int((robot.position.x - rect.left) // SATURATION_CELL_SIZE)
        row = int((robot.position.y - rect.top) // SATURATION_CELL_SIZE)
        occupied.add((clamp(col, 0, cols - 1), clamp(row, 0, rows - 1)))
    return len(occupied) / (cols * rows)


def update_dead_end_saturation(robots, branch, reference_density, dt):
    tracker = saturation_tracker
    if tracker.branch != branch:
        tracker.reset(branch)
    tip = tip_robots(robots, branch)
    tracker.tip_count = len(tip)
    if tip:
        forward_direction = BRANCH_DIRECTIONS[branch]
        signed_speeds = [robot.velocity.dot(forward_direction) for robot in tip]
        tracker.low_speed_ratio = sum(abs(speed) <= SATURATION_LOW_SPEED_THRESHOLD for speed in signed_speeds) / len(tip)
        average_density = sum(robot.density for robot in tip) / len(tip)
        tracker.average_density_ratio = average_density / max(reference_density, EPSILON)
        tracker.occupancy_ratio = tip_occupancy_ratio(tip, branch)
        front_progress = max(branch_progress(robot, branch) for robot in tip)
    else:
        tracker.low_speed_ratio = tracker.average_density_ratio = tracker.occupancy_ratio = 0.0
        front_progress = 0.0

    tracker.progress_history.append((simulation_time, front_progress))
    while tracker.progress_history and simulation_time - tracker.progress_history[0][0] > SATURATION_FRONT_WINDOW:
        tracker.progress_history.popleft()
    if len(tracker.progress_history) >= 2:
        values = [item[1] for item in tracker.progress_history]
        tracker.front_delta = max(values) - min(values)
    else:
        tracker.front_delta = float("inf")

    conditions = (
        tracker.tip_count >= SATURATION_MIN_TIP_ROBOTS
        and tracker.low_speed_ratio >= SATURATION_LOW_SPEED_RATIO
        and tracker.average_density_ratio >= SATURATION_DENSITY_RATIO
        and tracker.occupancy_ratio >= SATURATION_OCCUPANCY_RATIO
        and tracker.front_delta <= SATURATION_FRONT_PROGRESS_EPSILON
    )
    tracker.dwell = tracker.dwell + dt if conditions else 0.0
    tracker.saturated = tracker.dwell >= SATURATION_DWELL_TIME
    if tracker.saturated:
        metrics.saturation_events.append({
            "branch": branch,
            "time": simulation_time,
            "tip_count": tracker.tip_count,
            "low_speed_ratio": tracker.low_speed_ratio,
            "density_ratio": tracker.average_density_ratio,
            "occupancy": tracker.occupancy_ratio,
            "front_delta": tracker.front_delta,
        })
    return tracker.saturated

# =========================================================
# 14. Adaptive Shepherd election and pressure flow
# =========================================================


def adaptive_shepherd_count():
    effective_width = corridor_width - 2.0 * SHEPHERD_EDGE_MARGIN
    count = math.ceil(effective_width / SHEPHERD_TARGET_SLOT_SPACING) + 1
    return int(clamp(count, SHEPHERD_MIN_COUNT, SHEPHERD_MAX_COUNT))


def build_shepherd_slots(branch, count):
    usable_half = half_width - SHEPHERD_EDGE_MARGIN
    lateral = [0.0] if count <= 1 else [
        -usable_half + 2.0 * usable_half * index / (count - 1)
        for index in range(count)
    ]
    if branch == "UP":
        y = center_y - half_width - normal_length + SHEPHERD_BOUNDARY_WALL_CLEARANCE
        return [pygame.Vector2(center_x + value, y) for value in lateral]
    if branch == "LEFT":
        x = center_x - half_width - normal_length + SHEPHERD_BOUNDARY_WALL_CLEARANCE
        return [pygame.Vector2(x, center_y + value) for value in lateral]
    x = center_x + half_width + right_length - SHEPHERD_BOUNDARY_WALL_CLEARANCE
    return [pygame.Vector2(x, center_y + value) for value in lateral]


def reset_shepherd_roles(robots):
    for robot in robots:
        if robot.role == "SHEPHERD":
            robot.role = "NORMAL"
        robot.shepherd_anchor = None
        robot.shepherd_origin = None



def shepherd_candidates(robots, branch, required_count):
    """Return the leading robots already inside the original capture region."""
    capture_rect = early_capture_regions[branch]
    candidates = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and robot.connected_to_base
        and get_robot_region(robot.position) == branch
        and capture_rect.collidepoint(robot.position.x, robot.position.y)
    ]
    candidates.sort(
        key=lambda robot: (
            branch_progress(robot, branch),
            -robot.velocity.length_squared(),
            -robot.robot_id,
        ),
        reverse=True,
    )
    return candidates[:required_count]


def capture_region_ready_for_shepherd(robots, branch):
    required_count = adaptive_shepherd_count()
    capture_rect = early_capture_regions[branch]
    candidates = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and robot.connected_to_base
        and get_robot_region(robot.position) == branch
        and capture_rect.collidepoint(robot.position.x, robot.position.y)
    ]
    if len(candidates) < required_count:
        return False

    front_status = get_front_communication_status(robots, branch)
    return (
        front_status["connected_ratio"]
        >= RELAY_FRONT_REQUIRED_CONNECTED_RATIO
        and front_status["minimum_margin"] >= 0.0
    )


def assign_shepherd_slots(candidates, slots):
    """Assign the first-arrival candidates to the nearest remaining slots."""
    if len(candidates) < len(slots):
        return []

    unused = list(candidates[: len(slots)])
    assignment = []
    for slot in slots:
        chosen = min(
            unused,
            key=lambda robot: (
                robot.position.distance_squared_to(slot),
                robot.robot_id,
            ),
        )
        unused.remove(chosen)
        assignment.append((chosen, slot, 0.0))
    return assignment


def select_adaptive_shepherds(robots, branch, grid):
    reset_shepherd_roles(robots)
    required_count = adaptive_shepherd_count()
    slots = build_shepherd_slots(branch, required_count)
    candidates = shepherd_candidates(robots, branch, required_count)
    assignment = assign_shepherd_slots(candidates, slots)
    if len(assignment) != required_count:
        print(f"[Shepherd] insufficient candidates: {len(candidates)}/{required_count}")
        return []
    selected = []
    for robot, slot, score in assignment:
        robot.role = "SHEPHERD"
        robot.shepherd_anchor = slot.copy()
        robot.shepherd_origin = slot.copy()
        robot.velocity.update(0.0, 0.0)
        selected.append(robot)
        print(f"[Shepherd] robot={robot.robot_id}, score={score:.3f}")
    print(f"[Shepherd] adaptive count={required_count}")
    return selected


def get_shepherds(robots):
    return [robot for robot in robots if robot.role == "SHEPHERD"]


def shepherd_boundary_formed(robots):
    shepherds = get_shepherds(robots)
    return bool(shepherds) and all(
        robot.shepherd_anchor is not None
        and robot.position.distance_to(robot.shepherd_anchor) <= SHEPHERD_FORM_TOLERANCE
        for robot in shepherds
    )


def get_local_pressure_front_normals(robots, branch):
    shepherds = [
        robot
        for robot in robots
        if robot.role == "SHEPHERD" and get_robot_region(robot.position) == branch
    ]
    normals = [
        robot
        for robot in robots
        if robot.role == "NORMAL" and get_robot_region(robot.position) == branch
    ]
    if not shepherds:
        return normals
    boundary_progress = sum(branch_progress(robot, branch) for robot in shepherds) / len(shepherds)
    return [
        robot
        for robot in normals
        if boundary_progress - SHEPHERD_LOCAL_FLOW_DEPTH
        <= branch_progress(robot, branch)
        <= boundary_progress + SHEPHERD_LOCAL_FLOW_FORWARD_ALLOWANCE
    ]


def normal_backtracking_metrics(robots, branch):
    normals = get_local_pressure_front_normals(robots, branch)
    if not normals:
        return 1.0, 0.0, 0
    direction = get_backtrack_direction(branch)
    speeds = [robot.velocity.dot(direction) for robot in normals]
    moving_ratio = sum(speed >= FLOW_SPEED_THRESHOLD for speed in speeds) / len(normals)
    average_speed = sum(max(0.0, speed) for speed in speeds) / len(normals)
    return moving_ratio, average_speed, len(normals)


def release_shepherds_into_flow(robots):
    direction = get_backtrack_direction(active_branch)
    local = get_local_pressure_front_normals(robots, active_branch)
    positive = [max(0.0, robot.velocity.dot(direction)) for robot in local]
    speed = max(SHEPHERD_RELEASE_SPEED, (sum(positive) / len(positive) * 1.15) if positive else 0.0)
    speed = min(speed, MAX_SPEED * 0.45)
    released = 0
    for robot in robots:
        if robot.role != "SHEPHERD":
            continue
        robot.role = "NORMAL"
        robot.shepherd_anchor = None
        robot.shepherd_origin = None
        robot.velocity = direction * speed
        released += 1
    print(f"[Shepherd] released={released}, speed={speed:.2f}")

# =========================================================
# 15. SPH
# =========================================================


def compute_densities(robots, grid):
    self_contribution = spiky_kernel(0.0, SMOOTHING_LENGTH)
    h_sq = SMOOTHING_LENGTH**2
    for robot_i in robots:
        density = self_contribution
        for robot_j in iter_neighbor_candidates(robot_i, grid):
            if robot_i is robot_j:
                continue
            distance_sq = robot_i.position.distance_squared_to(robot_j.position)
            if distance_sq <= h_sq:
                density += spiky_kernel(math.sqrt(distance_sq), SMOOTHING_LENGTH)
        robot_i.density = max(density, EPSILON)


def compute_pressures(robots, reference_density):
    for robot in robots:
        ratio = robot.density / max(reference_density, EPSILON)
        robot.pressure = PRESSURE_GAIN * robot.density * (ratio**STIFFNESS_EXPONENT - 1.0)
        if phase == SimulationPhase.PRESSURE_PUSH and robot.role == "SHEPHERD":
            ramp = min(1.0, 0.25 + pressure_push_timer / max(PRESSURE_RAMP_TIME, EPSILON))
            robot.pressure += PRESSURE_GAIN * robot.density * SHEPHERD_PRESSURE_FACTOR * ramp


def compute_route_force(robot):
    region = get_robot_region(robot.position)
    junction_target = pygame.Vector2(center_x, center_y)
    force = pygame.Vector2()
    if robot.role in {"ANCHOR", "RELAY", "TRUNK_RELAY"}:
        return force
    independent = {
        SimulationPhase.MOVE_TO_JUNCTION,
        SimulationPhase.FINAL_JUNCTION_GATHER,
        SimulationPhase.RETURN_TO_BASE,
        SimulationPhase.DONE,
    }
    if phase not in independent and junction_anchor is not None:
        if not robot.connected_to_base or robot.received_command != phase.name:
            return force
        if phase in {
            SimulationPhase.EXPLORE_BRANCH,
            SimulationPhase.FORM_SHEPHERD_BOUNDARY,
            SimulationPhase.FILL_BEHIND_SHEPHERD,
            SimulationPhase.PRESSURE_PUSH,
            SimulationPhase.FLOW_BACKTRACK,
        } and robot.received_branch != active_branch:
            return force

    if phase == SimulationPhase.MOVE_TO_JUNCTION:
        y_distance = robot.position.y - INITIAL_INGRESS_TARGET_Y
        if y_distance > 0.0:
            scale = max(INITIAL_INGRESS_MIN_FORCE_SCALE, min(1.0, y_distance / INITIAL_INGRESS_BRAKE_DISTANCE))
            force.y = -INITIAL_INGRESS_FORCE * scale
        lane_error = robot.ingress_lane_x - robot.position.x
        force.x = clamp(INITIAL_INGRESS_LANE_GAIN * lane_error, -INITIAL_INGRESS_LANE_MAX_FORCE, INITIAL_INGRESS_LANE_MAX_FORCE)
    elif phase == SimulationPhase.EXPLORE_BRANCH:
        force = normalized_direction_toward(
            robot.position, get_branch_tip_target(active_branch)
        ) * ROUTE_FORCE * relay_motion_scale
    elif phase == SimulationPhase.FORM_SHEPHERD_BOUNDARY:
        if robot.role == "SHEPHERD":
            force = pygame.Vector2()
        elif region == active_branch:
            # Keep the original behavior: ordinary branch robots wait while the
            # selected Shepherds arrange themselves across the corridor.
            force = pygame.Vector2()
        else:
            force = normalized_direction_toward(
                robot.position,
                JUNCTION_STAGING_POSITION,
            ) * OUTLET_FORCE
    elif phase == SimulationPhase.FILL_BEHIND_SHEPHERD:
        if robot.role == "SHEPHERD":
            force = pygame.Vector2()
        elif region == active_branch:
            force = normalized_direction_toward(
                robot.position,
                get_shepherd_fill_target(active_branch),
            ) * ROUTE_FORCE * relay_motion_scale
        else:
            force = normalized_direction_toward(
                robot.position,
                JUNCTION_STAGING_POSITION,
            ) * OUTLET_FORCE
    elif phase == SimulationPhase.PRESSURE_PUSH:
        if robot.role == "NORMAL" and region == active_branch:
            force = get_backtrack_direction(active_branch) * PRESSURE_BACKTRACK_BODY_FORCE
        elif robot.role != "SHEPHERD":
            force = normalized_direction_toward(robot.position, JUNCTION_STAGING_POSITION) * OUTLET_FORCE
    elif phase == SimulationPhase.FLOW_BACKTRACK:
        target = junction_target if region == active_branch else JUNCTION_STAGING_POSITION
        force = normalized_direction_toward(robot.position, target) * FLOW_BACKTRACK_FORCE
    elif phase == SimulationPhase.JUNCTION_SWITCH:
        force = normalized_direction_toward(robot.position, JUNCTION_STAGING_POSITION) * OUTLET_FORCE
    elif phase == SimulationPhase.FINAL_JUNCTION_GATHER:
        target = junction_target if region in BRANCHES else JUNCTION_STAGING_POSITION
        force = normalized_direction_toward(robot.position, target) * FINAL_GATHER_FORCE
    elif phase == SimulationPhase.RETURN_TO_BASE:
        target = junction_target if region in BRANCHES else get_bottom_hold_point()
        force = normalized_direction_toward(robot.position, target) * OUTLET_FORCE

    if region in {"UP", "BOTTOM"}:
        force.x += CENTERING_GAIN * (center_x - robot.position.x)
    elif region in {"LEFT", "RIGHT"}:
        force.y += CENTERING_GAIN * (center_y - robot.position.y)
    return force


def compute_sph_forces(robots, grid):
    h_sq = SMOOTHING_LENGTH**2
    virtual_sq = VIRTUAL_PRESSURE_RADIUS**2
    backtrack_direction = get_backtrack_direction(active_branch)
    checked_pairs = set()
    for robot_i in robots:
        if robot_i.role in {"ANCHOR", "RELAY", "TRUNK_RELAY"}:
            robot_i.acceleration.update(0.0, 0.0)
            continue
        if robot_i.role == "SHEPHERD" and phase in {
            SimulationPhase.FORM_SHEPHERD_BOUNDARY,
            SimulationPhase.FILL_BEHIND_SHEPHERD,
            SimulationPhase.PRESSURE_PUSH,
        }:
            robot_i.acceleration.update(0.0, 0.0)
            continue

        pressure_force = pygame.Vector2()
        viscosity_force = pygame.Vector2()
        repulsion_force = pygame.Vector2()
        virtual_force = pygame.Vector2()
        cohesion_force = pygame.Vector2()
        neighbor_count = 0
        neighbor_center = pygame.Vector2()

        for robot_j in iter_neighbor_candidates(robot_i, grid):
            if robot_i is robot_j:
                continue
            pair = tuple(sorted((robot_i.robot_id, robot_j.robot_id)))
            r_ij = robot_i.position - robot_j.position
            distance_sq = r_ij.length_squared()
            if pair not in checked_pairs:
                checked_pairs.add(pair)
                distance = math.sqrt(max(distance_sq, 0.0))
                metrics.minimum_pair_distance = min(metrics.minimum_pair_distance, distance)
                if distance < ROBOT_RADIUS * 2.0:
                    metrics.safety_violations += 1

            if (
                phase == SimulationPhase.PRESSURE_PUSH
                and robot_i.role == "NORMAL"
                and robot_j.role == "SHEPHERD"
                and distance_sq <= virtual_sq
                and branch_progress(robot_i, active_branch) <= branch_progress(robot_j, active_branch) + 2.0
            ):
                distance = math.sqrt(max(distance_sq, EPSILON))
                ratio = max(0.0, 1.0 - distance / VIRTUAL_PRESSURE_RADIUS)
                ramp = min(1.0, 0.25 + pressure_push_timer / max(PRESSURE_RAMP_TIME, EPSILON))
                virtual_force += backtrack_direction * VIRTUAL_PRESSURE_FORCE * ratio**2 * ramp

            if distance_sq <= EPSILON or distance_sq > h_sq:
                continue
            neighbor_count += 1
            neighbor_center += robot_j.position
            distance = math.sqrt(distance_sq)
            gradient = spiky_gradient(r_ij, SMOOTHING_LENGTH)
            coefficient = (
                robot_i.pressure / max(robot_i.density**2, EPSILON)
                + robot_j.pressure / max(robot_j.density**2, EPSILON)
            )
            pressure_force += -coefficient * gradient
            v_ij = robot_i.velocity - robot_j.velocity
            approach = v_ij.dot(r_ij)
            if approach < 0.0:
                mu_ij = SMOOTHING_LENGTH * approach / (distance_sq + 0.01 * SMOOTHING_LENGTH**2)
                c_i_sq = (robot_i.pressure + PRESSURE_GAIN * robot_i.density) / max(robot_i.density, EPSILON)
                c_j_sq = (robot_j.pressure + PRESSURE_GAIN * robot_j.density) / max(robot_j.density, EPSILON)
                c_ij = 0.5 * (math.sqrt(max(c_i_sq, 0.0)) + math.sqrt(max(c_j_sq, 0.0)))
                mean_density = 0.5 * (robot_i.density + robot_j.density)
                pi_ij = (-VISCOSITY_XI1 * c_ij * mu_ij + VISCOSITY_XI2 * mu_ij**2) / max(mean_density, EPSILON)
                viscosity_force += -pi_ij * gradient
            if distance < SAFE_RADIUS:
                repulsion_force += REPULSION_GAIN * ((SAFE_RADIUS - distance) / SAFE_RADIUS) * (r_ij / distance)

        route_force = compute_route_force(robot_i)
        connectivity_force = compute_connectivity_force(robot_i, grid)
        pressure_phase_normal = (
            phase == SimulationPhase.PRESSURE_PUSH
            and robot_i.role == "NORMAL"
            and get_robot_region(robot_i.position) == active_branch
        )
        if 0 < neighbor_count < ISOLATION_NEIGHBOR_THRESHOLD and not pressure_phase_normal:
            local_center = neighbor_center / neighbor_count
            direction = local_center - robot_i.position
            if direction.length_squared() > EPSILON:
                ratio = (ISOLATION_NEIGHBOR_THRESHOLD - neighbor_count) / ISOLATION_NEIGHBOR_THRESHOLD
                cohesion_force = direction.normalize() * LOCAL_COHESION_GAIN * ratio
        if neighbor_count < ISOLATION_NEIGHBOR_THRESHOLD and not pressure_phase_normal:
            boost = (ISOLATION_NEIGHBOR_THRESHOLD - neighbor_count) / ISOLATION_NEIGHBOR_THRESHOLD
            route_force *= 1.0 + ISOLATION_ROUTE_BOOST * boost
        total = (
            pressure_force
            + viscosity_force
            + repulsion_force
            + virtual_force
            + cohesion_force
            + route_force
            + connectivity_force
            - DAMPING * robot_i.velocity
        )
        robot_i.acceleration = limit_vector(total, MAX_ACCELERATION)

# =========================================================
# 16. State machine
# =========================================================


def count_branch_roles(robots, branch):
    normal = shepherd = relay = 0
    for robot in robots:
        if get_robot_region(robot.position) != branch:
            continue
        if robot.role == "SHEPHERD":
            shepherd += 1
        elif robot.role == "RELAY":
            relay += 1
        else:
            normal += 1
    return normal, shepherd, relay


def update_metrics_per_frame(robots, dt):
    if base_station is not None:
        disconnected = sum(not robot.connected_to_base for robot in robots)
        metrics.disconnected_robot_seconds += disconnected * dt


def update_simulation_state(robots, dt, reference_density, spatial_grid):
    global phase, shepherd_form_timer, pressure_push_timer, flow_establish_timer
    global junction_switch_timer, final_gather_timer
    anchor = elect_junction_anchor(robots)

    if phase == SimulationPhase.MOVE_TO_JUNCTION:
        update_trunk_relay_deployment(robots, dt)
        robots_in_junction = sum(
            get_robot_region(robot.position) == "JUNCTION" and robot.role != "ANCHOR"
            for robot in robots
        )
        if (
            anchor is not None
            and anchor_deployment_ready(anchor, robots)
            and robots_in_junction >= JUNCTION_ENTRY_COUNT
        ):
            selected = choose_next_branch(anchor)
            if selected is None:
                begin_final_gather()
            else:
                saturation_tracker.reset(selected)
                phase = SimulationPhase.EXPLORE_BRANCH
                metrics.branch_events.append({"branch": selected, "started_at": simulation_time})

    elif phase == SimulationPhase.EXPLORE_BRANCH:
        update_relay_deployment(robots, dt)

        # Preserve the original timing: wait until the leading robots enter the
        # dead-end capture region. Only the required count is now width-adaptive.
        if capture_region_ready_for_shepherd(robots, active_branch):
            selected = select_adaptive_shepherds(
                robots,
                active_branch,
                spatial_grid,
            )
            if len(selected) == adaptive_shepherd_count():
                phase = SimulationPhase.FORM_SHEPHERD_BOUNDARY
                shepherd_form_timer = 0.0
                print(
                    f"[Shepherd] capture-region election: branch={active_branch}, "
                    f"count={len(selected)}"
                )

    elif phase == SimulationPhase.FORM_SHEPHERD_BOUNDARY:
        update_relay_deployment(robots, dt)
        shepherd_form_timer += dt
        if shepherd_boundary_formed(robots):
            phase = SimulationPhase.FILL_BEHIND_SHEPHERD
            saturation_tracker.reset(active_branch)
            print("[Shepherd] boundary formed; ordinary robots now fill behind it")
        elif shepherd_form_timer >= SHEPHERD_FORM_TIMEOUT:
            # Do not start pressure with an incomplete boundary. Return selected
            # robots to NORMAL and retry when the capture region is ready.
            reset_shepherd_roles(robots)
            phase = SimulationPhase.EXPLORE_BRANCH
            print("[Shepherd] boundary formation timeout; election will retry")

    elif phase == SimulationPhase.FILL_BEHIND_SHEPHERD:
        update_relay_deployment(robots, dt)
        saturated = update_dead_end_saturation(
            robots, active_branch, reference_density, dt
        )
        if saturated:
            phase = SimulationPhase.PRESSURE_PUSH
            pressure_push_timer = 0.0
            flow_establish_timer = 0.0
            metrics.pressure_events.append({
                "branch": active_branch,
                "started_at": simulation_time,
            })
            print(
                f"[Saturation] robots packed behind Shepherd boundary: "
                f"branch={active_branch}, count={saturation_tracker.tip_count}"
            )
            print("[Pressure] piston push started")

    elif phase == SimulationPhase.PRESSURE_PUSH:
        pressure_push_timer += dt
        moving_ratio, average_speed, normal_count = normal_backtracking_metrics(robots, active_branch)
        established = (
            pressure_push_timer >= SHEPHERD_MIN_PUSH_TIME
            and normal_count >= FLOW_MIN_NORMAL_COUNT
            and moving_ratio >= FLOW_RATIO_THRESHOLD
            and average_speed >= FLOW_AVERAGE_SPEED_THRESHOLD
        )
        flow_establish_timer = flow_establish_timer + dt if established else 0.0
        if (
            flow_establish_timer >= FLOW_ESTABLISH_DWELL_TIME
            or pressure_push_timer >= FLOW_FALLBACK_TIME
            or normal_count == 0
        ):
            release_shepherds_into_flow(robots)
            phase = SimulationPhase.FLOW_BACKTRACK
            if metrics.pressure_events:
                metrics.pressure_events[-1]["flow_at"] = simulation_time
                metrics.pressure_events[-1]["latency"] = pressure_push_timer
            print(f"[Pressure] flow ratio={moving_ratio:.2f}, avg={average_speed:.2f}")

    elif phase == SimulationPhase.FLOW_BACKTRACK:
        update_relay_retraction(robots, dt)
        remaining = sum(get_robot_region(robot.position) == active_branch for robot in robots)
        in_junction = sum(
            get_robot_region(robot.position) == "JUNCTION" and robot.role != "ANCHOR"
            for robot in robots
        )
        if remaining <= BRANCH_CLEAR_LIMIT and in_junction >= JUNCTION_SWITCH_COUNT and not get_active_branch_relays(robots):
            complete_active_branch(anchor, active_branch)
            phase = SimulationPhase.JUNCTION_SWITCH
            junction_switch_timer = 0.0

    elif phase == SimulationPhase.JUNCTION_SWITCH:
        junction_switch_timer += dt
        if junction_switch_timer >= JUNCTION_SWITCH_DWELL_TIME:
            selected = choose_next_branch(anchor)
            if selected is None:
                begin_final_gather()
            else:
                saturation_tracker.reset(selected)
                phase = SimulationPhase.EXPLORE_BRANCH
                metrics.branch_events.append({"branch": selected, "started_at": simulation_time})

    elif phase == SimulationPhase.FINAL_JUNCTION_GATHER:
        stragglers = sum(get_robot_region(robot.position) in BRANCHES for robot in robots)
        gather_ready = (
            stragglers == 0
            and not get_relays(robots)
            and not get_shepherds(robots)
            and sum(robot.connected_to_base for robot in robots) == len(robots)
        )
        final_gather_timer = final_gather_timer + dt if gather_ready else 0.0
        if final_gather_timer >= FINAL_GATHER_DWELL_TIME:
            begin_final_return(anchor, robots)

    elif phase == SimulationPhase.RETURN_TO_BASE:
        in_bottom = sum(get_robot_region(robot.position) == "BOTTOM" for robot in robots)
        special = sum(robot.role in {"ANCHOR", "RELAY", "TRUNK_RELAY", "SHEPHERD"} for robot in robots)
        if in_bottom >= RETURN_BOTTOM_TARGET_COUNT and special == 0:
            phase = SimulationPhase.DONE
            metrics.completion_time = simulation_time
            print(f"[DFS] done, robots={in_bottom}/{len(robots)}")
            save_experiment_logs(robots, "DONE")

# =========================================================
# 17. Initialization
# =========================================================


def reset_dfs_state():
    global phase, active_branch, branch_states, branch_order_plan
    global previous_branch_direction, junction_anchor, simulation_time
    global junction_switch_timer, final_gather_timer, shepherd_form_timer
    global pressure_push_timer, flow_establish_timer, communication_sequence
    global last_message_signature, relay_slots, relay_deploy_cooldown
    global relay_retract_cooldown, relay_retract_clear_timer, relay_motion_scale
    global trunk_relay_slots, trunk_relay_deploy_cooldown, base_station
    global metrics
    phase = SimulationPhase.MOVE_TO_JUNCTION
    active_branch = "UP"
    branch_states = {branch: "UNVISITED" for branch in BRANCHES}
    branch_order_plan = []
    previous_branch_direction = pygame.Vector2(0.0, -1.0)
    junction_anchor = None
    simulation_time = 0.0
    junction_switch_timer = final_gather_timer = shepherd_form_timer = 0.0
    pressure_push_timer = flow_establish_timer = 0.0
    communication_sequence = 0
    last_message_signature = None
    relay_slots = []
    relay_deploy_cooldown = relay_retract_cooldown = relay_retract_clear_timer = 0.0
    relay_motion_scale = 1.0
    trunk_relay_slots = []
    trunk_relay_deploy_cooldown = 0.0
    base_station = BaseStation(BASE_POSITION)
    initialize_trunk_relay_plan()
    saturation_tracker.reset()
    metrics = ExperimentMetrics()


def initialize_simulation():
    reset_dfs_state()
    robots = create_grid_robots(ROBOT_COUNT) if SPAWN_MODE == "grid" else create_random_robots(ROBOT_COUNT)
    if not robots:
        raise RuntimeError("No robots were created.")
    grid = build_spatial_grid(robots)
    compute_densities(robots, grid)
    mean_density = sum(robot.density for robot in robots) / len(robots)
    reference_density = mean_density * 0.70
    color_reference_density = mean_density * 0.68
    update_communication_system(robots, grid)
    print(f"robots={len(robots)}, mean_density={mean_density:.6f}, rho0={reference_density:.6f}")
    return robots, reference_density, color_reference_density


robots, reference_density, color_reference_density = initialize_simulation()

# =========================================================
# 18. Main loop
# =========================================================

running = True
paused = False
show_density_color = False
show_regions = True
show_comm_links = SHOW_COMM_LINKS_DEFAULT
communication_frame_counter = 0

while running:
    raw_dt = max(clock.tick(FPS) / 1000.0, 1.0 / 240.0)
    frame_dt = min(
        raw_dt,
        INITIAL_INGRESS_MAX_DT if phase == SimulationPhase.MOVE_TO_JUNCTION else NORMAL_PHYSICS_MAX_DT,
    )

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:
                robots, reference_density, color_reference_density = initialize_simulation()
            elif event.key == pygame.K_d:
                show_density_color = not show_density_color
            elif event.key == pygame.K_v:
                show_regions = not show_regions
            elif event.key == pygame.K_c:
                show_comm_links = not show_comm_links
            elif event.key == pygame.K_ESCAPE:
                running = False

    spatial_grid = build_spatial_grid(robots)
    if not paused:
        simulation_time += frame_dt
        communication_frame_counter += 1
        if communication_frame_counter % COMM_UPDATE_INTERVAL_FRAMES == 0:
            update_communication_system(robots, spatial_grid)
        substep_dt = frame_dt / SUBSTEPS
        for _ in range(SUBSTEPS):
            spatial_grid = build_spatial_grid(robots)
            compute_densities(robots, spatial_grid)
            compute_pressures(robots, reference_density)
            compute_sph_forces(robots, spatial_grid)
            for robot in robots:
                robot.update(substep_dt)
        update_anchor_entry_records(robots, simulation_time)
        # Rebuild immediately after role changes such as Anchor or Relay election.
        spatial_grid = build_spatial_grid(robots)
        update_communication_system(robots, spatial_grid)
        update_simulation_state(robots, frame_dt, reference_density, spatial_grid)
        update_metrics_per_frame(robots, frame_dt)
    else:
        update_communication_system(robots, spatial_grid)
        compute_densities(robots, spatial_grid)
        compute_pressures(robots, reference_density)

    screen.fill(BACKGROUND_COLOR)
    pygame.draw.polygon(screen, FLOOR_COLOR, cross_points)
    pygame.draw.polygon(screen, WALL_COLOR, cross_points, width=5)

    if show_regions:
        pygame.draw.rect(screen, JUNCTION_COLOR, junction_rect, width=2)
        pygame.draw.rect(screen, ANCHOR_COLOR, anchor_election_rect, width=1)
        pygame.draw.circle(screen, ANCHOR_COLOR, ANCHOR_PARK_POSITION, 4, width=1)
        pygame.draw.circle(screen, BASE_COLOR, BASE_POSITION, 7, width=2)
        pygame.draw.rect(
            screen,
            SHEPHERD_COLOR,
            early_capture_regions[active_branch],
            width=1,
        )
        pygame.draw.rect(
            screen,
            END_REGION_COLOR,
            get_saturation_rect(active_branch),
            width=1,
        )
        for branch, rect in dead_end_regions.items():
            pygame.draw.rect(
                screen,
                END_REGION_COLOR if branch == active_branch else (175, 175, 175),
                rect,
                width=2,
            )
        for robot in get_shepherds(robots):
            if robot.shepherd_anchor is not None:
                pygame.draw.circle(screen, SHEPHERD_COLOR, robot.shepherd_anchor, 3, width=1)
        draw_trunk_relay_plan(screen, robots)
        draw_relay_plan(screen, robots)

    pygame.draw.circle(screen, JUNCTION_COLOR, (center_x, center_y), 5)
    pygame.draw.circle(screen, BASE_COLOR, BASE_POSITION, 6)
    if show_comm_links:
        draw_communication_links(screen, robots)
    for robot in robots:
        robot.draw(screen, color_reference_density, show_density_color)

    normal_count, shepherd_count, relay_count = count_branch_roles(robots, active_branch)
    communication_stats = get_communication_stats(robots)
    front_comm = get_front_communication_status(robots, active_branch)
    hud_lines = [
        "Base-rooted SPH DFS: trunk + adaptive Shepherd + saturation piston",
        f"FPS={clock.get_fps():.1f} | robots={len(robots)} | phase={phase.name}",
        f"Anchor={junction_anchor.robot_id if junction_anchor else '-'} | score={junction_anchor.anchor_election_score:.3f}" if junction_anchor else "Anchor=-",
        f"Branch={active_branch if phase not in {SimulationPhase.MOVE_TO_JUNCTION, SimulationPhase.RETURN_TO_BASE, SimulationPhase.DONE} else '-'}",
        f"Order={' > '.join(branch_order_plan) if branch_order_plan else '-'}",
        f"States: U={branch_states['UP']} L={branch_states['LEFT']} R={branch_states['RIGHT']}",
        f"Base comm={communication_stats['connected']}/{len(robots)} | hop={communication_stats['max_hop']} | margin={communication_stats['margin']:.1f}",
        f"Base direct={communication_stats['direct']} | Anchor linked={communication_stats['anchor_connected']} | trunk={len(get_trunk_relays(robots))}/{len(trunk_relay_slots)}",
        f"Front comm ratio={front_comm['connected_ratio']:.2f} | relay need={front_comm['needs_relay']}",
        f"Relays={len(get_relays(robots))} | motion scale={relay_motion_scale:.2f}",
        f"Branch robots normal={normal_count} relay={relay_count} shepherd={shepherd_count}",
        f"Saturation: tip={saturation_tracker.tip_count} slow={saturation_tracker.low_speed_ratio:.2f}",
        f"density={saturation_tracker.average_density_ratio:.2f} occupancy={saturation_tracker.occupancy_ratio:.2f}",
        f"front_delta={saturation_tracker.front_delta:.2f} dwell={saturation_tracker.dwell:.2f} saturated={saturation_tracker.saturated}",
        f"Shepherd target={adaptive_shepherd_count()} | formed={shepherd_boundary_formed(robots)} | pressure t={pressure_push_timer:.2f}",
        f"Distance total={sum(robot.total_distance for robot in robots):.0f} | disconnect robot-s={metrics.disconnected_robot_seconds:.1f}",
    ]
    for index, text in enumerate(hud_lines):
        screen.blit(small_font.render(text, True, TEXT_COLOR), (15, 12 + index * 21))

    control_text = font.render(
        "SPACE pause | R reset | D density | V regions | C communication | ESC quit",
        True,
        TEXT_COLOR,
    )
    screen.blit(control_text, (15, SCREEN_HEIGHT - 30))
    pygame.display.flip()

if not metrics.saved:
    save_experiment_logs(robots, "USER_EXIT")
pygame.quit()
sys.exit()
