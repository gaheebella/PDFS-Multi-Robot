"""SPH-based single-junction Physical DFS prototype.

Implemented research components
-------------------------------
1. NORMAL-to-NORMAL local voting for distributed Junction decisions.
2. Fixed RIGHT -> UP -> LEFT priority agreed through peer consensus.
3. Dead-end saturation detection using speed, density, occupancy,
   front stagnation, and dwell time.
4. Width-adaptive Shepherd count and peer-auction role self-election.
5. Pressure-dominant SPH motion with only a weak directional bias.
6. Fixed Base-rooted LOS communication with reactive tail Breadcrumb relays.
7. The original dead-end first-arrival Shepherd selection timing is retained,
   but the Shepherd count is computed from corridor width.
8. Pressure starts only after the ordinary robots saturate behind the formed
   Shepherd boundary; Breadcrumb release logic and final gathering remain.

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

SCREEN_WIDTH = 1380
SCREEN_HEIGHT = 700
SIMULATION_VIEW_WIDTH = 840
HUD_PANEL_X = 850
HUD_PANEL_MARGIN = 14
HUD_PANEL_WIDTH = SCREEN_WIDTH - HUD_PANEL_X
HUD_PANEL_COLOR = (244, 247, 252)
HUD_PANEL_BORDER_COLOR = (196, 204, 218)
FPS = 60
SUBSTEPS = 1

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(
    "Pressure-Driven SPH | Distributed Decisions + Breadcrumb Relays"
)
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)
small_font = pygame.font.SysFont(None, 19)
hud_font = pygame.font.SysFont(None, 18)

BACKGROUND_COLOR = (248, 249, 252)
FLOOR_COLOR = (235, 239, 246)
WALL_COLOR = (96, 106, 124)
TEXT_COLOR = (58, 67, 82)
ROBOT_BASE_COLOR = (44, 92, 118)
SHEPHERD_COLOR = (106, 70, 150)
ANCHOR_COLOR = (48, 118, 82)
BASE_COLOR = (44, 72, 120)
TRUNK_RELAY_COLOR = (142, 82, 60)
RELAY_COLOR = (168, 112, 44)
ROBOT_OUTLINE_COLOR = (22, 42, 58)
DISCONNECTED_FILL_COLOR = (180, 92, 92)
PROXY_POINT_COLORS = {
    "UP": (46, 76, 130),
    "LEFT": (92, 54, 120),
    "RIGHT": (132, 86, 36),
}
RELAY_SLOT_COLOR = (196, 129, 65)
JUNCTION_COLOR = (153, 164, 181)
END_REGION_COLOR = (224, 171, 115)
COMM_LINK_SAFE_COLOR = (132, 190, 158)
COMM_LINK_WARNING_COLOR = (226, 177, 96)
COMM_LINK_DANGER_COLOR = (214, 103, 103)
DISCONNECTED_COLOR = (205, 96, 96)
ARTIFICIAL_WALL_COLOR = (220, 62, 62)
ARTIFICIAL_WALL_WIDTH = 3

# Branch colour identity is shared by the physical branch, its proxy subregion,
# projected proxy particles, branch labels, and branch boundaries.  The same
# RGB hue is drawn with different alpha values rather than using unrelated
# colours for the map and the analytical proxy partition.
BRANCH_COLORS = {
    "UP": (35, 112, 238),       # vivid blue
    "LEFT": (171, 63, 204),    # vivid purple
    "RIGHT": (242, 126, 32),   # vivid orange
}
BRANCH_PROXY_ALPHA = 112
BRANCH_FLOOR_ALPHA = 48
BRANCH_ASSIGNMENT_ALPHA = 220
BRANCH_BOUNDARY_WIDTH = 1


# =========================================================
# 2. Cross map
# =========================================================

center_x = 400
center_y = 350
MAP_SCALE = 0.55
corridor_width = round(120 * MAP_SCALE)
half_width = corridor_width // 2
normal_length = round(180 * MAP_SCALE)
right_length = normal_length * 2
base_length = round(normal_length * (2.0 / 3.0))

cross_points = [
    (center_x - half_width, center_y - half_width - normal_length),
    (center_x + half_width, center_y - half_width - normal_length),
    (center_x + half_width, center_y - half_width),
    (center_x + half_width + right_length, center_y - half_width),
    (center_x + half_width + right_length, center_y + half_width),
    (center_x + half_width, center_y + half_width),
    (center_x + half_width, center_y + half_width + base_length),
    (center_x - half_width, center_y + half_width + base_length),
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

ANCHOR_REGION_SIZE = round(70 * MAP_SCALE)
anchor_election_rect = pygame.Rect(
    center_x - ANCHOR_REGION_SIZE // 2,
    center_y - ANCHOR_REGION_SIZE // 2,
    ANCHOR_REGION_SIZE,
    ANCHOR_REGION_SIZE,
)

ANCHOR_PARK_POSITION = pygame.Vector2(
    center_x - 25 * MAP_SCALE,
    center_y - 25 * MAP_SCALE,
)
# Fixed communication root. It remains at the lower entrance throughout exploration.
BASE_POSITION = pygame.Vector2(
    center_x - 25 * MAP_SCALE,
    center_y + half_width + base_length - 14 * MAP_SCALE,
)
BASE_COMPRESSION_CENTER = pygame.Vector2(center_x, BASE_POSITION.y)
JUNCTION_STAGING_POSITION = pygame.Vector2(
    center_x + 10 * MAP_SCALE,
    center_y + 10 * MAP_SCALE,
)

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
    base_length,
)

END_REGION_DEPTH = round(48 * MAP_SCALE)

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
EARLY_CAPTURE_DEPTH = round(34 * MAP_SCALE)

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
FIXED_BRANCH_ORDER = ("RIGHT", "UP", "LEFT")
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
active_branch = FIXED_BRANCH_ORDER[0]
branch_states = {branch: "UNVISITED" for branch in BRANCHES}
branch_order_plan: list[str] = []
branch_gate_states = {branch: "OPEN" for branch in BRANCHES}
distributed_consensus_branch: Optional[str] = None
transfer_branch: Optional[str] = None
final_base_transfer_active = False
previous_branch_direction = pygame.Vector2(0.0, -1.0)  # incoming from BASE

junction_anchor: Optional["Robot"] = None
simulation_time = 0.0
junction_switch_timer = 0.0
final_gather_timer = 0.0
shepherd_form_timer = 0.0
pressure_push_timer = 0.0
flow_establish_timer = 0.0
shepherd_flow_timer = 0.0
shepherd_flow_start_depth = 0.0
viscoelastic_step = 0
viscoelastic_rest_lengths: dict[tuple[int, int], float] = {}
viscoelastic_last_seen: dict[tuple[int, int], int] = {}

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

# HydroSwarm-inspired Junction proxy partition state.  The partition is used
# only to score one DFS child at a time; robots are never dispatched to several
# branches simultaneously.
last_proxy_partition: dict[tuple[int, int], str] = {}
last_proxy_cell_centers: dict[tuple[int, int], pygame.Vector2] = {}
last_proxy_mass_stats: dict[str, dict] = {}
last_proxy_robot_assignment: dict[int, str] = {}
last_proxy_candidates: tuple[str, ...] = ()
# Candidate-by-candidate virtual SPH rollout results from the latest Junction
# decision.  They are display/logging data only and never alter real robots.
last_flow_rollout_scores: dict[str, dict] = {}

# =========================================================
# 4. Physics and control parameters
# =========================================================

ROBOT_COUNT = 352
BASE_RESERVE_HOLD_GAIN = 42.0
SPAWN_MODE = "grid"
ROBOT_RADIUS = 2.0 * MAP_SCALE
GRID_SPACING = 5.75 * MAP_SCALE
GRID_ROW_SPACING = 5.25 * MAP_SCALE

SMOOTHING_LENGTH = 28.0 * MAP_SCALE
PRESSURE_GAIN = 2800.0
SPH_MOTION_PRESSURE_BOOST = 2.20
SHEPHERD_PACKED_PRESSURE_BOOST = 5.00
STIFFNESS_EXPONENT = 0.5
VISCOSITY_XI1 = 0.9
VISCOSITY_XI2 = 1.2
MOTION_SPEED_MULTIPLIER = 3.0
DAMPING = 3.0
SAFE_RADIUS = 7.5 * MAP_SCALE
REPULSION_GAIN = 260.0
NORMAL_EQUILIBRIUM_SCALE = 1.20
# Physical-robot command conditioning.  Longitudinal motion remains fast,
# while high-frequency SPH reversals and corridor-crossing zigzags are damped.
ACCELERATION_FILTER_ALPHA = 0.18
CORRIDOR_LATERAL_VELOCITY_DAMPING = 12.0
SPH_PRESSURE_FORCE_LIMIT = 200.0 * MOTION_SPEED_MULTIPLIER
SPH_VISCOSITY_FORCE_LIMIT = 90.0 * MOTION_SPEED_MULTIPLIER
VISCOELASTIC_LINK_RADIUS = SAFE_RADIUS * 1.45
VISCOELASTIC_REST_MIN = ROBOT_RADIUS * 2.20
VISCOELASTIC_REST_MAX = SAFE_RADIUS * 1.65
VISCOELASTIC_ELASTIC_GAIN = 42.0
VISCOELASTIC_DASHPOT_GAIN = 8.0
VISCOELASTIC_REST_RELAXATION = 0.85
VISCOELASTIC_EQUILIBRIUM_ADAPTATION = 4.0
VISCOELASTIC_VELOCITY_CONSENSUS_GAIN = 6.0
VISCOELASTIC_FORCE_LIMIT = 75.0 * MOTION_SPEED_MULTIPLIER
VISCOELASTIC_LINK_STALE_STEPS = 3
COMPRESSION_RELEASE_DENSITY_RATIO = 1.20
COMPRESSION_RELEASE_RADIUS = (
    SAFE_RADIUS * NORMAL_EQUILIBRIUM_SCALE * 1.15
)
COMPRESSION_RELEASE_GAIN = 120.0
COMPRESSION_RELEASE_FORCE_LIMIT = 180.0 * MOTION_SPEED_MULTIPLIER
COMPRESSION_RELEASE_RAMP_TIME = 0.35
PRESSURE_POLICY_VERSION = "SPH_SELF_EXPANSION_V1"
VISCOELASTIC_MODEL_VERSION = "KELVIN_VOIGT_SPH_V2_SPACED"
MOTION_POLICY_VERSION = "FAST_STABLE_COMMAND_V3"

ROUTE_FORCE = 22.0 * MOTION_SPEED_MULTIPLIER
OUTLET_FORCE = 24.0 * MOTION_SPEED_MULTIPLIER
FLOW_BACKTRACK_FORCE = 46.0 * MOTION_SPEED_MULTIPLIER
FINAL_GATHER_FORCE = 58.0 * MOTION_SPEED_MULTIPLIER
PRESSURE_BACKTRACK_BODY_FORCE = 8.0 * MOTION_SPEED_MULTIPLIER
CENTERING_GAIN = 1.2

MAX_SPEED = 78.0 * MOTION_SPEED_MULTIPLIER
MAX_ACCELERATION = 300.0 * MOTION_SPEED_MULTIPLIER
PRESSURE_PUSH_MAX_SPEED = 42.0 * MOTION_SPEED_MULTIPLIER
FLOW_BACKTRACK_MAX_SPEED = 52.0 * MOTION_SPEED_MULTIPLIER
EPSILON = 1e-8

INITIAL_INGRESS_FORCE = 16.0 * MOTION_SPEED_MULTIPLIER
BASE_COMPRESSION_DURATION = 0.90
BASE_EXPANSION_BOOST_DURATION = 1.60
BASE_COMPRESSION_FORCE = 90.0 * MOTION_SPEED_MULTIPLIER
BASE_COMPRESSION_PRESSURE_SCALE = 0.35
BASE_EXPANSION_PRESSURE_SCALE = 2.80
BASE_EXPANSION_RAMP_FRACTION = 0.35
INITIAL_INGRESS_LANE_GAIN = 1.0
INITIAL_INGRESS_LANE_MAX_FORCE = 20.0
INITIAL_INGRESS_TARGET_Y = center_y + 10.0 * MAP_SCALE
INITIAL_INGRESS_BRAKE_DISTANCE = 52.0 * MAP_SCALE
INITIAL_INGRESS_MIN_FORCE_SCALE = 0.18
INITIAL_INGRESS_MAX_DT = 0.04
INITIAL_SAFE_MAX_SPEED = 36.0 * MOTION_SPEED_MULTIPLIER
INITIAL_JUNCTION_MAX_SPEED = 20.0 * MOTION_SPEED_MULTIPLIER
INITIAL_SAFE_MAX_ACCELERATION = 170.0 * MOTION_SPEED_MULTIPLIER
INITIAL_JUNCTION_SOFT_WALL_DEPTH = 18.0 * MAP_SCALE
INITIAL_JUNCTION_SOFT_WALL_FORCE = 120.0 * MOTION_SPEED_MULTIPLIER
INITIAL_JUNCTION_SOFT_WALL_DAMPING = 20.0
INITIAL_IMPULSE_POLICY_VERSION = "SOFT_RAMP_BRAKE_V1"

RETURN_EGRESS_FORCE = 42.0 * MOTION_SPEED_MULTIPLIER
RETURN_LANE_GAIN = 1.15
RETURN_LANE_MAX_FORCE = 22.0
RETURN_BRAKE_DISTANCE = 34.0 * MAP_SCALE
RETURN_MIN_FORCE_SCALE = 0.20
RETURN_TRUNK_RETRACT_DWELL = 0.55
RETURN_TRUNK_RELEASE_INITIAL_SPEED = 12.0
RETURN_TRUNK_READY_BOTTOM_TOLERANCE = 2
RETURN_TRUNK_READY_CONNECTED_RATIO = 0.97
RETURN_TRUNK_FORCE_RELEASE_TIMEOUT = 2.50
NORMAL_PHYSICS_MAX_DT = 0.05

ISOLATION_NEIGHBOR_THRESHOLD = 4
ISOLATION_ROUTE_BOOST = 1.1
LOCAL_COHESION_GAIN = 20.0

JUNCTION_ENTRY_COUNT = 18
ANCHOR_MOVE_SPEED = 42.0
ANCHOR_POSITION_TOLERANCE = 2.5 * MAP_SCALE
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
ANCHOR_LOCAL_COMM_RANGE = 56.0 * MAP_SCALE

# Dead-end saturation
SATURATION_MIN_TIP_ROBOTS = 18
SATURATION_LOW_SPEED_THRESHOLD = 4.0
SATURATION_LOW_SPEED_RATIO = 0.65
SATURATION_DENSITY_RATIO = 1.02
SATURATION_OCCUPANCY_RATIO = 0.16
SATURATION_FRONT_WINDOW = 0.20
SATURATION_FRONT_PROGRESS_EPSILON = 2.2 * MAP_SCALE
SATURATION_DWELL_TIME = 0.16
SATURATION_CELL_SIZE = 8.0 * MAP_SCALE
# A highly compressed pack keeps moving because the fill and SPH forces are
# intentionally strong.  In that case, density/count/coverage are sufficient
# evidence of a piston-ready pack and low velocity is not required.
SATURATION_PACKED_MIN_TIP_ROBOTS = 27
SATURATION_PACKED_ROBOTS_PER_SHEPHERD = 5.0
SATURATION_PACKED_DENSITY_RATIO = 1.08
SATURATION_PACKED_OCCUPANCY_RATIO = 0.30
SATURATION_PACKED_DWELL_TIME = 0.10
SATURATION_POLICY_VERSION = "PACKED_GEOMETRY_V2"

# Width-adaptive Shepherd count. Selection timing remains the same as the
# original code: leading robots are selected inside early_capture_regions.
SHEPHERD_BOUNDARY_WALL_CLEARANCE = 14.0 * MAP_SCALE
SHEPHERD_FILL_GAP = 38.0 * MAP_SCALE
SHEPHERD_FILL_REGION_DEPTH = 62.0 * MAP_SCALE
SHEPHERD_PACKED_EQUILIBRIUM_SCALE = 0.46
JUNCTION_TAIL_EQUILIBRIUM_SCALE = 1.55
SHEPHERD_FILL_COMPRESSION_PRESSURE_SCALE = 0.22
SHEPHERD_FILL_FORCE_MULTIPLIER = 3.0
SHEPHERD_MIN_COUNT = 5
SHEPHERD_MAX_COUNT = 14
SHEPHERD_EDGE_MARGIN = 12.0 * MAP_SCALE
SHEPHERD_TARGET_SLOT_SPACING = 12.5 * MAP_SCALE

SHEPHERD_FORM_SPEED = 50.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_RELEASE_SPEED = 32.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_FORM_TOLERANCE = 3.0 * MAP_SCALE
SHEPHERD_FORM_TIMEOUT = 1.25

# Continuous virtual Shepherd curtain.  The selected robots still form the
# visible SPH boundary, but this full-width virtual plane becomes active
# immediately after election.  It closes the temporary gaps between moving
# Shepherds so ordinary robots cannot leak toward the dead-end wall.
SHEPHERD_CURTAIN_CLEARANCE = max(ROBOT_RADIUS * 2.5, 6.0 * MAP_SCALE)
SHEPHERD_CURTAIN_INTERACTION_DEPTH = 24.0 * MAP_SCALE
SHEPHERD_CURTAIN_FORCE = 180.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_CURTAIN_MAX_FORCE = 220.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_CURTAIN_VELOCITY_DAMPING = 12.0
SHEPHERD_CURTAIN_RECOVERY_SPEED = 6.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_CURTAIN_DRAW_HALF_WIDTH = 3

# Piston motion: Shepherd boundary advances toward the parent junction.
SHEPHERD_PISTON_SPEED = 8.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_PISTON_MAX_TRAVEL = 42.0 * MAP_SCALE
SHEPHERD_LINE_BACKTRACK_SPEED = 12.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_SPEED_RAMP_TIME = 0.55
SHEPHERD_JUNCTION_DEPTH_TOLERANCE = max(ROBOT_RADIUS, 0.75)
SHEPHERD_JUNCTION_RELEASE_INSET = max(
    ROBOT_RADIUS * 2.5,
    2.0 * MAP_SCALE,
)
SHEPHERD_JUNCTION_RELEASE_SPEED = 12.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_POLICY_VERSION = "DENSITY_RELEASE_CURTAIN_V3"
SHEPHERD_PRESSURE_FACTOR = 4.0
VIRTUAL_PRESSURE_RADIUS = 60.0 * MAP_SCALE
VIRTUAL_PRESSURE_FORCE = 110.0
SHEPHERD_VIRTUAL_FORCE_LIMIT = 140.0 * MOTION_SPEED_MULTIPLIER
PRESSURE_RAMP_TIME = 0.8

SHEPHERD_LOCAL_FLOW_DEPTH = 58.0 * MAP_SCALE
SHEPHERD_LOCAL_FLOW_FORWARD_ALLOWANCE = 6.0 * MAP_SCALE
SHEPHERD_MIN_PUSH_TIME = 0.20
FLOW_SPEED_THRESHOLD = 1.5
FLOW_RATIO_THRESHOLD = 0.45
FLOW_AVERAGE_SPEED_THRESHOLD = 1.8
FLOW_ESTABLISH_DWELL_TIME = 0.12
FLOW_MIN_NORMAL_COUNT = 6
FLOW_FALLBACK_TIME = 1.25
BRANCH_CLEAR_LIMIT = 1

# Communication
COMM_RANGE = 54.0 * MAP_SCALE
COMM_LOS_SAMPLE_SPACING = 6.0 * MAP_SCALE
COMM_LOS_CLEARANCE = 0.0
COMM_SAFE_DISTANCE = 40.0 * MAP_SCALE
COMM_BARRIER_START = COMM_RANGE * 0.84
COMM_RECOVERY_RANGE = 84.0 * MAP_SCALE
COMM_RECOVERY_GAIN = 0.65
SHOW_COMM_LINKS_DEFAULT = False
COMM_UPDATE_INTERVAL_FRAMES = 1
ANCHOR_LINK_WARNING_DISTANCE = COMM_SAFE_DISTANCE * 0.82
ANCHOR_LINK_STOP_DISTANCE = COMM_RANGE * 0.90
ANCHOR_MIN_DIRECT_NEIGHBORS = 1
ANCHOR_READY_DIRECT_NEIGHBORS = 1

# Permanent Base-to-Junction trunk relays
TRUNK_RELAY_SPACING = 30.0 * MAP_SCALE
TRUNK_RELAY_SELECTION_RADIUS = 50.0 * MAP_SCALE
TRUNK_RELAY_DEPLOY_LOOKAHEAD = 12.0 * MAP_SCALE
TRUNK_RELAY_DEPLOY_COOLDOWN = 0.08

# Reactive Breadcrumb relays. No slot is planned in advance. Once the moving
# NORMAL tail has passed the latest breadcrumb far enough to threaten its
# communication margin, that tail robot stops exactly where it is.
BREADCRUMB_SPACING = COMM_SAFE_DISTANCE * 0.55
BREADCRUMB_DEPLOY_DISTANCE = COMM_SAFE_DISTANCE * 0.50
BREADCRUMB_FRONT_CLEARANCE = 10.0 * MAP_SCALE
BREADCRUMB_MIN_TRAVEL = 12.0 * MAP_SCALE
BREADCRUMB_DEPLOY_COOLDOWN = 0.08
BREADCRUMB_GUARD_PER_RELAY = 4
BREADCRUMB_GUARD_DEPTH = COMM_SAFE_DISTANCE * 0.85

# Branch relay
RELAY_SPACING = 30.0 * MAP_SCALE
RELAY_DEPLOY_LOOKAHEAD = 12.0 * MAP_SCALE
RELAY_SELECTION_RADIUS = 52.0 * MAP_SCALE
RELAY_END_CLEARANCE = 24.0 * MAP_SCALE
RELAY_LANE_MARGIN = 22.0 * MAP_SCALE
RELAY_MOVE_SPEED = 125.0 * MOTION_SPEED_MULTIPLIER
RELAY_POSITION_TOLERANCE = 2.5 * MAP_SCALE
RELAY_DEPLOY_COOLDOWN = 0.10
RELAY_PASS_MARGIN = 6.0 * MAP_SCALE
RELAY_RETRACT_DWELL_TIME = 0.40
RELAY_RETRACT_COOLDOWN = 0.45
RELAY_RELEASE_SPEED = 16.0 * MOTION_SPEED_MULTIPLIER
RELAY_FORMING_SPEED_SCALE = 0.40
RELAY_WAIT_SPEED_SCALE = 0.18
RELAY_DEPLOY_MARGIN = 5.0 * MAP_SCALE
RELAY_FRONT_FRACTION = 0.20
RELAY_FRONT_MIN_COUNT = 10
RELAY_FRONT_REQUIRED_CONNECTED_RATIO = 0.75

# Peer-to-peer branch consensus among NORMAL robots near the Junction.
DISTRIBUTED_VOTE_MIN_ROBOTS = 8
DISTRIBUTED_VOTE_QUORUM_RATIO = 0.60
DISTRIBUTED_VOTE_NEIGHBOR_RANGE = COMM_RANGE

# Proxy-Region-Based Flow-Preserving SPH-Aware DFS ordering.
# Structural complete-exploration loss is handled lexicographically first.
# Each candidate is evaluated only with the mobile robots assigned to its
# demand-constrained proxy subregion plus SPH-support boundary context.
# Therefore proxy mass, regional flow and regional disturbance are primary
# decision terms rather than a weak display-only prior.
BRANCH_COST_PREDICTED_FLOW_REWARD = 0.24
BRANCH_COST_DENSITY_DISTURBANCE_WEIGHT = 0.11
BRANCH_COST_VELOCITY_DISTURBANCE_WEIGHT = 0.10
BRANCH_COST_WALL_RISK_WEIGHT = 0.07
BRANCH_COST_COLLISION_RISK_WEIGHT = 0.07
BRANCH_COST_ROLLOUT_COMM_WEIGHT = 0.09
BRANCH_COST_RELAY_WEIGHT = 0.07
BRANCH_COST_LAMBDA_MODE_WEIGHT = 0.04
BRANCH_COST_STABILIZATION_WEIGHT = 0.04

BRANCH_COST_TRANSPORT_WEIGHT = 0.08
BRANCH_COST_PROXY_MASS_WEIGHT = 0.12
BRANCH_COST_SHAPE_WEIGHT = 0.05
BRANCH_COST_FLOW_PRIOR_WEIGHT = 0.06
BRANCH_COST_CONGESTION_WEIGHT = 0.08
BRANCH_COST_BACKTRACK_WEIGHT = 0.04
BRANCH_COST_SWITCH_WEIGHT = 0.04

# Candidate-specific short virtual rollout.  It is evaluated only at Junction
# decisions, so several candidates can be tested without affecting frame rate.
FLOW_ROLLOUT_HORIZON = 0.50
FLOW_ROLLOUT_DT = 0.05
FLOW_ROLLOUT_STEPS = max(1, int(round(FLOW_ROLLOUT_HORIZON / FLOW_ROLLOUT_DT)))
FLOW_ROLLOUT_MAX_ROBOTS = 190
FLOW_ROLLOUT_TARGET_DEPTH = 54.0 * MAP_SCALE
FLOW_ROLLOUT_ROUTE_GAIN = ROUTE_FORCE * 0.82
FLOW_ROLLOUT_VALVE_GAIN = 92.0
FLOW_ROLLOUT_GATE_SIGMA = 72.0 * MAP_SCALE
FLOW_ROLLOUT_REFERENCE_SPEED = 28.0
FLOW_ROLLOUT_MAX_SPEED = MAX_SPEED * 0.55
FLOW_ROLLOUT_MAX_ACCELERATION = MAX_ACCELERATION * 0.70
FLOW_ROLLOUT_WALL_CLEARANCE = 7.0 * MAP_SCALE
FLOW_ROLLOUT_COLLISION_DISTANCE = SAFE_RADIUS
FLOW_ROLLOUT_DENSITY_NORMALIZER = 0.35
FLOW_ROLLOUT_VELOCITY_NORMALIZER = FLOW_ROLLOUT_REFERENCE_SPEED

# Adaptive stiffness used both in virtual rollouts and in the real branch-entry
# phase.  A large turn temporarily lowers lambda; it then recovers smoothly.
STIFFNESS_EXPONENT_RIGID = STIFFNESS_EXPONENT
STIFFNESS_EXPONENT_SOFT = 0.22
STIFFNESS_EXPONENT_PRESSURE_PUSH = max(STIFFNESS_EXPONENT_RIGID, 0.62)
BRANCH_STIFFNESS_RECOVERY_TIME = 1.20
selected_branch_entry_lambda = STIFFNESS_EXPONENT_RIGID
branch_entry_timer = 0.0
return_trunk_release_pending = False
return_trunk_retract_timer = 0.0
return_trunk_last_released_id = None
return_trunk_force_timer = 0.0

# Branch-entrance/SPH-state measurement parameters
BRANCH_ENTRANCE_CONGESTION_RADIUS = 52.0 * MAP_SCALE
FLOW_DIRECTION_MIN_SPEED = 1.0
FLOW_DIRECTION_REFERENCE_SPEED = 12.0
CONGESTION_EXCESS_NORMALIZER = 1.0
# Longest corridor-to-opposite-branch path in the current cross map.
MAX_TRANSPORT_DISTANCE = max(BRANCH_LENGTHS.values()) + corridor_width

# HydroSwarm proxy region.  The Junction is treated as the aggregate proxy
# Ω_proxy.  It is partitioned into area-constrained temporary subregions whose
# quotas are proportional to the robot demand of each unvisited branch.
PROXY_CELL_SIZE = max(2, round(10 * MAP_SCALE))
PROXY_PARTITION_ITERATIONS = 160
PROXY_BIAS_LEARNING_RATE = 0.075
PROXY_DENSITY_MASS_MIN = 0.50
PROXY_DENSITY_MASS_MAX = 2.00
PROXY_FRONT_LAYER_LENGTH = SMOOTHING_LENGTH
PROXY_FRONT_LATERAL_SPACING = max(SAFE_RADIUS * 1.8, 1.0)
PROXY_ROLLOUT_CONTEXT_DISTANCE = SMOOTHING_LENGTH * 1.10
PROXY_ROLLOUT_MIN_PRIMARY = 6
PROXY_CONTEXT_HOLD_GAIN = 3.2
PROXY_CONTEXT_MAX_SPEED_SCALE = 0.28

# Geodesic EDF guidance and smooth virtual valves at unselected branch mouths.
EDF_FINITE_EPSILON = 1e-6
VIRTUAL_VALVE_RADIUS = 46.0 * MAP_SCALE
VIRTUAL_VALVE_GAIN = 92.0

# Centralized sampled-data approximation of HydroSwarm's local stability
# consensus. A dual-threshold readiness rule shortens Junction waiting while
# preserving the Base link, Anchor deployment, and minimum robot-count guards.
#
# Normal path: a moderately stable group may switch after a short dwell.
# Fast path: when enough robots have already gathered, a looser transient
# stability condition allows immediate Branch scoring instead of waiting for
# every particle to become nearly stationary.
JUNCTION_CONSENSUS_MIN_COUNT = 14
JUNCTION_CONSENSUS_STABLE_RATIO = 0.62
JUNCTION_CONSENSUS_SPEED_THRESHOLD = 5.5
JUNCTION_CONSENSUS_DENSITY_DELTA_RATIO = 0.14
JUNCTION_CONSENSUS_DWELL_TIME = 0.18

JUNCTION_FAST_READY_MIN_COUNT = JUNCTION_ENTRY_COUNT
JUNCTION_FAST_READY_STABLE_RATIO = 0.50
JUNCTION_FAST_READY_SPEED_THRESHOLD = 8.0
JUNCTION_FAST_READY_DENSITY_DELTA_RATIO = 0.22
JUNCTION_FAST_READY_DWELL_TIME = 0.10

# The old unconditional 1.40 s timeout made every child switch visibly pause.
# The shortened fallback still requires a minimally coherent Junction group.
JUNCTION_CONSENSUS_FALLBACK_TIME = 0.85
JUNCTION_FALLBACK_MIN_COUNT = 12
JUNCTION_FALLBACK_STABLE_RATIO = 0.35
JUNCTION_FALLBACK_SPEED_THRESHOLD = 10.0

CELL_SIZE = max(SMOOTHING_LENGTH, VIRTUAL_PRESSURE_RADIUS, COMM_RANGE)
SPH_CELL_SIZE = SMOOTHING_LENGTH

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
        return (
            region in {"BOTTOM", "JUNCTION"}
            or branch_gate_states.get(region) == "OPEN"
        )
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
    pixel_radius = max(1, int(round(radius)))
    diagonal = int(round(pixel_radius / math.sqrt(2.0)))
    test_points = [
        (x, y),
        (x + pixel_radius, y),
        (x - pixel_radius, y),
        (x, y + pixel_radius),
        (x, y - pixel_radius),
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


def constrain_base_reserve_to_bottom(robot: "Robot") -> None:
    """Keep Breadcrumb-front NORMAL guards in the Base corridor."""
    if (
        not robot.base_reserve
        or phase in {
            SimulationPhase.RETURN_TO_BASE,
            SimulationPhase.DONE,
        }
    ):
        return
    clearance = max(1.0, robot.radius)
    left = bottom_rect.left + clearance
    right = bottom_rect.right - clearance
    top = bottom_rect.top + clearance
    bottom = bottom_rect.bottom - clearance
    old_x, old_y = robot.position.x, robot.position.y
    robot.position.x = clamp(robot.position.x, left, right)
    robot.position.y = clamp(robot.position.y, top, bottom)
    if robot.position.x != old_x:
        robot.velocity.x = 0.0
    if robot.position.y != old_y:
        robot.velocity.y = max(0.0, robot.velocity.y)


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


def smoothstep01(value: float) -> float:
    value = clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def ramped_travel(
    elapsed: float,
    target_speed: float,
    ramp_time: float,
) -> float:
    """Distance under a bounded linear acceleration into target speed."""
    elapsed = max(0.0, elapsed)
    ramp_time = max(ramp_time, EPSILON)
    if elapsed < ramp_time:
        acceleration = target_speed / ramp_time
        return 0.5 * acceleration * elapsed**2
    return target_speed * (elapsed - 0.5 * ramp_time)


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
    """Blue-density palette for readability: sky blue -> blue -> navy."""
    ratio = density / max(reference_density, EPSILON)
    if ratio <= 0.85:
        return interpolate_color(
            (158, 214, 255),
            (102, 176, 255),
            ratio / 0.85,
        )
    if ratio <= 1.15:
        return interpolate_color(
            (102, 176, 255),
            (52, 118, 214),
            (ratio - 0.85) / 0.30,
        )
    if ratio <= 1.55:
        return interpolate_color(
            (52, 118, 214),
            (28, 74, 156),
            (ratio - 1.15) / 0.40,
        )
    return interpolate_color(
        (28, 74, 156),
        (16, 42, 96),
        min((ratio - 1.55) / 0.70, 1.0),
    )


def normalized_direction_toward(source, target):
    delta = target - source
    return delta.normalize() if delta.length_squared() > EPSILON else pygame.Vector2()


def get_bottom_hold_point():
    return pygame.Vector2(
        center_x,
        center_y + half_width + base_length - 18 * MAP_SCALE,
    )


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



def shepherd_curtain_active() -> bool:
    """Whether the continuous full-width Shepherd gate must be enforced."""
    return phase in {
        SimulationPhase.FORM_SHEPHERD_BOUNDARY,
        SimulationPhase.FILL_BEHIND_SHEPHERD,
        SimulationPhase.PRESSURE_PUSH,
        SimulationPhase.FLOW_BACKTRACK,
    }


def get_shepherd_curtain_depth(branch: str) -> float:
    """Current gate depth from the Junction mouth.

    During formation/filling, the gate is already closed at the planned
    Shepherd line.  During pressure push it follows the moving piston toward
    the Junction so robots cannot slip around or through individual Shepherds.
    """
    depth = get_shepherd_boundary_depth(branch)
    if phase == SimulationPhase.PRESSURE_PUSH:
        travel = min(
            SHEPHERD_PISTON_MAX_TRAVEL,
            ramped_travel(
                pressure_push_timer,
                SHEPHERD_PISTON_SPEED,
                SHEPHERD_SPEED_RAMP_TIME,
            ),
        )
        depth -= travel
    elif phase == SimulationPhase.FLOW_BACKTRACK:
        depth = (
            shepherd_flow_start_depth
            - ramped_travel(
                shepherd_flow_timer,
                SHEPHERD_LINE_BACKTRACK_SPEED,
                SHEPHERD_SPEED_RAMP_TIME,
            )
        )
    return max(0.0, depth)


def shepherd_slot_position_at_depth(
    anchor: pygame.Vector2,
    branch: str,
    depth: float,
) -> pygame.Vector2:
    """Keep each Shepherd's lateral slot while the whole line backtracks."""
    depth = clamp(depth, 0.0, BRANCH_LENGTHS[branch])
    if branch == "UP":
        return pygame.Vector2(anchor.x, center_y - half_width - depth)
    if branch == "LEFT":
        return pygame.Vector2(center_x - half_width - depth, anchor.y)
    return pygame.Vector2(center_x + half_width + depth, anchor.y)


def get_shepherd_normal_limit_depth(branch: str) -> float:
    return max(
        0.0,
        get_shepherd_curtain_depth(branch) - SHEPHERD_CURTAIN_CLEARANCE,
    )


def compute_shepherd_curtain_force(robot: "Robot") -> pygame.Vector2:
    """Continuous repulsion from the full-width virtual Shepherd curtain."""
    if (
        not shepherd_curtain_active()
        or robot.role != "NORMAL"
        or get_robot_region(robot.position) != active_branch
    ):
        return pygame.Vector2()

    depth = branch_depth_from_junction(robot.position, active_branch)
    limit_depth = get_shepherd_normal_limit_depth(active_branch)
    activation_depth = limit_depth - SHEPHERD_CURTAIN_INTERACTION_DEPTH
    if depth <= activation_depth:
        return pygame.Vector2()

    ratio = clamp(
        (depth - activation_depth)
        / max(SHEPHERD_CURTAIN_INTERACTION_DEPTH, EPSILON),
        0.0,
        1.5,
    )
    forward_speed = max(
        0.0,
        robot.velocity.dot(BRANCH_DIRECTIONS[active_branch]),
    )
    magnitude = min(
        SHEPHERD_CURTAIN_MAX_FORCE,
        (
            SHEPHERD_CURTAIN_FORCE * ratio**2
            + SHEPHERD_CURTAIN_VELOCITY_DAMPING
            * forward_speed
        ),
    )
    return get_backtrack_direction(active_branch) * magnitude


def constrain_normal_behind_shepherd_curtain(robot: "Robot") -> None:
    """Hard safety projection preventing leakage through Shepherd gaps.

    The force above provides smooth behaviour.  This projection is the final
    guard against a fast particle crossing the virtual plane in one time step.
    """
    if (
        not shepherd_curtain_active()
        or robot.role != "NORMAL"
        or get_robot_region(robot.position) != active_branch
    ):
        return

    limit_depth = get_shepherd_normal_limit_depth(active_branch)
    depth = branch_depth_from_junction(robot.position, active_branch)
    if depth <= limit_depth:
        return

    penetration = depth - limit_depth
    if active_branch == "UP":
        robot.position.y = center_y - half_width - limit_depth
    elif active_branch == "LEFT":
        robot.position.x = center_x - half_width - limit_depth
    else:
        robot.position.x = center_x + half_width + limit_depth

    forward_direction = BRANCH_DIRECTIONS[active_branch]
    forward_speed = robot.velocity.dot(forward_direction)
    if forward_speed > 0.0:
        robot.velocity -= forward_direction * forward_speed
    robot.velocity += get_backtrack_direction(active_branch) * min(
        SHEPHERD_CURTAIN_RECOVERY_SPEED,
        penetration * 2.0,
    )


def enforce_shepherd_curtain_for_swarm(robots) -> None:
    for robot in robots:
        constrain_normal_behind_shepherd_curtain(robot)

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
    branch_selection_events: list[dict] = field(default_factory=list)
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
        writer.writerow(["branch_selection_event_count", len(metrics.branch_selection_events)])
        for index, event in enumerate(metrics.branch_selection_events, start=1):
            component_text = "; ".join(
                f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in event["components"].items()
            )
            writer.writerow([
                f"branch_selection_{index}",
                (
                    f"time={event['time']:.6f}; selected={event['selected']}; "
                    f"cost={event['cost']:.6f}; max_structural_loss={event['max_structural_loss']}; "
                    f"{component_text}"
                ),
            ])
            for candidate, candidate_data in event.get("candidate_scores", {}).items():
                candidate_components = candidate_data.get("components", {})
                candidate_text = "; ".join(
                    f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
                    for key, value in candidate_components.items()
                )
                writer.writerow([
                    f"branch_selection_{index}_{candidate}",
                    f"cost={candidate_data.get('cost', 0.0):.6f}; {candidate_text}",
                ])
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
        self.received_gate_states = None
        self.received_sequence = -1


class Robot:
    def __init__(self, x: float, y: float, robot_id: int):
        self.robot_id = robot_id
        self.position = pygame.Vector2(x, y)
        self.previous_position = self.position.copy()
        self.ingress_lane_x = float(x)
        self.velocity = pygame.Vector2()
        self.acceleration = pygame.Vector2()
        self.filtered_acceleration = pygame.Vector2()
        self.radius = ROBOT_RADIUS
        self.density = 0.0
        self.density_ratio = 1.0
        self.pressure = 0.0
        self.last_sph_pressure_force = 0.0
        self.last_compression_release_force = 0.0
        self.last_shepherd_force = 0.0
        self.role = "NORMAL"

        self.shepherd_anchor: Optional[pygame.Vector2] = None
        self.shepherd_origin: Optional[pygame.Vector2] = None
        self.relay_anchor: Optional[pygame.Vector2] = None
        self.relay_index = -1
        self.anchor_position: Optional[pygame.Vector2] = None
        self.local_branch_states = branch_states.copy()
        self.selected_branch = None
        self.branch_gate_states = {branch: "OPEN" for branch in BRANCHES}
        self.branch_vote: Optional[str] = None
        self.branch_vote_confidence = 0.0
        self.distributed_branch_decision: Optional[str] = None
        self.transfer_target: Optional[str] = None
        self.base_reserve = False
        self.base_hold_position: Optional[pygame.Vector2] = None
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
        self.received_gate_states = None
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
            elif phase in {
                SimulationPhase.PRESSURE_PUSH,
                SimulationPhase.FLOW_BACKTRACK,
            }:
                target = shepherd_slot_position_at_depth(
                    self.shepherd_anchor,
                    active_branch,
                    get_shepherd_curtain_depth(active_branch),
                )
            else:
                target = None

            if target is not None:
                error = target - self.position
                # A selected Shepherd moves independently to its locally
                # claimed slot; distant Base connectivity is irrelevant.
                if phase == SimulationPhase.FORM_SHEPHERD_BOUNDARY:
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
                self.filtered_acceleration.update(0.0, 0.0)
                self.previous_position = old_position
                self._record_motion()
                return

        self.velocity += self.acceleration * dt
        if self.role == "NORMAL":
            region = get_robot_region(self.position)
            lateral_decay = math.exp(
                -CORRIDOR_LATERAL_VELOCITY_DAMPING * dt
            )
            if region in {"LEFT", "RIGHT"}:
                self.velocity.y *= lateral_decay
            elif region in {"UP", "BOTTOM"}:
                self.velocity.x *= lateral_decay
        speed_limit = MAX_SPEED
        if phase == SimulationPhase.MOVE_TO_JUNCTION:
            speed_limit = (
                INITIAL_JUNCTION_MAX_SPEED
                if get_robot_region(self.position) == "JUNCTION"
                else INITIAL_SAFE_MAX_SPEED
            )
        elif phase == SimulationPhase.PRESSURE_PUSH:
            speed_limit = PRESSURE_PUSH_MAX_SPEED
        elif phase == SimulationPhase.FLOW_BACKTRACK:
            speed_limit = FLOW_BACKTRACK_MAX_SPEED
        limit_vector(self.velocity, speed_limit)
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

        # Smooth virtual pressure is backed by a hard one-step guard so a fast
        # ordinary robot cannot pass through a temporary gap while Shepherds
        # are still moving laterally into their slots.
        constrain_normal_behind_shepherd_curtain(self)
        constrain_base_reserve_to_bottom(self)
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

        if base_station is not None and self.role != "BASE" and not self.connected_to_base:
            color = DISCONNECTED_FILL_COLOR

        draw_radius = max(1, round(self.radius + 1))
        pygame.draw.circle(surface, color, (x, y), draw_radius)

        if self.role in {"RELAY", "TRUNK_RELAY"}:
            ring_color = TRUNK_RELAY_COLOR if self.role == "TRUNK_RELAY" else RELAY_COLOR
            pygame.draw.circle(surface, ring_color, (x, y), draw_radius + 2, width=1)
        elif self.role == "ANCHOR":
            pygame.draw.circle(surface, ANCHOR_COLOR, (x, y), draw_radius + 2, width=1)
        elif self.role == "SHEPHERD":
            pygame.draw.circle(surface, SHEPHERD_COLOR, (x, y), draw_radius + 2, width=1)
        elif self.base_reserve:
            pygame.draw.circle(
                surface,
                BASE_COLOR,
                (x, y),
                draw_radius + 2,
                width=1,
            )

# =========================================================
# 9. Robot creation and spatial hash
# =========================================================


def create_grid_robots(robot_count: int):
    robots = []
    left = center_x - half_width + ROBOT_RADIUS + 4 * MAP_SCALE
    right = center_x + half_width - ROBOT_RADIUS - 4 * MAP_SCALE
    top = center_y + half_width + 12 * MAP_SCALE
    bottom = (
        center_y
        + half_width
        + base_length
        - ROBOT_RADIUS
        - 7 * MAP_SCALE
    )
    per_row = max(1, int((right - left) // GRID_SPACING) + 1)
    for robot_id in range(robot_count):
        row, column = divmod(robot_id, per_row)
        x = left + column * GRID_SPACING
        y = bottom - row * GRID_ROW_SPACING
        if y < top:
            print(f"Warning: only {len(robots)} robots fit in the entrance.")
            break
        robots.append(Robot(x, y, robot_id))
    return robots


def create_random_robots(robot_count: int):
    robots = []
    minimum_distance = ROBOT_RADIUS * 2 + 1
    left = center_x - half_width + ROBOT_RADIUS + 4 * MAP_SCALE
    right = center_x + half_width - ROBOT_RADIUS - 4 * MAP_SCALE
    top = center_y + half_width + 12 * MAP_SCALE
    bottom = (
        center_y
        + half_width
        + base_length
        - ROBOT_RADIUS
        - 7 * MAP_SCALE
    )
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


def physics_cell_key(position):
    return (
        int(position.x // SPH_CELL_SIZE),
        int(position.y // SPH_CELL_SIZE),
    )


def build_physics_grid(robots):
    grid = {}
    for robot in robots:
        grid.setdefault(
            physics_cell_key(robot.position),
            [],
        ).append(robot)
    return grid


def iter_physics_neighbor_candidates(robot, grid):
    cx, cy = physics_cell_key(robot.position)
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
    if anchor is None:
        return (
            distributed_consensus_branch,
            (
                f"DISTRIBUTED_{phase.name}"
                if distributed_consensus_branch is not None
                else "DISTRIBUTED_VOTING"
            ),
            branch_gate_states.copy(),
        )
    if not anchor.connected_to_base:
        return None, "WAIT_FOR_BASE_LINK", None
    return (
        anchor.selected_branch,
        phase.name,
        anchor.branch_gate_states.copy(),
    )


def propagate_base_message(robots, anchor):
    """Compute Base-rooted paths and mirror the NORMAL peer-consensus state.

    The Base is only a communication root/observer. It does not select a
    branch, assign roles, or issue motion commands.
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
        robot.received_gate_states = None
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

    selected_branch, command, gate_states = get_anchor_message(anchor)
    gate_signature = (
        tuple(sorted(gate_states.items()))
        if gate_states is not None
        else None
    )
    signature = (selected_branch, command, gate_signature)
    if signature != last_message_signature:
        communication_sequence += 1
        last_message_signature = signature
        gate_text = (
            ", ".join(
                f"{branch}={gate_states[branch]}"
                for branch in BRANCHES
            )
            if gate_states is not None
            else "-"
        )
        print(
            f"[Communication Observer] seq={communication_sequence}, "
            f"command={command}, branch={selected_branch}, gates={gate_text}"
        )

    base_station.received_branch = selected_branch
    base_station.received_command = command
    base_station.received_gate_states = (
        gate_states.copy() if gate_states is not None else None
    )
    base_station.received_sequence = communication_sequence
    for robot in robots:
        if not robot.connected_to_base:
            continue
        if anchor is None:
            robot.received_branch = robot.distributed_branch_decision
            robot.received_command = (
                "PEER_CONSENSUS"
                if robot.distributed_branch_decision is not None
                else "LOCAL_VOTING"
            )
            robot.received_gate_states = (
                branch_gate_states.copy()
                if robot.distributed_branch_decision is not None
                else None
            )
        else:
            robot.received_branch = selected_branch
            robot.received_command = command
            robot.received_gate_states = (
                gate_states.copy() if gate_states is not None else None
            )
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
# 11. Reactive tail Breadcrumb communication trail
# =========================================================


def initialize_trunk_relay_plan():
    global trunk_relay_slots, trunk_relay_deploy_cooldown
    # Deliberately keep this empty: trunk positions must never be precomputed.
    # The active controller uses update_relay_deployment(), which freezes an
    # actual tail robot at its current, already-traversed position.
    trunk_relay_slots = []
    trunk_relay_deploy_cooldown = 0.0


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
    candidate.filtered_acceleration.update(0.0, 0.0)
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


def release_next_trunk_relay_for_return(robots):
    """Release one Trunk Relay from the Junction side toward the Base.

    Sequential release prevents the fixed trunk from becoming a permanent
    deadlock while preserving a Base-rooted communication backbone during the
    final return.  The released robot becomes NORMAL and must reach BOTTOM and
    reconnect before the next Trunk Relay is released.
    """
    relays = sorted(
        (robot for robot in robots if robot.role == "TRUNK_RELAY"),
        key=lambda robot: robot.relay_index,
        reverse=True,
    )
    if not relays:
        return None

    robot = relays[0]
    released_index = robot.relay_index
    robot.role = "NORMAL"
    robot.relay_anchor = None
    robot.relay_index = -1
    robot.ingress_lane_x = float(robot.position.x)
    robot.velocity.update(0.0, RETURN_TRUNK_RELEASE_INITIAL_SPEED)
    robot.acceleration.update(0.0, 0.0)
    robot.filtered_acceleration.update(0.0, 0.0)
    print(
        f"[Base Trunk] retract robot={robot.robot_id}, "
        f"index={released_index}, remaining={len(relays) - 1}"
    )
    return robot


def release_trunk_relays_for_return(robots):
    """Emergency fallback: release every remaining Trunk Relay."""
    released = 0
    for robot in robots:
        if robot.role != "TRUNK_RELAY":
            continue
        robot.role = "NORMAL"
        robot.relay_anchor = None
        robot.relay_index = -1
        robot.ingress_lane_x = float(robot.position.x)
        robot.velocity.update(0.0, RETURN_TRUNK_RELEASE_INITIAL_SPEED)
        robot.acceleration.update(0.0, 0.0)
        released += 1
    print(f"[Base Trunk] emergency release={released}")


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
    relay_slots = []
    relay_deploy_cooldown = relay_retract_cooldown = relay_retract_clear_timer = 0.0
    relay_motion_scale = 1.0
    print(f"[Breadcrumb] reactive deployment enabled for branch={branch}")


def relay_path_progress(position, branch):
    """Piecewise progress along the path already traversed by the swarm."""
    junction_center = pygame.Vector2(center_x, center_y)
    base_vector = junction_center - BASE_POSITION
    base_length = base_vector.length()
    region = get_robot_region(position)
    if region == "BOTTOM":
        return clamp(
            (position - BASE_POSITION).dot(base_vector.normalize()),
            0.0,
            base_length,
        )
    branch_direction = BRANCH_DIRECTIONS[branch]
    if region == "JUNCTION":
        junction_approach = base_length - clamp(
            position.y - center_y,
            0.0,
            float(half_width),
        )
        return junction_approach + clamp(
            (position - junction_center).dot(branch_direction),
            0.0,
            float(half_width),
        )
    if region == branch:
        return (
            base_length
            + half_width
            + branch_depth_from_junction(position, branch)
        )
    return 0.0


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
    candidate.acceleration.update(0.0, 0.0)
    candidate.filtered_acceleration.update(0.0, 0.0)
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


def assign_breadcrumb_front_guards(
    robots,
    breadcrumb: "Robot",
    breadcrumb_progress: float,
) -> list["Robot"]:
    """Keep nearby NORMAL robots immediately ahead of a Base Breadcrumb."""
    if get_robot_region(breadcrumb.position) != "BOTTOM":
        return []
    candidates = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and not robot.base_reserve
        and robot.connected_to_base
        and get_robot_region(robot.position) == "BOTTOM"
        and breadcrumb_progress + ROBOT_RADIUS
        < relay_path_progress(robot.position, active_branch)
        <= breadcrumb_progress + BREADCRUMB_GUARD_DEPTH
        and robot.position.distance_to(breadcrumb.position) <= COMM_RANGE * 0.9
    ]
    candidates.sort(
        key=lambda robot: (
            relay_path_progress(robot.position, active_branch),
            robot.position.distance_squared_to(breadcrumb.position),
            robot.robot_id,
        )
    )
    selected = candidates[:BREADCRUMB_GUARD_PER_RELAY]
    for robot in selected:
        robot.base_reserve = True
        robot.base_hold_position = robot.position.copy()
    if selected:
        print(
            f"[Breadcrumb Guards] relay={breadcrumb.robot_id}, "
            f"normal_guards={[robot.robot_id for robot in selected]}"
        )
    return selected


def update_relay_deployment(robots, dt):
    global relay_deploy_cooldown, relay_motion_scale
    relay_deploy_cooldown = max(0.0, relay_deploy_cooldown - dt)
    relay_motion_scale = 1.0
    if phase not in {
        SimulationPhase.MOVE_TO_JUNCTION,
        SimulationPhase.EXPLORE_BRANCH,
        SimulationPhase.FORM_SHEPHERD_BOUNDARY,
        SimulationPhase.FILL_BEHIND_SHEPHERD,
    }:
        return
    if (
        phase == SimulationPhase.MOVE_TO_JUNCTION
        and simulation_time < BASE_COMPRESSION_DURATION
    ):
        return
    if relay_deploy_cooldown > 0.0 or base_station is None:
        return

    breadcrumbs = get_active_branch_relays(robots)
    last_node = breadcrumbs[-1] if breadcrumbs else base_station
    last_progress = (
        relay_path_progress(last_node.relay_anchor, active_branch)
        if breadcrumbs and last_node.relay_anchor is not None
        else 0.0
    )
    mobile = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and not robot.base_reserve
        and robot.connected_to_base
        and get_robot_region(robot.position)
        in {"BOTTOM", "JUNCTION", active_branch}
    ]
    ahead = [
        (relay_path_progress(robot.position, active_branch), robot)
        for robot in mobile
        if relay_path_progress(robot.position, active_branch)
        > last_progress + EPSILON
    ]
    if not ahead:
        return

    ahead.sort(key=lambda item: (item[0], item[1].robot_id))
    tail_progress = ahead[0][0]
    front_progress = ahead[-1][0]
    if tail_progress - last_progress < BREADCRUMB_SPACING:
        return
    if front_progress - tail_progress < BREADCRUMB_FRONT_CLEARANCE:
        return

    tail_band = [
        (progress, robot)
        for progress, robot in ahead
        if progress <= tail_progress + GRID_SPACING * 1.5
        and BREADCRUMB_DEPLOY_DISTANCE
        <= robot.position.distance_to(last_node.position)
        <= COMM_RANGE * 0.88
    ]
    if not tail_band:
        return
    tail_progress, tail_robot = min(
        tail_band,
        key=lambda item: (
            abs(
                item[1].position.distance_to(last_node.position)
                - BREADCRUMB_SPACING
            ),
            item[0],
            item[1].robot_id,
        ),
    )
    if tail_robot.total_distance < BREADCRUMB_MIN_TRAVEL:
        return

    tail_robot.role = "RELAY"
    tail_robot.relay_anchor = tail_robot.position.copy()
    tail_robot.relay_index = (
        breadcrumbs[-1].relay_index + 1 if breadcrumbs else 0
    )
    tail_robot.velocity.update(0.0, 0.0)
    tail_robot.acceleration.update(0.0, 0.0)
    tail_robot.filtered_acceleration.update(0.0, 0.0)
    relay_slots.append({
        "index": tail_robot.relay_index,
        "position": tail_robot.relay_anchor.copy(),
        "path_distance": tail_progress,
    })
    guards = assign_breadcrumb_front_guards(
        robots,
        tail_robot,
        tail_progress,
    )
    relay_deploy_cooldown = BREADCRUMB_DEPLOY_COOLDOWN
    print(
        f"[Breadcrumb] tail robot={tail_robot.robot_id}, "
        f"index={tail_robot.relay_index}, progress={tail_progress:.1f}, "
        f"front_guards={len(guards)}"
    )


def release_relay_into_backtracking(robot):
    index = robot.relay_index
    robot.role = "NORMAL"
    robot.relay_anchor = None
    robot.relay_index = -1
    robot.velocity = get_backtrack_direction(active_branch) * RELAY_RELEASE_SPEED
    robot.acceleration.update(0.0, 0.0)
    robot.filtered_acceleration.update(0.0, 0.0)
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
    nodes = ([base_station] if base_station is not None else []) + get_active_branch_relays(robots)
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
    junction_anchor.branch_gate_states = branch_gate_states.copy()
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


def get_branch_entrance(branch: str) -> pygame.Vector2:
    """Return the center point of a branch entrance at the Junction boundary."""
    if branch == "UP":
        return pygame.Vector2(center_x, center_y - half_width)
    if branch == "LEFT":
        return pygame.Vector2(center_x - half_width, center_y)
    if branch == "RIGHT":
        return pygame.Vector2(center_x + half_width, center_y)
    raise ValueError(f"Unknown branch: {branch}")


def get_region_entrance(region: str) -> Optional[pygame.Vector2]:
    """Return the Junction-side entrance of a corridor region."""
    if region == "UP":
        return pygame.Vector2(center_x, center_y - half_width)
    if region == "LEFT":
        return pygame.Vector2(center_x - half_width, center_y)
    if region == "RIGHT":
        return pygame.Vector2(center_x + half_width, center_y)
    if region == "BOTTOM":
        return pygame.Vector2(center_x, center_y + half_width)
    return None


def free_space_distance_to_branch(
    position: pygame.Vector2,
    branch: str,
) -> float:
    """Approximate wall-respecting geodesic distance to a branch entrance.

    The current map is a convex rectangular Junction connected to four convex
    corridor rectangles. A robot in another corridor must first reach that
    corridor's Junction entrance, pass through the Junction center, and then
    reach the candidate branch entrance. This prevents straight-line distances
    from incorrectly cutting through walls.
    """
    target_entrance = get_branch_entrance(branch)
    region = get_robot_region(position)
    junction_center = pygame.Vector2(center_x, center_y)

    if region == "JUNCTION":
        return position.distance_to(target_entrance)

    current_entrance = get_region_entrance(region)
    if current_entrance is not None:
        if region == branch:
            return position.distance_to(target_entrance)
        return (
            position.distance_to(current_entrance)
            + current_entrance.distance_to(junction_center)
            + junction_center.distance_to(target_entrance)
        )

    # Defensive fallback for a numerical boundary point.
    return position.distance_to(junction_center) + junction_center.distance_to(target_entrance)



def estimate_branch_robot_demand(branch: str) -> float:
    """Estimate fluid mass needed by a branch for proportional proxy sizing.

    The demand combines width-adaptive Shepherds, communication relays, and a
    length-dependent front-fluid term.  Only ratios are used, so this is a
    practical sizing rule analogous to HydroSwarm's target-area proportion.
    """
    shepherd_demand = adaptive_shepherd_count()
    relay_demand = max(0, math.ceil(BRANCH_LENGTHS[branch] / RELAY_SPACING) - 1)
    longitudinal_layers = max(
        1,
        math.ceil(BRANCH_LENGTHS[branch] / max(PROXY_FRONT_LAYER_LENGTH, EPSILON)),
    )
    lateral_robots = max(
        1,
        math.ceil(
            (corridor_width - 2.0 * SHEPHERD_EDGE_MARGIN)
            / max(PROXY_FRONT_LATERAL_SPACING, EPSILON)
        ),
    )
    front_fluid_demand = longitudinal_layers * lateral_robots
    return float(shepherd_demand + relay_demand + front_fluid_demand)


def proxy_cell_grid() -> tuple[dict[tuple[int, int], pygame.Vector2], int, int]:
    cols = max(1, int(math.ceil(junction_rect.width / PROXY_CELL_SIZE)))
    rows = max(1, int(math.ceil(junction_rect.height / PROXY_CELL_SIZE)))
    centers: dict[tuple[int, int], pygame.Vector2] = {}
    for row in range(rows):
        for col in range(cols):
            x = min(
                junction_rect.right - 0.5,
                junction_rect.left + (col + 0.5) * PROXY_CELL_SIZE,
            )
            y = min(
                junction_rect.bottom - 0.5,
                junction_rect.top + (row + 0.5) * PROXY_CELL_SIZE,
            )
            centers[(col, row)] = pygame.Vector2(x, y)
    return centers, cols, rows


def build_capacity_constrained_proxy_partition(
    branches: list[str],
) -> tuple[dict[tuple[int, int], str], dict[tuple[int, int], pygame.Vector2], dict[str, float]]:
    """Partition Ω_proxy using capacity-constrained geodesic Voronoi cells.

    Additive branch biases are iteratively adapted until each branch receives
    approximately its demand-proportional area quota.  Because every cell is
    assigned by distance to a branch mouth plus one branch-wide bias, the
    resulting subregions remain spatially coherent rather than arbitrary robot
    assignments.
    """
    branches = sorted(branches)
    centers, _, _ = proxy_cell_grid()
    if not branches:
        return {}, centers, {}
    if len(branches) == 1:
        only = branches[0]
        return {key: only for key in centers}, centers, {only: 1.0}

    demands = {branch: estimate_branch_robot_demand(branch) for branch in branches}
    total_demand = max(sum(demands.values()), EPSILON)
    quotas = {branch: demands[branch] / total_demand for branch in branches}
    biases = {branch: 0.0 for branch in branches}
    diagonal = max(math.hypot(junction_rect.width, junction_rect.height), EPSILON)
    partition: dict[tuple[int, int], str] = {}

    for _ in range(PROXY_PARTITION_ITERATIONS):
        counts = {branch: 0 for branch in branches}
        partition = {}
        for key, center in centers.items():
            selected = min(
                branches,
                key=lambda branch: (
                    center.distance_to(get_branch_entrance(branch)) / diagonal
                    + biases[branch],
                    branch,
                ),
            )
            partition[key] = selected
            counts[selected] += 1

        total_cells = max(len(centers), 1)
        maximum_error = 0.0
        for branch in branches:
            actual = counts[branch] / total_cells
            error = actual - quotas[branch]
            maximum_error = max(maximum_error, abs(error))
            # Too much area -> increase its additive distance penalty.
            biases[branch] += PROXY_BIAS_LEARNING_RATE * error
        if maximum_error <= 1.0 / total_cells:
            break

    return partition, centers, quotas


def project_robot_to_proxy(position: pygame.Vector2) -> pygame.Vector2:
    """Project a robot into Ω_proxy while retaining its lateral placement."""
    region = get_robot_region(position)
    margin = 1.0
    if region == "JUNCTION":
        x = clamp(position.x, junction_rect.left + margin, junction_rect.right - margin)
        y = clamp(position.y, junction_rect.top + margin, junction_rect.bottom - margin)
        return pygame.Vector2(x, y)
    if region == "UP":
        return pygame.Vector2(
            clamp(position.x, junction_rect.left + margin, junction_rect.right - margin),
            junction_rect.top + margin,
        )
    if region == "LEFT":
        return pygame.Vector2(
            junction_rect.left + margin,
            clamp(position.y, junction_rect.top + margin, junction_rect.bottom - margin),
        )
    if region == "RIGHT":
        return pygame.Vector2(
            junction_rect.right - margin,
            clamp(position.y, junction_rect.top + margin, junction_rect.bottom - margin),
        )
    return pygame.Vector2(
        clamp(position.x, junction_rect.left + margin, junction_rect.right - margin),
        junction_rect.bottom - margin,
    )


def nearest_proxy_cell(
    point: pygame.Vector2,
    centers: dict[tuple[int, int], pygame.Vector2],
) -> Optional[tuple[int, int]]:
    if not centers:
        return None
    col = int((point.x - junction_rect.left) // PROXY_CELL_SIZE)
    row = int((point.y - junction_rect.top) // PROXY_CELL_SIZE)
    key = (col, row)
    if key in centers:
        return key
    return min(centers, key=lambda candidate: centers[candidate].distance_squared_to(point))


def compute_proxy_mass_statistics(
    robots,
    branches: list[str],
    partition: dict[tuple[int, int], str],
    centers: dict[tuple[int, int], pygame.Vector2],
    quotas: dict[str, float],
    reference_density: float,
) -> tuple[dict[str, dict], dict[int, str]]:
    """Measure uniform-particle and density-weighted fluid mass per subregion."""
    stats = {
        branch: {
            "quota_fraction": quotas.get(branch, 0.0),
            "cell_count": sum(value == branch for value in partition.values()),
            "robot_count": 0,
            "density_mass": 0.0,
            "actual_mass_fraction": 0.0,
            "mass_deficit_cost": 1.0,
        }
        for branch in branches
    }
    robot_assignment: dict[int, str] = {}
    total_mass = 0.0
    for robot in robots:
        proxy_point = project_robot_to_proxy(robot.position)
        cell = nearest_proxy_cell(proxy_point, centers)
        if cell is None or cell not in partition:
            continue
        branch = partition[cell]
        density_mass = clamp(
            robot.density / max(reference_density, EPSILON),
            PROXY_DENSITY_MASS_MIN,
            PROXY_DENSITY_MASS_MAX,
        )
        stats[branch]["robot_count"] += 1
        stats[branch]["density_mass"] += density_mass
        robot_assignment[robot.robot_id] = branch
        total_mass += density_mass

    for branch in branches:
        quota = stats[branch]["quota_fraction"]
        actual = stats[branch]["density_mass"] / max(total_mass, EPSILON)
        stats[branch]["actual_mass_fraction"] = actual
        stats[branch]["mass_deficit_cost"] = clamp(
            max(0.0, quota - actual) / max(quota, EPSILON),
            0.0,
            1.0,
        )
    return stats, robot_assignment


def geodesic_edf_direction(
    position: pygame.Vector2,
    branch: str,
    final_target: Optional[pygame.Vector2] = None,
) -> pygame.Vector2:
    """Negative gradient direction of a piecewise geodesic EDF.

    For this rectilinear cross map the exact shortest free-space route is:
    current corridor -> its Junction mouth -> selected mouth -> branch target.
    The returned unit vector is therefore the analytic counterpart of
    -∇phi/||∇phi|| used by HydroSwarm, without a raster distance-field lookup.
    """
    region = get_robot_region(position)
    if region == branch:
        target = final_target if final_target is not None else get_branch_tip_target(branch)
        return normalized_direction_toward(position, target)
    if region == "JUNCTION":
        return normalized_direction_toward(position, get_branch_entrance(branch))
    current_entrance = get_region_entrance(region)
    if current_entrance is not None:
        return normalized_direction_toward(position, current_entrance)
    return normalized_direction_toward(position, pygame.Vector2(center_x, center_y))


def compute_virtual_valve_force(robot: "Robot") -> pygame.Vector2:
    """Smooth virtual-particle barrier at every unselected branch mouth."""
    if phase not in {
        SimulationPhase.EXPLORE_BRANCH,
        SimulationPhase.FORM_SHEPHERD_BOUNDARY,
        SimulationPhase.FILL_BEHIND_SHEPHERD,
        SimulationPhase.PRESSURE_PUSH,
        SimulationPhase.FLOW_BACKTRACK,
    }:
        return pygame.Vector2()
    if get_robot_region(robot.position) != "JUNCTION":
        return pygame.Vector2()

    force = pygame.Vector2()
    junction_center = pygame.Vector2(center_x, center_y)
    for branch in BRANCHES:
        if branch_gate_states.get(branch) != "CLOSED":
            continue
        entrance = get_branch_entrance(branch)
        distance = robot.position.distance_to(entrance)
        if distance >= VIRTUAL_VALVE_RADIUS:
            continue
        inward = junction_center - entrance
        if inward.length_squared() <= EPSILON:
            continue
        ratio = 1.0 - distance / VIRTUAL_VALVE_RADIUS
        force += inward.normalize() * VIRTUAL_VALVE_GAIN * ratio**2
    return force

def get_branch_ordering_robots(robots) -> list["Robot"]:
    """Return mobile Base-connected mass used by the branch-ordering layer."""
    eligible = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and robot.connected_to_base
        and get_robot_region(robot.position) != "OUTSIDE"
    ]
    if eligible:
        return eligible

    # During the first communication update, connectivity can be one frame old.
    # Falling back to NORMAL robots avoids an undefined branch score.
    return [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and get_robot_region(robot.position) != "OUTSIDE"
    ]


def compute_swarm_principal_axis(robots) -> tuple[pygame.Vector2, float]:
    """Return the 2-D principal axis and anisotropy confidence in [0, 1]."""
    if len(robots) < 2:
        return pygame.Vector2(0.0, 0.0), 0.0

    mean_x = sum(robot.position.x for robot in robots) / len(robots)
    mean_y = sum(robot.position.y for robot in robots) / len(robots)
    cov_xx = sum((robot.position.x - mean_x) ** 2 for robot in robots) / len(robots)
    cov_yy = sum((robot.position.y - mean_y) ** 2 for robot in robots) / len(robots)
    cov_xy = sum(
        (robot.position.x - mean_x) * (robot.position.y - mean_y)
        for robot in robots
    ) / len(robots)

    trace = cov_xx + cov_yy
    discriminant = math.sqrt(max(0.0, (cov_xx - cov_yy) ** 2 + 4.0 * cov_xy**2))
    lambda_max = 0.5 * (trace + discriminant)
    lambda_min = 0.5 * (trace - discriminant)
    confidence = clamp(
        (lambda_max - lambda_min) / max(lambda_max + lambda_min, EPSILON),
        0.0,
        1.0,
    )

    if lambda_max <= EPSILON:
        return pygame.Vector2(0.0, 0.0), 0.0

    # Principal eigenvector of the symmetric 2x2 covariance matrix.
    if abs(cov_xy) > EPSILON:
        axis = pygame.Vector2(lambda_max - cov_yy, cov_xy)
    elif cov_xx >= cov_yy:
        axis = pygame.Vector2(1.0, 0.0)
    else:
        axis = pygame.Vector2(0.0, 1.0)

    if axis.length_squared() <= EPSILON:
        return pygame.Vector2(0.0, 0.0), 0.0
    return axis.normalize(), confidence


def compute_transport_cost(branch: str, robots) -> float:
    if not robots:
        return 0.0
    average_distance = sum(
        free_space_distance_to_branch(robot.position, branch)
        for robot in robots
    ) / len(robots)
    return clamp(average_distance / max(MAX_TRANSPORT_DISTANCE, EPSILON), 0.0, 1.0)


def compute_shape_cost(
    branch: str,
    principal_axis: pygame.Vector2,
    shape_confidence: float,
) -> float:
    if principal_axis.length_squared() <= EPSILON or shape_confidence <= EPSILON:
        return 0.0
    alignment = abs(principal_axis.dot(BRANCH_DIRECTIONS[branch]))
    return clamp((1.0 - alignment) * shape_confidence, 0.0, 1.0)


def compute_flow_cost(branch: str, robots) -> float:
    if not robots:
        return 0.0
    mean_velocity = sum(
        (robot.velocity for robot in robots),
        pygame.Vector2(0.0, 0.0),
    ) / len(robots)
    mean_speed = mean_velocity.length()
    if mean_speed < FLOW_DIRECTION_MIN_SPEED:
        return 0.0

    alignment = clamp(
        mean_velocity.normalize().dot(BRANCH_DIRECTIONS[branch]),
        -1.0,
        1.0,
    )
    direction_cost = 0.5 * (1.0 - alignment)
    confidence = clamp(
        mean_speed / max(FLOW_DIRECTION_REFERENCE_SPEED, EPSILON),
        0.0,
        1.0,
    )
    return clamp(direction_cost * confidence, 0.0, 1.0)


def compute_congestion_cost(
    branch: str,
    robots,
    reference_density: float,
) -> float:
    entrance = get_branch_entrance(branch)
    nearby = [
        robot
        for robot in robots
        if robot.position.distance_to(entrance) <= BRANCH_ENTRANCE_CONGESTION_RADIUS
        and has_line_of_sight(robot.position, entrance)
    ]
    if not nearby:
        return 0.0

    excess_squared = [
        max(0.0, robot.density / max(reference_density, EPSILON) - 1.0) ** 2
        for robot in nearby
    ]
    raw = sum(excess_squared) / len(excess_squared)
    return clamp(raw / max(CONGESTION_EXCESS_NORMALIZER, EPSILON), 0.0, 1.0)



@dataclass
class RolloutParticle:
    """Lightweight proxy-region particle for candidate-specific SPH rollout.

    Primary particles belong to the candidate Branch subregion and are used
    for all branch metrics. Context particles lie within one SPH support of
    the proxy boundary; they participate in kernels and communication but are
    softly held near their original positions so that the candidate does not
    incorrectly pull the whole swarm into every virtual Branch.
    """

    robot_id: int
    position: pygame.Vector2
    velocity: pygame.Vector2
    initial_position: pygame.Vector2
    initial_velocity: pygame.Vector2
    initial_density: float
    is_primary: bool
    density: float = 0.0
    pressure: float = 0.0


def get_proxy_boundary_centers(
    branch: str,
    partition: dict[tuple[int, int], str],
    centers: dict[tuple[int, int], pygame.Vector2],
) -> list[pygame.Vector2]:
    """Return cell centers on the decision boundary of one proxy subregion."""
    boundary = []
    for key, owner in partition.items():
        if owner != branch:
            continue
        col, row = key
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (col + dx, row + dy)
            if neighbor in partition and partition[neighbor] != branch:
                boundary.append(centers[key])
                break
    # A single remaining Branch owns all cells and has no internal boundary.
    if not boundary:
        boundary = [get_branch_entrance(branch)]
    return boundary


def get_proxy_region_rollout_robots(
    robots,
    branch: str,
    robot_assignment: dict[int, str],
    partition: dict[tuple[int, int], str],
    centers: dict[tuple[int, int], pygame.Vector2],
) -> tuple[list["Robot"], set[int], set[int]]:
    """Select candidate-region robots plus one-support boundary context.

    Robots are never physically divided by this assignment. The sets exist
    only inside the branch decision layer.
    """
    eligible = get_branch_ordering_robots(robots)
    primary = [
        robot for robot in eligible
        if robot_assignment.get(robot.robot_id) == branch
    ]

    # Numerical fallback: guarantee a minimally meaningful regional sample.
    if len(primary) < PROXY_ROLLOUT_MIN_PRIMARY:
        already = {robot.robot_id for robot in primary}
        supplements = sorted(
            (robot for robot in eligible if robot.robot_id not in already),
            key=lambda robot: (
                free_space_distance_to_branch(robot.position, branch),
                robot.robot_id,
            ),
        )
        primary.extend(supplements[: PROXY_ROLLOUT_MIN_PRIMARY - len(primary)])

    primary_ids = {robot.robot_id for robot in primary}
    boundary_centers = get_proxy_boundary_centers(branch, partition, centers)
    context_candidates = []
    for robot in eligible:
        if robot.robot_id in primary_ids:
            continue
        proxy_point = project_robot_to_proxy(robot.position)
        distance = min(
            proxy_point.distance_to(center)
            for center in boundary_centers
        )
        if distance <= PROXY_ROLLOUT_CONTEXT_DISTANCE:
            context_candidates.append((distance, robot.robot_id, robot))

    context_candidates.sort(key=lambda item: (item[0], item[1]))
    remaining_capacity = max(0, FLOW_ROLLOUT_MAX_ROBOTS - len(primary))
    context = [item[2] for item in context_candidates[:remaining_capacity]]
    context_ids = {robot.robot_id for robot in context}

    # Preserve all primary particles whenever possible; trim only pathological
    # cases after prioritizing robots closest to the candidate entrance.
    if len(primary) > FLOW_ROLLOUT_MAX_ROBOTS:
        primary.sort(
            key=lambda robot: (
                free_space_distance_to_branch(robot.position, branch),
                robot.robot_id,
            )
        )
        primary = primary[:FLOW_ROLLOUT_MAX_ROBOTS]
        primary_ids = {robot.robot_id for robot in primary}
        context = []
        context_ids = set()

    return primary + context, primary_ids, context_ids


def rollout_region_allowed(position: pygame.Vector2, branch: str) -> bool:
    return get_robot_region(position) in {"BOTTOM", "JUNCTION", branch}


def is_rollout_walkable(
    position: pygame.Vector2,
    radius: float,
    branch: str,
) -> bool:
    """Candidate-specific map check independent of the real active_branch."""
    x = int(round(position.x))
    y = int(round(position.y))
    diagonal = int(round(radius / math.sqrt(2.0)))
    test_points = [
        (x, y),
        (x + radius, y),
        (x - radius, y),
        (x, y + radius),
        (x, y - radius),
        (x + diagonal, y + diagonal),
        (x + diagonal, y - diagonal),
        (x - diagonal, y + diagonal),
        (x - diagonal, y - diagonal),
    ]
    for px, py in test_points:
        px = int(round(px))
        py = int(round(py))
        if not (0 <= px < SCREEN_WIDTH and 0 <= py < SCREEN_HEIGHT):
            return False
        if walkable_mask.get_at((px, py)) == 0:
            return False
    return rollout_region_allowed(position, branch)


def rollout_edf_direction(
    position: pygame.Vector2,
    branch: str,
) -> pygame.Vector2:
    """Analytic geodesic EDF direction for a candidate Branch rollout."""
    region = get_robot_region(position)
    if region == branch:
        target = branch_point_at_depth(
            branch,
            min(BRANCH_LENGTHS[branch], FLOW_ROLLOUT_TARGET_DEPTH),
        )
        return normalized_direction_toward(position, target)
    if region == "JUNCTION":
        return normalized_direction_toward(position, get_branch_entrance(branch))
    current_entrance = get_region_entrance(region)
    if current_entrance is not None:
        return normalized_direction_toward(position, current_entrance)
    return normalized_direction_toward(position, pygame.Vector2(center_x, center_y))


def rollout_virtual_valve_force(
    position: pygame.Vector2,
    open_branch: str,
) -> pygame.Vector2:
    if get_robot_region(position) != "JUNCTION":
        return pygame.Vector2()
    force = pygame.Vector2()
    junction_center = pygame.Vector2(center_x, center_y)
    for closed_branch in BRANCHES:
        if closed_branch == open_branch:
            continue
        entrance = get_branch_entrance(closed_branch)
        distance = position.distance_to(entrance)
        if distance >= VIRTUAL_VALVE_RADIUS:
            continue
        inward = junction_center - entrance
        if inward.length_squared() <= EPSILON:
            continue
        ratio = 1.0 - distance / VIRTUAL_VALVE_RADIUS
        force += inward.normalize() * FLOW_ROLLOUT_VALVE_GAIN * ratio**2
    return force


def compute_rollout_densities(particles: list[RolloutParticle]) -> None:
    self_density = spiky_kernel(0.0, SMOOTHING_LENGTH)
    densities = [self_density for _ in particles]
    h_squared = SMOOTHING_LENGTH**2
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            delta = particles[i].position - particles[j].position
            distance_squared = delta.length_squared()
            if distance_squared > h_squared:
                continue
            value = spiky_kernel(math.sqrt(max(distance_squared, 0.0)), SMOOTHING_LENGTH)
            densities[i] += value
            densities[j] += value
    for particle, density in zip(particles, densities):
        particle.density = max(density, EPSILON)


def compute_rollout_pressures(
    particles: list[RolloutParticle],
    reference_density: float,
    stiffness_exponent: float,
) -> None:
    for particle in particles:
        ratio = particle.density / max(reference_density, EPSILON)
        particle.pressure = (
            PRESSURE_GAIN
            * particle.density
            * (ratio**stiffness_exponent - 1.0)
        )


def rollout_stiffness_for_branch(
    branch: str,
    incoming_direction: pygame.Vector2,
) -> tuple[float, float]:
    """Return temporary lambda and normalized material-mode transition cost."""
    turn_ratio = angle_between(
        incoming_direction,
        BRANCH_DIRECTIONS[branch],
    ) / math.pi
    mode_cost = clamp(turn_ratio, 0.0, 1.0)
    stiffness = (
        STIFFNESS_EXPONENT_RIGID
        - (STIFFNESS_EXPONENT_RIGID - STIFFNESS_EXPONENT_SOFT) * mode_cost
    )
    return stiffness, mode_cost


def evaluate_rollout_communication_risk(
    particles: list[RolloutParticle],
    robots,
) -> tuple[float, float, float]:
    """Predict Base connectivity of the candidate proxy-region particles.

    Context particles may provide physically valid intermediate links, but the
    connected ratio and robust margin are evaluated only for primary particles.
    """
    if base_station is None:
        return 1.0, 0.0, 0.0

    fixed_positions = [base_station.position.copy()]
    fixed_positions.extend(
        relay.position.copy()
        for relay in get_trunk_relays(robots)
    )
    if junction_anchor is not None:
        fixed_positions.append(junction_anchor.position.copy())

    positions = fixed_positions + [particle.position.copy() for particle in particles]
    fixed_count = len(fixed_positions)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in positions]
    range_squared = COMM_RANGE**2

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distance_squared = positions[i].distance_squared_to(positions[j])
            if distance_squared > range_squared:
                continue
            if not has_line_of_sight(positions[i], positions[j]):
                continue
            distance = math.sqrt(max(distance_squared, 0.0))
            margin = COMM_RANGE - distance
            adjacency[i].append((j, margin))
            adjacency[j].append((i, margin))

    best_margin = [float("-inf") for _ in positions]
    best_margin[0] = float("inf")
    heap = [(-1.0e12, 0)]
    while heap:
        _, current = heapq.heappop(heap)
        for neighbor, edge_margin in adjacency[current]:
            candidate_margin = min(best_margin[current], edge_margin)
            if candidate_margin <= best_margin[neighbor] + EPSILON:
                continue
            best_margin[neighbor] = candidate_margin
            heapq.heappush(heap, (-candidate_margin, neighbor))

    primary_margins = [
        best_margin[fixed_count + index]
        for index, particle in enumerate(particles)
        if particle.is_primary
    ]
    connected = [margin for margin in primary_margins if math.isfinite(margin)]
    connected_ratio = len(connected) / max(len(primary_margins), 1)
    robust_margin = min(connected) if connected else -COMM_RANGE
    margin_quality = clamp(
        robust_margin / max(COMM_SAFE_DISTANCE, EPSILON),
        0.0,
        1.0,
    )
    risk = clamp(
        0.75 * (1.0 - connected_ratio)
        + 0.25 * (1.0 - margin_quality),
        0.0,
        1.0,
    )
    return risk, connected_ratio, robust_margin


def evaluate_flow_preserving_rollout(
    branch: str,
    robots,
    reference_density: float,
    incoming_direction: pygame.Vector2,
    robot_assignment: dict[int, str],
    partition: dict[tuple[int, int], str],
    centers: dict[tuple[int, int], pygame.Vector2],
) -> dict:
    """Run a short non-mutating SPH look-ahead for one proxy subregion.

    Only robots assigned to the candidate Branch region receive the candidate
    EDF/valve control and contribute to branch metrics. Nearby robots from
    other regions are included as softly held SPH boundary context.
    """
    source_robots, primary_ids, context_ids = get_proxy_region_rollout_robots(
        robots,
        branch,
        robot_assignment,
        partition,
        centers,
    )
    if not source_robots or not primary_ids:
        return {
            "predicted_flow": 0.0,
            "density_disturbance": 1.0,
            "velocity_disturbance": 1.0,
            "wall_risk": 1.0,
            "collision_risk": 1.0,
            "rollout_comm": 1.0,
            "rollout_connected_ratio": 0.0,
            "rollout_margin": -COMM_RANGE,
            "stabilization": 1.0,
            "lambda_mode": 1.0,
            "rollout_lambda": STIFFNESS_EXPONENT_SOFT,
            "predicted_entry_ratio": 0.0,
            "rollout_robot_count": 0,
            "proxy_primary_count": 0,
            "proxy_context_count": 0,
        }

    rollout_lambda, lambda_mode_cost = rollout_stiffness_for_branch(
        branch,
        incoming_direction,
    )
    particles = [
        RolloutParticle(
            robot_id=robot.robot_id,
            position=robot.position.copy(),
            velocity=robot.velocity.copy(),
            initial_position=robot.position.copy(),
            initial_velocity=robot.velocity.copy(),
            initial_density=max(robot.density, EPSILON),
            is_primary=robot.robot_id in primary_ids,
        )
        for robot in source_robots
    ]

    density_disturbance_sum = 0.0
    velocity_disturbance_sum = 0.0
    wall_risk_sum = 0.0
    collision_risk_sum = 0.0
    density_samples = 0
    velocity_samples = 0
    collision_samples = 0
    primary_count = sum(particle.is_primary for particle in particles)
    h_squared = SMOOTHING_LENGTH**2

    for _ in range(FLOW_ROLLOUT_STEPS):
        compute_rollout_densities(particles)
        compute_rollout_pressures(
            particles,
            reference_density,
            rollout_lambda,
        )
        accelerations = [pygame.Vector2() for _ in particles]

        for i in range(len(particles)):
            for j in range(i + 1, len(particles)):
                particle_i = particles[i]
                particle_j = particles[j]
                r_ij = particle_i.position - particle_j.position
                distance_squared = r_ij.length_squared()
                pair_is_relevant = particle_i.is_primary or particle_j.is_primary
                if distance_squared <= EPSILON:
                    if pair_is_relevant:
                        collision_risk_sum += 1.0
                        collision_samples += 1
                    continue

                distance = math.sqrt(distance_squared)
                if pair_is_relevant and distance < FLOW_ROLLOUT_COLLISION_DISTANCE:
                    penetration = (
                        FLOW_ROLLOUT_COLLISION_DISTANCE - distance
                    ) / max(FLOW_ROLLOUT_COLLISION_DISTANCE, EPSILON)
                    collision_risk_sum += penetration**2
                    collision_samples += 1

                if distance_squared > h_squared:
                    continue

                gradient = spiky_gradient(r_ij, SMOOTHING_LENGTH)
                pressure_coefficient = (
                    particle_i.pressure / max(particle_i.density**2, EPSILON)
                    + particle_j.pressure / max(particle_j.density**2, EPSILON)
                )
                pressure_force = -pressure_coefficient * gradient
                accelerations[i] += pressure_force
                accelerations[j] -= pressure_force

                v_ij = particle_i.velocity - particle_j.velocity
                approach_value = v_ij.dot(r_ij)
                if approach_value < 0.0:
                    mu_ij = (
                        SMOOTHING_LENGTH
                        * approach_value
                        / (distance_squared + 0.01 * SMOOTHING_LENGTH**2)
                    )
                    c_i_squared = (
                        particle_i.pressure
                        + PRESSURE_GAIN * particle_i.density
                    ) / max(particle_i.density, EPSILON)
                    c_j_squared = (
                        particle_j.pressure
                        + PRESSURE_GAIN * particle_j.density
                    ) / max(particle_j.density, EPSILON)
                    c_ij = 0.5 * (
                        math.sqrt(max(c_i_squared, 0.0))
                        + math.sqrt(max(c_j_squared, 0.0))
                    )
                    mean_density = 0.5 * (
                        particle_i.density + particle_j.density
                    )
                    pi_ij = (
                        -VISCOSITY_XI1 * c_ij * mu_ij
                        + VISCOSITY_XI2 * mu_ij**2
                    ) / max(mean_density, EPSILON)
                    viscosity_force = -pi_ij * gradient
                    accelerations[i] += viscosity_force
                    accelerations[j] -= viscosity_force

                if distance < SAFE_RADIUS:
                    direction_away = r_ij / distance
                    penetration_ratio = (SAFE_RADIUS - distance) / SAFE_RADIUS
                    repulsion = (
                        REPULSION_GAIN
                        * penetration_ratio
                        * direction_away
                    )
                    accelerations[i] += repulsion
                    accelerations[j] -= repulsion

        for index, particle in enumerate(particles):
            region = get_robot_region(particle.position)
            if particle.is_primary:
                route_force = (
                    rollout_edf_direction(particle.position, branch)
                    * FLOW_ROLLOUT_ROUTE_GAIN
                )
                route_force += rollout_virtual_valve_force(
                    particle.position,
                    branch,
                )
            else:
                # Context particles approximate the neighboring fluid boundary
                # rather than being virtually assigned to this candidate.
                route_force = (
                    particle.initial_position - particle.position
                ) * PROXY_CONTEXT_HOLD_GAIN

            if region in {"UP", "BOTTOM"}:
                route_force.x += CENTERING_GAIN * (center_x - particle.position.x)
            elif region in {"LEFT", "RIGHT"}:
                route_force.y += CENTERING_GAIN * (center_y - particle.position.y)

            acceleration = (
                accelerations[index]
                + route_force
                - DAMPING * particle.velocity
            )
            limit_vector(acceleration, FLOW_ROLLOUT_MAX_ACCELERATION)
            particle.velocity += acceleration * FLOW_ROLLOUT_DT
            speed_limit = (
                FLOW_ROLLOUT_MAX_SPEED
                if particle.is_primary
                else FLOW_ROLLOUT_MAX_SPEED * PROXY_CONTEXT_MAX_SPEED_SCALE
            )
            limit_vector(particle.velocity, speed_limit)

            proposed_x = pygame.Vector2(
                particle.position.x + particle.velocity.x * FLOW_ROLLOUT_DT,
                particle.position.y,
            )
            if is_rollout_walkable(proposed_x, ROBOT_RADIUS, branch):
                particle.position.x = proposed_x.x
            else:
                if particle.is_primary:
                    wall_risk_sum += 1.0
                particle.velocity.x = 0.0

            proposed_y = pygame.Vector2(
                particle.position.x,
                particle.position.y + particle.velocity.y * FLOW_ROLLOUT_DT,
            )
            if is_rollout_walkable(proposed_y, ROBOT_RADIUS, branch):
                particle.position.y = proposed_y.y
            else:
                if particle.is_primary:
                    wall_risk_sum += 1.0
                particle.velocity.y = 0.0

            if particle.is_primary and not is_rollout_walkable(
                particle.position,
                ROBOT_RADIUS + FLOW_ROLLOUT_WALL_CLEARANCE,
                branch,
            ):
                wall_risk_sum += 0.35

            if particle.is_primary:
                relative_density_change = (
                    particle.density - particle.initial_density
                ) / max(particle.initial_density, reference_density, EPSILON)
                density_disturbance_sum += relative_density_change**2
                density_samples += 1

                velocity_change = particle.velocity - particle.initial_velocity
                velocity_disturbance_sum += (
                    velocity_change.length()
                    / max(FLOW_ROLLOUT_VELOCITY_NORMALIZER, EPSILON)
                ) ** 2
                velocity_samples += 1

    compute_rollout_densities(particles)

    entrance = get_branch_entrance(branch)
    direction = BRANCH_DIRECTIONS[branch]
    weighted_flux = 0.0
    weight_sum = 0.0
    entered = 0
    for particle in particles:
        if not particle.is_primary:
            continue
        distance = particle.position.distance_to(entrance)
        weight = math.exp(
            -(distance**2)
            / max(2.0 * FLOW_ROLLOUT_GATE_SIGMA**2, EPSILON)
        )
        weighted_flux += weight * max(0.0, particle.velocity.dot(direction))
        weight_sum += weight
        if get_robot_region(particle.position) == branch:
            entered += 1

    predicted_flow = clamp(
        weighted_flux
        / max(weight_sum * FLOW_ROLLOUT_REFERENCE_SPEED, EPSILON),
        0.0,
        1.0,
    )
    density_disturbance = clamp(
        (density_disturbance_sum / max(density_samples, 1))
        / max(FLOW_ROLLOUT_DENSITY_NORMALIZER**2, EPSILON),
        0.0,
        1.0,
    )
    velocity_disturbance = clamp(
        velocity_disturbance_sum / max(velocity_samples, 1),
        0.0,
        1.0,
    )
    wall_risk = clamp(
        wall_risk_sum
        / max(primary_count * FLOW_ROLLOUT_STEPS * 2.35, 1.0),
        0.0,
        1.0,
    )
    collision_risk = (
        clamp(collision_risk_sum / max(collision_samples, 1), 0.0, 1.0)
        if collision_samples
        else 0.0
    )
    rollout_comm, connected_ratio, robust_margin = evaluate_rollout_communication_risk(
        particles,
        robots,
    )
    stabilization = clamp(
        0.5 * density_disturbance + 0.5 * velocity_disturbance,
        0.0,
        1.0,
    )

    return {
        "predicted_flow": predicted_flow,
        "density_disturbance": density_disturbance,
        "velocity_disturbance": velocity_disturbance,
        "wall_risk": wall_risk,
        "collision_risk": collision_risk,
        "rollout_comm": rollout_comm,
        "rollout_connected_ratio": connected_ratio,
        "rollout_margin": robust_margin,
        "stabilization": stabilization,
        "lambda_mode": lambda_mode_cost,
        "rollout_lambda": rollout_lambda,
        "predicted_entry_ratio": entered / max(primary_count, 1),
        "rollout_robot_count": len(particles),
        "proxy_primary_count": primary_count,
        "proxy_context_count": len(context_ids),
    }


def branch_efficiency_cost(
    branch: str,
    robots,
    incoming_direction: pygame.Vector2,
    reference_density: float,
    proxy_mass_stats: dict[str, dict],
    robot_assignment: dict[int, str],
    partition: dict[tuple[int, int], str],
    centers: dict[tuple[int, int], pygame.Vector2],
):
    ordering_robots = get_branch_ordering_robots(robots)
    region_robots = [
        robot for robot in ordering_robots
        if robot_assignment.get(robot.robot_id) == branch
    ]
    if not region_robots:
        region_robots = ordering_robots

    region_axis, region_shape_confidence = compute_swarm_principal_axis(
        region_robots
    )
    rollout = evaluate_flow_preserving_rollout(
        branch,
        robots,
        reference_density,
        incoming_direction,
        robot_assignment,
        partition,
        centers,
    )

    # Every state prior is now measured from this Branch's proxy-region mass.
    transport = compute_transport_cost(branch, region_robots)
    proxy_mass = proxy_mass_stats.get(branch, {}).get("mass_deficit_cost", 1.0)
    shape = compute_shape_cost(
        branch,
        region_axis,
        region_shape_confidence,
    )
    flow_prior = compute_flow_cost(branch, region_robots)
    congestion = compute_congestion_cost(
        branch,
        region_robots,
        reference_density,
    )

    relay_count = max(
        0,
        math.ceil(BRANCH_LENGTHS[branch] / RELAY_SPACING) - 1,
    )
    max_relays = max(
        1,
        max(
            math.ceil(length / RELAY_SPACING) - 1
            for length in BRANCH_LENGTHS.values()
        ),
    )
    relay = relay_count / max_relays
    backtrack = BRANCH_LENGTHS[branch] / max(BRANCH_LENGTHS.values())
    switch = angle_between(
        incoming_direction,
        BRANCH_DIRECTIONS[branch],
    ) / math.pi

    total = (
        -BRANCH_COST_PREDICTED_FLOW_REWARD * rollout["predicted_flow"]
        + BRANCH_COST_DENSITY_DISTURBANCE_WEIGHT * rollout["density_disturbance"]
        + BRANCH_COST_VELOCITY_DISTURBANCE_WEIGHT * rollout["velocity_disturbance"]
        + BRANCH_COST_WALL_RISK_WEIGHT * rollout["wall_risk"]
        + BRANCH_COST_COLLISION_RISK_WEIGHT * rollout["collision_risk"]
        + BRANCH_COST_ROLLOUT_COMM_WEIGHT * rollout["rollout_comm"]
        + BRANCH_COST_RELAY_WEIGHT * relay
        + BRANCH_COST_LAMBDA_MODE_WEIGHT * rollout["lambda_mode"]
        + BRANCH_COST_STABILIZATION_WEIGHT * rollout["stabilization"]
        + BRANCH_COST_TRANSPORT_WEIGHT * transport
        + BRANCH_COST_PROXY_MASS_WEIGHT * proxy_mass
        + BRANCH_COST_SHAPE_WEIGHT * shape
        + BRANCH_COST_FLOW_PRIOR_WEIGHT * flow_prior
        + BRANCH_COST_CONGESTION_WEIGHT * congestion
        + BRANCH_COST_BACKTRACK_WEIGHT * backtrack
        + BRANCH_COST_SWITCH_WEIGHT * switch
    )

    components = {
        **rollout,
        "transport": transport,
        "proxy_mass": proxy_mass,
        "proxy_quota": proxy_mass_stats.get(branch, {}).get("quota_fraction", 0.0),
        "proxy_actual_mass": proxy_mass_stats.get(branch, {}).get("actual_mass_fraction", 0.0),
        "proxy_robot_count": proxy_mass_stats.get(branch, {}).get("robot_count", 0),
        "shape": shape,
        "shape_confidence": region_shape_confidence,
        "flow_prior": flow_prior,
        "congestion": congestion,
        "relay": relay,
        "backtrack": backtrack,
        "switch": switch,
        "structural_loss": structural_loss(branch),
        "ordering_robot_count": len(ordering_robots),
        "regional_robot_count": len(region_robots),
    }
    return total, components


def branch_is_feasible(branch: str, robots) -> bool:
    """Check deterministic resource/connectivity feasibility before scoring."""
    connected_normals = [
        robot
        for robot in robots
        if robot.role == "NORMAL" and robot.connected_to_base
    ]
    required_branch_relays = max(
        0,
        math.ceil(BRANCH_LENGTHS[branch] / RELAY_SPACING) - 1,
    )
    required_mobile_roles = required_branch_relays + adaptive_shepherd_count()
    return len(connected_normals) >= required_mobile_roles


def update_distributed_branch_consensus(robots) -> Optional[str]:
    """Run a local NORMAL-to-NORMAL vote without an Anchor decision maker."""
    global distributed_consensus_branch
    voters = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and robot.connected_to_base
        and get_robot_region(robot.position) == "JUNCTION"
    ]
    if len(voters) < DISTRIBUTED_VOTE_MIN_ROBOTS:
        return None

    for robot in voters:
        preferred = next(
            (
                branch
                for branch in FIXED_BRANCH_ORDER
                if robot.local_branch_states.get(branch) == "UNVISITED"
            ),
            None,
        )
        local_peers = [
            peer
            for peer in robot.comm_neighbors
            if getattr(peer, "role", None) == "NORMAL"
            and get_robot_region(peer.position) == "JUNCTION"
            and robot.position.distance_to(peer.position)
            <= DISTRIBUTED_VOTE_NEIGHBOR_RANGE
        ]
        peer_votes = [
            peer.branch_vote
            for peer in local_peers
            if peer.branch_vote in FIXED_BRANCH_ORDER
            and peer.local_branch_states.get(peer.branch_vote) == "UNVISITED"
        ]
        counts = {
            branch: peer_votes.count(branch)
            for branch in FIXED_BRANCH_ORDER
            if robot.local_branch_states.get(branch) == "UNVISITED"
        }
        if preferred is not None:
            counts[preferred] = counts.get(preferred, 0) + 1
        if counts:
            robot.branch_vote = max(
                counts,
                key=lambda branch: (
                    counts[branch],
                    -FIXED_BRANCH_ORDER.index(branch),
                ),
            )
            robot.branch_vote_confidence = (
                counts[robot.branch_vote] / max(sum(counts.values()), 1)
            )

    vote_counts = {
        branch: sum(robot.branch_vote == branch for robot in voters)
        for branch in FIXED_BRANCH_ORDER
    }
    selected = max(
        FIXED_BRANCH_ORDER,
        key=lambda branch: (
            vote_counts[branch],
            -FIXED_BRANCH_ORDER.index(branch),
        ),
    )
    quorum = max(
        DISTRIBUTED_VOTE_MIN_ROBOTS,
        math.ceil(len(voters) * DISTRIBUTED_VOTE_QUORUM_RATIO),
    )
    if vote_counts[selected] < quorum:
        return None

    distributed_consensus_branch = selected
    for robot in voters:
        robot.distributed_branch_decision = selected
    print(
        f"[Distributed Consensus] branch={selected}, "
        f"votes={vote_counts[selected]}/{len(voters)}"
    )
    return selected


def apply_consensus_branch_gates(open_branch: Optional[str]) -> None:
    """Apply the branch-mouth state agreed by the NORMAL peer consensus."""
    global branch_gate_states
    if open_branch is None:
        branch_gate_states = {branch: "OPEN" for branch in BRANCHES}
    else:
        branch_gate_states = {
            branch: "OPEN" if branch == open_branch else "CLOSED"
            for branch in BRANCHES
        }
    print(
        "[Distributed Gate Consensus] "
        + ", ".join(
            f"{branch}={branch_gate_states[branch]}"
            for branch in BRANCHES
        )
    )


def begin_cross_branch_transfer(robots, source: str, target: str) -> None:
    """Open the next branch while Shepherd pressure expels the source fluid."""
    global branch_gate_states, transfer_branch, final_base_transfer_active
    transfer_branch = target
    final_base_transfer_active = False
    branch_gate_states = {
        branch: (
            "OPEN"
            if branch in {source, target}
            else "CLOSED"
        )
        for branch in BRANCHES
    }
    for robot in robots:
        robot.transfer_target = (
            target
            if get_robot_region(robot.position) == source
            and robot.role in {"NORMAL", "SHEPHERD"}
            else None
        )
    print(
        f"[Cross-Branch Transfer] {source} -> {target}; "
        f"{source}=OPEN, {target}=OPEN"
    )


def begin_final_base_transfer(robots, source: str) -> None:
    """Use the final Shepherd piston to expel LEFT fluid toward Base."""
    global branch_gate_states, transfer_branch, final_base_transfer_active
    transfer_branch = None
    final_base_transfer_active = True
    branch_gate_states = {
        branch: "OPEN" if branch == source else "CLOSED"
        for branch in BRANCHES
    }
    for robot in robots:
        robot.transfer_target = (
            "BOTTOM"
            if get_robot_region(robot.position) == source
            and robot.role in {"NORMAL", "SHEPHERD"}
            else None
        )
    print(
        f"[Final Base Transfer] {source} -> BASE; "
        "UP=CLOSED, RIGHT=CLOSED"
    )


def close_all_branch_gates() -> None:
    global branch_gate_states
    branch_gate_states = {branch: "CLOSED" for branch in BRANCHES}
    print("[Gate] UP=CLOSED, LEFT=CLOSED, RIGHT=CLOSED")


def direction_toward_base_path(position: pygame.Vector2) -> pygame.Vector2:
    """Piecewise free-space direction from a branch/Junction to Base."""
    region = get_robot_region(position)
    if region in BRANCHES:
        return normalized_direction_toward(
            position,
            get_branch_entrance(region),
        )
    if region == "JUNCTION":
        return normalized_direction_toward(
            position,
            pygame.Vector2(center_x, center_y + half_width),
        )
    if region == "BOTTOM":
        return normalized_direction_toward(
            position,
            get_bottom_hold_point(),
        )
    return pygame.Vector2()


def get_sequence_stage() -> int:
    """Return the six-stage state shown in the user's flow sketch."""
    if final_base_transfer_active:
        return 5
    if phase in {SimulationPhase.RETURN_TO_BASE, SimulationPhase.DONE}:
        return 6
    if active_branch == "RIGHT":
        return (
            2
            if transfer_branch == "UP"
            and phase == SimulationPhase.FLOW_BACKTRACK
            else 1
        )
    if active_branch == "UP":
        return (
            3
            if transfer_branch == "LEFT"
            and phase in {
                SimulationPhase.PRESSURE_PUSH,
                SimulationPhase.FLOW_BACKTRACK,
            }
            else 2
        )
    if active_branch == "LEFT":
        return 4
    return 1


def next_unvisited_transfer_branch(source: str) -> Optional[str]:
    start = FIXED_BRANCH_ORDER.index(source) + 1
    return next(
        (
            branch
            for branch in FIXED_BRANCH_ORDER[start:]
            if branch_states.get(branch) == "UNVISITED"
        ),
        None,
    )


def choose_next_branch(anchor, robots, reference_density: float):
    """Select the next unvisited branch using the fixed DFS route.

    The branch route is always Base -> RIGHT -> UP -> LEFT -> Base.
    Proxy partitioning, demand-ratio allocation, and rollout scoring do not
    participate in branch ordering.
    """
    global active_branch, previous_branch_direction, branch_order_plan
    global last_proxy_partition, last_proxy_cell_centers
    global last_proxy_mass_stats, last_proxy_robot_assignment
    global last_proxy_candidates
    global last_flow_rollout_scores
    global selected_branch_entry_lambda, branch_entry_timer

    selected = distributed_consensus_branch
    if selected is None or branch_states.get(selected) != "UNVISITED":
        if anchor is not None:
            anchor.selected_branch = None
        return None

    if not branch_is_feasible(selected, robots):
        print(
            f"[DFS] warning: fixed branch {selected} did not pass resource "
            "feasibility; preserving the fixed order"
        )

    # Clear decision-time analytical displays because fixed ordering does not
    # construct or score temporary proxy regions.
    last_proxy_partition = {}
    last_proxy_cell_centers = {}
    last_proxy_mass_stats = {}
    last_proxy_robot_assignment = {}
    last_proxy_candidates = ()
    last_flow_rollout_scores = {}

    branch_states[selected] = "ACTIVE"
    for robot in robots:
        robot.local_branch_states[selected] = "ACTIVE"
        if robot.role == "NORMAL":
            robot.distributed_branch_decision = selected
    if anchor is not None:
        anchor.local_branch_states[selected] = "ACTIVE"
        anchor.selected_branch = selected
    apply_consensus_branch_gates(selected)
    active_branch = selected
    branch_order_plan.append(selected)
    selected_branch_entry_lambda, _ = rollout_stiffness_for_branch(
        selected,
        previous_branch_direction,
    )
    branch_entry_timer = 0.0
    initialize_relay_plan(selected)

    print(
        f"[DFS] fixed-order selected={selected}, "
        f"entry_lambda={selected_branch_entry_lambda:.3f}"
    )
    return selected

def complete_active_branch(anchor, branch, robots):
    global previous_branch_direction, distributed_consensus_branch
    branch_states[branch] = "VISITED"
    distributed_consensus_branch = None
    for robot in robots:
        robot.local_branch_states[branch] = "VISITED"
        robot.branch_vote = None
        robot.branch_vote_confidence = 0.0
        robot.distributed_branch_decision = None
    if anchor is not None:
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
    global transfer_branch, final_base_transfer_active
    relay_slots = []
    relay_motion_scale = 1.0
    final_gather_timer = 0.0
    transfer_branch = None
    final_base_transfer_active = False
    apply_consensus_branch_gates(None)
    phase = SimulationPhase.FINAL_JUNCTION_GATHER
    print("[DFS] final gather")


def begin_final_return(anchor, robots):
    global phase, relay_slots, relay_motion_scale
    global return_trunk_release_pending, return_trunk_retract_timer, return_trunk_last_released_id, return_trunk_force_timer
    relay_slots = []
    relay_motion_scale = 1.0
    close_all_branch_gates()
    release_anchor_for_final_return(anchor)
    return_trunk_release_pending = True
    return_trunk_retract_timer = 0.0
    return_trunk_last_released_id = None
    return_trunk_force_timer = 0.0
    phase = SimulationPhase.RETURN_TO_BASE
    print("[DFS] return to base")

# =========================================================
# 12-1. Junction stability consensus
# =========================================================


@dataclass
class JunctionConsensusTracker:
    dwell: float = 0.0
    fast_dwell: float = 0.0
    elapsed: float = 0.0
    candidate_count: int = 0
    stable_ratio: float = 0.0
    mean_speed: float = 0.0
    mean_density_delta_ratio: float = 0.0
    ready: bool = False
    ready_mode: str = "WAIT"
    previous_density: dict[int, float] = field(default_factory=dict)

    def reset(self):
        self.dwell = 0.0
        self.fast_dwell = 0.0
        self.elapsed = 0.0
        self.candidate_count = 0
        self.stable_ratio = 0.0
        self.mean_speed = 0.0
        self.mean_density_delta_ratio = 0.0
        self.ready = False
        self.ready_mode = "WAIT"
        self.previous_density.clear()

    def update(self, robots, dt: float, reference_density: float) -> bool:
        self.elapsed += dt
        candidates = [
            robot
            for robot in robots
            if robot.role == "NORMAL"
            and get_robot_region(robot.position) == "JUNCTION"
        ]
        self.candidate_count = len(candidates)
        if not candidates:
            self.stable_ratio = 0.0
            self.mean_speed = 0.0
            self.mean_density_delta_ratio = float("inf")
            self.dwell = 0.0
            self.ready = False
            return False

        stable_count = 0
        density_deltas = []
        speeds = []
        for robot in candidates:
            speed = robot.velocity.length()
            previous = self.previous_density.get(robot.robot_id, robot.density)
            density_delta_ratio = abs(robot.density - previous) / max(reference_density, EPSILON)
            self.previous_density[robot.robot_id] = robot.density
            speeds.append(speed)
            density_deltas.append(density_delta_ratio)
            if (
                robot.connected_to_base
                and speed <= JUNCTION_CONSENSUS_SPEED_THRESHOLD
                and density_delta_ratio <= JUNCTION_CONSENSUS_DENSITY_DELTA_RATIO
            ):
                stable_count += 1

        self.stable_ratio = stable_count / len(candidates)
        self.mean_speed = sum(speeds) / len(speeds)
        self.mean_density_delta_ratio = sum(density_deltas) / len(density_deltas)
        normal_conditions = (
            self.candidate_count >= JUNCTION_CONSENSUS_MIN_COUNT
            and self.stable_ratio >= JUNCTION_CONSENSUS_STABLE_RATIO
            and self.mean_speed <= JUNCTION_CONSENSUS_SPEED_THRESHOLD
            and self.mean_density_delta_ratio <= JUNCTION_CONSENSUS_DENSITY_DELTA_RATIO
        )
        fast_conditions = (
            self.candidate_count >= JUNCTION_FAST_READY_MIN_COUNT
            and self.stable_ratio >= JUNCTION_FAST_READY_STABLE_RATIO
            and self.mean_speed <= JUNCTION_FAST_READY_SPEED_THRESHOLD
            and self.mean_density_delta_ratio
            <= JUNCTION_FAST_READY_DENSITY_DELTA_RATIO
        )

        self.dwell = self.dwell + dt if normal_conditions else 0.0
        self.fast_dwell = self.fast_dwell + dt if fast_conditions else 0.0

        normal_ready = self.dwell >= JUNCTION_CONSENSUS_DWELL_TIME
        fast_ready = self.fast_dwell >= JUNCTION_FAST_READY_DWELL_TIME
        self.ready = normal_ready or fast_ready
        self.ready_mode = (
            "FAST" if fast_ready
            else "STABLE" if normal_ready
            else "WAIT"
        )
        return self.ready


junction_consensus_tracker = JunctionConsensusTracker()

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
    packed_min_count: int = SATURATION_PACKED_MIN_TIP_ROBOTS
    recognition_mode: str = "WAIT"
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
        self.packed_min_count = SATURATION_PACKED_MIN_TIP_ROBOTS
        self.recognition_mode = "WAIT"
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

    stalled_conditions = (
        tracker.tip_count >= SATURATION_MIN_TIP_ROBOTS
        and tracker.low_speed_ratio >= SATURATION_LOW_SPEED_RATIO
        and tracker.average_density_ratio >= SATURATION_DENSITY_RATIO
        and tracker.occupancy_ratio >= SATURATION_OCCUPANCY_RATIO
        and tracker.front_delta <= SATURATION_FRONT_PROGRESS_EPSILON
    )
    shepherd_count = sum(
        robot.role == "SHEPHERD"
        and get_robot_region(robot.position) == branch
        for robot in robots
    )
    tracker.packed_min_count = max(
        SATURATION_PACKED_MIN_TIP_ROBOTS,
        math.ceil(
            max(1, shepherd_count)
            * SATURATION_PACKED_ROBOTS_PER_SHEPHERD
        ),
    )
    density_evidence = (
        tracker.average_density_ratio >= SATURATION_PACKED_DENSITY_RATIO
    )
    geometry_evidence = (
        tracker.tip_count >= tracker.packed_min_count
    )
    packed_conditions = (
        tracker.tip_count >= SATURATION_PACKED_MIN_TIP_ROBOTS
        and tracker.occupancy_ratio >= SATURATION_PACKED_OCCUPANCY_RATIO
        and (density_evidence or geometry_evidence)
    )
    if packed_conditions:
        tracker.recognition_mode = (
            "PACKED_DENSITY"
            if density_evidence
            else "PACKED_GEOMETRY"
        )
        required_dwell = SATURATION_PACKED_DWELL_TIME
    elif stalled_conditions:
        tracker.recognition_mode = "STALLED"
        required_dwell = SATURATION_DWELL_TIME
    else:
        tracker.recognition_mode = "WAIT"
        required_dwell = SATURATION_DWELL_TIME

    recognized = packed_conditions or stalled_conditions
    tracker.dwell = tracker.dwell + dt if recognized else 0.0
    tracker.saturated = tracker.dwell >= required_dwell
    if tracker.saturated:
        metrics.saturation_events.append({
            "branch": branch,
            "time": simulation_time,
            "mode": tracker.recognition_mode,
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
            robot.filtered_acceleration.update(0.0, 0.0)
        robot.shepherd_anchor = None
        robot.shepherd_origin = None



def shepherd_candidates(robots, branch, required_count):
    """Return the leading robots already inside the original capture region."""
    capture_rect = early_capture_regions[branch]
    candidates = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
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
        and get_robot_region(robot.position) == branch
        and capture_rect.collidepoint(robot.position.x, robot.position.y)
    ]
    if len(candidates) < required_count:
        return False

    # The physical co-location inside the narrow dead-end capture region is
    # sufficient for immediate peer role election. No Base path, hop count,
    # communication margin, or global message is required.
    return True


def assign_shepherd_slots(candidates, slots):
    """Peer-auction Shepherd self-election using only local slot bids.

    Every NORMAL candidate proposes its nearest unclaimed slot. Conflicts are
    resolved deterministically by distance and robot id, equivalent to a local
    winner announcement among nearby peers.
    """
    if len(candidates) < len(slots):
        return []

    unassigned = list(candidates)
    open_slots = set(range(len(slots)))
    assignment = []
    while open_slots and unassigned:
        proposals: dict[int, list["Robot"]] = {}
        for robot in unassigned:
            slot_index = min(
                open_slots,
                key=lambda index: (
                    robot.position.distance_squared_to(slots[index]),
                    index,
                ),
            )
            proposals.setdefault(slot_index, []).append(robot)

        winners = []
        for slot_index, bidders in proposals.items():
            winner = min(
                bidders,
                key=lambda robot: (
                    robot.position.distance_squared_to(slots[slot_index]),
                    robot.robot_id,
                ),
            )
            score = winner.position.distance_to(slots[slot_index])
            assignment.append((winner, slots[slot_index], score))
            winners.append(winner)
            open_slots.remove(slot_index)
        unassigned = [
            robot for robot in unassigned
            if robot not in winners
        ]
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
        robot.filtered_acceleration.update(0.0, 0.0)
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


def force_complete_shepherd_boundary(robots) -> bool:
    """Finish local slot claims without consulting the Base network."""
    for robot in get_shepherds(robots):
        if robot.shepherd_anchor is None:
            return False
        if not is_walkable(robot.shepherd_anchor, robot.radius):
            return False
    for robot in get_shepherds(robots):
        robot.position = robot.shepherd_anchor.copy()
        robot.velocity.update(0.0, 0.0)
        robot.acceleration.update(0.0, 0.0)
    return shepherd_boundary_formed(robots)


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
    """Start continuous line-preserving backtracking to the Junction."""
    global shepherd_flow_timer, shepherd_flow_start_depth
    shepherd_flow_start_depth = get_shepherd_curtain_depth(active_branch)
    shepherd_flow_timer = 0.0
    retained = 0
    for robot in robots:
        if robot.role != "SHEPHERD":
            continue
        robot.velocity.update(0.0, 0.0)
        robot.acceleration.update(0.0, 0.0)
        robot.filtered_acceleration.update(0.0, 0.0)
        retained += 1
    print(
        f"[Shepherd] line retained to Junction={retained}, "
        f"start_depth={shepherd_flow_start_depth:.2f}"
    )


def release_shepherd_line_at_junction(robots) -> int:
    """Turn the intact line back into NORMAL robots inside the Junction."""
    if (
        phase != SimulationPhase.FLOW_BACKTRACK
        or get_shepherd_curtain_depth(active_branch)
        > SHEPHERD_JUNCTION_DEPTH_TOLERANCE
    ):
        return 0

    direction = get_backtrack_direction(active_branch)
    released = 0
    for robot in robots:
        if robot.role != "SHEPHERD":
            continue
        if robot.shepherd_anchor is not None:
            junction_slot = shepherd_slot_position_at_depth(
                robot.shepherd_anchor,
                active_branch,
                0.0,
            )
            target = (
                junction_slot
                + direction * SHEPHERD_JUNCTION_RELEASE_INSET
            )
            if is_walkable(target, robot.radius):
                robot.position = target
        robot.role = "NORMAL"
        robot.shepherd_anchor = None
        robot.shepherd_origin = None
        robot.velocity = (
            direction * SHEPHERD_JUNCTION_RELEASE_SPEED
        )
        robot.acceleration.update(0.0, 0.0)
        robot.filtered_acceleration.update(0.0, 0.0)
        released += 1

    if released:
        print(
            f"[Shepherd] Junction reached; released to NORMAL={released}, "
            f"next={transfer_branch or 'BASE'}"
        )
    return released

# =========================================================
# 15. SPH
# =========================================================


def compute_densities(robots, grid):
    self_contribution = spiky_kernel(0.0, SMOOTHING_LENGTH)
    h_sq = SMOOTHING_LENGTH**2
    for robot_i in robots:
        density = self_contribution
        for robot_j in iter_physics_neighbor_candidates(robot_i, grid):
            if robot_i is robot_j:
                continue
            distance_sq = robot_i.position.distance_squared_to(robot_j.position)
            if distance_sq <= h_sq:
                density += spiky_kernel(math.sqrt(distance_sq), SMOOTHING_LENGTH)
        robot_i.density = max(density, EPSILON)


def get_effective_stiffness_exponent() -> float:
    """Adaptive lambda schedule for real branch switching and pressure push."""
    if phase == SimulationPhase.PRESSURE_PUSH:
        return STIFFNESS_EXPONENT_PRESSURE_PUSH
    if phase == SimulationPhase.JUNCTION_SWITCH:
        return STIFFNESS_EXPONENT_SOFT
    if phase in {
        SimulationPhase.EXPLORE_BRANCH,
        SimulationPhase.FORM_SHEPHERD_BOUNDARY,
        SimulationPhase.FILL_BEHIND_SHEPHERD,
    }:
        recovery = clamp(
            branch_entry_timer / max(BRANCH_STIFFNESS_RECOVERY_TIME, EPSILON),
            0.0,
            1.0,
        )
        return (
            selected_branch_entry_lambda
            + (STIFFNESS_EXPONENT_RIGID - selected_branch_entry_lambda)
            * recovery
        )
    return STIFFNESS_EXPONENT_RIGID


def robot_is_in_shepherd_packing_zone(
    robot: "Robot",
    branch: str,
) -> bool:
    if (
        robot.role != "NORMAL"
        or get_robot_region(robot.position) != branch
    ):
        return False
    boundary_depth = get_shepherd_curtain_depth(branch)
    depth = branch_depth_from_junction(robot.position, branch)
    near_depth = max(
        0.0,
        boundary_depth - SHEPHERD_FILL_REGION_DEPTH,
    )
    return near_depth <= depth <= boundary_depth


def adaptive_equilibrium_radius(robot: "Robot") -> float:
    """Phase-local equilibrium spacing used by pair repulsion."""
    if (
        phase == SimulationPhase.FILL_BEHIND_SHEPHERD
        and robot.role == "NORMAL"
    ):
        if robot_is_in_shepherd_packing_zone(robot, active_branch):
            return max(
                ROBOT_RADIUS * 2.05,
                SAFE_RADIUS * SHEPHERD_PACKED_EQUILIBRIUM_SCALE,
            )
        if get_robot_region(robot.position) in {"JUNCTION", "BOTTOM"}:
            return SAFE_RADIUS * JUNCTION_TAIL_EQUILIBRIUM_SCALE
    return SAFE_RADIUS * NORMAL_EQUILIBRIUM_SCALE


def get_base_pressure_scale() -> float:
    """Continuous compression-release pulse suitable for physical robots."""
    if simulation_time < BASE_COMPRESSION_DURATION:
        return BASE_COMPRESSION_PRESSURE_SCALE

    elapsed = simulation_time - BASE_COMPRESSION_DURATION
    if elapsed >= BASE_EXPANSION_BOOST_DURATION:
        return SPH_MOTION_PRESSURE_BOOST

    progress = clamp(
        elapsed / max(BASE_EXPANSION_BOOST_DURATION, EPSILON),
        0.0,
        1.0,
    )
    rise = smoothstep01(
        progress / max(BASE_EXPANSION_RAMP_FRACTION, EPSILON)
    )
    fall = smoothstep01(
        (
            progress
            - (1.0 - BASE_EXPANSION_RAMP_FRACTION)
        )
        / max(BASE_EXPANSION_RAMP_FRACTION, EPSILON)
    )
    pressure_scale = (
        BASE_COMPRESSION_PRESSURE_SCALE
        + (
            BASE_EXPANSION_PRESSURE_SCALE
            - BASE_COMPRESSION_PRESSURE_SCALE
        )
        * rise
    )
    return pressure_scale + (
        SPH_MOTION_PRESSURE_BOOST - pressure_scale
    ) * fall


def get_base_compression_envelope() -> float:
    """Zero-slope-style pulse: no step impulse at compression start/end."""
    progress = clamp(
        simulation_time / max(BASE_COMPRESSION_DURATION, EPSILON),
        0.0,
        1.0,
    )
    return math.sin(math.pi * progress) ** 2


def compute_initial_junction_soft_wall_force(
    robot: "Robot",
) -> pygame.Vector2:
    """Brake before the three closed Junction faces instead of colliding."""
    if (
        phase != SimulationPhase.MOVE_TO_JUNCTION
        or robot.role != "NORMAL"
        or get_robot_region(robot.position) != "JUNCTION"
    ):
        return pygame.Vector2()

    depth = max(INITIAL_JUNCTION_SOFT_WALL_DEPTH, EPSILON)
    force = pygame.Vector2()
    faces = (
        (
            robot.position.x - junction_rect.left,
            pygame.Vector2(1.0, 0.0),
        ),
        (
            junction_rect.right - robot.position.x,
            pygame.Vector2(-1.0, 0.0),
        ),
        (
            robot.position.y - junction_rect.top,
            pygame.Vector2(0.0, 1.0),
        ),
    )
    for clearance, inward in faces:
        if clearance >= depth:
            continue
        ratio = clamp(1.0 - clearance / depth, 0.0, 1.0)
        outward_speed = max(0.0, -robot.velocity.dot(inward))
        force += inward * (
            INITIAL_JUNCTION_SOFT_WALL_FORCE * ratio**2
            + INITIAL_JUNCTION_SOFT_WALL_DAMPING * outward_speed
        )
    return force


def compute_pressures(robots, reference_density):
    effective_lambda = get_effective_stiffness_exponent()
    for robot in robots:
        ratio = robot.density / max(reference_density, EPSILON)
        robot.density_ratio = ratio
        raw_pressure = (
            PRESSURE_GAIN
            * robot.density
            * (ratio**effective_lambda - 1.0)
        )
        # Negative pressure creates tensile clumping in particle methods.
        # Cohesion is supplied by the explicit viscoelastic links, leaving
        # this SPH channel compressive and numerically stable.
        robot.pressure = max(0.0, raw_pressure)
        if robot.role == "NORMAL":
            if phase == SimulationPhase.MOVE_TO_JUNCTION:
                robot.pressure *= get_base_pressure_scale()
            elif phase == SimulationPhase.EXPLORE_BRANCH:
                robot.pressure *= SPH_MOTION_PRESSURE_BOOST
            elif phase == SimulationPhase.FILL_BEHIND_SHEPHERD:
                if robot_is_in_shepherd_packing_zone(
                    robot,
                    active_branch,
                ):
                    robot.pressure *= (
                        SHEPHERD_FILL_COMPRESSION_PRESSURE_SCALE
                    )
                else:
                    robot.pressure *= SPH_MOTION_PRESSURE_BOOST
            elif phase in {
                SimulationPhase.PRESSURE_PUSH,
                SimulationPhase.FLOW_BACKTRACK,
            }:
                robot.pressure *= SHEPHERD_PACKED_PRESSURE_BOOST
        if (
            phase in {
                SimulationPhase.PRESSURE_PUSH,
                SimulationPhase.FLOW_BACKTRACK,
            }
            and robot.role == "SHEPHERD"
        ):
            ramp = (
                1.0
                if phase == SimulationPhase.FLOW_BACKTRACK
                else smoothstep01(
                    pressure_push_timer
                    / max(PRESSURE_RAMP_TIME, EPSILON)
                )
            )
            robot.pressure += PRESSURE_GAIN * robot.density * SHEPHERD_PRESSURE_FACTOR * ramp


def compute_route_force(robot):
    region = get_robot_region(robot.position)
    junction_target = pygame.Vector2(center_x, center_y)
    force = pygame.Vector2()
    if robot.role in {"ANCHOR", "RELAY", "TRUNK_RELAY"}:
        return force
    if (
        robot.base_reserve
        and robot.base_hold_position is not None
        and phase not in {
            SimulationPhase.RETURN_TO_BASE,
            SimulationPhase.DONE,
        }
        and not (
            phase == SimulationPhase.MOVE_TO_JUNCTION
            and simulation_time < BASE_COMPRESSION_DURATION
        )
    ):
        hold_error = robot.base_hold_position - robot.position
        return limit_vector(
            hold_error * BASE_RESERVE_HOLD_GAIN,
            BASE_COMPRESSION_FORCE,
        )
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
        }:
            if robot.received_branch != active_branch:
                return force
            if (
                robot.received_gate_states is None
                or robot.received_gate_states.get(active_branch) != "OPEN"
            ):
                return force

    if phase == SimulationPhase.MOVE_TO_JUNCTION:
        if simulation_time < BASE_COMPRESSION_DURATION:
            compression = BASE_COMPRESSION_CENTER - robot.position
            distance = compression.length()
            if distance > EPSILON:
                compression_scale = clamp(
                    distance / max(half_width * 0.55, EPSILON),
                    0.15,
                    1.0,
                )
                force = (
                    compression.normalize()
                    * BASE_COMPRESSION_FORCE
                    * compression_scale
                    * get_base_compression_envelope()
                )
        else:
            y_distance = robot.position.y - INITIAL_INGRESS_TARGET_Y
            if y_distance > 0.0:
                scale = max(
                    INITIAL_INGRESS_MIN_FORCE_SCALE,
                    min(
                        1.0,
                        y_distance / INITIAL_INGRESS_BRAKE_DISTANCE,
                    ),
                )
                force.y = -INITIAL_INGRESS_FORCE * scale
            lane_error = robot.ingress_lane_x - robot.position.x
            force.x = clamp(
                INITIAL_INGRESS_LANE_GAIN * lane_error,
                -INITIAL_INGRESS_LANE_MAX_FORCE,
                INITIAL_INGRESS_LANE_MAX_FORCE,
            )
    elif phase == SimulationPhase.EXPLORE_BRANCH:
        if (
            transfer_branch == active_branch
            and robot.transfer_target != active_branch
        ):
            # Base-side robots receive no artificial suction. With no
            # Shepherd behind them, the low-density tail remains near Base.
            force = pygame.Vector2()
        else:
            force = (
                geodesic_edf_direction(robot.position, active_branch)
                * ROUTE_FORCE
                * relay_motion_scale
            )
    elif phase == SimulationPhase.FORM_SHEPHERD_BOUNDARY:
        if (
            transfer_branch == active_branch
            and robot.transfer_target != active_branch
        ):
            force = pygame.Vector2()
        elif robot.role == "SHEPHERD":
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
        if (
            transfer_branch == active_branch
            and robot.transfer_target != active_branch
        ):
            force = pygame.Vector2()
        elif robot.role == "SHEPHERD":
            force = pygame.Vector2()
        elif region == active_branch:
            force = (
                geodesic_edf_direction(
                    robot.position,
                    active_branch,
                    get_shepherd_fill_target(active_branch),
                )
                * ROUTE_FORCE
                * relay_motion_scale
                * SHEPHERD_FILL_FORCE_MULTIPLIER
            )
        else:
            force = (
                geodesic_edf_direction(
                    robot.position,
                    active_branch,
                    get_shepherd_fill_target(active_branch),
                )
                * OUTLET_FORCE
            )
    elif phase == SimulationPhase.PRESSURE_PUSH:
        if (
            final_base_transfer_active
            and robot.transfer_target == "BOTTOM"
            and region == active_branch
        ):
            force = (
                get_backtrack_direction(active_branch)
                * PRESSURE_BACKTRACK_BODY_FORCE
            )
        elif (
            final_base_transfer_active
            and robot.transfer_target == "BOTTOM"
            and region in {"JUNCTION", "BOTTOM"}
        ):
            force = direction_toward_base_path(robot.position) * ROUTE_FORCE
        elif (
            robot.transfer_target == transfer_branch
            and region == active_branch
        ):
            force = get_backtrack_direction(active_branch) * PRESSURE_BACKTRACK_BODY_FORCE
        elif (
            transfer_branch is not None
            and robot.transfer_target == transfer_branch
            and region in {"JUNCTION", transfer_branch}
        ):
            force = (
                geodesic_edf_direction(robot.position, transfer_branch)
                * ROUTE_FORCE
            )
        else:
            force = pygame.Vector2()
    elif phase == SimulationPhase.FLOW_BACKTRACK:
        if final_base_transfer_active:
            force = (
                direction_toward_base_path(robot.position) * ROUTE_FORCE
                if robot.transfer_target == "BOTTOM"
                else pygame.Vector2()
            )
        elif transfer_branch is not None:
            force = (
                geodesic_edf_direction(robot.position, transfer_branch)
                * ROUTE_FORCE
                if robot.transfer_target == transfer_branch
                else pygame.Vector2()
            )
        else:
            target = junction_target if region == active_branch else JUNCTION_STAGING_POSITION
            force = normalized_direction_toward(robot.position, target) * FLOW_BACKTRACK_FORCE
    elif phase == SimulationPhase.JUNCTION_SWITCH:
        force = normalized_direction_toward(robot.position, JUNCTION_STAGING_POSITION) * OUTLET_FORCE
    elif phase == SimulationPhase.FINAL_JUNCTION_GATHER:
        target = junction_target if region in BRANCHES else JUNCTION_STAGING_POSITION
        force = normalized_direction_toward(robot.position, target) * FINAL_GATHER_FORCE
    elif phase == SimulationPhase.RETURN_TO_BASE:
        if region in BRANCHES:
            force = normalized_direction_toward(robot.position, junction_target) * OUTLET_FORCE
        else:
            bottom_target = get_bottom_hold_point()
            y_distance = bottom_target.y - robot.position.y
            if y_distance > 0.0:
                scale = max(RETURN_MIN_FORCE_SCALE, min(1.0, y_distance / RETURN_BRAKE_DISTANCE))
                force.y = RETURN_EGRESS_FORCE * scale
            lane_error = robot.ingress_lane_x - robot.position.x
            force.x = clamp(RETURN_LANE_GAIN * lane_error, -RETURN_LANE_MAX_FORCE, RETURN_LANE_MAX_FORCE)

    force += compute_virtual_valve_force(robot)

    if region in {"UP", "BOTTOM"}:
        force.x += CENTERING_GAIN * (center_x - robot.position.x)
    elif region in {"LEFT", "RIGHT"}:
        force.y += CENTERING_GAIN * (center_y - robot.position.y)
    return force


def compute_sph_forces(
    robots,
    grid,
    communication_grid,
    dt=1.0 / FPS,
):
    global viscoelastic_step
    viscoelastic_step += 1
    h_sq = SMOOTHING_LENGTH**2
    viscoelastic_link_sq = VISCOELASTIC_LINK_RADIUS**2
    virtual_sq = VIRTUAL_PRESSURE_RADIUS**2
    backtrack_direction = get_backtrack_direction(active_branch)
    active_shepherds = get_shepherds(robots)
    checked_pairs = set()
    for robot_i in robots:
        if robot_i.role in {"ANCHOR", "RELAY", "TRUNK_RELAY"}:
            robot_i.acceleration.update(0.0, 0.0)
            robot_i.last_sph_pressure_force = 0.0
            robot_i.last_compression_release_force = 0.0
            robot_i.last_shepherd_force = 0.0
            continue
        if robot_i.role == "SHEPHERD" and phase in {
            SimulationPhase.FORM_SHEPHERD_BOUNDARY,
            SimulationPhase.FILL_BEHIND_SHEPHERD,
            SimulationPhase.PRESSURE_PUSH,
            SimulationPhase.FLOW_BACKTRACK,
        }:
            robot_i.acceleration.update(0.0, 0.0)
            robot_i.filtered_acceleration.update(0.0, 0.0)
            robot_i.last_sph_pressure_force = 0.0
            robot_i.last_compression_release_force = 0.0
            robot_i.last_shepherd_force = 0.0
            continue

        pressure_force = pygame.Vector2()
        viscosity_force = pygame.Vector2()
        viscoelastic_force = pygame.Vector2()
        compression_release_force = pygame.Vector2()
        repulsion_force = pygame.Vector2()
        virtual_force = pygame.Vector2()
        cohesion_force = pygame.Vector2()
        neighbor_count = 0
        neighbor_center = pygame.Vector2()

        if (
            phase in {
                SimulationPhase.PRESSURE_PUSH,
                SimulationPhase.FLOW_BACKTRACK,
            }
            and robot_i.role == "NORMAL"
        ):
            for shepherd in active_shepherds:
                distance_sq = robot_i.position.distance_squared_to(
                    shepherd.position
                )
                if (
                    distance_sq > virtual_sq
                    or branch_progress(robot_i, active_branch)
                    > branch_progress(shepherd, active_branch) + 2.0
                ):
                    continue
                distance = math.sqrt(max(distance_sq, EPSILON))
                ratio = max(
                    0.0,
                    1.0 - distance / VIRTUAL_PRESSURE_RADIUS,
                )
                ramp = (
                    1.0
                    if phase == SimulationPhase.FLOW_BACKTRACK
                    else smoothstep01(
                        pressure_push_timer
                        / max(PRESSURE_RAMP_TIME, EPSILON)
                    )
                )
                virtual_force += (
                    backtrack_direction
                    * VIRTUAL_PRESSURE_FORCE
                    * ratio**2
                    * ramp
                )

        for robot_j in iter_physics_neighbor_candidates(robot_i, grid):
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

            if distance_sq <= EPSILON or distance_sq > h_sq:
                continue
            neighbor_count += 1
            neighbor_center += robot_j.position
            distance = math.sqrt(distance_sq)
            direction_away = r_ij / distance
            gradient = spiky_gradient(r_ij, SMOOTHING_LENGTH)
            coefficient = (
                robot_i.pressure / max(robot_i.density**2, EPSILON)
                + robot_j.pressure / max(robot_j.density**2, EPSILON)
            )
            pressure_force += -coefficient * gradient
            pair_equilibrium_radius = 0.5 * (
                adaptive_equilibrium_radius(robot_i)
                + adaptive_equilibrium_radius(robot_j)
            )
            v_ij = robot_i.velocity - robot_j.velocity
            kernel_weight = (
                spiky_kernel(distance, SMOOTHING_LENGTH)
                / max(
                    spiky_kernel(0.0, SMOOTHING_LENGTH),
                    EPSILON,
                )
            )
            # Physical viscosity acts for shear and separation as well as
            # compression; nearby robots continuously reach velocity
            # consensus instead of only damping head-on approaches.
            viscosity_force += (
                robot_j.velocity - robot_i.velocity
            ) * (
                VISCOELASTIC_VELOCITY_CONSENSUS_GAIN
                * kernel_weight
            )

            viscoelastic_pair = (
                robot_i.role == "NORMAL"
                and robot_j.role == "NORMAL"
                and not robot_i.base_reserve
                and not robot_j.base_reserve
                and distance_sq <= viscoelastic_link_sq
            )
            if viscoelastic_pair:
                if pair not in viscoelastic_rest_lengths:
                    viscoelastic_rest_lengths[pair] = clamp(
                        pair_equilibrium_radius,
                        VISCOELASTIC_REST_MIN,
                        VISCOELASTIC_REST_MAX,
                    )
                if (
                    viscoelastic_last_seen.get(pair)
                    != viscoelastic_step
                ):
                    rest_length = viscoelastic_rest_lengths[pair]
                    equilibrium_target = clamp(
                        pair_equilibrium_radius,
                        VISCOELASTIC_REST_MIN,
                        VISCOELASTIC_REST_MAX,
                    )
                    equilibrium_adaptation = clamp(
                        VISCOELASTIC_EQUILIBRIUM_ADAPTATION * dt,
                        0.0,
                        1.0,
                    )
                    rest_length += (
                        equilibrium_target - rest_length
                    ) * equilibrium_adaptation
                    relaxation = clamp(
                        VISCOELASTIC_REST_RELAXATION * dt,
                        0.0,
                        1.0,
                    )
                    viscoelastic_rest_lengths[pair] = (
                        rest_length
                        + (distance - rest_length) * relaxation
                    )
                    viscoelastic_last_seen[pair] = viscoelastic_step

                rest_length = viscoelastic_rest_lengths[pair]
                extension = distance - rest_length
                radial_relative_speed = v_ij.dot(direction_away)
                viscoelastic_force += direction_away * (
                    -VISCOELASTIC_ELASTIC_GAIN * extension
                    -VISCOELASTIC_DASHPOT_GAIN
                    * radial_relative_speed
                )

            if (
                phase in {
                    SimulationPhase.PRESSURE_PUSH,
                    SimulationPhase.FLOW_BACKTRACK,
                }
                and robot_i.role == "NORMAL"
                and robot_j.role == "NORMAL"
                and distance < COMPRESSION_RELEASE_RADIUS
            ):
                average_density_ratio = 0.5 * (
                    robot_i.density_ratio
                    + robot_j.density_ratio
                )
                density_excess = max(
                    0.0,
                    average_density_ratio
                    - COMPRESSION_RELEASE_DENSITY_RATIO,
                )
                spacing_compression = max(
                    0.0,
                    1.0
                    - distance
                    / max(pair_equilibrium_radius, EPSILON),
                )
                release_intensity = max(
                    density_excess,
                    spacing_compression * 2.0,
                )
                release_weight = (
                    1.0
                    - distance / COMPRESSION_RELEASE_RADIUS
                ) ** 2
                release_ramp = (
                    1.0
                    if phase == SimulationPhase.FLOW_BACKTRACK
                    else smoothstep01(
                        pressure_push_timer
                        / max(
                            COMPRESSION_RELEASE_RAMP_TIME,
                            EPSILON,
                        )
                    )
                )
                compression_release_force += direction_away * (
                    COMPRESSION_RELEASE_GAIN
                    * release_intensity
                    * release_weight
                    * release_ramp
                )

            approach = v_ij.dot(r_ij)
            if approach < 0.0:
                mu_ij = SMOOTHING_LENGTH * approach / (distance_sq + 0.01 * SMOOTHING_LENGTH**2)
                c_i_sq = (robot_i.pressure + PRESSURE_GAIN * robot_i.density) / max(robot_i.density, EPSILON)
                c_j_sq = (robot_j.pressure + PRESSURE_GAIN * robot_j.density) / max(robot_j.density, EPSILON)
                c_ij = 0.5 * (math.sqrt(max(c_i_sq, 0.0)) + math.sqrt(max(c_j_sq, 0.0)))
                mean_density = 0.5 * (robot_i.density + robot_j.density)
                pi_ij = (-VISCOSITY_XI1 * c_ij * mu_ij + VISCOSITY_XI2 * mu_ij**2) / max(mean_density, EPSILON)
                viscosity_force += -pi_ij * gradient
            if distance < pair_equilibrium_radius:
                penetration_ratio = (
                    pair_equilibrium_radius - distance
                ) / max(pair_equilibrium_radius, EPSILON)
                repulsion_force += (
                    REPULSION_GAIN
                    * penetration_ratio
                    * (r_ij / distance)
                )

        limit_vector(pressure_force, SPH_PRESSURE_FORCE_LIMIT)
        limit_vector(viscosity_force, SPH_VISCOSITY_FORCE_LIMIT)
        limit_vector(viscoelastic_force, VISCOELASTIC_FORCE_LIMIT)
        limit_vector(
            compression_release_force,
            COMPRESSION_RELEASE_FORCE_LIMIT,
        )
        limit_vector(
            virtual_force,
            SHEPHERD_VIRTUAL_FORCE_LIMIT,
        )
        route_force = compute_route_force(robot_i)
        connectivity_force = compute_connectivity_force(
            robot_i,
            communication_grid,
        )
        shepherd_curtain_force = compute_shepherd_curtain_force(robot_i)
        initial_junction_wall_force = (
            compute_initial_junction_soft_wall_force(robot_i)
        )
        robot_i.last_sph_pressure_force = pressure_force.length()
        robot_i.last_compression_release_force = (
            compression_release_force.length()
        )
        robot_i.last_shepherd_force = (
            virtual_force + shepherd_curtain_force
        ).length()
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
            + viscoelastic_force
            + compression_release_force
            + repulsion_force
            + virtual_force
            + cohesion_force
            + route_force
            + connectivity_force
            + shepherd_curtain_force
            + initial_junction_wall_force
            - DAMPING * robot_i.velocity
        )
        acceleration_limit = (
            INITIAL_SAFE_MAX_ACCELERATION
            if phase == SimulationPhase.MOVE_TO_JUNCTION
            else MAX_ACCELERATION
        )
        raw_acceleration = limit_vector(total, acceleration_limit)
        robot_i.filtered_acceleration = (
            robot_i.filtered_acceleration
            * (1.0 - ACCELERATION_FILTER_ALPHA)
            + raw_acceleration * ACCELERATION_FILTER_ALPHA
        )
        robot_i.acceleration = robot_i.filtered_acceleration.copy()

    if viscoelastic_step % 5 == 0:
        stale_pairs = [
            pair
            for pair, last_seen in viscoelastic_last_seen.items()
            if (
                viscoelastic_step - last_seen
                > VISCOELASTIC_LINK_STALE_STEPS
            )
        ]
        for pair in stale_pairs:
            viscoelastic_last_seen.pop(pair, None)
            viscoelastic_rest_lengths.pop(pair, None)

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
    global shepherd_flow_timer
    global junction_switch_timer, final_gather_timer, branch_entry_timer
    global distributed_consensus_branch, transfer_branch
    global final_base_transfer_active
    global return_trunk_release_pending, return_trunk_retract_timer, return_trunk_last_released_id, return_trunk_force_timer

    if phase in {
        SimulationPhase.EXPLORE_BRANCH,
        SimulationPhase.FORM_SHEPHERD_BOUNDARY,
        SimulationPhase.FILL_BEHIND_SHEPHERD,
    }:
        branch_entry_timer += dt

    # No control Anchor is elected. NORMAL robots make branch decisions by
    # exchanging local votes; the Base only remains the communication root.
    anchor = None

    if phase == SimulationPhase.MOVE_TO_JUNCTION:
        update_relay_deployment(robots, dt)
        robots_in_junction = sum(
            get_robot_region(robot.position) == "JUNCTION"
            and robot.role == "NORMAL"
            for robot in robots
        )
        voted_branch = update_distributed_branch_consensus(robots)
        if (
            robots_in_junction >= DISTRIBUTED_VOTE_MIN_ROBOTS
            and voted_branch is not None
        ):
            selected = choose_next_branch(None, robots, reference_density)
            if selected is None:
                begin_final_gather()
            else:
                saturation_tracker.reset(selected)
                junction_consensus_tracker.reset()
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
                # Close a continuous full-width virtual gate immediately.  Any
                # ordinary robot already beyond the planned line is moved to
                # its safe Junction side before the next physics frame.
                enforce_shepherd_curtain_for_swarm(robots)
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
            if force_complete_shepherd_boundary(robots):
                phase = SimulationPhase.FILL_BEHIND_SHEPHERD
                saturation_tracker.reset(active_branch)
                print(
                    "[Shepherd] local slot fallback completed; "
                    "filling starts without Base communication"
                )
            else:
                reset_shepherd_roles(robots)
                phase = SimulationPhase.EXPLORE_BRANCH
                print(
                    "[Shepherd] invalid local slots; "
                    "dead-end election will retry"
                )

    elif phase == SimulationPhase.FILL_BEHIND_SHEPHERD:
        update_relay_deployment(robots, dt)
        saturated = update_dead_end_saturation(
            robots, active_branch, reference_density, dt
        )
        if saturated:
            next_branch = next_unvisited_transfer_branch(active_branch)
            if next_branch is not None:
                begin_cross_branch_transfer(
                    robots,
                    active_branch,
                    next_branch,
                )
            else:
                begin_final_base_transfer(
                    robots,
                    active_branch,
                )
            phase = SimulationPhase.PRESSURE_PUSH
            pressure_push_timer = 0.0
            flow_establish_timer = 0.0
            shepherd_flow_timer = 0.0
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
        shepherd_flow_timer += dt
        release_shepherd_line_at_junction(robots)
        update_relay_retraction(robots, dt)
        remaining = sum(get_robot_region(robot.position) == active_branch for robot in robots)
        in_junction = sum(
            get_robot_region(robot.position) == "JUNCTION" and robot.role != "ANCHOR"
            for robot in robots
        )
        transferred_count = (
            sum(
                robot.transfer_target == transfer_branch
                and get_robot_region(robot.position) == transfer_branch
                for robot in robots
            )
            if transfer_branch is not None
            else 0
        )
        transfer_ready = (
            transfer_branch is not None
            and transferred_count >= JUNCTION_SWITCH_COUNT
        )
        base_transferred_count = sum(
            robot.transfer_target == "BOTTOM"
            and get_robot_region(robot.position) == "BOTTOM"
            for robot in robots
        )
        base_transfer_ready = (
            final_base_transfer_active
            and base_transferred_count >= JUNCTION_SWITCH_COUNT
        )
        junction_ready = (
            transfer_branch is None
            and not final_base_transfer_active
            and in_junction >= JUNCTION_SWITCH_COUNT
        )
        if (
            remaining <= BRANCH_CLEAR_LIMIT
            and not get_active_branch_relays(robots)
            and (
                transfer_ready
                or base_transfer_ready
                or junction_ready
            )
        ):
            completed_branch = active_branch
            next_branch = transfer_branch
            complete_active_branch(anchor, completed_branch, robots)
            if final_base_transfer_active:
                reset_shepherd_roles(robots)
                final_base_transfer_active = False
                begin_final_return(None, robots)
                print(
                    f"[Final Base Transfer] completed "
                    f"{completed_branch} -> BASE; "
                    f"robots={base_transferred_count}"
                )
            elif next_branch is not None:
                reset_shepherd_roles(robots)
                distributed_consensus_branch = next_branch
                selected = choose_next_branch(
                    None,
                    robots,
                    reference_density,
                )
                saturation_tracker.reset(selected)
                phase = SimulationPhase.EXPLORE_BRANCH
                metrics.branch_events.append({
                    "branch": selected,
                    "started_at": simulation_time,
                    "transferred_from": completed_branch,
                })
                print(
                    f"[Cross-Branch Transfer] completed "
                    f"{completed_branch} -> {selected}; "
                    f"robots={transferred_count}"
                )
            else:
                reset_shepherd_roles(robots)
                phase = SimulationPhase.JUNCTION_SWITCH
                junction_switch_timer = 0.0
                junction_consensus_tracker.reset()

    elif phase == SimulationPhase.JUNCTION_SWITCH:
        junction_switch_timer += dt
        if not any(state == "UNVISITED" for state in branch_states.values()):
            begin_final_gather()
            return
        voted_branch = update_distributed_branch_consensus(robots)
        if voted_branch is not None:
            selected = choose_next_branch(None, robots, reference_density)
            if selected is None:
                begin_final_gather()
            else:
                saturation_tracker.reset(selected)
                junction_consensus_tracker.reset()
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
        global return_trunk_last_released_id, return_trunk_force_timer
        in_bottom = sum(get_robot_region(robot.position) == "BOTTOM" for robot in robots)
        trunk_relays = get_trunk_relays(robots)
        connected_count = sum(robot.connected_to_base for robot in robots)
        connected_ratio = connected_count / max(len(robots), 1)
        special = sum(
            robot.role in {"ANCHOR", "RELAY", "TRUNK_RELAY", "SHEPHERD"}
            for robot in robots
        )

        if return_trunk_release_pending:
            # Sequentially retract the Junction-side Trunk Relay.  Requiring a
            # particular released robot to be classified as BOTTOM caused the
            # chain to stall permanently when that robot hovered on a region
            # boundary.  Instead, use the live Base-connected ratio as the
            # safety guard and release one relay per dwell interval.
            safe_to_retract = connected_ratio >= RETURN_TRUNK_READY_CONNECTED_RATIO
            return_trunk_retract_timer = (
                return_trunk_retract_timer + dt if safe_to_retract else 0.0
            )
            return_trunk_force_timer += dt

            release_due = (
                return_trunk_retract_timer >= RETURN_TRUNK_RETRACT_DWELL
                or return_trunk_force_timer >= RETURN_TRUNK_FORCE_RELEASE_TIMEOUT
            )

            if trunk_relays and release_due:
                released = release_next_trunk_relay_for_return(robots)
                if released is not None:
                    return_trunk_last_released_id = released.robot_id
                return_trunk_retract_timer = 0.0
                return_trunk_force_timer = 0.0
                if not get_trunk_relays(robots):
                    return_trunk_release_pending = False
                return

            if not trunk_relays:
                return_trunk_release_pending = False
                return_trunk_retract_timer = 0.0
                return_trunk_force_timer = 0.0
                return_trunk_last_released_id = None

        if in_bottom >= RETURN_BOTTOM_TARGET_COUNT and special == 0:
            phase = SimulationPhase.DONE
            metrics.completion_time = simulation_time
            print(f"[DFS] done, robots={in_bottom}/{len(robots)}")
            save_experiment_logs(robots, "DONE")

def draw_branch_colour_fields(surface):
    """Tint each physical branch using the same hue as its proxy subregion."""
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    branch_rectangles = {
        "UP": up_rect,
        "LEFT": left_rect,
        "RIGHT": right_rect,
    }
    for branch, rect in branch_rectangles.items():
        pygame.draw.rect(
            overlay,
            (*BRANCH_COLORS[branch], BRANCH_FLOOR_ALPHA),
            rect,
        )
        pygame.draw.rect(
            overlay,
            (*BRANCH_COLORS[branch], 230),
            rect,
            width=3,
        )
    surface.blit(overlay, (0, 0))

    label_positions = {
        "UP": (center_x, up_rect.top + 18),
        "LEFT": (left_rect.left + 32, center_y),
        "RIGHT": (right_rect.right - 38, center_y),
    }
    for branch, position in label_positions.items():
        label = small_font.render(branch, True, (255, 255, 255))
        badge = label.get_rect(center=position).inflate(12, 6)
        pygame.draw.rect(
            surface,
            BRANCH_COLORS[branch],
            badge,
            border_radius=6,
        )
        surface.blit(label, label.get_rect(center=position))


def draw_branch_gates(surface):
    """Draw the Anchor-commanded artificial walls at closed branch mouths."""
    gate_lines = {
        "UP": (
            (center_x - half_width + 3, center_y - half_width),
            (center_x + half_width - 3, center_y - half_width),
        ),
        "LEFT": (
            (center_x - half_width, center_y - half_width + 3),
            (center_x - half_width, center_y + half_width - 3),
        ),
        "RIGHT": (
            (center_x + half_width, center_y - half_width + 3),
            (center_x + half_width, center_y + half_width - 3),
        ),
    }
    for branch, state in branch_gate_states.items():
        if state != "CLOSED":
            continue
        start, end = gate_lines[branch]
        pygame.draw.line(
            surface,
            ARTIFICIAL_WALL_COLOR,
            start,
            end,
            ARTIFICIAL_WALL_WIDTH,
        )
        midpoint = (
            round((start[0] + end[0]) / 2),
            round((start[1] + end[1]) / 2),
        )
        label = hud_font.render("CLOSED", True, (255, 255, 255))
        badge = label.get_rect(center=midpoint).inflate(8, 4)
        pygame.draw.rect(
            surface,
            ARTIFICIAL_WALL_COLOR,
            badge,
            border_radius=4,
        )
        surface.blit(label, label.get_rect(center=midpoint))


def draw_proxy_partition(surface):
    """Draw vivid, clearly separated decision-time proxy subregions."""
    if not last_proxy_partition or not last_proxy_cell_centers:
        return

    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    centroids = {branch: pygame.Vector2() for branch in BRANCHES}
    counts = {branch: 0 for branch in BRANCHES}

    # Strong translucent fill using the exact same Branch RGB identity.
    for key, branch in last_proxy_partition.items():
        center = last_proxy_cell_centers[key]
        rect = pygame.Rect(
            int(center.x - PROXY_CELL_SIZE / 2),
            int(center.y - PROXY_CELL_SIZE / 2),
            PROXY_CELL_SIZE,
            PROXY_CELL_SIZE,
        )
        pygame.draw.rect(
            overlay,
            (*BRANCH_COLORS[branch], BRANCH_PROXY_ALPHA),
            rect,
        )
        centroids[branch] += center
        counts[branch] += 1

    # Draw only inter-region boundaries, not every cell grid line.
    neighbour_offsets = {
        "left": (-1, 0),
        "right": (1, 0),
        "top": (0, -1),
        "bottom": (0, 1),
    }
    for (col, row), branch in last_proxy_partition.items():
        center = last_proxy_cell_centers[(col, row)]
        left = int(center.x - PROXY_CELL_SIZE / 2)
        top = int(center.y - PROXY_CELL_SIZE / 2)
        right = left + PROXY_CELL_SIZE
        bottom = top + PROXY_CELL_SIZE
        for side, (dc, dr) in neighbour_offsets.items():
            neighbour_branch = last_proxy_partition.get((col + dc, row + dr))
            if neighbour_branch == branch:
                continue
            if side == "left":
                p1, p2 = (left, top), (left, bottom)
            elif side == "right":
                p1, p2 = (right, top), (right, bottom)
            elif side == "top":
                p1, p2 = (left, top), (right, top)
            else:
                p1, p2 = (left, bottom), (right, bottom)
            pygame.draw.line(
                overlay,
                (*BRANCH_COLORS[branch], 255),
                p1,
                p2,
                BRANCH_BOUNDARY_WIDTH,
            )

    surface.blit(overlay, (0, 0))

    # Branch name badge at the centroid of each temporary proxy subregion.
    for branch in last_proxy_candidates:
        if counts.get(branch, 0) <= 0:
            continue
        center = centroids[branch] / counts[branch]
        label = small_font.render(branch, True, (255, 255, 255))
        badge = label.get_rect(center=(round(center.x), round(center.y))).inflate(10, 5)
        pygame.draw.rect(
            surface,
            BRANCH_COLORS[branch],
            badge,
            border_radius=5,
        )
        surface.blit(label, label.get_rect(center=badge.center))


def draw_proxy_robot_assignments(surface, robots):
    """Draw projected analytical assignments as darker dots for readability."""
    if not last_proxy_robot_assignment:
        return
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    for robot in robots:
        branch = last_proxy_robot_assignment.get(robot.robot_id)
        if branch is None:
            continue
        point = project_robot_to_proxy(robot.position)
        pygame.draw.circle(
            overlay,
            (*PROXY_POINT_COLORS[branch], 210),
            (round(point.x), round(point.y)),
            2,
        )
    surface.blit(overlay, (0, 0))


def draw_shepherd_curtain(surface):
    """Visualize the continuous gate that seals gaps between Shepherd robots."""
    if not shepherd_curtain_active():
        return
    depth = get_shepherd_curtain_depth(active_branch)
    color = BRANCH_COLORS[active_branch]
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    if active_branch == "UP":
        y = round(center_y - half_width - depth)
        pygame.draw.line(
            overlay,
            (*color, 235),
            (center_x - half_width + 2, y),
            (center_x + half_width - 2, y),
            SHEPHERD_CURTAIN_DRAW_HALF_WIDTH * 2,
        )
    elif active_branch == "LEFT":
        x = round(center_x - half_width - depth)
        pygame.draw.line(
            overlay,
            (*color, 235),
            (x, center_y - half_width + 2),
            (x, center_y + half_width - 2),
            SHEPHERD_CURTAIN_DRAW_HALF_WIDTH * 2,
        )
    else:
        x = round(center_x + half_width + depth)
        pygame.draw.line(
            overlay,
            (*color, 235),
            (x, center_y - half_width + 2),
            (x, center_y + half_width - 2),
            SHEPHERD_CURTAIN_DRAW_HALF_WIDTH * 2,
        )
    surface.blit(overlay, (0, 0))


# =========================================================
# 17. Initialization
# =========================================================


def reset_dfs_state():
    global phase, active_branch, branch_states, branch_order_plan
    global branch_gate_states, distributed_consensus_branch, transfer_branch
    global final_base_transfer_active
    global previous_branch_direction, junction_anchor, simulation_time
    global junction_switch_timer, final_gather_timer, shepherd_form_timer
    global pressure_push_timer, flow_establish_timer, communication_sequence
    global shepherd_flow_timer, shepherd_flow_start_depth
    global viscoelastic_step, viscoelastic_rest_lengths
    global viscoelastic_last_seen
    global last_message_signature, relay_slots, relay_deploy_cooldown
    global relay_retract_cooldown, relay_retract_clear_timer, relay_motion_scale
    global trunk_relay_slots, trunk_relay_deploy_cooldown, base_station
    global last_proxy_partition, last_proxy_cell_centers
    global last_proxy_mass_stats, last_proxy_robot_assignment
    global last_proxy_candidates
    global last_flow_rollout_scores
    global selected_branch_entry_lambda, branch_entry_timer
    global return_trunk_release_pending, return_trunk_retract_timer
    global metrics
    phase = SimulationPhase.MOVE_TO_JUNCTION
    active_branch = FIXED_BRANCH_ORDER[0]
    branch_states = {branch: "UNVISITED" for branch in BRANCHES}
    branch_order_plan = []
    branch_gate_states = {branch: "OPEN" for branch in BRANCHES}
    distributed_consensus_branch = None
    transfer_branch = None
    final_base_transfer_active = False
    previous_branch_direction = pygame.Vector2(0.0, -1.0)
    junction_anchor = None
    simulation_time = 0.0
    junction_switch_timer = final_gather_timer = shepherd_form_timer = 0.0
    pressure_push_timer = flow_establish_timer = 0.0
    shepherd_flow_timer = shepherd_flow_start_depth = 0.0
    viscoelastic_step = 0
    viscoelastic_rest_lengths = {}
    viscoelastic_last_seen = {}
    communication_sequence = 0
    last_message_signature = None
    relay_slots = []
    relay_deploy_cooldown = relay_retract_cooldown = relay_retract_clear_timer = 0.0
    relay_motion_scale = 1.0
    trunk_relay_slots = []
    trunk_relay_deploy_cooldown = 0.0
    base_station = BaseStation(BASE_POSITION)
    last_proxy_partition = {}
    last_proxy_cell_centers = {}
    last_proxy_mass_stats = {}
    last_proxy_robot_assignment = {}
    last_proxy_candidates = ()
    last_flow_rollout_scores = {}
    selected_branch_entry_lambda = STIFFNESS_EXPONENT_RIGID
    branch_entry_timer = 0.0
    return_trunk_release_pending = False
    return_trunk_retract_timer = 0.0
    return_trunk_last_released_id = None
    return_trunk_force_timer = 0.0
    saturation_tracker.reset()
    junction_consensus_tracker.reset()
    metrics = ExperimentMetrics()


def initialize_simulation():
    reset_dfs_state()
    robots = create_grid_robots(ROBOT_COUNT) if SPAWN_MODE == "grid" else create_random_robots(ROBOT_COUNT)
    if not robots:
        raise RuntimeError("No robots were created.")
    grid = build_spatial_grid(robots)
    compute_densities(robots, build_physics_grid(robots))
    mean_density = sum(robot.density for robot in robots) / len(robots)
    reference_density = mean_density * 0.62
    color_reference_density = mean_density * 0.68
    update_communication_system(robots, grid)
    print(
        f"robots={len(robots)}, breadcrumb_guards=0, "
        f"mean_density={mean_density:.6f}, rho0={reference_density:.6f}"
    )
    return robots, reference_density, color_reference_density


robots, reference_density, color_reference_density = initialize_simulation()

# =========================================================
# 18. Main loop
# =========================================================

running = True
paused = False
def wrap_hud_text(text: str, font_obj, max_width: int):
    """Wrap one HUD string to the width of the separate side panel."""
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font_obj.size(candidate)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_hud_panel(surface, lines):
    panel_rect = pygame.Rect(
        HUD_PANEL_X,
        0,
        HUD_PANEL_WIDTH,
        SCREEN_HEIGHT,
    )
    pygame.draw.rect(surface, HUD_PANEL_COLOR, panel_rect)
    pygame.draw.line(
        surface,
        HUD_PANEL_BORDER_COLOR,
        (HUD_PANEL_X, 0),
        (HUD_PANEL_X, SCREEN_HEIGHT),
        width=2,
    )

    x = HUD_PANEL_X + HUD_PANEL_MARGIN
    y = 12
    max_width = HUD_PANEL_WIDTH - 2 * HUD_PANEL_MARGIN
    line_height = hud_font.get_linesize() + 2

    for line in lines:
        wrapped = wrap_hud_text(line, hud_font, max_width)
        for wrapped_line in wrapped:
            if y + line_height >= SCREEN_HEIGHT - 72:
                break
            surface.blit(hud_font.render(wrapped_line, True, TEXT_COLOR), (x, y))
            y += line_height
        if y + line_height >= SCREEN_HEIGHT - 72:
            break

    controls = [
        "SPACE pause | R reset | D density",
        "V regions | C communication | ESC quit",
    ]
    controls_y = SCREEN_HEIGHT - 58
    for control in controls:
        surface.blit(hud_font.render(control, True, TEXT_COLOR), (x, controls_y))
        controls_y += line_height


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
        substep_dt = frame_dt / SUBSTEPS
        for _ in range(SUBSTEPS):
            physics_grid = build_physics_grid(robots)
            compute_densities(robots, physics_grid)
            compute_pressures(robots, reference_density)
            compute_sph_forces(
                robots,
                physics_grid,
                spatial_grid,
                substep_dt,
            )
            for robot in robots:
                robot.update(substep_dt)
        # Rebuild immediately after distributed role changes such as a
        # Breadcrumb or Shepherd self-election.
        spatial_grid = build_spatial_grid(robots)
        update_communication_system(robots, spatial_grid)
        update_simulation_state(robots, frame_dt, reference_density, spatial_grid)
        update_metrics_per_frame(robots, frame_dt)
    else:
        update_communication_system(robots, spatial_grid)
        compute_densities(robots, build_physics_grid(robots))
        compute_pressures(robots, reference_density)

    screen.fill(BACKGROUND_COLOR)
    pygame.draw.polygon(screen, FLOOR_COLOR, cross_points)
    draw_branch_colour_fields(screen)
    pygame.draw.polygon(screen, WALL_COLOR, cross_points, width=2)
    draw_branch_gates(screen)

    if show_regions:
        draw_proxy_partition(screen)
        draw_proxy_robot_assignments(screen, robots)
        pygame.draw.rect(screen, JUNCTION_COLOR, junction_rect, width=2)
        pygame.draw.circle(screen, BASE_COLOR, BASE_POSITION, 7, width=2)
        pygame.draw.rect(
            screen,
            BRANCH_COLORS[active_branch],
            early_capture_regions[active_branch],
            width=2,
        )
        pygame.draw.rect(
            screen,
            BRANCH_COLORS[active_branch],
            get_saturation_rect(active_branch),
            width=2,
        )
        for branch, rect in dead_end_regions.items():
            pygame.draw.rect(
                screen,
                BRANCH_COLORS[branch],
                rect,
                width=3 if branch == active_branch else 2,
            )
        for robot in get_shepherds(robots):
            if robot.shepherd_anchor is not None:
                pygame.draw.circle(
                    screen,
                    BRANCH_COLORS[active_branch],
                    robot.shepherd_anchor,
                    4,
                    width=2,
                )
        draw_shepherd_curtain(screen)
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
    shepherd_packing_count = sum(
        robot_is_in_shepherd_packing_zone(robot, active_branch)
        for robot in robots
    )
    shepherd_line_error = max(
        (
            robot.position.distance_to(
                shepherd_slot_position_at_depth(
                    robot.shepherd_anchor,
                    active_branch,
                    get_shepherd_curtain_depth(active_branch),
                )
            )
            for robot in get_shepherds(robots)
            if robot.shepherd_anchor is not None
        ),
        default=0.0,
    )
    force_samples = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and get_robot_region(robot.position) == active_branch
    ]
    force_sample_count = max(1, len(force_samples))
    average_sph_pressure_force = sum(
        robot.last_sph_pressure_force
        for robot in force_samples
    ) / force_sample_count
    average_compression_release_force = sum(
        robot.last_compression_release_force
        for robot in force_samples
    ) / force_sample_count
    average_shepherd_force = sum(
        robot.last_shepherd_force
        for robot in force_samples
    ) / force_sample_count
    hud_lines = [
        "Pressure-driven SPH | Distributed NORMAL consensus | Breadcrumb relay",
        f"FPS={clock.get_fps():.1f} | robots={len(robots)} | phase={phase.name}",
        f"Sketch sequence stage={get_sequence_stage()}/6",
        (
            "Base impulse=COMPRESS"
            if phase == SimulationPhase.MOVE_TO_JUNCTION
            and simulation_time < BASE_COMPRESSION_DURATION
            else "Base impulse=SPH EXPANSION"
            if phase == SimulationPhase.MOVE_TO_JUNCTION
            and simulation_time
            < BASE_COMPRESSION_DURATION + BASE_EXPANSION_BOOST_DURATION
            else "Base impulse=RELEASED"
        ),
        (
            f"Initial safety={INITIAL_IMPULSE_POLICY_VERSION} | "
            f"pressure scale={get_base_pressure_scale():.2f}"
        ),
        (
            "Decision=NORMAL peer consensus | "
            f"Junction voters={sum(robot.role == 'NORMAL' and get_robot_region(robot.position) == 'JUNCTION' for robot in robots)}"
        ),
        (
            f"Breadcrumb-front NORMAL guards={sum(robot.base_reserve and get_robot_region(robot.position) == 'BOTTOM' for robot in robots)}"
            f"/{sum(robot.base_reserve for robot in robots)}"
        ),
        f"Branch={active_branch if phase not in {SimulationPhase.MOVE_TO_JUNCTION, SimulationPhase.RETURN_TO_BASE, SimulationPhase.DONE} else '-'}",
        (
            f"Distributed decision=MOVE_{distributed_consensus_branch}"
            if distributed_consensus_branch
            else "Distributed decision=VOTING"
        ),
        (
            f"Pressure transfer={active_branch}->BASE | "
            f"eligible={sum(robot.transfer_target == 'BOTTOM' for robot in robots)}"
            if final_base_transfer_active
            else f"Pressure transfer={active_branch}->{transfer_branch} | "
            f"eligible={sum(robot.transfer_target == transfer_branch for robot in robots)}"
            if transfer_branch
            else "Pressure transfer=-"
        ),
        "Gates: " + " | ".join(
            f"{branch}={branch_gate_states[branch]}"
            for branch in BRANCHES
        ),
        f"Order={' > '.join(branch_order_plan) if branch_order_plan else '-'}",
        (
            "Last branch cost: "
            + (
                f"Q={metrics.branch_selection_events[-1]['components']['predicted_flow']:.2f} "
                f"dRho={metrics.branch_selection_events[-1]['components']['density_disturbance']:.2f} "
                f"dV={metrics.branch_selection_events[-1]['components']['velocity_disturbance']:.2f} "
                f"Comm={metrics.branch_selection_events[-1]['components']['rollout_comm']:.2f}"
                if metrics.branch_selection_events
                else "-"
            )
        ),
        (
            "Proxy mass: "
            + " | ".join(
                f"{branch} q={last_proxy_mass_stats.get(branch, {}).get('quota_fraction', 0.0):.2f} "
                f"m={last_proxy_mass_stats.get(branch, {}).get('actual_mass_fraction', 0.0):.2f}"
                for branch in last_proxy_candidates
            )
            if last_proxy_candidates
            else "Proxy mass: -"
        ),
        (
            "Proxy rollout candidates: "
            + " | ".join(
                f"{branch}:J={data['cost']:.2f},Q={data['components']['predicted_flow']:.2f},"
                f"n={data['components']['proxy_primary_count']}+{data['components']['proxy_context_count']}"
                for branch, data in sorted(last_flow_rollout_scores.items())
            )
            if last_flow_rollout_scores
            else "Proxy rollout candidates: -"
        ),
        (
            f"Adaptive lambda={get_effective_stiffness_exponent():.3f} "
            f"entry={selected_branch_entry_lambda:.3f} "
            f"recovery={branch_entry_timer:.2f}/{BRANCH_STIFFNESS_RECOVERY_TIME:.2f}"
        ),
        (
            f"Junction consensus: n={junction_consensus_tracker.candidate_count} "
            f"stable={junction_consensus_tracker.stable_ratio:.2f} "
            f"dv={junction_consensus_tracker.mean_density_delta_ratio:.3f} "
            f"dwell={junction_consensus_tracker.dwell:.2f} "
            f"fast={junction_consensus_tracker.fast_dwell:.2f} "
            f"mode={junction_consensus_tracker.ready_mode}"
        ),
        f"States: U={branch_states['UP']} L={branch_states['LEFT']} R={branch_states['RIGHT']}",
        f"Base comm={communication_stats['connected']}/{len(robots)} | hop={communication_stats['max_hop']} | margin={communication_stats['margin']:.1f}",
        f"Base direct={communication_stats['direct']} | Breadcrumbs={len(get_active_branch_relays(robots))}",
        f"Front comm ratio={front_comm['connected_ratio']:.2f} | relay need={front_comm['needs_relay']}",
        f"Reactive relays={len(get_relays(robots))} | preplanned slots=0",
        f"Branch robots normal={normal_count} relay={relay_count} shepherd={shepherd_count}",
        f"Saturation: tip={saturation_tracker.tip_count} slow={saturation_tracker.low_speed_ratio:.2f}",
        f"density={saturation_tracker.average_density_ratio:.2f} occupancy={saturation_tracker.occupancy_ratio:.2f}",
        (
            f"Saturation policy={SATURATION_POLICY_VERSION} | "
            f"geometric count={saturation_tracker.tip_count}/"
            f"{saturation_tracker.packed_min_count}"
        ),
        (
            f"front_delta={saturation_tracker.front_delta:.2f} "
            f"dwell={saturation_tracker.dwell:.2f} "
            f"mode={saturation_tracker.recognition_mode} "
            f"saturated={saturation_tracker.saturated}"
        ),
        (
            f"Spacing: Normal={NORMAL_EQUILIBRIUM_SCALE:.2f}x | "
            f"Shepherd-back={SHEPHERD_PACKED_EQUILIBRIUM_SCALE:.2f}x "
            f"(n={shepherd_packing_count}) | Junction-tail={JUNCTION_TAIL_EQUILIBRIUM_SCALE:.2f}x"
        ),
        f"Shepherd target={adaptive_shepherd_count()} | formed={shepherd_boundary_formed(robots)} | pressure t={pressure_push_timer:.2f}",
        f"Shepherd control=LOCAL_ONLY | active={len(get_shepherds(robots))}",
        (
            f"Motion={MOTION_POLICY_VERSION} | "
            f"Shepherd={SHEPHERD_POLICY_VERSION}"
        ),
        (
            f"Viscoelastic={VISCOELASTIC_MODEL_VERSION} | "
            f"active links={len(viscoelastic_rest_lengths)}"
        ),
        (
            f"Force avg: SPH={average_sph_pressure_force:.1f} | "
            f"self-expand={average_compression_release_force:.1f} | "
            f"Shepherd={average_shepherd_force:.1f}"
        ),
        f"Pressure policy={PRESSURE_POLICY_VERSION}",
        (
            f"Shepherd line depth={get_shepherd_curtain_depth(active_branch):.1f} "
            f"| max slot error={shepherd_line_error:.2f}"
        ),
        f"Distance total={sum(robot.total_distance for robot in robots):.0f} | disconnect robot-s={metrics.disconnected_robot_seconds:.1f}",
    ]
    draw_hud_panel(screen, hud_lines)
    pygame.display.flip()

if not metrics.saved:
    save_experiment_logs(robots, "USER_EXIT")
pygame.quit()
sys.exit()
