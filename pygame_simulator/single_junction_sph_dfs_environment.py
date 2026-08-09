"""SPH-based single-junction Physical DFS prototype.

Implemented research components
-------------------------------
1. Emmons-inspired, normalized local angular distributions are accumulated
   sequentially to infer a Junction from lateral expansion and persistent,
   traversable directional cohorts.
2. NORMAL-to-NORMAL local voting commits only branches discovered by the
   emergent-distribution observer.  The elected Anchor stores and relays the
   consensus; it does not decide on behalf of the swarm.
3. Branch order is selected from locally observable transport, deformation,
   turn, contact, communication, guard, and short SPH-rollout costs.
4. Eguchi-inspired command-observed velocity discrepancy is integrated per
   robot.  Detected collision positions become virtual boundary samples and
   weak/strong state-dependent obstacle repulsion sources.
5. Dead ends are confirmed from frontier-wide contact evidence, low observed
   forward velocity, density, lateral escape, and dwell time--not from a known
   terminal coordinate alone.
6. Width-adaptive Shepherd election, pressure backflow, Base-rooted LOS
   communication, and reactive Breadcrumb relays complete the physical DFS.

Scope limitation
----------------
The renderer and collision mask still provide the single-cross test fixture,
but Junction/dead-end decisions do not consume its labelled detection regions.
The code implements DFS child ordering at one junction, not recursive
multi-junction DFS-tree repair.
"""

from __future__ import annotations

import csv
import heapq
import math
import os
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
ROBOT_BASE_COLOR = (29, 74, 135)
SHEPHERD_COLOR = (106, 70, 150)
JUNCTION_GUARD_COLOR = (202, 56, 72)
FRONTIER_SHEPHERD_COLOR = (238, 132, 36)
BRANCH_LEADER_RING_COLOR = (255, 220, 55)
# High-luminance green makes the stationary Anchor immediately distinct from
# the navy NORMAL swarm, purple Shepherds, and brown Breadcrumb relays.
ANCHOR_COLOR = (20, 220, 90)
ANCHOR_RING_COLOR = (0, 72, 34)
COMM_BRIDGE_COLOR = (0, 190, 215)
COMM_BRIDGE_LINK_COLOR = (0, 154, 190)
BASE_COLOR = (44, 72, 120)
TRUNK_RELAY_COLOR = (142, 82, 60)
RELAY_COLOR = (168, 112, 44)
ROBOT_OUTLINE_COLOR = (12, 28, 52)
DISCONNECTED_FILL_COLOR = (180, 92, 92)
PROXY_POINT_COLORS = {
    "UP": (46, 76, 130),
    "LEFT": (92, 54, 120),
    "RIGHT": (132, 86, 36),
}
RELAY_SLOT_COLOR = (196, 129, 65)
JUNCTION_COLOR = (153, 164, 181)
END_REGION_COLOR = (224, 171, 115)
# Muted sage-mint stays lighter than the navy swarm without visual harshness.
COMM_LINK_SAFE_COLOR = (96, 158, 138)
COMM_LINK_WARNING_COLOR = (226, 177, 96)
COMM_LINK_DANGER_COLOR = (214, 103, 103)
COMM_LINK_WIDTH = 1
DISCONNECTED_COLOR = (205, 96, 96)
ARTIFICIAL_WALL_COLOR = (220, 62, 62)
ARTIFICIAL_WALL_WIDTH = 3
CONTACT_POINT_COLOR = (220, 45, 150)
INFERRED_JUNCTION_COLOR = (30, 170, 210)

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
MAP_SCALE = 0.70
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
ANCHOR_PARK_SLOTS = (
    pygame.Vector2(center_x - 25 * MAP_SCALE, center_y - 25 * MAP_SCALE),
    pygame.Vector2(center_x + 25 * MAP_SCALE, center_y - 25 * MAP_SCALE),
    pygame.Vector2(center_x - 25 * MAP_SCALE, center_y + 25 * MAP_SCALE),
    pygame.Vector2(center_x + 25 * MAP_SCALE, center_y + 25 * MAP_SCALE),
)
# Fixed communication root. It remains at the lower entrance throughout exploration.
BASE_POSITION = pygame.Vector2(
    center_x - 25 * MAP_SCALE,
    center_y + half_width + base_length - 14 * MAP_SCALE,
)
BASE_COMPRESSION_CENTER = pygame.Vector2(
    center_x,
    center_y + half_width + base_length * 0.60,
)
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
UP_EARLY_CAPTURE_DEPTH = round(64 * MAP_SCALE)
SHEPHERD_ELECTION_POLICY_VERSION = "BRANCH_SCALED_CAPTURE_V1"

early_capture_regions = {
    "UP": pygame.Rect(
        center_x - half_width,
        center_y - half_width - normal_length,
        corridor_width,
        UP_EARLY_CAPTURE_DEPTH,
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
    FORM_JUNCTION_GUARDS = auto()
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


@dataclass
class BranchEdge:
    """Topological edge metadata prepared for recursive multi-Junction DFS."""

    edge_id: str
    source_junction_id: str
    target_node_id: Optional[str]
    direction: str
    length: float


@dataclass
class JunctionState:
    """All state owned by one Junction and its elected Anchor."""

    junction_id: str
    rect: pygame.Rect
    anchor_region: pygame.Rect
    anchor_slots: list[pygame.Vector2]
    branch_edges: dict[str, BranchEdge]
    branch_states: dict[str, str]
    gate_states: dict[str, str]
    selected_branch: Optional[str] = None
    anchor_robot_id: Optional[int] = None
    selected_anchor_position: Optional[pygame.Vector2] = None
    parent_junction_id: Optional[str] = None
    parent_edge_id: Optional[str] = None
    depth: int = 0
    state_epoch: int = 1
    state_sequence: int = 0
    updated_at: float = 0.0
    _recorded_signature: Optional[tuple] = field(
        default=None,
        repr=False,
    )

    def edge_id_for_branch(self, branch: str) -> Optional[str]:
        edge = self.branch_edges.get(branch)
        return edge.edge_id if edge is not None else None

    def record_consensus(
        self,
        states: dict[str, str],
        gates: dict[str, str],
        selected_branch: Optional[str],
        updated_at: float,
    ) -> bool:
        """Store a versioned state snapshot and report whether it changed."""
        states_snapshot = dict(states)
        gates_snapshot = dict(gates)
        signature_after = (
            tuple(sorted(states_snapshot.items())),
            tuple(sorted(gates_snapshot.items())),
            selected_branch,
        )
        if self._recorded_signature == signature_after:
            return False
        self.branch_states.clear()
        self.branch_states.update(states_snapshot)
        self.gate_states.clear()
        self.gate_states.update(gates_snapshot)
        self.selected_branch = selected_branch
        self._recorded_signature = signature_after
        self.state_sequence += 1
        self.updated_at = updated_at
        return True


CURRENT_JUNCTION_ID = "J0"


def create_single_junction_registry() -> dict[str, JunctionState]:
    """Register the present map as J0 using the multi-Junction data model."""
    branch_edges = {
        branch: BranchEdge(
            edge_id=f"{CURRENT_JUNCTION_ID}:{branch}",
            source_junction_id=CURRENT_JUNCTION_ID,
            target_node_id=BRANCH_TARGET_NODE[branch],
            direction=branch,
            length=BRANCH_LENGTHS[branch],
        )
        for branch in BRANCHES
    }
    state = JunctionState(
        junction_id=CURRENT_JUNCTION_ID,
        rect=junction_rect.copy(),
        anchor_region=anchor_election_rect.copy(),
        anchor_slots=[slot.copy() for slot in ANCHOR_PARK_SLOTS],
        branch_edges=branch_edges,
        branch_states={branch: "UNVISITED" for branch in BRANCHES},
        # All mouths are physically and logically OPEN during free diffusion.
        # They close only after emergent distribution confirms a Junction.
        gate_states={branch: "OPEN" for branch in BRANCHES},
    )
    return {CURRENT_JUNCTION_ID: state}


junctions = create_single_junction_registry()
junction_anchors: dict[str, "Robot"] = {}


def get_junction_state(junction_id: str = CURRENT_JUNCTION_ID) -> JunctionState:
    return junctions[junction_id]


phase = SimulationPhase.MOVE_TO_JUNCTION
active_branch = FIXED_BRANCH_ORDER[0]
branch_states = get_junction_state().branch_states
branch_order_plan: list[str] = []
branch_gate_states = get_junction_state().gate_states
distributed_consensus_branch: Optional[str] = None
transfer_branch: Optional[str] = None
final_base_transfer_active = False
transfer_path_max_gap = 0.0
transfer_entrance_count = 0
transfer_gap_control = 0.0
transfer_target_motion_scale = 1.0
branch_fill_target_count = 0
branch_fill_current_count = 0
branch_fill_deficit_control = 0.0
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
pre_shepherd_branch: Optional[str] = None
pre_shepherd_pack_dwell = 0.0
pre_shepherd_pack_ready = False
draining_branch: Optional[str] = None
initial_release_flow_dwell = 0.0
initial_release_event_time: Optional[float] = None
initial_release_flow_count = 0
initial_release_flow_ratio = 0.0
initial_release_average_speed = 0.0
viscoelastic_step = 0
viscoelastic_rest_lengths: dict[tuple[int, int], float] = {}
viscoelastic_last_seen: dict[tuple[int, int], int] = {}

communication_sequence = 0
last_message_signature = None
communication_redundant_links: list[tuple[object, object]] = []
backtrack_bridge_required_count = 0
backtrack_bridge_candidate_count = 0
backtrack_bridge_candidate_dwell = 0.0
backtrack_bridge_risk_level = "STABLE"
backtrack_bridge_natural_redundancy = 0
backtrack_bridge_natural_margin = 0.0
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

# Inference state is separate from labelled map metadata. Contact positions
# contain only robot-observed poses, branch-local association, and timestamps.
detected_branch_candidates: set[str] = set()
collision_points: deque["CollisionPoint"] = deque(maxlen=256)
effective_branch_widths: dict[str, float] = {
    branch: 0.0 for branch in BRANCHES
}
junction_guard_groups: dict[str, list[int]] = {}
junction_guard_frontier_depths: dict[str, float] = {}
junction_guard_formation_timer = 0.0
junction_guard_stable_dwell = 0.0
junction_guard_status = "OPEN_FREE_DIFFUSION"
pending_branch_start: Optional[str] = None
thick_mouth_guard_layers: dict[str, int] = {
    branch: 0 for branch in BRANCHES
}
thick_mouth_guard_columns: dict[str, int] = {
    branch: 0 for branch in BRANCHES
}
frontier_line_branch: Optional[str] = None
frontier_line_depth = 0.0
observed_dead_end_depths: dict[str, float] = {}

# =========================================================
# 4. Physics and control parameters
# =========================================================

FLUID_BODY_POLICY_VERSION = "CONTINUOUS_SPH_BODY_680_LARGE_MAP_V4"
ROBOT_COUNT = int(os.environ.get("SPH_DFS_ROBOT_COUNT", "680"))
BASE_RESERVE_HOLD_GAIN = 42.0
SPAWN_MODE = "grid"
ROBOT_RADIUS = 1.60 * MAP_SCALE
GRID_SPACING = 4.00 * MAP_SCALE
GRID_ROW_SPACING = 3.80 * MAP_SCALE

SMOOTHING_LENGTH = 22.0 * MAP_SCALE
PRESSURE_GAIN = 2800.0
SPH_MOTION_PRESSURE_BOOST = 3.00
SHEPHERD_PACKED_PRESSURE_BOOST = 5.00
STIFFNESS_EXPONENT = 0.5
VISCOSITY_XI1 = 0.9
VISCOSITY_XI2 = 1.2
MOTION_SPEED_MULTIPLIER = 3.0
DAMPING = 4.0
SAFE_RADIUS = 7.5 * MAP_SCALE
REPULSION_GAIN = 260.0
NORMAL_EQUILIBRIUM_SCALE = 1.48
# Physical-robot command conditioning.  Longitudinal motion remains fast,
# while high-frequency SPH reversals and corridor-crossing zigzags are damped.
ACCELERATION_FILTER_ALPHA = 0.18
CORRIDOR_LATERAL_VELOCITY_DAMPING = 12.0
SPH_PRESSURE_FORCE_LIMIT = 420.0 * MOTION_SPEED_MULTIPLIER
SPH_VISCOSITY_FORCE_LIMIT = 150.0 * MOTION_SPEED_MULTIPLIER
VISCOELASTIC_LINK_RADIUS = SAFE_RADIUS * 1.45
VISCOELASTIC_REST_MIN = ROBOT_RADIUS * 2.05
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
COMPRESSION_RELEASE_GAIN = 135.0
COMPRESSION_RELEASE_FORCE_LIMIT = 180.0 * MOTION_SPEED_MULTIPLIER
COMPRESSION_RELEASE_RAMP_TIME = 0.24
PRESSURE_POLICY_VERSION = "SPH_SELF_EXPANSION_V1"
VISCOELASTIC_MODEL_VERSION = "KELVIN_VOIGT_SPH_V2_SPACED"
MOTION_POLICY_VERSION = "FAST_STABLE_COMMAND_V3"

ROUTE_FORCE = 22.0 * MOTION_SPEED_MULTIPLIER
WEAK_BRANCH_BIAS_FORCE = 3.5 * MOTION_SPEED_MULTIPLIER
OUTLET_FORCE = 24.0 * MOTION_SPEED_MULTIPLIER
# During Shepherd filling, robots already inside the active branch receive a
# strong longitudinal force.  Give the Base and Junction tails their own
# smoothly-ramped feed gains so they do not remain pooled in the lower trunk.
FILL_BASE_FEED_MULTIPLIER = 1.80
FILL_JUNCTION_FEED_MULTIPLIER = 1.35
FILL_FEED_RAMP_TIME = 0.50
# Keep the Base, Junction, and selected Branch as one lane-preserving cohort.
# The Branch head is throttled only while the mouth is sparse, while the
# crowded tail is gently accelerated and expanded until the hole closes.
FILL_BRANCH_CRUISE_MIN_SCALE = 0.30
FILL_TAIL_FEED_BOOST = 1.10
FILL_TAIL_EQUILIBRIUM_EXPANSION = 0.35
FILL_LANE_LOOKAHEAD = SMOOTHING_LENGTH * 0.80
FILL_LANE_WALL_CLEARANCE = max(ROBOT_RADIUS * 2.5, 3.0 * MAP_SCALE)
FILL_BRANCH_LANE_CENTERING_GAIN = 6.0
FILL_BRANCH_LANE_FORCE_LIMIT = ROUTE_FORCE * 2.00
BRANCH_FILL_QUOTA_COVERAGE = 1.00
BRANCH_FILL_QUOTA_AXIAL_PITCH_RATIO = 0.95
BRANCH_FILL_QUOTA_MIN_RESERVE_ROWS = 2
BRANCH_FILL_QUOTA_SOURCE_FORCE_BOOST = 1.80
BRANCH_FILL_QUOTA_FILTER_ALPHA = 0.20
BRANCH_FILL_QUOTA_POLICY_VERSION = "LENGTH_WIDTH_EQUILIBRIUM_QUOTA_V3"
COHORT_FLOW_POLICY_VERSION = "LANE_PRESERVING_DENSITY_CONTINUITY_V4"
FLOW_BACKTRACK_FORCE = 46.0 * MOTION_SPEED_MULTIPLIER
FINAL_GATHER_FORCE = 58.0 * MOTION_SPEED_MULTIPLIER
PRESSURE_BACKTRACK_BODY_FORCE = 8.0 * MOTION_SPEED_MULTIPLIER
CENTERING_GAIN = 1.2

MAX_SPEED = 78.0 * MOTION_SPEED_MULTIPLIER
MAX_ACCELERATION = 300.0 * MOTION_SPEED_MULTIPLIER
PRESSURE_PUSH_MAX_SPEED = 42.0 * MOTION_SPEED_MULTIPLIER
FLOW_BACKTRACK_MAX_SPEED = 52.0 * MOTION_SPEED_MULTIPLIER
EPSILON = 1e-8

INITIAL_INGRESS_FORCE = 2.5 * MOTION_SPEED_MULTIPLIER
BASE_COMPRESSION_DURATION = 0.65
BASE_EXPANSION_BOOST_DURATION = 3.20
BASE_COMPRESSION_FORCE = 80.0 * MOTION_SPEED_MULTIPLIER
BASE_COMPRESSION_RISE_FRACTION = 0.20
BASE_COMPRESSION_FALL_START_FRACTION = 0.80
BASE_COMPRESSION_PRESSURE_SCALE = 0.35
BASE_EXPANSION_PRESSURE_SCALE = 5.20
BASE_EXPANSION_RAMP_FRACTION = 0.22
BASE_PACKED_EQUILIBRIUM_SCALE = 0.60
BASE_EQUILIBRIUM_RELEASE_DURATION = 0.40
BASE_STORED_PRESSURE_FLOOR = 0.85
BASE_STORED_PRESSURE_DECAY_START = 2.80
BASE_STORED_PRESSURE_DURATION = 6.00
BASE_STORED_PRESSURE_RISE_TIME = 0.12
BASE_PISTON_REACTION_GAIN = 260.0 * MOTION_SPEED_MULTIPLIER
BASE_PISTON_REACTION_FORCE_LIMIT = 260.0 * MOTION_SPEED_MULTIPLIER
BASE_PISTON_REACTION_RISE_TIME = 0.25
BASE_PISTON_REACTION_DURATION = 5.00
BASE_PISTON_REACTION_DEPTH_START = 0.00
INITIAL_RELEASE_PRESSURE_FORCE_LIMIT = 260.0 * MOTION_SPEED_MULTIPLIER
INITIAL_RELEASE_VISCOSITY_MULTIPLIER = 1.35
INITIAL_RELEASE_EXTRA_DAMPING = 6.0
INITIAL_RELEASE_ACCELERATION_FILTER_ALPHA = 0.12
INITIAL_INGRESS_LANE_GAIN = 0.25
INITIAL_INGRESS_LANE_MAX_FORCE = 5.0
# Blind straight-ahead probe along the incoming corridor heading.  The target
# lies beyond the Junction envelope, but it does not assert that an opening is
# present; successful physical crossing is still required by the detector.
INITIAL_INGRESS_TARGET_Y = (
    center_y - half_width - 18.0 * MAP_SCALE
)
INITIAL_JUNCTION_PROBE_ROUTE_MULTIPLIER = 3.0
INITIAL_INGRESS_BRAKE_DISTANCE = 52.0 * MAP_SCALE
INITIAL_INGRESS_MIN_FORCE_SCALE = 0.10
INITIAL_INGRESS_MAX_DT = 0.04
INITIAL_SAFE_MAX_SPEED = 36.0 * MOTION_SPEED_MULTIPLIER
INITIAL_JUNCTION_MAX_SPEED = 26.0 * MOTION_SPEED_MULTIPLIER
INITIAL_SAFE_MAX_ACCELERATION = 200.0 * MOTION_SPEED_MULTIPLIER
INITIAL_JUNCTION_SOFT_WALL_DEPTH = 10.0 * MAP_SCALE
INITIAL_JUNCTION_SOFT_WALL_FORCE = 70.0 * MOTION_SPEED_MULTIPLIER
INITIAL_JUNCTION_SOFT_WALL_DAMPING = 5.0
INITIAL_WALL_RESTITUTION = 0.18
INITIAL_RELEASE_FLOW_MIN_ROBOTS = 18
INITIAL_RELEASE_FLOW_SPEED_THRESHOLD = 3.0
INITIAL_RELEASE_FLOW_RATIO_THRESHOLD = 0.40
INITIAL_RELEASE_FLOW_AVERAGE_SPEED_THRESHOLD = 4.0
INITIAL_RELEASE_FLOW_DWELL_TIME = 0.12
INITIAL_RELEASE_EVENT_DECAY_TIME = 0.45
INITIAL_RELEASE_EDF_BLEND_TIME = 0.25
INITIAL_IMPULSE_POLICY_VERSION = "EVENT_GATED_FLOW_RELEASE_V7"

RETURN_EGRESS_FORCE = 42.0 * MOTION_SPEED_MULTIPLIER
RETURN_LANE_GAIN = 1.15
RETURN_LANE_MAX_FORCE = 22.0
RETURN_BRAKE_DISTANCE = 34.0 * MAP_SCALE
RETURN_MIN_FORCE_SCALE = 0.20
# Repack the swarm while gathering inside the finite Base corridor.  Without
# this return-only spacing/pressure mode, normal SPH expansion can balance the
# downward route force and leave one or more robots hovering in the Junction,
# so the exact all-robots-in-BOTTOM completion condition never becomes true.
RETURN_PACKED_EQUILIBRIUM_SCALE = BASE_PACKED_EQUILIBRIUM_SCALE
RETURN_PACKING_PRESSURE_SCALE = BASE_COMPRESSION_PRESSURE_SCALE
RETURN_TRUNK_RETRACT_DWELL = 0.55
RETURN_TRUNK_RELEASE_INITIAL_SPEED = 12.0
RETURN_TRUNK_READY_BOTTOM_TOLERANCE = 2
RETURN_TRUNK_READY_CONNECTED_RATIO = 0.97
RETURN_TRUNK_FORCE_RELEASE_TIMEOUT = 2.50
RETURN_DONE_DWELL_TIME = 0.35
RETURN_ENTRY_RECOVERY_TIMEOUT = 1.50
RETURN_ENTRY_RECOVERY_DISTANCE = 18.0 * MAP_SCALE
RETURN_STRAGGLER_FORCE_MULTIPLIER = 1.80
NORMAL_PHYSICS_MAX_DT = 0.05

ISOLATION_NEIGHBOR_THRESHOLD = 6
ISOLATION_ROUTE_BOOST = 1.1
# Density-adaptive attraction for sparse sections of the continuous fluid
# body.  It is stronger than the former fixed cohesion only when both the
# local neighbor count and density are low, so packed regions are unaffected.
LOCAL_COHESION_GAIN = 32.0
LOCAL_COHESION_DENSITY_TARGET_RATIO = 1.15
LOCAL_COHESION_DENSITY_BOOST = 1.25
LOCAL_COHESION_FORCE_LIMIT = 60.0

JUNCTION_ENTRY_COUNT = 18
ANCHOR_MOVE_SPEED = 42.0
ANCHOR_POSITION_TOLERANCE = 2.5 * MAP_SCALE
JUNCTION_SWITCH_COUNT = 18
JUNCTION_SWITCH_DWELL_TIME = 0.25
RETURN_BOTTOM_TARGET_COUNT = ROBOT_COUNT
FINAL_GATHER_DWELL_TIME = 0.55

# Minimum-cost multi-criteria Anchor election
ANCHOR_ELECTION_MIN_CANDIDATES = 4
ANCHOR_ELECTION_WAIT_TIME = 0.22
ANCHOR_COST_WEIGHT_ARRIVAL = 0.25
ANCHOR_COST_WEIGHT_PARKING = 0.25
ANCHOR_COST_WEIGHT_OBSTRUCTION = 0.30
ANCHOR_COST_WEIGHT_COMMUNICATION = 0.20
ANCHOR_LOCAL_COMM_RANGE = 56.0 * MAP_SCALE
ANCHOR_OBSTRUCTION_RADIUS = 18.0 * MAP_SCALE
ANCHOR_POLICY_VERSION = "JUNCTION_SCOPED_MIN_COST_RELAY_V3"

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
SATURATION_PACKED_LATERAL_COVERAGE_RATIO = 0.45
SATURATION_GEOMETRY_LATERAL_COVERAGE_RATIO = 0.55
SATURATION_PACKED_DWELL_TIME = 0.10
SATURATION_POLICY_VERSION = "CROSS_SECTION_PACKED_FAST_START_V4"
PRESSURE_START_POLICY_VERSION = (
    "LOCAL_PACKED_OVERRIDES_GLOBAL_CONTINUITY_V1"
)
LONG_BRANCH_FILL_LENGTH_RATIO = 1.50
LONG_BRANCH_FILL_EQUILIBRIUM_SCALE = 1.58

# Do not start Backtracking while the selected Branch still contains the
# visible sparse middle band.  Base/Junction robots continue the ordinary
# FILL_BEHIND_SHEPHERD feed until most longitudinal slices are populated.
BRANCH_CONTINUITY_SLICE_DEPTH = SMOOTHING_LENGTH
BRANCH_CONTINUITY_MIN_ROBOTS_PER_SLICE = 8
BRANCH_CONTINUITY_REQUIRED_SLICE_RATIO = 0.90
BRANCH_CONTINUITY_MAX_DEPTH_GAP = (
    SAFE_RADIUS * LONG_BRANCH_FILL_EQUILIBRIUM_SCALE * 1.35
)
BRANCH_CONTINUITY_DWELL_TIME = 0.28
BRANCH_CONTINUITY_FILL_TIMEOUT = 3.50

# Width-adaptive Shepherd count. Selection timing remains the same as the
# original code: leading robots are selected inside early_capture_regions.
SHEPHERD_BOUNDARY_WALL_CLEARANCE = 14.0 * MAP_SCALE
SHEPHERD_FILL_GAP = 38.0 * MAP_SCALE
SHEPHERD_FILL_REGION_DEPTH = 62.0 * MAP_SCALE
SHEPHERD_PACKED_EQUILIBRIUM_SCALE = 0.46
# The RIGHT corridor is twice as long as the other branches.  Keep the dense
# layer immediately behind the Shepherd unchanged, but let the ordinary body
# behind that layer use a slightly wider equilibrium spacing.  This covers the
# long corridor without increasing the already expensive 680-robot simulation.
# Keep the Base/Junction tail compact enough to continuously feed the selected
# branch.  The former 1.55x spacing decompressed the tail and starved the
# Junction mouth during FILL_BEHIND_SHEPHERD.
JUNCTION_TAIL_EQUILIBRIUM_SCALE = 1.42
SHEPHERD_FILL_COMPRESSION_PRESSURE_SCALE = 0.22
SHEPHERD_FILL_FORCE_MULTIPLIER = 3.6
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
SHEPHERD_PIPELINE_POLICY_VERSION = "PACK_READY_IMMEDIATE_HANDOFF_V3"
PIPELINE_SOURCE_STRAGGLER_LIMIT = 6
PRE_SHEPHERD_PACK_DWELL_TIME = 0.08
SHEPHERD_PRESSURE_FACTOR = 4.0
VIRTUAL_PRESSURE_RADIUS = 60.0 * MAP_SCALE
VIRTUAL_PRESSURE_FORCE = 110.0
SHEPHERD_VIRTUAL_FORCE_LIMIT = 140.0 * MOTION_SPEED_MULTIPLIER
PRESSURE_RAMP_TIME = 0.55

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
# The local cohesion term cannot bridge two fluid components after their
# spacing exceeds the SPH support radius.  A broad Base-front equilibrium
# field extends the attraction range without assigning special robot roles.
COMM_LOS_SAMPLE_SPACING = 6.0 * MAP_SCALE
COMM_LOS_CLEARANCE = 0.0
COMM_SAFE_DISTANCE = 40.0 * MAP_SCALE
# Cohort-continuity control.  When the current Branch backtracks more slowly
# than Junction/Base robots enter the transfer Branch, the largest 1-D gap and
# source-mouth occupancy throttle the target cohort and boost source recovery.
# The target is tied to the actual spawn lattice rather than the much wider SPH
# support radius, so a visually sparse Branch entrance is no longer accepted.
TRANSFER_CONTINUITY_TARGET_GAP = (
    SAFE_RADIUS * NORMAL_EQUILIBRIUM_SCALE * 1.05
)
TRANSFER_CONTINUITY_DANGER_GAP = (
    SAFE_RADIUS * NORMAL_EQUILIBRIUM_SCALE * 1.90
)
TRANSFER_CONTINUITY_ENTRANCE_DEPTH = SMOOTHING_LENGTH * 1.50
TRANSFER_CONTINUITY_MIN_ENTRANCE_ROBOTS = JUNCTION_SWITCH_COUNT
BASE_FRONT_EQUILIBRIUM_RADIUS = COMM_RANGE * 2.80
BASE_FRONT_EQUILIBRIUM_DISTANCE = (
    SAFE_RADIUS * NORMAL_EQUILIBRIUM_SCALE
)
BASE_FRONT_BAND_DEPTH = SMOOTHING_LENGTH * 2.20
BASE_FRONT_BRANCH_TAIL_DEPTH = SMOOTHING_LENGTH * 5.00
BASE_FRONT_LINKS_PER_ROBOT = 4
BASE_FRONT_EQUILIBRIUM_EXPANSION = 0.55
BASE_FRONT_EQUILIBRIUM_GAIN = 3.00
BASE_FRONT_EQUILIBRIUM_DAMPING_GAIN = 0.65
BASE_FRONT_EQUILIBRIUM_FORCE_LIMIT = ROUTE_FORCE * 4.00
BASE_FRONT_EQUILIBRIUM_TAPER_START_RATIO = 0.85
BASE_FRONT_FILL_ACTIVATION_FLOOR = 0.45
BASE_FRONT_HANDOFF_ACTIVATION_FLOOR = 0.25
BASE_FRONT_TAIL_REACTION_SCALE = 0.70
BASE_FRONT_FOLLOW_MIN_WEIGHT = 0.35
BASE_FRONT_FOLLOW_FORCE_BOOST = 1.40
BASE_FRONT_FOLLOW_MAX_ROUTE_SCALE = 2.10
COLLECTIVE_EQUILIBRIUM_POLICY_VERSION = (
    "LANE_MAPPED_BIDIRECTIONAL_CROSS_SECTION_V3"
)
TRANSFER_CONTINUITY_MIN_TARGET_SCALE = 0.22
TRANSFER_CONTINUITY_SOURCE_FORCE_BOOST = 2.0
TRANSFER_CONTINUITY_FILTER_ALPHA = 0.25
COMM_BARRIER_START = COMM_RANGE * 0.84
COMM_CONTROL_POLICY_VERSION = "PREDICTIVE_PARENT_LEASH_V1"
COMM_GUARD_START = COMM_SAFE_DISTANCE * 0.78
COMM_GUARD_HARD_LIMIT = COMM_RANGE * 0.88
COMM_PARENT_SPRING_GAIN = 45.0
COMM_PARENT_DAMPING_GAIN = 14.0
COMM_PARENT_VELOCITY_MATCH_GAIN = 5.0
COMM_GUARD_FORCE_LIMIT = 220.0 * MOTION_SPEED_MULTIPLIER
COMM_RECOVERY_RANGE = 84.0 * MAP_SCALE
COMM_RECOVERY_GAIN = 12.0
COMM_MAX_LOCAL_NEIGHBORS = 16
SHOW_COMM_LINKS_DEFAULT = True
SHOW_DENSITY_COLOR_DEFAULT = True
COMM_UPDATE_INTERVAL_FRAMES = 1
BACKTRACK_BRIDGE_GUARD_COUNT = 4
BACKTRACK_BRIDGE_SLOT_SPREAD = 12.0 * MAP_SCALE
BACKTRACK_BRIDGE_CENTER_OFFSET = 8.0 * MAP_SCALE
BACKTRACK_BRIDGE_RECRUIT_RADIUS = 95.0 * MAP_SCALE
BACKTRACK_BRIDGE_POSITION_GAIN = 18.0
BACKTRACK_BRIDGE_DAMPING_GAIN = 7.0
BACKTRACK_BRIDGE_FORCE_LIMIT = 150.0 * MOTION_SPEED_MULTIPLIER
BACKTRACK_BRIDGE_EXTRA_LINKS_PER_SIDE = 3
BACKTRACK_BRIDGE_POSITION_TOLERANCE = 4.0 * MAP_SCALE
BACKTRACK_BRIDGE_TARGET_REDUNDANCY = 3
BACKTRACK_BRIDGE_STABLE_MARGIN = COMM_RANGE * 0.22
BACKTRACK_BRIDGE_DANGER_MARGIN = COMM_RANGE * 0.08
BACKTRACK_BRIDGE_DEPLOY_DWELL = 0.12
BACKTRACK_BRIDGE_RELEASE_DWELL = 0.80
BACKTRACK_BRIDGE_POLICY_VERSION = "EVENT_TRIGGERED_ADAPTIVE_BRIDGE_V2"
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
BREADCRUMB_GUARD_POLICY_VERSION = "NO_STATIC_GUARDS_FLUID_BODY_V1"
BREADCRUMB_GUARD_PER_RELAY = 0
BREADCRUMB_GUARD_DEPTH = COMM_SAFE_DISTANCE * 1.10

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
return_done_dwell = 0.0
return_entry_stall_timer = 0.0
return_last_bottom_count = 0

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

# Geodesic EDF guidance. Runtime branch-mouth control is implemented by
# physical JUNCTION_GUARD robots, not map-aware virtual valves/geofences.
EDF_FINITE_EPSILON = 1e-6
EDF_PROPULSION_POLICY_VERSION = "POST_RELEASE_BRANCH_CRUISE_EDF_V5"
EDF_PRESSURE_COUPLING_GAIN = 36.0 * MOTION_SPEED_MULTIPLIER
EDF_PRESSURE_FORCE_LIMIT = 48.0 * MOTION_SPEED_MULTIPLIER
EDF_PRESSURE_RAMP_TIME = 0.25
EXPLORATION_PRESSURE_FLOOR = 0.85
EXPLORATION_PRESSURE_FLOOR_RAMP_TIME = 0.45
EXPLORATION_CRUISE_EDF_MULTIPLIER = 1.75
EXPLORATION_EDF_FORCE_LIMIT = 64.0 * MOTION_SPEED_MULTIPLIER
TRANSFER_PRESSURE_FLOOR = 0.65
TRANSFER_EDF_FORCE_MULTIPLIER = 1.25
VIRTUAL_VALVE_RADIUS = 46.0 * MAP_SCALE
VIRTUAL_VALVE_GAIN = 92.0
FINAL_RETURN_VALVE_GAIN_MULTIPLIER = 4.0
FINAL_GATE_POLICY_VERSION = "PHYSICAL_GUARD_NO_GEOFENCE_V1"

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

# Emmons et al. observed normalized local robot densities over time.  This
# simulator uses the same principle in a heading-aligned angular frame and
# validates persistent motion cohorts before exposing a branch to DFS.
JUNCTION_DISTRIBUTION_WINDOW = 24
JUNCTION_OBSERVATION_RADIUS = corridor_width * 1.35
JUNCTION_FRONT_QUANTILE = 0.68
JUNCTION_MIN_OBSERVATION_ROBOTS = 18
JUNCTION_BASELINE_ALPHA = 0.035
JUNCTION_LATERAL_EXPANSION_RATIO = 1.28
JUNCTION_LATERAL_EXPANSION_MIN = (4.5 * MAP_SCALE) ** 2
JUNCTION_EXPANSION_DWELL_TIME = 0.14
JUNCTION_COHORT_HALF_ANGLE = math.radians(38.0)
JUNCTION_COHORT_MIN_SPEED = 1.2
JUNCTION_COHORT_MIN_ROBOTS = 8
JUNCTION_COHORT_MIN_FRACTION = 0.045
JUNCTION_COHORT_MIN_TRAVEL = 12.0 * MAP_SCALE
JUNCTION_COHORT_MIN_BRANCH_DEPTH = 12.0 * MAP_SCALE
JUNCTION_COHORT_DWELL_TIME = 0.22
JUNCTION_MIN_VALID_COHORTS = 2
JUNCTION_DISCOVERY_SETTLE_TIME = 1.00
JUNCTION_FRONT_BLOCK_MIN_CONTACTS = 4
JUNCTION_FRONT_BLOCK_MIN_SPAN_RATIO = 0.80
JUNCTION_PROBE_DEPTH = 24.0 * MAP_SCALE
JUNCTION_PROBE_BARRIER_GAIN = 70.0 * MOTION_SPEED_MULTIPLIER
JUNCTION_INFERENCE_POLICY_VERSION = "LOCAL_CROSSING_DENSITY_COHORT_V2"

# Junction-mouth Shepherd/Guard lifecycle.  Full-width guards are recruited
# only after Junction confirmation, using a leader-rooted minimum K-hop set.
JUNCTION_GUARD_COVERAGE = 12.5 * MAP_SCALE
JUNCTION_GUARD_MIN_COUNT = 5
JUNCTION_GUARD_MAX_COUNT = 11
JUNCTION_GUARD_BRANCH_INSET = 5.0 * MAP_SCALE
JUNCTION_GUARD_RECRUIT_RADIUS = 78.0 * MAP_SCALE
JUNCTION_GUARD_FRONTIER_MARGIN = ROBOT_RADIUS * 2.5
JUNCTION_GUARD_TERMINAL_DEPTH_EPSILON = ROBOT_RADIUS * 1.5
JUNCTION_GUARD_MOVE_SPEED = 46.0 * MOTION_SPEED_MULTIPLIER
JUNCTION_GUARD_POSITION_TOLERANCE = 3.0 * MAP_SCALE
JUNCTION_GUARD_FORM_DWELL = 0.10
JUNCTION_GUARD_FORM_TIMEOUT = 1.50
JUNCTION_MINIMAL_GUARD_COUNT = 2
JUNCTION_GUARD_MAX_HOPS = 4
JUNCTION_GUARD_POLICY_VERSION = "PERSISTENT_PHYSICAL_MOUTH_GUARD_V1"
PHYSICAL_GUARD_INFLUENCE_RADIUS = max(
    JUNCTION_GUARD_COVERAGE * 0.78,
    SAFE_RADIUS * 1.15,
)
PHYSICAL_GUARD_INWARD_GAIN = 118.0 * MOTION_SPEED_MULTIPLIER
PHYSICAL_GUARD_LATERAL_GAIN = 32.0 * MOTION_SPEED_MULTIPLIER
PHYSICAL_GUARD_FORCE_LIMIT = 105.0 * MOTION_SPEED_MULTIPLIER
THICK_MOUTH_GUARD_POLICY_VERSION = "ADAPTIVE_KHOP_LAYERED_MOUTH_WALL_V1"
THICK_MOUTH_GUARD_MIN_LAYERS = 2
THICK_MOUTH_GUARD_MAX_LAYERS = JUNCTION_GUARD_MAX_HOPS
THICK_MOUTH_GUARD_LAYER_SPACING = max(
    SAFE_RADIUS * 1.10,
    ROBOT_RADIUS * 2.25,
)
THICK_MOUTH_GUARD_FORM_DWELL = 0.18
THICK_MOUTH_GUARD_FORM_TIMEOUT = 4.0
THICK_MOUTH_GUARD_LARGE_SWARM_SIZE = 400
THICK_MOUTH_GUARD_VERY_LARGE_SWARM_SIZE = 900
FRONTIER_POLICY_VERSION = "LEADER_BIASED_DEFORMABLE_FRONTIER_V1"
FRONTIER_LINE_ADVANCE_SPEED = 52.0 * MOTION_SPEED_MULTIPLIER
FRONTIER_LINE_FORM_SPEED = 58.0 * MOTION_SPEED_MULTIPLIER
FRONTIER_LINE_LEAD_GAP = 12.0 * MAP_SCALE
FRONTIER_LINE_SUPPORT_QUANTILE = 0.98
FRONTIER_LINE_START_DEPTH = JUNCTION_GUARD_BRANCH_INSET
FRONTIER_LINE_POLICY_VERSION = "PERSISTENT_CROSS_SECTION_SHEPHERD_V1"

# Eguchi et al. integrate normalized command/observed speed discrepancy and
# record the position when the threshold is crossed.  The constants below are
# expressed per nominal simulator control tick, with a short refractory period
# to avoid duplicating one physical contact on every frame.
CONTACT_ERROR_ATTENUATION = 0.004
CONTACT_ERROR_THRESHOLD = 0.55
CONTACT_COMMAND_MIN_SPEED = 2.0
CONTACT_EVENT_COOLDOWN = 0.16
CONTACT_POINT_MERGE_RADIUS = 4.0 * MAP_SCALE
CONTACT_POINT_MAX_COUNT = 256
CONTACT_POINT_REPULSION_RADIUS = SMOOTHING_LENGTH * 1.60
CONTACT_POINT_WEAK_GAIN = 12.0 * MOTION_SPEED_MULTIPLIER
CONTACT_POINT_STRONG_GAIN = 42.0 * MOTION_SPEED_MULTIPLIER
CONTACT_POINT_FORCE_LIMIT = 90.0 * MOTION_SPEED_MULTIPLIER
CONTACT_EVENT_MEMORY = 0.65
INDIRECT_CONTACT_POLICY_VERSION = "EGUCHI_TRACKING_ERROR_V1"

BRANCH_WIDTH_SAMPLE_DEPTH = 44.0 * MAP_SCALE
BRANCH_WIDTH_MIN_FLOW_SAMPLES = 8
BRANCH_WIDTH_MIN_CONTACT_SAMPLES = 2
BRANCH_WIDTH_FLOW_WEIGHT = 0.65
BRANCH_WIDTH_CONTACT_WEIGHT = 0.35
BRANCH_CONTACT_RISK_WEIGHT = 0.08

DEAD_END_FRONTIER_DEPTH = 22.0 * MAP_SCALE
DEAD_END_MIN_FRONTIER_ROBOTS = 8
DEAD_END_LEADER_CONTACT_THRESHOLD = 0.80
DEAD_END_MEAN_CONTACT_THRESHOLD = 0.22
DEAD_END_FORWARD_SPEED_THRESHOLD = 8.0
# The frontier reference density can be below the initially compressed Base
# density (especially in reduced-count experiments), so use a local packed
# floor rather than demanding rho_front > rho_base.
DEAD_END_DENSITY_RATIO = 0.55
DEAD_END_LATERAL_ESCAPE_RATIO = 0.45
DEAD_END_CONFIRM_DWELL = 0.16
DEAD_END_SHEPHERD_DIRECT_CONTACT_RATIO = 0.60
DEAD_END_SHEPHERD_CONTACT_SPAN_RATIO = 0.55
DEAD_END_FORWARD_BUMPER_MEMORY = 0.30
DEAD_END_FORWARD_BUMPER_PROBE = ROBOT_RADIUS * 0.85
DEAD_END_INFERENCE_POLICY_VERSION = "SHEPHERD_FORWARD_BUMPER_SPAN_V3"

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
    """Accept every physical corridor independently of logical gate state.

    ``branch_gate_states`` is a distributed command state, not a simulator
    geofence. Only actual environment walls and locally sensed guard robots
    may affect locomotion.
    """
    region = get_robot_region(position)
    return region in {"BOTTOM", "JUNCTION", "UP", "LEFT", "RIGHT"}


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


def constrain_final_return_gate_crossing(
    robot: "Robot",
    old_position: pygame.Vector2,
) -> None:
    """Compatibility no-op: final return has no map-aware one-way gate."""
    return


def wall_collision_velocity(component: float) -> float:
    """Apply a small, damped rebound during pressure-driven forward flow."""
    if (
        phase in {
            SimulationPhase.MOVE_TO_JUNCTION,
            SimulationPhase.EXPLORE_BRANCH,
        }
        and simulation_time >= BASE_COMPRESSION_DURATION
    ):
        return -component * INITIAL_WALL_RESTITUTION
    return 0.0


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
    """Dark fluid-density palette: blue -> cobalt -> midnight navy."""
    ratio = density / max(reference_density, EPSILON)
    if ratio <= 0.85:
        return interpolate_color(
            (44, 95, 135),
            (31, 74, 119),
            ratio / 0.85,
        )
    if ratio <= 1.15:
        return interpolate_color(
            (31, 74, 119),
            (20, 53, 96),
            (ratio - 0.85) / 0.30,
        )
    if ratio <= 1.55:
        return interpolate_color(
            (20, 53, 96),
            (13, 36, 73),
            (ratio - 1.15) / 0.40,
        )
    return interpolate_color(
        (13, 36, 73),
        (6, 20, 48),
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
    observed_depth = observed_dead_end_depths.get(branch)
    if observed_depth is not None:
        return max(0.0, observed_depth)
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
    if phase == SimulationPhase.EXPLORE_BRANCH:
        return (
            frontier_line_branch == active_branch
            and frontier_line_depth > 0.0
        )
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
    if (
        phase == SimulationPhase.EXPLORE_BRANCH
        and frontier_line_branch == branch
    ):
        depth = frontier_line_depth
    elif phase == SimulationPhase.PRESSURE_PUSH:
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


def branch_fill_equilibrium_spacing(branch: str) -> float:
    """Return the intended center spacing for the ordinary Branch body."""
    equilibrium_scale = (
        LONG_BRANCH_FILL_EQUILIBRIUM_SCALE
        if BRANCH_LENGTHS[branch]
        >= normal_length * LONG_BRANCH_FILL_LENGTH_RATIO
        else NORMAL_EQUILIBRIUM_SCALE
    )
    return max(
        GRID_SPACING,
        SAFE_RADIUS * equilibrium_scale,
    )


def calculate_branch_fill_quota(robots, branch: str) -> int:
    """Calculate how many mobile robots are needed to fill one Branch.

    The target follows the physical Branch length and width.  Its lane count is
    derived from the same equilibrium spacing used by the SPH body, while a
    small mobile reserve remains outside the Branch for Junction continuity.
    """
    spacing = branch_fill_equilibrium_spacing(branch)
    usable_width = max(
        spacing,
        corridor_width - 2.0 * FILL_LANE_WALL_CLEARANCE,
    )
    lane_count = max(
        1,
        math.floor(
            (usable_width + EPSILON) / spacing
        )
        + 1,
    )
    usable_depth = max(
        spacing,
        get_shepherd_boundary_depth(branch)
        - SHEPHERD_CURTAIN_CLEARANCE,
    )
    axial_pitch = max(
        GRID_SPACING,
        spacing * BRANCH_FILL_QUOTA_AXIAL_PITCH_RATIO,
    )
    row_count = max(
        1,
        math.ceil(usable_depth / axial_pitch),
    )
    geometric_target = math.ceil(
        lane_count
        * row_count
        * BRANCH_FILL_QUOTA_COVERAGE
    )
    mobile_count = sum(
        robot.role in {"NORMAL", "SHEPHERD"}
        and not robot.base_reserve
        for robot in robots
    )
    reserve_count = max(
        JUNCTION_SWITCH_COUNT,
        lane_count * BRANCH_FILL_QUOTA_MIN_RESERVE_ROWS,
    )
    return int(
        clamp(
            geometric_target,
            1,
            max(1, mobile_count - reserve_count),
        )
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


def pre_shepherd_curtain_active() -> bool:
    return (
        pre_shepherd_branch is not None
        and phase in {
            SimulationPhase.PRESSURE_PUSH,
            SimulationPhase.FLOW_BACKTRACK,
            SimulationPhase.EXPLORE_BRANCH,
        }
    )


def get_pre_shepherd_normal_limit_depth(branch: str) -> float:
    return max(
        0.0,
        get_shepherd_boundary_depth(branch)
        - SHEPHERD_CURTAIN_CLEARANCE,
    )


def compute_pre_shepherd_curtain_force(
    robot: "Robot",
) -> pygame.Vector2:
    """Static full-width shield formed early in the transfer branch."""
    branch = pre_shepherd_branch
    if (
        not pre_shepherd_curtain_active()
        or branch is None
        or robot.role != "NORMAL"
        or get_robot_region(robot.position) != branch
    ):
        return pygame.Vector2()
    limit_depth = get_pre_shepherd_normal_limit_depth(branch)
    depth = branch_depth_from_junction(robot.position, branch)
    activation_depth = (
        limit_depth - SHEPHERD_CURTAIN_INTERACTION_DEPTH
    )
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
        robot.velocity.dot(BRANCH_DIRECTIONS[branch]),
    )
    magnitude = min(
        SHEPHERD_CURTAIN_MAX_FORCE,
        SHEPHERD_CURTAIN_FORCE * ratio**2
        + SHEPHERD_CURTAIN_VELOCITY_DAMPING * forward_speed,
    )
    return get_backtrack_direction(branch) * magnitude


def constrain_normal_behind_pre_shepherd_curtain(
    robot: "Robot",
) -> None:
    """Hard guard preventing normals from passing the prepared shield."""
    branch = pre_shepherd_branch
    if (
        not pre_shepherd_curtain_active()
        or branch is None
        or robot.role != "NORMAL"
        or get_robot_region(robot.position) != branch
    ):
        return
    limit_depth = get_pre_shepherd_normal_limit_depth(branch)
    depth = branch_depth_from_junction(robot.position, branch)
    if depth <= limit_depth:
        return
    penetration = depth - limit_depth
    if branch == "UP":
        robot.position.y = center_y - half_width - limit_depth
    elif branch == "LEFT":
        robot.position.x = center_x - half_width - limit_depth
    else:
        robot.position.x = center_x + half_width + limit_depth
    forward_direction = BRANCH_DIRECTIONS[branch]
    forward_speed = robot.velocity.dot(forward_direction)
    if forward_speed > 0.0:
        robot.velocity -= forward_direction * forward_speed
    robot.velocity += get_backtrack_direction(branch) * min(
        SHEPHERD_CURTAIN_RECOVERY_SPEED,
        penetration * 2.0,
    )


def enforce_pre_shepherd_curtain_for_swarm(robots) -> None:
    for robot in robots:
        constrain_normal_behind_pre_shepherd_curtain(robot)


def angle_between(a: pygame.Vector2, b: pygame.Vector2) -> float:
    if a.length_squared() <= EPSILON or b.length_squared() <= EPSILON:
        return 0.0
    dot = clamp(a.normalize().dot(b.normalize()), -1.0, 1.0)
    return math.acos(dot)

# =========================================================
# 7. Experiment metrics
# =========================================================


@dataclass
class CollisionPoint:
    position: pygame.Vector2
    detected_at: float
    robot_id: int
    branch: Optional[str]


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
    junction_inference_events: list[dict] = field(default_factory=list)
    contact_events: list[dict] = field(default_factory=list)
    dead_end_events: list[dict] = field(default_factory=list)
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
        writer.writerow(["junction_inference_event_count", len(metrics.junction_inference_events)])
        writer.writerow(["indirect_contact_event_count", len(metrics.contact_events)])
        writer.writerow(["dead_end_inference_event_count", len(metrics.dead_end_events)])
        for branch in BRANCHES:
            writer.writerow([
                f"effective_width_{branch.lower()}",
                f"{effective_branch_widths.get(branch, 0.0):.6f}",
            ])
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
        self.last_base_piston_force = 0.0
        self.last_edf_force = 0.0
        self.commanded_velocity = pygame.Vector2()
        self.observed_velocity = pygame.Vector2()
        self.tracking_error_integral = 0.0
        self.contact_detected = False
        self.last_contact_time = float("-inf")
        self.last_forward_obstacle_contact_time = float("-inf")
        self.latest_contact_point: Optional[pygame.Vector2] = None
        self.role = "NORMAL"

        self.shepherd_anchor: Optional[pygame.Vector2] = None
        self.shepherd_origin: Optional[pygame.Vector2] = None
        self.shepherd_branch: Optional[str] = None
        self.junction_guard_anchor: Optional[pygame.Vector2] = None
        self.junction_guard_branch: Optional[str] = None
        self.junction_guard_hop = -1
        self.junction_guard_parent_id: Optional[int] = None
        self.junction_guard_layer = -1
        self.is_branch_leader = False
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
        self.current_junction_id: Optional[str] = None
        self.anchor_junction_id: Optional[str] = None
        self.known_junction_states: dict[str, dict] = {}
        self.comm_bridge_target: Optional[pygame.Vector2] = None
        self.comm_bridge_index = -1
        self.comm_bridge_branch: Optional[str] = None

        self.anchor_region_entry_time: Optional[float] = None
        self.anchor_region_entry_times: dict[str, float] = {}
        self.was_in_anchor_region = anchor_election_rect.collidepoint(x, y)
        self.was_in_anchor_regions: dict[str, bool] = {
            CURRENT_JUNCTION_ID: self.was_in_anchor_region
        }
        self.anchor_election_cost = float("inf")
        self.anchor_candidate_position: Optional[pygame.Vector2] = None
        self.anchor_cost_components = {
            "arrival": 0.0,
            "parking": 0.0,
            "obstruction": 0.0,
            "communication": 0.0,
        }

        self.comm_neighbors: list[object] = []
        self.connected_to_base = False
        self.comm_hop = -1
        self.comm_parent: Optional[object] = None
        self.comm_path_margin = float("-inf")
        self.received_branch = None
        self.received_command = None
        self.received_gate_states = None
        self.received_sequence = -1
        self.received_junction_id: Optional[str] = None
        self.received_state_sequence = -1
        self.received_junction_id: Optional[str] = None
        self.received_state_sequence = -1

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

        if self.role == "JUNCTION_GUARD" and self.junction_guard_anchor is not None:
            error = self.junction_guard_anchor - self.position
            step = JUNCTION_GUARD_MOVE_SPEED * dt
            next_position = (
                self.junction_guard_anchor.copy()
                if error.length() <= step
                else self.position + error.normalize() * step
            ) if error.length_squared() > EPSILON else self.position.copy()
            next_position = limit_communication_proposed_position(
                self,
                next_position,
                old_position,
            )
            if is_walkable(next_position, self.radius):
                self.position = next_position
            self.velocity.update(0.0, 0.0)
            self.acceleration.update(0.0, 0.0)
            self.filtered_acceleration.update(0.0, 0.0)
            self.previous_position = old_position
            self._record_motion()
            return

        if (
            self.role == "FRONTIER_SHEPHERD"
            and phase == SimulationPhase.EXPLORE_BRANCH
            and self.shepherd_anchor is not None
            and self.shepherd_branch == frontier_line_branch
        ):
            # The selected entrance guard remains the same physical group for
            # the whole branch.  It advances as a transverse line instead of
            # dissolving into the NORMAL SPH body and being re-elected later.
            target = shepherd_slot_position_at_depth(
                self.shepherd_anchor,
                self.shepherd_branch,
                frontier_line_depth,
            )
            # Local forward bumper/proximity probe.  A communication guard or
            # NORMAL pressure can stop the robot without making this probe
            # positive; only the physical map mask immediately ahead does.
            outward = BRANCH_DIRECTIONS[self.shepherd_branch]
            forward_probe = (
                self.position
                + outward * DEAD_END_FORWARD_BUMPER_PROBE
            )
            if not is_walkable(forward_probe, self.radius):
                self.last_forward_obstacle_contact_time = simulation_time
            error = target - self.position
            desired_velocity = pygame.Vector2()
            if error.length_squared() > EPSILON:
                desired_velocity = (
                    error.normalize() * FRONTIER_LINE_FORM_SPEED
                )
            self.commanded_velocity = desired_velocity.copy()
            step = min(error.length(), FRONTIER_LINE_FORM_SPEED * dt)
            next_position = (
                self.position + error.normalize() * step
                if step > 0.0 and error.length_squared() > EPSILON
                else self.position.copy()
            )
            next_position = limit_communication_proposed_position(
                self,
                next_position,
                old_position,
            )
            if is_walkable(next_position, self.radius):
                self.position = next_position
            self.observed_velocity = (
                self.position - old_position
            ) / max(dt, EPSILON)
            self.velocity = self.observed_velocity.copy()
            self.acceleration.update(0.0, 0.0)
            self.filtered_acceleration.update(0.0, 0.0)
            update_indirect_contact_state(self, dt)
            self.previous_position = old_position
            self._record_motion()
            return

        if self.role == "PRE_SHEPHERD" and self.shepherd_anchor is not None:
            error = self.shepherd_anchor - self.position
            step = SHEPHERD_FORM_SPEED * dt
            next_position = (
                self.shepherd_anchor.copy()
                if error.length() <= step
                else self.position + error.normalize() * step
            ) if error.length_squared() > EPSILON else self.position.copy()
            next_position = limit_communication_proposed_position(
                self,
                next_position,
                old_position,
            )
            if is_walkable(next_position, self.radius):
                self.position = next_position
            self.velocity.update(0.0, 0.0)
            self.acceleration.update(0.0, 0.0)
            self.filtered_acceleration.update(0.0, 0.0)
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
                # A selected Shepherd claims its local slot, but the proposed
                # step is still bounded by its current Base-side parent link.
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
                next_position = limit_communication_proposed_position(
                    self,
                    next_position,
                    old_position,
                )
                if is_walkable(next_position, self.radius):
                    self.position = next_position
                self.velocity.update(0.0, 0.0)
                self.acceleration.update(0.0, 0.0)
                self.filtered_acceleration.update(0.0, 0.0)
                self.previous_position = old_position
                self._record_motion()
                return

        self.velocity += self.acceleration * dt
        if self.role in {"NORMAL", "FRONTIER_SHEPHERD"}:
            region = get_robot_region(self.position)
            lateral_decay = math.exp(
                -CORRIDOR_LATERAL_VELOCITY_DAMPING * dt
            )
            if region in {"LEFT", "RIGHT"}:
                self.velocity.y *= lateral_decay
            elif region in {"UP", "BOTTOM"}:
                self.velocity.x *= lateral_decay
        speed_limit = MAX_SPEED
        if (
            phase == SimulationPhase.MOVE_TO_JUNCTION
            or initial_pressure_release_active()
        ):
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
        apply_communication_velocity_guard(self, dt)
        # Eguchi's command velocity is sampled before collision constraints;
        # observed velocity is reconstructed from the realized displacement.
        self.commanded_velocity = self.velocity.copy()
        x_position = pygame.Vector2(self.position.x + self.velocity.x * dt, self.position.y)
        if is_walkable(x_position, self.radius):
            self.position.x = x_position.x
        else:
            self.velocity.x = wall_collision_velocity(self.velocity.x)
        y_position = pygame.Vector2(self.position.x, self.position.y + self.velocity.y * dt)
        if is_walkable(y_position, self.radius):
            self.position.y = y_position.y
        else:
            self.velocity.y = wall_collision_velocity(self.velocity.y)

        constrain_final_return_gate_crossing(self, old_position)
        # Smooth virtual pressure is backed by a hard one-step guard so a fast
        # ordinary robot cannot pass through a temporary gap while Shepherds
        # are still moving laterally into their slots.
        constrain_normal_behind_shepherd_curtain(self)
        constrain_normal_behind_pre_shepherd_curtain(self)
        constrain_base_reserve_to_bottom(self)
        constrain_communication_parent_separation(self, old_position)
        self.observed_velocity = (
            self.position - old_position
        ) / max(dt, EPSILON)
        update_indirect_contact_state(self, dt)
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
        elif self.role in {"SHEPHERD", "PRE_SHEPHERD"}:
            color = SHEPHERD_COLOR
        elif self.role == "JUNCTION_GUARD":
            color = JUNCTION_GUARD_COLOR
        elif self.role == "FRONTIER_SHEPHERD":
            color = FRONTIER_SHEPHERD_COLOR
        elif self.comm_bridge_target is not None:
            color = COMM_BRIDGE_COLOR
        elif show_density_color:
            color = density_to_color(self.density, color_reference_density)
        else:
            color = ROBOT_BASE_COLOR

        if base_station is not None and self.role != "BASE" and not self.connected_to_base:
            color = DISCONNECTED_FILL_COLOR

        draw_radius = max(1, round(self.radius + 1))
        pygame.draw.circle(surface, color, (x, y), draw_radius)
        if self.role == "NORMAL" and not show_density_color:
            pygame.draw.circle(
                surface,
                ROBOT_OUTLINE_COLOR,
                (x, y),
                draw_radius,
                width=1,
            )

        if self.role in {"RELAY", "TRUNK_RELAY"}:
            ring_color = TRUNK_RELAY_COLOR if self.role == "TRUNK_RELAY" else RELAY_COLOR
            pygame.draw.circle(surface, ring_color, (x, y), draw_radius + 2, width=1)
        elif self.role == "ANCHOR":
            pygame.draw.circle(
                surface,
                ANCHOR_RING_COLOR,
                (x, y),
                draw_radius + 3,
                width=2,
            )
        elif self.role in {"SHEPHERD", "PRE_SHEPHERD"}:
            pygame.draw.circle(surface, SHEPHERD_COLOR, (x, y), draw_radius + 2, width=1)
        elif self.role in {"JUNCTION_GUARD", "FRONTIER_SHEPHERD"}:
            pygame.draw.circle(surface, color, (x, y), draw_radius + 2, width=2)
            if self.is_branch_leader:
                pygame.draw.circle(
                    surface,
                    BRANCH_LEADER_RING_COLOR,
                    (x, y),
                    draw_radius + 4,
                    width=2,
                )
        elif self.comm_bridge_target is not None:
            pygame.draw.circle(
                surface,
                COMM_BRIDGE_LINK_COLOR,
                (x, y),
                draw_radius + 3,
                width=2,
            )
        elif self.base_reserve:
            pygame.draw.circle(
                surface,
                BASE_COLOR,
                (x, y),
                draw_radius + 2,
                width=1,
            )


def linear_quantile(values: list[float], probability: float) -> float:
    """Small dependency-free quantile helper with linear interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = clamp(probability, 0.0, 1.0) * (len(ordered) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    blend = index - lower
    return ordered[lower] * (1.0 - blend) + ordered[upper] * blend


def inferred_contact_branch(robot: "Robot") -> Optional[str]:
    region = get_robot_region(robot.position)
    if region in BRANCHES:
        return region
    if phase in {
        SimulationPhase.EXPLORE_BRANCH,
        SimulationPhase.FORM_SHEPHERD_BOUNDARY,
        SimulationPhase.FILL_BEHIND_SHEPHERD,
    }:
        return active_branch
    return None


def record_collision_point(robot: "Robot") -> None:
    """Record/merge an Eguchi collision position without reading wall geometry."""
    branch = inferred_contact_branch(robot)
    position = robot.position.copy()
    for existing in reversed(collision_points):
        if (
            existing.branch == branch
            and position.distance_to(existing.position)
            <= CONTACT_POINT_MERGE_RADIUS
        ):
            existing.position = existing.position.lerp(position, 0.35)
            existing.detected_at = simulation_time
            existing.robot_id = robot.robot_id
            robot.latest_contact_point = existing.position.copy()
            return
    collision_points.append(
        CollisionPoint(
            position=position,
            detected_at=simulation_time,
            robot_id=robot.robot_id,
            branch=branch,
        )
    )
    robot.latest_contact_point = position.copy()
    metrics.contact_events.append({
        "time": simulation_time,
        "robot_id": robot.robot_id,
        "branch": branch or "UNASSIGNED",
        "x": position.x,
        "y": position.y,
    })


def update_indirect_contact_state(robot: "Robot", dt: float) -> None:
    """Integrate normalized command/observed speed discrepancy (Eguchi Eq. 11)."""
    robot.contact_detected = False
    if (
        robot.role not in {"NORMAL", "FRONTIER_SHEPHERD"}
        or phase not in {
            SimulationPhase.MOVE_TO_JUNCTION,
            SimulationPhase.EXPLORE_BRANCH,
            SimulationPhase.FORM_SHEPHERD_BOUNDARY,
            SimulationPhase.FILL_BEHIND_SHEPHERD,
        }
    ):
        robot.tracking_error_integral = max(
            0.0,
            robot.tracking_error_integral
            - CONTACT_ERROR_ATTENUATION * dt * FPS,
        )
        return

    command_speed = robot.commanded_velocity.length()
    observed_speed = robot.observed_velocity.length()
    if command_speed < CONTACT_COMMAND_MIN_SPEED:
        discrepancy = -CONTACT_ERROR_ATTENUATION
    else:
        discrepancy = (
            abs(command_speed - observed_speed)
            / max(MAX_SPEED, EPSILON)
            - CONTACT_ERROR_ATTENUATION
        )
    robot.tracking_error_integral = max(
        0.0,
        robot.tracking_error_integral + discrepancy * dt * FPS,
    )
    if (
        robot.tracking_error_integral >= CONTACT_ERROR_THRESHOLD
        and simulation_time - robot.last_contact_time
        >= CONTACT_EVENT_COOLDOWN
    ):
        robot.contact_detected = True
        robot.last_contact_time = simulation_time
        record_collision_point(robot)
        # Eguchi resets I after recording a collision position.
        robot.tracking_error_integral = 0.0


def robot_contact_evidence(robot: "Robot") -> float:
    integral_evidence = clamp(
        robot.tracking_error_integral
        / max(CONTACT_ERROR_THRESHOLD, EPSILON),
        0.0,
        1.0,
    )
    recent_event = (
        1.0
        if simulation_time - robot.last_contact_time <= CONTACT_EVENT_MEMORY
        else 0.0
    )
    return max(integral_evidence, recent_event)


def compute_contact_point_repulsion_force(robot: "Robot") -> pygame.Vector2:
    """Use observed collision poses as weak/strong virtual boundary particles."""
    if robot.role not in {"NORMAL", "FRONTIER_SHEPHERD"} or not collision_points:
        return pygame.Vector2()
    region = get_robot_region(robot.position)
    gain = (
        CONTACT_POINT_WEAK_GAIN
        if phase == SimulationPhase.MOVE_TO_JUNCTION
        else CONTACT_POINT_STRONG_GAIN
    )
    force = pygame.Vector2()
    for point in collision_points:
        if point.branch is not None and region in BRANCHES and point.branch != region:
            continue
        offset = robot.position - point.position
        distance_sq = offset.length_squared()
        if distance_sq <= EPSILON:
            continue
        distance = math.sqrt(distance_sq)
        if distance >= CONTACT_POINT_REPULSION_RADIUS:
            continue
        kernel = (1.0 - distance / CONTACT_POINT_REPULSION_RADIUS) ** 2
        force += offset.normalize() * gain * kernel / max(distance, 1.0)
    return limit_vector(force, CONTACT_POINT_FORCE_LIMIT)


def estimate_effective_branch_width(robots, branch: str) -> float:
    """Fuse traversing-robot span with indirect left/right contact samples."""
    direction = BRANCH_DIRECTIONS[branch]
    normal = pygame.Vector2(-direction.y, direction.x)
    flow_samples = [
        robot.position.dot(normal)
        for robot in robots
        if robot.role == "NORMAL"
        and get_robot_region(robot.position) == branch
        and branch_depth_from_junction(robot.position, branch)
        <= BRANCH_WIDTH_SAMPLE_DEPTH
    ]
    flow_width = 0.0
    if len(flow_samples) >= BRANCH_WIDTH_MIN_FLOW_SAMPLES:
        flow_width = (
            linear_quantile(flow_samples, 0.95)
            - linear_quantile(flow_samples, 0.05)
        )

    center_lateral = get_branch_entrance(branch).dot(normal)
    contact_lateral = [
        point.position.dot(normal) - center_lateral
        for point in collision_points
        if point.branch == branch
    ]
    left = [value for value in contact_lateral if value < 0.0]
    right = [value for value in contact_lateral if value > 0.0]
    contact_width = 0.0
    if (
        len(left) >= BRANCH_WIDTH_MIN_CONTACT_SAMPLES
        and len(right) >= BRANCH_WIDTH_MIN_CONTACT_SAMPLES
    ):
        contact_width = linear_quantile(right, 0.50) - linear_quantile(left, 0.50)

    if flow_width > 0.0 and contact_width > 0.0:
        estimate = (
            BRANCH_WIDTH_FLOW_WEIGHT * flow_width
            + BRANCH_WIDTH_CONTACT_WEIGHT * contact_width
        )
    else:
        estimate = max(flow_width, contact_width)
    if estimate > 0.0:
        effective_branch_widths[branch] = clamp(
            estimate,
            ROBOT_RADIUS * 2.0,
            corridor_width * 1.5,
        )
    return effective_branch_widths.get(branch, 0.0)


def observed_branch_contact_risk(branch: str) -> float:
    entrance = get_branch_entrance(branch)
    nearby_count = sum(
        point.branch == branch
        and point.position.distance_to(entrance)
        <= BRANCH_ENTRANCE_CONGESTION_RADIUS
        for point in collision_points
    )
    return clamp(nearby_count / 10.0, 0.0, 1.0)


def required_junction_guard_count(robots, branch: str) -> int:
    width = estimate_effective_branch_width(robots, branch)
    if width <= EPSILON:
        width = float(corridor_width)
    return int(clamp(
        math.ceil(width / max(JUNCTION_GUARD_COVERAGE, EPSILON)),
        JUNCTION_GUARD_MIN_COUNT,
        JUNCTION_GUARD_MAX_COUNT,
    ))


def build_junction_guard_slots(
    branch: str,
    count: int,
    frontier_depth: Optional[float] = None,
) -> list[pygame.Vector2]:
    direction = BRANCH_DIRECTIONS[branch]
    normal = pygame.Vector2(-direction.y, direction.x)
    width = effective_branch_widths.get(branch, 0.0)
    if width <= EPSILON:
        width = float(corridor_width)
    usable_half = min(
        half_width - ROBOT_RADIUS * 2.0,
        max(ROBOT_RADIUS * 2.0, width * 0.5 - ROBOT_RADIUS),
    )
    lateral_offsets = [0.0] if count <= 1 else [
        -usable_half + 2.0 * usable_half * index / (count - 1)
        for index in range(count)
    ]
    line_depth = clamp(
        (
            JUNCTION_GUARD_BRANCH_INSET
            if frontier_depth is None
            else frontier_depth
        ),
        JUNCTION_GUARD_BRANCH_INSET,
        BRANCH_LENGTHS[branch] - ROBOT_RADIUS * 1.1,
    )
    mouth_center = get_branch_entrance(branch) + direction * line_depth
    return [mouth_center + normal * offset for offset in lateral_offsets]


def observed_branch_frontier_depth(robots, branch: str) -> Optional[float]:
    """Return the outer edge of the actually observed branch cohort.

    No map-length-derived probe point is used.  The line is placed just beyond
    the deepest NORMAL robot that physically crossed this branch mouth, so no
    already-diffused robot is stranded on the far side of its Shepherd border.
    """
    depths = [
        branch_depth_from_junction(robot.position, branch)
        for robot in robots
        if robot.role == "NORMAL"
        and not robot.base_reserve
        and get_robot_region(robot.position) == branch
    ]
    if not depths:
        return None
    return clamp(
        max(depths) + JUNCTION_GUARD_FRONTIER_MARGIN,
        JUNCTION_GUARD_BRANCH_INSET,
        BRANCH_LENGTHS[branch] - ROBOT_RADIUS * 1.1,
    )


def outward_branch_neighbor_count(robot: "Robot", branch: str) -> int:
    """Count locally connected peers that are measurably farther outward."""
    own_depth = branch_depth_from_junction(robot.position, branch)
    return sum(
        get_robot_region(peer.position) == branch
        and branch_depth_from_junction(peer.position, branch)
        > own_depth + JUNCTION_GUARD_TERMINAL_DEPTH_EPSILON
        for peer in robot.comm_neighbors
    )


def select_branch_guard_leader(
    robots,
    branch: str,
    available_ids: set[int],
) -> Optional["Robot"]:
    direction = BRANCH_DIRECTIONS[branch]
    normal = pygame.Vector2(-direction.y, direction.x)
    candidates = [
        robot
        for robot in robots
        if robot.robot_id in available_ids
        and robot.role == "NORMAL"
        and get_robot_region(robot.position) == branch
    ]
    if not candidates:
        return None

    # In the local communication graph a frontier terminal has no same-branch
    # neighbour at a larger outward depth.  Select the deepest such terminal;
    # global branch geometry is used only to express the robot's own odometry.
    terminal_candidates = [
        robot
        for robot in candidates
        if outward_branch_neighbor_count(robot, branch) == 0
    ]
    if terminal_candidates:
        candidates = terminal_candidates
    entrance = get_branch_entrance(branch)
    return min(
        candidates,
        key=lambda robot: (
            -branch_depth_from_junction(robot.position, branch),
            abs((robot.position - entrance).dot(normal)),
            -len(robot.comm_neighbors),
            robot.robot_id,
        ),
    )


def minimum_k_hop_guard_group(
    leader: "Robot",
    robots,
    available_ids: set[int],
    required_count: int,
    branch: str,
) -> tuple[list["Robot"], int]:
    by_id = {robot.robot_id: robot for robot in robots}
    selected_ids = {leader.robot_id}
    frontier_ids = {leader.robot_id}
    selected_hop = 0
    for hop in range(1, JUNCTION_GUARD_MAX_HOPS + 1):
        next_frontier: set[int] = set()
        for robot_id in frontier_ids:
            robot = by_id[robot_id]
            for peer in robot.comm_neighbors:
                peer_id = getattr(peer, "robot_id", -1)
                if (
                    peer_id in available_ids
                    and peer_id not in selected_ids
                    and getattr(peer, "role", None) == "NORMAL"
                    and get_robot_region(peer.position) == branch
                ):
                    next_frontier.add(peer_id)
        selected_ids.update(next_frontier)
        frontier_ids = next_frontier
        selected_hop = hop
        if len(selected_ids) >= required_count or not frontier_ids:
            break
    candidates = [by_id[robot_id] for robot_id in selected_ids]
    if len(candidates) < required_count:
        candidates = sorted(
            (
                robot
                for robot in robots
                if robot.robot_id in available_ids and robot.role == "NORMAL"
                and get_robot_region(robot.position) == branch
            ),
            key=lambda robot: (
                leader.position.distance_squared_to(robot.position),
                robot.robot_id,
            ),
        )[:required_count]
        selected_hop = JUNCTION_GUARD_MAX_HOPS + 1
    else:
        candidates.sort(
            key=lambda robot: (
                leader.position.distance_squared_to(robot.position),
                robot.robot_id,
            )
        )
        candidates = candidates[:required_count]
    return candidates, selected_hop


def required_thick_mouth_guard_layers(
    robots,
    branch: str,
    column_count: int,
) -> int:
    """Choose 2--4 physical layers from locally arriving pressure mass."""
    entrance = get_branch_entrance(branch)
    nearby_normals = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and not robot.base_reserve
        and get_robot_region(robot.position) in {"JUNCTION", branch}
        and robot.position.distance_to(entrance)
        <= JUNCTION_GUARD_RECRUIT_RADIUS
    ]
    mean_density_ratio = (
        sum(robot.density_ratio for robot in nearby_normals)
        / max(len(nearby_normals), 1)
    )
    layers = THICK_MOUTH_GUARD_MIN_LAYERS
    if (
        len(robots) >= THICK_MOUTH_GUARD_LARGE_SWARM_SIZE
        or len(nearby_normals) >= column_count * 3
        or mean_density_ratio >= 1.15
    ):
        layers += 1
    if (
        len(robots) >= THICK_MOUTH_GUARD_VERY_LARGE_SWARM_SIZE
        or (
            len(nearby_normals) >= column_count * 6
            and mean_density_ratio >= 1.30
        )
    ):
        layers += 1
    return int(clamp(
        layers,
        THICK_MOUTH_GUARD_MIN_LAYERS,
        THICK_MOUTH_GUARD_MAX_LAYERS,
    ))


def expand_k_hop_mouth_guard_group(
    seed_guards,
    robots,
    branch: str,
    target_count: int,
):
    """Grow an existing frontier guard into a bounded K-hop mouth wall."""
    if not seed_guards:
        return []
    entrance = get_branch_entrance(branch)
    selected = {robot.robot_id: robot for robot in seed_guards}
    leader = next(
        (robot for robot in seed_guards if robot.is_branch_leader),
        seed_guards[0],
    )
    leader.junction_guard_parent_id = None
    leader.junction_guard_hop = 0
    for robot in seed_guards:
        if robot is leader:
            continue
        robot.junction_guard_parent_id = leader.robot_id
        robot.junction_guard_hop = max(1, robot.junction_guard_hop)

    frontier = list(seed_guards)
    for hop in range(1, JUNCTION_GUARD_MAX_HOPS + 1):
        if len(selected) >= target_count or not frontier:
            break
        candidate_parents: dict[int, tuple["Robot", "Robot"]] = {}
        for parent in frontier:
            for peer in parent.comm_neighbors:
                peer_id = getattr(peer, "robot_id", -1)
                if peer_id < 0 or peer_id in selected:
                    continue
                if (
                    getattr(peer, "role", None) != "NORMAL"
                    or getattr(peer, "base_reserve", False)
                ):
                    continue
                region = get_robot_region(peer.position)
                if not (
                    region == branch
                    or (
                        region == "JUNCTION"
                        and peer.position.distance_to(entrance)
                        <= JUNCTION_GUARD_RECRUIT_RADIUS
                    )
                ):
                    continue
                previous = candidate_parents.get(peer_id)
                if (
                    previous is None
                    or parent.position.distance_squared_to(peer.position)
                    < previous[0].position.distance_squared_to(peer.position)
                ):
                    candidate_parents[peer_id] = (parent, peer)
        ordered = sorted(
            candidate_parents.values(),
            key=lambda item: (
                0 if get_robot_region(item[1].position) == branch else 1,
                item[1].position.distance_squared_to(entrance),
                item[1].robot_id,
            ),
        )
        next_frontier = []
        for parent, peer in ordered[:max(0, target_count - len(selected))]:
            peer.junction_guard_parent_id = parent.robot_id
            peer.junction_guard_hop = min(
                JUNCTION_GUARD_MAX_HOPS,
                max(0, parent.junction_guard_hop) + 1,
            )
            selected[peer.robot_id] = peer
            next_frontier.append(peer)
        frontier = next_frontier

    return list(selected.values())


def build_thick_mouth_guard_slots(
    branch: str,
    count: int,
    column_count: int,
) -> list[pygame.Vector2]:
    """Create full-width axial rows entirely on the Branch side of a mouth."""
    slots = []
    remaining = count
    layer = 0
    while remaining > 0:
        row_count = min(column_count, remaining)
        depth = (
            JUNCTION_GUARD_BRANCH_INSET
            + layer * THICK_MOUTH_GUARD_LAYER_SPACING
        )
        slots.extend(build_junction_guard_slots(branch, row_count, depth))
        remaining -= row_count
        layer += 1
    return slots


def thick_mouth_guards_formed(robots, selected_branch: str) -> bool:
    protected_branches = [
        branch
        for branch in junction_guard_groups
        if branch != selected_branch
        and branch_states.get(branch) == "UNVISITED"
    ]
    if not protected_branches:
        return True
    for branch in protected_branches:
        guards = [
            robot
            for robot in robots
            if robot.role == "JUNCTION_GUARD"
            and robot.junction_guard_branch == branch
        ]
        if (
            not guards
            or thick_mouth_guard_layers.get(branch, 0)
            < THICK_MOUTH_GUARD_MIN_LAYERS
            or any(
                robot.junction_guard_anchor is None
                or robot.position.distance_to(robot.junction_guard_anchor)
                > JUNCTION_GUARD_POSITION_TOLERANCE
                for robot in guards
            )
        ):
            return False
    return True


def release_junction_guard_roles(robots) -> None:
    global frontier_line_branch, frontier_line_depth
    for robot in robots:
        if robot.role not in {"JUNCTION_GUARD", "FRONTIER_SHEPHERD"}:
            continue
        robot.role = "NORMAL"
        robot.junction_guard_anchor = None
        robot.junction_guard_branch = None
        robot.junction_guard_hop = -1
        robot.junction_guard_parent_id = None
        robot.junction_guard_layer = -1
        robot.is_branch_leader = False
        robot.shepherd_anchor = None
        robot.shepherd_origin = None
        robot.shepherd_branch = None
        robot.velocity.update(0.0, 0.0)
        robot.acceleration.update(0.0, 0.0)
        robot.filtered_acceleration.update(0.0, 0.0)
    frontier_line_branch = None
    frontier_line_depth = 0.0


def begin_junction_guard_formation(robots) -> None:
    """Form a full border at every observed branch cohort's outer frontier."""
    global junction_guard_groups, junction_guard_formation_timer
    global junction_guard_frontier_depths
    global junction_guard_stable_dwell, junction_guard_status
    global distributed_consensus_branch
    global pending_branch_start
    release_junction_guard_roles(robots)
    # Only observed openings may be sealed.  Closing an undiscovered branch
    # would leak the simulator's ground-truth map into the controller.
    branch_gate_states.clear()
    branch_gate_states.update({
        branch: (
            "CLOSED" if branch in detected_branch_candidates else "OPEN"
        )
        for branch in BRANCHES
    })
    preserve_consensus_at_anchor(junction_anchor)
    print(
        "[Gate] inferred mouths only: "
        + ", ".join(
            f"{branch}={branch_gate_states[branch]}" for branch in BRANCHES
        )
    )
    distributed_consensus_branch = None
    junction_guard_groups = {}
    junction_guard_frontier_depths = {}
    junction_guard_formation_timer = 0.0
    junction_guard_stable_dwell = 0.0
    pending_branch_start = None
    for branch in BRANCHES:
        thick_mouth_guard_layers[branch] = 0
        thick_mouth_guard_columns[branch] = 0
    available_ids = {
        robot.robot_id
        for robot in robots
        if robot.role == "NORMAL" and not robot.base_reserve
    }
    unavailable = []
    for branch in FIXED_BRANCH_ORDER:
        if (
            branch not in detected_branch_candidates
            or branch_states.get(branch) != "UNVISITED"
        ):
            continue
        required_count = required_junction_guard_count(robots, branch)
        frontier_depth = observed_branch_frontier_depth(robots, branch)
        if frontier_depth is None:
            unavailable.append(branch)
            continue
        leader = select_branch_guard_leader(robots, branch, available_ids)
        if leader is None:
            unavailable.append(branch)
            continue
        candidates, selected_hop = minimum_k_hop_guard_group(
            leader,
            robots,
            available_ids,
            required_count,
            branch,
        )
        slots = build_junction_guard_slots(
            branch,
            len(candidates),
            frontier_depth,
        )
        assignment = assign_shepherd_slots(candidates, slots)
        if len(assignment) < JUNCTION_GUARD_MIN_COUNT:
            unavailable.append(branch)
            continue
        branch_ids = []
        for robot, slot, _ in assignment:
            robot.role = "JUNCTION_GUARD"
            robot.junction_guard_anchor = slot.copy()
            robot.junction_guard_branch = branch
            robot.junction_guard_hop = selected_hop
            robot.junction_guard_parent_id = (
                None if robot is leader else leader.robot_id
            )
            robot.junction_guard_layer = 0
            robot.is_branch_leader = robot is leader
            robot.velocity.update(0.0, 0.0)
            robot.filtered_acceleration.update(0.0, 0.0)
            branch_ids.append(robot.robot_id)
            available_ids.discard(robot.robot_id)
        junction_guard_groups[branch] = branch_ids
        junction_guard_frontier_depths[branch] = frontier_depth
        print(
            f"[Branch Frontier Guard] branch={branch}, "
            f"depth={frontier_depth:.1f}, full={len(branch_ids)}, "
            f"terminal_leader={leader.robot_id}, "
            f"outward_degree={outward_branch_neighbor_count(leader, branch)}, "
            f"k={selected_hop}"
        )
    junction_guard_status = (
        "FULL_GUARD_UNAVAILABLE:" + ",".join(unavailable)
        if unavailable
        else "FORMING_FULL_GUARDS"
    )


def junction_guards_formed(robots) -> bool:
    guards = [robot for robot in robots if robot.role == "JUNCTION_GUARD"]
    if not guards or any(
        branch_states.get(branch) == "UNVISITED"
        and branch in detected_branch_candidates
        and branch not in junction_guard_groups
        for branch in BRANCHES
    ):
        return False
    return all(
        robot.junction_guard_anchor is not None
        and robot.position.distance_to(robot.junction_guard_anchor)
        <= JUNCTION_GUARD_POSITION_TOLERANCE
        for robot in guards
    )


def commit_junction_guard_roles(robots, selected_branch: str) -> None:
    """Advance the selected line and thicken every unselected mouth guard."""
    global junction_guard_status, frontier_line_branch, frontier_line_depth
    for branch, robot_ids in junction_guard_groups.items():
        branch_guards = [
            robot for robot in robots if robot.robot_id in robot_ids
        ]
        if branch == selected_branch:
            line_slots = build_shepherd_slots(branch, len(branch_guards))
            assignment = assign_shepherd_slots(branch_guards, line_slots)
            for robot, slot, _ in assignment:
                robot.role = "FRONTIER_SHEPHERD"
                robot.junction_guard_anchor = None
                robot.junction_guard_branch = branch
                robot.junction_guard_parent_id = None
                robot.junction_guard_layer = 0
                robot.shepherd_anchor = slot.copy()
                robot.shepherd_origin = slot.copy()
                robot.shepherd_branch = branch
                robot.velocity.update(0.0, 0.0)
                robot.filtered_acceleration.update(0.0, 0.0)
            frontier_line_branch = branch
            frontier_line_depth = junction_guard_frontier_depths.get(
                branch,
                FRONTIER_LINE_START_DEPTH,
            )
            thick_mouth_guard_layers[branch] = 0
            thick_mouth_guard_columns[branch] = 0
            continue
        # The original frontier line becomes the seed of a bounded K-hop
        # cohort.  Extra NORMAL peers form axial rows behind the full-width
        # mouth line, so high SPH pressure is resisted by physical depth and
        # redundant communication rather than by a virtual geofence.
        column_count = max(
            JUNCTION_GUARD_MIN_COUNT,
            len(branch_guards),
        )
        desired_layers = required_thick_mouth_guard_layers(
            robots,
            branch,
            column_count,
        )
        target_count = column_count * desired_layers
        expanded_guards = expand_k_hop_mouth_guard_group(
            branch_guards,
            robots,
            branch,
            target_count,
        )
        actual_layers = math.ceil(
            len(expanded_guards) / max(column_count, 1)
        )
        mouth_slots = build_thick_mouth_guard_slots(
            branch,
            len(expanded_guards),
            column_count,
        )
        assignment = assign_shepherd_slots(expanded_guards, mouth_slots)
        leader = next(
            (robot for robot in branch_guards if robot.is_branch_leader),
            branch_guards[0],
        )
        branch_ids = []
        for robot, slot, _ in assignment:
            robot.role = "JUNCTION_GUARD"
            robot.junction_guard_anchor = slot.copy()
            robot.junction_guard_branch = branch
            robot.junction_guard_layer = int(round(
                (
                    branch_depth_from_junction(slot, branch)
                    - JUNCTION_GUARD_BRANCH_INSET
                )
                / max(THICK_MOUTH_GUARD_LAYER_SPACING, EPSILON)
            ))
            robot.is_branch_leader = robot is leader
            robot.shepherd_anchor = None
            robot.shepherd_origin = None
            robot.shepherd_branch = None
            robot.velocity.update(0.0, 0.0)
            robot.acceleration.update(0.0, 0.0)
            robot.filtered_acceleration.update(0.0, 0.0)
            branch_ids.append(robot.robot_id)
        junction_guard_groups[branch] = branch_ids
        thick_mouth_guard_columns[branch] = column_count
        thick_mouth_guard_layers[branch] = actual_layers
        print(
            f"[Thick Mouth Guard] branch={branch}, "
            f"columns={column_count}, layers={actual_layers}/"
            f"{desired_layers}, robots={len(branch_ids)}/{target_count}, "
            f"k<={JUNCTION_GUARD_MAX_HOPS}"
        )
    junction_guard_status = (
        f"FRONTIER={selected_branch};OTHERS=FORMING_THICK_KHOP_WALLS"
    )
    print(
        f"[Junction Guard] selected={selected_branch} -> persistent "
        f"FRONTIER_SHEPHERD line; ids="
        f"{[robot.robot_id for robot in get_frontier_shepherds(robots, selected_branch)]}; "
        f"start-depth={frontier_line_depth:.1f}; "
        "non-selected=recruited-as-thick-khop-mouth-walls"
    )


def get_frontier_shepherds(robots, branch: Optional[str] = None):
    return [
        robot
        for robot in robots
        if robot.role == "FRONTIER_SHEPHERD"
        and (branch is None or robot.shepherd_branch == branch)
    ]


def update_frontier_line_progress(robots, branch: str, dt: float) -> None:
    """Advance the intact line only as fast as the NORMAL body can follow."""
    global frontier_line_depth
    frontiers = get_frontier_shepherds(robots, branch)
    if not frontiers or frontier_line_branch != branch:
        return
    normal_depths = [
        branch_depth_from_junction(robot.position, branch)
        for robot in robots
        if robot.role == "NORMAL"
        and not robot.base_reserve
        and get_robot_region(robot.position) == branch
    ]
    # At the mouth there may be no NORMAL center inside the branch yet because
    # the line itself is the leading boundary.  Treat the Junction mouth as
    # zero-depth support so the line opens exactly one lead gap; after that it
    # can advance only when the NORMAL body follows.
    supported_front = (
        linear_quantile(
            normal_depths,
            FRONTIER_LINE_SUPPORT_QUANTILE,
        )
        if normal_depths
        else 0.0
    )
    supported_target = supported_front + FRONTIER_LINE_LEAD_GAP
    observed_limit = observed_dead_end_depths.get(branch)
    if observed_limit is not None:
        supported_target = min(supported_target, observed_limit)
    desired_depth = max(frontier_line_depth, supported_target)
    frontier_line_depth = min(
        desired_depth,
        frontier_line_depth + FRONTIER_LINE_ADVANCE_SPEED * dt,
    )


def promote_existing_frontier_line(
    robots,
    branch: str,
    observed_boundary_depth: Optional[float] = None,
):
    """Flatten the same moving frontier IDs into the return piston line."""
    global frontier_line_branch, frontier_line_depth
    frontiers = get_frontier_shepherds(robots, branch)
    if not frontiers:
        return []
    slots = build_shepherd_slots(
        branch,
        len(frontiers),
        observed_boundary_depth,
    )
    assignment = assign_shepherd_slots(frontiers, slots)
    if len(assignment) != len(frontiers):
        return []
    promoted = []
    for robot, slot, _ in assignment:
        robot.role = "SHEPHERD"
        robot.shepherd_anchor = slot.copy()
        robot.shepherd_origin = slot.copy()
        robot.shepherd_branch = branch
        robot.junction_guard_anchor = None
        robot.velocity.update(0.0, 0.0)
        robot.acceleration.update(0.0, 0.0)
        robot.filtered_acceleration.update(0.0, 0.0)
        promoted.append(robot)
    frontier_line_branch = None
    frontier_line_depth = 0.0
    print(
        f"[Frontier -> Shepherd] retained original IDs="
        f"{[robot.robot_id for robot in promoted]}; no re-election"
    )
    return promoted

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


def get_backtrack_bridge_slots(branch: str) -> list[pygame.Vector2]:
    """Return parallel Junction slots linking the Base side to one Branch."""
    outgoing = BRANCH_DIRECTIONS.get(
        branch,
        pygame.Vector2(1.0, 0.0),
    )
    incoming = pygame.Vector2(0.0, 1.0)
    diagonal = incoming + outgoing
    if diagonal.length_squared() <= EPSILON:
        diagonal = outgoing.copy()
    diagonal = diagonal.normalize()
    lateral = pygame.Vector2(-diagonal.y, diagonal.x)
    center = pygame.Vector2(junction_rect.center)
    slot_center = center + diagonal * BACKTRACK_BRIDGE_CENTER_OFFSET
    offsets = (
        -1.5,
        -0.5,
        0.5,
        1.5,
    )
    return [
        slot_center
        + lateral * BACKTRACK_BRIDGE_SLOT_SPREAD * offset
        for offset in offsets[:BACKTRACK_BRIDGE_GUARD_COUNT]
    ]


def clear_backtrack_bridge_guards(robots) -> None:
    for robot in robots:
        if robot.comm_bridge_target is None:
            continue
        robot.comm_bridge_target = None
        robot.comm_bridge_index = -1
        robot.comm_bridge_branch = None


def assess_backtrack_bridge_demand(robots) -> dict[str, object]:
    """Estimate how many temporary guards the natural mesh currently needs."""
    active_guards = {
        robot.robot_id
        for robot in robots
        if robot.comm_bridge_target is not None
    }
    branch_robots = [
        robot
        for robot in robots
        if get_robot_region(robot.position) == active_branch
    ]
    if not branch_robots:
        return {
            "required": 0,
            "risk": "STABLE",
            "redundancy": BACKTRACK_BRIDGE_TARGET_REDUNDANCY,
            "margin": COMM_RANGE,
            "connected_ratio": 1.0,
        }

    natural_pairs = set()
    base_interface_nodes = set()
    branch_interface_nodes = set()
    interface_margins = []
    for robot in robots:
        if robot.robot_id in active_guards:
            continue
        robot_region = get_robot_region(robot.position)
        for neighbor in robot.comm_neighbors:
            neighbor_id = getattr(neighbor, "robot_id", -1)
            if neighbor_id < 0 or neighbor_id in active_guards:
                continue
            pair = tuple(sorted((robot.robot_id, neighbor_id)))
            if pair in natural_pairs:
                continue
            natural_pairs.add(pair)
            neighbor_region = get_robot_region(neighbor.position)
            regions = {robot_region, neighbor_region}
            junction_robot_id = (
                robot.robot_id
                if robot_region == "JUNCTION"
                else neighbor_id
                if neighbor_region == "JUNCTION"
                else None
            )
            if junction_robot_id is None:
                continue
            if regions == {"BOTTOM", "JUNCTION"}:
                base_interface_nodes.add(junction_robot_id)
                interface_margins.append(
                    COMM_RANGE
                    - robot.position.distance_to(neighbor.position)
                )
            elif regions == {active_branch, "JUNCTION"}:
                branch_interface_nodes.add(junction_robot_id)
                interface_margins.append(
                    COMM_RANGE
                    - robot.position.distance_to(neighbor.position)
                )

    redundancy = min(
        len(base_interface_nodes),
        len(branch_interface_nodes),
    )
    if interface_margins:
        ordered_margins = sorted(interface_margins)
        robust_index = min(
            len(ordered_margins) - 1,
            int(0.20 * len(ordered_margins)),
        )
        robust_margin = ordered_margins[robust_index]
    else:
        robust_margin = -COMM_RANGE

    overall_connected_ratio = sum(
        robot.connected_to_base for robot in robots
    ) / max(len(robots), 1)
    branch_connected_ratio = sum(
        robot.connected_to_base for robot in branch_robots
    ) / max(len(branch_robots), 1)
    connected_ratio = min(
        overall_connected_ratio,
        branch_connected_ratio,
    )
    link_deficit = max(
        0,
        BACKTRACK_BRIDGE_TARGET_REDUNDANCY - redundancy,
    )
    disconnected_branch_count = sum(
        not robot.connected_to_base for robot in branch_robots
    )
    disconnected_demand = math.ceil(
        disconnected_branch_count
        / max(BACKTRACK_BRIDGE_EXTRA_LINKS_PER_SIDE, 1)
    )

    if (
        connected_ratio < 0.90
        or redundancy == 0
        or robust_margin <= BACKTRACK_BRIDGE_DANGER_MARGIN
    ):
        risk = "DANGER"
        required = max(2, link_deficit, disconnected_demand)
        if connected_ratio < 0.90:
            required = BACKTRACK_BRIDGE_GUARD_COUNT
    elif (
        connected_ratio < 1.0
        or redundancy < BACKTRACK_BRIDGE_TARGET_REDUNDANCY
        or robust_margin < BACKTRACK_BRIDGE_STABLE_MARGIN
    ):
        risk = "CAUTION"
        required = max(1, link_deficit, disconnected_demand)
        required = min(required, 2)
    else:
        risk = "STABLE"
        required = 0

    return {
        "required": int(clamp(
            required,
            0,
            BACKTRACK_BRIDGE_GUARD_COUNT,
        )),
        "risk": risk,
        "redundancy": redundancy,
        "margin": robust_margin,
        "connected_ratio": connected_ratio,
    }


def update_backtrack_bridge_guards(robots, dt) -> None:
    """Adaptively hold only the number of bridge guards required by risk."""
    global backtrack_bridge_required_count
    global backtrack_bridge_candidate_count
    global backtrack_bridge_candidate_dwell
    global backtrack_bridge_risk_level
    global backtrack_bridge_natural_redundancy
    global backtrack_bridge_natural_margin

    if phase != SimulationPhase.FLOW_BACKTRACK:
        clear_backtrack_bridge_guards(robots)
        backtrack_bridge_required_count = 0
        backtrack_bridge_candidate_count = 0
        backtrack_bridge_candidate_dwell = 0.0
        backtrack_bridge_risk_level = "STABLE"
        backtrack_bridge_natural_redundancy = 0
        backtrack_bridge_natural_margin = COMM_RANGE
        return

    assessment = assess_backtrack_bridge_demand(robots)
    desired_count = assessment["required"]
    backtrack_bridge_risk_level = assessment["risk"]
    backtrack_bridge_natural_redundancy = assessment["redundancy"]
    backtrack_bridge_natural_margin = assessment["margin"]

    if desired_count != backtrack_bridge_candidate_count:
        backtrack_bridge_candidate_count = desired_count
        backtrack_bridge_candidate_dwell = 0.0
    else:
        backtrack_bridge_candidate_dwell += dt

    dwell_required = (
        BACKTRACK_BRIDGE_DEPLOY_DWELL
        if desired_count > backtrack_bridge_required_count
        else BACKTRACK_BRIDGE_RELEASE_DWELL
    )
    if (
        desired_count != backtrack_bridge_required_count
        and backtrack_bridge_candidate_dwell >= dwell_required
    ):
        backtrack_bridge_required_count = desired_count
        backtrack_bridge_candidate_dwell = 0.0
        print(
            "[Backtrack Bridge] "
            f"risk={backtrack_bridge_risk_level}, "
            f"required={backtrack_bridge_required_count}, "
            f"natural_redundancy="
            f"{backtrack_bridge_natural_redundancy}, "
            f"margin={backtrack_bridge_natural_margin:.1f}"
        )

    slots = get_backtrack_bridge_slots(active_branch)
    existing_by_index = {}
    for robot in robots:
        if robot.comm_bridge_target is None:
            continue
        if (
            robot.role != "NORMAL"
            or robot.comm_bridge_branch != active_branch
            or robot.comm_bridge_index >= len(slots)
            or robot.comm_bridge_index
            >= backtrack_bridge_required_count
        ):
            robot.comm_bridge_target = None
            robot.comm_bridge_index = -1
            robot.comm_bridge_branch = None
            continue
        robot.comm_bridge_target = slots[robot.comm_bridge_index].copy()
        existing_by_index[robot.comm_bridge_index] = robot

    for slot_index, slot in enumerate(
        slots[:backtrack_bridge_required_count]
    ):
        if slot_index in existing_by_index:
            continue
        candidates = [
            robot
            for robot in robots
            if (
                robot.role == "NORMAL"
                and robot.comm_bridge_target is None
                and not robot.base_reserve
                and robot.transfer_target in {None, transfer_branch}
                and get_robot_region(robot.position) in {"BOTTOM", "JUNCTION"}
                and robot.position.distance_to(slot)
                <= BACKTRACK_BRIDGE_RECRUIT_RADIUS
            )
        ]
        if not candidates:
            continue
        candidate = min(
            candidates,
            key=lambda robot: (
                0
                if get_robot_region(robot.position) == "BOTTOM"
                else 1,
                robot.position.distance_squared_to(slot),
                robot.robot_id,
            ),
        )
        candidate.comm_bridge_target = slot.copy()
        candidate.comm_bridge_index = slot_index
        candidate.comm_bridge_branch = active_branch


def compute_backtrack_bridge_force(robot) -> pygame.Vector2:
    target = robot.comm_bridge_target
    if target is None or phase != SimulationPhase.FLOW_BACKTRACK:
        return pygame.Vector2()
    force = (
        (target - robot.position) * BACKTRACK_BRIDGE_POSITION_GAIN
        - robot.velocity * BACKTRACK_BRIDGE_DAMPING_GAIN
    )
    if force.length_squared() > BACKTRACK_BRIDGE_FORCE_LIMIT**2:
        force.scale_to_length(BACKTRACK_BRIDGE_FORCE_LIMIT)
    return force


def add_redundant_backtrack_bridge_links(
    robots,
    linked_pairs: set[tuple[int, int]],
) -> None:
    """Add explicit Base-side and Branch-side links for every bridge guard."""
    global communication_redundant_links
    communication_redundant_links = []
    if phase != SimulationPhase.FLOW_BACKTRACK:
        return

    bridges = sorted(
        (
            robot
            for robot in robots
            if robot.comm_bridge_target is not None
            and robot.comm_bridge_branch == active_branch
        ),
        key=lambda robot: robot.comm_bridge_index,
    )
    recorded_pairs = set()
    for bridge in bridges:
        for side_region in ("BOTTOM", active_branch):
            candidates = []
            for other in robots:
                if other is bridge or get_robot_region(other.position) != side_region:
                    continue
                distance_sq = bridge.position.distance_squared_to(other.position)
                if distance_sq > COMM_RANGE**2:
                    continue
                if not has_line_of_sight(bridge.position, other.position):
                    continue
                candidates.append((distance_sq, other.robot_id, other))
            candidates.sort(key=lambda item: (item[0], item[1]))
            for _, _, other in candidates[
                :BACKTRACK_BRIDGE_EXTRA_LINKS_PER_SIDE
            ]:
                pair = tuple(sorted((bridge.robot_id, other.robot_id)))
                if pair not in linked_pairs:
                    linked_pairs.add(pair)
                    bridge.comm_neighbors.append(other)
                    other.comm_neighbors.append(bridge)
                if pair not in recorded_pairs:
                    recorded_pairs.add(pair)
                    communication_redundant_links.append((bridge, other))


def get_backtrack_bridge_stats(robots) -> dict[str, object]:
    guards = [
        robot
        for robot in robots
        if robot.comm_bridge_target is not None
    ]
    ready = 0
    base_links = 0
    branch_links = 0
    settled = 0
    for guard in guards:
        base_count = sum(
            get_robot_region(neighbor.position) == "BOTTOM"
            for neighbor in guard.comm_neighbors
            if getattr(neighbor, "role", None) != "BASE"
        )
        branch_count = sum(
            get_robot_region(neighbor.position) == active_branch
            for neighbor in guard.comm_neighbors
            if getattr(neighbor, "role", None) != "BASE"
        )
        base_links += base_count
        branch_links += branch_count
        ready += base_count > 0 and branch_count > 0
        settled += (
            guard.position.distance_to(guard.comm_bridge_target)
            <= BACKTRACK_BRIDGE_POSITION_TOLERANCE
        )
    return {
        "guards": len(guards),
        "required": backtrack_bridge_required_count,
        "risk": backtrack_bridge_risk_level,
        "natural_redundancy": backtrack_bridge_natural_redundancy,
        "natural_margin": backtrack_bridge_natural_margin,
        "ready": ready,
        "settled": settled,
        "base_links": base_links,
        "branch_links": branch_links,
    }


def update_communication_neighbors(robots, grid):
    """Build a sparse local mesh and explicit Base links with LOS checks."""
    range_sq = COMM_RANGE**2
    for robot in robots:
        robot.comm_neighbors = []
    if base_station is None:
        return
    base_station.comm_neighbors = []

    linked_pairs = set()
    for robot in robots:
        candidates = []
        for other in iter_neighbor_candidates(robot, grid):
            if other is robot:
                continue
            distance_sq = robot.position.distance_squared_to(other.position)
            if distance_sq > range_sq:
                continue
            candidates.append((distance_sq, other.robot_id, other))
        candidates.sort(key=lambda item: (item[0], item[1]))
        for _, _, other in candidates[:COMM_MAX_LOCAL_NEIGHBORS]:
            pair = tuple(sorted((robot.robot_id, other.robot_id)))
            if pair in linked_pairs:
                continue
            if not has_line_of_sight(robot.position, other.position):
                continue
            linked_pairs.add(pair)
            robot.comm_neighbors.append(other)
            other.comm_neighbors.append(robot)

    for robot in robots:
        if base_station.position.distance_squared_to(robot.position) > range_sq:
            continue
        if not has_line_of_sight(base_station.position, robot.position):
            continue
        base_station.comm_neighbors.append(robot)
        robot.comm_neighbors.append(base_station)

    add_redundant_backtrack_bridge_links(robots, linked_pairs)


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


def get_anchor_message(
    anchor,
    junction_id: str = CURRENT_JUNCTION_ID,
):
    junction_state = get_junction_state(junction_id)
    if anchor is None:
        return (
            junction_id,
            junction_state.state_sequence,
            distributed_consensus_branch,
            (
                f"DISTRIBUTED_{phase.name}"
                if distributed_consensus_branch is not None
                else "DISTRIBUTED_VOTING"
            ),
            branch_gate_states.copy(),
        )
    if not anchor.connected_to_base:
        return (
            junction_id,
            junction_state.state_sequence,
            None,
            "WAIT_FOR_BASE_LINK",
            None,
        )
    return (
        anchor.anchor_junction_id or junction_id,
        junction_state.state_sequence,
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
        robot.received_junction_id = None
        robot.received_state_sequence = -1

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

    (
        message_junction_id,
        state_sequence,
        selected_branch,
        command,
        gate_states,
    ) = get_anchor_message(anchor)
    gate_signature = (
        tuple(sorted(gate_states.items()))
        if gate_states is not None
        else None
    )
    signature = (
        message_junction_id,
        state_sequence,
        selected_branch,
        command,
        gate_signature,
    )
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
            f"junction={message_junction_id}, "
            f"state_seq={state_sequence}, command={command}, "
            f"branch={selected_branch}, gates={gate_text}"
        )

    base_station.received_junction_id = message_junction_id
    base_station.received_state_sequence = state_sequence
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
        robot.received_junction_id = message_junction_id
        robot.received_state_sequence = state_sequence
        robot.received_sequence = communication_sequence
        if gate_states is not None:
            robot.known_junction_states[message_junction_id] = {
                "epoch": get_junction_state(
                    message_junction_id
                ).state_epoch,
                "sequence": state_sequence,
                "branch_states": get_junction_state(
                    message_junction_id
                ).branch_states.copy(),
                "gate_states": gate_states.copy(),
                "selected_branch": robot.received_branch,
            }


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


def get_communication_parent_velocity(robot):
    """Return the current velocity of the robot's Base-side parent."""
    parent = robot.comm_parent
    parent_velocity = getattr(parent, "velocity", None)
    return (
        parent_velocity.copy()
        if parent_velocity is not None
        else pygame.Vector2()
    )


def apply_communication_velocity_guard(robot, dt):
    """Prevent a connected mobile robot from outrunning its parent link."""
    parent = robot.comm_parent
    if (
        not robot.connected_to_base
        or parent is None
        or robot.role in {"ANCHOR", "RELAY", "TRUNK_RELAY"}
    ):
        return
    separation = robot.position - parent.position
    distance = separation.length()
    if distance <= EPSILON:
        return
    outward = separation / distance
    relative_velocity = (
        robot.velocity - get_communication_parent_velocity(robot)
    )
    relative_outward_speed = relative_velocity.dot(outward)
    maximum_outward_speed = max(
        0.0,
        (COMM_GUARD_HARD_LIMIT - distance) / max(dt, EPSILON),
    )
    if relative_outward_speed > maximum_outward_speed:
        robot.velocity -= outward * (
            relative_outward_speed - maximum_outward_speed
        )


def limit_communication_proposed_position(robot, proposed_position, old_position):
    """Clamp scripted motion to the hard Base-parent communication radius."""
    parent = robot.comm_parent
    if not robot.connected_to_base or parent is None:
        return proposed_position
    separation = proposed_position - parent.position
    distance = separation.length()
    if distance <= COMM_GUARD_HARD_LIMIT or distance <= EPSILON:
        return proposed_position
    bounded = (
        parent.position
        + separation * (COMM_GUARD_HARD_LIMIT / distance)
    )
    if is_walkable(bounded, robot.radius):
        return bounded
    return old_position.copy()


def constrain_communication_parent_separation(robot, old_position):
    """Final one-step guard against numerical overshoot of a parent link."""
    parent = robot.comm_parent
    if (
        not robot.connected_to_base
        or parent is None
        or robot.role in {"ANCHOR", "RELAY", "TRUNK_RELAY"}
    ):
        return
    bounded = limit_communication_proposed_position(
        robot,
        robot.position,
        old_position,
    )
    if bounded.distance_squared_to(robot.position) <= EPSILON:
        return
    robot.position = bounded
    separation = robot.position - parent.position
    distance = separation.length()
    if distance <= EPSILON:
        return
    outward = separation / distance
    parent_velocity = get_communication_parent_velocity(robot)
    relative_outward_speed = (
        robot.velocity - parent_velocity
    ).dot(outward)
    if relative_outward_speed > 0.0:
        robot.velocity -= outward * relative_outward_speed


def compute_connectivity_force(robot, grid):
    if base_station is None or robot.role in {"ANCHOR", "RELAY", "TRUNK_RELAY"}:
        return pygame.Vector2()
    if robot.connected_to_base:
        parent = robot.comm_parent
        if parent is None:
            return pygame.Vector2()
        delta = parent.position - robot.position
        distance = delta.length()
        if distance <= COMM_GUARD_START or distance <= EPSILON:
            return pygame.Vector2()
        toward_parent = delta / distance
        barrier_ratio = smoothstep01(
            (distance - COMM_GUARD_START)
            / max(
                COMM_GUARD_HARD_LIMIT - COMM_GUARD_START,
                EPSILON,
            )
        )
        parent_velocity = get_communication_parent_velocity(robot)
        relative_velocity = robot.velocity - parent_velocity
        outward_speed = max(
            0.0,
            -relative_velocity.dot(toward_parent),
        )
        spring_force = toward_parent * (
            COMM_PARENT_SPRING_GAIN
            * (distance - COMM_GUARD_START)
            * barrier_ratio
            + COMM_PARENT_DAMPING_GAIN
            * outward_speed
            * barrier_ratio
        )
        velocity_match_force = (
            parent_velocity - robot.velocity
        ) * (
            COMM_PARENT_VELOCITY_MATCH_GAIN
            * barrier_ratio
        )
        return limit_vector(
            spring_force + velocity_match_force,
            COMM_GUARD_FORCE_LIMIT,
        )
    target = find_nearest_connected_robot(robot, grid)
    if target is None:
        return pygame.Vector2()
    delta = target.position - robot.position
    distance = delta.length()
    if distance <= EPSILON:
        return pygame.Vector2()
    toward_target = delta / distance
    target_velocity = getattr(target, "velocity", pygame.Vector2())
    recovery_force = (
        toward_target
        * COMM_RECOVERY_GAIN
        * min(distance, COMM_RECOVERY_RANGE)
        + (target_velocity - robot.velocity)
        * COMM_PARENT_VELOCITY_MATCH_GAIN
    )
    return limit_vector(recovery_force, COMM_GUARD_FORCE_LIMIT)


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
        pygame.draw.line(
            surface,
            color,
            robot.position,
            parent.position,
            width=COMM_LINK_WIDTH,
        )
    for bridge, neighbor in communication_redundant_links:
        pygame.draw.line(
            surface,
            COMM_BRIDGE_LINK_COLOR,
            bridge.position,
            neighbor.position,
            width=max(2, COMM_LINK_WIDTH + 1),
        )

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
    junction_state = get_junction_state()
    endpoint = (
        junction_state.selected_anchor_position
        if junction_state.selected_anchor_position is not None
        else junction_state.anchor_slots[0]
    )
    vector = endpoint - BASE_POSITION
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


def get_relays_inside_branch(robots, branch):
    """Return only breadcrumbs physically blocking completion of *branch*."""
    return [
        robot
        for robot in get_active_branch_relays(robots)
        if get_robot_region(robot.position) == branch
    ]


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
    """Breadcrumbs mark the trail; NORMAL robots remain mobile SPH mass."""
    del robots, breadcrumb, breadcrumb_progress
    return []


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
        f"static_guards={len(guards)}"
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
        robot.current_junction_id = None
        for junction_id, junction_state in junctions.items():
            inside_junction = junction_state.rect.collidepoint(
                robot.position.x,
                robot.position.y,
            )
            inside_anchor_region = junction_state.anchor_region.collidepoint(
                robot.position.x,
                robot.position.y,
            )
            was_inside = robot.was_in_anchor_regions.get(junction_id, False)
            if inside_junction:
                robot.current_junction_id = junction_id
            if (
                inside_anchor_region
                and not was_inside
                and junction_id not in robot.anchor_region_entry_times
                and robot.role == "NORMAL"
            ):
                robot.anchor_region_entry_times[junction_id] = current_time
                if junction_id == CURRENT_JUNCTION_ID:
                    robot.anchor_region_entry_time = current_time
            robot.was_in_anchor_regions[junction_id] = inside_anchor_region
            if junction_id == CURRENT_JUNCTION_ID:
                robot.was_in_anchor_region = inside_anchor_region


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


def point_to_segment_distance(point, segment_start, segment_end):
    """Return the shortest distance from *point* to a finite line segment."""
    segment = segment_end - segment_start
    length_squared = segment.length_squared()
    if length_squared <= EPSILON:
        return point.distance_to(segment_start)
    projection = clamp(
        (point - segment_start).dot(segment) / length_squared,
        0.0,
        1.0,
    )
    closest = segment_start + projection * segment
    return point.distance_to(closest)


def anchor_branch_flow_penalty(
    park_position: pygame.Vector2,
    selected_branch: Optional[str],
    junction_state: JunctionState,
) -> float:
    """Penalize a slot lying on the open Branch side of the Junction."""
    if selected_branch not in BRANCH_DIRECTIONS:
        return 0.0
    center = pygame.Vector2(junction_state.rect.center)
    radial = park_position - center
    if radial.length_squared() <= EPSILON:
        return 1.0
    alignment = radial.normalize().dot(BRANCH_DIRECTIONS[selected_branch])
    return max(0.0, alignment)


def anchor_movement_obstruction_cost(
    candidate,
    robots,
    park_position: pygame.Vector2,
    junction_state: JunctionState,
):
    """Estimate how much parking this candidate disrupts Junction flow.

    The deployment corridor from the candidate to the fixed edge parking
    position is treated as a swept path. NORMAL robots close to that corridor
    contribute more cost, with an additional penalty for removing a robot from
    a locally dense part of the fluid body.
    """
    corridor_load = 0.0
    local_flow_load = 0.0
    start = candidate.position
    end = park_position
    for other in robots:
        if (
            other is candidate
            or other.role != "NORMAL"
            or not junction_state.rect.collidepoint(
                other.position.x,
                other.position.y,
            )
        ):
            continue

        corridor_distance = point_to_segment_distance(
            other.position,
            start,
            end,
        )
        if corridor_distance < ANCHOR_OBSTRUCTION_RADIUS:
            corridor_load += (
                1.0
                - corridor_distance / ANCHOR_OBSTRUCTION_RADIUS
            )

        local_distance = candidate.position.distance_to(other.position)
        if local_distance < ANCHOR_OBSTRUCTION_RADIUS:
            local_flow_load += (
                1.0
                - local_distance / ANCHOR_OBSTRUCTION_RADIUS
            )

    branch_flow_load = anchor_branch_flow_penalty(
        park_position,
        distributed_consensus_branch,
        junction_state,
    )
    return corridor_load + 0.5 * local_flow_load + 2.0 * branch_flow_load


def normalized_minimum_cost(value, minimum, maximum):
    """Map a raw metric to [0, 1], where the smallest value is best."""
    if maximum - minimum <= EPSILON:
        return 0.0
    return clamp((value - minimum) / (maximum - minimum), 0.0, 1.0)


def compute_anchor_candidate_costs(
    candidates,
    robots,
    junction_state: JunctionState,
):
    if not candidates:
        return

    junction_id = junction_state.junction_id
    entry_times = [
        candidate.anchor_region_entry_times[junction_id]
        for candidate in candidates
    ]
    min_time, max_time = min(entry_times), max(entry_times)

    parking_distances = {
        (candidate.robot_id, slot_index): candidate.position.distance_to(slot)
        for candidate in candidates
        for slot_index, slot in enumerate(junction_state.anchor_slots)
    }
    min_parking_distance = min(parking_distances.values(), default=0.0)
    max_parking_distance = max(parking_distances.values(), default=0.0)

    obstruction_loads = {
        (candidate.robot_id, slot_index): anchor_movement_obstruction_cost(
            candidate,
            robots,
            slot,
            junction_state,
        )
        for candidate in candidates
        for slot_index, slot in enumerate(junction_state.anchor_slots)
    }
    min_obstruction = min(obstruction_loads.values(), default=0.0)
    max_obstruction = max(obstruction_loads.values(), default=0.0)

    neighbor_info = {
        candidate.robot_id: local_visible_neighbor_count(candidate, robots)
        for candidate in candidates
    }
    max_neighbors = max((info[0] for info in neighbor_info.values()), default=1)
    max_margin = max((info[1] for info in neighbor_info.values()), default=1.0)

    for candidate in candidates:
        arrival_cost = normalized_minimum_cost(
            candidate.anchor_region_entry_times[junction_id],
            min_time,
            max_time,
        )
        neighbor_count, average_margin = neighbor_info[candidate.robot_id]
        communication_quality = (
            0.6 * neighbor_count / max(max_neighbors, 1)
            + 0.4 * average_margin / max(max_margin, EPSILON)
        )
        communication_cost = 1.0 - clamp(
            communication_quality,
            0.0,
            1.0,
        )

        slot_costs = []
        for slot_index, slot in enumerate(junction_state.anchor_slots):
            pair_key = (candidate.robot_id, slot_index)
            parking_cost = normalized_minimum_cost(
                parking_distances[pair_key],
                min_parking_distance,
                max_parking_distance,
            )
            obstruction_cost = normalized_minimum_cost(
                obstruction_loads[pair_key],
                min_obstruction,
                max_obstruction,
            )
            components = {
                "arrival": arrival_cost,
                "parking": parking_cost,
                "obstruction": obstruction_cost,
                "communication": communication_cost,
            }
            total_cost = (
                ANCHOR_COST_WEIGHT_ARRIVAL * arrival_cost
                + ANCHOR_COST_WEIGHT_PARKING * parking_cost
                + ANCHOR_COST_WEIGHT_OBSTRUCTION * obstruction_cost
                + ANCHOR_COST_WEIGHT_COMMUNICATION * communication_cost
            )
            slot_costs.append(
                (total_cost, slot_index, slot, components)
            )

        best_cost, _, best_slot, best_components = min(
            slot_costs,
            key=lambda item: (item[0], item[1]),
        )
        candidate.anchor_election_cost = best_cost
        candidate.anchor_candidate_position = best_slot.copy()
        candidate.anchor_cost_components = best_components


def elect_junction_anchor(
    robots,
    junction_id: str = CURRENT_JUNCTION_ID,
):
    global junction_anchor
    junction_state = get_junction_state(junction_id)
    existing_anchor = junction_anchors.get(junction_id)
    if existing_anchor is not None:
        return existing_anchor
    candidates = [
        robot
        for robot in robots
        if (
            robot.role == "NORMAL"
            and junction_id in robot.anchor_region_entry_times
            and junction_state.rect.collidepoint(
                robot.position.x,
                robot.position.y,
            )
        )
    ]
    if not candidates:
        return None
    first_entry = min(
        candidate.anchor_region_entry_times[junction_id]
        for candidate in candidates
    )
    if len(candidates) < ANCHOR_ELECTION_MIN_CANDIDATES and simulation_time - first_entry < ANCHOR_ELECTION_WAIT_TIME:
        return None
    compute_anchor_candidate_costs(candidates, robots, junction_state)
    elected_anchor = min(
        candidates,
        key=lambda robot: (
            robot.anchor_election_cost,
            robot.anchor_region_entry_times[junction_id],
            robot.robot_id,
        ),
    )
    elected_anchor.role = "ANCHOR"
    elected_anchor.anchor_junction_id = junction_id
    elected_anchor.anchor_position = (
        elected_anchor.anchor_candidate_position.copy()
    )
    elected_anchor.local_branch_states = junction_state.branch_states.copy()
    elected_anchor.selected_branch = junction_state.selected_branch
    elected_anchor.branch_gate_states = junction_state.gate_states.copy()
    junction_state.anchor_robot_id = elected_anchor.robot_id
    junction_state.selected_anchor_position = (
        elected_anchor.anchor_position.copy()
    )
    junction_anchors[junction_id] = elected_anchor
    if junction_id == CURRENT_JUNCTION_ID:
        junction_anchor = elected_anchor
    components = elected_anchor.anchor_cost_components
    print(
        f"[Anchor] junction={junction_id}, "
        f"robot={elected_anchor.robot_id}, "
        f"minimum_cost={elected_anchor.anchor_election_cost:.3f}, "
        f"arrival={components['arrival']:.3f}, "
        f"parking={components['parking']:.3f}, "
        f"obstruction={components['obstruction']:.3f}, "
        f"communication={components['communication']:.3f}, "
        f"entry={elected_anchor.anchor_region_entry_times[junction_id]:.3f}, "
        f"slot=({elected_anchor.anchor_position.x:.1f},"
        f"{elected_anchor.anchor_position.y:.1f})"
    )
    return elected_anchor


def preserve_consensus_at_anchor(
    anchor: Optional["Robot"],
    selected_branch: Optional[str] = None,
    clear_selection: bool = False,
    junction_id: str = CURRENT_JUNCTION_ID,
) -> None:
    """Store a NORMAL peer-consensus result without making Anchor decide it."""
    junction_state = get_junction_state(junction_id)
    effective_selection = (
        None
        if clear_selection
        else (
            selected_branch
            if selected_branch is not None
            else junction_state.selected_branch
        )
    )
    junction_state.record_consensus(
        branch_states,
        branch_gate_states,
        effective_selection,
        simulation_time,
    )
    if anchor is None or anchor.role != "ANCHOR":
        return
    anchor.local_branch_states = junction_state.branch_states.copy()
    if clear_selection:
        anchor.selected_branch = None
        anchor.distributed_branch_decision = None
    elif selected_branch is not None:
        anchor.selected_branch = selected_branch
        anchor.distributed_branch_decision = selected_branch
    anchor.branch_gate_states = junction_state.gate_states.copy()
    anchor.known_junction_states[junction_id] = {
        "epoch": junction_state.state_epoch,
        "sequence": junction_state.state_sequence,
        "branch_states": junction_state.branch_states.copy(),
        "gate_states": junction_state.gate_states.copy(),
        "selected_branch": junction_state.selected_branch,
        "selected_edge_id": junction_state.edge_id_for_branch(
            junction_state.selected_branch
        ) if junction_state.selected_branch is not None else None,
        "parent_junction_id": junction_state.parent_junction_id,
        "parent_edge_id": junction_state.parent_edge_id,
    }


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


def get_cohort_lane_offset(robot: "Robot") -> float:
    """Map a robot's deployment column to a wall-safe corridor lane."""
    lane_limit = max(
        0.0,
        half_width - FILL_LANE_WALL_CLEARANCE,
    )
    return clamp(
        robot.ingress_lane_x - center_x,
        -lane_limit,
        lane_limit,
    )


def get_lane_preserving_cohort_direction(
    robot: "Robot",
    branch: str,
    final_target: Optional[pygame.Vector2] = None,
) -> pygame.Vector2:
    """Route a whole-width cohort through the Junction without center collapse.

    The spawn-lane x coordinate is continuously mapped to a lateral Branch
    coordinate.  Robots therefore turn as parallel streamlines instead of all
    aiming at the same branch-mouth point.
    """
    region = get_robot_region(robot.position)
    lane_offset = get_cohort_lane_offset(robot)

    if region == "BOTTOM":
        target = pygame.Vector2(
            center_x + lane_offset,
            center_y + half_width - FILL_LANE_LOOKAHEAD * 0.35,
        )
        return normalized_direction_toward(robot.position, target)

    if region == "JUNCTION":
        if branch == "UP":
            target = pygame.Vector2(
                center_x + lane_offset,
                center_y - half_width - FILL_LANE_LOOKAHEAD,
            )
        elif branch == "LEFT":
            target = pygame.Vector2(
                center_x - half_width - FILL_LANE_LOOKAHEAD,
                center_y + lane_offset,
            )
        else:
            target = pygame.Vector2(
                center_x + half_width + FILL_LANE_LOOKAHEAD,
                center_y + lane_offset,
            )
        return normalized_direction_toward(robot.position, target)

    if region == branch:
        axial_target = (
            final_target.copy()
            if final_target is not None
            else get_branch_tip_target(branch)
        )
        if branch == "UP":
            axial_target.x = center_x + lane_offset
        else:
            axial_target.y = center_y + lane_offset
        return normalized_direction_toward(robot.position, axial_target)

    return geodesic_edf_direction(
        robot.position,
        branch,
        final_target,
    )


def get_collective_transfer_direction(
    robot: "Robot",
) -> tuple[pygame.Vector2, float]:
    """Blend the whole Base/Junction cohort from next-Branch to source-follow."""
    if transfer_branch is None:
        return pygame.Vector2(), 0.0
    next_direction = geodesic_edf_direction(
        robot.position,
        transfer_branch,
    )
    region = get_robot_region(robot.position)
    if region not in {"BOTTOM", "JUNCTION"}:
        return next_direction, 0.0

    frontier_weight = base_front_equilibrium_weight(robot)
    continuity_weight = collective_equilibrium_activation()
    follow_weight = (
        continuity_weight
        * (
            BASE_FRONT_FOLLOW_MIN_WEIGHT
            + (
                1.0 - BASE_FRONT_FOLLOW_MIN_WEIGHT
            )
            * frontier_weight
        )
    )
    source_direction = geodesic_edf_direction(
        robot.position,
        active_branch,
    )
    blended = (
        next_direction * (1.0 - follow_weight)
        + source_direction * follow_weight
    )
    if blended.length_squared() <= EPSILON:
        return next_direction, follow_weight
    return blended.normalize(), follow_weight


def compute_pressure_coupled_edf_force(
    robot: "Robot",
) -> pygame.Vector2:
    """Project available SPH pressure along the selected branch EDF."""
    if (
        robot.role != "NORMAL"
        or not robot.connected_to_base
    ):
        return pygame.Vector2()
    region = get_robot_region(robot.position)
    if (
        phase == SimulationPhase.EXPLORE_BRANCH
        and region in {"JUNCTION", active_branch}
    ):
        target_branch = active_branch
        ramp = smoothstep01(
            branch_entry_timer / max(EDF_PRESSURE_RAMP_TIME, EPSILON)
        )
        # Open sustained cruise from the measured RIGHT-flow event rather
        # than waiting for the full Base pulse timer.  A short smooth blend
        # avoids replacing the old pressure oscillation with an EDF step.
        cruise_blend = get_initial_release_cruise_blend()
        phase_multiplier = (
            1.0
            + (
                EXPLORATION_CRUISE_EDF_MULTIPLIER - 1.0
            )
            * cruise_blend
        )
        force_limit = (
            EDF_PRESSURE_FORCE_LIMIT
            + (
                EXPLORATION_EDF_FORCE_LIMIT
                - EDF_PRESSURE_FORCE_LIMIT
            )
            * cruise_blend
        )
    elif (
        phase in {
            SimulationPhase.PRESSURE_PUSH,
            SimulationPhase.FLOW_BACKTRACK,
        }
        and transfer_branch is not None
        and robot.transfer_target == transfer_branch
        and region in {"BOTTOM", "JUNCTION", transfer_branch}
    ):
        target_branch = transfer_branch
        ramp = (
            1.0
            if phase == SimulationPhase.FLOW_BACKTRACK
            else smoothstep01(
                pressure_push_timer / max(PRESSURE_RAMP_TIME, EPSILON)
            )
        )
        phase_multiplier = (
            TRANSFER_EDF_FORCE_MULTIPLIER
            * transfer_target_motion_scale
        )
        force_limit = EDF_PRESSURE_FORCE_LIMIT
    else:
        return pygame.Vector2()
    normalized_pressure = max(
        0.0,
        robot.pressure
        / max(PRESSURE_GAIN * robot.density, EPSILON),
    )
    magnitude = min(
        force_limit,
        EDF_PRESSURE_COUPLING_GAIN
        * normalized_pressure
        * ramp
        * phase_multiplier
        * relay_motion_scale,
    )
    if (
        phase in {
            SimulationPhase.PRESSURE_PUSH,
            SimulationPhase.FLOW_BACKTRACK,
        }
        and transfer_branch is not None
        and region in {"BOTTOM", "JUNCTION"}
    ):
        edf_direction, _ = get_collective_transfer_direction(
            robot,
        )
    else:
        edf_direction = geodesic_edf_direction(
            robot.position,
            target_branch,
        )
    return edf_direction * magnitude


def compute_virtual_valve_force(robot: "Robot") -> pygame.Vector2:
    """Compatibility no-op: runtime valves are physical robot formations."""
    return pygame.Vector2()

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
    effective_width = estimate_effective_branch_width(robots, branch)
    contact_risk = observed_branch_contact_risk(branch)
    narrowness = (
        clamp(1.0 - effective_width / max(corridor_width, EPSILON), 0.0, 1.0)
        if effective_width > 0.0
        else 0.0
    )

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
        + BRANCH_CONTACT_RISK_WEIGHT * max(contact_risk, narrowness)
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
        "observed_contact_risk": contact_risk,
        "effective_width": effective_width,
        "narrowness": narrowness,
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


def prepare_branch_candidate_scores(robots, reference_density: float) -> list[str]:
    """Evaluate only inferred, unvisited local openings with a short SPH rollout."""
    global last_proxy_partition, last_proxy_cell_centers
    global last_proxy_mass_stats, last_proxy_robot_assignment
    global last_proxy_candidates, last_flow_rollout_scores

    candidates = [
        branch
        for branch in FIXED_BRANCH_ORDER
        if branch in detected_branch_candidates
        and branch_states.get(branch) == "UNVISITED"
    ]
    signature = tuple(candidates)
    if (
        signature == last_proxy_candidates
        and all(branch in last_flow_rollout_scores for branch in candidates)
    ):
        return candidates
    if not candidates:
        last_proxy_candidates = ()
        last_flow_rollout_scores = {}
        return []

    partition, centers, quotas = build_capacity_constrained_proxy_partition(
        candidates
    )
    mass_stats, robot_assignment = compute_proxy_mass_statistics(
        robots,
        candidates,
        partition,
        centers,
        quotas,
        reference_density,
    )
    scores: dict[str, dict] = {}
    for branch in candidates:
        cost, components = branch_efficiency_cost(
            branch,
            robots,
            previous_branch_direction,
            reference_density,
            mass_stats,
            robot_assignment,
            partition,
            centers,
        )
        scores[branch] = {
            "cost": cost,
            "components": components,
        }
    last_proxy_partition = partition
    last_proxy_cell_centers = centers
    last_proxy_mass_stats = mass_stats
    last_proxy_robot_assignment = robot_assignment
    last_proxy_candidates = signature
    last_flow_rollout_scores = scores
    return candidates


def update_distributed_branch_consensus(
    robots,
    reference_density: float,
) -> Optional[str]:
    """Run a local NORMAL vote over inferred branches ranked by local costs."""
    global distributed_consensus_branch
    if (
        not junction_inference_tracker.confirmed
        or simulation_time - junction_inference_tracker.confirmed_at
        < JUNCTION_DISCOVERY_SETTLE_TIME
    ):
        return None
    candidates = prepare_branch_candidate_scores(robots, reference_density)
    if not candidates:
        return None
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
        local_candidates = [
            branch
            for branch in candidates
            if robot.local_branch_states.get(branch) == "UNVISITED"
        ]
        if not local_candidates:
            continue
        preferred = min(
            local_candidates,
            key=lambda branch: (
                last_flow_rollout_scores[branch]["cost"],
                FIXED_BRANCH_ORDER.index(branch),
            ),
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
            if peer.branch_vote in local_candidates
        ]
        counts = {
            branch: peer_votes.count(branch)
            for branch in local_candidates
        }
        counts[preferred] = counts.get(preferred, 0) + 1
        robot.branch_vote = max(
            counts,
            key=lambda branch: (
                counts[branch],
                -last_flow_rollout_scores[branch]["cost"],
            ),
        )
        robot.branch_vote_confidence = (
            counts[robot.branch_vote] / max(sum(counts.values()), 1)
        )

    vote_counts = {
        branch: sum(robot.branch_vote == branch for robot in voters)
        for branch in candidates
    }
    selected = max(
        candidates,
        key=lambda branch: (
            vote_counts[branch],
            -last_flow_rollout_scores[branch]["cost"],
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
        f"votes={vote_counts[selected]}/{len(voters)}, "
        f"local-cost={last_flow_rollout_scores[selected]['cost']:.3f}"
    )
    return selected


def apply_consensus_branch_gates(open_branch: Optional[str]) -> None:
    """Apply the branch-mouth state agreed by the NORMAL peer consensus."""
    new_gate_states: dict[str, str]
    if open_branch is None:
        new_gate_states = {branch: "OPEN" for branch in BRANCHES}
    else:
        new_gate_states = {
            branch: (
                "OPEN"
                if branch == open_branch
                or branch not in detected_branch_candidates
                else "CLOSED"
            )
            for branch in BRANCHES
        }
    branch_gate_states.clear()
    branch_gate_states.update(new_gate_states)
    preserve_consensus_at_anchor(
        junction_anchor,
        open_branch,
        clear_selection=open_branch is None,
    )
    print(
        "[Distributed Gate Consensus] "
        + ", ".join(
            f"{branch}={branch_gate_states[branch]}"
            for branch in BRANCHES
        )
    )


def begin_cross_branch_transfer(robots, source: str, target: str) -> None:
    """Open the next branch while Shepherd pressure expels the source fluid."""
    global transfer_branch, final_base_transfer_active
    transfer_branch = target
    final_base_transfer_active = False
    new_gate_states = {
        branch: (
            "OPEN"
            if branch in {
                source,
                target,
                draining_branch,
            }
            else "CLOSED"
        )
        for branch in BRANCHES
    }
    branch_gate_states.clear()
    branch_gate_states.update(new_gate_states)
    preserve_consensus_at_anchor(junction_anchor)
    for robot in robots:
        region = get_robot_region(robot.position)
        robot.transfer_target = (
            target
            if region in {
                source,
                "JUNCTION",
                "BOTTOM",
                draining_branch,
            }
            and robot.role in {"NORMAL", "SHEPHERD"}
            and not robot.base_reserve
            else None
        )
    junction_seed_count = sum(
        robot.transfer_target == target
        and get_robot_region(robot.position) == "JUNCTION"
        for robot in robots
    )
    print(
        f"[Cross-Branch Transfer] {source} -> {target}; "
        f"{source}=OPEN, {target}=OPEN, "
        f"junction_seed={junction_seed_count}"
    )


def begin_guarded_return_to_junction(robots, source: str) -> None:
    """Backtrack to the Junction before any next branch is selected."""
    global transfer_branch, final_base_transfer_active
    transfer_branch = None
    final_base_transfer_active = False
    branch_gate_states.clear()
    branch_gate_states.update({
        branch: "OPEN" if branch == source else "CLOSED"
        for branch in BRANCHES
    })
    preserve_consensus_at_anchor(junction_anchor)
    for robot in robots:
        robot.transfer_target = None
    print(
        f"[Guarded Backtrack] {source} -> JUNCTION; "
        "next branch remains closed until full guards are rebuilt"
    )


def finish_cross_branch_transfer(robots, target: str) -> None:
    """End only the transient handoff after the target becomes active."""
    global transfer_branch
    if transfer_branch != target:
        return
    cleared_count = 0
    for robot in robots:
        if robot.transfer_target != target:
            continue
        robot.transfer_target = None
        cleared_count += 1
    transfer_branch = None
    print(
        f"[Cross-Branch Transfer] handoff released; "
        f"active={target}, cleared={cleared_count}"
    )


def begin_final_base_transfer(robots, source: str) -> None:
    """Use the final Shepherd piston to expel LEFT fluid toward Base."""
    global transfer_branch, final_base_transfer_active
    transfer_branch = None
    final_base_transfer_active = True
    new_gate_states = {
        branch: "OPEN" if branch == source else "CLOSED"
        for branch in BRANCHES
    }
    branch_gate_states.clear()
    branch_gate_states.update(new_gate_states)
    preserve_consensus_at_anchor(junction_anchor)
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
    branch_gate_states.clear()
    branch_gate_states.update(
        {branch: "CLOSED" for branch in BRANCHES}
    )
    preserve_consensus_at_anchor(junction_anchor)
    print("[Gate] UP=CLOSED, LEFT=CLOSED, RIGHT=CLOSED")


def update_draining_branch_gate(robots) -> None:
    """Keep a completed source mouth open only for a few late normals."""
    global draining_branch
    branch = draining_branch
    if branch is None:
        return
    remaining = [
        robot
        for robot in robots
        if get_robot_region(robot.position) == branch
        and robot.role in {
            "NORMAL",
            "SHEPHERD",
            "PRE_SHEPHERD",
        }
    ]
    if remaining:
        branch_gate_states[branch] = "OPEN"
        preserve_consensus_at_anchor(junction_anchor)
        return
    branch_gate_states[branch] = "CLOSED"
    preserve_consensus_at_anchor(junction_anchor)
    draining_branch = None
    print(f"[Pipeline Drain] source gate closed: {branch}")


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
    candidates = [
        branch
        for branch in detected_branch_candidates
        if branch != source and branch_states.get(branch) == "UNVISITED"
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda branch: (
            last_flow_rollout_scores.get(branch, {}).get("cost", float("inf")),
            FIXED_BRANCH_ORDER.index(branch),
        ),
    )


def choose_next_branch(anchor, robots, reference_density: float):
    """Commit the branch selected by peer consensus over local rollout costs."""
    global active_branch, previous_branch_direction, branch_order_plan
    global last_proxy_partition, last_proxy_cell_centers
    global last_proxy_mass_stats, last_proxy_robot_assignment
    global last_proxy_candidates
    global last_flow_rollout_scores
    global detected_branch_candidates, collision_points
    global effective_branch_widths
    global selected_branch_entry_lambda, branch_entry_timer

    selected = distributed_consensus_branch
    if (
        selected is None
        or selected not in detected_branch_candidates
        or branch_states.get(selected) != "UNVISITED"
    ):
        if anchor is not None:
            anchor.selected_branch = None
        return None

    if not branch_is_feasible(selected, robots):
        print(
            f"[DFS] warning: inferred branch {selected} did not pass resource "
            "feasibility; preserving peer consensus"
        )

    branch_states[selected] = "ACTIVE"
    for robot in robots:
        robot.local_branch_states[selected] = "ACTIVE"
        if robot.role == "NORMAL":
            robot.distributed_branch_decision = selected
    if anchor is not None:
        anchor.local_branch_states[selected] = "ACTIVE"
        anchor.selected_branch = selected
    apply_consensus_branch_gates(selected)
    preserve_consensus_at_anchor(anchor, selected)
    active_branch = selected
    branch_order_plan.append(selected)
    selected_branch_entry_lambda, _ = rollout_stiffness_for_branch(
        selected,
        previous_branch_direction,
    )
    branch_entry_timer = 0.0
    initialize_relay_plan(selected)

    selected_score = last_flow_rollout_scores.get(
        selected,
        {"cost": 0.0, "components": {}},
    )
    metrics.branch_selection_events.append({
        "time": simulation_time,
        "selected": selected,
        "cost": selected_score["cost"],
        "max_structural_loss": max(
            (structural_loss(branch) for branch in detected_branch_candidates),
            default=0,
        ),
        "components": dict(selected_score["components"]),
        "candidate_scores": {
            branch: {
                "cost": data["cost"],
                "components": dict(data["components"]),
            }
            for branch, data in last_flow_rollout_scores.items()
        },
    })

    print(
        f"[DFS] local-cost selected={selected}, "
        f"cost={selected_score['cost']:.3f}, "
        f"entry_lambda={selected_branch_entry_lambda:.3f}"
    )
    return selected

def complete_active_branch(anchor, branch, robots):
    global previous_branch_direction, distributed_consensus_branch
    branch_states[branch] = "VISITED"
    distributed_consensus_branch = None
    detected_branch_candidates = set()
    collision_points = deque(maxlen=CONTACT_POINT_MAX_COUNT)
    effective_branch_widths = {branch: 0.0 for branch in BRANCHES}
    for robot in robots:
        robot.local_branch_states[branch] = "VISITED"
        robot.branch_vote = None
        robot.branch_vote_confidence = 0.0
        robot.distributed_branch_decision = None
    if anchor is not None:
        anchor.local_branch_states[branch] = "VISITED"
        preserve_consensus_at_anchor(
            anchor,
            clear_selection=True,
        )
    previous_branch_direction = get_backtrack_direction(branch)
    metrics.branch_events.append({"branch": branch, "completed_at": simulation_time})
    print(f"[DFS] completed={branch}")


def release_anchor_for_final_return(anchor):
    global junction_anchor
    if anchor is None:
        return
    junction_id = anchor.anchor_junction_id or CURRENT_JUNCTION_ID
    anchor.role = "NORMAL"
    anchor.anchor_position = None
    anchor.anchor_candidate_position = None
    anchor.anchor_junction_id = None
    anchor.selected_branch = None
    anchor.distributed_branch_decision = None
    anchor.velocity.update(0.0, 0.0)
    junction_anchors.pop(junction_id, None)
    junction_state = get_junction_state(junction_id)
    junction_state.anchor_robot_id = None
    junction_state.selected_anchor_position = None
    if junction_id == CURRENT_JUNCTION_ID:
        junction_anchor = None


def begin_final_gather():
    global phase, relay_slots, relay_motion_scale, final_gather_timer
    global transfer_branch, final_base_transfer_active
    release_transient_roles_for_final_return(robots)
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
    global return_done_dwell, return_entry_stall_timer
    global return_last_bottom_count
    release_transient_roles_for_final_return(robots)
    relay_slots = []
    relay_motion_scale = 1.0
    close_all_branch_gates()
    release_anchor_for_final_return(anchor)
    return_trunk_release_pending = True
    return_trunk_retract_timer = 0.0
    return_trunk_last_released_id = None
    return_trunk_force_timer = 0.0
    return_done_dwell = 0.0
    return_entry_stall_timer = 0.0
    return_last_bottom_count = sum(
        get_robot_region(robot.position) == "BOTTOM"
        for robot in robots
    )
    phase = SimulationPhase.RETURN_TO_BASE
    print("[DFS] return to base")


def recover_return_entry_stragglers(robots) -> int:
    """Move boundary-stalled NORMAL robots safely inside the Base corridor."""
    recovered = 0
    for robot in robots:
        if (
            robot.role != "NORMAL"
            or get_robot_region(robot.position) != "JUNCTION"
            or bottom_rect.top - robot.position.y
            > RETURN_ENTRY_RECOVERY_DISTANCE
        ):
            continue
        target = pygame.Vector2(
            clamp(
                robot.position.x,
                bottom_rect.left + robot.radius + 1.0,
                bottom_rect.right - robot.radius - 1.0,
            ),
            bottom_rect.top + robot.radius + 1.0,
        )
        if not is_walkable(target, robot.radius):
            continue
        robot.position = target
        robot.previous_position = target.copy()
        robot.velocity.update(0.0, RETURN_TRUNK_RELEASE_INITIAL_SPEED)
        robot.acceleration.update(0.0, 0.0)
        robot.filtered_acceleration.update(0.0, 0.0)
        recovered += 1
    if recovered:
        print(f"[Final Return Recovery] boundary_stragglers={recovered}")
    return recovered

# =========================================================
# 12-1. Environment inference from emergent local observations
# =========================================================


@dataclass
class SequentialJunctionInferenceTracker:
    """Emmons-inspired sequential distribution and cohort validator."""

    forward_direction: pygame.Vector2 = field(
        default_factory=lambda: pygame.Vector2(0.0, -1.0)
    )
    distribution_history: deque = field(
        default_factory=lambda: deque(maxlen=JUNCTION_DISTRIBUTION_WINDOW)
    )
    baseline_lateral_variance: float = 0.0
    lateral_variance: float = 0.0
    expansion_ratio: float = 1.0
    expansion_dwell: float = 0.0
    observation_count: int = 0
    sector_distribution: dict[str, float] = field(default_factory=dict)
    cohort_counts: dict[str, int] = field(
        default_factory=lambda: {branch: 0 for branch in BRANCHES}
    )
    cohort_travel: dict[str, float] = field(
        default_factory=lambda: {branch: 0.0 for branch in BRANCHES}
    )
    cohort_depth: dict[str, float] = field(
        default_factory=lambda: {branch: 0.0 for branch in BRANCHES}
    )
    cohort_dwell: dict[str, float] = field(
        default_factory=lambda: {branch: 0.0 for branch in BRANCHES}
    )
    cohort_origins: dict[tuple[int, str], pygame.Vector2] = field(default_factory=dict)
    cohort_member_ids: dict[str, set[int]] = field(
        default_factory=lambda: {branch: set() for branch in BRANCHES}
    )
    valid_branches: set[str] = field(default_factory=set)
    discovery_dwell: float = 0.0
    last_valid_branch_signature: frozenset[str] = field(
        default_factory=frozenset
    )
    forward_probe_status: str = "UNRESOLVED"
    forward_contact_count: int = 0
    confirmed: bool = False
    confirmed_at: float = float("inf")

    def reset(self):
        self.forward_direction = pygame.Vector2(0.0, -1.0)
        self.distribution_history.clear()
        self.baseline_lateral_variance = 0.0
        self.lateral_variance = 0.0
        self.expansion_ratio = 1.0
        self.expansion_dwell = 0.0
        self.observation_count = 0
        self.sector_distribution = {}
        self.cohort_counts = {branch: 0 for branch in BRANCHES}
        self.cohort_travel = {branch: 0.0 for branch in BRANCHES}
        self.cohort_depth = {branch: 0.0 for branch in BRANCHES}
        self.cohort_dwell = {branch: 0.0 for branch in BRANCHES}
        self.cohort_origins.clear()
        self.cohort_member_ids = {branch: set() for branch in BRANCHES}
        self.valid_branches.clear()
        self.discovery_dwell = 0.0
        self.last_valid_branch_signature = frozenset()
        self.forward_probe_status = "UNRESOLVED"
        self.forward_contact_count = 0
        self.confirmed = False
        self.confirmed_at = float("inf")

    def update(self, robots, dt: float) -> bool:
        global detected_branch_candidates
        mobile = [
            robot
            for robot in robots
            if robot.role == "NORMAL"
            and robot.connected_to_base
            and get_robot_region(robot.position) != "OUTSIDE"
        ]
        if len(mobile) < JUNCTION_MIN_OBSERVATION_ROBOTS:
            return False

        moving_velocity = sum(
            (robot.observed_velocity for robot in mobile),
            pygame.Vector2(),
        ) / len(mobile)
        if (
            moving_velocity.length() >= JUNCTION_COHORT_MIN_SPEED
            and moving_velocity.dot(self.forward_direction) > 0.0
        ):
            measured = moving_velocity.normalize()
            blended = self.forward_direction * 0.94 + measured * 0.06
            if blended.length_squared() > EPSILON:
                self.forward_direction = blended.normalize()

        swarm_center = sum(
            (robot.position for robot in mobile),
            pygame.Vector2(),
        ) / len(mobile)
        forward_positions = [
            (robot.position - swarm_center).dot(self.forward_direction)
            for robot in mobile
        ]
        front_threshold = linear_quantile(
            forward_positions,
            JUNCTION_FRONT_QUANTILE,
        )
        front_seed = [
            robot
            for robot, projection in zip(mobile, forward_positions)
            if projection >= front_threshold
        ]
        front_center = sum(
            (robot.position for robot in front_seed),
            pygame.Vector2(),
        ) / max(len(front_seed), 1)
        observed = [
            robot
            for robot in mobile
            if robot.position.distance_to(front_center)
            <= JUNCTION_OBSERVATION_RADIUS
        ]
        self.observation_count = len(observed)
        if len(observed) < JUNCTION_MIN_OBSERVATION_ROBOTS:
            return False

        observation_center = sum(
            (robot.position for robot in observed),
            pygame.Vector2(),
        ) / len(observed)
        lateral_direction = pygame.Vector2(
            -self.forward_direction.y,
            self.forward_direction.x,
        )
        lateral_coordinates = [
            (robot.position - observation_center).dot(lateral_direction)
            for robot in observed
        ]
        self.lateral_variance = sum(
            value * value for value in lateral_coordinates
        ) / len(lateral_coordinates)
        if self.baseline_lateral_variance <= EPSILON:
            self.baseline_lateral_variance = max(
                self.lateral_variance,
                JUNCTION_LATERAL_EXPANSION_MIN,
            )
        self.expansion_ratio = (
            self.lateral_variance
            / max(self.baseline_lateral_variance, EPSILON)
        )
        expanding = (
            self.lateral_variance - self.baseline_lateral_variance
            >= JUNCTION_LATERAL_EXPANSION_MIN
            and self.expansion_ratio >= JUNCTION_LATERAL_EXPANSION_RATIO
        )
        self.expansion_dwell = self.expansion_dwell + dt if expanding else 0.0
        # Preserve the narrow-corridor reference.  Letting the baseline follow
        # a slow expansion upward makes a genuine Junction look stationary;
        # only downward relaxation is allowed after initialization.
        if (
            not expanding
            and self.lateral_variance < self.baseline_lateral_variance
        ):
            self.baseline_lateral_variance += JUNCTION_BASELINE_ALPHA * (
                self.lateral_variance - self.baseline_lateral_variance
            )

        sector_names = (
            "FRONT",
            "FRONT_LEFT",
            "LEFT",
            "BACK_LEFT",
            "BACK",
            "BACK_RIGHT",
            "RIGHT",
            "FRONT_RIGHT",
        )
        sector_counts = {name: 0 for name in sector_names}
        for robot in observed:
            relative = robot.position - observation_center
            if relative.length_squared() <= EPSILON:
                continue
            angle = math.atan2(
                relative.dot(lateral_direction),
                relative.dot(self.forward_direction),
            )
            sector_index = int(round(angle / (math.pi / 4.0))) % 8
            sector_counts[sector_names[sector_index]] += 1
        total_sector_count = max(sum(sector_counts.values()), 1)
        self.sector_distribution = {
            name: count / total_sector_count
            for name, count in sector_counts.items()
        }
        self.distribution_history.append(dict(self.sector_distribution))

        alignment_threshold = math.cos(JUNCTION_COHORT_HALF_ANGLE)
        for branch in BRANCHES:
            direction = BRANCH_DIRECTIONS[branch]
            maximum_travel = self.cohort_travel[branch]
            maximum_depth = 0.0
            for robot in observed:
                velocity = robot.observed_velocity
                heading_aligned = (
                    velocity.length() >= JUNCTION_COHORT_MIN_SPEED
                    and velocity.normalize().dot(direction)
                    >= alignment_threshold
                )
                key = (robot.robot_id, branch)
                # Directional motion inside the Junction is not evidence that
                # an opening exists.  Count the robot only after it has crossed
                # the physical mouth and penetrated a non-trivial distance into
                # that branch.  This prevents a handful of shallow probes from
                # creating a false branch candidate.
                if get_robot_region(robot.position) != branch:
                    continue
                # The travel origin is the first aligned pose *inside* this
                # opening, never a pose in Base or the Junction.  Otherwise a
                # straight-ahead robot would incorrectly count its entire
                # incoming corridor journey as branch traversal.
                if heading_aligned:
                    self.cohort_origins.setdefault(
                        key,
                        robot.position.copy(),
                    )
                depth = branch_depth_from_junction(robot.position, branch)
                maximum_depth = max(maximum_depth, depth)
                if depth < JUNCTION_COHORT_MIN_BRANCH_DEPTH:
                    continue
                origin = self.cohort_origins.get(key)
                if origin is None:
                    continue
                travel = (robot.position - origin).dot(direction)
                maximum_travel = max(
                    maximum_travel,
                    travel,
                )
                if travel >= JUNCTION_COHORT_MIN_TRAVEL:
                    self.cohort_member_ids[branch].add(robot.robot_id)
            self.cohort_counts[branch] = len(self.cohort_member_ids[branch])
            self.cohort_travel[branch] = maximum_travel
            self.cohort_depth[branch] = maximum_depth
            fraction = self.cohort_counts[branch] / len(observed)
            cohort_valid_now = (
                self.cohort_counts[branch] >= JUNCTION_COHORT_MIN_ROBOTS
                and fraction >= JUNCTION_COHORT_MIN_FRACTION
                and maximum_travel >= JUNCTION_COHORT_MIN_TRAVEL
            )
            self.cohort_dwell[branch] = (
                self.cohort_dwell[branch] + dt
                if cohort_valid_now
                else max(0.0, self.cohort_dwell[branch] - dt)
            )
            if (
                self.cohort_dwell[branch] >= JUNCTION_COHORT_DWELL_TIME
                and branch not in self.valid_branches
            ):
                self.valid_branches.add(branch)
                print(
                    f"[Branch Evidence] {branch} OPEN only after physical "
                    f"crossing: unique={self.cohort_counts[branch]}, "
                    f"depth={self.cohort_depth[branch]:.1f}, "
                    f"travel={self.cohort_travel[branch]:.1f}, "
                    f"dwell={self.cohort_dwell[branch]:.2f}"
                )

        forward_branch = max(
            BRANCHES,
            key=lambda branch: BRANCH_DIRECTIONS[branch].dot(
                self.forward_direction
            ),
        )
        lateral_axis = pygame.Vector2(
            -self.forward_direction.y,
            self.forward_direction.x,
        )
        front_contacts = [
            point
            for point in collision_points
            if point.branch is None
            and half_width * 0.75
            <= (point.position - pygame.Vector2(center_x, center_y)).dot(
                self.forward_direction
            )
            <= half_width * 1.25
            and abs(
                (point.position - pygame.Vector2(center_x, center_y)).dot(
                    lateral_axis
                )
            )
            <= half_width * 0.95
        ]
        self.forward_contact_count = len(front_contacts)
        contact_span = (
            max(
                (point.position - pygame.Vector2(center_x, center_y)).dot(
                    lateral_axis
                )
                for point in front_contacts
            )
            - min(
                (point.position - pygame.Vector2(center_x, center_y)).dot(
                    lateral_axis
                )
                for point in front_contacts
            )
            if len(front_contacts) >= 2
            else 0.0
        )
        forward_blocked = (
            len(front_contacts) >= JUNCTION_FRONT_BLOCK_MIN_CONTACTS
            and contact_span
            >= corridor_width * JUNCTION_FRONT_BLOCK_MIN_SPAN_RATIO
        )
        if forward_branch in self.valid_branches:
            self.forward_probe_status = f"OPEN:{forward_branch}"
        elif forward_blocked:
            self.forward_probe_status = "BLOCKED_BY_CONTACT"
        else:
            self.forward_probe_status = "UNRESOLVED"

        junction_signature = (
            self.expansion_dwell >= JUNCTION_EXPANSION_DWELL_TIME
            and len(self.valid_branches) >= JUNCTION_MIN_VALID_COHORTS
            and self.forward_probe_status != "UNRESOLVED"
        )
        # Additional robots crossing an already validated opening strengthen
        # existing evidence; they do not reveal a new environmental feature.
        # Reset settling only when the inferred branch set itself changes or
        # when the Junction distribution signature is genuinely lost.
        valid_branch_signature = frozenset(self.valid_branches)
        branch_set_changed = (
            valid_branch_signature != self.last_valid_branch_signature
        )
        self.last_valid_branch_signature = valid_branch_signature
        if not junction_signature or branch_set_changed:
            self.discovery_dwell = 0.0
        else:
            self.discovery_dwell += dt
        newly_confirmed = (
            not self.confirmed
            and junction_signature
            and self.discovery_dwell >= JUNCTION_DISCOVERY_SETTLE_TIME
        )
        if newly_confirmed:
            self.confirmed = True
            self.confirmed_at = simulation_time
            detected_branch_candidates = set(self.valid_branches)
            metrics.junction_inference_events.append({
                "time": simulation_time,
                "branches": sorted(self.valid_branches),
                "expansion_ratio": self.expansion_ratio,
                "lateral_variance": self.lateral_variance,
                "sector_distribution": dict(self.sector_distribution),
                "cohort_counts": dict(self.cohort_counts),
                "cohort_travel": dict(self.cohort_travel),
                "cohort_depth": dict(self.cohort_depth),
                "forward_probe_status": self.forward_probe_status,
            })
            print(
                "[Junction Inference] sequential expansion + cohorts: "
                f"branches={sorted(self.valid_branches)}, "
                f"variance-ratio={self.expansion_ratio:.2f}, "
                f"physical-depths={{{', '.join(f'{branch}:{self.cohort_depth[branch]:.1f}' for branch in sorted(self.valid_branches))}}}"
            )
        elif self.confirmed:
            detected_branch_candidates.update(self.valid_branches)
        return self.confirmed


@dataclass
class DeadEndInferenceTracker:
    branch: Optional[str] = None
    dwell: float = 0.0
    frontier_count: int = 0
    leader_contact: float = 0.0
    mean_contact: float = 0.0
    mean_forward_speed: float = 0.0
    mean_density_ratio: float = 0.0
    lateral_escape_ratio: float = 0.0
    shepherd_direct_contact_ratio: float = 0.0
    shepherd_contact_span_ratio: float = 0.0
    shepherd_mean_forward_speed: float = 0.0
    confirmed_depth: float = 0.0
    confirmed: bool = False

    def reset(self, branch: Optional[str] = None):
        self.branch = branch
        self.dwell = 0.0
        self.frontier_count = 0
        self.leader_contact = 0.0
        self.mean_contact = 0.0
        self.mean_forward_speed = 0.0
        self.mean_density_ratio = 0.0
        self.lateral_escape_ratio = 0.0
        self.shepherd_direct_contact_ratio = 0.0
        self.shepherd_contact_span_ratio = 0.0
        self.shepherd_mean_forward_speed = 0.0
        self.confirmed_depth = 0.0
        self.confirmed = False

    def update(self, robots, branch: str, reference_density: float, dt: float) -> bool:
        global observed_dead_end_depths
        if self.branch != branch:
            self.reset(branch)
        branch_robots = [
            robot
            for robot in robots
            if robot.role in {"NORMAL", "FRONTIER_SHEPHERD"}
            and get_robot_region(robot.position) == branch
        ]
        estimate_effective_branch_width(robots, branch)
        if not branch_robots:
            self.dwell = 0.0
            return False

        maximum_depth = max(
            branch_depth_from_junction(robot.position, branch)
            for robot in branch_robots
        )
        frontier = [
            robot
            for robot in branch_robots
            if branch_depth_from_junction(robot.position, branch)
            >= maximum_depth - DEAD_END_FRONTIER_DEPTH
        ]
        self.frontier_count = len(frontier)
        if len(frontier) < DEAD_END_MIN_FRONTIER_ROBOTS:
            self.dwell = 0.0
            return False

        direction = BRANCH_DIRECTIONS[branch]
        lateral = pygame.Vector2(-direction.y, direction.x)
        leader = max(
            frontier,
            key=lambda robot: branch_depth_from_junction(robot.position, branch),
        )
        evidences = [robot_contact_evidence(robot) for robot in frontier]
        self.leader_contact = robot_contact_evidence(leader)
        self.mean_contact = sum(evidences) / len(evidences)
        self.mean_forward_speed = sum(
            max(0.0, robot.observed_velocity.dot(direction))
            for robot in frontier
        ) / len(frontier)
        self.mean_density_ratio = sum(
            robot.density / max(reference_density, EPSILON)
            for robot in frontier
        ) / len(frontier)
        self.lateral_escape_ratio = sum(
            abs(robot.observed_velocity.dot(lateral))
            > DEAD_END_FORWARD_SPEED_THRESHOLD
            for robot in frontier
        ) / len(frontier)
        frontier_shepherds = get_frontier_shepherds(robots, branch)
        directly_contacting_shepherds = [
            robot
            for robot in frontier_shepherds
            if simulation_time - robot.last_forward_obstacle_contact_time
            <= DEAD_END_FORWARD_BUMPER_MEMORY
        ]
        self.shepherd_direct_contact_ratio = (
            len(directly_contacting_shepherds)
            / max(len(frontier_shepherds), 1)
        )
        self.shepherd_mean_forward_speed = (
            sum(
                max(0.0, robot.observed_velocity.dot(direction))
                for robot in frontier_shepherds
            )
            / max(len(frontier_shepherds), 1)
        )
        contact_lateral_coordinates = [
            (robot.position - get_branch_entrance(branch)).dot(lateral)
            for robot in directly_contacting_shepherds
        ]
        contact_span = (
            max(contact_lateral_coordinates) - min(contact_lateral_coordinates)
            if len(contact_lateral_coordinates) >= 2
            else 0.0
        )
        self.shepherd_contact_span_ratio = clamp(
            contact_span / max(corridor_width, EPSILON),
            0.0,
            1.0,
        )
        conditions = (
            len(frontier_shepherds) >= JUNCTION_GUARD_MIN_COUNT
            and self.shepherd_direct_contact_ratio
            >= DEAD_END_SHEPHERD_DIRECT_CONTACT_RATIO
            and self.shepherd_contact_span_ratio
            >= DEAD_END_SHEPHERD_CONTACT_SPAN_RATIO
            and self.shepherd_mean_forward_speed
            <= DEAD_END_FORWARD_SPEED_THRESHOLD
        )
        self.dwell = self.dwell + dt if conditions else max(0.0, self.dwell - dt)
        newly_confirmed = (
            not self.confirmed
            and self.dwell >= DEAD_END_CONFIRM_DWELL
        )
        if newly_confirmed:
            self.confirmed = True
            # Store the contacted leader's measured displacement from the
            # observed mouth.  This is the Shepherd boundary reference; the
            # renderer's known branch length is not substituted here.
            self.confirmed_depth = max(
                0.0,
                linear_quantile(
                    [
                        (robot.position - get_branch_entrance(branch)).dot(
                            direction
                        )
                        for robot in directly_contacting_shepherds
                    ],
                    0.50,
                ),
            )
            observed_dead_end_depths[branch] = self.confirmed_depth
            metrics.dead_end_events.append({
                "time": simulation_time,
                "branch": branch,
                "frontier_count": self.frontier_count,
                "leader_contact": self.leader_contact,
                "mean_contact": self.mean_contact,
                "mean_forward_speed": self.mean_forward_speed,
                "mean_density_ratio": self.mean_density_ratio,
                "lateral_escape_ratio": self.lateral_escape_ratio,
                "shepherd_direct_contact_ratio": (
                    self.shepherd_direct_contact_ratio
                ),
                "shepherd_contact_span_ratio": (
                    self.shepherd_contact_span_ratio
                ),
                "shepherd_mean_forward_speed": (
                    self.shepherd_mean_forward_speed
                ),
                "observed_depth": self.confirmed_depth,
            })
            print(
                f"[Dead-end Inference] branch={branch}, "
                f"contact={self.mean_contact:.2f}, "
                f"forward={self.mean_forward_speed:.2f}, "
                f"rho={self.mean_density_ratio:.2f}, "
                f"bumper={self.shepherd_direct_contact_ratio:.2f}, "
                f"span={self.shepherd_contact_span_ratio:.2f}, "
                f"shepherd_v={self.shepherd_mean_forward_speed:.2f}, "
                f"observed_depth={self.confirmed_depth:.1f}"
            )
        return self.confirmed


junction_inference_tracker = SequentialJunctionInferenceTracker()
dead_end_inference_tracker = DeadEndInferenceTracker()


# =========================================================
# 12-2. Junction stability consensus
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
    lateral_coverage_ratio: float = 0.0
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
        self.lateral_coverage_ratio = 0.0
        self.front_delta = float("inf")
        self.tip_count = 0
        self.packed_min_count = SATURATION_PACKED_MIN_TIP_ROBOTS
        self.recognition_mode = "WAIT"
        self.saturated = False


saturation_tracker = SaturationTracker()


@dataclass
class BranchContinuityTracker:
    branch: Optional[str] = None
    dwell: float = 0.0
    wait_time: float = 0.0
    minimum_slice_count: int = 0
    covered_slice_ratio: float = 0.0
    maximum_depth_gap: float = float("inf")
    ready: bool = False
    timed_out: bool = False

    def reset(self, branch: Optional[str] = None):
        self.branch = branch
        self.dwell = 0.0
        self.wait_time = 0.0
        self.minimum_slice_count = 0
        self.covered_slice_ratio = 0.0
        self.maximum_depth_gap = float("inf")
        self.ready = False
        self.timed_out = False

    def update(self, robots, branch: str, dt: float) -> bool:
        if self.branch != branch:
            self.reset(branch)
        self.wait_time += dt
        maximum_depth = max(
            BRANCH_CONTINUITY_SLICE_DEPTH,
            get_shepherd_normal_limit_depth(branch),
        )
        slice_count = max(
            1,
            math.ceil(
                maximum_depth
                / BRANCH_CONTINUITY_SLICE_DEPTH
            ),
        )
        counts = [0 for _ in range(slice_count)]
        depths = []
        for robot in robots:
            if (
                robot.role != "NORMAL"
                or get_robot_region(robot.position) != branch
            ):
                continue
            depth = branch_depth_from_junction(
                robot.position,
                branch,
            )
            if depth > maximum_depth:
                continue
            depths.append(depth)
            slice_index = min(
                slice_count - 1,
                int(
                    depth
                    / BRANCH_CONTINUITY_SLICE_DEPTH
                ),
            )
            counts[slice_index] += 1

        self.minimum_slice_count = min(counts, default=0)
        covered_count = sum(
            count
            >= BRANCH_CONTINUITY_MIN_ROBOTS_PER_SLICE
            for count in counts
        )
        self.covered_slice_ratio = (
            covered_count / max(slice_count, 1)
        )
        ordered_depths = [0.0] + sorted(depths) + [
            maximum_depth
        ]
        self.maximum_depth_gap = max(
            (
                after - before
                for before, after in zip(
                    ordered_depths,
                    ordered_depths[1:],
                )
            ),
            default=maximum_depth,
        )
        conditions = (
            self.covered_slice_ratio
            >= BRANCH_CONTINUITY_REQUIRED_SLICE_RATIO
            and self.maximum_depth_gap
            <= BRANCH_CONTINUITY_MAX_DEPTH_GAP
        )
        self.dwell = self.dwell + dt if conditions else 0.0
        self.ready = (
            self.dwell >= BRANCH_CONTINUITY_DWELL_TIME
        )
        self.timed_out = (
            self.wait_time >= BRANCH_CONTINUITY_FILL_TIMEOUT
        )
        return self.ready


branch_continuity_tracker = BranchContinuityTracker()


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


def tip_lateral_coverage_ratio(robots_at_tip, branch):
    """Measure how much corridor width the compact layer covers.

    A valid Shepherd pack is normally a thin slab, so area occupancy is a poor
    trigger: making the slab denser can reduce its occupied area.  Lateral
    coverage measures the physically relevant cross-section instead.
    """
    if not robots_at_tip:
        return 0.0
    bin_count = max(
        1,
        math.ceil(
            (corridor_width - EPSILON)
            / SATURATION_CELL_SIZE
        ),
    )
    occupied_bins = set()
    for robot in robots_at_tip:
        lateral_position = (
            robot.position.x - (center_x - half_width)
            if branch == "UP"
            else robot.position.y - (center_y - half_width)
        )
        bin_index = int(
            lateral_position / max(SATURATION_CELL_SIZE, EPSILON)
        )
        occupied_bins.add(
            int(clamp(bin_index, 0, bin_count - 1))
        )
    return len(occupied_bins) / bin_count


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
        tracker.lateral_coverage_ratio = (
            tip_lateral_coverage_ratio(tip, branch)
        )
        front_progress = max(branch_progress(robot, branch) for robot in tip)
    else:
        tracker.low_speed_ratio = 0.0
        tracker.average_density_ratio = 0.0
        tracker.occupancy_ratio = 0.0
        tracker.lateral_coverage_ratio = 0.0
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
    density_packed_conditions = (
        tracker.tip_count >= SATURATION_PACKED_MIN_TIP_ROBOTS
        and tracker.lateral_coverage_ratio
        >= SATURATION_PACKED_LATERAL_COVERAGE_RATIO
        and density_evidence
    )
    geometry_packed_conditions = (
        geometry_evidence
        and tracker.lateral_coverage_ratio
        >= SATURATION_GEOMETRY_LATERAL_COVERAGE_RATIO
    )
    packed_conditions = (
        density_packed_conditions
        or geometry_packed_conditions
    )
    if packed_conditions:
        tracker.recognition_mode = (
            "PACKED_DENSITY"
            if density_packed_conditions
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


def build_shepherd_slots(
    branch,
    count,
    observed_boundary_depth: Optional[float] = None,
):
    usable_half = half_width - SHEPHERD_EDGE_MARGIN
    lateral = [0.0] if count <= 1 else [
        -usable_half + 2.0 * usable_half * index / (count - 1)
        for index in range(count)
    ]
    boundary_depth = (
        get_shepherd_boundary_depth(branch)
        if observed_boundary_depth is None
        else max(0.0, observed_boundary_depth)
    )
    direction = BRANCH_DIRECTIONS[branch]
    center = get_branch_entrance(branch) + direction * boundary_depth
    normal = pygame.Vector2(-direction.y, direction.x)
    return [center + normal * value for value in lateral]


def reset_shepherd_roles(robots):
    for robot in robots:
        if robot.role in {"SHEPHERD", "FRONTIER_SHEPHERD"}:
            robot.role = "NORMAL"
            robot.filtered_acceleration.update(0.0, 0.0)
            robot.shepherd_anchor = None
            robot.shepherd_origin = None
            robot.shepherd_branch = None
            robot.junction_guard_anchor = None
            robot.junction_guard_branch = None
            robot.junction_guard_hop = -1
            robot.junction_guard_parent_id = None
            robot.junction_guard_layer = -1
            robot.is_branch_leader = False


def release_transient_roles_for_final_return(robots):
    """Release stale local roles that would otherwise stay position-locked."""
    global pre_shepherd_branch
    global pre_shepherd_pack_dwell, pre_shepherd_pack_ready

    released_shepherds = 0
    released_relays = 0
    for robot in robots:
        robot.comm_bridge_target = None
        robot.comm_bridge_index = -1
        robot.comm_bridge_branch = None

        if robot.role in {
            "SHEPHERD",
            "PRE_SHEPHERD",
            "JUNCTION_GUARD",
            "FRONTIER_SHEPHERD",
        }:
            robot.role = "NORMAL"
            robot.shepherd_anchor = None
            robot.shepherd_origin = None
            robot.shepherd_branch = None
            robot.junction_guard_anchor = None
            robot.junction_guard_branch = None
            robot.junction_guard_hop = -1
            robot.junction_guard_parent_id = None
            robot.junction_guard_layer = -1
            robot.is_branch_leader = False
            robot.transfer_target = None
            robot.velocity.update(0.0, 0.0)
            robot.acceleration.update(0.0, 0.0)
            robot.filtered_acceleration.update(0.0, 0.0)
            released_shepherds += 1
        elif robot.role == "RELAY":
            # Branch Breadcrumbs are transient. Only TRUNK_RELAY robots remain
            # for the deliberate Junction-to-Base sequential retraction.
            robot.role = "NORMAL"
            robot.relay_anchor = None
            robot.relay_index = -1
            robot.transfer_target = None
            robot.velocity.update(0.0, RETURN_TRUNK_RELEASE_INITIAL_SPEED)
            robot.acceleration.update(0.0, 0.0)
            robot.filtered_acceleration.update(0.0, 0.0)
            released_relays += 1

    pre_shepherd_branch = None
    pre_shepherd_pack_dwell = 0.0
    pre_shepherd_pack_ready = False
    if released_shepherds or released_relays:
        print(
            "[Final Cleanup] "
            f"released_shepherds={released_shepherds}, "
            f"released_branch_relays={released_relays}"
        )



def shepherd_candidates(robots, branch, required_count):
    """Return the leading robots already inside the original capture region."""
    capture_rect = early_capture_regions[branch]
    candidates = [
        robot
        for robot in robots
        if robot.role in {"NORMAL", "FRONTIER_SHEPHERD"}
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
        if robot.role in {"NORMAL", "FRONTIER_SHEPHERD"}
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
        robot.shepherd_branch = branch
        robot.velocity.update(0.0, 0.0)
        robot.filtered_acceleration.update(0.0, 0.0)
        selected.append(robot)
        print(f"[Shepherd] robot={robot.robot_id}, score={score:.3f}")
    print(f"[Shepherd] adaptive count={required_count}")
    return selected


def get_pre_shepherds(robots, branch: Optional[str] = None):
    return [
        robot
        for robot in robots
        if robot.role == "PRE_SHEPHERD"
        and (
            branch is None
            or robot.shepherd_branch == branch
        )
    ]


def pre_shepherd_boundary_formed(robots, branch: str) -> bool:
    shepherds = get_pre_shepherds(robots, branch)
    return (
        len(shepherds) == adaptive_shepherd_count()
        and all(
            robot.shepherd_anchor is not None
            and robot.position.distance_to(robot.shepherd_anchor)
            <= SHEPHERD_FORM_TOLERANCE
            for robot in shepherds
        )
    )


def select_pre_shepherds(robots, branch: str, grid):
    """Elect the next branch shield without disturbing active Shepherds."""
    del grid
    global pre_shepherd_branch
    global pre_shepherd_pack_dwell, pre_shepherd_pack_ready
    if get_pre_shepherds(robots):
        return get_pre_shepherds(robots, branch)
    required_count = adaptive_shepherd_count()
    slots = build_shepherd_slots(branch, required_count)
    candidates = shepherd_candidates(robots, branch, required_count)
    assignment = assign_shepherd_slots(candidates, slots)
    if len(assignment) != required_count:
        return []
    selected = []
    pre_shepherd_branch = branch
    pre_shepherd_pack_dwell = 0.0
    pre_shepherd_pack_ready = False
    for robot, slot, score in assignment:
        robot.role = "PRE_SHEPHERD"
        robot.shepherd_anchor = slot.copy()
        robot.shepherd_origin = slot.copy()
        robot.shepherd_branch = branch
        robot.velocity.update(0.0, 0.0)
        robot.acceleration.update(0.0, 0.0)
        robot.filtered_acceleration.update(0.0, 0.0)
        selected.append(robot)
        print(
            f"[Pre-Shepherd] branch={branch}, "
            f"robot={robot.robot_id}, score={score:.3f}"
        )
    enforce_pre_shepherd_curtain_for_swarm(robots)
    print(
        f"[Pre-Shepherd] shield elected: "
        f"branch={branch}, count={len(selected)}"
    )
    return selected


def update_pre_shepherd_pack_readiness(
    robots,
    branch: str,
    reference_density: float,
    dt: float,
) -> bool:
    """Track packed mass behind the prepared next-branch curtain."""
    global pre_shepherd_pack_dwell, pre_shepherd_pack_ready
    tip = tip_robots(robots, branch)
    required_count = max(
        SATURATION_PACKED_MIN_TIP_ROBOTS,
        math.ceil(
            max(1, len(get_pre_shepherds(robots, branch)))
            * SATURATION_PACKED_ROBOTS_PER_SHEPHERD
        ),
    )
    average_density_ratio = (
        sum(robot.density for robot in tip)
        / max(len(tip), 1)
        / max(reference_density, EPSILON)
    )
    lateral_coverage = tip_lateral_coverage_ratio(tip, branch)
    packed = (
        len(tip) >= SATURATION_PACKED_MIN_TIP_ROBOTS
        and lateral_coverage
        >= SATURATION_PACKED_LATERAL_COVERAGE_RATIO
        and (
            len(tip) >= required_count
            or average_density_ratio
            >= SATURATION_PACKED_DENSITY_RATIO
        )
    )
    pre_shepherd_pack_dwell = (
        pre_shepherd_pack_dwell + dt
        if packed
        else 0.0
    )
    pre_shepherd_pack_ready = (
        pre_shepherd_pack_dwell
        >= PRE_SHEPHERD_PACK_DWELL_TIME
    )
    return pre_shepherd_pack_ready


def promote_pre_shepherds(robots, branch: str) -> bool:
    """Activate a prepared shield only after the prior line has returned."""
    global pre_shepherd_branch
    if (
        not pre_shepherd_boundary_formed(robots, branch)
        or not pre_shepherd_pack_ready
    ):
        return False
    for robot in get_pre_shepherds(robots, branch):
        robot.role = "SHEPHERD"
        robot.velocity.update(0.0, 0.0)
        robot.acceleration.update(0.0, 0.0)
        robot.filtered_acceleration.update(0.0, 0.0)
    pre_shepherd_branch = None
    print(
        f"[Pre-Shepherd] promoted after prior Junction return: "
        f"branch={branch}"
    )
    return True


def update_pre_shepherd_pipeline(
    robots,
    grid,
    reference_density: float,
    dt: float,
) -> None:
    """Prepare the next branch shield while the current line backtracks."""
    branch = transfer_branch
    if (
        phase not in {
            SimulationPhase.PRESSURE_PUSH,
            SimulationPhase.FLOW_BACKTRACK,
        }
        or branch is None
        or branch == active_branch
    ):
        return
    existing = get_pre_shepherds(robots, branch)
    if not existing and capture_region_ready_for_shepherd(
        robots,
        branch,
    ):
        select_pre_shepherds(robots, branch, grid)
    if get_pre_shepherds(robots, branch):
        enforce_pre_shepherd_curtain_for_swarm(robots)
        update_pre_shepherd_pack_readiness(
            robots,
            branch,
            reference_density,
            dt,
        )


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
        robot.shepherd_branch = None
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


def base_front_equilibrium_weight(robot: "Robot") -> float:
    """Continuous membership of the upper Base/Junction swarm frontier.

    The Junction part is the union of the incoming upper front and the
    branch-facing front.  In particular, a horizontal transfer must not reduce
    the eligible Junction cohort to only the few robots nearest the side gate.
    """
    if robot.role != "NORMAL" or robot.base_reserve:
        return 0.0
    region = get_robot_region(robot.position)
    if region == "BOTTOM":
        depth_from_top = max(
            0.0,
            robot.position.y - bottom_rect.top,
        )
        return 1.0 - smoothstep01(
            depth_from_top
            / max(BASE_FRONT_BAND_DEPTH, EPSILON)
        )
    if region == "JUNCTION":
        incoming_front = smoothstep01(
            (
                junction_rect.bottom - robot.position.y
            )
            / max(junction_rect.height, EPSILON)
        )
        branch_direction = BRANCH_DIRECTIONS[active_branch]
        axial_position = (
            robot.position
            - pygame.Vector2(center_x, center_y)
        ).dot(branch_direction)
        rear_extent = SMOOTHING_LENGTH * 0.75
        branch_front = smoothstep01(
            (
                axial_position + rear_extent
            )
            / max(
                half_width + rear_extent,
                EPSILON,
            )
        )
        return max(incoming_front, branch_front)
    return 0.0


def branch_tail_equilibrium_weight(robot: "Robot") -> float:
    """Continuous membership of the active Branch's rear/mouth cohort."""
    if (
        robot.role != "NORMAL"
        or robot.base_reserve
        or get_robot_region(robot.position) != active_branch
    ):
        return 0.0
    depth = branch_depth_from_junction(
        robot.position,
        active_branch,
    )
    if depth > BASE_FRONT_BRANCH_TAIL_DEPTH:
        return 0.0
    return 1.0 - smoothstep01(
        depth / max(BASE_FRONT_BRANCH_TAIL_DEPTH, EPSILON)
    )


def collective_equilibrium_activation() -> float:
    """Keep a weak cross-section active before the gap controller reacts."""
    control = smoothstep01(transfer_gap_control)
    if phase == SimulationPhase.FILL_BEHIND_SHEPHERD:
        floor = BASE_FRONT_FILL_ACTIVATION_FLOOR
    elif (
        phase in {
            SimulationPhase.PRESSURE_PUSH,
            SimulationPhase.FLOW_BACKTRACK,
        }
        and transfer_branch is not None
        and not final_base_transfer_active
    ):
        floor = BASE_FRONT_HANDOFF_ACTIVATION_FLOOR
    else:
        return 0.0
    return floor + (1.0 - floor) * control


def adaptive_equilibrium_radius(robot: "Robot") -> float:
    """Phase-local equilibrium spacing used by pair repulsion."""
    if (
        phase == SimulationPhase.RETURN_TO_BASE
        and robot.role == "NORMAL"
        and get_robot_region(robot.position) in {"JUNCTION", "BOTTOM"}
    ):
        return max(
            ROBOT_RADIUS * 2.05,
            SAFE_RADIUS * RETURN_PACKED_EQUILIBRIUM_SCALE,
        )
    if (
        phase in {
            SimulationPhase.MOVE_TO_JUNCTION,
            SimulationPhase.EXPLORE_BRANCH,
        }
        and robot.role == "NORMAL"
        and simulation_time
        < BASE_COMPRESSION_DURATION
        + BASE_EQUILIBRIUM_RELEASE_DURATION
    ):
        release_progress = smoothstep01(
            (
                simulation_time - BASE_COMPRESSION_DURATION
            )
            / max(BASE_EQUILIBRIUM_RELEASE_DURATION, EPSILON)
        )
        equilibrium_scale = (
            BASE_PACKED_EQUILIBRIUM_SCALE
            + (
                NORMAL_EQUILIBRIUM_SCALE
                - BASE_PACKED_EQUILIBRIUM_SCALE
            )
            * release_progress
        )
        return max(
            ROBOT_RADIUS * 2.05,
            SAFE_RADIUS * equilibrium_scale,
        )
    branch_feed_phase = phase in {
        SimulationPhase.EXPLORE_BRANCH,
        SimulationPhase.FORM_SHEPHERD_BOUNDARY,
        SimulationPhase.FILL_BEHIND_SHEPHERD,
    }
    region = get_robot_region(robot.position)
    if branch_feed_phase and robot.role == "NORMAL":
        if (
            phase == SimulationPhase.FILL_BEHIND_SHEPHERD
            and robot_is_in_shepherd_packing_zone(
                robot,
                active_branch,
            )
        ):
            return max(
                ROBOT_RADIUS * 2.05,
                SAFE_RADIUS * SHEPHERD_PACKED_EQUILIBRIUM_SCALE,
            )
        if region == active_branch:
            return max(
                ROBOT_RADIUS * 2.05,
                branch_fill_equilibrium_spacing(active_branch),
            )
        if (
            phase == SimulationPhase.FILL_BEHIND_SHEPHERD
            and region in {"JUNCTION", "BOTTOM"}
        ):
            continuity_expansion = (
                FILL_TAIL_EQUILIBRIUM_EXPANSION
                * smoothstep01(transfer_gap_control)
            )
            return (
                SAFE_RADIUS
                * JUNCTION_TAIL_EQUILIBRIUM_SCALE
                * (1.0 + continuity_expansion)
            )
    if (
        phase in {
            SimulationPhase.PRESSURE_PUSH,
            SimulationPhase.FLOW_BACKTRACK,
        }
        and transfer_branch is not None
        and not final_base_transfer_active
    ):
        front_weight = base_front_equilibrium_weight(robot)
        if front_weight > 0.0:
            expansion = (
                BASE_FRONT_EQUILIBRIUM_EXPANSION
                * front_weight
                * collective_equilibrium_activation()
            )
            return (
                SAFE_RADIUS
                * NORMAL_EQUILIBRIUM_SCALE
                * (1.0 + expansion)
            )
    return SAFE_RADIUS * NORMAL_EQUILIBRIUM_SCALE


def get_initial_release_event_decay() -> float:
    """Smoothly remove stored Base energy after RIGHT flow is established."""
    if initial_release_event_time is None:
        return 1.0
    elapsed = simulation_time - initial_release_event_time
    return 1.0 - smoothstep01(
        elapsed / max(INITIAL_RELEASE_EVENT_DECAY_TIME, EPSILON)
    )


def get_initial_release_cruise_blend() -> float:
    """Blend sustained branch EDF from the flow event, with a time fallback."""
    if initial_release_event_time is not None:
        elapsed = simulation_time - initial_release_event_time
        return smoothstep01(
            elapsed / max(INITIAL_RELEASE_EDF_BLEND_TIME, EPSILON)
        )
    if (
        simulation_time
        >= BASE_COMPRESSION_DURATION + BASE_EXPANSION_BOOST_DURATION
    ):
        return 1.0
    return 0.0


def update_initial_release_flow_event(
    robots: list["Robot"],
    dt: float,
) -> bool:
    """End the Base pulse when a sustained connected RIGHT flux is observed."""
    global initial_release_flow_dwell, initial_release_event_time
    global initial_release_flow_count, initial_release_flow_ratio
    global initial_release_average_speed

    if initial_release_event_time is not None:
        return True
    event_window_open = (
        phase == SimulationPhase.EXPLORE_BRANCH
        and active_branch == "RIGHT"
        and branch_gate_states.get("RIGHT") == "OPEN"
        and simulation_time >= BASE_COMPRESSION_DURATION
    )
    if not event_window_open:
        initial_release_flow_dwell = 0.0
        initial_release_flow_count = 0
        initial_release_flow_ratio = 0.0
        initial_release_average_speed = 0.0
        return False

    entrants = [
        robot
        for robot in robots
        if (
            robot.role == "NORMAL"
            and robot.connected_to_base
            and not robot.base_reserve
            and get_robot_region(robot.position) == "RIGHT"
        )
    ]
    forward_speeds = [
        robot.velocity.dot(BRANCH_DIRECTIONS["RIGHT"])
        for robot in entrants
    ]
    moving_count = sum(
        speed >= INITIAL_RELEASE_FLOW_SPEED_THRESHOLD
        for speed in forward_speeds
    )
    initial_release_flow_count = len(entrants)
    initial_release_flow_ratio = (
        moving_count / max(len(entrants), 1)
    )
    initial_release_average_speed = (
        sum(max(0.0, speed) for speed in forward_speeds)
        / max(len(entrants), 1)
    )
    flow_established = (
        initial_release_flow_count
        >= INITIAL_RELEASE_FLOW_MIN_ROBOTS
        and initial_release_flow_ratio
        >= INITIAL_RELEASE_FLOW_RATIO_THRESHOLD
        and initial_release_average_speed
        >= INITIAL_RELEASE_FLOW_AVERAGE_SPEED_THRESHOLD
    )
    initial_release_flow_dwell = (
        initial_release_flow_dwell + dt
        if flow_established
        else 0.0
    )
    if (
        initial_release_flow_dwell
        < INITIAL_RELEASE_FLOW_DWELL_TIME
    ):
        return False

    initial_release_event_time = simulation_time
    print(
        "[Initial Release] RIGHT flow established; "
        f"robots={initial_release_flow_count}, "
        f"moving={initial_release_flow_ratio:.2f}, "
        f"speed={initial_release_average_speed:.2f}"
    )
    return True


def get_base_pressure_scale() -> float:
    """Continuous pulse that yields to sustained RIGHT flow when established."""
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
    timed_scale = pressure_scale + (
        SPH_MOTION_PRESSURE_BOOST - pressure_scale
    ) * fall
    event_decay = get_initial_release_event_decay()
    return (
        SPH_MOTION_PRESSURE_BOOST
        + (timed_scale - SPH_MOTION_PRESSURE_BOOST)
        * event_decay
    )


def initial_pressure_release_active() -> bool:
    """Keep shock guards until timeout or the event-triggered decay finishes."""
    if phase not in {
        SimulationPhase.MOVE_TO_JUNCTION,
        SimulationPhase.EXPLORE_BRANCH,
    }:
        return False
    if simulation_time < BASE_COMPRESSION_DURATION:
        return False
    if initial_release_event_time is not None:
        return (
            simulation_time - initial_release_event_time
            < INITIAL_RELEASE_EVENT_DECAY_TIME
        )
    return (
        simulation_time
        < BASE_COMPRESSION_DURATION + BASE_EXPANSION_BOOST_DURATION
    )


def get_base_compression_envelope() -> float:
    """Fast smooth rise, sustained compression, then a smooth release."""
    progress = clamp(
        simulation_time / max(BASE_COMPRESSION_DURATION, EPSILON),
        0.0,
        1.0,
    )
    rise = smoothstep01(
        progress / max(BASE_COMPRESSION_RISE_FRACTION, EPSILON)
    )
    fall = 1.0 - smoothstep01(
        (
            progress - BASE_COMPRESSION_FALL_START_FRACTION
        )
        / max(
            1.0 - BASE_COMPRESSION_FALL_START_FRACTION,
            EPSILON,
        )
    )
    return rise * fall


def get_stored_compression_pressure_envelope() -> float:
    """Stored compression energy retained as a decaying SPH pressure floor."""
    elapsed = simulation_time - BASE_COMPRESSION_DURATION
    if elapsed <= 0.0 or elapsed >= BASE_STORED_PRESSURE_DURATION:
        return 0.0
    rise = smoothstep01(
        elapsed / max(BASE_STORED_PRESSURE_RISE_TIME, EPSILON)
    )
    decay = 1.0 - smoothstep01(
        (
            elapsed - BASE_STORED_PRESSURE_DECAY_START
        )
        / max(
            BASE_STORED_PRESSURE_DURATION
            - BASE_STORED_PRESSURE_DECAY_START,
            EPSILON,
        )
    )
    return rise * decay * get_initial_release_event_decay()


def get_base_piston_reaction_envelope() -> float:
    """Fast release followed by a bounded decay of the Base-wall reaction."""
    elapsed = simulation_time - BASE_COMPRESSION_DURATION
    if elapsed <= 0.0 or elapsed >= BASE_PISTON_REACTION_DURATION:
        return 0.0
    rise = smoothstep01(
        elapsed / max(BASE_PISTON_REACTION_RISE_TIME, EPSILON)
    )
    decay = 1.0 - smoothstep01(
        (
            elapsed - BASE_PISTON_REACTION_RISE_TIME
        )
        / max(
            BASE_PISTON_REACTION_DURATION
            - BASE_PISTON_REACTION_RISE_TIME,
            EPSILON,
        )
    )
    return rise * decay * get_initial_release_event_decay()


def compute_base_piston_reaction_force(
    robot: "Robot",
) -> pygame.Vector2:
    """Convert compressed SPH pressure at the Base wall into upward thrust."""
    if (
        phase not in {
            SimulationPhase.MOVE_TO_JUNCTION,
            SimulationPhase.EXPLORE_BRANCH,
        }
        or robot.role != "NORMAL"
        or get_robot_region(robot.position) != "BOTTOM"
    ):
        return pygame.Vector2()
    envelope = get_base_piston_reaction_envelope()
    if envelope <= 0.0:
        return pygame.Vector2()
    depth_fraction = clamp(
        (robot.position.y - bottom_rect.top)
        / max(bottom_rect.height, EPSILON),
        0.0,
        1.0,
    )
    wall_weight = smoothstep01(
        (
            depth_fraction - BASE_PISTON_REACTION_DEPTH_START
        )
        / max(
            1.0 - BASE_PISTON_REACTION_DEPTH_START,
            EPSILON,
        )
    )
    normalized_pressure = (
        robot.pressure
        / max(PRESSURE_GAIN * robot.density, EPSILON)
    )
    magnitude = min(
        BASE_PISTON_REACTION_FORCE_LIMIT,
        BASE_PISTON_REACTION_GAIN
        * normalized_pressure
        * wall_weight
        * envelope,
    )
    return pygame.Vector2(0.0, -magnitude)


def compute_initial_junction_soft_wall_force(
    robot: "Robot",
) -> pygame.Vector2:
    """Contain natural probes only after they traverse a short branch depth.

    The old barrier was placed on the three Junction faces and therefore
    suppressed the very lateral expansion needed for distribution inference.
    This safety valve sits beyond the local probe distance: robots may form and
    validate directional cohorts, but cannot drain deeply before DFS commits.
    """
    if (
        phase != SimulationPhase.MOVE_TO_JUNCTION
        or robot.role != "NORMAL"
    ):
        return pygame.Vector2()
    region = get_robot_region(robot.position)
    if region not in BRANCHES:
        return pygame.Vector2()
    branch_depth = branch_depth_from_junction(robot.position, region)
    if branch_depth <= JUNCTION_PROBE_DEPTH:
        return pygame.Vector2()
    outward = BRANCH_DIRECTIONS[region]
    overshoot = branch_depth - JUNCTION_PROBE_DEPTH
    ratio = clamp(
        overshoot / max(INITIAL_JUNCTION_SOFT_WALL_DEPTH, EPSILON),
        0.0,
        1.0,
    )
    outward_speed = max(0.0, robot.velocity.dot(outward))
    return -outward * (
        JUNCTION_PROBE_BARRIER_GAIN * ratio**2
        + INITIAL_JUNCTION_SOFT_WALL_DAMPING * outward_speed
    )


def compute_pressures(robots, reference_density):
    effective_lambda = get_effective_stiffness_exponent()
    exploration_floor_ramp = (
        smoothstep01(
            branch_entry_timer
            / max(EXPLORATION_PRESSURE_FLOOR_RAMP_TIME, EPSILON)
        )
        if phase == SimulationPhase.EXPLORE_BRANCH
        else 0.0
    )
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
        if robot.role in {"NORMAL", "FRONTIER_SHEPHERD"}:
            if phase == SimulationPhase.MOVE_TO_JUNCTION:
                robot.pressure *= get_base_pressure_scale()
            elif phase == SimulationPhase.EXPLORE_BRANCH:
                robot.pressure *= (
                    get_base_pressure_scale()
                    if simulation_time
                    < BASE_COMPRESSION_DURATION
                    + BASE_EXPANSION_BOOST_DURATION
                    else SPH_MOTION_PRESSURE_BOOST
                )
                if (
                    not robot.base_reserve
                    and get_robot_region(robot.position)
                    in {"JUNCTION", active_branch}
                ):
                    exploration_pressure_floor = (
                        PRESSURE_GAIN
                        * robot.density
                        * EXPLORATION_PRESSURE_FLOOR
                        * exploration_floor_ramp
                    )
                    robot.pressure = max(
                        robot.pressure,
                        exploration_pressure_floor,
                    )
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
            elif phase == SimulationPhase.RETURN_TO_BASE:
                robot.pressure *= RETURN_PACKING_PRESSURE_SCALE
            if phase in {
                SimulationPhase.MOVE_TO_JUNCTION,
                SimulationPhase.EXPLORE_BRANCH,
            }:
                stored_pressure_floor = (
                    PRESSURE_GAIN
                    * robot.density
                    * BASE_STORED_PRESSURE_FLOOR
                    * get_stored_compression_pressure_envelope()
                )
                robot.pressure = max(
                    robot.pressure,
                    stored_pressure_floor,
                )
            if (
                phase in {
                    SimulationPhase.PRESSURE_PUSH,
                    SimulationPhase.FLOW_BACKTRACK,
                }
                and transfer_branch is not None
                and robot.transfer_target == transfer_branch
                and get_robot_region(robot.position)
                in {"JUNCTION", transfer_branch}
            ):
                transfer_ramp = (
                    1.0
                    if phase == SimulationPhase.FLOW_BACKTRACK
                    else smoothstep01(
                        pressure_push_timer
                        / max(PRESSURE_RAMP_TIME, EPSILON)
                    )
                )
                transfer_pressure_floor = (
                    PRESSURE_GAIN
                    * robot.density
                    * TRANSFER_PRESSURE_FLOOR
                    * transfer_ramp
                )
                robot.pressure = max(
                    robot.pressure,
                    transfer_pressure_floor,
                )
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


def update_transfer_continuity_control(robots) -> None:
    """Balance source recovery against target advance and mouth occupancy."""
    global transfer_path_max_gap
    global transfer_entrance_count
    global transfer_gap_control
    global transfer_target_motion_scale
    global branch_fill_target_count
    global branch_fill_current_count
    global branch_fill_deficit_control

    filling_active = phase == SimulationPhase.FILL_BEHIND_SHEPHERD
    quota_feed_active = phase in {
        SimulationPhase.EXPLORE_BRANCH,
        SimulationPhase.FORM_SHEPHERD_BOUNDARY,
        SimulationPhase.FILL_BEHIND_SHEPHERD,
    }
    handoff_active = (
        phase in {
            SimulationPhase.PRESSURE_PUSH,
            SimulationPhase.FLOW_BACKTRACK,
        }
        and transfer_branch is not None
        and not final_base_transfer_active
    )
    if quota_feed_active:
        branch_fill_target_count = calculate_branch_fill_quota(
            robots,
            active_branch,
        )
        branch_fill_current_count = sum(
            robot.role == "NORMAL"
            and get_robot_region(robot.position) == active_branch
            for robot in robots
        )
        raw_fill_deficit = clamp(
            (
                branch_fill_target_count
                - branch_fill_current_count
            )
            / max(branch_fill_target_count, 1),
            0.0,
            1.0,
        )
        branch_fill_deficit_control = (
            branch_fill_deficit_control
            * (1.0 - BRANCH_FILL_QUOTA_FILTER_ALPHA)
            + raw_fill_deficit
            * BRANCH_FILL_QUOTA_FILTER_ALPHA
        )
    else:
        branch_fill_target_count = 0
        branch_fill_current_count = 0
        branch_fill_deficit_control = 0.0

    if not (filling_active or handoff_active):
        transfer_path_max_gap = 0.0
        transfer_entrance_count = 0
        transfer_gap_control = 0.0
        transfer_target_motion_scale = 1.0
        return

    path_regions = {"BOTTOM", "JUNCTION", active_branch}
    path_progress = sorted(
        relay_path_progress(robot.position, active_branch)
        for robot in robots
        if (
            robot.role == "NORMAL"
            and not robot.base_reserve
            and get_robot_region(robot.position) in path_regions
        )
    )
    transfer_path_max_gap = max(
        (
            after - before
            for before, after in zip(
                path_progress,
                path_progress[1:],
            )
        ),
        default=0.0,
    )
    entrance_robots = [
        robot
        for robot in robots
        if (
            robot.role == "NORMAL"
            and not robot.base_reserve
            and get_robot_region(robot.position) == active_branch
            and branch_depth_from_junction(
                robot.position,
                active_branch,
            )
            <= TRANSFER_CONTINUITY_ENTRANCE_DEPTH
        )
    ]
    transfer_entrance_count = len(entrance_robots)
    gap_control = clamp(
        (
            transfer_path_max_gap
            - TRANSFER_CONTINUITY_TARGET_GAP
        )
        / max(
            TRANSFER_CONTINUITY_DANGER_GAP
            - TRANSFER_CONTINUITY_TARGET_GAP,
            EPSILON,
        ),
        0.0,
        1.0,
    )
    entrance_control = clamp(
        1.0
        - transfer_entrance_count
        / max(
            TRANSFER_CONTINUITY_MIN_ENTRANCE_ROBOTS,
            1,
        ),
        0.0,
        1.0,
    )
    raw_control = max(gap_control, entrance_control)
    transfer_gap_control = (
        transfer_gap_control
        * (1.0 - TRANSFER_CONTINUITY_FILTER_ALPHA)
        + raw_control * TRANSFER_CONTINUITY_FILTER_ALPHA
    )
    shaped_control = smoothstep01(transfer_gap_control)
    minimum_motion_scale = (
        FILL_BRANCH_CRUISE_MIN_SCALE
        if filling_active
        else TRANSFER_CONTINUITY_MIN_TARGET_SCALE
    )
    transfer_target_motion_scale = (
        1.0
        - (
            1.0
            - minimum_motion_scale
        )
        * shaped_control
    )


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
    # Once ordering selects one branch, NORMAL robots in another branch retain
    # their locally learned inward heading. The physical mouth guard reinforces
    # this command at close range and keeps the communication chain present.
    if (
        phase in {
            SimulationPhase.EXPLORE_BRANCH,
            SimulationPhase.FORM_SHEPHERD_BOUNDARY,
            SimulationPhase.FILL_BEHIND_SHEPHERD,
            SimulationPhase.PRESSURE_PUSH,
            SimulationPhase.FLOW_BACKTRACK,
        }
        and region in BRANCHES
        and region != active_branch
        and robot.role == "NORMAL"
    ):
        return (
            normalized_direction_toward(
                robot.position,
                get_branch_entrance(region),
            )
            * OUTLET_FORCE
        )
    independent = {
        SimulationPhase.MOVE_TO_JUNCTION,
        SimulationPhase.FORM_JUNCTION_GUARDS,
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
            if y_distance > 0.0 and region in {"BOTTOM", "JUNCTION"}:
                scale = max(
                    INITIAL_INGRESS_MIN_FORCE_SCALE,
                    min(
                        1.0,
                        y_distance / INITIAL_INGRESS_BRAKE_DISTANCE,
                    ),
                )
                route_multiplier = (
                    INITIAL_JUNCTION_PROBE_ROUTE_MULTIPLIER
                    if region == "JUNCTION"
                    else 1.0
                )
                force.y = (
                    -INITIAL_INGRESS_FORCE * scale * route_multiplier
                )
            if region == "BOTTOM":
                lane_error = robot.ingress_lane_x - robot.position.x
                force.x = clamp(
                    INITIAL_INGRESS_LANE_GAIN * lane_error,
                    -INITIAL_INGRESS_LANE_MAX_FORCE,
                    INITIAL_INGRESS_LANE_MAX_FORCE,
                )
    elif phase == SimulationPhase.FORM_JUNCTION_GUARDS:
        if region in BRANCHES:
            force = normalized_direction_toward(
                robot.position,
                junction_target,
            ) * OUTLET_FORCE
        elif region == "BOTTOM":
            force = normalized_direction_toward(
                robot.position,
                JUNCTION_STAGING_POSITION,
            ) * WEAK_BRANCH_BIAS_FORCE
        else:
            force = normalized_direction_toward(
                robot.position,
                JUNCTION_STAGING_POSITION,
            ) * WEAK_BRANCH_BIAS_FORCE
    elif phase == SimulationPhase.EXPLORE_BRANCH:
        if (
            transfer_branch == active_branch
            and robot.transfer_target != active_branch
            and region not in {"BOTTOM", "JUNCTION"}
        ):
            # Base-side robots receive no artificial suction. With no
            # Shepherd behind them, the low-density tail remains near Base.
            force = pygame.Vector2()
        elif region in {"BOTTOM", "JUNCTION"}:
            quota_feed_scale = (
                0.65
                + BRANCH_FILL_QUOTA_SOURCE_FORCE_BOOST
                * branch_fill_deficit_control
            )
            force = (
                get_lane_preserving_cohort_direction(
                    robot,
                    active_branch,
                )
                * OUTLET_FORCE
                * quota_feed_scale
            )
        else:
            force = (
                get_lane_preserving_cohort_direction(
                    robot,
                    active_branch,
                )
                * WEAK_BRANCH_BIAS_FORCE
                * relay_motion_scale
            )
    elif phase == SimulationPhase.FORM_SHEPHERD_BOUNDARY:
        if (
            transfer_branch == active_branch
            and robot.transfer_target != active_branch
            and region not in {"BOTTOM", "JUNCTION"}
        ):
            force = pygame.Vector2()
        elif robot.role == "SHEPHERD":
            force = pygame.Vector2()
        elif region == active_branch:
            # Keep the original behavior: ordinary branch robots wait while the
            # selected Shepherds arrange themselves across the corridor.
            force = pygame.Vector2()
        else:
            force = (
                get_lane_preserving_cohort_direction(
                    robot,
                    active_branch,
                    get_shepherd_fill_target(active_branch),
                )
                * OUTLET_FORCE
                * (
                    1.0
                    + BRANCH_FILL_QUOTA_SOURCE_FORCE_BOOST
                    * branch_fill_deficit_control
                )
            )
    elif phase == SimulationPhase.FILL_BEHIND_SHEPHERD:
        if (
            transfer_branch == active_branch
            and robot.transfer_target != active_branch
            and region not in {"BOTTOM", "JUNCTION"}
        ):
            force = pygame.Vector2()
        elif robot.role == "SHEPHERD":
            force = pygame.Vector2()
        else:
            fill_direction = get_lane_preserving_cohort_direction(
                robot,
                active_branch,
                get_shepherd_fill_target(active_branch),
            )
            fill_feed_ramp = smoothstep01(
                branch_entry_timer
                / max(FILL_FEED_RAMP_TIME, EPSILON)
            )
            if region == active_branch:
                force = (
                    fill_direction
                    * ROUTE_FORCE
                    * relay_motion_scale
                    * SHEPHERD_FILL_FORCE_MULTIPLIER
                    * transfer_target_motion_scale
                )
            elif region == "BOTTOM":
                force = (
                    fill_direction
                    * OUTLET_FORCE
                    * FILL_BASE_FEED_MULTIPLIER
                    * fill_feed_ramp
                    * (
                        1.0
                        + BRANCH_FILL_QUOTA_SOURCE_FORCE_BOOST
                        * branch_fill_deficit_control
                    )
                    * (
                        1.0
                        + FILL_TAIL_FEED_BOOST
                        * smoothstep01(transfer_gap_control)
                    )
                )
            elif region == "JUNCTION":
                force = (
                    fill_direction
                    * OUTLET_FORCE
                    * FILL_JUNCTION_FEED_MULTIPLIER
                    * fill_feed_ramp
                    * (
                        1.0
                        + BRANCH_FILL_QUOTA_SOURCE_FORCE_BOOST
                        * branch_fill_deficit_control
                    )
                    * (
                        1.0
                        + FILL_TAIL_FEED_BOOST
                        * smoothstep01(transfer_gap_control)
                    )
                )
            else:
                force = fill_direction * OUTLET_FORCE
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
            force = (
                get_backtrack_direction(active_branch)
                * PRESSURE_BACKTRACK_BODY_FORCE
                * (
                    1.0
                    + TRANSFER_CONTINUITY_SOURCE_FORCE_BOOST
                    * transfer_gap_control
                )
            )
        elif (
            transfer_branch is not None
            and robot.transfer_target == transfer_branch
            and region != active_branch
        ):
            transfer_direction, follow_weight = (
                get_collective_transfer_direction(robot)
            )
            route_scale = clamp(
                transfer_target_motion_scale
                + BASE_FRONT_FOLLOW_FORCE_BOOST
                * follow_weight,
                TRANSFER_CONTINUITY_MIN_TARGET_SCALE,
                BASE_FRONT_FOLLOW_MAX_ROUTE_SCALE,
            )
            force = (
                transfer_direction
                * ROUTE_FORCE
                * route_scale
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
            if robot.transfer_target != transfer_branch:
                force = pygame.Vector2()
            elif region == active_branch:
                force = (
                    geodesic_edf_direction(
                        robot.position,
                        transfer_branch,
                    )
                    * ROUTE_FORCE
                    * (
                        1.0
                        + TRANSFER_CONTINUITY_SOURCE_FORCE_BOOST
                        * transfer_gap_control
                    )
                )
            else:
                transfer_direction, follow_weight = (
                    get_collective_transfer_direction(robot)
                )
                route_scale = clamp(
                    transfer_target_motion_scale
                    + BASE_FRONT_FOLLOW_FORCE_BOOST
                    * follow_weight,
                    TRANSFER_CONTINUITY_MIN_TARGET_SCALE,
                    BASE_FRONT_FOLLOW_MAX_ROUTE_SCALE,
                )
                force = (
                    transfer_direction
                    * ROUTE_FORCE
                    * route_scale
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
            force = (
                normalized_direction_toward(
                    robot.position,
                    junction_target,
                )
                * OUTLET_FORCE
                * RETURN_STRAGGLER_FORCE_MULTIPLIER
            )
        elif region == "JUNCTION":
            force = (
                direction_toward_base_path(robot.position)
                * RETURN_EGRESS_FORCE
                * RETURN_STRAGGLER_FORCE_MULTIPLIER
            )
            lane_error = robot.ingress_lane_x - robot.position.x
            force.x += clamp(
                RETURN_LANE_GAIN * lane_error,
                -RETURN_LANE_MAX_FORCE,
                RETURN_LANE_MAX_FORCE,
            )
        else:
            bottom_target = get_bottom_hold_point()
            y_distance = bottom_target.y - robot.position.y
            if y_distance > 0.0:
                scale = max(RETURN_MIN_FORCE_SCALE, min(1.0, y_distance / RETURN_BRAKE_DISTANCE))
                force.y = RETURN_EGRESS_FORCE * scale
            lane_error = robot.ingress_lane_x - robot.position.x
            force.x = clamp(RETURN_LANE_GAIN * lane_error, -RETURN_LANE_MAX_FORCE, RETURN_LANE_MAX_FORCE)

    force += compute_backtrack_bridge_force(robot)

    cohort_phase = phase in {
        SimulationPhase.EXPLORE_BRANCH,
        SimulationPhase.FORM_SHEPHERD_BOUNDARY,
        SimulationPhase.FILL_BEHIND_SHEPHERD,
    }
    lane_offset = (
        get_cohort_lane_offset(robot)
        if cohort_phase
        and (
            region in {"BOTTOM", "JUNCTION"}
            or region == active_branch
        )
        else 0.0
    )
    if region in {"UP", "BOTTOM"}:
        lateral_target = center_x + (
            lane_offset
            if region == "BOTTOM" or region == active_branch
            else 0.0
        )
        force.x += CENTERING_GAIN * (
            lateral_target - robot.position.x
        )
    elif region in {"LEFT", "RIGHT"}:
        lateral_target = center_y + (
            lane_offset if region == active_branch else 0.0
        )
        lateral_error = lateral_target - robot.position.y
        if (
            phase == SimulationPhase.FILL_BEHIND_SHEPHERD
            and robot.role == "NORMAL"
            and region == active_branch
        ):
            force.y += clamp(
                FILL_BRANCH_LANE_CENTERING_GAIN * lateral_error,
                -FILL_BRANCH_LANE_FORCE_LIMIT,
                FILL_BRANCH_LANE_FORCE_LIMIT,
            )
        else:
            force.y += CENTERING_GAIN * lateral_error
    return force


def compute_long_range_gap_attraction_force(
    robot: "Robot",
    spatial_grid,
) -> pygame.Vector2:
    """Join the broad Base/Junction frontier to the Branch rear as one body.

    No robot is elected or assigned a special role.  Every NORMAL robot in the
    continuous upper Base/Junction band and every NORMAL robot at the Branch
    rear receives an extended, reciprocal equilibrium interaction.  Candidate
    peers are matched by their deployment-lane coordinate rather than their
    current screen coordinate, so a 90-degree turn forms parallel links across
    the corridor width instead of concentrating them at one corner.
    """
    filling_active = phase == SimulationPhase.FILL_BEHIND_SHEPHERD
    handoff_active = (
        phase in {
            SimulationPhase.PRESSURE_PUSH,
            SimulationPhase.FLOW_BACKTRACK,
        }
        and transfer_branch is not None
        and not final_base_transfer_active
    )
    if (
        not (filling_active or handoff_active)
        or robot.role != "NORMAL"
        or robot.base_reserve
    ):
        return pygame.Vector2()

    front_weight = base_front_equilibrium_weight(robot)
    tail_weight = branch_tail_equilibrium_weight(robot)
    endpoint_weight = max(front_weight, tail_weight)
    control_weight = collective_equilibrium_activation()
    if endpoint_weight <= 0.0 or control_weight <= EPSILON:
        return pygame.Vector2()

    radius_sq = BASE_FRONT_EQUILIBRIUM_RADIUS**2
    center_cell_x, center_cell_y = cell_key(robot.position)
    cell_reach = max(
        1,
        math.ceil(
            BASE_FRONT_EQUILIBRIUM_RADIUS / CELL_SIZE
        ),
    )
    candidates = []
    for other in (
        other_robot
        for cell_dx in range(-cell_reach, cell_reach + 1)
        for cell_dy in range(-cell_reach, cell_reach + 1)
        for other_robot in spatial_grid.get(
            (
                center_cell_x + cell_dx,
                center_cell_y + cell_dy,
            ),
            (),
        )
    ):
        if (
            other is robot
            or other.role != "NORMAL"
            or other.base_reserve
        ):
            continue
        other_front_weight = base_front_equilibrium_weight(other)
        other_tail_weight = branch_tail_equilibrium_weight(other)
        if front_weight > 0.0:
            peer_weight = other_tail_weight
        else:
            peer_weight = other_front_weight
        if peer_weight <= 0.0:
            continue
        distance_sq = robot.position.distance_squared_to(
            other.position,
        )
        if (
            distance_sq > radius_sq
            or not has_line_of_sight(
                robot.position,
                other.position,
            )
        ):
            continue
        lane_offset = abs(
            get_cohort_lane_offset(other)
            - get_cohort_lane_offset(robot)
        )
        candidates.append(
            (
                lane_offset,
                distance_sq,
                other.robot_id,
                peer_weight,
                other,
            )
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        )
    )
    linked_candidates = candidates[
        :BASE_FRONT_LINKS_PER_ROBOT
    ]
    if not linked_candidates:
        return pygame.Vector2()

    force = pygame.Vector2()
    total_peer_weight = 0.0
    taper_start = (
        BASE_FRONT_EQUILIBRIUM_RADIUS
        * BASE_FRONT_EQUILIBRIUM_TAPER_START_RATIO
    )
    for _, _, _, peer_weight, other in linked_candidates:
        delta = other.position - robot.position
        distance = delta.length()
        if distance <= EPSILON:
            continue
        taper = (
            1.0
            if distance <= taper_start
            else 1.0 - smoothstep01(
                (distance - taper_start)
                / max(
                    BASE_FRONT_EQUILIBRIUM_RADIUS
                    - taper_start,
                    EPSILON,
                )
            )
        )
        stretch = max(
            0.0,
            distance - BASE_FRONT_EQUILIBRIUM_DISTANCE,
        )
        direction = delta / distance
        radial_peer_speed = (
            other.velocity - robot.velocity
        ).dot(direction)
        link_strength = max(
            0.0,
            BASE_FRONT_EQUILIBRIUM_GAIN * stretch
            + BASE_FRONT_EQUILIBRIUM_DAMPING_GAIN
            * radial_peer_speed,
        )
        force += (
            direction
            * link_strength
            * taper
            * peer_weight
        )
        total_peer_weight += peer_weight
    force /= max(total_peer_weight, EPSILON)
    reaction_scale = (
        BASE_FRONT_TAIL_REACTION_SCALE
        if tail_weight > 0.0
        else 1.0
    )
    force *= endpoint_weight * control_weight * reaction_scale
    return limit_vector(
        force,
        BASE_FRONT_EQUILIBRIUM_FORCE_LIMIT,
    )


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
        if robot_i.role in {
            "ANCHOR",
            "RELAY",
            "TRUNK_RELAY",
            "JUNCTION_GUARD",
        }:
            robot_i.acceleration.update(0.0, 0.0)
            robot_i.last_sph_pressure_force = 0.0
            robot_i.last_compression_release_force = 0.0
            robot_i.last_shepherd_force = 0.0
            robot_i.last_base_piston_force = 0.0
            robot_i.last_edf_force = 0.0
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
            robot_i.last_base_piston_force = 0.0
            robot_i.last_edf_force = 0.0
            continue

        pressure_force = pygame.Vector2()
        viscosity_force = pygame.Vector2()
        viscoelastic_force = pygame.Vector2()
        compression_release_force = pygame.Vector2()
        repulsion_force = pygame.Vector2()
        physical_guard_force = pygame.Vector2()
        virtual_force = pygame.Vector2()
        contact_obstacle_force = pygame.Vector2()
        cohesion_force = pygame.Vector2()
        gap_attraction_force = pygame.Vector2()
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
                robot_i.role in {"NORMAL", "FRONTIER_SHEPHERD"}
                and robot_j.role in {"NORMAL", "FRONTIER_SHEPHERD"}
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

            # A JUNCTION_GUARD communicates only its branch-facing orientation.
            # A nearby NORMAL uses that local message and relative position to
            # move toward the Junction. Overlapping influence disks across the
            # physical line replace the old map-aware virtual wall and also
            # recover robots remaining on the outer side of the line.
            if (
                robot_i.role == "NORMAL"
                and robot_j.role == "JUNCTION_GUARD"
                and robot_j.junction_guard_branch in BRANCHES
                and distance < PHYSICAL_GUARD_INFLUENCE_RADIUS
            ):
                ratio = (
                    1.0
                    - distance / PHYSICAL_GUARD_INFLUENCE_RADIUS
                )
                branch_direction = BRANCH_DIRECTIONS[
                    robot_j.junction_guard_branch
                ]
                inward = -branch_direction
                lateral = pygame.Vector2(
                    -branch_direction.y,
                    branch_direction.x,
                )
                lateral_sign = (
                    1.0 if r_ij.dot(lateral) >= 0.0 else -1.0
                )
                physical_guard_force += (
                    inward * PHYSICAL_GUARD_INWARD_GAIN * ratio**2
                    + lateral
                    * lateral_sign
                    * PHYSICAL_GUARD_LATERAL_GAIN
                    * ratio**2
                )

        release_active = initial_pressure_release_active()
        pressure_force_limit = (
            INITIAL_RELEASE_PRESSURE_FORCE_LIMIT
            if release_active
            else SPH_PRESSURE_FORCE_LIMIT
        )
        limit_vector(pressure_force, pressure_force_limit)
        if release_active:
            viscosity_force *= INITIAL_RELEASE_VISCOSITY_MULTIPLIER
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
        limit_vector(
            physical_guard_force,
            PHYSICAL_GUARD_FORCE_LIMIT,
        )
        route_force = compute_route_force(robot_i)
        connectivity_force = compute_connectivity_force(
            robot_i,
            communication_grid,
        )
        shepherd_curtain_force = compute_shepherd_curtain_force(robot_i)
        pre_shepherd_curtain_force = (
            compute_pre_shepherd_curtain_force(robot_i)
        )
        initial_junction_wall_force = (
            compute_initial_junction_soft_wall_force(robot_i)
        )
        base_piston_force = compute_base_piston_reaction_force(robot_i)
        edf_force = compute_pressure_coupled_edf_force(robot_i)
        contact_obstacle_force = compute_contact_point_repulsion_force(robot_i)
        robot_i.last_sph_pressure_force = pressure_force.length()
        robot_i.last_compression_release_force = (
            compression_release_force.length()
        )
        robot_i.last_shepherd_force = (
            virtual_force
            + shepherd_curtain_force
            + pre_shepherd_curtain_force
        ).length()
        robot_i.last_base_piston_force = base_piston_force.length()
        robot_i.last_edf_force = edf_force.length()
        pressure_phase_normal = (
            phase == SimulationPhase.PRESSURE_PUSH
            and robot_i.role == "NORMAL"
            and get_robot_region(robot_i.position) == active_branch
        )
        if 0 < neighbor_count < ISOLATION_NEIGHBOR_THRESHOLD and not pressure_phase_normal:
            local_center = neighbor_center / neighbor_count
            direction = local_center - robot_i.position
            if direction.length_squared() > EPSILON:
                neighbor_deficit = (
                    ISOLATION_NEIGHBOR_THRESHOLD - neighbor_count
                ) / ISOLATION_NEIGHBOR_THRESHOLD
                density_deficit = clamp(
                    (
                        LOCAL_COHESION_DENSITY_TARGET_RATIO
                        - robot_i.density_ratio
                    )
                    / max(
                        LOCAL_COHESION_DENSITY_TARGET_RATIO,
                        EPSILON,
                    ),
                    0.0,
                    1.0,
                )
                attraction_gain = (
                    LOCAL_COHESION_GAIN
                    * neighbor_deficit
                    * (
                        1.0
                        + LOCAL_COHESION_DENSITY_BOOST
                        * density_deficit
                    )
                )
                cohesion_force = limit_vector(
                    direction.normalize() * attraction_gain,
                    LOCAL_COHESION_FORCE_LIMIT,
                )
        gap_attraction_force = (
            compute_long_range_gap_attraction_force(
                robot_i,
                communication_grid,
            )
        )
        if neighbor_count < ISOLATION_NEIGHBOR_THRESHOLD and not pressure_phase_normal:
            boost = (ISOLATION_NEIGHBOR_THRESHOLD - neighbor_count) / ISOLATION_NEIGHBOR_THRESHOLD
            route_force *= 1.0 + ISOLATION_ROUTE_BOOST * boost
        total = (
            pressure_force
            + viscosity_force
            + viscoelastic_force
            + compression_release_force
            + repulsion_force
            + physical_guard_force
            + virtual_force
            + contact_obstacle_force
            + cohesion_force
            + gap_attraction_force
            + route_force
            + connectivity_force
            + shepherd_curtain_force
            + pre_shepherd_curtain_force
            + initial_junction_wall_force
            + base_piston_force
            + edf_force
            - (
                DAMPING
                + (
                    INITIAL_RELEASE_EXTRA_DAMPING
                    if release_active
                    else 0.0
                )
            )
            * robot_i.velocity
        )
        acceleration_limit = (
            INITIAL_SAFE_MAX_ACCELERATION
            if (
                phase == SimulationPhase.MOVE_TO_JUNCTION
                or release_active
            )
            else MAX_ACCELERATION
        )
        raw_acceleration = limit_vector(total, acceleration_limit)
        filter_alpha = (
            INITIAL_RELEASE_ACCELERATION_FILTER_ALPHA
            if release_active
            else ACCELERATION_FILTER_ALPHA
        )
        robot_i.filtered_acceleration = (
            robot_i.filtered_acceleration
            * (1.0 - filter_alpha)
            + raw_acceleration * filter_alpha
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


def start_shepherd_pressure_push(robots, branch):
    """Start a prepared Shepherd piston without repeating the fill wait."""
    global phase, pressure_push_timer, flow_establish_timer
    global shepherd_flow_timer
    next_branch = next_unvisited_transfer_branch(branch)
    if next_branch is not None:
        begin_guarded_return_to_junction(robots, branch)
    else:
        begin_final_base_transfer(
            robots,
            branch,
        )
    phase = SimulationPhase.PRESSURE_PUSH
    pressure_push_timer = 0.0
    flow_establish_timer = 0.0
    shepherd_flow_timer = 0.0
    packed_count = len(tip_robots(robots, branch))
    metrics.pressure_events.append({
        "branch": branch,
        "started_at": simulation_time,
    })
    print(
        f"[Saturation] robots packed behind Shepherd boundary: "
        f"branch={branch}, count={packed_count}"
    )
    print("[Pressure] piston push started")


def update_simulation_state(robots, dt, reference_density, spatial_grid):
    global phase, shepherd_form_timer, pressure_push_timer, flow_establish_timer
    global shepherd_flow_timer
    global junction_switch_timer, final_gather_timer, branch_entry_timer
    global distributed_consensus_branch, transfer_branch
    global final_base_transfer_active
    global return_trunk_release_pending, return_trunk_retract_timer, return_trunk_last_released_id, return_trunk_force_timer
    global return_done_dwell, return_entry_stall_timer
    global return_last_bottom_count
    global draining_branch
    global junction_guard_formation_timer, junction_guard_stable_dwell
    global junction_guard_status
    global pending_branch_start

    update_draining_branch_gate(robots)
    update_anchor_entry_records(robots, simulation_time)
    update_backtrack_bridge_guards(robots, dt)

    if phase in {
        SimulationPhase.EXPLORE_BRANCH,
        SimulationPhase.FORM_SHEPHERD_BOUNDARY,
        SimulationPhase.FILL_BEHIND_SHEPHERD,
    }:
        branch_entry_timer += dt
    update_initial_release_flow_event(robots, dt)

    # NORMAL robots decide first. The elected Anchor only stores and
    # retransmits their consensus and gate state.
    anchor = junction_anchor

    if phase == SimulationPhase.MOVE_TO_JUNCTION:
        update_relay_deployment(robots, dt)
        junction_confirmed = junction_inference_tracker.update(robots, dt)
        if junction_confirmed:
            if anchor is None:
                anchor = elect_junction_anchor(robots)
            begin_junction_guard_formation(robots)
            phase = SimulationPhase.FORM_JUNCTION_GUARDS
            print(
                "[Junction] confirmed while mouths were OPEN; "
                "forming branch-wise full guards before ordering"
            )

    elif phase == SimulationPhase.FORM_JUNCTION_GUARDS:
        junction_guard_formation_timer += dt
        if pending_branch_start is not None:
            thick_walls_ready = thick_mouth_guards_formed(
                robots,
                pending_branch_start,
            )
            junction_guard_stable_dwell = (
                junction_guard_stable_dwell + dt
                if thick_walls_ready
                else 0.0
            )
            if (
                junction_guard_formation_timer
                >= THICK_MOUTH_GUARD_FORM_TIMEOUT
                and not thick_walls_ready
                and not junction_guard_status.startswith("WAITING_FOR_THICK")
            ):
                junction_guard_status = "WAITING_FOR_THICK_KHOP_MOUTH_WALLS"
                print(
                    "[Thick Mouth Guard] formation is slow; selected Branch "
                    "flow remains paused until every physical wall is ready"
                )
            if (
                junction_guard_stable_dwell
                < THICK_MOUTH_GUARD_FORM_DWELL
            ):
                return
            selected = pending_branch_start
            pending_branch_start = None
            junction_guard_status = (
                f"FRONTIER={selected};OTHERS=THICK_KHOP_WALLS_READY"
            )
            saturation_tracker.reset(selected)
            dead_end_inference_tracker.reset(selected)
            junction_consensus_tracker.reset()
            phase = SimulationPhase.EXPLORE_BRANCH
            metrics.branch_events.append({
                "branch": selected,
                "started_at": simulation_time,
            })
            print(
                f"[Thick Mouth Guard] all unselected walls ready; "
                f"starting {selected} exploration"
            )
            return
        formed = junction_guards_formed(robots)
        junction_guard_stable_dwell = (
            junction_guard_stable_dwell + dt if formed else 0.0
        )
        guards_ready = (
            junction_guard_stable_dwell >= JUNCTION_GUARD_FORM_DWELL
        )
        if not guards_ready:
            if (
                junction_guard_formation_timer >= JUNCTION_GUARD_FORM_TIMEOUT
                and junction_guard_status == "FORMING_FULL_GUARDS"
            ):
                junction_guard_status = "WAITING_FOR_PHYSICAL_FULL_GUARDS"
                print(
                    "[Junction Guard] formation is slow; branch ordering "
                    "remains blocked until every entrance guard is in place"
                )
            return
        if junction_guard_status != "FULL_GUARDS_READY":
            print(
                "[Junction Guard] every inferred branch entrance has a "
                "physical full guard; branch ordering may begin"
            )
        junction_guard_status = "FULL_GUARDS_READY"
        voted_branch = update_distributed_branch_consensus(
            robots,
            reference_density,
        )
        if voted_branch is not None:
            if anchor is None:
                anchor = elect_junction_anchor(robots)
            preserve_consensus_at_anchor(anchor, voted_branch)
            if anchor is None or not anchor_deployment_ready(anchor, robots):
                return
            selected = choose_next_branch(
                anchor,
                robots,
                reference_density,
            )
            if selected is None:
                begin_final_gather()
            else:
                commit_junction_guard_roles(robots, selected)
                pending_branch_start = selected
                junction_guard_formation_timer = 0.0
                junction_guard_stable_dwell = 0.0
                # Stay in FORM_JUNCTION_GUARDS: all added K-hop layers must
                # physically reach their mouth slots before selected flow is
                # allowed to start.
                return

    elif phase == SimulationPhase.EXPLORE_BRANCH:
        update_relay_deployment(robots, dt)
        update_frontier_line_progress(robots, active_branch, dt)
        dead_end_confirmed = dead_end_inference_tracker.update(
            robots,
            active_branch,
            reference_density,
            dt,
        )
        if pre_shepherd_branch == active_branch:
            update_pre_shepherd_pack_readiness(
                robots,
                active_branch,
                reference_density,
                dt,
            )
            if (
                not get_shepherds(robots)
                and promote_pre_shepherds(
                    robots,
                    active_branch,
                )
            ):
                phase = SimulationPhase.FILL_BEHIND_SHEPHERD
                saturation_tracker.reset(active_branch)
                branch_continuity_tracker.reset(active_branch)
                print(
                    "[Pre-Shepherd] prior line cleared; "
                    "Branch continuity fill starts before piston push"
                )
            else:
                enforce_pre_shepherd_curtain_for_swarm(robots)
            return

        # Contact/density inference triggers a role transition of the same
        # entrance-guard IDs.  Never discard that line and elect unrelated
        # robots at the dead end.
        if dead_end_confirmed:
            observed_depth = dead_end_inference_tracker.confirmed_depth
            # The same frontier line keeps following the NORMAL front at its
            # existing advance rate.  It may form the return piston only after
            # physically reaching the locally contacted depth, so there is no
            # map-directed sprint to a known terminal coordinate.
            frontier_shepherds = get_frontier_shepherds(
                robots,
                active_branch,
            )
            physical_line_reached = bool(frontier_shepherds) and all(
                branch_depth_from_junction(robot.position, active_branch)
                >= observed_depth - JUNCTION_GUARD_POSITION_TOLERANCE
                for robot in frontier_shepherds
            )
            if not physical_line_reached:
                return
            selected = promote_existing_frontier_line(
                robots,
                active_branch,
                observed_depth,
            )
            if selected:
                phase = SimulationPhase.FORM_SHEPHERD_BOUNDARY
                shepherd_form_timer = 0.0
                # Close a continuous full-width virtual gate immediately.  Any
                # ordinary robot already beyond the planned line is moved to
                # its safe Junction side before the next physics frame.
                enforce_shepherd_curtain_for_swarm(robots)
                print(
                    f"[Shepherd] original frontier line flattened at dead-end: "
                    f"branch={active_branch}, count={len(selected)}"
                )

    elif phase == SimulationPhase.FORM_SHEPHERD_BOUNDARY:
        update_relay_deployment(robots, dt)
        shepherd_form_timer += dt
        if shepherd_boundary_formed(robots):
            phase = SimulationPhase.FILL_BEHIND_SHEPHERD
            saturation_tracker.reset(active_branch)
            branch_continuity_tracker.reset(active_branch)
            print("[Shepherd] boundary formed; ordinary robots now fill behind it")
        elif shepherd_form_timer >= SHEPHERD_FORM_TIMEOUT:
            if force_complete_shepherd_boundary(robots):
                phase = SimulationPhase.FILL_BEHIND_SHEPHERD
                saturation_tracker.reset(active_branch)
                branch_continuity_tracker.reset(active_branch)
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
        continuity_ready = branch_continuity_tracker.update(
            robots,
            active_branch,
            dt,
        )
        saturated = update_dead_end_saturation(
            robots, active_branch, reference_density, dt
        )
        # A dense local pack directly behind the Shepherd is sufficient
        # physical evidence for starting the piston.  The branch-wide
        # continuity tracker remains useful for a merely stalled/slow front,
        # but must not hold a genuinely packed boundary until the Base empties.
        local_packed_ready = (
            saturated
            and saturation_tracker.recognition_mode
            in {"PACKED_DENSITY", "PACKED_GEOMETRY"}
        )
        branch_quota_ready = (
            branch_fill_target_count > 0
            and branch_fill_current_count
            >= branch_fill_target_count
        )
        if (
            saturated
            and branch_quota_ready
            and (
                local_packed_ready
                or continuity_ready
                or branch_continuity_tracker.timed_out
            )
        ):
            if local_packed_ready and not continuity_ready:
                print(
                    "[Pressure Start] local packed boundary overrides "
                    "branch-wide continuity wait; "
                    f"mode={saturation_tracker.recognition_mode}, "
                    f"tip={saturation_tracker.tip_count}, "
                    f"density={saturation_tracker.average_density_ratio:.2f}, "
                    f"occupancy={saturation_tracker.occupancy_ratio:.2f}, "
                    f"quota={branch_fill_current_count}/"
                    f"{branch_fill_target_count}"
                )
            elif branch_continuity_tracker.timed_out:
                print(
                    "[Branch Continuity] fill timeout fallback; "
                    f"coverage="
                    f"{branch_continuity_tracker.covered_slice_ratio:.2f}, "
                    f"gap="
                    f"{branch_continuity_tracker.maximum_depth_gap:.1f}"
                )
            start_shepherd_pressure_push(
                robots,
                active_branch,
            )

    elif phase == SimulationPhase.PRESSURE_PUSH:
        update_pre_shepherd_pipeline(
            robots,
            spatial_grid,
            reference_density,
            dt,
        )
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
        update_pre_shepherd_pipeline(
            robots,
            spatial_grid,
            reference_density,
            dt,
        )
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
        pipeline_switch_ready = (
            transfer_branch is not None
            and pre_shepherd_branch == transfer_branch
            and pre_shepherd_boundary_formed(
                robots,
                transfer_branch,
            )
            and pre_shepherd_pack_ready
            and not get_shepherds(robots)
            and remaining <= PIPELINE_SOURCE_STRAGGLER_LIMIT
        )
        source_branch_relays = get_relays_inside_branch(
            robots,
            active_branch,
        )
        if (
            (
                remaining <= BRANCH_CLEAR_LIMIT
                or pipeline_switch_ready
            )
            and not source_branch_relays
            and not get_shepherds(robots)
            and (
                transfer_ready
                or base_transfer_ready
                or junction_ready
            )
        ):
            completed_branch = active_branch
            next_branch = transfer_branch
            if pipeline_switch_ready and remaining > 0:
                draining_branch = completed_branch
            complete_active_branch(anchor, completed_branch, robots)
            if final_base_transfer_active:
                reset_shepherd_roles(robots)
                final_base_transfer_active = False
                begin_final_return(anchor, robots)
                print(
                    f"[Final Base Transfer] completed "
                    f"{completed_branch} -> BASE; "
                    f"robots={base_transferred_count}"
                )
            elif next_branch is not None:
                reset_shepherd_roles(robots)
                distributed_consensus_branch = next_branch
                selected = choose_next_branch(
                    anchor,
                    robots,
                    reference_density,
                )
                if selected is not None:
                    finish_cross_branch_transfer(robots, selected)
                if draining_branch is not None:
                    branch_gate_states[draining_branch] = "OPEN"
                    preserve_consensus_at_anchor(anchor)
                saturation_tracker.reset(selected)
                dead_end_inference_tracker.reset(selected)
                if (
                    selected == pre_shepherd_branch
                    and promote_pre_shepherds(
                        robots,
                        selected,
                    )
                ):
                    phase = SimulationPhase.FILL_BEHIND_SHEPHERD
                    saturation_tracker.reset(selected)
                    branch_continuity_tracker.reset(selected)
                    print(
                        "[Pre-Shepherd] branch switch completed; "
                        "Branch continuity fill starts before piston push"
                    )
                else:
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
        begin_junction_guard_formation(robots)
        phase = SimulationPhase.FORM_JUNCTION_GUARDS
        print(
            "[Junction] backtracking complete; "
            "re-forming full guards before the next branch order"
        )

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
        if any(
            robot.role in {"RELAY", "SHEPHERD", "PRE_SHEPHERD"}
            for robot in robots
        ):
            release_transient_roles_for_final_return(robots)

        in_bottom = sum(get_robot_region(robot.position) == "BOTTOM" for robot in robots)
        trunk_relays = get_trunk_relays(robots)
        connected_count = sum(robot.connected_to_base for robot in robots)
        connected_ratio = connected_count / max(len(robots), 1)
        special = sum(
            robot.role in {
                "ANCHOR",
                "RELAY",
                "TRUNK_RELAY",
                "SHEPHERD",
                "PRE_SHEPHERD",
            }
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

        if in_bottom > return_last_bottom_count:
            return_last_bottom_count = in_bottom
            return_entry_stall_timer = 0.0
        elif in_bottom < RETURN_BOTTOM_TARGET_COUNT:
            return_entry_stall_timer += dt

        if return_entry_stall_timer >= RETURN_ENTRY_RECOVERY_TIMEOUT:
            recovered = recover_return_entry_stragglers(robots)
            return_entry_stall_timer = 0.0
            if recovered:
                in_bottom = sum(
                    get_robot_region(robot.position) == "BOTTOM"
                    for robot in robots
                )
                return_last_bottom_count = max(
                    return_last_bottom_count,
                    in_bottom,
                )

        done_ready = (
            in_bottom >= RETURN_BOTTOM_TARGET_COUNT
            and special == 0
            and not return_trunk_release_pending
        )
        return_done_dwell = (
            return_done_dwell + dt if done_ready else 0.0
        )
        if return_done_dwell >= RETURN_DONE_DWELL_TIME:
            phase = SimulationPhase.DONE
            metrics.completion_time = simulation_time
            print(f"[DFS] done, robots={in_bottom}/{len(robots)}")
            save_experiment_logs(robots, "DONE")

def draw_collision_points(surface):
    for point in collision_points:
        pygame.draw.circle(
            surface,
            CONTACT_POINT_COLOR,
            (round(point.position.x), round(point.position.y)),
            3,
            width=1,
        )


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
    """Draw peer-consensus gates whose state is retained by the Anchor."""
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


def draw_pre_shepherd_curtain(surface):
    """Visualize the prepared next-branch shield before activation."""
    branch = pre_shepherd_branch
    if not pre_shepherd_curtain_active() or branch is None:
        return
    depth = get_shepherd_boundary_depth(branch)
    color = BRANCH_COLORS[branch]
    overlay = pygame.Surface(
        (SCREEN_WIDTH, SCREEN_HEIGHT),
        pygame.SRCALPHA,
    )
    if branch == "UP":
        y = round(center_y - half_width - depth)
        endpoints = (
            (center_x - half_width + 2, y),
            (center_x + half_width - 2, y),
        )
    elif branch == "LEFT":
        x = round(center_x - half_width - depth)
        endpoints = (
            (x, center_y - half_width + 2),
            (x, center_y + half_width - 2),
        )
    else:
        x = round(center_x + half_width + depth)
        endpoints = (
            (x, center_y - half_width + 2),
            (x, center_y + half_width - 2),
        )
    pygame.draw.line(
        overlay,
        (*color, 170),
        endpoints[0],
        endpoints[1],
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
    global transfer_path_max_gap, transfer_entrance_count
    global transfer_gap_control
    global transfer_target_motion_scale
    global branch_fill_target_count
    global branch_fill_current_count
    global branch_fill_deficit_control
    global previous_branch_direction, junction_anchor, simulation_time
    global junctions, junction_anchors
    global junction_switch_timer, final_gather_timer, shepherd_form_timer
    global pressure_push_timer, flow_establish_timer, communication_sequence
    global shepherd_flow_timer, shepherd_flow_start_depth
    global pre_shepherd_branch
    global pre_shepherd_pack_dwell, pre_shepherd_pack_ready
    global draining_branch
    global initial_release_flow_dwell, initial_release_event_time
    global initial_release_flow_count, initial_release_flow_ratio
    global initial_release_average_speed
    global viscoelastic_step, viscoelastic_rest_lengths
    global viscoelastic_last_seen
    global last_message_signature, communication_redundant_links
    global backtrack_bridge_required_count
    global backtrack_bridge_candidate_count
    global backtrack_bridge_candidate_dwell
    global backtrack_bridge_risk_level
    global backtrack_bridge_natural_redundancy
    global backtrack_bridge_natural_margin
    global relay_slots, relay_deploy_cooldown
    global relay_retract_cooldown, relay_retract_clear_timer, relay_motion_scale
    global trunk_relay_slots, trunk_relay_deploy_cooldown, base_station
    global last_proxy_partition, last_proxy_cell_centers
    global last_proxy_mass_stats, last_proxy_robot_assignment
    global last_proxy_candidates
    global last_flow_rollout_scores
    global selected_branch_entry_lambda, branch_entry_timer
    global return_trunk_release_pending, return_trunk_retract_timer
    global return_trunk_last_released_id, return_trunk_force_timer
    global return_done_dwell, return_entry_stall_timer
    global return_last_bottom_count
    global junction_guard_groups, junction_guard_formation_timer
    global junction_guard_frontier_depths
    global junction_guard_stable_dwell, junction_guard_status
    global pending_branch_start
    global thick_mouth_guard_layers, thick_mouth_guard_columns
    global frontier_line_branch, frontier_line_depth
    global observed_dead_end_depths
    global metrics
    phase = SimulationPhase.MOVE_TO_JUNCTION
    active_branch = FIXED_BRANCH_ORDER[0]
    junctions = create_single_junction_registry()
    junction_anchors = {}
    branch_states = get_junction_state().branch_states
    branch_order_plan = []
    branch_gate_states = get_junction_state().gate_states
    distributed_consensus_branch = None
    transfer_branch = None
    final_base_transfer_active = False
    transfer_path_max_gap = 0.0
    transfer_entrance_count = 0
    transfer_gap_control = 0.0
    transfer_target_motion_scale = 1.0
    branch_fill_target_count = 0
    branch_fill_current_count = 0
    branch_fill_deficit_control = 0.0
    previous_branch_direction = pygame.Vector2(0.0, -1.0)
    junction_anchor = None
    simulation_time = 0.0
    junction_switch_timer = final_gather_timer = shepherd_form_timer = 0.0
    pressure_push_timer = flow_establish_timer = 0.0
    shepherd_flow_timer = shepherd_flow_start_depth = 0.0
    pre_shepherd_branch = None
    pre_shepherd_pack_dwell = 0.0
    pre_shepherd_pack_ready = False
    draining_branch = None
    initial_release_flow_dwell = 0.0
    initial_release_event_time = None
    initial_release_flow_count = 0
    initial_release_flow_ratio = 0.0
    initial_release_average_speed = 0.0
    viscoelastic_step = 0
    viscoelastic_rest_lengths = {}
    viscoelastic_last_seen = {}
    communication_sequence = 0
    last_message_signature = None
    communication_redundant_links = []
    backtrack_bridge_required_count = 0
    backtrack_bridge_candidate_count = 0
    backtrack_bridge_candidate_dwell = 0.0
    backtrack_bridge_risk_level = "STABLE"
    backtrack_bridge_natural_redundancy = 0
    backtrack_bridge_natural_margin = COMM_RANGE
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
    return_done_dwell = 0.0
    return_entry_stall_timer = 0.0
    return_last_bottom_count = 0
    junction_guard_groups = {}
    junction_guard_frontier_depths = {}
    junction_guard_formation_timer = 0.0
    junction_guard_stable_dwell = 0.0
    junction_guard_status = "OPEN_FREE_DIFFUSION"
    pending_branch_start = None
    thick_mouth_guard_layers = {branch: 0 for branch in BRANCHES}
    thick_mouth_guard_columns = {branch: 0 for branch in BRANCHES}
    frontier_line_branch = None
    frontier_line_depth = 0.0
    observed_dead_end_depths = {}
    saturation_tracker.reset()
    branch_continuity_tracker.reset()
    junction_consensus_tracker.reset()
    junction_inference_tracker.reset()
    dead_end_inference_tracker.reset()
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


show_density_color = SHOW_DENSITY_COLOR_DEFAULT
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
            update_transfer_continuity_control(robots)
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
    draw_collision_points(screen)

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
        for robot in (
            get_shepherds(robots)
            + get_pre_shepherds(robots)
        ):
            if robot.shepherd_anchor is not None:
                pygame.draw.circle(
                    screen,
                    BRANCH_COLORS[
                        robot.shepherd_branch
                        or active_branch
                    ],
                    robot.shepherd_anchor,
                    4,
                    width=2,
                )
        draw_shepherd_curtain(screen)
        draw_pre_shepherd_curtain(screen)
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
    backtrack_bridge_stats = get_backtrack_bridge_stats(robots)
    communication_parent_distances = [
        robot.position.distance_to(robot.comm_parent.position)
        for robot in robots
        if robot.connected_to_base and robot.comm_parent is not None
    ]
    communication_guarded_count = sum(
        distance > COMM_GUARD_START
        for distance in communication_parent_distances
    )
    communication_max_parent_distance = max(
        communication_parent_distances,
        default=0.0,
    )
    fluid_body_region_counts = {
        region: sum(
            robot.role == "NORMAL"
            and get_robot_region(robot.position) == region
            for robot in robots
        )
        for region in ("BOTTOM", "JUNCTION", active_branch)
    }
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
    average_edf_force = sum(
        robot.last_edf_force
        for robot in force_samples
    ) / force_sample_count
    base_piston_samples = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and get_robot_region(robot.position) == "BOTTOM"
    ]
    average_base_piston_force = sum(
        robot.last_base_piston_force
        for robot in base_piston_samples
    ) / max(1, len(base_piston_samples))
    active_capture_rect = early_capture_regions[active_branch]
    shepherd_candidate_count = sum(
        robot.role == "NORMAL"
        and get_robot_region(robot.position) == active_branch
        and active_capture_rect.collidepoint(
            robot.position.x,
            robot.position.y,
        )
        for robot in robots
    )
    return_branch_robot_count = sum(
        get_robot_region(robot.position) in BRANCHES
        for robot in robots
    )
    return_bottom_count = sum(
        get_robot_region(robot.position) == "BOTTOM"
        for robot in robots
    )
    return_junction_count = sum(
        get_robot_region(robot.position) == "JUNCTION"
        for robot in robots
    )
    return_special_count = sum(
        robot.role in {
            "ANCHOR",
            "RELAY",
            "TRUNK_RELAY",
            "SHEPHERD",
            "PRE_SHEPHERD",
        }
        for robot in robots
    )
    collective_front_count = sum(
        base_front_equilibrium_weight(robot) > 0.05
        for robot in robots
    )
    collective_tail_count = sum(
        robot.role == "NORMAL"
        and get_robot_region(robot.position) == active_branch
        and branch_depth_from_junction(
            robot.position,
            active_branch,
        )
        <= BASE_FRONT_BRANCH_TAIL_DEPTH
        for robot in robots
    )
    hud_lines = [
        "SPH Physical DFS | Emmons distribution | Eguchi contact inference",
        f"FPS={clock.get_fps():.1f} | robots={len(robots)} | phase={phase.name}",
        f"Sketch sequence stage={get_sequence_stage()}/6",
        (
            "Base impulse=COMPRESS"
            if phase == SimulationPhase.MOVE_TO_JUNCTION
            and simulation_time < BASE_COMPRESSION_DURATION
            else "Base impulse=SPH EXPANSION"
            if initial_pressure_release_active()
            else "Base impulse=FLOW COUPLED"
            if initial_release_event_time is not None
            else "Base impulse=RELEASED"
        ),
        (
            f"Initial safety={INITIAL_IMPULSE_POLICY_VERSION} | "
            f"pressure scale={get_base_pressure_scale():.2f}"
        ),
        (
            f"Base pulse: compress={get_base_compression_envelope():.2f} | "
            f"stored-SPH={get_stored_compression_pressure_envelope():.2f} | "
            f"density-piston={get_base_piston_reaction_envelope():.2f}"
        ),
        (
            "Release event="
            + (
                "RIGHT_FLOW"
                if initial_release_event_time is not None
                else "WAIT"
            )
            + f" | right={initial_release_flow_count} "
            + f"moving={initial_release_flow_ratio:.2f} "
            + f"speed={initial_release_average_speed:.1f} "
            + f"dwell={initial_release_flow_dwell:.2f} "
            + f"cruise={get_initial_release_cruise_blend():.2f}"
        ),
        (
            "Decision=NORMAL peer consensus | "
            f"Junction voters={sum(robot.role == 'NORMAL' and get_robot_region(robot.position) == 'JUNCTION' for robot in robots)}"
        ),
        (
            f"Anchor policy={ANCHOR_POLICY_VERSION} | "
            + (
                f"junction={junction_anchor.anchor_junction_id} "
                f"id={junction_anchor.robot_id} "
                f"cost={junction_anchor.anchor_election_cost:.3f} "
                f"stored={junction_anchor.selected_branch or '-'} "
                f"state-seq={get_junction_state().state_sequence} "
                f"connected={junction_anchor.connected_to_base}"
                if junction_anchor is not None
                else "WAITING_FOR_JUNCTION_NORMAL"
            )
        ),
        (
            f"Fluid body={FLUID_BODY_POLICY_VERSION} | "
            f"B={fluid_body_region_counts['BOTTOM']} "
            f"J={fluid_body_region_counts['JUNCTION']} "
            f"{active_branch}={fluid_body_region_counts[active_branch]}"
        ),
        (
            f"Breadcrumb policy={BREADCRUMB_GUARD_POLICY_VERSION} | "
            "static NORMAL guards=0"
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
        (
            f"Flow continuity: max-gap={transfer_path_max_gap:.1f} | "
            f"entrance={transfer_entrance_count}/"
            f"{TRANSFER_CONTINUITY_MIN_ENTRANCE_ROBOTS} | "
            f"control={transfer_gap_control:.2f} | "
            f"target-scale={transfer_target_motion_scale:.2f}"
        ),
        (
            f"Cohort flow={COHORT_FLOW_POLICY_VERSION} | "
            f"tail-expand="
            f"{FILL_TAIL_EQUILIBRIUM_EXPANSION * smoothstep01(transfer_gap_control):.2f}"
        ),
        (
            f"Branch quota={BRANCH_FILL_QUOTA_POLICY_VERSION} | "
            f"{active_branch}={branch_fill_current_count}/"
            f"{branch_fill_target_count} "
            f"deficit-control={branch_fill_deficit_control:.2f} "
            f"spacing={branch_fill_equilibrium_spacing(active_branch):.1f}"
        ),
        (
            f"Collective equilibrium={COLLECTIVE_EQUILIBRIUM_POLICY_VERSION} | "
            f"front={collective_front_count} | "
            f"tail={collective_tail_count} | "
            f"links<={BASE_FRONT_LINKS_PER_ROBOT} | "
            f"expand="
            f"{BASE_FRONT_EQUILIBRIUM_EXPANSION * collective_equilibrium_activation():.2f} | "
            f"follow-active={collective_equilibrium_activation():.2f}"
        ),
        (
            f"Shepherd trigger={SATURATION_POLICY_VERSION} | "
            f"tip={saturation_tracker.tip_count} "
            f"density={saturation_tracker.average_density_ratio:.2f} "
            f"width={saturation_tracker.lateral_coverage_ratio:.2f}/"
            f"{SATURATION_PACKED_LATERAL_COVERAGE_RATIO:.2f} "
            f"mode={saturation_tracker.recognition_mode} "
            f"ready={saturation_tracker.saturated}"
        ),
        "Gate commands (no geofence): " + " | ".join(
            f"{branch}={branch_gate_states[branch]}"
            for branch in BRANCHES
        ),
        (
            "Physical mouth guards: "
            + " | ".join(
                f"{branch}="
                f"{sum(robot.role == 'JUNCTION_GUARD' and robot.junction_guard_branch == branch for robot in robots)}"
                for branch in BRANCHES
            )
        ),
        (
            f"Thick K-hop walls={THICK_MOUTH_GUARD_POLICY_VERSION} | "
            f"pending={pending_branch_start or '-'} | "
            + " | ".join(
                f"{branch}={thick_mouth_guard_layers.get(branch, 0)}L/"
                f"{thick_mouth_guard_columns.get(branch, 0)}C"
                for branch in BRANCHES
            )
        ),
        (
            f"Persistent frontier={FRONTIER_LINE_POLICY_VERSION} | "
            f"branch={frontier_line_branch or '-'} "
            f"depth={frontier_line_depth:.1f} "
            f"ids={[robot.robot_id for robot in get_frontier_shepherds(robots)]}"
        ),
        (
            "Observed guard frontiers: "
            + " | ".join(
                f"{branch}={junction_guard_frontier_depths.get(branch, 0.0):.1f}"
                for branch in BRANCHES
            )
        ),
        f"Order={' > '.join(branch_order_plan) if branch_order_plan else '-'}",
        (
            "Last branch cost: "
            + (
                f"Q={metrics.branch_selection_events[-1]['components'].get('predicted_flow', 0.0):.2f} "
                f"dRho={metrics.branch_selection_events[-1]['components'].get('density_disturbance', 0.0):.2f} "
                f"dV={metrics.branch_selection_events[-1]['components'].get('velocity_disturbance', 0.0):.2f} "
                f"Comm={metrics.branch_selection_events[-1]['components'].get('rollout_comm', 0.0):.2f}"
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
        (
            f"Junction inference={JUNCTION_INFERENCE_POLICY_VERSION} | "
            f"confirmed={junction_inference_tracker.confirmed} "
            f"n={junction_inference_tracker.observation_count} "
            f"lateral={junction_inference_tracker.expansion_ratio:.2f} "
            f"dwell={junction_inference_tracker.expansion_dwell:.2f} "
            f"discovery={junction_inference_tracker.discovery_dwell:.2f}/"
            f"{JUNCTION_DISCOVERY_SETTLE_TIME:.2f} "
            f"valid={sorted(junction_inference_tracker.valid_branches)} "
            f"front={junction_inference_tracker.forward_probe_status}"
        ),
        (
            "Cohorts: "
            + " | ".join(
                f"{branch}=n{junction_inference_tracker.cohort_counts.get(branch, 0)}"
                f"/d{junction_inference_tracker.cohort_travel.get(branch, 0.0):.1f}"
                for branch in BRANCHES
            )
        ),
        (
            f"Indirect contact={INDIRECT_CONTACT_POLICY_VERSION} | "
            f"points={len(collision_points)} | events={len(metrics.contact_events)}"
        ),
        (
            f"Dead-end inference={DEAD_END_INFERENCE_POLICY_VERSION} | "
            f"frontier={dead_end_inference_tracker.frontier_count} "
            f"leader={dead_end_inference_tracker.leader_contact:.2f} "
            f"contact={dead_end_inference_tracker.mean_contact:.2f} "
            f"v={dead_end_inference_tracker.mean_forward_speed:.1f} "
            f"rho={dead_end_inference_tracker.mean_density_ratio:.2f} "
            f"escape={dead_end_inference_tracker.lateral_escape_ratio:.2f} "
            f"bumper={dead_end_inference_tracker.shepherd_direct_contact_ratio:.2f} "
            f"span={dead_end_inference_tracker.shepherd_contact_span_ratio:.2f} "
            f"shepherd-v={dead_end_inference_tracker.shepherd_mean_forward_speed:.1f} "
            f"depth={dead_end_inference_tracker.confirmed_depth:.1f} "
            f"dwell={dead_end_inference_tracker.dwell:.2f} "
            f"confirmed={dead_end_inference_tracker.confirmed}"
        ),
        (
            "Effective width: "
            + " | ".join(
                f"{branch}={effective_branch_widths.get(branch, 0.0):.1f}"
                for branch in BRANCHES
            )
        ),
        f"States: U={branch_states['UP']} L={branch_states['LEFT']} R={branch_states['RIGHT']}",
        f"Base comm={communication_stats['connected']}/{len(robots)} | hop={communication_stats['max_hop']} | margin={communication_stats['margin']:.1f}",
        (
            f"Comm control={COMM_CONTROL_POLICY_VERSION} | "
            f"guarded={communication_guarded_count} | "
            f"max parent gap={communication_max_parent_distance:.1f}/"
            f"{COMM_GUARD_HARD_LIMIT:.1f}"
        ),
        (
            f"Backtrack bridge={BACKTRACK_BRIDGE_POLICY_VERSION} | "
            f"risk={backtrack_bridge_stats['risk']} "
            f"required={backtrack_bridge_stats['required']} "
            f"guards={backtrack_bridge_stats['guards']} "
            f"settled={backtrack_bridge_stats['settled']} "
            f"ready={backtrack_bridge_stats['ready']}"
        ),
        (
            f"Natural bridge k="
            f"{backtrack_bridge_stats['natural_redundancy']} "
            f"margin={backtrack_bridge_stats['natural_margin']:.1f} | "
            f"links B={backtrack_bridge_stats['base_links']} "
            f"{active_branch}={backtrack_bridge_stats['branch_links']}"
        ),
        f"Base direct={communication_stats['direct']} | Breadcrumbs={len(get_active_branch_relays(robots))}",
        f"Front comm ratio={front_comm['connected_ratio']:.2f} | relay need={front_comm['needs_relay']}",
        f"Reactive relays={len(get_relays(robots))} | preplanned slots=0",
        f"Branch robots normal={normal_count} relay={relay_count} shepherd={shepherd_count}",
        f"Saturation: tip={saturation_tracker.tip_count} slow={saturation_tracker.low_speed_ratio:.2f}",
        (
            f"density={saturation_tracker.average_density_ratio:.2f} "
            f"occupancy={saturation_tracker.occupancy_ratio:.2f} "
            f"width={saturation_tracker.lateral_coverage_ratio:.2f}"
        ),
        (
            f"Saturation policy={SATURATION_POLICY_VERSION} | "
            f"start={PRESSURE_START_POLICY_VERSION} | "
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
            "Branch fill continuity: "
            f"min-slice={branch_continuity_tracker.minimum_slice_count}/"
            f"{BRANCH_CONTINUITY_MIN_ROBOTS_PER_SLICE} | "
            f"coverage={branch_continuity_tracker.covered_slice_ratio:.2f}/"
            f"{BRANCH_CONTINUITY_REQUIRED_SLICE_RATIO:.2f} | "
            f"gap={branch_continuity_tracker.maximum_depth_gap:.1f}/"
            f"{BRANCH_CONTINUITY_MAX_DEPTH_GAP:.1f} | "
            f"dwell={branch_continuity_tracker.dwell:.2f} "
            f"ready={branch_continuity_tracker.ready}"
        ),
        (
            f"Spacing: Normal={NORMAL_EQUILIBRIUM_SCALE:.2f}x | "
            f"Long-fill={LONG_BRANCH_FILL_EQUILIBRIUM_SCALE:.2f}x | "
            f"Shepherd-back={SHEPHERD_PACKED_EQUILIBRIUM_SCALE:.2f}x "
            f"(n={shepherd_packing_count}) | Junction-tail={JUNCTION_TAIL_EQUILIBRIUM_SCALE:.2f}x"
        ),
        f"Shepherd target={adaptive_shepherd_count()} | formed={shepherd_boundary_formed(robots)} | pressure t={pressure_push_timer:.2f}",
        (
            f"Election={SHEPHERD_ELECTION_POLICY_VERSION} | "
            f"candidates={shepherd_candidate_count}/"
            f"{adaptive_shepherd_count()}"
        ),
        (
            f"Shepherd control=LOCAL_ONLY | "
            f"active={len(get_shepherds(robots))} | "
            f"pre={len(get_pre_shepherds(robots))} "
            f"branch={pre_shepherd_branch or '-'} | "
            f"pre-pack={pre_shepherd_pack_ready} "
            f"dwell={pre_shepherd_pack_dwell:.2f} | "
            f"drain={draining_branch or '-'}"
        ),
        (
            f"Motion={MOTION_POLICY_VERSION} | "
            f"Shepherd={SHEPHERD_POLICY_VERSION}"
        ),
        f"Shepherd pipeline={SHEPHERD_PIPELINE_POLICY_VERSION}",
        (
            f"Propulsion={EDF_PROPULSION_POLICY_VERSION} | "
            f"EDF avg={average_edf_force:.1f} | "
            f"bias={WEAK_BRANCH_BIAS_FORCE:.1f}"
        ),
        (
            f"Viscoelastic={VISCOELASTIC_MODEL_VERSION} | "
            f"active links={len(viscoelastic_rest_lengths)}"
        ),
        (
            f"Force avg: SPH={average_sph_pressure_force:.1f} | "
            f"self-expand={average_compression_release_force:.1f} | "
            f"EDF={average_edf_force:.1f} | "
            f"Base-piston={average_base_piston_force:.1f} | "
            f"Shepherd={average_shepherd_force:.1f}"
        ),
        f"Pressure policy={PRESSURE_POLICY_VERSION}",
        (
            f"Final gate={FINAL_GATE_POLICY_VERSION} | "
            f"branch robots={return_branch_robot_count}"
        ),
        (
            f"Return status: B={return_bottom_count}/{len(robots)} "
            f"J={return_junction_count} "
            f"special={return_special_count} "
            f"pending-trunk={return_trunk_release_pending} "
            f"done-dwell={return_done_dwell:.2f}/"
            f"{RETURN_DONE_DWELL_TIME:.2f}"
        ),
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
