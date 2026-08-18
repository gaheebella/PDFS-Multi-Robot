"""SPH-based single-junction Physical DFS prototype.

Implemented research components
-------------------------------
1. Emmons-inspired, normalized local angular distributions are accumulated
   sequentially to infer a Junction from lateral expansion and persistent,
   traversable directional cohorts.
2. NORMAL-to-NORMAL local voting commits only branches discovered by the
   emergent-distribution observer. Completed children are remembered by
   informational Pebbles at their locally observed entrances.
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
import json
import math
import os
import random
import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import pygame

from junction_detection.integration.handoff_row_diagnostics import (
    HandoffRowDiagnostics,
)
from junction_detection.integration.handoff_motion_frame_bias_diagnostics import (
    MotionFrameBiasDiagnostics,
    circular_distance_deg,
)
from junction_detection.integration.mouth_pca_sample_distribution_diagnostics import (
    MouthPcaSampleDistributionDiagnostics,
)
from junction_detection.integration.mouth_crossing_origin_bias_diagnostics import (
    MouthCrossingOriginBiasDiagnostics,
)
from junction_detection.integration.local_mouth_crossing_sampling import (
    LocalMouthCrossingDiagnostics,
    LocalTrajectoryCrossingSampler,
)


HANDOFF_DIAGNOSTICS_ENABLED = (
    os.environ.get("SPH_DFS_HANDOFF_DIAGNOSTICS", "0") == "1"
)
HANDOFF_DIAGNOSTICS_OUTPUT_DIR = Path(
    os.environ.get(
        "SPH_DFS_HANDOFF_DIAGNOSTICS_OUTPUT_DIR",
        "/tmp/pdfs_handoff_diagnostics",
    )
)
handoff_diagnostics = HandoffRowDiagnostics()
MOTION_FRAME_BIAS_DIAGNOSTICS_ENABLED = (
    os.environ.get("SPH_DFS_MOTION_FRAME_BIAS_DIAGNOSTICS", "0") == "1"
)
MOTION_FRAME_BIAS_DIAGNOSTICS_OUTPUT_DIR = Path(
    os.environ.get(
        "SPH_DFS_MOTION_FRAME_BIAS_DIAGNOSTICS_OUTPUT_DIR",
        "/tmp/pdfs_handoff_motion_frame_bias_diagnostics",
    )
)
motion_frame_bias_diagnostics = MotionFrameBiasDiagnostics()
MOUTH_PCA_SAMPLE_DIAGNOSTICS_ENABLED = (
    os.environ.get("SPH_DFS_MOUTH_PCA_SAMPLE_DIAGNOSTICS", "0") == "1"
)
MOUTH_PCA_SAMPLE_DIAGNOSTICS_OUTPUT_DIR = Path(
    os.environ.get(
        "SPH_DFS_MOUTH_PCA_SAMPLE_DIAGNOSTICS_OUTPUT_DIR",
        "/tmp/pdfs_mouth_pca_sample_diagnostics",
    )
)
mouth_pca_sample_diagnostics = MouthPcaSampleDistributionDiagnostics()
MOUTH_CROSSING_ORIGIN_DIAGNOSTICS_ENABLED = (
    os.environ.get("SPH_DFS_MOUTH_CROSSING_ORIGIN_DIAGNOSTICS", "0") == "1"
)
MOUTH_CROSSING_ORIGIN_DIAGNOSTICS_OUTPUT_DIR = Path(
    os.environ.get(
        "SPH_DFS_MOUTH_CROSSING_ORIGIN_DIAGNOSTICS_OUTPUT_DIR",
        "/tmp/pdfs_mouth_crossing_origin_diagnostics",
    )
)
mouth_crossing_origin_diagnostics = MouthCrossingOriginBiasDiagnostics()
LOCAL_MOUTH_CROSSING_DIAGNOSTICS_ENABLED = (
    os.environ.get("SPH_DFS_LOCAL_MOUTH_CROSSING_DIAGNOSTICS", "0") == "1"
)
LOCAL_MOUTH_SAMPLE_MODE = os.environ.get(
    "SPH_DFS_MOUTH_SAMPLE_MODE", "heading_origin"
)
if LOCAL_MOUTH_SAMPLE_MODE not in {"heading_origin", "local_crossing"}:
    raise ValueError(
        "SPH_DFS_MOUTH_SAMPLE_MODE must be heading_origin or local_crossing"
    )
LOCAL_MOUTH_CROSSING_OUTPUT_DIR = Path(
    os.environ.get(
        "SPH_DFS_LOCAL_MOUTH_CROSSING_OUTPUT_DIR",
        "/tmp/pdfs_local_mouth_crossing",
    )
)
local_mouth_crossing_diagnostics = None
local_mouth_shepherd_branches: set[str] = set()
local_mouth_pressure_branches: set[str] = set()
local_mouth_backflow_branches: set[str] = set()


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
PEBBLE_COLOR = (20, 220, 90)
PEBBLE_RING_COLOR = (0, 72, 34)
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


def _record_evaluation_geometric_mouth_crossings(
    robot,
    old_position: pygame.Vector2,
    dt: float,
) -> None:
    """Interpolate first outward mouth-plane crossings for diagnostics only."""
    if not (
        MOUTH_CROSSING_ORIGIN_DIAGNOSTICS_ENABLED
        or LOCAL_MOUTH_CROSSING_DIAGNOSTICS_ENABLED
    ):
        return
    current = robot.position
    if LOCAL_MOUTH_CROSSING_DIAGNOSTICS_ENABLED:
        local_mouth_crossing_diagnostics.sampler.record_motion(
            robot.robot_id,
            (old_position.x, old_position.y),
            (current.x, current.y),
            simulation_time,
            dt,
        )
    segment_x = current.x - old_position.x
    segment_y = current.y - old_position.y
    candidates = []
    left_plane = center_x - half_width
    if old_position.x >= left_plane and current.x < left_plane and abs(segment_x) > EPSILON:
        candidates.append(("LEFT", (left_plane - old_position.x) / segment_x))
    right_plane = center_x + half_width
    if old_position.x <= right_plane and current.x > right_plane and abs(segment_x) > EPSILON:
        candidates.append(("RIGHT", (right_plane - old_position.x) / segment_x))
    up_plane = center_y - half_width
    if old_position.y >= up_plane and current.y < up_plane and abs(segment_y) > EPSILON:
        candidates.append(("UP", (up_plane - old_position.y) / segment_y))
    for branch, alpha in candidates:
        if not -EPSILON <= alpha <= 1.0 + EPSILON:
            continue
        alpha = clamp(alpha, 0.0, 1.0)
        crossing = old_position.lerp(current, alpha)
        lateral_coordinate = crossing.y if branch in {"LEFT", "RIGHT"} else crossing.x
        lateral_minimum = center_y - half_width if branch in {"LEFT", "RIGHT"} else center_x - half_width
        lateral_maximum = center_y + half_width if branch in {"LEFT", "RIGHT"} else center_x + half_width
        if not lateral_minimum - EPSILON <= lateral_coordinate <= lateral_maximum + EPSILON:
            continue
        mouth_crossing_time = simulation_time - dt + alpha * dt
        crossing_row = {
            "branch": branch,
            "robot_id": robot.robot_id,
            "frame": globals().get("headless_frame_count", 0),
            "mouth_crossing_time": mouth_crossing_time,
            "crossing_world_x": crossing.x,
            "crossing_world_y": crossing.y,
            "interpolation_alpha": alpha,
            "old_world_x": old_position.x,
            "old_world_y": old_position.y,
            "current_world_x": current.x,
            "current_world_y": current.y,
            "robot_role": robot.role,
            "evaluation_only_gt": True,
        }
        if MOUTH_CROSSING_ORIGIN_DIAGNOSTICS_ENABLED:
            mouth_crossing_origin_diagnostics.record_gt_crossing(crossing_row)
        if LOCAL_MOUTH_CROSSING_DIAGNOSTICS_ENABLED:
            local_mouth_crossing_diagnostics.record_gt_crossing(crossing_row)

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
# Research boundary: dynamic Branch UIDs reduce decision/memory dependence on
# map labels, but this prototype is not fully localization-free. Functions
# such as get_robot_region(), get_branch_entrance(), branch_depth_from_junction()
# and the existing guard/EDF/Shepherd/relay geometry still use these fixtures.
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
class BranchDescriptor:
    """Stable local identity for one physically observed outgoing cohort.

    ``fixture_key`` is only an adapter into the current cross-map geometry.
    Distributed DFS memory and decisions use ``uid``; the legacy
    UP/LEFT/RIGHT key is consulted only by existing motion/physics code.
    """

    uid: str
    junction_uid: str
    fixture_key: Optional[str]
    local_outgoing_direction: pygame.Vector2
    local_return_direction: pygame.Vector2
    observed_mouth_position: Optional[pygame.Vector2]
    observed_width: float
    visit_state: str = "UNVISITED"
    leader_id: Optional[int] = None
    cohort_member_ids: set[int] = field(default_factory=set)
    observed_mouth_samples: dict[int, pygame.Vector2] = field(
        default_factory=dict
    )
    cohort_downstream_travel: dict[int, float] = field(
        default_factory=dict
    )
    cohort_segment_anchors: dict[int, pygame.Vector2] = field(
        default_factory=dict
    )
    cohort_recent_segments: dict[int, pygame.Vector2] = field(
        default_factory=dict
    )
    cohort_segment_update_counts: dict[int, int] = field(
        default_factory=dict
    )
    direction_last_estimate: Optional[pygame.Vector2] = None
    direction_stability_reference: Optional[pygame.Vector2] = None
    direction_stable_dwell: float = 0.0
    direction_sample_count: int = 0
    direction_angular_spread: float = math.pi
    direction_is_stable: bool = False
    direction_mature_dwell: float = 0.0
    direction_is_mature: bool = False
    direction_downstream_travel: float = 0.0
    direction_refinement_count: int = 0
    direction_last_refinement_time: float = float("-inf")
    direction_last_diagnostic_time: float = float("-inf")
    motion_t: Optional[pygame.Vector2] = None
    motion_n: Optional[pygame.Vector2] = None
    motion_frame_locked: bool = False
    motion_frame_source: str = "UNLOCKED"
    motion_frame_sample_count: int = 0
    motion_frame_angular_spread: float = math.pi
    motion_observed_width: float = 0.0
    observed_flow_width: float = 0.0
    observed_physical_width: float = 0.0
    physical_width_confident: bool = False
    physical_width_source: str = "UNOBSERVED"
    physical_left_boundary_lateral: Optional[float] = None
    physical_right_boundary_lateral: Optional[float] = None
    physical_boundary_sample_count: int = 0
    discovered_at: float = 0.0


@dataclass(frozen=True)
class PebbleFlowState:
    """Immutable VISITED fact relayed through local robot communication."""

    branch_uid: str
    branch_key: Optional[str]
    mouth_position: tuple[float, float]
    ingress_direction_local: tuple[float, float]
    return_direction_local: tuple[float, float]
    completion_epoch: int


@dataclass
class JunctionState:
    """Simulation bookkeeping for one inferred Junction.

    Mobile robots never consult this object to learn whether a child was
    visited; that runtime knowledge comes from local Pebble messages.
    """

    junction_id: str
    rect: pygame.Rect
    branch_edges: dict[str, BranchEdge]
    branch_states: dict[str, str]
    gate_states: dict[str, str]
    selected_branch: Optional[str] = None
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
        branch_edges=branch_edges,
        branch_states={branch: "UNVISITED" for branch in BRANCHES},
        # All mouths are physically and logically OPEN during free diffusion.
        # They close only after emergent distribution confirms a Junction.
        gate_states={branch: "OPEN" for branch in BRANCHES},
    )
    return {CURRENT_JUNCTION_ID: state}


junctions = create_single_junction_registry()
def get_junction_state(junction_id: str = CURRENT_JUNCTION_ID) -> JunctionState:
    return junctions[junction_id]


phase = SimulationPhase.MOVE_TO_JUNCTION
active_branch = FIXED_BRANCH_ORDER[0]
active_branch_uid: Optional[str] = None
branch_states = get_junction_state().branch_states
# DFS ordering is UID-based. Fixture order is retained separately only for
# diagnostics that compare against the present cross-map.
branch_order_plan: list[str] = []
branch_fixture_order_plan: list[str] = []
branch_gate_states = get_junction_state().gate_states
# Despite the legacy variable name, this value is a BranchDescriptor UID.
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
branch_fill_feed_scale = 1.0
branch_fill_feed_state = "OPEN"
branch_fill_feed_last_logged_state = "OPEN"
previous_branch_direction = pygame.Vector2(0.0, -1.0)  # incoming from BASE

simulation_time = 0.0
branch_completion_epoch = 0
branch_dead_end_confirmed = {branch: False for branch in BRANCHES}
branch_backflow_started = {branch: False for branch in BRANCHES}
# No descriptor exists at startup. A UID is allocated only when local swarm
# motion validates a directional outgoing cohort.
branch_descriptors_by_uid: dict[str, BranchDescriptor] = {}
fixture_key_to_branch_uid: dict[str, str] = {}
branch_uid_to_fixture_key: dict[str, str] = {}
branch_discovery_counter = 0
# Compatibility mirror for older local-ingress and rendering paths.
branch_local_uids: dict[str, str] = {}
pebble_rx_logged: set[str] = set()
pending_pebble_robot_ids: dict[str, int] = {}
pebble_flow_last_log_time = float("-inf")
current_visited_branch_leakage = {branch: 0 for branch in BRANCHES}
current_pebble_guidance_active_normals = 0
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
last_decision_candidate_uids: tuple[str, ...] = ()
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
junction_guard_last_wait_log_time = float("-inf")
pending_branch_start: Optional[str] = None
local_guard_width_fallback_logged: set[str] = set()
local_guard_pending_logged: set[tuple[str, str]] = set()
thick_mouth_guard_layers: dict[str, int] = {
    branch: 0 for branch in BRANCHES
}
thick_mouth_guard_columns: dict[str, int] = {
    branch: 0 for branch in BRANCHES
}
frontier_line_branch: Optional[str] = None
frontier_line_depth = 0.0
frontier_line_lateral_center = 0.0
frontier_line_target_settled_ratio = 0.0
frontier_line_current_span = 0.0
frontier_line_target_span = 0.0
frontier_line_physical_coverage_ratio = 0.0
frontier_line_left_edge_gap = float("inf")
frontier_line_right_edge_gap = float("inf")
frontier_line_continuous = False
frontier_line_row_ready = False
frontier_line_last_diagnostic_time = float("-inf")
frontier_dead_end_transition_last_log_time = float("-inf")
observed_dead_end_depths: dict[str, float] = {}
last_distributed_vote_counts: dict[str, int] = {}
last_distributed_voter_count = 0
last_distributed_vote_quorum = 0
last_distributed_vote_status = "INACTIVE"

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
BRANCH_FILL_FEED_THROTTLE_START_RATIO = 0.82
BRANCH_FILL_FEED_CLOSED_RATIO = 1.00
BRANCH_FILL_FEED_FILTER_ALPHA = 0.35
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
JUNCTION_SWITCH_COUNT = 18
JUNCTION_SWITCH_DWELL_TIME = 0.25
FINAL_GATHER_DWELL_TIME = 0.55

PEBBLE_POLICY_VERSION = "LOCAL_VISITED_MARKER_FLOW_V2"
# Pebbles remain force-free markers. These gains only condition the NORMAL
# decision-layer force (route + EDF); no SPH, collision, or connectivity term
# is projected or attenuated.
PEBBLE_GUIDANCE_RADIUS = corridor_width * 1.10
PEBBLE_MOUTH_RADIUS = SMOOTHING_LENGTH * 1.60
PEBBLE_BRANCH_LATERAL_MARGIN = SMOOTHING_LENGTH * 0.75
PEBBLE_RECOVERY_FULL_DEPTH = SMOOTHING_LENGTH * 0.90
PEBBLE_CAUTION_GAIN = 0.45
PEBBLE_MOUTH_SUPPRESSION_GAIN = 0.45
PEBBLE_PREVENTIVE_RETURN_FORCE = ROUTE_FORCE * 0.75
PEBBLE_RECOVERY_GAIN = OUTLET_FORCE * 1.75
PEBBLE_GUIDANCE_FORCE_LIMIT = ROUTE_FORCE * 2.50
PEBBLE_CROSSING_MIN_RETURN_SPEED = 1.0
PEBBLE_FLOW_LOG_INTERVAL = 1.0
PEBBLE_INGRESS_MIN_OBSERVED_TRAVEL = ROBOT_RADIUS * 2.0
PEBBLE_INGRESS_DIRECTION_FILTER_ALPHA = 0.25
PEBBLE_INGRESS_COHORT_MIN_TRAVEL = corridor_width * 1.05

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
SHEPHERD_MAX_COUNT = 22
SHEPHERD_EDGE_MARGIN = 12.0 * MAP_SCALE
# Keep adjacent physical Shepherd influence zones overlapping. This produces a
# genuinely dense cross-section instead of relying on a five-robot fallback.
SHEPHERD_TARGET_SLOT_SPACING = SAFE_RADIUS * 0.85

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
# preserving the Base link and minimum robot-count guards.
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
if LOCAL_MOUTH_CROSSING_DIAGNOSTICS_ENABLED:
    local_mouth_crossing_diagnostics = LocalMouthCrossingDiagnostics(
        mode=LOCAL_MOUTH_SAMPLE_MODE,
        sampler=LocalTrajectoryCrossingSampler(
            history_size=JUNCTION_DISTRIBUTION_WINDOW,
            minimum_speed=JUNCTION_COHORT_MIN_SPEED,
            half_angle_rad=JUNCTION_COHORT_HALF_ANGLE,
        ),
    )
LOCAL_BRANCH_DIRECTION_DOWNSTREAM_START = corridor_width * 0.50
LOCAL_BRANCH_DIRECTION_SEGMENT_LENGTH = JUNCTION_COHORT_MIN_TRAVEL
LOCAL_BRANCH_DIRECTION_MIN_SAMPLES = max(
    5,
    math.ceil(JUNCTION_COHORT_MIN_ROBOTS * 0.75),
)
LOCAL_BRANCH_DIRECTION_OUTLIER_ANGLE = math.radians(28.0)
LOCAL_BRANCH_DIRECTION_STABLE_DELTA = math.radians(6.0)
LOCAL_BRANCH_DIRECTION_STABLE_SPREAD = math.radians(20.0)
LOCAL_BRANCH_DIRECTION_STABLE_DWELL = 0.20
LOCAL_BRANCH_DIRECTION_MATURE_DWELL = 0.25
LOCAL_BRANCH_DIRECTION_MIN_REFINEMENT_UPDATES = 3
LOCAL_BRANCH_DIRECTION_MIN_SECOND_PASS_SAMPLES = max(
    4,
    LOCAL_BRANCH_DIRECTION_MIN_SAMPLES - 1,
)
LOCAL_BRANCH_DIRECTION_FILTER_ALPHA = 0.28
JUNCTION_MIN_VALID_COHORTS = 2
JUNCTION_DISCOVERY_SETTLE_TIME = 1.00
JUNCTION_FRONT_BLOCK_MIN_CONTACTS = 4
JUNCTION_FRONT_BLOCK_MIN_SPAN_RATIO = 0.80
# The initial safety wall must sit beyond the trajectory-maturation window.
# It still limits uncommitted exploration, but no longer traps every cohort in
# the Junction-to-Branch turn before a recent straight segment can be seen.
JUNCTION_PROBE_DEPTH = (
    LOCAL_BRANCH_DIRECTION_DOWNSTREAM_START
    + LOCAL_BRANCH_DIRECTION_SEGMENT_LENGTH * 3.0
)
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
LOCAL_GUARD_MOUTH_TOLERANCE = ROBOT_RADIUS * 4.0
LOCAL_GUARD_LATERAL_RECRUIT_MARGIN = max(
    JUNCTION_GUARD_COVERAGE,
    SAFE_RADIUS,
)
LOCAL_GUARD_ORTHOGONAL_EPSILON = 1.0e-5
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
FRONTIER_LINE_TARGET_SETTLED_RATIO = 0.80
FRONTIER_LINE_MIN_SPAN_RATIO = 0.96
FRONTIER_LINE_TARGET_TOLERANCE = max(
    JUNCTION_GUARD_POSITION_TOLERANCE,
    ROBOT_RADIUS * 1.25,
)
FRONTIER_LINE_DRIFT_RESERVE = ROBOT_RADIUS * 1.50
FRONTIER_LINE_EDGE_CLEARANCE = ROBOT_RADIUS * 0.25
FRONTIER_LINE_MAX_EDGE_GAP = max(
    FRONTIER_LINE_TARGET_TOLERANCE,
    PHYSICAL_GUARD_INFLUENCE_RADIUS * 0.35,
)
FRONTIER_LINE_MAX_INTERNAL_GAP = PHYSICAL_GUARD_INFLUENCE_RADIUS * 1.35
FRONTIER_LINE_CENTER_CORRECTION_SPEED = ROBOT_RADIUS * 0.75
FRONTIER_LINE_CENTER_MIN_COHORT = 5
FRONTIER_LINE_CENTER_WINDOW = SMOOTHING_LENGTH * 2.0
FRONTIER_MOUTH_AXIS_MAX_FLOW_ANGLE = math.radians(25.0)
FRONTIER_MOUTH_AXIS_MIN_ANISOTROPY = 1.35
LOCAL_PHYSICAL_WIDTH_PROBE_AXIAL = max(
    JUNCTION_GUARD_BRANCH_INSET * 2.0,
    ROBOT_RADIUS * 5.0,
)
LOCAL_PHYSICAL_WIDTH_PROBE_STEP = max(0.5, ROBOT_RADIUS * 0.50)
LOCAL_PHYSICAL_WIDTH_MAX_PROBE_RANGE = JUNCTION_GUARD_RECRUIT_RADIUS
LOCAL_PHYSICAL_WIDTH_CONTACT_AXIAL_WINDOW = 44.0 * MAP_SCALE
LOCAL_PHYSICAL_WIDTH_MIN_SIDE_CONTACTS = 2
FRONTIER_LINE_POLICY_VERSION = "LOCAL_PHYSICAL_WIDTH_HANDOFF_V3"

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
DEAD_END_FRONTIER_PROGRESS_WINDOW = 0.28
DEAD_END_FRONTIER_PROGRESS_RATE_THRESHOLD = 2.5
DEAD_END_FAST_MIN_BRANCH_ROBOTS = 32
DEAD_END_FAST_CONFIRM_DWELL = 0.12
DEAD_END_INFERENCE_POLICY_VERSION = "SHEPHERD_CONTACT_STALL_PRIMARY_V4"

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
    visited_branch_reentry_counts: dict[str, int] = field(
        default_factory=lambda: {branch: 0 for branch in BRANCHES}
    )
    visited_branch_reentry_robot_ids: dict[str, set[int]] = field(
        default_factory=lambda: {branch: set() for branch in BRANCHES}
    )
    visited_branch_max_inside: dict[str, int] = field(
        default_factory=lambda: {branch: 0 for branch in BRANCHES}
    )
    visited_branch_robot_seconds: dict[str, float] = field(
        default_factory=lambda: {branch: 0.0 for branch in BRANCHES}
    )
    exploration_leak_reentry_counts: dict[str, int] = field(
        default_factory=lambda: {branch: 0 for branch in BRANCHES}
    )
    exploration_leak_reentry_robot_ids: dict[str, set[int]] = field(
        default_factory=lambda: {branch: set() for branch in BRANCHES}
    )
    exploration_leak_max_inside: dict[str, int] = field(
        default_factory=lambda: {branch: 0 for branch in BRANCHES}
    )
    exploration_leak_robot_seconds: dict[str, float] = field(
        default_factory=lambda: {branch: 0.0 for branch in BRANCHES}
    )
    initialized_visited_uids: set[str] = field(default_factory=set)
    pebble_guidance_activation_count: int = 0
    pebble_recovery_success_count: int = 0
    pebble_recovery_success_pairs: set[tuple[int, str]] = field(
        default_factory=set
    )
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
    pebble_distance = sum(robot.distance_by_role.get("PEBBLE", 0.0) for robot in robots)
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
        writer.writerow(["pebble_distance", f"{pebble_distance:.6f}"])
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
        writer.writerow([
            "pebble_guidance_activation_count",
            metrics.pebble_guidance_activation_count,
        ])
        writer.writerow([
            "pebble_recovery_success_count",
            metrics.pebble_recovery_success_count,
        ])
        for branch in BRANCHES:
            writer.writerow([
                f"effective_width_{branch.lower()}",
                f"{effective_branch_widths.get(branch, 0.0):.6f}",
            ])
            writer.writerow([
                f"visited_reentry_count_{branch.lower()}",
                metrics.visited_branch_reentry_counts[branch],
            ])
            writer.writerow([
                f"visited_max_inside_{branch.lower()}",
                metrics.visited_branch_max_inside[branch],
            ])
            writer.writerow([
                f"visited_robot_seconds_{branch.lower()}",
                f"{metrics.visited_branch_robot_seconds[branch]:.6f}",
            ])
            writer.writerow([
                f"exploration_leak_reentry_count_{branch.lower()}",
                metrics.exploration_leak_reentry_counts[branch],
            ])
            writer.writerow([
                f"exploration_leak_max_inside_{branch.lower()}",
                metrics.exploration_leak_max_inside[branch],
            ])
            writer.writerow([
                f"exploration_leak_robot_seconds_{branch.lower()}",
                f"{metrics.exploration_leak_robot_seconds[branch]:.6f}",
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
        # Lateral slot in the Branch's observed local frame. It is the only
        # frontier-line geometry a robot keeps while the line advances.
        self.frontier_local_lateral: Optional[float] = None
        self.junction_guard_anchor: Optional[pygame.Vector2] = None
        self.junction_guard_branch: Optional[str] = None
        self.junction_guard_branch_uid: Optional[str] = None
        self.junction_guard_hop = -1
        self.junction_guard_parent_id: Optional[int] = None
        self.junction_guard_layer = -1
        self.is_branch_leader = False
        self.relay_anchor: Optional[pygame.Vector2] = None
        self.relay_index = -1
        self.local_branch_states = branch_states.copy()
        self.known_visited_branches: set[str] = set()
        self.known_visited_branch_uids: set[str] = set()
        self.local_branch_uid_by_key: dict[str, str] = {}
        self.local_ingress_tangents_by_uid: dict[str, pygame.Vector2] = {}
        self.local_branch_ingress_points_by_uid: dict[str, pygame.Vector2] = {}
        self.local_ingress_observed_travel_by_uid: dict[str, float] = {}
        self.local_ingress_tangents: dict[str, pygame.Vector2] = {}
        self.local_branch_ingress_points: dict[str, pygame.Vector2] = {}
        self.local_ingress_observed_travel: dict[str, float] = {}
        self.local_return_mouth_crossings: dict[str, pygame.Vector2] = {}
        self.known_pebble_flow_states: dict[str, PebbleFlowState] = {}
        self.pebble_anchor: Optional[pygame.Vector2] = None
        self.pebble_branch_uid: Optional[str] = None
        self.pebble_branch_key: Optional[str] = None
        self.pebble_state: Optional[str] = None
        self.pebble_ingress_direction_local: Optional[pygame.Vector2] = None
        self.pebble_return_direction_local: Optional[pygame.Vector2] = None
        self.pebble_completion_epoch = 0
        self.last_pebble_guidance_force = 0.0
        self.last_pebble_guidance_weight = 0.0
        self.last_pebble_guidance_mode = "NONE"
        self.last_pebble_guidance_branch_uid: Optional[str] = None
        self.metric_pebble_guidance_active = False
        self.metric_inside_visited_uids: set[str] = set()
        self.metric_pebble_recovery_seen_uids: set[str] = set()
        self.branch_vote: Optional[str] = None
        self.branch_vote_confidence = 0.0
        self.distributed_branch_decision: Optional[str] = None
        self.transfer_target: Optional[str] = None
        self.base_reserve = False
        self.base_hold_position: Optional[pygame.Vector2] = None
        self.parent_branch = "BOTTOM"
        self.known_junction_states: dict[str, dict] = {}
        self.comm_bridge_target: Optional[pygame.Vector2] = None
        self.comm_bridge_index = -1
        self.comm_bridge_branch: Optional[str] = None

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

        self.total_distance = 0.0
        self.distance_by_role = {
            "NORMAL": 0.0,
            "PEBBLE": 0.0,
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

        if self.role == "PEBBLE" and self.pebble_anchor is not None:
            self.position = self.pebble_anchor.copy()
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
            # During the one-row full-Guard formation stage, the assigned
            # K-hop group is the connectivity structure.  A transient global
            # comm_parent can point behind the moving row and permanently
            # clamp an otherwise walkable anchor.  Thick-wall motion keeps the
            # existing communication clamp unchanged.
            if not (
                phase == SimulationPhase.FORM_JUNCTION_GUARDS
                and pending_branch_start is None
                and self.junction_guard_layer == 0
            ):
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
            and (
                phase == SimulationPhase.EXPLORE_BRANCH
                or (
                    phase == SimulationPhase.FORM_JUNCTION_GUARDS
                    and pending_branch_start == self.shepherd_branch
                )
            )
            and self.shepherd_anchor is not None
            and self.shepherd_branch == frontier_line_branch
        ):
            # The selected entrance guard remains the same physical group for
            # the whole branch.  It advances as a transverse line instead of
            # dissolving into the NORMAL SPH body and being re-elected later.
            # The row is expressed in the Branch's own observed frame:
            # target = observed_mouth + t * axial_depth + n * lateral_slot.
            target = frontier_shepherd_slot_target(self, frontier_line_depth)
            if target is None:
                target = self.position.copy()
            # Local forward bumper/proximity probe.  A communication guard or
            # NORMAL pressure can stop the robot without making this probe
            # positive; only the physical map mask immediately ahead does.
            outward = frontier_shepherd_forward_direction(self)
            if outward is not None and phase == SimulationPhase.EXPLORE_BRANCH:
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
        _record_evaluation_geometric_mouth_crossings(
            self,
            old_position,
            dt,
        )
        maybe_stage_pebble_at_return_crossing(self, old_position)
        update_indirect_contact_state(self, dt)
        self.acceleration.update(0.0, 0.0)
        self.previous_position = old_position
        self._record_motion()

    def draw(self, surface, color_reference_density, show_density_color):
        x, y = round(self.position.x), round(self.position.y)
        if self.role == "PEBBLE":
            color = PEBBLE_COLOR
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
        elif self.role == "PEBBLE":
            pygame.draw.circle(
                surface,
                PEBBLE_RING_COLOR,
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


def descriptor_local_basis(
    descriptor: BranchDescriptor,
) -> tuple[pygame.Vector2, pygame.Vector2]:
    """Return the measured local axial/transverse basis for one Branch."""
    outgoing = (
        descriptor.motion_t
        if descriptor.motion_frame_locked and descriptor.motion_t is not None
        else descriptor.local_outgoing_direction
    )
    if outgoing.length_squared() <= EPSILON:
        raise ValueError(f"Branch {descriptor.uid} has no outgoing direction")
    tangent = outgoing.normalize()
    normal = (
        descriptor.motion_n.normalize()
        if (
            descriptor.motion_frame_locked
            and descriptor.motion_n is not None
            and descriptor.motion_n.length_squared() > EPSILON
        )
        else pygame.Vector2(-tangent.y, tangent.x)
    )
    if abs(tangent.dot(normal)) >= LOCAL_GUARD_ORTHOGONAL_EPSILON:
        raise AssertionError(f"non-orthogonal local Guard basis: {descriptor.uid}")
    return tangent, normal


def branch_local_coordinates(
    position: pygame.Vector2,
    descriptor: BranchDescriptor,
) -> tuple[float, float]:
    """Project a world pose into the descriptor's mouth-centred local frame."""
    if descriptor.observed_mouth_position is None:
        raise ValueError(f"Branch {descriptor.uid} has no mouth observation")
    tangent, normal = descriptor_local_basis(descriptor)
    relative = position - descriptor.observed_mouth_position
    return relative.dot(tangent), relative.dot(normal)


def local_coordinates_to_world(
    descriptor: BranchDescriptor,
    axial: float,
    lateral: float,
) -> pygame.Vector2:
    """Inverse of branch_local_coordinates(): mouth + t*axial + n*lateral."""
    if descriptor.observed_mouth_position is None:
        raise ValueError(f"Branch {descriptor.uid} has no mouth observation")
    tangent, normal = descriptor_local_basis(descriptor)
    return (
        descriptor.observed_mouth_position
        + tangent * axial
        + normal * lateral
    )


def branch_motion_descriptor(
    branch: Optional[str],
) -> Optional[BranchDescriptor]:
    """Return the descriptor usable as a local motion frame for one Branch.

    Motion code calls this instead of reading BRANCH_DIRECTIONS/BRANCH_LENGTHS.
    ``None`` means nothing has been observed yet; local-frame motion callers
    pause until that observation exists rather than inventing fixture geometry.
    """
    if branch is None:
        return None
    branch_uid = branch_uid_for_fixture(branch)
    if branch_uid is None:
        return None
    descriptor = branch_descriptors_by_uid.get(branch_uid)
    if descriptor is None or descriptor.observed_mouth_position is None:
        return None
    if descriptor.local_outgoing_direction.length_squared() <= EPSILON:
        return None
    return descriptor


def observed_branch_axial_depth(
    position: pygame.Vector2,
    descriptor: BranchDescriptor,
) -> float:
    """Depth along a Branch measured only in its observed local frame."""
    return max(0.0, branch_local_coordinates(position, descriptor)[0])


def probe_local_boundary_center_limit(
    descriptor: BranchDescriptor,
    axial: float,
    side: float,
) -> Optional[float]:
    """Probe one physical side wall along local n with a robot footprint.

    This is the simulator equivalent of an edge robot extending laterally
    until its local collision/proximity sensor reports a boundary.  The probe
    reads neither a Branch label nor a fixture width.
    """
    if side not in {-1.0, 1.0}:
        raise ValueError("physical mouth probe side must be -1 or +1")
    if not is_walkable(
        local_coordinates_to_world(descriptor, axial, 0.0),
        ROBOT_RADIUS,
    ):
        return None
    last_walkable = 0.0
    distance = LOCAL_PHYSICAL_WIDTH_PROBE_STEP
    while distance <= LOCAL_PHYSICAL_WIDTH_MAX_PROBE_RANGE:
        candidate = local_coordinates_to_world(
            descriptor,
            axial,
            side * distance,
        )
        if not is_walkable(candidate, ROBOT_RADIUS):
            low = last_walkable
            high = distance
            for _ in range(14):
                middle = 0.5 * (low + high)
                middle_point = local_coordinates_to_world(
                    descriptor,
                    axial,
                    side * middle,
                )
                if is_walkable(middle_point, ROBOT_RADIUS):
                    low = middle
                else:
                    high = middle
            return side * low
        last_walkable = distance
        distance += LOCAL_PHYSICAL_WIDTH_PROBE_STEP
    return None


def update_local_physical_mouth_width(
    descriptor: BranchDescriptor,
) -> bool:
    """Separate cohort flow span from locally observed wall-to-wall width."""
    if descriptor.physical_width_confident:
        return True
    if (
        not descriptor.motion_frame_locked
        or descriptor.observed_mouth_position is None
    ):
        return False

    descriptor.observed_flow_width = max(
        descriptor.observed_flow_width,
        descriptor.motion_observed_width,
        descriptor.observed_width,
    )
    contact_laterals = []
    for point in collision_points:
        if point.branch != descriptor.fixture_key:
            continue
        axial, lateral = branch_local_coordinates(point.position, descriptor)
        if (
            0.0 <= axial <= LOCAL_PHYSICAL_WIDTH_CONTACT_AXIAL_WINDOW
            and abs(lateral) >= ROBOT_RADIUS * 2.0
        ):
            contact_laterals.append(lateral)
    left_contacts = [value for value in contact_laterals if value < 0.0]
    right_contacts = [value for value in contact_laterals if value > 0.0]

    # Probe a short distance downstream so the Junction opening itself cannot
    # be mistaken for an infinitely wide mouth.  Try a few local axial rows in
    # case the robust mouth centre lies directly on a rounded corner.
    probed_left = None
    probed_right = None
    for axial_scale in (1.0, 1.5, 2.0):
        probe_axial = LOCAL_PHYSICAL_WIDTH_PROBE_AXIAL * axial_scale
        left = probe_local_boundary_center_limit(
            descriptor,
            probe_axial,
            -1.0,
        )
        right = probe_local_boundary_center_limit(
            descriptor,
            probe_axial,
            1.0,
        )
        if left is not None and right is not None:
            probed_left = left
            probed_right = right
            break
    if probed_left is None or probed_right is None:
        return False

    contact_sides_ready = (
        len(left_contacts) >= LOCAL_PHYSICAL_WIDTH_MIN_SIDE_CONTACTS
        and len(right_contacts) >= LOCAL_PHYSICAL_WIDTH_MIN_SIDE_CONTACTS
    )
    left_center_limit = probed_left
    right_center_limit = probed_right
    source = "LOCAL_EDGE_CLEARANCE_PROBE"
    if contact_sides_ready:
        contact_left = linear_quantile(left_contacts, 0.50)
        contact_right = linear_quantile(right_contacts, 0.50)
        contact_tolerance = ROBOT_RADIUS * 4.0
        if (
            abs(contact_left - probed_left) <= contact_tolerance
            and abs(contact_right - probed_right) <= contact_tolerance
        ):
            left_center_limit = 0.5 * (probed_left + contact_left)
            right_center_limit = 0.5 * (probed_right + contact_right)
            source = "EGUCHI_CONTACTS+LOCAL_EDGE_PROBE"

    if right_center_limit - left_center_limit <= ROBOT_RADIUS * 4.0:
        return False
    lateral_midpoint = 0.5 * (left_center_limit + right_center_limit)
    _, motion_n = descriptor_local_basis(descriptor)
    descriptor.observed_mouth_position += (
        motion_n * lateral_midpoint
    )
    left_center_limit -= lateral_midpoint
    right_center_limit -= lateral_midpoint
    descriptor.physical_left_boundary_lateral = (
        left_center_limit - ROBOT_RADIUS
    )
    descriptor.physical_right_boundary_lateral = (
        right_center_limit + ROBOT_RADIUS
    )
    descriptor.observed_physical_width = (
        descriptor.physical_right_boundary_lateral
        - descriptor.physical_left_boundary_lateral
    )
    descriptor.physical_width_confident = True
    descriptor.physical_width_source = source
    descriptor.physical_boundary_sample_count = (
        len(left_contacts) + len(right_contacts) + 2
    )
    print(
        f"[LocalPhysicalWidth] uid={descriptor.uid} source={source} "
        f"flow={descriptor.observed_flow_width:.1f} "
        f"physical={descriptor.observed_physical_width:.1f} "
        f"left={descriptor.physical_left_boundary_lateral:.1f} "
        f"right={descriptor.physical_right_boundary_lateral:.1f} "
        f"samples={descriptor.physical_boundary_sample_count} "
        "confident=True"
    )
    return True


def local_physical_usable_half_width(
    descriptor: BranchDescriptor,
) -> float:
    """Robot-centre half span shared by Guard and Frontier rows."""
    if not descriptor.physical_width_confident:
        return 0.0
    return max(
        0.0,
        descriptor.observed_physical_width * 0.5
        - ROBOT_RADIUS
        - FRONTIER_LINE_EDGE_CLEARANCE,
    )


def local_frontier_shepherd_count(descriptor: BranchDescriptor) -> int:
    usable_width = local_physical_usable_half_width(descriptor) * 2.0
    return int(clamp(
        math.ceil(usable_width / max(SHEPHERD_TARGET_SLOT_SPACING, EPSILON))
        + 1,
        SHEPHERD_MIN_COUNT,
        SHEPHERD_MAX_COUNT,
    ))


def fit_slot_into_walkable_row(
    row_center: pygame.Vector2,
    slot: pygame.Vector2,
    clearance: float = ROBOT_RADIUS,
) -> pygame.Vector2:
    """Preserve the transverse row and move one member toward its centre.

    The returned object is the original ``slot`` when no fitting was needed,
    so callers can detect an adjustment by identity.
    """
    if is_walkable(slot, clearance):
        return slot
    low = 0.0
    high = 1.0
    fitted = row_center.copy()
    for _ in range(14):
        fraction = 0.5 * (low + high)
        candidate = row_center + (slot - row_center) * fraction
        if is_walkable(candidate, clearance):
            fitted = candidate
            low = fraction
        else:
            high = fraction
    return fitted


def frontier_shepherd_forward_direction(
    robot: "Robot",
) -> Optional[pygame.Vector2]:
    """Outward probe direction from the Branch's own observed tangent t."""
    descriptor = branch_motion_descriptor(robot.shepherd_branch)
    if descriptor is None:
        return None
    return descriptor_local_basis(descriptor)[0]


def frontier_shepherd_slot_target(
    robot: "Robot",
    axial_depth: float,
) -> Optional[pygame.Vector2]:
    """Place one FRONTIER_SHEPHERD in the Branch's observed local frame.

    Every line member shares ``axial_depth`` along t and keeps the lateral
    slot it claimed at commit time, so the row stays perpendicular to the
    measured tangent even when the Branch is not axis aligned.
    """
    descriptor = branch_motion_descriptor(robot.shepherd_branch)
    if descriptor is None or robot.frontier_local_lateral is None:
        return None
    axial = max(0.0, axial_depth)
    return local_coordinates_to_world(
        descriptor,
        axial,
        frontier_line_lateral_center + robot.frontier_local_lateral,
    )


def frontier_line_usable_half_width(descriptor: BranchDescriptor) -> float:
    """Use the same physical robot-centre span as the mouth Guard.

    Drift reserve limits centre correction; it must not shorten the physical
    row and create two side leaks.
    """
    return local_physical_usable_half_width(descriptor)


def build_frontier_line_local_slots(
    descriptor: BranchDescriptor,
    count: int,
    axial_depth: float,
) -> list[pygame.Vector2]:
    """Build the persistent frontier row in the Branch's observed local frame.

    The row centre and its orientation come from the descriptor
    (observed_mouth_position, t, n).  The lateral spread deliberately keeps the
    same observed width that formed the local Guard, so an existing
    axis-aligned map keeps its measured physical line width.
    """
    if descriptor.observed_mouth_position is None:
        return []
    usable_half = frontier_line_usable_half_width(descriptor)
    lateral_offsets = [0.0] if count <= 1 else [
        -usable_half + 2.0 * usable_half * index / (count - 1)
        for index in range(count)
    ]
    axial = max(0.0, axial_depth)
    row_center = local_coordinates_to_world(descriptor, axial, 0.0)
    slots = [
        local_coordinates_to_world(descriptor, axial, offset)
        for offset in lateral_offsets
    ]
    if (
        is_walkable(row_center, ROBOT_RADIUS)
        and not all(is_walkable(slot, ROBOT_RADIUS) for slot in slots)
    ):
        # Shrink every lateral offset by one shared factor.  Independent slot
        # fitting made one side collapse sooner and split the physical row.
        low = 0.0
        high = 1.0
        fitted_scale = 0.0
        for _ in range(14):
            scale = 0.5 * (low + high)
            candidates = [
                local_coordinates_to_world(
                    descriptor,
                    axial,
                    offset * scale,
                )
                for offset in lateral_offsets
            ]
            if all(is_walkable(candidate, ROBOT_RADIUS) for candidate in candidates):
                fitted_scale = scale
                low = scale
            else:
                high = scale
        lateral_offsets = [offset * fitted_scale for offset in lateral_offsets]
        slots = [
            local_coordinates_to_world(descriptor, axial, offset)
            for offset in lateral_offsets
        ]
    validate_local_guard_row(descriptor, row_center, slots)
    return slots


def local_guard_pending_reason(
    descriptor: Optional[BranchDescriptor],
) -> Optional[str]:
    if descriptor is None:
        return "NO_DESCRIPTOR"
    if descriptor.observed_mouth_position is None:
        return "NO_MOUTH_OBSERVATION"
    if descriptor.local_outgoing_direction.length_squared() <= EPSILON:
        return "NO_OUTGOING_DIRECTION"
    if len(descriptor.cohort_member_ids) < JUNCTION_COHORT_MIN_ROBOTS:
        return "INSUFFICIENT_COHORT_OBSERVATION"
    if not descriptor.direction_is_stable:
        return "DIRECTION_NOT_STABLE"
    if not descriptor.direction_is_mature:
        return "DIRECTION_NOT_MATURE"
    return None


def log_local_guard_pending(branch_uid: str, reason: str) -> None:
    signature = (branch_uid, reason)
    if signature in local_guard_pending_logged:
        return
    local_guard_pending_logged.add(signature)
    print(f"[LocalGuardPending] uid={branch_uid} reason={reason}")


def local_guard_observed_width(descriptor: BranchDescriptor) -> float:
    """Return only a confident locally observed physical wall width."""
    if (
        descriptor.physical_width_confident
        and descriptor.observed_physical_width > ROBOT_RADIUS * 2.0
    ):
        return descriptor.observed_physical_width
    if descriptor.uid not in local_guard_width_fallback_logged:
        local_guard_width_fallback_logged.add(descriptor.uid)
        print(
            f"[LocalGuardPending] uid={descriptor.uid} "
            "reason=PHYSICAL_WIDTH_UNCONFIRMED"
        )
    return 0.0


def local_guard_tube_contains(
    position: pygame.Vector2,
    descriptor: BranchDescriptor,
) -> bool:
    axial, lateral = branch_local_coordinates(position, descriptor)
    width = local_guard_observed_width(descriptor)
    return (
        axial >= -LOCAL_GUARD_MOUTH_TOLERANCE
        and abs(lateral)
        <= width * 0.5 + LOCAL_GUARD_LATERAL_RECRUIT_MARGIN
    )


def required_junction_guard_count(descriptor: BranchDescriptor) -> int:
    usable_width = local_physical_usable_half_width(descriptor) * 2.0
    return int(clamp(
        math.ceil(
            usable_width / max(JUNCTION_GUARD_COVERAGE, EPSILON)
        ) + 1,
        JUNCTION_GUARD_MIN_COUNT,
        JUNCTION_GUARD_MAX_COUNT,
    ))


def validate_local_guard_row(
    descriptor: BranchDescriptor,
    row_center: pygame.Vector2,
    slots: list[pygame.Vector2],
) -> None:
    tangent, normal = descriptor_local_basis(descriptor)
    if abs(tangent.dot(normal)) >= LOCAL_GUARD_ORTHOGONAL_EPSILON:
        raise AssertionError(f"invalid Guard basis for {descriptor.uid}")
    for slot in slots:
        if abs((slot - row_center).dot(tangent)) >= 1.0e-4:
            raise AssertionError(f"Guard row axial drift for {descriptor.uid}")


def build_local_junction_guard_slots(
    descriptor: BranchDescriptor,
    count: int,
    frontier_axial: Optional[float] = None,
    fit_walkable: bool = False,
) -> list[pygame.Vector2]:
    """Build one observed-mouth row using only descriptor-local geometry."""
    if descriptor.observed_mouth_position is None:
        return []
    tangent, normal = descriptor_local_basis(descriptor)
    usable_half = local_physical_usable_half_width(descriptor)
    if usable_half <= EPSILON:
        return []
    lateral_offsets = [0.0] if count <= 1 else [
        -usable_half + 2.0 * usable_half * index / (count - 1)
        for index in range(count)
    ]
    axial = max(
        JUNCTION_GUARD_BRANCH_INSET,
        frontier_axial if frontier_axial is not None
        else JUNCTION_GUARD_BRANCH_INSET,
    )
    row_center = descriptor.observed_mouth_position + tangent * axial
    slots = [row_center + normal * offset for offset in lateral_offsets]
    if fit_walkable and is_walkable(row_center, ROBOT_RADIUS):
        fitted_slots = []
        adjusted_count = 0
        max_adjustment = 0.0
        for slot in slots:
            # Preserve the exact transverse row and move only toward its
            # observed centre until the robot-radius footprint is walkable.
            fitted = fit_slot_into_walkable_row(row_center, slot)
            fitted_slots.append(fitted)
            if fitted is slot:
                continue
            adjusted_count += 1
            max_adjustment = max(
                max_adjustment,
                fitted.distance_to(slot),
            )
        if adjusted_count:
            print(
                f"[LocalGuardSlotFit] uid={descriptor.uid} "
                f"adjusted={adjusted_count}/{len(slots)} "
                f"max-shift={max_adjustment:.2f}"
            )
        slots = fitted_slots
    validate_local_guard_row(descriptor, row_center, slots)
    return slots


def observed_local_guard_frontier_axial(
    robots,
    descriptor: BranchDescriptor,
) -> Optional[float]:
    """Measure frontier depth from cohort poses projected from observed mouth."""
    cohort = [
        robot for robot in robots
        if robot.robot_id in descriptor.cohort_member_ids
        and robot.role == "NORMAL"
        and not robot.base_reserve
        and local_guard_tube_contains(robot.position, descriptor)
    ]
    if not cohort:
        return None
    depths = [
        branch_local_coordinates(robot.position, descriptor)[0]
        for robot in cohort
    ]
    return max(
        JUNCTION_GUARD_BRANCH_INSET,
        max(depths) + JUNCTION_GUARD_FRONTIER_MARGIN,
    )


def outward_branch_neighbor_count(
    robot: "Robot",
    descriptor: BranchDescriptor,
) -> int:
    """Count local-tube communication peers farther along measured t."""
    own_axial, _ = branch_local_coordinates(robot.position, descriptor)
    return sum(
        getattr(peer, "robot_id", -1) >= 0
        and getattr(peer, "role", None) == "NORMAL"
        and local_guard_tube_contains(peer.position, descriptor)
        and branch_local_coordinates(peer.position, descriptor)[0]
        > own_axial + JUNCTION_GUARD_TERMINAL_DEPTH_EPSILON
        for peer in robot.comm_neighbors
    )


def select_branch_guard_leader(
    robots,
    descriptor: BranchDescriptor,
    available_ids: set[int],
) -> Optional["Robot"]:
    candidates = [
        robot
        for robot in robots
        if robot.robot_id in available_ids
        and robot.robot_id in descriptor.cohort_member_ids
        and robot.role == "NORMAL"
        and not robot.base_reserve
        and robot.comm_neighbors
        and local_guard_tube_contains(robot.position, descriptor)
    ]
    if not candidates:
        return None
    terminal_candidates = [
        robot for robot in candidates
        if outward_branch_neighbor_count(robot, descriptor) == 0
    ]
    if terminal_candidates:
        candidates = terminal_candidates
    return min(
        candidates,
        key=lambda robot: (
            -branch_local_coordinates(robot.position, descriptor)[0],
            abs(branch_local_coordinates(robot.position, descriptor)[1]),
            -len(robot.comm_neighbors),
            robot.robot_id,
        ),
    )


def minimum_k_hop_guard_group(
    leader: "Robot",
    robots,
    available_ids: set[int],
    required_count: int,
    descriptor: BranchDescriptor,
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
                    and not getattr(peer, "base_reserve", False)
                    and local_guard_tube_contains(peer.position, descriptor)
                ):
                    next_frontier.add(peer_id)
        selected_ids.update(next_frontier)
        frontier_ids = next_frontier
        selected_hop = hop
        if len(selected_ids) >= required_count or not frontier_ids:
            break
    candidates = [by_id[robot_id] for robot_id in selected_ids]
    if len(candidates) < required_count:
        print(
            f"[LocalGuardRecruitFallback] uid={descriptor.uid} "
            "reason=K_HOP_COHORT_SHORTFALL"
        )
        candidates = sorted(
            (
                robot for robot in robots
                if robot.robot_id in available_ids
                and robot.role == "NORMAL"
                and not robot.base_reserve
                and local_guard_tube_contains(robot.position, descriptor)
            ),
            key=lambda robot: (
                0 if robot.robot_id in descriptor.cohort_member_ids else 1,
                leader.position.distance_squared_to(robot.position),
                robot.robot_id,
            ),
        )[:required_count]
        selected_hop = JUNCTION_GUARD_MAX_HOPS + 1
    else:
        candidates.sort(
            key=lambda robot: (
                0 if robot.robot_id in descriptor.cohort_member_ids else 1,
                leader.position.distance_squared_to(robot.position),
                robot.robot_id,
            )
        )
        candidates = candidates[:required_count]
    return candidates, selected_hop


def required_thick_mouth_guard_layers(
    robots,
    descriptor: BranchDescriptor,
    column_count: int,
) -> int:
    """Choose 2--4 layers from local mouth pressure without fixture regions."""
    nearby_normals = [
        robot for robot in robots
        if robot.role == "NORMAL"
        and not robot.base_reserve
        and local_guard_tube_contains(robot.position, descriptor)
        and robot.position.distance_to(descriptor.observed_mouth_position)
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
    descriptor: BranchDescriptor,
    target_count: int,
):
    """Grow an existing wall through the local-tube communication graph."""
    if not seed_guards:
        return []
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
    for _ in range(1, JUNCTION_GUARD_MAX_HOPS + 1):
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
                    or not local_guard_tube_contains(peer.position, descriptor)
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
                0 if item[1].robot_id in descriptor.cohort_member_ids else 1,
                item[1].position.distance_squared_to(
                    descriptor.observed_mouth_position
                ),
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
    if len(selected) < target_count:
        print(
            f"[LocalGuardRecruitFallback] uid={descriptor.uid} "
            f"reason=K_HOP_TARGET_SHORTFALL selected={len(selected)}/"
            f"{target_count}"
        )
    return list(selected.values())


def build_local_thick_mouth_guard_slots(
    descriptor: BranchDescriptor,
    count: int,
    column_count: int,
) -> list[pygame.Vector2]:
    """Create local t/n mouth rows without absolute Branch geometry."""
    slots = []
    remaining = count
    layer = 0
    while remaining > 0:
        row_count = min(column_count, remaining)
        axial = (
            JUNCTION_GUARD_BRANCH_INSET
            + layer * THICK_MOUTH_GUARD_LAYER_SPACING
        )
        slots.extend(build_local_junction_guard_slots(
            descriptor,
            row_count,
            axial,
        ))
        remaining -= row_count
        layer += 1
    return slots


def run_diagonal_local_guard_synthetic_validation() -> None:
    tangent = pygame.Vector2(0.819, -0.574).normalize()
    descriptor = BranchDescriptor(
        uid="SYNTHETIC-DIAGONAL",
        junction_uid="SYNTHETIC",
        fixture_key=None,
        local_outgoing_direction=tangent,
        local_return_direction=-tangent,
        observed_mouth_position=pygame.Vector2(100.0, 100.0),
        observed_width=60.0,
        observed_flow_width=48.0,
        observed_physical_width=60.0,
        physical_width_confident=True,
        physical_width_source="SYNTHETIC_LOCAL_BOUNDARIES",
        physical_left_boundary_lateral=-30.0,
        physical_right_boundary_lateral=30.0,
        cohort_member_ids=set(range(JUNCTION_COHORT_MIN_ROBOTS)),
    )
    slots = build_local_junction_guard_slots(descriptor, 5)
    local_t, local_n = descriptor_local_basis(descriptor)
    row_center = descriptor.observed_mouth_position + (
        local_t * JUNCTION_GUARD_BRANCH_INSET
    )
    validate_local_guard_row(descriptor, row_center, slots)
    lateral_values = [(slot - row_center).dot(local_n) for slot in slots]
    if lateral_values != sorted(lateral_values):
        raise AssertionError("synthetic local Guard slots are not transverse")
    true_normal = pygame.Vector2(-tangent.y, tangent.x)
    noisy_flow = tangent.rotate(6.0).normalize()
    locked_descriptor = BranchDescriptor(
        uid="SYNTHETIC-LOCKED-MOTION",
        junction_uid="SYNTHETIC",
        fixture_key=None,
        local_outgoing_direction=noisy_flow,
        local_return_direction=-noisy_flow,
        observed_mouth_position=pygame.Vector2(220.0, 180.0),
        observed_width=60.0,
        direction_last_estimate=noisy_flow.copy(),
        direction_is_stable=True,
        direction_is_mature=True,
        observed_physical_width=60.0,
        physical_width_confident=True,
        physical_width_source="SYNTHETIC_LOCAL_BOUNDARIES",
        physical_left_boundary_lateral=-30.0,
        physical_right_boundary_lateral=30.0,
    )
    sample_total = max(LOCAL_BRANCH_DIRECTION_MIN_SAMPLES + 2, 8)
    for index in range(sample_total):
        lateral = -28.0 + 56.0 * index / max(sample_total - 1, 1)
        locked_descriptor.observed_mouth_samples[index] = (
            locked_descriptor.observed_mouth_position
            + true_normal * lateral
            + tangent * (0.20 * ((index % 3) - 1))
        )
        locked_descriptor.cohort_recent_segments[index] = (
            noisy_flow.rotate((index % 3) - 1) * 18.0
        )
    if not lock_branch_motion_frame(locked_descriptor):
        raise AssertionError("synthetic motion frame did not lock")
    locked_t, locked_n = descriptor_local_basis(locked_descriptor)
    if locked_t.dot(tangent) < 0.999:
        raise AssertionError("synthetic motion frame followed noisy trajectory")
    frontier_slots = build_frontier_line_local_slots(
        locked_descriptor,
        7,
        40.0,
    )
    frontier_center = local_coordinates_to_world(
        locked_descriptor,
        40.0,
        0.0,
    )
    frontier_laterals = [
        (slot - frontier_center).dot(locked_n)
        for slot in frontier_slots
    ]
    if any(
        abs(left + right) > 1.0e-4
        for left, right in zip(frontier_laterals, reversed(frontier_laterals))
    ):
        raise AssertionError("synthetic Frontier slots are not symmetric")
    synthetic_usable_width = (
        local_physical_usable_half_width(locked_descriptor) * 2.0
    )
    synthetic_coverage = (
        max(frontier_laterals) - min(frontier_laterals)
    ) / max(synthetic_usable_width, EPSILON)
    if synthetic_coverage < 0.999:
        raise AssertionError("synthetic Frontier does not cover physical width")
    print(
        "[LocalGuardSynthetic] diagonal=PASS "
        f"dot={local_t.dot(local_n):.3e} slots={len(slots)} "
        f"locked-dot={locked_t.dot(tangent):.6f} "
        f"frontier-slots={len(frontier_slots)} "
        f"coverage={synthetic_coverage:.3f}"
    )


def local_guard_debug_summary(
    descriptor: BranchDescriptor,
    robots,
) -> str:
    reason = local_guard_pending_reason(descriptor)
    fixture = descriptor.fixture_key or "-"
    if reason is not None:
        return (
            f"{descriptor.uid} ({fixture}) pending={reason} "
            f"recent={descriptor.direction_sample_count} "
            f"downstream={descriptor.direction_downstream_travel:.1f} "
            f"t=({descriptor.local_outgoing_direction.x:.2f},"
            f"{descriptor.local_outgoing_direction.y:.2f}) "
            f"spread={math.degrees(descriptor.direction_angular_spread):.1f}deg "
            f"stable={descriptor.direction_is_stable}/"
            f"{descriptor.direction_stable_dwell:.2f}s "
            f"mature={descriptor.direction_is_mature}/"
            f"{descriptor.direction_mature_dwell:.2f}s"
        )
    tangent, normal = descriptor_local_basis(descriptor)
    mouth = descriptor.observed_mouth_position
    guards = [
        robot for robot in robots
        if robot.role == "JUNCTION_GUARD"
        and robot.junction_guard_branch_uid == descriptor.uid
    ]
    max_hop = max((robot.junction_guard_hop for robot in guards), default=0)
    return (
        f"{descriptor.uid} ({fixture}) mouth=({mouth.x:.0f},{mouth.y:.0f}) "
        f"t=({tangent.x:.2f},{tangent.y:.2f}) "
        f"n=({normal.x:.2f},{normal.y:.2f}) "
        f"w={local_guard_observed_width(descriptor):.1f} "
        f"recent={descriptor.direction_sample_count} "
        f"downstream={descriptor.direction_downstream_travel:.1f} "
        f"spread={math.degrees(descriptor.direction_angular_spread):.1f}deg "
        f"dir=STABLE+MATURE/"
        f"{descriptor.direction_stable_dwell:.2f}s "
        f"leader={descriptor.leader_id if descriptor.leader_id is not None else '-'} "
        f"k={max_hop} columns="
        f"{thick_mouth_guard_columns.get(fixture, 0)} layers="
        f"{thick_mouth_guard_layers.get(fixture, 0)}"
    )


def log_local_guard_formation_diagnostics(robots) -> None:
    """Report local-frame anchor errors without altering Guard geometry."""
    for branch_uid in ordered_discovered_branch_uids():
        descriptor = branch_descriptors_by_uid[branch_uid]
        guards = [
            robot for robot in robots
            if robot.role == "JUNCTION_GUARD"
            and robot.junction_guard_branch_uid == branch_uid
            and robot.junction_guard_anchor is not None
        ]
        if not guards:
            continue
        worst = max(
            guards,
            key=lambda robot: robot.position.distance_squared_to(
                robot.junction_guard_anchor
            ),
        )
        anchor_axial, anchor_lateral = branch_local_coordinates(
            worst.junction_guard_anchor,
            descriptor,
        )
        print(
            f"[LocalGuardPending] uid={branch_uid} "
            "reason=ANCHOR_SETTLING "
            f"max_error={worst.position.distance_to(worst.junction_guard_anchor):.1f} "
            f"anchor_local=({anchor_axial:.1f},{anchor_lateral:.1f}) "
            f"walkable={is_walkable(worst.junction_guard_anchor, worst.radius)}"
        )


def thick_mouth_guards_formed(robots, selected_branch: str) -> bool:
    visited = observed_visited_branches(robots)
    protected_branches = [
        branch
        for branch in junction_guard_groups
        if branch != selected_branch
        and branch not in visited
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
        robot.junction_guard_branch_uid = None
        robot.junction_guard_hop = -1
        robot.junction_guard_parent_id = None
        robot.junction_guard_layer = -1
        robot.is_branch_leader = False
        robot.shepherd_anchor = None
        robot.shepherd_origin = None
        robot.frontier_local_lateral = None
        robot.shepherd_branch = None
        robot.velocity.update(0.0, 0.0)
        robot.acceleration.update(0.0, 0.0)
        robot.filtered_acceleration.update(0.0, 0.0)
    frontier_line_branch = None
    frontier_line_depth = 0.0


def begin_junction_guard_formation(robots) -> None:
    """Form or preserve physical borders at observed branch frontiers.

    A mouth wall is a physical state, not a transient planning artifact. Once
    an unvisited branch has a valid thick K-hop wall, keep the same robots and
    slot assignments across branch switches. Only the selected branch
    is released later by ``commit_junction_guard_roles``.
    """
    global junction_guard_groups, junction_guard_formation_timer
    global junction_guard_frontier_depths
    global junction_guard_stable_dwell, junction_guard_status
    global distributed_consensus_branch
    global pending_branch_start
    global frontier_line_branch, frontier_line_depth
    global frontier_line_lateral_center
    global frontier_line_target_settled_ratio
    global frontier_line_current_span, frontier_line_target_span
    global frontier_line_physical_coverage_ratio
    global frontier_line_left_edge_gap, frontier_line_right_edge_gap
    global frontier_line_continuous
    global frontier_line_row_ready, frontier_line_last_diagnostic_time
    global frontier_dead_end_transition_last_log_time

    visited = observed_visited_branches(robots)
    # Preserve already-formed walls for branches that are still unvisited.
    # Releasing every guard here made a 3-layer wall collapse to a 1-layer
    # frontier and be re-estimated with a smaller column count on each switch.
    previous_groups = {
        branch: list(robot_ids)
        for branch, robot_ids in junction_guard_groups.items()
    }
    previous_layers = dict(thick_mouth_guard_layers)
    previous_columns = dict(thick_mouth_guard_columns)
    previous_depths = dict(junction_guard_frontier_depths)
    preserved_groups: dict[str, list[int]] = {}
    preserved_ids: set[int] = set()

    for branch, robot_ids in previous_groups.items():
        if (
            branch not in detected_branch_candidates
            or branch in visited
        ):
            continue
        branch_guards = [
            robot
            for robot in robots
            if robot.robot_id in robot_ids
            and robot.role == "JUNCTION_GUARD"
            and robot.junction_guard_branch == branch
            and robot.junction_guard_anchor is not None
        ]
        wall_is_valid = (
            len(branch_guards) >= JUNCTION_GUARD_MIN_COUNT
            and previous_layers.get(branch, 0)
            >= THICK_MOUTH_GUARD_MIN_LAYERS
            and previous_columns.get(branch, 0) > 0
        )
        if wall_is_valid:
            preserved_groups[branch] = [robot.robot_id for robot in branch_guards]
            preserved_ids.update(preserved_groups[branch])

    # Release only stale/selected transient roles. Valid unvisited walls stay
    # fixed in place and remain physical gatekeepers during the next transfer.
    for robot in robots:
        if robot.role == "JUNCTION_GUARD" and robot.robot_id in preserved_ids:
            continue
        if robot.role not in {"JUNCTION_GUARD", "FRONTIER_SHEPHERD"}:
            continue
        robot.role = "NORMAL"
        robot.junction_guard_anchor = None
        robot.junction_guard_branch = None
        robot.junction_guard_branch_uid = None
        robot.junction_guard_hop = -1
        robot.junction_guard_parent_id = None
        robot.junction_guard_layer = -1
        robot.is_branch_leader = False
        robot.shepherd_anchor = None
        robot.shepherd_origin = None
        robot.frontier_local_lateral = None
        robot.shepherd_branch = None
        robot.velocity.update(0.0, 0.0)
        robot.acceleration.update(0.0, 0.0)
        robot.filtered_acceleration.update(0.0, 0.0)

    frontier_line_branch = None
    frontier_line_depth = 0.0
    frontier_line_lateral_center = 0.0
    frontier_line_target_settled_ratio = 0.0
    frontier_line_current_span = 0.0
    frontier_line_target_span = 0.0
    frontier_line_physical_coverage_ratio = 0.0
    frontier_line_left_edge_gap = float("inf")
    frontier_line_right_edge_gap = float("inf")
    frontier_line_continuous = False
    frontier_line_row_ready = False
    frontier_line_last_diagnostic_time = float("-inf")
    frontier_dead_end_transition_last_log_time = float("-inf")
    # Only observed openings may be sealed. Closing an undiscovered branch
    # would leak the simulator's ground-truth map into the controller.
    branch_gate_states.clear()
    branch_gate_states.update({
        branch: (
            "CLOSED" if branch in detected_branch_candidates else "OPEN"
        )
        for branch in BRANCHES
    })
    record_distributed_consensus()
    print(
        "[Gate] inferred mouths only: "
        + ", ".join(
            f"{branch}={branch_gate_states[branch]}" for branch in BRANCHES
        )
    )
    distributed_consensus_branch = None
    junction_guard_groups = preserved_groups
    junction_guard_frontier_depths = {
        branch: previous_depths[branch]
        for branch in preserved_groups
        if branch in previous_depths
    }
    junction_guard_formation_timer = 0.0
    junction_guard_stable_dwell = 0.0
    pending_branch_start = None
    for branch in BRANCHES:
        if branch not in preserved_groups:
            thick_mouth_guard_layers[branch] = 0
            thick_mouth_guard_columns[branch] = 0
    available_ids = {
        robot.robot_id
        for robot in robots
        if robot.role == "NORMAL" and not robot.base_reserve
    }
    unavailable = []
    for fixture_key in sorted(detected_branch_candidates):
        if branch_uid_for_fixture(fixture_key) is not None:
            continue
        pending_uid = f"UNASSIGNED:{fixture_key}"
        log_local_guard_pending(pending_uid, "NO_DESCRIPTOR")
        unavailable.append(pending_uid)
    for branch_uid in ordered_discovered_branch_uids():
        descriptor = branch_descriptors_by_uid.get(branch_uid)
        reason = local_guard_pending_reason(descriptor)
        if reason is not None:
            log_local_guard_pending(branch_uid, reason)
            unavailable.append(branch_uid)
            continue
        if not lock_branch_motion_frame(descriptor):
            log_local_guard_pending(branch_uid, "MOTION_FRAME_UNAVAILABLE")
            unavailable.append(branch_uid)
            continue
        if not update_local_physical_mouth_width(descriptor):
            log_local_guard_pending(branch_uid, "PHYSICAL_WIDTH_UNCONFIRMED")
            unavailable.append(branch_uid)
            continue
        branch = descriptor.fixture_key
        if branch is None:
            log_local_guard_pending(branch_uid, "NO_FIXTURE_ADAPTER")
            unavailable.append(branch_uid)
            continue
        if (
            branch not in detected_branch_candidates
            or branch in visited
        ):
            continue
        if branch in preserved_groups:
            # This wall has already been physically formed. Do not re-elect
            # its leader or recompute its width during a branch switch.
            available_ids.difference_update(preserved_groups[branch])
            continue
        required_count = required_junction_guard_count(descriptor)
        frontier_depth = observed_local_guard_frontier_axial(
            robots,
            descriptor,
        )
        if frontier_depth is None:
            log_local_guard_pending(branch_uid, "NO_LOCAL_FRONTIER")
            unavailable.append(branch_uid)
            continue
        leader = select_branch_guard_leader(
            robots,
            descriptor,
            available_ids,
        )
        if leader is None:
            log_local_guard_pending(branch_uid, "NO_LOCAL_COHORT_LEADER")
            unavailable.append(branch_uid)
            continue
        descriptor.leader_id = leader.robot_id
        candidates, selected_hop = minimum_k_hop_guard_group(
            leader,
            robots,
            available_ids,
            required_count,
            descriptor,
        )
        slots = build_local_junction_guard_slots(
            descriptor,
            len(candidates),
            frontier_depth,
            fit_walkable=True,
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
            robot.junction_guard_branch_uid = branch_uid
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
        tangent, normal = descriptor_local_basis(descriptor)
        mouth = descriptor.observed_mouth_position
        print(
            f"[LocalGuard] uid={branch_uid} fixture={branch} "
            f"mouth=({mouth.x:.1f},{mouth.y:.1f}) "
            f"outgoing=({tangent.x:.3f},{tangent.y:.3f}) "
            f"normal=({normal.x:.3f},{normal.y:.3f}) "
            f"width={local_guard_observed_width(descriptor):.1f} "
            f"leader={leader.robot_id} required={required_count} "
            f"k={selected_hop}"
        )
        print(
            f"[LocalGuardSlot] uid={branch_uid} columns={len(branch_ids)} "
            "layers=1"
        )
        print(
            f"[Branch Frontier Guard] uid={branch_uid}, fixture={branch}, "
            f"depth={frontier_depth:.1f}, full={len(branch_ids)}, "
            f"terminal_leader={leader.robot_id}, "
            f"outward_degree="
            f"{outward_branch_neighbor_count(leader, descriptor)}, "
            f"k={selected_hop}"
        )
    junction_guard_status = (
        "FULL_GUARD_UNAVAILABLE:" + ",".join(unavailable)
        if unavailable
        else (
            "FORMING_FULL_GUARDS;PRESERVED="
            + ",".join(sorted(preserved_groups))
            if preserved_groups
            else "FORMING_FULL_GUARDS"
        )
    )

def junction_guard_readiness_snapshot(robots) -> dict[str, dict]:
    """Expose the exact per-Branch blockers used by Guard readiness."""
    visited = observed_visited_branches(robots)
    expected_branches = {
        branch
        for branch in detected_branch_candidates
        if branch not in visited
    }
    guard_branches = {
        robot.junction_guard_branch
        for robot in robots
        if robot.role == "JUNCTION_GUARD"
        and robot.junction_guard_branch is not None
    }
    by_id = {robot.robot_id: robot for robot in robots}
    snapshot: dict[str, dict] = {}
    for branch in sorted(expected_branches | guard_branches):
        group_ids = junction_guard_groups.get(branch)
        guards = [
            robot
            for robot in robots
            if robot.role == "JUNCTION_GUARD"
            and robot.junction_guard_branch == branch
        ]
        active_group_guard_ids = {
            robot.robot_id
            for robot in guards
            if robot.robot_id in set(group_ids or [])
        }
        missing_guard_count = max(
            0,
            len(group_ids or []) - len(active_group_guard_ids),
        )
        unsettled = []
        for robot in guards:
            anchor = robot.junction_guard_anchor
            error = (
                robot.position.distance_to(anchor)
                if anchor is not None
                else float("inf")
            )
            if (
                anchor is None
                or error > JUNCTION_GUARD_POSITION_TOLERANCE
            ):
                unsettled.append((error, robot))
        worst_error = 0.0
        worst_robot = None
        worst_anchor_walkable = None
        worst_anchor_comm_limited = None
        if unsettled:
            worst_error, worst_robot = max(
                unsettled,
                key=lambda item: item[0],
            )
            anchor = worst_robot.junction_guard_anchor
            worst_anchor_walkable = (
                is_walkable(anchor, worst_robot.radius)
                if anchor is not None
                else False
            )
            parent = worst_robot.comm_parent
            worst_anchor_comm_limited = bool(
                anchor is not None
                and worst_robot.connected_to_base
                and parent is not None
                and anchor.distance_to(parent.position)
                > COMM_GUARD_HARD_LIMIT
            )
        snapshot[branch] = {
            "missing_group": branch not in junction_guard_groups,
            "guard_count": len(guards),
            "unsettled_count": len(unsettled) + missing_guard_count,
            "max_anchor_error": worst_error,
            "worst_robot_id": (
                worst_robot.robot_id if worst_robot is not None else None
            ),
            "worst_anchor_walkable": worst_anchor_walkable,
            "worst_anchor_comm_limited": worst_anchor_comm_limited,
        }
    return snapshot


def log_junction_guard_wait_diagnostics(robots) -> None:
    global junction_guard_last_wait_log_time
    if simulation_time - junction_guard_last_wait_log_time < 1.0:
        return
    junction_guard_last_wait_log_time = simulation_time
    snapshot = junction_guard_readiness_snapshot(robots)
    if not snapshot:
        print("[JunctionGuardWait] no active Guard groups")
        return
    for branch, data in snapshot.items():
        error = data["max_anchor_error"]
        error_text = "inf" if not math.isfinite(error) else f"{error:.2f}"
        print(
            f"[JunctionGuardWait] branch={branch} "
            f"guards={data['guard_count']} "
            f"unsettled={data['unsettled_count']} "
            f"max-anchor-error={error_text} "
            f"robot={data['worst_robot_id'] if data['worst_robot_id'] is not None else '-'} "
            f"anchor-walkable={data['worst_anchor_walkable']} "
            f"comm-limited={data['worst_anchor_comm_limited']} "
            f"missing-group={data['missing_group']}"
        )


def junction_guard_wait_hud_summary(robots) -> str:
    snapshot = junction_guard_readiness_snapshot(robots)
    if not snapshot:
        return "Guard readiness: no active groups"
    parts = []
    for branch, data in snapshot.items():
        error = data["max_anchor_error"]
        error_text = "inf" if not math.isfinite(error) else f"{error:.1f}"
        robot_id = data["worst_robot_id"]
        parts.append(
            f"{branch}:unsettled={data['unsettled_count']} "
            f"err={error_text} robot={robot_id if robot_id is not None else '-'} "
            f"walkable={data['worst_anchor_walkable']}"
            f" comm-limited={data['worst_anchor_comm_limited']}"
        )
    return "Guard readiness: " + " | ".join(parts)


def distributed_vote_hud_summary() -> str:
    counts = (
        " | ".join(
            f"{branch_identity_label(branch_uid)}={count}"
            for branch_uid, count
            in sorted(last_distributed_vote_counts.items())
        )
        if last_distributed_vote_counts
        else "-"
    )
    return (
        f"Voting readiness: status={last_distributed_vote_status} "
        f"votes=[{counts}] voters={last_distributed_voter_count} "
        f"quorum={last_distributed_vote_quorum}"
    )


def junction_guards_formed(robots) -> bool:
    visited = observed_visited_branches(robots)
    guards = [robot for robot in robots if robot.role == "JUNCTION_GUARD"]
    if not guards or any(
        branch not in visited
        and branch in detected_branch_candidates
        and branch not in junction_guard_groups
        for branch in BRANCHES
    ):
        return False
    anchors_settled = all(
        robot.junction_guard_anchor is not None
        and robot.position.distance_to(robot.junction_guard_anchor)
        <= JUNCTION_GUARD_POSITION_TOLERANCE
        for robot in guards
    )
    if not anchors_settled:
        return False
    for branch in detected_branch_candidates:
        if branch in visited:
            continue
        descriptor = branch_motion_descriptor(branch)
        if (
            descriptor is None
            or not descriptor.physical_width_confident
        ):
            return False
        row = [
            robot for robot in guards
            if robot.junction_guard_branch == branch
            and robot.junction_guard_layer == 0
        ]
        if len(row) < required_junction_guard_count(descriptor):
            return False
        laterals = sorted(
            branch_local_coordinates(robot.position, descriptor)[1]
            for robot in row
        )
        usable_half = local_physical_usable_half_width(descriptor)
        usable_width = usable_half * 2.0
        span = laterals[-1] - laterals[0]
        left_gap = max(0.0, laterals[0] + usable_half)
        right_gap = max(0.0, usable_half - laterals[-1])
        maximum_gap = max(
            (
                right - left
                for left, right in zip(laterals, laterals[1:])
            ),
            default=0.0,
        )
        if (
            span / max(usable_width, EPSILON)
            < FRONTIER_LINE_MIN_SPAN_RATIO
            or left_gap > FRONTIER_LINE_MAX_EDGE_GAP
            or right_gap > FRONTIER_LINE_MAX_EDGE_GAP
            or maximum_gap > FRONTIER_LINE_MAX_INTERNAL_GAP
        ):
            return False
    return True


def commit_junction_guard_roles(robots, selected_branch: str) -> None:
    """Advance the selected line and thicken every unselected mouth guard."""
    global junction_guard_status, frontier_line_branch, frontier_line_depth
    global frontier_line_lateral_center
    global frontier_line_target_settled_ratio
    global frontier_line_current_span, frontier_line_target_span
    global frontier_line_physical_coverage_ratio
    global frontier_line_left_edge_gap, frontier_line_right_edge_gap
    global frontier_line_continuous
    global frontier_line_row_ready, frontier_line_last_diagnostic_time
    for branch, robot_ids in junction_guard_groups.items():
        branch_uid = branch_uid_for_fixture(branch)
        descriptor = branch_descriptors_by_uid.get(branch_uid)
        reason = local_guard_pending_reason(descriptor)
        if reason is not None:
            log_local_guard_pending(branch_uid or branch, reason)
            continue
        branch_guards = [
            robot for robot in robots if robot.robot_id in robot_ids
        ]
        if branch == selected_branch:
            # A preserved mouth wall can contain several complete axial rows.
            # Only its deepest (outward-most) cross-section becomes the moving
            # frontier. The support rows become NORMAL and join exploration.
            stored_columns = thick_mouth_guard_columns.get(branch, 0)
            seed_columns = (
                stored_columns if stored_columns > 0 else len(branch_guards)
            )
            column_count = int(clamp(
                max(local_frontier_shepherd_count(descriptor), seed_columns),
                JUNCTION_GUARD_MIN_COUNT,
                SHEPHERD_MAX_COUNT,
            ))
            # Recruit any missing cross-section members from the leader-rooted
            # K-hop neighbourhood before opening this mouth. The same dense
            # physical line is retained through exploration and backtracking.
            expanded_frontier_pool = expand_k_hop_mouth_guard_group(
                branch_guards,
                robots,
                descriptor,
                column_count,
            )
            if expanded_frontier_pool:
                branch_guards = expanded_frontier_pool
            deepest_layer = max(
                (robot.junction_guard_layer for robot in branch_guards),
                default=0,
            )
            frontier_guards = [
                robot
                for robot in branch_guards
                if robot.junction_guard_layer == deepest_layer
            ]
            if len(frontier_guards) < column_count:
                frontier_ids = {robot.robot_id for robot in frontier_guards}
                supplements = sorted(
                    (
                        robot
                        for robot in branch_guards
                        if robot.robot_id not in frontier_ids
                    ),
                    key=lambda robot: (
                        -robot.junction_guard_layer,
                        robot.robot_id,
                    ),
                )
                frontier_guards.extend(
                    supplements[:max(0, column_count - len(frontier_guards))]
                )
            elif len(frontier_guards) > column_count:
                frontier_guards = frontier_guards[:column_count]

            frontier_ids = {robot.robot_id for robot in frontier_guards}
            for robot in branch_guards:
                if robot.robot_id in frontier_ids:
                    continue
                robot.role = "NORMAL"
                robot.junction_guard_anchor = None
                robot.junction_guard_branch = None
                robot.junction_guard_branch_uid = None
                robot.junction_guard_hop = -1
                robot.junction_guard_parent_id = None
                robot.junction_guard_layer = -1
                robot.is_branch_leader = False
                robot.shepherd_anchor = None
                robot.shepherd_origin = None
                robot.frontier_local_lateral = None
                robot.shepherd_branch = None
                robot.velocity.update(0.0, 0.0)
                robot.acceleration.update(0.0, 0.0)
                robot.filtered_acceleration.update(0.0, 0.0)

            # The line starts exactly where the local Guard row already is, so
            # its slots are built on the same observed mouth frame instead of
            # the fixture corridor. Each member then keeps only its lateral
            # offset, and the whole row advances along the measured tangent.
            line_axial = junction_guard_frontier_depths.get(
                branch,
                FRONTIER_LINE_START_DEPTH,
            )
            line_slots = build_frontier_line_local_slots(
                descriptor,
                len(frontier_guards),
                line_axial,
            )
            if not line_slots:
                log_local_guard_pending(
                    branch_uid or branch,
                    "NO_LOCAL_FRONTIER_SLOTS",
                )
                continue
            assignment = assign_shepherd_slots(frontier_guards, line_slots)
            for robot, slot, _ in assignment:
                robot.role = "FRONTIER_SHEPHERD"
                robot.junction_guard_anchor = None
                robot.junction_guard_branch = branch
                robot.junction_guard_branch_uid = None
                robot.junction_guard_parent_id = None
                robot.junction_guard_layer = 0
                robot.shepherd_anchor = slot.copy()
                robot.shepherd_origin = slot.copy()
                robot.shepherd_branch = branch
                robot.frontier_local_lateral = branch_local_coordinates(
                    slot,
                    descriptor,
                )[1]
                robot.velocity.update(0.0, 0.0)
                robot.filtered_acceleration.update(0.0, 0.0)
            junction_guard_groups[branch] = [
                robot.robot_id for robot in frontier_guards
            ]
            frontier_line_branch = branch
            frontier_line_depth = line_axial
            frontier_line_lateral_center = 0.0
            frontier_line_target_settled_ratio = 0.0
            frontier_line_current_span = 0.0
            frontier_line_target_span = (
                max(robot.frontier_local_lateral for robot in frontier_guards)
                - min(robot.frontier_local_lateral for robot in frontier_guards)
                if len(frontier_guards) >= 2
                else 0.0
            )
            frontier_line_physical_coverage_ratio = 0.0
            frontier_line_left_edge_gap = float("inf")
            frontier_line_right_edge_gap = float("inf")
            frontier_line_continuous = False
            frontier_line_row_ready = False
            frontier_line_last_diagnostic_time = float("-inf")
            print(
                f"[FrontierLocalFrame] uid={branch_uid} fixture={branch} "
                f"axial={line_axial:.1f} members={len(assignment)} "
                f"flow-width={descriptor.observed_flow_width:.1f} "
                f"physical-width={local_guard_observed_width(descriptor):.1f} "
                "frame=OBSERVED_MOUTH_T_N"
            )
            thick_mouth_guard_layers[branch] = 0
            thick_mouth_guard_columns[branch] = 0
            continue
        existing_layers = thick_mouth_guard_layers.get(branch, 0)
        existing_columns = thick_mouth_guard_columns.get(branch, 0)
        existing_wall_is_valid = (
            existing_layers >= THICK_MOUTH_GUARD_MIN_LAYERS
            and existing_columns >= JUNCTION_GUARD_MIN_COUNT
            and len(branch_guards)
            >= THICK_MOUTH_GUARD_MIN_LAYERS * existing_columns
            and all(
                robot.role == "JUNCTION_GUARD"
                and robot.junction_guard_anchor is not None
                for robot in branch_guards
            )
        )
        if existing_wall_is_valid:
            # Keep an already formed wall unchanged until this branch itself
            # becomes selected. Do not re-estimate width or recruit a new row.
            print(
                f"[Thick Mouth Guard] preserved branch={branch}, "
                f"columns={existing_columns}, layers={existing_layers}, "
                f"robots={len(branch_guards)}"
            )
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
            descriptor,
            column_count,
        )
        target_count = column_count * desired_layers
        expanded_guards = expand_k_hop_mouth_guard_group(
            branch_guards,
            robots,
            descriptor,
            target_count,
        )
        actual_layers = math.ceil(
            len(expanded_guards) / max(column_count, 1)
        )
        mouth_slots = build_local_thick_mouth_guard_slots(
            descriptor,
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
            robot.junction_guard_branch_uid = branch_uid
            robot.junction_guard_layer = int(round(
                (
                    branch_local_coordinates(slot, descriptor)[0]
                    - JUNCTION_GUARD_BRANCH_INSET
                )
                / max(THICK_MOUTH_GUARD_LAYER_SPACING, EPSILON)
            ))
            robot.is_branch_leader = robot is leader
            robot.shepherd_anchor = None
            robot.shepherd_origin = None
            robot.frontier_local_lateral = None
            robot.shepherd_branch = None
            robot.velocity.update(0.0, 0.0)
            robot.acceleration.update(0.0, 0.0)
            robot.filtered_acceleration.update(0.0, 0.0)
            branch_ids.append(robot.robot_id)
        junction_guard_groups[branch] = branch_ids
        thick_mouth_guard_columns[branch] = column_count
        thick_mouth_guard_layers[branch] = actual_layers
        print(
            f"[LocalGuardSlot] uid={branch_uid} columns={column_count} "
            f"layers={actual_layers}"
        )
        print(
            f"[Thick Mouth Guard] uid={branch_uid}, fixture={branch}, "
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


def frontier_row_snapshot(
    frontiers,
    descriptor: BranchDescriptor,
) -> tuple[float, float, float, float, float, float, bool, bool]:
    """Measure settlement and full physical transverse coverage."""
    if not frontiers:
        return 0.0, 0.0, 0.0, 0.0, float("inf"), float("inf"), False, False
    targets = [
        frontier_shepherd_slot_target(robot, frontier_line_depth)
        for robot in frontiers
    ]
    settled = sum(
        target is not None
        and robot.position.distance_to(target)
        <= FRONTIER_LINE_TARGET_TOLERANCE
        for robot, target in zip(frontiers, targets)
    )
    settled_ratio = settled / len(frontiers)
    lateral_positions = [
        branch_local_coordinates(robot.position, descriptor)[1]
        for robot in frontiers
    ]
    current_span = (
        max(lateral_positions) - min(lateral_positions)
        if len(lateral_positions) >= 2
        else 0.0
    )
    target_laterals = [
        frontier_line_lateral_center + robot.frontier_local_lateral
        for robot in frontiers
        if robot.frontier_local_lateral is not None
    ]
    target_span = (
        max(target_laterals) - min(target_laterals)
        if len(target_laterals) >= 2
        else 0.0
    )
    usable_half = local_physical_usable_half_width(descriptor)
    usable_width = usable_half * 2.0
    ordered_laterals = sorted(lateral_positions)
    left_edge_gap = (
        max(0.0, ordered_laterals[0] + usable_half)
        if ordered_laterals else float("inf")
    )
    right_edge_gap = (
        max(0.0, usable_half - ordered_laterals[-1])
        if ordered_laterals else float("inf")
    )
    coverage_ratio = clamp(
        current_span / max(usable_width, EPSILON),
        0.0,
        1.0,
    )
    maximum_internal_gap = max(
        (
            right - left
            for left, right in zip(
                ordered_laterals,
                ordered_laterals[1:],
            )
        ),
        default=0.0,
    )
    continuous = (
        len(frontiers) >= local_frontier_shepherd_count(descriptor)
        and maximum_internal_gap <= FRONTIER_LINE_MAX_INTERNAL_GAP
    )
    span_ready = (
        target_span <= EPSILON
        or current_span >= target_span * FRONTIER_LINE_MIN_SPAN_RATIO
    )
    row_ready = (
        settled_ratio >= FRONTIER_LINE_TARGET_SETTLED_RATIO
        and span_ready
        and coverage_ratio >= FRONTIER_LINE_MIN_SPAN_RATIO
        and left_edge_gap <= FRONTIER_LINE_MAX_EDGE_GAP
        and right_edge_gap <= FRONTIER_LINE_MAX_EDGE_GAP
        and continuous
    )
    return (
        settled_ratio,
        current_span,
        target_span,
        coverage_ratio,
        left_edge_gap,
        right_edge_gap,
        continuous,
        row_ready,
    )


def update_frontier_lateral_center(
    robots,
    branch: str,
    descriptor: BranchDescriptor,
    dt: float,
) -> None:
    """Follow the robust median of the NORMAL cohort immediately behind."""
    global frontier_line_lateral_center
    samples = []
    for robot in robots:
        if (
            robot.role != "NORMAL"
            or robot.base_reserve
            or get_robot_region(robot.position) != branch
        ):
            continue
        axial, lateral = branch_local_coordinates(robot.position, descriptor)
        if (
            frontier_line_depth - FRONTIER_LINE_CENTER_WINDOW
            <= axial <= frontier_line_depth
        ):
            samples.append(lateral)
    if len(samples) < FRONTIER_LINE_CENTER_MIN_COHORT:
        return
    desired_center = linear_quantile(samples, 0.50)
    frontiers = get_frontier_shepherds(robots, branch)
    slot_half = max(
        (
            abs(robot.frontier_local_lateral)
            for robot in frontiers
            if robot.frontier_local_lateral is not None
        ),
        default=0.0,
    )
    physical_center_half = local_physical_usable_half_width(descriptor)
    # Drift reserve caps how far the row centre may follow the body; it does
    # not shorten the row itself. The physical wall clearance is the tighter
    # bound for the two edge robots.
    center_limit = min(
        FRONTIER_LINE_DRIFT_RESERVE,
        max(0.0, physical_center_half - slot_half),
    )
    desired_center = clamp(desired_center, -center_limit, center_limit)
    maximum_step = FRONTIER_LINE_CENTER_CORRECTION_SPEED * dt
    frontier_line_lateral_center += clamp(
        desired_center - frontier_line_lateral_center,
        -maximum_step,
        maximum_step,
    )


def refresh_frontier_row_readiness(robots, branch: str) -> bool:
    """Refresh the one full-width readiness gate without advancing depth."""
    global frontier_line_target_settled_ratio
    global frontier_line_current_span, frontier_line_target_span
    global frontier_line_physical_coverage_ratio
    global frontier_line_left_edge_gap, frontier_line_right_edge_gap
    global frontier_line_continuous, frontier_line_row_ready
    descriptor = branch_motion_descriptor(branch)
    frontiers = get_frontier_shepherds(robots, branch)
    if descriptor is None or not frontiers:
        frontier_line_row_ready = False
        return False
    (
        frontier_line_target_settled_ratio,
        frontier_line_current_span,
        frontier_line_target_span,
        frontier_line_physical_coverage_ratio,
        frontier_line_left_edge_gap,
        frontier_line_right_edge_gap,
        frontier_line_continuous,
        frontier_line_row_ready,
    ) = frontier_row_snapshot(frontiers, descriptor)
    return frontier_line_row_ready


def update_frontier_line_progress(robots, branch: str, dt: float) -> None:
    """Advance the intact line only as fast as the NORMAL body can follow."""
    global frontier_line_depth
    global frontier_line_target_settled_ratio
    global frontier_line_current_span, frontier_line_target_span
    global frontier_line_physical_coverage_ratio
    global frontier_line_left_edge_gap, frontier_line_right_edge_gap
    global frontier_line_continuous
    global frontier_line_row_ready
    frontiers = get_frontier_shepherds(robots, branch)
    if not frontiers or frontier_line_branch != branch:
        return
    # The advancing line lives in the observed local frame, so the support it
    # waits for is measured in that same frame (axial distance from the
    # observed mouth) rather than from the fixture Junction rectangle.
    descriptor = branch_motion_descriptor(branch)
    if descriptor is None:
        return
    refresh_frontier_row_readiness(robots, branch)
    if frontier_line_row_ready:
        update_frontier_lateral_center(robots, branch, descriptor, dt)
        refresh_frontier_row_readiness(robots, branch)
    if not frontier_line_row_ready:
        return
    normal_depths = [
        observed_branch_axial_depth(robot.position, descriptor)
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
    global frontier_line_lateral_center
    global frontier_line_target_settled_ratio
    global frontier_line_current_span, frontier_line_target_span
    global frontier_line_physical_coverage_ratio
    global frontier_line_left_edge_gap, frontier_line_right_edge_gap
    global frontier_line_continuous
    global frontier_line_row_ready
    frontiers = get_frontier_shepherds(robots, branch)
    if not frontiers:
        return []
    descriptor = branch_motion_descriptor(branch)
    if descriptor is None:
        return []
    boundary_depth = (
        frontier_line_depth
        if observed_boundary_depth is None
        else max(0.0, observed_boundary_depth)
    )
    local_slots = []
    for robot in frontiers:
        if robot.frontier_local_lateral is None:
            return []
        slot = local_coordinates_to_world(
            descriptor,
            boundary_depth,
            frontier_line_lateral_center + robot.frontier_local_lateral,
        )
        if not is_walkable(slot, robot.radius):
            return []
        local_slots.append((robot, slot))
    promoted = []
    for robot, slot in local_slots:
        robot.role = "SHEPHERD"
        robot.shepherd_anchor = slot.copy()
        robot.shepherd_origin = slot.copy()
        robot.shepherd_branch = branch
        # Keep the same local lateral identity through the handoff. Backflow
        # may still use its legacy geometry after this stationary boundary is
        # formed, but the transition itself does not rotate, shrink, or
        # re-auction the physical line.
        robot.junction_guard_anchor = None
        robot.velocity.update(0.0, 0.0)
        robot.acceleration.update(0.0, 0.0)
        robot.filtered_acceleration.update(0.0, 0.0)
        promoted.append(robot)
    frontier_line_branch = None
    frontier_line_depth = 0.0
    frontier_line_lateral_center = 0.0
    frontier_line_target_settled_ratio = 0.0
    frontier_line_current_span = 0.0
    frontier_line_target_span = 0.0
    frontier_line_physical_coverage_ratio = 0.0
    frontier_line_left_edge_gap = float("inf")
    frontier_line_right_edge_gap = float("inf")
    frontier_line_continuous = False
    frontier_line_row_ready = False
    print(
        f"[Frontier -> Shepherd] retained original IDs="
        f"{[robot.robot_id for robot in promoted]}; no re-election; "
        f"local-depth={boundary_depth:.1f}; local-slots-preserved=True"
    )
    return promoted


def _handoff_comm_components(all_robots, frontier_ids: set[int]):
    """Describe components from the already-populated comm_neighbors graph."""
    by_id = {robot.robot_id: robot for robot in all_robots}
    graph: dict[int, set[int]] = {robot_id: set() for robot_id in by_id}
    graph[-1] = set()
    for robot in all_robots:
        for neighbor in robot.comm_neighbors:
            neighbor_id = getattr(neighbor, "robot_id", None)
            if neighbor_id is None:
                continue
            graph.setdefault(robot.robot_id, set()).add(neighbor_id)
            graph.setdefault(neighbor_id, set()).add(robot.robot_id)
    component_by_id: dict[int, int] = {}
    component_rows = []
    for start in sorted(graph):
        if start in component_by_id:
            continue
        pending = [start]
        members = set()
        while pending:
            current = pending.pop()
            if current in members:
                continue
            members.add(current)
            pending.extend(graph.get(current, ()))
        component_id = min(members)
        for member in members:
            component_by_id[member] = component_id
        mobile_ids = sorted(member for member in members if member in by_id)
        roles = Counter(by_id[robot_id].role for robot_id in mobile_ids)
        component_rows.append({
            "component_id": component_id,
            "robot_count": len(mobile_ids),
            "contains_base": -1 in members,
            "connected_to_base_count": sum(
                by_id[robot_id].connected_to_base for robot_id in mobile_ids
            ),
            "frontier_candidate_count": sum(
                robot_id in frontier_ids for robot_id in mobile_ids
            ),
            "role_counts": json.dumps(roles, sort_keys=True),
            "robot_ids": json.dumps(mobile_ids),
        })
    return component_by_id, component_rows


def _evaluation_branch_geometry(branch: str):
    """Return fixture geometry only for diagnostics and visualization."""
    rectangles = {"LEFT": left_rect, "UP": up_rect, "RIGHT": right_rect}
    gt_tangents = {
        "LEFT": pygame.Vector2(-1.0, 0.0),
        "UP": pygame.Vector2(0.0, -1.0),
        "RIGHT": pygame.Vector2(1.0, 0.0),
    }
    mouth_centers = {
        "LEFT": pygame.Vector2(center_x - half_width, center_y),
        "UP": pygame.Vector2(center_x, center_y - half_width),
        "RIGHT": pygame.Vector2(center_x + half_width, center_y),
    }
    return rectangles[branch], gt_tangents[branch], mouth_centers[branch]


def _rect_boundary_violation(point: pygame.Vector2, rect: pygame.Rect) -> float:
    """Measure evaluation-only robot-centre clearance violation."""
    minimum_x = rect.left + ROBOT_RADIUS
    maximum_x = rect.right - ROBOT_RADIUS
    minimum_y = rect.top + ROBOT_RADIUS
    maximum_y = rect.bottom - ROBOT_RADIUS
    return max(
        0.0,
        minimum_x - point.x,
        point.x - maximum_x,
        minimum_y - point.y,
        point.y - maximum_y,
    )


def _record_motion_frame_projections(
    frontiers,
    descriptor: BranchDescriptor,
    contacted_depth: float,
) -> None:
    """Project retained slots with each recorded yaw for evaluation only."""
    if not MOTION_FRAME_BIAS_DIAGNOSTICS_ENABLED:
        return
    branch = descriptor.fixture_key or ""
    if not branch or not motion_frame_bias_diagnostics.needs_projection(branch):
        return
    source = motion_frame_bias_diagnostics.source_for_branch(branch)
    if source is None or descriptor.observed_mouth_position is None:
        return
    rectangle, gt_tangent, gt_mouth = _evaluation_branch_geometry(branch)
    gt_normal = pygame.Vector2(-gt_tangent.y, gt_tangent.x)
    lateral_slots = [
        frontier_line_lateral_center + robot.frontier_local_lateral
        for robot in frontiers
        if robot.frontier_local_lateral is not None
    ]
    if len(lateral_slots) != len(frontiers):
        return
    depths = (
        ("near", FRONTIER_LINE_START_DEPTH),
        ("middle", 0.5 * (FRONTIER_LINE_START_DEPTH + contacted_depth)),
        ("contacted", contacted_depth),
    )
    yaw_sources = (
        ("mouth_pca", source["mouth_pca_yaw_deg"]),
        ("recent_segment", source["recent_segment_yaw_deg"]),
        ("final", source["final_yaw_deg"]),
    )
    projection_rows = []
    lateral_midpoint = 0.5 * (min(lateral_slots) + max(lateral_slots))
    for yaw_source, yaw_value in yaw_sources:
        if yaw_value == "":
            continue
        yaw = math.radians(float(yaw_value))
        tangent = pygame.Vector2(math.cos(yaw), math.sin(yaw))
        normal = pygame.Vector2(-tangent.y, tangent.x)
        for depth_label, depth in depths:
            projected_center = (
                descriptor.observed_mouth_position
                + tangent * depth
                + normal * lateral_midpoint
            )
            expected_center = gt_mouth + gt_tangent * depth
            centerline_drift = (projected_center - expected_center).dot(gt_normal)
            for slot_index, (robot, lateral) in enumerate(zip(frontiers, lateral_slots)):
                target = (
                    descriptor.observed_mouth_position
                    + tangent * depth
                    + normal * lateral
                )
                projection_rows.append({
                    "branch": branch,
                    "yaw_source": yaw_source,
                    "yaw_deg": float(yaw_value),
                    "depth_label": depth_label,
                    "depth": depth,
                    "slot_index": slot_index,
                    "robot_id": robot.robot_id,
                    "lateral_slot": lateral,
                    "target_x": target.x,
                    "target_y": target.y,
                    "walkable": is_walkable(target, ROBOT_RADIUS),
                    "boundary_violation": _rect_boundary_violation(target, rectangle),
                    "predicted_centerline_lateral_drift": centerline_drift,
                    "corridor_min_x": rectangle.left,
                    "corridor_max_x": rectangle.right,
                    "corridor_min_y": rectangle.top,
                    "corridor_max_y": rectangle.bottom,
                    "evaluation_only_ground_truth": True,
                })
    motion_frame_bias_diagnostics.record_projections(branch, projection_rows)

    if branch == "RIGHT":
        sweep_rows = []
        for yaw_error in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
            yaw = math.radians(yaw_error)
            tangent = pygame.Vector2(math.cos(yaw), math.sin(yaw))
            normal = pygame.Vector2(-tangent.y, tangent.x)
            for depth in (50.0, 100.0, 150.0, 200.0, 250.0, 300.0):
                targets = [
                    descriptor.observed_mouth_position
                    + tangent * depth
                    + normal * lateral
                    for lateral in lateral_slots
                ]
                walkable = [is_walkable(target, ROBOT_RADIUS) for target in targets]
                violations = [
                    _rect_boundary_violation(target, rectangle) for target in targets
                ]
                sweep_rows.append({
                    "yaw_error_deg": yaw_error,
                    "axial_depth": depth,
                    "safe_slot_count": sum(walkable),
                    "unsafe_slot_count": len(walkable) - sum(walkable),
                    "slot_count": len(walkable),
                    "topmost_target_offset": max(target.y for target in targets) - center_y,
                    "bottommost_target_offset": min(target.y for target in targets) - center_y,
                    "maximum_boundary_violation": max(violations),
                    "origin_x": descriptor.observed_mouth_position.x,
                    "origin_y": descriptor.observed_mouth_position.y,
                    "slot_lateral_min": min(lateral_slots),
                    "slot_lateral_max": max(lateral_slots),
                    "evaluation_only_ground_truth": True,
                })
        motion_frame_bias_diagnostics.record_sweep(sweep_rows)


def _record_motion_frontier_frames(all_robots) -> None:
    """Record existing target tolerance and walkability for all frontiers."""
    if not MOTION_FRAME_BIAS_DIAGNOSTICS_ENABLED:
        return
    descriptor = branch_motion_descriptor(active_branch)
    frontiers = get_frontier_shepherds(all_robots, active_branch)
    if descriptor is None or not frontiers:
        return
    tangent, _ = descriptor_local_basis(descriptor)
    rows = []
    for robot in frontiers:
        target = frontier_shepherd_slot_target(robot, frontier_line_depth)
        if target is None or robot.frontier_local_lateral is None:
            continue
        actual_axial, actual_lateral = branch_local_coordinates(
            robot.position,
            descriptor,
        )
        target_axial, target_lateral = branch_local_coordinates(target, descriptor)
        rows.append({
            "frame": globals().get("headless_frame_count", 0),
            "timestamp": simulation_time,
            "branch": active_branch,
            "robot_id": robot.robot_id,
            "actual_axial": actual_axial,
            "target_axial": target_axial,
            "axial_error": actual_axial - target_axial,
            "actual_lateral": actual_lateral,
            "target_lateral": target_lateral,
            "lateral_error": actual_lateral - target_lateral,
            "target_distance_error": robot.position.distance_to(target),
            "target_attained": (
                robot.position.distance_to(target)
                <= FRONTIER_LINE_TARGET_TOLERANCE
            ),
            "target_walkable": is_walkable(target, ROBOT_RADIUS),
            "frontier_row_ready": frontier_line_row_ready,
            "contact_stall": (
                dead_end_inference_tracker.confirmation_mode == "CONTACT_STALL"
            ),
            "blocking_reason": dead_end_inference_tracker.blocking_reason,
            "motion_frame_yaw": _yaw_deg(tangent),
            "existing_target_tolerance": FRONTIER_LINE_TARGET_TOLERANCE,
        })
    motion_frame_bias_diagnostics.record_frontier_frame(rows)


def _record_handoff_resolution(
    frontiers,
    descriptor: BranchDescriptor,
    contacted_depth: float,
    resolved_depth: Optional[float],
    attempted_rows,
) -> None:
    """Copy resolver inputs/results to the downstream diagnostics sink."""
    if LOCAL_MOUTH_CROSSING_DIAGNOSTICS_ENABLED:
        local_mouth_crossing_diagnostics.record_handoff(
            branch=descriptor.fixture_key or "",
            timestamp=simulation_time,
            contacted_depth=contacted_depth,
            resolved_depth=resolved_depth,
            attempted_rows=attempted_rows,
        )
    if MOUTH_CROSSING_ORIGIN_DIAGNOSTICS_ENABLED:
        mouth_crossing_origin_diagnostics.record_handoff_outcome(
            descriptor.fixture_key or "",
            resolved_depth is not None,
        )
    if MOUTH_PCA_SAMPLE_DIAGNOSTICS_ENABLED:
        mouth_pca_sample_diagnostics.record_handoff_outcome(
            descriptor.fixture_key or "",
            resolved_depth is not None,
        )
    # Motion-frame projections are an independent, evaluation-only diagnostic.
    # Record them even when the pre-existing handoff-row diagnostics are disabled.
    _record_motion_frame_projections(frontiers, descriptor, contacted_depth)
    if not HANDOFF_DIAGNOSTICS_ENABLED:
        return
    outcome = "SUCCESS" if resolved_depth is not None else "FAILED"
    branch = descriptor.fixture_key or ""
    if not handoff_diagnostics.needs_outcome(branch, outcome):
        return
    all_robots = globals().get("robots", ())
    frontier_ids = {robot.robot_id for robot in frontiers}
    component_by_id, component_rows = _handoff_comm_components(
        all_robots,
        frontier_ids,
    )
    tangent, normal = descriptor_local_basis(descriptor)
    origin = descriptor.observed_mouth_position
    role_counts = Counter(robot.role for robot in all_robots)
    robot_rows = []
    for robot in all_robots:
        axial, lateral = branch_local_coordinates(robot.position, descriptor)
        is_frontier = robot.robot_id in frontier_ids
        if is_frontier and robot.frontier_local_lateral is not None:
            eligibility_reason = "INCLUDED_RETAINED_FRONTIER_WITH_SLOT"
        elif is_frontier:
            eligibility_reason = "INCLUDED_RETAINED_FRONTIER_MISSING_SLOT"
        elif robot.role != "FRONTIER_SHEPHERD":
            eligibility_reason = f"ROLE_{robot.role}_NOT_FRONTIER_SHEPHERD"
        elif robot.shepherd_branch != descriptor.fixture_key:
            eligibility_reason = "FRONTIER_FOR_OTHER_BRANCH"
        else:
            eligibility_reason = "NOT_PASSED_BY_EXISTING_FRONTIER_QUERY"
        slot_lateral = (
            "" if robot.frontier_local_lateral is None
            else frontier_line_lateral_center + robot.frontier_local_lateral
        )
        robot_rows.append({
            "frame": globals().get("headless_frame_count", 0),
            "timestamp": simulation_time,
            "branch": descriptor.fixture_key,
            "robot_id": robot.robot_id,
            "role": robot.role,
            "world_x": robot.position.x,
            "world_y": robot.position.y,
            "local_axial": axial,
            "local_lateral": lateral,
            "row_index": "",
            "row_quantization": "CONTINUOUS_DEPTH_SEARCH_NO_ROW_BIN",
            "slot_lateral": slot_lateral,
            "candidate_included": is_frontier,
            "eligibility_reason": eligibility_reason,
            "connected_to_base": robot.connected_to_base,
            "comm_hop": robot.comm_hop,
            "comm_neighbor_ids": json.dumps(sorted(
                getattr(neighbor, "robot_id", -999999)
                for neighbor in robot.comm_neighbors
            )),
            "comm_component": component_by_id.get(robot.robot_id, ""),
            "junction_guard_branch": robot.junction_guard_branch or "",
            "junction_guard_layer": robot.junction_guard_layer,
            "shepherd_branch": robot.shepherd_branch or "",
            "relay_index": robot.relay_index,
            "comm_bridge_branch": robot.comm_bridge_branch or "",
            "pebble_branch": robot.pebble_branch_key or "",
            "transfer_target": robot.transfer_target or "",
            "distributed_branch_decision": robot.distributed_branch_decision or "",
            "base_reserve": robot.base_reserve,
            "frontier_slot_locked": robot.frontier_local_lateral is not None,
            "known_visited_branches": json.dumps(sorted(robot.known_visited_branches)),
        })

    occupancy_rows = []
    for attempt_index, attempt in enumerate(attempted_rows):
        safe_ids = [item["robot_id"] for item in attempt["slots"] if item["walkable"]]
        unsafe_ids = [item["robot_id"] for item in attempt["slots"] if not item["walkable"]]
        occupancy_rows.append({
            "frame": globals().get("headless_frame_count", 0),
            "timestamp": simulation_time,
            "branch": descriptor.fixture_key,
            "attempt_index": attempt_index,
            "candidate_depth": attempt["depth"],
            "retreat_from_contacted_depth": contacted_depth - attempt["depth"],
            "safe_slot_count": len(safe_ids),
            "unsafe_slot_count": len(unsafe_ids),
            "all_slots_walkable": not unsafe_ids,
            "safe_robot_ids": json.dumps(safe_ids),
            "unsafe_robot_ids": json.dumps(unsafe_ids),
            "slot_world_coordinates": json.dumps([
                [item["robot_id"], item["world_x"], item["world_y"], item["walkable"]]
                for item in attempt["slots"]
            ]),
        })

    event = {
        "frame": globals().get("headless_frame_count", 0),
        "timestamp": simulation_time,
        "phase": phase.name,
        "subphase": dead_end_inference_tracker.confirmation_mode,
        "current_branch": branch,
        "current_branch_uid": descriptor.uid,
        "target_branch": transfer_branch or pending_branch_start or "",
        "result": outcome,
        "blocking_reason": (
            "READY_LOCAL_HANDOFF"
            if resolved_depth is not None
            else "NO_COMMON_LOCAL_HANDOFF_ROW"
        ),
        "robot_count": len(all_robots),
        "frontier_count": len(frontiers),
        "contacted_depth": contacted_depth,
        "resolved_depth": "" if resolved_depth is None else resolved_depth,
        "frontier_actual_depths": json.dumps([
            observed_branch_axial_depth(robot.position, descriptor)
            for robot in frontiers
        ]),
        "frontier_slot_laterals": json.dumps([
            None if robot.frontier_local_lateral is None
            else frontier_line_lateral_center + robot.frontier_local_lateral
            for robot in frontiers
        ]),
        "local_origin_x": "" if origin is None else origin.x,
        "local_origin_y": "" if origin is None else origin.y,
        "local_yaw_deg": math.degrees(math.atan2(tangent.y, tangent.x)),
        "local_t_x": tangent.x,
        "local_t_y": tangent.y,
        "local_n_x": normal.x,
        "local_n_y": normal.y,
        "frontier_lateral_center": frontier_line_lateral_center,
        "retreat_step": max(0.25, ROBOT_RADIUS * 0.25),
        "maximum_retreat": ROBOT_RADIUS * 6.0,
        "row_quantization": "CONTINUOUS_DEPTH_SEARCH_NO_ROW_BIN",
        "role_counts": json.dumps(role_counts, sort_keys=True),
        "normal_robot_count": role_counts["NORMAL"],
        "connected_robot_count": sum(robot.connected_to_base for robot in all_robots),
        "component_count": len(component_rows),
        "branch_visit_state": descriptor.visit_state,
        "visited_branches": json.dumps(sorted(observed_visited_branches(all_robots))),
        "branch_order": json.dumps(branch_fixture_order_plan),
        "junction_guard_groups": json.dumps({
            branch: len(robot_ids) for branch, robot_ids in junction_guard_groups.items()
        }, sort_keys=True),
        "frontier_line_branch": frontier_line_branch or "",
        "frontier_row_ready": frontier_line_row_ready,
        "branch_fill_state": branch_fill_feed_state,
    }
    handoff_diagnostics.record_handoff(
        event=event,
        robots=robot_rows,
        rows=occupancy_rows,
        components=component_rows,
    )


def _record_handoff_state(all_robots) -> None:
    """Record state/role residue only when an existing state field changes."""
    if not HANDOFF_DIAGNOSTICS_ENABLED:
        return
    role_counts = Counter(robot.role for robot in all_robots)
    branch_role_counts = {
        branch: Counter(
            robot.role for robot in all_robots
            if get_robot_region(robot.position) == branch
        )
        for branch in BRANCHES
    }
    subphase = (
        dead_end_inference_tracker.blocking_reason
        if phase == SimulationPhase.EXPLORE_BRANCH
        else (
            junction_guard_status
            if phase == SimulationPhase.FORM_JUNCTION_GUARDS
            else branch_fill_feed_state
        )
    )
    handoff_diagnostics.record_state({
        "frame": globals().get("headless_frame_count", 0),
        "timestamp": simulation_time,
        "phase": phase.name,
        "subphase": subphase,
        "current_branch": active_branch,
        "target_branch": transfer_branch or pending_branch_start or "",
        "handoff_status": (
            "CONFIRMED" if dead_end_inference_tracker.confirmed else "WAIT"
        ),
        "blocking_reason": dead_end_inference_tracker.blocking_reason,
        "role_counts": json.dumps(role_counts, sort_keys=True),
        "branch_role_counts": json.dumps({
            branch: dict(counts) for branch, counts in branch_role_counts.items()
        }, sort_keys=True),
        "normal_robot_count": role_counts["NORMAL"],
        "connected_robot_count": sum(robot.connected_to_base for robot in all_robots),
        "visited_branches": json.dumps(sorted(observed_visited_branches(all_robots))),
        "branch_order": json.dumps(branch_fixture_order_plan),
        "pending_branch_start": pending_branch_start or "",
        "transfer_branch": transfer_branch or "",
        "frontier_line_branch": frontier_line_branch or "",
        "frontier_count": len(get_frontier_shepherds(all_robots, active_branch)),
        "junction_guard_count": role_counts["JUNCTION_GUARD"],
        "shepherd_count": role_counts["SHEPHERD"],
        "relay_count": role_counts["RELAY"] + role_counts["TRUNK_RELAY"],
        "pebble_count": role_counts["PEBBLE"],
        "stale_transfer_target_count": sum(
            robot.transfer_target is not None for robot in all_robots
        ),
        "stale_frontier_slot_count": sum(
            robot.frontier_local_lateral is not None
            and robot.role != "FRONTIER_SHEPHERD"
            for robot in all_robots
        ),
    })


def resolve_local_frontier_handoff_depth(
    frontiers,
    descriptor: BranchDescriptor,
    contacted_depth: float,
) -> Optional[float]:
    """Find the deepest common local row supported by all retained slots.

    A real end wall can be slightly oblique to the locked motion frame. The
    contact depths then have a small spread, so the physical Shepherd boundary
    is the deepest nearby axial row where every original lateral slot still
    has robot-footprint clearance. This is a local t/n collision observation,
    not a fixture terminal coordinate.
    """
    lateral_slots = [
        frontier_line_lateral_center + robot.frontier_local_lateral
        for robot in frontiers
        if robot.frontier_local_lateral is not None
    ]
    attempted_rows = []

    def finish(result: Optional[float]) -> Optional[float]:
        _record_handoff_resolution(
            frontiers,
            descriptor,
            contacted_depth,
            result,
            attempted_rows,
        )
        return result

    if len(lateral_slots) != len(frontiers):
        return finish(None)

    def row_is_walkable(depth: float) -> bool:
        slots = []
        for robot, lateral in zip(frontiers, lateral_slots):
            world = local_coordinates_to_world(descriptor, depth, lateral)
            walkable = is_walkable(world, ROBOT_RADIUS)
            slots.append({
                "robot_id": robot.robot_id,
                "world_x": world.x,
                "world_y": world.y,
                "walkable": walkable,
            })
        attempted_rows.append({"depth": depth, "slots": slots})
        return all(item["walkable"] for item in slots)

    if row_is_walkable(contacted_depth):
        return finish(contacted_depth)
    retreat_step = max(0.25, ROBOT_RADIUS * 0.25)
    unsafe_depth = contacted_depth
    safe_depth = None
    maximum_retreat = ROBOT_RADIUS * 6.0
    retreat = retreat_step
    while retreat <= maximum_retreat + EPSILON:
        candidate = max(0.0, contacted_depth - retreat)
        if row_is_walkable(candidate):
            safe_depth = candidate
            break
        unsafe_depth = candidate
        retreat += retreat_step
    if safe_depth is None:
        return finish(None)
    # Recover the deepest common row to sub-pixel precision.
    low = safe_depth
    high = unsafe_depth
    if high < low:
        high = contacted_depth
    for _ in range(14):
        middle = 0.5 * (low + high)
        if row_is_walkable(middle):
            low = middle
        else:
            high = middle
    return finish(low)

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


def get_distributed_message(junction_id: str = CURRENT_JUNCTION_ID):
    junction_state = get_junction_state(junction_id)
    return (
        junction_id,
        junction_state.state_sequence,
        distributed_consensus_branch,
        f"DISTRIBUTED_{phase.name}",
        branch_gate_states.copy(),
    )


def pebble_flow_state_from_marker(
    pebble: "Robot",
) -> Optional[PebbleFlowState]:
    """Create a relay-safe fact from one force-free VISITED marker."""
    if (
        pebble.pebble_state != "VISITED"
        or not pebble.pebble_branch_uid
        or pebble.pebble_anchor is None
        or pebble.pebble_ingress_direction_local is None
        or pebble.pebble_return_direction_local is None
    ):
        return None
    ingress = pebble.pebble_ingress_direction_local
    return_direction = pebble.pebble_return_direction_local
    if (
        ingress.length_squared() <= EPSILON
        or return_direction.length_squared() <= EPSILON
    ):
        return None
    ingress = ingress.normalize()
    return_direction = return_direction.normalize()
    return PebbleFlowState(
        branch_uid=pebble.pebble_branch_uid,
        branch_key=pebble.pebble_branch_key,
        mouth_position=(pebble.pebble_anchor.x, pebble.pebble_anchor.y),
        ingress_direction_local=(ingress.x, ingress.y),
        return_direction_local=(return_direction.x, return_direction.y),
        completion_epoch=pebble.pebble_completion_epoch,
    )


def propagate_local_visited_knowledge(robots) -> None:
    """Exchange complete Pebble facts one local communication hop per frame."""
    global pebble_rx_logged
    snapshots = {
        robot.robot_id: dict(robot.known_pebble_flow_states)
        for robot in robots
    }
    for pebble in get_pebbles(robots):
        state = pebble_flow_state_from_marker(pebble)
        if state is not None:
            snapshots[pebble.robot_id][state.branch_uid] = state
    for robot in robots:
        previous_uids = set(robot.known_pebble_flow_states)
        learned_states = dict(robot.known_pebble_flow_states)
        for neighbor in robot.comm_neighbors:
            neighbor_id = getattr(neighbor, "robot_id", -1)
            if neighbor_id < 0:
                continue
            for branch_uid, state in snapshots.get(neighbor_id, {}).items():
                previous = learned_states.get(branch_uid)
                if (
                    previous is None
                    or state.completion_epoch > previous.completion_epoch
                ):
                    learned_states[branch_uid] = state
        newly_learned = set(learned_states) - previous_uids
        if not learned_states:
            continue
        robot.known_pebble_flow_states.update(learned_states)
        robot.known_visited_branch_uids.update(learned_states)
        for branch_uid, state in learned_states.items():
            branch = branch_fixture_for_uid(branch_uid)
            if branch is None:
                continue
            robot.known_visited_branches.add(branch)
            robot.local_branch_states[branch] = "VISITED"
            robot.branch_vote = None
            if (
                branch_uid in newly_learned
                and branch_uid not in pebble_rx_logged
            ):
                pebble_rx_logged.add(branch_uid)
                print(
                    f"[PebbleRx] robot={robot.robot_id} learned VISITED "
                    f"branch={branch_uid} ingress="
                    f"({state.ingress_direction_local[0]:.3f},"
                    f"{state.ingress_direction_local[1]:.3f})"
                )


def propagate_base_message(robots):
    """Compute Base-rooted paths and observe the peer-consensus state.

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
    ) = get_distributed_message()
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
        robot.received_branch = robot.distributed_branch_decision
        robot.received_command = (
            "PEER_CONSENSUS"
            if robot.distributed_branch_decision is not None
            else "LOCAL_VOTING"
        )
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
                "branch_states": robot.local_branch_states.copy(),
                "gate_states": gate_states.copy(),
                "selected_branch": robot.received_branch,
            }


def update_communication_system(robots, grid):
    update_communication_neighbors(robots, grid)
    propagate_local_visited_knowledge(robots)
    propagate_base_message(robots)


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
        or robot.role in {"PEBBLE", "RELAY", "TRUNK_RELAY"}
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
        or robot.role in {"PEBBLE", "RELAY", "TRUNK_RELAY"}
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
    if base_station is None or robot.role in {"PEBBLE", "RELAY", "TRUNK_RELAY"}:
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
            "pebble_connected": 0,
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
        "pebble_connected": sum(
            pebble.connected_to_base for pebble in get_pebbles(robots)
        ),
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
    forward = junction_inference_tracker.forward_direction
    if forward.length_squared() <= EPSILON:
        forward = pygame.Vector2(0.0, -1.0)
    else:
        forward = forward.normalize()
    length = base_length + corridor_width
    if length <= EPSILON:
        return 0.0
    return clamp((position - BASE_POSITION).dot(forward), 0.0, length)


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


def branch_uid_for_fixture(fixture_key: Optional[str]) -> Optional[str]:
    """Translate a map fixture key at the physics/decision boundary."""
    if fixture_key is None:
        return None
    return fixture_key_to_branch_uid.get(fixture_key)


def branch_fixture_for_uid(branch_uid: Optional[str]) -> Optional[str]:
    """Translate a dynamic UID only when legacy geometry needs a fixture."""
    if branch_uid is None:
        return None
    descriptor = branch_descriptors_by_uid.get(branch_uid)
    return descriptor.fixture_key if descriptor is not None else None


def branch_identity_label(branch_uid: Optional[str]) -> str:
    if branch_uid is None:
        return "-"
    fixture = branch_fixture_for_uid(branch_uid)
    return f"{branch_uid} ({fixture})" if fixture is not None else branch_uid


def discovered_branch_uids() -> set[str]:
    return set(branch_descriptors_by_uid)


def ordered_discovered_branch_uids() -> list[str]:
    return [
        descriptor.uid
        for descriptor in sorted(
            branch_descriptors_by_uid.values(),
            key=lambda item: (item.discovered_at, item.uid),
        )
    ]


def set_branch_descriptor_state(branch_uid: str, state: str) -> None:
    """Update UID state first and mirror it into fixture compatibility state."""
    descriptor = branch_descriptors_by_uid.get(branch_uid)
    if descriptor is None:
        return
    previous = descriptor.visit_state
    descriptor.visit_state = state
    fixture = descriptor.fixture_key
    if fixture is not None:
        branch_states[fixture] = state
    if previous != state:
        print(
            f"[BranchState] uid={branch_uid} fixture={fixture or '-'} "
            f"{previous}->{state}"
        )


def robust_displacement_direction(
    displacement_samples: list[pygame.Vector2],
    minimum_travel: float = LOCAL_BRANCH_DIRECTION_SEGMENT_LENGTH,
) -> tuple[Optional[pygame.Vector2], int, float]:
    """Estimate a cohort direction with a median centre and angular trimming."""
    usable = [
        displacement.copy()
        for displacement in displacement_samples
        if displacement.length() >= minimum_travel
    ]
    if len(usable) < LOCAL_BRANCH_DIRECTION_MIN_SAMPLES:
        return None, len(usable), math.pi
    units = [displacement.normalize() for displacement in usable]
    median_vector = pygame.Vector2(
        linear_quantile([unit.x for unit in units], 0.50),
        linear_quantile([unit.y for unit in units], 0.50),
    )
    if median_vector.length_squared() <= EPSILON:
        return None, len(usable), math.pi
    median_direction = median_vector.normalize()
    inlier_cosine = math.cos(LOCAL_BRANCH_DIRECTION_OUTLIER_ANGLE)
    inliers = [
        (displacement, unit)
        for displacement, unit in zip(usable, units)
        if unit.dot(median_direction) >= inlier_cosine
    ]
    if len(inliers) < LOCAL_BRANCH_DIRECTION_MIN_SAMPLES:
        return None, len(inliers), math.pi
    weighted = pygame.Vector2()
    for displacement, unit in inliers:
        weight = clamp(
            displacement.length()
            / max(minimum_travel, EPSILON),
            1.0,
            4.0,
        )
        weighted += unit * weight
    if weighted.length_squared() <= EPSILON:
        return None, len(inliers), math.pi
    estimate = weighted.normalize()
    angular_errors = sorted(
        math.acos(clamp(unit.dot(estimate), -1.0, 1.0))
        for _, unit in inliers
    )
    spread = linear_quantile(angular_errors, 0.80)
    return estimate, len(inliers), spread


def robust_mouth_cross_section_axis(
    descriptor: BranchDescriptor,
    flow_t: pygame.Vector2,
) -> tuple[Optional[pygame.Vector2], int, float, float]:
    """Infer the mouth cross-section axis from a robust local PCA.

    The mouth samples are robot-observed crossing poses.  Radial outliers are
    trimmed around their component-wise median before PCA, and the resulting
    principal axis is accepted only when it is both anisotropic and close to
    perpendicular to the independently observed cohort flow.
    """
    points = list(descriptor.observed_mouth_samples.values())
    if len(points) < LOCAL_BRANCH_DIRECTION_MIN_SAMPLES:
        return None, len(points), 0.0, 1.0
    center = pygame.Vector2(
        linear_quantile([point.x for point in points], 0.50),
        linear_quantile([point.y for point in points], 0.50),
    )
    distances = [point.distance_to(center) for point in points]
    cutoff = linear_quantile(distances, 0.90)
    inliers = [
        point for point, distance in zip(points, distances)
        if distance <= cutoff + EPSILON
    ]
    if len(inliers) < LOCAL_BRANCH_DIRECTION_MIN_SAMPLES:
        return None, len(inliers), 0.0, 1.0
    robust_center = pygame.Vector2(
        linear_quantile([point.x for point in inliers], 0.50),
        linear_quantile([point.y for point in inliers], 0.50),
    )
    offsets = [point - robust_center for point in inliers]
    covariance_xx = sum(offset.x * offset.x for offset in offsets) / len(offsets)
    covariance_xy = sum(offset.x * offset.y for offset in offsets) / len(offsets)
    covariance_yy = sum(offset.y * offset.y for offset in offsets) / len(offsets)
    trace = covariance_xx + covariance_yy
    discriminant = math.sqrt(max(
        0.0,
        (covariance_xx - covariance_yy) ** 2
        + 4.0 * covariance_xy * covariance_xy,
    ))
    major_value = 0.5 * (trace + discriminant)
    minor_value = max(0.0, 0.5 * (trace - discriminant))
    anisotropy = major_value / max(minor_value, EPSILON)
    if major_value <= EPSILON or anisotropy < FRONTIER_MOUTH_AXIS_MIN_ANISOTROPY:
        return None, len(inliers), 0.0, anisotropy
    if abs(covariance_xy) > EPSILON:
        axis = pygame.Vector2(
            major_value - covariance_yy,
            covariance_xy,
        )
    elif covariance_xx >= covariance_yy:
        axis = pygame.Vector2(1.0, 0.0)
    else:
        axis = pygame.Vector2(0.0, 1.0)
    if axis.length_squared() <= EPSILON:
        return None, len(inliers), 0.0, anisotropy
    axis = axis.normalize()
    reference_n = pygame.Vector2(-flow_t.y, flow_t.x)
    if axis.dot(reference_n) < 0.0:
        axis = -axis
    if abs(axis.dot(flow_t)) > math.sin(FRONTIER_MOUTH_AXIS_MAX_FLOW_ANGLE):
        return None, len(inliers), 0.0, anisotropy
    lateral_values = sorted(point.dot(axis) for point in inliers)
    width = max(
        0.0,
        linear_quantile(lateral_values, 0.95)
        - linear_quantile(lateral_values, 0.05),
    )
    return axis, len(inliers), width, anisotropy


def _yaw_deg(vector: pygame.Vector2) -> float:
    """Return a normalized diagnostic yaw without affecting estimation."""
    return (math.degrees(math.atan2(vector.y, vector.x)) + 180.0) % 360.0 - 180.0


def _motion_frame_pca_details(
    descriptor: BranchDescriptor,
    flow_t: pygame.Vector2,
) -> dict:
    """Recompute the existing robust PCA intermediates for observation only."""
    points = list(descriptor.observed_mouth_samples.values())
    result = {
        "mouth_point_count": len(points),
        "mouth_inlier_count": 0,
        "pca_covariance_xx": "",
        "pca_covariance_xy": "",
        "pca_covariance_yy": "",
        "pca_major_eigenvalue": "",
        "pca_minor_eigenvalue": "",
        "pca_explained_ratio": "",
        "pca_anisotropy": "",
        "mouth_cross_axis_yaw_deg": "",
        "mouth_pca_yaw_deg": "",
        "mouth_axis_sign_flip": False,
        "mouth_tangent_sign_flip": False,
        "mouth_pca_accepted": False,
        "mouth_rejection_reason": "",
    }
    if len(points) < LOCAL_BRANCH_DIRECTION_MIN_SAMPLES:
        result["mouth_rejection_reason"] = "INSUFFICIENT_MOUTH_POINTS"
        return result
    center = pygame.Vector2(
        linear_quantile([point.x for point in points], 0.50),
        linear_quantile([point.y for point in points], 0.50),
    )
    distances = [point.distance_to(center) for point in points]
    cutoff = linear_quantile(distances, 0.90)
    inliers = [
        point for point, distance in zip(points, distances)
        if distance <= cutoff + EPSILON
    ]
    result["mouth_inlier_count"] = len(inliers)
    if len(inliers) < LOCAL_BRANCH_DIRECTION_MIN_SAMPLES:
        result["mouth_rejection_reason"] = "INSUFFICIENT_PCA_INLIERS"
        return result
    robust_center = pygame.Vector2(
        linear_quantile([point.x for point in inliers], 0.50),
        linear_quantile([point.y for point in inliers], 0.50),
    )
    offsets = [point - robust_center for point in inliers]
    covariance_xx = sum(offset.x * offset.x for offset in offsets) / len(offsets)
    covariance_xy = sum(offset.x * offset.y for offset in offsets) / len(offsets)
    covariance_yy = sum(offset.y * offset.y for offset in offsets) / len(offsets)
    trace = covariance_xx + covariance_yy
    discriminant = math.sqrt(max(
        0.0,
        (covariance_xx - covariance_yy) ** 2
        + 4.0 * covariance_xy * covariance_xy,
    ))
    major = 0.5 * (trace + discriminant)
    minor = max(0.0, 0.5 * (trace - discriminant))
    anisotropy = major / max(minor, EPSILON)
    result.update({
        "pca_covariance_xx": covariance_xx,
        "pca_covariance_xy": covariance_xy,
        "pca_covariance_yy": covariance_yy,
        "pca_major_eigenvalue": major,
        "pca_minor_eigenvalue": minor,
        "pca_explained_ratio": major / max(trace, EPSILON),
        "pca_anisotropy": anisotropy,
    })
    if major <= EPSILON or anisotropy < FRONTIER_MOUTH_AXIS_MIN_ANISOTROPY:
        result["mouth_rejection_reason"] = "PCA_ANISOTROPY"
        return result
    if abs(covariance_xy) > EPSILON:
        axis = pygame.Vector2(major - covariance_yy, covariance_xy)
    elif covariance_xx >= covariance_yy:
        axis = pygame.Vector2(1.0, 0.0)
    else:
        axis = pygame.Vector2(0.0, 1.0)
    if axis.length_squared() <= EPSILON:
        result["mouth_rejection_reason"] = "ZERO_PCA_AXIS"
        return result
    axis = axis.normalize()
    reference_n = pygame.Vector2(-flow_t.y, flow_t.x)
    if axis.dot(reference_n) < 0.0:
        axis = -axis
        result["mouth_axis_sign_flip"] = True
    result["mouth_cross_axis_yaw_deg"] = _yaw_deg(axis)
    if abs(axis.dot(flow_t)) > math.sin(FRONTIER_MOUTH_AXIS_MAX_FLOW_ANGLE):
        result["mouth_rejection_reason"] = "PCA_NOT_TRANSVERSE_TO_FLOW"
        return result
    tangent = pygame.Vector2(axis.y, -axis.x).normalize()
    if tangent.dot(flow_t) < 0.0:
        tangent = -tangent
        result["mouth_tangent_sign_flip"] = True
    result["mouth_pca_yaw_deg"] = _yaw_deg(tangent)
    result["mouth_pca_accepted"] = True
    result["mouth_rejection_reason"] = ""
    return result


def _record_locked_motion_frame(
    descriptor: BranchDescriptor,
    flow_t: pygame.Vector2,
    sample_count: int,
    spread: float,
    robust_recent_available: bool,
    mouth_n: Optional[pygame.Vector2],
    motion_t: pygame.Vector2,
    source: str,
    origin_before: pygame.Vector2,
) -> None:
    """Record exact estimator outputs plus evaluation-only GT errors."""
    if not MOTION_FRAME_BIAS_DIAGNOSTICS_ENABLED:
        return
    branch = descriptor.fixture_key or descriptor.uid
    ground_truth_yaws = {"LEFT": 180.0, "UP": -90.0, "RIGHT": 0.0}
    gt_yaw = ground_truth_yaws.get(branch)
    if gt_yaw is None:
        return
    pca = _motion_frame_pca_details(descriptor, flow_t)
    mouth_yaw = pca["mouth_pca_yaw_deg"]
    recent_yaw = _yaw_deg(flow_t)
    final_yaw = _yaw_deg(motion_t)
    segment_lengths = [
        segment.length() for segment in descriptor.cohort_recent_segments.values()
    ]
    direction_last_yaw = (
        ""
        if descriptor.direction_last_estimate is None
        else _yaw_deg(descriptor.direction_last_estimate)
    )
    row = {
        "branch_id": descriptor.uid,
        "branch": branch,
        "frame": globals().get("headless_frame_count", 0),
        "timestamp": simulation_time,
        "motion_frame_source": source,
        "mouth_pca_yaw_deg": mouth_yaw,
        "recent_segment_yaw_deg": recent_yaw,
        "direction_last_estimate_yaw_deg": direction_last_yaw,
        "final_yaw_deg": final_yaw,
        "mouth_vs_recent_deg": (
            "" if mouth_yaw == "" else circular_distance_deg(mouth_yaw, recent_yaw)
        ),
        "mouth_vs_final_deg": (
            "" if mouth_yaw == "" else circular_distance_deg(mouth_yaw, final_yaw)
        ),
        "recent_vs_final_deg": circular_distance_deg(recent_yaw, final_yaw),
        "gt_yaw_deg": gt_yaw,
        "mouth_gt_error_deg": (
            "" if mouth_yaw == "" else circular_distance_deg(mouth_yaw, gt_yaw)
        ),
        "recent_gt_error_deg": circular_distance_deg(recent_yaw, gt_yaw),
        "final_gt_error_deg": circular_distance_deg(final_yaw, gt_yaw),
        "recent_segment_sample_count": sample_count,
        "recent_segment_stored_count": len(segment_lengths),
        "recent_segment_mean_length": (
            "" if not segment_lengths else sum(segment_lengths) / len(segment_lengths)
        ),
        "recent_segment_min_length": min(segment_lengths, default=""),
        "recent_segment_max_length": max(segment_lengths, default=""),
        "recent_angular_spread_deg": math.degrees(spread),
        "descriptor_age_at_lock": simulation_time - descriptor.discovered_at,
        "robust_recent_available": robust_recent_available,
        "fallback_used": not robust_recent_available,
        "selection_method": (
            "MOUTH_PCA_ORIENTATION_RECENT_SIGN_ONLY"
            if mouth_n is not None
            else "RECENT_SEGMENT_ORIENTATION"
        ),
        "weighted_fusion_used": False,
        "mouth_orientation_weight": "NOT_APPLICABLE_SOURCE_SELECTION",
        "recent_orientation_weight": "NOT_APPLICABLE_SOURCE_SELECTION",
        "recent_used_for_sign_alignment": mouth_n is not None,
        "origin_before_lock_x": origin_before.x,
        "origin_before_lock_y": origin_before.y,
        "final_origin_x": descriptor.observed_mouth_position.x,
        "final_origin_y": descriptor.observed_mouth_position.y,
        **pca,
    }
    motion_frame_bias_diagnostics.record_source(row)


def lock_branch_motion_frame(descriptor: BranchDescriptor) -> bool:
    """Freeze one robust t/n frame before any physical Guard starts moving."""
    if descriptor.motion_frame_locked:
        return True
    flow_t, sample_count, spread = robust_displacement_direction(
        list(descriptor.cohort_recent_segments.values()),
        LOCAL_BRANCH_DIRECTION_SEGMENT_LENGTH,
    )
    robust_recent_available = flow_t is not None
    if flow_t is None:
        fallback = descriptor.direction_last_estimate
        if fallback is None or fallback.length_squared() <= EPSILON:
            return False
        flow_t = fallback.normalize()
        sample_count = descriptor.direction_sample_count
        spread = descriptor.direction_angular_spread
    mouth_n, mouth_count, mouth_width, anisotropy = (
        robust_mouth_cross_section_axis(descriptor, flow_t)
    )
    if mouth_n is not None:
        motion_t = pygame.Vector2(mouth_n.y, -mouth_n.x).normalize()
        if motion_t.dot(flow_t) < 0.0:
            motion_t = -motion_t
        motion_n = pygame.Vector2(-motion_t.y, motion_t.x)
        source = "MOUTH_PCA_X_FLOW_SIGN"
    else:
        motion_t = flow_t.normalize()
        motion_n = pygame.Vector2(-motion_t.y, motion_t.x)
        source = "ROBUST_RECENT_SEGMENTS"
    if descriptor.observed_mouth_position is None:
        return False
    origin_before = descriptor.observed_mouth_position.copy()
    mouth_samples = list(descriptor.observed_mouth_samples.values())
    if mouth_samples:
        lateral_samples = sorted(
            point.dot(motion_n) for point in mouth_samples
        )
        # Crossing density is commonly biased toward the incoming turn.  The
        # robust envelope midpoint represents the physical cross-section
        # centre; a population median would drag every Guard slot to one wall.
        lateral_center = 0.5 * (
            linear_quantile(lateral_samples, 0.05)
            + linear_quantile(lateral_samples, 0.95)
        )
        descriptor.observed_mouth_position += motion_n * (
            lateral_center
            - descriptor.observed_mouth_position.dot(motion_n)
        )
    measured_width = (
        mouth_width
        if mouth_width > ROBOT_RADIUS * 2.0
        else descriptor.observed_width
    )
    descriptor.motion_t = motion_t.copy()
    descriptor.motion_n = motion_n.copy()
    descriptor.motion_frame_locked = True
    descriptor.motion_frame_source = source
    descriptor.motion_frame_sample_count = max(sample_count, mouth_count)
    descriptor.motion_frame_angular_spread = spread
    descriptor.motion_observed_width = max(
        measured_width,
        ROBOT_RADIUS * 2.0,
    )
    descriptor.observed_flow_width = descriptor.motion_observed_width
    _record_locked_motion_frame(
        descriptor,
        flow_t,
        sample_count,
        spread,
        robust_recent_available,
        mouth_n,
        motion_t,
        source,
        origin_before,
    )
    if MOUTH_PCA_SAMPLE_DIAGNOSTICS_ENABLED:
        branch = descriptor.fixture_key or ""
        gt_yaws = {"LEFT": 180.0, "UP": -90.0, "RIGHT": 0.0}
        if branch in gt_yaws:
            mouth_pca_sample_diagnostics.record_snapshot(
                branch=branch,
                branch_id=descriptor.uid,
                frame=globals().get("headless_frame_count", 0),
                timestamp=simulation_time,
                samples={
                    robot_id: (point.x, point.y)
                    for robot_id, point in descriptor.observed_mouth_samples.items()
                },
                flow_xy=(flow_t.x, flow_t.y),
                final_origin_xy=(
                    descriptor.observed_mouth_position.x,
                    descriptor.observed_mouth_position.y,
                ),
                motion_t_xy=(motion_t.x, motion_t.y),
                motion_n_xy=(motion_n.x, motion_n.y),
                gt_yaw_deg=gt_yaws[branch],
                branch_length=BRANCH_LENGTHS[branch],
                minimum_samples=LOCAL_BRANCH_DIRECTION_MIN_SAMPLES,
            )
    if MOUTH_CROSSING_ORIGIN_DIAGNOSTICS_ENABLED:
        branch = descriptor.fixture_key or ""
        gt_frames = {
            "LEFT": ((358.0, 350.0), (-1.0, 0.0), 180.0),
            "UP": ((400.0, 308.0), (0.0, -1.0), -90.0),
            "RIGHT": ((442.0, 350.0), (1.0, 0.0), 0.0),
        }
        if branch in gt_frames:
            gt_origin, gt_tangent, gt_yaw = gt_frames[branch]
            mouth_crossing_origin_diagnostics.record_snapshot(
                branch=branch,
                frame=globals().get("headless_frame_count", 0),
                timestamp=simulation_time,
                production_samples={
                    robot_id: (point.x, point.y)
                    for robot_id, point in descriptor.observed_mouth_samples.items()
                },
                flow_xy=(flow_t.x, flow_t.y),
                gt_origin_xy=gt_origin,
                gt_tangent_xy=gt_tangent,
                gt_yaw_deg=gt_yaw,
            )
    if LOCAL_MOUTH_CROSSING_DIAGNOSTICS_ENABLED:
        branch = descriptor.fixture_key or ""
        # These exact planes are evaluation labels only.  The sampler has
        # already selected and installed its pose before this scoring hook.
        evaluation_frames = {
            "LEFT": ((358.0, 350.0), (-1.0, 0.0), 180.0),
            "UP": ((400.0, 308.0), (0.0, -1.0), -90.0),
            "RIGHT": ((442.0, 350.0), (1.0, 0.0), 0.0),
        }
        if branch in evaluation_frames:
            gt_origin, gt_tangent, gt_yaw = evaluation_frames[branch]
            local_mouth_crossing_diagnostics.record_snapshot(
                branch=branch,
                frame=globals().get("headless_frame_count", 0),
                timestamp=simulation_time,
                flow_xy=(flow_t.x, flow_t.y),
                production_sample_ids=tuple(
                    descriptor.observed_mouth_samples
                ),
                gt_origin_xy=gt_origin,
                gt_tangent_xy=gt_tangent,
                gt_yaw_deg=gt_yaw,
            )
    print(
        f"[BranchMotionFrame] uid={descriptor.uid} source={source} "
        f"flow-t=({flow_t.x:.3f},{flow_t.y:.3f}) "
        f"motion-t=({motion_t.x:.3f},{motion_t.y:.3f}) "
        f"motion-n=({motion_n.x:.3f},{motion_n.y:.3f}) "
        f"segments={sample_count} mouth={mouth_count} "
        f"width={descriptor.motion_observed_width:.1f} "
        f"spread={math.degrees(spread):.2f}deg "
        f"mouth-anisotropy={anisotropy:.2f} locked=True"
    )
    return True


def recenter_descriptor_mouth_from_observations(
    descriptor: BranchDescriptor,
    direction: pygame.Vector2,
) -> None:
    """Centre the mouth on its fixed cohort-origin lateral envelope."""
    mouth_samples = list(descriptor.observed_mouth_samples.values())
    if (
        descriptor.observed_mouth_position is None
        or direction.length_squared() <= EPSILON
        or len(mouth_samples) < 2
    ):
        return
    normal = pygame.Vector2(-direction.y, direction.x)
    lateral_samples = [point.dot(normal) for point in mouth_samples]
    lateral_center = 0.5 * (
        min(lateral_samples) + max(lateral_samples)
    )
    descriptor.observed_mouth_position += normal * (
        lateral_center - descriptor.observed_mouth_position.dot(normal)
    )
    descriptor.observed_width = max(
        descriptor.observed_width,
        max(lateral_samples) - min(lateral_samples),
    )


def update_branch_direction_stability(
    descriptor: BranchDescriptor,
    robots,
    dt: float,
) -> bool:
    """Refine t from mature robots' recent, multi-frame motion segments."""
    # Stable is an intermediate state and never freezes refinement.  Mature
    # requires additional fresh downstream segments; only then is the tangent
    # finalized for Guard geometry.
    if descriptor.direction_is_mature:
        return True
    if descriptor.observed_mouth_position is None:
        descriptor.direction_is_stable = False
        descriptor.direction_is_mature = False
        return False
    by_id = {robot.robot_id: robot for robot in robots}
    reference = (
        descriptor.direction_last_estimate
        if descriptor.direction_last_estimate is not None
        else descriptor.local_outgoing_direction
    )
    downstream_start = LOCAL_BRANCH_DIRECTION_DOWNSTREAM_START
    segment_length = LOCAL_BRANCH_DIRECTION_SEGMENT_LENGTH
    # Direction labels and instantaneous velocity are deliberately excluded.
    # A robot first crosses a label-free radial downstream threshold.  Only
    # motion accumulated after that crossing can become a tangent sample.
    alignment_cosine = math.cos(math.radians(60.0))
    fresh_segment_count = 0
    for robot_id in descriptor.cohort_member_ids:
        robot = by_id.get(robot_id)
        mouth_sample = descriptor.observed_mouth_samples.get(robot_id)
        if robot is None or mouth_sample is None:
            continue
        downstream_travel = robot.position.distance_to(mouth_sample)
        previous_max = descriptor.cohort_downstream_travel.get(robot_id, 0.0)
        descriptor.cohort_downstream_travel[robot_id] = max(
            previous_max,
            downstream_travel,
        )
        if downstream_travel < downstream_start:
            continue
        anchor = descriptor.cohort_segment_anchors.get(robot_id)
        if anchor is None:
            descriptor.cohort_segment_anchors[robot_id] = (
                robot.position.copy()
            )
            continue
        segment = robot.position - anchor
        if segment.length() < segment_length:
            continue
        anchor_downstream = anchor.distance_to(mouth_sample)
        radial_growth = downstream_travel - anchor_downstream
        at_outbound_front = downstream_travel >= (
            previous_max - ROBOT_RADIUS * 0.25
        )
        if (
            not at_outbound_front
            or radial_growth < segment.length() * 0.25
            or (
                reference.length_squared() > EPSILON
                and segment.normalize().dot(reference.normalize())
                < alignment_cosine
            )
        ):
            continue
        descriptor.cohort_recent_segments[robot_id] = segment.copy()
        descriptor.cohort_segment_anchors[robot_id] = robot.position.copy()
        descriptor.cohort_segment_update_counts[robot_id] = (
            descriptor.cohort_segment_update_counts.get(robot_id, 0) + 1
        )
        fresh_segment_count += 1

    recent_segments = list(descriptor.cohort_recent_segments.values())
    estimate, sample_count, spread = robust_displacement_direction(
        recent_segments,
        segment_length,
    )
    descriptor.direction_sample_count = sample_count
    descriptor.direction_angular_spread = spread
    travel_samples = list(descriptor.cohort_downstream_travel.values())
    descriptor.direction_downstream_travel = (
        linear_quantile(travel_samples, 0.80)
        if travel_samples
        else 0.0
    )
    second_pass_samples = sum(
        update_count >= 2
        for update_count in descriptor.cohort_segment_update_counts.values()
    )
    if (
        simulation_time - descriptor.direction_last_diagnostic_time
        >= 1.0
    ):
        descriptor.direction_last_diagnostic_time = simulation_time
        print(
            f"[BranchDirection] uid={descriptor.uid} "
            f"recent-segments={sample_count}/"
            f"{len(descriptor.cohort_recent_segments)} "
            f"fresh={fresh_segment_count} second-pass={second_pass_samples} "
            f"downstream-p80={descriptor.direction_downstream_travel:.1f} "
            f"start={downstream_start:.1f} segment={segment_length:.1f} "
            f"t=({descriptor.local_outgoing_direction.x:.3f},"
            f"{descriptor.local_outgoing_direction.y:.3f}) "
            f"spread={math.degrees(spread):.1f}deg "
            f"dwell={descriptor.direction_stable_dwell:.2f} "
            f"stable={descriptor.direction_is_stable} "
            f"mature={descriptor.direction_is_mature}"
        )
    if estimate is None:
        descriptor.direction_stable_dwell = max(
            0.0,
            descriptor.direction_stable_dwell - dt,
        )
        descriptor.direction_is_stable = False
        descriptor.direction_mature_dwell = 0.0
        descriptor.direction_is_mature = False
        return False

    # Re-reading an unchanged set of samples must not manufacture stability.
    # Refinement advances only when at least one robot contributes a new
    # multi-frame segment farther along its outbound trajectory.
    if fresh_segment_count == 0:
        return descriptor.direction_is_mature

    refinement_dt = (
        dt
        if not math.isfinite(descriptor.direction_last_refinement_time)
        else clamp(
            simulation_time - descriptor.direction_last_refinement_time,
            dt,
            0.25,
        )
    )
    descriptor.direction_last_refinement_time = simulation_time
    descriptor.direction_refinement_count += 1

    previous_estimate = descriptor.direction_last_estimate
    stability_reference = descriptor.direction_stability_reference
    stable_now = False
    if previous_estimate is not None and stability_reference is not None:
        frame_delta = math.acos(clamp(
            previous_estimate.normalize().dot(estimate),
            -1.0,
            1.0,
        ))
        window_delta = math.acos(clamp(
            stability_reference.normalize().dot(estimate),
            -1.0,
            1.0,
        ))
        stable_now = (
            frame_delta <= LOCAL_BRANCH_DIRECTION_STABLE_DELTA
            and window_delta <= LOCAL_BRANCH_DIRECTION_STABLE_DELTA
            and spread <= LOCAL_BRANCH_DIRECTION_STABLE_SPREAD
        )
    if stable_now:
        descriptor.direction_stable_dwell += refinement_dt
    else:
        descriptor.direction_stable_dwell = 0.0
        descriptor.direction_stability_reference = estimate.copy()

    previous_direction = descriptor.local_outgoing_direction
    if previous_direction.length_squared() > EPSILON:
        filtered = (
            previous_direction.normalize()
            * (1.0 - LOCAL_BRANCH_DIRECTION_FILTER_ALPHA)
            + estimate * LOCAL_BRANCH_DIRECTION_FILTER_ALPHA
        )
        direction = (
            filtered.normalize()
            if filtered.length_squared() > EPSILON
            else estimate
        )
    else:
        direction = estimate
    descriptor.local_outgoing_direction = direction.copy()
    descriptor.local_return_direction = -direction
    descriptor.direction_last_estimate = estimate.copy()
    recenter_descriptor_mouth_from_observations(descriptor, direction)

    descriptor.direction_is_stable = (
        descriptor.direction_stable_dwell
        >= LOCAL_BRANCH_DIRECTION_STABLE_DWELL
    )
    descriptor.direction_mature_dwell = (
        descriptor.direction_mature_dwell + refinement_dt
        if (
            descriptor.direction_is_stable
            and descriptor.direction_refinement_count
            >= LOCAL_BRANCH_DIRECTION_MIN_REFINEMENT_UPDATES
            and second_pass_samples
            >= LOCAL_BRANCH_DIRECTION_MIN_SECOND_PASS_SAMPLES
        )
        else 0.0
    )
    was_mature = descriptor.direction_is_mature
    descriptor.direction_is_mature = (
        descriptor.direction_mature_dwell
        >= LOCAL_BRANCH_DIRECTION_MATURE_DWELL
    )
    if descriptor.direction_is_mature and not was_mature:
        print(
            f"[BranchDirectionMature] uid={descriptor.uid} "
            f"t=({direction.x:.3f},{direction.y:.3f}) "
            f"recent-segments={sample_count} "
            f"refinements={descriptor.direction_refinement_count} "
            f"second-pass={second_pass_samples} "
            f"downstream-p80={descriptor.direction_downstream_travel:.1f} "
            f"spread={math.degrees(spread):.2f}deg "
            f"stable-dwell={descriptor.direction_stable_dwell:.2f} "
            f"mature-dwell={descriptor.direction_mature_dwell:.2f}"
        )
    return descriptor.direction_is_mature


def register_discovered_branch(
    robots,
    fixture_key: str,
    member_ids: set[int],
    cohort_origins: dict[tuple[int, str], pygame.Vector2],
) -> BranchDescriptor:
    """Create/reassociate a UID from a physically validated local cohort."""
    global branch_discovery_counter

    members = [robot for robot in robots if robot.robot_id in member_ids]
    origins = [
        cohort_origins[(robot.robot_id, fixture_key)]
        for robot in members
        if (robot.robot_id, fixture_key) in cohort_origins
    ]
    mouth = (
        sum(origins, pygame.Vector2()) / len(origins)
        if origins
        else None
    )

    mouth_observations = {
        robot.robot_id: cohort_origins[(robot.robot_id, fixture_key)].copy()
        for robot in members
        if (robot.robot_id, fixture_key) in cohort_origins
    }
    initial_displacements = [
        robot.position - mouth_observations[robot.robot_id]
        for robot in members
        if robot.robot_id in mouth_observations
    ]
    initial_direction, initial_sample_count, initial_spread = (
        robust_displacement_direction(
            initial_displacements,
            JUNCTION_COHORT_MIN_TRAVEL,
        )
    )
    measured_direction = (
        initial_direction.copy()
        if initial_direction is not None
        else pygame.Vector2()
    )

    lateral_samples = []
    if measured_direction.length_squared() > EPSILON:
        lateral = pygame.Vector2(
            -measured_direction.y,
            measured_direction.x,
        )
        lateral_samples = [point.dot(lateral) for point in origins]
    observed_width = (
        max(lateral_samples) - min(lateral_samples)
        if len(lateral_samples) >= 2
        else 0.0
    )
    if mouth is not None and len(lateral_samples) >= 2:
        # The arithmetic mean can be biased toward the incoming turn. Recenter
        # the observed mouth on the cohort's own lateral envelope so a local
        # transverse row spans the measured tube rather than that turn bias.
        observed_lateral_center = 0.5 * (
            min(lateral_samples) + max(lateral_samples)
        )
        mouth += lateral * (
            observed_lateral_center - mouth.dot(lateral)
        )
    leader = max(
        members,
        key=lambda robot: (
            (robot.position - mouth).dot(measured_direction)
            if mouth is not None else 0.0,
            -robot.robot_id,
        ),
        default=None,
    )

    existing_uid = fixture_key_to_branch_uid.get(fixture_key)
    if existing_uid is None and mouth is not None:
        # Direction/proximity matching supports stable re-observation even if
        # a future fixture adapter is temporarily unavailable or renamed.
        for candidate in branch_descriptors_by_uid.values():
            if candidate.junction_uid != CURRENT_JUNCTION_ID:
                continue
            if candidate.fixture_key not in {None, fixture_key}:
                continue
            candidate_mouth = candidate.observed_mouth_position
            if (
                candidate_mouth is not None
                and candidate.local_outgoing_direction.length_squared()
                > EPSILON
                and measured_direction.length_squared() > EPSILON
                and candidate.local_outgoing_direction.dot(measured_direction)
                >= 0.90
                and candidate_mouth.distance_to(mouth) <= corridor_width
            ):
                existing_uid = candidate.uid
                break

    if existing_uid is None:
        existing_uid = f"{CURRENT_JUNCTION_ID}-B{branch_discovery_counter}"
        branch_discovery_counter += 1
        descriptor = BranchDescriptor(
            uid=existing_uid,
            junction_uid=CURRENT_JUNCTION_ID,
            fixture_key=fixture_key,
            local_outgoing_direction=measured_direction.copy(),
            local_return_direction=-measured_direction,
            observed_mouth_position=mouth.copy() if mouth is not None else None,
            observed_width=observed_width,
            leader_id=leader.robot_id if leader is not None else None,
            cohort_member_ids=set(member_ids),
            observed_mouth_samples=mouth_observations,
            direction_last_estimate=(
                measured_direction.copy()
                if measured_direction.length_squared() > EPSILON
                else None
            ),
            # Initial mouth-to-current displacement is provisional only.  It
            # never counts toward recent-segment maturity or Guard readiness.
            direction_sample_count=0,
            direction_angular_spread=math.pi,
            discovered_at=simulation_time,
        )
        branch_descriptors_by_uid[existing_uid] = descriptor
        print(
            f"[BranchDiscovery] uid={existing_uid} fixture={fixture_key} "
            f"members={len(member_ids)} direction=PROVISIONAL "
            f"({measured_direction.x:.3f},{measured_direction.y:.3f}) "
            f"samples={initial_sample_count} "
            f"width={observed_width:.1f}"
        )
    else:
        descriptor = branch_descriptors_by_uid[existing_uid]
        descriptor.fixture_key = fixture_key
        if measured_direction.length_squared() > EPSILON:
            descriptor.local_outgoing_direction = measured_direction.copy()
            descriptor.local_return_direction = -measured_direction
            descriptor.direction_last_estimate = measured_direction.copy()
        if mouth is not None:
            descriptor.observed_mouth_position = mouth.copy()
        descriptor.observed_width = max(descriptor.observed_width, observed_width)
        descriptor.cohort_member_ids.update(member_ids)
        descriptor.observed_mouth_samples.update(mouth_observations)
        if descriptor.leader_id is None and leader is not None:
            descriptor.leader_id = leader.robot_id

    fixture_key_to_branch_uid[fixture_key] = existing_uid
    branch_uid_to_fixture_key[existing_uid] = fixture_key
    branch_local_uids[fixture_key] = existing_uid
    for robot in robots:
        robot.local_branch_uid_by_key[fixture_key] = existing_uid
        if fixture_key in robot.local_ingress_tangents:
            robot.local_ingress_tangents_by_uid[existing_uid] = (
                robot.local_ingress_tangents[fixture_key].copy()
            )
        if fixture_key in robot.local_branch_ingress_points:
            robot.local_branch_ingress_points_by_uid[existing_uid] = (
                robot.local_branch_ingress_points[fixture_key].copy()
            )
        if fixture_key in robot.local_ingress_observed_travel:
            robot.local_ingress_observed_travel_by_uid[existing_uid] = (
                robot.local_ingress_observed_travel[fixture_key]
            )
    return descriptor


def record_distributed_consensus(
    selected_branch: Optional[str] = None,
    clear_selection: bool = False,
    junction_id: str = CURRENT_JUNCTION_ID,
) -> None:
    """Record consensus for diagnostics; robots retain their own local copy."""
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


def get_pebbles(robots) -> list["Robot"]:
    return [robot for robot in robots if robot.role == "PEBBLE"]


def observed_visited_branches(robots) -> set[str]:
    """Compatibility fixture view of UID-based Pebble observations."""
    return {
        pebble.pebble_branch_key
        for pebble in get_pebbles(robots)
        if pebble.pebble_state == "VISITED"
        and pebble.pebble_branch_key is not None
    }


def observed_visited_branch_uids(robots) -> set[str]:
    """Primary distributed VISITED facts, keyed by dynamic Branch UID."""
    return {
        pebble.pebble_branch_uid
        for pebble in get_pebbles(robots)
        if pebble.pebble_state == "VISITED"
        and pebble.pebble_branch_uid is not None
    }


def return_mobile_target_count(robots) -> int:
    """All non-marker robots must return; persistent Pebbles stay in place."""
    return len(robots) - len(get_pebbles(robots))


def mirror_local_ingress_observation_to_uid(robot: "Robot", branch: str) -> None:
    branch_uid = branch_uid_for_fixture(branch)
    if branch_uid is None:
        return
    tangent = robot.local_ingress_tangents.get(branch)
    mouth = robot.local_branch_ingress_points.get(branch)
    if tangent is not None:
        robot.local_ingress_tangents_by_uid[branch_uid] = tangent.copy()
    if mouth is not None:
        robot.local_branch_ingress_points_by_uid[branch_uid] = mouth.copy()
    robot.local_ingress_observed_travel_by_uid[branch_uid] = (
        robot.local_ingress_observed_travel.get(branch, 0.0)
    )


def update_local_ingress_tangents(robots) -> None:
    """Estimate ingress from each robot's own mouth-to-outbound trajectory."""
    for robot in robots:
        if robot.role not in {"NORMAL", "FRONTIER_SHEPHERD"}:
            continue
        branch = get_robot_region(robot.position)
        if branch not in BRANCHES:
            continue
        velocity = robot.observed_velocity
        if velocity.length_squared() <= EPSILON:
            continue
        mouth_observation = robot.local_branch_ingress_points.get(branch)
        if mouth_observation is None:
            robot.local_branch_ingress_points[branch] = robot.position.copy()
            robot.local_ingress_tangents[branch] = velocity.normalize()
            robot.local_ingress_observed_travel[branch] = 0.0
            mirror_local_ingress_observation_to_uid(robot, branch)
            continue
        outbound_displacement = robot.position - mouth_observation
        robot.local_ingress_observed_travel[branch] = max(
            robot.local_ingress_observed_travel.get(branch, 0.0),
            outbound_displacement.length(),
        )
        if (
            outbound_displacement.length()
            < PEBBLE_INGRESS_MIN_OBSERVED_TRAVEL
            or velocity.dot(outbound_displacement) <= 0.0
        ):
            mirror_local_ingress_observation_to_uid(robot, branch)
            continue
        observed_tangent = outbound_displacement.normalize()
        previous_tangent = robot.local_ingress_tangents.get(branch)
        if previous_tangent is None or previous_tangent.length_squared() <= EPSILON:
            robot.local_ingress_tangents[branch] = observed_tangent
            mirror_local_ingress_observation_to_uid(robot, branch)
            continue
        if previous_tangent.dot(observed_tangent) < 0.0:
            previous_tangent = -previous_tangent
        filtered = (
            previous_tangent.normalize()
            * (1.0 - PEBBLE_INGRESS_DIRECTION_FILTER_ALPHA)
            + observed_tangent * PEBBLE_INGRESS_DIRECTION_FILTER_ALPHA
        )
        if filtered.length_squared() > EPSILON:
            robot.local_ingress_tangents[branch] = filtered.normalize()
        mirror_local_ingress_observation_to_uid(robot, branch)


def locally_consensed_ingress_direction(
    marker: "Robot",
    branch: str,
) -> Optional[pygame.Vector2]:
    """Fuse only one-hop cohort trajectories that actually travelled outward."""
    sources = [marker] + [
        neighbor for neighbor in marker.comm_neighbors
        if getattr(neighbor, "robot_id", -1) >= 0
    ]
    observations = []
    for source in sources:
        tangent = getattr(source, "local_ingress_tangents", {}).get(branch)
        travel = getattr(source, "local_ingress_observed_travel", {}).get(
            branch,
            0.0,
        )
        if (
            tangent is None
            or tangent.length_squared() <= EPSILON
            or travel < PEBBLE_INGRESS_COHORT_MIN_TRAVEL
        ):
            continue
        observations.append((tangent.normalize(), travel))
    if not observations:
        return None
    reference, _ = max(observations, key=lambda item: item[1])
    consensus = pygame.Vector2()
    total_weight = 0.0
    for tangent, travel in observations:
        # Opposing samples are stale return trajectories, not ingress votes.
        if tangent.dot(reference) <= 0.0:
            continue
        weight = clamp(
            travel / max(PEBBLE_INGRESS_COHORT_MIN_TRAVEL, EPSILON),
            1.0,
            4.0,
        )
        consensus += tangent * weight
        total_weight += weight
    if total_weight <= EPSILON or consensus.length_squared() <= EPSILON:
        return reference.copy()
    return (consensus / total_weight).normalize()


def maybe_stage_pebble_at_return_crossing(
    robot: "Robot",
    old_position: pygame.Vector2,
) -> bool:
    """Freeze one returning NORMAL at its observed mouth-plane crossing.

    The plane is reconstructed from that robot's outbound observation. No
    predefined entrance coordinate, teleport, or position projection is used.
    The marker remains PENDING until the normal Branch-completion evidence is
    confirmed.
    """
    global pending_pebble_robot_ids
    branch_uid = active_branch_uid or branch_uid_for_fixture(active_branch)
    descriptor = branch_descriptors_by_uid.get(branch_uid)
    if (
        phase != SimulationPhase.FLOW_BACKTRACK
        or robot.role != "NORMAL"
        or descriptor is None
        or descriptor.visit_state != "ACTIVE"
        or active_branch in pending_pebble_robot_ids
    ):
        return False
    ingress = robot.local_ingress_tangents.get(active_branch)
    mouth_observation = robot.local_branch_ingress_points.get(active_branch)
    if (
        ingress is None
        or mouth_observation is None
        or ingress.length_squared() <= EPSILON
    ):
        return False
    ingress = ingress.normalize()
    return_direction = -ingress
    old_depth = (old_position - mouth_observation).dot(ingress)
    new_depth = (robot.position - mouth_observation).dot(ingress)
    return_speed = robot.observed_velocity.dot(return_direction)
    if not (
        old_depth > 0.0
        and new_depth <= 0.0
        and return_speed >= PEBBLE_CROSSING_MIN_RETURN_SPEED
    ):
        return False
    consensus_ingress = locally_consensed_ingress_direction(
        robot,
        active_branch,
    )
    if consensus_ingress is None:
        return False
    ingress = consensus_ingress
    return_direction = -ingress

    robot.local_return_mouth_crossings[active_branch] = robot.position.copy()
    robot.role = "PEBBLE"
    robot.pebble_anchor = robot.position.copy()
    robot.pebble_branch_uid = branch_uid
    robot.pebble_branch_key = active_branch
    robot.pebble_state = "PENDING_RETURN_CONFIRMATION"
    robot.pebble_ingress_direction_local = ingress.copy()
    robot.pebble_return_direction_local = return_direction.copy()
    robot.pebble_completion_epoch = 0
    robot.transfer_target = None
    robot.velocity.update(0.0, 0.0)
    robot.acceleration.update(0.0, 0.0)
    robot.filtered_acceleration.update(0.0, 0.0)
    pending_pebble_robot_ids[active_branch] = robot.robot_id
    print(
        f"[Pebble] branch={robot.pebble_branch_uid} robot={robot.robot_id} "
        "staged at observed return mouth crossing"
    )
    return True


def stage_pebble_from_returned_shepherd_line(
    robots,
    branch: str,
) -> Optional["Robot"]:
    """Use an actually returned Shepherd when completion follows immediately."""
    global pending_pebble_robot_ids
    branch_uid = branch_uid_for_fixture(branch)
    if branch_uid is None:
        return None
    if branch in pending_pebble_robot_ids:
        return next(
            (
                robot for robot in robots
                if robot.robot_id == pending_pebble_robot_ids[branch]
            ),
            None,
        )
    candidates = [
        robot for robot in robots
        if robot.role == "SHEPHERD"
        and robot.shepherd_branch == branch
        and branch in robot.local_ingress_tangents
        and branch in robot.local_branch_ingress_points
        and robot.position.distance_to(
            robot.local_branch_ingress_points[branch]
        ) <= PEBBLE_MOUTH_RADIUS
    ]
    if not candidates:
        return None
    pebble = min(
        candidates,
        key=lambda robot: (
            robot.position.distance_squared_to(
                robot.local_branch_ingress_points[branch]
            ),
            robot.robot_id,
        ),
    )
    ingress = locally_consensed_ingress_direction(pebble, branch)
    if ingress is None or ingress.length_squared() <= EPSILON:
        return None
    ingress = ingress.normalize()
    pebble.local_return_mouth_crossings[branch] = pebble.position.copy()
    pebble.role = "PEBBLE"
    pebble.pebble_anchor = pebble.position.copy()
    pebble.pebble_branch_uid = branch_uid
    pebble.pebble_branch_key = branch
    pebble.pebble_state = "PENDING_RETURN_CONFIRMATION"
    pebble.pebble_ingress_direction_local = ingress.copy()
    pebble.pebble_return_direction_local = -ingress
    pebble.pebble_completion_epoch = 0
    pebble.shepherd_anchor = None
    pebble.shepherd_origin = None
    pebble.frontier_local_lateral = None
    pebble.shepherd_branch = None
    pebble.transfer_target = None
    pebble.velocity.update(0.0, 0.0)
    pebble.acceleration.update(0.0, 0.0)
    pebble.filtered_acceleration.update(0.0, 0.0)
    pending_pebble_robot_ids[branch] = pebble.robot_id
    print(
        f"[Pebble] branch={pebble.pebble_branch_uid} "
        f"robot={pebble.robot_id} staged from returned Shepherd crossing"
    )
    return pebble


def create_branch_pebble(robots, branch_uid: str) -> Optional["Robot"]:
    """Promote a marker by UID; adapt to fixture only for local geometry."""
    global branch_completion_epoch, pending_pebble_robot_ids
    descriptor = branch_descriptors_by_uid.get(branch_uid)
    branch = branch_fixture_for_uid(branch_uid)
    if descriptor is None or branch is None:
        return None
    if branch_uid in observed_visited_branch_uids(robots):
        return next(
            pebble for pebble in get_pebbles(robots)
            if pebble.pebble_branch_uid == branch_uid
        )
    pending_id = pending_pebble_robot_ids.get(branch)
    pebble = next(
        (
            robot for robot in robots
            if robot.robot_id == pending_id
            and robot.role == "PEBBLE"
            and robot.pebble_branch_key == branch
        ),
        None,
    )
    if pebble is None:
        # Robust fallback for a cohort whose discrete trajectory did not cross
        # its recorded plane during the sampled backflow. The selected robot
        # is frozen where it physically stands; it is never moved to a map
        # entrance coordinate.
        candidates = [
            robot for robot in robots
            if robot.role == "NORMAL"
            and branch in robot.local_ingress_tangents
            and branch in robot.local_branch_ingress_points
            and robot.local_ingress_observed_travel.get(branch, 0.0)
            >= PEBBLE_INGRESS_COHORT_MIN_TRAVEL
        ]
        pebble = min(
            candidates,
            key=lambda robot: (
                robot.position.distance_squared_to(
                    robot.local_branch_ingress_points[branch]
                ),
                robot.robot_id,
            ),
            default=None,
        )
        if pebble is None:
            return None
        ingress = locally_consensed_ingress_direction(pebble, branch)
        if ingress is None:
            ingress = pebble.local_ingress_tangents[branch]
        if ingress.length_squared() <= EPSILON:
            return None
        ingress = ingress.normalize()
        pebble.role = "PEBBLE"
        pebble.pebble_anchor = pebble.position.copy()
        pebble.pebble_branch_uid = branch_uid
        pebble.pebble_branch_key = branch
        pebble.pebble_ingress_direction_local = ingress.copy()
        pebble.pebble_return_direction_local = -ingress
        pending_pebble_robot_ids[branch] = pebble.robot_id
        print(
            f"[Pebble] branch={pebble.pebble_branch_uid} "
            f"robot={pebble.robot_id} fallback frozen at current local pose"
        )
    ingress = pebble.pebble_ingress_direction_local
    return_direction = pebble.pebble_return_direction_local
    if (
        ingress is None
        or return_direction is None
        or ingress.length_squared() <= EPSILON
        or return_direction.length_squared() <= EPSILON
    ):
        return None
    ingress = ingress.normalize()
    return_direction = return_direction.normalize()
    descriptor.local_outgoing_direction = ingress.copy()
    descriptor.local_return_direction = return_direction.copy()
    if pebble.pebble_anchor is not None:
        descriptor.observed_mouth_position = pebble.pebble_anchor.copy()
    branch_completion_epoch += 1
    pebble.pebble_state = "VISITED"
    pebble.pebble_ingress_direction_local = ingress.copy()
    pebble.pebble_return_direction_local = return_direction.copy()
    pebble.pebble_completion_epoch = branch_completion_epoch
    pebble.known_visited_branches.add(branch)
    pebble.known_visited_branch_uids.add(branch_uid)
    state = pebble_flow_state_from_marker(pebble)
    if state is not None:
        pebble.known_pebble_flow_states[state.branch_uid] = state
    pebble.velocity.update(0.0, 0.0)
    pebble.acceleration.update(0.0, 0.0)
    print(
        f"[Pebble] uid={branch_uid} fixture={branch} robot={pebble.robot_id} "
        "created state=VISITED"
    )
    print(
        f"[Pebble] uid={branch_uid} fixture={branch} ingress_bearing="
        f"({ingress.x:.3f},{ingress.y:.3f}) return_bearing="
        f"({return_direction.x:.3f},{return_direction.y:.3f}) "
        f"epoch={branch_completion_epoch}"
    )
    return pebble


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


def structural_loss(branch: str, robots) -> int:
    reachable = reachable_nodes_without_branch(branch)
    visited_branches = observed_visited_branches(robots)
    loss = 0
    for candidate in BRANCHES:
        if candidate in visited_branches:
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
        phase == SimulationPhase.EXPLORE_BRANCH
        and region == "JUNCTION"
    ):
        # Quota closure controls only the source-side feed.  Robots that have
        # already entered the Branch retain the original pressure-coupled EDF
        # so their frontier/SPH behavior is unchanged.
        magnitude *= branch_fill_feed_scale
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
        "structural_loss": structural_loss(branch, robots),
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
    global last_proxy_candidates, last_decision_candidate_uids
    global last_flow_rollout_scores

    local_voters = [
        robot for robot in robots
        if robot.role == "NORMAL"
        and robot.connected_to_base
        and get_robot_region(robot.position) == "JUNCTION"
    ]
    knowledge_quorum = max(
        DISTRIBUTED_VOTE_MIN_ROBOTS,
        math.ceil(len(local_voters) * DISTRIBUTED_VOTE_QUORUM_RATIO),
    )
    quorum_visited_uids = {
        branch_uid for branch_uid in discovered_branch_uids()
        if sum(
            branch_uid in robot.known_visited_branch_uids
            for robot in local_voters
        ) >= knowledge_quorum
    }
    candidate_uids = [
        branch_uid
        for branch_uid in ordered_discovered_branch_uids()
        if branch_fixture_for_uid(branch_uid) in detected_branch_candidates
        and branch_uid not in quorum_visited_uids
    ]
    candidates = [
        branch_fixture_for_uid(branch_uid) for branch_uid in candidate_uids
    ]
    candidates = [branch for branch in candidates if branch is not None]
    uid_signature = tuple(candidate_uids)
    fixture_signature = tuple(candidates)
    if uid_signature != last_decision_candidate_uids:
        for branch_uid in sorted(quorum_visited_uids & discovered_branch_uids()):
            print(
                f"[VoteExclude] uid={branch_uid} "
                f"fixture={branch_fixture_for_uid(branch_uid) or '-'} "
                "reason=visited-quorum"
            )
    if (
        fixture_signature == last_proxy_candidates
        and uid_signature == last_decision_candidate_uids
        and all(branch in last_flow_rollout_scores for branch in candidates)
    ):
        return candidate_uids
    if not candidates:
        last_proxy_candidates = ()
        last_decision_candidate_uids = ()
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
    last_proxy_candidates = fixture_signature
    last_decision_candidate_uids = uid_signature
    last_flow_rollout_scores = scores
    return candidate_uids


def update_distributed_branch_consensus(
    robots,
    reference_density: float,
) -> Optional[str]:
    """Run a local NORMAL vote over inferred branches ranked by local costs."""
    global distributed_consensus_branch
    global last_distributed_vote_counts
    global last_distributed_voter_count, last_distributed_vote_quorum
    global last_distributed_vote_status
    voters = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and robot.connected_to_base
        and get_robot_region(robot.position) == "JUNCTION"
    ]
    last_distributed_voter_count = len(voters)
    last_distributed_vote_quorum = max(
        DISTRIBUTED_VOTE_MIN_ROBOTS,
        math.ceil(len(voters) * DISTRIBUTED_VOTE_QUORUM_RATIO),
    )
    if (
        not junction_inference_tracker.confirmed
        or simulation_time - junction_inference_tracker.confirmed_at
        < JUNCTION_DISCOVERY_SETTLE_TIME
    ):
        last_distributed_vote_counts = {}
        last_distributed_vote_status = "WAIT_DISCOVERY_SETTLE"
        return None
    candidate_uids = prepare_branch_candidate_scores(robots, reference_density)
    if not candidate_uids:
        last_distributed_vote_counts = {}
        last_distributed_vote_status = "NO_CANDIDATES"
        return None
    if len(voters) < DISTRIBUTED_VOTE_MIN_ROBOTS:
        last_distributed_vote_counts = {
            branch_uid: 0 for branch_uid in candidate_uids
        }
        last_distributed_vote_status = "INSUFFICIENT_VOTERS"
        return None

    locally_excluded_uids = set().union(
        *(robot.known_visited_branch_uids for robot in voters)
    )
    for branch_uid in sorted(locally_excluded_uids & set(candidate_uids)):
        print(
            f"[VoteExclude] uid={branch_uid} "
            f"fixture={branch_fixture_for_uid(branch_uid) or '-'} "
            "reason=local-pebble"
        )

    for robot in voters:
        local_candidates = [
            branch_uid
            for branch_uid in candidate_uids
            if branch_uid not in robot.known_visited_branch_uids
        ]
        if not local_candidates:
            robot.branch_vote = None
            continue
        preferred = min(
            local_candidates,
            key=lambda branch_uid: (
                last_flow_rollout_scores[
                    branch_fixture_for_uid(branch_uid)
                ]["cost"],
                candidate_uids.index(branch_uid),
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
            branch_uid: peer_votes.count(branch_uid)
            for branch_uid in local_candidates
        }
        counts[preferred] = counts.get(preferred, 0) + 1
        robot.branch_vote = max(
            counts,
            key=lambda branch_uid: (
                counts[branch_uid],
                -last_flow_rollout_scores[
                    branch_fixture_for_uid(branch_uid)
                ]["cost"],
            ),
        )
        robot.branch_vote_confidence = (
            counts[robot.branch_vote] / max(sum(counts.values()), 1)
        )

    vote_counts = {
        branch_uid: sum(robot.branch_vote == branch_uid for robot in voters)
        for branch_uid in candidate_uids
    }
    last_distributed_vote_counts = dict(vote_counts)
    print(
        "[BranchVote] "
        + ", ".join(
            f"{branch_identity_label(uid)}={vote_counts[uid]}"
            for uid in candidate_uids
        )
        + f" voters={len(voters)} quorum={last_distributed_vote_quorum}"
    )
    selected = max(
        candidate_uids,
        key=lambda branch_uid: (
            vote_counts[branch_uid],
            -last_flow_rollout_scores[
                branch_fixture_for_uid(branch_uid)
            ]["cost"],
        ),
    )
    quorum = last_distributed_vote_quorum
    if vote_counts[selected] < quorum:
        last_distributed_vote_status = "WAIT_QUORUM"
        return None

    last_distributed_vote_status = "CONSENSUS"
    distributed_consensus_branch = selected
    for robot in voters:
        robot.distributed_branch_decision = selected
    selected_fixture = branch_fixture_for_uid(selected)
    print(
        f"[Consensus] selected={branch_identity_label(selected)} "
        f"candidates={[branch_identity_label(uid) for uid in candidate_uids]} "
        f"votes={vote_counts[selected]}/{len(voters)}, "
        f"local-cost={last_flow_rollout_scores[selected_fixture]['cost']:.3f}"
    )
    return selected


def apply_consensus_branch_gates(
    open_branch: Optional[str],
    selected_uid: Optional[str] = None,
) -> None:
    """Adapt UID consensus to the fixture-keyed physical mouth gates."""
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
    record_distributed_consensus(
        selected_uid,
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
    record_distributed_consensus()
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
    record_distributed_consensus()
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
    record_distributed_consensus()
    for robot in robots:
        robot.transfer_target = (
            "BOTTOM"
            if get_robot_region(robot.position) == source
            and robot.role in {"NORMAL", "SHEPHERD"}
            else None
        )
    print(
        f"[Final Base Transfer] {source} -> BASE; "
        + ", ".join(
            f"{branch}={branch_gate_states[branch]}" for branch in BRANCHES
        )
    )


def close_all_branch_gates() -> None:
    branch_gate_states.clear()
    branch_gate_states.update(
        {branch: "CLOSED" for branch in BRANCHES}
    )
    record_distributed_consensus()
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
        record_distributed_consensus()
        return
    branch_gate_states[branch] = "CLOSED"
    record_distributed_consensus()
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


def next_unvisited_transfer_branch(source: str, robots) -> Optional[str]:
    """Return a fixture adapter selected from UID-based DFS memory."""
    source_uid = branch_uid_for_fixture(source)
    visited_uids = observed_visited_branch_uids(robots)
    candidate_uids = [
        branch_uid
        for branch_uid in ordered_discovered_branch_uids()
        if branch_uid != source_uid
        and branch_uid not in visited_uids
        and branch_fixture_for_uid(branch_uid) in detected_branch_candidates
    ]
    if not candidate_uids:
        return None
    selected_uid = min(
        candidate_uids,
        key=lambda branch_uid: (
            last_flow_rollout_scores.get(
                branch_fixture_for_uid(branch_uid), {}
            ).get("cost", float("inf")),
            candidate_uids.index(branch_uid),
        ),
    )
    return branch_fixture_for_uid(selected_uid)


def choose_next_branch(robots, reference_density: float):
    """Commit the branch selected by peer consensus over local rollout costs."""
    global active_branch, active_branch_uid, previous_branch_direction
    global branch_order_plan, branch_fixture_order_plan
    global last_proxy_partition, last_proxy_cell_centers
    global last_proxy_mass_stats, last_proxy_robot_assignment
    global last_proxy_candidates
    global last_flow_rollout_scores
    global detected_branch_candidates, collision_points
    global effective_branch_widths
    global selected_branch_entry_lambda, branch_entry_timer

    selected_uid = distributed_consensus_branch
    selected = branch_fixture_for_uid(selected_uid)
    if (
        selected_uid is None
        or selected is None
        or selected not in detected_branch_candidates
        or selected_uid in observed_visited_branch_uids(robots)
    ):
        return None

    if not branch_is_feasible(selected, robots):
        print(
            f"[DFS] warning: inferred branch "
            f"{branch_identity_label(selected_uid)} did not pass resource "
            "feasibility; preserving peer consensus"
        )

    set_branch_descriptor_state(selected_uid, "ACTIVE")
    for robot in robots:
        if robot.role == "NORMAL":
            robot.local_branch_states[selected] = "ACTIVE"
            robot.distributed_branch_decision = selected_uid
    apply_consensus_branch_gates(selected, selected_uid)
    record_distributed_consensus(selected_uid)
    active_branch = selected
    active_branch_uid = selected_uid
    branch_order_plan.append(selected_uid)
    branch_fixture_order_plan.append(selected)
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
        "selected": selected_uid,
        "fixture": selected,
        "cost": selected_score["cost"],
        "max_structural_loss": max(
            (structural_loss(branch, robots) for branch in detected_branch_candidates),
            default=0,
        ),
        "components": dict(selected_score["components"]),
        "candidate_scores": {
            branch_uid_for_fixture(branch) or branch: {
                "cost": data["cost"],
                "fixture": branch,
                "components": dict(data["components"]),
            }
            for branch, data in last_flow_rollout_scores.items()
        },
    })

    print(
        f"[DFS] local-cost selected={branch_identity_label(selected_uid)}, "
        f"cost={selected_score['cost']:.3f}, "
        f"entry_lambda={selected_branch_entry_lambda:.3f}"
    )
    return selected

def complete_active_branch(branch, robots, cohort_return_confirmed: bool):
    global previous_branch_direction, distributed_consensus_branch
    global active_branch_uid
    global last_proxy_candidates, last_decision_candidate_uids
    global last_flow_rollout_scores
    branch_uid = active_branch_uid or branch_uid_for_fixture(branch)
    descriptor = branch_descriptors_by_uid.get(branch_uid)
    if not (
        descriptor is not None
        and descriptor.visit_state == "ACTIVE"
        and branch_dead_end_confirmed.get(branch, False)
        and branch_backflow_started.get(branch, False)
        and cohort_return_confirmed
    ):
        print(
            f"[BranchComplete] rejected "
            f"branch={branch_identity_label(branch_uid)} "
            "missing ACTIVE/dead-end/backflow/return evidence"
        )
        return False
    pebble = create_branch_pebble(robots, branch_uid)
    if pebble is None:
        print(
            "[BranchComplete] waiting for local Pebble candidate "
            f"branch={branch_identity_label(branch_uid)}"
        )
        return False
    set_branch_descriptor_state(branch_uid, "VISITED")
    distributed_consensus_branch = None
    active_branch_uid = None
    last_proxy_candidates = ()
    last_decision_candidate_uids = ()
    last_flow_rollout_scores = {}
    collision_points = deque(maxlen=CONTACT_POINT_MAX_COUNT)
    effective_branch_widths = {branch: 0.0 for branch in BRANCHES}
    for robot in robots:
        robot.branch_vote = None
        robot.branch_vote_confidence = 0.0
        robot.distributed_branch_decision = None
    record_distributed_consensus(clear_selection=True)
    previous_branch_direction = get_backtrack_direction(branch)
    metrics.branch_events.append({"branch": branch, "completed_at": simulation_time})
    visited_count = len(observed_visited_branch_uids(robots))
    print(
        f"[BranchComplete] uid={pebble.pebble_branch_uid} fixture={branch}"
    )
    print(
        f"[JunctionComplete] visited={visited_count}/"
        f"{len(discovered_branch_uids()) or len(BRANCHES)}"
    )
    return True


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


def begin_final_return(robots):
    global phase, relay_slots, relay_motion_scale
    global return_trunk_release_pending, return_trunk_retract_timer, return_trunk_last_released_id, return_trunk_force_timer
    global return_done_dwell, return_entry_stall_timer
    global return_last_bottom_count
    release_transient_roles_for_final_return(robots)
    relay_slots = []
    relay_motion_scale = 1.0
    close_all_branch_gates()
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
                    origin_is_new = key not in self.cohort_origins
                    chosen_origin = robot.position.copy()
                    if (
                        origin_is_new
                        and LOCAL_MOUTH_CROSSING_DIAGNOSTICS_ENABLED
                    ):
                        local_estimate = (
                            local_mouth_crossing_diagnostics.sampler.estimate(
                                robot.robot_id
                            )
                        )
                        if local_estimate is not None:
                            local_mouth_crossing_diagnostics.record_candidate(
                                branch=branch,
                                robot_id=robot.robot_id,
                                heading_xy=(robot.position.x, robot.position.y),
                                heading_time=simulation_time,
                                local=local_estimate,
                            )
                            if LOCAL_MOUTH_SAMPLE_MODE == "local_crossing":
                                chosen_origin = pygame.Vector2(
                                    local_estimate.x,
                                    local_estimate.y,
                                )
                    self.cohort_origins.setdefault(
                        key,
                        chosen_origin,
                    )
                    if (
                        origin_is_new
                        and MOUTH_PCA_SAMPLE_DIAGNOSTICS_ENABLED
                    ):
                        mouth_pca_sample_diagnostics.record_crossing_origin({
                            "branch": branch,
                            "frame": globals().get("headless_frame_count", 0),
                            "crossing_timestamp": simulation_time,
                            "robot_id": robot.robot_id,
                            "sample_world_x": robot.position.x,
                            "sample_world_y": robot.position.y,
                            "robot_role": robot.role,
                            "robot_speed": robot.velocity.length(),
                            "observed_speed": velocity.length(),
                            "velocity_direction_deg": (
                                _yaw_deg(velocity)
                                if velocity.length_squared() > EPSILON
                                else ""
                            ),
                            "neighbor_count": len(robot.comm_neighbors),
                            "crossing_region": get_robot_region(robot.position),
                            "crossing_depth": branch_depth_from_junction(
                                robot.position,
                                branch,
                            ),
                        })
                    if (
                        origin_is_new
                        and MOUTH_CROSSING_ORIGIN_DIAGNOSTICS_ENABLED
                    ):
                        mouth_crossing_origin_diagnostics.record_heading_origin({
                            "branch": branch,
                            "robot_id": robot.robot_id,
                            "frame": globals().get("headless_frame_count", 0),
                            "heading_sample_time": simulation_time,
                            "heading_world_x": robot.position.x,
                            "heading_world_y": robot.position.y,
                            "robot_role": robot.role,
                            "observed_speed": velocity.length(),
                        })
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
                descriptor = register_discovered_branch(
                    robots,
                    branch,
                    self.cohort_member_ids[branch],
                    self.cohort_origins,
                )
                print(
                    f"[Branch Evidence] {descriptor.uid} ({branch}) OPEN "
                    "only after physical "
                    f"crossing: unique={self.cohort_counts[branch]}, "
                    f"depth={self.cohort_depth[branch]:.1f}, "
                    f"travel={self.cohort_travel[branch]:.1f}, "
                    f"dwell={self.cohort_dwell[branch]:.2f}"
                )

        for branch in self.valid_branches:
            branch_uid = branch_uid_for_fixture(branch)
            descriptor = branch_descriptors_by_uid.get(branch_uid)
            if descriptor is None:
                continue
            descriptor.cohort_member_ids.update(
                self.cohort_member_ids[branch]
            )
            for robot_id in self.cohort_member_ids[branch]:
                origin = self.cohort_origins.get((robot_id, branch))
                if origin is not None:
                    descriptor.observed_mouth_samples.setdefault(
                        robot_id,
                        origin.copy(),
                    )
            update_branch_direction_stability(descriptor, robots, dt)

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

        directions_mature = bool(self.valid_branches) and all(
            (
                descriptor := branch_descriptors_by_uid.get(
                    branch_uid_for_fixture(branch)
                )
            ) is not None
            and descriptor.direction_is_mature
            for branch in self.valid_branches
        )
        junction_signature = (
            self.expansion_dwell >= JUNCTION_EXPANSION_DWELL_TIME
            and len(self.valid_branches) >= JUNCTION_MIN_VALID_COHORTS
            and self.forward_probe_status != "UNRESOLVED"
            and directions_mature
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
                "branch_uids": ordered_discovered_branch_uids(),
                "expansion_ratio": self.expansion_ratio,
                "lateral_variance": self.lateral_variance,
                "sector_distribution": dict(self.sector_distribution),
                "cohort_counts": dict(self.cohort_counts),
                "cohort_travel": dict(self.cohort_travel),
                "cohort_depth": dict(self.cohort_depth),
                "branch_directions": {
                    uid: (
                        branch_descriptors_by_uid[uid]
                        .local_outgoing_direction.xy
                    )
                    for uid in ordered_discovered_branch_uids()
                },
                "forward_probe_status": self.forward_probe_status,
            })
            print(
                "[Junction Inference] sequential expansion + cohorts: "
                "branches="
                f"{[branch_identity_label(uid) for uid in ordered_discovered_branch_uids()]}, "
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
    shepherd_direct_contact_count: int = 0
    shepherd_direct_contact_ratio: float = 0.0
    shepherd_contact_span: float = 0.0
    shepherd_contact_span_ratio: float = 0.0
    shepherd_mean_forward_speed: float = 0.0
    frontier_forward_progress: float = 0.0
    frontier_progress_rate: float = float("inf")
    branch_robot_count: int = 0
    required_branch_robot_count: int = 0
    blocking_reason: str = "NO_FRONTIER_LINE"
    confirmation_mode: str = "WAIT"
    frontier_progress_history: deque = field(
        default_factory=lambda: deque(maxlen=32)
    )
    confirmed_depth: float = 0.0
    handoff_depth: float = 0.0
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
        self.shepherd_direct_contact_count = 0
        self.shepherd_direct_contact_ratio = 0.0
        self.shepherd_contact_span = 0.0
        self.shepherd_contact_span_ratio = 0.0
        self.shepherd_mean_forward_speed = 0.0
        self.frontier_forward_progress = 0.0
        self.frontier_progress_rate = float("inf")
        self.branch_robot_count = 0
        self.required_branch_robot_count = 0
        self.blocking_reason = "NO_FRONTIER_LINE"
        self.confirmation_mode = "WAIT"
        self.frontier_progress_history.clear()
        self.confirmed_depth = 0.0
        self.handoff_depth = 0.0
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
        self.branch_robot_count = len(branch_robots)
        descriptor = branch_motion_descriptor(branch)
        if descriptor is None or not descriptor.physical_width_confident:
            self.dwell = 0.0
            self.blocking_reason = "LOCAL_PHYSICAL_FRAME_UNAVAILABLE"
            return False
        if not branch_robots:
            self.dwell = 0.0
            self.blocking_reason = "INSUFFICIENT_BRANCH_ROBOTS"
            return False

        maximum_depth = max(
            observed_branch_axial_depth(robot.position, descriptor)
            for robot in branch_robots
        )
        frontier = [
            robot
            for robot in branch_robots
            if observed_branch_axial_depth(robot.position, descriptor)
            >= maximum_depth - DEAD_END_FRONTIER_DEPTH
        ]
        self.frontier_count = len(frontier)
        if len(frontier) < DEAD_END_MIN_FRONTIER_ROBOTS:
            self.dwell = 0.0
            self.blocking_reason = "INSUFFICIENT_FRONTIER_ROBOTS"
            return False

        direction, lateral = descriptor_local_basis(descriptor)
        leader = max(
            frontier,
            key=lambda robot: observed_branch_axial_depth(
                robot.position,
                descriptor,
            ),
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
        if frontier_shepherds:
            self.frontier_forward_progress = linear_quantile(
                [
                    observed_branch_axial_depth(robot.position, descriptor)
                    for robot in frontier_shepherds
                ],
                0.50,
            )
            self.frontier_progress_history.append((
                simulation_time,
                self.frontier_forward_progress,
            ))
            while (
                len(self.frontier_progress_history) >= 2
                and simulation_time
                - self.frontier_progress_history[0][0]
                > DEAD_END_FRONTIER_PROGRESS_WINDOW
            ):
                self.frontier_progress_history.popleft()
            history_span = (
                simulation_time - self.frontier_progress_history[0][0]
                if self.frontier_progress_history
                else 0.0
            )
            self.frontier_progress_rate = (
                max(
                    0.0,
                    self.frontier_forward_progress
                    - self.frontier_progress_history[0][1],
                )
                / max(history_span, EPSILON)
                if history_span >= DEAD_END_FAST_CONFIRM_DWELL
                else float("inf")
            )
        else:
            self.frontier_progress_history.clear()
            self.frontier_forward_progress = 0.0
            self.frontier_progress_rate = float("inf")
        directly_contacting_shepherds = [
            robot
            for robot in frontier_shepherds
            if simulation_time - robot.last_forward_obstacle_contact_time
            <= DEAD_END_FORWARD_BUMPER_MEMORY
        ]
        self.shepherd_direct_contact_count = len(
            directly_contacting_shepherds
        )
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
            branch_local_coordinates(robot.position, descriptor)[1]
            for robot in directly_contacting_shepherds
        ]
        self.shepherd_contact_span = (
            max(contact_lateral_coordinates) - min(contact_lateral_coordinates)
            if len(contact_lateral_coordinates) >= 2
            else 0.0
        )
        self.shepherd_contact_span_ratio = clamp(
            self.shepherd_contact_span
            / max(local_physical_usable_half_width(descriptor) * 2.0, EPSILON),
            0.0,
            1.0,
        )
        required_branch_robots = max(
            DEAD_END_FAST_MIN_BRANCH_ROBOTS,
            len(frontier_shepherds) * 2,
        )
        self.required_branch_robot_count = required_branch_robots
        contact_stall_conditions = (
            len(frontier_shepherds) >= JUNCTION_GUARD_MIN_COUNT
            and self.shepherd_direct_contact_ratio
            >= DEAD_END_SHEPHERD_DIRECT_CONTACT_RATIO
            and self.shepherd_contact_span_ratio
            >= DEAD_END_SHEPHERD_CONTACT_SPAN_RATIO
            and self.frontier_progress_rate
            <= DEAD_END_FRONTIER_PROGRESS_RATE_THRESHOLD
            and self.branch_robot_count >= required_branch_robots
            and self.lateral_escape_ratio
            <= DEAD_END_LATERAL_ESCAPE_RATIO
        )
        if len(frontier_shepherds) < JUNCTION_GUARD_MIN_COUNT:
            self.blocking_reason = "NO_FRONTIER_SHEPHERD_LINE"
        elif (
            self.shepherd_direct_contact_ratio
            < DEAD_END_SHEPHERD_DIRECT_CONTACT_RATIO
        ):
            self.blocking_reason = "FRONTIER_CONTACT_RATIO"
        elif (
            self.shepherd_contact_span_ratio
            < DEAD_END_SHEPHERD_CONTACT_SPAN_RATIO
        ):
            self.blocking_reason = "FRONTIER_CONTACT_SPAN"
        elif (
            self.frontier_progress_rate
            > DEAD_END_FRONTIER_PROGRESS_RATE_THRESHOLD
        ):
            self.blocking_reason = "FRONTIER_STILL_PROGRESSING"
        elif self.branch_robot_count < required_branch_robots:
            self.blocking_reason = "INSUFFICIENT_BRANCH_ROBOTS"
        elif self.lateral_escape_ratio > DEAD_END_LATERAL_ESCAPE_RATIO:
            self.blocking_reason = "LATERAL_ESCAPE_PRESENT"
        else:
            self.blocking_reason = "READY_CONTACT_STALL"
        conditions = contact_stall_conditions
        required_dwell = DEAD_END_FAST_CONFIRM_DWELL
        self.confirmation_mode = (
            "CONTACT_STALL"
            if contact_stall_conditions
            else "WAIT"
        )
        self.dwell = (
            self.dwell + dt
            if conditions
            else max(0.0, self.dwell - dt)
        )
        newly_confirmed = (
            not self.confirmed
            and self.dwell >= required_dwell
        )
        if newly_confirmed:
            self.confirmed = True
            # Confirm the wall at the median local axial contact depth. Keep
            # the nearby common walkable row separate: it is a placement
            # target, not a replacement for the measured Dead-end depth.
            self.confirmed_depth = max(
                0.0,
                linear_quantile(
                    [
                        observed_branch_axial_depth(
                            robot.position,
                            descriptor,
                        )
                        for robot in directly_contacting_shepherds
                    ],
                    0.50,
                ),
            )
            resolved_handoff_depth = resolve_local_frontier_handoff_depth(
                frontier_shepherds,
                descriptor,
                self.confirmed_depth,
            )
            if resolved_handoff_depth is None:
                self.confirmed = False
                self.blocking_reason = "NO_COMMON_LOCAL_HANDOFF_ROW"
                return False
            self.handoff_depth = resolved_handoff_depth
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
                "frontier_forward_progress": self.frontier_forward_progress,
                "frontier_progress_rate": self.frontier_progress_rate,
                "branch_robot_count": self.branch_robot_count,
                "confirmation_mode": self.confirmation_mode,
                "observed_depth": self.confirmed_depth,
                "handoff_depth": self.handoff_depth,
            })
            print(
                f"[Dead-end Inference] branch={branch}, "
                f"contact={self.mean_contact:.2f}, "
                f"forward={self.mean_forward_speed:.2f}, "
                f"rho={self.mean_density_ratio:.2f}, "
                f"bumper={self.shepherd_direct_contact_ratio:.2f}, "
                f"span={self.shepherd_contact_span_ratio:.2f}, "
                f"shepherd_v={self.shepherd_mean_forward_speed:.2f}, "
                f"frontier-progress={self.frontier_forward_progress:.1f}, "
                f"frontier-rate={self.frontier_progress_rate:.2f}, "
                f"branch-robots={self.branch_robot_count}, "
                f"fill={branch_fill_current_count}/"
                f"{branch_fill_target_count}, "
                f"feed={branch_fill_feed_state}, "
                f"dwell={self.dwell:.2f}, "
                f"mode={self.confirmation_mode}, "
                f"observed_depth={self.confirmed_depth:.1f}"
                f", handoff_depth={self.handoff_depth:.1f}"
            )
        return self.confirmed


junction_inference_tracker = SequentialJunctionInferenceTracker()
dead_end_inference_tracker = DeadEndInferenceTracker()


def log_frontier_explore_diagnostics(robots, branch: str) -> None:
    """Explain local-row readiness and the existing dead-end transition gate."""
    global frontier_line_last_diagnostic_time
    if simulation_time - frontier_line_last_diagnostic_time < 1.0:
        return
    descriptor = branch_motion_descriptor(branch)
    frontiers = get_frontier_shepherds(robots, branch)
    if descriptor is None or not frontiers:
        return
    frontier_line_last_diagnostic_time = simulation_time
    tangent, normal = descriptor_local_basis(descriptor)
    contacting = [
        robot for robot in frontiers
        if simulation_time - robot.last_forward_obstacle_contact_time
        <= DEAD_END_FORWARD_BUMPER_MEMORY
    ]
    contact_laterals = [
        branch_local_coordinates(robot.position, descriptor)[1]
        for robot in contacting
    ]
    contact_span = (
        max(contact_laterals) - min(contact_laterals)
        if len(contact_laterals) >= 2
        else 0.0
    )
    print(
        f"[FrontierMotion] branch={branch} uid={descriptor.uid} "
        f"robots={len(frontiers)} "
        f"physical-width={descriptor.observed_physical_width:.1f} "
        f"usable-width={local_physical_usable_half_width(descriptor) * 2.0:.1f} "
        f"axial={frontier_line_depth:.1f} "
        f"settled={frontier_line_target_settled_ratio:.2f} "
        f"row-ready={frontier_line_row_ready} "
        f"span={frontier_line_current_span:.1f}/"
        f"{frontier_line_target_span:.1f} "
        f"coverage={frontier_line_physical_coverage_ratio:.2f} "
        f"edge-gaps={frontier_line_left_edge_gap:.1f}/"
        f"{frontier_line_right_edge_gap:.1f} "
        f"continuous={frontier_line_continuous} "
        f"center={frontier_line_lateral_center:.2f} "
        f"motion-t=({tangent.x:.3f},{tangent.y:.3f}) "
        f"motion-n=({normal.x:.3f},{normal.y:.3f}) "
        f"forward-contact={len(contacting)}/{len(frontiers)} "
        f"contact-ratio="
        f"{len(contacting) / max(len(frontiers), 1):.2f} "
        f"contact-span={contact_span:.1f} "
        f"dead-end-dwell={dead_end_inference_tracker.dwell:.2f} "
        f"blocked={dead_end_inference_tracker.blocking_reason}"
    )
    transition_line_reached = (
        dead_end_inference_tracker.confirmed
        and all(
            observed_branch_axial_depth(robot.position, descriptor)
            >= dead_end_inference_tracker.confirmed_depth
            - JUNCTION_GUARD_POSITION_TOLERANCE
            for robot in frontiers
        )
    )
    if not dead_end_inference_tracker.confirmed:
        transition_blocker = dead_end_inference_tracker.blocking_reason
    elif not transition_line_reached:
        transition_blocker = "FRONTIER_BEHIND_CONFIRMED_LOCAL_DEPTH"
    elif dead_end_inference_tracker.blocking_reason.startswith(
        "LOCAL_HANDOFF"
    ):
        transition_blocker = dead_end_inference_tracker.blocking_reason
    else:
        transition_blocker = "READY_FOR_LOCAL_PROMOTE"
    log_dead_end_transition_diagnostics(
        robots,
        branch,
        transition_line_reached,
        transition_blocker,
    )


def log_dead_end_transition_diagnostics(
    robots,
    branch: str,
    physical_line_reached: bool,
    blocking_reason: str,
    force: bool = False,
) -> None:
    """Explain the local contact-depth to physical-line handoff gate."""
    global frontier_dead_end_transition_last_log_time
    if (
        not force
        and simulation_time - frontier_dead_end_transition_last_log_time
        < 1.0
    ):
        return
    descriptor = branch_motion_descriptor(branch)
    frontiers = get_frontier_shepherds(robots, branch)
    if descriptor is None:
        return
    if not frontiers:
        frontiers = [
            robot for robot in get_shepherds(robots)
            if robot.shepherd_branch == branch
        ]
    frontier_dead_end_transition_last_log_time = simulation_time
    depths = sorted(
        observed_branch_axial_depth(robot.position, descriptor)
        for robot in frontiers
    )
    minimum = depths[0] if depths else 0.0
    median = linear_quantile(depths, 0.50) if depths else 0.0
    maximum = depths[-1] if depths else 0.0
    print(
        f"[DeadEndTransition] branch={branch} uid={descriptor.uid} "
        f"frontier-shepherds={len(frontiers)} "
        f"direct-contact="
        f"{dead_end_inference_tracker.shepherd_direct_contact_count}/"
        f"{len(frontiers)} "
        f"direct-contact-ratio="
        f"{dead_end_inference_tracker.shepherd_direct_contact_ratio:.2f} "
        f"contact-span={dead_end_inference_tracker.shepherd_contact_span:.1f} "
        f"contact-span-ratio="
        f"{dead_end_inference_tracker.shepherd_contact_span_ratio:.2f} "
        f"frontier-local-axial={minimum:.1f}/{median:.1f}/{maximum:.1f} "
        f"frontier-progress-rate="
        f"{dead_end_inference_tracker.frontier_progress_rate:.2f} "
        f"branch-robots={dead_end_inference_tracker.branch_robot_count}/"
        f"{dead_end_inference_tracker.required_branch_robot_count} "
        f"lateral-escape="
        f"{dead_end_inference_tracker.lateral_escape_ratio:.2f} "
        f"dwell={dead_end_inference_tracker.dwell:.2f} "
        f"dead-end-confirmed={dead_end_inference_tracker.confirmed} "
        f"confirmed-local-axial="
        f"{dead_end_inference_tracker.confirmed_depth:.1f} "
        f"handoff-local-axial="
        f"{dead_end_inference_tracker.handoff_depth:.1f} "
        f"physical-line-reached={physical_line_reached} "
        f"blocking-reason={blocking_reason}"
    )


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
            robot.frontier_local_lateral = None
            robot.shepherd_branch = None
            robot.junction_guard_anchor = None
            robot.junction_guard_branch = None
            robot.junction_guard_branch_uid = None
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
            robot.frontier_local_lateral = None
            robot.shepherd_branch = None
            robot.junction_guard_anchor = None
            robot.junction_guard_branch = None
            robot.junction_guard_branch_uid = None
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
    stage_pebble_from_returned_shepherd_line(robots, active_branch)
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
        robot.frontier_local_lateral = None
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
            if robot_i is robot_j or robot_j.role == "PEBBLE":
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
    global branch_fill_feed_scale, branch_fill_feed_state
    global branch_fill_feed_last_logged_state

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
        fill_ratio = (
            branch_fill_current_count
            / max(branch_fill_target_count, 1)
        )
        raw_feed_scale = 1.0 - smoothstep01(
            (
                fill_ratio
                - BRANCH_FILL_FEED_THROTTLE_START_RATIO
            )
            / max(
                BRANCH_FILL_FEED_CLOSED_RATIO
                - BRANCH_FILL_FEED_THROTTLE_START_RATIO,
                EPSILON,
            )
        )
        branch_fill_feed_scale = (
            branch_fill_feed_scale
            * (1.0 - BRANCH_FILL_FEED_FILTER_ALPHA)
            + raw_feed_scale * BRANCH_FILL_FEED_FILTER_ALPHA
        )
        if fill_ratio >= BRANCH_FILL_FEED_CLOSED_RATIO:
            branch_fill_feed_state = "CLOSED"
        elif fill_ratio >= BRANCH_FILL_FEED_THROTTLE_START_RATIO:
            branch_fill_feed_state = "THROTTLED"
        else:
            branch_fill_feed_state = "OPEN"
        if branch_fill_feed_state != branch_fill_feed_last_logged_state:
            print(
                f"[Branch Feed] branch={active_branch}, "
                f"fill={branch_fill_current_count}/{branch_fill_target_count}, "
                f"state={branch_fill_feed_state}, "
                f"scale={branch_fill_feed_scale:.2f}"
            )
            branch_fill_feed_last_logged_state = branch_fill_feed_state
    else:
        branch_fill_target_count = 0
        branch_fill_current_count = 0
        branch_fill_deficit_control = 0.0
        branch_fill_feed_scale = 1.0
        branch_fill_feed_state = "OPEN"
        branch_fill_feed_last_logged_state = "OPEN"

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


def pebble_flow_local_coordinates(
    robot: "Robot",
    state: PebbleFlowState,
) -> tuple[float, float, float]:
    """Return observed ingress depth, lateral offset, and marker distance."""
    mouth = pygame.Vector2(state.mouth_position)
    ingress = pygame.Vector2(state.ingress_direction_local)
    if ingress.length_squared() <= EPSILON:
        return 0.0, float("inf"), float("inf")
    ingress = ingress.normalize()
    lateral = pygame.Vector2(-ingress.y, ingress.x)
    offset = robot.position - mouth
    return (
        offset.dot(ingress),
        abs(offset.dot(lateral)),
        offset.length(),
    )


def robot_is_inside_visited_flow(
    robot: "Robot",
    state: PebbleFlowState,
) -> bool:
    """Observer/control test using only a Pebble's locally measured frame."""
    ingress_depth, lateral_offset, _ = pebble_flow_local_coordinates(
        robot,
        state,
    )
    return (
        ingress_depth > 0.0
        and lateral_offset
        <= corridor_width + PEBBLE_BRANCH_LATERAL_MARGIN
    )


def compute_pebble_flow_guidance(
    robot: "Robot",
    decision_force: pygame.Vector2,
) -> pygame.Vector2:
    """Condition only route/EDF propulsion using locally relayed VISITED facts.

    Near a completed mouth, positive propulsion along the observed ingress
    tangent is smoothly attenuated and a weak return-bearing bias is added.
    Once the robot has physically crossed the observed mouth plane, recovery
    ramps up. Physical SPH, Kelvin-Voigt, cohesion, collision, and connectivity
    forces never enter this function.
    """
    robot.last_pebble_guidance_force = 0.0
    robot.last_pebble_guidance_weight = 0.0
    robot.last_pebble_guidance_mode = "NONE"
    robot.last_pebble_guidance_branch_uid = None
    if (
        robot.role != "NORMAL"
        or not robot.known_pebble_flow_states
        or phase in {SimulationPhase.MOVE_TO_JUNCTION, SimulationPhase.DONE}
    ):
        return decision_force

    active_uid = (
        robot.local_branch_uid_by_key.get(active_branch)
        if branch_states.get(active_branch) == "ACTIVE"
        else None
    )
    guided_force = decision_force.copy()
    return_bias = pygame.Vector2()
    strongest_weight = 0.0
    strongest_mode = "NONE"
    strongest_uid: Optional[str] = None

    for branch_uid, state in sorted(
        robot.known_pebble_flow_states.items()
    ):
        # A selected ACTIVE child is never conditioned by a stale/duplicate
        # fact carrying that child's own locally assigned UID.
        if active_uid is not None and branch_uid == active_uid:
            continue
        ingress = pygame.Vector2(state.ingress_direction_local)
        return_direction = pygame.Vector2(state.return_direction_local)
        if (
            ingress.length_squared() <= EPSILON
            or return_direction.length_squared() <= EPSILON
        ):
            continue
        ingress = ingress.normalize()
        return_direction = return_direction.normalize()
        ingress_depth, lateral_offset, marker_distance = (
            pebble_flow_local_coordinates(robot, state)
        )
        lateral_weight = 1.0 - smoothstep01(
            (
                lateral_offset - corridor_width
            ) / max(PEBBLE_BRANCH_LATERAL_MARGIN, EPSILON)
        )
        if lateral_weight <= 0.0:
            continue

        inside_weight = (
            smoothstep01(
                ingress_depth
                / max(PEBBLE_RECOVERY_FULL_DEPTH, EPSILON)
            )
            * lateral_weight
            if ingress_depth > 0.0
            else 0.0
        )
        proximity_weight = 1.0 - smoothstep01(
            (
                marker_distance - PEBBLE_MOUTH_RADIUS
            )
            / max(
                PEBBLE_GUIDANCE_RADIUS - PEBBLE_MOUTH_RADIUS,
                EPSILON,
            )
        )
        proximity_weight = clamp(proximity_weight, 0.0, 1.0)
        mouth_weight = 1.0 - smoothstep01(
            marker_distance / max(PEBBLE_MOUTH_RADIUS, EPSILON)
        )
        preventive_weight = (
            proximity_weight
            * (
                PEBBLE_CAUTION_GAIN
                + PEBBLE_MOUTH_SUPPRESSION_GAIN * mouth_weight
            )
            * lateral_weight
        )
        attenuation = clamp(
            max(preventive_weight, 0.95 * inside_weight),
            0.0,
            0.95,
        )
        positive_ingress = max(0.0, guided_force.dot(ingress))
        if positive_ingress > EPSILON and attenuation > EPSILON:
            guided_force -= ingress * positive_ingress * attenuation

        if inside_weight > EPSILON:
            mode = "RECOVERY"
            weight = max(inside_weight, preventive_weight)
            return_bias += return_direction * (
                PEBBLE_RECOVERY_GAIN * inside_weight
                + PEBBLE_PREVENTIVE_RETURN_FORCE * preventive_weight
            )
        elif preventive_weight > EPSILON:
            mode = "JUNCTION_SUPPRESS"
            weight = preventive_weight
            # This is a bearing command, not radial repulsion: every receiver
            # uses the same locally observed Branch-to-Junction direction.
            return_bias += (
                return_direction
                * PEBBLE_PREVENTIVE_RETURN_FORCE
                * preventive_weight
            )
        else:
            continue
        if weight > strongest_weight:
            strongest_weight = weight
            strongest_mode = mode
            strongest_uid = branch_uid

    return_bias = limit_vector(return_bias, PEBBLE_GUIDANCE_FORCE_LIMIT)
    guided_force += return_bias
    adjustment = guided_force - decision_force
    robot.last_pebble_guidance_force = adjustment.length()
    robot.last_pebble_guidance_weight = strongest_weight
    robot.last_pebble_guidance_mode = strongest_mode
    robot.last_pebble_guidance_branch_uid = strongest_uid
    return guided_force


def compute_route_force(robot):
    region = get_robot_region(robot.position)
    junction_target = pygame.Vector2(center_x, center_y)
    force = pygame.Vector2()
    if robot.role in {"PEBBLE", "RELAY", "TRUNK_RELAY"}:
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
            ) * branch_fill_feed_scale
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
                * branch_fill_feed_scale
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
                    * branch_fill_feed_scale
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
                    * branch_fill_feed_scale
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
            "PEBBLE",
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
            robot_i.last_pebble_guidance_force = 0.0
            robot_i.last_pebble_guidance_weight = 0.0
            robot_i.last_pebble_guidance_mode = "NONE"
            robot_i.last_pebble_guidance_branch_uid = None
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
            robot_i.last_pebble_guidance_force = 0.0
            robot_i.last_pebble_guidance_weight = 0.0
            robot_i.last_pebble_guidance_mode = "NONE"
            robot_i.last_pebble_guidance_branch_uid = None
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
            if robot_i is robot_j or robot_j.role == "PEBBLE":
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
                and robot_j.junction_guard_branch_uid
                in branch_descriptors_by_uid
                and distance < PHYSICAL_GUARD_INFLUENCE_RADIUS
            ):
                ratio = (
                    1.0
                    - distance / PHYSICAL_GUARD_INFLUENCE_RADIUS
                )
                guard_descriptor = branch_descriptors_by_uid[
                    robot_j.junction_guard_branch_uid
                ]
                branch_direction, lateral = descriptor_local_basis(
                    guard_descriptor
                )
                inward = -branch_direction
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
        decision_guidance_force = compute_pebble_flow_guidance(
            robot_i,
            route_force + edf_force,
        )
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
            + decision_guidance_force
            + connectivity_force
            + shepherd_curtain_force
            + pre_shepherd_curtain_force
            + initial_junction_wall_force
            + base_piston_force
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
    global current_visited_branch_leakage
    global current_pebble_guidance_active_normals
    global pebble_flow_last_log_time
    if base_station is not None:
        disconnected = sum(not robot.connected_to_base for robot in robots)
        metrics.disconnected_robot_seconds += disconnected * dt

    visited_states: dict[str, PebbleFlowState] = {}
    for pebble in get_pebbles(robots):
        state = pebble_flow_state_from_marker(pebble)
        if state is not None:
            visited_states[state.branch_uid] = state
    exploration_leak_phase = phase in {
        SimulationPhase.JUNCTION_SWITCH,
        SimulationPhase.FORM_JUNCTION_GUARDS,
        SimulationPhase.EXPLORE_BRANCH,
        SimulationPhase.FORM_SHEPHERD_BOUNDARY,
        SimulationPhase.FILL_BEHIND_SHEPHERD,
    }

    leakage = {branch: 0 for branch in BRANCHES}
    current_inside_by_robot: dict[int, set[str]] = {}
    for robot in robots:
        current_inside: set[str] = set()
        if robot.role == "NORMAL":
            for branch_uid, state in visited_states.items():
                if robot_is_inside_visited_flow(robot, state):
                    current_inside.add(branch_uid)
                    if state.branch_key in leakage:
                        leakage[state.branch_key] += 1
        current_inside_by_robot[robot.robot_id] = current_inside

    initialized_before = set(metrics.initialized_visited_uids)
    for robot in robots:
        current_inside = current_inside_by_robot[robot.robot_id]
        if (
            robot.last_pebble_guidance_mode == "RECOVERY"
            and robot.last_pebble_guidance_branch_uid in current_inside
        ):
            robot.metric_pebble_recovery_seen_uids.add(
                robot.last_pebble_guidance_branch_uid
            )
        entered = (
            current_inside - robot.metric_inside_visited_uids
        ) & initialized_before
        for branch_uid in entered:
            state = visited_states.get(branch_uid)
            if state is not None and state.branch_key in BRANCHES:
                branch = state.branch_key
                metrics.visited_branch_reentry_robot_ids[branch].add(
                    robot.robot_id
                )
                metrics.visited_branch_reentry_counts[branch] = len(
                    metrics.visited_branch_reentry_robot_ids[branch]
                )
                if exploration_leak_phase:
                    metrics.exploration_leak_reentry_robot_ids[branch].add(
                        robot.robot_id
                    )
                    metrics.exploration_leak_reentry_counts[branch] = len(
                        metrics.exploration_leak_reentry_robot_ids[branch]
                    )
        exited = robot.metric_inside_visited_uids - current_inside
        for branch_uid in exited:
            state = visited_states.get(branch_uid)
            if state is None:
                continue
            ingress_depth, _, _ = pebble_flow_local_coordinates(robot, state)
            if (
                ingress_depth <= 0.0
                and branch_uid in robot.metric_pebble_recovery_seen_uids
            ):
                metrics.pebble_recovery_success_pairs.add(
                    (robot.robot_id, branch_uid)
                )
                metrics.pebble_recovery_success_count = len(
                    metrics.pebble_recovery_success_pairs
                )
                robot.metric_pebble_recovery_seen_uids.discard(branch_uid)
        robot.metric_inside_visited_uids = current_inside

        guidance_active = robot.last_pebble_guidance_mode != "NONE"
        if guidance_active and not robot.metric_pebble_guidance_active:
            metrics.pebble_guidance_activation_count += 1
        robot.metric_pebble_guidance_active = guidance_active

    metrics.initialized_visited_uids.update(visited_states)
    current_visited_branch_leakage = leakage
    current_pebble_guidance_active_normals = sum(
        robot.role == "NORMAL"
        and robot.last_pebble_guidance_mode != "NONE"
        for robot in robots
    )
    for branch in BRANCHES:
        metrics.visited_branch_max_inside[branch] = max(
            metrics.visited_branch_max_inside[branch],
            leakage[branch],
        )
        metrics.visited_branch_robot_seconds[branch] += leakage[branch] * dt
        if exploration_leak_phase:
            metrics.exploration_leak_max_inside[branch] = max(
                metrics.exploration_leak_max_inside[branch],
                leakage[branch],
            )
            metrics.exploration_leak_robot_seconds[branch] += (
                leakage[branch] * dt
            )

    if (
        visited_states
        and simulation_time - pebble_flow_last_log_time
        >= PEBBLE_FLOW_LOG_INTERVAL
    ):
        pebble_flow_last_log_time = simulation_time
        representative = max(
            (
                robot for robot in robots
                if robot.role == "NORMAL"
                and robot.last_pebble_guidance_mode != "NONE"
            ),
            key=lambda robot: robot.last_pebble_guidance_weight,
            default=None,
        )
        if representative is not None:
            print(
                f"[PebbleFlow] robot={representative.robot_id} "
                f"branch={representative.last_pebble_guidance_branch_uid} "
                f"mode={representative.last_pebble_guidance_mode} "
                f"weight={representative.last_pebble_guidance_weight:.2f} "
                f"active_normals={current_pebble_guidance_active_normals}"
            )
        for state in visited_states.values():
            branch = state.branch_key
            if branch not in BRANCHES:
                continue
            print(
                f"[PebbleLeak] branch={branch} "
                f"normals_inside={leakage[branch]} "
                f"max={metrics.visited_branch_max_inside[branch]} "
                f"robot_s={metrics.visited_branch_robot_seconds[branch]:.2f}"
            )


def start_shepherd_pressure_push(robots, branch):
    """Start a prepared Shepherd piston without repeating the fill wait."""
    global phase, pressure_push_timer, flow_establish_timer
    global shepherd_flow_timer
    global branch_backflow_started
    branch_backflow_started[branch] = True
    next_branch = next_unvisited_transfer_branch(branch, robots)
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
    update_local_ingress_tangents(robots)
    update_backtrack_bridge_guards(robots, dt)

    if phase in {
        SimulationPhase.EXPLORE_BRANCH,
        SimulationPhase.FORM_SHEPHERD_BOUNDARY,
        SimulationPhase.FILL_BEHIND_SHEPHERD,
    }:
        branch_entry_timer += dt
    update_initial_release_flow_event(robots, dt)

    if phase == SimulationPhase.MOVE_TO_JUNCTION:
        update_relay_deployment(robots, dt)
        junction_confirmed = junction_inference_tracker.update(robots, dt)
        if junction_confirmed:
            begin_junction_guard_formation(robots)
            phase = SimulationPhase.FORM_JUNCTION_GUARDS
            print(
                "[Junction] confirmed while mouths were OPEN; "
                "forming branch-wise full guards before ordering"
            )

    elif phase == SimulationPhase.FORM_JUNCTION_GUARDS:
        junction_guard_formation_timer += dt
        if pending_branch_start is not None:
            unselected_walls_ready = thick_mouth_guards_formed(
                robots,
                pending_branch_start,
            )
            selected_frontier_ready = refresh_frontier_row_readiness(
                robots,
                pending_branch_start,
            )
            thick_walls_ready = (
                unselected_walls_ready and selected_frontier_ready
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
                    "[PhysicalCoverage] Branch opening remains paused: "
                    f"unselected-walls={unselected_walls_ready}, "
                    f"selected-frontier={selected_frontier_ready}, "
                    f"coverage={frontier_line_physical_coverage_ratio:.2f}, "
                    f"edge-gaps={frontier_line_left_edge_gap:.1f}/"
                    f"{frontier_line_right_edge_gap:.1f}, "
                    f"continuous={frontier_line_continuous}"
                )
                log_local_guard_formation_diagnostics(robots)
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
            if (
                junction_guard_formation_timer
                >= JUNCTION_GUARD_FORM_TIMEOUT
                and not formed
            ):
                log_junction_guard_wait_diagnostics(robots)
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
            record_distributed_consensus(voted_branch)
            selected = choose_next_branch(
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
        log_frontier_explore_diagnostics(robots, active_branch)
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
            branch_dead_end_confirmed[active_branch] = True
            observed_depth = dead_end_inference_tracker.confirmed_depth
            # The same frontier line keeps following the NORMAL front at its
            # existing advance rate.  It may form the return piston only after
            # physically reaching the locally contacted depth, so there is no
            # map-directed sprint to a known terminal coordinate.
            frontier_shepherds = get_frontier_shepherds(
                robots,
                active_branch,
            )
            descriptor = branch_motion_descriptor(active_branch)
            physical_line_reached = (
                descriptor is not None
                and bool(frontier_shepherds)
                and all(
                    observed_branch_axial_depth(robot.position, descriptor)
                    >= observed_depth - JUNCTION_GUARD_POSITION_TOLERANCE
                    for robot in frontier_shepherds
                )
            )
            if not physical_line_reached:
                blocking_reason = (
                    "LOCAL_DESCRIPTOR_UNAVAILABLE"
                    if descriptor is None
                    else (
                        "NO_FRONTIER_ROBOTS"
                        if not frontier_shepherds
                        else "FRONTIER_BEHIND_CONFIRMED_LOCAL_DEPTH"
                    )
                )
                log_dead_end_transition_diagnostics(
                    robots,
                    active_branch,
                    False,
                    blocking_reason,
                )
                return
            selected = promote_existing_frontier_line(
                robots,
                active_branch,
                dead_end_inference_tracker.handoff_depth,
            )
            if selected:
                log_dead_end_transition_diagnostics(
                    robots,
                    active_branch,
                    True,
                    "READY_LOCAL_HANDOFF",
                    force=True,
                )
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
            else:
                dead_end_inference_tracker.blocking_reason = (
                    "LOCAL_HANDOFF_TARGET_UNWALKABLE_OR_INCOMPLETE"
                )
                log_dead_end_transition_diagnostics(
                    robots,
                    active_branch,
                    physical_line_reached,
                    "LOCAL_HANDOFF_TARGET_UNWALKABLE_OR_INCOMPLETE",
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
        remaining = sum(
            robot.role != "PEBBLE"
            and get_robot_region(robot.position) == active_branch
            for robot in robots
        )
        in_junction = sum(
            get_robot_region(robot.position) == "JUNCTION" and robot.role != "PEBBLE"
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
            if not complete_active_branch(
                completed_branch,
                robots,
                transfer_ready or base_transfer_ready or junction_ready,
            ):
                return
            if final_base_transfer_active:
                reset_shepherd_roles(robots)
                final_base_transfer_active = False
                begin_final_return(robots)
                print(
                    f"[Final Base Transfer] completed "
                    f"{completed_branch} -> BASE; "
                    f"robots={base_transferred_count}"
                )
            elif next_branch is not None:
                reset_shepherd_roles(robots)
                distributed_consensus_branch = branch_uid_for_fixture(
                    next_branch
                )
                selected = choose_next_branch(
                    robots,
                    reference_density,
                )
                if selected is not None:
                    finish_cross_branch_transfer(robots, selected)
                if draining_branch is not None:
                    branch_gate_states[draining_branch] = "OPEN"
                    record_distributed_consensus()
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
        discovered_children = discovered_branch_uids()
        visited_children = observed_visited_branch_uids(robots)
        if discovered_children and discovered_children <= visited_children:
            print(
                f"[JunctionComplete] visited={len(visited_children)}/"
                f"{len(discovered_children)}"
            )
            begin_final_gather()
            return
        begin_junction_guard_formation(robots)
        phase = SimulationPhase.FORM_JUNCTION_GUARDS
        print(
            "[Junction] backtracking complete; "
            "re-forming full guards before the next branch order"
        )

    elif phase == SimulationPhase.FINAL_JUNCTION_GATHER:
        stragglers = sum(
            robot.role != "PEBBLE"
            and get_robot_region(robot.position) in BRANCHES
            for robot in robots
        )
        gather_ready = (
            stragglers == 0
            and not get_relays(robots)
            and not get_shepherds(robots)
            and sum(robot.connected_to_base for robot in robots) == len(robots)
        )
        final_gather_timer = final_gather_timer + dt if gather_ready else 0.0
        if final_gather_timer >= FINAL_GATHER_DWELL_TIME:
            begin_final_return(robots)

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
        elif in_bottom < return_mobile_target_count(robots):
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
            in_bottom >= return_mobile_target_count(robots)
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
    """Draw the current peer-consensus gate commands."""
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
    global phase, active_branch, active_branch_uid, branch_states
    global branch_order_plan, branch_fixture_order_plan
    global branch_gate_states, distributed_consensus_branch, transfer_branch
    global final_base_transfer_active
    global transfer_path_max_gap, transfer_entrance_count
    global transfer_gap_control
    global transfer_target_motion_scale
    global branch_fill_target_count
    global branch_fill_current_count
    global branch_fill_deficit_control
    global branch_fill_feed_scale, branch_fill_feed_state
    global branch_fill_feed_last_logged_state
    global previous_branch_direction, simulation_time
    global junctions
    global branch_completion_epoch
    global branch_dead_end_confirmed, branch_backflow_started
    global branch_descriptors_by_uid, fixture_key_to_branch_uid
    global branch_uid_to_fixture_key, branch_discovery_counter
    global branch_local_uids
    global pebble_rx_logged
    global pending_pebble_robot_ids
    global pebble_flow_last_log_time
    global current_visited_branch_leakage
    global current_pebble_guidance_active_normals
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
    global last_proxy_candidates, last_decision_candidate_uids
    global last_flow_rollout_scores
    global selected_branch_entry_lambda, branch_entry_timer
    global return_trunk_release_pending, return_trunk_retract_timer
    global return_trunk_last_released_id, return_trunk_force_timer
    global return_done_dwell, return_entry_stall_timer
    global return_last_bottom_count
    global junction_guard_groups, junction_guard_formation_timer
    global junction_guard_frontier_depths
    global junction_guard_stable_dwell, junction_guard_status
    global junction_guard_last_wait_log_time
    global pending_branch_start
    global local_guard_width_fallback_logged, local_guard_pending_logged
    global thick_mouth_guard_layers, thick_mouth_guard_columns
    global frontier_line_branch, frontier_line_depth
    global frontier_line_lateral_center
    global frontier_line_target_settled_ratio
    global frontier_line_current_span, frontier_line_target_span
    global frontier_line_physical_coverage_ratio
    global frontier_line_left_edge_gap, frontier_line_right_edge_gap
    global frontier_line_continuous
    global frontier_line_row_ready, frontier_line_last_diagnostic_time
    global frontier_dead_end_transition_last_log_time
    global observed_dead_end_depths
    global last_distributed_vote_counts
    global last_distributed_voter_count, last_distributed_vote_quorum
    global last_distributed_vote_status
    global metrics
    phase = SimulationPhase.MOVE_TO_JUNCTION
    active_branch = FIXED_BRANCH_ORDER[0]
    active_branch_uid = None
    junctions = create_single_junction_registry()
    branch_states = get_junction_state().branch_states
    branch_order_plan = []
    branch_fixture_order_plan = []
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
    branch_fill_feed_scale = 1.0
    branch_fill_feed_state = "OPEN"
    branch_fill_feed_last_logged_state = "OPEN"
    previous_branch_direction = pygame.Vector2(0.0, -1.0)
    branch_completion_epoch = 0
    branch_dead_end_confirmed = {branch: False for branch in BRANCHES}
    branch_backflow_started = {branch: False for branch in BRANCHES}
    branch_descriptors_by_uid = {}
    fixture_key_to_branch_uid = {}
    branch_uid_to_fixture_key = {}
    branch_discovery_counter = 0
    branch_local_uids = {}
    pebble_rx_logged = set()
    pending_pebble_robot_ids = {}
    pebble_flow_last_log_time = float("-inf")
    current_visited_branch_leakage = {
        branch: 0 for branch in BRANCHES
    }
    current_pebble_guidance_active_normals = 0
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
    last_decision_candidate_uids = ()
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
    junction_guard_last_wait_log_time = float("-inf")
    pending_branch_start = None
    local_guard_width_fallback_logged = set()
    local_guard_pending_logged = set()
    thick_mouth_guard_layers = {branch: 0 for branch in BRANCHES}
    thick_mouth_guard_columns = {branch: 0 for branch in BRANCHES}
    frontier_line_branch = None
    frontier_line_depth = 0.0
    frontier_line_lateral_center = 0.0
    frontier_line_target_settled_ratio = 0.0
    frontier_line_current_span = 0.0
    frontier_line_target_span = 0.0
    frontier_line_physical_coverage_ratio = 0.0
    frontier_line_left_edge_gap = float("inf")
    frontier_line_right_edge_gap = float("inf")
    frontier_line_continuous = False
    frontier_line_row_ready = False
    frontier_line_last_diagnostic_time = float("-inf")
    frontier_dead_end_transition_last_log_time = float("-inf")
    observed_dead_end_depths = {}
    last_distributed_vote_counts = {}
    last_distributed_voter_count = 0
    last_distributed_vote_quorum = DISTRIBUTED_VOTE_MIN_ROBOTS
    last_distributed_vote_status = "INACTIVE"
    saturation_tracker.reset()
    branch_continuity_tracker.reset()
    junction_consensus_tracker.reset()
    junction_inference_tracker.reset()
    dead_end_inference_tracker.reset()
    metrics = ExperimentMetrics()


def initialize_simulation():
    reset_dfs_state()
    run_diagonal_local_guard_synthetic_validation()
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
headless_fast = os.environ.get("SPH_DFS_HEADLESS_FAST", "0") == "1"
headless_max_frames = int(os.environ.get("SPH_DFS_MAX_FRAMES", "0"))
headless_frame_dt = float(
    os.environ.get("SPH_DFS_HEADLESS_DT", str(1.0 / FPS))
)
headless_frame_count = 0
capture_directory_value = os.environ.get("SPH_DFS_CAPTURE_DIR", "")
capture_directory = (
    Path(capture_directory_value)
    if capture_directory_value
    else None
)
capture_last_signature = None
if capture_directory is not None:
    capture_directory.mkdir(parents=True, exist_ok=True)
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
    raw_dt = (
        headless_frame_dt
        if headless_fast
        else max(clock.tick(FPS) / 1000.0, 1.0 / 240.0)
    )
    headless_frame_count += 1
    if headless_fast and phase == SimulationPhase.DONE:
        print(
            f"[Headless] frames={headless_frame_count}, phase=DONE, "
            f"pebbles={len(get_pebbles(robots))}, "
            f"visited={sorted(observed_visited_branches(robots))}"
        )
        running = False
        continue
    if headless_max_frames and headless_frame_count > headless_max_frames:
        print(
            f"[Headless] frames={headless_max_frames}, phase={phase.name}, "
            f"pebbles={len(get_pebbles(robots))}, "
            f"visited={sorted(observed_visited_branches(robots))}"
        )
        running = False
        continue
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
        _record_handoff_state(robots)
        _record_motion_frontier_frames(robots)
    else:
        update_communication_system(robots, spatial_grid)
        compute_densities(robots, build_physics_grid(robots))
        compute_pressures(robots, reference_density)

    if LOCAL_MOUTH_CROSSING_DIAGNOSTICS_ENABLED and active_branch in BRANCHES:
        if phase in {
            SimulationPhase.FORM_SHEPHERD_BOUNDARY,
            SimulationPhase.FILL_BEHIND_SHEPHERD,
            SimulationPhase.PRESSURE_PUSH,
            SimulationPhase.FLOW_BACKTRACK,
        }:
            local_mouth_shepherd_branches.add(active_branch)
        if phase == SimulationPhase.PRESSURE_PUSH:
            local_mouth_pressure_branches.add(active_branch)
        if phase == SimulationPhase.FLOW_BACKTRACK:
            local_mouth_backflow_branches.add(active_branch)

    # Headless verification exercises the identical physics/state/metrics
    # path above; only rasterization and display presentation are skipped.
    if headless_fast:
        if capture_directory is None:
            continue
        capture_probe_signature = (
            phase.name,
            active_branch,
            junction_guard_status
            if phase == SimulationPhase.FORM_JUNCTION_GUARDS
            else "-",
            int(frontier_line_depth // max(40.0 * MAP_SCALE, 1.0))
            if phase == SimulationPhase.EXPLORE_BRANCH
            else -1,
            dead_end_inference_tracker.confirmed,
        )
        if capture_probe_signature == capture_last_signature:
            continue

    screen.fill(BACKGROUND_COLOR)
    pygame.draw.polygon(screen, FLOOR_COLOR, cross_points)
    draw_branch_colour_fields(screen)
    pygame.draw.polygon(screen, WALL_COLOR, cross_points, width=2)
    draw_branch_gates(screen)
    # Eguchi contact points remain internal inference data; do not leave
    # persistent magenta robot/contact traces on the environment view.

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
        robot.role != "PEBBLE"
        and get_robot_region(robot.position) in BRANCHES
        for robot in robots
    )
    return_bottom_count = sum(
        get_robot_region(robot.position) == "BOTTOM"
        for robot in robots
    )
    return_junction_count = sum(
        robot.role != "PEBBLE"
        and get_robot_region(robot.position) == "JUNCTION"
        for robot in robots
    )
    return_special_count = sum(
        robot.role in {
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
            f"Pebble policy={PEBBLE_POLICY_VERSION} | "
            f"count={len(get_pebbles(robots))} | "
            f"visited-uids={sorted(observed_visited_branch_uids(robots))}/"
            f"{len(discovered_branch_uids()) or len(BRANCHES)} | "
            f"consensus={branch_identity_label(distributed_consensus_branch)}"
        ),
        (
            "Visited branch leakage: "
            + " | ".join(
                f"{branch}={current_visited_branch_leakage[branch]}"
                for branch in BRANCHES
            )
        ),
        (
            "Pebble guidance active normals="
            f"{current_pebble_guidance_active_normals} | "
            f"activations={metrics.pebble_guidance_activation_count} | "
            f"recoveries={metrics.pebble_recovery_success_count}"
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
        (
            f"Branch={branch_identity_label(active_branch_uid)}"
            if phase not in {
                SimulationPhase.MOVE_TO_JUNCTION,
                SimulationPhase.RETURN_TO_BASE,
                SimulationPhase.DONE,
            }
            else "Branch=-"
        ),
        (
            "Distributed decision=MOVE_"
            f"{branch_identity_label(distributed_consensus_branch)}"
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
            f"Branch fill={BRANCH_FILL_QUOTA_POLICY_VERSION} | "
            f"{active_branch}={branch_fill_current_count}/"
            f"{branch_fill_target_count} | "
            f"feed={branch_fill_feed_state} "
            f"scale={branch_fill_feed_scale:.2f} | "
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
            f"Guard phase status={junction_guard_status} "
            f"timer={junction_guard_formation_timer:.2f} "
            f"formed={junction_guards_formed(robots)}"
        ),
        junction_guard_wait_hud_summary(robots),
        distributed_vote_hud_summary(),
        (
            "Local Guard: "
            + (
                " || ".join(
                    local_guard_debug_summary(
                        branch_descriptors_by_uid[uid],
                        robots,
                    )
                    for uid in ordered_discovered_branch_uids()
                )
                if branch_descriptors_by_uid
                else "-"
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
            f"settled={frontier_line_target_settled_ratio:.2f} "
            f"span={frontier_line_current_span:.1f}/"
            f"{frontier_line_target_span:.1f} "
            f"coverage={frontier_line_physical_coverage_ratio:.2f} "
            f"edge={frontier_line_left_edge_gap:.1f}/"
            f"{frontier_line_right_edge_gap:.1f} "
            f"continuous={frontier_line_continuous} "
            f"center={frontier_line_lateral_center:.1f} "
            f"ready={frontier_line_row_ready} "
            f"ids={[robot.robot_id for robot in get_frontier_shepherds(robots)]}"
        ),
        (
            "Local physical widths: "
            + (
                " | ".join(
                    f"{branch_identity_label(uid)} "
                    f"flow={branch_descriptors_by_uid[uid].observed_flow_width:.1f} "
                    f"physical={branch_descriptors_by_uid[uid].observed_physical_width:.1f} "
                    f"conf={branch_descriptors_by_uid[uid].physical_width_confident}"
                    for uid in ordered_discovered_branch_uids()
                )
                if branch_descriptors_by_uid
                else "-"
            )
        ),
        (
            "Observed guard frontiers: "
            + " | ".join(
                f"{branch}={junction_guard_frontier_depths.get(branch, 0.0):.1f}"
                for branch in BRANCHES
            )
        ),
        (
            "Detected Branch UIDs: "
            + (
                " | ".join(
                    f"{branch_identity_label(uid)}="
                    f"{branch_descriptors_by_uid[uid].visit_state}"
                    for uid in ordered_discovered_branch_uids()
                )
                if branch_descriptors_by_uid
                else "-"
            )
        ),
        (
            "Pebble UIDs: "
            + (
                " | ".join(
                    branch_identity_label(pebble.pebble_branch_uid)
                    for pebble in get_pebbles(robots)
                )
                if get_pebbles(robots)
                else "-"
            )
        ),
        (
            "Order="
            + (
                " > ".join(branch_identity_label(uid) for uid in branch_order_plan)
                if branch_order_plan
                else "-"
            )
        ),
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
                f"{branch_identity_label(branch_uid_for_fixture(branch))} "
                f"q={last_proxy_mass_stats.get(branch, {}).get('quota_fraction', 0.0):.2f} "
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
            f"uids={[branch_identity_label(uid) for uid in ordered_discovered_branch_uids()]} "
            f"dir-stable="
            f"{sum(descriptor.direction_is_stable for descriptor in branch_descriptors_by_uid.values())}/"
            f"{len(branch_descriptors_by_uid)} "
            f"dir-mature="
            f"{sum(descriptor.direction_is_mature for descriptor in branch_descriptors_by_uid.values())}/"
            f"{len(branch_descriptors_by_uid)} "
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
            f"branch-n={dead_end_inference_tracker.branch_robot_count}"
        ),
        (
            "Dead-end progress: "
            f"frontier-contact={dead_end_inference_tracker.shepherd_direct_contact_ratio:.2f} "
            f"forward={dead_end_inference_tracker.frontier_forward_progress:.1f} "
            f"rate={dead_end_inference_tracker.frontier_progress_rate:.2f} "
            f"dwell={dead_end_inference_tracker.dwell:.2f} "
            f"blocking={dead_end_inference_tracker.blocking_reason} "
            f"mode={dead_end_inference_tracker.confirmation_mode} "
            f"confirmed={dead_end_inference_tracker.confirmed} "
            f"depth={dead_end_inference_tracker.confirmed_depth:.1f}"
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
            f"Return status: B={return_bottom_count}/"
            f"{return_mobile_target_count(robots)} "
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
    if capture_directory is not None:
        capture_signature = (
            phase.name,
            active_branch,
            junction_guard_status
            if phase == SimulationPhase.FORM_JUNCTION_GUARDS
            else "-",
            int(frontier_line_depth // max(40.0 * MAP_SCALE, 1.0))
            if phase == SimulationPhase.EXPLORE_BRANCH
            else -1,
            dead_end_inference_tracker.confirmed,
        )
        if capture_signature != capture_last_signature:
            capture_last_signature = capture_signature
            capture_name = (
                f"{headless_frame_count:05d}_{phase.name}_"
                f"{active_branch}_{max(0, round(frontier_line_depth)):04d}.png"
            )
            capture_path = capture_directory / capture_name
            pygame.image.save(screen, capture_path)
            print(f"[Capture] {capture_path}")

if not metrics.saved:
    save_experiment_logs(robots, "USER_EXIT")
if HANDOFF_DIAGNOSTICS_ENABLED:
    handoff_diagnostics.save(HANDOFF_DIAGNOSTICS_OUTPUT_DIR)
if MOTION_FRAME_BIAS_DIAGNOSTICS_ENABLED:
    motion_frame_bias_diagnostics.save(
        MOTION_FRAME_BIAS_DIAGNOSTICS_OUTPUT_DIR
    )
if MOUTH_PCA_SAMPLE_DIAGNOSTICS_ENABLED:
    mouth_pca_sample_diagnostics.save(
        MOUTH_PCA_SAMPLE_DIAGNOSTICS_OUTPUT_DIR
    )
if MOUTH_CROSSING_ORIGIN_DIAGNOSTICS_ENABLED:
    mouth_crossing_origin_diagnostics.save(
        MOUTH_CROSSING_ORIGIN_DIAGNOSTICS_OUTPUT_DIR
    )
if LOCAL_MOUTH_CROSSING_DIAGNOSTICS_ENABLED:
    local_mouth_crossing_diagnostics.record_run_summary(
        final_phase=phase.name,
        simulation_time=simulation_time,
        visited_branches=observed_visited_branches(robots),
        shepherd_formed_branches=local_mouth_shepherd_branches,
        pressure_branches=local_mouth_pressure_branches,
        backflow_branches=local_mouth_backflow_branches,
    )
    local_mouth_crossing_diagnostics.save(
        LOCAL_MOUTH_CROSSING_OUTPUT_DIR
    )
pygame.quit()
sys.exit()
