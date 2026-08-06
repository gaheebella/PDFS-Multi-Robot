"""SPH-based localization-free natural-spread DFS prototype (method B).

Implemented research components
-------------------------------
1. A manually seeded front Leader captures a width-adaptive K-hop tree and
   packs its Shepherds into a three-row curved physical cap before moving.
2. The tree is rebuilt from the live local communication graph while the cap
   slots keep it cohesive instead of allowing a long, arbitrary BFS chain.
3. Junction discovery never uses a forward/open-angle probe. Ordinary SPH
   motion continues until several robots naturally acquire a persistent
   non-corridor velocity and displacement while remaining locally connected.
4. The first lateral flow only registers a Junction candidate. The main body
   is then slowed and the already-occurring spread is range-limited while
   directional cohorts prove count, travel, persistence, coherence,
   connectivity, and multi-robot passability.
5. Actual SPH spread positions are clustered through the live communication
   graph. Inside each component, branch Leaders are elected by the Max-Min
   d-cluster heuristic (d Floodmax plus d Floodmin rounds, Rules 1-4,
   gateways, and convergecast parents) from Amis et al., INFOCOM 2000.
6. All elected branch caps form at their entrances first. Only one cap then
   explores at a time in locally evaluated rollout-cost order;
   inactive caps remain physical gates. A confirmed cap returns before the
   next branch is activated, matching the staged sequential DFS sketch.
7. Leader and Marker state is exchanged through bounded-hop packets over the
   same live communication graph used by the capture trees.
8. Local forward clearance, pressure, contact compression, density, speed, and
   dwell confirm a dead-end without exposing dead-end coordinates.
9. A physical Marker/Pebble broadcasts PENDING -> COMPLETED -> BLOCKING, and a
   retained two-row Shepherd group replaces each simulated virtual gate.
10. Compatible groups merge through local Leader packets; duplicate Leaders and
   surplus Shepherds are released into the remaining streams.
11. Reactive Breadcrumb robots are frozen only when the measured Base-path
   margin of an advancing K-hop stream needs support.

Scope limitation
----------------
The environment still contains one T/cross junction for controlled comparison.
Map rectangles are used by collision and rendering only; the d-hop controller
does not query Junction, Branch, or dead-end rectangles. Branch cohorts are
discovered from motion rather than map-labelled regions. Method A (boundary
surface tension) remains a separate follow-up experiment.
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


def integer_cli_option(name: str, default: Optional[int]) -> Optional[int]:
    prefix = f"--{name}="
    for argument in sys.argv[1:]:
        if argument.startswith(prefix):
            return int(argument[len(prefix):])
    return default


def string_cli_option(name: str, default: Optional[str]) -> Optional[str]:
    prefix = f"--{name}="
    for argument in sys.argv[1:]:
        if argument.startswith(prefix):
            return argument[len(prefix):]
    return default


HEADLESS_STEP_LIMIT = integer_cli_option("headless-steps", None)
HEADLESS_DT_MS = integer_cli_option("headless-dt-ms", 1000 // 60) or 16
HEADLESS_SNAPSHOT_PATH = string_cli_option("headless-snapshot", None)
INITIAL_KHOP_LEADER_ID = integer_cli_option("initial-leader-id", None)

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
    "Pressure-Driven SPH | Localization-free K-hop Capture Shepherd"
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
KHOP_LEADER_COLOR = (30, 92, 184)
MARKER_PENDING_COLOR = (230, 174, 48)
MARKER_COMPLETE_COLOR = (220, 70, 70)
GATEKEEPER_COLOR = (194, 52, 126)
KHOP_TREE_LINK_COLOR = (133, 82, 184)
DHOP_ACTIVE_STREAM_COLOR = (105, 210, 105)
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

# Pre-arm the final slot line before any leader can cross it.  The wider
# upstream band retains approaching candidates against that line until the
# width-adaptive peer auction can fill every slot.
SHEPHERD_PLACEMENT_STANDOFF = round(14 * MAP_SCALE)
SHEPHERD_ELECTION_BAND_DEPTH = round(60 * MAP_SCALE)
SHEPHERD_ELECTION_POLICY_VERSION = "PREARMED_SLOT_APPROACH_CAPTURE_V3"

early_capture_regions = {
    "UP": pygame.Rect(
        center_x - half_width,
        (
            center_y
            - half_width
            - normal_length
            + SHEPHERD_PLACEMENT_STANDOFF
        ),
        corridor_width,
        SHEPHERD_ELECTION_BAND_DEPTH,
    ),
    "LEFT": pygame.Rect(
        (
            center_x
            - half_width
            - normal_length
            + SHEPHERD_PLACEMENT_STANDOFF
        ),
        center_y - half_width,
        SHEPHERD_ELECTION_BAND_DEPTH,
        corridor_width,
    ),
    "RIGHT": pygame.Rect(
        (
            center_x
            + half_width
            + right_length
            - SHEPHERD_PLACEMENT_STANDOFF
            - SHEPHERD_ELECTION_BAND_DEPTH
        ),
        center_y - half_width,
        SHEPHERD_ELECTION_BAND_DEPTH,
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


class SpaceRecognitionState(Enum):
    """Motion-evidence state machine used before a Junction exists."""

    NORMAL_CORRIDOR = auto()
    SPREAD_ANOMALY = auto()
    JUNCTION_CANDIDATE = auto()
    CONTROLLED_SPREAD = auto()
    COHORT_FORMATION = auto()
    COHORT_VALIDATION = auto()
    JUNCTION_CONFIRMED = auto()
    NOT_A_JUNCTION = auto()
    COHORT_MERGE = auto()


class KHopGroupState(Enum):
    FORMING = auto()
    WAITING = auto()
    EXPLORING = auto()
    DEAD_END_CONFIRMED = auto()
    RETURNING = auto()
    MERGING = auto()
    GATEKEEPING = auto()
    RELEASED = auto()


class MarkerState(Enum):
    UNSET = auto()
    PENDING = auto()
    COMPLETED = auto()
    BLOCKING = auto()
    RELEASED = auto()


class KHopMessageKind(Enum):
    """Packets that robots may broadcast over their current local mesh."""

    LEADER_STATUS = auto()
    MARKER_STATUS = auto()
    RELAY_BEACON = auto()


@dataclass(frozen=True)
class KHopMessage:
    """Immutable bounded-hop packet delivered only through comm_neighbors."""

    kind: KHopMessageKind
    source_id: int
    source_sequence: int
    group_id: Optional[str]
    hop_count: int
    hop_limit: int
    created_at: float
    state: str
    via_id: Optional[int] = None
    heading: tuple[float, float] = (0.0, 0.0)
    marker_id: Optional[str] = None
    marker_state: Optional[MarkerState] = None
    dead_end_confirmed: bool = False
    pressure_ratio: float = 0.0
    density_ratio: float = 0.0
    compression_ratio: float = 0.0
    connectivity_ratio: float = 0.0

    @property
    def identity(self) -> tuple[KHopMessageKind, int, int]:
        return (self.kind, self.source_id, self.source_sequence)

    def forwarded(self, forwarder_id: int) -> "KHopMessage":
        return KHopMessage(
            kind=self.kind,
            source_id=self.source_id,
            source_sequence=self.source_sequence,
            group_id=self.group_id,
            hop_count=self.hop_count + 1,
            hop_limit=self.hop_limit,
            via_id=forwarder_id,
            created_at=self.created_at,
            state=self.state,
            heading=self.heading,
            marker_id=self.marker_id,
            marker_state=self.marker_state,
            dead_end_confirmed=self.dead_end_confirmed,
            pressure_ratio=self.pressure_ratio,
            density_ratio=self.density_ratio,
            compression_ratio=self.compression_ratio,
            connectivity_ratio=self.connectivity_ratio,
        )


@dataclass
class MarkerRecord:
    """Physical Pebble state broadcast by one robot at a split entrance."""

    marker_id: str
    group_id: str
    robot_id: int
    position: pygame.Vector2
    origin_leader_id: int
    state: MarkerState = MarkerState.PENDING
    dead_end_confirmed: bool = False
    sequence_number: int = 0
    completion_witness_id: Optional[int] = None
    last_broadcast_at: float = 0.0
    parent_junction_id: Optional[str] = None
    local_branch_direction: tuple[float, float] = (0.0, 0.0)
    cohort_robot_ids: tuple[int, ...] = ()
    estimated_passage_width: float = 0.0
    required_guard_count: int = 0
    branch_leader_id: Optional[int] = None
    rollout_cost: float = float("inf")
    rollout_rank: int = -1
    return_direction_local: tuple[float, float] = (0.0, 0.0)
    marker_epoch: int = 1


@dataclass
class KHopShepherdGroup:
    """One dynamically rebuilt Leader-rooted capture tree."""

    group_id: str
    leader_id: int
    heading: pygame.Vector2
    origin: pygame.Vector2
    display_branch: str
    parent_group_id: Optional[str] = None
    state: KHopGroupState = KHopGroupState.FORMING
    member_robot_ids: set[int] = field(default_factory=set)
    seed_robot_ids: set[int] = field(default_factory=set)
    tree_parent: dict[int, Optional[int]] = field(default_factory=dict)
    hop_by_robot: dict[int, int] = field(default_factory=dict)
    cap_slot_by_robot: dict[int, tuple[float, float]] = field(default_factory=dict)
    cap_mean_error: float = float("inf")
    cap_formation_dwell: float = 0.0
    cap_formation_elapsed: float = 0.0
    cap_anchor: Optional[pygame.Vector2] = None
    dhop_winner_by_robot: dict[int, list[int]] = field(default_factory=dict)
    dhop_sender_by_robot: dict[int, list[int]] = field(default_factory=dict)
    dhop_clusterhead_by_robot: dict[int, int] = field(default_factory=dict)
    dhop_rule_by_robot: dict[int, int] = field(default_factory=dict)
    dhop_clusterhead_ids: set[int] = field(default_factory=set)
    dhop_gateway_ids: set[int] = field(default_factory=set)
    dhop_parent_by_robot: dict[int, Optional[int]] = field(default_factory=dict)
    marker_id: Optional[str] = None
    dead_end_confirmed: bool = False
    dead_end_dwell: float = 0.0
    failure_dwell: float = 0.0
    return_reason: Optional[str] = None
    retry_count: int = 0
    leader_candidate_id: Optional[int] = None
    leader_candidate_dwell: float = 0.0
    average_heading: pygame.Vector2 = field(
        default_factory=lambda: pygame.Vector2(0.0, -1.0)
    )
    average_pressure: float = 0.0
    average_density_ratio: float = 0.0
    average_speed: float = 0.0
    forward_clearance: float = 0.0
    front_pressure_ratio: float = 0.0
    front_density_ratio: float = 0.0
    contact_compression_ratio: float = 0.0
    connectivity_ratio: float = 0.0
    pressure_evidence_ema: float = 0.0
    density_evidence_ema: float = 0.0
    compression_evidence_ema: float = 0.0
    connectivity_ema: float = 1.0
    heading_stability_ema: float = 0.0
    estimated_width: float = 0.0
    relay_deploy_cooldown: float = 0.0
    merge_partner_id: Optional[str] = None
    merge_dwell: float = 0.0
    merged_into: Optional[str] = None
    created_at: float = 0.0
    rollout_cost: float = float("inf")
    rollout_rank: int = -1


@dataclass
class DirectionalCohortEvidence:
    """Persistent evidence for one motion-created outgoing flow."""

    heading: pygame.Vector2
    robot_ids: set[int] = field(default_factory=set)
    dwell: float = 0.0
    mean_travel: float = 0.0
    direction_variance: float = 1.0
    connectivity_ratio: float = 0.0
    connected_to_base_ratio: float = 0.0
    lateral_width: float = 0.0
    valid: bool = False


@dataclass
class KHopCaptureState:
    """Global experiment observer; motion decisions remain local to groups."""

    stage: str = "UNINITIALIZED"
    root_group_id: Optional[str] = None
    split_origin: Optional[pygame.Vector2] = None
    groups: dict[str, KHopShepherdGroup] = field(default_factory=dict)
    markers: dict[str, MarkerRecord] = field(default_factory=dict)
    robot_by_id: dict[int, "Robot"] = field(default_factory=dict)
    split_dwell: float = 0.0
    capture_refresh_elapsed: float = 0.0
    retry_dwell: float = 0.0
    consensus_sequence: int = 0
    message_delivery_count: int = 0
    completion_consensus_dwell: float = 0.0
    last_split_group_size: int = 0
    controlled_spread_elapsed: float = 0.0
    split_cluster_signature: tuple[int, ...] = ()
    split_cluster_dwell: float = 0.0
    pending_split_clusters: list[KHopSplitCluster] = field(default_factory=list)
    last_cluster_headings: list[tuple[float, float]] = field(default_factory=list)
    last_cluster_sizes: list[int] = field(default_factory=list)
    branch_queue: list[str] = field(default_factory=list)
    active_group_id: Optional[str] = None
    sequential_visit_order: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    recognition_state: SpaceRecognitionState = (
        SpaceRecognitionState.NORMAL_CORRIDOR
    )
    anomaly_dwell: float = 0.0
    candidate_elapsed: float = 0.0
    candidate_cooldown: float = 0.0
    motion_origin_by_robot: dict[int, pygame.Vector2] = field(
        default_factory=dict
    )
    motion_dwell_by_robot: dict[int, float] = field(default_factory=dict)
    cohort_evidence: list[DirectionalCohortEvidence] = field(
        default_factory=list
    )
    candidate_observer_ids: set[int] = field(default_factory=set)
    straggler_wait_elapsed: float = 0.0
    return_to_base_dwell: float = 0.0


@dataclass
class KHopSplitCluster:
    """A persistent directional component observed near the spreading front."""

    heading: pygame.Vector2
    robot_ids: set[int]
    angle: float
    mean_travel: float = 0.0
    direction_variance: float = 1.0
    connectivity_ratio: float = 0.0
    connected_to_base_ratio: float = 0.0
    lateral_width: float = 0.0


@dataclass
class MaxMinDClusterResult:
    """Per-node output of the paper's 2d-round Max-Min heuristic."""

    depth: int
    winner_by_robot: dict[int, list[int]]
    sender_by_robot: dict[int, list[int]]
    clusterhead_by_robot: dict[int, int]
    rule_by_robot: dict[int, int]
    clusterhead_ids: set[int]
    gateway_ids: set[int]
    parent_by_robot: dict[int, Optional[int]]
    hop_to_clusterhead: dict[int, int]


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
        gate_states={branch: "CLOSED" for branch in BRANCHES},
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
branch_overflow_return_robot_ids: set[int] = set()
junction_buffer_control = 0.0
junction_buffer_slice_counts: list[int] = []
junction_buffer_lateral_coverage: list[float] = []
base_substitute_count = 0
base_substitute_robot_ids: set[int] = set()
base_substitute_initial_base_count = 0
base_substitute_reserve_target = 0
base_junction_feed_robot_ids: set[int] = set()
base_junction_feed_branch: Optional[str] = None
base_junction_feed_initial_count = 0
base_junction_feed_reserve_target = 0
branch_feed_continuity_deficit = 0
branch_feed_slice_counts: list[int] = []
branch_feed_slice_coverages: list[float] = []
base_junction_interface_gap = 0.0
base_junction_interface_deficit = 0
base_junction_interface_slice_counts: list[int] = []
base_junction_interface_slice_coverages: list[float] = []
base_tail_refill_robot_ids: set[int] = set()
hose_continuity_forces: dict[int, pygame.Vector2] = {}
hose_continuity_link_count = 0
handoff_source_continuity_deficit = 0
handoff_source_slice_counts: list[int] = []
handoff_source_slice_coverages: list[float] = []
handoff_source_continuity_control = 0.0
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

# Method B is the default experiment.  The legacy fixed-map DFS controller is
# retained below as a comparison implementation, but no longer drives motion.
KHOP_CAPTURE_ENABLED = True
khop_state = KHopCaptureState()

# =========================================================
# 4. Physics and control parameters
# =========================================================

FLUID_BODY_POLICY_VERSION = "CONTINUOUS_SPH_BODY_760_ADAPTIVE_HOSE_TAIL_V6"
# The extra robots are a moving hose tail, not a permanently parked Base pool.
# A measured Branch/J0 deficit releases only the required upper Base layers;
# the rest remain available for the next cross-Branch handoff.
ROBOT_COUNT = integer_cli_option("robot-count", 760) or 760
BASE_RESERVE_HOLD_GAIN = 42.0
SPAWN_MODE = "grid"
ROBOT_RADIUS = 1.60 * MAP_SCALE
GRID_SPACING = 4.00 * MAP_SCALE
GRID_ROW_SPACING = 3.65 * MAP_SCALE

SMOOTHING_LENGTH = 22.0 * MAP_SCALE
PRESSURE_GAIN = 2800.0
SPH_MOTION_PRESSURE_BOOST = 3.00
SHEPHERD_PACKED_PRESSURE_BOOST = 5.00
STIFFNESS_EXPONENT = 0.5
VISCOSITY_XI1 = 0.9
VISCOSITY_XI2 = 1.2
MOTION_SPEED_MULTIPLIER = 3.0
DAMPING = 9.0
SAFE_RADIUS = 7.5 * MAP_SCALE
REPULSION_GAIN = 440.0
# This is a physical centre-to-centre spacing, not merely a quota hint.  Keep
# it visibly wider than the spawn lattice so the ordinary body expands back
# through J0 and into Base instead of preserving its initially packed spacing.
NORMAL_EQUILIBRIUM_SCALE = 2.80
# Physical-robot command conditioning.  Longitudinal motion remains fast,
# while high-frequency SPH reversals and corridor-crossing zigzags are damped.
ACCELERATION_FILTER_ALPHA = 0.07
ACCELERATION_SLEW_RATE = 450.0
CORRIDOR_LATERAL_VELOCITY_DAMPING = 20.0
EQUILIBRIUM_REPULSION_FORCE_LIMIT = 180.0 * MOTION_SPEED_MULTIPLIER
SPH_PRESSURE_FORCE_LIMIT = 420.0 * MOTION_SPEED_MULTIPLIER
SPH_VISCOSITY_FORCE_LIMIT = 150.0 * MOTION_SPEED_MULTIPLIER
VISCOELASTIC_LINK_RADIUS = SAFE_RADIUS * 3.00
VISCOELASTIC_REST_MIN = ROBOT_RADIUS * 2.05
VISCOELASTIC_REST_MAX = SAFE_RADIUS * 3.00
VISCOELASTIC_ELASTIC_GAIN = 42.0
VISCOELASTIC_DASHPOT_GAIN = 18.0
VISCOELASTIC_REST_RELAXATION = 0.20
VISCOELASTIC_EQUILIBRIUM_ADAPTATION = 4.0
VISCOELASTIC_VELOCITY_CONSENSUS_GAIN = 12.0
VISCOELASTIC_FORCE_LIMIT = 75.0 * MOTION_SPEED_MULTIPLIER
VISCOELASTIC_LINK_STALE_STEPS = 3
COMPRESSION_RELEASE_DENSITY_RATIO = 1.20
COMPRESSION_RELEASE_RADIUS = (
    SAFE_RADIUS * NORMAL_EQUILIBRIUM_SCALE * 1.15
)
COMPRESSION_RELEASE_GAIN = 135.0
COMPRESSION_RELEASE_FORCE_LIMIT = 180.0 * MOTION_SPEED_MULTIPLIER
COMPRESSION_RELEASE_RAMP_TIME = 0.24
COMPRESSION_RELEASE_DIRECTIONAL_TIME = 0.50
PRESSURE_POLICY_VERSION = "DIRECTIONAL_SPH_SELF_EXPANSION_V2"
VISCOELASTIC_MODEL_VERSION = "KELVIN_VOIGT_SPH_V2_SPACED"
MOTION_POLICY_VERSION = "OVERDAMPED_CONTINUOUS_HOSE_COMMAND_V7"

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
FILL_BRANCH_LANE_CENTERING_GAIN = 12.0
FILL_BRANCH_LANE_FORCE_LIMIT = ROUTE_FORCE * 3.00
BRANCH_FILL_QUOTA_COVERAGE = 1.00
BRANCH_FILL_QUOTA_AXIAL_PITCH_RATIO = 0.95
BRANCH_FILL_QUOTA_MIN_RESERVE_ROWS = 2
BRANCH_FILL_QUOTA_SOURCE_FORCE_BOOST = 1.80
BRANCH_FILL_QUOTA_FILTER_ALPHA = 0.35
BRANCH_FILL_QUOTA_JUNCTION_FEED_FLOOR = 0.00
# A purely geometric count becomes much too small when the requested physical
# equilibrium is wide: it stops Base feeding even though the visible hose tail
# is still broken.  Preserve the large equilibrium force, but allocate a
# compressible, length-aware share of the whole NORMAL body to the Branch.
LONG_BRANCH_HOSE_TARGET_SHARE = 0.64
NORMAL_BRANCH_HOSE_TARGET_SHARE = 0.56
BRANCH_FILL_QUOTA_POLICY_VERSION = "LENGTH_WIDTH_HOSE_SHARE_QUOTA_V5"
BRANCH_OVERFLOW_HYSTERESIS_RATIO = 0.08
BRANCH_OVERFLOW_MIN_HYSTERESIS = 6
BRANCH_OVERFLOW_RETURN_FORCE = ROUTE_FORCE * 1.20
BRANCH_OVERFLOW_RETURN_SPEED = 20.0 * MOTION_SPEED_MULTIPLIER
BRANCH_OVERFLOW_RETURN_VELOCITY_GAIN = 7.0
BRANCH_OVERFLOW_POLICY_VERSION = "HOSE_SHARE_HYSTERESIS_TO_J0_V2"
UNIFORM_LANE_AXIAL_GAIN = 18.0
UNIFORM_LANE_AXIAL_DAMPING = 12.0
UNIFORM_LANE_MATCH_TOLERANCE = 1.0 * MAP_SCALE
UNIFORM_LANE_FORCE_LIMIT = 100.0 * MOTION_SPEED_MULTIPLIER
UNIFORM_LANE_POLICY_VERSION = "AXIAL_KELVIN_VOIGT_ROWS_V1"
HOSE_CONTINUITY_AXIAL_GAIN = 14.0
HOSE_CONTINUITY_DAMPING = 11.0
HOSE_CONTINUITY_FORCE_LIMIT = 120.0 * MOTION_SPEED_MULTIPLIER
HOSE_CONTINUITY_POLICY_VERSION = "BASE_J0_BRANCH_CONTINUOUS_HOSE_V2"
COHORT_FLOW_POLICY_VERSION = "CONTINUOUS_HOSE_LANE_ROWS_V6"
FLOW_BACKTRACK_FORCE = 46.0 * MOTION_SPEED_MULTIPLIER
FINAL_GATHER_FORCE = 58.0 * MOTION_SPEED_MULTIPLIER
PRESSURE_BACKTRACK_BODY_FORCE = 8.0 * MOTION_SPEED_MULTIPLIER
CENTERING_GAIN = 1.2

MAX_SPEED = 78.0 * MOTION_SPEED_MULTIPLIER
MAX_ACCELERATION = 150.0 * MOTION_SPEED_MULTIPLIER
PRESSURE_PUSH_MAX_SPEED = 42.0 * MOTION_SPEED_MULTIPLIER
FLOW_BACKTRACK_MAX_SPEED = 52.0 * MOTION_SPEED_MULTIPLIER
TRANSFER_JUNCTION_MERGE_SPEED = (
    FLOW_BACKTRACK_MAX_SPEED * 0.72
)
TRANSFER_JUNCTION_MERGE_FORCE_MULTIPLIER = 1.25
TRANSFER_JUNCTION_MERGE_VELOCITY_GAIN = 7.0
TRANSFER_JUNCTION_MERGE_FORCE_LIMIT = (
    280.0 * MOTION_SPEED_MULTIPLIER
)
TRANSFER_PACKED_TARGET_TAPER_START = 0.88
TRANSFER_PACKED_TARGET_TAPER_END = 1.00
TRANSFER_PACKED_TARGET_FEED_FLOOR = 0.00
TRANSFER_CONTINUITY_DEFICIT_FEED_FLOOR = 0.58
TRANSFER_HANDOFF_POLICY_VERSION = (
    "QUOTA_AND_2D_CONTINUITY_OVERRIDE_V3"
)
JUNCTION_BUFFER_AXIAL_SLICES = 4
JUNCTION_BUFFER_MIN_ROBOTS_PER_SLICE = 14
JUNCTION_BUFFER_LATERAL_SLICES = 4
JUNCTION_BUFFER_MIN_LATERAL_COVERAGE = 0.75
# Recruit only the live 2-D Junction deficit plus a small in-flight column.
# Remaining Base robots stop receiving route force automatically once J0 is
# spatially covered; no fixed half/reserve split is imposed.
JUNCTION_BUFFER_TRANSIT_HEADROOM = 12
BASE_SUBSTITUTE_MAX_ACTIVE = 240
BASE_JUNCTION_INTERFACE_TARGET_GAP = (
    SAFE_RADIUS * NORMAL_EQUILIBRIUM_SCALE * 1.05
)
BASE_JUNCTION_INTERFACE_FRONT_DEPTH = float(bottom_rect.height)
BASE_JUNCTION_INTERFACE_AXIAL_SLICES = 8
BASE_JUNCTION_INTERFACE_MIN_ROBOTS_PER_SLICE = 8
BASE_SUBSTITUTE_DRIVE_FLOOR = 0.85
BASE_SUBSTITUTE_RELEASE_FILTER_ALPHA = 0.18
BASE_SUBSTITUTE_MERGE_SPEED = (
    FLOW_BACKTRACK_MAX_SPEED * 0.95
)
BASE_SUBSTITUTE_FORCE_MULTIPLIER = 1.85
BASE_SUBSTITUTE_VELOCITY_GAIN = 9.0
BASE_SUBSTITUTE_FORCE_LIMIT = (
    360.0 * MOTION_SPEED_MULTIPLIER
)
BASE_TO_JUNCTION_FEED_SPEED = FLOW_BACKTRACK_MAX_SPEED * 0.90
BASE_TO_JUNCTION_VELOCITY_GAIN = 8.5
BASE_TO_JUNCTION_HOLD_DAMPING = 8.0
BASE_TO_JUNCTION_FORCE_LIMIT = 330.0 * MOTION_SPEED_MULTIPLIER
BASE_TAIL_REFILL_SPEED = FLOW_BACKTRACK_MAX_SPEED * 0.58
BASE_TAIL_REFILL_VELOCITY_GAIN = 8.0
BASE_TAIL_REFILL_FORCE_MULTIPLIER = 1.55
BASE_TAIL_REFILL_POLICY_VERSION = "J0_SURPLUS_TO_BASE_COVERAGE_V1"
JUNCTION_REFILL_SPEED = FLOW_BACKTRACK_MAX_SPEED * 0.42
JUNCTION_REFILL_FORCE_MULTIPLIER = 1.45
JUNCTION_REFILL_VELOCITY_GAIN = 8.0
JUNCTION_REFILL_POLICY_VERSION = "FILL_ALL_J0_SLICES_BEFORE_BRANCH_V1"
BASE_SUBSTITUTE_POLICY_VERSION = (
    "PREDICTIVE_BASE_J0_BIDIRECTIONAL_REFILL_V12"
)
BASE_TO_JUNCTION_POLICY_VERSION = "ADAPTIVE_BASE_HOSE_TAIL_FEED_V8"
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
INITIAL_INGRESS_TARGET_Y = center_y + 10.0 * MAP_SCALE
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
SATURATION_PACKED_DENSITY_RATIO = max(
    1.25,
    COMPRESSION_RELEASE_DENSITY_RATIO + 0.05,
)
SATURATION_PACKED_OCCUPANCY_RATIO = 0.30
SATURATION_PACKED_LATERAL_COVERAGE_RATIO = 0.45
SATURATION_GEOMETRY_LATERAL_COVERAGE_RATIO = 0.55
SATURATION_PACKED_DWELL_TIME = 0.20
SATURATION_POLICY_VERSION = "UNIFORM_BRANCH_DENSITY_RELEASE_V6"
PRESSURE_START_POLICY_VERSION = (
    "DENSITY_COMPRESSION_ONLY_V2"
)
LONG_BRANCH_FILL_LENGTH_RATIO = 1.50
LONG_BRANCH_FILL_EQUILIBRIUM_SCALE = 2.90

# Do not start Backtracking while the selected Branch still contains the
# visible sparse middle band.  Base/Junction robots continue the ordinary
# FILL_BEHIND_SHEPHERD feed until most longitudinal slices are populated.
BRANCH_CONTINUITY_SLICE_DEPTH = SMOOTHING_LENGTH
BRANCH_CONTINUITY_MIN_ROBOTS_PER_SLICE = 12
BRANCH_CONTINUITY_LATERAL_SLICES = 5
BRANCH_CONTINUITY_MIN_LATERAL_COVERAGE = 0.80
BRANCH_CONTINUITY_FEED_POLICY_VERSION = "BRANCH_2D_SLICE_DEFICIT_FEED_V2"
HANDOFF_SOURCE_CONTINUITY_POLICY_VERSION = "SOURCE_PREFIX_2D_NO_VOID_V2"
BRANCH_CONTINUITY_REQUIRED_SLICE_RATIO = 0.90
BRANCH_CONTINUITY_MAX_DEPTH_GAP = (
    SAFE_RADIUS * LONG_BRANCH_FILL_EQUILIBRIUM_SCALE * 1.35
)
BRANCH_CONTINUITY_DWELL_TIME = 0.28
BRANCH_CONTINUITY_FILL_TIMEOUT = 3.50

# Width-adaptive Shepherd count.  The slots and their blocking plane exist at
# this final coordinate from the first exploration frame.
SHEPHERD_BOUNDARY_WALL_CLEARANCE = float(SHEPHERD_PLACEMENT_STANDOFF)
SHEPHERD_CURTAIN_CLEARANCE = max(ROBOT_RADIUS * 2.5, 6.0 * MAP_SCALE)
# Keep the NORMAL stacking line closer than the dead-end guide edge while
# retaining a visible collision-free buffer.  This relative center-line gap is
# shared by UP, LEFT, and RIGHT regardless of Branch orientation or length.
SHEPHERD_STACK_LINE_GAP = 20.0 * MAP_SCALE
SHEPHERD_STACK_WALL_STANDOFF = (
    SHEPHERD_BOUNDARY_WALL_CLEARANCE
    + SHEPHERD_STACK_LINE_GAP
)
SHEPHERD_FILL_GAP = SHEPHERD_STACK_LINE_GAP
SHEPHERD_CURTAIN_INTERACTION_DEPTH = (
    SHEPHERD_FILL_GAP
    - SHEPHERD_CURTAIN_CLEARANCE
)
SHEPHERD_STACK_GATE_INTERACTION_DEPTH = 8.0 * MAP_SCALE
# The one-sided fill target remains exactly on the fixed stacking guide.
SHEPHERD_STACK_BRAKE_DISTANCE = 20.0 * MAP_SCALE
SHEPHERD_STACK_MIN_APPROACH_SCALE = 0.12
SHEPHERD_STACK_POLICY_VERSION = "CLOSE_UNIFORM_GAP_V7"
SHEPHERD_FILL_REGION_DEPTH = 62.0 * MAP_SCALE
SHEPHERD_PACKED_EQUILIBRIUM_SCALE = 0.46
# The RIGHT corridor is twice as long as the other branches.  Keep the dense
# layer immediately behind the Shepherd unchanged, but let the ordinary body
# behind that layer use a slightly wider equilibrium spacing.  This covers the
# long corridor without increasing the already expensive 680-robot simulation.
# Keep a uniformly wider Base/Junction tail while the quota controller retains
# enough source flow to connect it to the selected Branch.
JUNCTION_TAIL_EQUILIBRIUM_SCALE = 2.90
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
SHEPHERD_CURTAIN_FORCE = 180.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_CURTAIN_MAX_FORCE = 220.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_CURTAIN_VELOCITY_DAMPING = 12.0
SHEPHERD_CURTAIN_RECOVERY_SPEED = 6.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_CURTAIN_DRAW_HALF_WIDTH = 3

# Piston motion: Shepherd boundary advances toward the parent junction.
SHEPHERD_PISTON_SPEED = 16.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_PISTON_MAX_TRAVEL = 42.0 * MAP_SCALE
SHEPHERD_LINE_BACKTRACK_SPEED = 30.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_SPEED_RAMP_TIME = 0.30
SHEPHERD_JUNCTION_DEPTH_TOLERANCE = max(ROBOT_RADIUS, 0.75)
SHEPHERD_JUNCTION_RELEASE_INSET = max(
    ROBOT_RADIUS * 2.5,
    2.0 * MAP_SCALE,
)
SHEPHERD_JUNCTION_RELEASE_SPEED = 12.0 * MOTION_SPEED_MULTIPLIER
SHEPHERD_POLICY_VERSION = "STRONG_DIRECTIONAL_PISTON_V4"
SHEPHERD_PIPELINE_POLICY_VERSION = (
    "ACTIVE_BRANCH_IMMEDIATE_SHEPHERD_FILL_V5"
)
PIPELINE_SOURCE_STRAGGLER_LIMIT = JUNCTION_SWITCH_COUNT
PRE_SHEPHERD_PACK_DWELL_TIME = 0.08
SHEPHERD_PRESSURE_FACTOR = 7.0
VIRTUAL_PRESSURE_RADIUS = 60.0 * MAP_SCALE
VIRTUAL_PRESSURE_FORCE = 260.0
SHEPHERD_VIRTUAL_FORCE_LIMIT = 300.0 * MOTION_SPEED_MULTIPLIER
PRESSURE_RAMP_TIME = 0.35

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
SHOW_COMM_LINKS_DEFAULT = False
SHOW_DENSITY_COLOR_DEFAULT = True
COMM_UPDATE_INTERVAL_FRAMES = 1

# Localization-free K-hop Capture Shepherd (method B)
KHOP_CAPTURE_POLICY_VERSION = "NATURAL_SPREAD_MAXMIN_DFS_V6"
NATURAL_SPREAD_POLICY_VERSION = "MOTION_COHORT_JUNCTION_V1"
KHOP_CAPTURE_DEPTH = 4
DHOP_CLUSTER_DEPTH = KHOP_CAPTURE_DEPTH
KHOP_CAPTURE_REFRESH_TIME = 0.10
KHOP_MESSAGE_HOP_LIMIT = KHOP_CAPTURE_DEPTH + 2
KHOP_MARKER_MESSAGE_HOP_LIMIT = KHOP_CAPTURE_DEPTH * 2
KHOP_MESSAGE_MAX_AGE = 0.35
# A cap only needs to fully cover the corridor cross-section once, not
# several rows deep -- extra rows just mean more robots to capture and
# position before the barrier is functionally ready, slowing down exactly
# the window a gate is supposed to be protecting.
KHOP_CAPTURE_THICKNESS_ROWS = 1
KHOP_CAPTURE_COUNT_MARGIN = 1.20
KHOP_TREE_TARGET_DISTANCE = COMM_SAFE_DISTANCE * 0.58
KHOP_TREE_SPRING_GAIN = 24.0
KHOP_TREE_DAMPING_GAIN = 6.0
KHOP_TREE_FORCE_LIMIT = 150.0 * MOTION_SPEED_MULTIPLIER
KHOP_LEADER_FORCE = 25.0 * MOTION_SPEED_MULTIPLIER
KHOP_SHEPHERD_FORCE = 16.0 * MOTION_SPEED_MULTIPLIER
KHOP_CAP_ROWS = 3
# Must match khop_cap_line_spacing() (defined later, near
# khop_required_shepherd_count): the SPH pair-repulsion equilibrium
# between physical cap bodies has to agree with the slot-attraction
# target spacing, or the two forces fight each other and the line can
# never actually pack as tight as its slots ask for.
KHOP_CAP_EQUILIBRIUM_DISTANCE = ROBOT_RADIUS * 2.30
KHOP_CAP_POSITION_GAIN = 28.0 * MOTION_SPEED_MULTIPLIER
KHOP_CAP_VELOCITY_GAIN = 8.0
KHOP_CAP_FORCE_LIMIT = 220.0 * MOTION_SPEED_MULTIPLIER
KHOP_CAP_ANCHOR_GAIN = 45.0 * MOTION_SPEED_MULTIPLIER
KHOP_CAP_FORM_SPEED = 42.0 * MOTION_SPEED_MULTIPLIER
KHOP_LEADER_TRACK_SPEED = 10.0 * MOTION_SPEED_MULTIPLIER
KHOP_LEADER_CENTER_SPEED_GAIN = 0.60
KHOP_CAP_CENTERING_GAIN = 9.0 * MOTION_SPEED_MULTIPLIER
KHOP_CAP_CENTERING_FORCE_LIMIT = 220.0 * MOTION_SPEED_MULTIPLIER
KHOP_CAP_FORM_TOLERANCE = KHOP_CAP_EQUILIBRIUM_DISTANCE * 1.50
KHOP_CAP_FORM_DWELL = 0.35
KHOP_CAP_FORM_MAX_TIME = 4.00
# Ordinary particles are primarily SPH mass. Leaders and their captured
# Shepherd tree provide the directional seed; this deliberately small term
# only prevents an unassigned particle from stalling at a symmetric split.
KHOP_STREAM_FORCE = 2.5 * MOTION_SPEED_MULTIPLIER
KHOP_RETURN_FORCE = 28.0 * MOTION_SPEED_MULTIPLIER
KHOP_RETURN_DAMPING = 5.0
# A captured cap member (Leader/Shepherd) is only ever a small fraction of
# the swarm (~29 robots sized to the corridor width). Everyone else keeps
# khop_stream_id == None until it is individually captured or a Branch
# stream is assigned to it, which used to mean zero directed route force
# for the whole rest of the body -- raw SPH pressure alone diffuses it
# radially across the open Junction (and partway into every Branch) long
# before any physical cap has finished forming at a single mouth. This
# keeps the uncaptured body one coherent stream behind whichever front is
# currently leading, weak enough that the cap (KHOP_LEADER_FORCE /
# KHOP_SHEPHERD_FORCE, both well above this) always stays ahead of it.
# KHOP_STREAM_FORCE itself (2.5x) turned out too faint to meaningfully
# out-compete raw SPH pressure diffusion once ~700 robots are involved --
# the body still scattered even once this force pointed the right way.
KHOP_UNASSIGNED_FOLLOW_FORCE = 10.0 * MOTION_SPEED_MULTIPLIER
# Once every locally discovered Branch is COMPLETED/BLOCKING there is no more
# Junction left to guard, so every remaining physical role (Gatekeeper,
# Marker, stray Leader/Shepherd/Relay) is released back to NORMAL and the
# whole body is driven home to the fixed Base origin instead of being left
# wherever forward propulsion happened to stop.
KHOP_RETURN_TO_BASE_FORCE = 30.0 * MOTION_SPEED_MULTIPLIER
KHOP_RETURN_TO_BASE_DAMPING = 4.0
KHOP_RETURN_BASE_READY_RATIO = 0.95
KHOP_RETURN_BASE_DWELL = 0.50
KHOP_LOCAL_PROBE_STEP = max(ROBOT_RADIUS * 2.5, 3.0 * MAP_SCALE)
KHOP_LOCAL_PROBE_DISTANCE = COMM_RANGE * 1.55
KHOP_NAVIGATION_CLEARANCE = KHOP_LOCAL_PROBE_DISTANCE * 0.88
# Junction discovery thresholds use only robot motion and the live local mesh.
# Local wall clearance remains available exclusively for collision safety and
# dead-end confirmation after a Branch has already been registered.
NATURAL_SPREAD_LATERAL_ANGLE = math.radians(32.0)
NATURAL_SPREAD_MIN_SPEED = 2.0 * MOTION_SPEED_MULTIPLIER
NATURAL_SPREAD_MIN_TRAVEL = max(ROBOT_RADIUS * 3.0, 4.5 * MAP_SCALE)
# Witness-count anomaly threshold, expressed as a ratio of the locally
# observable population (khop_reachable_robot_count(leader,
# NATURAL_SPREAD_LOCAL_HOPS), see natural_spread_anomaly_threshold()) rather
# than a fixed absolute robot count. An absolute count's meaning silently
# shifts whenever cap/witness-pool sizes change for unrelated reasons (this
# is exactly what happened when the root cap grew from 29 to ~38 members in
# an earlier round: the same raw threshold suddenly meant a much smaller
# fraction of the locally reachable swarm). The ratio below is calibrated to
# reproduce the previously tuned absolute threshold (7) at the locally
# observed population measured at anomaly detection (~483 reachable robots).
NATURAL_SPREAD_ANOMALY_ROBOT_RATIO = 0.0145
NATURAL_SPREAD_ANOMALY_ROBOTS_MIN = 3
NATURAL_SPREAD_ANOMALY_DWELL = 0.22
NATURAL_SPREAD_LOCAL_HOPS = KHOP_CAPTURE_DEPTH + 2
NATURAL_SPREAD_CANDIDATE_TIMEOUT = 5.0
NATURAL_SPREAD_RETRY_COOLDOWN = 1.0
NATURAL_SPREAD_CANDIDATE_SLOW_SCALE = 0.32
NATURAL_SPREAD_LIMIT_RADIUS = COMM_SAFE_DISTANCE * 2.05
NATURAL_SPREAD_LIMIT_GAIN = 18.0 * MOTION_SPEED_MULTIPLIER
NATURAL_SPREAD_LIMIT_FORCE = 65.0 * MOTION_SPEED_MULTIPLIER
NATURAL_SPREAD_MIN_COHORTS = 2
NATURAL_SPREAD_COHORT_MIN_TRAVEL = COMM_SAFE_DISTANCE * 0.48
NATURAL_SPREAD_COHORT_MIN_DWELL = 0.48
NATURAL_SPREAD_COHORT_MAX_VARIANCE = 0.13
NATURAL_SPREAD_COHORT_MIN_CONNECTIVITY = 0.82
NATURAL_SPREAD_COHORT_MIN_BASE_CONNECTIVITY = 0.72
NATURAL_SPREAD_COHORT_MIN_WIDTH = ROBOT_RADIUS * 4.0
NATURAL_SPREAD_COHORT_MAX_ANGLE_STD = math.radians(25.0)
NATURAL_SPREAD_COHORT_MIN_SEPARATION = math.radians(52.0)
# Once >= NATURAL_SPREAD_MIN_COHORTS have validated, do not finalize the
# split immediately if another detected-but-not-yet-valid candidate is
# still accumulating its own dwell -- a straight-through exit's evidence
# converges slower (see detect_khop_straight_continuation_cluster) and
# would otherwise always lose the race to already-mature turning cohorts.
# Bounded well under NATURAL_SPREAD_CANDIDATE_TIMEOUT so an indefinitely
# stuck straggler still cannot block confirmation forever.
NATURAL_SPREAD_STRAGGLER_GRACE_TIME = 1.5
# KHOP_SPLIT_MIN_GROUP_SIZE is a physical/geometric floor -- the fewest
# robots that can plausibly hold a barrier line/gate or a connected capture
# tree, driven by corridor width (see khop_required_shepherd_count), not by
# how large the swarm's locally observable population happens to be. It is
# intentionally left as an absolute constant and used only for those
# physical-viability checks (cap formation completeness, connected-tree
# re-election, retry candidate pool, dead-end member-count failure).
KHOP_SPLIT_MIN_GROUP_SIZE = 18
KHOP_SPLIT_CLUSTER_MATCH_DEGREES = 18.0
KHOP_SPLIT_CLUSTER_MIN_RADIUS = COMM_SAFE_DISTANCE * 0.62
KHOP_SPLIT_CLUSTER_MAX_RADIUS = NATURAL_SPREAD_LIMIT_RADIUS * 1.05
KHOP_SPLIT_CLUSTER_BACKWARD_COSINE = -0.30
# Cluster/cohort-size validity floor (detect_khop_directional_clusters,
# detect_khop_straight_continuation_cluster, validate_directional_cohorts,
# split_khop_from_clusters), unlike KHOP_SPLIT_MIN_GROUP_SIZE above, IS a
# detection-evidence threshold -- "is this cluster big enough to trust as
# real cohort evidence" -- so it is normalized as a ratio of the locally
# observable population (khop_state.last_split_group_size, frozen at the
# moment CONTROLLED_SPREAD began) via khop_split_cluster_min_size() instead
# of the fixed absolute count it used to share with KHOP_SPLIT_MIN_GROUP_SIZE.
# Calibrated to reproduce the previously tuned absolute threshold (18) at
# the locally observed population measured at anomaly detection (~483).
KHOP_SPLIT_CLUSTER_MIN_RATIO = 0.0373
KHOP_SPLIT_CLUSTER_MIN_SIZE_FLOOR = 6
KHOP_STREAM_LATERAL_BAND = COMM_SAFE_DISTANCE * 0.45
KHOP_LEADER_REELECT_DISTANCE = COMM_SAFE_DISTANCE * 0.32
KHOP_LEADER_REELECT_DWELL = 0.35
KHOP_FRONT_NEIGHBOR_CONE_DEGREES = 55.0
KHOP_DEAD_END_CLEARANCE = max(ROBOT_RADIUS * 3.0, 6.0 * MAP_SCALE)
# These three must actually mean "a real queue has backed up behind the
# cap," not just "the Leader is touching a wall": with the old thresholds
# (density 0.25 / pressure 0.006 / contact 0.075) ordinary ambient density
# right after the front merely arrives already clears all three, so
# Pressure Push effectively started on a clearance+dwell timer regardless
# of how much was actually queued up. Raised to require density/pressure/
# contact compression that only build up once robots are genuinely packed
# in behind the cap.
KHOP_DEAD_END_DENSITY_RATIO = 0.70
KHOP_DEAD_END_MAX_SPEED = 7.0
KHOP_DEAD_END_MIN_PRESSURE_RATIO = 0.05
KHOP_DEAD_END_MIN_CONTACT_COMPRESSION = 0.28
KHOP_DEAD_END_DWELL_TIME = 0.80
KHOP_GROUP_FAILURE_DWELL_TIME = 0.80
KHOP_RETRY_DELAY = 0.50
KHOP_MAX_RETRIES = 2
KHOP_MARKER_RETURN_RADIUS = COMM_SAFE_DISTANCE * 0.85
# Keep the Pebble on the Junction side of a staged cap; the cap Leader waits
# farther outward at KHOP_BRANCH_STAGING_OFFSET. Both must clear the open
# Junction cross-section itself, not just sit a short step from the split
# point -- a small offset leaves the Marker/cap still inside the wide open
# area shared by every Branch, where nothing narrows the passage and the
# cap cannot act as a chokepoint. Sized generously past this map's
# Junction opening so the Marker/cap land inside the confined Branch
# corridor proper.
KHOP_MARKER_OFFSET = COMM_SAFE_DISTANCE * 2.20
KHOP_BRANCH_STAGING_OFFSET = COMM_SAFE_DISTANCE * 3.40
KHOP_GATEKEEPER_ROWS = 2
KHOP_GATEKEEPER_ROW_GAP = max(ROBOT_RADIUS * 2.4, 3.5 * MAP_SCALE)
KHOP_GATEKEEPER_FORCE_RADIUS = COMM_SAFE_DISTANCE * 0.70
KHOP_GATEKEEPER_FORCE = 125.0 * MOTION_SPEED_MULTIPLIER
KHOP_GATEKEEPER_FORCE_LIMIT = 220.0 * MOTION_SPEED_MULTIPLIER
KHOP_STATIONARY_TOLERANCE = 2.5 * MAP_SCALE
KHOP_STATIONARY_MOVE_SPEED = 70.0 * MOTION_SPEED_MULTIPLIER
KHOP_DYNAMIC_ROLES = {"NORMAL", "KHOP_LEADER", "KHOP_SHEPHERD"}
KHOP_EVIDENCE_FILTER_TIME = 0.30
KHOP_MERGE_HEADING_DEGREES = 24.0
KHOP_MERGE_RANGE = COMM_SAFE_DISTANCE * 0.90
KHOP_MERGE_DWELL_TIME = 0.45
KHOP_COMPLETION_CONSENSUS_DWELL = 0.35
KHOP_RELAY_DEPLOY_MARGIN = COMM_RANGE * 0.16
KHOP_RELAY_RELEASE_MARGIN = COMM_RANGE * 0.30
KHOP_RELAY_COOLDOWN = 0.45
KHOP_RELAY_MIN_FRONT_GAP = COMM_SAFE_DISTANCE * 0.72
KHOP_RELAY_MAX_PER_GROUP = 5
KHOP_SPH_EQUILIBRIUM_DISTANCE = max(
    ROBOT_RADIUS * 2.2,
    SMOOTHING_LENGTH * 0.70,
)
KHOP_SPH_ATTRACTION_GAIN = 18.0 * MOTION_SPEED_MULTIPLIER
KHOP_SPH_ATTRACTION_LIMIT = 55.0 * MOTION_SPEED_MULTIPLIER
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

# Geodesic EDF guidance and smooth virtual valves at unselected branch mouths.
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
FINAL_GATE_POLICY_VERSION = "ONE_WAY_RETURN_SEAL_V1"

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
    if KHOP_CAPTURE_ENABLED:
        # The collision mask represents real walls.  No Branch is made
        # impassable from a controller-side gate state in method B.
        return region != "OUTSIDE"
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
        or base_reserve_is_mobile(robot)
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


def base_reserve_is_mobile(robot: "Robot") -> bool:
    """Return whether a held Base particle is part of the live hose refill.

    Reserve is a state-dependent supply, not an immovable caste.  Selection by
    either the ordinary fill controller or the cross-Branch substitute
    controller temporarily overrides the Base positional hold and wall clamp.
    """
    return (
        not robot.base_reserve
        or robot.robot_id in base_junction_feed_robot_ids
        or robot.robot_id in base_substitute_robot_ids
    )


def constrain_final_return_gate_crossing(
    robot: "Robot",
    old_position: pygame.Vector2,
) -> None:
    """One-way hard gate: branch stragglers may exit, nobody may re-enter."""
    if phase not in {
        SimulationPhase.FINAL_JUNCTION_GATHER,
        SimulationPhase.RETURN_TO_BASE,
    }:
        return
    old_region = get_robot_region(old_position)
    new_region = get_robot_region(robot.position)
    if (
        old_region not in {"JUNCTION", "BOTTOM"}
        or new_region not in BRANCHES
        or branch_gate_states.get(new_region) != "CLOSED"
    ):
        return

    robot.position = old_position.copy()
    outward_direction = BRANCH_DIRECTIONS[new_region]
    outward_speed = robot.velocity.dot(outward_direction)
    if outward_speed > 0.0:
        robot.velocity -= outward_direction * outward_speed
    robot.acceleration.update(0.0, 0.0)
    robot.filtered_acceleration.update(0.0, 0.0)


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
        SimulationPhase.EXPLORE_BRANCH,
        SimulationPhase.FORM_SHEPHERD_BOUNDARY,
        SimulationPhase.FILL_BEHIND_SHEPHERD,
        SimulationPhase.PRESSURE_PUSH,
        SimulationPhase.FLOW_BACKTRACK,
    }


def shepherd_stack_staging_active() -> bool:
    return phase in {
        SimulationPhase.EXPLORE_BRANCH,
        SimulationPhase.FORM_SHEPHERD_BOUNDARY,
        SimulationPhase.FILL_BEHIND_SHEPHERD,
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
    if shepherd_stack_staging_active():
        return max(
            0.0,
            get_shepherd_boundary_depth(branch)
            - SHEPHERD_FILL_GAP,
        )
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

    The geometric lattice target is a lower bound only.  With a large physical
    equilibrium it is deliberately sparse and used alone would stop Base feed
    while the Base/J0/Branch hose is visibly disconnected.  A length-aware
    share of all NORMAL fluid therefore supplies the compressible Branch body,
    while the remaining share stays available for Junction and Base continuity.
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
        branch_depth_from_junction(
            get_shepherd_fill_target(branch),
            branch,
        ),
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
    normal_fluid_count = sum(robot.role == "NORMAL" for robot in robots)
    hose_share = (
        LONG_BRANCH_HOSE_TARGET_SHARE
        if BRANCH_LENGTHS[branch]
        >= normal_length * LONG_BRANCH_FILL_LENGTH_RATIO
        else NORMAL_BRANCH_HOSE_TARGET_SHARE
    )
    hose_target = math.ceil(normal_fluid_count * hose_share)
    reserve_count = max(
        JUNCTION_SWITCH_COUNT,
        lane_count * BRANCH_FILL_QUOTA_MIN_RESERVE_ROWS,
    )
    return int(
        clamp(
            max(geometric_target, hose_target),
            1,
            max(1, normal_fluid_count - reserve_count),
        )
    )


def branch_quota_source_feed_scale(region: str) -> float:
    """Throttle only Junction-to-Branch admission after quota is met."""
    deficit = smoothstep01(branch_fill_deficit_control)
    del region
    floor = BRANCH_FILL_QUOTA_JUNCTION_FEED_FLOOR
    maximum = 1.0 + BRANCH_FILL_QUOTA_SOURCE_FORCE_BOOST
    return floor + (maximum - floor) * deficit


def handoff_target_feed_scale() -> float:
    """Cap the next Branch at quota and leave surplus fluid in Junction.

    The former pre-pack-ready guard allowed hundreds of source robots into a
    short target Branch before its density trigger became ready.  Quota control
    now applies from the first transfer frame; density still decides when the
    Shepherd piston may start, but never how many robots may enter the Branch.
    """
    if (
        phase not in {
            SimulationPhase.PRESSURE_PUSH,
            SimulationPhase.FLOW_BACKTRACK,
        }
        or transfer_branch is None
        or branch_fill_target_count <= 0
    ):
        return 1.0
    fill_ratio = (
        branch_fill_current_count
        / max(branch_fill_target_count, 1)
    )
    taper = smoothstep01(
        (
            fill_ratio
            - TRANSFER_PACKED_TARGET_TAPER_START
        )
        / max(
            (
                TRANSFER_PACKED_TARGET_TAPER_END
                - TRANSFER_PACKED_TARGET_TAPER_START
            ),
            EPSILON,
        )
    )
    quota_scale = (
        1.0
        - (1.0 - TRANSFER_PACKED_TARGET_FEED_FLOOR)
        * taper
    )
    if branch_feed_continuity_deficit <= 0:
        return quota_scale
    slice_capacity = max(
        1,
        len(branch_feed_slice_counts)
        * BRANCH_CONTINUITY_MIN_ROBOTS_PER_SLICE,
    )
    continuity_need = clamp(
        branch_feed_continuity_deficit / slice_capacity,
        0.0,
        1.0,
    )
    continuity_scale = (
        TRANSFER_CONTINUITY_DEFICIT_FEED_FLOOR
        + (
            1.0 - TRANSFER_CONTINUITY_DEFICIT_FEED_FLOOR
        )
        * smoothstep01(continuity_need)
    )
    return max(quota_scale, continuity_scale)


def get_branch_fill_quota_branch() -> str:
    if (
        phase in {
            SimulationPhase.PRESSURE_PUSH,
            SimulationPhase.FLOW_BACKTRACK,
        }
        and transfer_branch is not None
        and not final_base_transfer_active
    ):
        return transfer_branch
    return active_branch


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
    interaction_depth = (
        SHEPHERD_STACK_GATE_INTERACTION_DEPTH
        if shepherd_stack_staging_active()
        else SHEPHERD_CURTAIN_INTERACTION_DEPTH
    )
    activation_depth = limit_depth - interaction_depth
    if depth <= activation_depth:
        return pygame.Vector2()

    ratio = clamp(
        (depth - activation_depth)
        / max(interaction_depth, EPSILON),
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
    if not shepherd_stack_staging_active():
        robot.velocity += get_backtrack_direction(active_branch) * min(
            SHEPHERD_CURTAIN_RECOVERY_SPEED,
            penetration * 2.0,
        )


def enforce_shepherd_curtain_for_swarm(robots) -> None:
    for robot in robots:
        constrain_normal_behind_shepherd_curtain(robot)


def get_pre_shepherd_staging_branch() -> Optional[str]:
    """Return the next Branch whose slot line is already pre-armed."""
    branch = (
        pre_shepherd_branch
        if pre_shepherd_branch is not None
        else transfer_branch
    )
    if (
        phase not in {
            SimulationPhase.PRESSURE_PUSH,
            SimulationPhase.FLOW_BACKTRACK,
        }
        or branch is None
        or branch == active_branch
    ):
        return None
    return branch


def pre_shepherd_curtain_active() -> bool:
    return get_pre_shepherd_staging_branch() is not None


def get_pre_shepherd_normal_limit_depth(branch: str) -> float:
    return max(
        0.0,
        get_shepherd_boundary_depth(branch)
        - SHEPHERD_FILL_GAP,
    )


def compute_pre_shepherd_curtain_force(
    robot: "Robot",
) -> pygame.Vector2:
    """Static full-width shield formed early in the transfer branch."""
    branch = get_pre_shepherd_staging_branch()
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
        limit_depth - SHEPHERD_STACK_GATE_INTERACTION_DEPTH
    )
    if depth <= activation_depth:
        return pygame.Vector2()
    ratio = clamp(
        (depth - activation_depth)
        / max(SHEPHERD_STACK_GATE_INTERACTION_DEPTH, EPSILON),
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
    branch = get_pre_shepherd_staging_branch()
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
    # During slot staging, clamp at the fixed stack line without injecting a
    # reverse Junction velocity.  Route pressure can therefore build layers
    # in place instead of producing the former back-and-forth shuttle.


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
    khop_leader_distance = sum(
        robot.distance_by_role.get("KHOP_LEADER", 0.0)
        for robot in robots
    )
    khop_shepherd_distance = sum(
        robot.distance_by_role.get("KHOP_SHEPHERD", 0.0)
        for robot in robots
    )
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
        writer.writerow(["khop_policy", KHOP_CAPTURE_POLICY_VERSION])
        writer.writerow(["junction_discovery_policy", NATURAL_SPREAD_POLICY_VERSION])
        writer.writerow(["space_recognition_state", khop_state.recognition_state.name])
        writer.writerow(["forward_open_angle_pre_detection", "DISABLED"])
        writer.writerow(["khop_capture_depth", KHOP_CAPTURE_DEPTH])
        writer.writerow(["dhop_cluster_depth", DHOP_CLUSTER_DEPTH])
        writer.writerow([
            "dhop_sequential_visit_order",
            " > ".join(khop_state.sequential_visit_order),
        ])
        writer.writerow(["khop_stage", khop_state.stage])
        writer.writerow(["khop_consensus_sequence", khop_state.consensus_sequence])
        writer.writerow(["khop_message_deliveries", khop_state.message_delivery_count])
        writer.writerow([
            "khop_active_relays",
            sum(robot.role == "RELAY" for robot in robots),
        ])
        writer.writerow(["khop_leader_distance", f"{khop_leader_distance:.6f}"])
        writer.writerow(["khop_shepherd_distance", f"{khop_shepherd_distance:.6f}"])
        writer.writerow(["khop_event_count", len(khop_state.events)])
        for group in sorted(khop_branch_groups(), key=lambda item: item.rollout_rank):
            writer.writerow([
                f"cohort_rollout_{group.group_id}",
                f"rank={group.rollout_rank}; cost={group.rollout_cost:.6f}",
            ])
        for index, event in enumerate(khop_state.events, start=1):
            writer.writerow([
                f"khop_event_{index}",
                "; ".join(f"{key}={value}" for key, value in event.items()),
            ])
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
        self.last_base_piston_force = 0.0
        self.last_edf_force = 0.0
        self.role = "NORMAL"

        # K-hop Capture Shepherd state.  ``khop_stream_id`` assigns ordinary
        # fluid robots to a locally discovered outgoing stream, while
        # ``khop_group_id`` is reserved for robots currently captured in the
        # Leader's K-hop tree.
        self.khop_stream_id: Optional[str] = None
        self.khop_group_id: Optional[str] = None
        self.khop_parent_id: Optional[int] = None
        self.khop_hop = -1
        self.khop_heading = pygame.Vector2(0.0, -1.0)
        self.khop_stationary_target: Optional[pygame.Vector2] = None
        self.marker_group_id: Optional[str] = None
        self.khop_inbox: list[KHopMessage] = []
        self.khop_tx_sequence = 0
        self.khop_velocity_ema = pygame.Vector2()
        self.khop_neighbor_count_ema = 0.0
        self.khop_last_marker_id: Optional[str] = None
        self.dhop_clusterhead_id: Optional[int] = None
        self.dhop_parent_id: Optional[int] = None
        self.dhop_gateway = False
        self.dhop_selection_rule = 0
        self.dhop_staging_group_id: Optional[str] = None

        self.shepherd_anchor: Optional[pygame.Vector2] = None
        self.shepherd_origin: Optional[pygame.Vector2] = None
        self.shepherd_branch: Optional[str] = None
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
            "KHOP_LEADER": 0.0,
            "KHOP_SHEPHERD": 0.0,
            "MARKER": 0.0,
            "GATEKEEPER": 0.0,
        }

    def _record_motion(self):
        distance = self.position.distance_to(self.previous_position)
        self.total_distance += distance
        self.distance_by_role[self.role] = self.distance_by_role.get(self.role, 0.0) + distance
        self.previous_position = self.position.copy()

    def update(self, dt: float):
        old_position = self.position.copy()

        if (
            KHOP_CAPTURE_ENABLED
            and self.role in {"KHOP_LEADER", "KHOP_SHEPHERD"}
        ):
            group = khop_state.groups.get(self.khop_group_id or "")
            if (
                group is not None
                and self.role == "KHOP_LEADER"
                and group.state == KHopGroupState.EXPLORING
            ):
                forward = group.heading.normalize()
                side = pygame.Vector2(-forward.y, forward.x)
                side_probe = COMM_RANGE * 1.25
                left_clearance = khop_local_clearance(
                    self.position,
                    -side,
                    side_probe,
                )
                right_clearance = khop_local_clearance(
                    self.position,
                    side,
                    side_probe,
                )
                sensed_width = khop_estimate_corridor_width(group)
                enclosed_cross_section = (
                    left_clearance + right_clearance
                    < sensed_width * 0.90
                )
                lateral_speed = (
                    clamp(
                        (right_clearance - left_clearance)
                        * KHOP_LEADER_CENTER_SPEED_GAIN,
                        -KHOP_LEADER_TRACK_SPEED * 0.80,
                        KHOP_LEADER_TRACK_SPEED * 0.80,
                    )
                    if enclosed_cross_section
                    else 0.0
                )
                forward_clearance = khop_local_clearance(
                    self.position,
                    forward,
                    KHOP_LOCAL_PROBE_DISTANCE,
                )
                forward_speed = (
                    0.0
                    if forward_clearance <= KHOP_DEAD_END_CLEARANCE * 1.15
                    else KHOP_LEADER_TRACK_SPEED
                )
                if (
                    khop_state.stage == "CONTROLLED_SPREAD"
                    and group.group_id == khop_state.root_group_id
                ):
                    # Stage 3 of the sketch: the original cap stays at the
                    # Junction while ordinary mass spreads and the new local
                    # components elect their own Leaders.
                    forward_speed = 0.0
                desired_velocity = (
                    forward * forward_speed
                    + side * lateral_speed
                )
                x_position = pygame.Vector2(
                    self.position.x + desired_velocity.x * dt,
                    self.position.y,
                )
                if is_walkable(x_position, self.radius):
                    self.position.x = x_position.x
                y_position = pygame.Vector2(
                    self.position.x,
                    self.position.y + desired_velocity.y * dt,
                )
                if is_walkable(y_position, self.radius):
                    self.position.y = y_position.y
                self.velocity = (self.position - old_position) / max(dt, EPSILON)
                self.acceleration.update(0.0, 0.0)
                self.filtered_acceleration.update(0.0, 0.0)
                self.previous_position = old_position
                self._record_motion()
                return
            cap_tracking_state = (
                group is not None
                and (
                    group.state in {
                        KHopGroupState.FORMING,
                        KHopGroupState.WAITING,
                    }
                    or self.role == "KHOP_SHEPHERD"
                    and group.state in {
                        KHopGroupState.EXPLORING,
                        KHopGroupState.DEAD_END_CONFIRMED,
                        KHopGroupState.RETURNING,
                        KHopGroupState.MERGING,
                    }
                )
            )
            if cap_tracking_state:
                if self.role == "KHOP_LEADER":
                    target = group.cap_anchor
                else:
                    slot = group.cap_slot_by_robot.get(self.robot_id)
                    target = (
                        khop_cap_world_target(group, slot)
                        if slot is not None
                        else None
                    )
                if target is not None:
                    error = target - self.position
                    step = KHOP_CAP_FORM_SPEED * dt
                    next_position = (
                        target.copy()
                        if error.length() <= step
                        else self.position + error.normalize() * step
                    ) if error.length_squared() > EPSILON else self.position.copy()
                    if is_walkable(next_position, self.radius):
                        self.position = next_position
                if group.state in {
                    KHopGroupState.FORMING,
                    KHopGroupState.WAITING,
                } or dt <= EPSILON:
                    self.velocity.update(0.0, 0.0)
                else:
                    self.velocity = (self.position - old_position) / dt
                self.acceleration.update(0.0, 0.0)
                self.filtered_acceleration.update(0.0, 0.0)
                self.previous_position = old_position
                self._record_motion()
                return

        if (
            KHOP_CAPTURE_ENABLED
            and self.role in {"MARKER", "GATEKEEPER"}
            and self.khop_stationary_target is not None
        ):
            error = self.khop_stationary_target - self.position
            step = KHOP_STATIONARY_MOVE_SPEED * dt
            next_position = (
                self.khop_stationary_target.copy()
                if error.length() <= step
                else self.position + error.normalize() * step
            ) if error.length_squared() > EPSILON else self.position.copy()
            if is_walkable(next_position, self.radius):
                self.position = next_position
            if error.length() <= KHOP_STATIONARY_TOLERANCE:
                self.position = self.khop_stationary_target.copy()
            self.velocity.update(0.0, 0.0)
            self.acceleration.update(0.0, 0.0)
            self.filtered_acceleration.update(0.0, 0.0)
            self.previous_position = old_position
            self._record_motion()
            return

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

        if (
            self.role in {
                "PRE_SHEPHERD",
                "SHEPHERD_CANDIDATE",
            }
            and self.shepherd_anchor is not None
        ):
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
        if self.role == "NORMAL" and not KHOP_CAPTURE_ENABLED:
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
        if not KHOP_CAPTURE_ENABLED:
            apply_communication_velocity_guard(self, dt)
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
        if not KHOP_CAPTURE_ENABLED:
            constrain_communication_parent_separation(self, old_position)
        self.acceleration.update(0.0, 0.0)
        self.previous_position = old_position
        self._record_motion()

    def draw(self, surface, color_reference_density, show_density_color):
        x, y = round(self.position.x), round(self.position.y)
        if self.role == "KHOP_LEADER":
            color = KHOP_LEADER_COLOR
        elif self.role == "MARKER":
            color = (
                MARKER_PENDING_COLOR
                if khop_marker_state_for_robot(self.robot_id)
                in {MarkerState.UNSET, MarkerState.PENDING}
                else MARKER_COMPLETE_COLOR
            )
        elif self.role == "GATEKEEPER":
            color = GATEKEEPER_COLOR
        elif self.role == "ANCHOR":
            color = ANCHOR_COLOR
        elif self.role == "TRUNK_RELAY":
            color = TRUNK_RELAY_COLOR
        elif self.role == "RELAY":
            color = RELAY_COLOR
        elif self.role in {
            "SHEPHERD",
            "PRE_SHEPHERD",
            "SHEPHERD_CANDIDATE",
            "KHOP_SHEPHERD",
        }:
            color = SHEPHERD_COLOR
        elif self.comm_bridge_target is not None:
            color = COMM_BRIDGE_COLOR
        elif (
            KHOP_CAPTURE_ENABLED
            and self.role == "NORMAL"
            and self.khop_stream_id == khop_state.active_group_id
        ):
            color = DHOP_ACTIVE_STREAM_COLOR
        elif show_density_color:
            color = density_to_color(self.density, color_reference_density)
        else:
            color = ROBOT_BASE_COLOR

        if (
            not KHOP_CAPTURE_ENABLED
            and base_station is not None
            and self.role != "BASE"
            and not self.connected_to_base
        ):
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

        if self.role == "KHOP_LEADER":
            pygame.draw.circle(
                surface,
                KHOP_LEADER_COLOR,
                (x, y),
                draw_radius + 4,
                width=2,
            )
        elif self.role in {"MARKER", "GATEKEEPER"}:
            pygame.draw.circle(
                surface,
                color,
                (x, y),
                draw_radius + 3,
                width=2,
            )
        elif self.role in {"RELAY", "TRUNK_RELAY"}:
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
        elif self.role in {
            "SHEPHERD",
            "PRE_SHEPHERD",
            "SHEPHERD_CANDIDATE",
            "KHOP_SHEPHERD",
        }:
            pygame.draw.circle(surface, SHEPHERD_COLOR, (x, y), draw_radius + 2, width=1)
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
    if KHOP_CAPTURE_ENABLED:
        return (
            "LOCAL_KHOP",
            khop_state.consensus_sequence,
            None,
            f"KHOP_{khop_state.stage}",
            None,
        )
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
    if KHOP_CAPTURE_ENABLED:
        update_khop_local_messages(robots)


# =========================================================
# 10A. Localization-free K-hop Capture Shepherd controller
# =========================================================


def khop_ema(previous: float, sample: float, dt: float, time_constant: float) -> float:
    """Time-step invariant exponential filter for noisy local evidence."""
    alpha = 1.0 - math.exp(-max(dt, 0.0) / max(time_constant, EPSILON))
    return previous + (sample - previous) * alpha


def update_khop_local_motion_estimates(
    robots: list["Robot"],
    dt: float,
) -> None:
    """Maintain each robot's own velocity filter, independent of group roles."""
    alpha = 1.0 - math.exp(
        -max(dt, 0.0) / max(KHOP_EVIDENCE_FILTER_TIME, EPSILON)
    )
    for robot in robots:
        robot.khop_velocity_ema = robot.khop_velocity_ema.lerp(
            robot.velocity,
            alpha,
        )


def khop_messages(
    robot: "Robot",
    kind: Optional[KHopMessageKind] = None,
    group_id: Optional[str] = None,
) -> list[KHopMessage]:
    """Read only fresh packets actually delivered to this robot."""
    return [
        message
        for message in robot.khop_inbox
        if simulation_time - message.created_at <= KHOP_MESSAGE_MAX_AGE
        and (kind is None or message.kind == kind)
        and (group_id is None or message.group_id == group_id)
    ]


def khop_build_source_messages(robots: list["Robot"]) -> list[KHopMessage]:
    """Create one local status packet per active Leader, Marker, and Relay."""
    packets: list[KHopMessage] = []
    for group in khop_state.groups.values():
        leader = khop_state.robot_by_id.get(group.leader_id)
        if leader is None or leader.role != "KHOP_LEADER":
            continue
        leader.khop_tx_sequence += 1
        packets.append(KHopMessage(
            kind=KHopMessageKind.LEADER_STATUS,
            source_id=leader.robot_id,
            source_sequence=leader.khop_tx_sequence,
            group_id=group.group_id,
            hop_count=0,
            hop_limit=KHOP_MESSAGE_HOP_LIMIT,
            created_at=simulation_time,
            state=group.state.name,
            heading=(group.heading.x, group.heading.y),
            marker_id=group.marker_id,
            dead_end_confirmed=group.dead_end_confirmed,
            pressure_ratio=group.front_pressure_ratio,
            density_ratio=group.front_density_ratio,
            compression_ratio=group.contact_compression_ratio,
            connectivity_ratio=group.connectivity_ratio,
        ))

    for marker in khop_state.markers.values():
        marker_robot = khop_state.robot_by_id.get(marker.robot_id)
        group = khop_state.groups.get(marker.group_id)
        if marker_robot is None or group is None:
            continue
        marker_robot.khop_tx_sequence += 1
        marker.last_broadcast_at = simulation_time
        packets.append(KHopMessage(
            kind=KHopMessageKind.MARKER_STATUS,
            source_id=marker_robot.robot_id,
            source_sequence=marker_robot.khop_tx_sequence,
            group_id=marker.group_id,
            hop_count=0,
            hop_limit=KHOP_MARKER_MESSAGE_HOP_LIMIT,
            created_at=simulation_time,
            state=marker.state.name,
            heading=(group.heading.x, group.heading.y),
            marker_id=marker.marker_id,
            marker_state=marker.state,
            dead_end_confirmed=marker.dead_end_confirmed,
        ))

    for relay in robots:
        if relay.role != "RELAY" or relay.marker_group_id is None:
            continue
        relay.khop_tx_sequence += 1
        packets.append(KHopMessage(
            kind=KHopMessageKind.RELAY_BEACON,
            source_id=relay.robot_id,
            source_sequence=relay.khop_tx_sequence,
            group_id=relay.marker_group_id,
            hop_count=0,
            hop_limit=KHOP_MESSAGE_HOP_LIMIT,
            created_at=simulation_time,
            state="RELAY",
            connectivity_ratio=1.0 if relay.connected_to_base else 0.0,
        ))
    return packets


def update_khop_local_messages(robots: list["Robot"]) -> None:
    """Flood bounded packets strictly over this frame's local neighbor graph.

    ``khop_state`` stores copies for experiment observation only. Controller
    decisions below read each robot's inbox, never an observer-created route.
    """
    for robot in robots:
        robot.khop_inbox = []
    if khop_state.stage == "UNINITIALIZED":
        return
    khop_state.robot_by_id = {robot.robot_id: robot for robot in robots}
    delivered = 0
    for packet in khop_build_source_messages(robots):
        source = khop_state.robot_by_id.get(packet.source_id)
        if source is None:
            continue
        queue: deque[tuple["Robot", KHopMessage]] = deque([(source, packet)])
        visited = {source.robot_id}
        while queue:
            receiver, received_packet = queue.popleft()
            receiver.khop_inbox.append(received_packet)
            delivered += 1
            if received_packet.hop_count >= received_packet.hop_limit:
                continue
            for neighbor in receiver.comm_neighbors:
                neighbor_id = getattr(neighbor, "robot_id", -1)
                if neighbor_id < 0 or neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                queue.append((
                    neighbor,
                    received_packet.forwarded(receiver.robot_id),
                ))
    khop_state.message_delivery_count += delivered


def khop_marker_packet(
    robot: "Robot",
    marker_id: Optional[str],
) -> Optional[KHopMessage]:
    if marker_id is None:
        return None
    candidates = [
        message
        for message in khop_messages(robot, KHopMessageKind.MARKER_STATUS)
        if message.marker_id == marker_id
    ]
    return min(candidates, key=lambda item: item.hop_count, default=None)


def khop_estimate_corridor_width(
    group: Optional[KHopShepherdGroup],
) -> float:
    """Estimate width from two robot-local wall rays, retaining the last value."""
    fallback = COMM_SAFE_DISTANCE * 1.85
    if group is None:
        return fallback
    leader = khop_state.robot_by_id.get(group.leader_id)
    if leader is None or group.heading.length_squared() <= EPSILON:
        return group.estimated_width or fallback
    side = pygame.Vector2(-group.heading.y, group.heading.x).normalize()
    # The old ray was shorter than half of this experiment's corridor, so a
    # centered Leader never saw either wall and always used the narrow
    # fallback.  A cap must measure both walls before choosing its row width.
    maximum = COMM_RANGE * 1.45
    left = khop_local_clearance(leader.position, -side, maximum)
    right = khop_local_clearance(leader.position, side, maximum)
    if left < maximum - KHOP_LOCAL_PROBE_STEP and right < maximum - KHOP_LOCAL_PROBE_STEP:
        measured = left + right + ROBOT_RADIUS * 2.0
        measured = clamp(
            measured,
            ROBOT_RADIUS * 6.0,
            COMM_RANGE * 2.0,
        )
        previous = group.estimated_width or fallback
        # A Leader brushing a wall or sitting at a closed end sees one nearly
        # zero side ray. That must not collapse a previously measured corridor
        # width and under-recruit the physical gate.
        if measured >= previous * 0.72:
            group.estimated_width = previous * 0.80 + measured * 0.20
    return group.estimated_width or fallback


def khop_cap_line_spacing() -> float:
    """Canonical slot-to-slot spacing for a single-row cap/gate line.

    Shared by khop_required_shepherd_count (how many robots a full-width
    line needs) and khop_cap_slot_offsets (where those slots actually sit),
    so the recruited headcount always matches what one tightly packed row
    can hold -- a mismatch here is what silently reintroduces extra rows
    even when KHOP_CAPTURE_THICKNESS_ROWS == 1.

    Deliberately near the physical minimum (just enough for two robot
    bodies to sit side by side without overlapping), not the much wider
    NORMAL_EQUILIBRIUM_SCALE spacing the general fluid body flows at --
    a gate line needs to be dense enough that nothing can fit through a
    gap between adjacent members, not comfortably spaced.
    """
    return ROBOT_RADIUS * 2.30


def khop_required_shepherd_count(
    group: Optional[KHopShepherdGroup] = None,
) -> int:
    """Estimate a capture group sized to one tightly packed cap/gate line."""
    robots_across = math.ceil(
        khop_estimate_corridor_width(group)
        / max(khop_cap_line_spacing(), EPSILON)
    ) + 1
    return max(
        KHOP_SPLIT_MIN_GROUP_SIZE,
        math.ceil(
            robots_across
            * KHOP_CAPTURE_THICKNESS_ROWS
            * KHOP_CAPTURE_COUNT_MARGIN
        ),
    )


def khop_cap_slot_offsets(
    group: KHopShepherdGroup,
    shepherd_count: int,
) -> list[tuple[float, float]]:
    """Build straight, corridor-width barrier rows in Leader-local
    (side, forward) axes.

    A tapered/crescent cap leaves both edges of the corridor uncovered near
    the walls, so the main body slips around it instead of being blocked --
    not something a real physical robot line could pass off as a gate. Every
    row here spans the full sensed corridor width evenly (spreading fewer
    robots wider rather than clustering them at the center) and rows stack
    directly behind one another at a fixed depth step, matching a straight
    Shepherd fence a real robot line could actually form.
    """
    if shepherd_count <= 0:
        return []
    spacing = khop_cap_line_spacing()
    half_width = max(
        spacing * 0.5,
        (khop_estimate_corridor_width(group) - ROBOT_RADIUS * 4.0) * 0.5,
    )
    per_row = max(2, math.floor(2.0 * half_width / spacing) + 1)
    rows = max(1, math.ceil(shepherd_count / per_row))
    slots: list[tuple[float, float]] = []
    remaining = shepherd_count
    for row_index in range(rows):
        count = min(per_row, remaining)
        if count <= 0:
            break
        remaining -= count
        lateral_values = (
            [0.0]
            if count == 1
            else [
                -half_width + 2.0 * half_width * index / (count - 1)
                for index in range(count)
            ]
        )
        # Rows stack straight behind the front one; none of them taper
        # inward, so the barrier's lateral reach never shrinks with depth.
        depth = spacing * (0.72 + row_index * 0.85)
        for lateral in lateral_values:
            slots.append((lateral, -depth))
    return slots


def khop_cap_world_target(
    group: KHopShepherdGroup,
    slot: tuple[float, float],
) -> Optional[pygame.Vector2]:
    leader = khop_state.robot_by_id.get(group.leader_id)
    if leader is None or group.heading.length_squared() <= EPSILON:
        return None
    forward = group.heading.normalize()
    side = pygame.Vector2(-forward.y, forward.x)
    lateral, axial = slot
    # A Leader can be off-center while turning into a branch. Contract only
    # the lateral coordinate until the local collision mask confirms that the
    # requested physical slot lies inside the corridor.
    for lateral_scale in (1.0, 0.86, 0.72, 0.58, 0.44, 0.30, 0.0):
        target = (
            leader.position
            + side * lateral * lateral_scale
            + forward * axial
        )
        if is_mask_clear_at(target, ROBOT_RADIUS * 1.35):
            return target
    return leader.position + forward * axial


def assign_khop_cap_slots(group: KHopShepherdGroup) -> None:
    """Keep Shepherd IDs on stable geometric slots around the live Leader."""
    member_ids = sorted(group.member_robot_ids - {group.leader_id})
    slots = khop_cap_slot_offsets(group, len(member_ids))
    old_slots = group.cap_slot_by_robot
    assigned: dict[int, tuple[float, float]] = {}
    remaining_slots = list(slots)
    slot_reuse_limit = max(
        ROBOT_RADIUS * 3.0,
        KHOP_CAP_EQUILIBRIUM_DISTANCE * 0.65,
    )
    for robot_id in member_ids:
        old_slot = old_slots.get(robot_id)
        if old_slot is None or not remaining_slots:
            continue
        nearest = min(
            remaining_slots,
            key=lambda slot: math.dist(slot, old_slot),
        )
        if math.dist(nearest, old_slot) <= slot_reuse_limit:
            assigned[robot_id] = nearest
            remaining_slots.remove(nearest)

    remaining_ids = [robot_id for robot_id in member_ids if robot_id not in assigned]
    while remaining_ids and remaining_slots:
        best_robot_id, best_slot = min(
            (
                (robot_id, slot)
                for robot_id in remaining_ids
                for slot in remaining_slots
            ),
            key=lambda pair: (
                khop_state.robot_by_id[pair[0]].position.distance_squared_to(
                    khop_cap_world_target(group, pair[1])
                    or khop_state.robot_by_id[pair[0]].position
                ),
                pair[0],
                pair[1],
            ),
        )
        assigned[best_robot_id] = best_slot
        remaining_ids.remove(best_robot_id)
        remaining_slots.remove(best_slot)
    group.cap_slot_by_robot = assigned


def update_khop_cap_formation(
    group: KHopShepherdGroup,
    dt: float,
) -> bool:
    """Return True only after the physical cap is packed and connected."""
    group.cap_formation_elapsed += dt
    errors = []
    for robot_id, slot in group.cap_slot_by_robot.items():
        robot = khop_state.robot_by_id.get(robot_id)
        target = khop_cap_world_target(group, slot)
        if robot is not None and target is not None:
            errors.append(robot.position.distance_to(target))
    group.cap_mean_error = (
        sum(errors) / len(errors)
        if errors
        else float("inf")
    )
    connected_enough = (
        len(group.member_robot_ids)
        >= min(KHOP_SPLIT_MIN_GROUP_SIZE, khop_required_shepherd_count(group))
    )
    packed = connected_enough and group.cap_mean_error <= KHOP_CAP_FORM_TOLERANCE
    group.cap_formation_dwell = (
        group.cap_formation_dwell + dt
        if packed
        else max(0.0, group.cap_formation_dwell - dt * 0.5)
    )
    return (
        group.cap_formation_dwell >= KHOP_CAP_FORM_DWELL
        or (
            connected_enough
            and group.cap_formation_elapsed >= KHOP_CAP_FORM_MAX_TIME
            and group.cap_mean_error <= KHOP_CAP_FORM_TOLERANCE * 1.8
        )
    )


def khop_required_gatekeeper_count(
    group: Optional[KHopShepherdGroup] = None,
) -> int:
    equilibrium_distance = max(
        ROBOT_RADIUS * 2.2,
        SAFE_RADIUS * NORMAL_EQUILIBRIUM_SCALE,
    )
    robots_across = math.ceil(
        khop_estimate_corridor_width(group)
        / max(equilibrium_distance * 0.78, EPSILON)
    ) + 1
    return max(4, robots_across * KHOP_GATEKEEPER_ROWS)


def khop_display_branch(heading: pygame.Vector2) -> str:
    """Return a display-only cardinal label; control never queries its rect."""
    unit = heading.normalize() if heading.length_squared() > EPSILON else heading
    return max(
        BRANCHES,
        key=lambda branch: unit.dot(BRANCH_DIRECTIONS[branch]),
    )


def khop_local_clearance(
    position: pygame.Vector2,
    heading: pygame.Vector2,
    maximum_distance: float = KHOP_LOCAL_PROBE_DISTANCE,
) -> float:
    """Ray-march a robot-local free-space sensor against collision geometry."""
    if heading.length_squared() <= EPSILON:
        return 0.0
    direction = heading.normalize()
    distance = KHOP_LOCAL_PROBE_STEP
    while distance <= maximum_distance + EPSILON:
        probe = position + direction * distance
        if not is_mask_clear_at(probe, ROBOT_RADIUS * 1.25):
            return max(0.0, distance - KHOP_LOCAL_PROBE_STEP)
        distance += KHOP_LOCAL_PROBE_STEP
    return maximum_distance


def khop_marker_state_for_robot(robot_id: int) -> MarkerState:
    for marker in khop_state.markers.values():
        if marker.robot_id == robot_id:
            return marker.state
    return MarkerState.UNSET


def initialize_khop_capture(robots: list["Robot"]) -> None:
    """Manually seed the initial front Leader, as agreed for experiment one."""
    global khop_state
    khop_state = KHopCaptureState(stage="FORMING_ROOT")
    khop_state.robot_by_id = {robot.robot_id: robot for robot in robots}
    if not robots:
        return

    for robot in robots:
        robot.khop_stream_id = "T0"
        robot.khop_group_id = None
        robot.khop_parent_id = None
        robot.khop_hop = -1
        robot.khop_heading.update(0.0, -1.0)
        robot.khop_stationary_target = None
        robot.marker_group_id = None
        robot.khop_inbox = []
        robot.khop_tx_sequence = 0
        robot.khop_velocity_ema.update(0.0, 0.0)
        robot.khop_neighbor_count_ema = 0.0
        robot.khop_last_marker_id = None
        robot.dhop_clusterhead_id = None
        robot.dhop_parent_id = None
        robot.dhop_gateway = False
        robot.dhop_selection_rule = 0
        robot.dhop_staging_group_id = None
        if robot.role in {
            "KHOP_LEADER",
            "KHOP_SHEPHERD",
            "MARKER",
            "GATEKEEPER",
        }:
            robot.role = "NORMAL"

    front_y = min(robot.position.y for robot in robots)
    front_candidates = [
        robot
        for robot in robots
        if robot.position.y <= front_y + GRID_ROW_SPACING * 1.55
    ]
    swarm_centroid_x = sum(robot.position.x for robot in robots) / len(robots)
    manually_selected = next(
        (
            robot
            for robot in front_candidates
            if robot.robot_id == INITIAL_KHOP_LEADER_ID
        ),
        None,
    )
    leader = manually_selected or min(
        front_candidates,
        key=lambda robot: (
            abs(robot.position.x - swarm_centroid_x),
            robot.robot_id,
        ),
    )
    root = KHopShepherdGroup(
        group_id="T0",
        leader_id=leader.robot_id,
        heading=pygame.Vector2(0.0, -1.0),
        origin=leader.position.copy(),
        display_branch="UP",
        state=KHopGroupState.FORMING,
        cap_anchor=leader.position.copy(),
        created_at=simulation_time,
    )
    khop_state.root_group_id = root.group_id
    khop_state.groups[root.group_id] = root
    rebuild_khop_capture_tree(root, robots)
    khop_state.events.append({
        "time": simulation_time,
        "event": "INITIAL_LEADER",
        "group": root.group_id,
        "leader": root.leader_id,
        "selection": "MANUAL" if manually_selected is not None else "FRONT_LOCAL",
    })
    print(
        f"[K-hop] initial Leader={leader.robot_id}, "
        f"selection={'MANUAL' if manually_selected is not None else 'FRONT_LOCAL'}, "
        f"K={KHOP_CAPTURE_DEPTH}, captured={len(root.member_robot_ids)}/"
        f"{khop_required_shepherd_count(root)}"
    )


def khop_capture_eligible(robot: "Robot", group_id: str) -> bool:
    if robot.role in {"BASE", "ANCHOR", "RELAY", "TRUNK_RELAY", "MARKER", "GATEKEEPER"}:
        return False
    if robot.khop_group_id not in {None, group_id}:
        return False
    reserved_leaders = {
        group.leader_id
        for group in khop_state.groups.values()
        if group.group_id != group_id and group.leader_id >= 0
    }
    if robot.robot_id in reserved_leaders:
        return False
    group = khop_state.groups.get(group_id)
    if group is not None and group.state == KHopGroupState.FORMING:
        return (
            not group.seed_robot_ids
            or robot.robot_id in group.seed_robot_ids
            or robot.khop_stream_id == group_id
        )
    return robot.khop_stream_id in {None, group_id}


def khop_reachable_robots(
    leader: "Robot",
    maximum_hop: int,
) -> list["Robot"]:
    """Return unique robots in the Leader's bounded local communication ball."""
    queue: deque[tuple["Robot", int]] = deque([(leader, 0)])
    visited = {leader.robot_id}
    reachable = [leader]
    while queue:
        current, hop = queue.popleft()
        if hop >= maximum_hop:
            continue
        for neighbor in current.comm_neighbors:
            neighbor_id = getattr(neighbor, "robot_id", -1)
            if neighbor_id < 0 or neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            reachable.append(neighbor)
            queue.append((neighbor, hop + 1))
    return reachable


def khop_reachable_robot_count(leader: "Robot", maximum_hop: int) -> int:
    """Count unique robots in the Leader's bounded local communication ball."""
    return len(khop_reachable_robots(leader, maximum_hop))


def natural_spread_anomaly_threshold() -> int:
    """Witness count needed to flag a natural-spread anomaly.

    Normalized as a ratio of khop_state.last_split_group_size (the locally
    observable population at NATURAL_SPREAD_LOCAL_HOPS, set immediately
    before this is consulted) instead of a fixed absolute robot count --
    see NATURAL_SPREAD_ANOMALY_ROBOT_RATIO for why.
    """
    return max(
        NATURAL_SPREAD_ANOMALY_ROBOTS_MIN,
        round(
            NATURAL_SPREAD_ANOMALY_ROBOT_RATIO * khop_state.last_split_group_size
        ),
    )


def khop_split_cluster_min_size() -> int:
    """Minimum robot count for a candidate cluster/cohort to be trusted.

    Normalized as a ratio of khop_state.last_split_group_size (frozen at the
    moment CONTROLLED_SPREAD began) instead of the fixed absolute count this
    used to share with KHOP_SPLIT_MIN_GROUP_SIZE -- see
    KHOP_SPLIT_CLUSTER_MIN_RATIO for why.
    """
    return max(
        KHOP_SPLIT_CLUSTER_MIN_SIZE_FLOOR,
        round(
            KHOP_SPLIT_CLUSTER_MIN_RATIO * khop_state.last_split_group_size
        ),
    )


def rebuild_khop_capture_tree(
    group: KHopShepherdGroup,
    robots: list["Robot"],
) -> None:
    """Rebuild a bounded BFS tree from the live communication neighbor graph."""
    robot_by_id = khop_state.robot_by_id
    leader = robot_by_id.get(group.leader_id)
    if leader is None:
        return

    target_count = khop_required_shepherd_count(group)
    cap_targets = [
        target
        for slot in khop_cap_slot_offsets(group, max(0, target_count - 1))
        if (target := khop_cap_world_target(group, slot)) is not None
    ]
    queue: deque[tuple["Robot", int]] = deque([(leader, 0)])
    selected: list["Robot"] = []
    parent: dict[int, Optional[int]] = {leader.robot_id: None}
    hop_by_robot = {leader.robot_id: 0}
    visited = {leader.robot_id}

    while queue and len(selected) < target_count:
        current, hop = queue.popleft()
        selected.append(current)
        if hop >= KHOP_CAPTURE_DEPTH:
            continue
        neighbors = sorted(
            (
                neighbor
                for neighbor in current.comm_neighbors
                if getattr(neighbor, "robot_id", -1) >= 0
                and neighbor.robot_id not in visited
                and khop_capture_eligible(neighbor, group.group_id)
            ),
            key=lambda neighbor: (
                min(
                    (
                        neighbor.position.distance_squared_to(target)
                        for target in cap_targets
                    ),
                    default=0.0,
                ),
                current.position.distance_squared_to(neighbor.position),
                -(
                    neighbor.position - group.origin
                ).dot(group.heading),
                neighbor.robot_id,
            ),
        )
        for neighbor in neighbors:
            visited.add(neighbor.robot_id)
            parent[neighbor.robot_id] = current.robot_id
            hop_by_robot[neighbor.robot_id] = hop + 1
            queue.append((neighbor, hop + 1))

    selected_ids = {robot.robot_id for robot in selected}
    for old_id in group.member_robot_ids - selected_ids:
        old_robot = robot_by_id.get(old_id)
        if old_robot is None or old_robot.role not in {"KHOP_LEADER", "KHOP_SHEPHERD"}:
            continue
        old_robot.role = "NORMAL"
        old_robot.khop_group_id = None
        old_robot.khop_parent_id = None
        old_robot.khop_hop = -1

    group.member_robot_ids = selected_ids
    group.tree_parent = {
        robot_id: parent[robot_id]
        for robot_id in selected_ids
    }
    group.hop_by_robot = {
        robot_id: hop_by_robot[robot_id]
        for robot_id in selected_ids
    }
    for captured in selected:
        captured.khop_stream_id = group.group_id
        captured.khop_group_id = group.group_id
        captured.khop_parent_id = parent[captured.robot_id]
        captured.khop_hop = hop_by_robot[captured.robot_id]
        captured.khop_heading = group.heading.copy()
        captured.role = (
            "KHOP_LEADER"
            if captured.robot_id == group.leader_id
            else "KHOP_SHEPHERD"
        )
    assign_khop_cap_slots(group)


def khop_has_forward_neighbor(
    robot: "Robot",
    heading: pygame.Vector2,
) -> bool:
    direction = heading.normalize()
    minimum_cosine = math.cos(math.radians(KHOP_FRONT_NEIGHBOR_CONE_DEGREES))
    for neighbor in robot.comm_neighbors:
        if getattr(neighbor, "robot_id", -1) < 0:
            continue
        delta = neighbor.position - robot.position
        distance = delta.length()
        if distance <= EPSILON or distance > COMM_SAFE_DISTANCE:
            continue
        if delta.dot(direction) / distance >= minimum_cosine:
            return True
    return False


def khop_front_candidate(
    group: KHopShepherdGroup,
    robots: list["Robot"],
) -> Optional["Robot"]:
    lateral = pygame.Vector2(-group.heading.y, group.heading.x)
    lateral_limit = max(
        khop_estimate_corridor_width(group) * 0.60,
        COMM_SAFE_DISTANCE * 0.55,
    )
    candidates = [
        robot
        for robot in robots
        if robot.khop_stream_id == group.group_id
        and robot.role not in {"MARKER", "GATEKEEPER", "ANCHOR", "RELAY", "TRUNK_RELAY"}
        and (
            not group.member_robot_ids
            or robot.robot_id in group.member_robot_ids
        )
        and abs((robot.position - group.origin).dot(lateral)) <= lateral_limit
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda robot: (
            not khop_has_forward_neighbor(robot, group.heading),
            (robot.position - group.origin).dot(group.heading),
            -robot.robot_id,
        ),
    )


def khop_tree_reachable_count(
    candidate: "Robot",
    group: KHopShepherdGroup,
) -> int:
    """Count current Tree members reachable from a candidate within K hops."""
    queue: deque[tuple["Robot", int]] = deque([(candidate, 0)])
    visited = {candidate.robot_id}
    while queue:
        current, hop = queue.popleft()
        if hop >= KHOP_CAPTURE_DEPTH:
            continue
        for neighbor in current.comm_neighbors:
            neighbor_id = getattr(neighbor, "robot_id", -1)
            if (
                neighbor_id not in group.member_robot_ids
                or neighbor_id in visited
            ):
                continue
            visited.add(neighbor_id)
            queue.append((neighbor, hop + 1))
    return len(visited)


def update_khop_group_statistics(
    group: KHopShepherdGroup,
    dt: float = KHOP_CAPTURE_REFRESH_TIME,
) -> None:
    """Aggregate only measurements available to members of one K-hop tree."""
    members = [
        khop_state.robot_by_id[robot_id]
        for robot_id in group.member_robot_ids
        if robot_id in khop_state.robot_by_id
    ]
    if not members:
        group.average_pressure = 0.0
        group.average_density_ratio = 0.0
        group.average_speed = 0.0
        group.front_pressure_ratio = 0.0
        group.front_density_ratio = 0.0
        group.contact_compression_ratio = 0.0
        group.connectivity_ratio = 0.0
        return
    velocity_alpha = 1.0 - math.exp(
        -max(dt, 0.0) / max(KHOP_EVIDENCE_FILTER_TIME, EPSILON)
    )
    for robot in members:
        robot.khop_velocity_ema = robot.khop_velocity_ema.lerp(
            robot.velocity,
            velocity_alpha,
        )
        local_neighbor_count = sum(
            getattr(neighbor, "robot_id", -1) in group.member_robot_ids
            for neighbor in robot.comm_neighbors
        )
        robot.khop_neighbor_count_ema = khop_ema(
            robot.khop_neighbor_count_ema,
            float(local_neighbor_count),
            dt,
            KHOP_EVIDENCE_FILTER_TIME,
        )
    moving_headings = [
        robot.khop_velocity_ema.normalize()
        for robot in members
        if robot.khop_velocity_ema.length_squared() > 1.0
    ]
    if moving_headings:
        mean_heading = sum(moving_headings, pygame.Vector2())
        if mean_heading.length_squared() > EPSILON:
            group.average_heading = mean_heading.normalize()
    group.average_pressure = sum(robot.pressure for robot in members) / len(members)
    group.average_density_ratio = sum(robot.density_ratio for robot in members) / len(members)
    group.average_speed = sum(robot.velocity.length() for robot in members) / len(members)
    ordered_front = sorted(
        members,
        key=lambda robot: (robot.position - group.origin).dot(group.heading),
        reverse=True,
    )
    front_count = min(
        len(ordered_front),
        max(5, math.ceil(len(ordered_front) * 0.35)),
    )
    front_members = ordered_front[:front_count]
    pressure_ratios = [
        robot.pressure
        / max(PRESSURE_GAIN * max(robot.density, EPSILON), EPSILON)
        for robot in front_members
    ]
    group.front_pressure_ratio = sum(pressure_ratios) / max(len(pressure_ratios), 1)
    group.front_density_ratio = sum(
        robot.density_ratio for robot in front_members
    ) / max(len(front_members), 1)

    contact_compressions: list[float] = []
    for robot in front_members:
        for neighbor in robot.comm_neighbors:
            neighbor_id = getattr(neighbor, "robot_id", -1)
            if neighbor_id not in group.member_robot_ids:
                continue
            distance = robot.position.distance_to(neighbor.position)
            if distance >= KHOP_SPH_EQUILIBRIUM_DISTANCE:
                continue
            contact_compressions.append(
                1.0 - distance / max(KHOP_SPH_EQUILIBRIUM_DISTANCE, EPSILON)
            )
    group.contact_compression_ratio = (
        sum(contact_compressions) / len(contact_compressions)
        if contact_compressions
        else 0.0
    )
    captured_ratio = clamp(
        len(members) / max(khop_required_shepherd_count(group), 1),
        0.0,
        1.0,
    )
    base_connected_ratio = sum(
        robot.connected_to_base for robot in members
    ) / len(members)
    group.connectivity_ratio = min(captured_ratio, base_connected_ratio)
    heading_stability = (
        sum(max(0.0, heading.dot(group.average_heading)) for heading in moving_headings)
        / len(moving_headings)
        if moving_headings
        else 0.0
    )
    group.pressure_evidence_ema = khop_ema(
        group.pressure_evidence_ema,
        group.front_pressure_ratio,
        dt,
        KHOP_EVIDENCE_FILTER_TIME,
    )
    group.density_evidence_ema = khop_ema(
        group.density_evidence_ema,
        group.front_density_ratio,
        dt,
        KHOP_EVIDENCE_FILTER_TIME,
    )
    group.compression_evidence_ema = khop_ema(
        group.compression_evidence_ema,
        group.contact_compression_ratio,
        dt,
        KHOP_EVIDENCE_FILTER_TIME,
    )
    group.connectivity_ema = khop_ema(
        group.connectivity_ema,
        group.connectivity_ratio,
        dt,
        KHOP_EVIDENCE_FILTER_TIME,
    )
    group.heading_stability_ema = khop_ema(
        group.heading_stability_ema,
        heading_stability,
        dt,
        KHOP_EVIDENCE_FILTER_TIME,
    )
    leader = khop_state.robot_by_id.get(group.leader_id)
    if leader is not None:
        group.forward_clearance = khop_local_clearance(
            leader.position,
            group.heading,
            KHOP_NAVIGATION_CLEARANCE,
        )


def update_khop_front_leader(
    group: KHopShepherdGroup,
    robots: list["Robot"],
    dt: float,
) -> None:
    # Branch Leaders are elected by the paper's Max-Min 2d-round heuristic.
    # Do not silently replace them with the former geometric front heuristic.
    if group.dhop_clusterhead_ids:
        group.leader_candidate_id = None
        group.leader_candidate_dwell = 0.0
        return
    if group.state != KHopGroupState.EXPLORING:
        group.leader_candidate_id = None
        group.leader_candidate_dwell = 0.0
        return
    if group.forward_clearance <= KHOP_NAVIGATION_CLEARANCE * 0.35:
        group.leader_candidate_id = None
        group.leader_candidate_dwell = 0.0
        return
    candidate = khop_front_candidate(group, robots)
    leader = khop_state.robot_by_id.get(group.leader_id)
    if candidate is None or leader is None or candidate.robot_id == leader.robot_id:
        group.leader_candidate_id = None
        group.leader_candidate_dwell = 0.0
        return
    # A locally exposed Leader remains authoritative even if another lane is
    # a few centimetres farther ahead.  Re-election is only needed when the
    # current Leader is demonstrably behind another neighbor-front.
    if not khop_has_forward_neighbor(leader, group.heading):
        group.leader_candidate_id = None
        group.leader_candidate_dwell = 0.0
        return
    lookahead = KHOP_LEADER_REELECT_DWELL * 0.50
    candidate_progress = (
        candidate.position
        + candidate.khop_velocity_ema * lookahead
        - group.origin
    ).dot(group.heading)
    leader_progress = (
        leader.position
        + leader.khop_velocity_ema * lookahead
        - group.origin
    ).dot(group.heading)
    if candidate.khop_neighbor_count_ema < 1.5:
        group.leader_candidate_id = None
        group.leader_candidate_dwell = 0.0
        return
    minimum_connected_tree = min(
        KHOP_SPLIT_MIN_GROUP_SIZE,
        len(group.member_robot_ids),
    )
    if khop_tree_reachable_count(candidate, group) < minimum_connected_tree:
        group.leader_candidate_id = None
        group.leader_candidate_dwell = 0.0
        return
    if candidate_progress - leader_progress < KHOP_LEADER_REELECT_DISTANCE:
        group.leader_candidate_id = None
        group.leader_candidate_dwell = 0.0
        return
    if group.leader_candidate_id == candidate.robot_id:
        group.leader_candidate_dwell += dt
    else:
        group.leader_candidate_id = candidate.robot_id
        group.leader_candidate_dwell = dt
    if group.leader_candidate_dwell < KHOP_LEADER_REELECT_DWELL:
        return

    old_leader = leader
    group.leader_id = candidate.robot_id
    old_leader.role = "KHOP_SHEPHERD"
    candidate.role = "KHOP_LEADER"
    group.leader_candidate_id = None
    group.leader_candidate_dwell = 0.0
    khop_state.consensus_sequence += 1
    khop_state.events.append({
        "time": simulation_time,
        "event": "LEADER_REELECTED",
        "group": group.group_id,
        "old_leader": old_leader.robot_id,
        "leader": candidate.robot_id,
    })
    print(
        f"[K-hop] front Leader changed: group={group.group_id}, "
        f"{old_leader.robot_id}->{candidate.robot_id}"
    )


def khop_groups_exchange_leader_packets(
    first: KHopShepherdGroup,
    second: KHopShepherdGroup,
) -> bool:
    first_leader = khop_state.robot_by_id.get(first.leader_id)
    second_leader = khop_state.robot_by_id.get(second.leader_id)
    if first_leader is None or second_leader is None:
        return False
    first_hears_second = any(
        message.source_id == second.leader_id
        for message in khop_messages(
            first_leader,
            KHopMessageKind.LEADER_STATUS,
        )
    )
    second_hears_first = any(
        message.source_id == first.leader_id
        for message in khop_messages(
            second_leader,
            KHopMessageKind.LEADER_STATUS,
        )
    )
    return first_hears_second and second_hears_first


def merge_khop_groups(
    survivor: KHopShepherdGroup,
    duplicate: KHopShepherdGroup,
    robots: list["Robot"],
) -> None:
    """Merge two locally communicating, co-directed active capture trees."""
    combined_heading = survivor.heading + duplicate.heading
    if combined_heading.length_squared() > EPSILON:
        survivor.heading = combined_heading.normalize()
    for robot in robots:
        if robot.khop_stream_id == duplicate.group_id:
            robot.khop_stream_id = survivor.group_id
            robot.khop_heading = survivor.heading.copy()
        if robot.marker_group_id == duplicate.group_id and robot.role == "RELAY":
            robot.marker_group_id = survivor.group_id

    survivor.member_robot_ids.update(duplicate.member_robot_ids)
    survivor_candidates = [
        khop_state.robot_by_id[robot_id]
        for robot_id in survivor.member_robot_ids
        if robot_id in khop_state.robot_by_id
    ]
    if survivor_candidates:
        survivor.leader_id = max(
            survivor_candidates,
            key=lambda robot: (
                (robot.position - survivor.origin).dot(survivor.heading),
                robot.khop_neighbor_count_ema,
                -robot.robot_id,
            ),
        ).robot_id

    duplicate_marker = khop_state.markers.get(duplicate.marker_id or "")
    if duplicate_marker is not None:
        duplicate_marker.state = MarkerState.RELEASED
        duplicate_marker.sequence_number += 1
        duplicate_marker_robot = khop_state.robot_by_id.get(
            duplicate_marker.robot_id
        )
        if duplicate_marker_robot is not None:
            duplicate_marker_robot.role = "NORMAL"
            duplicate_marker_robot.marker_group_id = None
            duplicate_marker_robot.khop_stationary_target = None
            duplicate_marker_robot.khop_stream_id = survivor.group_id
            duplicate_marker_robot.khop_heading = survivor.heading.copy()

    duplicate.merged_into = survivor.group_id
    duplicate.return_reason = f"MERGED_INTO:{survivor.group_id}"
    duplicate.state = KHopGroupState.RELEASED
    duplicate.member_robot_ids.clear()
    duplicate.tree_parent.clear()
    duplicate.hop_by_robot.clear()
    duplicate.cap_slot_by_robot.clear()
    survivor.merge_partner_id = None
    survivor.merge_dwell = 0.0
    rebuild_khop_capture_tree(survivor, robots)
    khop_state.consensus_sequence += 1
    khop_state.events.append({
        "time": simulation_time,
        "event": "GROUPS_MERGED",
        "group": survivor.group_id,
        "merged_group": duplicate.group_id,
        "leader": survivor.leader_id,
    })
    print(
        f"[K-hop Merge] {duplicate.group_id}->{survivor.group_id}, "
        f"Leader={survivor.leader_id}, captured={len(survivor.member_robot_ids)}"
    )


def update_khop_group_merging(robots: list["Robot"], dt: float) -> None:
    active = [
        group
        for group in khop_branch_groups()
        if group.state in {KHopGroupState.FORMING, KHopGroupState.EXPLORING}
    ]
    active.sort(key=lambda group: (group.created_at, group.group_id))
    compatible_pair: Optional[tuple[KHopShepherdGroup, KHopShepherdGroup]] = None
    for index, first in enumerate(active):
        first_leader = khop_state.robot_by_id.get(first.leader_id)
        if first_leader is None:
            continue
        for second in active[index + 1:]:
            second_leader = khop_state.robot_by_id.get(second.leader_id)
            if second_leader is None:
                continue
            if angle_between(first.heading, second.heading) > math.radians(
                KHOP_MERGE_HEADING_DEGREES
            ):
                continue
            if first_leader.position.distance_to(second_leader.position) > KHOP_MERGE_RANGE:
                continue
            if not khop_groups_exchange_leader_packets(first, second):
                continue
            compatible_pair = (first, second)
            break
        if compatible_pair is not None:
            break

    for group in active:
        if compatible_pair is None or group not in compatible_pair:
            group.merge_partner_id = None
            group.merge_dwell = max(0.0, group.merge_dwell - dt)
    if compatible_pair is None:
        return
    survivor, duplicate = compatible_pair
    if survivor.merge_partner_id == duplicate.group_id:
        survivor.merge_dwell += dt
    else:
        survivor.merge_partner_id = duplicate.group_id
        survivor.merge_dwell = dt
    duplicate.merge_partner_id = survivor.group_id
    duplicate.merge_dwell = survivor.merge_dwell
    if survivor.merge_dwell >= KHOP_MERGE_DWELL_TIME:
        merge_khop_groups(survivor, duplicate, robots)


def khop_choose_marker_robot(
    robots: list["Robot"],
    target: pygame.Vector2,
    reserved_ids: set[int],
) -> Optional["Robot"]:
    candidates = [
        robot
        for robot in robots
        if robot.robot_id not in reserved_ids
        and robot.role not in {"ANCHOR", "RELAY", "TRUNK_RELAY"}
    ]
    return min(
        candidates,
        key=lambda robot: (
            robot.position.distance_squared_to(target),
            robot.robot_id,
        ),
        default=None,
    )


def khop_largest_connected_component(robot_ids: set[int]) -> set[int]:
    remaining = set(robot_ids)
    largest: set[int] = set()
    while remaining:
        start = remaining.pop()
        component = {start}
        queue = deque([start])
        while queue:
            robot = khop_state.robot_by_id.get(queue.popleft())
            if robot is None:
                continue
            for neighbor in robot.comm_neighbors:
                neighbor_id = getattr(neighbor, "robot_id", -1)
                if neighbor_id not in remaining:
                    continue
                remaining.remove(neighbor_id)
                component.add(neighbor_id)
                queue.append(neighbor_id)
        if len(component) > len(largest):
            largest = component
    return largest


def dhop_shortest_path(
    start_id: int,
    target_id: int,
    neighbors_by_robot: dict[int, list[int]],
) -> list[int]:
    """Return one deterministic local path used by the convergecast phase."""
    if start_id == target_id:
        return [start_id]
    queue = deque([start_id])
    previous: dict[int, Optional[int]] = {start_id: None}
    while queue:
        current = queue.popleft()
        for neighbor_id in neighbors_by_robot.get(current, []):
            if neighbor_id in previous:
                continue
            previous[neighbor_id] = current
            if neighbor_id == target_id:
                queue.clear()
                break
            queue.append(neighbor_id)
    if target_id not in previous:
        return []
    path = [target_id]
    while path[-1] != start_id:
        parent_id = previous[path[-1]]
        if parent_id is None:
            return []
        path.append(parent_id)
    path.reverse()
    return path


def run_maxmin_d_cluster(
    robot_ids: set[int],
    depth: int = DHOP_CLUSTER_DEPTH,
) -> MaxMinDClusterResult:
    """Run Amis et al.'s synchronous Floodmax/Floodmin d-clustering.

    Every node starts with its own unique robot ID, logs one WINNER and SENDER
    per round, performs d Floodmax rounds followed by d Floodmin rounds, then
    applies Rules 1-3. Gateway detection and Rule 4 are resolved during the
    convergecast construction.
    """
    active_ids = {
        robot_id
        for robot_id in robot_ids
        if robot_id in khop_state.robot_by_id
    }
    neighbors_by_robot = {
        robot_id: sorted({
            neighbor_id
            for neighbor in khop_state.robot_by_id[robot_id].comm_neighbors
            if (neighbor_id := getattr(neighbor, "robot_id", -1)) in active_ids
        })
        for robot_id in active_ids
    }
    winner_by_robot = {robot_id: [] for robot_id in active_ids}
    sender_by_robot = {robot_id: [] for robot_id in active_ids}
    current_winner = {robot_id: robot_id for robot_id in active_ids}

    def flood_round(select_maximum: bool) -> None:
        nonlocal current_winner
        next_winner: dict[int, int] = {}
        for robot_id in sorted(active_ids):
            sources = [robot_id, *neighbors_by_robot[robot_id]]
            values = [current_winner[source_id] for source_id in sources]
            selected_value = (
                max(values) if select_maximum else min(values)
            )
            selected_sender = min(
                source_id
                for source_id in sources
                if current_winner[source_id] == selected_value
            )
            next_winner[robot_id] = selected_value
            winner_by_robot[robot_id].append(selected_value)
            sender_by_robot[robot_id].append(selected_sender)
        current_winner = next_winner

    for _ in range(max(1, depth)):
        flood_round(select_maximum=True)
    for _ in range(max(1, depth)):
        flood_round(select_maximum=False)

    clusterhead_by_robot: dict[int, int] = {}
    rule_by_robot: dict[int, int] = {}
    for robot_id in sorted(active_ids):
        max_winners = winner_by_robot[robot_id][:depth]
        min_winners = winner_by_robot[robot_id][depth:]
        # Rule 1: seeing one's own original ID in Floodmin proves that some
        # other node elected it, so it declares itself a clusterhead.
        if robot_id in min_winners:
            selected_head = robot_id
            selected_rule = 1
        else:
            # Rule 2: a node pair appears at least once in both d-round halves;
            # choose the minimum such ID for fairness/load distribution.
            node_pairs = set(max_winners) & set(min_winners)
            if node_pairs:
                selected_head = min(node_pairs)
                selected_rule = 2
            else:
                # Rule 3: no pair exists, so choose the largest Floodmax ID,
                # the only candidate guaranteed by the paper to survive.
                selected_head = max(max_winners, default=robot_id)
                selected_rule = 3
        clusterhead_by_robot[robot_id] = selected_head
        rule_by_robot[robot_id] = selected_rule

    clusterhead_ids = set(clusterhead_by_robot.values())
    # Paper Rule 4: if convergecast encounters a nearer clusterhead on the
    # selected path, that first clusterhead adopts the node as its child.
    for robot_id in sorted(active_ids):
        selected_head = clusterhead_by_robot[robot_id]
        path = dhop_shortest_path(
            robot_id,
            selected_head,
            neighbors_by_robot,
        )
        nearer_heads = [
            node_id
            for node_id in path[1:-1]
            if node_id in clusterhead_ids
        ]
        if nearer_heads:
            clusterhead_by_robot[robot_id] = nearer_heads[0]
            rule_by_robot[robot_id] = 4

    clusterhead_ids = set(clusterhead_by_robot.values())
    gateway_ids = {
        robot_id
        for robot_id in active_ids
        if any(
            clusterhead_by_robot.get(neighbor_id)
            != clusterhead_by_robot[robot_id]
            for neighbor_id in neighbors_by_robot[robot_id]
        )
    }
    parent_by_robot: dict[int, Optional[int]] = {}
    hop_to_clusterhead: dict[int, int] = {}
    for robot_id in sorted(active_ids):
        selected_head = clusterhead_by_robot[robot_id]
        path = dhop_shortest_path(
            robot_id,
            selected_head,
            neighbors_by_robot,
        )
        if not path:
            clusterhead_by_robot[robot_id] = robot_id
            clusterhead_ids.add(robot_id)
            parent_by_robot[robot_id] = None
            hop_to_clusterhead[robot_id] = 0
            rule_by_robot[robot_id] = 1
            continue
        parent_by_robot[robot_id] = path[1] if len(path) > 1 else None
        hop_to_clusterhead[robot_id] = len(path) - 1

    return MaxMinDClusterResult(
        depth=depth,
        winner_by_robot=winner_by_robot,
        sender_by_robot=sender_by_robot,
        clusterhead_by_robot=clusterhead_by_robot,
        rule_by_robot=rule_by_robot,
        clusterhead_ids=clusterhead_ids,
        gateway_ids=gateway_ids,
        parent_by_robot=parent_by_robot,
        hop_to_clusterhead=hop_to_clusterhead,
    )


def detect_persistent_non_corridor_motion(
    root: KHopShepherdGroup,
    dt: float,
) -> set[int]:
    """Return a connected set that *already* moved away from corridor flow.

    The observer is the live bounded-hop neighborhood of the root Leader. No
    wall ray, Junction rectangle, branch label, or pre-authored location is
    consulted. A robot must accumulate both lateral velocity dwell and actual
    travel before it contributes evidence.
    """
    leader = khop_state.robot_by_id.get(root.leader_id)
    if leader is None or root.heading.length_squared() <= EPSILON:
        return set()
    forward = root.heading.normalize()
    reachable = khop_reachable_robots(leader, NATURAL_SPREAD_LOCAL_HOPS)
    reachable_ids = {robot.robot_id for robot in reachable}
    active_ids: set[int] = set()
    for robot in reachable:
        if robot.role != "NORMAL":
            continue
        velocity = robot.khop_velocity_ema
        speed = velocity.length()
        lateral = (
            abs(forward.cross(velocity / speed))
            if speed > EPSILON
            else 0.0
        )
        distinct_motion = (
            speed >= NATURAL_SPREAD_MIN_SPEED
            and lateral >= math.sin(NATURAL_SPREAD_LATERAL_ANGLE)
            and velocity.dot(forward) > -speed * 0.30
        )
        if distinct_motion:
            khop_state.motion_origin_by_robot.setdefault(
                robot.robot_id,
                robot.position.copy(),
            )
            dwell = khop_state.motion_dwell_by_robot.get(robot.robot_id, 0.0)
            khop_state.motion_dwell_by_robot[robot.robot_id] = dwell + dt
            travel = robot.position.distance_to(
                khop_state.motion_origin_by_robot[robot.robot_id]
            )
            if (
                khop_state.motion_dwell_by_robot[robot.robot_id]
                >= NATURAL_SPREAD_ANOMALY_DWELL
                and travel >= NATURAL_SPREAD_MIN_TRAVEL
            ):
                active_ids.add(robot.robot_id)
        else:
            dwell = max(
                0.0,
                khop_state.motion_dwell_by_robot.get(robot.robot_id, 0.0)
                - dt * 1.5,
            )
            if dwell <= EPSILON:
                khop_state.motion_dwell_by_robot.pop(robot.robot_id, None)
                khop_state.motion_origin_by_robot.pop(robot.robot_id, None)
            else:
                khop_state.motion_dwell_by_robot[robot.robot_id] = dwell

    for robot_id in list(khop_state.motion_dwell_by_robot):
        if robot_id not in reachable_ids:
            khop_state.motion_dwell_by_robot.pop(robot_id, None)
            khop_state.motion_origin_by_robot.pop(robot_id, None)
    component = khop_largest_connected_component(active_ids)
    khop_state.candidate_observer_ids = component
    return component


def _motion_kmeans(
    observations: list[tuple[float, int, pygame.Vector2, float]],
    cluster_count: int,
) -> tuple[list[float], list[list[tuple[float, int, pygame.Vector2, float]]]]:
    """Deterministic 1-D angular clustering of robot-created headings."""
    ordered = sorted(observations, key=lambda item: (item[0], item[1]))
    centers = [
        ordered[min(
            len(ordered) - 1,
            round((index + 0.5) * len(ordered) / cluster_count - 0.5),
        )][0]
        for index in range(cluster_count)
    ]
    groups: list[list[tuple[float, int, pygame.Vector2, float]]] = []
    for _ in range(10):
        groups = [[] for _ in centers]
        for observation in ordered:
            selected = min(
                range(len(centers)),
                key=lambda index: abs(observation[0] - centers[index]),
            )
            groups[selected].append(observation)
        if any(not group for group in groups):
            return [], []
        updated = [
            sum(item[0] for item in group) / len(group)
            for group in groups
        ]
        if max(abs(a - b) for a, b in zip(updated, centers)) < math.radians(0.5):
            centers = updated
            break
        centers = updated
    combined = sorted(zip(centers, groups), key=lambda item: item[0])
    return [item[0] for item in combined], [item[1] for item in combined]


def detect_khop_directional_clusters(
    robots: list["Robot"],
    root: KHopShepherdGroup,
) -> list[KHopSplitCluster]:
    """Form candidate cohorts exclusively from observed travel directions."""
    origin = khop_state.split_origin
    if origin is None or root.heading.length_squared() <= EPSILON:
        return []
    incoming = root.heading.normalize()
    observations: list[tuple[float, int, pygame.Vector2, float]] = []
    for robot in robots:
        if robot.role not in KHOP_DYNAMIC_ROLES:
            continue
        # A robot still captured in root's own cap is exclusively the
        # straight-continuation candidate's evidence
        # (detect_khop_straight_continuation_cluster); letting the same
        # robot_id also land in a turning k-means cluster here let two
        # Branch groups independently pick it as their Max-Min head after
        # the split, producing two groups that shared one Leader.
        if robot.khop_group_id == root.group_id:
            continue
        relative = robot.position - origin
        travel = relative.length()
        if not (
            KHOP_SPLIT_CLUSTER_MIN_RADIUS
            <= travel
            <= KHOP_SPLIT_CLUSTER_MAX_RADIUS
        ):
            continue
        displacement_heading = relative / travel
        if displacement_heading.dot(incoming) < KHOP_SPLIT_CLUSTER_BACKWARD_COSINE:
            continue
        velocity = robot.khop_velocity_ema
        if velocity.length() >= NATURAL_SPREAD_MIN_SPEED:
            velocity_heading = velocity.normalize()
            if velocity_heading.dot(incoming) >= KHOP_SPLIT_CLUSTER_BACKWARD_COSINE:
                blended = displacement_heading * 0.72 + velocity_heading * 0.28
                observed_heading = (
                    blended.normalize()
                    if blended.length_squared() > EPSILON
                    else displacement_heading
                )
            else:
                observed_heading = displacement_heading
        else:
            observed_heading = displacement_heading
        angle = math.atan2(
            incoming.cross(observed_heading),
            incoming.dot(observed_heading),
        )
        observations.append((angle, robot.robot_id, observed_heading, travel))
    # A witness can only ever report a *turning* exit this way: nothing is
    # permitted ahead of the front Leader in its own heading, so a
    # straight-through exit has no one to slip past into new territory and
    # register as an observation here. It is measured separately below from
    # the root's own advance and merged in, rather than requiring >= 2
    # turning clusters up front.
    cluster_min_size = khop_split_cluster_min_size()
    maximum_count = min(4, len(observations) // max(cluster_min_size, 1))

    selected: list[KHopSplitCluster] = []
    for cluster_count in range(maximum_count, 0, -1):
        centers, angular_groups = _motion_kmeans(observations, cluster_count)
        if not centers or any(
            second - first < NATURAL_SPREAD_COHORT_MIN_SEPARATION
            for first, second in zip(centers, centers[1:])
        ):
            continue
        proposed: list[KHopSplitCluster] = []
        for center, angular_group in zip(centers, angular_groups):
            if len(angular_group) < cluster_min_size:
                proposed = []
                break
            angular_std = math.sqrt(
                sum((item[0] - center) ** 2 for item in angular_group)
                / len(angular_group)
            )
            if angular_std > NATURAL_SPREAD_COHORT_MAX_ANGLE_STD:
                proposed = []
                break
            raw_ids = {item[1] for item in angular_group}
            component = khop_largest_connected_component(raw_ids)
            connectivity_ratio = len(component) / max(1, len(raw_ids))
            if len(component) < cluster_min_size:
                proposed = []
                break
            component_samples = [item for item in angular_group if item[1] in component]
            heading_sum = sum((item[2] for item in component_samples), pygame.Vector2())
            if heading_sum.length_squared() <= EPSILON:
                proposed = []
                break
            mean_heading = heading_sum.normalize()
            direction_variance = 1.0 - heading_sum.length() / len(component_samples)
            connected_to_base_ratio = sum(
                bool(khop_state.robot_by_id[robot_id].connected_to_base)
                for robot_id in component
            ) / len(component)
            side = pygame.Vector2(-mean_heading.y, mean_heading.x)
            lateral_offsets = [
                (khop_state.robot_by_id[robot_id].position - origin).dot(side)
                for robot_id in component
            ]
            proposed.append(KHopSplitCluster(
                heading=mean_heading,
                robot_ids=component,
                angle=math.atan2(
                    incoming.cross(mean_heading),
                    incoming.dot(mean_heading),
                ),
                mean_travel=sum(item[3] for item in component_samples) / len(component_samples),
                direction_variance=direction_variance,
                connectivity_ratio=connectivity_ratio,
                connected_to_base_ratio=connected_to_base_ratio,
                lateral_width=max(lateral_offsets) - min(lateral_offsets),
            ))
        if proposed:
            selected = sorted(proposed, key=lambda cluster: cluster.angle)
            break

    straight_cluster = detect_khop_straight_continuation_cluster(
        root,
        origin,
        incoming,
    )
    if straight_cluster is not None:
        selected = [
            cluster
            for cluster in selected
            if angle_between(cluster.heading, incoming)
            >= NATURAL_SPREAD_COHORT_MIN_SEPARATION
        ]
        selected.append(straight_cluster)
    return sorted(selected, key=lambda cluster: cluster.angle)


def detect_khop_straight_continuation_cluster(
    root: KHopShepherdGroup,
    origin: pygame.Vector2,
    incoming: pygame.Vector2,
) -> Optional[KHopSplitCluster]:
    """Treat the root's own unblocked forward advance as a branch candidate.

    A straight-through exit can never be witnessed the way a turning one is:
    nobody is permitted ahead of the front Leader in its own heading, so
    nobody can slip past it into new territory and register as an
    observation. Its evidence instead comes directly from the Leader/cap's
    own measured forward progress, connectivity, and heading stability since
    the anomaly was flagged -- evaluated against the same criteria
    (validate_directional_cohorts) a turning cohort must pass.
    """
    leader = khop_state.robot_by_id.get(root.leader_id)
    if leader is None:
        return None
    members = {
        robot_id
        for robot_id in root.member_robot_ids
        if robot_id in khop_state.robot_by_id
    }
    if len(members) < khop_split_cluster_min_size():
        return None
    # A turning cohort's mean_travel is how far a new witness moved since it
    # first looked anomalous -- proof a fresh direction is actually usable.
    # The straight lane has no such moment to measure from (it never looked
    # anomalous), and CONTROLLED_SPREAD deliberately holds the root's own
    # advance near-still while cohorts are being validated, so displacement
    # alone stalls at whatever it was the instant the candidate opened.
    # root.forward_clearance is the same short-range probe already used to
    # confirm a dead-end (inverted here: a large clear run ahead is exactly
    # as much proof the lane is open as a witness's own travel would be).
    del origin
    forward_travel = max(
        0.0,
        (leader.position - root.origin).dot(incoming),
        root.forward_clearance,
    )
    connected_to_base_ratio = sum(
        khop_state.robot_by_id[robot_id].connected_to_base
        for robot_id in members
    ) / len(members)
    # direction_variance exists to express uncertainty in an *inferred*
    # heading: a turning cohort's true direction is unknown and must be
    # estimated from scattered witness velocity samples, which is exactly
    # what that noise measures. The straight lane has no such uncertainty
    # -- its heading isn't inferred from anyone's motion, it is root.heading
    # by construction. Per-member velocity noise (tried both raw and
    # EMA-smoothed) instead measures the cap's own in-place jitter from
    # holding its slot formation, which has a steady-state floor around
    # 0.13-0.3 that never reliably clears NATURAL_SPREAD_COHORT_MAX_VARIANCE
    # regardless of smoothing -- it was answering a different question.
    direction_variance = 0.0
    return KHopSplitCluster(
        heading=incoming.copy(),
        robot_ids=members,
        angle=0.0,
        mean_travel=forward_travel,
        direction_variance=direction_variance,
        connectivity_ratio=root.connectivity_ratio,
        connected_to_base_ratio=connected_to_base_ratio,
        lateral_width=root.estimated_width,
    )


def validate_directional_cohorts(
    clusters: list[KHopSplitCluster],
    dt: float,
) -> list[KHopSplitCluster]:
    """Persist cohort evidence and return only physically validated exits."""
    previous = list(khop_state.cohort_evidence)
    updated: list[DirectionalCohortEvidence] = []
    for cluster in clusters:
        match = min(
            previous,
            key=lambda evidence: angle_between(evidence.heading, cluster.heading),
            default=None,
        )
        if (
            match is None
            or angle_between(match.heading, cluster.heading)
            > math.radians(KHOP_SPLIT_CLUSTER_MATCH_DEGREES)
        ):
            match = DirectionalCohortEvidence(heading=cluster.heading.copy())
        else:
            previous.remove(match)
        instantaneous_valid = (
            len(cluster.robot_ids) >= khop_split_cluster_min_size()
            and cluster.mean_travel >= NATURAL_SPREAD_COHORT_MIN_TRAVEL
            and cluster.direction_variance <= NATURAL_SPREAD_COHORT_MAX_VARIANCE
            and cluster.connectivity_ratio >= NATURAL_SPREAD_COHORT_MIN_CONNECTIVITY
            and cluster.connected_to_base_ratio >= NATURAL_SPREAD_COHORT_MIN_BASE_CONNECTIVITY
            and cluster.lateral_width >= NATURAL_SPREAD_COHORT_MIN_WIDTH
        )
        match.heading = cluster.heading.copy()
        match.robot_ids = set(cluster.robot_ids)
        match.mean_travel = cluster.mean_travel
        match.direction_variance = cluster.direction_variance
        match.connectivity_ratio = cluster.connectivity_ratio
        match.connected_to_base_ratio = cluster.connected_to_base_ratio
        match.lateral_width = cluster.lateral_width
        match.dwell = match.dwell + dt if instantaneous_valid else max(0.0, match.dwell - dt)
        match.valid = match.dwell >= NATURAL_SPREAD_COHORT_MIN_DWELL
        updated.append(match)
    khop_state.cohort_evidence = updated
    valid_clusters: list[KHopSplitCluster] = []
    for cluster in clusters:
        evidence = min(
            updated,
            key=lambda item: angle_between(item.heading, cluster.heading),
            default=None,
        )
        if evidence is not None and evidence.valid:
            valid_clusters.append(cluster)
    return valid_clusters


def begin_khop_controlled_spread(
    root: KHopShepherdGroup,
    leader: "Robot",
) -> None:
    """Slow and bound an anomaly that natural SPH motion already created."""
    khop_state.stage = "CONTROLLED_SPREAD"
    khop_state.recognition_state = SpaceRecognitionState.SPREAD_ANOMALY
    khop_state.split_origin = leader.position.copy()
    khop_state.controlled_spread_elapsed = 0.0
    khop_state.candidate_elapsed = 0.0
    khop_state.split_cluster_signature = ()
    khop_state.split_cluster_dwell = 0.0
    khop_state.pending_split_clusters = []
    khop_state.cohort_evidence = []
    khop_state.last_cluster_headings = []
    khop_state.last_cluster_sizes = []
    khop_state.straggler_wait_elapsed = 0.0
    khop_state.events.append({
        "time": simulation_time,
        "event": "SPREAD_ANOMALY_DETECTED",
        "leader": leader.robot_id,
        "witnesses": sorted(khop_state.candidate_observer_ids),
    })
    print(
        f"[Natural spread] Leader={leader.robot_id}, "
        f"witnesses={len(khop_state.candidate_observer_ids)}; "
        "slowing the main body for cohort validation"
    )


def cancel_khop_junction_candidate() -> None:
    """Merge temporary cohorts and resume ordinary corridor motion."""
    khop_state.recognition_state = SpaceRecognitionState.NOT_A_JUNCTION
    khop_state.events.append({
        "time": simulation_time,
        "event": "JUNCTION_CANDIDATE_REJECTED",
        "elapsed": khop_state.candidate_elapsed,
        "cohorts": len(khop_state.cohort_evidence),
    })
    khop_state.recognition_state = SpaceRecognitionState.COHORT_MERGE
    khop_state.stage = "ADVANCING"
    khop_state.split_origin = None
    khop_state.controlled_spread_elapsed = 0.0
    khop_state.candidate_elapsed = 0.0
    khop_state.candidate_cooldown = NATURAL_SPREAD_RETRY_COOLDOWN
    khop_state.split_cluster_signature = ()
    khop_state.split_cluster_dwell = 0.0
    khop_state.pending_split_clusters = []
    khop_state.cohort_evidence = []
    khop_state.candidate_observer_ids = set()
    khop_state.last_cluster_headings = []
    khop_state.last_cluster_sizes = []
    khop_state.straggler_wait_elapsed = 0.0
    khop_state.motion_origin_by_robot.clear()
    khop_state.motion_dwell_by_robot.clear()
    khop_state.recognition_state = SpaceRecognitionState.NORMAL_CORRIDOR


def natural_cohort_rollout_cost(cluster: KHopSplitCluster) -> float:
    """Short local rollout proxy computed only from witnessed cohort data."""
    members = [
        khop_state.robot_by_id[robot_id]
        for robot_id in cluster.robot_ids
        if robot_id in khop_state.robot_by_id
    ]
    if not members:
        return float("inf")
    mean_density = sum(robot.density_ratio for robot in members) / len(members)
    mean_speed = sum(robot.khop_velocity_ema.length() for robot in members) / len(members)
    speed_variance = sum(
        (robot.khop_velocity_ema.length() - mean_speed) ** 2
        for robot in members
    ) / len(members)
    turn_cost = abs(cluster.angle) / math.pi
    density_cost = max(0.0, mean_density - 1.0)
    travel_credit = clamp(
        cluster.mean_travel / max(NATURAL_SPREAD_LIMIT_RADIUS, EPSILON),
        0.0,
        1.0,
    )
    return (
        3.0 * cluster.direction_variance
        + 2.5 * (1.0 - cluster.connectivity_ratio)
        + 2.0 * (1.0 - cluster.connected_to_base_ratio)
        + 0.8 * density_cost
        + 0.02 * math.sqrt(max(0.0, speed_variance))
        + 0.35 * turn_cost
        + 0.75 * (1.0 - travel_credit)
    )


def split_khop_from_clusters(
    robots: list["Robot"],
    clusters: list[KHopSplitCluster],
) -> None:
    global phase
    root = khop_state.groups.get(khop_state.root_group_id or "")
    origin = khop_state.split_origin
    if root is None or origin is None:
        return
    root.state = KHopGroupState.RELEASED
    for robot_id in root.member_robot_ids:
        robot = khop_state.robot_by_id.get(robot_id)
        if robot is not None and robot.role in {"KHOP_LEADER", "KHOP_SHEPHERD"}:
            robot.role = "NORMAL"
            robot.khop_group_id = None
            robot.khop_parent_id = None
            robot.khop_hop = -1
            robot.dhop_staging_group_id = None
    root.member_robot_ids.clear()
    root.tree_parent.clear()
    root.hop_by_robot.clear()
    root.cap_slot_by_robot.clear()

    for robot in robots:
        if robot.role in KHOP_DYNAMIC_ROLES:
            robot.khop_stream_id = None
            robot.khop_group_id = None
            robot.khop_parent_id = None
            robot.khop_hop = -1

    local_groups: list[KHopShepherdGroup] = []
    used_group_ids: set[str] = set()
    claimed_robot_ids: set[int] = set()
    for index, cluster in enumerate(clusters):
        heading = cluster.heading.normalize()
        # Clusters are independently detected (k-means turning candidates
        # plus the straight-continuation candidate) and are not guaranteed
        # disjoint -- a robot the two happened to agree on would otherwise
        # let its own Max-Min run pick it as Leader in both resulting
        # groups. First cluster in sorted (angle) order keeps a contested
        # robot; later ones treat it as already spoken for.
        cluster_robot_ids = cluster.robot_ids - claimed_robot_ids
        if len(cluster_robot_ids) < khop_split_cluster_min_size():
            continue
        claimed_robot_ids |= cluster_robot_ids
        dhop_result = run_maxmin_d_cluster(
            cluster_robot_ids,
            DHOP_CLUSTER_DEPTH,
        )
        member_counts: dict[int, int] = {}
        for selected_head in dhop_result.clusterhead_by_robot.values():
            member_counts[selected_head] = member_counts.get(selected_head, 0) + 1
        if not member_counts:
            continue
        # A directional component may contain multiple paper-valid d-clusters.
        # The largest one supplies this branch's physical cap; ties choose the
        # smaller ID, preserving Max-Min's Rule-2 fairness preference.
        primary_head = max(
            member_counts,
            key=lambda head_id: (member_counts[head_id], -head_id),
        )
        primary_members = {
            robot_id
            for robot_id, selected_head
            in dhop_result.clusterhead_by_robot.items()
            if selected_head == primary_head
        }
        primary_members.add(primary_head)
        display_branch = khop_display_branch(heading)
        base_group_id = f"T_{display_branch.lower()}"
        group_id = base_group_id
        suffix = 2
        while group_id in khop_state.groups or group_id in used_group_ids:
            group_id = f"{base_group_id}_{suffix}"
            suffix += 1
        used_group_ids.add(group_id)
        group = KHopShepherdGroup(
            group_id=group_id,
            leader_id=-1,
            heading=heading,
            origin=origin.copy(),
            display_branch=display_branch,
            parent_group_id=root.group_id,
            state=KHopGroupState.FORMING,
            seed_robot_ids=primary_members,
            estimated_width=max(cluster.lateral_width, root.estimated_width),
            dhop_winner_by_robot=dhop_result.winner_by_robot,
            dhop_sender_by_robot=dhop_result.sender_by_robot,
            dhop_clusterhead_by_robot=dhop_result.clusterhead_by_robot,
            dhop_rule_by_robot=dhop_result.rule_by_robot,
            dhop_clusterhead_ids=dhop_result.clusterhead_ids,
            dhop_gateway_ids=dhop_result.gateway_ids,
            dhop_parent_by_robot=dhop_result.parent_by_robot,
            created_at=simulation_time,
            rollout_cost=natural_cohort_rollout_cost(cluster),
        )
        for robot_id in cluster_robot_ids:
            robot = khop_state.robot_by_id.get(robot_id)
            if robot is not None:
                robot.khop_stream_id = group_id
                robot.khop_heading = heading.copy()
                robot.dhop_clusterhead_id = (
                    dhop_result.clusterhead_by_robot.get(robot_id)
                )
                robot.dhop_parent_id = dhop_result.parent_by_robot.get(robot_id)
                robot.dhop_gateway = robot_id in dhop_result.gateway_ids
                robot.dhop_selection_rule = dhop_result.rule_by_robot.get(
                    robot_id,
                    0,
                )
        group.leader_id = primary_head
        group.cap_anchor = (
            origin + heading * KHOP_BRANCH_STAGING_OFFSET
        )
        khop_state.groups[group_id] = group
        local_groups.append(group)

    seed_ids = {
        robot_id for group in local_groups for robot_id in group.seed_robot_ids
    }
    reserved_ids = seed_ids | {group.leader_id for group in local_groups}
    for index, group in enumerate(local_groups):
        marker_target = origin + group.heading * KHOP_MARKER_OFFSET
        marker_robot = khop_choose_marker_robot(robots, marker_target, reserved_ids)
        if marker_robot is None:
            continue
        reserved_ids.add(marker_robot.robot_id)
        marker_id = f"M_{group.group_id[2:]}"
        marker = MarkerRecord(
            marker_id=marker_id,
            group_id=group.group_id,
            robot_id=marker_robot.robot_id,
            position=marker_target.copy(),
            origin_leader_id=group.leader_id,
            parent_junction_id=root.group_id,
            local_branch_direction=(group.heading.x, group.heading.y),
            cohort_robot_ids=tuple(sorted(group.seed_robot_ids)),
            estimated_passage_width=group.estimated_width,
            required_guard_count=khop_required_gatekeeper_count(group),
            branch_leader_id=group.leader_id,
            rollout_cost=group.rollout_cost,
            return_direction_local=(-group.heading.x, -group.heading.y),
        )
        khop_state.markers[marker_id] = marker
        group.marker_id = marker_id
        marker_robot.role = "MARKER"
        marker_robot.marker_group_id = group.group_id
        marker_robot.khop_stream_id = None
        marker_robot.khop_group_id = None
        marker_robot.khop_stationary_target = marker_target.copy()

    for group in local_groups:
        if group.leader_id < 0:
            group.state = KHopGroupState.RELEASED
            continue
        rebuild_khop_capture_tree(group, robots)
        khop_state.events.append({
            "time": simulation_time,
            "event": "BRANCH_GROUP_CREATED",
            "group": group.group_id,
            "leader": group.leader_id,
            "marker": group.marker_id,
            "natural_cluster_size": len(group.seed_robot_ids),
            "heading": (group.heading.x, group.heading.y),
        })
        print(
            f"[Max-Min d-hop] group={group.group_id}, d={DHOP_CLUSTER_DEPTH}, "
            f"rounds={2 * DHOP_CLUSTER_DEPTH}, "
            f"Leader={group.leader_id}, marker={group.marker_id}, "
            f"component={len(group.dhop_clusterhead_by_robot)}, "
            f"heads={sorted(group.dhop_clusterhead_ids)}, "
            f"gateways={len(group.dhop_gateway_ids)}, "
            f"primary-members={len(group.seed_robot_ids)}, "
            f"captured={len(group.member_robot_ids)}"
        )

    rollout_order = sorted(
        local_groups,
        key=lambda group: (group.rollout_cost, group.leader_id),
    )
    for rank, group in enumerate(rollout_order, start=1):
        group.rollout_rank = rank
        marker = khop_state.markers.get(group.marker_id or "")
        if marker is not None:
            marker.rollout_rank = rank
    khop_state.branch_queue = [group.group_id for group in rollout_order]
    khop_state.events.append({
        "time": simulation_time,
        "event": "JUNCTION_CONFIRMED",
        "cohorts": len(local_groups),
        "rollout_order": tuple(khop_state.branch_queue),
        "rollout_costs": tuple(
            (group.group_id, round(group.rollout_cost, 6))
            for group in rollout_order
        ),
    })
    print(
        "[Natural rollout] "
        + " > ".join(
            f"{group.group_id}(J={group.rollout_cost:.3f})"
            for group in rollout_order
        )
    )
    khop_state.active_group_id = None
    khop_state.sequential_visit_order = []
    khop_state.stage = "SEQUENTIAL_DFS"
    khop_state.recognition_state = SpaceRecognitionState.JUNCTION_CONFIRMED
    khop_state.consensus_sequence += 1
    phase = SimulationPhase.EXPLORE_BRANCH


def update_khop_unassigned_streams(robots: list["Robot"]) -> None:
    """Let unassigned fluid robots join only Leaders whose packets they hear."""
    valid_group_ids = {group.group_id for group in khop_branch_groups()}
    for robot in robots:
        if robot.role != "NORMAL":
            continue
        current_group = khop_state.groups.get(robot.khop_stream_id or "")
        if (
            current_group is not None
            and current_group.state in {
                KHopGroupState.EXPLORING,
                KHopGroupState.DEAD_END_CONFIRMED,
                KHopGroupState.RETURNING,
                KHopGroupState.MERGING,
            }
        ):
            continue
        robot.khop_stream_id = None
        candidates = [
            message
            for message in khop_messages(robot, KHopMessageKind.LEADER_STATUS)
            if message.group_id in valid_group_ids
            and message.state == KHopGroupState.EXPLORING.name
        ]
        if not candidates:
            continue
        motion = robot.khop_velocity_ema
        selected = min(
            candidates,
            key=lambda message: (
                message.hop_count,
                -(
                    motion.normalize().dot(pygame.Vector2(message.heading))
                    if motion.length_squared() > 1.0
                    else 0.0
                ),
                message.source_id,
            ),
        )
        robot.khop_stream_id = selected.group_id
        robot.dhop_staging_group_id = None
        packet_heading = pygame.Vector2(selected.heading)
        if packet_heading.length_squared() > EPSILON:
            robot.khop_heading = packet_heading.normalize()


def khop_branch_groups() -> list[KHopShepherdGroup]:
    return [
        group
        for group in khop_state.groups.values()
        if group.parent_group_id is not None and group.merged_into is None
    ]


def release_waiting_group_fluid(
    group: KHopShepherdGroup,
    robots: list["Robot"],
) -> None:
    """Keep only the elected physical cap at an inactive branch entrance."""
    for robot in robots:
        if (
            robot.role == "NORMAL"
            and robot.khop_stream_id == group.group_id
        ):
            robot.khop_stream_id = None
            robot.dhop_staging_group_id = group.group_id


def activate_next_sequential_branch(robots: list["Robot"]) -> bool:
    """Activate left-turn, straight, then right-turn by local signed angle."""
    if khop_state.active_group_id is not None:
        return False
    while khop_state.branch_queue:
        group_id = khop_state.branch_queue.pop(0)
        group = khop_state.groups.get(group_id)
        if group is None or group.state != KHopGroupState.WAITING:
            continue
        group.state = KHopGroupState.EXPLORING
        group.dead_end_dwell = 0.0
        group.failure_dwell = 0.0
        khop_state.active_group_id = group_id
        khop_state.sequential_visit_order.append(group_id)
        khop_state.consensus_sequence += 1
        khop_state.events.append({
            "time": simulation_time,
            "event": "SEQUENTIAL_BRANCH_ACTIVATED",
            "group": group_id,
            "leader": group.leader_id,
            "visit_index": len(khop_state.sequential_visit_order),
        })
        print(
            f"[Sequential DFS] activate={group_id}, "
            f"Leader={group.leader_id}, order="
            f"{khop_state.sequential_visit_order}"
        )
        return True
    return False


def khop_group_relays(
    group_id: str,
    robots: list["Robot"],
) -> list["Robot"]:
    return sorted(
        [
            robot
            for robot in robots
            if robot.role == "RELAY" and robot.marker_group_id == group_id
        ],
        key=lambda robot: robot.relay_index,
    )


def release_khop_group_relays(
    group: KHopShepherdGroup,
    robots: list["Robot"],
) -> list["Robot"]:
    released = khop_group_relays(group.group_id, robots)
    for relay in released:
        relay.role = "NORMAL"
        relay.relay_anchor = None
        relay.relay_index = -1
        relay.marker_group_id = None
        relay.khop_stream_id = group.group_id
        relay.khop_heading = group.heading.copy()
    if released:
        khop_state.events.append({
            "time": simulation_time,
            "event": "KHOP_RELAYS_RELEASED",
            "group": group.group_id,
            "count": len(released),
        })
    return released


def release_khop_survivors_for_return(robots: list["Robot"]) -> int:
    """Stand down every remaining physical role once nothing is left to guard.

    Every locally discovered Branch is COMPLETED/BLOCKING at this point, so
    the Gatekeepers, Markers, and any straggling Leader/Shepherd/Relay no
    longer have a Junction to protect. Releasing them lets the whole body
    become one ordinary NORMAL swarm again for the walk back to Base,
    instead of leaving physical barriers frozen at each branch mouth
    forever.
    """
    released = 0
    for robot in robots:
        if robot.role in {
            "GATEKEEPER",
            "MARKER",
            "KHOP_LEADER",
            "KHOP_SHEPHERD",
            "RELAY",
        }:
            robot.role = "NORMAL"
            released += 1
        robot.khop_stream_id = None
        robot.khop_group_id = None
        robot.khop_parent_id = None
        robot.khop_hop = -1
        robot.khop_stationary_target = None
        robot.marker_group_id = None
        robot.relay_anchor = None
        robot.relay_index = -1
        robot.dhop_staging_group_id = None
    for group in khop_branch_groups():
        group.state = KHopGroupState.RELEASED
    return released


def update_khop_relay_deployment(
    group: KHopShepherdGroup,
    robots: list["Robot"],
    dt: float,
) -> None:
    """Freeze a traversed tail robot only when measured path margin is weak."""
    group.relay_deploy_cooldown = max(0.0, group.relay_deploy_cooldown - dt)
    if group.state not in {KHopGroupState.FORMING, KHopGroupState.EXPLORING}:
        return
    leader = khop_state.robot_by_id.get(group.leader_id)
    if leader is None or group.relay_deploy_cooldown > 0.0:
        return
    relays = khop_group_relays(group.group_id, robots)
    if len(relays) >= KHOP_RELAY_MAX_PER_GROUP:
        return
    path_at_risk = (
        not leader.connected_to_base
        or leader.comm_path_margin < KHOP_RELAY_DEPLOY_MARGIN
    )
    if not path_at_risk:
        return

    if relays:
        last_node: object = relays[-1]
    else:
        marker = khop_state.markers.get(group.marker_id or "")
        last_node = (
            khop_state.robot_by_id.get(marker.robot_id)
            if marker is not None
            else base_station
        )
    if last_node is None:
        return
    last_progress = (last_node.position - group.origin).dot(group.heading)
    leader_progress = (leader.position - group.origin).dot(group.heading)
    candidates = []
    for robot in robots:
        if (
            robot.role != "NORMAL"
            or robot.khop_stream_id != group.group_id
            or not robot.connected_to_base
        ):
            continue
        progress = (robot.position - group.origin).dot(group.heading)
        if progress <= last_progress + BREADCRUMB_DEPLOY_DISTANCE:
            continue
        if leader_progress - progress < KHOP_RELAY_MIN_FRONT_GAP:
            continue
        link_distance = robot.position.distance_to(last_node.position)
        if not BREADCRUMB_DEPLOY_DISTANCE <= link_distance <= COMM_RANGE * 0.90:
            continue
        candidates.append((progress, link_distance, robot))
    if not candidates:
        return
    _, _, relay = min(
        candidates,
        key=lambda item: (
            abs(item[1] - BREADCRUMB_SPACING),
            -item[0],
            item[2].robot_id,
        ),
    )
    relay.role = "RELAY"
    relay.relay_anchor = relay.position.copy()
    relay.relay_index = relays[-1].relay_index + 1 if relays else 0
    relay.marker_group_id = group.group_id
    relay.khop_group_id = None
    relay.khop_parent_id = None
    relay.khop_hop = -1
    relay.velocity.update(0.0, 0.0)
    relay.acceleration.update(0.0, 0.0)
    relay.filtered_acceleration.update(0.0, 0.0)
    group.relay_deploy_cooldown = KHOP_RELAY_COOLDOWN
    khop_state.events.append({
        "time": simulation_time,
        "event": "KHOP_RELAY_DEPLOYED",
        "group": group.group_id,
        "relay": relay.robot_id,
        "index": relay.relay_index,
        "leader_margin": leader.comm_path_margin,
    })
    print(
        f"[K-hop Relay] group={group.group_id}, robot={relay.robot_id}, "
        f"index={relay.relay_index}, leader_margin={leader.comm_path_margin:.1f}"
    )


def khop_gatekeeper_slots(group: KHopShepherdGroup) -> list[pygame.Vector2]:
    count = khop_required_gatekeeper_count(group)
    per_row = max(2, math.ceil(count / KHOP_GATEKEEPER_ROWS))
    perpendicular = pygame.Vector2(-group.heading.y, group.heading.x)
    usable_width = khop_estimate_corridor_width(group) - 2.0 * max(
        ROBOT_RADIUS * 2.5,
        4.0 * MAP_SCALE,
    )
    slots: list[pygame.Vector2] = []
    marker = khop_state.markers.get(group.marker_id or "")
    if marker is None:
        return slots
    for row in range(KHOP_GATEKEEPER_ROWS):
        axial_offset = row * KHOP_GATEKEEPER_ROW_GAP
        for index in range(per_row):
            if len(slots) >= count:
                break
            fraction = 0.5 if per_row == 1 else index / (per_row - 1)
            lateral_offset = (fraction - 0.5) * usable_width
            slot = (
                marker.position
                + group.heading * axial_offset
                + perpendicular * lateral_offset
            )
            if is_walkable(slot, ROBOT_RADIUS):
                slots.append(slot)
    return slots


def khop_reassign_released_streams(released: list["Robot"]) -> None:
    active_groups = [
        group
        for group in khop_branch_groups()
        if group.state in {
            KHopGroupState.FORMING,
            KHopGroupState.EXPLORING,
            KHopGroupState.DEAD_END_CONFIRMED,
            KHopGroupState.RETURNING,
        }
    ]
    if not active_groups:
        for robot in released:
            robot.khop_stream_id = None
        return
    active_groups.sort(
        key=lambda group: (
            sum(
                robot.khop_stream_id == group.group_id
                for robot in khop_state.robot_by_id.values()
            ),
            group.leader_id,
        )
    )
    for index, robot in enumerate(released):
        target_group = active_groups[index % len(active_groups)]
        robot.khop_stream_id = target_group.group_id
        robot.khop_heading = target_group.heading.copy()


def activate_khop_marker_gate(group: KHopShepherdGroup) -> None:
    marker = khop_state.markers.get(group.marker_id or "")
    if marker is None:
        group.state = KHopGroupState.RELEASED
        return
    marker.state = MarkerState.COMPLETED
    marker.dead_end_confirmed = True
    marker.completion_witness_id = group.leader_id
    marker.sequence_number += 1
    khop_state.consensus_sequence += 1

    members = [
        khop_state.robot_by_id[robot_id]
        for robot_id in group.member_robot_ids
        if robot_id in khop_state.robot_by_id
        and robot_id != marker.robot_id
    ]
    slots = khop_gatekeeper_slots(group)
    members.sort(
        key=lambda robot: (
            robot.position.distance_squared_to(marker.position),
            robot.robot_id,
        )
    )
    gatekeepers = members[:len(slots)]
    gatekeeper_ids = {robot.robot_id for robot in gatekeepers}
    for robot, slot in zip(gatekeepers, slots):
        robot.role = "GATEKEEPER"
        robot.marker_group_id = group.group_id
        robot.khop_stationary_target = slot.copy()
        robot.khop_stream_id = None
        robot.khop_group_id = None
        robot.khop_parent_id = None
        robot.khop_hop = -1

    released: list["Robot"] = []
    for robot_id in group.member_robot_ids - gatekeeper_ids:
        robot = khop_state.robot_by_id.get(robot_id)
        if robot is None or robot.robot_id == marker.robot_id:
            continue
        if robot.role in {"KHOP_LEADER", "KHOP_SHEPHERD"}:
            robot.role = "NORMAL"
        robot.khop_stationary_target = None
        robot.khop_group_id = None
        robot.khop_parent_id = None
        robot.khop_hop = -1
        released.append(robot)
    marker.state = MarkerState.BLOCKING
    marker.sequence_number += 1
    group.member_robot_ids = gatekeeper_ids
    group.tree_parent.clear()
    group.hop_by_robot.clear()
    group.cap_slot_by_robot.clear()
    group.state = KHopGroupState.GATEKEEPING
    released.extend(release_khop_group_relays(group, list(khop_state.robot_by_id.values())))
    khop_reassign_released_streams(released)
    khop_state.events.append({
        "time": simulation_time,
        "event": "MARKER_BLOCKING",
        "group": group.group_id,
        "marker": marker.marker_id,
        "gatekeepers": len(gatekeepers),
        "released": len(released),
    })
    print(
        f"[Marker] {marker.marker_id}: PENDING->COMPLETED->BLOCKING, "
        f"physical gatekeepers={len(gatekeepers)}, released={len(released)}"
    )


def release_unconfirmed_khop_group(group: KHopShepherdGroup) -> None:
    """Return without blocking when the Leader never confirmed a dead-end."""
    marker = khop_state.markers.get(group.marker_id or "")
    released: list["Robot"] = []
    for robot_id in group.member_robot_ids:
        robot = khop_state.robot_by_id.get(robot_id)
        if robot is None:
            continue
        if robot.role in {"KHOP_LEADER", "KHOP_SHEPHERD"}:
            robot.role = "NORMAL"
        robot.khop_group_id = None
        robot.khop_parent_id = None
        robot.khop_hop = -1
        robot.khop_stationary_target = None
        released.append(robot)
    if marker is not None:
        marker.state = MarkerState.RELEASED
        marker.dead_end_confirmed = False
        marker.sequence_number += 1
        marker_robot = khop_state.robot_by_id.get(marker.robot_id)
        if marker_robot is not None:
            marker_robot.role = "NORMAL"
            marker_robot.marker_group_id = None
            marker_robot.khop_stationary_target = None
            released.append(marker_robot)
    released.extend(
        release_khop_group_relays(group, list(khop_state.robot_by_id.values()))
    )
    group.member_robot_ids.clear()
    group.tree_parent.clear()
    group.hop_by_robot.clear()
    group.cap_slot_by_robot.clear()
    group.state = KHopGroupState.RELEASED
    khop_reassign_released_streams(released)
    khop_state.consensus_sequence += 1
    khop_state.events.append({
        "time": simulation_time,
        "event": "UNCONFIRMED_RETURN_RELEASED",
        "group": group.group_id,
        "reason": group.return_reason or "UNCONFIRMED",
    })
    print(
        f"[Marker] {marker.marker_id if marker else '-'}: "
        f"PENDING->RELEASED; group={group.group_id}, "
        f"reason={group.return_reason or 'UNCONFIRMED'}"
    )


def start_khop_retry(group: KHopShepherdGroup, robots: list["Robot"]) -> bool:
    """Re-arm a released Marker and send the merged main stream back in."""
    marker = khop_state.markers.get(group.marker_id or "")
    if marker is None or group.retry_count >= KHOP_MAX_RETRIES:
        return False
    candidates = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and robot.khop_group_id is None
    ]
    if len(candidates) < KHOP_SPLIT_MIN_GROUP_SIZE:
        return False

    marker_robot = khop_state.robot_by_id.get(marker.robot_id)
    if marker_robot is not None:
        marker_robot.role = "MARKER"
        marker_robot.marker_group_id = group.group_id
        marker_robot.khop_stream_id = None
        marker_robot.khop_stationary_target = marker.position.copy()
        candidates = [
            robot for robot in candidates
            if robot.robot_id != marker_robot.robot_id
        ]
    if len(candidates) < KHOP_SPLIT_MIN_GROUP_SIZE:
        return False

    # Once prior groups have merged, the remaining mobile mass is one stream.
    # Assign it locally to the released entrance instead of inventing a gate or
    # a new global Branch-order command.
    for robot in candidates:
        robot.khop_stream_id = group.group_id
        robot.khop_heading = group.heading.copy()
    new_leader = max(
        candidates,
        key=lambda robot: (
            (robot.position - marker.position).dot(group.heading),
            not khop_has_forward_neighbor(robot, group.heading),
            -robot.robot_id,
        ),
    )
    group.leader_id = new_leader.robot_id
    group.origin = marker.position.copy()
    group.cap_anchor = new_leader.position.copy()
    group.state = KHopGroupState.FORMING
    group.cap_mean_error = float("inf")
    group.cap_formation_dwell = 0.0
    group.cap_formation_elapsed = 0.0
    group.dead_end_confirmed = False
    group.dead_end_dwell = 0.0
    group.failure_dwell = 0.0
    group.return_reason = None
    group.leader_candidate_id = None
    group.leader_candidate_dwell = 0.0
    group.retry_count += 1
    marker.state = MarkerState.PENDING
    marker.dead_end_confirmed = False
    marker.origin_leader_id = new_leader.robot_id
    marker.sequence_number += 1
    rebuild_khop_capture_tree(group, robots)
    khop_state.consensus_sequence += 1
    khop_state.events.append({
        "time": simulation_time,
        "event": "BRANCH_RETRY",
        "group": group.group_id,
        "leader": group.leader_id,
        "retry": group.retry_count,
        "captured": len(group.member_robot_ids),
    })
    print(
        f"[K-hop Retry] group={group.group_id}, retry={group.retry_count}/"
        f"{KHOP_MAX_RETRIES}, Leader={group.leader_id}, "
        f"captured={len(group.member_robot_ids)}"
    )
    return True


def update_khop_dead_end_detection(
    group: KHopShepherdGroup,
    dt: float,
) -> None:
    if group.state != KHopGroupState.EXPLORING:
        return
    leader = khop_state.robot_by_id.get(group.leader_id)
    if leader is None:
        return
    if len(group.member_robot_ids) < KHOP_SPLIT_MIN_GROUP_SIZE:
        group.failure_dwell += dt
    else:
        group.failure_dwell = max(0.0, group.failure_dwell - dt)
    if group.failure_dwell >= KHOP_GROUP_FAILURE_DWELL_TIME:
        group.return_reason = "INSUFFICIENT_CONNECTED_ROBOTS"
        group.dead_end_confirmed = False
        group.state = KHopGroupState.RETURNING
        khop_state.events.append({
            "time": simulation_time,
            "event": "GROUP_RETURN_WITHOUT_CONFIRMATION",
            "group": group.group_id,
            "reason": group.return_reason,
            "members": len(group.member_robot_ids),
        })
        print(
            f"[K-hop] group={group.group_id} returning without completion: "
            f"members={len(group.member_robot_ids)}/"
            f"{KHOP_SPLIT_MIN_GROUP_SIZE}"
        )
        return
    forward_speed = max(0.0, leader.velocity.dot(group.heading))
    density_observed = (
        group.density_evidence_ema >= KHOP_DEAD_END_DENSITY_RATIO
    )
    compression_observed = (
        group.pressure_evidence_ema >= KHOP_DEAD_END_MIN_PRESSURE_RATIO
        or group.compression_evidence_ema
        >= KHOP_DEAD_END_MIN_CONTACT_COMPRESSION
    )
    local_dead_end = (
        group.forward_clearance <= KHOP_DEAD_END_CLEARANCE
        and density_observed
        and compression_observed
        and forward_speed <= KHOP_DEAD_END_MAX_SPEED
    )
    group.dead_end_dwell = (
        group.dead_end_dwell + dt
        if local_dead_end
        else max(0.0, group.dead_end_dwell - dt * 0.5)
    )
    if group.dead_end_dwell < KHOP_DEAD_END_DWELL_TIME:
        return
    group.dead_end_confirmed = True
    group.return_reason = "DEAD_END_CONFIRMED"
    group.state = KHopGroupState.DEAD_END_CONFIRMED
    khop_state.consensus_sequence += 1
    khop_state.events.append({
        "time": simulation_time,
        "event": "DEAD_END_CONFIRMED",
        "group": group.group_id,
        "leader": group.leader_id,
        "clearance": group.forward_clearance,
        "density_ratio": group.density_evidence_ema,
        "pressure_ratio": group.pressure_evidence_ema,
        "contact_compression": group.compression_evidence_ema,
        "connectivity_ratio": group.connectivity_ema,
    })
    group.state = KHopGroupState.RETURNING
    print(
        f"[K-hop Dead-end] group={group.group_id}, Leader={group.leader_id}, "
        f"clearance={group.forward_clearance:.1f}, "
        f"density={group.density_evidence_ema:.2f}, "
        f"pressure={group.pressure_evidence_ema:.3f}, "
        f"contact={group.compression_evidence_ema:.3f}; returning to Marker"
    )


def khop_marker_acknowledges_return(
    group: KHopShepherdGroup,
) -> tuple[bool, bool]:
    """Return (bidirectional_local_contact, dead_end_witness_received)."""
    marker = khop_state.markers.get(group.marker_id or "")
    leader = khop_state.robot_by_id.get(group.leader_id)
    if marker is None or leader is None:
        return False, False
    marker_robot = khop_state.robot_by_id.get(marker.robot_id)
    if marker_robot is None:
        return False, False
    marker_packet = khop_marker_packet(leader, marker.marker_id)
    leader_packets = [
        message
        for message in khop_messages(
            marker_robot,
            KHopMessageKind.LEADER_STATUS,
            group.group_id,
        )
        if message.source_id == group.leader_id
        and message.state in {
            KHopGroupState.DEAD_END_CONFIRMED.name,
            KHopGroupState.RETURNING.name,
            KHopGroupState.MERGING.name,
        }
    ]
    local_contact = (
        marker_packet is not None
        and marker_packet.hop_count <= 1
        and any(message.hop_count <= 1 for message in leader_packets)
    )
    witnessed = any(
        message.hop_count <= 1 and message.dead_end_confirmed
        for message in leader_packets
    )
    return local_contact, witnessed


def khop_completion_packets_agree(
    branch_groups: list[KHopShepherdGroup],
) -> bool:
    """Require every physical Marker to hear every BLOCKING Marker packet."""
    expected_marker_ids = {
        group.marker_id for group in branch_groups if group.marker_id is not None
    }
    if not expected_marker_ids:
        return False
    for group in branch_groups:
        marker = khop_state.markers.get(group.marker_id or "")
        if marker is None:
            return False
        marker_robot = khop_state.robot_by_id.get(marker.robot_id)
        if marker_robot is None:
            return False
        heard_blocking = {
            message.marker_id
            for message in khop_messages(
                marker_robot,
                KHopMessageKind.MARKER_STATUS,
            )
            if message.marker_state == MarkerState.BLOCKING
        }
        if not expected_marker_ids.issubset(heard_blocking):
            return False
    return True


def khop_retry_consensus_choice(
    retryable_groups: list[KHopShepherdGroup],
) -> Optional[KHopShepherdGroup]:
    """Choose a retry only after retry Markers share the same local view."""
    expected = {
        group.marker_id for group in retryable_groups if group.marker_id is not None
    }
    if not expected:
        return None
    for group in retryable_groups:
        marker = khop_state.markers.get(group.marker_id or "")
        marker_robot = (
            khop_state.robot_by_id.get(marker.robot_id)
            if marker is not None
            else None
        )
        if marker_robot is None:
            return None
        heard = {
            message.marker_id
            for message in khop_messages(
                marker_robot,
                KHopMessageKind.MARKER_STATUS,
            )
        }
        if not expected.issubset(heard):
            return None
    return min(
        retryable_groups,
        key=lambda group: (
            group.retry_count,
            -group.connectivity_ema,
            group.group_id,
        ),
    )


def update_khop_capture_state(
    robots: list["Robot"],
    dt: float,
) -> None:
    """Advance method B using local sensing and the current neighbor graph."""
    global phase
    if khop_state.stage == "UNINITIALIZED":
        initialize_khop_capture(robots)
    khop_state.robot_by_id = {robot.robot_id: robot for robot in robots}
    update_khop_local_motion_estimates(robots, dt)
    khop_state.capture_refresh_elapsed += dt

    if khop_state.stage == "RETURNING_TO_BASE":
        in_base = sum(
            get_robot_region(robot.position) == "BOTTOM"
            for robot in robots
        )
        khop_state.last_split_group_size = in_base
        ready = in_base >= math.ceil(
            len(robots) * KHOP_RETURN_BASE_READY_RATIO
        )
        khop_state.return_to_base_dwell = (
            khop_state.return_to_base_dwell + dt if ready else 0.0
        )
        if khop_state.return_to_base_dwell >= KHOP_RETURN_BASE_DWELL:
            khop_state.stage = "COMPLETE"
            phase = SimulationPhase.DONE
            metrics.completion_time = simulation_time
            khop_state.consensus_sequence += 1
            khop_state.events.append({
                "time": simulation_time,
                "event": "RETURNED_TO_BASE",
                "in_base": in_base,
                "total": len(robots),
            })
            print(
                f"[K-hop] swarm gathered back at Base: {in_base}/{len(robots)}"
            )
        return

    if khop_state.stage == "FORMING_ROOT":
        root = khop_state.groups.get(khop_state.root_group_id or "")
        if root is None:
            return
        if khop_state.capture_refresh_elapsed >= KHOP_CAPTURE_REFRESH_TIME:
            rebuild_khop_capture_tree(root, robots)
            update_khop_group_statistics(
                root,
                khop_state.capture_refresh_elapsed,
            )
            khop_state.capture_refresh_elapsed = 0.0
        if update_khop_cap_formation(root, dt):
            root.state = KHopGroupState.EXPLORING
            khop_state.stage = "ADVANCING"
            khop_state.consensus_sequence += 1
            khop_state.events.append({
                "time": simulation_time,
                "event": "ROOT_CAP_FORMED",
                "group": root.group_id,
                "leader": root.leader_id,
                "members": len(root.member_robot_ids),
                "mean_error": root.cap_mean_error,
            })
            print(
                f"[K-hop Cap] root formed: Leader={root.leader_id}, "
                f"members={len(root.member_robot_ids)}, "
                f"mean_error={root.cap_mean_error:.1f}"
            )
        return

    if khop_state.stage == "ADVANCING":
        root = khop_state.groups.get(khop_state.root_group_id or "")
        if root is None:
            return
        if khop_state.capture_refresh_elapsed >= KHOP_CAPTURE_REFRESH_TIME:
            rebuild_khop_capture_tree(root, robots)
            update_khop_group_statistics(
                root,
                khop_state.capture_refresh_elapsed,
            )
            update_khop_relay_deployment(
                root,
                robots,
                khop_state.capture_refresh_elapsed,
            )
            khop_state.capture_refresh_elapsed = 0.0
        leader = khop_state.robot_by_id.get(root.leader_id)
        if leader is None:
            return
        local_group_size = khop_reachable_robot_count(
            leader,
            NATURAL_SPREAD_LOCAL_HOPS,
        )
        khop_state.last_split_group_size = local_group_size
        khop_state.candidate_cooldown = max(
            0.0,
            khop_state.candidate_cooldown - dt,
        )
        natural_witnesses = (
            detect_persistent_non_corridor_motion(root, dt)
            if khop_state.candidate_cooldown <= 0.0
            else set()
        )
        split_detected = (
            len(natural_witnesses) >= natural_spread_anomaly_threshold()
        )
        khop_state.split_dwell = (
            khop_state.split_dwell + dt
            if split_detected
            else max(0.0, khop_state.split_dwell - dt * 1.5)
        )
        if khop_state.split_dwell >= NATURAL_SPREAD_ANOMALY_DWELL:
            begin_khop_controlled_spread(root, leader)
        return

    if khop_state.stage == "CONTROLLED_SPREAD":
        root = khop_state.groups.get(khop_state.root_group_id or "")
        if root is None:
            return
        khop_state.controlled_spread_elapsed += dt
        khop_state.candidate_elapsed += dt
        if khop_state.candidate_elapsed < 0.10:
            khop_state.recognition_state = SpaceRecognitionState.JUNCTION_CANDIDATE
        elif khop_state.candidate_elapsed < 0.22:
            khop_state.recognition_state = SpaceRecognitionState.CONTROLLED_SPREAD
        if khop_state.capture_refresh_elapsed >= KHOP_CAPTURE_REFRESH_TIME:
            rebuild_khop_capture_tree(root, robots)
            update_khop_group_statistics(
                root,
                khop_state.capture_refresh_elapsed,
            )
            khop_state.capture_refresh_elapsed = 0.0
        clusters = detect_khop_directional_clusters(robots, root)
        if clusters:
            khop_state.recognition_state = SpaceRecognitionState.COHORT_FORMATION
            khop_state.last_cluster_headings = [
                (cluster.heading.x, cluster.heading.y)
                for cluster in clusters
            ]
            khop_state.last_cluster_sizes = [
                len(cluster.robot_ids) for cluster in clusters
            ]
        valid_clusters = validate_directional_cohorts(clusters, dt)
        if clusters:
            khop_state.recognition_state = SpaceRecognitionState.COHORT_VALIDATION
        signature = tuple(
            round(
                math.degrees(cluster.angle)
                / KHOP_SPLIT_CLUSTER_MATCH_DEGREES
            )
            for cluster in clusters
        )
        if signature:
            if signature == khop_state.split_cluster_signature:
                khop_state.split_cluster_dwell += dt
            else:
                khop_state.split_cluster_signature = signature
                khop_state.split_cluster_dwell = dt
            khop_state.pending_split_clusters = [
                KHopSplitCluster(
                    heading=cluster.heading.copy(),
                    robot_ids=set(cluster.robot_ids),
                    angle=cluster.angle,
                )
                for cluster in clusters
            ]
        else:
            # A sparse boundary can lose one graph edge for a frame. Preserve
            # the last locally witnessed components and decay confidence
            # gradually instead of erasing the complete observation.
            khop_state.split_cluster_dwell = max(
                0.0,
                khop_state.split_cluster_dwell - dt * 0.20,
            )
        candidate_clusters = valid_clusters
        ready = (
            len(candidate_clusters) >= NATURAL_SPREAD_MIN_COHORTS
            and all(
                angle_between(first.heading, second.heading)
                >= NATURAL_SPREAD_COHORT_MIN_SEPARATION
                for index, first in enumerate(candidate_clusters)
                for second in candidate_clusters[index + 1:]
            )
        )
        if ready:
            # A still-forming candidate (e.g. a straight-through exit, whose
            # evidence converges slower than a turning cohort's) should not
            # be dropped just because other cohorts happened to validate
            # first. Wait a bounded grace period for it to either validate
            # or fall out of detection before finalizing without it.
            stragglers_pending = len(clusters) > len(candidate_clusters)
            if (
                stragglers_pending
                and khop_state.straggler_wait_elapsed
                < NATURAL_SPREAD_STRAGGLER_GRACE_TIME
                and khop_state.candidate_elapsed
                < NATURAL_SPREAD_CANDIDATE_TIMEOUT
            ):
                khop_state.straggler_wait_elapsed += dt
                return
            khop_state.straggler_wait_elapsed = 0.0
            khop_state.recognition_state = SpaceRecognitionState.JUNCTION_CONFIRMED
            split_khop_from_clusters(robots, candidate_clusters)
        else:
            khop_state.straggler_wait_elapsed = 0.0
            if khop_state.candidate_elapsed >= NATURAL_SPREAD_CANDIDATE_TIMEOUT:
                cancel_khop_junction_candidate()
        return

    if khop_state.stage not in {"SEQUENTIAL_DFS", "EXPLORING"}:
        return

    update_khop_unassigned_streams(robots)
    refresh_tree = (
        khop_state.capture_refresh_elapsed >= KHOP_CAPTURE_REFRESH_TIME
    )
    for group in khop_branch_groups():
        if group.state in {
            KHopGroupState.FORMING,
            KHopGroupState.WAITING,
            KHopGroupState.EXPLORING,
            KHopGroupState.DEAD_END_CONFIRMED,
            KHopGroupState.RETURNING,
        }:
            # A WAITING cap is already fully formed and stationary -- there
            # is no live front to track, so leave its roster and slots
            # exactly as they packed instead of re-running BFS capture every
            # refresh cycle. A gate that keeps quietly reselecting members
            # (even onto the same straight shape) reads as an unstable
            # formation rather than one fixed line holding position.
            if refresh_tree and group.state != KHopGroupState.WAITING:
                update_khop_front_leader(
                    group,
                    robots,
                    khop_state.capture_refresh_elapsed,
                )
                rebuild_khop_capture_tree(group, robots)
            update_khop_group_statistics(group, dt)
            if group.state == KHopGroupState.FORMING:
                if update_khop_cap_formation(group, dt):
                    group.state = KHopGroupState.WAITING
                    release_waiting_group_fluid(group, robots)
                    khop_state.consensus_sequence += 1
                    khop_state.events.append({
                        "time": simulation_time,
                        "event": "BRANCH_CAP_FORMED",
                        "group": group.group_id,
                        "leader": group.leader_id,
                        "members": len(group.member_robot_ids),
                        "mean_error": group.cap_mean_error,
                    })
                    print(
                        f"[K-hop Cap] group={group.group_id} staged, "
                        f"Leader={group.leader_id}, "
                        f"members={len(group.member_robot_ids)}, "
                        f"mean_error={group.cap_mean_error:.1f}"
                    )
            elif group.state != KHopGroupState.WAITING:
                update_khop_relay_deployment(group, robots, dt)
                update_khop_dead_end_detection(group, dt)

        if group.state == KHopGroupState.RETURNING:
            local_contact, dead_end_witnessed = khop_marker_acknowledges_return(
                group
            )
            if local_contact:
                group.state = KHopGroupState.MERGING
                if group.dead_end_confirmed and dead_end_witnessed:
                    activate_khop_marker_gate(group)
                else:
                    release_unconfirmed_khop_group(group)

    active_group = khop_state.groups.get(khop_state.active_group_id or "")
    if active_group is not None and active_group.state in {
        KHopGroupState.GATEKEEPING,
        KHopGroupState.RELEASED,
    }:
        khop_state.active_group_id = None
    branch_groups = khop_branch_groups()
    if (
        khop_state.active_group_id is None
        and branch_groups
        and all(group.state != KHopGroupState.FORMING for group in branch_groups)
    ):
        activate_next_sequential_branch(robots)

    if refresh_tree:
        update_khop_group_merging(
            robots,
            max(khop_state.capture_refresh_elapsed, dt),
        )
        khop_state.capture_refresh_elapsed = 0.0

    branch_groups = khop_branch_groups()
    physical_completion = branch_groups and all(
        group.state == KHopGroupState.GATEKEEPING
        for group in branch_groups
    )
    packet_consensus = (
        physical_completion and khop_completion_packets_agree(branch_groups)
    )
    khop_state.completion_consensus_dwell = (
        khop_state.completion_consensus_dwell + dt
        if packet_consensus
        else max(0.0, khop_state.completion_consensus_dwell - dt)
    )
    if (
        physical_completion
        and khop_state.completion_consensus_dwell
        >= KHOP_COMPLETION_CONSENSUS_DWELL
    ):
        khop_state.stage = "RETURNING_TO_BASE"
        khop_state.return_to_base_dwell = 0.0
        released = release_khop_survivors_for_return(robots)
        khop_state.consensus_sequence += 1
        khop_state.events.append({
            "time": simulation_time,
            "event": "ALL_BRANCHES_BLOCKING",
            "consensus_sequence": khop_state.consensus_sequence,
            "released": released,
        })
        print(
            "[K-hop] all locally discovered branches are COMPLETED/BLOCKING; "
            f"released={released}, walking home to Base"
        )
        return

    active_group_exists = any(
        group.state in {
            KHopGroupState.FORMING,
            KHopGroupState.WAITING,
            KHopGroupState.EXPLORING,
            KHopGroupState.DEAD_END_CONFIRMED,
            KHopGroupState.RETURNING,
            KHopGroupState.MERGING,
        }
        for group in branch_groups
    )
    retryable_groups = [
        group
        for group in branch_groups
        if group.state == KHopGroupState.RELEASED
        and group.retry_count < KHOP_MAX_RETRIES
    ]
    if not active_group_exists and retryable_groups:
        khop_state.retry_dwell += dt
        if khop_state.retry_dwell >= KHOP_RETRY_DELAY:
            selected_retry = khop_retry_consensus_choice(retryable_groups)
            if (
                selected_retry is not None
                and start_khop_retry(selected_retry, robots)
            ):
                khop_state.retry_dwell = 0.0
                khop_state.stage = "SEQUENTIAL_DFS"
    elif active_group_exists:
        khop_state.retry_dwell = 0.0
    elif any(group.state == KHopGroupState.RELEASED for group in branch_groups):
        khop_state.stage = "RETRY_REQUIRED"


def compute_khop_tree_force(robot: "Robot") -> pygame.Vector2:
    if not KHOP_CAPTURE_ENABLED or robot.khop_group_id is None:
        return pygame.Vector2()
    group = khop_state.groups.get(robot.khop_group_id)
    parent = khop_state.robot_by_id.get(
        robot.khop_parent_id
        if robot.khop_parent_id is not None
        else -1
    )
    if group is None or parent is None:
        return pygame.Vector2()
    delta = parent.position - robot.position
    distance = delta.length()
    if distance <= EPSILON:
        return pygame.Vector2()
    direction = delta / distance
    extension = max(0.0, distance - KHOP_TREE_TARGET_DISTANCE)
    parent_velocity = getattr(parent, "velocity", pygame.Vector2())
    damping = (parent_velocity - robot.velocity).dot(direction)
    force = direction * (
        KHOP_TREE_SPRING_GAIN * extension
        + KHOP_TREE_DAMPING_GAIN * damping
    )
    return limit_vector(force, KHOP_TREE_FORCE_LIMIT)


def compute_khop_cap_force(robot: "Robot") -> pygame.Vector2:
    """Pull one Shepherd toward its straight, corridor-width fence slot."""
    if not KHOP_CAPTURE_ENABLED or robot.role != "KHOP_SHEPHERD":
        return pygame.Vector2()
    group = khop_state.groups.get(robot.khop_group_id or "")
    if group is None or group.state in {
        KHopGroupState.GATEKEEPING,
        KHopGroupState.RELEASED,
    }:
        return pygame.Vector2()
    slot = group.cap_slot_by_robot.get(robot.robot_id)
    target = khop_cap_world_target(group, slot) if slot is not None else None
    leader = khop_state.robot_by_id.get(group.leader_id)
    if target is None or leader is None:
        return pygame.Vector2()
    position_error = target - robot.position
    velocity_error = leader.velocity - robot.velocity
    force = (
        position_error * KHOP_CAP_POSITION_GAIN
        + velocity_error * KHOP_CAP_VELOCITY_GAIN
    )
    return limit_vector(force, KHOP_CAP_FORCE_LIMIT)


def compute_khop_corridor_centering_force(robot: "Robot") -> pygame.Vector2:
    """Center a live Leader using only its two local side-wall ranges."""
    if not KHOP_CAPTURE_ENABLED or robot.role != "KHOP_LEADER":
        return pygame.Vector2()
    group = khop_state.groups.get(robot.khop_group_id or "")
    if group is None or group.state != KHopGroupState.EXPLORING:
        return pygame.Vector2()
    forward = group.heading.normalize()
    side = pygame.Vector2(-forward.y, forward.x)
    maximum = COMM_RANGE * 1.25
    left_clearance = khop_local_clearance(robot.position, -side, maximum)
    right_clearance = khop_local_clearance(robot.position, side, maximum)
    imbalance = right_clearance - left_clearance
    return limit_vector(
        side * imbalance * KHOP_CAP_CENTERING_GAIN,
        KHOP_CAP_CENTERING_FORCE_LIMIT,
    )


def compute_khop_unassigned_follow_force(robot: "Robot") -> pygame.Vector2:
    """Keep the un-captured swarm one coherent stream behind whichever
    front currently leads (the root before a split, the active Branch
    after), instead of leaving it to raw, undirected pressure diffusion.
    """
    if robot.role != "NORMAL" or khop_state.stage == "FORMING_ROOT":
        return pygame.Vector2()
    heading: Optional[pygame.Vector2] = None
    if khop_state.stage in {"ADVANCING", "CONTROLLED_SPREAD"}:
        root = khop_state.groups.get(khop_state.root_group_id or "")
        if root is not None:
            heading = root.heading
    elif khop_state.stage in {"SEQUENTIAL_DFS", "EXPLORING"}:
        target_group_id = khop_state.active_group_id
        if target_group_id is None and khop_state.branch_queue:
            # Every Branch cap can still be FORMING right after the split,
            # with none yet "active" -- the exact window that let the body
            # spread past every mouth before any cap was ready. Head for the
            # rollout-cheapest Branch (already decided at split time) so it
            # arrives behind that cap instead of fanning out toward all of
            # them at once.
            target_group_id = khop_state.branch_queue[0]
        active = khop_state.groups.get(target_group_id or "")
        if active is not None:
            heading = active.heading
    if heading is None or heading.length_squared() <= EPSILON:
        return pygame.Vector2()
    forward = heading.normalize()
    force = forward * KHOP_UNASSIGNED_FOLLOW_FORCE
    if khop_state.stage == "CONTROLLED_SPREAD":
        # Reinforce a genuinely witnessed lateral flow instead of dragging
        # it straight back onto the old corridor heading -- the whole point
        # is to let a real spread persist long enough to become a candidate.
        motion = robot.khop_velocity_ema
        if motion.length() >= NATURAL_SPREAD_MIN_SPEED:
            motion_heading = motion.normalize()
            lateral = abs(forward.cross(motion_heading))
            if (
                lateral >= math.sin(NATURAL_SPREAD_LATERAL_ANGLE)
                and motion_heading.dot(forward) > -0.30
            ):
                force += motion_heading * (
                    KHOP_UNASSIGNED_FOLLOW_FORCE * 0.70
                )
    return limit_vector(force, KHOP_TREE_FORCE_LIMIT)


def compute_khop_route_force(robot: "Robot") -> pygame.Vector2:
    if robot.role in {"MARKER", "GATEKEEPER", "ANCHOR", "RELAY", "TRUNK_RELAY"}:
        return pygame.Vector2()
    if khop_state.stage == "RETURNING_TO_BASE":
        # Nothing is left to guard once every locally discovered Branch is
        # COMPLETED/BLOCKING; walk the whole body back to the fixed,
        # always-known Base origin instead of leaving it wherever forward
        # propulsion happened to stop.
        direction = direction_toward_base_path(robot.position)
        if direction.length_squared() <= EPSILON:
            return -robot.velocity * KHOP_RETURN_TO_BASE_DAMPING
        desired_velocity = direction * (FLOW_BACKTRACK_MAX_SPEED * 0.55)
        return limit_vector(
            direction * KHOP_RETURN_TO_BASE_FORCE
            + (desired_velocity - robot.velocity)
            * KHOP_RETURN_TO_BASE_DAMPING,
            KHOP_TREE_FORCE_LIMIT,
        )
    if khop_state.stage == "COMPLETE":
        return -robot.velocity * KHOP_RETURN_DAMPING
    group = khop_state.groups.get(robot.khop_stream_id or "")
    if group is None:
        staged_group = khop_state.groups.get(
            robot.dhop_staging_group_id or ""
        )
        if staged_group is not None and staged_group.state in {
            KHopGroupState.FORMING,
            KHopGroupState.WAITING,
        }:
            direction = -staged_group.heading.normalize()
            return limit_vector(
                direction * (KHOP_RETURN_FORCE * 0.45)
                - robot.velocity * (KHOP_RETURN_DAMPING * 0.60),
                KHOP_TREE_FORCE_LIMIT,
            )
        return compute_khop_unassigned_follow_force(robot)
    if group.state in {
        KHopGroupState.GATEKEEPING,
        KHopGroupState.RELEASED,
    }:
        return pygame.Vector2()

    if group.state in {
        KHopGroupState.FORMING,
        KHopGroupState.WAITING,
    }:
        # Do not let the Leader run away while its bounded-hop Shepherds are
        # still packing into the physical cap. Slot forces do the forming.
        if robot.role == "KHOP_LEADER" and group.cap_anchor is not None:
            return limit_vector(
                (group.cap_anchor - robot.position) * KHOP_CAP_ANCHOR_GAIN
                - robot.velocity * KHOP_RETURN_DAMPING,
                KHOP_CAP_FORCE_LIMIT,
            )
        return -robot.velocity * KHOP_RETURN_DAMPING

    if (
        khop_state.stage == "CONTROLLED_SPREAD"
        and group.group_id == khop_state.root_group_id
        and khop_state.split_origin is not None
    ):
        # Only the captured Leader/Shepherd cap ever reaches this branch
        # (a plain NORMAL robot's khop_stream_id is never the root's, so it
        # is handled by compute_khop_unassigned_follow_force instead). Hold
        # the cap near the candidate point while the ordinary body's own
        # motion supplies all the directional evidence being validated.
        return -robot.velocity * KHOP_RETURN_DAMPING

    leader_packet = min(
        khop_messages(
            robot,
            KHopMessageKind.LEADER_STATUS,
            group.group_id,
        ),
        key=lambda message: message.hop_count,
        default=None,
    )
    effective_state = group.state
    effective_heading = group.heading
    if leader_packet is not None:
        try:
            effective_state = KHopGroupState[leader_packet.state]
        except KeyError:
            effective_state = group.state
        packet_heading = pygame.Vector2(leader_packet.heading)
        if packet_heading.length_squared() > EPSILON:
            effective_heading = packet_heading.normalize()

    if effective_state in {
        KHopGroupState.DEAD_END_CONFIRMED,
        KHopGroupState.RETURNING,
        KHopGroupState.MERGING,
    }:
        marker_packet = khop_marker_packet(robot, group.marker_id)
        marker_robot = (
            khop_state.robot_by_id.get(
                marker_packet.source_id
                if marker_packet.hop_count <= 1
                else marker_packet.via_id
                if marker_packet.via_id is not None
                else marker_packet.source_id
            )
            if marker_packet is not None
            else None
        )
        if marker_robot is not None:
            delta = marker_robot.position - robot.position
            direction = (
                delta.normalize()
                if delta.length_squared() > EPSILON
                else -group.heading
            )
            robot.khop_last_marker_id = marker_packet.marker_id
        else:
            # Until the Marker is a direct neighbor, reverse the locally stored
            # entry heading. No global Marker position is read here.
            direction = -effective_heading
        desired_velocity = direction * (FLOW_BACKTRACK_MAX_SPEED * 0.55)
        return limit_vector(
            direction * KHOP_RETURN_FORCE
            + (desired_velocity - robot.velocity) * KHOP_RETURN_DAMPING,
            KHOP_TREE_FORCE_LIMIT,
        )

    if robot.role == "KHOP_LEADER":
        magnitude = KHOP_LEADER_FORCE
    elif robot.role == "KHOP_SHEPHERD":
        magnitude = KHOP_SHEPHERD_FORCE
    else:
        magnitude = KHOP_STREAM_FORCE
    return effective_heading * magnitude


def compute_khop_gatekeeping_force(robot: "Robot") -> pygame.Vector2:
    """Local Marker-message steering plus physical gatekeeper repulsion."""
    if not KHOP_CAPTURE_ENABLED or robot.role in {"MARKER", "GATEKEEPER"}:
        return pygame.Vector2()
    total = pygame.Vector2()
    for message in khop_messages(robot, KHopMessageKind.MARKER_STATUS):
        if message.marker_state != MarkerState.BLOCKING or message.hop_count > 1:
            continue
        marker_robot = khop_state.robot_by_id.get(message.source_id)
        if marker_robot is None:
            continue
        delta = robot.position - marker_robot.position
        distance = delta.length()
        if distance <= EPSILON or distance >= KHOP_GATEKEEPER_FORCE_RADIUS * 1.25:
            continue
        heading = pygame.Vector2(message.heading)
        if heading.length_squared() <= EPSILON:
            continue
        heading = heading.normalize()
        ratio = 1.0 - distance / (KHOP_GATEKEEPER_FORCE_RADIUS * 1.25)
        side = pygame.Vector2(-heading.y, heading.x)
        if robot.velocity.dot(side) < 0.0:
            side = -side
        total += (
            -heading * KHOP_GATEKEEPER_FORCE * ratio**2
            + side * KHOP_GATEKEEPER_FORCE * 0.30 * ratio
        )
    for group in khop_branch_groups():
        if group.state not in {
            KHopGroupState.FORMING,
            KHopGroupState.WAITING,
            KHopGroupState.EXPLORING,
            KHopGroupState.DEAD_END_CONFIRMED,
            KHopGroupState.RETURNING,
            KHopGroupState.GATEKEEPING,
        }:
            continue
        total += compute_khop_barrier_segment_force(robot, group)
    return limit_vector(total, KHOP_GATEKEEPER_FORCE_LIMIT)


def khop_barrier_line_point(
    group: KHopShepherdGroup,
) -> Optional[pygame.Vector2]:
    """Current position of a group's blocking line, if it should have one.

    FORMING/WAITING hold at the fixed staging point (group.cap_anchor);
    GATEKEEPING holds at the fixed post-completion Marker; an actively
    advancing/retreating group (EXPLORING/DEAD_END_CONFIRMED/RETURNING)
    tracks its live Leader, since that is where its own physical cap
    actually is right now, not where it started.
    """
    if group.state in {KHopGroupState.FORMING, KHopGroupState.WAITING}:
        return group.cap_anchor
    if group.state == KHopGroupState.GATEKEEPING:
        marker = khop_state.markers.get(group.marker_id or "")
        return marker.position if marker is not None else None
    if group.state in {
        KHopGroupState.EXPLORING,
        KHopGroupState.DEAD_END_CONFIRMED,
        KHopGroupState.RETURNING,
    }:
        leader = khop_state.robot_by_id.get(group.leader_id)
        return leader.position if leader is not None else None
    return None


def compute_khop_barrier_segment_force(
    robot: "Robot",
    group: KHopShepherdGroup,
) -> pygame.Vector2:
    """Repel a non-member from a group's current line as a straight wall
    segment spanning the corridor width, instead of only from wherever its
    individual physical bodies happen to be right now. This is what gives
    full-width coverage immediately -- even mid-formation, before a single
    Shepherd body has actually reached its slot -- rather than protection
    only as good as however far the cap has already converged.
    """
    if (
        robot.khop_group_id == group.group_id
        or robot.khop_stream_id == group.group_id
    ):
        return pygame.Vector2()
    if group.heading.length_squared() <= EPSILON:
        return pygame.Vector2()
    anchor = khop_barrier_line_point(group)
    if anchor is None:
        return pygame.Vector2()
    forward = group.heading.normalize()
    side = pygame.Vector2(-forward.y, forward.x)
    half_width = max(
        khop_cap_line_spacing(),
        (khop_estimate_corridor_width(group) - ROBOT_RADIUS * 4.0) * 0.5,
    )
    relative = robot.position - anchor
    depth = relative.dot(forward)
    if depth <= 0.0:
        # Already on the safe (Junction/Base) side of the line; a robot
        # properly trailing the cap from behind should be free to pack up
        # close against it, not get shoved away by its own gate.
        return pygame.Vector2()
    lateral = clamp(relative.dot(side), -half_width, half_width)
    closest = anchor + side * lateral
    delta = robot.position - closest
    distance = delta.length()
    if distance <= EPSILON or distance >= KHOP_GATEKEEPER_FORCE_RADIUS:
        return pygame.Vector2()
    ratio = 1.0 - distance / KHOP_GATEKEEPER_FORCE_RADIUS
    away = delta / distance
    outward_component = max(0.0, away.dot(-forward))
    return (
        -forward * KHOP_GATEKEEPER_FORCE * ratio**2
        + away
        * KHOP_GATEKEEPER_FORCE
        * ratio**2
        * (0.35 + 0.65 * outward_component)
    )


def draw_khop_capture_network(surface, robots: list["Robot"]) -> None:
    for group in khop_state.groups.values():
        for child_id, parent_id in group.tree_parent.items():
            if parent_id is None:
                continue
            child = khop_state.robot_by_id.get(child_id)
            parent = khop_state.robot_by_id.get(parent_id)
            if child is None or parent is None:
                continue
            pygame.draw.line(
                surface,
                KHOP_TREE_LINK_COLOR,
                child.position,
                parent.position,
                width=1,
            )
    for marker in khop_state.markers.values():
        color = (
            MARKER_PENDING_COLOR
            if marker.state == MarkerState.PENDING
            else MARKER_COMPLETE_COLOR
        )
        pygame.draw.circle(surface, color, marker.position, 8, width=2)
        label = small_font.render(
            f"{marker.marker_id}:{marker.state.name}",
            True,
            color,
        )
        group = khop_state.groups.get(marker.group_id)
        heading = (
            group.heading.normalize()
            if group is not None and group.heading.length_squared() > EPSILON
            else pygame.Vector2(1.0, 0.0)
        )
        if heading.x < -0.5:
            label_position = (
                marker.position.x - label.get_width() - 10,
                marker.position.y - label.get_height() * 0.5,
            )
        elif heading.x > 0.5:
            label_position = (
                marker.position.x + 10,
                marker.position.y - label.get_height() * 0.5,
            )
        elif heading.y < -0.5:
            label_position = (
                marker.position.x - label.get_width() * 0.5,
                marker.position.y - label.get_height() - 10,
            )
        else:
            label_position = (
                marker.position.x - label.get_width() * 0.5,
                marker.position.y + 10,
            )
        surface.blit(label, label_position)


def save_khop_headless_snapshot(
    path: str,
    robots: list["Robot"],
    color_reference_density: float,
) -> None:
    """Save the physical K-hop frame used by automated visual checks."""
    snapshot = pygame.Surface((SIMULATION_VIEW_WIDTH, SCREEN_HEIGHT))
    snapshot.fill(BACKGROUND_COLOR)
    pygame.draw.polygon(snapshot, FLOOR_COLOR, cross_points)
    pygame.draw.polygon(snapshot, WALL_COLOR, cross_points, width=2)
    pygame.draw.circle(snapshot, JUNCTION_COLOR, (center_x, center_y), 5)
    pygame.draw.circle(snapshot, BASE_COLOR, BASE_POSITION, 6)
    draw_khop_capture_network(snapshot, robots)
    for robot in robots:
        robot.draw(snapshot, color_reference_density, True)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(snapshot, str(output_path))
    print(f"[Headless] snapshot={output_path.resolve()}")


def build_khop_hud_lines(robots: list["Robot"]) -> list[str]:
    communication_stats = get_communication_stats(robots)
    root_group = khop_state.groups.get(khop_state.root_group_id or "")
    lines = [
        "SPH | Natural-spread Junction discovery | K-hop Shepherd",
        f"FPS={clock.get_fps():.1f} | robots={len(robots)} | stage={khop_state.stage}",
        (
            f"Policy={KHOP_CAPTURE_POLICY_VERSION} | K={KHOP_CAPTURE_DEPTH} | "
            f"d={DHOP_CLUSTER_DEPTH} | "
            f"target/group={khop_required_shepherd_count(root_group)}"
        ),
        (
            f"Recognition={khop_state.recognition_state.name} | "
            f"policy={NATURAL_SPREAD_POLICY_VERSION}"
        ),
        (
            f"Natural anomaly: witnesses={len(khop_state.candidate_observer_ids)}/"
            f"{natural_spread_anomaly_threshold()} | "
            f"dwell={khop_state.split_dwell:.2f}s | "
            f"candidate={khop_state.candidate_elapsed:.2f}/"
            f"{NATURAL_SPREAD_CANDIDATE_TIMEOUT:.2f}s"
        ),
        (
            f"Motion cohorts: sizes={khop_state.last_cluster_sizes or '-'} | "
            f"valid={sum(evidence.valid for evidence in khop_state.cohort_evidence)}/"
            f"{NATURAL_SPREAD_MIN_COHORTS} | "
            "forward/open-angle pre-detection=OFF"
        ),
        (
            f"Consensus sequence={khop_state.consensus_sequence} | "
            f"packet deliveries={khop_state.message_delivery_count}"
        ),
        (
            f"Sequential active={khop_state.active_group_id or '-'} | "
            f"queue={khop_state.branch_queue or '-'} | "
            f"visited={khop_state.sequential_visit_order or '-'}"
        ),
        "Groups:",
    ]
    for group in sorted(khop_branch_groups(), key=lambda item: item.group_id):
        marker = khop_state.markers.get(group.marker_id or "")
        lines.extend([
            (
                f"  {group.group_id}: {group.state.name} | leader={group.leader_id} | "
                f"tree={len(group.member_robot_ids)} | retry={group.retry_count}/"
                f"{KHOP_MAX_RETRIES} | heading="
                f"({group.heading.x:+.1f},{group.heading.y:+.1f}) | "
                f"rollout={group.rollout_cost:.3f} rank={group.rollout_rank}"
            ),
            (
                f"    Max-Min heads={sorted(group.dhop_clusterhead_ids) or '-'} | "
                f"gateways={len(group.dhop_gateway_ids)} | "
                f"rounds={2 * DHOP_CLUSTER_DEPTH}"
            ),
            (
                f"    hop={max(group.hop_by_robot.values(), default=0)}/{KHOP_CAPTURE_DEPTH} | "
                f"cap-error={group.cap_mean_error:.1f}/"
                f"{KHOP_CAP_FORM_TOLERANCE:.1f} | "
                f"clear={group.forward_clearance:.1f} | rho={group.density_evidence_ema:.2f} | "
                f"P={group.pressure_evidence_ema:.3f} | "
                f"contact={group.compression_evidence_ema:.3f} | dead-end dwell="
                f"{group.dead_end_dwell:.2f}/{KHOP_DEAD_END_DWELL_TIME:.2f}"
            ),
            (
                f"    marker={marker.marker_id}:{marker.state.name} seq={marker.sequence_number}"
                if marker is not None
                else "    marker=-"
            ),
        ])
    lines.extend([
        (
            f"Physical gatekeepers={sum(robot.role == 'GATEKEEPER' for robot in robots)} | "
            f"Markers={sum(robot.role == 'MARKER' for robot in robots)} | "
            f"K-hop relays={sum(robot.role == 'RELAY' for robot in robots)}"
        ),
        (
            f"Communication observer={communication_stats['connected']}/{len(robots)} | "
            f"max hop={communication_stats['max_hop']}"
        ),
        "Split inputs: actual spread positions, local velocity, comm components, wall rays",
        "Controller inputs: local free-space, packets, Hop, heading, pressure/density/contact",
        "Map coordinates: collision/rendering only",
    ])
    return lines


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
    # Method B maintains the Leader-rooted K-hop tree directly.  Base-rooted
    # parent leashes would prevent independent Branch groups from separating.
    if KHOP_CAPTURE_ENABLED:
        return pygame.Vector2()
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


def get_cohort_lane_offset(
    robot: "Robot",
    branch: Optional[str] = None,
) -> float:
    """Map the deployment column onto equilibrium-spaced corridor lanes."""
    lane_limit = max(
        0.0,
        half_width - FILL_LANE_WALL_CLEARANCE,
    )
    if lane_limit <= EPSILON:
        return 0.0
    lane_branch = branch or active_branch
    spacing = branch_fill_equilibrium_spacing(lane_branch)
    lane_count = max(
        2,
        math.floor((2.0 * lane_limit + EPSILON) / spacing) + 1,
    )
    source_ratio = clamp(
        (robot.ingress_lane_x - (center_x - lane_limit))
        / max(2.0 * lane_limit, EPSILON),
        0.0,
        1.0,
    )
    lane_index = int(round(source_ratio * (lane_count - 1)))
    return (
        -lane_limit
        + 2.0 * lane_limit * lane_index / (lane_count - 1)
    )


def hose_path_branch() -> Optional[str]:
    """Branch defining the currently continuous Base/J0 hose."""
    if (
        phase in {
            SimulationPhase.PRESSURE_PUSH,
            SimulationPhase.FLOW_BACKTRACK,
        }
        and transfer_branch is not None
        and not final_base_transfer_active
    ):
        return transfer_branch
    if phase in {
        SimulationPhase.EXPLORE_BRANCH,
        SimulationPhase.FILL_BEHIND_SHEPHERD,
        SimulationPhase.PRESSURE_PUSH,
        SimulationPhase.FLOW_BACKTRACK,
    }:
        return active_branch
    return None


def hose_forward_direction(
    robot: "Robot",
    branch: str,
) -> pygame.Vector2:
    """Local forward tangent of the Base-Junction-Branch hose."""
    if get_robot_region(robot.position) == "BOTTOM":
        return pygame.Vector2(0.0, -1.0)
    return get_lane_preserving_cohort_direction(robot, branch)


def update_hose_continuity_forces(robots) -> None:
    """Join Base, J0, and the destination Branch as one damped hose.

    Robots retain their deployment lane, while adjacent robots in that lane
    are coupled in the one-dimensional path coordinate.  Because adjacency is
    computed across region labels, neither the Base/J0 seam nor the J0/Branch
    turn can open into a void while the body moves.
    """
    global hose_continuity_forces, hose_continuity_link_count
    hose_continuity_forces = {}
    hose_continuity_link_count = 0
    branch = hose_path_branch()
    if branch is None:
        return
    allowed_regions = {"BOTTOM", "JUNCTION", branch}
    lane_groups: dict[float, list[tuple[float, "Robot"]]] = {}
    for robot in robots:
        if (
            robot.role != "NORMAL"
            or get_robot_region(robot.position) not in allowed_regions
        ):
            continue
        lane_key = round(get_cohort_lane_offset(robot, branch), 3)
        lane_groups.setdefault(lane_key, []).append(
            (relay_path_progress(robot.position, branch), robot)
        )
        hose_continuity_forces[robot.robot_id] = pygame.Vector2()

    for lane_members in lane_groups.values():
        lane_members.sort(key=lambda item: (item[0], item[1].robot_id))
        for (rear_progress, rear), (front_progress, front) in zip(
            lane_members,
            lane_members[1:],
        ):
            rear_region = get_robot_region(rear.position)
            front_region = get_robot_region(front.position)
            if (
                phase == SimulationPhase.FILL_BEHIND_SHEPHERD
                and rear_region == active_branch
                and front_region == active_branch
            ):
                # The local row spring already handles the Branch body and
                # deliberately yields to the dense Shepherd packing layer.
                continue
            progress_gap = front_progress - rear_progress
            desired_gap = 0.5 * (
                adaptive_equilibrium_radius(rear)
                + adaptive_equilibrium_radius(front)
            )
            rear_tangent = hose_forward_direction(rear, branch)
            front_tangent = hose_forward_direction(front, branch)
            rear_speed = rear.velocity.dot(rear_tangent)
            front_speed = front.velocity.dot(front_tangent)
            scalar = (
                HOSE_CONTINUITY_AXIAL_GAIN
                * (progress_gap - desired_gap)
                + HOSE_CONTINUITY_DAMPING
                * (front_speed - rear_speed)
            )
            hose_continuity_forces[rear.robot_id] += (
                rear_tangent * scalar
            )
            hose_continuity_forces[front.robot_id] -= (
                front_tangent * scalar
            )
            hose_continuity_link_count += 1

    for force in hose_continuity_forces.values():
        limit_vector(force, HOSE_CONTINUITY_FORCE_LIMIT)


def compute_hose_continuity_force(robot: "Robot") -> pygame.Vector2:
    return hose_continuity_forces.get(
        robot.robot_id,
        pygame.Vector2(),
    ).copy()


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
    if (
        phase == SimulationPhase.FLOW_BACKTRACK
        and region == "JUNCTION"
    ):
        # A robot that has already escaped the source Branch must not be
        # blended back toward it.  Turn it directly into the established
        # next-Branch stream.
        return get_lane_preserving_cohort_direction(
            robot,
            transfer_branch,
        ), 0.0

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
    if KHOP_CAPTURE_ENABLED:
        return pygame.Vector2()
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
        if (
            region == transfer_branch
            and pre_shepherd_branch == transfer_branch
            and pre_shepherd_pack_ready
            and branch_feed_continuity_deficit <= 0
        ):
            return pygame.Vector2()
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
            * (
                handoff_target_feed_scale()
                if region in {"JUNCTION", transfer_branch}
                else 1.0
            )
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
    """Smooth virtual-particle barrier at every unselected branch mouth."""
    if KHOP_CAPTURE_ENABLED:
        return pygame.Vector2()
    if phase not in {
        SimulationPhase.EXPLORE_BRANCH,
        SimulationPhase.FORM_SHEPHERD_BOUNDARY,
        SimulationPhase.FILL_BEHIND_SHEPHERD,
        SimulationPhase.PRESSURE_PUSH,
        SimulationPhase.FLOW_BACKTRACK,
        SimulationPhase.FINAL_JUNCTION_GATHER,
        SimulationPhase.RETURN_TO_BASE,
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
        gain = VIRTUAL_VALVE_GAIN * (
            FINAL_RETURN_VALVE_GAIN_MULTIPLIER
            if phase in {
                SimulationPhase.FINAL_JUNCTION_GATHER,
                SimulationPhase.RETURN_TO_BASE,
            }
            else 1.0
        )
        force += inward.normalize() * gain * ratio**2
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
    new_gate_states: dict[str, str]
    if open_branch is None:
        new_gate_states = {branch: "OPEN" for branch in BRANCHES}
    else:
        new_gate_states = {
            branch: "OPEN" if branch == open_branch else "CLOSED"
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
    global base_substitute_initial_base_count
    global base_substitute_reserve_target
    global base_junction_feed_robot_ids
    transfer_branch = target
    final_base_transfer_active = False
    base_junction_feed_robot_ids = set()
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
            else None
        )
    base_candidates = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and robot.transfer_target == target
        and get_robot_region(robot.position) == "BOTTOM"
    ]
    base_substitute_initial_base_count = len(base_candidates)
    base_substitute_reserve_target = base_substitute_initial_base_count
    junction_seed_count = sum(
        robot.transfer_target == target
        and get_robot_region(robot.position) == "JUNCTION"
        for robot in robots
    )
    print(
        f"[Cross-Branch Transfer] {source} -> {target}; "
        f"{source}=OPEN, {target}=OPEN, "
        f"junction_seed={junction_seed_count}, "
        f"base-candidates={base_substitute_initial_base_count}"
    )


def finish_cross_branch_transfer(robots, target: str) -> None:
    """End only the transient handoff after the target becomes active."""
    global transfer_branch
    global base_substitute_initial_base_count
    global base_substitute_reserve_target
    if transfer_branch != target:
        return
    cleared_count = 0
    for robot in robots:
        if robot.transfer_target != target:
            continue
        robot.transfer_target = None
        cleared_count += 1
    transfer_branch = None
    base_substitute_initial_base_count = 0
    base_substitute_reserve_target = 0
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
            "SHEPHERD_CANDIDATE",
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
    preserve_consensus_at_anchor(anchor, selected)
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
            branch_depth_from_junction(
                get_shepherd_fill_target(branch),
                branch,
            ),
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
    packed_conditions = density_packed_conditions
    if density_packed_conditions:
        tracker.recognition_mode = "PACKED_DENSITY"
        required_dwell = SATURATION_PACKED_DWELL_TIME
    elif geometry_packed_conditions:
        # Enough robots cover the curtain, but they have not yet stored
        # compression energy.  Keep filling instead of starting by count.
        tracker.recognition_mode = "GEOMETRY_WAIT_DENSITY"
        required_dwell = SATURATION_PACKED_DWELL_TIME
    elif stalled_conditions:
        tracker.recognition_mode = "STALLED_WAIT_DENSITY"
        required_dwell = SATURATION_DWELL_TIME
    else:
        tracker.recognition_mode = "WAIT"
        required_dwell = SATURATION_DWELL_TIME

    recognized = packed_conditions
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
        if robot.role in {
            "SHEPHERD",
            "SHEPHERD_CANDIDATE",
        }:
            robot.role = "NORMAL"
            robot.filtered_acceleration.update(0.0, 0.0)
            robot.shepherd_anchor = None
            robot.shepherd_origin = None
            robot.shepherd_branch = None


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
            "SHEPHERD_CANDIDATE",
        }:
            robot.role = "NORMAL"
            robot.shepherd_anchor = None
            robot.shepherd_origin = None
            robot.shepherd_branch = None
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
    """Return leaders in the upstream band before the dead-end standoff."""
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

    # The physical co-location inside the upstream standoff capture band is
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


def reserve_shepherd_slots_progressively(
    robots,
    branch: str,
    reservation_role: str,
) -> tuple[list["Robot"], int]:
    """Reserve open slots as individual leaders enter the approach band."""
    slots = build_shepherd_slots(
        branch,
        adaptive_shepherd_count(),
    )
    reserved = [
        robot
        for robot in robots
        if robot.role == reservation_role
        and robot.shepherd_branch == branch
        and robot.shepherd_anchor is not None
    ]
    occupied_slots = set()
    for robot in reserved:
        slot_index = min(
            range(len(slots)),
            key=lambda index: robot.shepherd_anchor.distance_squared_to(
                slots[index]
            ),
        )
        occupied_slots.add(slot_index)

    open_slots = [
        index
        for index in range(len(slots))
        if index not in occupied_slots
    ]
    capture_rect = early_capture_regions[branch]
    available = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and get_robot_region(robot.position) == branch
        and capture_rect.collidepoint(
            robot.position.x,
            robot.position.y,
        )
    ]
    new_count = 0
    while open_slots and available:
        _, _, slot_index, robot = min(
            (
                robot.position.distance_squared_to(slots[index]),
                robot.robot_id,
                index,
                robot,
            )
            for robot in available
            for index in open_slots
        )
        slot = slots[slot_index]
        robot.role = reservation_role
        robot.shepherd_anchor = slot.copy()
        robot.shepherd_origin = slot.copy()
        robot.shepherd_branch = branch
        robot.velocity.update(0.0, 0.0)
        robot.acceleration.update(0.0, 0.0)
        robot.filtered_acceleration.update(0.0, 0.0)
        reserved.append(robot)
        available.remove(robot)
        open_slots.remove(slot_index)
        new_count += 1
        print(
            f"[Shepherd Slot] branch={branch}, "
            f"slot={slot_index}, robot={robot.robot_id}, "
            f"reserved={len(reserved)}/{len(slots)}"
        )
    return reserved, new_count


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
    """Progressively reserve next-branch slots during the current return."""
    del grid
    global pre_shepherd_branch
    global pre_shepherd_pack_dwell, pre_shepherd_pack_ready
    if (
        pre_shepherd_branch is not None
        and pre_shepherd_branch != branch
    ):
        return get_pre_shepherds(robots, branch)
    pre_shepherd_branch = branch
    selected, new_count = reserve_shepherd_slots_progressively(
        robots,
        branch,
        "PRE_SHEPHERD",
    )
    if new_count > 0:
        pre_shepherd_pack_dwell = 0.0
        pre_shepherd_pack_ready = False
    enforce_pre_shepherd_curtain_for_swarm(robots)
    if len(selected) == adaptive_shepherd_count():
        print(
            f"[Pre-Shepherd] all slots reserved: "
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
        and average_density_ratio
        >= SATURATION_PACKED_DENSITY_RATIO
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


def promote_pre_shepherds(
    robots,
    branch: str,
    require_packed: bool = True,
) -> bool:
    """Activate a prepared shield when its slots are physically complete.

    A shield prepared during the previous Branch may be promoted immediately
    once its Branch becomes active.  Its ordinary robots must then repeat the
    same FILL/packed-density sequence as the first Branch; requiring density
    before promotion creates a deadlock because packed spacing is enabled only
    after the Shepherd role and FILL phase are active.
    """
    global pre_shepherd_branch
    global pre_shepherd_pack_dwell, pre_shepherd_pack_ready
    if (
        not pre_shepherd_boundary_formed(robots, branch)
        or require_packed
        and not pre_shepherd_pack_ready
    ):
        return False
    for robot in get_pre_shepherds(robots, branch):
        robot.role = "SHEPHERD"
        robot.velocity.update(0.0, 0.0)
        robot.acceleration.update(0.0, 0.0)
        robot.filtered_acceleration.update(0.0, 0.0)
    pre_shepherd_branch = None
    pre_shepherd_pack_dwell = 0.0
    pre_shepherd_pack_ready = False
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
    if len(existing) < adaptive_shepherd_count():
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
    if robot.role != "NORMAL" or not base_reserve_is_mobile(robot):
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
        or not base_reserve_is_mobile(robot)
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
    if KHOP_CAPTURE_ENABLED:
        if robot.role in {"KHOP_LEADER", "KHOP_SHEPHERD"}:
            return KHOP_CAP_EQUILIBRIUM_DISTANCE
        if (
            khop_state.stage == "RETURNING_TO_BASE"
            and robot.role == "NORMAL"
        ):
            # The Base rectangle physically cannot hold the whole swarm at
            # ordinary flow spacing; without compacting, robots jam at the
            # entrance and the walk home can never reach the ready ratio.
            return max(
                ROBOT_RADIUS * 2.05,
                SAFE_RADIUS * RETURN_PACKED_EQUILIBRIUM_SCALE,
            )
        return SAFE_RADIUS * NORMAL_EQUILIBRIUM_SCALE
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
    if KHOP_CAPTURE_ENABLED:
        return pygame.Vector2()
    """Damped elastic pressure wall at the three initially closed faces."""
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
        if KHOP_CAPTURE_ENABLED:
            if robot.role in KHOP_DYNAMIC_ROLES:
                if (
                    khop_state.stage == "RETURNING_TO_BASE"
                    and robot.role == "NORMAL"
                ):
                    # Match the tighter RETURNING_TO_BASE equilibrium radius
                    # with a gentler pressure response, or the full motion
                    # boost fights the compaction and nobody can pack in.
                    robot.pressure *= RETURN_PACKING_PRESSURE_SCALE
                else:
                    robot.pressure *= SPH_MOTION_PRESSURE_BOOST
            continue
        if (
            robot.role == "NORMAL"
            or KHOP_CAPTURE_ENABLED
            and robot.role in KHOP_DYNAMIC_ROLES
        ):
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
                    base_reserve_is_mobile(robot)
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


def measure_branch_feed_continuity(
    robots,
    branch: str,
    maximum_depth: Optional[float] = None,
) -> tuple[int, list[int], list[float]]:
    """Return the NORMAL mass missing from each 2-D Branch cross-section.

    Branch quota alone cannot see a sparse middle band: a dense Shepherd-side
    pile can make the total count exceed quota while one or more upstream
    slices remain nearly empty.  Measure both axial population and lateral
    coverage so Base feeding continues until the entire Junction-to-Shepherd
    body is spatially continuous.
    """
    if maximum_depth is None:
        maximum_depth = branch_depth_from_junction(
            get_shepherd_fill_target(branch),
            branch,
        )
    maximum_depth = max(
        BRANCH_CONTINUITY_SLICE_DEPTH,
        maximum_depth,
    )
    slice_count = max(
        1,
        math.ceil(
            maximum_depth / BRANCH_CONTINUITY_SLICE_DEPTH
        ),
    )
    counts = [0 for _ in range(slice_count)]
    occupied_lateral_bins = [
        set() for _ in range(slice_count)
    ]
    branch_direction = BRANCH_DIRECTIONS[branch]
    lateral_direction = pygame.Vector2(
        -branch_direction.y,
        branch_direction.x,
    )
    junction_center = pygame.Vector2(center_x, center_y)

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
        slice_index = min(
            slice_count - 1,
            int(depth / BRANCH_CONTINUITY_SLICE_DEPTH),
        )
        counts[slice_index] += 1
        lateral_position = (
            robot.position - junction_center
        ).dot(lateral_direction)
        normalized_lateral = clamp(
            (lateral_position + half_width)
            / max(corridor_width, EPSILON),
            0.0,
            1.0 - EPSILON,
        )
        lateral_index = min(
            BRANCH_CONTINUITY_LATERAL_SLICES - 1,
            int(
                normalized_lateral
                * BRANCH_CONTINUITY_LATERAL_SLICES
            ),
        )
        occupied_lateral_bins[slice_index].add(
            lateral_index
        )

    lateral_coverages = [
        len(occupied) / BRANCH_CONTINUITY_LATERAL_SLICES
        for occupied in occupied_lateral_bins
    ]
    count_deficit = sum(
        max(0, BRANCH_CONTINUITY_MIN_ROBOTS_PER_SLICE - count)
        for count in counts
    )
    coverage_deficit = sum(
        math.ceil(
            max(
                0.0,
                BRANCH_CONTINUITY_MIN_LATERAL_COVERAGE - coverage,
            )
            * BRANCH_CONTINUITY_MIN_ROBOTS_PER_SLICE
        )
        for coverage in lateral_coverages
    )
    return (
        max(count_deficit, coverage_deficit),
        counts,
        lateral_coverages,
    )


def measure_base_junction_interface(
    robots,
) -> tuple[float, int, list[int], list[float]]:
    """Measure every axial/lateral cross-section along the whole Base trunk.

    A nearest-particle test is insufficient: one sparse chain can touch J0
    while the rest of the Base cross-section is visibly empty.  Divide the
    whole Base corridor into 2-D slices so isolated particles no longer hide
    a broad void and the body remains connected down to the Base station.
    """
    base_robots = [
        robot
        for robot in robots
        if robot.role == "NORMAL"
        and get_robot_region(robot.position) == "BOTTOM"
    ]
    if not base_robots:
        # An empty Base is a maximum coverage deficit, not a completed seam.
        # Surplus Junction robots will be assigned back into these empty slices.
        empty_counts = [
            0 for _ in range(BASE_JUNCTION_INTERFACE_AXIAL_SLICES)
        ]
        empty_coverages = [
            0.0 for _ in range(BASE_JUNCTION_INTERFACE_AXIAL_SLICES)
        ]
        full_deficit = (
            BASE_JUNCTION_INTERFACE_AXIAL_SLICES
            * BASE_JUNCTION_INTERFACE_MIN_ROBOTS_PER_SLICE
        )
        return (
            BASE_JUNCTION_INTERFACE_FRONT_DEPTH,
            full_deficit,
            empty_counts,
            empty_coverages,
        )

    counts = [
        0 for _ in range(BASE_JUNCTION_INTERFACE_AXIAL_SLICES)
    ]
    occupied_lateral_bins = [
        set() for _ in range(BASE_JUNCTION_INTERFACE_AXIAL_SLICES)
    ]
    axial_depths = []
    for robot in base_robots:
        axial_depth = robot.position.y - bottom_rect.top
        if not 0.0 <= axial_depth <= BASE_JUNCTION_INTERFACE_FRONT_DEPTH:
            continue
        axial_depths.append(axial_depth)
        normalized_axial = clamp(
            axial_depth
            / max(BASE_JUNCTION_INTERFACE_FRONT_DEPTH, EPSILON),
            0.0,
            1.0 - EPSILON,
        )
        slice_index = min(
            BASE_JUNCTION_INTERFACE_AXIAL_SLICES - 1,
            int(
                normalized_axial
                * BASE_JUNCTION_INTERFACE_AXIAL_SLICES
            ),
        )
        counts[slice_index] += 1
        normalized_lateral = clamp(
            (robot.position.x - bottom_rect.left)
            / max(bottom_rect.width, EPSILON),
            0.0,
            1.0 - EPSILON,
        )
        occupied_lateral_bins[slice_index].add(
            min(
                JUNCTION_BUFFER_LATERAL_SLICES - 1,
                int(
                    normalized_lateral
                    * JUNCTION_BUFFER_LATERAL_SLICES
                ),
            )
        )

    lateral_coverages = [
        len(occupied) / JUNCTION_BUFFER_LATERAL_SLICES
        for occupied in occupied_lateral_bins
    ]
    ordered_depths = [0.0] + sorted(axial_depths) + [
        BASE_JUNCTION_INTERFACE_FRONT_DEPTH
    ]
    interface_gap = max(
        (
            after - before
            for before, after in zip(
                ordered_depths,
                ordered_depths[1:],
            )
        ),
        default=BASE_JUNCTION_INTERFACE_FRONT_DEPTH,
    )
    count_deficit = sum(
        max(
            0,
            BASE_JUNCTION_INTERFACE_MIN_ROBOTS_PER_SLICE - count,
        )
        for count in counts
    )
    lateral_deficit = sum(
        math.ceil(
            max(
                0.0,
                JUNCTION_BUFFER_MIN_LATERAL_COVERAGE - coverage,
            )
            * BASE_JUNCTION_INTERFACE_MIN_ROBOTS_PER_SLICE
        )
        for coverage in lateral_coverages
    )
    return (
        interface_gap,
        max(count_deficit, lateral_deficit),
        counts,
        lateral_coverages,
    )


def update_junction_buffer_control(
    robots,
    handoff_active: bool,
) -> None:
    """Recruit only enough Base mass to keep J0 spatially continuous."""
    global junction_buffer_control
    global junction_buffer_slice_counts
    global junction_buffer_lateral_coverage
    global base_substitute_count
    global base_substitute_robot_ids
    global base_substitute_initial_base_count
    global base_substitute_reserve_target
    global base_junction_feed_robot_ids
    global base_junction_feed_branch
    global base_junction_feed_initial_count
    global base_junction_feed_reserve_target
    global branch_feed_continuity_deficit
    global branch_feed_slice_counts
    global branch_feed_slice_coverages
    global base_junction_interface_gap
    global base_junction_interface_deficit
    global base_junction_interface_slice_counts
    global base_junction_interface_slice_coverages
    global base_tail_refill_robot_ids

    ordinary_feed_active = phase in {
        SimulationPhase.EXPLORE_BRANCH,
        SimulationPhase.FORM_SHEPHERD_BOUNDARY,
        SimulationPhase.FILL_BEHIND_SHEPHERD,
    }
    if not handoff_active and not ordinary_feed_active:
        junction_buffer_control = 0.0
        junction_buffer_slice_counts = []
        junction_buffer_lateral_coverage = []
        base_substitute_count = 0
        base_substitute_robot_ids = set()
        base_junction_feed_robot_ids = set()
        base_junction_feed_branch = None
        base_junction_feed_initial_count = 0
        base_junction_feed_reserve_target = 0
        branch_feed_continuity_deficit = 0
        branch_feed_slice_counts = []
        branch_feed_slice_coverages = []
        base_junction_interface_gap = 0.0
        base_junction_interface_deficit = 0
        base_junction_interface_slice_counts = []
        base_junction_interface_slice_coverages = []
        base_tail_refill_robot_ids = set()
        return

    target_branch = transfer_branch if handoff_active else active_branch
    if target_branch is None:
        branch_feed_continuity_deficit = 0
        branch_feed_slice_counts = []
        branch_feed_slice_coverages = []
        base_junction_interface_gap = 0.0
        base_junction_interface_deficit = 0
        base_junction_interface_slice_counts = []
        base_junction_interface_slice_coverages = []
        base_tail_refill_robot_ids = set()
        return
    (
        branch_feed_continuity_deficit,
        branch_feed_slice_counts,
        branch_feed_slice_coverages,
    ) = measure_branch_feed_continuity(
        robots,
        target_branch,
    )
    (
        base_junction_interface_gap,
        base_junction_interface_deficit,
        base_junction_interface_slice_counts,
        base_junction_interface_slice_coverages,
    ) = measure_base_junction_interface(robots)
    target_direction = BRANCH_DIRECTIONS[target_branch]
    junction_center = pygame.Vector2(center_x, center_y)
    counts = [
        0 for _ in range(JUNCTION_BUFFER_AXIAL_SLICES)
    ]
    occupied_lateral_bins = [
        set() for _ in range(JUNCTION_BUFFER_AXIAL_SLICES)
    ]
    lateral_direction = pygame.Vector2(
        -target_direction.y,
        target_direction.x,
    )
    for robot in robots:
        if (
            robot.role != "NORMAL"
            or get_robot_region(robot.position) != "JUNCTION"
        ):
            continue
        axial_position = (
            robot.position - junction_center
        ).dot(target_direction)
        normalized_axial = clamp(
            (
                axial_position + half_width
            )
            / max(corridor_width, EPSILON),
            0.0,
            1.0 - EPSILON,
        )
        slice_index = min(
            JUNCTION_BUFFER_AXIAL_SLICES - 1,
            int(
                normalized_axial
                * JUNCTION_BUFFER_AXIAL_SLICES
            ),
        )
        counts[slice_index] += 1
        lateral_position = (
            robot.position - junction_center
        ).dot(lateral_direction)
        normalized_lateral = clamp(
            (
                lateral_position + half_width
            )
            / max(corridor_width, EPSILON),
            0.0,
            1.0 - EPSILON,
        )
        lateral_index = min(
            JUNCTION_BUFFER_LATERAL_SLICES - 1,
            int(
                normalized_lateral
                * JUNCTION_BUFFER_LATERAL_SLICES
            ),
        )
        occupied_lateral_bins[slice_index].add(
            lateral_index
        )

    lateral_coverage = [
        len(occupied) / JUNCTION_BUFFER_LATERAL_SLICES
        for occupied in occupied_lateral_bins
    ]

    raw_control = max(
        (
            deficit
            for count, coverage in zip(
                counts,
                lateral_coverage,
            )
            for deficit in (
                clamp(
                    1.0
                    - count
                    / max(
                        JUNCTION_BUFFER_MIN_ROBOTS_PER_SLICE,
                        1,
                    ),
                    0.0,
                    1.0,
                ),
                clamp(
                    1.0
                    - coverage
                    / max(
                        JUNCTION_BUFFER_MIN_LATERAL_COVERAGE,
                        EPSILON,
                    ),
                    0.0,
                    1.0,
                ),
            )
        ),
        default=1.0,
    )
    if raw_control >= junction_buffer_control:
        # An opening hole must recruit Base immediately.  Filtering the rising
        # edge made the old substitute force arrive only after the Junction had
        # already emptied.
        junction_buffer_control = raw_control
    else:
        # Release the priming cohort smoothly after the backtracking stream has
        # restored every axial/lateral Junction slice.
        junction_buffer_control = (
            junction_buffer_control
            * (1.0 - BASE_SUBSTITUTE_RELEASE_FILTER_ALPHA)
            + raw_control * BASE_SUBSTITUTE_RELEASE_FILTER_ALPHA
        )
    junction_buffer_slice_counts = counts
    junction_buffer_lateral_coverage = lateral_coverage
    junction_normals = sorted(
        (
            robot
            for robot in robots
            if robot.role == "NORMAL"
            and base_reserve_is_mobile(robot)
            and get_robot_region(robot.position) == "JUNCTION"
        ),
        key=lambda robot: (
            -robot.position.y,
            abs(robot.position.x - center_x),
            robot.robot_id,
        ),
    )
    retained_base_refill = sorted(
        (
            robot
            for robot in robots
            if robot.robot_id in base_tail_refill_robot_ids
            and robot.role == "NORMAL"
            and get_robot_region(robot.position) == "BOTTOM"
        ),
        key=lambda robot: robot.robot_id,
    )
    minimum_junction_body = (
        JUNCTION_BUFFER_AXIAL_SLICES
        * JUNCTION_BUFFER_MIN_ROBOTS_PER_SLICE
    )
    junction_surplus = max(
        0,
        len(junction_normals) - minimum_junction_body,
    )
    base_refill_target = min(
        base_junction_interface_deficit,
        len(retained_base_refill) + junction_surplus,
    )
    retained_base_refill = retained_base_refill[:base_refill_target]
    new_base_refill_count = max(
        0,
        base_refill_target - len(retained_base_refill),
    )
    base_tail_refill_robot_ids = {
        robot.robot_id
        for robot in (
            retained_base_refill
            + junction_normals[:new_base_refill_count]
        )
    }
    bottom_candidates = sorted(
        (
            robot
            for robot in robots
            if robot.role == "NORMAL"
            and robot.robot_id not in base_tail_refill_robot_ids
            and (
                not handoff_active
                or robot.transfer_target == target_branch
            )
            and get_robot_region(robot.position) == "BOTTOM"
        ),
        key=lambda robot: (
            robot.position.y,
            abs(robot.position.x - center_x),
            robot.robot_id,
        ),
    )
    count_deficit = sum(
        max(0, JUNCTION_BUFFER_MIN_ROBOTS_PER_SLICE - count)
        for count in counts
    )
    coverage_deficit = sum(
        math.ceil(
            max(
                0.0,
                JUNCTION_BUFFER_MIN_LATERAL_COVERAGE - coverage,
            )
            * JUNCTION_BUFFER_MIN_ROBOTS_PER_SLICE
        )
        for coverage in lateral_coverage
    )
    spatial_deficit = max(count_deficit, coverage_deficit)
    branch_quota_deficit = max(
        0,
        branch_fill_target_count - branch_fill_current_count,
    )
    dispatch_demand = min(
        BASE_SUBSTITUTE_MAX_ACTIVE,
        max(
            spatial_deficit,
            branch_quota_deficit,
            branch_feed_continuity_deficit,
        )
        + base_junction_interface_deficit
        + math.ceil(
            JUNCTION_BUFFER_TRANSIT_HEADROOM
            * smoothstep01(junction_buffer_control)
        ),
    )
    previous_ids = (
        base_substitute_robot_ids
        if handoff_active
        else base_junction_feed_robot_ids
    )
    junction_inflight_ids = {
        robot.robot_id
        for robot in robots
        if robot.robot_id in previous_ids
        and robot.role == "NORMAL"
        and get_robot_region(robot.position) == "JUNCTION"
    }
    # Maintain a live Base refill cohort independently of robots already in
    # J0.  Subtracting Junction residents from this count caused a pulse: the
    # residents immediately left for the Branch while no replacement was
    # recruited from Base, leaving the Junction visibly empty.
    new_dispatch_count = min(
        len(bottom_candidates),
        dispatch_demand,
    )
    selected_ids = junction_inflight_ids | {
        robot.robot_id
        for robot in bottom_candidates[:new_dispatch_count]
    }
    dispatch_count = len(selected_ids)
    if handoff_active:
        base_substitute_robot_ids = selected_ids
        base_substitute_count = len(selected_ids)
        base_substitute_initial_base_count = len(bottom_candidates)
        base_substitute_reserve_target = (
            len(bottom_candidates) - new_dispatch_count
        )
        base_junction_feed_robot_ids = set()
        base_junction_feed_branch = None
        base_junction_feed_initial_count = 0
        base_junction_feed_reserve_target = 0
    else:
        base_junction_feed_robot_ids = selected_ids
        base_junction_feed_branch = target_branch
        base_junction_feed_initial_count = len(bottom_candidates)
        base_junction_feed_reserve_target = (
            len(bottom_candidates) - new_dispatch_count
        )
        base_substitute_robot_ids = set()
        base_substitute_count = 0


def compute_base_to_junction_force(
    robot: "Robot",
) -> pygame.Vector2:
    """Lift the currently required upper Base cohort to the Junction mouth.

    Cohort size follows the measured Junction deficit.  Unselected robots
    naturally remain in Base, while selected robots preserve their original
    lanes so the connection rises as a broad SPH column instead of a stream.
    """
    if (
        get_robot_region(robot.position) != "BOTTOM"
        or robot.robot_id not in base_junction_feed_robot_ids
    ):
        return pygame.Vector2()
    wall_clearance = max(robot.radius * 2.0, 2.0 * MAP_SCALE)
    lane_x = clamp(
        robot.ingress_lane_x,
        bottom_rect.left + wall_clearance,
        bottom_rect.right - wall_clearance,
    )
    target = pygame.Vector2(
        lane_x,
        junction_rect.bottom - corridor_width * 0.22,
    )
    direction = normalized_direction_toward(robot.position, target)
    desired_velocity = direction * BASE_TO_JUNCTION_FEED_SPEED
    velocity_alignment = (
        desired_velocity - robot.velocity
    ) * BASE_TO_JUNCTION_VELOCITY_GAIN
    return limit_vector(
        direction * OUTLET_FORCE * FILL_BASE_FEED_MULTIPLIER
        + velocity_alignment,
        BASE_TO_JUNCTION_FORCE_LIMIT,
    )


def compute_base_substitute_force(
    robot: "Robot",
) -> pygame.Vector2:
    """Send the reserved Base-first cohort directly into the next Branch."""
    if (
        transfer_branch is None
        or get_robot_region(robot.position) != "BOTTOM"
        or robot.transfer_target != transfer_branch
        or robot.robot_id not in base_substitute_robot_ids
    ):
        return pygame.Vector2()
    branch_need_control = clamp(
        (
            branch_fill_target_count - branch_fill_current_count
        )
        / max(branch_fill_target_count, 1),
        0.0,
        1.0,
    )
    drive_signal = max(
        junction_buffer_control,
        branch_need_control,
    )
    drive_control = (
        BASE_SUBSTITUTE_DRIVE_FLOOR
        + (1.0 - BASE_SUBSTITUTE_DRIVE_FLOOR)
        * smoothstep01(drive_signal)
    )
    merge_direction = get_lane_preserving_cohort_direction(
        robot,
        transfer_branch,
    )
    desired_velocity = (
        merge_direction
        * BASE_SUBSTITUTE_MERGE_SPEED
        * drive_control
    )
    velocity_alignment = (
        desired_velocity - robot.velocity
    ) * BASE_SUBSTITUTE_VELOCITY_GAIN
    return limit_vector(
        (
            merge_direction
            * FLOW_BACKTRACK_FORCE
            * BASE_SUBSTITUTE_FORCE_MULTIPLIER
            * drive_control
            + velocity_alignment
        ),
        BASE_SUBSTITUTE_FORCE_LIMIT,
    )


def junction_buffer_spatially_ready() -> bool:
    """Require every J0 axial slice and its width before Branch release."""
    return (
        len(junction_buffer_slice_counts)
        == JUNCTION_BUFFER_AXIAL_SLICES
        and len(junction_buffer_lateral_coverage)
        == JUNCTION_BUFFER_AXIAL_SLICES
        and all(
            count >= JUNCTION_BUFFER_MIN_ROBOTS_PER_SLICE
            and coverage >= JUNCTION_BUFFER_MIN_LATERAL_COVERAGE
            for count, coverage in zip(
                junction_buffer_slice_counts,
                junction_buffer_lateral_coverage,
            )
        )
    )


def compute_base_tail_refill_force(
    robot: "Robot",
) -> pygame.Vector2:
    """Expand measured J0 surplus backward into empty upper-Base slices."""
    slice_index = (
        robot.robot_id % BASE_JUNCTION_INTERFACE_AXIAL_SLICES
    )
    axial_depth = (
        slice_index + 0.5
    ) * (
        BASE_JUNCTION_INTERFACE_FRONT_DEPTH
        / BASE_JUNCTION_INTERFACE_AXIAL_SLICES
    )
    wall_clearance = max(robot.radius * 2.0, 2.0 * MAP_SCALE)
    lane_x = clamp(
        robot.ingress_lane_x,
        bottom_rect.left + wall_clearance,
        bottom_rect.right - wall_clearance,
    )
    target = pygame.Vector2(
        lane_x,
        bottom_rect.top + axial_depth,
    )
    direction = normalized_direction_toward(
        robot.position,
        target,
    )
    distance = robot.position.distance_to(target)
    approach_scale = smoothstep01(
        distance
        / max(
            BASE_JUNCTION_INTERFACE_FRONT_DEPTH
            / BASE_JUNCTION_INTERFACE_AXIAL_SLICES,
            EPSILON,
        )
    )
    desired_velocity = (
        direction * BASE_TAIL_REFILL_SPEED * approach_scale
    )
    velocity_alignment = (
        desired_velocity - robot.velocity
    ) * BASE_TAIL_REFILL_VELOCITY_GAIN
    return limit_vector(
        direction
        * OUTLET_FORCE
        * BASE_TAIL_REFILL_FORCE_MULTIPLIER
        * approach_scale
        + velocity_alignment,
        BASE_TO_JUNCTION_FORCE_LIMIT,
    )


def compute_junction_refill_force(
    robot: "Robot",
    branch: str,
) -> pygame.Vector2:
    """Spread J0 robots into deficient slices instead of passing them through."""
    deficient_slices = [
        index
        for index, (count, coverage) in enumerate(
            zip(
                junction_buffer_slice_counts,
                junction_buffer_lateral_coverage,
            )
        )
        if count < JUNCTION_BUFFER_MIN_ROBOTS_PER_SLICE
        or coverage < JUNCTION_BUFFER_MIN_LATERAL_COVERAGE
    ]
    if not deficient_slices:
        return limit_vector(
            -robot.velocity * BASE_TO_JUNCTION_HOLD_DAMPING,
            BASE_TO_JUNCTION_FORCE_LIMIT,
        )
    # Distribute the cohort across every deficient cross-section.  The original
    # Base ingress lane becomes the lateral coordinate after the 90-degree turn.
    slice_index = deficient_slices[
        robot.robot_id % len(deficient_slices)
    ]
    branch_direction = BRANCH_DIRECTIONS[branch]
    lateral_direction = pygame.Vector2(
        -branch_direction.y,
        branch_direction.x,
    )
    axial_position = (
        -half_width
        + (
            slice_index + 0.5
        )
        * corridor_width
        / JUNCTION_BUFFER_AXIAL_SLICES
    )
    wall_clearance = max(robot.radius * 2.0, 2.0 * MAP_SCALE)
    lateral_position = clamp(
        get_cohort_lane_offset(robot),
        -half_width + wall_clearance,
        half_width - wall_clearance,
    )
    target = (
        pygame.Vector2(center_x, center_y)
        + branch_direction * axial_position
        + lateral_direction * lateral_position
    )
    direction = normalized_direction_toward(
        robot.position,
        target,
    )
    distance = robot.position.distance_to(target)
    approach_scale = smoothstep01(
        distance
        / max(
            corridor_width / JUNCTION_BUFFER_AXIAL_SLICES,
            EPSILON,
        )
    )
    desired_velocity = (
        direction * JUNCTION_REFILL_SPEED * approach_scale
    )
    velocity_alignment = (
        desired_velocity - robot.velocity
    ) * JUNCTION_REFILL_VELOCITY_GAIN
    return limit_vector(
        direction
        * OUTLET_FORCE
        * JUNCTION_REFILL_FORCE_MULTIPLIER
        * approach_scale
        + velocity_alignment,
        BASE_TO_JUNCTION_FORCE_LIMIT,
    )


def compute_base_junction_feed_override(
    robot: "Robot",
) -> Optional[pygame.Vector2]:
    """Fill every J0 slice first, then release surplus into the Branch."""
    if (
        phase not in {
            SimulationPhase.EXPLORE_BRANCH,
            SimulationPhase.FORM_SHEPHERD_BOUNDARY,
            SimulationPhase.FILL_BEHIND_SHEPHERD,
        }
        or base_junction_feed_branch != active_branch
    ):
        return None
    region = get_robot_region(robot.position)
    if (
        robot.robot_id in base_tail_refill_robot_ids
        and region in {"JUNCTION", "BOTTOM"}
    ):
        return compute_base_tail_refill_force(robot)
    if (
        region == "JUNCTION"
        and robot.role == "NORMAL"
        and not junction_buffer_spatially_ready()
    ):
        return compute_junction_refill_force(
            robot,
            active_branch,
        )
    if robot.robot_id not in base_junction_feed_robot_ids:
        return None
    if region == "BOTTOM":
        return compute_base_to_junction_force(robot)
    if (
        region == "JUNCTION"
        and branch_feed_continuity_deficit > 0
    ):
        merge_direction = get_lane_preserving_cohort_direction(
            robot,
            active_branch,
            get_shepherd_fill_target(active_branch),
        )
        desired_velocity = (
            merge_direction * BASE_TO_JUNCTION_FEED_SPEED
        )
        velocity_alignment = (
            desired_velocity - robot.velocity
        ) * BASE_TO_JUNCTION_VELOCITY_GAIN
        return limit_vector(
            merge_direction
            * OUTLET_FORCE
            * FILL_JUNCTION_FEED_MULTIPLIER
            + velocity_alignment,
            BASE_TO_JUNCTION_FORCE_LIMIT,
        )
    if (
        region == "JUNCTION"
        and branch_fill_target_count > 0
        and branch_fill_current_count >= branch_fill_target_count
    ):
        return limit_vector(
            -robot.velocity * BASE_TO_JUNCTION_HOLD_DAMPING,
            BASE_TO_JUNCTION_FORCE_LIMIT,
        )
    return None


def update_transfer_continuity_control(robots) -> None:
    """Balance source recovery against target advance and mouth occupancy."""
    global transfer_path_max_gap
    global transfer_entrance_count
    global transfer_gap_control
    global transfer_target_motion_scale
    global branch_fill_target_count
    global branch_fill_current_count
    global branch_fill_deficit_control
    global branch_overflow_return_robot_ids
    global junction_buffer_control
    global junction_buffer_slice_counts
    global junction_buffer_lateral_coverage
    global base_substitute_count
    global handoff_source_continuity_deficit
    global handoff_source_slice_counts
    global handoff_source_slice_coverages
    global handoff_source_continuity_control

    update_hose_continuity_forces(robots)
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
    update_junction_buffer_control(
        robots,
        handoff_active,
    )
    quota_control_active = quota_feed_active or handoff_active
    if quota_control_active:
        quota_branch = (
            transfer_branch
            if handoff_active
            else active_branch
        )
        branch_fill_target_count = calculate_branch_fill_quota(
            robots,
            quota_branch,
        )
        branch_members = [
            robot
            for robot in robots
            if robot.role == "NORMAL"
            and get_robot_region(robot.position) == quota_branch
        ]
        branch_fill_current_count = len(branch_members)
        overflow_hysteresis = max(
            BRANCH_OVERFLOW_MIN_HYSTERESIS,
            math.ceil(
                branch_fill_target_count
                * BRANCH_OVERFLOW_HYSTERESIS_RATIO
            ),
        )
        overflow_count = max(
            0,
            branch_fill_current_count
            - branch_fill_target_count
            - overflow_hysteresis,
        )
        if filling_active and quota_branch == active_branch:
            branch_overflow_return_robot_ids = {
                robot.robot_id
                for robot in sorted(
                    branch_members,
                    key=lambda candidate: (
                        branch_depth_from_junction(
                            candidate.position,
                            active_branch,
                        ),
                        candidate.robot_id,
                    ),
                )[:overflow_count]
            }
        else:
            branch_overflow_return_robot_ids = set()
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
        branch_overflow_return_robot_ids = set()

    if not (filling_active or handoff_active):
        transfer_path_max_gap = 0.0
        transfer_entrance_count = 0
        transfer_gap_control = 0.0
        transfer_target_motion_scale = 1.0
        handoff_source_continuity_deficit = 0
        handoff_source_slice_counts = []
        handoff_source_slice_coverages = []
        handoff_source_continuity_control = 0.0
        return

    path_regions = {"BOTTOM", "JUNCTION", active_branch}
    path_progress = sorted(
        relay_path_progress(robot.position, active_branch)
        for robot in robots
        if (
            robot.role == "NORMAL"
            and base_reserve_is_mobile(robot)
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
            and base_reserve_is_mobile(robot)
            and get_robot_region(robot.position) == active_branch
            and branch_depth_from_junction(
                robot.position,
                active_branch,
            )
            <= TRANSFER_CONTINUITY_ENTRANCE_DEPTH
        )
    ]
    transfer_entrance_count = len(entrance_robots)
    if handoff_active:
        source_depths = [
            branch_depth_from_junction(
                robot.position,
                active_branch,
            )
            for robot in robots
            if robot.role == "NORMAL"
            and base_reserve_is_mobile(robot)
            and get_robot_region(robot.position) == active_branch
        ]
        if source_depths:
            (
                handoff_source_continuity_deficit,
                handoff_source_slice_counts,
                handoff_source_slice_coverages,
            ) = measure_branch_feed_continuity(
                robots,
                active_branch,
                max(source_depths),
            )
            handoff_source_continuity_control = max(
                (
                    deficit
                    for count, coverage in zip(
                        handoff_source_slice_counts,
                        handoff_source_slice_coverages,
                    )
                    for deficit in (
                        clamp(
                            1.0
                            - count
                            / max(
                                BRANCH_CONTINUITY_MIN_ROBOTS_PER_SLICE,
                                1,
                            ),
                            0.0,
                            1.0,
                        ),
                        clamp(
                            1.0
                            - coverage
                            / max(
                                BRANCH_CONTINUITY_MIN_LATERAL_COVERAGE,
                                EPSILON,
                            ),
                            0.0,
                            1.0,
                        ),
                    )
                ),
                default=0.0,
            )
        else:
            handoff_source_continuity_deficit = 0
            handoff_source_slice_counts = []
            handoff_source_slice_coverages = []
            handoff_source_continuity_control = 0.0
    else:
        handoff_source_continuity_deficit = 0
        handoff_source_slice_counts = []
        handoff_source_slice_coverages = []
        handoff_source_continuity_control = 0.0
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
    base_interface_control = clamp(
        (
            base_junction_interface_gap
            - BASE_JUNCTION_INTERFACE_TARGET_GAP
        )
        / max(BASE_JUNCTION_INTERFACE_TARGET_GAP, EPSILON),
        0.0,
        1.0,
    )
    base_interface_density_control = clamp(
        base_junction_interface_deficit
        / max(
            BASE_JUNCTION_INTERFACE_AXIAL_SLICES
            * BASE_JUNCTION_INTERFACE_MIN_ROBOTS_PER_SLICE,
            1,
        ),
        0.0,
        1.0,
    )
    raw_control = max(
        gap_control,
        entrance_control,
        base_interface_control,
        base_interface_density_control,
        handoff_source_continuity_control,
    )
    if raw_control >= transfer_gap_control:
        # A newly opened seam must brake the target immediately; filtering the
        # rising edge lets the target pull away before the source can catch it.
        transfer_gap_control = raw_control
    else:
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
    if KHOP_CAPTURE_ENABLED:
        return compute_khop_route_force(robot)
    region = get_robot_region(robot.position)
    junction_target = pygame.Vector2(center_x, center_y)
    force = pygame.Vector2()
    if robot.role in {"ANCHOR", "RELAY", "TRUNK_RELAY"}:
        return force
    if (
        robot.base_reserve
        and not base_reserve_is_mobile(robot)
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

    base_junction_override = compute_base_junction_feed_override(robot)
    if base_junction_override is not None:
        return base_junction_override

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
            and region not in {"BOTTOM", "JUNCTION"}
        ):
            # Base-side robots receive no artificial suction. With no
            # Shepherd behind them, the low-density tail remains near Base.
            force = pygame.Vector2()
        elif region == "BOTTOM":
            force = compute_base_to_junction_force(robot)
        elif region == "JUNCTION":
            quota_feed_scale = branch_quota_source_feed_scale(region)
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
        elif region == "BOTTOM":
            force = compute_base_to_junction_force(robot)
        else:
            force = (
                get_lane_preserving_cohort_direction(
                    robot,
                    active_branch,
                    get_shepherd_fill_target(active_branch),
                )
                * OUTLET_FORCE
                * branch_quota_source_feed_scale(region)
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
                fill_target_depth = max(
                    0.0,
                    get_shepherd_boundary_depth(active_branch)
                    - SHEPHERD_FILL_GAP,
                )
                robot_depth = branch_depth_from_junction(
                    robot.position,
                    active_branch,
                )
                if robot.robot_id in branch_overflow_return_robot_ids:
                    # A finite corridor cannot realize the widened spacing
                    # while holding several times its geometric capacity.
                    # Return only the shallowest overflow to J0 with damping.
                    return_direction = get_backtrack_direction(
                        active_branch
                    )
                    desired_velocity = (
                        return_direction
                        * BRANCH_OVERFLOW_RETURN_SPEED
                    )
                    force = (
                        return_direction
                        * BRANCH_OVERFLOW_RETURN_FORCE
                        + (desired_velocity - robot.velocity)
                        * BRANCH_OVERFLOW_RETURN_VELOCITY_GAIN
                    )
                else:
                    depth_error = fill_target_depth - robot_depth
                    if depth_error > 0.0:
                        # Approach the packing line only from its Junction side.
                        # Once a particle reaches the fixed standoff, route
                        # attraction never reverses it toward the Junction;
                        # local SPH pressure and the curtain form the layers.
                        approach_scale = max(
                            SHEPHERD_STACK_MIN_APPROACH_SCALE,
                            smoothstep01(
                                depth_error
                                / max(
                                    SHEPHERD_STACK_BRAKE_DISTANCE,
                                    EPSILON,
                                )
                            ),
                        )
                        force = (
                            fill_direction
                            * ROUTE_FORCE
                            * relay_motion_scale
                            * SHEPHERD_FILL_FORCE_MULTIPLIER
                            * transfer_target_motion_scale
                            * approach_scale
                        )
                    else:
                        force = pygame.Vector2()
            elif region == "BOTTOM":
                force = compute_base_to_junction_force(robot)
            elif region == "JUNCTION":
                force = (
                    fill_direction
                    * OUTLET_FORCE
                    * FILL_JUNCTION_FEED_MULTIPLIER
                    * fill_feed_ramp
                    * branch_quota_source_feed_scale(region)
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
            if region == "BOTTOM":
                # The priority cohort continues into the target Branch.  Every
                # other Base robot still rises to the Junction mouth so the
                # Base/Junction body never tears into two separated pools.
                force = (
                    compute_base_substitute_force(robot)
                    if robot.robot_id in base_substitute_robot_ids
                    else compute_base_to_junction_force(robot)
                )
            elif (
                region == transfer_branch
                and pre_shepherd_branch == transfer_branch
                and pre_shepherd_pack_ready
                and branch_feed_continuity_deficit <= 0
            ):
                force = pygame.Vector2()
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
                if region in {"JUNCTION", transfer_branch}:
                    route_scale *= handoff_target_feed_scale()
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
            elif region == "BOTTOM":
                # Keep the entire Base column connected to Junction while the
                # strongest recruited front primes the next Branch.
                force = (
                    compute_base_substitute_force(robot)
                    if robot.robot_id in base_substitute_robot_ids
                    else compute_base_to_junction_force(robot)
                )
            elif (
                region == transfer_branch
                and pre_shepherd_branch == transfer_branch
                and pre_shepherd_pack_ready
                and branch_feed_continuity_deficit <= 0
            ):
                # Stop artificial tip suction after the prepared target pack
                # reaches compression.  Incoming source flow then occupies the
                # Junction cross-section instead of creating a second pile.
                force = pygame.Vector2()
            elif region == "JUNCTION":
                merge_direction = (
                    get_lane_preserving_cohort_direction(
                        robot,
                        transfer_branch,
                    )
                )
                feed_scale = handoff_target_feed_scale()
                desired_velocity = (
                    merge_direction
                    * TRANSFER_JUNCTION_MERGE_SPEED
                    * feed_scale
                )
                velocity_alignment = (
                    desired_velocity - robot.velocity
                ) * TRANSFER_JUNCTION_MERGE_VELOCITY_GAIN
                force = limit_vector(
                    (
                        merge_direction
                        * FLOW_BACKTRACK_FORCE
                        * TRANSFER_JUNCTION_MERGE_FORCE_MULTIPLIER
                        * feed_scale
                        + velocity_alignment
                    ),
                    TRANSFER_JUNCTION_MERGE_FORCE_LIMIT,
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

    force += compute_virtual_valve_force(robot)
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
        or not base_reserve_is_mobile(robot)
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
            or not base_reserve_is_mobile(other)
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
        if robot_i.role in {"ANCHOR", "RELAY", "TRUNK_RELAY"}:
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
        lane_spacing_force = pygame.Vector2()
        virtual_force = pygame.Vector2()
        cohesion_force = pygame.Vector2()
        khop_sph_attraction_force = pygame.Vector2()
        gap_attraction_force = pygame.Vector2()
        hose_continuity_force = compute_hose_continuity_force(robot_i)
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

            if distance_sq <= EPSILON:
                # Coincident particles previously had no direction and were
                # skipped forever. Use a deterministic reciprocal normal so
                # the pair separates without injecting random nondeterminism.
                phase_seed = (pair[0] * 73856093 + pair[1] * 19349663) % 360
                separation = pygame.Vector2(1.0, 0.0).rotate(phase_seed)
                if robot_i.robot_id > robot_j.robot_id:
                    separation = -separation
                repulsion_force += separation * EQUILIBRIUM_REPULSION_FORCE_LIMIT
                continue
            if distance_sq > h_sq:
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
            if (
                KHOP_CAPTURE_ENABLED
                and robot_i.role in KHOP_DYNAMIC_ROLES
                and robot_j.role in KHOP_DYNAMIC_ROLES
                and distance > KHOP_SPH_EQUILIBRIUM_DISTANCE
            ):
                attraction_span = max(
                    SMOOTHING_LENGTH - KHOP_SPH_EQUILIBRIUM_DISTANCE,
                    EPSILON,
                )
                stretch = clamp(
                    (distance - KHOP_SPH_EQUILIBRIUM_DISTANCE)
                    / attraction_span,
                    0.0,
                    1.0,
                )
                support = clamp(
                    1.0 - distance / max(SMOOTHING_LENGTH, EPSILON),
                    0.0,
                    1.0,
                )
                # Reciprocal, isotropic body cohesion. This is deliberately
                # separate from the directed Parent-Child tree spring.
                khop_sph_attraction_force += -direction_away * (
                    KHOP_SPH_ATTRACTION_GAIN * stretch * support
                )
            v_ij = robot_i.velocity - robot_j.velocity
            if (
                phase == SimulationPhase.FILL_BEHIND_SHEPHERD
                and robot_i.role == "NORMAL"
                and robot_j.role == "NORMAL"
                and get_robot_region(robot_i.position) == active_branch
                and get_robot_region(robot_j.position) == active_branch
                and robot_i.robot_id
                not in branch_overflow_return_robot_ids
                and robot_j.robot_id
                not in branch_overflow_return_robot_ids
                and not robot_is_in_shepherd_packing_zone(
                    robot_i,
                    active_branch,
                )
                and not robot_is_in_shepherd_packing_zone(
                    robot_j,
                    active_branch,
                )
                and abs(
                    get_cohort_lane_offset(robot_i)
                    - get_cohort_lane_offset(robot_j)
                )
                <= UNIFORM_LANE_MATCH_TOLERANCE
            ):
                branch_direction = BRANCH_DIRECTIONS[active_branch]
                signed_separation = (
                    robot_j.position - robot_i.position
                ).dot(branch_direction)
                if abs(signed_separation) > EPSILON:
                    desired_separation = math.copysign(
                        branch_fill_equilibrium_spacing(active_branch),
                        signed_separation,
                    )
                    axial_error = (
                        signed_separation - desired_separation
                    )
                    axial_peer_speed = (
                        robot_j.velocity - robot_i.velocity
                    ).dot(branch_direction)
                    lane_spacing_force += branch_direction * (
                        UNIFORM_LANE_AXIAL_GAIN * axial_error
                        + UNIFORM_LANE_AXIAL_DAMPING
                        * axial_peer_speed
                    )
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
                (
                    robot_i.role == "NORMAL"
                    and robot_j.role == "NORMAL"
                    or KHOP_CAPTURE_ENABLED
                    and robot_i.role in KHOP_DYNAMIC_ROLES
                    and robot_j.role in KHOP_DYNAMIC_ROLES
                )
                and base_reserve_is_mobile(robot_i)
                and base_reserve_is_mobile(robot_j)
                and distance_sq <= viscoelastic_link_sq
            )
            if viscoelastic_pair:
                khop_cap_pair = (
                    KHOP_CAPTURE_ENABLED
                    and robot_i.role in {"KHOP_LEADER", "KHOP_SHEPHERD"}
                    and robot_j.role in {"KHOP_LEADER", "KHOP_SHEPHERD"}
                )
                if khop_cap_pair:
                    # A newly captured pair may still carry the much wider
                    # NORMAL-particle rest length from a previous frame.
                    # Replace that role-stale memory immediately so it does
                    # not tear the dense cap apart for several seconds.
                    viscoelastic_rest_lengths[pair] = clamp(
                        pair_equilibrium_radius,
                        VISCOELASTIC_REST_MIN,
                        VISCOELASTIC_REST_MAX,
                    )
                elif pair not in viscoelastic_rest_lengths:
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
                    # Plastic relaxation may learn a stretched link, but must
                    # never learn a compressed particle distance as its new
                    # equilibrium.  The old bidirectional relaxation silently
                    # cancelled every spacing increase while the route force
                    # packed robots into the Branch.
                    if distance > equilibrium_target and not khop_cap_pair:
                        relaxation = clamp(
                            VISCOELASTIC_REST_RELAXATION * dt,
                            0.0,
                            1.0,
                        )
                        rest_length += (
                            min(distance, VISCOELASTIC_REST_MAX)
                            - rest_length
                        ) * relaxation
                    viscoelastic_rest_lengths[pair] = clamp(
                        rest_length,
                        VISCOELASTIC_REST_MIN,
                        VISCOELASTIC_REST_MAX,
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
                release_direction = direction_away.copy()
                axial_release = release_direction.dot(
                    backtrack_direction
                )
                if axial_release < 0.0:
                    # Remove the component aimed back into the advancing
                    # Shepherd line. Lateral spreading remains available.
                    release_direction -= (
                        backtrack_direction * axial_release
                    )
                compression_release_force += release_direction * (
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
        limit_vector(khop_sph_attraction_force, KHOP_SPH_ATTRACTION_LIMIT)
        # A robot may have many simultaneous close neighbours.  Bound their
        # summed separation command so a denser frame cannot flip the steering
        # direction in a single integration step.
        limit_vector(
            repulsion_force,
            EQUILIBRIUM_REPULSION_FORCE_LIMIT,
        )
        limit_vector(
            lane_spacing_force,
            UNIFORM_LANE_FORCE_LIMIT,
        )
        limit_vector(
            compression_release_force,
            COMPRESSION_RELEASE_FORCE_LIMIT,
        )
        if (
            (
                phase == SimulationPhase.PRESSURE_PUSH
                and pressure_push_timer
                <= COMPRESSION_RELEASE_DIRECTIONAL_TIME
                or phase == SimulationPhase.FLOW_BACKTRACK
                and robot_i.density_ratio
                >= COMPRESSION_RELEASE_DENSITY_RATIO
            )
            and robot_i.role == "NORMAL"
            and get_robot_region(robot_i.position) == active_branch
        ):
            # Stored-energy release is one-way: pressure may expand toward the
            # Junction or laterally, never back into the oncoming piston.
            for expansion_force in (
                pressure_force,
                compression_release_force,
            ):
                axial_force = expansion_force.dot(
                    backtrack_direction
                )
                if axial_force < 0.0:
                    expansion_force -= (
                        backtrack_direction * axial_force
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
        khop_tree_force = compute_khop_tree_force(robot_i)
        khop_cap_force = compute_khop_cap_force(robot_i)
        khop_centering_force = compute_khop_corridor_centering_force(robot_i)
        khop_gatekeeping_force = compute_khop_gatekeeping_force(robot_i)
        shepherd_curtain_force = compute_shepherd_curtain_force(robot_i)
        pre_shepherd_curtain_force = (
            compute_pre_shepherd_curtain_force(robot_i)
        )
        initial_junction_wall_force = (
            compute_initial_junction_soft_wall_force(robot_i)
        )
        base_piston_force = compute_base_piston_reaction_force(robot_i)
        edf_force = compute_pressure_coupled_edf_force(robot_i)
        robot_i.last_sph_pressure_force = pressure_force.length()
        robot_i.last_compression_release_force = (
            compression_release_force.length()
        )
        robot_i.last_shepherd_force = (
            virtual_force
            + shepherd_curtain_force
            + pre_shepherd_curtain_force
            + khop_gatekeeping_force
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
            + lane_spacing_force
            + virtual_force
            + cohesion_force
            + khop_sph_attraction_force
            + gap_attraction_force
            + hose_continuity_force
            + route_force
            + connectivity_force
            + khop_tree_force
            + khop_cap_force
            + khop_centering_force
            + khop_gatekeeping_force
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
        filtered_target = (
            robot_i.filtered_acceleration
            * (1.0 - filter_alpha)
            + raw_acceleration * filter_alpha
        )
        # Rate-limit steering acceleration as a physical actuator would.  This
        # removes the frame-to-frame left/right command reversal that looked
        # acceptable in simulation but could not be tracked by a real robot.
        acceleration_delta = (
            filtered_target - robot_i.filtered_acceleration
        )
        limit_vector(
            acceleration_delta,
            ACCELERATION_SLEW_RATE * dt,
        )
        robot_i.filtered_acceleration += acceleration_delta
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
        if robot.role in {
            "SHEPHERD",
            "SHEPHERD_CANDIDATE",
        }:
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
        begin_cross_branch_transfer(
            robots,
            branch,
            next_branch,
        )
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

    if KHOP_CAPTURE_ENABLED:
        update_khop_capture_state(robots, dt)
        return

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
                saturation_tracker.reset(selected)
                junction_consensus_tracker.reset()
                phase = SimulationPhase.EXPLORE_BRANCH
                metrics.branch_events.append({"branch": selected, "started_at": simulation_time})

    elif phase == SimulationPhase.EXPLORE_BRANCH:
        update_relay_deployment(robots, dt)
        if pre_shepherd_branch == active_branch:
            # The next-branch pipeline may have switched as soon as enough
            # robots entered the Branch, before every local slot was claimed.
            # Finish any missing slots from the active Branch itself, then
            # enter the same FILL state used by the first explored Branch.
            reserve_shepherd_slots_progressively(
                robots,
                active_branch,
                "PRE_SHEPHERD",
            )
            if (
                not get_shepherds(robots)
                and promote_pre_shepherds(
                    robots,
                    active_branch,
                    require_packed=False,
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

        # Reserve slots one leader at a time as robots enter the wide approach
        # band.  Reserved candidates move forward into already-visible slots;
        # no robot is selected after passing the line and pulled backward.
        reserved, _ = reserve_shepherd_slots_progressively(
            robots,
            active_branch,
            "SHEPHERD_CANDIDATE",
        )
        if len(reserved) == adaptive_shepherd_count():
            for robot in reserved:
                robot.role = "SHEPHERD"
            phase = SimulationPhase.FORM_SHEPHERD_BOUNDARY
            shepherd_form_timer = 0.0
            enforce_shepherd_curtain_for_swarm(robots)
            print(
                f"[Shepherd] progressive slots complete: "
                f"branch={active_branch}, count={len(reserved)}"
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
                    "upstream-standoff election will retry"
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
        # The piston starts only after physical compression is measured.
        # Robot count, cross-section geometry, a stalled front, continuity,
        # and timeout remain diagnostics; none can substitute for density.
        local_packed_ready = (
            saturated
            and saturation_tracker.recognition_mode
            == "PACKED_DENSITY"
        )
        branch_quota_ready = (
            branch_fill_target_count > 0
            and branch_fill_current_count
            >= branch_fill_target_count
        )
        if (
            local_packed_ready
            and branch_quota_ready
        ):
            print(
                "[Pressure Start] compressed SPH layer released; "
                f"tip={saturation_tracker.tip_count}, "
                f"density={saturation_tracker.average_density_ratio:.2f}/"
                f"{SATURATION_PACKED_DENSITY_RATIO:.2f}, "
                f"width={saturation_tracker.lateral_coverage_ratio:.2f}, "
                f"quota={branch_fill_current_count}/"
                f"{branch_fill_target_count}, "
                f"continuity={continuity_ready}"
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
                promoted_pre_shepherd = (
                    selected == pre_shepherd_branch
                    and promote_pre_shepherds(
                        robots,
                        selected,
                    )
                )
                if promoted_pre_shepherd:
                    saturation_tracker.reset(selected)
                    branch_continuity_tracker.reset(selected)
                    # Every DFS child repeats the same physical sequence as the
                    # first Branch.  Robot count/quota can finish filling, but
                    # only a newly measured packed-density dwell may release
                    # the Shepherd piston.
                    phase = SimulationPhase.FILL_BEHIND_SHEPHERD
                    print(
                        "[Pre-Shepherd] branch switch completed; "
                        "fresh density-compression fill starts"
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
        voted_branch = update_distributed_branch_consensus(robots)
        if voted_branch is not None:
            selected = choose_next_branch(
                anchor,
                robots,
                reference_density,
            )
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
        if any(
            robot.role in {
                "RELAY",
                "SHEPHERD",
                "PRE_SHEPHERD",
                "SHEPHERD_CANDIDATE",
            }
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
                "SHEPHERD_CANDIDATE",
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


def draw_planned_shepherd_slots(
    surface,
    branch: str,
    alpha: int = 210,
) -> None:
    """Show deterministic empty slots before any robot reaches their line."""
    overlay = pygame.Surface(
        (SCREEN_WIDTH, SCREEN_HEIGHT),
        pygame.SRCALPHA,
    )
    color = BRANCH_COLORS[branch]
    for slot in build_shepherd_slots(
        branch,
        adaptive_shepherd_count(),
    ):
        pygame.draw.circle(
            overlay,
            (*color, alpha),
            (round(slot.x), round(slot.y)),
            4,
            width=2,
        )
    surface.blit(overlay, (0, 0))


def draw_shepherd_curtain(surface):
    """Draw the planned/active Shepherd line at one continuous coordinate.

    Before election this is the fixed placement guide.  Formation and filling
    reuse that exact depth, so the line does not jump when Shepherd roles are
    assigned.  It becomes dynamic only during the intentional piston/backtrack
    motion.
    """
    curtain_active = shepherd_curtain_active()
    depth = (
        get_shepherd_curtain_depth(active_branch)
        if curtain_active
        else get_shepherd_boundary_depth(active_branch)
    )
    color = BRANCH_COLORS[active_branch]
    alpha = 235 if curtain_active else 170
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    if active_branch == "UP":
        y = round(center_y - half_width - depth)
        pygame.draw.line(
            overlay,
            (*color, alpha),
            (center_x - half_width + 2, y),
            (center_x + half_width - 2, y),
            SHEPHERD_CURTAIN_DRAW_HALF_WIDTH * 2,
        )
    elif active_branch == "LEFT":
        x = round(center_x - half_width - depth)
        pygame.draw.line(
            overlay,
            (*color, alpha),
            (x, center_y - half_width + 2),
            (x, center_y + half_width - 2),
            SHEPHERD_CURTAIN_DRAW_HALF_WIDTH * 2,
        )
    else:
        x = round(center_x + half_width + depth)
        pygame.draw.line(
            overlay,
            (*color, alpha),
            (x, center_y - half_width + 2),
            (x, center_y + half_width - 2),
            SHEPHERD_CURTAIN_DRAW_HALF_WIDTH * 2,
        )
    surface.blit(overlay, (0, 0))


def draw_pre_shepherd_curtain(surface):
    """Visualize the prepared next-branch shield before activation."""
    branch = get_pre_shepherd_staging_branch()
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
    global branch_overflow_return_robot_ids
    global junction_buffer_control
    global junction_buffer_slice_counts
    global junction_buffer_lateral_coverage
    global base_substitute_count
    global base_substitute_robot_ids
    global base_substitute_initial_base_count
    global base_substitute_reserve_target
    global base_junction_feed_robot_ids
    global base_junction_feed_branch
    global base_junction_feed_initial_count
    global base_junction_feed_reserve_target
    global branch_feed_continuity_deficit
    global branch_feed_slice_counts
    global branch_feed_slice_coverages
    global base_junction_interface_gap
    global base_junction_interface_deficit
    global base_junction_interface_slice_counts
    global base_junction_interface_slice_coverages
    global base_tail_refill_robot_ids
    global hose_continuity_forces, hose_continuity_link_count
    global handoff_source_continuity_deficit
    global handoff_source_slice_counts
    global handoff_source_slice_coverages
    global handoff_source_continuity_control
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
    global metrics
    global khop_state
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
    branch_overflow_return_robot_ids = set()
    junction_buffer_control = 0.0
    junction_buffer_slice_counts = []
    junction_buffer_lateral_coverage = []
    base_substitute_count = 0
    base_substitute_robot_ids = set()
    base_substitute_initial_base_count = 0
    base_substitute_reserve_target = 0
    base_junction_feed_robot_ids = set()
    base_junction_feed_branch = None
    base_junction_feed_initial_count = 0
    base_junction_feed_reserve_target = 0
    branch_feed_continuity_deficit = 0
    branch_feed_slice_counts = []
    branch_feed_slice_coverages = []
    base_junction_interface_gap = 0.0
    base_junction_interface_deficit = 0
    base_junction_interface_slice_counts = []
    base_junction_interface_slice_coverages = []
    base_tail_refill_robot_ids = set()
    hose_continuity_forces = {}
    hose_continuity_link_count = 0
    handoff_source_continuity_deficit = 0
    handoff_source_slice_counts = []
    handoff_source_slice_coverages = []
    handoff_source_continuity_control = 0.0
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
    saturation_tracker.reset()
    branch_continuity_tracker.reset()
    junction_consensus_tracker.reset()
    metrics = ExperimentMetrics()
    khop_state = KHopCaptureState()


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
    if KHOP_CAPTURE_ENABLED:
        initialize_khop_capture(robots)
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
headless_frame_counter = 0


while running:
    raw_dt = (
        HEADLESS_DT_MS / 1000.0
        if HEADLESS_STEP_LIMIT is not None
        else max(clock.tick(FPS) / 1000.0, 1.0 / 240.0)
    )
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
            if not KHOP_CAPTURE_ENABLED:
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
        if HEADLESS_STEP_LIMIT is not None:
            headless_frame_counter += 1
            if headless_frame_counter >= HEADLESS_STEP_LIMIT:
                headless_groups = []
                for group in khop_branch_groups():
                    group_leader = khop_state.robot_by_id.get(group.leader_id)
                    leader_position = (
                        tuple(round(value, 1) for value in group_leader.position)
                        if group_leader is not None
                        else "UNFORMED"
                    )
                    headless_groups.append((
                        group.group_id,
                        group.state.name,
                        len(group.member_robot_ids),
                        leader_position,
                        round(group.forward_clearance, 1),
                        round(group.density_evidence_ema, 2),
                        round(group.pressure_evidence_ema, 3),
                        round(group.compression_evidence_ema, 3),
                        round(group.dead_end_dwell, 2),
                        round(group.cap_mean_error, 1),
                    ))
                root_group = khop_state.groups.get(
                    khop_state.root_group_id or ""
                )
                root_leader = (
                    khop_state.robot_by_id.get(root_group.leader_id)
                    if root_group is not None
                    else None
                )
                print(
                    f"[Headless] frames={headless_frame_counter}, "
                    f"stage={khop_state.stage}, groups="
                    f"{headless_groups}, "
                    f"split-dwell={khop_state.split_dwell:.2f}, "
                    f"split-N={khop_state.last_split_group_size}, "
                    f"candidate={khop_state.candidate_elapsed:.2f}, "
                    f"recognition={khop_state.recognition_state.name}, "
                    f"witnesses={len(khop_state.candidate_observer_ids)}, "
                    f"clusters={khop_state.last_cluster_sizes}/"
                    f"{[(round(x, 2), round(y, 2)) for x, y in khop_state.last_cluster_headings]}, "
                    f"valid-cohorts={sum(e.valid for e in khop_state.cohort_evidence)}, "
                    f"root-cap={len(root_group.member_robot_ids) if root_group is not None else 0}/"
                    f"{round(root_group.cap_mean_error, 1) if root_group is not None else '-'}, "
                    f"active={khop_state.active_group_id or '-'}, "
                    f"queue={khop_state.branch_queue}, "
                    f"visited={khop_state.sequential_visit_order}, "
                    f"root-pos="
                    f"{tuple(round(value, 1) for value in root_leader.position) if root_leader is not None else '-'}"
                )
                if HEADLESS_SNAPSHOT_PATH:
                    save_khop_headless_snapshot(
                        HEADLESS_SNAPSHOT_PATH,
                        robots,
                        color_reference_density,
                    )
                running = False
                continue
    else:
        update_communication_system(robots, spatial_grid)
        compute_densities(robots, build_physics_grid(robots))
        compute_pressures(robots, reference_density)

    if HEADLESS_STEP_LIMIT is not None:
        # Physics/communication validation must not spend most of its time
        # rasterizing an invisible frame through SDL's dummy video driver.
        continue

    screen.fill(BACKGROUND_COLOR)
    pygame.draw.polygon(screen, FLOOR_COLOR, cross_points)
    if not KHOP_CAPTURE_ENABLED:
        draw_branch_colour_fields(screen)
    pygame.draw.polygon(screen, WALL_COLOR, cross_points, width=2)
    if not KHOP_CAPTURE_ENABLED:
        draw_branch_gates(screen)

    if show_regions:
        if KHOP_CAPTURE_ENABLED:
            draw_khop_capture_network(screen, robots)
        else:
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
            if phase in {
                SimulationPhase.MOVE_TO_JUNCTION,
                SimulationPhase.EXPLORE_BRANCH,
                SimulationPhase.FORM_SHEPHERD_BOUNDARY,
                SimulationPhase.FILL_BEHIND_SHEPHERD,
            }:
                draw_planned_shepherd_slots(
                    screen,
                    active_branch,
                )
            staging_branch = get_pre_shepherd_staging_branch()
            if staging_branch is not None:
                draw_planned_shepherd_slots(
                    screen,
                    staging_branch,
                    alpha=170,
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
            "SHEPHERD_CANDIDATE",
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
        "Pressure-driven SPH | Distributed NORMAL consensus | Breadcrumb relay",
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
            f"Handoff={TRANSFER_HANDOFF_POLICY_VERSION} | "
            f"merge={TRANSFER_JUNCTION_MERGE_SPEED:.1f} "
            f"feed={handoff_target_feed_scale():.2f}"
        ),
        (
            f"Base substitute={BASE_SUBSTITUTE_POLICY_VERSION} | "
            f"J-slices="
            f"{'/'.join(str(count) for count in junction_buffer_slice_counts) or '-'}"
            f"/{JUNCTION_BUFFER_MIN_ROBOTS_PER_SLICE} "
            f"cover="
            f"{'/'.join(f'{coverage:.2f}' for coverage in junction_buffer_lateral_coverage) or '-'} "
            f"control={junction_buffer_control:.2f} "
            f"dispatch={base_substitute_count}/"
            f"{BASE_SUBSTITUTE_MAX_ACTIVE} "
            f"base-left={base_substitute_reserve_target}/"
            f"{base_substitute_initial_base_count}"
        ),
        (
            f"Base/J0 feed={BASE_TO_JUNCTION_POLICY_VERSION} | "
            f"move={len(base_junction_feed_robot_ids)}/"
            f"{base_junction_feed_initial_count} "
            f"base-left={base_junction_feed_reserve_target} | "
            f"seam={base_junction_interface_gap:.1f}/"
            f"{BASE_JUNCTION_INTERFACE_TARGET_GAP:.1f} "
            f"missing={base_junction_interface_deficit} "
            f"slices="
            f"{'/'.join(str(count) for count in base_junction_interface_slice_counts) or '-'} "
            f"cover-min={min(base_junction_interface_slice_coverages, default=0.0):.2f}"
        ),
        (
            f"J0 release={JUNCTION_REFILL_POLICY_VERSION} | "
            f"ready={junction_buffer_spatially_ready()} | "
            f"required={JUNCTION_BUFFER_MIN_ROBOTS_PER_SLICE} "
            f"cover={JUNCTION_BUFFER_MIN_LATERAL_COVERAGE:.2f}"
        ),
        (
            f"Base tail={BASE_TAIL_REFILL_POLICY_VERSION} | "
            f"refill={len(base_tail_refill_robot_ids)} | "
            f"equilibrium=N{NORMAL_EQUILIBRIUM_SCALE:.2f} "
            f"J{JUNCTION_TAIL_EQUILIBRIUM_SCALE:.2f} "
            f"long={LONG_BRANCH_FILL_EQUILIBRIUM_SCALE:.2f}"
        ),
        (
            f"Branch continuity feed={BRANCH_CONTINUITY_FEED_POLICY_VERSION} | "
            f"missing={branch_feed_continuity_deficit} "
            f"thin={sum(count < BRANCH_CONTINUITY_MIN_ROBOTS_PER_SLICE for count in branch_feed_slice_counts)}/"
            f"{len(branch_feed_slice_counts)} "
            f"min={min(branch_feed_slice_counts, default=0)} "
            f"cover-min={min(branch_feed_slice_coverages, default=0.0):.2f}"
        ),
        (
            f"Source continuity={HANDOFF_SOURCE_CONTINUITY_POLICY_VERSION} | "
            f"missing={handoff_source_continuity_deficit} "
            f"thin={sum(count < BRANCH_CONTINUITY_MIN_ROBOTS_PER_SLICE for count in handoff_source_slice_counts)}/"
            f"{len(handoff_source_slice_counts)} "
            f"cover-min={min(handoff_source_slice_coverages, default=0.0):.2f} "
            f"control={handoff_source_continuity_control:.2f}"
        ),
        (
            f"Cohort flow={COHORT_FLOW_POLICY_VERSION} | "
            f"tail-expand="
            f"{FILL_TAIL_EQUILIBRIUM_EXPANSION * smoothstep01(transfer_gap_control):.2f}"
        ),
        (
            f"Branch quota={BRANCH_FILL_QUOTA_POLICY_VERSION} | "
            f"{get_branch_fill_quota_branch()}="
            f"{branch_fill_current_count}/"
            f"{branch_fill_target_count} "
            f"deficit-control={branch_fill_deficit_control:.2f} "
            f"spacing="
            f"{branch_fill_equilibrium_spacing(get_branch_fill_quota_branch()):.1f} "
            f"overflow={BRANCH_OVERFLOW_POLICY_VERSION}:"
            f"{len(branch_overflow_return_robot_ids)}"
        ),
        (
            f"Uniform rows={UNIFORM_LANE_POLICY_VERSION} | "
            f"C-lines=COMM_PARENT_TREE(not equilibrium)"
        ),
        (
            f"Hose={HOSE_CONTINUITY_POLICY_VERSION} | "
            f"branch={hose_path_branch() or '-'} | "
            f"links={hose_continuity_link_count}"
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
            f"density={saturation_tracker.average_density_ratio:.2f}/"
            f"{SATURATION_PACKED_DENSITY_RATIO:.2f} "
            f"width={saturation_tracker.lateral_coverage_ratio:.2f}/"
            f"{SATURATION_PACKED_LATERAL_COVERAGE_RATIO:.2f} "
            f"mode={saturation_tracker.recognition_mode} "
            f"ready={saturation_tracker.saturated}"
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
            f"approach-gap="
            f"{SHEPHERD_PLACEMENT_STANDOFF + SHEPHERD_ELECTION_BAND_DEPTH} "
            f"place-gap={SHEPHERD_BOUNDARY_WALL_CLEARANCE:.1f} "
            f"band={SHEPHERD_ELECTION_BAND_DEPTH} | "
            f"candidates={shepherd_candidate_count}/"
            f"{adaptive_shepherd_count()}"
        ),
        (
            f"Stack={SHEPHERD_STACK_POLICY_VERSION} | "
            f"wall-gap={SHEPHERD_STACK_WALL_STANDOFF:.1f} "
            f"line-gap={SHEPHERD_FILL_GAP:.1f} "
            f"brake={SHEPHERD_STACK_BRAKE_DISTANCE:.1f}"
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
    if KHOP_CAPTURE_ENABLED:
        hud_lines = build_khop_hud_lines(robots)
    draw_hud_panel(screen, hud_lines)
    pygame.display.flip()

if not metrics.saved and HEADLESS_STEP_LIMIT is None:
    save_experiment_logs(robots, "USER_EXIT")
pygame.quit()
sys.exit()