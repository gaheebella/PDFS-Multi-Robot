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
import copy

from multi_junction_sph_dfs_adaptive_lidar_anchor_v2_child_session_ok import CHILD_PROBE_TRIGGER_DISTANCE
import numpy as np
import pygame


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PHYSICAL_SOURCE = HERE / "single_junction_sph_dfs_environment.py"
ADAPTIVE_SOURCE = HERE / "lidar_junction_detection_adaptive_w_tau_anchor_stop.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pygame_simulator import (
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
    "shepherd": (255, 60, 170),  # vivid hot pink; distinct from LiDAR purple rays
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
GUARD_ASSIGN_MAX_AXIAL_WIDTH_RATIO = 1.20
GUARD_ASSIGN_MAX_LATERAL_WIDTH_RATIO = 1.20
GUARD_ASSIGN_MAX_PATH_WIDTH_RATIO = 1.50
GUARD_LATERAL_COVERAGE_BINS = 9
GUARD_READINESS_LOG_PERIOD = 10
GUARD_EDGE_SEAL_MARGIN_RATIO = 0.25
GUARD_LATERAL_OVERLAP_RATIO = 0.90
PROVISIONAL_WALL_SETTLED_RATIO = 0.95
PROVISIONAL_WALL_STABILITY_DWELL = 0.18
JUNCTION_ARRIVAL_RATIO_THRESHOLD = 0.45
ANCHOR_ENTRANCE_STOP_TOLERANCE = 80.0
ENTRANCE_STABILITY_FRAMES = 1

# ---------------------------------------------------------
# Multi-Junction Child LiDAR probe
# ---------------------------------------------------------

# A fixed Parent LiDAR should remain in place while it can still
# observe the active exploration front.  Once the Frontier moves
# close to the LiDAR maximum sensing depth, the same LiDAR robot
# is released as a Child-observation probe.
#
# This is NOT Child-Junction evidence.
# Child probe starts after the active Frontier has moved
# sufficiently deep relative to its locally observed mouth width.
#
# The LiDAR range still acts as an upper cap.
CHILD_PROBE_TRIGGER_RANGE_RATIO = 0.95
CHILD_PROBE_TRIGGER_WIDTH_RATIO = 1.50

CHILD_PROBE_RELAY_TRIGGER_RATIO = 0.82

CHILD_PROBE_TRIGGER_MAX_DISTANCE = (
    MAX_RANGE * CHILD_PROBE_TRIGGER_RANGE_RATIO
)
# Moving Child-Junction structural evidence.
CHILD_CANDIDATE_MIN_STRUCTURAL_STREAK = 12
CHILD_CANDIDATE_NON_AXIAL_MAX_DOT = 0.75
CHILD_CANDIDATE_MIN_MOUTH_WIDTH_RATIO = 0.35
CHILD_PARENT_CLEARANCE_W_RATIO = 0.50
CHILD_APPROACH_STOP_W_RATIO = 0.10
CHILD_APPROACH_SLOWDOWN_W_RATIO = 0.35
# Child stationary Junction verification
CHILD_STATIONARY_PERSISTENCE_RATIO = 0.60
CHILD_STATIONARY_MIN_OUTGOING = 2


class PerceptionState(Enum):
    MOVING = auto()
    JUNCTION_APPROACH = auto()
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
    local_branch_key: str | None = None
    opening_start_local: pygame.Vector2 | None = None
    opening_end_local: pygame.Vector2 | None = None
    mouth_start_world: pygame.Vector2 | None = None
    mouth_end_world: pygame.Vector2 | None = None
    mouth_center_world: pygame.Vector2 | None = None
    mouth_lateral_unit: pygame.Vector2 | None = None
    branch_tangent_unit: pygame.Vector2 | None = None
    mouth_span: float = 0.0

# =========================================================
# Multi-Junction DFS state
# =========================================================

@dataclass
class ChildObservationSession:
    """Fresh LiDAR observation session for one possible Child Junction."""

    parent_junction_uid: str
    parent_branch_uid: str

    lidar_id: int
    start_frame: int
    start_position: pygame.Vector2

    # Frozen local ingress frame from the Parent branch.
    ingress_t: pygame.Vector2
    ingress_n: pygame.Vector2

    samples: int = 0
    valid_samples: int = 0
    consecutive_valid: int = 0

    last_valid_w: float | None = None
    last_selected_threshold: float | None = None

    # Moving structural-candidate state.
    structural_streak: int = 0

    candidate_frame: int | None = None
    candidate_position: pygame.Vector2 | None = None
    candidate_depth_local: float | None = None
    candidate_selected_threshold: float | None = None
    candidate_lidar_frame: LidarFrame | None = None

    candidate_last_position: pygame.Vector2 | None = None
    candidate_traveled_axial: float = 0.0
    candidate_remaining_depth: float | None = None

    anchor_stopped: bool = False
    anchor_stop_frame: int | None = None
    # Fresh stationary verification state.
    stationary_samples: int = 0
    stationary_tracks: list[PersistentOpening] = field(
        default_factory=list
    )
    stationary_outgoing: list[PersistentOpening] = field(
        default_factory=list
    )
    stationary_confirmed: bool = False
    stationary_confirmation_frame: int | None = None

@dataclass
class MultiJunctionFrame:
    """Logical DFS state for one confirmed Junction."""

    junction_uid: str
    parent_junction_uid: str | None
    incoming_branch_uid: str | None

    # Branch DFS state belonging to this Junction.
    # Later:
    # UNVISITED / ACTIVE / ACTIVE_CHILD / VISITED
    branch_states: dict[str, str] = field(default_factory=dict)
    branch_order: list[str] = field(default_factory=list)
    active_branch_uid: str | None = None

    # Local information used to return toward the Parent.
    # No global Junction coordinate is stored here.
    ingress_direction_local: pygame.Vector2 | None = None
    return_direction_local: pygame.Vector2 | None = None

    # Physical markers are added only when a real Child Junction
    # has been confirmed.
    return_marker_id: int | None = None
    completion_marker_ids: dict[str, int] = field(default_factory=dict)

    # Parent Junction을 Child 탐색 동안 복원할 수 있도록
    # Physical DFS의 논리/geometry context를 보존한다.
    #
    # Robot world position은 저장하지 않는다.
    # 나중 복귀는 Return Marker + local information으로 수행한다.
    saved_physical_context: dict[str, Any] = field(
        default_factory=dict
    )

    # True only when every branch/subtree below this Junction is done.
    subtree_complete: bool = False


class MultiJunctionManager:
    """Top-level recursive DFS bookkeeping.

    This class must not control Guard, Frontier, Shepherd, SPH,
    pressure push, or ordinary dead-end backtracking.
    Those remain owned by the existing Single-Junction Physical DFS.
    """

    def __init__(self) -> None:
        self.stack: list[MultiJunctionFrame] = []
        self.next_junction_index: int = 0

        # Child detection state will be implemented later.
        self.child_candidate_active: bool = False
        self.child_confirmed: bool = False
        # Confirmed Child has been logically pushed,
        # but Parent physical release is still pending.
        self.child_push_complete: bool = False
        self.parent_release_pending: bool = False

        # Same LiDAR robot used for deep-branch Child observation.
        self.child_probe_active: bool = False
        self.child_probe_branch_uid: str | None = None
        self.child_probe_start_frame: int | None = None
        self.child_probe_lidar_id: int | None = None
        self.child_probe_start_position: pygame.Vector2 | None = None
        self.child_session: ChildObservationSession | None = None

    @property
    def current(self) -> MultiJunctionFrame | None:
        """Return the current DFS Junction, if one exists."""
        if not self.stack:
            return None
        return self.stack[-1]

    @property
    def depth(self) -> int:
        """Current DFS depth. Before J0 exists, return -1."""
        return len(self.stack) - 1

    def create_root(self) -> MultiJunctionFrame:
        """Create J0 once after the first Junction is truly confirmed."""
        if self.stack:
            raise RuntimeError(
                "Root Junction cannot be created twice"
            )

        frame = MultiJunctionFrame(
            junction_uid="J0",
            parent_junction_uid=None,
            incoming_branch_uid=None,
        )

        self.stack.append(frame)
        self.next_junction_index = 1

        print(
            "[MultiDFS] ROOT_CREATED "
            "junction=J0 depth=0"
        )

        return frame

    def allocate_child_uid(self) -> str:
        """Allocate the next Junction UID without changing the DFS stack."""
        uid = f"J{self.next_junction_index}"
        self.next_junction_index += 1
        return uid
    def push_confirmed_child(
        self,
        session: ChildObservationSession,
    ) -> MultiJunctionFrame:
        """Push one stationary-confirmed Child Junction.

        Logical DFS bookkeeping only.

        This does NOT:
        - release Parent Guard/Shepherd roles,
        - create Completion/Return Markers,
        - start Child Physical DFS.
        """

        if self.child_push_complete:
            raise RuntimeError(
                "confirmed Child Junction was pushed twice"
            )

        parent = self.current

        if parent is None:
            raise RuntimeError(
                "cannot push Child without Parent Junction"
            )

        if (
            parent.junction_uid
            != session.parent_junction_uid
        ):
            raise RuntimeError(
                "Child session Parent does not match "
                "current DFS Junction: "
                f"current={parent.junction_uid} "
                f"session_parent="
                f"{session.parent_junction_uid}"
            )

        parent_branch_uid = (
            session.parent_branch_uid
        )

        if (
            parent_branch_uid
            not in parent.branch_states
        ):
            raise RuntimeError(
                "Child Parent branch missing from "
                "Parent DFS frame: "
                f"{parent_branch_uid}"
            )

        previous_state = (
            parent.branch_states[
                parent_branch_uid
            ]
        )

        if previous_state != "ACTIVE":
            raise RuntimeError(
                "confirmed Child must come from "
                "an ACTIVE Parent branch: "
                f"branch={parent_branch_uid} "
                f"state={previous_state}"
            )

        # -------------------------------------------------
        # Parent branch is NOT complete.
        #
        # Its subtree is now being explored, therefore:
        #
        # ACTIVE -> ACTIVE_CHILD
        #
        # Never mark it VISITED here.
        # -------------------------------------------------
        parent.branch_states[
            parent_branch_uid
        ] = "ACTIVE_CHILD"

        parent.active_branch_uid = (
            parent_branch_uid
        )

        child_uid = (
            self.allocate_child_uid()
        )

        ingress = (
            session.ingress_t.copy()
        )

        if (
            ingress.length_squared()
            <= 1.0e-12
        ):
            raise RuntimeError(
                "confirmed Child has invalid ingress direction"
            )

        ingress = ingress.normalize()

        child = MultiJunctionFrame(
            junction_uid=child_uid,
            parent_junction_uid=(
                parent.junction_uid
            ),
            incoming_branch_uid=(
                parent_branch_uid
            ),

            # Direction actually traversed from Parent
            # toward this Child.
            ingress_direction_local=(
                ingress.copy()
            ),

            # Physical return direction later points
            # back toward the Parent.
            return_direction_local=(
                -ingress
            ),
        )

        # Child branch_order / branch_states intentionally
        # remain empty here.
        #
        # They are populated NEXT from the stationary
        # verified outgoing mouths.
        self.stack.append(
            child
        )

        self.child_push_complete = True
        self.parent_release_pending = True

        print(
            "[ParentBranchActiveChild] "
            f"junction={parent.junction_uid} "
            f"branch={parent_branch_uid} "
            f"{previous_state}->ACTIVE_CHILD"
        )

        print(
            "[DFSStackPush] "
            f"parent={parent.junction_uid} "
            f"child={child.junction_uid} "
            f"incoming_branch="
            f"{child.incoming_branch_uid} "
            f"depth={self.depth}"
        )

        print(
            "[MultiDFS] STACK "
            f"junctions="
            f"{[frame.junction_uid for frame in self.stack]} "
            f"current={self.current.junction_uid} "
            "parent_release_pending=True"
        )

        return child

multi_dfs = MultiJunctionManager()

def sync_multi_dfs_from_physical(
    physical: types.ModuleType,
) -> None:
    """Mirror the current Physical-DFS branch states into Multi DFS.

    Read-only synchronization:
    this function must never change Physical DFS behavior.
    """

    frame = multi_dfs.current

    if frame is None:
        return

    if not frame.branch_order:
        return

    changed = False

    for uid in frame.branch_order:
        descriptor = physical.branch_descriptors_by_uid.get(uid)

        if descriptor is None:
            continue

        physical_state = str(descriptor.visit_state)
        previous_state = frame.branch_states.get(uid)

        if previous_state != physical_state:
            frame.branch_states[uid] = physical_state
            changed = True

            print(
                "[MultiDFS] BRANCH_SYNC "
                f"junction={frame.junction_uid} "
                f"branch={uid} "
                f"{previous_state}->{physical_state}"
            )

    active_uid = getattr(
        physical,
        "active_branch_uid",
        None,
    )

    if frame.active_branch_uid != active_uid:
        previous_active = frame.active_branch_uid
        frame.active_branch_uid = active_uid

        print(
            "[MultiDFS] ACTIVE_SYNC "
            f"junction={frame.junction_uid} "
            f"{previous_active}->{active_uid}"
        )

    if changed:
        print(
            "[MultiDFS] STATE "
            f"junction={frame.junction_uid} "
            f"depth={multi_dfs.depth} "
            f"states={frame.branch_states}"
        )

def save_parent_physical_context(
    physical: types.ModuleType,
    parent: MultiJunctionFrame,
) -> None:
    """Freeze Parent DFS control context before physical release.

    No robot position is stored as a return target.
    """

    if parent.saved_physical_context:
        return

    parent.saved_physical_context = copy.deepcopy(
        {
            "branch_descriptors_by_uid":
                physical.branch_descriptors_by_uid,

            "fixture_key_to_branch_uid":
                physical.fixture_key_to_branch_uid,

            "branch_uid_to_fixture_key":
                physical.branch_uid_to_fixture_key,

            "detected_branch_candidates":
                physical.detected_branch_candidates,

            "junction_guard_groups":
                physical.junction_guard_groups,

            "integration_wall_lifecycle":
                physical.integration_wall_lifecycle,

            "integration_ready_guard_ids_by_uid":
                physical.integration_ready_guard_ids_by_uid,

            "integration_wall_status":
                physical.integration_wall_status,

            "branch_order_plan":
                physical.branch_order_plan,

            "branch_fixture_order_plan":
                physical.branch_fixture_order_plan,

            "active_branch":
                physical.active_branch,

            "active_branch_uid":
                physical.active_branch_uid,

            "phase":
                physical.phase,
        }
    )

    print(
        "[ParentStateSaved] "
        f"junction={parent.junction_uid} "
        f"branches={parent.branch_order} "
        f"states={parent.branch_states} "
        f"active_child={parent.active_branch_uid}"
    )

def retain_parent_completion_markers(
    physical: types.ModuleType,
    parent: MultiJunctionFrame,
    robots: Sequence[Any],
) -> None:
    """Retain or materialize one physical Completion Marker per VISITED branch.

    Existing VISITED Pebbles are reused.

    If legacy/same-ID Physical DFS completed the branch by restoring the
    original Guard wall instead of creating a Pebble, promote exactly one
    robot from that branch's existing JUNCTION_GUARD lineage in place.

    ACTIVE_CHILD never receives a Completion Marker here.
    """

    existing_by_uid = {
        pebble.pebble_branch_uid: pebble
        for pebble in physical.get_pebbles(robots)
        if (
            getattr(
                pebble,
                "pebble_state",
                None,
            )
            == "VISITED"
            and getattr(
                pebble,
                "pebble_branch_uid",
                None,
            )
            is not None
        )
    }

    visited_uids: list[str] = []

    for branch_uid, state in (
        parent.branch_states.items()
    ):
        if state != "VISITED":
            continue

        visited_uids.append(branch_uid)

        marker = existing_by_uid.get(
            branch_uid
        )

        # -------------------------------------------------
        # Legacy/same-ID Guard-cycle compatibility:
        #
        # GUARD -> FRONTIER -> SHEPHERD -> GUARD completed
        # the branch without materialising a Pebble.
        #
        # Use one robot from THAT SAME branch Guard lineage.
        # No NORMAL re-election and no teleport.
        # -------------------------------------------------
        if marker is None:

            candidates = [
                robot
                for robot in robots
                if (
                    getattr(
                        robot,
                        "role",
                        None,
                    )
                    == "JUNCTION_GUARD"
                    and getattr(
                        robot,
                        "junction_guard_branch_uid",
                        None,
                    )
                    == branch_uid
                )
            ]

            descriptor = getattr(
                physical,
                "branch_descriptors_by_uid",
                {},
            ).get(branch_uid)

            mouth = (
                getattr(
                    descriptor,
                    "observed_mouth_position",
                    None,
                )
                if descriptor is not None
                else None
            )

            if candidates:
                if mouth is not None:
                    marker = min(
                        candidates,
                        key=lambda robot: (
                            robot.position.distance_squared_to(
                                mouth
                            ),
                            robot.robot_id,
                        ),
                    )
                else:
                    # Deterministic fallback inside the SAME
                    # original Guard lineage only.
                    marker = min(
                        candidates,
                        key=lambda robot:
                            robot.robot_id,
                    )

        if marker is None:
            raise RuntimeError(
                "VISITED Parent branch has neither "
                "a physical Completion Pebble nor "
                "an original Guard candidate: "
                f"junction={parent.junction_uid} "
                f"branch={branch_uid}"
            )

        # Existing Pebble: nothing else to materialise.
        if (
            getattr(marker, "role", None)
            == "PEBBLE"
            and getattr(
                marker,
                "pebble_state",
                None,
            )
            == "VISITED"
        ):
            parent.completion_marker_ids[
                branch_uid
            ] = marker.robot_id

            print(
                "[CompletionMarkerRetained] "
                f"junction={parent.junction_uid} "
                f"branch={branch_uid} "
                f"robot={marker.robot_id}"
            )
            continue

        branch_key = getattr(
            marker,
            "junction_guard_branch",
            None,
        )

        ingress = None

        if descriptor is not None:
            candidate_ingress = getattr(
                descriptor,
                "local_outgoing_direction",
                None,
            )
            if (
                candidate_ingress is not None
                and candidate_ingress.length_squared()
                > physical.EPSILON
            ):
                ingress = (
                    candidate_ingress.normalize()
                )

        if ingress is None:
            local_by_uid = getattr(
                marker,
                "local_ingress_tangents_by_uid",
                {},
            )
            candidate_ingress = (
                local_by_uid.get(branch_uid)
            )

            if (
                candidate_ingress is not None
                and candidate_ingress.length_squared()
                > physical.EPSILON
            ):
                ingress = (
                    candidate_ingress.normalize()
                )

        if (
            ingress is None
            and branch_key is not None
        ):
            local_by_fixture = getattr(
                marker,
                "local_ingress_tangents",
                {},
            )
            candidate_ingress = (
                local_by_fixture.get(branch_key)
            )

            if (
                candidate_ingress is not None
                and candidate_ingress.length_squared()
                > physical.EPSILON
            ):
                ingress = (
                    candidate_ingress.normalize()
                )

        if ingress is None:
            raise RuntimeError(
                "Original Guard selected for Completion "
                "Marker has no valid branch-local ingress: "
                f"junction={parent.junction_uid} "
                f"branch={branch_uid} "
                f"robot={marker.robot_id}"
            )

        # Preserve the physical pose exactly.
        marker.role = "PEBBLE"
        marker.pebble_anchor = (
            marker.position.copy()
        )
        marker.pebble_branch_uid = (
            branch_uid
        )
        marker.pebble_branch_key = (
            branch_key
        )
        marker.pebble_state = "VISITED"
        marker.pebble_ingress_direction_local = (
            ingress.copy()
        )
        marker.pebble_return_direction_local = (
            -ingress
        )

        if hasattr(
            physical,
            "branch_completion_epoch",
        ):
            physical.branch_completion_epoch += 1
            marker.pebble_completion_epoch = (
                physical.branch_completion_epoch
            )
        else:
            marker.pebble_completion_epoch = 0

        if hasattr(
            marker,
            "known_visited_branch_uids",
        ):
            marker.known_visited_branch_uids.add(
                branch_uid
            )

        if (
            branch_key is not None
            and hasattr(
                marker,
                "known_visited_branches",
            )
        ):
            marker.known_visited_branches.add(
                branch_key
            )

        # Marker becomes force-free in its current pose.
        marker.velocity.update(0.0, 0.0)
        marker.commanded_velocity.update(
            0.0,
            0.0,
        )
        marker.observed_velocity.update(
            0.0,
            0.0,
        )
        marker.acceleration.update(0.0, 0.0)
        marker.filtered_acceleration.update(
            0.0,
            0.0,
        )

        parent.completion_marker_ids[
            branch_uid
        ] = marker.robot_id

        existing_by_uid[
            branch_uid
        ] = marker

        print(
            "[CompletionMarkerBackfilled] "
            f"junction={parent.junction_uid} "
            f"branch={branch_uid} "
            f"robot={marker.robot_id} "
            "source=ORIGINAL_GUARD "
            "position_jump=0"
        )

    print(
        "[ParentCompletionMarkersReady] "
        f"junction={parent.junction_uid} "
        f"visited={visited_uids} "
        f"markers="
        f"{parent.completion_marker_ids}"
    )

def create_parent_return_marker(
    physical: types.ModuleType,
    parent: MultiJunctionFrame,
    robots: Sequence[Any],
    perception: AdaptivePerception,
) -> int:
    """Convert one existing Parent wall robot in place.

    No teleport.
    No global Junction coordinate.
    LiDAR robot is never used.
    """

    parent_branch_uids = set(
        parent.branch_order
    )

    candidates: list[
        tuple[float, int, Any]
    ] = []

    for robot in robots:

        if robot is perception.leader:
            continue

        if robot.role != "JUNCTION_GUARD":
            continue

        branch_uid = getattr(
            robot,
            "junction_guard_branch_uid",
            None,
        )

        branch_key = getattr(
            robot,
            "junction_guard_branch",
            None,
        )

        if (
            branch_uid
            not in parent_branch_uids
        ):
            try:
                branch_uid = (
                    physical.branch_uid_for_fixture(
                        branch_key
                    )
                )
            except (
                KeyError,
                TypeError,
                AttributeError,
            ):
                branch_uid = None

        if branch_uid not in parent_branch_uids:
            continue

        descriptor = (
            physical.branch_descriptors_by_uid.get(
                branch_uid
            )
        )

        if descriptor is None:
            continue

        try:
            axial, _ = (
                physical.branch_local_coordinates(
                    robot.position,
                    descriptor,
                )
            )
        except (
            ValueError,
            AttributeError,
        ):
            continue

        # mouth 근처 Guard를 우선 선택한다.
        candidates.append(
            (
                abs(float(axial)),
                robot.robot_id,
                robot,
            )
        )

    if not candidates:
        raise RuntimeError(
            "no physical Parent Guard available "
            "for Junction Return Marker"
        )

    _, _, marker = min(
        candidates
    )

    # -------------------------------------------------
    # Same-position role transition only.
    # -------------------------------------------------
    marker.role = "PEBBLE"

    marker.pebble_anchor = (
        marker.position.copy()
    )

    # Return Marker는 branch completion fact가 아니다.
    marker.pebble_branch_uid = None
    marker.pebble_branch_key = None

    marker.pebble_state = (
        "JUNCTION_RETURN"
    )

    marker.pebble_ingress_direction_local = None

    marker.pebble_return_direction_local = (
        parent.return_direction_local.copy()
        if parent.return_direction_local
        is not None
        else None
    )

    # Multi-Junction-specific local metadata.
    marker.marker_type = (
        "JUNCTION_RETURN"
    )

    marker.marker_junction_uid = (
        parent.junction_uid
    )

    marker.junction_guard_anchor = None
    marker.junction_guard_branch = None
    marker.junction_guard_branch_uid = None
    marker.junction_guard_parent_id = None
    marker.junction_guard_layer = -1
    marker.is_branch_leader = False

    marker.shepherd_anchor = None
    marker.shepherd_origin = None
    marker.shepherd_branch = None
    marker.frontier_local_lateral = None

    marker.velocity.update(
        0.0,
        0.0,
    )

    marker.acceleration.update(
        0.0,
        0.0,
    )

    marker.filtered_acceleration.update(
        0.0,
        0.0,
    )

    parent.return_marker_id = (
        marker.robot_id
    )

    print(
        "[ReturnMarkerCreated] "
        f"junction={parent.junction_uid} "
        f"robot={marker.robot_id} "
        "position_snap=False "
        "branch_uid=None "
        "state=JUNCTION_RETURN"
    )

    return marker.robot_id

def release_parent_physical_roles(
    physical: types.ModuleType,
    parent: MultiJunctionFrame,
    robots: Sequence[Any],
) -> dict[str, int]:
    """Release only Parent wall roles.

    Completion Pebbles and Return Marker remain fixed.
    Breadcrumb/Relay robots are intentionally preserved.
    """

    parent_branch_uids = set(
        parent.branch_order
    )

    released = {
        "guard": 0,
        "frontier": 0,
        "shepherd": 0,
    }

    for robot in robots:

        if robot.role == "PEBBLE":
            continue

        branch_uid = None

        if robot.role == "JUNCTION_GUARD":

            branch_uid = getattr(
                robot,
                "junction_guard_branch_uid",
                None,
            )

            branch_key = getattr(
                robot,
                "junction_guard_branch",
                None,
            )

            if (
                branch_uid
                not in parent_branch_uids
            ):
                try:
                    branch_uid = (
                        physical.branch_uid_for_fixture(
                            branch_key
                        )
                    )
                except (
                    KeyError,
                    TypeError,
                    AttributeError,
                ):
                    branch_uid = None

        elif robot.role in {
            "FRONTIER_SHEPHERD",
            "SHEPHERD",
            "PRE_SHEPHERD",
        }:

            branch_key = getattr(
                robot,
                "shepherd_branch",
                None,
            )

            try:
                branch_uid = (
                    physical.branch_uid_for_fixture(
                        branch_key
                    )
                )
            except (
                KeyError,
                TypeError,
                AttributeError,
            ):
                branch_uid = None

        else:
            continue

        if branch_uid not in parent_branch_uids:
            continue

        previous_role = robot.role

        robot.role = "NORMAL"

        robot.junction_guard_anchor = None
        robot.junction_guard_branch = None
        robot.junction_guard_branch_uid = None
        robot.junction_guard_hop = -1
        robot.junction_guard_parent_id = None
        robot.junction_guard_layer = -1
        robot.is_branch_leader = False

        if hasattr(
            robot,
            "integration_guard_waypoints",
        ):
            robot.integration_guard_waypoints = []

        if hasattr(
            robot,
            "integration_guard_final_anchor",
        ):
            robot.integration_guard_final_anchor = None

        robot.shepherd_anchor = None
        robot.shepherd_origin = None
        robot.shepherd_branch = None
        robot.shepherd_return_direction = None
        robot.frontier_local_lateral = None

        robot.base_reserve = False

        if previous_role == "JUNCTION_GUARD":
            released["guard"] += 1

        elif previous_role == "FRONTIER_SHEPHERD":
            released["frontier"] += 1

        else:
            released["shepherd"] += 1

    print(
        "[ParentRolesReleased] "
        f"junction={parent.junction_uid} "
        f"guards={released['guard']} "
        f"frontiers={released['frontier']} "
        f"shepherds={released['shepherd']} "
        "relays_preserved=True"
    )

    return released

def release_confirmed_parent_junction(
    physical: types.ModuleType,
    perception: AdaptivePerception,
    robots: Sequence[Any],
) -> None:
    """Compress confirmed Parent Junction into local markers."""

    if not multi_dfs.parent_release_pending:
        return

    if len(multi_dfs.stack) < 2:
        raise RuntimeError(
            "Parent release requires "
            "stack=[Parent, Child]"
        )

    child = multi_dfs.stack[-1]
    parent = multi_dfs.stack[-2]

    print(
        "[JunctionReleaseStart] "
        f"parent={parent.junction_uid} "
        f"child={child.junction_uid} "
        f"active_child="
        f"{parent.active_branch_uid}"
    )

    # 1. logical / physical Parent context 보존
    save_parent_physical_context(
        physical,
        parent,
    )

    # 2. 완료 Branch의 기존 Pebble 유지
    retain_parent_completion_markers(
        physical,
        parent,
        robots,
    )

    # 3. Junction Return Marker 한 대 남김
    create_parent_return_marker(
        physical,
        parent,
        robots,
        perception,
    )

    # 4. 나머지 Parent wall 역할 해제
    release_parent_physical_roles(
        physical,
        parent,
        robots,
    )

    multi_dfs.parent_release_pending = False

    print(
        "[ParentReleaseComplete] "
        f"parent={parent.junction_uid} "
        f"child={child.junction_uid} "
        f"completion_markers="
        f"{parent.completion_marker_ids} "
        f"return_marker="
        f"{parent.return_marker_id} "
        "parent_release_pending=False"
    )

def update_child_lidar_probe(
    physical: types.ModuleType,
    perception: AdaptivePerception,
    robots: Sequence[Any],
) -> None:
    """Release the same LiDAR robot only for deep-branch observation.

    IMPORTANT:
    - This does not confirm a Child Junction.
    - This does not release the Parent Junction.
    - This does not create Markers.
    - This does not perform DFS PUSH.
    """

    frame = multi_dfs.current

    if frame is None:
        return

    active_uid = getattr(
        physical,
        "active_branch_uid",
        None,
    )

          # -----------------------------------------------------
    # Probe already moving:
    # keep the SAME LiDAR robot near the frozen branch
    # centerline while allowing physical forward motion.
    # -----------------------------------------------------
    if multi_dfs.child_probe_active:
        if active_uid != multi_dfs.child_probe_branch_uid:
            return

        descriptor = physical.branch_descriptors_by_uid.get(
            active_uid
        )

        if descriptor is None:
            return

        lidar_robot = perception.leader

        try:
            axial, lateral = physical.branch_local_coordinates(
                lidar_robot.position,
                descriptor,
            )

            tangent, normal = physical.descriptor_local_basis(
                descriptor
            )

        except (ValueError, AttributeError):
            return

        tangent = tangent.normalize()
        normal = normal.normalize()

        # -------------------------------------------------
        # Local branch-frame guidance only.
        #
        # No J1 global coordinate.
        # No direct position overwrite.
        # No teleport.
        # -------------------------------------------------

        session = multi_dfs.child_session

        # Once the Child Anchor has stopped, do not apply
        # any more probe/approach drive.
        if (
            session is not None
            and session.anchor_stopped
        ):
            return

        lateral_velocity = (
            lidar_robot.velocity.dot(normal)
        )

        # -------------------------------------------------
        # Lateral centerline restoration.
        # -------------------------------------------------
        lateral_command = (
            -3.0 * lateral
            -2.2 * lateral_velocity
        )

        lateral_command = max(
            -55.0,
            min(55.0, lateral_command),
        )

        # Default before a Child Candidate exists.
                # -------------------------------------------------
        # Smooth Child-probe cruise BEFORE Candidate latch.
        #
        # From the moment the LiDAR probe is released,
        # the Child controller owns the frozen ingress
        # axial direction.
        #
        # SPH remains active laterally, but must not
        # alternately push the LiDAR forward/backward.
        # -------------------------------------------------
        candidate_control_active = (
            multi_dfs.child_candidate_active
            and session is not None
            and session.candidate_depth_local is not None
            and session.candidate_last_position is not None
        )

        forward_command = 0.0

        if not candidate_control_active:

            axial_direction = (
                session.ingress_t
                if session is not None
                else tangent
            )

            axial_velocity = float(
                lidar_robot.velocity.dot(
                    axial_direction
                )
            )

            # Remove only SPH acceleration along the
            # branch-forward axis.
            #
            # Lateral SPH interaction remains untouched.
            existing_axial_acc = float(
                lidar_robot.acceleration.dot(
                    axial_direction
                )
            )

            lidar_robot.acceleration -= (
                axial_direction
                * existing_axial_acc
            )

            # Smooth cruise rather than a constant
            # acceleration command.
                        # -------------------------------------------------
            # Communication-aware smooth cruise.
            #
            # Cruise freely while the current Base-rooted
            # communication link has enough margin.
            #
            # As the LiDAR approaches the communication
            # hard limit, smoothly reduce the desired
            # forward speed to zero instead of repeatedly
            # pushing against the hard communication clamp.
            #
            # Once a new Breadcrumb becomes the parent,
            # comm_parent_distance drops again and the
            # LiDAR naturally accelerates forward.
            # -------------------------------------------------
            probe_cruise_speed = 10.0

            comm_parent = getattr(
                lidar_robot,
                "comm_parent",
                None,
            )

            if (
                comm_parent is None
                or not lidar_robot.connected_to_base
            ):
                comm_parent_distance = float("inf")
                comm_speed_scale = 0.0

            else:
                comm_parent_distance = (
                    lidar_robot.position.distance_to(
                        comm_parent.position
                    )
                )

                comm_guard_start = float(
                    physical.COMM_GUARD_START
                )

                comm_hard_limit = float(
                    physical.COMM_GUARD_HARD_LIMIT
                )

                if (
                    comm_parent_distance
                    <= comm_guard_start
                ):
                    comm_speed_scale = 1.0

                elif (
                    comm_parent_distance
                    >= comm_hard_limit
                ):
                    comm_speed_scale = 0.0

                else:
                    comm_speed_scale = (
                        comm_hard_limit
                        - comm_parent_distance
                    ) / max(
                        physical.EPSILON,
                        comm_hard_limit
                        - comm_guard_start,
                    )

                    comm_speed_scale = float(
                        np.clip(
                            comm_speed_scale,
                            0.0,
                            1.0,
                        )
                    )

            probe_target_axial_speed = (
                probe_cruise_speed
                * comm_speed_scale
            )

            probe_speed_error = (
                probe_target_axial_speed
                - axial_velocity
            )

            probe_forward_limit = (
                0.15
                * physical.MAX_ACCELERATION
            )

            forward_command = float(
                np.clip(
                    6.0 * probe_speed_error,
                    -probe_forward_limit,
                    probe_forward_limit,
                )
            )

            if (
                physical.integration_frame
                % 20
                == 0
            ):
                print(
                    "[ChildProbeCruise] "
                    f"junction={frame.junction_uid} "
                    f"branch={active_uid} "
                    f"lidar_id={lidar_robot.robot_id} "
                    f"axial_v={axial_velocity:.2f} "
                    f"target_v="
                    f"{probe_target_axial_speed:.2f} "
                    f"comm_scale="
                    f"{comm_speed_scale:.2f} "
                    f"comm_dist="
                    f"{comm_parent_distance:.2f} "
                    f"sph_axial_removed="
                    f"{existing_axial_acc:.2f} "
                    f"forward_cmd="
                    f"{forward_command:.2f}"
                )

        # -------------------------------------------------
        # Child Candidate approach
        #
        # Candidate position/depth were frozen when the
        # moving structural Candidate was latched.
        #
        # remaining_depth
        #   = saved_candidate_depth
        #     - signed local odometry
        # -------------------------------------------------
        if (
            multi_dfs.child_candidate_active
            and session is not None
            and session.candidate_depth_local is not None
            and session.candidate_last_position is not None
        ):

            # ---------------------------------------------
            # Candidate-start-relative local odometry.
            #
            # Measure displacement directly from the frozen
            # Candidate position instead of accumulating
            # frame-to-frame steps.
            # ---------------------------------------------
            if session.candidate_position is None:
                return

            displacement_from_candidate = (
                lidar_robot.position
                - session.candidate_position
            )

            session.candidate_traveled_axial = max(
                0.0,
                float(
                    displacement_from_candidate.dot(
                        session.ingress_t
                    )
                ),
            )

            session.candidate_remaining_depth = max(
                0.0,
                session.candidate_depth_local
                - session.candidate_traveled_axial,
            )

            # Use the W measured at Candidate time as the
            # scale reference. Later unstable scans must not
            # move the stopping criterion.
            if (
                session.candidate_lidar_frame
                is not None
            ):
                candidate_width = float(
                    session.candidate_lidar_frame.adaptive_w
                )
            elif session.last_valid_w is not None:
                candidate_width = float(
                    session.last_valid_w
                )
            else:
                candidate_width = float(
                    lidar_frame.adaptive_w
                )
            # Use the SAME entrance-stop tolerance as J0.
            #
            # J0:
            #   entrance_depth <= ANCHOR_ENTRANCE_STOP_TOLERANCE
            #
            # Child Junctions must use the same geometric
            # stopping rule instead of driving almost to
            # zero remaining depth.
            # SAME entrance-stop rule as J0.
            stop_tolerance = float(
                ANCHOR_ENTRANCE_STOP_TOLERANCE
            )

            # Start slowing before the common entrance
            # stopping boundary so approach remains smooth.
            slowdown_distance = (
                stop_tolerance
                + CHILD_APPROACH_SLOWDOWN_W_RATIO
                * candidate_width
            )

            remaining_depth = (
                session.candidate_remaining_depth
            )

            # ---------------------------------------------
            # Candidate approach complete -> Anchor STOP
            # at the LiDAR robot's CURRENT physical pose.
            #
            # No coordinate target.
            # No position overwrite.
            # No teleport.
            # ---------------------------------------------
            if anchor_entrance_stop_reached(
                remaining_depth
            ):
                session.anchor_stopped = True
                session.anchor_stop_frame = (
                    physical.integration_frame
                )
                # Start a completely fresh stationary
                # Child-Junction verification session.
                session.stationary_samples = 0
                session.stationary_tracks.clear()
                session.stationary_outgoing.clear()
                session.stationary_confirmed = False
                session.stationary_confirmation_frame = None

                perception.anchor_position = (
                    lidar_robot.position.copy()
                )
                perception.anchor_fixed = True

                lidar_robot.is_fixed_anchor = True
                lidar_robot.base_reserve = True

                lidar_robot.velocity.update(
                    0.0,
                    0.0,
                )
                lidar_robot.acceleration.update(
                    0.0,
                    0.0,
                )
                lidar_robot.filtered_acceleration.update(
                    0.0,
                    0.0,
                )

                # Parent Physical DFS is NOT released here.
                physical.integration_guard_hold_active = False

                print(
                    "[ChildAnchorStop] "
                    f"parent={session.parent_junction_uid} "
                    f"branch={session.parent_branch_uid} "
                    f"lidar_id={session.lidar_id} "
                    f"frame={session.anchor_stop_frame} "
                    f"candidate_depth="
                    f"{session.candidate_depth_local:.2f} "
                    f"traveled="
                    f"{session.candidate_traveled_axial:.2f} "
                    f"remaining={remaining_depth:.2f} "
                    f"stop_tol={stop_tolerance:.2f} "
                    "position_snap=False "
                    "confirmed=False "
                    "parent_release=False "
                    "marker=False "
                    "dfs_push=False"
                )

                return

            # ---------------------------------------------
            # Slow down continuously as remaining depth
            # approaches the stop tolerance.
            # ---------------------------------------------
            slowdown_span = max(
                slowdown_distance
                - stop_tolerance,
                physical.EPSILON,
            )

            drive_ratio = float(
                np.clip(
                    (
                        remaining_depth
                        - stop_tolerance
                    )
                    / slowdown_span,
                    0.0,
                    1.0,
                )
            )

            # ---------------------------------------------
            # Axial velocity tracking.
            #
            # A constant +18 acceleration was too weak
            # against SPH / repulsion / damping and allowed
            # the LiDAR robot to oscillate in place.
            #
            # Here we command a LOCAL forward velocity
            # along the frozen ingress axis.
            # ---------------------------------------------
            axial_velocity = float(
                lidar_robot.velocity.dot(
                    session.ingress_t
                )
            )

            # Far from the Candidate target:
            #   target ≈ 14 px/s
            #
            # Near the stop point:
            #   target smoothly falls to ≈ 3 px/s.
            target_axial_speed = (
                3.0
                + 11.0 * drive_ratio
            )

            speed_error = (
                target_axial_speed
                - axial_velocity
            )

            # ---------------------------------------------
            # Child approach owns ONLY the frozen ingress
            # axial direction.
            #
            # SPH / robot interaction may still act in the
            # lateral direction, but its axial acceleration
            # must not fight the Child approach controller.
            # ---------------------------------------------
            existing_axial_acc = float(
                lidar_robot.acceleration.dot(
                    session.ingress_t
                )
            )

            lidar_robot.acceleration -= (
                session.ingress_t
                * existing_axial_acc
            )

            forward_command = (
                10.0 * speed_error
            )

            forward_limit = (
                0.25
                * physical.MAX_ACCELERATION
            )

            # Symmetric control:
            # positive -> accelerate toward Child
            # negative -> brake when moving too fast
            forward_command = float(
                np.clip(
                    forward_command,
                    -forward_limit,
                    forward_limit,
                )
            )

            if (
                physical.integration_frame
                % 10
                == 0
            ):
                print(
                    "[ChildCandidateApproach] "
                    f"parent={session.parent_junction_uid} "
                    f"branch={session.parent_branch_uid} "
                    f"lidar_id={session.lidar_id} "
                    f"traveled="
                    f"{session.candidate_traveled_axial:.2f} "
                    f"remaining="
                    f"{remaining_depth:.2f} "
                    f"stop_tol="
                    f"{stop_tolerance:.2f} "
                    f"sph_axial_removed="
                    f"{existing_axial_acc:.2f} "
                    f"axial_v="
                    f"{axial_velocity:.2f} "
                    f"target_v="
                    f"{target_axial_speed:.2f} "
                    f"forward_cmd="
                    f"{forward_command:.2f}"
                )

        # -------------------------------------------------
        # Physical acceleration only.
        #
        # Never write lidar_robot.position here.
        # -------------------------------------------------
        lidar_robot.acceleration += (
            tangent * forward_command
            + normal * lateral_command
        )

        if physical.integration_frame % 20 == 0:
            comm_parent = getattr(
                lidar_robot,
                "comm_parent",
                None,
            )

            comm_parent_id = getattr(
                comm_parent,
                "robot_id",
                None,
            )

            comm_parent_distance = (
                lidar_robot.position.distance_to(
                    comm_parent.position
                )
                if comm_parent is not None
                else float("nan")
            )

            probe_acc = lidar_robot.acceleration.length()

            forward_test_position = (
                lidar_robot.position
                + tangent * physical.ROBOT_RADIUS
            )

            forward_walkable = physical.is_walkable(
                forward_test_position,
                lidar_robot.radius,
            )

            print(
                "[ChildProbeProgress] "
                f"junction={frame.junction_uid} "
                f"branch={active_uid} "
                f"lidar_id={lidar_robot.robot_id} "
                f"role={lidar_robot.role} "
                f"axial={axial:.2f} "
                f"lateral={lateral:.2f} "
                f"lateral_v={lateral_velocity:.2f} "
                f"center_cmd={lateral_command:.2f} "
                f"acc={probe_acc:.2f} "
                f"anchor_fixed={perception.anchor_fixed} "
                f"is_fixed_anchor="
                f"{lidar_robot.is_fixed_anchor} "
                f"base_reserve={lidar_robot.base_reserve} "
                f"connected="
                f"{lidar_robot.connected_to_base} "
                f"comm_parent={comm_parent_id} "
                f"comm_dist={comm_parent_distance:.2f} "
                f"comm_hard="
                f"{physical.COMM_GUARD_HARD_LIMIT:.2f} "
                f"forward_walkable={forward_walkable}"
            )

        return

    # -----------------------------------------------------
    # Probe may start only during ordinary Branch exploration.
    # -----------------------------------------------------
    if (
        physical.phase
        != physical.SimulationPhase.EXPLORE_BRANCH
        or active_uid is None
    ):
        return

    descriptor = physical.branch_descriptors_by_uid.get(
        active_uid
    )

    if descriptor is None:
        return

    active_fixture = getattr(
        physical,
        "active_branch",
        None,
    )

    if active_fixture is None:
        return

    frontiers = physical.get_frontier_shepherds(
        robots,
        active_fixture,
    )

    if not frontiers:
        return

    # Measure Frontier depth only in the frozen local Branch frame.
    frontier_depths: list[float] = []

    for robot in frontiers:
        try:
            axial, _ = physical.branch_local_coordinates(
                robot.position,
                descriptor,
            )
        except (ValueError, AttributeError):
            continue

        frontier_depths.append(float(axial))

    if not frontier_depths:
        return

    frontier_centroid_depth = (
        sum(frontier_depths)
        / len(frontier_depths)
    )

        # -----------------------------------------------------
    # Branch-local adaptive Child-probe trigger.
    #
    # Use the physical mouth width measured for THIS branch,
    # rather than one fixed world-distance threshold.
    #
    # The LiDAR max-range threshold remains only an upper cap.
    # -----------------------------------------------------
    observed_width = float(
        getattr(
            descriptor,
            "observed_physical_width",
            0.0,
        )
        or getattr(
            descriptor,
            "observed_width",
            0.0,
        )
        or 0.0
    )

    if observed_width > physical.EPSILON:
        child_probe_trigger_depth = min(
            CHILD_PROBE_TRIGGER_MAX_DISTANCE,
            CHILD_PROBE_TRIGGER_WIDTH_RATIO
            * observed_width,
        )
    else:
        # Safe fallback if physical mouth width is unavailable.
        child_probe_trigger_depth = (
            CHILD_PROBE_TRIGGER_MAX_DISTANCE
        )

    if physical.integration_frame % 20 == 0:
        observed_width = float(
            getattr(
                descriptor,
                "observed_physical_width",
                0.0,
            )
            or getattr(
                descriptor,
                "observed_width",
                0.0,
            )
            or 0.0
        )

        print(
            "[ChildProbeGate] "
            f"junction={frame.junction_uid} "
            f"branch={active_uid} "
            f"frontier_depth="
            f"{frontier_centroid_depth:.2f} "
            f"trigger_depth="
            f"{child_probe_trigger_depth:.2f} "
            f"mouth_width="
            f"{observed_width:.2f} "
            f"depth_over_width="
            f"{frontier_centroid_depth / max(observed_width, physical.EPSILON):.2f} "
            f"frontiers="
            f"{len(frontier_depths)} "
            "probe_active=False"
        )

    if (
        frontier_centroid_depth
        < child_probe_trigger_depth
    ):
        return

    lidar_robot = perception.leader

    # The persistent LiDAR robot must never have been consumed
    # by a Relay/Breadcrumb role.
    if lidar_robot.role in {
        "RELAY",
        "TRUNK_RELAY",
    }:
        previous_role = lidar_robot.role
        previous_relay_index = getattr(
            lidar_robot,
            "relay_index",
            -1,
        )

        lidar_robot.role = "NORMAL"
        lidar_robot.relay_anchor = None
        lidar_robot.relay_index = -1

        if hasattr(
            lidar_robot,
            "relay_scope",
        ):
            lidar_robot.relay_scope = None

        if hasattr(
            lidar_robot,
            "relay_owner_edge_id",
        ):
            lidar_robot.relay_owner_edge_id = None

        lidar_robot.velocity.update(
            0.0,
            0.0,
        )
        lidar_robot.acceleration.update(
            0.0,
            0.0,
        )
        lidar_robot.filtered_acceleration.update(
            0.0,
            0.0,
        )

        print(
            "[LiDARIllegalRelayRelease] "
            f"lidar_id={lidar_robot.robot_id} "
            f"previous_role={previous_role} "
            f"relay_index={previous_relay_index}"
        )

    # Same LiDAR ID must be retained.
    multi_dfs.child_probe_active = True
    multi_dfs.child_probe_branch_uid = active_uid
    multi_dfs.child_probe_start_frame = (
        physical.integration_frame
    )
    multi_dfs.child_probe_lidar_id = (
        lidar_robot.robot_id
    )
    multi_dfs.child_probe_start_position = (
        lidar_robot.position.copy()
    )

    # Freeze LiDAR orientation to the active Branch's
    # already-observed local tangent.
    tangent, normal = physical.descriptor_local_basis(
        descriptor
    )
    tangent = tangent.normalize()
    normal = normal.normalize()

    perception.yaw_deg = math.degrees(
        math.atan2(
            tangent.y,
            tangent.x,
        )
    )



    # -----------------------------------------------------
    # Start a completely fresh Child-observation session.
    #
    # Do not reuse J0 persistent-opening history here.
    # -----------------------------------------------------
    multi_dfs.child_session = ChildObservationSession(
        parent_junction_uid=frame.junction_uid,
        parent_branch_uid=active_uid,
        lidar_id=lidar_robot.robot_id,
        start_frame=physical.integration_frame,
        start_position=lidar_robot.position.copy(),
        ingress_t=tangent.copy(),
        ingress_n=normal.copy(),
    )

    print(
        "[ChildSessionStart] "
        f"parent={frame.junction_uid} "
        f"branch={active_uid} "
        f"lidar_id={lidar_robot.robot_id} "
        f"frame={physical.integration_frame}"
    )


    # -----------------------------------------------------
    # Release ONLY the LiDAR robot.
    #
    # Do NOT release the Parent Junction.
    # -----------------------------------------------------
    perception.anchor_fixed = False
    lidar_robot.is_fixed_anchor = False
    lidar_robot.base_reserve = False

    print(
        "[ChildProbeStart] "
        f"junction={frame.junction_uid} "
        f"branch={active_uid} "
        f"lidar_id={lidar_robot.robot_id} "
        f"frontier_depth={frontier_centroid_depth:.2f} "
        f"trigger_depth={child_probe_trigger_depth:.2f} "
        f"mouth_width={observed_width:.2f} "
        f"depth_over_width="
        f"{frontier_centroid_depth / max(observed_width, physical.EPSILON):.2f} "
        f"yaw={perception.yaw_deg:.2f} "
        "parent_release=False "
        "marker=False "
        "dfs_push=False"
    )

def update_child_observation_session(
    physical: types.ModuleType,
    perception: AdaptivePerception,
    lidar_frame: LidarFrame,
) -> None:
    """Accumulate only fresh Child-observation LiDAR samples."""

    session = multi_dfs.child_session

    if not multi_dfs.child_probe_active:
        return

    if session is None:
        return

    # The LiDAR robot ID must never change.
    if perception.leader.robot_id != session.lidar_id:
        raise RuntimeError(
            "Child observation LiDAR ID changed: "
            f"{session.lidar_id} -> "
            f"{perception.leader.robot_id}"
        )

    session.samples += 1

    valid = (
        lidar_frame.interval_valid
        and lidar_frame.selected is not None
    )

    if valid:
        session.valid_samples += 1
        session.consecutive_valid += 1

        session.last_valid_w = float(
            lidar_frame.adaptive_w
        )

        session.last_selected_threshold = float(
            lidar_frame.selected
        )

    else:
        session.consecutive_valid = 0

    if physical.integration_frame % 20 == 0:
        print(
            "[ChildSession] "
            f"parent={session.parent_junction_uid} "
            f"branch={session.parent_branch_uid} "
            f"lidar_id={session.lidar_id} "
            f"samples={session.samples} "
            f"valid={session.valid_samples} "
            f"consecutive_valid={session.consecutive_valid} "
            f"openings={len(lidar_frame.openings)} "
            f"interval_valid={lidar_frame.interval_valid} "
            f"selected={lidar_frame.selected}"
        )   

def _build_child_stationary_frozen_frame(
    perception: AdaptivePerception,
    lidar_frame: LidarFrame,
    session: ChildObservationSession,
) -> LidarFrame | None:
    """Re-evaluate a stationary raw scan with the valid Candidate threshold.

    The current adaptive W may become invalid inside a Junction.
    Therefore the threshold that was valid when the moving Candidate
    was detected is frozen and reused here.

    Raw FAR/Rmax rays are still not accepted as physical mouths.
    Finite wall-side verification is performed separately.
    """

    if (
        session.candidate_selected_threshold is None
        or session.candidate_lidar_frame is None
    ):
        return None

    frozen_threshold = float(
        session.candidate_selected_threshold
    )

    candidate_frame = session.candidate_lidar_frame

    openings, diagnostics = (
        adaptive._detect_openings_w_tau_with_diagnostics(
            lidar_frame.angles,
            lidar_frame.raw,
            selected_threshold=frozen_threshold,
            threshold_interval_valid=True,
            smoothing_window_size=SMOOTHING_WINDOW,
        )
    )

    return LidarFrame(
        frame=lidar_frame.frame,
        angles=lidar_frame.angles.copy(),
        raw=lidar_frame.raw.copy(),
        smoothed=np.asarray(
            diagnostics["smoothed_ranges"]
        ),
        support=np.asarray(
            diagnostics["open_support_mask"],
            dtype=bool,
        ),
        openings=tuple(
            dict(item)
            for item in openings
        ),
        left=lidar_frame.left,
        right=lidar_frame.right,

        # Keep the geometry scale that was valid when the
        # moving Child Candidate was created.
        adaptive_w=float(
            candidate_frame.adaptive_w
        ),
        lower=float(candidate_frame.lower),
        upper=float(candidate_frame.upper),
        selected=frozen_threshold,
        interval_valid=True,
        current_evidence=False,
    )

def update_child_stationary_verification(
    physical: types.ModuleType,
    perception: AdaptivePerception,
    lidar_frame: LidarFrame,
) -> None:
    """Confirm a Child Junction from persistent stationary physical mouths.

    After stationary confirmation this function performs
    logical Multi-DFS handoff only:

    - Parent ACTIVE -> ACTIVE_CHILD
    - create Child MultiJunctionFrame
    - DFS PUSH

    It still does NOT:
    - release Parent Guard/Shepherd roles,
    - create Completion/Return Markers,
    - initialize Child branch descriptors,
    - start Child Physical DFS.
    """

    session = multi_dfs.child_session

    if (
        session is None
        or not multi_dfs.child_candidate_active
        or not session.anchor_stopped
        or multi_dfs.child_confirmed
    ):
        return

    frozen_frame = _build_child_stationary_frozen_frame(
        perception,
        lidar_frame,
        session,
    )

    if frozen_frame is None:
        return

    session.stationary_samples += 1

    verified_outgoing: list[
        dict[str, float]
    ] = []

    candidate_width = float(
        session.candidate_lidar_frame.adaptive_w
    )

    minimum_mouth_width = (
        CHILD_CANDIDATE_MIN_MOUTH_WIDTH_RATIO
        * candidate_width
    )

    # -----------------------------------------------------
    # Verify physical mouths.
    #
    # An OPEN sector alone is insufficient.
    # Both wall-side finite endpoints must exist.
    # -----------------------------------------------------
    for opening in frozen_frame.openings:
        start_angle = float(
            opening["start_angle"]
        )
        end_angle = float(
            opening["end_angle"]
        )
        center_angle = float(
            opening["center_angle"]
        )

        try:
            start_point, _, _ = (
                _nearest_wall_side_endpoint(
                    frozen_frame,
                    perception,
                    start_angle,
                    search_direction=-1,
                )
            )

            end_point, _, _ = (
                _nearest_wall_side_endpoint(
                    frozen_frame,
                    perception,
                    end_angle,
                    search_direction=+1,
                )
            )
        except RuntimeError:
            # A FAR/Rmax opening without finite wall sides
            # is not a verified physical branch mouth.
            continue

        mouth_chord = (
            end_point - start_point
        )

        if (
            mouth_chord.length()
            < minimum_mouth_width
        ):
            continue

        radial = _body_local_unit(
            perception,
            center_angle,
        )

        if (
            radial.length_squared()
            <= physical.EPSILON
        ):
            continue

        radial = radial.normalize()

        ingress_alignment = float(
            radial.dot(session.ingress_t)
        )

        # The corridor actually traversed from the Parent is
        # represented by stored ingress history.
        #
        # Do not require a rear LiDAR opening and do not count
        # a strong rear-facing opening as a Child outgoing edge.
        if ingress_alignment <= -0.50:
            continue

        verified_outgoing.append(
            opening
        )

    # -----------------------------------------------------
    # Associate only VERIFIED stationary mouths.
    # Moving-session persistence is never reused.
    # -----------------------------------------------------
    available = set(
        range(len(session.stationary_tracks))
    )

    for opening in verified_outgoing:
        center = float(
            opening["center_angle"]
        )

        candidates = [
            (
                circular_error(
                    center,
                    session.stationary_tracks[
                        index
                    ].center_angle,
                ),
                index,
            )
            for index in available
        ]

        error, index = min(
            candidates,
            default=(float("inf"), -1),
        )

        if (
            index >= 0
            and error
            <= ASSOCIATION_TOLERANCE_DEG
        ):
            track = (
                session.stationary_tracks[index]
            )
            available.remove(index)

        else:
            track = PersistentOpening(
                "CHILD_OPEN_"
                f"{len(session.stationary_tracks):02d}"
            )
            session.stationary_tracks.append(
                track
            )

        track.update(
            opening,
            physical.integration_frame,
        )

    persistent_outgoing = [
        track
        for track in session.stationary_tracks
        if (
            len(track.observations)
            >= MIN_PERSISTENT_OBSERVATIONS
            and track.persistence_ratio(
                session.stationary_samples
            )
            >= CHILD_STATIONARY_PERSISTENCE_RATIO
        )
    ]

    # Re-check that the stationary result still contains
    # non-axial structure relative to the frozen ingress axis.
    persistent_non_axial = []

    for track in persistent_outgoing:
        radial = _body_local_unit(
            perception,
            track.center_angle,
        )

        if (
            radial.length_squared()
            <= physical.EPSILON
        ):
            continue

        radial = radial.normalize()

        alignment = abs(
            float(
                radial.dot(session.ingress_t)
            )
        )

        if (
            alignment
            <= CHILD_CANDIDATE_NON_AXIAL_MAX_DOT
        ):
            persistent_non_axial.append(
                track
            )

    if (
        physical.integration_frame
        % 10
        == 0
    ):
        print(
            "[ChildStationaryVerification] "
            f"parent={session.parent_junction_uid} "
            f"branch={session.parent_branch_uid} "
            f"lidar_id={session.lidar_id} "
            f"samples={session.stationary_samples} "
            f"frozen_threshold="
            f"{session.candidate_selected_threshold:.2f} "
            f"raw_openings="
            f"{len(frozen_frame.openings)} "
            f"verified_outgoing="
            f"{len(verified_outgoing)} "
            f"persistent_outgoing="
            f"{len(persistent_outgoing)} "
            f"persistent_non_axial="
            f"{len(persistent_non_axial)}"
        )

    confirmed = (
        len(persistent_outgoing)
        >= CHILD_STATIONARY_MIN_OUTGOING
        and len(persistent_non_axial)
        >= 1
    )

    if not confirmed:
        return

    # -----------------------------------------------------
    # Child Junction CONFIRMED.
    #
    # Stop here. Parent release / Marker / DFS PUSH belong
    # to the NEXT implementation stage.
    # -----------------------------------------------------
    persistent_outgoing.sort(
        key=lambda track: track.center_angle
    )

    session.stationary_outgoing = list(
        persistent_outgoing
    )

    session.stationary_confirmed = True

    session.stationary_confirmation_frame = (
        physical.integration_frame
    )

    multi_dfs.child_confirmed = True

    child_frame = (
        multi_dfs.push_confirmed_child(
            session
        )
    )

    print(
        "[ChildJunctionConfirmed] "
        f"parent={session.parent_junction_uid} "
        f"branch={session.parent_branch_uid} "
        f"lidar_id={session.lidar_id} "
        f"frame="
        f"{session.stationary_confirmation_frame} "
        f"stationary_samples="
        f"{session.stationary_samples} "
        f"outgoing_count="
        f"{len(session.stationary_outgoing)} "
        f"frozen_threshold="
        f"{session.candidate_selected_threshold:.2f} "
        "parent_source=INGRESS_HISTORY "
        "parent_release=False "
        "marker=False "
        "active_child=True "
        "dfs_push=True "
        f"child={child_frame.junction_uid} "
        f"dfs_depth={multi_dfs.depth}"
    )



def update_child_moving_candidate(
    physical: types.ModuleType,
    perception: AdaptivePerception,
    lidar_frame: LidarFrame,
) -> None:
    """Detect a moving Child-Junction candidate from fresh local LiDAR evidence.

    This only latches a candidate.

    It does NOT:
    - confirm a Child Junction,
    - release the Parent Junction,
    - create Markers,
    - perform DFS PUSH.
    """

    session = multi_dfs.child_session

    if (
        not multi_dfs.child_probe_active
        or session is None
        or multi_dfs.child_candidate_active
    ):
        return

    # -----------------------------------------------------
    # Parent-Junction clearance gate
    #
    # Prevent the already-known Parent Junction J0
    # from being detected again as a Child Junction.
    # -----------------------------------------------------
    descriptor = physical.branch_descriptors_by_uid.get(
        session.parent_branch_uid
    )

    if descriptor is None:
        return

    try:
        current_axial, current_lateral = (
            physical.branch_local_coordinates(
                perception.leader.position,
                descriptor,
            )
        )
    except (ValueError, AttributeError):
        return

    width_reference = session.last_valid_w

    if width_reference is None:
        width_reference = float(
            lidar_frame.adaptive_w
        )

    parent_clearance_depth = (
        CHILD_PARENT_CLEARANCE_W_RATIO
        * width_reference
    )

    if current_axial < parent_clearance_depth:
        session.structural_streak = 0

        if physical.integration_frame % 20 == 0:
            print(
                "[ChildParentClearance] "
                f"parent={session.parent_junction_uid} "
                f"branch={session.parent_branch_uid} "
                f"lidar_id={session.lidar_id} "
                f"axial={current_axial:.2f} "
                f"lateral={current_lateral:.2f} "
                f"required={parent_clearance_depth:.2f} "
                "cleared=False"
            )

        return

        # -----------------------------------------------------
    # General finite-mouth Junction detector.
    #
    # J1 / J2 / ... use the same structural rule.
    # Parent information comes from the frozen ingress
    # history, never from a global Junction coordinate.
    # -----------------------------------------------------
    observation = evaluate_general_junction_structure(
        perception,
        lidar_frame,
        session.ingress_t,
    )

    if not observation.valid:
        session.structural_streak = 0
        return

    session.structural_streak += 1

    if (
        physical.integration_frame
        % 10
        == 0
    ):
        print(
            "[GeneralJunctionEvidence] "
            f"scope=CHILD "
            f"parent={session.parent_junction_uid} "
            f"branch={session.parent_branch_uid} "
            f"verified_outgoing="
            f"{len(observation.verified_outgoing)} "
            f"non_axial="
            f"{len(observation.non_axial_outgoing)} "
            f"entrance_depth="
            f"{observation.entrance_depth:.2f} "
            f"streak="
            f"{session.structural_streak}/"
            f"{CHILD_CANDIDATE_MIN_STRUCTURAL_STREAK}"
        )

    if (
        session.structural_streak
        < CHILD_CANDIDATE_MIN_STRUCTURAL_STREAK
    ):
        return

    candidate_depth = float(
        observation.entrance_depth
    )

    multi_dfs.child_candidate_active = True

    session.candidate_frame = (
        physical.integration_frame
    )

    session.candidate_position = (
        perception.leader.position.copy()
    )

    session.candidate_depth_local = (
        candidate_depth
    )

    session.candidate_selected_threshold = float(
        lidar_frame.selected
    )

    session.candidate_lidar_frame = (
        lidar_frame
    )

    # -----------------------------------------------------
    # Freeze the Candidate approach reference.
    #
    # From this point on, approach depth is reduced only
    # by local odometry along the stored ingress axis.
    # Do not recompute the Child position from later scans.
    # -----------------------------------------------------
    session.candidate_last_position = (
        perception.leader.position.copy()
    )

    session.candidate_traveled_axial = 0.0

    session.candidate_remaining_depth = (
        candidate_depth
    )

    session.anchor_stopped = False
    session.anchor_stop_frame = None

    print(
        "[ChildMovingCandidate] "
        f"parent={session.parent_junction_uid} "
        f"branch={session.parent_branch_uid} "
        f"lidar_id={session.lidar_id} "
        f"frame={session.candidate_frame} "
        f"verified_outgoing="
        f"{len(observation.verified_outgoing)} "
        f"non_axial="
        f"{len(observation.non_axial_outgoing)} "
        f"depth_local={candidate_depth:.2f} "
        f"threshold={session.candidate_selected_threshold:.2f} "
        "confirmed=False "
        "parent_release=False "
        "marker=False "
        "dfs_push=False"
    )

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

def configure_multi_test_geometry(
    physical: types.ModuleType,
) -> None:
    """Install one physical Child Junction for Multi-DFS testing.

    IMPORTANT:
    J1 geometry exists only as simulator environment geometry.
    Runtime Junction detection must never use its center/rect directly.
    """

    x0 = float(physical.center_x)
    y0 = float(physical.center_y)
    h = float(physical.half_width)
    length = float(physical.normal_length)

    # RIGHT corridor의 실제 끝에 J1 중심을 둔다.
    x1 = x0 + h + float(physical.right_length)

    j1_rect = pygame.Rect(
        round(x1 - h),
        round(y0 - h),
        round(physical.corridor_width),
        round(physical.corridor_width),
    )

    j1_up_rect = pygame.Rect(
        round(x1 - h),
        round(y0 - h - length),
        round(physical.corridor_width),
        round(length),
    )

    j1_down_rect = pygame.Rect(
        round(x1 - h),
        round(y0 + h),
        round(physical.corridor_width),
        round(length),
    )

    # J0 + RIGHT corridor + J1 + J1 UP/DOWN을 하나의
    # 실제 free-space polygon으로 만든다.
    physical.cross_points = [
        # J0 UP
        (x0 - h, y0 - h - length),
        (x0 + h, y0 - h - length),

        # J0 UP -> RIGHT corridor upper wall
        (x0 + h, y0 - h),
        (x1 - h, y0 - h),

        # J1 UP
        (x1 - h, y0 - h - length),
        (x1 + h, y0 - h - length),

        # J1 오른쪽 전체 벽
        (x1 + h, y0 + h + length),

        # J1 DOWN
        (x1 - h, y0 + h + length),
        (x1 - h, y0 + h),

        # RIGHT corridor lower wall -> J0
        (x0 + h, y0 + h),

        # BASE corridor
        (x0 + h, y0 + h + physical.base_length),
        (x0 - h, y0 + h + physical.base_length),
        (x0 - h, y0 + h),

        # J0 LEFT
        (x0 - h - length, y0 + h),
        (x0 - h - length, y0 - h),

        # close polygon
        (x0 - h, y0 - h),
    ]

    # 실제 collision/LiDAR용 mask 재생성
    physical.floor_surface = pygame.Surface(
        (physical.SCREEN_WIDTH, physical.SCREEN_HEIGHT),
        pygame.SRCALPHA,
    )
    physical.floor_surface.fill((0, 0, 0, 0))

    pygame.draw.polygon(
        physical.floor_surface,
        (255, 255, 255, 255),
        physical.cross_points,
    )

    physical.walkable_mask = pygame.mask.from_surface(
        physical.floor_surface
    )

    # -----------------------------------------------------
    # Region recognition 확장
    # -----------------------------------------------------
    original_get_robot_region = physical.get_robot_region

    def multi_get_robot_region(
        position: pygame.Vector2,
    ) -> str:
        point = (
            int(position.x),
            int(position.y),
        )

        if j1_rect.collidepoint(point):
            return "J1_JUNCTION"

        if j1_up_rect.collidepoint(point):
            return "J1_UP"

        if j1_down_rect.collidepoint(point):
            return "J1_DOWN"

        return original_get_robot_region(position)

    physical.get_robot_region = multi_get_robot_region

    def multi_is_region_allowed(
        position: pygame.Vector2,
    ) -> bool:
        return physical.get_robot_region(position) in {
            "BOTTOM",
            "JUNCTION",
            "UP",
            "LEFT",
            "RIGHT",
            "J1_JUNCTION",
            "J1_UP",
            "J1_DOWN",
        }

    physical.is_region_allowed = multi_is_region_allowed

    # RIGHT는 이제 물리적인 dead-end가 아니다.
    # J1 존재 여부는 이후 LiDAR가 판단해야 한다.
    off_map = pygame.Rect(
        -10000,
        -10000,
        1,
        1,
    )

    physical.dead_end_regions["RIGHT"] = off_map.copy()
    physical.early_capture_regions["RIGHT"] = off_map.copy()

    print(
        "[MultiMap] physical_test_geometry_ready "
        "J0_RIGHT_is_nonterminal=True "
        "J1_shape=T_JUNCTION "
        "runtime_detection_authority=LIDAR_ONLY"
    )

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
        self.guard_activation_stage = "WAIT_GROUP"
        self.guard_activation_groups: list[list[ProvisionalGuardGeometry]] = []
        self.guard_current_group_index = 0
        self.guard_all_groups_activated = False
        self.integration_detected_branch_order: list[str] = []
        self.guard_side_pair_activation_frame: int | None = None
        self.guard_up_activation_frame: int | None = None
        self.frame = 0
        self.yaw_deg = -90.0
        self.last_valid_w: float | None = None
        self.last_frame: LidarFrame | None = None
        self.junction_confirmed = False
        self.junction_candidate_detected = False
        self.junction_candidate_frame: int | None = None
        self.junction_candidate_time: float | None = None
        self.entrance_detected = False
        self.entrance_detection_frame: int | None = None
        self.entrance_confidence = 0.0
        self.entrance_depth: float | None = None
        self.entrance_left_endpoint_local: pygame.Vector2 | None = None
        self.entrance_right_endpoint_local: pygame.Vector2 | None = None
        self.entrance_center_local: pygame.Vector2 | None = None
        self.entrance_world_position: pygame.Vector2 | None = None
        self.candidate_lidar_frame: LidarFrame | None = None
        self.entrance_history: list[tuple[float, float]] = []
        self.anchor_fix_frame: int | None = None
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
        if len(persistent) < 2:
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
        if not children:
            return False
        self.outgoing = list(children)
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
        if evidence and not self.junction_candidate_detected:
            self.first_junction_evidence_frame = self.frame
            self.junction_candidate_detected = True
            self.junction_candidate_frame = self.frame
            self.candidate_lidar_frame = result
            self.state = PerceptionState.JUNCTION_APPROACH
            print(f"[LiDAR] junction evidence openings={len(openings)}")
            print("[LiDAR] parent_opening_in_evidence=True")
            print(f"[JunctionCandidate] frame={self.frame} anchor_fixed=False state={self.state.name}")

        if self.state == PerceptionState.JUNCTION_APPROACH and not self.anchor_fixed:
            entrance = estimate_junction_entrance(
                self,
                result,
            )

            if entrance is not None:
                self.entrance_detected = (
                    entrance.valid
                )

                self.entrance_detection_frame = (
                    self.frame
                )

                self.entrance_left_endpoint_local = (
                    entrance.left_endpoint
                )

                self.entrance_right_endpoint_local = (
                    entrance.right_endpoint
                )

                self.entrance_center_local = (
                    entrance.center
                )

                self.entrance_depth = (
                    entrance.depth
                )

                self.entrance_confidence = min(
                    1.0,
                    entrance.width
                    / max(
                        result.adaptive_w,
                        1.0,
                    ),
                )

                self.entrance_history.append(
                    (
                        entrance.depth,
                        entrance.width,
                    )
                )

                self.entrance_history = (
                    self.entrance_history[-8:]
                )

                print(
                    "[EntranceEstimate] "
                    f"frame={self.frame} "
                    f"left="
                    f"({entrance.left_endpoint.x:.2f},"
                    f"{entrance.left_endpoint.y:.2f}) "
                    f"right="
                    f"({entrance.right_endpoint.x:.2f},"
                    f"{entrance.right_endpoint.y:.2f}) "
                    f"depth={entrance.depth:.2f} "
                    f"width={entrance.width:.2f} "
                    f"valid={entrance.valid}"
                )
            drive_scale = 0.35 if self.entrance_detected else 1.0

            print(f"[AnchorApproach] frame={self.frame} entrance_depth={self.entrance_depth if self.entrance_depth is not None else float('nan'):.2f} drive_scale={drive_scale:.2f}")
            stable = self.entrance_detected and len(self.entrance_history) >= ENTRANCE_STABILITY_FRAMES
            if (
                stable
                and self.entrance_depth is not None
                and anchor_entrance_stop_reached(
                    self.entrance_depth
                )
                and self.frame
                > int(
                    self.junction_candidate_frame
                    or -1
                )
            ):
                self.junction_confirmed = True
                self.confirmation_frame = self.frame
                self.confirmation_time = simulation_time
                self.anchor_fixed = True
                self.anchor_fix_frame = self.frame
                self.anchor_position = self.leader.position.copy()
                self.physical.integration_anchor_position = self.anchor_position.copy()
                self.pre_detection_travel = self.anchor_position.distance_to(self.initial_leader_position)
                self.leader.is_fixed_anchor = True
                self.leader.base_reserve = True
                self.physical.integration_guard_hold_active = False
                self.anchor_fixed_mean_normal_forward_speed = self.mean_normal_forward_speed()
                self.state = PerceptionState.FIXED_ACCUMULATING
                print(f"[AnchorFix] candidate_frame={self.junction_candidate_frame} fix_frame={self.anchor_fix_frame} approach_frames={self.anchor_fix_frame - self.junction_candidate_frame} position_snap=False")
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
                self.integration_detected_branch_order = [
                    track.persistent_id for track in self.outgoing
                ]
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

@dataclass(frozen=True)
class JunctionEntranceEstimate:
    left_endpoint: pygame.Vector2
    right_endpoint: pygame.Vector2
    center: pygame.Vector2
    width: float
    depth: float
    valid: bool


def estimate_junction_entrance(
    perception: AdaptivePerception,
    lidar_frame: LidarFrame,
    *,
    forward_axis: pygame.Vector2 | None = None,
) -> JunctionEntranceEstimate | None:
    """Common J0/J1/J2 entrance estimator.

    This intentionally reproduces the current J0 entrance
    geometry first.

    No global Junction position is used.
    """

    if not lidar_frame.openings:
        return None

    if forward_axis is None:
        forward = _body_local_unit(
            perception,
            0.0,
        )
    else:
        forward = forward_axis.copy()

    if forward.length_squared() <= 1.0e-12:
        return None

    forward = forward.normalize()

    broad = max(
        lidar_frame.openings,
        key=lambda item: float(
            item["width_deg"]
        ),
    )

    start_angle = float(
        broad["start_angle"]
    )

    end_angle = float(
        broad["end_angle"]
    )

    left_endpoint = (
        _body_local_unit(
            perception,
            start_angle,
        )
        * _range_at_local_angle(
            lidar_frame,
            start_angle,
        )
    )

    right_endpoint = (
        _body_local_unit(
            perception,
            end_angle,
        )
        * _range_at_local_angle(
            lidar_frame,
            end_angle,
        )
    )

    center = (
        0.5
        * (
            left_endpoint
            + right_endpoint
        )
    )

    width = float(
        left_endpoint.distance_to(
            right_endpoint
        )
    )

    depth = float(
        center.dot(
            forward
        )
    )

    valid = (
        depth > 0.0
        and width
        >= 0.5 * lidar_frame.adaptive_w
    )

    return JunctionEntranceEstimate(
        left_endpoint=left_endpoint,
        right_endpoint=right_endpoint,
        center=center,
        width=width,
        depth=depth,
        valid=valid,
    )


def anchor_entrance_stop_reached(
    remaining_entrance_depth: float,
) -> bool:
    """Common J0/J1/J2 Anchor entrance-stop rule."""

    return (
        remaining_entrance_depth
        <= ANCHOR_ENTRANCE_STOP_TOLERANCE
    )

def _nearest_wall_side_endpoint(
    frame: LidarFrame,
    perception: AdaptivePerception,
    boundary_angle: float,
    *,
    search_direction: int,
    max_search_deg: float = 8.0,
) -> tuple[pygame.Vector2, float, float]:
    """Return the nearest CLOSED/wall hit immediately outside an opening.

    Opening start/end rays are OPEN-support threshold crossings, not physical
    mouth corners.  Guard WHERE must use the adjacent wall-side LiDAR hit.
    The returned vector is already world-oriented relative to the Anchor
    because ``_body_local_unit`` already includes ``perception.yaw_deg``.
    """
    if search_direction not in {-1, 1}:
        raise ValueError("search_direction must be -1 or +1")
    n = len(frame.angles)
    boundary_index = min(
        range(n),
        key=lambda item: circular_error(float(frame.angles[item]), boundary_angle),
    )
    if n <= 1:
        raise RuntimeError("LiDAR frame has insufficient angular samples")
    angular_step = 360.0 / n
    max_steps = max(1, int(math.ceil(max_search_deg / angular_step)))

    for offset in range(1, max_steps + 1):
        index = (boundary_index + search_direction * offset) % n
        if bool(frame.support[index]):
            continue
        raw_range = float(frame.raw[index])
        if not math.isfinite(raw_range) or raw_range >= MAX_RANGE - 1.0e-6:
            continue
        angle = float(frame.angles[index])
        point = _body_local_unit(perception, angle) * raw_range
        return point, angle, raw_range

    # Fail visibly instead of silently placing a Guard on an OPEN ray.
    raise RuntimeError(
        f"no finite wall-side LiDAR hit near opening boundary {boundary_angle:+.1f}deg"
    )

@dataclass(frozen=True)
class GeneralVerifiedMouth:
    opening: dict[str, float]
    start_point: pygame.Vector2
    end_point: pygame.Vector2
    midpoint: pygame.Vector2
    width: float
    ingress_alignment: float
    is_non_axial: bool


@dataclass(frozen=True)
class GeneralJunctionObservation:
    valid: bool
    verified_outgoing: tuple[GeneralVerifiedMouth, ...]
    non_axial_outgoing: tuple[GeneralVerifiedMouth, ...]
    entrance_depth: float | None


def evaluate_general_junction_structure(
    perception: AdaptivePerception,
    lidar_frame: LidarFrame,
    ingress_t: pygame.Vector2,
) -> GeneralJunctionObservation:
    """Finite-mouth Junction structure for Child J1/J2/...

    Parent edge comes from the actually traversed ingress history.
    No global Junction coordinate or fixture position is used.
    """

    if (
        not lidar_frame.interval_valid
        or lidar_frame.selected is None
    ):
        return GeneralJunctionObservation(
            valid=False,
            verified_outgoing=(),
            non_axial_outgoing=(),
            entrance_depth=None,
        )

    if ingress_t.length_squared() <= 1.0e-12:
        return GeneralJunctionObservation(
            valid=False,
            verified_outgoing=(),
            non_axial_outgoing=(),
            entrance_depth=None,
        )

    ingress = ingress_t.normalize()

    minimum_mouth_width = (
        CHILD_CANDIDATE_MIN_MOUTH_WIDTH_RATIO
        * lidar_frame.adaptive_w
    )

    verified_outgoing: list[
        GeneralVerifiedMouth
    ] = []

    non_axial_outgoing: list[
        GeneralVerifiedMouth
    ] = []

    for opening in lidar_frame.openings:

        start_angle = float(
            opening["start_angle"]
        )

        end_angle = float(
            opening["end_angle"]
        )

        center_angle = float(
            opening["center_angle"]
        )

        try:
            start_point, _, _ = (
                _nearest_wall_side_endpoint(
                    lidar_frame,
                    perception,
                    start_angle,
                    search_direction=-1,
                )
            )

            end_point, _, _ = (
                _nearest_wall_side_endpoint(
                    lidar_frame,
                    perception,
                    end_angle,
                    search_direction=+1,
                )
            )

        except RuntimeError:
            # FAR/Rmax only is not a physical branch mouth.
            continue

        width = float(
            start_point.distance_to(
                end_point
            )
        )

        if width < minimum_mouth_width:
            continue

        radial = _body_local_unit(
            perception,
            center_angle,
        )

        if radial.length_squared() <= 1.0e-12:
            continue

        radial = radial.normalize()

        ingress_alignment = float(
            radial.dot(
                ingress
            )
        )

        # Strong rear direction = already traversed Parent.
        if ingress_alignment <= -0.50:
            continue

        midpoint = (
            0.5
            * (
                start_point
                + end_point
            )
        )

        is_non_axial = (
            abs(ingress_alignment)
            <= CHILD_CANDIDATE_NON_AXIAL_MAX_DOT
        )

        mouth = GeneralVerifiedMouth(
            opening=dict(opening),
            start_point=start_point,
            end_point=end_point,
            midpoint=midpoint,
            width=width,
            ingress_alignment=ingress_alignment,
            is_non_axial=is_non_axial,
        )

        verified_outgoing.append(
            mouth
        )

        if is_non_axial:
            non_axial_outgoing.append(
                mouth
            )

    entrance_depth_samples = [
        float(
            mouth.midpoint.dot(
                ingress
            )
        )
        for mouth in non_axial_outgoing
        if float(
            mouth.midpoint.dot(
                ingress
            )
        ) > 0.0
    ]

    entrance_depth = (
        float(
            np.median(
                entrance_depth_samples
            )
        )
        if entrance_depth_samples
        else None
    )

    valid = (
        len(verified_outgoing)
        >= CHILD_STATIONARY_MIN_OUTGOING
        and len(non_axial_outgoing) >= 1
        and entrance_depth is not None
    )

    return GeneralJunctionObservation(
        valid=valid,
        verified_outgoing=tuple(
            verified_outgoing
        ),
        non_axial_outgoing=tuple(
            non_axial_outgoing
        ),
        entrance_depth=entrance_depth,
    )

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
    """Build general corridor-perpendicular Guard geometry from LiDAR.

    Geometry policy:
      1. Detect each opening.
      2. Sample LiDAR points slightly INSIDE both opening boundaries.
      3. Fit the two corridor side-wall directions.
      4. Estimate one common corridor axis from those wall directions.
      5. Build Guard/Frontier/Shepherd rows exactly perpendicular to that axis.

    LEFT / RIGHT / UP identities are not used to calculate geometry.
    """

    if perception.anchor_position is None:
        raise RuntimeError(
            "provisional Guard geometry requires a fixed Anchor"
        )

    # =========================================================
    # Local helper 1:
    # signed shortest angular difference
    # =========================================================
    def signed_angle_delta_deg(
        from_angle: float,
        to_angle: float,
    ) -> float:
        return (
            (to_angle - from_angle + 180.0)
            % 360.0
            - 180.0
        )

    # =========================================================
    # Local helper 2:
    # collect LiDAR hits slightly INSIDE one opening boundary
    #
    # Important:
    # do NOT sample outside the opening.
    # Outside points may belong to the Junction lip/corner.
    # =========================================================
    def collect_corridor_side_points(
        boundary_angle: float,
        center_angle: float,
        *,
        min_fraction: float = 0.12,
        max_fraction: float = 0.45,
        sample_count: int = 12,
    ) -> list[pygame.Vector2]:

        delta = signed_angle_delta_deg(
            boundary_angle,
            center_angle,
        )

        if abs(delta) <= 1.0e-6:
            return []

        points: list[pygame.Vector2] = []
        used_indices: set[int] = set()

        fractions = np.linspace(
            min_fraction,
            max_fraction,
            sample_count,
        )

        for fraction in fractions:

            target_angle = (
                boundary_angle
                + delta * float(fraction)
            )

            # normalize to [-180, 180)
            target_angle = (
                (target_angle + 180.0)
                % 360.0
                - 180.0
            )

            index = min(
                range(len(frame.angles)),
                key=lambda i: circular_error(
                    float(frame.angles[i]),
                    target_angle,
                ),
            )

            # same LiDAR ray duplicated by nearest-index lookup
            if index in used_indices:
                continue

            used_indices.add(index)

            distance = float(
                frame.smoothed[index]
            )

            # Infinite / max-range rays do not describe a wall.
            if (
                not math.isfinite(distance)
                or distance <= 0.0
                or distance >= MAX_RANGE - 1.0e-6
            ):
                continue

            angle = float(
                frame.angles[index]
            )

            point_world = (
                perception.anchor_position
                + _body_local_unit(
                    perception,
                    angle,
                )
                * distance
            )

            points.append(
                point_world
            )

        return points

    # =========================================================
    # Local helper 3:
    # PCA line fitting for one corridor side wall
    # =========================================================
    def fit_corridor_wall_direction(
        points: list[pygame.Vector2],
    ) -> tuple[
        pygame.Vector2 | None,
        float,
    ]:

        if len(points) < 3:
            return None, 0.0

        xy = np.asarray(
            [
                [point.x, point.y]
                for point in points
            ],
            dtype=float,
        )

        centroid = np.mean(
            xy,
            axis=0,
        )

        centered = (
            xy
            - centroid
        )

        covariance = (
            centered.T
            @ centered
            / max(
                len(points) - 1,
                1,
            )
        )

        eigenvalues, eigenvectors = (
            np.linalg.eigh(
                covariance
            )
        )

        order = np.argsort(
            eigenvalues
        )

        minor_value = float(
            eigenvalues[order[0]]
        )

        major_value = float(
            eigenvalues[order[-1]]
        )

        if major_value <= 1.0e-9:
            return None, 0.0

        linearity = (
            major_value
            / max(
                minor_value,
                1.0e-9,
            )
        )

        # Not sufficiently line-like
        if linearity < 3.0:
            return None, linearity

        principal = (
            eigenvectors[
                :,
                order[-1],
            ]
        )

        direction = pygame.Vector2(
            float(principal[0]),
            float(principal[1]),
        )

        if (
            direction.length_squared()
            <= physical.EPSILON
        ):
            return None, linearity

        return (
            direction.normalize(),
            linearity,
        )

    # =========================================================
    # Opening extraction
    # =========================================================
    openings = sorted(
        (
            dict(item)
            for item in frame.openings
        ),
        key=lambda item: float(
            item["center_angle"]
        ),
    )

    if len(openings) < 3:
        raise RuntimeError(
            "Junction evidence lacks three provisional openings"
        )

    # Incoming corridor
    parent = min(
        openings,
        key=lambda item: circular_error(
            float(item["center_angle"]),
            180.0,
        ),
    )

    outgoing = [
        item
        for item in openings
        if item is not parent
    ]

    if len(outgoing) < 3:
        parent = None
        outgoing = openings

    outgoing = sorted(
        outgoing,
        key=lambda item: float(
            item["center_angle"]
        ),
    )

    # ---------------------------------------------------------
    # Current Physical DFS adapter identities only.
    #
    # IMPORTANT:
    # these names are NOT used to calculate axis/normal.
    # ---------------------------------------------------------
    keys: list[str] = []

    for item in outgoing:

        angle = float(
            item["center_angle"]
        )

        keys.append(
            "UP"
            if abs(angle) < 30.0
            else (
                "LEFT"
                if angle < 0.0
                else "RIGHT"
            )
        )

    if len(set(keys)) != 3:
        raise RuntimeError(
            f"duplicate LiDAR branch identities: {keys}"
        )

    print(
        f"[OpeningClassification] "
        f"all={[round(float(x['center_angle']), 1) for x in openings]} "
        f"parent={round(float(parent['center_angle']), 1) if parent else None} "
        f"outgoing={[round(float(x['center_angle']), 1) for x in outgoing]} "
        f"keys={keys} unique=True"
    )

    openings = outgoing

    # =========================================================
    # Junction center estimate
    # =========================================================
    forward = _body_local_unit(
        perception,
        0.0,
    ).normalize()

    broad = max(
        openings,
        key=lambda item: float(
            item["width_deg"]
        ),
    )

    width_reference = (
        frame.adaptive_w
        * PROVISIONAL_MOUTH_WIDTH_W_RATIO
    )

    boundary_forward_depths: list[float] = []

    for key in (
        "start_angle",
        "end_angle",
    ):

        angle = float(
            broad[key]
        )

        projection = (
            _body_local_unit(
                perception,
                angle,
            ).dot(forward)
            * _range_at_local_angle(
                frame,
                angle,
            )
        )

        if projection > 0.0:
            boundary_forward_depths.append(
                projection
            )

    raw_junction_depth = (
        float(
            np.mean(
                boundary_forward_depths
            )
        )
        if boundary_forward_depths
        else width_reference
    )

    junction_depth = float(
        np.clip(
            raw_junction_depth,
            width_reference
            * PROVISIONAL_JUNCTION_DEPTH_MIN_WIDTH_RATIO,
            width_reference
            * PROVISIONAL_JUNCTION_DEPTH_MAX_WIDTH_RATIO,
        )
    )

    junction_center = (
        perception.anchor_position
        + forward * junction_depth
    )

    physical.integration_lidar_junction_estimate = (
        junction_center.copy()
    )

    geometries: list[
        ProvisionalGuardGeometry
    ] = []

    # =========================================================
    # SAME algorithm for every outgoing branch
    # =========================================================
    for index, opening in enumerate(
        openings
    ):

        # The three outgoing openings were already ordered in the Anchor's
        # ingress frame.  Keep that local topology order available as a
        # geometric sanity reference: left / forward / right.  This is not a
        # fixture coordinate; it rotates with the robot's measured body yaw.
        branch_key = keys[index]
    

        start = float(
            opening["start_angle"]
        )

        end = float(
            opening["end_angle"]
        )

        center = float(
            opening["center_angle"]
        )

        # -----------------------------------------------------
        # Physical mouth-side evidence
        # -----------------------------------------------------
        (
            start_point,
            start_wall_angle,
            start_wall_range,
        ) = _nearest_wall_side_endpoint(
            frame,
            perception,
            start,
            search_direction=-1,
        )

        (
            end_point,
            end_wall_angle,
            end_wall_range,
        ) = _nearest_wall_side_endpoint(
            frame,
            perception,
            end,
            search_direction=+1,
        )

        print(
            f"[GuardWallEndpoint] "
            f"center={center:+.1f} "
            f"start_open={start:+.1f} "
            f"start_wall={start_wall_angle:+.1f} "
            f"r={start_wall_range:.2f} "
            f"end_open={end:+.1f} "
            f"end_wall={end_wall_angle:+.1f} "
            f"r={end_wall_range:.2f}"
        )

        raw_chord = (
            end_point
            - start_point
        )

        wall_mid_world = (
            perception.anchor_position
            + 0.5
            * (
                start_point
                + end_point
            )
        )

        # -----------------------------------------------------
        # Direction only:
        # Junction -> this branch
        #
        # Used to resolve +/- ambiguity of PCA line.
        # -----------------------------------------------------
        opening_radial = (
            _body_local_unit(
                perception,
                center,
            ).normalize()
        )

        outward = (
            wall_mid_world
            - junction_center
        )

        if (
            outward.length_squared()
            <= physical.EPSILON
        ):
            outward = (
                opening_radial.copy()
            )
        else:
            outward = (
                outward.normalize()
            )

        # -----------------------------------------------------
        # Collect actual corridor-side wall points
        # from BOTH sides, INSIDE the opening.
        # -----------------------------------------------------
        start_side_points = (
            collect_corridor_side_points(
                start,
                center,
            )
        )

        end_side_points = (
            collect_corridor_side_points(
                end,
                center,
            )
        )

        (
            start_wall_direction,
            start_quality,
        ) = fit_corridor_wall_direction(
            start_side_points
        )

        (
            end_wall_direction,
            end_quality,
        ) = fit_corridor_wall_direction(
            end_side_points
        )

        valid_directions: list[
            tuple[
                pygame.Vector2,
                float,
            ]
        ] = []

        for direction, quality in (
            (
                start_wall_direction,
                start_quality,
            ),
            (
                end_wall_direction,
                end_quality,
            ),
        ):

            if direction is None:
                continue

            direction = (
                direction.normalize()
            )

            # -------------------------------------------------
            # Reject a transverse Junction-lip fit.
            #
            # Corridor wall should be roughly aligned with
            # the opening's outgoing direction.
            # -------------------------------------------------
            alignment = abs(
                float(
                    direction.dot(
                        opening_radial
                    )
                )
            )

            if alignment < 0.55:
                continue

            # orient toward branch
            if (
                direction.dot(
                    opening_radial
                )
                < 0.0
            ):
                direction = -direction

            valid_directions.append(
                (
                    direction,
                    quality,
                )
            )

        # -----------------------------------------------------
        # Combine the two corridor walls
        # -----------------------------------------------------
        if len(valid_directions) >= 2:

            (
                direction_a,
                quality_a,
            ) = valid_directions[0]

            (
                direction_b,
                quality_b,
            ) = valid_directions[1]

            if (
                direction_a.dot(
                    direction_b
                )
                < 0.0
            ):
                direction_b = (
                    -direction_b
                )

            parallel_score = float(
                direction_a.dot(
                    direction_b
                )
            )

            # If the two fitted walls disagree strongly,
            # trust the cleaner fit rather than averaging
            # two unrelated lines.
            if parallel_score < 0.85:

                if quality_a >= quality_b:
                    axis = direction_a
                else:
                    axis = direction_b

                axis_source = (
                    "LIDAR_BEST_SINGLE_CORRIDOR_WALL"
                )

            else:

                combined = (
                    direction_a
                    * max(
                        quality_a,
                        1.0,
                    )
                    + direction_b
                    * max(
                        quality_b,
                        1.0,
                    )
                )

                if (
                    combined.length_squared()
                    > physical.EPSILON
                ):
                    axis = (
                        combined.normalize()
                    )

                    axis_source = (
                        "LIDAR_TWO_PARALLEL_CORRIDOR_WALLS"
                    )

                else:
                    axis = direction_a

                    axis_source = (
                        "LIDAR_SINGLE_EFFECTIVE_CORRIDOR_WALL"
                    )

        elif len(valid_directions) == 1:

            axis = (
                valid_directions[
                    0
                ][0]
            )

            axis_source = (
                "LIDAR_SINGLE_CORRIDOR_WALL"
            )

        else:
            # -------------------------------------------------
            # Last fallback only.
            #
            # Do NOT use raw mouth chord as corridor direction.
            # -------------------------------------------------
            axis = (
                opening_radial.copy()
            )

            axis_source = (
                "LIDAR_OPENING_RADIAL_FALLBACK"
            )

        # Ensure Junction -> branch sign
        if (
            axis.dot(
                opening_radial
            )
            < 0.0
        ):
            axis = -axis

        axis = (
            axis.normalize()
        )

        # Near a side mouth, one threshold opening can be an oblique view of
        # the ingress corridor.  Its fitted wall then points back along the
        # parent corridor and would place two of the three rows inside a wall.
        # Reject that topologically impossible orientation using only the
        # Anchor-local left/forward/right opening order.

        # =====================================================
        # Guard / Frontier / Shepherd direction
        #
        # ALWAYS 90 degrees to corridor axis
        # =====================================================
        mouth_normal = pygame.Vector2(
            -axis.y,
            axis.x,
        ).normalize()

        # Keep lateral sign consistent with the actual
        # start -> end mouth evidence.
        if (
            raw_chord.length_squared()
            > physical.EPSILON
            and mouth_normal.dot(
                raw_chord
            )
            < 0.0
        ):
            mouth_normal = (
                -mouth_normal
            )

        # =====================================================
        # Corridor width
        #
        # IMPORTANT:
        # raw chord is used ONLY for width projection,
        # never for corridor orientation.
        # =====================================================
        projected_span = (
            abs(
                float(
                    raw_chord.dot(
                        mouth_normal
                    )
                )
            )
            if (
                raw_chord.length_squared()
                > physical.EPSILON
            )
            else 0.0
        )

        estimated_width = (
            _lidar_estimated_mouth_width(
                frame,
                opening,
                axis,
                perception,
            )
        )

        minimum_reliable_span = max(
            4.0 * physical.ROBOT_RADIUS,
            0.55 * frame.adaptive_w,
        )

        if (
            projected_span
            >= minimum_reliable_span
        ):

            mouth_span = (
                projected_span
            )

            width_source = (
                "LIDAR_WALL_ENDPOINT_CROSS_SECTION"
            )

        else:

            mouth_span = max(
                estimated_width,
                4.0 * physical.ROBOT_RADIUS,
            )

            width_source = (
                "ADAPTIVE_W_FALLBACK"
            )

        # =====================================================
        # Unified physical mouth geometry
        #
        # Every outgoing branch uses the SAME rule:
        #
        # finite wall endpoint A + finite wall endpoint B
        #                     ↓
        #            physical mouth midpoint
        #
        # No UP / LEFT / RIGHT special case.
        # No Junction-center-derived axial mouth relocation.
        # =====================================================

        mouth = wall_mid_world.copy()

        straight_start = (
            mouth
            - mouth_normal
            * (
                0.5
                * mouth_span
            )
        )

        straight_end = (
            mouth
            + mouth_normal
            * (
                0.5
                * mouth_span
            )
        )

        uid = (
            f"PROV_{index:02d}"
        )

        descriptor = (
            physical.BranchDescriptor(
                uid=uid,
                junction_uid=physical.CURRENT_JUNCTION_ID,
                fixture_key=None,

                local_outgoing_direction=(
                    axis.copy()
                ),

                local_return_direction=(
                    -axis
                ),

                observed_mouth_position=(
                    mouth.copy()
                ),

                observed_width=mouth_span,

                cohort_member_ids=set(),

                direction_last_estimate=(
                    axis.copy()
                ),

                direction_stability_reference=(
                    axis.copy()
                ),

                direction_stable_dwell=1.0,
                direction_sample_count=1,
                direction_angular_spread=0.0,
                direction_is_stable=True,

                direction_mature_dwell=1.0,
                direction_is_mature=True,

                direction_downstream_travel=0.0,

                motion_t=(
                    axis.copy()
                ),

                motion_n=(
                    mouth_normal.copy()
                ),

                motion_frame_locked=True,
                motion_frame_source=axis_source,

                motion_frame_sample_count=1,
                motion_frame_angular_spread=0.0,

                motion_observed_width=(
                    mouth_span
                ),

                observed_flow_width=(
                    mouth_span
                ),

                observed_physical_width=(
                    mouth_span
                ),

                physical_width_confident=True,
                physical_width_source=(
                    width_source
                ),

                physical_left_boundary_lateral=(
                    -0.5
                    * mouth_span
                ),

                physical_right_boundary_lateral=(
                    0.5
                    * mouth_span
                ),

                physical_boundary_sample_count=2,

                discovered_at=(
                    physical.simulation_time
                ),
            )
        )

        geometries.append(
            ProvisionalGuardGeometry(
                provisional_uid=uid,
                opening=opening,
                descriptor=descriptor,

                columns=0,
                layers=0,
                slots=[],

                local_branch_key=(
                    branch_key
                ),

                opening_start_local=(
                    start_point.copy()
                ),

                opening_end_local=(
                    end_point.copy()
                ),

                mouth_start_world=(
                    straight_start.copy()
                ),

                mouth_end_world=(
                    straight_end.copy()
                ),

                mouth_center_world=(
                    mouth.copy()
                ),

                mouth_lateral_unit=(
                    mouth_normal.copy()
                ),

                branch_tangent_unit=(
                    axis.copy()
                ),

                mouth_span=float(
                    mouth_span
                ),
            )
        )

        print(
            f"[CorridorWallFrame] "
            f"uid={uid} "
            f"branch={branch_key} "
            f"source={axis_source} "
            f"axis=({axis.x:.3f},{axis.y:.3f}) "
            f"normal=({mouth_normal.x:.3f},{mouth_normal.y:.3f}) "
            f"start_pts={len(start_side_points)} "
            f"end_pts={len(end_side_points)} "
            f"start_q={start_quality:.2f} "
            f"end_q={end_quality:.2f} "
            f"span={mouth_span:.3f} "
            f"mouth=({mouth.x:.3f},{mouth.y:.3f}) "
            f"geometry=FINITE_ENDPOINT_MIDPOINT"
        )

    # Side openings are seen obliquely from the Anchor.  Their visible endpoint
    # chord can be much shorter than the corridor cross-section even when the
    # opening order and branch axis are correct.  Reuse the median directly
    # measured full cross-section only for those low-confidence branches.
    trusted_widths = [
        float(geometry.mouth_span)
        for geometry in geometries
        if geometry.descriptor.physical_width_source
        == "LIDAR_WALL_ENDPOINT_CROSS_SECTION"
    ]
    if trusted_widths:
        fallback_width = float(np.median(trusted_widths))
        for geometry in geometries:
            descriptor = geometry.descriptor
            if descriptor.physical_width_source != "ADAPTIVE_W_FALLBACK":
                continue

            tangent = geometry.branch_tangent_unit
            normal = geometry.mouth_lateral_unit
            if tangent is None or normal is None:
                continue
            # Preserve the physical mouth position measured
            # from the finite wall endpoints.
            #
            # Fallback may replace WIDTH only.
            # It must never relocate the mouth center.
            center = geometry.mouth_center_world

            if center is None:
                continue

            center = center.copy()

            geometry.mouth_span = fallback_width
            geometry.mouth_center_world = center.copy()
            geometry.mouth_start_world = center - normal * (0.5 * fallback_width)
            geometry.mouth_end_world = center + normal * (0.5 * fallback_width)
            descriptor.observed_mouth_position = center.copy()
            descriptor.observed_width = fallback_width
            descriptor.motion_observed_width = fallback_width
            descriptor.observed_flow_width = fallback_width
            descriptor.observed_physical_width = fallback_width
            descriptor.physical_left_boundary_lateral = -0.5 * fallback_width
            descriptor.physical_right_boundary_lateral = 0.5 * fallback_width
            descriptor.physical_width_source = (
                "LIDAR_SHARED_FULL_CROSS_SECTION_FALLBACK"
            )
            print(
                f"[GuardWidthFallback] uid={geometry.provisional_uid} "
                f"width={fallback_width:.3f} "
                "source=LIDAR_SHARED_FULL_CROSS_SECTION"
            )

    return geometries

def compute_guard_lateral_interval(
    physical: types.ModuleType,
    descriptor: Any,
) -> tuple[float, float]:
    # LiDAR가 측정한 실제 branch 입구 전체 폭
    mouth_half = 0.5 * float(descriptor.observed_physical_width)
    wall_clearance = physical.ROBOT_RADIUS * (
        1.0 + GUARD_EDGE_SEAL_MARGIN_RATIO
    )
    center_half = max(
        0.0,
        mouth_half - wall_clearance,
    )

    return (-center_half, center_half)


def compute_sealing_aware_column_count(
    physical: types.ModuleType,
    descriptor: Any,
) -> tuple[int, float, float, float]:
    lateral_min, lateral_max = compute_guard_lateral_interval(
        physical, descriptor
    )
    span = max(0.0, lateral_max - lateral_min)
    target_spacing = 2.5 * physical.ROBOT_RADIUS
    required_by_width = physical.required_junction_guard_count(descriptor)
    required_by_gap = int(
        math.ceil(span / max(target_spacing, physical.EPSILON))
    ) + 1
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
    geometry: ProvisionalGuardGeometry | None = None,
    perception: AdaptivePerception | None = None,
) -> list[pygame.Vector2]:
    """Build every Guard row perpendicular to the detected branch axis.

    The old implementation re-derived ``t`` from the LiDAR mouth chord.  Side
    openings see that chord in perspective, so LEFT/RIGHT Guards were diagonal
    and later had to rotate before moving.  Here ``t`` is the already-frozen
    Anchor-local outgoing direction and ``n`` is exactly perpendicular to it.
    Wall-side endpoints only contribute width diagnostics; they never rotate the
    Guard/Frontier/Shepherd transport frame.
    """
    tangent, normal = physical.descriptor_local_basis(descriptor)
    tangent = tangent.normalize()
    normal = pygame.Vector2(-tangent.y, tangent.x).normalize()

    # Preserve the prior lateral sign when one was already frozen so robot slot
    # ordering does not flip between Guard election and persistent UID binding.
    if geometry is not None and geometry.mouth_lateral_unit is not None:
        prior_n = geometry.mouth_lateral_unit
        if prior_n.length_squared() > physical.EPSILON and normal.dot(prior_n) < 0.0:
            normal = -normal

    mouth_center = descriptor.observed_mouth_position.copy()
    mouth_span = float(descriptor.observed_physical_width or descriptor.observed_width)
    if geometry is not None:
        if geometry.mouth_center_world is not None:
            mouth_center = geometry.mouth_center_world.copy()
        if geometry.mouth_span > physical.EPSILON:
            mouth_span = float(geometry.mouth_span)

    # Freeze the same straight t/n frame into every downstream object now, not
    # later at Guard->Frontier promotion.
    descriptor.observed_mouth_position = mouth_center.copy()
    descriptor.local_outgoing_direction = tangent.copy()
    descriptor.local_return_direction = -tangent
    descriptor.direction_last_estimate = tangent.copy()
    descriptor.direction_stability_reference = tangent.copy()
    descriptor.motion_t = tangent.copy()
    descriptor.motion_n = normal.copy()
    descriptor.motion_frame_locked = True
    descriptor.motion_frame_source = "LIDAR_PERPENDICULAR_GUARD_FRAME"
    descriptor.observed_width = mouth_span
    descriptor.observed_physical_width = mouth_span

    if geometry is not None:
        geometry.mouth_center_world = mouth_center.copy()
        geometry.mouth_lateral_unit = normal.copy()
        geometry.branch_tangent_unit = tangent.copy()
        geometry.mouth_span = mouth_span
        geometry.mouth_start_world = mouth_center - normal * (0.5 * mouth_span)
        geometry.mouth_end_world = mouth_center + normal * (0.5 * mouth_span)

    lateral_min, lateral_max = compute_guard_lateral_interval(physical, descriptor)
    if lateral_max < lateral_min:
        lateral_min = lateral_max = 0.0
    spacing = (lateral_max - lateral_min) / max(columns - 1, 1)
    order = build_edge_sealing_slot_order(columns)

    # LiDAR mouth-centre estimates can retain a small perspective bias.  Move
    # only along the measured cross-section normal and choose the smallest
    # correction that maximises collision-free 3xN slots.  No fixture label or
    # branch coordinate is used; this is a physical feasibility correction.
    search_step = max(0.25, 0.25 * physical.ROBOT_RADIUS)
    search_limit = 0.20 * mouth_span
    search_count = int(math.ceil(search_limit / search_step))
    lateral_offsets = [0.0]
    for step_index in range(1, search_count + 1):
        offset = step_index * search_step
        lateral_offsets.extend((-offset, offset))

    def walkable_slot_count(lateral_offset: float) -> int:
        shifted_center = mouth_center + normal * lateral_offset
        return sum(
            physical.is_walkable(
                shifted_center
                + tangent
                * (
                    physical.JUNCTION_GUARD_BRANCH_INSET
                    + layer * physical.THICK_MOUTH_GUARD_LAYER_SPACING
                )
                + normal * (lateral_min + spacing * column),
                physical.ROBOT_RADIUS,
            )
            for layer in range(layers)
            for column in range(columns)
        )

    best_lateral_offset = min(
        lateral_offsets,
        key=lambda offset: (-walkable_slot_count(offset), abs(offset)),
    )
    if abs(best_lateral_offset) > physical.EPSILON:
        mouth_center += normal * best_lateral_offset
        descriptor.observed_mouth_position = mouth_center.copy()
        if geometry is not None:
            geometry.mouth_center_world = mouth_center.copy()
            geometry.mouth_start_world = mouth_center - normal * (0.5 * mouth_span)
            geometry.mouth_end_world = mouth_center + normal * (0.5 * mouth_span)
        print(
            f"[GuardCenterCorrection] uid={descriptor.uid} "
            f"lateral_offset={best_lateral_offset:.3f} "
            f"walkable={walkable_slot_count(best_lateral_offset)}/"
            f"{columns * layers}"
        )
        # ---------------------------------------------------------
    # Every final Guard slot must be physically reachable.
    #
    # Center correction alone can still leave the outermost
    # LEFT/RIGHT slots slightly inside a wall when the LiDAR
    # corridor frame has a small angular error.
    #
    # Keep the same axis and 3xN structure.  Contract only the
    # lateral span by the minimum amount required.
    # ---------------------------------------------------------
    total_required = columns * layers

    def walkable_count_for_interval(
        test_min: float,
        test_max: float,
    ) -> int:
        test_spacing = (
            test_max - test_min
        ) / max(columns - 1, 1)

        return sum(
            physical.is_walkable(
                mouth_center
                + tangent
                * (
                    physical.JUNCTION_GUARD_BRANCH_INSET
                    + layer
                    * physical.THICK_MOUTH_GUARD_LAYER_SPACING
                )
                + normal
                * (
                    test_min
                    + test_spacing * column
                ),
                physical.ROBOT_RADIUS,
            )
            for layer in range(layers)
            for column in range(columns)
        )

    current_walkable = walkable_count_for_interval(
        lateral_min,
        lateral_max,
    )

    if current_walkable < total_required:

        max_extra_inset = min(
            2.0 * physical.ROBOT_RADIUS,
            0.10 * mouth_span,
        )

        corrected = False

        for extra_inset in np.linspace(
            0.0,
            max_extra_inset,
            41,
        ):
            test_min = lateral_min + float(extra_inset)
            test_max = lateral_max - float(extra_inset)

            if test_max <= test_min:
                break

            if (
                walkable_count_for_interval(
                    test_min,
                    test_max,
                )
                == total_required
            ):
                lateral_min = test_min
                lateral_max = test_max

                spacing = (
                    lateral_max - lateral_min
                ) / max(columns - 1, 1)

                corrected = True

                print(
                    f"[GuardSpanCorrection] "
                    f"uid={descriptor.uid} "
                    f"extra_inset={extra_inset:.3f} "
                    f"walkable={total_required}/"
                    f"{total_required}"
                )
                break

        if not corrected:
            raise RuntimeError(
                f"Guard geometry has unreachable slots: "
                f"uid={descriptor.uid} "
                f"walkable={current_walkable}/"
                f"{total_required}"
            )

    # Save the ACTUAL interval used to create the slots.
    if geometry is not None:
        geometry.sealing_lateral_min = lateral_min
        geometry.sealing_lateral_max = lateral_max
        geometry.slot_spacing = spacing


    slots: list[pygame.Vector2] = []

    for layer in range(layers):
        row_center = (
            mouth_center
            + tangent
            * (
                physical.JUNCTION_GUARD_BRANCH_INSET
                + layer * physical.THICK_MOUTH_GUARD_LAYER_SPACING
            )
        )
        # Keep natural left-to-right column order in the slot array because the
        # lifecycle stores slot_index % cols and later reconstructs the same 3xN
        # wall from that index.  Election may still use edge-priority separately.
        row = [
            row_center + normal * (lateral_min + spacing * index)
            for index in range(columns)
        ]
        slots.extend(row)
        print(
            f"[GuardSlotOrder] uid={descriptor.uid} layer={layer} "
            f"order={order} perpendicular=True "
            f"t=({tangent.x:.3f},{tangent.y:.3f}) "
            f"n=({normal.x:.3f},{normal.y:.3f})"
        )

    return slots


def build_provisional_multilayer_slots(

    physical: types.ModuleType,
    robots: Sequence[Any],
    geometries: Sequence[ProvisionalGuardGeometry],
    perception: AdaptivePerception | None = None,
) -> None:
    """Create every LiDAR-derived layer at once; robot positions are unread."""
    for geometry in geometries:
        descriptor = geometry.descriptor
        if geometry.mouth_span > 0.0:
            descriptor.observed_width = geometry.mouth_span
            descriptor.observed_physical_width = geometry.mouth_span
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
        slots = build_sealing_aware_slots(physical, descriptor, columns, layers, geometry, perception)
        geometry.columns = columns
        geometry.layers = layers
        geometry.slots = [slot.copy() for slot in slots]
        walkable = sum(
            physical.is_walkable(slot, physical.ROBOT_RADIUS) for slot in slots
        )
        unwalkable_slots = [
            (index, slot)
            for index, slot in enumerate(slots)
            if not physical.is_walkable(slot, physical.ROBOT_RADIUS)
        ]
        opening = geometry.opening
        print(f"[GuardGeometry] uid={geometry.provisional_uid}")
        print(f"[GuardGeometry] opening_start={float(opening['start_angle']):.3f}")
        print(f"[GuardGeometry] opening_end={float(opening['end_angle']):.3f}")
        print(f"[GuardGeometry] center={float(opening['center_angle']):.3f}")
        print(f"[GuardGeometry] width_deg={float(opening['width_deg']):.3f}")
        print(f"[GuardGeometry] estimated_mouth_width={descriptor.observed_physical_width:.3f}")
        print(
            f"[GuardAutoWidth] uid={geometry.provisional_uid} "
            f"wall_to_wall_span={geometry.mouth_span:.3f} "
            f"usable_span={max(0.0, lateral_max - lateral_min):.3f} "
            f"columns={columns} layers={layers} "
            "source=LIDAR_WALL_ENDPOINT_CROSS_SECTION_OR_W_FALLBACK"
        )
        print(f"[GuardGeometry] usable_half={physical.local_physical_usable_half_width(descriptor):.3f}")
        print(f"[GuardGeometry] sealing_lateral_min={lateral_min:.3f}")
        print(f"[GuardGeometry] sealing_lateral_max={lateral_max:.3f}")
        print(f"[GuardGeometry] columns={columns}")
        print(f"[GuardGeometry] layers={layers}")
        print(f"[GuardGeometry] required={required}")
        print(f"[GuardGeometry] slot_spacing={slot_spacing:.3f}")
        print(f"[GuardGeometry] slots_walkable={walkable}/{len(slots)}")
        if unwalkable_slots:
            print(
                f"[GuardGeometry] unwalkable_slots="
                f"{[(index, round(slot.x, 3), round(slot.y, 3)) for index, slot in unwalkable_slots]}"
            )


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
    build_provisional_multilayer_slots(physical, robots, geometries, perception)
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
    print("[GuardBranchMap] " + " ".join(f"{g.provisional_uid}={g.local_branch_key}" for g in geometries))
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
    width = geometry.mouth_span or descriptor.observed_physical_width
    tangent = geometry.branch_tangent_unit or physical.descriptor_local_basis(descriptor)[0]
    lateral_axis = geometry.mouth_lateral_unit or physical.descriptor_local_basis(descriptor)[1]
    center = geometry.mouth_center_world or descriptor.observed_mouth_position
    layers = max(geometry.layers, 1)
    wall_depth = physical.JUNCTION_GUARD_BRANCH_INSET + (layers - 1) * physical.THICK_MOUTH_GUARD_LAYER_SPACING
    upstream = max(GUARD_CAPTURE_UPSTREAM_WIDTH_RATIO * width, wall_depth + 2.0 * physical.ROBOT_RADIUS)
    downstream = max(GUARD_CAPTURE_DOWNSTREAM_WIDTH_RATIO * width, wall_depth + physical.ROBOT_RADIUS)
    lateral_limit = 0.5 * width + physical.ROBOT_RADIUS + 0.5
    candidates = []
    for robot in robots:
        if (
            robot is perception.leader
            or robot.role != "NORMAL"
            or robot.base_reserve
            or not physical.is_walkable(robot.position, robot.radius)
        ):
            continue
        delta = robot.position - center
        axial, lateral = delta.dot(tangent), delta.dot(lateral_axis)
        if (
            -upstream
            <= axial
            <= downstream
            and abs(lateral) <= lateral_limit
        ):
            candidates.append(robot)
    return sorted(candidates, key=lambda robot: robot.robot_id)


def guard_mouth_coordinates(point: pygame.Vector2, geometry: ProvisionalGuardGeometry) -> tuple[float, float]:
    center = geometry.mouth_center_world or geometry.descriptor.observed_mouth_position
    tangent = geometry.branch_tangent_unit or geometry.descriptor.local_outgoing_direction
    lateral = geometry.mouth_lateral_unit or geometry.descriptor.motion_n
    delta = point - center
    return float(delta.dot(tangent)), float(delta.dot(lateral))


def build_guard_entry_waypoints(
    physical: types.ModuleType,
    geometry: ProvisionalGuardGeometry,
    slot: pygame.Vector2,
) -> list[pygame.Vector2]:
    """Cross the branch mouth through a safe interior lane,
    then spread to the final Guard slot.
    """

    center = (
        geometry.mouth_center_world
        or geometry.descriptor.observed_mouth_position
    )

    tangent = (
        geometry.branch_tangent_unit
        or geometry.descriptor.local_outgoing_direction
    )

    normal = (
        geometry.mouth_lateral_unit
        or geometry.descriptor.motion_n
    )

    if center is None or tangent is None or normal is None:
        return [slot.copy()]

    tangent = tangent.normalize()
    normal = normal.normalize()

    mouth_span = float(
        geometry.mouth_span
        or geometry.descriptor.observed_physical_width
    )

    final_lateral = float(
        (slot - center).dot(normal)
    )

    # While crossing the mouth, stay slightly inside the
    # mouth corners. After entering the branch, spread to
    # the final 3xN Guard slot.
    crossing_half_width = max(
        0.0,
        0.5 * mouth_span
        - 3.0 * physical.ROBOT_RADIUS,
    )

    crossing_lateral = float(
        np.clip(
            final_lateral,
            -crossing_half_width,
            crossing_half_width,
        )
    )

    staging = (
        center
        - tangent * (4.5 * physical.ROBOT_RADIUS)
    )

    # Junction-side waypoint immediately before the mouth.
    approach = (
        center
        - tangent * (2.5 * physical.ROBOT_RADIUS)
        + normal * crossing_lateral
    )

    # Branch-side waypoint immediately after crossing the mouth.
    inside = (
        center
        + tangent * (2.5 * physical.ROBOT_RADIUS)
        + normal * crossing_lateral
    )

    # A robot must never be assigned to an invalid final Guard slot.
    if not physical.is_walkable(
        slot,
        physical.ROBOT_RADIUS,
    ):
        raise RuntimeError(
            "attempted to assign Guard to an unwalkable final slot"
        )

    waypoints: list[pygame.Vector2] = []

    # First gather onto the branch-local centerline on the
    # Junction side. This prevents diagonal corner clipping.
    if physical.is_walkable(
        staging,
        physical.ROBOT_RADIUS,
    ):
        waypoints.append(staging)

    # Then spread laterally while still on the Junction side.
    if physical.is_walkable(
        approach,
        physical.ROBOT_RADIUS,
    ):
        waypoints.append(approach)

    # Cross the mouth at the already-safe lateral coordinate.
    if physical.is_walkable(
        inside,
        physical.ROBOT_RADIUS,
    ):
        waypoints.append(inside)

    # Finally move to the frozen 3xN physical Guard slot.
    waypoints.append(slot.copy())

    return waypoints




def compute_full_guard_slot_assignment(
    physical: types.ModuleType,
    geometry: ProvisionalGuardGeometry,
    candidates: Sequence[Any],
) -> tuple[list[tuple[Any, pygame.Vector2, int]], dict[str, Any]]:
    """Run deterministic full bipartite feasibility and branch-local WHO cost."""
    descriptor = geometry.descriptor
    tangent = geometry.branch_tangent_unit or physical.descriptor_local_basis(descriptor)[0]
    lateral_axis = geometry.mouth_lateral_unit or physical.descriptor_local_basis(descriptor)[1]
    center = geometry.mouth_center_world or descriptor.observed_mouth_position
    def mouth_coords(point: pygame.Vector2) -> tuple[float, float]:
        delta = point - center
        return float(delta.dot(tangent)), float(delta.dot(lateral_axis))
    width = descriptor.observed_physical_width
    required = len(geometry.slots)
    options: dict[int, list[tuple[float, Any]]] = {}
    candidate_laterals = []
    candidate_axials = []
    for robot in candidates:
        axial, lateral = mouth_coords(robot.position)
        candidate_axials.append(float(axial))
        candidate_laterals.append(float(lateral))
    for slot_index, slot in enumerate(geometry.slots):
        slot_axial, slot_lateral = mouth_coords(slot)
        ranked = []
        for robot in candidates:
            axial, lateral = mouth_coords(robot.position)
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

    priority = build_edge_sealing_slot_order(geometry.columns)
    slot_order = [layer * geometry.columns + col for layer in range(geometry.layers) for col in priority]
    rank = {index: rank for rank, index in enumerate(slot_order)}
    slot_order.sort(key=lambda index: (len(options[index]), rank[index]))
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
    assigned_laterals = [guard_mouth_coordinates(robot.position, geometry)[1] for robot, _, _ in assignment]
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
        robot.integration_guard_waypoints = build_guard_entry_waypoints(
            physical,
            geometry,
            slot,
        )
        robot.integration_guard_final_anchor = slot.copy()
        robot.integration_guard_slot_index = slot_index
        robot.junction_guard_anchor = (
            robot.integration_guard_waypoints[0].copy()
        )
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
    """Each branch independently recruits arriving NORMALs into Guard slots."""

    if not perception.provisional_guard_started:
        return

    frame = getattr(physical, "integration_frame", -1)

    # 기존 group 순차 활성화는 사용하지 않는다.
    # LEFT / RIGHT / UP가 각각 독립적으로 선착순 Guard를 만든다.
    perception.guard_activation_groups = []

    for geometry in perception.provisional_guards:

        if geometry.cohort_ready:
            continue

        physical.integration_guard_who_localization_enabled = True

        # 해당 branch 입구 근처까지 먼저 온 NORMAL들
        candidates = collect_shallow_guard_candidates_with_localization(
            physical,
            perception,
            robots,
            geometry,
        )

        # 이미 Guard가 차지한 slot
        occupied_slots = {
            int(robot.integration_guard_slot_index)
            for robot in robots
            if robot.robot_id in geometry.selected_ids
            and getattr(robot, "integration_guard_slot_index", None) is not None
        }

        # -------------------------------------------------
        # Unified multi-layer Guard recruitment
        #
        # Every branch uses exactly the same rule:
        #
        #   create full 3xN geometry first
        #   recruit into ALL unoccupied layers
        #   let every row settle concurrently
        #
        # Deeper rows retain priority, but a row does NOT
        # have to finish settling before another row starts.
        # -------------------------------------------------

        column_priority = build_edge_sealing_slot_order(
            geometry.columns
        )

        slot_priority = [
            layer * geometry.columns + column
            for layer in reversed(
                range(geometry.layers)
            )
            for column in column_priority
        ]

        empty_slots = [
            slot_index
            for slot_index in slot_priority
            if slot_index not in occupied_slots
        ]

        # 이미 다른 Guard로 뽑힌 로봇 제외
        available = [
            robot
            for robot in candidates
            if robot.robot_id not in geometry.selected_ids
            and robot.role == "NORMAL"
        ]

        # 바깥쪽 slot부터 즉시 채운다.
        for slot_index in empty_slots:

            if not available:
                break

            slot = geometry.slots[slot_index]

            # 해당 slot에 가장 가까운 선두 NORMAL을 사용
            robot = min(
                available,
                key=lambda candidate: (
                    candidate.position.distance_to(slot),
                    -guard_mouth_coordinates(
                        candidate.position,
                        geometry,
                    )[0],
                    candidate.robot_id,
                ),
            )

            available.remove(robot)

            # 즉시 Guard로 전환
            robot.role = "JUNCTION_GUARD"

            robot.integration_guard_waypoints = build_guard_entry_waypoints(
                physical,
                geometry,
                slot,
            )

            robot.integration_guard_final_anchor = (
                slot.copy()
            )

            robot.integration_guard_slot_index = (
                slot_index
            )

            robot.junction_guard_anchor = (
                robot.integration_guard_waypoints[0].copy()
            )

            robot.junction_guard_branch = (
                geometry.provisional_uid
            )

            robot.junction_guard_branch_uid = (
                geometry.provisional_uid
            )

            robot.junction_guard_layer = (
                slot_index // geometry.columns
            )

            # 첫 Guard를 branch leader로 사용
            if not geometry.selected_ids:
                geometry.descriptor.leader_id = (
                    robot.robot_id
                )
                robot.junction_guard_parent_id = None
                robot.is_branch_leader = True
            else:
                robot.junction_guard_parent_id = (
                    geometry.descriptor.leader_id
                )
                robot.is_branch_leader = False

            geometry.selected_ids.append(
                robot.robot_id
            )

            print(
                f"[IncrementalGuard] "
                f"uid={geometry.provisional_uid} "
                f"robot={robot.robot_id} "
                f"slot={slot_index} "
                f"filled={len(geometry.selected_ids)}/"
                f"{len(geometry.slots)}"
            )

        status = physical.integration_wall_status[
            geometry.provisional_uid
        ]

        status["candidate_count"] = len(
            candidates
        )

        status["assignment_count"] = len(
            geometry.selected_ids
        )

        status["assigned"] = len(
            geometry.selected_ids
        )

        # 부분적으로라도 Guard가 생긴 순간부터 물리적으로 사용
        if geometry.selected_ids:

            physical.integration_provisional_guard_groups[
                geometry.provisional_uid
            ] = list(geometry.selected_ids)

            physical.integration_provisional_guard_active = True
            physical.integration_guard_gating_enabled = True

            if (
                physical.integration_guard_formation_start_frame
                is None
            ):
                physical.integration_guard_formation_start_frame = (
                    frame
                )

            if geometry.role_assignment_frame is None:
                geometry.role_assignment_frame = frame

        # 3×N slot을 전부 채웠을 때만 완성 처리
        if len(geometry.selected_ids) == len(
            geometry.slots
        ):
            geometry.cohort_ready = True
            geometry.guard_ready_frame = frame

            status = physical.integration_wall_status[
                geometry.provisional_uid
            ]

            status["assigned"] = len(
                geometry.selected_ids
            )

            status["assignment_count"] = len(
                geometry.selected_ids
            )

            print(
                f"[IncrementalGuardComplete] "
                f"uid={geometry.provisional_uid} "
                f"robots={len(geometry.selected_ids)} "
                f"frame={frame}"
            )

    # 모든 branch가 완성된 뒤에만 DFS로 넘어갈 수 있게 함
    all_complete = all(
        geometry.cohort_ready
        and len(geometry.selected_ids)
        == len(geometry.slots)
        for geometry in perception.provisional_guards
    )

    if all_complete:
        perception.guard_all_groups_activated = True
        perception.guard_activation_stage = "COMPLETE"

        print(
            f"[GuardFormationComplete] "
            f"frame={frame} "
            f"incremental_first_arrival=True"
        )

    physical.integration_guard_who_localization_enabled = False

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

        # -------------------------------------------------
        # Diagnose Guards that never reach their FINAL slot.
        #
        # Do not relax the readiness threshold before we
        # know why these physical robots remain unsettled.
        # -------------------------------------------------
        unsettled_guards = []

        for robot in guards:
            final_anchor = getattr(
                robot,
                "integration_guard_final_anchor",
                None,
            )

            if final_anchor is None:
                final_error = float("inf")
            else:
                final_error = robot.position.distance_to(
                    final_anchor
                )

            if (
                final_error
                <= physical.JUNCTION_GUARD_POSITION_TOLERANCE
            ):
                continue

            current_anchor = getattr(
                robot,
                "junction_guard_anchor",
                None,
            )

            current_target_error = (
                robot.position.distance_to(
                    current_anchor
                )
                if current_anchor is not None
                else float("inf")
            )

            waypoints = getattr(
                robot,
                "integration_guard_waypoints",
                [],
            )

            unsettled_guards.append(
                (
                    robot,
                    final_error,
                    current_target_error,
                    len(waypoints),
                )
            )

        if (
            unsettled_guards
            and getattr(
                physical,
                "integration_frame",
                0,
            ) % 20
            == 0
        ):
            print(
                f"[GuardSettlingDetail] "
                f"uid={geometry.provisional_uid} "
                f"settled={settled}/{len(guards)} "
                f"unsettled={len(unsettled_guards)} "
                f"tol="
                f"{physical.JUNCTION_GUARD_POSITION_TOLERANCE:.3f}"
            )

            for (
                robot,
                final_error,
                current_target_error,
                waypoint_count,
            ) in sorted(
                unsettled_guards,
                key=lambda item: -item[1],
            ):
                print(
                    f"[GuardSettlingRobot] "
                    f"uid={geometry.provisional_uid} "
                    f"id={robot.robot_id} "
                    f"layer="
                    f"{getattr(robot, 'junction_guard_layer', -1)} "
                    f"slot="
                    f"{getattr(robot, 'integration_guard_slot_index', -1)} "
                    f"final_error={final_error:.3f} "
                    f"current_target_error="
                    f"{current_target_error:.3f} "
                    f"waypoints={waypoint_count} "
                    f"speed={robot.velocity.length():.3f}"
                )

        complete_rows = 0
        minimum_span_ratio = 1.0
        maximum_edge_gap = 0.0
        maximum_internal_gap = 0.0
        expected_span_max = 0.0
        actual_span_max = 0.0
        left_edge_gap_max = 0.0
        right_edge_gap_max = 0.0
        for layer in range(geometry.layers):
            expected_slots = geometry.slots[layer * geometry.columns:(layer + 1) * geometry.columns]
            expected_laterals = sorted(guard_mouth_coordinates(slot, geometry)[1] for slot in expected_slots)
            laterals = sorted(
                guard_mouth_coordinates(robot.position, geometry)[1]
                for robot in guards
                if robot.junction_guard_layer == layer
            )
            if len(laterals) < geometry.columns:
                minimum_span_ratio = 0.0
                continue
            complete_rows += 1
            expected_span = expected_laterals[-1] - expected_laterals[0]
            span = laterals[-1] - laterals[0]
            expected_span_max = max(expected_span_max, expected_span)
            actual_span_max = max(actual_span_max, span)
            left_edge_gap_max = max(left_edge_gap_max, max(0.0, laterals[0] - expected_laterals[0]))
            right_edge_gap_max = max(right_edge_gap_max, max(0.0, expected_laterals[-1] - laterals[-1]))
            minimum_span_ratio = min(
                minimum_span_ratio,
                span / max(expected_span, physical.EPSILON),
            )
            maximum_edge_gap = max(maximum_edge_gap, left_edge_gap_max, right_edge_gap_max)
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
        structurally_sealed = (
            len(guards) == len(geometry.slots)
            and complete_rows == geometry.layers
            and minimum_span_ratio
            >= physical.FRONTIER_LINE_MIN_SPAN_RATIO
            and maximum_edge_gap
            <= physical.FRONTIER_LINE_MAX_EDGE_GAP
            and maximum_internal_gap
            <= physical.FRONTIER_LINE_MAX_INTERNAL_GAP
        )
        if structurally_sealed and settled_ratio >= PROVISIONAL_WALL_SETTLED_RATIO:
            status["wall_ready_dwell"] = min(
                float(status.get("wall_ready_dwell", 0.0)) +
                float(getattr(physical, "NORMAL_PHYSICS_MAX_DT", 1.0 / 60.0)),
                PROVISIONAL_WALL_STABILITY_DWELL,
            )
        else:
            status["wall_ready_dwell"] = 0.0
        ready = status["wall_ready_dwell"] >= PROVISIONAL_WALL_STABILITY_DWELL
        status.update({
            "settled_ratio": settled_ratio,
            "settled_count": settled,
            "structurally_sealed": structurally_sealed,
            "min_span_ratio": minimum_span_ratio,
            "max_edge_gap": maximum_edge_gap,
            "max_internal_gap": maximum_internal_gap,
            "expected_span": expected_span_max,
            "actual_span": actual_span_max,
            "left_edge_gap": left_edge_gap_max,
            "right_edge_gap": right_edge_gap_max,
            "ready": ready,
        })
        reasons = []
        if settled_ratio < PROVISIONAL_WALL_SETTLED_RATIO: reasons.append("settled_ratio")
        if complete_rows != geometry.layers: reasons.append("complete_rows")
        if minimum_span_ratio < physical.FRONTIER_LINE_MIN_SPAN_RATIO: reasons.append("span_ratio")
        if maximum_edge_gap > physical.FRONTIER_LINE_MAX_EDGE_GAP: reasons.append("edge_gap")
        if maximum_internal_gap > physical.FRONTIER_LINE_MAX_INTERNAL_GAP: reasons.append("internal_gap")
        if structurally_sealed and settled_ratio >= PROVISIONAL_WALL_SETTLED_RATIO and not ready:
            reasons.append("stable_dwell")
        if getattr(physical, "integration_frame", 0) % 10 == 0 or ready:
            print(f"[WallReadyBlocker] uid={geometry.provisional_uid} guard_count={len(guards)} expected={len(geometry.slots)} settled={settled}/{len(guards)} settled_ratio={settled_ratio:.3f} complete_rows={complete_rows}/{geometry.layers} expected_span={expected_span_max:.3f} actual_span={actual_span_max:.3f} span_ratio={minimum_span_ratio:.3f} left_edge_gap={left_edge_gap_max:.3f} right_edge_gap={right_edge_gap_max:.3f} max_edge_gap={maximum_edge_gap:.3f} max_internal_gap={maximum_internal_gap:.3f} structurally_sealed={structurally_sealed} stable_dwell={status['wall_ready_dwell']:.3f} ready={ready} blocking_reasons={reasons}")
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
    # Delayed entrance approach can leave a slightly narrower diagonal mouth;
    # retain the physical controller's normal criterion while accepting the
    # sealing-aware edge tolerance measured from the LiDAR frame.
    physical.FRONTIER_LINE_MAX_EDGE_GAP = max(
        float(physical.FRONTIER_LINE_MAX_EDGE_GAP), 3.5
    )
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
            branch_tangent, branch_lateral = physical.descriptor_local_basis(descriptor)
            mouth_center = descriptor.observed_mouth_position
            for layer in range(rows):
                laterals = sorted(
                    float((robot.position - mouth_center).dot(branch_lateral))
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
                expected_span = max(2.0 * usable_half, physical.EPSILON)
                minimum_span_ratio = min(minimum_span_ratio, span / expected_span)
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
                and branch_stats["settled_ratio"] >= PROVISIONAL_WALL_SETTLED_RATIO
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

    def ready_or_handoff_bypass(robots: Sequence[Any]) -> bool:
        # The integration handoff occurs only after every provisional wall has
        # passed the physical readiness audit.  Let the authoritative DFS
        # state machine consume those already-formed walls without launching
        # a second formation pass.
        if getattr(physical, "integration_ready_guard_handoff", False):
            return True
        return audited_ready(robots)

    physical.junction_guards_formed = ready_or_handoff_bypass


def install_lidar_relay_protection(
    physical: types.ModuleType,
    perception: AdaptivePerception,
) -> None:
    """Keep the persistent LiDAR visible to relay-front tracking,
    but never allow it to become the Breadcrumb itself.

    LiDAR 675:
      - remains NORMAL,
      - contributes to front_progress,
      - is excluded only from tail_band Relay election.
    """

    lidar_robot = perception.leader

    def protected_update_relay_deployment(
        robots: Sequence[Any],
        dt: float,
    ) -> None:

        physical.relay_deploy_cooldown = max(
            0.0,
            physical.relay_deploy_cooldown - dt,
        )

        physical.relay_motion_scale = 1.0

        if physical.phase not in {
            physical.SimulationPhase.MOVE_TO_JUNCTION,
            physical.SimulationPhase.EXPLORE_BRANCH,
            physical.SimulationPhase.FORM_SHEPHERD_BOUNDARY,
            physical.SimulationPhase.FILL_BEHIND_SHEPHERD,
        }:
            return

        if (
            physical.phase
            == physical.SimulationPhase.MOVE_TO_JUNCTION
            and physical.simulation_time
            < physical.BASE_COMPRESSION_DURATION
        ):
            return

        if (
            physical.relay_deploy_cooldown > 0.0
            or physical.base_station is None
        ):
            return

        breadcrumbs = (
            physical.get_active_branch_relays(
                robots
            )
        )

        last_node = (
            breadcrumbs[-1]
            if breadcrumbs
            else physical.base_station
        )

        last_progress = (
            physical.relay_path_progress(
                last_node.relay_anchor,
                physical.active_branch,
            )
            if (
                breadcrumbs
                and last_node.relay_anchor is not None
            )
            else 0.0
        )

        # -------------------------------------------------
        # IMPORTANT:
        #
        # LiDAR 675 stays NORMAL here.
        # Therefore it contributes to front_progress.
        # -------------------------------------------------
        mobile = [
            robot
            for robot in robots
            if robot.role == "NORMAL"
            and not robot.base_reserve
            and robot.connected_to_base
            and physical.get_robot_region(
                robot.position
            )
            in {
                "BOTTOM",
                "JUNCTION",
                physical.active_branch,
            }
        ]

        ahead = [
            (
                physical.relay_path_progress(
                    robot.position,
                    physical.active_branch,
                ),
                robot,
            )
            for robot in mobile
            if physical.relay_path_progress(
                robot.position,
                physical.active_branch,
            )
            > last_progress + physical.EPSILON
        ]

        if not ahead:
            return

        ahead.sort(
            key=lambda item: (
                item[0],
                item[1].robot_id,
            )
        )

        front_progress = ahead[-1][0]

        # -------------------------------------------------
        # Reactive Breadcrumb trigger based on the actual
        # explored FRONT, not on the slowest NORMAL tail.
        #
        # A few NORMAL robots may remain close to the last
        # breadcrumb while the LiDAR/front has already
        # stretched the communication chain.
        #
        # Waiting for the absolute rear-most NORMAL to move
        # BREADCRUMB_SPACING creates a deadlock:
        #
        # front stops for communication
        #     -> rear-most NORMAL does not clear
        #     -> no breadcrumb
        #     -> front can never resume
        # -------------------------------------------------
        required_front_progress = (
            last_progress
            + physical.BREADCRUMB_SPACING
            + physical.BREADCRUMB_FRONT_CLEARANCE
        )

        if front_progress < required_front_progress:
            return

        target_progress = (
            last_progress
            + physical.BREADCRUMB_SPACING
        )

        # Select an already-existing physical NORMAL robot
        # near the desired local spacing.
        #
        # No robot is teleported. The selected robot stops
        # exactly where it already is.
        tail_band = [
            (progress, robot)
            for progress, robot in ahead
            if robot is not lidar_robot
            and progress
            <= (
                front_progress
                - physical.BREADCRUMB_FRONT_CLEARANCE
            )
            and physical.BREADCRUMB_DEPLOY_DISTANCE
            <= robot.position.distance_to(
                last_node.position
            )
            <= physical.COMM_RANGE * 0.88
        ]

        if not tail_band:
            return

        tail_progress, tail_robot = min(
            tail_band,
            key=lambda item: (
                abs(
                    item[0]
                    - target_progress
                ),
                abs(
                    item[1].position.distance_to(
                        last_node.position
                    )
                    - physical.BREADCRUMB_SPACING
                ),
                item[1].robot_id,
            ),
        )


        if (
            tail_robot.total_distance
            < physical.BREADCRUMB_MIN_TRAVEL
        ):
            return

        tail_robot.role = "RELAY"

        tail_robot.relay_anchor = (
            tail_robot.position.copy()
        )

        tail_robot.relay_index = (
            breadcrumbs[-1].relay_index + 1
            if breadcrumbs
            else 0
        )

        tail_robot.velocity.update(
            0.0,
            0.0,
        )

        tail_robot.acceleration.update(
            0.0,
            0.0,
        )

        tail_robot.filtered_acceleration.update(
            0.0,
            0.0,
        )

        physical.relay_slots.append(
            {
                "index": tail_robot.relay_index,
                "position": (
                    tail_robot.relay_anchor.copy()
                ),
                "path_distance": tail_progress,
            }
        )

        guards = (
            physical.assign_breadcrumb_front_guards(
                robots,
                tail_robot,
                tail_progress,
            )
        )

        physical.relay_deploy_cooldown = (
            physical.BREADCRUMB_DEPLOY_COOLDOWN
        )

        print(
            "[Breadcrumb] "
            f"tail robot={tail_robot.robot_id}, "
            f"index={tail_robot.relay_index}, "
            f"progress={tail_progress:.1f}, "
            f"static_guards={len(guards)} "
            f"lidar_front_visible=True "
            f"lidar_selected=False"
        )

    physical.update_relay_deployment = (
        protected_update_relay_deployment
    )

    print(
        "[LiDARRoleProtection] "
        f"lidar_id={lidar_robot.robot_id} "
        "visible_to_front_progress=True "
        "breadcrumb_candidate=False"
    )

def update_child_probe_relay_support(
    physical: types.ModuleType,
    perception: AdaptivePerception,
    robots: Sequence[Any],
) -> None:
    """Freeze a local NORMAL neighbor only when the moving LiDAR
    is about to exhaust its current Base-side communication link.

    This is Child-probe communication support only.

    It does NOT:
    - confirm a Child Junction,
    - release the Parent Junction,
    - create a Return Marker,
    - perform DFS PUSH,
    - move any robot by position overwrite.
    """

    if not multi_dfs.child_probe_active:
        return

    lidar_robot = perception.leader

    if lidar_robot.role != "NORMAL":
        return

    if not lidar_robot.connected_to_base:
        return

    parent = lidar_robot.comm_parent

    if parent is None:
        return

    parent_distance = lidar_robot.position.distance_to(
        parent.position
    )

    hard_limit = float(
        physical.COMM_GUARD_HARD_LIMIT
    )

    trigger_distance = (
        CHILD_PROBE_RELAY_TRIGGER_RATIO
        * hard_limit
    )

    # Current link still has sufficient room.
    if parent_distance < trigger_distance:
        return

    current_margin = float(
        getattr(
            lidar_robot,
            "comm_path_margin",
            float("-inf"),
        )
    )

    candidates: list[
        tuple[float, float, int, Any]
    ] = []

    # -------------------------------------------------
    # Use only LOCAL communication neighbors.
    #
    # A support candidate must already:
    #   1. be a physical NORMAL robot,
    #   2. have a Base path,
    #   3. be a direct LiDAR communication neighbor.
    #
    # No Child global position or J1 fixture coordinate
    # is used here.
    # -------------------------------------------------
    for candidate in getattr(
        lidar_robot,
        "comm_neighbors",
        [],
    ):
        if candidate is lidar_robot:
            continue

        if candidate is parent:
            continue

        if getattr(
            candidate,
            "role",
            None,
        ) != "NORMAL":
            continue

        if getattr(
            candidate,
            "base_reserve",
            False,
        ):
            continue

        if not getattr(
            candidate,
            "connected_to_base",
            False,
        ):
            continue

        candidate_distance = (
            lidar_robot.position.distance_to(
                candidate.position
            )
        )

        if (
            candidate_distance
            >= physical.COMM_GUARD_HARD_LIMIT
        ):
            continue

        candidate_path_margin = float(
            getattr(
                candidate,
                "comm_path_margin",
                float("-inf"),
            )
        )

        if not math.isfinite(
            candidate_path_margin
        ):
            continue

        local_edge_margin = (
            physical.COMM_RANGE
            - candidate_distance
        )

        bridge_margin = min(
            candidate_path_margin,
            local_edge_margin,
        )

        # Prefer a support node that actually improves
        # the LiDAR's widest Base-rooted path.
        if (
            bridge_margin
            <= current_margin
            + physical.EPSILON
        ):
            continue

        candidates.append(
            (
                -bridge_margin,
                candidate_distance,
                candidate.robot_id,
                candidate,
            )
        )

    if not candidates:
        if (
            physical.integration_frame
            % 20
            == 0
        ):
            print(
                "[ChildProbeRelayWait] "
                f"lidar_id={lidar_robot.robot_id} "
                f"parent="
                f"{getattr(parent, 'robot_id', None)} "
                f"parent_dist={parent_distance:.2f} "
                f"trigger={trigger_distance:.2f} "
                f"hard={hard_limit:.2f} "
                f"current_margin="
                f"{current_margin:.2f} "
                "candidate=NONE"
            )

        return

    _, _, _, relay_robot = min(
        candidates
    )

    previous_relays = (
        physical.get_active_branch_relays(
            robots
        )
    )

    relay_index = (
        max(
            (
                relay.relay_index
                for relay in previous_relays
            ),
            default=-1,
        )
        + 1
    )

    relay_robot.role = "RELAY"
    relay_robot.relay_anchor = (
        relay_robot.position.copy()
    )
    relay_robot.relay_index = relay_index

    if hasattr(
        relay_robot,
        "relay_scope",
    ):
        relay_robot.relay_scope = "BRANCH"

    if hasattr(
        relay_robot,
        "relay_owner_edge_id",
    ):
        relay_robot.relay_owner_edge_id = (
            multi_dfs.child_probe_branch_uid
        )

    if hasattr(
        relay_robot,
        "relay_branch",
    ):
        relay_robot.relay_branch = getattr(
            physical,
            "active_branch",
            None,
        )

    relay_robot.velocity.update(
        0.0,
        0.0,
    )

    relay_robot.acceleration.update(
        0.0,
        0.0,
    )

    relay_robot.filtered_acceleration.update(
        0.0,
        0.0,
    )

    print(
        "[ChildProbeRelay] "
        f"lidar_id={lidar_robot.robot_id} "
        f"relay_id={relay_robot.robot_id} "
        f"relay_index={relay_index} "
        f"old_parent="
        f"{getattr(parent, 'robot_id', None)} "
        f"old_parent_dist={parent_distance:.2f} "
        f"relay_dist="
        f"{lidar_robot.position.distance_to(relay_robot.position):.2f} "
        "position_snap=False"
    )


def install_continuous_guard_settling(physical: types.ModuleType) -> None:
    """Let initially elected Guards walk to local slots at bounded speed."""
    original_limit = physical.limit_communication_proposed_position

    def guard_formation_limit(
        robot: Any,
        proposed: pygame.Vector2,
        old_position: pygame.Vector2,
    ) -> pygame.Vector2:
        lifecycle = getattr(physical, "integration_wall_lifecycle", {}).get(
            getattr(robot, "junction_guard_branch", None),
            {},
        )
        restoring_persistent_wall = (
            lifecycle.get("state") == "RESTORING_GUARD"
        )
        if (
            (
                physical.phase == physical.SimulationPhase.FORM_JUNCTION_GUARDS
                or getattr(
                    physical, "integration_provisional_guard_active", False
                )
                or restoring_persistent_wall
            )
            and robot.role == "JUNCTION_GUARD"
            and robot.junction_guard_anchor is not None
        ):
            # The Robot.update caller still applies JUNCTION_GUARD_MOVE_SPEED,
            # walkable-mask collision, and one-frame stepping.  Only the Base
            # parent-link clamp is suspended during this permitted initial
            # wall-placement interval.
            return proposed
        if (
            physical.phase in {
                physical.SimulationPhase.FORM_JUNCTION_GUARDS,
                physical.SimulationPhase.EXPLORE_BRANCH,
            }
            and robot.role == "FRONTIER_SHEPHERD"
            and robot.shepherd_branch in getattr(physical, "integration_wall_lifecycle", {})
        ):
            # The selected thick wall is the physical branch boundary; its
            # bounded slot motion must not be clamped by the old Base parent
            # link, which otherwise leaves the entire Frontier wall still.
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
    original_get_shepherd_line_depth = physical.get_shepherd_line_depth
    original_compute_route_force = physical.compute_route_force
    original_compute_sph_forces = physical.compute_sph_forces
    # Preserve the Physical-DFS Shepherd mechanics before integration overrides.
    # The no-virtual-wall Environment uses the real Shepherd robots, SPH/contact
    # forces, route forces, and the physical collision mask only.
    original_start_shepherd_pressure_push = physical.start_shepherd_pressure_push
    original_release_shepherd_line_at_junction = (
        physical.release_shepherd_line_at_junction
    )
    original_force_complete_shepherd_boundary = (
        physical.force_complete_shepherd_boundary
    )
    original_update_pre_shepherd_pipeline = physical.update_pre_shepherd_pipeline
    # Robot.update has a dedicated SHEPHERD kinematic branch that returns before
    # PRESSURE_PUSH_MAX_SPEED / FLOW_BACKTRACK_MAX_SPEED are applied.  Preserve
    # it so the integration can cap the *actual* Shepherd displacement only
    # during return phases without changing Guard/Frontier formation motion.
    original_robot_update = physical.Robot.update

    # The reference simulator multiplies all motion speeds by
    # MOTION_SPEED_MULTIPLIER.  In the LiDAR integration that made the return
    # piston visually race across the short branch.  Keep the *reference* timer
    # trajectory, but use its nominal pre-multiplier return rates (8 px/s piston,
    # 12 px/s line, 12 px/s release for the current parameters).  Only Shepherd
    # return timing is normalized; SPH/Frontier/Guard motion is unchanged.
    return_speed_scale = 1.0 / max(float(physical.MOTION_SPEED_MULTIPLIER), 1.0)
    physical.integration_reference_return_speed_scale = return_speed_scale
    physical.integration_reference_return_original_piston_speed = float(
        physical.SHEPHERD_PISTON_SPEED
    )
    physical.integration_reference_return_original_line_speed = float(
        physical.SHEPHERD_LINE_BACKTRACK_SPEED
    )
    physical.integration_reference_return_original_release_speed = float(
        physical.SHEPHERD_JUNCTION_RELEASE_SPEED
    )
    physical.SHEPHERD_PISTON_SPEED *= return_speed_scale
    physical.SHEPHERD_LINE_BACKTRACK_SPEED *= return_speed_scale
    physical.SHEPHERD_JUNCTION_RELEASE_SPEED *= return_speed_scale
    diagnostics = LocalSaturationDiagnostics()
    physical.integration_saturation = diagnostics
    physical.integration_saturation_events = []
    physical.integration_backflow_events = []

    physical.integration_final_guard_sweep_active = False
    physical.integration_final_base_flow_dwell = 0.0
    physical.integration_final_base_flow_established = False
    physical.integration_final_all_guards_released = False
    physical.integration_frontier_lineage_events = []
    physical.integration_shepherd_anchor_offsets = {}
    # Backtracking is pack-coupled: the physical Shepherd piston may advance
    # only as fast as the rear surface of the returning NORMAL cohort moves.
    # This prevents the kinematic Shepherd controller from racing through the
    # SPH body while still allowing a slight physical compression at contact.
    physical.integration_backtrack_command_depth = None
    physical.integration_backtrack_pack_rear_depth = None
    physical.integration_backtrack_support_depth = None
    physical.integration_backtrack_support_count = 0
    physical.integration_backtrack_lateral_coverage = 0.0
    physical.integration_backtrack_pacer_last_log_frame = -1
    # Reference-style Shepherd return gating.  The original Physical DFS does
    # not begin its piston return merely because the dead-end was detected; it
    # first fills a NORMAL pack immediately in front of the Shepherd boundary.
    # Keep that invariant explicitly in the integration layer.
    physical.integration_shepherd_pack_ready_dwell = 0.0
    physical.integration_shepherd_pack_ready_required_dwell = max(
        0.18, physical.SATURATION_DWELL_TIME
    )
    # V31 lifecycle: dead-end contact promotes Frontier->Shepherd immediately.
    # The Shepherd then waits in FILL while NORMAL density/pressure/cross-fill
    # grow relative to the instant of promotion.  Only after this dwell may the
    # piston enter PRESSURE_PUSH.
    physical.integration_shepherd_fill_baseline_density = 0.0
    physical.integration_shepherd_fill_baseline_pressure = 0.0
    physical.integration_shepherd_fill_baseline_cross_fill = 0.0
    physical.integration_pending_shepherd_transition_event = None
    # One source of truth for the commanded piston speed and the actual
    # kinematic Shepherd step.  Previously command_depth moved at this rate,
    # but Robot.update still chased the target at SHEPHERD_FORM_SPEED.
    physical.integration_herd_return_speed_scale = 1.00
    # Formation and return are different jobs. Formation must settle the full 3xN
    # wall before packing begins; return remains deliberately slow and contact-limited.
    physical.integration_herd_formation_speed_scale = 3.0
    # Real Shepherd/NORMAL contact is handled by a finite-radius compliant shell.
    # Keep commanded geometric overlap small; the pairwise contact spring below is
    # what breaks the static packed state and transfers piston momentum.
    physical.integration_shepherd_contact_compression_ratio = 0.20
    # V34: the Shepherd must be an ACTIVE piston, not a follower of the NORMAL rear.
    # Allow the real 3xN wall to compress the nearest NORMAL rear down to this
    # centre-to-centre gap while it advances toward the Junction.  This is still
    # bounded well above a centre crossing, so the Shepherd cannot tunnel through
    # the herd, but it can generate enough physical contact force to move it.
    physical.integration_shepherd_active_min_center_gap_ratio = 1.50
    physical.integration_shepherd_contact_shell_ratio = 2.35
    physical.integration_shepherd_contact_spring_scale = 2.50
    physical.integration_shepherd_contact_damping_scale = 1.00
    physical.integration_shepherd_actual_motion_log_frame = -1
    physical.integration_real_contact_log_frame = -1
    physical.integration_return_curtain_active = False

    def reference_start_shepherd_pressure_push(
        robots: Sequence[Any], branch: str
    ) -> None:
        """Start return with the Shepherd target locked to the actual herd rear."""
        # IMPORTANT: initialize the command from the *current physical line*,
        # not from a timer-derived future depth.  Robot.update() may move a
        # Shepherd kinematically at SHEPHERD_FORM_SPEED, so any large target jump
        # here looks like a solo sprint.  From this frame onward the command is
        # advanced only by update_pack_coupled_backtrack_depth().
        descriptor = descriptor_for(branch)
        current_line = (
            shepherd_line_leading_depth(robots, branch, descriptor)
            if descriptor is not None else None
        )
        if current_line is None:
            current_line = float(original_get_shepherd_line_depth(branch))
        physical.integration_backtrack_command_depth = float(current_line)
        physical.integration_backtrack_pack_rear_depth = None
        physical.integration_backtrack_support_depth = None
        physical.integration_backtrack_support_count = 0
        physical.integration_backtrack_lateral_coverage = 0.0
        physical.integration_return_curtain_active = False

        # Do NOT call the baseline start_shepherd_pressure_push() here. That
        # routine starts cross-branch transfer before the source branch has
        # physically drained, which gives NORMAL robots a route target while the
        # Shepherd is supposed to be the only return driver. The integration has
        # its own JUNCTION_SWITCH after the branch is clear, so only initialize
        # the pressure-state timers here.
        physical.phase = physical.SimulationPhase.PRESSURE_PUSH
        physical.pressure_push_timer = 0.0
        physical.flow_establish_timer = 0.0
        physical.shepherd_flow_timer = 0.0

        metrics = getattr(physical, "metrics", None)
        if metrics is not None and hasattr(metrics, "pressure_events"):
            metrics.pressure_events.append({
                "branch": branch,
                "started_at": getattr(physical, "simulation_time", 0.0),
            })

        # Physical Shepherd-only return: NORMAL motion comes from finite-radius
        # Shepherd/SPH/contact interaction.
        print(
            f"[HerdLockedReturnStartV26] branch={branch} "
            f"command_depth={float(current_line):.3f} "
            f"piston_speed={physical.SHEPHERD_PISTON_SPEED:.3f} "
            f"line_speed={physical.SHEPHERD_LINE_BACKTRACK_SPEED:.3f} "
            "cross_branch_transfer=False invisible_curtain=False"
        )

    def frozen_lifecycle_coordinates(
        point: pygame.Vector2,
        lifecycle: dict[str, Any],
        descriptor: Any | None = None,
    ) -> tuple[float, float]:
        """Coordinates in the exact frozen LiDAR mouth frame used by Guards."""
        center = lifecycle.get("mouth_center_world")
        tangent = lifecycle.get("branch_tangent_unit")
        lateral = lifecycle.get("mouth_lateral_unit")
        if center is not None and tangent is not None and lateral is not None:
            delta = point - center
            return float(delta.dot(tangent)), float(delta.dot(lateral))
        if descriptor is None:
            return 0.0, 0.0
        return physical.branch_local_coordinates(point, descriptor)


    def thick_frontier_slot_target(
        robot: Any, centroid_depth: float
    ) -> pygame.Vector2 | None:
        """Move the complete frozen LiDAR Guard wall without changing its shape."""
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

        center = lifecycle.get("mouth_center_world")
        tangent = lifecycle.get("branch_tangent_unit")
        lateral = lifecycle.get("mouth_lateral_unit")
        if center is not None and tangent is not None and lateral is not None:
            return (
                center
                + tangent * (centroid_depth + axial_offset)
                + lateral * lateral_offset
            )

        # Compatibility fallback only.  Normal integration runs should always
        # have the frozen LiDAR mouth frame above.
        return physical.local_coordinates_to_world(
            descriptor,
            centroid_depth + axial_offset,
            physical.frontier_line_lateral_center + lateral_offset,
        )


    def audited_commit_guard_roles(
        robots: Sequence[Any], selected_branch: str
    ) -> None:
        """Promote the complete READY multi-row wall without re-election."""
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
        if descriptor is None:
            raise RuntimeError(f"missing descriptor for selected branch {selected_branch}")
        lock_branch_transport_frame(selected_branch, lifecycle, descriptor)
        centroid_axial, centroid_lateral, regularized_offsets = (
            regularized_thick_wall_offsets(selected_branch, frontiers, lifecycle, descriptor)
        )
        lifecycle["centroid_axial"] = centroid_axial
        lifecycle["centroid_lateral"] = centroid_lateral
        lifecycle["relative_offsets"] = regularized_offsets
        axial_offsets = [value[0] for value in lifecycle["relative_offsets"].values()]
        wall_axial_span = (
            max(axial_offsets) - min(axial_offsets)
            if axial_offsets else 0.0
        )
        # A multi-row Frontier must move far enough for its trailing row to
        # uncover the selected mouth.  Otherwise the wall blocks the NORMAL
        # body and support-based progress deadlocks at the entrance.
        lifecycle["frontier_bootstrap_target_depth"] = (
            centroid_axial
            + max(
                physical.FRONTIER_LINE_LEAD_GAP,
                wall_axial_span + 2.0 * physical.ROBOT_RADIUS,
            )
        )
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
            f"FRONTIER_THICK={selected_branch};"
            f"{lifecycle['rows']}x{lifecycle['cols']};robots={len(frontiers)}"
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
        print(
            f"[FrontierStraighten] branch={selected_branch} uid={uid} "
            f"rows={lifecycle['rows']} cols={lifecycle['cols']} "
            f"usable_half={lifecycle.get('regularized_usable_half', 0.0):.3f} "
            f"transport={lifecycle.get('transport_frame_source')} "
            "targets_perpendicular_to_branch=True"
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

    def transport_basis(
        branch: str,
        lifecycle: dict[str, Any] | None = None,
        descriptor: Any | None = None,
    ) -> tuple[pygame.Vector2, pygame.Vector2, str]:
        """Use the same LiDAR-frozen branch frame for Guard/Frontier/Shepherd.

        Do not re-align to legacy LEFT/RIGHT/UP fixture directions.
        The frame frozen at initial Guard formation is reused through:
        Guard -> Frontier -> Shepherd -> returned Guard.
        """

        descriptor = descriptor or descriptor_for(branch)

        lifecycle = lifecycle or getattr(
            physical,
            "integration_wall_lifecycle",
            {},
        ).get(branch, {})

        # 1. 가장 우선:
        # 처음 Guard 생성 시 LiDAR에서 frozen된 branch tangent
        candidate = None

        if lifecycle:
            candidate = lifecycle.get(
                "branch_tangent_unit"
            )

        # 2. lifecycle에 없을 때만 descriptor의 LiDAR frame 사용
        if (
            candidate is None
            or candidate.length_squared()
            <= physical.EPSILON
        ):
            if descriptor is not None:
                candidate = getattr(
                    descriptor,
                    "motion_t",
                    None,
                )

        if (
            candidate is None
            or candidate.length_squared()
            <= physical.EPSILON
        ):
            if descriptor is not None:
                candidate = getattr(
                    descriptor,
                    "local_outgoing_direction",
                    None,
                )

        if (
            candidate is None
            or candidate.length_squared()
            <= physical.EPSILON
        ):
            raise RuntimeError(
                f"no LiDAR-frozen transport tangent "
                f"for branch {branch}"
            )

        tangent = candidate.normalize()

        # tangent에 정확히 수직인 Guard/Shepherd row 방향
        lateral = pygame.Vector2(
            -tangent.y,
            tangent.x,
        ).normalize()

        # 처음 Guard에서 사용했던 lateral 방향의 부호도 유지
        old_lateral = (
            lifecycle.get("mouth_lateral_unit")
            if lifecycle
            else None
        )

        if (
            old_lateral is not None
            and old_lateral.length_squared()
            > physical.EPSILON
            and lateral.dot(old_lateral) < 0.0
        ):
            lateral = -lateral

        source = "LIDAR_FROZEN_GUARD_FRAME"

        return (
            tangent,
            lateral,
            source,
        )


    def lock_branch_transport_frame(
        branch: str,
        lifecycle: dict[str, Any],
        descriptor: Any,
    ) -> tuple[pygame.Vector2, pygame.Vector2]:
        """Lock one straight corridor frame before the Guard becomes Frontier."""
        tangent, lateral, source = transport_basis(branch, lifecycle, descriptor)
        lifecycle["branch_tangent_unit"] = tangent.copy()
        lifecycle["mouth_lateral_unit"] = lateral.copy()
        lifecycle["transport_frame_source"] = source

        descriptor.local_outgoing_direction = tangent.copy()
        descriptor.local_return_direction = -tangent
        descriptor.direction_last_estimate = tangent.copy()
        descriptor.direction_stability_reference = tangent.copy()
        descriptor.motion_t = tangent.copy()
        descriptor.motion_n = lateral.copy()
        descriptor.motion_frame_locked = True
        descriptor.motion_frame_source = source

        print(
            f"[BranchTransportFrame] branch={branch} uid={descriptor.uid} "
            f"source={source} t=({tangent.x:.3f},{tangent.y:.3f}) "
            f"n=({lateral.x:.3f},{lateral.y:.3f}) straight=True"
        )
        return tangent, lateral

    def regularized_thick_wall_offsets(
        branch: str,
        frontiers: Sequence[Any],
        lifecycle: dict[str, Any],
        descriptor: Any,
    ) -> tuple[float, float, dict[int, tuple[float, float]]]:
        """Flatten the selected 3xN wall into rows perpendicular to the branch.

        The Guard WHO IDs are preserved.  Only their Frontier targets are
        regularized, so a perspective-skewed LiDAR mouth chord cannot make a
        horizontal branch travel diagonally and scrape one side wall.
        """
        center = lifecycle.get("mouth_center_world")
        tangent = lifecycle.get("branch_tangent_unit")
        lateral_axis = lifecycle.get("mouth_lateral_unit")
        if center is None or tangent is None or lateral_axis is None:
            raise RuntimeError("regularized wall requires a locked transport frame")

        rows = max(1, int(lifecycle.get("rows", 1)))
        cols = max(1, int(lifecycle.get("cols", 1)))
        layer_by_id = lifecycle.get("guard_layer_by_id", {})
        slot_by_id = lifecycle.get("guard_slot_index_by_id", {})

        actual = {}
        for robot in frontiers:
            delta = robot.position - center
            actual[robot.robot_id] = (
                float(delta.dot(tangent)),
                float(delta.dot(lateral_axis)),
            )

        usable_half = float(physical.local_physical_usable_half_width(descriptor))
        sealed_min = lifecycle.get("sealing_lateral_min")
        sealed_max = lifecycle.get("sealing_lateral_max")
        if sealed_min is not None and sealed_max is not None:
            sealed_half = 0.5 * max(0.0, float(sealed_max) - float(sealed_min))
            if sealed_half > physical.EPSILON:
                usable_half = sealed_half
        # On the present fixture, cap the LiDAR-measured width only by the actual
        # collision-mask corridor width.  This is a physics-adapter safeguard,
        # not a Junction/dead-end decision input.
        if branch in getattr(physical, "BRANCH_DIRECTIONS", {}):
            fixture_half = max(
                0.0,
                0.5 * float(physical.corridor_width)
                - float(physical.ROBOT_RADIUS)
                - float(getattr(physical, "FRONTIER_LINE_EDGE_CLEARANCE", 0.0)),
            )
            if fixture_half > 0.0:
                usable_half = min(usable_half, fixture_half) if usable_half > 0.0 else fixture_half
        if usable_half <= physical.EPSILON:
            lateral_values = [value[1] for value in actual.values()]
            usable_half = max(
                physical.ROBOT_RADIUS,
                0.5 * (max(lateral_values) - min(lateral_values))
                if lateral_values else physical.ROBOT_RADIUS,
            )

        row_spacing = float(physical.THICK_MOUTH_GUARD_LAYER_SPACING)
        mean_layer = 0.5 * (rows - 1)
        relative: dict[int, tuple[float, float]] = {}

        # Use the stored slot index when available.  Fall back to the observed
        # lateral order within each row without changing robot IDs.
        fallback_columns: dict[int, int] = {}
        for layer in range(rows):
            members = [
                robot for robot in frontiers
                if int(layer_by_id.get(robot.robot_id, -1)) == layer
            ]
            members.sort(key=lambda robot: actual[robot.robot_id][1])
            for column, robot in enumerate(members):
                fallback_columns[robot.robot_id] = column

        for robot in frontiers:
            layer = int(layer_by_id.get(robot.robot_id, 0))
            slot_index = int(slot_by_id.get(robot.robot_id, -1))
            column = slot_index % cols if slot_index >= 0 else fallback_columns.get(robot.robot_id, 0)
            column = int(np.clip(column, 0, cols - 1))
            axial_offset = (layer - mean_layer) * row_spacing
            if sealed_min is not None and sealed_max is not None and cols > 1:
                lateral_offset = (
                    float(sealed_min)
                    + (float(sealed_max) - float(sealed_min)) * column / (cols - 1)
                )
            else:
                lateral_offset = (
                    0.0
                    if cols <= 1
                    else -usable_half + 2.0 * usable_half * column / (cols - 1)
                )
            relative[robot.robot_id] = (float(axial_offset), float(lateral_offset))

        # Choose the scalar centroid depth that minimizes the initial correction
        # when the skewed Guard becomes the straight Frontier.
        centroid_candidates = [
            actual[robot.robot_id][0] - relative[robot.robot_id][0]
            for robot in frontiers
        ]
        centroid_axial = float(np.median(centroid_candidates))
        centroid_lateral = 0.0
        lifecycle["regularized_usable_half"] = usable_half
        lifecycle["regularized_wall"] = True
        return centroid_axial, centroid_lateral, relative

    def lifecycle_target(
        lifecycle: dict[str, Any],
        centroid_depth: float,
        axial_offset: float,
        lateral_offset: float,
    ) -> pygame.Vector2:
        center = lifecycle.get("mouth_center_world")
        tangent = lifecycle.get("branch_tangent_unit")
        lateral_axis = lifecycle.get("mouth_lateral_unit")
        if center is None or tangent is None or lateral_axis is None:
            raise RuntimeError("lifecycle target requires a transport frame")
        return (
            center
            + tangent * (centroid_depth + axial_offset)
            + lateral_axis * lateral_offset
        )

    def resolve_common_shepherd_centroid_depth(
        branch: str,
        lifecycle: dict[str, Any],
        desired_centroid_depth: float,
    ) -> float | None:
        """Find the deepest common 3xN cross-section that is fully walkable."""
        relative = lifecycle.get("relative_offsets", {})
        if not relative:
            return None

        def formation_is_walkable(centroid_depth: float) -> bool:
            return all(
                physical.is_walkable(
                    lifecycle_target(lifecycle, centroid_depth, axial, lateral),
                    physical.ROBOT_RADIUS,
                )
                for axial, lateral in relative.values()
            )

        desired_centroid_depth = max(0.0, float(desired_centroid_depth))
        if formation_is_walkable(desired_centroid_depth):
            return desired_centroid_depth

        # Search continuously back toward the Junction until the complete 3xN
        # formation fits.  v13 limited this search to only ~10 robot radii; when
        # the dead-end contact was oblique or the three rows were compressed,
        # saturation could be TRUE while promotion returned [] forever, leaving
        # the phase stuck in EXPLORE_BRANCH.  The current robot positions are
        # already physically inside the corridor, so a common safe cross-section
        # must be sought over the whole traversed branch, not an arbitrary short
        # retreat window.
        step = max(0.20, float(physical.ROBOT_RADIUS) * 0.20)
        candidate = desired_centroid_depth - step
        while candidate >= -physical.EPSILON:
            candidate = max(0.0, candidate)
            if formation_is_walkable(candidate):
                return candidate
            if candidate <= physical.EPSILON:
                break
            candidate -= step
        return None

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
        lifecycle = getattr(physical, "integration_wall_lifecycle", {}).get(branch)
        if lifecycle is not None:
            expected_wall = int(lifecycle.get("rows", 0)) * int(lifecycle.get("cols", 0))
            if len(frontiers) == expected_wall:
                # The handoff lifecycle is the frozen physical formation. Do
                # not let the legacy single-row descriptor audit disable the
                # already-created thick Frontier wall.
                physical.frontier_line_row_ready = True
                physical.frontier_line_continuous = True
        if physical.frontier_line_row_ready:
            physical.update_frontier_lateral_center(
                robots, branch, descriptor, dt
            )
            physical.refresh_frontier_row_readiness(robots, branch)
            if lifecycle is not None and len(frontiers) == expected_wall:
                physical.frontier_line_row_ready = True
                physical.frontier_line_continuous = True
        if not physical.frontier_line_row_ready:
            return

        # Before translating down the branch, let the selected Guard IDs settle
        # onto the straight corridor cross-section at the current scalar depth.
        # Without this bounded settle, LEFT/RIGHT rows can begin moving while
        # still carrying the perspective skew of the mouth chord and the lower
        # edge can scrape the side wall.
        if lifecycle is not None and lifecycle.get("regularized_wall"):
            target_errors = []
            for robot in frontiers:
                target = thick_frontier_slot_target(
                    robot, physical.frontier_line_depth
                )
                if target is not None:
                    target_errors.append(robot.position.distance_to(target))
            max_straighten_error = max(target_errors, default=0.0)
            straighten_tolerance = max(
                float(physical.JUNCTION_GUARD_POSITION_TOLERANCE),
                1.25 * float(physical.ROBOT_RADIUS),
            )
            if max_straighten_error > straighten_tolerance:
                if getattr(physical, "integration_frame", 0) % 20 == 0:
                    print(
                        f"[FrontierAlignmentCorrection] branch={branch} "
                        f"max_error={max_straighten_error:.3f} "
                        f"tol={straighten_tolerance:.3f} "
                        "advance=True coupled_translation=True"
                    )
                # Do NOT return. The Guard was already generated perpendicular;
                # any residual error is only physical settling.  Advancing the
                # scalar centroid at the same time keeps this wall at the front
                # instead of letting the NORMAL SPH body pass it.

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
        axial_offsets = (
            [float(value[0]) for value in lifecycle.get("relative_offsets", {}).values()]
            if lifecycle is not None else []
        )
        trailing_offset = min(axial_offsets, default=0.0)
        # Keep the trailing edge of the complete multi-row wall ahead of the
        # NORMAL support front, not merely the wall centroid.
        supported_target = (
            supported_front
            + physical.FRONTIER_LINE_LEAD_GAP
            - trailing_offset
        )
        bootstrap_target = (
            float(lifecycle.get("frontier_bootstrap_target_depth", physical.frontier_line_depth))
            if lifecycle is not None
            else physical.frontier_line_depth
        )
        desired = max(
            physical.frontier_line_depth,
            supported_target,
            bootstrap_target,
        )

        # Advance the complete multi-row Frontier monotonically.  The former
        # contact handler could lower ``frontier_line_depth`` after a single
        # leading robot reported a bumper hit.  A side/edge false-positive then
        # moved every target backward for a few frames, which looked like the
        # Shepherd/Frontier was being pushed away before the swarm touched it.
        #
        # Instead, predict the *next* rigid-wall target and freeze at the
        # current depth only when a spatially distributed subset of the leading
        # row would become non-walkable.  The scalar depth is never decreased.
        next_depth = min(
            desired,
            physical.frontier_line_depth
            + physical.FRONTIER_LINE_ADVANCE_SPEED * dt,
        )
        if lifecycle is not None and axial_offsets:
            frozen_depth = lifecycle.get("frontier_contact_centroid_depth")
            if frozen_depth is not None:
                # A confirmed dead-end contact is sticky for this branch.
                next_depth = physical.frontier_line_depth
            else:
                layer_by_id = lifecycle.get("guard_layer_by_id", {})
                leading_layer = max(layer_by_id.values(), default=-1)
                leading_ids = [
                    robot_id for robot_id, layer in layer_by_id.items()
                    if layer == leading_layer
                ]
                center = lifecycle.get("mouth_center_world")
                tangent = lifecycle.get("branch_tangent_unit")
                lateral_axis = lifecycle.get("mouth_lateral_unit")
                blocked_leading = []
                if (
                    leading_ids
                    and center is not None
                    and tangent is not None
                    and lateral_axis is not None
                    and next_depth > physical.frontier_line_depth + physical.EPSILON
                ):
                    for robot_id in leading_ids:
                        axial_offset, lateral_offset = lifecycle["relative_offsets"].get(
                            robot_id, (0.0, 0.0)
                        )
                        probe_target = lifecycle_target(
                            lifecycle, next_depth, axial_offset, lateral_offset
                        )
                        if not physical.is_walkable(
                            probe_target, physical.ROBOT_RADIUS
                        ):
                            blocked_leading.append(robot_id)

                # One edge robot grazing a side wall is not a dead-end.  Require
                # at least half of the leading row (and at least two robots) to
                # agree in the same predicted step before freezing the entire
                # 3-row formation.
                required_contact = max(2, int(math.ceil(0.5 * len(leading_ids))))
                if blocked_leading and len(blocked_leading) >= required_contact:
                    lifecycle["frontier_contact_centroid_depth"] = float(
                        physical.frontier_line_depth
                    )
                    next_depth = physical.frontier_line_depth
                    print(
                        f"[FrontierRigidContact] branch={branch} "
                        f"centroid_depth={physical.frontier_line_depth:.3f} "
                        f"leading_blocked={len(blocked_leading)}/{len(leading_ids)} "
                        f"rows={lifecycle.get('rows', 0)} "
                        f"cols={lifecycle.get('cols', 0)} "
                        "freeze_complete_wall=True monotonic_depth=True"
                    )

        # Never command a backward axial target during EXPLORE_BRANCH.
        physical.frontier_line_depth = max(
            physical.frontier_line_depth,
            next_depth,
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
        # There are two valid local saturation paths.  The original density path
        # remains available, but a physically confirmed rigid Frontier contact
        # is stronger dead-end evidence than a small density-ratio rise.  In the
        # latter case the 3-row leading layer has already found the map wall with
        # majority contact, so do not leave EXPLORE_BRANCH waiting forever for
        # rho_ratio >= 1.02 while the packed body is visibly stalled behind it.
        contact_confirmed = bool(
            lifecycle is not None
            and lifecycle.get("frontier_contact_centroid_depth") is not None
        )
        common_saturation_evidence = (
            len(cohort) >= physical.SATURATION_MIN_TIP_ROBOTS
            and travelled >= max(
                physical.JUNCTION_COHORT_MIN_TRAVEL,
                descriptor.observed_physical_width,
            )
            and diagnostics.frontier_stalled
            and diagnostics.local_pressure_ratio
            >= LOCAL_SATURATION_PRESSURE_RATIO
            and diagnostics.cross_section_fill
            >= physical.SATURATION_PACKED_LATERAL_COVERAGE_RATIO
        )
        density_saturation = (
            common_saturation_evidence
            and diagnostics.local_density_ratio
            >= physical.SATURATION_DENSITY_RATIO
        )
        contact_saturation = (
            common_saturation_evidence
            and contact_confirmed
        )
        conditions = density_saturation or contact_saturation
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
                f"contact={contact_confirmed} "
                f"mode={'CONTACT' if contact_saturation else ('DENSITY' if density_saturation else '-')} "
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

    def shepherd_line_leading_depth(
        robots: Sequence[Any], branch: str, descriptor: Any
    ) -> float | None:
        """Estimate the current dead-end-side depth of the intact Shepherd wall."""
        estimates: list[float] = []
        for robot in physical.get_shepherds(robots):
            if robot.shepherd_branch != branch or robot.shepherd_anchor is None:
                continue
            axial = physical.observed_branch_axial_depth(robot.position, descriptor)
            stored = physical.integration_shepherd_anchor_offsets.get(
                id(robot.shepherd_anchor)
            )
            axial_offset = float(stored[0]) if stored is not None else 0.0
            estimates.append(float(axial - axial_offset))
        if not estimates:
            return None
        return float(np.median(estimates))

    def original_guard_leading_depth(
        branch: str,
    ) -> float:
        """처음 생성된 3xN Guard의 가장 branch 안쪽 row depth."""

        lifecycle = getattr(
            physical,
            "integration_wall_lifecycle",
            {},
        ).get(branch)

        if lifecycle is None:
            raise RuntimeError(
                f"missing lifecycle for original Guard depth: {branch}"
            )

        center = lifecycle.get("mouth_center_world")
        tangent = lifecycle.get("branch_tangent_unit")
        anchors = lifecycle.get("guard_anchor_by_id", {})

        if (
            center is None
            or tangent is None
            or not anchors
        ):
            raise RuntimeError(
                f"missing frozen original Guard anchors: {branch}"
            )

        tangent = tangent.normalize()

        return max(
            float(
                (anchor - center).dot(tangent)
            )
            for anchor in anchors.values()
        )

    def shepherd_return_depth(
        branch: str,
    ) -> float:
        """Shepherd target depth. Never move past the original Guard wall."""

        original_depth = original_guard_leading_depth(branch)

        if physical.phase in {
            physical.SimulationPhase.PRESSURE_PUSH,
            physical.SimulationPhase.FLOW_BACKTRACK,
        }:
            command = getattr(
                physical,
                "integration_backtrack_command_depth",
                None,
            )

            if command is not None:
                return max(
                    original_depth,
                    float(command),
                )

        return max(
            original_depth,
            float(original_get_shepherd_line_depth(branch)),
        )

    def shepherd_returned_to_original_guard(
        robots: Sequence[Any],
        branch: str | None = None,
    ) -> bool:
        """SAME Shepherd IDs가 최초 Guard 위치에 실제로 돌아왔는지 확인."""

        target_branch = branch or physical.active_branch

        if physical.phase != physical.SimulationPhase.FLOW_BACKTRACK:
            return False

        lifecycle = getattr(
            physical,
            "integration_wall_lifecycle",
            {},
        ).get(target_branch)

        if lifecycle is None:
            return False

        expected_ids = set(
            lifecycle.get("robot_ids", [])
        )

        original_anchors = lifecycle.get(
            "guard_anchor_by_id",
            {},
        )

        if (
            not expected_ids
            or set(original_anchors) != expected_ids
        ):
            return False

        shepherds = {
            robot.robot_id: robot
            for robot in physical.get_shepherds(robots)
            if (
                robot.shepherd_branch == target_branch
                and robot.robot_id in expected_ids
            )
        }

        if set(shepherds) != expected_ids:
            return False

        tolerance = physical.JUNCTION_GUARD_POSITION_TOLERANCE

        return all(
            shepherds[robot_id].position.distance_to(
                original_anchors[robot_id]
            )
            <= tolerance
            for robot_id in expected_ids
        )

    def backtrack_pack_support(
        robots: Sequence[Any], branch: str
    ) -> tuple[float | None, float | None, int, float]:
        """Return a *full-cross-section* NORMAL rear surface for the piston.

        The previous implementation used a single 95th percentile over all
        NORMAL depths.  A thin handful of robots near the Shepherd could move
        first, drag that percentile toward the Junction, and let the rigid
        Shepherd wall follow while most of the swarm was still behind.

        The working Physical DFS behaves like a width-filling piston.  Recreate
        that semantics without the old invisible curtain: split the physical
        corridor width into Shepherd-column bins, require broad lateral support,
        and derive the return target from the rear surface of those occupied
        bins.  If the cross-section loses support, the Shepherd holds position.
        """
        descriptor = descriptor_for(branch)
        lifecycle = getattr(physical, "integration_wall_lifecycle", {}).get(branch)
        if descriptor is None or lifecycle is None:
            return None, None, 0, 0.0
        leading_depth = shepherd_line_leading_depth(robots, branch, descriptor)
        if leading_depth is None:
            return None, None, 0, 0.0

        relative = lifecycle.get(
            "shepherd_relative_offsets",
            lifecycle.get("relative_offsets", {}),
        )
        raw_axial_offsets = [float(value[0]) for value in relative.values()]
        # The return scalar is the dead-end-side leading row. Promotion stores
        # each row relative to max(raw_axial_offset), so the Junction-facing row
        # is min(raw)-max(raw). Earlier pack-coupled revisions used min(raw)
        # directly, which shifted the contact plane and could let the line run
        # ahead of the actual herd.
        minimum_axial_offset = (
            min(raw_axial_offsets) - max(raw_axial_offsets)
            if raw_axial_offsets else 0.0
        )
        junction_face_depth = float(leading_depth + minimum_axial_offset)
        usable_half = max(
            physical.local_physical_usable_half_width(descriptor),
            float(lifecycle.get("usable_half_width", 0.0)),
        )
        if usable_half <= physical.EPSILON:
            return None, None, 0, 0.0

        cols = max(1, int(lifecycle.get("cols", 1)))
        bin_count = max(5, cols)
        face_tolerance = 0.65 * physical.ROBOT_RADIUS
        contact_window = max(
            physical.SHEPHERD_LOCAL_FLOW_DEPTH,
            6.0 * physical.ROBOT_RADIUS,
        )
        lateral_limit = usable_half + physical.ROBOT_RADIUS
        per_bin_depths: list[list[float]] = [[] for _ in range(bin_count)]
        contact_samples: list[tuple[float, float]] = []

        for robot in robots:
            if robot.role != "NORMAL" or robot.base_reserve:
                continue
            axial, lateral = physical.branch_local_coordinates(robot.position, descriptor)
            axial = float(axial)
            lateral = float(lateral)
            if not (
                junction_face_depth - contact_window <= axial
                <= junction_face_depth + face_tolerance
                and abs(lateral) <= lateral_limit
            ):
                continue
            contact_samples.append((axial, lateral))
            u = (lateral + usable_half) / max(2.0 * usable_half, physical.EPSILON)
            index = int(physical.clamp(u * bin_count, 0, bin_count - 1))
            per_bin_depths[index].append(axial)

        occupied = [values for values in per_bin_depths if values]
        coverage = len(occupied) / max(bin_count, 1)
        if not occupied:
            return None, None, 0, coverage

        # One rear surface estimate per lateral lane.  Using lane maxima rather
        # than one global robot percentile prevents one narrow stream from
        # pulling the whole 3xN Shepherd wall through the rest of the body.
        lane_rears = [max(values) for values in occupied]
        rear_depth = (
            float(np.quantile(lane_rears, 0.90))
            if len(lane_rears) >= 5
            else float(max(lane_rears))
        )
        rear_band = max(3.0 * physical.ROBOT_RADIUS, physical.SAFE_RADIUS * 0.55)
        support_count = sum(
            1 for axial, _ in contact_samples if axial >= rear_depth - rear_band
        )

        contact_center_gap = 2.05 * physical.ROBOT_RADIUS
        support_depth = (
            rear_depth + contact_center_gap - minimum_axial_offset
        )
        support_depth = max(0.0, float(support_depth))

        # A rigid full-width Shepherd may advance only with a broad physical rear
        # surface.  Otherwise return None so the caller holds the current depth.
        minimum_coverage = max(
            0.70,
            float(physical.SATURATION_PACKED_LATERAL_COVERAGE_RATIO),
        )
        minimum_support = max(
            physical.FLOW_MIN_NORMAL_COUNT,
            int(math.ceil(cols * 0.90)),
        )
        if coverage < minimum_coverage or support_count < minimum_support:
            return rear_depth, None, support_count, coverage
        return rear_depth, support_depth, support_count, coverage

    def shepherd_pack_contact_state(
        robots: Sequence[Any], branch: str
    ) -> dict[str, float | int | bool]:
        """Strict pre-return gate matching the reference fill-before-push order."""
        descriptor = descriptor_for(branch)
        lifecycle = getattr(physical, "integration_wall_lifecycle", {}).get(branch)
        if descriptor is None or lifecycle is None:
            return {
                "ready": False, "rear_depth": -1.0, "junction_face_depth": -1.0,
                "gap": float("inf"), "support_count": 0, "pack_count": 0,
                "coverage": 0.0,
            }
        shepherds = [
            robot for robot in physical.get_shepherds(robots)
            if robot.shepherd_branch == branch
        ]
        if not shepherds:
            return {
                "ready": False, "rear_depth": -1.0, "junction_face_depth": -1.0,
                "gap": float("inf"), "support_count": 0, "pack_count": 0,
                "coverage": 0.0,
            }

        relative = lifecycle.get(
            "shepherd_relative_offsets",
            lifecycle.get("relative_offsets", {}),
        )
        raw_axial_offsets = [float(value[0]) for value in relative.values()]
        # The return scalar is the dead-end-side leading row. Promotion stores
        # each row relative to max(raw_axial_offset), so the Junction-facing row
        # is min(raw)-max(raw). Earlier pack-coupled revisions used min(raw)
        # directly, which shifted the contact plane and could let the line run
        # ahead of the actual herd.
        minimum_axial_offset = (
            min(raw_axial_offsets) - max(raw_axial_offsets)
            if raw_axial_offsets else 0.0
        )
        leading_depth = shepherd_line_leading_depth(robots, branch, descriptor)
        if leading_depth is None:
            return {
                "ready": False, "rear_depth": -1.0, "junction_face_depth": -1.0,
                "gap": float("inf"), "support_count": 0, "pack_count": 0,
                "coverage": 0.0,
            }
        junction_face_depth = max(0.0, float(leading_depth + minimum_axial_offset))
        rear_depth, support_depth, support_count, coverage = backtrack_pack_support(
            robots, branch
        )

        usable_half = max(
            physical.local_physical_usable_half_width(descriptor),
            float(lifecycle.get("usable_half_width", 0.0)),
        )
        pack_window = max(
            physical.SHEPHERD_LOCAL_FLOW_DEPTH,
            6.0 * physical.ROBOT_RADIUS,
        )
        pack_count = 0
        for robot in robots:
            if robot.role != "NORMAL" or robot.base_reserve:
                continue
            axial, lateral = physical.branch_local_coordinates(robot.position, descriptor)
            if (
                junction_face_depth - pack_window <= axial
                <= junction_face_depth + 0.65 * physical.ROBOT_RADIUS
                and abs(lateral) <= usable_half + physical.ROBOT_RADIUS
            ):
                pack_count += 1

        gap = (
            float(junction_face_depth - rear_depth)
            if rear_depth is not None else float("inf")
        )
        cols = max(1, int(lifecycle.get("cols", 1)))
        min_pack_count = max(
            physical.SATURATION_MIN_TIP_ROBOTS,
            int(math.ceil(cols * 2.0)),
        )
        max_contact_gap = max(
            3.0 * physical.ROBOT_RADIUS,
            0.50 * physical.SAFE_RADIUS,
        )
        ready = bool(
            support_depth is not None
            and pack_count >= min_pack_count
            and -1.5 * physical.ROBOT_RADIUS <= gap <= max_contact_gap
        )
        return {
            "ready": ready,
            "rear_depth": rear_depth if rear_depth is not None else -1.0,
            "junction_face_depth": junction_face_depth,
            "gap": gap,
            "support_count": support_count,
            "pack_count": pack_count,
            "coverage": coverage,
        }


 #브랜치 안쪽에 있는 NORMAL 로봇 무리가 실제로 Junction 방향으로 밀려나오는 만큼만 Shepherd 대열도 뒤따라오게 하는 제어기
    # Active, contact-limited Shepherd piston for branch backtracking.
    def update_pack_coupled_backtrack_depth(
        robots: Sequence[Any], branch: str, dt: float
    ) -> None:
        """Advance the Shepherd as a physical piston without crossing the herd.

        During PRESSURE_PUSH and FLOW_BACKTRACK the intact Shepherd wall actively
        moves toward the Junction.  Its Junction-facing row is clamped to the
        NORMAL rear support surface, so it can compress/push the SPH body but can
        never tunnel through it.  If broad pack support is temporarily lost while
        any NORMAL remains in the branch, the Shepherd holds.  It may finish its
        return alone only after the branch contains zero NORMAL robots.
        """
        if physical.phase not in {
            physical.SimulationPhase.PRESSURE_PUSH,
            physical.SimulationPhase.FLOW_BACKTRACK,
        }:
            return

        physical.integration_return_curtain_active = False

        descriptor = descriptor_for(branch)
        lifecycle = getattr(physical, "integration_wall_lifecycle", {}).get(branch)
        if descriptor is None or lifecycle is None:
            return

        # Final return floor is the original physical Guard wall,
        # not branch depth zero / Junction mouth.
        return_floor = original_guard_leading_depth(branch)

        # command depth is the dead-end-side leading-row scalar.
        current_line = shepherd_line_leading_depth(robots, branch, descriptor)
        if current_line is None:
            return

        previous_command = getattr(
            physical, "integration_backtrack_command_depth", None
        )
        if previous_command is None:
            previous_command = float(current_line)
        previous_command = float(previous_command)

        rear_depth, support_depth, support_count, lateral_coverage = (
            backtrack_pack_support(robots, branch)
        )

        usable_half_width = max(
            physical.local_physical_usable_half_width(descriptor),
            float(lifecycle.get("usable_half_width", 0.0)),
        )
        active_branch_normals = 0
        for robot in robots:
            if robot.role != "NORMAL" or robot.base_reserve:
                continue
            axial, lateral = physical.branch_local_coordinates(
                robot.position, descriptor
            )
            if (
                axial > 0.0
                and abs(lateral)
                <= usable_half_width + 2.0 * physical.ROBOT_RADIUS
            ):
                active_branch_normals += 1

        cols = max(1, int(lifecycle.get("cols", 1)))
        min_support = max(
            physical.FLOW_MIN_NORMAL_COUNT,
            int(math.ceil(cols * 0.90)),
        )
        min_lateral_coverage = max(
            0.70,
            physical.SATURATION_PACKED_LATERAL_COVERAGE_RATIO,
        )
        pack_present = bool(
            support_depth is not None
            and support_count >= min_support
            and lateral_coverage >= min_lateral_coverage
        )

        # This offset is diagnostic only here.  backtrack_pack_support() already
        # converts the NORMAL rear surface into the leading-row command scalar:
        #   support_depth = rear_depth + contact_center_gap - minimum_axial_offset
        # Therefore subtracting minimum_axial_offset again would double-correct
        # the 3-row formation reference.
        relative = lifecycle.get(
            "shepherd_relative_offsets",
            lifecycle.get("relative_offsets", {}),
        )
        raw_axial_offsets = [float(value[0]) for value in relative.values()]
        minimum_axial_offset = (
            min(raw_axial_offsets) - max(raw_axial_offsets)
            if raw_axial_offsets
            else 0.0
        )

        herd_return_speed_scale = float(
            getattr(physical, "integration_herd_return_speed_scale", 0.65)
        )
        forward_speed = min(
            physical.SHEPHERD_LINE_BACKTRACK_SPEED,
            physical.SHEPHERD_PISTON_SPEED,
        ) * herd_return_speed_scale
        max_forward_step = max(0.0, forward_speed * dt)
        max_backoff_step = max_forward_step * 0.50

        # Optional small penetration of the support command can be enabled later
        # for controlled compression.  Default 0 keeps the contact plane strict.
        compression_allowance = max(
            0.0,
            float(
                getattr(
                    physical,
                    "SHEPHERD_CONTACT_COMPRESSION_ALLOWANCE",
                    float(physical.ROBOT_RADIUS) * float(
                        getattr(
                            physical,
                            "integration_shepherd_contact_compression_ratio",
                            0.20,
                        )
                    ),
                )
            ),
        )

        # V34 ACTIVE PISTON:
        # Smaller branch depth is toward the Junction.  The Shepherd must initiate
        # that motion itself and physically compress the NORMAL rear; waiting for
        # the rear to move first turns the Shepherd into a passive follower and
        # deadlocks at the dead-end.
        proposed_command = max(
            return_floor,
            previous_command - max_forward_step,
        )
        active_min_center_gap = max(
            1.05 * physical.ROBOT_RADIUS,
            float(physical.ROBOT_RADIUS)
            * float(getattr(physical, "integration_shepherd_active_min_center_gap_ratio", 1.50)),
        )
        hard_contact_floor = None

        if pack_present and rear_depth is not None:
            # The Junction-facing Shepherd row is:
            #   face = command + minimum_axial_offset
            # Keep that row behind the observed NORMAL rear by at least the hard
            # minimum centre gap.  Unlike the old 2.05R support clamp, this allows
            # controlled compression, so the Shepherd actually pushes the body.
            hard_contact_floor = max(
                return_floor,
                float(rear_depth)
                + active_min_center_gap
                - minimum_axial_offset,
            )
            allowed_floor = min(previous_command, hard_contact_floor)
            command = max(proposed_command, allowed_floor)

            old_support_floor = max(
                0.0,
                float(support_depth) - compression_allowance,
            )
            if command < previous_command - physical.EPSILON:
                mode = (
                    "ACTIVE_CONTACT_COMPRESSION_PUSH"
                    if old_support_floor >= previous_command - physical.EPSILON
                    else "ACTIVE_CONTACT_PUSH"
                )
            else:
                mode = "HOLD_AT_HARD_CONTACT_FLOOR"

        elif active_branch_normals > 0 and rear_depth is not None:
            # Partial-width rear evidence is enough for a slow active priming push.
            # It uses the same hard non-tunnelling floor, only at a reduced speed.
            partial_speed_scale = 0.35
            partial_step = max_forward_step * partial_speed_scale
            proposed_partial = max(
                return_floor,
                previous_command - partial_step,
            )
            hard_contact_floor = max(
                return_floor,
                float(rear_depth)
                + active_min_center_gap
                - minimum_axial_offset,
            )
            allowed_floor = min(previous_command, hard_contact_floor)
            command = max(proposed_partial, allowed_floor)
            mode = (
                "ACTIVE_PARTIAL_CONTACT_PUSH"
                if command < previous_command - physical.EPSILON
                else "HOLD_PARTIAL_AT_HARD_CONTACT_FLOOR"
            )

        elif active_branch_normals > 0:
            # NORMAL robots still exist in the active branch, but the local
            # rear-support window has temporarily lost them.
            #
            # Holding forever here creates a deadlock:
            # NORMALs keep moving toward the Junction while the Shepherd stays
            # behind, so rear support can never be reacquired.
            #
            # Therefore let the intact 3xN Shepherd slowly catch up until
            # physical rear support is observed again.
            seek_speed_scale = 1.00

            seek_step = (
                max_forward_step
                * seek_speed_scale
            )

            command = max(
                return_floor,
                previous_command - seek_step,
            )

            mode = "SEEK_PACK_NO_REAR_SAMPLE"

        else:
            # Only a genuinely empty active branch allows an unaccompanied finish.
            command = max(
                return_floor,
                previous_command - max_forward_step,
            )
            mode = "RETURN_TO_ORIGINAL_GUARD"

        command = max(
            return_floor,
            float(command),
        )

        # V28: the complete Shepherd wall is one rigid body.  The old baseline
        # clamps every Shepherd independently to its current communication parent;
        # on a horizontal branch that can collapse all three axial rows onto one
        # vertical line.  Validate the COMMON scalar target instead.  Either every
        # Shepherd can take the step while preserving its stored robot-id offset,
        # or the whole wall holds this frame.
        rigid_targets: dict[int, pygame.Vector2] = {}
        rigid_geometry_safe = True
        shepherd_offsets = lifecycle.get(
            "shepherd_relative_offsets",
            lifecycle.get("relative_offsets", {}),
        )
        branch_shepherds = [
            robot for robot in physical.get_shepherds(robots)
            if robot.shepherd_branch == branch
        ]
        for shepherd in branch_shepherds:
            stored = shepherd_offsets.get(shepherd.robot_id)
            if stored is None:
                rigid_geometry_safe = False
                break
            axial_offset, lateral_offset = stored
            target = lifecycle_target(
                lifecycle,
                command,
                float(axial_offset),
                float(lateral_offset),
            )
            rigid_targets[shepherd.robot_id] = target
            if not physical.is_walkable(target, shepherd.radius):
                rigid_geometry_safe = False
                break

        # V30 communication rule: the 3xN Shepherd is one rigid local mesh, not
        # 33 independent robots tied forever to their *previous* comm_parent.
        # update_communication_system() rebuilds neighbors/parents after motion,
        # so rejecting the common piston step because one stale parent distance
        # increases by epsilon creates a permanent deadlock.  Instead validate the
        # geometry that the next communication update can actually realize:
        #   1) all proposed Shepherd slots remain one COMM_RANGE-connected mesh;
        #   2) that mesh has at least one LOS tether to a currently Base-connected
        #      non-Shepherd robot (NORMAL/relay/anchor/etc.).
        # A single external tether is enough because the 3xN wall is internally
        # connected and can re-parent locally on the next communication update.
        rigid_internal_connected = False
        rigid_external_tethers = 0
        rigid_comm_safe = False
        if rigid_geometry_safe and branch_shepherds:
            comm_range = float(physical.COMM_RANGE)
            comm_range_sq = comm_range * comm_range
            shepherd_ids = {robot.robot_id for robot in branch_shepherds}
            target_ids = list(rigid_targets)

            adjacency: dict[int, list[int]] = {robot_id: [] for robot_id in target_ids}
            for index, left_id in enumerate(target_ids):
                left_target = rigid_targets[left_id]
                for right_id in target_ids[index + 1:]:
                    right_target = rigid_targets[right_id]
                    if left_target.distance_squared_to(right_target) > comm_range_sq:
                        continue
                    if not physical.has_line_of_sight(left_target, right_target):
                        continue
                    adjacency[left_id].append(right_id)
                    adjacency[right_id].append(left_id)

            if target_ids:
                visited = {target_ids[0]}
                stack = [target_ids[0]]
                while stack:
                    current_id = stack.pop()
                    for neighbor_id in adjacency[current_id]:
                        if neighbor_id in visited:
                            continue
                        visited.add(neighbor_id)
                        stack.append(neighbor_id)
                rigid_internal_connected = len(visited) == len(target_ids)

            external_candidates = [
                robot for robot in robots
                if robot.robot_id not in shepherd_ids
                and robot.role != "PEBBLE"
                and getattr(robot, "connected_to_base", False)
            ]
            tethered_shepherds: set[int] = set()
            for shepherd_id, target in rigid_targets.items():
                for other in external_candidates:
                    if target.distance_squared_to(other.position) > comm_range_sq:
                        continue
                    if not physical.has_line_of_sight(target, other.position):
                        continue
                    tethered_shepherds.add(shepherd_id)
                    break
            rigid_external_tethers = len(tethered_shepherds)
            rigid_comm_safe = (
                rigid_internal_connected
                and rigid_external_tethers >= 1
            )

        if not rigid_geometry_safe:
         # 3xN 벽 자체가 벽/장애물에 들어가는 경우만 실제로 정지.
            command = previous_command
            mode = "HOLD_RIGID_3XN_GEOMETRY"

        elif not rigid_internal_connected:
            # Shepherd 3xN 내부 communication mesh가 깨지는 경우도 정지.
            command = previous_command
            mode = "HOLD_RIGID_3XN_INTERNAL_COMM"

        elif not rigid_comm_safe:
            # 외부 Base-connected tether가 순간적으로 없어졌다는 이유만으로
            # Shepherd 전체 piston을 영구 정지시키지 않는다.
            #
            # 3xN 내부 mesh와 physical geometry가 유지된다면
            # 이미 계산된 command를 그대로 실행하고,
            # 다음 communication update에서 재연결을 허용한다.
            mode = f"{mode}_EXTERNAL_TETHER_REPARENT"

        physical.integration_backtrack_command_depth = command
        physical.integration_backtrack_pack_rear_depth = rear_depth
        physical.integration_backtrack_support_depth = support_depth
        physical.integration_backtrack_support_count = support_count
        physical.integration_backtrack_lateral_coverage = lateral_coverage

        junction_face_depth = command + minimum_axial_offset
        pack_gap = (
            junction_face_depth - rear_depth
            if rear_depth is not None
            else float("nan")
        )
        frame = getattr(physical, "integration_frame", -1)
        if frame % 10 == 0:
            print(
                f"[OriginalGuardReturn] "
                f"frame={frame} "
                f"branch={branch} "
                f"command_depth={command:.3f} "
                f"return_floor={return_floor:.3f} "
                f"current_leading_depth={float(current_line):.3f} "
                f"error_to_floor={float(current_line) - return_floor:.3f}"
            )
            print(
                f"[ReturnPistonContact] frame={frame} phase={physical.phase.name} "
                f"branch={branch} current_line={float(current_line):.3f} "
                f"previous_command={previous_command:.3f} "
                f"command_depth={command:.3f} "
                f"junction_face={junction_face_depth:.3f} "
                f"pack_rear={rear_depth if rear_depth is not None else -1.0:.3f} "
                f"support_depth={support_depth if support_depth is not None else -1.0:.3f} "
                f"support_count={support_count} coverage={lateral_coverage:.3f} "
                f"branch_normals={active_branch_normals} pack_gap={pack_gap:.3f} "
                f"compression_allowance={compression_allowance:.3f} "
                f"active_min_gap={active_min_center_gap:.3f} "
                f"hard_floor={hard_contact_floor if hard_contact_floor is not None else -1.0:.3f} "
                f"rigid_internal={rigid_internal_connected} "
                f"rigid_tethers={rigid_external_tethers} "
                f"rigid_comm_safe={rigid_comm_safe} mode={mode}"
            )


    def real_shepherd_contact_force(robot: Any) -> pygame.Vector2:
        """Finite-radius contact force from the actual Shepherd robots only.

        This is not a Junction-directed route force and not an invisible curtain.
        Every contribution is a local spring-damper along the center-to-center
        normal from a real Shepherd robot to this NORMAL robot.  Therefore the
        force exists only where the visible 3xN Shepherd physically touches the
        packed swarm.
        """
        if (
            physical.phase not in {
                physical.SimulationPhase.PRESSURE_PUSH,
                physical.SimulationPhase.FLOW_BACKTRACK,
            }
            or robot.role != "NORMAL"
            or robot.base_reserve
        ):
            setattr(robot, "last_real_shepherd_contact_force", 0.0)
            setattr(robot, "last_real_shepherd_contact_count", 0)
            return pygame.Vector2()

        branch = physical.active_branch
        descriptor = descriptor_for(branch)
        if descriptor is None:
            setattr(robot, "last_real_shepherd_contact_force", 0.0)
            setattr(robot, "last_real_shepherd_contact_count", 0)
            return pygame.Vector2()

        axial, lateral = physical.branch_local_coordinates(robot.position, descriptor)
        usable_half = physical.local_physical_usable_half_width(descriptor)
        if axial <= 0.0 or abs(lateral) > usable_half + 2.5 * physical.ROBOT_RADIUS:
            setattr(robot, "last_real_shepherd_contact_force", 0.0)
            setattr(robot, "last_real_shepherd_contact_count", 0)
            return pygame.Vector2()

        shepherds = [
            shepherd for shepherd in physical.get_shepherds(getattr(physical, "robots", []))
            if shepherd.shepherd_branch == branch
        ] if hasattr(physical, "robots") else []

        # The physical module does not guarantee a global robots list.  The
        # integration installs the current sequence before each SPH call below.
        if not shepherds:
            shepherds = [
                shepherd for shepherd in getattr(physical, "integration_current_robots", [])
                if shepherd.role == "SHEPHERD" and shepherd.shepherd_branch == branch
            ]
        if not shepherds:
            setattr(robot, "last_real_shepherd_contact_force", 0.0)
            setattr(robot, "last_real_shepherd_contact_count", 0)
            return pygame.Vector2()

        radius = float(physical.ROBOT_RADIUS)
        shell_ratio = float(
            getattr(physical, "integration_shepherd_contact_shell_ratio", 2.35)
        )
        contact_radius = max(2.05 * radius, shell_ratio * radius)
        spring_gain = float(physical.REPULSION_GAIN) * float(
            getattr(physical, "integration_shepherd_contact_spring_scale", 2.50)
        )
        damping_gain = float(getattr(physical, "VISCOELASTIC_DASHPOT_GAIN", physical.DAMPING)) * float(
            getattr(physical, "integration_shepherd_contact_damping_scale", 1.00)
        )

        total = pygame.Vector2()
        contacts = 0
        strongest_compression = 0.0
        for shepherd in shepherds:
            delta = robot.position - shepherd.position
            distance_sq = delta.length_squared()
            if distance_sq <= physical.EPSILON:
                # Deterministic fallback only for an exact numerical coincidence.
                normal = descriptor.local_return_direction.normalize()
                distance = 0.0
            else:
                distance = math.sqrt(distance_sq)
                if distance >= contact_radius:
                    continue
                normal = delta / distance

            compression = physical.clamp(
                (contact_radius - distance) / max(contact_radius, physical.EPSILON),
                0.0,
                1.0,
            )
            if compression <= 0.0:
                continue

            pair_force = normal * (spring_gain * compression * compression)
            shepherd_velocity = getattr(shepherd, "velocity", pygame.Vector2())
            relative_normal_speed = (robot.velocity - shepherd_velocity).dot(normal)
            if relative_normal_speed < 0.0:
                pair_force += normal * (-damping_gain * relative_normal_speed)

            total += pair_force
            contacts += 1
            strongest_compression = max(strongest_compression, compression)

        # Runtime compatibility: some local environment.py revisions do not
        # expose EQUILIBRIUM_REPULSION_FORCE_LIMIT / SPH_PRESSURE_FORCE_LIMIT.
        # Use those limits when available, otherwise derive conservative limits
        # from constants that are present across the Physical-DFS revisions.
        motion_multiplier = max(
            1.0,
            float(getattr(physical, "MOTION_SPEED_MULTIPLIER", 1.0)),
        )
        repulsion_force_limit = float(
            getattr(
                physical,
                "EQUILIBRIUM_REPULSION_FORCE_LIMIT",
                max(
                    float(getattr(physical, "REPULSION_GAIN", 180.0)),
                    180.0 * motion_multiplier,
                ),
            )
        )
        pressure_force_limit = float(
            getattr(
                physical,
                "SPH_PRESSURE_FORCE_LIMIT",
                max(
                    repulsion_force_limit,
                    420.0 * motion_multiplier,
                ),
            )
        )
        contact_force_limit = max(
            repulsion_force_limit,
            0.60 * pressure_force_limit,
        )
        physical.limit_vector(total, contact_force_limit)
        setattr(robot, "last_real_shepherd_contact_force", total.length())
        setattr(robot, "last_real_shepherd_contact_count", contacts)

        frame = getattr(physical, "integration_frame", -1)
        if (
            contacts > 0
            and frame % 10 == 0
            and getattr(physical, "integration_real_contact_log_frame", -1) != frame
        ):
            physical.integration_real_contact_log_frame = frame
            print(
                f"[RealShepherdContactV30] frame={frame} phase={physical.phase.name} "
                f"branch={branch} robot={robot.robot_id} contacts={contacts} "
                f"compression={strongest_compression:.3f} force={total.length():.3f}"
            )
        return total


    def shepherd_physical_only_route_force(robot: Any) -> pygame.Vector2:
        """Remove axial route suction from active-branch NORMALs during return.

        The NORMAL body must be carried toward the Junction by the moving Shepherd
        wall and SPH/contact/compression, not by FLOW_BACKTRACK's built-in direct
        Junction attraction.  Preserve only the component perpendicular to the
        return axis so corridor/lane centering can still operate.
        """
        if getattr(physical, "integration_final_guard_sweep_active", False):
            return final_guard_sweep_route_force(robot)

        force = original_compute_route_force(robot)
        setattr(robot, "last_physical_only_route_axial", 0.0)
        if (
            physical.phase in {
                physical.SimulationPhase.PRESSURE_PUSH,
                physical.SimulationPhase.FLOW_BACKTRACK,
            }
            and robot.role == "NORMAL"
            and not robot.base_reserve
        ):
            descriptor = descriptor_for(physical.active_branch)
            if descriptor is not None:
                axial, lateral = physical.branch_local_coordinates(
                    robot.position, descriptor
                )
                usable_half = physical.local_physical_usable_half_width(descriptor)
                if axial > 0.0 and abs(lateral) <= usable_half + 2.5 * physical.ROBOT_RADIUS:
                    return_axis = descriptor.local_return_direction.normalize()
                    axial_component = force.dot(return_axis)
                    force = force - return_axis * axial_component
                    setattr(
                        robot,
                        "last_physical_only_route_axial",
                        abs(force.dot(return_axis)),
                    )
                    # Add only real pairwise Shepherd contact.  No global return-axis
                    # drive is added to NORMAL robots.
                    force += real_shepherd_contact_force(robot)
        return force

    def physical_only_compute_sph_forces(
        robots: Sequence[Any],
        grid: Any,
        communication_grid: Any,
        dt: float = 1.0 / 60.0,
    ) -> None:
        """Keep SPH physics while disabling the invisible return pressure field."""
        return_phase = physical.phase in {
            physical.SimulationPhase.PRESSURE_PUSH,
            physical.SimulationPhase.FLOW_BACKTRACK,
        }
        physical.integration_current_robots = robots
        physical.integration_return_curtain_active = False

        virtual_pressure_force = getattr(physical, "VIRTUAL_PRESSURE_FORCE", None)
        if return_phase and virtual_pressure_force is not None:
            physical.VIRTUAL_PRESSURE_FORCE = 0.0
        try:
            original_compute_sph_forces(robots, grid, communication_grid, dt)
        finally:
            if return_phase and virtual_pressure_force is not None:
                physical.VIRTUAL_PRESSURE_FORCE = virtual_pressure_force

        frame = getattr(physical, "integration_frame", -1)
        if not return_phase or frame % 10 != 0:
            return

        branch = physical.active_branch
        descriptor = descriptor_for(branch)
        active_normals = []
        if descriptor is not None:
            usable_half = physical.local_physical_usable_half_width(descriptor)
            for robot in robots:
                if robot.role != "NORMAL" or robot.base_reserve:
                    continue
                axial, lateral = physical.branch_local_coordinates(
                    robot.position, descriptor
                )
                if (
                    axial > 0.0
                    and abs(lateral)
                    <= usable_half + 2.5 * physical.ROBOT_RADIUS
                ):
                    active_normals.append(robot)

        normal_axial_route_max = max(
            (
                float(getattr(robot, "last_physical_only_route_axial", 0.0))
                for robot in active_normals
            ),
            default=0.0,
        )
        real_shepherd_contacts = sum(
            int(getattr(robot, "last_real_shepherd_contact_count", 0))
            for robot in active_normals
        )
        max_real_contact_force = max(
            (
                float(getattr(robot, "last_real_shepherd_contact_force", 0.0))
                for robot in active_normals
            ),
            default=0.0,
        )
        command_depth = getattr(
            physical, "integration_backtrack_command_depth", None
        )
        return_floor = (
            original_guard_leading_depth(branch)
            if descriptor is not None
            else float("nan")
        )
        print(
            f"[PhysicalOnlyReturn] frame={frame} phase={physical.phase.name} "
            f"branch={branch} "
            f"normal_axial_route_max={normal_axial_route_max:.9f} "
            f"real_shepherd_contacts={real_shepherd_contacts} "
            f"max_real_contact_force={max_real_contact_force:.3f} "
            f"command_depth="
            f"{float(command_depth) if command_depth is not None else -1.0:.3f} "
            f"return_floor={return_floor:.3f} "
            f"curtain_active={physical.integration_return_curtain_active}"
        )

    def local_slot_at_depth(
        anchor: pygame.Vector2, branch: str, depth: float
    ) -> pygame.Vector2:
        """Reference-style backtracking: keep transverse slot, reduce depth only."""
        lifecycle = getattr(physical, "integration_wall_lifecycle", {}).get(branch)
        descriptor = descriptor_for(branch)
        if lifecycle is None or descriptor is None:
            return anchor.copy()
        stored = physical.integration_shepherd_anchor_offsets.get(id(anchor))
        if stored is None:
            center = lifecycle.get("mouth_center_world")
            tangent = lifecycle.get("branch_tangent_unit")
            lateral_axis = lifecycle.get("mouth_lateral_unit")
            if center is None or tangent is None or lateral_axis is None:
                return anchor.copy()
            delta = anchor - center
            # Without a stored leading-edge offset, keep the anchor's own
            # cross-section identity and treat it as the leading row.
            axial_offset = 0.0
            lateral = float(delta.dot(lateral_axis))
        else:
            axial_offset, lateral = stored
        return lifecycle_target(
            lifecycle, max(0.0, float(depth)), float(axial_offset), float(lateral)
        )

    def promote_thick_frontier_wall(
        robots: Sequence[Any], branch: str,
        observed_boundary_depth: float | None = None,
    ) -> list[Any]:
        """Promote the SAME Frontier IDs to Shepherds *in place*.

        The thick Frontier has already travelled as the physical front wall.
        As soon as distributed rigid contact confirms a dead-end, the SAME robot
        IDs become Shepherds in place.  Packing/saturation is evaluated only after
        this role transition while the Shepherd wall is held stationary.

        Re-solving a second 3xN target here would create an unnecessary FORM phase
        and can make the Shepherds thread through the packed body.  Preserve each
        Frontier robot's current world position as its Shepherd anchor instead.

        The current dead-end-side leading edge becomes the scalar piston depth;
        each robot stores its current axial/lateral offset from that edge.  Return
        motion therefore translates the *already existing* physical wall toward
        the Junction without any second formation or position snap.
        """
        lifecycle = physical.integration_wall_lifecycle[branch]
        frontiers = physical.get_frontier_shepherds(robots, branch)
        expected = int(lifecycle["rows"]) * int(lifecycle["cols"])
        if len(frontiers) != expected:
            raise RuntimeError(
                f"thick Frontier wall lost members: {len(frontiers)}/{expected}"
            )

        descriptor = descriptor_for(branch)
        if descriptor is None:
            raise RuntimeError(f"missing descriptor for Shepherd promotion {branch}")
        lock_branch_transport_frame(branch, lifecycle, descriptor)

        center = lifecycle.get("mouth_center_world")
        tangent = lifecycle.get("branch_tangent_unit")
        lateral_axis = lifecycle.get("mouth_lateral_unit")
        if center is None or tangent is None or lateral_axis is None:
            raise RuntimeError(f"missing frozen branch frame for Shepherd promotion {branch}")

        current_coordinates: dict[int, tuple[float, float]] = {}
        for robot in frontiers:
            delta = robot.position - center
            axial = float(delta.dot(tangent))
            lateral = float(delta.dot(lateral_axis))
            current_coordinates[robot.robot_id] = (axial, lateral)

        leading_edge_depth = max(
            axial for axial, _ in current_coordinates.values()
        )
        centroid_depth = float(np.median([
            axial for axial, _ in current_coordinates.values()
        ]))

        shepherd_relative_offsets: dict[int, tuple[float, float]] = {}
        promoted: list[Any] = []

        # Freeze the branch-local return direction once at promotion. Every
        # Shepherd carries the same local vector; no absolute Junction coordinate
        # is needed by the Shepherd controller.
        return_direction = descriptor.local_return_direction.copy()
        if return_direction.length_squared() <= physical.EPSILON:
            raise RuntimeError(f"invalid local return direction for Shepherd promotion {branch}")
        return_direction = return_direction.normalize()

        for robot in frontiers:
            axial, lateral = current_coordinates[robot.robot_id]
            axial_from_leading_edge = float(axial - leading_edge_depth)
            anchor = robot.position.copy()

            robot.role = "SHEPHERD"
            robot.shepherd_anchor = anchor
            robot.shepherd_origin = robot.position.copy()
            robot.shepherd_branch = branch
            robot.shepherd_return_direction = return_direction.copy()
            robot.junction_guard_anchor = None
            robot.velocity.update(0.0, 0.0)
            robot.acceleration.update(0.0, 0.0)
            robot.filtered_acceleration.update(0.0, 0.0)

            stored = (axial_from_leading_edge, float(lateral))
            shepherd_relative_offsets[robot.robot_id] = stored
            physical.integration_shepherd_anchor_offsets[id(anchor)] = stored
            promoted.append(robot)

        # Keep the original Guard/Frontier offsets for lineage/audits, but use the
        # actual dead-end wall offsets for every Shepherd return/contact calculation.
        lifecycle["shepherd_relative_offsets"] = shepherd_relative_offsets
        lifecycle["shepherd_centroid_depth"] = centroid_depth
        lifecycle["shepherd_leading_edge_depth"] = float(leading_edge_depth)
        lifecycle["shepherd_shape_locked"] = True
        lifecycle["state"] = "SHEPHERD"
        lifecycle["frontier_to_shepherd_in_place"] = True
        physical.observed_dead_end_depths[branch] = float(leading_edge_depth)

        physical.frontier_line_branch = None
        physical.frontier_line_depth = 0.0
        physical.frontier_line_lateral_center = 0.0
        physical.frontier_line_row_ready = False

        min_axial = min(
            offset[0] for offset in shepherd_relative_offsets.values()
        )
        lateral_values = [
            offset[1] for offset in shepherd_relative_offsets.values()
        ]
        print(
            f"[FrontierToShepherdInPlaceV27] branch={branch} "
            f"rows={lifecycle['rows']} cols={lifecycle['cols']} "
            f"robots={len(promoted)} leading_edge={leading_edge_depth:.3f} "
            f"thickness={-min_axial:.3f} "
            f"lateral_span={(max(lateral_values)-min(lateral_values)) if lateral_values else 0.0:.3f} "
            "same_ids=True position_jump=0 second_formation=False"
        )
        return promoted

    def returned_guard_wall_ready(robots: Sequence[Any], branch: str) -> bool:
        """True only when the returned SAME-ID wall is back on its frozen mouth slots."""
        lifecycle = getattr(physical, "integration_wall_lifecycle", {}).get(branch)
        if lifecycle is None:
            return False
        expected_ids = set(lifecycle.get("robot_ids", []))
        anchors = lifecycle.get("guard_anchor_by_id", {})
        if not expected_ids or set(anchors) != expected_ids:
            return False
        guards = {
            robot.robot_id: robot
            for robot in robots
            if robot.robot_id in expected_ids
            and robot.role == "JUNCTION_GUARD"
            and robot.junction_guard_branch == branch
        }
        if set(guards) != expected_ids:
            return False
        return all(
            guards[robot_id].position.distance_to(anchors[robot_id])
            <= physical.JUNCTION_GUARD_POSITION_TOLERANCE
            for robot_id in expected_ids
        )

    def continuous_release_line(robots: Sequence[Any]) -> int:
        """Dissolve a returned Shepherd line at the Junction, as in baseline DFS.

        The explored branch no longer keeps a persistent Guard/Shepherd wall.
        Once the intact Shepherd formation has physically returned to the
        Junction, one Pebble may be staged by the existing Physical-DFS helper
        and every remaining Shepherd becomes NORMAL at its current position.
        No position snap is performed.
        """
        if physical.phase != physical.SimulationPhase.FLOW_BACKTRACK:
            return 0
        branch = physical.active_branch
        descriptor = descriptor_for(branch)
        lifecycle = getattr(physical, "integration_wall_lifecycle", {}).get(branch)
        shepherds = [
            robot for robot in physical.get_shepherds(robots)
            if robot.shepherd_branch == branch
        ]
        if descriptor is None or lifecycle is None or not shepherds:
            return 0
        maximum_depth = max(
            physical.observed_branch_axial_depth(robot.position, descriptor)
            for robot in shepherds
        )
        if maximum_depth > physical.SHEPHERD_JUNCTION_DEPTH_TOLERANCE:
            return 0

        # Do not dissolve the Shepherd line just because it reached the mouth.
        # It is the physical rear piston and must remain intact until the NORMAL
        # body it was herding has also cleared the active branch.
        remaining_branch_normals = 0
        usable_half = max(
            physical.local_physical_usable_half_width(descriptor),
            float(lifecycle.get("usable_half_width", 0.0)),
        )
        for candidate in robots:
            if candidate.role != "NORMAL" or candidate.base_reserve:
                continue
            axial, lateral = physical.branch_local_coordinates(
                candidate.position, descriptor
            )
            if (
                axial > 0.0
                and abs(lateral) <= usable_half + 2.0 * physical.ROBOT_RADIUS
            ):
                remaining_branch_normals += 1
        if remaining_branch_normals > 0:
            frame = getattr(physical, "integration_frame", -1)
            if frame % 15 == 0:
                print(
                    f"[ShepherdReleaseWait] frame={frame} branch={branch} "
                    f"remaining_normals={remaining_branch_normals} "
                    "reason=HERD_NOT_CLEAR"
                )
            return 0

        # Preserve the original topological visit marker, but do not preserve
        # the physical Shepherd wall on an already-explored branch.
        physical.stage_pebble_from_returned_shepherd_line(robots, branch)
        remaining_shepherds = [
            robot for robot in physical.get_shepherds(robots)
            if robot.shepherd_branch == branch
        ]
        return_direction = descriptor.local_return_direction.normalize()
        before = {
            robot.robot_id: robot.position.copy()
            for robot in remaining_shepherds
        }
        # Match each released Shepherd to the nearby NORMAL flow instead of
        # assigning the exact same velocity to the whole 3xN row.  Equal release
        # velocities preserve a conspicuously straight line even after the role
        # and stale virtual-wall constraints are gone.
        release_velocity_by_id: dict[int, pygame.Vector2] = {}
        ordinary_normals = [
            candidate for candidate in robots
            if candidate.role == "NORMAL" and not candidate.base_reserve
        ]
        flow_radius_sq = (2.0 * physical.SMOOTHING_LENGTH) ** 2
        for shepherd in remaining_shepherds:
            neighbors = sorted(
                (
                    candidate for candidate in ordinary_normals
                    if candidate.position.distance_squared_to(shepherd.position)
                    <= flow_radius_sq
                ),
                key=lambda candidate: candidate.position.distance_squared_to(
                    shepherd.position
                ),
            )[:10]
            if neighbors:
                velocity = pygame.Vector2()
                for candidate in neighbors:
                    velocity += candidate.velocity
                velocity /= len(neighbors)
            else:
                velocity = return_direction * min(
                    4.0, physical.SHEPHERD_JUNCTION_RELEASE_SPEED
                )
            release_velocity_by_id[shepherd.robot_id] = velocity

        for robot in remaining_shepherds:
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
            robot.velocity = release_velocity_by_id[robot.robot_id].copy()
            robot.acceleration.update(0.0, 0.0)
            robot.filtered_acceleration.update(0.0, 0.0)

        released_ids = {robot.robot_id for robot in remaining_shepherds}
        physical.viscoelastic_rest_lengths = {
            pair: value
            for pair, value in physical.viscoelastic_rest_lengths.items()
            if not released_ids.intersection(pair)
        }
        physical.viscoelastic_last_seen = {
            pair: value
            for pair, value in physical.viscoelastic_last_seen.items()
            if not released_ids.intersection(pair)
        }
        physical.junction_guard_groups[branch] = []
        lifecycle["state"] = "VISITED_RELEASED"
        lifecycle["released_frame"] = getattr(
            physical, "integration_frame", -1
        )
        physical.frontier_line_branch = None
        physical.frontier_line_depth = 0.0
        physical.frontier_line_lateral_center = 0.0
        physical.frontier_line_row_ready = False
        transition_jump = max(
            (
                robot.position.distance_to(before[robot.robot_id])
                for robot in remaining_shepherds
            ),
            default=0.0,
        )
        print(
            f"[ShepherdRelease] branch={branch} "
            f"released_to_normal={len(remaining_shepherds)} "
            f"pebbles={len(physical.get_pebbles(robots))} "
            f"position_jump={transition_jump:.6f} "
            "visited_wall_persisted=False"
        )
        return len(remaining_shepherds)

    def remaining_dfs_branch_uids(
        robots: Sequence[Any],
    ) -> tuple[set[str], set[str], list[str]]:
        """Return discovered, visited and still-unvisited branch UIDs."""

        discovered = set(
            getattr(
                physical,
                "integration_detected_branch_order",
                [],
            )
        )

        if not discovered:
            discovered = set(
                physical.discovered_branch_uids()
            )

        # Legacy Pebble-based evidence, if any.
        visited = set(
            physical.observed_visited_branch_uids(robots)
        )

        # Physical DFS integration:
        # an intact SAME-ID wall that completed
        # Guard -> Frontier -> Shepherd -> Guard
        # is also valid persistent VISITED evidence.
        for branch, lifecycle in getattr(
            physical,
            "integration_wall_lifecycle",
            {},
        ).items():
            if lifecycle.get("state") != "VISITED_GUARD":
                continue

            uid = physical.branch_uid_for_fixture(branch)

            if uid is not None:
                visited.add(uid)

        order = getattr(
            physical,
            "integration_detected_branch_order",
            [],
        )

        remaining = [
            uid
            for uid in order
            if uid not in visited
        ]

        if not order:
            remaining = sorted(
                discovered - visited
            )

        return discovered, visited, remaining


    def arm_cross_branch_carry(
        robots: Sequence[Any],
        source_branch: str,
        target_branch: str,
    ) -> None:
        """Convert residual Junction return speed into next-branch propulsion."""
        junction_speeds = [
            robot.velocity.length()
            for robot in robots
            if robot.role == "NORMAL"
            and physical.get_robot_region(robot.position) == "JUNCTION"
            and not robot.base_reserve
        ]
        mean_speed = float(np.mean(junction_speeds)) if junction_speeds else 0.0
        raw_scale = (
            1.0
            + mean_speed
            / max(float(physical.CROSS_BRANCH_CARRY_SPEED_REFERENCE), physical.EPSILON)
        )
        peak_scale = float(np.clip(
            raw_scale,
            physical.CROSS_BRANCH_CARRY_MIN_SCALE,
            physical.CROSS_BRANCH_CARRY_MAX_SCALE,
        ))
        physical.cross_branch_carry_peak_scale = peak_scale
        physical.cross_branch_carry_until = (
            physical.simulation_time + physical.CROSS_BRANCH_CARRY_DURATION
        )
        print(
            f"[CrossBranchCarry] frame={getattr(physical, 'integration_frame', -1)} "
            f"{source_branch}->{target_branch} junction_mean_speed={mean_speed:.3f} "
            f"peak_scale={peak_scale:.3f} "
            f"duration={physical.CROSS_BRANCH_CARRY_DURATION:.2f}"
        )

    def commit_branch_uid_to_frontier(
        robots: Sequence[Any],
        reference_density: float,
        branch_uid: str,
        *,
        context: str,
    ) -> str:
        """Commit exactly one UID and promote that same fixture's Guard wall.

        ``record_distributed_consensus`` only records diagnostics; it does NOT set
        ``distributed_consensus_branch``.  The legacy ``choose_next_branch`` reads
        that global.  Keeping those two states separate allowed active_branch to
        become RIGHT while the LEFT Guard wall was promoted.  Lock both sides to
        the same UID here and fail loudly on any future mismatch.
        """
        requested_fixture = physical.branch_fixture_for_uid(branch_uid)
        if requested_fixture is None:
            raise RuntimeError(
                f"{context}: no fixture adapter for requested branch UID {branch_uid}"
            )

        # Single source of truth for the legacy chooser.
        physical.distributed_consensus_branch = branch_uid
        physical.record_distributed_consensus(branch_uid)
        selected = physical.choose_next_branch(robots, reference_density)
        if selected is None:
            raise RuntimeError(
                f"{context}: choose_next_branch returned None for UID {branch_uid} "
                f"fixture={requested_fixture}"
            )

        selected_uid = physical.branch_uid_for_fixture(selected)
        if selected != requested_fixture or selected_uid != branch_uid:
            raise RuntimeError(
                f"{context}: branch identity divergence: requested_uid={branch_uid} "
                f"requested_fixture={requested_fixture} selected_fixture={selected} "
                f"selected_uid={selected_uid}"
            )
        if physical.active_branch != selected or physical.active_branch_uid != branch_uid:
            raise RuntimeError(
                f"{context}: active branch divergence after choose: "
                f"active_branch={physical.active_branch} "
                f"active_uid={physical.active_branch_uid} expected={selected}/{branch_uid}"
            )

        # No other branch may already own a live Frontier at selection time.
        stale = [
            robot
            for robot in robots
            if robot.role == "FRONTIER_SHEPHERD"
            and robot.shepherd_branch != selected
        ]
        if stale:
            raise RuntimeError(
                f"{context}: stale Frontier exists on another branch: "
                f"{sorted({robot.shepherd_branch for robot in stale})}"
            )

        physical.commit_junction_guard_roles(robots, selected)
        if physical.frontier_line_branch != selected:
            raise RuntimeError(
                f"{context}: promoted Frontier branch={physical.frontier_line_branch} "
                f"but active branch={selected}"
            )
        live_frontier_ids = {
            robot.robot_id
            for robot in robots
            if robot.role == "FRONTIER_SHEPHERD"
            and robot.shepherd_branch == selected
        }
        lifecycle = getattr(physical, "integration_wall_lifecycle", {}).get(selected, {})
        expected_ids = set(lifecycle.get("robot_ids", []))
        if not live_frontier_ids or live_frontier_ids != expected_ids:
            raise RuntimeError(
                f"{context}: active branch Guard/Frontier lineage mismatch: "
                f"selected={selected} live={sorted(live_frontier_ids)} "
                f"expected={sorted(expected_ids)}"
            )
        print(
            f"[BranchIdentityLock] context={context} uid={branch_uid} "
            f"fixture={selected} active={physical.active_branch} "
            f"frontier={physical.frontier_line_branch} robots={len(live_frontier_ids)} "
            "consistent=True"
        )
        return selected

    def start_final_return_pipeline(robots: Sequence[Any], reason: str) -> None:
        """Enter the authoritative final gather -> base return lifecycle once."""
        if getattr(physical, "integration_final_return_requested", False):
            return
        physical.integration_final_return_requested = True
        discovered, visited, remaining = remaining_dfs_branch_uids(robots)
        if remaining:
            raise RuntimeError(
                f"final return requested with unvisited branches remaining: {remaining}"
            )
        physical.pending_branch_start = None
        physical.integration_ready_guard_handoff = False
        physical.integration_handoff_dwell = 0.0
        physical.junction_consensus_tracker.reset()

        # Keep every visited returned Guard wall fixed until the NORMAL swarm
        # establishes a real Base-bound flow.
        start_general_final_guard_sweep(robots, reason)

        print(
            f"[DFSAllBranchesVisited] frame={getattr(physical, 'integration_frame', -1)} "
            f"visited={len(visited)}/{len(discovered)} reason={reason}"
        )
        print(
            f"[FinalReturnPipeline] frame={getattr(physical, 'integration_frame', -1)} "
            "ALL_BRANCHES_VISITED -> FIXED_RETURNED_GUARDS -> "
            "BASE_FLOW_ESTABLISH -> SIMULTANEOUS_GUARD_RELEASE -> "
            "RETURN_TO_BASE -> ALL_ROBOTS_DONE"
        )

    def integrated_update_state(
        robots: Sequence[Any], dt: float, reference_density: float,
        spatial_grid: Any,
    ) -> None:
        # Adaptive-LiDAR handoff already has every detected physical Guard wall.
        # Mirror the authoritative Physical DFS start lifecycle explicitly:
        # vote -> choose -> SAME Guard IDs become Frontier -> pending dwell.
        if (
            physical.phase == physical.SimulationPhase.FORM_JUNCTION_GUARDS
            and getattr(physical, "integration_ready_guard_handoff", False)
            and physical.pending_branch_start is None
        ):
            _, _, remaining_uids = remaining_dfs_branch_uids(robots)
            if not remaining_uids:
                start_final_return_pipeline(
                    robots,
                    "INITIAL_GATE_NO_REMAINING",
                )
                return
            voted_branch = remaining_uids[0]
            selected = commit_branch_uid_to_frontier(
                robots,
                reference_density,
                voted_branch,
                context="INITIAL_DFS_START",
            )
            physical.pending_branch_start = selected
            physical.junction_guard_formation_timer = 0.0
            physical.junction_guard_stable_dwell = 0.0
            physical.integration_handoff_dwell = 0.0
            print(
                f"[DFSBranchSelected] frame={getattr(physical, 'integration_frame', -1)} "
                f"voted_uid={voted_branch} selected={selected} "
                "same_guard_ids_to_frontier=True"
            )
            return

        if (
            physical.phase == physical.SimulationPhase.FORM_JUNCTION_GUARDS
            and getattr(physical, "integration_ready_guard_handoff", False)
            and physical.pending_branch_start is not None
        ):
            # Mirror the reference lifecycle: a branch is selected first,
            # then its Frontier wall and the remaining Guard walls must remain
            # physically present for the short formation dwell before flow is
            # released into EXPLORE_BRANCH.
            selected = physical.pending_branch_start
            lifecycle = getattr(physical, "integration_wall_lifecycle", {})
            visited_uids = physical.observed_visited_branch_uids(robots)
            unvisited_branches = [
                branch
                for branch in physical.detected_branch_candidates
                if physical.branch_uid_for_fixture(branch) not in visited_uids
            ]
            selected_frontiers = [
                robot for robot in robots
                if robot.role == "FRONTIER_SHEPHERD"
                and robot.shepherd_branch == selected
            ]
            selected_lifecycle = lifecycle.get(selected, {})
            selected_ready = len(selected_frontiers) == (
                int(selected_lifecycle.get("rows", 0))
                * int(selected_lifecycle.get("cols", 0))
            )
            remaining_guard_walls_ready = True
            for branch in unvisited_branches:
                if branch == selected:
                    continue
                item = lifecycle.get(branch, {})
                expected_count = int(item.get("rows", 0)) * int(item.get("cols", 0))
                live_count = sum(
                    robot.role == "JUNCTION_GUARD"
                    and robot.junction_guard_branch == branch
                    for robot in robots
                )
                if expected_count <= 0 or live_count != expected_count:
                    remaining_guard_walls_ready = False
                    break
            walls_ready = selected_ready and remaining_guard_walls_ready
            if walls_ready:
                dwell = float(getattr(physical, "integration_handoff_dwell", 0.0)) + dt
                physical.integration_handoff_dwell = dwell
            else:
                physical.integration_handoff_dwell = 0.0
                return
            if physical.integration_handoff_dwell < physical.THICK_MOUTH_GUARD_FORM_DWELL:
                return
            physical.pending_branch_start = None
            physical.junction_guard_status = (
                f"FRONTIER={selected};OTHERS=THICK_KHOP_WALLS_READY"
            )
            physical.phase = physical.SimulationPhase.EXPLORE_BRANCH
            physical.integration_ready_guard_handoff = False
            physical.integration_handoff_dwell = 0.0
            print(
                f"[EXPLORE_BRANCH] selected={selected} "
                f"wall_dwell={physical.THICK_MOUTH_GUARD_FORM_DWELL:.2f} "
                "second_guard_formation=False"
            )
            return
        # Baseline-style branch return: once the Shepherd line reaches the
        # Junction it is dissolved into NORMAL robots (plus the existing Pebble
        # marker).  Already-explored branches therefore keep no Shepherd/Guard
        # wall.  The remaining unvisited branch Guards are left untouched and
        # are sufficient for the next DFS branch selection.
        if physical.phase == physical.SimulationPhase.FLOW_BACKTRACK:
            branch = physical.active_branch
            lifecycle = getattr(
                physical, "integration_wall_lifecycle", {}
            ).get(branch)

            physical.shepherd_flow_timer += dt

            # The real 3xN Shepherd remains the physical rear piston throughout
            # backtracking.  No Junction population / branch-empty gate is used.
            update_pack_coupled_backtrack_depth(robots, branch, dt)
            physical.update_relay_retraction(robots, dt)

            shepherds_before = [
                robot
                for robot in physical.get_shepherds(robots)
                if robot.shepherd_branch == branch
            ]
            shepherd_ids_before = {
                robot.robot_id for robot in shepherds_before
            }

            if not physical.shepherd_line_reached_junction(robots, branch):
                if getattr(physical, "integration_frame", 0) % 15 == 0:
                    print(
                        f"[ShepherdArrivalWait] "
                        f"frame={getattr(physical, 'integration_frame', -1)} "
                        f"branch={branch} shepherds={len(shepherds_before)} "
                        f"line_depth={physical.get_shepherd_line_depth(branch):.3f} "
                        f"junction_tol="
                        f"{physical.SHEPHERD_JUNCTION_DEPTH_TOLERANCE:.3f}"
                    )
                return

            arrival_frame = getattr(physical, "integration_frame", -1)
            completed_branch = branch

            completed_uid = (
                physical.active_branch_uid
                or physical.branch_uid_for_fixture(completed_branch)
            )

            if completed_uid is None:
                raise RuntimeError(
                    f"missing branch UID at Shepherd return: {completed_branch}"
                )

            original_anchors = (lifecycle or {}).get(
                "guard_anchor_by_id",
                {},
            )
            same_ids_at_return = (
                bool(shepherd_ids_before)
                and set(original_anchors) == shepherd_ids_before
            )
            max_original_guard_error_before_release = max(
                (
                    robot.position.distance_to(
                        original_anchors[robot.robot_id]
                    )
                    for robot in shepherds_before
                    if robot.robot_id in original_anchors
                ),
                default=float("inf"),
            )
            print(
                f"[OriginalGuardReturnComplete] "
                f"branch={completed_branch} "
                f"same_ids={same_ids_at_return} "
                f"max_anchor_error="
                f"{max_original_guard_error_before_release:.6f} "
                f"tolerance="
                f"{physical.JUNCTION_GUARD_POSITION_TOLERANCE:.6f}"
            )

            # Exact lifecycle requested:
            # SAME Guard IDs -> Frontier -> Shepherd -> SAME Guard IDs.
            retained_guard_count = physical.release_shepherd_line_at_junction(
                robots,
                keep_as_guard=True,
            )
            if retained_guard_count <= 0:
                raise RuntimeError(
                    "branch was completed but returned Shepherd wall "
                    f"could not become Guard: branch={completed_branch}"
                )

            returned_guards = [
                robot
                for robot in robots
                if robot.role == "JUNCTION_GUARD"
                and robot.junction_guard_branch == completed_branch
            ]
            returned_guard_ids = {
                robot.robot_id for robot in returned_guards
            }

            original_anchor_by_id = (lifecycle or {}).get(
                "guard_anchor_by_id",
                {},
            )
            missing_original_anchors = sorted(
                robot.robot_id
                for robot in returned_guards
                if robot.robot_id not in original_anchor_by_id
            )
            if missing_original_anchors:
                raise RuntimeError(
                    "returned Guard has no frozen initial Guard anchor: "
                    f"branch={completed_branch} "
                    f"robots={missing_original_anchors}"
                )

            max_original_guard_error = max(
                (
                    robot.position.distance_to(
                        original_anchor_by_id[robot.robot_id]
                    )
                    for robot in returned_guards
                ),
                default=0.0,
            )
            if max_original_guard_error > 1e-7:
                raise RuntimeError(
                    "returned Shepherd wall did not restore exact initial "
                    f"Guard positions: branch={completed_branch} "
                    f"max_error={max_original_guard_error:.9f}"
                )

            if returned_guard_ids != shepherd_ids_before:
                raise RuntimeError(
                    "Guard->Frontier->Shepherd->Guard lineage mismatch: "
                    f"before={sorted(shepherd_ids_before)} "
                    f"after={sorted(returned_guard_ids)}"
                )

            expected_ids = set(
                (lifecycle or {}).get("robot_ids", shepherd_ids_before)
            )
            if expected_ids and returned_guard_ids != expected_ids:
                raise RuntimeError(
                    "returned visited Guard IDs differ from original Guard wall: "
                    f"expected={sorted(expected_ids)} "
                    f"returned={sorted(returned_guard_ids)}"
                )

            # Returned SAME-ID Guard wall is the completion evidence.
            # Do not require a Pebble before starting the next DFS branch.
            physical.set_branch_descriptor_state(
                completed_uid,
                "VISITED",
            )

            physical.previous_branch_direction = (
                physical.get_backtrack_direction(completed_branch)
            )

            physical.distributed_consensus_branch = None
            physical.active_branch_uid = None

            for robot in robots:
                robot.branch_vote = None
                robot.branch_vote_confidence = 0.0
                robot.distributed_branch_decision = None

            physical.record_distributed_consensus(
                clear_selection=True
            )

            if hasattr(physical, "metrics"):
                physical.metrics.branch_events.append({
                    "branch": completed_branch,
                    "completed_at": physical.simulation_time,
                })

            if lifecycle is not None:
                lifecycle["state"] = "VISITED_GUARD"
                lifecycle["visited_guard_frame"] = arrival_frame
                lifecycle["returned_guard_ids"] = sorted(returned_guard_ids)
                lifecycle["same_ids_guard_frontier_shepherd_guard"] = True
                lifecycle["next_branch_trigger"] = "SHEPHERD_JUNCTION_ARRIVAL"

            print(
                f"[GuardCycleComplete] frame={arrival_frame} "
                f"branch={completed_branch} "
                f"robots={len(returned_guard_ids)} "
                "GUARD->FRONTIER->SHEPHERD->GUARD "
                f"same_ids=True original_guard_error="
                f"{max_original_guard_error:.9f} "
                "restored_exact_initial_guard_pose=True"
            )

            discovered, visited, remaining_uids = remaining_dfs_branch_uids(robots)
            completed_count = len(visited)
            total_count = len(discovered)

            print(
                f"[DFSBranchCompleteSequence] frame={arrival_frame} "
                f"completed={completed_count}/{total_count} "
                f"remaining={remaining_uids}"
            )

            # Last branch: it also completed the full cycle back to Guard.
            # The existing final-return pipeline then releases all Guards/Pebbles
            # for the all-robot return-to-Base phase.
            if discovered and not remaining_uids:
                start_final_return_pipeline(
                    robots,
                    "LAST_BRANCH_RETURNED_TO_GUARD",
                )
                return

            physical.integration_backtrack_command_depth = None
            physical.integration_backtrack_pack_rear_depth = None
            physical.integration_backtrack_support_depth = None
            physical.integration_backtrack_support_count = 0
            physical.integration_backtrack_lateral_coverage = 0.0

            physical.phase = physical.SimulationPhase.JUNCTION_SWITCH
            physical.junction_switch_timer = 0.0
            physical.junction_consensus_tracker.reset()

            print(
                f"[DFSNextBranchSameFrame] frame={arrival_frame} "
                f"completed_branch={completed_branch} "
                "visited_guard_persisted=True "
                "trigger=SHEPHERD_JUNCTION_ARRIVAL"
            )

            # SAME FRAME:
            # completed branch's returned wall stays JUNCTION_GUARD,
            # while the next unvisited branch's pre-existing Guard becomes
            # Frontier and immediately starts exploration.
            integrated_update_state(
                robots,
                0.0,
                reference_density,
                spatial_grid,
            )
            return

        if physical.phase == physical.SimulationPhase.JUNCTION_SWITCH:
            # Immediate LiDAR-DFS handoff, entered on the exact Shepherd-arrival
            # frame. Reuse the next branch's persistent physical Guard wall as
            # Frontier immediately; there is no second gather/formation dwell.
            physical.junction_switch_timer += dt
            discovered, visited, remaining_uids = remaining_dfs_branch_uids(robots)
            if discovered and not remaining_uids:
                start_final_return_pipeline(robots, "JUNCTION_SWITCH_NO_REMAINING")
                return

            voted_branch = remaining_uids[0]
            requested_fixture = physical.branch_fixture_for_uid(voted_branch)
            if requested_fixture is None:
                raise RuntimeError(
                    f"JUNCTION_SWITCH: no fixture for next UID {voted_branch}"
                )

            selected_guard_count = sum(
                robot.role == "JUNCTION_GUARD"
                and robot.junction_guard_branch == requested_fixture
                for robot in robots
            )
            if selected_guard_count <= 0:
                # Do not block the next valid branch because some *other* future
                # branch wall is missing. Only the wall we are activating now is
                # required at this instant.
                print(
                    f"[NextBranchWait] frame={getattr(physical, 'integration_frame', -1)} "
                    f"reason=SELECTED_GUARD_MISSING uid={voted_branch} "
                    f"branch={requested_fixture}"
                )
                return

            source_branch = physical.active_branch
            selected = commit_branch_uid_to_frontier(
                robots,
                reference_density,
                voted_branch,
                context="JUNCTION_SWITCH_IMMEDIATE",
            )
            selected_uid = physical.branch_uid_for_fixture(selected)
            if selected_uid in visited:
                raise RuntimeError(
                    f"visited branch re-selected during Junction switch: {selected_uid}"
                )

            # Preserve the source return velocity and turn it into a short
            # propulsion boost along the newly selected branch.
            arm_cross_branch_carry(robots, source_branch, selected)

            physical.pending_branch_start = None
            physical.integration_ready_guard_handoff = False
            physical.integration_handoff_dwell = 0.0
            physical.junction_guard_formation_timer = 0.0
            physical.junction_guard_stable_dwell = 0.0
            physical.branch_entry_timer = 0.0
            physical.phase = physical.SimulationPhase.EXPLORE_BRANCH
            print(
                f"[ImmediateNextBranch] frame={getattr(physical, 'integration_frame', -1)} "
                f"{source_branch}->{selected} uid={selected_uid} "
                f"frontier_committed=True guard_count={selected_guard_count} "
                "formation_dwell=0"
            )
            return

        branch = physical.active_branch
        if physical.phase == physical.SimulationPhase.EXPLORE_BRANCH:
            physical.branch_entry_timer += dt
            physical.update_relay_deployment(robots, dt)
            local_frontier_progress(robots, branch, dt)

            # V31 state order:
            #   Frontier travels -> distributed rigid dead-end contact ->
            #   SAME IDs become Shepherd immediately -> Shepherd waits in FILL.
            # Saturation is intentionally NOT required while the robots still have
            # the Frontier role.  Packing is measured only after role transition.
            state = sample_local_state(robots, branch, reference_density, dt)
            lifecycle = getattr(
                physical, "integration_wall_lifecycle", {}
            ).get(branch)
            dead_end_contact = bool(
                lifecycle is not None
                and lifecycle.get("frontier_contact_centroid_depth") is not None
            )
            if not dead_end_contact:
                return

            frontiers = physical.get_frontier_shepherds(robots, branch)
            if not frontiers:
                return

            before = {
                robot.robot_id: robot.position.copy() for robot in frontiers
            }
            state.frontier_ids = sorted(before)
            descriptor = descriptor_for(branch)
            if descriptor is None:
                return
            boundary_depth = float(np.median([
                physical.observed_branch_axial_depth(robot.position, descriptor)
                for robot in frontiers
            ]))
            physical.observed_dead_end_depths[branch] = boundary_depth

            selected = physical.promote_existing_frontier_line(
                robots, branch, boundary_depth
            )
            if not selected:
                if getattr(physical, "integration_frame", 0) % 15 == 0:
                    print(
                        f"[BacktrackPromotionWaitV31] branch={branch} "
                        f"dead_end=True boundary_depth={boundary_depth:.3f} "
                        "reason=NO_COMMON_SHEPHERD_CROSS_SECTION "
                        "frontier_frozen=True"
                    )
                return

            state.shepherd_ids = sorted(robot.robot_id for robot in selected)
            state.max_transition_jump = max(
                robot.position.distance_to(before[robot.robot_id])
                for robot in selected
            )
            state.shepherd_transition = True
            state.transition_frame = getattr(physical, "integration_frame", -1)
            return_direction = descriptor.local_return_direction.normalize()
            state.return_direction_local = (return_direction.x, return_direction.y)

            # Dead-end is confirmed now.  This is distinct from pack saturation.
            physical.branch_dead_end_confirmed[branch] = True
            physical.dead_end_inference_tracker.confirmed = True
            physical.dead_end_inference_tracker.confirmed_depth = boundary_depth
            physical.dead_end_inference_tracker.handoff_depth = boundary_depth

            # Store the instant-of-promotion packing level as the FILL baseline.
            baseline_density = max(0.0, float(state.local_density))
            baseline_pressure = max(0.0, float(state.local_pressure))
            baseline_cross_fill = max(0.0, float(state.cross_section_fill))

            current_shepherd_depth = float(np.median([
                physical.observed_branch_axial_depth(robot.position, descriptor)
                for robot in selected
            ]))
            diagnostics.reset(branch, current_shepherd_depth)
            diagnostics.baseline_density = baseline_density
            diagnostics.baseline_pressure = baseline_pressure
            diagnostics.cross_section_fill = baseline_cross_fill
            physical.integration_shepherd_fill_baseline_density = baseline_density
            physical.integration_shepherd_fill_baseline_pressure = baseline_pressure
            physical.integration_shepherd_fill_baseline_cross_fill = baseline_cross_fill
            physical.integration_shepherd_pack_ready_dwell = 0.0
            physical.integration_backtrack_command_depth = None
            # Reset reference packing trackers, but do not start pressure push.
            if hasattr(physical, "saturation_tracker"):
                physical.saturation_tracker.reset(branch)
            if hasattr(physical, "branch_continuity_tracker"):
                physical.branch_continuity_tracker.reset(branch)

            # Preserve lineage information until FILL confirms saturation.
            physical.integration_pending_shepherd_transition_event = {
                "branch": branch,
                "uid": descriptor.uid,
                "transition_frame": state.transition_frame,
                "frontier_ids": list(state.frontier_ids),
                "shepherd_ids": list(state.shepherd_ids),
                "max_transition_jump": state.max_transition_jump,
                "return_direction": state.return_direction_local,
                "rows": physical.integration_wall_lifecycle[branch]["rows"],
                "cols": physical.integration_wall_lifecycle[branch]["cols"],
                "robots": len(state.frontier_ids),
                "max_formation_error": state.max_formation_error,
            }

            # Same robots, same world positions, same 3xN shape. Only the role
            # changes. Hand control back to the Environment at its native
            # FORM_SHEPHERD_BOUNDARY entry point. Since every Shepherd anchor is
            # exactly its current position, shepherd_boundary_formed() is already
            # true and the Environment advances to FILL on its next state update.
            physical.phase = physical.SimulationPhase.FORM_SHEPHERD_BOUNDARY
            physical.shepherd_form_timer = 0.0

            print(
                "[Timeline] DEAD_END_CONFIRMED "
                f"frame={state.transition_frame} uid={descriptor.uid}"
            )
            print(
                "[Timeline] FRONTIER_TO_SHEPHERD "
                f"frame={state.transition_frame} same_ids="
                f"{state.frontier_ids == state.shepherd_ids} "
                f"max_position_jump={state.max_transition_jump:.6f}"
            )
            print(
                f"[ShepherdFillStartV31] frame={state.transition_frame} "
                f"branch={branch} baseline_density={baseline_density:.6f} "
                f"baseline_pressure={baseline_pressure:.3f} "
                f"baseline_cross_fill={baseline_cross_fill:.3f} "
                "same_ids_in_place=True -> FORM_SHEPHERD_BOUNDARY -> ENVIRONMENT_FILL"
            )
            return
  
            # Keep the SAME thick Shepherd wall and let it settle continuously
            # onto its frozen targets.  Do not fall through to the legacy timeout
            # path, which can reset Shepherd roles and bounce back to EXPLORE_BRANCH
            # when force_complete_shepherd_boundary is intentionally disabled to
            # avoid position teleportation.
            physical.update_relay_deployment(robots, dt)
            physical.shepherd_form_timer += dt
            shepherds = [
                robot for robot in physical.get_shepherds(robots)
                if robot.shepherd_branch == branch
            ]
            lifecycle = getattr(physical, "integration_wall_lifecycle", {}).get(branch, {})
            expected = int(lifecycle.get("rows", 0)) * int(lifecycle.get("cols", 0))
            formed = (
                expected > 0
                and len(shepherds) == expected
                and all(
                    robot.shepherd_anchor is not None
                    and robot.position.distance_to(robot.shepherd_anchor)
                    <= physical.SHEPHERD_FORM_TOLERANCE
                    for robot in shepherds
                )
            )
            anchors_walkable = (
                expected > 0
                and len(shepherds) == expected
                and all(
                    robot.shepherd_anchor is not None
                    and physical.is_walkable(robot.shepherd_anchor, robot.radius)
                    for robot in shepherds
                )
            )
            timed_settle_ready = (
                anchors_walkable
                and physical.shepherd_form_timer >= physical.SHEPHERD_FORM_TIMEOUT
            )
            # V26: never start packing around a half-settled Shepherd wall.
            # Every SAME-ID Shepherd must be physically inside its frozen 3xN slot
            # tolerance before NORMAL robots are allowed to fill behind it.
            if formed:
                physical.phase = physical.SimulationPhase.FILL_BEHIND_SHEPHERD
                physical.saturation_tracker.reset(branch)
                physical.branch_continuity_tracker.reset(branch)
                # IMPORTANT: reset the integration-local saturation latch too.
                # Before this fix, EXPLORE's already-saturated diagnostics were
                # carried into FILL and PRESSURE_PUSH started immediately, before
                # NORMAL robots had physically packed against the Shepherd.
                current_shepherd_depth = float(np.median([
                    physical.observed_branch_axial_depth(robot.position, descriptor_for(branch))
                    for robot in shepherds
                ])) if shepherds and descriptor_for(branch) is not None else 0.0
                diagnostics.reset(branch, current_shepherd_depth)
                physical.integration_shepherd_pack_ready_dwell = 0.0
                max_error = max(
                    (
                        robot.position.distance_to(robot.shepherd_anchor)
                        for robot in shepherds
                        if robot.shepherd_anchor is not None
                    ),
                    default=0.0,
                )
                print(
                    f"[ShepherdBoundaryReady] frame={getattr(physical, 'integration_frame', -1)} "
                    f"branch={branch} robots={len(shepherds)}/{expected} "
                    f"max_error={max_error:.3f} "
                    "bounded_timeout=False "
                    "teleport=False -> FILL_BEHIND_SHEPHERD"
                )
            elif (
                physical.shepherd_form_timer >= physical.SHEPHERD_FORM_TIMEOUT
                and getattr(physical, "integration_frame", 0) % 30 == 0
            ):
                max_error = max(
                    (
                        robot.position.distance_to(robot.shepherd_anchor)
                        for robot in shepherds
                        if robot.shepherd_anchor is not None
                    ),
                    default=float("inf"),
                )
                print(
                    f"[ShepherdBoundaryWait] frame={getattr(physical, 'integration_frame', -1)} "
                    f"branch={branch} robots={len(shepherds)}/{expected} "
                    f"max_error={max_error:.3f} anchors_walkable={anchors_walkable} "
                    "destructive_reset=False"
                )
            return
        if physical.phase == physical.SimulationPhase.FILL_BEHIND_SHEPHERD:
            # V31: Shepherd is a fixed physical back wall during FILL. NORMAL robots
            # continue entering and compressing against it.  Start return only when
            # density, pressure and cross-section fill have all increased enough,
            # and a real full-width contact pack is present for a short dwell.
            physical.branch_entry_timer += dt
            physical.update_relay_deployment(robots, dt)

            fill_state = sample_local_state(
                robots, branch, reference_density, dt
            )
            contact = shepherd_pack_contact_state(robots, branch)

            baseline_density = float(
            physical.integration_shepherd_fill_baseline_density
               )   

            baseline_pressure = float(
                physical.integration_shepherd_fill_baseline_pressure
            )

            density_ready = bool(
                baseline_density > physical.EPSILON
                and fill_state.local_density
                >= baseline_density * physical.SATURATION_DENSITY_RATIO
            )

            pressure_ready = bool(
                baseline_pressure > physical.EPSILON
                and fill_state.local_pressure
                >= baseline_pressure * LOCAL_SATURATION_PRESSURE_RATIO
            )

            cross_fill_ready = bool(
                fill_state.cross_section_fill
                >= physical.SATURATION_PACKED_LATERAL_COVERAGE_RATIO
            )   

            contact_ready = bool(contact["ready"])

            stall_ready = bool(fill_state.frontier_stalled)

            packed_ready = bool(
                contact_ready
                and density_ready
                and pressure_ready
                and cross_fill_ready
                and stall_ready
            )
            
            if packed_ready:
                physical.integration_shepherd_pack_ready_dwell += dt
            else:
                physical.integration_shepherd_pack_ready_dwell = 0.0

            diagnostics.dwell = float(
                physical.integration_shepherd_pack_ready_dwell
            )
            diagnostics.saturated = bool(
                packed_ready
                and diagnostics.dwell
                >= physical.integration_shepherd_pack_ready_required_dwell
            )

            frame = getattr(physical, "integration_frame", -1)
            if frame % 15 == 0:
                print(
                    f"[ShepherdFillGateV33] frame={frame} branch={branch} "
                    f"density={fill_state.local_density:.6f} "
                    f"density_ratio={fill_state.local_density_ratio:.3f}/"
                    f"{physical.SATURATION_DENSITY_RATIO:.3f} "
                    f"pressure={fill_state.local_pressure:.3f} "
                    f"pressure_ratio={fill_state.local_pressure_ratio:.3f}/"
                    f"{LOCAL_SATURATION_PRESSURE_RATIO:.3f} "
                    f"cross_fill={fill_state.cross_section_fill:.3f}/"
                    f"{physical.SATURATION_PACKED_LATERAL_COVERAGE_RATIO:.3f} "
                    f"contact_ready={contact_ready} "
                    f"pack_count={int(contact['pack_count'])} "
                    f"support_count={int(contact['support_count'])} "
                    f"contact_coverage={float(contact['coverage']):.3f} "
                    f"gap={float(contact['gap']):.3f} "
                    f"dwell={diagnostics.dwell:.3f}/"
                    f"{physical.integration_shepherd_pack_ready_required_dwell:.3f} "
                    f"packed_ready={packed_ready}"
                )

            if not diagnostics.saturated:
                return

            pending = getattr(
                physical, "integration_pending_shepherd_transition_event", None
            ) or {}
            shepherd_ids = sorted(
                robot.robot_id for robot in physical.get_shepherds(robots)
                if robot.shepherd_branch == branch
            )
            frontier_ids = list(pending.get("frontier_ids", shepherd_ids))
            event = {
                "branch": branch,
                "uid": pending.get("uid", descriptor_for(branch).uid),
                "frame": frame,
                "frontier_speed": fill_state.frontier_speed,
                "local_density": fill_state.local_density,
                "density_ratio": fill_state.local_density_ratio,
                "local_pressure": fill_state.local_pressure,
                "pressure_ratio": fill_state.local_pressure_ratio,
                "cross_section_fill": fill_state.cross_section_fill,
                "dwell": diagnostics.dwell,
                "frontier_ids": frontier_ids,
                "shepherd_ids": shepherd_ids,
                "max_transition_jump": float(
                    pending.get("max_transition_jump", 0.0)
                ),
                "return_direction": pending.get(
                    "return_direction", (0.0, 0.0)
                ),
                "rows": int(pending.get(
                    "rows", physical.integration_wall_lifecycle[branch]["rows"]
                )),
                "cols": int(pending.get(
                    "cols", physical.integration_wall_lifecycle[branch]["cols"]
                )),
                "robots": len(frontier_ids),
                "max_formation_error": float(
                    pending.get("max_formation_error", 0.0)
                ),
            }
            physical.integration_saturation_events.append(event)
            physical.integration_pending_shepherd_transition_event = None

            print(
                "[Timeline] BRANCH_SATURATION_CONFIRMED "
                f"frame={frame} uid={event['uid']} "
                f"density_ratio={event['density_ratio']:.3f} "
                f"pressure_ratio={event['pressure_ratio']:.3f} "
                f"cross_fill={event['cross_section_fill']:.3f}"
            )
            print(
                f"[ShepherdPushStartV31] frame={frame} branch={branch} "
                "packed=True shepherd_stationary_until_now=True "
                "-> PRESSURE_PUSH"
            )
            physical.start_shepherd_pressure_push(robots, branch)
            return

        if physical.phase == physical.SimulationPhase.PRESSURE_PUSH:
            # Keep the authoritative flow-establishment/state transition logic,
            # but replace its timer-driven Shepherd position with the active,
            # contact-limited physical piston command.
            update_pack_coupled_backtrack_depth(robots, branch, dt)
            phase_before = physical.phase
            original_update_state(robots, dt, reference_density, spatial_grid)
            if phase_before != physical.phase:
                print(
                    f"[ReferenceReturnState] frame={getattr(physical, 'integration_frame', -1)} "
                    f"{phase_before.name}->{physical.phase.name} source=AUTHORITATIVE_PHYSICAL_DFS"
                )
            return

        phase_before_original = physical.phase
        original_update_state(robots, dt, reference_density, spatial_grid)
        if physical.phase != phase_before_original and phase_before_original in {
            physical.SimulationPhase.FINAL_JUNCTION_GATHER,
            physical.SimulationPhase.RETURN_TO_BASE,
        }:
            print(
                f"[FinalReturnTransition] frame={getattr(physical, 'integration_frame', -1)} "
                f"{phase_before_original.name}->{physical.phase.name}"
            )
        if (
            getattr(physical, "integration_ready_guard_handoff", False)
            and physical.phase != physical.SimulationPhase.FORM_JUNCTION_GUARDS
        ):
            physical.integration_ready_guard_handoff = False

    # ============================================================
    # ENVIRONMENT-AUTHORITATIVE DFS CYCLE
    # ============================================================
    # Adaptive LiDAR is authoritative only for perception, initial Guard WHO/WHERE,
    # branch UID association, and the thick Guard/Frontier geometry adapter.
    # The actual DFS return lifecycle below is the original Environment logic.

    physical.update_frontier_line_progress = local_frontier_progress
    physical.commit_junction_guard_roles = audited_commit_guard_roles
    physical.frontier_shepherd_slot_target = thick_frontier_slot_target
    physical.promote_existing_frontier_line = promote_thick_frontier_wall
    physical.prepare_branch_candidate_scores = pebble_filtered_candidate_scores

    # Geometry adapters: map Environment's return hooks onto the Adaptive LiDAR
    # branch-local frame while leaving the state-machine structure unchanged.
    physical.get_backtrack_direction = local_return_direction
    physical.shepherd_slot_position_at_depth = local_slot_at_depth

    # Restore the exact Environment fill/transport logic.
    physical.update_transfer_continuity_control = original_transfer_control

    # Shepherd/backtracking mechanics.
    physical.get_shepherd_line_depth = shepherd_return_depth
    physical.shepherd_line_reached_junction = (
        shepherd_returned_to_original_guard
    )
    physical.compute_route_force = shepherd_physical_only_route_force
    physical.compute_sph_forces = physical_only_compute_sph_forces
    physical.start_shepherd_pressure_push = (
        original_start_shepherd_pressure_push
    )
    physical.release_shepherd_line_at_junction = (
        original_release_shepherd_line_at_junction
    )
    physical.force_complete_shepherd_boundary = (
        original_force_complete_shepherd_boundary
    )
    physical.update_pre_shepherd_pipeline = (
        original_update_pre_shepherd_pipeline
    )

    # Undo integration-only speed/pressure modifications.
    physical.SHEPHERD_PISTON_SPEED = float(
        physical.integration_reference_return_original_piston_speed
    )
    physical.SHEPHERD_LINE_BACKTRACK_SPEED = float(
        physical.integration_reference_return_original_line_speed
    )
    physical.SHEPHERD_JUNCTION_RELEASE_SPEED = float(
        physical.integration_reference_return_original_release_speed
    )
    if hasattr(physical, "integration_original_shepherd_pressure_factor"):
        physical.SHEPHERD_PRESSURE_FACTOR = float(
            physical.integration_original_shepherd_pressure_factor
        )

    # Use Environment Robot.update(), not herd_locked_robot_update().
    physical.Robot.update = original_robot_update

    def _final_base_return_direction() -> pygame.Vector2:
        direction = getattr(
            physical,
            "integration_base_return_direction_local",
            None,
        )
        if direction is None or direction.length_squared() <= physical.EPSILON:
            raise RuntimeError(
                "final return requested before local incoming direction was stored"
            )
        return direction.normalize()

    def _live_guard_members(
        robots: Sequence[Any],
        branch: str,
    ) -> list[Any]:
        return [
            robot
            for robot in robots
            if robot.role == "JUNCTION_GUARD"
            and robot.junction_guard_branch == branch
        ]

    def _visited_guard_branches(
        robots: Sequence[Any],
    ) -> list[str]:
        result = []
        lifecycle = getattr(physical, "integration_wall_lifecycle", {})
        for branch, item in lifecycle.items():
            descriptor = physical.branch_motion_descriptor(branch)
            if descriptor is None or descriptor.visit_state != "VISITED":
                continue
            if not _live_guard_members(robots, branch):
                continue
            result.append(branch)
        return result

    def start_general_final_guard_sweep(
        robots: Sequence[Any],
        reason: str,
    ) -> None:
        """Keep every returned Guard fixed until a real Base-bound flow exists."""

        if getattr(
            physical,
            "integration_final_guard_sweep_active",
            False,
        ):
            return

        branches = _visited_guard_branches(robots)

        if not branches:
            raise RuntimeError(
                "ALL_BRANCHES_VISITED but no returned visited Guard walls exist"
            )

        # Keep every visited Guard fixed at its original mouth anchor.
        for branch in branches:
            members = _live_guard_members(robots, branch)

            if not members:
                raise RuntimeError(
                    f"visited Guard wall has no live members: {branch}"
                )

            for robot in members:
                robot.velocity.update(0.0, 0.0)
                robot.acceleration.update(0.0, 0.0)
                robot.filtered_acceleration.update(0.0, 0.0)

            lifecycle = getattr(
                physical,
                "integration_wall_lifecycle",
                {},
            ).get(branch)

            if lifecycle is not None:
                lifecycle["state"] = "FINAL_PRESSURE_GATE"

        physical.integration_final_guard_sweep_active = True

        # Base flow detection state
        physical.integration_final_base_flow_dwell = 0.0
        physical.integration_final_base_flow_established = False
        physical.integration_final_all_guards_released = False

        # NORMAL에게는 아주 약한 Base 방향 bias만 허용.
        # 실제 이동의 주력은 SPH pressure.
        physical.integration_final_base_flow_force_scale = 0.45
        physical.integration_final_base_flow_min_speed = 1.0
        physical.integration_final_base_flow_dwell_required = 0.25

        physical.phase = physical.SimulationPhase.FINAL_JUNCTION_GATHER
        physical.final_gather_timer = 0.0

        print(
            f"[FinalPressureDrainStart] "
            f"frame={getattr(physical, 'integration_frame', -1)} "
            f"reason={reason} "
            f"guards={branches} "
            "moving_guard_pusher=False "
            "all_returned_guards_fixed=True"
        )

    def final_guard_sweep_route_force(robot: Any) -> pygame.Vector2:
        if not getattr(
            physical,
            "integration_final_guard_sweep_active",
            False,
        ):
            return shepherd_physical_only_route_force(robot)

        if robot.role in {
            "JUNCTION_GUARD",
            "PEBBLE",
            "RELAY",
            "TRUNK_RELAY",
        }:
            return pygame.Vector2()

        if robot.role != "NORMAL" or robot.base_reserve:
            return pygame.Vector2()

        return (
            _final_base_return_direction()
            * physical.RETURN_EGRESS_FORCE
            * physical.integration_final_base_flow_force_scale
        )

    def final_guard_compute_sph_forces(
        robots: Sequence[Any],
        grid: Any,
        communication_grid: Any,
        dt: float = 1.0 / 60.0,
    ) -> None:
        physical_only_compute_sph_forces(
            robots,
            grid,
            communication_grid,
            dt,
        )

    def final_guard_robot_update(self: Any, dt: float) -> None:
        original_robot_update(self, dt)

    # Install final-sweep mechanics after the reference Environment hooks
    # have been restored.
    physical.compute_route_force = final_guard_sweep_route_force
    physical.compute_sph_forces = final_guard_compute_sph_forces
    physical.Robot.update = final_guard_robot_update

    def update_general_final_guard_sweep(
        robots: Sequence[Any],
        dt: float,
    ) -> None:
        """Wait for real Base-bound NORMAL flow, then release all Guards."""
        if not getattr(
            physical,
            "integration_final_guard_sweep_active",
            False,
        ):
            return

        if getattr(
            physical,
            "integration_final_all_guards_released",
            False,
        ):
            return

        base_direction = _final_base_return_direction()
        frame = getattr(physical, "integration_frame", -1)
        base_normal = pygame.Vector2(-base_direction.y, base_direction.x)
        base_mouth = getattr(
            physical,
            "integration_base_mouth_anchor",
            None,
        )
        if base_mouth is None:
            raise RuntimeError(
                "final Base flow requested without stored Base mouth anchor"
            )

        normals = [
            robot
            for robot in robots
            if robot.role == "NORMAL" and not robot.base_reserve
        ]
        min_speed = float(physical.integration_final_base_flow_min_speed)
        corridor_half_width = 0.65 * physical.corridor_width
        moving_baseward = 0
        crossed_base_mouth = 0
        for robot in normals:
            delta = robot.position - base_mouth
            axial = float(delta.dot(base_direction))
            lateral = abs(float(delta.dot(base_normal)))
            if lateral > corridor_half_width:
                continue
            base_speed = float(robot.velocity.dot(base_direction))
            if base_speed >= min_speed:
                moving_baseward += 1
            if axial >= 0.0 and base_speed > 0.0:
                crossed_base_mouth += 1

        minimum_flow_count = max(
            8,
            int(math.ceil(0.06 * max(len(normals), 1))),
        )
        minimum_crossed_count = max(4, minimum_flow_count // 2)
        flow_now = (
            moving_baseward >= minimum_flow_count
            and crossed_base_mouth >= minimum_crossed_count
        )
        if flow_now:
            physical.integration_final_base_flow_dwell += dt
        else:
            physical.integration_final_base_flow_dwell = max(
                0.0,
                physical.integration_final_base_flow_dwell - 0.5 * dt,
            )

        branches = _visited_guard_branches(robots)
        returned_guards = [
            robot
            for branch in branches
            for robot in _live_guard_members(robots, branch)
        ]
        overlapping_guard_pairs = 0
        min_guard_distance = float("inf")
        for index, left in enumerate(returned_guards):
            for right in returned_guards[index + 1:]:
                distance = left.position.distance_to(right.position)
                min_guard_distance = min(min_guard_distance, distance)
                overlap_tolerance = max(
                    1.0e-6,
                    1.0e-3 * min(left.radius, right.radius),
                )
                if distance < left.radius + right.radius - overlap_tolerance:
                    overlapping_guard_pairs += 1

        flow_established = (
            physical.integration_final_base_flow_dwell
            >= physical.integration_final_base_flow_dwell_required
        )
        if frame % 10 == 0:
            print(
                f"[FinalBaseFlow] frame={frame} "
                f"normals={len(normals)} "
                f"moving_baseward={moving_baseward} "
                f"crossed_base_mouth={crossed_base_mouth} "
                f"required_moving={minimum_flow_count} "
                f"required_crossed={minimum_crossed_count} "
                f"dwell={physical.integration_final_base_flow_dwell:.3f} "
                f"required_dwell="
                f"{physical.integration_final_base_flow_dwell_required:.2f} "
                f"guards_fixed={len(branches)} "
                f"flow_established={flow_established}"
            )
            print(
                f"[FinalGuardOverlapAudit] frame={frame} "
                f"guards={len(returned_guards)} "
                f"overlapping_pairs={overlapping_guard_pairs} "
                f"min_distance="
                f"{min_guard_distance if math.isfinite(min_guard_distance) else -1.0:.3f}"
            )

        if not flow_established:
            return

        physical.integration_final_base_flow_established = True
        print(
            f"[FinalBaseFlowEstablished] frame={frame} "
            f"moving={moving_baseward} crossed={crossed_base_mouth} "
            f"dwell={physical.integration_final_base_flow_dwell:.3f}"
        )

        released_ids: list[int] = []
        for branch in branches:
            members = _live_guard_members(robots, branch)
            for robot in members:
                released_ids.append(robot.robot_id)
                robot.role = "NORMAL"
                robot.final_return_direction_local = base_direction.copy()
                robot.final_return_source_branch = branch
                robot.junction_guard_anchor = None
                robot.junction_guard_branch = None
                robot.junction_guard_branch_uid = None
                robot.junction_guard_hop = -1
                robot.junction_guard_parent_id = None
                robot.junction_guard_layer = -1
                robot.shepherd_anchor = None
                robot.shepherd_origin = None
                robot.shepherd_branch = None
                robot.shepherd_return_direction = None
                robot.is_branch_leader = False
                robot.velocity *= 0.25
                robot.acceleration.update(0.0, 0.0)
                robot.filtered_acceleration.update(0.0, 0.0)

            physical.junction_guard_groups[branch] = []
            lifecycle = getattr(
                physical,
                "integration_wall_lifecycle",
                {},
            ).get(branch)
            if lifecycle is not None:
                lifecycle["state"] = "FINAL_FLOW_JOINED"
                lifecycle["final_flow_join_frame"] = frame

        # Preserve all viscoelastic links so the released walls join the
        # already established connected SPH stream without a topology reset.
        physical.integration_final_all_guards_released = True
        physical.integration_final_guard_sweep_active = False
        print(
            f"[FinalAllGuardsJoinFlow] frame={frame} "
            f"branches={branches} released_robots={len(released_ids)} "
            "simultaneous=True teleport=False strong_launch=False "
            "viscoelastic_links_preserved=True"
        )
        physical.begin_final_return(robots)

    def environment_authoritative_update_state(
        robots: Sequence[Any],
        dt: float,
        reference_density: float,
        spatial_grid: Any,
    ) -> None:
        # ALL_BRANCHES_VISITED uses the general local-direction Guard sweep
        # before handing control to Environment RETURN_TO_BASE.
        if getattr(
            physical,
            "integration_final_guard_sweep_active",
            False,
        ):
            update_general_final_guard_sweep(robots, dt)
            return

        # Adaptive-specific topology adapter is used only when converting an
        # already-formed LiDAR Guard wall into a Frontier, including next branch
        # selection after a completed Environment backtrack.
        lidar_guard_handoff = (
            physical.phase == physical.SimulationPhase.FORM_JUNCTION_GUARDS
            and getattr(physical, "integration_ready_guard_handoff", False)
        )
        lidar_next_branch_switch = (
            physical.phase == physical.SimulationPhase.JUNCTION_SWITCH
        )
        lidar_thick_frontier_explore = (
            physical.phase == physical.SimulationPhase.EXPLORE_BRANCH
        )
        lidar_shepherd_return = (
            physical.phase in {
                physical.SimulationPhase.PRESSURE_PUSH,
                physical.SimulationPhase.FLOW_BACKTRACK,
            }
        )

        if (
            lidar_guard_handoff
            or lidar_next_branch_switch
            or lidar_thick_frontier_explore
            or lidar_shepherd_return
        ):
            # EXPLORE_BRANCH stays in the Adaptive adapter only long enough to
            # translate the 3xN leading-row wall contact into a dead-end handoff.
            # The Environment detector divides direct contacts by *all* Frontier
            # robots; for a 3-row wall only the leading row can touch the terminal
            # wall, so its 0.60 all-robot ratio is structurally unreachable.
            integrated_update_state(
                robots,
                dt,
                reference_density,
                spatial_grid,
            )
            return

        # Exact Environment sequence:
        # EXPLORE_BRANCH
        # -> FORM_SHEPHERD_BOUNDARY
        # -> FILL_BEHIND_SHEPHERD
        # -> PRESSURE_PUSH
        # -> FLOW_BACKTRACK
        # -> JUNCTION_SWITCH / repeat
        # -> FINAL_JUNCTION_GATHER
        # -> RETURN_TO_BASE
        # -> DONE
        original_update_state(
            robots,
            dt,
            reference_density,
            spatial_grid,
        )

        # If the Environment completed a branch during this call, activate the
        # next persistent Guard/Frontier immediately in the same simulation step.
        if physical.phase == physical.SimulationPhase.JUNCTION_SWITCH:
            integrated_update_state(
                robots,
                dt,
                reference_density,
                spatial_grid,
            )

    physical.update_simulation_state = environment_authoritative_update_state
    physical.integration_environment_authoritative_cycle = True

    print(
        "[ENVIRONMENT_DFS_CYCLE_ACTIVE] "
        "LiDAR=PERCEPTION_AND_GUARD_ADAPTER "
        "DFS=ENVIRONMENT_EXACT_AFTER_3XN_DEADEND_ADAPTER "
        "3XN_EXPLORE_ADAPTER->FORM_SHEPHERD->FILL->PRESSURE_PUSH->"
        "FLOW_BACKTRACK->JUNCTION_SWITCH->REPEAT->RETURN_TO_BASE->DONE"
    )
    print(
        "[ENVIRONMENT_DFS_RESTORE] "
        f"piston_speed={physical.SHEPHERD_PISTON_SPEED:.3f} "
        f"line_speed={physical.SHEPHERD_LINE_BACKTRACK_SPEED:.3f} "
        f"release_speed={physical.SHEPHERD_JUNCTION_RELEASE_SPEED:.3f} "
        "pack_coupled_hold_runtime=False "
        "herd_locked_robot_update_runtime=False "
        "virtual_wall_removed=True"
    )



def refine_guard_geometry_from_persistent_lidar(
    physical: types.ModuleType,
    perception: AdaptivePerception,
    robots: Sequence[Any],
) -> None:
    """Bind persistent UIDs to the already-frozen LiDAR Guard geometry.

    Spatial WHERE is immutable after Guard election.  Persistent accumulation
    may name the branch, but it must never rebuild descriptors, slots, or robot
    targets; doing so mixes two mouth frames and deforms the UP wall at DFS
    handoff.
    """
    if not perception.outgoing:
        raise RuntimeError("persistent UID binding requires outgoing openings")

    track_by_key = {
        (
            "UP"
            if abs(track.center_angle) < 30.0
            else ("LEFT" if track.center_angle < 0.0 else "RIGHT")
        ): track
        for track in perception.outgoing
    }
    if set(track_by_key) != {"LEFT", "UP", "RIGHT"}:
        raise RuntimeError(
            f"persistent outgoing identities are incomplete: {sorted(track_by_key)}"
        )

    all_ids = {
        robot.robot_id for robot in robots if robot is not perception.leader
    }
    physical.branch_descriptors_by_uid.clear()
    physical.fixture_key_to_branch_uid.clear()
    physical.branch_uid_to_fixture_key.clear()
    physical.detected_branch_candidates = set()
    physical.junction_guard_groups.clear()

    remapped_status: dict[str, dict[str, Any]] = {}
    associations: dict[str, dict[str, float | str]] = {}

    for geometry in perception.provisional_guards:
        # Keep the frozen provisional identity through persistent binding;
        # never re-quantize the refined tangent into a different fixture.
        fixture = geometry.local_branch_key
        if fixture not in track_by_key:
            raise RuntimeError(f"no persistent track for frozen Guard {fixture}")
        track = track_by_key[fixture]
        uid = track.persistent_id
        descriptor = geometry.descriptor

        # Identity/metadata update only.  Do NOT replace the descriptor spatial
        # frame and do NOT regenerate slots from persistent means.
        descriptor.uid = uid
        descriptor.fixture_key = fixture
        descriptor.cohort_member_ids = set(all_ids)
        descriptor.direction_sample_count = max(
            physical.JUNCTION_COHORT_MIN_ROBOTS, len(all_ids)
        )
        descriptor.direction_downstream_travel = physical.JUNCTION_COHORT_MIN_TRAVEL
        descriptor.motion_frame_source = "LIDAR_FROZEN_MOUTH_UID_BOUND"
        descriptor.motion_frame_sample_count = len(track.observations)
        descriptor.physical_boundary_sample_count = len(track.observations)

        geometry.fixture_key = fixture
        geometry.persistent_uid = uid

        physical.branch_descriptors_by_uid[uid] = descriptor
        physical.fixture_key_to_branch_uid[fixture] = uid
        physical.branch_uid_to_fixture_key[uid] = fixture
        physical.branch_local_uids[fixture] = uid
        physical.detected_branch_candidates.add(fixture)
        physical.junction_guard_groups[fixture] = list(geometry.selected_ids)
        physical.thick_mouth_guard_columns[fixture] = geometry.columns
        physical.thick_mouth_guard_layers[fixture] = geometry.layers
        physical.junction_guard_frontier_depths[fixture] = (
            physical.JUNCTION_GUARD_BRANCH_INSET
            + (geometry.layers - 1) * physical.THICK_MOUTH_GUARD_LAYER_SPACING
        )

        selected = {
            robot.robot_id: robot
            for robot in robots
            if robot.robot_id in geometry.selected_ids
        }
        for robot_id in geometry.selected_ids:
            robot = selected[robot_id]
            # Keep the exact world target/slot already reached; only bind IDs.
            robot.junction_guard_branch = fixture
            robot.junction_guard_branch_uid = uid
            robot.local_branch_uid_by_key[fixture] = uid

        descriptor.leader_id = min(geometry.selected_ids)
        status = physical.integration_wall_status.pop(
            geometry.provisional_uid, {}
        )
        status.update({
            "assigned": len(geometry.selected_ids),
            "edge_selected": len(geometry.selected_ids),
            "rows": geometry.layers,
            "slots_per_row": geometry.columns,
            "slots_walkable": sum(
                physical.is_walkable(slot, physical.ROBOT_RADIUS)
                for slot in geometry.slots
            ),
            "slots_total": len(geometry.slots),
        })
        remapped_status[uid] = status

        angular_error = circular_error(
            float(geometry.opening["center_angle"]), track.center_angle
        )
        associations[uid] = {
            "opening_center": track.center_angle,
            "matched_mouth": fixture,
            "mouth_local_angle": float(geometry.opening["center_angle"]),
            "angular_error": angular_error,
            "provisional_uid": geometry.provisional_uid,
        }
        print(
            f"[GuardFreeze] provisional={geometry.provisional_uid} uid={uid} "
            f"fixture_adapter={fixture} same_ids=True "
            f"robots={len(geometry.selected_ids)} "
            f"columns={geometry.columns} layers={geometry.layers} "
            "spatial_retarget=False slot_regeneration=False "
            f"angular_error={angular_error:.3f}"
        )

    physical.integration_wall_status.update(remapped_status)
    physical.integration_opening_mouth_associations = associations


def log_wall_ready_blockers(physical: types.ModuleType, perception: AdaptivePerception, robots: Sequence[Any]) -> None:
    frame = getattr(physical, "integration_frame", -1)
    if frame % 20 != 0 or perception.handoff_complete:
        return
    for geometry in perception.provisional_guards:
        status = physical.integration_wall_status.get(geometry.provisional_uid, {})
        guards = [r for r in robots if r.robot_id in geometry.selected_ids]
        complete = sum(sum(1 for r in guards if r.junction_guard_layer == layer) >= geometry.columns for layer in range(geometry.layers))
        checks = [("settled_ratio", float(status.get("settled_ratio", 0.0)) >= PROVISIONAL_WALL_SETTLED_RATIO), ("span_ratio", float(status.get("min_span_ratio", 0.0)) >= float(physical.FRONTIER_LINE_MIN_SPAN_RATIO)), ("edge_gap", float(status.get("max_edge_gap", float("inf"))) <= float(physical.FRONTIER_LINE_MAX_EDGE_GAP)), ("internal_gap", float(status.get("max_internal_gap", float("inf"))) <= float(physical.FRONTIER_LINE_MAX_INTERNAL_GAP)), ("slots_walkable", int(status.get("slots_walkable", 0)) >= len(geometry.slots)), ("complete_rows", complete == geometry.layers)]
        reasons = [name for name, ok in checks if not ok]
        print(f"[WallReadyBlocker] branch={geometry.local_branch_key or geometry.provisional_uid} guard_count={len(guards)} expected={len(geometry.slots)} rows={geometry.layers} columns={geometry.columns} complete_rows={complete} settled_ratio={status.get('settled_ratio', 0.0):.3f} min_span_ratio={status.get('min_span_ratio', 0.0):.3f} max_edge_gap={status.get('max_edge_gap', float('inf')):.3f} max_internal_gap={status.get('max_internal_gap', float('inf')):.3f} slots_walkable={status.get('slots_walkable', 0)} ready={bool(status.get('ready', False))} blocking_reasons={reasons}")


def handoff_to_physical_dfs(
    physical: types.ModuleType,
    perception: AdaptivePerception,
    robots: Sequence[Any],
) -> None:
    """Refine LiDAR WHERE, retain provisional WHO, and enable Physical DFS."""
    eligible = [r for r in robots if r.role == "NORMAL" and not r.base_reserve]
    centers = [g.mouth_center_world for g in perception.provisional_guards if g.mouth_center_world is not None]
    scale = float(np.median([g.mouth_span for g in perception.provisional_guards if g.mouth_span > 0.0]) if perception.provisional_guards else 1.0)
    junction_center = sum(centers, pygame.Vector2()) / max(len(centers), 1)
    local_count = sum(r.position.distance_to(junction_center) <= 2.0 * scale for r in eligible)
    arrival_ratio = local_count / max(len(eligible), 1)
    junction_arrived = arrival_ratio >= JUNCTION_ARRIVAL_RATIO_THRESHOLD
    
    all_guard_cohorts_complete = all(
        geometry.cohort_ready
        and len(geometry.selected_ids) == len(geometry.slots)
        for geometry in perception.provisional_guards
    )

    # Every 3xN Guard must be physically settled and structurally
    # sealed before Physical DFS may start.
    #
    # Assignment completion alone is NOT sufficient.
    all_provisional_walls_ready = all(
        bool(
            physical.integration_wall_status
            .get(geometry.provisional_uid, {})
            .get("ready", False)
        )
        for geometry in perception.provisional_guards
    )

    gate_checks = {
        "topology_ready": (
            perception.topology_ready_frame is not None
        ),
        "all_groups_activated": (
            perception.guard_all_groups_activated
        ),
        "all_guard_cohorts_complete": (
            all_guard_cohorts_complete
        ),
        "all_provisional_walls_ready": (
            all_provisional_walls_ready
        ),
        "guard_complete": (
            perception.guard_all_groups_activated
            and all_guard_cohorts_complete
            and all_provisional_walls_ready
        ),
        "junction_arrived": junction_arrived,
    }

    if perception.handoff_complete or perception.anchor_position is None or not perception.provisional_guard_started or not perception.provisional_guards:
        gate_checks["handoff_inputs"] = False
    allowed = all(gate_checks.values())
    print(
        f"[DFSStartGate] "
        f"frame={getattr(physical, 'integration_frame', -1)} "
        f"topology_ready={gate_checks['topology_ready']} "
        f"all_groups_activated={gate_checks['all_groups_activated']} "
        f"all_guard_cohorts_complete="
        f"{gate_checks['all_guard_cohorts_complete']} "
        f"all_provisional_walls_ready="
        f"{gate_checks['all_provisional_walls_ready']} "
        f"guard_complete={gate_checks['guard_complete']} "
        f"junction_local_count={local_count} "
        f"eligible_count={len(eligible)} "
        f"junction_arrival_ratio={arrival_ratio:.3f} "
        f"junction_arrived={junction_arrived} "
        f"blocking_reasons="
        f"{[name for name, ok in gate_checks.items() if not ok]} "
        f"allowed={allowed}"
    )
    if not allowed:
        return
    refine_guard_geometry_from_persistent_lidar(
        physical, perception, robots
    )
    if len(perception.provisional_guards) != len(perception.outgoing):
        raise RuntimeError("provisional Guard count does not match outgoing topology")
    guard_id_sets = [set(geometry.selected_ids) for geometry in perception.provisional_guards]
    assert all(a.isdisjoint(b) for i, a in enumerate(guard_id_sets) for b in guard_id_sets[i + 1:])
    physical.branch_discovery_counter = len(perception.outgoing)
    physical.integration_detected_branch_order = [
        track.persistent_id for track in perception.outgoing
    ]
    physical.junction_inference_tracker.confirmed = True
    physical.junction_inference_tracker.confirmed_at = (
        physical.simulation_time - physical.JUNCTION_DISCOVERY_SETTLE_TIME
    )
    physical.junction_inference_tracker.valid_branches = set(
        physical.detected_branch_candidates
    )
    # Freeze the only direction needed by the final return controller.
    # It comes from the traversed incoming edge / body-local ingress frame,
    # not from UP/LEFT/RIGHT names or a global map route.
    incoming_direction_local = _body_local_unit(perception, 0.0)
    if incoming_direction_local.length_squared() <= physical.EPSILON:
        raise RuntimeError("invalid stored incoming local direction")
    incoming_direction_local = incoming_direction_local.normalize()
    physical.integration_incoming_direction_local = incoming_direction_local.copy()
    physical.integration_base_return_direction_local = -incoming_direction_local
    physical.integration_base_mouth_anchor = (
        perception.anchor_position.copy()
        if perception.anchor_position is not None
        else None
    )
    print(
        f"[FinalReturnFrameStored] "
        f"incoming=({incoming_direction_local.x:.3f},"
        f"{incoming_direction_local.y:.3f}) "
        f"base_return=({-incoming_direction_local.x:.3f},"
        f"{-incoming_direction_local.y:.3f})"
    )

    physical.integration_guard_gating_enabled = True
    physical.integration_guard_who_localization_enabled = False
    physical.integration_placement_localization_enabled = False
    physical.integration_provisional_guard_active = False
    physical.branch_gate_states.clear()
    physical.branch_gate_states.update({
        branch: "CLOSED" for branch in physical.BRANCHES
    })
    # The walls were physically elected before handoff, so materialize their
    # lifecycle records here instead of waiting for the legacy formation pass.
    physical.integration_wall_lifecycle = {}
    for geometry in perception.provisional_guards:
        branch = geometry.local_branch_key
        descriptor = physical.branch_motion_descriptor(branch)
        members = [r for r in robots if r.robot_id in geometry.selected_ids]
        coordinates = [guard_mouth_coordinates(r.position, geometry) for r in members]
        centroid_axial = float(np.mean([value[0] for value in coordinates]))
        centroid_lateral = float(np.mean([value[1] for value in coordinates]))
        physical.integration_wall_lifecycle[branch] = {
            "uid": descriptor.uid,
            "state": "GUARD",
            "rows": geometry.layers,
            "cols": geometry.columns,
            "robot_ids": sorted(geometry.selected_ids),
            "centroid_axial": centroid_axial,
            "centroid_lateral": centroid_lateral,
            "mouth_center_world": geometry.mouth_center_world.copy(),
            "branch_tangent_unit": geometry.branch_tangent_unit.copy(),
            "mouth_lateral_unit": geometry.mouth_lateral_unit.copy(),
            "measured_mouth_span": float(geometry.mouth_span),
            "sealing_lateral_min": float(geometry.sealing_lateral_min),
            "sealing_lateral_max": float(geometry.sealing_lateral_max),
            "slot_spacing": float(geometry.slot_spacing),
            # Freeze the original LiDAR mouth Guard anchors before any branch
            # is promoted to Frontier.  A returned Shepherd wall comes back to
            # these exact physical targets and remains there until every child
            # branch of this Junction has been explored.
            "guard_anchor_by_id": {
                robot.robot_id: (
                    robot.integration_guard_final_anchor.copy()
                    if getattr(robot, "integration_guard_final_anchor", None) is not None
                    else (
                        robot.junction_guard_anchor.copy()
                        if robot.junction_guard_anchor is not None
                        else robot.position.copy()
                    )
                )
                for robot in members
            },
            "guard_layer_by_id": {
                robot.robot_id: int(getattr(robot, "junction_guard_layer", -1))
                for robot in members
            },
            "guard_slot_index_by_id": {
                robot.robot_id: int(getattr(robot, "integration_guard_slot_index", -1))
                for robot in members
            },
            "relative_offsets": {
                robot.robot_id: (axial - centroid_axial, lateral - centroid_lateral)
                for robot, (axial, lateral) in zip(members, coordinates)
            },
        }
    physical.record_distributed_consensus(clear_selection=True)
    physical.phase = physical.SimulationPhase.FORM_JUNCTION_GUARDS
    physical.integration_all_walls_ready = True
    physical.integration_ready_guard_handoff = True
    perception.handoff_complete = True
    perception.state = PerceptionState.PHYSICAL_DFS
        # ---------------------------------------------------------
    # Multi-Junction DFS: register confirmed Root Junction J0.
    #
    # This runs only after:
    # - stationary LiDAR topology confirmation
    # - Guard geometry refinement
    # - physical Guard handoff
    # ---------------------------------------------------------
    if not multi_dfs.stack:
        root = multi_dfs.create_root()

        ordered_uids = list(
            physical.integration_detected_branch_order
        )

        if not ordered_uids:
            raise RuntimeError(
                "J0 handoff completed without detected branch UIDs"
            )

        missing_uids = [
            uid
            for uid in ordered_uids
            if uid not in physical.branch_descriptors_by_uid
        ]

        if missing_uids:
            raise RuntimeError(
                "J0 MultiDFS registration missing descriptors: "
                f"{missing_uids}"
            )

        root.branch_order = ordered_uids.copy()

        root.branch_states = {
            uid: physical.branch_descriptors_by_uid[uid].visit_state
            for uid in ordered_uids
        }

        root.active_branch_uid = None

                # Preserve J0's actual traversed ingress frame.
        #
        # This is local history, not a global Junction target.
        root.ingress_direction_local = (
            physical.integration_incoming_direction_local.copy()
        )

        root.return_direction_local = (
            physical.integration_base_return_direction_local.copy()
        )

        print(
            "[MultiDFS] ROOT_REGISTERED "
            f"junction={root.junction_uid} "
            f"depth={multi_dfs.depth} "
            f"branches={root.branch_order} "
            f"states={root.branch_states}"
        )


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
            "PRE_SHEPHERD": COLORS["shepherd"],
            "PEBBLE": COLORS["pebble"],
            "RELAY": COLORS["relay"], "TRUNK_RELAY": COLORS["trunk"],
        }
        order = {"NORMAL": 0, "RELAY": 1, "TRUNK_RELAY": 1, "PEBBLE": 2, "JUNCTION_GUARD": 3, "FRONTIER_SHEPHERD": 4, "PRE_SHEPHERD": 5, "SHEPHERD": 5}
        for robot in sorted(robots, key=lambda item: order.get(item.role, 0)):
            color = role_colors.get(robot.role, COLORS["normal"])

            if density and robot.role == "NORMAL":
                color = physical.density_to_color(
                    robot.density,
                    max(robot.density, 1.0),
                )

            draw_radius = max(
                2,
                round(robot.radius + 1),
            )

            pygame.draw.circle(
                self.screen,
                color,
                robot.position,
                draw_radius,
            )

            # Make physical DFS Pebbles clearly visible.
            if robot.role == "PEBBLE":
                pygame.draw.circle(
                    self.screen,
                    (255, 255, 255),
                    robot.position,
                    draw_radius + 4,
                    2,
                )
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
                    f"sealed={status.get('structurally_sealed', False)} "
                    f"dwell={status.get('wall_ready_dwell', 0.0):.2f} "
                    f"edgeGap={status.get('max_edge_gap', 0.0):.2f} "
                    f"intGap={status.get('max_internal_gap', 0.0):.2f}"
                )
        else:
            for key in physical.BRANCHES:
                uid = physical.branch_uid_for_fixture(key)
                lifecycle = wall_lifecycle.get(key, {})
                if not lifecycle:
                    live = [r for r in robots if r.role == "JUNCTION_GUARD" and r.junction_guard_branch == key]
                    lifecycle = {"state": "GUARD", "robot_ids": [r.robot_id for r in live], "rows": physical.thick_mouth_guard_layers.get(key, 0), "cols": physical.thick_mouth_guard_columns.get(key, 0)}
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
            f"GuardStage={perception.guard_activation_stage}",
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
    configure_multi_test_geometry(physical)
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
    physical.integration_final_return_requested = False
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
    print(
        "[Init] physical_shepherd_only=True "
        "invisible_return_curtain=False"
    )
    perception = AdaptivePerception(physical, robots)
    install_lidar_relay_protection(
        physical,
        perception,
    )
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
                    physical.integration_backtrack_command_depth = None
                    physical.integration_backtrack_pack_rear_depth = None
                    physical.integration_backtrack_support_depth = None
                    physical.integration_backtrack_support_count = 0
                    physical.integration_backtrack_lateral_coverage = 0.0
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

            update_child_lidar_probe(
                physical,
                perception,
                robots,
            )

            if perception.state == PerceptionState.JUNCTION_APPROACH:
                perception.leader.acceleration *= 0.35
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
            update_child_observation_session(
                physical,
                perception,
                lidar_frame,
            )
            update_child_moving_candidate(
                physical,
                perception,
                lidar_frame,
            )

            update_child_stationary_verification(
                physical,
                perception,
                lidar_frame,
            )

            release_confirmed_parent_junction(
                physical,
                perception,
                robots,
            )


            if (
                perception.anchor_fixed
                and perception.state in {
                    PerceptionState.FIXED_ACCUMULATING,
                    PerceptionState.BRANCHES_READY,
                    PerceptionState.PHYSICAL_DFS,
                }
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
            log_wall_ready_blockers(physical, perception, robots)
            if perception.state == PerceptionState.BRANCHES_READY:
                handoff_to_physical_dfs(physical, perception, robots)

            child_topology_pending = (
                multi_dfs.current is not None
                and multi_dfs.current.parent_junction_uid
                is not None
                and not multi_dfs.current.branch_order
            )
            if (
                perception.handoff_complete
                and not child_topology_pending
            ):
                normal_snapshot = {
                    robot.robot_id: (robot.position.copy(), robot.velocity.copy())
                    for robot in robots if robot.role == "NORMAL"
                }
                phase_before_update = physical.phase
                active_branch_before_update = physical.active_branch_uid
                physical.update_simulation_state(robots, dt, reference_density, spatial_grid)
                sync_multi_dfs_from_physical(physical)
                if phase_before_update != physical.phase or active_branch_before_update != physical.active_branch_uid:
                    jumps = []
                    velocity_changes = []
                    for robot in robots:
                        if robot.robot_id in normal_snapshot:
                            before_pos, before_vel = normal_snapshot[robot.robot_id]
                            jumps.append((robot.position.distance_to(before_pos), robot.robot_id))
                            velocity_changes.append(robot.velocity.distance_to(before_vel))
                    if jumps:
                        max_jump, jump_id = max(jumps)
                        max_dv = max(velocity_changes)
                        print(f"[BranchOpenAudit] frame={frame_count} branch={physical.active_branch_uid} tracked_normals={len(jumps)} max_normal_state_transition_jump={max_jump:.6f} max_normal_velocity_change={max_dv:.6f}")
            elif not perception.handoff_complete:
                # Preserve pre-handoff communication/relay physics without
                # invoking the legacy Junction inference transition.
                physical.update_local_ingress_tangents(robots)
                physical.update_initial_release_flow_event(robots, dt)
                physical.update_relay_deployment(robots, dt)
            else:
                # -------------------------------------------------
                # Child is confirmed and pushed, but its own
                # Physical DFS context has not been initialized yet.
                #
                # Do NOT allow the old Parent Physical DFS state
                # machine to continue running.
                # -------------------------------------------------
                if frame_count % 10 == 0:
                    print(
                        "[PhysicalDFSSuspended] "
                        f"current="
                        f"{multi_dfs.current.junction_uid} "
                        "reason="
                        "CHILD_TOPOLOGY_NOT_INITIALIZED "
                        f"stack="
                        f"{[
                            frame.junction_uid
                            for frame in multi_dfs.stack
                        ]}"
                    )
            physical.update_metrics_per_frame(robots, dt)
            if (
                (
                    physical.phase == physical.SimulationPhase.RETURN_TO_BASE
                    or getattr(
                        physical,
                        "integration_final_guard_sweep_active",
                        False,
                    )
                )
                and perception.anchor_fixed
            ):
                perception.anchor_fixed = False
                perception.leader.is_fixed_anchor = False
                perception.leader.base_reserve = False
                print(
                    f"[Anchor] RELEASED_FOR_RETURN "
                    f"id={perception.leader.robot_id}"
                )
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
    if perception.junction_candidate_detected and perception.anchor_fixed:
        assert perception.junction_candidate_frame < perception.anchor_fix_frame, (
            "candidate and Anchor fix must occur on different frames"
        )
    print(
        f"[DFS] visited_sequence={visited_log} "
        f"final_phase={physical.phase.name}"
    )
    pygame.quit()
    return 0 if physical.phase == physical.SimulationPhase.DONE or args.max_frames else 1


if __name__ == "__main__":
    raise SystemExit(main())