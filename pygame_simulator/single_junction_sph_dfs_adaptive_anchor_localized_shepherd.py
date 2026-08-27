"""Adaptive Anchor to Physical-DFS Guard lifecycle integration boundary.

This file deliberately imports the existing Adaptive-W/LiDAR/SPH baseline and
does not modify either source implementation.  It proves the integration path
through LiDAR-only Branch discovery, one-shot localization-assisted Guard *ID
election* for every outgoing Branch, local-frame physical Guard formation, and
the same-ID Guard-to-Frontier transition. Fixture/global geometry is discarded
after election. Dead-end handling, backtracking, Junction return, and DFS
completion remain intentionally disabled.

The explicit BLOCKED phase report is a safety property, not an emulation of a
completed Physical DFS controller.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pygame_simulator import (  # noqa: E402
    lidar_junction_detection_adaptive_w_tau_anchor_stop as adaptive,
)


EXPERIMENT_NAME = (
    "EXP-Physical-Guard-Lifecycle-01 Three-Branch Junction Guard and Frontier Transition Validation"
)
BASELINE_MARGIN_RATIO = 0.05
BASELINE_ALPHA = 0.5
BASELINE_NOISE_MODEL = "none"
BASELINE_MAP_CASE = adaptive.M1_PRE_CORRIDOR_CASE
REFERENCE_PROTOTYPE = (
    PROJECT_ROOT
    / "pygame_simulator/single_junction_sph_dfs_anchor_junction_detection.py"
)
ADAPTIVE_SOURCE = (
    PROJECT_ROOT
    / "pygame_simulator/lidar_junction_detection_adaptive_w_tau_anchor_stop.py"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "pygame_simulator/single_junction_sph_dfs_adaptive_anchor_localized_shepherd_output"
)

# Values are reused from the audited reference implementation.
REFERENCE_MAP_SCALE = 0.70
REFERENCE_SAFE_RADIUS = 7.5 * REFERENCE_MAP_SCALE
REFERENCE_SHEPHERD_EDGE_MARGIN = 12.0 * REFERENCE_MAP_SCALE
REFERENCE_SHEPHERD_TARGET_SLOT_SPACING = REFERENCE_SAFE_RADIUS * 0.85
REFERENCE_SHEPHERD_MIN_COUNT = 5
REFERENCE_SHEPHERD_MAX_COUNT = 22
REFERENCE_EARLY_CAPTURE_DEPTH = 34.0 * REFERENCE_MAP_SCALE
ANCHOR_BRANCH_OBSERVATION_SAMPLES = 12
BRANCH_ASSOCIATION_TOLERANCE_DEG = max(
    2.0 * float(adaptive.FROZEN_PARAMETERS["merge_gap_deg"]),
    float(adaptive.FROZEN_PARAMETERS["min_opening_width_deg"]),
)
BRANCH_MIN_OBSERVATIONS = ANCHOR_BRANCH_OBSERVATION_SAMPLES // 2 + 1
REQUIRED_CHILD_COUNT_FOR_BASELINE = 3

# Reused verbatim/scaled from the audited Physical DFS reference.
JUNCTION_GUARD_COVERAGE = 12.5 * REFERENCE_MAP_SCALE
JUNCTION_GUARD_MIN_COUNT = 5
JUNCTION_GUARD_MAX_COUNT = 11
JUNCTION_GUARD_BRANCH_INSET = 5.0 * REFERENCE_MAP_SCALE
JUNCTION_GUARD_RECRUIT_RADIUS = 78.0 * REFERENCE_MAP_SCALE
JUNCTION_GUARD_POSITION_TOLERANCE = 3.0 * REFERENCE_MAP_SCALE
JUNCTION_GUARD_FORM_DWELL = 0.10
FRONTIER_LINE_TARGET_SETTLED_RATIO = 0.80
FRONTIER_LINE_MIN_SPAN_RATIO = 0.96
FRONTIER_LINE_EDGE_CLEARANCE = adaptive.ROBOT_RADIUS * 0.25
PHYSICAL_GUARD_INFLUENCE_RADIUS = max(
    JUNCTION_GUARD_COVERAGE * 0.78,
    adaptive.SAFE_RADIUS * 1.15,
)
FRONTIER_LINE_MAX_EDGE_GAP = max(
    JUNCTION_GUARD_POSITION_TOLERANCE,
    PHYSICAL_GUARD_INFLUENCE_RADIUS * 0.35,
)
FRONTIER_LINE_MAX_INTERNAL_GAP = PHYSICAL_GUARD_INFLUENCE_RADIUS * 1.35
PHYSICAL_GUARD_INWARD_GAIN = 118.0 * 3.0
PHYSICAL_GUARD_LATERAL_GAIN = 32.0 * 3.0
PHYSICAL_GUARD_FORCE_LIMIT = 105.0 * 3.0
JUNCTION_GUARD_MOVE_SPEED = 46.0 * 3.0
FRONTIER_LINE_ADVANCE_SPEED = 52.0 * 3.0
FRONTIER_LINE_FORM_SPEED = 58.0 * 3.0
FRONTIER_LINE_LEAD_GAP = 12.0 * REFERENCE_MAP_SCALE
FRONTIER_LINE_SUPPORT_QUANTILE = 0.98
FRONTIER_LINE_TARGET_TOLERANCE = max(
    JUNCTION_GUARD_POSITION_TOLERANCE,
    adaptive.ROBOT_RADIUS * 1.25,
)

# Shadow-only wall topology scales.  Both values are imported from the frozen
# detector; no new angular tuning threshold is introduced here.
LOCAL_WALL_BAND_DEG = float(
    adaptive.FROZEN_PARAMETERS["boundary_search_deg"]
)
LOCAL_WALL_DIRECTION_TOLERANCE_DEG = LOCAL_WALL_BAND_DEG

STATE_ANCHOR_FIXED = "ANCHOR_FIXED"
STATE_BRANCHES_VALIDATED = "BRANCHES_VALIDATED"
STATE_FORM_ALL_JUNCTION_GUARDS = "FORM_ALL_JUNCTION_GUARDS"
STATE_ALL_BRANCH_GUARDS_READY = "ALL_BRANCH_GUARDS_READY"
STATE_FRONTIER_SELECTED = "FRONTIER_SELECTED"
STATE_SELECTED_BRANCH_EXPLORATION_READY = "SELECTED_BRANCH_EXPLORATION_READY"

ROLE_NORMAL = "NORMAL"
ROLE_JUNCTION_GUARD = "JUNCTION_GUARD"
ROLE_FRONTIER_SHEPHERD = "FRONTIER_SHEPHERD"

BRANCH_GUI_COLORS = {
    "LIDAR_BRANCH_00": (171, 63, 204),
    "LIDAR_BRANCH_01": (35, 112, 238),
    "LIDAR_BRANCH_02": (242, 126, 32),
}
ANCHOR_GUI_COLOR = (0, 200, 230)
PARENT_GUI_COLOR = (140, 148, 160)
GUARD_SLOT_GUI_COLOR = (255, 215, 60)
GUARD_ROBOT_GUI_COLOR = (210, 45, 65)
FRONTIER_GUI_COLOR = (255, 145, 30)
SELECTED_GUI_COLOR = (0, 200, 90)


@dataclass(frozen=True)
class LidarBranchCandidate:
    """A Branch identity derived only from an Anchor-local Opening group."""

    uid: str
    start_angle_deg: float
    end_angle_deg: float
    center_angle_deg: float
    angular_width_deg: float
    confidence: float
    local_direction_x: float
    local_direction_y: float
    visit_state: str = "UNVISITED"


@dataclass(frozen=True)
class ShepherdElectionRecord:
    branch_uid: str
    fixture_adapter: str
    candidate_count: int
    required_count: int
    selected_ids: tuple[int, ...]
    election_frame: int
    purpose: str = "ID_ELECTION"
    localization_disabled_after_election: bool = True


@dataclass
class PersistentOpening:
    uid: str
    is_parent: bool
    observed_count: int = 0
    last_seen_frame: int = -1
    sine_sum: float = 0.0
    cosine_sum: float = 0.0
    start_angles: list[float] = field(default_factory=list)
    end_angles: list[float] = field(default_factory=list)
    widths: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)

    @property
    def center_angle_deg(self) -> float:
        return math.degrees(math.atan2(self.sine_sum, self.cosine_sum))

    def update(self, opening: dict[str, float], frame: int) -> None:
        center = float(opening["center_angle"])
        radians = math.radians(center)
        self.sine_sum += math.sin(radians)
        self.cosine_sum += math.cos(radians)
        self.start_angles.append(float(opening["start_angle"]))
        self.end_angles.append(float(opening["end_angle"]))
        self.widths.append(float(opening["width_deg"]))
        self.confidences.append(float(opening["confidence"]))
        self.observed_count += 1
        self.last_seen_frame = frame

    def persistence(self, sample_count: int) -> float:
        return self.observed_count / max(sample_count, 1)

    def candidate(self) -> LidarBranchCandidate:
        center = self.center_angle_deg
        radians = math.radians(center)
        return LidarBranchCandidate(
            uid=self.uid,
            start_angle_deg=float(np.median(self.start_angles)),
            end_angle_deg=float(np.median(self.end_angles)),
            center_angle_deg=center,
            angular_width_deg=float(np.median(self.widths)),
            confidence=float(np.mean(self.confidences)),
            local_direction_x=math.cos(radians),
            local_direction_y=math.sin(radians),
        )


@dataclass
class LocalGuardFrame:
    branch: LidarBranchCandidate
    anchor_position: np.ndarray
    mouth_origin: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    observed_mouth_width: float
    usable_half_width: float
    axial_inset: float
    lateral_offsets: tuple[float, ...]
    slots: tuple[np.ndarray, ...]
    elected_ids: tuple[int, ...]
    slot_by_robot_id: dict[int, np.ndarray]
    formation_frame: int | None = None
    ready_dwell: float = 0.0
    robot_seconds: float = 0.0
    max_penetration: float = 0.0
    preformation_leakage_count: int = 0
    frontier_depth: float | None = None
    normal_support_front: float = 0.0
    latest_metrics: dict[str, Any] = field(default_factory=dict)

    def serializable(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch.uid,
            "center_angle_deg": self.branch.center_angle_deg,
            "observed_mouth_width": self.observed_mouth_width,
            "mouth_origin": self.mouth_origin.tolist(),
            "tangent": self.tangent.tolist(),
            "normal": self.normal.tolist(),
            "usable_half_width": self.usable_half_width,
            "axial_inset": self.axial_inset,
            "lateral_offsets": list(self.lateral_offsets),
            "slots": [slot.tolist() for slot in self.slots],
            "elected_ids": list(self.elected_ids),
            "slot_by_robot_id": {
                str(robot_id): slot.tolist()
                for robot_id, slot in self.slot_by_robot_id.items()
            },
            "formation_frame": self.formation_frame,
            "frontier_depth": self.frontier_depth,
            "normal_support_front": self.normal_support_front,
            "latest_metrics": self.latest_metrics,
            "protected_robot_seconds": self.robot_seconds,
            "protected_max_axial_penetration": self.max_penetration,
        }


@dataclass
class LocalWallFit:
    side: str
    angle_band: tuple[float, float]
    points: np.ndarray
    inlier_points: np.ndarray
    point_on_line: np.ndarray | None
    direction: np.ndarray | None
    direction_angle_deg: float | None
    residual: float | None
    span: float
    range_min: float | None
    range_max: float | None
    valid: bool
    invalid_reason: str

    def serializable(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "angle_band": list(self.angle_band),
            "sample_count": int(len(self.points)),
            "inlier_count": int(len(self.inlier_points)),
            "points": self.points.tolist(),
            "inlier_points": self.inlier_points.tolist(),
            "point_on_line": (
                None if self.point_on_line is None
                else self.point_on_line.tolist()
            ),
            "direction": (
                None if self.direction is None else self.direction.tolist()
            ),
            "direction_angle_deg": self.direction_angle_deg,
            "residual": self.residual,
            "span": self.span,
            "range_min": self.range_min,
            "range_max": self.range_max,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
        }


@dataclass
class LocalBranchGeometryShadow:
    branch: LidarBranchCandidate
    left_fit: LocalWallFit
    right_fit: LocalWallFit
    axis: np.ndarray | None
    normal: np.ndarray | None
    estimated_axis_angle_deg: float | None
    axis_difference_deg: float | None
    parallel_error_deg: float | None
    confidence: float
    physical_width: float | None
    usable_width: float | None
    left_normal_coord: float | None
    right_normal_coord: float | None
    mouth_origin: np.ndarray | None
    slots: tuple[np.ndarray, ...]
    lateral_offsets: tuple[float, ...]
    geometry_metrics: dict[str, Any]
    evaluation_only: dict[str, Any] = field(default_factory=dict)

    def serializable(self) -> dict[str, Any]:
        return {
            "branch": asdict(self.branch),
            "left_fit": self.left_fit.serializable(),
            "right_fit": self.right_fit.serializable(),
            "axis": None if self.axis is None else self.axis.tolist(),
            "normal": None if self.normal is None else self.normal.tolist(),
            "estimated_axis_angle_deg": self.estimated_axis_angle_deg,
            "axis_difference_deg": self.axis_difference_deg,
            "parallel_error_deg": self.parallel_error_deg,
            "confidence": self.confidence,
            "physical_width": self.physical_width,
            "usable_width": self.usable_width,
            "left_normal_coord": self.left_normal_coord,
            "right_normal_coord": self.right_normal_coord,
            "mouth_origin": (
                None if self.mouth_origin is None
                else self.mouth_origin.tolist()
            ),
            "slots": [slot.tolist() for slot in self.slots],
            "lateral_offsets": list(self.lateral_offsets),
            "geometry_metrics": self.geometry_metrics,
            "evaluation_only": self.evaluation_only,
        }


@dataclass(frozen=True)
class AuditRow:
    function: str
    purpose: str
    localization_gt_dependency: str
    action: str


AUDIT_ROWS = (
    AuditRow("shepherd_candidates", "capture-region candidates", "YES: region, pose, progress", "PARTIAL_REUSE"),
    AuditRow("capture_region_ready_for_shepherd", "candidate quorum", "YES: capture rectangle", "PARTIAL_REUSE"),
    AuditRow("adaptive_shepherd_count", "width-scaled count", "YES: fixture corridor_width constant", "PARTIAL_REUSE"),
    AuditRow("assign_shepherd_slots", "deterministic nearest-slot auction", "YES: pose-to-global-slot distance", "DO_NOT_REUSE"),
    AuditRow("select_adaptive_shepherds", "elect and assign global anchors", "YES: build_shepherd_slots/global targets", "DO_NOT_REUSE"),
    AuditRow("build_shepherd_slots", "legacy Shepherd targets", "YES: BRANCH_DIRECTIONS/entrance/width", "DO_NOT_REUSE"),
    AuditRow("descriptor_local_basis", "observed Branch t/n basis", "NO fixture dependency in function", "REUSE"),
    AuditRow("branch_local_coordinates", "world-to-observed local frame", "NO fixture dependency in function", "REUSE"),
    AuditRow("local_coordinates_to_world", "observed local-to-world frame", "NO fixture dependency in function", "REUSE"),
    AuditRow("build_local_junction_guard_slots", "observed-mouth Guard row", "LOCAL descriptor; walkability adapter optional", "REUSE"),
    AuditRow("build_frontier_line_local_slots", "observed moving frontier row", "LOCAL descriptor; walkability adapter", "REUSE"),
    AuditRow("frontier_row_snapshot", "local physical row evidence", "LOCAL descriptor/robot observations", "REUSE"),
    AuditRow("junction_guards_formed", "physical Guard readiness", "mostly LOCAL; callers retain fixture keys", "PARTIAL_REUSE"),
    AuditRow("promote_existing_frontier_line", "frontier-to-piston handoff", "LOCAL descriptor/contact depth", "REUSE"),
    AuditRow("get_robot_region", "fixture region classifier", "YES: global rectangles", "PARTIAL_REUSE: ELECTION ONLY"),
    AuditRow("branch_depth_from_junction", "fixture axial depth", "YES: center/branch label", "DO_NOT_REUSE"),
    AuditRow("get_branch_tip_target", "fixture terminal target", "YES: branch terminal coordinate", "DO_NOT_REUSE"),
    AuditRow("DeadEndInferenceTracker.update", "contact/stall/density dead end", "LOCAL evidence", "REUSE"),
    AuditRow("update_dead_end_saturation", "packed/stalled dwell", "LOCAL physical evidence", "REUSE"),
    AuditRow("normal_backtracking_metrics", "reverse-flow evidence", "MIXED: legacy branch classifier", "PARTIAL_REUSE"),
    AuditRow("FLOW_BACKTRACK transition", "branch clear/Junction return", "YES: get_robot_region", "DO_NOT_REUSE"),
)


PHASE_STATUS = {
    "Adaptive LiDAR": "PASS",
    "Junction detection": "PASS",
    "Anchor fixation": "PASS",
    "Opening detection": "PASS",
    "Branch candidate creation": "PASS",
    "Branch selection": "BLOCKED",
    "Initial 3-Branch Guard election": "BLOCKED",
    "Shepherd localization trigger": "PASS",
    "Shepherd candidates": "BLOCKED",
    "Shepherd ID election": "BLOCKED",
    "Guard IDs elected-only": "BLOCKED",
    "Post-election Guard localization": "BLOCKED",
    "LIDAR_BRANCH_00 Guard row": "BLOCKED",
    "LIDAR_BRANCH_01 Guard row": "BLOCKED",
    "LIDAR_BRANCH_02 Guard row": "BLOCKED",
    "Guard settled": "BLOCKED",
    "Guard coverage": "BLOCKED",
    "Guard edge gaps": "BLOCKED",
    "Guard internal gaps": "BLOCKED",
    "Unselected Branch leakage": "BLOCKED",
    "All Branch Guards ready": "BLOCKED",
    "Selected Guard -> Frontier": "BLOCKED",
    "Selected Branch flow": "BLOCKED",
    "Selected Branch exploration": "BLOCKED",
    "Dead-end detection": "BLOCKED",
    "Shepherd Backtracking": "BLOCKED",
    "Junction return": "BLOCKED",
    "Next Branch": "BLOCKED",
    "All Branches visited": "BLOCKED",
    "DONE": "BLOCKED",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def exp_adaptive_threshold_01_configuration() -> Iterator[None]:
    """Apply 5% only inside this process without editing the source module.

    The imported implementation currently exposes its margin as a module
    constant and binds it as a helper default.  Both are temporarily replaced
    so the baseline is exactly Tmin=1.05*W.  The formula and detector are not
    reimplemented, and the original objects are restored on exit.
    """

    original_ratio = adaptive.ADAPTIVE_W_MARGIN_RATIO
    original_interval = adaptive.compute_adaptive_safe_threshold_interval

    def baseline_interval(
        adaptive_worst_wall_range: float,
        max_range: float,
        tau: float,
        margin_ratio: float = BASELINE_MARGIN_RATIO,
    ) -> tuple[float, float, bool]:
        return original_interval(
            adaptive_worst_wall_range,
            max_range,
            tau,
            margin_ratio=margin_ratio,
        )

    adaptive.ADAPTIVE_W_MARGIN_RATIO = BASELINE_MARGIN_RATIO
    adaptive.compute_adaptive_safe_threshold_interval = baseline_interval
    try:
        yield
    finally:
        adaptive.compute_adaptive_safe_threshold_interval = original_interval
        adaptive.ADAPTIVE_W_MARGIN_RATIO = original_ratio


def circular_angle_error_deg(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def incoming_parent_angle_local_deg(snapshot: adaptive.AdaptiveSnapshot) -> float:
    """Return the rear direction from the proprioceptive ingress heading.

    LiDAR angle zero is the robot's body-forward direction, so the corridor it
    arrived through is exactly 180 degrees in the Anchor-local scan frame.
    Position, map labels, and Junction coordinates are not consulted.
    """

    del snapshot
    return -180.0


class AnchorOpeningTracker:
    """Associate fixed-Anchor openings using only bounded local scan history."""

    def __init__(self) -> None:
        self.sample_count = 0
        self.parent = PersistentOpening("PARENT_00", True)
        self.children: list[PersistentOpening] = []
        self.rows: list[dict[str, Any]] = []

    @property
    def complete(self) -> bool:
        return self.sample_count >= ANCHOR_BRANCH_OBSERVATION_SAMPLES

    def observe(self, snapshot: adaptive.AdaptiveSnapshot) -> None:
        if self.complete:
            return
        openings = dfs_pre_merge_opening_observations(snapshot)
        if not openings:
            return
        parent_angle = incoming_parent_angle_local_deg(snapshot)
        parent_index = min(
            range(len(openings)),
            key=lambda index: circular_angle_error_deg(
                float(openings[index]["center_angle"]), parent_angle
            ),
        )
        self.sample_count += 1
        self.parent.update(openings[parent_index], snapshot.physics_frame)
        outgoing_angles: list[float] = []
        for index, opening in enumerate(openings):
            if index == parent_index:
                continue
            center = float(opening["center_angle"])
            outgoing_angles.append(center)
            matches = [
                item for item in self.children
                if circular_angle_error_deg(
                    center, item.center_angle_deg
                ) <= BRANCH_ASSOCIATION_TOLERANCE_DEG
            ]
            if matches:
                track = min(
                    matches,
                    key=lambda item: circular_angle_error_deg(
                        center, item.center_angle_deg
                    ),
                )
                error = circular_angle_error_deg(center, track.center_angle_deg)
            else:
                track = PersistentOpening(
                    f"LIDAR_BRANCH_{len(self.children):02d}", False
                )
                self.children.append(track)
                error = 0.0
            track.update(opening, snapshot.physics_frame)
            persistence = track.persistence(self.sample_count)
            row = {
                "frame": snapshot.physics_frame,
                "opening_center": center,
                "matched_branch_id": track.uid,
                "angular_error": error,
                "observation_count": track.observed_count,
                "persistence": persistence,
            }
            self.rows.append(row)
            print(
                "[BranchAssociation] "
                f"frame={snapshot.physics_frame} opening_center={center:.1f} "
                f"matched_branch_id={track.uid} angular_error={error:.2f} "
                f"observation_count={track.observed_count} "
                f"persistence={persistence:.3f}"
            )
        print(
            "[AnchorBranchObservation] "
            f"frame={snapshot.physics_frame} "
            f"raw_opening_count={len(snapshot.opening_groups)} "
            f"parent_angle={openings[parent_index]['center_angle']:.1f} "
            f"outgoing_angles={[round(value, 1) for value in outgoing_angles]} "
            f"persistent_branch_count={len(self.children)}"
        )

    def confirmed_children(self) -> tuple[LidarBranchCandidate, ...]:
        minimum_width = float(
            adaptive.FROZEN_PARAMETERS["min_opening_width_deg"]
        )
        confirmed = [
            item.candidate()
            for item in self.children
            if item.observed_count >= BRANCH_MIN_OBSERVATIONS
            and float(np.median(item.widths)) >= minimum_width
        ]
        return tuple(sorted(confirmed, key=lambda item: item.center_angle_deg))

    def summary(self) -> dict[str, Any]:
        children = self.confirmed_children()
        return {
            "observation_samples": self.sample_count,
            "association_tolerance_deg": BRANCH_ASSOCIATION_TOLERANCE_DEG,
            "minimum_observations": BRANCH_MIN_OBSERVATIONS,
            "parent": {
                "uid": self.parent.uid,
                "center_angle_deg": self.parent.center_angle_deg,
                "observed_count": self.parent.observed_count,
                "persistence": self.parent.persistence(self.sample_count),
            },
            "children": [
                {
                    **asdict(candidate),
                    "observed_count": next(
                        item.observed_count
                        for item in self.children if item.uid == candidate.uid
                    ),
                    "persistence": next(
                        item.persistence(self.sample_count)
                        for item in self.children if item.uid == candidate.uid
                    ),
                }
                for candidate in children
            ],
        }


def dfs_pre_merge_opening_observations(
    snapshot: adaptive.AdaptiveSnapshot,
) -> list[dict[str, float]]:
    """Expose pre-merge local support components to DFS, read-only.

    Junction confirmation continues to consume the detector's unchanged
    post-merge/refined groups.  DFS needs edge multiplicity, so it preserves
    distinct connected support components before the detector's short-gap
    robustness merge.  Threshold, smoothing, and minimum width are imported
    unchanged from the Adaptive detector.
    """

    angles, _, angular_steps = adaptive._validate_circular_scan(
        snapshot.angles_deg, snapshot.raw_ranges
    )
    smoothed = np.asarray(snapshot.smoothed_ranges, dtype=float)
    support = smoothed >= float(snapshot.adaptive_selected_threshold)
    minimum_width = float(
        adaptive.FROZEN_PARAMETERS["min_opening_width_deg"]
    )
    observations = []
    for run in adaptive._circular_runs(support, value=True):
        width = adaptive._run_width_deg(run, angular_steps)
        if width < minimum_width or width >= 359.0:
            continue
        start = float(angles[int(run[0])])
        end = float(angles[int(run[-1])])
        center = float(adaptive._normalize_angles(start + width / 2.0))
        observations.append({
            "start_angle": start,
            "end_angle": end,
            "center_angle": center,
            "width_deg": float(width),
            "mean_range_m": float(np.mean(smoothed[run])),
            "peak_range_m": float(np.max(smoothed[run])),
            "confidence": float(np.mean(support[run])),
            "start_refined": 0.0,
            "end_refined": 0.0,
        })
    observations.sort(key=lambda item: item["center_angle"])
    return observations


def select_branch_from_lidar_order(
    candidates: Sequence[LidarBranchCandidate],
) -> LidarBranchCandidate | None:
    """Select the smallest local turn; no fixture label enters policy."""

    return min(
        candidates,
        key=lambda item: (
            abs(item.center_angle_deg), item.center_angle_deg, item.uid
        ),
        default=None,
    )


def _component_rows(
    mask: np.ndarray,
    angles: np.ndarray,
    raw: np.ndarray,
    smoothed: np.ndarray,
    angular_steps: np.ndarray,
    stage: str,
) -> list[dict[str, Any]]:
    rows = []
    for index, run in enumerate(adaptive._circular_runs(mask, value=True)):
        rows.append({
            "stage": stage,
            "component_id": index,
            "start_angle_deg": float(angles[int(run[0])]),
            "end_angle_deg": float(angles[int(run[-1])]),
            "angular_width_deg": adaptive._run_width_deg(run, angular_steps),
            "supporting_ray_count": int(len(run)),
            "raw_min": float(np.min(raw[run])),
            "raw_mean": float(np.mean(raw[run])),
            "raw_max": float(np.max(raw[run])),
            "smoothed_min": float(np.min(smoothed[run])),
            "smoothed_mean": float(np.mean(smoothed[run])),
            "smoothed_max": float(np.max(smoothed[run])),
        })
    return rows


def opening_pipeline_diagnostic(
    snapshot: adaptive.AdaptiveSnapshot,
    geometry: adaptive.GeometryCase,
) -> dict[str, Any]:
    """Audit detector stages without changing any detector decision."""

    angles, raw, angular_steps = adaptive._validate_circular_scan(
        snapshot.angles_deg, snapshot.raw_ranges
    )
    smoothed = np.asarray(snapshot.smoothed_ranges, dtype=float)
    threshold = float(snapshot.adaptive_selected_threshold)
    pre_merge = smoothed >= threshold
    post_merge = adaptive._fill_short_circular_gaps(
        pre_merge.copy(),
        angular_steps,
        float(adaptive.FROZEN_PARAMETERS["merge_gap_deg"]),
    )
    pre_rows = _component_rows(
        pre_merge, angles, raw, smoothed, angular_steps, "PRE_MERGE"
    )
    post_rows = _component_rows(
        post_merge, angles, raw, smoothed, angular_steps, "POST_MERGE"
    )
    minimum_width = float(
        adaptive.FROZEN_PARAMETERS["min_opening_width_deg"]
    )
    width_kept = [
        row for row in post_rows
        if row["angular_width_deg"] >= minimum_width
    ]
    final_rows = []
    for index, opening in enumerate(snapshot.opening_groups):
        center = float(opening["center_angle"])
        nearest = int(np.argmin(np.abs(
            (angles - center + 180.0) % 360.0 - 180.0
        )))
        final_rows.append({
            "stage": "FINAL_REFINED",
            "component_id": index,
            **dict(opening),
            "center_raw_range": float(raw[nearest]),
            "center_smoothed_range": float(smoothed[nearest]),
            "center_support": bool(post_merge[nearest]),
        })

    # Evaluation-only fixture axes diagnose observability; these values never
    # flow back into persistence, selection, control, or detector decisions.
    expected_sector_rows = []
    for branch_index, spec in enumerate(geometry.branches):
        world_standard_deg = 90.0 - float(spec.angle_deg)
        local_angle = adaptive._normalize_angles(
            world_standard_deg - snapshot.lidar_yaw_deg
        )
        nearest = int(np.argmin(np.abs(
            (angles - local_angle + 180.0) % 360.0 - 180.0
        )))
        sector_mask = np.abs(
            (angles - local_angle + 180.0) % 360.0 - 180.0
        ) <= 15.0
        expected_sector_rows.append({
            "evaluation_only_branch_index": branch_index,
            "local_axis_angle_deg": float(local_angle),
            "axis_raw_range": float(raw[nearest]),
            "axis_smoothed_range": float(smoothed[nearest]),
            "axis_support": bool(post_merge[nearest]),
            "sector_raw_min": float(np.min(raw[sector_mask])),
            "sector_raw_mean": float(np.mean(raw[sector_mask])),
            "sector_raw_max": float(np.max(raw[sector_mask])),
            "sector_smoothed_min": float(np.min(smoothed[sector_mask])),
            "sector_smoothed_mean": float(np.mean(smoothed[sector_mask])),
            "sector_smoothed_max": float(np.max(smoothed[sector_mask])),
            "sector_support_rays": int(np.count_nonzero(post_merge[sector_mask])),
            "sector_ray_count": int(np.count_nonzero(sector_mask)),
            "axis_hits_physical_wall": bool(raw[nearest] < adaptive.LIDAR_MAX_RANGE),
        })

    return {
        "frame": snapshot.physics_frame,
        "selected_threshold": threshold,
        "pre_merge_component_count": len(pre_rows),
        "post_merge_component_count": len(post_rows),
        "post_width_filter_component_count": len(width_kept),
        "final_refined_component_count": len(final_rows),
        "components": pre_rows + post_rows + final_rows,
        "expected_sectors_evaluation_only": expected_sector_rows,
        "fixed_anchor_scan_is_static_wall_raycast": True,
    }


def selected_branch_world_direction_for_physics(
    branch: LidarBranchCandidate,
    lidar_yaw_deg: float,
) -> np.ndarray:
    """Rotate Anchor-local LiDAR angle into the simulator physics basis."""

    radians = math.radians(lidar_yaw_deg + branch.center_angle_deg)
    return np.array([math.cos(radians), math.sin(radians)], dtype=float)


class LocalBranchEntryController:
    """Weak SPH-compatible directional bias using an Anchor-local frame only."""

    def __init__(self) -> None:
        self.active = False
        self.branch: LidarBranchCandidate | None = None
        self.anchor_position: np.ndarray | None = None
        self.world_direction = np.zeros(2)
        self.corridor_width = adaptive.BASELINE_CORRIDOR_WIDTH
        self.excluded_robot_ids: set[int] = set()
        self.timeline: list[dict[str, Any]] = []

    def activate(
        self,
        branch: LidarBranchCandidate,
        snapshot: adaptive.AdaptiveSnapshot,
    ) -> None:
        self.active = True
        self.branch = branch
        self.anchor_position = snapshot.leader_position.copy()
        self.world_direction = selected_branch_world_direction_for_physics(
            branch, snapshot.lidar_yaw_deg
        )
        self.corridor_width = (
            snapshot.estimated_corridor_width
            or adaptive.BASELINE_CORRIDOR_WIDTH
        )
        print(
            "[BranchSelected] "
            f"branch_id={branch.uid} center_angle={branch.center_angle_deg:.2f} "
            "selection_rule=MIN_ABS_ANCHOR_LOCAL_TURN"
        )

    def activate_from_guard_frame(
        self,
        branch: LidarBranchCandidate,
        frame: LocalGuardFrame,
        excluded_robot_ids: set[int],
    ) -> None:
        """Open the selected local physical frame after all Guards are ready."""
        self.active = True
        self.branch = branch
        self.anchor_position = frame.mouth_origin.copy()
        self.world_direction = frame.tangent.copy()
        self.corridor_width = frame.observed_mouth_width
        self.excluded_robot_ids = set(excluded_robot_ids)

    def apply_after_physics_step(self, world: adaptive.SimulatorWorld) -> None:
        if not self.active or self.anchor_position is None:
            return
        normal = np.array([-self.world_direction[1], self.world_direction[0]])
        entry_acceleration = (
            adaptive.LOCAL_FORWARD_DRIVE_FORCE
            * adaptive.LOCAL_FOLLOWER_DRIVE_WEIGHT
        )
        for robot in world.robots:
            if (
                robot.robot_id in world.fixed_robot_ids
                or robot.robot_id in self.excluded_robot_ids
            ):
                continue
            relative = robot.position - self.anchor_position
            axial = float(np.dot(relative, self.world_direction))
            lateral = abs(float(np.dot(relative, normal)))
            # Existing physical scales define the local recruitment cohort.
            if axial < -self.corridor_width or lateral > self.corridor_width:
                continue
            robot.velocity += (
                self.world_direction * entry_acceleration * adaptive.DT
            )
            robot.velocity = adaptive._limit(robot.velocity, adaptive.MAX_SPEED)

    def sample(self, world: adaptive.SimulatorWorld, frame: int) -> dict[str, Any]:
        if not self.active or self.anchor_position is None or self.branch is None:
            return {
                "frame": frame,
                "branch_id": "",
                "entry_robot_count": 0,
                "front_axial_progress": 0.0,
                "mean_axial_progress": 0.0,
                "mean_axial_velocity": 0.0,
                "lateral_spread": 0.0,
            }
        normal = np.array([-self.world_direction[1], self.world_direction[0]])
        rows = []
        for robot in world.robots:
            if (
                robot.robot_id in world.fixed_robot_ids
                or robot.robot_id in self.excluded_robot_ids
            ):
                continue
            relative = robot.position - self.anchor_position
            rows.append((
                float(np.dot(relative, self.world_direction)),
                float(np.dot(relative, normal)),
                float(np.dot(robot.observed_velocity, self.world_direction)),
            ))
        entered = [
            row for row in rows
            if row[0] > 2.0 * adaptive.ROBOT_RADIUS
            and abs(row[1]) <= self.corridor_width / 2.0
        ]
        result = {
            "frame": frame,
            "branch_id": self.branch.uid,
            "entry_robot_count": len(entered),
            "front_axial_progress": max((row[0] for row in rows), default=0.0),
            "mean_axial_progress": float(np.mean([row[0] for row in entered])) if entered else 0.0,
            "mean_axial_velocity": float(np.mean([row[2] for row in entered])) if entered else 0.0,
            "lateral_spread": float(np.ptp([row[1] for row in entered])) if entered else 0.0,
        }
        self.timeline.append(result)
        print(
            "[BranchEntryControl] "
            f"frame={frame} branch_id={self.branch.uid} "
            f"local_direction=({self.branch.local_direction_x:.3f},"
            f"{self.branch.local_direction_y:.3f}) "
            f"entry_robot_count={result['entry_robot_count']} "
            f"front_axial_progress={result['front_axial_progress']:.2f} "
            f"mean_axial_progress={result['mean_axial_progress']:.2f}"
        )
        return result


def lidar_ray_to_anchor_local_point(
    theta_deg: float,
    measured_range: float,
) -> np.ndarray:
    """Convert one body/LiDAR-relative ray to Anchor-local Cartesian."""

    radians = math.radians(theta_deg)
    return np.array([
        measured_range * math.cos(radians),
        measured_range * math.sin(radians),
    ], dtype=float)


def _undirected_angle_error_deg(left: float, right: float) -> float:
    difference = abs(float(adaptive._normalize_angles(left - right)))
    return min(difference, 180.0 - difference)


def _fit_local_wall_line(
    side: str,
    boundary_angle_deg: float,
    snapshot: adaptive.AdaptiveSnapshot,
) -> LocalWallFit:
    """Fit one boundary-adjacent wall using ordered local LiDAR returns."""

    angles = np.asarray(snapshot.angles_deg, dtype=float)
    ranges = np.asarray(snapshot.raw_ranges, dtype=float)
    angular_distance = np.abs(
        (angles - boundary_angle_deg + 180.0) % 360.0 - 180.0
    )
    mask = (
        angular_distance <= LOCAL_WALL_BAND_DEG
    ) & (
        ranges < adaptive.LIDAR_MAX_RANGE - adaptive.DETECTOR_EPSILON
    ) & (ranges > adaptive.DETECTOR_EPSILON)
    indices = np.flatnonzero(mask)
    points = np.asarray([
        lidar_ray_to_anchor_local_point(angles[index], ranges[index])
        for index in indices
    ], dtype=float).reshape(-1, 2)
    angle_band = (
        float(adaptive._normalize_angles(boundary_angle_deg - LOCAL_WALL_BAND_DEG)),
        float(adaptive._normalize_angles(boundary_angle_deg + LOCAL_WALL_BAND_DEG)),
    )
    range_min = float(np.min(ranges[indices])) if len(indices) else None
    range_max = float(np.max(ranges[indices])) if len(indices) else None
    empty = np.empty((0, 2), dtype=float)
    if len(points) < 2:
        return LocalWallFit(
            side, angle_band, points, empty, None, None, None, None, 0.0,
            range_min, range_max, False, "INSUFFICIENT_WALL_RETURNS",
        )

    # Ordered adjacent-point directions expose the dominant local wall
    # segment without a RANSAC dependency.  A medoid on undirected angles is
    # robust to a short orthogonal corner segment in the same angular band.
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    usable_segments = np.flatnonzero(lengths > adaptive.DETECTOR_EPSILON)
    if not len(usable_segments):
        return LocalWallFit(
            side, angle_band, points, empty, None, None, None, None, 0.0,
            range_min, range_max, False, "ZERO_WALL_SPAN",
        )
    segment_angles = np.asarray([
        math.degrees(math.atan2(deltas[index, 1], deltas[index, 0]))
        for index in usable_segments
    ])
    medoid_index = min(
        range(len(segment_angles)),
        key=lambda index: sum(
            _undirected_angle_error_deg(
                segment_angles[index], candidate
            ) for candidate in segment_angles
        ),
    )
    medoid_angle = float(segment_angles[medoid_index])
    accepted_segments = [
        int(segment_index)
        for segment_index, angle in zip(usable_segments, segment_angles)
        if _undirected_angle_error_deg(angle, medoid_angle)
        <= LOCAL_WALL_DIRECTION_TOLERANCE_DEG
    ]
    point_indices = sorted({
        point_index
        for segment_index in accepted_segments
        for point_index in (segment_index, segment_index + 1)
    })
    inliers = points[point_indices] if point_indices else empty
    if len(inliers) < 2:
        return LocalWallFit(
            side, angle_band, points, inliers, None, None, None, None, 0.0,
            range_min, range_max, False, "INSUFFICIENT_DIRECTION_SUPPORT",
        )

    point_on_line = np.mean(inliers, axis=0)
    centered = inliers - point_on_line
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    direction = axes[0]
    direction /= max(float(np.linalg.norm(direction)), adaptive.EPSILON)
    normal = np.array([-direction[1], direction[0]])
    residual = float(np.sqrt(np.mean((centered @ normal) ** 2)))
    axial = centered @ direction
    span = float(np.ptp(axial))
    median_step = float(np.median(np.diff(angles)))
    minimum_samples = int(math.ceil(
        float(adaptive.FROZEN_PARAMETERS["merge_gap_deg"])
        / max(median_step, adaptive.EPSILON)
    )) + 1
    valid = len(inliers) >= minimum_samples and span >= adaptive.SAFE_RADIUS
    invalid_reason = "" if valid else (
        "INSUFFICIENT_DIRECTION_SUPPORT"
        if len(inliers) < minimum_samples
        else "CORNER_ONLY_SPAN_BELOW_SAFE_RADIUS"
    )
    return LocalWallFit(
        side=side,
        angle_band=angle_band,
        points=points,
        inlier_points=inliers,
        point_on_line=point_on_line,
        direction=direction,
        direction_angle_deg=float(math.degrees(math.atan2(
            direction[1], direction[0]
        ))),
        residual=residual,
        span=span,
        range_min=range_min,
        range_max=range_max,
        valid=valid,
        invalid_reason=invalid_reason,
    )


def _estimate_local_branch_geometry(
    branch: LidarBranchCandidate,
    snapshot: adaptive.AdaptiveSnapshot,
    guard_count: int,
) -> LocalBranchGeometryShadow:
    """Estimate geometry exclusively from the Anchor-local LiDAR scan."""

    left = _fit_local_wall_line(
        "LEFT", branch.start_angle_deg, snapshot
    )
    right = _fit_local_wall_line(
        "RIGHT", branch.end_angle_deg, snapshot
    )
    outward = np.array([
        math.cos(math.radians(branch.center_angle_deg)),
        math.sin(math.radians(branch.center_angle_deg)),
    ])
    aligned_directions = []
    for fit in (left, right):
        if not fit.valid or fit.direction is None:
            continue
        direction = fit.direction.copy()
        if float(np.dot(direction, outward)) < 0.0:
            direction *= -1.0
        aligned_directions.append(direction)
    axis = None
    if aligned_directions:
        combined = np.sum(aligned_directions, axis=0)
        if float(np.linalg.norm(combined)) > adaptive.EPSILON:
            axis = combined / float(np.linalg.norm(combined))
    normal = None if axis is None else np.array([-axis[1], axis[0]])
    axis_angle = (
        None if axis is None
        else float(math.degrees(math.atan2(axis[1], axis[0])))
    )
    difference = (
        None if axis_angle is None
        else circular_angle_error_deg(axis_angle, branch.center_angle_deg)
    )
    parallel_error = (
        _undirected_angle_error_deg(
            left.direction_angle_deg, right.direction_angle_deg
        )
        if left.direction_angle_deg is not None
        and right.direction_angle_deg is not None
        else None
    )
    valid_fits = [fit for fit in (left, right) if fit.valid]
    fit_quality = float(np.mean([
        1.0 / (1.0 + float(fit.residual or 0.0)
               / max(fit.span, adaptive.EPSILON))
        for fit in valid_fits
    ])) if valid_fits else 0.0
    confidence = len(valid_fits) / 2.0 * fit_quality
    if len(valid_fits) == 2 and parallel_error is not None:
        confidence *= max(0.0, 1.0 - parallel_error / 90.0)

    physical_width = None
    usable_width = None
    left_coord = None
    right_coord = None
    mouth_origin = None
    slots: tuple[np.ndarray, ...] = ()
    offsets: tuple[float, ...] = ()
    if (
        axis is not None and normal is not None
        and left.valid and right.valid
        and left.point_on_line is not None
        and right.point_on_line is not None
    ):
        left_coord = float(np.dot(left.point_on_line, normal))
        right_coord = float(np.dot(right.point_on_line, normal))
        physical_width = abs(right_coord - left_coord)
        usable_width = max(
            0.0,
            physical_width
            - 2.0 * (adaptive.ROBOT_RADIUS + FRONTIER_LINE_EDGE_CLEARANCE),
        )
        # The closest supported outward axial coordinate on each wall defines
        # a local mouth cross-section; no map entrance coordinate is queried.
        axial_edges = []
        for fit in (left, right):
            projections = fit.inlier_points @ axis
            outward_projections = projections[projections >= 0.0]
            axial_edges.append(float(np.min(
                outward_projections if len(outward_projections)
                else projections
            )))
        axial_mouth = float(np.mean(axial_edges))
        center_coord = 0.5 * (left_coord + right_coord)
        mouth_origin = axis * axial_mouth + normal * center_coord
        usable_half = usable_width * 0.5
        offsets = tuple(np.linspace(
            -usable_half, usable_half, guard_count
        ).tolist())
        row_center = mouth_origin + axis * JUNCTION_GUARD_BRANCH_INSET
        slots = tuple(row_center + normal * value for value in offsets)

    if slots:
        adjacent_gaps = [
            float(np.linalg.norm(right_slot - left_slot))
            for left_slot, right_slot in zip(slots, slots[1:])
        ]
        row_span = float(np.linalg.norm(slots[-1] - slots[0]))
        geometry_metrics = {
            "status": "CONFIRMED",
            "slot_count": len(slots),
            "row_span": row_span,
            "estimated_usable_width": usable_width,
            "theoretical_coverage_ratio": (
                row_span / max(float(usable_width), adaptive.EPSILON)
            ),
            "left_edge_gap": 0.0,
            "right_edge_gap": 0.0,
            "max_internal_slot_gap": max(adjacent_gaps, default=0.0),
        }
    else:
        geometry_metrics = {
            "status": "UNCONFIRMED",
            "slot_count": 0,
            "row_span": None,
            "estimated_usable_width": usable_width,
            "theoretical_coverage_ratio": None,
            "left_edge_gap": None,
            "right_edge_gap": None,
            "max_internal_slot_gap": None,
        }
    return LocalBranchGeometryShadow(
        branch=branch,
        left_fit=left,
        right_fit=right,
        axis=axis,
        normal=normal,
        estimated_axis_angle_deg=axis_angle,
        axis_difference_deg=difference,
        parallel_error_deg=parallel_error,
        confidence=confidence,
        physical_width=physical_width,
        usable_width=usable_width,
        left_normal_coord=left_coord,
        right_normal_coord=right_coord,
        mouth_origin=mouth_origin,
        slots=slots,
        lateral_offsets=offsets,
        geometry_metrics=geometry_metrics,
    )


def _anchor_local_rotation(lidar_yaw_deg: float) -> np.ndarray:
    radians = math.radians(lidar_yaw_deg)
    return np.array([
        [math.cos(radians), -math.sin(radians)],
        [math.sin(radians), math.cos(radians)],
    ])


def _anchor_local_to_world(
    local_point: np.ndarray,
    snapshot: adaptive.AdaptiveSnapshot,
) -> np.ndarray:
    rotation = _anchor_local_rotation(snapshot.lidar_yaw_deg)
    return snapshot.leader_position + rotation @ local_point


def _evaluate_shadow_geometry_only(
    shadow: LocalBranchGeometryShadow,
    snapshot: adaptive.AdaptiveSnapshot,
    geometry: adaptive.GeometryCase,
    discovered_branches: Sequence[LidarBranchCandidate],
) -> dict[str, Any]:
    """Synthetic-map comparison; never feeds the estimator/controller."""

    fixture = _fixture_adapter_for_local_direction(
        shadow.branch, geometry, discovered_branches
    )
    gt_world_standard_deg = 90.0 - float(fixture.angle_deg)
    gt_local_axis_deg = float(adaptive._normalize_angles(
        gt_world_standard_deg - snapshot.lidar_yaw_deg
    ))
    slot_rows = []
    for index, local_slot in enumerate(shadow.slots):
        world_slot = _anchor_local_to_world(local_slot, snapshot)
        wall_clearance = min(
            adaptive._nearest_point(world_slot, wall)[1]
            for wall in geometry.walls
        )
        slot_rows.append({
            "slot_index": index,
            "local_position": local_slot.tolist(),
            "world_position_evaluation_only": world_slot.tolist(),
            "walkable": bool(
                geometry.contains(world_slot)
                and wall_clearance >= adaptive.ROBOT_RADIUS
            ),
            "wall_clearance": float(wall_clearance),
        })
    walkable_count = sum(row["walkable"] for row in slot_rows)
    return {
        "label": "EVALUATION_ONLY",
        "gt_local_axis_angle_deg": gt_local_axis_deg,
        "estimated_axis_error_deg": (
            None if shadow.estimated_axis_angle_deg is None
            else circular_angle_error_deg(
                shadow.estimated_axis_angle_deg, gt_local_axis_deg
            )
        ),
        "gt_corridor_width": float(fixture.width),
        "estimated_width_error": (
            None if shadow.physical_width is None
            else shadow.physical_width - float(fixture.width)
        ),
        "walkable_slots": walkable_count,
        "total_slots": len(slot_rows),
        "slot_walkability_ratio": (
            walkable_count / len(slot_rows) if slot_rows else None
        ),
        "slots": slot_rows,
    }


class LocalBranchGeometryShadowManager:
    """Local LiDAR geometry producer; evaluation-only GT stays isolated."""

    def __init__(self) -> None:
        self.frame: int | None = None
        self.results: dict[str, LocalBranchGeometryShadow] = {}

    def analyze(
        self,
        snapshot: adaptive.AdaptiveSnapshot,
        branches: Sequence[LidarBranchCandidate],
        target_branches: Sequence[LidarBranchCandidate],
        elections: Sequence[ShepherdElectionRecord],
        geometry_evaluation_only: adaptive.GeometryCase,
    ) -> None:
        if self.results:
            return
        counts = {item.branch_uid: item.required_count for item in elections}
        self.frame = snapshot.physics_frame
        for branch in target_branches:
            shadow = _estimate_local_branch_geometry(
                branch, snapshot, counts.get(branch.uid, 0)
            )
            shadow.evaluation_only = _evaluate_shadow_geometry_only(
                shadow, snapshot, geometry_evaluation_only, branches
            )
            self.results[branch.uid] = shadow
            for fit in (shadow.left_fit, shadow.right_fit):
                print(
                    "[LocalWallSamples] "
                    f"branch={branch.uid} side={fit.side} "
                    f"sample_count={len(fit.points)} "
                    f"angle_band={list(fit.angle_band)} "
                    f"range_min={fit.range_min} range_max={fit.range_max}"
                )
                print(
                    "[LocalWallFit] "
                    f"branch={branch.uid} side={fit.side} "
                    f"direction_angle={fit.direction_angle_deg} "
                    f"residual={fit.residual} span={fit.span:.3f} "
                    f"valid={fit.valid} reason={fit.invalid_reason or 'NONE'}"
                )
            print(
                "[LocalBranchAxis] "
                f"branch={branch.uid} "
                f"opening_center_angle={branch.center_angle_deg:.3f} "
                f"estimated_axis_angle={shadow.estimated_axis_angle_deg} "
                f"difference_deg={shadow.axis_difference_deg} "
                f"left_wall_angle={shadow.left_fit.direction_angle_deg} "
                f"right_wall_angle={shadow.right_fit.direction_angle_deg} "
                f"parallel_error_deg={shadow.parallel_error_deg} "
                f"confidence={shadow.confidence:.6f}"
            )
            print(
                "[LocalMouthWidth] "
                f"branch={branch.uid} physical_width={shadow.physical_width} "
                f"usable_width={shadow.usable_width} "
                f"left_normal_coord={shadow.left_normal_coord} "
                f"right_normal_coord={shadow.right_normal_coord}"
            )
            print(
                "[LocalMouthOrigin] "
                f"branch={branch.uid} origin="
                f"{None if shadow.mouth_origin is None else shadow.mouth_origin.tolist()} "
                "source=LOCAL_LIDAR_WALL_TOPOLOGY"
            )
            print(
                "[ShadowGuardSlots] "
                f"branch={branch.uid} count={len(shadow.slots)} "
                f"row_center="
                f"{None if not shadow.slots else np.mean(shadow.slots, axis=0).tolist()} "
                f"axis_t={None if shadow.axis is None else shadow.axis.tolist()} "
                f"normal_n={None if shadow.normal is None else shadow.normal.tolist()} "
                f"lateral_offsets={list(shadow.lateral_offsets)}"
            )
            metrics = shadow.geometry_metrics
            evaluation = shadow.evaluation_only
            print(
                "[ShadowGuardGeometry] "
                f"branch={branch.uid} "
                f"coverage_ratio={metrics['theoretical_coverage_ratio']} "
                f"left_edge_gap={metrics['left_edge_gap']} "
                f"right_edge_gap={metrics['right_edge_gap']} "
                f"max_internal_gap={metrics['max_internal_slot_gap']} "
                f"walkable_slots={evaluation['walkable_slots']} "
                f"total_slots={evaluation['total_slots']}"
            )

    def serializable(self) -> dict[str, Any]:
        return {
            "analysis_frame": self.frame,
            "runtime_inputs": (
                "ANCHOR_LOCAL_LIDAR_ANGLES_RANGES_AND_PERSISTENT_OPENINGS"
            ),
            "controller_output": "CONFIRMED_LOCAL_FRAME_ONLY",
            "results": {
                uid: result.serializable()
                for uid, result in self.results.items()
            },
        }



def _required_local_guard_count(observed_mouth_width: float) -> int:
    usable_width = max(
        0.0,
        observed_mouth_width
        - 2.0 * (adaptive.ROBOT_RADIUS + FRONTIER_LINE_EDGE_CLEARANCE),
    )
    return int(np.clip(
        math.ceil(usable_width / JUNCTION_GUARD_COVERAGE) + 1,
        JUNCTION_GUARD_MIN_COUNT,
        JUNCTION_GUARD_MAX_COUNT,
    ))


def _elect_guard_ids_with_localization(
    robots: Sequence[adaptive.RobotState],
    branch: LidarBranchCandidate,
    geometry: adaptive.GeometryCase,
    discovered_branches: Sequence[LidarBranchCandidate],
    required_count: int,
    excluded_ids: set[int],
    frame: int,
) -> ShepherdElectionRecord:
    """LOCALIZATION EXCEPTION ONLY: elect IDs near one protected mouth."""

    fixture = _fixture_adapter_for_local_direction(
        branch, geometry, discovered_branches
    )
    radians = math.radians(fixture.angle_deg)
    direction = np.array([math.sin(radians), math.cos(radians)])
    mouth = direction * (geometry.junction_size * 0.5)
    ranked: list[tuple[int, float, float, int]] = []
    for robot in robots:
        if robot.robot_id in excluded_ids:
            continue
        distance = float(np.linalg.norm(robot.position - mouth))
        progress = float(np.dot(robot.position, direction))
        ranked.append((
            int(distance > JUNCTION_GUARD_RECRUIT_RADIUS),
            distance,
            -progress,
            robot.robot_id,
        ))
    ranked.sort()
    selected_ids = tuple(row[3] for row in ranked[:required_count])
    record = ShepherdElectionRecord(
        branch_uid=branch.uid,
        fixture_adapter=f"ANGLE_{fixture.angle_deg:+.1f}_MOUTH_CAPTURE",
        candidate_count=len(ranked),
        required_count=required_count,
        selected_ids=selected_ids,
        election_frame=frame,
        localization_disabled_after_election=(
            len(selected_ids) == required_count
        ),
    )
    print(
        "[GuardElection] "
        f"branch={branch.uid} required_count={required_count} "
        f"candidate_count={len(ranked)} selected_ids={list(selected_ids)}"
    )
    if record.localization_disabled_after_election:
        print(
            "[ShepherdElectionComplete] "
            f"protected_branch={branch.uid} "
            f"selected_ids={list(selected_ids)} "
            "localization_disabled_after_election=True"
        )
        print(
            "[GuardLocalizationClosed] "
            f"branch={branch.uid} selected_ids={list(selected_ids)} "
            "localization_disabled=True"
        )
    return record



class LocalGuardManager:
    """Three-Branch Guard lifecycle with a one-shot localization boundary."""

    def __init__(self) -> None:
        self.state = STATE_ANCHOR_FIXED
        self.branches: tuple[LidarBranchCandidate, ...] = ()
        self.selected_branch: LidarBranchCandidate | None = None
        self.protected_branches: tuple[LidarBranchCandidate, ...] = ()
        self.elections: list[ShepherdElectionRecord] = []
        self.frames: dict[str, LocalGuardFrame] = {}
        self.elected_ids: set[int] = set()
        self.guard_ids: set[int] = set()
        self.frontier_ids: set[int] = set()
        self.roles: dict[int, str] = {}
        self.guard_ids_locked = False
        self.localization_election_closed = False
        self.geometry_blockers: dict[str, str] = {}
        self.formation_rows: list[dict[str, Any]] = []
        self.leakage_rows: list[dict[str, Any]] = []
        self.selected_flow_rows: list[dict[str, Any]] = []
        self.all_ready_frame: int | None = None
        self.selection_frame: int | None = None
        self.ready_frame: int | None = None

    def elect_all(
        self,
        world: adaptive.SimulatorWorld,
        snapshot: adaptive.AdaptiveSnapshot,
        branches: Sequence[LidarBranchCandidate],
    ) -> None:
        """Localization exception: elect disjoint IDs for all Branches once."""
        if self.elections:
            return
        self.branches = tuple(branches)
        self.state = STATE_BRANCHES_VALIDATED
        observed_width = float(
            snapshot.estimated_corridor_width
            or adaptive.BASELINE_CORRIDOR_WIDTH
        )
        excluded = set(world.fixed_robot_ids)
        for branch in self.branches:
            required = _required_local_guard_count(observed_width)
            record = _elect_guard_ids_with_localization(
                world.robots,
                branch,
                world.geometry,
                self.branches,
                required,
                excluded,
                snapshot.physics_frame,
            )
            self.elections.append(record)
            excluded.update(record.selected_ids)
            print(
                "[AllBranchGuardElection] "
                f"branch={branch.uid} required={required} "
                f"candidate_count={record.candidate_count} "
                f"ids={list(record.selected_ids)}"
            )
        self.elected_ids = {
            robot_id
            for record in self.elections
            for robot_id in record.selected_ids
        }
        self.guard_ids_locked = (
            len(self.elections) == len(self.branches)
            and all(
                len(record.selected_ids) == record.required_count
                for record in self.elections
            )
            and len(self.elected_ids) == sum(
                record.required_count for record in self.elections
            )
        )
        self.localization_election_closed = True
        self.state = STATE_FORM_ALL_JUNCTION_GUARDS
        print(
            "[GuardLocalizationClosed] "
            f"branches={[item.uid for item in self.branches]} "
            "localization_after_election=DISABLED"
        )

    def build_from_local_geometry(
        self,
        world: adaptive.SimulatorWorld,
        snapshot: adaptive.AdaptiveSnapshot,
        geometry_results: dict[str, LocalBranchGeometryShadow],
    ) -> None:
        """Create rows only when local axis, mouth, width, and slots exist."""
        if self.frames or not self.guard_ids_locked:
            return
        robot_by_id = {robot.robot_id: robot for robot in world.robots}
        rotation = _anchor_local_rotation(snapshot.lidar_yaw_deg)
        election_by_uid = {item.branch_uid: item for item in self.elections}
        for branch in self.branches:
            shadow = geometry_results.get(branch.uid)
            election = election_by_uid.get(branch.uid)
            missing = []
            if shadow is None or shadow.axis is None:
                missing.append("PHYSICAL_AXIS_UNCONFIRMED")
            if shadow is None or shadow.normal is None:
                missing.append("PHYSICAL_NORMAL_UNCONFIRMED")
            if shadow is None or shadow.mouth_origin is None:
                missing.append("MOUTH_ORIGIN_UNCONFIRMED")
            if shadow is None or shadow.physical_width is None:
                missing.append("PHYSICAL_WIDTH_UNCONFIRMED")
            if shadow is None or shadow.usable_width is None:
                missing.append("USABLE_WIDTH_UNCONFIRMED")
            elif shadow.usable_width <= adaptive.EPSILON:
                missing.append("USABLE_WIDTH_NONPOSITIVE")
            if (
                shadow is not None
                and shadow.physical_width is not None
                and shadow.physical_width <= 2.0 * adaptive.ROBOT_RADIUS
            ):
                missing.append("WIDTH_BELOW_ROBOT_DIAMETER")
            if shadow is None or not shadow.slots:
                missing.append("LOCAL_SLOTS_UNCONFIRMED")
            if election is None or len(election.selected_ids) != election.required_count:
                missing.append("ELECTION_QUORUM_MISSING")
            if (
                shadow is not None
                and election is not None
                and len(shadow.slots) != len(election.selected_ids)
            ):
                missing.append("SLOT_COUNT_MISMATCH")
            if missing:
                self.geometry_blockers[branch.uid] = "+".join(missing)
                print(
                    "[GuardGeometryBlocked] "
                    f"branch={branch.uid} reason={self.geometry_blockers[branch.uid]} "
                    "gt_fallback=False"
                )
                continue

            tangent = rotation @ shadow.axis
            tangent /= max(float(np.linalg.norm(tangent)), adaptive.EPSILON)
            normal = rotation @ shadow.normal
            normal /= max(float(np.linalg.norm(normal)), adaptive.EPSILON)
            mouth_origin = _anchor_local_to_world(shadow.mouth_origin, snapshot)
            slots = tuple(
                _anchor_local_to_world(slot, snapshot)
                for slot in shadow.slots
            )
            ordered_ids = tuple(sorted(
                election.selected_ids,
                key=lambda robot_id: (
                    float(np.dot(
                        robot_by_id[robot_id].position - mouth_origin,
                        normal,
                    )),
                    robot_id,
                ),
            ))
            slot_by_robot_id = {
                robot_id: slot.copy()
                for robot_id, slot in zip(ordered_ids, slots)
            }
            frame = LocalGuardFrame(
                branch=branch,
                anchor_position=snapshot.leader_position.copy(),
                mouth_origin=mouth_origin,
                tangent=tangent,
                normal=normal,
                observed_mouth_width=float(shadow.physical_width),
                usable_half_width=float(shadow.usable_width) * 0.5,
                axial_inset=JUNCTION_GUARD_BRANCH_INSET,
                lateral_offsets=shadow.lateral_offsets,
                slots=slots,
                elected_ids=ordered_ids,
                slot_by_robot_id=slot_by_robot_id,
            )
            self.frames[branch.uid] = frame
            self.guard_ids.update(ordered_ids)
            self.roles.update({
                robot_id: ROLE_JUNCTION_GUARD
                for robot_id in ordered_ids
            })
            print(
                "[GuardLocalFrame] "
                f"branch={branch.uid} t={tangent.tolist()} n={normal.tolist()} "
                f"mouth={mouth_origin.tolist()} width={shadow.physical_width:.3f}"
            )
            print(
                "[GuardSlots] "
                f"branch={branch.uid} count={len(slots)} "
                f"slots={[slot.tolist() for slot in slots]}"
            )

    def _target_for(self, frame: LocalGuardFrame, robot_id: int) -> np.ndarray:
        if robot_id not in self.frontier_ids:
            return frame.slot_by_robot_id[robot_id]
        lateral = float(np.dot(
            frame.slot_by_robot_id[robot_id]
            - (frame.mouth_origin + frame.tangent * frame.axial_inset),
            frame.normal,
        ))
        depth = frame.frontier_depth or frame.axial_inset
        return frame.mouth_origin + frame.tangent * depth + frame.normal * lateral

    def advance_roles_before_physics(
        self,
        world: adaptive.SimulatorWorld,
    ) -> dict[int, np.ndarray]:
        """Move role robots first; caller then freezes them for NORMAL step."""
        realized_velocities: dict[int, np.ndarray] = {}
        robot_by_id = {robot.robot_id: robot for robot in world.robots}
        for frame in self.frames.values():
            for robot_id in frame.elected_ids:
                robot = robot_by_id[robot_id]
                old_position = robot.position.copy()
                target = self._target_for(frame, robot_id)
                error = target - old_position
                distance = float(np.linalg.norm(error))
                speed = (
                    FRONTIER_LINE_FORM_SPEED
                    if robot_id in self.frontier_ids
                    else JUNCTION_GUARD_MOVE_SPEED
                )
                step = min(distance, speed * adaptive.DT)
                new_position = (
                    old_position.copy()
                    if distance <= adaptive.EPSILON
                    else old_position + error / distance * step
                )
                robot.position = new_position
                realized_velocity = (
                    new_position - old_position
                ) / adaptive.DT
                realized_velocities[robot_id] = realized_velocity
                robot.observed_velocity = realized_velocity.copy()
                robot.velocity = (
                    realized_velocity.copy()
                    if robot_id in self.frontier_ids
                    else np.zeros(2)
                )
                robot.acceleration = np.zeros(2)
        return realized_velocities

    def restore_role_observations(
        self,
        world: adaptive.SimulatorWorld,
        realized_velocities: dict[int, np.ndarray],
    ) -> None:
        robot_by_id = {robot.robot_id: robot for robot in world.robots}
        for robot_id, realized_velocity in realized_velocities.items():
            robot = robot_by_id[robot_id]
            robot.observed_velocity = realized_velocity.copy()
            robot.velocity = (
                realized_velocity.copy()
                if robot_id in self.frontier_ids
                else np.zeros(2)
            )
            robot.acceleration = np.zeros(2)

    def _row_metrics(
        self,
        frame: LocalGuardFrame,
        robot_by_id: dict[int, adaptive.RobotState],
    ) -> dict[str, Any]:
        robots = [robot_by_id[item] for item in frame.elected_ids]
        targets = [self._target_for(frame, item) for item in frame.elected_ids]
        errors = [
            float(np.linalg.norm(robot.position - target))
            for robot, target in zip(robots, targets)
        ]
        settled_ratio = sum(
            error <= FRONTIER_LINE_TARGET_TOLERANCE for error in errors
        ) / max(len(errors), 1)
        row_center = frame.mouth_origin + frame.tangent * (
            frame.frontier_depth or frame.axial_inset
        )
        laterals = sorted(float(np.dot(
            robot.position - row_center,
            frame.normal,
        )) for robot in robots)
        span = laterals[-1] - laterals[0] if len(laterals) >= 2 else 0.0
        usable_width = frame.usable_half_width * 2.0
        coverage = span / max(usable_width, adaptive.EPSILON)
        left_gap = max(
            0.0, laterals[0] + frame.usable_half_width
        ) if laterals else math.inf
        right_gap = max(
            0.0, frame.usable_half_width - laterals[-1]
        ) if laterals else math.inf
        internal_gap = max(
            (right - left for left, right in zip(laterals, laterals[1:])),
            default=0.0,
        )
        ready = (
            len(robots) == len(frame.slots)
            and settled_ratio >= FRONTIER_LINE_TARGET_SETTLED_RATIO
            and coverage >= FRONTIER_LINE_MIN_SPAN_RATIO
            and left_gap <= FRONTIER_LINE_MAX_EDGE_GAP
            and right_gap <= FRONTIER_LINE_MAX_EDGE_GAP
            and internal_gap <= FRONTIER_LINE_MAX_INTERNAL_GAP
        )
        return {
            "settled_ratio": settled_ratio,
            "coverage_ratio": coverage,
            "left_edge_gap": left_gap,
            "right_edge_gap": right_gap,
            "max_internal_gap": internal_gap,
            "instant_ready": ready,
            "max_slot_error": max(errors, default=math.inf),
        }

    def _update_frontier_progress(
        self,
        world: adaptive.SimulatorWorld,
        frame: LocalGuardFrame,
        row_ready: bool,
    ) -> None:
        if frame.frontier_depth is None:
            return
        support = []
        excluded = set(self.roles) | set(world.fixed_robot_ids)
        for robot in world.robots:
            if robot.robot_id in excluded:
                continue
            relative = robot.position - frame.mouth_origin
            axial = float(np.dot(relative, frame.tangent))
            lateral = float(np.dot(relative, frame.normal))
            if axial >= 0.0 and abs(lateral) <= frame.observed_mouth_width * 0.5:
                support.append(axial)
        frame.normal_support_front = (
            float(np.quantile(support, FRONTIER_LINE_SUPPORT_QUANTILE))
            if support else 0.0
        )
        if row_ready:
            desired = max(
                frame.frontier_depth,
                frame.normal_support_front + FRONTIER_LINE_LEAD_GAP,
            )
            frame.frontier_depth = min(
                desired,
                frame.frontier_depth
                + FRONTIER_LINE_ADVANCE_SPEED * adaptive.SAMPLE_PERIOD,
            )

    def _commit_frontier(self, frame_number: int) -> None:
        selected = select_branch_from_lidar_order(self.branches)
        if selected is None:
            return
        self.selected_branch = selected
        self.selection_frame = frame_number
        self.protected_branches = tuple(
            branch for branch in self.branches if branch.uid != selected.uid
        )
        selected_frame = self.frames[selected.uid]
        selected_frame.frontier_depth = selected_frame.axial_inset
        self.frontier_ids = set(selected_frame.elected_ids)
        self.guard_ids.difference_update(self.frontier_ids)
        self.roles.update({
            robot_id: ROLE_FRONTIER_SHEPHERD
            for robot_id in self.frontier_ids
        })
        self.state = STATE_FRONTIER_SELECTED
        print(
            f"[BranchSelected] branch={selected.uid} "
            "selection_rule=PERSISTENT_LOCAL_ORDER"
        )
        print(
            "[GuardToFrontier] "
            f"branch={selected.uid} same_ids={list(selected_frame.elected_ids)} "
            "new_election=False"
        )

    def sample(
        self,
        world: adaptive.SimulatorWorld,
        frame_number: int,
        selected_flow: dict[str, Any],
    ) -> None:
        if not self.frames:
            return
        robot_by_id = {robot.robot_id: robot for robot in world.robots}
        all_ready = len(self.frames) == len(self.branches)
        selected_uid = self.selected_branch.uid if self.selected_branch else None
        for branch_uid, frame in self.frames.items():
            metrics = self._row_metrics(frame, robot_by_id)
            frame.ready_dwell = (
                frame.ready_dwell + adaptive.SAMPLE_PERIOD
                if metrics["instant_ready"] else 0.0
            )
            row_ready = frame.ready_dwell >= JUNCTION_GUARD_FORM_DWELL
            all_ready = all_ready and row_ready
            if branch_uid == selected_uid:
                self._update_frontier_progress(world, frame, row_ready)
            normal_rows = []
            for robot in world.robots:
                if robot.robot_id in self.roles or robot.robot_id in world.fixed_robot_ids:
                    continue
                relative = robot.position - frame.mouth_origin
                axial = float(np.dot(relative, frame.tangent))
                lateral = float(np.dot(relative, frame.normal))
                if axial > frame.axial_inset and abs(lateral) <= frame.observed_mouth_width * 0.5:
                    normal_rows.append((axial, robot.robot_id))
            leakage_count = len(normal_rows)
            penetration = max((item[0] for item in normal_rows), default=0.0)
            frame.robot_seconds += leakage_count * adaptive.SAMPLE_PERIOD
            frame.max_penetration = max(frame.max_penetration, penetration)
            if frame.formation_frame is None and row_ready:
                frame.formation_frame = frame_number
            row = {
                "frame": frame_number,
                "branch": branch_uid,
                **metrics,
                "row_ready": row_ready,
            }
            frame.latest_metrics = row
            self.formation_rows.append(row)
            self.leakage_rows.append({
                "frame": frame_number,
                "branch": branch_uid,
                "entry_robot_count": leakage_count,
                "max_axial_penetration": penetration,
            })
            print(
                "[GuardFormation] "
                f"frame={frame_number} branch={branch_uid} "
                f"settled_ratio={metrics['settled_ratio']:.3f} "
                f"coverage={metrics['coverage_ratio']:.3f} "
                f"left_gap={metrics['left_edge_gap']:.2f} "
                f"right_gap={metrics['right_edge_gap']:.2f} "
                f"max_internal_gap={metrics['max_internal_gap']:.2f}"
            )
            if branch_uid != selected_uid:
                print(
                    "[UnselectedGuardHold] "
                    f"frame={frame_number} branch={branch_uid} "
                    f"max_slot_error={metrics['max_slot_error']:.2f} "
                    f"leakage_count={leakage_count}"
                )
            else:
                print(
                    "[FrontierProgress] "
                    f"frame={frame_number} branch={branch_uid} "
                    f"frontier_depth={frame.frontier_depth:.2f} "
                    f"row_ready={row_ready} "
                    f"normal_support_front={frame.normal_support_front:.2f}"
                )
        flow_row = {
            "frame": frame_number,
            "branch": selected_uid or "",
            "entry_count": selected_flow.get("entry_robot_count", 0),
            "mean_axial_progress": selected_flow.get("mean_axial_progress", 0.0),
            "mean_axial_velocity": selected_flow.get("mean_axial_velocity", 0.0),
        }
        self.selected_flow_rows.append(flow_row)
        if self.state == STATE_FORM_ALL_JUNCTION_GUARDS and all_ready:
            self.state = STATE_ALL_BRANCH_GUARDS_READY
            self.all_ready_frame = frame_number
            print(
                "[AllBranchGuardsReady] "
                f"frame={frame_number} branches={list(self.frames)}"
            )
            self._commit_frontier(frame_number)
        if (
            self.state == STATE_FRONTIER_SELECTED
            and flow_row["entry_count"] > 0
            and flow_row["mean_axial_velocity"] > 0.0
        ):
            self.state = STATE_SELECTED_BRANCH_EXPLORATION_READY
            self.ready_frame = frame_number
            print(
                "[SelectedBranchExplorationReady] "
                f"frame={frame_number} selected={flow_row['branch']}"
            )



def _fixture_adapter_for_local_direction(
    branch: LidarBranchCandidate,
    geometry: adaptive.GeometryCase,
    discovered_branches: Sequence[LidarBranchCandidate],
) -> adaptive.BranchSpec:
    """Map an already-discovered Branch to a capture region for election.

    This adapter is called only by ``detect_shepherds_with_localization``.
    It does not discover or select a Branch.
    """

    ordered_local = sorted(
        discovered_branches, key=lambda item: item.center_angle_deg
    )
    ordered_fixture = sorted(
        geometry.branches, key=lambda spec: -float(spec.angle_deg)
    )
    branch_rank = ordered_local.index(branch)
    return ordered_fixture[min(branch_rank, len(ordered_fixture) - 1)]



def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(
        key for row in rows for key in row
    ))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_headless(args: argparse.Namespace) -> dict[str, Any]:
    config = adaptive.DetectorExperimentConfig(
        threshold_mode="w-tau",
        worst_wall_range=args.worst_wall_range,
        noise_model=BASELINE_NOISE_MODEL,
        noise_fraction=args.noise_fraction,
        noise_sigma=None,
        noise_seed=args.noise_seed,
        threshold_alpha=BASELINE_ALPHA,
        smoothing_window=5,
        dump_beams=False,
        anchor_stop_on_detect=True,
        evaluate_interval=False,
    )
    tracker = AnchorOpeningTracker()
    entry_controller = LocalBranchEntryController()
    guard_manager = LocalGuardManager()
    geometry_shadow = LocalBranchGeometryShadowManager()
    localization_trigger_frame: int | None = None
    with exp_adaptive_threshold_01_configuration():
        session = adaptive.AdaptiveSession(BASELINE_MAP_CASE, config)
        world = session.runner.world
        original_world_step = world.step

        def controlled_world_step() -> None:
            realized_velocities = guard_manager.advance_roles_before_physics(world)
            preexisting_fixed = set(world.fixed_robot_ids)
            world.fixed_robot_ids.update(guard_manager.roles)
            original_world_step()
            world.fixed_robot_ids.intersection_update(preexisting_fixed)
            guard_manager.restore_role_observations(world, realized_velocities)
            entry_controller.apply_after_physics_step(world)

        world.step = controlled_world_step
        discovery_logged = False
        for _ in range(args.frames):
            snapshot = session.advance_physics_frame()
            if snapshot is None or not snapshot.anchor_fixed:
                continue
            tracker.observe(snapshot)
            if tracker.complete and not discovery_logged:
                candidates_now = tracker.confirmed_children()
                print(
                    "[BranchDiscoveryComplete] "
                    f"parent_count={int(tracker.parent.observed_count > 0)} "
                    f"child_count={len(candidates_now)} "
                    f"child_ids={[item.uid for item in candidates_now]} "
                    f"child_angles={[round(item.center_angle_deg, 2) for item in candidates_now]}"
                )
                discovery_logged = True
                if len(candidates_now) == REQUIRED_CHILD_COUNT_FOR_BASELINE:
                    localization_trigger_frame = snapshot.physics_frame
                    guard_manager.elect_all(world, snapshot, candidates_now)
                    geometry_shadow.analyze(
                        snapshot,
                        candidates_now,
                        candidates_now,
                        guard_manager.elections,
                        world.geometry,
                    )
                    guard_manager.build_from_local_geometry(
                        world, snapshot, geometry_shadow.results
                    )
            if (
                guard_manager.selected_branch is not None
                and not entry_controller.active
                and guard_manager.selected_branch.uid in guard_manager.frames
            ):
                selected_frame = guard_manager.frames[
                    guard_manager.selected_branch.uid
                ]
                entry_controller.activate_from_guard_frame(
                    guard_manager.selected_branch,
                    selected_frame,
                    set(guard_manager.roles),
                )
            entry_row = entry_controller.sample(
                world, snapshot.physics_frame
            )
            guard_manager.sample(
                world, snapshot.physics_frame, entry_row
            )

    detection = next(
        (item for item in session.snapshots if item.junction_detected), None
    )
    candidates = tracker.confirmed_children()
    selected = guard_manager.selected_branch
    detection_diagnostic = (
        None if detection is None
        else opening_pipeline_diagnostic(detection, session.runner.geometry)
    )
    final_entry = (
        entry_controller.timeline[-1]
        if entry_controller.timeline
        else entry_controller.sample(
            session.runner.world, session.next_physics_frame - 1
        )
    )
    anchor = session.anchor_fix_position
    phase_status = dict(PHASE_STATUS)
    phase_status["Adaptive LiDAR"] = "PASS" if session.snapshots else "FAIL"
    phase_status["Junction detection"] = "PASS" if detection is not None else "FAIL"
    phase_status["Anchor fixation"] = "PASS" if session.anchor_fixed else "FAIL"
    phase_status["Opening detection"] = "PASS" if detection is not None else "FAIL"
    phase_status["Branch candidate creation"] = (
        "PASS" if len(candidates) == REQUIRED_CHILD_COUNT_FOR_BASELINE else "FAIL"
    )
    phase_status["Branch selection"] = "PASS" if selected is not None else "BLOCKED"
    phase_status["Selected Branch exploration"] = (
        "PASS" if entry_controller.active else "BLOCKED"
    )
    phase_status["Local SPH Branch entry"] = (
        "PASS" if final_entry["entry_robot_count"] > 0 else "BLOCKED"
    )
    phase_status["Shepherd localization trigger"] = (
        "PASS" if localization_trigger_frame is not None else "BLOCKED"
    )
    elections = guard_manager.elections
    complete_elections = sum(
        len(item.selected_ids) == item.required_count for item in elections
    )
    expected_branches = {item.uid for item in candidates}
    election_pass = (
        len(elections) == REQUIRED_CHILD_COUNT_FOR_BASELINE
        and complete_elections == len(elections)
        and guard_manager.guard_ids_locked
    )
    phase_status["Initial 3-Branch Guard election"] = (
        "PASS" if election_pass else "FAIL"
    )
    if election_pass:
        phase_status["Shepherd candidates"] = "PASS"
        phase_status["Shepherd ID election"] = "PASS"
        phase_status["Guard IDs elected-only"] = "PASS"
        phase_status["Post-election Guard localization"] = "PASS"
    for branch_uid in expected_branches:
        local_frame = guard_manager.frames.get(branch_uid)
        phase_status[f"{branch_uid} Guard row"] = (
            "PASS"
            if local_frame is not None and local_frame.formation_frame is not None
            else "BLOCKED"
        )
    latest_metrics = [
        frame.latest_metrics for frame in guard_manager.frames.values()
        if frame.latest_metrics
    ]
    all_rows_ready = (
        len(latest_metrics) == REQUIRED_CHILD_COUNT_FOR_BASELINE
        and all(row["row_ready"] for row in latest_metrics)
    )
    phase_status["Guard settled"] = "PASS" if all_rows_ready else "BLOCKED"
    phase_status["Guard coverage"] = (
        "PASS" if latest_metrics and all(
            row["coverage_ratio"] >= FRONTIER_LINE_MIN_SPAN_RATIO
            for row in latest_metrics
        ) else "BLOCKED"
    )
    phase_status["Guard edge gaps"] = (
        "PASS" if latest_metrics and all(
            row["left_edge_gap"] <= FRONTIER_LINE_MAX_EDGE_GAP
            and row["right_edge_gap"] <= FRONTIER_LINE_MAX_EDGE_GAP
            for row in latest_metrics
        ) else "BLOCKED"
    )
    phase_status["Guard internal gaps"] = (
        "PASS" if latest_metrics and all(
            row["max_internal_gap"] <= FRONTIER_LINE_MAX_INTERNAL_GAP
            for row in latest_metrics
        ) else "BLOCKED"
    )
    phase_status["Unselected Branch leakage"] = (
        "PASS" if all_rows_ready and all(
            row["entry_robot_count"] == 0
            for row in guard_manager.leakage_rows[-REQUIRED_CHILD_COUNT_FOR_BASELINE:]
        ) else "BLOCKED"
    )
    selected_flow_valid = bool(
        guard_manager.selected_flow_rows
        and guard_manager.selected_flow_rows[-1]["entry_count"] > 0
        and guard_manager.selected_flow_rows[-1]["mean_axial_velocity"] > 0.0
    )
    phase_status["Selected Branch flow"] = (
        "PASS" if selected_flow_valid else "BLOCKED"
    )
    shadow_results = geometry_shadow.results
    for branch_uid in expected_branches:
        shadow = shadow_results.get(branch_uid)
        prefix = f"{branch_uid} shadow"
        phase_status[f"{prefix} left wall fit"] = (
            "PASS" if shadow is not None and shadow.left_fit.valid else "FAIL"
        )
        phase_status[f"{prefix} right wall fit"] = (
            "PASS" if shadow is not None and shadow.right_fit.valid else "FAIL"
        )
        phase_status[f"{prefix} physical axis"] = (
            "PASS" if shadow is not None and shadow.axis is not None else "FAIL"
        )
        phase_status[f"{prefix} mouth width"] = (
            "PASS"
            if shadow is not None and shadow.physical_width is not None
            else "FAIL"
        )
        phase_status[f"{prefix} Guard row geometry"] = (
            "PASS" if shadow is not None and bool(shadow.slots) else "FAIL"
        )
    phase_status["All Branch Guards ready"] = (
        "PASS" if guard_manager.all_ready_frame is not None else "BLOCKED"
    )
    phase_status["Selected Guard -> Frontier"] = (
        "PASS" if guard_manager.frontier_ids else "BLOCKED"
    )
    phase_status["Selected Branch exploration"] = (
        "PASS"
        if guard_manager.state == STATE_SELECTED_BRANCH_EXPLORATION_READY
        else "BLOCKED"
    )
    old_vs_new_geometry = []
    for branch_uid, shadow in shadow_results.items():
        old_frame = guard_manager.frames.get(branch_uid)
        old_vs_new_geometry.append({
            "branch": branch_uid,
            "old_opening_center_axis_deg": shadow.branch.center_angle_deg,
            "new_wall_axis_deg": shadow.estimated_axis_angle_deg,
            "axis_difference_deg": shadow.axis_difference_deg,
            "old_row_orientation_deg": float(adaptive._normalize_angles(
                shadow.branch.center_angle_deg + 90.0
            )),
            "new_row_orientation_deg": (
                None if shadow.estimated_axis_angle_deg is None
                else float(adaptive._normalize_angles(
                    shadow.estimated_axis_angle_deg + 90.0
                ))
            ),
            "old_observed_mouth_width": (
                None if old_frame is None else old_frame.observed_mouth_width
            ),
            "new_physical_width": shadow.physical_width,
            "old_final_max_internal_gap": (
                None if old_frame is None
                else old_frame.latest_metrics.get("max_internal_gap")
            ),
            "new_theoretical_max_internal_gap": (
                shadow.geometry_metrics["max_internal_slot_gap"]
            ),
        })
    result = {
        "experiment": EXPERIMENT_NAME,
        "map_case": BASELINE_MAP_CASE,
        "frames": args.frames,
        "margin_ratio": BASELINE_MARGIN_RATIO,
        "alpha": BASELINE_ALPHA,
        "noise_model": BASELINE_NOISE_MODEL,
        "tau": config.tau,
        "first_junction_confirmation_frame": session.confirmation_frame,
        "anchor_fix_frame": session.anchor_fix_frame,
        "anchor_fix_position": None if anchor is None else anchor.tolist(),
        "adaptive_w": None if detection is None else detection.adaptive_worst_wall_range,
        "tmin": None if detection is None else detection.adaptive_lower_bound,
        "selected_threshold": None if detection is None else detection.adaptive_selected_threshold,
        "opening_count": 0 if detection is None else len(detection.opening_groups),
        "detected_opening_groups": (
            [] if detection is None
            else [dict(item) for item in detection.opening_groups]
        ),
        "opening_pipeline_diagnostic": detection_diagnostic,
        "persistent_openings": tracker.summary(),
        "branch_candidates": [asdict(item) for item in candidates],
        "expected_child_count_eval_only": REQUIRED_CHILD_COUNT_FOR_BASELINE,
        "selected_branch_uid": None if selected is None else selected.uid,
        "shepherd_localization_trigger_frame": localization_trigger_frame,
        "branch_entry_control": {
            "active": entry_controller.active,
            "equation": "F_total = F_SPH + 0.5*LOCAL_FORWARD_DRIVE_FORCE*t_branch",
            "world_direction": entry_controller.world_direction.tolist(),
            "final": final_entry,
            "timeline": entry_controller.timeline,
        },
        "elections": [asdict(item) for item in elections],
        "guard_ids_locked": guard_manager.guard_ids_locked,
        "localization_election_closed": guard_manager.localization_election_closed,
        "localization_runtime_uses_after_election": 0,
        "guard_roles": {
            str(robot_id): role
            for robot_id, role in sorted(guard_manager.roles.items())
        },
        "local_branch_geometry_shadow": geometry_shadow.serializable(),
        "old_vs_new_branch_geometry": old_vs_new_geometry,
        "guard_controller_committed": bool(guard_manager.frames),
        "guard_controller_geometry_policy": (
            "CONFIRMED_LOCAL_PHYSICAL_FRAME_ONLY_NO_GT_FALLBACK"
        ),
        "geometry_blockers": dict(guard_manager.geometry_blockers),
        "protected_branches": [
            item.uid for item in guard_manager.protected_branches
        ],
        "guard_state": guard_manager.state,
        "guard_frames": {
            uid: frame.serializable()
            for uid, frame in guard_manager.frames.items()
        },
        "guard_formation_rows": guard_manager.formation_rows,
        "guard_leakage_rows": guard_manager.leakage_rows,
        "selected_flow_rows": guard_manager.selected_flow_rows,
        "all_branch_guards_ready_frame": guard_manager.all_ready_frame,
        "branch_selection_frame": guard_manager.selection_frame,
        "selected_branch_exploration_ready_frame": guard_manager.ready_frame,
        "phase_status": phase_status,
        "largest_blocker": (
            "LiDAR-only persistent Branch discovery did not produce 3/3 children."
            if len(candidates) != REQUIRED_CHILD_COUNT_FOR_BASELINE
            else "Three-Branch Guard election quorum was not reached."
            if not election_pass
            else (
                "Anchor-fixed LiDAR exposes only one valid longitudinal wall "
                "per affected Branch; physical width and local slots remain "
                "UNCONFIRMED without an additional local viewpoint."
            )
            if shadow_results and any(
                shadow.physical_width is None
                for shadow in shadow_results.values()
            )
            else "Local Guard rows did not satisfy physical formation metrics."
            if guard_manager.all_ready_frame is None
            else "Selected Branch flow was not maintained after Guard formation."
            if guard_manager.ready_frame is None
            else "NONE_WITHIN_TASK_SCOPE"
        ),
        "branch_association_rows": tracker.rows,
    }
    return result


def _save_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "integration_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _write_csv(
        output_dir / "physical_dfs_phase_status.csv",
        [
            {"phase": phase, "status": status}
            for phase, status in result["phase_status"].items()
        ],
    )
    _write_csv(
        output_dir / "physical_dfs_reference_audit.csv",
        [asdict(row) for row in AUDIT_ROWS],
    )
    _write_csv(
        output_dir / "shepherd_elections.csv",
        [dict(row) for row in result["elections"]],
    )
    _write_csv(
        output_dir / "branch_associations.csv",
        result["branch_association_rows"],
    )
    _write_csv(
        output_dir / "branch_entry_timeline.csv",
        result["branch_entry_control"]["timeline"],
    )
    _write_csv(
        output_dir / "guard_formation.csv",
        result["guard_formation_rows"],
    )
    _write_csv(
        output_dir / "guard_leakage.csv",
        result["guard_leakage_rows"],
    )
    _write_csv(
        output_dir / "selected_branch_flow.csv",
        result["selected_flow_rows"],
    )
    shadow = result["local_branch_geometry_shadow"]
    wall_sample_rows = []
    wall_fit_rows = []
    branch_geometry_rows = []
    shadow_slot_rows = []
    shadow_geometry_rows = []
    walkability_rows = []
    for branch_uid, geometry_row in shadow["results"].items():
        for side_key in ("left_fit", "right_fit"):
            fit = geometry_row[side_key]
            wall_sample_rows.append({
                "branch": branch_uid,
                "side": fit["side"],
                "angle_band_start_deg": fit["angle_band"][0],
                "angle_band_end_deg": fit["angle_band"][1],
                "sample_count": fit["sample_count"],
                "range_min": fit["range_min"],
                "range_max": fit["range_max"],
            })
            wall_fit_rows.append({
                "branch": branch_uid,
                "side": fit["side"],
                "inlier_count": fit["inlier_count"],
                "direction_angle_deg": fit["direction_angle_deg"],
                "residual": fit["residual"],
                "span": fit["span"],
                "valid": fit["valid"],
                "invalid_reason": fit["invalid_reason"],
            })
        branch = geometry_row["branch"]
        evaluation = geometry_row["evaluation_only"]
        branch_geometry_rows.append({
            "branch": branch_uid,
            "opening_center_angle_deg": branch["center_angle_deg"],
            "estimated_axis_angle_deg": geometry_row["estimated_axis_angle_deg"],
            "axis_difference_deg": geometry_row["axis_difference_deg"],
            "parallel_error_deg": geometry_row["parallel_error_deg"],
            "confidence": geometry_row["confidence"],
            "physical_width": geometry_row["physical_width"],
            "usable_width": geometry_row["usable_width"],
            "old_observed_width": (
                result["guard_frames"].get(branch_uid, {})
                .get("observed_mouth_width")
            ),
            "gt_axis_angle_deg_evaluation_only": evaluation["gt_local_axis_angle_deg"],
            "gt_axis_error_deg_evaluation_only": evaluation["estimated_axis_error_deg"],
            "gt_width_evaluation_only": evaluation["gt_corridor_width"],
            "gt_width_error_evaluation_only": evaluation["estimated_width_error"],
        })
        metrics = geometry_row["geometry_metrics"]
        shadow_geometry_rows.append({
            "branch": branch_uid,
            **metrics,
            "slot_walkability_ratio_evaluation_only": evaluation["slot_walkability_ratio"],
        })
        for index, slot in enumerate(geometry_row["slots"]):
            shadow_slot_rows.append({
                "branch": branch_uid,
                "status": "CONFIRMED",
                "slot_index": index,
                "local_x": slot[0],
                "local_y": slot[1],
                "lateral_offset": geometry_row["lateral_offsets"][index],
            })
        if not geometry_row["slots"]:
            shadow_slot_rows.append({
                "branch": branch_uid,
                "status": "UNCONFIRMED_PHYSICAL_WIDTH",
                "slot_index": None,
                "local_x": None,
                "local_y": None,
                "lateral_offset": None,
            })
        for slot in evaluation["slots"]:
            walkability_rows.append({"branch": branch_uid, **slot})
        if not evaluation["slots"]:
            walkability_rows.append({
                "branch": branch_uid,
                "slot_index": None,
                "local_position": None,
                "world_position_evaluation_only": None,
                "walkable": None,
                "wall_clearance": None,
                "status": "NOT_EVALUATED_NO_CONFIRMED_SLOTS",
            })
    _write_csv(output_dir / "local_wall_samples.csv", wall_sample_rows)
    _write_csv(output_dir / "local_wall_fits.csv", wall_fit_rows)
    _write_csv(output_dir / "local_branch_geometry.csv", branch_geometry_rows)
    _write_csv(output_dir / "shadow_guard_slots.csv", shadow_slot_rows)
    _write_csv(output_dir / "shadow_guard_geometry.csv", shadow_geometry_rows)
    _write_csv(
        output_dir / "old_vs_new_branch_geometry.csv",
        result["old_vs_new_branch_geometry"],
    )
    _write_csv(
        output_dir / "shadow_guard_walkability_evaluation_only.csv",
        walkability_rows,
    )
    diagnostic = result.get("opening_pipeline_diagnostic")
    if diagnostic is not None:
        _write_csv(
            output_dir / "opening_pipeline_components.csv",
            diagnostic["components"],
        )
        _write_csv(
            output_dir / "expected_sectors_evaluation_only.csv",
            diagnostic["expected_sectors_evaluation_only"],
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--worst-wall-range", type=float, default=100.0)
    parser.add_argument(
        "--noise-fraction", type=float, default=adaptive.DEFAULT_NOISE_FRACTION
    )
    parser.add_argument("--noise-seed", type=int, default=adaptive.DEFAULT_NOISE_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--gui",
        action="store_true",
        help="show Adaptive Anchor plus local protected-Branch Guard overlays",
    )
    parser.add_argument("--show-profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pause-on-detect", action="store_true")
    parser.add_argument(
        "--verify-regression",
        action="store_true",
        help="repeat an independent direct run and compare Anchor signatures",
    )
    args = parser.parse_args(argv)
    if args.frames <= 0:
        parser.error("--frames must be positive")
    return args


def _run_gui(args: argparse.Namespace) -> None:
    import pygame

    config = adaptive.DetectorExperimentConfig(
        threshold_mode="w-tau",
        worst_wall_range=args.worst_wall_range,
        noise_model=BASELINE_NOISE_MODEL,
        noise_fraction=args.noise_fraction,
        noise_sigma=None,
        noise_seed=args.noise_seed,
        threshold_alpha=BASELINE_ALPHA,
        smoothing_window=5,
        dump_beams=False,
        anchor_stop_on_detect=True,
        evaluate_interval=False,
    )
    with exp_adaptive_threshold_01_configuration():
        pygame.init()
        session = adaptive.AdaptiveSession(BASELINE_MAP_CASE, config)
        tracker = AnchorOpeningTracker()
        controller = LocalBranchEntryController()
        guard_manager = LocalGuardManager()
        geometry_shadow = LocalBranchGeometryShadowManager()
        world = session.runner.world
        original_world_step = world.step

        def controlled_world_step() -> None:
            realized_velocities = guard_manager.advance_roles_before_physics(world)
            preexisting_fixed = set(world.fixed_robot_ids)
            world.fixed_robot_ids.update(guard_manager.roles)
            original_world_step()
            world.fixed_robot_ids.intersection_update(preexisting_fixed)
            guard_manager.restore_role_observations(world, realized_velocities)
            controller.apply_after_physics_step(world)

        world.step = controlled_world_step
        renderer = adaptive.AdaptiveRenderer(
            pygame, session.runner.geometry, args.show_profile
        )
        clock = pygame.time.Clock()
        paused = False
        running = True
        pause_consumed = False
        while running and session.next_physics_frame < args.frames:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
            if not paused:
                snapshot = session.advance_physics_frame()
                if snapshot is not None and snapshot.anchor_fixed:
                    tracker.observe(snapshot)
                    children = tracker.confirmed_children()
                    if (
                        tracker.complete
                        and len(children) == REQUIRED_CHILD_COUNT_FOR_BASELINE
                        and not guard_manager.elections
                    ):
                        guard_manager.elect_all(world, snapshot, children)
                        geometry_shadow.analyze(
                            snapshot,
                            children,
                            children,
                            guard_manager.elections,
                            world.geometry,
                        )
                        guard_manager.build_from_local_geometry(
                            world, snapshot, geometry_shadow.results
                        )
                    if (
                        guard_manager.selected_branch is not None
                        and not controller.active
                        and guard_manager.selected_branch.uid
                        in guard_manager.frames
                    ):
                        selected_frame = guard_manager.frames[
                            guard_manager.selected_branch.uid
                        ]
                        controller.activate_from_guard_frame(
                            guard_manager.selected_branch,
                            selected_frame,
                            set(guard_manager.roles),
                        )
                    entry_row = controller.sample(
                        world, snapshot.physics_frame
                    )
                    guard_manager.sample(
                        world, snapshot.physics_frame, entry_row
                    )
                    if args.pause_on_detect and not pause_consumed:
                        paused = True
                        pause_consumed = True
            renderer.draw(session, paused)
            snapshot = (
                session.snapshots[session.view_index]
                if session.snapshots and session.view_index >= 0 else None
            )
            if snapshot is not None and snapshot.anchor_fixed:
                anchor_screen = renderer.world_to_screen(
                    snapshot.leader_position, snapshot
                )
                children = tracker.confirmed_children()
                selected_uid = (
                    guard_manager.selected_branch.uid
                    if guard_manager.selected_branch is not None else None
                )
                # Fixed Anchor: bright cyan diamond with a dark outline.
                diamond = [
                    (anchor_screen[0], anchor_screen[1] - 9),
                    (anchor_screen[0] + 9, anchor_screen[1]),
                    (anchor_screen[0], anchor_screen[1] + 9),
                    (anchor_screen[0] - 9, anchor_screen[1]),
                ]
                pygame.draw.polygon(renderer.screen, ANCHOR_GUI_COLOR, diamond)
                pygame.draw.polygon(renderer.screen, (15, 25, 35), diamond, 2)
                parent_direction = selected_branch_world_direction_for_physics(
                    tracker.parent.candidate(), snapshot.lidar_yaw_deg
                )
                pygame.draw.line(
                    renderer.screen,
                    PARENT_GUI_COLOR,
                    anchor_screen,
                    renderer.world_to_screen(
                        snapshot.leader_position + parent_direction * 28.0,
                        snapshot,
                    ),
                    width=2,
                )
                for branch in children:
                    direction = selected_branch_world_direction_for_physics(
                        branch, snapshot.lidar_yaw_deg
                    )
                    endpoint = snapshot.leader_position + direction * 35.0
                    color = BRANCH_GUI_COLORS.get(branch.uid, (230, 220, 130))
                    pygame.draw.line(
                        renderer.screen,
                        color,
                        anchor_screen,
                        renderer.world_to_screen(endpoint, snapshot),
                        width=2,
                    )
                    if branch.uid == selected_uid:
                        pygame.draw.line(
                            renderer.screen,
                            SELECTED_GUI_COLOR,
                            anchor_screen,
                            renderer.world_to_screen(
                                snapshot.leader_position + direction * 43.0,
                                snapshot,
                            ),
                            width=5,
                        )
                robot_by_id = {
                    robot.robot_id: robot for robot in world.robots
                }
                for guard_frame in guard_manager.frames.values():
                    slot_points = [
                        renderer.world_to_screen(slot, snapshot)
                        for slot in guard_frame.slots
                    ]
                    for slot_point in slot_points:
                        pygame.draw.rect(
                            renderer.screen,
                            GUARD_SLOT_GUI_COLOR,
                            pygame.Rect(slot_point[0] - 4, slot_point[1] - 4, 8, 8),
                            width=2,
                        )
                    for robot_id in guard_frame.elected_ids:
                        guard_point = renderer.world_to_screen(
                            robot_by_id[robot_id].position, snapshot
                        )
                        role = guard_manager.roles.get(robot_id, ROLE_NORMAL)
                        fill = (
                            FRONTIER_GUI_COLOR
                            if role == ROLE_FRONTIER_SHEPHERD
                            else GUARD_ROBOT_GUI_COLOR
                        )
                        pygame.draw.circle(
                            renderer.screen, fill, guard_point, 6
                        )
                        pygame.draw.circle(
                            renderer.screen, (20, 25, 35), guard_point, 7, 2
                        )
                # Elected IDs remain a ring-only diagnostic until local slot
                # geometry is confirmed and a physical role is committed.
                for robot_id in sorted(guard_manager.elected_ids - set(guard_manager.roles)):
                    pending_point = renderer.world_to_screen(
                        robot_by_id[robot_id].position, snapshot
                    )
                    pygame.draw.circle(
                        renderer.screen, GUARD_ROBOT_GUI_COLOR, pending_point, 6, 2
                    )
                for shadow in geometry_shadow.results.values():
                    branch_color = BRANCH_GUI_COLORS.get(
                        shadow.branch.uid, (80, 220, 110)
                    )
                    if shadow.axis is not None:
                        axis_world_end = _anchor_local_to_world(
                            shadow.axis * 55.0, snapshot
                        )
                        pygame.draw.line(
                            renderer.screen,
                            branch_color,
                            anchor_screen,
                            renderer.world_to_screen(axis_world_end, snapshot),
                            width=4,
                        )
                    for fit, color in (
                        (shadow.left_fit, (255, 150, 70)),
                        (shadow.right_fit, (90, 180, 255)),
                    ):
                        if (
                            fit.direction is None
                            or fit.point_on_line is None
                            or len(fit.inlier_points) < 2
                        ):
                            continue
                        projections = (
                            fit.inlier_points - fit.point_on_line
                        ) @ fit.direction
                        local_start = (
                            fit.point_on_line
                            + fit.direction * float(np.min(projections))
                        )
                        local_end = (
                            fit.point_on_line
                            + fit.direction * float(np.max(projections))
                        )
                        pygame.draw.line(
                            renderer.screen,
                            color,
                            renderer.world_to_screen(
                                _anchor_local_to_world(local_start, snapshot),
                                snapshot,
                            ),
                            renderer.world_to_screen(
                                _anchor_local_to_world(local_end, snapshot),
                                snapshot,
                            ),
                            width=3,
                        )
                    if shadow.mouth_origin is not None:
                        mouth_screen = renderer.world_to_screen(
                            _anchor_local_to_world(
                                shadow.mouth_origin, snapshot
                            ),
                            snapshot,
                        )
                        pygame.draw.circle(
                            renderer.screen, (255, 255, 255), mouth_screen, 6, 2
                        )
                final_entry = (
                    controller.timeline[-1]
                    if controller.timeline else {"entry_robot_count": 0}
                )
                protected = ",".join(
                    item.uid for item in guard_manager.protected_branches
                ) or "-"
                latest_guard_lines = []
                for branch_uid, guard_frame in guard_manager.frames.items():
                    metrics = guard_frame.latest_metrics
                    latest_guard_lines.append(
                        f"{branch_uid}: guards={len(guard_frame.elected_ids)} "
                        f"coverage={100.0 * metrics.get('coverage_ratio', 0.0):.1f}% "
                        f"gaps={metrics.get('left_edge_gap', 0.0):.1f}/"
                        f"{metrics.get('right_edge_gap', 0.0):.1f}"
                    )
                latest_leakage = sum(
                    row["entry_robot_count"]
                    for row in guard_manager.leakage_rows[-3:]
                ) if guard_manager.leakage_rows else 0
                shadow_lines = []
                for branch_uid, shadow in geometry_shadow.results.items():
                    shadow_lines.append(
                        f"{branch_uid} wall-axis: "
                        f"{shadow.estimated_axis_angle_deg} deg "
                        f"width={shadow.physical_width} "
                        f"slots={len(shadow.slots)}"
                    )
                lines = (
                    f"STATE: {guard_manager.state}",
                    f"Persistent parent: {tracker.parent.center_angle_deg:+.1f} deg",
                    f"Persistent children: {len(children)}/3",
                    f"Selected: {selected_uid or '-'}",
                    f"Protected: {protected}",
                    f"Guard IDs locked: {guard_manager.guard_ids_locked}",
                    f"Branch-entry robots: {final_entry['entry_robot_count']}",
                    f"Unselected leakage: {latest_leakage}",
                    *(
                        f"{uid} BLOCKED: {reason}"
                        for uid, reason in guard_manager.geometry_blockers.items()
                    ),
                    *latest_guard_lines,
                    *shadow_lines,
                )
                for index, line in enumerate(lines):
                    renderer.text(
                        line,
                        (880, 790 + 20 * index),
                        font=renderer.small_font,
                    )
                pygame.display.flip()
            clock.tick(60)
        pygame.quit()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    before = {
        "adaptive": _sha256(ADAPTIVE_SOURCE),
        "reference": _sha256(REFERENCE_PROTOTYPE),
    }
    if args.gui:
        _run_gui(args)
    result = _run_headless(args)
    if args.verify_regression:
        direct = _run_headless(args)
        keys = (
            "first_junction_confirmation_frame",
            "anchor_fix_frame",
            "anchor_fix_position",
            "adaptive_w",
            "tmin",
            "selected_threshold",
            "opening_count",
        )
        result["adaptive_anchor_regression"] = {
            "equivalent": all(result[key] == direct[key] for key in keys),
            "compared_fields": list(keys),
            "direct_signature": {key: direct[key] for key in keys},
        }
        result["deterministic_replay"] = {
            "persistent_descriptors_equal": (
                result["persistent_openings"]
                == direct["persistent_openings"]
            ),
            "branch_candidates_equal": (
                result["branch_candidates"] == direct["branch_candidates"]
            ),
            "selected_branch_equal": (
                result["selected_branch_uid"]
                == direct["selected_branch_uid"]
            ),
            "trigger_frame_equal": (
                result["shepherd_localization_trigger_frame"]
                == direct["shepherd_localization_trigger_frame"]
            ),
            "election_equal": result["elections"] == direct["elections"],
            "protected_branches_equal": (
                result["protected_branches"] == direct["protected_branches"]
            ),
            "guard_slot_ordering_equal": (
                result["guard_frames"] == direct["guard_frames"]
            ),
            "guard_state_equal": (
                result["guard_state"] == direct["guard_state"]
            ),
            "all_ready_frame_equal": (
                result["all_branch_guards_ready_frame"]
                == direct["all_branch_guards_ready_frame"]
            ),
            "ready_frame_equal": (
                result["selected_branch_exploration_ready_frame"]
                == direct["selected_branch_exploration_ready_frame"]
            ),
            "local_branch_geometry_shadow_equal": (
                result["local_branch_geometry_shadow"]
                == direct["local_branch_geometry_shadow"]
            ),
            "direct_elections": direct["elections"],
        }
    after = {
        "adaptive": _sha256(ADAPTIVE_SOURCE),
        "reference": _sha256(REFERENCE_PROTOTYPE),
    }
    result["source_hash_before"] = before
    result["source_hash_after"] = after
    result["source_hash_unchanged"] = before == after
    _save_outputs(args.output_dir, result)
    print(json.dumps(result, indent=2))
    print(f"[Output] {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
