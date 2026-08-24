"""Local-history features for persistent physical branch identity.

The helper consumes body-local motion, scan-local gap geometry, and persistent
candidate-relative geometry.  It has no map, global pose, branch label, or GT
input.  Policy is intentionally kept outside the feature definitions so an
audit-only pass can precede any identity decision.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.pointcloud.general_branch_candidate import (
    FROZEN_ENDPOINT_FRAME_RESOLUTION_DEG,
)

FROZEN_ASSOCIATION_DOT = 0.5
FROZEN_SPATIAL_ASSOCIATION_W = 0.12
DERIVED_TANGENT_PERP_DOT_MAX = math.sqrt(1.0 - FROZEN_ASSOCIATION_DOT ** 2)
DERIVED_IDENTITY_DOT = math.cos(
    math.acos(FROZEN_ASSOCIATION_DOT)
    + math.radians(FROZEN_ENDPOINT_FRAME_RESOLUTION_DEG)
)


def unit(vector: np.ndarray) -> np.ndarray:
    """Return a finite unit vector, or a zero vector for degenerate evidence."""
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    return value / norm if math.isfinite(norm) and norm > 1.0e-12 else np.zeros(2)


def rotation(angle_rad: float) -> np.ndarray:
    """Return a planar local-frame rotation."""
    return np.array([
        [math.cos(angle_rad), -math.sin(angle_rad)],
        [math.sin(angle_rad), math.cos(angle_rad)],
    ])


def wrap_angle(angle_rad: float) -> float:
    """Wrap one relative local orientation to [-pi, pi)."""
    return float((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass
class LocalMotionHistory:
    """Dead-reckoned local trajectory built only from body-local velocity."""

    history_window_seconds: float
    initial_body_yaw_rad: float | None = None
    odom_position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_time: float | None = None
    samples: list[tuple[float, np.ndarray, float]] = field(default_factory=list)

    def observe(self, timestamp: float, body_yaw_rad: float, observed_velocity_world: np.ndarray) -> None:
        """Integrate a velocity expressed in the current body frame.

        The simulator exposes observed velocity in its storage frame.  It is
        immediately transformed to body-local coordinates; only yaw change
        relative to the first sample is retained in the odometry frame.
        """
        if self.initial_body_yaw_rad is None:
            self.initial_body_yaw_rad = float(body_yaw_rad)
        relative_yaw = wrap_angle(float(body_yaw_rad) - self.initial_body_yaw_rad)
        if self.last_time is not None:
            dt = max(0.0, float(timestamp) - self.last_time)
            body_local_velocity = rotation(float(body_yaw_rad)).T @ np.asarray(observed_velocity_world, dtype=float)
            self.odom_position = self.odom_position + rotation(relative_yaw) @ body_local_velocity * dt
        self.last_time = float(timestamp)
        self.samples.append((float(timestamp), self.odom_position.copy(), relative_yaw))

    @property
    def current_yaw_odom(self) -> float:
        return 0.0 if not self.samples else float(self.samples[-1][2])

    def local_point_to_odom(self, point_local: np.ndarray) -> np.ndarray:
        """Express current scan-local geometry in the history odometry frame."""
        return self.odom_position + rotation(self.current_yaw_odom) @ np.asarray(point_local, dtype=float)

    def local_axis_to_odom(self, axis_local: np.ndarray) -> np.ndarray:
        return unit(rotation(self.current_yaw_odom) @ np.asarray(axis_local, dtype=float))

    def odom_axis_to_local(self, axis_odom: np.ndarray) -> np.ndarray:
        return unit(rotation(self.current_yaw_odom).T @ np.asarray(axis_odom, dtype=float))

    def motion_axis_local(self) -> np.ndarray:
        """Estimate recent directed motion over the frozen SPH time scale."""
        if len(self.samples) < 2:
            return np.zeros(2)
        end_time, end_position, _ = self.samples[-1]
        target_time = end_time - self.history_window_seconds
        start = min(self.samples, key=lambda row: abs(row[0] - target_time))[1]
        return self.odom_axis_to_local(end_position - start)

    def parent_axis_local(self) -> np.ndarray:
        """Return the direction from the current LiDAR pose toward its traversed path."""
        return -self.motion_axis_local()

    def motion_reliability(self) -> float:
        """Return directed displacement divided by traversed path length."""
        if len(self.samples) < 2:
            return 0.0
        end_time = self.samples[-1][0]
        recent = [row for row in self.samples if row[0] >= end_time - self.history_window_seconds]
        if len(recent) < 2:
            return 0.0
        vectors = np.diff(np.asarray([row[1] for row in recent]), axis=0)
        path_length = float(np.sum(np.linalg.norm(vectors, axis=1)))
        displacement = float(np.linalg.norm(recent[-1][1] - recent[0][1]))
        return displacement / path_length if path_length > 1.0e-12 else 0.0

    def distance_to_traversed_path(self, point_odom: np.ndarray) -> float:
        """Return distance to the full locally traversed polyline."""
        point = np.asarray(point_odom, dtype=float)
        positions = [row[1] for row in self.samples]
        if not positions:
            return math.inf
        if len(positions) == 1:
            return float(np.linalg.norm(point - positions[0]))
        distances = []
        for start, end in zip(positions, positions[1:]):
            edge = end - start
            denominator = float(np.dot(edge, edge))
            ratio = 0.0 if denominator <= 1.0e-12 else float(np.clip(np.dot(point - start, edge) / denominator, 0.0, 1.0))
            distances.append(float(np.linalg.norm(point - (start + ratio * edge))))
        return min(distances)

    def snapshot(self) -> dict[str, Any]:
        """Expose diagnostic-only local history endpoints and axes."""
        motion = self.motion_axis_local()
        parent = -motion
        return {
            "motion_axis_local": motion,
            "parent_axis_local": parent,
            "motion_reliability": self.motion_reliability(),
            "history_start_odom": self.samples[0][1].copy() if self.samples else np.zeros(2),
            "history_end_odom": self.odom_position.copy(),
            "sample_count": len(self.samples),
        }


@dataclass(frozen=True)
class CandidateObservation:
    """One accepted General Candidate observation in a common local frame."""

    observation_id: str
    timestamp: float
    frame: int
    source_gap_id: int
    opening_normal_local: np.ndarray
    opening_normal_odom: np.ndarray
    gap_tangent_local: np.ndarray
    gap_tangent_odom: np.ndarray
    center_local: np.ndarray
    center_odom: np.ndarray
    endpoint_a_local: np.ndarray
    endpoint_b_local: np.ndarray
    endpoint_a_odom: np.ndarray
    endpoint_b_odom: np.ndarray
    gap_width: float
    gap_width_over_W: float
    free_continuation: float
    incident_wall_alignment_deg: float


@dataclass
class PersistentCandidateIdentity:
    """Persistent local geometry and observation history for one branch ID."""

    candidate_id: str
    first_seen_time: float
    last_seen_time: float
    opening_normal_odom: np.ndarray
    gap_tangent_odom: np.ndarray
    center_odom: np.ndarray
    endpoints_odom: tuple[np.ndarray, ...]
    gap_width: float
    source_gap_ids: list[int] = field(default_factory=list)
    observation_ids: list[str] = field(default_factory=list)

    def append(self, observation: CandidateObservation) -> None:
        """Append evidence without resetting the persistent candidate state."""
        self.last_seen_time = observation.timestamp
        self.source_gap_ids.append(observation.source_gap_id)
        self.observation_ids.append(observation.observation_id)


def make_observation(
    observation_id: str,
    timestamp: float,
    frame: int,
    source_gap_id: int,
    descriptor: Any,
    history: LocalMotionHistory,
) -> CandidateObservation:
    """Transform one EXP-038 descriptor into the common local odometry frame."""
    return CandidateObservation(
        observation_id=observation_id,
        timestamp=float(timestamp),
        frame=int(frame),
        source_gap_id=int(source_gap_id),
        opening_normal_local=np.asarray(descriptor.opening_normal_local).copy(),
        opening_normal_odom=history.local_axis_to_odom(descriptor.opening_normal_local),
        gap_tangent_local=np.asarray(descriptor.gap_tangent_local).copy(),
        gap_tangent_odom=history.local_axis_to_odom(descriptor.gap_tangent_local),
        center_local=np.asarray(descriptor.gap_center_local).copy(),
        center_odom=history.local_point_to_odom(descriptor.gap_center_local),
        endpoint_a_local=np.asarray(descriptor.endpoint_a_local).copy(),
        endpoint_b_local=np.asarray(descriptor.endpoint_b_local).copy(),
        endpoint_a_odom=history.local_point_to_odom(descriptor.endpoint_a_local),
        endpoint_b_odom=history.local_point_to_odom(descriptor.endpoint_b_local),
        gap_width=float(descriptor.gap_width),
        gap_width_over_W=float(descriptor.gap_width_over_W_hat),
        free_continuation=float(descriptor.free_continuation),
        incident_wall_alignment_deg=float(descriptor.gap_boundary_wall_alignment_error_deg),
    )


def incoming_features(
    observation: CandidateObservation,
    history: LocalMotionHistory,
    width_hat: float,
) -> dict[str, float | bool]:
    """Measure directed parent alignment and overlap with traversed local history."""
    snapshot = history.snapshot()
    parent = np.asarray(snapshot["parent_axis_local"])
    motion = np.asarray(snapshot["motion_axis_local"])
    center_direction = unit(observation.center_local)
    path_distance = history.distance_to_traversed_path(observation.center_odom)
    return {
        "incoming_axis_dot": float(np.dot(observation.opening_normal_local, parent)),
        "parent_direction_dot": float(np.dot(center_direction, parent)),
        "parent_tangent_abs_dot": abs(float(np.dot(observation.gap_tangent_local, motion))),
        "traversed_path_distance_over_W": path_distance / max(float(width_hat), 1.0e-12),
        "motion_reliability": float(snapshot["motion_reliability"]),
        "history_available": bool(np.linalg.norm(parent) > 0.0),
    }


def _termination_assignment_distance(
    observation: CandidateObservation,
    identity: PersistentCandidateIdentity,
) -> float:
    if len(identity.endpoints_odom) < 2:
        return min(
            float(np.linalg.norm(observation.endpoint_a_odom - identity.endpoints_odom[0])),
            float(np.linalg.norm(observation.endpoint_b_odom - identity.endpoints_odom[0])),
        )
    old_a, old_b = identity.endpoints_odom[:2]
    direct = 0.5 * (
        float(np.linalg.norm(observation.endpoint_a_odom - old_a))
        + float(np.linalg.norm(observation.endpoint_b_odom - old_b))
    )
    swapped = 0.5 * (
        float(np.linalg.norm(observation.endpoint_a_odom - old_b))
        + float(np.linalg.norm(observation.endpoint_b_odom - old_a))
    )
    return min(direct, swapped)


def pairwise_features(
    observation: CandidateObservation,
    identity: PersistentCandidateIdentity,
    width_hat: float,
) -> dict[str, float | int]:
    """Measure label-free continuity against one persistent candidate."""
    scale = max(float(width_hat), 1.0e-12)
    center_distance = float(np.linalg.norm(observation.center_odom - identity.center_odom))
    termination_distance = _termination_assignment_distance(observation, identity)
    axis_dot = float(np.dot(observation.opening_normal_odom, identity.opening_normal_odom))
    normal_dot = axis_dot
    tangent_dot = abs(float(np.dot(observation.gap_tangent_odom, identity.gap_tangent_odom)))
    return {
        "axis_dot": axis_dot,
        "normal_dot": normal_dot,
        "tangent_dot": tangent_dot,
        "center_distance_over_W": center_distance / scale,
        "termination_distance_over_W": termination_distance / scale,
        "gap_width_difference_over_W": abs(observation.gap_width - identity.gap_width) / scale,
        "creation_time_difference": observation.timestamp - identity.first_seen_time,
        "shared_termination_count": 0,
        "spatial_overlap": max(0.0, 1.0 - center_distance / max(observation.gap_width, identity.gap_width, 1.0e-12)),
        "center_within_smaller_half_width": center_distance <= 0.5 * min(observation.gap_width, identity.gap_width),
        "physical_gap_disks_overlap": center_distance <= 0.5 * (observation.gap_width + identity.gap_width),
    }


def incoming_path_match(features: dict[str, float | bool]) -> bool:
    """Classify a known parent path from directed local traversal evidence.

    The direction gate is the existing 0.5 association gate.  The tangent
    bound is derived from its orthogonal complement, not fitted to benchmark
    cases.  A nonzero motion reliability merely requires usable history.
    """
    return bool(
        features["history_available"]
        and float(features["motion_reliability"]) > 0.0
        and float(features["incoming_axis_dot"]) > FROZEN_ASSOCIATION_DOT
        and float(features["parent_tangent_abs_dot"]) < DERIVED_TANGENT_PERP_DOT_MAX
    )


def best_existing_match(
    pair_rows: list[tuple[str, dict[str, float | int | bool]]],
) -> tuple[str | None, dict[str, float | int | bool] | None, str]:
    """Associate by the frozen direction gate, then physical mouth overlap.

    The fallback contains no fitted distance: two observations overlap when
    their centers lie within the smaller physical gap's half-width.  This
    represents the same open-space mouth rather than a map-specific angle.
    """
    directional = [row for row in pair_rows if float(row[1]["axis_dot"]) > FROZEN_ASSOCIATION_DOT]
    if directional:
        candidate_id, features = max(directional, key=lambda row: float(row[1]["axis_dot"]))
        return candidate_id, features, "FROZEN_DIRECTION_ASSOCIATION"
    # EXP-039's 60-degree direction association is widened only by the frozen
    # 20-degree endpoint-frame resolution.  Physical gap-disk overlap is also
    # required, preventing angle-only merging of distinct mouths.
    overlapping = [
        row for row in pair_rows
        if float(row[1]["axis_dot"]) > DERIVED_IDENTITY_DOT
        and bool(row[1]["physical_gap_disks_overlap"])
    ]
    if overlapping:
        candidate_id, features = min(overlapping, key=lambda row: float(row[1]["center_distance_over_W"]))
        return candidate_id, features, "ENDPOINT_RESOLUTION_AWARE_GAP_OVERLAP"
    return None, None, "NO_EXISTING_IDENTITY_MATCH"


def identity_from_observation(candidate_id: str, observation: CandidateObservation) -> PersistentCandidateIdentity:
    """Create a persistent record from one accepted gap observation."""
    return PersistentCandidateIdentity(
        candidate_id=candidate_id,
        first_seen_time=observation.timestamp,
        last_seen_time=observation.timestamp,
        opening_normal_odom=observation.opening_normal_odom.copy(),
        gap_tangent_odom=observation.gap_tangent_odom.copy(),
        center_odom=observation.center_odom.copy(),
        endpoints_odom=(observation.endpoint_a_odom.copy(), observation.endpoint_b_odom.copy()),
        gap_width=observation.gap_width,
        source_gap_ids=[observation.source_gap_id],
        observation_ids=[observation.observation_id],
    )


def self_test() -> None:
    """Exercise incoming and duplicate features without map or GT inputs."""
    history = LocalMotionHistory(1.0)
    for index in range(4):
        history.observe(float(index), 0.0, np.array([1.0, 0.0]))
    class Descriptor:
        opening_normal_local = np.array([-1.0, 0.0])
        gap_tangent_local = np.array([0.0, 1.0])
        gap_center_local = np.array([-1.0, 0.0])
        endpoint_a_local = np.array([-1.0, -0.5])
        endpoint_b_local = np.array([-1.0, 0.5])
        gap_width = 1.0
        gap_width_over_W_hat = 1.0
        free_continuation = 2.0
        gap_boundary_wall_alignment_error_deg = 0.0
    observation = make_observation("O0", 3.0, 3, 0, Descriptor(), history)
    incoming = incoming_features(observation, history, 1.0)
    assert incoming_path_match(incoming)
    assert incoming["traversed_path_distance_over_W"] <= FROZEN_SPATIAL_ASSOCIATION_W
    identity = identity_from_observation("C0", observation)
    repeated = make_observation("O1", 4.0, 4, 1, Descriptor(), history)
    pair = pairwise_features(repeated, identity, 1.0)
    assert pair["center_distance_over_W"] <= FROZEN_SPATIAL_ASSOCIATION_W
    assert pair["termination_distance_over_W"] <= FROZEN_SPATIAL_ASSOCIATION_W
    matched_id, _, reason = best_existing_match([("C0", pair)])
    assert matched_id == "C0" and reason == "FROZEN_DIRECTION_ASSOCIATION"
    # A direction-distorted re-observation still merges when both physical gap
    # disks overlap within the smaller mouth half-width.
    distorted = dict(pair)
    distorted.update({"axis_dot": 0.49, "center_distance_over_W": 0.42, "physical_gap_disks_overlap": True})
    matched_id, _, reason = best_existing_match([("C0", distorted)])
    assert matched_id == "C0" and reason == "ENDPOINT_RESOLUTION_AWARE_GAP_OVERLAP"


if __name__ == "__main__":
    self_test()
