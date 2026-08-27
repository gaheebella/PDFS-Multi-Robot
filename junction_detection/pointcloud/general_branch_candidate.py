"""Orientation-relative representation for a locally observed physical gap.

This module does not classify directions as side, angled, or forward.  It
represents an already accepted scan-derived gap in its own tangent/normal frame
and checks one orientation-invariant topology condition: an observed incident
wall must support the gap-boundary tangent.  The angular resolution reuses the
20-degree corner resolution already frozen in the EXP-033 endpoint extractor;
it is not fitted to branch labels or benchmark outcomes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

FROZEN_ENDPOINT_FRAME_RESOLUTION_DEG = 20.0


def _unit(vector: np.ndarray) -> np.ndarray:
    """Return a finite unit vector and reject degenerate geometry."""
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("degenerate vector in general branch geometry")
    return value / norm


def _angle_deg(vector: np.ndarray) -> float:
    """Return a vector bearing in local degrees."""
    return float(math.degrees(math.atan2(float(vector[1]), float(vector[0]))))


def _signed_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Return signed rotation from first to second in [-180, 180)."""
    a, b = _unit(first), _unit(second)
    return float(math.degrees(math.atan2(a[0] * b[1] - a[1] * b[0], np.dot(a, b))))


def _axial_difference_deg(first: float, second: float) -> float:
    """Return acute difference between two unoriented line axes."""
    return float(abs((first - second + 90.0) % 180.0 - 90.0))


@dataclass(frozen=True)
class OrientationRelativeGapDescriptor:
    """Continuous local geometry shared by side, angled, and axial gaps."""

    endpoint_a_local: np.ndarray
    endpoint_b_local: np.ndarray
    gap_center_local: np.ndarray
    gap_width: float
    gap_width_over_W_hat: float
    gap_tangent_local: np.ndarray
    gap_normal_candidate_1: np.ndarray
    gap_normal_candidate_2: np.ndarray
    opening_normal_local: np.ndarray
    free_space_direction_local: np.ndarray
    free_continuation: float
    corridor_axis_local: np.ndarray
    corridor_lateral_axis_local: np.ndarray
    relative_opening_angle_deg: float
    endpoint_a_type: str
    endpoint_b_type: str
    endpoint_a_wall_tangent_deg: tuple[float, ...]
    endpoint_b_wall_tangent_deg: tuple[float, ...]
    gap_tangent_deg_local: float
    opening_normal_deg_local: float
    free_space_direction_deg_local: float
    normal_free_alignment: float
    gap_boundary_wall_alignment_error_deg: float
    geometry_support: bool
    rejection_reason: str


@dataclass(frozen=True)
class GeneralBranchCandidate:
    """A direction-label-free hypothesis passed to later validation layers."""

    candidate_id: str
    timestamp: float
    topology_type: str
    descriptor: OrientationRelativeGapDescriptor
    candidate_reliability: float


def _incident_tangent_angles(
    endpoint: dict[str, Any],
    segments: dict[int, dict[str, Any]],
) -> tuple[float, ...]:
    """Collect fitted incident-wall axes for one scan-derived endpoint."""
    return tuple(float(segments[index]["orientation_deg"]) for index in endpoint["segment_ids"])


def describe_accepted_gap(
    gap: dict[str, Any],
    endpoints: dict[int, dict[str, Any]],
    segments: dict[int, dict[str, Any]],
    corridor_axis_local: np.ndarray,
) -> OrientationRelativeGapDescriptor:
    """Build a GT-free orientation-relative descriptor for one gap.

    The free-space direction is the same local center-bearing ray already used
    by EXP-033 to measure continuation.  Its dot product selects one of the two
    gap normals without consulting a branch direction or map.
    """
    first = endpoints[int(gap["endpoint_a"])]
    second = endpoints[int(gap["endpoint_b"])]
    endpoint_a = np.asarray(first["point"], dtype=float)
    endpoint_b = np.asarray(second["point"], dtype=float)
    center = np.asarray(gap.get("gap_center", 0.5 * (endpoint_a + endpoint_b)), dtype=float)
    tangent = _unit(endpoint_b - endpoint_a)
    normal_1 = np.array([-tangent[1], tangent[0]], dtype=float)
    normal_2 = -normal_1
    if "estimated_direction_local" in gap:
        angle = math.radians(float(gap["estimated_direction_local"]))
        free_direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)
    else:
        free_direction = _unit(center)
    opening_normal = normal_1 if float(np.dot(normal_1, free_direction)) >= float(np.dot(normal_2, free_direction)) else normal_2
    corridor_axis = _unit(corridor_axis_local)
    corridor_lateral = np.array([-corridor_axis[1], corridor_axis[0]], dtype=float)
    tangent_angle = _angle_deg(tangent)
    first_angles = _incident_tangent_angles(first, segments)
    second_angles = _incident_tangent_angles(second, segments)
    all_wall_angles = first_angles + second_angles
    boundary_error = min(
        (_axial_difference_deg(angle, tangent_angle) for angle in all_wall_angles),
        default=math.inf,
    )
    # Physical invariant: at least one observed incident wall locally supports
    # the gap-boundary axis.  The tolerance is the frozen EXP-033 corner
    # resolution, independent of corridor or branch orientation.
    geometry_support = boundary_error <= FROZEN_ENDPOINT_FRAME_RESOLUTION_DEG
    accepted_upstream = bool(gap.get("candidate_valid", True))
    if not accepted_upstream:
        rejection = "GAP_NOT_ACCEPTED_UPSTREAM"
    elif not geometry_support:
        rejection = "NO_INCIDENT_WALL_SUPPORT_ALONG_GAP_TANGENT"
    else:
        rejection = "NONE"
    return OrientationRelativeGapDescriptor(
        endpoint_a_local=endpoint_a.copy(),
        endpoint_b_local=endpoint_b.copy(),
        gap_center_local=center.copy(),
        gap_width=float(gap["gap_width"]),
        gap_width_over_W_hat=float(gap.get("gap_width_ratio_W", gap.get("gap_width_over_W_hat", math.nan))),
        gap_tangent_local=tangent,
        gap_normal_candidate_1=normal_1,
        gap_normal_candidate_2=normal_2,
        opening_normal_local=opening_normal,
        free_space_direction_local=_unit(free_direction),
        free_continuation=float(gap.get("continuation_depth", gap.get("free_continuation", math.nan))),
        corridor_axis_local=corridor_axis,
        corridor_lateral_axis_local=corridor_lateral,
        relative_opening_angle_deg=_signed_angle_deg(corridor_axis, opening_normal),
        endpoint_a_type=str(first["endpoint_type"]),
        endpoint_b_type=str(second["endpoint_type"]),
        endpoint_a_wall_tangent_deg=first_angles,
        endpoint_b_wall_tangent_deg=second_angles,
        gap_tangent_deg_local=tangent_angle,
        opening_normal_deg_local=_angle_deg(opening_normal),
        free_space_direction_deg_local=_angle_deg(free_direction),
        normal_free_alignment=float(np.clip(np.dot(opening_normal, _unit(free_direction)), -1.0, 1.0)),
        gap_boundary_wall_alignment_error_deg=boundary_error,
        geometry_support=geometry_support,
        rejection_reason=rejection,
    )


def build_general_branch_candidate(
    candidate_id: str,
    timestamp: float,
    topology_type: str,
    gap: dict[str, Any],
    endpoints: dict[int, dict[str, Any]],
    segments: dict[int, dict[str, Any]],
    corridor_axis_local: np.ndarray,
) -> tuple[OrientationRelativeGapDescriptor, GeneralBranchCandidate | None]:
    """Describe a gap and create a candidate only with physical frame support."""
    descriptor = describe_accepted_gap(gap, endpoints, segments, corridor_axis_local)
    if descriptor.rejection_reason != "NONE":
        return descriptor, None
    # Reliability remains a transparent continuous observable, not a new
    # acceptance threshold or validation state.
    candidate = GeneralBranchCandidate(
        candidate_id=candidate_id,
        timestamp=float(timestamp),
        topology_type=str(topology_type),
        descriptor=descriptor,
        candidate_reliability=descriptor.normal_free_alignment,
    )
    return descriptor, candidate


def self_test() -> None:
    """Verify horizontal, vertical, and angled gaps use one constructor."""
    corridor = np.array([1.0, 0.0])
    for index, angle_deg in enumerate((0.0, 90.0, 37.0)):
        angle = math.radians(angle_deg)
        tangent = np.array([math.cos(angle), math.sin(angle)])
        center = np.array([30.0, 12.0])
        first, second = center - 5.0 * tangent, center + 5.0 * tangent
        endpoints = {
            0: {"point": first, "endpoint_type": "CORNER", "segment_ids": [0]},
            1: {"point": second, "endpoint_type": "WALL_TERMINATION", "segment_ids": [1]},
        }
        segments = {
            0: {"orientation_deg": angle_deg},
            1: {"orientation_deg": angle_deg},
        }
        gap = {
            "endpoint_a": 0,
            "endpoint_b": 1,
            "gap_center": center,
            "gap_width": 10.0,
            "gap_width_ratio_W": 1.0,
            "continuation_depth": 8.0,
            "candidate_valid": True,
        }
        descriptor, candidate = build_general_branch_candidate(
            f"S{index}", 0.0, "COMPLETE", gap, endpoints, segments, corridor
        )
        assert candidate is not None and descriptor.geometry_support
        assert math.isclose(float(np.linalg.norm(descriptor.gap_tangent_local)), 1.0)
        assert math.isclose(float(np.linalg.norm(descriptor.opening_normal_local)), 1.0)
    # An arbitrary diagonal between orthogonal walls has no observed boundary
    # segment along its own diagonal tangent and is not promoted.
    endpoints = {
        0: {"point": np.array([10.0, 0.0]), "endpoint_type": "WALL_TERMINATION", "segment_ids": [0]},
        1: {"point": np.array([20.0, 10.0]), "endpoint_type": "WALL_TERMINATION", "segment_ids": [1]},
    }
    segments = {0: {"orientation_deg": 0.0}, 1: {"orientation_deg": 90.0}}
    gap = {"endpoint_a": 0, "endpoint_b": 1, "gap_center": np.array([15.0, 5.0]), "gap_width": math.sqrt(200.0), "gap_width_ratio_W": 1.4, "continuation_depth": 8.0, "candidate_valid": True}
    descriptor, candidate = build_general_branch_candidate("D", 0.0, "COMPLETE", gap, endpoints, segments, corridor)
    assert candidate is None
    assert descriptor.rejection_reason == "NO_INCIDENT_WALL_SUPPORT_ALONG_GAP_TANGENT"


if __name__ == "__main__":
    self_test()
