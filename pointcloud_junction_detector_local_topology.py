"""
pointcloud_junction_detector.py

General 2D Junction Point-Cloud Opening Detector.

Research separation
-------------------
Simulator-only information:
    - wall geometry
    - Anchor world x/y/yaw

Detector-visible information:
    - Anchor-local LiDAR angle/range only

The detector never receives:
    - expected branch/way count
    - expected branch directions
    - map or wall geometry
    - Anchor global pose
    - Junction coordinates

Problem definition
------------------
Detector input:
    P = {(theta_i, r_i)} for i = 1..N
    where theta_i is the Anchor-local bearing and r_i is the measured range.

Detector output:
    A variable-length list of open angular sectors. Each sector contains
    start_angle, end_angle, center_angle, and width_deg.
    The number of outputs is inferred from the scan; it is never provided.

Local topology output:
    Using only the detected openings plus the robot's recent local motion direction
    expressed in the same Anchor-local frame, the classifier identifies the
    incoming corridor, excludes it from the new exits, and classifies the
    local topology as DEAD_END / CORRIDOR / JUNCTION / UNKNOWN.

Pipeline
--------
Arbitrary test geometry -> ray casting -> Anchor-local P={(theta,r)}
-> circular smoothing -> adaptive open-support extraction
-> circular connected components -> range-gradient boundary refinement
-> opening start/end angles and automatically inferred opening count
-> incoming-corridor matching from local motion direction
-> local topology classification without global map / absolute position
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


EPSILON = 1.0e-10


@dataclass(frozen=True)
class LidarScan:
    """Anchor-local 2D LiDAR scan; no global map/pose is stored."""

    angle_deg: np.ndarray
    range_m: np.ndarray
    hit: np.ndarray
    local_x: np.ndarray
    local_y: np.ndarray
    max_range_m: float

    def detector_input(self) -> tuple[np.ndarray, np.ndarray]:
        """Return only information permitted to the localization-free detector."""
        return self.angle_deg.copy(), self.range_m.copy()

    def valid_local_points(self) -> np.ndarray:
        """Return physical LiDAR returns only in the Anchor-local frame."""
        return np.column_stack((self.local_x[self.hit], self.local_y[self.hit]))


# -----------------------------------------------------------------------------
# Geometry / ray casting
# -----------------------------------------------------------------------------

def cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def ray_segment_distance(
    ray_origin: Sequence[float],
    ray_direction: Sequence[float],
    seg_start: Sequence[float],
    seg_end: Sequence[float],
    eps: float = EPSILON,
) -> Optional[float]:
    """Nearest forward intersection distance of a ray and a finite segment.

    Handles ordinary intersections, zero-length segments, parallel lines, and
    collinear overlap. Returns None when there is no forward intersection.
    """
    o = np.asarray(ray_origin, dtype=float)
    d = np.asarray(ray_direction, dtype=float)
    p = np.asarray(seg_start, dtype=float)
    q = np.asarray(seg_end, dtype=float)

    if any(v.shape != (2,) for v in (o, d, p, q)):
        raise ValueError("ray/segment points must contain exactly two values")

    d_norm = float(np.linalg.norm(d))
    if d_norm <= eps:
        raise ValueError("ray_direction must be non-zero")

    s = q - p
    s_norm = float(np.linalg.norm(s))
    offset = p - o

    if s_norm <= eps:
        if abs(cross2d(offset, d)) > eps * d_norm:
            return None
        t = float(np.dot(offset, d) / (d_norm * d_norm))
        return None if t < -eps else max(0.0, t) * d_norm

    denominator = cross2d(d, s)
    parallel_tolerance = eps * d_norm * s_norm
    if abs(denominator) <= parallel_tolerance:
        if abs(cross2d(offset, d)) > eps * d_norm:
            return None
        projections = np.array(
            [np.dot(p - o, d), np.dot(q - o, d)], dtype=float
        ) / (d_norm * d_norm)
        if float(np.max(projections)) < -eps:
            return None
        return max(0.0, float(np.min(projections))) * d_norm

    t = cross2d(offset, s) / denominator
    u = cross2d(offset, d) / denominator
    if t < -eps or u < -eps or u > 1.0 + eps:
        return None
    return max(0.0, float(t)) * d_norm


def simulate_lidar_scan(
    wall_segments: Sequence[Sequence[Sequence[float]]],
    anchor_xy: Sequence[float],
    *,
    anchor_yaw_deg: float = 0.0,
    angle_min_deg: float = -180.0,
    angle_max_deg: float = 180.0,
    angle_step_deg: float = 1.0,
    max_range_m: float = 6.0,
    noise_std_m: float = 0.0,
    dropout_probability: float = 0.0,
    occlusion_probability: float = 0.0,
    visible_boundary_ratio: float = 1.0,
    seed: Optional[int] = None,
) -> LidarScan:
    """Ray-cast a 2D LiDAR scan and return it in the Anchor-local frame.

    Sensor degradation is applied only after ideal ray casting:
      - ``noise_std_m``: Gaussian range noise on physical returns
      - ``dropout_probability``: independent random loss of physical returns
      - ``occlusion_probability``: probability of one contiguous angular block
        being hidden in this scan
      - ``visible_boundary_ratio``: fraction of physical wall returns retained

    These options belong to the simulator only. The opening detector still
    receives only Anchor-local ``(theta, range)`` samples.
    """
    walls = np.asarray(wall_segments, dtype=float)
    anchor = np.asarray(anchor_xy, dtype=float)

    if walls.ndim != 3 or walls.shape[1:] != (2, 2):
        raise ValueError("wall_segments must have shape (N, 2, 2)")
    if anchor.shape != (2,):
        raise ValueError("anchor_xy must have length 2")
    if angle_step_deg <= 0.0:
        raise ValueError("angle_step_deg must be > 0")
    if angle_max_deg <= angle_min_deg:
        raise ValueError("angle_max_deg must exceed angle_min_deg")
    if angle_max_deg - angle_min_deg > 360.0 + 1.0e-9:
        raise ValueError("LiDAR FOV cannot exceed 360 degrees")
    if max_range_m <= 0.0:
        raise ValueError("max_range_m must be > 0")
    if noise_std_m < 0.0:
        raise ValueError("noise_std_m must be >= 0")
    if not 0.0 <= dropout_probability <= 1.0:
        raise ValueError("dropout_probability must be in [0, 1]")
    if not 0.0 <= occlusion_probability <= 1.0:
        raise ValueError("occlusion_probability must be in [0, 1]")
    if not 0.0 <= visible_boundary_ratio <= 1.0:
        raise ValueError("visible_boundary_ratio must be in [0, 1]")

    angles_local = np.arange(
        angle_min_deg, angle_max_deg, angle_step_deg, dtype=float
    )
    ranges = np.full(angles_local.size, float(max_range_m), dtype=float)
    hits = np.zeros(angles_local.size, dtype=bool)
    yaw_rad = np.deg2rad(float(anchor_yaw_deg))

    for i, local_angle_deg in enumerate(angles_local):
        world_angle = yaw_rad + np.deg2rad(local_angle_deg)
        direction = np.array([np.cos(world_angle), np.sin(world_angle)], dtype=float)
        nearest = float(max_range_m)
        found = False
        for wall in walls:
            distance = ray_segment_distance(anchor, direction, wall[0], wall[1])
            if distance is not None and EPSILON < distance <= nearest:
                nearest = float(distance)
                found = True
        if found:
            ranges[i] = nearest
            hits[i] = True

    rng = np.random.default_rng(seed)
    if noise_std_m > 0.0 and hits.any():
        hit_idx = np.flatnonzero(hits)
        ranges[hit_idx] += rng.normal(0.0, noise_std_m, hit_idx.size)
        ranges[hit_idx] = np.clip(ranges[hit_idx], 0.0, max_range_m)

    # Build one simulator-only loss mask so every degradation mechanism has
    # the same no-return semantics: range=max_range and hit=False.
    drop_mask = np.zeros(ranges.size, dtype=bool)

    # Independent ray dropout. Restrict it to physical returns so an already
    # open/no-return direction is not counted as a synthetic sensor failure.
    if dropout_probability > 0.0 and hits.any():
        drop_mask |= (rng.random(ranges.size) < dropout_probability) & hits

    # Contiguous angular occlusion. This mimics one obstacle/object masking a
    # block of neighboring LiDAR rays rather than isolated random failures.
    if (
        occlusion_probability > 0.0
        and hits.any()
        and rng.random() < occlusion_probability
    ):
        min_span = max(2, ranges.size // 24)
        max_span_exclusive = max(min_span + 1, ranges.size // 5 + 1)
        span = int(rng.integers(min_span, max_span_exclusive))
        start = int(rng.integers(0, ranges.size))
        occluded_indices = (np.arange(span, dtype=int) + start) % ranges.size
        drop_mask[occluded_indices] |= hits[occluded_indices]

    # Partial boundary visibility. Randomly retain only the requested fraction
    # of physical wall returns. This is useful for stress-testing incomplete
    # point clouds without changing the detector interface.
    if visible_boundary_ratio < 1.0 and hits.any():
        hit_indices = np.flatnonzero(hits)
        keep_count = int(np.ceil(visible_boundary_ratio * hit_indices.size))
        keep_count = max(0, min(keep_count, hit_indices.size))
        keep_mask = np.zeros(ranges.size, dtype=bool)
        if keep_count > 0:
            keep_indices = rng.choice(hit_indices, size=keep_count, replace=False)
            keep_mask[keep_indices] = True
        drop_mask |= hits & ~keep_mask

    if np.any(drop_mask):
        hits[drop_mask] = False
        ranges[drop_mask] = max_range_m

    theta = np.deg2rad(angles_local)
    local_x = ranges * np.cos(theta)
    local_y = ranges * np.sin(theta)

    return LidarScan(
        angle_deg=angles_local,
        range_m=ranges,
        hit=hits,
        local_x=local_x,
        local_y=local_y,
        max_range_m=float(max_range_m),
    )


# -----------------------------------------------------------------------------
# General N-way test environment generator (simulator only)
# -----------------------------------------------------------------------------

def _normalize_angle_360(angle_deg: float) -> float:
    return float(angle_deg % 360.0)


def _split_wrapped_interval(start_deg: float, end_deg: float) -> list[tuple[float, float]]:
    """Split a circular [start,end] CCW interval into non-wrapped [0,360] pieces."""
    start = _normalize_angle_360(start_deg)
    width = (end_deg - start_deg) % 360.0
    if width <= EPSILON:
        return []
    end_unwrapped = start + width
    if end_unwrapped <= 360.0 + EPSILON:
        return [(start, min(end_unwrapped, 360.0))]
    return [(start, 360.0), (0.0, end_unwrapped - 360.0)]


def _merge_linear_intervals(
    intervals: Sequence[tuple[float, float]],
    eps: float = 1.0e-9,
) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted((float(a), float(b)) for a, b in intervals)
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + eps:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def make_n_way_junction_walls(
    branch_angles_deg: Sequence[float],
    *,
    corridor_width_m: float = 2.0,
    central_radius_m: float = 1.6,
    branch_length_m: float = 10.0,
    arc_step_deg: float = 2.0,
    close_branch_ends: bool = False,
) -> np.ndarray:
    """Create a radial N-way junction as wall line segments.

    This function may know N because it creates *test ground truth* only.
    The detector never receives N or branch angles.

    Geometry:
      - circular central chamber
      - one straight corridor for each supplied branch direction
      - central-circle boundary is removed only where a branch opens
      - branch ends are open by default
    """
    angles = np.asarray(branch_angles_deg, dtype=float)
    if angles.ndim != 1 or angles.size < 1:
        raise ValueError("branch_angles_deg must be a non-empty 1D sequence")
    if not np.all(np.isfinite(angles)):
        raise ValueError("branch angles must be finite")
    if corridor_width_m <= 0.0:
        raise ValueError("corridor_width_m must be positive")
    if central_radius_m <= corridor_width_m / 2.0:
        raise ValueError("central_radius_m must exceed half the corridor width")
    if branch_length_m <= central_radius_m:
        raise ValueError("branch_length_m must exceed central_radius_m")
    if arc_step_deg <= 0.0:
        raise ValueError("arc_step_deg must be positive")

    # Remove duplicate directions modulo 360.
    normalized = np.mod(angles, 360.0)
    normalized = np.sort(normalized)
    if normalized.size > 1:
        circular_gaps = np.diff(np.r_[normalized, normalized[0] + 360.0])
        if np.min(circular_gaps) < 1.0e-6:
            raise ValueError("branch directions must be unique modulo 360 degrees")

    half_width = corridor_width_m / 2.0
    r0 = float(np.sqrt(central_radius_m**2 - half_width**2))
    half_opening_deg = float(np.rad2deg(np.arcsin(half_width / central_radius_m)))

    # Validate that neighboring openings do not overlap on the central chamber.
    if normalized.size > 1:
        min_center_gap = float(
            np.min(np.diff(np.r_[normalized, normalized[0] + 360.0]))
        )
        if min_center_gap <= 2.0 * half_opening_deg + 1.0e-6:
            raise ValueError(
                "branch openings overlap; increase central_radius_m, "
                "decrease corridor_width_m, or separate branch angles"
            )

    walls: list[list[list[float]]] = []
    opening_intervals: list[tuple[float, float]] = []

    for angle_deg in normalized:
        phi = np.deg2rad(float(angle_deg))
        direction = np.array([np.cos(phi), np.sin(phi)], dtype=float)
        normal = np.array([-np.sin(phi), np.cos(phi)], dtype=float)

        left_start = r0 * direction + half_width * normal
        right_start = r0 * direction - half_width * normal
        left_end = branch_length_m * direction + half_width * normal
        right_end = branch_length_m * direction - half_width * normal

        walls.append([left_start.tolist(), left_end.tolist()])
        walls.append([right_start.tolist(), right_end.tolist()])
        if close_branch_ends:
            walls.append([left_end.tolist(), right_end.tolist()])

        opening_intervals.extend(
            _split_wrapped_interval(
                float(angle_deg - half_opening_deg),
                float(angle_deg + half_opening_deg),
            )
        )

    # Add central circular wall only in angular gaps between branch openings.
    merged_openings = _merge_linear_intervals(opening_intervals)
    complement: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merged_openings:
        if start > cursor + 1.0e-9:
            complement.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < 360.0 - 1.0e-9:
        complement.append((cursor, 360.0))

    for start_deg, end_deg in complement:
        width = end_deg - start_deg
        if width <= 1.0e-9:
            continue
        segments = max(1, int(np.ceil(width / arc_step_deg)))
        arc_angles = np.linspace(start_deg, end_deg, segments + 1)
        points = np.column_stack(
            (
                central_radius_m * np.cos(np.deg2rad(arc_angles)),
                central_radius_m * np.sin(np.deg2rad(arc_angles)),
            )
        )
        for p0, p1 in zip(points[:-1], points[1:]):
            walls.append([p0.tolist(), p1.tolist()])

    return np.asarray(walls, dtype=float)


def evenly_spaced_branch_angles(num_ways: int, rotation_deg: float = 0.0) -> np.ndarray:
    """Convenience function for tests only; detector does not use it."""
    if num_ways < 1:
        raise ValueError("num_ways must be >= 1")
    return rotation_deg + np.arange(num_ways, dtype=float) * (360.0 / num_ways)


# -----------------------------------------------------------------------------
# Detector utilities: angle/range only
# -----------------------------------------------------------------------------

def _normalize_angles(angles_deg: Any) -> Any:
    normalized = (np.asarray(angles_deg) + 180.0) % 360.0 - 180.0
    if np.ndim(angles_deg) == 0:
        return float(normalized)
    return normalized


def _validate_circular_scan(
    angles_deg: Sequence[float], ranges: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.asarray(angles_deg, dtype=float)
    values = np.asarray(ranges, dtype=float)
    if angles.ndim != 1 or values.ndim != 1 or angles.shape != values.shape:
        raise ValueError("angles_deg and ranges must be equal-length 1D arrays")
    if angles.size < 8:
        raise ValueError("at least 8 LiDAR rays are required")
    if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(values)):
        raise ValueError("angles/ranges must contain finite values")
    if np.any(values < 0.0):
        raise ValueError("ranges cannot be negative")
    if np.any(np.diff(angles) <= 0.0):
        raise ValueError("angles_deg must be strictly increasing")

    steps = np.diff(np.r_[angles, angles[0] + 360.0])
    if np.any(steps <= 0.0):
        raise ValueError("angles must describe one circular revolution")
    if abs(float(np.sum(steps)) - 360.0) > max(1.0, 2.0 * float(np.median(steps))):
        raise ValueError("detector expects an approximately 360-degree scan")
    return angles, values, steps


def smooth_ranges(ranges: Sequence[float], window_size: int = 5) -> np.ndarray:
    """Centered circular moving average."""
    values = np.asarray(ranges, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("ranges must be a non-empty 1D sequence")
    if not isinstance(window_size, (int, np.integer)) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if window_size > values.size:
        raise ValueError("window_size cannot exceed scan length")
    half = window_size // 2
    return np.mean([np.roll(values, s) for s in range(-half, half + 1)], axis=0)


def circular_median_filter(ranges: Sequence[float], window_size: int = 5) -> np.ndarray:
    """Robust circular median prefilter for isolated high-range sensor losses.

    A true opening spans multiple adjacent bearings, so it survives a short
    median window. Isolated dropout / partial-visibility spikes surrounded by
    wall returns are suppressed. Only range values are used; no hit mask or
    global information is exposed to the detector.
    """
    values = np.asarray(ranges, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("ranges must be a non-empty 1D sequence")
    if not isinstance(window_size, (int, np.integer)) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if window_size > values.size:
        raise ValueError("window_size cannot exceed scan length")
    if window_size == 1:
        return values.copy()
    half = window_size // 2
    stack = np.vstack([np.roll(values, shift) for shift in range(-half, half + 1)])
    return np.median(stack, axis=0)


def circular_range_gradient(
    angles_deg: Sequence[float], ranges: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Forward circular range gradient in m/degree at ray boundaries."""
    angles, values, steps = _validate_circular_scan(angles_deg, ranges)
    gradient = (np.roll(values, -1) - values) / steps
    boundary_angles = _normalize_angles(angles + 0.5 * steps)
    return boundary_angles, gradient


def _automatic_gradient_threshold(
    gradient: np.ndarray,
    mad_scale: float,
    minimum: float,
) -> float:
    magnitudes = np.abs(gradient)
    median = float(np.median(magnitudes))
    mad = float(np.median(np.abs(magnitudes - median)))
    robust_sigma = 1.4826 * mad
    return max(float(minimum), median + mad_scale * robust_sigma)


def _circular_runs(mask: np.ndarray, value: bool = True) -> list[np.ndarray]:
    """Return circular connected components of a boolean mask as index arrays."""
    mask = np.asarray(mask, dtype=bool)
    n = mask.size
    if n == 0:
        return []
    target = mask == value
    if not np.any(target):
        return []
    if np.all(target):
        return [np.arange(n, dtype=int)]

    starts = np.flatnonzero(target & ~np.roll(target, 1))
    runs: list[np.ndarray] = []
    for start in starts:
        indices = [int(start)]
        cursor = (int(start) + 1) % n
        while target[cursor] and cursor != start:
            indices.append(cursor)
            cursor = (cursor + 1) % n
        runs.append(np.asarray(indices, dtype=int))
    return runs


def _run_width_deg(run: np.ndarray, angular_steps: np.ndarray) -> float:
    return float(np.sum(angular_steps[run]))


def _fill_short_circular_gaps(
    mask: np.ndarray,
    angular_steps: np.ndarray,
    max_gap_deg: float,
) -> np.ndarray:
    """Fill short false runs surrounded by open samples."""
    if max_gap_deg <= 0.0:
        return mask.copy()
    result = np.asarray(mask, dtype=bool).copy()
    if np.all(result) or not np.any(result):
        return result
    for run in _circular_runs(result, value=False):
        if _run_width_deg(run, angular_steps) <= max_gap_deg:
            before = (int(run[0]) - 1) % result.size
            after = (int(run[-1]) + 1) % result.size
            if result[before] and result[after]:
                result[run] = True
    return result


def _boundary_angle_before_ray(
    ray_index: int,
    boundary_angles: np.ndarray,
) -> float:
    return float(boundary_angles[(ray_index - 1) % boundary_angles.size])


def _boundary_angle_after_ray(
    ray_index: int,
    boundary_angles: np.ndarray,
) -> float:
    return float(boundary_angles[ray_index % boundary_angles.size])


def _circular_index_window(center: int, radius: int, n: int) -> np.ndarray:
    offsets = np.arange(-radius, radius + 1, dtype=int)
    return (center + offsets) % n


def _refine_boundary_from_gradient(
    target_gradient_index: int,
    gradient: np.ndarray,
    boundary_angles: np.ndarray,
    *,
    positive: bool,
    search_radius_samples: int,
    minimum_strength: float,
    fallback_angle: float,
) -> tuple[float, float, bool]:
    candidates = _circular_index_window(
        target_gradient_index, search_radius_samples, gradient.size
    )
    local = gradient[candidates]
    if positive:
        best_local = int(np.argmax(local))
        strength = float(local[best_local])
        valid = strength >= minimum_strength
    else:
        best_local = int(np.argmin(local))
        strength = float(-local[best_local])
        valid = strength >= minimum_strength
    if not valid:
        return float(fallback_angle), max(0.0, strength), False
    idx = int(candidates[best_local])
    return float(boundary_angles[idx]), strength, True


def _positive_ccw_width(start_angle: float, end_angle: float) -> float:
    return float((end_angle - start_angle) % 360.0)


def _detect_openings_with_diagnostics(
    angles_deg: Sequence[float],
    ranges: Sequence[float],
    *,
    smoothing_window_size: int = 5,
    median_prefilter_window_size: int = 1,
    wall_reference_quantile: float = 0.25,
    far_range_fraction: float = 0.55,
    merge_gap_deg: float = 3.0,
    min_opening_width_deg: float = 5.0,
    gradient_threshold: Optional[float] = None,
    gradient_mad_scale: float = 4.0,
    min_gradient_threshold: float = 0.05,
    boundary_search_deg: float = 6.0,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Detect an arbitrary number of openings using only local angle/range.

    No expected way count or expected direction appears anywhere in this
    function. The number of outputs is the number of connected open angular
    components found in the current scan.

    Method
    ------
    1) circular moving-average smoothing
    2) infer near-wall reference and far-range ceiling from the scan itself
    3) classify angular samples that are sufficiently far as "open support"
    4) merge only short internal gaps
    5) each remaining circular connected component is one opening candidate
    6) refine candidate start/end with local positive/negative range gradients
    """
    angles, raw, angular_steps = _validate_circular_scan(angles_deg, ranges)

    if smoothing_window_size <= 0 or smoothing_window_size % 2 == 0:
        raise ValueError("smoothing_window_size must be a positive odd integer")
    if median_prefilter_window_size <= 0 or median_prefilter_window_size % 2 == 0:
        raise ValueError("median_prefilter_window_size must be a positive odd integer")
    if not 0.0 <= wall_reference_quantile < 1.0:
        raise ValueError("wall_reference_quantile must be in [0,1)")
    if not 0.0 < far_range_fraction < 1.0:
        raise ValueError("far_range_fraction must be in (0,1)")
    if merge_gap_deg < 0.0:
        raise ValueError("merge_gap_deg must be non-negative")
    if not 0.0 < min_opening_width_deg < 360.0:
        raise ValueError("min_opening_width_deg must be in (0,360)")
    if boundary_search_deg < 0.0:
        raise ValueError("boundary_search_deg must be non-negative")

    # Suppress isolated max-range spikes before averaging. This improves
    # dropout/partial-visibility robustness while preserving broad openings.
    prefiltered = circular_median_filter(raw, median_prefilter_window_size)
    smoothed = smooth_ranges(prefiltered, smoothing_window_size)

    # These are inferred from the scan; the detector is not passed sensor/map metadata.
    wall_reference = float(np.quantile(smoothed, wall_reference_quantile))
    range_ceiling = float(np.max(prefiltered))
    dynamic_span = max(0.0, range_ceiling - wall_reference)

    # If there is no meaningful contrast, no opening can be supported by this baseline.
    if dynamic_span <= 1.0e-6:
        diagnostics = {
            "prefiltered_ranges": prefiltered,
            "prefiltered_ranges": prefiltered,
        "smoothed_ranges": smoothed,
            "open_support_mask": np.zeros(raw.size, dtype=bool),
            "open_threshold": range_ceiling,
            "wall_reference": wall_reference,
            "range_ceiling": range_ceiling,
            "boundary_angles": np.array([], dtype=float),
            "gradient": np.array([], dtype=float),
            "gradient_threshold": 0.0,
            "start_angles": [],
            "end_angles": [],
        }
        return [], diagnostics

    open_threshold = wall_reference + far_range_fraction * dynamic_span
    open_support = smoothed >= open_threshold
    open_support = _fill_short_circular_gaps(
        open_support, angular_steps, merge_gap_deg
    )

    boundary_angles, gradient = circular_range_gradient(angles, smoothed)
    grad_threshold = (
        float(gradient_threshold)
        if gradient_threshold is not None
        else _automatic_gradient_threshold(
            gradient, gradient_mad_scale, min_gradient_threshold
        )
    )

    median_step = float(np.median(angular_steps))
    search_radius_samples = int(np.ceil(boundary_search_deg / median_step))

    openings: list[dict[str, float]] = []
    for run in _circular_runs(open_support, value=True):
        coarse_width = _run_width_deg(run, angular_steps)
        if coarse_width < min_opening_width_deg:
            continue
        if coarse_width >= 359.0:
            # Entire scan is far: there is no observable wall/opening separation.
            continue

        start_ray = int(run[0])
        end_ray = int(run[-1])
        coarse_start = _boundary_angle_before_ray(start_ray, boundary_angles)
        coarse_end = _boundary_angle_after_ray(end_ray, boundary_angles)

        start_grad_index = (start_ray - 1) % gradient.size
        end_grad_index = end_ray % gradient.size

        start_angle, start_strength, start_refined = _refine_boundary_from_gradient(
            start_grad_index,
            gradient,
            boundary_angles,
            positive=True,
            search_radius_samples=search_radius_samples,
            minimum_strength=grad_threshold,
            fallback_angle=coarse_start,
        )
        end_angle, end_strength, end_refined = _refine_boundary_from_gradient(
            end_grad_index,
            gradient,
            boundary_angles,
            positive=False,
            search_radius_samples=search_radius_samples,
            minimum_strength=grad_threshold,
            fallback_angle=coarse_end,
        )

        width = _positive_ccw_width(start_angle, end_angle)
        # Gradient refinement can jump to a neighboring lobe under heavy noise.
        # Fall back to the connected-component boundaries if that becomes implausible.
        if width < min_opening_width_deg or width > min(359.0, coarse_width + 2.0 * boundary_search_deg + 2.0):
            start_angle = coarse_start
            end_angle = coarse_end
            width = _positive_ccw_width(start_angle, end_angle)
            start_refined = False
            end_refined = False

        if width < min_opening_width_deg:
            continue

        center_angle = float(_normalize_angles(start_angle + width / 2.0))
        mean_range = float(np.mean(smoothed[run]))
        peak_range = float(np.max(smoothed[run]))
        contrast_score = float(
            np.clip((mean_range - wall_reference) / max(dynamic_span, EPSILON), 0.0, 1.0)
        )
        boundary_score = float(
            np.clip(
                min(start_strength, end_strength) / max(grad_threshold, EPSILON),
                0.0,
                1.0,
            )
        )
        confidence = float(0.7 * contrast_score + 0.3 * boundary_score)

        openings.append(
            {
                "start_angle": float(_normalize_angles(start_angle)),
                "end_angle": float(_normalize_angles(end_angle)),
                "center_angle": center_angle,
                "width_deg": float(width),
                "mean_range_m": mean_range,
                "peak_range_m": peak_range,
                "confidence": confidence,
                "start_refined": float(start_refined),
                "end_refined": float(end_refined),
            }
        )

    openings.sort(key=lambda item: item["center_angle"])
    diagnostics: dict[str, Any] = {
        "smoothed_ranges": smoothed,
        "open_support_mask": open_support,
        "open_threshold": float(open_threshold),
        "wall_reference": float(wall_reference),
        "range_ceiling": float(range_ceiling),
        "boundary_angles": boundary_angles,
        "gradient": gradient,
        "gradient_threshold": float(grad_threshold),
        "start_angles": [o["start_angle"] for o in openings],
        "end_angles": [o["end_angle"] for o in openings],
    }
    return openings, diagnostics


def detect_openings(
    angles_deg: Sequence[float],
    ranges: Sequence[float],
    **kwargs: Any,
) -> list[dict[str, float]]:
    """Detect open angular sectors from Anchor-local range-vs-angle data only.

    Parameters
    ----------
    angles_deg:
        Anchor-local LiDAR bearing angles. No world heading is accepted here.
    ranges:
        Measured range associated with each angle.

    Returns
    -------
    list of dict
        Variable-length output. Each item contains ``start_angle``,
        ``end_angle``, ``center_angle`` and ``width_deg`` (plus diagnostics
        such as confidence). ``len(result)`` is the detected opening count.

    Notes
    -----
    The detector is intentionally NOT given a branch count, branch labels,
    expected branch directions, wall geometry, map, Anchor global pose, or
    Junction coordinates. 3-way/4-way/5-way are therefore test cases only,
    not algorithm modes.
    """
    openings, _ = _detect_openings_with_diagnostics(angles_deg, ranges, **kwargs)
    return openings


def detect_openings_from_point_cloud(
    point_cloud_theta_r: Sequence[Sequence[float]],
    **kwargs: Any,
) -> list[dict[str, float]]:
    """Detect openings directly from P={(theta_i, r_i)}.

    This is the research-facing API matching the Junction-detector problem
    definition. The input must be an ``(N, 2)`` array-like object whose first
    column is Anchor-local angle [deg] and second column is range [m].

    The samples are sorted by angle before detection. No way-count or geometry
    information is accepted.
    """
    cloud = np.asarray(point_cloud_theta_r, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 2:
        raise ValueError("point_cloud_theta_r must have shape (N, 2): [theta_deg, range_m]")
    if cloud.shape[0] < 8:
        raise ValueError("at least 8 point-cloud samples are required")
    if not np.all(np.isfinite(cloud)):
        raise ValueError("point cloud must contain only finite values")

    order = np.argsort(cloud[:, 0])
    angles = cloud[order, 0]
    ranges = cloud[order, 1]
    return detect_openings(angles, ranges, **kwargs)


# -----------------------------------------------------------------------------
# Local topology classification: detected openings + local motion only
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class LocalTopologyResult:
    """Localization-free local topology classification result.

    The classifier receives no global map, absolute position, Junction pose,
    branch labels, branch count, or expected branch directions.  It only uses:

      1) the openings already inferred from the local LiDAR scan, and
      2) the robot's recent local motion direction expressed in the SAME local frame.

    ``incoming_opening_index`` points to the opening that leads back toward the
    path the robot came from.  Every other opening is considered a locally
    observable outgoing option.
    """

    topology: str
    is_junction: bool
    total_opening_count: int
    incoming_opening_index: Optional[int]
    outgoing_opening_indices: tuple[int, ...]
    outgoing_opening_count: int
    local_motion_direction_deg: float
    incoming_direction_local_deg: float
    incoming_match_error_deg: Optional[float]


def circular_angular_distance_deg(angle_a: float, angle_b: float) -> float:
    """Smallest unsigned angular distance in degrees, in [0, 180]."""
    return abs(float(_normalize_angles(float(angle_a) - float(angle_b))))


def _angle_inside_ccw_sector(angle_deg: float, start_deg: float, end_deg: float) -> bool:
    """Return True when angle lies inside the circular CCW [start,end] sector."""
    width = _positive_ccw_width(float(start_deg), float(end_deg))
    offset = _positive_ccw_width(float(start_deg), float(angle_deg))
    return offset <= width + 1.0e-9


def _angular_distance_to_sector_deg(
    angle_deg: float,
    start_deg: float,
    end_deg: float,
) -> float:
    """Angular distance from an angle to a wrapped opening sector."""
    if _angle_inside_ccw_sector(angle_deg, start_deg, end_deg):
        return 0.0
    return min(
        circular_angular_distance_deg(angle_deg, start_deg),
        circular_angular_distance_deg(angle_deg, end_deg),
    )


def classify_local_topology(
    openings: Sequence[dict[str, float]],
    *,
    local_motion_direction_deg: float,
    incoming_tolerance_deg: float = 20.0,
    min_new_exits_for_junction: int = 2,
) -> LocalTopologyResult:
    """Classify DEAD_END / CORRIDOR / JUNCTION using only local information.

    Parameters
    ----------
    openings:
        Output of ``detect_openings(...)``.  No ground-truth branch data is
        accepted here.
    local_motion_direction_deg:
        Direction in which the robot has recently been moving, expressed in
        the same Anchor-local angular frame as the LiDAR scan.  In a real robot
        this can come from recent local odometry / velocity / trajectory.  It
        does NOT need a global map or absolute position.
    incoming_tolerance_deg:
        Maximum angular gap allowed between the expected back direction and a
        detected opening sector.  The sector itself has zero error when the
        expected direction lies inside it.
    min_new_exits_for_junction:
        Number of NEW outgoing openings (after removing the incoming corridor)
        required to call the location a Junction.  Default 2 means:

            incoming + 0 new exits -> DEAD_END
            incoming + 1 new exit  -> CORRIDOR
            incoming + >=2 exits   -> JUNCTION

    Notes
    -----
    The expected incoming direction is simply 180 degrees opposite the recent
    local motion direction in the local frame.  This is a local geometric relation,
    not a global localization requirement.
    """
    if incoming_tolerance_deg < 0.0 or incoming_tolerance_deg > 180.0:
        raise ValueError("incoming_tolerance_deg must be in [0, 180]")
    if min_new_exits_for_junction < 2:
        raise ValueError("min_new_exits_for_junction must be >= 2")

    travel_local = float(_normalize_angles(local_motion_direction_deg))
    expected_incoming = float(_normalize_angles(travel_local + 180.0))
    total = len(openings)

    if total == 0:
        return LocalTopologyResult(
            topology="UNKNOWN",
            is_junction=False,
            total_opening_count=0,
            incoming_opening_index=None,
            outgoing_opening_indices=tuple(),
            outgoing_opening_count=0,
            local_motion_direction_deg=travel_local,
            incoming_direction_local_deg=expected_incoming,
            incoming_match_error_deg=None,
        )

    sector_errors = np.asarray(
        [
            _angular_distance_to_sector_deg(
                expected_incoming,
                opening["start_angle"],
                opening["end_angle"],
            )
            for opening in openings
        ],
        dtype=float,
    )
    best_idx = int(np.argmin(sector_errors))
    best_error = float(sector_errors[best_idx])

    # If none of the detected sectors is sufficiently close to the direction
    # back toward the travelled path, we avoid forcing a topology label.
    if best_error > incoming_tolerance_deg:
        return LocalTopologyResult(
            topology="UNKNOWN",
            is_junction=False,
            total_opening_count=total,
            incoming_opening_index=None,
            outgoing_opening_indices=tuple(range(total)),
            outgoing_opening_count=total,
            local_motion_direction_deg=travel_local,
            incoming_direction_local_deg=expected_incoming,
            incoming_match_error_deg=best_error,
        )

    incoming_idx = best_idx
    outgoing_indices = tuple(i for i in range(total) if i != incoming_idx)
    outgoing_count = len(outgoing_indices)

    if outgoing_count == 0:
        topology = "DEAD_END"
    elif outgoing_count < min_new_exits_for_junction:
        topology = "CORRIDOR"
    else:
        topology = "JUNCTION"

    return LocalTopologyResult(
        topology=topology,
        is_junction=(topology == "JUNCTION"),
        total_opening_count=total,
        incoming_opening_index=incoming_idx,
        outgoing_opening_indices=outgoing_indices,
        outgoing_opening_count=outgoing_count,
        local_motion_direction_deg=travel_local,
        incoming_direction_local_deg=expected_incoming,
        incoming_match_error_deg=best_error,
    )


def simulated_local_motion_direction(
    branch_angles_deg: Sequence[float],
    *,
    incoming_branch_index: int,
    anchor_yaw_deg: float,
) -> float:
    """TEST-ONLY helper that simulates recent local motion into the Junction.

    ``branch_angles_deg`` and ``incoming_branch_index`` are ground truth used
    only by the simulator/test harness.  They are NOT passed to
    ``classify_local_topology``.  A real robot should replace this helper with
    its measured recent local local motion direction.

    If a branch points outward from the Junction at angle ``b``, travelling
    from that branch into the Junction points at ``b + 180 deg`` in world
    coordinates.  Subtracting the LiDAR-frame yaw expresses that motion in the
    Anchor-local frame.
    """
    angles = np.asarray(branch_angles_deg, dtype=float)
    if angles.ndim != 1 or angles.size == 0:
        raise ValueError("branch_angles_deg must be a non-empty 1D sequence")
    if not 0 <= incoming_branch_index < angles.size:
        raise ValueError(
            f"incoming_branch_index must be in [0, {angles.size - 1}]"
        )
    travel_world = float(angles[incoming_branch_index] + 180.0)
    return float(_normalize_angles(travel_world - float(anchor_yaw_deg)))


# -----------------------------------------------------------------------------
# Save / visualization
# -----------------------------------------------------------------------------

def save_local_scan_csv(scan: LidarScan, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_deg", "range_m", "hit", "local_x_m", "local_y_m"])
        for a, r, h, x, y in zip(
            scan.angle_deg, scan.range_m, scan.hit, scan.local_x, scan.local_y
        ):
            writer.writerow(
                [f"{a:.6f}", f"{r:.6f}", int(h), f"{x:.6f}", f"{y:.6f}"]
            )


def plot_results(
    walls: np.ndarray,
    anchor_xy: Sequence[float],
    scan: LidarScan,
    openings: Sequence[dict[str, float]],
    diagnostics: dict[str, Any],
    *,
    topology_result: Optional[LocalTopologyResult] = None,
    anchor_yaw_deg: float = 0.0,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    anchor = np.asarray(anchor_xy, dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    world_ax, cloud_ax, range_ax = axes

    for wall in walls:
        world_ax.plot(
            [wall[0, 0], wall[1, 0]], [wall[0, 1], wall[1, 1]], linewidth=1.7
        )
    world_ax.scatter(anchor[0], anchor[1], marker="x", s=80, label="Anchor")
    yaw = np.deg2rad(anchor_yaw_deg)
    world_ax.arrow(
        anchor[0],
        anchor[1],
        0.7 * np.cos(yaw),
        0.7 * np.sin(yaw),
        width=0.015,
        length_includes_head=True,
    )
    world_ax.set_aspect("equal", adjustable="box")
    world_ax.set_title("A. Simulator ground truth")
    world_ax.set_xlabel("world x [m]")
    world_ax.set_ylabel("world y [m]")
    world_ax.grid(True, alpha=0.3)
    world_ax.legend()

    cloud_ax.scatter(
        scan.local_x[scan.hit], scan.local_y[scan.hit], s=9, label="LiDAR returns"
    )
    if np.any(~scan.hit):
        cloud_ax.scatter(
            scan.local_x[~scan.hit],
            scan.local_y[~scan.hit],
            s=5,
            alpha=0.15,
            label="No return / max range",
        )
    cloud_ax.scatter([0.0], [0.0], marker="x", s=80, label="Anchor local origin")
    for i, opening in enumerate(openings):
        theta = np.deg2rad(opening["center_angle"])
        endpoint = 0.85 * scan.max_range_m * np.array([np.cos(theta), np.sin(theta)])
        cloud_ax.plot(
            [0.0, endpoint[0]],
            [0.0, endpoint[1]],
            linestyle="--",
            linewidth=1.2,
            label="Detected opening center" if i == 0 else None,
        )
    if topology_result is not None:
        incoming_theta = np.deg2rad(topology_result.incoming_direction_local_deg)
        incoming_endpoint = 0.72 * scan.max_range_m * np.array(
            [np.cos(incoming_theta), np.sin(incoming_theta)]
        )
        cloud_ax.plot(
            [0.0, incoming_endpoint[0]],
            [0.0, incoming_endpoint[1]],
            linestyle=":",
            linewidth=1.5,
            label="Expected incoming direction",
        )

    cloud_ax.set_aspect("equal", adjustable="box")
    if topology_result is None:
        cloud_ax.set_title("B. Anchor-local point cloud")
    else:
        cloud_ax.set_title(
            f"B. Anchor-local point cloud | {topology_result.topology}"
        )
    cloud_ax.set_xlabel("local x [m]")
    cloud_ax.set_ylabel("local y [m]")
    cloud_ax.grid(True, alpha=0.3)
    cloud_ax.legend(fontsize=8)

    smoothed = np.asarray(diagnostics["smoothed_ranges"], dtype=float)
    open_mask = np.asarray(diagnostics["open_support_mask"], dtype=bool)
    range_ax.plot(scan.angle_deg, scan.range_m, linewidth=0.8, alpha=0.45, label="raw")
    range_ax.plot(scan.angle_deg, smoothed, linewidth=1.4, label="smoothed")
    range_ax.axhline(
        diagnostics["open_threshold"], linestyle="--", linewidth=1.1, label="adaptive open threshold"
    )
    if np.any(open_mask):
        range_ax.scatter(
            scan.angle_deg[open_mask],
            smoothed[open_mask],
            s=8,
            alpha=0.35,
            label="open support",
        )

    for i, opening in enumerate(openings):
        start = opening["start_angle"]
        end = opening["end_angle"]
        label = "detected opening" if i == 0 else None
        if start <= end:
            range_ax.axvspan(start, end, alpha=0.10, label=label)
        else:
            range_ax.axvspan(start, 180.0, alpha=0.10, label=label)
            range_ax.axvspan(-180.0, end, alpha=0.10)

    gradient_ax = range_ax.twinx()
    gradient_ax.plot(
        diagnostics["boundary_angles"],
        diagnostics["gradient"],
        linewidth=0.7,
        alpha=0.28,
        label="range gradient",
    )
    threshold = float(diagnostics["gradient_threshold"])
    gradient_ax.axhline(threshold, linestyle=":", linewidth=1.0)
    gradient_ax.axhline(-threshold, linestyle=":", linewidth=1.0)
    gradient_ax.set_ylabel("range gradient [m/deg]")

    range_ax.set_title("C. Detector-visible range profile")
    range_ax.set_xlabel("local angle [deg]")
    range_ax.set_ylabel("range [m]")
    range_ax.set_xlim(float(scan.angle_deg[0]), float(scan.angle_deg[-1]))
    range_ax.set_ylim(0.0, scan.max_range_m * 1.05)
    range_ax.grid(True, alpha=0.3)
    h1, l1 = range_ax.get_legend_handles_labels()
    h2, l2 = gradient_ax.get_legend_handles_labels()
    range_ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")

    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"[saved] {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# Evaluation / benchmark / failure-condition analysis (simulator side only)
# -----------------------------------------------------------------------------

def _sector_iou_deg(a: dict[str, float], b: dict[str, float]) -> float:
    """Exact angular IoU for two circular sectors whose widths are < 360 deg."""
    sa = _normalize_angle_360(float(a["start_angle"]))
    sb = _normalize_angle_360(float(b["start_angle"]))
    wa = float(a["width_deg"])
    wb = float(b["width_deg"])
    if wa <= 0.0 or wb <= 0.0:
        return 0.0
    best_intersection = 0.0
    for shift in (-360.0, 0.0, 360.0):
        a0, a1 = sa, sa + wa
        b0, b1 = sb + shift, sb + shift + wb
        best_intersection = max(best_intersection, max(0.0, min(a1, b1) - max(a0, b0)))
    union = wa + wb - best_intersection
    return 0.0 if union <= EPSILON else float(best_intersection / union)


def physical_mouth_openings_from_geometry(
    branch_angles_deg: Sequence[float],
    *,
    anchor_xy: Sequence[float],
    anchor_yaw_deg: float,
    corridor_width_m: float,
    central_radius_m: float,
) -> list[dict[str, float]]:
    """Physical branch-mouth sectors used ONLY as a secondary diagnostic.

    Each physical branch mouth is represented by the two points where its side
    walls meet the central chamber.  Those points are projected into the same
    Anchor-local angular frame as the LiDAR scan.  This keeps evaluation valid
    for a shifted Anchor and arbitrary local yaw without leaking geometry into
    the detector.
    """
    anchor = np.asarray(anchor_xy, dtype=float)
    if anchor.shape != (2,):
        raise ValueError("anchor_xy must have length 2")
    half_width = float(corridor_width_m) / 2.0
    if central_radius_m <= half_width:
        raise ValueError("central_radius_m must exceed half corridor width")
    r0 = float(np.sqrt(float(central_radius_m) ** 2 - half_width ** 2))
    result: list[dict[str, float]] = []
    for b in np.asarray(branch_angles_deg, dtype=float):
        phi = np.deg2rad(float(b))
        d = np.array([np.cos(phi), np.sin(phi)], dtype=float)
        n = np.array([-np.sin(phi), np.cos(phi)], dtype=float)
        p1 = r0 * d + half_width * n
        p2 = r0 * d - half_width * n
        pm = r0 * d

        def local_angle(point: np.ndarray) -> float:
            v = point - anchor
            return float(_normalize_angles(np.rad2deg(np.arctan2(v[1], v[0])) - anchor_yaw_deg))

        center = local_angle(pm)
        e1 = local_angle(p1)
        e2 = local_angle(p2)
        o1 = float(_normalize_angles(e1 - center))
        o2 = float(_normalize_angles(e2 - center))
        # For valid radial branch mouths, the center ray lies between endpoints.
        if abs(o1 - o2) > 180.0:
            if o1 < 0.0:
                o1 += 360.0
            else:
                o2 += 360.0
        lo, hi = sorted((o1, o2))
        start = float(_normalize_angles(center + lo))
        end = float(_normalize_angles(center + hi))
        width = _positive_ccw_width(start, end)
        if width > 180.0:
            # Select the shorter physical branch-mouth sector.
            start, end = end, start
            width = 360.0 - width
        result.append(
            {
                "start_angle": start,
                "end_angle": end,
                "center_angle": center,
                "width_deg": float(width),
            }
        )
    result.sort(key=lambda x: x["center_angle"])
    return result



def _interpolate_threshold_crossing_angle(
    angle_a: float,
    range_a: float,
    angle_b: float,
    range_b: float,
    threshold: float,
) -> float:
    """Interpolate a circular angular threshold crossing between adjacent rays."""
    delta = (float(angle_b) - float(angle_a)) % 360.0
    if delta > 180.0:
        delta -= 360.0
    denom = float(range_b) - float(range_a)
    frac = 0.5 if abs(denom) <= EPSILON else float(np.clip((threshold - float(range_a)) / denom, 0.0, 1.0))
    return float(_normalize_angles(float(angle_a) + frac * delta))


def ground_truth_openings_from_ideal_scan(
    walls: np.ndarray,
    *,
    anchor_xy: Sequence[float],
    anchor_yaw_deg: float,
    angle_step_deg: float,
    max_range_m: float,
    wall_reference_quantile: float = 0.25,
    far_range_fraction: float = 0.55,
    min_opening_width_deg: float = 5.0,
) -> tuple[list[dict[str, float]], float]:
    """Simulator-only GT for the detector's observable long-free-path sector.

    The current range-profile detector estimates bearings that provide a
    sufficiently long free path. It does not estimate the entire physical
    corridor mouth. Primary boundary accuracy is therefore measured against an
    ideal noise-free LiDAR scan using the same declared free-path definition.
    Geometry remains evaluation-only and is never fed to the detector.
    """
    ideal = simulate_lidar_scan(
        walls, anchor_xy, anchor_yaw_deg=anchor_yaw_deg,
        angle_step_deg=angle_step_deg, max_range_m=max_range_m,
        noise_std_m=0.0, dropout_probability=0.0, occlusion_probability=0.0,
        visible_boundary_ratio=1.0, seed=0,
    )
    angles = np.asarray(ideal.angle_deg, dtype=float)
    ranges = np.asarray(ideal.range_m, dtype=float)
    _, _, angular_steps = _validate_circular_scan(angles, ranges)
    wall_reference = float(np.quantile(ranges, wall_reference_quantile))
    ceiling = float(np.max(ranges))
    threshold = wall_reference + far_range_fraction * max(0.0, ceiling - wall_reference)
    mask = ranges >= threshold
    sectors: list[dict[str, float]] = []
    for run in _circular_runs(mask, value=True):
        if _run_width_deg(run, angular_steps) < min_opening_width_deg:
            continue
        first, last = int(run[0]), int(run[-1])
        prev_idx, next_idx = (first - 1) % ranges.size, (last + 1) % ranges.size
        start = _interpolate_threshold_crossing_angle(
            angles[prev_idx], ranges[prev_idx], angles[first], ranges[first], threshold
        )
        end = _interpolate_threshold_crossing_angle(
            angles[last], ranges[last], angles[next_idx], ranges[next_idx], threshold
        )
        width = _positive_ccw_width(start, end)
        if width < min_opening_width_deg or width >= 359.0:
            continue
        sectors.append({
            "start_angle": start,
            "end_angle": end,
            "center_angle": float(_normalize_angles(start + width / 2.0)),
            "width_deg": float(width),
        })
    sectors.sort(key=lambda item: item["center_angle"])
    return sectors, float(threshold)


# Compatibility alias. This is the physical-mouth diagnostic, not primary GT.
def ground_truth_openings_from_geometry(*args: Any, **kwargs: Any) -> list[dict[str, float]]:
    return physical_mouth_openings_from_geometry(*args, **kwargs)

def _match_openings(
    ground_truth: Sequence[dict[str, float]],
    detected: Sequence[dict[str, float]],
) -> list[tuple[int, int, float]]:
    """One-to-one greedy matching ranked by IoU then center-angle agreement."""
    pairs: list[tuple[float, float, int, int]] = []
    for gi, gt in enumerate(ground_truth):
        for di, det in enumerate(detected):
            iou = _sector_iou_deg(gt, det)
            center_err = circular_angular_distance_deg(gt["center_angle"], det["center_angle"])
            pairs.append((-iou, center_err, gi, di))
    pairs.sort()
    used_g: set[int] = set()
    used_d: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for neg_iou, _center_err, gi, di in pairs:
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi)
        used_d.add(di)
        matches.append((gi, di, -neg_iou))
        if len(matches) >= min(len(ground_truth), len(detected)):
            break
    return matches


def evaluate_detection(
    ground_truth: Sequence[dict[str, float]],
    detected: Sequence[dict[str, float]],
    *,
    expected_topology: Optional[str] = None,
    detected_topology: Optional[str] = None,
    min_acceptable_iou: float = 0.50,
) -> dict[str, Any]:
    """Compare detected openings with simulator-only geometric ground truth."""
    matches = _match_openings(ground_truth, detected)
    gt_n, det_n = len(ground_truth), len(detected)
    matched = len(matches)
    fp = max(0, det_n - matched)
    fn = max(0, gt_n - matched)
    precision = matched / det_n if det_n else (1.0 if gt_n == 0 else 0.0)
    recall = matched / gt_n if gt_n else (1.0 if det_n == 0 else 0.0)
    f1 = 0.0 if precision + recall <= EPSILON else 2.0 * precision * recall / (precision + recall)

    ious: list[float] = []
    center_errors: list[float] = []
    start_errors: list[float] = []
    end_errors: list[float] = []
    for gi, di, iou in matches:
        gt, det = ground_truth[gi], detected[di]
        ious.append(float(iou))
        center_errors.append(circular_angular_distance_deg(gt["center_angle"], det["center_angle"]))
        start_errors.append(circular_angular_distance_deg(gt["start_angle"], det["start_angle"]))
        end_errors.append(circular_angular_distance_deg(gt["end_angle"], det["end_angle"]))

    def mean_or_nan(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    failure_reasons: list[str] = []
    if gt_n != det_n:
        failure_reasons.append("opening_count_mismatch")
    if fp > 0:
        failure_reasons.append("false_positive_opening")
    if fn > 0:
        failure_reasons.append("missed_opening")
    if ious and min(ious) < min_acceptable_iou:
        failure_reasons.append("low_opening_iou")
    topology_correct: Optional[bool] = None
    if expected_topology is not None and detected_topology is not None:
        topology_correct = expected_topology == detected_topology
        if not topology_correct:
            failure_reasons.append("topology_mismatch")

    return {
        "ground_truth_count": gt_n,
        "detected_count": det_n,
        "count_correct": gt_n == det_n,
        "matched_openings": matched,
        "false_positive": fp,
        "false_negative": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou": mean_or_nan(ious),
        "min_iou": float(min(ious)) if ious else float("nan"),
        "center_mae_deg": mean_or_nan(center_errors),
        "start_mae_deg": mean_or_nan(start_errors),
        "end_mae_deg": mean_or_nan(end_errors),
        "boundary_mae_deg": mean_or_nan(start_errors + end_errors),
        "expected_topology": expected_topology,
        "detected_topology": detected_topology,
        "topology_correct": topology_correct,
        "failure_reasons": ";".join(failure_reasons) if failure_reasons else "NONE",
    }


def _expected_topology_from_way_count(way_count: int) -> str:
    if way_count <= 0:
        return "UNKNOWN"
    new_exits = max(0, way_count - 1)
    if new_exits == 0:
        return "DEAD_END"
    if new_exits == 1:
        return "CORRIDOR"
    return "JUNCTION"


def _evaluate_case(
    branch_angles: Sequence[float],
    *,
    anchor_xy: tuple[float, float],
    anchor_yaw_deg: float,
    corridor_width_m: float,
    central_radius_m: float,
    branch_length_m: float,
    max_range_m: float,
    noise_std_m: float,
    dropout_probability: float,
    occlusion_probability: float,
    visible_boundary_ratio: float,
    angle_step_deg: float,
    incoming_branch_index: int,
    incoming_tolerance_deg: float,
    seed: int,
    min_acceptable_iou: float,
) -> dict[str, Any]:
    _walls, _scan, openings, _diag = run_case(
        branch_angles,
        anchor_xy=anchor_xy,
        anchor_yaw_deg=anchor_yaw_deg,
        corridor_width_m=corridor_width_m,
        central_radius_m=central_radius_m,
        branch_length_m=branch_length_m,
        max_range_m=max_range_m,
        noise_std_m=noise_std_m,
        dropout_probability=dropout_probability,
        occlusion_probability=occlusion_probability,
        visible_boundary_ratio=visible_boundary_ratio,
        angle_step_deg=angle_step_deg,
        seed=seed,
    )
    motion = simulated_local_motion_direction(
        branch_angles,
        incoming_branch_index=incoming_branch_index,
        anchor_yaw_deg=anchor_yaw_deg,
    )
    topology = classify_local_topology(
        openings,
        local_motion_direction_deg=motion,
        incoming_tolerance_deg=incoming_tolerance_deg,
    )
    gt, evaluation_free_path_threshold_m = ground_truth_openings_from_ideal_scan(
        _walls,
        anchor_xy=anchor_xy,
        anchor_yaw_deg=anchor_yaw_deg,
        angle_step_deg=angle_step_deg,
        max_range_m=max_range_m,
    )
    metrics = evaluate_detection(
        gt,
        openings,
        expected_topology=_expected_topology_from_way_count(len(branch_angles)),
        detected_topology=topology.topology,
        min_acceptable_iou=min_acceptable_iou,
    )
    mouth_gt = physical_mouth_openings_from_geometry(
        branch_angles,
        anchor_xy=anchor_xy,
        anchor_yaw_deg=anchor_yaw_deg,
        corridor_width_m=corridor_width_m,
        central_radius_m=central_radius_m,
    )
    mouth_metrics = evaluate_detection(mouth_gt, openings, min_acceptable_iou=0.0)
    metrics.update({
        "evaluation_boundary_definition": "ideal_free_path_sector",
        "evaluation_free_path_threshold_m": evaluation_free_path_threshold_m,
        "physical_mouth_mean_iou": mouth_metrics["mean_iou"],
        "physical_mouth_boundary_mae_deg": mouth_metrics["boundary_mae_deg"],
    })
    metrics.update(
        {
            "seed": seed,
            "way_count": len(branch_angles),
            "noise_std_m": noise_std_m,
            "dropout_probability": dropout_probability,
            "occlusion_probability": occlusion_probability,
            "visible_boundary_ratio": visible_boundary_ratio,
            "anchor_x": anchor_xy[0],
            "anchor_y": anchor_xy[1],
            "anchor_yaw_deg": anchor_yaw_deg,
        }
    )
    return metrics


def _write_metrics_csv(rows: Sequence[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved metrics] {path}")


def _summarize_rows(rows: Sequence[dict[str, Any]], label: str) -> None:
    if not rows:
        return
    count_acc = float(np.mean([bool(r["count_correct"]) for r in rows]))
    topo_vals = [r["topology_correct"] for r in rows if r["topology_correct"] is not None]
    topo_acc = float(np.mean(topo_vals)) if topo_vals else float("nan")
    mean_ious = [float(r["mean_iou"]) for r in rows if np.isfinite(float(r["mean_iou"]))]
    center_mae = [float(r["center_mae_deg"]) for r in rows if np.isfinite(float(r["center_mae_deg"]))]
    boundary_mae = [float(r["boundary_mae_deg"]) for r in rows if np.isfinite(float(r["boundary_mae_deg"]))]
    failures = [r for r in rows if r["failure_reasons"] != "NONE"]
    print(f"\n[{label}] runs={len(rows)}")
    print(f"  opening-count accuracy : {100.0 * count_acc:.1f}%")
    print(f"  topology accuracy      : {100.0 * topo_acc:.1f}%" if np.isfinite(topo_acc) else "  topology accuracy      : n/a")
    print(f"  free-sector mean IoU   : {np.mean(mean_ious):.3f}" if mean_ious else "  free-sector mean IoU   : n/a")
    print(f"  center-angle MAE       : {np.mean(center_mae):.2f} deg" if center_mae else "  center-angle MAE       : n/a")
    print(f"  free-sector boundary MAE: {np.mean(boundary_mae):.2f} deg" if boundary_mae else "  free-sector boundary MAE: n/a")
    mouth_mae = [float(r.get("physical_mouth_boundary_mae_deg", float("nan"))) for r in rows]
    mouth_mae = [v for v in mouth_mae if np.isfinite(v)]
    print(f"  physical-mouth boundary MAE (diagnostic): {np.mean(mouth_mae):.2f} deg" if mouth_mae else "  physical-mouth boundary MAE (diagnostic): n/a")
    print(f"  runs with any failure  : {len(failures)}/{len(rows)}")
    if failures:
        first = failures[0]
        print(f"  first failure          : way={first['way_count']}, seed={first['seed']}, reasons={first['failure_reasons']}")


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    ways = [int(x.strip()) for x in args.benchmark_ways.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    for n in ways:
        central_radius = max(float(args.central_radius), 0.55 * n)
        branch_angles = evenly_spaced_branch_angles(n, rotation_deg=args.rotation)
        for k in range(args.runs):
            rows.append(
                _evaluate_case(
                    branch_angles,
                    anchor_xy=(args.anchor_x, args.anchor_y),
                    anchor_yaw_deg=args.anchor_yaw,
                    corridor_width_m=args.corridor_width,
                    central_radius_m=central_radius,
                    branch_length_m=args.branch_length,
                    max_range_m=args.max_range,
                    noise_std_m=args.noise,
                    dropout_probability=args.dropout,
                    occlusion_probability=args.occlusion,
                    visible_boundary_ratio=args.visible_boundary_ratio,
                    angle_step_deg=args.angle_step,
                    incoming_branch_index=min(args.incoming_branch_index, n - 1),
                    incoming_tolerance_deg=args.incoming_tolerance,
                    seed=args.seed + k,
                    min_acceptable_iou=args.min_acceptable_iou,
                )
            )
    _summarize_rows(rows, "benchmark")
    return rows


def run_failure_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Ablation-style sweep to expose detector failure conditions."""
    conditions = [
        ("baseline", 0.0, 0.0, 0.0, 1.0, (0.0, 0.0)),
        ("noise_0.03", 0.03, 0.0, 0.0, 1.0, (0.0, 0.0)),
        ("noise_0.08", 0.08, 0.0, 0.0, 1.0, (0.0, 0.0)),
        ("dropout_0.05", 0.0, 0.05, 0.0, 1.0, (0.0, 0.0)),
        ("dropout_0.15", 0.0, 0.15, 0.0, 1.0, (0.0, 0.0)),
        ("occlusion_0.40", 0.0, 0.0, 0.40, 1.0, (0.0, 0.0)),
        ("occlusion_0.80", 0.0, 0.0, 0.80, 1.0, (0.0, 0.0)),
        ("visibility_0.90", 0.0, 0.0, 0.0, 0.90, (0.0, 0.0)),
        ("visibility_0.70", 0.0, 0.0, 0.0, 0.70, (0.0, 0.0)),
        ("anchor_offset", 0.0, 0.0, 0.0, 1.0, (0.35, -0.25)),
        ("combined_mild", 0.03, 0.05, 0.40, 0.90, (0.20, -0.15)),
        ("combined_hard", 0.08, 0.15, 0.80, 0.70, (0.35, -0.25)),
    ]
    ways = [int(x.strip()) for x in args.benchmark_ways.split(",") if x.strip()]
    all_rows: list[dict[str, Any]] = []
    for label, noise, dropout, occlusion, visibility, offset in conditions:
        rows: list[dict[str, Any]] = []
        for n in ways:
            central_radius = max(float(args.central_radius), 0.55 * n)
            branch_angles = evenly_spaced_branch_angles(n, rotation_deg=args.rotation)
            for k in range(args.runs):
                row = _evaluate_case(
                    branch_angles,
                    anchor_xy=(args.anchor_x + offset[0], args.anchor_y + offset[1]),
                    anchor_yaw_deg=args.anchor_yaw,
                    corridor_width_m=args.corridor_width,
                    central_radius_m=central_radius,
                    branch_length_m=args.branch_length,
                    max_range_m=args.max_range,
                    noise_std_m=noise,
                    dropout_probability=dropout,
                    occlusion_probability=occlusion,
                    visible_boundary_ratio=visibility,
                    angle_step_deg=args.angle_step,
                    incoming_branch_index=min(args.incoming_branch_index, n - 1),
                    incoming_tolerance_deg=args.incoming_tolerance,
                    seed=args.seed + k,
                    min_acceptable_iou=args.min_acceptable_iou,
                )
                row["condition"] = label
                rows.append(row)
                all_rows.append(row)
        _summarize_rows(rows, label)
    return all_rows

# -----------------------------------------------------------------------------
# CLI / regression test
# -----------------------------------------------------------------------------

def run_case(
    branch_angles_deg: Sequence[float],
    *,
    anchor_xy: tuple[float, float] = (0.0, 0.0),
    anchor_yaw_deg: float = 0.0,
    corridor_width_m: float = 2.0,
    central_radius_m: float = 1.6,
    branch_length_m: float = 10.0,
    max_range_m: float = 6.0,
    noise_std_m: float = 0.0,
    dropout_probability: float = 0.0,
    occlusion_probability: float = 0.0,
    visible_boundary_ratio: float = 1.0,
    angle_step_deg: float = 1.0,
    seed: int = 7,
) -> tuple[np.ndarray, LidarScan, list[dict[str, float]], dict[str, Any]]:
    walls = make_n_way_junction_walls(
        branch_angles_deg,
        corridor_width_m=corridor_width_m,
        central_radius_m=central_radius_m,
        branch_length_m=branch_length_m,
    )
    scan = simulate_lidar_scan(
        walls,
        anchor_xy,
        anchor_yaw_deg=anchor_yaw_deg,
        angle_step_deg=angle_step_deg,
        max_range_m=max_range_m,
        noise_std_m=noise_std_m,
        dropout_probability=dropout_probability,
        occlusion_probability=occlusion_probability,
        visible_boundary_ratio=visible_boundary_ratio,
        seed=seed,
    )
    detector_angles, detector_ranges = scan.detector_input()
    point_cloud_theta_r = np.column_stack((detector_angles, detector_ranges))

    # Research-facing detector input: P={(theta_i, r_i)} only.
    # The test geometry and its branch count/directions stay outside this boundary.
    openings = detect_openings_from_point_cloud(point_cloud_theta_r)
    _, diagnostics = _detect_openings_with_diagnostics(detector_angles, detector_ranges)
    return walls, scan, openings, diagnostics


def regression_test_way_counts(
    way_counts: Sequence[int] = (3, 4, 5, 6, 7, 8),
) -> list[tuple[int, int, bool]]:
    """Test different N using the exact same detector without count hints."""
    results: list[tuple[int, int, bool]] = []
    for n in way_counts:
        # Use a larger central chamber automatically so high-N openings remain separate.
        central_radius = max(1.6, 0.55 * n)
        branch_angles = evenly_spaced_branch_angles(n, rotation_deg=11.0)
        _, _, openings, _ = run_case(
            branch_angles,
            central_radius_m=central_radius,
            noise_std_m=0.0,
            anchor_xy=(0.0, 0.0),
            anchor_yaw_deg=23.0,
        )
        detected = len(openings)
        results.append((n, detected, detected == n))
    return results


def _parse_branch_angles(text: Optional[str], ways: int, rotation_deg: float) -> np.ndarray:
    if text:
        values = [float(part.strip()) for part in text.split(",") if part.strip()]
        if not values:
            raise ValueError("--branch-angles did not contain any angles")
        return np.asarray(values, dtype=float)
    return evenly_spaced_branch_angles(ways, rotation_deg=rotation_deg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ways", type=int, default=4, help="TEST ONLY: number of openings used to generate synthetic geometry")
    parser.add_argument(
        "--branch-angles",
        type=str,
        default=None,
        help='TEST ONLY: synthetic opening directions, e.g. "0,65,160,245"',
    )
    parser.add_argument("--rotation", type=float, default=0.0)
    parser.add_argument("--corridor-width", type=float, default=2.0)
    parser.add_argument("--central-radius", type=float, default=1.8)
    parser.add_argument("--branch-length", type=float, default=10.0)
    parser.add_argument("--anchor-x", type=float, default=0.0)
    parser.add_argument("--anchor-y", type=float, default=0.0)
    parser.add_argument("--anchor-yaw", type=float, default=0.0)
    parser.add_argument("--angle-step", type=float, default=1.0)
    parser.add_argument("--max-range", type=float, default=6.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--incoming-branch-index",
        type=int,
        default=0,
        help=(
            "TEST ONLY: which synthetic branch the robot arrived from; "
            "used only to simulate a local recent-local motion direction"
        ),
    )
    parser.add_argument(
        "--local-motion-direction", "--travel-direction-local",
        dest="local_motion_direction",
        type=float,
        default=None,
        help=(
            "LOCAL INPUT: recent motion direction in the LiDAR frame [deg]. "
            "If omitted in simulation, it is synthesized from --incoming-branch-index."
        ),
    )
    parser.add_argument(
        "--incoming-tolerance",
        type=float,
        default=20.0,
        help="max angular gap [deg] for matching the incoming corridor",
    )
    parser.add_argument(
        "--occlusion",
        type=float,
        default=0.0,
        help="SIMULATOR ONLY: probability of one contiguous angular occlusion block",
    )
    parser.add_argument(
        "--visible-boundary-ratio",
        type=float,
        default=1.0,
        help="SIMULATOR ONLY: fraction of physical wall returns kept in the scan",
    )
    parser.add_argument("--csv", type=str, default="general_local_scan.csv")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="compare this run with simulator-only geometric ground truth")
    parser.add_argument("--benchmark", action="store_true", help="run repeated multi-seed evaluation")
    parser.add_argument("--failure-sweep", action="store_true", help="sweep sensor degradation and Anchor offset conditions")
    parser.add_argument("--benchmark-ways", type=str, default="2,3,4,5", help="comma-separated way counts for benchmark/sweep")
    parser.add_argument("--runs", type=int, default=20, help="runs per way count and condition")
    parser.add_argument("--metrics-csv", type=str, default=None, help="save benchmark/sweep metrics CSV")
    parser.add_argument("--min-acceptable-iou", type=float, default=0.50, help="opening IoU below this is flagged as a failure")
    parser.add_argument("--seed", type=int, default=7, help="base random seed")
    parser.add_argument("--regression", action="store_true")
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be > 0")
    if not 0.0 <= args.min_acceptable_iou <= 1.0:
        parser.error("--min-acceptable-iou must be in [0,1]")

    if args.failure_sweep:
        rows = run_failure_sweep(args)
        if args.metrics_csv:
            _write_metrics_csv(rows, args.metrics_csv)
        return

    if args.benchmark:
        rows = run_benchmark(args)
        if args.metrics_csv:
            _write_metrics_csv(rows, args.metrics_csv)
        return

    if args.regression:
        print("=== Way-count-agnostic regression ===")
        for expected, detected, ok in regression_test_way_counts():
            print(f"N={expected}: detected={detected} -> {'PASS' if ok else 'FAIL'}")
        return

    branch_angles = _parse_branch_angles(args.branch_angles, args.ways, args.rotation)
    walls, scan, openings, diagnostics = run_case(
        branch_angles,
        anchor_xy=(args.anchor_x, args.anchor_y),
        anchor_yaw_deg=args.anchor_yaw,
        corridor_width_m=args.corridor_width,
        central_radius_m=args.central_radius,
        branch_length_m=args.branch_length,
        max_range_m=args.max_range,
        noise_std_m=args.noise,
        dropout_probability=args.dropout,
        occlusion_probability=args.occlusion,
        visible_boundary_ratio=args.visible_boundary_ratio,
        angle_step_deg=args.angle_step,
        seed=args.seed,
    )

    # The topology classifier does not receive branch angles/count.
    # For this synthetic test only, recent local motion can be generated from
    # one chosen incoming branch. In a real robot, pass measured local motion
    # via --local-motion-direction / the API instead.
    if args.local_motion_direction is None:
        local_motion_direction = simulated_local_motion_direction(
            branch_angles,
            incoming_branch_index=args.incoming_branch_index,
            anchor_yaw_deg=args.anchor_yaw,
        )
        travel_source = "synthetic test motion from incoming branch"
    else:
        local_motion_direction = float(args.local_motion_direction)
        travel_source = "explicit local motion input"

    topology_result = classify_local_topology(
        openings,
        local_motion_direction_deg=local_motion_direction,
        incoming_tolerance_deg=args.incoming_tolerance,
    )

    save_local_scan_csv(scan, args.csv)

    print("=== General Point-Cloud Junction Opening Detector ===")
    print(f"test-only ground-truth openings : {len(branch_angles)}")
    print(f"Detected openings              : {len(openings)}")
    print("Detector was given only P={(theta_i, r_i)}; no way count or branch directions.")
    for i, opening in enumerate(openings):
        print(
            f"  Opening {i}: start={opening['start_angle']:.1f} deg, "
            f"end={opening['end_angle']:.1f} deg, "
            f"center={opening['center_angle']:.1f} deg, "
            f"width={opening['width_deg']:.1f} deg, "
            f"confidence={opening['confidence']:.2f}"
        )

    print("\n=== Localization-Free Local Topology ===")
    print(f"Recent local motion direction (local) : {topology_result.local_motion_direction_deg:.1f} deg")
    print(f"Travel-direction source         : {travel_source}")
    print(f"Expected incoming direction     : {topology_result.incoming_direction_local_deg:.1f} deg")
    print(f"Incoming opening index          : {topology_result.incoming_opening_index}")
    print(f"New outgoing openings           : {topology_result.outgoing_opening_count}")
    print(f"Topology                        : {topology_result.topology}")
    print(f"Junction?                       : {topology_result.is_junction}")
    if topology_result.incoming_match_error_deg is not None:
        print(f"Incoming-sector angular gap     : {topology_result.incoming_match_error_deg:.1f} deg")

    if topology_result.incoming_opening_index is not None:
        inc = openings[topology_result.incoming_opening_index]
        print(
            f"  Incoming opening: [{inc['start_angle']:.1f}, {inc['end_angle']:.1f}] deg, "
            f"center={inc['center_angle']:.1f} deg"
        )
    for local_rank, opening_idx in enumerate(topology_result.outgoing_opening_indices):
        opening = openings[opening_idx]
        print(
            f"  New exit {local_rank}: opening={opening_idx}, "
            f"[{opening['start_angle']:.1f}, {opening['end_angle']:.1f}] deg, "
            f"center={opening['center_angle']:.1f} deg"
        )

    print("\nOpening detector receives only: Anchor-local P={(theta_i, r_i)}")
    print("Topology classifier additionally receives only: recent local motion direction in the same local frame")
    print("Neither stage receives: global map/pose, junction coordinates, expected way count, or expected branch directions")

    if args.evaluate:
        gt_openings, eval_free_path_threshold = ground_truth_openings_from_ideal_scan(
            walls,
            anchor_xy=(args.anchor_x, args.anchor_y),
            anchor_yaw_deg=args.anchor_yaw,
            angle_step_deg=args.angle_step,
            max_range_m=args.max_range,
        )
        metrics = evaluate_detection(
            gt_openings,
            openings,
            expected_topology=_expected_topology_from_way_count(len(branch_angles)),
            detected_topology=topology_result.topology,
            min_acceptable_iou=args.min_acceptable_iou,
        )
        mouth_gt = physical_mouth_openings_from_geometry(
            branch_angles,
            anchor_xy=(args.anchor_x, args.anchor_y),
            anchor_yaw_deg=args.anchor_yaw,
            corridor_width_m=args.corridor_width,
            central_radius_m=args.central_radius,
        )
        mouth_metrics = evaluate_detection(mouth_gt, openings, min_acceptable_iou=0.0)
        metrics["evaluation_free_path_threshold_m"] = eval_free_path_threshold
        metrics["physical_mouth_mean_iou"] = mouth_metrics["mean_iou"]
        metrics["physical_mouth_boundary_mae_deg"] = mouth_metrics["boundary_mae_deg"]
        print("\n=== Detection Accuracy Evaluation ===")
        print("Primary boundary target : ideal free-path sector (not full physical mouth)")
        print(f"Free-path threshold     : {metrics['evaluation_free_path_threshold_m']:.3f} m")
        print(f"Ground-truth openings : {metrics['ground_truth_count']}")
        print(f"Detected openings     : {metrics['detected_count']}")
        print(f"Count correct         : {metrics['count_correct']}")
        print(f"Matched / FP / FN     : {metrics['matched_openings']} / {metrics['false_positive']} / {metrics['false_negative']}")
        print(f"Precision/Recall/F1   : {metrics['precision']:.3f} / {metrics['recall']:.3f} / {metrics['f1']:.3f}")
        print(f"Free-sector mean/min IoU: {metrics['mean_iou']:.3f} / {metrics['min_iou']:.3f}")
        print(f"Center-angle MAE      : {metrics['center_mae_deg']:.2f} deg")
        print(f"Start / end MAE       : {metrics['start_mae_deg']:.2f} / {metrics['end_mae_deg']:.2f} deg")
        print(f"Free-sector boundary MAE: {metrics['boundary_mae_deg']:.2f} deg")
        print(f"Physical-mouth boundary MAE (diagnostic): {metrics['physical_mouth_boundary_mae_deg']:.2f} deg")
        print(f"Expected topology     : {metrics['expected_topology']}")
        print(f"Detected topology     : {metrics['detected_topology']}")
        print(f"Topology correct      : {metrics['topology_correct']}")
        print(f"Failure reasons       : {metrics['failure_reasons']}")

    plot_results(
        walls,
        (args.anchor_x, args.anchor_y),
        scan,
        openings,
        diagnostics,
        topology_result=topology_result,
        anchor_yaw_deg=args.anchor_yaw,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()