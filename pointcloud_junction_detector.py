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

Pipeline
--------
Arbitrary test geometry -> ray casting -> Anchor-local P={(theta,r)}
-> circular smoothing -> adaptive open-support extraction
-> circular connected components -> range-gradient boundary refinement
-> opening start/end angles and automatically inferred opening count
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
    seed: Optional[int] = None,
) -> LidarScan:
    """Ray-cast a 2D LiDAR scan and return it in the Anchor-local frame."""
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

    if dropout_probability > 0.0 and hits.any():
        dropout = (rng.random(ranges.size) < dropout_probability) & hits
        hits[dropout] = False
        ranges[dropout] = max_range_m

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

    smoothed = smooth_ranges(raw, smoothing_window_size)

    # These are inferred from the scan; the detector is not passed sensor/map metadata.
    wall_reference = float(np.quantile(smoothed, wall_reference_quantile))
    range_ceiling = float(np.max(raw))
    dynamic_span = max(0.0, range_ceiling - wall_reference)

    # If there is no meaningful contrast, no opening can be supported by this baseline.
    if dynamic_span <= 1.0e-6:
        diagnostics = {
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
    cloud_ax.set_aspect("equal", adjustable="box")
    cloud_ax.set_title("B. Anchor-local point cloud")
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
    parser.add_argument("--csv", type=str, default="general_local_scan.csv")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--regression", action="store_true")
    args = parser.parse_args()

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
        angle_step_deg=args.angle_step,
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

    print("\nDetector receives only: Anchor-local P={(theta_i, r_i)}")
    print("Detector does NOT receive: way count, branch labels/directions, walls/map, global pose, junction coordinates")

    plot_results(
        walls,
        (args.anchor_x, args.anchor_y),
        scan,
        openings,
        diagnostics,
        anchor_yaw_deg=args.anchor_yaw,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()