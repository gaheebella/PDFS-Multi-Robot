"""Generate an ideal 2D LiDAR point cloud for a four-way junction."""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


Point = Tuple[float, float]
Segment = Tuple[Point, Point]
Intersection = Tuple[float, Point]


def create_4way_junction(
    corridor_width: float = 2.0,
    branch_length: float = 12.0,
) -> Tuple[list[Segment], Point]:
    """Create the boundary walls of a symmetric four-way cross junction.

    The free space is the union of one horizontal and one vertical corridor.
    The returned walls are the eight boundary segments extending away from the
    central square.  Branch ends deliberately remain open.

    Args:
        corridor_width: Constant width of every corridor branch.
        branch_length: Distance from the anchor to the far end of each wall.

    Returns:
        A wall-segment list and the anchor at ``(0, 0)``.
    """
    if corridor_width <= 0.0:
        raise ValueError("corridor_width must be positive")

    half_width = corridor_width / 2.0
    if branch_length <= half_width:
        raise ValueError("branch_length must exceed half the corridor width")

    h = half_width
    length = float(branch_length)
    walls: list[Segment] = [
        # Upper and lower walls of the left branch.
        ((-length, h), (-h, h)),
        ((-length, -h), (-h, -h)),
        # Upper and lower walls of the right branch.
        ((h, h), (length, h)),
        ((h, -h), (length, -h)),
        # Left and right walls of the upper branch.
        ((-h, h), (-h, length)),
        ((h, h), (h, length)),
        # Left and right walls of the lower branch.
        ((-h, -length), (-h, -h)),
        ((h, -length), (h, -h)),
    ]
    return walls, (0.0, 0.0)


def _cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    """Return the scalar 2D cross product of two vectors."""
    return float(a[0] * b[1] - a[1] * b[0])


def ray_segment_intersection(
    ray_origin: Sequence[float],
    ray_direction: Sequence[float],
    segment_start: Sequence[float],
    segment_end: Sequence[float],
    epsilon: float = 1.0e-10,
) -> Optional[Intersection]:
    """Find the nearest intersection between a ray and a finite segment.

    Args:
        ray_origin: The ray's starting point.
        ray_direction: Ray direction, normally a unit vector.
        segment_start: First endpoint of the wall segment.
        segment_end: Second endpoint of the wall segment.
        epsilon: Tolerance used for parallel and boundary comparisons.

    Returns:
        ``(distance, (x, y))`` for an intersection on the forward ray and
        within the segment, otherwise ``None``.  Collinear overlap returns the
        nearest forward point, so the function also behaves sensibly for that
        otherwise-degenerate case.
    """
    origin = np.asarray(ray_origin, dtype=float)
    direction = np.asarray(ray_direction, dtype=float)
    start = np.asarray(segment_start, dtype=float)
    end = np.asarray(segment_end, dtype=float)

    if any(vector.shape != (2,) for vector in (origin, direction, start, end)):
        raise ValueError("all points and vectors must contain exactly two values")

    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= epsilon:
        raise ValueError("ray_direction must be non-zero")

    segment = end - start
    segment_norm = float(np.linalg.norm(segment))
    offset = start - origin

    # A zero-length segment is treated as a point lying on (or off) the ray.
    if segment_norm <= epsilon:
        if abs(_cross_2d(offset, direction)) > epsilon * direction_norm:
            return None
        ray_parameter = float(np.dot(offset, direction) / direction_norm**2)
        if ray_parameter < -epsilon:
            return None
        ray_parameter = max(0.0, ray_parameter)
        point = origin + ray_parameter * direction
        return ray_parameter * direction_norm, (float(point[0]), float(point[1]))

    denominator = _cross_2d(direction, segment)
    parallel_tolerance = epsilon * direction_norm * segment_norm
    if abs(denominator) <= parallel_tolerance:
        # Parallel lines only intersect when they are collinear.  Project both
        # endpoints onto the ray and select the closest forward overlap.
        if abs(_cross_2d(offset, direction)) > epsilon * direction_norm:
            return None
        projections = np.array(
            [
                np.dot(start - origin, direction),
                np.dot(end - origin, direction),
            ],
            dtype=float,
        ) / direction_norm**2
        if float(np.max(projections)) < -epsilon:
            return None
        ray_parameter = max(0.0, float(np.min(projections)))
        point = origin + ray_parameter * direction
        return ray_parameter * direction_norm, (float(point[0]), float(point[1]))

    # Solve origin + t*direction = start + u*segment using 2D cross products.
    ray_parameter = _cross_2d(offset, segment) / denominator
    segment_parameter = _cross_2d(offset, direction) / denominator

    if ray_parameter < -epsilon:
        return None
    if segment_parameter < -epsilon or segment_parameter > 1.0 + epsilon:
        return None

    # Clamp values accepted only through tolerance back onto the exact domains.
    ray_parameter = max(0.0, ray_parameter)
    segment_parameter = min(1.0, max(0.0, segment_parameter))
    ray_point = origin + ray_parameter * direction
    segment_point = start + segment_parameter * segment
    point = 0.5 * (ray_point + segment_point)
    distance = ray_parameter * direction_norm
    return distance, (float(point[0]), float(point[1]))


def simulate_lidar(
    walls: Iterable[Segment],
    anchor: Sequence[float],
    angle_step_deg: float = 1.0,
    max_range: float = 10.0,
    noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cast ideal 2D LiDAR rays against arbitrary wall segments.

    Rays span ``[-180, 180)`` degrees.  For each ray, the closest wall hit is
    retained; if there is no hit within ``max_range``, its endpoint is placed
    exactly at ``max_range``.  Optional zero-mean Gaussian measurement noise
    is applied after ideal ray casting.  No junction topology is assumed.

    Args:
        walls: Arbitrary iterable of wall line segments.
        anchor: LiDAR origin in world coordinates.
        angle_step_deg: Positive angular increment in degrees.
        max_range: Maximum measurable distance.
        noise_std: Standard deviation of Gaussian range noise.  Zero disables
            noise.  Noisy measurements are clipped only at zero.
        rng: Optional NumPy random generator for reproducible noise.

    Returns:
        Arrays ``(angles_deg, ranges, hit_points)``.  ``hit_points`` contains
        one world-coordinate endpoint per ray, including max-range endpoints.
    """
    if angle_step_deg <= 0.0:
        raise ValueError("angle_step_deg must be positive")
    if max_range <= 0.0:
        raise ValueError("max_range must be positive")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")

    anchor_array = np.asarray(anchor, dtype=float)
    if anchor_array.shape != (2,):
        raise ValueError("anchor must contain exactly two values")

    wall_list = list(walls)
    angles_deg = np.arange(-180.0, 180.0, angle_step_deg, dtype=float)
    ranges = np.full(angles_deg.shape, float(max_range), dtype=float)
    hit_points = np.empty((angles_deg.size, 2), dtype=float)

    for index, angle_deg in enumerate(angles_deg):
        angle_rad = math.radians(float(angle_deg))
        direction = np.array([math.cos(angle_rad), math.sin(angle_rad)])
        nearest_distance = float(max_range)

        for segment_start, segment_end in wall_list:
            intersection = ray_segment_intersection(
                anchor_array, direction, segment_start, segment_end
            )
            if intersection is not None and intersection[0] < nearest_distance:
                nearest_distance = intersection[0]

        ranges[index] = nearest_distance
        hit_points[index] = anchor_array + nearest_distance * direction

    if noise_std > 0.0:
        generator = rng if rng is not None else np.random.default_rng()
        ranges = np.maximum(0.0, ranges + generator.normal(0.0, noise_std, ranges.size))
        angles_rad = np.deg2rad(angles_deg)
        directions = np.column_stack((np.cos(angles_rad), np.sin(angles_rad)))
        hit_points = anchor_array + ranges[:, np.newaxis] * directions

    return angles_deg, ranges, hit_points


def polar_to_xy(
    angles_deg: Sequence[float], ranges: Sequence[float]
) -> np.ndarray:
    """Convert polar LiDAR measurements to anchor-local Cartesian points.

    Args:
        angles_deg: Measurement angles in degrees.
        ranges: Range associated with each angle.

    Returns:
        An ``(N, 2)`` array whose columns are local ``x`` and ``y``.
    """
    angles = np.asarray(angles_deg, dtype=float)
    radial_ranges = np.asarray(ranges, dtype=float)
    if angles.ndim != 1 or radial_ranges.ndim != 1:
        raise ValueError("angles_deg and ranges must be one-dimensional")
    if angles.shape != radial_ranges.shape:
        raise ValueError("angles_deg and ranges must have the same length")

    angles_rad = np.deg2rad(angles)
    return np.column_stack(
        (radial_ranges * np.cos(angles_rad), radial_ranges * np.sin(angles_rad))
    )


def smooth_ranges(ranges: Sequence[float], window_size: int = 5) -> np.ndarray:
    """Smooth a circular LiDAR scan with a centered moving average.

    The first and last samples are neighbors, as required for a 360-degree
    scan.  An odd window keeps the filtered value centered on each input ray.

    Args:
        ranges: One-dimensional range measurements.
        window_size: Positive odd number of circularly averaged samples.

    Returns:
        Smoothed ranges with the same shape as the input.
    """
    values = np.asarray(ranges, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("ranges must be a non-empty one-dimensional sequence")
    if not isinstance(window_size, (int, np.integer)) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd for centered smoothing")
    if window_size > values.size:
        raise ValueError("window_size cannot exceed the number of ranges")
    if not np.all(np.isfinite(values)):
        raise ValueError("ranges must contain only finite values")

    half_window = window_size // 2
    shifted = [np.roll(values, shift) for shift in range(-half_window, half_window + 1)]
    return np.mean(shifted, axis=0)


def circular_range_gradient(
    angles_deg: Sequence[float], ranges: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate forward range change per degree on a circular scan.

    Returns one gradient at the angular midpoint between every ray and its
    successor.  The final successor is the first ray plus 360 degrees.
    """
    angles = np.asarray(angles_deg, dtype=float)
    values = np.asarray(ranges, dtype=float)
    if angles.ndim != 1 or values.ndim != 1 or angles.shape != values.shape:
        raise ValueError("angles_deg and ranges must be equal-length 1D arrays")
    if angles.size < 3:
        raise ValueError("at least three LiDAR samples are required")
    if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(values)):
        raise ValueError("angles_deg and ranges must contain only finite values")
    if np.any(np.diff(angles) <= 0.0):
        raise ValueError("angles_deg must be strictly increasing")

    angular_steps = np.diff(np.append(angles, angles[0] + 360.0))
    if np.any(angular_steps <= 0.0):
        raise ValueError("angles_deg must span less than one full revolution")

    gradient = (np.roll(values, -1) - values) / angular_steps
    boundary_angles = angles + 0.5 * angular_steps
    boundary_angles = _normalize_angles(boundary_angles)
    return boundary_angles, gradient


def _normalize_angles(angles_deg: Any) -> Any:
    """Normalize scalar or array angles to the half-open interval [-180, 180)."""
    normalized = (np.asarray(angles_deg) + 180.0) % 360.0 - 180.0
    if np.ndim(angles_deg) == 0:
        return float(normalized)
    return normalized


def _automatic_gradient_threshold(
    gradient: np.ndarray,
    threshold_mad_scale: float,
    min_gradient_threshold: float,
) -> float:
    """Estimate a robust gradient-magnitude threshold using median and MAD."""
    magnitudes = np.abs(gradient)
    median = float(np.median(magnitudes))
    mad = float(np.median(np.abs(magnitudes - median)))
    robust_sigma = 1.4826 * mad
    return max(float(min_gradient_threshold), median + threshold_mad_scale * robust_sigma)


def _group_boundary_candidates(
    candidate_indices: np.ndarray,
    boundary_angles: np.ndarray,
    gradient: np.ndarray,
    max_gap_deg: float,
) -> list[float]:
    """Group nearby same-sign boundary candidates on a circular angle domain."""
    if candidate_indices.size == 0:
        return []

    positions = (boundary_angles[candidate_indices] + 180.0) % 360.0
    order = np.argsort(positions)
    sorted_indices = candidate_indices[order]
    sorted_positions = positions[order]
    groups: list[list[int]] = [[int(sorted_indices[0])]]

    for previous_position, position, index in zip(
        sorted_positions[:-1], sorted_positions[1:], sorted_indices[1:]
    ):
        if position - previous_position <= max_gap_deg:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])

    # The last and first groups may be adjacent through the circular seam.
    wrap_gap = sorted_positions[0] + 360.0 - sorted_positions[-1]
    if len(groups) > 1 and wrap_gap <= max_gap_deg:
        groups[0] = groups[-1] + groups[0]
        groups.pop()

    representatives = []
    for group in groups:
        group_array = np.asarray(group, dtype=int)
        strongest_index = int(group_array[np.argmax(np.abs(gradient[group_array]))])
        representatives.append(float(boundary_angles[strongest_index]))
    return sorted(representatives)


def _opening_width(start_angle: float, end_angle: float) -> float:
    """Return positive counter-clockwise width from start to end."""
    return float((end_angle - start_angle) % 360.0)


def _detect_openings_with_diagnostics(
    angles_deg: Sequence[float],
    ranges: Sequence[float],
    smoothing_window_size: int = 5,
    gradient_threshold: Optional[float] = None,
    threshold_mad_scale: float = 4.0,
    min_gradient_threshold: float = 0.05,
    candidate_group_gap_deg: float = 3.0,
    min_opening_width_deg: float = 5.0,
    max_opening_width_deg: Optional[float] = 120.0,
) -> Tuple[list[dict[str, float]], dict[str, Any]]:
    """Run the range-discontinuity baseline and retain plot diagnostics."""
    angles = np.asarray(angles_deg, dtype=float)
    raw_ranges = np.asarray(ranges, dtype=float)
    if angles.ndim != 1 or raw_ranges.ndim != 1 or angles.shape != raw_ranges.shape:
        raise ValueError("angles_deg and ranges must be equal-length 1D arrays")
    if np.any(raw_ranges < 0.0):
        raise ValueError("ranges cannot be negative")
    if threshold_mad_scale < 0.0 or min_gradient_threshold < 0.0:
        raise ValueError("threshold parameters must be non-negative")
    if candidate_group_gap_deg <= 0.0:
        raise ValueError("candidate_group_gap_deg must be positive")
    if min_opening_width_deg <= 0.0 or min_opening_width_deg >= 360.0:
        raise ValueError("min_opening_width_deg must be in (0, 360)")
    if max_opening_width_deg is not None:
        if max_opening_width_deg <= min_opening_width_deg or max_opening_width_deg > 360.0:
            raise ValueError("max_opening_width_deg must exceed the minimum and be <= 360")
    if gradient_threshold is not None and gradient_threshold <= 0.0:
        raise ValueError("gradient_threshold must be positive when supplied")

    smoothed = smooth_ranges(raw_ranges, smoothing_window_size)
    boundary_angles, gradient = circular_range_gradient(angles, smoothed)
    threshold = (
        float(gradient_threshold)
        if gradient_threshold is not None
        else _automatic_gradient_threshold(
            gradient, threshold_mad_scale, min_gradient_threshold
        )
    )

    start_candidates = np.flatnonzero(gradient >= threshold)
    end_candidates = np.flatnonzero(gradient <= -threshold)
    starts = _group_boundary_candidates(
        start_candidates, boundary_angles, gradient, candidate_group_gap_deg
    )
    ends = _group_boundary_candidates(
        end_candidates, boundary_angles, gradient, candidate_group_gap_deg
    )

    # Pair every rising edge with its next falling edge around the circle.  No
    # branch count or expected direction is used anywhere in this baseline.
    openings: list[dict[str, float]] = []
    used_end_indices: set[int] = set()
    for start in starts:
        ranked_ends = sorted(
            enumerate(ends), key=lambda item: _opening_width(start, item[1])
        )
        for end_index, end in ranked_ends:
            width = _opening_width(start, end)
            if end_index in used_end_indices or width <= 0.0:
                continue
            if width < min_opening_width_deg:
                break
            if max_opening_width_deg is not None and width > max_opening_width_deg:
                break
            center = _normalize_angles(start + width / 2.0)
            openings.append(
                {
                    "start_angle": float(_normalize_angles(start)),
                    "end_angle": float(_normalize_angles(end)),
                    "center_angle": float(center),
                    "width_deg": float(width),
                }
            )
            used_end_indices.add(end_index)
            break

    openings.sort(key=lambda opening: opening["center_angle"])
    diagnostics: dict[str, Any] = {
        "smoothed_ranges": smoothed,
        "boundary_angles": boundary_angles,
        "gradient": gradient,
        "gradient_threshold": threshold,
        "start_angles": [opening["start_angle"] for opening in openings],
        "end_angles": [opening["end_angle"] for opening in openings],
    }
    return openings, diagnostics


def detect_openings(
    angles_deg: Sequence[float],
    ranges: Sequence[float],
    smoothing_window_size: int = 5,
    gradient_threshold: Optional[float] = None,
    threshold_mad_scale: float = 4.0,
    min_gradient_threshold: float = 0.05,
    candidate_group_gap_deg: float = 3.0,
    min_opening_width_deg: float = 5.0,
    max_opening_width_deg: Optional[float] = 120.0,
) -> list[dict[str, float]]:
    """Detect opening intervals using only angle-distance measurements.

    This is intentionally a range-discontinuity *baseline*, not a final
    junction detector.  Rising gradients propose opening starts and falling
    gradients propose ends.  Candidate grouping and width limits make its
    sensitivity configurable for noise, shallow branches, narrow openings,
    max-range clipping, natural range peaks, and circular wrap-around.

    Neither walls, environment geometry, branch count, nor expected branch
    directions are accepted or inferred from hard-coded constants.

    Args:
        angles_deg: Strictly increasing ray angles covering a circular scan.
        ranges: Range measurement for each angle.
        smoothing_window_size: Odd circular moving-average window.
        gradient_threshold: Manual absolute gradient threshold in range/degree.
            If omitted, a median/MAD robust threshold is estimated.
        threshold_mad_scale: Robust-sigma multiplier for the automatic threshold.
        min_gradient_threshold: Floor for the automatic threshold.
        candidate_group_gap_deg: Maximum angular gap within a boundary group.
        min_opening_width_deg: Reject narrower paired intervals.
        max_opening_width_deg: Optional upper width limit, useful for rejecting
            broad natural peaks.  ``None`` disables the upper limit.

    Returns:
        Opening dictionaries with start, end, center, and positive circular
        width.  A wrap-around opening has ``start_angle > end_angle``.
    """
    openings, _ = _detect_openings_with_diagnostics(
        angles_deg=angles_deg,
        ranges=ranges,
        smoothing_window_size=smoothing_window_size,
        gradient_threshold=gradient_threshold,
        threshold_mad_scale=threshold_mad_scale,
        min_gradient_threshold=min_gradient_threshold,
        candidate_group_gap_deg=candidate_group_gap_deg,
        min_opening_width_deg=min_opening_width_deg,
        max_opening_width_deg=max_opening_width_deg,
    )
    return openings


def visualize_results(
    walls: Iterable[Segment],
    anchor: Sequence[float],
    angles_deg: np.ndarray,
    ranges: np.ndarray,
    hit_points: np.ndarray,
    max_range: float,
    ray_stride: int = 15,
    openings: Optional[Sequence[dict[str, float]]] = None,
    detector_diagnostics: Optional[dict[str, Any]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Visualize geometry, point cloud, and optional baseline detections."""
    if ray_stride <= 0:
        raise ValueError("ray_stride must be positive")

    wall_list = list(walls)
    anchor_array = np.asarray(anchor, dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    geometry_axis, cloud_axis, range_axis = axes

    for start, end in wall_list:
        geometry_axis.plot(
            [start[0], end[0]], [start[1], end[1]], color="black", linewidth=2
        )
    for point in hit_points[::ray_stride]:
        geometry_axis.plot(
            [anchor_array[0], point[0]],
            [anchor_array[1], point[1]],
            color="tab:orange",
            alpha=0.35,
            linewidth=0.8,
        )
    geometry_axis.scatter(*anchor_array, marker="*", s=130, color="red", zorder=3)
    geometry_axis.set_title("A. Four-way junction and sampled rays")
    geometry_axis.set_xlabel("x")
    geometry_axis.set_ylabel("y")
    geometry_axis.set_aspect("equal", adjustable="box")
    geometry_axis.grid(alpha=0.2)

    # With measurement noise, range values alone cannot reliably distinguish a
    # wall return from a max-range return, so avoid assigning ground-truth labels.
    cloud_axis.scatter(
        hit_points[:, 0],
        hit_points[:, 1],
        s=12,
        color="tab:blue",
        label="LiDAR endpoint",
    )
    cloud_axis.scatter(*anchor_array, marker="*", s=130, color="red", label="anchor")
    if openings:
        direction_length = max_range * 0.85
        for index, opening in enumerate(openings):
            center_rad = math.radians(opening["center_angle"])
            endpoint = anchor_array + direction_length * np.array(
                [math.cos(center_rad), math.sin(center_rad)]
            )
            cloud_axis.plot(
                [anchor_array[0], endpoint[0]],
                [anchor_array[1], endpoint[1]],
                color="tab:green",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label="opening center" if index == 0 else None,
            )
    cloud_axis.set_title("B. LiDAR point cloud")
    cloud_axis.set_xlabel("x")
    cloud_axis.set_ylabel("y")
    cloud_axis.set_aspect("equal", adjustable="box")
    cloud_axis.grid(alpha=0.2)
    cloud_axis.legend(loc="upper right", fontsize=8)

    range_axis.plot(
        angles_deg, ranges, color="tab:blue", linewidth=0.9, alpha=0.55, label="raw range"
    )
    if detector_diagnostics is not None:
        smoothed_ranges = np.asarray(detector_diagnostics["smoothed_ranges"])
        range_axis.plot(
            angles_deg,
            smoothed_ranges,
            color="black",
            linewidth=1.4,
            label="smoothed range",
        )

        for index, opening in enumerate(openings or []):
            start = opening["start_angle"]
            end = opening["end_angle"]
            span_label = "detected opening" if index == 0 else None
            if start <= end:
                range_axis.axvspan(
                    start, end, color="tab:green", alpha=0.12, label=span_label
                )
            else:
                range_axis.axvspan(
                    start, 180.0, color="tab:green", alpha=0.12, label=span_label
                )
                range_axis.axvspan(-180.0, end, color="tab:green", alpha=0.12)

        start_angles = np.asarray(detector_diagnostics["start_angles"], dtype=float)
        end_angles = np.asarray(detector_diagnostics["end_angles"], dtype=float)
        if start_angles.size:
            start_ranges = np.interp(start_angles, angles_deg, smoothed_ranges)
            range_axis.scatter(
                start_angles,
                start_ranges,
                marker="^",
                s=55,
                color="tab:green",
                zorder=5,
                label="opening start",
            )
        if end_angles.size:
            end_ranges = np.interp(end_angles, angles_deg, smoothed_ranges)
            range_axis.scatter(
                end_angles,
                end_ranges,
                marker="v",
                s=55,
                color="tab:red",
                zorder=5,
                label="opening end",
            )

        gradient_axis = range_axis.twinx()
        boundary_angles = np.asarray(detector_diagnostics["boundary_angles"])
        gradient = np.asarray(detector_diagnostics["gradient"])
        threshold = float(detector_diagnostics["gradient_threshold"])
        gradient_axis.plot(
            boundary_angles,
            gradient,
            color="tab:purple",
            linewidth=0.7,
            alpha=0.35,
            label="range gradient",
        )
        gradient_axis.axhline(
            threshold,
            color="tab:orange",
            linestyle=":",
            linewidth=1.1,
            label=f"gradient threshold (+/-{threshold:.2f})",
        )
        gradient_axis.axhline(
            -threshold, color="tab:orange", linestyle=":", linewidth=1.1
        )
        gradient_axis.set_ylabel("range change / deg", color="tab:purple")
        gradient_axis.tick_params(axis="y", labelcolor="tab:purple")

        range_handles, range_labels = range_axis.get_legend_handles_labels()
        gradient_handles, gradient_labels = gradient_axis.get_legend_handles_labels()
        range_axis.legend(
            range_handles + gradient_handles,
            range_labels + gradient_labels,
            loc="upper right",
            fontsize=7,
        )
    else:
        range_axis.legend(loc="upper right", fontsize=8)
    range_axis.set_title("C. Range profile")
    range_axis.set_xlabel("angle [deg]")
    range_axis.set_ylabel("range")
    range_axis.set_xlim(-180.0, 180.0)
    range_axis.set_ylim(0.0, max(max_range, float(np.max(ranges))) * 1.05)
    range_axis.grid(alpha=0.3)

    figure.tight_layout()
    return figure, axes


def main() -> None:
    """Generate a junction scan, detect baseline openings, and visualize it."""
    max_range = 10.0
    noise_std = 0.05
    walls, anchor = create_4way_junction()
    angles_deg, ranges, hit_points = simulate_lidar(
        walls,
        anchor,
        angle_step_deg=1.0,
        max_range=max_range,
        noise_std=noise_std,
        rng=np.random.default_rng(7),
    )
    local_points = polar_to_xy(angles_deg, ranges)

    # The anchor is (0, 0), so local Cartesian points equal world endpoints.
    if not np.allclose(local_points + np.asarray(anchor), hit_points):
        raise RuntimeError("polar and ray-cast point clouds are inconsistent")

    # Only angle-distance data enter the detector.  Geometry and branch count
    # remain strictly outside this range-discontinuity baseline.
    openings, diagnostics = _detect_openings_with_diagnostics(
        angles_deg,
        ranges,
        smoothing_window_size=5,
        min_opening_width_deg=5.0,
    )

    print(f"Detected openings: {len(openings)}")
    for index, opening in enumerate(openings):
        print(
            f"Opening {index}: start={opening['start_angle']:.1f}\N{DEGREE SIGN}, "
            f"end={opening['end_angle']:.1f}\N{DEGREE SIGN}, "
            f"center={opening['center_angle']:.1f}\N{DEGREE SIGN}, "
            f"width={opening['width_deg']:.1f}\N{DEGREE SIGN}"
        )

    visualize_results(
        walls,
        anchor,
        angles_deg,
        ranges,
        hit_points,
        max_range=max_range,
        openings=openings,
        detector_diagnostics=diagnostics,
    )
    plt.show()


if __name__ == "__main__":
    main()
