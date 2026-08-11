"""Generate an ideal 2D LiDAR point cloud for a four-way junction."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cast ideal 2D LiDAR rays against arbitrary wall segments.

    Rays span ``[-180, 180)`` degrees.  For each ray, the closest wall hit is
    retained; if there is no hit within ``max_range``, its endpoint is placed
    exactly at ``max_range``.  No junction topology is assumed.

    Args:
        walls: Arbitrary iterable of wall line segments.
        anchor: LiDAR origin in world coordinates.
        angle_step_deg: Positive angular increment in degrees.
        max_range: Maximum measurable distance.

    Returns:
        Arrays ``(angles_deg, ranges, hit_points)``.  ``hit_points`` contains
        one world-coordinate endpoint per ray, including max-range endpoints.
    """
    if angle_step_deg <= 0.0:
        raise ValueError("angle_step_deg must be positive")
    if max_range <= 0.0:
        raise ValueError("max_range must be positive")

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


def visualize_results(
    walls: Iterable[Segment],
    anchor: Sequence[float],
    angles_deg: np.ndarray,
    ranges: np.ndarray,
    hit_points: np.ndarray,
    max_range: float,
    ray_stride: int = 15,
) -> Tuple[plt.Figure, np.ndarray]:
    """Visualize geometry, LiDAR endpoints, and the angular range profile."""
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

    # Max-range endpoints represent no-return directions, so distinguish them
    # from actual wall returns while keeping the complete point cloud visible.
    no_return = np.isclose(ranges, max_range, rtol=0.0, atol=1.0e-9)
    cloud_axis.scatter(
        hit_points[~no_return, 0],
        hit_points[~no_return, 1],
        s=12,
        color="tab:blue",
        label="wall return",
    )
    cloud_axis.scatter(
        hit_points[no_return, 0],
        hit_points[no_return, 1],
        s=10,
        color="tab:orange",
        alpha=0.55,
        label="max-range endpoint",
    )
    cloud_axis.scatter(*anchor_array, marker="*", s=130, color="red", label="anchor")
    cloud_axis.set_title("B. Ideal LiDAR point cloud")
    cloud_axis.set_xlabel("x")
    cloud_axis.set_ylabel("y")
    cloud_axis.set_aspect("equal", adjustable="box")
    cloud_axis.grid(alpha=0.2)
    cloud_axis.legend(loc="upper right", fontsize=8)

    range_axis.plot(angles_deg, ranges, color="tab:blue", linewidth=1.2)
    range_axis.set_title("C. Range profile")
    range_axis.set_xlabel("angle [deg]")
    range_axis.set_ylabel("range")
    range_axis.set_xlim(-180.0, 180.0)
    range_axis.set_ylim(0.0, max_range * 1.05)
    range_axis.grid(alpha=0.3)

    figure.tight_layout()
    return figure, axes


def main() -> None:
    """Generate the noiseless junction scan and display its diagnostics."""
    max_range = 10.0
    walls, anchor = create_4way_junction()
    angles_deg, ranges, hit_points = simulate_lidar(
        walls, anchor, angle_step_deg=1.0, max_range=max_range
    )
    local_points = polar_to_xy(angles_deg, ranges)

    # The anchor is (0, 0), so local Cartesian points equal world endpoints.
    if not np.allclose(local_points + np.asarray(anchor), hit_points):
        raise RuntimeError("polar and ray-cast point clouds are inconsistent")

    visualize_results(
        walls, anchor, angles_deg, ranges, hit_points, max_range=max_range
    )
    plt.show()


if __name__ == "__main__":
    main()
