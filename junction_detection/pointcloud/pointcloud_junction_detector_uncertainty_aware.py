"""Physics/uncertainty-aware opening detector for Anchor-local 2D LiDAR.

This experimental detector is deliberately separate from the protected
baseline.  Runtime inputs are one circular local angle/range scan plus sensor
range uncertainty and the robot diameter.  Map geometry, global pose, Branch
count/direction, and ground truth are neither accepted nor inferred.

The detector treats exact/near-maximum returns as a censored open-space core,
then follows statistically significant range ramps outward to the observable
mouth boundaries.  Angular quantization determines boundary intervals and the
minimum-width test is expressed as a conservative local chord length rather
than a fixed number of degrees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np

from junction_detection.pointcloud import pointcloud_junction_detector as baseline


EPSILON = baseline.EPSILON


@dataclass(frozen=True)
class DetectorStages:
    """Ablation switches; each stage remains independent of simulator GT."""

    adaptive_discontinuity: bool = True
    uncertainty_merge: bool = True
    physical_minimum_width: bool = True
    wall_support_uncertainty: bool = True


def _normal_familywise_z(ray_count: int, false_alarm_probability: float) -> float:
    """Two-sided Bonferroni normal critical value for one circular scan."""
    if ray_count < 1:
        raise ValueError("ray_count must be positive")
    if not 0.0 < false_alarm_probability < 1.0:
        raise ValueError("false_alarm_probability must be in (0, 1)")
    tail = false_alarm_probability / (2.0 * ray_count)
    return float(NormalDist().inv_cdf(1.0 - tail))


def _circular_indices(start: int, end: int, size: int) -> np.ndarray:
    """Inclusive circular indices from ``start`` to ``end`` counter-clockwise."""
    count = (end - start) % size + 1
    return (start + np.arange(count, dtype=int)) % size


def _point_span(
    indices: np.ndarray,
    angles_deg: np.ndarray,
    ranges: np.ndarray,
    ceiling_mask: np.ndarray,
) -> tuple[float, float, int]:
    """Physical endpoint span and representative range of observed returns."""
    usable = indices[~ceiling_mask[indices]]
    if usable.size < 2:
        return 0.0, 0.0, int(usable.size)
    theta = np.radians(angles_deg[usable])
    points = np.column_stack((ranges[usable] * np.cos(theta), ranges[usable] * np.sin(theta)))
    span = float(np.linalg.norm(points[-1] - points[0]))
    return span, float(np.mean(ranges[usable])), int(usable.size)


def _mouth_chord(
    start_outer_ray: int,
    end_outer_ray: int,
    angles_deg: np.ndarray,
    ranges: np.ndarray,
    angular_sigma_deg: float,
    noise_std_m: float,
    critical_z: float,
) -> tuple[float, float, float]:
    """Return observed chord, propagated sigma, and conservative lower bound."""
    indices = np.asarray([start_outer_ray, end_outer_ray], dtype=int)
    theta = np.radians(angles_deg[indices])
    radii = ranges[indices]
    points = np.column_stack((radii * np.cos(theta), radii * np.sin(theta)))
    chord = float(np.linalg.norm(points[1] - points[0]))
    angular_sigma_rad = math.radians(angular_sigma_deg)
    sigma = math.sqrt(
        2.0 * noise_std_m**2
        + angular_sigma_rad**2 * float(np.sum(radii**2))
    )
    return chord, sigma, max(0.0, chord - critical_z * sigma)


def _merge_core_gaps(
    core_mask: np.ndarray,
    angular_steps: np.ndarray,
    boundary_half_interval_deg: float,
) -> np.ndarray:
    """Merge only gaps whose two quantization intervals overlap."""
    return baseline._fill_short_circular_gaps(
        core_mask,
        angular_steps,
        2.0 * boundary_half_interval_deg,
    )


def _difference_model(
    ranges: np.ndarray,
    ceiling_mask: np.ndarray,
    noise_std_m: float,
    critical_z: float,
) -> tuple[np.ndarray, float, float]:
    """Build a robust, noise-aware threshold for adjacent range changes."""
    differences = np.roll(ranges, -1) - ranges
    background = np.abs(differences[~(ceiling_mask | np.roll(ceiling_mask, -1))])
    if background.size:
        center = float(np.median(background))
        robust_sigma = 1.4826 * float(np.median(np.abs(background - center)))
    else:
        center = 0.0
        robust_sigma = 0.0
    numerical_sigma = np.finfo(float).eps * max(1.0, float(np.max(ranges))) * 32.0
    difference_sigma = max(robust_sigma, math.sqrt(2.0) * noise_std_m, numerical_sigma)
    return differences, center + critical_z * difference_sigma, difference_sigma


def _expand_boundaries(
    run: np.ndarray,
    differences: np.ndarray,
    threshold_m: float,
) -> tuple[int, int, float, float]:
    """Follow one significant monotone ramp outward on each side of a core."""
    size = differences.size
    start_gradient = (int(run[0]) - 1) % size
    end_gradient = int(run[-1]) % size

    cursor = start_gradient
    start_score = max(0.0, float(differences[cursor]))
    visited = 0
    while differences[cursor] > threshold_m and visited < size - 1:
        start_gradient = cursor
        cursor = (cursor - 1) % size
        visited += 1

    cursor = end_gradient
    end_score = max(0.0, float(-differences[cursor]))
    visited = 0
    while -differences[cursor] > threshold_m and visited < size - 1:
        end_gradient = cursor
        cursor = (cursor + 1) % size
        visited += 1
    return start_gradient, end_gradient, start_score, end_score


def _detect_openings_with_diagnostics(
    angles_deg: Sequence[float],
    ranges: Sequence[float],
    *,
    noise_std_m: float = 0.0,
    robot_diameter_m: float = 2.70,
    false_alarm_probability: float = 0.01,
    stages: DetectorStages = DetectorStages(),
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Detect openings from permitted local observations and physical metadata.

    ``false_alarm_probability`` is a declared statistical family-wise error
    rate, not a geometry-tuned distance or angle.  The default robot diameter
    is the known 2 * 1.35 simulation robot radius and may be replaced by the
    real platform specification.
    """
    angles, raw, angular_steps = baseline._validate_circular_scan(angles_deg, ranges)
    if noise_std_m < 0.0:
        raise ValueError("noise_std_m must be non-negative")
    if robot_diameter_m <= 0.0:
        raise ValueError("robot_diameter_m must be positive")

    ray_count = raw.size
    critical_z = _normal_familywise_z(ray_count, false_alarm_probability)
    median_step = float(np.median(angular_steps))
    angular_sigma_deg = median_step / math.sqrt(12.0)
    boundary_half_interval_deg = critical_z * angular_sigma_deg
    range_ceiling = float(np.max(raw))
    numerical_tolerance = np.finfo(float).eps * max(1.0, range_ceiling) * 32.0
    ceiling_tolerance = max(critical_z * noise_std_m, numerical_tolerance)
    core_mask = raw >= range_ceiling - ceiling_tolerance
    if stages.uncertainty_merge:
        core_mask = _merge_core_gaps(core_mask, angular_steps, boundary_half_interval_deg)

    differences, difference_threshold, difference_sigma = _difference_model(
        raw, core_mask, noise_std_m, critical_z
    )
    boundary_angles = baseline._normalize_angles(angles + 0.5 * angular_steps)
    openings: list[dict[str, float]] = []

    for run in baseline._circular_runs(core_mask, value=True):
        if run.size == ray_count:
            continue
        if stages.adaptive_discontinuity:
            start_gradient, end_gradient, start_jump, end_jump = _expand_boundaries(
                run, differences, difference_threshold
            )
        else:
            start_gradient = (int(run[0]) - 1) % ray_count
            end_gradient = int(run[-1]) % ray_count
            start_jump = max(0.0, float(differences[start_gradient]))
            end_jump = max(0.0, float(-differences[end_gradient]))

        start_angle = float(boundary_angles[start_gradient])
        end_angle = float(boundary_angles[end_gradient])
        width_deg = baseline._positive_ccw_width(start_angle, end_angle)
        if width_deg <= 0.0 or width_deg >= 360.0 - median_step:
            continue

        start_outer_ray = start_gradient
        end_outer_ray = (end_gradient + 1) % ray_count
        mouth_width, mouth_sigma, mouth_lower = _mouth_chord(
            start_outer_ray,
            end_outer_ray,
            angles,
            raw,
            angular_sigma_deg,
            noise_std_m,
            critical_z,
        )
        if stages.physical_minimum_width and mouth_lower < robot_diameter_m:
            continue

        start_support_indices = _circular_indices(
            (start_gradient + 1) % ray_count,
            (int(run[0]) - 1) % ray_count,
            ray_count,
        )
        end_support_indices = _circular_indices(
            (int(run[-1]) + 1) % ray_count,
            end_gradient,
            ray_count,
        )
        start_span, start_range, start_count = _point_span(
            start_support_indices, angles, raw, core_mask
        )
        end_span, end_range, end_count = _point_span(
            end_support_indices, angles, raw, core_mask
        )
        angular_sigma_rad = math.radians(angular_sigma_deg)
        start_span_sigma = math.sqrt(2.0) * (
            noise_std_m + start_range * angular_sigma_rad
        )
        end_span_sigma = math.sqrt(2.0) * (
            noise_std_m + end_range * angular_sigma_rad
        )
        start_supported = start_span > critical_z * start_span_sigma
        end_supported = end_span > critical_z * end_span_sigma
        if stages.wall_support_uncertainty and not (start_supported or end_supported):
            continue

        center_angle = float(baseline._normalize_angles(start_angle + width_deg * 0.5))
        run_mean = float(np.mean(raw[run]))
        boundary_score = min(start_jump, end_jump) / max(
            difference_threshold, numerical_tolerance
        )
        confidence = float(np.clip(boundary_score / critical_z, 0.0, 1.0))
        openings.append({
            "start_angle": start_angle,
            "end_angle": end_angle,
            "center_angle": center_angle,
            "width_deg": float(width_deg),
            "mean_range_m": run_mean,
            "peak_range_m": float(np.max(raw[run])),
            "confidence": confidence,
            "start_refined": float(stages.adaptive_discontinuity),
            "end_refined": float(stages.adaptive_discontinuity),
            "boundary_uncertainty_deg": boundary_half_interval_deg,
            "estimated_mouth_width_m": mouth_width,
            "mouth_width_sigma_m": mouth_sigma,
            "mouth_width_lower_m": mouth_lower,
            "start_wall_support_span_m": start_span,
            "end_wall_support_span_m": end_span,
            "start_wall_support_count": float(start_count),
            "end_wall_support_count": float(end_count),
            "start_wall_supported": float(start_supported),
            "end_wall_supported": float(end_supported),
        })

    openings.sort(key=lambda item: item["center_angle"])
    diagnostics = {
        "range_ceiling_m": range_ceiling,
        "ceiling_tolerance_m": ceiling_tolerance,
        "open_core_mask": core_mask,
        "adjacent_range_difference_m": differences,
        "difference_sigma_m": difference_sigma,
        "difference_threshold_m": difference_threshold,
        "critical_z": critical_z,
        "angular_sigma_deg": angular_sigma_deg,
        "boundary_half_interval_deg": boundary_half_interval_deg,
        "robot_diameter_m": robot_diameter_m,
        "stages": stages,
    }
    return openings, diagnostics


def detect_openings(
    angles_deg: Sequence[float],
    ranges: Sequence[float],
    *,
    noise_std_m: float = 0.0,
    robot_diameter_m: float = 2.70,
    false_alarm_probability: float = 0.01,
    stages: DetectorStages = DetectorStages(),
) -> list[dict[str, float]]:
    """Return variable-count local opening intervals without map/GT inputs."""
    openings, _ = _detect_openings_with_diagnostics(
        angles_deg,
        ranges,
        noise_std_m=noise_std_m,
        robot_diameter_m=robot_diameter_m,
        false_alarm_probability=false_alarm_probability,
        stages=stages,
    )
    return openings


def detect_openings_from_point_cloud(
    point_cloud_theta_r: Sequence[Sequence[float]],
    **kwargs: Any,
) -> list[dict[str, float]]:
    """Detect from an ``(N, 2)`` Anchor-local ``[angle, range]`` array."""
    cloud = np.asarray(point_cloud_theta_r, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 2 or not np.all(np.isfinite(cloud)):
        raise ValueError("point_cloud_theta_r must be a finite (N, 2) array")
    order = np.argsort(cloud[:, 0])
    return detect_openings(cloud[order, 0], cloud[order, 1], **kwargs)


def run_synthetic_sanity() -> None:
    """Check wrap handling, resolution scaling, and rigid scan rotation."""
    angles = np.arange(-180.0, 180.0, 1.0)
    ranges = np.full(360, 4.0)
    ranges[(angles >= 170.0) | (angles < -170.0)] = 12.0
    ranges[(angles >= -4.0) & (angles < 5.0)] = 12.0
    first = detect_openings(angles, ranges, robot_diameter_m=0.5)
    assert len(first) == 2
    rotated_ranges = np.roll(ranges, 37)
    second = detect_openings(angles, rotated_ranges, robot_diameter_m=0.5)
    expected = sorted(baseline._normalize_angles(item["center_angle"] + 37.0) for item in first)
    actual = sorted(item["center_angle"] for item in second)
    assert np.allclose(expected, actual)
    coarse = detect_openings(
        angles[::4], ranges[::4], robot_diameter_m=0.5
    )
    assert len(coarse) == 2


if __name__ == "__main__":
    run_synthetic_sanity()
    print("uncertainty-aware detector sanity: PASS")
