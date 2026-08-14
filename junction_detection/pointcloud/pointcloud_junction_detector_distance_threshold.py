"""Distance-threshold experiment derived from the sensor-enhanced detector.

This module deliberately reuses the existing simulator, geometry, circular
moving average, connected components, and gradient boundary refinement from
``pointcloud_junction_detector_sensor_enhanced.py`` without modifying it.

The only detector experiment here is the opening-support threshold:

    statistical_margin = k * robust wall spread
    relative_margin = alpha * estimated local wall distance
    T_open = estimated local wall distance + max(statistical_margin, relative_margin)

Both terms are estimated from Anchor-local LiDAR ranges. No map, wall geometry,
branch count/direction, global Anchor pose, or central radius enters detection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from junction_detection.pointcloud import (
    pointcloud_junction_detector_sensor_enhanced as baseline,
)


EPSILON = baseline.EPSILON
LidarScan = baseline.LidarScan
cross2d = baseline.cross2d
ray_segment_distance = baseline.ray_segment_distance
simulate_lidar_scan = baseline.simulate_lidar_scan
make_n_way_junction_walls = baseline.make_n_way_junction_walls
evenly_spaced_branch_angles = baseline.evenly_spaced_branch_angles
smooth_ranges = baseline.smooth_ranges
circular_range_gradient = baseline.circular_range_gradient
save_local_scan_csv = baseline.save_local_scan_csv


def estimate_reference_wall_distance(
    smoothed_ranges: Sequence[float],
    raw_ranges: Sequence[float],
) -> tuple[float, float, dict[str, Any]]:
    """Estimate the dominant nearby-wall distance and its robust variation.

    The upper range ceiling is first excluded because exact/no-return readings
    describe open space or sensor saturation, not the Junction's nearby wall.
    A square-root-rule histogram is then formed from the remaining smoothed
    ranges. Among bins that reach the lower half of the distribution, the
    densest bin is treated as the short-range wall mode. This identifies a
    population instead of assuming that one fixed quantile is the wall.

    The median of that mode estimates ``d_wall``. Its scaled median absolute
    deviation (1.4826 * MAD) estimates ordinary variation around the wall mode
    while resisting long corridor returns and isolated sensor noise.

    Only range samples are used. The histogram bin count follows the standard
    square-root sample-size rule and does not encode a metric distance.
    """
    smoothed = np.asarray(smoothed_ranges, dtype=float)
    raw = np.asarray(raw_ranges, dtype=float)
    if smoothed.ndim != 1 or raw.ndim != 1 or smoothed.shape != raw.shape:
        raise ValueError("smoothed_ranges and raw_ranges must be equal-length 1D arrays")
    if smoothed.size < 8 or not np.all(np.isfinite(smoothed)) or not np.all(np.isfinite(raw)):
        raise ValueError("finite LiDAR scans with at least 8 samples are required")
    if np.any(smoothed < 0.0) or np.any(raw < 0.0):
        raise ValueError("LiDAR ranges cannot be negative")

    range_ceiling = float(np.max(raw))
    numerical_tolerance = max(
        1.0e-9,
        np.finfo(float).eps * max(1.0, abs(range_ceiling)) * 32.0,
    )
    non_ceiling = smoothed[smoothed < range_ceiling - numerical_tolerance]
    if non_ceiling.size < 4:
        raise ValueError("not enough non-ceiling ranges to estimate a nearby wall")

    bin_count = max(2, int(np.ceil(np.sqrt(non_ceiling.size))))
    counts, edges = np.histogram(non_ceiling, bins=bin_count)
    lower_guard = float(np.median(non_ceiling))
    lower_bin_indices = np.flatnonzero(edges[:-1] <= lower_guard)
    if lower_bin_indices.size == 0:
        raise RuntimeError("failed to identify a lower-distance histogram bin")
    modal_index = int(
        lower_bin_indices[np.argmax(counts[lower_bin_indices])]
    )

    lower_edge = float(edges[modal_index])
    upper_edge = float(edges[modal_index + 1])
    if modal_index == bin_count - 1:
        wall_population = non_ceiling[
            (non_ceiling >= lower_edge) & (non_ceiling <= upper_edge)
        ]
    else:
        wall_population = non_ceiling[
            (non_ceiling >= lower_edge) & (non_ceiling < upper_edge)
        ]
    if wall_population.size == 0:
        raise RuntimeError("estimated wall mode contains no samples")

    wall_reference = float(np.median(wall_population))
    wall_mad = float(np.median(np.abs(wall_population - wall_reference)))
    robust_wall_spread = 1.4826 * wall_mad
    diagnostics = {
        "estimator": "lower-distance histogram mode median + scaled MAD",
        "range_ceiling": range_ceiling,
        "non_ceiling_sample_count": int(non_ceiling.size),
        "histogram_bin_count": bin_count,
        "wall_mode_lower_m": lower_edge,
        "wall_mode_upper_m": upper_edge,
        "wall_population_count": int(wall_population.size),
        "wall_mad_m": wall_mad,
    }
    return wall_reference, robust_wall_spread, diagnostics


def _detect_openings_with_diagnostics(
    angles_deg: Sequence[float],
    ranges: Sequence[float],
    *,
    smoothing_window_size: int = 5,
    wall_margin_mad_scale: float = 6.0,
    relative_margin_ratio: float = 0.05,
    merge_gap_deg: float = 3.0,
    min_opening_width_deg: float = 5.0,
    gradient_threshold: Optional[float] = None,
    gradient_mad_scale: float = 4.0,
    min_gradient_threshold: float = 0.05,
    boundary_search_deg: float = 6.0,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Detect openings using a scan-derived wall-distance threshold.

    The connected-component and gradient-refinement stages intentionally match
    the existing sensor-enhanced baseline. Only open-support extraction changes.
    ``wall_margin_mad_scale`` and ``relative_margin_ratio`` are dimensionless
    experimental hyperparameters. The larger of their statistical and relative
    margins is used, avoiding both a fixed metric margin and a near-zero clean-
    scan threshold.
    """
    angles, raw, angular_steps = baseline._validate_circular_scan(angles_deg, ranges)
    if smoothing_window_size <= 0 or smoothing_window_size % 2 == 0:
        raise ValueError("smoothing_window_size must be a positive odd integer")
    if wall_margin_mad_scale <= 0.0:
        raise ValueError("wall_margin_mad_scale must be positive")
    if relative_margin_ratio < 0.0:
        raise ValueError("relative_margin_ratio must be non-negative")
    if merge_gap_deg < 0.0:
        raise ValueError("merge_gap_deg must be non-negative")
    if not 0.0 < min_opening_width_deg < 360.0:
        raise ValueError("min_opening_width_deg must be in (0, 360)")
    if boundary_search_deg < 0.0:
        raise ValueError("boundary_search_deg must be non-negative")

    smoothed = smooth_ranges(raw, smoothing_window_size)
    raw_stats = {
        "min": float(np.min(raw)),
        "median": float(np.median(raw)),
        "max": float(np.max(raw)),
        "mean": float(np.mean(raw)),
    }
    smoothed_stats = {
        "min": float(np.min(smoothed)),
        "median": float(np.median(smoothed)),
        "max": float(np.max(smoothed)),
        "mean": float(np.mean(smoothed)),
    }

    try:
        wall_reference, robust_wall_spread, reference_diagnostics = (
            estimate_reference_wall_distance(smoothed, raw)
        )
    except ValueError:
        diagnostics = {
            "smoothed_ranges": smoothed,
            "open_support_mask": np.zeros(raw.size, dtype=bool),
            "open_threshold": float(np.max(raw)),
            "wall_reference": float(np.min(smoothed)),
            "robust_wall_spread": 0.0,
            "statistical_margin": 0.0,
            "relative_margin": 0.0,
            "distance_margin": 0.0,
            "relative_margin_ratio": float(relative_margin_ratio),
            "selected_margin_source": "none",
            "range_ceiling": float(np.max(raw)),
            "boundary_angles": np.array([], dtype=float),
            "gradient": np.array([], dtype=float),
            "gradient_threshold": 0.0,
            "start_angles": [],
            "end_angles": [],
            "raw_stats": raw_stats,
            "smoothed_stats": smoothed_stats,
            "reference_estimator": {},
            "legacy_55_threshold_debug_only": float("nan"),
            "smoothing_window_size": int(smoothing_window_size),
        }
        return [], diagnostics

    # Preserve the prior machine-scale spread floor for numerical stability.
    # The relative term below provides the meaningful clean-scan minimum when
    # relative_margin_ratio > 0, without introducing a fixed metric distance.
    effective_spread = max(
        robust_wall_spread,
        np.finfo(float).eps * max(1.0, abs(wall_reference)) * 32.0,
    )
    statistical_margin = wall_margin_mad_scale * effective_spread
    relative_margin = relative_margin_ratio * wall_reference
    distance_margin = max(statistical_margin, relative_margin)
    selected_margin_source = (
        "statistical"
        if statistical_margin >= relative_margin
        else "relative"
    )
    open_threshold = wall_reference + distance_margin
    range_ceiling = float(np.max(raw))
    if open_threshold >= range_ceiling - EPSILON:
        open_support = np.zeros(raw.size, dtype=bool)
    else:
        open_support = smoothed > open_threshold
        open_support = baseline._fill_short_circular_gaps(
            open_support, angular_steps, merge_gap_deg
        )

    boundary_angles, gradient = circular_range_gradient(angles, smoothed)
    grad_threshold = (
        float(gradient_threshold)
        if gradient_threshold is not None
        else baseline._automatic_gradient_threshold(
            gradient, gradient_mad_scale, min_gradient_threshold
        )
    )
    median_step = float(np.median(angular_steps))
    search_radius_samples = int(np.ceil(boundary_search_deg / median_step))

    openings: list[dict[str, float]] = []
    dynamic_open_span = max(range_ceiling - open_threshold, EPSILON)
    for run in baseline._circular_runs(open_support, value=True):
        coarse_width = baseline._run_width_deg(run, angular_steps)
        if coarse_width < min_opening_width_deg or coarse_width >= 359.0:
            continue

        start_ray = int(run[0])
        end_ray = int(run[-1])
        coarse_start = baseline._boundary_angle_before_ray(
            start_ray, boundary_angles
        )
        coarse_end = baseline._boundary_angle_after_ray(end_ray, boundary_angles)
        start_angle, start_strength, start_refined = (
            baseline._refine_boundary_from_gradient(
                (start_ray - 1) % gradient.size,
                gradient,
                boundary_angles,
                positive=True,
                search_radius_samples=search_radius_samples,
                minimum_strength=grad_threshold,
                fallback_angle=coarse_start,
            )
        )
        end_angle, end_strength, end_refined = (
            baseline._refine_boundary_from_gradient(
                end_ray % gradient.size,
                gradient,
                boundary_angles,
                positive=False,
                search_radius_samples=search_radius_samples,
                minimum_strength=grad_threshold,
                fallback_angle=coarse_end,
            )
        )

        width = baseline._positive_ccw_width(start_angle, end_angle)
        if width < min_opening_width_deg or width > min(
            359.0, coarse_width + 2.0 * boundary_search_deg + 2.0
        ):
            start_angle, end_angle = coarse_start, coarse_end
            width = baseline._positive_ccw_width(start_angle, end_angle)
            start_refined = False
            end_refined = False
        if width < min_opening_width_deg:
            continue

        center_angle = float(
            baseline._normalize_angles(start_angle + width / 2.0)
        )
        mean_range = float(np.mean(smoothed[run]))
        peak_range = float(np.max(smoothed[run]))
        contrast_score = float(
            np.clip((mean_range - open_threshold) / dynamic_open_span, 0.0, 1.0)
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
                "start_angle": float(baseline._normalize_angles(start_angle)),
                "end_angle": float(baseline._normalize_angles(end_angle)),
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
    # Retained only as a labelled diagnostic comparison. It never determines
    # open_support or any result returned by this detector.
    legacy_debug_threshold = wall_reference + 0.55 * (
        range_ceiling - wall_reference
    )
    diagnostics = {
        "smoothed_ranges": smoothed,
        "open_support_mask": open_support,
        "open_threshold": float(open_threshold),
        "wall_reference": float(wall_reference),
        "robust_wall_spread": float(robust_wall_spread),
        "effective_wall_spread": float(effective_spread),
        "statistical_margin": float(statistical_margin),
        "relative_margin": float(relative_margin),
        "distance_margin": float(distance_margin),
        "wall_margin_mad_scale": float(wall_margin_mad_scale),
        "relative_margin_ratio": float(relative_margin_ratio),
        "selected_margin_source": selected_margin_source,
        "range_ceiling": range_ceiling,
        "boundary_angles": boundary_angles,
        "gradient": gradient,
        "gradient_threshold": float(grad_threshold),
        "start_angles": [opening["start_angle"] for opening in openings],
        "end_angles": [opening["end_angle"] for opening in openings],
        "raw_stats": raw_stats,
        "smoothed_stats": smoothed_stats,
        "reference_estimator": reference_diagnostics,
        "legacy_55_threshold_debug_only": float(legacy_debug_threshold),
        "smoothing_window_size": int(smoothing_window_size),
    }
    return openings, diagnostics


def detect_openings(
    angles_deg: Sequence[float],
    ranges: Sequence[float],
    **kwargs: Any,
) -> list[dict[str, float]]:
    """Detect sectors using only Anchor-local LiDAR angle and range arrays."""
    openings, _ = _detect_openings_with_diagnostics(angles_deg, ranges, **kwargs)
    return openings


def detect_openings_from_point_cloud(
    point_cloud_theta_r: Sequence[Sequence[float]],
    **kwargs: Any,
) -> list[dict[str, float]]:
    """Detect openings from an ``(N, 2)`` local ``[angle, range]`` cloud."""
    cloud = np.asarray(point_cloud_theta_r, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 2:
        raise ValueError("point_cloud_theta_r must have shape (N, 2)")
    if cloud.shape[0] < 8 or not np.all(np.isfinite(cloud)):
        raise ValueError("at least 8 finite point-cloud samples are required")
    order = np.argsort(cloud[:, 0])
    return detect_openings(cloud[order, 0], cloud[order, 1], **kwargs)


def run_case(
    branch_angles_deg: Sequence[float],
    *,
    anchor_xy: tuple[float, float] = (0.0, 0.0),
    anchor_yaw_deg: float = 0.0,
    corridor_width_m: float = 2.0,
    central_radius_m: float = 1.8,
    branch_length_m: float = 10.0,
    max_range_m: float = 6.0,
    noise_std_m: float = 0.01,
    dropout_probability: float = 0.0,
    occlusion_probability: float = 0.0,
    visible_boundary_ratio: float = 1.0,
    angle_step_deg: float = 1.0,
    seed: int = 7,
    detector_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[np.ndarray, LidarScan, list[dict[str, float]], dict[str, Any]]:
    """Simulate one test case and keep geometry outside the detector boundary."""
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
    kwargs = {} if detector_kwargs is None else dict(detector_kwargs)
    openings, diagnostics = _detect_openings_with_diagnostics(
        detector_angles, detector_ranges, **kwargs
    )
    return walls, scan, openings, diagnostics


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
    """Show geometry, point cloud, raw, smoothed, and threshold stages."""
    anchor = np.asarray(anchor_xy, dtype=float)
    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    world_axis, cloud_axis, raw_axis = axes[0]
    smooth_axis, detection_axis, gradient_axis = axes[1]

    for wall in walls:
        world_axis.plot(
            [wall[0, 0], wall[1, 0]], [wall[0, 1], wall[1, 1]], linewidth=1.5
        )
    world_axis.scatter(anchor[0], anchor[1], marker="x", s=80, label="Anchor")
    yaw = np.deg2rad(anchor_yaw_deg)
    world_axis.arrow(
        anchor[0], anchor[1], 0.7 * np.cos(yaw), 0.7 * np.sin(yaw),
        width=0.015, length_includes_head=True,
    )
    world_axis.set_title("A. Simulator-only geometry")
    world_axis.set_xlabel("world x [m]")
    world_axis.set_ylabel("world y [m]")
    world_axis.set_aspect("equal", adjustable="box")
    world_axis.grid(alpha=0.3)
    world_axis.legend(fontsize=8)

    cloud_axis.scatter(
        scan.local_x[scan.hit], scan.local_y[scan.hit], s=9, label="LiDAR returns"
    )
    if np.any(~scan.hit):
        cloud_axis.scatter(
            scan.local_x[~scan.hit], scan.local_y[~scan.hit],
            s=7, alpha=0.18, label="no return / max range",
        )
    cloud_axis.scatter(0.0, 0.0, marker="x", s=80, label="Anchor local origin")
    for index, opening in enumerate(openings):
        theta = np.deg2rad(opening["center_angle"])
        endpoint = 0.85 * scan.max_range_m * np.array([np.cos(theta), np.sin(theta)])
        cloud_axis.plot(
            [0.0, endpoint[0]], [0.0, endpoint[1]], linestyle="--", linewidth=1.1,
            label="detected center" if index == 0 else None,
        )
    cloud_axis.set_title("B. Anchor-local point cloud")
    cloud_axis.set_xlabel("local x [m]")
    cloud_axis.set_ylabel("local y [m]")
    cloud_axis.set_aspect("equal", adjustable="box")
    cloud_axis.grid(alpha=0.3)
    cloud_axis.legend(fontsize=8)

    smoothed = np.asarray(diagnostics["smoothed_ranges"], dtype=float)
    raw_axis.plot(scan.angle_deg, scan.range_m, color="tab:blue", linewidth=0.9)
    raw_axis.set_title("C. Raw LiDAR range")
    raw_axis.set_ylabel("range [m]")

    smooth_axis.plot(
        scan.angle_deg, scan.range_m, color="tab:blue", linewidth=0.7,
        alpha=0.3, label="raw",
    )
    smooth_axis.plot(
        scan.angle_deg, smoothed, color="black", linewidth=1.3,
        label=(
            "circular moving average "
            f"(window={diagnostics['smoothing_window_size']})"
        ),
    )
    smooth_axis.set_title("D. Raw vs moving average")
    smooth_axis.set_ylabel("range [m]")
    smooth_axis.legend(fontsize=8)

    open_mask = np.asarray(diagnostics["open_support_mask"], dtype=bool)
    detection_axis.plot(
        scan.angle_deg, smoothed, color="black", linewidth=1.25, label="smoothed"
    )
    detection_axis.axhline(
        diagnostics["wall_reference"], color="tab:orange", linestyle="-.",
        linewidth=1.3, label="estimated reference wall distance",
    )
    detection_axis.axhline(
        diagnostics["open_threshold"], color="tab:red", linestyle="--",
        linewidth=1.3,
        label=(
            "threshold = wall + max(statistical, relative) "
            f"[{diagnostics['selected_margin_source']}]"
        ),
    )
    if np.any(open_mask):
        detection_axis.scatter(
            scan.angle_deg[open_mask], smoothed[open_mask], s=8,
            color="tab:green", alpha=0.35, label="open support",
        )
    for index, opening in enumerate(openings):
        start, end = opening["start_angle"], opening["end_angle"]
        label = "detected opening" if index == 0 else None
        if start <= end:
            detection_axis.axvspan(start, end, color="tab:green", alpha=0.10, label=label)
        else:
            detection_axis.axvspan(start, 180.0, color="tab:green", alpha=0.10, label=label)
            detection_axis.axvspan(-180.0, end, color="tab:green", alpha=0.10)
    detection_axis.set_title("E. Distance-threshold opening support")
    detection_axis.set_xlabel("Anchor-local angle [deg]")
    detection_axis.set_ylabel("range [m]")
    detection_axis.legend(fontsize=7, loc="upper right")

    gradient_axis.plot(
        diagnostics["boundary_angles"], diagnostics["gradient"],
        color="tab:purple", linewidth=0.8,
    )
    gradient_threshold = float(diagnostics["gradient_threshold"])
    gradient_axis.axhline(gradient_threshold, color="tab:red", linestyle=":")
    gradient_axis.axhline(-gradient_threshold, color="tab:red", linestyle=":")
    gradient_axis.set_title("F. Boundary-refinement range gradient")
    gradient_axis.set_xlabel("Anchor-local angle [deg]")
    gradient_axis.set_ylabel("range gradient [m/deg]")

    for axis in (raw_axis, smooth_axis, detection_axis, gradient_axis):
        axis.set_xlim(float(scan.angle_deg[0]), float(scan.angle_deg[-1]))
        axis.grid(alpha=0.3)
    for axis in (raw_axis, smooth_axis, detection_axis):
        axis.set_ylim(0.0, scan.max_range_m * 1.05)

    figure.tight_layout()
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180, bbox_inches="tight")
        print(f"[saved] {path}")
    if show:
        plt.show()
    else:
        plt.close(figure)


def regression_test_way_counts(
    way_counts: Sequence[int] = (3, 4, 5),
    *,
    noise_std_m: float = 0.01,
    wall_margin_mad_scale: float = 6.0,
    relative_margin_ratio: float = 0.05,
) -> list[tuple[int, int, bool]]:
    """Compare synthetic expected count outside the count-agnostic detector."""
    results: list[tuple[int, int, bool]] = []
    for count in way_counts:
        central_radius = max(1.8, 0.55 * count)
        branch_angles = evenly_spaced_branch_angles(count, rotation_deg=11.0)
        _, _, openings, _ = run_case(
            branch_angles,
            central_radius_m=central_radius,
            noise_std_m=noise_std_m,
            dropout_probability=0.0,
            occlusion_probability=0.0,
            visible_boundary_ratio=1.0,
            anchor_xy=(0.0, 0.0),
            anchor_yaw_deg=23.0,
            detector_kwargs={
                "wall_margin_mad_scale": wall_margin_mad_scale,
                "relative_margin_ratio": relative_margin_ratio,
            },
        )
        detected = len(openings)
        results.append((count, detected, detected == count))
    return results


def _parse_branch_angles(
    text: Optional[str], ways: int, rotation_deg: float
) -> np.ndarray:
    """Parse simulator-only directions or create evenly spaced test branches."""
    if text:
        values = [float(part.strip()) for part in text.split(",") if part.strip()]
        if not values:
            raise ValueError("--branch-angles did not contain any angles")
        return np.asarray(values, dtype=float)
    return evenly_spaced_branch_angles(ways, rotation_deg=rotation_deg)


def _print_statistics(label: str, statistics: dict[str, float]) -> None:
    print(
        f"{label}: min={statistics['min']:.4f} m, "
        f"median={statistics['median']:.4f} m, "
        f"mean={statistics['mean']:.4f} m, max={statistics['max']:.4f} m"
    )


def main() -> None:
    """Run the weak-noise five-way distance-threshold baseline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ways", type=int, default=5, help="SIMULATOR TEST ONLY")
    parser.add_argument("--branch-angles", type=str, default=None, help="SIMULATOR TEST ONLY")
    parser.add_argument("--rotation", type=float, default=11.0)
    parser.add_argument("--corridor-width", type=float, default=2.0)
    parser.add_argument("--central-radius", type=float, default=None)
    parser.add_argument("--branch-length", type=float, default=10.0)
    parser.add_argument("--anchor-x", type=float, default=0.0)
    parser.add_argument("--anchor-y", type=float, default=0.0)
    parser.add_argument("--anchor-yaw", type=float, default=23.0)
    parser.add_argument("--angle-step", type=float, default=1.0)
    parser.add_argument("--max-range", type=float, default=6.0)
    # 0.01 m is an experimental starting point, not a professor-specified or
    # claimed real-sensor noise model. Robustness degradations stay available.
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--occlusion", type=float, default=0.0)
    parser.add_argument("--visible-boundary-ratio", type=float, default=1.0)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--wall-margin-mad-scale", type=float, default=6.0)
    parser.add_argument(
        "--relative-margin-ratio",
        type=float,
        default=0.05,
        help=(
            "experimental minimum margin as a fraction of estimated wall "
            "distance; 0.05 is a starting value, not a calibrated constant"
        ),
    )
    parser.add_argument("--merge-gap-deg", type=float, default=3.0)
    parser.add_argument("--min-opening-width-deg", type=float, default=5.0)
    parser.add_argument("--boundary-search-deg", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--csv", type=str, default="distance_threshold_local_scan.csv")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--regression", action="store_true")
    args = parser.parse_args()

    if args.regression:
        print("=== Distance-threshold way-count sanity test ===")
        for expected, detected, passed in regression_test_way_counts(
            noise_std_m=args.noise,
            wall_margin_mad_scale=args.wall_margin_mad_scale,
            relative_margin_ratio=args.relative_margin_ratio,
        ):
            print(f"N={expected}: detected={detected} -> {'PASS' if passed else 'FAIL'}")
        return

    branch_angles = _parse_branch_angles(args.branch_angles, args.ways, args.rotation)
    central_radius = (
        max(1.8, 0.55 * len(branch_angles))
        if args.central_radius is None
        else float(args.central_radius)
    )
    detector_kwargs = {
        "smoothing_window_size": args.smoothing_window,
        "wall_margin_mad_scale": args.wall_margin_mad_scale,
        "relative_margin_ratio": args.relative_margin_ratio,
        "merge_gap_deg": args.merge_gap_deg,
        "min_opening_width_deg": args.min_opening_width_deg,
        "boundary_search_deg": args.boundary_search_deg,
    }
    walls, scan, openings, diagnostics = run_case(
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
        seed=args.seed,
        detector_kwargs=detector_kwargs,
    )
    save_local_scan_csv(scan, args.csv)

    print("=== Distance-Based Point-Cloud Opening Detector ===")
    print("Baseline noise 0.01 m is an experimental starting value, not a specified sensor model.")
    print(
        f"Sensor: noise={args.noise:.4f} m, dropout={args.dropout:.3f}, "
        f"occlusion={args.occlusion:.3f}, visibility={args.visible_boundary_ratio:.3f}"
    )
    print(f"Circular moving-average window: {args.smoothing_window}")
    print("Detector input: Anchor-local angle/range only")
    print(f"Synthetic expected count (test harness only): {len(branch_angles)}")
    _print_statistics("Raw ranges", diagnostics["raw_stats"])
    _print_statistics("Smoothed ranges", diagnostics["smoothed_stats"])
    print(f"Reference estimator: {diagnostics['reference_estimator']['estimator']}")
    print(f"Estimated reference wall distance: {diagnostics['wall_reference']:.4f} m")
    print(f"Robust wall spread (1.4826*MAD): {diagnostics['robust_wall_spread']:.6f} m")
    print(f"Statistical margin: {diagnostics['statistical_margin']:.4f} m")
    print(f"Relative wall-distance margin: {diagnostics['relative_margin']:.4f} m")
    print(f"Selected distance margin: {diagnostics['distance_margin']:.4f} m")
    print(f"Selected margin source: {diagnostics['selected_margin_source']}")
    print(f"Wall MAD scale k: {diagnostics['wall_margin_mad_scale']:.2f}")
    print(f"Relative margin ratio: {diagnostics['relative_margin_ratio']:.4f}")
    print(f"Final opening threshold: {diagnostics['open_threshold']:.4f} m")
    print(
        "Legacy 55% threshold (debug only, not used): "
        f"{diagnostics['legacy_55_threshold_debug_only']:.4f} m"
    )
    print(f"Detected openings: {len(openings)}")
    for index, opening in enumerate(openings):
        print(
            f"  Opening {index}: start={opening['start_angle']:.1f} deg, "
            f"end={opening['end_angle']:.1f} deg, "
            f"center={opening['center_angle']:.1f} deg, "
            f"width={opening['width_deg']:.1f} deg, "
            f"confidence={opening['confidence']:.2f}"
        )

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
