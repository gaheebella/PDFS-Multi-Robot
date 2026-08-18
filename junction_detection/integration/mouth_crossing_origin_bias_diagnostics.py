"""Evaluation-only A/B diagnostics for mouth sample origin bias.

Production heading-aligned origins are compared with interpolated geometric
mouth-plane crossings for exactly matched robot IDs. Results are never exposed
to detector, PCA lock, handoff, or control decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from junction_detection.integration.mouth_pca_sample_distribution_diagnostics import (
    _pca,
    _quantile,
    _signed_angle_delta,
    _write_rows,
)


BRANCH_ORDER = ("LEFT", "UP", "RIGHT")


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean, or NaN for an empty sequence."""
    return sum(values) / len(values) if values else float("nan")


def _population_std(values: Sequence[float]) -> float:
    """Return population standard deviation."""
    if not values:
        return float("nan")
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def _pearson(first: Sequence[float], second: Sequence[float]) -> float:
    """Return Pearson correlation without adding a significance threshold."""
    if len(first) != len(second) or len(first) < 2:
        return float("nan")
    first_mean, second_mean = _mean(first), _mean(second)
    numerator = sum((x - first_mean) * (y - second_mean) for x, y in zip(first, second))
    denominator = math.sqrt(
        sum((x - first_mean) ** 2 for x in first)
        * sum((y - second_mean) ** 2 for y in second)
    )
    return numerator / denominator if denominator > 1.0e-12 else float("nan")


def _ranks(values: Sequence[float]) -> list[float]:
    """Return average ranks, preserving ties for Spearman correlation."""
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = 0.5 * (start + end - 1)
        for offset in range(start, end):
            result[indexed[offset][0]] = rank
        start = end
    return result


def _spearman(first: Sequence[float], second: Sequence[float]) -> float:
    """Return Spearman rank correlation."""
    return _pearson(_ranks(first), _ranks(second))


def _linear_fit(independent: Sequence[float], dependent: Sequence[float]) -> tuple[float, float, float]:
    """Return ordinary least-squares slope, intercept, and R squared."""
    if len(independent) != len(dependent) or len(independent) < 2:
        return float("nan"), float("nan"), float("nan")
    x_mean, y_mean = _mean(independent), _mean(dependent)
    denominator = sum((x - x_mean) ** 2 for x in independent)
    if denominator <= 1.0e-12:
        return float("nan"), y_mean, float("nan")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(independent, dependent)) / denominator
    intercept = y_mean - slope * x_mean
    residual = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(independent, dependent))
    total = sum((y - y_mean) ** 2 for y in dependent)
    r_squared = 1.0 - residual / total if total > 1.0e-12 else float("nan")
    return slope, intercept, r_squared


@dataclass
class MouthCrossingOriginBiasDiagnostics:
    """Collect first crossings and production origins, then run matched A/B PCA."""

    gt_crossings: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    heading_origins: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    comparison_rows: list[dict[str, Any]] = field(default_factory=list)
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    pca_rows: list[dict[str, Any]] = field(default_factory=list)
    temporal_rows: list[dict[str, Any]] = field(default_factory=list)
    snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    _crossing_counts: dict[str, int] = field(default_factory=dict)

    def record_gt_crossing(self, row: Mapping[str, Any]) -> None:
        """Store the first outward geometric mouth-plane crossing per robot/Branch."""
        key = (str(row["branch"]), int(row["robot_id"]))
        if key in self.gt_crossings:
            return
        branch = key[0]
        stored = dict(row)
        stored["crossing_order"] = self._crossing_counts.get(branch, 0)
        self._crossing_counts[branch] = int(stored["crossing_order"]) + 1
        self.gt_crossings[key] = stored

    def record_heading_origin(self, row: Mapping[str, Any]) -> None:
        """Store the production setdefault pose without modifying that dictionary."""
        key = (str(row["branch"]), int(row["robot_id"]))
        self.heading_origins.setdefault(key, dict(row))

    def record_handoff_outcome(self, branch: str, success: bool) -> None:
        """Attach observed lifecycle outcome after the existing resolver decides."""
        result = "SUCCESS" if success else "FAILED"
        for row in self.pca_rows:
            if row["branch"] == branch:
                row["result"] = result

    def record_snapshot(
        self,
        *,
        branch: str,
        frame: int,
        timestamp: float,
        production_samples: Mapping[int, tuple[float, float]],
        flow_xy: tuple[float, float],
        gt_origin_xy: tuple[float, float],
        gt_tangent_xy: tuple[float, float],
        gt_yaw_deg: float,
    ) -> None:
        """Analyze only robot IDs possessing both production and GT positions."""
        if branch in self.snapshots:
            return
        production_ids = set(int(robot_id) for robot_id in production_samples)
        crossing_ids = {robot_id for candidate, robot_id in self.gt_crossings if candidate == branch}
        matched_ids = sorted(production_ids & crossing_ids)
        rows: list[dict[str, Any]] = []
        origin_x, origin_y = gt_origin_xy
        tangent_x, tangent_y = gt_tangent_xy
        normal_x, normal_y = -tangent_y, tangent_x
        for robot_id in matched_ids:
            heading_x, heading_y = production_samples[robot_id]
            crossing = self.gt_crossings[(branch, robot_id)]
            heading = self.heading_origins.get((branch, robot_id), {})
            crossing_x = float(crossing["crossing_world_x"])
            crossing_y = float(crossing["crossing_world_y"])
            heading_dx, heading_dy = heading_x - origin_x, heading_y - origin_y
            crossing_dx, crossing_dy = crossing_x - origin_x, crossing_y - origin_y
            heading_axial = heading_dx * tangent_x + heading_dy * tangent_y
            heading_lateral = heading_dx * normal_x + heading_dy * normal_y
            crossing_axial = crossing_dx * tangent_x + crossing_dy * tangent_y
            crossing_lateral = crossing_dx * normal_x + crossing_dy * normal_y
            axial_error = heading_axial - crossing_axial
            lateral_error = heading_lateral - crossing_lateral
            heading_time = float(heading.get("heading_sample_time", timestamp))
            crossing_time = float(crossing["mouth_crossing_time"])
            rows.append({
                "branch": branch,
                "robot_id": robot_id,
                "crossing_order": crossing["crossing_order"],
                "mouth_crossing_frame": crossing["frame"],
                "heading_sample_frame": heading.get("frame", frame),
                "mouth_crossing_time": crossing_time,
                "heading_sample_time": heading_time,
                "crossing_world_x": crossing_x,
                "crossing_world_y": crossing_y,
                "heading_world_x": heading_x,
                "heading_world_y": heading_y,
                "crossing_axial": crossing_axial,
                "crossing_lateral": crossing_lateral,
                "heading_axial": heading_axial,
                "heading_lateral": heading_lateral,
                "axial_error": axial_error,
                "lateral_error": lateral_error,
                "euclidean_error": math.hypot(heading_x - crossing_x, heading_y - crossing_y),
                "time_delay": heading_time - crossing_time,
                "crossing_interpolation_alpha": crossing["interpolation_alpha"],
                "evaluation_only_gt": True,
            })
        rows.sort(key=lambda row: (int(row["crossing_order"]), int(row["robot_id"])))
        self.comparison_rows.extend(rows)
        lateral = [float(row["crossing_lateral"]) for row in rows]
        axial_error = [float(row["axial_error"]) for row in rows]
        crossing_order = [float(row["crossing_order"]) for row in rows]
        crossing_time = [float(row["mouth_crossing_time"]) for row in rows]
        slope, intercept, r_squared = _linear_fit(lateral, axial_error)
        order_slope, _, order_r_squared = _linear_fit(crossing_order, axial_error)
        time_slope, _, time_r_squared = _linear_fit(crossing_time, axial_error)
        delays = [float(row["time_delay"]) for row in rows]
        lateral_error = [float(row["lateral_error"]) for row in rows]
        self.summary_rows.append({
            "branch": branch,
            "production_sample_count": len(production_ids),
            "gt_crossing_count": len(crossing_ids),
            "matched_count": len(matched_ids),
            "production_only_count": len(production_ids - crossing_ids),
            "gt_only_count": len(crossing_ids - production_ids),
            "mean_axial_error": _mean(axial_error),
            "median_axial_error": _quantile(axial_error, 0.50),
            "std_axial_error": _population_std(axial_error),
            "minimum_axial_error": min(axial_error),
            "maximum_axial_error": max(axial_error),
            "mean_lateral_error": _mean(lateral_error),
            "std_lateral_error": _population_std(lateral_error),
            "mean_euclidean_error": _mean([float(row["euclidean_error"]) for row in rows]),
            "mean_time_delay": _mean(delays),
            "median_time_delay": _quantile(delays, 0.50),
            "lateral_vs_axial_error_pearson": _pearson(lateral, axial_error),
            "lateral_vs_axial_error_spearman": _spearman(lateral, axial_error),
            "lateral_vs_axial_error_slope": slope,
            "lateral_vs_axial_error_intercept": intercept,
            "lateral_vs_axial_error_r_squared": r_squared,
            "crossing_order_vs_axial_error_pearson": _pearson(crossing_order, axial_error),
            "crossing_order_vs_axial_error_slope": order_slope,
            "crossing_order_vs_axial_error_r_squared": order_r_squared,
            "crossing_time_vs_axial_error_pearson": _pearson(crossing_time, axial_error),
            "crossing_time_vs_axial_error_slope": time_slope,
            "crossing_time_vs_axial_error_r_squared": time_r_squared,
            "pca_frame": frame,
            "pca_timestamp": timestamp,
        })
        production_pca_rows = [
            {"sample_world_x": float(production_samples[robot_id][0]), "sample_world_y": float(production_samples[robot_id][1])}
            for robot_id in matched_ids
        ]
        crossing_pca_rows = [
            {"sample_world_x": float(self.gt_crossings[(branch, robot_id)]["crossing_world_x"]), "sample_world_y": float(self.gt_crossings[(branch, robot_id)]["crossing_world_y"])}
            for robot_id in matched_ids
        ]
        production_raw = _pca(production_pca_rows, flow_xy, radial_trim=False)
        production_inlier = _pca(production_pca_rows, flow_xy, radial_trim=True)
        crossing_raw = _pca(crossing_pca_rows, flow_xy, radial_trim=False)
        crossing_inlier = _pca(crossing_pca_rows, flow_xy, radial_trim=True)
        production_error = abs(_signed_angle_delta(production_inlier["yaw_deg"], gt_yaw_deg))
        crossing_error = abs(_signed_angle_delta(crossing_inlier["yaw_deg"], gt_yaw_deg))
        reduction = production_error - crossing_error
        self.pca_rows.append({
            "branch": branch,
            "matched_sample_count": len(matched_ids),
            "production_raw_count": len(production_pca_rows),
            "production_inlier_count": len(production_inlier["inlier_indices"]),
            "production_trim_count": len(production_pca_rows) - len(production_inlier["inlier_indices"]),
            "crossing_raw_count": len(crossing_pca_rows),
            "crossing_inlier_count": len(crossing_inlier["inlier_indices"]),
            "crossing_trim_count": len(crossing_pca_rows) - len(crossing_inlier["inlier_indices"]),
            "production_raw_yaw_deg": production_raw["yaw_deg"],
            "production_inlier_yaw_deg": production_inlier["yaw_deg"],
            "crossing_raw_yaw_deg": crossing_raw["yaw_deg"],
            "crossing_inlier_yaw_deg": crossing_inlier["yaw_deg"],
            "gt_yaw_deg": gt_yaw_deg,
            "production_error_deg": production_error,
            "crossing_error_deg": crossing_error,
            "error_reduction_deg": reduction,
            "error_reduction_percent": 100.0 * reduction / production_error if production_error > 1.0e-12 else 0.0,
            "residual_bias_deg": crossing_error,
            "production_raw_covariance_xy": production_raw["covariance_xy"],
            "production_inlier_covariance_xy": production_inlier["covariance_xy"],
            "crossing_raw_covariance_xy": crossing_raw["covariance_xy"],
            "crossing_inlier_covariance_xy": crossing_inlier["covariance_xy"],
            "production_raw_anisotropy": production_raw["anisotropy"],
            "production_inlier_anisotropy": production_inlier["anisotropy"],
            "crossing_raw_anisotropy": crossing_raw["anisotropy"],
            "crossing_inlier_anisotropy": crossing_inlier["anisotropy"],
            "result": "NOT_OBSERVED",
            "evaluation_only_gt": True,
        })
        boundaries = [round(index * len(rows) / 3) for index in range(4)]
        for index, label in enumerate(("EARLY", "MIDDLE", "LATE")):
            group = rows[boundaries[index]:boundaries[index + 1]]
            group_lateral = [float(row["crossing_lateral"]) for row in group]
            group_axial_error = [float(row["axial_error"]) for row in group]
            group_slope, _, group_r_squared = _linear_fit(group_lateral, group_axial_error)
            self.temporal_rows.append({
                "branch": branch,
                "segment": label,
                "sample_count": len(group),
                "first_crossing_order": group[0]["crossing_order"],
                "last_crossing_order": group[-1]["crossing_order"],
                "first_crossing_time": group[0]["mouth_crossing_time"],
                "last_crossing_time": group[-1]["mouth_crossing_time"],
                "mean_axial_error": _mean(group_axial_error),
                "median_axial_error": _quantile(group_axial_error, 0.50),
                "std_axial_error": _population_std(group_axial_error),
                "mean_lateral_error": _mean([float(row["lateral_error"]) for row in group]),
                "mean_time_delay": _mean([float(row["time_delay"]) for row in group]),
                "lateral_vs_axial_error_pearson": _pearson(group_lateral, group_axial_error),
                "lateral_vs_axial_error_slope": group_slope,
                "lateral_vs_axial_error_r_squared": group_r_squared,
            })
        self.snapshots[branch] = {
            "rows": rows,
            "production_pca": production_inlier,
            "crossing_pca": crossing_inlier,
            "gt_origin_xy": gt_origin_xy,
            "gt_tangent_xy": gt_tangent_xy,
        }

    def save(self, output_dir: str | Path) -> None:
        """Validate and save four CSVs plus six required figures."""
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        _write_rows(directory / "mouth_crossing_vs_heading_samples.csv", self.comparison_rows)
        _write_rows(directory / "mouth_crossing_branch_summary.csv", self.summary_rows)
        _write_rows(directory / "mouth_crossing_pca_comparison.csv", self.pca_rows)
        _write_rows(directory / "mouth_crossing_temporal_bias.csv", self.temporal_rows)
        self._validate()
        self._save_plots(directory)

    def _validate(self) -> None:
        """Assert matched IDs and exact production RIGHT PCA reproduction."""
        if set(self.snapshots) != set(BRANCH_ORDER):
            raise AssertionError("missing mouth crossing A/B snapshot")
        if len({(row["branch"], int(row["robot_id"])) for row in self.comparison_rows}) != len(self.comparison_rows):
            raise AssertionError("duplicate matched robot row")
        right = next(row for row in self.pca_rows if row["branch"] == "RIGHT")
        if abs(float(right["production_inlier_yaw_deg"]) - 0.763325338752395) > 1.0e-9:
            raise AssertionError("RIGHT production PCA was not reproduced")

    def _save_plots(self, directory: Path) -> None:
        """Render matched-pair, correlation, and PCA A/B visualizations."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        for branch in BRANCH_ORDER:
            rows = [row for row in self.comparison_rows if row["branch"] == branch]
            figure, axis = plt.subplots(figsize=(7, 5.5))
            for row in rows:
                axis.plot(
                    [float(row["crossing_lateral"]), float(row["heading_lateral"])],
                    [float(row["crossing_axial"]), float(row["heading_axial"])],
                    color="0.75",
                    linewidth=0.6,
                    zorder=1,
                )
            axis.scatter([float(row["crossing_lateral"]) for row in rows], [float(row["crossing_axial"]) for row in rows], s=18, label="interpolated GT crossing", color="black", zorder=3)
            axis.scatter([float(row["heading_lateral"]) for row in rows], [float(row["heading_axial"]) for row in rows], s=18, label="production heading origin", color="tab:blue", zorder=2)
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set(title=f"{branch}: production origin vs geometric crossing", xlabel="GT-local lateral", ylabel="GT-local axial")
            axis.legend()
            axis.grid(alpha=0.25)
            figure.tight_layout()
            figure.savefig(directory / f"heading_vs_true_crossing_scatter_{branch.lower()}.png", dpi=170)
            plt.close(figure)

        figure, axis = plt.subplots(figsize=(9, 5.5))
        colors = {"LEFT": "tab:blue", "UP": "tab:orange", "RIGHT": "tab:green"}
        for branch in BRANCH_ORDER:
            rows = [row for row in self.comparison_rows if row["branch"] == branch]
            x = [float(row["crossing_lateral"]) for row in rows]
            y = [float(row["axial_error"]) for row in rows]
            summary = next(row for row in self.summary_rows if row["branch"] == branch)
            slope = float(summary["lateral_vs_axial_error_slope"])
            intercept = float(summary["lateral_vs_axial_error_intercept"])
            axis.scatter(x, y, s=18, alpha=0.65, color=colors[branch], label=f"{branch} samples")
            fit_x = np.linspace(min(x), max(x), 100)
            axis.plot(fit_x, slope * fit_x + intercept, color=colors[branch], linewidth=2, label=f"{branch} fit (R²={float(summary['lateral_vs_axial_error_r_squared']):.3f})")
        axis.set(title="Heading-origin axial error vs true crossing lateral", xlabel="GT crossing lateral", ylabel="heading axial error")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(directory / "axial_error_vs_lateral_by_branch.png", dpi=170)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8, 4.8))
        branches = [row["branch"] for row in self.pca_rows]
        x = np.arange(len(branches), dtype=float)
        width = 0.34
        axis.bar(x - width / 2, [float(row["production_error_deg"]) for row in self.pca_rows], width, label="production origin PCA")
        axis.bar(x + width / 2, [float(row["crossing_error_deg"]) for row in self.pca_rows], width, label="GT crossing PCA")
        axis.set(xticks=x, xticklabels=branches, ylabel="absolute yaw error [deg]", title="Production vs geometric-crossing PCA")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(directory / "production_vs_crossing_pca_yaw.png", dpi=170)
        plt.close(figure)

        right = self.snapshots["RIGHT"]
        rows = right["rows"]
        figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        for axis, source in zip(axes, ("production", "crossing")):
            lateral_key = "heading_lateral" if source == "production" else "crossing_lateral"
            axial_key = "heading_axial" if source == "production" else "crossing_axial"
            axis.scatter([float(row[lateral_key]) for row in rows], [float(row[axial_key]) for row in rows], s=20)
            pca = right[f"{source}_pca"]
            tangent_x, tangent_y = float(pca["tangent_x"]), float(pca["tangent_y"])
            gt_tangent_x, gt_tangent_y = right["gt_tangent_xy"]
            gt_normal_x, gt_normal_y = -gt_tangent_y, gt_tangent_x
            local_tangent = (
                tangent_x * gt_normal_x + tangent_y * gt_normal_y,
                tangent_x * gt_tangent_x + tangent_y * gt_tangent_y,
            )
            axis.quiver(0.0, 0.0, local_tangent[0] * 12.0, local_tangent[1] * 12.0, angles="xy", scale_units="xy", scale=1, color="tab:red", label="PCA tangent")
            axis.set(title=f"RIGHT {source}", xlabel="GT-local lateral")
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.grid(alpha=0.25)
            axis.legend()
        axes[0].set_ylabel("GT-local axial")
        figure.suptitle("RIGHT matched cloud: production origin vs true crossing")
        figure.tight_layout()
        figure.savefig(directory / "right_production_vs_crossing_cloud.png", dpi=170)
        plt.close(figure)


def run_synthetic_test() -> None:
    """Check correlations and that exact plane points yield zero PCA error."""
    assert abs(_pearson([-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]) - 1.0) < 1.0e-12
    assert abs(_spearman([3.0, 1.0, 2.0], [30.0, 10.0, 20.0]) - 1.0) < 1.0e-12
    slope, intercept, r_squared = _linear_fit([-1.0, 0.0, 1.0], [1.0, 2.0, 3.0])
    assert (slope, intercept, r_squared) == (1.0, 2.0, 1.0)
    points = [{"sample_world_x": 0.0, "sample_world_y": value} for value in range(-10, 11)]
    result = _pca(points, (1.0, 0.0), radial_trim=True)
    assert abs(float(result["yaw_deg"])) < 1.0e-12


if __name__ == "__main__":
    run_synthetic_test()
    print("mouth_crossing_origin_bias_diagnostics synthetic test: PASS")
