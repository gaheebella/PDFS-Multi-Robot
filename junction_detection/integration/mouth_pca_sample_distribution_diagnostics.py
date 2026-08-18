"""Evaluation-only diagnostics for mouth-crossing PCA input samples.

The diagnostics reproduce the existing radial trim and PCA calculations for
analysis, but expose no result to the simulator control path.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


BRANCH_ORDER = ("LEFT", "UP", "RIGHT")


def _quantile(values: Sequence[float], probability: float) -> float:
    """Match the production linear-interpolation quantile helper."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(max(float(probability), 0.0), 1.0) * (len(ordered) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    blend = index - lower
    return ordered[lower] * (1.0 - blend) + ordered[upper] * blend


def _yaw_deg(x: float, y: float) -> float:
    """Return a normalized vector yaw in degrees."""
    return (math.degrees(math.atan2(y, x)) + 180.0) % 360.0 - 180.0


def _signed_angle_delta(first: float, second: float) -> float:
    """Return first minus second on the shortest circular arc."""
    return (float(first) - float(second) + 180.0) % 360.0 - 180.0


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write heterogeneous dictionaries with stable first-seen columns."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pca(
    samples: Sequence[Mapping[str, Any]],
    flow_xy: tuple[float, float],
    *,
    radial_trim: bool,
) -> dict[str, Any]:
    """Reproduce production PCA, optionally including its 90% radial trim."""
    points = [(float(row["sample_world_x"]), float(row["sample_world_y"])) for row in samples]
    if not points:
        return {"accepted": False, "inlier_indices": [], "yaw_deg": float("nan")}
    trim_center_x = _quantile([point[0] for point in points], 0.50)
    trim_center_y = _quantile([point[1] for point in points], 0.50)
    distances = [math.hypot(x - trim_center_x, y - trim_center_y) for x, y in points]
    cutoff = _quantile(distances, 0.90) if radial_trim else float("inf")
    inlier_indices = [
        index for index, distance in enumerate(distances)
        if distance <= cutoff + 1.0e-9
    ]
    inlier_points = [points[index] for index in inlier_indices]
    if len(inlier_points) < 2:
        return {
            "accepted": False,
            "inlier_indices": inlier_indices,
            "trim_center_x": trim_center_x,
            "trim_center_y": trim_center_y,
            "trim_cutoff": cutoff,
            "yaw_deg": float("nan"),
        }
    center_x = _quantile([point[0] for point in inlier_points], 0.50)
    center_y = _quantile([point[1] for point in inlier_points], 0.50)
    offsets = [(x - center_x, y - center_y) for x, y in inlier_points]
    covariance_xx = sum(x * x for x, _ in offsets) / len(offsets)
    covariance_xy = sum(x * y for x, y in offsets) / len(offsets)
    covariance_yy = sum(y * y for _, y in offsets) / len(offsets)
    trace = covariance_xx + covariance_yy
    discriminant = math.sqrt(max(
        0.0,
        (covariance_xx - covariance_yy) ** 2 + 4.0 * covariance_xy ** 2,
    ))
    major = 0.5 * (trace + discriminant)
    minor = max(0.0, 0.5 * (trace - discriminant))
    anisotropy = major / max(minor, 1.0e-9)
    if abs(covariance_xy) > 1.0e-9:
        axis_x, axis_y = major - covariance_yy, covariance_xy
    elif covariance_xx >= covariance_yy:
        axis_x, axis_y = 1.0, 0.0
    else:
        axis_x, axis_y = 0.0, 1.0
    axis_length = math.hypot(axis_x, axis_y)
    if axis_length <= 1.0e-9:
        return {"accepted": False, "inlier_indices": inlier_indices, "yaw_deg": float("nan")}
    axis_x, axis_y = axis_x / axis_length, axis_y / axis_length
    flow_x, flow_y = flow_xy
    flow_length = max(math.hypot(flow_x, flow_y), 1.0e-9)
    flow_x, flow_y = flow_x / flow_length, flow_y / flow_length
    reference_n = (-flow_y, flow_x)
    axis_sign_flip = axis_x * reference_n[0] + axis_y * reference_n[1] < 0.0
    if axis_sign_flip:
        axis_x, axis_y = -axis_x, -axis_y
    tangent_x, tangent_y = axis_y, -axis_x
    tangent_sign_flip = tangent_x * flow_x + tangent_y * flow_y < 0.0
    if tangent_sign_flip:
        tangent_x, tangent_y = -tangent_x, -tangent_y
    arithmetic_mean_x = sum(point[0] for point in inlier_points) / len(inlier_points)
    arithmetic_mean_y = sum(point[1] for point in inlier_points) / len(inlier_points)
    return {
        "accepted": major > 1.0e-9 and anisotropy >= 1.35,
        "inlier_indices": inlier_indices,
        "trim_center_x": trim_center_x,
        "trim_center_y": trim_center_y,
        "trim_cutoff": cutoff,
        "center_x": center_x,
        "center_y": center_y,
        "mean_x": arithmetic_mean_x,
        "mean_y": arithmetic_mean_y,
        "covariance_xx": covariance_xx,
        "covariance_xy": covariance_xy,
        "covariance_yy": covariance_yy,
        "major_eigenvalue": major,
        "minor_eigenvalue": minor,
        "anisotropy": anisotropy,
        "axis_x": axis_x,
        "axis_y": axis_y,
        "tangent_x": tangent_x,
        "tangent_y": tangent_y,
        "yaw_deg": _yaw_deg(tangent_x, tangent_y),
        "axis_sign_flip": axis_sign_flip,
        "tangent_sign_flip": tangent_sign_flip,
    }


@dataclass
class MouthPcaSampleDistributionDiagnostics:
    """Capture one-shot mouth origins and analyze the samples used at lock."""

    crossing_metadata: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    sample_rows: list[dict[str, Any]] = field(default_factory=list)
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    temporal_rows: list[dict[str, Any]] = field(default_factory=list)
    robot_rows: list[dict[str, Any]] = field(default_factory=list)
    leave_one_out_rows: list[dict[str, Any]] = field(default_factory=list)
    covariance_rows: list[dict[str, Any]] = field(default_factory=list)
    prefix_rows: list[dict[str, Any]] = field(default_factory=list)
    snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    _crossing_counts: dict[str, int] = field(default_factory=dict)

    def record_crossing_origin(self, row: Mapping[str, Any]) -> None:
        """Record the first heading-aligned in-Branch origin for one robot."""
        key = (str(row["branch"]), int(row["robot_id"]))
        if key in self.crossing_metadata:
            return
        branch = key[0]
        order = self._crossing_counts.get(branch, 0)
        self._crossing_counts[branch] = order + 1
        stored = dict(row)
        stored["crossing_order"] = order
        self.crossing_metadata[key] = stored

    def record_handoff_outcome(self, branch: str, success: bool) -> None:
        """Attach an observed baseline outcome without influencing it."""
        for row in self.summary_rows:
            if row["branch"] == branch:
                row["handoff_result"] = "SUCCESS" if success else "FAILED"

    def record_snapshot(
        self,
        *,
        branch: str,
        branch_id: str,
        frame: int,
        timestamp: float,
        samples: Mapping[int, tuple[float, float]],
        flow_xy: tuple[float, float],
        final_origin_xy: tuple[float, float],
        motion_t_xy: tuple[float, float],
        motion_n_xy: tuple[float, float],
        gt_yaw_deg: float,
        branch_length: float,
        minimum_samples: int,
    ) -> None:
        """Analyze the exact immutable sample dictionary used by production."""
        if branch in self.snapshots:
            return
        ordered: list[dict[str, Any]] = []
        fallback_order = 10**9
        for dictionary_order, (robot_id, point) in enumerate(samples.items()):
            metadata = dict(self.crossing_metadata.get((branch, int(robot_id)), {}))
            metadata.setdefault("branch", branch)
            metadata.setdefault("robot_id", int(robot_id))
            metadata.setdefault("frame", frame)
            metadata.setdefault("crossing_timestamp", timestamp)
            metadata.setdefault("timestamp", metadata["crossing_timestamp"])
            metadata.setdefault("crossing_order", fallback_order + dictionary_order)
            metadata["sample_world_x"] = float(point[0])
            metadata["sample_world_y"] = float(point[1])
            ordered.append(metadata)
        ordered.sort(key=lambda row: (int(row["crossing_order"]), int(row["robot_id"])))
        for index, row in enumerate(ordered):
            row["crossing_order"] = index
        raw_pca = _pca(ordered, flow_xy, radial_trim=False)
        trimmed_pca = _pca(ordered, flow_xy, radial_trim=True)
        inlier_orders = {ordered[index]["crossing_order"] for index in trimmed_pca["inlier_indices"]}
        origin_x, origin_y = final_origin_xy
        motion_t_x, motion_t_y = motion_t_xy
        motion_n_x, motion_n_y = motion_n_xy
        gt_radians = math.radians(gt_yaw_deg)
        gt_t = (math.cos(gt_radians), math.sin(gt_radians))
        gt_n = (-gt_t[1], gt_t[0])
        for row in ordered:
            dx = float(row["sample_world_x"]) - origin_x
            dy = float(row["sample_world_y"]) - origin_y
            local_axial = dx * motion_t_x + dy * motion_t_y
            local_lateral = dx * motion_n_x + dy * motion_n_y
            row.update({
                "branch_id": branch_id,
                "pca_frame": frame,
                "pca_timestamp": timestamp,
                "sample_local_axial": local_axial,
                "sample_local_lateral": local_lateral,
                "crossing_side": "NEGATIVE" if local_lateral < 0.0 else ("POSITIVE" if local_lateral > 0.0 else "ZERO"),
                "raw_or_inlier": "INLIER" if row["crossing_order"] in inlier_orders else "TRIMMED_OUT",
                "trimmed_out": row["crossing_order"] not in inlier_orders,
                "pca_center_x": trimmed_pca["center_x"],
                "pca_center_y": trimmed_pca["center_y"],
                "gt_local_axial": dx * gt_t[0] + dy * gt_t[1],
                "gt_local_lateral": dx * gt_n[0] + dy * gt_n[1],
            })
            self.sample_rows.append(dict(row))
        lateral = [float(row["sample_local_lateral"]) for row in ordered]
        negative = [value for value in lateral if value < 0.0]
        positive = [value for value in lateral if value > 0.0]
        mean_lateral = sum(lateral) / len(lateral)
        std_lateral = math.sqrt(sum((value - mean_lateral) ** 2 for value in lateral) / len(lateral))
        summary = {
            "branch": branch,
            "branch_id": branch_id,
            "branch_length": branch_length,
            "pca_frame": frame,
            "pca_timestamp": timestamp,
            "raw_sample_count": len(ordered),
            "inlier_count": len(trimmed_pca["inlier_indices"]),
            "trimmed_count": len(ordered) - len(trimmed_pca["inlier_indices"]),
            "negative_side_count": len(negative),
            "positive_side_count": len(positive),
            "zero_side_count": len(lateral) - len(negative) - len(positive),
            "negative_side_mean_lateral": sum(negative) / len(negative) if negative else "",
            "positive_side_mean_lateral": sum(positive) / len(positive) if positive else "",
            "median_lateral": _quantile(lateral, 0.50),
            "mean_lateral": mean_lateral,
            "lateral_standard_deviation": std_lateral,
            "minimum_lateral": min(lateral),
            "maximum_lateral": max(lateral),
            "centroid_lateral_offset": mean_lateral,
            "raw_covariance_xy": raw_pca["covariance_xy"],
            "inlier_covariance_xy": trimmed_pca["covariance_xy"],
            "raw_anisotropy": raw_pca["anisotropy"],
            "inlier_anisotropy": trimmed_pca["anisotropy"],
            "raw_pca_yaw_deg": raw_pca["yaw_deg"],
            "inlier_pca_yaw_deg": trimmed_pca["yaw_deg"],
            "raw_pca_error_to_gt_deg": abs(_signed_angle_delta(raw_pca["yaw_deg"], gt_yaw_deg)),
            "inlier_pca_error_to_gt_deg": abs(_signed_angle_delta(trimmed_pca["yaw_deg"], gt_yaw_deg)),
            "trim_yaw_change_deg": _signed_angle_delta(trimmed_pca["yaw_deg"], raw_pca["yaw_deg"]),
            "arithmetic_mean_x": trimmed_pca["mean_x"],
            "arithmetic_mean_y": trimmed_pca["mean_y"],
            "production_robust_center_x": trimmed_pca["center_x"],
            "production_robust_center_y": trimmed_pca["center_y"],
            "handoff_result": "NOT_OBSERVED",
        }
        segments = ("EARLY", "MIDDLE", "LATE")
        boundaries = [round(index * len(ordered) / 3) for index in range(4)]
        for segment_index, label in enumerate(segments):
            group = ordered[boundaries[segment_index]:boundaries[segment_index + 1]]
            result = _pca(group, flow_xy, radial_trim=True)
            temporal_row = {
                "branch": branch,
                "segment": label,
                "first_crossing_order": group[0]["crossing_order"],
                "last_crossing_order": group[-1]["crossing_order"],
                "sample_count": len(group),
                "inlier_count": len(result["inlier_indices"]),
                "centroid_x": sum(float(row["sample_world_x"]) for row in group) / len(group),
                "centroid_y": sum(float(row["sample_world_y"]) for row in group) / len(group),
                "pca_axis_x": result["axis_x"],
                "pca_axis_y": result["axis_y"],
                "estimated_tangent_yaw_deg": result["yaw_deg"],
                "yaw_error_to_gt_deg": _signed_angle_delta(result["yaw_deg"], gt_yaw_deg),
                "evaluation_only_gt_yaw_deg": gt_yaw_deg,
            }
            self.temporal_rows.append(temporal_row)
            summary[f"{label.lower()}_pca_yaw_deg"] = result["yaw_deg"]
            summary[f"{label.lower()}_pca_error_to_gt_deg"] = temporal_row["yaw_error_to_gt_deg"]
        for row in ordered:
            self.robot_rows.append({
                "branch": branch,
                "robot_id": row["robot_id"],
                "number_of_samples": 1,
                "first_crossing_time": row["crossing_timestamp"],
                "last_crossing_time": row["crossing_timestamp"],
                "mean_local_lateral": row["sample_local_lateral"],
                "mean_local_axial": row["sample_local_axial"],
                "crossing_side": row["crossing_side"],
                "raw_or_inlier": row["raw_or_inlier"],
            })
        for count in range(minimum_samples, len(ordered) + 1):
            result = _pca(ordered[:count], flow_xy, radial_trim=True)
            self.prefix_rows.append({
                "branch": branch,
                "accumulated_sample_count": count,
                "last_crossing_order": ordered[count - 1]["crossing_order"],
                "last_crossing_timestamp": ordered[count - 1]["crossing_timestamp"],
                "yaw_deg": result["yaw_deg"],
                "yaw_error_to_gt_deg": _signed_angle_delta(result["yaw_deg"], gt_yaw_deg),
            })
        removal_influences = []
        for removed in ordered:
            reduced = [
                row for row in ordered
                if int(row["robot_id"]) != int(removed["robot_id"])
            ]
            result = _pca(reduced, flow_xy, radial_trim=True)
            removal_influences.append((
                abs(_signed_angle_delta(result["yaw_deg"], trimmed_pca["yaw_deg"])),
                int(removed["robot_id"]),
                int(removed["crossing_order"]),
            ))
        largest_influence, largest_robot, largest_order = max(removal_influences)
        summary.update({
            "largest_single_sample_yaw_influence_deg": largest_influence,
            "largest_single_sample_order": largest_order,
            "largest_single_sample_robot_id": largest_robot,
            # Production stores exactly one sample per robot, so these two
            # counterfactuals are identical in the current implementation.
            "largest_single_robot_yaw_influence_deg": largest_influence,
            "largest_single_robot_id": largest_robot,
        })
        self.summary_rows.append(summary)
        if branch == "RIGHT":
            self._record_right_influence(ordered, flow_xy, trimmed_pca)
        self.snapshots[branch] = {
            "raw": [dict(row) for row in ordered],
            "raw_pca": raw_pca,
            "pca": trimmed_pca,
            "gt_yaw_deg": gt_yaw_deg,
        }

    def _record_right_influence(
        self,
        ordered: Sequence[Mapping[str, Any]],
        flow_xy: tuple[float, float],
        baseline: Mapping[str, Any],
    ) -> None:
        """Compute end-to-end sample/robot removal and covariance influence."""
        baseline_yaw = float(baseline["yaw_deg"])
        for removal_type in ("SAMPLE", "ROBOT"):
            for removed in ordered:
                reduced = [row for row in ordered if int(row["robot_id"]) != int(removed["robot_id"])]
                result = _pca(reduced, flow_xy, radial_trim=True)
                self.leave_one_out_rows.append({
                    "removal_type": removal_type,
                    "removed_sample_order": removed["crossing_order"],
                    "removed_robot_id": removed["robot_id"],
                    "removed_was_inlier": removed["raw_or_inlier"] == "INLIER",
                    "baseline_yaw_deg": baseline_yaw,
                    "recomputed_yaw_deg": result["yaw_deg"],
                    "yaw_change_deg": _signed_angle_delta(result["yaw_deg"], baseline_yaw),
                    "absolute_yaw_change_deg": abs(_signed_angle_delta(result["yaw_deg"], baseline_yaw)),
                    "recomputed_inlier_count": len(result["inlier_indices"]),
                })
        inlier_indices = set(int(index) for index in baseline["inlier_indices"])
        center_x = float(baseline["center_x"])
        center_y = float(baseline["center_y"])
        inlier_count = len(inlier_indices)
        contribution_rows = []
        for index, row in enumerate(ordered):
            contribution = (
                (float(row["sample_world_x"]) - center_x)
                * (float(row["sample_world_y"]) - center_y)
                if index in inlier_indices else 0.0
            )
            contribution_rows.append({
                "branch": "RIGHT",
                "crossing_order": row["crossing_order"],
                "robot_id": row["robot_id"],
                "crossing_timestamp": row["crossing_timestamp"],
                "sample_local_axial": row["sample_local_axial"],
                "sample_local_lateral": row["sample_local_lateral"],
                "raw_or_inlier": row["raw_or_inlier"],
                "covariance_product": contribution,
                "covariance_xy_component": contribution / inlier_count if index in inlier_indices else 0.0,
                "absolute_covariance_product": abs(contribution),
            })
        ranked = sorted(contribution_rows, key=lambda row: float(row["absolute_covariance_product"]), reverse=True)
        rank_by_order = {int(row["crossing_order"]): rank + 1 for rank, row in enumerate(ranked)}
        for row in contribution_rows:
            row["absolute_contribution_rank"] = rank_by_order[int(row["crossing_order"])]
            self.covariance_rows.append(row)

    def save(self, output_dir: str | Path) -> None:
        """Validate and write the six required CSVs and seven PNGs."""
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        _write_rows(directory / "mouth_pca_samples.csv", self.sample_rows)
        _write_rows(directory / "mouth_pca_branch_summary.csv", self.summary_rows)
        _write_rows(directory / "mouth_pca_temporal_segments.csv", self.temporal_rows)
        _write_rows(directory / "mouth_pca_robot_contributions.csv", self.robot_rows)
        _write_rows(directory / "right_pca_leave_one_out.csv", self.leave_one_out_rows)
        _write_rows(directory / "right_covariance_contributions.csv", self.covariance_rows)
        self._validate()
        self._save_plots(directory)
        top = sorted(self.covariance_rows, key=lambda row: float(row["absolute_covariance_product"]), reverse=True)[:10]
        print("[MouthPCADiagnostics] Top RIGHT covariance contributors:")
        for row in top:
            print(
                "  robot={robot_id} order={crossing_order} time={crossing_timestamp} "
                "axial={sample_local_axial:.3f} lateral={sample_local_lateral:.3f} "
                "product={covariance_product:.6f}".format(**row)
            )

    def _validate(self) -> None:
        """Assert exact snapshot, trimming, and one-sample-per-robot invariants."""
        if set(self.snapshots) != set(BRANCH_ORDER):
            raise AssertionError(f"missing Branch snapshots: {set(BRANCH_ORDER) - set(self.snapshots)}")
        for branch in BRANCH_ORDER:
            rows = [row for row in self.sample_rows if row["branch"] == branch]
            if len({int(row["robot_id"]) for row in rows}) != len(rows):
                raise AssertionError(f"duplicate robot mouth sample in {branch}")
            summary = next(row for row in self.summary_rows if row["branch"] == branch)
            if int(summary["raw_sample_count"]) != len(rows):
                raise AssertionError(f"raw sample count mismatch in {branch}")
            if int(summary["inlier_count"]) != sum(not bool(row["trimmed_out"]) for row in rows):
                raise AssertionError(f"inlier count mismatch in {branch}")
        right = next(row for row in self.summary_rows if row["branch"] == "RIGHT")
        if abs(float(right["inlier_covariance_xy"]) + 4.285752199565801) > 1.0e-9:
            raise AssertionError("RIGHT production covariance was not reproduced")

    def _save_plots(self, directory: Path) -> None:
        """Render sample geometry, temporal yaw, and covariance influence."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for branch in BRANCH_ORDER:
            snapshot = self.snapshots[branch]
            rows = snapshot["raw"]
            pca = snapshot["pca"]
            figure, axis = plt.subplots(figsize=(7, 6))
            trimmed = [row for row in rows if bool(row["trimmed_out"])]
            inliers = [row for row in rows if not bool(row["trimmed_out"])]
            axis.scatter([row["sample_world_x"] for row in rows], [row["sample_world_y"] for row in rows], s=24, facecolors="none", edgecolors="0.65", label="raw")
            axis.scatter([row["sample_world_x"] for row in inliers], [row["sample_world_y"] for row in inliers], s=18, color="tab:blue", label="PCA inlier")
            if trimmed:
                axis.scatter([row["sample_world_x"] for row in trimmed], [row["sample_world_y"] for row in trimmed], s=35, marker="x", color="tab:red", label="trimmed")
            center_x, center_y = float(pca["center_x"]), float(pca["center_y"])
            scale = 25.0
            axis.scatter([center_x], [center_y], marker="*", s=120, color="black", label="PCA center")
            axis.quiver(center_x, center_y, float(pca["axis_x"]) * scale, float(pca["axis_y"]) * scale, angles="xy", scale_units="xy", scale=1, color="tab:purple", label="PCA major axis")
            axis.quiver(center_x, center_y, float(pca["tangent_x"]) * scale, float(pca["tangent_y"]) * scale, angles="xy", scale_units="xy", scale=1, color="tab:green", label="derived tangent")
            gt = math.radians(float(snapshot["gt_yaw_deg"]))
            axis.quiver(center_x, center_y, math.cos(gt) * scale, math.sin(gt) * scale, angles="xy", scale_units="xy", scale=1, color="black", linestyle="--", label="GT eval only")
            axis.set(
                title=(
                    f"{branch} mouth PCA: tangent={float(pca['yaw_deg']):.4f} deg "
                    "(independent axis scales)"
                ),
                xlabel="world x",
                ylabel="world y",
            )
            axis.legend(fontsize=8)
            axis.grid(alpha=0.25)
            figure.tight_layout()
            figure.savefig(directory / f"mouth_pca_samples_{branch.lower()}.png", dpi=170)
            plt.close(figure)

        figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
        for axis, branch in zip(axes, BRANCH_ORDER):
            rows = [row for row in self.sample_rows if row["branch"] == branch]
            axis.scatter([row["gt_local_axial"] for row in rows], [row["gt_local_lateral"] for row in rows], c=["tab:red" if row["trimmed_out"] else "tab:blue" for row in rows], s=18)
            axis.axvline(0.0, color="black", linewidth=0.8)
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set(title=branch, xlabel="GT-local axial (eval only)", aspect="equal")
            axis.grid(alpha=0.2)
        axes[0].set_ylabel("GT-local lateral (eval only)")
        figure.suptitle("Mouth PCA sample clouds at common scale")
        figure.tight_layout()
        figure.savefig(directory / "mouth_pca_branch_comparison.png", dpi=170)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(9, 4.8))
        for branch in BRANCH_ORDER:
            rows = [row for row in self.prefix_rows if row["branch"] == branch]
            axis.plot([row["accumulated_sample_count"] for row in rows], [row["yaw_error_to_gt_deg"] for row in rows], label=branch)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(title="PCA tangent yaw error over crossing order", xlabel="accumulated sample count", ylabel="signed yaw error [deg]")
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(directory / "mouth_pca_yaw_over_crossing_order.png", dpi=170)
        plt.close(figure)

        rows = sorted(self.covariance_rows, key=lambda row: int(row["crossing_order"]))
        figure, axis = plt.subplots(figsize=(10, 4.8))
        colors = ["tab:blue" if row["raw_or_inlier"] == "INLIER" else "tab:red" for row in rows]
        axis.bar([int(row["crossing_order"]) for row in rows], [float(row["covariance_xy_component"]) for row in rows], color=colors, width=0.8)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(title="RIGHT covariance XY contributions", xlabel="crossing order", ylabel="contribution / inlier count")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(directory / "right_covariance_contributors.png", dpi=170)
        plt.close(figure)


def run_synthetic_test() -> None:
    """Verify quantile, angle wrapping, and a simple vertical PCA fixture."""
    assert _quantile([0.0, 10.0], 0.25) == 2.5
    assert _signed_angle_delta(-179.0, 179.0) == 2.0
    samples = [
        {"sample_world_x": 0.0, "sample_world_y": float(index)}
        for index in range(-10, 11)
    ]
    result = _pca(samples, (1.0, 0.0), radial_trim=True)
    assert len(result["inlier_indices"]) == 19
    assert abs(result["yaw_deg"]) < 0.1


if __name__ == "__main__":
    run_synthetic_test()
    print("mouth_pca_sample_distribution_diagnostics synthetic test: PASS")
