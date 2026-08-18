"""Map-free trajectory sampling and evaluation-only A/B diagnostics.

The sampling algorithm in this module consumes only a robot's causal motion
history.  Geometric mouth planes are accepted by the diagnostics methods only
to score an already-selected sample; they never flow back into ``estimate``.
"""

from __future__ import annotations

import csv
import math
from collections import deque
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
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


@dataclass(frozen=True)
class LocalCrossingEstimate:
    """One causal pose selected from a robot's local trajectory history."""

    x: float
    y: float
    timestamp: float
    method: str
    interpolation_alpha: float
    history_segments: int


@dataclass
class LocalTrajectoryCrossingSampler:
    """Estimate a transition pose without map, Branch direction, or PCA.

    Starting at the current outbound segment, the sampler walks backward over
    the contiguous run whose speed and direction satisfy the *existing*
    production cohort thresholds.  At the first angular threshold transition,
    it interpolates the causal pose between the previous and first aligned
    observations.  The reference is the robot's current observed displacement,
    not a labelled Branch direction.
    """

    history_size: int
    minimum_speed: float
    half_angle_rad: float
    histories: dict[int, deque[dict[str, float]]] = field(default_factory=dict)

    def record_motion(
        self,
        robot_id: int,
        previous_xy: tuple[float, float],
        current_xy: tuple[float, float],
        timestamp: float,
        dt: float,
    ) -> None:
        """Append one realized self-motion segment to bounded causal history."""
        dx = current_xy[0] - previous_xy[0]
        dy = current_xy[1] - previous_xy[1]
        distance = math.hypot(dx, dy)
        speed = distance / max(dt, 1.0e-12)
        direction_x = dx / distance if distance > 1.0e-12 else 0.0
        direction_y = dy / distance if distance > 1.0e-12 else 0.0
        history = self.histories.setdefault(
            int(robot_id), deque(maxlen=self.history_size)
        )
        history.append({
            "start_x": float(previous_xy[0]),
            "start_y": float(previous_xy[1]),
            "end_x": float(current_xy[0]),
            "end_y": float(current_xy[1]),
            "start_time": float(timestamp - dt),
            "end_time": float(timestamp),
            "speed": speed,
            "direction_x": direction_x,
            "direction_y": direction_y,
        })

    def estimate(self, robot_id: int) -> LocalCrossingEstimate | None:
        """Return the first pose in the current persistent self-motion run."""
        history = list(self.histories.get(int(robot_id), ()))
        if not history:
            return None
        reference = next(
            (
                (row["direction_x"], row["direction_y"])
                for row in reversed(history)
                if row["speed"] >= self.minimum_speed
            ),
            None,
        )
        if reference is None:
            return None
        threshold = math.cos(self.half_angle_rad)

        def score(row: Mapping[str, float]) -> float:
            return (
                float(row["direction_x"]) * reference[0]
                + float(row["direction_y"]) * reference[1]
            )

        first_index = len(history) - 1
        accepted = 0
        for index in range(len(history) - 1, -1, -1):
            row = history[index]
            if row["speed"] < self.minimum_speed or score(row) < threshold:
                break
            first_index = index
            accepted += 1
        first = history[first_index]
        prior = history[first_index - 1] if first_index > 0 else None
        if prior is not None and prior["speed"] >= self.minimum_speed:
            prior_score = score(prior)
            first_score = score(first)
            denominator = first_score - prior_score
            if prior_score < threshold <= first_score and denominator > 1.0e-12:
                alpha = min(1.0, max(0.0, (threshold - prior_score) / denominator))
                return LocalCrossingEstimate(
                    x=first["start_x"] + alpha * (first["end_x"] - first["start_x"]),
                    y=first["start_y"] + alpha * (first["end_y"] - first["start_y"]),
                    timestamp=first["start_time"] + alpha * (first["end_time"] - first["start_time"]),
                    method="interpolated_local_heading_transition",
                    interpolation_alpha=alpha,
                    history_segments=accepted,
                )
        return LocalCrossingEstimate(
            x=first["start_x"],
            y=first["start_y"],
            timestamp=first["start_time"],
            method="first_persistent_self_motion_pose",
            interpolation_alpha=0.0,
            history_segments=accepted,
        )


@dataclass
class LocalMouthCrossingDiagnostics:
    """Collect sampling, PCA, handoff, and lifecycle measurements for one run."""

    mode: str
    sampler: LocalTrajectoryCrossingSampler
    gt_crossings: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    candidates: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    sample_rows: list[dict[str, Any]] = field(default_factory=list)
    branch_rows: list[dict[str, Any]] = field(default_factory=list)
    pca_rows: list[dict[str, Any]] = field(default_factory=list)
    handoff_rows: list[dict[str, Any]] = field(default_factory=list)
    run_rows: list[dict[str, Any]] = field(default_factory=list)
    snapshots: set[str] = field(default_factory=set)

    def record_gt_crossing(self, row: Mapping[str, Any]) -> None:
        """Store evaluation-only first geometric crossing for later scoring."""
        key = (str(row["branch"]), int(row["robot_id"]))
        self.gt_crossings.setdefault(key, dict(row))

    def record_candidate(
        self,
        *,
        branch: str,
        robot_id: int,
        heading_xy: tuple[float, float],
        heading_time: float,
        local: LocalCrossingEstimate,
    ) -> None:
        """Store both A and B poses before the selected one enters production PCA."""
        self.candidates.setdefault((branch, int(robot_id)), {
            "branch": branch,
            "robot_id": int(robot_id),
            "heading_x": float(heading_xy[0]),
            "heading_y": float(heading_xy[1]),
            "heading_time": float(heading_time),
            "local_x": local.x,
            "local_y": local.y,
            "local_time": local.timestamp,
            "local_method": local.method,
            "local_interpolation_alpha": local.interpolation_alpha,
            "local_history_segments": local.history_segments,
        })

    def record_snapshot(
        self,
        *,
        branch: str,
        frame: int,
        timestamp: float,
        flow_xy: tuple[float, float],
        production_sample_ids: Sequence[int],
        gt_origin_xy: tuple[float, float],
        gt_tangent_xy: tuple[float, float],
        gt_yaw_deg: float,
    ) -> None:
        """Run identical robust PCA on matched heading, local, and GT samples."""
        if branch in self.snapshots:
            return
        descriptor_ids = set(map(int, production_sample_ids))
        candidate_ids = {
            robot_id
            for candidate, robot_id in self.candidates
            if candidate == branch and robot_id in descriptor_ids
        }
        gt_ids = {robot_id for candidate, robot_id in self.gt_crossings if candidate == branch}
        matched = sorted(candidate_ids & gt_ids)
        origin_x, origin_y = gt_origin_xy
        tangent_x, tangent_y = gt_tangent_xy
        normal_x, normal_y = -tangent_y, tangent_x
        rows = []
        for robot_id in matched:
            candidate = self.candidates[(branch, robot_id)]
            gt = self.gt_crossings[(branch, robot_id)]
            crossing_x = float(gt["crossing_world_x"])
            crossing_y = float(gt["crossing_world_y"])

            def local_coordinates(x: float, y: float) -> tuple[float, float]:
                dx, dy = x - origin_x, y - origin_y
                return dx * tangent_x + dy * tangent_y, dx * normal_x + dy * normal_y

            crossing_axial, crossing_lateral = local_coordinates(crossing_x, crossing_y)
            heading_axial, heading_lateral = local_coordinates(candidate["heading_x"], candidate["heading_y"])
            local_axial, local_lateral = local_coordinates(candidate["local_x"], candidate["local_y"])
            rows.append({
                **candidate,
                "gt_x": crossing_x,
                "gt_y": crossing_y,
                "gt_time": gt["mouth_crossing_time"],
                "gt_axial": crossing_axial,
                "gt_lateral": crossing_lateral,
                "heading_axial": heading_axial,
                "heading_lateral": heading_lateral,
                "local_axial": local_axial,
                "local_lateral": local_lateral,
                "heading_axial_error": heading_axial - crossing_axial,
                "local_axial_error": local_axial - crossing_axial,
                "heading_lateral_error": heading_lateral - crossing_lateral,
                "local_lateral_error": local_lateral - crossing_lateral,
                "heading_delay": candidate["heading_time"] - float(gt["mouth_crossing_time"]),
                "local_delay": candidate["local_time"] - float(gt["mouth_crossing_time"]),
                "mode": self.mode,
                "evaluation_only_gt": True,
            })
        self.sample_rows.extend(rows)
        heading_errors = [float(row["heading_axial_error"]) for row in rows]
        local_errors = [float(row["local_axial_error"]) for row in rows]
        self.branch_rows.append({
            "branch": branch,
            "mode": self.mode,
            "candidate_count": len(candidate_ids),
            "gt_count": len(gt_ids),
            "matched_count": len(matched),
            "heading_mean_axial_error": _mean(heading_errors),
            "heading_median_axial_error": _quantile(heading_errors, 0.50),
            "heading_std_axial_error": _std(heading_errors),
            "heading_max_abs_axial_error": max(map(abs, heading_errors), default=0.0),
            "local_mean_axial_error": _mean(local_errors),
            "local_median_axial_error": _quantile(local_errors, 0.50),
            "local_std_axial_error": _std(local_errors),
            "local_max_abs_axial_error": max(map(abs, local_errors), default=0.0),
            "heading_mean_lateral_error": _mean([float(row["heading_lateral_error"]) for row in rows]),
            "local_mean_lateral_error": _mean([float(row["local_lateral_error"]) for row in rows]),
            "heading_mean_delay": _mean([float(row["heading_delay"]) for row in rows]),
            "local_mean_delay": _mean([float(row["local_delay"]) for row in rows]),
        })

        def pca(source: str) -> dict[str, Any]:
            if source == "gt":
                values = [(float(row["gt_x"]), float(row["gt_y"])) for row in rows]
            else:
                values = [(float(row[f"{source}_x"]), float(row[f"{source}_y"])) for row in rows]
            pca_rows = [{"sample_world_x": x, "sample_world_y": y} for x, y in values]
            return _pca(pca_rows, flow_xy, radial_trim=True)

        heading_pca, local_pca, gt_pca = pca("heading"), pca("local"), pca("gt")
        heading_error = abs(_signed_angle_delta(heading_pca["yaw_deg"], gt_yaw_deg))
        local_error = abs(_signed_angle_delta(local_pca["yaw_deg"], gt_yaw_deg))
        self.pca_rows.append({
            "branch": branch,
            "mode": self.mode,
            "matched_count": len(matched),
            "heading_inlier_count": len(heading_pca["inlier_indices"]),
            "local_inlier_count": len(local_pca["inlier_indices"]),
            "gt_inlier_count": len(gt_pca["inlier_indices"]),
            "heading_inlier_yaw_deg": heading_pca["yaw_deg"],
            "local_inlier_yaw_deg": local_pca["yaw_deg"],
            "gt_inlier_yaw_deg": gt_pca["yaw_deg"],
            "gt_yaw_deg": gt_yaw_deg,
            "heading_error_deg": heading_error,
            "local_error_deg": local_error,
            "yaw_reduction_deg": heading_error - local_error,
            "yaw_reduction_percent": 100.0 * (heading_error - local_error) / heading_error if heading_error > 1.0e-12 else 0.0,
            "heading_covariance_xy": heading_pca["covariance_xy"],
            "local_covariance_xy": local_pca["covariance_xy"],
            "gt_covariance_xy": gt_pca["covariance_xy"],
            "heading_anisotropy": heading_pca["anisotropy"],
            "local_anisotropy": local_pca["anisotropy"],
            "gt_anisotropy": gt_pca["anisotropy"],
            "frame": frame,
            "timestamp": timestamp,
        })
        self.snapshots.add(branch)

    def record_handoff(
        self,
        *,
        branch: str,
        timestamp: float,
        contacted_depth: float,
        resolved_depth: float | None,
        attempted_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        """Store the first actual resolver outcome for a Branch in this mode."""
        if any(row["branch"] == branch for row in self.handoff_rows):
            return
        safe_counts = [sum(bool(slot["walkable"]) for slot in attempt["slots"]) for attempt in attempted_rows]
        slot_count = max((len(attempt["slots"]) for attempt in attempted_rows), default=0)
        self.handoff_rows.append({
            "branch": branch,
            "mode": self.mode,
            "success": resolved_depth is not None,
            "failure_reason": "" if resolved_depth is not None else "NO_COMMON_LOCAL_HANDOFF_ROW",
            "contacted_depth": contacted_depth,
            "resolved_depth": "" if resolved_depth is None else resolved_depth,
            "tested_row_count": len(attempted_rows),
            "maximum_safe_slot_count": max(safe_counts, default=0),
            "required_slot_count": slot_count,
            "full_row_exists": any(count == slot_count and slot_count > 0 for count in safe_counts),
            "handoff_time": timestamp,
        })

    def record_run_summary(
        self,
        *,
        final_phase: str,
        simulation_time: float,
        visited_branches: Sequence[str],
        shepherd_formed_branches: Sequence[str],
        pressure_branches: Sequence[str],
        backflow_branches: Sequence[str],
    ) -> None:
        """Store final lifecycle facts after the simulator exits."""
        self.run_rows[:] = [{
            "mode": self.mode,
            "final_phase": final_phase,
            "done": final_phase == "DONE",
            "simulation_time": simulation_time,
            "visited_branches": ",".join(sorted(visited_branches)),
            "shepherd_formed_branches": ",".join(sorted(shepherd_formed_branches)),
            "pressure_branches": ",".join(sorted(pressure_branches)),
            "backflow_branches": ",".join(sorted(backflow_branches)),
        }]

    def save(self, output_dir: str | Path) -> None:
        """Write raw per-run tables for the separate A/B aggregation step."""
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        _write_rows(directory / "local_mouth_crossing_samples.csv", self.sample_rows)
        _write_rows(directory / "local_mouth_crossing_branch_summary.csv", self.branch_rows)
        _write_rows(directory / "local_mouth_crossing_pca_comparison.csv", self.pca_rows)
        _write_rows(directory / "local_mouth_crossing_handoff_comparison.csv", self.handoff_rows)
        _write_rows(directory / "local_mouth_crossing_run_summary.csv", self.run_rows)


def _synthetic_test() -> None:
    sampler = LocalTrajectoryCrossingSampler(6, 1.0, math.radians(30.0))
    sampler.record_motion(1, (0.0, 0.0), (0.0, 1.0), 1.0, 1.0)
    sampler.record_motion(1, (0.0, 1.0), (1.0, 2.0), 2.0, 1.0)
    sampler.record_motion(1, (1.0, 2.0), (2.0, 2.0), 3.0, 1.0)
    estimate = sampler.estimate(1)
    if estimate is None or not 1.0 <= estimate.x <= 2.0:
        raise AssertionError("local transition interpolation invariant failed")
    print("local_mouth_crossing_sampling synthetic test: PASS")


if __name__ == "__main__":
    _synthetic_test()
