"""Evaluation-only diagnostics for Branch motion-frame angular bias.

This sink accepts estimator intermediates and already-computed projection
results.  It writes CSV/PNG artifacts only and returns no control decision.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


def circular_distance_deg(first: float, second: float) -> float:
    """Return the unsigned shortest angular separation in degrees."""
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write dictionaries while preserving first-seen field order."""
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


@dataclass
class MotionFrameBiasDiagnostics:
    """Accumulate motion-frame source, projection, and lag observations."""

    source_rows: list[dict[str, Any]] = field(default_factory=list)
    projection_rows: list[dict[str, Any]] = field(default_factory=list)
    frontier_rows: list[dict[str, Any]] = field(default_factory=list)
    sweep_rows: list[dict[str, Any]] = field(default_factory=list)
    _source_branches: set[str] = field(default_factory=set)
    _projected_branches: set[str] = field(default_factory=set)
    _sweep_recorded: bool = False

    def record_source(self, row: Mapping[str, Any]) -> None:
        """Store the one-time frame-lock inputs for one Branch."""
        branch = str(row["branch"])
        if branch in self._source_branches:
            return
        self._source_branches.add(branch)
        self.source_rows.append(dict(row))

    def source_for_branch(self, branch: str) -> dict[str, Any] | None:
        """Return a read-only diagnostic lookup used only for more logging."""
        return next((row for row in self.source_rows if row["branch"] == branch), None)

    def needs_projection(self, branch: str) -> bool:
        """Return whether the Branch still needs its one projection snapshot."""
        return branch not in self._projected_branches

    def record_projections(
        self, branch: str, rows: Sequence[Mapping[str, Any]],
    ) -> None:
        """Store evaluation-only slot projections at representative depths."""
        if branch in self._projected_branches:
            return
        self._projected_branches.add(branch)
        self.projection_rows.extend(dict(row) for row in rows)

    def record_frontier_frame(
        self, rows: Sequence[Mapping[str, Any]],
    ) -> None:
        """Store lightweight existing-tolerance status for frontier members."""
        self.frontier_rows.extend(dict(row) for row in rows)

    def record_sweep(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Store one synthetic yaw-error by depth sensitivity sweep."""
        if self._sweep_recorded:
            return
        self._sweep_recorded = True
        self.sweep_rows.extend(dict(row) for row in rows)

    def save(self, output_dir: str | Path) -> None:
        """Validate and save the four required CSVs and PNGs."""
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        _write_rows(
            directory / "motion_frame_source_diagnostics.csv",
            self.source_rows,
        )
        _write_rows(
            directory / "motion_frame_slot_projection.csv",
            self.projection_rows,
        )
        robot_583 = [
            row for row in self.frontier_rows if int(row["robot_id"]) == 583
        ]
        _write_rows(directory / "robot_583_divergence.csv", robot_583)
        _write_rows(
            directory / "synthetic_angular_bias_sweep.csv",
            self.sweep_rows,
        )
        self._validate(robot_583)
        self._save_plots(directory, robot_583)

    def _validate(self, robot_583: Sequence[Mapping[str, Any]]) -> None:
        """Check source uniqueness and slot/sweep count consistency."""
        if len({row["branch"] for row in self.source_rows}) != len(self.source_rows):
            raise AssertionError("duplicate motion-frame source row")
        for row in self.sweep_rows:
            if int(row["safe_slot_count"]) + int(row["unsafe_slot_count"]) != int(row["slot_count"]):
                raise AssertionError("synthetic sweep slot count mismatch")
        for row in robot_583:
            if row["target_attained"] not in {True, False}:
                raise AssertionError("target_attained must be an existing boolean")

    def _save_plots(
        self,
        directory: Path,
        robot_583: Sequence[Mapping[str, Any]],
    ) -> None:
        """Render diagnostic figures without feeding results back to control."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return

        if self.source_rows:
            branches = [str(row["branch"]) for row in self.source_rows]
            labels = ("mouth_pca_yaw_deg", "recent_segment_yaw_deg", "final_yaw_deg", "gt_yaw_deg")
            figure, axis = plt.subplots(figsize=(9, 4.5))
            x = np.arange(len(branches), dtype=float)
            width = 0.19
            for index, key in enumerate(labels):
                values = []
                for row in self.source_rows:
                    if row[key] == "":
                        values.append(float("nan"))
                        continue
                    value = float(row[key])
                    ground_truth = float(row["gt_yaw_deg"])
                    # Unwrap each source around its Branch GT solely for a
                    # readable plot (for example, -172.8 deg is 187.2 deg
                    # beside LEFT's 180 deg). CSV values remain unchanged.
                    values.append(
                        ground_truth
                        + (value - ground_truth + 180.0) % 360.0
                        - 180.0
                    )
                axis.bar(x + (index - 1.5) * width, values, width, label=key.replace("_yaw_deg", ""))
            axis.set(
                xticks=x,
                xticklabels=branches,
                ylabel="GT-centered yaw [deg]",
                title="Branch motion-frame yaw sources (circularly unwrapped)",
            )
            axis.legend(fontsize=8)
            axis.grid(axis="y", alpha=0.25)
            figure.tight_layout()
            figure.savefig(directory / "branch_motion_frame_yaw_comparison.png", dpi=160)
            plt.close(figure)

        right = [
            row for row in self.projection_rows
            if row["branch"] == "RIGHT" and row["depth_label"] == "contacted"
        ]
        if right:
            figure, axis = plt.subplots(figsize=(8, 5))
            for source in ("mouth_pca", "recent_segment", "final"):
                rows = [row for row in right if row["yaw_source"] == source]
                axis.scatter(
                    [row["target_x"] for row in rows],
                    [row["target_y"] for row in rows],
                    marker="o" if source == "final" else "x",
                    label=f"{source} ({sum(bool(row['walkable']) for row in rows)}/{len(rows)} safe)",
                )
            first = right[0]
            axis.axhline(float(first["corridor_min_y"]), color="black", linewidth=1)
            axis.axhline(float(first["corridor_max_y"]), color="black", linewidth=1)
            axis.set(xlabel="world x", ylabel="world y", title="RIGHT contacted-depth slot projections")
            axis.legend()
            axis.grid(alpha=0.25)
            figure.tight_layout()
            figure.savefig(directory / "right_yaw_source_slot_projection.png", dpi=160)
            plt.close(figure)

        if self.sweep_rows:
            yaws = sorted({float(row["yaw_error_deg"]) for row in self.sweep_rows})
            depths = sorted({float(row["axial_depth"]) for row in self.sweep_rows})
            safe = {
                (float(row["yaw_error_deg"]), float(row["axial_depth"])): int(row["safe_slot_count"])
                for row in self.sweep_rows
            }
            values = np.asarray([[safe[(yaw, depth)] for depth in depths] for yaw in yaws])
            figure, axis = plt.subplots(figsize=(8, 4.5))
            image = axis.imshow(values, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=max(values.max(), 1))
            axis.set(xticks=np.arange(len(depths)), xticklabels=[f"{value:g}" for value in depths], yticks=np.arange(len(yaws)), yticklabels=[f"{value:g}" for value in yaws], xlabel="axial depth", ylabel="yaw error [deg]", title="Safe full-width slots")
            figure.colorbar(image, ax=axis, label="safe slot count")
            figure.tight_layout()
            figure.savefig(directory / "angular_bias_vs_depth_heatmap.png", dpi=160)
            plt.close(figure)

        if robot_583:
            times = [float(row["timestamp"]) for row in robot_583]
            figure, axis = plt.subplots(figsize=(9, 4.5))
            axis.plot(times, [float(row["actual_axial"]) for row in robot_583], label="actual axial")
            axis.plot(times, [float(row["target_axial"]) for row in robot_583], label="target axial")
            axis.set(xlabel="simulation time [s]", ylabel="local axial", title="Robot 583 frontier divergence")
            flags = axis.twinx()
            flags.step(times, [int(bool(row["target_walkable"])) for row in robot_583], where="post", label="target walkable", color="tab:green", alpha=0.6)
            flags.step(times, [int(bool(row["frontier_row_ready"])) for row in robot_583], where="post", label="row ready", color="tab:red", alpha=0.6)
            flags.set(ylim=(-0.05, 1.05), ylabel="boolean")
            handles, labels = axis.get_legend_handles_labels()
            extra_handles, extra_labels = flags.get_legend_handles_labels()
            axis.legend(handles + extra_handles, labels + extra_labels, loc="upper left")
            figure.tight_layout()
            figure.savefig(directory / "robot_583_divergence_timeline.png", dpi=160)
            plt.close(figure)


def run_synthetic_test() -> None:
    """Check circular distance and aggregate invariants."""
    assert circular_distance_deg(179.0, -179.0) == 2.0
    diagnostics = MotionFrameBiasDiagnostics()
    diagnostics.record_sweep(({
        "yaw_error_deg": 0.0,
        "axial_depth": 50.0,
        "safe_slot_count": 20,
        "unsafe_slot_count": 0,
        "slot_count": 20,
    },))
    diagnostics._validate(())


if __name__ == "__main__":
    run_synthetic_test()
    print("handoff_motion_frame_bias_diagnostics synthetic test: PASS")
