"""Evaluation-only A/B/C replay for one failed local handoff snapshot.

This module is deliberately downstream of the production handoff resolver.  It
never returns a depth or any other control value to the simulator.  The same
retained robot IDs, lateral slots, contacted depth, retreat schedule, and
walkability callback are replayed in production, GT-orientation, and GT-full
frames to isolate frame error from slot geometry.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


Point = tuple[float, float]
WalkabilityProbe = Callable[[Point], bool]


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write dictionaries while preserving stable first-seen column order."""
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


def _target(origin: Point, tangent: Point, normal: Point, depth: float,
            lateral: float) -> Point:
    """Convert one local axial/lateral slot to world coordinates."""
    return (
        origin[0] + tangent[0] * depth + normal[0] * lateral,
        origin[1] + tangent[1] * depth + normal[1] * lateral,
    )


def _yaw_deg(vector: Point) -> float:
    return math.degrees(math.atan2(vector[1], vector[0]))


def _as_bool(value: Any) -> bool:
    """Accept both in-memory booleans and their CSV text representation."""
    return value if isinstance(value, bool) else str(value).lower() == "true"


def _fixture_margin(point: Point, bounds: Mapping[str, float]) -> tuple[float, str]:
    """Return signed robot-centre clearance to the RIGHT corridor boundaries.

    The fixture limits are used only to explain the already-computed collision
    result.  They never participate in resolver success/failure.
    """
    x, y = point
    margins = {
        "TOP": y - float(bounds["center_top"]),
        "BOTTOM": float(bounds["center_bottom"]) - y,
        "TERMINAL": float(bounds["center_terminal"]) - x,
    }
    boundary = min(margins, key=margins.get)
    value = margins[boundary]
    return value, ("NONE" if value >= 0.0 else boundary)


@dataclass
class HandoffGtFrameIsolationDiagnostics:
    """Capture and shadow-replay only the first failed RIGHT handoff event."""

    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    occupancy_rows: list[dict[str, Any]] = field(default_factory=list)
    target_rows: list[dict[str, Any]] = field(default_factory=list)
    snapshot_rows: list[dict[str, Any]] = field(default_factory=list)
    captured: bool = False

    def capture_and_replay(
        self,
        *,
        frame: int,
        timestamp: float,
        branch: str,
        phase: str,
        frontier_rows: Sequence[Mapping[str, Any]],
        contacted_depth: float,
        production_origin: Point,
        production_tangent: Point,
        production_normal: Point,
        gt_origin: Point,
        gt_tangent: Point,
        gt_normal: Point,
        retreat_step: float,
        maximum_retreat: float,
        robot_radius: float,
        observed_width: float,
        observed_physical_width: float,
        physical_left_boundary_lateral: float | None,
        physical_right_boundary_lateral: float | None,
        physical_width_source: str,
        environment_state: Mapping[str, Any],
        fixture_center_bounds: Mapping[str, float],
        walkable: WalkabilityProbe,
        production_attempted_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        """Freeze one snapshot and replay the unmodified depth-search schedule."""
        if self.captured or branch != "RIGHT":
            return
        self.captured = True
        if not frontier_rows:
            raise AssertionError("RIGHT failure snapshot has no retained frontiers")

        modes = (
            ("production", production_origin, production_tangent, production_normal),
            ("gt_orientation", production_origin, gt_tangent, gt_normal),
            ("gt_full_frame", gt_origin, gt_tangent, gt_normal),
        )
        common = {
            "frame": frame,
            "timestamp": timestamp,
            "branch": branch,
            "phase": phase,
            "retained_frontier_count": len(frontier_rows),
            "retained_frontier_ids": json.dumps(
                [int(row["robot_id"]) for row in frontier_rows]
            ),
            "contacted_depth": contacted_depth,
            "production_origin_x": production_origin[0],
            "production_origin_y": production_origin[1],
            "production_t_x": production_tangent[0],
            "production_t_y": production_tangent[1],
            "production_n_x": production_normal[0],
            "production_n_y": production_normal[1],
            "production_yaw_deg": _yaw_deg(production_tangent),
            "gt_origin_x": gt_origin[0],
            "gt_origin_y": gt_origin[1],
            "gt_t_x": gt_tangent[0],
            "gt_t_y": gt_tangent[1],
            "gt_n_x": gt_normal[0],
            "gt_n_y": gt_normal[1],
            "gt_yaw_deg": _yaw_deg(gt_tangent),
            "observed_width": observed_width,
            "observed_physical_width": observed_physical_width,
            "physical_left_boundary_lateral": physical_left_boundary_lateral,
            "physical_right_boundary_lateral": physical_right_boundary_lateral,
            "physical_width_source": physical_width_source,
            "retreat_step": retreat_step,
            "maximum_retreat": maximum_retreat,
            "robot_radius": robot_radius,
            "walkability_environment_state": json.dumps(
                dict(environment_state), sort_keys=True
            ),
        }
        self.snapshot_rows.append({"record_type": "EVENT", **common})
        for row in frontier_rows:
            self.snapshot_rows.append({
                "record_type": "RETAINED_FRONTIER",
                **common,
                "robot_id": int(row["robot_id"]),
                "world_x": float(row["world_x"]),
                "world_y": float(row["world_y"]),
                "actual_local_depth": float(row["actual_local_depth"]),
                "actual_local_lateral": float(row["actual_local_lateral"]),
                "assigned_lateral_slot": float(row["lateral_slot"]),
                "is_robot_583": int(row["robot_id"]) == 583,
            })

        for mode, origin, tangent, normal in modes:
            self._replay_mode(
                mode=mode,
                common=common,
                frontier_rows=frontier_rows,
                origin=origin,
                tangent=tangent,
                normal=normal,
                contacted_depth=contacted_depth,
                retreat_step=retreat_step,
                maximum_retreat=maximum_retreat,
                fixture_center_bounds=fixture_center_bounds,
                walkable=walkable,
            )
        if production_attempted_rows is not None:
            shadow = [row for row in self.occupancy_rows if row["mode"] == "production"]
            if len(shadow) != len(production_attempted_rows):
                raise AssertionError("production shadow tested-row count diverged")
            for replayed, actual in zip(shadow, production_attempted_rows):
                actual_unsafe = sorted(
                    int(slot["robot_id"])
                    for slot in actual["slots"] if not slot["walkable"]
                )
                if (
                    abs(float(replayed["depth"]) - float(actual["depth"])) > 1e-9
                    or sorted(json.loads(replayed["unsafe_ids"])) != actual_unsafe
                ):
                    raise AssertionError("production shadow slot result diverged")
            self.summary_rows[0]["production_shadow_matches_resolver"] = True
        self._validate()

    def _replay_mode(
        self,
        *,
        mode: str,
        common: Mapping[str, Any],
        frontier_rows: Sequence[Mapping[str, Any]],
        origin: Point,
        tangent: Point,
        normal: Point,
        contacted_depth: float,
        retreat_step: float,
        maximum_retreat: float,
        fixture_center_bounds: Mapping[str, float],
        walkable: WalkabilityProbe,
    ) -> None:
        """Replay the production resolver algorithm without returning control."""
        mode_occupancy: list[dict[str, Any]] = []

        def row_is_walkable(depth: float, search_stage: str) -> bool:
            attempt_index = len(mode_occupancy)
            unsafe_ids: list[int] = []
            safe_ids: list[int] = []
            for frontier in frontier_rows:
                robot_id = int(frontier["robot_id"])
                lateral = float(frontier["lateral_slot"])
                point = _target(origin, tangent, normal, depth, lateral)
                is_safe = bool(walkable(point))
                margin, violation = _fixture_margin(point, fixture_center_bounds)
                if is_safe:
                    violation = "NONE"
                (safe_ids if is_safe else unsafe_ids).append(robot_id)
                self.target_rows.append({
                    "mode": mode,
                    "frame": common["frame"],
                    "timestamp": common["timestamp"],
                    "branch": common["branch"],
                    "attempt_index": attempt_index,
                    "search_stage": search_stage,
                    "depth": depth,
                    "robot_id": robot_id,
                    "is_robot_583": robot_id == 583,
                    "lateral_slot": lateral,
                    "target_x": point[0],
                    "target_y": point[1],
                    "walkable": is_safe,
                    "boundary_margin": margin,
                    "wall_violation_direction": violation,
                })
            mode_occupancy.append({
                "mode": mode,
                "frame": common["frame"],
                "timestamp": common["timestamp"],
                "branch": common["branch"],
                "attempt_index": attempt_index,
                "search_stage": search_stage,
                "depth": depth,
                "retreat_from_contacted_depth": contacted_depth - depth,
                "safe": len(safe_ids),
                "unsafe": len(unsafe_ids),
                "unsafe_ids": json.dumps(unsafe_ids),
                "safe_ids": json.dumps(safe_ids),
                "all_slots_walkable": not unsafe_ids,
            })
            return not unsafe_ids

        resolved_depth: float | None = None
        if row_is_walkable(contacted_depth, "CONTACTED"):
            resolved_depth = contacted_depth
        else:
            unsafe_depth = contacted_depth
            safe_depth: float | None = None
            retreat = retreat_step
            while retreat <= maximum_retreat + 1e-9:
                candidate = max(0.0, contacted_depth - retreat)
                if row_is_walkable(candidate, "RETREAT"):
                    safe_depth = candidate
                    break
                unsafe_depth = candidate
                retreat += retreat_step
            if safe_depth is not None:
                low = safe_depth
                high = unsafe_depth if unsafe_depth >= low else contacted_depth
                for _ in range(14):
                    middle = 0.5 * (low + high)
                    if row_is_walkable(middle, "BISECTION"):
                        low = middle
                    else:
                        high = middle
                resolved_depth = low

        self.occupancy_rows.extend(mode_occupancy)
        # Persistent means unsafe at every tested row.  A full-safe row must
        # therefore make the intersection empty rather than being skipped.
        unsafe_sets = [
            set(json.loads(row["unsafe_ids"])) for row in mode_occupancy
        ]
        persistent = sorted(set.intersection(*unsafe_sets)) if unsafe_sets else []
        full_rows = [row for row in mode_occupancy if row["all_slots_walkable"]]
        self.summary_rows.append({
            "mode": mode,
            "branch": common["branch"],
            "frame": common["frame"],
            "timestamp": common["timestamp"],
            "production_yaw": common["production_yaw_deg"],
            "evaluation_yaw": _yaw_deg(tangent),
            "evaluation_origin_x": origin[0],
            "evaluation_origin_y": origin[1],
            "contacted_depth": contacted_depth,
            "tested_rows": len(mode_occupancy),
            "max_safe_slots": max(int(row["safe"]) for row in mode_occupancy),
            "frontier_count": len(frontier_rows),
            "full_safe_row": bool(full_rows),
            "first_full_safe_depth": (
                "" if not full_rows else full_rows[0]["depth"]
            ),
            "resolved_depth": "" if resolved_depth is None else resolved_depth,
            "retreat_required": (
                "" if resolved_depth is None else contacted_depth - resolved_depth
            ),
            "persistent_unsafe_robot": json.dumps(persistent),
            "robot_583_persistent": 583 in persistent,
        })

    def _validate(self) -> None:
        """Assert snapshot identity and row/target count invariants."""
        if len(self.summary_rows) != 3:
            raise AssertionError("A/B/C replay did not produce exactly three modes")
        frontier_count = int(self.summary_rows[0]["frontier_count"])
        for row in self.occupancy_rows:
            if int(row["safe"]) + int(row["unsafe"]) != frontier_count:
                raise AssertionError("shadow row occupancy does not match snapshot")
        for summary in self.summary_rows:
            rows = [row for row in self.occupancy_rows if row["mode"] == summary["mode"]]
            targets = [row for row in self.target_rows if row["mode"] == summary["mode"]]
            if len(targets) != len(rows) * frontier_count:
                raise AssertionError("shadow slot targets do not match tested rows")

    def save(self, output_dir: str | Path) -> None:
        """Write the four requested CSV files and four causal plots."""
        if not self.captured:
            return
        self._validate()
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        _write_rows(directory / "gt_frame_handoff_summary.csv", self.summary_rows)
        _write_rows(directory / "gt_frame_row_occupancy.csv", self.occupancy_rows)
        _write_rows(directory / "gt_frame_slot_targets.csv", self.target_rows)
        _write_rows(directory / "gt_frame_snapshot.csv", self.snapshot_rows)
        self._save_plots(directory)

    def _save_plots(self, directory: Path) -> None:
        """Render comparisons exclusively from the saved shadow observations."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {
            "production": "tab:red",
            "gt_orientation": "tab:orange",
            "gt_full_frame": "tab:green",
        }
        figure, axis = plt.subplots(figsize=(9, 5))
        for mode in colors:
            rows = sorted(
                (row for row in self.occupancy_rows if row["mode"] == mode),
                key=lambda row: float(row["retreat_from_contacted_depth"]),
            )
            axis.plot(
                [float(row["retreat_from_contacted_depth"]) for row in rows],
                [int(row["safe"]) for row in rows], marker=".", label=mode,
                color=colors[mode],
            )
        axis.axhline(int(self.summary_rows[0]["frontier_count"]), linestyle="--", color="black", alpha=0.5)
        axis.set(xlabel="retreat from contacted depth", ylabel="safe retained slots", title="RIGHT handoff row occupancy: identical snapshot")
        axis.legend(); axis.grid(alpha=0.3); figure.tight_layout()
        figure.savefig(directory / "right_row_occupancy_by_frame_mode.png", dpi=170)
        plt.close(figure)

        self._plot_target_modes(
            directory / "right_slot_targets_production_vs_gt_orientation.png",
            ("production", "gt_orientation"),
            "Production vs GT orientation (same contacted depth)",
            selection="contacted",
        )
        self._plot_target_modes(
            directory / "right_slot_targets_gt_full_frame.png",
            ("gt_full_frame",),
            "GT full frame slot targets (first full-safe row)",
            selection="first_full_safe",
        )

        persistent_ids: set[int] = set()
        for summary in self.summary_rows:
            persistent_ids.update(json.loads(summary["persistent_unsafe_robot"]))
        figure, axis = plt.subplots(figsize=(9, 5))
        persistent_targets: list[dict[str, Any]] = []
        for mode in colors:
            mode_rows = [row for row in self.target_rows if row["mode"] == mode]
            if not mode_rows:
                continue
            selected = [
                row for row in mode_rows
                if int(row["attempt_index"]) == 0
                and int(row["robot_id"]) in persistent_ids
            ]
            persistent_targets.extend(selected)
            axis.scatter([float(row["target_x"]) for row in selected], [float(row["target_y"]) for row in selected], label=mode, color=colors[mode])
            for row in selected:
                axis.annotate(str(row["robot_id"]), (float(row["target_x"]), float(row["target_y"])))
        self._draw_bounds(axis)
        if persistent_targets:
            xs = [float(row["target_x"]) for row in persistent_targets]
            ys = [float(row["target_y"]) for row in persistent_targets]
            axis.set_xlim(min(xs) - 2.0, max(xs) + 2.0)
            axis.set_ylim(min(ys) - 3.0, max(ys) + 3.0)
        axis.set_aspect("equal", adjustable="box"); axis.set(title="Persistent unsafe slot comparison", xlabel="world x", ylabel="world y")
        axis.legend(); axis.grid(alpha=0.3); figure.tight_layout()
        figure.savefig(directory / "persistent_unsafe_slot_comparison.png", dpi=170)
        plt.close(figure)

    def _draw_bounds(self, axis: Any) -> None:
        event = self.snapshot_rows[0]
        state = json.loads(event["walkability_environment_state"])
        # is_walkable rounds the centre to a pixel and tests an integer-radius
        # footprint.  The half-pixel term is the continuous centre limit that
        # corresponds to that exact collision implementation.
        pixel_radius = float(state.get("collision_pixel_radius", round(float(event["robot_radius"]))))
        top = float(state["corridor_top"]) + pixel_radius - 0.5
        bottom = float(state["corridor_bottom"]) - pixel_radius + 0.5
        terminal = float(state["right_terminal_x"]) - pixel_radius + 0.5
        axis.axhline(top, color="black", linestyle=":", alpha=0.6)
        axis.axhline(bottom, color="black", linestyle=":", alpha=0.6)
        axis.axvline(terminal, color="black", linestyle=":", alpha=0.6)

    def _plot_target_modes(
        self,
        path: Path,
        modes: Sequence[str],
        title: str,
        *,
        selection: str,
    ) -> None:
        """Plot one explicitly comparable target row for selected modes."""
        import matplotlib.pyplot as plt
        colors = {"production": "tab:red", "gt_orientation": "tab:orange", "gt_full_frame": "tab:green"}
        figure, axis = plt.subplots(figsize=(9, 5))
        for mode in modes:
            mode_rows = [row for row in self.target_rows if row["mode"] == mode]
            if selection == "contacted":
                selected_attempt = 0
            elif selection == "first_full_safe":
                full_row = next(
                    row for row in self.occupancy_rows
                    if row["mode"] == mode and _as_bool(row["all_slots_walkable"])
                )
                selected_attempt = int(full_row["attempt_index"])
            else:
                raise ValueError(f"unknown target-row selection: {selection}")
            selected = [row for row in mode_rows if int(row["attempt_index"]) == selected_attempt]
            axis.scatter([float(row["target_x"]) for row in selected], [float(row["target_y"]) for row in selected], label=mode, color=colors[mode], marker="x")
            unsafe = [row for row in selected if not _as_bool(row["walkable"])]
            axis.scatter([float(row["target_x"]) for row in unsafe], [float(row["target_y"]) for row in unsafe], facecolors="none", edgecolors="black", s=90)
        self._draw_bounds(axis)
        axis.set_aspect("equal", adjustable="box"); axis.set(title=title, xlabel="world x", ylabel="world y")
        axis.legend(); axis.grid(alpha=0.3); figure.tight_layout(); figure.savefig(path, dpi=170); plt.close(figure)


def run_synthetic_test() -> None:
    """Verify A/B/C replay and circularly identical retained-slot accounting."""
    diagnostic = HandoffGtFrameIsolationDiagnostics()
    frontiers = [
        {"robot_id": 582, "world_x": 8.0, "world_y": -0.5, "actual_local_depth": 8.0, "actual_local_lateral": -0.5, "lateral_slot": -0.5},
        {"robot_id": 583, "world_x": 8.0, "world_y": 0.5, "actual_local_depth": 8.0, "actual_local_lateral": 0.5, "lateral_slot": 0.5},
    ]
    diagnostic.capture_and_replay(
        frame=10, timestamp=1.0, branch="RIGHT", phase="TEST",
        frontier_rows=frontiers, contacted_depth=10.0,
        production_origin=(0.0, 0.0), production_tangent=(1.0, 0.0), production_normal=(0.0, 1.0),
        gt_origin=(0.0, 0.0), gt_tangent=(1.0, 0.0), gt_normal=(0.0, 1.0),
        retreat_step=0.25, maximum_retreat=1.0, robot_radius=0.1,
        observed_width=2.0, observed_physical_width=2.0,
        physical_left_boundary_lateral=-1.0, physical_right_boundary_lateral=1.0,
        physical_width_source="SYNTHETIC",
        environment_state={"corridor_top": -1.0, "corridor_bottom": 1.0, "right_terminal_x": 10.0},
        fixture_center_bounds={"center_top": -0.9, "center_bottom": 0.9, "center_terminal": 9.9},
        walkable=lambda point: -0.9 <= point[1] <= 0.9 and point[0] <= 9.9,
    )
    if any(row["max_safe_slots"] != 2 for row in diagnostic.summary_rows):
        raise AssertionError("synthetic replay failed to find the common row")
    if any(row["tested_rows"] != 16 for row in diagnostic.summary_rows):
        raise AssertionError("synthetic replay did not preserve resolver schedule")


if __name__ == "__main__":
    run_synthetic_test()
    print("handoff_gt_frame_isolation_diagnostics synthetic test: PASS")
