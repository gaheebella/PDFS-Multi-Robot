"""Anchor-local Point Cloud confirmation for provisional Junction candidates.

The module enforces two explicit information domains:

``simulate_polygon_lidar`` is simulator-only. It may receive world geometry and
an Anchor world pose in order to emulate a 2D LiDAR.

``confirm_junction_topology`` is algorithm-visible. It receives only circular
Anchor-local angle/range arrays, reuses the existing sensor-enhanced opening
detector without overriding its defaults, and applies only the graph-topology
definition ``degree >= 3``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


class AnchorJunctionState(str, Enum):
    """Lifecycle states for one provisional Anchor candidate."""

    NO_ANCHOR = "NO_ANCHOR"
    PROVISIONAL_ANCHOR = "PROVISIONAL_ANCHOR"
    CONFIRMED_ANCHOR = "CONFIRMED_ANCHOR"
    REJECTED = "REJECTED"
    RELEASED = "RELEASED"


@dataclass(frozen=True)
class AnchorLocalScan:
    """Detector-safe local scan plus local point-cloud presentation fields."""

    angles_deg: np.ndarray
    ranges: np.ndarray
    hit: np.ndarray
    local_x: np.ndarray
    local_y: np.ndarray
    max_range: float
    range_input_unit: str

    def detector_input(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of the only fields visible to topology inference."""
        return self.angles_deg.copy(), self.ranges.copy()


@dataclass(frozen=True)
class PointCloudTopologyResult:
    """Timestamp-free result produced strictly from angle/range input."""

    openings: tuple[dict[str, float], ...]
    opening_count: int
    is_junction: bool
    reason: str

    @property
    def opening_directions_deg(self) -> tuple[float, ...]:
        return tuple(float(item["center_angle"]) for item in self.openings)


@dataclass(frozen=True)
class PointCloudTopologyDecision(PointCloudTopologyResult):
    """Topology result augmented with lifecycle time outside the algorithm."""

    timestamp: float


@dataclass(frozen=True)
class LifecycleTransition:
    """One auditable state transition independent of simulator controls."""

    timestamp: float
    anchor_id: Optional[int]
    previous_state: AnchorJunctionState
    state: AnchorJunctionState
    reason: str
    local_reference_yaw_deg: Optional[float]
    pointcloud_opening_count: Optional[int]


@dataclass
class AnchorPointCloudConfirmation:
    """State machine shared by the simulator and false-candidate fixture."""

    state: AnchorJunctionState = AnchorJunctionState.NO_ANCHOR
    anchor_id: Optional[int] = None
    local_reference_yaw_deg: Optional[float] = None
    provisional_time: Optional[float] = None
    decision_time: Optional[float] = None
    release_time: Optional[float] = None
    decision: Optional[PointCloudTopologyDecision] = None
    release_reason: str = ""
    transitions: list[LifecycleTransition] = field(default_factory=list)

    def observe_no_anchor(self, *, timestamp: float) -> None:
        """Record the initial state once for complete lifecycle diagnostics."""
        if self.state != AnchorJunctionState.NO_ANCHOR:
            raise RuntimeError(f"cannot record NO_ANCHOR from {self.state}")
        if self.transitions:
            return
        self._transition(
            timestamp,
            AnchorJunctionState.NO_ANCHOR,
            "awaiting_lateral_expansion_candidate",
        )

    def _transition(
        self,
        timestamp: float,
        state: AnchorJunctionState,
        reason: str,
        opening_count: Optional[int] = None,
    ) -> None:
        previous = self.state
        self.state = state
        self.transitions.append(
            LifecycleTransition(
                timestamp=float(timestamp),
                anchor_id=self.anchor_id,
                previous_state=previous,
                state=state,
                reason=reason,
                local_reference_yaw_deg=self.local_reference_yaw_deg,
                pointcloud_opening_count=opening_count,
            )
        )

    def begin_provisional(
        self,
        *,
        timestamp: float,
        anchor_id: int,
        local_reference_yaw_deg: float,
    ) -> None:
        """Register a candidate triggered outside this class by SPH evidence."""
        if self.state != AnchorJunctionState.NO_ANCHOR:
            raise RuntimeError(f"cannot elect a provisional Anchor from {self.state}")
        self.anchor_id = int(anchor_id)
        self.local_reference_yaw_deg = float(local_reference_yaw_deg)
        self.provisional_time = float(timestamp)
        self._transition(
            timestamp,
            AnchorJunctionState.PROVISIONAL_ANCHOR,
            "lateral_expansion_junction_candidate",
        )

    def evaluate(
        self,
        *,
        timestamp: float,
        angles_deg: Sequence[float],
        ranges: Sequence[float],
    ) -> PointCloudTopologyDecision:
        """Confirm or reject using only Anchor-local angle/range measurements."""
        if self.state != AnchorJunctionState.PROVISIONAL_ANCHOR:
            raise RuntimeError(f"cannot evaluate Point Cloud from {self.state}")
        result = confirm_junction_topology(
            angles_deg=angles_deg,
            ranges=ranges,
        )
        decision = PointCloudTopologyDecision(
            openings=result.openings,
            opening_count=result.opening_count,
            is_junction=result.is_junction,
            reason=result.reason,
            timestamp=float(timestamp),
        )
        self.decision = decision
        self.decision_time = float(timestamp)
        self._transition(
            timestamp,
            (
                AnchorJunctionState.CONFIRMED_ANCHOR
                if decision.is_junction
                else AnchorJunctionState.REJECTED
            ),
            decision.reason,
            decision.opening_count,
        )
        return decision

    def release(self, *, timestamp: float, reason: str) -> None:
        """Close a rejected or confirmed candidate without adding a cooldown."""
        if self.state == AnchorJunctionState.RELEASED:
            return
        if self.state == AnchorJunctionState.NO_ANCHOR:
            raise RuntimeError("cannot release before provisional election")
        self.release_time = float(timestamp)
        self.release_reason = str(reason)
        opening_count = self.decision.opening_count if self.decision else None
        self._transition(
            timestamp,
            AnchorJunctionState.RELEASED,
            reason,
            opening_count,
        )


def polygon_wall_segments(
    polygon_points: Sequence[Sequence[float]],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    """Convert a simulator polygon boundary into closed ray-casting segments."""
    points = tuple((float(point[0]), float(point[1])) for point in polygon_points)
    if len(points) < 3:
        raise ValueError("a wall polygon requires at least three points")
    return tuple(
        (points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )


def simulate_polygon_lidar(
    *,
    polygon_points: Sequence[Sequence[float]],
    anchor_xy: Sequence[float],
    anchor_reference_yaw_deg: float,
    max_range_world_units: float,
    range_input_unit: str = "pygame_world_unit",
) -> AnchorLocalScan:
    """Simulator-only adapter from world geometry to local angle/range.

    The imported ray caster supports sensor degradation, but this integration
    intentionally uses its ideal defaults. No noise, dropout, or occlusion is
    introduced in the provisional-confirmation stage.
    """
    from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import (
        simulate_lidar_scan,
    )

    scan = simulate_lidar_scan(
        polygon_wall_segments(polygon_points),
        anchor_xy,
        anchor_yaw_deg=float(anchor_reference_yaw_deg),
        max_range_m=float(max_range_world_units),
    )
    return AnchorLocalScan(
        angles_deg=scan.angle_deg.copy(),
        ranges=scan.range_m.copy(),
        hit=scan.hit.copy(),
        local_x=scan.local_x.copy(),
        local_y=scan.local_y.copy(),
        max_range=float(scan.max_range_m),
        range_input_unit=range_input_unit,
    )


def confirm_junction_topology(
    *,
    angles_deg: Sequence[float],
    ranges: Sequence[float],
) -> PointCloudTopologyResult:
    """Classify topology from local measurements using degree > 2 only.

    The existing detector defaults are intentionally not repeated or overridden
    here. The number three is a graph-topology definition, not a known map way
    count: degree 1 is a dead end, degree 2 a corridor, and degree >=3 a Junction.
    """
    from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import (
        detect_openings,
    )

    openings = tuple(detect_openings(angles_deg, ranges))
    is_junction = len(openings) >= 3
    return PointCloudTopologyResult(
        openings=openings,
        opening_count=len(openings),
        is_junction=is_junction,
        reason=(
            "pointcloud_topological_degree_ge_3"
            if is_junction
            else "pointcloud_topological_degree_le_2"
        ),
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def save_confirmation_artifacts(
    *,
    output_dir: str | Path,
    confirmation: AnchorPointCloudConfirmation,
    scan: Optional[AnchorLocalScan],
    anchor_position_drift: float,
    swarm_directions_deg: Sequence[float] = (),
    swarm_candidate_rows: Sequence[dict[str, Any]] = (),
) -> None:
    """Save lifecycle, opening, comparison CSVs and required PNG diagnostics."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    transition_rows = [
        {
            "timestamp": item.timestamp,
            "anchor_id": "" if item.anchor_id is None else item.anchor_id,
            "state": item.state.value,
            "transition": f"{item.previous_state.value}->{item.state.value}",
            "transition_reason": item.reason,
            "local_reference_yaw": (
                "" if item.local_reference_yaw_deg is None
                else item.local_reference_yaw_deg
            ),
            "pointcloud_opening_count": (
                "" if item.pointcloud_opening_count is None
                else item.pointcloud_opening_count
            ),
        }
        for item in confirmation.transitions
    ]
    _write_csv(
        directory / "anchor_junction_lifecycle.csv",
        transition_rows,
        (
            "timestamp", "anchor_id", "state", "transition",
            "transition_reason", "local_reference_yaw",
            "pointcloud_opening_count",
        ),
    )

    decision = confirmation.decision
    opening_rows = []
    if decision is not None:
        for opening_id, opening in enumerate(decision.openings):
            opening_rows.append({
                "timestamp": decision.timestamp,
                "opening_id": opening_id,
                "center_deg": opening["center_angle"],
                "start_deg": opening["start_angle"],
                "end_deg": opening["end_angle"],
                "width_deg": opening["width_deg"],
                "confidence": opening.get("confidence", ""),
            })
    _write_csv(
        directory / "anchor_pointcloud_openings.csv",
        opening_rows,
        (
            "timestamp", "opening_id", "center_deg", "start_deg", "end_deg",
            "width_deg", "confidence",
        ),
    )

    final_count = "" if decision is None else decision.opening_count
    summary_rows = [{
        "provisional_time": (
            "" if confirmation.provisional_time is None
            else confirmation.provisional_time
        ),
        "decision": (
            "NOT_EVALUATED" if decision is None
            else "CONFIRMED" if decision.is_junction else "REJECTED"
        ),
        "decision_time": (
            "" if confirmation.decision_time is None else confirmation.decision_time
        ),
        "anchor_id": "" if confirmation.anchor_id is None else confirmation.anchor_id,
        "final_pointcloud_opening_count": final_count,
        "opening_directions_deg": (
            "" if decision is None
            else json.dumps(decision.opening_directions_deg)
        ),
        "confirmation_reason": "" if decision is None else decision.reason,
        "release_reason": confirmation.release_reason,
        "anchor_position_drift": anchor_position_drift,
        "range_input_unit": "" if scan is None else scan.range_input_unit,
    }]
    _write_csv(
        directory / "anchor_junction_confirmation_summary.csv",
        summary_rows,
        tuple(summary_rows[0]),
    )
    _write_csv(
        directory / "pointcloud_vs_swarm_candidates.csv",
        list(swarm_candidate_rows),
        (
            "timestamp", "source", "candidate_id", "direction_deg",
            "robot_count", "circular_spread_deg",
        ),
    )
    _save_confirmation_plots(
        directory=directory,
        confirmation=confirmation,
        scan=scan,
        swarm_directions_deg=swarm_directions_deg,
    )


def _save_confirmation_plots(
    *,
    directory: Path,
    confirmation: AnchorPointCloudConfirmation,
    scan: Optional[AnchorLocalScan],
    swarm_directions_deg: Sequence[float],
) -> None:
    """Generate required plots with Point Cloud and swarm sources separated."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        print(f"[PointCloudConfirmation] PNG skipped: {error}")
        return

    state_order = {
        AnchorJunctionState.NO_ANCHOR: 0,
        AnchorJunctionState.PROVISIONAL_ANCHOR: 1,
        AnchorJunctionState.CONFIRMED_ANCHOR: 2,
        AnchorJunctionState.REJECTED: 3,
        AnchorJunctionState.RELEASED: 4,
    }
    figure, axis = plt.subplots(figsize=(8, 3.5))
    times = [item.timestamp for item in confirmation.transitions]
    states = [state_order[item.state] for item in confirmation.transitions]
    if times:
        axis.step(times, states, where="post", marker="o")
    axis.set_yticks(list(state_order.values()), [item.value for item in state_order])
    axis.set(xlabel="simulation time [s]", ylabel="Anchor lifecycle")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(directory / "junction_confirmation_timeline.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 3.5))
    if confirmation.decision is not None:
        axis.scatter(
            [confirmation.decision.timestamp],
            [confirmation.decision.opening_count],
            s=45,
        )
    axis.axhline(3, linestyle="--", color="tab:red", label="topological degree 3")
    axis.set(xlabel="simulation time [s]", ylabel="detected opening count")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "pointcloud_opening_count_over_time.png", dpi=150)
    plt.close(figure)

    if scan is not None:
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.scatter(
            scan.local_x[scan.hit],
            scan.local_y[scan.hit],
            s=8,
            alpha=0.7,
            label="wall hit",
        )
        axis.scatter(
            scan.local_x[~scan.hit],
            scan.local_y[~scan.hit],
            s=7,
            alpha=0.25,
            label="max-range no return",
        )
        if confirmation.decision is not None:
            for index, opening in enumerate(confirmation.decision.openings):
                angle = math.radians(opening["center_angle"])
                axis.plot(
                    [0.0, scan.max_range * math.cos(angle)],
                    [0.0, scan.max_range * math.sin(angle)],
                    linewidth=2,
                    label="detected opening" if index == 0 else None,
                )
        axis.scatter([0.0], [0.0], marker="*", s=100, color="black", label="Anchor")
        axis.set_aspect("equal", adjustable="box")
        axis.set(xlabel=f"local x [{scan.range_input_unit}]", ylabel=f"local y [{scan.range_input_unit}]")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(directory / "final_anchor_local_pointcloud.png", dpi=150)
        plt.close(figure)

    figure = plt.figure(figsize=(7, 5))
    axis = figure.add_subplot(111, projection="polar")
    pointcloud_directions = (
        confirmation.decision.opening_directions_deg
        if confirmation.decision is not None else ()
    )
    for index, direction in enumerate(pointcloud_directions):
        angle = math.radians(direction)
        axis.plot(
            [angle, angle], [0.0, 1.0], linewidth=2.5, color="tab:orange",
            label="Point Cloud opening" if index == 0 else None,
        )
    for index, direction in enumerate(swarm_directions_deg):
        angle = math.radians(float(direction))
        axis.plot(
            [angle, angle], [0.0, 0.82], linestyle="--", linewidth=2,
            color="tab:blue", label="Relative swarm candidate" if index == 0 else None,
        )
    axis.set_title("Point Cloud openings vs Relative Swarm directions")
    if pointcloud_directions or swarm_directions_deg:
        axis.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
    figure.tight_layout()
    figure.savefig(directory / "pointcloud_vs_swarm_directions.png", dpi=150)
    plt.close(figure)


def run_false_corridor_fixture(output_dir: str | Path) -> PointCloudTopologyDecision:
    """Exercise PROVISIONAL→REJECTED→RELEASED on a straight corridor.

    Corridor geometry is test-only and is consumed only by the ray caster. The
    confirmation state machine receives no expected opening count or label.
    """
    corridor_width = 84.0
    half_width = corridor_width / 2.0
    half_length = 300.0
    polygon = (
        (-half_width, -half_length),
        (half_width, -half_length),
        (half_width, half_length),
        (-half_width, half_length),
    )
    confirmation = AnchorPointCloudConfirmation()
    confirmation.observe_no_anchor(timestamp=0.0)
    confirmation.begin_provisional(
        timestamp=1.0,
        anchor_id=7,
        local_reference_yaw_deg=0.0,
    )
    scan = simulate_polygon_lidar(
        polygon_points=polygon,
        anchor_xy=(0.0, 0.0),
        anchor_reference_yaw_deg=0.0,
        max_range_world_units=corridor_width,
    )
    angles, ranges = scan.detector_input()
    decision = confirmation.evaluate(
        timestamp=1.0,
        angles_deg=angles,
        ranges=ranges,
    )
    if decision.is_junction:
        raise AssertionError("straight corridor was incorrectly confirmed as Junction")
    confirmation.release(timestamp=1.0, reason="pointcloud_non_junction_release")
    expected = [
        AnchorJunctionState.NO_ANCHOR,
        AnchorJunctionState.PROVISIONAL_ANCHOR,
        AnchorJunctionState.REJECTED,
        AnchorJunctionState.RELEASED,
    ]
    if [item.state for item in confirmation.transitions] != expected:
        raise AssertionError("false-candidate lifecycle did not reject and release")
    save_confirmation_artifacts(
        output_dir=output_dir,
        confirmation=confirmation,
        scan=scan,
        anchor_position_drift=0.0,
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the test-only straight-corridor provisional-Anchor fixture."
    )
    parser.add_argument(
        "--output-dir",
        default="junction_detection/integration/output/false_corridor_confirmation",
    )
    args = parser.parse_args()
    decision = run_false_corridor_fixture(args.output_dir)
    print(
        "false_corridor: "
        f"openings={decision.opening_count}, decision=REJECTED, final=RELEASED"
    )


if __name__ == "__main__":
    main()
