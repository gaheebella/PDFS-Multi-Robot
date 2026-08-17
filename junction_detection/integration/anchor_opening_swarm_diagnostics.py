"""Opening-conditioned diagnostics for an existing Anchor-local swarm result.

This module is intentionally downstream-only.  It receives Point Cloud opening
intervals, Anchor-relative robot trend diagnostics, existing cohort IDs, and an
ID-only communication graph.  It has no simulator geometry, global position,
DFS phase, Branch label, expected direction, or control output.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Hashable, Mapping, Optional, Sequence

import numpy as np


RobotId = Hashable


@dataclass(frozen=True)
class OpeningInterval:
    """One detector-produced circular opening interval."""

    opening_id: int
    start_deg: float
    end_deg: float
    center_deg: float
    width_deg: float
    confidence: Optional[float] = None


@dataclass(frozen=True)
class RobotTrendInput:
    """Algorithm-safe subset of an existing validator robot diagnostic."""

    robot_id: RobotId
    motion_state: str
    representative_bearing_deg: float
    radial_slope: float
    ci_low: float
    existing_cohort_id: Optional[int]


@dataclass(frozen=True)
class CohortInput:
    """Existing validator cohort statistics; no regrouping is performed."""

    cohort_id: int
    member_robot_ids: tuple[RobotId, ...]
    circular_mean_deg: float
    circular_spread_deg: float
    mean_resultant_length: float
    mean_radial_slope: float
    minimum_ci_low: float


@dataclass(frozen=True)
class Assignment:
    """One robot's exact interval-membership result at one sample."""

    timestamp: float
    robot: RobotTrendInput
    assignment_state: str
    assigned_opening_id: Optional[int]
    matching_opening_ids: tuple[int, ...]
    nearest_opening_id: Optional[int]
    nearest_center_distance_deg: Optional[float]
    nearest_boundary_gap_deg: Optional[float]
    anchor_hop_distance: Optional[int]


def normalize_angle_deg(angle_deg: float) -> float:
    """Normalize an angle to ``[-180, 180)``."""
    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


def circular_distance_deg(first_deg: float, second_deg: float) -> float:
    """Return the unsigned shortest circular distance in degrees."""
    return abs(normalize_angle_deg(float(first_deg) - float(second_deg)))


def angle_in_opening(angle_deg: float, opening: OpeningInterval) -> bool:
    """Test exact circular interval membership without angular tolerance."""
    width = (opening.end_deg - opening.start_deg) % 360.0
    offset = (float(angle_deg) - opening.start_deg) % 360.0
    return offset <= width


def compute_anchor_hops(
    anchor_id: RobotId,
    neighbor_graph: Mapping[RobotId, Sequence[RobotId]],
) -> dict[RobotId, int]:
    """Compute directed BFS hop counts from existing ``comm_neighbors`` edges."""
    hops: dict[RobotId, int] = {anchor_id: 0}
    queue: deque[RobotId] = deque((anchor_id,))
    while queue:
        current = queue.popleft()
        for neighbor in neighbor_graph.get(current, ()):
            if neighbor in hops:
                continue
            hops[neighbor] = hops[current] + 1
            queue.append(neighbor)
    return hops


def _circular_statistics(angles_deg: Sequence[float]) -> tuple[Any, Any, Any]:
    if not angles_deg:
        return "", "", ""
    radians = np.deg2rad(np.asarray(angles_deg, dtype=float))
    mean_cos = float(np.mean(np.cos(radians)))
    mean_sin = float(np.mean(np.sin(radians)))
    resultant = min(1.0, math.hypot(mean_cos, mean_sin))
    mean_deg = normalize_angle_deg(math.degrees(math.atan2(mean_sin, mean_cos)))
    spread = (
        180.0 if resultant <= np.finfo(float).eps
        else min(180.0, math.degrees(math.sqrt(max(0.0, -2.0 * math.log(resultant)))))
    )
    return mean_deg, spread, resultant


def _hop_bucket(hop: Optional[int]) -> str:
    if hop is None:
        return "disconnected"
    if hop <= 4:
        return f"hop_{hop}"
    return "hop_5_plus"


def classify_robot(
    *,
    timestamp: float,
    robot: RobotTrendInput,
    openings: Sequence[OpeningInterval],
    anchor_hop_distance: Optional[int],
) -> Assignment:
    """Classify one robot using motion state and exact opening intervals only."""
    if robot.motion_state != "progressing":
        return Assignment(
            timestamp, robot, "not_progressing", None, (), None, None, None,
            anchor_hop_distance,
        )

    matching = tuple(
        opening.opening_id for opening in openings
        if angle_in_opening(robot.representative_bearing_deg, opening)
    )
    if len(matching) == 1:
        state = "inside_one_opening"
        assigned = matching[0]
    elif len(matching) > 1:
        state = "inside_multiple_openings"
        assigned = None
    else:
        state = "outside_all_openings"
        assigned = None

    nearest_id: Optional[int] = None
    nearest_center: Optional[float] = None
    nearest_boundary: Optional[float] = None
    if state == "outside_all_openings" and openings:
        nearest = min(
            openings,
            key=lambda opening: (
                circular_distance_deg(robot.representative_bearing_deg, opening.center_deg),
                opening.opening_id,
            ),
        )
        nearest_id = nearest.opening_id
        nearest_center = circular_distance_deg(
            robot.representative_bearing_deg, nearest.center_deg
        )
        nearest_boundary = min(
            circular_distance_deg(robot.representative_bearing_deg, nearest.start_deg),
            circular_distance_deg(robot.representative_bearing_deg, nearest.end_deg),
        )
    return Assignment(
        timestamp, robot, state, assigned, matching, nearest_id,
        nearest_center, nearest_boundary, anchor_hop_distance,
    )


@dataclass
class OpeningSwarmDiagnostics:
    """Accumulate downstream-only assignments and composition summaries."""

    assignment_rows: list[dict[str, Any]] = field(default_factory=list)
    cohort_rows: list[dict[str, Any]] = field(default_factory=list)
    cohort_membership_rows: list[dict[str, Any]] = field(default_factory=list)
    opening_rows: list[dict[str, Any]] = field(default_factory=list)
    latest_assignments: tuple[Assignment, ...] = ()
    latest_openings: tuple[OpeningInterval, ...] = ()
    latest_cohorts: tuple[CohortInput, ...] = ()
    sample_count: int = 0
    maximum_robot_count: int = 0

    def add_sample(
        self,
        *,
        timestamp: float,
        openings: Sequence[OpeningInterval],
        robots: Sequence[RobotTrendInput],
        cohorts: Sequence[CohortInput],
        anchor_id: RobotId,
        neighbor_graph: Mapping[RobotId, Sequence[RobotId]],
    ) -> None:
        """Analyze one existing validator sample without changing its result."""
        opening_tuple = tuple(openings)
        robot_tuple = tuple(robots)
        cohort_tuple = tuple(cohorts)
        hops = compute_anchor_hops(anchor_id, neighbor_graph)
        assignments = tuple(
            classify_robot(
                timestamp=timestamp,
                robot=robot,
                openings=opening_tuple,
                anchor_hop_distance=hops.get(robot.robot_id),
            )
            for robot in robot_tuple
        )
        if len(assignments) != len(robot_tuple):
            raise AssertionError("assignment count does not equal robot diagnostic count")
        self.sample_count += 1
        self.maximum_robot_count = max(self.maximum_robot_count, len(robot_tuple))
        self.latest_assignments = assignments
        self.latest_openings = opening_tuple
        self.latest_cohorts = cohort_tuple
        assignment_by_id = {item.robot.robot_id: item for item in assignments}
        if len(assignment_by_id) != len(assignments):
            raise AssertionError("duplicate robot ID in one diagnostic sample")

        for item in assignments:
            self.assignment_rows.append({
                "timestamp": timestamp,
                "robot_id": item.robot.robot_id,
                "motion_state": item.robot.motion_state,
                "representative_bearing_deg": item.robot.representative_bearing_deg,
                "existing_cohort_id": (
                    "" if item.robot.existing_cohort_id is None
                    else item.robot.existing_cohort_id
                ),
                "assignment_state": item.assignment_state,
                "assigned_opening_id": (
                    "" if item.assigned_opening_id is None
                    else item.assigned_opening_id
                ),
                "matching_opening_ids": json.dumps(item.matching_opening_ids),
                "nearest_opening_id": (
                    "" if item.nearest_opening_id is None else item.nearest_opening_id
                ),
                "nearest_center_distance_deg": (
                    "" if item.nearest_center_distance_deg is None
                    else item.nearest_center_distance_deg
                ),
                "nearest_boundary_gap_deg": (
                    "" if item.nearest_boundary_gap_deg is None
                    else item.nearest_boundary_gap_deg
                ),
                "radial_slope": item.robot.radial_slope,
                "ci_low": item.robot.ci_low,
                "anchor_hop_distance": (
                    "" if item.anchor_hop_distance is None
                    else item.anchor_hop_distance
                ),
            })

        for cohort in cohort_tuple:
            members = [assignment_by_id[robot_id] for robot_id in cohort.member_robot_ids]
            if len(members) != len(cohort.member_robot_ids):
                raise AssertionError("cohort member is missing from robot diagnostics")
            state_counts = Counter(item.assignment_state for item in members)
            opening_counts = Counter(
                item.assigned_opening_id for item in members
                if item.assignment_state == "inside_one_opening"
            )
            classified = (
                state_counts["inside_one_opening"]
                + state_counts["inside_multiple_openings"]
                + state_counts["outside_all_openings"]
            )
            if classified != len(members):
                raise AssertionError("cohort classification sum does not match cohort size")
            dominant = (
                max(opening_counts, key=lambda key: (opening_counts[key], -int(key)))
                if opening_counts else None
            )
            inside_count = state_counts["inside_one_opening"]
            dominant_count = 0 if dominant is None else opening_counts[dominant]
            hop_counts = Counter(_hop_bucket(item.anchor_hop_distance) for item in members)
            bearings = [item.robot.representative_bearing_deg for item in members]
            circular_mean, circular_spread, _ = _circular_statistics(bearings)
            self.cohort_rows.append({
                "timestamp": timestamp,
                "cohort_id": cohort.cohort_id,
                "total_robot_count": len(members),
                "inside_one_count": inside_count,
                "ambiguous_count": state_counts["inside_multiple_openings"],
                "outside_all_count": state_counts["outside_all_openings"],
                "dominant_opening_id": "" if dominant is None else dominant,
                "dominant_opening_count": dominant_count,
                "dominant_opening_ratio": (
                    "" if inside_count == 0 else dominant_count / inside_count
                ),
                "represented_opening_count": len(opening_counts),
                "opening_purity": (
                    "" if inside_count == 0 else dominant_count / inside_count
                ),
                "purity_unavailable_reason": (
                    "no_inside_one_opening_robots" if inside_count == 0 else ""
                ),
                "bearing_min_deg": min(bearings) if bearings else "",
                "bearing_max_deg": max(bearings) if bearings else "",
                "circular_mean_deg": circular_mean,
                "circular_spread_deg": circular_spread,
                "mean_resultant_length": cohort.mean_resultant_length,
                "mean_radial_slope": cohort.mean_radial_slope,
                "minimum_ci_low": cohort.minimum_ci_low,
                "hop_1_count": hop_counts["hop_1"],
                "hop_2_count": hop_counts["hop_2"],
                "hop_3_count": hop_counts["hop_3"],
                "hop_4_count": hop_counts["hop_4"],
                "hop_5_plus_count": hop_counts["hop_5_plus"],
                "disconnected_count": hop_counts["disconnected"],
            })
            for opening in opening_tuple:
                count = opening_counts[opening.opening_id]
                self.cohort_membership_rows.append({
                    "timestamp": timestamp,
                    "cohort_id": cohort.cohort_id,
                    "opening_id": opening.opening_id,
                    "robot_count": count,
                    "fraction_of_inside_one": (
                        "" if inside_count == 0 else count / inside_count
                    ),
                    "fraction_of_total_cohort": count / len(members),
                })

        for opening in opening_tuple:
            members = [
                item for item in assignments
                if item.assignment_state == "inside_one_opening"
                and item.assigned_opening_id == opening.opening_id
            ]
            cohort_ids = sorted({
                item.robot.existing_cohort_id for item in members
                if item.robot.existing_cohort_id is not None
            })
            bearings = [item.robot.representative_bearing_deg for item in members]
            circular_mean, circular_spread, _ = _circular_statistics(bearings)
            hop_counts = Counter(_hop_bucket(item.anchor_hop_distance) for item in members)
            self.opening_rows.append({
                "timestamp": timestamp,
                "opening_id": opening.opening_id,
                "center_deg": opening.center_deg,
                "start_deg": opening.start_deg,
                "end_deg": opening.end_deg,
                "width_deg": opening.width_deg,
                "pointcloud_confidence": (
                    "" if opening.confidence is None else opening.confidence
                ),
                "progressing_robot_count": len(members),
                "distinct_cohort_count": len(cohort_ids),
                "cohort_ids": json.dumps(cohort_ids),
                "circular_mean_deg": circular_mean,
                "circular_spread_deg": circular_spread,
                "mean_radial_slope": (
                    "" if not members
                    else float(np.mean([item.robot.radial_slope for item in members]))
                ),
                "hop_1_count": hop_counts["hop_1"],
                "hop_2_count": hop_counts["hop_2"],
                "hop_3_count": hop_counts["hop_3"],
                "hop_4_count": hop_counts["hop_4"],
                "hop_5_plus_count": hop_counts["hop_5_plus"],
                "disconnected_count": hop_counts["disconnected"],
                "empty_opening": not members,
            })

    def save(self, output_dir: str | Path) -> None:
        """Write all CSV and PNG diagnostics to a caller-selected directory."""
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        _write_rows(directory / "opening_robot_assignments.csv", self.assignment_rows)
        _write_rows(directory / "cohort_opening_composition.csv", self.cohort_rows)
        _write_rows(
            directory / "cohort_opening_membership_long.csv",
            self.cohort_membership_rows,
        )
        _write_rows(directory / "opening_swarm_summary.csv", self.opening_rows)
        self._save_plots(directory)

    def _save_plots(self, directory: Path) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        if self.cohort_rows:
            latest_time = max(row["timestamp"] for row in self.cohort_rows)
            rows = [row for row in self.cohort_rows if row["timestamp"] == latest_time]
            labels = [str(row["cohort_id"]) for row in rows]
            figure, axis = plt.subplots(figsize=(8, 4))
            bottom = np.zeros(len(rows))
            for key, label in (
                ("inside_one_count", "inside one"),
                ("ambiguous_count", "ambiguous"),
                ("outside_all_count", "outside all"),
            ):
                values = np.asarray([row[key] for row in rows], dtype=float)
                axis.bar(labels, values, bottom=bottom, label=label)
                bottom += values
            axis.set(xlabel="existing cohort ID", ylabel="robot count")
            axis.legend()
            figure.tight_layout()
            figure.savefig(directory / "cohort_opening_composition.png", dpi=160)
            plt.close(figure)

        if self.opening_rows:
            figure, axis = plt.subplots(figsize=(8, 4))
            opening_ids = sorted({row["opening_id"] for row in self.opening_rows})
            for opening_id in opening_ids:
                rows = [row for row in self.opening_rows if row["opening_id"] == opening_id]
                axis.plot(
                    [row["timestamp"] for row in rows],
                    [row["progressing_robot_count"] for row in rows],
                    label=f"opening {opening_id}",
                )
            axis.set(xlabel="simulation time [s]", ylabel="progressing robot count")
            axis.legend()
            axis.grid(alpha=0.3)
            figure.tight_layout()
            figure.savefig(directory / "opening_robot_counts_over_time.png", dpi=160)
            plt.close(figure)

        if self.latest_assignments:
            figure = plt.figure(figsize=(7, 6))
            axis = figure.add_subplot(111, projection="polar")
            for opening in self.latest_openings:
                start = math.radians(opening.start_deg)
                width = math.radians((opening.end_deg - opening.start_deg) % 360.0)
                axis.bar(start, 1.0, width=width, bottom=0.0, align="edge", alpha=0.12)
                for boundary in (opening.start_deg, opening.end_deg):
                    angle = math.radians(boundary)
                    axis.plot([angle, angle], [0.0, 1.0], color="black", linewidth=0.7)
            progressing = [
                item for item in self.latest_assignments
                if item.robot.motion_state == "progressing"
            ]
            cohort_ids = sorted({
                item.robot.existing_cohort_id for item in progressing
                if item.robot.existing_cohort_id is not None
            })
            colors = {cohort_id: f"C{index % 10}" for index, cohort_id in enumerate(cohort_ids)}
            for item in progressing:
                assigned = item.assignment_state == "inside_one_opening"
                axis.scatter(
                    [math.radians(item.robot.representative_bearing_deg)],
                    [0.72 if assigned else 0.88],
                    marker="o" if assigned else "x",
                    s=12,
                    color=colors.get(item.robot.existing_cohort_id, "0.5"),
                    alpha=0.65,
                )
            axis.plot([0.0, 0.0], [0.0, 1.0], "--", color="tab:red", label="Anchor local 0°")
            axis.set_title("Opening intervals and existing cohort bearings")
            axis.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12))
            figure.tight_layout()
            figure.savefig(directory / "opening_assignment_polar.png", dpi=160)
            plt.close(figure)

        if self.cohort_rows:
            latest_time = max(row["timestamp"] for row in self.cohort_rows)
            rows = [row for row in self.cohort_rows if row["timestamp"] == latest_time]
            figure, axis = plt.subplots(figsize=(8, 4))
            labels = [str(row["cohort_id"]) for row in rows]
            bottom = np.zeros(len(rows))
            for key in (
                "hop_1_count", "hop_2_count", "hop_3_count", "hop_4_count",
                "hop_5_plus_count", "disconnected_count",
            ):
                values = np.asarray([row[key] for row in rows], dtype=float)
                axis.bar(labels, values, bottom=bottom, label=key)
                bottom += values
            axis.set(xlabel="existing cohort ID", ylabel="robot count")
            axis.legend(fontsize=8)
            figure.tight_layout()
            figure.savefig(directory / "anchor_hop_composition_by_cohort.png", dpi=160)
            plt.close(figure)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write heterogeneous diagnostics, preserving first-seen column order."""
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


def save_full_lifecycle_comparison(
    output_dir: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Save externally assembled A/B/C lifecycle rows and a compact plot."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _write_rows(directory / "full_lifecycle_comparison.csv", rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axis = plt.subplots(figsize=(8, 4))
    labels = [str(row.get("run_type", "")) for row in rows]
    times = [float(row.get("simulation_time", 0.0) or 0.0) for row in rows]
    colors = ["tab:green" if row.get("done_reached") else "tab:orange" for row in rows]
    axis.bar(labels, times, color=colors)
    axis.set(ylabel="simulation time [s]", title="Full lifecycle comparison")
    figure.tight_layout()
    figure.savefig(directory / "full_lifecycle_comparison.png", dpi=160)
    plt.close(figure)


def run_synthetic_tests() -> None:
    """Validate wraparound, overlap, outside-all, no-opening, BFS, and sums."""
    wrap = OpeningInterval(0, 170.0, -170.0, -180.0, 20.0)
    overlap_a = OpeningInterval(0, -20.0, 20.0, 0.0, 40.0)
    overlap_b = OpeningInterval(1, 10.0, 30.0, 20.0, 20.0)
    robot = lambda rid, bearing, state="progressing", cohort=0: RobotTrendInput(
        rid, state, bearing, 0.2, 0.1, cohort
    )
    assert classify_robot(
        timestamp=0.0, robot=robot(1, 179.0), openings=(wrap,), anchor_hop_distance=1,
    ).assignment_state == "inside_one_opening"
    assert classify_robot(
        timestamp=0.0, robot=robot(2, 15.0), openings=(overlap_a, overlap_b),
        anchor_hop_distance=2,
    ).assignment_state == "inside_multiple_openings"
    outside = classify_robot(
        timestamp=0.0, robot=robot(3, 90.0), openings=(overlap_a,),
        anchor_hop_distance=None,
    )
    assert outside.assignment_state == "outside_all_openings"
    assert outside.nearest_opening_id == 0
    no_opening = classify_robot(
        timestamp=0.0, robot=robot(4, 0.0), openings=(), anchor_hop_distance=3,
    )
    assert no_opening.assignment_state == "outside_all_openings"
    assert no_opening.nearest_opening_id is None
    hops = compute_anchor_hops(0, {0: (1,), 1: (2,), 2: (3,), 3: ()})
    assert hops == {0: 0, 1: 1, 2: 2, 3: 3}

    diagnostics = OpeningSwarmDiagnostics()
    robots = (robot(1, -10.0), robot(2, 15.0), robot(3, 90.0))
    cohort = CohortInput(0, (1, 2, 3), 0.0, 1.0, 0.9, 0.2, 0.1)
    diagnostics.add_sample(
        timestamp=1.0,
        openings=(overlap_a, overlap_b),
        robots=robots,
        cohorts=(cohort,),
        anchor_id=0,
        neighbor_graph={0: (1,), 1: (2,), 2: (3,), 3: ()},
    )
    row = diagnostics.cohort_rows[-1]
    assert row["inside_one_count"] + row["ambiguous_count"] + row["outside_all_count"] == 3


if __name__ == "__main__":
    run_synthetic_tests()
    print("anchor_opening_swarm_diagnostics synthetic tests: PASS")
