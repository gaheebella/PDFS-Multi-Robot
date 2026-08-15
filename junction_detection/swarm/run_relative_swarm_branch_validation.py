"""Deterministic synthetic validation for relative swarm Branch candidates.

The synthetic harness creates only Anchor-relative range/bearing observations.
It never constructs or passes simulator world x/y coordinates to the validator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from junction_detection.swarm.relative_swarm_branch_validator import (
    RelativeSwarmBranchValidator,
    RobotObservation,
    ValidationResult,
)


@dataclass(frozen=True)
class SyntheticCase:
    """Simulator-only frames and their semantic expected Branch count."""

    name: str
    description: str
    semantic_expected_branch_count: int
    required_to_pass: bool
    known_limitation_reason: str | None
    visualization_reference_directions_deg: tuple[float, ...]
    frames: tuple[tuple[RobotObservation, ...], ...]
    measurement_noise_std_m: float
    bearing_noise_std_deg: float


@dataclass(frozen=True)
class SyntheticCaseResult:
    """One completed synthetic case and all temporal validator outputs."""

    case: SyntheticCase
    validation_results: tuple[ValidationResult, ...]

    @property
    def final_result(self) -> ValidationResult:
        """Return the final temporal validation result."""
        return self.validation_results[-1]


RangeFunction = Callable[[float, int], float]


def _irregular_timestamps(rng: np.random.Generator, count: int = 18) -> np.ndarray:
    """Generate deterministic positive but nonuniform frame intervals."""
    increments = rng.uniform(0.52, 0.68, size=count - 1)
    return np.r_[0.0, np.cumsum(increments)]


def _plateau_range(timestamp: float, _: int) -> float:
    """Move outward initially, then stop at the local wall proxy distance."""
    return 1.0 + min(0.42 * timestamp, 1.25)


def _progress_range(timestamp: float, _: int) -> float:
    """Continue moving radially through a synthetic open corridor."""
    return 1.0 + 0.36 * timestamp


def _temporarily_stalled_range(timestamp: float, frame_index: int) -> float:
    """Progress overall but repeat one previous range at a single frame."""
    if frame_index == 14:
        return 1.0 + 0.36 * (timestamp - 0.62)
    return _progress_range(timestamp, frame_index)


def _clique_neighbor_map(
    groups: Sequence[Sequence[str]],
) -> dict[str, set[str]]:
    """Create disconnected all-to-all neighbor groups for synthetic tests."""
    neighbors: dict[str, set[str]] = {}
    for group in groups:
        group_tuple = tuple(group)
        for robot_id in group_tuple:
            if robot_id in neighbors:
                raise ValueError(f"robot appears in multiple groups: {robot_id}")
            neighbors[robot_id] = {
                other for other in group_tuple if other != robot_id
            }
    return neighbors


def _chain_neighbor_map(robot_ids: Sequence[str]) -> dict[str, set[str]]:
    """Create a sparse R0--R1--... chain neighbor graph."""
    ordered = tuple(robot_ids)
    if len(set(ordered)) != len(ordered):
        raise ValueError("chain robot IDs must be unique")
    neighbors = {robot_id: set() for robot_id in ordered}
    for first, second in zip(ordered[:-1], ordered[1:]):
        neighbors[first].add(second)
        neighbors[second].add(first)
    return neighbors


def _merge_neighbor_maps(
    *maps: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Merge disjoint or partially overlapping synthetic neighbor maps."""
    merged: dict[str, set[str]] = {}
    for neighbor_map in maps:
        for robot_id, neighbor_ids in neighbor_map.items():
            merged.setdefault(robot_id, set()).update(neighbor_ids)
    return merged


def _add_undirected_edges(
    neighbor_map: dict[str, set[str]],
    edges: Sequence[tuple[str, str]],
) -> dict[str, set[str]]:
    """Return a copy with manually specified bridge links added."""
    result = {robot_id: set(neighbor_ids) for robot_id, neighbor_ids in neighbor_map.items()}
    for first, second in edges:
        if first == second:
            raise ValueError("manual neighbor edges cannot be self-links")
        if first not in result or second not in result:
            raise ValueError("manual edge endpoint is absent from the neighbor map")
        result[first].add(second)
        result[second].add(first)
    return result


def _build_case(
    *,
    name: str,
    description: str,
    semantic_expected_branch_count: int,
    required_to_pass: bool,
    known_limitation_reason: str | None,
    visualization_reference_directions_deg: tuple[float, ...] = (),
    timestamps: np.ndarray,
    robot_models: dict[str, tuple[RangeFunction, float]],
    neighbor_map: dict[str, set[str]],
    rng: np.random.Generator,
    measurement_noise_std_m: float,
    bearing_noise_std_deg: float,
) -> SyntheticCase:
    """Build local relative frames without constructing global positions."""
    if set(neighbor_map) != set(robot_models):
        raise ValueError("every synthetic robot must appear in the neighbor map")
    frames: list[tuple[RobotObservation, ...]] = []
    for frame_index, timestamp in enumerate(timestamps):
        frame: list[RobotObservation] = []
        for robot_id in sorted(robot_models):
            range_function, nominal_bearing = robot_models[robot_id]
            anchor_range = range_function(float(timestamp), frame_index)
            anchor_range += float(rng.normal(0.0, measurement_noise_std_m))
            bearing = nominal_bearing + float(rng.normal(0.0, bearing_noise_std_deg))
            frame.append(
                RobotObservation(
                    timestamp=float(timestamp),
                    robot_id=robot_id,
                    anchor_range_m=max(0.0, anchor_range),
                    anchor_bearing_deg=bearing,
                    neighbor_ids=tuple(sorted(neighbor_map[robot_id])),
                )
            )
        frames.append(tuple(frame))
    return SyntheticCase(
        name=name,
        description=description,
        semantic_expected_branch_count=semantic_expected_branch_count,
        required_to_pass=required_to_pass,
        known_limitation_reason=known_limitation_reason,
        visualization_reference_directions_deg=visualization_reference_directions_deg,
        frames=tuple(frames),
        measurement_noise_std_m=measurement_noise_std_m,
        bearing_noise_std_deg=bearing_noise_std_deg,
    )


def create_synthetic_cases(
    *,
    seed: int = 17,
    measurement_noise_std_m: float = 0.01,
    bearing_noise_std_deg: float = 0.4,
) -> tuple[SyntheticCase, ...]:
    """Create five regressions, one limitation case, and one sparse-graph case."""
    if measurement_noise_std_m < 0.0 or bearing_noise_std_deg < 0.0:
        raise ValueError("synthetic noise standard deviations must be non-negative")
    master_rng = np.random.default_rng(seed)
    timestamps = _irregular_timestamps(master_rng)

    wall_ids = tuple(f"wall_{index}" for index in range(6))
    case_1 = _build_case(
        name="case_1_wall",
        description="All robots plateau after initial outward motion.",
        semantic_expected_branch_count=0,
        required_to_pass=True,
        known_limitation_reason=None,
        timestamps=timestamps,
        robot_models={
            robot_id: (_plateau_range, -30.0 + 12.0 * index)
            for index, robot_id in enumerate(wall_ids)
        },
        neighbor_map=_clique_neighbor_map((wall_ids,)),
        rng=np.random.default_rng(int(master_rng.integers(0, 2**32 - 1))),
        measurement_noise_std_m=measurement_noise_std_m,
        bearing_noise_std_deg=bearing_noise_std_deg,
    )

    branch_ids = tuple(f"branch_{index}" for index in range(4))
    one_wall_ids = tuple(f"wall_{index}" for index in range(4))
    case_2 = _build_case(
        name="case_2_one_branch",
        description="One connected progressing cohort and one plateau group.",
        semantic_expected_branch_count=1,
        required_to_pass=True,
        known_limitation_reason=None,
        timestamps=timestamps,
        robot_models={
            **{
                robot_id: (_progress_range, 34.0 + 1.8 * (index - 1.5))
                for index, robot_id in enumerate(branch_ids)
            },
            **{
                robot_id: (_plateau_range, -120.0 + 8.0 * index)
                for index, robot_id in enumerate(one_wall_ids)
            },
        },
        neighbor_map=_clique_neighbor_map((branch_ids, one_wall_ids)),
        rng=np.random.default_rng(int(master_rng.integers(0, 2**32 - 1))),
        measurement_noise_std_m=measurement_noise_std_m,
        bearing_noise_std_deg=bearing_noise_std_deg,
    )

    branch_a = tuple(f"branch_a_{index}" for index in range(4))
    branch_b = tuple(f"branch_b_{index}" for index in range(4))
    two_wall_ids = tuple(f"wall_{index}" for index in range(3))
    case_3 = _build_case(
        name="case_3_two_branches",
        description="Two disconnected progressing cohorts at distinct bearings.",
        semantic_expected_branch_count=2,
        required_to_pass=True,
        known_limitation_reason=None,
        timestamps=timestamps,
        robot_models={
            **{
                robot_id: (_progress_range, -65.0 + 1.5 * (index - 1.5))
                for index, robot_id in enumerate(branch_a)
            },
            **{
                robot_id: (_progress_range, 82.0 + 1.5 * (index - 1.5))
                for index, robot_id in enumerate(branch_b)
            },
            **{
                robot_id: (_plateau_range, 0.0 + 8.0 * index)
                for index, robot_id in enumerate(two_wall_ids)
            },
        },
        neighbor_map=_clique_neighbor_map((branch_a, branch_b, two_wall_ids)),
        rng=np.random.default_rng(int(master_rng.integers(0, 2**32 - 1))),
        measurement_noise_std_m=measurement_noise_std_m,
        bearing_noise_std_deg=bearing_noise_std_deg,
    )

    stall_ids = tuple(f"cohort_{index}" for index in range(5))
    case_4 = _build_case(
        name="case_4_temporary_stall",
        description="One cohort member stalls for one frame while peers progress.",
        semantic_expected_branch_count=1,
        required_to_pass=True,
        known_limitation_reason=None,
        timestamps=timestamps,
        robot_models={
            robot_id: (
                _temporarily_stalled_range if index == 2 else _progress_range,
                112.0 + 1.4 * (index - 2.0),
            )
            for index, robot_id in enumerate(stall_ids)
        },
        neighbor_map=_clique_neighbor_map((stall_ids,)),
        rng=np.random.default_rng(int(master_rng.integers(0, 2**32 - 1))),
        measurement_noise_std_m=measurement_noise_std_m,
        bearing_noise_std_deg=bearing_noise_std_deg,
    )

    wrap_ids = tuple(f"wrap_{index}" for index in range(4))
    wrap_bearings = (178.5, 179.5, -179.5, -178.5)
    case_5 = _build_case(
        name="case_5_bearing_wraparound",
        description="One progressing cohort spans the -180/+180 bearing seam.",
        semantic_expected_branch_count=1,
        required_to_pass=True,
        known_limitation_reason=None,
        timestamps=timestamps,
        robot_models={
            robot_id: (_progress_range, wrap_bearings[index])
            for index, robot_id in enumerate(wrap_ids)
        },
        neighbor_map=_clique_neighbor_map((wrap_ids,)),
        rng=np.random.default_rng(int(master_rng.integers(0, 2**32 - 1))),
        measurement_noise_std_m=measurement_noise_std_m,
        bearing_noise_std_deg=bearing_noise_std_deg,
    )
    bridged_a = tuple(f"bridge_a_{index}" for index in range(4))
    bridged_b = tuple(f"bridge_b_{index}" for index in range(4))
    bridge_id = "central_bridge"
    bridged_map = _merge_neighbor_maps(
        _clique_neighbor_map((bridged_a,)),
        _clique_neighbor_map((bridged_b,)),
        {bridge_id: set()},
    )
    bridged_map = _add_undirected_edges(
        bridged_map,
        ((bridged_a[-1], bridge_id), (bridge_id, bridged_b[0])),
    )
    case_6 = _build_case(
        name="case_6_bridged_two_branches",
        description=(
            "Two semantic Branches merge into one progressing connected component "
            "through a central progressing robot."
        ),
        semantic_expected_branch_count=2,
        required_to_pass=False,
        known_limitation_reason=(
            "two semantic branches are merged by one progressing connected component"
        ),
        visualization_reference_directions_deg=(-58.0, 58.0),
        timestamps=timestamps,
        robot_models={
            **{
                robot_id: (_progress_range, -58.0 + 1.6 * (index - 1.5))
                for index, robot_id in enumerate(bridged_a)
            },
            **{
                robot_id: (_progress_range, 58.0 + 1.6 * (index - 1.5))
                for index, robot_id in enumerate(bridged_b)
            },
            bridge_id: (_progress_range, 0.0),
        },
        neighbor_map=bridged_map,
        rng=np.random.default_rng(int(master_rng.integers(0, 2**32 - 1))),
        measurement_noise_std_m=measurement_noise_std_m,
        bearing_noise_std_deg=bearing_noise_std_deg,
    )

    sparse_ids = tuple(f"sparse_{index}" for index in range(5))
    case_7 = _build_case(
        name="case_7_sparse_one_branch",
        description="Five progressing robots connected only as a sparse chain.",
        semantic_expected_branch_count=1,
        required_to_pass=True,
        known_limitation_reason=None,
        timestamps=timestamps,
        robot_models={
            robot_id: (_progress_range, -132.0 + 1.5 * (index - 2.0))
            for index, robot_id in enumerate(sparse_ids)
        },
        neighbor_map=_chain_neighbor_map(sparse_ids),
        rng=np.random.default_rng(int(master_rng.integers(0, 2**32 - 1))),
        measurement_noise_std_m=measurement_noise_std_m,
        bearing_noise_std_deg=bearing_noise_std_deg,
    )
    return case_1, case_2, case_3, case_4, case_5, case_6, case_7


def run_synthetic_case(
    case: SyntheticCase,
    *,
    temporal_window_s: float,
    minimum_observations: int,
    confidence_multiplier: float,
    minimum_cohort_size: int,
) -> SyntheticCaseResult:
    """Run one case through the public local-observation validator interface."""
    validator = RelativeSwarmBranchValidator(
        temporal_window_s=temporal_window_s,
        minimum_observations=minimum_observations,
        confidence_multiplier=confidence_multiplier,
        minimum_cohort_size=minimum_cohort_size,
    )
    results = tuple(validator.update(frame) for frame in case.frames)
    return SyntheticCaseResult(case=case, validation_results=results)


def _latest_observations(case: SyntheticCase) -> dict[str, RobotObservation]:
    """Index the final local-relative observation frame."""
    return {str(observation.robot_id): observation for observation in case.frames[-1]}


def plot_case_result(result: SyntheticCaseResult, output_path: Path) -> None:
    """Visualize ranges, slopes, local reconstruction, and Branch directions."""
    final = result.final_result
    final_states = {str(trend.robot_id): trend.state for trend in final.trends}
    colors = {
        "progressing": "tab:green",
        "non_progressing": "tab:blue",
        "uncertain": "tab:orange",
    }
    figure = plt.figure(figsize=(15, 11))
    range_axis = figure.add_subplot(2, 2, 1)
    slope_axis = figure.add_subplot(2, 2, 2)
    local_axis = figure.add_subplot(2, 2, 3)
    bearing_axis = figure.add_subplot(2, 2, 4, projection="polar")

    robot_ids = sorted(str(observation.robot_id) for observation in result.case.frames[0])
    for robot_id in robot_ids:
        times = []
        ranges = []
        for frame in result.case.frames:
            observation = next(obs for obs in frame if str(obs.robot_id) == robot_id)
            times.append(observation.timestamp)
            ranges.append(observation.anchor_range_m)
        state = final_states[robot_id]
        range_axis.plot(times, ranges, color=colors[state], alpha=0.75, label=robot_id)
    range_axis.set_title("A. Anchor-relative radial range vs time")
    range_axis.set_xlabel("timestamp [s]")
    range_axis.set_ylabel("Anchor-relative range [m]")
    range_axis.grid(alpha=0.3)

    trends = list(final.trends)
    x_positions = np.arange(len(trends))
    for index, trend in enumerate(trends):
        if not np.isfinite(trend.radial_slope_mps):
            continue
        lower_error = trend.radial_slope_mps - trend.slope_ci_low_mps
        upper_error = trend.slope_ci_high_mps - trend.radial_slope_mps
        slope_axis.errorbar(
            index,
            trend.radial_slope_mps,
            yerr=[[lower_error], [upper_error]],
            fmt="o",
            color=colors[trend.state],
            capsize=3,
        )
    slope_axis.axhline(0.0, color="black", linewidth=0.8)
    slope_axis.set_xticks(x_positions, [str(trend.robot_id) for trend in trends], rotation=60)
    slope_axis.set_title("B. Final OLS radial slope and confidence interval")
    slope_axis.set_ylabel("radial slope [m/s]")
    slope_axis.grid(alpha=0.3)

    latest = _latest_observations(result.case)
    local_positions: dict[str, np.ndarray] = {}
    for robot_id, observation in latest.items():
        angle_rad = np.deg2rad(observation.anchor_bearing_deg)
        local_positions[robot_id] = observation.anchor_range_m * np.array(
            [np.cos(angle_rad), np.sin(angle_rad)]
        )
    drawn_edges: set[tuple[str, str]] = set()
    for robot_id, observation in latest.items():
        for neighbor_id_raw in observation.neighbor_ids:
            neighbor_id = str(neighbor_id_raw)
            if neighbor_id not in local_positions:
                continue
            edge = tuple(sorted((robot_id, neighbor_id)))
            if edge in drawn_edges:
                continue
            drawn_edges.add(edge)
            points = np.vstack((local_positions[robot_id], local_positions[neighbor_id]))
            local_axis.plot(points[:, 0], points[:, 1], color="0.75", linewidth=0.7)
    for state in ("progressing", "non_progressing", "uncertain"):
        members = [robot_id for robot_id in robot_ids if final_states[robot_id] == state]
        if not members:
            continue
        points = np.vstack([local_positions[robot_id] for robot_id in members])
        local_axis.scatter(points[:, 0], points[:, 1], color=colors[state], label=state)
    local_axis.scatter(0.0, 0.0, marker="*", s=140, color="red", label="Anchor")
    for candidate in final.branch_candidates:
        theta = np.deg2rad(candidate.estimated_direction_deg)
        radius = max(observation.anchor_range_m for observation in latest.values())
        local_axis.plot(
            [0.0, radius * np.cos(theta)], [0.0, radius * np.sin(theta)],
            linestyle="--", linewidth=1.5, label=f"Branch {candidate.cohort_id}",
        )
    for index, direction_deg in enumerate(
        result.case.visualization_reference_directions_deg
    ):
        theta = np.deg2rad(direction_deg)
        radius = max(observation.anchor_range_m for observation in latest.values())
        local_axis.plot(
            [0.0, radius * np.cos(theta)],
            [0.0, radius * np.sin(theta)],
            color="black",
            linestyle=":",
            linewidth=1.2,
            label="semantic Branch reference (test only)" if index == 0 else None,
        )
    local_axis.set_title("C. Reconstructed Anchor-local cohort graph")
    local_axis.set_xlabel("local x from range/bearing [m]")
    local_axis.set_ylabel("local y from range/bearing [m]")
    local_axis.set_aspect("equal", adjustable="box")
    local_axis.grid(alpha=0.3)
    local_axis.legend(fontsize=8)

    for state in ("progressing", "non_progressing", "uncertain"):
        members = [robot_id for robot_id in robot_ids if final_states[robot_id] == state]
        if not members:
            continue
        bearing_axis.scatter(
            np.deg2rad([latest[robot_id].anchor_bearing_deg for robot_id in members]),
            [latest[robot_id].anchor_range_m for robot_id in members],
            color=colors[state], label=state,
        )
    max_radius = max(observation.anchor_range_m for observation in latest.values())
    for candidate in final.branch_candidates:
        theta = np.deg2rad(candidate.estimated_direction_deg)
        bearing_axis.plot(
            [theta, theta], [0.0, max_radius], linewidth=2.0,
            label=f"cohort {candidate.cohort_id}: {candidate.estimated_direction_deg:.1f} deg",
        )
    for index, direction_deg in enumerate(
        result.case.visualization_reference_directions_deg
    ):
        theta = np.deg2rad(direction_deg)
        bearing_axis.plot(
            [theta, theta],
            [0.0, max_radius],
            color="black",
            linestyle=":",
            linewidth=1.2,
            label="semantic reference (test only)" if index == 0 else None,
        )
    bearing_axis.set_title("D. Circular bearing and detected Branch direction")
    bearing_axis.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, -0.25))

    figure.suptitle(
        f"{result.case.name}: semantic expected="
        f"{result.case.semantic_expected_branch_count}, "
        f"detected={len(final.branch_candidates)}",
        fontsize=14,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--range-noise-std", type=float, default=0.01)
    parser.add_argument("--bearing-noise-std", type=float, default=0.4)
    parser.add_argument("--temporal-window", type=float, default=4.0)
    parser.add_argument("--minimum-observations", type=int, default=5)
    parser.add_argument(
        "--confidence-multiplier",
        type=float,
        default=1.96,
        help=(
            "normal-approximation confidence multiplier for synthetic validation; "
            "1.96 does not guarantee exact small-sample 95%% coverage; evaluate "
            "Student-t, bootstrap, or empirical calibration on real SPH/robot data"
        ),
    )
    parser.add_argument("--minimum-cohort-size", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/relative_swarm_branch_validation"),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run all required scenarios and fail if an expected count is missed."""
    args = _parse_args()
    cases = create_synthetic_cases(
        seed=args.seed,
        measurement_noise_std_m=args.range_noise_std,
        bearing_noise_std_deg=args.bearing_noise_std,
    )
    results = tuple(
        run_synthetic_case(
            case,
            temporal_window_s=args.temporal_window,
            minimum_observations=args.minimum_observations,
            confidence_multiplier=args.confidence_multiplier,
            minimum_cohort_size=args.minimum_cohort_size,
        )
        for case in cases
    )

    print("=== Relative Swarm Branch Validation ===")
    print("Validator input: timestamp, robot ID, Anchor-relative range/bearing, neighbor IDs")
    print("Validator input excludes: global x/y, map, walls, known Branch count/directions")
    print(
        f"Synthetic-only noise: range={args.range_noise_std:.4f} m, "
        f"bearing={args.bearing_noise_std:.3f} deg, seed={args.seed}"
    )
    print(
        f"Inference settings: window={args.temporal_window:.2f} s, "
        f"minimum_observations={args.minimum_observations}, "
        f"confidence_multiplier={args.confidence_multiplier:.3f}, "
        f"minimum_cohort_size={args.minimum_cohort_size}"
    )

    all_passed = True
    for result in results:
        final = result.final_result
        detected = len(final.branch_candidates)
        passed = detected == result.case.semantic_expected_branch_count
        if result.case.required_to_pass:
            status = "PASS" if passed else "FAIL"
            all_passed &= passed
        elif passed:
            status = "DIAGNOSTIC MATCH"
        else:
            status = "KNOWN LIMITATION"
        print(
            f"{result.case.name}: semantic_expected="
            f"{result.case.semantic_expected_branch_count}, "
            f"detected={detected} -> {status}"
        )
        if status == "KNOWN LIMITATION":
            print(f"  KNOWN LIMITATION: {result.case.known_limitation_reason}")
        print(
            f"  progressing={len(final.progressing_robot_ids)}, "
            f"non_progressing={len(final.non_progressing_robot_ids)}, "
            f"uncertain={len(final.uncertain_robot_ids)}"
        )
        for candidate in final.branch_candidates:
            print(
                f"  cohort={candidate.cohort_id}, robots={candidate.robot_count}, "
                f"direction={candidate.estimated_direction_deg:.2f} deg, "
                f"spread={candidate.circular_bearing_spread_deg:.2f} deg, "
                f"mean_slope={candidate.mean_radial_slope_mps:.4f} m/s, "
                f"min_CI_low={candidate.min_slope_ci_low_mps:.4f} m/s"
            )
        if not args.no_plots:
            path = args.output_dir / f"{result.case.name}.png"
            plot_case_result(result, path)
            print(f"  plot={path}")

    if not all_passed:
        raise SystemExit("one or more required synthetic validation cases failed")


if __name__ == "__main__":
    main()
