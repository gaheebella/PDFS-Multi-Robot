"""Quantify bearing-gap sensitivity of the relative swarm Branch validator.

This module is an evaluation harness, not a second detector. Synthetic semantic
Branch counts, directions, and bridge identities are retained only by the
evaluation layer. The validator itself receives only timestamp, robot ID,
Anchor-relative range/bearing, and neighbor IDs through ``RobotObservation``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations
from math import isfinite, sqrt
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from junction_detection.swarm.relative_swarm_branch_validator import (
    RelativeSwarmBranchValidator,
    RobotObservation,
    ValidationResult,
    circular_distance_deg,
)


TWO_BRANCH_SEPARATIONS_DEG = (5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120)
GRADUAL_CHAIN_SEPARATIONS_DEG = (30, 60, 90, 120)
TWO_BRANCH_WIDTHS_DEG = (2, 5, 10, 15, 20, 30)
WIDE_BRANCH_WIDTHS_DEG = (2, 5, 10, 15, 20, 30, 40, 50, 60, 80)
BEARING_GAP_THRESHOLDS_DEG = (5, 10, 15, 20, 25, 30, 40, 50, 60)
GRADUAL_CHAIN_GAPS_DEG = (10, 15, 20, 25, 30)
GRADUAL_BRIDGE_COUNTS = (0, 1, 2, 3, 5)
NEIGHBOR_STRUCTURES = ("clique", "chain")


@dataclass(frozen=True)
class SensitivityConfiguration:
    """One parameter combination shared by all seeds in an aggregate."""

    experiment: str
    branch_separation_deg: float | None
    branch_width_deg: float | None
    bearing_gap_threshold_deg: float
    bridge_robot_count: int
    neighbor_structure: str
    ambiguous_configuration: bool = False


@dataclass(frozen=True)
class SyntheticScanSequence:
    """Algorithm-visible frames plus evaluation-only semantic annotations."""

    frames: tuple[tuple[RobotObservation, ...], ...]
    expected_branch_count: int
    reference_directions_deg: tuple[float, ...]
    bridge_robot_ids: tuple[str, ...]


@dataclass
class SummaryAccumulator:
    """Online aggregate that avoids retaining the full raw experiment in memory."""

    run_count: int = 0
    evaluated_run_count: int = 0
    exact_count: int = 0
    merge_count: int = 0
    over_split_count: int = 0
    direction_count: int = 0
    direction_sum: float = 0.0
    direction_square_sum: float = 0.0
    detected_count_sum: float = 0.0
    rejected_robot_sum: float = 0.0
    bearing_incompatible_edge_sum: float = 0.0

    def add_ambiguous(self) -> None:
        """Record one deliberately unevaluated overlapping semantic setup."""
        self.run_count += 1

    def add_result(
        self,
        *,
        exact: bool,
        merge: bool,
        over_split: bool,
        direction_error_deg: float,
        detected_count: int,
        rejected_robot_count: int,
        incompatible_edge_count: int,
    ) -> None:
        """Accumulate one evaluated validator run."""
        self.run_count += 1
        self.evaluated_run_count += 1
        self.exact_count += int(exact)
        self.merge_count += int(merge)
        self.over_split_count += int(over_split)
        self.detected_count_sum += detected_count
        self.rejected_robot_sum += rejected_robot_count
        self.bearing_incompatible_edge_sum += incompatible_edge_count
        if isfinite(direction_error_deg):
            self.direction_count += 1
            self.direction_sum += direction_error_deg
            self.direction_square_sum += direction_error_deg * direction_error_deg

    def as_row(self, configuration: SensitivityConfiguration) -> dict[str, object]:
        """Convert aggregate counters into one summary CSV row."""
        evaluated = self.evaluated_run_count
        direction_mean = (
            self.direction_sum / self.direction_count
            if self.direction_count
            else float("nan")
        )
        direction_variance = (
            max(
                0.0,
                self.direction_square_sum / self.direction_count
                - direction_mean * direction_mean,
            )
            if self.direction_count
            else float("nan")
        )
        return {
            **_configuration_columns(configuration),
            "run_count": self.run_count,
            "evaluated_run_count": evaluated,
            "exact_success_count": self.exact_count,
            "merge_count": self.merge_count,
            "over_split_count": self.over_split_count,
            "exact_count_accuracy": _safe_ratio(self.exact_count, evaluated),
            "merge_rate": _safe_ratio(self.merge_count, evaluated),
            "over_split_rate": _safe_ratio(self.over_split_count, evaluated),
            "direction_mae_mean_deg": direction_mean,
            "direction_mae_std_deg": sqrt(direction_variance),
            "detected_branch_count_mean": _safe_ratio(
                self.detected_count_sum, evaluated
            ),
            "rejected_robot_count_mean": _safe_ratio(
                self.rejected_robot_sum, evaluated
            ),
            "bearing_incompatible_edge_count_mean": _safe_ratio(
                self.bearing_incompatible_edge_sum, evaluated
            ),
        }


RAW_FIELDNAMES = (
    "experiment",
    "case_configuration",
    "seed",
    "expected_branch_count",
    "detected_branch_count",
    "status",
    "merge",
    "over_split",
    "ambiguous_configuration",
    "branch_separation_deg",
    "branch_width_deg",
    "bearing_gap_threshold_deg",
    "bridge_robot_count",
    "neighbor_structure",
    "range_noise_std_m",
    "bearing_noise_std_deg",
    "circular_direction_error_deg",
    "reference_directions_deg",
    "detected_directions_deg",
    "progressing_count",
    "returning_count",
    "non_progressing_count",
    "insufficient_count",
    "rejected_robot_count",
    "rejected_bridge_robot_count",
    "bearing_incompatible_edge_count",
    "progressing_component_count",
    "bearing_subcohort_count",
    "valid_cohort_robot_counts",
    "bridge_assignments",
)

SUMMARY_FIELDNAMES = (
    "experiment",
    "branch_separation_deg",
    "branch_width_deg",
    "bearing_gap_threshold_deg",
    "bridge_robot_count",
    "neighbor_structure",
    "ambiguous_configuration",
    "run_count",
    "evaluated_run_count",
    "exact_success_count",
    "merge_count",
    "over_split_count",
    "exact_count_accuracy",
    "merge_rate",
    "over_split_rate",
    "direction_mae_mean_deg",
    "direction_mae_std_deg",
    "detected_branch_count_mean",
    "rejected_robot_count_mean",
    "bearing_incompatible_edge_count_mean",
)


def _safe_ratio(numerator: float, denominator: int) -> float:
    """Return a ratio or NaN when a semantic group was not evaluated."""
    return float(numerator / denominator) if denominator else float("nan")


def _normalize_angle_deg(angle_deg: float) -> float:
    """Normalize an evaluation-only reference angle to [-180, 180)."""
    return float((angle_deg + 180.0) % 360.0 - 180.0)


def _parse_number_list(
    text: str | None, cast: Callable[[str], float | int]
) -> tuple:
    """Parse an optional comma-separated numeric CLI list."""
    if text is None or not text.strip():
        return ()
    return tuple(cast(token.strip()) for token in text.split(",") if token.strip())


def _parse_structures(text: str) -> tuple[str, ...]:
    """Parse and validate comma-separated topology names."""
    structures = tuple(token.strip() for token in text.split(",") if token.strip())
    invalid = set(structures) - set(NEIGHBOR_STRUCTURES)
    if invalid:
        raise ValueError(f"unknown neighbor structures: {sorted(invalid)}")
    if not structures:
        raise ValueError("at least one neighbor structure is required")
    return structures


def _clique_neighbor_map(robot_ids: Sequence[str]) -> dict[str, set[str]]:
    """Return reciprocal all-to-all links for one semantic group."""
    ordered = tuple(robot_ids)
    return {
        robot_id: {other for other in ordered if other != robot_id}
        for robot_id in ordered
    }


def _chain_neighbor_map(robot_ids: Sequence[str]) -> dict[str, set[str]]:
    """Return reciprocal links only between adjacent ordered robots."""
    ordered = tuple(robot_ids)
    neighbors = {robot_id: set() for robot_id in ordered}
    for first, second in zip(ordered[:-1], ordered[1:]):
        neighbors[first].add(second)
        neighbors[second].add(first)
    return neighbors


def _topology_neighbor_map(
    robot_ids: Sequence[str], structure: str
) -> dict[str, set[str]]:
    """Build one of the explicitly requested intra-Branch topologies."""
    if structure == "clique":
        return _clique_neighbor_map(robot_ids)
    if structure == "chain":
        return _chain_neighbor_map(robot_ids)
    raise ValueError(f"unsupported neighbor structure: {structure}")


def _merge_neighbor_maps(
    *neighbor_maps: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Merge synthetic topology fragments without changing validator behavior."""
    merged: dict[str, set[str]] = {}
    for neighbor_map in neighbor_maps:
        for robot_id, neighbors in neighbor_map.items():
            merged.setdefault(robot_id, set()).update(neighbors)
    return merged


def _add_reciprocal_edge(
    neighbor_map: dict[str, set[str]], first: str, second: str
) -> None:
    """Add a manually specified reciprocal synthetic bridge edge."""
    neighbor_map[first].add(second)
    neighbor_map[second].add(first)


def _make_frames(
    *,
    nominal_bearings_deg: dict[str, float],
    neighbor_map: dict[str, set[str]],
    rng: np.random.Generator,
    range_noise_std_m: float,
    bearing_noise_std_deg: float,
) -> tuple[tuple[RobotObservation, ...], ...]:
    """Create five irregular-compatible samples using local measurements only."""
    if set(nominal_bearings_deg) != set(neighbor_map):
        raise ValueError("every synthetic robot must have a neighbor-map entry")
    timestamps = (0.0, 0.83, 1.91, 2.88, 3.94)
    frames: list[tuple[RobotObservation, ...]] = []
    for timestamp in timestamps:
        frame: list[RobotObservation] = []
        for robot_id in sorted(nominal_bearings_deg):
            radius = 1.0 + 0.36 * timestamp
            radius += float(rng.normal(0.0, range_noise_std_m))
            bearing = nominal_bearings_deg[robot_id]
            bearing += float(rng.normal(0.0, bearing_noise_std_deg))
            frame.append(
                RobotObservation(
                    timestamp=timestamp,
                    robot_id=robot_id,
                    anchor_range_m=max(0.0, radius),
                    anchor_bearing_deg=bearing,
                    neighbor_ids=tuple(sorted(neighbor_map[robot_id])),
                )
            )
        frames.append(tuple(frame))
    return tuple(frames)


def _build_two_branch_sequence(
    configuration: SensitivityConfiguration,
    rng: np.random.Generator,
    range_noise_std_m: float,
    bearing_noise_std_deg: float,
) -> SyntheticScanSequence:
    """Build two equal progressing groups, optionally joined by one bridge."""
    separation = float(configuration.branch_separation_deg)
    width = float(configuration.branch_width_deg)
    rotation = float(rng.uniform(-180.0, 180.0))
    first_center = rotation - 0.5 * separation
    second_center = rotation + 0.5 * separation
    first_ids = tuple(f"r{index}" for index in range(3))
    second_ids = tuple(f"r{index}" for index in range(3, 6))
    offsets = np.linspace(-0.5 * width, 0.5 * width, len(first_ids))
    bearings = {
        **{
            robot_id: first_center + float(offset)
            for robot_id, offset in zip(first_ids, offsets)
        },
        **{
            robot_id: second_center + float(offset)
            for robot_id, offset in zip(second_ids, offsets)
        },
    }
    neighbors = _merge_neighbor_maps(
        _topology_neighbor_map(first_ids, configuration.neighbor_structure),
        _topology_neighbor_map(second_ids, configuration.neighbor_structure),
    )
    bridge_ids: tuple[str, ...] = ()
    if configuration.bridge_robot_count:
        bridge_id = "r6"
        bridge_ids = (bridge_id,)
        bearings[bridge_id] = rotation
        neighbors[bridge_id] = set()
        _add_reciprocal_edge(neighbors, first_ids[-1], bridge_id)
        _add_reciprocal_edge(neighbors, bridge_id, second_ids[0])
    return SyntheticScanSequence(
        frames=_make_frames(
            nominal_bearings_deg=bearings,
            neighbor_map=neighbors,
            rng=rng,
            range_noise_std_m=range_noise_std_m,
            bearing_noise_std_deg=bearing_noise_std_deg,
        ),
        expected_branch_count=2,
        reference_directions_deg=(
            _normalize_angle_deg(first_center),
            _normalize_angle_deg(second_center),
        ),
        bridge_robot_ids=bridge_ids,
    )


def _build_wide_branch_sequence(
    configuration: SensitivityConfiguration,
    rng: np.random.Generator,
    range_noise_std_m: float,
    bearing_noise_std_deg: float,
) -> SyntheticScanSequence:
    """Build one semantic Branch spanning an explicitly controlled width."""
    width = float(configuration.branch_width_deg)
    center = float(rng.uniform(-180.0, 180.0))
    robot_ids = tuple(f"r{index}" for index in range(5))
    offsets = np.linspace(-0.5 * width, 0.5 * width, len(robot_ids))
    bearings = {
        robot_id: center + float(offset)
        for robot_id, offset in zip(robot_ids, offsets)
    }
    neighbors = _topology_neighbor_map(robot_ids, configuration.neighbor_structure)
    return SyntheticScanSequence(
        frames=_make_frames(
            nominal_bearings_deg=bearings,
            neighbor_map=neighbors,
            rng=rng,
            range_noise_std_m=range_noise_std_m,
            bearing_noise_std_deg=bearing_noise_std_deg,
        ),
        expected_branch_count=1,
        reference_directions_deg=(_normalize_angle_deg(center),),
        bridge_robot_ids=(),
    )


def _build_gradual_chain_sequence(
    configuration: SensitivityConfiguration,
    rng: np.random.Generator,
    range_noise_std_m: float,
    bearing_noise_std_deg: float,
) -> SyntheticScanSequence:
    """Build two endpoint groups joined through gradual intermediate bearings."""
    separation = float(configuration.branch_separation_deg)
    rotation = float(rng.uniform(-180.0, 180.0))
    first_center = rotation - 0.5 * separation
    second_center = rotation + 0.5 * separation
    first_ids = ("r0", "r1", "r2")
    second_ids = ("r3", "r4", "r5")
    bearings = {
        **{
            robot_id: first_center + offset
            for robot_id, offset in zip(first_ids, (-1.0, 0.0, 1.0))
        },
        **{
            robot_id: second_center + offset
            for robot_id, offset in zip(second_ids, (-1.0, 0.0, 1.0))
        },
    }
    neighbors = _merge_neighbor_maps(
        _clique_neighbor_map(first_ids), _clique_neighbor_map(second_ids)
    )
    bridge_ids = tuple(
        f"r{6 + index}" for index in range(configuration.bridge_robot_count)
    )
    for index, bridge_id in enumerate(bridge_ids):
        fraction = (index + 1.0) / (len(bridge_ids) + 1.0)
        bearings[bridge_id] = first_center + fraction * separation
        neighbors[bridge_id] = set()
    path = (first_ids[-1], *bridge_ids, second_ids[0])
    for first, second in zip(path[:-1], path[1:]):
        _add_reciprocal_edge(neighbors, first, second)
    return SyntheticScanSequence(
        frames=_make_frames(
            nominal_bearings_deg=bearings,
            neighbor_map=neighbors,
            rng=rng,
            range_noise_std_m=range_noise_std_m,
            bearing_noise_std_deg=bearing_noise_std_deg,
        ),
        expected_branch_count=2,
        reference_directions_deg=(
            _normalize_angle_deg(first_center),
            _normalize_angle_deg(second_center),
        ),
        bridge_robot_ids=bridge_ids,
    )


def _run_validator(
    frames: Sequence[Sequence[RobotObservation]], bearing_gap_threshold_deg: float
) -> ValidationResult:
    """Run the unchanged public validator interface and return its final result."""
    validator = RelativeSwarmBranchValidator(
        temporal_window_s=4.0,
        minimum_observations=5,
        confidence_level=0.95,
        minimum_cohort_size=2,
        neighbor_edge_policy="reciprocal",
        maximum_neighbor_bearing_gap_deg=bearing_gap_threshold_deg,
    )
    final_result: ValidationResult | None = None
    for frame in frames:
        final_result = validator.update(frame)
    if final_result is None:
        raise ValueError("synthetic sequence must contain at least one frame")
    return final_result


def _direction_mae_deg(
    detected_directions_deg: Sequence[float], reference_directions_deg: Sequence[float]
) -> float:
    """Compute optimal one-to-one circular MAE when counts match.

    For count errors, unmatched semantic directions are undefined, so the metric
    falls back to the mean nearest-reference error for available candidates.
    """
    detected = tuple(detected_directions_deg)
    references = tuple(reference_directions_deg)
    if not detected or not references:
        return float("nan")
    if len(detected) == len(references):
        return min(
            float(
                np.mean(
                    [
                        circular_distance_deg(candidate, reference)
                        for candidate, reference in zip(detected, ordering)
                    ]
                )
            )
            for ordering in permutations(references)
        )
    return float(
        np.mean(
            [
                min(circular_distance_deg(candidate, ref) for ref in references)
                for candidate in detected
            ]
        )
    )


def _configuration_columns(
    configuration: SensitivityConfiguration,
) -> dict[str, object]:
    """Return consistent configuration columns for raw and summary outputs."""
    return {
        "experiment": configuration.experiment,
        "branch_separation_deg": configuration.branch_separation_deg,
        "branch_width_deg": configuration.branch_width_deg,
        "bearing_gap_threshold_deg": configuration.bearing_gap_threshold_deg,
        "bridge_robot_count": configuration.bridge_robot_count,
        "neighbor_structure": configuration.neighbor_structure,
        "ambiguous_configuration": configuration.ambiguous_configuration,
    }


def _configuration_json(configuration: SensitivityConfiguration) -> str:
    """Serialize a stable human-readable case configuration."""
    return json.dumps(
        _configuration_columns(configuration), sort_keys=True, separators=(",", ":")
    )


def _ambiguous_raw_row(
    configuration: SensitivityConfiguration,
    seed: int,
    range_noise_std_m: float,
    bearing_noise_std_deg: float,
) -> dict[str, object]:
    """Create a raw row for overlapping semantic intervals without inference."""
    return {
        **{field: "" for field in RAW_FIELDNAMES},
        **_configuration_columns(configuration),
        "case_configuration": _configuration_json(configuration),
        "seed": seed,
        "expected_branch_count": 2,
        "status": "AMBIGUOUS",
        "range_noise_std_m": range_noise_std_m,
        "bearing_noise_std_deg": bearing_noise_std_deg,
    }


def _evaluated_raw_row(
    *,
    configuration: SensitivityConfiguration,
    seed: int,
    sequence: SyntheticScanSequence,
    result: ValidationResult,
    range_noise_std_m: float,
    bearing_noise_std_deg: float,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build one raw CSV row and compact numeric values for aggregation."""
    detected_directions = tuple(
        candidate.estimated_direction_deg for candidate in result.branch_candidates
    )
    detected_count = len(detected_directions)
    expected_count = sequence.expected_branch_count
    exact = detected_count == expected_count
    merge = detected_count < expected_count
    over_split = detected_count > expected_count
    direction_error = _direction_mae_deg(
        detected_directions, sequence.reference_directions_deg
    )
    diagnostics_by_id = {
        str(diagnostic.robot_id): diagnostic for diagnostic in result.robot_diagnostics
    }
    rejected_robot_count = sum(
        diagnostic.motion_state == "progressing" and diagnostic.excluded
        for diagnostic in result.robot_diagnostics
    )
    rejected_bridge_count = sum(
        diagnostics_by_id[robot_id].excluded for robot_id in sequence.bridge_robot_ids
    )
    incompatible_edge_count = sum(
        edge.reason == "bearing_incompatible_neighbor_link"
        for edge in result.rejected_neighbor_edges
    )
    bridge_assignments = []
    for robot_id in sequence.bridge_robot_ids:
        diagnostic = diagnostics_by_id[robot_id]
        assignment = (
            f"cohort_{diagnostic.final_cohort_id}"
            if diagnostic.final_cohort_id is not None
            else diagnostic.exclusion_reason
        )
        bridge_assignments.append(f"{robot_id}:{assignment}")
    total_subcohorts = len(result.bearing_subcohorts) + len(
        result.rejected_bearing_subcohorts
    )
    row = {
        **_configuration_columns(configuration),
        "case_configuration": _configuration_json(configuration),
        "seed": seed,
        "expected_branch_count": expected_count,
        "detected_branch_count": detected_count,
        "status": "PASS" if exact else "FAIL",
        "merge": merge,
        "over_split": over_split,
        "range_noise_std_m": range_noise_std_m,
        "bearing_noise_std_deg": bearing_noise_std_deg,
        "circular_direction_error_deg": direction_error,
        "reference_directions_deg": ";".join(
            f"{direction:.6f}" for direction in sequence.reference_directions_deg
        ),
        "detected_directions_deg": ";".join(
            f"{direction:.6f}" for direction in detected_directions
        ),
        "progressing_count": len(result.progressing_robot_ids),
        "returning_count": len(result.returning_robot_ids),
        "non_progressing_count": len(result.non_progressing_robot_ids),
        "insufficient_count": len(result.insufficient_robot_ids),
        "rejected_robot_count": rejected_robot_count,
        "rejected_bridge_robot_count": rejected_bridge_count,
        "bearing_incompatible_edge_count": incompatible_edge_count,
        "progressing_component_count": len(result.progressing_components),
        "bearing_subcohort_count": total_subcohorts,
        "valid_cohort_robot_counts": ";".join(
            str(candidate.robot_count) for candidate in result.branch_candidates
        ),
        "bridge_assignments": ";".join(bridge_assignments),
    }
    aggregate_values = {
        "exact": exact,
        "merge": merge,
        "over_split": over_split,
        "direction_error_deg": direction_error,
        "detected_count": detected_count,
        "rejected_robot_count": rejected_robot_count,
        "incompatible_edge_count": incompatible_edge_count,
    }
    return row, aggregate_values


def _experiment_configurations(args: argparse.Namespace) -> tuple[SensitivityConfiguration, ...]:
    """Expand CLI lists into deterministic experiment parameter combinations."""
    requested = (
        ("two-branch", "wide-one-branch", "gradual-chain")
        if args.experiment == "all"
        else (args.experiment,)
    )
    custom_separations = _parse_number_list(args.branch_separations, float)
    custom_widths = _parse_number_list(args.branch_widths, float)
    custom_gaps = _parse_number_list(args.bearing_gap_thresholds, float)
    bridge_counts = _parse_number_list(args.bridge_counts, int) or GRADUAL_BRIDGE_COUNTS
    structures = _parse_structures(args.neighbor_structures)
    configurations: list[SensitivityConfiguration] = []
    if "two-branch" in requested:
        separations = custom_separations or TWO_BRANCH_SEPARATIONS_DEG
        widths = custom_widths or TWO_BRANCH_WIDTHS_DEG
        gaps = custom_gaps or BEARING_GAP_THRESHOLDS_DEG
        for separation in separations:
            for gap in gaps:
                for width in widths:
                    # Two equal angular intervals overlap when center separation
                    # is no greater than their shared full width. Such semantic
                    # labels are not uniquely separable and are reported apart.
                    ambiguous = separation <= width
                    for bridge_count in (0, 1):
                        for structure in structures:
                            configurations.append(
                                SensitivityConfiguration(
                                    experiment="two-branch",
                                    branch_separation_deg=float(separation),
                                    branch_width_deg=float(width),
                                    bearing_gap_threshold_deg=float(gap),
                                    bridge_robot_count=bridge_count,
                                    neighbor_structure=structure,
                                    ambiguous_configuration=ambiguous,
                                )
                            )
    if "wide-one-branch" in requested:
        widths = custom_widths or WIDE_BRANCH_WIDTHS_DEG
        gaps = custom_gaps or BEARING_GAP_THRESHOLDS_DEG
        for width in widths:
            for gap in gaps:
                for structure in structures:
                    configurations.append(
                        SensitivityConfiguration(
                            experiment="wide-one-branch",
                            branch_separation_deg=None,
                            branch_width_deg=float(width),
                            bearing_gap_threshold_deg=float(gap),
                            bridge_robot_count=0,
                            neighbor_structure=structure,
                        )
                    )
    if "gradual-chain" in requested:
        separations = custom_separations or GRADUAL_CHAIN_SEPARATIONS_DEG
        gaps = custom_gaps or GRADUAL_CHAIN_GAPS_DEG
        for separation in separations:
            for bridge_count in bridge_counts:
                for gap in gaps:
                    configurations.append(
                        SensitivityConfiguration(
                            experiment="gradual-chain",
                            branch_separation_deg=float(separation),
                            branch_width_deg=2.0,
                            bearing_gap_threshold_deg=float(gap),
                            bridge_robot_count=int(bridge_count),
                            neighbor_structure="gradual-chain",
                        )
                    )
    return tuple(configurations)


def run_sensitivity(args: argparse.Namespace) -> tuple[Path, Path, list[dict[str, object]]]:
    """Execute selected experiments and write raw plus aggregate CSV files."""
    configurations = _experiment_configurations(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.raw_csv or output_dir / "relative_swarm_sensitivity_raw.csv"
    summary_path = args.summary_csv or output_dir / "relative_swarm_sensitivity_summary.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    accumulators: dict[SensitivityConfiguration, SummaryAccumulator] = defaultdict(
        SummaryAccumulator
    )
    total_rows = len(configurations) * args.runs
    completed_rows = 0
    evaluated_rows = 0
    start_time = perf_counter()
    with raw_path.open("w", newline="", encoding="utf-8") as raw_stream:
        writer = csv.DictWriter(raw_stream, fieldnames=RAW_FIELDNAMES)
        writer.writeheader()
        for configuration in configurations:
            accumulator = accumulators[configuration]
            for seed_offset in range(args.runs):
                seed = args.seed_start + seed_offset
                if configuration.ambiguous_configuration:
                    writer.writerow(
                        _ambiguous_raw_row(
                            configuration,
                            seed,
                            args.range_noise_std,
                            args.bearing_noise_std,
                        )
                    )
                    accumulator.add_ambiguous()
                else:
                    rng = np.random.default_rng(seed)
                    if configuration.experiment == "two-branch":
                        sequence = _build_two_branch_sequence(
                            configuration,
                            rng,
                            args.range_noise_std,
                            args.bearing_noise_std,
                        )
                    elif configuration.experiment == "wide-one-branch":
                        sequence = _build_wide_branch_sequence(
                            configuration,
                            rng,
                            args.range_noise_std,
                            args.bearing_noise_std,
                        )
                    else:
                        sequence = _build_gradual_chain_sequence(
                            configuration,
                            rng,
                            args.range_noise_std,
                            args.bearing_noise_std,
                        )
                    result = _run_validator(
                        sequence.frames, configuration.bearing_gap_threshold_deg
                    )
                    raw_row, values = _evaluated_raw_row(
                        configuration=configuration,
                        seed=seed,
                        sequence=sequence,
                        result=result,
                        range_noise_std_m=args.range_noise_std,
                        bearing_noise_std_deg=args.bearing_noise_std,
                    )
                    writer.writerow(raw_row)
                    accumulator.add_result(**values)
                    evaluated_rows += 1
                completed_rows += 1
                if completed_rows % 5000 == 0 or completed_rows == total_rows:
                    elapsed = perf_counter() - start_time
                    print(
                        f"progress={completed_rows}/{total_rows}, "
                        f"evaluated={evaluated_rows}, elapsed={elapsed:.1f}s"
                    )
    summary_rows = [
        accumulators[configuration].as_row(configuration)
        for configuration in configurations
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as summary_stream:
        writer = csv.DictWriter(summary_stream, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(
        f"completed raw_rows={total_rows}, evaluated_runs={evaluated_rows}, "
        f"summary_rows={len(summary_rows)}, elapsed={perf_counter() - start_time:.1f}s"
    )
    return raw_path, summary_path, summary_rows


def _weighted_metric_grid(
    summary_rows: Sequence[dict[str, object]],
    *,
    experiment: str,
    x_key: str,
    y_key: str,
    numerator_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate summary counters into a weighted 2D metric grid."""
    rows = [
        row
        for row in summary_rows
        if row["experiment"] == experiment
        and not row["ambiguous_configuration"]
        and int(row["evaluated_run_count"]) > 0
    ]
    if not rows:
        return np.array([]), np.array([]), np.empty((0, 0))
    x_values = np.array(sorted({float(row[x_key]) for row in rows}))
    y_values = np.array(sorted({float(row[y_key]) for row in rows}))
    numerator = np.zeros((len(y_values), len(x_values)), dtype=float)
    denominator = np.zeros_like(numerator)
    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}
    for row in rows:
        x_position = x_index[float(row[x_key])]
        y_position = y_index[float(row[y_key])]
        numerator[y_position, x_position] += float(row[numerator_key])
        denominator[y_position, x_position] += int(row["evaluated_run_count"])
    grid = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )
    return x_values, y_values, grid


def _save_heatmap(
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    grid: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    colorbar_label: str,
    output_path: Path,
) -> None:
    """Save a bounded [0,1] sensitivity heatmap with explicit axis units."""
    if not grid.size:
        return
    figure, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(
        np.ma.masked_invalid(grid),
        origin="lower",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    axis.set_xticks(np.arange(len(x_values)), [f"{value:g}" for value in x_values])
    axis.set_yticks(np.arange(len(y_values)), [f"{value:g}" for value in y_values])
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def create_plots(
    summary_rows: Sequence[dict[str, object]], output_dir: Path
) -> tuple[Path, ...]:
    """Create the requested heatmaps and threshold trade-off diagnostics."""
    created: list[Path] = []
    separations, gaps, accuracy_grid = _weighted_metric_grid(
        summary_rows,
        experiment="two-branch",
        x_key="branch_separation_deg",
        y_key="bearing_gap_threshold_deg",
        numerator_key="exact_success_count",
    )
    if accuracy_grid.size:
        path = output_dir / "two_branch_exact_accuracy_heatmap.png"
        _save_heatmap(
            x_values=separations,
            y_values=gaps,
            grid=accuracy_grid,
            title="Two-Branch exact count accuracy",
            x_label="Semantic Branch center separation [deg]",
            y_label="Maximum neighbor bearing gap [deg]",
            colorbar_label="Exact Branch count accuracy",
            output_path=path,
        )
        created.append(path)
    widths, wide_gaps, over_grid = _weighted_metric_grid(
        summary_rows,
        experiment="wide-one-branch",
        x_key="branch_width_deg",
        y_key="bearing_gap_threshold_deg",
        numerator_key="over_split_count",
    )
    if over_grid.size:
        path = output_dir / "wide_branch_over_split_heatmap.png"
        _save_heatmap(
            x_values=widths,
            y_values=wide_gaps,
            grid=over_grid,
            title="Wide one-Branch over-split rate",
            x_label="Semantic Branch angular width [deg]",
            y_label="Maximum neighbor bearing gap [deg]",
            colorbar_label="Over-split rate",
            output_path=path,
        )
        created.append(path)
    if accuracy_grid.size and over_grid.size:
        common_gaps = sorted(set(gaps) & set(wide_gaps))
        merge_rates = []
        over_rates = []
        two_rows = [
            row
            for row in summary_rows
            if row["experiment"] == "two-branch"
            and not row["ambiguous_configuration"]
        ]
        wide_rows = [
            row for row in summary_rows if row["experiment"] == "wide-one-branch"
        ]
        for gap in common_gaps:
            selected_two = [
                row for row in two_rows if float(row["bearing_gap_threshold_deg"]) == gap
            ]
            selected_wide = [
                row for row in wide_rows if float(row["bearing_gap_threshold_deg"]) == gap
            ]
            merge_rates.append(
                sum(int(row["merge_count"]) for row in selected_two)
                / sum(int(row["evaluated_run_count"]) for row in selected_two)
            )
            over_rates.append(
                sum(int(row["over_split_count"]) for row in selected_wide)
                / sum(int(row["evaluated_run_count"]) for row in selected_wide)
            )
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(common_gaps, merge_rates, marker="o", label="Two-Branch merge rate")
        axis.plot(common_gaps, over_rates, marker="s", label="Wide-Branch over-split rate")
        axis.set_xlabel("Maximum neighbor bearing gap [deg]")
        axis.set_ylabel("Error rate")
        axis.set_ylim(-0.02, 1.02)
        axis.set_title("Merge vs over-split threshold trade-off")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        path = output_dir / "merge_over_split_tradeoff.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        created.append(path)

        minimum_separations = []
        for gap_index, _gap in enumerate(gaps):
            qualifying = separations[accuracy_grid[gap_index] >= 0.95]
            minimum_separations.append(
                float(np.min(qualifying)) if qualifying.size else float("nan")
            )
        figure, axis = plt.subplots(figsize=(8, 5))
        values = np.asarray(minimum_separations, dtype=float)
        finite = np.isfinite(values)
        axis.plot(gaps[finite], values[finite], marker="o")
        axis.set_xlabel("Maximum neighbor bearing gap [deg]")
        axis.set_ylabel("Minimum separation reaching >=95% accuracy [deg]")
        axis.set_title("Minimum reliably separable Branch angle")
        axis.grid(alpha=0.3)
        figure.tight_layout()
        path = output_dir / "minimum_separable_angle.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        created.append(path)
    return tuple(created)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure failure boundaries of the current fixed circular neighbor "
            "bearing-gap rule without changing validator behavior."
        )
    )
    parser.add_argument(
        "--experiment",
        choices=("two-branch", "wide-one-branch", "gradual-chain", "all"),
        default="all",
        help="experiment family to execute (default: all)",
    )
    parser.add_argument(
        "--runs", type=int, default=50,
        help="number of consecutive random seeds per parameter combination (default: 50)",
    )
    parser.add_argument(
        "--seed-start", type=int, default=1000,
        help="first random seed; run k uses seed-start+k",
    )
    parser.add_argument(
        "--branch-separations", default=None,
        help=(
            "comma-separated Branch center separations [deg], e.g. 20,30,60; "
            "experiment-specific defaults are used when omitted"
        ),
    )
    parser.add_argument(
        "--branch-widths", default=None,
        help=(
            "comma-separated full angular widths [deg], e.g. 5,20,40; "
            "experiment-specific defaults are used when omitted"
        ),
    )
    parser.add_argument(
        "--bearing-gap-thresholds", default=None,
        help=(
            "comma-separated maximum neighbor bearing gaps [deg], e.g. 10,20,30"
        ),
    )
    parser.add_argument(
        "--bridge-counts", default=None,
        help="comma-separated gradual-chain intermediate robot counts, e.g. 0,1,3,5",
    )
    parser.add_argument(
        "--neighbor-structures", default="clique,chain",
        help="comma-separated intra-Branch topologies: clique,chain",
    )
    parser.add_argument(
        "--range-noise-std", type=float, default=0.01,
        help="Anchor-range Gaussian noise standard deviation [m]",
    )
    parser.add_argument(
        "--bearing-noise-std", type=float, default=0.4,
        help="Anchor-bearing Gaussian noise standard deviation [deg]",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("junction_detection/swarm/output/relative_swarm_sensitivity"),
        help="directory for default CSV and PNG outputs",
    )
    parser.add_argument(
        "--raw-csv", type=Path, default=None,
        help="optional raw per-run CSV path",
    )
    parser.add_argument(
        "--summary-csv", type=Path, default=None,
        help="optional parameter-aggregate CSV path",
    )
    parser.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.range_noise_std < 0.0 or args.bearing_noise_std < 0.0:
        parser.error("noise standard deviations must be non-negative")
    return args


def main() -> None:
    """Run selected sensitivity experiments and report saved artifacts."""
    args = _parse_args()
    configurations = _experiment_configurations(args)
    total_rows = len(configurations) * args.runs
    ambiguous_rows = sum(
        configuration.ambiguous_configuration for configuration in configurations
    ) * args.runs
    print(
        f"experiment={args.experiment}, configurations={len(configurations)}, "
        f"raw_rows={total_rows}, planned_validator_runs={total_rows - ambiguous_rows}"
    )
    print(
        "Validator input remains: timestamp, robot ID, Anchor-relative "
        "range/bearing, neighbor IDs"
    )
    raw_path, summary_path, summary_rows = run_sensitivity(args)
    print(f"raw_csv={raw_path}")
    print(f"summary_csv={summary_path}")
    if not args.no_plots:
        for path in create_plots(summary_rows, args.output_dir):
            print(f"plot={path}")


if __name__ == "__main__":
    main()
