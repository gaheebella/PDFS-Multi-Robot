"""Evaluate the unchanged stable-motion tangent on inlet/turning SPH flows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from junction_detection.integration.trajectory_stability_diagnostics import (
    angular_error_deg,
    annotate_trajectory_stability,
    circular_statistics_deg,
    equal_robot_robust_tangent,
    mouth_pca_tangent,
    normalize_angle_deg,
    signed_angle_delta_deg,
)
from pygame_simulator.general_junction_sph_benchmark import SphParameters
from pygame_simulator.general_junction_sph_inlet_benchmark import (
    InletBenchmarkGeometry,
    create_inlet_benchmark_geometries,
    run_inlet_geometry_synthetic_test,
    simulate_inlet_sph_case,
)


def _short_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _case_key(case: InletBenchmarkGeometry) -> str:
    return case.geometry.case_id


def _attach_causal_pca(
    segments: list[dict[str, Any]],
    crossings: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Lock the same first-cohort mouth PCA used by the radial benchmark."""
    crossing_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    motion_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in crossings:
        crossing_groups[(str(row["case_id"]), str(row["branch_id"]))].append(row)
    for row in segments:
        motion_groups[(str(row["case_id"]), str(row["branch_id"]))].append(row)
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in motion_groups.items():
        mouth_rows = sorted(
            crossing_groups.get(key, []),
            key=lambda row: (int(row["frame"]), int(row["robot_id"])),
        )[:12]
        points = [
            (float(row["crossing_x"]), float(row["crossing_y"]))
            for row in mouth_rows
        ]
        outward_angles = [
            normalize_angle_deg(math.degrees(math.atan2(
                float(row["outward_dy"]), float(row["outward_dx"])
            )))
            for row in mouth_rows
            if math.hypot(
                float(row["outward_dx"]), float(row["outward_dy"])
            ) > 1.0e-12
        ]
        tangent = (
            mouth_pca_tangent(points, outward_angles)
            if len(points) >= 4 and outward_angles
            else None
        )
        record = {
            "tangent": tangent,
            "sample_count": len(mouth_rows),
            "lock_frame": max(
                (int(row["frame"]) for row in mouth_rows), default=""
            ),
        }
        records[key] = record
        for row in rows:
            row["pca_tangent_deg"] = "" if tangent is None else tangent
            row["pca_lock_sample_count"] = record["sample_count"]
            row["pca_lock_frame"] = record["lock_frame"]
    return records


def _phase_summary(
    annotated: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in annotated:
        groups[(str(row["case_id"]), str(row["branch_id"]))].append(row)
    output = []
    for (case_id, branch_id), rows in sorted(groups.items()):
        selections = {
            phase: [row for row in rows if row["lifecycle_phase"] == phase]
            for phase in (
                "JUNCTION_TURNING",
                "BRANCH_STRAIGHTENING",
                "BRANCH_FLOW",
            )
        }
        selections["STABLE_GATE"] = [
            row for row in rows if bool(row["stable_candidate"])
        ]
        for phase, selected in selections.items():
            if not selected:
                continue
            deltas = [
                float(row["temporal_delta_deg"])
                for row in selected
                if math.isfinite(float(row["temporal_delta_deg"]))
            ]
            _, dispersion, resultant = circular_statistics_deg([
                float(row["motion_angle_deg"]) for row in selected
            ])
            gt_errors = [
                signed_angle_delta_deg(
                    float(row["motion_angle_deg"]),
                    float(row["gt_branch_angle_deg"]),
                )
                for row in selected
            ]
            output.append({
                "case_id": case_id,
                "branch_id": branch_id,
                "turn_severity": selected[0]["turn_severity"],
                "phase": phase,
                "segment_count": len(selected),
                "robot_count": len({int(row["robot_id"]) for row in selected}),
                "median_temporal_delta_deg": (
                    float(np.median(deltas)) if deltas else float("nan")
                ),
                "circular_dispersion_deg": dispersion,
                "resultant_length": resultant,
                "median_persistence_updates": float(np.median([
                    float(row["persistence_updates"]) for row in selected
                ])),
                "mean_signed_gt_error_deg": float(np.mean(gt_errors)),
                "mean_absolute_gt_error_deg": float(np.mean(np.abs(gt_errors))),
                "p90_absolute_gt_error_deg": float(np.percentile(np.abs(gt_errors), 90)),
                "stable_candidate_ratio": float(np.mean([
                    bool(row["stable_candidate"]) for row in selected
                ])),
            })
    return output


def _bias_statistics(errors: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(errors, dtype=float)
    if values.size == 0:
        return {
            "positive_error_count": 0,
            "negative_error_count": 0,
            "zero_error_count": 0,
            "individual_signed_mean_deg": "",
            "individual_signed_median_deg": "",
            "individual_mae_deg": "",
            "individual_p90_deg": "",
            "signed_error_skew": "",
        }
    mean = float(np.mean(values))
    std = float(np.std(values))
    skew = (
        float(np.mean(((values - mean) / std) ** 3))
        if std > 1.0e-12 else 0.0
    )
    return {
        "positive_error_count": int(np.sum(values > 1.0e-9)),
        "negative_error_count": int(np.sum(values < -1.0e-9)),
        "zero_error_count": int(np.sum(np.abs(values) <= 1.0e-9)),
        "individual_signed_mean_deg": mean,
        "individual_signed_median_deg": float(np.median(values)),
        "individual_mae_deg": float(np.mean(np.abs(values))),
        "individual_p90_deg": float(np.percentile(np.abs(values), 90)),
        "signed_error_skew": skew,
    }


def _comparison_rows(
    cases: Sequence[InletBenchmarkGeometry],
    annotated: Sequence[Mapping[str, Any]],
    pca_records: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in annotated:
        groups[(str(row["case_id"]), str(row["branch_id"]))].append(row)
    comparisons = []
    biases = []
    for case in cases:
        geometry = case.geometry
        for branch in case.outgoing_branches:
            key = (geometry.case_id, branch.branch_id)
            rows = groups.get(key, [])
            stable_rows = [row for row in rows if bool(row["stable_candidate"])]
            stable = equal_robot_robust_tangent(rows) if rows else None
            pca_record = pca_records.get(key, {})
            pca = pca_record.get("tangent")
            individual_errors = [
                signed_angle_delta_deg(
                    float(row["motion_angle_deg"]), branch.angle_deg
                )
                for row in stable_rows
            ]
            bias = _bias_statistics(individual_errors)
            stable_signed = (
                "" if stable is None
                else signed_angle_delta_deg(stable, branch.angle_deg)
            )
            pca_signed = (
                "" if pca is None
                else signed_angle_delta_deg(pca, branch.angle_deg)
            )
            comparison = {
                "case_id": geometry.case_id,
                "seed": geometry.seed,
                "topology": geometry.topology,
                "rotation_deg": geometry.rotation_deg,
                "incoming_travel_angle_deg": case.incoming_travel_angle_deg,
                "branch_id": branch.branch_id,
                "gt_tangent_deg": branch.angle_deg,
                "turn_angle_deg": case.turn_angle_deg(branch),
                "turn_severity": case.turn_severity(branch),
                "length_group": geometry.length_group,
                "corridor_width_group": geometry.width_group,
                "branch_length": branch.length,
                "corridor_width": branch.width,
                "pca_tangent_deg": "" if pca is None else pca,
                "pca_signed_error_deg": pca_signed,
                "pca_error_deg": (
                    "" if pca is None
                    else angular_error_deg(pca, branch.angle_deg)
                ),
                "pca_lock_frame": pca_record.get("lock_frame", ""),
                "pca_lock_sample_count": pca_record.get("sample_count", 0),
                "stable_tangent_deg": "" if stable is None else stable,
                "stable_signed_error_deg": stable_signed,
                "stable_error_deg": (
                    "" if stable is None
                    else angular_error_deg(stable, branch.angle_deg)
                ),
                # Evaluation-only projection: the transverse miss implied by
                # an angular error at the physical end of this corridor.
                "pca_predicted_lateral_drift": (
                    "" if pca_signed == "" else branch.length * math.sin(
                        math.radians(float(pca_signed))
                    )
                ),
                "stable_predicted_lateral_drift": (
                    "" if stable_signed == "" else branch.length * math.sin(
                        math.radians(float(stable_signed))
                    )
                ),
                "stable_robot_count": len({
                    int(row["robot_id"]) for row in stable_rows
                }),
                "stable_segment_count": len(stable_rows),
                "total_segment_count": len(rows),
                "estimator_available": stable is not None,
                **bias,
            }
            comparisons.append(comparison)
            biases.append({
                "case_id": geometry.case_id,
                "branch_id": branch.branch_id,
                "turn_severity": case.turn_severity(branch),
                "aggregate_signed_error_deg": stable_signed,
                **bias,
            })
    return comparisons, biases


def _aggregate_errors(
    rows: Sequence[Mapping[str, Any]],
    dimension: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[dimension])].append(row)
    output = []
    for value, group in sorted(groups.items()):
        for estimator, column in (
            ("pca", "pca_error_deg"),
            ("stable", "stable_error_deg"),
        ):
            errors = [
                float(row[column]) for row in group if row[column] != ""
            ]
            if not errors:
                continue
            output.append({
                "group_dimension": dimension,
                "group_value": value,
                "estimator": estimator,
                "count": len(errors),
                "mean_error_deg": float(np.mean(errors)),
                "median_error_deg": float(np.median(errors)),
                "p90_error_deg": float(np.percentile(errors, 90)),
                "max_error_deg": float(np.max(errors)),
            })
    return output


def _generalization_rows(
    comparisons: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dimensions = (
        "gt_tangent_deg",
        "length_group",
        "corridor_width_group",
        "topology",
        "seed",
        "rotation_deg",
        "turn_severity",
    )
    all_rows = []
    for dimension in dimensions:
        all_rows.extend(_aggregate_errors(comparisons, dimension))
    severity_rows = _aggregate_errors(comparisons, "turn_severity")
    return all_rows, severity_rows


def _group_plot(
    directory: Path,
    comparisons: Sequence[Mapping[str, Any]],
    key: str,
    filename: str,
    label: str,
) -> None:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in comparisons:
        groups[str(row[key])].append(row)
    labels = list(groups)
    if key in {"gt_tangent_deg", "rotation_deg", "turn_angle_deg"}:
        labels.sort(key=float)
    else:
        labels.sort()
    pca = [
        np.mean([
            float(row["pca_error_deg"])
            for row in groups[value] if row["pca_error_deg"] != ""
        ]) for value in labels
    ]
    stable = [
        np.mean([
            float(row["stable_error_deg"])
            for row in groups[value] if row["stable_error_deg"] != ""
        ]) for value in labels
    ]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.plot(x, pca, "o-", label="mouth PCA")
    axis.plot(x, stable, "o-", label="stable motion")
    axis.set_xticks(x, labels, rotation=45, ha="right")
    axis.set(xlabel=label, ylabel="mean absolute error [deg]", title=f"Error vs {label}")
    axis.legend()
    figure.savefig(directory / filename, dpi=160)
    plt.close(figure)


def _save_plots(
    directory: Path,
    annotated: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    biases: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
) -> None:
    available = [row for row in comparisons if row["estimator_available"]]
    example = max(available, key=lambda row: float(row["turn_angle_deg"]))
    key = (str(example["case_id"]), str(example["branch_id"]))
    rows = [
        row for row in annotated
        if (str(row["case_id"]), str(row["branch_id"])) == key
    ]
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    example_robot_ids = sorted({int(row["robot_id"]) for row in rows})[:12]
    case_traces = [
        row for row in traces
        if str(row["case_id"]) == key[0]
        and int(row["robot_id"]) in example_robot_ids
    ]
    for robot_id in example_robot_ids:
        robot_rows = [
            row for row in case_traces if int(row["robot_id"]) == robot_id
        ]
        axis.plot(
            [float(row["x"]) for row in robot_rows],
            [float(row["y"]) for row in robot_rows],
            alpha=0.55,
        )
    axis.set_aspect("equal")
    axis.set(title=f"Inlet turning trajectories: {key}", xlabel="world x", ylabel="world y")
    figure.savefig(directory / "inlet_turning_stable_trajectory_examples.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(4, 1, figsize=(10, 12), constrained_layout=True)
    axes[0].scatter(
        [float(row["time"]) for row in rows],
        [float(row["motion_angle_deg"]) for row in rows],
        c=[bool(row["stable_candidate"]) for row in rows],
        s=7, cmap="coolwarm", alpha=0.55,
    )
    axes[0].axhline(float(example["gt_tangent_deg"]), color="black", linestyle="--")
    axes[0].set(ylabel="motion angle [deg]")
    axes[1].scatter(
        [float(row["time"]) for row in rows],
        [float(row["cohort_dispersion_deg"]) for row in rows],
        s=6, alpha=0.4,
    )
    axes[1].set(ylabel="cohort dispersion [deg]")
    axes[1].scatter(
        [float(row["time"]) for row in rows],
        [float(row["cohort_resultant_length"]) for row in rows],
        s=6, alpha=0.35, color="tab:orange", label="resultant",
    )
    axes[1].legend(loc="lower right")
    axes[2].scatter(
        [float(row["time"]) for row in rows],
        [float(row["persistence_updates"]) for row in rows],
        s=6, alpha=0.4,
    )
    axes[2].set(ylabel="persistence [updates]")
    axes[3].scatter(
        [float(row["time"]) for row in rows],
        [angular_error_deg(
            float(row["motion_angle_deg"]), float(row["gt_branch_angle_deg"])
        ) for row in rows],
        c=[bool(row["stable_candidate"]) for row in rows],
        s=7, cmap="coolwarm", alpha=0.55,
    )
    axes[3].set(xlabel="time [s]", ylabel="GT abs. error [deg]")
    figure.savefig(directory / "turning_to_stable_direction_timeline.png", dpi=160)
    plt.close(figure)

    pca = [float(row["pca_error_deg"]) for row in available if row["pca_error_deg"] != ""]
    stable = [float(row["stable_error_deg"]) for row in available]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.boxplot([pca, stable], tick_labels=["mouth PCA", "stable motion"])
    axis.set(title="Inlet SPH tangent error", ylabel="absolute error [deg]")
    figure.savefig(directory / "pca_vs_stable_error_distribution.png", dpi=160)
    plt.close(figure)

    _group_plot(directory, available, "turn_severity", "error_vs_turn_severity.png", "turn severity")
    _group_plot(directory, available, "gt_tangent_deg", "error_vs_branch_angle.png", "Branch angle")
    _group_plot(directory, available, "corridor_width_group", "error_vs_branch_width.png", "width group")
    _group_plot(directory, available, "length_group", "error_vs_branch_length.png", "length group")
    _group_plot(directory, available, "rotation_deg", "rotation_invariance.png", "rotation [deg]")

    labels = [f"{row['case_id']}:{row['branch_id']}" for row in biases]
    positive = [int(row["positive_error_count"]) for row in biases]
    negative = [-int(row["negative_error_count"]) for row in biases]
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    x = np.arange(len(labels))
    axis.bar(x, positive, label="positive bias")
    axis.bar(x, negative, label="negative bias")
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(x, labels, rotation=90, fontsize=6)
    axis.set(ylabel="stable sample count", title="Stable signed-bias balance")
    axis.legend()
    figure.savefig(directory / "stable_signed_bias_balance.png", dpi=160)
    plt.close(figure)

    stress = [
        row for row in comparisons
        if row["case_id"] == "inlet_stress_long_sharp"
        and row["pca_predicted_lateral_drift"] != ""
        and row["stable_predicted_lateral_drift"] != ""
    ]
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    labels = [row["branch_id"] for row in stress]
    x = np.arange(len(labels))
    axis.bar(x - 0.18, [
        abs(float(row["pca_predicted_lateral_drift"])) for row in stress
    ], width=0.36, label="PCA")
    axis.bar(x + 0.18, [
        abs(float(row["stable_predicted_lateral_drift"])) for row in stress
    ], width=0.36, label="stable")
    axis.set_xticks(x, labels)
    axis.set(ylabel="predicted lateral drift [px]", title="Long-Branch inlet stress abstraction")
    axis.legend()
    figure.savefig(directory / "stress_case_long_branch_comparison.png", dpi=160)
    plt.close(figure)


def run_inlet_benchmark(
    output_dir: Path,
    seed: int,
    quick: bool = False,
) -> dict[str, Any]:
    """Execute the balanced physical inlet benchmark and unchanged estimator."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    cases, rejected = create_inlet_benchmark_geometries(seed)
    if quick:
        cases = cases[:2]
    parameters = (
        SphParameters(steps=1600, particle_count=64)
        if quick else SphParameters(steps=1600)
    )
    case_rows: list[dict[str, Any]] = list(rejected)
    segments: list[dict[str, Any]] = []
    crossings: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        print(
            f"[SPH inlet benchmark] {index}/{len(cases)} "
            f"{_case_key(case)}",
            flush=True,
        )
        result = simulate_inlet_sph_case(case, parameters)
        case_rows.append(result.case_row)
        segments.extend(result.segment_rows)
        crossings.extend(result.crossing_rows)
        traces.extend(result.trace_rows)
    pca_records = _attach_causal_pca(segments, crossings)
    # These are exactly the pre-existing features, percentile gate, and
    # equal-robot component-median estimator.  No inlet-specific tuning occurs.
    annotated = annotate_trajectory_stability(segments)
    phase_summary = _phase_summary(annotated)
    comparisons, biases = _comparison_rows(cases, annotated, pca_records)
    generalization, severity = _generalization_rows(comparisons)
    _write_rows(directory / "sph_inlet_benchmark_cases.csv", case_rows)
    _write_rows(directory / "sph_inlet_trajectory_segments.csv", annotated)
    _write_rows(directory / "sph_inlet_mouth_crossings.csv", crossings)
    _write_rows(directory / "sph_inlet_turning_stability_summary.csv", phase_summary)
    _write_rows(directory / "sph_inlet_tangent_comparison.csv", comparisons)
    _write_rows(directory / "sph_inlet_error_by_turn_severity.csv", severity)
    _write_rows(directory / "sph_inlet_generalization_summary.csv", generalization)
    _write_rows(directory / "sph_inlet_signed_bias_summary.csv", biases)
    _save_plots(directory, annotated, comparisons, biases, traces)

    available = [row for row in comparisons if row["estimator_available"]]
    pca_errors = [float(row["pca_error_deg"]) for row in available if row["pca_error_deg"] != ""]
    stable_errors = [float(row["stable_error_deg"]) for row in available]
    expected_branches = len(comparisons)
    coverage = len(available) / max(expected_branches, 1)
    phase_by_name: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in phase_summary:
        phase_by_name[str(row["phase"])].append(row)
    turning_dispersion = float(np.median([
        float(row["circular_dispersion_deg"])
        for row in phase_by_name["JUNCTION_TURNING"]
    ]))
    stable_dispersion = float(np.median([
        float(row["circular_dispersion_deg"])
        for row in phase_by_name["STABLE_GATE"]
    ]))
    turning_resultant = float(np.median([
        float(row["resultant_length"])
        for row in phase_by_name["JUNCTION_TURNING"]
    ]))
    stable_resultant = float(np.median([
        float(row["resultant_length"])
        for row in phase_by_name["STABLE_GATE"]
    ]))
    severity_groups = defaultdict(list)
    for row in available:
        severity_groups[str(row["turn_severity"])].append(
            float(row["stable_error_deg"])
        )
    # Case A requires generalization within every requested physical grouping,
    # not merely a favorable global mean. This is evaluation logic only and
    # does not alter the stable gate or tangent estimator.
    systematic_failure_groups = []
    for dimension in (
        "turn_severity", "corridor_width_group", "length_group", "topology"
    ):
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in available:
            grouped[str(row[dimension])].append(row)
        for group_value, group_rows in grouped.items():
            paired = [
                row for row in group_rows if row["pca_error_deg"] != ""
            ]
            if paired and np.mean([
                float(row["stable_error_deg"]) for row in paired
            ]) > np.mean([
                float(row["pca_error_deg"]) for row in paired
            ]):
                systematic_failure_groups.append(
                    f"{dimension}={group_value}"
                )
    stress_rows = [
        row for row in comparisons
        if row["case_id"] == "inlet_stress_long_sharp"
    ]
    stress_available = [row for row in stress_rows if row["estimator_available"]]
    stress_failure = (
        len(stress_available) != len(stress_rows)
        or any(
            row["pca_error_deg"] != ""
            and float(row["stable_error_deg"]) > float(row["pca_error_deg"])
            for row in stress_available
        )
    )
    stable_better = (
        stable_errors
        and pca_errors
        and float(np.mean(stable_errors)) < float(np.mean(pca_errors))
    )
    signal = (
        stable_dispersion < turning_dispersion
        and stable_resultant > turning_resultant
    )
    classification = (
        "A"
        if coverage >= 0.95 and signal and stable_better
        and not systematic_failure_groups and not stress_failure
        else ("B" if signal and stable_better else "C")
    )
    rotation_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in available:
        if str(row["case_id"]).startswith("inlet_rotated_"):
            rotation_groups[(str(row["seed"]), str(row["branch_id"]))].append(
                float(row["stable_error_deg"])
            )
    rotation_span = max(
        (max(values) - min(values) for values in rotation_groups.values() if len(values) > 1),
        default=float("nan"),
    )
    summary = {
        "case": classification,
        "physical_case_count": len(cases),
        "expected_outgoing_branch_count": expected_branches,
        "estimated_branch_count": len(available),
        "branch_coverage_ratio": coverage,
        "trajectory_segment_count": len(annotated),
        "turning_median_dispersion_deg": turning_dispersion,
        "stable_median_dispersion_deg": stable_dispersion,
        "turning_median_resultant": turning_resultant,
        "stable_median_resultant": stable_resultant,
        "pca_mean_error_deg": float(np.mean(pca_errors)),
        "pca_median_error_deg": float(np.median(pca_errors)),
        "pca_p90_error_deg": float(np.percentile(pca_errors, 90)),
        "pca_max_error_deg": float(np.max(pca_errors)),
        "stable_mean_error_deg": float(np.mean(stable_errors)),
        "stable_median_error_deg": float(np.median(stable_errors)),
        "stable_p90_error_deg": float(np.percentile(stable_errors, 90)),
        "stable_max_error_deg": float(np.max(stable_errors)),
        "rotation_max_stable_error_span_deg": rotation_span,
        "systematic_failure_groups": systematic_failure_groups,
        "stress_branch_coverage_ratio": (
            len(stress_available) / max(len(stress_rows), 1)
        ),
        "stress_condition_failure": stress_failure,
        "benchmark_boundary": (
            "production-family SPH physics and inlet turning; no DFS/Guard/Shepherd/Handoff"
        ),
    }
    (directory / "sph_inlet_benchmark_result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            f"/tmp/pdfs_general_sph_inlet_tangent_{_short_head()}"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--synthetic-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.synthetic_test:
        run_inlet_geometry_synthetic_test()
        print("general SPH inlet geometry synthetic test: PASS")
    summary = run_inlet_benchmark(args.output_dir, args.seed, args.quick)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
