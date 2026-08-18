"""Run and analyze the geometry-general physical SPH Branch benchmark."""

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
from pygame_simulator.general_junction_sph_benchmark import (
    JunctionGeometry,
    SphParameters,
    collision_mask_sanity,
    create_physical_benchmark_geometries,
    run_geometry_synthetic_test,
    simulate_sph_case,
)


def _short_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _rank(values: Sequence[float]) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty(len(data), dtype=float)
    ranks[order] = np.arange(len(data), dtype=float)
    return ranks


def _late_stable_tangent(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Use the persistent half of local-gate samples without GT/progress."""
    stable = [row for row in rows if bool(row["stable_candidate"])]
    if not stable:
        return None
    persistence_gate = float(np.median([float(row["persistence_updates"]) for row in stable]))
    late = [row for row in stable if float(row["persistence_updates"]) >= persistence_gate]
    return equal_robot_robust_tangent(late)


def _attach_pca_labels(
    segments: list[dict[str, Any]],
    crossings: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Compute production-equivalent transverse-mouth PCA from actual crossings."""
    crossing_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    motion_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in crossings:
        crossing_groups[(str(row["case_id"]), str(row["branch_id"]))].append(row)
    for row in segments:
        motion_groups[(str(row["case_id"]), str(row["branch_id"]))].append(row)
    estimates: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in motion_groups.items():
        mouth_rows = sorted(
            crossing_groups.get(key, []),
            key=lambda row: (int(row["frame"]), int(row["robot_id"])),
        )
        # Freeze the diagnostic baseline at the first causal cohort.  Twelve
        # unique robots is a sample-count condition, not geometry/time/GT, and
        # prevents later Branch flow from leaking into the mouth-PCA baseline.
        mouth_rows = mouth_rows[:12]
        points = [(float(row["crossing_x"]), float(row["crossing_y"])) for row in mouth_rows]
        outward_angles = [
            normalize_angle_deg(math.degrees(math.atan2(float(row["outward_dy"]), float(row["outward_dx"]))))
            for row in mouth_rows
            if math.hypot(float(row["outward_dx"]), float(row["outward_dy"])) > 1.0e-12
        ]
        if len(points) < 4:
            # Sign may still be obtained from realized Branch motion, but the
            # PCA geometry itself requires multiple independent crossings.
            tangent = None
        else:
            tangent = mouth_pca_tangent(points, outward_angles or [float(row["motion_angle_deg"]) for row in rows])
        estimates[key] = {
            "tangent": tangent,
            "sample_count": len(mouth_rows),
            "lock_frame": max((int(row["frame"]) for row in mouth_rows), default=""),
        }
        for row in rows:
            row["pca_tangent_deg"] = "" if tangent is None else tangent
            row["pca_lock_sample_count"] = len(mouth_rows)
            row["pca_lock_frame"] = estimates[key]["lock_frame"]
    return estimates


def _stability_summary(annotated: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in annotated:
        groups[(str(row["case_id"]), str(row["branch_id"]))].append(row)
    output = []
    for (case_id, branch_id), rows in sorted(groups.items()):
        selections = {
            "TURNING": [row for row in rows if row["lifecycle_phase"] == "JUNCTION_TURNING"],
            "STABLE_GATE": [row for row in rows if bool(row["stable_candidate"])],
        }
        for epoch, selected in selections.items():
            if not selected:
                continue
            deltas = [float(row["temporal_delta_deg"]) for row in selected if math.isfinite(float(row["temporal_delta_deg"]))]
            curvatures = [float(row["curvature_deg_per_unit"]) for row in selected if math.isfinite(float(row["curvature_deg_per_unit"]))]
            _, dispersion, resultant = circular_statistics_deg([float(row["motion_angle_deg"]) for row in selected])
            signed_bias = [signed_angle_delta_deg(float(row["motion_angle_deg"]), float(row["gt_branch_angle_deg"])) for row in selected]
            output.append({
                "case_id": case_id,
                "branch_id": branch_id,
                "evaluation_epoch": epoch,
                "segment_count": len(selected),
                "robot_count": len({int(row["robot_id"]) for row in selected}),
                "median_angular_change_deg": float(np.median(deltas)) if deltas else float("nan"),
                "median_curvature_deg_per_unit": float(np.median(curvatures)) if curvatures else float("nan"),
                "circular_dispersion_deg": dispersion,
                "resultant_length": resultant,
                "median_persistence_updates": float(np.median([float(row["persistence_updates"]) for row in selected])),
                "median_displacement_length": float(np.median([float(row["displacement_length"]) for row in selected])),
                "mean_signed_gt_bias_deg": float(np.mean(signed_bias)),
                "median_signed_gt_bias_deg": float(np.median(signed_bias)),
                "mean_absolute_gt_error_deg": float(np.mean(np.abs(signed_bias))),
                "p90_absolute_gt_error_deg": float(np.percentile(np.abs(signed_bias), 90)),
                "signed_bias_variance_deg2": float(np.var(signed_bias)),
            })
    return output


def _tangent_rows(
    cases: Sequence[Mapping[str, Any]],
    annotated: Sequence[Mapping[str, Any]],
    pca_estimates: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    case_by_id = {str(row["case_id"]): row for row in cases}
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in annotated:
        groups[(str(row["case_id"]), str(row["branch_id"]))].append(row)
    output = []
    for (case_id, branch_id), rows in sorted(groups.items()):
        case = case_by_id[case_id]
        ground_truth = float(rows[0]["gt_branch_angle_deg"])
        pca_record = pca_estimates.get((case_id, branch_id), {})
        pca = pca_record.get("tangent")
        stable = equal_robot_robust_tangent(rows)
        late = _late_stable_tangent(rows)
        output.append({
            "case_id": case_id,
            "topology": case["topology"],
            "seed": case["seed"],
            "branch_id": branch_id,
            "gt_tangent_deg": ground_truth,
            "pca_tangent_deg": "" if pca is None else pca,
            "pca_lock_frame": pca_record.get("lock_frame", ""),
            "pca_lock_sample_count": pca_record.get("sample_count", 0),
            "pca_signed_error_deg": "" if pca is None else signed_angle_delta_deg(pca, ground_truth),
            "pca_error_deg": "" if pca is None else angular_error_deg(pca, ground_truth),
            "stable_tangent_deg": "" if stable is None else stable,
            "stable_signed_error_deg": "" if stable is None else signed_angle_delta_deg(stable, ground_truth),
            "stable_error_deg": "" if stable is None else angular_error_deg(stable, ground_truth),
            "late_stable_tangent_deg": "" if late is None else late,
            "late_stable_error_deg": "" if late is None else angular_error_deg(late, ground_truth),
            "length_group": case["length_group"],
            "corridor_width_group": case["corridor_width_group"],
            "rotation_deg": case["rotation_deg"],
            "source_family": case["source_family"],
            "stable_segment_count": sum(bool(row["stable_candidate"]) for row in rows),
            "segment_count": len(rows),
            "robot_count": len({int(row["robot_id"]) for row in rows}),
        })
    return output


def _generalization_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    dimensions = ("gt_tangent_deg", "length_group", "corridor_width_group", "topology", "seed", "rotation_deg")
    for dimension in dimensions:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[dimension])].append(row)
        for value, group in sorted(groups.items()):
            for estimator, column in (("pca", "pca_error_deg"), ("stable", "stable_error_deg"), ("late_stable", "late_stable_error_deg")):
                errors = [float(row[column]) for row in group if row[column] != ""]
                if errors:
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


def _correlation_rows(tangent_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paired = [row for row in tangent_rows if row["pca_signed_error_deg"] != "" and row["stable_signed_error_deg"] != ""]
    pca = np.asarray([float(row["pca_signed_error_deg"]) for row in paired])
    flow = np.asarray([float(row["stable_signed_error_deg"]) for row in paired])
    pearson = (
        float(np.corrcoef(pca, flow)[0, 1])
        if len(paired) >= 2 and float(np.std(pca)) > 0.0 and float(np.std(flow)) > 0.0
        else float("nan")
    )
    ranked_pca, ranked_flow = _rank(pca), _rank(flow)
    spearman = (
        float(np.corrcoef(ranked_pca, ranked_flow)[0, 1])
        if len(paired) >= 2 and float(np.std(ranked_pca)) > 0.0 and float(np.std(ranked_flow)) > 0.0
        else float("nan")
    )
    same_sign = int(np.sum(np.sign(pca) == np.sign(flow)))
    pca_better = int(np.sum(np.abs(pca) < np.abs(flow)))
    flow_better = int(np.sum(np.abs(flow) < np.abs(pca)))
    summary = {
        "paired_branch_count": len(paired),
        "pearson_signed_error": pearson,
        "spearman_signed_error": spearman,
        "same_sign_count": same_sign,
        "opposite_sign_count": len(paired) - same_sign,
        "pca_lower_absolute_error_count": pca_better,
        "flow_lower_absolute_error_count": flow_better,
        "equal_absolute_error_count": len(paired) - pca_better - flow_better,
    }
    output = []
    for row, pca_error, flow_error in zip(paired, pca, flow):
        output.append({
            "case_id": row["case_id"], "branch_id": row["branch_id"],
            "pca_signed_error_deg": pca_error,
            "stable_signed_error_deg": flow_error,
            "same_error_sign": bool(np.sign(pca_error) == np.sign(flow_error)),
            "lower_absolute_error": "PCA" if abs(pca_error) < abs(flow_error) else ("FLOW" if abs(flow_error) < abs(pca_error) else "EQUAL"),
            **summary,
        })
    return output, summary


def _geometry_plot(directory: Path, geometries: Sequence[JunctionGeometry]) -> None:
    selected = [geometries[0]]
    for predicate in (
        lambda geometry: geometry.topology == "3-way",
        lambda geometry: geometry.topology == "5-way",
        lambda geometry: geometry.rotation_deg == 60.0,
    ):
        match = next((geometry for geometry in geometries if predicate(geometry)), None)
        if match is not None and match not in selected:
            selected.append(match)
    while len(selected) < 4:
        selected.append(geometries[len(selected) % len(geometries)])
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    for axis, geometry in zip(axes.flat, selected):
        circle = plt.Circle(geometry.center, geometry.central_radius, color="0.9")
        axis.add_patch(circle)
        center = geometry.center_array
        for branch in geometry.branches:
            start = center + branch.tangent * (geometry.central_radius * 0.68)
            corners = np.asarray([
                start - branch.normal * branch.width * 0.5,
                start + branch.normal * branch.width * 0.5,
                center + branch.tangent * (geometry.central_radius + branch.length) + branch.normal * branch.width * 0.5,
                center + branch.tangent * (geometry.central_radius + branch.length) - branch.normal * branch.width * 0.5,
            ])
            axis.fill(corners[:, 0], corners[:, 1], color="0.84", edgecolor="0.25")
            axis.arrow(center[0], center[1], branch.tangent[0] * geometry.central_radius, branch.tangent[1] * geometry.central_radius, width=0.25, color="tab:blue")
        axis.set_aspect("equal")
        axis.autoscale_view()
        axis.set_title(geometry.case_id)
    fig.savefig(directory / "sph_geometry_examples.png", dpi=160)
    plt.close(fig)


def _plots(directory: Path, geometries: Sequence[JunctionGeometry], annotated: Sequence[Mapping[str, Any]], tangent_rows: Sequence[Mapping[str, Any]]) -> None:
    _geometry_plot(directory, geometries)
    example_key = (str(annotated[0]["case_id"]), str(annotated[0]["branch_id"]))
    example = [row for row in annotated if (str(row["case_id"]), str(row["branch_id"])) == example_key]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for robot_id in sorted({int(row["robot_id"]) for row in example})[:8]:
        robot_rows = [row for row in example if int(row["robot_id"]) == robot_id]
        axes[0].plot([float(row["progress_fraction"]) for row in robot_rows], [float(row["motion_angle_deg"]) for row in robot_rows], alpha=0.55)
    axes[0].set(title=f"Actual SPH turning/stable: {example_key}", xlabel="GT progress (evaluation only)", ylabel="motion angle [deg]")
    axes[1].scatter([float(row["progress_fraction"]) for row in example], [float(row["stability_score"]) for row in example], c=[bool(row["stable_candidate"]) for row in example], s=8, cmap="coolwarm")
    axes[1].set(xlabel="GT progress (evaluation only)", ylabel="local-only stability score")
    fig.savefig(directory / "sph_turning_vs_stable_examples.png", dpi=160)
    plt.close(fig)

    sampled = annotated[:: max(1, len(annotated) // 8000)]
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.scatter([float(row["progress_fraction"]) for row in sampled], [angular_error_deg(float(row["motion_angle_deg"]), float(row["gt_branch_angle_deg"])) for row in sampled], s=4, alpha=0.2)
    axis.set(title="Actual SPH direction error over progress", xlabel="GT progress (evaluation only)", ylabel="motion error [deg]")
    fig.savefig(directory / "sph_direction_error_over_progress.png", dpi=160)
    plt.close(fig)

    pca = [float(row["pca_error_deg"]) for row in tangent_rows if row["pca_error_deg"] != ""]
    stable = [float(row["stable_error_deg"]) for row in tangent_rows if row["stable_error_deg"] != ""]
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.boxplot([pca, stable], tick_labels=["mouth PCA", "stable motion"])
    axis.set(title="Actual SPH tangent error", ylabel="absolute error [deg]")
    fig.savefig(directory / "sph_pca_vs_stable_error_distribution.png", dpi=160)
    plt.close(fig)

    def grouped(filename: str, key: str, label: str) -> None:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in tangent_rows:
            groups[str(row[key])].append(row)
        labels = list(sorted(groups, key=lambda value: float(value) if key in {"gt_tangent_deg", "rotation_deg"} else value))
        x = np.arange(len(labels))
        pca_mean = [np.mean([float(row["pca_error_deg"]) for row in groups[value] if row["pca_error_deg"] != ""]) for value in labels]
        stable_mean = [np.mean([float(row["stable_error_deg"]) for row in groups[value] if row["stable_error_deg"] != ""]) for value in labels]
        fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        axis.plot(x, pca_mean, "o-", label="mouth PCA")
        axis.plot(x, stable_mean, "o-", label="stable motion")
        axis.set_xticks(x, labels, rotation=45, ha="right")
        axis.set(title=f"Actual SPH error vs {label}", xlabel=label, ylabel="mean absolute error [deg]")
        axis.legend()
        fig.savefig(directory / filename, dpi=160)
        plt.close(fig)

    grouped("sph_error_vs_branch_angle.png", "gt_tangent_deg", "Branch angle")
    grouped("sph_error_vs_length.png", "length_group", "length group")
    grouped("sph_error_vs_width.png", "corridor_width_group", "width group")
    grouped("sph_rotation_invariance.png", "rotation_deg", "rotation [deg]")

    paired = [row for row in tangent_rows if row["pca_signed_error_deg"] != "" and row["stable_signed_error_deg"] != ""]
    fig, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
    axis.scatter([float(row["pca_signed_error_deg"]) for row in paired], [float(row["stable_signed_error_deg"]) for row in paired], alpha=0.65)
    axis.axhline(0.0, color="0.5", linewidth=1)
    axis.axvline(0.0, color="0.5", linewidth=1)
    axis.set(title="PCA vs stable-flow signed error", xlabel="PCA signed error [deg]", ylabel="stable-flow signed error [deg]")
    fig.savefig(directory / "sph_pca_vs_flow_signed_error_scatter.png", dpi=160)
    plt.close(fig)


def run_physical_benchmark(output_dir: Path, seed: int, quick: bool = False) -> dict[str, Any]:
    """Run actual SPH cases, analyze local trajectories, and save artifacts."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    geometries, rejected = create_physical_benchmark_geometries(seed)
    if quick:
        geometries = geometries[:2]
    params = SphParameters(steps=180, particle_count=64) if quick else SphParameters()
    case_rows: list[dict[str, Any]] = list(rejected)
    segments: list[dict[str, Any]] = []
    crossings: list[dict[str, Any]] = []
    for index, geometry in enumerate(geometries, 1):
        print(f"[SPH benchmark] {index}/{len(geometries)} {geometry.case_id}", flush=True)
        result = simulate_sph_case(geometry, params)
        case_rows.append(result.case_row)
        segments.extend(result.segment_rows)
        crossings.extend(result.crossing_rows)
    pca_estimates = _attach_pca_labels(segments, crossings)
    annotated = annotate_trajectory_stability(segments)
    stability = _stability_summary(annotated)
    tangents = _tangent_rows(case_rows, annotated, pca_estimates)
    generalization = _generalization_rows(tangents)
    correlations, correlation_summary = _correlation_rows(tangents)
    _write_rows(directory / "sph_benchmark_cases.csv", case_rows)
    _write_rows(directory / "sph_trajectory_segments.csv", annotated)
    _write_rows(directory / "sph_mouth_crossings.csv", crossings)
    _write_rows(directory / "sph_trajectory_stability_summary.csv", stability)
    _write_rows(directory / "sph_tangent_comparison.csv", tangents)
    _write_rows(directory / "sph_tangent_generalization_summary.csv", generalization)
    _write_rows(directory / "sph_pca_flow_error_correlation.csv", correlations)
    _plots(directory, geometries, annotated, tangents)

    turning = [row for row in stability if row["evaluation_epoch"] == "TURNING"]
    stable_rows = [row for row in stability if row["evaluation_epoch"] == "STABLE_GATE"]
    turning_delta = float(np.median([float(row["median_angular_change_deg"]) for row in turning]))
    stable_delta = float(np.median([float(row["median_angular_change_deg"]) for row in stable_rows]))
    turning_resultant = float(np.median([float(row["resultant_length"]) for row in turning]))
    stable_resultant = float(np.median([float(row["resultant_length"]) for row in stable_rows]))
    pca_errors = [float(row["pca_error_deg"]) for row in tangents if row["pca_error_deg"] != ""]
    stable_errors = [float(row["stable_error_deg"]) for row in tangents if row["stable_error_deg"] != ""]
    signal = stable_delta < turning_delta and stable_resultant > turning_resultant
    stable_better = bool(pca_errors and stable_errors and np.mean(stable_errors) < np.mean(pca_errors))
    rotation_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in tangents:
        if str(row["case_id"]).startswith("rotated_cross_") and row["stable_error_deg"] != "":
            rotation_groups[(str(row["seed"]), str(row["branch_id"]))].append(float(row["stable_error_deg"]))
    rotation_span = max((max(values) - min(values) for values in rotation_groups.values() if len(values) > 1), default=float("nan"))
    classification = "A" if signal and stable_better else ("B" if signal else "C")
    summary = {
        "case": classification,
        "physical_case_count": len(geometries),
        "branch_result_count": len(tangents),
        "trajectory_segment_count": len(annotated),
        "turning_median_delta_deg": turning_delta,
        "stable_median_delta_deg": stable_delta,
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
        "rotation_max_error_span_deg": rotation_span,
        "correlation": correlation_summary,
        "benchmark_boundary": "physical SPH pressure/viscosity/collision with radial local drive; no production DFS/Guard/Shepherd/Handoff",
    }
    (directory / "sph_benchmark_result.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def run_synthetic_test() -> None:
    """Run geometry validity and a short end-to-end physical analysis test."""
    run_geometry_synthetic_test()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(f"/tmp/pdfs_general_sph_branch_direction_{_short_head()}"))
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--synthetic-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.synthetic_test:
        run_synthetic_test()
        print("general SPH geometry synthetic test: PASS")
    summary = run_physical_benchmark(args.output_dir, args.seed, args.quick)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
