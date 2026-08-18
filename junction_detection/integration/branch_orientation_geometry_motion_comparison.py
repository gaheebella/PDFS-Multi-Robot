"""Evaluation-only comparison of PCA, stable motion, and opening geometry.

The physical inlet simulation, PCA baseline, stable-motion estimator, LiDAR ray
casting, and opening detector are imported unchanged.  This module only adapts
the shared JunctionGeometry to wall segments and evaluates detected opening
centers; no fusion or production policy is implemented.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from junction_detection.integration.general_sph_inlet_tangent_analysis import (
    _attach_causal_pca,
    _comparison_rows,
)
from junction_detection.integration.stable_tangent_failure_boundary_sweep import (
    create_boundary_cases,
)
from junction_detection.integration.trajectory_stability_diagnostics import (
    annotate_trajectory_stability,
    angular_error_deg,
    signed_angle_delta_deg,
)
from junction_detection.pointcloud.pointcloud_junction_detector import (
    _merge_linear_intervals,
    _split_wrapped_interval,
    detect_openings,
    simulate_lidar_scan,
)
from junction_detection.pointcloud.pointcloud_junction_detector_local_topology import (
    _match_openings,
    ground_truth_openings_from_geometry,
)
from pygame_simulator.general_junction_sph_benchmark import (
    JunctionGeometry,
    SphParameters,
    collision_mask_sanity,
    validate_geometry,
)
from pygame_simulator.general_junction_sph_inlet_benchmark import (
    InletBenchmarkGeometry,
    simulate_inlet_sph_case,
)


# Reuses the diagnostic correctness boundary declared in the preceding failure
# sweep. It labels evaluation outcomes only and is never passed to a detector.
EVALUATION_CORRECT_DEG = 5.0


def _short_head() -> str:
    """Return the current Git revision abbreviation without mutation."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write dictionaries using their union of fields."""
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def geometry_wall_segments(geometry: JunctionGeometry) -> np.ndarray:
    """Adapt general SPH radial corridors to the existing LiDAR wall format.

    This is the same circle-with-openings construction used by the Point Cloud
    generator, extended only to preserve per-Branch lengths already present in
    JunctionGeometry. All cases in this comparison have a common corridor width.
    """
    widths = {round(branch.width, 12) for branch in geometry.branches}
    if len(widths) != 1:
        raise ValueError("comparison adapter requires a common corridor width")
    half_width = geometry.branches[0].width * 0.5
    radius = geometry.central_radius
    if radius <= half_width:
        raise ValueError("central radius must exceed corridor half-width")
    r0 = math.sqrt(radius * radius - half_width * half_width)
    walls: list[list[list[float]]] = []
    openings: list[tuple[float, float]] = []
    half_opening_deg = math.degrees(math.asin(half_width / radius))
    for branch in geometry.branches:
        tangent, normal = branch.tangent, branch.normal
        left_start = r0 * tangent + half_width * normal
        right_start = r0 * tangent - half_width * normal
        end_axial = radius + branch.length
        left_end = end_axial * tangent + half_width * normal
        right_end = end_axial * tangent - half_width * normal
        walls.extend([
            [left_start.tolist(), left_end.tolist()],
            [right_start.tolist(), right_end.tolist()],
        ])
        openings.extend(_split_wrapped_interval(
            branch.angle_deg - half_opening_deg,
            branch.angle_deg + half_opening_deg,
        ))

    cursor = 0.0
    for start, end in _merge_linear_intervals(openings):
        if start > cursor + 1.0e-9:
            _append_arc(walls, radius, cursor, start)
        cursor = max(cursor, end)
    if cursor < 360.0 - 1.0e-9:
        _append_arc(walls, radius, cursor, 360.0)
    return np.asarray(walls, dtype=float)


def _append_arc(
    walls: list[list[list[float]]],
    radius: float,
    start_deg: float,
    end_deg: float,
) -> None:
    """Append central-wall segments with the existing 2-degree resolution."""
    count = max(1, int(math.ceil((end_deg - start_deg) / 2.0)))
    angles = np.linspace(start_deg, end_deg, count + 1)
    points = np.column_stack((
        radius * np.cos(np.radians(angles)),
        radius * np.sin(np.radians(angles)),
    ))
    walls.extend([[p0.tolist(), p1.tolist()] for p0, p1 in zip(points[:-1], points[1:])])


def _representative_cases(seed: int) -> list[InletBenchmarkGeometry]:
    """Select 12 paired nominal/production cases from the prior boundary design."""
    cases = create_boundary_cases(seed)
    keep = {
        (40.0, "nominal", "nominal"),
        (40.0, "long", "production"),
        (80.0, "nominal", "nominal"),
        (80.0, "long", "production"),
        (130.0, "nominal", "nominal"),
        (130.0, "long", "production"),
    }
    selected = []
    for case in cases:
        target = case.outgoing_branches[0]
        key = (
            case.turn_angle_deg(target),
            case.geometry.length_group,
            case.geometry.width_group,
        )
        if key in keep:
            selected.append(case)
    return selected


def run_sanity_test() -> None:
    """Validate case balance, walls, collision mask, and an ideal scan."""
    cases = _representative_cases(19)
    assert len(cases) == 12
    assert {case.turn_severity(case.outgoing_branches[0]) for case in cases} == {
        "shallow", "medium", "sharp"
    }
    for case in cases:
        assert validate_geometry(case.geometry)[0]
        assert collision_mask_sanity(case.geometry)["pass"]
        walls = geometry_wall_segments(case.geometry)
        scan = simulate_lidar_scan(
            walls, case.geometry.center, max_range_m=500.0
        )
        assert scan.range_m.shape == (360,)
        assert detect_openings(*scan.detector_input())


def _rankdata(values: Sequence[float]) -> np.ndarray:
    """Return average ranks for a small Spearman calculation."""
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=float)
    index = 0
    while index < array.size:
        end = index + 1
        while end < array.size and array[order[end]] == array[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1) + 1.0
        index = end
    return ranks


def _correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Return Pearson correlation or NaN for degenerate input."""
    if len(x) < 2 or np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def _geometry_observation(case: InletBenchmarkGeometry) -> dict[str, Any]:
    """Run the unchanged ideal LiDAR/opening detector and evaluation matching."""
    geometry = case.geometry
    walls = geometry_wall_segments(geometry)
    maximum_extent = geometry.central_radius + max(
        branch.length for branch in geometry.branches
    )
    scan = simulate_lidar_scan(
        walls,
        geometry.center,
        angle_step_deg=1.0,
        max_range_m=maximum_extent * 1.25,
        noise_std_m=0.0,
        dropout_probability=0.0,
        seed=geometry.seed,
    )
    openings = detect_openings(*scan.detector_input())
    gt_openings = ground_truth_openings_from_geometry(
        [branch.angle_deg for branch in geometry.branches],
        anchor_xy=geometry.center,
        anchor_yaw_deg=0.0,
        corridor_width_m=geometry.branches[0].width,
        central_radius_m=geometry.central_radius,
    )
    matches = _match_openings(gt_openings, openings)
    target = case.outgoing_branches[0]
    target_gt = min(
        range(len(gt_openings)),
        key=lambda index: angular_error_deg(
            gt_openings[index]["center_angle"], target.angle_deg
        ),
    )
    match = next((item for item in matches if item[0] == target_gt), None)
    if match is None:
        return {
            "geometry_tangent_deg": "",
            "geometry_error_deg": "",
            "geometry_signed_error_deg": "",
            "pointcloud_confidence": "",
            "opening_width_deg": "",
            "opening_start_refined": "",
            "opening_end_refined": "",
            "pointcloud_opening_count": len(openings),
        }
    opening = openings[match[1]]
    tangent = float(opening["center_angle"])
    return {
        "geometry_tangent_deg": tangent,
        "geometry_error_deg": angular_error_deg(tangent, target.angle_deg),
        "geometry_signed_error_deg": signed_angle_delta_deg(tangent, target.angle_deg),
        "pointcloud_confidence": opening["confidence"],
        "opening_width_deg": opening["width_deg"],
        "opening_start_refined": opening["start_refined"],
        "opening_end_refined": opening["end_refined"],
        "pointcloud_opening_count": len(openings),
    }


def _comparison_data(cases: Sequence[InletBenchmarkGeometry]) -> list[dict[str, Any]]:
    """Generate physical trajectories and compare the three unchanged estimates."""
    parameters = SphParameters(steps=1600)
    segments: list[dict[str, Any]] = []
    crossings: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        print(f"[geometry-motion] {index}/{len(cases)} {case.geometry.case_id}", flush=True)
        result = simulate_inlet_sph_case(case, parameters)
        segments.extend(result.segment_rows)
        crossings.extend(result.crossing_rows)
    pca_records = _attach_causal_pca(segments, crossings)
    annotated = annotate_trajectory_stability(segments)
    comparisons, _ = _comparison_rows(cases, annotated, pca_records)
    target_comparisons = {
        str(row["case_id"]): row
        for row in comparisons if row["branch_id"] == "B1"
    }
    stable_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in annotated:
        if row["branch_id"] == "B1" and bool(row["stable_candidate"]):
            stable_groups[str(row["case_id"])].append(row)

    output = []
    for case in cases:
        target = case.outgoing_branches[0]
        motion = target_comparisons[case.geometry.case_id]
        geometry = _geometry_observation(case)
        stable_rows = stable_groups[case.geometry.case_id]
        stable_available = bool(motion["estimator_available"])
        geometry_available = geometry["geometry_tangent_deg"] != ""
        stable_error = motion["stable_error_deg"]
        geometry_error = geometry["geometry_error_deg"]
        disagreement = (
            angular_error_deg(
                float(motion["stable_tangent_deg"]),
                float(geometry["geometry_tangent_deg"]),
            )
            if stable_available and geometry_available else ""
        )
        motion_correct = stable_available and float(stable_error) < EVALUATION_CORRECT_DEG
        geometry_correct = geometry_available and float(geometry_error) < EVALUATION_CORRECT_DEG
        category = (
            "motion_correct_geometry_correct" if motion_correct and geometry_correct
            else "motion_wrong_geometry_correct" if geometry_correct
            else "motion_correct_geometry_wrong" if motion_correct
            else "motion_wrong_geometry_wrong"
        )
        output.append({
            "case_id": case.geometry.case_id,
            "seed": case.geometry.seed,
            "turn_severity": case.turn_severity(target),
            "turn_angle_deg": case.turn_angle_deg(target),
            "width": target.width,
            "branch_length": target.length,
            "length_width_ratio": target.length / target.width,
            "gt_tangent_deg": target.angle_deg,
            "pca_tangent_deg": motion["pca_tangent_deg"],
            "pca_error_deg": motion["pca_error_deg"],
            "stable_tangent_deg": motion["stable_tangent_deg"],
            "stable_error_deg": stable_error,
            "stable_signed_error_deg": motion["stable_signed_error_deg"],
            "geometry_tangent_deg": geometry["geometry_tangent_deg"],
            "geometry_error_deg": geometry_error,
            "geometry_signed_error_deg": geometry["geometry_signed_error_deg"],
            "stable_geometry_disagreement_deg": disagreement,
            "stable_coverage": int(stable_available),
            "stable_sample_count": motion["stable_segment_count"],
            "stable_dispersion_deg": (
                "" if not stable_rows else float(np.median([
                    float(row["cohort_dispersion_deg"]) for row in stable_rows
                ]))
            ),
            "stable_resultant": (
                "" if not stable_rows else float(np.median([
                    float(row["cohort_resultant_length"]) for row in stable_rows
                ]))
            ),
            "pointcloud_confidence": geometry["pointcloud_confidence"],
            "opening_width_deg": geometry["opening_width_deg"],
            "opening_start_refined": geometry["opening_start_refined"],
            "opening_end_refined": geometry["opening_end_refined"],
            "pointcloud_opening_count": geometry["pointcloud_opening_count"],
            "complementarity_category": category,
        })
    return output


def _summary(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Summarize complementarity and select Case A/B/C from observed outputs."""
    categories = defaultdict(int)
    for row in rows:
        categories[str(row["complementarity_category"])] += 1
    paired = [
        row for row in rows
        if row["stable_error_deg"] != "" and row["geometry_error_deg"] != ""
    ]
    stable_signed = [float(row["stable_signed_error_deg"]) for row in paired]
    geometry_signed = [float(row["geometry_signed_error_deg"]) for row in paired]
    disagreements = [float(row["stable_geometry_disagreement_deg"]) for row in paired]
    stable_errors = [float(row["stable_error_deg"]) for row in paired]
    geometry_errors = [float(row["geometry_error_deg"]) for row in paired]
    motion_failure_rows = [
        row for row in rows
        if row["stable_error_deg"] == ""
        or float(row["stable_error_deg"]) >= EVALUATION_CORRECT_DEG
    ]
    corrected_failures = sum(
        row["geometry_error_deg"] != ""
        and float(row["geometry_error_deg"]) < EVALUATION_CORRECT_DEG
        for row in motion_failure_rows
    )
    geometry_coverage = sum(row["geometry_error_deg"] != "" for row in rows)
    if geometry_coverage < len(rows):
        classification = "C"
    elif motion_failure_rows and corrected_failures == len(motion_failure_rows):
        classification = "A"
    else:
        classification = "B"
    summary = [
        {"metric": "case_count", "value": len(rows)},
        {"metric": "paired_estimator_count", "value": len(paired)},
        {"metric": "geometry_coverage", "value": geometry_coverage / len(rows)},
        {"metric": "motion_failure_count", "value": len(motion_failure_rows)},
        {"metric": "motion_failures_corrected_by_geometry", "value": corrected_failures},
        {"metric": "motion_correct_geometry_correct", "value": categories["motion_correct_geometry_correct"]},
        {"metric": "motion_wrong_geometry_correct", "value": categories["motion_wrong_geometry_correct"]},
        {"metric": "motion_correct_geometry_wrong", "value": categories["motion_correct_geometry_wrong"]},
        {"metric": "motion_wrong_geometry_wrong", "value": categories["motion_wrong_geometry_wrong"]},
        {"metric": "stable_geometry_signed_error_pearson", "value": _correlation(stable_signed, geometry_signed)},
        {"metric": "stable_geometry_signed_error_spearman", "value": _correlation(_rankdata(stable_signed), _rankdata(geometry_signed))},
        {"metric": "disagreement_vs_stable_error_pearson", "value": _correlation(disagreements, stable_errors)},
        {"metric": "disagreement_vs_max_error_pearson", "value": _correlation(disagreements, np.maximum(stable_errors, geometry_errors))},
        {"metric": "classification", "value": classification},
    ]
    return summary, classification


def _save_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Save the single requested estimator-error comparison plot."""
    labels = [str(row["case_id"]).replace("boundary_", "") for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    figure, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    axis.bar(x - width, [
        np.nan if row["pca_error_deg"] == "" else float(row["pca_error_deg"])
        for row in rows
    ], width, label="PCA")
    axis.bar(x, [
        np.nan if row["stable_error_deg"] == "" else float(row["stable_error_deg"])
        for row in rows
    ], width, label="stable motion")
    axis.bar(x + width, [float(row["geometry_error_deg"]) for row in rows], width, label="opening geometry")
    axis.axhline(EVALUATION_CORRECT_DEG, color="black", linestyle="--", linewidth=1, label="evaluation boundary")
    axis.set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
    axis.set(ylabel="absolute tangent error [deg]", title="Geometry vs motion Branch orientation")
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_comparison(output_dir: Path, seed: int) -> str:
    """Run 12 representative physical cases and persist minimal artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _comparison_data(_representative_cases(seed))
    summary, classification = _summary(rows)
    _write_rows(output_dir / "geometry_motion_comparison.csv", rows)
    _write_rows(output_dir / "geometry_motion_summary.csv", summary)
    _save_plot(output_dir / "geometry_vs_motion_error_comparison.png", rows)
    print(f"classification=Case {classification}")
    print(f"artifacts={output_dir}")
    return classification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(f"/tmp/pdfs_geometry_motion_comparison_{_short_head()}"),
    )
    parser.add_argument("--sanity-test", action="store_true")
    args = parser.parse_args()
    if args.sanity_test:
        run_sanity_test()
        print("geometry-motion comparison sanity test: PASS")
    run_comparison(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
