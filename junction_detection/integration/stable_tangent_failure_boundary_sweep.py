"""Small physical sweep around the stable-tangent inlet failure boundary.

This module changes neither the SPH model nor either estimator.  It constructs
only a balanced set of inlet geometries, then delegates simulation, causal PCA,
stability annotation, and stable tangent aggregation to the existing modules.
GT geometry is used only to construct walls and evaluate angular error.
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
from junction_detection.integration.trajectory_stability_diagnostics import (
    annotate_trajectory_stability,
)
from pygame_simulator.general_junction_sph_benchmark import (
    JunctionGeometry,
    SphParameters,
    collision_mask_sanity,
    validate_geometry,
)
from pygame_simulator.general_junction_sph_inlet_benchmark import (
    InletBenchmarkGeometry,
    _branches,
    simulate_inlet_sph_case,
)


def _short_head() -> str:
    """Return the current commit abbreviation without changing Git state."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write heterogeneous dictionaries with a stable union header."""
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


def create_boundary_cases(seed: int = 20260818) -> list[InletBenchmarkGeometry]:
    """Create 16 targeted cases without a full factorial expansion.

    The inlet outward direction is -90 degrees, so travel into the Junction is
    +90 degrees. B1 is the evaluation target at the requested relative turn;
    B2 is a geometry-only alternate outlet. No world-axis label is used.
    """
    configurations = (
        # Reference and joint-extreme checks for shallow/medium turns.
        (40.0, "nominal", 105.0, "nominal", 24.0),
        (40.0, "long", 256.424384, "production", 84.0),
        (80.0, "nominal", 105.0, "nominal", 24.0),
        (80.0, "long", 256.424384, "production", 84.0),
        # The sharp neighborhood isolates length and width effects.
        (130.0, "nominal", 105.0, "nominal", 24.0),
        (130.0, "long", 256.424384, "nominal", 24.0),
        (130.0, "nominal", 105.0, "production", 84.0),
        (130.0, "long", 256.424384, "production", 84.0),
    )
    cases = []
    for config_index, (
        turn_angle, length_group, target_length, width_group, width
    ) in enumerate(configurations):
        for seed_offset in (0, 1):
            # Choosing the negative signed turn avoids any LEFT/RIGHT semantic;
            # a rigid rotation would produce the same relative experiment.
            target_angle = 90.0 - turn_angle
            radius = 115.0 if width_group == "production" else 42.0
            geometry = JunctionGeometry(
                case_id=(
                    f"boundary_t{int(turn_angle):03d}_{length_group}_"
                    f"{width_group}_s{seed_offset}"
                ),
                topology="3-way",
                seed=seed + config_index * 101 + seed_offset,
                center=(0.0, 0.0),
                central_radius=radius,
                branches=_branches(
                    (-90.0, target_angle, 160.0),
                    (126.0, target_length, 150.0),
                    (width, width, width),
                ),
                rotation_deg=0.0,
                length_group=length_group,
                width_group=width_group,
                source_family="STABLE_TANGENT_FAILURE_BOUNDARY",
            )
            case = InletBenchmarkGeometry(geometry, "B0")
            valid, reason = validate_geometry(geometry)
            if not valid:
                raise ValueError(f"{geometry.case_id}: {reason}")
            cases.append(case)
    return cases


def run_sanity_test() -> None:
    """Check the small design, relative turns, and analytical collision masks."""
    cases = create_boundary_cases(11)
    assert len(cases) == 16
    targets = [case.outgoing_branches[0] for case in cases]
    assert {case.turn_severity(target) for case, target in zip(cases, targets)} == {
        "shallow", "medium", "sharp"
    }
    assert {target.length for target in targets} == {105.0, 256.424384}
    assert {target.width for target in targets} == {24.0, 84.0}
    assert all(collision_mask_sanity(case.geometry)["pass"] for case in cases)


def _result_rows(cases: Sequence[InletBenchmarkGeometry]) -> list[dict[str, Any]]:
    """Run existing physics and estimators, retaining only target Branch B1."""
    parameters = SphParameters(steps=1600)
    segments: list[dict[str, Any]] = []
    crossings: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        print(f"[boundary] {index}/{len(cases)} {case.geometry.case_id}", flush=True)
        result = simulate_inlet_sph_case(case, parameters)
        segments.extend(result.segment_rows)
        crossings.extend(result.crossing_rows)
    pca_records = _attach_causal_pca(segments, crossings)
    annotated = annotate_trajectory_stability(segments)
    comparisons, _ = _comparison_rows(cases, annotated, pca_records)
    rows = []
    for row in comparisons:
        if row["branch_id"] != "B1":
            continue
        available = bool(row["estimator_available"])
        stable_error = row["stable_error_deg"]
        pca_error = row["pca_error_deg"]
        # This is an evaluation label, not an estimator threshold. It marks a
        # reliability failure when output is absent, >=5 deg, or worse than PCA.
        failure = (
            not available
            or stable_error == ""
            or float(stable_error) >= 5.0
            or (pca_error != "" and float(stable_error) > float(pca_error))
        )
        rows.append({
            "case_id": row["case_id"],
            "turn_angle_deg": row["turn_angle_deg"],
            "turn_severity": row["turn_severity"],
            "branch_length": row["branch_length"],
            "corridor_width": row["corridor_width"],
            "length_width_ratio": (
                float(row["branch_length"]) / float(row["corridor_width"])
            ),
            "seed": row["seed"],
            "pca_angular_error_deg": pca_error,
            "stable_angular_error_deg": stable_error,
            "stable_coverage": 1 if available else 0,
            "stable_signed_bias_deg": row["stable_signed_error_deg"],
            "stable_sample_count": row["stable_segment_count"],
            "stable_robot_count": row["stable_robot_count"],
            "predicted_lateral_drift": row["stable_predicted_lateral_drift"],
            "failure": failure,
        })
    return rows


def _summary_rows(results: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Aggregate the two seeds per physical condition and classify breadth."""
    groups: dict[tuple[float, float, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        groups[(
            float(row["turn_angle_deg"]),
            float(row["branch_length"]),
            float(row["corridor_width"]),
        )].append(row)
    summary = []
    failing_conditions = []
    for (turn, length, width), rows in sorted(groups.items()):
        available = [row for row in rows if row["stable_coverage"]]
        errors = [float(row["stable_angular_error_deg"]) for row in available]
        failures = sum(bool(row["failure"]) for row in rows)
        if failures:
            failing_conditions.append((turn, length, width, failures))
        summary.append({
            "turn_angle_deg": turn,
            "turn_severity": rows[0]["turn_severity"],
            "branch_length": length,
            "corridor_width": width,
            "length_width_ratio": length / width,
            "seed_count": len(rows),
            "available_seed_count": len(available),
            "failure_seed_count": failures,
            "failure_repeatability": failures / len(rows),
            "stable_mean_error_deg": "" if not errors else float(np.mean(errors)),
            "stable_max_error_deg": "" if not errors else float(np.max(errors)),
            "mean_abs_end_drift": (
                "" if not available else float(np.mean([
                    abs(float(row["predicted_lateral_drift"])) for row in available
                ]))
            ),
        })
    # Case A means a compact, reproducible sharp-turn boundary. Failures in
    # shallow/medium conditions or one-seed-only failures indicate Case B.
    localized = bool(failing_conditions) and all(
        turn >= 105.0 for turn, _, _, _ in failing_conditions
    )
    reproducible = all(failures == 2 for *_, failures in failing_conditions)
    sparse = len(failing_conditions) <= len(groups) / 2
    classification = "A" if localized and reproducible and sparse else "B"
    summary.append({
        "turn_angle_deg": "ALL",
        "turn_severity": "FINAL",
        "seed_count": len(results),
        "available_seed_count": sum(int(row["stable_coverage"]) for row in results),
        "failure_seed_count": sum(bool(row["failure"]) for row in results),
        "failure_repeatability": "",
        "stable_mean_error_deg": "",
        "stable_max_error_deg": "",
        "mean_abs_end_drift": "",
        "classification": classification,
    })
    return summary, classification


def _save_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Plot turn versus normalized length with width and seed encoded."""
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for width, marker in ((24.0, "o"), (84.0, "s")):
        selected = [row for row in rows if float(row["corridor_width"]) == width]
        scatter = axis.scatter(
            [float(row["length_width_ratio"]) for row in selected],
            [float(row["turn_angle_deg"]) for row in selected],
            c=[
                float(row["stable_angular_error_deg"])
                if row["stable_angular_error_deg"] != "" else 20.0
                for row in selected
            ],
            s=95,
            marker=marker,
            cmap="magma",
            vmin=0.0,
            vmax=20.0,
            edgecolors=["cyan" if bool(row["failure"]) else "white" for row in selected],
            linewidths=1.5,
            label=f"width={width:g}",
        )
    axis.set(
        xlabel="normalized branch length (length / width)",
        ylabel="relative turn angle [deg]",
        title="Stable-motion tangent failure boundary (cyan edge = failure)",
    )
    axis.legend()
    figure.colorbar(scatter, ax=axis, label="stable angular error [deg]")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_sweep(output_dir: Path, seed: int) -> str:
    """Run the minimal boundary sweep and save two CSVs and one PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = create_boundary_cases(seed)
    results = _result_rows(cases)
    summary, classification = _summary_rows(results)
    _write_rows(output_dir / "failure_boundary_results.csv", results)
    _write_rows(output_dir / "failure_boundary_summary.csv", summary)
    _save_plot(output_dir / "stable_error_failure_boundary.png", results)
    print(f"classification=Case {classification}")
    print(f"artifacts={output_dir}")
    return classification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(f"/tmp/pdfs_stable_tangent_boundary_{_short_head()}"),
    )
    parser.add_argument("--sanity-test", action="store_true")
    args = parser.parse_args()
    if args.sanity_test:
        run_sanity_test()
        print("stable tangent boundary sanity test: PASS")
    run_sweep(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
