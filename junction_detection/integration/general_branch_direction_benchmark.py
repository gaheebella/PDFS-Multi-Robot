"""General/local Branch-direction benchmark generator and runner.

The broad geometry sweep is a deterministic kinematic benchmark for the
estimator interface.  It deliberately exposes only local trajectory fields to
the candidate.  Ground-truth geometry remains in evaluation columns.  The
production SPH fixture is not edited; its current RIGHT failure is retained as
an independently identified stress case, not as a tuning target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from junction_detection.integration.trajectory_stability_diagnostics import (
    analyze_benchmark,
    angular_error_deg,
    normalize_angle_deg,
    run_synthetic_test as run_analysis_synthetic_test,
)


@dataclass(frozen=True)
class BenchmarkCase:
    """One parameterized Junction benchmark before rigid rotations."""

    case_id: str
    seed: int
    way_count: int
    branch_angles_deg: tuple[float, ...]
    branch_length: float
    length_group: str
    corridor_width: float
    corridor_width_group: str
    rotation_deg: float
    source: str = "KINEMATIC_LOCAL_TRAJECTORY_BENCHMARK"

    def as_row(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "seed": self.seed,
            "way_count": self.way_count,
            "branch_angles_deg": json.dumps(self.branch_angles_deg),
            "branch_length": self.branch_length,
            "length_group": self.length_group,
            "corridor_width": self.corridor_width,
            "corridor_width_group": self.corridor_width_group,
            "rotation_deg": self.rotation_deg,
            "source": self.source,
        }


def _short_head() -> str:
    """Read the current Git identity without mutating repository state."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def create_benchmark_cases(seed: int = 20260817) -> list[BenchmarkCase]:
    """Create a balanced 3/4/5-way, length, width, seed, and rotation sweep."""
    length_values = {"short": 90.0, "medium": 150.0, "long": 260.0}
    width_values = {"narrow": 58.0, "nominal": 84.0, "wide": 116.0}
    rotations = (0.0, 30.0, 60.0, 120.0)
    # Base angles contain both axis-aligned and arbitrary directions.  A
    # rotation is applied rigidly to the entire Junction, never snapped.
    topology_angles = {
        3: (0.0, 110.0, -125.0),
        4: (0.0, 70.0, 160.0, -110.0),
        5: (0.0, 45.0, 110.0, 175.0, -95.0),
    }
    cases: list[BenchmarkCase] = []
    index = 0
    for length_index, (length_group, length) in enumerate(length_values.items()):
        for width_index, (width_group, width) in enumerate(width_values.items()):
            way_count = (3, 4, 5)[(length_index + 2 * width_index) % 3]
            case_seed = seed + 101 * length_index + 17 * width_index
            for rotation in rotations:
                angles = tuple(normalize_angle_deg(angle + rotation) for angle in topology_angles[way_count])
                cases.append(BenchmarkCase(
                    case_id=f"general_{index:02d}_{length_group}_{width_group}_rot{int(rotation):03d}",
                    seed=case_seed,
                    way_count=way_count,
                    branch_angles_deg=angles,
                    branch_length=length,
                    length_group=length_group,
                    corridor_width=width,
                    corridor_width_group=width_group,
                    rotation_deg=rotation,
                ))
            index += 1

    # Preserve the observed production RIGHT geometry as one stress case.  Its
    # PCA yaw is injected as a measured baseline label, never into the motion
    # gate or candidate estimator.
    cases.append(BenchmarkCase(
        case_id="stress_case_right_long_rot000",
        seed=seed + 999,
        way_count=4,
        branch_angles_deg=(0.0,),
        branch_length=256.424384,
        length_group="long",
        corridor_width=84.0,
        corridor_width_group="nominal",
        rotation_deg=0.0,
        source="PRODUCTION_RIGHT_STRESS_RECONSTRUCTION",
    ))
    return cases


def _mouth_points(
    rng: np.random.Generator,
    gt_angle: float,
    width: float,
    robot_count: int,
    pca_bias_deg: float,
) -> dict[int, tuple[float, float]]:
    """Generate observed crossing origins along a biased transverse envelope."""
    tangent_angle = math.radians(gt_angle + pca_bias_deg)
    normal = np.asarray([-math.sin(tangent_angle), math.cos(tangent_angle)])
    tangent = np.asarray([math.cos(tangent_angle), math.sin(tangent_angle)])
    laterals = np.linspace(-0.45 * width, 0.45 * width, robot_count)
    points: dict[int, tuple[float, float]] = {}
    for robot_id, lateral in enumerate(laterals):
        point = normal * lateral + tangent * rng.normal(0.0, width * 0.008)
        points[robot_id] = (float(point[0]), float(point[1]))
    return points


def _generate_branch_segments(
    case: BenchmarkCase,
    branch_index: int,
    gt_angle: float,
    robot_count: int = 14,
    updates: int = 21,
) -> list[dict[str, Any]]:
    """Generate causal local robot trajectories from a turning entrance.

    GT supplies the physical corridor direction to the data generator, just as
    wall geometry supplies it in a simulator.  It is not used by the analysis
    gate.  Noise parameters scale with width and are seed-driven.
    """
    rng = np.random.default_rng(case.seed + branch_index * 7919)
    stress = case.case_id.startswith("stress_case_right_long")
    pca_bias = 0.76332534 if stress else float(rng.uniform(-2.4, 2.4))
    mouth_points = _mouth_points(rng, gt_angle, case.corridor_width, robot_count, pca_bias)
    incoming_angle = normalize_angle_deg(gt_angle - rng.choice((55.0, 75.0, 105.0)))
    turn_delta = normalize_angle_deg(incoming_angle - gt_angle)
    base_step = case.branch_length / (updates * 1.10)
    width_noise = 0.45 + 0.012 * case.corridor_width
    rows: list[dict[str, Any]] = []
    positions = {robot_id: np.asarray(mouth_points[robot_id], dtype=float) for robot_id in range(robot_count)}
    previous_angles = np.full(robot_count, incoming_angle, dtype=float)
    for update in range(updates):
        progress = (update + 1) / updates
        # Smooth turn decay is dimensionless in normalized progress.  Wider
        # corridors permit more lateral meander; no estimator threshold reads
        # width, length, or this model.
        turn_fraction = math.exp(-5.0 * progress)
        common_wobble = rng.normal(0.0, width_noise * (0.25 + turn_fraction))
        angles = []
        for robot_id in range(robot_count):
            individual = rng.normal(0.0, width_noise * (0.15 + 0.75 * turn_fraction))
            inertia = 0.12 * normalize_angle_deg(previous_angles[robot_id] - gt_angle)
            angle = normalize_angle_deg(gt_angle + turn_delta * turn_fraction + inertia + common_wobble + individual)
            previous_angles[robot_id] = angle
            angles.append(angle)
        radians = np.radians(np.asarray(angles))
        resultant = float(math.hypot(float(np.mean(np.cos(radians))), float(np.mean(np.sin(radians)))))
        mean_angle = normalize_angle_deg(math.degrees(math.atan2(float(np.mean(np.sin(radians))), float(np.mean(np.cos(radians))))))
        neighbor_spread = [abs(normalize_angle_deg(angle - mean_angle)) for angle in angles]
        for robot_id, angle in enumerate(angles):
            step = max(base_step * float(rng.normal(1.0, 0.045)), base_step * 0.5)
            direction = np.asarray([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
            previous = positions[robot_id].copy()
            current = previous + direction * step
            positions[robot_id] = current
            phase = "JUNCTION_TURNING" if progress <= 1.0 / 3.0 else "EXPLORE_BRANCH"
            rows.append({
                "environment_id": case.case_id,
                "case_id": case.case_id,
                "seed": case.seed,
                "branch_id": f"B{branch_index}",
                "gt_branch_angle_deg": gt_angle,
                "robot_id": robot_id,
                "frame": update,
                "time": update / 10.0,
                "previous_x": float(previous[0]),
                "previous_y": float(previous[1]),
                "current_x": float(current[0]),
                "current_y": float(current[1]),
                "dx": float(current[0] - previous[0]),
                "dy": float(current[1] - previous[1]),
                "displacement_length": step,
                "motion_angle_deg": angle,
                "observed_speed": step * 10.0,
                "segment_update_count": update + 1,
                "lifecycle_phase": phase,
                "anchor_confirmed": True,
                "branch_discovered": update >= 2,
                "neighbor_count": robot_count - 1,
                "neighbor_motion_mean_deg": mean_angle,
                "neighbor_motion_dispersion_deg": float(np.median(neighbor_spread)),
                "local_heading_consensus": resultant,
                "progress_fraction": progress,
                "mouth_x": mouth_points[robot_id][0],
                "mouth_y": mouth_points[robot_id][1],
                "information_scope": "SELF_MOTION_AND_LOCAL_NEIGHBOR_MOTION",
                "pca_tangent_deg": 0.76332534 if stress else "",
            })
    return rows


def generate_trajectory_dataset(cases: Sequence[BenchmarkCase]) -> list[dict[str, Any]]:
    """Generate all robot-level segments for a case collection."""
    rows: list[dict[str, Any]] = []
    for case in cases:
        for branch_index, angle in enumerate(case.branch_angles_deg):
            rows.extend(_generate_branch_segments(case, branch_index, angle))
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save_right_stress_analysis(output_dir: Path) -> dict[str, Any]:
    """Evaluate the measured RIGHT snapshot after the general benchmark."""
    comparisons = _read_csv(output_dir / "branch_tangent_estimator_comparison.csv")
    row = next(item for item in comparisons if item["case_id"] == "stress_case_right_long_rot000")
    candidate_error = float(row["candidate_error_deg"])
    depth = 256.424384
    pca_error = 0.76332534
    pca_drift = depth * math.sin(math.radians(pca_error))
    candidate_drift = depth * math.sin(math.radians(candidate_error))
    # Exact production observations from the frozen failure snapshot.
    gt_robot_583_clearance = 0.217203
    pca_robot_583_clearance = -1.618905
    sensitivity = (gt_robot_583_clearance - pca_robot_583_clearance) / max(abs(pca_drift), 1.0e-12)
    candidate_clearance = gt_robot_583_clearance - sensitivity * abs(candidate_drift)
    result = {
        "case_id": "stress_case_right_long",
        "measured_production_pca_error_deg": pca_error,
        "candidate_error_deg": candidate_error,
        "contacted_depth": depth,
        "pca_depth_amplified_drift_px": pca_drift,
        "candidate_depth_amplified_drift_px": candidate_drift,
        "production_safe_slots": 19,
        "candidate_safe_slots": 20 if candidate_clearance >= 0.0 else 19,
        "gt_safe_slots": 20,
        "production_robot_583_clearance_px": pca_robot_583_clearance,
        "candidate_robot_583_clearance_px": candidate_clearance,
        "gt_robot_583_clearance_px": gt_robot_583_clearance,
        "shadow_model": "first-order orientation sensitivity calibrated from frozen production/GT snapshot",
    }
    (output_dir / "right_stress_case_shadow_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    labels = ["production PCA", "stable motion", "GT"]
    axes[0].bar(labels, [pca_error, candidate_error, 0.0], color=["tab:red", "tab:blue", "tab:green"])
    axes[0].set(title="RIGHT tangent error", ylabel="absolute error [deg]")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(labels, [19, result["candidate_safe_slots"], 20], color=["tab:red", "tab:blue", "tab:green"])
    axes[1].set(title="Frozen snapshot shadow row", ylabel="safe slots", ylim=(18, 20.4))
    axes[1].tick_params(axis="x", rotation=20)
    fig.savefig(output_dir / "right_stress_case_shadow_comparison.png", dpi=160)
    plt.close(fig)
    return result


def run_benchmark(output_dir: Path, seed: int) -> dict[str, Any]:
    """Generate, analyze, and persist one complete benchmark sweep."""
    cases = create_benchmark_cases(seed)
    segments = generate_trajectory_dataset(cases)
    result = analyze_benchmark([case.as_row() for case in cases], segments, output_dir)
    right = _save_right_stress_analysis(output_dir)
    scope = {
        "broad_geometry_source": "deterministic kinematic local-trajectory benchmark",
        "production_fixture_support": "current simulator is one axis-aligned 4-way cross (three outgoing children plus ingress)",
        "rotation_method": "exact rigid rotation of full generated Junction and all local observations",
        "gt_policy": "evaluation columns and error scoring only",
        "candidate_inputs": ["robot displacement", "update order", "local neighbor motion"],
        "warning": "Broad geometry results establish estimator/interface behavior, not yet a physical SPH generalization claim.",
    }
    (output_dir / "benchmark_scope.json").write_text(json.dumps(scope, indent=2, sort_keys=True), encoding="utf-8")
    result["right_stress"] = right
    return result


def run_synthetic_test() -> None:
    """Exercise case generation, wrap-around, and local-only analysis."""
    run_analysis_synthetic_test()
    cases = create_benchmark_cases(7)
    assert {case.way_count for case in cases} >= {3, 4, 5}
    assert {case.length_group for case in cases} >= {"short", "medium", "long"}
    assert {case.corridor_width_group for case in cases} >= {"narrow", "nominal", "wide"}
    sample = _generate_branch_segments(cases[0], 0, cases[0].branch_angles_deg[0], robot_count=4, updates=8)
    assert sample and all(row["anchor_confirmed"] for row in sample)
    assert all("gt_branch_angle_deg" in row for row in sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(f"/tmp/pdfs_general_branch_direction_{_short_head()}"))
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--synthetic-test", action="store_true", help="run helper checks before the benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.synthetic_test:
        run_synthetic_test()
        print("general_branch_direction_benchmark synthetic test: PASS")
    result = run_benchmark(args.output_dir, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
