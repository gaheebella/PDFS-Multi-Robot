"""M0/M1 audit of body-local forward propulsion without compression energy."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pygame_simulator.pre_exploration_general_pipeline_simulator import (
    DAMPING, EPSILON, GRID_ROW_SPACING, LOCAL_FORWARD_DRIVE_FORCE,
    NORMAL_EQUILIBRIUM_SCALE, PRESSURE_GAIN, REPULSION_GAIN, ROBOT_RADIUS,
    SAFE_RADIUS, SMOOTHING_LENGTH, SPH_PRESSURE_SCALE, STIFFNESS_EXPONENT,
    GeometryBuilder, SimulatorWorld, _gradient, run_headless,
)

DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/no_compression_exploration_audit"
CASES = ("M0_STRAIGHT", "M1_CROSS_BASELINE")
MODES = ("production_compression", "local_forward")
METRICS = (
    "gt_mean_forward_progress", "mean_speed_sanity", "max_speed",
    "min_inter_robot_distance", "overlap_pair_count", "mean_neighbor_degree",
    "reference_front_lateral_span", "reference_front_lateral_variance",
    "boundary_fraction", "boundary_component_count",
    "boundary_largest_component_fraction", "wall_contact_count",
)


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _initial_regime() -> dict:
    """Measure t=0 local-forward pressure and pair forces without integration."""
    world = SimulatorWorld(GeometryBuilder.build("M1_CROSS_BASELINE"), "local_forward")
    neighbors = world._neighbors()
    world._densities(neighbors)
    equilibrium = SAFE_RADIUS * NORMAL_EQUILIBRIUM_SCALE
    pressures = []
    pressure_magnitudes = []
    repulsion_magnitudes = []
    elastic_magnitudes = []
    for robot in world.robots:
        ratio = robot.density / max(world.reference_density, EPSILON)
        pressure = max(0.0, PRESSURE_GAIN * robot.density * (ratio**STIFFNESS_EXPONENT - 1.0)) * SPH_PRESSURE_SCALE
        robot.pressure = pressure
        pressures.append(pressure)
    for robot in world.robots:
        pressure_force = np.zeros(2)
        repulsion_force = np.zeros(2)
        elastic_force = np.zeros(2)
        for peer in neighbors[robot.robot_id]:
            offset = robot.position - peer.position
            distance = float(np.linalg.norm(offset))
            if distance <= EPSILON:
                continue
            coefficient = robot.pressure / max(robot.density**2, EPSILON) + peer.pressure / max(peer.density**2, EPSILON)
            pressure_force += -coefficient * _gradient(offset)
            if distance < equilibrium:
                repulsion_force += REPULSION_GAIN * (equilibrium-distance) / equilibrium * offset / distance
            if distance <= SAFE_RADIUS * 1.45:
                elastic_force += offset / distance * (-42.0 * (distance-equilibrium))
        pressure_magnitudes.append(float(np.linalg.norm(pressure_force)))
        repulsion_magnitudes.append(float(np.linalg.norm(repulsion_force)))
        elastic_magnitudes.append(float(np.linalg.norm(elastic_force)))
    positions = np.array([robot.position for robot in world.robots])
    return {
        "propulsion_mode": "local_forward", "robot_count": len(world.robots),
        "columns": 28, "x_spacing": 2.8, "longitudinal_spacing": GRID_ROW_SPACING,
        "robot_diameter": 2.0 * ROBOT_RADIUS,
        "initial_mean_density": float(np.mean([robot.density for robot in world.robots])),
        "reference_density": world.reference_density,
        "initial_mean_pressure": float(np.mean(pressures)),
        "initial_mean_pressure_force": float(np.mean(pressure_magnitudes)),
        "initial_mean_repulsion_force": float(np.mean(repulsion_magnitudes)),
        "initial_mean_elastic_force": float(np.mean(elastic_magnitudes)),
        "local_forward_drive_force": LOCAL_FORWARD_DRIVE_FORCE,
        "linear_damping": DAMPING, "normal_equilibrium_radius": equilibrium,
        "initial_min_neighbor_distance": world.sanity()["min_inter_robot_distance"],
        "initial_overlap_pair_count": world.sanity()["overlap_pair_count"],
        "initial_mean_neighbor_degree": float(np.mean([len(neighbors[robot.robot_id]) for robot in world.robots])),
        "initial_lateral_span": float(np.ptp(positions[:, 0])),
        "initial_longitudinal_span": float(np.ptp(positions[:, 1])),
    }


def _phase_summary(rows: list[dict], case: str, mode: str) -> list[dict]:
    result = []
    for phase in dict.fromkeys(row["gt_phase"] for row in rows):
        selected = [row for row in rows if row["gt_phase"] == phase]
        item = {"map_case": case, "propulsion_mode": mode, "phase": phase, "sample_count": len(selected)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_min"] = float(np.min(values))
            item[f"{metric}_max"] = float(np.max(values))
        item.update({
            "max_nan_inf_state_count": max(row["nan_inf_state_count"] for row in selected),
            "max_outside_free_space_robot_count": max(row["outside_free_space_robot_count"] for row in selected),
            "final_wall_projection_correction_count": selected[-1]["wall_projection_correction_count"],
        })
        result.append(item)
    return result


def run(frames: int, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    runners = {(case, mode): run_headless(case, frames, mode) for case in CASES for mode in MODES}
    local_rows = [row for case in CASES for row in runners[(case, "local_forward")].rows]
    _write(output / "no_compression_initial_state.csv", [_initial_regime()])
    _write(output / "no_compression_timeline.csv", local_rows)

    summaries = {
        key: _phase_summary(runner.rows, *key)
        for key, runner in runners.items()
    }
    _write(output / "no_compression_m0_summary.csv", summaries[("M0_STRAIGHT", "local_forward")])
    _write(output / "no_compression_m1_summary.csv", summaries[("M1_CROSS_BASELINE", "local_forward")])

    comparison = []
    for case in CASES:
        phases = dict.fromkeys(row["gt_phase"] for mode in MODES for row in runners[(case, mode)].rows)
        for phase in phases:
            for metric in METRICS:
                item = {"map_case": case, "phase": phase, "metric": metric}
                for mode in MODES:
                    values = [row[metric] for row in runners[(case, mode)].rows if row["gt_phase"] == phase]
                    item[mode] = float(np.mean(values)) if values else math.nan
                item["no_compression_minus_compression"] = item["local_forward"] - item["production_compression"]
                comparison.append(item)
    _write(output / "compression_vs_no_compression.csv", comparison)

    m0 = runners[("M0_STRAIGHT", "local_forward")].rows
    m1 = runners[("M1_CROSS_BASELINE", "local_forward")].rows
    compression_m1 = runners[("M1_CROSS_BASELINE", "production_compression")].rows
    def first_split_time(rows):
        return next((row["timestamp"] for row in rows if row["boundary_component_count"] > 1), math.nan)
    severe_pair_collision = min(row["min_inter_robot_distance"] for row in m1) < ROBOT_RADIUS
    natural_expansion = max(row["reference_front_lateral_span"] for row in m1) > max(row["reference_front_lateral_span"] for row in m0)
    topology_split = max(row["boundary_component_count"] for row in m1) > max(row["boundary_component_count"] for row in m0)
    verdict = "B. PARTIALLY_VALID" if severe_pair_collision and natural_expansion and topology_split else "A. NO_COMPRESSION_VALID" if natural_expansion and topology_split else "D. UNSTABLE" if severe_pair_collision else "C. PROPULSION_INSUFFICIENT"
    verdict_row = {
        "verdict": verdict,
        "m0_forward_progress": m0[-1]["gt_mean_forward_progress"],
        "m0_min_distance": min(row["min_inter_robot_distance"] for row in m0),
        "m0_max_boundary_components": max(row["boundary_component_count"] for row in m0),
        "m1_forward_progress": m1[-1]["gt_mean_forward_progress"],
        "m1_min_distance": min(row["min_inter_robot_distance"] for row in m1),
        "m1_max_boundary_components": max(row["boundary_component_count"] for row in m1),
        "compression_m1_first_boundary_split_time": first_split_time(compression_m1),
        "local_forward_m1_first_boundary_split_time": first_split_time(m1),
        "m0_max_lateral_span": max(row["reference_front_lateral_span"] for row in m0),
        "m1_max_lateral_span": max(row["reference_front_lateral_span"] for row in m1),
        "natural_expansion_observed": natural_expansion,
        "reason": "stable corridor travel and Junction-specific expansion were observed, but localized sub-radius pair collisions remain" if verdict.startswith("B") else "see measured fields",
    }
    _write(output / "no_compression_verdict.csv", [verdict_row])

    figure, axes = plt.subplots(3, 2, figsize=(13, 11))
    plot_metrics = (
        "gt_mean_forward_progress", "mean_speed_sanity", "min_inter_robot_distance",
        "reference_front_lateral_span", "boundary_fraction", "boundary_component_count",
    )
    for (case, mode), runner in runners.items():
        label = f"{case} / {mode}"
        for axis, metric in zip(axes.flat, plot_metrics):
            axis.plot([row["timestamp"] for row in runner.rows], [row[metric] for row in runner.rows], label=label)
    for axis, metric in zip(axes.flat, plot_metrics):
        axis.set_title(metric); axis.set_xlabel("time [s]"); axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    figure.suptitle("No-compression local-forward exploration audit — GT phases evaluation-only")
    figure.tight_layout()
    figure.savefig(output / "no_compression_exploration_audit.png", dpi=150)
    plt.close(figure)
    return verdict_row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    verdict = run(args.frames, args.output_dir)
    print(f"verdict={verdict['verdict']} frames={args.frames} output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
