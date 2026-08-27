"""Evaluation-only repeated surface-population audit.

The production simulator currently exposes no seed injection mechanism.  This
runner therefore records requested seeds separately from ``seed_controlled``
and never labels uncontrolled repetitions as independent seeded experiments.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC = ROOT / "pygame_simulator/single_junction_sph_dfs_lidar_front_trigger_diagnostics.py"


def _float(value):
    return None if value in (None, "") else float(value)


def _stats(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {k: "" for k in ("mean", "std", "median", "p10", "p90")}
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "median": float(np.median(values)), "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90))}


def _events(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["event"]: row["timestamp_or_delta_s"] for row in csv.DictReader(handle)}


def _run_one(run_id, requested_seed, args, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"SDL_VIDEODRIVER": "dummy", "SPH_DFS_MAX_FRAMES": str(args.frames), "SPH_DFS_DIAGNOSTIC_OUTPUT": str(run_dir)})
    # Deliberately do not set a seed: the simulator has no supported seed path.
    subprocess.run([sys.executable, str(DIAGNOSTIC)], cwd=ROOT, env=env, check=True)
    with (run_dir / "local_front_surface_peak_timeline.csv").open(newline="", encoding="utf-8") as handle:
        timeline = list(csv.DictReader(handle))
    with (run_dir / "local_front_surface_phase_summary.csv").open(newline="", encoding="utf-8") as handle:
        phase_rows = list(csv.DictReader(handle))
    events = _events(run_dir / "lidar_front_trigger_event_summary.csv")
    count_peak = max(timeline, key=lambda row: float(row["forward_zero_neighbor_count"])) if timeline else {}
    fraction_peak = max(timeline, key=lambda row: float(row["forward_zero_neighbor_fraction"])) if timeline else {}
    phases = {row["phase"]: row for row in phase_rows}
    c = phases.get("SPH_CORRIDOR", {}); o = phases.get("SPH_OPENING_APPROACH", {}); j = phases.get("SPH_JUNCTION_REGION", {})
    c_count, o_count, j_count = _float(c.get("surface_count_mean")), _float(o.get("surface_count_mean")), _float(j.get("surface_count_mean"))
    c_fraction, o_fraction, j_fraction = _float(c.get("surface_fraction_mean")), _float(o.get("surface_fraction_mean")), _float(j.get("surface_fraction_mean"))
    count_pattern = c_count is not None and o_count is not None and j_count is not None and o_count > c_count and j_count < o_count
    fraction_pattern = c_fraction is not None and o_fraction is not None and j_fraction is not None and o_fraction > c_fraction and j_fraction < o_fraction
    lateral = _float(events.get("first_existing_expansion_ratio_onset"))
    peak_time = _float(count_peak.get("timestamp"))
    return {
        "run_id": run_id, "requested_seed": requested_seed, "seed_controlled": False,
        "simulation_frames": args.frames, "sample_count": len(timeline),
        "frontmost_crossing_time": _float(events.get("frontmost_boundary_crossing")),
        "front_center_crossing_time": _float(events.get("front_cohort_center_crossing")),
        "observed_center_crossing_time": _float(events.get("observed_cohort_center_crossing")),
        "surface_count_peak_time": peak_time, "surface_count_peak_value": _float(count_peak.get("forward_zero_neighbor_count")),
        "surface_fraction_peak_time": _float(fraction_peak.get("timestamp")), "surface_fraction_peak_value": _float(fraction_peak.get("forward_zero_neighbor_fraction")),
        "surface_peak_phase": count_peak.get("evaluation_only_sph_phase", ""),
        "lateral_ratio_gt_1_28_time": lateral, "lateral_dwell_onset_time": _float(events.get("first_positive_lateral_dwell")),
        "delta_t_peak_minus_lateral": None if peak_time is None or lateral is None else peak_time - lateral,
        "count_rise_peak_fall": count_pattern, "fraction_rise_peak_fall": fraction_pattern,
        "corridor_surface_fraction_mean": _float(c.get("surface_fraction_mean")),
        "opening_surface_fraction_mean": _float(o.get("surface_fraction_mean")),
        "junction_surface_fraction_mean": _float(j.get("surface_fraction_mean")),
        "opening_minus_corridor_fraction": None if c_fraction is None or o_fraction is None else o_fraction - c_fraction,
        "junction_minus_opening_fraction": None if o_fraction is None or j_fraction is None else j_fraction - o_fraction,
        "peak_front_center_boundary_distance": _float(count_peak.get("front_center_boundary_distance")),
        "peak_observed_center_boundary_distance": _float(count_peak.get("observed_center_boundary_distance")),
        "peak_frontmost_boundary_distance": _float(count_peak.get("frontmost_boundary_distance")),
        "event_order_pattern": "LATERAL_BEFORE_SURFACE_PEAK" if lateral is not None and peak_time is not None and lateral < peak_time else "SURFACE_PEAK_BEFORE_LATERAL" if lateral is not None and peak_time is not None else "MISSING_EVENT",
    }


def _write_summary(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "local_front_surface_multiseed_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["run_id"]); writer.writeheader(); writer.writerows(rows)
    valid = [r for r in rows if r["front_center_crossing_time"] is not None and r["surface_count_peak_time"] is not None]
    delta = _stats([r["delta_t_peak_minus_lateral"] for r in valid])
    summary = {
        "run_count": len(rows), "valid_run_count": len(valid), "seed_controlled": False,
        "count_rise_peak_fall_fraction": sum(r["count_rise_peak_fall"] for r in valid) / max(len(valid), 1),
        "fraction_rise_peak_fall_fraction": sum(r["fraction_rise_peak_fall"] for r in valid) / max(len(valid), 1),
        "opening_gt_corridor_fraction": sum((r["opening_minus_corridor_fraction"] or 0) > 0 for r in valid) / max(len(valid), 1),
        "junction_lt_opening_fraction": sum((r["junction_minus_opening_fraction"] or 0) < 0 for r in valid) / max(len(valid), 1),
        "lateral_before_peak_fraction": sum(r["event_order_pattern"] == "LATERAL_BEFORE_SURFACE_PEAK" for r in valid) / max(len(valid), 1),
        "peak_before_lateral_fraction": sum(r["event_order_pattern"] == "SURFACE_PEAK_BEFORE_LATERAL" for r in valid) / max(len(valid), 1),
        "surface_peak_phase_opening_fraction": sum(r["surface_peak_phase"] == "SPH_OPENING_APPROACH" for r in valid) / max(len(valid), 1),
    }
    summary.update({f"delta_t_peak_minus_lateral_{k}": v for k, v in delta.items()})
    peak_pos = _stats([r["peak_front_center_boundary_distance"] for r in valid])
    summary.update({f"peak_front_center_boundary_distance_{k}": v for k, v in peak_pos.items()})
    with (output / "local_front_surface_multiseed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["metric", "value"]); writer.writerows(summary.items())
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    axes[0, 0].hist([r["surface_fraction_peak_value"] for r in valid], bins=max(1, min(10, len(valid))), color="tab:blue"); axes[0, 0].set_title("surface fraction peak")
    axes[0, 1].hist([r["peak_front_center_boundary_distance"] for r in valid if r["peak_front_center_boundary_distance"] is not None], bins=max(1, min(10, len(valid))), color="tab:green"); axes[0, 1].set_title("peak boundary-relative position")
    axes[1, 0].hist([r["delta_t_peak_minus_lateral"] for r in valid if r["delta_t_peak_minus_lateral"] is not None], bins=max(1, min(10, len(valid))), color="tab:orange"); axes[1, 0].set_title("peak time - lateral onset")
    for phase, color in (("corridor_surface_fraction_mean", "tab:blue"), ("opening_surface_fraction_mean", "tab:green"), ("junction_surface_fraction_mean", "tab:red")):
        axes[1, 1].hist([r[phase] for r in valid if r[phase] is not None], alpha=0.5, label=phase, color=color)
    axes[1, 1].legend(); axes[1, 1].set_title("phase surface fraction")
    axes[2, 0].bar(np.arange(len(valid)), [int(r["fraction_rise_peak_fall"]) for r in valid]); axes[2, 0].set_title("fraction rise-peak-fall")
    axes[2, 1].bar(np.arange(len(valid)), [int(r["event_order_pattern"] == "LATERAL_BEFORE_SURFACE_PEAK") for r in valid]); axes[2, 1].set_title("lateral before peak")
    fig.suptitle("Multi-seed surface peak audit (evaluation-only; seed control status recorded)"); fig.tight_layout(); fig.savefig(output / "local_front_surface_multiseed_audit.png", dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "junction_detection/integration/output/local_front_surface_multiseed")
    args = parser.parse_args()
    rows = [_run_one(index + 1, seed, args, args.output_dir / f"run_{index + 1:03d}") for index, seed in enumerate(args.seeds)]
    _write_summary(rows, args.output_dir)
    print(f"requested_seeds={args.seeds} runs={len(rows)} seed_controlled=False output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
