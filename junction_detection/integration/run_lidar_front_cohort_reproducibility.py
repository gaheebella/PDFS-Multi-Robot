"""Repeated evaluation runner for the fixed FRONT_COHORT diagnostic.

This is evaluation-only.  The current simulator has no explicit seed
injection path, so runs are labelled ``uncontrolled`` unless a future
simulator exposes one.  The runner never claims uncontrolled repetitions are
independent seeds.
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

ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC = ROOT / "pygame_simulator/single_junction_sph_dfs_lidar_front_trigger_diagnostics.py"


def _value(row, key):
    value = row.get(key, "")
    return None if value in ("", None) else float(value)


def _stats(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {name: "" for name in ("mean", "median", "std", "min", "max", "p10", "p90")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def _event_map(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["event"]: row["timestamp_or_delta_s"] for row in csv.DictReader(handle)}


def _front_events(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next((r for r in rows if r["cohort_name"] == "FRONT_COHORT"), {})
    return row


def _run_one(run_id, args, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "SDL_VIDEODRIVER": "dummy",
        "SPH_DFS_MAX_FRAMES": str(args.frames),
        "SPH_DFS_DIAGNOSTIC_OUTPUT": str(run_dir),
    })
    # No simulator seed mechanism was found; do not inject a fake seed.
    subprocess.run([sys.executable, str(DIAGNOSTIC)], cwd=ROOT, env=env, check=True)
    events = _event_map(run_dir / "lidar_front_trigger_event_summary.csv")
    front = _front_events(run_dir / "lidar_front_trigger_cohort_events.csv")
    timeline = run_dir / "lidar_front_trigger_timeline.csv"
    with timeline.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts = [float(r["front_cohort_robot_count"]) for r in rows]
    corridor = [r for r in rows if r["evaluation_only_sph_phase"] == "SPH_CORRIDOR"]
    ratio11 = sum(float(r["front_cohort_expansion_ratio"]) > 1.1 for r in corridor)
    ratio128 = sum(float(r["front_cohort_expansion_ratio"]) > 1.28 for r in corridor)
    dwell = sum(float(r["front_cohort_expansion_dwell"]) > 0.0 for r in corridor)
    sustained = sum(r["front_cohort_sustained_marker"] == "True" for r in corridor)
    sustained_time = _value(front, "sustained_onset")
    front_center = _value(events, "front_cohort_center_crossing")
    frontmost = _value(events, "frontmost_boundary_crossing")
    return {
        "run_id": run_id,
        "seed": "uncontrolled",
        "seed_controlled": False,
        "simulation_frames": args.frames,
        "simulation_time_s": float(args.frames) / 60.0,
        "diagnostic_sampling_interval_s": 0.1,
        "initial_diagnostic_offset_y": os.environ.get("SPH_DFS_DIAGNOSTIC_START_OFFSET_Y", "20.0"),
        "front_quantile": "0.68",
        "lateral_expansion_ratio_threshold": "1.28",
        "front_cohort_mean_robot_count": float(np.mean(counts)) if counts else "",
        "frontmost_crossing": frontmost,
        "front_center_crossing": front_center,
        "ratio_onset": _value(front, "ratio_onset"),
        "positive_dwell_onset": _value(front, "positive_dwell_onset"),
        "sustained_onset": sustained_time,
        "delta_t_frontmost": None if sustained_time is None or frontmost is None else sustained_time - frontmost,
        "delta_t_front_center": None if sustained_time is None or front_center is None else sustained_time - front_center,
        "junction_onset_detected": sustained_time is not None,
        "front_center_crossing_detected": front_center is not None,
        "frontmost_crossing_detected": frontmost is not None,
        "corridor_ratio_gt_1_1_count": ratio11,
        "corridor_ratio_gt_1_28_count": ratio128,
        "corridor_positive_dwell_count": dwell,
        "corridor_sustained_count": sustained,
        "corridor_false_positive": bool(ratio128 or dwell or sustained),
    }


def _write_outputs(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["run_id"]
    with (output / "front_cohort_reproducibility_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    valid = [r for r in rows if r["junction_onset_detected"] and r["front_center_crossing_detected"]]
    onset = [r for r in rows if r["junction_onset_detected"]]
    false_runs = [r for r in rows if r["corridor_false_positive"]]
    summary = {
        "total_runs": len(rows), "valid_runs": len(valid),
        "junction_onset_detection_count": len(onset),
        "junction_onset_detection_rate": len(onset) / max(len(rows), 1),
        "missed_onset_count": len(rows) - len(onset),
        "missed_onset_rate": (len(rows) - len(onset)) / max(len(rows), 1),
        "corridor_false_positive_run_count": len(false_runs),
        "corridor_false_positive_run_rate": len(false_runs) / max(len(rows), 1),
        "front_cohort_mean_robot_count_mean": float(np.mean([r["front_cohort_mean_robot_count"] for r in rows])) if rows else "",
        "front_cohort_mean_robot_count_std": float(np.std([r["front_cohort_mean_robot_count"] for r in rows])) if rows else "",
    }
    for name, values in (("delta_t_front_center", [r["delta_t_front_center"] for r in valid]), ("delta_t_frontmost", [r["delta_t_frontmost"] for r in valid])):
        summary.update({f"{name}_{key}": value for key, value in _stats(values).items()})
    with (output / "front_cohort_reproducibility_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["metric", "value"]); writer.writerows(summary.items())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    indices = np.arange(1, len(rows) + 1)
    axes[0, 0].hist([r["delta_t_front_center"] for r in valid if r["delta_t_front_center"] is not None], bins=min(10, max(1, len(valid))), color="tab:blue")
    axes[0, 0].set_title("Δt front center")
    axes[0, 1].scatter([r["front_center_crossing"] for r in rows], [r["sustained_onset"] for r in rows])
    axes[0, 1].set_xlabel("front center crossing"); axes[0, 1].set_ylabel("sustained onset")
    axes[1, 0].bar(indices, [r["corridor_ratio_gt_1_28_count"] for r in rows]); axes[1, 0].set_title("corridor ratio > 1.28 count")
    axes[1, 1].plot(indices, [r["front_cohort_mean_robot_count"] for r in rows], "o-"); axes[1, 1].set_title("FRONT_COHORT mean robot count")
    fig.suptitle("FRONT_COHORT reproducibility (evaluation-only)"); fig.tight_layout()
    fig.savefig(output / "front_cohort_reproducibility.png", dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "junction_detection/integration/output/lidar_front_cohort_reproducibility")
    args = parser.parse_args()
    rows = []
    for run_id in range(1, args.runs + 1):
        rows.append(_run_one(run_id, args, args.output_dir / f"run_{run_id:03d}"))
    _write_outputs(rows, args.output_dir)
    print(f"runs={len(rows)} seed_controlled=False output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
