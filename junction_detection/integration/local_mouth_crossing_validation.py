"""Aggregate two local-mouth-sampling simulator runs into A/B artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


TABLES = (
    "local_mouth_crossing_samples.csv",
    "local_mouth_crossing_branch_summary.csv",
    "local_mouth_crossing_pca_comparison.csv",
    "local_mouth_crossing_handoff_comparison.csv",
    "local_mouth_crossing_run_summary.csv",
)
BRANCHES = ("LEFT", "UP", "RIGHT")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(baseline_dir: Path, local_dir: Path, output_dir: Path) -> None:
    """Write five combined CSVs and five comparison figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = {name: _read(baseline_dir / name) for name in TABLES}
    local = {name: _read(local_dir / name) for name in TABLES}
    # Sampling and PCA tables from B contain matched A/B/GT values from the
    # exact descriptor population used by that run.  Outcome/run tables need
    # both actual executions.
    _write(output_dir / TABLES[0], local[TABLES[0]])
    _write(output_dir / TABLES[1], local[TABLES[1]])
    _write(output_dir / TABLES[2], local[TABLES[2]])
    _write(output_dir / TABLES[3], baseline[TABLES[3]] + local[TABLES[3]])
    _write(output_dir / TABLES[4], baseline[TABLES[4]] + local[TABLES[4]])
    _plots(local, baseline, output_dir)


def _plots(
    local: dict[str, list[dict[str, str]]],
    baseline: dict[str, list[dict[str, str]]],
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    samples = local[TABLES[0]]
    right = [row for row in samples if row["branch"] == "RIGHT"]
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.scatter([float(row["gt_lateral"]) for row in right], [float(row["heading_axial"]) for row in right], s=22, label="heading origin")
    axis.scatter([float(row["gt_lateral"]) for row in right], [float(row["local_axial"]) for row in right], s=22, label="local proxy")
    axis.scatter([float(row["gt_lateral"]) for row in right], [float(row["gt_axial"]) for row in right], s=18, color="black", label="GT crossing")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(xlabel="GT-local lateral", ylabel="GT-local axial", title="RIGHT sampling clouds")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "production_vs_local_vs_gt_crossing_right.png", dpi=170)
    plt.close(figure)

    pca = {row["branch"]: row for row in local[TABLES[2]]}
    x = np.arange(len(BRANCHES))
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(x - 0.25, [float(pca[b]["heading_error_deg"]) for b in BRANCHES], 0.25, label="baseline")
    axis.bar(x, [float(pca[b]["local_error_deg"]) for b in BRANCHES], 0.25, label="local proxy")
    axis.bar(x + 0.25, [0.0] * len(BRANCHES), 0.25, label="GT")
    axis.set(xticks=x, xticklabels=BRANCHES, ylabel="absolute yaw error [deg]", title="PCA yaw before/after")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "pca_yaw_before_after.png", dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    positions = np.arange(len(BRANCHES))
    heading_data = [[float(row["heading_axial_error"]) for row in samples if row["branch"] == branch] for branch in BRANCHES]
    local_data = [[float(row["local_axial_error"]) for row in samples if row["branch"] == branch] for branch in BRANCHES]
    axis.boxplot(heading_data, positions=positions - 0.18, widths=0.30, patch_artist=True, boxprops={"facecolor": "tab:blue", "alpha": 0.55})
    axis.boxplot(local_data, positions=positions + 0.18, widths=0.30, patch_artist=True, boxprops={"facecolor": "tab:orange", "alpha": 0.55})
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(xticks=positions, xticklabels=BRANCHES, ylabel="axial error [px]", title="Sampling axial error: blue=heading, orange=local")
    figure.tight_layout()
    figure.savefig(output_dir / "sampling_axial_error_before_after.png", dpi=170)
    plt.close(figure)

    handoff = baseline[TABLES[3]] + local[TABLES[3]]
    right_by_mode = {row["mode"]: row for row in handoff if row["branch"] == "RIGHT"}
    figure, axis = plt.subplots(figsize=(7, 5))
    modes = ("heading_origin", "local_crossing")
    safe = [int(right_by_mode[mode]["maximum_safe_slot_count"]) for mode in modes]
    required = [int(right_by_mode[mode]["required_slot_count"]) for mode in modes]
    axis.bar(("baseline", "local proxy"), required, color="0.85", label="required")
    axis.bar(("baseline", "local proxy"), safe, color=("tab:blue", "tab:orange"), label="walkable")
    for index, value in enumerate(safe):
        axis.text(index, value - 1.0, f"{value}/{required[index]}", ha="center", color="white", fontweight="bold")
    axis.set(ylim=(0, max(required) + 2), ylabel="slot count", title="RIGHT handoff row occupancy")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "right_handoff_row_before_after.png", dpi=170)
    plt.close(figure)

    outcomes = {(row["mode"], row["branch"]): row["success"] == "True" for row in handoff}
    figure, axis = plt.subplots(figsize=(9, 5))
    baseline_values = [int(outcomes.get(("heading_origin", branch), False)) for branch in BRANCHES]
    local_values = [int(outcomes.get(("local_crossing", branch), False)) for branch in BRANCHES]
    axis.bar(x - 0.18, baseline_values, 0.36, label="baseline")
    axis.bar(x + 0.18, local_values, 0.36, label="local proxy")
    axis.set(xticks=x, xticklabels=BRANCHES, yticks=(0, 1), yticklabels=("failed/not reached", "success"), title="Handoff outcome before/after")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "handoff_outcome_before_after.png", dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.baseline_dir, args.local_dir, args.output_dir)
    print(f"local mouth crossing A/B artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
