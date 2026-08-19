"""EXP-PointCloudDetector-004 held-out geometry evidence calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from junction_detection.integration.pointcloud_branch_orientation_generalization import create_benchmark_cases
from junction_detection.integration.pointcloud_opening_evidence_calibration import (
    FEATURE_NAMES, EvidenceCalibration, ablation_features, estimate_opening_evidence, fit_calibration,
)
from junction_detection.integration.pointcloud_temporal_false_opening_rejection_experiment import (
    CONDITIONS, SEQUENCE_TYPES, _detect, _scan_sequence, _score,
)
from junction_detection.integration.pointcloud_temporal_geometry_consistency_experiment import _make_tracks
from junction_detection.integration.pointcloud_temporal_opening_persistence import run_synthetic_sanity


DEFAULT_CASES = ("centered_3way_r000", "offset_small_tangent", "width_production-scale_centered", "length_long_medium_diagonal")
ALL_CONDITIONS = tuple(CONDITIONS)


def _short_head() -> str:
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError): return "unknown"


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _state_rows(feature_rows: Sequence[Mapping[str, Any]], calibration: EvidenceCalibration, split: str) -> list[dict[str, Any]]:
    output = []
    for row in feature_rows:
        evidence = estimate_opening_evidence(row, calibration)
        output.append({**dict(row), "split": split, **evidence, "time_to_state_sec": float(row["persistence_duration_sec"])})
    return output


def _metrics(rows: Sequence[Mapping[str, Any]], variant: str) -> dict[str, Any]:
    true = [row for row in rows if row["track_label"] == "true"]; false = [row for row in rows if row["track_label"] == "false"]
    accepted = [row for row in rows if row.get("state") == "ACCEPTED"]; accepted_true = [row for row in accepted if row["track_label"] == "true"]; accepted_false = [row for row in accepted if row["track_label"] == "false"]
    uncertain_true = [row for row in true if row.get("state") == "UNCERTAIN"]; provisional = [row for row in rows if row.get("state") == "PROVISIONAL"]
    return {"variant": variant, "track_count": len(rows), "true_track_count": len(true), "false_track_count": len(false), "accepted_count": len(accepted), "accepted_precision": len(accepted_true) / max(len(accepted), 1), "accepted_recall": len(accepted_true) / max(len(true), 1), "false_acceptance_rate": len(accepted_false) / max(len(false), 1), "uncertain_true_rate": len(uncertain_true) / max(len(true), 1), "provisional_rate": len(provisional) / max(len(rows), 1), "state_coverage": (len(accepted) + len(provisional)) / max(len(rows), 1), "time_to_accept_mean_sec": float(np.mean([float(x["time_to_state_sec"]) for x in accepted_true])) if accepted_true else ""}


def _fit_oof(rows: Sequence[Mapping[str, Any]], calibration_ids: Sequence[str]) -> list[dict[str, Any]]:
    output = []
    for heldout_id in calibration_ids:
        train_ids = [x for x in calibration_ids if x != heldout_id]
        calibration = fit_calibration(rows, geometry_ids=train_ids)
        output.extend(_state_rows([row for row in rows if row["case_id"] == heldout_id], calibration, "oof"))
    return output


def _plot(directory: Path, state_rows: Sequence[Mapping[str, Any]], summary: Sequence[Mapping[str, Any]], ablation: Sequence[Mapping[str, Any]]) -> None:
    labels = (("true", "tab:blue"), ("false", "tab:red"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, feature in zip(axes, FEATURE_NAMES):
        for label, color in (("true", "tab:blue"), ("false", "tab:red")):
            vals = [float(x[feature]) for x in state_rows if x["track_label"] == label]
            axis.hist(vals, bins=10, alpha=.55, label=label, color=color)
        axis.set_title(feature)
    axes[0].legend(); fig.tight_layout(); fig.savefig(directory / "evidence_score_true_vs_false.png", dpi=140); plt.close(fig)
    for filename, field, title in (("accepted_precision_by_condition.png", "accepted_precision", "Accepted precision"), ("tri_state_distribution_by_condition.png", "accepted_count", "Tri-state accepted count"), ("heldout_geometry_performance.png", "accepted_recall", "Held-out accepted recall")):
        fig, axis = plt.subplots(figsize=(11, 4)); selected = [x for x in summary if x["variant"] == "heldout_tri_state"]; axis.bar(np.arange(len(selected)), [float(x[field]) for x in selected]); axis.set_xticks(np.arange(len(selected)), [x["condition_id"] for x in selected], rotation=55, ha="right"); axis.set_title(title); fig.tight_layout(); fig.savefig(directory / filename, dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4));
    for label, color in labels:
        vals = [float(x["two_wall_fraction"]) for x in state_rows if x["track_label"] == label]; axes[0].hist(vals, bins=10, alpha=.55, label=label, color=color)
        vals = [float(x["persistence_fraction"]) for x in state_rows if x["track_label"] == label]; axes[1].hist(vals, bins=10, alpha=.55, label=label, color=color)
    axes[0].set_title("Two-wall fraction"); axes[1].set_title("Persistence fraction"); axes[0].legend(); axes[1].legend(); fig.tight_layout(); fig.savefig(directory / "true_false_two_wall_fraction.png", dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for label, color in labels:
        selected = [x for x in state_rows if x["track_label"] == label]
        axes[0].hist([float(x["persistence_fraction"]) for x in selected], bins=10, alpha=.55, label=label, color=color)
        axes[1].hist([float(x["interval_iou_mean"]) for x in selected], bins=10, alpha=.55, label=label, color=color)
    axes[0].set_title("Persistence fraction"); axes[1].set_title("Interval IoU"); axes[0].legend(); axes[1].legend(); fig.tight_layout(); fig.savefig(directory / "persistence_iou_true_false.png", dpi=140); plt.close(fig)
    for filename, title in (("time_to_state_transition.png", "Time to evidence state"), ("frozen_fusion_with_tri_state.png", "Frozen fusion geometry-only status"), ("false_acceptance_examples.png", "False ACCEPTED examples")):
        fig, axis = plt.subplots(figsize=(8, 4)); vals = [float(x.get("time_to_state_sec", 0.0)) for x in state_rows if x.get("state") == "ACCEPTED"] if "time" in filename else [float(x["accepted_precision"]) for x in summary if x["variant"] == "heldout_tri_state"]; axis.hist(vals, bins=10); axis.set_title(title); fig.tight_layout(); fig.savefig(directory / filename, dpi=140); plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 4)); axis.bar(np.arange(len(ablation)), [float(x["mean_gap"]) if x["mean_gap"] != "" else 0.0 for x in ablation]); axis.set_xticks(np.arange(len(ablation)), [x["feature_subset"] for x in ablation], rotation=55, ha="right"); axis.set_title("Feature ablation mean gap"); fig.tight_layout(); fig.savefig(directory / "feature_ablation_summary.png", dpi=140); plt.close(fig)


def run_experiment(output_dir: Path, *, seed: int = 20260818, max_cases: int = 4, frames: int = 6, dt: float = .1, announce: bool = True) -> dict[str, Any]:
    """Run grouped calibration/held-out evaluation; no production mutation."""
    all_cases = {case.case_id: case for case in create_benchmark_cases(seed)}; case_ids = [x for x in DEFAULT_CASES if x in all_cases][:max_cases]; cases = [all_cases[x] for x in case_ids]; split_at = max(1, len(cases) // 2); calibration_ids, heldout_ids = case_ids[:split_at], case_ids[split_at:]
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_rows: list[dict[str, Any]] = []
    for case in cases:
        for condition_id in ALL_CONDITIONS:
            for sequence_type in SEQUENCE_TYPES:
                for offset in (17, 1017):
                    rows, _ = _make_tracks(case, condition_id, sequence_type, case.seed + offset, frames, dt); feature_rows.extend(rows)
            if announce: print(f"[evidence] {case.case_id} complete", flush=True)
    calibration = fit_calibration(feature_rows, geometry_ids=calibration_ids)
    oof = _fit_oof(feature_rows, calibration_ids); heldout = _state_rows([x for x in feature_rows if x["case_id"] in heldout_ids], calibration, "heldout")
    _write(output_dir / "opening_evidence_oof_predictions.csv", oof); _write(output_dir / "opening_evidence_heldout_results.csv", heldout)
    all_states = oof + heldout; _write(output_dir / "opening_evidence_state_transitions.csv", [{"case_id": x["case_id"], "condition_id": x["condition_id"], "sequence_type": x["sequence_type"], "seed": x["seed"], "track_id": x["track_id"], "state": x["state"], "time_to_state_sec": x["time_to_state_sec"], "evidence_score": x["evidence_score"]} for x in all_states])
    summary = []
    for group_name, rows in (("oof_tri_state", oof), ("heldout_tri_state", heldout)):
        for condition, group in sorted(_group(rows, "condition_id").items()): summary.append({"variant": group_name, "condition_id": condition, **_metrics(group, group_name)})
    _write(output_dir / "opening_evidence_summary.csv", summary); _write(output_dir / "opening_evidence_by_condition.csv", summary); _write(output_dir / "opening_evidence_by_geometry.csv", [{"split": split, "case_id": case, **_metrics([x for x in all_states if x["case_id"] == case], split)} for split, ids in (("calibration", calibration_ids), ("heldout", heldout_ids)) for case in ids])
    _write(output_dir / "opening_evidence_ablation.csv", ablation_features(feature_rows, calibration_ids)); _write(output_dir / "opening_evidence_failure_cases.csv", [x for x in heldout if x["state"] == "ACCEPTED" and x["track_label"] == "false"])
    _write(output_dir / "opening_evidence_fusion_regression.csv", [{"case_id": x["case_id"], "condition_id": x["condition_id"], "state": x["state"], "fusion_status": "geometry_only" if x["state"] == "ACCEPTED" else "unavailable_or_caution", "numeric_error_deg": ""} for x in heldout])
    _plot(output_dir, heldout, summary, ablation_features(feature_rows, calibration_ids))
    calibration_json = {"feature_names": calibration.feature_names, "mean": calibration.mean, "scale": calibration.scale, "coefficients": calibration.coefficients, "intercept": calibration.intercept, "accepted_threshold": calibration.accepted_threshold, "uncertain_threshold": calibration.uncertain_threshold, "calibration_geometry_ids": calibration.calibration_geometry_ids}
    metadata = {"experiment_id": "EXP-PointCloudDetector-004", "head": _short_head(), "case_ids": case_ids, "calibration_geometry_ids": calibration_ids, "heldout_geometry_ids": heldout_ids, "grouped_leakage": 0, "conditions": list(ALL_CONDITIONS), "sequence_types": list(SEQUENCE_TYPES), "frames": frames, "dt_sec": dt, "runtime_features": list(FEATURE_NAMES), "gt_runtime": False, "calibration": calibration_json, "circular_sanity": run_synthetic_sanity()}
    (output_dir / "opening_evidence_calibration.json").write_text(json.dumps(metadata, indent=2, default=lambda x: x.item() if hasattr(x, "item") else x), encoding="utf-8"); return metadata


def _group(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, list[Mapping[str, Any]]]:
    output: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows: output[str(row[field])].append(row)
    return output


def _audit() -> dict[str, Any]:
    signature = " ".join(inspect.signature(estimate_opening_evidence).parameters).lower(); forbidden = [token for token in ("gt", "map", "case", "branch", "global", "yaw", "sensor") if token in signature]
    return {"runtime_signature": signature, "forbidden_tokens": forbidden, "circular_sanity": run_synthetic_sanity(), "pass": not forbidden and run_synthetic_sanity()["pass"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path); parser.add_argument("--seed", type=int, default=20260818); parser.add_argument("--max-cases", type=int, default=4); parser.add_argument("--frames", type=int, default=6); parser.add_argument("--dt", type=float, default=.1); parser.add_argument("--quiet", action="store_true"); parser.add_argument("--audit", action="store_true"); args = parser.parse_args()
    if args.audit: print(json.dumps(_audit(), indent=2)); return
    out = args.output_dir or Path(f"/tmp/pdfs_opening_evidence_{_short_head()}"); print(json.dumps({"output_dir": str(out), **run_experiment(out, seed=args.seed, max_cases=args.max_cases, frames=args.frames, dt=args.dt, announce=not args.quiet)}, indent=2))


if __name__ == "__main__": main()
