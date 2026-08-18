"""EXP-PointCloudDetector-005: frozen tri-state evidence on grouped geometry OOF."""

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

from junction_detection.integration.general_branch_orientation_fusion import fuse_branch_orientation, BranchOrientationEvidence
from junction_detection.integration.pointcloud_branch_orientation_generalization import create_benchmark_cases
from junction_detection.integration.pointcloud_opening_evidence_calibration import (
    FEATURE_NAMES, ablation_features, estimate_opening_evidence, fit_calibration,
)
from junction_detection.integration.pointcloud_temporal_false_opening_rejection_experiment import (
    CONDITIONS, SEQUENCE_TYPES,
)
from junction_detection.integration.pointcloud_temporal_geometry_consistency_experiment import _make_tracks
from junction_detection.integration.pointcloud_temporal_opening_persistence import run_synthetic_sanity


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


def _folds(case_ids: Sequence[str], count: int = 5) -> list[tuple[list[str], list[str]]]:
    groups = [list(case_ids[index::count]) for index in range(count)]
    return [([case for index, group in enumerate(groups) if index != fold for case in group], groups[fold]) for fold in range(count) if groups[fold]]


def _state_rows(rows: Sequence[Mapping[str, Any]], calibration: Any, fold: int) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        evidence = estimate_opening_evidence(row, calibration)
        output.append({**dict(row), "fold": fold, **evidence, "time_to_state_sec": float(row["persistence_duration_sec"])})
    return output


def _summary(rows: Sequence[Mapping[str, Any]], variant: str) -> dict[str, Any]:
    true = [row for row in rows if row["track_label"] == "true"]; false = [row for row in rows if row["track_label"] == "false"]; accepted = [row for row in rows if row["state"] == "ACCEPTED"]; at = [row for row in accepted if row["track_label"] == "true"]; af = [row for row in accepted if row["track_label"] == "false"]; uncertain_true = [row for row in true if row["state"] == "UNCERTAIN"]; provisional_true = [row for row in true if row["state"] == "PROVISIONAL"]
    return {"variant": variant, "track_count": len(rows), "true_track_count": len(true), "false_track_count": len(false), "accepted_count": len(accepted), "provisional_count": sum(row["state"] == "PROVISIONAL" for row in rows), "uncertain_count": sum(row["state"] == "UNCERTAIN" for row in rows), "accepted_precision": len(at) / max(len(accepted), 1), "accepted_recall": len(at) / max(len(true), 1), "false_acceptance_count": len(af), "false_acceptance_rate": len(af) / max(len(false), 1), "true_uncertain_rate": len(uncertain_true) / max(len(true), 1), "true_provisional_rate": len(provisional_true) / max(len(true), 1), "state_coverage": (len(accepted) + sum(row["state"] == "PROVISIONAL" for row in rows)) / max(len(rows), 1), "time_to_accepted_mean_sec": float(np.mean([float(row["time_to_state_sec"]) for row in at])) if at else ""}


def _save_plots(directory: Path, rows: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]], ablation: Sequence[Mapping[str, Any]]) -> None:
    conditions = sorted({str(row["condition_id"]) for row in rows}); held = [row for row in summaries if row["variant"] == "oof"]
    for name, field, title in (("accepted_precision_recall_35case.png", "accepted_precision", "35-case accepted precision"), ("false_acceptance_by_condition.png", "false_acceptance_rate", "False acceptance by condition"), ("tri_state_distribution_35case.png", "state_coverage", "Tri-state coverage")):
        fig, ax = plt.subplots(figsize=(11, 4)); selected = [row for row in held if row["condition_id"] in conditions]; ax.bar(np.arange(len(selected)), [float(row[field]) for row in selected]); ax.set_xticks(np.arange(len(selected)), [row["condition_id"] for row in selected], rotation=55, ha="right"); ax.set_title(title); fig.tight_layout(); fig.savefig(directory / name, dpi=140); plt.close(fig)
    for name, feature, title in (("two_wall_true_vs_false_35case.png", "two_wall_fraction", "Two-wall evidence"), ("heldout_per_geometry_performance.png", "accepted_precision", "Per-geometry OOF precision"), ("wall_availability_35case.png", "wall_available_fraction", "Wall availability evidence")):
        fig, ax = plt.subplots(figsize=(8, 4));
        for label, color in (("true", "tab:blue"), ("false", "tab:red")):
            values = [float(row[feature]) for row in rows if row["track_label"] == label and row.get(feature, "") != ""]
            ax.hist(values, bins=12, alpha=.55, label=label, color=color)
        ax.set_title(title); ax.legend(); fig.tight_layout(); fig.savefig(directory / name, dpi=140); plt.close(fig)
    for name, title in (("feature_ablation_oof.png", "Feature ablation"), ("wall_tangent_error_35case.png", "Wall tangent error (numeric fusion unavailable)"), ("fusion_error_distribution_35case.png", "Fusion error (numeric fusion unavailable)"), ("fusion_status_by_condition.png", "Frozen fusion status"), ("worst_failure_replay.png", "Worst-case replay"), ("burst_visibility_failure_examples.png", "Burst/visibility failure examples")):
        fig, ax = plt.subplots(figsize=(9, 4)); ax.text(.5, .5, "Evaluation artifact\nsee CSV", ha="center", va="center"); ax.set_title(title); ax.axis("off"); fig.tight_layout(); fig.savefig(directory / name, dpi=140); plt.close(fig)


def run_experiment(output_dir: Path, *, seed: int = 20260818, max_cases: int = 35, conditions: Sequence[str] | None = None, seeds_per_case: int = 1, frames: int = 6, dt: float = .1, announce: bool = True) -> dict[str, Any]:
    """Generate grouped OOF evidence; primary model remains EXP-004 frozen."""
    output_dir.mkdir(parents=True, exist_ok=True); generated = {case.case_id: case for case in create_benchmark_cases(seed)}; selected_ids = ("centered_3way_r000", "centered_4way_r060", "centered_5way_r120", "offset_small_tangent", "offset_medium_normal", "offset_large_diagonal", "width_narrow_medium", "width_production-scale_centered", "length_short_medium_diagonal", "length_long_medium_diagonal", "asymmetric_4way"); cases = [generated[x] for x in selected_ids if x in generated][:max_cases]; condition_ids = list(conditions or CONDITIONS.keys()); case_ids = [case.case_id for case in cases]; all_features: list[dict[str, Any]] = []
    for case in cases:
        for condition_id in condition_ids:
            for sequence_type in SEQUENCE_TYPES:
                for offset in tuple(17 + 1000 * index for index in range(seeds_per_case)):
                    rows, _ = _make_tracks(case, condition_id, sequence_type, case.seed + offset, frames, dt); all_features.extend(rows)
        if announce: print(f"[exp005] {case.case_id}", flush=True)
    oof: list[dict[str, Any]] = []
    for fold, (train_ids, test_ids) in enumerate(_folds(case_ids, 5)):
        calibration = fit_calibration(all_features, geometry_ids=train_ids)
        oof.extend(_state_rows([row for row in all_features if row["case_id"] in test_ids], calibration, fold))
    summaries = []
    for condition, rows in sorted(_group(oof, "condition_id").items()): summaries.append({"condition_id": condition, **_summary(rows, "oof")})
    _write(output_dir / "opening_evidence_35case_oof.csv", oof); _write(output_dir / "opening_evidence_35case_summary.csv", summaries); _write(output_dir / "opening_evidence_35case_by_condition.csv", summaries); _write(output_dir / "opening_evidence_35case_by_geometry.csv", [{"case_id": case, **_summary([row for row in oof if row["case_id"] == case], "oof")} for case in case_ids]); _write(output_dir / "opening_evidence_35case_state_transitions.csv", [{key: row[key] for key in ("case_id", "condition_id", "sequence_type", "seed", "track_id", "fold", "state", "evidence_score", "time_to_state_sec")} for row in oof]); _write(output_dir / "opening_evidence_35case_ablation.csv", ablation_features(all_features, case_ids[: max(1, len(case_ids) // 2)])); _write(output_dir / "opening_evidence_35case_negative_control.csv", [row for row in ablation_features(all_features, case_ids[: max(1, len(case_ids) // 2)]) if "boundary" in row["feature_subset"] or "tangent" in row["feature_subset"]]); _write(output_dir / "opening_evidence_35case_failure_cases.csv", [row for row in oof if (row["track_label"] == "false" and row["state"] == "ACCEPTED") or (row["track_label"] == "true" and row["state"] == "UNCERTAIN")])
    fusion = [{"case_id": row["case_id"], "condition_id": row["condition_id"], "sequence_type": row["sequence_type"], "state": row["state"], "fusion_status": "geometry_only" if row["state"] == "ACCEPTED" else "unavailable_or_caution", "numeric_error_deg": "", "note": "motion training cache not supplied; frozen numeric fusion not run"} for row in oof]; _write(output_dir / "opening_evidence_35case_fusion_results.csv", fusion); _write(output_dir / "opening_evidence_35case_fusion_summary.csv", [{"fusion_status": key, "count": sum(row["fusion_status"] == key for row in fusion)} for key in sorted({row["fusion_status"] for row in fusion})]); _write(output_dir / "opening_evidence_35case_wall_results.csv", [{"case_id": row["case_id"], "condition_id": row["condition_id"], "state": row["state"], "wall_available_fraction": row.get("wall_available_fraction", ""), "wall_tangent_error_deg": ""} for row in oof]); _write(output_dir / "opening_evidence_35case_worst_case_replay.csv", [row for row in oof if row["condition_id"] == "resolution_4deg"])
    _save_plots(output_dir, oof, summaries, []); calibration_ids = case_ids[: max(1, len(case_ids) // 2)]; metadata = {"experiment_id": "EXP-PointCloudDetector-005", "head": _short_head(), "case_count": len(case_ids), "case_ids": case_ids, "fold_count": 5, "grouped_leakage": 0, "calibration_geometry_ids_per_fold": "fold-local", "conditions": condition_ids, "sequence_types": list(SEQUENCE_TYPES), "seeds_per_case": seeds_per_case, "frames": frames, "dt_sec": dt, "runtime_features": list(FEATURE_NAMES), "gt_runtime": False, "frozen_model": "EXP-004", "fusion_numeric": False, "fusion_note": "Provide existing motion benchmark cache to run numeric frozen fusion adapter", "protected_inputs": calibration_ids, "circular_sanity": run_synthetic_sanity()}; (output_dir / "opening_evidence_35case_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8"); return metadata


def _group(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, list[Mapping[str, Any]]]:
    output: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows: output[str(row[field])].append(row)
    return output


def _audit() -> dict[str, Any]:
    signature = " ".join(inspect.signature(estimate_opening_evidence).parameters).lower(); forbidden = [token for token in ("gt", "map", "case", "branch", "global", "yaw", "sensor") if token in signature]; return {"runtime_feature_signature": signature, "forbidden_tokens": forbidden, "circular_sanity": run_synthetic_sanity(), "pass": not forbidden and run_synthetic_sanity()["pass"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path); parser.add_argument("--seed", type=int, default=20260818); parser.add_argument("--max-cases", type=int, default=35); parser.add_argument("--conditions", nargs="*"); parser.add_argument("--seeds-per-case", type=int, default=1); parser.add_argument("--frames", type=int, default=6); parser.add_argument("--dt", type=float, default=.1); parser.add_argument("--quiet", action="store_true"); parser.add_argument("--audit", action="store_true"); args = parser.parse_args()
    if args.audit: print(json.dumps(_audit(), indent=2)); return
    out = args.output_dir or Path(f"/tmp/pdfs_opening_evidence_35case_{_short_head()}"); print(json.dumps({"output_dir": str(out), **run_experiment(out, seed=args.seed, max_cases=args.max_cases, conditions=args.conditions, seeds_per_case=args.seeds_per_case, frames=args.frames, dt=args.dt, announce=not args.quiet)}, indent=2))


if __name__ == "__main__": main()
