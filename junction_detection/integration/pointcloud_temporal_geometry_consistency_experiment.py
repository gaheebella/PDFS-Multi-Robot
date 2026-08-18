"""Evaluation-only LiDAR multi-scan opening geometry consistency pilot.

The experiment does not add a production acceptance threshold.  It records
whether local opening, boundary, wall, support, and width observables separate
post-hoc true and false tracks.  Ground truth is used only after runtime
feature extraction for scoring.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from junction_detection.integration.pointcloud_branch_orientation_generalization import (
    _gt_opening, angular_error_deg, create_benchmark_cases, wall_segments,
)
from junction_detection.integration.pointcloud_temporal_false_opening_rejection_experiment import (
    CONDITIONS, SEQUENCE_TYPES, _detect, _scan_sequence, _positive_overlap_matches,
)
from junction_detection.integration.pointcloud_temporal_opening_persistence import (
    TemporalOpeningPersistence, TemporalPersistenceConfig, circular_angle_distance,
    circular_interval_iou, run_synthetic_sanity,
)
from junction_detection.integration.pointcloud_wall_orientation_sensor_robustness import _candidate_diagnostics
from junction_detection.integration.pointcloud_wall_parallel_orientation import estimate_wall_parallel_tangent


PILOT_CASE_IDS = (
    "centered_3way_r000", "offset_small_tangent", "width_production-scale_centered", "centered_4way_r060",
)
PILOT_CONDITIONS = (
    "clean", "dropout_0.05", "dropout_0.15", "visibility_0.90", "visibility_0.70",
    "noise_0.08", "resolution_4deg", "occlusion_0.40",
)


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _short_head() -> str:
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError): return "unknown"


def _axial_std(values: Sequence[float]) -> float:
    """Circular standard deviation for line orientations modulo 180 degrees."""
    if len(values) < 2: return 0.0
    radians = np.deg2rad(np.asarray(values, dtype=float) * 2.0)
    resultant = float(np.hypot(np.mean(np.cos(radians)), np.mean(np.sin(radians))))
    return float(np.rad2deg(np.sqrt(max(0.0, -2.0 * np.log(max(resultant, 1e-12)))) / 2.0))


def _track_features(track: Mapping[str, Any], frame_count: int) -> dict[str, Any]:
    intervals = track["intervals"]
    starts = [x["start_angle"] for x in intervals]; ends = [x["end_angle"] for x in intervals]
    widths = [x["width_deg"] for x in intervals]
    tangents = [x["wall_tangent_deg"] for x in intervals if x["wall_tangent_deg"] != ""]
    mouth = [x["mouth_width_m"] for x in intervals if x["mouth_width_m"] != ""]
    spans = [x["wall_span_m"] for x in intervals if x["wall_span_m"] != ""]
    ious = [circular_interval_iou(intervals[i - 1], intervals[i]) for i in range(1, len(intervals))]
    boundary = []
    for series in (starts, ends):
        if series:
            ref = series[0]; boundary.append(float(np.std([ref + ((v - ref + 180) % 360 - 180) for v in series])))
    modes = [x["wall_mode"] for x in intervals]
    return {
        "track_id": track["track_id"], "observation_count": len(intervals), "persistence_fraction": len(intervals) / max(frame_count, 1),
        "interval_iou_mean": "" if not ious else float(np.mean(ious)), "boundary_std_deg": max(boundary, default=0.0),
        "width_std_deg": float(np.std(widths)) if len(widths) > 1 else 0.0,
        "wall_tangent_count": len(tangents), "wall_tangent_axial_std_deg": _axial_std(tangents),
        "two_wall_fraction": sum(mode == "two_wall_parallel" for mode in modes) / max(len(modes), 1),
        "wall_available_fraction": sum(not mode.startswith("unavailable") for mode in modes) / max(len(modes), 1),
        "wall_span_mean_m": "" if not spans else float(np.mean(spans)), "wall_span_std_m": "" if len(spans) < 2 else float(np.std(spans)),
        "mouth_width_mean_m": "" if not mouth else float(np.mean(mouth)), "mouth_width_std_m": "" if len(mouth) < 2 else float(np.std(mouth)),
        "wall_disagreement_mean_deg": "" if not [x["wall_disagreement_deg"] for x in intervals if x["wall_disagreement_deg"] != ""] else float(np.mean([x["wall_disagreement_deg"] for x in intervals if x["wall_disagreement_deg"] != ""])),
        "persistence_duration_sec": track["last_time"] - track["first_time"],
    }


def _make_tracks(case: Any, condition_id: str, sequence_type: str, seed: int, frames: int, dt: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scans = _scan_sequence(case, condition_id, sequence_type, seed, frames, dt)
    tracks: list[dict[str, Any]] = []; next_id = 0
    for frame, scan in enumerate(scans):
        candidates = _detect("combined", *scan.detector_input(), CONDITIONS[condition_id].get("noise_std_m", 0.0))
        observed: list[dict[str, Any]] = []
        for candidate in candidates:
            estimate = estimate_wall_parallel_tangent(scan.angle_deg, scan.range_m, scan.max_range_m, candidate)
            diagnostics = _candidate_diagnostics(scan.angle_deg, scan.range_m, scan.max_range_m, candidate, estimate.estimate_mode)
            observed.append({**candidate, "wall_tangent_deg": "" if estimate.tangent_deg is None else float(estimate.tangent_deg),
                             "wall_mode": estimate.estimate_mode, "mouth_width_m": candidate.get("mouth_width_lower_m", ""),
                             "wall_span_m": diagnostics["selected_wall_span_m"], "wall_disagreement_deg": "" if estimate.wall_disagreement_deg is None else float(estimate.wall_disagreement_deg)})
        used: set[int] = set()
        for candidate in observed:
            scored = []
            for index, track in enumerate(tracks):
                if index in used or not track["intervals"] or track["last_frame"] != frame - 1: continue
                previous = track["intervals"][-1]
                score = circular_interval_iou(previous, candidate)
                if score >= TemporalPersistenceConfig().association_iou or circular_angle_distance(previous["center_angle"], candidate["center_angle"]) <= TemporalPersistenceConfig().association_center_deg:
                    scored.append((score, index))
            if scored: index = max(scored)[1]; used.add(index)
            else:
                index = len(tracks); tracks.append({"track_id": next_id, "first_time": frame * dt, "last_time": frame * dt, "last_frame": frame, "intervals": []}); next_id += 1; used.add(index)
            tracks[index]["intervals"].append(candidate); tracks[index]["last_time"] = frame * dt; tracks[index]["last_frame"] = frame
    feature_rows = []
    gt = [_gt_opening(case, branch) for branch in case.branches]
    for track in tracks:
        features = _track_features(track, frames)
        overlap = max((max((circular_interval_iou(gt_item, interval) for interval in track["intervals"]), default=0.0) for gt_item in gt), default=0.0)
        feature_rows.append({"case_id": case.case_id, "condition_id": condition_id, "sequence_type": sequence_type, "seed": seed, "track_label": "true" if overlap > 0.0 else "false", "max_gt_iou_posthoc": overlap, **features})
    return feature_rows, scans


def _scan_metrics(case: Any, scans: Sequence[Any], detections: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    gt = [_gt_opening(case, branch) for branch in case.branches]; matches = _positive_overlap_matches(gt, list(detections)); matched_gt = {x[0] for x in matches}; matched_det = {x[1] for x in matches}
    return {"gt_opening_count": len(gt), "detected_opening_count": len(detections), "correct_count": len(detections) == len(gt), "precision": len(matches) / max(len(detections), 1), "recall": len(matches) / max(len(gt), 1), "matching_coverage": len(matched_gt) / max(len(gt), 1), "false_positive_count": len(detections) - len(matched_det), "missed_opening_count": len(gt) - len(matched_gt)}


def run_experiment(output_dir: Path, *, seed: int = 20260818, frames: int = 6, dt: float = .1, announce: bool = True) -> dict[str, Any]:
    """Run the small geometry-consistency pilot without adding a decision rule."""
    all_cases = {case.case_id: case for case in create_benchmark_cases(seed)}; cases = [all_cases[x] for x in PILOT_CASE_IDS if x in all_cases]
    feature_rows: list[dict[str, Any]] = []; result_rows: list[dict[str, Any]] = []
    for case in cases:
        for condition_id in PILOT_CONDITIONS:
            for sequence_type in SEQUENCE_TYPES:
                for seed_offset in (17, 1017):
                    sequence_seed = case.seed + seed_offset; features, scans = _make_tracks(case, condition_id, sequence_type, sequence_seed, frames, dt); feature_rows.extend(features)
                    final = scans[-1]; single = _detect("combined", *final.detector_input(), CONDITIONS[condition_id].get("noise_std_m", 0.0)); result_rows.append({"case_id": case.case_id, "condition_id": condition_id, "sequence_type": sequence_type, "seed": sequence_seed, "variant": "single_uncertainty", **_scan_metrics(case, scans, single)})
                    # Existing temporal persistence is evaluated using its unchanged lifecycle.
                    layer = TemporalOpeningPersistence()
                    for frame, scan in enumerate(scans): layer.update(frame * dt, _detect("combined", *scan.detector_input(), CONDITIONS[condition_id].get("noise_std_m", 0.0)))
                    accepted = [{"start_angle": x["latest_start_angle"], "end_angle": x["latest_end_angle"], "center_angle": x["latest_center_angle"], "width_deg": x["latest_width_deg"]} for x in layer.finalize() if x["accepted"]]
                    result_rows.append({"case_id": case.case_id, "condition_id": condition_id, "sequence_type": sequence_type, "seed": sequence_seed, "variant": "temporal_persistence", **_scan_metrics(case, scans, accepted)})
                    result_rows.append({"case_id": case.case_id, "condition_id": condition_id, "sequence_type": sequence_type, "seed": sequence_seed, "variant": "temporal_geometry_evidence", "decision_rule": "none; feature discriminability only", **_scan_metrics(case, scans, accepted)})
            if announce: print(f"[temporal-geometry] {case.case_id} {condition_id}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True); _write(output_dir / "temporal_geometry_results.csv", result_rows); _write(output_dir / "temporal_geometry_tracks.csv", feature_rows)
    comparison = []
    for feature in ("persistence_fraction", "interval_iou_mean", "boundary_std_deg", "width_std_deg", "wall_tangent_axial_std_deg", "two_wall_fraction", "wall_available_fraction", "wall_span_std_m", "mouth_width_std_m"):
        for label in ("true", "false"):
            values = [float(row[feature]) for row in feature_rows if row["track_label"] == label and row[feature] != ""]
            comparison.append({"feature": feature, "label": label, "count": len(values), "mean": "" if not values else float(np.mean(values)), "median": "" if not values else float(np.median(values)), "p90": "" if not values else float(np.percentile(values, 90))})
    _write(output_dir / "temporal_geometry_feature_comparison.csv", comparison)
    failure = [row for row in feature_rows if row["track_label"] == "false" and row["persistence_fraction"] >= .5]
    _write(output_dir / "temporal_geometry_failure_cases.csv", failure)
    summary = []
    for (variant, condition), rows in sorted(_group(result_rows, ("variant", "condition_id")).items()):
        summary.append({"variant": variant, "condition_id": condition, "runs": len(rows), "false_positive_count": sum(int(x["false_positive_count"]) for x in rows), "missed_opening_count": sum(int(x["missed_opening_count"]) for x in rows), "precision": float(np.mean([x["precision"] for x in rows])), "recall": float(np.mean([x["recall"] for x in rows])), "matching_coverage": float(np.mean([x["matching_coverage"] for x in rows]))})
    _write(output_dir / "temporal_geometry_summary.csv", summary); _save_plots(output_dir, feature_rows, summary)
    metadata = {"experiment_id": "EXP-PointCloudDetector-003", "head": _short_head(), "case_count": len(cases), "conditions": list(PILOT_CONDITIONS), "sequence_types": list(SEQUENCE_TYPES), "frames": frames, "dt_sec": dt, "runtime_inputs": ["local angle", "local range", "timestamp", "opening interval", "wall fit observables"], "gt_runtime": False, "decision_rule": "none; discriminability analysis", "circular_sanity": run_synthetic_sanity()}
    (output_dir / "temporal_geometry_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8"); return metadata


def _group(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[tuple[str, ...], list[Mapping[str, Any]]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows: groups[tuple(str(row[f]) for f in fields)].append(row)
    return groups


def _save_plots(directory: Path, tracks: Sequence[Mapping[str, Any]], summary: Sequence[Mapping[str, Any]]) -> None:
    features = ("boundary_std_deg", "interval_iou_mean", "wall_tangent_axial_std_deg", "two_wall_fraction")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, feature in zip(axes.flat, features):
        for label, color in (("true", "tab:blue"), ("false", "tab:red")):
            values = [float(x[feature]) for x in tracks if x["track_label"] == label and x[feature] != ""]
            axis.hist(values, bins=10, alpha=.55, label=label, color=color)
        axis.set_title(feature); axis.legend()
    fig.tight_layout(); fig.savefig(directory / "true_vs_false_opening_geometry_features.png", dpi=140); plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 4)); labels = [x["condition_id"] for x in summary if x["variant"] in ("temporal_persistence", "temporal_geometry_evidence")]; vals = [x["false_positive_count"] for x in summary if x["variant"] in ("temporal_persistence", "temporal_geometry_evidence")]; axis.bar(np.arange(len(vals)), vals); axis.set_xticks(np.arange(len(vals)), labels, rotation=60, ha="right"); axis.set_title("False-opening comparison"); fig.tight_layout(); fig.savefig(directory / "false_positive_comparison.png", dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, color in (("true", "tab:blue"), ("false", "tab:red")):
        selected = [x for x in tracks if x["track_label"] == label]
        axes[0].hist([float(x["wall_tangent_axial_std_deg"]) for x in selected], bins=10, alpha=.55, label=label, color=color)
        axes[1].hist([float(x["wall_available_fraction"]) for x in selected], bins=10, alpha=.55, label=label, color=color)
    axes[0].set_title("Wall tangent axial variation"); axes[1].set_title("Wall availability fraction"); axes[0].legend(); axes[1].legend(); fig.tight_layout(); fig.savefig(directory / "temporal_geometry_wall_consistency.png", dpi=140); plt.close(fig)


def _audit() -> dict[str, Any]:
    signature = " ".join(inspect.signature(_make_tracks).parameters).lower()
    forbidden = [token for token in ("gt", "map", "global", "yaw", "branch_id", "case_id") if token in signature]
    return {"runtime_helper_signature": signature, "forbidden_runtime_tokens": forbidden, "circular_sanity": run_synthetic_sanity(), "axial_359_1_sanity_deg": _axial_std([359.0, 1.0]), "pass": not forbidden and run_synthetic_sanity()["pass"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path); parser.add_argument("--seed", type=int, default=20260818); parser.add_argument("--frames", type=int, default=6); parser.add_argument("--dt", type=float, default=.1); parser.add_argument("--quiet", action="store_true"); parser.add_argument("--audit", action="store_true"); args = parser.parse_args()
    if args.audit: print(json.dumps(_audit(), indent=2)); return
    out = args.output_dir or Path(f"/tmp/pdfs_temporal_geometry_opening_{_short_head()}"); metadata = run_experiment(out, seed=args.seed, frames=args.frames, dt=args.dt, announce=not args.quiet); print(json.dumps({"output_dir": str(out), **metadata}, indent=2))


if __name__ == "__main__": main()
