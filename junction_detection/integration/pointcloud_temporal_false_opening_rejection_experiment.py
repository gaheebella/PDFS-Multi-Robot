"""EXP-PointCloudDetector-002: temporal false-opening rejection.

Diagnostics-only harness.  Existing detector, wall estimator, reliability, and
fusion implementations are imported unchanged.  The only new runtime state is
an Anchor-local interval history from sequential scans.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import subprocess
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from junction_detection.integration.pointcloud_branch_orientation_generalization import (
    _gt_opening, angular_error_deg, create_benchmark_cases, normalize_angle_deg, wall_segments,
)
from junction_detection.integration.pointcloud_detector_physics_experiment import (
    _positive_overlap_matches, _stats, ROBOT_DIAMETER_M, FAMILYWISE_FALSE_ALARM_PROBABILITY,
)
from junction_detection.integration.pointcloud_wall_parallel_orientation import estimate_wall_parallel_tangent
from junction_detection.integration.general_branch_orientation_fusion import BranchOrientationEvidence, fuse_branch_orientation
from junction_detection.pointcloud import pointcloud_junction_detector as baseline
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import simulate_lidar_scan
from junction_detection.pointcloud.pointcloud_junction_detector_uncertainty_aware import (
    DetectorStages, detect_openings as detect_uncertainty_aware,
)
from junction_detection.integration.pointcloud_temporal_opening_persistence import (
    TemporalOpeningPersistence, TemporalPersistenceConfig, run_synthetic_sanity,
)

CONDITIONS = {
    "clean": {}, "noise_0.03": {"noise_std_m": .03}, "noise_0.08": {"noise_std_m": .08},
    "dropout_0.05": {"dropout_probability": .05}, "dropout_0.15": {"dropout_probability": .15},
    "occlusion_0.40": {"occlusion_probability": .40}, "occlusion_0.80": {"occlusion_probability": .80},
    "visibility_0.90": {"visible_boundary_ratio": .90}, "visibility_0.70": {"visible_boundary_ratio": .70},
    "resolution_2deg": {"angle_step_deg": 2.0}, "resolution_4deg": {"angle_step_deg": 4.0},
}
SEQUENCE_TYPES = ("independent", "burst")


def _short_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _detect(variant: str, angles: np.ndarray, ranges: np.ndarray, noise: float) -> list[dict[str, float]]:
    """Invoke protected detectors with only local observations."""
    if variant == "baseline": return baseline.detect_openings(angles, ranges)
    return detect_uncertainty_aware(angles, ranges, noise_std_m=noise, robot_diameter_m=ROBOT_DIAMETER_M,
                                    false_alarm_probability=FAMILYWISE_FALSE_ALARM_PROBABILITY,
                                    stages=DetectorStages(True, True, True, True))


def _scan_sequence(case: Any, condition_id: str, sequence_type: str, seed: int, frames: int, dt: float) -> list[Any]:
    """Generate sequential scans; burst reuses a stochastic local loss mask."""
    params = CONDITIONS[condition_id]
    extent = case.central_radius + max(branch.length for branch in case.branches)
    kwargs = dict(params); kwargs.setdefault("angle_step_deg", 1.0)
    scans = []
    persistent_loss: np.ndarray | None = None
    for frame in range(frames):
        scan = simulate_lidar_scan(wall_segments(case), case.anchor_xy, anchor_yaw_deg=case.anchor_yaw_deg,
                                   max_range_m=extent * 1.30, seed=seed + 104729 * frame, **kwargs)
        if sequence_type == "burst" and frame == 0:
            ideal = simulate_lidar_scan(wall_segments(case), case.anchor_xy, anchor_yaw_deg=case.anchor_yaw_deg,
                                        max_range_m=extent * 1.30, angle_step_deg=kwargs["angle_step_deg"], seed=0)
            persistent_loss = ideal.hit & ~scan.hit
        if sequence_type == "burst" and frame > 0 and persistent_loss is not None:
            hit = scan.hit.copy(); ranges = scan.range_m.copy(); hit[persistent_loss] = False; ranges[persistent_loss] = scan.max_range_m
            theta = np.deg2rad(scan.angle_deg)
            scan = replace(scan, hit=hit, range_m=ranges, local_x=ranges * np.cos(theta), local_y=ranges * np.sin(theta))
        scans.append(scan)
    return scans


def _score(case: Any, detections: Sequence[Mapping[str, float]], angles: np.ndarray, ranges: np.ndarray, max_range: float) -> dict[str, Any]:
    gt = [_gt_opening(case, branch) for branch in case.branches]
    matches = _positive_overlap_matches(gt, list(detections)); matched_gt = {a for a, _, _ in matches}; matched_det = {b for _, b, _ in matches}
    walls = []
    for gt_index, det_index, _iou in matches:
        estimate = estimate_wall_parallel_tangent(angles, ranges, max_range, detections[det_index])
        if estimate.tangent_deg is not None:
            walls.append((float(estimate.tangent_deg), gt[gt_index]["center_angle"]))
    errors = [angular_error_deg(a, b) for a, b in walls]
    return {"gt_opening_count": len(gt), "detected_opening_count": len(detections),
            "matched_opening_count": len(matches), "missed_opening_count": len(gt) - len(matched_gt),
            "false_positive_count": len(detections) - len(matched_det),
            "opening_matching_coverage": len(matched_gt) / max(len(gt), 1),
            "precision": len(matches) / max(len(detections), 1), "recall": len(matches) / max(len(gt), 1),
            "f1": 2 * len(matches) / max(len(gt) + len(detections), 1),
            "wall_tangent_availability": len(walls) / max(len(matches), 1),
            "wall_error_mean_deg": "" if not errors else float(np.mean(errors)),
            "wall_error_p90_deg": "" if not errors else float(np.percentile(errors, 90)),
            "wall_error_max_deg": "" if not errors else float(np.max(errors))}


def run_experiment(output_dir: Path, *, seed: int = 20260818, sequence_length: int = 6, dt: float = .1, announce: bool = True, max_cases: int | None = None) -> dict[str, Any]:
    """Run static-Anchor temporal benchmark and write requested CSV/PNG files."""
    cases = create_benchmark_cases(seed)
    if max_cases is not None: cases = cases[:max(1, int(max_cases))]
    results: list[dict[str, Any]] = []; tracks_rows: list[dict[str, Any]] = []
    condition_summary: list[dict[str, Any]] = []; ablation_rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        for condition_id in CONDITIONS:
            for sequence_type in SEQUENCE_TYPES:
                for seed_index, sequence_seed in enumerate((case.seed + 17, case.seed + 1017)):
                    scans = _scan_sequence(case, condition_id, sequence_type, sequence_seed, sequence_length, dt)
                    layer = TemporalOpeningPersistence()
                    for frame, scan in enumerate(scans):
                        candidates = _detect("combined", *scan.detector_input(), CONDITIONS[condition_id].get("noise_std_m", 0.0))
                        layer.update(frame * dt, candidates)
                    accepted = [dict(summary, start_angle=summary["latest_start_angle"], end_angle=summary["latest_end_angle"], center_angle=summary["latest_center_angle"], width_deg=summary["latest_width_deg"]) for summary in layer.finalize() if summary["accepted"]]
                    final_scan = scans[-1]
                    single = _detect("combined", *final_scan.detector_input(), CONDITIONS[condition_id].get("noise_std_m", 0.0))
                    base = _detect("baseline", *final_scan.detector_input(), 0.0)
                    scores = {"baseline": _score(case, base, final_scan.angle_deg, final_scan.range_m, final_scan.max_range_m),
                              "single_scan_uncertainty": _score(case, single, final_scan.angle_deg, final_scan.range_m, final_scan.max_range_m),
                              "temporal_persistence": _score(case, accepted, final_scan.angle_deg, final_scan.range_m, final_scan.max_range_m)}
                    for variant, score in scores.items():
                        results.append({"case_id": case.case_id, "case_index": case_index, "seed": sequence_seed, "condition_id": condition_id,
                                        "sequence_type": sequence_type, "variant": variant, "sequence_length": sequence_length, "sampling_interval_sec": dt,
                                        **score})
                    for track in layer.finalize():
                        tracks_rows.append({"case_id": case.case_id, "seed": sequence_seed, "condition_id": condition_id, "sequence_type": sequence_type, **track})
                    # An ablation row records observable stages without changing detector thresholds.
                    for variant in ("single_scan_uncertainty", "temporal_persistence"):
                        ablation_rows.append({"case_id": case.case_id, "seed": sequence_seed, "condition_id": condition_id, "sequence_type": sequence_type,
                                              "variant": variant, "false_positive_count": scores[variant]["false_positive_count"],
                                              "missed_opening_count": scores[variant]["missed_opening_count"], "opening_matching_coverage": scores[variant]["opening_matching_coverage"],
                                              "wall_tangent_availability": scores[variant]["wall_tangent_availability"]})
                    if announce and (len(results) // 3) % 100 == 0:
                        print(f"[temporal] case {case_index + 1}/{len(cases)} {condition_id} {sequence_type}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True); _write(output_dir / "temporal_opening_results.csv", results); _write(output_dir / "temporal_opening_tracks.csv", tracks_rows); _write(output_dir / "temporal_ablation.csv", ablation_rows)
    fusion_rows = []
    for row in results:
        available = float(row["wall_tangent_availability"]) > 0.0
        fused = fuse_branch_orientation(
            BranchOrientationEvidence(0.0, 1.0, available, "geometry"),
            BranchOrientationEvidence(None, None, False, "motion"),
        )
        fusion_rows.append({"case_id": row["case_id"], "condition_id": row["condition_id"], "sequence_type": row["sequence_type"], "variant": row["variant"], "status": fused.status, "geometry_used": fused.geometry_used, "motion_used": fused.motion_used, "orientation_error_deg": ""})
    _write(output_dir / "temporal_fusion_regression.csv", fusion_rows)
    for fields, name in [(["variant", "condition_id", "sequence_type"], "temporal_opening_summary.csv"), (["variant", "condition_id"], "temporal_opening_by_condition.csv"), (["variant", "condition_id"], "temporal_wall_summary.csv")]:
        groups = defaultdict(list)
        for row in results: groups[tuple(row[x] for x in fields)].append(row)
        summary = []
        for key, rows in sorted(groups.items()):
            summary.append({**dict(zip(fields, key)), "runs": len(rows), "false_positive_count": sum(r["false_positive_count"] for r in rows),
                            "missed_opening_count": sum(r["missed_opening_count"] for r in rows), "opening_matching_coverage": float(np.mean([r["opening_matching_coverage"] for r in rows])),
                            "precision": float(np.mean([r["precision"] for r in rows])), "recall": float(np.mean([r["recall"] for r in rows])), "f1": float(np.mean([r["f1"] for r in rows])),
                            "wall_tangent_availability": float(np.mean([r["wall_tangent_availability"] for r in rows])),
                            "wall_error_p90_deg": float(np.mean([r["wall_error_p90_deg"] for r in rows if r["wall_error_p90_deg"] != ""])) if any(r["wall_error_p90_deg"] != "" for r in rows) else ""})
        _write(output_dir / name, summary)
        if name == "temporal_opening_summary.csv": condition_summary = summary
    _save_plots(output_dir, results, tracks_rows, condition_summary)
    metadata = {"experiment_id": "EXP-PointCloudDetector-002", "head": _short_head(), "case_count": len(cases), "sequence_length": sequence_length, "sampling_interval_sec": dt,
                "seeds": 2, "runtime_inputs": ["Anchor-local angles", "Anchor-local ranges", "timestamp/frame index"],
                "sensor_audit": {"simulator_hit_exists": True, "detector_runtime_hit_available": False, "primary": "temporal-only"},
                "persistence_config": TemporalPersistenceConfig().__dict__, "sanity": run_synthetic_sanity()}
    (output_dir / "temporal_experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _save_plots(directory: Path, rows: Sequence[Mapping[str, Any]], tracks: Sequence[Mapping[str, Any]], summary: Sequence[Mapping[str, Any]]) -> None:
    """Write compact requested diagnostic figures."""
    variants = ["single_scan_uncertainty", "temporal_persistence"]
    conds = list(CONDITIONS)
    for filename, field, title in [("false_positive_before_after_temporal.png", "false_positive_count", "False positives before/after temporal persistence"),
                                   ("opening_precision_recall_by_condition.png", "precision", "Opening precision by condition"),
                                   ("wall_availability_before_after_temporal.png", "wall_tangent_availability", "Wall availability before/after temporal persistence")]:
        fig, ax = plt.subplots(figsize=(12, 4)); x = np.arange(len(conds)); width = .38
        for i, variant in enumerate(variants):
            vals = [np.mean([float(r[field]) for r in rows if r["variant"] == variant and r["condition_id"] == c]) for c in conds]
            ax.bar(x + (i - .5) * width, vals, width, label=variant)
        ax.set_xticks(x, conds, rotation=45, ha="right"); ax.set_title(title); ax.legend(); fig.tight_layout(); fig.savefig(directory / filename, dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); true = [float(t["persistence_duration_sec"]) for t in tracks if t["accepted"]]; false = [float(t["persistence_duration_sec"]) for t in tracks if not t["accepted"]]; ax.hist([true, false], bins=10, label=["accepted", "rejected"], alpha=.7); ax.set(xlabel="track duration [s]", ylabel="count", title="True/false track persistence (post-hoc GT-free status)"); ax.legend(); fig.tight_layout(); fig.savefig(directory / "true_vs_false_opening_persistence.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar([0, 1], [np.mean([float(r["false_positive_count"]) for r in rows if r["variant"] == v]) for v in variants]); ax.set_xticks([0, 1], variants, rotation=15); ax.set_title("Temporal ablation summary"); fig.tight_layout(); fig.savefig(directory / "temporal_ablation_summary.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); durations = [float(t["persistence_duration_sec"]) for t in tracks if t["accepted"]]; ax.hist(durations, bins=8); ax.set(xlabel="confirmation/persistence time [s]", ylabel="accepted tracks", title="Time-to-confirm proxy"); fig.tight_layout(); fig.savefig(directory / "time_to_confirm_true_opening.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); vals = [float(r["wall_tangent_availability"]) for r in rows if r["variant"] in variants]; ax.hist(vals, bins=10); ax.set(xlabel="wall availability", ylabel="runs", title="Frozen fusion geometry evidence availability"); fig.tight_layout(); fig.savefig(directory / "fusion_error_before_after_temporal.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot([float(t["persistence_duration_sec"]) for t in tracks[:200]], ".", alpha=.5); ax.set(xlabel="track index", ylabel="duration [s]", title="Independent/burst local dropout track examples"); fig.tight_layout(); fig.savefig(directory / "dropout_sequence_examples.png", dpi=140); plt.close(fig)


def _audits() -> dict[str, Any]:
    """Run runtime signature and circular/localization leakage checks."""
    signature = " ".join(inspect.signature(TemporalOpeningPersistence.update).parameters).lower()
    forbidden = [token for token in ("gt", "map", "wall", "branch", "global", "yaw") if token in signature]
    return {"temporal_update_signature": signature, "forbidden_runtime_tokens": forbidden, "circular_sanity": run_synthetic_sanity(), "pass": not forbidden and run_synthetic_sanity()["pass"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path); parser.add_argument("--seed", type=int, default=20260818); parser.add_argument("--sequence-length", type=int, default=6); parser.add_argument("--dt", type=float, default=.1); parser.add_argument("--max-cases", type=int); parser.add_argument("--quiet", action="store_true"); parser.add_argument("--audit", action="store_true"); args = parser.parse_args()
    if args.audit: print(json.dumps(_audits(), indent=2)); return
    out = args.output_dir or Path(f"/tmp/pdfs_temporal_opening_{_short_head()}"); metadata = run_experiment(out, seed=args.seed, sequence_length=args.sequence_length, dt=args.dt, announce=not args.quiet, max_cases=args.max_cases)
    print(json.dumps({"output_dir": str(out), **metadata}, indent=2))


if __name__ == "__main__":
    main()
