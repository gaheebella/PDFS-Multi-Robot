"""EXP-036 general wall-topology plus SPH branch-validation benchmark.

M0--M5 are run through the same frozen EXP-034/035 implementation.  Map case
identity is consumed only by the simulator environment constructor. Candidate
generation and motion validation receive no map-specific branch logic.  GT
branch identity is used after each run solely for recall/false-positive audit.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pygame_simulator.pre_exploration_general_pipeline_simulator import CASES, GeometryBuilder
from pygame_simulator.pre_exploration_persistent_partial_sph_validation import (
    BASELINE_REFERENCE,
    PersistentPartialAuditRunner,
    _candidate_summaries,
    _candidate_state_self_test,
)

EXPERIMENT_ID = "EXP-036"
DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/general_wall_topology_sph_branch_validation_benchmark"
BENCHMARK_CASES = tuple(CASES)


def _write(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    """Write heterogeneous benchmark rows and retain empty audit headers."""
    if fields is None:
        if not rows:
            return
        fields = list(rows[0])
        for row in rows[1:]:
            fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _branch_label(index: int, angle: float) -> str:
    """Mirror the evaluation-only labels used by the existing GT helper."""
    return {0.0: "FORWARD", -90.0: "RIGHT", 90.0: "LEFT"}.get(float(angle), f"BRANCH_{index}")


def _canonical(value: Any) -> Any:
    """Normalize NaNs before deterministic record comparison."""
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def _run_worker(spec: tuple[str, int, bool, str]) -> dict[str, Any]:
    """Run one unchanged deterministic simulation in an isolated process."""
    case_id, frames, rear_start, run_role = spec
    runner = PersistentPartialAuditRunner(case_id, rear_start)
    for frame in range(frames):
        runner.step(frame)
    summaries = _candidate_summaries(runner, runner.world.time)
    candidate_details = []
    for candidate, summary in zip(runner.candidates, summaries):
        evidence = candidate.best_evidence or candidate.last_evidence
        # General post-hoc association uses the actual branch rectangle edge,
        # not an orthogonal-mouth endpoint assumption.  The distance gate is
        # the unchanged EXP-033/034 evaluation tolerance (0.12 * W_hat).
        endpoint_world = candidate.anchor_position_eval + np.array(
            [[math.cos(candidate.anchor_yaw_rad), -math.sin(candidate.anchor_yaw_rad)],
             [math.sin(candidate.anchor_yaw_rad), math.cos(candidate.anchor_yaw_rad)]]
        ) @ candidate.endpoint_local
        matches = []
        for index, branch in enumerate(runner.geometry.branches):
            direction_world = np.array([
                math.sin(math.radians(branch.angle_deg)),
                math.cos(math.radians(branch.angle_deg)),
            ])
            direction_local = np.array(
                [[math.cos(candidate.anchor_yaw_rad), math.sin(candidate.anchor_yaw_rad)],
                 [-math.sin(candidate.anchor_yaw_rad), math.cos(candidate.anchor_yaw_rad)]]
            ) @ direction_world
            alignment = float(np.dot(candidate.free_axis_local, direction_local))
            vertices = np.asarray(runner.geometry.free_rects[2 + index].vertices)
            distances = []
            for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
                edge = end - start
                ratio = float(np.clip(np.dot(endpoint_world - start, edge) / np.dot(edge, edge), 0.0, 1.0))
                distances.append(float(np.linalg.norm(endpoint_world - (start + ratio * edge))))
            edge_distance = min(distances)
            if alignment > 0.0 and edge_distance <= 0.12 * candidate.estimated_width:
                matches.append((edge_distance, -alignment, _branch_label(index, branch.angle_deg)))
        general_label = min(matches)[2] if matches else "FALSE"
        partial_event = next((row for row in runner.events if row["candidate_id"] == candidate.candidate_id and row["event"] == "PARTIAL_BRANCH_CANDIDATE"), None)
        candidate_details.append({
            **summary,
            "branch_eval_posthoc": general_label,
            "map_case": case_id,
            "partial_observed": partial_event is not None,
            "termination_count": len(candidate.endpoint_ids),
            "gap_width": candidate.gap_width,
            "gap_width_ratio": candidate.gap_width / candidate.estimated_width if math.isfinite(candidate.gap_width) else math.nan,
            "free_space_continuation": candidate.free_space_evidence,
            "candidate_lifetime": runner.world.time - candidate.created_time,
            "progress": evidence.get("forward_progress", 0.0),
        })
    return {
        "map_case": case_id,
        "run_role": run_role,
        "events": runner.events,
        "candidate_details": candidate_details,
        "timeline": runner.state_timeline,
        "angular": runner.angular_shadow_rows,
        "world_time": runner.world.time,
        "candidate_count": len(runner.candidates),
    }


def _pathway(row: dict[str, Any]) -> str:
    """Classify one candidate's topology/motion order without changing state."""
    if row["branch_eval_posthoc"] == "FALSE":
        return "FALSE_CANDIDATE"
    partial = bool(row["partial_observed"])
    complete = math.isfinite(float(row["t_complete"]))
    motion = math.isfinite(float(row["t_motion_supported"]))
    if not partial and complete:
        return "DIRECT_COMPLETE"
    if complete and motion:
        return "COMPLETE_BEFORE_MOTION" if row["t_complete"] < row["t_motion_supported"] else "MOTION_BEFORE_COMPLETE"
    if not complete and motion:
        return "PERSISTENT_PARTIAL_MOTION_SUPPORTED"
    if complete:
        return "COMPLETE_ONLY"
    return "PARTIAL_NO_SUPPORT"


def _branch_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a GT-evaluation matrix including missed and false candidates."""
    geometry = GeometryBuilder.build(result["map_case"])
    labels = [_branch_label(index, branch.angle_deg) for index, branch in enumerate(geometry.branches)]
    by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    false_rows = []
    for candidate in result["candidate_details"]:
        label = candidate["branch_eval_posthoc"]
        if label in by_label:
            by_label[label].append(candidate)
        else:
            false_rows.append(candidate)
    rows = []
    for label in labels:
        candidate = by_label[label][0] if by_label[label] else None
        pathway = "MISSED" if candidate is None else _pathway(candidate)
        rows.append({
            "map_case": result["map_case"],
            "candidate_id": "NONE" if candidate is None else candidate["candidate_id"],
            "branch_eval_posthoc": label,
            "candidate_created": candidate is not None,
            "t_candidate_created": math.nan if candidate is None else candidate["t_created"],
            "t_partial": math.nan if candidate is None else candidate["t_partial_confirmed"],
            "partial_observed": False if candidate is None else candidate["partial_observed"],
            "partial_duration": 0.0 if candidate is None else candidate["partial_duration"],
            "complete": False if candidate is None else math.isfinite(candidate["t_complete"]),
            "t_complete": math.nan if candidate is None else candidate["t_complete"],
            "t_first_motion_evidence": math.nan if candidate is None else candidate["t_first_motion_evidence"],
            "t_motion_condition_met": math.nan if candidate is None else candidate["t_motion_threshold_met"],
            "motion_supported": False if candidate is None else math.isfinite(candidate["t_motion_supported"]),
            "t_motion_supported": math.nan if candidate is None else candidate["t_motion_supported"],
            "t_final": math.nan if candidate is None else candidate["t_run_end"],
            "time_order_class": pathway,
            "supporting_robot_count": 0 if candidate is None else candidate["support_robot_count_at_transition"],
            "stable_support_total": 0 if candidate is None else candidate["stable_support_total"],
            "motion_tangent_deg": math.nan if candidate is None else candidate["motion_tangent_deg"],
            "reliability": 0.0 if candidate is None else candidate["reliability"],
            "dispersion": math.nan if candidate is None else candidate["dispersion"],
            "backflow_ratio": 0.0 if candidate is None else candidate["backflow_ratio"],
            "progress": 0.0 if candidate is None else candidate["progress"],
            "free_space_half_plane_entries": 0 if candidate is None else candidate["free_space_half_plane_entries"],
            "termination_count": 0 if candidate is None else candidate["termination_count"],
            "gap_width": math.nan if candidate is None else candidate["gap_width"],
            "gap_width_ratio": math.nan if candidate is None else candidate["gap_width_ratio"],
            "free_space_continuation": math.nan if candidate is None else candidate["free_space_continuation"],
            "candidate_lifetime": 0.0 if candidate is None else candidate["candidate_lifetime"],
            "final_state": "MISSED" if candidate is None else candidate["final_state"],
            "topology_confirmed": False if candidate is None else math.isfinite(candidate["t_complete"]),
            "final_validated": False if candidate is None else candidate["final_state"] == "MOTION_SUPPORTED",
            "true_candidate_eval": candidate is not None,
            "false_candidate_eval": False,
        })
    for candidate in false_rows:
        rows.append({
            "map_case": result["map_case"],
            "candidate_id": candidate["candidate_id"],
            "branch_eval_posthoc": "FALSE",
            "candidate_created": True,
            "t_candidate_created": candidate["t_created"],
            "t_partial": candidate["t_partial_confirmed"],
            "partial_observed": candidate["partial_observed"],
            "partial_duration": candidate["partial_duration"],
            "complete": math.isfinite(candidate["t_complete"]),
            "t_complete": candidate["t_complete"],
            "t_first_motion_evidence": candidate["t_first_motion_evidence"],
            "t_motion_condition_met": candidate["t_motion_threshold_met"],
            "motion_supported": math.isfinite(candidate["t_motion_supported"]),
            "t_motion_supported": candidate["t_motion_supported"],
            "t_final": candidate["t_run_end"],
            "time_order_class": "FALSE_CANDIDATE",
            "supporting_robot_count": candidate["support_robot_count_at_transition"],
            "stable_support_total": candidate["stable_support_total"],
            "motion_tangent_deg": candidate["motion_tangent_deg"],
            "reliability": candidate["reliability"],
            "dispersion": candidate["dispersion"],
            "backflow_ratio": candidate["backflow_ratio"],
            "progress": candidate["progress"],
            "free_space_half_plane_entries": candidate["free_space_half_plane_entries"],
            "termination_count": candidate["termination_count"],
            "gap_width": candidate["gap_width"],
            "gap_width_ratio": candidate["gap_width_ratio"],
            "free_space_continuation": candidate["free_space_continuation"],
            "candidate_lifetime": candidate["candidate_lifetime"],
            "final_state": candidate["final_state"],
            "topology_confirmed": math.isfinite(candidate["t_complete"]),
            "final_validated": candidate["final_state"] == "MOTION_SUPPORTED",
            "true_candidate_eval": False,
            "false_candidate_eval": True,
        })
    return rows


def _geometry_summary(case_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate branch recall and false-validation metrics for one map."""
    true_rows = [row for row in rows if row["branch_eval_posthoc"] != "FALSE"]
    candidates = [row for row in rows if row["candidate_created"]]
    true_candidates = [row for row in true_rows if row["candidate_created"]]
    false_candidates = [row for row in rows if row["false_candidate_eval"]]
    complete_true = [row for row in true_candidates if row["complete"]]
    motion_true = [row for row in true_candidates if row["motion_supported"]]
    false_motion = [row for row in false_candidates if row["motion_supported"]]
    mean = lambda values: float(np.mean(values)) if values else math.nan
    return {
        "map_case": case_id,
        "gt_outgoing_count_eval": len(true_rows),
        "candidate_count": len(candidates),
        "true_candidate_count": len(true_candidates),
        "false_candidate_count": len(false_candidates),
        "complete_true_count": len(complete_true),
        "motion_supported_true_count": len(motion_true),
        "false_motion_supported_count": len(false_motion),
        "candidate_recall": len(true_candidates) / max(1, len(true_rows)),
        "complete_recall": len(complete_true) / max(1, len(true_rows)),
        "final_support_recall": len(motion_true) / max(1, len(true_rows)),
        "false_positive_rate": len(false_candidates) / max(1, len(candidates)),
        "mean_candidate_latency": mean([row["t_candidate_created"] for row in true_candidates]),
        "mean_complete_latency": mean([row["t_complete"] for row in complete_true]),
        "mean_motion_latency": mean([row["t_motion_supported"] for row in motion_true]),
        "mean_validation_latency": mean([row["t_motion_supported"] - row["t_candidate_created"] for row in motion_true]),
    }


def _pathway_rows(case_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["time_order_class"] for row in rows if row["candidate_created"])
    return {
        "map_case": case_id,
        "complete_before_motion_count": counts["COMPLETE_BEFORE_MOTION"],
        "motion_before_complete_count": counts["MOTION_BEFORE_COMPLETE"],
        "persistent_partial_motion_count": counts["PERSISTENT_PARTIAL_MOTION_SUPPORTED"],
        "complete_only_count": counts["COMPLETE_ONLY"],
        "partial_no_support_count": counts["PARTIAL_NO_SUPPORT"],
        "direct_complete_count": counts["DIRECT_COMPLETE"],
        "false_candidate_count": counts["FALSE_CANDIDATE"],
    }


def _plot_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    true = [row for row in rows if row["branch_eval_posthoc"] != "FALSE"]
    data = np.array([[row["partial_observed"], row["complete"], row["motion_supported"], row["false_candidate_eval"]] for row in true], dtype=float)
    fig, axis = plt.subplots(figsize=(8, max(5, 0.38 * len(true))), constrained_layout=True)
    axis.imshow(data, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    axis.set(xticks=range(4), xticklabels=["PARTIAL", "COMPLETE", "MOTION", "FALSE"], yticks=range(len(true)), yticklabels=[f"{row['map_case']}:{row['branch_eval_posthoc']}" for row in true], title="General branch-validation matrix")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_timing(path: Path, rows: list[dict[str, Any]]) -> None:
    true = [row for row in rows if row["candidate_created"] and not row["false_candidate_eval"]]
    labels = [f"{row['map_case']}:{row['branch_eval_posthoc']}" for row in true]
    x = np.arange(len(true))
    fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    axis.scatter(x, [row["t_candidate_created"] for row in true], label="candidate")
    axis.scatter(x, [row["t_complete"] for row in true], label="complete")
    axis.scatter(x, [row["t_motion_supported"] for row in true], label="motion")
    axis.set_xticks(x, labels, rotation=45, ha="right")
    axis.set(ylabel="time [s]", title="Validation timing by geometry/branch")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_pathways(path: Path, pathways: list[dict[str, Any]]) -> None:
    keys = ("complete_before_motion_count", "motion_before_complete_count", "persistent_partial_motion_count", "complete_only_count", "partial_no_support_count", "direct_complete_count", "false_candidate_count")
    totals = [sum(int(row[key]) for row in pathways) for key in keys]
    fig, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    axis.bar(range(len(keys)), totals)
    axis.set(xticks=range(len(keys)), xticklabels=[key.replace("_count", "").replace("_", "\n") for key in keys], ylabel="candidate count", title="Topology vs motion pathways")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_recall(path: Path, angular: list[dict[str, Any]], geometry: list[dict[str, Any]]) -> None:
    cases = [row["map_case"] for row in geometry if row["map_case"] != "M0_STRAIGHT"]
    angular_by = {row["map_case"]: row for row in angular}
    geo_by = {row["map_case"]: row for row in geometry}
    x = np.arange(len(cases)); width = 0.36
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.bar(x - width / 2, [angular_by[case]["angular_recall_at_first_candidate"] for case in cases], width, label="angular")
    axis.bar(x + width / 2, [geo_by[case]["final_support_recall"] for case in cases], width, label="integrated motion")
    axis.set_xticks(x, cases, rotation=25, ha="right")
    axis.set(ylim=(0, 1.05), ylabel="outgoing branch recall", title="Angular vs integrated branch recall")
    axis.legend(); axis.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _baseline_exact(result: dict[str, Any]) -> bool:
    for event, expected in BASELINE_REFERENCE.items():
        values = [float(row["timestamp"]) for row in result["events"] if row["event"] == event]
        if len(values) != 2 or any(not math.isclose(value, expected, abs_tol=1e-12) for value in values):
            return False
    return True


def run_benchmark(output: Path, frames: int, m0_frames: int, workers: int, replay: bool) -> dict[str, Any]:
    """Run all existing geometries with identical frozen implementation."""
    output.mkdir(parents=True, exist_ok=True)
    specs = []
    for case_id in BENCHMARK_CASES:
        count = m0_frames if case_id == "M0_STRAIGHT" else frames
        rear = case_id != "M0_STRAIGHT"
        specs.append((case_id, count, rear, "MAIN"))
        if replay:
            specs.append((case_id, count, rear, "REPLAY"))
    if workers <= 1:
        results = [_run_worker(spec) for spec in specs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_run_worker, specs))
    main = {row["map_case"]: row for row in results if row["run_role"] == "MAIN"}
    repeated = {row["map_case"]: row for row in results if row["run_role"] == "REPLAY"}
    def comparable(row: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in row.items() if key != "run_role"}
    deterministic = {
        case_id: (
            not replay
            or _canonical(comparable(main[case_id]))
            == _canonical(comparable(repeated[case_id]))
        )
        for case_id in BENCHMARK_CASES
    }
    baseline_pass = _baseline_exact(main["M1_CROSS_BASELINE"])
    m0_pass = main["M0_STRAIGHT"]["candidate_count"] == 0

    branch_rows = []
    timeline_rows = []
    for case_id in BENCHMARK_CASES:
        branch_rows.extend(_branch_rows(main[case_id]))
        timeline_rows.extend({"map_case": case_id, **row} for row in main[case_id]["timeline"])
    geometry_rows = [_geometry_summary(case_id, [row for row in branch_rows if row["map_case"] == case_id]) for case_id in BENCHMARK_CASES]
    pathway_rows = [_pathway_rows(case_id, [row for row in branch_rows if row["map_case"] == case_id]) for case_id in BENCHMARK_CASES]
    motion_rows = [row for row in branch_rows if row["candidate_created"]]
    false_rows = [row for row in branch_rows if row["false_candidate_eval"]]
    angular_rows = []
    for case_id in BENCHMARK_CASES:
        result = main[case_id]
        angular = result["angular"]
        first_time = min((row["t_created"] for row in result["candidate_details"]), default=math.nan)
        at_first = min(angular, key=lambda row: abs(row["timestamp"] - first_time)) if angular and math.isfinite(first_time) else None
        gt_count = len(GeometryBuilder.build(case_id).branches)
        angular_rows.append({
            "map_case": case_id,
            "gt_outgoing_count_eval": gt_count,
            "angular_match_at_first_candidate": 0 if at_first is None else at_first["angular_outgoing_count_eval_only"],
            "angular_recall_at_first_candidate": 0.0 if gt_count == 0 else (0 if at_first is None else at_first["angular_outgoing_count_eval_only"] / gt_count),
            "angular_max_outgoing_match": max((row["angular_outgoing_count_eval_only"] for row in angular), default=0),
            "integrated_candidate_count": next(row["true_candidate_count"] for row in geometry_rows if row["map_case"] == case_id),
            "integrated_motion_supported_count": next(row["motion_supported_true_count"] for row in geometry_rows if row["map_case"] == case_id),
        })

    incomplete = any(row["final_support_recall"] < 1.0 for row in geometry_rows if row["map_case"] != "M0_STRAIGHT")
    false_validated = any(row["false_motion_supported_count"] > 0 for row in geometry_rows)
    if not baseline_pass or not all(deterministic.values()):
        verdict = "F_BASELINE_OR_IMPLEMENTATION_INCONSISTENCY"
    elif not m0_pass:
        verdict = "E_M0_NEGATIVE_CONTROL_REGRESSION"
    elif false_validated:
        verdict = "C_FALSE_BRANCH_VALIDATION_PRESENT"
    elif incomplete:
        verdict = "B_GENERAL_PIPELINE_RECALL_INCOMPLETE"
    else:
        verdict = "A_GENERAL_BRANCH_VALIDATION_PIPELINE_CONSISTENT"

    benchmark_rows = []
    for case_id in BENCHMARK_CASES:
        geometry = GeometryBuilder.build(case_id)
        benchmark_rows.append({
            "map_case": case_id,
            "incoming_width": geometry.incoming_width,
            "junction_size": geometry.junction_size,
            "outgoing_branch_count": len(geometry.branches),
            "branch_angles_deg": ";".join(str(branch.angle_deg) for branch in geometry.branches),
            "branch_widths": ";".join(str(branch.width) for branch in geometry.branches),
            "frames": m0_frames if case_id == "M0_STRAIGHT" else frames,
            "seed": "N/A_DETERMINISTIC",
            "main_replay_equal": deterministic[case_id],
        })
    _write(output / "benchmark_cases.csv", benchmark_rows)
    _write(output / "branch_candidate_timeline.csv", timeline_rows)
    _write(output / "branch_level_summary.csv", branch_rows)
    _write(output / "geometry_level_summary.csv", geometry_rows)
    _write(output / "validation_pathways.csv", pathway_rows)
    _write(output / "motion_metrics.csv", motion_rows)
    _write(output / "false_candidate_audit.csv", false_rows, list(branch_rows[0]))
    _write(output / "angular_shadow_comparison.csv", angular_rows)
    _write(output / "verdict.csv", [{
        "experiment_id": EXPERIMENT_ID,
        "verdict": verdict,
        "M1_baseline_exact": baseline_pass,
        "M0_negative_control": m0_pass,
        "all_deterministic_replays_equal": all(deterministic.values()),
        "seed": "N/A_DETERMINISTIC",
        "runtime_GT_map_used": False,
        "map_specific_algorithm_logic": False,
        "detector_modified": False,
        "motion_threshold_modified": False,
        "SPH_force_modified": False,
        "general_posthoc_match_audited": True,
    }])
    _plot_matrix(output / "branch_validation_generalization_matrix.png", branch_rows)
    _plot_timing(output / "validation_timing_by_geometry.png", branch_rows)
    _plot_pathways(output / "topology_vs_motion_pathways.png", pathway_rows)
    _plot_recall(output / "angular_vs_integrated_branch_recall.png", angular_rows, geometry_rows)
    return {"verdict": verdict, "baseline": baseline_pass, "m0": m0_pass, "deterministic": deterministic, "geometry": geometry_rows, "pathways": pathway_rows}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--m0-frames", type=int, default=120)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _candidate_state_self_test()
    result = run_benchmark(args.output, args.frames, args.m0_frames, args.workers, not args.no_replay)
    print(f"EXP-036 verdict={result['verdict']} baseline={result['baseline']} M0={result['m0']}")
    print(f"deterministic={result['deterministic']}")
    for row in result["geometry"]:
        print(row)
    print(f"pathways={result['pathways']}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
