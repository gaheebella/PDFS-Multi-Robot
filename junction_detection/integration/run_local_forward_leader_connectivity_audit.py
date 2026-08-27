"""Audit LOCAL_FORWARD leader connectivity before/after local graph coupling."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.integration.run_junction_shadow_detection import (
    EvidenceThresholds,
    replay_detector,
)
from pygame_simulator import pre_exploration_general_pipeline_simulator as current_sim

DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/local_forward_leader_connectivity"
DEFAULT_BEFORE_MODULE = Path("/tmp/pdfs_pre_exploration_before_leader_connectivity.py")
DEFAULT_FROZEN_VERDICT = ROOT / "junction_detection/integration/output/junction_shadow_detection/junction_shadow_verdict.csv"
DEFAULT_BEFORE_WINDOW = ROOT / "junction_detection/integration/output/junction_shadow_detection/junction_shadow_timeline_m1.csv"
CASES = ("M0_STRAIGHT", "M1_CROSS_BASELINE")


def _write(path: Path, rows: list[dict]) -> None:
    """Write homogeneous dictionaries to CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read(path: Path) -> list[dict]:
    """Read a prior timeline while restoring scalar CSV types."""
    rows = []
    for source in csv.DictReader(path.open()):
        row = {}
        for key, value in source.items():
            if value in {"True", "False"}:
                row[key] = value == "True"
            else:
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
        rows.append(row)
    return rows


def _load_module(path: Path):
    """Load the preserved pre-change simulator snapshot."""
    spec = importlib.util.spec_from_file_location("pre_exploration_before_connectivity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load before module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _distance_graph(robots, radius: float) -> dict[int, list]:
    """Build a local distance graph for independent audit measurement."""
    grid = {}
    for robot in robots:
        key = tuple(np.floor(robot.position / radius).astype(int))
        grid.setdefault(key, []).append(robot)
    graph = {}
    for robot in robots:
        key = tuple(np.floor(robot.position / radius).astype(int))
        peers = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for peer in grid.get((key[0] + dx, key[1] + dy), ()):
                    if peer is not robot and np.linalg.norm(peer.position - robot.position) <= radius:
                        peers.append(peer)
        graph[robot.robot_id] = peers
    return graph


def _connectivity(world, module) -> dict:
    """Measure leader support and communication connectivity post-step."""
    robots = world.robots
    by_id = {robot.robot_id: robot for robot in robots}
    leader = by_id[world.lidar_robot_id]
    support_range = float(module.SMOOTHING_LENGTH)
    communication_range = float(getattr(module, "LOCAL_COMMUNICATION_RANGE", 54.0 * module.MAP_SCALE))
    support = _distance_graph(robots, support_range)
    communication = _distance_graph(robots, communication_range)
    forward = np.array([math.cos(leader.body_yaw_rad), math.sin(leader.body_yaw_rad)])
    rear = [
        peer for peer in support[leader.robot_id]
        if float(np.dot(peer.position - leader.position, forward)) <= 0.0
    ]
    nearest = min(
        (float(np.linalg.norm(peer.position - leader.position)) for peer in support[leader.robot_id]),
        default=math.inf,
    )
    gap = min(
        (float(np.linalg.norm(peer.position - leader.position)) for peer in rear),
        default=math.inf,
    )
    hops = {leader.robot_id: 0}
    queue = [leader.robot_id]
    while queue:
        current = queue.pop(0)
        for peer in communication[current]:
            if peer.robot_id not in hops:
                hops[peer.robot_id] = hops[current] + 1
                queue.append(peer.robot_id)
    return {
        "leader_to_nearest_follower_distance": nearest,
        "leader_to_front_pack_gap": gap,
        "normalized_leader_gap": gap / communication_range,
        "leader_connected": bool(rear),
        "leader_connected_component_size": len(hops),
        "connected_to_leader_count": max(0, len(hops) - 1),
        "disconnected_count": len(robots) - len(hops),
        "leader_max_hop": max(hops.values()),
        "communication_edge_count": sum(len(peers) for peers in communication.values()) // 2,
        "support_edge_count": sum(len(peers) for peers in support.values()) // 2,
        "leader_forward_speed": float(np.dot(leader.velocity, forward)),
        "leader_drive_scale": float(leader.propulsion_weight),
        "communication_range": communication_range,
        "support_range": support_range,
    }


def _run(module, case: str, frames: int) -> list[dict]:
    """Run one physical case and append independent connectivity metrics."""
    runner = module.SimulationRunner(case, "local_forward")
    rows = []
    for frame in range(frames):
        row = runner.step(frame)
        if row is not None:
            row.update(_connectivity(runner.world, module))
            rows.append(row)
    return rows


def _frozen_thresholds(path: Path) -> EvidenceThresholds:
    """Load the detector thresholds measured before the physics change."""
    row = next(csv.DictReader(path.open()))
    return EvidenceThresholds(
        sph=float(row["sph_threshold"]),
        boundary=float(row["boundary_threshold"]),
        lidar=float(row["lidar_threshold"]),
        sph_margin=0.0,
        boundary_margin=0.0,
        lidar_margin=0.0,
    )


def _episodes(rows: list[dict], field: str) -> list[tuple[float, float]]:
    """Return sampled boolean episodes."""
    result = []
    start = None
    previous = None
    for row in rows:
        timestamp = float(row["timestamp"])
        if row[field] and start is None:
            start = timestamp
        elif not row[field] and start is not None:
            result.append((start, float(previous) + current_sim.SAMPLE_PERIOD))
            start = None
        previous = timestamp
    if start is not None:
        result.append((start, float(previous) + current_sim.SAMPLE_PERIOD))
    return result


def _summary(label: str, case: str, rows: list[dict]) -> dict:
    """Summarize physical stability and leader attachment."""
    disconnected = [row for row in rows if not row["leader_connected"]]
    finite_gaps = [float(row["leader_to_front_pack_gap"]) for row in rows if math.isfinite(float(row["leader_to_front_pack_gap"]))]
    return {
        "version": label,
        "map_case": case,
        "sample_count": len(rows),
        "leader_max_front_pack_gap": max(finite_gaps, default=math.inf),
        "leader_max_normalized_gap": max((float(row["normalized_leader_gap"]) for row in rows if math.isfinite(float(row["normalized_leader_gap"]))), default=math.inf),
        "leader_disconnected_sample_count": len(disconnected),
        "leader_disconnected_duration": len(disconnected) * current_sim.SAMPLE_PERIOD,
        "leader_min_component_size": min(int(row["leader_connected_component_size"]) for row in rows),
        "max_disconnected_robot_count": max(int(row["disconnected_count"]) for row in rows),
        "leader_max_hop": max(int(row["leader_max_hop"]) for row in rows),
        "mean_communication_edge_count": float(np.mean([row["communication_edge_count"] for row in rows])),
        "mean_support_edge_count": float(np.mean([row["support_edge_count"] for row in rows])),
        "leader_mean_forward_speed": float(np.mean([row["leader_forward_speed"] for row in rows])),
        "leader_max_forward_speed": max(float(row["leader_forward_speed"]) for row in rows),
        "leader_min_drive_scale": min(float(row["leader_drive_scale"]) for row in rows),
        "min_inter_robot_distance": min(float(row["min_inter_robot_distance"]) for row in rows),
        "max_overlap_pair_count": max(int(row["overlap_pair_count"]) for row in rows),
        "max_speed": max(float(row["max_speed"]) for row in rows),
        "max_lateral_span": max(float(row["local_front_lateral_span"]) for row in rows),
        "min_boundary_largest_component_fraction": min(float(row["boundary_largest_component_fraction"]) for row in rows),
        "max_boundary_second_component_fraction": max(float(row["boundary_second_component_fraction"]) for row in rows),
        "max_boundary_component_count": max(int(row["boundary_component_count"]) for row in rows),
        "max_nan_inf_state_count": max(int(row["nan_inf_state_count"]) for row in rows),
        "max_outside_free_space_robot_count": max(int(row["outside_free_space_robot_count"]) for row in rows),
    }


def _plot(before: dict[str, list[dict]], after: dict[str, list[dict]], output: Path) -> None:
    """Plot leader connectivity, expansion and frozen shadow regression."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex="col")
    for column, case in enumerate(CASES):
        series = [("after", after[case], "-")]
        if case in before:
            series.insert(0, ("before", before[case], "--"))
        for label, rows, style in series:
            time = [row["timestamp"] for row in rows]
            axes[0, column].plot(time, [row["leader_to_front_pack_gap"] for row in rows], style, label=f"{label} leader gap")
            axes[0, column].plot(time, [row["support_range"] for row in rows], ":", color="black", alpha=.5, label="support range" if label == "after" else None)
            axes[1, column].plot(time, [row["leader_connected_component_size"] for row in rows], style, label=f"{label} comm comp")
            axes[1, column].plot(time, [row["disconnected_count"] for row in rows], style, alpha=.7, label=f"{label} disconnected")
            axes[2, column].plot(time, [row["local_front_lateral_span"] for row in rows], style, label=f"{label} lateral span")
        time = [row["timestamp"] for row in after[case]]
        axes[2, column].step(time, [float(row["shadow_fusion_trigger"]) * 100.0 for row in after[case]], where="post", label="frozen fusion ×100", alpha=.7)
        axes[0, column].set_title(case); axes[2, column].set_xlabel("time [s]")
        for axis in axes[:, column]: axis.legend(fontsize=7)
    axes[0, 0].set_ylabel("leader gap")
    axes[1, 0].set_ylabel("connectivity count")
    axes[2, 0].set_ylabel("span / trigger")
    fig.suptitle("LOCAL_FORWARD leader connectivity audit")
    fig.tight_layout()
    fig.savefig(output / "leader_connectivity_audit.png", dpi=150)
    plt.close(fig)


def run(frames: int, output: Path, before_module: Path, frozen_verdict: Path, reuse_m0_after: Path | None = None) -> dict:
    """Execute paired legacy/current runs and save the complete audit."""
    output.mkdir(parents=True, exist_ok=True)
    legacy = _load_module(before_module)
    # The reported defect is M1 leader separation. M0 has no pre-change
    # detachment evidence and is retained as the after-change negative control,
    # avoiding one redundant 600-frame physical run.
    before = {"M1_CROSS_BASELINE": _run(legacy, "M1_CROSS_BASELINE", frames)}
    after_raw = {
        "M0_STRAIGHT": _read(reuse_m0_after) if reuse_m0_after else _run(current_sim, "M0_STRAIGHT", frames),
        "M1_CROSS_BASELINE": _run(current_sim, "M1_CROSS_BASELINE", frames),
    }
    thresholds = _frozen_thresholds(frozen_verdict)
    after = {case: replay_detector(rows, thresholds) for case, rows in after_raw.items()}
    _write(output / "leader_connectivity_timeline_m0.csv", after["M0_STRAIGHT"])
    _write(output / "leader_connectivity_timeline_m1.csv", after["M1_CROSS_BASELINE"])
    summaries = [
        _summary(label, case, rows)
        for label, values in (("before", before), ("after", after))
        for case, rows in values.items()
    ]
    _write(output / "leader_connectivity_summary.csv", summaries)
    comparisons = []
    for case in before:
        old = next(row for row in summaries if row["version"] == "before" and row["map_case"] == case)
        new = next(row for row in summaries if row["version"] == "after" and row["map_case"] == case)
        for metric in old:
            if metric not in {"version", "map_case", "sample_count"}:
                comparisons.append({"map_case": case, "metric": metric, "before": old[metric], "after": new[metric], "delta": float(new[metric]) - float(old[metric])})
    _write(output / "leader_connectivity_before_after.csv", comparisons)

    m0_after = next(row for row in summaries if row["version"] == "after" and row["map_case"] == "M0_STRAIGHT")
    m1_before = next(row for row in summaries if row["version"] == "before" and row["map_case"] == "M1_CROSS_BASELINE")
    m1_after = next(row for row in summaries if row["version"] == "after" and row["map_case"] == "M1_CROSS_BASELINE")
    m0_fusion = _episodes(after["M0_STRAIGHT"], "shadow_fusion_trigger")
    m1_fusion = _episodes(after["M1_CROSS_BASELINE"], "shadow_fusion_trigger")
    attached = m0_after["leader_disconnected_sample_count"] == 0 and m1_after["leader_disconnected_sample_count"] == 0
    stable = max(m0_after["max_nan_inf_state_count"], m1_after["max_nan_inf_state_count"]) == 0 and max(m0_after["max_outside_free_space_robot_count"], m1_after["max_outside_free_space_robot_count"]) == 0
    expansion_ratio = m1_after["max_lateral_span"] / max(m1_before["max_lateral_span"], 1e-9)
    detector_ok = not m0_fusion and bool(m1_fusion)
    overlap_regression = m1_after["max_overlap_pair_count"] > m1_before["max_overlap_pair_count"]
    if attached and stable and not overlap_regression and expansion_ratio >= 0.80 and detector_ok:
        verdict_name = "A. LEADER_FRONT_CONNECTIVITY_VALID"
    elif attached and expansion_ratio < 0.80:
        verdict_name = "D. OVERCONSTRAINED"
    elif not attached:
        verdict_name = "C. LEADER_DETACHMENT_REMAINS"
    elif not stable or overlap_regression:
        verdict_name = "E. UNSTABLE"
    else:
        verdict_name = "B. PARTIALLY_VALID"
    verdict = {
        "verdict": verdict_name,
        "frames_per_case": frames,
        "m0_after_max_gap": m0_after["leader_max_front_pack_gap"],
        "m0_after_disconnected_duration": m0_after["leader_disconnected_duration"],
        "m1_before_max_gap": m1_before["leader_max_front_pack_gap"],
        "m1_after_max_gap": m1_after["leader_max_front_pack_gap"],
        "m1_after_disconnected_duration": m1_after["leader_disconnected_duration"],
        "m1_lateral_expansion_retention": expansion_ratio,
        "m1_before_max_overlap_pairs": m1_before["max_overlap_pair_count"],
        "m1_after_max_overlap_pairs": m1_after["max_overlap_pair_count"],
        "overlap_regression": overlap_regression,
        "m0_frozen_fusion_episode_count": len(m0_fusion),
        "m1_frozen_fusion_episode_count": len(m1_fusion),
        "m1_frozen_fusion_first_time": m1_fusion[0][0] if m1_fusion else math.nan,
        "frozen_sph_threshold": thresholds.sph,
        "frozen_boundary_threshold": thresholds.boundary,
        "frozen_lidar_threshold": thresholds.lidar,
        "runtime_gt_or_global_control": False,
        "tether_used": False,
    }
    _write(output / "leader_connectivity_verdict.csv", [verdict])
    _plot(before, after, output)
    return verdict


def run_targeted(frames: int, output: Path, frozen_verdict: Path, before_window: Path) -> dict:
    """Run only current M1 for candidate screening; never rerun saved baselines."""
    output.mkdir(parents=True, exist_ok=True)
    after_raw = _run(current_sim, "M1_CROSS_BASELINE", frames)
    thresholds = _frozen_thresholds(frozen_verdict)
    after = replay_detector(after_raw, thresholds)
    before_rows = _read(before_window)
    end_time = float(after[-1]["timestamp"])
    before_window_rows = [row for row in before_rows if float(row["timestamp"]) <= end_time + 1e-9]
    if not before_window_rows:
        raise ValueError("saved before-M1 window has no comparable samples")
    before_overlap = max(int(row["overlap_pair_count"]) for row in before_window_rows)
    before_span = max(float(row["local_front_lateral_span"]) for row in before_window_rows)
    after_overlap = max(int(row["overlap_pair_count"]) for row in after)
    after_span = max(float(row["local_front_lateral_span"]) for row in after)
    fusion = _episodes(after, "shadow_fusion_trigger")
    disconnected = sum(not row["leader_connected"] for row in after)
    max_gap = max(float(row["leader_to_front_pack_gap"]) for row in after)
    min_component = min(int(row["leader_connected_component_size"]) for row in after)
    stable = max(int(row["nan_inf_state_count"]) for row in after) == 0 and max(int(row["outside_free_space_robot_count"]) for row in after) == 0
    overlap_ok = after_overlap <= before_overlap
    expansion_retention = after_span / max(before_span, 1e-9)
    passed = disconnected == 0 and max_gap < current_sim.SMOOTHING_LENGTH and min_component == current_sim.ROBOT_COUNT and stable and overlap_ok and expansion_retention >= 0.80 and bool(fusion)
    verdict = {
        "targeted_verdict": "PASS" if passed else "FAIL",
        "frames": frames,
        "saved_before_window_source": str(before_window.resolve()),
        "saved_before_window_samples": len(before_window_rows),
        "leader_max_gap": max_gap,
        "leader_disconnected_duration": disconnected * current_sim.SAMPLE_PERIOD,
        "leader_min_component_size": min_component,
        "before_window_max_overlap_pairs": before_overlap,
        "after_max_overlap_pairs": after_overlap,
        "overlap_regression": not overlap_ok,
        "before_window_max_lateral_span": before_span,
        "after_max_lateral_span": after_span,
        "lateral_expansion_retention": expansion_retention,
        "frozen_fusion_episode_count": len(fusion),
        "frozen_fusion_first_time": fusion[0][0] if fusion else math.nan,
        "max_nan_inf": max(int(row["nan_inf_state_count"]) for row in after),
        "max_outside": max(int(row["outside_free_space_robot_count"]) for row in after),
        "frozen_sph_threshold": thresholds.sph,
        "frozen_boundary_threshold": thresholds.boundary,
        "frozen_lidar_threshold": thresholds.lidar,
    }
    _write(output / "leader_connectivity_targeted_m1_timeline.csv", after)
    _write(output / "leader_connectivity_targeted_m1_verdict.csv", [verdict])
    return verdict


def run_final_m1_only(frames: int, output: Path, frozen_verdict: Path, baseline_summary: Path, m0_timeline: Path) -> dict:
    """Run the accepted candidate on M1 once, reusing full saved baselines."""
    saved_summary = _read(baseline_summary)
    before_m1 = next(row for row in saved_summary if row["version"] == "before" and row["map_case"] == "M1_CROSS_BASELINE")
    after_m0 = next(row for row in saved_summary if row["version"] == "after" and row["map_case"] == "M0_STRAIGHT")
    m0 = _read(m0_timeline)
    thresholds = _frozen_thresholds(frozen_verdict)
    m1 = replay_detector(_run(current_sim, "M1_CROSS_BASELINE", frames), thresholds)
    after_m1 = _summary("after", "M1_CROSS_BASELINE", m1)
    summaries = [before_m1, after_m0, after_m1]
    _write(output / "leader_connectivity_timeline_m1.csv", m1)
    _write(output / "leader_connectivity_summary.csv", summaries)
    comparison = []
    for metric in before_m1:
        if metric not in {"version", "map_case", "sample_count"}:
            comparison.append({"map_case": "M1_CROSS_BASELINE", "metric": metric, "before": before_m1[metric], "after": after_m1[metric], "delta": float(after_m1[metric]) - float(before_m1[metric])})
    _write(output / "leader_connectivity_before_after.csv", comparison)
    m0_fusion = _episodes(m0, "shadow_fusion_trigger")
    m1_fusion = _episodes(m1, "shadow_fusion_trigger")
    expansion_ratio = float(after_m1["max_lateral_span"]) / max(float(before_m1["max_lateral_span"]), 1e-9)
    overlap_regression = int(after_m1["max_overlap_pair_count"]) > int(before_m1["max_overlap_pair_count"])
    attached = int(after_m1["leader_disconnected_sample_count"]) == 0
    stable = int(after_m1["max_nan_inf_state_count"]) == 0 and int(after_m1["max_outside_free_space_robot_count"]) == 0
    detector_ok = not m0_fusion and bool(m1_fusion)
    if attached and stable and not overlap_regression and expansion_ratio >= 0.80 and detector_ok:
        verdict_name = "A. LEADER_FRONT_CONNECTIVITY_VALID"
    elif attached and expansion_ratio < 0.80:
        verdict_name = "D. OVERCONSTRAINED"
    elif not attached:
        verdict_name = "C. LEADER_DETACHMENT_REMAINS"
    elif not stable or overlap_regression:
        verdict_name = "E. UNSTABLE"
    else:
        verdict_name = "B. PARTIALLY_VALID"
    verdict = {
        "verdict": verdict_name,
        "frames": frames,
        "leader_max_gap": after_m1["leader_max_front_pack_gap"],
        "leader_disconnected_duration": after_m1["leader_disconnected_duration"],
        "leader_min_component_size": after_m1["leader_min_component_size"],
        "before_max_overlap_pairs": before_m1["max_overlap_pair_count"],
        "after_max_overlap_pairs": after_m1["max_overlap_pair_count"],
        "overlap_regression": overlap_regression,
        "lateral_expansion_retention": expansion_ratio,
        "m0_frozen_fusion_episode_count": len(m0_fusion),
        "m1_frozen_fusion_episode_count": len(m1_fusion),
        "m1_frozen_fusion_first_time": m1_fusion[0][0] if m1_fusion else math.nan,
        "max_nan_inf": after_m1["max_nan_inf_state_count"],
        "max_outside": after_m1["max_outside_free_space_robot_count"],
        "frozen_sph_threshold": thresholds.sph,
        "frozen_boundary_threshold": thresholds.boundary,
        "frozen_lidar_threshold": thresholds.lidar,
    }
    _write(output / "leader_connectivity_verdict.csv", [verdict])
    _plot({}, {"M0_STRAIGHT": m0, "M1_CROSS_BASELINE": m1}, output)
    return verdict


def main(argv=None) -> None:
    """Run the command-line audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--before-module", type=Path, default=DEFAULT_BEFORE_MODULE)
    parser.add_argument("--frozen-verdict", type=Path, default=DEFAULT_FROZEN_VERDICT)
    parser.add_argument("--reuse-m0-after", type=Path)
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument("--before-window", type=Path, default=DEFAULT_BEFORE_WINDOW)
    parser.add_argument("--final-m1-only", action="store_true")
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--m0-timeline", type=Path)
    args = parser.parse_args(argv)
    if args.final_m1_only:
        if args.baseline_summary is None or args.m0_timeline is None:
            parser.error("--final-m1-only requires --baseline-summary and --m0-timeline")
        verdict = run_final_m1_only(args.frames, args.output_dir, args.frozen_verdict, args.baseline_summary, args.m0_timeline)
        print(f"verdict={verdict['verdict']} output={args.output_dir.resolve()}")
    elif args.targeted:
        verdict = run_targeted(args.frames, args.output_dir, args.frozen_verdict, args.before_window)
        print(f"targeted_verdict={verdict['targeted_verdict']} output={args.output_dir.resolve()}")
    else:
        verdict = run(args.frames, args.output_dir, args.before_module, args.frozen_verdict, args.reuse_m0_after)
        print(f"verdict={verdict['verdict']} output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
