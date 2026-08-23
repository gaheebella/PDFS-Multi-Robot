"""Validate the frozen LiDAR-profile detector's local active-Anchor transition.

This evaluation runner does not tune the detector and does not run a
Point Cloud detector. Ground-truth fields are copied only into audit outputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR","/tmp/pdfs_mpl_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from junction_detection.integration.run_lidar_local_corridor_estimation import _rear_start
from junction_detection.pointcloud.lidar_profile_junction_detector import GeometryProfileConfig,LidarProfileJunctionDetector
from pygame_simulator.pre_exploration_general_pipeline_simulator import LIDAR_MAX_RANGE,SimulationRunner

DEFAULT_OUTPUT=ROOT/"junction_detection/integration/output/active_anchor_transition"


def _write(path: Path, rows: list[dict]) -> None:
    """Write rows with stable columns."""
    if not rows: return
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run_case(case_id: str, frames: int, rear_start: bool=False) -> SimulationRunner:
    """Run the frozen profile detector with active control but no Point Cloud."""
    profile=LidarProfileJunctionDetector(GeometryProfileConfig(LIDAR_MAX_RANGE))
    runner=SimulationRunner(case_id,"local_forward",profile_detector=profile,hold_on_profile_detection=True)
    if rear_start: _rear_start(runner)
    for frame in range(frames): runner.step(frame)
    return runner


def first_time(rows: list[dict], key: str) -> float:
    """Return the first sampled timestamp whose Boolean field is true."""
    return next((float(row["timestamp"]) for row in rows if row.get(key)),math.nan)


def timeline(rows: list[dict]) -> list[dict]:
    """Select the state-machine columns requested by the active-Anchor audit."""
    return [{
        "frame":row["frame"],"time":row["timestamp"],"profile_junction_detected":row.get("profile_junction_detected",False),"detection_latched":row.get("junction_detection_latched",False),"phase":row.get("active_junction_phase",row["initialization_phase"]),
        "forward_propulsion_enabled":row.get("local_forward_propulsion_active",False),"leader_braking":row.get("leader_braking_active",False),
        "leader_speed":row.get("leader_speed",math.nan),"stationary":row.get("leader_stationary",False),
        "stationary_dwell":row.get("stationary_dwell_steps",0),"stationary_dwell_target":row.get("stationary_dwell_target",0),
        "anchor_active":row.get("provisional_fixed_anchor",False),"pointcloud_ready":row.get("pointcloud_ready",False),
        "first_detection_time":row.get("profile_detection_time",math.nan),"detection_latch_time":row.get("detection_latch_time",math.nan),"hold_entry_time":row.get("hold_entry_time",math.nan),"braking_start_time":row.get("braking_start_time",math.nan),"stationary_confirmation_time":row.get("stationary_confirmation_time",math.nan),"anchor_entry_time":row.get("anchor_entry_time",math.nan),
        "distance_since_detection_eval":row.get("leader_displacement_since_detection_eval",0.0),"distance_during_braking_eval":row.get("leader_displacement_during_braking_eval",0.0),"distance_since_anchor_eval":row.get("leader_displacement_since_anchor",0.0),
        "anchor_heading_drift_eval":row.get("leader_heading_change_since_anchor",0.0),"leader_gap":row.get("leader_to_front_pack_gap",math.nan),"connected_component_size":row.get("leader_connected_component_size",0),"follower_mean_speed":row.get("follower_mean_speed",0.0),
        "follower_moving_fraction":row.get("follower_moving_fraction",0.0),"mean_lateral_sph_force":row.get("mean_lateral_sph_force",0.0),
        "lateral_span":row["swarm_lateral_span_sanity"],"min_robot_distance":row["min_inter_robot_distance"],"overlap_pairs":row["overlap_pair_count"],"max_speed":row["max_speed"],"nan_inf_state_count":row["nan_inf_state_count"],"outside_free_space_robot_count":row["outside_free_space_robot_count"],"GT_phase_eval_only":row["gt_phase"],
    } for row in rows]


def summarize(case_id: str, rows: list[dict]) -> dict:
    """Summarize transitions, drift, follower motion, and physical sanity."""
    detection=first_time(rows,"profile_junction_detected"); latch=first_time(rows,"junction_detection_latched"); braking=first_time(rows,"leader_braking_active"); anchor=first_time(rows,"provisional_fixed_anchor")
    anchor_rows=[row for row in rows if row.get("provisional_fixed_anchor")]
    hold_rows=[row for row in rows if row.get("suspect_hold_active")]
    braking_rows=[row for row in rows if row.get("leader_braking_active")]
    pre_detection=[row for row in rows if not row.get("junction_detection_latched")]
    return {
        "map_case":case_id,"sample_count":len(rows),"first_profile_detection_time":detection,"detection_latch_time":latch,"hold_entry_time":first_time(rows,"suspect_hold_active"),"braking_start_time":braking,
        "stationary_confirmation_time":anchor,"anchor_entry_time":anchor,"detection_to_anchor_latency":anchor-detection if math.isfinite(anchor) and math.isfinite(detection) else math.nan,
        "leader_speed_at_detection":next((float(row["leader_speed"]) for row in rows if row.get("profile_junction_detected")),math.nan),
        "leader_distance_after_detection":max((float(row["leader_displacement_since_detection_eval"]) for row in hold_rows),default=0.0),"leader_distance_during_braking":max((float(row["leader_displacement_during_braking_eval"]) for row in braking_rows),default=0.0),
        "max_anchor_drift":max((float(row["leader_displacement_since_anchor"]) for row in anchor_rows),default=0.0),
        "max_anchor_heading_drift_deg":max((float(row["leader_heading_change_since_anchor"]) for row in anchor_rows),default=0.0),
        "post_anchor_follower_mean_speed":sum(float(row["follower_mean_speed"]) for row in anchor_rows)/max(len(anchor_rows),1),
        "post_anchor_follower_moving_fraction":sum(float(row["follower_moving_fraction"]) for row in anchor_rows)/max(len(anchor_rows),1),
        "post_anchor_mean_lateral_sph_force":sum(float(row["mean_lateral_sph_force"]) for row in anchor_rows)/max(len(anchor_rows),1),
        "pre_detection_max_overlap_pairs":max((int(row["overlap_pair_count"]) for row in pre_detection),default=0),"braking_max_overlap_pairs":max((int(row["overlap_pair_count"]) for row in braking_rows),default=0),"braking_min_robot_distance":min((float(row["min_inter_robot_distance"]) for row in braking_rows),default=math.nan),
        "max_overlap_pairs":max(int(row["overlap_pair_count"]) for row in rows),"max_nan_inf":max(int(row["nan_inf_state_count"]) for row in rows),"max_outside_free_space":max(int(row["outside_free_space_robot_count"]) for row in rows),
        "pointcloud_ready_before_anchor_count":sum(bool(row.get("pointcloud_ready")) and not bool(row.get("provisional_fixed_anchor")) for row in rows),"pointcloud_ready_final":bool(rows[-1].get("pointcloud_ready",False)),
    }


def plot_audit(path: Path, m0: list[dict], m1: list[dict]) -> None:
    """Save the single permitted transition overview figure."""
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=False)
    for axis,rows,title in zip(axes,(m0,m1),("M0 straight","M1 cross")):
        times=[row["timestamp"] for row in rows]
        axis.plot(times,[row["leader_speed"] for row in rows],label="leader speed")
        axis.plot(times,[float(row.get("junction_detection_latched",False))*8 for row in rows],label="detection latched x8")
        axis.plot(times,[float(row.get("provisional_fixed_anchor",False))*6 for row in rows],label="anchor x6")
        axis.set(title=title,ylabel="speed / state"); axis.grid(alpha=.25); axis.legend(loc="upper right")
    axes[-1].set_xlabel("time [s]"); fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig)


def main(argv=None) -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--frames",type=int,default=600); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); args=parser.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    runners={"M0_STRAIGHT":run_case("M0_STRAIGHT",args.frames),"M1_CROSS_BASELINE_REAR_START":run_case("M1_CROSS_BASELINE",args.frames,rear_start=True)}
    _write(args.output/"active_anchor_timeline_m0.csv",timeline(runners["M0_STRAIGHT"].rows))
    _write(args.output/"active_anchor_timeline_m1.csv",timeline(runners["M1_CROSS_BASELINE_REAR_START"].rows))
    summaries=[summarize(case,runner.rows) for case,runner in runners.items()]; _write(args.output/"active_anchor_summary.csv",summaries)
    m0,m1=summaries
    local_contract=LidarProfileJunctionDetector.RUNTIME_INPUTS==("angles_deg","ranges")
    passed=not math.isfinite(m0["first_profile_detection_time"]) and not math.isfinite(m0["detection_latch_time"]) and math.isfinite(m1["anchor_entry_time"]) and m1["max_anchor_drift"]<=1e-12 and m1["max_anchor_heading_drift_deg"]<=1e-12 and m1["max_nan_inf"]==0 and m1["post_anchor_follower_moving_fraction"]>0 and m1["post_anchor_mean_lateral_sph_force"]>0 and m1["pointcloud_ready_before_anchor_count"]==0 and local_contract
    verdict="ACTIVE_ANCHOR_TRANSITION_VALID" if passed else "VALIDATION_FAILED"
    _write(args.output/"active_anchor_verdict.csv",[{"verdict":verdict,"m0_false_transition":math.isfinite(m0["detection_latch_time"]),"m1_anchor_ready":math.isfinite(m1["anchor_entry_time"]),"runtime_inputs":"angles_deg,ranges,leader_local_velocity,body_heading,local_physics","runtime_local_contract":local_contract,"pointcloud_detector_executed":False}])
    plot_audit(args.output/"active_anchor_audit.png",runners["M0_STRAIGHT"].rows,runners["M1_CROSS_BASELINE_REAR_START"].rows)
    print(f"verdict={verdict} output={args.output.resolve()}")


if __name__=="__main__": main()
