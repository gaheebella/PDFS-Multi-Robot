"""Compare baseline and local support-aware viewpoint movement physics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR","/tmp/pdfs_mpl_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from junction_detection.integration.run_lidar_local_corridor_estimation import _rear_start
from junction_detection.pointcloud.lidar_profile_junction_detector import GeometryProfileConfig,LidarProfileJunctionDetector
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import detect_openings
from pygame_simulator.pre_exploration_general_pipeline_simulator import (
    ActiveViewpointConfig,LIDAR_MAX_RANGE,SimulationRunner,SMOOTHING_LENGTH,
)

DEFAULT_OUTPUT=ROOT/"junction_detection/integration/output/viewpoint_movement_physics"


def _write(path: Path, rows: list[dict[str,Any]]) -> None:
    """Write heterogeneous dictionaries as one CSV table."""
    if not rows:
        return
    fields=list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def run_case(case_id: str, frames: int, max_rescans: int, improved: bool, rear_start: bool=False) -> SimulationRunner:
    """Run one case; improved control uses only local motion/support inputs."""
    profile=LidarProfileJunctionDetector(GeometryProfileConfig(LIDAR_MAX_RANGE))
    config=ActiveViewpointConfig(0.10,max_rescans,improved,improved)
    runner=SimulationRunner(case_id,"local_forward",profile_detector=profile,hold_on_profile_detection=True,
                            pointcloud_detector=detect_openings,active_viewpoint_config=config)
    if rear_start:
        _rear_start(runner)
    for frame in range(frames):
        runner.step(frame)
    return runner


def _mode_rows(runner: SimulationRunner, mode: str) -> list[dict[str,Any]]:
    """Copy physics-step local diagnostics with an experiment mode label."""
    return [{"mode":mode,**row} for row in runner.world.viewpoint_motion_physics_rows]


def _step_rows(runner: SimulationRunner, mode: str) -> list[dict[str,Any]]:
    """Copy per-step displacement decomposition records."""
    return [{"mode":mode,**row} for row in runner.world.viewpoint_motion_step_records]


def _physical_summary(runner: SimulationRunner, mode: str) -> dict[str,Any]:
    """Summarize sampled physical and communication safety metrics."""
    active=[row for row in runner.rows if row.get("active_viewpoint_state") in ("VIEWPOINT_ADVANCE","VIEWPOINT_REBRAKING")]
    if not active:
        return {"mode":mode,"active_sample_count":0,"min_inter_robot_distance":"","max_overlap_pair_count":0,"wall_contacts_added":0,"max_outside":0,"max_nan_inf":0,"leader_max_speed":0.0,"follower_mean_speed_max":0.0,"direct_support_all_active":True,"communication_connected_all_active":True,"minimum_communication_component":len(runner.world.robots)}
    return {
        "mode":mode,"active_sample_count":len(active),"min_inter_robot_distance":min(row["min_inter_robot_distance"] for row in active),
        "max_overlap_pair_count":max(row["overlap_pair_count"] for row in active),
        "wall_contacts_added":max(row["wall_contact_count"] for row in active)-min(row["wall_contact_count"] for row in active),
        "max_outside":max(row["outside_free_space_robot_count"] for row in active),"max_nan_inf":max(row["nan_inf_state_count"] for row in active),
        "leader_max_speed":max(row["leader_speed"] for row in active),"follower_mean_speed_max":max(row["follower_mean_speed"] for row in active),
        "direct_support_all_active":all(row["leader_connected"] for row in active),
        "communication_connected_all_active":all(row["leader_communication_connected"] for row in active),
        "minimum_communication_component":min(row["leader_connected_component_size"] for row in active),
        "maximum_support_gap":max(row["leader_to_front_pack_gap"] for row in active),
        "pointcloud_call_count_secondary":runner.pointcloud_call_count,
    }


def _comparison(baseline: list[dict[str,Any]], improved: list[dict[str,Any]]) -> list[dict[str,Any]]:
    """Pair baseline and improved movement outcomes by bounded step index."""
    result=[]
    by_baseline={int(row["step_index"]):row for row in baseline}; by_improved={int(row["step_index"]):row for row in improved}
    for step in sorted(set(by_baseline)|set(by_improved)):
        b=by_baseline.get(step,{}); i=by_improved.get(step,{})
        result.append({
            "step_index":step,"target_advance":i.get("target_advance",b.get("target_advance","")),
            "baseline_actual":b.get("actual_advance",""),"improved_actual":i.get("actual_advance",""),
            "baseline_overshoot_ratio":b.get("overshoot_ratio",""),"improved_overshoot_ratio":i.get("overshoot_ratio",""),
            "overshoot_ratio_reduction":(float(b["overshoot_ratio"])-float(i["overshoot_ratio"]) if b and i else ""),
            "baseline_brake_start_progress":b.get("brake_start_progress",""),"improved_brake_start_progress":i.get("brake_start_progress",""),
            "baseline_speed_at_brake":b.get("speed_at_brake_start",""),"improved_speed_at_brake":i.get("speed_at_brake_start",""),
            "baseline_braking_plus_dwell":(float(b["braking_distance"])+float(b["dwell_distance"]) if b else ""),
            "improved_braking_plus_dwell":(float(i["braking_distance"])+float(i["dwell_distance"]) if i else ""),
            "baseline_support_gap_anchor":b.get("support_gap_at_anchor",""),"improved_support_gap_anchor":i.get("support_gap_at_anchor",""),
            "baseline_direct_support_anchor":b.get("direct_support_at_anchor",""),"improved_direct_support_anchor":i.get("direct_support_at_anchor",""),
            "improved_stop_reason":i.get("stop_reason",""),
        })
    return result


def _deterministic_equal(reference: SimulationRunner, replay: SimulationRunner) -> bool:
    """Require exact local physics-step and Anchor-step diagnostic replay."""
    a=reference.world.viewpoint_motion_physics_rows; b=replay.world.viewpoint_motion_physics_rows
    if len(a)!=len(b) or reference.world.viewpoint_motion_step_records!=replay.world.viewpoint_motion_step_records:
        return False
    keys=("viewpoint_step_index","state","leader_speed","leader_forward_speed","local_progress","predicted_stopping_distance","support_gap","direct_support_present")
    return all(all(left[key]==right[key] for key in keys) for left,right in zip(a,b))


def _plots(output: Path, baseline: list[dict[str,Any]], improved: list[dict[str,Any]], timeline: list[dict[str,Any]]) -> None:
    """Generate the requested compact motion/control diagnostics."""
    steps=range(1,len(improved)+1); target=[float(row["target_advance"]) for row in improved]
    fig,axis=plt.subplots(figsize=(7,4)); width=.25; x=np.arange(len(improved))
    axis.bar(x-width,[float(row["actual_advance"]) for row in baseline[:len(improved)]],width,label="baseline actual")
    axis.bar(x,[float(row["actual_advance"]) for row in improved],width,label="improved actual")
    axis.bar(x+width,target,width,label="target"); axis.set(xticks=x,xticklabels=list(steps),xlabel="viewpoint step",ylabel="local distance",title="Target vs actual displacement"); axis.legend(); fig.tight_layout(); fig.savefig(output/"target_vs_actual_displacement.png",dpi=150); plt.close(fig)
    improved_t=[row for row in timeline if row["mode"]=="IMPROVED"]
    fig,axis=plt.subplots(figsize=(7,4)); axis.plot([row["local_progress"] for row in improved_t],[row["leader_forward_speed"] for row in improved_t]); axis.set(xlabel="step-local progress",ylabel="forward speed",title="Improved speed vs progress"); axis.grid(alpha=.25); fig.tight_layout(); fig.savefig(output/"speed_vs_progress.png",dpi=150); plt.close(fig)
    fig,axis=plt.subplots(figsize=(7,4)); axis.plot([row["local_progress"] for row in improved_t],[row["support_gap"] for row in improved_t],label="direct support gap"); axis.axhline(SMOOTHING_LENGTH,color="tab:red",linestyle="--",label="SPH support radius"); axis.set(xlabel="step-local progress",ylabel="gap",title="Support gap vs progress"); axis.legend(); axis.grid(alpha=.25); fig.tight_layout(); fig.savefig(output/"support_gap_vs_progress.png",dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4)); axes[0].plot(list(steps),[float(row["overshoot_ratio"]) for row in baseline[:len(improved)]],"o-",label="baseline"); axes[0].plot(list(steps),[float(row["overshoot_ratio"]) for row in improved],"o-",label="improved"); axes[0].set(title="Overshoot ratio",xlabel="step"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(list(steps),[float(row["support_gap_at_anchor"]) for row in improved],"o-"); axes[1].axhline(SMOOTHING_LENGTH,color="tab:red",linestyle="--"); axes[1].set(title="Improved Anchor support gap",xlabel="step"); axes[1].grid(alpha=.25); fig.tight_layout(); fig.savefig(output/"viewpoint_motion_audit.png",dpi=150); plt.close(fig)


def main(argv=None) -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-frames",type=int,default=300); parser.add_argument("--single-frames",type=int,default=240)
    parser.add_argument("--improved-frames",type=int,default=600); parser.add_argument("--m0-frames",type=int,default=600)
    parser.add_argument("--replay-frames",type=int,default=300); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args(argv); args.output.mkdir(parents=True,exist_ok=True)

    baseline=run_case("M1_CROSS_BASELINE",args.baseline_frames,3,False,True)
    single=run_case("M1_CROSS_BASELINE",args.single_frames,1,True,True)
    improved=run_case("M1_CROSS_BASELINE",args.improved_frames,3,True,True)
    m0=run_case("M0_STRAIGHT",args.m0_frames,3,True,False)
    replay=run_case("M1_CROSS_BASELINE",args.replay_frames,3,True,True)
    if len(baseline.world.viewpoint_motion_step_records)<3 or len(improved.world.viewpoint_motion_step_records)<3 or len(single.world.viewpoint_motion_step_records)!=1:
        raise RuntimeError("bounded movement sequence did not complete")

    baseline_steps=_step_rows(baseline,"BASELINE"); improved_steps=_step_rows(improved,"IMPROVED")
    timeline=_mode_rows(baseline,"BASELINE")+_mode_rows(improved,"IMPROVED")
    support=[{key:row[key] for key in ("mode","physics_frame","timestamp","viewpoint_step_index","state","local_progress","predicted_stopping_distance","support_gap","direct_support_present","communication_connected","communication_component_size","brake_trigger_reason")} for row in timeline]
    comparison=_comparison(baseline_steps,improved_steps)
    deterministic=_deterministic_equal(improved,replay)
    m0_ok=(not m0.world.junction_detection_latched and not m0.world.provisional_fixed_anchor and not m0.world.viewpoint_motion_physics_rows and m0.pointcloud_call_count==0)
    improved_support=all(bool(row["direct_support_at_anchor"]) for row in improved_steps)
    improved_safe=all(float(row["max_outside"])==0 and float(row["max_nan_inf"])==0 and int(row["max_overlap_pair_count"])==0 for row in [_physical_summary(improved,"IMPROVED")])
    baseline_error=float(np.mean([abs(float(row["overshoot_ratio"])) for row in baseline_steps])); improved_error=float(np.mean([abs(float(row["overshoot_ratio"])) for row in improved_steps]))
    support_guard_count=sum(row["stop_reason"]=="DIRECT_SUPPORT_GUARD" for row in improved_steps)
    if improved_safe and improved_support and improved_error<baseline_error*0.5:
        verdict="VIEWPOINT_MOVEMENT_PHYSICS_VALID"
    elif improved_error<baseline_error*0.5 and support_guard_count:
        verdict="VIEWPOINT_MOVEMENT_OVERSHOOT_IMPROVED_SUPPORT_LIMITED"
    elif improved_support:
        verdict="VIEWPOINT_MOVEMENT_SUPPORT_VALID_OVERSHOOT_HIGH"
    else:
        verdict="VIEWPOINT_MOVEMENT_PHYSICS_INVALID"

    _write(args.output/"viewpoint_motion_timeline.csv",timeline)
    _write(args.output/"viewpoint_motion_steps.csv",baseline_steps+improved_steps)
    _write(args.output/"viewpoint_motion_baseline_vs_improved.csv",comparison)
    _write(args.output/"viewpoint_support_diagnostics.csv",support)
    physical=[_physical_summary(baseline,"BASELINE"),_physical_summary(single,"IMPROVED_SINGLE_STEP"),_physical_summary(improved,"IMPROVED"),_physical_summary(m0,"M0_NEGATIVE_CONTROL")]
    _write(args.output/"viewpoint_physics_summary.csv",physical+[{"mode":"M0_ASSERTIONS","m0_negative_control_pass":m0_ok,"deterministic_replay_pass":deterministic,"baseline_mean_abs_overshoot_ratio":baseline_error,"improved_mean_abs_overshoot_ratio":improved_error,"support_guard_count":support_guard_count}])
    _write(args.output/"viewpoint_physics_verdict.csv",[{"verdict":verdict,"global_pose_used_for_control":False,"GT_used_for_control":False,"detectors_modified":False,"m0_negative_control_pass":m0_ok,"deterministic_replay_pass":deterministic}])
    _plots(args.output,baseline_steps,improved_steps,timeline)
    print(json.dumps({"verdict":verdict,"baseline_actual":[row["actual_advance"] for row in baseline_steps],"improved_actual":[row["actual_advance"] for row in improved_steps],"improved_stop_reasons":[row["stop_reason"] for row in improved_steps],"support_at_anchor":[row["direct_support_at_anchor"] for row in improved_steps],"m0":m0_ok,"deterministic":deterministic,"output":str(args.output.resolve())},indent=2))


if __name__=="__main__":
    main()
