"""Audit fixed-Anchor Point Cloud visibility without changing detector/control."""

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
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from junction_detection.integration.pointcloud_wall_parallel_orientation import estimate_wall_parallel_tangent
from junction_detection.integration.run_lidar_local_corridor_estimation import _rear_start
from junction_detection.pointcloud.lidar_profile_junction_detector import GeometryProfileConfig,LidarProfileJunctionDetector
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import detect_openings
from pygame_simulator.pre_exploration_general_pipeline_simulator import LIDAR_MAX_RANGE,SimulationRunner

DEFAULT_OUTPUT=ROOT/"junction_detection/integration/output/provisional_anchor_pointcloud_observation"


def _write(path: Path, rows: list[dict[str,Any]]) -> None:
    """Write a homogeneous CSV table."""
    if not rows: return
    fields=list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _normalize(angle: float) -> float:
    return float((angle+180.0)%360.0-180.0)


def _contains(opening: dict[str,float], angle: float) -> bool:
    """Return whether a circular direction lies inside an opening interval."""
    return (angle-float(opening["start_angle"]))%360.0<=float(opening["width_deg"])+1e-9


def _snapshot(runner: SimulationRunner, context: str) -> dict[str,Any]:
    """Copy the current body-local scan and evaluation-only pose."""
    scan=runner.last_visual[0].lidar_scan
    leader=next(robot for robot in runner.world.robots if robot.robot_id==runner.world.lidar_robot_id)
    hit=scan.ranges<scan.max_range-np.finfo(float).eps*max(1.0,scan.max_range)*64.0
    return {"context":context,"angles":scan.angles_deg.copy(),"ranges":scan.ranges.copy(),"hit":hit,"max_range":scan.max_range,"position_eval":leader.position.copy(),"yaw_eval":math.degrees(leader.body_yaw_rad),"frame":runner.rows[-1]["frame"],"time":runner.rows[-1]["timestamp"]}


def run_case(case_id: str, frames: int, rear_start: bool=False) -> tuple[SimulationRunner,dict[str,Any]|None,dict[str,Any]|None]:
    """Run active Anchor control and invoke Point Cloud exactly at ready edge."""
    profile=LidarProfileJunctionDetector(GeometryProfileConfig(LIDAR_MAX_RANGE))
    runner=SimulationRunner(case_id,"local_forward",profile_detector=profile,hold_on_profile_detection=True,pointcloud_detector=detect_openings)
    if rear_start: _rear_start(runner)
    detection=None; anchor=None
    for frame in range(frames):
        row=runner.step(frame)
        if row is None: continue
        if detection is None and row["profile_junction_detected"]:
            detection=_snapshot(runner,"DETECTION_POSE_EVAL_ONLY")
        if anchor is None and row["pointcloud_called"]:
            anchor=_snapshot(runner,"PROVISIONAL_ANCHOR")
            anchor["openings"]=list(runner.last_pointcloud_openings)
    return runner,detection,anchor


def _gt_directions_eval_only(runner: SimulationRunner, snapshot: dict[str,Any]) -> tuple[list[dict[str,Any]],dict[str,Any]]:
    """Convert simulator branch axes to local directions after detector output."""
    yaw=float(snapshot["yaw_eval"]); outgoing=[]
    for index,branch in enumerate(runner.geometry.branches):
        direction=np.array([math.sin(math.radians(branch.angle_deg)),math.cos(math.radians(branch.angle_deg))])
        world=math.degrees(math.atan2(float(direction[1]),float(direction[0])))
        outgoing.append({"branch_id":index,"local_angle_deg":_normalize(world-yaw)})
    incoming={"branch_id":"INCOMING","local_angle_deg":_normalize(-90.0-yaw)}
    return outgoing,incoming


def evaluate_snapshot(runner: SimulationRunner, snapshot: dict[str,Any], openings: list[dict[str,float]]) -> tuple[dict[str,Any],list[dict[str,Any]]]:
    """Post-hoc match openings and measure existing wall-tangent support."""
    outgoing,incoming=_gt_directions_eval_only(runner,snapshot)
    unmatched=set(range(len(openings))); matched=[]; opening_rows=[]
    for branch in outgoing:
        candidates=[index for index in unmatched if _contains(openings[index],branch["local_angle_deg"])]
        if candidates:
            index=min(candidates,key=lambda value:abs(_normalize(openings[value]["center_angle"]-branch["local_angle_deg"])))
            unmatched.remove(index); matched.append((branch,index))
    incoming_matches=[index for index in unmatched if _contains(openings[index],incoming["local_angle_deg"])]
    incoming_index=(min(incoming_matches,key=lambda value:abs(_normalize(openings[value]["center_angle"]-incoming["local_angle_deg"]))) if incoming_matches else None)
    if incoming_index is not None: unmatched.remove(incoming_index)
    center_errors=[]; fitted_points=[]; usable_sides=[]
    for index,opening in enumerate(openings):
        estimate=estimate_wall_parallel_tangent(snapshot["angles"],snapshot["ranges"],snapshot["max_range"],opening)
        branch=next((item for item,opening_index in matched if opening_index==index),None)
        incoming_match=index==incoming_index
        error=(abs(_normalize(opening["center_angle"]-branch["local_angle_deg"])) if branch is not None else abs(_normalize(opening["center_angle"]-incoming["local_angle_deg"])) if incoming_match else math.nan)
        if branch is not None: center_errors.append(error)
        fitted_points.append(estimate.fitted_point_count); usable_sides.append(estimate.usable_wall_sides)
        opening_rows.append({
            "map_case":runner.geometry.case_id,"scan_context":snapshot["context"],"opening_id":index,"start_angle_deg":opening["start_angle"],"end_angle_deg":opening["end_angle"],"center_angle_deg":opening["center_angle"],"angular_width_deg":opening["width_deg"],"confidence":opening.get("confidence",""),
            "wall_tangent_deg":"" if estimate.tangent_deg is None else estimate.tangent_deg,"usable_wall_sides":estimate.usable_wall_sides,"fitted_wall_point_count":estimate.fitted_point_count,"line_fit_residual":"" if estimate.line_fit_residual_m is None else estimate.line_fit_residual_m,"wall_estimate_mode":estimate.estimate_mode,
            "matched_GT_branch_eval_only":branch["branch_id"] if branch is not None else "INCOMING" if incoming_match else "","center_error_deg_eval_only":error,
        })
    hit=np.asarray(snapshot["hit"]); angles=np.asarray(snapshot["angles"])
    summary={
        "scan_context":snapshot["context"],"frame":snapshot["frame"],"time":snapshot["time"],"opening_count":len(openings),"GT_outgoing_branch_count_eval_only":len(outgoing),"matched_outgoing_count_eval_only":len(matched),"missed_outgoing_count_eval_only":len(outgoing)-len(matched),"incoming_opening_count_eval_only":int(incoming_index is not None),"false_opening_count_eval_only":len(unmatched),"mean_center_error_deg_eval_only":float(np.mean(center_errors)) if center_errors else math.nan,
        "valid_lidar_point_count":int(np.count_nonzero(hit)),"max_range_no_return_count":int(np.count_nonzero(~hit)),"front_hit_support":int(np.count_nonzero(hit&(np.abs(angles)<=45.0))),"left_hit_support":int(np.count_nonzero(hit&(angles>45.0)&(angles<135.0))),"right_hit_support":int(np.count_nonzero(hit&(angles<-45.0)&(angles>-135.0))),
        "wall_tangent_available_count":sum(value>0 for value in usable_sides),"total_fitted_wall_point_count":sum(fitted_points),"mean_usable_wall_sides":float(np.mean(usable_sides)) if usable_sides else 0.0,
        "GT_outgoing_local_angles_eval_only":json.dumps([item["local_angle_deg"] for item in outgoing]),"GT_incoming_local_angle_eval_only":incoming["local_angle_deg"],
    }
    return summary,opening_rows


def timeline(rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    """Select lifecycle and exactly-once Point Cloud audit fields."""
    return [{"map_case":row["map_case"],"frame":row["frame"],"timestamp":row["timestamp"],"profile_junction_detected":row["profile_junction_detected"],"junction_detection_latched":row["junction_detection_latched"],"braking_active":row["leader_braking_active"],"leader_speed":row["leader_speed"],"stationary_dwell_steps":row["stationary_dwell_steps"],"provisional_fixed_anchor":row["provisional_fixed_anchor"],"pointcloud_ready":row["pointcloud_ready"],"pointcloud_called":row["pointcloud_called"],"pointcloud_opening_count":row["pointcloud_opening_count"],"primary_pointcloud_call_count":row["primary_pointcloud_call_count"],"pointcloud_called_before_ready_count":row["pointcloud_called_before_ready_count"],"anchor_entry_time":row["anchor_entry_time"],"anchor_x_eval_only":row["anchor_x_eval"],"anchor_y_eval_only":row["anchor_y_eval"],"anchor_heading_local_deg":row["anchor_heading_deg"],"detection_to_anchor_distance_eval_only":row["leader_displacement_since_detection_eval"],"anchor_drift_eval_only":row["leader_displacement_since_anchor"],"GT_phase_eval_only":row["gt_phase"],"nan_inf":row["nan_inf_state_count"],"outside":row["outside_free_space_robot_count"]} for row in rows]


def plot_snapshot(path: Path, snapshot: dict[str,Any], openings: list[dict[str,float]], gt_angles: list[float]) -> None:
    """Plot one Anchor-local Point Cloud with runtime and GT layers separated."""
    theta=np.deg2rad(snapshot["angles"]); x=snapshot["ranges"]*np.cos(theta); y=snapshot["ranges"]*np.sin(theta); hit=snapshot["hit"]
    fig,axis=plt.subplots(figsize=(6.5,6.5)); axis.scatter(x[hit],y[hit],s=8,label="wall return"); axis.scatter(x[~hit],y[~hit],s=5,alpha=.18,label="max-range")
    for index,opening in enumerate(openings):
        angle=math.radians(opening["center_angle"]); axis.plot([0,snapshot["max_range"]*math.cos(angle)],[0,snapshot["max_range"]*math.sin(angle)],color="tab:orange",linewidth=2,label="detected opening" if index==0 else None)
    for index,angle_deg in enumerate(gt_angles):
        angle=math.radians(angle_deg); axis.plot([0,snapshot["max_range"]*math.cos(angle)],[0,snapshot["max_range"]*math.sin(angle)],"--",color="tab:red",label="GT outgoing EVAL ONLY" if index==0 else None)
    axis.scatter([0],[0],marker="*",s=120,color="black",label="viewpoint"); axis.set_aspect("equal"); axis.set(title=snapshot["context"],xlabel="Anchor-local x",ylabel="Anchor-local y"); axis.grid(alpha=.25); axis.legend(loc="upper right"); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)


def main(argv=None) -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--frames",type=int,default=600); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); args=parser.parse_args(argv); args.output.mkdir(parents=True,exist_ok=True)
    m0,_,_=run_case("M0_STRAIGHT",args.frames); m1,detection,anchor=run_case("M1_CROSS_BASELINE",args.frames,rear_start=True)
    if detection is None or anchor is None: raise RuntimeError("M1 did not reach detection and primary Point Cloud observation")
    detection_openings=list(detect_openings(detection["angles"],detection["ranges"])); anchor_openings=list(anchor["openings"])
    detection_summary,detection_rows=evaluate_snapshot(m1,detection,detection_openings); anchor_summary,anchor_rows=evaluate_snapshot(m1,anchor,anchor_openings)
    _write(args.output/"provisional_anchor_pointcloud_timeline_m0.csv",timeline(m0.rows)); _write(args.output/"provisional_anchor_pointcloud_timeline_m1.csv",timeline(m1.rows)); _write(args.output/"provisional_anchor_pointcloud_openings.csv",detection_rows+anchor_rows)
    m0_summary={"map_case":"M0_STRAIGHT","profile_detection_count":sum(row["profile_junction_detected"] for row in m0.rows),"pointcloud_ready_count":sum(row["pointcloud_ready"] for row in m0.rows),"primary_pointcloud_call_count":m0.primary_pointcloud_call_count,"pointcloud_called_before_ready_count":m0.pointcloud_called_before_ready_count}
    detection_time=next(row["timestamp"] for row in m1.rows if row["profile_junction_detected"]); anchor_time=next(row["timestamp"] for row in m1.rows if row["provisional_fixed_anchor"])
    m1_summary={"map_case":"M1_CROSS_BASELINE_REAR_START","profile_detection_count":sum(row["profile_junction_detected"] for row in m1.rows),"pointcloud_ready_count":sum(row["pointcloud_ready"] for row in m1.rows),"profile_detection_time":detection_time,"anchor_time":anchor_time,"detection_to_anchor_latency":anchor_time-detection_time,"detection_to_anchor_displacement":next(row["leader_displacement_since_detection_eval"] for row in m1.rows if row["pointcloud_called"]),"anchor_x_eval_only":float(m1.world.anchor_position[0]),"anchor_y_eval_only":float(m1.world.anchor_position[1]),"anchor_heading_local_deg":math.degrees(m1.world.anchor_heading_rad),"pointcloud_call_time":m1.pointcloud_call_time,"primary_pointcloud_call_count":m1.primary_pointcloud_call_count,"pointcloud_called_before_ready_count":m1.pointcloud_called_before_ready_count,"max_anchor_drift":max(row["leader_displacement_since_anchor"] for row in m1.rows),"max_nan_inf":max(row["nan_inf_state_count"] for row in m1.rows),"max_outside":max(row["outside_free_space_robot_count"] for row in m1.rows),**{f"anchor_{key}":value for key,value in anchor_summary.items() if key not in ("scan_context","frame","time")}}
    _write(args.output/"provisional_anchor_pointcloud_summary.csv",[m0_summary,m1_summary])
    comparison=[]
    for key in ("opening_count","matched_outgoing_count_eval_only","missed_outgoing_count_eval_only","incoming_opening_count_eval_only","false_opening_count_eval_only","mean_center_error_deg_eval_only","valid_lidar_point_count","max_range_no_return_count","wall_tangent_available_count","total_fitted_wall_point_count","mean_usable_wall_sides"):
        comparison.append({"metric":key,"detection_pose":detection_summary[key],"provisional_anchor_pose":anchor_summary[key],"anchor_minus_detection":anchor_summary[key]-detection_summary[key]})
    _write(args.output/"anchor_pose_visibility_comparison.csv",comparison)
    matched=anchor_summary["matched_outgoing_count_eval_only"]
    verdict="PROVISIONAL_ANCHOR_VIEWPOINT_SUFFICIENT" if matched==anchor_summary["GT_outgoing_branch_count_eval_only"] else "PROVISIONAL_ANCHOR_VIEWPOINT_PARTIAL" if matched>1 else "PROVISIONAL_ANCHOR_VIEWPOINT_INSUFFICIENT"
    _write(args.output/"provisional_anchor_pointcloud_verdict.csv",[{"verdict":verdict,"evaluation_only":True,"production_viewpoint_threshold":False,"pointcloud_detector_modified":False,"profile_detector_modified":False,"braking_modified":False,"final_fixed_anchor_promoted":False}])
    gt_angles=json.loads(anchor_summary["GT_outgoing_local_angles_eval_only"]); plot_snapshot(args.output/"pointcloud_at_detection_pose.png",detection,detection_openings,json.loads(detection_summary["GT_outgoing_local_angles_eval_only"])); plot_snapshot(args.output/"pointcloud_at_anchor_pose.png",anchor,anchor_openings,gt_angles)
    fig,axes=plt.subplots(1,2,figsize=(11,4)); labels=["detection","anchor"]; axes[0].bar(labels,[detection_summary["matched_outgoing_count_eval_only"],anchor_summary["matched_outgoing_count_eval_only"]]); axes[0].set(title="Matched outgoing branches (EVAL ONLY)",ylim=(0,3)); axes[1].bar(labels,[detection_summary["total_fitted_wall_point_count"],anchor_summary["total_fitted_wall_point_count"]]); axes[1].set(title="Observed wall-fit point support"); fig.tight_layout(); fig.savefig(args.output/"provisional_anchor_pointcloud_audit.png",dpi=150); plt.close(fig)
    print(f"verdict={verdict} anchor_openings={len(anchor_openings)} matched={matched}/{anchor_summary['GT_outgoing_branch_count_eval_only']} output={args.output.resolve()}")


if __name__=="__main__": main()
