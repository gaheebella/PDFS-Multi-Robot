"""Re-evaluate Point Cloud visibility using frozen safe viewpoint motion."""

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

from junction_detection.integration.run_active_viewpoint_acquisition import (
    _evaluate_scans,_gt_directions_eval_only,_scan_snapshot,_timeline,_write,plot_snapshot,run_case,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import ActiveViewpointConfig,SMOOTHING_LENGTH

DEFAULT_OUTPUT=ROOT/"junction_detection/integration/output/safe_active_viewpoint_visibility"
UNSAFE_SCANS=ROOT/"junction_detection/integration/output/active_viewpoint_acquisition/active_viewpoint_scans.csv"


def _branch_visibility(visibility: list[dict[str,Any]], scan_index: int, angle: float) -> bool:
    """Return one post-hoc GT branch visibility flag."""
    return any(int(row["scan_index"])==scan_index and float(row["GT_branch_angle_deg_eval_only"])==angle and bool(row["visible_as_opening_eval_only"]) for row in visibility)


def _safe_scan_rows(runner, scans: list[dict[str,Any]], visibility: list[dict[str,Any]]) -> list[dict[str,Any]]:
    """Create the requested scan table by joining motion and scan records."""
    steps={int(row["step_index"]):row for row in runner.world.viewpoint_motion_step_records}
    result=[]
    for row in scans:
        index=int(row["scan_index"]); step=steps.get(index,{})
        result.append({
            "map_case":row["map_case"],"scan_index":index,"timestamp":row["timestamp"],"anchor_time":row["timestamp"],
            "step_local_advance":float(step.get("actual_advance",0.0)),"cumulative_local_advance":row["cumulative_local_advance"],
            "cumulative_width_ratio":row["cumulative_advance_width_ratio"],"target_advance":float(step.get("target_advance",0.0)),
            "actual_advance":float(step.get("actual_advance",0.0)),"overshoot_ratio":float(step.get("overshoot_ratio",0.0)),
            "leader_speed_at_scan":row["leader_speed_at_scan"],"support_gap":row["leader_front_pack_gap"],
            "direct_support_present":row["leader_support_connected"],"communication_component":row["leader_component_size"],
            "opening_count":row["opening_count"],"matched_outgoing_eval_only":row["matched_outgoing_count_eval_only"],
            "missed_outgoing_eval_only":row["missed_outgoing_count_eval_only"],"false_openings_eval_only":row["false_opening_count_eval_only"],
            "plus90_visible_eval_only":_branch_visibility(visibility,index,90.0),"minus90_visible_eval_only":_branch_visibility(visibility,index,-90.0),
            "incoming_visible_eval_only":bool(row["incoming_opening_count_eval_only"]),"forward_visible_eval_only":_branch_visibility(visibility,index,0.0),
            "mean_center_error_deg_eval_only":row["mean_center_error_deg_eval_only"],"mean_GT_mouth_IoU_eval_only":row["mean_GT_mouth_interval_iou_eval_only"],
            "valid_hit_count":row["valid_lidar_point_count"],"max_range_no_return_count":row["max_range_no_return_count"],
            "total_wall_fitted_points":row["total_fitted_wall_point_count"],"wall_tangent_available_count":row["wall_tangent_available_count"],
            "front_hit_support":row["front_hit_support"],"left_hit_support":row["left_hit_support"],"right_hit_support":row["right_hit_support"],
            "anchor_x_eval_only":row["anchor_x_eval_only"],"anchor_y_eval_only":row["anchor_y_eval_only"],"anchor_yaw_eval_only":row["anchor_yaw_eval_only"],
        })
    return result


def _safe_opening_rows(runner, rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    """Normalize existing wall/opening evaluation into the requested schema."""
    result=[]
    for row in rows:
        match=row["matched_GT_branch_eval_only"]
        if isinstance(match,int):
            gt_class="OUTGOING"; gt_direction=runner.geometry.branches[match].angle_deg
        elif match=="INCOMING":
            gt_class="INCOMING"; gt_direction=-180.0
        else:
            gt_class="FALSE"; gt_direction=""
        result.append({
            "map_case":row["map_case"],"scan_index":row["scan_index"],"opening_id":row["opening_id"],
            "start_angle_deg":row["start_angle_deg"],"end_angle_deg":row["end_angle_deg"],"center_angle_deg":row["center_angle_deg"],
            "angular_width_deg":row["angular_width_deg"],"confidence":row["confidence"],
            "two_wall_parallel":row["wall_estimate_mode"]=="two_wall_parallel","usable_wall_sides":row["usable_wall_sides"],
            "wall_fitted_points":row["fitted_wall_point_count"],"line_residual":row["line_fit_residual"],
            "tangent_available":row["wall_tangent_deg"]!="","tangent_deg":row["wall_tangent_deg"],
            "GT_class_eval_only":gt_class,"matched_GT_direction_eval_only":gt_direction,
            "center_error_deg_eval_only":row["center_error_deg_eval_only"],"IoU_eval_only":row["GT_mouth_interval_iou_eval_only"],
        })
    return result


def _physics_rows(runner) -> list[dict[str,Any]]:
    """Join safe movement decomposition with sampled physical sanity."""
    result=[]
    for step in runner.world.viewpoint_motion_step_records:
        index=int(step["step_index"])
        samples=[row for row in runner.rows if int(row.get("viewpoint_scan_index",-1))==index and row.get("active_viewpoint_state") in ("VIEWPOINT_ADVANCE","VIEWPOINT_REBRAKING")]
        result.append({
            "step_index":index,"target_advance":step["target_advance"],"actual_advance":step["actual_advance"],"overshoot_ratio":step["overshoot_ratio"],
            "brake_start_speed":step["speed_at_brake_start"],"predicted_stop_distance":step["predicted_stop_distance_at_trigger"],
            "braking_distance":step["braking_distance"],"dwell_distance":step["dwell_distance"],
            "support_gap_start":step["support_gap_start"],"support_gap_anchor":step["support_gap_at_anchor"],
            "direct_support_anchor":step["direct_support_at_anchor"],"support_guard_triggered":step["stop_reason"]=="DIRECT_SUPPORT_GUARD",
            "stop_reason":step["stop_reason"],"communication_component":step["communication_component_at_anchor"],
            "min_robot_distance":min(row["min_inter_robot_distance"] for row in samples),"overlap_pairs":max(row["overlap_pair_count"] for row in samples),
            "outside_count":max(row["outside_free_space_robot_count"] for row in samples),"nan_inf_count":max(row["nan_inf_state_count"] for row in samples),
            "leader_max_speed":max(row["leader_speed"] for row in samples),"follower_mean_speed_max":max(row["follower_mean_speed"] for row in samples),
        })
    return result


def _timeline_safe(runner) -> list[dict[str,Any]]:
    """Extend the existing sampled lifecycle with safe-motion fields."""
    rows=_timeline(runner)
    extras=("viewpoint_step_actual_progress","viewpoint_remaining_progress","viewpoint_predicted_stopping_distance","viewpoint_brake_trigger_reason")
    for output,source in zip(rows,runner.rows):
        output.update({key:source.get(key,"") for key in extras})
    return rows


def _unsafe_reference() -> list[dict[str,Any]]:
    """Load the frozen previous unsafe visibility run for evaluation only."""
    if not UNSAFE_SCANS.exists():
        return []
    with UNSAFE_SCANS.open(newline="",encoding="utf-8") as handle:
        return [{"source":"PREVIOUS_UNSAFE",**row} for row in csv.DictReader(handle)]


def _deterministic_equal(first, second) -> bool:
    """Compare movement, scan, detector, GT evaluation, and support outputs."""
    if first.world.viewpoint_motion_step_records!=second.world.viewpoint_motion_step_records:
        return False
    if len(first.pointcloud_history)!=len(second.pointcloud_history):
        return False
    first_scans,_,first_visibility=_evaluate_scans(first); second_scans,_,second_visibility=_evaluate_scans(second)
    keys=("scan_index","cumulative_local_advance","opening_count","matched_outgoing_count_eval_only","leader_front_pack_gap")
    for a,b,raw_a,raw_b in zip(first_scans,second_scans,first.pointcloud_history,second.pointcloud_history):
        if any(a[key]!=b[key] for key in keys) or not np.array_equal(raw_a["ranges"],raw_b["ranges"]):
            return False
    return first_visibility==second_visibility


def _plot(path: Path, scans: list[dict[str,Any]], physics: list[dict[str,Any]]) -> None:
    """Plot visibility, geometry support, safety gap, and overshoot."""
    x=[row["cumulative_width_ratio"] for row in scans]
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    axes[0,0].plot(x,[row["matched_outgoing_eval_only"] for row in scans],"o-",label="matched outgoing")
    axes[0,0].plot(x,[row["opening_count"] for row in scans],"s--",label="all openings"); axes[0,0].set(title="Visibility (GT post-hoc)",xlabel="cumulative advance / width"); axes[0,0].legend(); axes[0,0].grid(alpha=.25)
    axes[0,1].plot(x,[row["total_wall_fitted_points"] for row in scans],"o-",label="wall fitted points")
    axes[0,1].plot(x,[row["valid_hit_count"] for row in scans],"s--",label="valid hits"); axes[0,1].set(title="Point Cloud support",xlabel="cumulative advance / width"); axes[0,1].legend(); axes[0,1].grid(alpha=.25)
    axes[1,0].plot(x,[row["support_gap"] for row in scans],"o-"); axes[1,0].axhline(SMOOTHING_LENGTH,color="tab:red",linestyle="--",label="SPH support radius"); axes[1,0].set(title="Direct support gap",xlabel="cumulative advance / width"); axes[1,0].legend(); axes[1,0].grid(alpha=.25)
    axes[1,1].bar([row["step_index"] for row in physics],[row["overshoot_ratio"] for row in physics]); axes[1,1].set(title="Safe movement overshoot",xlabel="step",ylabel="ratio"); axes[1,1].grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)


def main(argv=None) -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--frames",type=int,default=600); parser.add_argument("--replay-frames",type=int,default=300); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args(argv); args.output.mkdir(parents=True,exist_ok=True)
    config=ActiveViewpointConfig(0.10,3,True,True)
    m0=run_case("M0_STRAIGHT",args.frames,config); m1=run_case("M1_CROSS_BASELINE",args.frames,config,rear_start=True)
    replay=run_case("M1_CROSS_BASELINE",args.replay_frames,config,rear_start=True)
    if len(m1.pointcloud_history)!=4 or len(m1.world.viewpoint_motion_step_records)!=3:
        raise RuntimeError("safe fixed-viewpoint scan sequence incomplete")
    scans_raw,openings_raw,visibility=_evaluate_scans(m1); scans=_safe_scan_rows(m1,scans_raw,visibility); openings=_safe_opening_rows(m1,openings_raw); physics=_physics_rows(m1)
    deterministic=_deterministic_equal(m1,replay)
    m0_ok=(not m0.world.junction_detection_latched and not m0.world.provisional_fixed_anchor and not m0.world.viewpoint_motion_physics_rows and m0.pointcloud_call_count==0)
    safe=all(row["direct_support_anchor"] and row["overlap_pairs"]==0 and row["outside_count"]==0 and row["nan_inf_count"]==0 for row in physics)
    first=scans[0]["matched_outgoing_eval_only"]; best=max(row["matched_outgoing_eval_only"] for row in scans)
    plus_first=next((row for row in scans if row["plus90_visible_eval_only"]),None); minus_first=next((row for row in scans if row["minus90_visible_eval_only"]),None)
    if not safe:
        verdict="SAFE_ACTIVE_VIEWPOINT_PHYSICS_REGRESSION"
    elif best==3 and plus_first is not None and minus_first is not None:
        verdict="SAFE_ACTIVE_VIEWPOINT_VISIBILITY_EFFECTIVE"
    elif best>first:
        verdict="SAFE_ACTIVE_VIEWPOINT_VISIBILITY_PARTIAL"
    else:
        verdict="SAFE_ACTIVE_VIEWPOINT_VISIBILITY_INEFFECTIVE"

    _write(args.output/"safe_viewpoint_timeline_m0.csv",_timeline_safe(m0)); _write(args.output/"safe_viewpoint_timeline_m1.csv",_timeline_safe(m1))
    _write(args.output/"safe_viewpoint_scans.csv",scans); _write(args.output/"safe_viewpoint_openings.csv",openings); _write(args.output/"safe_viewpoint_visibility.csv",visibility); _write(args.output/"safe_viewpoint_physics.csv",physics)
    unsafe=_unsafe_reference(); comparison=unsafe+[{"source":"SAFE",**row} for row in scans]
    summary={"verdict":verdict,"scan_count":len(scans),"initial_opening_count":scans[0]["opening_count"],"initial_outgoing_match":first,"best_outgoing_match":best,"best_wall_fitted_points":max(row["total_wall_fitted_points"] for row in scans),"visibility_improved":best>first,"safe_movement_maintained":safe,"plus90_first_scan":"" if plus_first is None else plus_first["scan_index"],"plus90_first_advance":"" if plus_first is None else plus_first["cumulative_local_advance"],"minus90_first_scan":"" if minus_first is None else minus_first["scan_index"],"minus90_first_advance":"" if minus_first is None else minus_first["cumulative_local_advance"],"profile_detection_time":next(row["timestamp"] for row in m1.rows if row["profile_junction_detected"]),"initial_anchor_time":m1.pointcloud_history[0]["timestamp"],"m0_negative_control_pass":m0_ok,"deterministic_replay_pass":deterministic,"GT_used_for_control":False,"global_pose_used_for_control":False,"movement_physics_modified":False,"detectors_modified":False}
    _write(args.output/"safe_viewpoint_summary.csv",[summary]); _write(args.output/"safe_viewpoint_verdict.csv",[{"verdict":verdict,"safe_movement_maintained":safe,"visibility_improved":best>first,"m0_negative_control_pass":m0_ok,"deterministic_replay_pass":deterministic}]); _write(args.output/"safe_vs_unsafe_visibility.csv",comparison)
    for record in m1.pointcloud_history:
        snapshot=_scan_snapshot(record); outgoing,_=_gt_directions_eval_only(m1,snapshot); plot_snapshot(args.output/f"scan_{record['scan_index']:02d}_pointcloud.png",snapshot,record["openings"],[item["local_angle_deg"] for item in outgoing])
    _plot(args.output/"safe_viewpoint_audit.png",scans,physics)
    print(json.dumps({"verdict":verdict,"visibility":[(row["scan_index"],row["cumulative_width_ratio"],row["opening_count"],row["matched_outgoing_eval_only"]) for row in scans],"safe":safe,"m0":m0_ok,"deterministic":deterministic,"output":str(args.output.resolve())},indent=2))


if __name__=="__main__":
    main()
