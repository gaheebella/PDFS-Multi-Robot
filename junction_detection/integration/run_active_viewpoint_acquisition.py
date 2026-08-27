"""Evaluate local forward viewpoint acquisition after a provisional Anchor.

Ground truth and global poses are consumed only after each fixed Point Cloud
scan. They never select an advance direction, distance, stopping time, or the
number of rescans.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from junction_detection.integration.pointcloud_temporal_opening_persistence import circular_interval_iou
from junction_detection.integration.run_lidar_local_corridor_estimation import _rear_start
from junction_detection.integration.run_provisional_anchor_pointcloud_observation import (
    _gt_directions_eval_only,
    _normalize,
    evaluate_snapshot,
    plot_snapshot,
)
from junction_detection.integration.pointcloud_wall_parallel_orientation import estimate_wall_parallel_tangent
from junction_detection.pointcloud.lidar_profile_junction_detector import GeometryProfileConfig,LidarProfileJunctionDetector
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import detect_openings
from pygame_simulator.pre_exploration_general_pipeline_simulator import (
    ActiveViewpointConfig,
    LIDAR_MAX_RANGE,
    SimulationRunner,
)

DEFAULT_OUTPUT=ROOT/"junction_detection/integration/output/active_viewpoint_acquisition"


def _write(path: Path, rows: list[dict[str,Any]]) -> None:
    """Write rows while preserving fields added by later records."""
    if not rows:
        return
    fields=list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def run_case(case_id: str, frames: int, config: ActiveViewpointConfig, rear_start: bool=False) -> SimulationRunner:
    """Run one deterministic physical case with opt-in active rescans."""
    profile=LidarProfileJunctionDetector(GeometryProfileConfig(LIDAR_MAX_RANGE))
    runner=SimulationRunner(
        case_id,"local_forward",profile_detector=profile,
        hold_on_profile_detection=True,pointcloud_detector=detect_openings,
        active_viewpoint_config=config,
    )
    if rear_start:
        _rear_start(runner)
    for frame in range(frames):
        runner.step(frame)
    return runner


def _scan_snapshot(record: dict[str,Any]) -> dict[str,Any]:
    """Adapt a runtime scan record to the existing post-hoc evaluator."""
    return {
        "context":f"SCAN_{record['scan_index']}","angles":record["angles_deg"],
        "ranges":record["ranges"],"hit":record["hit"],"max_range":record["max_range"],
        "position_eval":record["position_eval"],"yaw_eval":record["yaw_eval"],
        "frame":record["frame"],"time":record["timestamp"],
    }


def _gt_mouth_interval_eval_only(runner: SimulationRunner, snapshot: dict[str,Any], branch_id: int) -> dict[str,float]:
    """Return the shorter angular interval subtended by a GT branch mouth."""
    branch=runner.geometry.branches[branch_id]
    radians=math.radians(branch.angle_deg)
    direction=np.array([math.sin(radians),math.cos(radians)])
    lateral=np.array([-direction[1],direction[0]])
    center=direction*(runner.geometry.junction_size/2.0-2.0)
    endpoints=(center-lateral*branch.width/2.0,center+lateral*branch.width/2.0)
    bearings=[]
    for point in endpoints:
        relative=point-np.asarray(snapshot["position_eval"])
        bearings.append(_normalize(math.degrees(math.atan2(float(relative[1]),float(relative[0])))-float(snapshot["yaw_eval"])))
    first,second=bearings
    if (second-first)%360.0 <= 180.0:
        start,end=first,second
    else:
        start,end=second,first
    return {"start_angle":start,"end_angle":end}


def _evaluate_scans(runner: SimulationRunner) -> tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    """Evaluate visibility, wall support, and mouth IoU after runtime ends."""
    scan_rows=[]; opening_rows=[]; visibility_rows=[]
    for record in runner.pointcloud_history:
        snapshot=_scan_snapshot(record)
        summary,rows=evaluate_snapshot(runner,snapshot,record["openings"])
        outgoing,_=_gt_directions_eval_only(runner,snapshot)
        matched_ids=set()
        for row in rows:
            match=row["matched_GT_branch_eval_only"]
            if isinstance(match,int):
                matched_ids.add(match)
                gt_interval=_gt_mouth_interval_eval_only(runner,snapshot,match)
                detected={"start_angle":row["start_angle_deg"],"end_angle":row["end_angle_deg"]}
                row["GT_mouth_interval_iou_eval_only"]=circular_interval_iou(detected,gt_interval)
            else:
                row["GT_mouth_interval_iou_eval_only"]=""
            row.update({"scan_index":record["scan_index"],"cumulative_local_advance":record["cumulative_local_advance"]})
            opening_rows.append(row)
        matched_ious=[float(row["GT_mouth_interval_iou_eval_only"]) for row in rows if row["GT_mouth_interval_iou_eval_only"]!=""]
        scan_rows.append({
            "map_case":runner.geometry.case_id,"scan_index":record["scan_index"],"frame":record["frame"],"timestamp":record["timestamp"],
            "cumulative_local_advance":record["cumulative_local_advance"],"cumulative_advance_width_ratio":record["cumulative_local_advance"]/record["estimated_corridor_width"],
            "actual_step_advance":record["cumulative_local_advance"]-(runner.pointcloud_history[record["scan_index"]-1]["cumulative_local_advance"] if record["scan_index"] else 0.0),
            "estimated_corridor_width":record["estimated_corridor_width"],"leader_speed_at_scan":record["leader_speed_at_scan"],
            **{key:value for key,value in summary.items() if key not in ("scan_context","frame","time")},
            "mean_GT_mouth_interval_iou_eval_only":float(np.mean(matched_ious)) if matched_ious else math.nan,
            "anchor_x_eval_only":float(record["position_eval"][0]),"anchor_y_eval_only":float(record["position_eval"][1]),"anchor_yaw_eval_only":record["yaw_eval"],
            "leader_support_connected":record["connectivity"]["leader_connected"],"leader_communication_connected":record["connectivity"]["leader_communication_connected"],
            "leader_component_size":record["connectivity"]["leader_connected_component_size"],"leader_front_pack_gap":record["connectivity"]["leader_to_front_pack_gap"],
        })
        for branch in outgoing:
            visibility_rows.append({
                "scan_index":record["scan_index"],"cumulative_local_advance":record["cumulative_local_advance"],
                "GT_branch_id_eval_only":branch["branch_id"],"GT_branch_angle_deg_eval_only":runner.geometry.branches[branch["branch_id"]].angle_deg,
                "GT_local_angle_deg_eval_only":branch["local_angle_deg"],"visible_as_opening_eval_only":branch["branch_id"] in matched_ids,
            })
    return scan_rows,opening_rows,visibility_rows


def _timeline(runner: SimulationRunner) -> list[dict[str,Any]]:
    """Select lifecycle fields including local progress and physical sanity."""
    keys=(
        "map_case","frame","timestamp","profile_junction_detected","junction_detection_latched","active_junction_phase",
        "active_viewpoint_state","viewpoint_scan_index","viewpoint_step_fraction","viewpoint_step_target","viewpoint_step_progress",
        "viewpoint_cumulative_advance","trusted_corridor_width","trusted_corridor_orientation_deg","trusted_corridor_sign_source",
        "local_forward_propulsion_active","leader_braking_active","leader_speed","follower_mean_speed","stationary_dwell_steps",
        "provisional_fixed_anchor","pointcloud_ready","pointcloud_called","pointcloud_call_count","pointcloud_opening_count",
        "min_inter_robot_distance","overlap_pair_count","wall_contact_count","outside_free_space_robot_count","nan_inf_state_count","max_speed",
        "leader_connected","leader_communication_connected","leader_connected_component_size","leader_to_front_pack_gap","normalized_leader_gap",
        "leader_displacement_since_detection_eval","anchor_x_eval","anchor_y_eval","gt_phase",
    )
    return [{key:row.get(key,"") for key in keys} for row in runner.rows]


def _physical_rows(runner: SimulationRunner) -> list[dict[str,Any]]:
    fields=("frame","timestamp","active_viewpoint_state","viewpoint_scan_index","min_inter_robot_distance","overlap_pair_count","wall_contact_count","outside_free_space_robot_count","nan_inf_state_count","max_speed","follower_mean_speed","leader_connected","leader_communication_connected","leader_connected_component_size","leader_to_front_pack_gap")
    return [{key:row.get(key,"") for key in fields} for row in runner.rows]


def _plot_audit(path: Path, scan_rows: list[dict[str,Any]], visibility_rows: list[dict[str,Any]]) -> None:
    """Plot the visibility curve and wall-fit support in one compact figure."""
    x=[row["cumulative_advance_width_ratio"] for row in scan_rows]
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    axes[0].plot(x,[row["matched_outgoing_count_eval_only"] for row in scan_rows],"o-",label="matched outgoing")
    axes[0].plot(x,[row["opening_count"] for row in scan_rows],"s--",label="all openings")
    for branch_id in sorted({row["GT_branch_id_eval_only"] for row in visibility_rows}):
        subset=[row for row in visibility_rows if row["GT_branch_id_eval_only"]==branch_id]
        label=f"GT {subset[0]['GT_branch_angle_deg_eval_only']:+.0f} deg visible"
        axes[0].step(x,[int(row["visible_as_opening_eval_only"]) for row in subset],where="mid",alpha=.55,label=label)
    axes[0].set(xlabel="cumulative local advance / estimated width",ylabel="count / visibility",title="Outgoing visibility curve (GT post-hoc)"); axes[0].legend(fontsize=8); axes[0].grid(alpha=.25)
    axes[1].plot(x,[row["total_fitted_wall_point_count"] for row in scan_rows],"o-",label="fitted wall points")
    axes[1].plot(x,[row["wall_tangent_available_count"] for row in scan_rows],"s--",label="openings with tangent")
    axes[1].set(xlabel="cumulative local advance / estimated width",title="Wall/tangent support"); axes[1].legend(); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)


def _deterministic_equal(first: SimulationRunner, second: SimulationRunner) -> bool:
    """Compare the complete local active scan sequence exactly."""
    if len(first.pointcloud_history)!=len(second.pointcloud_history):
        return False
    for a,b in zip(first.pointcloud_history,second.pointcloud_history):
        if a["scan_index"]!=b["scan_index"] or a["cumulative_local_advance"]!=b["cumulative_local_advance"]:
            return False
        if not np.array_equal(a["ranges"],b["ranges"]):
            return False
        if a["openings"]!=b["openings"]:
            return False
    return True


def main(argv=None) -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames",type=int,default=600)
    parser.add_argument("--replay-frames",type=int,default=300)
    parser.add_argument("--viewpoint-step-fraction",type=float,default=0.10)
    parser.add_argument("--viewpoint-max-rescans",type=int,default=3)
    parser.add_argument("--reuse-existing-m0",action="store_true",help="reuse the unchanged existing 600-frame M0 CSV")
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args(argv); args.output.mkdir(parents=True,exist_ok=True)
    config=ActiveViewpointConfig(args.viewpoint_step_fraction,args.viewpoint_max_rescans)
    m0=run_case("M0_STRAIGHT",0 if args.reuse_existing_m0 else args.frames,config)
    m1=run_case("M1_CROSS_BASELINE",args.frames,config,rear_start=True)
    if not m1.pointcloud_history or m1.pointcloud_history[0]["scan_index"]!=0:
        raise RuntimeError("initial M1 provisional Anchor scan was not reproduced")
    scan_rows,opening_rows,visibility_rows=_evaluate_scans(m1)
    if not args.reuse_existing_m0:
        _write(args.output/"active_viewpoint_timeline_m0.csv",_timeline(m0))
    _write(args.output/"active_viewpoint_timeline_m1.csv",_timeline(m1))
    _write(args.output/"active_viewpoint_scans.csv",scan_rows)
    _write(args.output/"active_viewpoint_openings.csv",opening_rows)
    _write(args.output/"active_viewpoint_visibility.csv",visibility_rows)
    _write(args.output/"active_viewpoint_physical_sanity.csv",_physical_rows(m1))
    for record in m1.pointcloud_history:
        snapshot=_scan_snapshot(record); outgoing,_=_gt_directions_eval_only(m1,snapshot)
        plot_snapshot(args.output/f"scan_{record['scan_index']:02d}_pointcloud.png",snapshot,record["openings"],[item["local_angle_deg"] for item in outgoing])
    _plot_audit(args.output/"active_viewpoint_audit.png",scan_rows,visibility_rows)

    first_matched=scan_rows[0]["matched_outgoing_count_eval_only"]
    best_matched=max(row["matched_outgoing_count_eval_only"] for row in scan_rows)
    side_first={}
    for angle in (-90.0,90.0):
        visible=[row for row in visibility_rows if row["GT_branch_angle_deg_eval_only"]==angle and row["visible_as_opening_eval_only"]]
        side_first[angle]=visible[0] if visible else None
    active_rows=[row for row in m1.rows if row.get("active_viewpoint_state") in ("VIEWPOINT_ADVANCE","VIEWPOINT_REBRAKING")]
    physics_unsafe=(any(int(row["nan_inf_state_count"]) or int(row["outside_free_space_robot_count"]) for row in active_rows)
                    or any(not bool(row["leader_communication_connected"]) or not bool(row["leader_connected"]) for row in active_rows))
    if physics_unsafe:
        verdict="ACTIVE_VIEWPOINT_ACQUISITION_PHYSICS_UNSAFE"
    elif best_matched==len(m1.geometry.branches) and all(side_first.values()):
        verdict="ACTIVE_VIEWPOINT_ACQUISITION_EFFECTIVE"
    elif best_matched>first_matched:
        verdict="ACTIVE_VIEWPOINT_ACQUISITION_PARTIAL"
    else:
        verdict="ACTIVE_VIEWPOINT_ACQUISITION_INEFFECTIVE"

    replay=run_case("M1_CROSS_BASELINE",args.replay_frames,config,rear_start=True)
    replay_reference=[record for record in m1.pointcloud_history if record["frame"]<args.replay_frames]
    reference_proxy=type("ReplayReference",(),{"pointcloud_history":replay_reference})()
    deterministic=_deterministic_equal(reference_proxy,replay)
    m0_summary={"map_case":"M0_STRAIGHT","profile_detection_count":0,"junction_latched":False,"provisional_anchor":False,"active_advance_sample_count":0,"pointcloud_call_count":0,"reused_existing_600_frame_result":args.reuse_existing_m0}
    m1_summary={
        "map_case":"M1_CROSS_BASELINE_REAR_START","scan_count":len(scan_rows),"step_fraction":config.step_fraction,"max_rescans":config.max_rescans,
        "estimated_corridor_width":m1.world.trusted_corridor_width,"trusted_corridor_orientation_deg":m1.world.trusted_corridor_orientation_deg,"forward_sign_source":m1.world.trusted_corridor_sign_source,
        "initial_matched_outgoing":first_matched,"best_matched_outgoing":best_matched,"initial_opening_count":scan_rows[0]["opening_count"],"final_opening_count":scan_rows[-1]["opening_count"],
        "cumulative_local_advance":scan_rows[-1]["cumulative_local_advance"],"detection_to_first_anchor_displacement_eval_only":next(row["leader_displacement_since_detection_eval"] for row in m1.rows if row["pointcloud_called"]),
        "min_inter_robot_distance":min(row["min_inter_robot_distance"] for row in active_rows),"max_overlap_pair_count":max(row["overlap_pair_count"] for row in active_rows),
        "max_wall_contact_count":max(row["wall_contact_count"] for row in active_rows),"max_outside":max(row["outside_free_space_robot_count"] for row in active_rows),"max_nan_inf":max(row["nan_inf_state_count"] for row in active_rows),
        "leader_max_speed":max(row["leader_speed"] for row in active_rows),"follower_mean_speed_max":max(row["follower_mean_speed"] for row in active_rows),
        "leader_support_connected_all_active":all(row["leader_connected"] for row in active_rows),"leader_communication_connected_all_active":all(row["leader_communication_connected"] for row in active_rows),"minimum_leader_component_size":min(row["leader_connected_component_size"] for row in active_rows),"maximum_front_pack_gap":max(row["leader_to_front_pack_gap"] for row in active_rows),
        "minus90_first_visible_scan_eval_only":"" if side_first[-90.0] is None else side_first[-90.0]["scan_index"],"minus90_first_visible_advance_eval_only":"" if side_first[-90.0] is None else side_first[-90.0]["cumulative_local_advance"],
        "plus90_first_visible_scan_eval_only":"" if side_first[90.0] is None else side_first[90.0]["scan_index"],"plus90_first_visible_advance_eval_only":"" if side_first[90.0] is None else side_first[90.0]["cumulative_local_advance"],
        "deterministic_replay":deterministic,
    }
    _write(args.output/"active_viewpoint_summary.csv",[m0_summary,m1_summary])
    _write(args.output/"active_viewpoint_verdict.csv",[{"verdict":verdict,"GT_used_for_control":False,"global_position_used_for_control":False,"fixed_world_step_used":False,"detector_threshold_modified":False,"final_fixed_anchor_promoted":False,"deterministic_replay":deterministic}])
    print(json.dumps({"verdict":verdict,"scan_count":len(scan_rows),"visibility":[(row["scan_index"],row["cumulative_advance_width_ratio"],row["opening_count"],row["matched_outgoing_count_eval_only"]) for row in scan_rows],"deterministic_replay":deterministic,"output":str(args.output.resolve())},indent=2))


if __name__=="__main__":
    main()
