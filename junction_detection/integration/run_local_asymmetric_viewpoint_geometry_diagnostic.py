"""Probe generic local ghost viewpoints without moving the swarm.

Local features and candidate directions use only the fixed Anchor scan,
frozen corridor frame, and estimated corridor width. Geometry, branch labels,
and global positions are confined to virtual ray casting and post-hoc scoring.
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

os.environ.setdefault("MPLCONFIGDIR","/tmp/pdfs_mpl_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from junction_detection.integration.run_lidar_local_corridor_estimation import _rear_start
from junction_detection.integration.run_provisional_anchor_pointcloud_observation import (
    _gt_directions_eval_only,_normalize,evaluate_snapshot,
)
from junction_detection.integration.run_active_viewpoint_acquisition import _gt_mouth_interval_eval_only
from junction_detection.integration.pointcloud_temporal_opening_persistence import circular_interval_iou
from junction_detection.pointcloud.lidar_profile_junction_detector import GeometryProfileConfig,LidarProfileJunctionDetector
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import detect_openings
from pygame_simulator.pre_exploration_general_pipeline_simulator import (
    ActiveViewpointConfig,LIDAR_MAX_RANGE,SimulationRunner,
)

DEFAULT_OUTPUT=ROOT/"junction_detection/integration/output/local_asymmetric_viewpoint_geometry"
SAFE_FORWARD=ROOT/"junction_detection/integration/output/safe_active_viewpoint_visibility/safe_viewpoint_scans.csv"
UNSAFE_FORWARD=ROOT/"junction_detection/integration/output/active_viewpoint_acquisition/active_viewpoint_scans.csv"
CANDIDATE_COMPONENTS={
    "A0":(0.0,0.0),"F":(1.0,0.0),"L":(0.0,1.0),"R":(0.0,-1.0),
    "FL":(1.0/math.sqrt(2.0),1.0/math.sqrt(2.0)),
    "FR":(1.0/math.sqrt(2.0),-1.0/math.sqrt(2.0)),
}


def _write(path: Path, rows: list[dict[str,Any]]) -> None:
    """Write heterogeneous dictionaries as one CSV table."""
    if not rows:
        return
    fields=list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _acquire_m1_anchor(max_frames: int=120) -> tuple[SimulationRunner,dict[str,Any]]:
    """Run only until the already-validated initial fixed Anchor scan exists."""
    profile=LidarProfileJunctionDetector(GeometryProfileConfig(LIDAR_MAX_RANGE))
    runner=SimulationRunner("M1_CROSS_BASELINE","local_forward",profile_detector=profile,hold_on_profile_detection=True,
                            pointcloud_detector=detect_openings,active_viewpoint_config=ActiveViewpointConfig(0.10,0,True,True))
    _rear_start(runner)
    for frame in range(max_frames):
        runner.step(frame)
        if runner.pointcloud_history:
            return runner,runner.pointcloud_history[0]
    raise RuntimeError("M1 initial provisional Anchor was not reached")


def _acquire_m0_snapshot(frame_count: int) -> tuple[SimulationRunner,dict[str,Any]]:
    """Run straight-corridor M0 to the same bounded observation time."""
    profile=LidarProfileJunctionDetector(GeometryProfileConfig(LIDAR_MAX_RANGE))
    runner=SimulationRunner("M0_STRAIGHT","local_forward",profile_detector=profile,hold_on_profile_detection=True)
    for frame in range(frame_count):
        runner.step(frame)
    observation,_=runner.last_visual
    leader=next(robot for robot in runner.world.robots if robot.robot_id==runner.world.lidar_robot_id)
    scan=observation.lidar_scan; margin=np.finfo(float).eps*max(1.0,scan.max_range)*64.0
    return runner,{"scan_index":0,"frame":frame_count-1,"timestamp":runner.world.time,"angles_deg":scan.angles_deg.copy(),"ranges":scan.ranges.copy(),
                   "hit":scan.ranges<scan.max_range-margin,"max_range":scan.max_range,"position_eval":leader.position.copy(),
                   "yaw_eval":math.degrees(leader.body_yaw_rad),"estimated_corridor_width":float(profile.stable_corridor_width),
                   "corridor_forward":np.array([math.cos(leader.body_yaw_rad),math.sin(leader.body_yaw_rad)])}


def _longest_span(mask: np.ndarray, angular_step: float) -> float:
    """Return the longest contiguous true run in a non-wrapping sector."""
    longest=current=0
    for value in mask:
        current=current+1 if value else 0; longest=max(longest,current)
    return float(longest*angular_step)


def _wall_fit(angles: np.ndarray, ranges: np.ndarray, mask: np.ndarray) -> dict[str,Any]:
    """Fit one local lateral wall without geometry or branch metadata."""
    selected=np.flatnonzero(mask)
    if len(selected)<2:
        return {"available":False,"points":len(selected),"tangent_deg":math.nan,"residual":math.nan,"angular_support_deg":0.0}
    theta=np.deg2rad(angles[selected]); points=np.column_stack((ranges[selected]*np.cos(theta),ranges[selected]*np.sin(theta)))
    centroid=np.mean(points,axis=0); centered=points-centroid; values,vectors=np.linalg.eigh(centered.T@centered)
    tangent=vectors[:,int(np.argmax(values))]; normal=np.array([-tangent[1],tangent[0]])
    residual=float(np.sqrt(np.mean((centered@normal)**2)))
    tangent_deg=float((math.degrees(math.atan2(float(tangent[1]),float(tangent[0])))+90.0)%180.0-90.0)
    return {"available":True,"points":len(selected),"tangent_deg":tangent_deg,"residual":residual,"angular_support_deg":float(np.ptp(angles[selected]))}


def local_visibility_features(record: dict[str,Any], case: str) -> dict[str,Any]:
    """Extract left/right raw evidence solely from one local range profile."""
    angles=np.asarray(record["angles_deg"],dtype=float); ranges=np.asarray(record["ranges"],dtype=float); maximum=float(record["max_range"])
    step=float(np.median(np.diff(angles))); margin=np.finfo(float).eps*max(1.0,maximum)*128.0
    no_return=ranges>=maximum-margin; free=ranges>=0.90*maximum; hit=~no_return
    left_sector=(angles>=45.0)&(angles<=135.0); right_sector=(angles<=-45.0)&(angles>=-135.0)
    result={"case":case}
    side_data={}
    for name,sector in (("left",left_sector),("right",right_sector)):
        indices=np.flatnonzero(sector); local_no_return=no_return[indices]; local_free=free[indices]
        discontinuity=np.abs(np.diff(ranges[indices])); terminations=(hit[indices][1:]!=hit[indices][:-1])
        fit=_wall_fit(angles,ranges,sector&hit)
        side_data[name]={"no_return_fraction":float(np.mean(local_no_return)),"longest_no_return_span_deg":_longest_span(local_no_return,step),
                         "free_span_deg":_longest_span(local_free,step),"wall_support_count":int(np.count_nonzero(sector&hit)),
                         "range_discontinuity_strength":float(np.max(discontinuity)) if len(discontinuity) else 0.0,
                         "tangent_available":fit["available"],"tangent_deg":fit["tangent_deg"],"wall_fit_residual":fit["residual"],
                         "usable_angular_range_deg":fit["angular_support_deg"],"wall_termination_evidence":int(np.count_nonzero(terminations)),
                         "mean_range":float(np.mean(ranges[indices]))}
        result.update({f"{name}_{key}":value for key,value in side_data[name].items()})
    mirror_angles=np.arange(45.0,136.0,step); lookup={round(float(angle),8):float(value) for angle,value in zip(angles,ranges)}
    left=np.array([lookup[round(float(angle),8)] for angle in mirror_angles]); right=np.array([lookup[round(float(-angle),8)] for angle in mirror_angles])
    differences=left-right; numerical=max(1.0e-8,np.finfo(float).eps*maximum*128.0)
    result["left_right_asymmetry"]=float(np.mean(np.abs(differences))/maximum)
    result["signed_mean_range_asymmetry"] = float(np.mean(differences)/maximum)
    result["geometry_symmetric"]=bool(np.max(np.abs(differences))<=numerical)
    result["local_preferred_side_by_mean_range"]=("AMBIGUOUS" if abs(float(np.mean(differences)))<=numerical else "LEFT" if np.mean(differences)>0.0 else "RIGHT")
    return result


def _candidate_records(runner: SimulationRunner, anchor: dict[str,Any], case: str) -> list[dict[str,Any]]:
    """Generate generic local ghost scans and run the frozen detector."""
    width=float(anchor["estimated_corridor_width"]); step=0.10*width
    forward=np.asarray(anchor.get("corridor_forward",runner.world.trusted_corridor_forward),dtype=float); forward=forward/np.linalg.norm(forward)
    left=np.array([-forward[1],forward[0]]); origin=np.asarray(anchor["position_eval"],dtype=float); yaw=float(anchor["yaw_eval"])
    results=[]
    for candidate,(forward_scale,lateral_scale) in CANDIDATE_COMPONENTS.items():
        forward_offset=step*forward_scale; lateral_offset=step*lateral_scale; position=origin+forward*forward_offset+left*lateral_offset
        scan=runner.world.sensor.scan(runner.geometry,position,yaw); margin=np.finfo(float).eps*max(1.0,scan.max_range)*64.0
        openings=list(detect_openings(scan.angles_deg.copy(),scan.ranges.copy()))
        snapshot={"context":candidate,"angles":scan.angles_deg.copy(),"ranges":scan.ranges.copy(),"hit":scan.ranges<scan.max_range-margin,
                  "max_range":scan.max_range,"position_eval":position.copy(),"yaw_eval":yaw,"frame":anchor["frame"],"time":anchor["timestamp"]}
        summary,opening_rows=evaluate_snapshot(runner,snapshot,openings)
        outgoing,_=_gt_directions_eval_only(runner,snapshot); matched={row["matched_GT_branch_eval_only"] for row in opening_rows if isinstance(row["matched_GT_branch_eval_only"],int)}
        lateral_false=sum(
            1 for row in opening_rows
            if row["matched_GT_branch_eval_only"]=="" and abs(float(row["center_angle_deg"]))>=45.0
        )
        results.append({"case":case,"candidate":candidate,"local_forward_offset":forward_offset,"local_lateral_offset":lateral_offset,
                        "step_width_ratio":0.0 if candidate=="A0" else 0.10,"candidate_valid":runner.geometry.walkable(position),
                        "candidate_point_inside_free_space":runner.geometry.contains(position),"position_eval":position,"snapshot":snapshot,"openings":openings,"opening_rows":opening_rows,
                        "opening_count":summary["opening_count"],"outgoing_match_count":summary["matched_outgoing_count_eval_only"],"outgoing_total":summary["GT_outgoing_branch_count_eval_only"],
                        "plus90_visible_eval":next((branch["branch_id"] in matched for branch in outgoing if runner.geometry.branches[branch["branch_id"]].angle_deg==90.0),False),
                        "minus90_visible_eval":next((branch["branch_id"] in matched for branch in outgoing if runner.geometry.branches[branch["branch_id"]].angle_deg==-90.0),False),
                        "incoming_match":summary["incoming_opening_count_eval_only"],"false_opening_count":summary["false_opening_count_eval_only"],
                        "lateral_false_opening_count":lateral_false,
                        "valid_lidar_hits":summary["valid_lidar_point_count"],"max_range_count":summary["max_range_no_return_count"],
                        "total_wall_fitted_points":summary["total_fitted_wall_point_count"],"mean_center_error_eval":summary["mean_center_error_deg_eval_only"]})
    anchor_match=next(row["outgoing_match_count"] for row in results if row["candidate"]=="A0")
    anchor_openings=next(row["opening_count"] for row in results if row["candidate"]=="A0")
    for row in results:
        row["visibility_gain_vs_anchor"]=row["outgoing_match_count"]-anchor_match
        row["opening_count_gain_vs_anchor"]=row["opening_count"]-anchor_openings
        row["new_side_branch_count"]=int(row["plus90_visible_eval"])+int(row["minus90_visible_eval"])
    return results


def _candidate_csv(rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    excluded={"position_eval","snapshot","openings","opening_rows"}
    return [{key:value for key,value in row.items() if key not in excluded} for row in rows]


def _opening_csv(runner: SimulationRunner, candidates: list[dict[str,Any]]) -> list[dict[str,Any]]:
    rows=[]
    for candidate in candidates:
        forward=np.asarray(candidate["snapshot"].get("corridor_forward",runner.world.trusted_corridor_forward),dtype=float)
        corridor_world=math.degrees(math.atan2(float(forward[1]),float(forward[0])))
        corridor_body=_normalize(corridor_world-float(candidate["snapshot"]["yaw_eval"]))
        for row in candidate["opening_rows"]:
            match=row["matched_GT_branch_eval_only"]
            if isinstance(match,int):
                branch_type="OUTGOING"; gt=runner.geometry.branches[match].angle_deg
                gt_interval=_gt_mouth_interval_eval_only(runner,candidate["snapshot"],match)
                detected={"start_angle":row["start_angle_deg"],"end_angle":row["end_angle_deg"]}
                iou=circular_interval_iou(detected,gt_interval)
            elif match=="INCOMING": branch_type="INCOMING"; gt=-180.0
            else: branch_type="FALSE"; gt=""
            if not isinstance(match,int): iou=""
            rows.append({"case":candidate["case"],"candidate":candidate["candidate"],"opening_id":row["opening_id"],
                         "center_angle_body":row["center_angle_deg"],"center_angle_corridor":_normalize(float(row["center_angle_deg"])-corridor_body),"angular_width":row["angular_width_deg"],
                         "confidence":row["confidence"],"GT_match_eval_only":gt,"GT_branch_type_eval_only":branch_type,
                         "center_error_eval_only":row["center_error_deg_eval_only"],"IoU_eval_only":iou,
                         "wall_support":row["usable_wall_sides"],"wall_fitted_points":row["fitted_wall_point_count"],
                         "line_residual":row["line_fit_residual"],"tangent_support":row["wall_tangent_deg"]!="","tangent_deg":row["wall_tangent_deg"]})
    return rows


def _forward_history() -> list[dict[str,Any]]:
    """Read prior evaluation CSVs only for contextual comparison."""
    result=[]
    for label,path in (("SAFE_FORWARD",SAFE_FORWARD),("UNSAFE_FORWARD",UNSAFE_FORWARD)):
        if not path.exists(): continue
        with path.open(newline="",encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                result.append({"source":label,"scan_index":row["scan_index"],"cumulative_width_ratio":row.get("cumulative_width_ratio",row.get("cumulative_advance_width_ratio","")),
                               "outgoing_match":row.get("matched_outgoing_eval_only",row.get("matched_outgoing_count_eval_only","")),"opening_count":row["opening_count"]})
    return result


def _deterministic_probe(runner,anchor,case,reference) -> bool:
    repeated=_candidate_records(runner,anchor,case)
    for first,second in zip(reference,repeated):
        if first["candidate"]!=second["candidate"] or first["openings"]!=second["openings"] or first["outgoing_match_count"]!=second["outgoing_match_count"]:
            return False
        if not np.array_equal(first["snapshot"]["ranges"],second["snapshot"]["ranges"]): return False
    return True


def _plot(path: Path, runner,anchor,candidates,features) -> None:
    """Show candidate layout, visibility scores, profiles, and local evidence."""
    by_name={row["candidate"]:row for row in candidates}; non_forward=[row for row in candidates if row["candidate"] in ("L","R","FL","FR")]
    best=max(non_forward,key=lambda row:(row["outgoing_match_count"],row["new_side_branch_count"],row["total_wall_fitted_points"],-list(CANDIDATE_COMPONENTS).index(row["candidate"])))
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for wall in runner.geometry.walls: axes[0,0].plot([wall[0][0],wall[1][0]],[wall[0][1],wall[1][1]],color="black",linewidth=1)
    label_offsets={"A0":(-22,-15),"F":(8,-15),"L":(-25,8),"R":(8,-15),"FL":(-32,10),"FR":(8,8)}
    anchor_point=np.asarray(anchor["position_eval"],dtype=float)
    for row in candidates:
        point=row["position_eval"]; color="tab:green" if row["visibility_gain_vs_anchor"]>0 else "tab:blue"
        axes[0,0].plot([anchor_point[0],point[0]],[anchor_point[1],point[1]],color=color,alpha=.35)
        axes[0,0].scatter(point[0],point[1],color=color)
        axes[0,0].annotate(f"{row['candidate']} {row['outgoing_match_count']}/{row['outgoing_total']}",point,xytext=label_offsets[row["candidate"]],textcoords="offset points")
    width=float(anchor["estimated_corridor_width"])
    axes[0,0].set(xlim=(anchor_point[0]-.65*width,anchor_point[0]+.65*width),ylim=(anchor_point[1]-.65*width,anchor_point[1]+.65*width),title="Virtual candidates near A0 (EVAL ONLY)",aspect="equal")
    names=[row["candidate"] for row in candidates]; axes[0,1].bar(names,[row["outgoing_match_count"] for row in candidates],label="outgoing match"); axes[0,1].plot(names,[row["opening_count"] for row in candidates],"o--",color="tab:orange",label="openings"); axes[0,1].set(title="Candidate visibility",ylim=(0,3.2)); axes[0,1].legend(); axes[0,1].grid(axis="y",alpha=.25)
    for name in ("A0","F",best["candidate"]):
        row=by_name[name]; axes[1,0].plot(row["snapshot"]["angles"],row["snapshot"]["ranges"],label=name)
    axes[1,0].set(title="A0 / F / best non-forward profiles",xlabel="body-local angle [deg]",ylabel="range"); axes[1,0].legend(); axes[1,0].grid(alpha=.25)
    labels=("no_return_fraction","free_span_deg","wall_support_count","range_discontinuity_strength","wall_termination_evidence")
    x=np.arange(len(labels)); left=[features[f"left_{key}"] for key in labels]; right=[features[f"right_{key}"] for key in labels]
    axes[1,1].bar(x-.18,left,.36,label="left"); axes[1,1].bar(x+.18,right,.36,label="right"); axes[1,1].set(xticks=x,xticklabels=labels,title="A0 local raw evidence"); axes[1,1].tick_params(axis="x",rotation=25); axes[1,1].legend(); axes[1,1].grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)


def main(argv=None) -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); parser.add_argument("--max-anchor-frames",type=int,default=120); args=parser.parse_args(argv); args.output.mkdir(parents=True,exist_ok=True)
    m1,anchor=_acquire_m1_anchor(args.max_anchor_frames); anchor["corridor_forward"]=m1.world.trusted_corridor_forward.copy()
    features_m1=local_visibility_features(anchor,"M1_CROSS_BASELINE")
    candidates_m1=_candidate_records(m1,anchor,"M1_CROSS_BASELINE")
    m0,m0_anchor=_acquire_m0_snapshot(int(anchor["frame"])+1); features_m0=local_visibility_features(m0_anchor,"M0_STRAIGHT"); candidates_m0=_candidate_records(m0,m0_anchor,"M0_STRAIGHT")
    deterministic=_deterministic_probe(m1,anchor,"M1_CROSS_BASELINE",candidates_m1) and _deterministic_probe(m0,m0_anchor,"M0_STRAIGHT",candidates_m0)
    anchor_row=next(row for row in candidates_m1 if row["candidate"]=="A0"); non_forward=[row for row in candidates_m1 if row["candidate"] in ("L","R","FL","FR")]
    best=max(non_forward,key=lambda row:(row["outgoing_match_count"],row["new_side_branch_count"],row["total_wall_fitted_points"],-list(CANDIDATE_COMPONENTS).index(row["candidate"])))
    gain=[row for row in non_forward if row["outgoing_match_count"]>anchor_row["outgoing_match_count"]]
    false_regression=any(row["candidate"]!="A0" and row["false_opening_count"]>anchor_row["false_opening_count"] for row in candidates_m1)
    m0_anchor_false=next(row["false_opening_count"] for row in candidates_m0 if row["candidate"]=="A0")
    m0_candidate_regression=any(row["false_opening_count"]>m0_anchor_false for row in candidates_m0)
    m0_lateral_false_regression=any(row["lateral_false_opening_count"]>next(item["lateral_false_opening_count"] for item in candidates_m0 if item["candidate"]=="A0") for row in candidates_m0)
    if false_regression: verdict="FALSE_OPENING_REGRESSION"
    elif gain and features_m1["geometry_symmetric"]: verdict="NON_FORWARD_GAIN_EXISTS_BUT_LOCAL_DIRECTION_AMBIGUOUS"
    elif gain: verdict="NON_FORWARD_VIEWPOINT_VISIBILITY_GAIN_VALID"
    else: verdict="NO_BOUNDED_NON_FORWARD_VIEWPOINT_GAIN"
    summary={"verdict":verdict,"anchor_outgoing_match":anchor_row["outgoing_match_count"],"best_candidate":best["candidate"] if gain else "NONE_NO_GAIN","representative_non_forward_candidate":best["candidate"],"best_outgoing_match":best["outgoing_match_count"],
             "best_visibility_gain":best["visibility_gain_vs_anchor"],"plus90_first_candidate":next((row["candidate"] for row in candidates_m1 if row["plus90_visible_eval"]),""),
             "minus90_first_candidate":next((row["candidate"] for row in candidates_m1 if row["minus90_visible_eval"]),""),"local_geometry_symmetric":features_m1["geometry_symmetric"],
             "local_preferred_side":features_m1["local_preferred_side_by_mean_range"],"local_evidence_matches_best_direction":"NOT_TESTABLE_NO_GAIN" if not gain else best["candidate"],
             "m0_anchor_false_openings":m0_anchor_false,"m0_candidate_false_regression":m0_candidate_regression,"m0_lateral_false_regression":m0_lateral_false_regression,"deterministic_replay":deterministic,
             "GT_used_in_local_features":False,"map_used_in_local_direction_definition":False,"actual_swarm_movement_performed":False,"detectors_modified":False,"movement_physics_modified":False}
    _write(args.output/"viewpoint_candidates.csv",_candidate_csv(candidates_m1)+_candidate_csv(candidates_m0)); _write(args.output/"local_visibility_features.csv",[features_m1,features_m0])
    _write(args.output/"viewpoint_openings.csv",_opening_csv(m1,candidates_m1)+_opening_csv(m0,candidates_m0)); _write(args.output/"viewpoint_summary.csv",[summary]); _write(args.output/"viewpoint_verdict.csv",[{"verdict":verdict,"deterministic_replay":deterministic,"m0_candidate_false_regression":m0_candidate_regression,"m0_lateral_false_regression":m0_lateral_false_regression,"GT_map_leakage":False}]); _write(args.output/"forward_history_comparison.csv",_forward_history()+[{"source":"VIRTUAL_0.10W","candidate":row["candidate"],"cumulative_width_ratio":row["step_width_ratio"],"outgoing_match":row["outgoing_match_count"],"opening_count":row["opening_count"]} for row in candidates_m1])
    _plot(args.output/"viewpoint_geometry_audit.png",m1,anchor,candidates_m1,features_m1)
    print(json.dumps({"verdict":verdict,"features":features_m1,"candidates":[(row["candidate"],row["candidate_valid"],row["opening_count"],row["outgoing_match_count"],row["new_side_branch_count"]) for row in candidates_m1],"m0_false_regression":m0_candidate_regression,"deterministic":deterministic,"output":str(args.output.resolve())},indent=2))


if __name__=="__main__": main()
