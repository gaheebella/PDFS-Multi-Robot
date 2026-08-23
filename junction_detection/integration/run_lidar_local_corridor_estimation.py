"""Validate scan-only corridor estimation and rear-start Junction detection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR","/tmp/pdfs_mpl_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from junction_detection.pointcloud.lidar_profile_junction_detector import GeometryProfileConfig,LidarProfileJunctionDetector
from pygame_simulator.pre_exploration_general_pipeline_simulator import (
    GeometryBuilder,GeometryCase,GroundTruthEvaluator,LIDAR_MAX_RANGE,SimulationRunner,_rect,_union_boundary,
)

DEFAULT_OUTPUT=ROOT/"junction_detection/integration/output/lidar_local_corridor_estimation"
GT_CORRIDOR_WIDTH=84.0
REAR_START_SHIFT=160.0


def _write(path: Path, rows: list[dict]) -> None:
    """Write homogeneous audit dictionaries."""
    if not rows: return
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _rear_start(runner: SimulationRunner, shift: float=REAR_START_SHIFT) -> None:
    """Extend only the evaluation corridor and move the unchanged swarm rearward."""
    original=runner.geometry; entrance=float(original.entrance_y); length=original.incoming_length+shift
    incoming=_rect(np.array([0.0,entrance-0.5*length]),np.array([0.0,1.0]),original.incoming_width,length)
    rects=(incoming,)+original.free_rects[1:]
    geometry=GeometryCase(original.case_id,original.incoming_width,length,original.junction_size,original.branches,rects,_union_boundary(rects),original.entrance_y)
    runner.geometry=geometry; runner.world.geometry=geometry; runner.gt=GroundTruthEvaluator(geometry)
    for robot in runner.world.robots: robot.position[1]-=shift
    runner.world.initial_mean_y=float(np.mean([robot.position[1] for robot in runner.world.robots]))
    runner.world.initial_front_y=float(max(robot.position[1] for robot in runner.world.robots))
    leader=next(robot for robot in runner.world.robots if robot.robot_id==runner.world.lidar_robot_id)
    runner.world.initial_lidar_position=leader.position.copy()


def _snapshot(runner: SimulationRunner, row: dict) -> dict:
    """Copy one scan/profile state for post-hoc plotting."""
    scan=runner.last_visual[0].lidar_scan; result=runner.last_profile_result
    return {"angles":scan.angles_deg.copy(),"ranges":scan.ranges.copy(),"expected":result["expected_ranges"].copy(),"candidate":result["open_candidate_mask"].copy(),"confirmed":result["confirmed_opening_mask"].copy(),"frame":row["frame"],"time":row["timestamp"],"width":result["estimated_corridor_width"],"offset":result["estimated_offset"],"frozen":result["corridor_model_frozen"]}


def run_case(case_id: str, frames: int, rear_start: bool=False) -> tuple[SimulationRunner,dict]:
    """Run a scan-only detector; map data remains outside its call boundary."""
    detector=LidarProfileJunctionDetector(GeometryProfileConfig(LIDAR_MAX_RANGE))
    runner=SimulationRunner(case_id,"local_forward",profile_detector=detector)
    if rear_start: _rear_start(runner)
    snapshots={"clear":None,"before_detection":None,"first_detection":None}; previous_clear=None
    for frame in range(frames):
        row=runner.step(frame)
        if row is None: continue
        leader=next(robot for robot in runner.world.robots if robot.robot_id==runner.world.lidar_robot_id)
        row["lidar_x_eval"]=float(leader.position[0]); row["lidar_y_eval"]=float(leader.position[1])
        if runner.geometry.entrance_y is None:
            row["GT_opening_within_sensor_range_eval_only"]=False
        else:
            half=0.5*runner.geometry.incoming_width; y=runner.geometry.entrance_y
            corner_distance=min(math.hypot(leader.position[0]-side,y-leader.position[1]) for side in (-half,half))
            row["GT_opening_within_sensor_range_eval_only"]=corner_distance<=LIDAR_MAX_RANGE+1e-9
        current=_snapshot(runner,row)
        if row["corridor_model_initialized"] and not row["profile_junction_detected"]:
            if snapshots["clear"] is None: snapshots["clear"]=current
            previous_clear=current
        if snapshots["first_detection"] is None and row["profile_junction_detected"]:
            snapshots["before_detection"]=previous_clear; snapshots["first_detection"]=current
    return runner,snapshots


def timeline(rows: list[dict]) -> list[dict]:
    """Select corridor-model, profile, and evaluation-only fields."""
    return [{
        "frame":row["frame"],"time":row["timestamp"],"map_case":row["map_case"],
        "left_wall_range":row["left_wall_range"],"right_wall_range":row["right_wall_range"],"width_observation":row["width_observation"],
        "estimated_corridor_width":row["estimated_corridor_width"],"estimated_offset":row["estimated_offset"],
        "corridor_model_initialized":row["corridor_model_initialized"],"corridor_model_update_enabled":row["corridor_model_update_enabled"],"corridor_model_frozen":row["corridor_model_frozen"],
        "opening_candidate_count":row["opening_candidate_count"],"opening_group_count":row["opening_group_count"],"junction_detected":row["profile_junction_detected"],"opening_groups_json":row["opening_groups_json"],
        "leader_speed":row["leader_speed"],"lidar_x_eval":row["lidar_x_eval"],"lidar_y_eval":row["lidar_y_eval"],
        "GT_corridor_width_eval_only":GT_CORRIDOR_WIDTH,"GT_phase_eval_only":row["gt_phase"],"GT_opening_within_sensor_range_eval_only":row["GT_opening_within_sensor_range_eval_only"],"profile_max_error":row["profile_max_abs_valid_delta"],"nan_inf_state_count":row["nan_inf_state_count"],
    } for row in rows]


def _first(rows: list[dict], key: str) -> dict|None:
    return next((row for row in rows if row[key]),None)


def summarize(case_id: str, rows: list[dict]) -> dict:
    """Summarize width accuracy, state transition, and contamination."""
    initialized=[row for row in rows if row["corridor_model_initialized"]]
    widths=np.asarray([row["estimated_corridor_width"] for row in initialized],dtype=float)
    offsets=np.asarray([row["estimated_offset"] for row in initialized],dtype=float)
    candidate=_first(rows,"opening_candidate_count"); group=_first(rows,"opening_group_count"); detection=_first(rows,"profile_junction_detected"); initialization=_first(rows,"corridor_model_initialized")
    before=next((rows[index-1] for index,row in enumerate(rows) if row["profile_junction_detected"] and index>0),None)
    first_visible_index=next((index for index,row in enumerate(rows) if row["GT_opening_within_sensor_range_eval_only"]),len(rows))
    return {
        "map_case":case_id,"sample_count":len(rows),"model_initialization_time":initialization["timestamp"] if initialization else math.nan,
        "estimated_width_mean":float(np.mean(widths)) if len(widths) else math.nan,"estimated_width_min":float(np.min(widths)) if len(widths) else math.nan,"estimated_width_max":float(np.max(widths)) if len(widths) else math.nan,"estimated_width_error_eval_only":float(np.mean(widths)-GT_CORRIDOR_WIDTH) if len(widths) else math.nan,
        "estimated_offset_mean":float(np.mean(offsets)) if len(offsets) else math.nan,"estimated_offset_min":float(np.min(offsets)) if len(offsets) else math.nan,"estimated_offset_max":float(np.max(offsets)) if len(offsets) else math.nan,
        "first_candidate_time":candidate["timestamp"] if candidate else math.nan,"first_group_time":group["timestamp"] if group else math.nan,"first_detection_time":detection["timestamp"] if detection else math.nan,
        "first_detection_frame":detection["frame"] if detection else -1,"first_detection_groups":detection["opening_groups_json"] if detection else "[]","first_detection_gt_phase_eval_only":detection["gt_phase"] if detection else "NONE",
        "width_before_detection":before["estimated_corridor_width"] if before else math.nan,"width_at_detection":detection["estimated_corridor_width"] if detection else math.nan,"model_frozen_at_detection":detection["corridor_model_frozen"] if detection else False,
        "clear_sample_count_before_detection":sum(not row["profile_junction_detected"] for row in rows[:rows.index(detection)]) if detection else len(rows),
        "model_frozen_sample_count":sum(row["corridor_model_frozen"] for row in rows),"false_corridor_trigger_count":sum(row["profile_junction_detected"] for row in rows[:first_visible_index]),
        "max_profile_error":max(row["profile_max_abs_valid_delta"] for row in rows),"max_nan_inf":max(row["nan_inf_state_count"] for row in rows),
    }


def plot_profile(path: Path, snapshot: dict, title: str) -> None:
    """Plot measured/expected profiles with learned-model annotation."""
    fig,axis=plt.subplots(figsize=(10,4.8)); angles=snapshot["angles"]
    axis.plot(angles,snapshot["ranges"],color="black",linewidth=1.1,label="measured rho(theta)")
    axis.plot(angles,snapshot["expected"],color="tab:blue",linestyle="--",label="expected from local model")
    axis.fill_between(angles,0,snapshot["ranges"],where=snapshot["candidate"],color="orange",alpha=.24,label="OPEN_CANDIDATE")
    axis.fill_between(angles,0,snapshot["ranges"],where=snapshot["confirmed"],color="magenta",alpha=.18,label="opening group")
    annotation=f"width_hat={snapshot['width']:.3f}  offset_hat={snapshot['offset']:+.3f}  frozen={snapshot['frozen']}"
    axis.set(xlabel="body-relative angle [deg]",ylabel="range",xlim=(-180,179),ylim=(0,LIDAR_MAX_RANGE*1.05),title=f"{title} | frame={snapshot['frame']} t={snapshot['time']:.3f}s\n{annotation}")
    axis.grid(alpha=.25); axis.legend(loc="lower center",ncol=4); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)


def main(argv=None) -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--frames",type=int,default=600); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); args=parser.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    m0,m0_shots=run_case("M0_STRAIGHT",args.frames); m1,m1_shots=run_case("M1_CROSS_BASELINE",args.frames,rear_start=True)
    _write(args.output/"local_corridor_timeline_m0.csv",timeline(m0.rows)); _write(args.output/"local_corridor_timeline_m1.csv",timeline(m1.rows))
    summaries=[summarize("M0_STRAIGHT",m0.rows),summarize("M1_CROSS_BASELINE_REAR_START",m1.rows)]; _write(args.output/"local_corridor_summary.csv",summaries)
    config_fields=set(GeometryProfileConfig.__dataclass_fields__)
    m0s,m1s=summaries; local_contract=LidarProfileJunctionDetector.RUNTIME_INPUTS==("angles_deg","ranges") and "corridor_width" not in config_fields
    contamination=abs(m1s["width_at_detection"]-m1s["width_before_detection"]) if math.isfinite(m1s["width_at_detection"]) else math.inf
    passed=m0s["false_corridor_trigger_count"]==0 and m1s["false_corridor_trigger_count"]==0 and abs(m0s["estimated_width_error_eval_only"])<1e-8 and m1s["clear_sample_count_before_detection"]>0 and math.isfinite(m1s["first_detection_time"]) and contamination<1e-8 and m1s["model_frozen_at_detection"] and local_contract and m0s["max_nan_inf"]==m1s["max_nan_inf"]==0
    verdict="LOCAL_CORRIDOR_ESTIMATION_VALID" if passed else "VALIDATION_FAILED"
    _write(args.output/"local_corridor_verdict.csv",[{"verdict":verdict,"runtime_inputs":"angles_deg,ranges","known_corridor_width_runtime":False,"GT_dependency":False,"map_dependency":False,"legacy_shadow_dependency":False,"rear_start_shift_eval_only":REAR_START_SHIFT,"noise_std":0.0,"dropout":0.0,"occlusion":0.0}])
    plot_profile(args.output/"lidar_profile_clear.png",m1_shots["clear"],"rear-start M1 clear corridor")
    plot_profile(args.output/"lidar_profile_before_detection.png",m1_shots["before_detection"],"rear-start M1 before detection")
    plot_profile(args.output/"lidar_profile_first_detection.png",m1_shots["first_detection"],"rear-start M1 first detection")
    print(f"verdict={verdict} output={args.output.resolve()}")


if __name__=="__main__": main()
