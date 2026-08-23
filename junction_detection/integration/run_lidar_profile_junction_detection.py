"""Evaluate the ideal moving-LiDAR geometry-profile Junction baseline."""

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

from junction_detection.pointcloud.lidar_profile_junction_detector import (
    GeometryProfileConfig,
    LidarProfileJunctionDetector,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import (
    LIDAR_MAX_RANGE,
    MIN_SPEED,
    SimulationRunner,
)

DEFAULT_OUTPUT=ROOT/"junction_detection/integration/output/lidar_profile_junction_detection"
GROUP_FIELDS=("map_case","frame","time","group_id","start_angle_deg","end_angle_deg","center_angle_deg","angular_width_deg","beam_count","start_body_angle_deg","end_body_angle_deg","center_body_angle_deg","mean_range","max_range","mean_delta_range","max_delta_range","junction_detected")


def _write(path: Path, rows: list[dict], fieldnames=None) -> None:
    """Write a CSV, retaining headers for an empty group table."""
    names=list(fieldnames or (list(rows[0]) if rows else []))
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=names); writer.writeheader(); writer.writerows(rows)


def _new_detector() -> LidarProfileJunctionDetector:
    """Create the fixed ideal/noiseless baseline configuration."""
    return LidarProfileJunctionDetector(GeometryProfileConfig(LIDAR_MAX_RANGE))


def run_case(case_id: str, frames: int) -> tuple[SimulationRunner,dict,dict|None]:
    """Run continuously from motion start without legacy shadow actuation."""
    runner=SimulationRunner(case_id,"local_forward",profile_detector=_new_detector())
    corridor_snapshot=None; detection_snapshot=None
    for frame in range(frames):
        row=runner.step(frame)
        if row is None: continue
        leader=next(robot for robot in runner.world.robots if robot.robot_id==runner.world.lidar_robot_id)
        # Evaluation-only pose is appended after the local detector decision.
        row["lidar_x_eval"]=float(leader.position[0]); row["lidar_y_eval"]=float(leader.position[1])
        scan=runner.last_visual[0].lidar_scan; result=runner.last_profile_result
        snapshot={"angles":scan.angles_deg.copy(),"ranges":scan.ranges.copy(),"expected":result["expected_ranges"].copy(),"candidate":result["open_candidate_mask"].copy(),"confirmed":result["confirmed_opening_mask"].copy(),"frame":frame,"time":row["timestamp"],"groups":list(result["opening_groups"])}
        if corridor_snapshot is None: corridor_snapshot=snapshot
        if detection_snapshot is None and row["profile_junction_detected"]: detection_snapshot=snapshot
    return runner,corridor_snapshot,detection_snapshot


def timeline(rows: list[dict]) -> list[dict]:
    """Return compact runtime/evaluation timeline rows."""
    return [{
        "frame":row["frame"],"time":row["timestamp"],"map_case":row["map_case"],
        "lidar_x_eval":row["lidar_x_eval"],"lidar_y_eval":row["lidar_y_eval"],
        "estimated_corridor_width":row["estimated_corridor_width"],"max_range":row["profile_max_range"],"estimated_offset":row["estimated_offset"],
        "opening_group_count":row["opening_group_count"],"junction_detected":row["profile_junction_detected"],"detector_state":row["profile_detector_state"],
        "leader_speed":row["leader_speed"],"measured_profile_summary":json.dumps({"min":row["measured_profile_min"],"mean":row["measured_profile_mean"],"max":row["measured_profile_max"]},separators=(",",":")),
        "expected_profile_summary":json.dumps({"min":row["expected_profile_min"],"mean":row["expected_profile_mean"],"max":row["expected_profile_max"]},separators=(",",":")),
        "max_abs_valid_profile_delta":row["profile_max_abs_valid_delta"],"GT_phase_eval_only":row["gt_phase"],
        "nan_inf_state_count":row["nan_inf_state_count"],"noise_std":0.0,"dropout":0.0,"occlusion":0.0,
    } for row in rows]


def group_rows(case_id: str, rows: list[dict]) -> list[dict]:
    """Expand every detected circular group into long format."""
    output=[]
    for row in rows:
        for group in json.loads(row["opening_groups_json"]):
            output.append({"map_case":case_id,"frame":row["frame"],"time":row["timestamp"],**group,"junction_detected":row["profile_junction_detected"]})
    return output


def summarize(case_id: str, rows: list[dict]) -> dict:
    """Summarize detection timing, normal profile error, and sanity."""
    detections=[row for row in rows if row["profile_junction_detected"]]
    first=detections[0] if detections else None
    corridor_false=[row for row in detections if row["gt_phase"] in ("CORRIDOR","CORRIDOR_ONLY")]
    return {
        "map_case":case_id,"sample_count":len(rows),"detection_count":len(detections),"corridor_false_detection_count":len(corridor_false),
        "first_detection_frame":first["frame"] if first else -1,"first_detection_time":first["timestamp"] if first else math.nan,
        "first_detection_group_count":first["opening_group_count"] if first else 0,"first_detection_groups":first["opening_groups_json"] if first else "[]",
        "first_detection_gt_phase_eval_only":first["gt_phase"] if first else "NONE","first_detection_leader_speed":first["leader_speed"] if first else math.nan,
        "max_valid_profile_excess":max(row["profile_max_abs_valid_delta"] for row in rows),"max_group_count":max(row["opening_group_count"] for row in rows),
        "max_nan_inf":max(row["nan_inf_state_count"] for row in rows),"noise_std":0.0,"dropout":0.0,"occlusion":0.0,
    }


def plot_profile(path: Path, snapshot: dict, title: str) -> None:
    """Plot measured and analytic expected body-relative profiles."""
    fig,axis=plt.subplots(figsize=(10,4.8)); angles=snapshot["angles"]
    axis.plot(angles,snapshot["ranges"],label="measured rho(theta)",color="black",linewidth=1.1)
    axis.plot(angles,snapshot["expected"],label="expected corridor wall",color="tab:blue",linestyle="--")
    axis.fill_between(angles,0,snapshot["ranges"],where=snapshot["candidate"],color="orange",alpha=.22,label="OPEN_CANDIDATE")
    axis.fill_between(angles,0,snapshot["ranges"],where=snapshot["confirmed"],color="magenta",alpha=.20,label="confirmed group")
    axis.set(xlabel="body-relative angle [deg]",ylabel="range",xlim=(-180,179),ylim=(0,LIDAR_MAX_RANGE*1.05),title=f"{title} | frame={snapshot['frame']} t={snapshot['time']:.3f}s")
    axis.grid(alpha=.25); axis.legend(loc="lower center",ncol=4); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)


def main(argv=None) -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--frames",type=int,default=600); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); args=parser.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    m0,m0_corridor,_=run_case("M0_STRAIGHT",args.frames); m1,_,m1_detection=run_case("M1_CROSS_BASELINE",args.frames)
    _write(args.output/"lidar_profile_timeline_m0.csv",timeline(m0.rows)); _write(args.output/"lidar_profile_timeline_m1.csv",timeline(m1.rows))
    _write(args.output/"lidar_profile_groups_m0.csv",group_rows("M0_STRAIGHT",m0.rows),GROUP_FIELDS); _write(args.output/"lidar_profile_groups_m1.csv",group_rows("M1_CROSS_BASELINE",m1.rows),GROUP_FIELDS)
    summaries=[summarize("M0_STRAIGHT",m0.rows),summarize("M1_CROSS_BASELINE",m1.rows)]; _write(args.output/"lidar_profile_summary.csv",summaries)
    local_contract=set(LidarProfileJunctionDetector.RUNTIME_INPUTS)=={"angles_deg","ranges"}
    passed=summaries[0]["detection_count"]==0 and summaries[1]["detection_count"]>0 and summaries[1]["first_detection_leader_speed"]>=MIN_SPEED and all(row["max_nan_inf"]==0 for row in summaries) and local_contract
    verdict="LIDAR_PROFILE_BASELINE_VALID" if passed else "VALIDATION_FAILED"
    _write(args.output/"lidar_profile_verdict.csv",[{"verdict":verdict,"runtime_inputs":"angles_deg,ranges","corridor_width_configuration":"NONE_LOCAL_ESTIMATE","max_range_configuration":LIDAR_MAX_RANGE,"legacy_shadow_dependency":False,"GT_dependency":False,"noise_std":0.0,"dropout":0.0,"occlusion":0.0}])
    plot_profile(args.output/"lidar_profile_corridor.png",m0_corridor,"M0 ideal corridor")
    if m1_detection is not None: plot_profile(args.output/"lidar_profile_junction.png",m1_detection,"M1 first profile detection")
    print(f"verdict={verdict} output={args.output.resolve()}")


if __name__=="__main__": main()
