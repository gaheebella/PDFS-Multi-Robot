"""Validate scan-only local corridor orientation correction in ideal geometry."""

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

from junction_detection.integration.run_lidar_local_corridor_estimation import _rear_start
from junction_detection.pointcloud.lidar_profile_junction_detector import (
    GeometryProfileConfig,LidarProfileJunctionDetector,expected_corridor_ranges_from_walls,
)
from junction_detection.pointcloud.pointcloud_junction_detector import _circular_runs,_validate_circular_scan
from pygame_simulator.pre_exploration_general_pipeline_simulator import (
    GeometryBuilder,LIDAR_MAX_RANGE,LidarSensor,SimulationRunner,
)

DEFAULT_OUTPUT=ROOT/"junction_detection/integration/output/lidar_local_corridor_orientation"
M0_YAWS=(0.0,5.0,-5.0,10.0,-10.0,15.0,-15.0)
M1_YAWS=(0.0,10.0,-10.0)
GT_WIDTH=84.0


def _write(path: Path, rows: list[dict]) -> None:
    """Write one homogeneous CSV table."""
    if not rows: return
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _axial_error(estimate: float, truth: float) -> float:
    """Return signed error between 180-degree-periodic line directions."""
    return float((estimate-truth+90.0)%180.0-90.0)


def _orientation_off(scan, min_beams: int=2) -> dict:
    """Evaluate the frozen pre-orientation formulation for comparison only."""
    angles,ranges,_=_validate_circular_scan(scan.angles_deg,scan.ranges)
    left=float(ranges[int(np.argmin(np.abs(angles-90.0)))])
    right=float(ranges[int(np.argmin(np.abs(angles+90.0)))])
    expected,valid=expected_corridor_ranges_from_walls(angles,left,right,scan.max_range,0.0)
    candidates=valid&(ranges-expected>max(1.0e-8,np.finfo(float).eps*scan.max_range*128.0))
    groups=[run for run in _circular_runs(candidates) if len(run)>=min_beams]
    return {"width":left+right,"profile_error":float(np.max(np.abs(ranges[valid]-expected[valid]))),"group_count":len(groups)}


def synthetic_orientation_tests() -> list[dict]:
    """Exercise exact wall fitting, yaw sweep, and axial wrap-around."""
    sensor=LidarSensor(); geometry=GeometryBuilder.build("M0_STRAIGHT")
    rows=[]
    for yaw in M0_YAWS:
        scan=sensor.scan(geometry,np.array([0.0,-100.0]),90.0-yaw)
        detector=LidarProfileJunctionDetector(GeometryProfileConfig(sensor.max_range,initialization_scan_count=1))
        result=detector.detect(scan.angles_deg,scan.ranges)
        rows.append({"test":f"ideal_yaw_{yaw:+g}","truth_deg":yaw,"estimate_deg":result["current_corridor_orientation_deg"],"error_deg":_axial_error(result["current_corridor_orientation_deg"],yaw),"parallel_error_deg":result["parallel_error_deg"],"passed":abs(_axial_error(result["current_corridor_orientation_deg"],yaw))<1e-8 and result["side_walls_valid"]})
    wrap=LidarProfileJunctionDetector._axial_mean_deg((5.0,-175.0))
    rows.append({"test":"axial_5_minus175","truth_deg":5.0,"estimate_deg":wrap,"error_deg":_axial_error(wrap,5.0),"parallel_error_deg":0.0,"passed":abs(_axial_error(wrap,5.0))<1e-8})
    return rows


def run_m0_sweep() -> tuple[list[dict],dict[float,dict]]:
    """Evaluate physical M0 scans at controlled sensor/body yaw offsets."""
    sensor=LidarSensor(); geometry=GeometryBuilder.build("M0_STRAIGHT")
    rows=[]; snapshots={}
    for yaw in M0_YAWS:
        scan=sensor.scan(geometry,np.array([0.0,-100.0]),90.0-yaw)
        detector=LidarProfileJunctionDetector(GeometryProfileConfig(sensor.max_range))
        result=None
        for _ in range(detector.config.initialization_scan_count): result=detector.detect(scan.angles_deg,scan.ranges)
        assert result is not None
        off=_orientation_off(scan,detector.config.min_beam_count)
        rows.append({
            "yaw_eval_only":yaw,"psi_hat_deg":result["stable_corridor_orientation_deg"],"orientation_error_eval_only":_axial_error(result["stable_corridor_orientation_deg"],yaw),
            "left_wall_orientation_deg":result["left_wall_orientation_deg"],"right_wall_orientation_deg":result["right_wall_orientation_deg"],"parallel_error_deg":result["parallel_error_deg"],
            "estimated_corridor_width":result["estimated_corridor_width"],"width_error_eval_only":result["estimated_corridor_width"]-GT_WIDTH,"estimated_offset":result["estimated_offset"],
            "profile_max_error":result["profile_max_abs_valid_delta"],"opening_group_count":result["opening_group_count"],"false_junction_count":int(result["profile_junction_detected"]),
            "orientation_off_width":off["width"],"orientation_off_width_error_eval_only":off["width"]-GT_WIDTH,"orientation_off_profile_error":off["profile_error"],"orientation_off_group_count":off["group_count"],
        })
        snapshots[yaw]={"angles":scan.angles_deg.copy(),"ranges":scan.ranges.copy(),"expected":result["expected_ranges"].copy(),"psi":result["stable_corridor_orientation_deg"],"width":result["estimated_corridor_width"]}
    return rows,snapshots


def _m1_row(row: dict, yaw: float) -> dict:
    """Select runtime diagnostics and append explicitly evaluation-only truth."""
    return {
        "frame":row["frame"],"time":row["timestamp"],"yaw_eval_only":yaw,
        "psi_current_deg":row["current_corridor_orientation_deg"],"psi_stable_deg":row["stable_corridor_orientation_deg"],"orientation_error_eval_only":_axial_error(row["stable_corridor_orientation_deg"],yaw) if math.isfinite(row["stable_corridor_orientation_deg"]) else math.nan,
        "left_wall_orientation_deg":row["left_wall_orientation_deg"],"right_wall_orientation_deg":row["right_wall_orientation_deg"],"parallel_error_deg":row["parallel_error_deg"],
        "left_wall_perp_distance":row["left_wall_range"],"right_wall_perp_distance":row["right_wall_range"],"estimated_corridor_width":row["estimated_corridor_width"],"estimated_offset":row["estimated_offset"],
        "orientation_initialized":row["orientation_initialized"],"orientation_frozen":row["orientation_frozen"],"opening_candidate_count":row["opening_candidate_count"],"opening_group_count":row["opening_group_count"],"junction_detected":row["profile_junction_detected"],
        "opening_groups_json":row["opening_groups_json"],"GT_phase_eval_only":row["gt_phase"],"GT_opening_visible_eval_only":row["GT_opening_visible_eval_only"],"noise_std":0.0,"dropout":0.0,"occlusion":0.0,
    }


def run_m1(yaw: float, frames: int) -> tuple[list[dict],dict|None]:
    """Run the existing rear-start M1 physics with only initial local yaw changed."""
    detector=LidarProfileJunctionDetector(GeometryProfileConfig(LIDAR_MAX_RANGE))
    runner=SimulationRunner("M1_CROSS_BASELINE","local_forward",profile_detector=detector)
    _rear_start(runner)
    body_yaw=math.radians(90.0-yaw)
    for robot in runner.world.robots: robot.body_yaw_rad=body_yaw
    runner.world.lidar_yaw_deg=math.degrees(body_yaw)
    output=[]; first_snapshot=None
    for frame in range(frames):
        row=runner.step(frame)
        if row is None: continue
        leader=next(robot for robot in runner.world.robots if robot.robot_id==runner.world.lidar_robot_id)
        half=0.5*runner.geometry.incoming_width; entrance=float(runner.geometry.entrance_y)
        corner_distance=min(math.hypot(leader.position[0]-side,entrance-leader.position[1]) for side in (-half,half))
        row["GT_opening_visible_eval_only"]=corner_distance<=LIDAR_MAX_RANGE+1e-9
        output.append(_m1_row(row,yaw))
        if first_snapshot is None and row["profile_junction_detected"]:
            scan=runner.last_visual[0].lidar_scan; result=runner.last_profile_result
            first_snapshot={"angles":scan.angles_deg.copy(),"ranges":scan.ranges.copy(),"expected":result["expected_ranges"].copy(),"psi":result["stable_corridor_orientation_deg"],"width":result["estimated_corridor_width"]}
    return output,first_snapshot


def summarize_m1(yaw: float, rows: list[dict]) -> dict:
    """Summarize CLEAR-to-DETECTED timing and frozen-model contamination."""
    initialized=next((row for row in rows if row["orientation_initialized"]),None)
    candidate=next((row for row in rows if row["opening_candidate_count"]>0),None)
    grouped=next((row for row in rows if row["opening_group_count"]>0),None)
    detected=next((row for row in rows if row["junction_detected"]),None)
    visible_index=next((index for index,row in enumerate(rows) if row["GT_opening_visible_eval_only"]),len(rows))
    before=[row for row in rows[:visible_index] if row["orientation_initialized"]]
    frozen=[row for row in rows if row["orientation_frozen"]]
    freeze=frozen[0] if frozen else None
    orientation_contamination=max((abs(_axial_error(row["psi_stable_deg"],freeze["psi_stable_deg"])) for row in frozen),default=math.inf)
    width_contamination=max((abs(row["estimated_corridor_width"]-freeze["estimated_corridor_width"]) for row in frozen),default=math.inf)
    return {
        "case":"M1_CROSS_BASELINE_REAR_START","yaw_eval_only":yaw,"sample_count":len(rows),"orientation_initialization_time":initialized["time"] if initialized else math.nan,
        "psi_before_opening":float(np.mean([row["psi_stable_deg"] for row in before])) if before else math.nan,"orientation_error_before_opening_eval_only":float(np.mean([row["orientation_error_eval_only"] for row in before])) if before else math.nan,
        "width_before_opening":float(np.mean([row["estimated_corridor_width"] for row in before])) if before else math.nan,"clear_samples":sum(not row["junction_detected"] for row in rows[:rows.index(detected)]) if detected else len(rows),
        "first_candidate_time":candidate["time"] if candidate else math.nan,"first_group_time":grouped["time"] if grouped else math.nan,"first_detection_time":detected["time"] if detected else math.nan,
        "orientation_at_freeze":freeze["psi_stable_deg"] if freeze else math.nan,"width_at_freeze":freeze["estimated_corridor_width"] if freeze else math.nan,"opening_group_count":detected["opening_group_count"] if detected else 0,"opening_groups":detected["opening_groups_json"] if detected else "[]",
        "false_trigger_before_opening_visibility":sum(row["junction_detected"] for row in rows[:visible_index]),"orientation_contamination_after_freeze":orientation_contamination,"width_contamination_after_freeze":width_contamination,
    }


def plot_profile(path: Path, snapshot: dict, yaw: float) -> None:
    """Save one corrected measured/expected range profile."""
    fig,axis=plt.subplots(figsize=(9,4.5)); axis.plot(snapshot["angles"],snapshot["ranges"],color="black",label="measured"); axis.plot(snapshot["angles"],snapshot["expected"],"--",color="tab:blue",label="orientation-corrected expected")
    axis.set(xlabel="body-relative angle [deg]",ylabel="range",xlim=(-180,179),ylim=(0,LIDAR_MAX_RANGE*1.05),title=f"yaw eval={yaw:+.0f} deg | psi_hat={snapshot['psi']:+.3f} deg | width={snapshot['width']:.3f}")
    axis.grid(alpha=.25); axis.legend(); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)


def main(argv=None) -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--m1-frames",type=int,default=180); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); args=parser.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    synthetic=synthetic_orientation_tests(); m0,snapshots=run_m0_sweep()
    all_m1=[]; summaries=[]; m1_snapshots={}
    for yaw in M1_YAWS:
        rows,snapshot=run_m1(yaw,args.m1_frames); all_m1.extend(rows); summaries.append(summarize_m1(yaw,rows)); m1_snapshots[yaw]=snapshot
    _write(args.output/"orientation_synthetic_tests.csv",synthetic); _write(args.output/"orientation_sweep_m0.csv",m0); _write(args.output/"orientation_timeline_m1.csv",all_m1); _write(args.output/"orientation_summary.csv",summaries)
    passed=(all(row["passed"] for row in synthetic) and all(abs(row["orientation_error_eval_only"])<1e-8 and abs(row["width_error_eval_only"])<1e-8 and row["false_junction_count"]==0 for row in m0) and all(math.isfinite(row["first_detection_time"]) and row["clear_samples"]>0 and row["false_trigger_before_opening_visibility"]==0 and row["orientation_contamination_after_freeze"]<1e-8 and row["width_contamination_after_freeze"]<1e-8 for row in summaries))
    verdict="LOCAL_CORRIDOR_ORIENTATION_VALID" if passed else "VALIDATION_FAILED"
    _write(args.output/"orientation_verdict.csv",[{"verdict":verdict,"runtime_inputs":"angles_deg,ranges","global_yaw_runtime":False,"known_width_runtime":False,"map_runtime":False,"GT_runtime":False,"noise_std":0.0,"dropout":0.0,"occlusion":0.0}])
    for yaw in (0.0,10.0,-10.0):
        snapshot=m1_snapshots[yaw] or snapshots[yaw]
        label="minus10" if yaw<0 else str(int(yaw))
        plot_profile(args.output/f"orientation_profile_yaw{label}.png",snapshot,yaw)
    print(f"verdict={verdict} output={args.output.resolve()}")


if __name__=="__main__": main()
