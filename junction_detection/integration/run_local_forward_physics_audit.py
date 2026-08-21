"""Audit centered-leader, locally propagated LOCAL_FORWARD physics on M0/M1."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pygame_simulator.pre_exploration_general_pipeline_simulator import (
    GRID_SPACING, ROBOT_RADIUS, GeometryBuilder, SimulatorWorld, run_headless,
)

DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/local_forward_physics_audit"
BEFORE_TIMELINE = ROOT / "junction_detection/integration/output/no_compression_exploration_audit/no_compression_timeline.csv"
CASES = ("M0_STRAIGHT", "M1_CROSS_BASELINE")
METRICS = (
    "gt_mean_forward_progress", "mean_speed_sanity", "max_speed",
    "min_inter_robot_distance", "overlap_pair_count", "mean_density",
    "density_std", "mean_neighbor_degree", "reference_front_lateral_span",
    "reference_front_lateral_variance", "boundary_count", "boundary_fraction",
    "boundary_component_count", "boundary_largest_component_fraction",
    "wall_contact_count", "outside_free_space_robot_count",
)


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read(path: Path) -> list[dict]:
    rows=[]
    for row in csv.DictReader(path.open()):
        converted={}
        for key,value in row.items():
            try: converted[key]=float(value)
            except (TypeError,ValueError): converted[key]=value
        rows.append(converted)
    return rows


def _summary(rows: list[dict], case: str) -> list[dict]:
    result=[]
    for phase in dict.fromkeys(row["gt_phase"] for row in rows):
        selected=[row for row in rows if row["gt_phase"]==phase]
        item={"map_case":case,"phase":phase,"sample_count":len(selected)}
        for metric in METRICS:
            values=np.asarray([row[metric] for row in selected],dtype=float)
            item[f"{metric}_mean"]=float(np.mean(values)); item[f"{metric}_min"]=float(np.min(values)); item[f"{metric}_max"]=float(np.max(values))
        item.update({"max_nan_inf_state_count":max(row["nan_inf_state_count"] for row in selected),"final_wall_projection_correction_count":selected[-1]["wall_projection_correction_count"]})
        result.append(item)
    return result


def _initial_metadata() -> dict:
    world=SimulatorWorld(GeometryBuilder.build("M1_CROSS_BASELINE"),"local_forward")
    leader=next(robot for robot in world.robots if robot.robot_id==world.lidar_robot_id)
    neighbors=world._neighbors()
    return {"lidar_robot_id":leader.robot_id,"lidar_initial_x":float(leader.position[0]),"lidar_initial_y":float(leader.position[1]),"front_row_center_x":world.initial_front_center_x,"front_row_center_offset":float(leader.position[0]-world.initial_front_center_x),"leader_heading_hop":leader.heading_hop,"leader_propulsion_weight":leader.propulsion_weight,"max_follower_heading_hop":max(robot.heading_hop for robot in world.robots),"follower_propulsion_weight":next(robot.propulsion_weight for robot in world.robots if robot.robot_id!=leader.robot_id),"initial_mean_density":float(np.mean([robot.density for robot in world.robots])),"initial_density_std":float(np.std([robot.density for robot in world.robots])),"initial_mean_neighbor_degree":float(np.mean([len(neighbors[robot.robot_id]) for robot in world.robots])),"initial_min_distance":world.sanity()["min_inter_robot_distance"]}


def run(frames: int, output: Path, before_timeline: Path) -> dict:
    output.mkdir(parents=True,exist_ok=True)
    runners={case:run_headless(case,frames,"local_forward") for case in CASES}
    after={case:runners[case].rows for case in CASES}
    _write(output/"local_forward_m0_timeline.csv",after["M0_STRAIGHT"])
    _write(output/"local_forward_m1_timeline.csv",after["M1_CROSS_BASELINE"])
    _write(output/"local_forward_m0_summary.csv",_summary(after["M0_STRAIGHT"],"M0_STRAIGHT"))
    _write(output/"local_forward_m1_summary.csv",_summary(after["M1_CROSS_BASELINE"],"M1_CROSS_BASELINE"))

    before_rows=_read(before_timeline)
    comparison=[]
    for case in CASES:
        before=[row for row in before_rows if row["map_case"]==case]
        for scope,phase in (("ALL",None),("JUNCTION_REGION","JUNCTION_REGION")):
            if phase and case=="M0_STRAIGHT": continue
            old=[row for row in before if phase is None or row["gt_phase"]==phase]
            new=[row for row in after[case] if phase is None or row["gt_phase"]==phase]
            for metric in METRICS:
                old_values=[float(row[metric]) for row in old if metric in row]
                new_values=[float(row[metric]) for row in new]
                comparison.append({"map_case":case,"scope":scope,"metric":metric,"before_mean":float(np.mean(old_values)) if old_values else math.nan,"after_mean":float(np.mean(new_values)),"before_min":float(np.min(old_values)) if old_values else math.nan,"after_min":float(np.min(new_values)),"before_max":float(np.max(old_values)) if old_values else math.nan,"after_max":float(np.max(new_values))})
    _write(output/"local_forward_before_after_comparison.csv",comparison)

    m0=after["M0_STRAIGHT"]; m1=after["M1_CROSS_BASELINE"]; initial=_initial_metadata()
    m0_severe=min(row["min_inter_robot_distance"] for row in m0)<ROBOT_RADIUS
    m1_severe=min(row["min_inter_robot_distance"] for row in m1)<ROBOT_RADIUS
    m0_dominant=min(row["boundary_largest_component_fraction"] for row in m0)>0.95
    m1_expands=max(row["reference_front_lateral_span"] for row in m1)>max(row["reference_front_lateral_span"] for row in m0)
    m1_topology=min(row["boundary_largest_component_fraction"] for row in m1)<min(row["boundary_largest_component_fraction"] for row in m0)
    centered=abs(initial["front_row_center_offset"])<=GRID_SPACING/2+1e-9
    valid=(not m0_severe and not m1_severe and m0_dominant and m1_expands and m1_topology and centered)
    verdict={"verdict":"A. LOCAL_FORWARD_PHYSICS_VALID" if valid else "B. PARTIALLY_VALID","frames":frames,**initial,"m0_final_progress":m0[-1]["gt_mean_forward_progress"],"m0_min_distance":min(row["min_inter_robot_distance"] for row in m0),"m0_max_overlap_pairs":max(row["overlap_pair_count"] for row in m0),"m0_min_largest_component_fraction":min(row["boundary_largest_component_fraction"] for row in m0),"m1_final_progress":m1[-1]["gt_mean_forward_progress"],"m1_min_distance":min(row["min_inter_robot_distance"] for row in m1),"m1_max_overlap_pairs":max(row["overlap_pair_count"] for row in m1),"m1_max_lateral_span":max(row["reference_front_lateral_span"] for row in m1),"m1_min_largest_component_fraction":min(row["boundary_largest_component_fraction"] for row in m1),"max_nan_inf":max(row["nan_inf_state_count"] for rows in after.values() for row in rows),"max_outside":max(row["outside_free_space_robot_count"] for rows in after.values() for row in rows),"reason":"M0 remains one dominant corridor body; M1 expands laterally with stable pair spacing and no GT-directed force" if valid else "see measured fields"}
    _write(output/"local_forward_physics_verdict.csv",[verdict])
    return verdict


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames",type=int,default=180)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--before-timeline",type=Path,default=BEFORE_TIMELINE)
    args=parser.parse_args(argv)
    verdict=run(args.frames,args.output_dir,args.before_timeline)
    print(f"verdict={verdict['verdict']} output={args.output_dir.resolve()}")


if __name__=="__main__": main()
