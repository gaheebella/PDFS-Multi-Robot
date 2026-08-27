"""Headless validation for the clean pre-exploration research simulator."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pygame_simulator.pre_exploration_general_pipeline_simulator import (
    CASES, DEFAULT_OUTPUT, ROBOT_COUNT, run_headless, save_case,
)

METRICS = (
    "local_front_lateral_span", "local_front_lateral_velocity_variance",
    "motion_bearing_spread", "boundary_angular_spread",
    "mean_neighbor_degree", "lidar_left_wall_support",
    "lidar_right_wall_support", "lidar_forward_range",
    "lidar_free_space_angular_span", "lidar_range_profile_change",
)


def _stats(values):
    values=np.asarray(values,dtype=float)
    return {"mean":float(np.mean(values)),"median":float(np.median(values)),"std":float(np.std(values)),"min":float(np.min(values)),"max":float(np.max(values)),"mean_abs_temporal_change":float(np.mean(np.abs(np.diff(values)))) if len(values)>1 else 0.0}


def _production_fidelity(new_rows, output):
    production=ROOT/"junction_detection/integration/output/lidar_front_trigger_multiseed/seed_505/lidar_front_trigger_timeline.csv"
    fields=["metric","production_mean","new_mean","absolute_difference","note"]
    rows=[]
    if production.exists():
        old=list(csv.DictReader(production.open()))
        old_corridor=[row for row in old if row.get("evaluation_only_sph_phase")=="SPH_CORRIDOR"] or old
        mappings=(
            ("reference_front_fraction",[float(row["front_cohort_robot_count"])/680 for row in old_corridor],[float(row["reference_front_size"])/ROBOT_COUNT for row in new_rows],"normalized cohort size"),
            ("reference_front_lateral_span",[float(row["front_cohort_lateral_span"]) for row in old_corridor],[float(row["reference_front_lateral_span"]) for row in new_rows],"same spatial unit"),
            ("reference_front_lateral_variance",[float(row["lateral_variance"]) for row in old_corridor],[float(row["reference_front_lateral_variance"]) for row in new_rows],"same spatial unit squared"),
            ("local_front_fraction",[float(row["local_front_cohort_robot_count"])/680 for row in old_corridor],[float(row["local_front_size"])/ROBOT_COUNT for row in new_rows],"normalized; definitions aligned but harness differs"),
        )
        for name,old_values,new_values,note in mappings:
            old_mean=float(np.mean(old_values)); new_mean=float(np.mean(new_values))
            rows.append({"metric":name,"production_mean":old_mean,"new_mean":new_mean,"absolute_difference":abs(new_mean-old_mean),"note":note})
    with (output/"baseline_fidelity_comparison.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    if rows:
        figure,axis=plt.subplots(figsize=(9,4)); x=np.arange(len(rows)); width=.36
        axis.bar(x-width/2,[row["production_mean"] for row in rows],width,label="production artifact")
        axis.bar(x+width/2,[row["new_mean"] for row in rows],width,label="new M1")
        axis.set_xticks(x,[row["metric"] for row in rows],rotation=20,ha="right"); axis.legend(); axis.grid(axis="y",alpha=.2); figure.tight_layout(); figure.savefig(output/"baseline_fidelity_comparison.png",dpi=150); plt.close(figure)
    return rows


def run(frames,output):
    output.mkdir(parents=True,exist_ok=True); all_rows=[]; sanity=[]; runners={}
    for case in CASES:
        runner=run_headless(case,frames); runners[case]=runner; save_case(runner,output); all_rows.extend(runner.rows)
        last=runner.rows[-1]
        sanity.append({"map_case":case,"sample_count":len(runner.rows),"max_outside_free_space_robot_count":max(row["outside_free_space_robot_count"] for row in runner.rows),"wall_contact_count":last["wall_contact_count"],"wall_projection_correction_count":last["wall_projection_correction_count"],"max_nan_inf_state_count":max(row["nan_inf_state_count"] for row in runner.rows),"max_speed":max(row["max_speed"] for row in runner.rows),"min_inter_robot_distance":min(row["min_inter_robot_distance"] for row in runner.rows)})
    with (output/"pre_exploration_timeline.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    with (output/"pre_exploration_physical_sanity.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(sanity[0])); writer.writeheader(); writer.writerows(sanity)
    phase_rows=[]
    for case in CASES:
        phases=sorted({row["gt_phase"] for row in runners[case].rows})
        for phase in phases:
            selected=[row for row in runners[case].rows if row["gt_phase"]==phase]
            for metric in METRICS:
                phase_rows.append({"map_case":case,"phase":phase,"metric":metric,**_stats([row[metric] for row in selected])})
    with (output/"pre_exploration_phase_summary.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(phase_rows[0])); writer.writeheader(); writer.writerows(phase_rows)
    reference=runners["M0_STRAIGHT"].rows
    consistency=[]
    for metric in METRICS:
        base=float(np.mean([row[metric] for row in reference]))
        directions=[]
        for case in CASES[1:]:
            opening=[row for row in runners[case].rows if row["gt_phase"] in ("OPENING_APPROACH","BOUNDARY_CROSSING")]
            value=float(np.mean([row[metric] for row in opening])) if opening else math.nan
            difference=value-base if math.isfinite(value) else math.nan
            direction="NO_DATA" if not math.isfinite(difference) else "UP" if difference>0 else "DOWN" if difference<0 else "SAME"
            if math.isfinite(difference): directions.append(direction)
            consistency.append({"metric":metric,"map_case":case,"m0_corridor_mean":base,"junction_opening_mean":value,"absolute_difference":difference,"relative_change":difference/max(abs(base),1e-12) if math.isfinite(difference) else math.nan,"effect_direction":direction})
    with (output/"pre_exploration_geometry_consistency.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(consistency[0])); writer.writeheader(); writer.writerows(consistency)
    summaries=[]
    for case in CASES:
        rows=runners[case].rows; phases=sorted(set(row["gt_phase"] for row in rows))
        summaries.append({"map_case":case,"robot_count":ROBOT_COUNT,"sample_count":len(rows),"phases":";".join(phases),"lidar_robot_id":rows[0]["lidar_robot_id"],"max_outside":max(row["outside_free_space_robot_count"] for row in rows),"final_wall_corrections":rows[-1]["wall_projection_correction_count"],"mean_local_front_span":float(np.mean([row["local_front_lateral_span"] for row in rows])),"mean_motion_spread":float(np.mean([row["motion_bearing_spread"] for row in rows])),"mean_lidar_free_span":float(np.mean([row["lidar_free_space_angular_span"] for row in rows]))})
    with (output/"pre_exploration_run_summary.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    fidelity=_production_fidelity(runners["M1_CROSS_BASELINE"].rows,output)
    with (output/"baseline_fidelity_verdict.csv").open("w",newline="",encoding="utf-8") as handle:
        fields=["verdict","reason"]; writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        writer.writerow({"verdict":"BASELINE_FIDELITY_WEAK","reason":"bounded 240-robot cohort omits production initial compression pulse, communication guards, and state-coupled ingress forces; smoke also shows sub-diameter minimum spacing"})
    with (output/"pre_exploration_metric_ranking.csv").open("w",newline="",encoding="utf-8") as handle:
        fields=["metric","classification","reason"]; writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for metric in METRICS: writer.writerow({"metric":metric,"classification":"NOT_EVALUATED_FIDELITY_WEAK","reason":"generality inference deferred until M1 baseline fidelity is reasonable"})
    figure,axes=plt.subplots(3,2,figsize=(13,11)); plot_metrics=("local_front_lateral_span","local_front_lateral_velocity_variance","motion_bearing_spread","boundary_angular_spread","mean_neighbor_degree","lidar_free_space_angular_span")
    for case in CASES:
        rows=runners[case].rows; time=[row["timestamp"] for row in rows]
        for axis,metric in zip(axes.flat,plot_metrics): axis.plot(time,[row[metric] for row in rows],label=case)
    for axis,metric in zip(axes.flat,plot_metrics): axis.set_title(metric); axis.set_xlabel("time [s]"); axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=7); figure.suptitle("Pre-exploration local evidence — GT used only for retrospective phases"); figure.tight_layout(); figure.savefig(output/"pre_exploration_validation.png",dpi=150); plt.close(figure)
    return summaries,fidelity


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--frames",type=int,default=600); parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args(argv); summaries,fidelity=run(args.frames,args.output_dir); print(f"cases={len(summaries)} fidelity_rows={len(fidelity)} output={args.output_dir.resolve()}")


if __name__=="__main__": main()
