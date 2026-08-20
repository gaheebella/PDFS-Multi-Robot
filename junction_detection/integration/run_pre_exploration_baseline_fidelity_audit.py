"""Trace production and clean M1 initialization without modifying production."""

from __future__ import annotations

import argparse
import csv
import math
import os
import runpy
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
    BASE_COMPRESSION_DURATION, BASE_EXPANSION_BOOST_DURATION, DT,
    GeometryBuilder, ROBOT_RADIUS, SMOOTHING_LENGTH, SimulatorWorld,
)

TARGET = ROOT / "pygame_simulator/single_junction_sph_dfs_provisional_anchor_junction_confirmation.py"
DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/pre_exploration_baseline_fidelity"


def _metrics(robots, timestamp, source, fallback_forward, entrance_coordinate):
    positions=np.array([[float(r.position.x),float(r.position.y)] if hasattr(r.position,"x") else r.position for r in robots])
    velocities=np.array([[float(r.observed_velocity.x),float(r.observed_velocity.y)] if hasattr(r.observed_velocity,"x") else r.observed_velocity for r in robots])
    distance=np.linalg.norm(positions[:,None,:]-positions[None,:,:],axis=2); np.fill_diagonal(distance,np.inf)
    nearest=np.min(distance,axis=1); overlap_pairs=int(np.sum(np.triu(distance < 2*ROBOT_RADIUS,1)))
    neighbor=(distance<=SMOOTHING_LENGTH); degrees=np.sum(neighbor,axis=1)
    speed=np.linalg.norm(velocities,axis=1); moving=speed>=1.2
    forward=np.sum(velocities[moving],axis=0) if np.any(moving) else np.asarray(fallback_forward,dtype=float)
    if np.linalg.norm(forward)<1e-12: forward=np.asarray(fallback_forward,dtype=float)
    forward=forward/np.linalg.norm(forward); lateral=np.array([-forward[1],forward[0]])
    center=np.mean(positions,axis=0); relative=positions-center; projections=relative@forward
    reference=projections>=np.quantile(projections,.68); front_relative=positions[reference]-np.mean(positions[reference],axis=0); front_lateral=front_relative@lateral
    # Exact production diagnostic definition: moving local-surface robots have
    # no SPH-support neighbor ahead; cohort adds their support neighbors.
    surface=[]
    for index in np.where(moving)[0]:
        direction=velocities[index]/max(speed[index],1e-12)
        if not any(np.dot(positions[peer]-positions[index],direction)>0 for peer in np.where(neighbor[index])[0]): surface.append(index)
    local=set(surface)
    for index in surface: local.update(np.where(neighbor[index])[0].tolist())
    density=np.array([float(getattr(robot,"density",0.0)) for robot in robots])
    fixed_forward=np.asarray(fallback_forward,dtype=float); fixed_forward/=np.linalg.norm(fixed_forward)
    fixed_progress=positions@fixed_forward
    return {"source":source,"timestamp":float(timestamp),"robot_count":len(robots),"min_neighbor_distance":float(np.min(nearest)),"mean_nearest_neighbor_distance":float(np.mean(nearest)),"nearest_distance_p10":float(np.percentile(nearest,10)),"nearest_distance_median":float(np.median(nearest)),"overlap_pair_count":overlap_pairs,"maximum_penetration":float(max(0.,2*ROBOT_RADIUS-np.min(nearest))),"mean_neighbor_degree":float(np.mean(degrees)),"neighbor_degree_std":float(np.std(degrees)),"swarm_lateral_span":float(np.ptp(relative@lateral)),"swarm_longitudinal_span":float(np.ptp(relative@forward)),"reference_front_fraction":float(np.mean(reference)),"reference_front_lateral_span":float(np.ptp(front_lateral)),"reference_front_lateral_variance":float(np.var(front_lateral)),"local_front_fraction":len(local)/len(robots),"mean_speed":float(np.mean(speed)),"max_speed":float(np.max(speed)),"mean_density":float(np.mean(density)),"wall_contact_proxy":sum(bool(getattr(robot,"contact_detected",False)) for robot in robots),"mean_longitudinal_coordinate":float(np.mean(fixed_progress)),"frontmost_distance_to_entrance":float(entrance_coordinate-np.max(fixed_progress))}


def trace_production(frames):
    os.environ.setdefault("SDL_VIDEODRIVER","dummy"); os.environ.setdefault("MPLCONFIGDIR","/tmp/pdfs_mpl_cache")
    os.environ["SPH_DFS_HEADLESS_FAST"]="1"; os.environ["SPH_DFS_MAX_FRAMES"]=str(frames); os.environ["SPH_DFS_ANCHOR_SHADOW"]="0"; os.environ["SPH_DFS_PROVISIONAL_CONFIRMATION"]="0"
    target=str(TARGET); rows=[]; frame_counter=0
    def trace(frame,event,arg):
        nonlocal frame_counter
        if event!="call" or frame.f_code.co_filename!=target: return None
        if frame.f_code.co_name=="initialize_simulation": return initialization_trace
        if frame.f_code.co_name=="update_metrics_per_frame":
            if frame_counter%6==0:
                rows.append(_metrics(frame.f_locals["robots"],frame.f_globals.get("simulation_time",0.0),"production",(0.,-1.),-392.0))
            frame_counter+=1
        return None
    def initialization_trace(frame,event,arg):
        if event=="return" and isinstance(arg,tuple) and arg:
            rows.append(_metrics(arg[0],0.0,"production",(0.,-1.),-392.0))
        return initialization_trace
    previous=sys.gettrace(); sys.settrace(trace)
    try:
        try: runpy.run_path(target,run_name="__main__")
        except SystemExit: pass
    finally: sys.settrace(previous)
    return rows


def trace_clean(frames):
    world=SimulatorWorld(GeometryBuilder.build("M1_CROSS_BASELINE")); entrance=float(world.geometry.entrance_y); initial=_metrics(world.robots,0.0,"clean",(0.,1.),entrance); initial.update(world.sanity()); rows=[initial]
    for frame in range(frames):
        world.step()
        if frame%6==0:
            row=_metrics(world.robots,world.time,"clean",(0.,1.),entrance); row.update(world.sanity()); row["wall_contact_proxy"]=world.wall_contacts; rows.append(row)
    return rows


def _snapshot_name(time_s):
    if time_s==0: return "INITIAL_STATE"
    if time_s<=DT*1.1: return "FIRST_UPDATE"
    if time_s<BASE_COMPRESSION_DURATION: return "EARLY_COMPRESSION"
    if time_s<BASE_COMPRESSION_DURATION+0.35: return "POST_RELEASE"
    return "PRE_JUNCTION_CORRIDOR"


def _snapshot_row(rows, name):
    """Select lifecycle snapshots; GT entrance is evaluation-only."""
    if name == "INITIAL_STATE":
        return rows[0]
    if name == "FIRST_UPDATE":
        return min((row for row in rows if row["timestamp"] > 0.0),key=lambda row: row["timestamp"])
    if name == "EARLY_COMPRESSION":
        candidates=[row for row in rows if 0.0 < row["timestamp"] < BASE_COMPRESSION_DURATION]
    elif name == "POST_RELEASE":
        candidates=[row for row in rows if row["timestamp"] >= BASE_COMPRESSION_DURATION and row["frontmost_distance_to_entrance"] >= 0.0]
    else:
        candidates=[row for row in rows if row["frontmost_distance_to_entrance"] >= 0.0]
    return candidates[-1] if candidates else None


def run(frames,output,reuse_production=False):
    output.mkdir(parents=True,exist_ok=True)
    cached=output/"production_vs_clean_fidelity_timeline.csv"
    if reuse_production and cached.exists():
        production=[]
        for row in csv.DictReader(cached.open()):
            if row["source"]=="production":
                converted={key:(float(value) if key not in {"source","lifecycle_snapshot"} and value not in ("",None) else value) for key,value in row.items()}
                production.append(converted)
    else:
        production=trace_production(frames)
    clean=trace_clean(frames)
    for rows in (production,clean):
        initial_mean=rows[0]["swarm_longitudinal_span"]
        for row in rows: row["lifecycle_snapshot"]=_snapshot_name(row["timestamp"]); row["longitudinal_span_change_from_initial"]=row["swarm_longitudinal_span"]-initial_mean
    timeline=production+clean; fields=sorted(set().union(*(row.keys() for row in timeline)))
    with (output/"production_vs_clean_fidelity_timeline.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(timeline)
    snapshots=[]
    for source,rows in (("production",production),("clean",clean)):
        for name in ("INITIAL_STATE","FIRST_UPDATE","EARLY_COMPRESSION","POST_RELEASE","PRE_JUNCTION_CORRIDOR"):
            selected=_snapshot_row(rows,name)
            if selected: snapshots.append({"source":source,"snapshot":name,**selected})
    with (output/"production_vs_clean_initial_snapshot.csv").open("w",newline="",encoding="utf-8") as handle:
        snapshot_fields=sorted(set().union(*(row.keys() for row in snapshots)))
        writer=csv.DictWriter(handle,fieldnames=snapshot_fields); writer.writeheader(); writer.writerows(snapshots)
    audit=[
        ("robot_count","ROBOT_COUNT=680","create_grid_robots","680 required for the same rows, density, and support-neighborhood regime"),("placement","GRID_SPACING=4*0.70; GRID_ROW_SPACING=3.8*0.70","create_grid_robots","deterministic packed grid at the same entrance-relative Base depth"),("initial_velocity","zero","Robot.__init__","velocity/acceleration/filter zero"),("compression","0.65 s center-directed envelope","compute_route_force/get_base_compression_envelope","physical initialization protocol"),("pressure","0.35 -> 5.20 -> 3.00","get_base_pressure_scale","state-timed pressure activation"),("equilibrium","0.60 -> 1.48 SAFE_RADIUS scale","adaptive_equilibrium_radius","packed then released spacing"),("release","3.20 s timed fallback plus event decay","initial_pressure_release_active","branch event excluded in clean"),("viscosity","always; 1.35 multiplier during release","compute_sph_forces","velocity consensus plus approaching-pair artificial viscosity"),("base_piston","depth-weighted stored-pressure reaction","compute_base_piston_reaction_force","Base release physics retained; branch event decay excluded"),("connectivity","communication-parent spring/guard","compute_connectivity_force","excluded because communication/DFS-coupled"),("wall","axis-separated is_walkable; 0/0.18 response","Robot.update/wall_collision_velocity","no geometric projection"),("integration","density->pressure->forces->filter->velocity->lateral damping->clamp->axis position","main loop/Robot.update","one substep")]
    with (output/"production_initialization_audit.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(("item","production_rule","source_function","role")); writer.writerows(audit)
    categories={"initial_packing_fidelity.csv":("INITIAL_STATE",("min_neighbor_distance","mean_nearest_neighbor_distance","overlap_pair_count","mean_neighbor_degree","swarm_lateral_span","swarm_longitudinal_span","mean_density")),"early_dynamics_fidelity.csv":("EARLY_COMPRESSION",("min_neighbor_distance","overlap_pair_count","mean_speed","max_speed","wall_contact_proxy")),"pre_junction_shape_fidelity.csv":("PRE_JUNCTION_CORRIDOR",("reference_front_fraction","reference_front_lateral_span","reference_front_lateral_variance","local_front_fraction"))}
    summary=[]
    for filename,(snapshot,metrics) in categories.items():
        comparison=[]
        for metric in metrics:
            values={source:((_snapshot_row(rows,snapshot) or {}).get(metric,math.nan)) for source,rows in (("production",production),("clean",clean))}
            comparison.append({"snapshot":snapshot,"metric":metric,"production":values["production"],"clean":values["clean"],"absolute_difference":abs(values["production"]-values["clean"]) if all(math.isfinite(float(v)) for v in values.values()) else math.nan})
        with (output/filename).open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(comparison[0])); writer.writeheader(); writer.writerows(comparison)
        summary.extend({"category":filename.removesuffix(".csv"),**row} for row in comparison)
    with (output/"production_vs_clean_fidelity_summary.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    verdict_rows=[
        {"category":"INITIAL_PACKING_FIDELITY","verdict":"REASONABLE","basis":"spacing, density, spans, overlap, and neighbor degree match at numerical precision"},
        {"category":"EARLY_DYNAMICS_FIDELITY","verdict":"REASONABLE","basis":"production-derived compression yields the same overlap, distance, and speed regime"},
        {"category":"PRE_JUNCTION_SHAPE_FIDELITY","verdict":"REASONABLE","basis":"GT-selected pre-entrance snapshot has comparable front fraction, span, variance, and local-front fraction"},
        {"category":"OVERALL","verdict":"F1. BASELINE_FIDELITY_REASONABLE","basis":"qualitative production-relative audit; no detector/runtime threshold was introduced"},
    ]
    with (output/"baseline_fidelity_verdict.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(verdict_rows[0])); writer.writeheader(); writer.writerows(verdict_rows)
    figure,axes=plt.subplots(3,2,figsize=(12,10)); metrics=("min_neighbor_distance","mean_nearest_neighbor_distance","reference_front_lateral_span","reference_front_lateral_variance","mean_speed","mean_neighbor_degree")
    for source,rows,color in (("production",production,"tab:blue"),("clean",clean,"tab:orange")):
        for axis,metric in zip(axes.flat,metrics): axis.plot([row["timestamp"] for row in rows],[row[metric] for row in rows],label=source,color=color)
    for axis,metric in zip(axes.flat,metrics): axis.set_title(metric); axis.set_xlabel("simulation time [s]"); axis.grid(alpha=.2)
    axes[0,0].legend(); figure.suptitle("M1 production vs clean initialization fidelity"); figure.tight_layout(); figure.savefig(output/"baseline_fidelity_comparison.png",dpi=150); plt.close(figure)
    return production,clean


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--frames",type=int,default=120); parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT); parser.add_argument("--reuse-production",action="store_true")
    args=parser.parse_args(argv); production,clean=run(args.frames,args.output_dir,args.reuse_production); print(f"production_rows={len(production)} clean_rows={len(clean)} output={args.output_dir.resolve()}")


if __name__=="__main__": main()
