"""Seed-controlled evaluation runner for lateral/topology shadow signals."""
from __future__ import annotations
import argparse, csv, os, subprocess, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC = ROOT / "pygame_simulator/single_junction_sph_dfs_lidar_front_trigger_diagnostics.py"
SEEDS = [1, 7, 17, 23, 42, 101, 202, 303, 404, 505]

def f(v):
    return None if v in (None, "") else float(v)

def events(path):
    with path.open(newline="", encoding="utf-8") as h:
        return {r["event"]: f(r["timestamp_or_delta_s"]) for r in csv.DictReader(h)}

def run_one(seed, args, out):
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env.update({"SDL_VIDEODRIVER":"dummy", "SPH_DFS_MAX_FRAMES":str(args.frames), "SPH_DFS_SEED":str(seed), "SPH_DFS_DIAGNOSTIC_OUTPUT":str(out)})
    subprocess.run([sys.executable, str(DIAGNOSTIC)], cwd=ROOT, env=env, check=True)
    with (out / "local_front_surface_peak_timeline.csv").open(newline="", encoding="utf-8") as h: surface = list(csv.DictReader(h))
    with (out / "local_sph_boundary_component_timeline.csv").open(newline="", encoding="utf-8") as h: topo = list(csv.DictReader(h))
    ev = events(out / "lidar_front_trigger_event_summary.csv")
    peak = max(surface, key=lambda r: float(r["forward_zero_neighbor_fraction"])) if surface else {}
    split = next((r for r in topo if int(r["boundary_gap120_component_count"]) >= 2), None)
    phases = {}
    with (out / "local_sph_boundary_component_phase_summary.csv").open(newline="", encoding="utf-8") as h:
        for r in csv.DictReader(h):
            if r["threshold"] == "120": phases[r["phase"]] = r
    return {"seed":seed,"sample_count":len(surface),"frontmost_crossing":ev.get("frontmost_boundary_crossing"),"front_center_crossing":ev.get("front_cohort_center_crossing"),"lateral_sustained_onset":ev.get("first_sustained_lateral_onset"),"boundary_first_split":f(split["timestamp"]) if split else None,"delta_lateral_front_center":None if ev.get("first_sustained_lateral_onset") is None or ev.get("front_cohort_center_crossing") is None else ev["first_sustained_lateral_onset"]-ev["front_cohort_center_crossing"],"delta_split_front_center":None if split is None or ev.get("front_cohort_center_crossing") is None else f(split["timestamp"])-ev["front_cohort_center_crossing"],"delta_split_vs_lateral":None if split is None or ev.get("first_sustained_lateral_onset") is None else f(split["timestamp"])-ev["first_sustained_lateral_onset"],"surface_peak_time":f(peak.get("timestamp")),"surface_peak_fraction":f(peak.get("forward_zero_neighbor_fraction")),"corridor_component_ge2_fraction":float(phases.get("SPH_CORRIDOR",{}).get("component_count_mean",0) or 0),"opening_component_ge2_fraction":float(phases.get("SPH_OPENING_APPROACH",{}).get("component_count_mean",0) or 0),"corridor_largest_component_fraction_mean":f(phases.get("SPH_CORRIDOR",{}).get("largest_component_fraction_mean")),"opening_largest_component_fraction_mean":f(phases.get("SPH_OPENING_APPROACH",{}).get("largest_component_fraction_mean"))}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--seeds",nargs="+",type=int,default=SEEDS); p.add_argument("--frames",type=int,default=600); p.add_argument("--output-dir",type=Path,default=ROOT/"junction_detection/integration/output/lidar_front_trigger_multiseed"); a=p.parse_args()
    rows=[run_one(s,a,a.output_dir/f"seed_{s:03d}") for s in a.seeds]; a.output_dir.mkdir(parents=True,exist_ok=True)
    with (a.output_dir/"multiseed_run_summary.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (a.output_dir/"multiseed_event_alignment.csv").open("w",newline="",encoding="utf-8") as h:
        fields=["seed","frontmost_crossing","front_center_crossing","lateral_sustained_onset","boundary_first_split","delta_lateral_front_center","delta_split_front_center","delta_split_vs_lateral"]; w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows({k:r[k] for k in fields} for r in rows)
    valid=[r for r in rows if r["front_center_crossing"] is not None]; summary={"run_count":len(rows),"valid_run_count":len(valid),"lateral_onset_detection_rate":sum(r["lateral_sustained_onset"] is not None for r in valid)/max(len(valid),1),"boundary_split_detection_rate":sum(r["boundary_first_split"] is not None for r in valid)/max(len(valid),1)}
    for name in ("delta_lateral_front_center","delta_split_front_center","delta_split_vs_lateral"):
        x=np.asarray([r[name] for r in valid if r[name] is not None]); summary[f"{name}_mean"]=float(x.mean()) if len(x) else ""; summary[f"{name}_std"]=float(x.std()) if len(x) else ""; summary[f"{name}_median"]=float(np.median(x)) if len(x) else ""
    with (a.output_dir/"multiseed_lateral_vs_boundary_summary.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.writer(h); w.writerow(["metric","value"]); w.writerows(summary.items())
    fig,ax=plt.subplots(2,2,figsize=(10,7)); idx=np.arange(len(rows)); ax[0,0].bar(idx,[r["delta_lateral_front_center"] or np.nan for r in rows]); ax[0,0].set_title("lateral Δt vs front center"); ax[0,1].bar(idx,[r["delta_split_front_center"] or np.nan for r in rows]); ax[0,1].set_title("split Δt vs front center"); ax[1,0].bar(idx,[r["delta_split_vs_lateral"] or np.nan for r in rows]); ax[1,0].set_title("split - lateral onset"); ax[1,1].bar(idx,[r["opening_component_ge2_fraction"] for r in rows],label="opening"); ax[1,1].bar(idx,[r["corridor_component_ge2_fraction"] for r in rows],alpha=.6,label="corridor"); ax[1,1].legend(); ax[1,1].set_title("component response")
    fig.suptitle("Seed validation (GT references evaluation-only)"); fig.tight_layout(); fig.savefig(a.output_dir/"multiseed_validation.png",dpi=150); plt.close(fig)
    print(f"seeds={a.seeds} output={a.output_dir.resolve()}")
if __name__ == "__main__": main()
