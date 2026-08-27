"""Short, non-consuming RNG/initial-state effectiveness audit."""
from __future__ import annotations
import argparse,csv,hashlib,os,subprocess,sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
TARGET=ROOT/"pygame_simulator/single_junction_sph_dfs_lidar_front_trigger_diagnostics.py"

def run_one(run_id,seed,frames,out):
    out.mkdir(parents=True,exist_ok=True); env=os.environ.copy(); env.update({"SDL_VIDEODRIVER":"dummy","SPH_DFS_MAX_FRAMES":str(frames),"SPH_DFS_SEED":str(seed),"SPH_DFS_DIAGNOSTIC_OUTPUT":str(out)})
    subprocess.run([sys.executable,str(TARGET)],cwd=ROOT,env=env,check=True)
    with (out/"seed_effectiveness_run_metadata.csv").open(newline="",encoding="utf-8") as h:return next(csv.DictReader(h))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--seeds",nargs="+",type=int,default=[1,7,17]); p.add_argument("--frames",type=int,default=3); p.add_argument("--repeats",type=int,default=2); p.add_argument("--output-dir",type=Path,default=ROOT/"junction_detection/integration/output/lidar_front_trigger_seed_audit"); a=p.parse_args(); rows=[]; run_id=0
    for seed in a.seeds:
        for repeat in range(a.repeats):
            run_id+=1; row=run_one(run_id,seed,a.frames,a.output_dir/f"seed_{seed:03d}_repeat_{repeat+1}"); row.update({"run_id":run_id,"requested_seed":seed,"repeat":repeat+1}); rows.append(row)
    a.output_dir.mkdir(parents=True,exist_ok=True); fields=list(rows[0]);
    with (a.output_dir/"seed_effectiveness_run_summary.csv").open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)
    base=rows[0]; pairs=[]
    for row in rows[1:]:
        pairs.append({"seed_a":base["requested_seed"],"seed_b":row["requested_seed"],"checkpoint":"raw_initial_state","hash_equal":base["raw_initial_state_hash"]==row["raw_initial_state_hash"]})
    with (a.output_dir/"seed_effectiveness_pairwise_comparison.csv").open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
    print(f"runs={len(rows)} output={a.output_dir.resolve()}")
if __name__=="__main__":main()
