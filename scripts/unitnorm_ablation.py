"""Hard normalization versus soft penalties for quadratic fermionic PINN.

All comparisons use separately trained networks and equal optimizer steps.
A unit-norm wave does not guarantee correct quantum dynamics by itself.
"""
import csv,json,statistics
from pathlib import Path
from reduced_fermionic_pinn import train

SEEDS=(1,2,3,7,11)
CONFIGS=(("uniform",False,"uniform"),("global_projected",False,"global"),
         ("uniform",True,"uniform_unitnorm"),("global_projected",True,"global_unitnorm"))

def run(output="outputs/unitnorm_ablation"):
    dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for temp in (.03,.3):
        for seed in SEEDS:
            for mode,unit_norm,label in CONFIGS:
                result=train(mode=mode,temp=temp,seed=seed,steps=1200,unit_norm=unit_norm)
                result["setting"]=label
                rows.append(result)
                print(json.dumps(result),flush=True)
    with (dest/"runs.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    (dest/"runs.json").write_text(json.dumps(rows,indent=2))
    agg=[]
    for T in (.03,.3):
        for _,_,label in CONFIGS:
            s=[r for r in rows if (r["T_mev"],r["setting"])==(T,label)]
            m=lambda k:statistics.mean(x[k] for x in s)
            agg.append({"T_mev":T,"setting":label,"n_seeds":len(s),
                        "terminal_error":m("terminal_occupation_abs_error"),
                        "trajectory_rmse":m("occupation_rmse"),
                        "norm_drift":m("max_wave_norm_drift"),
                        "runtime_seconds":m("runtime_seconds")})
    (dest/"aggregate.json").write_text(json.dumps(agg,indent=2))
    lines=["# Hard unit-norm quantum neural output ablation","",
           "Finite 20-mode quadratic bath, 4 windows, same 1200 optimization steps.",
           "All results descriptive means across five seeds at each temperature.",
           "Hard normalization is exact for the wavefunction norm but",
           "does NOT guarantee a valid general CPTP quantum map.","",
           "| T (meV) | setting | terminal error | trajectory RMSE | norm drift | CPU sec |",
           "|---:|---|---:|---:|---:|---:|"]
    for r in agg:
        lines.append(f"| {r['T_mev']} | {r['setting']} | {r['terminal_error']:.5e} | "
                     f"{r['trajectory_rmse']:.5e} | {r['norm_drift']:.3e} | "
                     f"{r['runtime_seconds']:.2f} |")
    lines+=["","Exact quadratic propagation is still much cheaper than all neural methods.",
            "No claim about interacting SIAM, source DQME, physical strong-memory advantage",
            "or novelty of normalization networks."]
    (dest/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    return agg

if __name__=="__main__":
    run()
