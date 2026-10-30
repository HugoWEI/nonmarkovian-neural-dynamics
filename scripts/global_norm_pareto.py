"""Prespecified Pareto sweep for physical norm versus terminal occupation.

Controls are FIXED before inspecting outcomes. Direct quadratic propagation
is substantially cheaper than any of these models. No novelty claimed.
"""
import csv,json,statistics
from pathlib import Path
from reduced_fermionic_pinn import train

SEEDS=(1,2,3,7,11)
CONFIGS=(
 ("uniform",.5,0.,"uniform"),
 ("uniform",.5,1.,"uniform_norm1"),
 ("global_projected",.25,0.,"global_w025"),
 ("global_projected",.5,0.,"global_w050"),
 ("global_projected",.75,0.,"global_w075"),
 ("global_projected",.5,.1,"global_w050_norm01"),
 ("global_projected",.5,1.,"global_w050_norm1"),
)

def run(output="outputs/global_norm_pareto",steps=1200):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    allrows=[]
    for temp in (.03,.3):
        for seed in SEEDS:
            for mode,gw,nw,label in CONFIGS:
                result=train(mode=mode,temp=temp,seed=seed,steps=steps,
                             goal_weight=gw,norm_weight=nw)
                result["setting"]=label
                allrows.append(result)
                print(json.dumps(result),flush=True)
    with (out/"runs.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0]))
        writer.writeheader();writer.writerows(allrows)
    (out/"runs.json").write_text(json.dumps(allrows,indent=2))
    summary=[]
    for temp in (.03,.3):
        for _,_,_,label in CONFIGS:
            s=[r for r in allrows if r["T_mev"]==temp and r["setting"]==label]
            m=lambda k: statistics.mean(r[k] for r in s)
            summary.append({"T_mev":temp,"setting":label,"n":len(s),
                "terminal":m("terminal_occupation_abs_error"),
                "trajectory":m("occupation_rmse"),
                "norm":m("max_wave_norm_drift"),
                "seconds":m("runtime_seconds")})
    (out/"aggregate.json").write_text(json.dumps(summary,indent=2))
    lines=["# Norm/terminal trade-off: preregistered small-system sweep","",
           "A finite 20-mode noninteracting bath; NOT the original PINN-DQME.",
           "Same 1,200 steps/seed/temperature for all seven settings.",
           "Results report MEANS over five seeds, not a significance claim.","",
           "| T (meV) | setting | terminal err | traj RMSE | norm drift | train seconds |",
           "|---:|---|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['T_mev']} | {r['setting']} | {r['terminal']:.5e} | "
                     f"{r['trajectory']:.5e} | {r['norm']:.5e} | {r['seconds']:.2f} |")
    lines+=["","Lagrange norm penalty enforces a physical state constraint only",
            "approximately; neither trace preservation nor complete positivity",
            "of a generic quantum dynamical map is established.",
            "Training uses an explicit small-system propagator. The direct",
            "solver is faster; this study isolates optimization trade-offs."]
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    return summary

if __name__=="__main__":
    run()
