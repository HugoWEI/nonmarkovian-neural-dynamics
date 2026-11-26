"""Ablation: global-target versus local-target and ordinary piecewise PINN.

The global adjoint is computed using an explicit *small* quadratic propagator.
Therefore this experiment is a mathematical diagnostic/negative-control study,
NOT evidence for a cost-effective neural algorithm in interacting many-body QFT.
"""
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path
from reduced_fermionic_pinn import train

MODES=("uniform","projected","integrated","global_projected","global_integrated")
SEEDS=(1,2,3,7,11)

def run(output,steps=1200,seeds=SEEDS):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for temperature in (.03,.3):
        for seed in seeds:
            for mode in MODES:
                r=train(mode=mode,temp=temperature,seed=seed,steps=steps)
                rows.append(r)
                print(json.dumps(r),flush=True)
    with (out/"runs.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (out/"runs.json").write_text(json.dumps(rows,indent=2))
    aggregate=[]
    for T in (.03,.3):
        for mode in MODES:
            sample=[r for r in rows if r["T_mev"]==T and r["mode"]==mode]
            record={"T_mev":T,"mode":mode,"n_seeds":len(sample)}
            for field in ("occupation_rmse","terminal_occupation_abs_error",
                          "max_wave_norm_drift","runtime_seconds"):
                x=[r[field] for r in sample]
                record[field+"_mean"]=statistics.mean(x)
                record[field+"_median"]=statistics.median(x)
            aggregate.append(record)
    (out/"aggregate.json").write_text(json.dumps(aggregate,indent=2))
    lines=["# Full-horizon observable adjoint — controlled exploratory ablation","",
        "**Quadratic 20-mode finite-bath dot, not a converged continuum or interacting SIAM.**",
        "All methods share network width, 4-window training, initializations, 1200 total steps,",
        "and collocation points. Global adjoint explicitly uses exact small-system U(T,t),",
        "**which is already enough to solve this quadratic model**; report that cost.","",
        "| T (meV) | loss | mean trajectory RMSE | mean terminal error | mean norm drift | mean CPU s |",
        "|---:|---|---:|---:|---:|---:|"]
    for r in aggregate:
        lines.append(f"| {r['T_mev']} | {r['mode']} | "
            f"{r['occupation_rmse_mean']:.6e} | {r['terminal_occupation_abs_error_mean']:.6e} | "
            f"{r['max_wave_norm_drift_mean']:.6e} | {r['runtime_seconds_mean']:.3f} |")
    for T in (.03,.3):
        group=[r for r in aggregate if r["T_mev"]==T]
        base=next(r for r in group if r["mode"]=="uniform")
        for r in group:
            r["terminal_improvement_vs_uniform"]=(
                1-r["terminal_occupation_abs_error_mean"]/
                base["terminal_occupation_abs_error_mean"]
            )
    lines += ["","Means over five seeds are descriptive and are NOT statistical proof.",
        "The terminal observable is O=w^dag F w. Its adjoint uses the current neural",
        "state propagated to the *global* terminal time, not the exact target state.",
        "Finite-bath direct diagonalization remains far cheaper than the network.",
        "A lower training or trajectory residual does not imply physics advancement."]
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    return aggregate

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--steps",type=int,default=1200)
    p.add_argument("--output",default="outputs/global_goal")
    a=p.parse_args()
    run(a.output,a.steps)
