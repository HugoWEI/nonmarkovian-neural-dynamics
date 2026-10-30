"""Prespecified Hamiltonian-sensitivity test: not cross-system transfer.

Each PINN is retrained separately for every Hamiltonian. We test robustness
of a training objective, NOT zero-shot generalization of a pretrained model.
"""
import csv,json,statistics
from pathlib import Path
from reduced_fermionic_pinn import train

PARAMETERS=(
 ("original",-0.5,0.10),
 ("detuned",-0.25,0.10),
 ("stronger_coupling",-0.5,0.25),
 ("joint_shift",-0.85,0.20),
)
SEEDS=(1,3,7)

def run(out="outputs/global_parameter_robustness",steps=1200):
    dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,eps,gamma in PARAMETERS:
        for temp in (.03,.3):
            for seed in SEEDS:
                for mode in ("uniform","global_projected"):
                    value=train(mode=mode,temp=temp,seed=seed,steps=steps,
                                goal_weight=.5,norm_weight=0.,
                                dot_energy=eps,gamma=gamma)
                    value["parameter_case"]=name
                    rows.append(value)
                    print(json.dumps(value),flush=True)
    with (dest/"runs.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (dest/"runs.json").write_text(json.dumps(rows,indent=2))
    summary=[]
    for name,eps,gamma in PARAMETERS:
        for T in (.03,.3):
            for mode in ("uniform","global_projected"):
                subset=[r for r in rows if (r["parameter_case"],r["T_mev"],r["mode"])==(name,T,mode)]
                avg=lambda k:statistics.mean(r[k] for r in subset)
                summary.append({"case":name,"dot_energy":eps,"gamma":gamma,"T":T,
                    "mode":mode,"mean_terminal":avg("terminal_occupation_abs_error"),
                    "mean_global_rmse":avg("occupation_rmse"),
                    "mean_norm_drift":avg("max_wave_norm_drift"),
                    "mean_runtime":avg("runtime_seconds")})
    (dest/"aggregate.json").write_text(json.dumps(summary,indent=2))
    lines=["# Hamiltonian-parameter robustness, global-terminal objective","",
        "All 48 runs have a separate trained network; no transfer-learning claim.",
        "Finite 20-mode quadratic bath, 1,200 total optimizer steps and three seeds.",
        "Same setup/steps per paired global-terminal and uniform baseline.","",
        "| Hamiltonian | T | loss | mean final occupation error | trajectory RMSE | norm drift | CPU s |",
        "|---|---:|---|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['case']} | {r['T']} | {r['mode']} | "
              f"{r['mean_terminal']:.6e} | {r['mean_global_rmse']:.6e} | "
              f"{r['mean_norm_drift']:.6e} | {r['mean_runtime']:.2f} |")
    lines+=["", "Direct exact propagation remains cheaper. This tests only",
             "sensitivity to problem parameters, not quantum error backflow or",
             "a novel interacting-system solver."]
    (dest/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    return summary

if __name__=="__main__":
    run()
