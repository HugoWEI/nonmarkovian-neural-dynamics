"""Prespecified Hamiltonian robustness under exact unit-norm architecture.

This is NOT transfer: every model is retrained at each physical parameter.
"""
import csv,json,statistics
from pathlib import Path
from reduced_fermionic_pinn import train

PARAMS=(("original",-.5,.1),("detuned",-.25,.1),
        ("stronger_coupling",-.5,.25),("joint_shift",-.85,.2))
SEEDS=(1,3,7)
def run(output="outputs/unitnorm_parameters"):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,eps,gamma in PARAMS:
        for T in (.03,.3):
            for seed in SEEDS:
                for mode in ("uniform","global_projected"):
                    result=train(mode=mode,temp=T,seed=seed,steps=1200,
                                 dot_energy=eps,gamma=gamma,unit_norm=True)
                    result["parameter_case"]=name
                    rows.append(result)
                    print(json.dumps(result),flush=True)
    with (out/"runs.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (out/"runs.json").write_text(json.dumps(rows,indent=2))
    aggregate=[]
    for name,eps,gamma in PARAMS:
        for temp in (.03,.3):
            for mode in ("uniform","global_projected"):
                group=[r for r in rows if (r["parameter_case"],r["T_mev"],r["mode"])==(name,temp,mode)]
                mean=lambda k:statistics.mean(r[k] for r in group)
                aggregate.append({
                    "case":name,"eps":eps,"Gamma":gamma,"T":temp,
                    "mode":mode,"terminal":mean("terminal_occupation_abs_error"),
                    "trajectory":mean("occupation_rmse"),
                    "norm":mean("max_wave_norm_drift"),
                    "runtime":mean("runtime_seconds")
                })
    (out/"aggregate.json").write_text(json.dumps(aggregate,indent=2))
    summary=["# Hard-norm neural objective Hamiltonian stress test","",
        "Every neural model retrained on its Hamiltonian, not pretrained transfer.",
        "4 Hamiltonian parameter cases × 2 temperatures × 3 seeds × 2 losses.",
        "Physically unit-norm wavefunctions, but not certified valid quantum maps.",
        "Exact small-system propagator in goal weights; no efficiency claim.","",
        "| case | T | loss | terminal error | trajectory RMSE | norm drift | CPU s |",
        "|---|---:|---|---:|---:|---:|---:|"]
    for r in aggregate:
        summary.append(f"| {r['case']} | {r['T']} | {r['mode']} | "
            f"{r['terminal']:.5e} | {r['trajectory']:.5e} | "
            f"{r['norm']:.3e} | {r['runtime']:.2f} |")
    summary+=["","A failure in any case is not resolved by selecting only successful",
              "Hamiltonians. No inference about the source interacting SIAM."]
    (out/"SUMMARY.md").write_text("\n".join(summary)+"\n")
    return aggregate
if __name__=="__main__":
    run()
