"""Source-parameterized full-ADO steady state and voltage-quench test.

NOT a reproduction of Cao et al. stored initial hierarchy: preparation
is explicitly an independently computed unbiased (muL=muR=0) HEOM steady
state, followed by a right-lead voltage quench to muR=0.05 meV.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from heom_convergence import make_solver


def diagnostics(states):
    traces=[complex(state.tr()) for state in states]
    eigs=[
        np.linalg.eigvalsh((state.full()+state.full().conj().T)/2).real
        for state in states
    ]
    return {"max_trace_error":float(max(abs(t-1) for t in traces)),
            "minimum_eigenvalue":float(min(e.min() for e in eigs))}


def run(ratio, nk, depth, output, t_end=1.5, n_times=31):
    out=Path(output)
    out.mkdir(parents=True,exist_ok=True)
    start=perf_counter()
    unbiased,up,down=make_solver(ratio,nk,depth,mu_right_mev=0.)
    steady_rho,full_ados=unbiased.steady_state(use_mkl=False)
    ss_seconds=perf_counter()-start
    labels_a=list(unbiased.ados.labels)
    t=np.linspace(0.,t_end,n_times)
    stay=unbiased.run(full_ados,t,e_ops={"up":up,"down":down})
    biased,up_b,down_b=make_solver(ratio,nk,depth,mu_right_mev=.05)
    labels_b=list(biased.ados.labels)
    if labels_a!=labels_b:
        raise AssertionError("ADO ordering changed under chemical-potential quench")
    quench=biased.run(full_ados,t,e_ops={"up":up_b,"down":down_b})

    equilibrium=np.stack([np.real(stay.e_data["up"]),np.real(stay.e_data["down"])],axis=1)
    population=np.stack([np.real(quench.e_data["up"]),np.real(quench.e_data["down"])],axis=1)
    stationary_drift=float(np.max(abs(equilibrium-equilibrium[0])))
    quench_change=float(np.max(abs(population-population[0])))
    if stationary_drift>1e-5:
        raise AssertionError(f"Computed HEOM steady state not stationary: {stationary_drift}")
    if not np.isfinite(population).all():
        raise FloatingPointError("Nonfinite occupations in voltage quench")

    report={
        "T_over_Gamma":ratio, "Nk":nk, "depth":depth,
        "n_ados":len(labels_a),
        "steady_state_seconds":ss_seconds,
        "steady_state_n_up":float(equilibrium[0,0]),
        "steady_state_n_down":float(equilibrium[0,1]),
        "stationary_drift":stationary_drift,
        "max_quench_occupation_change":quench_change,
        "terminal_n_up_after_quench":float(population[-1,0]),
        "terminal_n_down_after_quench":float(population[-1,1]),
        "steady_density":diagnostics([steady_rho]),
        "quench_density":diagnostics(quench.states),
        "equilibrium_density":diagnostics(stay.states),
    }
    with (out/"trajectories.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["time_ps","n_up_unbiased","n_down_unbiased","n_up_biased","n_down_biased"])
        writer.writerows([[float(t[i]),*equilibrium[i],*population[i]] for i in range(len(t))])
    (out/"results.json").write_text(json.dumps(report,indent=2))
    lines=[
        f"# Correlated-state HEOM voltage quench T/Gamma={ratio}", "",
        f"Padé Nk={nk}, depth={depth}, {len(labels_a)} ADOs.",
        "**Not the authors' original stored correlated state or published neural results.**",
        "Initial *full ADO hierarchy* is QuTiP's computed unbiased steady state;",
        "then muR changes from 0 to 0.05 meV while muL remains 0.",
        "", "| Diagnostic | Measured value |",
        "|---|---:|"
    ]
    for key,val in report.items():
        if isinstance(val,(int,float)) and key!="T_over_Gamma":
            lines.append(f"| {key} | {val:.7g} |")
    lines += ["", f"Quench density diagnostics: {report['quench_density']}",
              "", "Stationarity and physicality checks do not establish",
              "convergence across Padé order or hierarchy depth.",
              "Current measurements require first-tier ADO observables and are not yet included."]
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(report),flush=True)
    return report


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ratio",required=True,type=float)
    ap.add_argument("--nk",required=True,type=int)
    ap.add_argument("--depth",default=1,type=int)
    ap.add_argument("--output",default=None)
    args=ap.parse_args()
    run(args.ratio,args.nk,args.depth,args.output or f"outputs/heom_quench/T{args.ratio}")
