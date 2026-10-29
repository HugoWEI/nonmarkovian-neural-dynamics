"""Independent two-axis convergence study for the source-anchored fermionic HEOM.

**Not** a Cao et al. 2026 PINN-DQME reproduction: the initial state is a
factorized empty impurity and has no pre-existing correlated ADOs.

Each temperature uses progressively larger Padé order Nk at fixed hierarchy
depth d and larger depth at fixed Nk. This separates the two error sources.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import comb

if __package__:
    from .fermionic_heom_reference import system, HBAR_MEV_PS
else:
    from fermionic_heom_reference import system, HBAR_MEV_PS


def make_solver(ratio: float, nk: int, depth: int, rtol=1e-8, atol=1e-10):
    from qutip.solver.heom import HEOMSolver, LorentzianPadeBath
    h, up, down, nup, ndown = system()
    opts = {
        "nsteps": 20000, "rtol": rtol, "atol": atol,
        "store_states": True, "progress_bar": "",
    }
    baths = [
        LorentzianPadeBath(
            op, 0.1/HBAR_MEV_PS, 5.0/HBAR_MEV_PS,
            mu/HBAR_MEV_PS, ratio*0.1/HBAR_MEV_PS, nk,
            tag=f"{spin}_{lead}"
        )
        for spin, op in (("up",up),("down",down))
        for lead, mu in (("L",0.0),("R",0.05))
    ]
    solver = HEOMSolver(h, baths, max_depth=depth, options=opts)
    return solver, nup, ndown


def solve(ratio: float, nk: int, depth: int, t_end: float, n_times: int):
    from qutip import tensor, basis
    start = time.monotonic()
    solver, nup, ndown = make_solver(ratio,nk,depth)
    construct_seconds=time.monotonic()-start
    rho0=tensor(basis(2,0),basis(2,0)).proj()
    times=np.linspace(0,t_end,n_times)
    init=time.monotonic()
    result=solver.run(rho0,times,e_ops={"n_up":nup,"n_down":ndown})
    integrate_seconds=time.monotonic()-init
    pop=np.vstack([
        np.real(result.e_data["n_up"]),
        np.real(result.e_data["n_down"])
    ]).T
    states=result.states
    if len(states)!=n_times:
        raise RuntimeError("HEOM did not save expected system states")
    traces=np.array([complex(s.tr()) for s in states])
    min_eigs=np.array([
        np.linalg.eigvalsh((s.full()+s.full().conj().T)/2).min().real
        for s in states
    ])
    herm_err=max(float(np.max(abs(s.full()-s.full().conj().T))) for s in states)
    return {
        "ratio":ratio,"Nk":nk,"depth":depth,
        "n_ados":len(solver.ados.labels),
        "construct_seconds":construct_seconds,
        "integrate_seconds":integrate_seconds,
        "max_trace_error":float(np.max(abs(traces-1.0))),
        "min_density_eigenvalue":float(min_eigs.min()),
        "max_hermiticity_error":herm_err,
        "terminal_n_up":float(pop[-1,0]),
        "terminal_n_down":float(pop[-1,1]),
        "times":times, "occupations":pop,
    }


def compare(a,b):
    assert len(a["times"])==len(b["times"])
    delta=abs(a["occupations"]-b["occupations"])
    return {"max_occupation_difference":float(delta.max()),
            "rms_occupation_difference":float(np.sqrt(np.mean(delta**2))),
            "terminal_occupation_difference":float(delta[-1].max())}


def run(ratio, out, t_end=1.5, n_times=31):
    output=Path(out)
    output.mkdir(parents=True,exist_ok=True)
    orders=(1,3) if ratio==3 else (1,3,7)
    # The highest source-matched Nk for each temperature is tested
    # at both depths 1 and 2. No depth 3 until cost is separately evaluated.
    records=[]
    traces={}
    for nk in orders:
        for depth in (1,2):
            solved=solve(ratio,nk,depth,t_end,n_times)
            traces[(nk,depth)]=solved
            record={k:v for k,v in solved.items() if k not in ("times","occupations")}
            records.append(record)
            print(json.dumps(record),flush=True)
            filename=f"T{ratio}_Nk{nk}_depth{depth}.csv"
            with (output/filename).open("w",newline="") as f:
                writer=csv.writer(f)
                writer.writerow(("time_ps","n_up","n_down"))
                writer.writerows(
                    (float(t),float(o[0]),float(o[1]))
                    for t,o in zip(solved["times"],solved["occupations"])
                )
    differences=[]
    for nk in orders:
        differences.append({
            "vary":"depth","fixed_Nk":nk,"depths":[1,2],
            **compare(traces[(nk,1)],traces[(nk,2)])
        })
    for depth in (1,2):
        for lo,hi in zip(orders[:-1],orders[1:]):
            differences.append({
                "vary":"Pade","fixed_depth":depth,"Nk_pair":[lo,hi],
                **compare(traces[(lo,depth)],traces[(hi,depth)])
            })
    results={"temperature_ratio":ratio,"records":records,"differences":differences,
             "initial_state":"empty impurity; factorized bath hierarchy",
             "source_matched_order":orders[-1],
             "converged":False}
    # Do NOT infer convergence from comparing just two depth/order values.
    (output/"results.json").write_text(json.dumps(results,indent=2))
    summary=[
        f"# HEOM convergence diagnostics T/Gamma={ratio}", "",
        "**NOT a converged result or reproduction of source initial ADOs.**",
        "Initial dot is empty and hierarchy ADOs are factorized at t=0.",
        "Comparisons use common timestamps; two adjacent truncations do not prove convergence.",
        "",
        "| Nk | depth | ADOs | final n_up | trace error | minimum eigenvalue | integration sec |",
        "|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for r in records:
        summary.append(
            f"| {r['Nk']} | {r['depth']} | {r['n_ados']} | {r['terminal_n_up']:.6f} | "
            f"{r['max_trace_error']:.2e} | {r['min_density_eigenvalue']:.2e} | "
            f"{r['integrate_seconds']:.2f} |"
        )
    summary.extend(["", "## Adjacent truncation differences",
                    "| Vary | fixed setting | pair | maximum delta n |",
                    "|---|---|---|---:|"])
    for x in differences:
        if x["vary"]=="depth":
            fixed,pair=f"Nk={x['fixed_Nk']}",str(x["depths"])
        else:
            fixed,pair=f"depth={x['fixed_depth']}",str(x["Nk_pair"])
        summary.append(
            f"| {x['vary']} | {fixed} | {pair} | "
            f"{x['max_occupation_difference']:.6e} |"
        )
    summary += ["", "No manuscript-level numerical convergence can be claimed",
                "until increasing both depth and Nk ceases to change physical observables",
                "at a prespecified tolerance, and integrator error is checked.",
                "State preparation differs from the original correlated input hierarchy."]
    (output/"SUMMARY.md").write_text("\n".join(summary)+"\n")
    print("\n".join(summary),flush=True)
    return results


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ratio",type=float,choices=(0.3,3.0),required=True)
    p.add_argument("--output",default=None)
    p.add_argument("--t-end",type=float,default=1.5)
    args=p.parse_args()
    run(args.ratio,args.output or f"outputs/heom_convergence/T{args.ratio}",
        t_end=args.t_end)
