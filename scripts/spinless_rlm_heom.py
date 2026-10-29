"""Spinless Lorentzian fermionic HEOM vs independent quadratic RLM.

Same physical Hamiltonian, thermal spectral density, factorized empty-dot
initial state. Not the correlated spinful SIAM benchmark of Cao et al.
Hierarchy convergence and integration tolerances must be checked separately.
"""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from qutip import basis,destroy
from qutip.solver.heom import HEOMSolver,LorentzianPadeBath
from nonmarkovian_neural_dynamics.exact_rlm import HBAR,reference_with_checks

def heom(times,temperature,nk,depth,eps=-.5,gamma=.1,bandwidth=5.,
         rtol=1e-8,atol=1e-10):
    c=destroy(2); n=c.dag()*c
    H=eps/HBAR*n
    bath=LorentzianPadeBath(c,gamma/HBAR,bandwidth/HBAR,0.,
                            temperature/HBAR,nk)
    start=time.perf_counter()
    sol=HEOMSolver(H,bath,max_depth=depth,options={
        "nsteps":30000,"rtol":rtol,"atol":atol,
        "store_states":True,"progress_bar":""})
    setup=time.perf_counter()-start
    start=time.perf_counter()
    result=sol.run(basis(2,0).proj(),times,e_ops={"n":n})
    elapsed=time.perf_counter()-start
    occ=np.real(np.asarray(result.e_data["n"]))
    trace=np.array([s.tr() for s in result.states])
    min_eig=min(np.linalg.eigvalsh(
        (s.full()+s.full().conj().T)/2).min().real
        for s in result.states)
    return occ,{"Nk":nk,"depth":depth,"n_ados":len(sol.ados.labels),
                "setup_seconds":setup,"integrate_seconds":elapsed,
                "max_trace_error":float(max(abs(trace-1))),
                "min_density_eigenvalue":float(min_eig),
                "terminal_occupation":float(occ[-1])}

def run(ratio,out,depths=(2,3,4,5),duration=6.):
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    if ratio not in (.3,3.): raise ValueError("Supported T/Gamma=0.3 or 3")
    temp=.1*ratio;nk=7 if ratio==.3 else 3
    times=np.linspace(0,duration,121)
    reference,referr=reference_with_checks(times,temp)
    curves={};runs=[]
    for depth in depths:
        pred,record=heom(times,temp,nk,depth)
        curves[depth]=pred
        record.update({"T_over_Gamma":ratio,
                       "max_vs_exact":float(max(abs(pred-reference))),
                       "rmse_vs_exact":float(np.sqrt(np.mean((pred-reference)**2)))})
        print(json.dumps(record),flush=True)
        runs.append(record)
    with (output/"trajectories.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["time_ps","quadratic_exact"]+[f"heom_d{d}" for d in depths])
        w.writerows([times[i],reference[i]]+[curves[d][i] for d in depths]
                    for i in range(len(times)))
    adjacent=[{"depths":[a,b],
               "max_difference":float(max(abs(curves[a]-curves[b])))}
              for a,b in zip(depths[:-1],depths[1:])]
    results={"T_over_Gamma":ratio,"Nk":nk,"reference_checks":referr,
             "runs":runs,"adjacent":adjacent,
             "initial":"empty dot, factorized thermal bath",
             "paper_SIAM_reproduction":False,"convergence_certified":False}
    (output/"results.json").write_text(json.dumps(results,indent=2))
    lines=[f"# Independent spinless RLM/HEOM benchmark T/Gamma={ratio}","",
           "Source-style Lorentzian parameters; NOT the interacting correlated SIAM.",
           f"Quadratic reference grid max change: {referr['resolution_max']:.3e}; "
           f"cutoff max change: {referr['cutoff_max']:.3e}.","",
           "| Depth | ADOs | Max error vs exact | RMSE vs exact | Min eigenvalue | Runtime s |",
           "|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        lines.append(f"| {r['depth']} | {r['n_ados']} | {r['max_vs_exact']:.6e} | "
                     f"{r['rmse_vs_exact']:.6e} | {r['min_density_eigenvalue']:.3e} | "
                     f"{r['integrate_seconds']:.2f} |")
    lines+=["","## Adjacent hierarchy differences"]
    for pair in adjacent:
        lines.append(f"- {pair['depths']}: {pair['max_difference']:.6e}")
    lines+=["","A green job is not proof of converged dynamics; refine both hierarchy and "
             "integrator, check physicality, and compare independently.",
             "Neural training is not included in this experiment."]
    (output/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    return results

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ratio",type=float,required=True)
    p.add_argument("--depths",default="2,3,4,5")
    p.add_argument("--output",required=True)
    a=p.parse_args()
    run(a.ratio,a.output,tuple(int(v) for v in a.depths.split(",")))
