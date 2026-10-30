"""Reduced-channel but INTERACTING fermionic HEOM, correlated ADO quench.

Spinful Anderson impurity with U=1 meV, one thermodynamic Lorentzian
lead per spin (instead of two). Prepare full equilibrium ADO hierarchy,
then quench lead mu from 0 to +0.05 meV. Depths 2 and 3 at Nk=3.

This is independent *source-inspired* U>0 non-Markovian physics, NOT
a reproduction of the original authors' two-lead correlated SIAM state.
"""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from qutip.solver.heom import HEOMSolver,LorentzianPadeBath
from fermionic_heom_reference import system,HBAR_MEV_PS

def build_solver(ratio,nk,depth,mu=0.):
    h,up,down,nup,ndown=system()
    baths=[LorentzianPadeBath(op,.1/HBAR_MEV_PS,5./HBAR_MEV_PS,
            mu/HBAR_MEV_PS,ratio*.1/HBAR_MEV_PS,nk,tag=f"{spin}_lead")
            for spin,op in (("up",up),("down",down))]
    solver=HEOMSolver(h,baths,max_depth=depth,options={
         "nsteps":30000,"rtol":1e-8,"atol":1e-10,
         "store_states":True,"progress_bar":""})
    return solver,nup,ndown

def physicality(states):
    err=max(abs(complex(r.tr())-1.) for r in states)
    smallest=min(float(np.min(np.linalg.eigvalsh(
        (r.full()+r.full().conj().T)/2).real)) for r in states)
    herm=max(float(np.max(abs(r.full()-r.full().conj().T))) for r in states)
    return {"trace_error":float(err),
            "minimum_eigenvalue":smallest,"hermiticity_error":herm}

def solve(ratio,nk,depth,duration=6.,n_times=121):
    t=np.linspace(0,duration,n_times)
    t0=time.perf_counter()
    equilibrium,nup,ndown=build_solver(ratio,nk,depth,mu=0.)
    rho,ados=equilibrium.steady_state(use_mkl=False)
    prepare_seconds=time.perf_counter()-t0
    reference=equilibrium.run(ados,t,e_ops={"up":nup,"down":ndown})
    biased,nu,nd=build_solver(ratio,nk,depth,mu=.05)
    if list(equilibrium.ados.labels)!=list(biased.ados.labels):
        raise RuntimeError("Correlated ADO label mismatch after quench")
    t0=time.perf_counter()
    quench=biased.run(ados,t,e_ops={"up":nu,"down":nd})
    evolution_seconds=time.perf_counter()-t0
    eqpop=np.stack([np.real(reference.e_data["up"]),
                    np.real(reference.e_data["down"])],axis=1)
    pop=np.stack([np.real(quench.e_data["up"]),
                  np.real(quench.e_data["down"])],axis=1)
    drift=float(max(abs(eqpop-eqpop[0]).ravel()))
    if drift>1e-5: raise AssertionError(f"Pre-quench ADO stationarity failed: {drift}")
    if not np.isfinite(pop).all():raise FloatingPointError("Nonfinite quench occupations")
    return {
        "ratio":ratio,"Nk":nk,"depth":depth,"n_ados":len(equilibrium.ados.labels),
        "prepare_seconds":prepare_seconds,
        "evolution_seconds":evolution_seconds,
        "stationary_occupation_drift":drift,
        "initial_occupation_up":float(eqpop[0,0]),
        "final_occupation_up":float(pop[-1,0]),
        "max_quench_response":float(np.max(abs(pop-pop[0]))),
        "density_physicality":physicality(quench.states),
        "times":t,"quench_occupation":pop
    }

def run(ratio,output,nk=3,depths=(2,3)):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    results=[];curves={}
    for depth in depths:
        val=solve(ratio,nk,depth)
        curves[depth]=val["quench_occupation"]
        results.append({k:v for k,v in val.items() if k not in ("times","quench_occupation")})
        print(json.dumps(results[-1]),flush=True)
    t=val["times"]
    with (out/"trajectories.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["time_ps"]+[f"n_up_d{d}" for d in depths]+[f"n_down_d{d}" for d in depths])
        writer.writerows([[t[i]]+[curves[d][i,0] for d in depths]+[curves[d][i,1] for d in depths]
                          for i in range(len(t))])
    comparison={"max_occupation_delta":float(np.max(abs(curves[depths[0]]-curves[depths[1]]))),
                "rms_occupation_delta":float(np.sqrt(np.mean(
                    (curves[depths[0]]-curves[depths[1]])**2)))}
    report={"T_over_Gamma":ratio,"Nk":nk,
            "depths":list(depths),"runs":results,"depth_comparison":comparison,
            "initial":"full HEOM equilibrium ADOs at mu=0",
            "quench":"lead chemical potential 0 to 0.05 meV",
            "model":"spinful U=1 meV, one Lorentzian lead per spin",
            "convergence_certified":False}
    (out/"results.json").write_text(json.dumps(report,indent=2))
    summary=[
        f"# Interacting correlated HEOM, T/Gamma={ratio}", "",
        "Spinful U=1 meV Anderson model with ONE lead/spin, full correlated steady ADO initial state.",
        "**Not the paper's two-lead correlated SIAM benchmark.**",
        "Low-temperature Nk=3 is below source Nk=7, so Padé-order convergence is unknown.",
        "", "| Depth | ADOs | initial n_up | terminal n_up | minimum density eigenvalue | prep sec |",
        "|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        summary.append(f"| {r['depth']} | {r['n_ados']} | {r['initial_occupation_up']:.7f} | "
            f"{r['final_occupation_up']:.7f} | {r['density_physicality']['minimum_eigenvalue']:.4e} | "
            f"{r['prepare_seconds']:.2f} |")
    summary+=["",f"Full-trajectory depth discrepancy: {comparison['max_occupation_delta']:.6e}.",
              "Converge depth AND Padé order independently before treating this as ML ground truth."]
    (out/"SUMMARY.md").write_text("\n".join(summary)+"\n")
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ratio",required=True,type=float)
    p.add_argument("--nk",default=3,type=int)
    p.add_argument("--depths",default="2,3")
    p.add_argument("--output",required=True)
    a=p.parse_args()
    run(a.ratio,a.output,a.nk,tuple(int(x) for x in a.depths.split(",")))
