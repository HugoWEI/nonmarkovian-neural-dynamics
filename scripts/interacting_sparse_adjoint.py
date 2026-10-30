"""Interacting HEOM sensitivity experiment, with fixed correlated initial hierarchy."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from nonmarkovian_neural_dynamics.linear_adjoint import sparse_generator, adjoint_trajectory, contract_sensitivity, fixed_initial_fd

def occupation_functional(op, n_ados):
    """Trace(op * rho0) as c^dagger y, compatible with row-major ADO flatten."""
    matrix=op.full()
    dim=matrix.shape[0]
    if matrix.shape!=(dim,dim):
        raise ValueError("Observable must be square")
    # Trace(O rho) = sum_ij O_ji rho_ij, with rho in C order.
    c=np.zeros(n_ados*dim*dim, dtype=np.complex128)
    c[:dim*dim]=matrix.T.conj().reshape(-1)
    return c


def build(ratio, nk, depth, mu_mev, U_mev):
    from qutip.solver.heom import HEOMSolver, LorentzianPadeBath
    from fermionic_heom_reference import system, HBAR_MEV_PS
    H,up,down,nup,ndown=system()
    H=H + ((U_mev-1.)/HBAR_MEV_PS)*nup*ndown
    baths=[
        LorentzianPadeBath(op, .1/HBAR_MEV_PS,5./HBAR_MEV_PS,
                           mu_mev/HBAR_MEV_PS,ratio*.1/HBAR_MEV_PS,nk,
                           tag=f"{spin}_single")
        for spin,op in (("up",up),("down",down))
    ]
    solver=HEOMSolver(H,baths,max_depth=depth,options={
        "nsteps":30000,"rtol":1e-9,"atol":1e-11,
        "normalize_output":False,"progress_bar":""})
    return solver,nup


def run(ratio,nk,depth,output,T=6.,samples=81):
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    eq,op=build(ratio,nk,depth,0.,1.)
    _,ado=eq.steady_state(use_mkl=False)
    setup_s=time.perf_counter()-t0
    y0=np.array(ado._ado_state,dtype=np.complex128).reshape(-1)
    labels=list(eq.ados.labels)

    central,op=build(ratio,nk,depth,.05,1.)
    L=sparse_generator(central)
    c=occupation_functional(op,len(labels))
    # Validate the interpretation of the HEOM flattened state.
    from qutip import expect
    rho0=ado.rho
    assert abs(np.real(np.vdot(c,y0))-np.real(expect(op,rho0)))<1e-10

    t0=time.perf_counter()
    shared_trajectory=adjoint_trajectory(L,y0,c,T,samples)
    common_forward_adjoint_s=time.perf_counter()-t0
    nominal=float(np.real(np.vdot(c,shared_trajectory[1][-1])))
    rows=[]
    for name,delta in (("mu",.002),("U",.005)):
        if name=="mu":
            plus,_=build(ratio,nk,depth,.05+delta,1.)
            minus,_=build(ratio,nk,depth,.05-delta,1.)
        else:
            plus,_=build(ratio,nk,depth,.05,1.+delta)
            minus,_=build(ratio,nk,depth,.05,1.-delta)
        if list(plus.ados.labels)!=labels or list(minus.ados.labels)!=labels:
            raise AssertionError("HEOM ADO basis/order changed with parameter")
        Lp=sparse_generator(plus)
        Lm=sparse_generator(minus)
        dL=(Lp-Lm)*(1/(2*delta))
        t0=time.perf_counter()
        gradient=contract_sensitivity(dL,shared_trajectory)
        contraction_s=time.perf_counter()-t0
        adjoint_s=common_forward_adjoint_s+contraction_s
        metadata={"quadrature_samples":samples}
        t0=time.perf_counter()
        fd=fixed_initial_fd(Lp,Lm,y0,c,T,delta)
        fd_s=time.perf_counter()-t0
        absolute=float(abs(gradient-fd))
        scale=max(abs(fd),1e-3)
        row={
            "T_over_Gamma":ratio,"Nk":nk,"depth":depth,
            "n_ados":len(labels),"parameter":name,"step_mev":delta,
            "adjoint_gradient_per_mev":gradient,
            "finite_difference_per_mev":fd,
            "absolute_error_per_mev":absolute,
            "scaled_relative_error":absolute/scale,
            "terminal_occupation":nominal,
            "steady_setup_seconds":setup_s,
            "adjoint_seconds":adjoint_s,
            "shared_forward_backward_seconds":common_forward_adjoint_s,
            "marginal_contraction_seconds":contraction_s,
            "finite_difference_seconds":fd_s,
            **metadata
        }
        print(json.dumps(row),flush=True)
        rows.append(row)
        if not np.isfinite(gradient) or not np.isfinite(fd):
            raise FloatingPointError("Nonfinite HEOM sensitivity")
        if absolute>4e-4+0.02*abs(fd):
            raise AssertionError(f"Adjoint finite-difference mismatch: {row}")
    with (directory/"results.json").open("w") as f:
        json.dump(rows,f,indent=2)
    with (directory/"results.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    lines=[
        f"# Sparse interacting HEOM adjoint, T/Gamma={ratio}, Nk={nk}, depth={depth}",
        "", "**Full correlated steady initial ADOs are held fixed.**",
        "The adjoint uses the sparse HEOM generator (Krylov expm action),",
        "NOT an exact diagonalized many-body propagator. This is classical",
        "sensitivity analysis, not a new neural algorithm.",
        "", "| derivative | adjoint | central FD | abs error | adjoint CPU s | FD CPU s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for x in rows:
        lines.append(f"| {x['parameter']} | {x['adjoint_gradient_per_mev']:.7e} | "
                    f"{x['finite_difference_per_mev']:.7e} | "
                    f"{x['absolute_error_per_mev']:.2e} | "
                    f"{x['adjoint_seconds']:.3f} | {x['finite_difference_seconds']:.3f} |")
    lines+=["", "Checking sensitivities at different hierarchy depths does not",
             "prove either reference convergence or neural computational advantage.",
             "Parameter derivative of a separately re-equilibrated initial hierarchy",
             "would require a further initial-state sensitivity term."]
    (directory/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    return rows


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ratio",required=True,type=float)
    p.add_argument("--nk",type=int,default=3)
    p.add_argument("--depth",type=int,default=2)
    p.add_argument("--samples",type=int,default=81)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    run(args.ratio,args.nk,args.depth,args.output,samples=args.samples)
