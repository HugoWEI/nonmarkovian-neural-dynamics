"""Controlled interacting HEOM hierarchy error-backflow decomposition.

Low depth starts from *projection of high-depth correlated steady ADOs*,
NOT its separately relaxed steady state. This makes the truncation error
mathematically well defined, with a potentially nonzero omitted-tier
initial condition and a high-tier residual.

High-depth adjoint and trajectory are used ONLY for validation;
POD-rank approximations measure whether they could be cached/reused.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.integrate import simpson

from interacting_sparse_adjoint import build,occupation_functional
from nonmarkovian_neural_dynamics.linear_adjoint import (
    sparse_generator,adjoint_trajectory
)
from nonmarkovian_neural_dynamics.hierarchy_defect import (
    hierarchy_retained_indices,compute_defect,linear_observable_defect
)
from nonmarkovian_neural_dynamics.dual_compression import CompressedDual


def run(ratio,nk,depth_hi,output,samples=81):
    if depth_hi<2: raise ValueError("Need at least depth2")
    depth_lo=depth_hi-1
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    eq,op=build(ratio,nk,depth_hi,0.,1.)
    _,ado=eq.steady_state(use_mkl=False)
    initial_s=time.perf_counter()-t0
    y0hi=np.asarray(ado._ado_state,dtype=complex).reshape(-1)
    hi,op=build(ratio,nk,depth_hi,.05,1.)
    lo,_=build(ratio,nk,depth_lo,.05,1.)
    Ahi=sparse_generator(hi)
    Alo=sparse_generator(lo)
    indices=hierarchy_retained_indices(lo.ados.labels,hi.ados.labels,16)
    c=occupation_functional(op,len(hi.ados.labels))
    t0=time.perf_counter()
    times,full,dual=adjoint_trajectory(Ahi,y0hi,c,T=6.,samples=samples)
    full_s=time.perf_counter()-t0
    t0=time.perf_counter()
    lowembedded,r,e0=compute_defect(Alo,Ahi,y0hi,indices,times)
    low_s=time.perf_counter()-t0
    true=float(np.real(np.vdot(c,lowembedded[-1]-full[-1])))
    initial,residual,full_est=linear_observable_defect(dual,r,e0,times)
    quadrature_error=abs(true-full_est)
    # This tolerance tests the numerical quadrature, not HEOM convergence.
    if quadrature_error>2e-4:
        raise AssertionError(f"Full dual defect not matched: {true} vs {full_est}")

    t0=time.perf_counter()
    pod=CompressedDual(times,dual)
    pod_s=time.perf_counter()-t0
    compressed=[]
    for rank in (2,4,8,16,32):
        t0=time.perf_counter()
        con=pod.contract(r,rank)
        basis,coeff,_=pod.factors(rank)
        z0_approx=(basis[0]@coeff)/pod.sqrtw[0]
        init_comp=float(np.real(np.vdot(z0_approx,e0)))
        init_bound=float(np.linalg.norm(dual[0]-z0_approx)*np.linalg.norm(e0))
        estimate=init_comp+con["estimated_gradient"]
        certified=init_bound+con["certified_discrete_error_bound"]
        row={
            "rank":rank,
            "compressed_full_defect_estimate":estimate,
            "error_vs_full_discrete_estimate":abs(estimate-full_est),
            "error_vs_true_terminal_difference":abs(estimate-true),
            "certified_discrete_compression_bound":certified,
            "initial_contribution_compressed":init_comp,
            "residual_contribution_compressed":con["estimated_gradient"],
            "dual_relative_L2_error":con["relative_dual_L2_error"],
            "full_dual_complex_entries":con["full_complex_entries"],
            "compressed_dual_complex_entries":con["compressed_complex_entries"],
            "contract_seconds":time.perf_counter()-t0,
        }
        if row["error_vs_full_discrete_estimate"]>certified+2e-8:
            raise AssertionError("Combined initial+residual POD bound failure")
        compressed.append(row)
        print(json.dumps({"rank_result":row}),flush=True)
    omitted=np.ones(Ahi.shape[0],dtype=bool)
    omitted[indices]=False
    omitted_residual_fraction=float(
        np.linalg.norm(r[:,omitted])/max(np.linalg.norm(r),1e-30))
    report={
        "T_over_Gamma":ratio,"Nk":nk,"depth_lo":depth_lo,"depth_hi":depth_hi,
        "n_ados_low":len(lo.ados.labels),"n_ados_high":len(hi.ados.labels),
        "full_steady_state_seconds":initial_s,
        "full_dual_seconds":full_s,
        "low_forward_and_defect_seconds":low_s,
        "pod_fit_seconds":pod_s,
        "true_terminal_low_minus_high":true,
        "initial_mismatch_contribution":initial,
        "omitted_tier_residual_contribution":residual,
        "full_defect_prediction":full_est,
        "full_defect_quadrature_error":quadrature_error,
        "fraction_residual_norm_on_omitted_indices":omitted_residual_fraction,
        "comparison_initial":"project same high-depth full correlated ADOs to low depth",
        "model":"one-lead interacting SIAM with post-steady-state mu quench",
        "compressed":compressed,
    }
    (out/"results.json").write_text(json.dumps(report,indent=2))
    lines=[
        f"# Source-inspired interacting HEOM hierarchy error decomposition, T/Gamma={ratio}, Nk={nk}",
        "",
        "**The low-depth initial ADOs are projected from the SAME high-depth steady state.**",
        "Thus this is a valid nested-hierarchy truncation experiment; it is not",
        "a comparison of separately prepared steady states.",
        "", f"Depths {depth_lo}->{depth_hi}; low/high ADOs {len(lo.ados.labels)}/{len(hi.ados.labels)}.",
        f"True terminal low-minus-high occupation = {true:.8e}.",
        f"Initial omitted-tier error contribution = {initial:.8e}.",
        f"Dynamic omitted-tier residual contribution = {residual:.8e}.",
        f"Sum from high-depth adjoint = {full_est:.8e};",
        f"numerical Simpson residual = {quadrature_error:.3e}.",
        f"Residual support fraction in omitted tiers = {omitted_residual_fraction:.4f}.",
        "", "| POD rank | terminal defect estimate | error vs true terminal defect | bound vs full discrete defect |",
        "|---:|---:|---:|---:|",
    ]
    for row in compressed:
        lines.append(f"| {row['rank']} | {row['compressed_full_defect_estimate']:.7e} | "
                     f"{row['error_vs_true_terminal_difference']:.2e} | "
                     f"{row['certified_discrete_compression_bound']:.2e} |")
    lines+=["","**Bound scope:** only the compression error for sampled finite HEOM",
            "adjoint quadrature plus initial omitted-tier term. Does not bound",
            "HEOM truncation relative to physical continuum or quadrature error.",
            "No neural solver has been trained or accelerated by this pilot."]
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="compressed"}),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ratio",type=float,required=True)
    p.add_argument("--nk",type=int,required=True)
    p.add_argument("--high-depth",type=int,default=3)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    run(a.ratio,a.nk,a.high_depth,a.output)
