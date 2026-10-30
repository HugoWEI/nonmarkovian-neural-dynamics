"""First-order top-tier memory feedback: cheap *predictive* interacting test.

Unlike full-high-depth dual error reconstruction, this predictor uses:
  (i) low-depth unbiased correlated ADO steady state;
 (ii) a low-depth sparse forward + adjoint;
(iii) a first-order forced response of the omitted top tier.
It does NOT use a high-depth correlated steady state or high-depth adjoint.

The higher-depth forward evolution from the same embedded initial condition
is run strictly for validation. This is an established one-iteration
Schur/Volterra memory approximation, NOT a new neural solver.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import expm_multiply

from interacting_sparse_adjoint import build,occupation_functional
from nonmarkovian_neural_dynamics.linear_adjoint import sparse_generator
from nonmarkovian_neural_dynamics.hierarchy_defect import hierarchy_retained_indices
from nonmarkovian_neural_dynamics.memory_feedback import (
    partition_nested,first_order_feedback
)


def run(ratio,nk,depth_hi,output,mu_mev=.05):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    depth_lo=depth_hi-1
    t0=time.perf_counter()
    equilibrium,op=build(ratio,nk,depth_lo,0.,1.)
    _,ados=equilibrium.steady_state(use_mkl=False)
    preparation_s=time.perf_counter()-t0
    y0lo=np.asarray(ados._ado_state,dtype=complex).reshape(-1)
    t0=time.perf_counter()
    low,op=build(ratio,nk,depth_lo,mu_mev,1.)
    Alo=sparse_generator(low)
    low_generator_s=time.perf_counter()-t0
    t0=time.perf_counter()
    high,_=build(ratio,nk,depth_hi,mu_mev,1.)
    Ahi=sparse_generator(high)
    high_generator_s=time.perf_counter()-t0
    indices=hierarchy_retained_indices(low.ados.labels,high.ados.labels,16)
    A,B,C,D,omitted=partition_nested(Ahi,Alo,indices)
    c_low=occupation_functional(op,len(low.ados.labels))
    c_hi=occupation_functional(op,len(high.ados.labels))
    t=np.linspace(0,6,81)
    t0=time.perf_counter()
    predicted,diagnostics=first_order_feedback(A,B,C,D,y0lo,c_low,t)
    predictor_s=time.perf_counter()-t0
    baseline=float(np.real(np.vdot(c_low,diagnostics["low_trajectory"][-1])))
    correction=predicted
    # Independent higher-depth validation from *exactly the same* embedded
    # initial state, with omitted-tier ADOs explicitly zero.
    y0hi=np.zeros(Ahi.shape[0],dtype=complex)
    y0hi[indices]=y0lo
    t0=time.perf_counter()
    yhi=expm_multiply(Ahi,y0hi,start=0.,stop=6.,num=81,
                      traceA=np.sum(Ahi.diagonal()),endpoint=True)
    high_s=time.perf_counter()-t0
    reference=float(np.real(np.vdot(c_hi,yhi[-1])))
    true_correction=reference-baseline
    predicted_total=baseline+correction
    baseline_error=abs(reference-baseline)
    corrected_error=abs(reference-predicted_total)
    report={
      "ratio":ratio,"Nk":nk,"low_depth":depth_lo,"high_depth":depth_hi,
      "post_quench_mu_mev":mu_mev,
      "low_generator_build_seconds":low_generator_s,
      "high_generator_build_seconds":high_generator_s,
      "n_ados_lo":len(low.ados.labels),"n_ados_hi":len(high.ados.labels),
      "lower_correlated_state_preparation_seconds":preparation_s,
      "memory_predictor_seconds":predictor_s,
      "independent_high_depth_forward_seconds":high_s,
      "predictor_total_seconds_no_exact_high_forward":(
          preparation_s+low_generator_s+high_generator_s+predictor_s),
      "direct_high_total_seconds_same_initial":(
          preparation_s+low_generator_s+high_generator_s+high_s),
      "baseline_low_depth_terminal_occupation":baseline,
      "independent_high_depth_terminal_occupation":reference,
      "true_high_minus_low_correction":true_correction,
      "first_order_memory_correction":correction,
      "uncorrected_abs_error":baseline_error,
      "corrected_abs_error":corrected_error,
      "error_reduction_fraction":(
           1.-corrected_error/max(baseline_error,1e-15)),
      "top_tier_initial_state":"zero on omitted tiers; low tier is correlated equilibrium at mu=0",
      "high_depth_steady_state_used":False,
      "high_depth_adjoint_used":False,
      "convergence_certified":False,
    }
    if not np.isfinite(list(v for v in report.values() if isinstance(v,(int,float)))).all():
        raise ValueError("Nonfinite predictions")
    (out/"results.json").write_text(json.dumps(report,indent=2))
    lines=[
        f"# First-order omitted-tier memory feedback, T/Gamma={ratio}, Nk={nk}",
        "", "This predictor uses no high-depth adjoint and no high-depth steady-state preparation.",
        "For a controlled comparison both truncations start from the **same**",
        "embedded depth-low correlated ADO state; omitted top tiers start at zero.",
        "It is **not** a full depth-high steady state or a reproduction of Cao et al.",
        "", "| Diagnostic | Value |", "|---|---:|",
    ]
    for key in ("baseline_low_depth_terminal_occupation",
                "independent_high_depth_terminal_occupation",
                "true_high_minus_low_correction","first_order_memory_correction",
                "uncorrected_abs_error","corrected_abs_error",
                "error_reduction_fraction","memory_predictor_seconds",
                "independent_high_depth_forward_seconds",
                "low_generator_build_seconds","high_generator_build_seconds",
                "predictor_total_seconds_no_exact_high_forward",
                "direct_high_total_seconds_same_initial"):
        lines.append(f"| {key} | {report[key]:.8g} |")
    lines+=["",
        "A positive error reduction would validate a first-order block-memory",
        "correction on this problem, NOT a novel algorithm or a speedup at scale.",
        "The top-tier initial ADOs were deliberately set to zero and the",
        "higher-depth solver was run only as a withheld reference.",
        "This does not certify deeper hierarchy or Padé convergence."]
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(report),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ratio",type=float,required=True)
    p.add_argument("--nk",type=int,required=True)
    p.add_argument("--high-depth",type=int,default=3)
    p.add_argument("--output",required=True)
    p.add_argument("--mu",type=float,default=.05)
    a=p.parse_args()
    run(a.ratio,a.nk,a.high_depth,a.output,mu_mev=a.mu)
