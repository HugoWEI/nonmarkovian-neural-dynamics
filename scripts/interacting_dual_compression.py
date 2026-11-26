"""Reusable approximate HEOM dual via quadrature-weighted POD compression.

Evaluate controlled rank truncations of an already computed adjoint, without
ever constructing an exact many-body propagator. The bound applies only
to numerical quadrature contractions, not reference physics error.
"""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np

from nonmarkovian_neural_dynamics.linear_adjoint import (
    sparse_generator,adjoint_trajectory
)
from nonmarkovian_neural_dynamics.dual_compression import CompressedDual
from interacting_sparse_adjoint import build,occupation_functional


def run(ratio,nk,depth,output,rank_list=(2,4,8,16,32),samples=81):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    eq,op=build(ratio,nk,depth,0.,1.)
    _,ado=eq.steady_state(use_mkl=False)
    steady_s=time.perf_counter()-t0
    y0=np.asarray(ado._ado_state,dtype=np.complex128).reshape(-1)
    labels=list(eq.ados.labels)
    central,op=build(ratio,nk,depth,.05,1.)
    L=sparse_generator(central)
    c=occupation_functional(op,len(labels))
    t0=time.perf_counter()
    grid,forward,dual=adjoint_trajectory(L,y0,c,6.,samples)
    adjoint_s=time.perf_counter()-t0
    t0=time.perf_counter()
    pod=CompressedDual(grid,dual)
    pod_fit_s=time.perf_counter()-t0
    rows=[]
    for p,step in (("mu",.002),("U",.005)):
        if p=="mu":
            plus,_=build(ratio,nk,depth,.05+step,1.)
            minus,_=build(ratio,nk,depth,.05-step,1.)
        else:
            plus,_=build(ratio,nk,depth,.05,1.+step)
            minus,_=build(ratio,nk,depth,.05,1.-step)
        if list(plus.ados.labels)!=labels or list(minus.ados.labels)!=labels:
            raise ValueError("Parameter perturbation changed ADO basis ordering")
        dL=(sparse_generator(plus)-sparse_generator(minus))*(.5/step)
        source=(dL@forward.T).T
        for rank in rank_list:
            t0=time.perf_counter()
            value=pod.contract(source,rank)
            contract_seconds=time.perf_counter()-t0
            rows.append({
                "T_over_Gamma":ratio,"Nk":nk,"depth":depth,"n_ados":len(labels),
                "parameter":p,"steady_state_seconds":steady_s,
                "full_dual_seconds":adjoint_s,
                "pod_fit_seconds":pod_fit_s,
                "compressed_contract_seconds":contract_seconds,
                **value,
            })
            print(json.dumps(rows[-1]),flush=True)
    (out/"results.json").write_text(json.dumps(rows,indent=2))
    with (out/"results.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    lines=[
        f"# POD-compressed HEOM adjoint, T/Gamma={ratio}, Nk={nk}, depth={depth}",
        "",
        "This is established projection-based model reduction of a",
        "**precomputed full HEOM adjoint**. No neural speedup or novelty claim.",
        "The Cauchy-Schwarz bound certifies only the *discretized*",
        "Simpson quadrature contraction at this finite hierarchy/Pade order.",
        "",f"Full dual setup {adjoint_s:.3f}s; compression fit {pod_fit_s:.3f}s.",
        "", "| parameter | rank | full gradient | compressed gradient | observed error | C-S bound | dual relative L2 error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['parameter']} | {r['rank']} | {r['full_discrete_gradient']:.6e} | "
                     f"{r['estimated_gradient']:.6e} | {r['gradient_absolute_error']:.3e} | "
                     f"{r['certified_discrete_error_bound']:.3e} | "
                     f"{r['relative_dual_L2_error']:.3e} |")
    lines+=["",
        "Good rank compression could permit amortized reuse over many",
        "training residual evaluations at a *fixed* Hamiltonian, observable,",
        "and terminal time. The one-time full adjoint and POD fit remain costly.",
        "The bound does not certify errors from omitted HEOM tiers, Padé",
        "truncation, collocation/time quadrature, or changing initial states."]
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    return rows


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ratio",required=True,type=float)
    p.add_argument("--nk",required=True,type=int)
    p.add_argument("--depth",required=True,type=int)
    p.add_argument("--samples",default=81,type=int)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    run(a.ratio,a.nk,a.depth,a.output,samples=a.samples)
