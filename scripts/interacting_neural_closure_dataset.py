"""Interacting fermionic HEOM dataset for amortized higher-order memory learning.

Source-like low-T Nk7, interacting U>0, depth2 correlated HEOM steady ADOs
at mu=0,U=1, embedded with zero omitted top-tier ADOs at depth3.
The initial state is HELD FIXED for every sudden U/mu parameter quench.
NOT fully re-equilibrated high-depth state, not Cao et al.'s two-lead state.

For each parameter pair store low-depth O(t), 1st Born/Volterra corrected O(t),
and withheld depth3 O(t). Never use the withheld target to construct the
first-order physics features.
"""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import expm_multiply
from interacting_sparse_adjoint import build,occupation_functional
from nonmarkovian_neural_dynamics.linear_adjoint import sparse_generator
from nonmarkovian_neural_dynamics.hierarchy_defect import hierarchy_retained_indices
from nonmarkovian_neural_dynamics.memory_feedback import partition_nested


def forcing_solution(generator,forcing,grid):
    grid=np.asarray(grid)
    def rhs(t,y):
        i=int(np.clip(np.searchsorted(grid,t,side="right")-1,0,len(grid)-2))
        a=(t-grid[i])/(grid[i+1]-grid[i])
        return generator@y+(1-a)*forcing[i]+a*forcing[i+1]
    sol=solve_ivp(rhs,(float(grid[0]),float(grid[-1])),
                  np.zeros(generator.shape[0],dtype=complex),
                  t_eval=grid,rtol=1e-7,atol=1e-9,method="RK45")
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def evaluate_case(mu,U,y0,grid,nk=7):
    ratio=.3
    t0=time.perf_counter()
    low,op=build(ratio,nk,2,mu,U)
    high,_=build(ratio,nk,3,mu,U)
    Alo=sparse_generator(low);Ahi=sparse_generator(high)
    idx=hierarchy_retained_indices(low.ados.labels,high.ados.labels,16)
    A,B,C,D,missing=partition_nested(Ahi,Alo,idx)
    build_s=time.perf_counter()-t0
    c_low=occupation_functional(op,len(low.ados.labels))
    c_hi=occupation_functional(op,len(high.ados.labels))

    t0=time.perf_counter()
    x=expm_multiply(Alo,y0,start=0,stop=grid[-1],num=len(grid),
                    traceA=np.sum(Alo.diagonal()))
    nlo=np.real(x@c_low.conj())
    top=forcing_solution(D,(C@x.T).T,grid)
    dx=forcing_solution(A,(B@top.T).T,grid)
    first=nlo+np.real(dx@c_low.conj())
    first_s=time.perf_counter()-t0

    initial_hi=np.zeros(Ahi.shape[0],dtype=complex)
    initial_hi[idx]=y0
    t0=time.perf_counter()
    exact_hi=expm_multiply(Ahi,initial_hi,start=0,stop=grid[-1],num=len(grid),
                           traceA=np.sum(Ahi.diagonal()))
    target=np.real(exact_hi@c_hi.conj())
    high_s=time.perf_counter()-t0
    return nlo,first,target,{"mu":mu,"U":U,"build_seconds":build_s,
        "first_order_seconds":first_s,"high_depth_seconds":high_s,
        "terminal_uncorrected_abs_error":float(abs(nlo[-1]-target[-1])),
        "terminal_first_order_abs_error":float(abs(first[-1]-target[-1]))}


def generate(out="outputs/neural_memory_dataset"):
    path=Path(out);path.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    eq,_=build(.3,7,2,0.,1.)
    _,ado=eq.steady_state(use_mkl=False)
    initial=np.asarray(ado._ado_state,dtype=complex).reshape(-1)
    setup_seconds=time.perf_counter()-t0
    U_train=(.8,1.,1.2)
    mu_train=(.025,.075,.125,.2)
    specs=[(u,mu,"train") for u in U_train for mu in mu_train]
    specs += [(u,mu,"heldout") for u in (.9,1.1) for mu in (.05,.15)]
    specs += [(1.3,.22,"extrapolate"),(.7,.015,"extrapolate")]
    t=np.linspace(0,12,161)
    low=[];first=[];high=[];parameters=[];split=[];records=[]
    for u,mu,group in specs:
        a,b,c,meta=evaluate_case(mu,u,initial,t)
        low.append(a);first.append(b);high.append(c)
        parameters.append([u,mu]);split.append(group)
        records.append({**meta,"split":group})
        print(json.dumps(records[-1]),flush=True)
    np.savez_compressed(path/"trajectories.npz",
        time=t,params=np.array(parameters),split=np.array(split),
        low=np.array(low),first=np.array(first),high=np.array(high))
    doc={"physics":"low-T source-Pade Nk7 interacting one-lead Anderson",
         "initial_state":"same depth2 correlated equilibrium projected to depth3; higher tiers zero",
         "horizon_ps":12,"n_time":len(t),"n_train":split.count("train"),
         "n_heldout":split.count("heldout"),
         "n_extrapolate":split.count("extrapolate"),
         "depth2_correlated_setup_seconds":setup_seconds,
         "records":records}
    (path/"metadata.json").write_text(json.dumps(doc,indent=2))
    print(json.dumps({"dataset_summary":{
        "setup_seconds":setup_seconds,
        "n_train":doc["n_train"],"n_test":doc["n_heldout"]+doc["n_extrapolate"],
        "generation_seconds":sum(x["build_seconds"]+x["first_order_seconds"]+
                                  x["high_depth_seconds"] for x in records)}}),flush=True)
    return path

if __name__=="__main__":
    generate()
