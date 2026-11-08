"""Benchmark *non-neural* exact RLM propagation versus neural CPU timing.

On same GitHub Actions hosted CPU as the other runs. The direct 21x21
eigensolution already provides all 201 times, so adjoint construction is
not a computational advantage in this small model.
"""
import json,statistics,time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.special import expit
from nonmarkovian_neural_dynamics.exact_rlm import quadrature,HBAR

def solve_one():
    energies,weights=quadrature(n=2,cutoff=20.)
    V=np.sqrt(.1*5.**2/(energies**2+25)*weights/(2*np.pi))
    H=np.diag(np.r_[-.5,energies])
    H[0,1:]=V;H[1:,0]=V
    eigen,vec=eigh(H)
    t=np.linspace(0,6,201)
    wave=(vec@(vec[0,:][:,None]*np.exp(1j*np.outer(eigen,t)/HBAR))).T
    occ=np.abs(wave)**2@np.r_[0,expit(-energies/.03)]
    return occ

def run(directory="outputs/exact_cost"):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    for _ in range(20): solve_one()
    repeats=[]
    for _ in range(200):
        t0=time.perf_counter()
        data=solve_one()
        repeats.append(time.perf_counter()-t0)
    r={"n_iterations":len(repeats),
       "median_direct_seconds":statistics.median(repeats),
       "mean_direct_seconds":statistics.mean(repeats),
       "p90_direct_seconds":float(np.percentile(repeats,90)),
       "terminal_occupation":float(data[-1]),
       "work":"construct H, diagonalize, propagate 201 times, occupation"}
    (out/"results.json").write_text(json.dumps(r,indent=2))
    (out/"SUMMARY.md").write_text(
        "# Small quadratic physical reference cost\n\n"+
        f"200 independent full-solution repeats, median {r['median_direct_seconds']:.5g} s.\n\n"+
        "Direct diagonalization is an unusually favourable baseline for this toy; "+
        "a proof-of-concept neural loss cannot claim cost benefit here.\n")
    print(json.dumps(r),flush=True)
    return r
if __name__=="__main__":
    run()
