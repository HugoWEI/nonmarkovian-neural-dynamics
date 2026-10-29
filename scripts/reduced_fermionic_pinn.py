"""Exploratory piecewise neural solver for a finite-bath quadratic spinless dot.

Three losses: standard residual (uniform), pointwise adjoint projection,
and integrated adjoint projection. All use identical seeds, architecture,
training steps and physical time windows. Note: a finite discrete bath is
NOT the published continuum, spinful, interacting PINN-DQME problem.
"""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.special import expit
from nonmarkovian_neural_dynamics.exact_rlm import HBAR,quadrature

def train(mode="uniform",temp=.03,seed=7,steps=1200,nwindows=4,horizon=6.,width=40):
    import torch
    if mode not in ("uniform","projected","integrated"): raise ValueError(mode)
    if steps % nwindows: raise ValueError("Steps must divide evenly between windows")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    dtype=torch.float64
    # 20 discrete fermionic bath modes: a small exactly solved learning test,
    # NOT a continuum-extrapolated physical reference.
    E,Q=quadrature(n=2,cutoff=20.)
    V=np.sqrt(.1*5.**2/(E**2+5.**2)*Q/(2*np.pi))
    H=np.diag(np.r_[-.5,E])
    H[0,1:]=V;H[1:,0]=V
    A=H/HBAR;N=len(A)
    eigen,vec=eigh(A)
    occupation_weights=np.r_[0,expit(-E/temp)]
    time_grid=np.linspace(0,horizon,201)
    exact_wave=(vec@(vec[0,:][:,None]*np.exp(1j*np.outer(eigen,time_grid)))).T
    exact_n=np.abs(exact_wave)**2@occupation_weights
    torch_A=torch.tensor(A,dtype=dtype)
    W=torch.tensor(occupation_weights,dtype=dtype)
    window_duration=horizon/nwindows
    wave_initial=np.zeros(N,dtype=complex);wave_initial[0]=1.
    predictions=[];times=[]
    started=time.perf_counter()
    for window in range(nwindows):
        initial_real=torch.tensor(wave_initial.real,dtype=dtype)
        initial_imag=torch.tensor(wave_initial.imag,dtype=dtype)
        net=torch.nn.Sequential(torch.nn.Linear(1,width),torch.nn.Tanh(),
            torch.nn.Linear(width,width),torch.nn.Tanh(),
            torch.nn.Linear(width,2*N)).to(dtype=dtype)
        optimizer=torch.optim.Adam(net.parameters(),lr=.004)
        collocation=torch.linspace(0,window_duration,41,dtype=dtype)[:,None]
        dt=1.e-3
        # Dual sensitivity of the terminal occupation: z(t)=exp(-iA(T-t))*F*w_pred(T).
        propagators=np.einsum("jm,tm,km->tjk",vec,
            np.exp(-1j*np.outer(window_duration-collocation.numpy().ravel(),eigen)),
            vec.conj())
        dual_matrices=torch.tensor(propagators,dtype=torch.complex128)
        def evaluate(t):
            d=(t*net(t/window_duration)).reshape(-1,2,N)
            return initial_real[None,:]+d[:,0],initial_imag[None,:]+d[:,1]
        for _ in range(steps//nwindows):
            optimizer.zero_grad(set_to_none=True)
            re,im=evaluate(collocation)
            rp,ip=evaluate(collocation+dt)
            rm,imn=evaluate(collocation-dt)
            # Wave obeys w'=+i A w; initial dot amplitude is 1.
            resid_re=(rp-rm)/(2*dt)+im@torch_A.T
            resid_im=(ip-imn)/(2*dt)-re@torch_A.T
            ordinary=(resid_re.square()+resid_im.square()).mean()
            if mode=="uniform":
                loss=ordinary
            else:
                with torch.no_grad():
                    rT,iT=evaluate(torch.tensor([[window_duration]],dtype=dtype))
                    grad=W*(rT[0]+1j*iT[0])
                    adjoint=dual_matrices@grad
                projection=torch.real((adjoint.conj()*(resid_re+1j*resid_im)).sum(1))
                denom=torch.clamp(adjoint.abs().square().sum(1).mean(),min=1e-10)
                if mode=="projected":
                    auxiliary=projection.square().mean()/denom/N
                else:
                    auxiliary=projection.mean().square()/denom/N
                loss=.5*ordinary+.5*auxiliary
            loss.backward()
            optimizer.step()
        local_times=np.linspace(0,window_duration,51)
        with torch.no_grad():
            re,im=evaluate(torch.tensor(local_times[:,None],dtype=dtype))
            vectors=re.numpy()+1j*im.numpy()
        if window:
            local_times=local_times[1:];vectors=vectors[1:]
        times.extend(window*window_duration+local_times)
        predictions.extend(vectors)
        wave_initial=vectors[-1]
    elapsed=time.perf_counter()-started
    times=np.asarray(times); wave=np.asarray(predictions)
    estimated=np.abs(wave)**2@occupation_weights
    truth=np.interp(times,time_grid,exact_n)
    return {"mode":mode,"T_mev":temp,"seed":seed,"fermionic_modes":N,
            "total_steps":steps,"windows":nwindows,
            "occupation_rmse":float(np.sqrt(np.mean((estimated-truth)**2))),
            "terminal_occupation_abs_error":float(abs(estimated[-1]-exact_n[-1])),
            "max_wave_norm_drift":float(max(abs(np.abs(wave).square().sum(1)-1))),
            "runtime_seconds":elapsed}

def run(out,steps=1200):
    dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for temperature in (.03,.3):
        for seed in (1,2,7):
            for mode in ("uniform","projected","integrated"):
                result=train(mode,temperature,seed,steps=steps)
                rows.append(result)
                print(json.dumps(result),flush=True)
    with (dest/"results.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (dest/"results.json").write_text(json.dumps(rows,indent=2))
    lines=["# Finite-bath spinless PINN comparison","",
        "**Not the original PINN-DQME or a continuum-converged neural calculation.**",
        "20 discrete bath modes, four windows, same width, seeds and optimization steps.",
        "Dual coefficients estimated from the current neural terminal wavefunction,",
        "not from hidden exact target trajectories.","",
        "| T (meV) | seed | loss | occupation RMSE | terminal error | norm drift | CPU secs |",
        "|---:|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['T_mev']} | {r['seed']} | {r['mode']} | "
                     f"{r['occupation_rmse']:.5e} | {r['terminal_occupation_abs_error']:.5e} | "
                     f"{r['max_wave_norm_drift']:.5e} | {r['runtime_seconds']:.2f} |")
    lines += ["","Do NOT infer a new method's superiority from trajectory RMSE alone.",
              "Report goal-observable errors, train+evaluation time, and a non-neural baseline.",
              "All three variants remain dramatically more expensive than the exact 21x21 solver.",
              "A sign-aware dual-integral penalty is not a certified error bound when",
              "its adjoint is computed from an approximate neural terminal state."]
    (dest/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    return rows

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps",type=int,default=1200)
    ap.add_argument("--output",default="outputs/reduced_pinn")
    args=ap.parse_args()
    run(args.output,args.steps)
