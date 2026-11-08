"""Independent Padé coefficient reconstruction from pinned Cao et al. data.

Rebuild the Lorentzian fermionic bath expansion (QuTiP guide convention), then
compare every term to the original T/Gamma=3 and 0.3 res_corr files.
No full HEOM/PINN claim. Chemical-potential phase handled separately later.
"""
import json
from pathlib import Path
from urllib.request import urlopen
import numpy as np
from scipy.linalg import eigvalsh

HBAR = 0.658211928
REV = "7fb6ffb6115db42b299cc5c5092b27bfa9050628"
ROOT = f"https://raw.githubusercontent.com/caolong-cn/NQS-DQME_fermions/{REV}/examples"
CASES = {"high": ("pinn_case1_T=3.0/res_corr.data", .3, 3),
         "low": ("case1_T=0.3/res_corr.data", .03, 7)}


def poles_weights(n):
    a = np.zeros((2*n, 2*n))
    for i in range(2*n-1):
        x = 1/np.sqrt((2*i+1)*(2*i+3))
        a[i,i+1] = a[i+1,i] = x
    eps = -2/eigvalsh(a)[:n]
    ap = np.zeros((2*n-1, 2*n-1))
    for i in range(2*n-2):
        x = 1/np.sqrt((2*i+3)*(2*i+5))
        ap[i,i+1] = ap[i+1,i] = x
    chi = -2/eigvalsh(ap)[:n-1]
    kappa = np.array([
        .5*n*(2*n+1)*np.prod(chi**2-eps[j]**2) /
        np.prod([eps[k]**2-eps[j]**2 if k!=j else 1. for k in range(n)])
        for j in range(n)
    ])
    return eps, kappa


def reconstruct(T_mev, nk, gamma_mev=.1, W_mev=5.):
    eps, kappa = poles_weights(nk)
    T, gamma, W = T_mev/HBAR, gamma_mev/HBAR, W_mev/HBAR
    beta = 1/T
    z = 1j*beta*W
    f = .5 - sum(2*k*z/(z*z+e*e) for k,e in zip(kappa,eps))
    eta = [.5*gamma*W*f]
    rates = [W]
    for e,k in zip(eps,kappa):
        eta.append(-1j*(k/beta)*gamma*W*W/(W*W-e*e/(beta*beta)))
        rates.append(e/beta)
    return np.array(eta), np.array(rates)


def source_coefficients(raw):
    lines = [row.split() for row in raw.splitlines() if row.strip()]
    n = int(lines[4][0])
    block = [
        (int(x[5]), complex(float(x[6]),float(x[7])), -float(x[10]))
        for x in lines[5:]
        if tuple(map(int,x[1:5])) == (2,1,1,1)
    ]
    block.sort()
    if len(block) != n:
        raise ValueError("Missing source coefficient terms")
    return np.array([b[1] for b in block]), np.array([b[2] for b in block])


def run(out="outputs/pade_source_match"):
    directory = Path(out)
    directory.mkdir(parents=True,exist_ok=True)
    results = {}
    for case,(filename,T_mev,nk) in CASES.items():
        with urlopen(f"{ROOT}/{filename}",timeout=30) as response:
            raw=response.read().decode("utf-8")
        e,v=source_coefficients(raw)
        er,vr=reconstruct(T_mev,nk)
        result = {"T_mev":T_mev,"number_of_terms":nk+1,
                  "max_coefficient_error":float(np.max(abs(e-er))),
                  "max_decay_rate_error":float(np.max(abs(v-vr))),
                  "slowest_decay_rate":float(min(vr))}
        results[case]=result
        if result["max_coefficient_error"]>1e-8 or result["max_decay_rate_error"]>1e-8:
            raise AssertionError(f"Published Padé coefficients not reproduced: {result}")
    (directory/"results.json").write_text(json.dumps(results,indent=2))
    lines=["# Exact-source Padé coefficient reconstruction","",
           f"Pinned upstream revision: `{REV}`",
           "Lorentzian coupling Gamma=0.1 meV, bandwidth W=5 meV, hbar=0.658211928 meV ps.",
           "Chemical-potential phase conventions are not tested here.",
           "| Dataset | max coefficient error | max rate error |",
           "|---|---:|---:|"]
    for label,r in results.items():
        lines.append(f"| {label} | {r['max_coefficient_error']:.3e} | {r['max_decay_rate_error']:.3e} |")
    lines+=["", "Both errors must be under 1e-8 for each dataset.",
            "This checks the correlation expansion, NOT a converged HEOM or PINN-DQME benchmark."]
    (directory/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
    return results


if __name__ == "__main__":
    run()
