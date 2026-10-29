"""Verify upstream chemical-potential sign indexing against QuTiP phases.

Original Liouville code adds 1j*(-1)**(sign_index+1)*mu/hbar to gamma,
where original gamma has negative real part. QuTiP C_sigma uses exp(-vk t),
with vk = rate - sigma*1j*mu/hbar.
"""
import numpy as np
if __package__:
    from .pade_source_match import HBAR, reconstruct
else:
    from pade_source_match import HBAR, reconstruct


def upstream_exponents(rates, mu_mev, sign_index):
    if sign_index not in (0,1):
        raise ValueError("sign_index must be 0 or 1")
    return -np.asarray(rates,dtype=complex) + 1j * ((-1)**(sign_index+1))*mu_mev/HBAR


def qutip_exponents(rates, mu_mev, sigma):
    if sigma not in (-1,1):
        raise ValueError("sigma must be +1 or -1")
    vk=np.asarray(rates,dtype=complex) - sigma*1j*mu_mev/HBAR
    return -vk


def compare():
    results=[]
    for thermal,nk in ((0.3,3),(0.03,7)):
        _,rates=reconstruct(thermal,nk)
        for mu in (0.0,0.05):
            for idx,sigma in ((0,-1),(1,1)):
                upstream=upstream_exponents(rates,mu,idx)
                qutip=qutip_exponents(rates,mu,sigma)
                diff=float(max(abs(upstream-qutip)))
                assert diff<1e-14
                results.append((thermal,mu,idx,sigma,diff))
    return results


if __name__ == "__main__":
    for record in compare():
        print(record)
