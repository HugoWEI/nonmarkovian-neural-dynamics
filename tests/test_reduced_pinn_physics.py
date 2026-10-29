"""Independent observable math tests before neural training."""
import numpy as np
from scipy.linalg import expm
from nonmarkovian_neural_dynamics.exact_rlm import HBAR,quadrature,occupation

def test_discrete_fermionic_occupation_is_probability():
    e,w=quadrature(2,20.)
    V=np.sqrt(.1*25/(e*e+25)*w/(2*np.pi))
    h=np.diag(np.r_[-.5,e])
    h[0,1:]=V;h[1:,0]=V
    t=2.
    u=expm(-1j*h*t/HBAR)
    f=1/(1+np.exp(e/.3))
    n=np.sum(abs(u[0,1:])**2*f)
    assert 0<=n<=1
    assert abs(np.sum(abs(u[0,:])**2)-1.)<1e-12
    assert np.isfinite(occupation(np.array([0.,t]),.3,n=3)).all()
