"""Independent quadratic resonant-level model (RLM) reference.

J(E)=Gamma W^2/(E^2+W^2), with V_j^2=J(E_j)w_j/(2*pi).
The empty-dot occupation is sum_j f(E_j)|U_dj(t)|^2, U=e^(-iHt/hbar).
This tests a single spinless dot with one factorized Gaussian thermal lead,
NOT the published interacting/correlated SIAM PINN-DQME initial state.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.special import expit

HBAR=0.658211928

def quadrature(n=96,cutoff=40.):
    if n<2 or cutoff<=10: raise ValueError("Invalid quadrature")
    edges=np.array([-cutoff,-10.,-3.,-.5,-.1,0.,.1,.5,3.,10.,cutoff])
    x,w=np.polynomial.legendre.leggauss(n)
    energies=[];weights=[]
    for a,b in zip(edges[:-1],edges[1:]):
        energies.extend(.5*(b-a)*x+.5*(a+b))
        weights.extend(.5*(b-a)*w)
    return np.array(energies),np.array(weights)

def occupation(times_ps,temperature_mev,n=96,cutoff=40.,
               eps=-.5,gamma=.1,bandwidth=5.,mu=0.,dot_initial=0.):
    if temperature_mev<=0 or not 0<=dot_initial<=1:
        raise ValueError("Invalid temperature or dot occupancy")
    e,w=quadrature(n,cutoff)
    V=np.sqrt(gamma*bandwidth**2/(e**2+bandwidth**2)*w/(2*np.pi))
    h=np.diag(np.r_[eps,e])
    h[0,1:]=V;h[1:,0]=V
    eig,vec=eigh(h,check_finite=False)
    row=(vec[0,:][None,:]*np.exp(-1j*np.outer(times_ps,eig)/HBAR))@vec.conj().T
    f=expit(-(e-mu)/temperature_mev)
    return dot_initial*abs(row[:,0])**2+abs(row[:,1:])**2@f

def reference_with_checks(times,temperature):
    ref=occupation(times,temperature,n=96,cutoff=40.)
    coarse=occupation(times,temperature,n=80,cutoff=40.)
    wider=occupation(times,temperature,n=144,cutoff=60.)
    return ref,{"resolution_max":float(max(abs(ref-coarse))),
                "cutoff_max":float(max(abs(ref-wider))),
                "reference_terminal":float(ref[-1])}
