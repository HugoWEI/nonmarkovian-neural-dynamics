"""First Born/Volterra correction from an omitted HEOM hierarchy tier.

Partition L_high=[[A,B],[C,D]]. For a low-depth solution x0'=A x0
and top-tier initial u(0)=0, approximate u'=D u + C x0(t).
Then the leading occupation correction is
delta O(T) = Re integral z_A(t)^dagger B u(t) dt,
with z_A(t)=exp(A^dagger(T-t))c.

This is an established block elimination / first-order memory approximation,
not a novel theorem or neural model. Use it only with **same initial state**
projected/embedded across the two hierarchy truncations.
"""
import numpy as np
from scipy.integrate import simpson,solve_ivp


def first_order_feedback(A,B,C,D,x0,c,times,*,rtol=1e-7,atol=1e-9):
    from nonmarkovian_neural_dynamics.linear_adjoint import adjoint_trajectory
    t=np.asarray(times,dtype=float)
    if t[0]!=0 or len(t)<5 or len(t)%2==0:
        raise ValueError("Uniform Simpson time grid must start at 0")
    _,x,z=adjoint_trajectory(A,x0,c,T=t[-1],samples=len(t))
    forcing=(C@x.T).T
    q=D.shape[0]
    def rhs(now,u):
        i=int(np.clip(np.searchsorted(t,now,side="right")-1,0,len(t)-2))
        alpha=(now-t[i])/(t[i+1]-t[i])
        f=(1-alpha)*forcing[i]+alpha*forcing[i+1]
        return D@u+f
    sol=solve_ivp(rhs,(float(t[0]),float(t[-1])),
                  np.zeros(q,dtype=complex),t_eval=t,method="RK45",
                  rtol=rtol,atol=atol)
    if not sol.success:
        raise RuntimeError(f"Omitted-tier response failed: {sol.message}")
    u=sol.y.T
    response=(B@u.T).T
    integral=float(np.real(simpson(
        np.einsum("ij,ij->i",z.conj(),response),x=t)))
    return integral,{"low_trajectory":x,"top_tier":u,
                     "interpolated_forcing":forcing,
                     "estimated_occupation_correction":integral}


def partition_nested(Ahi,Alo,indices):
    n=Ahi.shape[0]
    mask=np.ones(n,dtype=bool)
    mask[indices]=False
    omitted=np.flatnonzero(mask)
    block=Ahi[indices,:][:,indices].tocsr()
    err=(block-Alo).tocsr()
    max_deviation=max(np.abs(err.data),default=0.) if err.nnz else 0.
    if max_deviation>1e-9:
        raise AssertionError(f"Projected high-depth block != low generator: {max_deviation}")
    return (
        Alo,Ahi[indices,:][:,omitted].tocsr(),
        Ahi[omitted,:][:,indices].tocsr(),
        Ahi[omitted,:][:,omitted].tocsr(),omitted
    )
