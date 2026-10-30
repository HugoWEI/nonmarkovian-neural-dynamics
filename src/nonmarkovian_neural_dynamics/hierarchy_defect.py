"""Goal-oriented HEOM hierarchy truncation defect with consistent initial ADOs.

The error identity is exact for nested time-independent linear systems:
e'(t) = L_hi e(t) + r(t),
r= P L_lo y_lo - L_hi P y_lo,
e(0)=P y_lo(0)-y_hi(0).
No neural novelty. A high-depth dual is expensive; low-rank compression
and coarse-dual substitutes require explicit validation.
"""
import numpy as np
from scipy.integrate import simpson
from scipy.sparse.linalg import expm_multiply


def hierarchy_retained_indices(low_labels,high_labels,blockdim):
    high_map={tuple(x):i for i,x in enumerate(high_labels)}
    if len(high_map)!=len(high_labels):
        raise ValueError("Duplicate hierarchy labels")
    out=[]
    for label in low_labels:
        idx=high_map.get(tuple(label))
        if idx is None:
            raise ValueError("Low hierarchy is not nested in high hierarchy")
        out.extend(range(blockdim*idx,blockdim*(idx+1)))
    return np.asarray(out,dtype=np.int64)


def compute_defect(Llo,Lhi,y0hi,indices,times):
    """Propagate projected initial state, then evaluate high-depth residual."""
    t=np.asarray(times,dtype=float)
    y0hi=np.asarray(y0hi,dtype=np.complex128).reshape(-1)
    indices=np.asarray(indices,dtype=int)
    if Lhi.shape!=(len(y0hi),len(y0hi)) or Llo.shape!=(len(indices),len(indices)):
        raise ValueError("Hierarchy generator/index dimension mismatch")
    ylo=expm_multiply(Llo,y0hi[indices],start=float(t[0]),stop=float(t[-1]),
                      num=len(t),endpoint=True,traceA=np.sum(Llo.diagonal()))
    yhat=np.zeros((len(t),len(y0hi)),dtype=np.complex128)
    yhat[:,indices]=ylo
    source=- (Lhi@yhat.T).T
    source[:,indices]+=(Llo@ylo.T).T
    init=yhat[0]-y0hi
    return yhat,source,init


def linear_observable_defect(dual,source,initial,times):
    """Known full-high-depth dual, numerical Simpson defect identity."""
    z=np.asarray(dual)
    source=np.asarray(source)
    init=np.asarray(initial)
    initial_value=float(np.real(np.vdot(z[0],init)))
    residual_value=float(np.real(simpson(
        np.einsum("ij,ij->i",z.conj(),source),x=times)))
    return initial_value,residual_value,initial_value+residual_value
