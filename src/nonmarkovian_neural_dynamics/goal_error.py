"""Exact global residual/observable identity for a finite quadratic fermion.

For w'=+i(H/hbar)w and r=w_hat' - i(H/hbar)w_hat, define
e=w_hat-w. Then
e(T)=U(T)e(0)+sum_j U(T-t_j) jump_j + integral_0^T U(T-s)r(s) ds.
For O(w)=w^dag F w, the *exact* terminal-error identity is
O(w_hat)-O(w)=2 Re[w_hat^dag F e] - e^dag F e.

Uses an explicit eigendecomposition of H, so this is only a validation
identity/diagnostic on the small quadratic model. It is NOT a scalable
PINN improvement or an independent physical benchmark when used as a loss.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import simpson
from scipy.linalg import eigh

HBAR = 0.658211928

def terminal_defect(H, segments, *, initial_error=None, jumps=(), hbar=HBAR):
    H=np.asarray(H, dtype=complex)
    if H.ndim!=2 or H.shape[0]!=H.shape[1]:
        raise ValueError("H must be square")
    if not np.allclose(H,H.conj().T,atol=1e-12):
        raise ValueError("H must be Hermitian")
    energies, vectors=eigh(H)
    n=len(energies)
    if not segments:
        raise ValueError("At least one time interval is required")
    end=float(segments[-1][0][-1])
    result=np.zeros(n,dtype=complex)

    def propagate(x, dt):
        x=np.asarray(x,dtype=complex)
        return vectors@(np.exp(1j*energies*dt/hbar)*(vectors.conj().T@x))

    if initial_error is not None:
        result+=propagate(initial_error,end-float(segments[0][0][0]))
    previous_end=None
    for grid,residual in segments:
        grid=np.asarray(grid,dtype=float)
        residual=np.asarray(residual,dtype=complex)
        if grid.ndim!=1 or len(grid)<3 or len(grid)%2==0:
            raise ValueError("Use odd number of samples >=3 on each interval")
        if residual.shape!=(len(grid),n) or np.any(np.diff(grid)<=0):
            raise ValueError("Wrong residual shape or non-increasing time")
        if previous_end is not None and not np.isclose(grid[0],previous_end):
            raise ValueError("Segments must meet at interfaces")
        previous_end=grid[-1]
        weighted=(residual@vectors.conj())*np.exp(
            1j*(end-grid[:,None])*energies[None,:]/hbar
        )
        result+=vectors@simpson(weighted,x=grid,axis=0)
    for t,jump in jumps:
        result+=propagate(jump,end-float(t))
    return result

def quadratic_occupation_error(predicted, defect, occupation_weights):
    """O(predicted)-O(predicted-defect), for diagonal F."""
    predicted=np.asarray(predicted,dtype=complex)
    defect=np.asarray(defect,dtype=complex)
    weights=np.asarray(occupation_weights,dtype=float)
    if predicted.shape!=defect.shape or predicted.shape!=weights.shape:
        raise ValueError("Shape mismatch")
    if np.any((weights<0)|(weights>1)):
        raise ValueError("Fermionic occupations must lie between 0 and 1")
    linear=2*np.real(np.vdot(predicted,weights*defect))
    quadratic=np.real(np.vdot(defect,weights*defect))
    return float(linear-quadratic),float(linear),float(quadratic)
