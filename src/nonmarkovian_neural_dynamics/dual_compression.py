"""Quadrature-weighted POD compression of a precomputed HEOM dual trajectory.

A classical reduced-order / proper-orthogonal-decomposition technique,
NOT a new neural algorithm. The full adjoint must first be generated;
compressing it can only benefit repeated reuse (e.g., many residuals or
parameters) at fixed Hamiltonian, observable and final horizon.

The Cauchy-Schwarz error bound is valid for the *discretized quadrature*
contraction; it does NOT bound HEOM truncation or continuous-time quadrature.
"""
import numpy as np


def simpson_weights(times):
    t=np.asarray(times,dtype=float)
    n=len(t)
    if n<5 or n%2!=1 or np.any(np.diff(t)<=0):
        raise ValueError("Require odd >=5 increasing uniform times")
    widths=np.diff(t)
    if not np.allclose(widths,widths[0],rtol=1e-10,atol=1e-13):
        raise ValueError("Only uniform Simpson grid supported")
    w=np.ones(n)
    w[1:-1:2]=4.
    w[2:-1:2]=2.
    return w*widths[0]/3.


class CompressedDual:
    def __init__(self, times, dual):
        self.times=np.asarray(times,dtype=float)
        self.dual=np.asarray(dual,dtype=np.complex128)
        if self.dual.ndim!=2 or self.dual.shape[0]!=len(self.times):
            raise ValueError("Dual/time dimension mismatch")
        self.weights=simpson_weights(self.times)
        self.sqrtw=np.sqrt(self.weights)
        self.weighted=self.sqrtw[:,None]*self.dual
        gram=self.weighted@self.weighted.conj().T
        values,vectors=np.linalg.eigh((gram+gram.conj().T)/2.)
        descending=np.argsort(values)[::-1]
        self.eigenvalues=np.maximum(values[descending].real,0.)
        self.left_vectors=vectors[:,descending]
        self.energy=float(np.sum(abs(self.weighted)**2))
        self.n_samples,self.n_state=self.dual.shape

    def factors(self,rank):
        if not 1<=rank<=self.n_samples:
            raise ValueError("Invalid POD rank")
        basis=self.left_vectors[:,:rank]
        state_factors=basis.conj().T@self.weighted
        residual=self.weighted-basis@state_factors
        weighted_error=float(np.linalg.norm(residual))
        return basis,state_factors,weighted_error

    def contract(self,source,rank):
        """Return compressed gradient plus a rigorous sampled-quadrature bound."""
        source=np.asarray(source,dtype=np.complex128)
        if source.shape!=self.dual.shape:
            raise ValueError("Source/dual shape mismatch")
        b,state,error=self.factors(rank)
        weighted_source=self.sqrtw[:,None]*source
        source_coeff=b.conj().T@weighted_source
        estimate=float(np.real(np.vdot(state,source_coeff)))
        exact=float(np.real(np.sum(
            self.weights[:,None]*self.dual.conj()*source)))
        bound=error*float(np.linalg.norm(weighted_source))
        if abs(estimate-exact)>bound+2e-9:
            raise AssertionError("Cauchy-Schwarz compression bound violated")
        return {
            "rank":rank,
            "estimated_gradient":estimate,
            "full_discrete_gradient":exact,
            "gradient_absolute_error":abs(estimate-exact),
            "certified_discrete_error_bound":bound,
            "relative_dual_L2_error":error/max(np.sqrt(self.energy),1e-14),
            "full_complex_entries":self.n_samples*self.n_state,
            "compressed_complex_entries":rank*(self.n_samples+self.n_state),
        }
