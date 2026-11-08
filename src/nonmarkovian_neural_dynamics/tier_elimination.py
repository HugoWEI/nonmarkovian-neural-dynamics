"""Exact elimination of the deepest HEOM tier for stationary states.

HEOM couples each auxiliary density operator (ADO) only to its neighbouring
tiers.  Order the depth-(n+1) state as y = (x, u), where x holds tiers 0..n and
u holds the deepest tier n+1, so that

    L_high = [[A, B],
              [C, D]].

The deepest tier has no tier below it, so D only contains the system
Liouvillian and the bath decay rates of each ADO.  D is therefore block
diagonal (one block per ADO, of size d^2 for a d-level system), and exactly
diagonal whenever the system Hamiltonian is diagonal in the chosen basis
(as in the Anderson impurity used here).

At stationarity, C x + D u = 0, hence u = -D^{-1} C x exactly, and

    (A - B D^{-1} C) x = 0.

The full depth-(n+1) stationary hierarchy is thus obtained from a linear
system of the depth-n size plus a block-diagonal solve.  This is ordinary
Gaussian elimination (a Schur complement), related to the HEOM "terminator"
literature; nothing is approximated, and the result must agree with a direct
solve to linear-algebra precision.  For time evolution the deepest tier still
carries memory, so dynamics are run on the full hierarchy.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def block_inverse(D, block: int):
    """Invert a block-diagonal sparse matrix with square blocks of size ``block``."""
    D = sp.csr_matrix(D)
    n = D.shape[0]
    if n % block:
        raise ValueError("matrix size is not a multiple of the block size")
    coo = D.tocoo()
    if np.any(coo.row // block != coo.col // block):
        raise ValueError("D has couplings between different ADOs")
    if coo.nnz == n and np.all(coo.row == coo.col):
        diag = D.diagonal()
        if np.any(diag == 0):
            raise np.linalg.LinAlgError("singular diagonal entry in deepest tier")
        return sp.diags(1.0 / diag, format="csr")
    dense = np.zeros((n // block, block, block), dtype=complex)
    dense[coo.row // block, coo.row % block, coo.col % block] = coo.data
    return sp.csr_matrix(sp.block_diag(
        [sp.csr_array(b) for b in np.linalg.inv(dense)], format="csr"))


def schur_generator(A, B, C, D, block: int):
    """Return A - B D^{-1} C and D^{-1}."""
    Dinv = block_inverse(D, block)
    return (sp.csr_matrix(A) - sp.csr_matrix(B) @ Dinv @ sp.csr_matrix(C)).tocsc(), Dinv


def stationary_null_vector(L, trace_row):
    """Solve L x = 0 subject to trace_row . x = 1 (replaces the first row)."""
    L = sp.lil_matrix(L, dtype=complex)
    L[0, :] = 0
    L = L.tocsr()
    L = L + sp.csr_matrix(
        (np.asarray(trace_row, dtype=complex)[np.nonzero(trace_row)],
         (np.zeros(np.count_nonzero(trace_row), dtype=int),
          np.flatnonzero(trace_row))),
        shape=L.shape,
    )
    rhs = np.zeros(L.shape[0], dtype=complex)
    rhs[0] = 1.0
    return spsolve(L.tocsc(), rhs)


def trace_row(n_total: int, dim: int):
    """Linear functional giving Tr(rho) for a row-major ADO vector (rho first)."""
    row = np.zeros(n_total, dtype=complex)
    row[[i * dim + i for i in range(dim)]] = 1.0
    return row


def eliminated_steady_state(A, B, C, D, retained, omitted, dim: int):
    """Exact deepest-tier-eliminated stationary hierarchy.

    Parameters
    ----------
    A, B, C, D : blocks of the high-depth generator (retained/omitted order).
    retained, omitted : integer index arrays into the high-depth vector.
    dim : Hilbert-space dimension of the system (ADO block is dim**2).

    Returns the full high-depth stationary ADO vector and the Schur generator.
    """
    block = dim * dim
    S, Dinv = schur_generator(A, B, C, D, block)
    x = stationary_null_vector(S, trace_row(S.shape[0], dim))
    u = -(Dinv @ (sp.csr_matrix(C) @ x))
    y = np.zeros(len(retained) + len(omitted), dtype=complex)
    y[retained] = x
    y[omitted] = u
    return y, S


def deepest_tier_split(labels, n_block: int):
    """Retained/omitted vector indices for the deepest tier of a HEOM basis.

    ``labels`` are QuTiP HEOM ADO labels: one occupation number per bath
    exponent, so the tier of an ADO is the sum of its label.  The physical density operator
    (empty label) must be the first ADO.
    """
    tiers = np.array([sum(l) for l in labels])
    if tiers[0] != 0:
        raise ValueError("first ADO must be the physical density operator")
    top = tiers.max()
    ados_kept = np.flatnonzero(tiers < top)
    ados_top = np.flatnonzero(tiers == top)
    expand = lambda a: (a[:, None] * n_block + np.arange(n_block)).reshape(-1)
    return expand(ados_kept), expand(ados_top)


def split_generator(L, retained, omitted):
    L = sp.csr_matrix(L)
    return (L[retained][:, retained], L[retained][:, omitted],
            L[omitted][:, retained], L[omitted][:, omitted])
