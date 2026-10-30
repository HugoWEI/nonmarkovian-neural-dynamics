"""Sparse interacting-HEOM observable adjoint with *independent* FD check.

We compute \partial_mu O(T) and \partial_U O(T) with a fixed, fully
correlated initial hierarchy. No exact propagator is constructed or
diagonalized; sparse matrix-exponential *actions* are used for verification.
This is established continuous adjoint sensitivity analysis, not a new
method or proof of neural efficiency.

Important: derivatives HOLD THE INITIAL CORRELATED STEADY ADO STATE FIXED.
They are not total derivatives of a parameter-dependent re-equilibrated
state. Bath exponent labels are checked for compatibility.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy.sparse.linalg import expm_multiply


def sparse_generator(solver):
    """Return an explicit CSR generator; QuTiP exposes it as rhs(0)."""
    rhs = solver.rhs(0.0)
    mat = rhs.data.as_scipy().tocsr().astype(np.complex128)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("HEOM generator must be square")
    return mat


def adjoint_sensitivity(L, dL, state, observable, T=6., samples=81):
    """Exact finite-hierarchy adjoint identity, numerically quadrature-approximated.

    \partial_p O(T) = Re \int_0^T z(t)^* (dL/dp) y(t) dt,
    z(t) = exp[L^*(T-t)] c, y(t)=exp[L t]y0.

    Assumes a fixed initial hierarchy, a time-independent generator, and
    a linear observable represented by c* y. It does not approximate the
    potentially expensive generator: the 'approximate' part is the chosen
    hierarchy depth and finite quadrature grid.
    """
    if samples < 5 or samples % 2 != 1:
        raise ValueError("Simpson quadrature needs odd sample count >= 5")
    state = np.asarray(state, dtype=np.complex128).reshape(-1)
    c = np.asarray(observable, dtype=np.complex128).reshape(-1)
    if L.shape[0] != len(state) or len(c) != len(state):
        raise ValueError("HEOM/observable dimension mismatch")
    grid = np.linspace(0., float(T), samples)
    trace_L = np.sum(L.diagonal())
    forward = expm_multiply(L, state, start=0., stop=T, num=samples,
                            endpoint=True, traceA=trace_L)
    LH = L.conj().T.tocsr()
    adjoint_forward = expm_multiply(
        LH, c, start=0., stop=T, num=samples, endpoint=True,
        traceA=np.conj(trace_L)
    )
    adjoint = adjoint_forward[::-1]
    source = (dL @ forward.T).T
    integrand = np.einsum("ij,ij->i", adjoint.conj(), source)
    value = float(np.real(simpson(integrand, x=grid)))
    terminal = float(np.real(np.vdot(c, forward[-1])))
    return value, terminal, {
        "integrand_abs_max": float(np.max(np.abs(integrand))),
        "quadrature_samples": samples,
        "terminal_occupation": terminal,
    }


def fixed_initial_fd(L_plus, L_minus, y0, c, T, step):
    yp = expm_multiply(L_plus*T, y0, traceA=T*np.sum(L_plus.diagonal()))
    ym = expm_multiply(L_minus*T, y0, traceA=T*np.sum(L_minus.diagonal()))
    return float(np.real((np.vdot(c,yp)-np.vdot(c,ym))/(2*step)))


