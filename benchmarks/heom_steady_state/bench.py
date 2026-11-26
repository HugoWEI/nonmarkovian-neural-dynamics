"""HEOM stationary-state benchmark: QuTiP direct solve vs exact deepest-tier elimination.

Model (same as the paper note): spinful Anderson impurity, U = 1 meV, eps = -0.5 meV,
one Lorentzian lead per spin (Gamma = 0.1 meV, W = 5 meV), Pade expansion with Nk poles.
Energies are in meV / hbar (units of 1/ps), exactly as in the repository.

For each case it builds the HEOM generator once, then times
  (a) QuTiP's HEOMSolver.steady_state()  (direct sparse solve of the full hierarchy)
  (b) exact elimination of the deepest tier (Schur complement)
and reports the stationary <n_up> at mu = 0.05 meV (a non-trivial value used to
cross-check against the Julia run), agreement between (a) and (b), and timings.

Usage:  python bench.py              # default cases
        python bench.py --big        # also the Nk=7, depth-4 case (needs lots of RAM for (a))
Writes results_python.json next to this file.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

HBAR = 0.658211928  # meV ps
GAMMA, W, U, EPS, MU = 0.1, 5.0, 1.0, -0.5, 0.05  # meV
RATIO = 0.3  # T / Gamma

CASES = [(3, 3), (3, 4), (7, 3), (5, 4)]
BIG = [(7, 4)]


def build(nk, depth):
    from qutip import destroy, qeye, sigmaz, tensor
    from qutip.solver.heom import HEOMSolver, LorentzianPadeBath
    up = tensor(destroy(2), qeye(2))
    down = tensor(sigmaz(), destroy(2))  # Jordan-Wigner
    nup, ndown = up.dag() * up, down.dag() * down
    H = (EPS / HBAR) * (nup + ndown) + (U / HBAR) * nup * ndown
    baths = [LorentzianPadeBath(op, GAMMA / HBAR, W / HBAR, MU / HBAR,
                                RATIO * GAMMA / HBAR, nk, tag=s)
             for s, op in (("up", up), ("down", down))]
    solver = HEOMSolver(H, baths, max_depth=depth,
                        options={"progress_bar": "", "normalize_output": False})
    return solver, nup, baths


def generator(solver):
    return solver.rhs(0.0).data.as_scipy().tocsr().astype(np.complex128)


def deepest_split(labels, block):
    tiers = np.array([sum(l) for l in labels])
    top = tiers.max()
    keep, last = np.flatnonzero(tiers < top), np.flatnonzero(tiers == top)
    expand = lambda a: (a[:, None] * block + np.arange(block)).reshape(-1)
    return expand(keep), expand(last)


def solve_with_trace(L, dim):
    """Solve L x = 0 with Tr(rho) = 1 (rho = first ADO), replacing row 0."""
    L = sp.lil_matrix(L)
    L[0, :] = 0
    for i in range(dim):
        L[0, i * dim + i] = 1.0
    rhs = np.zeros(L.shape[0], dtype=complex)
    rhs[0] = 1.0
    return spsolve(L.tocsc(), rhs)


def eliminated(L, labels, dim):
    block = dim * dim
    keep, last = deepest_split(labels, block)
    A = L[keep][:, keep]; B = L[keep][:, last]
    C = L[last][:, keep]; D = L[last][:, last].tocoo()
    if not np.all(D.row == D.col):
        raise ValueError("deepest-tier block is not diagonal for this model")
    dinv = sp.diags(1.0 / L[last][:, last].diagonal())
    S = (A - B @ dinv @ C).tocsc()
    x = solve_with_trace(S, dim)
    y = np.zeros(L.shape[0], dtype=complex)
    y[keep] = x
    y[last] = -(dinv @ (C @ x))
    return y


def n_up_of(y, dim, nup):
    rho = y[:dim * dim].reshape(dim, dim)
    return float(np.real(np.trace(nup.full() @ rho)))


def run(nk, depth, skip_direct=False):
    solver, nup, baths = build(nk, depth)
    labels = list(solver.ados.labels)
    t0 = time.perf_counter(); L = generator(solver); t_gen = time.perf_counter() - t0
    dim = 4
    rec = dict(Nk=nk, depth=depth, ados=len(labels), unknowns=int(L.shape[0]),
               generator_s=round(t_gen, 3))

    t0 = time.perf_counter(); y_el = eliminated(L, labels, dim)
    rec["eliminated_s"] = round(time.perf_counter() - t0, 3)
    rec["eliminated_residual"] = float(np.linalg.norm(L @ y_el))
    rec["n_up_eliminated"] = n_up_of(y_el, dim, nup)

    if not skip_direct:
        try:
            t0 = time.perf_counter(); _, ado = solver.steady_state(use_mkl=False)
            rec["direct_qutip_s"] = round(time.perf_counter() - t0, 3)
            y_d = np.asarray(ado._ado_state, dtype=complex).reshape(-1)
            rec["n_up_direct"] = n_up_of(y_d, dim, nup)
            rec["max_abs_diff"] = float(np.max(np.abs(y_d - y_el)))
        except MemoryError:
            rec["direct_qutip_s"] = "out of memory"
    # bath exponents, so the Julia script can check it builds the same model
    rec["exponents"] = [[str(e.type), [e.ck.real, e.ck.imag], [e.vk.real, e.vk.imag]]
                        for e in baths[0].exponents]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smallest case only")
    args = ap.parse_args()
    cases = [(3, 3)] if args.quick else CASES + (BIG if args.big else [])
    import qutip, scipy
    out = dict(machine=platform.platform(), processor=platform.processor(),
               python=platform.python_version(), qutip=qutip.__version__,
               scipy=scipy.__version__, T_over_Gamma=RATIO, mu_meV=MU, cases=[])
    for nk, depth in cases:
        rec = run(nk, depth)
        out["cases"].append(rec)
        show = {k: v for k, v in rec.items() if k != "exponents"}
        print(json.dumps(show), flush=True)
        dest = Path(__file__).with_name("results_python.json")
        dest.write_text(json.dumps(out, indent=1))  # saved after every case
    print("done; results in results_python.json")


if __name__ == "__main__":
    main()
