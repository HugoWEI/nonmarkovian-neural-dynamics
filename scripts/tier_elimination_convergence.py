"""Hierarchy-depth convergence of the interacting correlated quench, using exact
deepest-tier elimination to prepare each depth's correlated stationary state.

Model: spinful Anderson impurity, U=1 meV, eps=-0.5 meV, one Lorentzian
fermionic lead per spin (Gamma=0.1 meV, W=5 meV), Pade bath expansion.
For every (T/Gamma, Nk, depth): prepare the fully correlated stationary ADO
hierarchy at mu=0, quench the lead chemical potential to mu=0.05 meV and
evolve n_up(t) for 6 ps with a tight-tolerance explicit integrator.

Usage: python scripts/tier_elimination_convergence.py [--out DIR] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interacting_sparse_adjoint import build, occupation_functional  # noqa: E402
from nonmarkovian_neural_dynamics.linear_adjoint import sparse_generator  # noqa: E402
from nonmarkovian_neural_dynamics.tier_elimination import (  # noqa: E402
    deepest_tier_split, eliminated_steady_state, split_generator)

CASES_FULL = [(3.0, 3, (2, 3, 4, 5)), (0.3, 3, (2, 3, 4, 5)),
              (0.3, 5, (2, 3, 4)), (0.3, 7, (2, 3, 4))]
CASES_QUICK = [(3.0, 3, (2, 3)), (0.3, 3, (2, 3))]


def evolve(L, y0, times, rtol, atol):
    sol = solve_ivp(lambda t, y: L @ y, (times[0], times[-1]), y0,
                    t_eval=times, method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def run_case(ratio, nk, depth, times, rtol=1e-10, atol=1e-12):
    t0 = time.perf_counter()
    eq, op = build(ratio, nk, depth, 0.0, 1.0)
    quench, _ = build(ratio, nk, depth, 0.05, 1.0)
    if list(eq.ados.labels) != list(quench.ados.labels):
        raise AssertionError("ADO ordering changed between mu values")
    L_eq, L_q = sparse_generator(eq), sparse_generator(quench)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    ret, om = deepest_tier_split(list(eq.ados.labels), 16)
    y0, _ = eliminated_steady_state(*split_generator(L_eq, ret, om), ret, om, 4)
    steady_s = time.perf_counter() - t0
    stationarity = float(np.linalg.norm(L_eq @ y0))

    c = occupation_functional(op, len(eq.ados.labels))
    t0 = time.perf_counter()
    traj = evolve(L_q, y0, times, rtol, atol)
    dyn_s = time.perf_counter() - t0
    n_up = np.real(traj @ c.conj())
    rho_T = traj[-1, :16].reshape(4, 4)
    rho_T = 0.5 * (rho_T + rho_T.conj().T)
    record = dict(T_over_Gamma=ratio, Nk=nk, depth=depth,
                  n_ados=len(eq.ados.labels), dim=int(L_eq.shape[0]),
                  build_s=build_s, eliminated_steady_s=steady_s,
                  dynamics_s=dyn_s, stationarity_residual=stationarity,
                  n_up_initial=float(n_up[0]), n_up_final=float(n_up[-1]),
                  min_rho_eigenvalue_final=float(np.linalg.eigvalsh(rho_T).min()),
                  trace_final=float(np.real(np.trace(rho_T))))
    return record, n_up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/tier_elimination_convergence")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    times = np.linspace(0.0, 6.0, 81)
    records, curves = [], {}
    for ratio, nk, depths in (CASES_QUICK if args.quick else CASES_FULL):
        for depth in depths:
            rec, n_up = run_case(ratio, nk, depth, times)
            records.append(rec)
            curves[f"T{ratio}_Nk{nk}_d{depth}"] = n_up
            print(json.dumps(rec), flush=True)
            prev = f"T{ratio}_Nk{nk}_d{depth - 1}"
            if prev in curves:
                diff = np.abs(n_up - curves[prev])
                summary = dict(T_over_Gamma=ratio, Nk=nk,
                               depths=f"{depth - 1}->{depth}",
                               max_abs_diff=float(diff.max()),
                               final_abs_diff=float(diff[-1]))
                records.append(summary)
                print(json.dumps(summary), flush=True)
            np.savez_compressed(out / "trajectories.npz", times=times, **curves)
            (out / "summary.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
