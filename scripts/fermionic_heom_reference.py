"""QuTiP fermionic HEOM: spinful interacting Anderson impurity, Lorentzian leads.

A *continuum-bath intermediate reference*, NOT an exact reproduction of the
authors' dissipaton coefficients; see docs/continuum_heom_protocol.md.
QuTiP is an optional dependency, installed by the GitHub Actions workflow.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np

HBAR_MEV_PS = 0.658211928


def system():
    from qutip import tensor, destroy, qeye, sigmaz
    # Jordan-Wigner mapping ensures two fermionic channels anticommute.
    up = tensor(destroy(2), qeye(2))
    down = tensor(sigmaz(), destroy(2))
    nup = up.dag() * up
    ndown = down.dag() * down
    h = (-0.5 / HBAR_MEV_PS) * (nup + ndown) + (1.0 / HBAR_MEV_PS) * nup * ndown
    return h, up, down, nup, ndown


def compute(temperature_ratio, depth, *, nk=1, t_end=1.5, n_times=31):
    from qutip import tensor, basis
    from qutip.solver.heom import HEOMSolver, LorentzianPadeBath
    h, up, down, nup, ndown = system()
    # Gamma=0.1 meV for the T/Gamma labels; provisional single-lead spectral
    # normalization 0.05 meV per lead. Not validated against upstream eta_k.
    cutoff = 5.0 / HBAR_MEV_PS
    coupling = 0.05 / HBAR_MEV_PS
    temperature = temperature_ratio * 0.1 / HBAR_MEV_PS
    baths = [
        LorentzianPadeBath(op, coupling, cutoff, mu / HBAR_MEV_PS,
                           temperature, nk, tag=f"{spin}_{lead}")
        for spin, op in (("up", up), ("down", down))
        for lead, mu in (("L", 0.0), ("R", 0.05))
    ]
    solver = HEOMSolver(h, baths, max_depth=depth,
                        options={"nsteps": 20000, "rtol": 1e-7, "atol": 1e-9})
    rho0 = tensor(basis(2, 0), basis(2, 0)).proj()
    times = np.linspace(0.0, t_end, n_times)
    result = solver.run(rho0, times, e_ops={"up": nup, "down": ndown})
    pop_up = np.real(np.asarray(result.e_data["up"], dtype=complex))
    pop_down = np.real(np.asarray(result.e_data["down"], dtype=complex))
    if not (np.isfinite(pop_up).all() and np.isfinite(pop_down).all()):
        raise FloatingPointError("HEOM produced nonfinite occupations")
    if abs(pop_up[0]) > 1e-9 or abs(pop_down[0]) > 1e-9:
        raise AssertionError("Initial impurity occupation mismatch")
    return times, pop_up, pop_down


def run(out_dir="outputs/fermionic_heom", depths=(1, 2), nk=1):
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    runs = []
    compare = []
    for temp in (3.0, 0.3):
        traces = {}
        for depth in depths:
            t, up, down = compute(temp, depth, nk=nk)
            traces[depth] = (up, down)
            runs.append({"T_over_Gamma": temp, "depth": depth, "Nk": nk,
                         "terminal_n_up": float(up[-1]),
                         "terminal_n_down": float(down[-1]),
                         "minimum_occupation": float(min(up.min(), down.min())),
                         "maximum_occupation": float(max(up.max(), down.max()))})
            with (dest/f"T{temp}_depth{depth}_Nk{nk}.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time_ps", "occupation_up", "occupation_down"])
                writer.writerows(zip(t, up, down))
            print(json.dumps(runs[-1]), flush=True)
        lo, hi = depths[-2:]
        max_diff = float(max(np.max(abs(traces[lo][0]-traces[hi][0])),
                             np.max(abs(traces[lo][1]-traces[hi][1]))))
        compare.append({"T_over_Gamma": temp, "depth_pair": [lo,hi],
                        "max_depth_difference": max_diff,
                        "depth_converged_at_1e-3": max_diff < 1e-3})
    (dest/"results.json").write_text(json.dumps({"runs": runs, "depth_tests": compare}, indent=2))
    lines = [
        "# Intermediate fermionic HEOM experiment", "",
        "**Not the paper's exact bath fit or a converged HEOM reference.**",
        "QuTiP, spinful U=1 meV Anderson impurity, two Lorentzian leads per spin.",
        "Using provisional Gamma_lead=0.05 meV, W=5 meV, mu=(0,0.05) meV.",
        f"Padé Nk={nk}; hierarchy depth sweep: {list(depths)}.", "",
        "| T/Gamma | depth | terminal n_up | terminal n_down |",
        "|---:|---:|---:|---:|"
    ]
    for r in runs:
        lines.append(f"| {r['T_over_Gamma']} | {r['depth']} | {r['terminal_n_up']:.6f} | {r['terminal_n_down']:.6f} |")
    lines += ["", "## Hierarchy-depth comparison", "",
              "| T/Gamma | max delta n | criterion delta < 0.001 met? |",
              "|---:|---:|---|"]
    for c in compare:
        lines.append(f"| {c['T_over_Gamma']} | {c['max_depth_difference']:.3e} | {c['depth_converged_at_1e-3']} |")
    lines += ["", "Converging depth does NOT establish convergence of the Padé expansion.",
              "Confirm Nk, numerical tolerances, bath normalization, and current conventions",
              "before comparing to published DQME observables or claiming an accurate reference.",
              "No neural model is trained here."]
    (dest/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    return {"runs": runs, "depth_tests": compare}


if __name__ == "__main__":
    run()
