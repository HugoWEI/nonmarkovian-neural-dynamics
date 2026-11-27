"""Quickstart: stationary state of a HEOM model by exact deepest-tier elimination.

Spinful Anderson impurity with one Lorentzian lead per spin (energies in meV/hbar).
Compares QuTiP's own steady-state solver with the two elimination methods.
"""
import time

from qutip import destroy, expect, qeye, sigmaz, tensor
from qutip.solver.heom import HEOMSolver, LorentzianPadeBath

from nonmarkovian_neural_dynamics import steady_state

HBAR = 0.658211928                      # meV ps
up = tensor(destroy(2), qeye(2))
dn = tensor(sigmaz(), destroy(2))       # Jordan-Wigner
n_up, n_dn = up.dag() * up, dn.dag() * dn
H = (-0.5 * (n_up + n_dn) + 1.0 * n_up * n_dn) / HBAR

baths = [LorentzianPadeBath(op, 0.1 / HBAR, 5.0 / HBAR, 0.05 / HBAR, 0.03 / HBAR, 3, tag=s)
         for s, op in (("up", up), ("down", dn))]
solver = HEOMSolver(H, baths, max_depth=4, options={"progress_bar": ""})

for name, run in [("QuTiP steady_state", lambda: solver.steady_state()),
                  ("elimination, direct", lambda: steady_state(solver, method="direct")),
                  ("elimination, gmres ", lambda: steady_state(solver, method="gmres"))]:
    t = time.perf_counter()
    rho, ados = run()
    print(f"{name}: <n_up> = {expect(n_up, rho):.12f}   ({time.perf_counter() - t:.2f} s)")
