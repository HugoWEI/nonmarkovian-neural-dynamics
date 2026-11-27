"""The one-call API matches QuTiP's own steady state on a small fermionic model."""
import numpy as np
import pytest

qutip = pytest.importorskip("qutip")

from nonmarkovian_neural_dynamics.tier_elimination import steady_state


def _solver():
    from qutip import destroy, qeye, sigmaz, tensor
    from qutip.solver.heom import HEOMSolver, LorentzianPadeBath
    up, dn = tensor(destroy(2), qeye(2)), tensor(sigmaz(), destroy(2))
    H = -0.5 * (up.dag() * up + dn.dag() * dn) + up.dag() * up * dn.dag() * dn
    baths = [LorentzianPadeBath(op, 0.1, 5.0, 0.05, 0.03, 2, tag=t)
             for t, op in (("u", up), ("d", dn))]
    return HEOMSolver(H, baths, max_depth=3, options={"progress_bar": ""}), up


@pytest.mark.parametrize("method,tol", [("direct", 1e-11), ("gmres", 1e-8)])
def test_matches_qutip(method, tol):
    solver, up = _solver()
    rho_ref, ados_ref = solver.steady_state()
    rho, ados = steady_state(solver, method=method)
    assert abs(qutip.expect(up.dag() * up, rho) - qutip.expect(up.dag() * up, rho_ref)) < tol
    assert np.max(np.abs((rho - rho_ref).full())) < tol
    assert abs(rho.tr() - 1) < 1e-12
