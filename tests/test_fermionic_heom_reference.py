"""Physical Hamiltonian checks for the QuTiP fermionic HEOM setup."""
import numpy as np
import pytest

qutip = pytest.importorskip("qutip")
from scripts.fermionic_heom_reference import system, HBAR_MEV_PS


def test_jordan_wigner_fermion_car():
    H, up, down, nup, ndown = system()
    eye = qutip.tensor(qutip.qeye(2), qutip.qeye(2))
    z = up * down + down * up
    assert z.norm() < 1e-12
    assert (up * down.dag() + down.dag() * up).norm() < 1e-12
    assert (up * up.dag() + up.dag() * up - eye).norm() < 1e-12
    assert (down * down.dag() + down.dag() * down - eye).norm() < 1e-12
    assert (H - H.dag()).norm() < 1e-12


def test_interacting_occupation_energies():
    H, up, down, nup, ndown = system()
    energies = np.sort(np.real(H.eigenenergies()))
    target = np.sort(np.array([0., -0.5, -0.5, 0.]) / HBAR_MEV_PS)
    np.testing.assert_allclose(energies, target, atol=1e-12)
