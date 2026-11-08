"""Physical consistency tests for a finite-bath interacting Anderson model."""
import numpy as np

from nonmarkovian_neural_dynamics.finite_anderson import (
    anderson_hamiltonian, thermal_diagonal, run_reference,
)


def test_hermiticity_and_spin_conservation():
    h = anderson_hamiltonian((-0.7, 0.85), (0.4, 0.35), interaction=1.0)
    np.testing.assert_allclose(h, h.conj().T, atol=1e-14)
    for spin in (0, 1):
        sector = np.array([
            sum((s >> k) & 1 for k in range(spin, 6, 2))
            for s in range(len(h))
        ])
        assert np.max(np.abs((sector[:, None]-sector[None, :])*h)) < 1e-12


def test_thermal_probabilities():
    for temp in (0.3, 3.0):
        p = thermal_diagonal((-0.7, 0.85), temp)
        assert abs(p.sum()-1) < 1e-12
        assert (p >= 0).all()


def test_exact_energy_number_and_occupations():
    for levels in (1, 2):
        for temp in (0.3, 3.0):
            results = run_reference(temp, levels=levels,
                                    times=np.linspace(0, 5, 31))
            populations = results[:, 1:1+2+2*levels]
            assert np.ptp(populations.sum(axis=1)) < 1e-11
            assert np.ptp(results[:, -1]) < 1e-11
            assert populations.min() >= -1e-12
            assert populations.max() <= 1+1e-12


def test_dot_current_matches_time_derivative():
    t = np.linspace(0, 3, 301)
    results = run_reference(0.3, levels=1, times=t)
    derivative = np.gradient(results[:, 1:3], t, axis=0)
    np.testing.assert_allclose(
        derivative[2:-2], results[2:-2, -3:-1], atol=4e-4
    )
