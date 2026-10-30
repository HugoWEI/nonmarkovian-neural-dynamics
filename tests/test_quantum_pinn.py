"""Analytical/physical tests for the independent zero-temperature benchmark."""
import numpy as np
import pytest

from scripts.quantum_pinn import exact_states, matrix


def test_exact_initial_state():
    np.testing.assert_allclose(exact_states([0.], 0.8, 0.2)[0], [1., 0.])


def test_damped_pseudomode_energy_identity():
    g, lam = 0.8, 0.2
    a = matrix(g, lam)
    for x in exact_states(np.linspace(0, 10, 30), g, lam):
        # d/dt ||x||^2 = 2 x^T A x = -2 lambda B^2.
        assert abs(2 * (x @ a @ x) + 2 * lam * x[1]**2) < 1e-12
        assert x @ x <= 1 + 1e-12


def test_strong_memory_revival_but_weak_memory_monotonic_decay():
    strong = exact_states(np.linspace(0, 10, 401), 0.8, 0.2)[:, 0]**2
    weak = exact_states(np.linspace(0, 10, 401), 0.2, 2.)[:, 0]**2
    assert np.maximum(np.diff(strong), 0).sum() > 0.5
    assert np.maximum(np.diff(weak), 0).sum() < 1e-10
    assert np.all((strong >= 0) & (strong <= 1 + 1e-12))


def test_pinn_optimization_smoke():
    pytest.importorskip("torch")
    from scripts.quantum_pinn import benchmark
    result = benchmark(steps=15, mode="uniform")
    assert np.isfinite(result["final_loss"])
    assert result["population_rmse"] >= 0
