"""Exact deepest-tier elimination reproduces the full stationary state."""
import numpy as np
import pytest
import scipy.sparse as sp

from nonmarkovian_neural_dynamics.tier_elimination import (
    block_inverse, deepest_tier_split, eliminated_steady_state,
    split_generator, stationary_null_vector, trace_row)


def _generator_with_known_stationary_state(rng, n_keep, n_top, block, diagonal_top):
    """HEOM-like nearest-tier structure with a prescribed null vector v."""
    n1, n2 = n_keep * block, n_top * block
    cplx = lambda *shape: rng.normal(size=shape) + 1j * rng.normal(size=shape)
    A, B, C = cplx(n1, n1), 0.3 * cplx(n1, n2), 0.3 * cplx(n2, n1)
    if diagonal_top:
        D = np.diag(-2.0 - rng.random(n2) + 1j * rng.normal(size=n2))
    else:
        D = sp.block_diag([sp.csr_array(np.diag(-2.0 - rng.random(block))
                           + 0.2 * rng.normal(size=(block, block)))
                           for _ in range(n_top)]).toarray()
    # Choose v, then shift only the retained block A so that L v = 0
    # (keeps the deepest tier block D untouched).
    v = cplx(n1 + n2)
    x, u = v[:n1], v[n1:]
    u[:] = -np.linalg.solve(D, C @ x)          # top-tier stationarity
    r = A @ x + B @ u
    A = A - np.outer(r, x.conj()) / np.vdot(x, x)
    L = sp.csr_matrix(np.block([[A, B], [C, D]]))
    return L, v / (trace_row(n1 + n2, int(np.sqrt(block))) @ v)


@pytest.mark.parametrize("diagonal_top", [True, False])
def test_schur_elimination_matches_full_solve(diagonal_top):
    rng = np.random.default_rng(3)
    block = 4
    L, expected = _generator_with_known_stationary_state(
        rng, 3, 5, block, diagonal_top)
    n1 = 3 * block
    retained, omitted = np.arange(n1), np.arange(n1, L.shape[0])
    y, _ = eliminated_steady_state(*split_generator(L, retained, omitted),
                                   retained, omitted, 2)
    full = stationary_null_vector(L, trace_row(L.shape[0], 2))
    assert np.max(np.abs(y - expected)) < 1e-9
    assert np.max(np.abs(full - expected)) < 1e-9
    assert np.linalg.norm(L @ y) < 1e-9


def test_block_inverse_rejects_cross_ado_coupling():
    D = sp.csr_matrix(np.array([[1.0, 0, 0, 0], [0, 1, 0, 0],
                                [0, 0, 1, 0], [1, 0, 0, 1]]))
    with pytest.raises(ValueError):
        block_inverse(D, 2)


def test_deepest_tier_split_uses_label_sum():
    labels = [(0, 0), (1, 0), (0, 1), (1, 1)]
    keep, top = deepest_tier_split(labels, 2)
    assert keep.tolist() == [0, 1, 2, 3, 4, 5]
    assert top.tolist() == [6, 7]


def test_qutip_interacting_heom_steady_state_matches_direct():
    pytest.importorskip("qutip")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from interacting_sparse_adjoint import build
    from nonmarkovian_neural_dynamics.linear_adjoint import sparse_generator
    solver, _ = build(0.3, 2, 3, 0.0, 1.0)
    L = sparse_generator(solver)
    keep, top = deepest_tier_split(list(solver.ados.labels), 16)
    y, _ = eliminated_steady_state(*split_generator(L, keep, top), keep, top, 4)
    _, ado = solver.steady_state(use_mkl=False)
    direct = np.asarray(ado._ado_state, dtype=complex).reshape(-1)
    assert np.max(np.abs(y - direct)) < 1e-11
    assert np.linalg.norm(L @ y) < 1e-12
