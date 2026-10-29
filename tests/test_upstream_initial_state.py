"""Regression checks for published reduced-density diagnostic parsing."""
import numpy as np
from scripts.upstream_initial_state_audit import parse_first_diagonal


def fake_table(probabilities):
    n_aux = 16
    state_bits = ((0,0,0,0),(1,0,1,0),(0,1,0,1),(1,1,1,1))
    rows = []
    for prob, bits in zip(probabilities, state_bits):
        rows.append(" ".join(map(str,[prob, 0.0]+[0]*n_aux+list(bits))))
    rows.append("0.01 0.002 1 "+ "0 "* (n_aux + 3))
    return "\n".join(rows)


def test_extracts_zero_tier_probabilities():
    p = [0.7954950930330533,0.09138879687552733,
         0.09138879687461894,0.02172731321281027]
    found = parse_first_diagonal(fake_table(p))
    np.testing.assert_allclose(found, p, rtol=0, atol=1e-15)
    assert found[0] != 1.
