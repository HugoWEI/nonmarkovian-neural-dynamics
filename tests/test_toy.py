import numpy as np
import pytest
from nonmarkovian_neural_dynamics.toy import check_adjoint_identity, exact_state, generator


@pytest.mark.parametrize("lam", [2.0, 0.2, 0.02])
def test_adjoint_error_identity(lam):
    assert check_adjoint_identity(lam=lam)["absolute_identity_error"] < 1e-7


def test_initial_condition():
    np.testing.assert_allclose(exact_state(0, g=1.0, lam=0.2), [1, 0])


def test_invalid_parameters():
    with pytest.raises(ValueError):
        generator(g=0, lam=0.2)
