"""Exact source-compatible benchmark coefficients, without network or QuTiP."""
import numpy as np
from scripts.pade_source_match import reconstruct


def test_high_temperature_first_coefficient():
    eta, rates = reconstruct(0.3, 3)
    np.testing.assert_allclose(
        eta[0], complex(0.28852172265373072, -0.16136646927170406),
        rtol=0, atol=1e-10,
    )
    np.testing.assert_allclose(
        rates, [7.5963375735117342, 1.4318760879309518,
                4.3532491698512104, 12.389386199282718],
        atol=1e-10, rtol=0,
    )


def test_low_temperature_slowest_decay_and_coefficient():
    eta, rates = reconstruct(0.03, 7)
    np.testing.assert_allclose(
        eta[0], complex(0.28852172265373072, 0.90793314764449873),
        rtol=0, atol=1e-9,
    )
    np.testing.assert_allclose(
        rates[1], 0.14318758989079544, rtol=0, atol=1e-10,
    )
    assert len(eta) == 8
