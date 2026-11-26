"""Linear memory toy and an exact dual-weighted-residual consistency test.

a'(t) = -g^2 * integral_0^t exp(-lam*(t-s)) a(s) ds
b(t) = g * integral_0^t exp(-lam*(t-s)) a(s) ds
x=(a,b), x'=A x, A=[[0,-g],[g,-lam]], x(0)=[1,0].

This is NOT a fermionic DQME simulation and NOT a neural result.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from scipy.integrate import simpson


def generator(g: float, lam: float) -> np.ndarray:
    if g <= 0 or lam < 0:
        raise ValueError("Require g>0 and lam>=0")
    return np.array([[0.0, -g], [g, -lam]], dtype=float)


def exact_state(t: float, g: float, lam: float) -> np.ndarray:
    return expm(generator(g, lam) * float(t)) @ np.array([1.0, 0.0])


def adjoint_weights(t: float, t_end: float, g: float, lam: float) -> np.ndarray:
    """w(t) = exp(A.T * (T-t)) c, with observable c=(1,0)."""
    return expm(generator(g, lam).T * (t_end - t)) @ np.array([1.0, 0.0])


def check_adjoint_identity(
    g: float = 1.0, lam: float = 0.2, t_end: float = 4.0,
    epsilon: float = 0.03, n: int = 1001,
) -> dict[str, float]:
    """Verify observable error = integral of adjoint times residual.

    Construct xhat(t) = xexact(t) + epsilon*(t/T)^2*v with v=(1,0).
    We use its analytic residual: this is NOT a trained approximation.
    """
    if t_end <= 0 or n < 3 or n % 2 == 0:
        raise ValueError("Require t_end>0 and odd n>=3")
    A = generator(g, lam)
    v = np.array([1.0, 0.0])
    t_grid = np.linspace(0.0, t_end, n)
    integrand = np.empty(n)
    for k, t in enumerate(t_grid):
        h, hprime = (t/t_end)**2, 2.0*t/t_end**2
        residual = epsilon*(hprime*v - h*(A @ v))
        integrand[k] = adjoint_weights(t, t_end, g, lam) @ residual
    weighted_residual = float(simpson(integrand, x=t_grid))
    true_error = float(epsilon*v[0])
    return {
        "g": g, "lambda": lam, "T": t_end,
        "exact_observable_error": true_error,
        "adjoint_residual_integral": weighted_residual,
        "absolute_identity_error": abs(true_error - weighted_residual),
        "reference_amplitude": float(exact_state(t_end, g, lam)[0]),
    }
