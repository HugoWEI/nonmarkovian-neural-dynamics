"""Exactly solvable finite-bath interacting Anderson impurity.

Spinful dot with Hubbard U, discrete fermionic lead levels and thermal initial
product state. Exact many-body evolution (NOT HEOM or continuum-bath DQME).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.special import expit


def hopping_element(dst: int, src: int, i: int, j: int) -> int:
    """Fock-basis matrix element <dst|c_i^dagger c_j|src>."""
    if not (src & (1 << j)):
        return 0
    middle = src ^ (1 << j)
    if middle & (1 << i) or dst != (middle | (1 << i)):
        return 0
    sign_j = (-1) ** ((src & ((1 << j) - 1)).bit_count())
    sign_i = (-1) ** ((middle & ((1 << i) - 1)).bit_count())
    return sign_i * sign_j


def anderson_hamiltonian(
    bath_energies=(-0.7,), bath_couplings=(0.55,),
    eps_dot=-0.5, interaction=1.0,
):
    if len(bath_energies) != len(bath_couplings) or not bath_energies:
        raise ValueError("Non-empty, matching bath energies and couplings required")
    n_modes = 2 + 2 * len(bath_energies)
    dim = 1 << n_modes
    h = np.zeros((dim, dim), dtype=np.complex128)
    for state in range(dim):
        occ = [(state >> k) & 1 for k in range(n_modes)]
        h[state, state] = (
            eps_dot * (occ[0] + occ[1])
            + interaction * occ[0] * occ[1]
            + sum(e * (occ[2 + 2*j] + occ[3 + 2*j])
                  for j, e in enumerate(bath_energies))
        )
        for j, coupling in enumerate(bath_couplings):
            for spin in (0, 1):
                dot, bath = spin, 2 + 2*j + spin
                for i, k in ((dot, bath), (bath, dot)):
                    for dst in range(dim):
                        entry = hopping_element(dst, state, i, k)
                        if entry:
                            h[dst, state] += coupling * entry
    return h


def thermal_diagonal(bath_energies, temperature,
                     chemical_potential=0.0, impurity_occupations=(0, 0)):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    nmodes = 2 + 2 * len(bath_energies)
    fillings = [
        expit(-(energy - chemical_potential) / temperature)
        for energy in bath_energies for _ in range(2)
    ]
    probabilities = np.zeros(1 << nmodes)
    for state in range(len(probabilities)):
        if (state & 1, (state >> 1) & 1) != tuple(impurity_occupations):
            continue
        prob = 1.0
        for mode, f in enumerate(fillings, start=2):
            prob *= f if state & (1 << mode) else (1 - f)
        probabilities[state] = prob
    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-12)
    return probabilities


def number_operator(mode, dim):
    return np.diag([float((state >> mode) & 1) for state in range(dim)]).astype(complex)


def exact_evolution(h, initial_probabilities, times):
    """time, all occupations, d(n_up)/dt, d(n_down)/dt, system energy."""
    energies, v = eigh(h)
    nmodes = int(np.log2(h.shape[0]))
    occupancy = np.array([
        [(state >> mode) & 1 for state in range(h.shape[0])]
        for mode in range(nmodes)
    ], dtype=float)
    output = []
    for t in times:
        u = (v * np.exp(-1j * energies * t)) @ v.conj().T
        rho = (u * initial_probabilities[None, :]) @ u.conj().T
        diagonal = (np.abs(u) ** 2) @ initial_probabilities
        occupations = occupancy @ diagonal
        currents = []
        for mode in (0, 1):
            n = number_operator(mode, h.shape[0])
            currents.append(np.real(1j * np.trace(rho @ (h @ n - n @ h))))
        energy = np.real(np.trace(rho @ h))
        output.append([float(t), *occupations, *currents, energy])
    return np.array(output)


def run_reference(temperature=0.3, levels=1, interaction=1.0, times=None):
    if levels == 1:
        energies, couplings = (-0.7,), (0.55,)
    elif levels == 2:
        energies, couplings = (-0.7, 0.85), (0.40, 0.35)
    else:
        raise ValueError("Only 1 or 2 bath levels supported in this benchmark")
    h = anderson_hamiltonian(energies, couplings, interaction=interaction)
    p = thermal_diagonal(energies, temperature)
    if times is None:
        times = np.linspace(0, 8, 161)
    return exact_evolution(h, p, times)
