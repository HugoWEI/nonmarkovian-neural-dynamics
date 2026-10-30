# Finite-temperature interacting Anderson model: Phase-1 intermediate results
October 29, 2026 — locally executed CPU calculations, PyTorch 2.10.0+cpu, NumPy 2.3.5, SciPy 1.17.0

## What was simulated
An exactly diagonalized **spinful interacting finite-reservoir Anderson Hamiltonian**:
H = eps_d (n_d_up + n_d_down) + U n_d_up n_d_down +
sum_{k,s} eps_k n_{ks} + sum_{k,s} [ V_k d_s^dagger c_{ks} + h.c. ]

Reference: Hilbert-space dimension 16 for one reservoir level per spin (four fermionic modes) and 64 for two levels per spin (six modes). Couplings for the 16-state example: eps_d = -0.5, U = 1, epsilon_bath = -0.7, V = 0.55. Both reservoirs initialized thermally at chemical potential 0; impurity initialized empty. These are **our benchmark choices**, not extracted parameters from the published Cao et al. SIAM.

Finite-bath unitary evolution and its many-body thermal ensemble provide the exact numerical reference at each time. No dissipaton hierarchy or continuum bath is involved. Finite-size recurrences must not be misinterpreted as a direct reproduction of continuum-bath non-Markovian memory.

## Validation
- Hamiltonian Hermiticity and spin-sector particle-number conservation.
- Exact evolution: total particle-number conservation and energy conservation to within ~1e-15 across tested temperatures T=0.3 and 3.0, and one- and two-level baths.
- Dot-occupation derivative agrees with independently evaluated commutator current within finite-difference tolerance (4e-4).

## Global PINN baseline
A neural network represents four initial thermal pure-state components simultaneously. The initial state is enforced exactly by a t * NN(t) correction. The ODE is enforced using a centered 1e-3 finite difference. Double precision, Adam with lr 0.003, two tanh hidden layers of width 48, 3000 steps, seed 7.

| T | horizon | occupation RMSE | terminal error | max trace error | CPU seconds |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 2 | 0.00156319 | 0.00236418 | 0.00926821 | 5.34 |
| 3.0 | 2 | 0.00526902 | 0.00534058 | 0.01238188 | 4.90 |
| 0.3 | 5 | 0.238086 | 0.106546 | 0.526630 | 4.84 |
| 3.0 | 5 | 0.148088 | 0.078378 | 0.406863 | 4.98 |

## Piecewise baseline
Same width, optimizer, seed, overall horizon 5 and **3000 total steps**, allocated as five subdomains x 600 steps. Each subdomain is initialized with the previous subdomain's *predicted* terminal wavefunction (no exact reference injected between windows). 41 collocation times/window, versus 81 for the global model. Thus same optimization step count is **not** the same total computational workload.

| T | horizon | occupation RMSE | terminal error | max trace error | CPU seconds |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 5 | 0.00439988 | 0.00627582 | 0.0179972 | 4.65 |
| 3.0 | 5 | 0.00265162 | 0.00413111 | 0.0132257 | 3.88 |

On this small example piecewise learning dramatically improves reconstruction at both temperatures. **This is an existing technique from PINN-DQME, not our discovery.**

## Crucial limitations
1. These models have finite discrete baths rather than continuum-thermal reservoirs. Low temperature here is not necessarily associated with enhanced continuum memory or a failure like in the paper.
2. Our reference is finite-system exact diagonalization, **not HEOM**.
3. Neural models do not outperform the inexpensive exact reference in total cost.
4. Single seed and unoptimized architectures: not an error-budgeted or statistically controlled solver comparison.
5. We have not reproduced the low-temperature PINN-DQME failure or shown a novel error-backflow-aware improvement.
6. The published upstream T=3.0 PINN tutorial requires checkpoint files absent at checked paths. The T=0.3 upstream example uses RBM/TDVP rather than PINN. See [reproduction audit](../docs/upstream_reproduction_audit.md).

## Next priority
Obtain or reconstruct a fully specified continuum-bath SIAM dissipaton hierarchy and a converged HEOM reference, then reproduce the published error accumulation and define an impurity-observable-specific error-sensitivity objective. Evaluate against a **piecewise PINN baseline**, not just one global neural network.
