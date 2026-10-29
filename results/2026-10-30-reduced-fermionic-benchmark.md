# Project B: reduced spinless fermionic benchmark and first neural ablation

**Date:** 2026-10-30. Two independent GitHub Actions experiments passed:
- HEOM-versus-exact: https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40975924524
- Fixed-depth Padé refinement: https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40976045674

## Physics and why this is a meaningful numerical control

We replaced the spinful interacting Anderson impurity with a **spinless, quadratic resonant-level model** with one fermionic Lorentzian reservoir. We use the source-inspired values dot energy epsilon_d=-0.5 meV, Gamma=0.1 meV, Lorentzian width W=5 meV, and temperatures T=0.03, 0.3 meV. The dot starts empty; the reservoir is a factorized Gaussian thermal state at mu=0. Its dynamics are not the authors' correlated initial SIAM and this is NOT a reproduction of their PINN-DQME results.

Two independent algorithms predict the dot occupation n_d(t) over 0..6ps:

1. **Non-neural exact quadratic reference:** discretize the Lorentzian spectral density J(E)=Gamma W²/(E²+W²), set V_j²=J(E_j)deltaE_j/(2 pi), exactly evolve the one-particle Hamiltonian, and calculate n_d(t)=sum_j f(E_j)|U_dj(t)|². Code: [`exact_rlm.py`](../src/nonmarkovian_neural_dynamics/exact_rlm.py).
2. **QuTiP fermionic HEOM:** same spinless dot, lead, and factorized initial condition; source-type Lorentzian Padé expansion, varying the hierarchy depth and Padé order. Code: [`spinless_rlm_heom.py`](../scripts/spinless_rlm_heom.py).

The reference's *observed* quadrature-resolution change (800->960 nodes) is below ~2e-10 and the observed spectral-cutoff change (40->60 meV with adjusted quadrature) is approximately 5.5e-6. The latter is a **diagnostic, not a certified upper bound**.

## Verified GitHub numerical results

Maximum dot occupation deviation over the 6ps trajectory, compared with the independent quadratic reference:

| T/Gamma | Source-style Nk | HEOM depth 2 | HEOM depth 5 |
|---:|---:|---:|---:|
| 3.0 | 3 | 2.09059e-5 | 2.09066e-5 |
| 0.3 | 7 | 1.92871e-4 | 1.92871e-4 |

Increasing depth beyond 2 scarcely improves the remaining error in the spinless model. This contrasts sharply with the **interacting spinful** model, whose high-dimensional hierarchy depth produced substantial long-time errors.

At depth 4, increasing Padé order improves independent-reference agreement:

| T/Gamma | Nk | max error |
|---:|---:|---:|
| 3.0 | 3 | 2.0906e-5 |
| 3.0 | 5 | 6.0533e-6 |
| 3.0 | 9 | 6.2309e-6 |
| 0.3 | 7 | 1.9287e-4 |
| 0.3 | 9 | 4.1267e-5 |
| 0.3 | 11 | 8.6006e-6 |
| 0.3 | 13 | 5.8706e-6 |

**Conclusion:** For this reduced benchmark, high hierarchy depth is not needed for the tested dot occupation, while Padé order matters much more at low temperature. Agreement approaches the independently observed spectral-cutoff sensitivity. This provides a credible **small-system physical validation bed** for neural methods, but we have not shown a neural advantage over accurate direct simulation.

## Exploratory piecewise neural experiment: LOCAL runs only

We also implemented a **different, small 20-mode finite-bath neural PINN** with exact quadratic dynamics and four sequential time windows. Losses compare (a) ordinary physics residual, (b) a pointwise adjoint projection, and (c) an integrated adjoint residual. Three seeds at each of the same two temperatures and 1200 optimization steps total were tested locally. These are **not continuum-extrapolated neural simulations**, and they have not yet been independently verified in GitHub Actions.

Mean across three seeds:

| T (meV) | training loss | occupation trajectory RMSE | terminal occupation abs error |
|---:|---|---:|---:|
| .03 | uniform | .03152 | .02357 |
| .03 | pointwise adjoint | .02305 | .04037 |
| .03 | integrated adjoint | .02415 | .03200 |
| .3 | uniform | .03058 | .02381 |
| .3 | pointwise adjoint | .02308 | .03938 |
| .3 | integrated adjoint | .02390 | .02985 |

**Caution:** The proposed adjoint losses lower *average trajectory RMSE* in these settings but **increase terminal occupation error**, which is their nominal target. There is no demonstrated meaningful gain in total computational cost; the direct quadratic solver is far cheaper. Moreover, the usual adjoint formula is a linearization of the quadratic observable, and replacing its exact sensitivity with the network's current prediction can introduce errors.

The first hosted neural run failed because of a NumPy metric typo; the typo was fixed and a smoke test added. This report does not claim its remote success.

## Research decision

Keep the reduced model as a **verification suite and architecture test bed**, not a paper by itself. Next:
1. Confirm the hosted neural run and compare identical seeds/hardware/cost.
2. Test whether a correctly constructed **nonlinear terminal population adjoint** and full-horizon sensitivity allocation can improve the actual targeted occupation while retaining physical norm.
3. Compare to direct numerical solvers and time-domain PINN with matched total cost.
4. Pursue the original strongly non-Markovian SIAM only after sourcing or constructing a reliable correlated ADO initial state and converged long-time reference.

No established original research contribution yet.
