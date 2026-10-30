# Research status — October 30, 2026

**Project:** Error-backflow-aware neural simulation of non-Markovian quantum dynamics  
**Honest overall status:** Source-anchored physics references and goal-oriented neural numerical ablations established; **no original or computationally advantageous non-Markovian neural solver yet**.

## What we have established

1. **Exact mathematical sanity checks.** Adjoint-residual identities verified to floating-point integration precision. Global nonlinear observable identity checked for smooth residuals and time-window jumps. Related DWR/goal-PINN ideas are established and must not be presented as new.
2. **Source-anchored fermionic reservoir.** Independently reproduced low/high temperature Lorentzian Padé coefficients from Cao et al.'s input data, with machine-precision matches for QuTiP C+/C-. Source low-T slow mode decay time ~6.98ps vs high-T ~0.698ps.
3. **Interacting spinful two-lead numerical physics.** HEOM at original-type bath parameters shows growing depth truncation error with longer times: max depth-2/3 occupation difference over 6ps ~0.005631 at low T (Nk=7), vs ~0.000725 at high T (Nk=3), in an independent factorized-empty-dot benchmark. The original author's fully correlated ADO initial condition has not been reproduced.
4. **Independent reduced spinless exact reference.** Quadratic spectral calculation checked against QuTiP HEOM; at fixed depth4 low T, Padé Nk7→Nk13 reduces max occupancy error from ~1.93e-4 to ~5.87e-6. This validates numerical reference methods but is not the interacting SIAM.
5. **Target-aware neural controls.** Verified 50-run initial test: mean terminal occupation error reduced ~81.1% low T and ~79.4% high T on a 20-mode quadratic toy, but full-trajectory RMSE and norm drift worsen. Five seeds each temperature, same nominal optimizer steps.
6. **Physicality experiments.** 70-run norm-penalty Pareto study: lower norm drift sacrifices substantial terminal improvement. 40-run exactly norm-preserving architecture achieves ~1e-15 norm drift and better terminal accuracy than equally constrained uniform PINN at original parameters.
7. **Robustness negative result.** 48-run unconstrained parameter study and 48-run exactly normalized parameter study expose severe Hamiltonian dependence: hard-normalized global-target loss performs *worse* than uniform hard-norm PINN for detuned and joint-shift cases.
8. **Compute-cost negative result.** Direct diagonalization and 201-time occupation evaluation of the 21×21 toy is ~0.0002219s median on hosted CPU; PINN training is typically 0.7–1.1s. **No speedup.**

## Closest literature / novelty warning

- Multigoal dual-weighted-residual PINNs (2025): https://link.springer.com/article/10.1007/s44379-025-00012-4
- Global-in-time quantum neural Galerkin, PRL 2026: https://doi.org/10.1103/kqvx-dl54
- Cao et al. PINN-DQME 2026: https://pubmed.ncbi.nlm.nih.gov/42138408/
- Non-Markovian quantum master-equation PINNs 2026: https://doi.org/10.1088/2632-2153/ae4b84

Details: [goal-oriented PINN prior art](prior_art_goal_oriented_quantum_pinn.md).

## Scientific decision

**Stop tweaking the exactly solvable 21-mode toy as if it were a research contribution.** It is valuable for unit tests and for a controlled negative result, not publication by itself.

**Proceed with an independent *interacting* correlated full-ADO benchmark**: spinful single-impurity Anderson U=1meV with one Lorentzian lead per spin, prepared in a full HEOM steady state followed by a chemical-potential quench. [Phase 3 issue](https://github.com/HugoWEI/nonmarkovian-neural-dynamics/issues/5). First benchmark depth2/3 and temperature cases, then independently refine Padé, hierarchy, ODE tolerance and physical observables. This is **not** a replacement for reproducing the original two-lead initial conditions.

Only after a converged reference exists should we attempt an *efficient* approximate error-sensitivity scheme without assuming the exact many-body propagator, benchmark against strong non-neural solvers and the published piecewise PINN, and perform a detailed prior-art check.

## Portfolio wording (accurate today)

> Developing reproducible numerical and neural benchmarks for finite-temperature non-Markovian open quantum dynamics. Reconstructed published fermionic reservoir correlation expansions, validated HEOM truncation effects, and studied goal-oriented PINN training with exact quantum-norm constraints and cross-parameter failure tests. Work in progress; no original solver performance claim yet.


## Phase 3 verified follow-up: interacting correlated one-lead HEOM

The reduced-channel **interacting** benchmark now runs on GitHub Actions with full correlated HEOM steady-state ADO initialization and a lead chemical-potential quench over 6 ps, at both temperatures.

- At T/Gamma=3 and Nk=3, the full-trajectory maximum depth2/3 occupation discrepancy is 6.03709e-5.
- At T/Gamma=0.3 and Nk=3, the depth2/3 discrepancy is 0.001657801.
- Refining low T to Nk=5 and Nk=7 yields depth2/3 maximum differences of 0.001589744 and 0.001586066, respectively; at depth3, Nk5→Nk7 shifts the terminal occupation only 4.978e-5.
- At low T, Nk7 depth3 requires ~47.9s to prepare a 5,489-ADO correlated steady state (actual measured GitHub CPU runtime). This is a real computational burden, unlike the trivial quadratic toy.
- A separate low-T Nk3 depth4 test reduces the depth3/4 full-trajectory difference to 0.000174625, compared with 0.001657801 for depth2/3. This supports a hierarchy-convergence trend but is **not** a source-order Nk7 convergence certificate.

See [interacting Nk7 source order](../results/2026-10-30-interacting-low-nk7-depth3.md), [depth4 pilot](../results/2026-10-30-interacting-low-depth4-pilot.md), and [Phase 3 issue](https://github.com/HugoWEI/nonmarkovian-neural-dynamics/issues/5).

**Still open:** a fully converged, physically equivalent reference in the hard low-temperature regime; an actual efficient learned error-backflow method; and a positive cost-benefit result against strong numerical solvers.
